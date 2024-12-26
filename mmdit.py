import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional

def attention(q, k, v, heads):
    q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=heads), (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    return rearrange(out, 'b h n d -> b n (h d)')

def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# class PatchEmbed(nn.Module):
#     # temporarily flatten the whole image only
#     def __init__(self, in_channels=1, input_shape=28*28, output_shape=128, patch_size=7):
#         super().__init__()
#         self.proj = nn.Conv2d(in_channels=in_channels, out_channels=output_shape, kernel_size=patch_size, stride=patch_size)


class PatchEmbed(nn.Module):
    # temporarily flatten the whole image only
    def __init__(self, input_dim=28*28, output_size=128, patch_size=16):
        super().__init__()
        self.l = nn.Linear(input_dim, output_size)

    def forward(self, x):
        return self.l(rearrange(x, 'b ... -> b (...)'))


# class LabelEmbedder(nn.Module):
#     def __init__(self, input_dim=10, embedding_size=128):
#         super().__init__()
#         self.embed = nn.Embedding(input_dim, embedding_size)
    
#     def forward(self, x):
#         return self.embed(x)


class VectorEmbedder(nn.Module):
    def __init__(self, input_dim=10, hidden_size=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, x):
        return self.mlp(x)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def split_qkv(qkv, heads):
    qkv = rearrange(qkv, 'b n (x h d) -> x b n d h', h=heads, x=3)
    return qkv[0], qkv[1], qkv[2]


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        pre_only: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.ln_q = nn.LayerNorm(self.head_dim)
        self.ln_k = nn.LayerNorm(self.head_dim)
        
        self.proj = nn.Linear(dim, dim)
        
    def pre_attention(self, x):
        # x is b n c
        qkv = self.qkv(x) # b n 3d
        q, k, v = split_qkv(qkv, self.num_heads)
        q = rearrange(self.ln_q(q), 'b n d h -> b n (d h)')
        k = rearrange(self.ln_k(k), 'b n d h -> b n (d h)')
        return q, k, v
    
    def post_attention(self, x):
        return self.proj(x)
    
    def forward(self, x):
        q, k, v = self.pre_attention(x)
        out = attention(q, k, v, self.num_heads)
        return self.post_attention(out)
    
    
class DismantledBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        self.attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        mlp_hidden_dim = int(4 * hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        
    def pre_attention(self, x, c):  # this c is y in the figure
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
        return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    
    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
    def forward(self, x, c):
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.num_heads)
        return self.post_attention(attn, *intermediates)
    

def block_mixing(context, x, context_block, x_block, c):
    assert context is not None, "block_mixing called with None context"
    context_qkv, context_intermediates = context_block.pre_attention(context, c)

    x_qkv, x_intermediates = x_block.pre_attention(x, c)

    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    q, k, v = tuple(o)

    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = (attn[:, : context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1] :])

    context = context_block.post_attention(context_attn, *context_intermediates)

    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.context_block = DismantledBlock(*args, **kwargs)
        self.x_block = DismantledBlock(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)
    
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, total_out_channels: Optional[int] = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
            if (total_out_channels is None)
            else nn.Linear(hidden_size, total_out_channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class MMDiT(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_size=32,
        num_classes=10,
        depth=2,
        num_heads=8,
        qkv_bias=False,
        patch_size=16,
        out_channels=1,
        total_out_channels=None,
    ):
        super().__init__()
                
        self.x_embedder = PatchEmbed(input_shape=input_size, output_shape=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = VectorEmbedder(num_classes, hidden_size)        
        self.context_embedder = VectorEmbedder(num_classes, hidden_size)
        
        self.joint_blocks = nn.ModuleList(
            [JointBlock(hidden_size, num_heads, qkv_bias) for i in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels, total_out_channels)
        
    def forward_core_with_concat(self, x, c_mod, context):
        # context is B, L', D
        # x is B, L, D
        for block in self.joint_blocks:
            context, x = block(context, x, c=c_mod)

        x = self.final_layer(x, c_mod)  # (N, T, patch_size ** 2 * out_channels)
        return x
    
    def forward(self, x, t, y, context):
        x = self.x_embedder(x)
        c = self.t_embedder(t)
        y = self.y_embedder(y)
        c = c + y
        
        context = self.context_embedder(context)
        x = self.forward_core_with_concat(x, c, context)
        return x
    
