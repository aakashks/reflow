import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Optional
from dataclasses import dataclass
from torch import Tensor

from loguru import logger


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ImageEmbedder(nn.Module):
    def __init__(self, in_channels, image_size, embed_dim, patch_size):
        super().__init__()
        num_patches = image_size // patch_size
        self.patch_size = patch_size
        
        self.conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=5, stride=1, padding='same'),
            nn.SiLU(),
            nn.GroupNorm(32, embed_dim // 2),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=5, stride=1, padding='same'),
            nn.SiLU(),
            nn.GroupNorm(32, embed_dim // 2),
        )
        
        self.embed = nn.Linear(embed_dim // 2 * patch_size**2, embed_dim)
        
        self.pos_embedder = nn.Embedding(num_patches ** 2, embed_dim)
        nn.init.xavier_uniform_(self.pos_embedder.weight)

    def patchify(self, x):
        return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)

    def forward(self, x):
        """
        x: (N, C, H, W)
        returns: (N, L, D) where L is no of patches and D is embedding dimension
        """
        x = self.conv_seq(x)
        x = self.patchify(x)
        x = self.embed(x)
        x += self.pos_embedder(torch.arange(x.shape[1], device=x.device))
        return x


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.2):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.use_dropout = dropout_prob > 0
        self.embed = nn.Embedding(self.num_classes + int(self.use_dropout), hidden_size)
    
    def forward(self, labels):
        labels = labels.to(torch.int32)
        
        if self.training and self.use_dropout:
            drop_ids = torch.bernoulli(labels, self.dropout_prob).bool()
            labels = torch.where(drop_ids, self.num_classes, labels)
        
        return self.embed(labels)


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
    def timestep_embedding(t: Tensor, dim: int, max_period=10000):
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

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def split_qkv(qkv, head_dim):
    qkv = rearrange(qkv, 'b n (split head head_dim) -> split b n head head_dim', split=3, head_dim=head_dim)
    return qkv[0], qkv[1], qkv[2]


def precompute_freqs(dim: int, maxlen: int, theta: float = 1e4):
    scale = torch.arange(0, dim, 2, dtype=torch.float64)[: (dim // 2)] / dim
    freqs = 1.0 / (theta ** scale)
    t = torch.arange(maxlen, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freq_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return freq_cis


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
        q, k, v = split_qkv(qkv, self.head_dim)
        # q = rearrange(self.ln_q(q), 'b n h d -> b n (h d)')
        # k = rearrange(self.ln_k(k), 'b n h d -> b n (h d)')
        q = self.ln_q(q)
        k = self.ln_k(k)
        return q, k, v
    
    def post_attention(self, x):
        return self.proj(x)

    @staticmethod    
    def apply_rope(q, k, freq_cis):
        (q_r, q_i), (k_r, k_i) = map(lambda t: rearrange(t, 'b n h (d split) -> split b h n d', split=2), (q, k))
        
        freq_cis = rearrange(freq_cis.to(q.device), 'n d cs -> cs 1 1 n d', cs=2)
        freq_cos, freq_sin = freq_cis

        q_out_r = q_r * freq_cos - q_i * freq_sin
        q_out_i = q_r * freq_sin + q_i * freq_cos
        k_out_r = k_r * freq_cos - k_i * freq_sin
        k_out_i = k_r * freq_sin + k_i * freq_cos
        
        q_out = torch.stack([q_out_r, q_out_i], dim=-1)
        k_out = torch.stack([k_out_r, k_out_i], dim=-1)
        
        q, k = map(lambda t: rearrange(t.to(q.dtype), 'b h n d split -> b h n (d split)', split=2), (q_out, k_out))
        return q, k
    
    def forward(self, x, freq_cis):
        q, k, v = self.pre_attention(x)
        q, k = self.apply_rope(q, k, freq_cis)
        v = rearrange(v, 'b n h d -> b h n d')
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.post_attention(out)

    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, bias=False, elementwise_affine=False, eps=1e-6)
        self.linear = (
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(), 
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x1 = F.silu(self.lin1(x))
        x2 = self.lin2(x)
        return self.lin(x1 * x2)


    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
    ):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, qkv_bias)
        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, 4 * dim)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        
    def forward(self, x, freq_cis, c=None):
        if c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(c).chunk(6, dim=-1)

            attn_out = self.attn(modulate(self.attn_norm(x), shift_msa, scale_msa), freq_cis)
            x = x + gate_msa.unsqueeze(1) * attn_out
            
            ffn_out = self.mlp(modulate(self.ffn_norm(x), shift_mlp, scale_mlp))
            x = x + gate_mlp.unsqueeze(1) * ffn_out
            
        else:
            x = x + self.attn(self.attn_norm(x), freq_cis)
            x = x + self.mlp(self.ffn_norm(x))
            
        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        num_channels=1,
        depth=6,
        num_heads=4,
        hidden_dim=64,
        cfg_dropout_prob=0.2,
        patch_size=2,
        qkv_bias=False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.x_embedder = ImageEmbedder(num_channels, input_size, hidden_dim, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.y_embedder = LabelEmbedder(num_classes, hidden_dim, cfg_dropout_prob)        
    
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, qkv_bias)
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_dim, patch_size, num_channels)
        
        max_pos_encoding = (input_size // patch_size) ** 2
        self.freq_cis = precompute_freqs(hidden_dim // num_heads, max_pos_encoding)
        
        logger.info(f"Initialized DiT with {self.get_num_params()} parameters")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def unpatchify(self, x):
        b, n, hw = x.shape
        n = int(math.sqrt(n))
        out = rearrange(x, 'b (n1 n2) (c h w) -> b c (n1 h) (n2 w)', n1=n, n2=n, c=self.num_channels, h=self.patch_size)
        return out

    
    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        y = self.y_embedder(y)

        adaln_context = t.to(x.dtype) + y.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freq_cis, adaln_context)

        x = self.final_layer(x, adaln_context)
        return self.unpatchify(x)
