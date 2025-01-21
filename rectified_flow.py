import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

class RectifiedFlow:
    def __init__(self, model):
        self.model = model

    def forward_pass(self, x0, y, logit_sampling=False):
        x1 = torch.randn_like(x0, device=x0.device, requires_grad=False)
        t = torch.rand(x0.size(0), requires_grad=False, device=x0.device)
        
        if logit_sampling:
            t = torch.randn_like(t)
            t = torch.sigmoid(t)

        t_ = t.view(x0.size(0), 1, 1, 1)
        xt = t_ * x1 + (1 - t_) * x0

        vt = self.model(xt, t, y)
        
        return F.mse_loss(vt, (x0 - x1))
    
    
    @torch.inference_mode()
    def simple_euler(self, z, cond, uncond=None, cfg=2.5, steps=25):
        b = z.size(0)
        images = []

        dt = 1.0 / steps
        dt = torch.full((b,), dt, device=z.device)
        dt = dt.view(b, 1, 1, 1)
        
        for i in tqdm(range(steps, -1, -1), desc='Sampling Steps', leave=False):
            t = torch.full((b,), i / steps, device=z.device)
            vc = self.model(z, t, cond)
            
            if uncond is not None:
                vu = self.model(z, t, uncond)
                vc = cfg * vc + (1 - cfg) * vu
            
            z = z + dt*vc
            images.append(z.cpu())

        return images
