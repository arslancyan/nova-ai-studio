"""Regression test for the native DDIM-style sampler path."""
from __future__ import annotations
import torch
from .diffusion import GaussianDiffusion

class TinyModel(torch.nn.Module):
    def forward(self, x, text_ids, t):
        return torch.zeros_like(x)

def main():
    torch.manual_seed(3)
    diffusion = GaussianDiffusion(steps=8)
    model = TinyModel()
    x = torch.randn(1,2,2,4,4)
    ids = torch.ones(1,8,dtype=torch.long)
    t = torch.tensor([7])
    prev = torch.tensor([3])
    out = diffusion.ddim_step(model,x,ids,t,prev,eta=0.0)
    if out.shape != x.shape or not torch.isfinite(out).all():
        raise AssertionError("DDIM output is invalid")
    out2 = diffusion.ddim_step(model,x,ids,t,prev,eta=0.0)
    if not torch.equal(out,out2):
        raise AssertionError("eta=0 DDIM path is not deterministic")
    print("DDIM sampler regression passed")

if __name__ == "__main__":
    main()
