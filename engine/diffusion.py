"""Gaussian diffusion utilities for the native NOVA model."""
from __future__ import annotations

import torch


def linear_beta_schedule(steps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, steps)


class GaussianDiffusion:
    def __init__(self, steps=1000):
        if steps < 2:
            raise ValueError("Diffusion requires at least two steps")
        self.steps = steps
        self.betas = linear_beta_schedule(steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, clean, t, noise=None):
        if clean.ndim != 5:
            raise ValueError("Expected clean latent/video tensor [B,C,T,H,W]")
        if t.ndim != 1 or t.shape[0] != clean.shape[0]:
            raise ValueError("t must have shape [B]")
        if torch.any(t < 0) or torch.any(t >= self.steps):
            raise ValueError("Diffusion timestep out of range")

        noise = noise if noise is not None else torch.randn_like(clean)
        a = self.alpha_bars.to(clean.device)[t].view(-1, 1, 1, 1, 1)
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def step(self, model, x, text_ids, t):
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape [B]")

        betas = self.betas.to(x.device)
        alpha = self.alphas.to(x.device)
        alpha_bar = self.alpha_bars.to(x.device)

        pred_noise = model(x, text_ids, t)
        beta_t = betas[t].view(-1, 1, 1, 1, 1)
        alpha_t = alpha[t].view(-1, 1, 1, 1, 1)
        abar_t = alpha_bar[t].view(-1, 1, 1, 1, 1)

        mean = (
            x - beta_t / (1 - abar_t).clamp_min(1e-12).sqrt() * pred_noise
        ) / alpha_t.sqrt()

        nonzero = (t > 0).view(-1, 1, 1, 1, 1)
        noise = torch.randn_like(x)
        return torch.where(nonzero, mean + beta_t.sqrt() * noise, mean)
