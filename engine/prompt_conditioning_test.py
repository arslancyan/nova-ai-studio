"""Regression test for text conditioning in the native latent denoiser.

This test checks that changing only the prompt changes the denoiser output
under identical latent noise and timestep. It does not claim semantic quality.
"""
from __future__ import annotations

import torch

from .diffusion import GaussianDiffusion
from .latent_model import build_latent_model
from .tokenizer import encode, vocab_size


def main() -> None:
    torch.manual_seed(23)
    model = build_latent_model(
        latent_channels=8,
        text_vocab_size=vocab_size(),
        latent_tokens=2 * 8 * 8,
        latent_frames=2,
        latent_height=8,
        latent_width=8,
        model_dim=64,
        text_dim=64,
        num_heads=4,
        num_layers=2,
    )
    model.eval()

    diffusion = GaussianDiffusion(steps=32)
    clean = torch.randn(1, 8, 2, 8, 8)
    timestep = torch.tensor([16])
    noise = torch.randn_like(clean)
    noisy = diffusion.q_sample(clean, timestep, noise=noise)[0]

    red = torch.tensor([encode("a red circle moving in a dark scene")])
    blue = torch.tensor([encode("a blue square moving in a dark scene")])

    red_pred = model(noisy, red, timestep)
    blue_pred = model(noisy, blue, timestep)

    difference = torch.mean(torch.abs(red_pred - blue_pred)).item()
    assert torch.isfinite(red_pred).all()
    assert torch.isfinite(blue_pred).all()
    assert difference > 1e-7, f"prompt conditioning collapsed: difference={difference}"

    print(f"native prompt-conditioning regression passed: mean_abs_difference={difference:.8f}")


if __name__ == "__main__":
    main()
