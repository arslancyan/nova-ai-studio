"""Verify that the native NOVA latent denoiser can optimize a tiny batch.

This is a regression test only. It is not meaningful model training.
"""
from __future__ import annotations

import torch

from .autoencoder import build_autoencoder
from .diffusion import GaussianDiffusion
from .latent_model import build_latent_model
from .tokenizer import encode, vocab_size


def main():
    torch.manual_seed(11)
    autoencoder = build_autoencoder()
    autoencoder.eval()

    video = torch.zeros(1, 3, 8, 32, 32)
    video[:, 0, :, 8:24, 8:24] = 1.0
    video[:, 1, :, 12:20, 10:26] = -0.5

    with torch.no_grad():
        latent = autoencoder.encode(video)

    model = build_latent_model(
        latent_channels=latent.shape[1],
        text_vocab_size=vocab_size(),
        latent_tokens=latent[0].numel() // latent.shape[1],
        latent_frames=latent.shape[2],
        latent_height=latent.shape[3],
        latent_width=latent.shape[4],
        model_dim=64,
        text_dim=64,
        num_heads=4,
        num_layers=2,
    )
    diffusion = GaussianDiffusion(steps=32)
    ids = torch.tensor([encode("a red square moving in a dark scene")])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    losses = []
    timestep = torch.tensor([16])
    for _ in range(5):
        noisy, noise = diffusion.q_sample(latent, timestep)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(noisy, ids, timestep)
        loss = torch.nn.functional.mse_loss(predicted, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], f"latent model did not learn: {losses}"
    print(f"native latent learning regression passed: {losses}")


if __name__ == "__main__":
    main()
