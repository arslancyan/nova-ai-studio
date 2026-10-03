"""Smoke-test the NOVA latent denoiser contract."""
from __future__ import annotations

import torch

from .latent_model import build_latent_model


def main() -> None:
    torch.manual_seed(0)
    model = build_latent_model(
        latent_channels=8,
        text_vocab_size=97,
        model_dim=64,
        text_dim=64,
        num_heads=4,
        num_layers=2,
        latent_tokens=2 * 8 * 8,
    )
    latent = torch.randn(2, 8, 2, 8, 8)
    text_ids = torch.randint(0, 97, (2, 32))
    timesteps = torch.tensor([10, 500])
    predicted = model(latent, text_ids, timesteps)

    assert predicted.shape == latent.shape
    assert torch.isfinite(predicted).all()

    loss = torch.nn.functional.mse_loss(predicted, torch.randn_like(predicted))
    loss.backward()
    print(
        "NOVA latent denoiser smoke test passed "
        f"(shape={tuple(predicted.shape)}, loss={float(loss):.6f})"
    )


if __name__ == "__main__":
    main()
