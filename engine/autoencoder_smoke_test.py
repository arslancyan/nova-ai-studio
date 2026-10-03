"""Smoke-test the native NOVA video autoencoder.

This test uses random initialization only and verifies the tensor contract
needed by the first representation-learning stage.
"""
from __future__ import annotations

import torch

from .autoencoder import build_autoencoder


def main() -> None:
    torch.manual_seed(0)
    model = build_autoencoder(in_channels=3, latent_channels=8)
    video = torch.randn(2, 3, 8, 32, 32)
    reconstruction, latent = model(video)

    assert reconstruction.shape == video.shape, (
        f"reconstruction shape {tuple(reconstruction.shape)} "
        f"!= input shape {tuple(video.shape)}"
    )
    assert latent.shape == (2, 8, 2, 8, 8), (
        f"unexpected latent shape: {tuple(latent.shape)}"
    )
    assert torch.isfinite(reconstruction).all()
    assert torch.isfinite(latent).all()

    loss = torch.nn.functional.l1_loss(reconstruction, video)
    loss.backward()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0
    print(
        "NOVA autoencoder smoke test passed "
        f"(params={trainable:,}, latent={tuple(latent.shape)}, loss={float(loss):.6f})"
    )


if __name__ == "__main__":
    main()
