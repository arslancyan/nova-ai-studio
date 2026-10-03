"""Verify that the random-init NOVA autoencoder can actually learn.

This is a tiny optimization test, not model training. It catches silent
gradient/optimizer/forward-path regressions that a shape-only smoke test misses.
"""
from __future__ import annotations

import torch

from .autoencoder import build_autoencoder


def main() -> None:
    torch.manual_seed(7)
    model = build_autoencoder(in_channels=3, latent_channels=8)
    model.train()

    # One deterministic synthetic clip. Keep the test small enough for CI CPU.
    video = torch.zeros(1, 3, 8, 32, 32)
    video[:, 0, :, 8:24, 8:24] = 1.0
    video[:, 1, :, 12:20, 10:26] = -0.5

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, _ = model(video)
        loss = (
            torch.nn.functional.l1_loss(reconstruction, video)
            + 0.1 * torch.nn.functional.mse_loss(reconstruction, video)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], (
        f"autoencoder did not learn on the tiny overfit test: {losses}"
    )
    print(f"NOVA autoencoder overfit test passed: {losses}")


if __name__ == "__main__":
    main()
