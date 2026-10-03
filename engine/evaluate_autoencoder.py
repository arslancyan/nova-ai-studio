"""Evaluate a trained NOVA autoencoder on a held-out deterministic split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .autoencoder import build_autoencoder


@torch.no_grad()
def evaluate(
    root="data/synthetic",
    checkpoint="checkpoints/nova_ae.pt",
    val_fraction=0.1,
    seed=0,
    batch_size=8,
):
    root = Path(root)
    clips = torch.load(root / "clips.pt", map_location="cpu", weights_only=True).float()
    if clips.ndim != 5 or clips.shape[1] != 3:
        raise ValueError("Expected [N,3,T,H,W] clips")

    generator = torch.Generator().manual_seed(seed)
    val_size = max(1, int(round(len(clips) * val_fraction)))
    train_size = len(clips) - val_size
    _, val_set = random_split(
        TensorDataset(clips), [train_size, val_size], generator=generator
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_autoencoder(
        in_channels=3,
        latent_channels=int(pack.get("latent_channels", 8)),
    ).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    total_l1 = 0.0
    total_mse = 0.0
    batches = 0

    for (video,) in loader:
        video = video.to(device)
        reconstruction, _ = model(video)
        total_l1 += float(torch.nn.functional.l1_loss(reconstruction, video))
        total_mse += float(torch.nn.functional.mse_loss(reconstruction, video))
        batches += 1

    l1 = total_l1 / max(1, batches)
    mse = total_mse / max(1, batches)
    psnr = float("inf") if mse == 0 else -10.0 * torch.log10(torch.tensor(mse)).item()

    print(
        f"validation_l1={l1:.6f} validation_mse={mse:.6f} "
        f"validation_psnr={psnr:.3f}dB"
    )
    return {"l1": l1, "mse": mse, "psnr_db": psnr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/synthetic")
    parser.add_argument("--checkpoint", default="checkpoints/nova_ae.pt")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    evaluate(
        root=args.root,
        checkpoint=args.checkpoint,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
    )
