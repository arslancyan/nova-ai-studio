"""Train the native NOVA video autoencoder with validation and resumable checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from .autoencoder import build_autoencoder


def _loss(reconstruction, video):
    return (
        torch.nn.functional.l1_loss(reconstruction, video)
        + 0.1 * torch.nn.functional.mse_loss(reconstruction, video)
    )


def train(
    root="data/synthetic",
    epochs=10,
    batch_size=8,
    lr=2e-4,
    val_fraction=0.1,
    seed=0,
    out="checkpoints/nova_ae.pt",
    resume=None,
):
    root = Path(root)
    out_path = Path(out)

    clips = torch.load(
        root / "clips.pt", map_location="cpu", weights_only=True
    ).float()
    if clips.ndim != 5 or clips.shape[1] != 3:
        raise ValueError("Expected [N,3,T,H,W] clips")
    if clips.shape[0] < 2:
        raise ValueError("Need at least two clips for train/validation split")
    if not torch.isfinite(clips).all():
        raise ValueError("Dataset contains NaN or infinity")
    if clips.min() < -1.0001 or clips.max() > 1.0001:
        raise ValueError("Dataset values must stay in [-1, 1]")

    captions = (root / "captions.txt").read_text(encoding="utf-8").splitlines()
    if len(captions) != clips.shape[0]:
        raise ValueError("Caption count must match clip count")

    generator = torch.Generator().manual_seed(seed)
    val_size = max(1, int(round(len(clips) * val_fraction)))
    train_size = len(clips) - val_size
    train_set, val_set = random_split(
        TensorDataset(clips), [train_size, val_size], generator=generator
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_autoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")

    if resume:
        pack = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(pack["state_dict"])
        if "optimizer" in pack:
            optimizer.load_state_dict(pack["optimizer"])
        if "scaler" in pack and use_amp:
            scaler.load_state_dict(pack["scaler"])
        start_epoch = int(pack.get("epoch", 0))
        best_val = float(pack.get("best_val", best_val))
        print(f"resumed from {resume} at epoch {start_epoch}")

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    history_path = out_path.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_total = 0.0

        for (video,) in loader:
            video = video.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device,
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                reconstruction, _ = model(video)
                loss = _loss(reconstruction, video)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for (video,) in val_loader:
                video = video.to(device)
                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16 if use_amp else torch.float32,
                    enabled=use_amp,
                ):
                    reconstruction, _ = model(video)
                    val_total += float(_loss(reconstruction, video))

        train_loss = train_total / max(1, len(loader))
        val_loss = val_total / max(1, len(val_loader))
        improved = val_loss < best_val
        best_val = min(best_val, val_loss)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "device": device,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "latent_channels": 8,
            "input_shape": [3, int(clips.shape[2]), int(clips.shape[3]), int(clips.shape[4])],
            "epoch": epoch + 1,
            "best_val": best_val,
            "seed": seed,
            "dataset": str(root),
            "training": {
                "epochs_target": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "val_fraction": val_fraction,
            },
        }
        torch.save(checkpoint, out_path)
        if improved:
            torch.save(checkpoint, out_path.with_name(out_path.stem + ".best" + out_path.suffix))
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} saved={out_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/nova_ae.pt")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(
        root=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        out=args.out,
        resume=args.resume,
    )
