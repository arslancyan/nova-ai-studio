"""Train the native NOVA autoencoder on prepared real-video tensors.

The prepared dataset contains one [3,T,H,W] tensor per clip plus train/val CSVs.
This script keeps provenance metadata alongside the training checkpoint and can
resume interrupted runs.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .autoencoder import build_autoencoder


class PreparedVideoDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = str(csv_path)
        self.rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
        if not self.rows:
            raise ValueError(f"empty dataset: {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        video = torch.load(
            row["prepared_path"], map_location="cpu", weights_only=True
        ).float()
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"{row['sample_id']}: expected [3,T,H,W]")
        if not torch.isfinite(video).all():
            raise ValueError(f"{row['sample_id']}: non-finite tensor")
        if video.min() < -1.0001 or video.max() > 1.0001:
            raise ValueError(f"{row['sample_id']}: values outside [-1,1]")
        return video


def loss_fn(reconstruction, video):
    return (
        torch.nn.functional.l1_loss(reconstruction, video)
        + 0.1 * torch.nn.functional.mse_loss(reconstruction, video)
    )


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    train_csv="data/real/train.csv",
    val_csv="data/real/val.csv",
    epochs=10,
    batch_size=2,
    lr=2e-4,
    out="checkpoints/nova_real_ae.pt",
    seed=7,
    resume=None,
):
    _set_seed(seed)
    train_ds = PreparedVideoDataset(train_csv)
    val_ds = PreparedVideoDataset(val_csv)
    sample = train_ds[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_autoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best = float("inf")
    if resume:
        pack = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(pack["state_dict"])
        if "optimizer" in pack:
            optimizer.load_state_dict(pack["optimizer"])
        if use_amp and pack.get("scaler"):
            scaler.load_state_dict(pack["scaler"])
        start_epoch = int(pack.get("epoch", 0))
        best = float(pack.get("best_val", best))
        seed = int(pack.get("seed", seed))
        print(f"resuming from epoch={start_epoch}")

    generator = torch.Generator().manual_seed(seed + start_epoch)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0.0
        for video in loader:
            video = video.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device,
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                reconstruction, _ = model(video)
                loss = loss_fn(reconstruction, video)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for video in val_loader:
                video = video.to(device)
                with torch.autocast(
                    device_type=device,
                    dtype=torch.float16 if use_amp else torch.float32,
                    enabled=use_amp,
                ):
                    reconstruction, _ = model(video)
                    val_total += float(loss_fn(reconstruction, video))

        train_loss = total / max(1, len(loader))
        val_loss = val_total / max(1, len(val_loader))
        best = min(best, val_loss)

        pack = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "epoch": epoch + 1,
            "best_val": best,
            "input_shape": list(sample.shape),
            "dataset_train": train_csv,
            "dataset_val": val_csv,
            "latent_channels": 8,
            "seed": seed,
            "native_random_init": True,
            "training": {
                "epochs_target": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "val_fraction": len(val_ds) / max(1, len(train_ds) + len(val_ds)),
            },
        }
        torch.save(pack, out_path)
        if val_loss <= best:
            torch.save(pack, out_path.with_name(out_path.stem + ".best" + out_path.suffix))
        print(f"epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data/real/train.csv")
    p.add_argument("--val-csv", default="data/real/val.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="checkpoints/nova_real_ae.pt")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", default=None)
    a = p.parse_args()
    train(
        train_csv=a.train_csv,
        val_csv=a.val_csv,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        out=a.out,
        seed=a.seed,
        resume=a.resume,
    )
