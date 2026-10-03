"""Train the native NOVA autoencoder on prepared real-video tensors.

The prepared dataset contains one [3,T,H,W] tensor per clip plus train/val CSVs.
This script keeps provenance metadata alongside the training checkpoint.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from .autoencoder import build_autoencoder

class PreparedVideoDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
        if not self.rows:
            raise ValueError(f"empty dataset: {csv_path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        video = torch.load(row["prepared_path"], map_location="cpu", weights_only=True).float()
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"{row['sample_id']}: expected [3,T,H,W]")
        return video

def loss_fn(reconstruction, video):
    return torch.nn.functional.l1_loss(reconstruction, video) + 0.1 * torch.nn.functional.mse_loss(reconstruction, video)

def train(train_csv="data/real/train.csv", val_csv="data/real/val.csv",
          epochs=10, batch_size=2, lr=2e-4, out="checkpoints/nova_real_ae.pt"):
    train_ds, val_ds = PreparedVideoDataset(train_csv), PreparedVideoDataset(val_csv)
    sample = train_ds[0]
    model = build_autoencoder().to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    out_path = Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(epochs):
        model.train(); total = 0.0
        for video in loader:
            video = video.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(video)
            loss = loss_fn(reconstruction, video)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss)
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for video in val_loader:
                video = video.to(device)
                reconstruction, _ = model(video)
                val_total += float(loss_fn(reconstruction, video))
        train_loss = total / len(loader)
        val_loss = val_total / len(val_loader)
        pack = {
            "state_dict": model.state_dict(), "epoch": epoch + 1,
            "best_val": min(best, val_loss), "input_shape": list(sample.shape),
            "dataset_train": train_csv, "dataset_val": val_csv,
            "training": {"epochs_target": epochs, "batch_size": batch_size, "learning_rate": lr},
        }
        torch.save(pack, out_path)
        if val_loss < best:
            best = val_loss
            torch.save(pack, out_path.with_name(out_path.stem + ".best" + out_path.suffix))
        print(f"epoch={epoch+1} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train-csv", default="data/real/train.csv")
    p.add_argument("--val-csv", default="data/real/val.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="checkpoints/nova_real_ae.pt")
    a=p.parse_args()
    train(a.train_csv,a.val_csv,a.epochs,a.batch_size,a.lr,a.out)
