"""Train NOVA latent diffusion after the autoencoder is trained.

This trains the native denoiser from random initialization. No pretrained
video model or proprietary generator is used.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import csv

from .autoencoder import build_autoencoder
from .diffusion import GaussianDiffusion
from .latent_model import build_latent_model
from .tokenizer import encode, vocab_size


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PreparedVideoDataset(Dataset):
    def __init__(self, csv_path):
        self.rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
        if not self.rows:
            raise ValueError(f"empty dataset: {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        video = torch.load(row["prepared_path"], map_location="cpu", weights_only=True).float()
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError(f"{row['sample_id']}: expected [3,T,H,W]")
        return video, row["caption"]

def load_prepared_csv(train_csv, val_csv):
    train_ds = PreparedVideoDataset(train_csv)
    val_ds = PreparedVideoDataset(val_csv)
    train_videos, train_captions = zip(*(train_ds[i] for i in range(len(train_ds))))
    val_videos, val_captions = zip(*(val_ds[i] for i in range(len(val_ds))))
    return (
        torch.stack(list(train_videos)), list(train_captions),
        torch.stack(list(val_videos)), list(val_captions),
    )

def train(
    data_root="data/synthetic",
    train_csv=None,
    val_csv=None,
    autoencoder_checkpoint="checkpoints/nova_ae.pt",
    epochs=5,
    batch_size=8,
    lr=2e-4,
    val_fraction=0.1,
    seed=7,
    out="checkpoints/nova_latent.pt",
    resume=None,
    history_out="checkpoints/nova_latent_history.jsonl",
    model_dim=128,
    num_heads=4,
    num_layers=4,
    diffusion_steps=1000,
):
    _set_seed(seed)
    if bool(train_csv) != bool(val_csv):
        raise ValueError("train_csv and val_csv must be provided together")

    if train_csv and val_csv:
        train_clips, train_captions, val_clips, val_captions = load_prepared_csv(train_csv, val_csv)
        clips = torch.cat([train_clips, val_clips], dim=0)
        captions = train_captions + val_captions
        explicit_split = True
        train_count = train_clips.shape[0]
    else:
        root = Path(data_root)
        clips = torch.load(root / "clips.pt", map_location="cpu", weights_only=True).float()
        captions = (root / "captions.txt").read_text(encoding="utf-8").splitlines()
        explicit_split = False
        train_count = 0

    if clips.ndim != 5 or clips.shape[1] != 3:
        raise ValueError("Expected [N,3,T,H,W] clips")
    if len(captions) != clips.shape[0]:
        raise ValueError("Caption count must match clip count")
    if not torch.isfinite(clips).all():
        raise ValueError("Dataset contains NaN or infinity")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae_pack = torch.load(
        autoencoder_checkpoint, map_location=device, weights_only=True
    )
    autoencoder = build_autoencoder(
        in_channels=3,
        latent_channels=int(ae_pack.get("latent_channels", 8)),
    ).to(device)
    autoencoder.load_state_dict(ae_pack["state_dict"])
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        sample_latent = autoencoder.encode(clips[:1].to(device))
    latent_channels = sample_latent.shape[1]
    latent_frames = sample_latent.shape[2]
    latent_height = sample_latent.shape[3]
    latent_width = sample_latent.shape[4]
    latent_tokens = latent_frames * latent_height * latent_width

    model = build_latent_model(
        latent_channels=latent_channels,
        text_vocab_size=vocab_size(),
        latent_tokens=latent_tokens,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    diffusion = GaussianDiffusion(steps=diffusion_steps)
    text_ids = torch.tensor(
        [encode(caption) for caption in captions], dtype=torch.long
    )

    generator = torch.Generator().manual_seed(seed)
    if explicit_split:
        train_idx = torch.arange(train_count)
        val_idx = torch.arange(train_count, clips.shape[0])
        if val_idx.numel() == 0:
            raise ValueError("Prepared validation split is empty")
    else:
        permutation = torch.randperm(clips.shape[0], generator=generator)
        val_count = max(1, int(round(clips.shape[0] * val_fraction)))
        val_idx = permutation[:val_count]
        train_idx = permutation[val_count:]
        if train_idx.numel() == 0:
            raise ValueError("Dataset is too small for a non-empty training split")

    train_loader = DataLoader(
        TensorDataset(clips[train_idx], text_ids[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(clips[val_idx], text_ids[val_idx]),
        batch_size=batch_size,
        shuffle=False,
    )

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
        print(f"resuming from epoch={start_epoch}")

    history_path = Path(history_out)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    def run_validation() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for video, ids in val_loader:
                video = video.to(device)
                ids = ids.to(device)
                clean_latent = autoencoder.encode(video)
                timestep = torch.randint(
                    0, diffusion.steps, (video.shape[0],), device=device
                )
                noisy, noise = diffusion.q_sample(clean_latent, timestep)
                predicted = model(noisy, ids, timestep)
                total += float(
                    torch.nn.functional.mse_loss(predicted, noise).detach()
                ) * video.shape[0]
                count += video.shape[0]
        model.train()
        return total / max(1, count)

    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0.0
        count = 0
        for video, ids in train_loader:
            video = video.to(device)
            ids = ids.to(device)
            with torch.no_grad():
                clean_latent = autoencoder.encode(video)

            timestep = torch.randint(
                0, diffusion.steps, (video.shape[0],), device=device
            )
            noisy, noise = diffusion.q_sample(clean_latent, timestep)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if use_amp else "cpu",
                dtype=torch.float16 if use_amp else torch.float32,
                enabled=use_amp,
            ):
                predicted = model(noisy, ids, timestep)
                loss = torch.nn.functional.mse_loss(predicted, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total += float(loss.detach()) * video.shape[0]
            count += video.shape[0]

        train_loss = total / max(1, count)
        val_loss = run_validation()
        best_val = min(best_val, val_loss)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val": best_val,
                        "seed": seed,
                        "model_dim": model_dim,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "diffusion_steps": diffusion_steps,
                    }
                )
                + "\n"
            )

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "latent_channels": latent_channels,
                "latent_tokens": latent_tokens,
                "epoch": epoch + 1,
                "best_val": best_val,
                "seed": seed,
                "model_dim": model_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "diffusion_steps": diffusion_steps,
                "dataset_mode": "prepared_csv" if explicit_split else "synthetic",
                "train_csv": train_csv,
                "val_csv": val_csv,
            },
            out,
        )

    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the native NOVA latent denoiser")
    parser.add_argument("--data-root", default="data/synthetic")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--autoencoder-checkpoint", default="checkpoints/nova_ae.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="checkpoints/nova_latent.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--history-out", default="checkpoints/nova_latent_history.jsonl")
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    args = parser.parse_args()
    train(
        data_root=args.data_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        out=args.out,
        resume=args.resume,
        history_out=args.history_out,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        diffusion_steps=args.diffusion_steps,
    )
