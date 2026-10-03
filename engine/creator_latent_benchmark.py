"""Run a CPU-friendly native latent benchmark on creator-owned videos.

This profile is intentionally small: it validates that the creator-owned
dataset can reach the native latent diffusion stage without pretending that
four clips are enough for production-quality generation.

Raw videos remain local. No network access or downloads are performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .autoencoder import build_autoencoder
from .diffusion import GaussianDiffusion
from .latent_model import build_latent_model
from .tokenizer import encode, vocab_size


def load_csv(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def load_clips(rows, height=32, width=32, frames=8):
    clips, captions = [], []
    for row in rows:
        video = torch.load(row["prepared_path"], map_location="cpu", weights_only=True).float()
        if tuple(video.shape[:1]) != (3,):
            raise ValueError(f"{row['sample_id']}: expected [3,T,H,W]")
        video = F.interpolate(
            video.unsqueeze(0),
            size=(frames, height, width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        clips.append(video)
        captions.append(row["caption"])
    return torch.stack(clips), captions


def run(
    train_csv,
    val_csv,
    autoencoder_checkpoint,
    epochs=10,
    lr=3e-4,
    seed=11,
    model_dim=32,
    text_dim=32,
    num_heads=4,
    num_layers=1,
    diffusion_steps=8,
    out="checkpoints/nova_creator_latent_micro.pt",
):
    random.seed(seed)
    torch.manual_seed(seed)

    train, train_captions = load_clips(load_csv(train_csv))
    val, val_captions = load_clips(load_csv(val_csv))

    ae_pack = torch.load(autoencoder_checkpoint, map_location="cpu", weights_only=True)
    ae = build_autoencoder(
        in_channels=3,
        latent_channels=int(ae_pack.get("latent_channels", 8)),
    )
    ae.load_state_dict(ae_pack["state_dict"])
    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        sample_latent = ae.encode(train[:1])

    latent_shape = tuple(sample_latent.shape[1:])
    latent_frames, latent_height, latent_width = latent_shape[1:]
    latent_tokens = latent_frames * latent_height * latent_width

    model = build_latent_model(
        latent_channels=latent_shape[0],
        text_vocab_size=vocab_size(),
        latent_tokens=latent_tokens,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        model_dim=model_dim,
        text_dim=text_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    diffusion = GaussianDiffusion(steps=diffusion_steps)
    train_ids = torch.tensor([encode(x) for x in train_captions], dtype=torch.long)
    val_ids = torch.tensor([encode(x) for x in val_captions], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    def loss_batch(clips, ids):
        with torch.no_grad():
            clean = ae.encode(clips)
        timestep = torch.randint(0, diffusion.steps, (clips.shape[0],))
        noisy, noise = diffusion.q_sample(clean, timestep)
        predicted = model(noisy, ids, timestep)
        return F.mse_loss(predicted, noise)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for index in range(len(train)):
            loss = loss_batch(train[index:index + 1], train_ids[index:index + 1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_batch(val, val_ids))
        train_loss = total / len(train)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    # Conditioning regression: identical noise/timestep but different prompts
    # must produce different denoiser outputs.
    model.eval()
    torch.manual_seed(seed + 1)
    clean = torch.randn(2, *latent_shape)
    timestep = torch.tensor([diffusion_steps // 2, diffusion_steps // 2])
    prompt_ids = torch.tensor([
        encode("a red circle moving in a dark scene"),
        encode("a blue square moving in a dark scene"),
    ])
    with torch.no_grad():
        outputs = model(clean, prompt_ids, timestep)
    separation = float((outputs[0] - outputs[1]).abs().mean())
    if not math.isfinite(separation) or separation <= 1e-7:
        raise AssertionError("prompt conditioning regression failed")

    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epochs,
            "best_val": min(item["val_loss"] for item in history),
            "seed": seed,
            "native_random_init": True,
            "dataset_mode": "creator_owned_micro",
            "train_csv": train_csv,
            "val_csv": val_csv,
            "latent_shape": list(latent_shape),
            "model_dim": model_dim,
            "text_dim": text_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "diffusion_steps": diffusion_steps,
            "prompt_conditioning_mean_abs": separation,
        },
        destination,
    )
    history_path = destination.with_name(destination.stem + "_history.jsonl")
    history_path.write_text(
        "\n".join(json.dumps(item) for item in history) + "\n",
        encoding="utf-8",
    )
    print(f"prompt_conditioning_mean_abs={separation:.8f}")
    print(f"saved={destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--autoencoder-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--text-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--out", default="checkpoints/nova_creator_latent_micro.pt")
    args = parser.parse_args()
    run(**vars(args))
