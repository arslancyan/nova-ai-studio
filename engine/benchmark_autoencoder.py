"""Benchmark a native NOVA autoencoder on prepared video clips.

Reports L1, MSE and PSNR over a CSV-defined split. It is deliberately
checkpoint-agnostic so synthetic and rights-documented real data use the same
measurement path.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from .autoencoder import build_autoencoder


def benchmark(checkpoint, csv_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_autoencoder(
        in_channels=3,
        latent_channels=int(pack.get("latent_channels", 8)),
    ).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty benchmark split: {csv_path}")

    l1_total = 0.0
    mse_total = 0.0
    count = 0
    with torch.no_grad():
        for row in rows:
            video = torch.load(
                row["prepared_path"], map_location=device, weights_only=True
            ).float().unsqueeze(0)
            reconstruction, latent = model(video)
            l1 = torch.nn.functional.l1_loss(reconstruction, video)
            mse = torch.nn.functional.mse_loss(reconstruction, video)
            l1_total += float(l1)
            mse_total += float(mse)
            count += 1

    l1 = l1_total / count
    mse = mse_total / count
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(4.0 / mse)
    result = {
        "checkpoint": str(checkpoint),
        "split": str(csv_path),
        "samples": count,
        "l1": l1,
        "mse": mse,
        "psnr_db": psnr,
        "input_range": "[-1,1]",
        "native_random_init": True,
    }
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    benchmark(args.checkpoint, args.csv)
