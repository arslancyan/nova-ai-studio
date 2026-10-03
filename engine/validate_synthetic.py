"""Validate the deterministic synthetic NOVA dataset contract."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def validate(root="data/synthetic"):
    root = Path(root)
    clips = torch.load(root / "clips.pt", map_location="cpu", weights_only=True)
    captions = (root / "captions.txt").read_text(encoding="utf-8").splitlines()

    if clips.ndim != 5:
        raise ValueError(f"Expected 5D clips, got {clips.ndim}D")
    if clips.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {clips.shape[1]}")
    if clips.shape[0] == 0:
        raise ValueError("Dataset is empty")
    if len(captions) != clips.shape[0]:
        raise ValueError("Caption count does not match clip count")
    if not torch.isfinite(clips).all():
        raise ValueError("Dataset contains NaN or infinity")
    if clips.min() < -1.0001 or clips.max() > 1.0001:
        raise ValueError("Dataset values must stay in [-1, 1]")
    if any(not caption.strip() for caption in captions):
        raise ValueError("Dataset contains an empty caption")

    print(
        "synthetic dataset valid: "
        f"samples={clips.shape[0]} shape={tuple(clips.shape[1:])}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/synthetic")
    args = parser.parse_args()
    validate(args.root)
