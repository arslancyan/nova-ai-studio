"""Validate the documented real-video dataset contract.

This checks metadata completeness, split integrity, prepared tensors, captions,
and value ranges. It does not make a legal determination about licenses.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

REQUIRED = [
    "sample_id", "source_id", "video_path", "creator",
    "license_or_permission", "source_url", "acquired_at",
    "permitted_use", "attribution_required", "caption", "notes",
]


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows:
        missing = [field for field in REQUIRED + ["prepared_path"] if field not in rows[0]]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    return rows


def validate(root="data/real", require_rights_verified=False):
    root = Path(root)
    metadata = read_csv(root / "metadata.csv")
    train = read_csv(root / "train.csv")
    val = read_csv(root / "val.csv")

    if not metadata:
        raise ValueError("metadata.csv is empty")
    if not train:
        raise ValueError("train.csv is empty")
    if not val:
        raise ValueError("val.csv is empty; keep at least one held-out clip")

    metadata_ids = {row["sample_id"] for row in metadata}
    train_ids = {row["sample_id"] for row in train}
    val_ids = {row["sample_id"] for row in val}

    if len(metadata_ids) != len(metadata):
        raise ValueError("duplicate sample_id in metadata.csv")
    if train_ids & val_ids:
        raise ValueError("train/val split overlap detected")
    if not train_ids <= metadata_ids or not val_ids <= metadata_ids:
        raise ValueError("split contains unknown sample_id")

    if require_rights_verified:
        for row in metadata:
            if row.get("training_rights_verified", "").strip().lower() != "true":
                raise ValueError(f"{row.get('sample_id', '<unknown>')}: training_rights_verified must be true")

    for row in metadata:
        for field in REQUIRED:
            if not row.get(field, "").strip():
                raise ValueError(f"{row.get('sample_id', '<unknown>')}: empty {field}")
        prepared = Path(row["prepared_path"])
        if not prepared.exists():
            raise FileNotFoundError(f"{row['sample_id']}: missing {prepared}")
        tensor = torch.load(prepared, map_location="cpu", weights_only=True)
        if tensor.ndim != 4 or tuple(tensor.shape[:1]) != (3,):
            raise ValueError(f"{row['sample_id']}: expected [3,T,H,W], got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{row['sample_id']}: NaN/inf detected")
        if float(tensor.min()) < -1.0001 or float(tensor.max()) > 1.0001:
            raise ValueError(f"{row['sample_id']}: values outside [-1,1]")

    print(
        "real dataset valid: "
        f"samples={len(metadata)} train={len(train)} val={len(val)} "
        "provenance_fields=complete"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/real")
    parser.add_argument("--require-rights-verified", action="store_true")
    args = parser.parse_args()
    validate(args.root, require_rights_verified=args.require_rights_verified)
