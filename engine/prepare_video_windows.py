"""Create fixed-duration training windows from local, rights-documented videos.

Windows increase sample count without pretending they create new visual
diversity. Every generated row keeps source provenance and an explicit rights
status. No network access or downloading is performed.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path

import torch

REQUIRED = [
    "sample_id",
    "video_path",
    "creator",
    "ownership",
    "permission",
    "media_category",
    "caption",
    "notes",
]


def read_metadata(path, require_rights_verified=False):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata CSV is empty")
    missing = [x for x in REQUIRED if x not in rows[0]]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))

    source_ids = [row["sample_id"].strip() for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("sample_id must uniquely identify each source video")

    if require_rights_verified and "training_rights_verified" not in rows[0]:
        raise ValueError(
            "training_rights_verified column is required with --require-rights-verified"
        )

    for row in rows:
        ownership = row["ownership"].strip().lower()
        if ownership not in {"creator-owned", "owned"}:
            raise ValueError(f'{row["sample_id"]}: not marked creator-owned')
        if not row["creator"].strip() or not row["permission"].strip():
            raise ValueError(
                f'{row["sample_id"]}: creator and permission evidence are required'
            )
        if require_rights_verified:
            verified = row.get("training_rights_verified", "").strip().lower()
            if verified != "true":
                raise ValueError(
                    f'{row["sample_id"]}: training_rights_verified must be true'
                )
    return rows


def duration(path):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return float(out)


def decode_window(path, start, seconds, frames, height, width):
    if frames < 4 or frames % 4:
        raise ValueError("frames must be >=4 and divisible by 4")
    if height < 4 or width < 4 or height % 4 or width % 4:
        raise ValueError("height/width must be >=4 and divisible by 4")

    fps = frames / seconds
    vf = (
        f"fps={fps:.8f},scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(path),
        "-t",
        f"{seconds:.6f}",
        "-vf",
        vf,
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    expected = frames * height * width * 3
    if len(result.stdout) != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes, got {len(result.stdout)}"
        )
    raw = torch.frombuffer(bytearray(result.stdout), dtype=torch.uint8)
    return (
        raw.view(frames, height, width, 3)
        .permute(3, 0, 1, 2)
        .float()
        / 127.5
        - 1.0
    )


def prepare(
    metadata,
    output="data/creator_windows",
    window_seconds=4.0,
    stride_seconds=2.0,
    frames=16,
    height=64,
    width=64,
    max_windows_per_source=24,
    require_rights_verified=False,
):
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds and stride_seconds must be positive")
    rows = read_metadata(metadata, require_rights_verified=require_rights_verified)

    root = Path(output)
    clips_dir = root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    prepared = []

    for row in rows:
        source = Path(row["video_path"])
        if not source.exists():
            raise FileNotFoundError(source)

        total = duration(source)
        if total < window_seconds:
            starts = [0.0]
        else:
            count = min(
                max_windows_per_source,
                max(1, math.floor((total - window_seconds) / stride_seconds) + 1),
            )
            starts = [
                min(i * stride_seconds, total - window_seconds)
                for i in range(count)
            ]

        for index, start in enumerate(starts):
            tensor = decode_window(
                source, start, window_seconds, frames, height, width
            )
            sample_id = f'{row["sample_id"]}-w{index:03d}'
            destination = clips_dir / f"{sample_id}.pt"
            torch.save(tensor, destination)

            prepared.append(
                {
                    **{k: row.get(k, "").strip() for k in REQUIRED},
                    "source_id": row["sample_id"].strip(),
                    "source_duration_seconds": f"{total:.6f}",
                    "window_index": str(index),
                    "source_window_start": f"{start:.6f}",
                    "source_window_seconds": f"{window_seconds:.6f}",
                    "training_rights_verified": row.get(
                        "training_rights_verified", "manual_review_required"
                    ).strip(),
                    "prepared_path": str(destination),
                }
            )

    train_rows = []
    val_rows = []
    by_source = {}
    for row in prepared:
        by_source.setdefault(row["source_id"], []).append(row)

    for source_rows in by_source.values():
        source_rows.sort(key=lambda x: float(x["source_window_start"]))
        if len(source_rows) == 1:
            train_rows.extend(source_rows)
        else:
            # Hold out the final chronological window for every source.
            val_rows.append(source_rows[-1])
            train_rows.extend(source_rows[:-1])

    fields = list(prepared[0].keys())
    for name, split in [
        ("metadata.csv", prepared),
        ("train.csv", train_rows),
        ("val.csv", val_rows),
    ]:
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split)

    print(
        f"windows={len(prepared)} train={len(train_rows)} "
        f"val={len(val_rows)} sources={len(rows)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="data/creator_windows")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--max-windows-per-source", type=int, default=24)
    parser.add_argument("--require-rights-verified", action="store_true")
    args = parser.parse_args()
    prepare(**vars(args))
