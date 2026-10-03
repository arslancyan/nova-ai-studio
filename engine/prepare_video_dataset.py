"""Prepare rights-documented local videos for native NOVA training.

No downloading is performed. The user supplies local video files and a
completed metadata CSV. FFmpeg converts each source into a fixed RGB tensor.
"""
from __future__ import annotations
import argparse, csv, random, subprocess
from pathlib import Path
import torch

REQUIRED = [
    "sample_id", "source_id", "video_path", "creator",
    "license_or_permission", "source_url", "acquired_at",
    "permitted_use", "attribution_required", "caption", "notes",
]

def read_metadata(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Metadata CSV is empty")
    missing = [x for x in REQUIRED if x not in rows[0]]
    if missing:
        raise ValueError("Missing metadata columns: " + ", ".join(missing))
    return rows

def video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"{video_path}: invalid duration {duration}")
    return duration

def decode_frames(video_path: Path, frames: int, height: int, width: int):
    duration = video_duration(video_path)
    sample_fps = frames / duration
    vf = (
        f"fps={sample_fps:.8f},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    command = [
        "ffmpeg", "-v", "error", "-i", str(video_path),
        "-vf", vf, "-frames:v", str(frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    expected = frames * height * width * 3
    if len(result.stdout) != expected:
        raise ValueError(f"{video_path}: expected {expected} bytes, got {len(result.stdout)}")
    raw = torch.frombuffer(bytearray(result.stdout), dtype=torch.uint8)
    return raw.view(frames, height, width, 3).permute(3, 0, 1, 2).float() / 127.5 - 1.0

def prepare(metadata="engine/REAL_VIDEO_TEMPLATE.csv", output="data/real",
            frames=8, height=64, width=64, val_fraction=0.1, seed=0):
    rows = read_metadata(Path(metadata))
    output = Path(output)
    clips_dir = output / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    prepared = []

    for row in rows:
        values = {k: row.get(k, "").strip() for k in REQUIRED}
        if any(not values[k] for k in REQUIRED):
            raise ValueError(f"{values.get('sample_id', '<unknown>')}: incomplete metadata")
        source = Path(values["video_path"])
        if not source.exists():
            raise FileNotFoundError(f"{values['sample_id']}: missing video {source}")
        clip = decode_frames(source, frames, height, width)
        destination = clips_dir / f"{values['sample_id']}.pt"
        torch.save(clip, destination)
        prepared.append({**values, "prepared_path": str(destination)})

    random.Random(seed).shuffle(prepared)
    val_size = max(1, int(round(len(prepared) * val_fraction))) if len(prepared) > 1 else 0
    train_rows, val_rows = prepared[:-val_size] if val_size else prepared, prepared[-val_size:] if val_size else []

    for name, split in (("train.csv", train_rows), ("val.csv", val_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=prepared[0].keys())
            writer.writeheader()
            writer.writerows(split)

    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prepared[0].keys())
        writer.writeheader()
        writer.writerows(prepared)
    print(f"prepared={len(prepared)} train={len(train_rows)} val={len(val_rows)} output={output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="engine/REAL_VIDEO_TEMPLATE.csv")
    p.add_argument("--output", default="data/real")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prepare(a.metadata, a.output, a.frames, a.height, a.width, a.val_fraction, a.seed)
