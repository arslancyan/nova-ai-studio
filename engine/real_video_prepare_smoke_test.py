"""End-to-end smoke test for local real-video preparation.

The test generates its own tiny MP4 with FFmpeg, so no external media or
copyrighted source is involved. It also verifies that sampling reaches both
halves of the source instead of silently taking only the beginning.
"""
from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import torch

from .prepare_video_dataset import prepare
from .validate_real_dataset import validate


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "synthetic_source.mp4"

        # 2-second source: red first half, blue second half.
        command = [
            "ffmpeg", "-v", "error",
            "-f", "lavfi", "-i",
            "color=c=red:s=64x64:r=8:d=1",
            "-f", "lavfi", "-i",
            "color=c=blue:s=64x64:r=8:d=1",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-y", str(source),
        ]
        subprocess.run(command, check=True)

        metadata = root / "metadata.csv"
        with metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "sample_id", "source_id", "video_path", "creator",
                "license_or_permission", "source_url", "acquired_at",
                "permitted_use", "attribution_required", "caption", "notes",
            ])
            writer.writeheader()
            writer.writerow({
                "sample_id": "smoke-001",
                "source_id": "ci-generated",
                "video_path": str(source),
                "creator": "NOVA CI",
                "license_or_permission": "generated in CI",
                "source_url": "local://ci-generated",
                "acquired_at": "CI",
                "permitted_use": "test only",
                "attribution_required": "no",
                "caption": "a color field changing over time",
                "notes": "Synthetic smoke-test media; not training data.",
            })

        output = root / "data"
        prepare(metadata, output, frames=8, height=64, width=64, val_fraction=0.5, seed=0)
        validate(output)

        # Also exercise the real-dataset checksum path without asserting legal rights.
        from .dataset_manifest import build_manifest
        build_manifest(
            str(output),
            str(output / "manifest.json"),
            "CI-generated media only; no external rights claim.",
        )

        tensor = torch.load(output / "clips" / "smoke-001.pt", map_location="cpu", weights_only=True)
        left = tensor[:, :4].mean(dim=(0, 2, 3))
        right = tensor[:, 4:].mean(dim=(0, 2, 3))
        if float(left[0]) <= float(left[2]):
            raise AssertionError("first half was not predominantly red")
        if float(right[2]) <= float(right[0]):
            raise AssertionError("second half was not predominantly blue")

        print("real video preparation smoke test passed")


if __name__ == "__main__":
    main()
