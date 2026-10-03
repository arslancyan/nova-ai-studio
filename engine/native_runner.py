"""Run a NOVA Creative Director job through the native engine and export MP4.

This runner is intentionally explicit: if native checkpoints are unavailable it
fails clearly instead of falling back to the preview renderer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export_video import export_video
from .latent_inference import generate_job


def run_job(job: dict, output_mp4: str) -> Path:
    root = Path(output_mp4)
    root.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = root.with_suffix(".pt")
    generate_job(job, output=str(tensor_path))
    fps = int(job.get("fps", 8))
    export_video(str(tensor_path), str(root), fps=fps)
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Path to a NOVA generation job JSON")
    parser.add_argument("--output", default="outputs/nova_native.mp4")
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        raise FileNotFoundError(f"job file not found: {job_path}")

    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = run_job(job, args.output)
    print(f"native NOVA video saved: {output}")


if __name__ == "__main__":
    main()
