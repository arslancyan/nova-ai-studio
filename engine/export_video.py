"""Convert a native NOVA tensor sample into an MP4.

This utility only accepts a tensor produced by the native NOVA inference path.
It never calls an external video generator.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image


def export_video(
    tensor_path: str,
    output_path: str,
    fps: int = 8,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive")
    video = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("Expected one generated video")
        video = video[0]
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("Expected [3,T,H,W] video tensor")
    if not torch.isfinite(video).all():
        raise ValueError("Video tensor contains NaN or infinity")

    video = video.clamp(-1, 1)
    frames = video.permute(1, 2, 3, 0)
    workdir = Path(tempfile.mkdtemp(prefix="nova-native-"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for index, frame in enumerate(frames):
            pixels = ((frame + 1.0) * 127.5).round().byte().numpy()
            Image.fromarray(pixels, "RGB").save(
                workdir / f"frame_{index:05d}.png"
            )

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to export NOVA native video")

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(workdir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)
        return output
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tensor")
    parser.add_argument("output")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    path = export_video(args.tensor, args.output, args.fps)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
