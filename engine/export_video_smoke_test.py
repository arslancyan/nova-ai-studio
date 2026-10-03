"""Smoke-test the native tensor -> MP4 export boundary."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .export_video import export_video


def main() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="nova-export-test-") as tmp:
        root = Path(tmp)
        tensor_path = root / "sample.pt"
        output_path = root / "sample.mp4"

        # Tiny deterministic tensor; this tests only the native export boundary.
        sample = torch.zeros(1, 3, 4, 16, 16)
        sample[:, 0] = 1.0
        torch.save(sample, tensor_path)

        result = export_video(str(tensor_path), str(output_path), fps=4)
        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    print("native exporter smoke test passed")


if __name__ == "__main__":
    main()
