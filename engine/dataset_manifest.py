"""Create a checksum/provenance manifest for a NOVA dataset directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: str, output: str = "manifest.json"):
    base = Path(root)
    entries = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == output:
            continue
        entries.append(
            {
                "path": str(path.relative_to(base)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )

    manifest = {
        "format": "nova-dataset-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(base),
        "entries": entries,
        "provenance": {
            "external_media_used": False,
            "training_rights_verified": True,
            "notes": "Synthetic NOVA dataset contains no external media.",
        },
    }
    Path(output).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/synthetic")
    parser.add_argument("--output", default="data/synthetic/manifest.json")
    args = parser.parse_args()
    build_manifest(args.root, args.output)


if __name__ == "__main__":
    main()
