"""Validate YouTube-derived candidates before they enter NOVA training data.

This tool intentionally does not download videos. It creates a conservative
candidate gate from metadata supplied by an administrator. A video must have
an explicit CC BY/public-domain declaration or documented direct permission
before it can be marked training_rights_verified.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ALLOWED_LICENSE_MARKERS = (
    "cc by",
    "creative commons attribution",
    "public domain",
    "cc0",
)

REQUIRED = {
    "sample_id",
    "source_id",
    "creator",
    "license_or_permission",
    "source_url",
    "permitted_use",
    "attribution_required",
}


def is_explicitly_eligible(row: dict) -> bool:
    license_text = (row.get("license_or_permission") or "").strip().lower()
    permitted = (row.get("permitted_use") or "").strip().lower()

    has_explicit_license = any(marker in license_text for marker in ALLOWED_LICENSE_MARKERS)
    commercial = "commercial" in permitted
    training = "train" in permitted or "machine learning" in permitted or "ml" in permitted

    return has_explicit_license and commercial and training


def validate(input_csv: Path, output_csv: Path) -> tuple[int, int]:
    with input_csv.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fields = list(reader.fieldnames or [])
        missing = REQUIRED - set(fields)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        approved = []
        rejected = []

        for row in reader:
            if is_explicitly_eligible(row):
                row["training_rights_verified"] = "manual_review_required"
                row["decision"] = "candidate"
                approved.append(row)
            else:
                row["training_rights_verified"] = "false"
                row["decision"] = "reject"
                rejected.append(row)

    out_fields = fields + ["training_rights_verified", "decision"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(approved + rejected)

    print(f"candidates={len(approved)} rejected={len(rejected)}")
    return len(approved), len(rejected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    validate(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
