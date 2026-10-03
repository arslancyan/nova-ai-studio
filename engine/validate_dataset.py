"""Validate the NOVA provenance manifest before training.

This checks documentation completeness. It does not make legal judgments
about whether a license permits machine-learning or commercial use.
"""
import csv
from pathlib import Path

REQUIRED = [
    "source_id", "source_name", "creator", "license_or_permission",
    "url", "acquired_at", "permitted_use", "attribution_required", "notes"
]

def validate_manifest(path="engine/DATASET_TEMPLATE.csv"):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)

    errors = [f"missing header: {field}" for field in REQUIRED if field not in headers]
    seen = set()

    for number, row in enumerate(rows, start=2):
        source_id = row.get("source_id", "").strip()
        if not source_id:
            errors.append(f"row {number}: source_id is empty")
        elif source_id in seen:
            errors.append(f"row {number}: duplicate source_id '{source_id}'")
        seen.add(source_id)

        for field in REQUIRED:
            if not row.get(field, "").strip():
                errors.append(f"row {number}: {field} is empty")

    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))

    print(f"NOVA dataset manifest: PASS ({len(rows)} documented sources)")

if __name__ == "__main__":
    validate_manifest()
