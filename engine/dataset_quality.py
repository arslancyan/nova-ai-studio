"""Quality gate for creator-owned prepared video datasets.

Checks provenance completeness, source diversity, split integrity, caption
coverage and window concentration before native training. It never decides
legal permission itself; rights verification remains an explicit metadata
field.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(train_csv, val_csv, min_sources=4, require_rights_verified=False):
    train = read_rows(train_csv)
    val = read_rows(val_csv)
    all_rows = train + val
    errors, warnings = [], []

    if not all_rows:
        errors.append("dataset is empty")
        return {"ok": False, "errors": errors, "warnings": warnings}

    required = {
        "source_id", "creator", "ownership", "permission", "caption",
        "prepared_path", "source_window_start", "training_rights_verified"
    }
    missing = sorted(required - set(all_rows[0]))
    if missing:
        errors.append("missing columns: " + ", ".join(missing))

    sources = sorted({r.get("source_id", "").strip() for r in all_rows if r.get("source_id")})
    train_sources = {r.get("source_id", "").strip() for r in train}
    val_sources = {r.get("source_id", "").strip() for r in val}

    if len(sources) < min_sources:
        warnings.append(f"only {len(sources)} independent sources; target is {min_sources}+")
    if not val_sources:
        errors.append("validation split has no sources")

    counts = Counter(r.get("source_id", "").strip() for r in train)
    if counts:
        largest = max(counts.values())
        smallest = min(counts.values())
        if smallest and largest / smallest > 6:
            warnings.append("training windows are highly concentrated in one source")

    if require_rights_verified:
        bad = [r.get("source_id", "") for r in all_rows if r.get("training_rights_verified", "").strip().lower() != "true"]
        if bad:
            errors.append("rights verification missing/false for: " + ", ".join(sorted(set(bad))))

    for r in all_rows:
        sid = r.get("source_id", "").strip()
        if r.get("ownership", "").strip().lower() not in {"creator-owned", "owned"}:
            errors.append(f"{sid}: ownership must be creator-owned")
        if not r.get("creator", "").strip():
            errors.append(f"{sid}: creator is missing")
        if not r.get("permission", "").strip():
            errors.append(f"{sid}: permission evidence is missing")
        if len(r.get("caption", "").strip()) < 8:
            warnings.append(f"{sid}: caption is very short")
        if not Path(r.get("prepared_path", "")).exists():
            warnings.append(f"{sid}: prepared file is not present at audit time")

    report = {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "source_count": len(sources),
        "train_window_count": len(train),
        "validation_window_count": len(val),
        "train_windows_by_source": dict(sorted(counts.items())),
        "validation_sources": sorted(val_sources),
        "all_sources_in_training": sorted(train_sources),
        "independent_source_count": len(sources),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--min-sources", type=int, default=4)
    parser.add_argument("--require-rights-verified", action="store_true")
    parser.add_argument("--report", default="dataset_quality_report.json")
    args = parser.parse_args()
    report = audit(
        args.train_csv,
        args.val_csv,
        min_sources=args.min_sources,
        require_rights_verified=args.require_rights_verified,
    )
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)
