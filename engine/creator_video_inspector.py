"""Inspect creator-owned local videos and build an auditable provenance manifest.

No network access and no downloading are performed.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess
from pathlib import Path

REQUIRED = ["sample_id","video_path","creator","ownership","permission","media_category","caption","notes"]

def sha256(path, chunk_size=1024*1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def ffprobe(path):
    command = ["ffprobe","-v","error","-select_streams","v:0",
               "-show_entries","stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration,size",
               "-of","json",str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    frames = stream.get("nb_frames")
    return {
        "codec": stream.get("codec_name",""),
        "width": int(stream["width"]) if stream.get("width") else None,
        "height": int(stream["height"]) if stream.get("height") else None,
        "frame_rate": stream.get("avg_frame_rate") or stream.get("r_frame_rate",""),
        "frames": int(frames) if frames and str(frames).isdigit() else None,
        "duration_seconds": float(fmt["duration"]) if fmt.get("duration") else None,
        "size_bytes": int(fmt["size"]) if fmt.get("size") else path.stat().st_size,
    }

def inspect(input_csv, output_json):
    rows = list(csv.DictReader(Path(input_csv).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("input CSV is empty")
    missing = [field for field in REQUIRED if field not in rows[0]]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))
    manifest = []
    for row in rows:
        path = Path(row["video_path"]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'{row["sample_id"]}: missing {path}')
        if row["ownership"].strip().lower() not in {"creator-owned","owned"}:
            raise ValueError(f'{row["sample_id"]}: creator inspector only accepts creator-owned media')
        if not row["permission"].strip():
            raise ValueError(f'{row["sample_id"]}: permission is required')
        media = ffprobe(path)
        width, height = media["width"], media["height"]
        aspect = None if not width or not height else round(width / height, 4)
        manifest.append({
            "sample_id": row["sample_id"],
            "video_path": str(path.resolve()),
            "creator": row["creator"],
            "ownership": row["ownership"],
            "permission": row["permission"],
            "media_category": row["media_category"],
            "caption": row["caption"],
            "notes": row["notes"],
            "sha256": sha256(path),
            "aspect_ratio": aspect,
            "media": media,
        })
    payload = {"schema_version":"creator-owned-0.1","training_rights_verified":True,
               "network_access":False,"downloaded":False,"samples":manifest}
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(f"creator-owned manifest written: {destination} samples={len(manifest)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/creator_owned/manifest.json")
    args = parser.parse_args()
    inspect(args.input, args.output)
