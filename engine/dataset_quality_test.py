"""Regression tests for creator dataset quality gates."""
from pathlib import Path

from .dataset_quality import audit


def _write(path, rows):
    path.write_text(
        "source_id,creator,ownership,permission,caption,prepared_path,source_window_start,training_rights_verified\n"
        + "\n".join(",".join(row) for row in rows),
        encoding="utf-8",
    )


def test_quality_gate_accepts_balanced_creator_sources(tmp_path):
    rows = [
        ("a","Creator A","creator-owned","self-created","red character walking",str(tmp_path/"a.pt"),"0","true"),
        ("b","Creator B","creator-owned","self-created","blue car driving",str(tmp_path/"b.pt"),"0","true"),
        ("c","Creator C","creator-owned","self-created","shop at dusk",str(tmp_path/"c.pt"),"0","true"),
        ("d","Creator D","creator-owned","self-created","motorcycle street",str(tmp_path/"d.pt"),"0","true"),
    ]
    for row in rows:
        Path(row[5]).touch()
    train=tmp_path/"train.csv"; val=tmp_path/"val.csv"
    _write(train,rows)
    _write(val,rows)
    result=audit(train,val,require_rights_verified=True)
    assert result["ok"] is True


def test_quality_gate_rejects_missing_rights(tmp_path):
    rows=[
        ("a","Creator A","creator-owned","self-created","red character",str(tmp_path/"a.pt"),"0","false"),
        ("b","Creator B","creator-owned","self-created","blue car",str(tmp_path/"b.pt"),"0","true"),
        ("c","Creator C","creator-owned","self-created","shop scene",str(tmp_path/"c.pt"),"0","true"),
        ("d","Creator D","creator-owned","self-created","street scene",str(tmp_path/"d.pt"),"0","true"),
    ]
    train=tmp_path/"train.csv"; val=tmp_path/"val.csv"
    _write(train,rows); _write(val,rows)
    result=audit(train,val,require_rights_verified=True)
    assert result["ok"] is False
