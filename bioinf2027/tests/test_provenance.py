"""Tests of the provenance records and of exact endpoint parsing."""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, features, provenance  # noqa: E402


def test_file_record_holds_portable_path_size_and_digest(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(tmp_path))
    target = tmp_path / "inputs" / "table.csv"
    target.parent.mkdir()
    target.write_bytes(b"a,b\n1,2\n")
    record = provenance.file_record(target)
    assert record == {
        "path": "$IGA_STORAGE_ROOT/inputs/table.csv",
        "bytes": 8,
        "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
    }
    assert provenance.file_record(tmp_path / "missing.csv") == {
        "path": "$IGA_STORAGE_ROOT/missing.csv",
        "exists": False,
    }


def test_folder_record_lists_every_file_and_a_listing_digest_that_recomputes(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(tmp_path))
    folder = tmp_path / "shards"
    (folder / "train").mkdir(parents=True)
    (folder / "test").mkdir()
    (folder / "train" / "b.npz").write_bytes(b"bb")
    (folder / "train" / "a.npz").write_bytes(b"a")
    (folder / "test" / "c.npz").write_bytes(b"ccc")
    paths = [folder / "train" / "b.npz", folder / "test" / "c.npz", folder / "train" / "a.npz"]

    record = provenance.folder_record(paths, folder)

    assert record["path"] == "$IGA_STORAGE_ROOT/shards"
    assert record["files"] == 3
    assert record["total_bytes"] == 6
    assert [entry["path"] for entry in record["entries"]] == ["test/c.npz", "train/a.npz", "train/b.npz"]
    assert record["entries"][1] == {"path": "train/a.npz", "bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()}
    listing = "\n".join("%s %d %s" % (e["path"], e["bytes"], e["sha256"]) for e in record["entries"])
    assert record["sha256_of_listing"] == hashlib.sha256(listing.encode()).hexdigest()


def test_strict_json_writes_non_finite_numbers_as_null():
    value = {"a": float("nan"), "b": [np.float64(1.5), np.inf], "c": np.int64(3), "d": np.bool_(True)}
    assert provenance.strict_json(value) == {"a": None, "b": [1.5, None], "c": 3, "d": True}


def test_read_endpoint_returns_the_exact_double_of_every_decimal(tmp_path):
    # The default parser reads this decimal one unit in the last place away.
    text = "2.5591081235012836"
    path = tmp_path / "endpoint.csv"
    path.write_text("construct,y_primary\nB,%s\nA,0.25\n" % text)
    table = features.read_endpoint(path)
    assert table["construct"].tolist() == ["A", "B"]
    assert table["y_primary"].iloc[1] == float(text)
    assert math.isclose(table["y_primary"].iloc[0], 0.25, rel_tol=0, abs_tol=0)
