"""Tests of E6's input check (plan Section 3.4, stop rule 1): the list of inputs and the stop on a changed file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e6_input_records as records  # noqa: E402
import make_e6_stats as stats  # noqa: E402
from bioinf2027.provenance import sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_every_listed_input_has_a_record_with_its_path():
    listed = records.listed()
    assert len(listed) == 15
    assert len({name for name, *_ in listed}) == 15
    for name, path, record_file, entry, read_by in listed:
        assert (REPO / record_file).exists()
        assert records.config.portable(path) == entry["path"], name
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0
        assert read_by


def test_the_stored_z_and_its_index_are_listed():
    names = {name for name, *_ in records.listed()}
    for screen in ("RPE1", "K562-essential"):
        assert {f"{screen}_z", f"{screen}_z_index", f"{screen}_eligibility", f"{screen}_single_cell"} <= names


def _fake_listed(tmp_path: Path, recorded: str):
    path = tmp_path / "input.npy"
    path.write_bytes(b"stored z")
    entry = {"path": str(path), "bytes": path.stat().st_size, "sha256": recorded}
    return lambda: [("RPE1_z", path, "bioinf2027/results/E1/input_records.json", entry, "test")]


def test_verify_inputs_passes_on_an_unchanged_file(tmp_path, monkeypatch):
    path = tmp_path / "input.npy"
    path.write_bytes(b"stored z")
    monkeypatch.setattr(stats.input_records, "listed", _fake_listed(tmp_path, sha256(path)))
    monkeypatch.setattr(stats.config, "portable", str)  # the test file lies outside the data root
    out = stats.verify_inputs()
    assert out["RPE1_z"]["sha256"] == sha256(path)
    json.dumps(out)


def test_verify_inputs_stops_on_a_changed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(stats.input_records, "listed", _fake_listed(tmp_path, "0" * 64))
    with pytest.raises(SystemExit, match="stop rule 1"):
        stats.verify_inputs()
