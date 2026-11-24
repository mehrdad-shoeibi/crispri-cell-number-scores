"""Check E0 runs once per code, environment and inputs, and a start with identical digests reuses its recorded pass
(the author's answer of 2026-11-15, bioinf2027/GATE.md)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e2_fits as mf  # noqa: E402

INPUTS = {"input_records": {"path": "bioinf2027/results/E1/input_records.json", "bytes": 1, "sha256": "a" * 64}}


@pytest.fixture()
def counted(tmp_path, monkeypatch):
    calls = []

    def check():
        calls.append(1)
        return [{"fit": "G5|existing|z|forest|0|all training rows|0", "sides_equal": {"val": True}, "passed": True}]

    monkeypatch.setattr(mf, "check_e0", check)
    monkeypatch.setattr(mf, "E0_RECORDS", tmp_path / "e0")
    monkeypatch.setattr(mf, "FRESH_E0", False)
    real = mf.file_record

    def record(path):  # the test's records lie outside the repository and the data root
        return real(path) if Path(path).resolve().is_relative_to(mf.REPO) else {"path": Path(path).name}

    monkeypatch.setattr(mf, "file_record", record)
    return calls


def test_the_check_runs_once_and_a_start_with_identical_digests_reuses_its_pass(counted, tmp_path):
    first = mf.e0_checked(INPUTS)
    second = mf.e0_checked(INPUTS)
    assert first["ran"] and not second["ran"] and len(counted) == 1
    assert second["result"] == first["result"] and second["key"] == first["key"]
    record = json.loads(next((tmp_path / "e0").glob("e0_*.json")).read_text())
    assert record["passed"] is True and set(record["digests"]) == {"code", "environment", "inputs"}
    assert "bioinf2027/scripts/make_e2_fits.py" in record["digests"]["code"]
    assert record["digests"]["environment"]["path"] == "bioinf2027/requirements.txt"


def test_any_change_of_an_input_digest_runs_it_again(counted):
    mf.e0_checked(INPUTS)
    changed = {"input_records": dict(INPUTS["input_records"], sha256="b" * 64)}
    assert mf.e0_checked(changed)["ran"] and len(counted) == 2


def test_a_change_of_the_code_or_the_environment_runs_it_again(counted, monkeypatch):
    mf.e0_checked(INPUTS)
    real = mf.e0_digests

    def code_changed(inputs):
        d = real(inputs)
        d["code"]["bioinf2027/scripts/make_e2_fits.py"] = "0" * 64
        return d

    monkeypatch.setattr(mf, "e0_digests", code_changed)
    assert mf.e0_checked(INPUTS)["ran"] and len(counted) == 2

    def environment_changed(inputs):
        d = real(inputs)
        d["environment"] = dict(d["environment"], sha256="0" * 64)
        return d

    monkeypatch.setattr(mf, "e0_digests", environment_changed)
    assert mf.e0_checked(INPUTS)["ran"] and len(counted) == 3


def test_fresh_e0_runs_it_regardless(counted, monkeypatch):
    mf.e0_checked(INPUTS)
    monkeypatch.setattr(mf, "FRESH_E0", True)
    assert mf.e0_checked(INPUTS)["ran"] and len(counted) == 2


def test_a_record_that_is_not_a_pass_or_whose_digests_do_not_give_its_key_is_not_reused(counted, tmp_path):
    mf.e0_checked(INPUTS)
    path = next((tmp_path / "e0").glob("e0_*.json"))
    record = json.loads(path.read_text())
    path.write_text(json.dumps(dict(record, passed=False)))
    assert mf.e0_checked(INPUTS)["ran"] and len(counted) == 2
    record = json.loads(path.read_text())
    record["digests"]["inputs"]["input_records"]["sha256"] = "c" * 64
    path.write_text(json.dumps(record))
    assert mf.e0_checked(INPUTS)["ran"] and len(counted) == 3


def test_a_failed_check_records_nothing(tmp_path, monkeypatch):
    def failing():
        raise mf.Stop("check E0 failed")

    monkeypatch.setattr(mf, "check_e0", failing)
    monkeypatch.setattr(mf, "E0_RECORDS", tmp_path / "e0")
    monkeypatch.setattr(mf, "FRESH_E0", False)
    with pytest.raises(mf.Stop):
        mf.e0_checked(INPUTS)
    assert not (tmp_path / "e0").exists() or not list((tmp_path / "e0").iterdir())


# --------------------------------------------------------------------------- gate G3, items 1 and 2 (the author's answers of 2026-11-16)

def test_check_e0_refits_four_e1b_fits_each_in_e1b_s_manifest():
    assert len(mf.E0_FITS) == 4 and len(set(mf.E0_FITS)) == 4
    assert {f[3] for f in mf.E0_FITS} == {"forest", "boosting", "ridge"} and {f[1] for f in mf.E0_FITS} == {"existing", "rebuilt"}
    files = json.loads((mf.e1f.OUT / "predictions_manifest.json").read_text())["files"]
    for cell, label, _, head, base, seed in mf.E0_FITS:
        assert mf.FitSpec(cell, label, base, head, seed).key() in files


def test_check_1_at_load_is_recorded_with_the_pass_only_when_it_ran(counted, tmp_path, monkeypatch):
    plain = mf.e0_checked(INPUTS)
    assert "check_1_at_load" not in json.loads((tmp_path / "e0" / Path(plain["record"]["path"]).name).read_text())
    monkeypatch.setattr(mf, "FRESH_E0", True)
    verified = {"train": {"sha256": "d" * 64, "matches_input_records": True}}
    with_check = mf.e0_checked(INPUTS, verified)
    record = json.loads((tmp_path / "e0" / Path(with_check["record"]["path"]).name).read_text())
    assert record["check_1_at_load"]["files"] == verified and record["check_1_at_load"]["passed"] is True and record["fresh"] is True


def test_the_check_only_run_takes_no_encoder(monkeypatch):
    import subprocess

    usage = subprocess.run([sys.executable, "-B", str(Path(mf.__file__)), "--check-e0-only", "scgpt"], capture_output=True, text=True)
    assert usage.returncode != 0 and "--check-e0-only [--fresh-e0] [--verify-inputs]" in usage.stderr


def test_check_e0_fits_each_of_its_heads_as_fit_does():
    import numpy as np

    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(60, 4)), rng.normal(size=60)
    x_val, val = rng.normal(size=(20, 4)), np.where(np.arange(20) % 4 == 0, np.nan, rng.normal(size=20))
    for _, label, _, head, base, seed in mf.E0_FITS:
        predict = mf.e0_model(mf.FitSpec("G6", label, base, head, seed), x, y, x_val, val)
        assert predict(x_val).shape == (20,)
    ridge = mf.FitSpec("G6", "rebuilt", "z", "ridge", 0)
    mask = np.isfinite(val)
    model, scaler, _ = mf.heads.fit_ridge_by_validation(x, y, x_val[mask], val[mask])
    assert np.array_equal(mf.e0_model(ridge, x, y, x_val, val)(x_val), model.predict(scaler.transform(x_val)))
