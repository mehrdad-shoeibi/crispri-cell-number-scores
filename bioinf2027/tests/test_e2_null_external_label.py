"""make_e2_null_external.py --label: with the default label every path is (d)'s; another label writes apart."""
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e2_null_external as ne  # noqa: E402
from bioinf2027 import config  # noqa: E402


@pytest.fixture
def label(monkeypatch):
    def set_label(value: str, smoke: bool = False) -> None:
        monkeypatch.setattr(ne, "LABEL", value)
        monkeypatch.setattr(ne, "SMOKE", smoke)

    return set_label


def test_default_label_keeps_the_paths_of_d(label):
    label("rebuilt")
    root = config.storage_root() / "bioinf2027"
    assert ne.outputs_dir() == ne.RESULTS
    assert ne.results_dir() == ne.RESULTS
    assert ne.predictions_dir() == root / "E2" / "predictions_null_external"
    assert ne.plan() == ne.PLAN and ne.status() == ne.POST_HOC
    label("rebuilt", smoke=True)
    assert ne.results_dir() == root / "scratch" / "E2_null_external_smoke"
    assert ne.predictions_dir() == root / "scratch" / "E2_null_external_smoke" / "predictions_null_external"


def test_existing_label_writes_apart(label):
    label("existing")
    root = config.storage_root() / "bioinf2027"
    assert ne.outputs_dir() == ne.FOLLOWUP / "existing_label"
    assert ne.results_dir() == ne.FOLLOWUP / "existing_label"
    assert ne.predictions_dir() == root / "E2" / "predictions_null_external_existing"
    assert ne.plan() == ne.FOLLOWUP_PLAN and "follow-up (4)" in ne.status()
    label("existing", smoke=True)
    assert ne.results_dir() == root / "scratch" / "E2_null_external_existing_smoke"
    assert ne.predictions_dir() == root / "scratch" / "E2_null_external_existing_smoke" / "predictions_null_external_existing"


def test_features_record_is_ds_for_every_label(label):
    for value in ne.LABELS:
        label(value)
        assert ne.FEATURES_JSON == ne.RESULTS / "features.json"


def test_reading_on_the_existing_label_covers_every_state():
    assert set(ne.READING_EXISTING) == {"below zero", "contains zero", "above zero"}


def test_vcc_margin_qualifier_reads_its_row(label, tmp_path):
    import pandas as pd

    label("existing")
    vcc = pd.DataFrame([
        {"encoder": "scgpt", "cell": "G5", "label": "existing", "head": "forest", "quantity": "N0 - C1", "r2": 0.02, "r2_lo": -0.15, "r2_hi": 0.13},
        {"encoder": "scgpt", "cell": "G6", "label": "existing", "head": "forest", "quantity": "N0 - C1", "r2": 0.25, "r2_lo": 0.15, "r2_hi": 0.39},
    ])
    assert "is not shown" in ne.vcc_margin(vcc, "scgpt", "G5") and "vcc.csv:2)" in ne.vcc_margin(vcc, "scgpt", "G5")
    assert "is shown" in ne.vcc_margin(vcc, "scgpt", "G6") and "vcc.csv:3)" in ne.vcc_margin(vcc, "scgpt", "G6")
    with pytest.raises(ne.Stop):
        ne.vcc_margin(vcc, "stack", "G5")
