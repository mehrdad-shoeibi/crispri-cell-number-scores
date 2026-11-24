"""The E4 builder writes feature files that load without pickle.

The fits and statistics load every E4 feature file with allow_pickle=False. The builder's main stage first
wrote the construct names as object arrays (pandas' to_numpy of strings), and every smoke fit failed on
loading them (2026-11-10). These tests hold the builder's text conversion and its refusal to save objects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e4_features as feat  # noqa: E402


def test_text_turns_pandas_strings_into_fixed_width_text():
    names = pd.Series(["RPL3_P1", "SF3B1_P2"]).astype(str)
    assert names.to_numpy().dtype == object
    converted = feat.text(names)
    assert converted.dtype.kind == "U"
    assert converted.tolist() == ["RPL3_P1", "SF3B1_P2"]


def test_save_npz_round_trips_without_pickle(tmp_path):
    arrays = {"RPE1_x": np.ones((2, 3), dtype=np.float32), "RPE1_constructs": feat.text(pd.Series(["a", "bb"]).astype(str))}
    feat.save_npz(tmp_path / "f.npz", arrays)
    with np.load(tmp_path / "f.npz", allow_pickle=False) as store:
        assert store["RPE1_constructs"].tolist() == ["a", "bb"]
        assert np.array_equal(store["RPE1_x"], arrays["RPE1_x"])


def test_save_npz_refuses_object_arrays(tmp_path):
    arrays = {"RPE1_x": np.ones(2), "RPE1_constructs": pd.Series(["a", "bb"]).astype(str).to_numpy()}
    with pytest.raises(TypeError, match="RPE1_constructs"):
        feat.save_npz(tmp_path / "f.npz", arrays)
    assert not (tmp_path / "f.npz").exists()
