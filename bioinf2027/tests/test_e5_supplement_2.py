"""make_e5_supplement_2.py: comparator names, the view of one comparator's statistics, and the readings."""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e5_supplement_2 as s2  # noqa: E402


def test_comparator_names():
    assert s2.comparator_name("J", "a") == "[Bc; FN; F_mean]"
    assert s2.comparator_name("F", "b") == "[Bc; F_mean; F_K562]"
    assert s2.comparator_name("FN", "a") == "[Bc; FN]" and s2.comparator_name("Bc", "b") == "Bc"


def test_view_maps_one_comparator():
    stats = {"names": [], "index_sha256": "x", "point": {f"{x}:{m}": float(i) for i, (x, m) in enumerate((x, m) for x in s2.COMPARATORS for m in ("bar", "e", "C", "S"))},
             "draws": {f"{x}:{m}": np.full(3, float(i)) for i, (x, m) in enumerate((x, m) for x in s2.COMPARATORS for m in ("bar", "e", "C", "S"))}}
    v = s2.view(stats, "F")
    assert v["point"] == {"bar": stats["point"]["F:bar"], "e": stats["point"]["F:e"], "C": stats["point"]["F:C"], "S": stats["point"]["F:S"]}
    assert v["index_sha256"] == "x"


def test_below_both_controls_needs_both_upper_ends_below_zero():
    assert s2.smaller_below({"minus_gaussian_hi": -0.02, "minus_shuffled_hi": -0.01})
    assert not s2.smaller_below({"minus_gaussian_hi": -0.02, "minus_shuffled_hi": 0.01})


def test_readings_of_the_score_against_the_vector_cover_every_state():
    assert set(s2.READ_SCORE) == {"above zero", "below zero", "contains zero"}


def test_default_is_not_a_smoke_and_writes_to_the_results():
    assert not s2.SMOKE and s2.out_dir() == s2.OUT
