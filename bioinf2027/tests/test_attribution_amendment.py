"""Reading, row and lookup logic of the attribution amendment (bioinf2027/plans/E1b_attribution.md, Section 11)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e1b_attribution_amendment as am  # noqa: E402


def test_h1_reading_is_withdrawn_when_a_block_of_the_same_width_also_raises_r2():
    above = am.READINGS["h1"][0]
    assert am.reading("h1", "above zero", ("contains zero", "below zero")) == above
    assert am.reading("h1", "above zero", ("above zero", "contains zero")).startswith("not read")
    assert am.reading("h1", "above zero", ("contains zero", "above zero")).startswith("not read")
    assert am.reading("h1", "contains zero", ("above zero", "above zero")) == am.READINGS["h1"][1]


def test_d_half_prime_is_read_on_its_upper_bound_only():
    assert am.reading("D1/2'", "above zero", ("", "")) == "not read"
    assert am.reading("D1/2'", "contains zero", ("", "")) == "not read"
    assert am.reading("D1/2'", "below zero", ("", "")) == am.READINGS["D1/2'"][2]


def _frame(states: dict, d_half_hi: dict) -> pd.DataFrame:
    rows = []
    for head in ("forest", "boosting"):
        for name in am.READ:
            rows.append({"cell": "G6", "label": "existing", "head": head, "quantity": name, "state": states[head].get(name, "contains zero"),
                         "r2_hi": d_half_hi[head] if name == "D1/2'" else 0.1})
    return pd.DataFrame(rows)


def test_rows_follow_the_states_and_boosting_agreement():
    states = {"forest": {"Delta'": "above zero", "h3": "above zero", "h3'": "contains zero"},
              "boosting": {"Delta'": "contains zero", "h3": "above zero", "h3'": "contains zero"}}
    out = am.s2_rows(_frame(states, {"forest": -0.01, "boosting": 0.02}))
    forest = out[out["head"] == "forest"].set_index("row")
    assert forest.loc["B'", "holds"] and forest.loc["C'", "holds"] and forest.loc["E'", "holds"]
    assert not forest.loc["E", "holds"] and not forest.loc["F", "holds"]
    assert not forest.loc["B'", "boosting_agrees"] and not forest.loc["C'", "boosting_agrees"]
    assert forest.loc["E'", "boosting_agrees"]


def test_row_f_holds_when_neither_contrast_is_above_zero():
    states = {"forest": {"h3": "below zero", "h3'": "contains zero"}, "boosting": {}}
    out = am.s2_rows(_frame(states, {"forest": 0.05, "boosting": 0.05}))
    forest = out[out["head"] == "forest"].set_index("row")
    assert forest.loc["F", "holds"] and not forest.loc["E", "holds"] and not forest.loc["E'", "holds"]


class _Source:
    def __init__(self, keys):
        self.files = {k: {} for k in keys}

    def get(self, *arguments, **keywords):
        return None


def test_lookup_refuses_manifests_that_share_a_key():
    with pytest.raises(am.Stop):
        am.Lookup(_Source(["a", "b"]), _Source(["c"]), _Source(["b"]))
    am.Lookup(_Source(["a"]), _Source(["b"]), _Source(["c"]))
