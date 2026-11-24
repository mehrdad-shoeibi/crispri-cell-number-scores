"""The two rules fixed before E6's full pass (bioinf2027/reports/E6.md, departures 7 and 8).

- Departure 7: a state under a gene-fold assignment is its p-value set against a given Holm threshold.
- Departure 8: the one-sided p-value of E1_design.md, Section 23.2, p = (1 + #{theta* >= margin}) / (B + 1),
  computed where T is Holm-positive, R is not and the R5 gate passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e6_stats as e6  # noqa: E402


def test_one_sided_matches_the_e1b_formula():
    draws = np.array([0.05, 0.12, 0.10, np.nan, 0.30])
    assert e6.one_sided((0.1, draws), 0.10) == (1 + 3) / (5 + 1)
    assert e6.one_sided(None, 0.10) != e6.one_sided(None, 0.10)  # NaN when the statistic is missing


def test_assignment_state_uses_the_given_threshold():
    positive = (0.2, np.full(999, 0.2))
    assert e6.assignment_state(positive, 0.05) == "Holm-positive"
    assert e6.assignment_state(positive, 0.0) == "inconclusive"
    assert e6.assignment_state(None, 0.05) == "not fitted"


def test_one_sided_rows_only_where_the_conditions_hold(monkeypatch):
    rows = []
    for direction, gate in (("RPE1 to K562-essential", False), ("K562-essential to RPE1", True)):
        for arm in e6.FAMILY:
            rows.append({"direction": direction, "statistic": "T", "arm": arm, "state": "Holm-positive", "r5_gate_passed": gate})
            rows.append({"direction": direction, "statistic": "R", "arm": arm,
                         "state": "Holm-positive" if arm == "z" else "inconclusive", "r5_gate_passed": gate})
    family = pd.DataFrame(rows)
    value = (0.05, np.full(99, 0.05))
    monkeypatch.setattr(e6, "arm_r", lambda *a, **k: value)
    monkeypatch.setattr(e6, "arm_rho", lambda *a, **k: value)
    out = e6.one_sided_rows(family).set_index(["direction", "arm"])
    assert not out.loc[("RPE1 to K562-essential", "x"), "considered"]
    assert out.loc[("RPE1 to K562-essential", "x"), "why_not_considered"] == "the R5 gate fails"
    assert not out.loc[("K562-essential to RPE1", "z"), "considered"]
    assert out.loc[("K562-essential to RPE1", "z"), "why_not_considered"] == "R Holm-positive"
    assert out.loc[("K562-essential to RPE1", "x"), "considered"]
    assert out.loc[("K562-essential to RPE1", "x"), "p_R_at_least_0.10"] == 1 / 100
