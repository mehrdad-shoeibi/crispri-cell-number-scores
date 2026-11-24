"""Tests of the fit tiers (E4 and E6 plans, Section 16): categories, selection, reuse refusal, pass equality."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fit_tiers  # noqa: E402
import make_e4_fits as e4  # noqa: E402
import make_e6_fits as e6  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REFITS = 20  # bioinf2027/results/E1/protocol.json, as priced in bioinf2027/diag/E4_E6_runtime.json


def _e6_tasks():
    e6.DATA["e4"] = True
    tasks = e6.task_list(REFITS, (0,)) + e6.shared_c2_tasks()
    for source in e6.SCREENS:
        for k in range(5):
            d = {"source": source, "endpoint": e6.PRIMARY, "fit_set": f"fold {k}", "z_minus": True}
            for seed in e6.SEEDS:
                tasks.append((d, FitSpec(e6.context_name(source, e6.PRIMARY, f"fold {k}") + " | z- of D2", "rebuilt", "z", "forest", seed)))
    return tasks


def test_categories_are_the_priced_ones():
    e4.DATA["thinned"] = True
    counts: dict = {}
    for d, s in e4.task_list(REFITS):
        key = ("E4", fit_tiers.e4_category(d, s))
        counts[key] = counts.get(key, 0) + 1
    for d, s in _e6_tasks():
        key = ("E6", fit_tiers.e6_category(d, s))
        counts[key] = counts.get(key, 0) + 1
    saved = pd.read_csv(REPO / "bioinf2027" / "diag" / "E4_E6_runtime_categories.csv")
    expected = {(r.item, r.category): int(r.fits) for r in saved.itertuples() if r.item != "D2"}
    assert counts == expected


def test_tiers_as_decided():
    e4.DATA["thinned"] = True
    assert [len(e4.run_list(REFITS, t)) for t in (1, 2, 3)] == [78, 328, 1552]
    listed = e4.task_list(REFITS)
    dropped = [t for t in listed if fit_tiers.e4_tier(*t) is None]
    assert len(dropped) == 1008 and all(fit_tiers.e4_category(*t).startswith("unread") for t in dropped)
    tiers = pd.Series([fit_tiers.e6_tier(d, s) for d, s in _e6_tasks()]).value_counts().to_dict()
    assert tiers == {1: 340, 2: 8750, 3: 4418}


def test_parse_tier():
    assert fit_tiers.parse_tier(["--smoke", "--tier", "2"]) == (2, ["--smoke"])
    assert fit_tiers.parse_tier(["--plan"]) == (None, ["--plan"])
    with pytest.raises(SystemExit):
        fit_tiers.parse_tier(["--tier", "4"])
    with pytest.raises(SystemExit):
        fit_tiers.parse_tier(["--tier"])


def test_differences():
    earlier = {"inputs": {"features": {"sha256": "a"}, "d2": None}, "heads": {"forest": {"n": 500}}, "seeds": [0, 1, 2]}
    assert fit_tiers.differences(earlier, json.loads(json.dumps(earlier))) == []
    later = json.loads(json.dumps(earlier))
    later["inputs"]["d2"] = {"sha256": "b"}  # absent before: no earlier fit used it
    assert fit_tiers.differences(earlier, later) == []
    later["inputs"]["features"]["sha256"] = "c"
    later["heads"]["forest"]["n"] = 5
    assert sorted(fit_tiers.differences(earlier, later)) == ["heads.forest.n", "inputs.features.sha256"]
    assert fit_tiers.differences(earlier, {"heads": earlier["heads"], "seeds": [0, 1, 2]}) == ["inputs"]


def test_a_later_tier_refuses_other_inputs(tmp_path):
    protocol = {"inputs": {"features": {"sha256": "a"}}, "heads": {"forest": {"n_estimators": 500}}, "seeds": {"heads": [0, 1, 2]},
                "refits": 20, "fitting_code": {"bioinf2027/src/bioinf2027/transfer.py": "x"}, "smoke": False, "tier": 1}
    (tmp_path / "protocol_tier1.json").write_text(json.dumps(protocol))
    same = dict(protocol, tier=2)
    assert e4.refuse_if_inputs_differ(tmp_path, same) and e6.refuse_if_inputs_differ(tmp_path, same)
    for change in ({"inputs": {"features": {"sha256": "b"}}}, {"refits": 50}, {"fitting_code": {"bioinf2027/src/bioinf2027/transfer.py": "y"}},
                   {"heads": {"forest": {"n_estimators": 5}}}):
        with pytest.raises(SystemExit):
            e4.refuse_if_inputs_differ(tmp_path, {**same, **change})
        with pytest.raises(SystemExit):
            e6.refuse_if_inputs_differ(tmp_path, {**same, **change})


def test_compare_passes(tmp_path):
    family, full = tmp_path / "family", tmp_path / "full"
    family.mkdir()
    full.mkdir()
    a = pd.DataFrame({"cell": ["G5", "G6"], "arm": ["z", "x"], "estimate": [0.1, np.nan], "state": ["inconclusive", "Holm-positive"],
                      "precondition_matched_passed": [None, None]})
    b = pd.DataFrame({"cell": ["G5", "G6", "G1"], "arm": ["z", "x", "z"], "estimate": [0.1, np.nan, 0.3],
                      "state": ["inconclusive", "Holm-positive", "x"], "precondition_matched_passed": [True, False, True]})
    a.to_csv(family / "t.csv", index=False)
    b.to_csv(full / "t.csv", index=False)
    a["note"] = ["", "text"]  # an empty text cell is read back as missing in both tables
    b["note"] = ["", "text", ""]
    a.to_csv(family / "t.csv", index=False)
    b.to_csv(full / "t.csv", index=False)
    pairs = {"t.csv": ("t.csv", ["cell", "arm"], ("precondition_matched_passed",))}
    assert fit_tiers.compare_passes(family, full, pairs)["passed"]
    b.loc[1, "note"] = "other"
    b.to_csv(full / "t.csv", index=False)
    assert fit_tiers.compare_passes(family, full, pairs)["files"]["t.csv"]["differing"] == [{"column": "note", "rows": 1}]
    b.loc[1, "note"] = "text"
    b.to_csv(full / "t.csv", index=False)
    b.loc[0, "estimate"] = 0.1 + 1e-15
    b.to_csv(full / "t.csv", index=False)
    result = fit_tiers.compare_passes(family, full, pairs)
    assert not result["passed"] and result["files"]["t.csv"]["differing"] == [{"column": "estimate", "rows": 1}]
    b.loc[0, "estimate"] = 0.1
    b.iloc[1:].to_csv(full / "t.csv", index=False)
    assert fit_tiers.compare_passes(family, full, pairs)["files"]["t.csv"]["rows_missing_from_the_full_pass"] == 1
    b.to_csv(full / "t.csv", index=False)
    assert not fit_tiers.compare_passes(family, full, {"t.csv": ("t.csv", ["cell", "arm"], ())})["passed"]


def test_e4_smoke_subset_is_g5_draw_1_seed_0():
    e4.DATA["thinned"] = True
    subset = [t for t in e4.run_list(1, 3) if e4.smoke_subset(t)]
    assert subset and all(s.seed == 0 for _, s in subset)
    assert {d["cell"] for d, _ in subset} <= {"G5", "G5-DT-RPE1", "G5-DT-K562-essential"}
    assert all("draw 1" in d["name"] or "shared" in d["name"] and d["name"].endswith(" 1") or d["kind"] in ("disjoint", "corner")
               for d, _ in subset)
    assert {fit_tiers.e4_tier(d, s) for d, s in subset} == {1, 2, 3}


def test_parse_workers():
    assert fit_tiers.parse_workers(["--tier", "1"], 12) == (12, ["--tier", "1"])
    assert fit_tiers.parse_workers(["--workers", "3", "--smoke"], 12) == (3, ["--smoke"])
    for bad in (["--workers", "0"], ["--workers", "x"], ["--workers"]):
        with pytest.raises(SystemExit):
            fit_tiers.parse_workers(bad, 12)
