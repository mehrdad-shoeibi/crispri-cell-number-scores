"""The reading rules of the E4, D2 and E6 statistics scripts against the tables of their plans.

E4: the attribution of a Section 6.4 change to a step (bioinf2027/plans/E4.md, Section 11.1).
D2: the share lost with non-positive denominators (D2.md, Section 11.2, item 7) and the reading table
(Section 11.1, with 11.2, items 8 to 10). E6: the reading per arm and direction (E6.md, Section 7).
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

import make_d2_stats as d2  # noqa: E402
import make_e4_stats as e4  # noqa: E402
import make_e6_stats as e6  # noqa: E402

HP, HN, INC = "Holm-positive", "Holm-negative", "inconclusive"


# --------------------------------------------------------------------------- E4


def interval(lo: float, hi: float) -> dict:
    return {"estimate": (lo + hi) / 2, "lo": lo, "hi": hi}


POS, NEG, NONE = interval(0.01, 0.05), interval(-0.05, -0.01), interval(-0.02, 0.02)
STEP_NAMES = ("total change (matched minus existing-label)",
              "path 1, label step (labels-matched minus existing-label)",
              "path 1, feature step (matched minus labels-matched)",
              "path 2, feature step at the existing label (fourth minus existing-label)",
              "path 2, label step at matched features (matched minus fourth)",
              "interaction (path 1 feature step minus path 2 feature step)")


def steps(total, label1, feature1, feature2, label2, interaction=NONE) -> dict:
    return dict(zip(STEP_NAMES, (total, label1, feature1, feature2, label2, interaction)))


def test_e4_state_of():
    assert e4.state_of(POS) == "positive"
    assert e4.state_of(NEG) == "negative"
    assert e4.state_of(NONE) == "not resolved"
    assert e4.state_of(interval(float("nan"), float("nan"))) == "not computed"


def test_e4_step_credited_only_when_resolved_with_the_same_sign_on_both_paths():
    assert e4.attribution(steps(POS, POS, NONE, NONE, POS)) == "credited to the label step (positive)"
    assert e4.attribution(steps(NEG, NONE, NEG, NEG, NONE)) == "credited to the feature step (negative)"
    both = e4.attribution(steps(POS, POS, POS, POS, POS))
    assert both == "credited to the label step (positive) and the feature step (positive)"


def test_e4_step_resolved_on_one_path_only_is_not_attributable():
    reading = e4.attribution(steps(POS, POS, NONE, NONE, NONE))
    assert reading.startswith("not attributable (resolved on one path only: label step)")
    assert "interaction" in reading
    # opposite signs on the two paths are not credited either
    assert e4.attribution(steps(POS, NONE, POS, NEG, NONE)).startswith("not attributable")


def test_e4_no_step_resolved():
    assert e4.attribution(steps(POS, NONE, NONE, NONE, NONE)) == "resolved, not attributable to one step"
    assert e4.attribution(steps(NONE, NONE, NONE, NONE, NONE)) == "no change resolved"


# --------------------------------------------------------------------------- D2


def test_d2_share_lost_keeps_infinite_replicates_in_the_interval():
    z = (0.5, np.array([0.5, 0.4, 0.3] + [0.5] * 37))
    zm = (0.2, np.array([0.2, 0.4, 0.35] + [0.2] * 37))
    c1 = (0.1, np.array([0.1, 0.4, 0.4] + [0.1] * 37))
    out = d2.share_lost(z, zm, c1)
    assert out["estimate"] == pytest.approx(0.75)
    assert out["replicates_with_a_non_positive_denominator"] == 2
    assert out["finite_replicates"] == 38
    assert out["lo"] == -np.inf  # two of 40 replicates are -inf: at the 2.5th percentile
    assert out["hi"] == pytest.approx(0.75)


def test_d2_share_lost_point_with_a_non_positive_denominator():
    out = d2.share_lost((0.1, np.array([0.1])), (0.05, np.array([0.05])), (0.2, np.array([0.2])))
    assert out["estimate"] == np.inf


def d2_frames(vcc: dict, external: dict | None = None, share_lo=(0.8, 0.8), z_pre_lo=(0.01, 0.01), zm_pre_lo=(0.01, 0.01),
              drop_lo=(0.01, 0.01)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Family and secondary frames for G5 and G6: vcc maps a statistic to its state in both cells (or a pair)."""
    rows = []
    external = external or {}
    for i, cell in enumerate(("G5", "G6")):
        for name in ("dR2_token", "dR2_rest", "VCC noise check R2(z~) - R2(z-)"):
            s = vcc[name] if isinstance(vcc[name], str) else vcc[name][i]
            rows.append({"cell": cell, "scope": "VCC test", "statistic": name, "state": s, "estimate": 0.1, "lo": 0.05, "hi": 0.15,
                         "detectable_2.80se": 0.07, "r5_gate_passed": True})
        for screen in d2.SCREENS:
            for name in ("drho_token", "R(z-)", "external noise check rho(z~) - rho(z-)"):
                s = external.get((screen, name), INC)
                rows.append({"cell": cell, "scope": screen, "statistic": name, "state": s if isinstance(s, str) else s[i], "estimate": 0.0,
                             "lo": -0.1, "hi": 0.1, "detectable_2.80se": 0.1, "r5_gate_passed": True})
    secondary = []
    for i, cell in enumerate(("G5", "G6")):
        for quantity, lo in (("precondition of z: R2(z) - R2(C1)", z_pre_lo[i]), ("precondition of z-: R2(z-) - R2(C1)", zm_pre_lo[i]),
                             ("share lost", share_lo[i]), ("positive control of the drop: R2(z) - R2(z-) predicting B2 native", drop_lo[i])):
            secondary.append({"cell": cell, "quantity": quantity, "estimate": lo + 0.1, "lo": lo, "hi": lo + 0.2, "detectable_2.80se": 0.1})
    return pd.DataFrame(rows), pd.DataFrame(secondary)


@pytest.fixture
def informative(monkeypatch):
    monkeypatch.setitem(d2.DATA, "uninformative", {"G5": False, "G6": False})


def vcc_reading(frames) -> pd.Series:
    out = d2.readings(*frames)
    return out[out["scope"] == "VCC"].iloc[0]


def test_d2_reading_a_prime_needs_the_margin_in_both_cells(informative):
    states = {"dR2_token": HP, "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": HP}
    assert vcc_reading(d2_frames(states))["reading"] == "(a')"
    mixed = vcc_reading(d2_frames(states, share_lo=(0.8, 0.5)))
    assert (mixed["G5"], mixed["G6"], mixed["reading"]) == ("(a')", "(a)", "(a)")


def test_d2_readings_b_c_e(informative):
    assert vcc_reading(d2_frames({"dR2_token": HP, "dR2_rest": HP, "VCC noise check R2(z~) - R2(z-)": HP}))["reading"] == "(b)"
    assert vcc_reading(d2_frames({"dR2_token": HP, "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": INC}))["reading"] == "(c)"
    assert vcc_reading(d2_frames({"dR2_token": HN, "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": HN}))["reading"] == "(e)"
    e_noise = vcc_reading(d2_frames({"dR2_token": HN, "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": INC}))
    assert e_noise["statement"] == "not attributable to the token"


def test_d2_reading_d_is_qualified_by_the_positive_control_of_the_drop(informative):
    states = {"dR2_token": INC, "dR2_rest": HP, "VCC noise check R2(z~) - R2(z-)": INC}
    assert vcc_reading(d2_frames(states))["statement"] == d2.READING_TEXT["(d)"]
    qualified = vcc_reading(d2_frames(states, drop_lo=(0.01, -0.01)))
    assert qualified["reading"] == "(d)" and qualified["statement"].endswith(d2.DROP_NOT_REMOVED)


def test_d2_reading_f_and_disagreement(informative):
    states = {"dR2_token": HP, "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": HP}
    out = vcc_reading(d2_frames(states, z_pre_lo=(0.01, -0.01)))
    assert (out["G6"], out["reading"]) == ("(f)", "undetermined")
    differ = vcc_reading(d2_frames({"dR2_token": (HP, INC), "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": HP}))
    assert differ["reading"] == "undetermined"


def test_d2_uninformative_cell_makes_every_statement_undetermined(monkeypatch):
    monkeypatch.setitem(d2.DATA, "uninformative", {"G5": True, "G6": False})
    frames = d2_frames({"dR2_token": HP, "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": HP},
                       external={("RPE1", "drho_token"): HP, ("RPE1", "external noise check rho(z~) - rho(z-)"): HP})
    out = d2.readings(*frames)
    assert out[out["scope"] == "VCC"].iloc[0]["statement"] == d2.READING_TEXT["uninformative"]
    rpe1 = out[out["scope"] == "RPE1"].iloc[0]
    assert rpe1["statement"] == d2.READING_TEXT["uninformative"]
    assert rpe1["R(z-) reading"].startswith("not read")


def test_d2_external_statements_and_the_precondition_of_r_z_minus(informative):
    external = {("RPE1", "drho_token"): HP, ("RPE1", "external noise check rho(z~) - rho(z-)"): HP, ("RPE1", "R(z-)"): HP,
                ("K562-essential", "drho_token"): (HP, INC)}
    states = {"dR2_token": INC, "dR2_rest": INC, "VCC noise check R2(z~) - R2(z-)": INC}
    out = d2.readings(*d2_frames(states, external=external)).set_index("scope")
    assert out.loc["RPE1", "statement"].startswith("removing the token lowers")
    assert out.loc["RPE1", "R(z-) reading"] == "z- carries information beyond the specified trivial predictors"
    assert out.loc["K562-essential", "statement"] == "undetermined"
    failed = d2.readings(*d2_frames(states, external=external, zm_pre_lo=(0.01, -0.01))).set_index("scope")
    assert failed.loc["RPE1", "R(z-) reading"] == "not read: z- fails its own precondition in a cell of the pair"


# --------------------------------------------------------------------------- E6


def e6_frames(t: dict, r: dict, r_minus: dict | None = None, r_z_minus=INC, label_below=False, c1_b2_below=False, z_b2_below=False,
              raw_b1_estimate=0.05):
    """Family, positive-control and secondary frames for the direction K562-essential to RPE1."""
    direction = "K562-essential to RPE1"
    other = "RPE1 to K562-essential"
    rows = []
    for d in (direction, other):
        for arm in e6.FAMILY:
            rows.append({"direction": d, "statistic": "T", "arm": arm, "state": t.get(arm, INC), "not_robust_to_the_gene_fold_assignment": False, "not_robust_to_fold_assignment": False, "not_robust_to_head_fitting": False})
            rows.append({"direction": d, "statistic": "R", "arm": arm, "state": r.get(arm, INC), "not_robust_to_the_gene_fold_assignment": False, "not_robust_to_fold_assignment": False, "not_robust_to_head_fitting": False})
        for arm in e6.MASKED:
            rows.append({"direction": d, "statistic": "R-", "arm": arm, "state": (r_minus or {}).get(arm, INC),
                         "not_robust_to_the_gene_fold_assignment": False, "not_robust_to_fold_assignment": False, "not_robust_to_head_fitting": False})
        rows.append({"direction": d, "statistic": "R(z-)", "arm": "z-", "state": r_z_minus, "not_robust_to_the_gene_fold_assignment": False, "not_robust_to_fold_assignment": False, "not_robust_to_head_fitting": False})
    positive = []
    for d in (direction, other):
        positive += [{"direction": d, "endpoint": "27/788", "control": "label construction", "below_0.80": label_below},
                     {"direction": d, "endpoint": "27/788", "control": "pipeline [C1; B2]", "below_0.80": c1_b2_below},
                     {"direction": d, "endpoint": "27/788", "control": "pipeline [z; B2]", "below_0.80": z_b2_below}]
    secondary = [{"direction": direction, "contrast": f"{arm} minus the target screen's raw {b}", "estimate": raw_b1_estimate if b == "B1" else 0.05}
                 for arm in e6.FAMILY for b in ("B1", "B2")]
    return pd.DataFrame(rows), pd.DataFrame(positive), pd.DataFrame(secondary)


@pytest.fixture
def gate(monkeypatch):
    def set_gate(rpe1: bool, k562: bool = False):
        monkeypatch.setitem(e6.DATA, "r5", {"G1 | RPE1": {"passed": rpe1}, "G1 | K562-essential": {"passed": k562}})
    return set_gate


def e6_row(frames, arm: str, direction: str = "K562-essential to RPE1") -> pd.Series:
    out = e6.readings(*frames)
    return out[(out["direction"] == direction) & (out["arm"] == arm)].iloc[0]


def test_e6_z_needs_r_of_z_minus(gate):
    gate(True)
    without = e6_row(e6_frames({"z": HP}, {"z": HP}, r_z_minus=INC), "z")
    assert "R(z-) is not" in without["statement"]
    with_ = e6_row(e6_frames({"z": HP}, {"z": HP}, r_z_minus=HP), "z")
    assert with_["statement"].startswith("trained on the source screen")


def test_e6_expression_arm_qualified_by_its_masked_arm(gate):
    gate(True)
    plain = e6_row(e6_frames({"x": HP}, {"x": HP}), "x")
    assert plain["statement"].endswith("since R- of the masked arm is not Holm-positive")
    masked = e6_row(e6_frames({"x": HP}, {"x": HP}, r_minus={"x-": HP}), "x")
    assert masked["statement"].endswith("R- of the masked arm is also Holm-positive")


def test_e6_validity_rules(gate):
    gate(False)
    assert "not answerable" in e6_row(e6_frames({"m": HP}, {"m": HP}), "m")["statement"]
    gate(True)
    assert e6_row(e6_frames({"m": HP}, {"m": HP}, label_below=True), "m")["statement"].startswith("no reading")
    assert "limited by the head" in e6_row(e6_frames({"m": HN}, {}, c1_b2_below=True), "m")["statement"]
    assert e6_row(e6_frames({"m": HN}, {}), "m")["statement"] == "the arm ranks the target constructs below the Gaussian control"
    assert "z masks the signal" in e6_row(e6_frames({"z": INC}, {}, z_b2_below=True), "z")["markers"]


def test_e6_r_minus_without_transfer_and_the_baseline_marker(gate):
    gate(True)
    row = e6_row(e6_frames({"x": INC}, {}, r_minus={"x-": HP}), "x-")
    assert row["statement"].startswith("information beyond the trivial predictors without a resolved transfer")
    marked = e6_row(e6_frames({"x": HP}, {"x": INC}, raw_b1_estimate=-0.01), "x")
    assert "untrained B1" in marked["markers"]


def test_e6_fold_and_head_fitting_markers_reach_the_readings(gate):
    gate(True)
    family, positive, secondary = e6_frames({"x": "Holm-positive"}, {"x": "Holm-positive"}, {"x-": "Holm-positive"})
    d = "K562-essential to RPE1"
    family.loc[(family.direction == d) & (family.statistic == "R") & (family.arm == "x"), "not_robust_to_fold_assignment"] = True
    family.loc[(family.direction == d) & (family.statistic == "T") & (family.arm == "x"), "not_robust_to_head_fitting"] = True
    family.loc[(family.direction == d) & (family.statistic == "R-") & (family.arm == "x-"), "not_robust_to_fold_assignment"] = True
    out = e6.readings(family, positive, secondary)
    x = out[(out.direction == d) & (out.arm == "x")].iloc[0]
    assert "R not robust to fold assignment" in x["markers"] and "T not robust to head fitting" in x["markers"]
    masked = out[(out.direction == d) & (out.arm == "x-")].iloc[0]
    assert masked["markers"] == "R- not robust to fold assignment"
    m = out[(out.direction == d) & (out.arm == "m")].iloc[0]
    assert m["markers"] == ""


def test_e4_readings_carry_the_inherited_markers():
    rows = []
    for cell in e4.PRIMARY:
        for screen in e4.SCREENS:
            for arm in e4.FAMILY:
                rows.append({"cell": cell, "screen": screen, "arm": arm, "state": "Holm-positive", "read": "yes", "estimate": 0.1, "lo": 0.05,
                             "hi": 0.15, "detectable_2.80se": 0.08, "not_robust_to_the_feature_draw": False,
                             "not_robust_to_label_mode": arm == "x" and cell == "G5", "not_robust_to_head_fitting": arm == "m" and cell == "G6"})
    out = e4.reading_rows(pd.DataFrame(rows))
    x = out[(out.screen == e4.SCREENS[0]) & (out.arm == "x")].iloc[0]["statement"]
    m = out[(out.screen == e4.SCREENS[0]) & (out.arm == "m")].iloc[0]["statement"]
    z = out[(out.screen == e4.SCREENS[0]) & (out.arm == "z")].iloc[0]["statement"]
    assert x.endswith("; not robust to the label mode in G5") and m.endswith("; not robust to head fitting in G6")
    assert "not robust" not in z
