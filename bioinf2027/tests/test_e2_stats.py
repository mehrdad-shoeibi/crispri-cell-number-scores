"""Logic of E2's statistics (bioinf2027/scripts/make_e2_stats.py) on small synthetic data; no data-root input is read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e2_stats as ms  # noqa: E402
from bioinf2027 import e1cells, resample  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

e1s = ms.e1s
B = 40
SCREENS = ms.SCREENS


class Source:
    """A manifest's lookup over predictions held in memory, keyed as make_e1b_stats.Predictions keys them."""

    def __init__(self, values: dict) -> None:
        self.values = values
        self.files = {k: {} for k in values}

    def get(self, context, label, representation, head="forest", seed=0, variant=e1s.ALL_ROWS, realisation=0):
        return self.values.get(FitSpec(context, label, representation, head, int(seed), variant, int(realisation)).key())


# --------------------------------------------------------------------------- synthetic cells, predictions and draws


def make_cell(name: str, seed: int) -> e1cells.CellData:
    rng = np.random.default_rng(seed)
    genes = np.repeat([f"G{g}" for g in range(10)], 3)
    vcc = pd.DataFrame({"split": "test", "batch": [f"b{i % 4}" for i in range(genes.size)], "target_gene": genes})
    vcc["y"] = rng.normal(size=genes.size)
    vcc["existing_label"] = vcc["y"] + rng.normal(scale=0.7, size=genes.size)
    data = e1cells.CellData(name=name, spec=e1cells.CELLS[name], vcc=vcc, x=np.zeros((genes.size, 1), np.float32),
                            z=np.zeros((genes.size, 1), np.float32), target_column=np.zeros(genes.size, np.int64))
    for screen in SCREENS:
        data.external[screen] = pd.DataFrame({"construct": [f"c{i:02d}" for i in range(20)], "target_ensg": np.repeat([f"E{g}" for g in range(10)], 2),
                                              "y": rng.normal(size=20)})
        data.external_source[screen] = name
    return data


E1B_NAMES = ("z", "C1", "C3", "B0", "B1", "B2", "TR", "[B0; z]", "[B0; C1]", "[B0; C3]", "[z; B2]", "[C1; B2]", "x", "C2")
NOISE = {"z": 0.6, "scgpt": 0.5, "C1": 2.0, "C1[scgpt]": 2.0, "B0": 1.0}


def make_predictions(cells: dict, encoders=("scgpt",)) -> tuple[dict, dict]:
    rng = np.random.default_rng(7)
    e1b, e2 = {}, {}
    e2_names = [n for e in encoders for n in ms.roles(e).values() if n not in E1B_NAMES] + ["N0[geneformer]"]
    for cell, data in cells.items():
        for label, column in (("existing", "existing_label"), ("rebuilt", "y")):
            y = data.vcc[column].to_numpy()
            for head, seeds in ms.HEADS.items():
                for seed in seeds:
                    for name in E1B_NAMES + tuple(e2_names):
                        scale = NOISE.get(name, 1.2)
                        values = {"test": y + rng.normal(scale=scale, size=y.size)}
                        for screen in SCREENS:
                            ext = data.external[screen]["y"].to_numpy()
                            values[screen] = ext + rng.normal(scale=scale, size=ext.size)
                        if name in ms.e2_names():
                            encoder = ms.e2_names()[name]
                            e2[FitSpec(f"E2 | {encoder} | {cell}", label, name, head, seed).key()] = values
                        else:
                            e1b[FitSpec(cell, label, name, head, seed).key()] = values
    return e1b, e2


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(ms, "SMOKE", False)
    monkeypatch.setattr(ms, "DATA", {"missing": [], "order_marker": ""})
    monkeypatch.setattr(ms, "CACHE", {})
    monkeypatch.setattr(e1s, "B", B)
    monkeypatch.setattr(resample, "BLOCK", B)
    monkeypatch.setattr(e1s, "parallel", lambda function, arguments: [function(*a) for a in arguments])
    cells = {"G5": make_cell("G5", 1), "G6": make_cell("G6", 2)}
    external = {(s, c): cells[c].external[s] for c in cells for s in SCREENS}
    monkeypatch.setattr(e1s, "DATA", {"cells": cells, "external": external, "external_source": {c: c for c in cells}})
    draws = {}
    for i, screen in enumerate(SCREENS):
        groups = cells["G5"].external[screen]["target_ensg"].astype(str).to_numpy()
        draws[screen] = {"draw": resample.cluster_draw(groups, B, 11 + i), "point": e1s._ones(groups.size), "groups": groups}
    monkeypatch.setattr(e1s, "STATE", {"draws": draws, "vcc_draws": {}})
    e1b, e2 = make_predictions(cells)
    sources = {"e1b": Source(e1b), "e2": Source(e2)}
    lookup = ms.Lookup([sources["e1b"]], [sources["e2"]], ms.e2_names())
    ms.DATA["lookup"] = lookup
    e1s.DATA["predictions"] = lookup
    return {"cells": cells, "sources": sources}


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# --------------------------------------------------------------------------- names and the lookup


def test_roles_follow_the_fitting_scripts():
    assert ms.roles("scgpt")["C1"] == "C1[scgpt]" and ms.roles("scgpt")["[e; B2]"] == "[scgpt; B2]"
    assert ms.roles("pca50")["C2-50"] == "C2-50[pca50]" and ms.roles("pca50")["N0"] == "N0[pca50]"
    assert ms.roles("pca768") == {"e": "pca768", "[B0; e]": "[B0; pca768]", "C1": "C1", "[B0; C1]": "[B0; C1]"}
    assert ms.roles(ms.GENEFORMER)["e"] == "z" and ms.roles(ms.GENEFORMER)["N0"] == "N0[geneformer]"
    names = ms.e2_names()
    assert names["C1[stack]"] == "stack" and names["N0[geneformer]"] == ms.GENEFORMER
    assert "C1" not in names and "[B0; C1]" not in names and "z" not in names
    assert not ms.available("scgpt", "[e; B2]", "G5") and ms.available("scgpt", "[e; B2]", "G6")


def test_lookup_routes_e2_names_to_e2_contexts_and_refuses_shared_keys():
    e2_key = FitSpec("E2 | scgpt | G6", "existing", "C1[scgpt]", "forest", 0).key()
    e1_key = FitSpec("G6", "existing", "C1", "forest", 0).key()
    lookup = ms.Lookup([Source({e1_key: {"test": np.zeros(2)}})], [Source({e2_key: {"test": np.ones(2)}})], ms.e2_names())
    assert lookup.get("G6", "existing", "C1[scgpt]", "forest", 0)["test"][0] == 1.0
    assert lookup.get("G6", "existing", "C1", "forest", 0)["test"][0] == 0.0
    assert lookup.get("G5", "existing", "C1[scgpt]", "forest", 0) is None
    with pytest.raises(ms.Stop):
        ms.Lookup([Source({e1_key: {}})], [Source({e1_key: {}})], ms.e2_names())


def test_planned_fits_equal_the_fitting_scripts_tasks():
    first, third = ms.planned_fits(False)
    assert (len(first), len(third)) == (784, 390)  # 756 + the 28 N0 fits of PCA-50 (E2.md, Section 17)
    smoke_first, smoke_third = ms.planned_fits(True)
    assert (len(smoke_first), len(smoke_third)) == (336, 130)  # 324 + 12
    assert ms.e2f.SMOKE is False and ms.e2e6.SMOKE is False
    assert all(t[3].seed == 0 for t in smoke_first.values())


def test_manifest_discovery_keeps_each_mode_apart(tmp_path):
    for name in ("predictions_manifest_scgpt.json", "predictions_manifest_stack_pca50_pca768_gfnull.json", "predictions_manifest_e6_scgpt.json",
                 "predictions_manifest_e6_scgpt_smoke.json", "protocol_scgpt.json"):
        (tmp_path / name).write_text("{}")
    real = ms.manifest_paths(tmp_path, tmp_path, False)
    assert [p.name for p in real["i_ii"]] == ["predictions_manifest_scgpt.json", "predictions_manifest_stack_pca50_pca768_gfnull.json"]
    assert [p.name for p in real["iii"]] == ["predictions_manifest_e6_scgpt.json"]
    assert [p.name for p in ms.manifest_paths(tmp_path, tmp_path, True)["iii"]] == ["predictions_manifest_e6_scgpt_smoke.json"]


def test_a_manifest_with_a_failed_fit_or_a_changed_protocol_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "SMOKE", False)
    monkeypatch.setattr(ms, "file_record", lambda path: {"path": str(path)})
    protocol = tmp_path / "protocol_x.json"
    protocol.write_text(json.dumps({"smoke": False, "checks": {"E0": [{"passed": True}]}}))
    good = {"protocol": {"path": str(protocol), "sha256": ms.sha256(protocol)}, "fits": 1, "completed": 1, "failed": [], "files": {"k": {}}}
    path = tmp_path / "predictions_manifest_x.json"
    path.write_text(json.dumps(good))
    assert ms.check_manifest(path, "i_ii")["fits"] == 1
    for change in ({"failed": [{"fit": "k"}]}, {"completed": 0}, {"protocol": {"path": str(protocol), "sha256": "0" * 64}}):
        path.write_text(json.dumps({**good, **change}))
        with pytest.raises(ms.Stop):
            ms.check_manifest(path, "i_ii")
    path.write_text(json.dumps(good))
    with pytest.raises(ms.Stop):
        ms.check_manifest(path, "iii")  # a (iii) protocol must hold check E6-0


def test_readings_of_the_attribution_note_are_written_for_the_encoder():
    text = ms.named(ms.att.READINGS["d2"][0], "stack")
    assert "1,600 columns" in text and "768" not in text
    text = ms.named(ms.att.READINGS["d3'"][0], "scgpt")
    assert "scgpt gives this forest" in text and " z " not in text
    assert ms.named("the size of z's margin", "pca50") == "the size of pca50's margin"


# --------------------------------------------------------------------------- VCC


def test_vcc_quantities_are_the_contrasts_inside_each_replicate(synthetic):
    frame = ms.vcc_frame(("scgpt", ms.GENEFORMER))
    cell = synthetic["cells"]["G6"]
    y = cell.vcc["existing_label"].to_numpy()
    e2, e1b = synthetic["sources"]["e2"].values, synthetic["sources"]["e1b"].values

    def mean_r2(values, context, label, name):
        return np.mean([r2(y if label == "existing" else cell.vcc["y"].to_numpy(), values[FitSpec(context, label, name, "forest", s).key()]["test"])
                        for s in (0, 1, 2)])

    row = ms.pick(frame, "scgpt", "G6", "existing", "e - C1")
    direct = mean_r2(e2, "E2 | scgpt | G6", "existing", "scgpt") - mean_r2(e2, "E2 | scgpt | G6", "existing", "C1[scgpt]")
    assert abs(row.r2 - direct) < 1e-12 and row.fact == "P-a"
    assert np.isfinite(row.batch_resampled_lo) and np.isfinite(row.batch_resampled_hi)
    rebuilt = ms.pick(frame, "scgpt", "G6", "rebuilt", "e - C1")
    assert rebuilt.fact == "P-r"
    drop = ms.pick(frame, "scgpt", "G6", ms.PAIRED, "P-d")
    assert abs(drop.r2 - (row.r2 - rebuilt.r2)) < 1e-12 and drop.fact == "P-d"
    d3, delta, d_half = (ms.pick(frame, "scgpt", "G6", "existing", q) for q in ("d3", "Delta", "D1/2"))
    assert abs(delta.r2 - (row.r2 - d3.r2)) < 1e-12
    assert abs(d_half.r2 - (d3.r2 - row.r2 / 2)) < 1e-12
    # the replicates are paired: P-d's interval is that of the difference of the two margins inside each replicate
    a = e1s.vcc_r2("G6", "existing", "scgpt")["r2"][1] - e1s.vcc_r2("G6", "existing", "C1[scgpt]")["r2"][1]
    b = e1s.vcc_r2("G6", "rebuilt", "scgpt")["r2"][1] - e1s.vcc_r2("G6", "rebuilt", "C1[scgpt]")["r2"][1]
    lo, hi, _ = e1s.resample_interval(a - b)
    assert abs(drop.r2_lo - lo) < 1e-12 and abs(drop.r2_hi - hi) < 1e-12
    # Geneformer's rows are E1b's z and C1, the reference of check 7
    z = ms.pick(frame, ms.GENEFORMER, "G6", "existing", "e - C1")
    assert z.first == "z" and z.second == "C1"
    assert abs(z.r2 - (mean_r2(e1b, "G6", "existing", "z") - mean_r2(e1b, "G6", "existing", "C1"))) < 1e-12
    n0 = ms.pick(frame, ms.GENEFORMER, "G6", "existing", "N0 - C1")
    assert n0.first == "N0[geneformer]"


def test_readings_are_made_in_g6_on_the_existing_label_under_the_forest_only(synthetic):
    frame = ms.vcc_frame(("scgpt",))
    read = frame[frame.reading != ""]
    assert set(read.cell) == {"G6"} and set(read["head"]) == {"forest"}
    d1 = ms.pick(frame, "scgpt", "G6", "existing", "d1")
    assert d1.reading and "scgpt" in d1.reading
    assert ms.pick(frame, "scgpt", "G6", "rebuilt", "d1").reading == ""
    assert ms.pick(frame, "scgpt", "G5", "existing", "d1").note.startswith("not the submitted statistic")
    holm = frame[frame.descriptive_holm_g6 != ""]
    assert set(holm.quantity) == set(ms.ATTRIBUTION_READ) and set(holm.label) == {"existing"}
    assert "positive control" in ms.pick(frame, "scgpt", "G6", "rebuilt", "[e; B2] - [C1; B2]").role
    assert ms.pick(frame, "scgpt", "G5", "existing", "[e; B2] - [C1; B2]") is None


def test_the_mse_version_of_p_d_is_paired_inside_each_replicate(synthetic):
    frame = ms.label_scale_frame(("scgpt",)).set_index(["encoder", "cell", "quantity"])
    cell = synthetic["cells"]["G6"]
    e2 = synthetic["sources"]["e2"].values
    out = {}
    for label, column in (("existing", "existing_label"), ("rebuilt", "y")):
        y = cell.vcc[column].to_numpy()

        def mse(name):
            return np.mean([np.mean((y - e2[FitSpec("E2 | scgpt | G6", label, name, "forest", s).key()]["test"]) ** 2) for s in (0, 1, 2)])

        out[label] = mse("scgpt") - mse("C1[scgpt]")
        row = frame.loc[("scgpt", "G6", f"MSE of e minus MSE of C1 (forest), {label} label")]
        assert abs(row.estimate - out[label]) < 1e-12
        variance = frame.loc[("both labels", "G6", f"variance of the {label} label over the test rows")]
        assert abs(variance.estimate - np.var(y)) < 1e-12
    paired = frame.loc[("scgpt", "G6", "MSE of e minus MSE of C1, existing label minus rebuilt label, one joint draw")]
    assert abs(paired.estimate - (out["existing"] - out["rebuilt"])) < 1e-12 and paired.lo <= paired.estimate <= paired.hi


def test_e6s_t_is_the_spearman_contrast_of_the_assembled_folds(monkeypatch):
    rng = np.random.default_rng(3)
    genes = {s: np.repeat([f"E{g}" for g in range(10)], 2) for s in SCREENS}
    tables = {s: pd.DataFrame({"y": rng.normal(size=20), "B1": rng.normal(size=20), "B2": rng.normal(size=20)}) for s in SCREENS}
    union = sorted(set(genes[SCREENS[0]].tolist()))
    folds = ms.e6f.fold_of(union, ms.e6f.FOLD_SEED)
    draws = {s: {"draw": resample.cluster_draw(genes[s], B, 5), "point": e1s._ones(20)} for s in SCREENS}
    monkeypatch.setattr(ms.e6s, "DATA", {"genes": genes, "folds": {ms.e6f.FOLD_SEED: folds}, "tables": {ms.e6f.PRIMARY: tables},
                                         "n": {s: 20 for s in SCREENS}, "draws": draws})
    monkeypatch.setattr(ms, "SMOKE", False)
    monkeypatch.setattr(ms, "CACHE", {})
    monkeypatch.setattr(resample, "BLOCK", B)
    specs, values, full = {}, {}, {}
    for source in ms.SOURCES:
        target = ms.e6f.other(source)
        fold = np.array([folds[g] for g in genes[target]])
        for role in ms.E6_ROLES:
            name = ms.roles("scgpt")[role]
            for seed in (0, 1, 2):
                vector = tables[target]["y"].to_numpy() + rng.normal(scale=0.5 if role == "e" else 2.0, size=20)
                full[(source, role, seed)] = vector
                for k in range(5):
                    spec = FitSpec(f"E2 | E6 | scgpt | {source}>{target} | fold {k}", "rebuilt", name, "forest", seed)
                    specs[("scgpt", source, k, role, seed)] = spec
                    values[spec.key()] = {target: vector[fold == k]}
    monkeypatch.setattr(ms, "DATA", {"e6_specs": specs, "e6_sources": [Source(values)], "missing": [], "order_marker": ""})
    monkeypatch.setattr(ms.e6s, "assembled", lambda source, endpoint, model, head, seed, **kw: tables[ms.e6f.other(source)]["y"].to_numpy() * 0.5)
    frame = ms.e6_frame(("scgpt",))
    for source in ms.SOURCES:
        target = ms.e6f.other(source)
        y = tables[target]["y"].to_numpy()

        def rho(role):
            return np.mean([sps.spearmanr(full[(source, role, s)], y).statistic for s in (0, 1, 2)])

        row = ms.stat(frame, encoder="scgpt", direction=f"{source} to {target}", quantity="T")
        assert abs(row["estimate"] - (rho("e") - rho("C1"))) < 1e-12
        raw = ms.stat(frame, encoder="scgpt", direction=f"{source} to {target}", quantity="e minus the target screen's raw B1")
        assert abs(raw["estimate"] - (rho("e") - sps.spearmanr(tables[target]["B1"], y).statistic)) < 1e-12
    assert set(frame[frame.quantity == "T"].role) == {"family"}


def test_a_missing_head_seed_stops_the_real_run_and_is_listed_in_the_smoke_run(synthetic, monkeypatch):
    values = synthetic["sources"]["e2"].values
    del values[FitSpec("E2 | scgpt | G6", "existing", "C3[scgpt]", "forest", 2).key()]
    with pytest.raises(ms.Stop):
        ms.score("G6", "existing", "C3[scgpt]", "forest")
    monkeypatch.setattr(ms, "SMOKE", True)
    ms.CACHE.clear()
    assert ms.wanted_seeds("C3[scgpt]", "forest") == (0,) and ms.wanted_seeds("C3", "forest") == (0, 1, 2)
    del values[FitSpec("E2 | scgpt | G6", "existing", "C3[scgpt]", "forest", 0).key()]
    assert ms.score("G6", "existing", "C3[scgpt]", "forest") is None and ms.DATA["missing"]


def test_pattern_outcomes_follow_the_table_of_section_2():
    assert ms.pattern_outcome(0.01, 0.02, -0.1) == "pattern"
    assert ms.pattern_outcome(0.01, 0.02, 0.03) == "margin on the size-fixed label"
    assert ms.pattern_outcome(0.01, -0.02, 0.03) == "margin on the size-fixed label"
    assert ms.pattern_outcome(0.01, -0.02, -0.1) == "no change shown"
    assert ms.pattern_outcome(-0.01, 0.2, 0.3) == "no margin"
    assert ms.pattern_outcome(np.nan, 0.2, 0.3) == "undefined"


def test_pattern_and_attribution_rows_from_the_vcc_table(synthetic):
    frame = ms.vcc_frame((ms.GENEFORMER, "scgpt"))
    pattern = ms.pattern_frame(frame)
    row = pattern[pattern.encoder == "scgpt"].iloc[0]
    assert row.outcome == ms.pattern_outcome(row["P-a_lo"], row["P-d_lo"], row["P-r_lo"])
    assert row["P-a_descriptive_holm"].endswith("(descriptive, not a decision rule)")
    assert pattern[pattern.encoder == "stack"].iloc[0].outcome == "undefined"
    assert pattern.iloc[-1].encoder == "all read encoders"
    s2 = ms.attribution_s2_frame(frame)
    scgpt = s2[(s2.encoder == "scgpt") & (s2["head"] == "forest")].set_index("row")
    assert scgpt.loc["D", "holds"] == (not (scgpt.loc["A", "holds"] or scgpt.loc["B", "holds"] or scgpt.loc["C", "holds"]))
    assert {"B'", "C'"} <= set(scgpt.index)


def test_s2_rows_follow_the_states():
    class R:
        def __init__(self, state, hi=0.1):
            self.state, self.r2_hi = state, hi

    holds = ms.s2_holds({"d3": R("above zero"), "d3'": R("contains zero"), "Delta": R("above zero"), "D1/2": R("below zero", -0.01),
                         "Delta'": R("contains zero"), "D1/2'": R("contains zero", 0.2)})
    assert holds["A'"] and not holds["A"] and holds["B"] and holds["C"] and not holds["D"] and not holds["B'"] and not holds["C'"]
    assert ms.s2_holds({})["D"]


# --------------------------------------------------------------------------- external (ii)


def test_external_t_and_l_are_e1bs_spearman_contrasts(synthetic):
    scorer = e1s.Scorer()
    e1s.STATE["scorer"] = scorer
    ms.request_external(scorer, ("scgpt", ms.GENEFORMER))
    scorer.run()
    frame = ms.external_frame(("scgpt",))
    e2 = synthetic["sources"]["e2"].values
    y = synthetic["cells"]["G6"].external["RPE1"]["y"].to_numpy()

    def rho(name, label):
        return np.mean([sps.spearmanr(e2[FitSpec("E2 | scgpt | G6", label, name, "forest", s).key()]["RPE1"], y).statistic for s in (0, 1, 2)])

    t = {}
    for label in ms.LABELS:
        row = ms.stat(frame, encoder="scgpt", cell="G6", screen="RPE1", label=label, head="forest", quantity="T")
        t[label] = row["estimate"]
        assert abs(t[label] - (rho("scgpt", label) - rho("C1[scgpt]", label))) < 1e-12
    l_row = ms.stat(frame, encoder="scgpt", cell="G6", screen="RPE1", label=ms.L_LABEL, head="forest", quantity="L")
    assert abs(l_row["estimate"] - (t["rebuilt"] - t["existing"])) < 1e-12
    roles = set(frame[(frame.quantity.isin(["T", "L"])) & (frame["head"] == "forest") & (frame.label != "existing")].role)
    assert roles == {"family"}
    assert len(frame[(frame.quantity == "e - z") & (frame["head"] == "forest")]) == 8


def _family_inputs(p_t: dict, estimates: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ext, e6 = [], []
    for encoder in ms.READ_ENCODERS:
        for cell in ms.CELLS:
            for screen in SCREENS:
                for quantity, label in (("T", "rebuilt"), ("L", ms.L_LABEL)):
                    p = p_t[encoder] if quantity == "T" else 0.9
                    ext.append({"encoder": encoder, "cell": cell, "screen": screen, "label": label, "head": "forest", "quantity": quantity,
                                "estimate": estimates[encoder], "lo": 0.0, "hi": 0.0, "p": p, "se": 0.01, "detectable_2.80se": 0.028})
            ext.append({"encoder": encoder, "cell": "G6", "screen": "RPE1", "label": "rebuilt", "head": "forest",
                        "quantity": f"rho({ms.roles(encoder)['[C1; B2]']})", "estimate": 0.9, "lo": 0.0, "hi": 0.0, "p": 0.0, "se": 0.0,
                        "detectable_2.80se": 0.0})
        for source in ms.SOURCES:
            direction = f"{source} to {ms.e6f.other(source)}"
            for quantity, estimate in (("T", 0.1), (f"rho({ms.roles(encoder)['[C1; B2]']})", 0.7), (f"rho({ms.roles(encoder)['[e; B2]']})", 0.9),
                                       ("e minus the target screen's raw B1", -0.2), ("e minus the target screen's raw B2", 0.1)):
                e6.append({"encoder": encoder, "direction": direction, "quantity": quantity, "estimate": estimate, "lo": 0.0, "hi": 0.0,
                           "p": 0.0002 if quantity == "T" else 0.5, "se": 0.01, "detectable_2.80se": 0.028})
    return pd.DataFrame(ext), pd.DataFrame(e6)


def _valid() -> dict:
    return {"excluded": {(c, s): False for c in ms.CELLS for s in SCREENS}, "head_limited": {(c, s): False for c in ms.CELLS for s in SCREENS},
            "e6_label_ok": {f"{s} to {ms.e6f.other(s)}": True for s in ms.SOURCES}}


def test_the_family_holds_30_statistics_adjusted_together(monkeypatch):
    monkeypatch.setattr(ms, "DATA", {"order_marker": ""})
    external, e6 = _family_inputs({"scgpt": 0.0002, "stack": 0.0002, "pca50": 0.5}, {"scgpt": 0.2, "stack": -0.2, "pca50": 0.01})
    pre = {"scgpt": {"G5": True, "G6": True, "both cells": True}, "stack": {"G5": True, "G6": False, "both cells": False},
           "pca50": {"G5": True, "G6": True, "both cells": True}}
    family = ms.family_frame(external, e6, pre, _valid())
    assert len(family) == 30 and (family.test == "ii").sum() == 24 and (family.test == "iii").sum() == 6
    rejected, thresholds = resample.holm(family["p"].to_numpy(), 0.05)
    assert np.array_equal(family["rejected"].to_numpy(), rejected) and np.allclose(family["holm_threshold"], thresholds)
    assert set(family[(family.test == "ii") & (family.encoder == "stack")].read) == {False}
    assert set(family[(family.test == "iii") & (family.encoder == "stack")].read) == {True}
    assert all(m.startswith("Stack") for m in family[family.encoder == "stack"].markers)
    scgpt_t = family[(family.encoder == "scgpt") & (family.test == "ii") & (family.statistic == "T")]
    assert set(scgpt_t.state) == {"Holm-positive"}
    readings = ms.external_readings(family, external, pre, _valid()).set_index(["encoder", "screen"])
    assert readings.loc[("scgpt", "RPE1"), "statement"].startswith("transfer resolved; R is not computed")
    assert readings.loc[("stack", "RPE1"), "statement"].startswith("uninformative about transfer")
    assert readings.loc[("stack", "RPE1"), "L_reading"] == "not read"
    assert readings.loc[("pca50", "RPE1"), "statement"] == "neither a transfer nor a failure is resolved"
    assert readings.loc[("scgpt", "RPE1"), "own_width_[C1; B2]_G6_rebuilt"] == 0.9
    e6_read = ms.e6_readings(family, e6, _valid()).set_index(["encoder", "direction"])
    row = e6_read.loc[("scgpt", "RPE1 to K562-essential")]
    assert row["T"] == "Holm-positive" and row.statement.startswith("trained on one Replogle screen, scgpt ranks")
    assert row.head_limited and "untrained B1" in row.qualifiers and "untrained B2" not in row.qualifiers


def test_a_family_with_a_missing_statistic_is_refused(monkeypatch):
    monkeypatch.setattr(ms, "DATA", {"order_marker": ""})
    external, e6 = _family_inputs({e: 0.5 for e in ms.READ_ENCODERS}, {e: 0.0 for e in ms.READ_ENCODERS})
    with pytest.raises(ms.Stop):
        ms.family_frame(external.iloc[1:], e6, {e: {"both cells": True} for e in ms.READ_ENCODERS}, _valid())


# --------------------------------------------------------------------------- E6 (iii) and check 7


def test_assembled_predictions_take_each_constructs_fold(monkeypatch):
    genes = np.array([f"E{g}" for g in range(10)])
    folds = {g: i % 5 for i, g in enumerate(genes)}
    monkeypatch.setattr(ms.e6s, "DATA", {"genes": {"K562-essential": genes}, "folds": {ms.e6f.FOLD_SEED: folds}})
    specs, values = {}, {}
    for k in range(5):
        spec = FitSpec(f"E2 | E6 | scgpt | RPE1>K562-essential | fold {k}", "rebuilt", "scgpt", "forest", 0)
        specs[("scgpt", "RPE1", k, "e", 0)] = spec
        rows = np.flatnonzero(np.array([folds[g] for g in genes]) == k)
        values[spec.key()] = {"K562-essential": rows.astype(float)}
    monkeypatch.setattr(ms, "DATA", {"e6_specs": specs, "e6_sources": [Source(values)]})
    out = ms.assembled_e2("scgpt", "RPE1", "e", 0)
    assert np.array_equal(out, np.arange(10, dtype=float))
    del values[specs[("scgpt", "RPE1", 3, "e", 0)].key()]
    assert ms.assembled_e2("scgpt", "RPE1", "e", 0) is None


def test_check_7_compares_to_1e_12_and_treats_two_missing_values_as_equal():
    checks: list = []
    ms.compare(checks, "a", 0.1 + 1e-13, 0.1)
    ms.compare(checks, "b", np.nan, np.nan)
    ms.compare(checks, "c", 0.1 + 1e-11, 0.1)
    ms.compare(checks, "d", np.nan, 0.1)
    passed = [c["difference"] <= ms.TOLERANCE for c in checks]
    assert passed == [True, True, False, False]
