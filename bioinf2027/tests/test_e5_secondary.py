"""E5's secondary analyses on synthetic data (bioinf2027/plans/E5.md, Sections 3, 7, 8, 16.1 and 18).

The fold loop must give the family's predictions exactly, the recomputed family must equal the family's row, and
each secondary construction (prefix folds, the family baseline, the B2 residual, the forest, VCC) must do what the
plan says. No file under the data root is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import e5fit  # noqa: E402
import e5lib  # noqa: E402
import make_e5_fits as fits  # noqa: E402
import make_e5_secondary as sec  # noqa: E402

WIDTHS = {"Geneformer": 6, "scGPT": 4, "GenePT": 9}
SYMBOLS = ([f"RPL{i}" for i in range(1, 13)] + [f"RPS{i}" for i in range(1, 9)] + [f"PSMA{i}" for i in range(1, 8)]
           + [f"MED{i}" for i in range(1, 11)] + [f"POLR2{c}" for c in "ABCDE"] + [f"GENE{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + ["GENEAA", "GENEAB"])


@pytest.fixture
def synthetic(monkeypatch):
    """Two screens of 60 genes (50 shared), Bc, both endpoints and the secondary columns, three small embeddings."""
    rng = np.random.default_rng(7)
    genes = [f"ENSG{i:011d}" for i in range(len(SYMBOLS))]
    symbol_of = dict(zip(genes, SYMBOLS))
    embeddings = {n: {g: rng.normal(size=w) for g in genes} for n, w in WIDTHS.items()}
    screens = {}
    for j, s in enumerate(fits.SCREENS):
        own = genes[j * 10: j * 10 + 60]
        f = pd.DataFrame(index=pd.Index(own, name="target_ensg"))
        for c in ("B1", "detection_rate", "control_sd", "ceiling", "B0", "B2"):
            f[c] = rng.normal(size=len(own))
            f[f"rank_{c}"] = e5lib.ranks01(f[c].to_numpy())
        signal = np.array([embeddings["Geneformer"][g][0] for g in own])
        f["a"] = 0.5 * f["B1"] + signal + 0.5 * rng.normal(size=len(own))
        f["b"] = 0.3 * f["B2"] + signal + 0.5 * rng.normal(size=len(own))
        for c in ("b_10kb", "b_full", "b_depth", "b50", "a10"):
            f[c] = f["b"] + 0.1 * rng.normal(size=len(own))
        f.loc[f.index[:8], "b50"] = np.nan  # fewer genes hold the 50-cell version
        f["symbol"] = [symbol_of[g] for g in own]
        x = pd.DataFrame(rng.normal(size=(len(own), 10)), index=own)  # the observed shift the positive control reads
        x[0] += f["b"].to_numpy()
        screens[s] = {"frame": f, "x": x}
    within = {s: e5lib.within_screen_folds(screens[s]["frame"].index) for s in fits.SCREENS}
    union = sorted(set(screens["RPE1"]["frame"].index) | set(screens["K562-essential"]["frame"].index))
    order = np.random.default_rng(fits.UNION_FOLD_SEED).permutation(len(union))
    data = {"screens": screens, "within_folds": within, "union_folds": {union[i]: int(p % 5) for p, i in enumerate(order)},
            "embeddings": embeddings, "covered": {s: set(screens[s]["frame"].index) for s in fits.SCREENS},
            "gates_b": {"gates": {s: {lv: {"reliability_gate_passed": True, "beyond_bc_gate_passed": lv == "27"} for lv in ("27", "50")} for s in fits.SCREENS}},
            "record": {"inputs": {"gates_b": {"sha256": "0" * 64}}}}
    for s in fits.SCREENS:
        screens[s]["frame"]["b_minus_B2"] = sec.residual_on(screens[s]["frame"], "b", "rank_B2", within[s])
    prefix_of = {g: sec.symbol_prefix(symbol_of[g]) for g in genes}
    pc = pd.DataFrame([{"setting": st, "endpoint": e, "nominal_width": w, "resolved": True} for st in fits.SETTINGS for e in "ab" for w in WIDTHS.values()])
    pc50 = pd.DataFrame([{"setting": st, "endpoint": "b50", "nominal_width": w, "version": "planned", "resolved": False}
                         for st in fits.SETTINGS for w in WIDTHS.values()])  # unlike the 27-cell control, not resolved
    ours = {"prefix_of": prefix_of, "prefix_folds": {**{s: sec.prefix_folds(screens[s]["frame"].index, prefix_of, e5lib.WITHIN_FOLD_SEED) for s in fits.SCREENS},
                                                     "union": sec.prefix_folds(union, prefix_of, fits.UNION_FOLD_SEED)},
            "gates_a": {s: {"reliability_gate_passed": True, "beyond_bc_gate_passed": True} for s in fits.SCREENS},
            "pc": pc, "pc_revised": pc.iloc[:0], "pc50": pc50, "inputs": {"embeddings": {n: {"sha256": n} for n in WIDTHS}}, "code": {"x": "1"}}
    monkeypatch.setitem(fits.BC_COLUMNS, "b50", fits.BC_COLUMNS["b"])
    monkeypatch.setattr(fits, "DATA", data)
    monkeypatch.setattr(sec, "DATA", ours)
    monkeypatch.setattr(fits, "B", 60)
    monkeypatch.setattr(sec, "B", 60)
    return data


def family_models(name: str, genes):
    vectors = fits.DATA["embeddings"][name]
    width = next(iter(vectors.values())).size
    shuffled = e5fit.shuffled({g: vectors[g] for g in genes}, genes)
    return {"Bc": None, "[Bc; e]": fits.embedding_provider(name, vectors), "[Bc; C]": fits.gaussian_provider(width),
            "[Bc; S]": fits.embedding_provider(name, shuffled)}


# --------------------------------------------------------------------------- the harness, unchanged


@pytest.mark.parametrize("setting", ["within RPE1", "RPE1 to K562-essential"])
@pytest.mark.parametrize("endpoint", ["a", "b"])
def test_fold_loop_gives_the_family_predictions_exactly(synthetic, setting, endpoint):
    genes = fits.setting_genes(setting)
    theirs = fits.run_setting(setting, endpoint, family_models("scGPT", genes), genes)
    models, width = sec.models_for("scGPT", genes)
    ours = sec.fit_folds(sec.setting_rows(setting, endpoint, sec.bar_columns("Bc", endpoint), genes), models)
    assert width == 4 and ours["genes"] == theirs["genes"] and np.array_equal(ours["y"], theirs["y"])
    for mine, family in zip(("bar", "e", "C", "S"), ("Bc", "[Bc; e]", "[Bc; C]", "[Bc; S]")):
        assert np.array_equal(ours["predictions"][mine], theirs["predictions"][family])
    assert len(ours["fold_ids"]) == 5 and ours["folds"].size == len(ours["genes"])


def test_recomputed_family_equals_the_family_row_and_a_difference_stops(synthetic, monkeypatch):
    monkeypatch.setattr(fits, "SMOKE", True)  # one refit in the family's row; no statistic compared here depends on it
    monkeypatch.setattr(sec, "SMOKE", False)
    rows = []
    for setting in ("within K562-essential", "K562-essential to RPE1"):
        family = fits.family_task(("GenePT", "a", setting))
        ours = sec.main_task((sec.PRIMARY, "GenePT", "a", setting))["row"]
        rows.append((ours, family))
    table = pd.DataFrame([f for _, f in rows])
    check = sec.check_against_family([o for o, _ in rows], table)
    assert check["compared"] and check["largest_difference"] == 0.0 and "iu_p" in check["quantities"]
    table.loc[0, "spearman_[Bc; e]"] += 1e-6
    with pytest.raises(sec.Stop):
        sec.check_against_family([o for o, _ in rows], table)
    assert sec.check_against_family([], None)["compared"] is False


# --------------------------------------------------------------------------- gene families


def test_symbol_prefix_takes_the_characters_before_the_first_digit():
    assert [sec.symbol_prefix(s) for s in ("RPL13A", "rps6", "PSMA1", "MED12", "POLR2A", "AASS", "C1orf109", "HLA-DRB1", "7SK")] == \
        ["RPL", "RPS", "PSMA", "MED", "POLR", "AASS", "C", "HLA-DRB", "7SK"]


def test_prefix_folds_keep_each_family_in_one_fold_and_are_fixed(synthetic):
    prefix_of = sec.DATA["prefix_of"]
    folds = sec.prefix_folds(sorted(prefix_of), prefix_of, 11)
    assert folds == sec.prefix_folds(sorted(prefix_of), prefix_of, 11)
    for p in set(prefix_of.values()):
        assert len({folds[g] for g in prefix_of if prefix_of[g] == p}) == 1
    assert set(folds.values()) == set(range(5))
    R = sec.setting_rows("within RPE1", "a", sec.bar_columns("Bc", "a"), None, "prefix-grouped folds")
    for k in range(5):
        train = {prefix_of[g] for g in R["src"] if R["folds"][g] != k}
        test = {prefix_of[g] for g in R["tgt"] if R["folds"][g] == k}
        assert not train & test


def test_family_baseline_takes_the_family_mean_otherwise_the_training_mean(synthetic):
    frame = pd.DataFrame({"y": [1.0, 3.0, 10.0, 7.0, 0.0]}, index=["g1", "g2", "g3", "g4", "g5"])
    sec.DATA["prefix_of"] = {"g1": "RPL", "g2": "RPL", "g3": "MED", "g4": "RPL", "g5": "XYZ"}
    R = {"src_frame": frame, "src_label": "y", "src": ["g1", "g2", "g3", "g4", "g5"], "tgt": ["g1", "g2", "g3", "g4", "g5"],
         "folds": {"g1": 1, "g2": 1, "g3": 1, "g4": 0, "g5": 0}, "ks": (0, 1)}
    genes, pred, with_family = sec.family_neighbour(R)
    assert genes == ["g4", "g5", "g1", "g2", "g3"]
    # fold 0: training g1, g2 (RPL), g3 (MED); g4 (RPL) -> 2.0, g5 (no family) -> the training mean 14 / 3
    # fold 1: training g4 (RPL), g5; g1 and g2 (RPL) -> 7.0, g3 (MED, none in training) -> 3.5
    assert np.allclose(pred, [2.0, 14.0 / 3.0, 7.0, 7.0, 3.5]) and with_family == 3


# --------------------------------------------------------------------------- the secondary columns


def test_b2_residual_is_out_of_fold(synthetic):
    f = fits.DATA["screens"]["RPE1"]["frame"].copy()
    folds_of = fits.DATA["within_folds"]["RPE1"]
    before = sec.residual_on(f, "b", "rank_B2", folds_of)
    folds = np.array([folds_of[g] for g in f.index])
    assert np.array_equal(before, e5lib.out_of_fold_residuals(f[["rank_B2"]].to_numpy(), f["b"].to_numpy(), folds))
    moved = f.index[folds == 0]
    f.loc[moved, "b"] += 5.0
    after = sec.residual_on(f, "b", "rank_B2", folds_of)
    assert np.allclose(after[folds == 0] - before[folds == 0], 5.0)  # the fold's own prediction does not use its labels


def test_added_columns_are_gene_medians_and_the_assigned_50_cell_version():
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index(["G1", "G2"], name="target_ensg"))
    genes_b = pd.DataFrame({"target_ensg": ["G2", "G1"], "target_symbol": ["MED4", "RPL3"], "b27_10kb": [0.2, 0.1], "b27_full": [2.0, 1.0],
                            "b27_depth": [0.4, 0.3], "b50": [5.0, 6.0], "b50_gem": [7.0, 8.0]})
    g1 = pd.DataFrame({"target_ensg": ["G1", "G1", "G1", "G2", "G2"], "n_pool": [30, 40, 20, 27, 100], "B0": [1.0, 3.0, 99.0, 2.0, 4.0],
                       "B2": [0.5, 0.7, 9.0, 0.1, 0.3]})
    out = sec.add_columns(frame, genes_b, g1, "b50_gem")
    assert out.loc["G1", "symbol"] == "RPL3" and out.loc["G2", "b_full"] == 2.0 and out.loc["G1", "b50"] == 8.0
    assert out.loc["G1", "B0"] == 2.0 and out.loc["G2", "B2"] == pytest.approx(0.2)  # the 20-cell construct is not eligible
    assert list(out["rank_B0"]) == [0.5, 1.0]


def test_bars_and_names():
    assert sec.bar_columns("Bc", "a") == ["rank_B1", "rank_detection_rate", "rank_control_sd", "rank_ceiling"]
    assert sec.bar_columns("Bc", "b_full") == ["rank_B1", "rank_detection_rate", "rank_control_sd"]
    assert sec.bar_columns("B1", "a") == ["rank_B1"] and sec.bar_columns(None, "B1") == []
    assert sec.bar_columns("[Bc; B0]", "b")[-1] == "rank_B0"
    assert sec.model_names("[Bc; B0]")["e"] == "[Bc; B0; e]" and sec.model_names(None) == {"e": "e", "C": "C", "S": "S"}
    assert [sec.endpoint_of(x) for x in ("a", "a10", "b50", "b_minus_B2", "B1", "detection_rate")] == ["a", "a", "b", "b", "none", "none"]


def test_every_main_item_runs_and_is_labelled(synthetic):
    for item, spec in sec.ITEMS.items():
        for label in spec["labels"]:
            out = sec.main_task((item, "Geneformer", label, "K562-essential to RPE1"))
            row = out["row"]
            assert row["role"] == spec["role"] and row["item"] == item and isinstance(row["svd_fallbacks"], int)
            assert np.isfinite(row["spearman_e"]) and np.isfinite(row["iu_p"])
            assert row["power_2.80se"] == pytest.approx(2.80 * max(row["minus_gaussian_se"], row["minus_shuffled_se"]))
            if spec["bar"] is None:
                assert np.isnan(row["spearman_bar"]) and row["gates_passed"] is None and row["positive_control_resolved"] is None
            elif label != "b50":
                assert np.isfinite(row["minus_bar_estimate"]) and row["positive_control_resolved"] is True
            if label == "b50":  # the 50-cell gates and the 50-cell control; the genes holding the version
                assert row["gates_passed"] is False and row["positive_control_resolved"] is False and row["n_test_genes"] == 52
                assert row["positive_control_version"] == "planned, 50-cell control"
            if item == "(b) residualised on B2":
                assert row["reading"] in (sec.RESPONSE_PER_KNOCKDOWN, "not read as response per unit knockdown")
                assert (row["reading"] == sec.RESPONSE_PER_KNOCKDOWN) == row["both_resolved_above_zero"]
            if item == sec.PRIMARY:
                assert out["prefix_row"]["clusters"] == "symbol prefixes" and out["prefix_row"]["n_clusters"] < row["n_test_genes"]
                assert {r["model"] for r in out["folds"]} == {"Bc", "[Bc; e]", "[Bc; C]", "[Bc; S]", "mean predictor (R-squared only)"}


def test_fold_rows_are_within_fold_and_centred_spearman(synthetic):
    rng = np.random.default_rng(2)
    y, folds = rng.normal(size=30), np.repeat(np.arange(3), 10)
    pred = y + rng.normal(size=30) + folds * 3.0
    result = {"y": y, "folds": folds, "fold_ids": [0, 1, 2], "train_sizes": [20, 20, 20], "predictions": {"e": pred},
              "records": {"e": [{"alpha": 1.0, "at_grid_edge": False}] * 3}, "mean_predictor": np.zeros(30)}
    rows = sec.fold_rows(result, {"setting": "x"}, {"e": "[Bc; e]"})
    centred = pred - np.repeat([pred[folds == k].mean() for k in range(3)], 10)
    assert rows[0]["fold_1_spearman"] == pytest.approx(spearmanr(pred[10:20], y[10:20])[0])
    assert rows[0]["pooled_centred_within_fold"] == pytest.approx(spearmanr(centred, y)[0])
    assert rows[1]["model"] == "mean predictor (R-squared only)" and "pooled_spearman" not in rows[1]


def test_between_embedding_contrasts_and_the_family_baseline(synthetic):
    results = {n: sec.main_task((sec.PRIMARY, n, "b", "within RPE1"))["result"] for n in sec.EMBEDDINGS}
    out = sec.contrast_task(("b", "within RPE1", results))
    assert [(r["first"], r["second"]) for r in out["between"]] == [("Geneformer", "scGPT"), ("Geneformer", "GenePT"), ("scGPT", "GenePT")]
    assert not any("rank" in k for r in out["between"] for k in r)
    assert out["between"][0]["e_difference_estimate"] == pytest.approx(
        spearmanr(results["Geneformer"]["predictions"]["e"], results["Geneformer"]["y"])[0]
        - spearmanr(results["scGPT"]["predictions"]["e"], results["scGPT"]["y"])[0])
    assert len(out["baseline"]) == 3 and out["baseline"][0]["test_genes_with_a_training_family_member"] > 0
    broken = {**results, "scGPT": {**results["scGPT"], "genes": results["scGPT"]["genes"][::-1]}}
    with pytest.raises(sec.Stop):
        sec.contrast_task(("b", "within RPE1", broken))


def test_power_rows_use_the_larger_se():
    row = {"embedding": "scGPT", "width": 512, "endpoint": "a", "setting": "within RPE1", "n_test_genes": 9, "iu_estimate": 0.01,
           "minus_gaussian_se": 0.02, "minus_shuffled_se": 0.03, "replicate_index_sha256": ""}
    out = sec.power_rows([row], None)[0]
    assert out["power_2.80se"] == pytest.approx(0.084) and out["holm_first_step_detectable"] == pytest.approx(3.92 * 0.03)


# --------------------------------------------------------------------------- the forest


def test_seed_averaged_spearman_is_the_mean_over_seeds():
    a = {"names": ["e"], "point": {"e": 0.2}, "draws": {"e": np.array([0.1, 0.3])}, "index_sha256": "x"}
    b = {"names": ["e"], "point": {"e": 0.4}, "draws": {"e": np.array([0.5, 0.1])}, "index_sha256": "x"}
    out = sec.seed_averaged([a, b])
    assert out["point"]["e"] == pytest.approx(0.3) and np.allclose(out["draws"]["e"], [0.3, 0.2])
    with pytest.raises(sec.Stop):
        sec.seed_averaged([a, {**b, "index_sha256": "y"}])


def test_forest_on_rank_columns_is_fixed_by_its_seed(synthetic, monkeypatch):
    monkeypatch.setattr(sec, "SMOKE", True)  # 5 trees
    genes = fits.setting_genes("within RPE1")
    R = sec.setting_rows("within RPE1", "a", sec.bar_columns("Bc", "a"), genes)
    models, _ = sec.models_for("scGPT", genes)
    one = sec.fit_folds(R, models, head="forest", seed=0)
    assert np.array_equal(one["predictions"]["e"], sec.fit_folds(R, models, head="forest", seed=0)["predictions"]["e"])
    assert not np.array_equal(one["predictions"]["e"], sec.fit_folds(R, models, head="forest", seed=1)["predictions"]["e"])
    assert one["records"]["bar"][0]["columns"] == 4 and one["records"]["e"][0]["columns"] == 4 + 4
    assert all(np.isfinite(p).all() for p in one["predictions"].values())


def test_forest_fit_is_reused_only_with_the_same_specification(synthetic, monkeypatch, tmp_path):
    monkeypatch.setattr(sec, "SMOKE", True)
    monkeypatch.setattr(sec, "data_dir", lambda: tmp_path)
    first = sec.forest_task(("scGPT", "b", "within RPE1", 0))
    again = sec.forest_task(("scGPT", "b", "within RPE1", 0))
    assert not first["reused"] and again["reused"] and again["result"]["genes"] == first["result"]["genes"]
    for m in ("bar", "e", "C", "S"):
        assert np.array_equal(again["result"]["predictions"][m], first["result"]["predictions"][m])
    sec.DATA["code"] = {"x": "2"}
    assert not sec.forest_task(("scGPT", "b", "within RPE1", 0))["reused"]
    row = sec.forest_stats_task(("scGPT", "b", "within RPE1", [again]))
    assert row["head"] == "forest" and row["bar"] == "Bc (rank columns)" and row["fits_reused"] == 1 and row["width"] == 4


# --------------------------------------------------------------------------- VCC


def vcc_inputs():
    rng = np.random.default_rng(5)
    rows = []
    for split, n in (("train", 30), ("val", 6), ("test", 20)):
        for i in range(n):
            rows.append({"split": split, "target_gene": f"V{split}{i}", "B1": rng.normal(), "detection_rate": rng.uniform(),
                         "control_sd": rng.uniform(), "b27": rng.normal(), "b10": rng.normal(), "b50": rng.normal()})
    genes = pd.DataFrame(rows)
    g3 = genes[["split", "target_gene"]].assign(y=rng.normal(size=len(genes)))
    g4 = genes[["split", "target_gene"]].assign(y=rng.normal(size=len(genes)))
    ensembl_of = {g: f"ENSV{i:05d}" for i, g in enumerate(genes.target_gene)}
    ensembl_of["Vtest0"] = "ENSG00000000010"  # a VCC test gene that is also a Replogle gene
    ensembl_of["Vtrain0"] = "ENSG00000000011"  # a VCC training gene that is also a Replogle gene
    del ensembl_of["Vtest1"]  # a symbol without an identifier is dropped
    return genes, g3, g4, ensembl_of


def test_vcc_frame_rows_and_task(synthetic):
    genes, g3, g4, ensembl_of = vcc_inputs()
    frame = sec.vcc_frame(genes, g3, g4, ensembl_of)
    assert len(frame) == len(genes) - 1 and frame.loc["ENSG00000000011", "split"] == "train"
    assert frame.loc["ENSG00000000010", "a27"] == float(g3.loc[g3.target_gene == "Vtest0", "y"].iloc[0])
    rng = np.random.default_rng(1)
    for n in WIDTHS:
        fits.DATA["embeddings"][n].update({g: rng.normal(size=WIDTHS[n]) for g in frame.index})
    sec.DATA.update({"vcc_frame": frame, "vcc_covered": set(frame.index), "vcc_excluded": set(frame.index[frame.split.isin(["train", "val"])])})
    R = sec.vcc_rows(sec.VCC_TEST, "a27", "a27")
    assert len(R["src"]) == 30 and len(R["tgt"]) == 19 and R["ks"] == (0,)
    R = sec.vcc_rows("VCC training to RPE1", "a27", "a")
    assert "ENSG00000000011" not in R["tgt"] and "ENSG00000000010" in R["tgt"] and len(R["tgt"]) == 59
    out = sec.vcc_task(("GenePT", "VCC training to K562-essential", "a10", "a10", "(a), G4 to G2, 10 cells"))
    assert out["row"]["role"] == "secondary, descriptive" and out["row"]["n_test_genes"] == 59 and np.isfinite(out["row"]["iu_p"])
    with pytest.raises(sec.Stop):
        sec.vcc_frame(genes, pd.concat([g3, pd.DataFrame([{"split": "test", "target_gene": "nowhere", "y": 1.0}])]), g4, ensembl_of)


def test_agreement_is_the_spearman_of_the_two_endpoints(synthetic):
    frames = {s: fits.DATA["screens"][s]["frame"] for s in fits.SCREENS}
    rows = sec.agreement_rows({s: 1 for s in frames}, frames, {"RPE1": set(frames["RPE1"].index[:40])}, (("a", "b50"),))
    f = frames["RPE1"]
    covered = rows[0]
    part = f.iloc[:40].dropna(subset=["b50"])
    assert covered["subset"] == "covered genes" and covered["n_genes"] == len(part)
    assert covered["spearman"] == pytest.approx(spearmanr(part["a"], part["b50"])[0])
    assert covered["lo"] <= covered["spearman"] <= covered["hi"]
    assert rows[1]["subset"] == "all genes with both labels" and rows[1]["n_genes"] == 52


# --------------------------------------------------------------------------- the size association (Section 8)


def test_size_frame_takes_b0_as_the_gene_median_of_eligible_constructs():
    genes_b = pd.DataFrame({"target_ensg": ["G1", "G2"], "b27": [1.0, 2.0], "b50": [3.0, np.nan], "all_cells": [5.0, 6.0], "n_scaled": [7.0, 8.0]})
    g1 = pd.DataFrame({"target_ensg": ["G1", "G1", "G2", "G2"], "n_pool": [30, 40, 10, 60], "B0": [3.0, 4.0, 1.0, 5.0]})
    f = sec.size_frame(genes_b, g1, "b27", "b50")
    assert list(f["B0"]) == [3.5, 5.0] and list(f["fixed"]) == [1.0, 2.0] and np.isnan(f.loc["G2", "other_version"])


def test_family_b_column_follows_the_gates_file():
    gates = {"versions_used_by_the_family": {"within RPE1": {"version_used_by_the_family": "27 cells"}},
             "gates": {"RPE1": {"27": {"primary_version": "pooled controls"}}}}
    assert sec.family_b_column(gates, "RPE1") == "b27"
    gates["gates"]["RPE1"]["27"]["primary_version"] = "gem-group-matched controls"
    assert sec.family_b_column(gates, "RPE1") == "b27_gem"
    gates["versions_used_by_the_family"]["within RPE1"]["version_used_by_the_family"] = "50 cells"
    gates["gates"]["RPE1"]["50"] = {"primary_version": "pooled controls"}
    assert sec.family_b_column(gates, "RPE1") == "b50"


def size_frames(fixed_sign: float):
    rng = np.random.default_rng(4)
    frames = {}
    for s in ("RPE1", "K562-essential"):
        b0 = rng.normal(size=300)
        frames[s] = pd.DataFrame({"B0": b0, "fixed": fixed_sign * b0 + 0.5 * rng.normal(size=300), "other_version": rng.normal(size=300),
                                  "all_cells": b0 + rng.normal(size=300), "n_scaled": b0 + 0.3 * rng.normal(size=300)},
                                 index=[f"G{i}" for i in range(300)])
        frames[s].iloc[:20, frames[s].columns.get_loc("other_version")] = np.nan
    return frames


def test_size_statement_where_the_n_scaled_association_is_resolved_and_the_fixed_cell_one_is_not(monkeypatch):
    monkeypatch.setattr(sec, "B", 200)
    columns = {s: {"fixed": "b27", "other_version": "b50"} for s in ("RPE1", "K562-essential")}
    rows = sec.size_rows(size_frames(-1.0), {"RPE1": 1, "K562-essential": 2}, columns)  # fixed: resolved below zero, so not above
    assert len(rows) == 8
    rpe1 = {r["statistic"]: r for r in rows if r["screen"] == "RPE1"}
    f = size_frames(-1.0)["RPE1"]
    assert rpe1["n-scaled, all cells"]["spearman_with_B0"] == pytest.approx(spearmanr(f["n_scaled"], f["B0"])[0])
    assert rpe1["n-scaled, all cells"]["resolved_above_zero"] and not rpe1["fixed cells (the family's version)"]["resolved_above_zero"]
    assert rpe1["fixed cells (the family's version)"]["column"] == "b27" and rpe1["n-scaled, all cells"]["in_the_rule"]
    assert not rpe1["all cells"]["in_the_rule"] and rpe1["all cells"]["statement"] == sec.SIZE_STATEMENT
    other = rpe1["fixed cells (the other version, beside)"]
    assert other["n_genes"] == 280 and other["statement"] == "" and not other["in_the_rule"]
    both = sec.size_rows(size_frames(1.0), {"RPE1": 1, "K562-essential": 2}, columns)  # both resolved above zero: no statement
    assert all(r["statement"] == "" for r in both)


# --------------------------------------------------------------------------- the 50-cell ceiling and positive control


@pytest.mark.parametrize("setting", ["within K562-essential", "K562-essential to RPE1"])
def test_ceiling_row_is_the_family_ceiling_task_with_the_genes_as_an_argument(synthetic, setting):
    ours = sec.ceiling_row(setting, "b", fits.shared_genes())
    theirs = fits.ceiling_task((setting, "b"))
    assert ours.keys() == theirs.keys()
    for k in ours:
        assert ours[k] == theirs[k] or (isinstance(ours[k], float) and np.isnan(ours[k]) and np.isnan(theirs[k]))


def test_prefit50_tasks_use_the_50_cell_label(synthetic, monkeypatch):
    sec.DATA["shared50"] = sec.shared_with_b50()
    assert len(sec.DATA["shared50"]) == 42  # 50 shared genes, 8 of them without the 50-cell version in K562-essential
    ceiling = sec.ceiling50_task("RPE1 to K562-essential")
    assert ceiling["endpoint"] == "b50" and ceiling["n_test_genes"] == 42 and np.isfinite(ceiling["difference_p"])
    assert isinstance(ceiling["svd_fallbacks"], int)
    planned = sec.pc50_task(("within RPE1", "b50", 4))
    assert planned["endpoint"] == "b50" and planned["version"] == "planned" and planned["n_test_genes"] == 50
    assert np.isfinite(planned["iu_p"]) and planned["fold_widths"] == "4;4;4;4;4"
    monkeypatch.setattr(fits, "REVISED_COMPONENTS", 3)  # the revised control at a width above the component limit
    revised = sec.pc50_task(("K562-essential to RPE1", "b50", 4, True))
    assert revised["version"] == fits.REVISED and revised["fold_padding"] == "1;1;1;1;1" and revised["n_test_genes"] == 50


def test_main_part_needs_the_complete_50_cell_control(synthetic, monkeypatch, tmp_path):
    from bioinf2027.provenance import sha256

    monkeypatch.setattr(sec, "out_dir", lambda: tmp_path)
    monkeypatch.setattr(sec.config, "portable", lambda p: str(p))  # tmp_path is outside both roots
    monkeypatch.setattr(sec, "file_record", lambda p: {"sha256": sha256(p)})
    with pytest.raises(sec.Stop):
        sec.load_prefit50()
    _, wanted = sec.prefit50_args()
    assert len(wanted) == 4 * 3 + 4  # planned at 512, 768 and 1,536 and revised at 1,536, in every setting
    table = pd.DataFrame([{"setting": a[0], "endpoint": "b50", "nominal_width": a[2], "version": fits.REVISED if len(a) > 3 else "planned",
                           "resolved": True} for a in wanted])

    def write(t, smoke=False):
        t.to_csv(tmp_path / "positive_control_50.csv", index=False)
        record = {"smoke": smoke, "outputs": {"positive_control_50.csv": {"sha256": sha256(tmp_path / "positive_control_50.csv")}}}
        (tmp_path / "secondary_prefit50.json").write_text(__import__("json").dumps(record))

    write(table)
    sec.load_prefit50()
    assert len(sec.DATA["pc50"]) == 16 and "prefit50" in sec.DATA["inputs"]
    write(table, smoke=True)
    with pytest.raises(sec.Stop):
        sec.load_prefit50()
    write(table.iloc[1:])
    with pytest.raises(sec.Stop):
        sec.load_prefit50()


# --------------------------------------------------------------------------- records


def test_predictions_are_saved_with_their_index(tmp_path, monkeypatch):
    from bioinf2027.provenance import sha256

    monkeypatch.setattr(sec, "file_record", lambda p: {"bytes": p.stat().st_size, "sha256": sha256(p)})  # tmp_path is outside both roots
    result = {"genes": ["G1", "G2"], "y": np.array([1.0, 2.0]), "folds": np.array([0, 1]), "predictions": {"e": np.array([0.5, 0.25])}}
    record = sec.save_predictions(tmp_path / "p.npz", [(("item", "scGPT", "a", "within RPE1"), result)])
    with np.load(tmp_path / "p.npz", allow_pickle=False) as store:
        assert list(store["r0_genes"]) == ["G1", "G2"] and np.array_equal(store["r0_e"], [0.5, 0.25])
        assert '"scGPT"' in str(store["index"])
    assert record["bytes"] > 0 and len(record["sha256"]) == 64


def test_smoke_outputs_go_to_the_scratch_folder(monkeypatch):
    monkeypatch.setattr(sec, "SMOKE", True)
    assert sec.out_dir().as_posix().endswith("bioinf2027/scratch/E5_secondary_smoke")
    assert sec.data_dir().as_posix().endswith("bioinf2027/scratch/E5_secondary_smoke/data")
    assert sec.active_settings() == ("within RPE1", "RPE1 to K562-essential")
    monkeypatch.setattr(sec, "SMOKE", False)
    assert sec.out_dir().as_posix().endswith("bioinf2027/results/E5/secondary") and len(sec.active_settings()) == 4


def test_a_saved_task_is_reused_with_the_same_values_and_redone_when_its_specification_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(sec, "data_dir", lambda: tmp_path)
    monkeypatch.setitem(sec.DATA, "code", {"script": "abc"})
    monkeypatch.setitem(sec.DATA, "inputs", {"family": {"sha256": "def"}})
    calls = []

    def task(args):
        calls.append(args)
        rng = np.random.default_rng(3)
        return {"key": args, "result": {"genes": ["G1", "G2"], "y": rng.normal(size=2), "predictions": {"e": rng.normal(size=2)}},
                "row": {"spearman_e": np.float64(0.1) + 1e-17, "both_resolved_above_zero": np.bool_(True), "n": np.int64(7),
                        "lo": float("nan"), "label": "b"}}
    first, reused = sec.saved("main", task, ("item", "Geneformer", "b", "within RPE1"), ("item", "Geneformer", "b", "within RPE1"))
    assert not reused and len(calls) == 1
    again, reused = sec.saved("main", task, ("item", "Geneformer", "b", "within RPE1"), ("item", "Geneformer", "b", "within RPE1"))
    assert reused and len(calls) == 1
    assert again["key"] == first["key"] and isinstance(again["key"], tuple)
    assert np.array_equal(again["result"]["y"], first["result"]["y"]) and again["result"]["y"].dtype == first["result"]["y"].dtype
    assert np.array_equal(again["result"]["predictions"]["e"], first["result"]["predictions"]["e"])
    for k, v in first["row"].items():
        assert type(again["row"][k]) is type(v)
        assert (np.isnan(v) and np.isnan(again["row"][k])) if k == "lo" else again["row"][k] == v
    monkeypatch.setattr(sec, "B", sec.B + 1)  # another number of replicates: done again
    sec.saved("main", task, ("item", "Geneformer", "b", "within RPE1"), ("item", "Geneformer", "b", "within RPE1"))
    assert len(calls) == 2


def _slow_square(x: int) -> dict:
    import time
    time.sleep(0.05 * (4 - x))  # the first tasks end last
    return {"x": x, "square": np.array([float(x) ** 2])}


def test_slots_map_returns_what_map_returns_in_the_listed_order(monkeypatch):
    monkeypatch.setattr(sec, "WORKERS", 3)
    monkeypatch.setattr(sec, "WORKERS_FILE", None)
    listed = [0, 1, 2, 3]
    got = sec.slots_map(_slow_square, listed, "test")
    want = list(map(_slow_square, listed))
    assert [g["x"] for g in got] == listed
    assert all(np.array_equal(g["square"], w["square"]) and g["square"].dtype == w["square"].dtype for g, w in zip(got, want))
    assert sec.DATA["shed"]["test"] == 0


def _fails(x: int) -> int:
    if x == 2:
        raise ValueError("no")
    return x


def test_slots_map_stops_when_a_task_fails(monkeypatch):
    monkeypatch.setattr(sec, "WORKERS", 2)
    monkeypatch.setattr(sec, "WORKERS_FILE", None)
    with pytest.raises(sec.Stop, match="1 tasks of test failed"):
        sec.slots_map(_fails, [0, 1, 2, 3], "test")
