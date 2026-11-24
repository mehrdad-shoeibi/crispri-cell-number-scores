"""Tests of the D1b script's bookkeeping: sample sizes, exclusions, guards, routes and tables."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_d1b as d1b  # noqa: E402

B = 40


def _source_table(seed: int = 0) -> pd.DataFrame:
    """Three splits; in each, some target genes are features and some are not."""
    rng = np.random.default_rng(seed)
    frames = []
    for split, n_genes in (("train", 30), ("val", 14), ("test", 20)):
        for g in range(n_genes):
            is_feature = g % 3 != 0
            for batch in range(6):
                n_cells = int(rng.integers(10, 60))
                delta = -abs(rng.normal(1.0, 0.3)) if is_feature else np.nan
                label = (0.0 if np.isnan(delta) else -delta) + 0.02 * n_cells + rng.normal(0, 0.1)
                frames.append(
                    {
                        "split": split,
                        "batch": f"b{batch}",
                        "target_gene": f"{split}_g{g}",
                        "label": label,
                        "own_delta": delta,
                        "target_is_selected_feature": is_feature,
                        "n_perturbed_cells": n_cells,
                    }
                )
    return pd.DataFrame(frames)


# ----------------------------------------------------------------- part (a) and (b)


def test_part_a_reports_the_rows_and_genes_used_and_the_split_totals():
    table = _source_table()
    rows, quintiles = d1b.part_a(table, b=B)
    for split, n_genes in (("train", 30), ("val", 14), ("test", 20)):
        row = rows[rows["split"] == split].iloc[0]
        whole = table[table["split"] == split]
        used = whole[whole["target_is_selected_feature"]]
        assert row["n_rows_in_split"] == len(whole) == n_genes * 6
        assert row["n_target_genes_in_split"] == n_genes
        assert row["n_rows"] == len(used)
        assert row["n_target_genes"] == used["target_gene"].nunique()
        assert row["bootstrap_b"] == B
        assert "spearman_n_skipped" in row and "partial_n_skipped" in row
        assert row["few_clusters_descriptive"] == (used["target_gene"].nunique() < d1b.FEW_CLUSTERS)
        q = quintiles[quintiles["split"] == split]
        assert len(q) == d1b.N_QUINTILES
        assert q["n_rows"].sum() == len(used)
        assert (q["n_rows_in_split"] == len(whole)).all()
        assert q["n_target_genes"].notna().all()


def test_readability_rows_count_each_exclusion_reason_separately():
    rng = np.random.default_rng(1)
    n = 120
    groups = np.repeat(np.arange(40), 3).astype(str)
    selected = np.repeat(np.arange(40) % 2 == 0, 3)
    control = rng.uniform(0.5, 3.0, size=n)
    control[:9] = 0.001  # below the floor
    control[9:12] = np.nan  # missing
    own = np.where(selected, -rng.uniform(0.1, 1.0, size=n), np.nan)
    label = np.nan_to_num(control) + rng.normal(0, 0.2, size=n)

    rows = pd.DataFrame(
        d1b._readability_rows("screen", "construct", label, control, own, groups, selected, b=B)
    )
    ratio_all = rows[(rows.predictor == d1b.RATIO) & (rows.subset == "all")].iloc[0]
    has_mean = np.isfinite(control)
    has_delta = has_mean & np.isfinite(own)
    assert ratio_all["n_excluded_missing_control_mean"] == int((~has_mean).sum())
    assert ratio_all["n_excluded_no_own_delta"] == int((has_mean & ~np.isfinite(own)).sum())
    assert ratio_all["n_excluded_low_control_mean"] == int((has_delta & (control <= 0.01)).sum())
    assert ratio_all["n_rows"] == int((has_delta & (control > 0.01)).sum())
    assert ratio_all["n_rows_in_scope"] == n and ratio_all["n_target_genes_in_scope"] == 40

    mean_all = rows[(rows.predictor == "control_mean_expression") & (rows.subset == "all")].iloc[0]
    assert mean_all["n_rows"] == int(has_mean.sum())
    assert mean_all["n_excluded_missing_control_mean"] == int((~has_mean).sum())
    assert set(rows["row_unit"]) == {"construct"}


def test_readability_rows_label_the_source_ratio_row_by_the_rows_it_covers():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(30), 4).astype(str)
    selected = np.repeat(np.arange(30) % 2 == 0, 4)
    control = rng.uniform(0.5, 3.0, size=120)
    own = np.where(selected, -rng.uniform(0.1, 1.0, size=120), np.nan)
    rows = pd.DataFrame(
        d1b._readability_rows("VCC train", "row", control, control, own, groups, selected, b=B)
    )
    ratio_subsets = set(rows.loc[rows.predictor == d1b.RATIO, "subset"])
    assert ratio_subsets == {d1b.VCC_RATIO_ALL, "target_in_features"}


# ----------------------------------------------------------------- first-run guard


@pytest.fixture()
def first_run(tmp_path, monkeypatch):
    path = tmp_path / "first_run.csv"
    pd.DataFrame(
        {
            "table": ["a", "a", "b", "b"],
            "key": [
                "x",
                "y",
                "VCC train|negated_knockdown_fraction|all|spearman",
                "K562-essential|negated_knockdown_fraction|all|spearman",
            ],
            "value": [0.25, -0.5, 0.125, 0.75],
        }
    ).to_csv(path, index=False)
    monkeypatch.setattr(d1b, "FIRST_RUN_FILE", path)
    monkeypatch.setattr(d1b, "FIRST_RUN_COUNTS", {"a": 2, "b": 2})
    return path


def test_first_run_guard_passes_on_identical_values_and_declared_new_ones(first_run):
    report = d1b.compare_with_first_run(
        "a", {"x": 0.25, "y": -0.5, "new": 1.0}, allowed_new=lambda key: key == "new"
    )
    assert report["passed"] and report["values_compared"] == 2 and report["values_new"] == 1


def test_first_run_guard_stops_on_an_undeclared_new_value(first_run):
    with pytest.raises(RuntimeError):
        d1b.compare_with_first_run("a", {"x": 0.25, "y": -0.5, "new": 1.0})


def test_first_run_guard_stops_on_a_changed_value(first_run):
    with pytest.raises(RuntimeError):
        d1b.compare_with_first_run("a", {"x": 0.25 + 1e-9, "y": -0.5})


def test_first_run_guard_stops_on_a_missing_value(first_run):
    with pytest.raises(RuntimeError):
        d1b.compare_with_first_run("a", {"x": 0.25})


def test_first_run_guard_stops_on_a_value_that_is_not_finite(first_run):
    with pytest.raises(RuntimeError):
        d1b.compare_with_first_run("a", {"x": 0.25, "y": float("nan")})
    with pytest.raises(RuntimeError):
        d1b.compare_with_first_run("a", {"x": float("nan"), "y": -0.5})


def test_first_run_guard_stops_when_the_reference_lacks_rows(first_run, monkeypatch):
    monkeypatch.setattr(d1b, "FIRST_RUN_COUNTS", {"a": 3, "b": 2})
    with pytest.raises(RuntimeError):
        d1b.compare_with_first_run("a", {"x": 0.25, "y": -0.5})


def test_first_run_guard_follows_the_renamed_ratio(first_run):
    values = {
        f"VCC train|{d1b.RATIO}|{d1b.VCC_RATIO_ALL}|spearman": 0.125,
        f"K562-essential|{d1b.RATIO}|all|spearman": 0.75,
    }
    assert d1b.compare_with_first_run("b", values)["passed"]


def test_first_run_reference_file_exists_and_covers_every_table():
    reference = pd.read_csv(d1b.OUT_DIR / "D1b_first_run_point_estimates.csv")
    counts = reference.groupby("table").size().to_dict()
    assert counts == d1b.FIRST_RUN_COUNTS
    assert not reference.duplicated(["table", "key"]).any()
    assert reference["value"].notna().all()


# ----------------------------------------------------------------- control-mean routes


def _screen(n_genes: int = 5, pool: int = 100):
    endpoint = pd.DataFrame({"n_nt_pool": [pool, pool]})
    return SimpleNamespace(endpoint=endpoint, x_native=np.zeros((2, n_genes)))


def _write_pass_outputs(directory, *, all_passed=True, control_cells=100, n_genes=5, array=True, digest=None):
    mean = np.arange(n_genes, dtype=float)
    record = None
    if array:
        np.save(directory / "mu_nt_rpe1.npy", mean)
        record = {"sha256": digest or hashlib.sha256((directory / "mu_nt_rpe1.npy").read_bytes()).hexdigest()}
    checks = {
        "all_passed": all_passed,
        "checks": {
            "stored_index": {"passed": True},
            "checkpoint_prefix": {"passed": all_passed, "max_abs_difference_of_sums": 0.0},
            "identity": {
                "passed": all_passed,
                "identity_max_abs_diff": 0.0 if all_passed else 0.2,
                "control_cells": control_cells,
                "n_genes": n_genes,
            },
        },
    }
    if all_passed and record:
        checks["control_mean_array"] = record
    (directory / "D1b_rpe1_control_mean_checks.json").write_text(json.dumps(checks))


@pytest.fixture()
def pass_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(d1b, "OUT_DIR", tmp_path)
    monkeypatch.setattr(d1b, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(d1b.config, "portable", lambda path: "portable/" + Path(path).name)
    return tmp_path


def test_repeated_pass_route_says_not_run_only_without_a_checks_file(pass_dirs):
    with pytest.raises(d1b.ControlMeanUnavailable) as error:
        d1b.external_control_mean("RPE1", _screen())
    assert "not been run" in error.value.report["blocked_reason"]


def test_repeated_pass_route_reports_a_failed_pass_with_its_verdicts(pass_dirs):
    _write_pass_outputs(pass_dirs, all_passed=False, array=False)
    with pytest.raises(d1b.ControlMeanUnavailable) as error:
        d1b.external_control_mean("RPE1", _screen())
    report = error.value.report
    assert "failed" in report["blocked_reason"]
    assert report["checkpoint_prefix_passed"] is False
    assert report["identity_max_abs_diff"] == 0.2


def test_repeated_pass_route_refuses_a_left_over_array_under_a_failed_checks_file(pass_dirs):
    _write_pass_outputs(pass_dirs, all_passed=False, array=True)
    with pytest.raises(d1b.ControlMeanUnavailable):
        d1b.external_control_mean("RPE1", _screen())


def test_repeated_pass_route_returns_the_mean_when_every_check_passed(pass_dirs):
    _write_pass_outputs(pass_dirs)
    mean, report = d1b.external_control_mean("RPE1", _screen())
    assert mean.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert report["identity_passed"] and report["route"].startswith("repeated pass")


def test_repeated_pass_route_blocks_when_the_array_is_absent(pass_dirs):
    _write_pass_outputs(pass_dirs)
    (pass_dirs / "mu_nt_rpe1.npy").unlink()
    with pytest.raises(d1b.ControlMeanUnavailable) as error:
        d1b.external_control_mean("RPE1", _screen())
    assert "absent" in error.value.report["blocked_reason"]


def test_repeated_pass_route_blocks_an_array_with_another_digest(pass_dirs):
    _write_pass_outputs(pass_dirs, digest="0" * 64)
    with pytest.raises(d1b.ControlMeanUnavailable) as error:
        d1b.external_control_mean("RPE1", _screen())
    assert "digest" in error.value.report["blocked_reason"]


def test_repeated_pass_route_refuses_a_mean_from_another_gene_space_or_pool(pass_dirs):
    _write_pass_outputs(pass_dirs)
    with pytest.raises(RuntimeError):
        d1b.external_control_mean("RPE1", _screen(n_genes=6))
    with pytest.raises(RuntimeError):
        d1b.external_control_mean("RPE1", _screen(pool=99))


# ----------------------------------------------------------------- part (c) assembly


def _per_screen(seed: int, n_genes: int = 120):
    rng = np.random.default_rng(seed)
    genes = np.repeat(np.arange(n_genes), 2)
    selected = np.repeat(np.arange(n_genes) % 4 == 0, 2)
    y = rng.normal(size=genes.size)
    return pd.DataFrame(
        {
            "construct": [f"c{seed}_{i}" for i in range(genes.size)],
            "target_symbol": [f"G{g}" for g in genes],
            "target_ensg": [f"E{g}" for g in genes],
            "target_is_selected_feature": selected,
            "y_primary": y,
            "n_perturbed_pool": rng.integers(27, 300, size=genes.size).astype(float),
            "control_mean_target_gene": y + rng.normal(scale=0.5, size=genes.size),
            "own_delta_native": -y + rng.normal(scale=0.1, size=genes.size),
            "own_delta_mapped_zero_filled": np.where(selected, -y, 0.0),
            "release_own_delta_zero_filled": np.where(selected, -y, 0.0),
            "release_max_abs": np.abs(y) + 1.0,
            "release_max_abs_target_zeroed": np.where(selected, 1.0, np.abs(y) + 1.0),
            "release_target_unique_largest": selected,
            "release_target_ties_largest": np.zeros(genes.size, dtype=bool),
            "release_target_mapped_under_another_symbol": np.zeros(genes.size, dtype=bool),
            "cell_max_abs": np.abs(y) + 1.0,
            "cell_max_abs_target_zeroed": np.where(selected, 1.0, np.abs(y) + 1.0),
            "cell_target_unique_largest": selected,
            "cell_target_ties_largest": np.zeros(genes.size, dtype=bool),
            "cell_target_mapped_under_another_symbol": np.zeros(genes.size, dtype=bool),
        }
    )


def test_part_c_assembles_the_tables_and_the_family_mean(monkeypatch):
    objects = ("first", "second")
    monkeypatch.setattr(d1b, "FIRST_RUN_OBJECTS", objects)
    monkeypatch.setattr(d1b, "INPUT_TYPE", {"first": "expression only", "second": "magnitude summaries only"})

    def fake_objects(name, arrays):
        y = arrays["y_primary"].to_numpy()
        selected = arrays["target_is_selected_feature"].to_numpy()
        rng = np.random.default_rng(7)
        columns = np.column_stack(
            [np.where(selected, y, 0) + rng.normal(scale=0.3, size=y.size), rng.normal(size=y.size)]
        )
        columns = np.column_stack([columns, np.where(selected, y, 0.0)])
        families = {
            "first": {"group": "archived object, first run", "columns": [0], "n_seeds": 1, "split_meaning": "availability"},
            "second": {"group": "archived object, first run", "columns": [1], "n_seeds": 1, "split_meaning": "summary_availability"},
            d1b.B2_RELEASE: {"group": "trivial baseline", "columns": [2], "n_seeds": 1, "split_meaning": "positive_control_release"},
        }
        for title, (source, zeroed) in d1b.MAX_FAMILIES.items():
            column = f"{source}_max_abs" + ("_target_zeroed" if zeroed else "")
            columns = np.column_stack([columns, arrays[column].to_numpy()])
            families[title] = {"group": "predictor without a model", "columns": [columns.shape[1] - 1], "n_seeds": 1,
                               "split_meaning": "summary_availability",
                               "input": "release pseudobulk" if source == "release" else "cell-level delta"}
        return columns, families

    monkeypatch.setattr(d1b, "external_objects", fake_objects)
    monkeypatch.setattr(d1b, "d1_changed_constructs", lambda name: 60)
    per_screen = {"RPE1": _per_screen(1), "K562-essential": _per_screen(2)}
    vcc_ids = {"train": {"E1", "E2", "E4"}, "val": {"E8"}, "test": set()}
    vcc_symbols = {"train": {"G1", "G2"}, "val": set(), "test": set()}
    rows, differences, partials, composition, overlap, tertiles, masking, identifier = d1b.part_c(
        per_screen, vcc_ids, vcc_symbols, b=60
    )
    assert identifier.empty
    for screen in ("RPE1", "K562-essential"):
        part = masking[masking["screen"] == screen]
        assert len(part) == 4
        primary_rows = part[part["role"].str.startswith("primary")]
        assert len(primary_rows) == 1
        first = primary_rows.iloc[0]
        assert first["input"] == "cell-level delta" and first["subset"] == "all"
        assert first["call"] in d1b.MASKING_CALLS and first["masked_score_call"] in d1b.MASKED_SCORE_CALLS
        assert first["plain_minus_target_zeroed"] == pytest.approx(
            first["spearman_max_abs"] - first["spearman_max_abs_target_zeroed"])
        assert first["difference_lo"] <= first["plain_minus_target_zeroed"] <= first["difference_hi"]
        assert first["constructs_whose_max_abs_changes"] == 60
    row = overlap[(overlap["screen"] == "RPE1") & (overlap["subset"] == "all")].iloc[0]
    assert row["vcc_training_target_constructs_by_symbol"] == 4

    assert set(rows["construct_set"]) == {
        "all eligible constructs", "without VCC training targets", "without VCC training or validation targets",
    }
    mean = differences[(differences["object"] == "mean of the twelve first-run objects")]
    primary = mean[mean["construct_set"] == "all eligible constructs"]
    assert set(primary["role"]) == {"post-hoc summary"} and len(primary) == 2
    positive = differences[(differences["object"] == d1b.B2_RELEASE) & (differences["construct_set"] == "all eligible constructs")]
    assert (positive["spearman_out"] == 0.0).all() and positive["in_minus_out"].notna().all()
    assert set(tertiles["control_mean_tertile"]) == {1, 2, 3}
    for screen in ("RPE1", "K562-essential"):
        single = differences[
            (differences["screen"] == screen)
            & (differences["construct_set"] == "all eligible constructs")
            & differences["object"].isin(objects)
        ]
        family = primary[primary["screen"] == screen].iloc[0]
        assert family["in_minus_out"] == pytest.approx(single["in_minus_out"].mean())
        assert family["in_minus_out_lo"] <= family["in_minus_out"] <= family["in_minus_out_hi"]

    reduced = rows[(rows["construct_set"] == "without VCC training targets") & (rows["subset"] == "all")]
    assert (reduced["n_rows"] == 240 - 6).all()  # E1, E2 and E4 have two constructs each
    fewer = rows[(rows["construct_set"] == "without VCC training or validation targets") & (rows["subset"] == "all")]
    assert (fewer["n_rows"] == 240 - 8).all()
    row = overlap[(overlap["screen"] == "RPE1") & (overlap["subset"] == "all")].iloc[0]
    assert row["vcc_training_target_constructs"] == 6 and row["vcc_validation_target_constructs"] == 2
    assert row["not_a_vcc_target_constructs"] == 240 - 8
    assert set(composition["subset"]) == {"all", "target_in_features", "target_not_in_features"}
    assert set(partials["statistic"]) == {"partial_spearman_given_control_mean"}


def test_family_mean_skips_replicates_with_an_undefined_member():
    names = ["a", "b"]
    point = np.array([[0, 0, 0, 0.2], [0, 0, 0, 0.4]])
    draws = np.zeros((4, 2, 4))
    draws[:, 0, 3] = [0.1, 0.2, 0.3, np.nan]
    draws[:, 1, 3] = [0.3, 0.4, 0.5, 0.6]
    out = d1b._family_mean(names, point, draws, ["a", "b"])
    assert out["estimate"] == pytest.approx(0.3)
    assert out["n_skipped"] == 1


def test_verdict_names_the_position_of_an_interval_relative_to_zero():
    assert d1b._verdict(0.1, 0.3) == "positive"
    assert d1b._verdict(-0.3, -0.1) == "negative"
    assert d1b._verdict(-0.1, 0.1) == "inconclusive"
    assert d1b._verdict(float("nan"), 0.1) == "undefined"


def test_part_c_stops_when_the_masked_count_differs_from_d1(monkeypatch):
    monkeypatch.setattr(d1b, "d1_changed_constructs", lambda name: 59)
    arrays = _per_screen(3)
    matrix = np.column_stack([arrays[f"{s}_max_abs" + ("_target_zeroed" if z else "")].to_numpy()
                              for s, z in d1b.MAX_FAMILIES.values()])
    families = {t: {"group": "predictor without a model", "columns": [i], "n_seeds": 1, "split_meaning": "baseline"}
                for i, t in enumerate(d1b.MAX_FAMILIES)}
    names, (_, point, draws) = d1b._joint(matrix, arrays, families, np.ones(len(arrays), dtype=bool), b=20)
    with pytest.raises(RuntimeError, match="D1 counts"):
        d1b.masking_contrast("RPE1", arrays, matrix, families, names, point, draws, b=20)


def test_largest_coordinate_flags_separate_unique_targets_from_ties():
    block = np.array([[5.0, 1.0, 2.0], [3.0, -3.0, 0.0], [1.0, 4.0, 0.5], [0.0, 0.0, 9.0]])
    feature = np.array([0, 1, 0, 2])
    selected = np.array([True, True, True, False])
    flags = d1b.largest_coordinate_flags(block, feature, selected)
    assert flags["unique_largest"].tolist() == [True, False, False, False]
    assert flags["ties_largest"].tolist() == [False, True, False, False]
    assert flags["changed"].tolist() == [True, False, False, False]
    assert flags["max_abs_target_zeroed"].tolist() == [2.0, 3.0, 4.0, 9.0]


def test_seed_convention_names_every_kind_of_row():
    assert d1b.seed_convention({"group": "archived object, first run", "n_seeds": 3}).startswith("mean over")
    assert d1b.seed_convention({"group": "archived object, first run", "n_seeds": 1}) == "single archived run"
    assert d1b.seed_convention({"group": "archived object, single seed", "n_seeds": 1}) == "single archived seed"
    assert "averaged" in d1b.seed_convention({"group": "archived encoder arm or control, seed-averaged", "n_seeds": 3})
    assert d1b.seed_convention({"group": "trivial baseline", "n_seeds": 1}) == "no model"


def test_max_response_by_argmax_reports_ties_separately():
    arrays = _per_screen(4)
    selected = arrays["target_is_selected_feature"].to_numpy()
    ties = selected & (np.arange(len(arrays)) % 3 == 0)
    arrays["cell_target_ties_largest"] = ties
    arrays["cell_target_unique_largest"] = selected & ~ties
    table = d1b.max_response_by_argmax({"RPE1": arrays}, b=30)
    cell = table[table["input"] == "cell-level delta"]
    assert set(cell["subset"]) == {
        "target_in_features, argmax is the target coordinate alone",
        "target_in_features, the target ties another coordinate at the largest",
        "target_in_features, argmax is another coordinate",
    }
    assert cell["n_rows"].sum() == selected.sum()


def test_vcc_control_cells_follow_the_recorded_counts(tmp_path, monkeypatch):
    path = tmp_path / "sizes.json"
    path.write_text(json.dumps({"nt_per_batch": {"b0": 707, "b1": 800}}))
    monkeypatch.setattr(d1b, "subsample_sizes_file", lambda: path)
    table = pd.DataFrame({"batch": ["b1", "b0", "b1"]})
    assert d1b.vcc_control_cells(table).tolist() == [800.0, 707.0, 800.0]
    with pytest.raises(RuntimeError):
        d1b.vcc_control_cells(pd.DataFrame({"batch": ["b2"]}))


def test_cells_against_pool_lists_every_difference():
    screen = SimpleNamespace(constructs=np.array(["a", "b", "c"]),
                             endpoint=pd.DataFrame({"n_perturbed_pool": [30, 40, 50]}))
    report = d1b._cells_against_pool(screen, np.array([30, 41, 50]), "test")
    assert report["constructs_that_differ"] == 1
    assert report["first_differences"] == [{"construct": "b", "cells": 41, "n_perturbed_pool": 40}]


def _write_h5(path, obs=None, var=None, matrix=None):
    h5py = pytest.importorskip("h5py")

    with h5py.File(path, "w") as handle:
        for name, table in (("obs", obs), ("var", var)):
            if table is None:
                continue
            group = handle.create_group(name)
            group.attrs["_index"] = "_index"
            for key, values in table.items():
                group.create_dataset(key, data=np.array([str(v).encode() for v in values], dtype="S"))
        if matrix is not None:
            handle.create_dataset("X", data=np.asarray(matrix, dtype=np.float32))


def test_release_inputs_map_by_first_symbol_and_find_reverse_cases(tmp_path, monkeypatch):
    symbols = np.array(["S1", "S2", "S3"])
    monkeypatch.setattr(d1b.features, "hvg_symbols", lambda: symbols)
    monkeypatch.setattr(d1b.config, "portable", lambda path: "portable/" + Path(path).name)
    path = tmp_path / "bulk.h5ad"
    # genes: S1 (E1), S2 (E2), S2 again (E9, not chosen), X (E3); construct c3 targets E2 under the symbol "ALIAS"
    matrix = np.array([[12.0, 0.5, 0.1, 3.0], [-1.0, 4.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0], [0.3, -2.0, 7.0, 1.0]])
    _write_h5(path, obs={"_index": ["c1", "c2", "non-targeting_1", "c3"]},
              var={"_index": ["E1", "E2", "E9", "E3"], "gene_name": ["S1", "S2", "S2", "X"]}, matrix=matrix)
    monkeypatch.setattr(d1b, "release_bulk_file", lambda name: path)
    frame, check = d1b.release_inputs("RPE1", np.array(["c1", "c2", "c3"]), np.array(["S1", "S2", "ALIAS"]),
                                      np.array(["E1", "E2", "E2"]))
    assert check["release_rows_without_controls"] == 3 and check["selected_genes_mapped"] == 2
    assert frame["release_own_delta_zero_filled"].tolist() == [10.0, 4.0, 0.0]  # clipped at 10
    assert frame["release_max_abs_at_clip"].tolist() == [True, False, False]
    assert frame["release_target_unique_largest"].tolist() == [True, True, False]
    assert check["constructs_outside_the_features_with_the_target_mapped_under_another_symbol"] == 1
    assert check["reverse_cases"][0]["construct"] == "c3"
    assert frame["release_target_mapped_under_another_symbol"].tolist() == [False, False, True]
    assert check["target_column_with_another_ensembl_identifier"] == 0


def test_cell_level_reverse_check_reads_the_annotation_only(tmp_path, monkeypatch):
    monkeypatch.setattr(d1b.features, "hvg_symbols", lambda: np.array(["S1", "S2"]))
    monkeypatch.setattr(d1b.config, "portable", lambda path: "portable/" + Path(path).name)
    path = tmp_path / "cells.h5ad"
    _write_h5(path, var={"_index": ["E1", "E2", "E7"], "gene_name": ["S1", "S2", "S7"]})
    monkeypatch.setattr(d1b, "single_cell_file", lambda name: path)
    flag, report = d1b.cell_level_reverse_check("RPE1", np.array(["a", "b"]), np.array(["S1", "OTHER"]),
                                                np.array(["E1", "E2"]), np.array([True, False]))
    assert flag.tolist() == [False, True]
    assert report["features_mapped"] == 2
    assert report["reverse_cases"][0]["mapped_as"] == "S2"


def test_archived_reproduction_compares_with_the_stored_values(tmp_path, monkeypatch):
    arrays = _per_screen(5)
    y = arrays["y_primary"].to_numpy()
    rng = np.random.default_rng(9)
    prediction = y + rng.normal(scale=1.0, size=y.size)
    from scipy import stats as sps

    rho = float(sps.spearmanr(prediction, y).statistic)
    max_rho = float(sps.spearmanr(arrays["cell_max_abs"].to_numpy(), y).statistic)
    primary = {"screens": {k: {"arm_rho": {"z": {"rho": rho}}} for k in ("RPE1", "K562")}}
    secondary = {"screens": {k: {"rho": {"MAX_LINEAR": {"rho": max_rho}}} for k in ("RPE1", "K562")}}
    (tmp_path / "phase4_primary.json").write_text(json.dumps(primary))
    (tmp_path / "phase4_secondary.json").write_text(json.dumps(secondary))
    monkeypatch.setattr(d1b, "phase4_file", lambda kind: tmp_path / f"phase4_{kind}.json")
    monkeypatch.setattr(d1b.config, "portable", lambda path: "portable/" + Path(path).name)
    monkeypatch.setattr(d1b, "external_objects", lambda name, a: (prediction[:, None], {"z arm": {"arm": "z", "columns": [0]}}))
    report = d1b.check_archived_reproduction({"RPE1": arrays, "K562-essential": arrays})
    assert report["passed"] and report["encoder_max_abs_difference"] == 0.0
    secondary["screens"]["K562"]["rho"]["MAX_LINEAR"]["rho"] = max_rho + 0.01
    (tmp_path / "phase4_secondary.json").write_text(json.dumps(secondary))
    with pytest.raises(RuntimeError, match="max-response"):
        d1b.check_archived_reproduction({"RPE1": arrays, "K562-essential": arrays})


def test_inventory_records_the_d1_summary_that_gates_the_masking_contrast():
    import inspect

    source = inspect.getsource(d1b.inventory)
    assert "D1_EXTERNAL_SUMMARY" in source
    assert "D1_EXTERNAL_SUMMARY" in inspect.getsource(d1b.d1_changed_constructs)
