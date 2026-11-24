"""E1b, the checks that run before any E1b fit is read.

Design: bioinf2027/plans/E1_design.md, Section 10 and the sections that amend it: 21.1 item 4 and
23.5 item 4 (the harness reproduces the submitted values), 21.3 item 2 (check f), 25.2 item 2, 26.1
item 3 and 27 item 2 (the external x arrays), 26.2 item 3 (the support of the heads' inputs) and
Section 10, dependence and coverage (the timing probe).

1. External x, both screens: the persisted mapped arrays against the mapping of the native delta,
   to 1e-6, with the control-cell count each rests on. The mapping of the native delta (the full
   pass) is saved as E1b's external x; the harness keeps the submitted arrays.
2. Check (f), on every VCC row whose target is a feature: unmodified x at the target coordinate
   against the batch-local own delta of the label pass (G5, G6), re-referenced x against -B2 native
   (G1, G2; G3 and G4 at the gene level), to 1e-4. A failure stops E1b.
3. Harness: the submitted analysis rebuilt with this pipeline (the per-arm heads of
   earlier_revision/phase4/phase4_primary.json, the existing label, all modelled training rows,
   batch-local features, the existing 27-cell endpoints, the submitted external arrays), every head
   on one thread. Every stored arm value on both screens and every stored VCC test value must be
   reproduced to 0.005 (Spearman, and R-squared on VCC test). A failure stops E1b (gate G3).
4. Support of the heads' inputs, per cell: VCC training range and quartiles against each screen for
   B0, the four magnitude summaries, the norms of x and z and the target coordinate of x.
5. Timing probe: one forest fit per representation in G5 and G6 and of every other kind in G6; it
   fixes the number of refits (50, or 20 if 50 project more than 48 hours of CPU). Its predictions
   are neither kept nor scored.

Outputs: bioinf2027/results/E1/checks.json, harness.csv, check_f.csv, support.csv; the full-pass
external x and the harness predictions under $IGA_STORAGE_ROOT/bioinf2027/E1b/.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, e1cells, e1inputs, features, heads, representations as rep  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E1"
PLAN = "bioinf2027/plans/E1_design.md, Sections 10, 21.1, 21.3, 23.5, 25.2, 26 and 27"
SPLITS = e1inputs.SPLITS
SCREENS = e1inputs.SCREENS
WORKERS = 14
X_TOLERANCE = 1e-6
F_TOLERANCE = 1e-4
HARNESS_TOLERANCE = 0.005
SUBMITTED_ARMS = ("z", "zm", "C1", "C2", "C3", "C4", "C5")
SEEDS = (0, 1, 2)
REFITS = 50
REFITS_CUT = 20
REFIT_LIMIT_HOURS = 48.0
SUPPORT_CELLS = ("G5", "G6", "G1", "G2", "G3", "G4")
CHECK_F_CELLS = ("G5", "G6", "G1", "G2", "G3", "G4")

DATA: dict = {}
SMOKE = False


def e1b_root() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E1b_checks_smoke"
    return config.storage_root() / "bioinf2027" / "E1b"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(sps.spearmanr(a, b).statistic)


# --------------------------------------------------------------------------- 1. external x


def external_x_check() -> dict:
    records = json.loads((REPO / "bioinf2027" / "diag" / "D1b_rpe1_control_mean_checks.json").read_text())
    k562 = json.loads((REPO / "bioinf2027" / "endpoints" / "external_control_means.json").read_text())
    controls = {
        "RPE1": {"control_cells": int(records["checks"]["identity"]["control_cells"]), "source": "bioinf2027/diag/D1b_rpe1_control_mean_checks.json, checks.identity: the native delta reproduces from all pass cells"},
        "K562-essential": {"control_cells": int(k562["arrays"]["K562-essential"]["provenance"]["state_nt_count"]), "source": "bioinf2027/endpoints/external_control_means.json: the accumulator state behind the native delta"},
    }
    out = {}
    folder = e1b_root() / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    for screen in SCREENS:
        endpoint = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        inputs = e1inputs.load_screen_inputs(screen, endpoint)
        DATA[("screen", screen)] = (endpoint, inputs)
        full = inputs.x["full_pass"]
        entry = {"constructs": int(full.shape[0]), "features_mapped": inputs.notes["features_mapped"], "native_genes": inputs.notes["native_genes"],
                 "native_delta_control_cells": controls[screen]}
        comparisons = {}
        for name in [k for k in inputs.x if k not in ("native", "full_pass")]:
            diff = np.abs(inputs.x[name].astype(np.float64) - full.astype(np.float64))
            comparisons[name] = {"max_abs_difference": float(diff.max()), "constructs_above_tolerance": int((diff.max(axis=1) > X_TOLERANCE).sum()),
                                 "equal_to_the_full_pass_mapping": bool(diff.max() <= X_TOLERANCE)}
        if "mapped_with_ids" in inputs.x:
            diff = np.abs(inputs.x["mapped_with_ids"].astype(np.float64) - inputs.x["submitted"].astype(np.float64))
            comparisons["mapped_with_ids_against_submitted"] = {"max_abs_difference": float(diff.max()), "constructs_above_tolerance": int((diff.max(axis=1) > X_TOLERANCE).sum())}
        entry["against_the_full_pass_mapping"] = comparisons
        path = folder / f"x_{screen}_full_pass_mapped.npy"
        np.save(path, full)
        entry["full_pass_mapping"] = file_record(path)
        if screen == "RPE1":
            entry["e1b_uses"] = "the full-pass mapping (Section 27, item 2)"
            entry["harness_uses"] = "the submitted array"
            entry["passed"] = True
        else:
            passed = comparisons["submitted"]["equal_to_the_full_pass_mapping"]
            entry["e1b_uses"] = "the submitted array, equal to the full-pass mapping" if passed else "none: the check failed and E1b stops for the author (Section 25.2, item 2)"
            entry["harness_uses"] = "the submitted array"
            entry["passed"] = bool(passed)
        out[screen] = entry
        print("[x] %s: %s" % (screen, json.dumps({k: v["max_abs_difference"] for k, v in comparisons.items()})), flush=True)
    out["tolerance"] = X_TOLERANCE
    return out


def e1b_external_x(screen: str) -> np.ndarray:
    _, inputs = DATA[("screen", screen)]
    return inputs.x["full_pass"] if screen == "RPE1" else inputs.x["submitted"]


# --------------------------------------------------------------------------- 2. check (f)


def check_f() -> tuple[dict, list[dict]]:
    rows_out = []
    summary = {}
    for cell in CHECK_F_CELLS:
        data = DATA[("cell", cell)]
        mask = data.target_column >= 0
        idx = np.flatnonzero(mask)
        x_target = data.x[idx, data.target_column[idx]].astype(np.float64)
        own = -data.vcc["B2"].to_numpy(dtype=np.float64)[idx]
        diff = np.abs(x_target - own)
        reference = "unmodified x against the batch-local own delta" if data.spec["features"] == "batch" else (
            "re-referenced x against -B2 native" if data.spec["unit"] == "row" else "gene-level re-referenced x against -B2 native")
        for split in SPLITS:
            in_split = (data.vcc["split"].to_numpy()[idx] == split)
            rows_out.append({"cell": cell, "split": split, "comparison": reference, "rows": int(in_split.sum()),
                             "max_abs_difference": float(diff[in_split].max()) if in_split.any() else float("nan"),
                             "rows_above_tolerance": int((diff[in_split] > F_TOLERANCE).sum())})
        summary[cell] = {"comparison": reference, "rows": int(idx.size), "max_abs_difference": float(diff.max()), "rows_above_tolerance": int((diff > F_TOLERANCE).sum()),
                         "passed": bool((diff <= F_TOLERANCE).all())}
        print("[f] %s %s max %.3g, above %d" % (cell, reference, diff.max(), (diff > F_TOLERANCE).sum()), flush=True)
    summary["tolerance"] = F_TOLERANCE
    summary["passed"] = bool(all(v["passed"] for k, v in summary.items() if isinstance(v, dict)))
    return summary, rows_out


# --------------------------------------------------------------------------- 3. harness


def submitted_blocks() -> dict:
    """Features of the submitted analysis (earlier_revision/pipeline/run_phase1_select.py:24-38)."""
    rows = DATA["rows"]
    order = np.argsort(rows.shard_position, kind="stable")
    split_of = rows.keys["split"].to_numpy()[order]
    by = {s: order[split_of == s] for s in SPLITS}
    x = {s: rows.x[by[s]] for s in SPLITS}
    z = {s: rows.z[by[s]] for s in SPLITS}
    y = {s: rows.existing[by[s]].astype(np.float32) for s in SPLITS}
    z_scaler = rep.BlockScaler(z["train"])
    m_stats = rep.fit_magnitude(x["train"])
    c1_scaler = rep.BlockScaler(rep.c1_features(len(y["train"]), "train"))
    projection = rep.c2_projection()
    c2_scaler = rep.BlockScaler(rep.c2_features(x["train"], projection))

    def assemble(zs, xs, m, split):
        n = zs.shape[0]
        return {
            "z": zs,
            "zm": np.concatenate([zs, m], axis=1),
            "C1": c1_scaler(rep.c1_features(n, split)),
            "C2": c2_scaler(rep.c2_features(xs, projection)),
            "C3": zs[rep.c3_permutation(n, split)],
            "C4": np.concatenate([zs, m[rep.c4_permutation(n, split)]], axis=1).astype(np.float32),
            "C5": np.concatenate([zs, rep.c5_gaussian(n, split)], axis=1).astype(np.float32),
        }

    blocks = {}
    for split in SPLITS:
        zs = z_scaler(z[split])
        blocks[split] = assemble(zs, x[split], m_stats.transform(rep.magnitude(x[split])), split)
    stored_m = {}
    for screen in SCREENS:
        _, inputs = DATA[("screen", screen)]
        zs = z_scaler(inputs.z)
        with np.load(e1inputs.paths()[f"{screen}_submitted_magnitude"]) as store:
            m_submitted = np.asarray(store["m_zscored"], dtype=np.float32)
        m_recomputed = m_stats.transform(rep.magnitude(inputs.x["submitted"]))
        stored_m[screen] = float(np.abs(m_submitted.astype(np.float64) - m_recomputed).max())
        blocks[screen] = assemble(zs, inputs.x["submitted"], m_submitted, screen)
        m_full = m_stats.transform(rep.magnitude(inputs.x["full_pass"]))
        blocks[screen + " full-pass x"] = assemble(zs, inputs.x["full_pass"], m_full, screen)
    return {"blocks": blocks, "y": y, "magnitude_recomputation_max_abs_difference": stored_m,
            "near_constant_z_coordinates": z_scaler.near_constant(), "test_order": by["test"]}


def harness_fit(arm: str, seed: int) -> dict:
    harness = DATA["harness"]
    blocks, y = harness["blocks"], harness["y"]
    head = heads.forest(seed) if DATA["submitted_heads"][arm] == "RF" else heads.boosting(seed)
    start = time.time()
    with heads.single_thread():
        head.fit(blocks["train"][arm], y["train"])
        out = {key: np.asarray(head.predict(block[arm]), dtype=np.float64) for key, block in blocks.items() if key != "train"}
    out["seconds"] = time.time() - start
    folder = e1b_root() / "predictions" / "harness"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{arm}_seed{seed}.npz"
    np.savez(path, **{k.replace(" ", "_"): v for k, v in out.items() if k != "seconds"})
    out["record"] = file_record(path)
    return {"arm": arm, "seed": seed, **out}


def harness_summary(results: list[dict]) -> tuple[dict, list[dict]]:
    stored = json.loads((REPO / "earlier_revision" / "phase4" / "phase4_primary.json").read_text())
    test_table = json.loads((REPO / "earlier_revision" / "phase1" / "test_table.json").read_text())
    predictions = np.load(REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz", allow_pickle=True)
    stored_test_y = np.load(REPO / "earlier_revision" / "phase1" / "preds" / "y_test.npy")
    y_test = DATA["harness"]["y"]["test"]
    rows = []
    by = {(r["arm"], r["seed"]): r for r in results}
    endpoints = {s: DATA[("screen", s)][0] for s in SCREENS}
    for arm in SUBMITTED_ARMS:
        per_seed = [by[(arm, s)] for s in SEEDS]
        r2 = [heads.pooled_r2(y_test, r["test"]) for r in per_seed]
        rho = [spearman(y_test, r["test"]) for r in per_seed]
        stored_preds = np.load(REPO / "earlier_revision" / "phase1" / "preds" / f"test_{arm}.npy")
        rows.append({"arm": arm, "scope": "VCC test", "head": DATA["submitted_heads"][arm], "quantity": "R-squared, mean over seeds",
                     "stored": test_table[arm]["r2_mean"], "reproduced": float(np.mean(r2)), "difference": float(np.mean(r2) - test_table[arm]["r2_mean"]),
                     "prediction_max_abs_difference": float(max(np.abs(r["test"] - stored_preds[i]).max() for i, r in enumerate(per_seed)))})
        rows.append({"arm": arm, "scope": "VCC test", "head": DATA["submitted_heads"][arm], "quantity": "Spearman, mean over seeds",
                     "stored": test_table[arm]["spearman_mean"], "reproduced": float(np.mean(rho)), "difference": float(np.mean(rho) - test_table[arm]["spearman_mean"])})
        for screen, short in (("RPE1", "RPE1"), ("K562-essential", "K562")):
            mean_prediction = np.mean([r[screen] for r in per_seed], axis=0)
            constructs = np.asarray(predictions[f"{short}_constructs"]).astype(str)
            if not np.array_equal(constructs, endpoints[screen]["construct"].astype(str).to_numpy()):
                raise RuntimeError("the stored %s predictions are not in construct order" % screen)
            column = list(np.asarray(predictions["arms"]).astype(str)).index(arm)
            stored_prediction = np.asarray(predictions[f"{short}_P"])[:, column]
            value = spearman(mean_prediction, endpoints[screen]["y_primary"].to_numpy())
            rows.append({"arm": arm, "scope": screen, "head": DATA["submitted_heads"][arm], "quantity": "external Spearman of the seed-mean prediction",
                         "stored": stored["screens"][short]["arm_rho"][arm]["rho"], "reproduced": value,
                         "difference": value - stored["screens"][short]["arm_rho"][arm]["rho"],
                         "prediction_max_abs_difference": float(np.abs(mean_prediction - stored_prediction).max())})
    for row in rows:
        row["passed"] = bool(abs(row["difference"]) <= HARNESS_TOLERANCE)
    summary = {
        "tolerance": HARNESS_TOLERANCE,
        "vcc_test_label_equals_the_stored_label": bool(np.array_equal(y_test, stored_test_y)),
        "external_magnitude_recomputed_from_the_submitted_x_max_abs_difference": DATA["harness"]["magnitude_recomputation_max_abs_difference"],
        "near_constant_z_coordinates": DATA["harness"]["near_constant_z_coordinates"],
        "heads": DATA["submitted_heads"],
        "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "forest_n_jobs": 1,
        "seconds_per_fit": {f"{r['arm']}_seed{r['seed']}": round(r["seconds"], 1) for r in results},
        "prediction_files": {f"{r['arm']}_seed{r['seed']}": r["record"] for r in results},
        "largest_absolute_difference": float(max(abs(r["difference"]) for r in rows)),
        "passed": bool(all(r["passed"] for r in rows)),
    }
    return summary, rows


# --------------------------------------------------------------------------- 4. support


def _quartiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "q25": float(np.percentile(values, 25)), "median": float(np.median(values)),
            "q75": float(np.percentile(values, 75)), "max": float(values.max())}


def support() -> list[dict]:
    out = []
    for cell in SUPPORT_CELLS:
        data = DATA[("cell", cell)]
        train = data.split("train")
        x_train = data.x[train]
        quantities = {
            "B0": (data.vcc["B0"].to_numpy()[train], None),
            "norm of x": (np.linalg.norm(x_train.astype(np.float64), axis=1), None),
            "norm of z": (np.linalg.norm(data.z[train].astype(np.float64), axis=1), None),
        }
        summaries = rep.magnitude(x_train)
        for j, name in enumerate(("mean |x|", "norm of x (summary)", "max |x|", "sd of x")):
            if j == 1:
                continue
            quantities[name] = (summaries[:, j], j)
        in_feature = (data.target_column >= 0) & train
        quantities["x at the target coordinate"] = (data.x[in_feature, data.target_column[in_feature]], "target")
        for screen in SCREENS:
            table = data.external[screen]
            _, inputs = DATA[("screen", screen)]
            x_ext = e1b_external_x(screen)
            external = {
                "B0": table["B0"].to_numpy(),
                "norm of x": np.linalg.norm(x_ext.astype(np.float64), axis=1),
                "norm of z": np.linalg.norm(inputs.z.astype(np.float64), axis=1),
            }
            ext_summaries = rep.magnitude(x_ext)
            for j, name in ((0, "mean |x|"), (2, "max |x|"), (3, "sd of x")):
                external[name] = ext_summaries[:, j]
            has = inputs.target_column >= 0
            external["x at the target coordinate"] = x_ext[has, inputs.target_column[has]]
            for name, (vcc_values, _) in quantities.items():
                vcc_q = _quartiles(vcc_values)
                ext_values = np.asarray(external[name], dtype=np.float64)
                ext_q = _quartiles(ext_values)
                out.append({
                    "cell": cell, "screen": screen, "quantity": name, "n_vcc_training": int(np.asarray(vcc_values).size), "n_screen": int(ext_values.size),
                    **{f"vcc_training_{k}": v for k, v in vcc_q.items()},
                    **{f"screen_{k}": v for k, v in ext_q.items()},
                    "screen_share_below_vcc_training_min": float((ext_values < vcc_q["min"]).mean()),
                    "screen_share_above_vcc_training_max": float((ext_values > vcc_q["max"]).mean()),
                })
    return out


# --------------------------------------------------------------------------- 5. timing probe


def probe_blocks(cell: str) -> dict:
    data = DATA[("cell", cell)]
    train = data.split("train")
    y = data.vcc["y"].to_numpy(dtype=np.float64)[train]
    x = data.x[train]
    m = rep.fit_magnitude(x).transform(rep.magnitude(x))
    b = data.vcc[["B0", "B1", "B2"]].to_numpy(dtype=np.float32)[train]
    blocks = {
        "z": rep.BlockScaler(data.z[train])(data.z[train]),
        "x": x,
        "m": m,
        "[x; m]": np.concatenate([x, m], axis=1),
        "C1": rep.c1_features(int(train.sum()), "train"),
    }
    if cell == "G6":
        blocks["C2"] = rep.c2_features(x, rep.c2_projection())
        blocks["C3"] = blocks["z"][rep.c3_permutation(int(train.sum()), "train")]
        blocks["TR"] = b
        blocks["[TR; x]"] = np.concatenate([b, x], axis=1)
    return {"y": y, "blocks": blocks}


def probe_fit(cell: str, name: str, head: str) -> dict:
    probe = DATA[("probe", cell)]
    model = heads.forest(0) if head == "forest" else heads.boosting(0)
    start = time.time()
    with heads.single_thread():
        model.fit(probe["blocks"][name], probe["y"])
    return {"cell": cell, "representation": name, "head": head, "rows": int(probe["y"].size), "columns": int(probe["blocks"][name].shape[1]),
            "seconds": time.time() - start}


def timing_summary(results: list[dict]) -> dict:
    seconds = {(r["cell"], r["representation"], r["head"]): r["seconds"] for r in results}
    refit_reps = ("z", "x", "m", "[x; m]", "C1")
    per_refit = {cell: sum(seconds[(cell, rep_name, "forest")] for rep_name in refit_reps) for cell in e1cells.PRIMARY}
    hours = REFITS * sum(per_refit.values()) / 3600.0
    return {
        "fits": [{**r, "seconds": round(r["seconds"], 1)} for r in results],
        "refit_representations": list(refit_reps),
        "seconds_per_refit_set": {k: round(v, 1) for k, v in per_refit.items()},
        "projected_cpu_hours_for_50_refits": round(hours, 1),
        "limit_hours": REFIT_LIMIT_HOURS,
        "refits": REFITS if hours <= REFIT_LIMIT_HOURS else REFITS_CUT,
        "rule": "Section 10, dependence and coverage: if 50 refits project more than 48 hours of CPU, their number is cut to 20",
    }


# --------------------------------------------------------------------------- main


def run_parallel(tasks: list[tuple]) -> list:
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = [pool.submit(function, *args) for function, *args in tasks]
        return [future.result() for future in futures]


def main() -> None:
    started = time.time()
    stored = json.loads((REPO / "earlier_revision" / "phase4" / "phase4_primary.json").read_text())
    DATA["submitted_heads"] = {arm: stored["heads"][arm] for arm in SUBMITTED_ARMS}
    x_check = external_x_check()
    DATA["rows"] = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    modelled = e1cells.load_tables("G2")["vcc"]
    for cell in sorted(set(SUPPORT_CELLS) | set(CHECK_F_CELLS)):
        DATA[("cell", cell)] = e1cells.build_cell(cell, DATA["rows"], means, modelled=modelled)
    f_summary, f_rows = check_f()
    DATA["harness"] = submitted_blocks()
    for cell in e1cells.PRIMARY:
        DATA[("probe", cell)] = probe_blocks(cell)
    tasks = [(harness_fit, arm, seed) for arm in SUBMITTED_ARMS for seed in SEEDS]
    probes = [(cell, name, "forest") for cell in e1cells.PRIMARY for name in DATA[("probe", cell)]["blocks"]]
    probes.append(("G6", "x", "boosting"))
    tasks += [(probe_fit, *p) for p in probes]
    print("[fits] %d harness fits and %d probe fits on %d workers" % (len(SUBMITTED_ARMS) * len(SEEDS), len(probes), WORKERS), flush=True)
    results = run_parallel(tasks)
    harness_results = [r for r in results if "arm" in r]
    probe_results = [r for r in results if "representation" in r]
    h_summary, h_rows = harness_summary(harness_results)
    timing = timing_summary(probe_results)
    support_rows = support()

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(h_rows).to_csv(OUT / "harness.csv", index=False)
    pd.DataFrame(f_rows).to_csv(OUT / "check_f.csv", index=False)
    pd.DataFrame(support_rows).to_csv(OUT / "support.csv", index=False)
    checks = {
        "plan": PLAN,
        "external_x": x_check,
        "check_f": f_summary,
        "harness": h_summary,
        "timing_probe": timing,
        "support": "bioinf2027/results/E1/support.csv",
        "all_passed": bool(x_check["RPE1"]["passed"] and x_check["K562-essential"]["passed"] and f_summary["passed"] and h_summary["passed"]),
        "seconds": round(time.time() - started, 1),
        "inputs": {
            "input_records": file_record(REPO / "bioinf2027" / "results" / "E1" / "input_records.json"),
            "cell_manifests": {cell: file_record(REPO / "bioinf2027" / "endpoints" / cell / "manifest.json") for cell in sorted(set(SUPPORT_CELLS) | set(CHECK_F_CELLS))},
            "submitted_values": file_record(REPO / "earlier_revision" / "phase4" / "phase4_primary.json"),
            "submitted_vcc_test_values": file_record(REPO / "earlier_revision" / "phase1" / "test_table.json"),
            "submitted_external_predictions": file_record(REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz"),
            "external_control_means_record": file_record(REPO / "bioinf2027" / "endpoints" / "external_control_means.json"),
            "rpe1_control_mean_checks": file_record(REPO / "bioinf2027" / "diag" / "D1b_rpe1_control_mean_checks.json"),
        },
    }
    (OUT / "checks.json").write_text(json.dumps(strict_json(checks), indent=1, allow_nan=False) + "\n")
    print("[checks] external x %s / %s, check (f) %s, harness %s (largest difference %.4f), refits %d, all passed %s" % (
        x_check["RPE1"]["passed"], x_check["K562-essential"]["passed"], f_summary["passed"], h_summary["passed"],
        h_summary["largest_absolute_difference"], timing["refits"], checks["all_passed"]), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        # Every stage on the real inputs with 5 trees and 20 boosting iterations; outputs go to
        # the scratch folder and are never reported.
        SMOKE = True
        heads.FOREST["n_estimators"] = 5
        heads.BOOSTING["max_iter"] = 20
        OUT = config.storage_root() / "bioinf2027" / "scratch" / "E1b_checks_smoke"
        PLAN = PLAN + "; smoke run, 5 trees, never reported"
    elif sys.argv[1:]:
        raise SystemExit("unknown arguments: %s" % sys.argv[1:])
    main()
