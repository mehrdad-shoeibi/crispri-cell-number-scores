"""E4, gate G3 option (a): where the builder's all-cell external z differs from the stored z.

Read-only diagnostic, as bioinf2027/GATE.md (commit bb53562), Section 2, describes and the author chose
(bioinf2027/plans/E4.md, Section 14). It loads the inputs exactly as the main stage of
bioinf2027/scripts/make_e4_features.py does and calls its functions unchanged: per external construct,
the builder's z from every perturbed cell of the construct and every control cell
(`screen_features`, as in its check) against the stored z (`stored_features`). It changes no cell set
and fits nothing.

Outputs:
- bioinf2027/diag/E4_external_z.csv, one row per construct: screen, construct, embedding run, number of
  perturbed cells, largest absolute difference, the norm of the difference over the norm of the stored
  z, the largest absolute difference after removing the shift shared by the construct's embedding run,
  and the largest ratio of the difference to the between-construct SD of the stored z per coordinate.
- bioinf2027/diag/E4_external_z.json: per screen and embedding run, the distribution of these
  per-construct values; the shift shared by the run's constructs (median of the difference over them,
  per coordinate), which is the control mean's share: a difference in the control mean shifts every
  construct of a run alike, and the stored control mean is not among the recorded inputs, so its share is
  estimated this way and named as an estimate; the per-coordinate SD of the stored z between constructs;
  and, as a consistency check, the largest differences of x and z, which must equal the failed check's.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import e1inputs  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

import make_e4_features as feat  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "bioinf2027" / "diag"
RECORD = REPO / "bioinf2027" / "results" / "E4" / "features.json"
QUANTILES = (0.0, 0.5, 0.9, 0.99, 1.0)


def load() -> dict:
    """The main stage's loading, step for step (make_e4_features.main up to stored_features)."""
    counts, manifest = feat.builder.load_counts()
    feat.STORE["counts"] = counts
    feat.builder.prepare(counts)
    fc = feat.load_feature_counts()
    feat.STORE["features"] = fc["arrays"]
    for screen in feat.SCREENS:
        for key in ("pert_rows", "nt_rows"):
            if not np.array_equal(fc["arrays"]["screen_" + screen][key], counts["screen_" + screen][key]):
                raise RuntimeError("the E4 extraction and the label cache list different %s cells" % screen)
    feat.STORE["emb"] = feat.embedding_rows()
    feat.stored_features()
    return {"label_cache": manifest["manifest"], "e4_extraction": fc["manifest"],
            "d2_cell_lists": file_record(REPO / "bioinf2027" / "results" / "D2" / "units.json")}


def quantiles(values: np.ndarray) -> dict:
    return {f"q{q:g}": float(np.quantile(values, q)) for q in QUANTILES}


def embedding_run(screen: str, constructs: np.ndarray) -> np.ndarray:
    index = pd.read_parquet(e1inputs.paths()[f"{screen}_z_index"])
    if "source" not in index.columns:
        return np.array(["one run"] * constructs.size)
    source = dict(zip(index["construct"].astype(str), index["source"].astype(str)))
    return np.array([source[c] for c in constructs.tolist()])


def screen_diagnostic(screen: str) -> tuple[pd.DataFrame, dict]:
    arrays = feat.builder.DATA["screens"][screen]
    constructs = arrays["table"]["construct"].astype(str).to_numpy()
    stored_x, stored_z = feat.STORE["stored_external"][screen]
    controls = np.arange(feat.STORE["counts"]["screen_" + screen]["nt_rows"].size)  # as in all_cell_work
    built_x = np.empty_like(stored_x)
    built_z = np.empty_like(stored_z)
    for k, pool in enumerate(arrays["pools"]):
        built_x[k], built_z[k] = feat.screen_features(screen, pool, controls)
    difference = built_z - stored_z
    runs = embedding_run(screen, constructs)
    sd = stored_z.std(axis=0, ddof=1)
    shift = np.zeros_like(difference)
    per_run = {}
    control_mean = feat.STORE["emb"][f"{screen}_nt"][controls].astype(np.float64).mean(axis=0)
    for run in sorted(set(runs.tolist())):
        members = runs == run
        shared = np.median(difference[members], axis=0)
        shift[members] = shared
        residual = np.abs(difference[members] - shared).max(axis=1)
        largest = np.abs(difference[members]).max(axis=1)
        relative = np.linalg.norm(difference[members], axis=1) / np.linalg.norm(stored_z[members], axis=1)
        per_run[run] = {
            "constructs": int(members.sum()),
            "largest_absolute_difference": quantiles(largest),
            "relative_difference_norm": quantiles(relative),
            "largest_absolute_difference_after_the_shared_shift": quantiles(residual),
            "shared_shift": {"norm": float(np.linalg.norm(shared)), "largest_absolute_coordinate": float(np.abs(shared).max()),
                             "norm_over_the_builder_control_mean_norm": float(np.linalg.norm(shared) / np.linalg.norm(control_mean)),
                             "largest_ratio_to_the_between_construct_sd": float((np.abs(shared) / sd).max()),
                             "share_of_the_squared_difference": float((shared ** 2).sum() * members.sum() / (difference[members] ** 2).sum())},
        }
    frame = pd.DataFrame({
        "screen": screen, "construct": constructs, "embedding_run": runs,
        "n_perturbed_cells": [len(p) for p in arrays["pools"]],
        "largest_absolute_difference": np.abs(difference).max(axis=1),
        "relative_difference_norm": np.linalg.norm(difference, axis=1) / np.linalg.norm(stored_z, axis=1),
        "largest_absolute_difference_after_the_run_shift": np.abs(difference - shift).max(axis=1),
        "largest_ratio_to_the_between_construct_sd": (np.abs(difference) / sd[None, :]).max(axis=1),
    })
    summary = {
        "constructs": int(constructs.size),
        "control_cells": int(controls.size),
        "consistency_with_the_failed_check": {"x_max_abs_difference": float(np.abs(built_x - stored_x).max()),
                                              "z_max_abs_difference": float(np.abs(difference).max())},
        "between_construct_sd_of_the_stored_z": quantiles(sd),
        "stored_z_norm": quantiles(np.linalg.norm(stored_z, axis=1)),
        "by_embedding_run": per_run,
        "relation_with_the_number_of_perturbed_cells": {
            "spearman_of_largest_difference_with_cells": float(pd.Series(frame["largest_absolute_difference"]).corr(frame["n_perturbed_cells"], method="spearman")),
            "spearman_after_the_run_shift": float(pd.Series(frame["largest_absolute_difference_after_the_run_shift"]).corr(frame["n_perturbed_cells"], method="spearman"))},
    }
    return frame, summary


def main() -> None:
    started = time.time()
    inputs = load()
    recorded = json.loads(RECORD.read_text())["builder_checks"]
    frames, report = [], {"plan": "bioinf2027/GATE.md (bb53562), Section 2; bioinf2027/plans/E4.md, Section 14",
                          "inputs": {**inputs, "failed_check": file_record(RECORD)},
                          "control_mean_note": ("the stored control mean is not among the recorded inputs; its share is estimated as the shift "
                                                "shared by the constructs of one embedding run (median of the difference over them, per coordinate)"),
                          "screens": {}}
    for screen in feat.SCREENS:
        frame, summary = screen_diagnostic(screen)
        summary["consistency_with_the_failed_check"]["recorded_z_max_abs_difference"] = recorded[screen]["z_max_abs_difference"]
        summary["consistency_with_the_failed_check"]["recorded_x_max_abs_difference"] = recorded[screen]["x_max_abs_difference"]
        frames.append(frame)
        report["screens"][screen] = summary
        print("[e4 z] %s: largest z difference %.3g (recorded %.3g); %s" % (
            screen, summary["consistency_with_the_failed_check"]["z_max_abs_difference"], recorded[screen]["z_max_abs_difference"],
            {r: (v["constructs"], v["shared_shift"]["largest_absolute_coordinate"], v["largest_absolute_difference_after_the_shared_shift"]["q1"])
             for r, v in summary["by_embedding_run"].items()}), flush=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(DIAG / "E4_external_z.csv", index=False)
    report["outputs"] = {"per_construct": file_record(DIAG / "E4_external_z.csv")}
    report["seconds"] = round(time.time() - started, 1)
    (DIAG / "E4_external_z.json").write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
    print("[e4 z] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
