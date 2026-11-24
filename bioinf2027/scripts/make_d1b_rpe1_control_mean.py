"""D1b, RPE1: the control mean recomputed from the retained single-cell file.

The recorded RPE1 expression pass kept only a state written after 40 of its 42
chunks, so the control mean behind the persisted delta matrix cannot be read
from a retained artifact. This script repeats that pass on the single-cell
file and accepts its control mean only if three checks pass:

  stored index       the stored control index array equals the control rows
                     derived from the cell annotations, both by the rule of the
                     recorded pass and by the rule of the endpoint builder, and
                     the pass has the recorded numbers of cells, chunks and
                     constructs,
  checkpoint prefix  after 40 chunks every running sum and count equals the
                     retained mid-pass state, and the two hold the same constructs,
  identity           every recomputed construct delta equals the persisted
                     delta matrix, and the control pool has the recorded size.

The design is fixed in bioinf2027/plans/D1b.md, Section 7.2. A checks file is
written to bioinf2027/diag/ on every run, including a run that fails or stops
with an error. The control mean is written to the local cache only when all
three checks pass, before the checks file, and the checks file records its
digest so that it cannot be confused with an array from another run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, singlecell  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "diag"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
CHECKS_FILE = OUT_DIR / "D1b_rpe1_control_mean_checks.json"
ARRAY_FILE = CACHE_DIR / "mu_nt_rpe1.npy"

CONTROL_LABEL = "non-targeting"
CHUNK = 5000
SNAPSHOT_CHUNKS = 40
TOLERANCE = 1e-9  # absolute, on means and deltas, and the absolute part on running sums
SUM_RELATIVE_TOLERANCE = 1e-12  # relative part on running sums of order 1e4 to 1e5
READ_DTYPE = np.float32  # the recorded pass cast every chunk to single precision on read

FIRST_RUN_RECORD = "bioinf2027/diag/D1b_provenance.json at commit bd78b5e"
RECORDED = {
    "control_cells": {
        "value": 11485,
        "source": "earlier_revision/experiment_specs/phase2_rpe1_topup_and_xcell.md:55-56; "
        "$IGA_STORAGE_ROOT/IGA_GENEFORMER_EXT/logs/phase2_embed_xcell.json, acc_trace.nt_count",
    },
    "pass_cells": {
        "value": 208606,
        "source": "earlier_revision/experiment_specs/phase2_rpe1_topup_and_xcell.md:55-56; "
        "$IGA_STORAGE_ROOT/IGA_GENEFORMER_EXT/logs/phase2_embed_xcell.json, acc_trace.total_cells",
    },
    "pass_chunks": {
        "value": 42,
        "source": "$IGA_STORAGE_ROOT/IGA_GENEFORMER_EXT/logs/phase2_embed_xcell.json, chunks; "
        "earlier_revision/experiment_specs/phase2_rpe1_topup_and_xcell.md:72",
    },
    "constructs": {
        "value": 1932,
        "source": "earlier_revision/experiment_specs/phase2_rpe1_topup_and_xcell.md:24",
    },
    "snapshot_chunks": {
        "value": 40,
        "source": "earlier_revision/experiment_specs/phase2_rpe1_topup_and_xcell.md:72-73; "
        + FIRST_RUN_RECORD
        + ", state_updates",
    },
    "snapshot_cells": {
        "value": 200000,
        "source": FIRST_RUN_RECORD + ", state_total_cells",
    },
    "snapshot_control_cells": {
        "value": 11007,
        "source": FIRST_RUN_RECORD
        + ", state_nt_count; earlier_revision/experiment_specs/phase4_prep.md:490",
    },
}
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def recorded(name: str) -> int:
    return int(RECORDED[name]["value"])


def input_paths() -> dict[str, Path]:
    root = config.storage_root()
    encoder = root / "IGA_GENEFORMER_EXT"
    replogle = root / "IGA_MAIN" / "data" / "replogle"
    return {
        "single_cell_counts": replogle / "raw_singlecell" / "rpe1_raw_singlecell_01.h5ad",
        "stored_control_index": encoder / "scratch" / "nt_oidx.npy",
        "construct_list": root
        / "IGA_MAIN"
        / "results_v4"
        / "transfer_pretest_lock_v1"
        / "arm1_rpe1_construct_list.csv",
        "persisted_delta_matrix": encoder / "x_cell" / "x_cell_rpe1_native.npz",
        "mid_pass_state": encoder / "checkpoints" / "rpe1_xcell" / "xcell_state.pkl",
        "recorded_manifest": encoder / "manifests" / "phase2_rpe1_manifest.json",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value):
    """The same structure with every non-finite number written as null."""
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_report(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CHECKS_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
    os.replace(temporary, CHECKS_FILE)


def quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {}
    points = np.quantile(values, QUANTILES)
    return {f"q{int(q * 100):03d}": float(v) for q, v in zip(QUANTILES, points)}


def manifest_comparison(manifest: dict, digest: str) -> str:
    """Equal, different, or key absent: the construct list against the recorded manifest."""
    recorded_digest = manifest.get("input_arm_list_sha256")
    if recorded_digest is None:
        return "key absent"
    return "equal" if recorded_digest == digest else "different"


def pass_definition(
    gene: np.ndarray, transcript: np.ndarray, constructs: np.ndarray, stored_control: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rows and group codes of the pass, and the stored-index check.

    Group 0 is the control set, group ``i + 1`` is the i-th construct in sorted
    order. Control cells are the rows of the stored index array. A cell enters
    the pass when it is a control cell or carries one of the constructs; a
    control cell that also carries a listed construct counts as a control cell,
    as in the recorded pass.
    """
    n_cells = gene.shape[0]
    stored_control = np.asarray(stored_control, dtype=np.int64)
    in_range = bool(stored_control.size == 0 or (stored_control.min() >= 0 and stored_control.max() < n_cells))
    recorded_rule = np.flatnonzero(gene == CONTROL_LABEL)
    endpoint_rule = np.flatnonzero(np.char.find(np.char.lower(gene.astype(str)), CONTROL_LABEL) >= 0)

    check = {
        "stored_rows": int(stored_control.size),
        "stored_rows_in_range": in_range,
        "stored_rows_strictly_ascending": bool((np.diff(stored_control) > 0).all()),
        "rows_by_recorded_rule": int(recorded_rule.size),
        "rows_by_endpoint_rule": int(endpoint_rule.size),
        "stored_equals_recorded_rule": bool(np.array_equal(stored_control, recorded_rule)),
        "stored_equals_endpoint_rule": bool(np.array_equal(stored_control, endpoint_rule)),
        "recorded_rule_text": "gene annotation equal to 'non-targeting'",
        "endpoint_rule_text": "gene annotation containing 'non-targeting', ignoring case",
    }
    if not in_range:
        check.update({"passed": False, "reason": "a stored row lies outside the single-cell file"})
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), check

    is_control = np.zeros(n_cells, dtype=bool)
    is_control[stored_control] = True
    position = {name: index for index, name in enumerate(constructs.tolist())}
    construct_code = np.array([position.get(name, -1) for name in transcript.tolist()], dtype=np.int64)
    rows = np.flatnonzero(is_control | (construct_code >= 0))
    codes = np.where(is_control[rows], 0, construct_code[rows] + 1)
    n_chunks = -(-rows.size // CHUNK)

    check.update(
        {
            "control_cells_also_carrying_a_listed_construct": int(
                (is_control & (construct_code >= 0)).sum()
            ),
            "pass_cells": int(rows.size),
            "pass_chunks": int(n_chunks),
            "constructs_listed": int(constructs.size),
            "constructs_listed_unique": bool(len(set(constructs.tolist())) == constructs.size),
            "constructs_with_cells": int(np.unique(codes[codes > 0]).size),
        }
    )
    check["passed"] = bool(
        check["stored_rows_strictly_ascending"]
        and check["stored_equals_recorded_rule"]
        and check["stored_equals_endpoint_rule"]
        and check["stored_rows"] == recorded("control_cells")
        and check["pass_cells"] == recorded("pass_cells")
        and check["pass_chunks"] == recorded("pass_chunks")
        and check["constructs_listed"] == recorded("constructs")
        and check["constructs_listed_unique"]
        and check["constructs_with_cells"] == recorded("constructs")
    )
    return rows, codes, check


def checkpoint_prefix_check(
    snapshot: singlecell.GroupSums | None, state: dict, constructs: np.ndarray
) -> dict:
    """Compare the state after 40 chunks with the retained mid-pass state.

    Every difference is collected and the largest is taken with numpy, so a
    single non-finite difference makes the check fail instead of being skipped.
    """
    if snapshot is None:
        return {"passed": False, "reason": "the repeated pass did not reach the snapshot chunk"}
    position = {name: index + 1 for index, name in enumerate(constructs.tolist())}
    recorded_counts = {str(k): int(v) for k, v in state["count"].items()}
    recorded_sums = {str(k): v for k, v in state["sum"].items()}
    seen_here = {constructs[g - 1] for g in (np.flatnonzero(snapshot.counts[1:] > 0) + 1)}

    unknown = sorted(name for name in recorded_counts if name not in position)
    without_sum = sorted(name for name in recorded_counts if name not in recorded_sums)
    count_mismatch = sorted(
        name
        for name, count in recorded_counts.items()
        if name in position and count != int(snapshot.counts[position[name]])
    )
    def compare(recorded_sum, repeated_sum):
        """Largest absolute difference, largest relative one, and the excess over the tolerance."""
        recorded_sum = np.asarray(recorded_sum, dtype=np.float64)
        if recorded_sum.shape != repeated_sum.shape:
            return np.inf, np.inf, np.inf
        gap = np.abs(recorded_sum - repeated_sum)
        allowed = TOLERANCE + SUM_RELATIVE_TOLERANCE * np.abs(recorded_sum)
        scale = np.maximum(np.abs(recorded_sum), 1.0)
        return gap.max(), (gap / scale).max(), (gap - allowed).max()

    results = [compare(state["nt_sum"], snapshot.sums[0])]
    for name in recorded_counts:
        group = position.get(name)
        if group is None or name not in recorded_sums:
            continue
        results.append(compare(recorded_sums[name], snapshot.sums[group]))
    results = np.asarray(results, dtype=np.float64)
    worst = float(np.max(results[:, 0]))
    worst_relative = float(np.max(results[:, 1]))
    excess = float(np.max(results[:, 2]))

    check = {
        "recorded_updates": int(state["updates"]),
        "recorded_cells": int(state["total_cells"]),
        "recorded_control_cells": int(state["nt_count"]),
        "repeated_chunks": int(snapshot.n_chunks),
        "repeated_cells": int(snapshot.n_cells),
        "repeated_control_cells": int(snapshot.counts[0]),
        "constructs_in_recorded_state": len(recorded_counts),
        "constructs_seen_in_repeated_pass": len(seen_here),
        "construct_sets_equal": bool(set(recorded_counts) == seen_here),
        "constructs_unknown_to_the_list": unknown[:10],
        "constructs_with_a_count_but_no_sum": without_sum[:10],
        "constructs_with_a_different_count": count_mismatch[:10],
        "max_abs_difference_of_sums": worst,
        "max_relative_difference_of_sums": worst_relative,
        "tolerance": "absolute %g plus relative %g of the recorded sum, element by element" % (TOLERANCE, SUM_RELATIVE_TOLERANCE),
        "largest_excess_over_the_tolerance": excess,
    }
    check["passed"] = bool(
        check["recorded_updates"] == recorded("snapshot_chunks") == check["repeated_chunks"]
        and check["recorded_cells"] == recorded("snapshot_cells") == check["repeated_cells"]
        and check["recorded_control_cells"]
        == recorded("snapshot_control_cells")
        == check["repeated_control_cells"]
        and check["construct_sets_equal"]
        and not unknown
        and not without_sum
        and not count_mismatch
        and np.isfinite(excess)
        and excess <= 0.0
    )
    return check


def identity_check(
    state: singlecell.GroupSums,
    persisted_ids: np.ndarray,
    persisted: np.ndarray,
    constructs: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """Recomputed construct deltas against the persisted delta matrix."""
    check = {
        "control_cells": int(state.counts[0]),
        "pass_cells": int(state.n_cells),
        "pass_chunks": int(state.n_chunks),
        "n_genes": int(state.sums.shape[1]),
        "persisted_shape": list(persisted.shape),
        "constructs_compared": int(persisted.shape[0]) if persisted.ndim == 2 else 0,
        "construct_ids_equal_and_in_the_same_order": bool(np.array_equal(persisted_ids, constructs)),
        "constructs_without_cells": int((state.counts[1:] <= 0).sum()),
        "identity_max_abs_diff": float("nan"),
        "identity_tolerance": TOLERANCE,
    }
    if state.counts[1:].size:
        check["perturbed_cells_per_construct"] = quantiles(state.counts[1:])
    shape_ok = persisted.ndim == 2 and persisted.shape == (constructs.size, state.sums.shape[1])
    check["persisted_shape_matches"] = bool(shape_ok)
    if state.counts[0] <= 0:
        check.update({"passed": False, "reason": "no control cell was accumulated"})
        return None, check
    control_mean = state.sums[0] / state.counts[0]
    if shape_ok and check["construct_ids_equal_and_in_the_same_order"] and check["constructs_without_cells"] == 0:
        means = state.sums[1:] / state.counts[1:, None]
        check["identity_max_abs_diff"] = float(np.max(np.abs(means - control_mean - persisted)))
    check["passed"] = bool(
        shape_ok
        and check["construct_ids_equal_and_in_the_same_order"]
        and check["constructs_without_cells"] == 0
        and check["control_cells"] == recorded("control_cells")
        and np.isfinite(check["identity_max_abs_diff"])
        and check["identity_max_abs_diff"] <= TOLERANCE
    )
    return control_mean, check


def save_control_mean(control_mean: np.ndarray) -> dict:
    """Write the array through a temporary name and return its digest."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARRAY_FILE.with_name(ARRAY_FILE.stem + ".tmp.npy")
    np.save(temporary, np.asarray(control_mean, dtype=np.float64))
    os.replace(temporary, ARRAY_FILE)
    return {
        "path": config.portable(ARRAY_FILE),
        "sha256": sha256(ARRAY_FILE),
        "shape": [int(n) for n in control_mean.shape],
        "dtype": "float64",
        "definition": "control sum / control count over the 11,485 control cells, the plain mean "
        "of the recorded pass, log1p(counts / cell total * 10,000) per cell",
    }


def run(report: dict) -> int:
    import h5py

    paths = input_paths()
    constructs = np.array(sorted(pd.read_csv(paths["construct_list"])["construct"].astype(str)))
    stored_control = np.load(paths["stored_control_index"]).astype(np.int64)
    manifest = json.loads(paths["recorded_manifest"].read_text())

    list_digest = sha256(paths["construct_list"])
    report["input_digests"] = {
        "construct_list_sha256": list_digest,
        "construct_list_against_recorded_manifest": manifest_comparison(manifest, list_digest),
        "stored_control_index_sha256": sha256(paths["stored_control_index"]),
        "persisted_delta_matrix_sha256": sha256(paths["persisted_delta_matrix"]),
        "mid_pass_state_sha256": sha256(paths["mid_pass_state"]),
        "recorded_manifest_sha256": sha256(paths["recorded_manifest"]),
    }

    print("reading the cell annotations", flush=True)
    with h5py.File(paths["single_cell_counts"], "r") as handle:
        gene = singlecell.read_annotation(handle["obs"], "gene")
        transcript = singlecell.read_annotation(handle["obs"], "gene_transcript")
        matrix = handle["X"]
        report["single_cell_matrix"] = {
            "cells": int(matrix.shape[0]),
            "genes": int(matrix.shape[1]),
            "stored_dtype": str(matrix.dtype),
            "read_dtype": np.dtype(READ_DTYPE).name,
        }

        rows, codes, stored = pass_definition(gene, transcript, constructs, stored_control)
        report["checks"]["stored_index"] = stored
        print("check, stored index:", "PASS" if stored["passed"] else "FAIL", flush=True)
        if not stored["passed"]:
            report["all_passed"] = False
            write_report(report)
            print("stopping: the pass definition does not match the recorded pass", flush=True)
            return 1

        print("repeating the pass over %d cells" % rows.size, flush=True)
        state, snapshot = singlecell.accumulate_group_sums(
            matrix,
            rows,
            codes,
            n_groups=constructs.size + 1,
            chunk=CHUNK,
            snapshot_after=recorded("snapshot_chunks"),
            read_dtype=READ_DTYPE,
        )

    with open(paths["mid_pass_state"], "rb") as handle:
        mid_state = pickle.load(handle)
    prefix = checkpoint_prefix_check(snapshot, mid_state, constructs)
    del mid_state
    if snapshot is not None:
        prefix["perturbed_cells_per_construct_at_the_snapshot"] = quantiles(
            snapshot.counts[1:][snapshot.counts[1:] > 0]
        )
    report["checks"]["checkpoint_prefix"] = prefix
    print(
        "check, checkpoint prefix: %s (max abs difference of sums %s)"
        % ("PASS" if prefix["passed"] else "FAIL", prefix.get("max_abs_difference_of_sums")),
        flush=True,
    )

    store = np.load(paths["persisted_delta_matrix"], allow_pickle=True)
    control_mean, identity = identity_check(
        state, np.asarray(store["ids"]).astype(str), np.asarray(store["x"]), constructs
    )
    report["checks"]["identity"] = identity
    print(
        "check, identity: %s (max abs difference %s)"
        % ("PASS" if identity["passed"] else "FAIL", identity["identity_max_abs_diff"]),
        flush=True,
    )

    report["all_passed"] = bool(stored["passed"] and prefix["passed"] and identity["passed"])
    print("hashing the single-cell file", flush=True)
    report["input_digests"]["single_cell_counts_sha256"] = sha256(paths["single_cell_counts"])

    if report["all_passed"]:
        report["control_mean_array"] = save_control_mean(control_mean)
    write_report(report)
    if not report["all_passed"]:
        print("stopping: a check failed, the control mean is not written", flush=True)
        return 1
    print("all three checks passed; control mean written to the local cache", flush=True)
    return 0


def main() -> int:
    report = {
        "plan": "bioinf2027/plans/D1b.md, Section 7.2",
        "paths_relative_to": "$" + config.STORAGE_VARIABLE,
        "inputs": {name: config.portable(path) for name, path in input_paths().items()},
        "recorded_values": RECORDED,
        "chunk": CHUNK,
        "normalisation": "log1p(counts / cell total * 10000) in double precision, total over all "
        "genes, after a single-precision cast on read as in the recorded pass",
        "checks": {},
        "all_passed": False,
    }
    try:
        return run(report)
    except Exception as error:  # the checks file must describe every run, a failed one too
        report["all_passed"] = False
        report["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        write_report(report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
