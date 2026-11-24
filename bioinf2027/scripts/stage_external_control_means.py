"""Stage the two stored external control-mean arrays under the data root and record their provenance.

E1a's external baseline check (bioinf2027/plans/E1_design.md, Section 7, own-coordinate summaries)
compares the E1a external B1 with the stored control means named there,
bioinf2027/cache/mu_nt_rpe1.npy and bioinf2027/cache/mu_nt_k562_essential.npy. Both lie in the
git-ignored repository cache. This script copies them, byte for byte, to
$IGA_STORAGE_ROOT/bioinf2027/cache/external_control_means/, leaves the repository copies in place, and
checks each copy:

- RPE1: its digest must equal the one bioinf2027/diag/D1b_rpe1_control_mean_checks.json records for
  the array that bioinf2027/scripts/make_d1b_rpe1_control_mean.py wrote and verified.
- K562-essential: the array must equal nt_sum / nt_count of the retained accumulator state
  ($IGA_STORAGE_ROOT/IGA_GENEFORMER_EXT/k562_essential/checkpoints/xcell_state.pkl), the
  quantity D1b recomputes in every run, and the state's control count must equal the endpoint's.

Output: the staged arrays and the committed record bioinf2027/endpoints/external_control_means.json.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "bioinf2027" / "cache"
TARGET = config.storage_root() / "bioinf2027" / "cache" / "external_control_means"
RECORD = REPO / "bioinf2027" / "endpoints" / "external_control_means.json"
ARRAYS = {"RPE1": "mu_nt_rpe1.npy", "K562-essential": "mu_nt_k562_essential.npy"}
RPE1_CHECKS = REPO / "bioinf2027" / "diag" / "D1b_rpe1_control_mean_checks.json"
K562_STATE = ("IGA_GENEFORMER_EXT", "k562_essential/checkpoints/xcell_state.pkl")


def stage(name: str) -> dict:
    source = SOURCE / name
    target = TARGET / name
    digest = sha256(source)
    if target.exists():
        if sha256(target) != digest:
            raise RuntimeError("%s exists under the data root with other contents" % name)
    else:
        TARGET.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if sha256(target) != digest:
        raise RuntimeError("the staged copy of %s differs from its source" % name)
    return {"source": file_record(source), "staged": file_record(target)}


def check_rpe1(record: dict) -> dict:
    recorded = json.loads(RPE1_CHECKS.read_text())["control_mean_array"]["sha256"]
    passed = record["staged"]["sha256"] == recorded
    if not passed:
        raise RuntimeError("the RPE1 control-mean array does not carry the digest D1b recorded")
    return {"check": "digest equals the one recorded by the verified repeated pass", "recorded_in": file_record(RPE1_CHECKS),
            "recorded_sha256": recorded, "passed": passed}


def check_k562(record: dict) -> dict:
    state_path = config.storage_root() / K562_STATE[0] / K562_STATE[1]
    with open(state_path, "rb") as handle:
        state = pickle.load(handle)
    mu_state = np.asarray(state["nt_sum"], dtype=np.float64) / float(state["nt_count"])
    mu_array = np.load(TARGET / ARRAYS["K562-essential"])
    endpoint_pool = int(features.load_screen("K562-essential").endpoint["n_nt_pool"].iloc[0])
    difference = float(np.max(np.abs(mu_array.astype(np.float64) - mu_state))) if mu_array.shape == mu_state.shape else float("inf")
    out = {"check": "array equals nt_sum / nt_count of the retained accumulator state, and the state's control count equals the endpoint's",
           "state": file_record(state_path), "state_nt_count": int(state["nt_count"]), "endpoint_control_pool": endpoint_pool,
           "shape": list(mu_array.shape), "max_abs_difference": difference}
    out["passed"] = bool(difference == 0.0 and out["state_nt_count"] == endpoint_pool)
    if not out["passed"]:
        raise RuntimeError("the K562-essential control-mean array does not equal the accumulator state's mean")
    return out


def main() -> None:
    staged = {screen: stage(name) for screen, name in ARRAYS.items()}
    staged["RPE1"]["provenance"] = check_rpe1(staged["RPE1"])
    staged["K562-essential"]["provenance"] = check_k562(staged["K562-essential"])
    record = {"purpose": "inputs of the E1a external baseline check, staged under the data root",
              "plan": "bioinf2027/plans/E1_design.md, Section 7; bioinf2027/reviews/E1a_audit_r2.md, hygiene findings",
              "arrays": staged}
    RECORD.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[stage] both arrays staged and checked", flush=True)


if __name__ == "__main__":
    main()
