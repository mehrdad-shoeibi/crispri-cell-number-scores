"""E2, Stack's set-order sensitivity (bioinf2027/plans/E2.md, Sections 3 and 17).

The per-row features rebuilt with the order-2 embeddings ($IGA_STORAGE_ROOT/bioinf2027/E2/embeddings/stack_order2/units/,
the 96 units holding VCC G6 test rows or K562-essential cells); every cell outside those units keeps its order-1 vector,
so each batch's control mean is defined. On the VCC test rows and on the K562-essential constructs: per dimension, the
Pearson correlation across rows of the feature under the two orders; one minus the median over the dimensions,
separately for the two sets; every Stack value is flagged "depends on the set order" if either exceeds 0.10.

Check first: the order-1 features rebuilt here equal the saved Stack features (features.json, stack.npz) to 1e-5 of
each set's largest absolute value. The unit files are streamed (sums per row, construct and batch), never held whole.
Output: bioinf2027/results/E2/order_sensitivity.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import e2cells  # noqa: E402
import make_e2_features as e2feat  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "order_sensitivity.json"
PLAN = "bioinf2027/plans/E2.md, Sections 3 and 17"
THRESHOLD = 0.10
TOLERANCE = 1e-5
SCREEN = "K562-essential"
FLAG = "depends on the set order"


class Stop(RuntimeError):
    """A check failed: nothing is written."""


def slots(P: dict, rows) -> dict:
    """For each part used, the accumulator slot of each of its cells (-1: not used)."""
    test = np.flatnonzero((rows["split"] == "test").to_numpy())
    vcc_pert = np.full(P["vcc_perturbed_test"]["rows"].size, -1)
    for j, k in enumerate(test):
        idx = np.asarray(rows.iloc[k]["part_index"])
        if (vcc_pert[idx] >= 0).any():
            raise Stop("a VCC test cell belongs to two rows")
        vcc_pert[idx] = j
    batches = sorted(set(P["vcc_control"]["group"].tolist()))
    batch_id = {b: i for i, b in enumerate(batches)}
    vcc_ctrl = np.array([batch_id[b] for b in P["vcc_control"]["group"].tolist()])
    cache = e2cells._cache(f"screen_{SCREEN}")
    table = features.read_endpoint(config.screen_paths()[SCREEN]["endpoint"])
    order = np.argsort(cache["pert_code"], kind="stable")
    bounds = np.searchsorted(cache["pert_code"][order], np.arange(cache["constructs"].size + 1))
    by_construct = {c: i for i, c in enumerate(cache["constructs"].tolist())}
    part_rows = P[f"{SCREEN}_perturbed"]["rows"]
    k_pert = np.full(part_rows.size, -1)
    for t, construct in enumerate(table["construct"].astype(str)):
        i = by_construct[construct]
        cells = cache["pert_rows"][order[bounds[i]: bounds[i + 1]]]
        if cells.size:
            idx = np.searchsorted(part_rows, cells)
            if not np.array_equal(part_rows[idx], cells) or (k_pert[idx] >= 0).any():
                raise Stop("a K562-essential cell is missing from its part or belongs to two constructs")
            k_pert[idx] = t
    k_ctrl = np.zeros(P[f"{SCREEN}_control"]["rows"].size, dtype=int)
    return {"vcc_perturbed_test": (vcc_pert, test.size), "vcc_control": (vcc_ctrl, len(batches)),
            f"{SCREEN}_perturbed": (k_pert, len(table)), f"{SCREEN}_control": (k_ctrl, 1),
            "_test_rows": test, "_batch_of_row": np.array([batch_id[b] for b in rows.iloc[test]["batch"].tolist()])}


class Sums:
    def __init__(self, S: dict, width: int) -> None:
        parts = {p: v[1] for p, v in S.items() if not p.startswith("_")}
        self.sum = {p: np.zeros((n, width)) for p, n in parts.items()}
        self.count = {p: np.zeros(n, dtype=np.int64) for p, n in parts.items()}

    def add(self, part: str, slot: np.ndarray, vectors: np.ndarray) -> None:
        keep = slot >= 0
        np.add.at(self.sum[part], slot[keep], vectors[keep].astype(np.float64))
        np.add.at(self.count[part], slot[keep], 1)


def stream(folder: Path, digests: dict):
    for path in sorted(folder.glob("*.npz")):
        digests[f"{folder.parent.name}/{path.name}"] = sha256(path)
        with np.load(path, allow_pickle=False) as z:
            yield z["part"], z["index"], z["vectors"]


def arms(sums: Sums, S: dict) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        pert = sums.sum["vcc_perturbed_test"] / sums.count["vcc_perturbed_test"][:, None]
        ctrl = sums.sum["vcc_control"] / sums.count["vcc_control"][:, None]
        vcc = pert - ctrl[S["_batch_of_row"]]
        k = sums.sum[f"{SCREEN}_perturbed"] / sums.count[f"{SCREEN}_perturbed"][:, None]
        k = k - sums.sum[f"{SCREEN}_control"][0] / sums.count[f"{SCREEN}_control"][0]
    return {"VCC test rows": vcc, f"{SCREEN} constructs": k}


def correlation_per_dimension(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[keep] - a[keep].mean(axis=0), b[keep] - b[keep].mean(axis=0)
    denominator = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(axis=0) / denominator, np.nan)


def main() -> None:
    started = time.time()
    P = e2cells.parts()
    rows, _ = e2feat.row_table(P)
    S = slots(P, rows)
    root = config.storage_root() / "bioinf2027" / "E2" / "embeddings"
    width = None
    digests: dict = {}
    covered = {p: np.zeros(S[p][0].size, dtype=bool) for p in S if not p.startswith("_")}
    order2 = None
    for part, index, vectors in stream(root / "stack_order2" / "units", digests):
        if order2 is None:
            width = vectors.shape[1]
            order2, order1 = Sums(S, width), Sums(S, width)
        for p in np.unique(part):
            if p not in covered:
                continue
            sel = part == p
            if covered[p][index[sel]].any() or np.unique(index[sel]).size != int(sel.sum()):
                raise Stop("an order-2 cell of %s appears twice" % p)
            covered[p][index[sel]] = True
            order2.add(p, S[p][0][index[sel]], vectors[sel])
    seen = {p: np.zeros(S[p][0].size, dtype=bool) for p in covered}
    for part, index, vectors in stream(root / "stack" / "units", digests):
        for p in np.unique(part):
            if p not in covered:
                continue
            sel = part == p
            seen[p][index[sel]] = True
            order1.add(p, S[p][0][index[sel]], vectors[sel])
            outside = sel.copy()
            outside[sel] = ~covered[p][index[sel]]
            order2.add(p, S[p][0][index[outside]], vectors[outside])
    for p in covered:
        if not seen[p].all():
            raise Stop("order 1 does not cover every cell of %s" % p)
        if not np.array_equal(order1.count[p], order2.count[p]):
            raise Stop("the two orders count other cells for %s" % p)
    a1, a2 = arms(order1, S), arms(order2, S)
    saved_record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "features.json").read_text())["outputs"]["stack"]
    saved_path = config.storage_root() / "bioinf2027" / "E2" / "features" / "stack.npz"
    if sha256(saved_path) != saved_record["sha256"]:
        raise Stop("stack.npz does not match features.json")
    with np.load(saved_path, allow_pickle=False) as z:
        saved = {"VCC test rows": z["vcc"][S["_test_rows"]].astype(np.float64), f"{SCREEN} constructs": z[f"{SCREEN}_arm"].astype(np.float64)}
    check = {}
    for name in a1:
        both = np.isfinite(a1[name]).all(axis=1)
        if not np.array_equal(both, np.isfinite(saved[name]).all(axis=1)):
            raise Stop("order 1 and the saved features differ in which %s are defined" % name)
        scale = float(np.abs(saved[name][both]).max())
        diff = float(np.abs(a1[name][both] - saved[name][both]).max())
        check[name] = {"rows": int(both.sum()), "largest_absolute_difference": diff, "scale": scale, "passed": diff <= TOLERANCE * scale}
        if not check[name]["passed"]:
            raise Stop("order 1 does not reproduce the saved Stack features: %s" % json.dumps(check))
    results = {}
    for name in a1:
        r = correlation_per_dimension(a1[name], a2[name])
        finite = r[np.isfinite(r)]
        changed = np.isfinite(a2[name]).all(axis=1) & (np.abs(a2[name] - a1[name]).max(axis=1) > 0)
        results[name] = {"rows": int(np.isfinite(a1[name]).all(axis=1).sum()), "rows_whose_feature_changed": int(changed.sum()),
                         "dimensions": int(r.size), "dimensions_with_a_correlation": int(finite.size),
                         "median_correlation": float(np.median(finite)), "one_minus_median_correlation": float(1.0 - np.median(finite)),
                         "correlation_quantiles": {q: float(np.quantile(finite, float(q))) for q in ("0.0", "0.05", "0.25", "0.5", "0.75", "1.0")},
                         "exceeds_threshold": bool(1.0 - np.median(finite) > THRESHOLD)}
    flagged = any(v["exceeds_threshold"] for v in results.values())
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(digests.items())).encode()).hexdigest()
    record = {"plan": PLAN, "threshold": THRESHOLD, "results": results, "flag": FLAG if flagged else "", "flagged": flagged,
              "order_1_check": {"tolerance_relative_to_scale": TOLERANCE, **check},
              "cells_in_order_2_units": {p: int(c.sum()) for p, c in covered.items()},
              "inputs": {"unit_files": len(digests), "unit_files_sha256_of_sorted_list": combined, "stack_features": file_record(saved_path),
                         "features_json": file_record(REPO / "bioinf2027" / "results" / "E2" / "features.json")},
              "unit_file_sha256": digests,
              "code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (sys.modules[__name__], e2feat, e2cells)},
              "seconds": round(time.time() - started, 1)}
    OUT.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 order] %s; flagged: %s; %.0f s" % ("; ".join(f"{k}: 1 - median r = {v['one_minus_median_correlation']:.4f}" for k, v in results.items()),
                                                 flagged, time.time() - started), flush=True)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: make_e2_order_sensitivity.py")
    main()
