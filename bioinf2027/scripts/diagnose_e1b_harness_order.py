"""E1b, second diagnostic of the harness failure: were the submitted external C1 and C3 rows assigned in another construct order?

The first diagnostic (bioinf2027/scripts/diagnose_e1b_harness_controls.py) found no construction of the
external C1 and C3 that reproduces the stored predictions. Here the refitted submitted heads (identical
to the submitted ones: their VCC test predictions equal the stored ones exactly) predict
- C1: the Gaussian rows of the screen's stream, scaled with the VCC training scaler and unscaled;
- C3: the screen's z, scaled with the VCC training scaler, unshuffled;
and the sorted predictions are compared with the sorted stored predictions. Equal multisets mean that
the stored predictions are ours in another order; the order is then recovered by matching values and
compared with the construct orders of the submitted inputs. No E1b fit is read and no E1b statistic
is computed.

Output: bioinf2027/results/E1/harness_order_diagnostic.json.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1inputs, heads, representations as rep  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

import make_e1b_checks as checks  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E1" / "harness_order_diagnostic.json"
SHORT = {"RPE1": "RPE1", "K562-essential": "K562"}
MATCH = 1e-9


def construct_orders(screen: str, sorted_constructs: np.ndarray) -> dict:
    """Construct orders of the submitted inputs, as permutations of the sorted constructs."""
    paths = config.screen_paths()[screen]
    position = {c: i for i, c in enumerate(sorted_constructs.tolist())}
    orders = {}
    raw = pd.read_csv(paths["endpoint"])
    orders["endpoint file order"] = raw["construct"].astype(str).tolist()
    eligibility = pd.read_csv(paths["eligibility"])
    eligible = eligibility[eligibility["eligible"].astype(bool)]
    orders["eligibility file order"] = eligible["construct"].astype(str).tolist()
    index = pd.read_parquet(e1inputs.paths()[f"{screen}_z_index"])
    orders["embedding index order"] = index["construct"].astype(str).tolist()
    with np.load(e1inputs.paths()[f"{screen}_native"], allow_pickle=True) as store:
        orders["native delta order"] = [str(c) for c in store["ids"]]
    out = {}
    for name, order in orders.items():
        if sorted(order) == sorted_constructs.tolist():
            out[name] = np.array([position[c] for c in order])
    return out


def recover(stored: np.ndarray, ours: np.ndarray) -> dict:
    """Whether the stored values are a reordering of ours, and if so which reordering."""
    s, o = np.sort(stored), np.sort(ours)
    multiset = float(np.abs(s - o).max())
    out = {"sorted_max_abs_difference": multiset, "same_multiset": bool(multiset <= MATCH),
           "distinct_values": bool(np.unique(np.round(ours, 12)).size == ours.size)}
    if out["same_multiset"] and out["distinct_values"]:
        # stored[j] = ours[pi[j]]
        pi = np.empty(ours.size, dtype=np.int64)
        pi[np.argsort(stored, kind="stable")] = np.argsort(ours, kind="stable")
        out["permutation"] = pi
        out["identity"] = bool(np.array_equal(pi, np.arange(ours.size)))
    return out


def main() -> None:
    stored_values = json.loads((REPO / "earlier_revision" / "phase4" / "phase4_primary.json").read_text())
    checks.DATA["submitted_heads"] = {arm: stored_values["heads"][arm] for arm in checks.SUBMITTED_ARMS}
    for screen in checks.SCREENS:
        endpoint = checks.features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        checks.DATA[("screen", screen)] = (endpoint, e1inputs.load_screen_inputs(screen, endpoint))
    rows = e1inputs.load_vcc_rows()
    checks.DATA["rows"] = rows
    harness = checks.submitted_blocks()
    blocks, y = harness["blocks"], harness["y"]
    order = np.argsort(rows.shard_position, kind="stable")
    train = order[rows.keys["split"].to_numpy()[order] == "train"]
    z_scaler = rep.BlockScaler(rows.z[train])
    c1_scaler = rep.BlockScaler(rep.c1_features(train.size, "train"))
    models = {}
    for arm in ("C1", "C3"):
        for seed in checks.SEEDS:
            model = heads.boosting(seed)
            with heads.single_thread():
                model.fit(blocks["train"][arm], y["train"])
            models[(arm, seed)] = model
    predictions = np.load(REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz", allow_pickle=True)
    arms = list(np.asarray(predictions["arms"]).astype(str))
    out = {"purpose": "whether the submitted external C1 and C3 rows were assigned in another construct order", "screens": {}}
    for screen in checks.SCREENS:
        endpoint, inputs = checks.DATA[("screen", screen)]
        constructs = endpoint["construct"].astype(str).to_numpy()
        n = constructs.size
        stored = np.asarray(predictions[f"{SHORT[screen]}_P"])
        candidates = {
            ("C1", "screen stream, VCC training scaler"): c1_scaler(rep.c1_features(n, screen)),
            ("C1", "screen stream, unscaled"): rep.c1_features(n, screen),
            ("C3", "screen z, VCC training scaler, unshuffled"): z_scaler(inputs.z),
        }
        orders = construct_orders(screen, constructs)
        entry = {"construct_orders_available": sorted(orders)}
        for (arm, name), block in candidates.items():
            with heads.single_thread():
                ours = np.mean([models[(arm, s)].predict(block) for s in checks.SEEDS], axis=0)
            result = recover(stored[:, arms.index(arm)], ours)
            record = {k: v for k, v in result.items() if k != "permutation"}
            if "permutation" in result:
                pi = result["permutation"]
                matches = {}
                for order_name, positions in orders.items():
                    if arm == "C1":
                        # rows generated in that construct order: the construct at positions[i] got Gaussian row i
                        expected = np.empty(n, dtype=np.int64)
                        expected[positions] = np.arange(n)
                        matches[order_name] = bool(np.array_equal(pi, expected))
                    else:
                        for offset in ("train", "val", "test", "RPE1", "K562-essential"):
                            perm = rep.c3_permutation(n, offset)
                            # z rows taken in that order, then shuffled: construct at positions[i] got z of positions[perm[i]]
                            expected = np.empty(n, dtype=np.int64)
                            expected[positions] = positions[perm]
                            matches[f"{order_name}, stream {offset}"] = bool(np.array_equal(pi, expected))
                record["orders_reproducing_the_stored_assignment"] = sorted(k for k, v in matches.items() if v)
            entry[f"{arm}: {name}"] = record
        out["screens"][screen] = entry
    out["inputs"] = {"stored_predictions": file_record(REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz"),
                     "first_diagnostic": file_record(REPO / "bioinf2027" / "results" / "E1" / "harness_controls_diagnostic.json")}
    OUT.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print(json.dumps(strict_json(out["screens"]), indent=1), flush=True)


if __name__ == "__main__":
    main()
