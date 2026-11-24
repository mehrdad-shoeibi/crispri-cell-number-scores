"""E1b, diagnostic of the harness failure: how were the external C1 and C3 features of the submitted analysis built?

The pre-run checks (bioinf2027/results/E1/checks.json, bioinf2027/results/E1/harness.csv) reproduce every
VCC test value of the seven submitted arms exactly, so the fitted boosting heads of C1 and C3 are the
submitted ones; their external values do not reproduce, so the external control features differ from
those the submitted analysis used, whose script was not retained. This diagnostic refits only those
heads (seeds 0 to 2, the submitted configuration, every head on one thread) and compares the
seed-mean external predictions under candidate constructions of the external C1 and C3 with the stored
predictions (earlier_revision/phase4/phase4_ext_predictions.npz). No E1b fit is read and no statistic
of E1b is computed; the candidates are fixed here, before any comparison.

Output: bioinf2027/results/E1/harness_controls_diagnostic.csv and .json.
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

from bioinf2027 import e1inputs, heads, representations as rep  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

import make_e1b_checks as checks  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E1"
SHORT = {"RPE1": "RPE1", "K562-essential": "K562"}
OFFSETS = ("train", "val", "test", "RPE1", "K562-essential")


def candidates(screen: str, z_ext: np.ndarray, harness: dict) -> dict:
    """Every candidate external block, named for how it is built."""
    rows = checks.DATA["rows"]
    order = np.argsort(rows.shard_position, kind="stable")
    train = order[rows.keys["split"].to_numpy()[order] == "train"]
    z_scaler = rep.BlockScaler(rows.z[train])
    c1_scaler = rep.BlockScaler(rep.c1_features(train.size, "train"))
    n = z_ext.shape[0]
    out = {}
    for offset in OFFSETS:
        raw = rep.c1_features(n, offset)
        out[("C1", f"Gaussian, stream {offset}, scaled with the VCC training scaler")] = c1_scaler(raw)
        out[("C1", f"Gaussian, stream {offset}, unscaled")] = raw
        out[("C1", f"Gaussian, stream {offset}, scaled with a scaler fitted on the screen")] = rep.BlockScaler(raw)(raw)
    for offset in OFFSETS:
        perm = rep.c3_permutation(n, offset)
        out[("C3", f"z scaled with the VCC training scaler, shuffled with stream {offset}")] = z_scaler(z_ext)[perm]
        out[("C3", f"z unscaled, shuffled with stream {offset}")] = z_ext[perm].astype(np.float32)
        out[("C3", f"z scaled with a scaler fitted on the screen, shuffled with stream {offset}")] = rep.BlockScaler(z_ext)(z_ext)[perm]
    out[("C3", "z scaled with the VCC training scaler, not shuffled")] = z_scaler(z_ext)
    return out


def main() -> None:
    stored = json.loads((REPO / "earlier_revision" / "phase4" / "phase4_primary.json").read_text())
    checks.DATA["submitted_heads"] = {arm: stored["heads"][arm] for arm in checks.SUBMITTED_ARMS}
    for screen in checks.SCREENS:
        endpoint = checks.features.read_endpoint(checks.config.screen_paths()[screen]["endpoint"])
        checks.DATA[("screen", screen)] = (endpoint, e1inputs.load_screen_inputs(screen, endpoint))
    checks.DATA["rows"] = e1inputs.load_vcc_rows()
    harness = checks.submitted_blocks()
    blocks, y = harness["blocks"], harness["y"]
    predictions = np.load(REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz", allow_pickle=True)
    arms = list(np.asarray(predictions["arms"]).astype(str))
    models = {}
    for arm in ("C1", "C3"):
        for seed in checks.SEEDS:
            model = heads.boosting(seed)
            with heads.single_thread():
                model.fit(blocks["train"][arm], y["train"])
            models[(arm, seed)] = model
    rows = []
    for screen in checks.SCREENS:
        endpoint, inputs = checks.DATA[("screen", screen)]
        stored_p = np.asarray(predictions[f"{SHORT[screen]}_P"])
        label = endpoint["y_primary"].to_numpy()
        for (arm, name), block in candidates(screen, inputs.z, harness).items():
            with heads.single_thread():
                mean_prediction = np.mean([models[(arm, s)].predict(block) for s in checks.SEEDS], axis=0)
            target = stored_p[:, arms.index(arm)]
            rows.append({"screen": screen, "arm": arm, "construction": name,
                         "max_abs_difference_from_the_stored_prediction": float(np.abs(mean_prediction - target).max()),
                         "external_spearman": checks.spearman(mean_prediction, label),
                         "stored_external_spearman": stored["screens"][SHORT[screen]]["arm_rho"][arm]["rho"]})
    frame = pd.DataFrame(rows).sort_values(["screen", "arm", "max_abs_difference_from_the_stored_prediction"])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "harness_controls_diagnostic.csv", index=False)
    best = frame.groupby(["screen", "arm"]).head(1)
    summary = {"purpose": "which construction of the external C1 and C3 reproduces the stored predictions",
               "candidates": int(len(frame)) // 4,
               "best": json.loads(best.to_json(orient="records")),
               "inputs": {"stored_predictions": file_record(REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz"),
                          "checks": file_record(OUT / "checks.json")}}
    (OUT / "harness_controls_diagnostic.json").write_text(json.dumps(strict_json(summary), indent=1, allow_nan=False) + "\n")
    print(best.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
