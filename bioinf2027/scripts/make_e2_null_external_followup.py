"""E2 (d), follow-up analyses (1) to (3), post hoc, from saved files (bioinf2027/plans/E2_null_external_followup.md,
Section 2; gate G7, item 2, the author's answer of 2026-11-17, bioinf2027/GATE.md). Analysis (4) is
make_e2_null_external.py --label existing.

On E1b's external constructs of each Replogle screen, against the cell's external label y (G5 and G6), with E1b's joint
external draws (target-gene cluster bootstrap, 10,000 replicates, seeds 20260901 and 20260902):
(1) rho(n_t, y), n_t the construct's eligible perturbed cells as (d) counts them (its features.npz), with the reading of
    the plan note against (d)'s T (null_external.csv);
(2) rho(prediction of the N0x head of (d), n_t), mean over head seeds inside each replicate, and the spread of the
    predictions (standard deviation over constructs, and its ratio to that of the same fit's VCC test predictions);
(3a) rho(prediction of e, n_t); (3b) the partial Spearman of e's prediction with y given n_t (partial correlation of
    the three weighted ranks) and rho(e, y) minus it, paired; (3c) rho(e, y) inside each fifth of n_t and their mean,
    descriptive. e is E2's encoder on the rebuilt label (E1b's z for Geneformer), forest, head seeds 0 to 2.
Outputs: bioinf2027/results/E2/null_external_followup/saved.csv and saved.json. Every value is post hoc and
unadjusted and not part of E2's family; Stack's values are marked; no value is read as an explanation of why the
external values are negative (gate G7, item 6).

With --label existing (the plan note, Section 5; the stop of 2026-11-17 06:50, item 4): (2), (3a) and (3b) for the
heads trained on VCC's all-cells label, the null's from (4) and e's saved forest predictions, without (1)'s rule and
(3c); outputs under null_external_followup/existing_label/. In both modes rho(e, y) must equal E2's external.csv.
--out-scratch NAME writes the outputs to $IGA_STORAGE_ROOT/bioinf2027/scratch/NAME instead (the check that the default run is
unchanged).
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, resample  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e2_null_external as ne  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E2_null_external_followup.md"
OUT = REPO / "bioinf2027" / "results" / "E2" / "null_external_followup"
LABEL = "rebuilt"
LABELS = ("rebuilt", "existing")
FIFTHS = (20, 40, 60, 80)
STATUS_BY_LABEL = {"rebuilt": "post hoc, unadjusted, not part of E2's family (gate G7, item 2; %s)" % PLAN,
                   "existing": "post hoc, unadjusted, not part of E2's family (the stop of 2026-11-17 06:50, item 4; %s, Section 5)" % PLAN}
STATUS = STATUS_BY_LABEL[LABEL]
EXTERNAL_CSV = REPO / "bioinf2027" / "results" / "E2" / "stats" / "external.csv"
TOLERANCE = 1e-12
WORKERS = 4
OUT_OVERRIDE: Path | None = None
DATA: dict = {}


def out_dir() -> Path:
    """The outputs' folder: saved.csv of (1) to (3) for the rebuilt label, existing_label/ for the all-cells label."""
    if OUT_OVERRIDE is not None:
        return OUT_OVERRIDE
    return OUT if LABEL == "rebuilt" else OUT / f"{LABEL}_label"


def set_label(label: str) -> None:
    global LABEL, STATUS
    if label not in LABELS:
        raise SystemExit("--label must be one of %s" % ", ".join(LABELS))
    LABEL, STATUS = label, STATUS_BY_LABEL[label]


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def state(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    return "above zero" if lo > 0 else ("below zero" if hi < 0 else "contains zero")


# --------------------------------------------------------------------------- weighted statistics


def partial_spearman(p: np.ndarray, y: np.ndarray, n: np.ndarray, draw: resample.ClusterDraw) -> np.ndarray:
    """Per replicate: the partial correlation of the weighted ranks of p and y given those of n."""
    out = np.full(draw.b, np.nan)
    for start in range(0, draw.b, resample.BLOCK):
        w = draw.weights[start : start + resample.BLOCK]
        rp, ry, rn = (resample.weighted_ranks(v, w) for v in (p, y, n))
        r_py = resample._weighted_corr(rp, ry, w)
        r_pn = resample._weighted_corr(rp, rn, w)
        r_yn = resample._weighted_corr(ry, rn, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start : start + resample.BLOCK] = (r_py - r_pn * r_yn) / np.sqrt((1.0 - r_pn ** 2) * (1.0 - r_yn ** 2))
    return out


def fifths(n: np.ndarray) -> list[np.ndarray]:
    """Boolean masks of the five fifths of n (cut at its 20th, 40th, 60th and 80th percentiles over n > 0)."""
    cuts = np.percentile(n[n > 0], FIFTHS)
    edges = np.concatenate([[-np.inf], cuts, [np.inf]])
    return [(n > edges[k]) & (n <= edges[k + 1]) for k in range(5)]


def both(statistic, *args) -> tuple[float, np.ndarray]:
    """(point, replicates) of a statistic computed with the point weights and with E1b's draws (args end with screen)."""
    *values, screen = args
    draws = DATA["draws"][screen]
    return float(statistic(*values, draws["point"])[0]), statistic(*values, draws["draw"])


def seed_mean(pairs: list) -> tuple[float, np.ndarray]:
    return float(np.mean([p for p, _ in pairs])), np.mean(np.vstack([r for _, r in pairs]), axis=0)


def spearman(x, y, draw, mask=None):
    return resample.weighted_spearman(x, y, draw, mask=mask)[:, 0]


# --------------------------------------------------------------------------- one combination


def combination(encoder: str, cell: str, screen: str) -> list[dict]:
    """(2) and (3) for one encoder, cell and screen; each statistic per head seed, then the mean over seeds."""
    import make_e2_stats as st

    y, n = DATA["label"][(cell, screen)], DATA["n"][screen]
    e, null = st.roles(encoder)["e"], ne.name_of(encoder)
    rows, spread = [], {}
    for role, representation in (("null", null), ("e", e)):
        seeds = st.wanted_seeds(representation, "forest") if role == "e" else ne.SEEDS
        values = [st.DATA["lookup"].get(cell, LABEL, representation, "forest", s) for s in seeds]
        if any(v is None or screen not in v for v in values):
            raise Stop("%s, %s, %s: a saved prediction of %s is missing" % (encoder, cell, screen, representation))
        predictions = [np.asarray(v[screen], dtype=np.float64) for v in values]
        if any(not np.isfinite(p).all() for p in predictions):
            raise Stop("%s, %s, %s: a prediction of %s is not finite" % (encoder, cell, screen, representation))
        spread[role] = {"sd_external": float(np.mean([p.std(ddof=1) for p in predictions])),
                        "sd_vcc_test": float(np.mean([np.asarray(v["test"], dtype=np.float64)[np.isfinite(v["test"])].std(ddof=1) for v in values]))}
        with_n = seed_mean([both(spearman, p, n, screen) for p in predictions])
        rows.append(row("(2)" if role == "null" else "(3a)", encoder, cell, screen, f"rho(prediction of {representation}, n_t)", with_n,
                        reading("head", role, with_n)))
        if role == "e":
            rho_y = seed_mean([both(spearman, p, y, screen) for p in predictions])
            reference = DATA["external"].get((encoder, cell, screen))
            if reference is None or not abs(rho_y[0] - reference) <= TOLERANCE:
                raise Stop("%s, %s, %s: rho(e, y) %r differs from external.csv %r" % (encoder, cell, screen, rho_y[0], reference))
            partial = seed_mean([both(partial_spearman, p, y, n, screen) for p in predictions])
            change = (rho_y[0] - partial[0], rho_y[1] - partial[1])
            rows.append(row("(3b)", encoder, cell, screen, f"partial rho({representation}, y | n_t)", partial, partial_reading(partial)))
            rows.append(row("(3b)", encoder, cell, screen, f"rho({representation}, y) - partial rho({representation}, y | n_t)", change, ""))
            rows.append(row("(3b)", encoder, cell, screen, f"rho({representation}, y)", rho_y, ""))
            if LABEL != "rebuilt":
                continue
            masks = fifths(n)
            per_fifth = []
            for k, mask in enumerate(masks):
                value = seed_mean([(float(spearman(p, y, DATA["draws"][screen]["point"], mask)[0]), spearman(p, y, DATA["draws"][screen]["draw"], mask))
                                   for p in predictions])
                per_fifth.append(value)
                rows.append(row("(3c)", encoder, cell, screen, f"rho({representation}, y), fifth {k + 1} of n_t ({int(mask.sum())} constructs)", value,
                                "descriptive"))
            mean = (float(np.mean([v[0] for v in per_fifth])), np.mean(np.vstack([v[1] for v in per_fifth]), axis=0))
            rows.append(row("(3c)", encoder, cell, screen, f"rho({representation}, y), mean of the five fifths of n_t", mean, "descriptive"))
    for role in ("null", "e"):
        rows.append({"analysis": "(2)" if role == "null" else "(3a)", "encoder": encoder, "cell": cell, "screen": screen,
                     "quantity": "spread of the predictions of %s" % (null if role == "null" else e), "estimate": spread[role]["sd_external"],
                     "lo": np.nan, "hi": np.nan, "p": np.nan, "se": np.nan, "state": "",
                     "reading": "descriptive: standard deviation over external constructs %.4g, over VCC test rows %.4g (ratio %.3f)" % (
                         spread[role]["sd_external"], spread[role]["sd_vcc_test"], spread[role]["sd_external"] / spread[role]["sd_vcc_test"]),
                     "role": STATUS, "markers": DATA["markers"][encoder]})
    return rows


def reading(_, role: str, value) -> str:
    import make_e1b_stats as e1s

    s = e1s.summarise(value)
    who = "the null's head" if role == "null" else "e's head"
    st_ = state(s["lo"], s["hi"])
    if st_ == "contains zero":
        return f"{who} does not rank the external constructs by cell count"
    if st_ == "undefined":
        return "undefined interval"
    return f"{who} ranks the external constructs by cell count ({'positive' if st_ == 'above zero' else 'negative'})"


def partial_reading(value) -> str:
    import make_e1b_stats as e1s

    s = e1s.summarise(value)
    st_ = state(s["lo"], s["hi"])
    return {"below zero": "e's external ranking stays below zero with the cell count held fixed (rank-linear)",
            "contains zero": "e's external ranking is not shown below zero once the cell count is held fixed",
            "above zero": "e's external ranking is above zero once the cell count is held fixed",
            }.get(st_, "undefined interval")


def row(analysis, encoder, cell, screen, quantity, value, text) -> dict:
    import make_e1b_stats as e1s

    s = e1s.summarise(value)
    return {"analysis": analysis, "encoder": encoder, "cell": cell, "screen": screen, "quantity": quantity, "estimate": s["estimate"],
            "lo": s["lo"], "hi": s["hi"], "p": s["p"], "se": s["se"], "state": state(s["lo"], s["hi"]), "reading": text, "role": STATUS,
            "markers": DATA["markers"].get(encoder, "")}


def combination_task(encoder, cell, screen):
    return combination(encoder, cell, screen)


# --------------------------------------------------------------------------- the run


def run() -> None:
    import make_e1b_stats as e1s
    import make_e2_stats as st

    started = time.time()
    e1s.WORKERS = WORKERS
    st.DATA["missing"] = []
    st.DATA["order_marker"] = st.order_marker()
    predictions = st.load_predictions()
    inputs = st.load_cells()
    ne.LABEL = LABEL
    null_manifest = ne.outputs_dir() / "predictions_manifest.json"
    manifest = json.loads(null_manifest.read_text())
    if manifest["failed"] or manifest["completed"] != manifest["fits"] or not manifest["vcc_predictions_equal_to_the_saved_n0_fits"]:
        raise Stop("the null's fits for the %s label are not complete or their check did not pass" % LABEL)
    external = pd.read_csv(EXTERNAL_CSV, float_precision="round_trip")
    external = external[(external["label"] == LABEL) & (external["head"] == "forest")]
    DATA["external"] = {}
    for encoder in ne.ENCODERS:
        representation = st.roles(encoder)["e"]
        for (cell, screen), group in external[(external.encoder == encoder) & (external.quantity == f"rho({representation})")].groupby(["cell", "screen"]):
            if len(group) != 1:
                raise Stop("%d rows of rho(%s) in external.csv for %s, %s" % (len(group), representation, cell, screen))
            DATA["external"][(encoder, cell, screen)] = float(group["estimate"].iloc[0])
    lookup = st.DATA["lookup"]
    st.DATA["lookup"] = st.Lookup(lookup.e1b, lookup.e2 + [e1s.Predictions(null_manifest)], {**st.e2_names(), **{ne.name_of(e): e for e in ne.ENCODERS}})
    e1s.DATA["predictions"] = st.DATA["lookup"]
    DATA["draws"] = e1s.STATE["draws"]
    DATA["markers"] = {e: st.markers(e, "Replogle") for e in ne.ENCODERS}
    null = ne.load_null()
    DATA["n"], DATA["label"], constructs = {}, {}, {}
    for screen in st.SCREENS:
        table = e1s.DATA["external"][(screen, e1s.DATA["cells"]["G5"].external_source[screen])]
        names = table["construct"].astype(str).to_numpy()
        index = {c: i for i, c in enumerate(null[f"{screen}__constructs"].tolist())}
        missing = [c for c in names if c not in index]
        if missing:
            raise Stop("%s: %d external constructs have no n_t" % (screen, len(missing)))
        n = null[f"{screen}__n"][[index[c] for c in names]].astype(np.float64)
        if (n <= 0).any():
            raise Stop("%s: an external construct has no eligible perturbed cell" % screen)
        DATA["n"][screen], constructs[screen] = n, int(names.size)
        for cell in ne.CELLS:
            table = e1s.DATA["external"][(screen, e1s.DATA["cells"][cell].external_source[screen])]
            if not np.array_equal(table["construct"].astype(str).to_numpy(), names):
                raise Stop("%s: the external constructs differ between cells" % screen)
            DATA["label"][(cell, screen)] = table["y"].to_numpy(dtype=np.float64)
    print("[e2 null follow-up] inputs loaded, %.0f s" % (time.time() - started), flush=True)

    rows = []
    observed = pd.read_csv(ne.RESULTS / "null_external.csv", float_precision="round_trip")
    for cell in (ne.CELLS if LABEL == "rebuilt" else ()):
        for screen in st.SCREENS:
            value = both(spearman, DATA["n"][screen], DATA["label"][(cell, screen)], screen)
            s = e1s.summarise(value)
            m = max(abs(s["lo"]), abs(s["hi"]))
            rows.append(row("(1)", "", cell, screen, "rho(n_t, y)", value, "the cell-count baseline; M = %.4f" % m))
            for encoder in ne.ENCODERS:
                t = observed[(observed.encoder == encoder) & (observed.cell == cell) & (observed.screen == screen) & (observed.quantity == "T")]
                if len(t) != 1:
                    raise Stop("%d T rows of null_external.csv for %s, %s, %s" % (len(t), encoder, cell, screen))
                t = t.iloc[0]
                if not t["hi"] < 0:
                    continue
                t_min = abs(float(t["hi"]))
                text = ("no monotone function of the cell count alone ranks the constructs as far below zero as T does" if m < t_min
                        else "a monotone function of the cell count alone can reach T's size")
                rows.append({"analysis": "(1)", "encoder": encoder, "cell": cell, "screen": screen, "quantity": "M against |T|min",
                             "estimate": m, "lo": np.nan, "hi": np.nan, "p": np.nan, "se": np.nan, "state": "",
                             "reading": "%s (M %.4f, |T|min %.4f, T %+.4f [%+.4f, %+.4f], null_external.csv)" % (text, m, t_min, t["estimate"], t["lo"], t["hi"]),
                             "role": STATUS, "markers": DATA["markers"][encoder]})
    import make_e1b_stats as e1s_mod

    tasks = [(encoder, cell, screen) for encoder in ne.ENCODERS for cell in ne.CELLS for screen in st.SCREENS]
    for result in e1s_mod.parallel(combination_task, tasks):
        rows.extend(result)
    if st.DATA["missing"]:
        raise Stop("missing: %s" % st.DATA["missing"][:5])
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if LABEL != "rebuilt":
        frame.insert(4, "training_label", LABEL)
    frame.to_csv(out / "saved.csv", index=False)
    record = {"plan": PLAN, "status": STATUS,
              "settings": {"replicates": e1s.B, "external_seeds": e1s.EXT_SEED, "label": LABEL, "head": "forest", "fifths_at_percentiles": list(FIFTHS)},
              "external_replicate_index_sha256": {s: DATA["draws"][s]["draw"].index_sha256 for s in st.SCREENS},
              "constructs": constructs,
              "inputs": {**inputs, "predictions": predictions, "null_predictions": file_record(null_manifest), "null_features": file_record(ne.FEATURES_JSON),
                         "null_external_csv": file_record(ne.RESULTS / "null_external.csv")},
              "code": ne.code_digests(), "outputs": {"saved.csv": file_record(out / "saved.csv")}, "rows": int(len(frame)),
              "seconds": round(time.time() - started, 1)}
    if LABEL != "rebuilt":
        record["check_rho_e_y_equals_external_csv"] = {"values_compared": len(DATA["external"]), "tolerance": TOLERANCE}
        record["inputs"]["external_csv"] = file_record(EXTERNAL_CSV)
    (out / "saved.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 null follow-up] %d rows written, %.0f s" % (len(frame), time.time() - started), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--workers" in arguments:
        i = arguments.index("--workers")
        WORKERS = int(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if "--label" in arguments:
        i = arguments.index("--label")
        set_label(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if "--out-scratch" in arguments:
        i = arguments.index("--out-scratch")
        name = arguments[i + 1]
        if not name or "/" in name or name.startswith("."):
            raise SystemExit("--out-scratch takes a plain folder name")
        OUT_OVERRIDE = config.storage_root() / "bioinf2027" / "scratch" / name
        arguments = arguments[:i] + arguments[i + 2:]
    if arguments != ["--saved"]:
        raise SystemExit("usage: make_e2_null_external_followup.py --saved [--label rebuilt|existing] [--out-scratch NAME] [--workers N]")
    run()
