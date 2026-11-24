"""Runtime of E4 and E6 under the options put to the author for opening gate G2 (bioinf2027/GATE.md).

Read-only: it fits nothing, changes no run and reads only saved files. Every listed fit of E6, E4 and
D2's head fits is priced as bioinf2027/scripts/progress_status.py prices E1b's unseen kinds: E1b's
recorded seconds per training row and column per head over its finished fits (E1b keeps running, so the
basis is recorded with its count of finished fits). Twelve workers then take the fits in the order of
the option (tier first, longest first within a tier), each fit available from its release time: E4 from
the end of E1b, E6 after its check (its harness refit and its timing probe, in one process), D2's fits
and E6's R(z-) fits from the end of D2's embeddings. The simulated tier ends give the estimates; a band
repeats them with every CPU price times 0.7 and 1.3.

Categories (revised after bioinf2027/reviews/G2_runtime_proposal_r1.md):
- family: the fits behind E4's F (matched forests of z, x, m and [x; m] in G5 and G6, draws 1 to 3, seeds
  0 to 2, and the corner z fits there), E6's 24 statistics with their validity controls (27/788,
  gene-disjoint, forest, seeds 0 to 2; R(z-) once D2 exists) and D2's head fits.
- marker: fits whose result is a marker on a family statement. E6: gene-fold sensitivity, refits (E6
  plan, Sections 3.2 and 6, item 9). E4, under the reading that F inherits E1's rules (E4 plan, Section
  10.2, item 7; E1 note, Sections 21.2, item 9, and 26, item 4): the source-mode fits and the refits of
  the family arms in G5 and G6 ("not robust to the label mode", "not robust to head fitting").
- paired: the inputs of E6's paired E6 - E1 differences complete (E6 plan, Section 6, item 6, with
  Section 13.1, items 3 and 4): z, x, m, [x; m] and C1 at the other endpoints, C2 at all four, the
  size-matched arm (paired with G1 and G2), and E1b's x basis at 27/788.
- z - C2: E4's matched C2 in G5 and G6 (E6's C2 at 27/788 is among the paired inputs).
- unread: E4 fits that bioinf2027/scripts/make_e4_stats.py never reads.

Options: planned (E6 and E4 beside each other after E1b on one pool of 12 workers; D2's fits and R(z-)
when released); order (tier 1 family, tier 2 markers, tier 3 the rest), with variants; the correction
(unread E4 fits dropped); the E4 rule (below). E4 rule: while the projection exceeds 200 hours of CPU
(E4 plan, Section 11.2, item 2, upper end; E6's threshold): (a) every E4 fit outside the family, its
marker fits and the corner fits runs with head seed 0 only; (b) every E4 fit of G1 and G2, of the fourth
corner, the size factor, the depth contrast, the draw-level positive control and the secondary heads runs
on feature replicate 1 only.

Also: the spread over head seeds of E1b's T (arm minus C1, external Spearman, per seed) on E1b's finished
fits, and the head seeds' agreement on predictions.

Outputs: bioinf2027/diag/E4_E6_runtime.json and bioinf2027/diag/E4_E6_runtime_categories.csv.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import heapq  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import e1cells  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import progress_status as ps  # noqa: E402
from fit_tiers import FAMILY_ARMS, E6_FAMILY_MODELS, e4_category, e4_unread, e6_category  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "bioinf2027" / "diag"
WORKERS = 12
E4_THRESHOLD_HOURS = 200.0
BAND = (0.7, 1.3)
E6_ROWS = {"RPE1": 1932, "K562-essential": 1903}
CONSTRUCTS_PER_GENE = 1.07  # 1,932 RPE1 constructs of 1,811 genes, for the size-matched draws
D2_RESTART_HOURS = 0.5      # two restarts of the GPU jobs, one watcher period each, rounded up


# --------------------------------------------------------------------------- categories


def e6_rows(d: dict) -> int:
    n = E6_ROWS[d["source"]]
    if "draw" in d:
        return int(round({"27/788": 98, "10/788": 140}[d["endpoint"]] * CONSTRUCTS_PER_GENE))
    return n if d["fit_set"] == "whole screen" else int(round(n * 0.8))


def replicate(d: dict) -> str:
    """The feature replicate of an E4 fit ("" where the fit has none: corner, cell-disjoint)."""
    if "k" in d:
        return str(d["k"])
    variant = str(d.get("variant", ""))
    return variant.split()[-1] if variant[-1:].isdigit() else ""


def priced(tasks, rows_of, category, model) -> pd.DataFrame:
    out = []
    for d, s in tasks:
        n = rows_of(d)
        out.append({"category": category(d, s), "head": s.head, "representation": s.representation, "seed": s.seed,
                    "variant": s.variant, "replicate": replicate(d), "cell": d.get("cell", d.get("source", "")), "kind": d.get("kind", ""),
                    "seconds": ps.price("", s.head, s.representation, n, model, by_kind=False), "rows": n})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- task lists


def e6_tasks(model: dict):
    import make_e6_fits as e6

    e6.DATA["e4"] = True  # E4's features exist (the matched sensitivity of Section 6, item 11)

    def hours(listed) -> float:
        return priced(listed, e6_rows, e6_category, model)["seconds"].sum() / 3600

    refit_tasks = [t for t in e6.task_list(e6.REFITS, e6.SEEDS) if t[1].variant.startswith("refit ")]
    refits = e6.REFITS if hours(refit_tasks) <= e6.REFIT_LIMIT_HOURS else e6.REFITS_CUT
    tasks = e6.task_list(refits, e6.SEEDS) + e6.shared_c2_tasks()
    seeds = e6.SEEDS
    if hours(tasks) > e6.CPU_LIMIT_HOURS:
        seeds = (0,)
        tasks = e6.task_list(refits, seeds) + e6.shared_c2_tasks()
    z_minus = []
    for source in e6.SCREENS:
        for k in range(5):
            d = {"source": source, "endpoint": e6.PRIMARY, "fit_set": f"fold {k}", "z_minus": True}
            for seed in e6.SEEDS:
                z_minus.append((d, FitSpec(f"E6 | {source} | fold {k} | z- of D2", "rebuilt", "z", "forest", seed)))
    rule = {"refits": refits, "seeds_of_the_700_control_endpoints": list(seeds), "refit_hours_at_50": round(hours(refit_tasks), 1)}
    return tasks + z_minus, rule


def e4_tasks(refits: int):
    import make_e4_fits as e4

    e4.DATA["thinned"] = True  # the thinned features exist (bioinf2027/results/E4/features_thinned.json)
    return e4.task_list(refits)


def d2_tasks(refits: int):
    import make_d2_fits as d2

    return d2.task_list(refits)


def apply_e4_rule(frame: pd.DataFrame, step: str) -> pd.DataFrame:
    protected = frame["category"].str.startswith("family") | frame["category"].str.startswith("marker") | (frame["kind"] == "corner")
    keep = ~frame["category"].str.startswith("unread")
    if step in ("a", "b"):
        keep &= protected | (frame["seed"] == 0)
    if step == "b":
        variants = frame["category"].isin(["matched, G1 and G2", "fourth corner (existing label)", "size factor", "depth-thinned",
                                           "draw-level B2", "secondary heads"])
        keep &= ~variants | frame["replicate"].isin(["1", ""])
    return frame[keep]


# --------------------------------------------------------------------------- E1b's head seeds


def e1b_seeds() -> dict:
    """Agreement of head seeds' external predictions, and the spread over seeds of T (arm minus C1), on E1b's finished fits."""
    root = ps.config.storage_root() / "bioinf2027" / "E1b" / "predictions"
    groups = {}
    for path in root.rglob("*.npz"):
        if path.parent.name == "harness":
            continue
        with np.load(path, allow_pickle=False) as store:
            spec = json.loads(str(store["spec"]))
            if spec["head"] != "forest" or spec["label"] != "rebuilt" or spec["variant"] != "all training rows" or spec["realisation"] != 0:
                continue
            groups.setdefault((spec["context"], spec["representation"]), {})[spec["seed"]] = {
                k: store[k] for k in ("RPE1", "K562-essential") if k in store.files}
    agreement, spread = [], []
    labels = {}
    for (context, representation), seeds in groups.items():
        if len(seeds) < 3:
            continue
        for screen in seeds[0]:
            for a, b in ((0, 1), (0, 2), (1, 2)):
                agreement.append(float(spearmanr(seeds[a][screen], seeds[b][screen]).statistic))
        control = groups.get((context, "C1"), {})
        if representation == "C1" or len(control) < 3:
            continue
        if context not in labels:
            tables = e1cells.load_tables(context)
            labels[context] = {screen: tables[screen]["y"].to_numpy(dtype=np.float64) for screen in e1cells.CELLS[context]["screens"]}
        for screen in seeds[0]:
            y = labels[context][screen]
            t = [float(spearmanr(seeds[s][screen], y).statistic - spearmanr(control[s][screen], y).statistic) for s in (0, 1, 2)]
            spread.append({"context": context, "representation": representation, "screen": screen, "sd": float(np.std(t, ddof=1)),
                           "range": float(max(t) - min(t)), "mean": float(np.mean(t))})
    frame = pd.DataFrame(spread)
    finite = [v for v in agreement if np.isfinite(v)]
    out = {"prediction_agreement": {"pairs": len(agreement), "pairs_with_a_constant_prediction": len(agreement) - len(finite),
                                    "median": float(np.median(finite)) if finite else None, "minimum": float(np.min(finite)) if finite else None}}
    if len(frame):
        worst = frame.loc[frame["range"].idxmax()]
        out["t_over_head_seeds"] = {"statistics": int(len(frame)), "sd_median": float(frame["sd"].median()), "sd_max": float(frame["sd"].max()),
                                    "range_median": float(frame["range"].median()), "range_max": float(frame["range"].max()),
                                    "range_max_at": {k: worst[k] for k in ("context", "representation", "screen")},
                                    "note": "T per head seed from one seed of the arm and the same seed of C1; the family averages three"}
    return out


# --------------------------------------------------------------------------- schedule


def simulate(jobs: list[dict]) -> dict:
    """List scheduling on WORKERS workers: each free worker takes the released job of the lowest tier, longest first."""
    jobs = sorted(jobs, key=lambda j: j["release"])
    free = [0.0] * WORKERS
    heapq.heapify(free)
    ready, ends, i = [], {}, 0
    while i < len(jobs) or ready:
        now = heapq.heappop(free)
        while i < len(jobs) and jobs[i]["release"] <= now:
            heapq.heappush(ready, (jobs[i]["tier"], -jobs[i]["seconds"], i, jobs[i]))
            i += 1
        if not ready:
            heapq.heappush(free, jobs[i]["release"])
            continue
        _, _, _, job = heapq.heappop(ready)
        end = now + job["seconds"]
        ends[job["group"]] = max(ends.get(job["group"], 0.0), end)
        heapq.heappush(free, end)
    return ends


TIER_RULES = {
    # option: the categories (by prefix) that run before G2 besides the families; None for the planned order
    "planned": None,
    "order, families only": (),
    "order": ("marker",),
    "order, with E6 Section 6, item 6 complete": ("marker", "paired"),
    "order, with Section 6, item 6 and z - C2": ("marker", "paired", "z - C2"),
}


def main() -> None:
    now = datetime.now()
    rows = ps.train_rows()
    finished = ps.e1b_finished()
    model = ps.pricing(finished, rows)
    e1 = ps.e1b_status(rows)
    d2s = ps.d2_status()
    e1_end = e1["remaining_wall_hours"] * 3600
    d2_end = (d2s["remaining_gpu_hours"] + D2_RESTART_HOURS) * 3600
    refits = json.loads((REPO / "bioinf2027" / "results" / "E1" / "protocol.json").read_text())["refits"]

    e6_list, e6_rule = e6_tasks(model)
    e6f = priced(e6_list, e6_rows, e6_category, model)
    e4f = priced(e4_tasks(refits), lambda d: rows.get(d["cell"], 3000), e4_category, model)
    d2f = priced(d2_tasks(refits), lambda d: rows.get(d["cell"], 3000), lambda d, s: "family: D2 head fits", model)
    check = sum(ps.price("", "forest", r, int(round(n * 0.8)), model, by_kind=False) for n in E6_ROWS.values()
                for r in ("z", "x", "[x; m]", "C1", "m", "TR")) + ps.price("", "forest", "x", rows["G1"], model, by_kind=False)

    categories = pd.concat([f.assign(item=name) for name, f in (("E6", e6f), ("E4", e4f), ("D2", d2f))]).groupby(
        ["item", "category"]).agg(fits=("seconds", "size"), cpu_hours=("seconds", lambda v: v.sum() / 3600)).reset_index()

    steps = {"none": e4f, "0": apply_e4_rule(e4f, "0"), "a": apply_e4_rule(e4f, "a"), "b": apply_e4_rule(e4f, "b")}
    rule_hours = {k: round(v["seconds"].sum() / 3600, 1) for k, v in steps.items()}
    chosen = "0"
    for step in ("a", "b"):
        if rule_hours[chosen] <= E4_THRESHOLD_HOURS:
            break
        chosen = step

    def jobs(e4_frame: pd.DataFrame, option: str, scale: float) -> list[dict]:
        before = TIER_RULES[option]
        out = []
        for f, item, release in ((e6f, "E6", e1_end + check * scale), (e4_frame, "E4", e1_end), (d2f, "D2", d2_end)):
            for r in f.itertuples():
                late = item == "D2" or r.category == "family: R(z-)"
                rel = d2_end if late else release
                if before is None:
                    tier, group = (0 if late else 1), ("D2 fits" if item == "D2" else "E6 and E4")
                else:
                    family = r.category.startswith("family")
                    early = any(r.category.startswith(p) for p in before)
                    tier = 0 if family else (1 if early else 2)
                    group = "families" if family else ("before G2, other" if early else "after G2")
                out.append({"tier": tier, "seconds": r.seconds * scale, "release": rel, "group": group})
        return out

    def at(seconds: float) -> str:
        return (now + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M")

    def run(e4_frame: pd.DataFrame, option: str, scale: float = 1.0) -> dict:
        ends = simulate(jobs(e4_frame, option, scale))
        last = max(ends.values())
        g2 = last if option == "planned" else max(e1_end, *(v for k, v in ends.items() if k != "after G2"))
        return {"gate_G2_fits_end": at(g2), "last_fit_end": at(last), "tier_ends": {k: at(v) for k, v in sorted(ends.items(), key=lambda kv: kv[1])}}

    options = {}
    for name in TIER_RULES:
        options[name] = run(e4f, name)
    options["correction only, planned"] = run(steps["0"], "planned")
    options["correction and order"] = run(steps["0"], "order")
    options["E4 rule with the correction, planned"] = run(steps[chosen], "planned")
    options["E4 rule with the correction, and order"] = run(steps[chosen], "order")
    band = {name: {f"x{s}": run(frame, option, s) for s in BAND} for name, frame, option in
            (("planned", e4f, "planned"), ("order", e4f, "order"), ("correction and order", steps["0"], "order"),
             ("correction, order, Section 6, item 6 and z - C2", steps["0"], "order, with Section 6, item 6 and z - C2"))}
    options["correction, order, Section 6, item 6 and z - C2"] = run(steps["0"], "order, with Section 6, item 6 and z - C2")

    DIAG.mkdir(parents=True, exist_ok=True)
    categories.round(2).to_csv(DIAG / "E4_E6_runtime_categories.csv", index=False)
    report = {
        "computed_at": now.strftime("%Y-%m-%d %H:%M ") + time.strftime("%Z"), "script": "bioinf2027/scripts/estimate_e4_e6_runtime.py",
        "review": "bioinf2027/reviews/G2_runtime_proposal_r1.md",
        "pricing": {"basis": "E1b's finished fits under $IGA_STORAGE_ROOT/bioinf2027/E1b/predictions, seconds per row and column per head",
                    "e1b_fits_finished": len(finished), "seconds_per_row_and_column": model["k"], "timing_probe_per_head": model["probe_k"],
                    "workers": WORKERS, "band": list(BAND),
                    "e6_rows": "four fifths of the source screen's constructs per fold, all for the whole screen, the drawn genes times 1.07 for the size-matched arm",
                    "e4_and_d2_rows": "the cell's VCC training rows"},
        "releases": {"E1b_end": at(e1_end), "E6_check_hours": round(check / 3600, 1), "D2_embeddings_end": at(d2_end),
                     "D2_restart_allowance_hours": D2_RESTART_HOURS},
        "d2_gpu": {"remaining_gpu_hours": round(d2s["remaining_gpu_hours"], 1), "units_done": d2s["finished"], "units": d2s["units"]},
        "cpu_hours": {"E6": round(e6f["seconds"].sum() / 3600, 1), "E4": round(e4f["seconds"].sum() / 3600, 1), "D2 fits": round(d2f["seconds"].sum() / 3600, 1)},
        "fits": {"E6": int(len(e6f)), "E4": int(len(e4f)), "D2 fits": int(len(d2f))},
        "e6_rule_as_its_plan_states": e6_rule,
        "e1b_head_seeds": e1b_seeds(),
        "e4_rule": {"threshold_cpu_hours": E4_THRESHOLD_HOURS, "cpu_hours_after": rule_hours,
                    "fits_after": {k: int(len(v)) for k, v in steps.items()}, "last_step_applied": chosen},
        "options": options,
        "band": band,
        "not_included": "E1b's, E6's, E4's and D2's statistics, the audits and reviews, D2's aggregation; the fits running at the start count as not started",
        "outputs": {"categories": file_record(DIAG / "E4_E6_runtime_categories.csv")},
    }
    (DIAG / "E4_E6_runtime.json").write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
    print(categories.round(1).to_string())
    print(json.dumps({k: report[k] for k in ("computed_at", "releases", "d2_gpu", "cpu_hours", "fits", "e6_rule_as_its_plan_states",
                                              "e1b_head_seeds", "e4_rule")}, indent=1, default=str))
    for k, v in options.items():
        print("%-45s G2 fits %s   last fit %s" % (k, v["gate_G2_fits_end"], v["last_fit_end"]))
    for k, v in band.items():
        print("band %-30s %s" % (k, {s: (r["gate_G2_fits_end"], r["last_fit_end"]) for s, r in v.items()}))


if __name__ == "__main__":
    main()
