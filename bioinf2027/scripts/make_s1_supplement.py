"""S1 supplement: the all-cells label and cell number, on VCC and on the Replogle screens, post hoc
(bioinf2027/plans/S1_supplement.md; the answer of 2026-11-17 to the stop of 06:50, bioinf2027/GATE.md). E1a's builder
(make_e1a_labels.py, bioinf2027/src/bioinf2027/endpoint.py), its read-outs (make_e1a_readouts.py) and, for (d), E2's
saved predictions through make_e2_null_external_followup.py's loaders are reused unchanged. No head is fitted.

Parts (--part, in the order given; default all):
- vcc: (a) rho(existing label, n_pool) - rho(rebuilt label, n_pool) on the rows of G5 and G6, R2's target-gene
  bootstrap (seed 20270301), both labels on the same replicate; (b) the all-cells label's mechanical part on R2's
  simulated subsets (reproduced from default_rng([20270406, k])), its within-(target gene, guide) association and the
  excess, as R2; and the curve on G6's rows with at least 60 perturbed cells (nested subsets, default_rng([20271213, k]),
  m = 10, 15, 20, 30, 45, 60), the all-cells label beside G6's rebuilt label.
- external: (c) on E1b's external constructs, the target-gene label with all eligible perturbed cells against 788
  pooled controls (E1's 25 draws, median) and against all pooled controls; rho with the cell count beside the 27/788
  and 10/788 labels' and the paired difference (E1b's external draws); the within-construct subsampling on constructs
  with at least 150 eligible cells (default_rng([20271214, screen index, construct position]), m = 27, 40, 60, 100, 150).
- control: (d) on (c)'s all-cells label, from saved predictions only: the size-matched null with heads trained on the
  all-cells VCC label and the encoders' all-cells-label heads, each against C1 of the encoder's width; rho(n, y) beside.
Checks (plan Section 0): the existing label reproduced on every full pool (2e-6); (a)'s separate values equal
R2_between.csv (1e-12); R2's simulated sizes and the first 50 rebuilt simulated labels per cell reproduced; the 27/788
external label reproduced for every construct (1e-12); E1b's draws and E2's external.csv reproduced for (d).
--smoke: 200 replicates, 30 rows per VCC cell, 40 constructs per screen, outputs under
$IGA_STORAGE_ROOT/bioinf2027/scratch/S1_supplement_smoke/. Every value is post hoc and unadjusted; no value is read as an
explanation of the negative external values (G7, item 6).
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
import multiprocessing  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, endpoint, resample, stats  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.stats import percentile_interval  # noqa: E402

import make_e1a_labels as builder  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/S1_supplement.md"
STATUS = "post hoc, unadjusted, not part of any family (the stop of 2026-11-17 06:50, S1 supplement; %s)" % PLAN
OUT = REPO / "bioinf2027" / "results" / "S1_supplement"
READOUTS = REPO / "bioinf2027" / "endpoints" / "readouts"
PARTS = ("vcc", "external", "control")
VCC_CELLS = ("G5", "G6")
SPLITS = tuple(builder.SPLITS)  # train, val, test (a dict in the builder)
B = stats.BOOTSTRAP_B
SEED = stats.BOOTSTRAP_SEED  # 20270301, R2's
SIMULATION_SEED = builder.SIMULATION_SEED  # 20270406, R2's simulation
SIMULATED_SIZES = builder.SIMULATED_SIZES
CURVE_SEED = 20271213
CURVE_FLOOR = 60
CURVE_SIZES = (10, 15, 20, 30, 45, 60)
SUB_SEED = 20271214
SUB_FLOOR = 150
SUB_SIZES = (27, 40, 60, 100, 150)
N_CTRL_EXTERNAL = 788
EXISTING_TOLERANCE = 2e-6
TOLERANCE = 1e-12
SCREEN_INDEX = {"RPE1": 0, "K562-essential": 1}
SMOKE_ROWS = 30
SMOKE_CONSTRUCTS = 40
WORKERS = 4
SMOKE = False
DATA: dict = {}

READ_A = {"above zero": "on the same rows, the all-cells label's association with pool size exceeds the size-fixed label's",
          "below zero": "on the same rows, the size-fixed label's association with pool size exceeds the all-cells label's",
          "contains zero": "not shown to differ on the same rows"}
READ_C = {"above zero": "on the same constructs, the all-cells label's association with the perturbed-cell count exceeds the size-fixed label's",
          "below zero": "on the same constructs, the size-fixed label's association with the perturbed-cell count exceeds the all-cells label's",
          "contains zero": "not shown to differ on the same constructs"}
READ_NULL = {"above zero": ("with heads trained on the all-cells label, the size-matched null ranks the constructs' all-cells external label above the "
                            "Gaussian control: the null carries cell count across screens where the label depends on it"),
             "contains zero": "the null is not shown to carry cell count to the all-cells external label",
             "below zero": "the null ranks the all-cells external label below the Gaussian control"}


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "S1_supplement_smoke" if SMOKE else OUT


def my_digest() -> str:
    return sha256(Path(__file__).resolve())


def state(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    return "above zero" if lo > 0 else ("below zero" if hi < 0 else "contains zero")


def write_record(folder: Path, part: str, started: float, outputs: list[str], extra: dict) -> None:
    record = {"plan": PLAN, "status": STATUS, "smoke": SMOKE, "part": part, "replicates": B,
              "seeds": {"R2 bootstrap": SEED, "R2 simulation": SIMULATION_SEED, "VCC curve": CURVE_SEED, "external subsampling": SUB_SEED,
                        "label draws": list(endpoint.SEEDS)},
              "code": {config.portable(Path(__file__).resolve()): my_digest(), config.portable(Path(builder.__file__).resolve()): sha256(Path(builder.__file__).resolve()),
                       "bioinf2027/src/bioinf2027/endpoint.py": sha256(REPO / "bioinf2027" / "src" / "bioinf2027" / "endpoint.py"),
                       "bioinf2027/src/bioinf2027/stats.py": sha256(REPO / "bioinf2027" / "src" / "bioinf2027" / "stats.py")},
              **extra, "outputs": {n: file_record(folder / n) for n in outputs}, "seconds": round(time.time() - started, 1)}
    (folder / f"s1_supplement_{part}.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")


def parallel(function, tasks: list) -> list:
    if WORKERS <= 1 or len(tasks) <= 1:
        return [function(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(function, tasks))


# --------------------------------------------------------------------------- inputs


def load_counts() -> dict:
    counts, record = builder.load_counts()
    DATA["counts_record"] = record["manifest"]
    return counts


def load_table(cell: str, name: str) -> tuple[pd.DataFrame, dict]:
    manifest_path = REPO / "bioinf2027" / "endpoints" / cell / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["tables"][name]
    path = builder.endpoint_root() / cell / name
    if sha256(path) != record["sha256"]:
        raise Stop("%s/%s does not match its manifest" % (cell, name))
    return pd.read_csv(path, float_precision="round_trip"), {"manifest": file_record(manifest_path), "table": record}


def prepare_vcc(counts: dict) -> dict:
    rows = builder.vcc_row_table(counts)
    train = counts["vcc_train"]
    DATA["target_slot"] = {t: i for i, t in enumerate(train["targets"].tolist())}
    batches = sorted(set(train["nt_batch"].tolist()))
    DATA["batch_members"] = {b: np.flatnonzero(train["nt_batch"] == b) for b in batches}
    DATA["values"] = builder.vcc_values(counts, "all", {})
    inputs = {}
    DATA["cells"] = {}
    for cell in VCC_CELLS:
        cell_rows = builder.cell_rows(rows, builder.CELLS[cell])
        labels, record = load_table(cell, "vcc_labels.csv")
        simulation, record_sim = load_table(cell, "pool_size_simulation.csv")
        keys = ["split", "batch", "target_gene", "n_pool"]
        if len(cell_rows) != len(labels) or not cell_rows[keys].reset_index(drop=True).astype(str).equals(labels[keys].astype(str)):
            raise Stop("%s: the builder's rows differ from vcc_labels.csv" % cell)
        DATA["cells"][cell] = {"rows": cell_rows, "labels": labels, "simulation": simulation}
        inputs[cell] = {"vcc_labels.csv": record, "pool_size_simulation.csv": record_sim}
    return inputs


def row_values(cell: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Source- and external-mode values of row k's perturbed cells and of its batch's control cells (the target gene)."""
    row = DATA["cells"][cell]["rows"].iloc[k]
    t = DATA["target_slot"][row.target_gene]
    pos = row.positions
    members = DATA["batch_members"][row.batch]
    v = DATA["values"]
    return v[f"{row.split}_src"][pos], v["nt_src"][members, t], v[f"{row.split}_ext"][pos], v["nt_ext"][members, t]


# --------------------------------------------------------------------------- (a) and (b)


def vcc_row_task(args) -> dict:
    """For one row: the existing label on the full pool, on R2's simulated subsets, and on the curve's nested subsets."""
    cell, k = args
    spec = builder.CELLS[cell]
    rows = DATA["cells"][cell]["rows"]
    sizes = rows["n_pool"].to_numpy()
    src, ctrl_src, ext, ctrl_ext = row_values(cell, k)
    out = {"cell": cell, "k": k, "full": endpoint.ad_log1p(src, ctrl_src), "simulation": [], "curve": None}
    n = src.size
    if n >= 2 * spec["n_pert"]:
        rng = np.random.default_rng([SIMULATION_SEED, k])
        candidates = sizes[(sizes >= spec["n_pert"]) & (sizes <= n)]
        for j, size in enumerate(rng.choice(candidates, size=SIMULATED_SIZES, replace=True)):
            subset = np.sort(rng.choice(n, size=int(size), replace=False))
            entry = {"simulation": j, "size": int(size), "existing": endpoint.ad_log1p(src[subset], ctrl_src)}
            if k < DATA["rebuilt_check_rows"]:
                entry["rebuilt"] = endpoint.subsampled_label(ext[subset], ctrl_ext, spec["n_pert"], spec["n_ctrl"])[0]["y"]
            out["simulation"].append(entry)
    if cell == "G6" and n >= CURVE_FLOOR:
        order = np.random.default_rng([CURVE_SEED, k]).permutation(n)
        curve = {}
        for m in CURVE_SIZES:
            cells = order[:m]
            curve[m] = {"existing": endpoint.ad_log1p(src[cells], ctrl_src),
                        "rebuilt": endpoint.subsampled_label(ext[cells], ctrl_ext, spec["n_pert"], spec["n_ctrl"])[0]["y"]}
        curve["full"] = {"existing": out["full"], "rebuilt": endpoint.subsampled_label(ext, ctrl_ext, spec["n_pert"], spec["n_ctrl"])[0]["y"]}
        out["curve"] = curve
    return out


def paired_corr(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, groups: np.ndarray, seed: int = SEED) -> dict:
    """Spearman of y1 and of y2 with x and their difference, each replicate drawing target genes with replacement exactly as
    stats.cluster_bootstrap_corr does; replicates with an undefined value are left out."""
    finite = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    x, y1, y2, groups = x[finite], y1[finite], y2[finite], np.asarray(groups)[finite]
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        nan = float("nan")
        return {"estimate": nan, "lo": nan, "hi": nan, "p": nan, "se": nan, "first": nan, "second": nan, "rows": int(x.size),
                "target_genes": int(len(unique)), "replicates_left_out": B}
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    s1, s2 = stats._safe_corr(x, y1)[0], stats._safe_corr(x, y2)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        picked = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[p] for p in picked])
        a, b = stats._safe_corr(x[index], y1[index])[0], stats._safe_corr(x[index], y2[index])[0]
        if np.isfinite(a) and np.isfinite(b):
            draws.append(a - b)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    draws = np.asarray(draws)
    return {"estimate": float(s1 - s2), "lo": float(lo), "hi": float(hi), "p": resample.p_value(draws), "se": float(np.std(draws, ddof=1)),
            "first": float(s1), "second": float(s2), "rows": int(x.size), "target_genes": int(len(unique)), "replicates_left_out": int(B - draws.size)}


def centred(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"v": values, "g": groups})
    return (frame["v"] - frame.groupby("g")["v"].transform("mean")).to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats as sps

    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(sps.rankdata(x), sps.rankdata(y))[0, 1])


def mechanical(observed: pd.DataFrame, simulated: pd.DataFrame) -> dict:
    """R2's within-(gene, guide) association, its mechanical part and the excess, with R2's target-gene bootstrap."""
    if len(observed) < 3 or len(simulated) < 3:
        nan = float("nan")
        return {k: nan for k in ("observed_within_gene_and_guide", "observed_lo", "observed_hi", "mechanical", "mechanical_lo", "mechanical_hi",
                                 "sign_of_observed", "excess", "excess_lo", "excess_hi")} | {
            "observed_rows": int(len(observed)), "simulated_labels": int(len(simulated)), "target_genes": 0, "replicates_undefined": B}
    obs = spearman(observed["c_n"].to_numpy(), observed["c_y"].to_numpy())
    mech = spearman(simulated["c_n"].to_numpy(), simulated["c_y"].to_numpy())
    sign = 1.0 if obs >= 0 else -1.0
    genes = np.unique(np.r_[observed["target_gene"].to_numpy(), simulated["target_gene"].to_numpy()])
    members_o = {g: np.flatnonzero(observed["target_gene"].to_numpy() == g) for g in genes}
    members_s = {g: np.flatnonzero(simulated["target_gene"].to_numpy() == g) for g in genes}
    rng = np.random.default_rng(SEED)
    draws = np.full((B, 3), np.nan)
    on, oy, sn, sy = (f[c].to_numpy() for f, c in ((observed, "c_n"), (observed, "c_y"), (simulated, "c_n"), (simulated, "c_y")))
    for r in range(B):
        picked = genes[rng.integers(0, genes.size, size=genes.size)]
        io = np.concatenate([members_o[g] for g in picked])
        is_ = np.concatenate([members_s[g] for g in picked])
        ob = spearman(on[io], oy[io]) if io.size else np.nan
        me = spearman(sn[is_], sy[is_]) if is_.size else np.nan
        draws[r] = (ob, me, sign * (ob - me))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return {"observed_within_gene_and_guide": obs, "observed_lo": lo[0], "observed_hi": hi[0], "mechanical": mech, "mechanical_lo": lo[1],
            "mechanical_hi": hi[1], "sign_of_observed": sign, "excess": sign * (obs - mech), "excess_lo": lo[2], "excess_hi": hi[2],
            "observed_rows": int(len(observed)), "simulated_labels": int(len(simulated)), "target_genes": int(genes.size),
            "replicates_undefined": int(np.isnan(draws[:, 2]).sum())}


def gene_bootstrap_mean(values: np.ndarray, genes: np.ndarray, seed: int = SEED) -> dict:
    """Mean over rows, with a target-gene bootstrap (R2's draw order)."""
    unique, inverse = np.unique(genes, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for r in range(B):
        picked = rng.integers(0, len(unique), size=len(unique))
        draws[r] = float(np.mean(values[np.concatenate([members[p] for p in picked])]))
    lo, hi, _ = percentile_interval(draws)
    return {"estimate": float(np.mean(values)), "lo": lo, "hi": hi, "rows": int(values.size), "target_genes": int(len(unique))}


def run_vcc() -> None:
    import make_e1a_readouts as readouts

    started = time.time()
    counts = load_counts()
    inputs = prepare_vcc(counts)
    DATA["rebuilt_check_rows"] = 50
    tasks = []
    for cell in VCC_CELLS:  # the smoke takes rows spread over the table, so every split has some
        count = len(DATA["cells"][cell]["rows"])
        ks = range(count) if not SMOKE else sorted({int(k) for k in np.linspace(0, count - 1, SMOKE_ROWS)})
        tasks += [(cell, k) for k in ks]
    print("[s1 vcc] %d rows, %.0f s" % (len(tasks), time.time() - started), flush=True)
    results = parallel(vcc_row_task, tasks)
    print("[s1 vcc] labels done, %.0f s" % (time.time() - started), flush=True)
    a_rows, b_rows, curve_rows, checks = [], [], [], {}
    for cell in VCC_CELLS:
        mine = [r for r in results if r["cell"] == cell]
        labels = DATA["cells"][cell]["labels"]
        stored = labels["existing_label"].to_numpy(dtype=np.float64)[[r["k"] for r in mine]]
        full = np.array([r["full"] for r in mine], dtype=np.float64)
        largest = float(np.nanmax(np.abs(full - stored)))
        if not largest <= EXISTING_TOLERANCE or np.isnan(full).sum() != np.isnan(stored).sum():
            raise Stop("%s: the all-cells builder reproduces the stored existing label only to %g" % (cell, largest))
        simulation = DATA["cells"][cell]["simulation"]
        rows = DATA["cells"][cell]["rows"]
        sim_mine = pd.DataFrame([{"k": r["k"], **s} for r in mine for s in r["simulation"]])
        key = rows[["split", "batch", "target_gene"]].astype(str).agg("|".join, axis=1).to_numpy()
        sim_mine["row"] = key[sim_mine["k"].to_numpy()] if len(sim_mine) else []
        stored_sim = simulation.assign(row=simulation[["split", "batch", "target_gene"]].astype(str).agg("|".join, axis=1))
        stored_sim = stored_sim[stored_sim["row"].isin(set(sim_mine["row"]))].reset_index(drop=True)
        if not np.array_equal(stored_sim["size"].to_numpy(), sim_mine["size"].to_numpy()):
            raise Stop("%s: R2's simulated sizes are not reproduced" % cell)
        checked = sim_mine[sim_mine["k"] < DATA["rebuilt_check_rows"]]
        reference = stored_sim.loc[checked.index, "y"].to_numpy(dtype=np.float64)
        rebuilt_diff = float(np.nanmax(np.abs(checked["rebuilt"].to_numpy(dtype=np.float64) - reference))) if len(checked) else 0.0
        if not rebuilt_diff <= TOLERANCE:
            raise Stop("%s: the rebuilt simulated labels are reproduced only to %g" % (cell, rebuilt_diff))
        checks[cell] = {"existing_label_full_pool_largest_difference": largest, "tolerance": EXISTING_TOLERANCE, "rows": len(mine),
                        "simulated_sizes_equal_pool_size_simulation_csv": True, "rebuilt_simulated_labels_checked": int(len(checked)),
                        "rebuilt_simulated_largest_difference": rebuilt_diff}
        # (a)
        part_labels = labels.iloc[[r["k"] for r in mine]].reset_index(drop=True)
        saved = pd.read_csv(READOUTS / "R2_between.csv", float_precision="round_trip")
        for split in SPLITS + ("all",):
            p = part_labels if split == "all" else part_labels[part_labels["split"] == split]
            value = paired_corr(p["n_pool"].to_numpy(dtype=np.float64), p["existing_label"].to_numpy(dtype=np.float64), p["y"].to_numpy(dtype=np.float64),
                                p["target_gene"].to_numpy())
            separate = {}
            for name, column in (("existing label against pool size, same rows", "existing_label"), ("rebuilt label against pool size", "y")):
                est = stats.cluster_bootstrap_corr(p[column].to_numpy(dtype=np.float64), p["n_pool"].to_numpy(dtype=np.float64), p["target_gene"].to_numpy())
                separate[column] = est
                if not SMOKE:
                    hit = saved[(saved["cell"] == cell) & (saved["split"] == split) & (saved["label"] == name)]
                    for c, v in (("spearman", est.spearman), ("spearman_lo", est.spearman_lo), ("spearman_hi", est.spearman_hi)):
                        if len(hit) != 1 or not abs(float(hit[c].iloc[0]) - v) <= TOLERANCE:
                            raise Stop("%s, %s, %s: %s does not reproduce R2_between.csv" % (cell, split, name, c))
            if abs(value["first"] - separate["existing_label"].spearman) > TOLERANCE or abs(value["second"] - separate["y"].spearman) > TOLERANCE:
                raise Stop("%s, %s: the paired point estimates differ from the separate ones" % (cell, split))
            a_rows.append({"analysis": "(a)", "cell": cell, "split": split, "quantity": "rho(existing label, n_pool) - rho(rebuilt label, n_pool)", **value,
                           "existing_lo": separate["existing_label"].spearman_lo, "existing_hi": separate["existing_label"].spearman_hi,
                           "rebuilt_lo": separate["y"].spearman_lo, "rebuilt_hi": separate["y"].spearman_hi, "state": state(value["lo"], value["hi"]),
                           "descriptive": bool(value["target_genes"] < 50), "reading": READ_A.get(state(value["lo"], value["hi"]), ""), "role": STATUS})
        checks[cell]["R2_between_csv_reproduced"] = not SMOKE
        # (b)
        single = part_labels[(part_labels["n_guides"] == 1) & part_labels["existing_label"].notna()].copy()
        single["group"] = single["target_gene"] + "|" + single["guides"].astype(str)
        single = single[single.groupby("group")["existing_label"].transform("size") >= 3]
        single["c_y"] = centred(single["existing_label"].to_numpy(), single["group"].to_numpy())
        single["c_n"] = centred(single["n_pool"].to_numpy(float), single["group"].to_numpy())
        sim = sim_mine.merge(rows[["split", "target_gene"]].reset_index().rename(columns={"index": "k"}), on="k", how="left")
        sim = sim[np.isfinite(sim["existing"].to_numpy(dtype=np.float64))].copy()
        sim["c_y"] = centred(sim["existing"].to_numpy(dtype=np.float64), sim["row"].to_numpy())
        sim["c_n"] = centred(sim["size"].to_numpy(float), sim["row"].to_numpy())
        for split in SPLITS + ("all",):
            o = single if split == "all" else single[single["split"] == split]
            s_ = sim if split == "all" else sim[sim["split"] == split]
            m = mechanical(o, s_)
            b_rows.append({"analysis": "(b)", "cell": cell, "split": split, "label": "existing (all-cells) label", **m,
                           "state_mechanical": state(m["mechanical_lo"], m["mechanical_hi"]), "state_excess": state(m["excess_lo"], m["excess_hi"]),
                           "reading": "" if split != "all" else "; ".join(x for x in (
                               "the all-cells label rises with the number of perturbed cells of the same row by construction"
                               if m["mechanical_lo"] > 0 else "",
                               {"contains zero": "the all-cells label's within-gene association with pool size is not shown beyond its mechanical part",
                                "above zero": "the all-cells label's within-gene association with pool size exceeds its mechanical part",
                                "below zero": "the all-cells label's within-gene association with pool size is below its mechanical part"}.get(
                                   state(m["excess_lo"], m["excess_hi"]), "")) if x), "role": STATUS})
        # the curve
        if cell == "G6" and any(r["curve"] is not None for r in mine):
            curved = [r for r in mine if r["curve"] is not None]
            genes = rows["target_gene"].to_numpy()[[r["k"] for r in curved]]
            for label in ("existing", "rebuilt"):
                full_values = np.array([r["curve"]["full"][label] for r in curved], dtype=np.float64)
                q75, q25 = np.nanpercentile(full_values, [75, 25])
                iqr = float(q75 - q25)
                for m_ in CURVE_SIZES:
                    shift = (np.array([r["curve"][m_][label] for r in curved], dtype=np.float64) - full_values) / iqr
                    keep = np.isfinite(shift)
                    value = gene_bootstrap_mean(shift[keep], genes[keep])
                    curve_rows.append({"analysis": "(b), curve", "cell": cell, "label": "existing (all-cells) label" if label == "existing" else "G6 rebuilt label",
                                       "m": m_, "quantity": "mean of (label at m - label on the full pool) / between-gene IQR", **value,
                                       "between_gene_iqr": iqr, "rows_with_at_least": CURVE_FLOOR, "role": "descriptive; " + STATUS})
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(a_rows).to_csv(folder / "vcc_paired.csv", index=False)
    pd.DataFrame(b_rows).to_csv(folder / "vcc_mechanical.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(folder / "vcc_curve.csv", index=False)
    write_record(folder, "vcc", started, ["vcc_paired.csv", "vcc_mechanical.csv", "vcc_curve.csv"],
                 {"checks": checks, "inputs": {"count_cache": DATA["counts_record"], **inputs, "R2_between.csv": file_record(READOUTS / "R2_between.csv")},
                  "readouts_code": sha256(Path(readouts.__file__).resolve())})
    print("[s1 vcc] written, %.0f s" % (time.time() - started), flush=True)


# --------------------------------------------------------------------------- (c)


def prepare_external(counts: dict) -> dict:
    import make_e1b_stats as e1s
    import make_e2_stats as st

    inputs = st.load_cells()
    DATA["draws"] = e1s.STATE["draws"]
    DATA["external"] = {}
    for screen in builder.SCREENS:
        arrays = counts["screen_" + screen]
        order = np.argsort(arrays["pert_code"], kind="stable")
        codes = arrays["pert_code"][order]
        bounds = np.searchsorted(codes, np.arange(arrays["constructs"].size + 1))
        pools = [order[bounds[i] : bounds[i + 1]] for i in range(arrays["constructs"].size)]
        by_construct = {c: i for i, c in enumerate(arrays["constructs"].tolist())}
        values = builder.screen_values(counts, screen, "all")
        names = e1s.DATA["external"][(screen, e1s.DATA["cells"]["G5"].external_source[screen])]["construct"].astype(str).to_numpy()
        # the size-fixed comparison labels are 27/788 and 10/788 (G1's and G2's external tables, plan Sections 0 and 3); E1b's G5
        # and G6 external labels are 27/700 and 10/700 and serve only (d)'s reproduction of external.csv
        fixed = {}
        for cell in ("G1", "G2"):
            table, record = load_table(cell, f"external_{screen}.csv")
            if not np.array_equal(table["construct"].astype(str).to_numpy(), names):
                raise Stop("%s: %s's external constructs differ from E1b's" % (screen, cell))
            fixed[cell] = table
            inputs.setdefault("external_tables", {})[f"{cell}/{screen}"] = record
        index = np.array([by_construct[c] for c in names])
        DATA["external"][screen] = {"names": names, "pools": [pools[i] for i in index], "slot": arrays["construct_slot"][index], "values": values,
                                    "y27": fixed["G1"]["y"].to_numpy(dtype=np.float64), "y10": fixed["G2"]["y"].to_numpy(dtype=np.float64),
                                    "n_pool": fixed["G1"]["n_pool"].to_numpy(dtype=np.float64)}
    return inputs


def construct_task(args) -> dict:
    screen, k = args
    E = DATA["external"][screen]
    pool = E["values"]["pert"][E["pools"][k]]
    controls = E["values"]["nt"][:, E["slot"][k]]
    out = {"screen": screen, "k": k, "n": int(pool.size),
           "y27_check": endpoint.subsampled_label(pool, controls, 27, N_CTRL_EXTERNAL)[0]["y"],
           "all_788": endpoint.subsampled_label(pool, controls, pool.size, N_CTRL_EXTERNAL)[0]["y"],
           "all_controls": endpoint.ad_log1p(pool, controls), "sub": None}
    if pool.size >= SUB_FLOOR:
        order = np.random.default_rng([SUB_SEED, SCREEN_INDEX[screen], k]).permutation(pool.size)
        sub = {}
        for m in SUB_SIZES:
            cells = pool[order[:m]]
            sub[m] = {"all": endpoint.subsampled_label(cells, controls, m, N_CTRL_EXTERNAL)[0]["y"],
                      "fixed27": endpoint.subsampled_label(cells, controls, 27, N_CTRL_EXTERNAL)[0]["y"]}
        sub["full"] = {"all": out["all_788"], "fixed27": out["y27_check"]}
        out["sub"] = sub
    return out


def weighted_rho(x: np.ndarray, y: np.ndarray, screen: str) -> tuple[float, np.ndarray]:
    draws = DATA["draws"][screen]
    return float(resample.weighted_spearman(x[:, None], y, draws["point"])[0, 0]), resample.weighted_spearman(x[:, None], y, draws["draw"])[:, 0]


def summary(point: float, draws: np.ndarray) -> dict:
    lo, hi, missing = percentile_interval(draws)
    return {"estimate": point, "lo": lo, "hi": hi, "p": resample.p_value(draws), "se": float(np.nanstd(draws, ddof=1)), "state": state(lo, hi),
            "replicates_not_finite": missing}


def run_external() -> None:
    started = time.time()
    counts = load_counts()
    inputs = prepare_external(counts)
    tasks = []
    for s in builder.SCREENS:  # the smoke takes constructs spread over the table
        count = len(DATA["external"][s]["names"])
        tasks += [(s, k) for k in (range(count) if not SMOKE else sorted({int(k) for k in np.linspace(0, count - 1, SMOKE_CONSTRUCTS)}))]
    print("[s1 external] %d constructs, %.0f s" % (len(tasks), time.time() - started), flush=True)
    results = parallel(construct_task, tasks)
    print("[s1 external] labels done, %.0f s" % (time.time() - started), flush=True)
    label_rows, assoc_rows, sub_rows, checks = [], [], [], {}
    for screen in builder.SCREENS:
        E = DATA["external"][screen]
        mine = sorted((r for r in results if r["screen"] == screen), key=lambda r: r["k"])
        ks = np.array([r["k"] for r in mine])
        y27 = E["y27"][ks]
        recomputed = np.array([r["y27_check"] for r in mine], dtype=np.float64)
        largest = float(np.nanmax(np.abs(recomputed - y27)))
        if not largest <= TOLERANCE or np.isnan(recomputed).sum() != np.isnan(y27).sum():
            raise Stop("%s: the builder reproduces the 27/788 external label only to %g" % (screen, largest))
        n = np.array([r["n"] for r in mine], dtype=np.float64)
        if not np.array_equal(n, E["n_pool"][ks]):
            raise Stop("%s: the eligible perturbed cells differ from the external table's n_pool" % screen)
        checks[screen] = {"y27_reproduced_largest_difference": largest, "constructs": len(mine), "tolerance": TOLERANCE,
                          "eligible_cells_equal_n_pool": True}
        y_all = np.array([r["all_788"] for r in mine], dtype=np.float64)
        y_ctrl = np.array([r["all_controls"] for r in mine], dtype=np.float64)
        for r, a, c in zip(mine, y_all, y_ctrl):
            label_rows.append({"screen": screen, "construct": E["names"][r["k"]], "n_eligible": r["n"], "y_all_cells_788": a, "y_all_cells_all_controls": c,
                               "y_27_788": E["y27"][r["k"]], "y_10_788": E["y10"][r["k"]]})
        if SMOKE:
            continue  # the draws are over every construct; the smoke checks the labels only
        values = {"all cells against 788 pooled controls": y_all, "all cells against all pooled controls": y_ctrl, "27 against 788": y27,
                  "10 against 788": E["y10"]}
        rhos = {name: weighted_rho(n, v, screen) for name, v in values.items()}
        for name, (point, draws) in rhos.items():
            assoc_rows.append({"analysis": "(c)", "screen": screen, "quantity": f"rho({name} label, n)", "constructs": len(n), **summary(point, draws),
                               "reading": "", "role": STATUS})
        for name in ("all cells against 788 pooled controls", "all cells against all pooled controls"):
            for fixed in ("27 against 788", "10 against 788"):
                point = rhos[name][0] - rhos[fixed][0]
                s = summary(point, rhos[name][1] - rhos[fixed][1])
                assoc_rows.append({"analysis": "(c)", "screen": screen, "quantity": f"rho({name} label, n) - rho({fixed} label, n)", "constructs": len(n), **s,
                                   "reading": READ_C.get(s["state"], "") if (name.endswith("788 pooled controls") and fixed == "27 against 788") else "",
                                   "role": STATUS})
        sub = [r for r in mine if r["sub"] is not None]
        weights = DATA["draws"][screen]["draw"].weights[:, [r["k"] for r in sub]]
        for label in (("all", "fixed27") if sub else ()):
            full_values = np.array([r["sub"]["full"][label] for r in sub], dtype=np.float64)
            q75, q25 = np.nanpercentile(full_values, [75, 25])
            iqr = float(q75 - q25)
            for m in SUB_SIZES:
                shift = (np.array([r["sub"][m][label] for r in sub], dtype=np.float64) - full_values) / iqr
                keep = np.isfinite(shift)
                w = weights[:, keep]
                draws = (w @ shift[keep]) / w.sum(axis=1)
                s = summary(float(np.mean(shift[keep])), draws)
                sub_rows.append({"analysis": "(c), subsampling", "screen": screen, "label": "all eligible cells against 788" if label == "all" else "27 against 788",
                                 "m": m, "quantity": "mean of (label at m - label on the full pool) / between-construct IQR", "constructs": int(keep.sum()),
                                 "constructs_with_at_least": SUB_FLOOR, "between_construct_iqr": iqr, **s, "role": "descriptive; " + STATUS})
            centred_y, centred_m = [], []
            for r in sub:
                ys = np.array([r["sub"][m][label] for m in SUB_SIZES], dtype=np.float64)
                lm = np.log(np.array(SUB_SIZES, dtype=np.float64))
                ok = np.isfinite(ys)
                centred_y.append(np.where(ok, ys - np.nanmean(ys), np.nan))
                centred_m.append(np.where(ok, lm - lm[ok].mean(), np.nan))
            cy, cm = np.concatenate(centred_y), np.concatenate(centred_m)
            rep = np.repeat(weights, len(SUB_SIZES), axis=1)
            ok = np.isfinite(cy)
            point_draw = resample.ClusterDraw(weights=np.ones((1, int(ok.sum()))), clusters=np.array([]), seed=-1, index_sha256="")
            boot = resample.ClusterDraw(weights=rep[:, ok], clusters=np.array([]), seed=-1, index_sha256=DATA["draws"][screen]["draw"].index_sha256)
            point = float(resample.weighted_spearman(cm[ok][:, None], cy[ok], point_draw)[0, 0])
            s = summary(point, resample.weighted_spearman(cm[ok][:, None], cy[ok], boot)[:, 0])
            sub_rows.append({"analysis": "(c), subsampling", "screen": screen, "label": "all eligible cells against 788" if label == "all" else "27 against 788",
                             "m": "within construct", "quantity": "Spearman of the centred label against the centred log m (mechanical)", "constructs": len(sub),
                             "constructs_with_at_least": SUB_FLOOR, **s, "role": "descriptive; " + STATUS})
        checks[screen]["draws_index_sha256"] = DATA["draws"][screen]["draw"].index_sha256
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(label_rows).to_csv(folder / "external_labels.csv", index=False)
    outputs = ["external_labels.csv"]
    if not SMOKE:
        pd.DataFrame(assoc_rows).to_csv(folder / "external_association.csv", index=False)
        pd.DataFrame(sub_rows).to_csv(folder / "external_subsampling.csv", index=False)
        outputs += ["external_association.csv", "external_subsampling.csv"]
    write_record(folder, "external", started, outputs, {"checks": checks, "inputs": {"count_cache": DATA["counts_record"], **inputs}})
    print("[s1 external] written, %.0f s" % (time.time() - started), flush=True)


# --------------------------------------------------------------------------- (d)


def run_control() -> None:
    import make_e1b_stats as e1s
    import make_e2_null_external as ne
    import make_e2_stats as st

    started = time.time()
    folder = out_dir()
    record = json.loads((folder / "s1_supplement_external.json").read_text())
    if sha256(folder / "external_labels.csv") != record["outputs"]["external_labels.csv"]["sha256"]:
        raise Stop("external_labels.csv does not match its record")
    labels = pd.read_csv(folder / "external_labels.csv", float_precision="round_trip")
    e1s.WORKERS = WORKERS
    st.DATA["missing"] = []
    st.DATA["order_marker"] = st.order_marker()
    predictions = st.load_predictions()
    inputs = st.load_cells()
    ne.LABEL = "existing"
    null_manifest = ne.outputs_dir() / "predictions_manifest.json"
    manifest = json.loads(null_manifest.read_text())
    if manifest["failed"] or manifest["completed"] != manifest["fits"] or not manifest["vcc_predictions_equal_to_the_saved_n0_fits"]:
        raise Stop("the all-cells-label null fits are not complete or their check did not pass")
    lookup = st.DATA["lookup"]
    st.DATA["lookup"] = st.Lookup(lookup.e1b, lookup.e2 + [e1s.Predictions(null_manifest)], {**st.e2_names(), **{ne.name_of(e): e for e in ne.ENCODERS}})
    DATA["draws"] = e1s.STATE["draws"]
    recorded = json.loads((REPO / "bioinf2027" / "results" / "E1" / "stats.json").read_text())
    external = pd.read_csv(REPO / "bioinf2027" / "results" / "E2" / "stats" / "external.csv", float_precision="round_trip")
    external = external[(external["label"] == "existing") & (external["head"] == "forest")]
    rows, compared = [], 0
    for screen in builder.SCREENS:
        table = e1s.DATA["external"][(screen, e1s.DATA["cells"]["G5"].external_source[screen])]
        names = table["construct"].astype(str).to_numpy()
        mine = labels[labels["screen"] == screen].set_index("construct")
        if SMOKE:  # the smoke labels only some constructs; the others take the size-fixed label so the code path runs
            fill = pd.DataFrame({"y_all_cells_788": table["y"].to_numpy(dtype=np.float64), "n_eligible": table["n_pool"].to_numpy(dtype=np.float64)}, index=names)
            fill.update(mine[["y_all_cells_788", "n_eligible"]])
            mine = fill
        if len(mine) != len(names) or set(mine.index) != set(names):
            raise Stop("%s: %d all-cells labels for %d constructs" % (screen, len(mine), len(names)))
        y_all = mine.loc[names, "y_all_cells_788"].to_numpy(dtype=np.float64)
        n = mine.loc[names, "n_eligible"].to_numpy(dtype=np.float64)
        point, draws = weighted_rho(n, y_all, screen)
        rows.append({"analysis": "(d)", "encoder": "", "cell": "", "screen": screen, "quantity": "rho(n, all-cells label)", **summary(point, draws),
                     "reading": "the cell-count baseline on the all-cells label", "role": STATUS, "markers": ""})
        for cell in ("G5", "G6"):
            y_fixed = e1s.DATA["external"][(screen, e1s.DATA["cells"][cell].external_source[screen])]["y"].to_numpy(dtype=np.float64)
            for encoder in ne.ENCODERS:
                names_by_role = st.roles(encoder)
                per = {}
                for role, representation in (("e", names_by_role["e"]), ("C1", names_by_role["C1"]), ("null", ne.name_of(encoder))):
                    seeds = st.wanted_seeds(representation, "forest") if role != "null" else ne.SEEDS
                    values = [st.DATA["lookup"].get(cell, "existing", representation, "forest", s) for s in seeds]
                    if any(v is None or screen not in v for v in values):
                        raise Stop("%s, %s, %s: a saved all-cells-label prediction of %s is missing" % (encoder, cell, screen, representation))
                    preds = [np.asarray(v[screen], dtype=np.float64) for v in values]
                    pairs = [weighted_rho(p, y_all, screen) for p in preds]
                    per[role] = (float(np.mean([a for a, _ in pairs])), np.mean(np.vstack([b for _, b in pairs]), axis=0))
                    if role != "null":
                        fixed = [weighted_rho(p, y_fixed, screen)[0] for p in preds]
                        quantity = "rho(%s)" % representation
                        hit = external[(external.encoder == encoder) & (external.cell == cell) & (external.screen == screen) & (external.quantity == quantity)]
                        if len(hit) != 1 or not abs(float(np.mean(fixed)) - float(hit["estimate"].iloc[0])) <= TOLERANCE:
                            raise Stop("%s, %s, %s: %s on the size-fixed label does not reproduce external.csv" % (encoder, cell, screen, quantity))
                        compared += 1
                markers = st.markers(encoder, "Replogle")
                for role in ("null", "e"):
                    s = summary(per[role][0] - per["C1"][0], per[role][1] - per["C1"][1])
                    rows.append({"analysis": "(d)", "encoder": encoder, "cell": cell, "screen": screen,
                                 "quantity": "rho(%s, all-cells label) - rho(%s, all-cells label)" % (ne.name_of(encoder) if role == "null" else names_by_role["e"],
                                                                                                     names_by_role["C1"]), **s,
                                 "reading": READ_NULL.get(s["state"], "") if role == "null" else "", "role": STATUS, "markers": markers})
                for role in ("null", "e", "C1"):
                    rows.append({"analysis": "(d)", "encoder": encoder, "cell": cell, "screen": screen,
                                 "quantity": "rho(%s, all-cells label)" % {"null": ne.name_of(encoder), "e": names_by_role["e"], "C1": names_by_role["C1"]}[role],
                                 **summary(*per[role]), "reading": "", "role": STATUS, "markers": markers})
    digests = {s: DATA["draws"][s]["draw"].index_sha256 for s in builder.SCREENS}
    if json.dumps(recorded).find(digests["RPE1"]) < 0 or json.dumps(recorded).find(digests["K562-essential"]) < 0:
        raise Stop("the external draws are not E1b's")
    pd.DataFrame(rows).to_csv(folder / "external_control.csv", index=False)
    write_record(folder, "control", started, ["external_control.csv"], {
        "checks": {"draws_equal_E1b": True, "rho_e_and_C1_on_the_size_fixed_label_equal_external_csv": {"compared": compared, "tolerance": TOLERANCE}},
        "inputs": {**inputs, "predictions": predictions, "null_predictions": file_record(null_manifest),
                   "external_labels.csv": file_record(folder / "external_labels.csv")}, "external_replicate_index_sha256": digests})
    print("[s1 control] %d rows, %.0f s" % (len(rows), time.time() - started), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
        B = 200
    if "--workers" in arguments:
        i = arguments.index("--workers")
        WORKERS = int(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    parts = []
    while "--part" in arguments:
        i = arguments.index("--part")
        parts.append(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if arguments or not set(parts) <= set(PARTS) or len(set(parts)) != len(parts):
        raise SystemExit("usage: make_s1_supplement.py [--smoke] [--workers N] [--part vcc|external|control ...]")
    for part in parts or PARTS:
        {"vcc": run_vcc, "external": run_external, "control": run_control}[part]()
