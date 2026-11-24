"""E5, endpoint (b): the target-excluded energy distance at fixed cell numbers, its sensitivities and its gates.

Design: bioinf2027/plans/E5.md, Sections 4, 5, 16.1 and 17 (fixed before any value). For each construct of the
two Replogle screens with at least 27 eligible cells: the energy distance between its perturbed cells and the
control cells on the first 50 control-cell components of the 1,303 features both screens map, the target
removed from the features and from every cell's normalisation total, at 27 (and, pools of at least 50, at 50)
perturbed cells against 788 controls, on the cells of E1's G1 draws; its halves (E1's keys), the composition
null and the gem-group-matched version, the depth-matched and full-space and 10 kb sensitivities, the all-cells
version and the n-scaled statistic, and the SAC-label approximation on the first draw. Then the gene values,
the composition rule, the reliability and beyond-Bc gates, and the version each family setting uses, written
before any fit. No embedding is read.

Inputs: the E4 count cache and the E1a label cache (each file checked against its manifest), the screens'
endpoint tables, and the gem groups and gene positions of the two screen files (their digests checked first).
Outputs: bioinf2027/results/E5/endpoint_b/. With --smoke: 24 constructs per screen, outputs to
$IGA_STORAGE_ROOT/bioinf2027/scratch/E5_endpoint_b_smoke/, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, endpoint, features, singlecell  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import e5lib  # noqa: E402
import make_e1a_extract as first  # noqa: E402
import make_e1a_label_extract as second  # noqa: E402
import make_e1a_labels as e1labels  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E5" / "endpoint_b"
PLAN = "bioinf2027/plans/E5.md, Sections 4, 5, 16.1 and 17"
SCREENS = ("RPE1", "K562-essential")
N_CTRL = 788
LEVELS = (27, 50)
HALF_FLOOR = {27: 54, 50: 100}
N_PC = 50
WINDOW = 10_000
FEATURES_IN_BOTH = 1303
COMPOSITION_RULE = 0.25
RELIABILITY_GATE = 0.5
BEYOND_BC_GATE = 0.2
SETTINGS = {"within RPE1": "RPE1", "within K562-essential": "K562-essential",
            "K562-essential to RPE1": "RPE1", "RPE1 to K562-essential": "K562-essential"}
WORKERS = 6
SMOKE = False
SMOKE_CONSTRUCTS = 24
CHECK_CONSTRUCTS = 20
DATA: dict = {}


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E5_endpoint_b_smoke" if SMOKE else OUT


def e4_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "cache" / "E4"


def checked(folder: Path, name: str, record: dict) -> dict:
    path = folder / f"{name}.npz"
    if sha256(path) != record["sha256"]:
        raise Stop("check (i): %s does not match its manifest" % path.name)
    return np.load(path, allow_pickle=False)


# --------------------------------------------------------------------------- inputs


def feature_mask() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((e4_dir() / "manifest.json").read_text())
    columns = {}
    for screen in SCREENS:
        with checked(e4_dir(), "screen_" + screen, manifest["cache"]["screen_" + screen]) as store:
            columns[screen] = store["feature_columns"]
    both = np.logical_and.reduce([columns[s] >= 0 for s in SCREENS])
    if int(both.sum()) != FEATURES_IN_BOTH:
        raise Stop("the features both screens map are %d, not %d" % (both.sum(), FEATURES_IN_BOTH))
    return both, features.hvg_symbols()[both]


def load_screen(screen: str, both: np.ndarray, symbols: np.ndarray) -> dict:
    import h5py

    m1 = json.loads((second.cache_dir() / "manifest.json").read_text())
    m4 = json.loads((e4_dir() / "manifest.json").read_text())
    with checked(second.cache_dir(), "screen_" + screen, m1["cache"]["screen_" + screen]) as store:
        lab = {k: store[k] for k in ("constructs", "construct_col", "construct_slot", "nt_rows", "nt_counts", "nt_total",
                                     "pert_rows", "pert_code", "pert_count", "pert_total")}
    with checked(e4_dir(), "screen_" + screen, m4["cache"]["screen_" + screen]) as store:
        if not (np.array_equal(store["pert_rows"], lab["pert_rows"]) and np.array_equal(store["nt_rows"], lab["nt_rows"])):
            raise Stop("check (v): the E4 cache's rows of %s differ from the E1a label cache's" % screen)
        pert_f = np.ascontiguousarray(store["pert_counts"][:, both])
        nt_f = np.ascontiguousarray(store["nt_counts"][:, both])
    table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
    order = np.argsort(lab["pert_code"], kind="stable")
    codes = lab["pert_code"][order]
    bounds = np.searchsorted(codes, np.arange(lab["constructs"].size + 1))
    pools = [order[bounds[i]: bounds[i + 1]] for i in range(lab["constructs"].size)]
    by_construct = {c: i for i, c in enumerate(lab["constructs"].tolist())}
    index = np.array([by_construct[c] for c in table["construct"].astype(str)])
    with h5py.File(first.screen_files()[screen], "r") as handle:
        gem = np.asarray(singlecell.read_annotation(handle["obs"], "gem_group")).astype(str)
        if all(g.isdigit() for g in set(gem.tolist())):
            gem = gem.astype(np.int64)  # ascending order of the identifier is numeric order
        var = {k: np.asarray(singlecell.read_annotation(handle["var"], k)) for k in ("gene_name", "gene_id", "chr", "start", "end")}
    position = {s: i for i, s in enumerate(symbols.tolist())}
    construct_col = lab["construct_col"][index]
    target_symbol = var["gene_name"][construct_col].astype(str)
    if not np.array_equal(var["gene_id"][construct_col].astype(str), table["target_ensg"].astype(str).to_numpy()):
        raise Stop("%s: the target columns do not carry the target identifiers" % screen)
    chrom = var["chr"].astype(str)
    start = var["start"].astype(np.int64)
    end = var["end"].astype(np.int64)
    neighbours = []
    for c in construct_col:
        near = np.flatnonzero((chrom == chrom[c]) & (start <= end[c] + WINDOW) & (end >= start[c] - WINDOW))
        names = {str(var["gene_name"][g]) for g in near if g != c} - {str(var["gene_name"][c])}
        neighbours.append(np.array(sorted(position[n] for n in names if n in position), dtype=np.int64))
    target_feature = np.array([position.get(s, -1) for s in target_symbol], dtype=np.int64)
    nt_gem = gem[lab["nt_rows"]]
    return {
        "screen": screen, "table": table, "pools": [pools[i] for i in index], "slot": lab["construct_slot"][index],
        "pert_f": pert_f, "nt_f": nt_f, "pert_total": lab["pert_total"], "nt_total": lab["nt_total"],
        "pert_count": lab["pert_count"], "nt_counts": lab["nt_counts"], "pert_gem": gem[lab["pert_rows"]], "nt_gem": nt_gem,
        "ctrl_by_gem": {g: np.flatnonzero(nt_gem == g) for g in sorted(set(nt_gem.tolist()))},
        "decile_members": None,
        "target_symbol": target_symbol, "target_feature": target_feature, "neighbours": neighbours,
        "symbol_mismatches": int((target_symbol != table["target_symbol"].astype(str).to_numpy()).sum()),
        "control_halves": endpoint.half_split(lab["nt_rows"].size, e1labels.HALF_CONTROLS_SEED),
        "deciles": np.quantile(lab["nt_total"].astype(np.float64), np.arange(1, 10) / 10.0),
    }


def add_deciles(S: dict) -> None:
    decile = np.searchsorted(S["deciles"], S["nt_total"].astype(np.float64), side="right")
    S["decile_members"] = {str(d): np.flatnonzero(decile == d) for d in range(10)}


# --------------------------------------------------------------------------- one construct


def _pooled(rng, n_pool, n, n_ctrl):
    return rng.choice(n_pool, size=n, replace=False), rng.choice(n_ctrl, size=N_CTRL, replace=False)


def _by_group(rng, groups_needed: list[tuple[str, int]], members: dict) -> np.ndarray | None:
    out = []
    for g, need in groups_needed:
        pool = members.get(g)
        if need == 0:
            continue
        if pool is None or pool.size < need:
            return None
        out.append(pool[rng.choice(pool.size, size=need, replace=False)])
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def level_values(S: dict, k: int, n: int, Xs, Ys, Xv, Yv, neighbour_scores) -> dict:
    pool = S["pools"][k]
    n_pool, n_ctrl = pool.size, Ys.shape[0]
    gem_p = S["pert_gem"][pool]
    groups, counts = np.unique(gem_p, return_counts=True)
    alloc = e5lib.largest_remainder(counts, N_CTRL)
    edges = S["deciles"]
    dec_members = S["decile_members"]
    draws = {name: np.full(len(endpoint.SEEDS), np.nan) for name in ("b", "full", "null", "gem", "depth", "10kb")}
    for i, seed in enumerate(endpoint.SEEDS):
        f, s = endpoint.draw(seed, n_pool, n, n_ctrl, N_CTRL)
        draws["b"][i] = e5lib.energy_distance(Xs[f], Ys[s])
        draws["full"][i] = e5lib.energy_distance_gram(Xv[f], Yv[s])
        if neighbour_scores is not None:
            draws["10kb"][i] = e5lib.energy_distance(neighbour_scores[0][f], neighbour_scores[1][s])
        else:
            draws["10kb"][i] = draws["b"][i]  # no neighbour among the features: the same statistic
        # composition null: the drawn cells' gem groups, as many controls of each, then 788 from the rest
        rng = np.random.default_rng(seed)
        f2 = rng.choice(n_pool, size=n, replace=False)
        g_drawn, c_drawn = np.unique(gem_p[f2], return_counts=True)
        slots = _by_group(rng, list(zip(g_drawn.tolist(), c_drawn.tolist())), S["ctrl_by_gem"])
        if slots is not None:
            rest = np.delete(np.arange(n_ctrl), slots)
            controls = rest[rng.choice(rest.size, size=N_CTRL, replace=False)]
            draws["null"][i] = e5lib.energy_distance(Ys[slots], Ys[controls])
        # gem-group-matched controls, in proportion to the construct's eligible cells
        rng = np.random.default_rng(seed)
        f3 = rng.choice(n_pool, size=n, replace=False)
        matched = _by_group(rng, list(zip(groups.tolist(), alloc.tolist())), S["ctrl_by_gem"])
        if matched is not None:
            draws["gem"][i] = e5lib.energy_distance(Xs[f3], Ys[matched])
        # depth-matched controls, per decile of the control totals, in proportion to the drawn cells' deciles
        rng = np.random.default_rng(seed)
        f4 = rng.choice(n_pool, size=n, replace=False)
        dec_p = np.searchsorted(edges, S["pert_total"][pool][f4].astype(np.float64), side="right")
        need = e5lib.largest_remainder(np.bincount(dec_p, minlength=10), N_CTRL)
        depth = _by_group(rng, [(str(d), int(need[d])) for d in range(10)], dec_members)
        if depth is not None:
            draws["depth"][i] = e5lib.energy_distance(Xs[f4], Ys[depth])
    out = {}
    for name, values in draws.items():
        summary = endpoint.summarise_draws(values)
        out[f"b{n}_{name}" if name != "b" else f"b{n}"] = summary["y"]
        out[f"b{n}_{name}_valid" if name != "b" else f"b{n}_valid"] = int(np.isfinite(values).sum())
    out[f"b{n}_draw1"] = float(draws["b"][0])
    # halves (E1's keys), pooled and gem-group-matched controls from each control half
    if n_pool >= HALF_FLOOR[n]:
        halves = endpoint.half_split(n_pool, [e1labels.HALF_CELLS_SEED, k])
        for name, part, ctrl in (("A", halves[0], S["control_halves"][0]), ("B", halves[1], S["control_halves"][1])):
            pooled = np.full(len(endpoint.SEEDS), np.nan)
            matched_half = np.full(len(endpoint.SEEDS), np.nan)
            members = {g: np.flatnonzero(S["nt_gem"][ctrl] == g) for g in groups.tolist()}
            for i, seed in enumerate(endpoint.SEEDS):
                f, s = endpoint.draw(seed, part.size, n, ctrl.size, N_CTRL)
                pooled[i] = e5lib.energy_distance(Xs[part[f]], Ys[ctrl[s]])
                rng = np.random.default_rng(seed)
                f5 = rng.choice(part.size, size=n, replace=False)
                g_half, c_half = np.unique(gem_p[part], return_counts=True)
                chosen = _by_group(rng, list(zip(g_half.tolist(), e5lib.largest_remainder(c_half, N_CTRL).tolist())), members)
                if chosen is not None:
                    matched_half[i] = e5lib.energy_distance(Xs[part[f5]], Ys[ctrl[chosen]])
            out[f"b{n}_half_{name}"] = endpoint.summarise_draws(pooled)["y"]
            out[f"b{n}_gem_half_{name}"] = endpoint.summarise_draws(matched_half)["y"]
    return out


def construct_values(k: int) -> dict:
    S = DATA["screen"]
    pool = S["pools"][k]
    slot = int(S["slot"][k])
    t_feature = int(S["target_feature"][k])
    keep = np.ones(S["nt_f"].shape[1], dtype=bool)
    if t_feature >= 0:
        keep[t_feature] = False
    ct_p = S["pert_count"][pool]
    ct_c = S["nt_counts"][:, slot]
    Xv = e5lib.values(S["pert_f"][pool][:, keep], S["pert_total"][pool], ct_p)
    Yv = e5lib.values(S["nt_f"][:, keep], S["nt_total"], ct_c)
    mean, vectors = e5lib.basis(Yv, N_PC) if t_feature >= 0 else DATA["basis0"]
    Xs, Ys = e5lib.scores(Xv, mean, vectors), e5lib.scores(Yv, mean, vectors)
    neighbours = S["neighbours"][k]
    neighbour_scores = None
    if neighbours.size:
        kept_index = np.flatnonzero(keep)
        sub = ~np.isin(kept_index, neighbours)
        m2, v2 = e5lib.basis(Yv[:, sub], N_PC)
        neighbour_scores = (e5lib.scores(Xv[:, sub], m2, v2), e5lib.scores(Yv[:, sub], m2, v2))
    row = S["table"].iloc[k]
    target_values_c = endpoint.external_values(ct_c, S["nt_total"]).astype(np.float64)
    out = {"construct": str(row["construct"]), "target_ensg": str(row["target_ensg"]), "target_symbol": str(S["target_symbol"][k]),
           "position": int(k), "n_pool": int(pool.size), "target_is_feature": bool(t_feature >= 0), "n_neighbour_features": int(neighbours.size),
           "B1": float(target_values_c.mean()), "detection_rate": float((ct_c > 0).mean()), "control_sd": float(target_values_c.std(ddof=1)),
           "depth_log_ratio": float(np.log(np.median(S["pert_total"][pool].astype(np.float64)) / np.median(S["nt_total"].astype(np.float64))))}
    for n in LEVELS:
        if pool.size >= n:
            out.update(level_values(S, k, n, Xs, Ys, Xv, Yv, neighbour_scores))
    all_cells = e5lib.energy_distance_gram(Xs, Ys)
    n, m = pool.size, Ys.shape[0]
    out["all_cells"] = all_cells
    out["n_scaled"] = n * m / (n + m) * all_cells
    # SAC-label approximation on the first 27-cell draw: E1's one-gene statistic over the 1,303 features, target included
    # (pools of at least 27 cells; every Replogle construct here has them, VCC's 10-cell genes may not)
    out["sac_label_draw1"], out["sac_label_features"] = float("nan"), 0
    if pool.size >= 27:
        f, s = endpoint.draw(endpoint.SEEDS[0], pool.size, 27, m, N_CTRL)
        a = endpoint.external_values(S["pert_f"][pool][f], S["pert_total"][pool][f])
        b = endpoint.external_values(S["nt_f"][s], S["nt_total"][s])
        stats = np.array([endpoint.ad_log1p(a[:, g], b[:, g]) for g in range(a.shape[1])])
        out["sac_label_draw1"] = float(np.nanmean(stats)) if np.isfinite(stats).any() else float("nan")
        out["sac_label_features"] = int(np.isfinite(stats).sum())
    return out


def check_iii(k: int) -> dict:
    """E1's one-gene statistic at the target on the same draws equals the stored G1 label (the cells of (a))."""
    S = DATA["screen"]
    pool = S["pools"][k]
    tv_p = endpoint.external_values(S["pert_count"][pool], S["pert_total"][pool])
    tv_c = endpoint.external_values(S["nt_counts"][:, int(S["slot"][k])], S["nt_total"])
    summary, draws = endpoint.subsampled_label(tv_p, tv_c, 27, N_CTRL)
    return {"y": summary["y"], "y_draws_1_12": float(np.median(draws[:12][np.isfinite(draws[:12])])),
            "y_draws_13_25": float(np.median(draws[12:][np.isfinite(draws[12:])]))}


# --------------------------------------------------------------------------- genes and gates


def gene_table(constructs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, part in constructs.groupby("target_ensg", sort=True):
        part = part.sort_values("position")
        first_row = part.iloc[0]
        entry = {"target_ensg": gene, "target_symbol": first_row.target_symbol, "n_constructs": int(len(part)),
                 "n_pool_max": int(part.n_pool.max()), "target_is_feature": bool(first_row.target_is_feature),
                 "B1": first_row.B1, "detection_rate": first_row.detection_rate, "control_sd": first_row.control_sd,
                 "depth_log_ratio": float(part.depth_log_ratio.median()), "all_cells": float(part.all_cells.median()),
                 "n_scaled": float(part.n_scaled.median())}
        for column in constructs.columns:
            if column.startswith("b") and column[1:3] in ("27", "50") and not column.endswith("_valid"):
                values = part[column].to_numpy(dtype=np.float64)
                entry[column] = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")
        rows.append(entry)
    return pd.DataFrame(rows)


def gates(screen: str, genes: pd.DataFrame) -> dict:
    folds_of = e5lib.within_screen_folds(genes.target_ensg)
    folds = genes.target_ensg.map(folds_of).to_numpy()
    bc = np.column_stack([e5lib.ranks01(genes[c].to_numpy(dtype=np.float64)) for c in ("B1", "detection_rate", "control_sd")])
    out = {}

    def column(name: str) -> np.ndarray:
        return genes[name].to_numpy(dtype=np.float64) if name in genes else np.full(len(genes), np.nan)

    for n in LEVELS:
        b = column(f"b{n}")
        null = column(f"b{n}_null")
        finite = np.isfinite(b)
        q25, q75 = (np.quantile(b[finite], [0.25, 0.75]) if finite.any() else (np.nan, np.nan))
        median_null = float(np.nanmedian(null)) if np.isfinite(null).any() else float("nan")
        exceeds = bool(np.isfinite(median_null) and median_null > COMPOSITION_RULE * (q75 - q25))
        prefix = f"b{n}_gem" if exceeds else f"b{n}"
        half_a, half_b = column(f"{prefix}_half_A"), column(f"{prefix}_half_B")
        both = np.isfinite(half_a) & np.isfinite(half_b)
        r = float(spearmanr(half_a[both], half_b[both])[0]) if both.sum() > 2 else float("nan")
        res_a = e5lib.out_of_fold_residuals(bc, np.where(both, half_a, np.nan), folds)
        res_b = e5lib.out_of_fold_residuals(bc, np.where(both, half_b, np.nan), folds)
        ok = np.isfinite(res_a) & np.isfinite(res_b)
        r_bc = float(spearmanr(res_a[ok], res_b[ok])[0]) if ok.sum() > 2 else float("nan")
        out[str(n)] = {
            "genes": int(finite.sum()), "genes_with_halves": int(both.sum()),
            "composition_null_median": median_null, "between_gene_iqr": float(q75 - q25),
            "composition_rule_exceeded": exceeds, "primary_version": "gem-group-matched controls" if exceeds else "pooled controls",
            "reliability_spearman": r, "spearman_brown_upper_bound": 2 * r / (1 + r) if np.isfinite(r) else float("nan"),
            "reliability_gate_passed": bool(np.isfinite(r) and r >= RELIABILITY_GATE),
            "beyond_bc_spearman": r_bc, "beyond_bc_gate_passed": bool(np.isfinite(r_bc) and r_bc >= BEYOND_BC_GATE),
            "spearman_with_depth": float(spearmanr(genes["depth_log_ratio"][finite], b[finite])[0]) if finite.sum() > 2 else float("nan"),
        }
    return out


def versions(gate_values: dict) -> dict:
    out = {}
    for setting, target in SETTINGS.items():
        use = 27 if gate_values[target]["27"]["reliability_gate_passed"] else 50
        out[setting] = {"target_screen": target, "version_used_by_the_family": f"{use} cells",
                        "reason": "the target screen's 27-cell reliability gate " + ("passed" if use == 27 else "failed")}
    return out


# --------------------------------------------------------------------------- run


def run_pool(function, items: list[int]) -> list:
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        return list(pool.map(function, items, chunksize=4))


def main() -> None:
    started = time.time()
    single_cell = first.verify_single_cell_inputs()
    both, symbols = feature_mask()
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    record = {"plan": PLAN, "smoke": SMOKE, "single_cell_inputs": single_cell,
              "caches": {"E1a_labels": file_record(second.cache_dir() / "manifest.json"), "E4": file_record(e4_dir() / "manifest.json")},
              "features": {"count": int(both.sum()), "rule": "feature symbols that both screens map (E4 cache feature_columns >= 0)"},
              "seeds": {"draws": list(endpoint.SEEDS), "halves": [e1labels.HALF_CELLS_SEED, e1labels.HALF_CONTROLS_SEED],
                        "within-screen folds": e5lib.WITHIN_FOLD_SEED},
              "settings": {"components": N_PC, "controls": N_CTRL, "levels": list(LEVELS), "half floors": HALF_FLOOR, "window": WINDOW},
              "screens": {}, "checks": {}}
    gate_values = {}
    for screen in SCREENS:
        t0 = time.time()
        S = load_screen(screen, both, symbols)
        add_deciles(S)
        DATA["screen"] = S
        DATA["basis0"] = e5lib.basis(e5lib.values(S["nt_f"], S["nt_total"]), N_PC)
        eligible = [k for k, p in enumerate(S["pools"]) if p.size >= LEVELS[0]]
        if SMOKE:
            features_first = sorted(eligible, key=lambda k: (S["target_feature"][k] < 0, k))
            eligible = sorted(set(features_first[:4]) | set(eligible[:SMOKE_CONSTRUCTS - 4]))
        # check (iii): the cells of (a)
        g1 = pd.read_csv(config.storage_root() / "bioinf2027" / "endpoints" / "G1" / f"external_{screen}.csv", float_precision="round_trip")
        manifest = json.loads((REPO / "bioinf2027" / "endpoints" / "G1" / "manifest.json").read_text())
        recorded = manifest["tables"][f"external_{screen}.csv"]["sha256"]
        if sha256(config.storage_root() / "bioinf2027" / "endpoints" / "G1" / f"external_{screen}.csv") != recorded:
            raise Stop("check (i): G1's external_%s.csv does not match its manifest" % screen)
        g1 = g1.set_index("construct")
        probe = eligible[:: max(1, len(eligible) // CHECK_CONSTRUCTS)][:CHECK_CONSTRUCTS]
        largest = 0.0
        for k in probe:
            ours = check_iii(k)
            theirs = g1.loc[str(S["table"].iloc[k]["construct"])]
            for key in ("y", "y_draws_1_12", "y_draws_13_25"):
                a_, b_ = ours[key], float(theirs[key])
                d = 0.0 if (np.isnan(a_) and np.isnan(b_)) else abs(a_ - b_)
                largest = max(largest, d)
                if not d == 0.0:
                    raise Stop("check (iii): %s %s differs from G1 by %r" % (S["table"].iloc[k]["construct"], key, d))
        record["checks"][f"iii {screen}"] = {"constructs": len(probe), "largest_difference": largest, "passed": True}
        print("[e5 b] %s: %d constructs, check (iii) equal on %d, loading %.0f s" % (screen, len(eligible), len(probe), time.time() - t0), flush=True)
        results = run_pool(construct_values, eligible)
        constructs = pd.DataFrame(results)
        # check (iv): a second computation of 20 constructs, same seeds
        largest = 0.0
        for k in probe:
            again = construct_values(k)
            saved = constructs[constructs.position == k].iloc[0]
            for key, value in again.items():
                if isinstance(value, float):
                    other = float(saved[key])
                    if np.isnan(value) != np.isnan(other):
                        raise Stop("check (iv): %s %s is NaN in one computation only" % (again["construct"], key))
                    d = 0.0 if np.isnan(value) else abs(value - other)
                    largest = max(largest, d)
                    if d > 1e-10:
                        raise Stop("check (iv): %s %s differs by %r on the second computation" % (again["construct"], key, d))
        record["checks"][f"iv {screen}"] = {"constructs": len(probe), "largest_difference": largest, "tolerance": 1e-10, "passed": True}
        genes = gene_table(constructs)
        gate_values[screen] = gates(screen, genes)
        constructs.to_csv(folder / f"constructs_{screen}.csv", index=False)
        genes.to_csv(folder / f"genes_{screen}.csv", index=False)
        first_draw = constructs[["b27_draw1", "sac_label_draw1"]].dropna()
        record["screens"][screen] = {
            "constructs": int(len(constructs)), "genes": int(len(genes)), "target_symbol_mismatches_with_the_endpoint_table": S["symbol_mismatches"],
            "targets_among_the_features": int(constructs.target_is_feature.sum()), "constructs_with_neighbour_features": int((constructs.n_neighbour_features > 0).sum()),
            "constructs_at_50": int(constructs.b50.notna().sum()) if "b50" in constructs else 0,
            "sac_label_spearman_first_draw": float(spearmanr(first_draw.b27_draw1, first_draw.sac_label_draw1)[0]) if len(first_draw) > 2 else float("nan"),
            "seconds": round(time.time() - t0, 1)}
        print("[e5 b] %s done in %.0f s" % (screen, time.time() - t0), flush=True)
        DATA.pop("screen", None)
        del S
    gates_file = {"plan": PLAN, "smoke": SMOKE, "gates": gate_values, "versions_used_by_the_family": versions(gate_values),
                  "rules": {"composition": "median composition null above %.2f of the between-gene IQR" % COMPOSITION_RULE,
                            "reliability": "uncorrected split-half Spearman at least %.1f" % RELIABILITY_GATE,
                            "beyond Bc": "split-half Spearman of out-of-fold Bc residuals at least %.1f" % BEYOND_BC_GATE}}
    (folder / "gates.json").write_text(json.dumps(strict_json(gates_file), indent=1, allow_nan=False) + "\n")
    record["outputs"] = {p.name: file_record(p) for p in sorted(folder.glob("*.csv"))} | {"gates.json": file_record(folder / "gates.json")}
    record["seconds"] = round(time.time() - started, 1)
    (folder / "endpoint_b.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e5 b] done in %.0f s; versions: %s" % (time.time() - started, json.dumps(gates_file["versions_used_by_the_family"])), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
    if arguments:
        raise SystemExit("usage: make_e5_endpoint_b.py [--smoke]")
    main()
