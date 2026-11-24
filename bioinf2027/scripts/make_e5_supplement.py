"""E5 supplement: the results review's analyses 1 and 3 to 8, post hoc (bioinf2027/plans/E5_supplement.md; gate G7, item 3,
the author's answer of 2026-11-17, bioinf2027/GATE.md). Analysis 2 (DepMap) is its own part, after the author's answer
on the release (decided at G4 on 2026-11-17: DepMap 24Q4 Public from figshare+ 27993248, not the current release; --part
depmap after make_e5_depmap_download.py). E5's harness (make_e5_fits.py, make_e5_secondary.py) is reused unchanged.

Parts (--part, in the order given; default: all but depmap):
- fits: on endpoint (b), the four settings, the three vectors, ridge, E5's gene bootstrap with the setting's seed:
    analysis 1, the symbol-prefix family mean FN as one more bar column (built inside each outer fold, leave one out on
    the training rows): [Bc; FN] - Bc and [Bc; FN; e] against [Bc; FN; C] and [Bc; FN; S];
    analysis 3, the family's models under folds by k-means clusters (k = 50) of each vector (unit length, n_init 10,
    random_state 20271207), the cluster labels permuted with E5's fold seeds, fold = position mod 5;
    analysis 6, E5's item "[Bc; B0] as the bar" on b_minus_B2 (diagnostic).
  Each task is one process (fit_slots.py; --workers, --workers-file), saved as it ends and reused when unchanged. Check:
  the family's statistic for GenePT, (b), within RPE1 recomputed equals family.csv (1e-9), or nothing is written.
- depth: analysis 4, per screen, rho(b27, depth_log_ratio), rho(b27_depth, depth_log_ratio), their paired difference and
  rho(b27, b27_depth), gene bootstrap (seeds 20271121, 20271122).
- pool: analysis 5, constructs with at least 150 eligible cells (at most 200 per screen, default_rng([20271205, screen index])), the
  fixed-cell endpoint recomputed on nested subsets of 27, 40, 60 and 100 cells of the pool (default_rng([20271206,
  screen index, position])) exactly as make_e5_endpoint_b.py computes b27 (the full pool must reproduce the saved b27
  to 1e-10); D_m = mean (b at m - b full) / the screen's between-gene IQR, construct bootstrap (seed 20271206 + screen
  index).
- lists: analysis 7, per screen and vector, every target gene with whether it has a vector and why not.
- breakdown: analysis 8, from the family's saved out-of-fold predictions, the mean rank differences of [Bc; e] against
  [Bc; C] and [Bc; S] by fifths of the label and for the ten largest symbol prefixes (descriptive).
--smoke: 200 replicates, two settings, six constructs per screen in pool, outputs under
$IGA_STORAGE_ROOT/bioinf2027/scratch/E5_supplement_smoke/. Outputs of a real run: bioinf2027/results/E5/supplement/.
Every value is post hoc and unadjusted and not part of E5's family.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, endpoint, resample  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.stats import percentile_interval  # noqa: E402

import e5fit  # noqa: E402
import e5lib  # noqa: E402
import fit_slots  # noqa: E402
import make_e5_secondary as sec  # noqa: E402
import safe_pickle  # noqa: E402

fits = sec.fits
REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E5_supplement.md"
STATUS = "post hoc, unadjusted, not part of E5's family (gate G7, item 3; %s)" % PLAN
OUT = REPO / "bioinf2027" / "results" / "E5" / "supplement"
PARTS = ("fits", "depth", "pool", "lists", "breakdown")
EMBEDDINGS = sec.EMBEDDINGS
K_CLUSTERS = 50
KMEANS_SEED = 20271207
POOL_FLOOR = 150
POOL_SAMPLE = 200
POOL_SAMPLE_SEED = 20271205
POOL_SUBSET_SEED = 20271206
POOL_LEVELS = (27, 40, 60, 100)
SMOKE_POOL = 6
B0_ON_B2 = "[Bc; B0] as the bar, (b) residualised on B2"
CHECK_KEY = ("GenePT", "within RPE1")
WORKERS = 4
WORKERS_FILE: Path | None = None
SMOKE = False
DATA: dict = {}

READ_FN = ("beside the target's control-cell profile and the mean label of its symbol-prefix family, the vector ranks unseen target genes "
           "by the downstream endpoint above both a random vector of its width and the same vectors assigned to other genes")
NOT_FN = "not shown beside the symbol-prefix family mean"
READ_CLUSTERS = "the vector's excess holds when genes are held out by clusters of the vector itself"
NOT_CLUSTERS = "not shown when genes are held out by clusters of the vector itself"


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E5_supplement_smoke" if SMOKE else OUT


def tasks_dir() -> Path:
    return (config.storage_root() / "bioinf2027" / "scratch" / "E5_supplement_smoke" / "tasks" if SMOKE
            else config.storage_root() / "bioinf2027" / "E5" / "supplement" / "tasks")


def my_digest() -> str:
    return sha256(Path(__file__).resolve())


def state(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    return "above zero" if lo > 0 else ("below zero" if hi < 0 else "contains zero")


# --------------------------------------------------------------------------- analysis 1: the family mean as a bar


def family_feature(tr: list, te: list, y_tr: np.ndarray, prefix_of: dict) -> tuple[np.ndarray, np.ndarray]:
    """FN of the training genes (leave one out within the prefix, else the mean of the other training genes) and of the
    test genes (the training mean of the prefix, else the training mean)."""
    p_tr = [prefix_of[g] for g in tr]
    sums, counts = {}, {}
    for p, y in zip(p_tr, y_tr):
        sums[p] = sums.get(p, 0.0) + float(y)
        counts[p] = counts.get(p, 0) + 1
    total, n = float(np.sum(y_tr)), len(tr)
    fn_tr = np.array([(sums[p] - y) / (counts[p] - 1) if counts[p] > 1 else (total - y) / (n - 1) for p, y in zip(p_tr, y_tr)], dtype=np.float64)
    fn_te = np.array([sums[prefix_of[g]] / counts[prefix_of[g]] if prefix_of[g] in counts else total / n for g in te], dtype=np.float64)
    return fn_tr, fn_te


def fit_folds_fn(R: dict, models: dict) -> dict:
    """make_e5_secondary.fit_folds' ridge path with FN appended to the bar's columns; also Bc alone on the same rows."""
    fs, ft = R["src_frame"], R["tgt_frame"]
    genes, folds, fold_ids, train_sizes, mean_pred, fn_test = [], [], [], [], [], []
    names = list(models) + ["Bc"]
    predictions, records = {m: [] for m in names}, {m: [] for m in names}
    for k in R["ks"]:
        tr = [g for g in R["src"] if R["folds"][g] != k]
        te = [g for g in R["tgt"] if R["folds"][g] == k]
        if not te:
            continue
        bc_tr, y_tr = fs.loc[tr, R["bc"]].to_numpy(dtype=np.float64), fs.loc[tr, R["src_label"]].to_numpy(dtype=np.float64)
        bc_te = ft.loc[te, R["bc"]].to_numpy(dtype=np.float64)
        fn_tr, fn_te = family_feature(tr, te, y_tr, sec.DATA["prefix_of"])
        bcf_tr, bcf_te = np.column_stack([bc_tr, fn_tr]), np.column_stack([bc_te, fn_te])
        genes += te
        folds += [k] * len(te)
        fold_ids.append(k)
        train_sizes.append(len(tr))
        mean_pred.append(np.full(len(te), float(np.mean(y_tr))))
        fn_test.append(fn_te)
        for name, provider in models.items():
            if provider is None:
                pred, rec = e5fit.fit_predict(bcf_tr, y_tr, np.array(tr), bcf_te)
            else:
                pred, rec = e5fit.fit_predict(bcf_tr, y_tr, np.array(tr), bcf_te, provider(R["source"], tr, k, tr), provider(R["target"], te, k, tr))
            predictions[name].append(np.asarray(pred, dtype=np.float64))
            records[name].append(rec)
        pred, rec = e5fit.fit_predict(bc_tr, y_tr, np.array(tr), bc_te)
        predictions["Bc"].append(np.asarray(pred, dtype=np.float64))
        records["Bc"].append(rec)
    y = ft.loc[genes, R["tgt_label"]].to_numpy(dtype=np.float64)
    return {"genes": genes, "y": y, "folds": np.array(folds, dtype=np.int64), "fold_ids": fold_ids, "train_sizes": train_sizes,
            "predictions": {m: np.concatenate(v) for m, v in predictions.items()}, "mean_predictor": np.concatenate(mean_pred),
            "records": records, "fn_test": np.concatenate(fn_test)}


def edges_of(result: dict) -> int:
    return int(sum(r.get("at_grid_edge", False) for m in ("e", "C", "S") for r in result["records"][m]))


def fn_task(args) -> dict:
    _, name, setting = args
    e5fit.SVD_FALLBACKS[0] = 0
    started = time.time()
    genes = fits.setting_genes(setting)
    R = sec.setting_rows(setting, "b", sec.bar_columns("Bc", "b"), genes)
    models, width = sec.models_for(name, genes)
    result = fit_folds_fn(R, models)
    nn_genes, neighbour, with_family = sec.family_neighbour(R)  # the family baseline's test-gene value is FN's
    if nn_genes != result["genes"] or not np.allclose(neighbour, result["fn_test"], rtol=0, atol=1e-12):
        raise Stop("%s, %s: FN of the test genes differs from the family baseline" % (name, setting))
    stats = fits.spearman_draws(result, sec.SETTING_SEEDS[setting])
    base = sec.base_of("family mean as a bar", name, width, "b", setting, result, "ridge", "gene folds", "genes")
    base.update({"item": "analysis 1: [Bc; FN] as the bar", "role": STATUS, "bar": "[Bc; FN]"})
    row = {**sec.contrast_row(stats, base), **sec.context("b", setting, width), "ridge_alpha_at_grid_edge": edges_of(result)}
    fb = fits.difference(stats, "bar", "Bc")
    row.update({"spearman_Bc_alone": float(stats["point"]["Bc"]), **{f"bar_minus_Bc_{k}": v for k, v in fb.items() if k != "draws"},
                "test_genes_with_a_training_family_member": with_family})
    row["reading"] = READ_FN if row["both_resolved_above_zero"] else NOT_FN
    row["reading_bar_minus_Bc"] = ("the family mean adds to the control-cell profile" if fb["lo"] > 0 else "the family mean is not shown to add to the "
                                   "control-cell profile")
    row["svd_fallbacks"] = int(e5fit.SVD_FALLBACKS[0])
    row["seconds"] = round(time.time() - started, 1)
    return {"key": list(args), "row": row}


# --------------------------------------------------------------------------- analysis 3: folds by clusters of the vectors


def cluster_folds() -> dict:
    """Per vector: within each screen over its covered genes, between the screens over their union; fold of a gene."""
    from sklearn.cluster import KMeans

    out, sizes = {}, {}
    sets = {s: sorted(fits.DATA["covered"][s]) for s in sec.SCREENS}
    sets["union"] = sorted(set(sets["RPE1"]) | set(sets["K562-essential"]))
    for name in EMBEDDINGS:
        vectors = fits.DATA["embeddings"][name]
        for where, genes in sets.items():
            x = np.vstack([vectors[g] for g in genes])
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            labels = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=KMEANS_SEED).fit_predict(x)
            seed = e5lib.WITHIN_FOLD_SEED if where != "union" else fits.UNION_FOLD_SEED
            order = np.random.default_rng(seed).permutation(K_CLUSTERS)
            fold_of_cluster = {int(c): int(position % e5lib.N_FOLDS) for position, c in enumerate(order)}
            folds = {g: fold_of_cluster[int(c)] for g, c in zip(genes, labels)}
            out[(name, where)] = folds
            counts = np.bincount(labels, minlength=K_CLUSTERS)
            sizes[f"{name} | {where}"] = {"genes": len(genes), "cluster_sizes_min_median_max": [int(counts.min()), float(np.median(counts)), int(counts.max())],
                                          "fold_sizes": [int(sum(1 for f in folds.values() if f == k)) for k in range(e5lib.N_FOLDS)]}
    return {"folds": out, "sizes": sizes}


def cluster_task(args) -> dict:
    _, name, setting = args
    e5fit.SVD_FALLBACKS[0] = 0
    started = time.time()
    genes = fits.setting_genes(setting)
    source, target = fits.SETTINGS[setting]
    R = sec.setting_rows(setting, "b", sec.bar_columns("Bc", "b"), genes)
    R["folds"] = DATA["clusters"]["folds"][(name, source if source == target else "union")]
    models, width = sec.models_for(name, genes)
    result = sec.fit_folds(R, models)
    stats = fits.spearman_draws(result, sec.SETTING_SEEDS[setting])
    base = sec.base_of("clusters", name, width, "b", setting, result, "ridge", "folds by k-means clusters of the vector (k = 50)", "genes")
    base.update({"item": "analysis 3: folds by clusters of the vector", "role": STATUS, "bar": "Bc"})
    row = {**sec.contrast_row(stats, base), **sec.context("b", setting, width), "ridge_alpha_at_grid_edge": edges_of(result)}
    row["reading"] = READ_CLUSTERS if row["both_resolved_above_zero"] else NOT_CLUSTERS
    row["svd_fallbacks"] = int(e5fit.SVD_FALLBACKS[0])
    row["seconds"] = round(time.time() - started, 1)
    return {"key": list(args), "row": row}


def b0_task(args) -> dict:
    _, name, setting = args
    out = sec.main_task((B0_ON_B2, name, "b_minus_B2", setting))
    row = {**out["row"], "role": "diagnostic; " + STATUS, "reading": "diagnostic, no reading (printed beside E5's B2 row)"}
    return {"key": list(args), "row": row}


def check_task(args) -> dict:
    _, name, setting = args
    out = sec.main_task((sec.PRIMARY, name, "b", setting))
    return {"key": list(args), "row": out["row"]}


# --------------------------------------------------------------------------- analysis 2: DepMap's gene effect as a bar


DEPMAP_RECORD = OUT / "depmap_download.json"
READ_F = ("beside the target's control-cell profile and an independent fitness score, the vector ranks unseen target genes by the "
          "downstream endpoint above both controls")
NOT_F = "not shown beside the fitness score"
NOT_CURRENT = "DepMap 24Q4 Public (figshare+ 27993248), not the current DepMap release (26Q3)"
DEPMAP_BARS = {"depmap_a": ("F_mean",), "depmap_b": ("F_mean", "F_K562")}


def load_depmap() -> dict:
    """F_mean (mean over cell lines) and F_K562 per upper-case symbol, from the recorded 24Q4 files (checked at load)."""
    record = json.loads(DEPMAP_RECORD.read_text())
    if not record.get("complete") or record.get("release") != "DepMap 24Q4 Public":
        raise Stop("the DepMap download is not complete or not 24Q4")
    paths = {}
    for name, f in record["files"].items():
        path = config.storage_root() / f["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != f["sha256"]:
            raise Stop("%s does not match its download record" % name)
        paths[name] = path
    effect = pd.read_csv(paths["CRISPRGeneEffect.csv"], index_col=0)
    models = pd.read_csv(paths["Model.csv"])
    k562 = models[models["StrippedCellLineName"].astype(str).str.upper() == "K562"]["ModelID"].astype(str).tolist()
    k562_rows = [m for m in k562 if m in effect.index]
    if len(k562_rows) != 1:
        raise Stop("K562 has %d models in Model.csv with a row in CRISPRGeneEffect.csv: %s" % (len(k562_rows), k562_rows))
    symbols = pd.Series([str(c).split(" (")[0].strip().upper() for c in effect.columns], index=effect.columns)
    counts = symbols.value_counts()
    single = symbols[symbols.map(counts) == 1]
    f_mean = effect[single.index].mean(axis=0, skipna=True)
    f_k562 = effect.loc[k562_rows[0], single.index]
    return {"F_mean": dict(zip(single.values, f_mean.to_numpy(dtype=np.float64))), "F_K562": dict(zip(single.values, f_k562.to_numpy(dtype=np.float64))),
            "several": set(counts[counts > 1].index), "k562_model": k562_rows[0], "cell_lines": int(effect.shape[0]), "genes": int(effect.shape[1]),
            "record": file_record(DEPMAP_RECORD)}


def add_depmap_columns(D: dict) -> dict:
    """The F columns and their within-screen ranks on each screen's frame; the mapping of every target gene recorded."""
    coverage = {}
    for s in sec.SCREENS:
        frame = fits.DATA["screens"][s]["frame"]
        upper = frame["symbol"].astype(str).str.upper()
        for column in ("F_mean", "F_K562"):
            values = np.array([D[column].get(u, np.nan) for u in upper], dtype=np.float64)
            frame[column] = values
            frame[f"rank_{column}"] = e5lib.ranks01(values)
        covered = fits.DATA["covered"][s]
        held = frame.index.isin(list(covered))
        coverage[s] = {"covered_genes": int(held.sum()), "with_F_mean": int((held & frame["F_mean"].notna()).sum()),
                       "with_F_K562": int((held & frame["F_K562"].notna()).sum()),
                       "several_columns": int((held & upper.isin(D["several"])).sum()),
                       "misses": int((held & frame["F_mean"].isna() & ~upper.isin(D["several"])).sum())}
        share = coverage[s]["with_F_mean"] / coverage[s]["covered_genes"]
        if share < 0.8:
            raise Stop("DepMap's F_mean covers %.3f of %s's covered genes" % (share, s))
    return coverage


def depmap_task(args) -> dict:
    kind, name, setting = args
    e5fit.SVD_FALLBACKS[0] = 0
    started = time.time()
    genes = fits.setting_genes(setting)
    bc = sec.bar_columns("Bc", "b")
    cols = bc + [f"rank_{c}" for c in DEPMAP_BARS[kind]]
    R = sec.setting_rows(setting, "b", cols, genes)
    models, width = sec.models_for(name, genes)
    result = sec.fit_folds(R, models)
    alone = sec.fit_folds({**R, "bc": bc}, {"Bc": None})
    if alone["genes"] != result["genes"]:
        raise Stop("%s, %s: Bc alone was scored on other genes" % (name, setting))
    result["predictions"]["Bc"] = alone["predictions"]["Bc"]
    stats = fits.spearman_draws(result, sec.SETTING_SEEDS[setting])
    bar = "[Bc; %s]" % "; ".join(DEPMAP_BARS[kind])
    base = sec.base_of("depmap", name, width, "b", setting, result, "ridge", "gene folds", "genes")
    base.update({"item": "analysis 2%s: %s as the bar" % ("a" if kind == "depmap_a" else "b", bar), "role": STATUS, "bar": bar, "depmap_release": NOT_CURRENT})
    row = {**sec.contrast_row(stats, base), **sec.context("b", setting, width), "ridge_alpha_at_grid_edge": edges_of(result)}
    fb = fits.difference(stats, "bar", "Bc")
    row.update({"spearman_Bc_alone": float(stats["point"]["Bc"]), **{f"bar_minus_Bc_{k}": v for k, v in fb.items() if k != "draws"}})
    row["reading"] = READ_F if row["both_resolved_above_zero"] else NOT_F
    row["reading_bar_minus_Bc"] = ("the fitness score adds to the control-cell profile" if fb["lo"] > 0 else "the fitness score is not shown to add to the "
                                   "control-cell profile")
    row["qualifier"] = ("DepMap's scores come from cancer cell lines in other screens; K562 is the cell line of one of the two screens, and RPE1 is not "
                        "a cancer line; " + NOT_CURRENT)
    row["svd_fallbacks"] = int(e5fit.SVD_FALLBACKS[0])
    row["seconds"] = round(time.time() - started, 1)
    return {"key": list(args), "row": row}


def run_depmap() -> None:
    started = time.time()
    sec.load_replogle()
    sec.load_prefit50()
    D = load_depmap()
    coverage = add_depmap_columns(D)
    DATA["depmap_digest"] = D["record"]["sha256"]
    settings = sec.active_settings()
    listed = [(kind, name, setting) for kind in DEPMAP_BARS for name in sorted(EMBEDDINGS, key=lambda n: -fits.WIDTHS[n]) for setting in settings]
    print("[e5 supplement depmap] %d tasks, %d cell lines, %d genes, K562 %s, coverage %s, %.0f s" % (
        len(listed), D["cell_lines"], D["genes"], D["k562_model"], json.dumps(coverage), time.time() - started), flush=True)
    killed = WORKERS_FILE.with_name(WORKERS_FILE.name + ".killed") if WORKERS_FILE is not None else None
    done, failed, shed = fit_slots.run_slots(listed, saved_task, lambda t: " | ".join(t), lambda: fit_slots.limit_from(WORKERS_FILE, WORKERS),
                                             "e5 supplement depmap", started, killed=killed)
    if failed:
        raise Stop("%d tasks failed: %s" % (len(failed), "; ".join("%s: %s" % (f["fit"], f["error"]) for f in failed[:5])))
    rows = [o["row"] for o in sorted(done, key=lambda o: (list(DEPMAP_BARS).index(o["key"][0]), EMBEDDINGS.index(o["key"][1]),
                                                          list(sec.SETTINGS).index(o["key"][2])))]
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "depmap.csv", index=False)
    mapping = []
    for s in sec.SCREENS:
        frame = fits.DATA["screens"][s]["frame"]
        for g in sorted(fits.DATA["covered"][s]):
            u = str(frame.loc[g, "symbol"]).upper()
            mapping.append({"screen": s, "target_ensg": g, "target_symbol": frame.loc[g, "symbol"],
                            "mapping": "several columns" if u in D["several"] else ("match" if np.isfinite(frame.loc[g, "F_mean"]) else "miss"),
                            "F_mean": frame.loc[g, "F_mean"], "F_K562": frame.loc[g, "F_K562"]})
    pd.DataFrame(mapping).to_csv(folder / "depmap_mapping.csv", index=False)
    write_record(folder, "depmap", started, ["depmap.csv", "depmap_mapping.csv"], {
        "depmap": {"release": NOT_CURRENT, "download_record": D["record"], "cell_lines": D["cell_lines"], "genes": D["genes"], "k562_model": D["k562_model"],
                   "coverage": coverage, "symbol_rule": "upper-case symbol before ' (' of the column name; a symbol with several columns is not used"},
        "tasks": {"listed": len(listed), "reused": int(sum(o["reused"] for o in done)), "shed_and_queued_again": shed},
        "svd_fallbacks_total": int(sum(r.get("svd_fallbacks", 0) for r in rows))})
    print("[e5 supplement depmap] %d rows, %.0f s" % (len(rows), time.time() - started), flush=True)


TASKS = {"fn": fn_task, "clusters": cluster_task, "b0_on_b2": b0_task, "check": check_task, "depmap_a": depmap_task, "depmap_b": depmap_task}


def task_spec(key) -> str:
    return json.dumps(strict_json({"key": [str(k) for k in key], "replicates": fits.B, "smoke": SMOKE, "settings": list(sec.active_settings()),
                                   "code": {**sec.DATA["code"], "make_e5_supplement.py": my_digest()}, "inputs": sec.DATA["inputs"],
                                   "clusters": DATA.get("clusters_digest", ""), "depmap": DATA.get("depmap_digest", "")}), sort_keys=True)


def saved_task(args) -> dict:
    """One task, saved as it ends and reused when its specification is unchanged (read back without running code)."""
    spec = task_spec(args)
    path = tasks_dir() / ("%s_%s.pkl" % (args[0], hashlib.sha256(spec.encode()).hexdigest()[:24]))
    if path.is_file():
        value = safe_pickle.load(path, safe_pickle.NUMPY_ARRAYS)
        if value["spec"] == spec:
            return {**value["output"], "reused": True}
    output = TASKS[args[0]](args)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("%s.tmp%d" % (path.name, os.getpid()))
    with open(temporary, "wb") as handle:
        pickle.dump({"spec": spec, "output": output}, handle, protocol=4)
    os.replace(temporary, path)
    return {**output, "reused": False}


def run_fits() -> None:
    started = time.time()
    sec.load_replogle()
    sec.load_prefit50()
    family = sec.load_family()
    sec.ITEMS[B0_ON_B2] = {"labels": ("b_minus_B2",), "bar": "[Bc; B0]", "folds": "gene folds", "role": "diagnostic"}
    DATA["clusters"] = cluster_folds()
    digest = hashlib.sha256()
    for key in sorted(DATA["clusters"]["folds"]):
        folds = DATA["clusters"]["folds"][key]
        digest.update(("%s|%s|" % key).encode() + ";".join("%s=%d" % (g, folds[g]) for g in sorted(folds)).encode())
    DATA["clusters_digest"] = digest.hexdigest()
    settings = sec.active_settings()
    listed = [("check",) + CHECK_KEY] if CHECK_KEY[1] in settings else []
    listed += [(kind, name, setting) for kind in ("fn", "clusters", "b0_on_b2") for name in sorted(EMBEDDINGS, key=lambda n: -fits.WIDTHS[n])
               for setting in settings]
    print("[e5 supplement fits] %d tasks, inputs and clusters ready, %.0f s" % (len(listed), time.time() - started), flush=True)
    killed = WORKERS_FILE.with_name(WORKERS_FILE.name + ".killed") if WORKERS_FILE is not None else None
    done, failed, shed = fit_slots.run_slots(listed, saved_task, lambda t: " | ".join(t), lambda: fit_slots.limit_from(WORKERS_FILE, WORKERS),
                                             "e5 supplement fits", started, killed=killed)
    if failed:
        raise Stop("%d tasks failed: %s" % (len(failed), "; ".join("%s: %s" % (f["fit"], f["error"]) for f in failed[:5])))
    by_kind = {}
    for o in done:
        by_kind.setdefault(o["key"][0], []).append(o)
    check = sec.check_against_family([o["row"] for o in by_kind.get("check", [])], family) if by_kind.get("check") else {"compared": False}
    rows = [o["row"] for kind in ("fn", "clusters", "b0_on_b2") for o in sorted(by_kind.get(kind, []), key=lambda o: (
        EMBEDDINGS.index(o["key"][1]), list(sec.SETTINGS).index(o["key"][2])))]
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "fits.csv", index=False)
    write_record(folder, "fits", started, ["fits.csv"], {
        "checks": {"family recomputed (GenePT, b, within RPE1)": check}, "clusters": DATA["clusters"]["sizes"], "clusters_digest": DATA["clusters_digest"],
        "kmeans": {"k": K_CLUSTERS, "n_init": 10, "random_state": KMEANS_SEED, "vectors": "unit length"},
        "tasks": {"listed": len(listed), "reused": int(sum(o["reused"] for o in done)), "shed_and_queued_again": shed, "folder": config.portable(tasks_dir())},
        "svd_fallbacks_total": int(sum(r.get("svd_fallbacks", 0) for r in rows))})
    print("[e5 supplement fits] %d rows, family check %s, %.0f s" % (len(rows), json.dumps({k: check.get(k) for k in ("compared", "largest_difference")}),
                                                                     time.time() - started), flush=True)


# --------------------------------------------------------------------------- analysis 4: depth


def run_depth() -> None:
    started = time.time()
    record = json.loads((fits.ENDPOINT_B / "endpoint_b.json").read_text())
    rows = []
    for i, screen in enumerate(sec.SCREENS):
        genes = fits.checked_csv(fits.ENDPOINT_B / f"genes_{screen}.csv", record["outputs"][f"genes_{screen}.csv"]["sha256"])
        g = genes[["target_ensg", "b27", "b27_depth", "depth_log_ratio"]].dropna()
        seed = sec.SETTING_SEEDS[f"within {screen}"]
        point = resample.ClusterDraw(weights=np.ones((1, len(g))), clusters=np.array([]), seed=-1, index_sha256="")
        draw = resample.cluster_draw(g["target_ensg"].astype(str).to_numpy(), fits.B, seed)
        cols = g[["b27", "b27_depth"]].to_numpy(dtype=np.float64)
        depth = g["depth_log_ratio"].to_numpy(dtype=np.float64)
        p, d = resample.weighted_spearman(cols, depth, point)[0], resample.weighted_spearman(cols, depth, draw)
        pb, db = resample.weighted_spearman(g[["b27"]].to_numpy(dtype=np.float64), g["b27_depth"].to_numpy(dtype=np.float64), point)[0], \
            resample.weighted_spearman(g[["b27"]].to_numpy(dtype=np.float64), g["b27_depth"].to_numpy(dtype=np.float64), draw)
        values = {"rho(b27, depth_log_ratio)": (p[0], d[:, 0]), "rho(b27_depth, depth_log_ratio)": (p[1], d[:, 1]),
                  "rho(b27, depth) - rho(b27_depth, depth)": (p[0] - p[1], d[:, 0] - d[:, 1]), "rho(b27, b27_depth)": (pb[0], db[:, 0])}
        summary = {}
        for q, (est, reps) in values.items():
            lo, hi, _ = percentile_interval(reps)
            summary[q] = (float(est), lo, hi)
            rows.append({"analysis": "4", "screen": screen, "quantity": q, "n_genes": int(len(g)), "estimate": float(est), "lo": lo, "hi": hi,
                         "p": resample.p_value(reps), "state": state(lo, hi), "seed": seed, "replicate_index_sha256": draw.index_sha256, "role": STATUS})
        diff, after = summary["rho(b27, depth) - rho(b27_depth, depth)"], summary["rho(b27_depth, depth_log_ratio)"]
        if state(diff[1], diff[2]) == "contains zero":
            text = "depth-matched controls are not shown to change the association"
        elif state(after[1], after[2]) == "contains zero":
            text = "depth-matched controls remove the association with the depth ratio"
        else:
            text = "depth-matched controls reduce the association, which remains"
        rows.append({"analysis": "4", "screen": screen, "quantity": "reading", "n_genes": int(len(g)), "reading": text, "role": STATUS})
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "depth.csv", index=False)
    write_record(folder, "depth", started, ["depth.csv"], {"inputs_depth": {s: file_record(fits.ENDPOINT_B / f"genes_{s}.csv") for s in sec.SCREENS}})
    print("[e5 supplement depth] written, %.0f s" % (time.time() - started), flush=True)


# --------------------------------------------------------------------------- analysis 5: nested pools


def construct_scores(k: int):
    """Xs, Ys of construct k exactly as make_e5_endpoint_b.construct_values builds them (features, values, basis)."""
    import make_e5_endpoint_b as eb

    S = eb.DATA["screen"]
    pool = S["pools"][k]
    slot = int(S["slot"][k])
    t_feature = int(S["target_feature"][k])
    keep = np.ones(S["nt_f"].shape[1], dtype=bool)
    if t_feature >= 0:
        keep[t_feature] = False
    Xv = e5lib.values(S["pert_f"][pool][:, keep], S["pert_total"][pool], S["pert_count"][pool])
    Yv = e5lib.values(S["nt_f"][:, keep], S["nt_total"], S["nt_counts"][:, slot])
    mean, vectors = e5lib.basis(Yv, eb.N_PC) if t_feature >= 0 else eb.DATA["basis0"]
    return e5lib.scores(Xv, mean, vectors), e5lib.scores(Yv, mean, vectors)


def pool_task(args) -> dict:
    import make_e5_endpoint_b as eb

    i_screen, k = args
    Xs, Ys = construct_scores(k)
    n_pool, n_ctrl = Xs.shape[0], Ys.shape[0]
    out = {"position": int(k), "n_pool": int(n_pool)}

    def fixed(cells: np.ndarray) -> float:
        values = np.full(len(endpoint.SEEDS), np.nan)
        for i, seed in enumerate(endpoint.SEEDS):
            f, s = endpoint.draw(seed, cells.size, 27, n_ctrl, eb.N_CTRL)
            values[i] = e5lib.energy_distance(Xs[cells[f]], Ys[s])
        return float(endpoint.summarise_draws(values)["y"])

    out["b_full"] = fixed(np.arange(n_pool))
    permutation = np.random.default_rng([POOL_SUBSET_SEED, i_screen, int(k)]).permutation(n_pool)
    for m in POOL_LEVELS:
        out[f"b_{m}"] = fixed(np.sort(permutation[:m]))
    return out


def run_pool() -> None:
    import make_e5_endpoint_b as eb

    started = time.time()
    single_cell = eb.first.verify_single_cell_inputs()
    both, symbols = eb.feature_mask()
    gates = json.loads((fits.ENDPOINT_B / "gates.json").read_text())
    record = json.loads((fits.ENDPOINT_B / "endpoint_b.json").read_text())
    rows, summary = [], []
    for i_screen, screen in enumerate(sec.SCREENS):
        t0 = time.time()
        S = eb.load_screen(screen, both, symbols)
        eb.DATA["screen"] = S
        eb.DATA["basis0"] = e5lib.basis(e5lib.values(S["nt_f"], S["nt_total"]), eb.N_PC)
        saved = fits.checked_csv(fits.ENDPOINT_B / f"constructs_{screen}.csv", record["outputs"][f"constructs_{screen}.csv"]["sha256"]).set_index("position")
        eligible = [k for k, p in enumerate(S["pools"]) if p.size >= POOL_FLOOR]
        take = min(SMOKE_POOL if SMOKE else POOL_SAMPLE, len(eligible))
        chosen = sorted(np.random.default_rng([POOL_SAMPLE_SEED, i_screen]).choice(len(eligible), size=take, replace=False).tolist())
        ks = [eligible[j] for j in chosen]
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
            results = list(pool.map(pool_task, [(i_screen, k) for k in ks]))
        largest = 0.0
        for r in results:
            d = abs(r["b_full"] - float(saved.loc[r["position"], "b27"]))
            largest = max(largest, d)
            if not d <= 1e-10:
                raise Stop("%s, construct %d: the full pool gives %r, the saved b27 %r" % (screen, r["position"], r["b_full"], saved.loc[r["position"], "b27"]))
        frame = pd.DataFrame(results)
        frame.insert(0, "screen", screen)
        frame.insert(1, "construct", [str(saved.loc[p, "construct"]) for p in frame["position"]])
        rows.append(frame)
        iqr = float(gates["gates"][screen]["27"]["between_gene_iqr"])
        draw = resample.cluster_draw(frame["construct"].to_numpy(), fits.B, POOL_SUBSET_SEED + i_screen)
        weights = draw.weights / draw.weights.sum(axis=1, keepdims=True)
        states = []
        for m in POOL_LEVELS:
            diff = (frame[f"b_{m}"].to_numpy() - frame["b_full"].to_numpy()) / iqr
            reps = weights @ diff
            lo, hi, _ = percentile_interval(reps)
            states.append(state(lo, hi))
            summary.append({"analysis": "5", "screen": screen, "quantity": f"D_{m}", "constructs": int(len(frame)), "estimate": float(diff.mean()), "lo": lo,
                            "hi": hi, "p": resample.p_value(reps), "state": state(lo, hi), "between_gene_iqr": iqr, "role": STATUS})
        from scipy.stats import spearmanr

        summary.append({"analysis": "5", "screen": screen, "quantity": "rho(n_pool, b full), sampled constructs, descriptive", "constructs": int(len(frame)),
                        "estimate": float(spearmanr(frame["n_pool"], frame["b_full"])[0]), "role": STATUS})
        text = ("no mechanical shift of the fixed-cell endpoint with the size of the pool it is drawn from is shown (pools of 27 to 100 cells)"
                if all(s == "contains zero" for s in states) else "a mechanical shift with pool size is shown (see D_m)")
        summary.append({"analysis": "5", "screen": screen, "quantity": "reading", "constructs": int(len(frame)), "reading": text,
                        "check_full_pool_largest_difference": largest, "role": STATUS})
        print("[e5 supplement pool] %s: %d constructs, full pool equal to the saved b27 (largest %.1e), %.0f s" % (screen, len(frame), largest, time.time() - t0), flush=True)
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(folder / "pool_constructs.csv", index=False)
    pd.DataFrame(summary).to_csv(folder / "pool.csv", index=False)
    write_record(folder, "pool", started, ["pool.csv", "pool_constructs.csv"], {"single_cell_inputs": single_cell, "levels": list(POOL_LEVELS),
                                                                               "pool_floor": POOL_FLOOR, "sample": SMOKE_POOL if SMOKE else POOL_SAMPLE})
    print("[e5 supplement pool] written, %.0f s" % (time.time() - started), flush=True)


# --------------------------------------------------------------------------- analysis 7: coverage lists


def run_lists() -> None:
    started = time.time()
    sec.load_replogle()
    scgpt_record = json.loads((fits.RESULTS / "identity" / "scgpt_gene.json").read_text())["inputs"]["vocab.json"]
    vocab_path = config.storage_root() / scgpt_record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(vocab_path) != scgpt_record["sha256"]:
        raise Stop("scGPT's vocab.json does not match its record")
    vocab = json.loads(vocab_path.read_text())
    vocab_upper = {}
    for symbol in vocab:
        vocab_upper.setdefault(str(symbol).upper(), []).append(str(symbol))
    record = json.loads((fits.ENDPOINT_B / "endpoint_b.json").read_text())
    rows = []
    for screen in sec.SCREENS:
        genes = fits.checked_csv(fits.ENDPOINT_B / f"genes_{screen}.csv", record["outputs"][f"genes_{screen}.csv"]["sha256"])
        covered = fits.DATA["covered"][screen]
        frame = fits.DATA["screens"][screen]["frame"]
        bc_ok = set(frame.index[frame[[f"rank_{c}" for c in fits.BC_COLUMNS["b"]]].notna().all(axis=1)])
        for ensembl, symbol in zip(genes["target_ensg"].astype(str), genes["target_symbol"].astype(str)):
            entry = {"screen": screen, "target_ensg": ensembl, "target_symbol": symbol, "in_the_covered_set": ensembl in covered,
                     "has_Bc": ensembl in bc_ok}
            removed_by = []
            for name in EMBEDDINGS:
                has = ensembl in fits.DATA["embeddings"][name]
                entry[f"{name}_vector"] = has
                if has:
                    entry[f"{name}_reason"] = ""
                    continue
                removed_by.append(name)
                if name == "Geneformer":
                    entry[f"{name}_reason"] = "the Ensembl ID has no token"
                elif name == "scGPT":
                    other = [s for s in vocab_upper.get(symbol.upper(), []) if s != symbol]
                    entry[f"{name}_reason"] = ("the symbol is in the vocabulary under another case (%s)" % ", ".join(other)) if other else "the symbol is not in the vocabulary"
                else:
                    entry[f"{name}_reason"] = "the upper-case symbol is not in the file"
            entry["removed_from_the_covered_set_by"] = ";".join(removed_by + ([] if ensembl in bc_ok else ["Bc"]))
            rows.append(entry)
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "coverage_lists.csv", index=False)
    counts = {s: {"genes": int((table.screen == s).sum()), "covered": int(table[(table.screen == s)]["in_the_covered_set"].sum()),
                  **{f"{n} without a vector": int((~table[(table.screen == s)][f"{n}_vector"]).sum()) for n in EMBEDDINGS}} for s in sec.SCREENS}
    write_record(folder, "lists", started, ["coverage_lists.csv"], {"counts": counts, "scgpt_vocab": file_record(vocab_path)})
    print("[e5 supplement lists] %s, %.0f s" % (json.dumps(counts), time.time() - started), flush=True)


# --------------------------------------------------------------------------- analysis 8: breakdown


def run_breakdown() -> None:
    started = time.time()
    sec.load_replogle()
    main = json.loads((fits.RESULTS / "secondary" / "secondary_main.json").read_text())["predictions"]
    path = config.storage_root() / main["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != main["sha256"]:
        raise Stop("predictions_main.npz does not match secondary_main.json")
    prefix_of = sec.DATA["prefix_of"]
    rows = []
    with np.load(path, allow_pickle=False) as z:
        index = json.loads(str(z["index"]))
        for i, key in enumerate(index):
            item, name, label, setting = key
            if item != sec.PRIMARY or label != "b":
                continue
            genes = z[f"r{i}_genes"].astype(str)
            y = z[f"r{i}_y"]
            r = {m: e5lib.ranks01(z[f"r{i}_{m}"]) for m in ("e", "C", "S")}
            dc, ds = r["e"] - r["C"], r["e"] - r["S"]
            cuts = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
            fifth = np.searchsorted(cuts, y, side="right")
            for f in range(5):
                mask = fifth == f
                rows.append({"analysis": "8", "embedding": name, "setting": setting, "group": f"fifth {f + 1} of (b)", "genes": int(mask.sum()),
                             "mean_rank_of_b": float(e5lib.ranks01(y)[mask].mean()), "mean_rank_e_minus_C": float(dc[mask].mean()),
                             "mean_rank_e_minus_S": float(ds[mask].mean()), "role": "descriptive; " + STATUS})
            prefixes = pd.Series([prefix_of[g] for g in genes])
            top = prefixes.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[:10]
            ry = e5lib.ranks01(y)
            for p in top:
                mask = (prefixes == p).to_numpy()
                rows.append({"analysis": "8", "embedding": name, "setting": setting, "group": f"prefix {p}", "genes": int(mask.sum()),
                             "mean_rank_of_b": float(ry[mask].mean()), "mean_rank_e_minus_C": float(dc[mask].mean()),
                             "mean_rank_e_minus_S": float(ds[mask].mean()), "role": "descriptive; " + STATUS})
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "breakdown.csv", index=False)
    write_record(folder, "breakdown", started, ["breakdown.csv"], {"predictions": file_record(path)})
    print("[e5 supplement breakdown] %d rows, %.0f s" % (len(rows), time.time() - started), flush=True)


# --------------------------------------------------------------------------- records


def write_record(folder: Path, part: str, started: float, outputs: list[str], extra: dict) -> None:
    record = {"plan": PLAN, "status": STATUS, "smoke": SMOKE, "part": part, "replicates": fits.B,
              "seeds": {"settings": sec.SETTING_SEEDS, "within folds": e5lib.WITHIN_FOLD_SEED, "union folds": fits.UNION_FOLD_SEED,
                        "inner folds": e5fit.INNER_SEED, "shuffle": e5fit.SHUFFLE_SEED, "kmeans": KMEANS_SEED, "pool sample": POOL_SAMPLE_SEED,
                        "pool subsets": POOL_SUBSET_SEED},
              "settings": list(sec.active_settings()), "inputs": sec.DATA.get("inputs", {}),
              "code": {**sec.DATA.get("code", {}), config.portable(Path(__file__).resolve()): my_digest()}, **extra,
              "outputs": {n: file_record(folder / n) for n in outputs}, "seconds": round(time.time() - started, 1)}
    (folder / f"supplement_{part}.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        sec.SMOKE = True
        arguments.remove("--smoke")
        sec.B = 200
        fits.B = 200
    for flag in ("--workers", "--workers-file"):
        if flag in arguments:
            i = arguments.index(flag)
            if flag == "--workers":
                WORKERS = int(arguments[i + 1])
            else:
                WORKERS_FILE = config.storage_root() / arguments[i + 1]
            arguments = arguments[:i] + arguments[i + 2:]
    parts = []
    while "--part" in arguments:
        i = arguments.index("--part")
        parts.append(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if arguments or not set(parts) <= set(PARTS + ("depmap",)) or len(set(parts)) != len(parts):
        raise SystemExit("usage: make_e5_supplement.py [--smoke] [--workers N] [--workers-file PATH] [--part fits|depth|pool|lists|breakdown|depmap ...]")
    for part in parts or PARTS:
        {"fits": run_fits, "depth": run_depth, "pool": run_pool, "lists": run_lists, "breakdown": run_breakdown, "depmap": run_depmap}[part]()
