"""E5, the fits: the gates of endpoint (a), the oracle ceiling and the pipeline positive control (--prefit, before
any embedding is loaded), the revised positive control above width 768 (--positive-control-revised, Section 20),
then the family (--family, after the embeddings exist).

Design: bioinf2027/plans/E5.md, Sections 3, 5 to 8, 16 and 18 (fixed before any fit). Rows are target genes of
the two Replogle screens; endpoint (a) is G1's label, endpoint (b) the version that
bioinf2027/results/E5/endpoint_b/gates.json assigns; Bc is control-only; the harness is
bioinf2027/scripts/e5fit.py. Outputs: bioinf2027/results/E5/prefit/ (this step). With --smoke: 200 bootstrap
replicates and the 768 width only, outputs to $IGA_STORAGE_ROOT/bioinf2027/scratch/E5_fits_smoke/, never reported.
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

from bioinf2027 import config, e1cells, e1inputs, features, resample  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.stats import percentile_interval  # noqa: E402

import e5fit  # noqa: E402
import e5lib  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "bioinf2027" / "results" / "E5"
ENDPOINT_B = RESULTS / "endpoint_b"
PLAN = "bioinf2027/plans/E5.md, Sections 3, 5 to 8, 16 and 18 to 20"
SCREENS = ("RPE1", "K562-essential")
SETTINGS = {"within RPE1": ("RPE1", "RPE1"), "within K562-essential": ("K562-essential", "K562-essential"),
            "RPE1 to K562-essential": ("RPE1", "K562-essential"), "K562-essential to RPE1": ("K562-essential", "RPE1")}
SETTING_SEEDS = {"within RPE1": 20271121, "within K562-essential": 20271122, "RPE1 to K562-essential": 20271123, "K562-essential to RPE1": 20271124}
UNION_FOLD_SEED = 20271204
WIDTHS = {"Geneformer": 768, "scGPT": 512, "GenePT": 1536}
BC_COLUMNS = {"a": ("B1", "detection_rate", "control_sd", "ceiling"), "b": ("B1", "detection_rate", "control_sd")}
RELIABILITY_GATE, BEYOND_BC_GATE = 0.5, 0.2
REVISED_COMPONENTS = 768  # Section 20 (gate G3, item 4): above this width the positive control is padded
REVISED = "revised (G3, item 4)"
B = 10_000
WORKERS = 4
SMOKE = False
DATA: dict = {}


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def out_dir(name: str) -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E5_fits_smoke" / name
    return RESULTS / name


def other(screen: str) -> str:
    return SCREENS[1] if screen == SCREENS[0] else SCREENS[0]


def checked_csv(path: Path, sha: str) -> pd.DataFrame:
    if sha256(path) != sha:
        raise Stop("an input does not match its record: %s" % path.name)
    return pd.read_csv(path, float_precision="round_trip")


# --------------------------------------------------------------------------- inputs


def gene_median(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("target_ensg")[column].median()


def load() -> dict:
    record = {}
    g1 = e1cells.load_tables("G1")
    g1_manifest = json.loads((REPO / "bioinf2027" / "endpoints" / "G1" / "manifest.json").read_text())
    supplement = json.loads((REPO / "bioinf2027" / "endpoints" / "readouts" / "supplement" / "supplement.json").read_text())
    b_record = json.loads((ENDPOINT_B / "endpoint_b.json").read_text())
    gates_b = json.loads((ENDPOINT_B / "gates.json").read_text())
    if b_record.get("smoke") or gates_b.get("smoke"):
        raise Stop("endpoint (b) outputs come from a smoke run")
    if sha256(ENDPOINT_B / "gates.json") != b_record["outputs"]["gates.json"]["sha256"]:
        raise Stop("gates.json does not match endpoint_b.json")
    symbols = features.hvg_symbols()
    present = {s: np.array([v in set(e1inputs.native_symbols(s).tolist()) for v in symbols.tolist()]) for s in SCREENS}
    both = present[SCREENS[0]] & present[SCREENS[1]]
    out = {"screens": {}, "gates_b": gates_b, "both": int(both.sum())}
    for s in SCREENS:
        table = g1[s]
        eligible = table[table["n_pool"] >= 27]
        halves_record = g1_manifest["tables"][f"external_{s}_halves.csv"]
        halves = checked_csv(config.storage_root() / "bioinf2027" / "endpoints" / "G1" / f"external_{s}_halves.csv", halves_record["sha256"])
        halves = halves[(halves["level"] == 27) & (halves["version"] == "pooled control halves")]
        ceil_record = supplement["per_row_ceilings"][f"ceiling_G1_{s}.csv"]
        ceiling = checked_csv(config.storage_root() / ceil_record["path"].split("$IGA_STORAGE_ROOT/", 1)[1], ceil_record["sha256"])
        ceiling = ceiling[ceiling["construct"].isin(set(eligible["construct"]))]
        genes_b = checked_csv(ENDPOINT_B / f"genes_{s}.csv", b_record["outputs"][f"genes_{s}.csv"]["sha256"]).set_index("target_ensg")
        inputs = e1inputs.load_screen_inputs(s, table)
        if not np.array_equal(inputs.constructs, table["construct"].astype(str).to_numpy()):
            raise Stop("%s: the screen inputs list other constructs than G1" % s)
        x = pd.DataFrame(inputs.x["full_pass"][:, both], index=table["target_ensg"].astype(str).to_numpy())
        x = x.loc[x.index.isin(set(eligible["target_ensg"].astype(str)))].groupby(level=0).mean()
        frame = pd.DataFrame({"a": gene_median(eligible, "y"), "a_half_A": gene_median(halves[halves.half == "A"], "y"),
                              "a_half_B": gene_median(halves[halves.half == "B"], "y"), "ceiling": gene_median(ceiling, "ceiling")})
        frame = frame.join(genes_b[["B1", "detection_rate", "control_sd"]], how="left")
        within = gates_b["versions_used_by_the_family"][f"within {s}"]["version_used_by_the_family"]
        level = within.split()[0]
        primary = gates_b["gates"][s][level]["primary_version"]
        column = f"b{level}_gem" if primary.startswith("gem") else f"b{level}"
        frame["b"] = genes_b[column]
        frame["b_column"] = column
        frame.index.name = "target_ensg"
        for c in ("B1", "detection_rate", "control_sd", "ceiling"):
            frame[f"rank_{c}"] = e5lib.ranks01(frame[c].to_numpy(dtype=np.float64))
        out["screens"][s] = {"frame": frame, "x": x}
        record[s] = {"genes": int(len(frame)), "b_column": column, "genes_with_b": int(frame["b"].notna().sum()),
                     "genes_with_halves": int((frame.a_half_A.notna() & frame.a_half_B.notna()).sum())}
    union = sorted(set(out["screens"][SCREENS[0]]["frame"].index) | set(out["screens"][SCREENS[1]]["frame"].index))
    order = np.random.default_rng(UNION_FOLD_SEED).permutation(len(union))
    out["union_folds"] = {union[i]: int(p % 5) for p, i in enumerate(order)}
    out["within_folds"] = {s: e5lib.within_screen_folds(out["screens"][s]["frame"].index) for s in SCREENS}
    out["record"] = {**record, "union_genes": len(union), "features_mapped_in_both": out["both"],
                     "inputs": {"G1_manifest": file_record(REPO / "bioinf2027" / "endpoints" / "G1" / "manifest.json"),
                                "supplement": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "supplement" / "supplement.json"),
                                "endpoint_b": file_record(ENDPOINT_B / "endpoint_b.json"), "gates_b": file_record(ENDPOINT_B / "gates.json")}}
    return out


# --------------------------------------------------------------------------- gates of (a)


def gates_a() -> dict:
    out = {}
    for s in SCREENS:
        f = DATA["screens"][s]["frame"]
        a, b = f["a_half_A"].to_numpy(dtype=np.float64), f["a_half_B"].to_numpy(dtype=np.float64)
        both = np.isfinite(a) & np.isfinite(b)
        r = float(spearmanr(a[both], b[both])[0])
        ranks = f[[f"rank_{c}" for c in BC_COLUMNS["a"]]].to_numpy(dtype=np.float64)
        folds = f.index.map(DATA["within_folds"][s]).to_numpy()
        ra = e5lib.out_of_fold_residuals(ranks, np.where(both, a, np.nan), folds)
        rb = e5lib.out_of_fold_residuals(ranks, np.where(both, b, np.nan), folds)
        ok = np.isfinite(ra) & np.isfinite(rb)
        r_bc = float(spearmanr(ra[ok], rb[ok])[0])
        out[s] = {"genes_with_halves": int(both.sum()), "reliability_spearman": r, "spearman_brown_upper_bound": 2 * r / (1 + r),
                  "reliability_gate_passed": bool(r >= RELIABILITY_GATE), "beyond_bc_spearman": r_bc,
                  "beyond_bc_gate_passed": bool(r_bc >= BEYOND_BC_GATE), "genes_in_residual_spearman": int(ok.sum())}
    return out


# --------------------------------------------------------------------------- one setting


def rows_of(setting: str, endpoint: str, genes_subset: set | None) -> dict:
    source, target = SETTINGS[setting]
    fs, ft = DATA["screens"][source]["frame"], DATA["screens"][target]["frame"]
    bc = [f"rank_{c}" for c in BC_COLUMNS[endpoint]]

    def usable(frame):
        ok = frame[endpoint].notna() & frame[bc].notna().all(axis=1)
        if genes_subset is not None:
            ok &= frame.index.isin(genes_subset)
        return frame.index[ok]

    src, tgt = usable(fs), usable(ft)
    if source == target:
        folds = DATA["within_folds"][source]
    else:
        folds = DATA["union_folds"]
    return {"source": source, "target": target, "src": list(src), "tgt": list(tgt), "folds": folds, "bc": bc}


def run_setting(setting: str, endpoint: str, models: dict, genes_subset: set | None = None, pca_width: int | None = None,
                pad_to: int | None = None) -> dict:
    """Pooled out-of-fold predictions of each model over the setting's test genes.

    ``models`` maps a name to None (Bc alone) or to a function (screen, genes, fold, train_genes) giving the input
    rows; with ``pca_width`` the input is reduced by principal components fitted on the fold's training rows, and with
    ``pad_to`` followed by the seeded Gaussian columns of Section 20 up to that width.
    """
    R = rows_of(setting, endpoint, genes_subset)
    fs, ft = DATA["screens"][R["source"]]["frame"], DATA["screens"][R["target"]]["frame"]
    test_genes, predictions, records = [], {m: [] for m in models}, {m: [] for m in models}
    for k in range(5):
        tr = [g for g in R["src"] if R["folds"][g] != k]
        te = [g for g in R["tgt"] if R["folds"][g] == k]
        if not te:
            continue
        bc_tr, y_tr = fs.loc[tr, R["bc"]].to_numpy(dtype=np.float64), fs.loc[tr, endpoint].to_numpy(dtype=np.float64)
        bc_te = ft.loc[te, R["bc"]].to_numpy(dtype=np.float64)
        test_genes += te
        for name, provider in models.items():
            if provider is None:
                pred, rec = e5fit.fit_predict(bc_tr, y_tr, np.array(tr), bc_te)
            else:
                x_tr, x_te = provider(R["source"], tr, k, tr), provider(R["target"], te, k, tr)
                if pca_width is not None:
                    x_tr, x_te, width = reduce(x_tr, x_te, pca_width)
                    rec_width = width
                    if pad_to is not None and pad_to > width:
                        x_tr = np.hstack([x_tr, e5fit.padding_block(pad_to, width, tr)])
                        x_te = np.hstack([x_te, e5fit.padding_block(pad_to, width, te)])
                pred, rec = e5fit.fit_predict(bc_tr, y_tr, np.array(tr), bc_te, x_tr, x_te)
                if pca_width is not None:
                    rec = {**rec, "width": rec_width}
            predictions[name].append(pred)
            records[name].append(rec)
    y = ft.loc[test_genes, endpoint].to_numpy(dtype=np.float64)
    return {"genes": test_genes, "y": y, "predictions": {m: np.concatenate(v) for m, v in predictions.items()}, "records": records}


def reduce(x_tr: np.ndarray, x_te: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, int]:
    mean = x_tr.mean(axis=0)
    u, s, vt = e5fit.thin_svd(x_tr - mean)
    rank = int((s > s[0] * 1e-10).sum())
    w = min(int(width), rank)
    return (x_tr - mean) @ vt[:w].T, (x_te - mean) @ vt[:w].T, w


def spearman_draws(result: dict, seed: int) -> dict:
    names = list(result["predictions"])
    columns = np.column_stack([result["predictions"][m] for m in names])
    point = resample.ClusterDraw(weights=np.ones((1, len(result["y"]))), clusters=np.array([]), seed=-1, index_sha256="")
    draw = resample.cluster_draw(np.array(result["genes"]), B, seed)
    p = resample.weighted_spearman(columns, result["y"], point)[0]
    d = resample.weighted_spearman(columns, result["y"], draw)
    return {"names": names, "point": dict(zip(names, p)), "draws": {m: d[:, j] for j, m in enumerate(names)}, "index_sha256": draw.index_sha256}


def difference(stats: dict, a: str, b: str) -> dict:
    point = float(stats["point"][a] - stats["point"][b])
    draws = stats["draws"][a] - stats["draws"][b]
    lo, hi, _ = percentile_interval(draws)
    return {"estimate": point, "lo": lo, "hi": hi, "p": resample.p_value(draws), "se": float(np.nanstd(draws, ddof=1)), "draws": draws}


# --------------------------------------------------------------------------- ceiling and positive control


def shared_genes() -> set:
    return set(DATA["screens"][SCREENS[0]]["frame"].index) & set(DATA["screens"][SCREENS[1]]["frame"].index)


def other_label(endpoint: str):
    def provider(screen, genes, fold, train):
        frame = DATA["screens"][other(screen)]["frame"]
        return frame.loc[genes, [endpoint]].to_numpy(dtype=np.float64)
    return provider


def other_x(shuffle: bool):
    vectors = {}
    for s in SCREENS:
        x = DATA["screens"][s]["x"]
        vectors[s] = {g: x.loc[g].to_numpy(dtype=np.float64) for g in x.index}
    if shuffle:
        genes = sorted(shared_genes())
        vectors = {s: e5fit.shuffled({g: vectors[s][g] for g in genes}, genes) for s in SCREENS}

    def provider(screen, genes, fold, train):
        source = vectors[other(screen)]
        return np.vstack([source[g] for g in genes])
    return provider


def gaussian(width_of_fold: dict):
    def provider(screen, genes, fold, train):
        return e5fit.gaussian_block(width_of_fold[fold], genes)
    return provider


def ceiling_task(args) -> dict:
    setting, endpoint = args
    genes = shared_genes()
    result = run_setting(setting, endpoint, {"Bc": None, "[Bc; other screen's label]": other_label(endpoint)}, genes)
    stats = spearman_draws(result, SETTING_SEEDS[setting])
    d = difference(stats, "[Bc; other screen's label]", "Bc")
    return {"setting": setting, "endpoint": endpoint, "n_test_genes": len(result["genes"]),
            "spearman_Bc": stats["point"]["Bc"], "spearman_with_other_label": stats["point"]["[Bc; other screen's label]"],
            **{f"difference_{k}": v for k, v in d.items() if k != "draws"},
            "ridge_alpha_at_grid_edge": int(sum(r.get("at_grid_edge", False) for r in result["records"]["[Bc; other screen's label]"]))}


def positive_control_task(args) -> dict:
    """(setting, endpoint, width) as planned (Section 18); with a fourth element True the revised control of Section 20."""
    setting, endpoint, width, *rest = args
    revised = bool(rest and rest[0])
    genes = shared_genes()
    components, pad = (min(width, REVISED_COMPONENTS), width) if revised else (width, None)
    reduced = run_setting(setting, endpoint, {"Bc": None, "[Bc; PC]": other_x(False)}, genes, pca_width=components, pad_to=pad)
    widths = {k: rec["width"] for k, rec in enumerate(reduced["records"]["[Bc; PC]"])}
    shuffled_run = run_setting(setting, endpoint, {"[Bc; S]": other_x(True)}, genes, pca_width=components, pad_to=pad)
    gaussian_run = run_setting(setting, endpoint, {"[Bc; C]": gaussian({k: width for k in widths} if revised else widths)}, genes)
    for extra in (shuffled_run, gaussian_run):
        if extra["genes"] != reduced["genes"]:
            raise Stop("the positive control's runs differ in their test genes")
        reduced["predictions"].update(extra["predictions"])
        reduced["records"].update(extra["records"])
    stats = spearman_draws(reduced, SETTING_SEEDS[setting])
    dg, ds = difference(stats, "[Bc; PC]", "[Bc; C]"), difference(stats, "[Bc; PC]", "[Bc; S]")
    iu = e5fit.iu_statistic((dg["estimate"], dg["draws"]), (ds["estimate"], ds["draws"]))
    edges = sum(r.get("at_grid_edge", False) for m in ("[Bc; PC]", "[Bc; C]", "[Bc; S]") for r in reduced["records"][m])
    return {"setting": setting, "endpoint": endpoint, "nominal_width": width, "fold_widths": ";".join(str(widths[k]) for k in sorted(widths)),
            "n_test_genes": len(reduced["genes"]), **{f"spearman_{m}": stats["point"][m] for m in ("Bc", "[Bc; PC]", "[Bc; C]", "[Bc; S]")},
            "minus_gaussian": dg["estimate"], "minus_gaussian_lo": dg["lo"], "minus_gaussian_hi": dg["hi"], "minus_gaussian_p": dg["p"],
            "minus_shuffled": ds["estimate"], "minus_shuffled_lo": ds["lo"], "minus_shuffled_hi": ds["hi"], "minus_shuffled_p": ds["p"],
            "iu_estimate": iu["estimate"], "iu_p": iu["p"],
            # resolved: both differences above zero at p below 0.05 (Section 8's reading); below zero is a failure of the control
            "resolved": bool(iu["p"] < 0.05 and iu["estimate"] > 0), "resolved_below_zero": bool(iu["p"] < 0.05 and iu["estimate"] < 0),
            "ridge_alpha_at_grid_edge": int(edges),
            **({"version": REVISED, "fold_padding": ";".join(str(width - widths[k]) for k in sorted(widths))} if revised else {}),
            "replicate_index_sha256": stats["index_sha256"]}


def run_prefit() -> None:
    started = time.time()
    DATA.update(load())
    folder = out_dir("prefit")
    folder.mkdir(parents=True, exist_ok=True)
    g_a = gates_a()
    context = multiprocessing.get_context("fork")
    ceiling_args = [(s, e) for s in SETTINGS for e in ("a", "b")]
    widths = (768,) if SMOKE else tuple(sorted(set(WIDTHS.values())))
    pc_args = [(s, e, w) for s in SETTINGS for e in ("a", "b") for w in widths]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        ceiling = list(pool.map(ceiling_task, ceiling_args))
        pc = list(pool.map(positive_control_task, pc_args))
    pd.DataFrame(ceiling).to_csv(folder / "ceiling.csv", index=False)
    pd.DataFrame(pc).to_csv(folder / "positive_control.csv", index=False)
    (folder / "gates_a.json").write_text(json.dumps(strict_json({"plan": PLAN, "smoke": SMOKE, "gates": g_a,
                                                                 "rules": {"reliability": RELIABILITY_GATE, "beyond Bc": BEYOND_BC_GATE}}), indent=1, allow_nan=False) + "\n")
    record = {"plan": PLAN, "smoke": SMOKE, "replicates": B, "seeds": {"settings": SETTING_SEEDS, "union folds": UNION_FOLD_SEED,
                                                                        "within folds": e5lib.WITHIN_FOLD_SEED, "inner folds": e5fit.INNER_SEED,
                                                                        "shuffle": e5fit.SHUFFLE_SEED},
              "ridge_grid": list(e5fit.RIDGE_GRID), "inputs": DATA["record"],
              "code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (sys.modules[__name__], e5fit, e5lib)},
              "outputs": {n: file_record(folder / n) for n in ("ceiling.csv", "positive_control.csv", "gates_a.json")},
              "seconds": round(time.time() - started, 1)}
    (folder / "prefit.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e5 prefit] done in %.0f s; gates (a): %s; positive control resolved: %d of %d" % (
        time.time() - started, json.dumps({s: (v["reliability_gate_passed"], v["beyond_bc_gate_passed"]) for s, v in g_a.items()}),
        sum(r["resolved"] for r in pc), len(pc)), flush=True)


def run_positive_control_revised() -> None:
    """Section 20 (gate G3, item 4): the positive control at each width above 768, revised; the planned file is untouched."""
    started = time.time()
    DATA.update(load())
    folder = out_dir("prefit")
    folder.mkdir(parents=True, exist_ok=True)
    widths = sorted(w for w in set(WIDTHS.values()) if w > REVISED_COMPONENTS)
    args = [(s, e, w, True) for s in SETTINGS for e in ("a", "b") for w in widths]
    if SMOKE:
        args = args[:2]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(args)), mp_context=context) as pool:
        pc = list(pool.map(positive_control_task, args))
    pd.DataFrame(pc).to_csv(folder / "positive_control_revised.csv", index=False)
    record = {"plan": "bioinf2027/plans/E5.md, Section 20 (gate G3, item 4)", "smoke": SMOKE, "replicates": B,
              "components": REVISED_COMPONENTS, "padding_seed_text": "E5 | PC padding | <width> | <Ensembl ID>",
              "seeds": {"settings": SETTING_SEEDS, "union folds": UNION_FOLD_SEED, "within folds": e5lib.WITHIN_FOLD_SEED,
                        "inner folds": e5fit.INNER_SEED, "shuffle": e5fit.SHUFFLE_SEED},
              "ridge_grid": list(e5fit.RIDGE_GRID), "inputs": DATA["record"],
              "planned_control": file_record(RESULTS / "prefit" / "positive_control.csv"),
              "code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (sys.modules[__name__], e5fit, e5lib)},
              "outputs": {"positive_control_revised.csv": file_record(folder / "positive_control_revised.csv")},
              "seconds": round(time.time() - started, 1)}
    (folder / "prefit_revised.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e5 positive control revised] done in %.0f s; resolved: %d of %d" % (time.time() - started, sum(r["resolved"] for r in pc), len(pc)), flush=True)


# --------------------------------------------------------------------------- the family (Section 8)

EMBEDDINGS = {"Geneformer": "geneformer_token", "scGPT": "scgpt_gene", "GenePT": "genept"}
REFITS = 20


def load_embeddings() -> dict:
    out = {}
    for name, file in EMBEDDINGS.items():
        record = json.loads((RESULTS / "identity" / f"{file}.json").read_text())["output"]
        path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != record["sha256"]:
            raise Stop("the %s vectors do not match their record" % name)
        with np.load(path, allow_pickle=False) as z:
            out[name] = {str(g): z["vectors"][i].astype(np.float64) for i, g in enumerate(z["ensembl"])}
        if len({v.size for v in out[name].values()}) != 1:
            raise Stop("the %s vectors have more than one width" % name)
    return out


def covered() -> dict:
    """Per screen, the genes all three embeddings and Bc cover (Section 3); coverage below 80 % stops (Section 11, 5)."""
    out = {}
    for s in SCREENS:
        frame = DATA["screens"][s]["frame"]
        genes = set(frame.index)
        both = set(frame.index[frame[[f"rank_{c}" for c in BC_COLUMNS["b"]]].notna().all(axis=1)])
        for name, vectors in DATA["embeddings"].items():
            share = len(genes & set(vectors)) / len(genes)
            if share < 0.8:
                raise Stop("%s covers %.3f of %s's genes" % (name, share, s))
            both &= set(vectors)
        out[s] = both
    return out


def embedding_provider(name: str, vectors: dict):
    def provider(screen, genes, fold, train):
        return np.vstack([vectors[g] for g in genes])
    return provider


def gaussian_provider(width: int):
    def provider(screen, genes, fold, train):
        return e5fit.gaussian_block(width, genes)
    return provider


def setting_genes(setting: str) -> set:
    source, target = SETTINGS[setting]
    return DATA["covered"][source] | DATA["covered"][target]


def family_task(args) -> dict:
    name, endpoint, setting = args
    e5fit.SVD_FALLBACKS[0] = 0
    vectors = DATA["embeddings"][name]
    width = next(iter(vectors.values())).size
    genes = setting_genes(setting)
    shuffled = e5fit.shuffled({g: vectors[g] for g in genes}, genes)
    models = {"Bc": None, "[Bc; e]": embedding_provider(name, vectors), "[Bc; C]": gaussian_provider(width),
              "[Bc; S]": embedding_provider(name, shuffled)}
    result = run_setting(setting, endpoint, models, genes)
    stats = spearman_draws(result, SETTING_SEEDS[setting])
    dg, ds, db = (difference(stats, "[Bc; e]", m) for m in ("[Bc; C]", "[Bc; S]", "Bc"))
    iu = e5fit.iu_statistic((dg["estimate"], dg["draws"]), (ds["estimate"], ds["draws"]))
    refit_share = refit_sensitivity(setting, endpoint, models, genes, iu["estimate"])
    edges = sum(r.get("at_grid_edge", False) for m in ("[Bc; e]", "[Bc; C]", "[Bc; S]") for r in result["records"][m])
    return {"embedding": name, "width": width, "endpoint": endpoint, "setting": setting, "n_test_genes": len(result["genes"]),
            **{f"spearman_{m}": stats["point"][m] for m in models},
            **{f"minus_gaussian_{k}": v for k, v in dg.items() if k != "draws"}, **{f"minus_shuffled_{k}": v for k, v in ds.items() if k != "draws"},
            **{f"minus_Bc_{k}": v for k, v in db.items() if k != "draws"},
            "iu_estimate": iu["estimate"], "iu_p": iu["p"], "refit_share_same_sign": refit_share, "ridge_alpha_at_grid_edge": int(edges),
            "power_2.80se": 2.80 * max(dg["se"], ds["se"]), "holm_first_step_detectable": (3.08 + 0.84) * max(dg["se"], ds["se"]),
            "svd_fallbacks": int(e5fit.SVD_FALLBACKS[0]), "replicate_index_sha256": stats["index_sha256"]}


def refit_sensitivity(setting: str, endpoint: str, models: dict, genes: set, primary: float) -> float:
    """20 resamples with replacement of each fold's training genes (seed 20271129); share of the smaller difference with the primary sign."""
    R = rows_of(setting, endpoint, genes)
    fs, ft = DATA["screens"][R["source"]]["frame"], DATA["screens"][R["target"]]["frame"]
    rng = np.random.default_rng([e5fit.REFIT_SEED, list(SETTINGS).index(setting)])  # one generator per setting (Section 18)
    signs = []
    for _ in range(1 if SMOKE else REFITS):
        preds = {m: [] for m in ("[Bc; e]", "[Bc; C]", "[Bc; S]")}
        y_all = []
        for k in range(5):
            tr = [g for g in R["src"] if R["folds"][g] != k]
            te = [g for g in R["tgt"] if R["folds"][g] == k]
            if not te:
                continue
            tr = [tr[i] for i in rng.integers(0, len(tr), len(tr))]
            bc_tr, y_tr = fs.loc[tr, R["bc"]].to_numpy(dtype=np.float64), fs.loc[tr, endpoint].to_numpy(dtype=np.float64)
            bc_te = ft.loc[te, R["bc"]].to_numpy(dtype=np.float64)
            # duplicated training genes stay in one inner fold: the inner folds are assigned by gene
            for m in preds:
                pred, _ = e5fit.fit_predict(bc_tr, y_tr, np.array(tr), bc_te, models[m](R["source"], tr, k, tr), models[m](R["target"], te, k, tr))
                preds[m].append(pred)
            y_all.append(ft.loc[te, endpoint].to_numpy(dtype=np.float64))
        y = np.concatenate(y_all)
        rho = {m: spearmanr(np.concatenate(v), y)[0] for m, v in preds.items()}
        smaller = min(rho["[Bc; e]"] - rho["[Bc; C]"], rho["[Bc; e]"] - rho["[Bc; S]"])
        signs.append(np.sign(smaller) == np.sign(primary))
    return float(np.mean(signs))


def run_family() -> None:
    started = time.time()
    DATA.update(load())
    DATA["embeddings"] = load_embeddings()
    DATA["covered"] = covered()
    prefit = json.loads((RESULTS / "prefit" / "prefit.json").read_text())
    pc = pd.read_csv(RESULTS / "prefit" / "positive_control.csv")
    pc_revised = pd.read_csv(RESULTS / "prefit" / "positive_control_revised.csv")
    gates_a = json.loads((RESULTS / "prefit" / "gates_a.json").read_text())["gates"]
    gates_b = DATA["gates_b"]["gates"]
    folder = out_dir("family")
    folder.mkdir(parents=True, exist_ok=True)
    args = [(n, e, s) for n in EMBEDDINGS for e in ("a", "b") for s in SETTINGS]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        rows = list(pool.map(family_task, args))
    frame = pd.DataFrame(rows)
    rejected, thresholds = resample.holm(frame["iu_p"].to_numpy(), 0.05)
    frame["holm_threshold"], frame["holm_rejected"] = thresholds, rejected
    readable, reasons, versions, planned_resolved = [], [], [], []
    for r in frame.itertuples():
        target = SETTINGS[r.setting][1]
        g = gates_a[target] if r.endpoint == "a" else gates_b[target]["27"]
        gate_ok = bool(g["reliability_gate_passed"] and g["beyond_bc_gate_passed"])
        planned = pc[(pc.setting == r.setting) & (pc.endpoint == r.endpoint) & (pc.nominal_width == r.width)]
        table = pc_revised if r.width > REVISED_COMPONENTS else pc  # Section 20: GenePT read under the revised control
        control = table[(table.setting == r.setting) & (table.endpoint == r.endpoint) & (table.nominal_width == r.width)]
        pc_ok = bool(len(control) and control["resolved"].iloc[0])
        versions.append(REVISED if r.width > REVISED_COMPONENTS else "planned")
        planned_resolved.append(bool(len(planned) and planned["resolved"].iloc[0]))
        readable.append(gate_ok and pc_ok)
        reasons.append("" if gate_ok and pc_ok else ("gates failed" if not gate_ok else "positive control not resolved: not informative"))
    frame["readable"], frame["not_read_because"] = readable, reasons
    frame["positive_control_version"], frame["planned_positive_control_resolved"] = versions, planned_resolved
    frame["reading"] = [
        ("" if not rd else ("beside the target's control-cell profile, the identity vector ranks the endpoint of unseen target genes better than "
                            "both a random vector of its width and the same vectors assigned to other genes" if (rj and r.minus_gaussian_estimate > 0 and r.minus_shuffled_estimate > 0)
                            else ("a difference below zero" if min(r.minus_gaussian_estimate, r.minus_shuffled_estimate) < 0 and r.iu_p < 0.05 else "not shown")))
        for r, rj, rd in zip(frame.itertuples(), rejected, readable)]
    frame.to_csv(folder / "family.csv", index=False)
    record = {"plan": PLAN, "smoke": SMOKE, "replicates": B, "refits": 1 if SMOKE else REFITS, "statistics": len(frame),
              "svd_fallbacks_total": int(frame["svd_fallbacks"].sum()),
              "covered_genes": {s: len(v) for s, v in DATA["covered"].items()}, "inputs": DATA["record"],
              "embeddings": {n: json.loads((RESULTS / "identity" / f"{f}.json").read_text())["output"] for n, f in EMBEDDINGS.items()},
              "prefit": file_record(RESULTS / "prefit" / "prefit.json"), "prefit_outputs": prefit["outputs"],
              "positive_control_revised": file_record(RESULTS / "prefit" / "prefit_revised.json"),
              "code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (sys.modules[__name__], e5fit, e5lib)},
              "outputs": {"family.csv": file_record(folder / "family.csv")}, "seconds": round(time.time() - started, 1)}
    (folder / "family.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e5 family] done in %.0f s; %d statistics, Holm-rejected %d, readable %d" % (time.time() - started, len(frame), int(rejected.sum()), int(sum(readable))), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
        B = 200
    if arguments == ["--prefit"]:
        run_prefit()
    elif arguments == ["--positive-control-revised"]:
        run_positive_control_revised()
    elif arguments == ["--family"]:
        run_family()
    else:
        raise SystemExit("usage: make_e5_fits.py --prefit | --positive-control-revised | --family [--smoke]")
