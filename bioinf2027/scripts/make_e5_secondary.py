"""E5, the secondary analyses and the diagnostics: everything of Section 8 that is not the family, unadjusted.

Design: bioinf2027/plans/E5.md, Sections 3 to 8, 11 and 16 to 20 (fixed before any value). Built on the family's
harness, imported unchanged: bioinf2027/scripts/make_e5_fits.py (inputs, coverage, settings, folds, seeds, the
controls and the gene bootstrap) and bioinf2027/scripts/e5fit.py (the ridge on Bc residuals, the penalty by inner
folds, the intersection-union statistic). Five parts, each with its own outputs and record:

- size: the size association of (b) (Sections 2, 4, 8 and 17): the Spearman between genes of B0 with the fixed-cell
  energy distance (the family's version), the all-cells energy distance and the n-scaled energy test statistic,
  gene bootstrap, unadjusted, and the statement rule of Section 8; from genes_<screen>.csv and G1's B0 only.
- prefit50: the oracle ceiling and the pipeline positive control of the 50-cell version (Section 16.1), as
  make_e5_fits.ceiling_task and positive_control_task with the label b50 (the latter called unchanged; widths 512 and
  768 as planned, 1,536 as planned and revised, Section 20). The main part's 50-cell rows read this control and
  the 50-cell gates, and the main part refuses to run without these outputs.
- main (ridge, the two Replogle screens; a real run needs the finished family and prefit50): the family recomputed, since its
  predictions are not saved, and checked against bioinf2027/results/E5/family/family.csv; from it the fold
  diagnostics (the Spearman within each fold and the pooled Spearman of predictions centred within fold, as E6,
  with R-squared and the mean predictor's R-squared beside), the between-embedding contrasts (descriptive, in a
  fixed order, no ranking), power (2.80 SE and the size detectable at Holm's first step over 24, (3.08 + 0.84) SE),
  the family nearest-neighbour baseline and the gene bootstrap with symbol prefixes as clusters. Then B1 alone as
  the bar; the 10 kb, full-space, depth-matched and 50-cell versions of (b); (b) residualised on the own shift B2
  out of fold, with its reading rule; prefix-grouped folds; the diagnostics (B1, the control detection rate and B0
  predicted from e alone, and [Bc; B0] as the bar); the agreement of (a) and (b) between genes.
- forest: the family's statistics under E1b's random forest (bioinf2027.heads.FOREST), head seeds 0 to 2, every
  Spearman averaged over the seeds inside each replicate (E1_design.md, Section 10); Bc enters as its within-screen
  rank columns, without the spline. Each fit is kept under the data root and reused when its specification matches.
- vcc: VCC training genes to VCC test genes, (a) at G3 and G4 and (b) at 27, 10 and 50 cells, and to the Replogle
  genes that are not VCC training or validation targets ((a) at 27 and 10 cells); Bc of three columns (E1a's
  ceiling exists for VCC at row level only); bootstrap seed 20271125; descriptive.

Every row is labelled "secondary, unadjusted" or "diagnostic"; none enters Holm. Gates and positive-control
columns carry the family's rule for the setting, endpoint and width (the positive control is the ridge's; the
50-cell rows carry the 50-cell gates and control).
Outputs: bioinf2027/results/E5/secondary/ (tables and secondary_<part>.json), out-of-fold predictions and the
forest fits under $IGA_STORAGE_ROOT/bioinf2027/E5/secondary/. With --smoke: 200 replicates, the prefit50, main and
forest parts in two settings (one within, one between the screens), the forest at 5 trees and head seed 0; outputs to
$IGA_STORAGE_ROOT/bioinf2027/scratch/E5_secondary_smoke/, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import functools  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, e1inputs, heads, resample  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, sha256, strict_json  # noqa: E402
from bioinf2027.stats import percentile_interval  # noqa: E402

import e5fit  # noqa: E402
import e5lib  # noqa: E402
import fit_slots  # noqa: E402
import make_e5_fits as fits  # noqa: E402
import safe_pickle  # noqa: E402

REPO = fits.REPO
RESULTS = fits.RESULTS
ENDPOINT_B = fits.ENDPOINT_B
PLAN = "bioinf2027/plans/E5.md, Sections 3 to 8, 11 and 16 to 20 (secondary, unadjusted, and diagnostics)"
SCREENS = fits.SCREENS
SETTINGS = fits.SETTINGS
SETTING_SEEDS = fits.SETTING_SEEDS
EMBEDDINGS = tuple(fits.EMBEDDINGS)
VCC_SEED = 20271125
FOREST_SEEDS = (0, 1, 2)
POWER_UNADJUSTED = 2.80  # 1.96 + 0.84
POWER_HOLM_FIRST = 3.08 + 0.84  # Holm's first step over 24 tests, as in the family's rows
TOLERANCE = 1e-9  # the recomputed family against family.csv (same code, one thread per worker)
PARTS = ("vcc", "size", "prefit50", "main", "forest")  # the default order; --part runs the parts in the order given
SIZE_STATEMENT = ("a size association appears for a statistic that grows with the number of cells and is absent at fixed "
                  "cells, as for the target-gene endpoint")  # Section 8, verbatim
PC50 = "50-cell control"
SMOKE_SETTINGS = ("within RPE1", "RPE1 to K562-essential")
SMOKE_TREES = 5
B = fits.B
WORKERS = 4
SMOKE = False
SLOTS = False  # --slots: the main part runs one process per task (fit_slots.py); changes no value
WORKERS_FILE = None  # --workers-file: the number of tasks in progress, read as the main part runs
DATA: dict = {}  # this script's inputs; the harness's stay in fits.DATA

PRIMARY = "the family, recomputed"
ITEMS = {  # item -> labels, bar, folds, role (Section 8)
    PRIMARY: {"labels": ("a", "b"), "bar": "Bc", "folds": "gene folds", "role": "family, recomputed"},
    "B1 as the bar": {"labels": ("a", "b"), "bar": "B1", "folds": "gene folds", "role": "secondary, unadjusted"},
    "10 kb neighbours removed": {"labels": ("b_10kb",), "bar": "Bc", "folds": "gene folds", "role": "secondary, unadjusted"},
    "full space": {"labels": ("b_full",), "bar": "Bc", "folds": "gene folds", "role": "secondary, unadjusted"},
    "depth-matched controls": {"labels": ("b_depth",), "bar": "Bc", "folds": "gene folds", "role": "secondary, unadjusted"},
    "50-cell version": {"labels": ("b50",), "bar": "Bc", "folds": "gene folds", "role": "secondary, unadjusted"},
    "(b) residualised on B2": {"labels": ("b_minus_B2",), "bar": "Bc", "folds": "gene folds", "role": "secondary, unadjusted"},
    "prefix-grouped folds": {"labels": ("a", "b"), "bar": "Bc", "folds": "prefix-grouped folds", "role": "secondary, unadjusted"},
    "[Bc; B0] as the bar": {"labels": ("a", "b"), "bar": "[Bc; B0]", "folds": "gene folds", "role": "diagnostic"},
    "B1 from e": {"labels": ("B1",), "bar": None, "folds": "gene folds", "role": "diagnostic"},
    "detection rate from e": {"labels": ("detection_rate",), "bar": None, "folds": "gene folds", "role": "diagnostic"},
    "B0 from e": {"labels": ("B0",), "bar": None, "folds": "gene folds", "role": "diagnostic"},
}
PREFIX_ROWS = "prefixes as bootstrap clusters"
RESPONSE_PER_KNOCKDOWN = "response per unit knockdown (both differences resolved above zero, unadjusted)"
VCC_TEST = "VCC training to VCC test"
VCC_SETTINGS = {VCC_TEST: None, "VCC training to RPE1": "RPE1", "VCC training to K562-essential": "K562-essential"}
VCC_LABELS = {  # (source column, target column, name)
    VCC_TEST: (("a27", "a27", "(a), G3, 27 cells"), ("a10", "a10", "(a), G4, 10 cells"), ("b27", "b27", "(b), 27 cells"),
               ("b10", "b10", "(b), 10 cells"), ("b50", "b50", "(b), 50 cells")),
    "Replogle": (("a27", "a", "(a), G3 to G1, 27 cells"), ("a10", "a10", "(a), G4 to G2, 10 cells")),
}
VCC_BC = ["rank_B1", "rank_detection_rate", "rank_control_sd"]


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def out_dir() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E5_secondary_smoke"
    return RESULTS / "secondary"


def data_dir() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E5_secondary_smoke" / "data"
    return config.storage_root() / "bioinf2027" / "E5" / "secondary"


def active_settings() -> tuple:
    return SMOKE_SETTINGS if SMOKE else tuple(SETTINGS)


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


# --------------------------------------------------------------------------- gene families (Section 3)

_LEADING = re.compile(r"[^0-9]*")


def symbol_prefix(symbol: str) -> str:
    """The characters before the first digit of the upper-case symbol (RPL13A -> RPL); a symbol without one is itself."""
    text = str(symbol).strip().upper()
    head = _LEADING.match(text).group(0)
    return head if head else text


def prefix_folds(genes, prefix_of: dict, seed: int) -> dict:
    """Five folds that keep every prefix whole: the sorted prefixes permuted with the seed, fold = position mod 5."""
    prefixes = sorted({prefix_of[g] for g in genes})
    order = np.random.default_rng(seed).permutation(len(prefixes))
    fold_of = {prefixes[i]: int(position % e5lib.N_FOLDS) for position, i in enumerate(order)}
    return {g: fold_of[prefix_of[g]] for g in genes}


# --------------------------------------------------------------------------- inputs


def code_digests() -> dict:
    modules = (sys.modules[__name__], fits, e5fit, e5lib, heads) + ((fit_slots,) if SLOTS else ())
    return {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in modules}


def checked_json(path: Path, sha: str | None = None) -> dict:
    if sha is not None and sha256(path) != sha:
        raise Stop("an input does not match its record: %s" % path.name)
    value = json.loads(path.read_text())
    if value.get("smoke"):
        raise Stop("%s comes from a smoke run" % path.name)
    return value


def add_columns(frame: pd.DataFrame, genes_b: pd.DataFrame, g1: pd.DataFrame, b50_column: str) -> pd.DataFrame:
    """The labels and inputs of the secondary analyses beside the harness's frame (gene rows, Ensembl index)."""
    out = frame.copy()
    gb = genes_b.set_index("target_ensg") if "target_ensg" in genes_b.columns else genes_b
    out["symbol"] = gb["target_symbol"].reindex(out.index).astype(str)
    for name, column in (("b_10kb", "b27_10kb"), ("b_full", "b27_full"), ("b_depth", "b27_depth"), ("b50", b50_column)):
        out[name] = gb[column].reindex(out.index).astype(np.float64)
    eligible = g1[g1["n_pool"] >= 27]
    for c in ("B0", "B2"):
        out[c] = eligible.groupby(eligible["target_ensg"].astype(str))[c].median().reindex(out.index).astype(np.float64)
        out[f"rank_{c}"] = e5lib.ranks01(out[c].to_numpy(dtype=np.float64))
    return out


def residual_on(frame: pd.DataFrame, label: str, column: str, folds_of: dict) -> np.ndarray:
    """The label minus its out-of-fold prediction from the rank spline of one column (Section 17's spline and folds)."""
    folds = np.array([folds_of[g] for g in frame.index])
    return e5lib.out_of_fold_residuals(frame[[column]].to_numpy(dtype=np.float64), frame[label].to_numpy(dtype=np.float64), folds)


def load_replogle() -> None:
    """The harness's inputs (make_e5_fits.load, load_embeddings, covered) and the secondary columns, each checked."""
    if DATA.get("loaded"):  # several parts in one process load once
        return
    fits.DATA.clear()
    fits.DATA.update(fits.load())
    fits.DATA["embeddings"] = fits.load_embeddings()
    fits.DATA["covered"] = fits.covered()
    b_record = checked_json(ENDPOINT_B / "endpoint_b.json")
    g1 = e1cells.load_tables("G1")
    inputs = {"harness": fits.DATA["record"], "G1_manifest": file_record(REPO / "bioinf2027" / "endpoints" / "G1" / "manifest.json")}
    DATA["label_sources"] = {}
    for s in SCREENS:
        if fits.DATA["record"][s]["b_column"] != "b27":
            raise Stop("%s: the family's (b) is %s; the sensitivities of Section 4 are fixed for the 27-cell pooled version" % (s, fits.DATA["record"][s]["b_column"]))
        genes_b = fits.checked_csv(ENDPOINT_B / f"genes_{s}.csv", b_record["outputs"][f"genes_{s}.csv"]["sha256"])
        level50 = fits.DATA["gates_b"]["gates"][s]["50"]["primary_version"]
        b50_column = "b50_gem" if level50.startswith("gem") else "b50"
        frame = add_columns(fits.DATA["screens"][s]["frame"], genes_b, g1[s], b50_column)
        frame["b_minus_B2"] = residual_on(frame, "b", "rank_B2", fits.DATA["within_folds"][s])
        fits.DATA["screens"][s]["frame"] = frame
        inputs[f"genes_{s}.csv"] = file_record(ENDPOINT_B / f"genes_{s}.csv")
        DATA["label_sources"][s] = {"a": "G1 y, median over the gene's eligible constructs", "b": "b27", "b_10kb": "b27_10kb", "b_full": "b27_full",
                                    "b_depth": "b27_depth", "b50": b50_column,
                                    "b_minus_B2": "b27 minus its out-of-fold rank-spline prediction from B2 (G1, gene median), within-screen folds",
                                    "B1": "B1 (control cells)", "detection_rate": "control detection rate", "B0": "G1 B0, gene median"}
    # run_setting reads Bc's columns by label; the 50-cell label takes (b)'s three (in this process; the file is unchanged)
    fits.BC_COLUMNS.setdefault("b50", fits.BC_COLUMNS["b"])
    prefix_of = {}
    for s in SCREENS:  # a gene's prefix from its symbol in the first screen that holds it
        frame = fits.DATA["screens"][s]["frame"]
        for g, symbol in zip(frame.index, frame["symbol"]):
            prefix_of.setdefault(str(g), symbol_prefix(symbol))
    DATA["prefix_of"] = prefix_of
    DATA["prefix_folds"] = {s: prefix_folds(fits.DATA["screens"][s]["frame"].index, prefix_of, e5lib.WITHIN_FOLD_SEED) for s in SCREENS}
    DATA["prefix_folds"]["union"] = prefix_folds(sorted(fits.DATA["union_folds"]), prefix_of, fits.UNION_FOLD_SEED)
    prefit = checked_json(RESULTS / "prefit" / "prefit.json")
    revised = checked_json(RESULTS / "prefit" / "prefit_revised.json")
    DATA["pc"] = fits.checked_csv(RESULTS / "prefit" / "positive_control.csv", prefit["outputs"]["positive_control.csv"]["sha256"])
    DATA["pc_revised"] = fits.checked_csv(RESULTS / "prefit" / "positive_control_revised.csv",
                                          revised["outputs"]["positive_control_revised.csv"]["sha256"])
    DATA["gates_a"] = checked_json(RESULTS / "prefit" / "gates_a.json", prefit["outputs"]["gates_a.json"]["sha256"])["gates"]
    inputs.update({"prefit": file_record(RESULTS / "prefit" / "prefit.json"), "prefit_revised": file_record(RESULTS / "prefit" / "prefit_revised.json"),
                   "embeddings": {n: json.loads((RESULTS / "identity" / f"{f}.json").read_text())["output"] for n, f in fits.EMBEDDINGS.items()}})
    DATA["inputs"] = inputs
    DATA["code"] = code_digests()
    DATA["covered_counts"] = {s: len(v) for s, v in fits.DATA["covered"].items()}
    DATA["prefixes"] = {s: len({prefix_of[g] for g in fits.DATA["screens"][s]["frame"].index}) for s in SCREENS}
    DATA["loaded"] = True


def load_family() -> pd.DataFrame | None:
    """The family's table, for the check of the recomputation; a real run of the main part needs it."""
    folder = RESULTS / "family"
    if not (folder / "family.json").is_file():
        if SMOKE:
            return None
        raise Stop("the family has not finished: %s is missing" % config.portable(folder / "family.json"))
    record = checked_json(folder / "family.json")
    table = fits.checked_csv(folder / "family.csv", record["outputs"]["family.csv"]["sha256"])
    DATA["inputs"]["family"] = file_record(folder / "family.json")
    return table


# --------------------------------------------------------------------------- rows and fits


def endpoint_of(label: str) -> str:
    if label in ("a", "a27", "a10"):
        return "a"
    if label.startswith("b"):
        return "b"
    return "none"


def bar_columns(bar: str | None, label: str) -> list:
    """The unpenalised columns of a model's bar: Bc (four columns for (a), three for (b)), B1 alone, [Bc; B0] or none."""
    if bar is None:
        return []
    if bar == "B1":
        return ["rank_B1"]
    bc = [f"rank_{c}" for c in fits.BC_COLUMNS["a" if label == "a" else "b"]]
    return bc + (["rank_B0"] if bar == "[Bc; B0]" else [])


def model_names(bar: str | None) -> dict:
    if bar is None:
        return {"e": "e", "C": "C", "S": "S"}
    inner = bar.strip("[]")
    return {"bar": bar, "e": f"[{inner}; e]", "C": f"[{inner}; C]", "S": f"[{inner}; S]"}


def usable(frame: pd.DataFrame, label: str, columns: list, genes) -> list:
    """As make_e5_fits.rows_of: rows with the label and every bar column, among the given genes."""
    ok = frame[label].notna() & frame[columns].notna().all(axis=1)
    if genes is not None:
        ok &= frame.index.isin(genes)
    return list(frame.index[ok])


def setting_rows(setting: str, label: str, columns: list, genes, folds: str = "gene folds") -> dict:
    """make_e5_fits.rows_of with the label, the bar's columns and the fold assignment as arguments."""
    source, target = SETTINGS[setting]
    fs, ft = fits.DATA["screens"][source]["frame"], fits.DATA["screens"][target]["frame"]
    if folds == "gene folds":
        fold_of = fits.DATA["within_folds"][source] if source == target else fits.DATA["union_folds"]
    else:
        fold_of = DATA["prefix_folds"][source] if source == target else DATA["prefix_folds"]["union"]
    return {"source": source, "target": target, "src_frame": fs, "tgt_frame": ft, "src_label": label, "tgt_label": label,
            "bc": list(columns), "src": usable(fs, label, columns, genes), "tgt": usable(ft, label, columns, genes),
            "folds": fold_of, "ks": tuple(range(e5lib.N_FOLDS))}


def forest_model(seed: int):
    model = heads.forest(seed)
    if SMOKE:
        model.set_params(n_estimators=SMOKE_TREES)
    return model


def fit_folds(R: dict, models: dict, head: str = "ridge", seed: int = 0) -> dict:
    """The fold loop of make_e5_fits.run_setting with the rows of ``R``; the ridge path calls e5fit.fit_predict exactly
    as run_setting does (bioinf2027/tests/test_e5_secondary.py checks the predictions are equal). With head "forest":
    the forest on [the bar's rank columns; the input] (bar alone: the rank columns), head seed ``seed``."""
    fs, ft = R["src_frame"], R["tgt_frame"]
    genes, fold_ids, folds, train_sizes, mean_pred = [], [], [], [], []
    predictions, records = {m: [] for m in models}, {m: [] for m in models}
    for k in R["ks"]:
        tr = [g for g in R["src"] if R["folds"][g] != k]
        te = [g for g in R["tgt"] if R["folds"][g] == k]
        if not te:
            continue
        bc_tr, y_tr = fs.loc[tr, R["bc"]].to_numpy(dtype=np.float64), fs.loc[tr, R["src_label"]].to_numpy(dtype=np.float64)
        bc_te = ft.loc[te, R["bc"]].to_numpy(dtype=np.float64)
        genes += te
        fold_ids.append(k)
        folds += [k] * len(te)
        train_sizes.append(len(tr))
        mean_pred.append(np.full(len(te), float(np.mean(y_tr))))
        for name, provider in models.items():
            if head == "ridge":
                if provider is None:
                    pred, rec = e5fit.fit_predict(bc_tr, y_tr, np.array(tr), bc_te)
                else:
                    pred, rec = e5fit.fit_predict(bc_tr, y_tr, np.array(tr), bc_te, provider(R["source"], tr, k, tr), provider(R["target"], te, k, tr))
            else:
                x_tr = bc_tr if provider is None else np.hstack([bc_tr, provider(R["source"], tr, k, tr)])
                x_te = bc_te if provider is None else np.hstack([bc_te, provider(R["target"], te, k, tr)])
                started = time.time()
                with heads.single_thread():
                    pred = forest_model(seed).fit(x_tr, y_tr).predict(x_te)
                rec = {"seconds": time.time() - started, "columns": int(x_tr.shape[1])}
            predictions[name].append(np.asarray(pred, dtype=np.float64))
            records[name].append(rec)
    y = ft.loc[genes, R["tgt_label"]].to_numpy(dtype=np.float64)
    return {"genes": genes, "y": y, "folds": np.array(folds, dtype=np.int64), "fold_ids": fold_ids, "train_sizes": train_sizes,
            "predictions": {m: np.concatenate(v) if v else np.empty(0) for m, v in predictions.items()},
            "mean_predictor": np.concatenate(mean_pred) if mean_pred else np.empty(0), "records": records}


def family_neighbour(R: dict) -> tuple[list, np.ndarray, int]:
    """Section 3's family baseline, in fit_folds' order: a test gene takes the mean training label of the genes with its
    prefix, otherwise the training mean. Returns the test genes, the predictions and how many had a family member."""
    fs = R["src_frame"]
    genes, pred, with_family = [], [], 0
    for k in R["ks"]:
        tr = [g for g in R["src"] if R["folds"][g] != k]
        te = [g for g in R["tgt"] if R["folds"][g] == k]
        if not te:
            continue
        y_tr = fs.loc[tr, R["src_label"]].to_numpy(dtype=np.float64)
        means = pd.Series(y_tr).groupby(np.array([DATA["prefix_of"][g] for g in tr])).mean().to_dict()
        overall = float(np.mean(y_tr))
        for g in te:
            hit = means.get(DATA["prefix_of"][g])
            with_family += hit is not None
            pred.append(overall if hit is None else float(hit))
        genes += te
    return genes, np.array(pred, dtype=np.float64), int(with_family)


def models_for(name: str, genes, with_bar: bool = True) -> tuple[dict, int]:
    """The family's providers (make_e5_fits.family_task): e, the Gaussian control of its width and the shuffled e."""
    vectors = fits.DATA["embeddings"][name]
    width = int(next(iter(vectors.values())).size)
    shuffled = e5fit.shuffled({g: vectors[g] for g in genes}, genes)
    models = {"e": fits.embedding_provider(name, vectors), "C": fits.gaussian_provider(width), "S": fits.embedding_provider(name, shuffled)}
    return ({"bar": None, **models} if with_bar else models), width


def compact(result: dict) -> dict:
    return {k: result[k] for k in ("genes", "y", "folds", "fold_ids", "train_sizes", "predictions", "mean_predictor")}


# --------------------------------------------------------------------------- statistics


def contrast_row(stats: dict, base: dict, with_bar: bool = True) -> dict:
    """The family's statistic on one table of draws: e - C, e - S, their intersection-union, e - bar, and power."""
    dg, ds = fits.difference(stats, "e", "C"), fits.difference(stats, "e", "S")
    iu = e5fit.iu_statistic((dg["estimate"], dg["draws"]), (ds["estimate"], ds["draws"]))
    row = {**base, "spearman_bar": float(stats["point"]["bar"]) if with_bar else float("nan"),
           **{f"spearman_{m}": float(stats["point"][m]) for m in ("e", "C", "S")},
           **{f"minus_gaussian_{k}": v for k, v in dg.items() if k != "draws"},
           **{f"minus_shuffled_{k}": v for k, v in ds.items() if k != "draws"}}
    if with_bar:
        db = fits.difference(stats, "e", "bar")
        row.update({f"minus_bar_{k}": v for k, v in db.items() if k != "draws"})
    else:
        row.update({f"minus_bar_{k}": float("nan") for k in ("estimate", "lo", "hi", "p", "se")})
    se = max(dg["se"], ds["se"])
    row.update({"iu_estimate": iu["estimate"], "iu_p": iu["p"], "both_resolved_above_zero": bool(iu["p"] < 0.05 and iu["estimate"] > 0),
                "power_2.80se": POWER_UNADJUSTED * se, "replicate_index_sha256": stats["index_sha256"]})
    return row


def seed_averaged(per_seed: list[dict]) -> dict:
    """Spearman per head seed, averaged over the seeds inside each replicate (E1_design.md, Section 10)."""
    if len({p["index_sha256"] for p in per_seed}) != 1:
        raise Stop("the head seeds were scored on different replicates")
    names = per_seed[0]["names"]
    return {"names": names, "point": {m: float(np.mean([p["point"][m] for p in per_seed])) for m in names},
            "draws": {m: np.mean([p["draws"][m] for p in per_seed], axis=0) for m in names}, "index_sha256": per_seed[0]["index_sha256"]}


def context(label: str, setting: str, width: int) -> dict:
    """The family's gate and the setting's (ridge) positive control for the endpoint and width; none for a diagnostic label."""
    out = {"gates_passed": None, "positive_control_resolved": None, "positive_control_version": ""}
    endpoint = endpoint_of(label)
    if endpoint == "none" or setting not in SETTINGS:
        return out
    target = SETTINGS[setting][1]
    g = DATA["gates_a"][target] if endpoint == "a" else fits.DATA["gates_b"]["gates"][target]["50" if label == "b50" else "27"]
    out["gates_passed"] = bool(g["reliability_gate_passed"] and g["beyond_bc_gate_passed"])
    revised = width > fits.REVISED_COMPONENTS
    version = fits.REVISED if revised else "planned"
    if label == "b50":  # the 50-cell version's own control (part prefit50; Section 16.1)
        table = DATA["pc50"]
        hit = table[(table["setting"] == setting) & (table["endpoint"] == "b50") & (table["nominal_width"] == width) & (table["version"] == version)]
        version = f"{version}, {PC50}"
    else:
        table = DATA["pc_revised"] if revised else DATA["pc"]
        hit = table[(table["setting"] == setting) & (table["endpoint"] == endpoint) & (table["nominal_width"] == width)]
    out["positive_control_resolved"] = bool(hit["resolved"].iloc[0]) if len(hit) else None
    out["positive_control_version"] = version
    return out


def fold_rows(result: dict, base: dict, names: dict) -> list[dict]:
    """Within-fold Spearman and the pooled Spearman with predictions centred within fold, as E6; R-squared beside."""
    y, folds = result["y"], result["folds"]
    rows = []
    for model, pred in result["predictions"].items():
        centred = pred.copy()
        for k in result["fold_ids"]:
            centred[folds == k] -= pred[folds == k].mean()
        entry = {**base, "model": names[model], "pooled_spearman": spearman(pred, y), "pooled_centred_within_fold": spearman(centred, y),
                 "pooled_r2": heads.pooled_r2(y, pred)}
        for i, k in enumerate(result["fold_ids"]):
            mask = folds == k
            record = result["records"][model][i]
            entry.update({f"fold_{k}_test_genes": int(mask.sum()), f"fold_{k}_training_genes": int(result["train_sizes"][i]),
                          f"fold_{k}_spearman": spearman(pred[mask], y[mask]), f"fold_{k}_alpha": record.get("alpha", float("nan")),
                          f"fold_{k}_at_grid_edge": record.get("at_grid_edge")})
        rows.append(entry)
    rows.append({**base, "model": "mean predictor (R-squared only)", "pooled_r2": heads.pooled_r2(y, result["mean_predictor"])})
    return rows


def base_of(item: str, name: str, width: int, label: str, setting: str, result: dict, head: str, folds: str, clusters: str) -> dict:
    spec = ITEMS.get(item, {"bar": "Bc", "role": "secondary, unadjusted"})
    source = DATA.get("label_sources", {}).get(SETTINGS[setting][1], {}).get(label, label) if setting in SETTINGS else label
    return {"item": item, "role": spec["role"], "embedding": name, "width": width, "endpoint": endpoint_of(label), "label": label,
            "label_source": source, "setting": setting, "bar": spec["bar"] or "none (intercept)", "head": head, "folds": folds,
            "clusters": clusters, "n_training_genes": ";".join(str(n) for n in result["train_sizes"]), "n_test_genes": len(result["genes"])}


# --------------------------------------------------------------------------- the main part


def main_task(args) -> dict:
    item, name, label, setting = args
    spec = ITEMS[item]
    e5fit.SVD_FALLBACKS[0] = 0
    started = time.time()
    genes = fits.setting_genes(setting)
    R = setting_rows(setting, label, bar_columns(spec["bar"], label), genes, spec["folds"])
    models, width = models_for(name, genes, with_bar=spec["bar"] is not None)
    result = fit_folds(R, models)
    stats = fits.spearman_draws(result, SETTING_SEEDS[setting])
    base = base_of(item, name, width, label, setting, result, "ridge", spec["folds"], "genes")
    edges = int(sum(r.get("at_grid_edge", False) for m in ("e", "C", "S") for r in result["records"][m]))
    row = {**contrast_row(stats, base, spec["bar"] is not None), **context(label, setting, width), "ridge_alpha_at_grid_edge": edges}
    if item == "(b) residualised on B2":
        row["reading"] = RESPONSE_PER_KNOCKDOWN if row["both_resolved_above_zero"] else "not read as response per unit knockdown"
    out = {"key": args, "result": compact(result)}
    if item == PRIMARY:
        out["folds"] = fold_rows(result, {k: base[k] for k in ("embedding", "width", "endpoint", "setting", "n_test_genes")}, model_names("Bc"))
        clusters = [DATA["prefix_of"][g] for g in result["genes"]]
        by_prefix = fits.spearman_draws({**result, "genes": clusters}, SETTING_SEEDS[setting])
        prefix_base = {**base, "item": PREFIX_ROWS, "role": "secondary, unadjusted", "clusters": "symbol prefixes"}
        out["prefix_row"] = {**contrast_row(by_prefix, prefix_base), **context(label, setting, width), "ridge_alpha_at_grid_edge": edges,
                             "n_clusters": len(set(clusters))}
    row["svd_fallbacks"] = int(e5fit.SVD_FALLBACKS[0])
    row["seconds"] = round(time.time() - started, 1)
    if "prefix_row" in out:
        out["prefix_row"]["svd_fallbacks"] = row["svd_fallbacks"]
    out["row"] = row
    return out


def task_spec(stage: str, key, extra: str = "") -> str:
    """What one saved task of the main part depends on: its key, the replicates, the settings, the code (by digest)
    and every input of the part (by digest)."""
    return json.dumps(strict_json({"stage": stage, "key": [str(k) for k in key], "replicates": B, "smoke": SMOKE,
                                   "settings": list(active_settings()), "code": DATA["code"], "inputs": DATA["inputs"], "extra": extra}),
                      sort_keys=True)


def saved(stage: str, function, args, key, extra: str = "") -> tuple:
    """One task of the main part, saved as it ends and reused when its specification is unchanged, so that a stop loses
    at most the tasks in progress (gate G4, item 4). The saved value is the task's own return value, pickled (protocol
    4) and read back without running code (safe_pickle, numpy arrays and scalars only), so no value changes.
    Returns (the task's value, whether it was reused)."""
    spec = task_spec(stage, key, extra)
    path = data_dir() / "main_tasks" / ("%s_%s.pkl" % (stage, hashlib.sha256(spec.encode()).hexdigest()[:24]))
    if path.is_file():
        value = safe_pickle.load(path, safe_pickle.NUMPY_ARRAYS)
        if value["spec"] == spec:
            return value["output"], True
    output = function(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("%s.tmp%d" % (path.name, os.getpid()))
    with open(temporary, "wb") as handle:
        pickle.dump({"spec": spec, "output": output}, handle, protocol=4)
    os.replace(temporary, path)
    return output, False


def main_task_saved(args) -> tuple:
    return saved("main", main_task, args, args)


def contrast_task_saved(args) -> tuple:
    """Stage two's task; its specification also holds a digest of the stage-one results it reads."""
    label, setting, results = args
    digest = hashlib.sha256()
    for n in sorted(results):
        r = results[n]
        digest.update(("%s|%s|" % (n, "|".join(str(g) for g in r["genes"]))).encode())
        digest.update(np.ascontiguousarray(r["y"], dtype=np.float64).tobytes())
        for m in sorted(r["predictions"]):
            digest.update(m.encode())
            digest.update(np.ascontiguousarray(r["predictions"][m], dtype=np.float64).tobytes())
    return saved("contrast", contrast_task, args, (label, setting), digest.hexdigest())


def _numbered(function, task: tuple) -> tuple:
    return task[0], function(task[1])


def slots_map(function, listed: list, stage: str) -> list:
    """The values pool.map would return, in the listed order, each task in a process of its own, so that the memory
    rules can take one task away at a time (the author's change of 2026-11-15, bioinf2027/GATE.md; fit_slots.py)."""
    killed = WORKERS_FILE.with_name(WORKERS_FILE.name + ".killed") if WORKERS_FILE is not None else None
    numbered = list(enumerate(listed))
    done, failed, shed = fit_slots.run_slots(
        numbered, functools.partial(_numbered, function), lambda t: "%s task %d (%s)" % (stage, t[0], " | ".join(a for a in (t[1] if isinstance(t[1], tuple) else ()) if isinstance(a, str))),
        lambda: fit_slots.limit_from(WORKERS_FILE, WORKERS), "e5 secondary main, %s" % stage, time.time(), killed=killed)
    if failed:
        raise Stop("%d tasks of %s failed: %s" % (len(failed), stage, "; ".join("%s: %s" % (f["fit"], f["error"]) for f in failed[:5])))
    DATA.setdefault("shed", {})[stage] = shed
    values = dict(done)
    assert sorted(values) == list(range(len(listed)))
    return [values[i] for i in range(len(listed))]


def main_args() -> list:
    args = [(item, name, label, setting) for item, spec in ITEMS.items() for name in EMBEDDINGS for label in spec["labels"] for setting in active_settings()]
    return sorted(args, key=lambda a: -fits.WIDTHS[a[1]])  # the widest first; stable, so the order is fixed


def contrast_task(args) -> dict:
    """Per endpoint and setting: the between-embedding contrasts and the family baseline, on one table of draws."""
    label, setting, results = args
    names = [n for n in EMBEDDINGS if n in results]
    first = results[names[0]]
    for n in names[1:]:
        r = results[n]
        if r["genes"] != first["genes"] or not np.array_equal(r["y"], first["y"]):
            raise Stop("%s, %s: the embeddings were scored on different test genes" % (label, setting))
        if not np.array_equal(r["predictions"]["bar"], first["predictions"]["bar"]):
            raise Stop("%s, %s: the Bc predictions differ between embeddings" % (label, setting))
    R = setting_rows(setting, label, bar_columns("Bc", label), fits.setting_genes(setting))
    nn_genes, neighbour, with_family = family_neighbour(R)
    if nn_genes != first["genes"]:
        raise Stop("%s, %s: the family baseline is not on the family's test genes" % (label, setting))
    predictions = {"family neighbour": neighbour, "Bc": first["predictions"]["bar"]}
    for n in names:
        predictions.update({f"{n} e": results[n]["predictions"]["e"], f"{n} C": results[n]["predictions"]["C"]})
    stats = fits.spearman_draws({"genes": first["genes"], "y": first["y"], "predictions": predictions}, SETTING_SEEDS[setting])
    for n in names:  # the excess over the vector's own Gaussian control, per replicate
        stats["point"][f"{n} e-C"] = stats["point"][f"{n} e"] - stats["point"][f"{n} C"]
        stats["draws"][f"{n} e-C"] = stats["draws"][f"{n} e"] - stats["draws"][f"{n} C"]
    common = {"endpoint": endpoint_of(label), "setting": setting, "n_test_genes": len(first["genes"]), "replicate_index_sha256": stats["index_sha256"]}
    between = []
    for a, b in itertools.combinations(names, 2):  # the fixed order of EMBEDDINGS, not an order by value
        raw, excess = fits.difference(stats, f"{a} e", f"{b} e"), fits.difference(stats, f"{a} e-C", f"{b} e-C")
        between.append({"role": "secondary, descriptive (no ranking)", **common, "first": a, "second": b,
                        **{f"e_difference_{k}": v for k, v in raw.items() if k != "draws"},
                        **{f"excess_over_gaussian_difference_{k}": v for k, v in excess.items() if k != "draws"}})
    lo, hi, _ = percentile_interval(stats["draws"]["family neighbour"])
    baseline = []
    for n in names:
        d = fits.difference(stats, f"{n} e", "family neighbour")
        bc = fits.difference(stats, "Bc", "family neighbour")
        baseline.append({"role": "secondary, unadjusted", **common, "embedding": n, "test_genes_with_a_training_family_member": with_family,
                         "spearman_family_neighbour": stats["point"]["family neighbour"], "spearman_family_neighbour_lo": lo,
                         "spearman_family_neighbour_hi": hi, "spearman_Bc": stats["point"]["Bc"], "spearman_[Bc; e]": stats["point"][f"{n} e"],
                         **{f"e_minus_family_neighbour_{k}": v for k, v in d.items() if k != "draws"},
                         **{f"Bc_minus_family_neighbour_{k}": v for k, v in bc.items() if k != "draws"}})
    return {"between": between, "baseline": baseline}


def agreement_rows(seed_of: dict, frames: dict, covered: dict, pairs: tuple) -> list[dict]:
    """Spearman between genes of (a) and each version of (b), with a gene-bootstrap interval."""
    rows = []
    for screen, frame in frames.items():
        for a_col, b_col in pairs:
            for subset, genes in (("covered genes", covered.get(screen)), ("all genes with both labels", None)):
                ok = frame[a_col].notna() & frame[b_col].notna()
                if genes is not None:
                    ok &= frame.index.isin(genes)
                g = [str(x) for x in frame.index[ok]]
                a, b = frame.loc[ok, a_col].to_numpy(dtype=np.float64), frame.loc[ok, b_col].to_numpy(dtype=np.float64)
                point = resample.ClusterDraw(weights=np.ones((1, len(g))), clusters=np.array([]), seed=-1, index_sha256="")
                draw = resample.cluster_draw(np.array(g), B, seed_of[screen])
                rho = float(resample.weighted_spearman(b, a, point)[0, 0])
                draws = resample.weighted_spearman(b, a, draw)[:, 0]
                lo, hi, missing = percentile_interval(draws)
                rows.append({"role": "secondary, unadjusted", "screen": screen, "first": a_col, "second": b_col, "subset": subset,
                             "n_genes": len(g), "spearman": rho, "lo": lo, "hi": hi, "replicates_not_finite": missing,
                             "seed": seed_of[screen], "replicate_index_sha256": draw.index_sha256})
    return rows


def check_against_family(rows: list[dict], family: pd.DataFrame | None) -> dict:
    """The recomputed family equals family.csv: points always; in a real run also intervals, p-values and SE."""
    if family is None:
        return {"compared": False, "reason": "no family table (smoke run)"}
    pairs = {"spearman_bar": "spearman_Bc", "spearman_e": "spearman_[Bc; e]", "spearman_C": "spearman_[Bc; C]", "spearman_S": "spearman_[Bc; S]",
             "minus_gaussian_estimate": "minus_gaussian_estimate", "minus_shuffled_estimate": "minus_shuffled_estimate",
             "minus_bar_estimate": "minus_Bc_estimate", "iu_estimate": "iu_estimate"}
    if not SMOKE:
        for k in ("lo", "hi", "p", "se"):
            pairs.update({f"minus_gaussian_{k}": f"minus_gaussian_{k}", f"minus_shuffled_{k}": f"minus_shuffled_{k}", f"minus_bar_{k}": f"minus_Bc_{k}"})
        pairs.update({"iu_p": "iu_p", "power_2.80se": "power_2.80se"})
    largest = 0.0
    for r in rows:
        hit = family[(family["embedding"] == r["embedding"]) & (family["endpoint"] == r["endpoint"]) & (family["setting"] == r["setting"])]
        if len(hit) != 1 or int(hit["n_test_genes"].iloc[0]) != r["n_test_genes"]:
            raise Stop("the family has no single row with %d test genes for %s, %s, %s" % (r["n_test_genes"], r["embedding"], r["endpoint"], r["setting"]))
        for ours, theirs in pairs.items():
            a, b = float(r[ours]), float(hit[theirs].iloc[0])
            if np.isnan(a) != np.isnan(b):
                raise Stop("%s, %s, %s: %s is missing on one side only" % (r["embedding"], r["endpoint"], r["setting"], ours))
            d = 0.0 if np.isnan(a) else abs(a - b)
            largest = max(largest, d)
            if d > TOLERANCE:
                raise Stop("%s, %s, %s: %s differs from family.csv by %r" % (r["embedding"], r["endpoint"], r["setting"], ours, d))
    return {"compared": True, "statistics": len(rows), "quantities": sorted(pairs), "largest_difference": largest, "tolerance": TOLERANCE}


def power_rows(rows: list[dict], family: pd.DataFrame | None) -> list[dict]:
    out = []
    for r in rows:
        se = max(r["minus_gaussian_se"], r["minus_shuffled_se"])
        entry = {"role": "family statistic, power", "embedding": r["embedding"], "width": r["width"], "endpoint": r["endpoint"], "setting": r["setting"],
                 "n_test_genes": r["n_test_genes"], "iu_estimate": r["iu_estimate"], "se_gaussian": r["minus_gaussian_se"],
                 "se_shuffled": r["minus_shuffled_se"], "se_larger": se, "power_2.80se": POWER_UNADJUSTED * se,
                 "holm_first_step_detectable": POWER_HOLM_FIRST * se, "replicate_index_sha256": r["replicate_index_sha256"]}
        if family is not None:
            hit = family[(family["embedding"] == r["embedding"]) & (family["endpoint"] == r["endpoint"]) & (family["setting"] == r["setting"])]
            entry["family_holm_first_step_detectable"] = float(hit["holm_first_step_detectable"].iloc[0]) if len(hit) == 1 else float("nan")
        out.append(entry)
    return out


def save_predictions(path: Path, results: list[tuple]) -> dict:
    """Out-of-fold predictions of every fit, with the index of what each block holds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, index = {}, []
    for i, (key, result) in enumerate(results):
        index.append([str(k) for k in key])
        arrays[f"r{i}_genes"] = np.array([str(g) for g in result["genes"]], dtype=str)
        arrays[f"r{i}_y"] = np.asarray(result["y"], dtype=np.float64)
        arrays[f"r{i}_folds"] = np.asarray(result["folds"], dtype=np.int64)
        for m, p in result["predictions"].items():
            arrays[f"r{i}_{m}"] = np.asarray(p, dtype=np.float64)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, index=np.array(json.dumps(index)), **arrays)
    os.replace(temporary, path)
    return file_record(path)


def sort_rows(rows: list[dict]) -> pd.DataFrame:
    items = {k: i for i, k in enumerate(list(ITEMS) + [PREFIX_ROWS])}
    settings = {k: i for i, k in enumerate(list(SETTINGS) + list(VCC_SETTINGS))}

    def key(r):
        return (items.get(r.get("item"), 99), EMBEDDINGS.index(r["embedding"]) if r.get("embedding") in EMBEDDINGS else 9,
                str(r.get("label", "")), settings.get(r.get("setting"), 99))
    return pd.DataFrame(sorted(rows, key=key))


def write_record(folder: Path, part: str, started: float, outputs: list[str], extra: dict) -> None:
    record = {"plan": PLAN, "smoke": SMOKE, "part": part, "replicates": B,
              "seeds": {"settings": SETTING_SEEDS, "VCC": VCC_SEED, "union folds": fits.UNION_FOLD_SEED, "within folds": e5lib.WITHIN_FOLD_SEED,
                        "prefix folds": {"within": e5lib.WITHIN_FOLD_SEED, "between": fits.UNION_FOLD_SEED}, "inner folds": e5fit.INNER_SEED,
                        "shuffle": e5fit.SHUFFLE_SEED, "forest head seeds": list(FOREST_SEEDS[:1] if SMOKE else FOREST_SEEDS)},
              "ridge_grid": list(e5fit.RIDGE_GRID), "forest": {**heads.FOREST, **({"n_estimators": SMOKE_TREES} if SMOKE else {})},
              "settings": list(active_settings()), "inputs": DATA.get("inputs", {}), "code": DATA.get("code", {}), **extra,
              "outputs": {n: file_record(folder / n) for n in outputs}, "seconds": round(time.time() - started, 1)}
    (folder / f"secondary_{part}.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")


def run_main() -> None:
    started = time.time()
    load_replogle()
    load_prefit50()
    family = load_family()
    folder = out_dir()
    if SLOTS:
        print("[e5 secondary main] inputs loaded, %.0f s" % (time.time() - started), flush=True)
        stage1_saved = slots_map(main_task_saved, main_args(), "stage one")
        stage1 = [o for o, _ in stage1_saved]
        found = {(o["key"][1], o["key"][2], o["key"][3]): o for o in stage1 if o["key"][0] == PRIMARY}
        primary = {k: found[k] for k in sorted(found, key=lambda k: (EMBEDDINGS.index(k[0]), k[1], list(SETTINGS).index(k[2])))}
        stage2_args = [(label, setting, {n: primary[(n, label, setting)]["result"] for n in EMBEDDINGS})
                       for label in ITEMS[PRIMARY]["labels"] for setting in active_settings()]
        stage2_saved = slots_map(contrast_task_saved, stage2_args, "stage two")
        stage2 = [o for o, _ in stage2_saved]
    else:
        context_ = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context_) as pool:
            stage1_saved = list(pool.map(main_task_saved, main_args()))
            stage1 = [o for o, _ in stage1_saved]
            found = {(o["key"][1], o["key"][2], o["key"][3]): o for o in stage1 if o["key"][0] == PRIMARY}
            primary = {k: found[k] for k in sorted(found, key=lambda k: (EMBEDDINGS.index(k[0]), k[1], list(SETTINGS).index(k[2])))}
            stage2_args = [(label, setting, {n: primary[(n, label, setting)]["result"] for n in EMBEDDINGS})
                           for label in ITEMS[PRIMARY]["labels"] for setting in active_settings()]
            stage2_saved = list(pool.map(contrast_task_saved, stage2_args))
            stage2 = [o for o, _ in stage2_saved]
    reused = {"stage one": int(sum(r for _, r in stage1_saved)), "stage two": int(sum(r for _, r in stage2_saved))}
    primary_rows = [o["row"] for o in primary.values()]
    check = check_against_family(primary_rows, family)
    secondary = [o["row"] for o in stage1 if ITEMS[o["key"][0]]["role"] == "secondary, unadjusted"] + [o["prefix_row"] for o in primary.values()]
    diagnostics = [o["row"] for o in stage1 if ITEMS[o["key"][0]]["role"] == "diagnostic"]
    frames = {s: fits.DATA["screens"][s]["frame"] for s in SCREENS}
    agreement = agreement_rows({s: SETTING_SEEDS[f"within {s}"] for s in SCREENS}, frames, fits.DATA["covered"], (("a", "b"), ("a", "b50")))
    folder.mkdir(parents=True, exist_ok=True)
    tables = {"secondary.csv": sort_rows(secondary), "diagnostics.csv": sort_rows(diagnostics),
              "folds.csv": pd.DataFrame([r for o in primary.values() for r in o["folds"]]),
              "between_embeddings.csv": pd.DataFrame([r for o in stage2 for r in o["between"]]),
              "family_baseline.csv": pd.DataFrame([r for o in stage2 for r in o["baseline"]]),
              "power.csv": sort_rows(power_rows(primary_rows, family)), "agreement.csv": pd.DataFrame(agreement)}
    for name, table in tables.items():
        table.to_csv(folder / name, index=False)
    predictions = save_predictions(data_dir() / "predictions_main.npz", [(o["key"], o["result"]) for o in stage1])
    svd = int(sum(o["row"]["svd_fallbacks"] for o in stage1))
    write_record(folder, "main", started, list(tables), {
        "checks": {"family recomputed": check, "inputs": "every input checked against its record at load (Section 11, check 1)"},
        "items": ITEMS, "svd_fallbacks_total": svd, "covered_genes": DATA["covered_counts"], "prefixes": DATA["prefixes"],
        "label_sources": DATA["label_sources"],
        "statistics": {"secondary": len(secondary), "diagnostics": len(diagnostics), "family recomputed": len(primary_rows)},
        "predictions": predictions,
        "saved_tasks": {"folder": config.portable(data_dir() / "main_tasks"), "tasks": {"stage one": len(stage1), "stage two": len(stage2)},
                        "reused": reused, "rule": "each task saved as it ends and reused when its specification is unchanged (gate G4, item 4)"},
        **({"runner": "one process per task (fit_slots.py)", "tasks_shed_and_queued_again": DATA.get("shed", {})} if SLOTS else {})})
    print("[e5 secondary main] done in %.0f s; %d secondary rows, %d diagnostics, family check %s, SVD fallbacks %d, reused %s" % (
        time.time() - started, len(secondary), len(diagnostics), json.dumps({k: check.get(k) for k in ("compared", "largest_difference")}), svd,
        json.dumps(reused)), flush=True)


# --------------------------------------------------------------------------- the size association of (b) (Section 8)


def family_b_column(gates: dict, screen: str) -> str:
    """The (b) column the family uses in the within-screen setting (as make_e5_fits.load reads gates.json)."""
    level = gates["versions_used_by_the_family"][f"within {screen}"]["version_used_by_the_family"].split()[0]
    return f"b{level}_gem" if gates["gates"][screen][level]["primary_version"].startswith("gem") else f"b{level}"


def size_frame(genes_b: pd.DataFrame, g1: pd.DataFrame, fixed: str, other: str) -> pd.DataFrame:
    """Per gene: B0 (G1, median over the eligible constructs) and the three statistics of (b), and the other version."""
    gb = genes_b.set_index("target_ensg")
    gb.index = gb.index.astype(str)
    out = pd.DataFrame({"fixed": gb[fixed], "other_version": gb[other], "all_cells": gb["all_cells"], "n_scaled": gb["n_scaled"]}).astype(np.float64)
    eligible = g1[g1["n_pool"] >= 27]
    out["B0"] = eligible.groupby(eligible["target_ensg"].astype(str))["B0"].median().reindex(out.index).astype(np.float64)
    return out


def size_rows(frames: dict, seeds: dict, columns: dict) -> list[dict]:
    """Spearman with B0 of each statistic, one gene bootstrap per screen; the statement of Section 8 where its rule holds."""
    rows = []
    reported = (("fixed cells (the family's version)", "fixed"), ("all cells", "all_cells"), ("n-scaled, all cells", "n_scaled"))
    for screen, f in frames.items():
        groups = [reported, (("fixed cells (the other version, beside)", "other_version"),)]  # one bootstrap per group
        screen_rows = []
        for statistics in groups:
            names = [c for _, c in statistics]
            ok = f[["B0"] + names].notna().all(axis=1)
            genes = np.array([str(g) for g in f.index[ok]])
            b0 = f.loc[ok, "B0"].to_numpy(dtype=np.float64)
            values = f.loc[ok, names].to_numpy(dtype=np.float64)
            point = resample.ClusterDraw(weights=np.ones((1, genes.size)), clusters=np.array([]), seed=-1, index_sha256="")
            draw = resample.cluster_draw(genes, B, seeds[screen])
            estimate = resample.weighted_spearman(values, b0, point)[0]
            draws = resample.weighted_spearman(values, b0, draw)
            for j, (label, column) in enumerate(statistics):
                lo, hi, missing = percentile_interval(draws[:, j])
                p = resample.p_value(draws[:, j])
                screen_rows.append({"role": "secondary, unadjusted", "screen": screen, "statistic": label,
                                    "column": columns[screen].get(column, column), "in_the_rule": column in ("fixed", "n_scaled"),
                                    "n_genes": int(genes.size),
                                    "spearman_with_B0": float(estimate[j]), "lo": lo, "hi": hi, "p": p,
                                    "se": float(np.nanstd(draws[:, j], ddof=1)), "replicates_not_finite": missing,
                                    "resolved_above_zero": bool(p < 0.05 and estimate[j] > 0), "seed": seeds[screen],
                                    "replicate_index_sha256": draw.index_sha256})
        resolved = {r["statistic"]: r["resolved_above_zero"] for r in screen_rows}
        holds = resolved["n-scaled, all cells"] and not resolved["fixed cells (the family's version)"]
        for r in screen_rows:  # the statement belongs to the screen; it is written on its three reported statistics
            r["statement"] = SIZE_STATEMENT if (holds and r["column"] != columns[screen]["other_version"]) else ""
        rows += screen_rows
    return rows


def run_size() -> None:
    started = time.time()
    b_record = checked_json(ENDPOINT_B / "endpoint_b.json")
    gates = checked_json(ENDPOINT_B / "gates.json", b_record["outputs"]["gates.json"]["sha256"])
    g1 = e1cells.load_tables("G1")
    frames, columns = {}, {}
    for s in SCREENS:
        genes_b = fits.checked_csv(ENDPOINT_B / f"genes_{s}.csv", b_record["outputs"][f"genes_{s}.csv"]["sha256"])
        other = "b50_gem" if gates["gates"][s]["50"]["primary_version"].startswith("gem") else "b50"
        columns[s] = {"fixed": family_b_column(gates, s), "other_version": other}
        frames[s] = size_frame(genes_b, g1[s], columns[s]["fixed"], other)
    rows = size_rows(frames, {s: SETTING_SEEDS[f"within {s}"] for s in SCREENS}, columns)
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "size_association.csv", index=False)
    DATA.setdefault("code", code_digests())
    inputs = {"endpoint_b": file_record(ENDPOINT_B / "endpoint_b.json"), "gates_b": file_record(ENDPOINT_B / "gates.json"),
              "G1_manifest": file_record(REPO / "bioinf2027" / "endpoints" / "G1" / "manifest.json"),
              **{f"genes_{s}.csv": file_record(ENDPOINT_B / f"genes_{s}.csv") for s in SCREENS}}
    write_record(folder, "size", started, ["size_association.csv"], {
        "inputs": inputs, "columns": columns, "statement": SIZE_STATEMENT,
        "rule": "Section 8: the statement where the Spearman with B0 of the n-scaled statistic on all cells is resolved above zero (bootstrap "
                "p below 0.05, estimate above zero, unadjusted) and that of the fixed-cell energy distance is not; otherwise no statement",
        "statements": {s: next(r["statement"] for r in rows if r["screen"] == s) for s in SCREENS}})
    print("[e5 secondary size] done in %.0f s; statements: %s" % (time.time() - started, json.dumps({s: bool(next(
        r["statement"] for r in rows if r["screen"] == s)) for s in SCREENS})), flush=True)


# --------------------------------------------------------------------------- the 50-cell ceiling and positive control (Section 16.1)


def shared_with_b50() -> set:
    """The genes both screens hold, each with the 50-cell version in both (the ceiling's input is the other screen's label)."""
    frames = [fits.DATA["screens"][s]["frame"] for s in SCREENS]
    return {g for g in fits.shared_genes() if all(np.isfinite(f.loc[g, "b50"]) for f in frames)}


def ceiling_row(setting: str, label: str, genes: set) -> dict:
    """make_e5_fits.ceiling_task with the label and the genes as arguments (it reads the shared genes itself)."""
    result = fits.run_setting(setting, label, {"Bc": None, "[Bc; other screen's label]": fits.other_label(label)}, genes)
    stats = fits.spearman_draws(result, SETTING_SEEDS[setting])
    d = fits.difference(stats, "[Bc; other screen's label]", "Bc")
    return {"setting": setting, "endpoint": label, "n_test_genes": len(result["genes"]),
            "spearman_Bc": stats["point"]["Bc"], "spearman_with_other_label": stats["point"]["[Bc; other screen's label]"],
            **{f"difference_{k}": v for k, v in d.items() if k != "draws"},
            "ridge_alpha_at_grid_edge": int(sum(r.get("at_grid_edge", False) for r in result["records"]["[Bc; other screen's label]"]))}


def ceiling50_task(setting: str) -> dict:
    e5fit.SVD_FALLBACKS[0] = 0
    row = ceiling_row(setting, "b50", DATA["shared50"])
    return {**row, "svd_fallbacks": int(e5fit.SVD_FALLBACKS[0])}


def pc50_task(args) -> dict:
    """make_e5_fits.positive_control_task, unchanged, with the label b50 (its rows: the shared genes holding b50)."""
    e5fit.SVD_FALLBACKS[0] = 0
    row = fits.positive_control_task(args)
    row.setdefault("version", "planned")
    return {**row, "svd_fallbacks": int(e5fit.SVD_FALLBACKS[0])}


def prefit50_args() -> tuple[list, list]:
    widths = sorted(set(fits.WIDTHS.values()))
    planned = [(s, "b50", w) for s in active_settings() for w in widths]
    revised = [(s, "b50", w, True) for s in active_settings() for w in widths if w > fits.REVISED_COMPONENTS]
    return list(active_settings()), planned + revised


def run_prefit50() -> None:
    started = time.time()
    load_replogle()
    DATA["shared50"] = shared_with_b50()
    ceiling_args, pc_args = prefit50_args()
    context_ = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context_) as pool:
        ceiling = list(pool.map(ceiling50_task, ceiling_args))
        pc = list(pool.map(pc50_task, pc_args))
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ceiling).to_csv(folder / "ceiling_50.csv", index=False)
    pd.DataFrame(pc).to_csv(folder / "positive_control_50.csv", index=False)
    genes = {"shared": len(fits.shared_genes()), "shared with b50 in both screens (the ceiling)": len(DATA["shared50"]),
             **{f"shared with b50 in {s} (the positive control's rows there)": int(fits.DATA["screens"][s]["frame"].loc[sorted(fits.shared_genes()), "b50"].notna().sum())
                for s in SCREENS}}
    write_record(folder, "prefit50", started, ["ceiling_50.csv", "positive_control_50.csv"], {
        "genes": genes, "components": fits.REVISED_COMPONENTS, "padding_seed_text": "E5 | PC padding | <width> | <Ensembl ID>",
        "label_sources": {s: DATA["label_sources"][s]["b50"] for s in SCREENS},
        "svd_fallbacks_total": int(sum(r["svd_fallbacks"] for r in ceiling + pc)),
        "rule": "Sections 6, 16.1, 18 and 20: the 27-cell prefit's ceiling and positive control with the label b50; resolved: "
                "intersection-union p below 0.05 with both differences above zero"})
    print("[e5 secondary prefit50] done in %.0f s; positive control resolved: %d of %d" % (
        time.time() - started, sum(bool(r["resolved"]) for r in pc), len(pc)), flush=True)


def load_prefit50() -> None:
    """The 50-cell control the main part's 50-cell rows read; the main part stops without it (smoke reads the smoke's)."""
    folder = out_dir()
    path = folder / "secondary_prefit50.json"
    if not path.is_file():
        raise Stop("the 50-cell ceiling and positive control have not been computed (--part prefit50): %s is missing" % config.portable(path))
    record = json.loads(path.read_text())
    if bool(record.get("smoke")) != SMOKE:
        raise Stop("secondary_prefit50.json comes from a %s run" % ("smoke" if record.get("smoke") else "real"))
    table = fits.checked_csv(folder / "positive_control_50.csv", record["outputs"]["positive_control_50.csv"]["sha256"])
    _, wanted = prefit50_args()
    for a in wanted:
        version = fits.REVISED if len(a) > 3 else "planned"
        hit = table[(table["setting"] == a[0]) & (table["endpoint"] == "b50") & (table["nominal_width"] == a[2]) & (table["version"] == version)]
        if len(hit) != 1:
            raise Stop("positive_control_50.csv has no single row for %s, width %d, %s" % (a[0], a[2], version))
    DATA["pc50"] = table
    DATA.setdefault("inputs", {})["prefit50"] = file_record(path)


# --------------------------------------------------------------------------- the forest part


def forest_spec(name: str, label: str, setting: str, seed: int, R: dict) -> str:
    genes = hashlib.sha256(("|".join(R["src"]) + "#" + "|".join(R["tgt"])).encode()).hexdigest()
    embedding = DATA["inputs"]["embeddings"][name]["sha256"]
    return json.dumps({"key": [name, label, setting, seed], "forest": forest_model(seed).get_params()["n_estimators"], "heads": heads.FOREST,
                       "code": DATA["code"], "embedding": embedding, "genes": genes,
                       "inputs": {k: v["sha256"] for k, v in fits.DATA["record"]["inputs"].items()}}, sort_keys=True)


def forest_task(args) -> dict:
    """One head seed of one family statistic under the forest; a saved fit with the same specification is reused."""
    name, label, setting, seed = args
    genes = fits.setting_genes(setting)
    R = setting_rows(setting, label, bar_columns("Bc", label), genes)
    spec = forest_spec(name, label, setting, seed, R)
    path = data_dir() / "forest_fits" / (hashlib.sha256(json.dumps(args).encode()).hexdigest()[:24] + ".npz")
    if path.is_file():
        with np.load(path, allow_pickle=False) as store:
            if str(store["spec"]) == spec:
                return {"key": args, "reused": True, "seconds": float(store["seconds"]), "path": path,
                        "result": {"genes": [str(g) for g in store["genes"]], "y": store["y"], "folds": store["folds"],
                                   "fold_ids": [int(k) for k in store["fold_ids"]], "train_sizes": [int(n) for n in store["train_sizes"]],
                                   "predictions": {m: store[f"p_{m}"] for m in ("bar", "e", "C", "S")}, "mean_predictor": store["mean_predictor"]}}
    models, _ = models_for(name, genes)
    started = time.time()
    result = compact(fit_folds(R, models, head="forest", seed=seed))
    seconds = time.time() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, spec=np.array(spec), seconds=np.array(seconds), genes=np.array(result["genes"], dtype=str), y=result["y"],
             folds=result["folds"], fold_ids=np.array(result["fold_ids"]), train_sizes=np.array(result["train_sizes"]),
             mean_predictor=result["mean_predictor"], **{f"p_{m}": v for m, v in result["predictions"].items()})
    os.replace(temporary, path)
    return {"key": args, "reused": False, "seconds": seconds, "path": path, "result": result}


def forest_stats_task(args) -> dict:
    name, label, setting, fitted = args
    first = fitted[0]["result"]
    for f in fitted[1:]:
        if f["result"]["genes"] != first["genes"] or not np.array_equal(f["result"]["y"], first["y"]):
            raise Stop("%s, %s, %s: the head seeds differ in their test genes" % (name, label, setting))
    stats = seed_averaged([fits.spearman_draws(f["result"], SETTING_SEEDS[setting]) for f in fitted])
    width = int(next(iter(fits.DATA["embeddings"][name].values())).size)
    base = base_of("forest", name, width, label, setting, first, "forest", "gene folds", "genes")
    base.update({"item": "forest", "role": "secondary, unadjusted", "bar": "Bc (rank columns)"})
    return {**contrast_row(stats, base), **context(label, setting, width), "head_seeds": ";".join(str(f["key"][3]) for f in fitted),
            "forest_seconds": round(sum(f["seconds"] for f in fitted), 1), "fits_reused": int(sum(f["reused"] for f in fitted)), "svd_fallbacks": 0}


def run_forest() -> None:
    started = time.time()
    load_replogle()
    folder = out_dir()
    seeds = FOREST_SEEDS[:1] if SMOKE else FOREST_SEEDS
    keys = [(n, label, s) for n in EMBEDDINGS for label in ("a", "b") for s in active_settings()]
    args = sorted([(n, label, s, seed) for n, label, s in keys for seed in seeds], key=lambda a: -fits.WIDTHS[a[0]])
    context_ = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context_) as pool:
        fitted = list(pool.map(forest_task, args))
        by_key = {k: sorted([f for f in fitted if tuple(f["key"][:3]) == k], key=lambda f: f["key"][3]) for k in keys}
        rows = list(pool.map(forest_stats_task, [(*k, by_key[k]) for k in keys]))
    folder.mkdir(parents=True, exist_ok=True)
    sort_rows(rows).to_csv(folder / "forest.csv", index=False)
    write_record(folder, "forest", started, ["forest.csv"], {
        "fits": len(fitted), "fits_reused": int(sum(f["reused"] for f in fitted)), "forest_seconds_total": round(sum(f["seconds"] for f in fitted), 1),
        "forest_fits": folder_record([f["path"] for f in fitted], data_dir() / "forest_fits"), "covered_genes": DATA["covered_counts"],
        "rule": "Spearman per head seed, averaged over the seeds inside each replicate (E1_design.md, Section 10); Bc as its rank columns"})
    print("[e5 secondary forest] done in %.0f s; %d statistics from %d fits (%d reused)" % (
        time.time() - started, len(rows), len(fitted), sum(f["reused"] for f in fitted)), flush=True)


# --------------------------------------------------------------------------- the VCC part (descriptive)


def vcc_frame(vcc_genes: pd.DataFrame, g3: pd.DataFrame, g4: pd.DataFrame, ensembl_of: dict) -> pd.DataFrame:
    """One row per VCC target gene: Bc and (b) from vcc_genes.csv, (a) from G3 and G4, indexed by Ensembl identifier."""
    t = vcc_genes[["split", "target_gene", "B1", "detection_rate", "control_sd", "b27", "b10", "b50"]].copy()
    t["split"], t["target_gene"] = t["split"].astype(str), t["target_gene"].astype(str)
    keys = set(zip(t["split"], t["target_gene"]))
    for name, table in (("a27", g3), ("a10", g4)):
        labels = {(str(s), str(g)): float(y) for s, g, y in zip(table["split"], table["target_gene"], table["y"])}
        if not set(labels) <= keys:
            raise Stop("VCC: %s holds genes that vcc_genes.csv does not" % name)
        t[name] = [labels.get(k, float("nan")) for k in zip(t["split"], t["target_gene"])]
    t["target_ensg"] = t["target_gene"].map(ensembl_of)
    t = t[t["target_ensg"].notna()].copy()
    if t["target_ensg"].duplicated().any():
        raise Stop("VCC: two target genes share an Ensembl identifier")
    t = t.set_index("target_ensg")
    t.index = t.index.astype(str)
    for c in ("B1", "detection_rate", "control_sd"):
        t[f"rank_{c}"] = e5lib.ranks01(t[c].to_numpy(dtype=np.float64))
    return t


def load_vcc() -> None:
    load_replogle()
    b_record = checked_json(ENDPOINT_B / "endpoint_b.json")
    vcc_record = checked_json(ENDPOINT_B / "endpoint_b_vcc.json")
    genes = fits.checked_csv(ENDPOINT_B / "vcc_genes.csv", vcc_record["outputs"]["vcc_genes.csv"]["sha256"])
    g3, g4 = e1cells.load_tables("G3"), e1cells.load_tables("G4")
    annotation = e1inputs.vcc_annotation_file()
    digest = sha256(annotation)
    if digest != b_record["single_cell_inputs"]["train"]["sha256"]:
        raise Stop("the VCC training file does not match its record")
    ensembl_of, unmatched = e1inputs.vcc_target_ensembl(genes["target_gene"].astype(str))
    frame = vcc_frame(genes, g3["vcc"], g4["vcc"], ensembl_of)
    for s in SCREENS:  # (a) at 10 cells on the Replogle genes: G2's label, gene median over E5's eligible constructs
        g2 = g4[s]
        eligible = g2[g2["n_pool"] >= 27]
        f = fits.DATA["screens"][s]["frame"]
        f["a10"] = eligible.groupby(eligible["target_ensg"].astype(str))["y"].median().reindex(f.index).astype(np.float64)
    covered = set(frame.index[frame[VCC_BC].notna().all(axis=1)])
    shares = {}
    for name, vectors in fits.DATA["embeddings"].items():
        shares[name] = len(set(frame.index) & set(vectors)) / len(frame)
        if shares[name] < 0.8:
            raise Stop("%s covers %.3f of the VCC genes" % (name, shares[name]))
        covered &= set(vectors)
    DATA["vcc_frame"] = frame
    DATA["vcc_covered"] = covered
    DATA["vcc_excluded"] = set(frame.index[frame["split"].isin(["train", "val"])])
    DATA["vcc_record"] = {"genes": int(len(frame)), "covered": len(covered), "coverage": shares, "symbols_without_an_identifier": list(unmatched),
                          "splits": {k: int(v) for k, v in frame["split"].value_counts().sort_index().items()}}
    DATA["inputs"].update({"endpoint_b_vcc": file_record(ENDPOINT_B / "endpoint_b_vcc.json"), "vcc_genes.csv": file_record(ENDPOINT_B / "vcc_genes.csv"),
                           "G3_manifest": file_record(REPO / "bioinf2027" / "endpoints" / "G3" / "manifest.json"),
                           "G4_manifest": file_record(REPO / "bioinf2027" / "endpoints" / "G4" / "manifest.json"),
                           "vcc_annotation": {"path": config.portable(annotation), "bytes": annotation.stat().st_size, "sha256": digest}})


def vcc_rows(setting: str, src_label: str, tgt_label: str) -> dict:
    vf = DATA["vcc_frame"]
    covered = DATA["vcc_covered"]
    src = usable(vf[vf["split"] == "train"], src_label, VCC_BC, covered)
    screen = VCC_SETTINGS[setting]
    if screen is None:
        tf, target = vf, "VCC"
        tgt = usable(vf[vf["split"] == "test"], tgt_label, VCC_BC, covered)
    else:
        tf, target = fits.DATA["screens"][screen]["frame"], screen
        tgt = [g for g in usable(tf, tgt_label, VCC_BC, fits.DATA["covered"][screen]) if g not in DATA["vcc_excluded"]]
    if set(src) & set(tgt):
        raise Stop("%s: a gene is both a training and a test gene" % setting)
    return {"source": "VCC", "target": target, "src_frame": vf, "tgt_frame": tf, "src_label": src_label, "tgt_label": tgt_label,
            "bc": list(VCC_BC), "src": src, "tgt": tgt, "folds": {**{g: 1 for g in src}, **{g: 0 for g in tgt}}, "ks": (0,)}


def vcc_task(args) -> dict:
    name, setting, src_label, tgt_label, title = args
    e5fit.SVD_FALLBACKS[0] = 0
    started = time.time()
    R = vcc_rows(setting, src_label, tgt_label)
    models, width = models_for(name, set(R["src"]) | set(R["tgt"]))
    result = fit_folds(R, models)
    stats = fits.spearman_draws(result, VCC_SEED)
    base = {"item": "VCC", "role": "secondary, descriptive", "embedding": name, "width": width, "endpoint": endpoint_of(src_label), "label": title,
            "label_source": f"{src_label} (VCC training) to {tgt_label}", "setting": setting, "bar": "Bc (B1, detection rate, control SD)",
            "head": "ridge", "folds": "VCC training genes to test genes", "clusters": "genes", "n_training_genes": str(len(R["src"])),
            "n_test_genes": len(result["genes"])}
    edges = int(sum(r.get("at_grid_edge", False) for m in ("e", "C", "S") for r in result["records"][m]))
    row = {**contrast_row(stats, base), "ridge_alpha_at_grid_edge": edges, "svd_fallbacks": int(e5fit.SVD_FALLBACKS[0]),
           "seconds": round(time.time() - started, 1)}
    return {"key": args, "row": row, "result": compact(result)}


def vcc_args() -> list:
    args = []
    for name in EMBEDDINGS:
        for setting, screen in VCC_SETTINGS.items():
            for src_label, tgt_label, title in VCC_LABELS[VCC_TEST if screen is None else "Replogle"]:
                args.append((name, setting, src_label, tgt_label, title))
    return sorted(args, key=lambda a: -fits.WIDTHS[a[0]])


def run_vcc() -> None:
    started = time.time()
    load_vcc()
    folder = out_dir()
    context_ = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context_) as pool:
        fitted = list(pool.map(vcc_task, vcc_args()))
    vf = DATA["vcc_frame"]
    frames = {f"VCC {s}": vf[vf["split"] == s] for s in ("train", "val", "test")}
    agreement = agreement_rows({k: VCC_SEED for k in frames}, frames, {k: DATA["vcc_covered"] for k in frames}, (("a27", "b27"), ("a10", "b10")))
    folder.mkdir(parents=True, exist_ok=True)
    sort_rows([f["row"] for f in fitted]).to_csv(folder / "vcc.csv", index=False)
    pd.DataFrame(agreement).to_csv(folder / "vcc_agreement.csv", index=False)
    predictions = save_predictions(data_dir() / "predictions_vcc.npz", [(f["key"], f["result"]) for f in fitted])
    write_record(folder, "vcc", started, ["vcc.csv", "vcc_agreement.csv"], {
        "vcc": DATA["vcc_record"], "svd_fallbacks_total": int(sum(f["row"]["svd_fallbacks"] for f in fitted)), "predictions": predictions,
        "rule": "descriptive; Bc of three columns on both endpoints (E1a's VCC ceiling is per row of G1 and G2, not per gene of G3 and G4)"})
    print("[e5 secondary vcc] done in %.0f s; %d statistics" % (time.time() - started, len(fitted)), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
        B = 200
        fits.B = 200
    if "--workers" in arguments:  # changes no value
        i = arguments.index("--workers")
        WORKERS = int(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if "--slots" in arguments:  # changes no value
        SLOTS = True
        arguments.remove("--slots")
    if "--workers-file" in arguments:  # changes no value
        i = arguments.index("--workers-file")
        WORKERS_FILE = config.storage_root() / arguments[i + 1]
        arguments = arguments[:i] + arguments[i + 2:]
    parts = []
    while "--part" in arguments:
        i = arguments.index("--part")
        parts.append(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if arguments or not set(parts) <= set(PARTS) or len(set(parts)) != len(parts):
        raise SystemExit("usage: make_e5_secondary.py [--smoke] [--workers N] [--slots] [--workers-file PATH-UNDER-DATA-ROOT] "
                         "[--part vcc|size|prefit50|main|forest ...] (parts run in the order given; --slots and --workers-file with --part main only)")
    if (SLOTS or WORKERS_FILE is not None) and parts != ["main"]:
        raise SystemExit("--slots and --workers-file apply to the main part only (--part main)")
    for part in parts or PARTS:
        {"main": run_main, "vcc": run_vcc, "forest": run_forest, "size": run_size, "prefit50": run_prefit50}[part]()
