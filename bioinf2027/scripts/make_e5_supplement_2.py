"""E5 supplement 2: five analyses the review of E5's supplement finds missing, post hoc (bioinf2027/plans/E5_supplement_2.md;
the answer of 2026-11-17 to the stop of 06:50, item 7, bioinf2027/GATE.md). E5's harness and the first supplement
(make_e5_supplement.py, make_e5_secondary.py, make_e5_fits.py) are reused unchanged.

Parts (--part, in the order given; default all):
- fits (plan Sections 1 and 4): on analysis 2's rows (version a: genes holding F_mean; b: F_mean and F_K562), per
  vector and setting, the comparators Bc, [Bc; FN], [Bc; F] and [Bc; FN; F], each with e, C and S, on the same
  replicates; the smaller of [X; e] - [X; C] and [X; e] - [X; S] and X - Bc for each; rho([Bc; F]) - rho([Bc; e]);
  rho([Bc; FN; F]) - rho([Bc; F]) and - rho([Bc; FN]); every value also with symbol prefixes as bootstrap clusters.
  Check: the [Bc; F] statistics equal analysis 2's depmap.csv (1e-9; point estimates only in a smoke run).
- route (Section 2): E5's diagnostic "B1 from e" with F_mean as the label ("F_mean from e").
- b0 (Section 3): comparator [Bc; F_mean; B0] with e, C and S on version a's rows (diagnostic), and per screen the
  Spearman of B0 with F_mean over the covered genes holding F_mean (gene bootstrap, seeds 20271121 and 20271122).
- hgnc (Section 5): the scGPT, GenePT and DepMap misses against the HGNC complete set (a count; no refit).
--smoke: 200 replicates, two settings, outputs under $IGA_STORAGE_ROOT/bioinf2027/scratch/E5_supplement_2_smoke/.
Every value is post hoc and unadjusted and not part of E5's family; DepMap values are 24Q4, not the current release.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, resample  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.stats import percentile_interval  # noqa: E402

import e5fit  # noqa: E402
import e5lib  # noqa: E402
import fit_slots  # noqa: E402
import make_e5_supplement as sup  # noqa: E402
import safe_pickle  # noqa: E402

sec, fits = sup.sec, sup.fits
REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E5_supplement_2.md"
STATUS = "post hoc, unadjusted, not part of E5's family (the stop of 2026-11-17 06:50, item 7; %s)" % PLAN
OUT = REPO / "bioinf2027" / "results" / "E5" / "supplement_2"
FIRST = REPO / "bioinf2027" / "results" / "E5" / "supplement"
PARTS = ("fits", "route", "b0", "hgnc")
VERSIONS = {"a": ("F_mean",), "b": ("F_mean", "F_K562")}
COMPARATORS = ("Bc", "FN", "F", "J")  # Bc; [Bc; FN]; [Bc; F]; [Bc; FN; F]
B0_SEEDS = {"RPE1": 20271121, "K562-essential": 20271122}
ROUTE = "F_mean from e"
B0_ITEM = "[Bc; F_mean; B0] as the comparator"
TOLERANCE = 1e-9
WORKERS = 4
WORKERS_FILE: Path | None = None
SMOKE = False
DATA: dict = {}

READ_JOINT = ("beside the target's control-cell profile, the symbol-prefix family mean and the fitness score together, the vector ranks unseen "
              "target genes by the downstream endpoint above both a random vector of its width and the same vectors assigned to other genes")
BELOW_JOINT = "beside family and fitness together, the vector ranks unseen target genes below both controls"
NOT_JOINT = "not shown beside family and fitness together"
READ_SCORE = {"above zero": "on the same rows, the profile with the fitness score ranks unseen target genes better than the profile with the vector",
              "below zero": "on the same rows, the profile with the vector ranks unseen target genes better than the profile with the fitness score",
              "contains zero": "not shown to differ on the same rows"}


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E5_supplement_2_smoke" if SMOKE else OUT


def tasks_dir() -> Path:
    return (config.storage_root() / "bioinf2027" / "scratch" / "E5_supplement_2_smoke" / "tasks" if SMOKE
            else config.storage_root() / "bioinf2027" / "E5" / "supplement_2" / "tasks")


def my_digest() -> str:
    return sha256(Path(__file__).resolve())


def state(lo: float, hi: float) -> str:
    return sup.state(lo, hi)


def comparator_name(x: str, version: str) -> str:
    f = "; ".join(VERSIONS[version])
    return {"Bc": "Bc", "FN": "[Bc; FN]", "F": "[Bc; %s]" % f, "J": "[Bc; FN; %s]" % f}[x]


# --------------------------------------------------------------------------- Sections 1 and 4


def view(stats: dict, x: str) -> dict:
    """The statistics of one comparator's four models under the names contrast_row expects."""
    names = {"bar": f"{x}:bar", "e": f"{x}:e", "C": f"{x}:C", "S": f"{x}:S"}
    return {"names": list(names), "point": {k: stats["point"][v] for k, v in names.items()}, "draws": {k: stats["draws"][v] for k, v in names.items()},
            "index_sha256": stats["index_sha256"]}


def smaller_below(row: dict) -> bool:
    return bool(row["minus_gaussian_hi"] < 0 and row["minus_shuffled_hi"] < 0)


def same_rows_task(args) -> dict:
    """Section 1 for one version, vector and setting: four comparator fits on the same rows, gene and prefix draws."""
    _, version, name, setting = args
    e5fit.SVD_FALLBACKS[0] = 0
    started = time.time()
    genes = fits.setting_genes(setting)
    bc = sec.bar_columns("Bc", "b")
    R = sec.setting_rows(setting, "b", bc + [f"rank_{c}" for c in VERSIONS[version]], genes)
    models, width = sec.models_for(name, genes)
    results = {"Bc": sec.fit_folds({**R, "bc": bc}, models), "F": sec.fit_folds(R, models),
               "FN": sup.fit_folds_fn({**R, "bc": bc}, models), "J": sup.fit_folds_fn(R, models)}
    order = results["Bc"]["genes"]
    if any(r["genes"] != order for r in results.values()):
        raise Stop("%s, %s, %s: the comparators were scored on other genes" % (version, name, setting))
    predictions = {f"{x}:{m}": results[x]["predictions"][m] for x in COMPARATORS for m in ("bar", "e", "C", "S")}
    combined = {"genes": order, "y": results["Bc"]["y"], "predictions": predictions}
    seed = sec.SETTING_SEEDS[setting]
    by_gene = fits.spearman_draws(combined, seed)
    clusters = [sec.DATA["prefix_of"][g] for g in order]
    by_prefix = fits.spearman_draws({**combined, "genes": clusters}, seed)
    rows, comparisons = [], []
    for x in COMPARATORS:
        base = sec.base_of("same rows", name, width, "b", setting, results[x], "ridge", "gene folds", "genes")
        base.update({"item": "supplement 2, Section 1: %s on analysis 2%s's rows" % (comparator_name(x, version), version), "role": STATUS,
                     "bar": comparator_name(x, version), "version": version, "comparator": x, "depmap_release": sup.NOT_CURRENT})
        row = {**sec.contrast_row(view(by_gene, x), base), **sec.context("b", setting, width)}
        pref = sec.contrast_row(view(by_prefix, x), {})
        row.update({f"prefix_{k}": pref[k] for k in ("minus_gaussian_estimate", "minus_gaussian_lo", "minus_gaussian_hi", "minus_shuffled_lo",
                                                     "minus_shuffled_hi", "iu_estimate", "iu_p", "both_resolved_above_zero", "replicate_index_sha256")})
        row["n_prefix_clusters"] = len(set(clusters))
        if x != "Bc":
            for label, stats in (("", by_gene), ("prefix_", by_prefix)):
                d = fits.difference(stats, f"{x}:bar", "Bc:bar")
                row.update({f"{label}bar_minus_Bc_{k}": v for k, v in d.items() if k != "draws"})
        row["reading"] = ""
        if x == "J":
            row["reading"] = READ_JOINT if row["both_resolved_above_zero"] else (BELOW_JOINT if smaller_below(row) else NOT_JOINT)
        if row["both_resolved_above_zero"]:
            row["reading_prefix_clusters"] = "holds with prefix clusters" if row["prefix_both_resolved_above_zero"] else "not shown with prefix clusters"
        else:
            row["reading_prefix_clusters"] = ""
        rows.append(row)
    for quantity, a, b, kind in (("rho([Bc; F]) - rho([Bc; e])", "F:bar", "Bc:e", "(2)"),
                                 ("rho([Bc; FN; F]) - rho([Bc; F])", "J:bar", "F:bar", "descriptive"),
                                 ("rho([Bc; FN; F]) - rho([Bc; FN])", "J:bar", "FN:bar", "descriptive")):
        g, p = fits.difference(by_gene, a, b), fits.difference(by_prefix, a, b)
        entry = {"item": "supplement 2, Section 1", "role": STATUS, "version": version, "embedding": name, "width": width, "setting": setting,
                 "n_test_genes": len(order), "quantity": quantity.replace("F]", "%s]" % "; ".join(VERSIONS[version])), "analysis": kind,
                 **{k: v for k, v in g.items() if k != "draws"}, "state": state(g["lo"], g["hi"]),
                 **{f"prefix_{k}": v for k, v in p.items() if k != "draws"}, "prefix_state": state(p["lo"], p["hi"]),
                 "replicate_index_sha256": by_gene["index_sha256"], "prefix_replicate_index_sha256": by_prefix["index_sha256"],
                 "depmap_release": sup.NOT_CURRENT}
        entry["reading"] = READ_SCORE[entry["state"]] if kind == "(2)" and entry["state"] in READ_SCORE else ""
        comparisons.append(entry)
    svd = int(e5fit.SVD_FALLBACKS[0])
    for row in rows:
        row["svd_fallbacks"] = svd
        row["seconds"] = round(time.time() - started, 1)
    return {"key": list(args), "rows": rows, "comparisons": comparisons}


def check_against_depmap(rows: list[dict]) -> dict:
    """The [Bc; F] comparator equals analysis 2 (depmap.csv): estimates, intervals and p (point estimates only in a smoke run)."""
    saved = pd.read_csv(FIRST / "depmap.csv", float_precision="round_trip")
    columns = ["spearman_bar", "spearman_e", "spearman_C", "spearman_S", "minus_gaussian_estimate", "minus_shuffled_estimate", "iu_estimate",
               "bar_minus_Bc_estimate"]
    if not SMOKE:
        columns += ["minus_gaussian_lo", "minus_gaussian_hi", "minus_gaussian_p", "minus_shuffled_lo", "minus_shuffled_hi", "minus_shuffled_p", "iu_p",
                    "bar_minus_Bc_lo", "bar_minus_Bc_hi", "bar_minus_Bc_p", "replicate_index_sha256"]
    largest, compared = 0.0, 0
    for row in (r for r in rows if r["comparator"] == "F"):
        hit = saved[saved["item"].str.startswith("analysis 2%s" % row["version"]) & (saved["embedding"] == row["embedding"]) & (saved["setting"] == row["setting"])]
        if len(hit) != 1:
            raise Stop("%d rows of depmap.csv for %s, %s, %s" % (len(hit), row["version"], row["embedding"], row["setting"]))
        hit = hit.iloc[0]
        if int(hit["n_test_genes"]) != int(row["n_test_genes"]):
            raise Stop("%s, %s, %s: %d genes against analysis 2's %d" % (row["version"], row["embedding"], row["setting"], row["n_test_genes"], hit["n_test_genes"]))
        for c in columns:
            if c == "replicate_index_sha256":
                if hit[c] != row[c]:
                    raise Stop("%s, %s, %s: other replicates than analysis 2" % (row["version"], row["embedding"], row["setting"]))
                continue
            difference = abs(float(hit[c]) - float(row[c]))
            largest = max(largest, difference)
            compared += 1
            if not difference <= TOLERANCE:
                raise Stop("%s, %s, %s: %s %r differs from depmap.csv %r" % (row["version"], row["embedding"], row["setting"], c, row[c], hit[c]))
    return {"compared": compared, "largest_difference": largest, "tolerance": TOLERANCE, "columns": columns}


# --------------------------------------------------------------------------- Sections 2 and 3


def route_task(args) -> dict:
    _, name, setting = args
    out = sec.main_task((ROUTE, name, "F_mean", setting))
    row = {**out["row"], "role": "diagnostic; " + STATUS, "reading": "diagnostic, no reading (printed beside Section 1)",
           "depmap_release": sup.NOT_CURRENT}
    return {"key": list(args), "rows": [row], "comparisons": []}


def b0_task(args) -> dict:
    _, name, setting = args
    e5fit.SVD_FALLBACKS[0] = 0
    started = time.time()
    genes = fits.setting_genes(setting)
    R = sec.setting_rows(setting, "b", sec.bar_columns("[Bc; B0]", "b") + ["rank_F_mean"], genes)
    models, width = sec.models_for(name, genes)
    result = sec.fit_folds(R, models)
    stats = fits.spearman_draws(result, sec.SETTING_SEEDS[setting])
    base = sec.base_of(B0_ITEM, name, width, "b", setting, result, "ridge", "gene folds", "genes")
    base.update({"item": B0_ITEM, "role": "diagnostic; " + STATUS, "bar": "[Bc; F_mean; B0]", "depmap_release": sup.NOT_CURRENT})
    row = {**sec.contrast_row(stats, base), **sec.context("b", setting, width), "reading": "diagnostic, no reading (B0 is observed after the perturbation)",
           "svd_fallbacks": int(e5fit.SVD_FALLBACKS[0]), "seconds": round(time.time() - started, 1)}
    return {"key": list(args), "rows": [row], "comparisons": []}


TASKS = {"same_rows": same_rows_task, "route": route_task, "b0": b0_task}


def task_spec(key) -> str:
    return json.dumps(strict_json({"key": [str(k) for k in key], "replicates": fits.B, "smoke": SMOKE, "settings": list(sec.active_settings()),
                                   "code": {**sec.DATA["code"], "make_e5_supplement.py": sup.my_digest(), "make_e5_supplement_2.py": my_digest()},
                                   "inputs": sec.DATA["inputs"], "depmap": DATA.get("depmap_digest", "")}), sort_keys=True)


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


def load_inputs() -> dict:
    """E5's Replogle inputs, the 50-cell gates and DepMap's columns; DepMap's coverage must equal analysis 2's."""
    sec.load_replogle()
    sec.load_prefit50()
    D = sup.load_depmap()
    coverage = sup.add_depmap_columns(D)
    first = json.loads((FIRST / "supplement_depmap.json").read_text())["depmap"]["coverage"]
    if coverage != first:
        raise Stop("DepMap's coverage %s differs from analysis 2's %s" % (coverage, first))
    DATA["depmap_digest"] = D["record"]["sha256"]
    sec.ITEMS[ROUTE] = {"labels": ("F_mean",), "bar": None, "folds": "gene folds", "role": "diagnostic"}
    return {"coverage": coverage, "record": D["record"]}


def run_tasks(listed: list, name: str, started: float) -> list:
    killed = WORKERS_FILE.with_name(WORKERS_FILE.name + ".killed") if WORKERS_FILE is not None else None
    done, failed, shed = fit_slots.run_slots(listed, saved_task, lambda t: " | ".join(t), lambda: fit_slots.limit_from(WORKERS_FILE, WORKERS),
                                             name, started, killed=killed)
    if failed:
        raise Stop("%d tasks failed: %s" % (len(failed), "; ".join("%s: %s" % (f["fit"], f["error"]) for f in failed[:5])))
    DATA["shed"] = shed
    return done


def ordered(done: list) -> list:
    return sorted(done, key=lambda o: tuple(sup.EMBEDDINGS.index(k) if k in sup.EMBEDDINGS else (list(sec.SETTINGS).index(k) if k in sec.SETTINGS else k)
                                            for k in o["key"][1:]))


def run_fits() -> None:
    started = time.time()
    loaded = load_inputs()
    settings = sec.active_settings()
    listed = [("same_rows", v, name, s) for v in VERSIONS for name in sorted(sup.EMBEDDINGS, key=lambda n: -fits.WIDTHS[n]) for s in settings]
    print("[e5 supplement 2 fits] %d tasks, %.0f s" % (len(listed), time.time() - started), flush=True)
    done = ordered(run_tasks(listed, "e5 supplement 2 fits", started))
    rows = [r for o in done for r in o["rows"]]
    comparisons = [c for o in done for c in o["comparisons"]]
    check = check_against_depmap(rows)
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "same_rows.csv", index=False)
    pd.DataFrame(comparisons).to_csv(folder / "comparisons.csv", index=False)
    write_record(folder, "fits", started, ["same_rows.csv", "comparisons.csv"], {
        "checks": {"[Bc; F] equals analysis 2 (depmap.csv)": check}, "depmap": {"release": sup.NOT_CURRENT, "coverage": loaded["coverage"],
                                                                                "download_record": loaded["record"]},
        "tasks": {"listed": len(listed), "reused": int(sum(o["reused"] for o in done)), "shed_and_queued_again": DATA["shed"],
                  "folder": config.portable(tasks_dir())}})
    print("[e5 supplement 2 fits] %d rows, %d comparisons, check %s, %.0f s" % (len(rows), len(comparisons), json.dumps(
        {k: check[k] for k in ("compared", "largest_difference")}), time.time() - started), flush=True)


def run_route_and_b0(part: str) -> None:
    started = time.time()
    loaded = load_inputs()
    settings = sec.active_settings()
    listed = [(part, name, s) for name in sorted(sup.EMBEDDINGS, key=lambda n: -fits.WIDTHS[n]) for s in settings]
    print("[e5 supplement 2 %s] %d tasks, %.0f s" % (part, len(listed), time.time() - started), flush=True)
    done = ordered(run_tasks(listed, "e5 supplement 2 " + part, started))
    rows = [r for o in done for r in o["rows"]]
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    outputs = [f"{part}.csv"]
    pd.DataFrame(rows).to_csv(folder / outputs[0], index=False)
    extra = {"depmap": {"release": sup.NOT_CURRENT, "coverage": loaded["coverage"], "download_record": loaded["record"]},
             "tasks": {"listed": len(listed), "reused": int(sum(o["reused"] for o in done)), "shed_and_queued_again": DATA["shed"]}}
    if part == "b0":
        association = []
        for screen in sec.SCREENS:
            frame = fits.DATA["screens"][screen]["frame"]
            genes = sorted(g for g in fits.DATA["covered"][screen] if np.isfinite(frame.loc[g, "F_mean"]) and np.isfinite(frame.loc[g, "B0"]))
            b0, f = frame.loc[genes, "B0"].to_numpy(dtype=np.float64), frame.loc[genes, "F_mean"].to_numpy(dtype=np.float64)
            point = resample.ClusterDraw(weights=np.ones((1, len(genes))), clusters=np.array([]), seed=-1, index_sha256="")
            draw = resample.cluster_draw(np.array(genes), fits.B, B0_SEEDS[screen])
            estimate = float(resample.weighted_spearman(b0[:, None], f, point)[0, 0])
            draws = resample.weighted_spearman(b0[:, None], f, draw)[:, 0]
            lo, hi, missing = percentile_interval(draws)
            association.append({"item": "Spearman of B0 with F_mean", "role": "descriptive; " + STATUS, "screen": screen, "n_genes": len(genes),
                                "estimate": estimate, "lo": lo, "hi": hi, "p": resample.p_value(draws), "state": state(lo, hi),
                                "replicates_not_finite": missing, "seed": B0_SEEDS[screen], "replicate_index_sha256": draw.index_sha256,
                                "depmap_release": sup.NOT_CURRENT})
        pd.DataFrame(association).to_csv(folder / "b0_with_f_mean.csv", index=False)
        outputs.append("b0_with_f_mean.csv")
    write_record(folder, part, started, outputs, extra)
    print("[e5 supplement 2 %s] %d rows, %.0f s" % (part, len(rows), time.time() - started), flush=True)


# --------------------------------------------------------------------------- Section 5


def run_hgnc() -> None:
    started = time.time()
    record_path = OUT / "hgnc_download.json"
    if not record_path.is_file():
        raise Stop("no HGNC download record; (9) is recorded as a limitation")
    record = json.loads(record_path.read_text())
    path = config.storage_root() / record["file"]["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if not record.get("complete") or sha256(path) != record["file"]["sha256"]:
        raise Stop("the HGNC file does not match its download record")
    hgnc = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    approved = {s.upper(): s for s in hgnc["symbol"] if s}
    other: dict[str, set] = {}
    for _, r in hgnc.iterrows():
        for column in ("prev_symbol", "alias_symbol"):
            for s in str(r[column]).split("|"):
                s = s.strip().strip('"')
                if s:
                    other.setdefault(s.upper(), set()).add(r["symbol"])
    lists = json.loads((FIRST / "supplement_lists.json").read_text())
    vocab_record = lists["inputs"]["scgpt_vocab"] if "scgpt_vocab" in lists.get("inputs", {}) else lists["scgpt_vocab"]
    vocab_path = config.storage_root() / vocab_record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(vocab_path) != vocab_record["sha256"]:
        raise Stop("scGPT's vocabulary does not match its record")
    vocab = set(json.loads(vocab_path.read_text()))
    coverage = list(csv.DictReader(open(FIRST / "coverage_lists.csv")))
    mapping = list(csv.DictReader(open(FIRST / "depmap_mapping.csv")))
    misses = []
    for r in coverage:
        for source in ("scGPT", "GenePT"):
            if r[f"{source}_vector"] == "False":
                misses.append({"source": source, "screen": r["screen"], "target_ensg": r["target_ensg"], "symbol": r["target_symbol"]})
    for r in mapping:
        if r["mapping"] == "miss":
            misses.append({"source": "DepMap", "screen": r["screen"], "target_ensg": r["target_ensg"], "symbol": r["target_symbol"]})
    for m in misses:
        u = m["symbol"].upper()
        if u in approved:
            m["hgnc"], m["approved_symbol"] = "an approved symbol", approved[u]
        elif u in other and len(other[u]) == 1:
            m["hgnc"], m["approved_symbol"] = "a previous or alias symbol of one approved gene", next(iter(other[u]))
        elif u in other:
            m["hgnc"], m["approved_symbol"] = "a previous or alias symbol of several approved genes", "|".join(sorted(other[u]))
        else:
            m["hgnc"], m["approved_symbol"] = "not in HGNC", ""
        if m["source"] == "scGPT":
            one = m["hgnc"] in ("an approved symbol", "a previous or alias symbol of one approved gene")
            m["approved_symbol_in_scgpt_vocabulary"] = bool(one and m["approved_symbol"] in vocab) if one else ""
        else:
            m["approved_symbol_in_scgpt_vocabulary"] = ""
    frame = pd.DataFrame(misses)
    counts = []
    for source in ("scGPT", "GenePT", "DepMap"):
        part = frame[frame["source"] == source]
        for screen, group in [("both screens, distinct genes", part.drop_duplicates("target_ensg"))] + [(s, part[part["screen"] == s]) for s in sec.SCREENS]:
            entry = {"source": source, "screen": screen, "misses": int(len(group))}
            for kind in ("an approved symbol", "a previous or alias symbol of one approved gene", "a previous or alias symbol of several approved genes",
                         "not in HGNC"):
                entry[kind] = int((group["hgnc"] == kind).sum())
            if source == "scGPT":
                entry["approved symbol in scGPT's vocabulary"] = int((group["approved_symbol_in_scgpt_vocabulary"] == True).sum())  # noqa: E712
            counts.append({**entry, "role": "count only, no refit and no change of the covered gene set; " + STATUS})
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "hgnc_misses.csv", index=False)
    pd.DataFrame(counts).to_csv(folder / "hgnc_counts.csv", index=False)
    write_record(folder, "hgnc", started, ["hgnc_misses.csv", "hgnc_counts.csv"], {
        "hgnc": {"download_record": file_record(record_path), "rows": int(len(hgnc)), "approved_symbols": len(approved),
                 "matching": "case-insensitive, on the columns symbol, prev_symbol and alias_symbol"},
        "scgpt_vocabulary": {**vocab_record, "tokens": len(vocab)}})
    print("[e5 supplement 2 hgnc] %d misses, %.0f s" % (len(frame), time.time() - started), flush=True)


def write_record(folder: Path, part: str, started: float, outputs: list[str], extra: dict) -> None:
    record = {"plan": PLAN, "status": STATUS, "smoke": SMOKE, "part": part, "replicates": fits.B,
              "seeds": {"settings": sec.SETTING_SEEDS, "within folds": e5lib.WITHIN_FOLD_SEED, "union folds": fits.UNION_FOLD_SEED,
                        "inner folds": e5fit.INNER_SEED, "shuffle": e5fit.SHUFFLE_SEED, "B0 with F_mean": B0_SEEDS},
              "settings": list(sec.active_settings()), "inputs": sec.DATA.get("inputs", {}),
              "code": {**sec.DATA.get("code", {}), config.portable(Path(sup.__file__).resolve()): sup.my_digest(),
                       config.portable(Path(__file__).resolve()): my_digest()}, **extra,
              "outputs": {n: file_record(folder / n) for n in outputs}, "seconds": round(time.time() - started, 1)}
    (folder / f"supplement_2_{part}.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        sup.SMOKE = True
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
    if arguments or not set(parts) <= set(PARTS) or len(set(parts)) != len(parts):
        raise SystemExit("usage: make_e5_supplement_2.py [--smoke] [--workers N] [--workers-file PATH] [--part fits|route|b0|hgnc ...]")
    for part in parts or PARTS:
        {"fits": run_fits, "route": lambda: run_route_and_b0("route"), "b0": lambda: run_route_and_b0("b0"), "hgnc": run_hgnc}[part]()
