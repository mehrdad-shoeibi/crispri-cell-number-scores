"""The closing round of missing analyses, from saved files only (plan: bioinf2027/plans/missing_analyses_closing.md;
the author's answer to the stop of 2026-11-17 17:10, item 9, bioinf2027/GATE.md). Post hoc, unadjusted, no reading.

1. S1 supplement, missing 4: the all-cells VCC label within (target gene, guide), restricted to the rows that enter
   R2's simulation, with the S1 supplement's target-gene bootstrap; the saved mechanical term printed beside it.
2. S1 supplement, missing 6: constructs excluded at the 27-cell floor per screen (D3), beside part (c).
3. E5 supplement 2, analysis 8: Holm within each version's 12 for the joint comparator.
4. E5 supplement 2, analyses 2 and 6: limitations; the saved task files are inspected and their contents recorded.
Usage, from the repository root (through run_item.py): python bioinf2027/scripts/make_missing_analyses_closing.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bioinf2027" / "src"))
sys.path.insert(0, str(REPO / "bioinf2027" / "scripts"))

from bioinf2027 import config, stats  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
import make_s1_supplement as s1sup  # noqa: E402  (spearman() and centred() reused unchanged)
import safe_pickle  # noqa: E402

B = REPO / "bioinf2027"
OUT = B / "results" / "missing_analyses_closing"
PLAN = "bioinf2027/plans/missing_analyses_closing.md"
STATUS = "post hoc, unadjusted, not part of any family, no reading (the author's answer to the stop of 2026-11-17 17:10, item 9; %s)" % PLAN
SEED, REPLICATES, TOLERANCE = stats.BOOTSTRAP_SEED, stats.BOOTSTRAP_B, 1e-12  # R2's and the S1 supplement's: 20270301, 10,000
SPLITS = ("train", "val", "test")
CELLS = ("G5", "G6")
S1 = B / "results" / "S1_supplement"
D3 = B / "diag" / "D3_cell_counts.csv"
SAME = B / "results" / "E5" / "supplement_2" / "same_rows.csv"


TASK_RECORDS: dict = {}  # every saved task file read, with its size and SHA-256


class Stop(Exception):
    pass


spearman, centred = s1sup.spearman, s1sup.centred


def observed_term(observed: pd.DataFrame, simulated_genes: np.ndarray) -> dict:
    """The observed term and its interval exactly as the S1 supplement's mechanical() draws them (genes from the union of
    the observed and the simulated rows' genes, one generator with seed 20270301, percentile interval)."""
    obs = spearman(observed["c_n"].to_numpy(), observed["c_y"].to_numpy())
    genes = np.unique(np.r_[observed["target_gene"].to_numpy(), simulated_genes])
    members = {g: np.flatnonzero(observed["target_gene"].to_numpy() == g) for g in genes}
    rng = np.random.default_rng(SEED)
    draws = np.full(REPLICATES, np.nan)
    on, oy = observed["c_n"].to_numpy(), observed["c_y"].to_numpy()
    for r in range(REPLICATES):
        picked = genes[rng.integers(0, genes.size, size=genes.size)]
        io = np.concatenate([members[g] for g in picked])
        draws[r] = spearman(on[io], oy[io]) if io.size else np.nan
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"estimate": obs, "lo": float(lo), "hi": float(hi), "replicates_undefined": int(np.isnan(draws).sum()), "target_genes_drawn": int(genes.size)}


def single_guide_groups(labels: pd.DataFrame) -> pd.DataFrame:
    """The S1 supplement's observed rows (make_s1_supplement.py:357-361): one guide, finite all-cells label, groups of at
    least three rows, label and pool size centred within the group."""
    single = labels[(labels["n_guides"] == 1) & labels["existing_label"].notna()].copy()
    single["group"] = single["target_gene"] + "|" + single["guides"].astype(str)
    single = single[single.groupby("group")["existing_label"].transform("size") >= 3].copy()
    single["c_y"] = centred(single["existing_label"].to_numpy(dtype=np.float64), single["group"].to_numpy())
    single["c_n"] = centred(single["n_pool"].to_numpy(float), single["group"].to_numpy())
    return single


def load_cell(cell: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((B / "endpoints" / cell / "manifest.json").read_text())["tables"]
    out, records = [], {}
    for name in ("vcc_labels.csv", "pool_size_simulation.csv"):
        path = Path(config.storage_root()) / "bioinf2027" / "endpoints" / cell / name
        if sha256(path) != manifest[name]["sha256"]:
            raise Stop("%s of %s does not match bioinf2027/endpoints/%s/manifest.json" % (name, cell, cell))
        out.append(pd.read_csv(path, float_precision="round_trip"))
        records[name] = file_record(path)
    records["manifest"] = file_record(B / "endpoints" / cell / "manifest.json")
    return out[0], out[1], records


def run_s1_within(inputs: dict, checks: dict) -> pd.DataFrame:
    saved = pd.read_csv(S1 / "vcc_mechanical.csv", float_precision="round_trip")
    saved["line"] = saved.index + 2
    rows = []
    for cell in CELLS:
        labels, simulation, records = load_cell(cell)
        inputs[cell] = records
        key = lambda f: (f["split"] + "|" + f["batch"] + "|" + f["target_gene"]).to_numpy()  # noqa: E731
        sim_rows = pd.DataFrame({"row": key(simulation), "split": simulation["split"].to_numpy(), "target_gene": simulation["target_gene"].to_numpy()})
        sim_rows = sim_rows.drop_duplicates("row")
        labels = labels.assign(row=key(labels))
        if not set(sim_rows["row"]) <= set(labels["row"]):
            raise Stop("%s: simulated rows missing from vcc_labels.csv" % cell)
        full = single_guide_groups(labels)
        restricted = single_guide_groups(labels[labels["row"].isin(set(sim_rows["row"]))])
        for split in SPLITS + ("all",):
            pick = (lambda f: f) if split == "all" else (lambda f: f[f["split"] == split])  # noqa: E731
            sim_genes = pick(sim_rows)["target_gene"].to_numpy()
            hit = saved[(saved["cell"] == cell) & (saved["split"] == split)]
            if len(hit) != 1:
                raise Stop("vcc_mechanical.csv has %d rows for %s, %s" % (len(hit), cell, split))
            hit = hit.iloc[0]
            # check: the unrestricted computation reproduces the saved observed term
            again = observed_term(pick(full), sim_genes)
            diffs = {c: abs(again[k] - float(hit[c])) for k, c in (("estimate", "observed_within_gene_and_guide"), ("lo", "observed_lo"), ("hi", "observed_hi"))}
            if int(len(pick(full))) != int(hit["observed_rows"]) or max(diffs.values()) > TOLERANCE:
                raise Stop("%s, %s: the saved observed term is not reproduced (%s)" % (cell, split, diffs))
            checks.setdefault("observed_term_reproduced", {})["%s %s" % (cell, split)] = {"rows": int(len(pick(full))), "largest_difference": max(diffs.values())}
            part = pick(restricted)
            value = observed_term(part, sim_genes)
            rows.append({"analysis": "S1 supplement, missing 4", "cell": cell, "split": split, "label": "existing (all-cells) label",
                         "quantity": "within (target gene, guide), on the rows that enter R2's simulation",
                         "estimate": value["estimate"], "lo": value["lo"], "hi": value["hi"],
                         "observed_rows": int(len(part)), "observed_groups": int(part["group"].nunique()), "observed_target_genes": int(part["target_gene"].nunique()),
                         "simulated_rows": int(len(pick(sim_rows))), "target_genes_drawn": value["target_genes_drawn"],
                         "replicates_undefined": value["replicates_undefined"],
                         "saved_observed_all_single_guide_rows": float(hit["observed_within_gene_and_guide"]), "saved_observed_lo": float(hit["observed_lo"]),
                         "saved_observed_hi": float(hit["observed_hi"]), "saved_observed_rows": int(hit["observed_rows"]),
                         "saved_mechanical": float(hit["mechanical"]), "saved_mechanical_lo": float(hit["mechanical_lo"]), "saved_mechanical_hi": float(hit["mechanical_hi"]),
                         "saved_simulated_labels": int(hit["simulated_labels"]),
                         "saved_line": "bioinf2027/results/S1_supplement/vcc_mechanical.csv:%d" % hit["line"],
                         "excess": "not formed: the mechanical term on the restricted rows needs the simulated all-cells labels per row, not saved (a limitation)",
                         "role": STATUS})
            print("[closing s1-4] %s %s: %d of %d simulated rows in groups of at least three; %+.3f [%+.3f, %+.3f]" % (
                cell, split, len(part), len(pick(sim_rows)), value["estimate"], value["lo"], value["hi"]), flush=True)
    return pd.DataFrame(rows)


def run_s1_floor() -> pd.DataFrame:
    d3 = pd.read_csv(D3)
    d3["line"] = d3.index + 2
    assoc = pd.read_csv(S1 / "external_association.csv", float_precision="round_trip")
    assoc["line"] = assoc.index + 2
    labels = pd.read_csv(S1 / "external_labels.csv").groupby("screen").size()
    rows = []
    for screen, assoc_line in (("RPE1", 2), ("K562-essential", 10)):
        r = d3[d3["screen"] == screen].iloc[0]
        a = assoc[assoc["line"] == assoc_line].iloc[0]
        if a["screen"] != screen or a["quantity"] != "rho(all cells against 788 pooled controls label, n)":
            raise Stop("external_association.csv:%d is not %s's all-cells association" % (assoc_line, screen))
        if not int(r["n_eligible"]) == int(a["constructs"]) == int(labels[screen]):
            raise Stop("%s: n_eligible %s, part (c) %s, external_labels.csv %s" % (screen, r["n_eligible"], a["constructs"], labels[screen]))
        if int(r["denom2_post_join"]) - int(r["n_eligible"]) != int(r["excluded_too_few_cells"]):
            raise Stop("%s: D3's exclusions do not add up" % screen)
        rows.append({"analysis": "S1 supplement, missing 6", "screen": screen, "constructs_after_join": int(r["denom2_post_join"]),
                     "with_at_least_27_cells": int(r["n_eligible"]), "excluded_too_few_cells": int(r["excluded_too_few_cells"]),
                     "excluded_target_not_in_var": int(r["excluded_target_not_in_var"]), "excluded_both": int(r["excluded_both"]),
                     "d3_line": "bioinf2027/diag/D3_cell_counts.csv:%d" % r["line"],
                     "part_c_all_cells_association": float(a["estimate"]), "part_c_lo": float(a["lo"]), "part_c_hi": float(a["hi"]),
                     "part_c_line": "bioinf2027/results/S1_supplement/external_association.csv:%d" % assoc_line, "role": "no new computation; " + STATUS})
    return pd.DataFrame(rows)


def holm_adjusted(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for step, i in enumerate(order):
        running = max(running, min(1.0, (m - step) * p[i]))
        adjusted[i] = running
    return adjusted


def run_holm(checks: dict) -> pd.DataFrame:
    same = pd.read_csv(SAME, float_precision="round_trip")
    same["line"] = same.index + 2
    out = []
    for version in ("a", "b"):
        part = same[(same["version"] == version) & (same["comparator"] == "J")].copy()
        if len(part) != 12 or not np.isfinite(part["iu_p"].to_numpy()).all():
            raise Stop("same_rows.csv: version %s has %d joint rows or a missing p" % (version, len(part)))
        p = part["iu_p"].to_numpy(dtype=np.float64)
        adjusted = holm_adjusted(p)
        # Holm's step-down decisions, as the report-table script computes them (make_e5_supplement_report_tables.py, holm())
        order = sorted(range(len(p)), key=lambda i: p[i])
        steps = [False] * len(p)
        for step, i in enumerate(order):
            if p[i] > 0.05 / (len(p) - step):
                break
            steps[i] = True
        if [bool(a <= 0.05) for a in adjusted] != steps:
            raise Stop("version %s: the adjusted p-values and the step-down decisions disagree" % version)
        for (_, r), a, kept in zip(part.iterrows(), adjusted, steps):
            out.append({"analysis": "E5 supplement 2, analysis 8", "version": version, "embedding": r["embedding"], "setting": r["setting"],
                        "n_test_genes": int(r["n_test_genes"]), "iu_estimate": float(r["iu_estimate"]), "iu_p": float(r["iu_p"]),
                        "holm_adjusted_p": float(a), "holm_rejected_at_0.05": bool(kept), "both_resolved_above_zero": bool(r["both_resolved_above_zero"]),
                        "prefix_both_resolved_above_zero": bool(r["prefix_both_resolved_above_zero"]), "depmap_release": r["depmap_release"],
                        "line": "bioinf2027/results/E5/supplement_2/same_rows.csv:%d" % r["line"], "role": "descriptive; " + STATUS})
        frame = pd.DataFrame(out[-12:])
        positive = frame[frame["iu_estimate"] > 0]
        checks.setdefault("holm", {})[version] = {"rejected": int(frame["holm_rejected_at_0.05"].sum()),
                                                  "rejected_with_estimate_above_zero": int(positive["holm_rejected_at_0.05"].sum()),
                                                  "estimate_above_zero": int(len(positive)),
                                                  "rejected_with_estimate_below_zero": int(frame[(frame["iu_estimate"] < 0)]["holm_rejected_at_0.05"].sum()),
                                                  "not_rejected_lines": [x for x, k in zip(frame["line"], frame["holm_rejected_at_0.05"]) if not k]}
    # the review's own computation as the report quotes it (bioinf2027/reports/E5.md:829-830; the review,
    # bioinf2027/reviews/E5_supplement_2_results_r1.md:57): version a keeps every row with both intervals above zero and the
    # negative one; version b keeps 9 of the 11 with both intervals above zero and drops lines 65 and 81
    frame = pd.DataFrame(out)
    statement = {"a": {"kept_among_resolved": "all", "dropped_among_resolved": [], "kept_negative": 1},
                 "b": {"kept_among_resolved": 9, "dropped_among_resolved": ["bioinf2027/results/E5/supplement_2/same_rows.csv:65",
                                                                            "bioinf2027/results/E5/supplement_2/same_rows.csv:81"], "kept_negative": 0}}
    comparison = {}
    for version, expect in statement.items():
        v = frame[frame["version"] == version]
        resolved = v[v["both_resolved_above_zero"]]
        observed = {"resolved_above_zero": int(len(resolved)), "kept_among_resolved": int(resolved["holm_rejected_at_0.05"].sum()),
                    "dropped_among_resolved": sorted(resolved.loc[~resolved["holm_rejected_at_0.05"], "line"]),
                    "kept_negative": int(v[(v["iu_estimate"] < 0) & v["holm_rejected_at_0.05"]].shape[0])}
        wanted_kept = observed["resolved_above_zero"] if expect["kept_among_resolved"] == "all" else expect["kept_among_resolved"]
        differences = [k for k, a, b in (("kept_among_resolved", observed["kept_among_resolved"], wanted_kept),
                                         ("dropped_among_resolved", observed["dropped_among_resolved"], sorted(expect["dropped_among_resolved"])),
                                         ("kept_negative", observed["kept_negative"], expect["kept_negative"])) if a != b]
        comparison[version] = {**observed, "differences_from_the_review": differences}
        checks["holm"][version]["both_resolved_above_zero"] = observed["resolved_above_zero"]
        print("[closing holm] version %s: %d of 12 kept; among the %d with both intervals above zero %d kept, dropped %s; against the review: %s" % (
            version, int(v["holm_rejected_at_0.05"].sum()), observed["resolved_above_zero"], observed["kept_among_resolved"],
            ", ".join(x.split("/")[-1] for x in observed["dropped_among_resolved"]) or "none", "no difference" if not differences else differences), flush=True)
    checks["holm"]["comparison_with_the_review"] = {"source": "bioinf2027/reports/E5.md:829-830; bioinf2027/reviews/E5_supplement_2_results_r1.md:57",
                                                    **comparison}
    return frame


def inspect_tasks(pattern: str) -> dict:
    """Keys of every saved task file of one kind; a list or array of per-gene values would be recorded here."""
    files = sorted(glob.glob(str(Path(config.storage_root()) / pattern)))
    keys, array_entries = set(), 0
    TASK_RECORDS.update({config.portable(Path(f)): file_record(Path(f)) for f in files})
    for f in files:
        value = safe_pickle.load(Path(f), safe_pickle.NUMPY_ARRAYS)
        output = value["output"]
        keys |= set(output)
        for item in output.values():
            for entry in (item if isinstance(item, list) else [item]):
                if isinstance(entry, np.ndarray) or (isinstance(entry, dict) and any(isinstance(v, (np.ndarray, list)) and len(v) > 50 for v in entry.values())):
                    array_entries += 1
    return {"pattern": "$IGA_STORAGE_ROOT/" + pattern, "files": len(files), "output_keys": sorted(keys), "entries_with_per_gene_arrays": array_entries}


def run_limitations(checks: dict) -> pd.DataFrame:
    b0 = inspect_tasks("bioinf2027/E5/supplement_2/tasks/b0_*.pkl")
    same = inspect_tasks("bioinf2027/E5/supplement_2/tasks/same_*.pkl")
    fn = inspect_tasks("bioinf2027/E5/supplement/tasks/fn_*.pkl")
    checks["task_files"] = {"b0": b0, "same": same, "fn": fn}
    if b0["entries_with_per_gene_arrays"] or same["entries_with_per_gene_arrays"] or fn["entries_with_per_gene_arrays"]:
        raise Stop("a saved task file holds per-gene arrays; the plan's limitation does not hold (%s)" % json.dumps(checks["task_files"]))
    return pd.DataFrame([
        {"analysis": "E5 supplement 2, analysis 2", "quantity": "rho([Bc; F_mean; B0]) - rho([Bc; F_mean]), paired",
         "status": "limitation: needs a refit (the author's rule)", "why": "the saved task files hold summary rows only, no predictions and no replicates",
         "files_inspected": "%s (%d files), %s (%d files)" % (b0["pattern"], b0["files"], same["pattern"], same["files"]),
         "point_estimates_where_saved": "bioinf2027/results/E5/supplement_2/b0.csv (column spearman_bar); bioinf2027/results/E5/supplement_2/same_rows.csv "
                                         "(comparator F, column spearman_bar); no difference is formed", "role": STATUS},
        {"analysis": "E5 supplement 2, analysis 6", "quantity": "prefix-clustered intervals for analysis 1 on its 1,773 and 1,765 rows",
         "status": "limitation: needs a refit (the author's rule)", "why": "analysis 1's saved task files hold one summary row each, no predictions and no replicates",
         "files_inspected": "%s (%d files)" % (fn["pattern"], fn["files"]),
         "point_estimates_where_saved": "bioinf2027/results/E5/supplement/fits.csv (item analysis 1, gene clusters); the family mean with prefix clusters on "
                                         "analysis 2's rows (same_rows.csv, comparator FN) is not on analysis 1's rows and is not substituted", "role": STATUS},
        {"analysis": "S1 supplement, missing 4 (part)", "quantity": "the mechanical term and the excess on exactly the restricted rows",
         "status": "limitation: needs the all-cells label rebuilt on each simulated subset (single-cell data), outside 'from saved files'",
         "why": "the S1 run computed the simulated all-cells labels but saved only their summary (vcc_mechanical.csv)",
         "files_inspected": "bioinf2027/results/S1_supplement/ and $IGA_STORAGE_ROOT/bioinf2027/endpoints/<cell>/pool_size_simulation.csv (rebuilt label only)",
         "point_estimates_where_saved": "the observed term on the simulated rows is in s1_within_on_simulated_rows.csv; the saved mechanical term beside it", "role": STATUS},
    ])


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, checks = {}, {}
    outputs = {}
    tables = {"s1_within_on_simulated_rows.csv": run_s1_within(inputs, checks), "s1_floor_exclusions.csv": run_s1_floor(),
              "e5_joint_holm.csv": run_holm(checks), "limitations.csv": run_limitations(checks)}
    for name, frame in tables.items():
        frame.to_csv(OUT / name, index=False)
        outputs[name] = file_record(OUT / name)
    for path in (S1 / "vcc_mechanical.csv", S1 / "external_association.csv", S1 / "external_labels.csv", D3, SAME):
        inputs[config.portable(path)] = file_record(path)
    inputs["task_files"] = dict(sorted(TASK_RECORDS.items()))
    code = {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (sys.modules[__name__], s1sup, safe_pickle)}
    record = {"plan": PLAN, "status": STATUS, "seed": SEED, "replicates": REPLICATES, "code": code,
              "inputs": inputs, "checks": checks, "seconds": round(time.time() - started, 1), "outputs": outputs}
    (OUT / "closing.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[closing] done in %.0f s: %s" % (time.time() - started, ", ".join(outputs)), flush=True)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: make_missing_analyses_closing.py")
    try:
        main()
    except Stop as stop:
        raise SystemExit("stopped, nothing written beyond this point: %s" % stop)
