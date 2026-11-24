"""Check of per-task saving in E5's secondary main part (gate G4, item 4: no value may change; smoke first).

Compares a main-part smoke of new code (first: per-task saving, stopped by the memory rules and resumed from its
saved tasks; then: one process per task) with an earlier smoke (first: made in one go by make_e5_secondary.py at
7062cdf, without saving; then: the smoke of per-task saving made at 20f5f53, and again the smoke made at 7062cdf). The check as first written required every table to have the same columns and every table
cell (the timing column excepted), every prediction array and the record's checks, counts and inputs to be equal.
Its result is kept as it comes out ("passed_as_first_written").

The corrected check (the author's answer of 2026-11-15, bioinf2027/GATE.md) passes when every cell of the
reference's columns (the timing column excepted) and every prediction array is equal, no column of the reference is
missing, and the record's counts, items, seeds, replicates and settings are equal, with these differences listed as
expected when E5's family table exists for the new run only: the column family_holm_first_step_detectable of
power.csv, the family check (compared, within its tolerance, against not compared) and the family table among the
inputs. Inputs that only the reference lists (parts it ran beside the main part) are listed; every input the new run
lists must be the reference's, the family table excepted. Every difference is listed in any case.
Usage: check_e5_saving_smoke.py <new folder> <reference folder> [<name>] (both under
$IGA_STORAGE_ROOT/bioinf2027/scratch/); the result goes to bioinf2027/results/E5/checks/<name>.json (default
saving_smoke).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bioinf2027" / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bioinf2027 import config  # noqa: E402

TABLES = ["secondary.csv", "diagnostics.csv", "folds.csv", "between_embeddings.csv", "family_baseline.csv", "power.csv", "agreement.csv"]
FIRST_RECORD_FIELDS = ("checks", "statistics", "svd_fallbacks_total", "covered_genes", "prefixes", "inputs", "items", "seeds", "replicates", "settings")
EQUAL_RECORD_FIELDS = ("statistics", "svd_fallbacks_total", "covered_genes", "prefixes", "items", "seeds", "replicates", "settings")
FAMILY_COLUMNS = {"power.csv": ["family_holm_first_step_detectable"]}  # written only when the family table exists
FAMILY_CHECK = "family recomputed"


def walk(a, b, path: str = "") -> list[str]:
    if type(a) is not type(b):
        return ["%s: %s; reference %s" % (path, json.dumps(a)[:300], json.dumps(b)[:300])]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append("%s.%s: only in the reference: %s" % (path, k, json.dumps(b[k])[:300]))
            elif k not in b:
                out.append("%s.%s: only in the new: %s" % (path, k, json.dumps(a[k])[:300]))
            else:
                out.extend(walk(a[k], b[k], "%s.%s" % (path, k)))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return ["%s: length %d; reference %d" % (path, len(a), len(b))]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in walk(x, y, "%s[%d]" % (path, i))]
    return [] if a == b else ["%s: %s; reference %s" % (path, json.dumps(a)[:300], json.dumps(b)[:300])]


def main() -> int:
    root = config.storage_root() / "bioinf2027" / "scratch"
    new, ref = root / sys.argv[1], root / sys.argv[2]
    tables, first_ok, cells_compared, cells_differing = {}, True, 0, 0
    for name in TABLES:
        a = pd.read_csv(new / name, dtype=str, keep_default_na=False)
        b = pd.read_csv(ref / name, dtype=str, keep_default_na=False)
        entry = {"rows": len(a), "rows_reference": len(b), "columns_only_in_new": [c for c in a.columns if c not in b.columns],
                 "columns_only_in_reference": [c for c in b.columns if c not in a.columns]}
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            first_ok = False
        common = [c for c in b.columns if c in a.columns and c != "seconds"]
        if len(a) == len(b):
            diff = a[common] != b[common]
            entry.update(cells_compared=int(a[common].size), cells_differing=int(diff.values.sum()),
                         columns_differing={c: int(n) for c, n in diff.sum().items() if n})
            cells_compared += entry["cells_compared"]
            cells_differing += entry["cells_differing"]
            first_ok = first_ok and entry["cells_differing"] == 0
        tables[name] = entry
    with np.load(new / "data" / "predictions_main.npz", allow_pickle=False) as x, np.load(ref / "data" / "predictions_main.npz", allow_pickle=False) as y:
        names = sorted(set(x.files) | set(y.files))
        arrays_differing = [k for k in names if k not in x.files or k not in y.files or not (
            x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f"))]
        arrays = {"new": len(x.files), "reference": len(y.files), "differing": arrays_differing}
    first_ok = first_ok and not arrays_differing
    ra, rb = json.loads((new / "secondary_main.json").read_text()), json.loads((ref / "secondary_main.json").read_text())
    first_ok = first_ok and all(ra.get(k) == rb.get(k) for k in FIRST_RECORD_FIELDS)

    family_new = bool(ra["checks"].get(FAMILY_CHECK, {}).get("compared"))
    family_ref = bool(rb["checks"].get(FAMILY_CHECK, {}).get("compared"))
    family_only_new = family_new and not family_ref
    expected = {"columns": FAMILY_COLUMNS if family_only_new else {}, "family_check": family_only_new,
                "inputs": ["family"] if family_only_new else []}
    problems = []
    for name, entry in tables.items():
        if entry["columns_only_in_reference"]:
            problems.append("%s: reference columns missing %s" % (name, entry["columns_only_in_reference"]))
        if sorted(entry["columns_only_in_new"]) != sorted(expected["columns"].get(name, [])):
            problems.append("%s: columns only in the new %s, expected %s" % (name, entry["columns_only_in_new"], expected["columns"].get(name, [])))
        if entry.get("cells_differing", 1) or entry["rows"] != entry["rows_reference"]:
            problems.append("%s: cells or rows differ" % name)
    if arrays_differing or arrays["new"] != arrays["reference"]:
        problems.append("prediction arrays differ")
    problems += ["record field %s differs" % k for k in EQUAL_RECORD_FIELDS if ra.get(k) != rb.get(k)]
    other_checks = sorted(set(ra["checks"]) | set(rb["checks"]))
    for k in other_checks:
        if k == FAMILY_CHECK and family_only_new:
            check = ra["checks"][k]
            if not check["largest_difference"] <= check["tolerance"]:
                problems.append("the family check exceeds its tolerance")
        elif ra["checks"].get(k) != rb["checks"].get(k):
            problems.append("record check %s differs" % k)
    for k, v in ra["inputs"].items():
        if k in expected["inputs"]:
            continue
        if rb["inputs"].get(k) != v:
            problems.append("input %s is not the reference's" % k)
    only_reference_inputs = sorted(k for k in rb["inputs"] if k not in ra["inputs"])
    result = {"check": "E5 secondary main part: a main-part smoke of new code, stopped and resumed, against an earlier smoke; the two "
                       "folders are named under new and reference (gate G4, item 4)",
              "new": config.portable(new), "reference": config.portable(ref), "passed_as_first_written": first_ok,
              "passed": not problems, "rule": "corrected check (the author's answer of 2026-11-15, bioinf2027/GATE.md); see this script's docstring",
              "problems": problems, "expected_differences": expected, "inputs_only_in_the_reference": only_reference_inputs,
              "table_cells_of_the_reference_columns_compared": cells_compared, "table_cells_differing": cells_differing,
              "tables": tables, "prediction_arrays": arrays, "record_differences": walk(ra, rb),
              "saved_tasks_new": ra.get("saved_tasks"), "made": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out = REPO / "bioinf2027" / "results" / "E5" / "checks" / ("%s.json" % (sys.argv[3] if len(sys.argv) > 3 else "saving_smoke"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("passed: %s (problems %s); passed as first written: %s; %d table cells of the reference's columns compared, %d differ; %d prediction arrays, %d differ; "
          "columns only in the new: %s; %d record differences"
          % (not problems, problems, first_ok, cells_compared, cells_differing, arrays["new"], len(arrays_differing),
             {k: v["columns_only_in_new"] for k, v in tables.items() if v["columns_only_in_new"]}, len(result["record_differences"])))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
