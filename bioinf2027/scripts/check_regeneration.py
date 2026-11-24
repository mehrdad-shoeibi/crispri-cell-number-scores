"""Compare regenerated tables with the versions held at a git revision.

A regeneration may add columns or rename them, but every value that existed
before must come out unchanged. For each table this script reports the columns
compared, renamed, added and removed, and, for every compared column, whether
all values are identical; numeric columns also get the largest absolute
difference. With --tolerance, a numeric column whose values differ by no more
than the tolerance, with the same missing values, is reported as within
tolerance; the default tolerance is zero, so by default every value must be
identical. The result is written to bioinf2027/diag/<item>_regeneration_check.json
and the script exits with a non-zero code when any compared value differs by
more than the tolerance.

Usage, from the repository root:

    python bioinf2027/scripts/check_regeneration.py --item D1 --against a2bc766 \
        --rename n=n_rows --rename n_groups=n_target_genes \
        bioinf2027/diag/D1_label_vs_own_delta_vcc.csv ...
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "diag"


def read_at_revision(repo: Path, revision: str, path: str) -> pd.DataFrame:
    text = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    return pd.read_csv(io.StringIO(text), float_precision="round_trip")


def compare_tables(
    old: pd.DataFrame, new: pd.DataFrame, rename: dict[str, str], tolerance: float = 0.0
) -> dict:
    """Value-by-value comparison of the columns the two versions share."""
    mapped = {column: rename.get(column, column) for column in old.columns}
    report = {
        "rows_before": int(len(old)),
        "rows_after": int(len(new)),
        "renamed": {k: v for k, v in mapped.items() if k != v and v in new.columns},
        "removed": sorted(k for k, v in mapped.items() if v not in new.columns),
        "added": sorted(set(new.columns) - set(mapped.values())),
        "columns": {},
    }
    same_rows = len(old) == len(new)
    for before, after in mapped.items():
        if after not in new.columns:
            continue
        a, b = old[before], new[after]
        entry = {"compared_as": after}
        if not same_rows:
            entry["identical"] = False
            entry["within_tolerance"] = False
        elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
            entry["identical"] = bool(np.array_equal(x, y, equal_nan=True))
            both = np.isfinite(x) & np.isfinite(y)
            entry["max_abs_difference"] = (
                float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0
            )
            entry["n_values_differing"] = int((x[both] != y[both]).sum())
            entry["missing_pattern_identical"] = bool(np.array_equal(~np.isfinite(x), ~np.isfinite(y)))
            entry["within_tolerance"] = bool(
                entry["identical"]
                or (entry["missing_pattern_identical"] and entry["max_abs_difference"] <= tolerance)
            )
        else:
            entry["identical"] = bool(a.fillna("").astype(str).equals(b.fillna("").astype(str)))
            entry["within_tolerance"] = entry["identical"]
        report["columns"][before] = entry
    report["all_values_identical"] = bool(
        same_rows and all(entry["identical"] for entry in report["columns"].values())
    )
    report["all_values_within_tolerance"] = bool(
        same_rows and all(entry["within_tolerance"] for entry in report["columns"].values())
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--item", required=True)
    parser.add_argument("--against", required=True, help="git revision holding the earlier tables")
    parser.add_argument("--rename", action="append", default=[], help="old=new column name")
    parser.add_argument(
        "--tolerance", type=float, default=0.0, help="largest absolute difference accepted in a numeric column"
    )
    parser.add_argument("tables", nargs="+", help="repository-relative table paths")
    options = parser.parse_args()

    rename = dict(item.split("=", 1) for item in options.rename)
    repo = config.repo_root()
    revision = subprocess.run(
        ["git", "rev-parse", options.against], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    tables = {}
    for path in options.tables:
        old = read_at_revision(repo, revision, path)
        new = pd.read_csv(repo / path, float_precision="round_trip")
        tables[path] = compare_tables(old, new, rename, options.tolerance)

    passed = all(table["all_values_within_tolerance"] for table in tables.values())
    result = {
        "item": options.item,
        "compared_against": revision,
        "rename": rename,
        "tolerance": options.tolerance,
        "tables": tables,
        "all_values_identical": all(table["all_values_identical"] for table in tables.values()),
        "all_values_within_tolerance": passed,
    }
    out = OUT_DIR / f"{options.item}_regeneration_check.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    for path, table in tables.items():
        if table["all_values_identical"]:
            state = "identical"
        elif table["all_values_within_tolerance"]:
            state = "within tolerance"
        else:
            state = "DIFFERS"
        print(f"{path}: {state}; added {table['added']}; renamed {table['renamed']}")
    print("wrote", config.portable(out))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
