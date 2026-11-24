"""The family pass against the full pass on smoke fits (E4 and E6 plans, Section 16): the record of the check.

Read-only on the smoke outputs: reads the stats.json of the smoke statistics of E4 and E6 (full pass, with
its family_pass_check) and the family-pass tables, and writes bioinf2027/diag/family_pass_smoke.json with
each check, the number of finite values compared per table and the smoke run records it rests on.
Smoke values are never reported; only the equality is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "diag" / "family_pass_smoke.json"


def main() -> None:
    scratch = config.storage_root() / "bioinf2027" / "scratch"
    record = {"plans": "bioinf2027/plans/E4.md and E6.md, Section 16", "items": {}}
    for item in ("E4", "E6"):
        folder = scratch / f"{item}_stats_smoke"
        stats = json.loads((folder / "stats.json").read_text())
        check = stats["family_pass_check"]
        counts = {}
        for name in check["files"]:
            table = pd.read_csv(folder / "family_pass" / name)
            counts[name] = int(np.isfinite(table.select_dtypes("number").to_numpy(dtype=np.float64)).sum())
        runs = json.loads((REPO / "bioinf2027" / "runs" / f"{item}.json").read_text())["runs"]
        smoke = [{"label": r["label"], "commit": r["commit"], "returncode": r["returncode"], "finished": r["finished"]}
                 for r in runs if r["label"].startswith(f"{item.lower()}-") and "smoke" in r["label"] and ("tier" in r["label"] or r["label"].endswith(("-family", "-full")))]
        record["items"][item] = {"full_pass_stats": file_record(folder / "stats.json"),
                                 "family_pass_stats": file_record(folder / "family_pass" / "stats.json"),
                                 "family_pass_check": check, "finite_values_per_family_table": counts, "smoke_runs": smoke}
    record["passed"] = all(v["family_pass_check"]["passed"] for v in record["items"].values())
    OUT.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("family pass equal to the full pass on smoke fits: %s (%s)" % (record["passed"], config.portable(OUT)))


if __name__ == "__main__":
    main()
