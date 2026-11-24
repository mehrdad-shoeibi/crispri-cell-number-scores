"""The read status of E1b's L readings, derived from bioinf2027/results/E1/readings.csv without rerunning the statistics.

Gate G3, 2026-11-12 (bioinf2027/GATE.md, top section, item 3), and bioinf2027/plans/E1b_posthoc.md,
Section 7. readings.csv (bioinf2027/scripts/make_e1b_stats.py, reading_rows) gives an L reading for
every arm. Under bioinf2027/plans/E1_design.md, Section 21.1, item 2, the L of an arm that fails the
VCC precondition in either cell of the primary pair is reported and not read. The audited file stays
unchanged. This script writes a derived table that carries the read status, and fact sheets, figures
and tables take E1b readings from it only (guard: bioinf2027/tests/test_e1_readings_guard.py).

- Every column of readings.csv is kept except L_reading.
- L_read_status is "read" only where precondition_passed_in_both_cells is true, and otherwise the
  NOT_READ text.
- L_reading carries the computed text only where the status is "read"; elsewhere it carries NOT_READ.
- The precondition flag is checked against preconditions.csv (rebuilt label, G5 and G6, passes)
  before anything is written.

The same is done for the `read` column of bioinf2027/results/E1/primary_family.csv, which is true
for every T and L although rule 3 of Section 11 reads none of them while no arm passes the
precondition (second results review, non-blocking 5, bioinf2027/reviews/E1b_results_r2.md):
primary_family_read_status.csv keeps the identifying columns, the estimate, interval and state, and
adds `read_under_rule_3` and `read_status`.

Outputs: bioinf2027/results/E1/derived/readings_read_status.csv and readings_read_status.json;
primary_family_read_status.csv and primary_family_read_status.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import e1cells  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "bioinf2027" / "results" / "E1"
OUT = E1 / "derived"
RULE = "bioinf2027/plans/E1_design.md, Section 21.1, item 2"
NOT_READ = "not read (fails the VCC precondition in a cell of the primary pair; %s)" % RULE
READ = "read"
RULE_3 = "bioinf2027/plans/E1_design.md, Section 11, rule 3, and Section 21.1, item 2"
FAMILY_NOT_READ = "not read (the arm fails the VCC precondition in a cell of the primary pair; %s)" % RULE_3
FAMILY_NOT_ANSWERABLE = "not read (not answerable: the R5 gate fails; bioinf2027/plans/E1_design.md, Section 23.3)"
FAMILY_COLUMNS = ["cell", "screen", "statistic", "arm", "estimate", "lo", "hi", "state", "read"]


def precondition_flags(preconditions: pd.DataFrame) -> dict:
    out = {}
    for arm, rows in preconditions[(preconditions.cell.isin(e1cells.PRIMARY)) & (preconditions.label == "rebuilt")].groupby("arm"):
        out[arm] = bool(len(rows) == len(e1cells.PRIMARY) and rows["passes"].astype(bool).all())
    return out


def derive(readings: pd.DataFrame, preconditions: pd.DataFrame) -> pd.DataFrame:
    flags = precondition_flags(preconditions)
    for row in readings.itertuples():
        if row.arm not in flags or bool(row.precondition_passed_in_both_cells) != flags[row.arm]:
            raise SystemExit("readings.csv and preconditions.csv disagree on the precondition of %s on %s" % (row.arm, row.screen))
    derived = readings.drop(columns=["L_reading"]).copy()
    passed = readings["precondition_passed_in_both_cells"].astype(bool)
    derived["L_read_status"] = [READ if p else NOT_READ for p in passed]
    derived["L_reading"] = [text if p else NOT_READ for p, text in zip(passed, readings["L_reading"])]
    return derived


def derive_family(family: pd.DataFrame, preconditions: pd.DataFrame) -> pd.DataFrame:
    """The read status of every primary statistic: T and L only where the arm passes the precondition in both cells."""
    flags = precondition_flags(preconditions)
    derived = family[FAMILY_COLUMNS].copy()
    read, status = [], []
    for row in family.itertuples():
        if row.arm not in flags:
            raise SystemExit("primary_family.csv names an arm without a precondition row: %s" % row.arm)
        if row.statistic in ("T", "L"):
            ok = bool(row.read) and flags[row.arm]
            read.append(ok)
            status.append(READ if ok else FAMILY_NOT_READ)
        else:
            ok = bool(row.read) and flags[row.arm]
            read.append(ok)
            status.append(READ if ok else (FAMILY_NOT_ANSWERABLE if not bool(row.read) else FAMILY_NOT_READ))
    derived["read_under_rule_3"] = read
    derived["read_status"] = status
    return derived


def main() -> None:
    family = pd.read_csv(E1 / "primary_family.csv", float_precision="round_trip")
    family_derived = derive_family(family, pd.read_csv(E1 / "preconditions.csv", float_precision="round_trip"))
    OUT.mkdir(parents=True, exist_ok=True)
    family_derived.to_csv(OUT / "primary_family_read_status.csv", index=False)
    family_record = {"decided": "bioinf2027/GATE.md, top section (gate G3, 2026-11-12), item 3; bioinf2027/reviews/E1b_results_r2.md, non-blocking 5",
                     "rule": RULE_3, "derived_from": {"primary_family.csv": file_record(E1 / "primary_family.csv"), "preconditions.csv": file_record(E1 / "preconditions.csv")},
                     "rows": int(len(family_derived)), "read": int(family_derived["read_under_rule_3"].sum()),
                     "not_read": int((~family_derived["read_under_rule_3"]).sum()), "output": file_record(OUT / "primary_family_read_status.csv")}
    (OUT / "primary_family_read_status.json").write_text(json.dumps(strict_json(family_record), indent=1, allow_nan=False) + "\n")
    print("[read status] primary family: %d statistics, %d read, %d not read" % (family_record["rows"], family_record["read"], family_record["not_read"]), flush=True)
    readings = pd.read_csv(E1 / "readings.csv")
    preconditions = pd.read_csv(E1 / "preconditions.csv", float_precision="round_trip")
    derived = derive(readings, preconditions)
    OUT.mkdir(parents=True, exist_ok=True)
    derived.to_csv(OUT / "readings_read_status.csv", index=False)
    record = {"decided": "bioinf2027/GATE.md, top section (gate G3, 2026-11-12), item 3", "plan": "bioinf2027/plans/E1b_posthoc.md, Section 7",
              "rule": RULE, "derived_from": {"readings.csv": file_record(E1 / "readings.csv"), "preconditions.csv": file_record(E1 / "preconditions.csv")},
              "rows": int(len(derived)), "L_read": int((derived["L_read_status"] == READ).sum()), "L_not_read": int((derived["L_read_status"] != READ).sum()),
              "output": file_record(OUT / "readings_read_status.csv")}
    (OUT / "readings_read_status.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[read status] %d rows: %d L read, %d not read" % (record["rows"], record["L_read"], record["L_not_read"]), flush=True)


if __name__ == "__main__":
    main()
