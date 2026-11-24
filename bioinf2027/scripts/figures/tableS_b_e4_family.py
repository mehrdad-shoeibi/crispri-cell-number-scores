"""Table S-b: E4, feature-side cell number, the 16 family statistics F from E4's full pass over tiers 1 to 3, with each row's
in-distribution check under matched features and its read status (figure and table plan, supplement; values without an
interpretation). From bioinf2027/results/E4/family.csv (columns `precondition_matched_*` and `read`) and the statements of
bioinf2027/results/E4/readings.csv, which begin "computed and reported, not read". The full pass reproduces the family values
of the pass of gate G2 (bioinf2027/results/E4/stats.json, key `family_pass_check`); the script stops if that check did not pass.
Usage, from the repository root: python bioinf2027/scripts/figures/tableS_b_e4_family.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

E4 = tablelib.REPO / "bioinf2027" / "results" / "E4"
FAMILY, READINGS, STATS = E4 / "family.csv", E4 / "readings.csv", E4 / "stats.json"
NOT_READ = "computed and reported, not read"


def main() -> None:
    stats = json.loads(STATS.read_text())
    if stats.get("pass") != "full" or not stats["family_pass_check"]["passed"]:
        raise SystemExit("bioinf2027/results/E4/stats.json is not a full pass whose family values equal the family pass")
    family = pd.read_csv(FAMILY, float_precision="round_trip")
    readings = pd.read_csv(READINGS)
    statement = {(r.screen, r.arm): r.statement for r in readings.itertuples()}
    if not all(str(s).startswith(NOT_READ) for s in statement.values()):
        raise SystemExit("a statement of readings.csv does not begin %r" % NOT_READ)
    if family["precondition_matched_passed"].astype(bool).any():
        raise SystemExit("an arm passes the in-distribution check under matched features; the status text below would be wrong")
    f = lambda v: ("%+.3f" % v).replace("-", "−")  # noqa: E731
    frame = pd.DataFrame({"Configuration": family["cell"], "Screen": family["screen"], "Feature set": family["arm"], "Statistic": family["statistic"],
                          "Estimate": family["estimate"].map(f), "Interval": ["[%s, %s]" % (f(a), f(b)) for a, b in zip(family["lo"], family["hi"])],
                          "p": family["p"].map(lambda v: "%.4f" % v), "Holm threshold": family["holm_threshold"].map(lambda v: "%.4f" % v),
                          "State": family["state"],
                          "In-distribution check, matched features": ["%s (lower bound %s): not passed" % (f(e), f(lo)) for e, lo in
                                                                      zip(family["precondition_matched_estimate"], family["precondition_matched_lo"])],
                          "Read": family["read"],
                          "Status": [statement[(s, a)] for s, a in zip(family["screen"], family["arm"])]})
    tablelib.save("tableS_b_e4_family", frame, [FAMILY, READINGS, STATS], Path(__file__).resolve(), align="llllrlrrllll",
                  extra={"sources": {"values": tablelib.rel(FAMILY), "status": tablelib.rel(READINGS), "equality with the family pass": tablelib.rel(STATS)},
                         "status_of_every_row": NOT_READ,
                         "family": "16 statistics, Holm at family-wise 0.05; E4's full pass over tiers 1 to 3, whose family values equal those of the pass of gate G2",
                         "in_distribution_check": "VCC-test R-squared minus that of the Gaussian control under matched features, with its lower bound (family.csv, "
                                                  "columns precondition_matched_estimate and precondition_matched_lo); every arm fails in both configurations"})


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_b_e4_family.py")
    main()
