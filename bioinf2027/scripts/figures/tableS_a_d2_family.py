"""Table S-a: D2, the target gene's own token removed, the 18 family statistics with their read status (figure and table
plan, supplement; values without an interpretation). From bioinf2027/results/D2/family.csv and the status of every row in
bioinf2027/results/D2/supplement/family_read_status.csv ("not read" or "read: ..."), printed as saved.
Usage, from the repository root: python bioinf2027/scripts/figures/tableS_a_d2_family.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

D2 = tablelib.REPO / "bioinf2027" / "results" / "D2"
FAMILY, STATUS = D2 / "family.csv", D2 / "supplement" / "family_read_status.csv"


def main() -> None:
    family = pd.read_csv(FAMILY, float_precision="round_trip")
    status = pd.read_csv(STATUS, float_precision="round_trip")
    keys = ["cell", "scope", "statistic"]
    if not (family[keys].values == status[keys].values).all():
        raise SystemExit("family.csv and family_read_status.csv list other rows")
    f = lambda v: ("%+.3f" % v).replace("-", "−")  # noqa: E731
    frame = pd.DataFrame({"Configuration": family["cell"], "Scope": family["scope"], "Statistic": family["statistic"],
                          "Estimate": family["estimate"].map(f), "Interval": ["[%s, %s]" % (f(a), f(b)) for a, b in zip(family["lo"], family["hi"])],
                          "p": family["p"].map(lambda v: "%.4f" % v), "Holm threshold": family["holm_threshold"].map(lambda v: "%.4f" % v),
                          "State": family["state"], "Read status": status["read_status"]})
    tablelib.save("tableS_a_d2_family", frame, [FAMILY, STATUS], Path(__file__).resolve(), align="lllrlrrll",
                  extra={"sources": {"values": tablelib.rel(FAMILY), "read status": tablelib.rel(STATUS)},
                         "family": "18 statistics, Holm at family-wise 0.05; no statistic is interpreted beyond its read status",
                         "read_status_as_saved": sorted(set(status["read_status"].astype(str)))})


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_a_d2_family.py")
    main()
