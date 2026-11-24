"""Table S-c: the external values of the VCC-trained heads, values without an interpretation (figure and table plan,
supplement). E1b's T and L of z, x, m and [x; m] in G5 and G6 on both screens with the read status of
bioinf2027/results/E1/derived/primary_family_read_status.csv (every row "not read"), and E2's test (ii), T and L of
scGPT, Stack and PCA-50, with the columns `read` and `why_not_read` of bioinf2027/results/E2/stats/family.csv. Stack's
values carry their marker. Usage, from the repository root: python bioinf2027/scripts/figures/tableS_c_external_values.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

R = tablelib.REPO / "bioinf2027" / "results"
E1_FAMILY = R / "E1" / "primary_family.csv"
E1_STATUS = R / "E1" / "derived" / "primary_family_read_status.csv"
E2_FAMILY = R / "E2" / "stats" / "family.csv"


def f(v: float) -> str:
    return ("%+.3f" % v).replace("-", "−")


def main() -> None:
    status = pd.read_csv(E1_STATUS, float_precision="round_trip")
    family = pd.read_csv(E1_FAMILY, float_precision="round_trip")
    keys = ["cell", "screen", "statistic", "arm"]
    if not (family[keys].values == status[keys].values).all():
        raise SystemExit("primary_family.csv and primary_family_read_status.csv list other rows")
    e1 = status[status["statistic"].isin(["T", "L"])]
    rows = [{"Item": "E1b", "Encoder or feature set": "Geneformer z" if r.arm == "z" else r.arm, "Configuration": r.cell, "Screen": r.screen,
             "Statistic": r.statistic, "Estimate": f(r.estimate), "Interval": "[%s, %s]" % (f(r.lo), f(r.hi)), "Holm state": r.state,
             "Read status": r.read_status, "Markers": ""} for r in e1.itertuples()]
    e2 = pd.read_csv(E2_FAMILY, float_precision="round_trip")
    e2 = e2[e2["test"] == "ii"]
    for r in e2.itertuples():
        rows.append({"Item": "E2 (ii)", "Encoder or feature set": r.encoder, "Configuration": r.cell, "Screen": r.screen, "Statistic": r.statistic,
                     "Estimate": f(r.estimate), "Interval": "[%s, %s]" % (f(r.lo), f(r.hi)), "Holm state": r.state,
                     "Read status": ("read" if bool(r.read) else "not read: %s" % r.why_not_read), "Markers": "" if pd.isna(r.markers) else r.markers})
    frame = pd.DataFrame(rows)
    tablelib.save("tableS_c_external_values", frame, [E1_FAMILY, E1_STATUS, E2_FAMILY], Path(__file__).resolve(), align="llllllllll",
                  extra={"sources": {"E1b values and read status": tablelib.rel(E1_STATUS), "E1b family": tablelib.rel(E1_FAMILY),
                                     "E2 test (ii)": tablelib.rel(E2_FAMILY)},
                         "read_status_as_saved": sorted(set(frame["Read status"])),
                         "families": "E1b's first family of 60 (Holm 0.05; T and L shown) and E2's family of 30 (Holm 0.05; the 24 statistics of test (ii) shown)"})


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_c_external_values.py")
    main()
