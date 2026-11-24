"""Table S-d: E2 (d), the external size-matched null, and its follow-ups, values without an explanation (figure and table
plan, supplement; G7, item 6: no statement about why the external values are negative). Two parts, one row per encoder,
configuration (G5, G6) and screen:
- part 1, size-fixed (rebuilt) training label: T, T_N0 = rho(N0x) - rho(C1), e - N0x (bioinf2027/results/E2/null_external/null_external.csv),
  and the follow-ups (2), (3a), (3b) and (3c) (bioinf2027/results/E2/null_external_followup/saved.csv);
- part 2, all-cells (existing) training label: T, T_N0, e - N0x (bioinf2027/results/E2/null_external_followup/existing_label/null_external.csv),
  and analyses 1 and 3 of the review of (4): (2), (3a), (3b) (existing_label/saved.csv);
- the external positive control on the all-cells external label: null - C1 and e - C1 (bioinf2027/results/S1_supplement/external_control.csv).
Part 1 also gives per configuration and screen the cell count against the size-fixed external label, rho(n_t, y)
(bioinf2027/results/E2/null_external_followup/saved.csv, analysis (1)). Stack's values carry their marker.
Usage, from the repository root: python bioinf2027/scripts/figures/tableS_d_e2d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

R = tablelib.REPO / "bioinf2027" / "results"
D = R / "E2" / "null_external" / "null_external.csv"
D4 = R / "E2" / "null_external_followup" / "existing_label" / "null_external.csv"
S5 = R / "E2" / "null_external_followup" / "existing_label" / "saved.csv"
S13 = R / "E2" / "null_external_followup" / "saved.csv"
S1D = R / "S1_supplement" / "external_control.csv"
ENCODERS = ("geneformer", "scgpt", "stack", "pca50")


def f(v: float) -> str:
    return ("%+.3f" % v).replace("-", "−")


def value(frame: pd.DataFrame, **keys) -> str:
    hit = frame
    for k, v in keys.items():
        hit = hit[hit[k] == v]
    if len(hit) != 1:
        raise SystemExit("%d rows for %s" % (len(hit), keys))
    r = hit.iloc[0]
    return "%s [%s, %s]" % (f(r.estimate), f(r.lo), f(r.hi))


def main() -> None:
    d, d4, s5, s13, s1 = (pd.read_csv(p, float_precision="round_trip") for p in (D, D4, S5, S13, S1D))
    first, second = [], []
    for cell in ("G5", "G6"):
        for screen in ("RPE1", "K562-essential"):
            first.append({"Encoder": "cell count", "Configuration": cell, "Screen": screen,
                          "rho(n_t, size-fixed external label)": value(s13, analysis="(1)", cell=cell, screen=screen, quantity="rho(n_t, y)")})
            for e in ENCODERS:
                null = "N0x[%s]" % e
                e_name = "z" if e == "geneformer" else e
                c1 = "C1" if e == "geneformer" else "C1[%s]" % e
                marker = d[(d.encoder == e)]["markers"].dropna()
                marker = marker.iloc[0] if len(marker) else ""
                key = {"Encoder": e, "Configuration": cell, "Screen": screen}
                first.append({**key,
                              "T": value(d, encoder=e, cell=cell, screen=screen, quantity="T"),
                              "T_N0": value(d, encoder=e, cell=cell, screen=screen, quantity="T_N0"),
                              "e − N0x": value(d, encoder=e, cell=cell, screen=screen, quantity="e - N0x"),
                              "rho(null's prediction, n_t)": value(s13, analysis="(2)", cell=cell, screen=screen, quantity="rho(prediction of %s, n_t)" % null),
                              "rho(e's prediction, n_t)": value(s13, analysis="(3a)", cell=cell, screen=screen, quantity="rho(prediction of %s, n_t)" % e_name),
                              "e's partial given n_t": value(s13, analysis="(3b)", cell=cell, screen=screen, quantity="partial rho(%s, y | n_t)" % e_name),
                              "rho(e, y), mean over fifths of n_t": value(s13, analysis="(3c)", cell=cell, screen=screen,
                                                                         quantity="rho(%s, y), mean of the five fifths of n_t" % e_name),
                              "Markers": marker})
                second.append({**key,
                               "T": value(d4, encoder=e, cell=cell, screen=screen, quantity="T"),
                               "T_N0": value(d4, encoder=e, cell=cell, screen=screen, quantity="T_N0"),
                               "e − N0x": value(d4, encoder=e, cell=cell, screen=screen, quantity="e - N0x"),
                               "rho(null's prediction, n_t)": value(s5, analysis="(2)", encoder=e, cell=cell, screen=screen, quantity="rho(prediction of %s, n_t)" % null),
                               "rho(e's prediction, n_t)": value(s5, analysis="(3a)", encoder=e, cell=cell, screen=screen, quantity="rho(prediction of %s, n_t)" % e_name),
                               "e's partial given n_t": value(s5, analysis="(3b)", encoder=e, cell=cell, screen=screen, quantity="partial rho(%s, y | n_t)" % e_name),
                               "All-cells external label: null − C1": value(s1, encoder=e, cell=cell, screen=screen,
                                                                           quantity="rho(%s, all-cells label) - rho(%s, all-cells label)" % (null, c1)),
                               "All-cells external label: e − C1": value(s1, encoder=e, cell=cell, screen=screen,
                                                                        quantity="rho(%s, all-cells label) - rho(%s, all-cells label)" % (e_name, c1)),
                               "Markers": marker})
    status = ("post hoc, unadjusted, not part of any family; no statement about why the external values are negative (G7, item 6); "
              "e − C1 on the all-cells external label carries no reading (bioinf2027/plans/S1_supplement.md, Section 4)")
    intervals = "E1b's joint external target-gene draws (seeds 20260901 and 20260902), head seeds averaged inside each replicate"
    not_shown = "(1) M against |T|min and (3c) per fifth of n_t, in bioinf2027/results/E2/null_external_followup/saved.csv"
    script = Path(__file__).resolve()
    one = pd.DataFrame(first).fillna("")
    tablelib.save("tableS_d_e2d", one, [D, S13], script, align="l" * len(one.columns),
                  extra={"part": "1 of 2, heads trained on the size-fixed (rebuilt) VCC label; external label at 27 against 788 cells",
                         "status": status, "intervals": intervals, "not_shown": not_shown})
    two = pd.DataFrame(second).fillna("")
    tablelib.save("tableS_d_e2d_all_cells", two, [D, D4, S5, S1D], script, align="l" * len(two.columns),
                  extra={"part": "2 of 2, heads trained on the all-cells (existing) VCC label; external label at 27 against 788 cells except "
                                 "the last two value columns (all eligible perturbed cells against 788 pooled controls)",
                         "status": status, "intervals": intervals})


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_d_e2d.py")
    main()
