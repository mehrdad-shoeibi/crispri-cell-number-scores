"""Table S-j: the closing round of missing analyses (the author's answer to the stop of 2026-11-17 17:10, item 9;
bioinf2027/GATE.md; plan bioinf2027/plans/missing_analyses_closing.md), printed from
bioinf2027/results/missing_analyses_closing/ as saved; post hoc, unadjusted, diagnostics without a reading.
Four tables: (1) S1 supplement, missing 4, the all-cells label within (target gene, guide) on the rows that enter R2's
simulation, beside the saved terms; (2) S1 supplement, missing 6, constructs excluded at the 27-cell floor beside part
(c); (3) E5 supplement 2, analysis 8, Holm within each version's 12 for the joint comparator; (4) the limitations.
Usage, from the repository root: python bioinf2027/scripts/figures/tableS_j_closing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

FOLDER = tablelib.REPO / "bioinf2027" / "results" / "missing_analyses_closing"
RECORD = FOLDER / "closing.json"
DEPMAP = "DepMap 24Q4, not the current release"


def f(v: float) -> str:
    return ("%+.3f" % v).replace("-", "−")


def iv(lo: float, hi: float) -> str:
    return "[%s, %s]" % (f(lo), f(hi))


def c(n) -> str:
    return "{:,}".format(int(n))


def main() -> None:
    script = Path(__file__).resolve()
    status = {"status": "post hoc, unadjusted, not part of any family, no reading", "record": tablelib.rel(RECORD)}

    s1 = FOLDER / "s1_within_on_simulated_rows.csv"
    t = pd.read_csv(s1, float_precision="round_trip")
    frame = pd.DataFrame({
        "Configuration": t["cell"], "Split": t["split"].map({"train": "training", "val": "validation", "test": "test", "all": "all"}),
        "Within (gene, guide), simulated rows": [f(v) for v in t["estimate"]], "Interval": [iv(a, b) for a, b in zip(t["lo"], t["hi"])],
        "Rows used / simulated rows": ["%s / %s" % (c(a), c(b)) for a, b in zip(t["observed_rows"], t["simulated_rows"])],
        "Target genes drawn": t["target_genes_drawn"].map(c),
        "Saved, all single-guide rows": ["%s %s (%s rows)" % (f(a), iv(b, d), c(e)) for a, b, d, e in
                                         zip(t["saved_observed_all_single_guide_rows"], t["saved_observed_lo"], t["saved_observed_hi"], t["saved_observed_rows"])],
        "Saved [part from the estimator], all simulated rows": ["%s %s" % (f(a), iv(b, d)) for a, b, d in zip(t["saved_mechanical"], t["saved_mechanical_lo"], t["saved_mechanical_hi"])],
        "Saved line": t["saved_line"].str.replace("bioinf2027/results/S1_supplement/", "", regex=False)})
    tablelib.save("tableS_j_closing_1_s1_within", frame, [s1, RECORD], script, align="llllrrlll",
                  extra={**status, "note": "validation rows fall below 50 target genes and are descriptive; the paired [remainder] on the same rows is not "
                                           "formed (limitations, table 4)", "term": "[part from the estimator] stands for the report's mechanical part"})

    s6 = FOLDER / "s1_floor_exclusions.csv"
    t = pd.read_csv(s6, float_precision="round_trip")
    frame = pd.DataFrame({"Screen": t["screen"], "Constructs after the join": t["constructs_after_join"].map(c),
                          "With at least 27 cells": t["with_at_least_27_cells"].map(c), "Excluded, too few cells": t["excluded_too_few_cells"].map(c),
                          "Target not a gene of the file": t["excluded_target_not_in_var"].map(c), "Both": t["excluded_both"].map(c),
                          "Part (c), all-cells label against n": ["%s %s" % (f(a), iv(b, d)) for a, b, d in zip(t["part_c_all_cells_association"], t["part_c_lo"], t["part_c_hi"])],
                          "Lines": ["%s; %s" % (a.replace("bioinf2027/diag/", ""), b.replace("bioinf2027/results/S1_supplement/", "")) for a, b in zip(t["d3_line"], t["part_c_line"])]})
    tablelib.save("tableS_j_closing_2_s1_floor", frame, [s6, RECORD], script, align="lrrrrrll", extra={**status, "computation": "none (D3's counts as saved)"})

    h = FOLDER / "e5_joint_holm.csv"
    t = pd.read_csv(h, float_precision="round_trip")
    frame = pd.DataFrame({"Version": t["version"].map({"a": "a (F_mean)", "b": "b (F_mean, F_K562)"}), "Vector": t["embedding"], "Setting": t["setting"],
                          "Genes": t["n_test_genes"].map(c), "Smaller difference": [f(v) for v in t["iu_estimate"]], "p": t["iu_p"].map(lambda v: "%.4f" % v),
                          "Holm-adjusted p (within the version's 12)": t["holm_adjusted_p"].map(lambda v: "%.4f" % v),
                          "Holm at 0.05": t["holm_rejected_at_0.05"].map({True: "kept", False: "not kept"}),
                          "Both intervals above zero (gene clusters)": t["both_resolved_above_zero"].map({True: "yes", False: "no"}),
                          "Also with prefix clusters": t["prefix_both_resolved_above_zero"].map({True: "yes", False: "no"}),
                          "DepMap release": DEPMAP, "Line": t["line"].str.replace("bioinf2027/results/E5/supplement_2/", "", regex=False)})
    tablelib.save("tableS_j_closing_3_e5_holm", frame, [h, RECORD], script, align="lllrrrrllll",
                  extra={**status, "depmap_release": DEPMAP, "note": "descriptive; Holm is not a family of the plan"})

    lim = FOLDER / "limitations.csv"
    t = pd.read_csv(lim)
    frame = pd.DataFrame({"Analysis": t["analysis"], "Quantity": t["quantity"], "Status": t["status"], "Why": t["why"], "Files inspected": t["files_inspected"]})
    tablelib.save("tableS_j_closing_4_limitations", frame, [lim, RECORD], script, align="lllll", extra=status)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_j_closing.py")
    main()
