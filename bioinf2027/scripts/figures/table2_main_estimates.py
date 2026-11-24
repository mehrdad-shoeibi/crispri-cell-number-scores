"""Table 2: main estimates with intervals (the stop of 2026-11-17 06:50, figure and table plan, item 6: one compact table
in the main text with the columns quantity, estimate, interval, N and scope; the full version with file and line is a
supplement table). The rows are the author's (answer to the stop of 2026-11-17 17:10, items 4 and 5; bioinf2027/GATE.md):
16 rows in the main table, S1 (4), S2 (5), between the screens on the all-cells label (1), S3 (1), S4 (2), S5 (3); the
submitted configuration (+0.1645), the row beside DepMap's mean gene effect and the Replogle paired differences in the
full version only. Two settings of one quantity share a row as "first; second". Every value is read from the cited
line of a saved file; each row checks the identifying columns of its line before using it, and a range over several
lines lists every line.
Usage, from the repository root: python bioinf2027/scripts/figures/table2_main_estimates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

REPO = tablelib.REPO
B = REPO / "bioinf2027"
F = {"R2": B / "endpoints/readouts/R2_between.csv", "S1p": B / "results/S1_supplement/vcc_paired.csv",
     "S1x": B / "results/S1_supplement/external_association.csv", "S1c": B / "results/S1_supplement/external_control.csv",
     "S1l": B / "results/S1_supplement/external_labels.csv", "posthoc": B / "results/E1/posthoc/posthoc_vcc.csv",
     "pre": B / "results/E1/preconditions.csv", "vcc_test": B / "results/E1/vcc_test.csv", "e2vcc": B / "results/E2/stats/vcc.csv",
     "D1": B / "diag/D1_label_vs_own_delta_external.csv", "E6": B / "results/E6/family.csv", "E6s": B / "results/E6/secondary.csv",
     "E2e6": B / "results/E2/stats/e6.csv", "E5": B / "results/E5/family/family.csv", "dep": B / "results/E5/supplement/depmap.csv",
     "s2c": B / "results/E5/supplement_2/comparisons.csv", "same": B / "results/E5/supplement_2/same_rows.csv",
     "D3": B / "diag/D3_cell_counts.csv"}
TABLES = {k: pd.read_csv(v, float_precision="round_trip") for k, v in F.items()}
DEPMAP = "DepMap 24Q4, not the current release"


def line(key: str, n: int, **expect) -> pd.Series:
    row = TABLES[key].iloc[n - 2]
    for column, value in expect.items():
        if str(row[column]) != str(value):
            raise SystemExit("%s:%d: %s is %r, expected %r" % (tablelib.rel(F[key]), n, column, row[column], value))
    return row


def s(v: float) -> str:
    return ("%+.3f" % v).replace("-", "−")


def iv(lo: float, hi: float) -> str:
    return "[%s, %s]" % (s(lo), s(hi))


def c(n) -> str:
    return "{:,}".format(int(n))


def where(key: str, lines) -> str:
    return "%s:%s" % (tablelib.rel(F[key]), ", ".join(str(n) for n in lines))


def rows_n(r) -> str:
    return "%s rows, %s genes" % (c(r.n_rows), c(r.n_target_genes))


rows = []


def add(claim, quantity, estimate, interval, n, scope, source, full_only=False):
    rows.append({"Claim": claim, "Quantity": quantity, "Estimate": estimate, "Interval": interval, "N": n, "Scope": scope,
                 "File and line": source, "full_only": full_only})


# S1: the label against cells per row or construct, VCC on the 27-cell (G5) and 10-cell (G6) rows
a5, a6 = line("R2", 25, cell="G5", split="all", label="existing label against pool size, same rows"), line("R2", 33, cell="G6", split="all", label="existing label against pool size, same rows")
add("S1", "VCC all-cells label against perturbed cells per row", "%s; %s" % (s(a5.spearman), s(a6.spearman)),
    "%s; %s" % (iv(a5.spearman_lo, a5.spearman_hi), iv(a6.spearman_lo, a6.spearman_hi)), "%s; %s" % (rows_n(a5), rows_n(a6)),
    "Spearman; 27-cell rows (G5); 10-cell rows (G6); all splits", where("R2", [25, 33]))
b5, b6 = line("R2", 24, cell="G5", split="all", label="rebuilt label against pool size"), line("R2", 32, cell="G6", split="all", label="rebuilt label against pool size")
add("S1", "VCC size-fixed label against perturbed cells per row, same rows", "%s; %s" % (s(b5.spearman), s(b6.spearman)),
    "%s; %s" % (iv(b5.spearman_lo, b5.spearman_hi), iv(b6.spearman_lo, b6.spearman_hi)), "%s; %s" % (rows_n(b5), rows_n(b6)),
    "Spearman; 27 cells (G5); 10 cells (G6); 700 controls of the row's batch", where("R2", [24, 32]))
p5, p6 = line("S1p", 5, cell="G5", split="all"), line("S1p", 9, cell="G6", split="all")
for p, a in ((p5, a5), (p6, a6)):
    if abs(p["first"] - a.spearman) > 1e-12:
        raise SystemExit("vcc_paired.csv: the first term is not R2's all-cells Spearman")
add("S1", "Their paired difference", "%s; %s" % (s(p5.estimate), s(p6.estimate)), "%s; %s" % (iv(p5.lo, p5.hi), iv(p6.lo, p6.hi)),
    "%s rows; %s rows" % (c(p5.rows), c(p6.rows)), "all-cells minus size-fixed, same rows; G5; G6; post hoc", where("S1p", [5, 9]))
x1, x2 = line("S1x", 2, screen="RPE1", quantity="rho(all cells against 788 pooled controls label, n)"), line("S1x", 10, screen="K562-essential", quantity="rho(all cells against 788 pooled controls label, n)")
add("S1", "Replogle all-cells label against perturbed cells per construct", "%s; %s" % (s(x1.estimate), s(x2.estimate)),
    "%s; %s" % (iv(x1.lo, x1.hi), iv(x2.lo, x2.hi)), "%s; %s constructs" % (c(x1.constructs), c(x2.constructs)),
    "Spearman; RPE1; K562-essential; all eligible cells against 788 pooled controls; post hoc", where("S1x", [2, 10]))
d1, d2 = line("S1x", 6, screen="RPE1"), line("S1x", 14, screen="K562-essential")
for d, a, z in ((d1, x1, line("S1x", 4, screen="RPE1", quantity="rho(27 against 788 label, n)")),
                (d2, x2, line("S1x", 12, screen="K562-essential", quantity="rho(27 against 788 label, n)"))):
    if abs(d.estimate - (a.estimate - z.estimate)) > 1e-9:
        raise SystemExit("external_association.csv: a paired difference is not all-cells minus 27/788")
add("S1", "Replogle paired difference, all-cells minus size-fixed label (27 against 788)", "%s; %s" % (s(d1.estimate), s(d2.estimate)),
    "%s; %s" % (iv(d1.lo, d1.hi), iv(d2.lo, d2.hi)), "%s; %s constructs" % (c(d1.constructs), c(d2.constructs)),
    "same constructs; RPE1; K562-essential; post hoc", where("S1x", [6, 14]), full_only=True)

# S2: VCC test rows, G6, forest against forest (the author's item 4)
e, f, g = (line("e2vcc", 155, encoder="geneformer", cell="G6", head="forest", label="existing", quantity="e - C1"),
           line("e2vcc", 156, encoder="geneformer", cell="G6", head="forest", label="rebuilt", quantity="e - C1"),
           line("e2vcc", 157, encoder="geneformer", cell="G6", head="forest", label="existing minus rebuilt", quantity="P-d"))
pre = line("pre", 98, cell="G6", label="rebuilt", arm="z")
if abs(pre.r2_difference_estimate - f.r2) > 1e-12:
    raise SystemExit("preconditions.csv:98 and vcc.csv:156 differ")
vn = "%s rows, %s genes" % (c(e.n_test_rows), c(e.n_test_genes))
add("S2", "Embedding over the Gaussian control, all-cells label", s(e.r2), iv(e.r2_lo, e.r2_hi), vn,
    "VCC test R², G6; Geneformer z minus C1, both forest; post hoc", where("e2vcc", [155]))
r = line("posthoc", 14, cell="G6", quantity="a0")
add("S2", "The same in the submitted configuration", ("%+.4f" % r.r2_difference).replace("-", "−"), iv(r.r2_lo, r.r2_hi),
    "%s rows, %s genes" % (c(r.n_test_rows), c(r.n_test_genes)), "VCC test R², G6; z (forest) minus C1 (boosting); post hoc",
    where("posthoc", [14]), full_only=True)
add("S2", "Embedding over the Gaussian control, size-fixed label", s(f.r2), iv(f.r2_lo, f.r2_hi), vn,
    "VCC test R², G6, 10 cells; z minus C1, both forest", where("e2vcc", [156]) + "; " + where("pre", [98]))
add("S2", "Paired drop from the all-cells to the size-fixed label", s(g.r2), iv(g.r2_lo, g.r2_hi), vn,
    "VCC test R², G6; same rows; post hoc", where("e2vcc", [157]))
r = line("vcc_test", 642, cell="G6", label="existing", representation="B0", head="forest")
add("S2", "Cell count alone, all-cells label", s(r.row_r2), iv(r.row_r2_lo, r.row_r2_hi), "%s rows, %s genes" % (c(r.n_test_rows), c(r.n_test_genes)),
    "VCC test R², G6, forest", where("vcc_test", [642]))
null_lines = (170, 534, 876, 1236)
nulls = [line("e2vcc", n, cell="G6", label="existing", head="forest", quantity="N0 - C1") for n in null_lines]
if sorted(x.encoder for x in nulls) != ["geneformer", "pca50", "scgpt", "stack"]:
    raise SystemExit("the null rows are not the four encoders")
add("S2", "Size-matched null over the Gaussian control, all-cells label", "%s to %s" % (s(min(x.r2 for x in nulls)), s(max(x.r2 for x in nulls))),
    "lower ends %s to %s" % (s(min(x.r2_lo for x in nulls)), s(max(x.r2_lo for x in nulls))), vn,
    "VCC test R², G6, forest; Geneformer, scGPT, Stack, PCA-50", where("e2vcc", null_lines))

# Between the screens on the all-cells label: S1 supplement, part (d), 16 combinations
ctrl = TABLES["S1c"].assign(line=TABLES["S1c"].index + 2)
nx = ctrl[ctrl.quantity.str.match(r"^rho\(N0x\[\w+\], all-cells label\) - rho\(C1(\[\w+\])?, all-cells label\)$")]
if len(nx) != 16 or sorted(set(nx.encoder)) != ["geneformer", "pca50", "scgpt", "stack"] or sorted(set(nx.cell)) != ["G5", "G6"]:
    raise SystemExit("external_control.csv: the null rows are not 4 encoders x 2 cells x 2 screens")
labels = TABLES["S1l"].groupby("screen").size()
add("S1", "Size-matched null over the Gaussian control, Replogle all-cells label", "%s to %s" % (s(nx.estimate.min()), s(nx.estimate.max())),
    "lower ends %s to %s" % (s(nx.lo.min()), s(nx.lo.max())), "%s; %s constructs" % (c(labels["RPE1"]), c(labels["K562-essential"])),
    "Spearman; heads trained on VCC's all-cells label; 4 encoders × G5, G6 × 2 screens; Stack marked; post hoc",
    where("S1c", list(nx.line)) + "; " + tablelib.rel(F["S1l"]) + " (rows per screen)")

# S3: the size-fixed label against the target's own shift
o1, o2 = (line("D1", 3, screen="RPE1", predictor="negated_own_delta_native", outcome="y_primary", subset="all"),
          line("D1", 7, screen="K562-essential", predictor="negated_own_delta_native", outcome="y_primary", subset="all"))
add("S3", "Replogle size-fixed label against the target's own shift", "%s; %s" % (s(o1.spearman), s(o2.spearman)),
    "%s; %s" % (iv(o1.spearman_lo, o1.spearman_hi), iv(o2.spearman_lo, o2.spearman_hi)),
    "%s; %s constructs" % (c(o1.n_rows), c(o2.n_rows)), "Spearman; RPE1; K562-essential; 27 against 788", where("D1", [3, 7]))

# S4: between the Replogle screens
t1, t2 = line("E6", 2, direction="RPE1 to K562-essential", statistic="T", arm="z"), line("E6", 14, direction="K562-essential to RPE1", statistic="T", arm="z")
d3 = TABLES["D3"].set_index("screen")
add("S4", "Embedding over the Gaussian control between the screens", "%s; %s" % (s(t1.estimate), s(t2.estimate)),
    "%s; %s" % (iv(t1.lo, t1.hi), iv(t2.lo, t2.hi)), "%s; %s target-screen constructs" % (c(d3.loc["K562-essential", "n_eligible"]), c(d3.loc["RPE1", "n_eligible"])),
    "external Spearman; RPE1 → K562-essential; K562-essential → RPE1; Geneformer z; gene-disjoint folds; Holm over 24",
    where("E6", [2, 14]) + "; " + where("D3", [5, 6]))
sec = TABLES["E6s"].assign(line=TABLES["E6s"].index + 2)
gap6 = sec[sec.contrast.str.match(r"^(z|x|m|\[x; m\]) minus the target screen's raw B[12]$")]
e2e6 = TABLES["E2e6"].assign(line=TABLES["E2e6"].index + 2)
gap2 = e2e6[e2e6.encoder.isin(["scgpt", "stack", "pca50"]) & e2e6.quantity.str.match(r"^e minus the target screen's raw B[12]$")]
if len(gap6) != 16 or len(gap2) != 12:
    raise SystemExit("the distances below B1 and B2 are not 16 and 12 rows")
gaps = pd.concat([gap6[["estimate", "lo", "hi"]], gap2[["estimate", "lo", "hi"]]])
add("S4", "Representations minus the target screen's untrained control mean (B1) or own shift (B2)", "%s to %s" % (s(gaps.estimate.min()), s(gaps.estimate.max())),
    "upper ends %s to %s" % (s(gaps.hi.min()), s(gaps.hi.max())), "%s; %s target-screen constructs" % (c(d3.loc["K562-essential", "n_eligible"]), c(d3.loc["RPE1", "n_eligible"])),
    "external Spearman; z, x, m, [x; m], scGPT, Stack, PCA-50 × 2 directions × B1, B2; Stack marked; unadjusted",
    where("E6s", list(gap6.line)) + "; " + where("E2e6", list(gap2.line)))

# S5: the identity task, endpoint (b)
fam = TABLES["E5"].assign(line=TABLES["E5"].index + 2)
fam = fam[fam.endpoint == "b"]
if len(fam) != 12:
    raise SystemExit("family.csv: endpoint (b) does not have 12 rows")
add("S5", "Identity vectors over random and shuffled vectors (smaller difference)", "%s to %s" % (s(fam.iu_estimate.min()), s(fam.iu_estimate.max())),
    "each p %s" % ("%.4f" % fam.iu_p.max()), "%s to %s genes" % (c(fam.n_test_genes.min()), c(fam.n_test_genes.max())),
    "Spearman, endpoint (b); 3 vectors × 4 settings; Holm over 24", where("E5", list(fam.line)))
dep = TABLES["dep"].assign(line=TABLES["dep"].index + 2)
dep = dep[dep.item.str.startswith("analysis 2a")]
pos = dep[dep.both_resolved_above_zero]
add("S5", "The same beside DepMap's mean gene effect", "%s to %s in %d of %d" % (s(pos.iu_estimate.min()), s(pos.iu_estimate.max()), len(pos), len(dep)), "",
    "%s to %s genes" % (c(dep.n_test_genes.min()), c(dep.n_test_genes.max())), "%s; post hoc, unadjusted" % DEPMAP, where("dep", list(dep.line)), full_only=True)
two = TABLES["s2c"].assign(line=TABLES["s2c"].index + 2)
two = two[two.analysis == "(2)"]
if len(two) != 24:
    raise SystemExit("comparisons.csv: analysis (2) does not have 24 rows")
add("S5", "DepMap's score over each vector, same rows", "%s to %s" % (s(two.estimate.min()), s(two.estimate.max())),
    "lower ends %s to %s" % (s(two.lo.min()), s(two.lo.max())), "%s to %s genes" % (c(two.n_test_genes.min()), c(two.n_test_genes.max())),
    "Spearman; 24 combinations (mean, and mean with the K562 value, × 3 vectors × 4 settings); %s; post hoc, unadjusted" % DEPMAP, where("s2c", list(two.line)))
same = TABLES["same"].assign(line=TABLES["same"].index + 2)
joint = same[(same.version == "a") & (same.comparator == "J")]
held = joint[joint.both_resolved_above_zero]
failed = joint[~joint.both_resolved_above_zero]
if len(joint) != 12 or set(failed.setting) != {"K562-essential to RPE1"} or not held.prefix_both_resolved_above_zero.all():
    raise SystemExit("same_rows.csv: the joint comparator's rows do not match the author's statement (9 of 12, failing from K562-essential to RPE1)")
lows = held[["minus_gaussian_lo", "minus_shuffled_lo"]].min(axis=1)
add("S5", "Identity vectors beside the family mean and DepMap's score together (smaller difference)",
    "%s to %s in %d of %d" % (s(held.iu_estimate.min()), s(held.iu_estimate.max()), len(held), len(joint)),
    "lower ends %s to %s" % (s(lows.min()), s(lows.max())), "%s to %s genes" % (c(joint.n_test_genes.min()), c(joint.n_test_genes.max())),
    "Spearman, endpoint (b), mean gene effect; not shown from K562-essential to RPE1 (no vector); the 9 hold with prefix clusters; %s; post hoc, unadjusted" % DEPMAP,
    where("same", list(joint.line)))

full = pd.DataFrame(rows)
main = full[~full.full_only].drop(columns=["File and line", "Claim", "full_only"])
if len(main) != 16:
    raise SystemExit("the main table has %d rows, not the author's 16" % len(main))
tablelib.save("table2_main_estimates", main, list(F.values()), Path(__file__).resolve(), align="lllll")
tablelib.save("tableS2_main_estimates_full", full.assign(**{"Main table": ["no" if x else "yes" for x in full.full_only]}).drop(columns=["full_only"]),
              list(F.values()), Path(__file__).resolve(), align="llllllll")
