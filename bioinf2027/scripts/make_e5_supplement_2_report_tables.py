"""Markdown tables of the section on E5's supplement 2 in bioinf2027/reports/E5.md, printed from the saved result files only
(bioinf2027/results/E5/supplement_2/). Run from the repository root; prints to standard output."""
import sys
from pathlib import Path

import pandas as pd

S = Path("bioinf2027/results/E5/supplement_2")
EMB = ["Geneformer", "scGPT", "GenePT"]
SET = ["within RPE1", "within K562-essential", "RPE1 to K562-essential", "K562-essential to RPE1"]
SHORT = {"within RPE1": "within RPE1", "within K562-essential": "within K562-ess.", "RPE1 to K562-essential": "RPE1 → K562-ess.",
         "K562-essential to RPE1": "K562-ess. → RPE1"}
VERSION = {"a": "F_mean", "b": "F_mean; F_K562"}
out = []


def p(s=""):
    out.append(s)


def ci(e, lo, hi):
    return "%+.3f [%+.3f, %+.3f]" % (e, lo, hi)


def pv(x):
    return "<0.001" if x < 0.001 else "%.3f" % x


rows = pd.read_csv(S / "same_rows.csv")
rows["line"] = rows.index + 2
for v in ("a", "b"):
    p("**Table S2-%s. The smaller of e − C and e − S beside four comparators on the same rows, version %s (F = %s)** (`same_rows.csv`; "
      "gene-clustered intervals; p the larger of the two; the last column the joint comparator's state under symbol-prefix clusters; DepMap "
      "24Q4, not the current release)." % ("1a" if v == "a" else "1b", v, VERSION[v]))
    p()
    p("| Vector | Setting | Genes | Bc | [Bc; FN] | [Bc; F] | [Bc; FN; F] (p) | joint, prefix clusters | Lines |")
    p("|---|---|---|---|---|---|---|---|---|")
    for e in EMB:
        for s in SET:
            g = rows[(rows.version == v) & (rows.embedding == e) & (rows.setting == s)].set_index("comparator")
            j = g.loc["J"]
            state = "above zero" if j["prefix_both_resolved_above_zero"] else ("—" if not j["both_resolved_above_zero"] else "not shown")
            p("| %s | %s | %d | %+.3f | %+.3f | %+.3f | %+.3f (%s) | %s | %s |" % (
                e, SHORT[s], j["n_test_genes"], g.loc["Bc", "iu_estimate"], g.loc["FN", "iu_estimate"], g.loc["F", "iu_estimate"], j["iu_estimate"],
                pv(j["iu_p"]), state, "%d-%d" % (g["line"].min(), g["line"].max())))
    p()

comp = pd.read_csv(S / "comparisons.csv")
comp["line"] = comp.index + 2
p("**Table S2-2. The fitness score against each vector on the same rows, and the joint comparator against its parts** (`comparisons.csv`; "
  "gene-clustered intervals, symbol-prefix-clustered beside; the second and third quantities do not depend on the vector).")
p()
p("| Version | Vector | Setting | ρ([Bc; F]) − ρ([Bc; e]) | prefix clusters | ρ([Bc; FN; F]) − ρ([Bc; F]) | ρ([Bc; FN; F]) − ρ([Bc; FN]) | Lines |")
p("|---|---|---|---|---|---|---|---|")
for v in ("a", "b"):
    for e in EMB:
        for s in SET:
            g = comp[(comp.version == v) & (comp.embedding == e) & (comp.setting == s)]
            two = g[g.analysis == "(2)"].iloc[0]
            jf = g[g.quantity.str.endswith("rho([Bc; %s])" % VERSION[v])].iloc[0]
            jn = g[g.quantity.str.endswith("rho([Bc; FN])")].iloc[0]
            p("| %s | %s | %s | %s | [%+.3f, %+.3f] | %s | %s | %d-%d |" % (
                v, e, SHORT[s], ci(two["estimate"], two["lo"], two["hi"]), two["prefix_lo"], two["prefix_hi"], ci(jf["estimate"], jf["lo"], jf["hi"]),
                ci(jn["estimate"], jn["lo"], jn["hi"]), g["line"].min(), g["line"].max()))
p()

route = pd.read_csv(S / "route.csv")
route["line"] = route.index + 2
p("**Table S2-3. Diagnostic: each vector alone predicting F_mean out of fold** (`route.csv`; ridge with an intercept; no reading).")
p()
p("| Vector | Setting | Genes | ρ(e) | ρ(C) | ρ(S) | e − C | e − S | Line |")
p("|---|---|---|---|---|---|---|---|---|")
for e in EMB:
    for s in SET:
        r = route[(route.embedding == e) & (route.setting == s)].iloc[0]
        p("| %s | %s | %d | %.3f | %+.3f | %+.3f | %s | %s | %d |" % (e, SHORT[s], r["n_test_genes"], r["spearman_e"], r["spearman_C"], r["spearman_S"],
                                                          ci(r["minus_gaussian_estimate"], r["minus_gaussian_lo"], r["minus_gaussian_hi"]),
                                                          ci(r["minus_shuffled_estimate"], r["minus_shuffled_lo"], r["minus_shuffled_hi"]), r["line"]))
p()

b0 = pd.read_csv(S / "b0.csv")
b0["line"] = b0.index + 2
assoc = pd.read_csv(S / "b0_with_f_mean.csv")
assoc["line"] = assoc.index + 2
p("**Table S2-4. Diagnostic: [Bc; F_mean; B0] as the comparator, and B0 against F_mean** (`b0.csv`, `b0_with_f_mean.csv`; B0 is observed "
  "after the perturbation; no reading).")
p()
p("| Vector | Setting | ρ([Bc; F_mean; B0]) | ρ(with e) | smaller of e − C, e − S (p) | Line |")
p("|---|---|---|---|---|---|")
for e in EMB:
    for s in SET:
        r = b0[(b0.embedding == e) & (b0.setting == s)].iloc[0]
        p("| %s | %s | %.3f | %.3f | %+.3f (%s) | %d |" % (e, SHORT[s], r["spearman_bar"], r["spearman_e"], r["iu_estimate"], pv(r["iu_p"]), r["line"]))
p()
for _, r in assoc.iterrows():
    p("- %s: Spearman of B0 with F_mean %s over %d genes (`b0_with_f_mean.csv:%d`)." % (r["screen"], ci(r["estimate"], r["lo"], r["hi"]), r["n_genes"], r["line"]))
p()

counts = pd.read_csv(S / "hgnc_counts.csv")
counts["line"] = counts.index + 2
p("**Table S2-5. Symbol misses against HGNC (a count; no refit and no change of the covered genes)** (`hgnc_counts.csv`; per symbol in "
  "`hgnc_misses.csv`).")
p()
p("| Source | Screen | Misses | Approved symbol | Previous or alias of one gene | of several | Not in HGNC | Approved symbol in scGPT's vocabulary | Line |")
p("|---|---|---|---|---|---|---|---|---|")
for _, r in counts.iterrows():
    voc = "" if pd.isna(r["approved symbol in scGPT's vocabulary"]) else "%d" % r["approved symbol in scGPT's vocabulary"]
    p("| %s | %s | %d | %d | %d | %d | %d | %s | %d |" % (r["source"], r["screen"], r["misses"], r["an approved symbol"],
                                                       r["a previous or alias symbol of one approved gene"], r["a previous or alias symbol of several approved genes"],
                                                       r["not in HGNC"], voc, r["line"]))
p()
sys.stdout.write("\n".join(out) + "\n")
