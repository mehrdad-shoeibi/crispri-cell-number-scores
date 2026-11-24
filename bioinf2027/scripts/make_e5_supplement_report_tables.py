"""Markdown tables for the supplement section of bioinf2027/reports/E5.md, printed from the saved result files only
(bioinf2027/results/E5/supplement/). Run from the repository root; prints to standard output."""
import json
import sys
from pathlib import Path

import pandas as pd

S = Path("bioinf2027/results/E5/supplement")
EMB = ["Geneformer", "scGPT", "GenePT"]
SET = ["within RPE1", "within K562-essential", "RPE1 to K562-essential", "K562-essential to RPE1"]
SHORT = {"within RPE1": "within RPE1", "within K562-essential": "within K562-ess.", "RPE1 to K562-essential": "RPE1 → K562-ess.",
         "K562-essential to RPE1": "K562-ess. → RPE1"}
out = []


def p(s=""):
    out.append(s)


def ci(e, lo, hi):
    return "%+.3f [%+.3f, %+.3f]" % (e, lo, hi)


def pv(x):
    return "<0.001" if x < 0.001 else "%.3f" % x


def ordered(frame):
    frame = frame.copy()
    frame["_e"] = frame.embedding.map({e: i for i, e in enumerate(EMB)})
    frame["_s"] = frame.setting.map({s: i for i, s in enumerate(SET)})
    return frame.sort_values(["_e", "_s"])


def holm(pvalues, alpha=0.05):
    """Holm's step-down decisions at family-wise alpha for the given p-values (descriptive here)."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    rejected = [False] * len(pvalues)
    for step, i in enumerate(order):
        if pvalues[i] > alpha / (len(pvalues) - step):
            break
        rejected[i] = True
    return rejected


def family_iu(embedding, setting):
    row = FAMILY[(FAMILY.endpoint == "b") & (FAMILY.embedding == embedding) & (FAMILY.setting == setting)]
    assert len(row) == 1
    return row["iu_estimate"].iloc[0]


def bar_table(frame, title, bar_name, with_holm=False):
    frame = ordered(frame)
    if with_holm:
        frame["holm"] = holm(list(frame["iu_p"]))
    p(title)
    p()
    p("| Vector | Setting | Test genes | ρ(Bc) | ρ(%s) | %s − Bc | ρ(%s with e) | smaller of e − C, e − S (p) | both above zero |%s family (b), smaller | Line |"
      % (bar_name, bar_name, bar_name, " Holm within the 12 (descriptive) |" if with_holm else ""))
    p("|---|---|---|---|---|---|---|---|---|%s---|---|" % ("---|" if with_holm else ""))
    for _, r in frame.iterrows():
        p("| %s | %s | %d | %.3f | %.3f | %s | %.3f | %+.3f (%s) | %s |%s %+.3f | %d |" % (
            r["embedding"], SHORT[r["setting"]], r["n_test_genes"], r["spearman_Bc_alone"], r["spearman_bar"],
            ci(r["bar_minus_Bc_estimate"], r["bar_minus_Bc_lo"], r["bar_minus_Bc_hi"]), r["spearman_e"], r["iu_estimate"], pv(r["iu_p"]),
            "yes" if r["both_resolved_above_zero"] else "no", (" %s |" % ("kept" if r["holm"] else "not kept")) if with_holm else "",
            family_iu(r["embedding"], r["setting"]), r["line"]))
    p()


FAMILY = pd.read_csv("bioinf2027/results/E5/family/family.csv")
SECONDARY = pd.read_csv("bioinf2027/results/E5/secondary/secondary.csv")
SECONDARY["line"] = SECONDARY.index + 2
fits = pd.read_csv(S / "fits.csv")
fits["line"] = fits.index + 2
dep = pd.read_csv(S / "depmap.csv")
dep["line"] = dep.index + 2
bar_table(fits[fits.item.str.startswith("analysis 1")], "**Table S1. Analysis 1: the symbol-prefix family mean FN beside Bc** (`fits.csv`, item \"analysis 1\"; "
          "FN built inside each outer fold, leave one out on the training rows).", "[Bc; FN]")
bar_table(dep[dep.item.str.startswith("analysis 2a")], "**Table S2a. Analysis 2a: DepMap's mean gene effect F_mean beside Bc** (`depmap.csv`; DepMap 24Q4 "
          "Public, not the current release; Holm within the 12 is descriptive, not planned).", "[Bc; F_mean]", with_holm=True)
bar_table(dep[dep.item.str.startswith("analysis 2b")], "**Table S2b. Analysis 2b: F_mean and the K562 gene effect F_K562 beside Bc** (`depmap.csv`; rows "
          "holding both; 24Q4, not the current release).", "[Bc; F_mean; F_K562]", with_holm=True)

p("**Table S3. Analysis 3: folds by k-means clusters of each vector (k = 50)** (`fits.csv`, item \"analysis 3\").")
p()
p("| Vector | Setting | Test genes | ρ(Bc) | ρ([Bc; e]) | e − C | e − S | smaller (p) | both above zero | family (b), smaller | Line |")
p("|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in ordered(fits[fits.item.str.startswith("analysis 3")]).iterrows():
    p("| %s | %s | %d | %.3f | %.3f | %s | %s | %+.3f (%s) | %s | %+.3f | %d |" % (
        r["embedding"], SHORT[r["setting"]], r["n_test_genes"], r["spearman_bar"], r["spearman_e"],
        ci(r["minus_gaussian_estimate"], r["minus_gaussian_lo"], r["minus_gaussian_hi"]),
        ci(r["minus_shuffled_estimate"], r["minus_shuffled_lo"], r["minus_shuffled_hi"]), r["iu_estimate"], pv(r["iu_p"]),
        "yes" if r["both_resolved_above_zero"] else "no", family_iu(r["embedding"], r["setting"]), r["line"]))
p()

p("**Table S4. Analysis 6, diagnostic: [Bc; B0] as the comparator on the part of (b) not predicted by the own shift** (`fits.csv`, "
  "item \"[Bc; B0] as the bar, (b) residualised on B2\"; beside E5's B2 row, `bioinf2027/results/E5/secondary/secondary.csv`).")
p()
p("| Vector | Setting | ρ(Bc), `secondary.csv` | ρ([Bc; B0]) | ρ([Bc; B0; e]) | smaller of e − C, e − S (p) | [Bc; B0; e] − [Bc; B0] | Lines (fits, secondary) |")
p("|---|---|---|---|---|---|---|---|")
for _, r in ordered(fits[fits.item.str.contains("residualised on B2")]).iterrows():
    b = SECONDARY[(SECONDARY["item"] == "(b) residualised on B2") & (SECONDARY.embedding == r["embedding"]) & (SECONDARY.setting == r["setting"])]
    assert len(b) == 1
    p("| %s | %s | %.3f | %.3f | %.3f | %+.3f (%s) | %s | %d, %d |" % (r["embedding"], SHORT[r["setting"]], b["spearman_bar"].iloc[0], r["spearman_bar"], r["spearman_e"],
                                                                r["iu_estimate"], pv(r["iu_p"]), ci(r["minus_bar_estimate"], r["minus_bar_lo"], r["minus_bar_hi"]),
                                                                r["line"], b["line"].iloc[0]))
p()

depth = pd.read_csv(S / "depth.csv")
depth["line"] = depth.index + 2
p("**Table S5. Analysis 4: depth matching and the depth ratio** (`depth.csv`; gene bootstrap, seeds 20271121 and 20271122).")
p()
p("| Screen | Quantity | Genes | Estimate [interval] | Line |")
p("|---|---|---|---|---|")
for _, r in depth[depth.quantity != "reading"].iterrows():
    p("| %s | %s | %d | %s | %d |" % (r["screen"], r["quantity"], r["n_genes"], ci(r["estimate"], r["lo"], r["hi"]), r["line"]))
p()

pool = pd.read_csv(S / "pool.csv")
pool["line"] = pool.index + 2
p("**Table S6. Analysis 5: the fixed-cell endpoint on nested subsets of the pool** (`pool.csv`; D_m = mean of (b at m cells − b on the "
  "full pool) over the sampled constructs, in units of the screen's between-gene IQR of b27; construct bootstrap).")
p()
p("| Screen | Quantity | Constructs | Estimate [interval] | Line |")
p("|---|---|---|---|---|")
for _, r in pool[pool.quantity.str.startswith("D_") | pool.quantity.str.startswith("rho")].iterrows():
    value = ci(r["estimate"], r["lo"], r["hi"]) if pd.notna(r["lo"]) else "%+.3f" % r["estimate"]
    p("| %s | %s | %d | %s | %d |" % (r["screen"], r["quantity"], r["constructs"], value, r["line"]))
p()

lists = json.loads((S / "supplement_lists.json").read_text())["counts"]
p("**Table S7. Analysis 7: coverage** (`supplement_lists.json`, key `counts`; per gene in `coverage_lists.csv`).")
p()
p("| Screen | Target genes | Covered by all three and Bc | Geneformer without a vector | scGPT without a vector | GenePT without a vector |")
p("|---|---|---|---|---|---|")
for s, c in lists.items():
    p("| %s | %d | %d | %d | %d | %d |" % (s, c["genes"], c["covered"], c["Geneformer without a vector"], c["scGPT without a vector"], c["GenePT without a vector"]))
p()

br = pd.read_csv(S / "breakdown.csv")
p("**Table S8. Analysis 8, descriptive: mean rank difference [Bc; e] − [Bc; C] by fifth of (b)** (`breakdown.csv`; ranks over the "
  "setting's test genes divided by their number; the same against [Bc; S] and by symbol prefix in the file).")
p()
p("| Vector | Setting | fifth 1 (lowest b) | fifth 2 | fifth 3 | fifth 4 | fifth 5 (highest b) |")
p("|---|---|---|---|---|---|---|")
for e in EMB:
    for s in SET:
        g = br[(br.embedding == e) & (br.setting == s) & br.group.str.startswith("fifth")]
        vals = [g[g.group == "fifth %d of (b)" % k]["mean_rank_e_minus_C"].iloc[0] for k in range(1, 6)]
        p("| %s | %s | %s |" % (e, SHORT[s], " | ".join("%+.3f" % v for v in vals)))
p()
sys.stdout.write("\n".join(out) + "\n")
