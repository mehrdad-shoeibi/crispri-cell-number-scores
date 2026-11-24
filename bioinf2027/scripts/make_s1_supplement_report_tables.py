"""Markdown tables of bioinf2027/reports/S1_supplement.md, printed from the saved result files only
(bioinf2027/results/S1_supplement/). Run from the repository root; prints to standard output."""
import sys
from pathlib import Path

import pandas as pd

S = Path("bioinf2027/results/S1_supplement")
SPLIT = {"train": "training", "val": "validation", "test": "test", "all": "all splits"}
out = []


def p(s=""):
    out.append(s)


def ci(e, lo, hi, signed=True):
    f = "%+.3f" if signed else "%.3f"
    return (f + " [" + f + ", " + f + "]") % (e, lo, hi)


a = pd.read_csv(S / "vcc_paired.csv")
a["line"] = a.index + 2
p("**Table 1. (a) VCC: the two labels' association with pool size on the same rows, and their paired difference** (`vcc_paired.csv`; "
  "Spearman over rows, R2's target-gene bootstrap, seed 20270301; validation descriptive below 50 target genes).")
p()
p("| Cell | Split | Rows | Target genes | All-cells label | Rebuilt label | Paired difference | Line |")
p("|---|---|---|---|---|---|---|---|")
for _, r in a.iterrows():
    p("| %s | %s | %d | %d | %s | %s | %s | %d |" % (r["cell"], SPLIT[r["split"]], r["rows"], r["target_genes"], ci(r["first"], r["existing_lo"], r["existing_hi"]),
                                                  ci(r["second"], r["rebuilt_lo"], r["rebuilt_hi"]), ci(r["estimate"], r["lo"], r["hi"]), r["line"]))
p()

b = pd.read_csv(S / "vcc_mechanical.csv")
b["line"] = b.index + 2
p("**Table 2. (b) VCC: the all-cells label within (target gene, guide), its mechanical part and the excess** (`vcc_mechanical.csv`; as R2: "
  "observed on single-guide groups of at least three rows, mechanical on R2's simulated subsets, excess = sign(observed) × (observed − "
  "mechanical); target-gene bootstrap, seed 20270301).")
p()
p("| Cell | Split | Observed within (gene, guide) | Mechanical | Excess | Observed rows | Simulated labels | Line |")
p("|---|---|---|---|---|---|---|---|")
for _, r in b.iterrows():
    p("| %s | %s | %s | %s | %s | %d | %d | %d |" % (r["cell"], SPLIT[r["split"]], ci(r["observed_within_gene_and_guide"], r["observed_lo"], r["observed_hi"]),
                                                  ci(r["mechanical"], r["mechanical_lo"], r["mechanical_hi"]), ci(r["excess"], r["excess_lo"], r["excess_hi"]),
                                                  r["observed_rows"], r["simulated_labels"], r["line"]))
p()

c = pd.read_csv(S / "vcc_curve.csv")
c["line"] = c.index + 2
p("**Table 3. (b) VCC, descriptive: the label on nested subsets of G6 rows with at least 60 perturbed cells** (`vcc_curve.csv`; mean over "
  "rows of (label at m − label on the full pool), in units of that label's IQR over the same rows' full-pool labels; target-gene bootstrap).")
p()
p("| m | All-cells label | G6 rebuilt label (10 against 700) | Rows | Lines |")
p("|---|---|---|---|---|")
for m in sorted(c["m"].unique()):
    e = c[(c["m"] == m) & c["label"].str.startswith("existing")].iloc[0]
    r = c[(c["m"] == m) & c["label"].str.startswith("G6")].iloc[0]
    p("| %d | %s | %s | %d | %d, %d |" % (m, ci(e["estimate"], e["lo"], e["hi"]), ci(r["estimate"], r["lo"], r["hi"]), e["rows"], e["line"], r["line"]))
p()

x = pd.read_csv(S / "external_association.csv")
x["line"] = x.index + 2
p("**Table 4. (c) Replogle: the label with all eligible perturbed cells against the perturbed-cell count** (`external_association.csv`; "
  "weighted Spearman over E1b's external constructs, E1b's target-gene draws, seeds 20260901 and 20260902).")
p()
p("| Screen | Quantity | Constructs | Estimate [interval] | Line |")
p("|---|---|---|---|---|")
for _, r in x.iterrows():
    p("| %s | %s | %d | %s | %d |" % (r["screen"], r["quantity"], r["constructs"], ci(r["estimate"], r["lo"], r["hi"]), r["line"]))
p()

s = pd.read_csv(S / "external_subsampling.csv")
s["line"] = s.index + 2
p("**Table 5. (c) Replogle, descriptive: nested subsets of constructs with at least 150 eligible cells** (`external_subsampling.csv`; mean "
  "over constructs of (label at m − label on the full pool) in units of the label's IQR over the same constructs; the last row per label "
  "is the within-construct Spearman of the centred label against the centred log m).")
p()
p("| Screen | Label | m | Constructs | Estimate [interval] | Line |")
p("|---|---|---|---|---|---|")
for _, r in s.iterrows():
    p("| %s | %s | %s | %d | %s | %d |" % (r["screen"], r["label"], r["m"], r["constructs"], ci(r["estimate"], r["lo"], r["hi"]), r["line"]))
p()

d = pd.read_csv(S / "external_control.csv")
d["line"] = d.index + 2
diff = d[d["quantity"].str.contains(" - ")]
p("**Table 6. (d) The external positive control on the all-cells label, from saved predictions only** (`external_control.csv`; forest "
  "heads trained on VCC's all-cells label, head seeds 0 to 2 averaged; E1b's draws; Stack marked).")
p()
p("| Screen | Cell | Encoder | Null − C1 | Encoder − C1 | Lines |")
p("|---|---|---|---|---|---|")
for screen in ("RPE1", "K562-essential"):
    base = d[(d["screen"] == screen) & d["quantity"].str.startswith("rho(n,")].iloc[0]
    for cell in ("G5", "G6"):
        for encoder in ("geneformer", "scgpt", "stack", "pca50"):
            g = diff[(diff["screen"] == screen) & (diff["cell"] == cell) & (diff["encoder"] == encoder)]
            null = g[g["quantity"].str.startswith("rho(N0x")].iloc[0]
            enc = g[~g["quantity"].str.startswith("rho(N0x")].iloc[0]
            p("| %s | %s | %s | %s | %s | %d, %d |" % (screen, cell, encoder, ci(null["estimate"], null["lo"], null["hi"]), ci(enc["estimate"], enc["lo"], enc["hi"]),
                                                    null["line"], enc["line"]))
    p("| %s | — | ρ(n, all-cells label) | %s | | %d |" % (screen, ci(base["estimate"], base["lo"], base["hi"]), base["line"]))
p()
sys.stdout.write("\n".join(out) + "\n")
