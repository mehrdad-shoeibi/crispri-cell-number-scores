"""Markdown tables for the section of bioinf2027/reports/E2.md on the follow-ups of (d), printed from the saved result
files only (bioinf2027/results/E2/null_external_followup/, bioinf2027/results/E2/stats/vcc.csv). Run from the repository
root; prints to standard output."""
import sys
from pathlib import Path

import pandas as pd

F = Path("bioinf2027/results/E2/null_external_followup")
ENC = {"geneformer": "Geneformer", "scgpt": "scGPT", "stack": "Stack (marked)", "pca50": "PCA-50"}
ORDER = list(ENC)
out = []


def p(s=""):
    out.append(s)


def ci(e, lo, hi):
    return "%+.3f [%+.3f, %+.3f]" % (e, lo, hi)


saved = pd.read_csv(F / "saved.csv")
saved["line"] = saved.index + 2

# (1)
p("**Table F1. (1) The cell count against the size-fixed external label** (`saved.csv`, analysis (1)).")
p()
p("| Cell | Screen | ρ(n_t, y) | Line |")
p("|---|---|---|---|")
for _, r in saved[(saved.analysis == "(1)") & (saved.quantity == "rho(n_t, y)")].iterrows():
    p("| %s | %s | %s | %d |" % (r["cell"], r["screen"], ci(r["estimate"], r["lo"], r["hi"]), r["line"]))
p()
m = saved[(saved.analysis == "(1)") & (saved.quantity == "M against |T|min")]
reach = m[m.reading.str.startswith("a monotone function")]
p("Plan reading over the %d combinations of (d) with T below zero: \"no monotone function of the cell count alone ranks the constructs as far below "
  "zero as T does\" in %d; \"a monotone function of the cell count alone can reach T's size\" in %d (%s) (`saved.csv`, rows \"M against |T|min\")." % (
      len(m), len(m) - len(reach), len(reach),
      "; ".join("%s %s %s, line %d" % (ENC[r["encoder"]], r["cell"], r["screen"], r["line"]) for _, r in reach.iterrows())))
p()

# (2), (3a), (3b)
p("**Table F2. (2) and (3): the heads' external predictions against n_t, and e's ranking with n_t held fixed** (rebuilt label, forest, mean over "
  "head seeds; `saved.csv`).")
p()
p("| Encoder | Cell | Screen | ρ(null's prediction, n_t) | ρ(e's prediction, n_t) | ρ(e, y) | partial ρ(e, y given n_t) | ρ(e, y) − partial |")
p("|---|---|---|---|---|---|---|---|")
for enc in ORDER:
    for cell in ("G5", "G6"):
        for screen in ("RPE1", "K562-essential"):
            g = saved[(saved.encoder == enc) & (saved.cell == cell) & (saved.screen == screen)]
            def q(prefix, analysis, exact=False):
                hit = g[(g.analysis == analysis) & ((g.quantity == prefix) if exact else g.quantity.str.startswith(prefix))]
                if len(hit) != 1:
                    raise SystemExit("%d rows for %s %s %s %s" % (len(hit), enc, cell, screen, prefix))
                r = hit.iloc[0]
                return ci(r["estimate"], r["lo"], r["hi"])
            e = {"geneformer": "z"}.get(enc, enc)
            p("| %s | %s | %s | %s | %s | %s | %s | %s |" % (ENC[enc], cell, screen, q("rho(prediction of N0x", "(2)"), q("rho(prediction of %s, n_t)" % e, "(3a)"),
                                                           q("rho(%s, y)" % e, "(3b)", exact=True), q("partial rho", "(3b)"), q("rho(%s, y) - partial" % e, "(3b)")))
p()

# (4)
null = pd.read_csv(F / "existing_label" / "null_external.csv")
null["line"] = null.index + 2
p("**Table F3. (4) The size-matched null with heads trained on the all-cells label** (`existing_label/null_external.csv`; T, ρ(e) and ρ(C1) on "
  "the all-cells label equal `bioinf2027/results/E2/stats/external.csv`).")
p()
p("| Encoder | Cell | Screen | T (all-cells training label) | ρ(null) | ρ(null) − ρ(C1) | ρ(e) − ρ(null) |")
p("|---|---|---|---|---|---|---|")
for enc in ORDER:
    for cell in ("G5", "G6"):
        for screen in ("RPE1", "K562-essential"):
            g = null[(null.encoder == enc) & (null.cell == cell) & (null.screen == screen)].set_index("quantity")
            name = [q for q in g.index if q.startswith("rho(N0x")][0]
            p("| %s | %s | %s | %s | %+.3f | %s | %s |" % (ENC[enc], cell, screen, ci(*g.loc["T", ["estimate", "lo", "hi"]]), g.loc[name, "estimate"],
                                                        ci(*g.loc["T_N0", ["estimate", "lo", "hi"]]), ci(*g.loc["e - N0x", ["estimate", "lo", "hi"]])))
p()
vcc = pd.read_csv("bioinf2027/results/E2/stats/vcc.csv")
vcc["line"] = vcc.index + 2
x = vcc[(vcc.quantity == "N0 - C1") & (vcc["head"] == "forest") & (vcc.label == "existing")]
p("**Table F4. The null's in-distribution margin over C1 on the all-cells label** (VCC test rows; `bioinf2027/results/E2/stats/vcc.csv`, "
  "columns `r2` and `spearman`, quantity \"N0 - C1\", forest; its digest is recorded in `bioinf2027/results/E2/stats/stats.json`).")
p()
p("| Encoder | Cell | R² margin | Spearman margin | Line |")
p("|---|---|---|---|---|")
for enc in ORDER:
    for cell in ("G5", "G6"):
        r = x[(x.encoder == enc) & (x.cell == cell)].iloc[0]
        p("| %s | %s | %s | %s | %d |" % (ENC[enc], cell, ci(r["r2"], r["r2_lo"], r["r2_hi"]), ci(r["spearman"], r["spearman_lo"], r["spearman_hi"]), r["line"]))
p()
sys.stdout.write("\n".join(out) + "\n")
