"""Markdown tables for bioinf2027/reports/E5.md, printed from the saved result files only."""
import json
import sys
from pathlib import Path

import pandas as pd

R = Path("bioinf2027/results/E5")
S = R / "secondary"
out = []


def p(line=""):
    out.append(line)


def f3(x):
    return "%+.3f" % x


def ci(est, lo, hi):
    return "%+.3f [%+.3f, %+.3f]" % (est, lo, hi)


def pv(x):
    return "<0.001" if x < 0.001 else "%.3f" % x


ORDER_S = ["within RPE1", "within K562-essential", "RPE1 to K562-essential", "K562-essential to RPE1"]
ORDER_E = ["Geneformer", "scGPT", "GenePT"]
SHORT = {"within RPE1": "within RPE1", "within K562-essential": "within K562-ess.", "RPE1 to K562-essential": "RPE1 → K562-ess.",
         "K562-essential to RPE1": "K562-ess. → RPE1"}


def key(frame):
    frame = frame.copy()
    frame["_s"] = frame["setting"].map({s: i for i, s in enumerate(ORDER_S)})
    if "embedding" in frame:
        frame["_e"] = frame["embedding"].map({s: i for i, s in enumerate(ORDER_E)})
        return frame.sort_values(["endpoint", "_e", "_s"])
    return frame.sort_values(["endpoint", "_s"])


fam = key(pd.read_csv(R / "family" / "family.csv"))
folds = pd.read_csv(S / "folds.csv")
r2 = folds.pivot_table(index=["embedding", "endpoint", "setting"], columns="model", values="pooled_r2", aggfunc="first")

# ---------------------------------------------------------------- table 1: the family
p("**Table 1. E5's family of 24** (ridge, gene folds, gene bootstrap, 10,000 replicates; `bioinf2027/results/E5/family/family.csv`; R² from `bioinf2027/results/E5/secondary/folds.csv`, pooled over folds).")
p()
p("| Vector | End-point | Setting | Test genes | ρ(Bc) | ρ([Bc; e]) | R² Bc / [Bc; e] | [Bc; e] − [Bc; C] | [Bc; e] − [Bc; S] | p (IU) | Holm | Read | Positive control |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in fam.iterrows():
    rr = r2.loc[(r["embedding"], r["endpoint"], r["setting"])]
    read = "read" if r["readable"] else "not read (positive control not resolved)"
    pc = r["positive_control_version"]
    if pc != "planned":
        pc = "revised (planned: %s)" % ("resolved" if r["planned_positive_control_resolved"] else "not resolved")
    p("| %s | (%s) | %s | %d | %.3f | %.3f | %.3f / %.3f | %s | %s | %s | %s | %s | %s |" % (
        r["embedding"], r["endpoint"], SHORT[r["setting"]], r["n_test_genes"], r["spearman_Bc"], r["spearman_[Bc; e]"],
        rr["Bc"], rr["[Bc; e]"], ci(r["minus_gaussian_estimate"], r["minus_gaussian_lo"], r["minus_gaussian_hi"]),
        ci(r["minus_shuffled_estimate"], r["minus_shuffled_lo"], r["minus_shuffled_hi"]), pv(r["iu_p"]),
        "rejected" if r["holm_rejected"] else "not rejected", read, pc))
p()
fj = json.loads((R / "family" / "family.json").read_text())
p("Holm rejected %d of 24; %d readable; SVD fallbacks %s in all (`family.json`)." % (
    int(fam["holm_rejected"].sum()), int(fam["readable"].sum()), fj.get("svd_fallbacks_total", fam["svd_fallbacks"].sum())))
p()

# ---------------------------------------------------------------- table 2: (b) beside its qualifiers
diag = pd.read_csv(S / "diagnostics.csv")
sec = pd.read_csv(S / "secondary.csv")
base = pd.read_csv(S / "family_baseline.csv")
forest = pd.read_csv(S / "forest.csv")
p("**Table 2. Endpoint (b): each family value beside the qualifiers the results review asks for** (all secondary or diagnostic, unadjusted). IU = the smaller of the two differences against the Gaussian and the shuffled control. Files: `family.csv`; `diagnostics.csv` (item \"[Bc; B0] as the bar\"); `secondary.csv` (item \"prefixes as bootstrap clusters\": the lower ends of both differences); `family_baseline.csv`; `forest.csv`.")
p()
p("| Vector | Setting | Family IU (gene clusters) | IU, B0 in the bar | Lower ends, prefix clusters (vs C; vs S) | Prefix-family mean: ρ | [Bc; e] − prefix-family mean | Forest IU | Forest [Bc; e] − Bc |")
p("|---|---|---|---|---|---|---|---|---|")
for _, r in fam[fam.endpoint == "b"].iterrows():
    k = (diag.item == "[Bc; B0] as the bar") & (diag.embedding == r["embedding"]) & (diag.endpoint == "b") & (diag.setting == r["setting"])
    d = diag[k].iloc[0]
    k = (sec.item == "prefixes as bootstrap clusters") & (sec.embedding == r["embedding"]) & (sec.endpoint == "b") & (sec.setting == r["setting"])
    c = sec[k].iloc[0]
    k = (base.embedding == r["embedding"]) & (base.endpoint == "b") & (base.setting == r["setting"])
    b = base[k].iloc[0]
    k = (forest.embedding == r["embedding"]) & (forest.endpoint == "b") & (forest.setting == r["setting"])
    fo = forest[k].iloc[0]
    p("| %s | %s | %s (p %s) | %s (p %s) | %+.3f; %+.3f | %.3f [%.3f, %.3f] | %s (p %s) | %s (p %s) | %s |" % (
        r["embedding"], SHORT[r["setting"]], f3(r["iu_estimate"]), pv(r["iu_p"]), f3(d["iu_estimate"]), pv(d["iu_p"]),
        c["minus_gaussian_lo"], c["minus_shuffled_lo"],
        b["spearman_family_neighbour"], b["spearman_family_neighbour_lo"], b["spearman_family_neighbour_hi"],
        ci(b["e_minus_family_neighbour_estimate"], b["e_minus_family_neighbour_lo"], b["e_minus_family_neighbour_hi"]), pv(b["e_minus_family_neighbour_p"]),
        f3(fo["iu_estimate"]), pv(fo["iu_p"]), ci(fo["minus_bar_estimate"], fo["minus_bar_lo"], fo["minus_bar_hi"])))
p()
p("Bar with B0, between the screens: ρ([Bc; B0]) against ρ(Bc) (`diagnostics.csv`):")
p()
p("| Setting | ρ(Bc) | ρ([Bc; B0]) |")
p("|---|---|---|")
for s in ORDER_S:
    d = diag[(diag.item == "[Bc; B0] as the bar") & (diag.endpoint == "b") & (diag.setting == s)].iloc[0]
    fb = fam[(fam.endpoint == "b") & (fam.setting == s)].iloc[0]
    p("| %s | %.3f | %.3f |" % (SHORT[s], fb["spearman_Bc"], d["spearman_bar"]))
p()

# ---------------------------------------------------------------- table 3: (b) sensitivities
items = ["B1 as the bar", "prefix-grouped folds", "depth-matched controls", "10 kb neighbours removed", "full space", "50-cell version", "(b) residualised on B2"]
p("**Table 3. Endpoint (b), sensitivities** (ridge; IU estimate; * where both differences are above zero with p below 0.05, unadjusted; `bioinf2027/results/E5/secondary/secondary.csv`). The B2 row is \"the part of (b) not predicted by the target's own shift\" (the plan's phrase, \"response per unit knockdown\", is not a dose-normalised quantity; review point 4).")
p()
p("| Vector | Setting | " + " | ".join(items) + " |")
p("|---|---|" + "---|" * len(items))
for e in ORDER_E:
    for s in ORDER_S:
        cells = []
        for it in items:
            k = (sec.item == it) & (sec.embedding == e) & (sec.endpoint == "b") & (sec.setting == s)
            if not k.any():
                cells.append("—")
                continue
            c = sec[k].iloc[0]
            mark = "" if c["positive_control_resolved"] else " (control not resolved)"
            cells.append("%s%s%s" % (f3(c["iu_estimate"]), "*" if c["both_resolved_above_zero"] else "", mark))
        p("| %s | %s | %s |" % (e, SHORT[s], " | ".join(cells)))
p()

# ---------------------------------------------------------------- table 4: (a) secondary
p("**Table 4. Endpoint (a): the family value beside its secondary versions** (IU estimate; * as in Table 3; `family.csv`, `secondary.csv`, `diagnostics.csv`, `forest.csv`).")
p()
items_a = ["B1 as the bar", "prefix-grouped folds", "prefixes as bootstrap clusters"]
p("| Vector | Setting | Family | " + " | ".join(items_a) + " | B0 in the bar | Forest |")
p("|---|---|---|" + "---|" * len(items_a) + "---|---|")
for _, r in fam[fam.endpoint == "a"].iterrows():
    cells = []
    for it in items_a:
        c = sec[(sec.item == it) & (sec.embedding == r["embedding"]) & (sec.endpoint == "a") & (sec.setting == r["setting"])].iloc[0]
        cells.append("%s%s" % (f3(c["iu_estimate"]), "*" if c["both_resolved_above_zero"] else ""))
    d = diag[(diag.item == "[Bc; B0] as the bar") & (diag.embedding == r["embedding"]) & (diag.endpoint == "a") & (diag.setting == r["setting"])].iloc[0]
    fo = forest[(forest.embedding == r["embedding"]) & (forest.endpoint == "a") & (forest.setting == r["setting"])].iloc[0]
    p("| %s | %s | %s%s | %s | %s%s | %s%s |" % (r["embedding"], SHORT[r["setting"]], f3(r["iu_estimate"]),
                                              " (Holm)" if r["holm_rejected"] else "", " | ".join(cells),
                                              f3(d["iu_estimate"]), "*" if d["both_resolved_above_zero"] else "",
                                              f3(fo["iu_estimate"]), "*" if fo["both_resolved_above_zero"] else ""))
p()

# ---------------------------------------------------------------- table 5: gates, composition, depth, O3
g = json.loads((R / "endpoint_b" / "gates.json").read_text())["gates"]
ga = json.loads((R / "prefit" / "gates_a.json").read_text())["gates"]
eb = json.loads((R / "endpoint_b" / "endpoint_b.json").read_text())
p("**Table 5. Endpoints and their gates, before any embedding fit** (`bioinf2027/results/E5/endpoint_b/gates.json`, `bioinf2027/results/E5/prefit/gates_a.json`).")
p()
p("| Screen | Endpoint | Genes (with halves) | Split-half ρ (gate ≥ 0.5) | Spearman-Brown | Beyond Bc ρ (gate ≥ 0.2) | Composition null median / between-gene IQR | ρ with depth ratio |")
p("|---|---|---|---|---|---|---|---|")
for screen in ("RPE1", "K562-essential"):
    a = ga[screen]
    p("| %s | (a) | %d with halves | %.3f | %.3f | %.3f | — | — |" % (screen, a["genes_with_halves"], a["reliability_spearman"], a["spearman_brown_upper_bound"], a["beyond_bc_spearman"]))
    for v in ("27", "50"):
        b = g[screen][v]
        p("| %s | (b), %s cells | %d (%d) | %.3f | %.3f | %.3f | %.4f / %.3f | %+.3f |" % (
            screen, v, b["genes"], b["genes_with_halves"], b["reliability_spearman"], b["spearman_brown_upper_bound"],
            b["beyond_bc_spearman"], b["composition_null_median"], b["between_gene_iqr"], b["spearman_with_depth"]))
p()

# ---------------------------------------------------------------- table 6: ceiling, positive control
ceil = key(pd.read_csv(R / "prefit" / "ceiling.csv"))
pc = pd.read_csv(R / "prefit" / "positive_control.csv")
pcr = pd.read_csv(R / "prefit" / "positive_control_revised.csv")
p("**Table 6. Oracle ceiling and pipeline positive control** (1,511 genes both screens hold; `bioinf2027/results/E5/prefit/ceiling.csv`, `positive_control.csv`, `positive_control_revised.csv`). PC: IU estimate (p); resolved = p below 0.05 with both differences above zero.")
p()
p("| Setting | End-point | ρ(Bc) | Ceiling: other screen's label − Bc | PC 512 | PC 768 | PC 1,536 planned | PC 1,536 revised |")
p("|---|---|---|---|---|---|---|---|")
for _, c in ceil.iterrows():
    cells = []
    for w in (512, 768, 1536):
        x = pc[(pc.setting == c["setting"]) & (pc.endpoint == c["endpoint"]) & (pc.nominal_width == w)].iloc[0]
        cells.append("%s (%s)%s" % (f3(x["iu_estimate"]), pv(x["iu_p"]), " resolved" if x["resolved"] else ""))
    x = pcr[(pcr.setting == c["setting"]) & (pcr.endpoint == c["endpoint"])].iloc[0]
    cells.append("%s (%s)%s" % (f3(x["iu_estimate"]), pv(x["iu_p"]), " resolved" if x["resolved"] else ""))
    p("| %s | (%s) | %.3f | %s | %s |" % (SHORT[c["setting"]], c["endpoint"], c["spearman_Bc"],
                                        ci(c["difference_estimate"], c["difference_lo"], c["difference_hi"]), " | ".join(cells)))
p()

# ---------------------------------------------------------------- table 7: size association and agreement
size = pd.read_csv(S / "size_association.csv")
agr = pd.read_csv(S / "agreement.csv")
p("**Table 7. Size association of (b) and agreement of (a) with (b)** (gene bootstrap, one draw per screen; `bioinf2027/results/E5/secondary/size_association.csv`, `agreement.csv`).")
p()
p("| Screen | Statistic | Genes | Spearman with B0 |")
p("|---|---|---|---|")
for _, r in size.iterrows():
    p("| %s | %s | %d | %s |" % (r["screen"], r["statistic"], r["n_genes"], ci(r["spearman_with_B0"], r["lo"], r["hi"])))
p()
p("| Screen | Pair | Genes | Spearman |")
p("|---|---|---|---|")
for _, r in agr.iterrows():
    p("| %s | (%s) with (%s), %s | %d | %s |" % (r["screen"], r["first"], r["second"], r["subset"], r["n_genes"], ci(r["spearman"], r["lo"], r["hi"])))
p()

# ---------------------------------------------------------------- table 8: diagnostics from e
p("**Table 8. Diagnostics: what each vector predicts on its own** (ridge with an intercept, out of fold; ρ of e's prediction, with the Gaussian and shuffled controls' ρ; `bioinf2027/results/E5/secondary/diagnostics.csv`).")
p()
p("| Vector | Target | " + " | ".join(SHORT[s] for s in ORDER_S) + " |")
p("|---|---|" + "---|" * 4)
for e in ORDER_E:
    for it in ("B1 from e", "detection rate from e", "B0 from e"):
        cells = []
        for s in ORDER_S:
            d = diag[(diag.item == it) & (diag.embedding == e) & (diag.setting == s)].iloc[0]
            cells.append("%.3f (C %+.3f, S %+.3f)" % (d["spearman_e"], d["spearman_C"], d["spearman_S"]))
        p("| %s | %s | %s |" % (e, it.replace(" from e", ""), " | ".join(cells)))
p()

# ---------------------------------------------------------------- table 9: between embeddings
be = key(pd.read_csv(S / "between_embeddings.csv"))
p("**Table 9. Contrasts between the vectors, descriptive, no ranking** (ρ([Bc; e₁]) − ρ([Bc; e₂]); `bioinf2027/results/E5/secondary/between_embeddings.csv`).")
p()
p("| End-point | Setting | Geneformer − scGPT | Geneformer − GenePT | scGPT − GenePT |")
p("|---|---|---|---|---|")
for (ep, s), grp in be.groupby(["endpoint", "setting"], sort=False):
    cells = []
    for a, b in (("Geneformer", "scGPT"), ("Geneformer", "GenePT"), ("scGPT", "GenePT")):
        x = grp[(grp["first"] == a) & (grp["second"] == b)].iloc[0]
        cells.append(ci(x["e_difference_estimate"], x["e_difference_lo"], x["e_difference_hi"]))
    p("| (%s) | %s | %s |" % (ep, SHORT[s], " | ".join(cells)))
p()

# ---------------------------------------------------------------- table 10: VCC
vcc = pd.read_csv(S / "vcc.csv")
p("**Table 10. VCC, descriptive** (ridge; Bc of three columns on both endpoints, Section 21; no positive control; not read; `bioinf2027/results/E5/secondary/vcc.csv`).")
p()
p("| Vector | Label | Setting | Training / test genes | ρ(Bc) | ρ([Bc; e]) | [Bc; e] − [Bc; C] | [Bc; e] − [Bc; S] |")
p("|---|---|---|---|---|---|---|---|")
for _, r in vcc.iterrows():
    p("| %s | %s | %s | %d / %d | %.3f | %.3f | %s | %s |" % (r["embedding"], r["label"], r["setting"], r["n_training_genes"], r["n_test_genes"],
                                                  r["spearman_bar"], r["spearman_e"],
                                                  ci(r["minus_gaussian_estimate"], r["minus_gaussian_lo"], r["minus_gaussian_hi"]),
                                                  ci(r["minus_shuffled_estimate"], r["minus_shuffled_lo"], r["minus_shuffled_hi"])))
p()

# ---------------------------------------------------------------- table 11: runs
runs = json.loads(Path("bioinf2027/runs/E5.json").read_text())["runs"]
p("**Table 11. E5's runs** (`bioinf2027/runs/E5.json`; times from the run records).")
p()
p("| Label | Command (script and arguments) | Started | Finished | Exit |")
p("|---|---|---|---|---|")
for r in runs:
    cmd = r["command"].replace("python bioinf2027/scripts/", "")
    p("| %s | `%s` | %s | %s | %s |" % (r["label"], cmd, r["started"][:19].replace("T", " "), str(r.get("finished", ""))[:19].replace("T", " "), r["returncode"]))
p()

sys.stdout.write("\n".join(out) + "\n")
