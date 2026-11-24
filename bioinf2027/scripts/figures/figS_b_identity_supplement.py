"""Figure S-b: the identity task's supplement panels that the stop of 2026-11-17 06:50 moves out of Fig. 5 (figure plan
item 5: "The B0 diagnostic, the cluster folds and endpoint (a) go to the supplement"). Per vector and setting, the
smaller of e − C and e − S, drawn with the interval of the term that gives it (both terms in the panel data):
- (a) endpoint (a) beside Bc, E5's family (bioinf2027/results/E5/family/family.csv, Holm over 24);
- (b) endpoint (b) with the cell count B0 added to the comparator, a diagnostic: B0 is observed after the perturbation
  (bioinf2027/results/E5/secondary/diagnostics.csv, item "[Bc; B0] as the bar", unadjusted);
- (c) endpoint (b) with folds formed by k-means clusters of each vector (k = 50; E5 supplement analysis 3,
  bioinf2027/results/E5/supplement/fits.csv, post hoc, unadjusted), beside the family's gene folds (family.csv).
Usage, from the repository root:
  python bioinf2027/scripts/figures/figS_b_identity_supplement.py [--width 86|178] [--formats pdf,png] [--data-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

E5 = figlib.REPO / "bioinf2027" / "results" / "E5"
FAMILY, DIAGNOSTICS, FITS = E5 / "family" / "family.csv", E5 / "secondary" / "diagnostics.csv", E5 / "supplement" / "fits.csv"
SETTINGS = ("within RPE1", "within K562-essential", "RPE1 to K562-essential", "K562-essential to RPE1")


def table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["line"] = frame.index + 2
    return frame


def entry(panel: str, series: str, r: pd.Series, path: Path, status: str) -> dict:
    smaller = "gaussian" if r.minus_gaussian_estimate <= r.minus_shuffled_estimate else "shuffled"
    if abs(min(r.minus_gaussian_estimate, r.minus_shuffled_estimate) - r.iu_estimate) > 1e-12:
        raise SystemExit("%s:%d: iu_estimate is not the smaller difference" % (path.name, r.line))
    return {"panel": panel, "series": series, "vector": r.embedding, "setting": r.setting, "genes": int(r.n_test_genes), "estimate": r.iu_estimate,
            "lo": r["minus_%s_lo" % smaller], "hi": r["minus_%s_hi" % smaller], "p": r.iu_p, "interval_of": "e − C" if smaller == "gaussian" else "e − S",
            "e_minus_C": "%+.3f [%+.3f, %+.3f]" % (r.minus_gaussian_estimate, r.minus_gaussian_lo, r.minus_gaussian_hi),
            "e_minus_S": "%+.3f [%+.3f, %+.3f]" % (r.minus_shuffled_estimate, r.minus_shuffled_lo, r.minus_shuffled_hi),
            "status": status, "source": "%s:%d" % (figlib.tablelib.rel(path), r.line)}


def one(frame: pd.DataFrame, path: Path, **keys) -> pd.Series:
    hit = frame
    for k, v in keys.items():
        hit = hit[hit[k] == v]
    if len(hit) != 1:
        raise SystemExit("%d rows of %s for %s" % (len(hit), path.name, keys))
    return hit.iloc[0]


def panel_data() -> pd.DataFrame:
    family, diagnostics, fits = table(FAMILY), table(DIAGNOSTICS), table(FITS)
    rows = []
    for setting in SETTINGS:
        for vector in figlib.VECTORS:
            rows.append(entry("a", "endpoint (a), beside Bc", one(family, FAMILY, endpoint="a", embedding=vector, setting=setting), FAMILY,
                              "family of 24, Holm"))
            rows.append(entry("b", "endpoint (b), beside [Bc; B0]",
                              one(diagnostics, DIAGNOSTICS, item="[Bc; B0] as the bar", endpoint="b", embedding=vector, setting=setting), DIAGNOSTICS,
                              "diagnostic, unadjusted"))
            rows.append(entry("c", "endpoint (b), gene folds (the family)", one(family, FAMILY, endpoint="b", embedding=vector, setting=setting), FAMILY,
                              "family of 24, Holm"))
            rows.append(entry("c", "endpoint (b), folds by clusters of the vector",
                              one(fits, FITS, item="analysis 3: folds by clusters of the vector", embedding=vector, setting=setting), FITS,
                              "post hoc, unadjusted"))
    return pd.DataFrame(rows)


def draw(frame: pd.DataFrame, args, data: Path) -> None:
    """Three rows (panels) of four settings; rows of each axis are the series, one point per vector in its fixed colour."""
    plt = figlib.pyplot()
    panels = ("a", "b", "c")
    groups_of = {p: list(dict.fromkeys(frame[frame.panel == p].series)) for p in panels}
    fig, axes = plt.subplots(len(panels), len(SETTINGS), figsize=(args.width * figlib.MM, 120 * figlib.MM), constrained_layout=True,
                             sharey="row", sharex="row", gridspec_kw={"height_ratios": [len(groups_of[p]) + 1 for p in panels]})
    step = 0.24
    for i, panel in enumerate(panels):
        groups = groups_of[panel]
        y = {g: len(groups) - 1 - k for k, g in enumerate(groups)}
        for j, setting in enumerate(SETTINGS):
            ax = axes[i][j]
            part = frame[(frame.panel == panel) & (frame.setting == setting)]
            for r in part.itertuples():
                k = figlib.VECTORS.index(r.vector)
                yy = y[r.series] + step - k * step
                colour = figlib.COLOURS[r.vector]
                ax.plot([r.lo, r.hi], [yy, yy], color=colour, linewidth=0.8, solid_capstyle="butt")
                ax.plot([r.estimate], [yy], marker="o", markersize=2.8, color=colour, linestyle="none")
            for k in range(len(groups) - 1):
                ax.axhline(k + 0.5, color="#DDDDDD", linewidth=0.4)
            ax.axvline(0.0, color="#000000", linewidth=0.5)
            ax.set_yticks(list(y.values()))
            ax.set_yticklabels([figlib.wrap(figlib.say("figS_b", g), 36) for g in groups])
            ax.set_ylim(-0.6, len(groups) - 0.4)
            ax.set_xlabel(figlib.wrap(figlib.say("figS_b", "x"), 22))
            if i == 0:
                ax.set_title(figlib.word(setting), fontsize=figlib.BASE_PT, pad=3)
            if j == 0:
                figlib.panel_letter(ax, panel)
    handles = [axes[0][0].plot([], [], marker="o", color=figlib.COLOURS[v], linestyle="none", markersize=2.8, label=figlib.word(v))[0] for v in figlib.VECTORS]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, handletextpad=0.2, columnspacing=1.2)
    figlib.save_figure(fig, "figS_b_identity_supplement", args, data, Path(__file__).resolve())


def main() -> None:
    args = figlib.arguments(__doc__.split("\n")[0])
    frame = panel_data()
    data = figlib.save_data("figS_b_identity_supplement", frame, [FAMILY, DIAGNOSTICS, FITS], Path(__file__).resolve(),
                            extra={"intervals": "gene bootstrap, 10,000 replicates, setting seeds 20271121 to 20271124; each value drawn with the "
                                                "interval of the term that gives the smaller difference",
                                   "bounds": "bioinf2027/factsheets/FS5_S5.md (the B0 panel is a diagnostic; no ranking or pooling of vectors)"})
    if not args.data_only:
        draw(frame, args, data)


if __name__ == "__main__":
    main()
