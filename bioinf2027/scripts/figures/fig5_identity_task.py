"""Fig. 5: the identity task on the target-excluded downstream endpoint (figure and table plan, Fig. 5; the stop of
2026-11-17 06:50, figure plan item 5: panel (a) the Spearman of Bc, [Bc; e], [Bc; F_mean] and the joint comparator on
the same rows; panel (b) the smaller difference beside Bc, Bc with the family mean, Bc with F_mean and the joint
comparator; the B0 diagnostic, the cluster folds and endpoint (a) go to the supplement). Endpoint (b) at 27 against 788
cells, the covered target genes holding a DepMap 24Q4 score (not the current release), per vector and setting, from
bioinf2027/results/E5/supplement_2/same_rows.csv (post hoc, unadjusted, gene-clustered intervals):
- (a) ρ of each comparator alone (column spearman_bar of rows Bc, FN, F and J) and ρ([Bc; e]) (spearman_e of row Bc),
  point values (the file holds no interval for an absolute Spearman);
- (b) per comparator the smaller of e − C and e − S (iu_estimate), drawn with the interval of the term that gives it;
  both terms' intervals are in the panel data.
Version a: F = F_mean; version b (supplement): F = F_mean and F_K562.
Usage, from the repository root:
  python bioinf2027/scripts/figures/fig5_identity_task.py [--version a|b] [--width 86|178] [--formats pdf,png] [--data-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

SAME_ROWS = figlib.REPO / "bioinf2027" / "results" / "E5" / "supplement_2" / "same_rows.csv"
SETTINGS = ("within RPE1", "within K562-essential", "RPE1 to K562-essential", "K562-essential to RPE1")
COMPARATORS = (("Bc", "Bc"), ("FN", "[Bc; FN]"), ("F", "[Bc; F]"), ("J", "[Bc; FN; F]"))


def options():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", choices=("a", "b"), default="a")
    parser.add_argument("--width", type=int, choices=figlib.WIDTHS_MM, default=178)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def panel_data(args) -> pd.DataFrame:
    rows = pd.read_csv(SAME_ROWS, float_precision="round_trip")
    rows["line"] = rows.index + 2
    rows = rows[rows.version == args.version]
    out = []
    for setting in SETTINGS:
        for vector in figlib.VECTORS:
            for key, shown in COMPARATORS:
                hit = rows[(rows.setting == setting) & (rows.embedding == vector) & (rows.comparator == key)]
                if len(hit) != 1:
                    raise SystemExit("%d rows of same_rows.csv for %s" % (len(hit), (args.version, setting, vector, key)))
                r = hit.iloc[0]
                src = "bioinf2027/results/E5/supplement_2/same_rows.csv:%d" % r.line
                base = {"setting": setting, "vector": vector, "comparator": shown, "genes": int(r.n_test_genes),
                        "depmap_release": r.depmap_release, "source": src}
                out.append({"panel": "a", "quantity": "rho(%s)" % shown, "estimate": r.spearman_bar, "lo": float("nan"), "hi": float("nan"), **base})
                if key == "Bc":
                    out.append({"panel": "a", "quantity": "rho([Bc; e])", "estimate": r.spearman_e, "lo": float("nan"), "hi": float("nan"), **base})
                smaller = "gaussian" if r.minus_gaussian_estimate <= r.minus_shuffled_estimate else "shuffled"
                out.append({"panel": "b", "quantity": "smaller of e − C and e − S beside %s" % shown, "estimate": r.iu_estimate,
                            "lo": r["minus_%s_lo" % smaller], "hi": r["minus_%s_hi" % smaller], "p": r.iu_p,
                            "interval_of": "e − C" if smaller == "gaussian" else "e − S",
                            "e_minus_C": "%+.3f [%+.3f, %+.3f]" % (r.minus_gaussian_estimate, r.minus_gaussian_lo, r.minus_gaussian_hi),
                            "e_minus_S": "%+.3f [%+.3f, %+.3f]" % (r.minus_shuffled_estimate, r.minus_shuffled_lo, r.minus_shuffled_hi),
                            "prefix_clusters_both_above_zero": bool(r.prefix_both_resolved_above_zero), **base})
                if abs(min(r.minus_gaussian_estimate, r.minus_shuffled_estimate) - r.iu_estimate) > 1e-12:
                    raise SystemExit("%s: iu_estimate is not the smaller difference" % src)
    return pd.DataFrame(out)


def draw(frame: pd.DataFrame, args, data: Path) -> None:
    """Two rows of four settings; rows of each axis are the comparators, with one point per vector in its fixed colour."""
    plt = figlib.pyplot()
    fig, axes = plt.subplots(2, len(SETTINGS), figsize=(args.width * figlib.MM, 100 * figlib.MM), constrained_layout=True,
                             sharey="row", sharex="row")
    step = 0.24
    for i, panel in enumerate(("a", "b")):
        part_all = frame[frame.panel == panel]
        groups = list(dict.fromkeys(part_all.quantity))
        key = lambda g: g.replace("smaller of e − C and e − S beside ", "beside ")  # noqa: E731
        row_label = lambda s: s if "\n" in s else figlib.wrap(s, 36)  # noqa: E731  (a label broken by hand in labels.toml keeps its breaks)
        shown = [row_label(figlib.say("fig5", key(g) + ("_b" if args.version == "b" and key(g) + "_b" in figlib.labels()["fig5"] else "")))
                 for g in groups]
        y = {g: len(groups) - 1 - k for k, g in enumerate(groups)}
        for j, setting in enumerate(SETTINGS):
            ax = axes[i][j]
            part = part_all[part_all.setting == setting]
            for r in part.itertuples():
                k = figlib.VECTORS.index(r.vector)
                yy = y[r.quantity] + step - k * step
                colour = figlib.COLOURS[r.vector]
                if panel == "b":
                    ax.plot([r.lo, r.hi], [yy, yy], color=colour, linewidth=0.8, solid_capstyle="butt")
                ax.plot([r.estimate], [yy], marker="o", markersize=2.8, color=colour, linestyle="none")
            for k in range(len(groups) - 1):
                ax.axhline(k + 0.5, color="#DDDDDD", linewidth=0.4)
            if panel == "b":
                ax.axvline(0.0, color="#000000", linewidth=0.5)
            ax.set_yticks(list(y.values()))
            ax.set_yticklabels(shown)
            ax.set_ylim(-0.6, len(groups) - 0.4)
            ax.set_xlabel(figlib.wrap(figlib.say("fig5", "x_" + panel).replace("\n", " "), 22))
            if i == 0:
                ax.set_title(figlib.word(setting), fontsize=figlib.BASE_PT, pad=3)
            if j == 0:
                figlib.panel_letter(ax, panel)
    handles = [axes[0][0].plot([], [], marker="o", color=figlib.COLOURS[v], linestyle="none", markersize=2.8, label=figlib.word(v))[0] for v in figlib.VECTORS]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, handletextpad=0.2, columnspacing=1.2)
    figlib.save_figure(fig, "fig5_identity_task_%s" % args.version, args, data, Path(__file__).resolve())


def main() -> None:
    args = options()
    frame = panel_data(args)
    data = figlib.save_data("fig5_identity_task_%s" % args.version, frame, [SAME_ROWS], Path(__file__).resolve(),
                            extra={"version": args.version, "status": "post hoc, unadjusted (E5 supplement 2); DepMap 24Q4, not the current release",
                                   "intervals": "gene-clustered bootstrap, 10,000 replicates, setting seeds 20271121 to 20271124; panel (b) draws the "
                                                "interval of the term that gives the smaller difference",
                                   "bounds": "bioinf2027/factsheets/FS5_S5.md (no ranking or pooling of vectors; nothing beyond gene family and "
                                             "fitness together without the failing setting, the magnitudes and the score's name and release)"})
    if not args.data_only:
        draw(frame, args, data)


if __name__ == "__main__":
    main()
