"""Fig. 3: the in-distribution difference over the Gaussian control on both VCC labels per encoder, with the
size-matched null and the cell-count model, and the size-matched null on the Replogle screens (figure and table plan,
Fig. 3; the stop of 2026-11-17 06:50, figure plan item 4: G6 and R-squared in the main figure, G5, PCA-768 and the
Spearman version in the supplement in the same layout; the author's answer to the stop of 2026-11-17 17:10: a fourth
panel, G5 to the supplement; bioinf2027/GATE.md). Forest heads, head seeds averaged:
- (a) e − C1 on the all-cells label and on the size-fixed label, and their paired drop P-d (VCC test rows,
  target-gene cluster intervals, bioinf2027/results/E2/stats/vcc.csv; Geneformer's rows are E1b's z; FS2 Sections 7
  and 11);
- (b) the size-matched null N0 − C1 and e − N0 on the all-cells label (N0 is not computed for PCA-768);
- (c) e − B0 (the cell-count head) and the attribution quantities Δ and D½ on the all-cells label, with the B0 head
  alone (trivial heads of E1b) as a reference;
- (d) on the Spearman scale whatever the metric of (a) to (c): the external size-matched null minus the Gaussian
  control of its width on the Replogle screens, with heads trained on VCC's all-cells label, on the all-cells external
  label (S1 supplement, part (d), bioinf2027/results/S1_supplement/external_control.csv) beside the size-fixed external
  label (follow-up (4) of E2 (d), bioinf2027/results/E2/null_external_followup/existing_label/null_external.csv,
  quantity T_N0); E1b's target-gene draws; Stack's values on the Replogle screens carry their marker.
Usage, from the repository root:
  python bioinf2027/scripts/figures/fig3_vcc_difference.py [--cell G6|G5] [--metric r2|spearman] [--pca768]
      [--width 86|178] [--formats pdf,png] [--data-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

R = figlib.REPO / "bioinf2027" / "results"
VCC = R / "E2" / "stats" / "vcc.csv"
S1_CONTROL = R / "S1_supplement" / "external_control.csv"
FOLLOWUP_4 = R / "E2" / "null_external_followup" / "existing_label" / "null_external.csv"
PANELS = {
    "a": (("existing", "e - C1", "e − C1, all-cells label"), ("rebuilt", "e - C1", "e − C1, size-fixed label"),
          ("existing minus rebuilt", "P-d", "paired drop P-d")),
    "b": (("existing", "N0 - C1", "N0 − C1, all-cells label"), ("existing", "e - N0", "e − N0, all-cells label")),
    "c": (("existing", "e - B0", "e − B0, all-cells label"), ("existing", "Delta", "Δ, all-cells label"),
          ("existing", "D1/2", "D½, all-cells label")),
}
REFERENCE = ("trivial heads (E1b)", "existing", "R2(B0)", "B0 head alone (absolute), all-cells label")
SCREENS = ("RPE1", "K562-essential")
EXTERNAL_LABELS = ("all-cells external label", "size-fixed external label")
D_ENCODERS = ("geneformer", "scgpt", "stack", "pca50")


def options():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cell", choices=("G6", "G5"), default="G6")
    parser.add_argument("--metric", choices=("r2", "spearman"), default="r2")
    parser.add_argument("--pca768", action="store_true", help="add PCA-768 (the width-matched reference) to (a) to (c)")
    parser.add_argument("--width", type=int, choices=figlib.WIDTHS_MM, default=178)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def name(args) -> str:
    return "fig3_vcc_difference_%s_%s%s" % (args.cell, args.metric, "_pca768" if args.pca768 else "")


def table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["line"] = frame.index + 2
    return frame


def one(frame: pd.DataFrame, path: Path, **keys) -> pd.Series:
    hit = frame
    for k, v in keys.items():
        hit = hit[hit[k] == v]
    if len(hit) != 1:
        raise SystemExit("%d rows of %s for %s" % (len(hit), figlib.tablelib.rel(path), keys))
    return hit.iloc[0]


def panel_data(args) -> pd.DataFrame:
    vcc = table(VCC)
    encoders = {k: v for k, v in figlib.FILE_NAMES.items() if k != "pca768" or args.pca768}
    m, lo, hi = args.metric, args.metric + "_lo", args.metric + "_hi"
    rows = []

    def pick(encoder: str, label: str, quantity: str):
        hit = vcc[(vcc.encoder == encoder) & (vcc.cell == args.cell) & (vcc["head"] == "forest") & (vcc.label == label) & (vcc.quantity == quantity)]
        if len(hit) > 1:
            raise SystemExit("%d rows of vcc.csv for %s" % (len(hit), (encoder, args.cell, label, quantity)))
        return None if hit.empty else hit.iloc[0]

    for panel, entries in PANELS.items():
        for label, quantity, shown in entries:
            for key, encoder in encoders.items():
                r = pick(key, label, quantity)
                if r is None:
                    if (key, quantity) == ("pca768", "N0 - C1") or (key, quantity) == ("pca768", "e - N0"):
                        continue
                    raise SystemExit("no row of vcc.csv for %s" % ((key, args.cell, label, quantity),))
                rows.append({"panel": panel, "quantity": shown, "encoder": encoder, "screen": "VCC test", "cell": args.cell, "metric": args.metric,
                             "estimate": r[m], "lo": r[lo], "hi": r[hi], "n": "%d rows, %d target genes" % (r.n_test_rows, r.n_test_genes),
                             "descriptive": int(r.n_test_genes) < 50, "marker": "", "source": "bioinf2027/results/E2/stats/vcc.csv:%d" % r.line})
    enc, label, quantity, shown = REFERENCE
    r = pick(enc, label, quantity)
    if r is None:
        raise SystemExit("no B0 reference row in vcc.csv")
    rows.append({"panel": "c", "quantity": shown, "encoder": "B0", "screen": "VCC test", "cell": args.cell, "metric": args.metric, "estimate": r[m],
                 "lo": r[lo], "hi": r[hi], "n": "%d rows, %d target genes" % (r.n_test_rows, r.n_test_genes),
                 "descriptive": int(r.n_test_genes) < 50, "marker": "", "source": "bioinf2027/results/E2/stats/vcc.csv:%d" % r.line})
    s1, f4 = table(S1_CONTROL), table(FOLLOWUP_4)
    for key in D_ENCODERS:
        shown = figlib.FILE_NAMES[key]
        c1 = "C1" if key == "geneformer" else "C1[%s]" % key
        for screen in SCREENS:
            marker = figlib.STACK_MARKER if key == "stack" else ""
            r = one(s1, S1_CONTROL, analysis="(d)", encoder=key, cell=args.cell, screen=screen,
                    quantity="rho(N0x[%s], all-cells label) - rho(%s, all-cells label)" % (key, c1))
            rows.append({"panel": "d", "quantity": "N0 − C1, %s" % EXTERNAL_LABELS[0], "encoder": shown, "screen": screen, "cell": args.cell,
                         "metric": "spearman", "estimate": r.estimate, "lo": r.lo, "hi": r.hi, "n": "", "descriptive": False, "marker": marker,
                         "source": "%s:%d" % (figlib.tablelib.rel(S1_CONTROL), r.line)})
            r = one(f4, FOLLOWUP_4, encoder=key, cell=args.cell, screen=screen, label="existing", head="forest", quantity="T_N0")
            rows.append({"panel": "d", "quantity": "N0 − C1, %s" % EXTERNAL_LABELS[1], "encoder": shown, "screen": screen, "cell": args.cell,
                         "metric": "spearman", "estimate": r.estimate, "lo": r.lo, "hi": r.hi, "n": "", "descriptive": False, "marker": marker,
                         "source": "%s:%d" % (figlib.tablelib.rel(FOLLOWUP_4), r.line)})
    return pd.DataFrame(rows)


def draw(frame: pd.DataFrame, args, data: Path) -> None:
    plt = figlib.pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(args.width * figlib.MM, 118 * figlib.MM), constrained_layout=True)
    axes = axes.ravel()
    axis = figlib.say("fig3", {"r2": "x_r2", "spearman": "x_spearman"}[args.metric])
    encoders = [e for e in dict.fromkeys(frame[frame.panel != "d"].encoder) if e != "B0"]
    for ax, panel in zip(axes, ("a", "b", "c", "d")):
        part = frame[frame.panel == panel].reset_index(drop=True)
        if panel == "d":
            row = lambda screen, encoder: figlib.say("fig3", "d_row", screen=figlib.word(screen),  # noqa: E731
                                                     encoder=figlib.word(encoder) + (figlib.word("stack_marker") if encoder == "Stack" else ""))
            groups = [row(s, e) for e in part.encoder.drop_duplicates() for s in SCREENS]
            y = {g: len(groups) - 1 - i for i, g in enumerate(groups)}
            for r in part.itertuples():
                g = row(r.screen, r.encoder)
                filled = r.quantity.endswith(EXTERNAL_LABELS[0])
                yy = y[g] + (0.14 if filled else -0.14)
                colour = figlib.COLOURS[r.encoder]
                ax.plot([r.lo, r.hi], [yy, yy], color=colour, linewidth=0.8, solid_capstyle="butt")
                ax.plot([r.estimate], [yy], marker="o", markersize=3.2, color=colour, mfc=colour if filled else "white", mew=0.8, linestyle="none")
            ax.set_yticks(list(y.values()))
            ax.set_yticklabels([figlib.wrap(g, 26) for g in y])
            ax.set_ylim(-0.6, len(groups) - 0.4)
            ax.axvline(0.0, color="#000000", linewidth=0.5)
            ax.set_xlabel(figlib.say("fig3", "x_d"))
            handles = [ax.plot([], [], marker="o", color="#222222", mfc="#222222" if i == 0 else "white", mew=0.8, linestyle="none", markersize=3.2,
                               label=figlib.say("fig3", key))[0] for i, key in enumerate(("d_all", "d_fixed"))]
            ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.2, borderaxespad=0.0,
                      columnspacing=1.0)
        else:
            groups = list(dict.fromkeys(part.quantity))
            members = [e for e in encoders if e in set(part.encoder)] + (["B0"] if "B0" in set(part.encoder) else [])
            step = 0.8 / max(len(members), 1)
            y = {g: len(groups) - 1 - i for i, g in enumerate(groups)}
            for r in part.itertuples():
                k = members.index(r.encoder)
                yy = y[r.quantity] + 0.4 - step / 2 - k * step if r.encoder != "B0" else y[r.quantity]
                colour = figlib.COLOURS.get(r.encoder, figlib.GREYS.get(r.encoder, "#000000"))
                ax.plot([r.lo, r.hi], [yy, yy], color=colour, linewidth=0.8, solid_capstyle="butt")
                ax.plot([r.estimate], [yy], marker="o", markersize=3.0, color=colour, linestyle="none")
            ax.set_yticks(list(y.values()))
            ax.set_yticklabels([figlib.wrap(figlib.say("fig3", g), 26) for g in y])
            ax.set_ylim(-0.6, len(groups) - 0.4)
            for k in range(len(groups) - 1):
                ax.axhline(k + 0.5, color="#DDDDDD", linewidth=0.4)
            ax.axvline(0.0, color="#000000", linewidth=0.5)
            ax.set_xlabel(axis)
        figlib.panel_letter(ax, panel)
    handles = [axes[0].plot([], [], marker="o", color=figlib.COLOURS.get(e, figlib.GREYS.get(e)), linestyle="none", markersize=3.0,
                            label=figlib.say("fig3", "legend_B0") if e == "B0" else figlib.word(e))[0] for e in encoders + ["B0"]]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, handletextpad=0.2, columnspacing=1.2)
    figlib.save_figure(fig, name(args), args, data, Path(__file__).resolve())


def main() -> None:
    args = options()
    frame = panel_data(args)
    data = figlib.save_data(name(args), frame, [VCC, S1_CONTROL, FOLLOWUP_4], Path(__file__).resolve(),
                            extra={"cell": args.cell, "metric": args.metric, "pca768": args.pca768,
                                   "scope": "(a) to (c): VCC test rows, forest heads, head seeds averaged; target-gene cluster bootstrap, 10,000 "
                                            "replicates, seed 20270301; intervals on fewer than 50 target genes are descriptive. (d): Replogle "
                                            "constructs, Spearman scale, forest heads trained on VCC's all-cells label, head seeds 0 to 2 averaged, "
                                            "E1b's target-gene draws; the null has one control draw (seed 20271131); S1 supplement part (d) and "
                                            "follow-up (4) of E2 (d), both post hoc and unadjusted",
                                   "marker": "* Stack: %s (bioinf2027/results/E2/pretraining_check.json)" % figlib.STACK_MARKER,
                                   "bounds": "bioinf2027/factsheets/FS2_S2.md (no ranking of encoders; not 'the difference is due to cell count'); "
                                             "FS1_S1.md (part (d) says nothing about the encoders' path or their negative external values)"})
    if not args.data_only:
        draw(frame, args, data)


if __name__ == "__main__":
    main()
