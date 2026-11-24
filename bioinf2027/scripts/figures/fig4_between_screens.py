"""Fig. 4: ranking between the two Replogle screens per representation, beside the untrained baselines (figure and
table plan, Fig. 4; S4). Gene-disjoint predictions at 27 against 788 cells, forest heads, E1b's joint target-gene
draws (seeds 20260901 and 20260902), both directions:
- (a) T, the external Spearman minus that of C1: z, x, m and [x; m] from E6's family (Holm over 24,
  bioinf2027/results/E6/family.csv) and scGPT, Stack and PCA-50 from E2's test (iii) (Holm within E2's 30,
  bioinf2027/results/E2/stats/family.csv), each with its saved state;
- (b) the absolute external Spearman of each representation, beside the target screen's untrained control-cell mean B1
  and own shift B2 and TR trained on the source screen (bioinf2027/results/E6/family_pass/supplement/absolute.csv,
  bioinf2027/results/E6/transfer.csv, bioinf2027/results/E2/stats/e6.csv);
- (c): each encoder minus C2, the random projection of x (bioinf2027/results/E2/supplement/e_minus_c2.csv); by the
  author's answer to the stop of 2026-11-17 17:10 (bioinf2027/GATE.md) a supplement figure of its own (--part c2).
Stack's values carry their marker. Usage, from the repository root:
  python bioinf2027/scripts/figures/fig4_between_screens.py [--part main|c2] [--width 86|178] [--formats pdf,png] [--data-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

R = figlib.REPO / "bioinf2027" / "results"
E6_FAMILY, E2_FAMILY = R / "E6" / "family.csv", R / "E2" / "stats" / "family.csv"
ABSOLUTE, TRANSFER, E2_E6 = R / "E6" / "family_pass" / "supplement" / "absolute.csv", R / "E6" / "transfer.csv", R / "E2" / "stats" / "e6.csv"
MINUS_C2 = R / "E2" / "supplement" / "e_minus_c2.csv"
DIRECTIONS = ("RPE1 to K562-essential", "K562-essential to RPE1")
E2_ENCODERS = (("scgpt", "scGPT"), ("stack", "Stack"), ("pca50", "PCA-50"))


def options():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", choices=("main", "c2"), default="main", help="main: panels (a) and (b); c2: the supplement figure of (c)")
    parser.add_argument("--width", type=int, choices=figlib.WIDTHS_MM, default=178)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["line"] = frame.index + 2
    return frame


def one(frame: pd.DataFrame, path: Path, **keys) -> pd.Series:
    hit = frame
    for k, v in keys.items():
        hit = hit[hit[k] == v]
    if len(hit) != 1:
        raise SystemExit("%d rows of %s for %s" % (len(hit), path.name, keys))
    return hit.iloc[0]


def row(panel, direction, shown, r, path, estimate="estimate", lo="lo", hi="hi", state="", marker=""):
    return {"panel": panel, "direction": direction, "representation": shown, "estimate": r[estimate], "lo": r[lo], "hi": r[hi],
            "state": state, "marker": marker, "source": "%s:%d" % (figlib.tablelib.rel(path), r.line)}


def panel_data(args) -> pd.DataFrame:
    # each part reads only the files it plots, so that its record lists every file it opens
    if args.part == "main":
        e6, e2, absolute, transfer, e2e6 = table(E6_FAMILY), table(E2_FAMILY), table(ABSOLUTE), table(TRANSFER), table(E2_E6)
    else:
        c2 = table(MINUS_C2)
    rows = []
    for d in DIRECTIONS:
        if args.part == "c2":
            for key, shown in (("geneformer", "z"),) + E2_ENCODERS:
                r = one(c2, MINUS_C2, encoder=key, direction=d)
                rows.append(row("c", d, shown, r, MINUS_C2, state=r.state, marker=figlib.STACK_MARKER if key == "stack" else ""))
            continue
        for arm in figlib.REPRESENTATIONS:
            r = one(e6, E6_FAMILY, direction=d, statistic="T", arm=arm)
            rows.append(row("a", d, arm, r, E6_FAMILY, state=r.state))
        for key, shown in E2_ENCODERS:
            r = one(e2, E2_FAMILY, test="iii", encoder=key, direction=d, statistic="T")
            rows.append(row("a", d, shown, r, E2_FAMILY, state=r.state, marker=figlib.STACK_MARKER if key == "stack" else ""))
        for arm in figlib.REPRESENTATIONS:
            r = one(absolute, ABSOLUTE, direction=d, quantity="Spearman of %s" % arm)
            rows.append(row("b", d, arm, r, ABSOLUTE))
        for key, shown in E2_ENCODERS:
            r = one(e2e6, E2_E6, encoder=key, direction=d, quantity="rho(%s)" % key)
            rows.append(row("b", d, shown, r, E2_E6, marker=figlib.STACK_MARKER if key == "stack" else ""))
        for raw in ("B1", "B2"):
            r = one(absolute, ABSOLUTE, direction=d, quantity="Spearman of the target screen's raw %s" % raw)
            rows.append(row("b", d, "%s, target screen, untrained" % raw, r, ABSOLUTE))
        r = one(transfer, TRANSFER, direction=d, endpoint="27/788", model="TR", head="forest", subset="all")
        rows.append(row("b", d, "TR, trained on the source screen", r, TRANSFER, estimate="spearman", lo="spearman_lo", hi="spearman_hi"))
    return pd.DataFrame(rows)


def draw(frame: pd.DataFrame, args, data: Path) -> None:
    plt = figlib.pyplot()
    panels = list(dict.fromkeys(frame.panel))
    fig, axes = plt.subplots(len(DIRECTIONS), len(panels), figsize=(args.width * figlib.MM, (135 if args.part == "main" else 70) * figlib.MM),
                             constrained_layout=True, squeeze=False, gridspec_kw={"width_ratios": [1, 1.4] if len(panels) == 2 else [1]})
    label = {k: figlib.say("fig4", "x_" + k) for k in ("a", "b", "c")}
    shown = lambda r: figlib.say("fig4", r) if r in figlib.labels()["fig4"] else figlib.word(r)  # noqa: E731
    for i, d in enumerate(DIRECTIONS):
        for j, panel in enumerate(panels):
            ax = axes[i][j]
            part = frame[(frame.panel == panel) & (frame.direction == d)].reset_index(drop=True)
            names = [figlib.wrap(shown(r) + (figlib.word("stack_marker") if m else ""), 40) for r, m in zip(part.representation, part.marker)]
            colours = [figlib.COLOURS.get(r, figlib.GREYS.get(r.split(",")[0], "#555555")) for r in part.representation]
            figlib.interval_rows(ax, part, names, colours, xlabel=label[panel])
            if i == 0 and len(panels) > 1:
                figlib.panel_letter(ax, panel)
            ax.set_title(figlib.word(d), fontsize=figlib.BASE_PT, pad=3)
    figlib.save_figure(fig, name(args), args, data, Path(__file__).resolve())


def name(args) -> str:
    return "fig4_between_screens" if args.part == "main" else "figS_c_between_screens_minus_c2"


def main() -> None:
    args = options()
    frame = panel_data(args)
    inputs = [E6_FAMILY, E2_FAMILY, ABSOLUTE, TRANSFER, E2_E6] if args.part == "main" else [MINUS_C2]
    data = figlib.save_data(name(args), frame, inputs, Path(__file__).resolve(),
                            extra={"scope": "gene-disjoint predictions, 27 against 788 cells, forest, E1b's joint target-gene draws",
                                   "families": "E6's T: Holm over E6's 24; E2's T of test (iii): Holm within E2's 30",
                                   "marker": "* Stack: %s (bioinf2027/results/E2/pretraining_check.json)" % figlib.STACK_MARKER,
                                   "bounds": "bioinf2027/factsheets/FS4_S4.md (above a Gaussian control only; none shown beyond the trivial predictors)"})
    if not args.data_only:
        draw(frame, args, data)


if __name__ == "__main__":
    main()
