"""Fig. 2: the endpoint and cell number before and after size control, and what the size-fixed endpoint tracks
(figure and table plan, Fig. 2; the author's answers to the stops of 2026-11-17 17:10 and 20:30, bioinf2027/GATE.md:
candidate B is the main figure; the scatter, the validation rows and the full share panel go to a supplement version).

Main version (--version main), four panels:
- (a) curve: S1 (b), G6 rows with at least 60 perturbed cells, the mean of (label at m − label on the full pool) in
  units of the IQR of these rows' full-pool labels (bioinf2027/results/S1_supplement/vcc_curve.csv), beside the two
  Replogle screens' subsampling of S1 (c), constructs with at least 150 perturbed cells
  (bioinf2027/results/S1_supplement/external_subsampling.csv);
- (b) association: Spearman of the label with cell number, VCC all-cells and size-fixed labels on the same rows of G5
  and G6 (bioinf2027/endpoints/readouts/R2_between.csv), and on both Replogle screens the label with all eligible
  perturbed cells beside the 27-cell label, each against 788 pooled controls (S1 (c),
  bioinf2027/results/S1_supplement/external_association.csv);
- (c) own shift: Spearman of the size-fixed label with the target's own shift (R3, bioinf2027/endpoints/readouts/R1_R3_R6.csv),
  VCC training and test and both screens, G5 and G6;
- (d) share: the share of the replicable rank variance of the cell-disjoint half labels explained by B0 to B4
  (bioinf2027/endpoints/readouts/supplement/5_6_R5_detail.csv), G5 and G6 only, one value per data set and setting:
  the one E1a's R5 rule uses (bioinf2027/reports/E1a.md:83-84: on VCC the deciding version, 27-cell halves in G5 and
  10-cell halves in G6; on the screens the pooled control halves); point values, one fold seed.
Supplement version (--version supplement), four panels: (a) and (b) the scatter, VCC rows of G5, label against
perturbed cells per row, one panel per label on the same axes ($IGA_STORAGE_ROOT/bioinf2027/endpoints/G5/vcc_labels.csv,
checked against bioinf2027/endpoints/G5/manifest.json; R2's rows); (c) the own shift with the validation rows; (d) the
share with every row (the upper bounds and G1 and G2).
All figure text comes from labels.toml (section fig2); labels are drawn in greys and fills, so that no label takes an
encoder's colour.
Usage, from the repository root:
  python bioinf2027/scripts/figures/fig2_endpoint_cell_number.py [--version main|supplement] [--width 86|178]
      [--formats pdf,png] [--data-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

B = figlib.REPO / "bioinf2027"
R2 = B / "endpoints" / "readouts" / "R2_between.csv"
R3 = B / "endpoints" / "readouts" / "R1_R3_R6.csv"
R5 = B / "endpoints" / "readouts" / "supplement" / "5_6_R5_detail.csv"
S1X = B / "results" / "S1_supplement" / "external_association.csv"
CURVE = B / "results" / "S1_supplement" / "vcc_curve.csv"
SUBSAMPLING = B / "results" / "S1_supplement" / "external_subsampling.csv"
G5_MANIFEST = B / "endpoints" / "G5" / "manifest.json"
R3_LINES = {("G5", "train"): 147, ("G5", "val"): 159, ("G5", "test"): 171, ("G6", "train"): 183, ("G6", "val"): 195,
            ("G6", "test"): 207, ("G5", "RPE1"): 530, ("G5", "K562-essential"): 533, ("G6", "RPE1"): 536, ("G6", "K562-essential"): 539}
RULE_LEVEL = {"G5": 27, "G6": 10}  # the half-label level E1a's R5 rule uses in each setting (bioinf2027/reports/E1a.md:83-84)
DATASETS = ("VCC", "RPE1", "K562-essential")
LABEL_STYLE = {"all-cells": dict(color="#222222", mfc="#222222"), "size-fixed": dict(color="#888888", mfc="white")}
SETTING_MARKER = {"G5": "o", "G6": "s"}
SCREEN_MARKER = {"VCC": "o", "RPE1": "s", "K562-essential": "^"}


def options():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", choices=("main", "supplement"), default="main")
    parser.add_argument("--width", type=int, choices=figlib.WIDTHS_MM, default=178)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def name(args) -> str:
    return "fig2_endpoint_cell_number" if args.version == "main" else "figS_d_endpoint_cell_number"


def table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["line"] = frame.index + 2
    return frame


def src(path: Path, line: int) -> str:
    return "%s:%d" % (figlib.tablelib.rel(path), line)


def storage_root() -> Path:
    sys.path.insert(0, str(B / "src"))
    from bioinf2027 import config
    return Path(config.storage_root())


def scatter_rows() -> tuple[pd.DataFrame, dict]:
    """The G5 rows of R2, from the data-root label table after its SHA-256 is checked against the G5 manifest."""
    entry = json.loads(G5_MANIFEST.read_text())["tables"]["vcc_labels.csv"]
    path = storage_root() / "bioinf2027" / "endpoints" / "G5" / "vcc_labels.csv"
    if figlib.tablelib.sha256(path) != entry["sha256"]:
        raise SystemExit("vcc_labels.csv of G5 does not match bioinf2027/endpoints/G5/manifest.json")
    labels = pd.read_csv(path, float_precision="round_trip")
    labels["line"] = labels.index + 2
    rows = labels[labels["y"].notna() & labels["existing_label"].notna()]
    expected = table(R2).set_index("line").loc[24]
    if len(rows) != int(expected.n_rows) or rows.target_gene.nunique() != int(expected.n_target_genes):
        raise SystemExit("the scatter rows (%d, %d genes) are not R2's rows (%d, %d)" % (
            len(rows), rows.target_gene.nunique(), expected.n_rows, expected.n_target_genes))
    out = []
    for label, column in (("all-cells", "existing_label"), ("size-fixed", "y")):
        out.append(pd.DataFrame({"panel": "scatter", "dataset": "VCC", "setting": "G5", "label": label, "x_perturbed_cells": rows["n_pool"],
                                 "estimate": rows[column], "lo": float("nan"), "hi": float("nan"), "split": rows["split"],
                                 "target_gene": rows["target_gene"], "source": ["vcc_labels.csv:%d" % n for n in rows["line"]]}))
    return pd.concat(out, ignore_index=True), {"$IGA_STORAGE_ROOT/bioinf2027/endpoints/G5/vcc_labels.csv": entry}


def panel_data(args) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    rows = []
    main = args.version == "main"
    r3, r5 = table(R3), table(R5)
    if main:  # the association and the curve are read only by the main version, so that each record lists what it reads
        r2, s1x = table(R2), table(S1X)
        for cell, (first, second) in (("G5", (25, 24)), ("G6", (33, 32))):
            for line, label in ((first, "all-cells"), (second, "size-fixed")):
                r = r2.set_index("line").loc[line]
                if r.cell != cell or r.split != "all" or (label == "all-cells") != r.label.startswith("existing"):
                    raise SystemExit("R2_between.csv:%d is not the %s row of %s" % (line, label, cell))
                rows.append({"panel": "association", "dataset": "VCC", "setting": cell, "label": label, "estimate": r.spearman, "lo": r.spearman_lo,
                             "hi": r.spearman_hi, "n": "%d rows, %d target genes" % (r.n_rows, r.n_target_genes), "source": src(R2, line)})
        for screen, lines in (("RPE1", (2, 4)), ("K562-essential", (10, 12))):
            for line, label, quantity in ((lines[0], "all-cells", "rho(all cells against 788 pooled controls label, n)"),
                                          (lines[1], "size-fixed", "rho(27 against 788 label, n)")):
                r = s1x.set_index("line").loc[line]
                if r.screen != screen or r.quantity != quantity:
                    raise SystemExit("external_association.csv:%d is not %s, %s" % (line, screen, quantity))
                rows.append({"panel": "association", "dataset": screen, "label": label, "estimate": r.estimate, "lo": r.lo, "hi": r.hi,
                             "n": "%d constructs" % r.constructs, "source": src(S1X, line)})
        for r in table(CURVE).itertuples():
            rows.append({"panel": "curve", "dataset": "VCC", "setting": r.cell, "label": "all-cells" if r.label.startswith("existing") else "size-fixed",
                         "x_perturbed_cells": r.m, "estimate": r.estimate, "lo": r.lo, "hi": r.hi,
                         "n": "%d rows with at least %d perturbed cells, %d target genes" % (r.rows, r.rows_with_at_least, r.target_genes),
                         "source": src(CURVE, r.line)})
        for r in table(SUBSAMPLING).itertuples():
            if not str(r.m).isdigit():
                continue  # the within-construct Spearman row is not a point of the curve
            rows.append({"panel": "curve", "dataset": r.screen, "label": "all-cells" if r.label.startswith("all eligible") else "size-fixed",
                         "x_perturbed_cells": int(r.m), "estimate": r.estimate, "lo": r.lo, "hi": r.hi,
                         "n": "%d constructs with at least %d perturbed cells" % (r.constructs, r.constructs_with_at_least), "source": src(SUBSAMPLING, r.line)})
    for (cell, part), line in R3_LINES.items():
        r = r3.set_index("line").loc[line]
        external = part in ("RPE1", "K562-essential")
        ok = r.cell == cell and r.readout == "R3" and (r.screen == part if external else (r.subset == "all rows" and r.label == "median, external mode"))
        if not ok:
            raise SystemExit("R1_R3_R6.csv:%d is not R3 of %s, %s" % (line, cell, part))
        if main and part == "val":
            continue  # the validation rows go to the supplement version (the author's answer to the stop of 20:30)
        rows.append({"panel": "own shift", "dataset": part if external else "VCC", "split": "" if external else part, "setting": cell,
                     "estimate": r.spearman, "lo": r.spearman_lo, "hi": r.spearman_hi,
                     "n": "%d %s, %d target genes%s" % (r.n_rows, "constructs" if external else "rows", r.n_target_genes, "; descriptive" if bool(r.descriptive) else ""),
                     "source": src(R3, line)})
    for r in r5.itertuples():
        if main and not (r.cell in RULE_LEVEL and int(r.level) == RULE_LEVEL[r.cell] and not r.version.startswith("upper bound")):
            continue  # the upper bounds and G1 and G2 go to the supplement version
        rows.append({"panel": "share", "dataset": r.side, "setting": r.cell, "level": int(r.level), "version": r.version,
                     "estimate": r.replicable_share_explained, "lo": float("nan"), "hi": float("nan"),
                     "n": "%d rows, %d target genes" % (r.rows, r.target_genes), "source": src(R5, r.line)})
    frame = pd.DataFrame(rows)
    if main and len(frame[frame.panel == "share"]) != 6:
        raise SystemExit("the main share panel should hold 6 rows (3 data sets x G5, G6), not %d" % len(frame[frame.panel == "share"]))
    scatter, data_root = (None, {}) if main else scatter_rows()
    return frame, scatter, data_root


def say(key: str, **fill) -> str:
    return figlib.say("fig2", key, **fill)


def legend_handles(ax, entries):
    return [ax.plot([], [], **style, label=label)[0] for label, style in entries]


def dot_rows(ax, part: pd.DataFrame, groups: list, group_label, series: list, series_of, style, xlabel: str, zero=True, above=False):
    """One row per group, top to bottom, with one point and interval per series, offset within the row."""
    keys = [s for s, _ in series]
    offsets = {s: (len(keys) - 1) / 2 * 0.28 - i * 0.28 for i, s in enumerate(keys)}
    y = {g: len(groups) - 1 - i for i, g in enumerate(groups)}
    for _, r in part.iterrows():
        g, s = group_label(r), series_of(r)
        yy = y[g] + offsets[s]
        st = style(s)
        ax.plot([r.lo, r.hi], [yy, yy], color=st["color"], linewidth=0.8, solid_capstyle="butt")
        ax.plot([r.estimate], [yy], marker=st.get("marker", "o"), markersize=3.2, color=st["color"], mfc=st.get("mfc", st["color"]), mew=0.7, linestyle="none")
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels(list(y.keys()))
    ax.set_ylim(-0.6, len(groups) - 0.4)
    if zero:
        ax.axvline(0.0, color="#000000", linewidth=0.5)
    ax.set_xlabel(xlabel)
    entries = [(legend, dict(marker=style(s).get("marker", "o"), color=style(s)["color"], mfc=style(s).get("mfc", style(s)["color"]), mew=0.7,
                             linestyle="none", markersize=3.2)) for s, legend in series]
    handles = legend_handles(ax, entries)
    if above:
        ax.legend(handles=handles, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles), handletextpad=0.2,
                  borderaxespad=0.0, columnspacing=1.0)
    else:
        ax.legend(handles=handles, frameon=False, loc="best", handletextpad=0.2, borderaxespad=0.2)


def dataset_name(r) -> str:
    return figlib.say("names", r.dataset)


def draw(frame: pd.DataFrame, scatter, args, data: Path) -> None:
    plt = figlib.pyplot()
    main = args.version == "main"
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import ScaledTranslation
    fig = plt.figure(figsize=(args.width * figlib.MM, 118 * figlib.MM), layout="constrained")
    top, bottom = fig.subfigures(2, 1)  # each row lays out its own columns, so long row labels in one row do not move the other
    if main:
        axes = list(top.subplots(1, 2)) + list(bottom.subplots(1, 2))
    else:  # the two scatter panels on the same axes
        axes = list(top.subplots(1, 2, sharex=True, sharey=True)) + list(bottom.subplots(1, 2, width_ratios=[1, 1.3]))
    labels = (("all-cells", say("label_all")), ("size-fixed", say("label_fixed")))
    panels = ("curve", "association", "own shift", "share") if main else ("scatter all-cells", "scatter size-fixed", "own shift", "share")
    for ax, panel, letter in zip(axes, panels, "abcd"):
        if panel.startswith("scatter"):
            label = panel.split(" ")[1]
            g = scatter[scatter.label == label]
            st = LABEL_STYLE[label]
            ax.scatter(g.x_perturbed_cells, g.estimate, s=0.8, alpha=0.35 if label == "all-cells" else 0.5, color=st["color"], linewidths=0, rasterized=True)
            ax.set_xscale("log")
            ax.set_xticks([27, 40, 60, 100])
            ax.set_xticklabels(["27", "40", "60", "100"])
            ax.minorticks_off()
            ax.set_xlabel(say("scatter_x"))
            if label == "all-cells":
                ax.set_ylabel(say("scatter_y"))
            ax.set_title(dict(labels)[label], fontsize=figlib.BASE_PT, pad=3)
        else:
            part = frame[frame.panel == panel].reset_index(drop=True)
            if panel == "curve":
                for (dataset, label), g in part.groupby(["dataset", "label"], sort=False):
                    st = LABEL_STYLE[label]
                    ax.errorbar(g.x_perturbed_cells, g.estimate, yerr=[g.estimate - g.lo, g.hi - g.estimate], marker=SCREEN_MARKER[dataset],
                                markersize=3.0, linewidth=0.8, elinewidth=0.6, capsize=0, color=st["color"], mfc=st["mfc"], mew=0.7,
                                linestyle="-" if label == "all-cells" else "--")
                ax.axhline(0.0, color="#000000", linewidth=0.5)
                ax.set_xscale("log")
                ticks = [t for t in (10, 15, 20, 30, 45, 60, 100, 150) if t <= part.x_perturbed_cells.max()]
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks])
                ax.minorticks_off()
                ax.set_xlabel(say("curve_x"))
                ax.set_ylabel(say("curve_y"))
                h1 = legend_handles(ax, [(say("curve_vcc") if d == "VCC" else figlib.say("names", d),
                                          dict(marker=SCREEN_MARKER[d], color="#222222", mfc="#222222", linestyle="none", markersize=3.0)) for d in DATASETS])
                h2 = legend_handles(ax, [(text, dict(color=LABEL_STYLE[k]["color"], linestyle="-" if k == "all-cells" else "--", linewidth=0.8))
                                         for k, text in labels])
                ax.legend(handles=h1 + h2, frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
            elif panel == "association":
                group = lambda r: say("assoc_vcc_" + r.setting) if r.dataset == "VCC" else dataset_name(r)  # noqa: E731
                groups = list(dict.fromkeys(group(r) for _, r in part.iterrows()))
                dot_rows(ax, part, groups, group, list(labels), lambda r: r.label, lambda s: LABEL_STYLE[s], say("assoc_x"), above=True)
            elif panel == "own shift":
                group = lambda r: say("own_vcc", split=figlib.say("names", r.split)) if r.dataset == "VCC" else dataset_name(r)  # noqa: E731
                groups = list(dict.fromkeys(group(r) for _, r in part.iterrows()))
                series = [("G5", figlib.say("terms", "G5")), ("G6", figlib.say("terms", "G6"))]
                dot_rows(ax, part, groups, group, series, lambda r: r.setting,
                         lambda s: dict(color="#222222", mfc="#222222" if s == "G5" else "white", marker=SETTING_MARKER[s]), say("own_x"), zero=False)
            else:  # share: point values only
                part = part.assign(lo=part.estimate, hi=part.estimate)
                if main:
                    names = [say("share_row", dataset=figlib.say("names", r.dataset), setting=figlib.say("terms", r.setting)) for _, r in part.iterrows()]
                else:
                    names = [say("share_row_full", dataset=figlib.say("names", r.dataset), setting=figlib.say("terms", r.setting), level=int(r.level),
                                 version=say("version_upper") if r.version.startswith("upper bound") else say("version_deciding")) for _, r in part.iterrows()]
                figlib.interval_rows(ax, part, names, ["#222222"] * len(part), zero=False, xlabel=say("share_x"))
                ax.xaxis.set_major_locator(MaxNLocator(5))
                if not main:
                    ax.tick_params(axis="y", labelsize=figlib.MIN_PT)
        # the supplement's scatter (a) has its top y tick label in the corner, so its letter stands to the left of the tick labels
        x, where = (0.0, ax.transAxes + ScaledTranslation(-13 / 72, 0, fig.dpi_scale_trans)) if panel == "scatter all-cells" else (-0.02, ax.transAxes)
        ax.text(x, 1.02, figlib.say("names", "panel", letter=letter), transform=where, fontweight="bold", va="bottom", ha="right")
    figlib.save_figure(fig, name(args), args, data, Path(__file__).resolve())


def main() -> None:
    args = options()
    frame, scatter, data_root = panel_data(args)
    inputs = ([R2, R3, R5, S1X, CURVE, SUBSAMPLING] if args.version == "main" else [R2, R3, R5, G5_MANIFEST])  # R2: the scatter's row check
    scatter_record = {}
    if scatter is not None:
        # the per-row points stay on the data root (about 12,000 values); the record keeps their SHA-256
        out = storage_root() / "bioinf2027" / "figures" / ("%s_rows.csv" % name(args))
        out.parent.mkdir(parents=True, exist_ok=True)
        scatter.to_csv(out, index=False)
        scatter_record = {"path": "$IGA_STORAGE_ROOT/bioinf2027/figures/%s" % out.name, "rows": len(scatter),
                          "bytes": out.stat().st_size, "sha256": figlib.tablelib.sha256(out),
                          "source": "each row's line of $IGA_STORAGE_ROOT/bioinf2027/endpoints/G5/vcc_labels.csv, columns n_pool, existing_label and y"}
    data = figlib.save_data(name(args), frame, inputs, Path(__file__).resolve(),
                            extra={"version": args.version, "data_root_inputs": data_root, "scatter_points": scatter_record,
                                   "intervals": ("VCC: target-gene cluster bootstrap, 10,000 replicates, seed 20270301 (R2, R3); external association: "
                                                 "E1b's target-gene draws, seeds 20260901 and 20260902 (S1 (c)); external R3: the readouts' target-gene "
                                                 "bootstrap; the curve: S1 (b) and (c) bootstraps; the share has no interval (one fold seed)")
                                                if args.version == "main" else
                                                ("the scatter has no interval; own shift: R3's target-gene cluster bootstrap (VCC, 10,000 replicates, "
                                                 "seed 20270301) and the readouts' bootstrap (external); the share has no interval (one fold seed)"),
                                   "share_rule": "main version: the row E1a's R5 rule uses per data set and setting (bioinf2027/reports/E1a.md:83-84)",
                                   "external_labels": "association panel: 27 against 788 and all eligible cells against 788 pooled controls (G1's "
                                                      "external table and S1 (c)), not G5's 27 against 700",
                                   "bounds": "bioinf2027/factsheets/FS1_S1.md and FS3_S3.md; validation values are descriptive"})
    if not args.data_only:
        draw(frame, scatter, args, data)


if __name__ == "__main__":
    main()
