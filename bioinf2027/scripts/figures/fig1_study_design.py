"""Fig. 1: how the two labels are built, and which data set each analysis uses; a plain SVG with editable text and a
PDF drawn from that SVG (the author's answer to the stop of 2026-11-17 20:30, bioinf2027/GATE.md: a new form for rule
O6, two panels, no lists of encoders, controls or trivial predictors, no columns of text boxes joined by arrows, no
numbered stack and no result).

- (a) For one batch, cells as dots: two example rows, one with few and one with many perturbed cells, and the batch's
  control cells. The all-cells label uses every perturbed cell of the row against every control cell of the batch; the
  size-fixed label draws a fixed number of perturbed and control cells, repeatedly, and takes the median over the
  draws (one draw is drawn, the text says how often it is repeated, so that no numbered stack appears). The numbers
  of the constructions come from the plan lines the fact sheets cite (27 or 10 perturbed cells, 700 or 788 control
  cells, 25 draws), each checked against its line; the counts of dots are a drawing, not data.
- (b) A grid of the three data sets (with their analysed counts) by the five analyses of the Results; a filled mark
  where an analysis uses a data set for its claim, an open mark where it uses it descriptively only. Every count and
  mark is listed with its file and line in bioinf2027/figures/fig1_study_design_data.csv; the lines are found by the
  text they hold, and the script stops if a text is missing.

The SVG is written in millimetres from five kinds of element only (rect, line, polygon, circle, text; one text element
per line). The PDF and the 600 dpi PNG are drawn from that SVG file alone: the file is parsed back and each element
rendered with the plotting library; an element of any other kind stops the script (convert an SVG saved by an editor
with the editor itself). All text comes from labels.toml (sections fig1 and names). Liberation Sans, metric-compatible
with Arial, which the SVG names first; no text below 6 pt.
Usage, from the repository root:
  python bioinf2027/scripts/figures/fig1_study_design.py [--width 178] [--formats svg,pdf,png] [--data-only]
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

B = figlib.REPO / "bioinf2027"
F = B / "factsheets"
RETENTION = B / "diag" / "D3_vcc_retention.csv"
COUNTS = B / "diag" / "D3_cell_counts.csv"
DESIGN = B / "plans" / "E1_design.md"
PT = 25.4 / 72  # millimetres per point
FONT = "Liberation Sans"
SVG_FONT = "Arial, Liberation Sans, Helvetica, sans-serif"
PERTURBED, CONTROL, RING = "#555555", "#C8C8C8", "#000000"

# Design numbers, each checked against the text of its cited lines.
DESIGN_NUMBERS = (  # the configuration table of the E1 design note (G1, G2, G5, G6) and its draws
    ("27", "perturbed cells of the size-fixed label in G1 and G5", DESIGN, 1169, 1169, "| G1 | 27 | 788 pooled |"),
    ("10", "perturbed cells of the size-fixed label in G2 and G6", DESIGN, 1170, 1170, "| G2 | 10 | 788 pooled |"),
    ("700", "control cells of the size-fixed label in G5 and G6 (VCC: of the row's batch)", DESIGN, 1173, 1173, "| G5 | 27 | 700 of the row's batch |"),
    ("788", "pooled control cells of the size-fixed label in G1 and G2", DESIGN, 1170, 1170, "| 10 | 788 pooled | 788 pooled |"),
    ("25", "draws per row or construct, median across draws", DESIGN, 201, 201, "Twenty-five draws per row"),
)
# Marks of panel (b): (analysis, data set, kind, fact sheet, the text of the line that states it).
MARKS = (
    (1, "VCC", "used", "FS1_S1.md", "- Rows per cell, all splits: G1 and G5 6,077 rows"),
    (1, "RPE1", "used", "FS1_S1.md", "- (c) Replogle, the label with all eligible perturbed cells against 788 pooled controls"),
    (1, "K562-essential", "used", "FS1_S1.md", "- (c) Replogle, the label with all eligible perturbed cells against 788 pooled controls"),
    (2, "VCC", "used", "FS2_S2.md", "G6 holds the 3,972 VCC test rows and 93 target genes"),
    (3, "VCC", "used", "FS3_S3.md", "- VCC G6: training +0.953"),
    (3, "RPE1", "used", "FS3_S3.md", "- RPE1: Spearman of the label with the negated own delta"),
    (3, "K562-essential", "used", "FS3_S3.md", "- K562-essential: +0.976 [+0.973, +0.978]"),
    (4, "RPE1", "used", "FS4_S4.md", "- Target-screen constructs: 1,932 (RPE1) and 1,903 (K562-essential)"),
    (4, "K562-essential", "used", "FS4_S4.md", "- Target-screen constructs: 1,932 (RPE1) and 1,903 (K562-essential)"),
    (5, "RPE1", "used", "FS5_S5.md", "- Rows: Replogle RPE1 and K562-essential essential-gene CRISPRi screens"),
    (5, "K562-essential", "used", "FS5_S5.md", "- Rows: Replogle RPE1 and K562-essential essential-gene CRISPRi screens"),
    (5, "VCC", "descriptive", "FS5_S5.md", "- No VCC value is interpreted"),
)
DATASET_KEYS = (("VCC", "dataset_vcc"), ("RPE1", "dataset_rpe1"), ("K562-essential", "dataset_k562"))


def options():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--width", type=int, choices=(178,), default=178, help="final width in mm (two columns)")
    parser.add_argument("--formats", default="svg,pdf,png")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def say(key: str, **fill) -> str:
    return figlib.say("fig1", key, **fill)


def find_line(path: Path, text: str) -> int:
    hits = [i + 1 for i, line in enumerate(path.read_text().split("\n")) if text in line]
    if len(hits) != 1:
        raise SystemExit("%d lines of %s hold %r" % (len(hits), figlib.tablelib.rel(path), text))
    return hits[0]


def panel_data() -> tuple[pd.DataFrame, dict]:
    rows = []
    for number, what, path, first, last, text in DESIGN_NUMBERS:
        held = " ".join(" ".join(path.read_text().split("\n")[first - 1:last]).split())
        if text not in held:
            raise SystemExit("%s:%d-%d does not hold %r" % (figlib.tablelib.rel(path), first, last, text))
        rows.append({"panel": "a", "item": what, "value": number, "source": "%s:%d-%d" % (figlib.tablelib.rel(path), first, last)})
    retention = pd.read_csv(RETENTION)
    retention["line"] = retention.index + 2
    genes = {}
    for split in ("train", "val", "test"):
        r = retention[(retention.split == split) & (retention.perturbed_cell_floor == 10)].iloc[0]
        genes[split] = int(r.target_genes_retained)
        rows.append({"panel": "b", "item": "VCC %s target genes analysed (at least 10 perturbed cells)" % split, "value": str(genes[split]),
                     "source": "%s:%d" % (figlib.tablelib.rel(RETENTION), r.line)})
    counts = pd.read_csv(COUNTS)
    counts["line"] = counts.index + 2
    constructs = {}
    for screen in ("RPE1", "K562-essential"):
        r = counts[counts.screen == screen].iloc[0]
        constructs[screen] = int(r.n_eligible)
        rows.append({"panel": "b", "item": "%s constructs with at least 27 perturbed cells" % screen, "value": "{:,}".format(constructs[screen]),
                     "source": "%s:%d" % (figlib.tablelib.rel(COUNTS), r.line)})
    for analysis, dataset, kind, sheet, text in MARKS:
        rows.append({"panel": "b", "item": "analysis %d uses %s (%s)" % (analysis, dataset, kind), "value": kind,
                     "source": "bioinf2027/factsheets/%s:%d" % (sheet, find_line(F / sheet, text))})
    design = {number: number for number, *_ in DESIGN_NUMBERS}  # each checked against its cited line above
    return pd.DataFrame(rows), {"genes": genes, "constructs": constructs, "design": design}


def text_width_mm(text: str, pt: float, bold: bool = False) -> float:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    prop = FontProperties(family=[FONT, "DejaVu Sans"], weight="bold" if bold else "normal", size=pt)
    return TextPath((0, 0), text, prop=prop).get_extents().width * PT


def wrap(text: str, width_mm: float, pt: float) -> list[str]:
    lines, line = [], ""
    for w in text.split(" "):
        trial = (line + " " + w).strip()
        if line and text_width_mm(trial, pt) > width_mm:
            lines.append(line)
            line = w
        else:
            line = trial
    return lines + [line]


class Svg:
    """A plain SVG in millimetres: rect, line, polygon, circle and one text element per line."""

    def __init__(self, width: float, height: float):
        self.width, self.height, self.items, self.texts = width, height, [], []

    def rect(self, x, y, w, h, fill="none", stroke="#000000", sw=0.25, rx=0.0):
        self.items.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="%.2f" fill="%s" stroke="%s" stroke-width="%.2f"/>'
                          % (x, y, w, h, rx, fill, stroke, sw))

    def line(self, x1, y1, x2, y2, stroke="#000000", sw=0.25):
        self.items.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="%.2f"/>' % (x1, y1, x2, y2, stroke, sw))

    def circle(self, cx, cy, r, fill, stroke="none", sw=0.0):
        self.items.append('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="%s" stroke="%s" stroke-width="%.2f"/>' % (cx, cy, r, fill, stroke, sw))

    def text(self, x, y, s, pt=figlib.BASE_PT, bold=False, anchor="start", fill="#000000"):
        if pt < figlib.MIN_PT:
            raise SystemExit("text below %.0f pt: %r" % (figlib.MIN_PT, s))
        self.texts.append(s)
        esc = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.items.append('<text x="%.2f" y="%.2f" font-family="%s" font-size="%.3f" data-pt="%.1f"%s text-anchor="%s" fill="%s">%s</text>'
                          % (x, y, SVG_FONT, pt * PT, pt, ' font-weight="bold"' if bold else "", anchor, fill, esc))

    def write(self, path: Path) -> None:
        figlib.check_symbols(self.texts)
        head = ('<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="%.0fmm" height="%.0fmm" '
                'viewBox="0 0 %.0f %.0f">\n' % (self.width, self.height, self.width, self.height))
        path.write_text(head + "\n".join("  " + i for i in self.items) + "\n</svg>\n", encoding="utf-8")


R_DOT, STEP = 0.75, 2.6


def cells(svg: Svg, x0: float, y0: float, columns: int, rows: int, n: int, fill: str, used=None) -> float:
    """A grid of n dots; `used` is the set of dot indices drawn with a ring. Returns the y below the grid."""
    for k in range(n):
        cx, cy = x0 + (k % columns) * STEP, y0 + (k // columns) * STEP
        ring = used is not None and k in used
        svg.circle(cx, cy, R_DOT, fill, RING if ring else "none", 0.3 if ring else 0.0)
    return y0 + ((n - 1) // columns) * STEP


def batch(svg: Svg, x: float, y: float, used_ctrl, used_few, used_many, labels: bool) -> None:
    """The batch's control cells and two rows of perturbed cells, from the top-left corner (x, y)."""
    small = 6.5
    if labels:
        svg.text(x, y, say("controls"), pt=small)
    cells(svg, x + 1, y + 2.5, 12, 3, 36, CONTROL, used_ctrl)
    if labels:
        svg.text(x, y + 13.5, say("row_few"), pt=small)
    cells(svg, x + 1, y + 16, 12, 1, 11, PERTURBED, used_few)
    if labels:
        svg.text(x, y + 22, say("row_many"), pt=small)
    cells(svg, x + 1, y + 24.5, 12, 3, 33, PERTURBED, used_many)


def build(width: float, counts: dict) -> Svg:
    svg = Svg(width, 112.0)
    pt, small = figlib.BASE_PT, 6.5
    letter = lambda x, y, k: svg.text(x, y, figlib.say("names", "panel", letter=k), pt=pt, bold=True)  # noqa: E731
    # Panel (a): the data, the all-cells label and the size-fixed label, side by side, without boxes or arrows.
    letter(1.0, 4.5, "a")
    batch(svg, 5.0, 13.0, None, None, None, labels=True)
    xa = 62.0
    svg.text(xa, 6.5, say("all_cells_head"), pt=pt, bold=True)
    batch(svg, xa, 13.0, set(range(36)), set(range(11)), set(range(33)), labels=False)
    svg.text(xa, 49.4, say("all_cells_text_1"), pt=small)
    svg.text(xa, 52.3, say("all_cells_text_2"), pt=small)
    xs = 112.0
    svg.text(xs, 6.5, say("size_fixed_head"), pt=pt, bold=True)
    # three draws, the later ones behind the first, each a frame of the same batch
    front = 12.0  # one draw, framed; the text below says how often it is repeated
    svg.rect(xs - 1.5, front, 36.0, 33.5, fill="#FFFFFF", stroke="#555555", sw=0.3, rx=0.8)
    used_ctrl = {k for k in range(36) if k % 7 != 3}           # most of the batch's controls
    used_few = {k for k in range(11) if k != 4}                 # the same number (10) of perturbed cells from each row
    used_many = set(sorted(k for k in range(33) if k % 3 == 1)[:10])
    batch(svg, xs, front + 1.0, used_ctrl, used_few, used_many, labels=False)
    svg.text(xs + 34.5 - 1.5, front + 32.3, say("one_draw"), pt=small, anchor="end", fill="#555555")
    d = counts["design"]
    svg.text(xs, 49.4, say("size_fixed_text_1", n_pert_1=d["27"], n_pert_2=d["10"]), pt=small)
    svg.text(xs, 52.3, say("size_fixed_text_2", n_ctrl_1=d["700"], n_ctrl_2=d["788"]), pt=small)
    svg.text(xs, 55.2, say("size_fixed_text_3", draws=d["25"]), pt=small)
    # legend of panel (a)
    ly = 55.2
    svg.circle(6.0, ly - 0.9, R_DOT, PERTURBED)
    svg.text(7.6, ly, say("legend_perturbed"), pt=small)
    svg.circle(28.0, ly - 0.9, R_DOT, CONTROL)
    svg.text(29.6, ly, say("legend_control"), pt=small)
    svg.circle(46.0, ly - 0.9, R_DOT, CONTROL, RING, 0.3)
    svg.text(47.6, ly, say("legend_used"), pt=small)
    # Panel (b): data sets by analyses.
    top = 63.5
    letter(1.0, top + 2.5, "b")
    x_rows, x0, col = 5.0, 62.0, (width - 62.0 - 2.0) / 5
    header_lines = [wrap(say("analysis_%d" % k), col - 2.0, small) for k in range(1, 6)]
    depth = max(len(h) for h in header_lines)
    for k, lines in enumerate(header_lines):
        cx = x0 + (k + 0.5) * col
        for j, line in enumerate(lines):
            svg.text(cx, top + 2.5 + (depth - len(lines) + j) * 2.9, line, pt=small, anchor="middle")
    grid_top = top + 2.5 + depth * 2.9
    svg.line(x_rows, grid_top, width - 2.0, grid_top, stroke="#000000", sw=0.25)
    row_h = 9.0
    marks = {(a, d): kind for a, d, kind, _, _ in MARKS}
    g = counts["genes"]
    count_text = {"VCC": say("count_vcc", train=g["train"], val=g["val"], test=g["test"]),
                  "RPE1": say("count_screen", n="{:,}".format(counts["constructs"]["RPE1"])),
                  "K562-essential": say("count_screen", n="{:,}".format(counts["constructs"]["K562-essential"]))}
    for i, (dataset, key) in enumerate(DATASET_KEYS):
        y = grid_top + i * row_h
        svg.text(x_rows, y + 4.0, say(key), pt=pt, bold=True)
        svg.text(x_rows, y + 7.0, count_text[dataset], pt=small)
        for k in range(1, 6):
            kind = marks.get((k, dataset))
            cx = x0 + (k - 0.5) * col
            if kind == "used":
                svg.circle(cx, y + row_h / 2, 1.4, "#222222")
            elif kind == "descriptive":
                svg.circle(cx, y + row_h / 2, 1.25, "#FFFFFF", "#222222", 0.35)
        svg.line(x_rows, y + row_h, width - 2.0, y + row_h, stroke="#BBBBBB", sw=0.2)
    ly = grid_top + 3 * row_h + 4.5
    svg.circle(x0 + 1.0, ly - 0.9, 1.4, "#222222")
    svg.text(x0 + 3.2, ly, say("mark_used"), pt=small)
    svg.circle(x0 + 22.0, ly - 0.9, 1.25, "#FFFFFF", "#222222", 0.35)
    svg.text(x0 + 24.2, ly, say("mark_descriptive"), pt=small)
    svg.height = float(int(ly + 2.5))
    return svg


def render(svg_path: Path, plt):
    """Draw the SVG file's elements with the plotting library; stop on any element of another kind."""
    from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
    root = ET.parse(svg_path).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    vb = [float(v) for v in root.get("viewBox").split()]
    fig = plt.figure(figsize=(vb[2] / 25.4, vb[3] / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, vb[2])
    ax.set_ylim(vb[3], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    counts = {"rect": 0, "line": 0, "polygon": 0, "circle": 0, "text": 0}
    for el in root:
        tag = el.tag.replace(ns, "")
        if tag not in counts:
            raise SystemExit("%s holds a <%s> element, which this renderer does not draw" % (svg_path.name, tag))
        counts[tag] += 1
        stroke = el.get("stroke", "none")
        sw = float(el.get("stroke-width", "0")) / PT
        if tag == "rect":
            x, y, w, h, rx = (float(el.get(k)) for k in ("x", "y", "width", "height", "rx"))
            fill = el.get("fill")
            face = "none" if fill == "none" else fill
            if rx > 0:
                ax.add_patch(FancyBboxPatch((x + rx, y + rx), w - 2 * rx, h - 2 * rx, boxstyle="round,pad=%.3f" % rx, facecolor=face,
                                            edgecolor="none" if stroke == "none" else stroke, linewidth=sw))
            else:
                ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="none" if stroke == "none" else stroke, linewidth=sw))
        elif tag == "line":
            x1, y1, x2, y2 = (float(el.get(k)) for k in ("x1", "y1", "x2", "y2"))
            ax.plot([x1, x2], [y1, y2], color=stroke, linewidth=sw, solid_capstyle="butt")
        elif tag == "polygon":
            pts = [tuple(float(v) for v in p.split(",")) for p in el.get("points").split()]
            ax.add_patch(Polygon(pts, closed=True, facecolor=el.get("fill"), edgecolor="none"))
        elif tag == "circle":
            cx, cy, r = (float(el.get(k)) for k in ("cx", "cy", "r"))
            ax.add_patch(Circle((cx, cy), r, facecolor=el.get("fill"), edgecolor="none" if stroke == "none" else stroke, linewidth=sw))
        else:
            ha = {"start": "left", "middle": "center", "end": "right"}[el.get("text-anchor", "start")]
            ax.text(float(el.get("x")), float(el.get("y")), el.text or "", fontsize=float(el.get("font-size")) / PT, ha=ha, va="baseline",
                    fontweight=el.get("font-weight", "normal"), color=el.get("fill", "#000000"), family=[FONT, "DejaVu Sans"])
    return fig, counts


def main() -> None:
    args = options()
    frame, counts = panel_data()
    sheets = sorted({F / sheet for _, _, _, sheet, _ in MARKS})
    data = figlib.save_data("fig1_study_design", frame, [RETENTION, COUNTS, DESIGN] + sheets, Path(__file__).resolve(),
                            extra={"content": "design only, no result; the dots are a drawing, their counts are not data",
                                   "svg": "plain SVG, millimetre user units, one text element per line, font %s" % SVG_FONT,
                                   "marks": "filled: the analysis uses the data set for its claim; open: descriptive only"})
    if args.data_only:
        return
    plt = figlib.pyplot()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    svg_path = figlib.FIGURES / ("fig1_study_design_%dmm.svg" % args.width)
    build(float(args.width), counts).write(svg_path)
    fig, element_counts = render(svg_path, plt)
    drawn = argparse.Namespace(width=args.width, formats=",".join(f for f in formats if f != "svg") or "pdf")
    figlib.save_figure(fig, "fig1_study_design", drawn, data, Path(__file__).resolve(),
                       extra={"svg": {svg_path.name: figlib.tablelib.file_record(svg_path)}, "svg_elements": element_counts,
                              "drawn_from": "the PDF and PNG are drawn from the SVG file by this script's renderer"})


if __name__ == "__main__":
    main()
