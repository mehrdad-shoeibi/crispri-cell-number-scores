"""Shared helpers of the figure scripts (the stop of 2026-11-17 06:50, figure and table plan; bioinf2027/GATE.md).

Every figure is drawn from saved files only, one command each, in two steps:
1. panel data: every plotted value with its file and line goes to bioinf2027/figures/<name>_data.csv, with a record
   <name>_data.json holding the inputs (each checked as the tables' inputs are, `tablelib.verify_inputs`), the code's
   SHA-256 and the outputs; this step needs no plotting library;
2. drawing: a vector PDF master (bioinf2027/figures/) and a PNG at 600 dpi ($IGA_STORAGE_ROOT/bioinf2027/figures/, its
   digest in the drawing's record), at a width of 86 mm (one column) or 178 mm (two columns), no text below 6 pt at
   final size, no caption or title that states a result (captions, titles and alt text are the author's).
The plotting library is loaded only for step 2: matplotlib 3.10.8, installed in the pipeline environment on the author's
answer to the G4 question of 2026-11-17 13:40 (bioinf2027/GATE.md; bioinf2027/figures/plotting_environment.json).
Text is set in Liberation Sans (metric-compatible with Arial), with DejaVu Sans for any glyph it lacks; PDF fonts are
embedded as TrueType.

Text: every axis label, legend entry, tick label and panel heading comes from labels.toml next to this file (say(),
word()); the drawing stops if a report symbol appears in any figure text (the author's answer to the stop of
2026-11-17 20:30). Colours: one fixed colour per encoder, representation and vector in every figure, from Paul Tol's "muted" scheme
(designed to be distinguishable under the common colour-vision deficiencies); baselines and controls in greys. Orders
are fixed and never sorted by value (no review allows a ranking).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

REPO = tablelib.REPO
FIGURES = REPO / "bioinf2027" / "figures"
MM = 1 / 25.4
WIDTHS_MM = (86, 178)
DPI = 600
MIN_PT = 6.0
BASE_PT = 7.0

# Paul Tol, "muted": indigo, cyan, teal, green, olive, sand, rose, wine, purple; pale grey for missing values.
COLOURS = {
    "Geneformer": "#332288", "scGPT": "#88CCEE", "Stack": "#44AA99", "PCA-50": "#117733", "PCA-768": "#999933",
    "x": "#CC6677", "m": "#DDCC77", "[x; m]": "#AA4499", "GenePT": "#882255",
}
COLOURS["z"] = COLOURS["Geneformer"]
GREYS = {"C1": "#777777", "C": "#777777", "S": "#AAAAAA", "N0": "#444444", "Bc": "#BBBBBB", "B0": "#222222", "TR": "#555555"}
REPRESENTATIONS = ("z", "x", "m", "[x; m]")
VECTORS = ("Geneformer", "scGPT", "GenePT")
FILE_NAMES = {"geneformer": "Geneformer", "scgpt": "scGPT", "stack": "Stack", "pca50": "PCA-50", "pca768": "PCA-768"}
STACK_MARKER = "possibly in Stack's pre-training data [VERIFY]"


LABELS_FILE = Path(__file__).resolve().with_name("labels.toml")
_LABELS: dict | None = None
# Report symbols that may not appear in any figure text (the author's answer to the stop of 2026-11-17 20:30): the
# configuration codes, the controls, the trivial predictors, the comparators and the one-letter feature names.
SYMBOL = re.compile(r"(?<![\w\-])(G[1-6]|C[1-3]|N0|B[0-4]|Bc|FN|F_mean|F_K562|TR|P-d|IU|IQR|R5|[zxmeTLS])(?![\w])|[ρΔ½ₑˣ⁻]")


def labels() -> dict:
    """The figure text, from labels.toml next to this file."""
    global _LABELS
    if _LABELS is None:
        import tomllib
        _LABELS = tomllib.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return _LABELS


READ: set = set()  # every entry of labels.toml this drawing read, recorded with the drawing
SYMBOL_KEYS = {"[x; m]": "x_m", "B0 to B4": "B0_to_B4", "P-d": "P_d", "Δ": "Delta", "D½": "D_half"}


def say(section: str, key: str, **fill) -> str:
    """One entry of labels.toml with its {placeholders} filled: a [terms] key from labels.toml, anything else from the
    script; a missing entry or placeholder stops the drawing."""
    try:
        value = labels()[section][key]
    except KeyError:
        raise SystemExit("labels.toml has no entry [%s] %r" % (section, key))
    READ.add("[%s] %s" % (section, key))
    mapping = dict(fill)
    for name in re.findall(r"\{(\w+)\}", value):
        if name not in mapping:
            if name not in labels()["terms"]:
                raise SystemExit("labels.toml [%s] %r refers to {%s}, which is neither a term nor filled by the script" % (section, key, name))
            mapping[name] = labels()["terms"][name]
            READ.add("[terms] %s" % name)
    return value.format_map(mapping)


def word(key: str) -> str:
    """The figure word for a report symbol ([terms]) or a name ([names]) of labels.toml."""
    key = SYMBOL_KEYS.get(key, key)
    for section in ("terms", "names"):
        if key in labels()[section]:
            READ.add("[%s] %s" % (section, key))
            return labels()[section][key]
    raise SystemExit("labels.toml has no entry for %r in [terms] or [names]" % key)


def wrap(text: str, width: int = 34) -> str:
    """A long tick label over two or more lines, so that the axes keep their width."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False)) if len(text) > width else text


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.02, say("names", "panel", letter=letter), transform=ax.transAxes, fontweight="bold", va="bottom", ha="right")


def check_symbols(texts) -> None:
    found = sorted({t for t in texts if t and SYMBOL.search(t)})
    if found:
        raise SystemExit("report symbols in figure text (change labels.toml): %s" % found[:6])


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--width", type=int, choices=WIDTHS_MM, default=178, help="final width in mm")
    parser.add_argument("--formats", default="pdf,png", help="comma-separated, from pdf and png")
    parser.add_argument("--data-only", action="store_true", help="write the panel data and stop")
    return parser.parse_args()


def save_data(name: str, frame: pd.DataFrame, inputs: list[Path], script: Path, extra: dict | None = None) -> Path:
    """Write <name>_data.csv and its record; every row must carry its source file and line."""
    if "source" not in frame.columns or frame["source"].isna().any() or (frame["source"] == "").any():
        raise SystemExit("every plotted value needs its source file and line")
    verified = tablelib.verify_inputs(inputs)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}_data.csv"
    frame.to_csv(out, index=False)
    record = {"figure": name, "code": {tablelib.rel(script): tablelib.sha256(script), tablelib.rel(Path(__file__)): tablelib.sha256(Path(__file__)),
                                       tablelib.rel(Path(tablelib.__file__)): tablelib.sha256(Path(tablelib.__file__))},
              "inputs": {tablelib.rel(p): tablelib.file_record(p) for p in inputs}, "inputs_verified": verified,
              "environment": tablelib.environment(), **(extra or {}),
              "outputs": {out.name: tablelib.file_record(out)}}
    (FIGURES / f"{name}_data.json").write_text(json.dumps(tablelib.strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[%s] %d plotted values written to %s" % (name, len(frame), tablelib.rel(out)), flush=True)
    return out


def pyplot():
    """The plotting library, or a stop that names the open question."""
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("not drawn: matplotlib is not installed in this environment (bioinf2027/requirements.txt pins "
                         "matplotlib 3.10.8); the panel data are written")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": BASE_PT, "axes.titlesize": BASE_PT, "axes.labelsize": BASE_PT, "xtick.labelsize": BASE_PT,
                         "ytick.labelsize": BASE_PT, "legend.fontsize": BASE_PT, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": ["Liberation Sans", "DejaVu Sans"], "svg.hashsalt": "bioinf2027"})
    return plt


def check_text(fig) -> None:
    texts = [t for t in fig.findobj(lambda o: hasattr(o, "get_fontsize") and hasattr(o, "get_text")) if t.get_text().strip() and t.get_visible()]
    small = [t.get_text() for t in texts if t.get_fontsize() < MIN_PT]
    if small:
        raise SystemExit("text below %.0f pt: %s" % (MIN_PT, small[:5]))
    check_symbols([t.get_text() for t in texts])


def data_root_figures() -> Path:
    """$IGA_STORAGE_ROOT/bioinf2027/figures, where the PNGs and other bulky drawing outputs are kept."""
    from bioinf2027 import config
    folder = Path(config.storage_root()) / "bioinf2027" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def check_png(path: Path, width_mm: int) -> None:
    """The PNG holds 600 dots per inch at its final width."""
    from PIL import Image
    with Image.open(path) as im:
        expected = round(width_mm * MM * DPI)
        if abs(im.size[0] - expected) > 2:
            raise SystemExit("%s is %d px wide, not %d (600 dpi at %d mm)" % (path.name, im.size[0], expected, width_mm))


def save_figure(fig, name: str, args: argparse.Namespace, data: Path, script: Path, extra: dict | None = None) -> None:
    check_text(fig)
    outputs = {}
    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        if fmt not in ("pdf", "png"):
            raise SystemExit("format %r: use pdf or png" % fmt)
        # the vector master stays in the repository; the 600 dpi PNG goes to the data root with its digest recorded here
        folder = FIGURES if fmt == "pdf" else data_root_figures()
        path = folder / f"{name}_{args.width}mm.{fmt}"
        fig.savefig(path, dpi=DPI, metadata={"CreationDate": None} if fmt == "pdf" else None)
        if fmt == "png":
            check_png(path, args.width)
        outputs[path.name] = tablelib.file_record(path)
    import matplotlib
    record = {"figure": name, "width_mm": args.width, "dpi_png": DPI, "minimum_text_pt": MIN_PT, "panel_data": tablelib.rel(data),
              "environment": {**tablelib.environment(), "matplotlib": matplotlib.__version__},
              "code": {tablelib.rel(script): tablelib.sha256(script), tablelib.rel(Path(__file__)): tablelib.sha256(Path(__file__)),
                       tablelib.rel(LABELS_FILE): tablelib.sha256(LABELS_FILE)},
              "labels_read": sorted(READ), **(extra or {}), "outputs": outputs}
    (FIGURES / f"{name}_{args.width}mm.json").write_text(json.dumps(tablelib.strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[%s] drawn at %d mm: %s" % (name, args.width, ", ".join(outputs)), flush=True)


def interval_rows(ax, frame: pd.DataFrame, labels: list[str], colours: list[str], zero: bool = True, xlabel: str = "") -> None:
    """Points with their 95 % intervals, one row per entry, top to bottom in the given order."""
    y = list(range(len(frame)))[::-1]
    for yy, (_, r), colour in zip(y, frame.iterrows(), colours):
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=colour, linewidth=1.0, solid_capstyle="butt")
        ax.plot([r["estimate"]], [yy], marker="o", markersize=3.0, color=colour, linestyle="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if zero:
        ax.axvline(0.0, color="#000000", linewidth=0.5)
    ax.set_xlabel(xlabel)
