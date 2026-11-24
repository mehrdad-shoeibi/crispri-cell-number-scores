"""Figure S-a (optional): the external values of the VCC-trained heads as intervals per encoder or feature set,
configuration and screen, each with its saved status (figure and table plan, supplement; values without an
interpretation). The values of Table S-c: E1b's T and L of z, x, m and [x; m] with the read status of
bioinf2027/results/E1/derived/primary_family_read_status.csv (every row "not read"), and E2's test (ii), T and L of
scGPT, Stack and PCA-50, with the columns read and why_not_read of bioinf2027/results/E2/stats/family.csv. Stack's
values carry their marker. Usage, from the repository root:
  python bioinf2027/scripts/figures/figS_a_external_values.py [--width 86|178] [--formats pdf,png] [--data-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib  # noqa: E402

R = figlib.REPO / "bioinf2027" / "results"
E1_STATUS = R / "E1" / "derived" / "primary_family_read_status.csv"
E2_FAMILY = R / "E2" / "stats" / "family.csv"
SCREENS = ("RPE1", "K562-essential")


def panel_data() -> pd.DataFrame:
    e1 = pd.read_csv(E1_STATUS, float_precision="round_trip")
    e1["line"] = e1.index + 2
    e2 = pd.read_csv(E2_FAMILY, float_precision="round_trip")
    e2["line"] = e2.index + 2
    rows = []
    for r in e1[e1.statistic.isin(["T", "L"])].itertuples():
        if not str(r.read_status).startswith("not read"):
            raise SystemExit("primary_family_read_status.csv:%d is read; this figure shows values that are not read" % r.line)
        rows.append({"statistic": r.statistic, "configuration": r.cell, "screen": r.screen, "representation": r.arm, "estimate": r.estimate,
                     "lo": r.lo, "hi": r.hi, "holm_state": r.state, "status": r.read_status, "marker": "",
                     "source": "bioinf2027/results/E1/derived/primary_family_read_status.csv:%d" % r.line})
    for r in e2[e2.test == "ii"].itertuples():
        if bool(r.read):
            raise SystemExit("family.csv:%d of E2 is read; this figure shows values that are not read" % r.line)
        rows.append({"statistic": r.statistic, "configuration": r.cell, "screen": r.screen, "representation": figlib.FILE_NAMES[r.encoder],
                     "estimate": r.estimate, "lo": r.lo, "hi": r.hi, "holm_state": r.state, "status": "not read: %s" % r.why_not_read,
                     "marker": figlib.STACK_MARKER if r.encoder == "stack" else "", "source": "bioinf2027/results/E2/stats/family.csv:%d" % r.line})
    return pd.DataFrame(rows)


def draw(frame: pd.DataFrame, args, data: Path) -> None:
    plt = figlib.pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(args.width * figlib.MM, 120 * figlib.MM), constrained_layout=True, sharey="row")
    row_labels = {}
    order = list(figlib.REPRESENTATIONS) + ["scGPT", "Stack", "PCA-50"]
    for i, statistic in enumerate(("T", "L")):
        for j, screen in enumerate(SCREENS):
            ax = axes[i][j]
            part = frame[(frame.statistic == statistic) & (frame.screen == screen)].copy()
            part["key"] = part.representation.map(order.index)
            part = part.sort_values(["key", "configuration"]).reset_index(drop=True)
            labels = [figlib.say("figS_a", "row", representation=figlib.word(r) + (figlib.word("stack_marker") if m else ""), setting=figlib.word(c))
                      for r, c, m in zip(part.representation, part.configuration, part.marker)]
            figlib.interval_rows(ax, part, labels, [figlib.COLOURS[r] for r in part.representation],
                                 xlabel=figlib.say("figS_a", "x_" + statistic))
            if i == 0:
                ax.set_title(figlib.word(screen), fontsize=figlib.BASE_PT, pad=3)
            if j == 0:
                row_labels[i] = labels
            else:  # the same rows as the left panel, so their labels are shown once
                if labels != row_labels[i]:
                    raise SystemExit("the two screens list other rows")
                ax.tick_params(labelleft=False)
    figlib.save_figure(fig, "figS_a_external_values", args, data, Path(__file__).resolve())


def main() -> None:
    args = figlib.arguments(__doc__.split("\n")[0])
    frame = panel_data()
    data = figlib.save_data("figS_a_external_values", frame, [E1_STATUS, E2_FAMILY], Path(__file__).resolve(),
                            extra={"status_of_every_value": "not read", "read_status_as_saved": sorted(set(frame.status)),
                                   "marker": "* Stack: %s (bioinf2027/results/E2/pretraining_check.json)" % figlib.STACK_MARKER,
                                   "bounds": "bioinf2027/factsheets/FS6_no_reading.md (values without an interpretation)"})
    if not args.data_only:
        draw(frame, args, data)


if __name__ == "__main__":
    main()
