"""Tables S-e (E5's supplement, analyses 1 to 8), S-f (E5's supplement 2) and S-g (the S1 supplement): the detail
tables of the reports, saved as tables (figure and table plan, supplement; the stop of 2026-11-17 06:50: "the S1
supplement table"). Each table is the one its report prints, taken from the output of the report's own table script
(bioinf2027/scripts/make_e5_supplement_report_tables.py, make_e5_supplement_2_report_tables.py and
make_s1_supplement_report_tables.py), so the values are those of the report; no value is computed here. The report's
bold table heading is not carried over (captions are the author's); the JSON record keeps the report's table number.
Tables whose values rest on DepMap carry the column "DepMap release" (24Q4, not the current release), and rows with
Stack's Replogle values a Marker column. The inputs are the result folder of each script and every result file the
script names outside it.
Usage, from the repository root: python bioinf2027/scripts/figures/tableS_e_supplement_detail.py
"""
from __future__ import annotations

import contextlib
import io
import re
import runpy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

REPO = tablelib.REPO
SCRIPTS = REPO / "bioinf2027" / "scripts"
SOURCES = (  # (prefix of the saved tables, report table script, result folder it reads, report)
    ("tableS_e_e5_supplement", SCRIPTS / "make_e5_supplement_report_tables.py", REPO / "bioinf2027/results/E5/supplement", "bioinf2027/reports/E5.md"),
    ("tableS_f_e5_supplement_2", SCRIPTS / "make_e5_supplement_2_report_tables.py", REPO / "bioinf2027/results/E5/supplement_2", "bioinf2027/reports/E5.md"),
    ("tableS_g_s1_supplement", SCRIPTS / "make_s1_supplement_report_tables.py", REPO / "bioinf2027/results/S1_supplement", "bioinf2027/reports/S1_supplement.md"),
)
HEADING = re.compile(r"^\*\*Table ([^.]+)\.")


def printed(script: Path) -> str:
    """The report table script's standard output, run from the repository root as its docstring states."""
    buffer = io.StringIO()
    with contextlib.chdir(REPO), contextlib.redirect_stdout(buffer):
        runpy.run_path(str(script), run_name="__main__")
    return buffer.getvalue()


def tables(text: str) -> list[tuple[str, str, pd.DataFrame]]:
    """The pipe tables of a markdown text, each with the report's table number and heading from the line above it."""
    out, number, head, block = [], None, "", []
    for line in text.split("\n") + [""]:
        m = HEADING.match(line)
        if m:
            number, head = m.group(1), line
            continue
        if line.startswith("|"):
            block.append(line)
            continue
        if block:
            cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in block]
            if not set("".join(cells[1])) <= set("-: "):
                raise SystemExit("table after %r has no separator line" % number)
            if any(len(r) != len(cells[0]) for r in cells):
                raise SystemExit("table %r has rows of unequal length" % number)
            out.append((number, head, pd.DataFrame(cells[2:], columns=cells[0])))
            block = []
    return out


def slug(number: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", number.lower()).strip("_")


STACK = "possibly in Stack's pre-training data [VERIFY] (bioinf2027/results/E2/pretraining_check.json)"
DEPMAP = "DepMap 24Q4, not the current release"
DEPMAP_WORDS = ("DepMap", "F_mean", "F_K562", "fitness")


def read_by(script: Path) -> list[Path]:
    """Files a report table script names by a repository path outside its result folder."""
    return [REPO / m for m in re.findall(r"[\"'](bioinf2027/results/[^\"']+\.(?:csv|json))[\"']", script.read_text())]


def main() -> None:
    for prefix, script, folder, report in SOURCES:
        found = tables(printed(script))
        if not found or any(n is None for n, _, _ in found):
            raise SystemExit("%s printed a table without a heading" % tablelib.rel(script))
        inputs = sorted(set(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".csv", ".json")) | set(read_by(script)))
        for number, head, frame in found:
            extra = {"printed_by": {tablelib.rel(script): tablelib.sha256(script)}, "report": report, "report_table": "Table %s" % number}
            text = head + " " + " ".join(frame.columns) + " " + " ".join(frame.astype(str).values.ravel())
            if any(w in text for w in DEPMAP_WORDS):
                # a table of several sources (the HGNC counts) carries the release on DepMap's rows only
                on = frame["Source"].str.contains("DepMap") if "Source" in frame.columns else pd.Series(True, index=frame.index)
                frame = frame.assign(**{"DepMap release": [DEPMAP if o else "" for o in on]})
                extra["depmap_release"] = DEPMAP
            stack = frame.apply(lambda col: col.astype(str).str.lower().eq("stack")).any(axis=1)
            if stack.any():
                frame = frame.assign(Marker=["Stack: " + STACK if s else "" for s in stack])
                extra["marker"] = "Stack: " + STACK
            tablelib.save("%s_%s" % (prefix, slug(number)), frame, inputs, Path(__file__).resolve(), align="l" * len(frame.columns), extra=extra)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_e_supplement_detail.py")
    main()
