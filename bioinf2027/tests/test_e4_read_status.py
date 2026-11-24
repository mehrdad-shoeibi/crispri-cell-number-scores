"""Guard on E4's read status (bioinf2027/reviews/E4_results_r1.md, non-blocking 9).

An F whose arm fails E1b's VCC precondition in its cell is computed and reported, not read
(bioinf2027/plans/E4.md, Section 11.2, item 3). family.csv carries that status in its `read` column and
readings.csv in its `statement` column. These tests check both against bioinf2027/results/E1/preconditions.csv,
for the family pass and, once it exists, the full pass, and check that no fact sheet, figure or table cites
E4's family or readings without the status.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / "results" / "E4"
PASSES = [p for p in (E4 / "family_pass", E4) if (p / "family.csv").exists()]
NOT_READ = "computed and reported, not read"
TEXT_SUFFIXES = (".py", ".ipynb", ".R", ".r", ".sh", ".md", ".tex", ".csv", ".json", ".txt", ".tsv", ".svg", ".html")


def passes_e1_precondition() -> dict:
    """(cell, arm) -> True when the arm passes E1b's VCC precondition on the rebuilt label."""
    pre = pd.read_csv(ROOT / "results" / "E1" / "preconditions.csv")
    pre = pre[pre["label"] == "rebuilt"]
    return {(c, a): bool(p) for c, a, p in zip(pre["cell"], pre["arm"], pre["passes"])}


@pytest.mark.parametrize("folder", PASSES, ids=lambda p: str(p.relative_to(ROOT)))
def test_family_rows_of_failing_arms_are_not_read(folder):
    flags = passes_e1_precondition()
    family = pd.read_csv(folder / "family.csv")
    assert len(family) == 16
    for row in family.itertuples():
        assert (row.cell, row.arm) in flags
        assert bool(row.precondition_E1_passed) == flags[(row.cell, row.arm)]
        if not flags[(row.cell, row.arm)]:
            assert str(row.read).startswith("no: the arm fails E1's precondition"), (row.cell, row.screen, row.arm)


@pytest.mark.parametrize("folder", PASSES, ids=lambda p: str(p.relative_to(ROOT)))
def test_readings_of_failing_arms_are_not_read(folder):
    flags = passes_e1_precondition()
    readings = pd.read_csv(folder / "readings.csv")
    for row in readings.itertuples():
        if not (flags[("G5", row.arm)] and flags[("G6", row.arm)]):
            assert str(row.statement).startswith(NOT_READ), (row.screen, row.arm)


def output_files():
    for name in ("factsheets", "figures", "tables"):
        if (ROOT / name).exists():
            yield from (p for p in sorted((ROOT / name).rglob("*")) if p.is_file() and p.suffix in TEXT_SUFFIXES)


def test_outputs_citing_e4_family_or_readings_carry_the_status():
    offenders = []
    for path in output_files():
        text = path.read_text(errors="replace")
        cites = any(f"results/E4/{sub}{name}" in text for sub in ("", "family_pass/") for name in ("family.csv", "readings.csv"))
        if cites and NOT_READ not in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "these cite E4's family or readings without '%s': %s" % (NOT_READ, offenders)
