"""Guard on D2's read status (bioinf2027/reviews/D2_results_r1.md, non-blocking 4).

family.csv's `read` column encodes only E1's R5 gate and an uninformative cell. Under reading (f)
(bioinf2027/plans/D2.md, Section 11.1) no VCC statistic of a cell is read where z fails its VCC
precondition. The derived status in bioinf2027/results/D2/supplement/family_read_status.csv carries that.
These tests check it against bioinf2027/results/E1/preconditions.csv and family.csv, check readings.csv's
VCC reading, and check that no fact sheet, figure or table cites D2's family without the status.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
D2 = ROOT / "results" / "D2"
STATUS = D2 / "supplement" / "family_read_status.csv"
TEXT_SUFFIXES = (".py", ".ipynb", ".R", ".r", ".sh", ".md", ".tex", ".csv", ".json", ".txt", ".tsv", ".svg", ".html")


def z_passes() -> dict:
    pre = pd.read_csv(ROOT / "results" / "E1" / "preconditions.csv")
    pre = pre[(pre["label"] == "rebuilt") & (pre["arm"] == "z")]
    return {c: bool(p) for c, p in zip(pre["cell"], pre["passes"])}


@pytest.mark.skipif(not STATUS.exists(), reason="the D2 supplement has not been written")
def test_derived_status_follows_reading_f_and_the_r5_gate():
    passes = z_passes()
    family = pd.read_csv(D2 / "family.csv")
    status = pd.read_csv(STATUS)
    assert len(status) == len(family) == 18
    assert (status[["cell", "scope", "statistic"]].to_numpy() == family[["cell", "scope", "statistic"]].to_numpy()).all()
    for f, s in zip(family.itertuples(), status.itertuples()):
        if f.scope == "VCC test" and not passes[f.cell]:
            assert s.read_status.startswith("not read: reading (f)"), (f.cell, f.statistic)
        elif f.statistic == "R(z-)" and not f.r5_gate_passed:
            assert s.read_status.startswith("not read: E1's R5 gate fails"), (f.cell, f.scope)
        else:
            assert s.read_status.startswith("read: "), (f.cell, f.scope, f.statistic)


SUPPLEMENT_TABLES = sorted((D2 / "supplement").glob("*.csv")) if (D2 / "supplement").exists() else []
TAGS = ("secondary, unadjusted", "post hoc")


@pytest.mark.parametrize("path", [p for p in SUPPLEMENT_TABLES if p.name != "family_read_status.csv"], ids=lambda p: p.name)
def test_every_supplement_row_carries_its_adjustment(path):
    """Results review r2, non-blocking 8: no supplement interval travels without its label.

    Every table with intervals (columns lo and hi) must carry an adjustment label on every row; a table
    of descriptive medians without intervals (manipulation_summary.csv) is held to it only if it has the
    column.
    """
    table = pd.read_csv(path)
    if "adjustment" not in table.columns:
        assert not {"lo", "hi"} <= set(table.columns), path.name
        return
    labels = table["adjustment"].astype(str)
    assert labels.map(lambda s: any(tag in s for tag in TAGS)).all(), path.name


@pytest.mark.skipif(not STATUS.exists(), reason="the D2 supplement has not been written")
def test_derived_status_has_a_status_on_every_row():
    status = pd.read_csv(STATUS)["read_status"].astype(str)
    assert status.map(lambda s: s.startswith(("read: ", "not read: "))).all()


def test_vcc_reading_is_f_where_z_fails_in_both_cells():
    passes = z_passes()
    readings = pd.read_csv(D2 / "readings.csv")
    vcc = readings[readings["scope"] == "VCC"].iloc[0]
    if not passes["G5"] and not passes["G6"]:
        assert vcc["reading"] == "(f)"
        assert str(vcc["statement"]).startswith("no D2 reading")


def output_files():
    for name in ("factsheets", "figures", "tables"):
        if (ROOT / name).exists():
            yield from (p for p in sorted((ROOT / name).rglob("*")) if p.is_file() and p.suffix in TEXT_SUFFIXES)


def test_outputs_citing_d2_family_carry_the_status():
    offenders = []
    for path in output_files():
        text = path.read_text(errors="replace")
        if "results/D2/family.csv" in text and "not read" not in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "these cite D2's family without its read status: %s" % offenders
