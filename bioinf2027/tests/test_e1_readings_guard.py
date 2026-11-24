"""Guard on E1b's L readings (gate G3, 2026-11-12, item 3; bioinf2027/plans/E1b_posthoc.md, Section 7).

No fact sheet, figure or table script may take an L reading from bioinf2027/results/E1/readings.csv;
they use bioinf2027/results/E1/derived/readings_read_status.csv, which carries the read status of
bioinf2027/plans/E1_design.md, Section 21.1, item 2.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e1b_read_status as rs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "results" / "E1"
WRITERS = {"make_e1b_stats.py", "make_e1b_read_status.py"}
# results/E1/readings.csv as a path; the E1 results folder joined with "readings.csv"; os.path.join(..., "E1", "readings.csv")
REFERENCE = re.compile(r"results/E1/readings\.csv"
                       r"|[\"']?E1[\"']?\s*\)?\s*/\s*[\"']readings\.csv[\"']"
                       r"|[\"']E1[\"']\s*,\s*[\"']readings\.csv[\"']")
# a variable that holds the E1 results folder, e.g. RESULTS = REPO / "bioinf2027" / "results" / "E1"
HOLDER = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*[\"']results[\"']\s*/\s*[\"']E1[\"']\s*$|^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*results/E1[\"']", re.M)
TEXT_SUFFIXES = (".py", ".ipynb", ".R", ".r", ".sh", ".md", ".tex", ".csv", ".json", ".txt", ".tsv", ".svg", ".html")


def refers_to_e1_readings(text: str) -> bool:
    if REFERENCE.search(text):
        return True
    for match in HOLDER.finditer(text):
        name = match.group(1) or match.group(2)
        if re.search(r"\b%s\s*/\s*[\"']readings\.csv[\"']" % re.escape(name), text):
            return True
    return False


def not_read_texts() -> set:
    """The L readings of arms whose L is not read, whole and clause by clause (second results review, non-blocking 6)."""
    readings = pd.read_csv(E1 / "readings.csv")
    failing = readings[~readings["precondition_passed_in_both_cells"].astype(bool)]
    texts = set(failing["L_reading"].astype(str))
    return texts | {clause.strip() for text in texts for clause in text.split(";") if len(clause.strip()) > 20}


def test_derived_file_matches_readings_and_reads_no_failing_arm():
    readings = pd.read_csv(E1 / "readings.csv")
    derived = pd.read_csv(ROOT / "results" / "E1" / "derived" / "readings_read_status.csv")
    assert list(derived[["screen", "arm"]].itertuples(index=False)) == list(readings[["screen", "arm"]].itertuples(index=False))
    for column in readings.columns.drop("L_reading"):
        pd.testing.assert_series_equal(derived[column], readings[column], check_names=True)
    passed = readings["precondition_passed_in_both_cells"].astype(bool)
    for p, status, text, original in zip(passed, derived["L_read_status"], derived["L_reading"], readings["L_reading"]):
        if p:
            assert status == rs.READ and text == original
        else:
            assert status == rs.NOT_READ and text == rs.NOT_READ


def test_derivation_is_reproducible():
    readings = pd.read_csv(E1 / "readings.csv")
    preconditions = pd.read_csv(E1 / "preconditions.csv", float_precision="round_trip")
    derived = pd.read_csv(ROOT / "results" / "E1" / "derived" / "readings_read_status.csv")
    pd.testing.assert_frame_equal(rs.derive(readings, preconditions).reset_index(drop=True), derived, check_dtype=False)


def test_no_failing_arm_is_read_in_a_constructed_case():
    readings = pd.DataFrame({"screen": ["RPE1", "RPE1"], "arm": ["z", "x"], "precondition_passed_in_both_cells": [True, False],
                             "L_reading": ["label matching raises the arm's contrast over C1", "label matching raises the arm's contrast over C1"]})
    preconditions = pd.DataFrame({"cell": ["G5", "G6", "G5", "G6"], "label": ["rebuilt"] * 4, "arm": ["z", "z", "x", "x"], "passes": [True, True, True, False]})
    derived = rs.derive(readings, preconditions)
    assert derived["L_read_status"].tolist() == [rs.READ, rs.NOT_READ]
    assert derived["L_reading"].tolist() == ["label matching raises the arm's contrast over C1", rs.NOT_READ]


def test_reference_pattern():
    assert refers_to_e1_readings('E1 / "readings.csv"')
    assert refers_to_e1_readings('RESULTS / "E1" / "readings.csv"')
    assert refers_to_e1_readings("see bioinf2027/results/E1/readings.csv")
    assert refers_to_e1_readings('os.path.join(root, "results", "E1", "readings.csv")')
    assert refers_to_e1_readings('RESULTS = REPO / "bioinf2027" / "results" / "E1"\nframe = pd.read_csv(RESULTS / "readings.csv")')
    assert not refers_to_e1_readings('OUT = REPO / "bioinf2027" / "results" / "E4"\nframe.to_csv(OUT / "readings.csv")')
    assert not refers_to_e1_readings('OUT / "readings.csv"')
    assert not refers_to_e1_readings("bioinf2027/results/E1/derived/readings_read_status.csv")


def output_files():
    for folder in ("factsheets", "figures", "tables"):
        if (ROOT / folder).exists():
            yield from (p for p in sorted((ROOT / folder).rglob("*")) if p.is_file() and p.suffix in TEXT_SUFFIXES)


def test_no_script_fact_sheet_figure_or_table_reads_e1_readings():
    offenders = []
    for path in sorted((ROOT / "scripts").rglob("*")):
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.name not in WRITERS and refers_to_e1_readings(path.read_text(errors="replace")):
            offenders.append(str(path.relative_to(ROOT)))
    for path in output_files():
        if refers_to_e1_readings(path.read_text(errors="replace")):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, "these take E1b readings from readings.csv, not from results/E1/derived/readings_read_status.csv: %s" % offenders


def test_no_fact_sheet_figure_or_table_carries_a_not_read_l_reading():
    texts = not_read_texts()
    offenders = [str(p.relative_to(ROOT)) for p in output_files() if any(t in p.read_text(errors="replace") for t in texts)]
    assert not offenders, "these carry the L reading of an arm whose L is not read: %s" % offenders


def test_not_read_texts_include_each_clause():
    texts = not_read_texts()
    assert "the failure is not measurably changed by label matching" in texts
    assert "no change from label matching is resolved" in texts


def test_family_read_status_reads_no_statistic_of_a_failing_arm():
    family = pd.read_csv(E1 / "primary_family.csv", float_precision="round_trip")
    preconditions = pd.read_csv(E1 / "preconditions.csv", float_precision="round_trip")
    derived = pd.read_csv(ROOT / "results" / "E1" / "derived" / "primary_family_read_status.csv")
    pd.testing.assert_frame_equal(rs.derive_family(family, preconditions).reset_index(drop=True), derived, check_dtype=False)
    flags = rs.precondition_flags(preconditions)
    for row in derived.itertuples():
        if row.read_under_rule_3:
            assert row.statistic in ("T", "L", "R", "R-") and flags[row.arm] and bool(row.read)
        else:
            assert row.read_status in (rs.FAMILY_NOT_READ, rs.FAMILY_NOT_ANSWERABLE)


def test_family_read_status_in_a_constructed_case():
    family = pd.DataFrame({"cell": ["G5"] * 3, "screen": ["RPE1"] * 3, "statistic": ["T", "L", "R"], "arm": ["z", "x", "z"],
                           "estimate": [0.1, 0.2, 0.0], "lo": [0.05, 0.1, -0.1], "hi": [0.15, 0.3, 0.1], "state": ["Holm-positive"] * 2 + ["inconclusive"],
                           "read": [True, True, False]})
    preconditions = pd.DataFrame({"cell": ["G5", "G6", "G5", "G6"], "label": ["rebuilt"] * 4, "arm": ["z", "z", "x", "x"], "passes": [True, True, True, False]})
    derived = rs.derive_family(family, preconditions)
    assert derived["read_under_rule_3"].tolist() == [True, False, False]
    assert derived["read_status"].tolist() == [rs.READ, rs.FAMILY_NOT_READ, rs.FAMILY_NOT_ANSWERABLE]


def test_fact_sheets_figures_and_tables_citing_the_family_cite_its_read_status():
    offenders = [str(p.relative_to(ROOT)) for p in output_files()
                 if "primary_family.csv" in p.read_text(errors="replace") and "primary_family_read_status.csv" not in p.read_text(errors="replace")]
    assert not offenders, "these cite primary_family.csv without results/E1/derived/primary_family_read_status.csv: %s" % offenders
