"""Table S-i: every analysis with its status (the stop of 2026-11-17 06:50, figure and table plan, item 7: "planned or
post hoc, its family and adjustment, interpreted or not, where it is reported"). One row per analysis:
- the status and the family as the cited plan or report line states them (each cited text is checked to appear on its
  lines before the table is written);
- for a family, how many statistics are read, counted from the saved read-status columns; otherwise the line of the
  plan or report that says how the values are read, or the report's claim bounds;
- the report section (its heading found in the report, so the line follows the report), and the verdict of every
  audit and results-review round saved under bioinf2027/reviews/ for the item, as each file's first line gives it.
Not analyses, so not listed: D0 (number provenance of the submitted text), C1 (content of the earlier version), D3 (cell
counts), the fact sheets. Usage, from the repository root: python bioinf2027/scripts/figures/tableS_i_analysis_status.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tablelib  # noqa: E402

REPO = tablelib.REPO
B = REPO / "bioinf2027"
RES = B / "results"
UNADJUSTED = ("bioinf2027/plans/E1_design.md", 544, 545, "Everything else is secondary and\nunadjusted")
READ_FILES = []


def text(path: str, first: int, last: int) -> str:
    return "\n".join((REPO / path).read_text().split("\n")[first - 1:last])


def cite(source: tuple) -> str:
    path, first, last, snippet = source
    held = " ".join(text(path, first, last).split())
    if " ".join(snippet.split()) not in held:
        raise SystemExit("%s:%d-%d does not hold %r" % (path, first, last, snippet))
    return "%s:%d" % (path, first) if first == last else "%s:%d-%d" % (path, first, last)


def heading(report: str, snippet: str) -> str:
    """The line of the report heading that starts with the snippet; a snippet ending in $ must be the whole line."""
    lines = (REPO / report).read_text().split("\n")
    hits = [i + 1 for i, line in enumerate(lines) if (line == snippet[:-1] if snippet.endswith("$") else line.startswith(snippet))]
    if len(hits) != 1:
        raise SystemExit("%d headings in %s hold %r" % (len(hits), report, snippet))
    return "%s:%d" % (report, hits[0])


def verdicts(prefix) -> tuple[str, str]:
    """Every saved round of the item's audits and results reviews; a tuple of prefixes adds the later ones by name
    (for example an amendment's audit)."""
    if prefix is None:
        return "not yet", "not yet"
    prefixes = prefix if isinstance(prefix, tuple) else (prefix,)
    out = []
    for kind in ("audit", "results"):
        parts = []
        for k, pre in enumerate(prefixes):
            files = sorted((B / "reviews").glob("%s_%s_r*.md" % (pre, kind)), key=lambda p: int(re.search(r"_r(\d+)\.md$", p.name).group(1)))
            parts += [(pre.replace(prefixes[0] + "_", "") + " " if k else "", f) for f in files]
        labelled = []
        for label, f in parts:
            first = f.read_text().split("\n")[0]
            m = re.match(r"^(?:AUDIT|VERDICT): *([A-Z]+)", first)
            if not m:
                raise SystemExit("%s has no verdict on its first line" % tablelib.rel(f))
            labelled.append("%s%s %s" % (label, re.search(r"_(r\d+)\.md$", f.name).group(1), m.group(1)))
            READ_FILES.append(f)
        out.append("; ".join(labelled) if labelled else "none saved")
    return out[0], out[1]


def count(path: Path, column: str, test) -> tuple[int, int]:
    READ_FILES.append(path)
    values = pd.read_csv(path)[column]
    return int(values.map(test).sum()), len(values)


def read_e1b() -> str:
    n, m = count(RES / "E1/derived/primary_family_read_status.csv", "read_status", lambda s: not str(s).startswith("not read"))
    return "%d of %d read (T and L: the arm fails the VCC precondition; R and R⁻: the R5 gate fails; saved read status)" % (n, m)


def read_e1b_second() -> str:
    n, m = count(RES / "E1/reduces_to_family.csv", "statement", lambda s: str(s) != "not made")
    return "%d of %d statements made (the R5 gate fails)" % (n, m)


def read_d2() -> str:
    path = RES / "D2/supplement/family_read_status.csv"
    READ_FILES.append(path)
    status = pd.read_csv(path)["read_status"].astype(str)
    read = status[status.str.startswith("read: ")].str.replace("read: ", "", regex=False).str.split(";").str[0]
    reasons = status[status.str.startswith("not read: ")].str.replace("not read: ", "", regex=False).value_counts()
    return "%d of %d read, each %s; %d not read (%s) (saved read status)" % (
        len(read), len(status), " or ".join("'%s'" % r for r in sorted(set(read))), int(reasons.sum()),
        "; ".join("%d: %s" % (n, r) for r, n in reasons.items()))


def read_e4() -> str:
    n, m = count(RES / "E4/family_pass/readings.csv", "statement", lambda s: not str(s).startswith("computed and reported, not read"))
    return "%d of %d feature sets and screens read (every statement 'computed and reported, not read')" % (n, m)


def read_e4_full() -> str:
    stats = RES / "E4/stats.json"
    READ_FILES.append(stats)
    record = json.loads(stats.read_text())
    if record.get("pass") != "full" or not record["family_pass_check"]["passed"]:
        raise SystemExit("bioinf2027/results/E4/stats.json is not a full pass whose family values equal the family pass")
    n, m = count(RES / "E4/readings.csv", "statement", lambda s: not str(s).startswith("computed and reported, not read"))
    return ("%d of %d feature sets and screens read (every statement 'computed and reported, not read'); the family values equal "
            "those of the pass of G2 (stats.json, family_pass_check)" % (n, m))


def read_e6() -> str:
    n, m = count(RES / "E6/family.csv", "read", lambda v: bool(v))
    return "%d of %d read (column read)" % (n, m)


def read_e2() -> str:
    READ_FILES.append(RES / "E2/stats/family.csv")
    f = pd.read_csv(RES / "E2/stats/family.csv")
    parts = ["(%s) %d of %d" % (t, int(g["read"].sum()), len(g)) for t, g in f.groupby("test")]
    return "read: " + ", ".join(parts) + " (column read)"


def read_e5() -> str:
    READ_FILES.append(RES / "E5/family/family.csv")
    f = pd.read_csv(RES / "E5/family/family.csv")
    parts = ["endpoint (%s) %d of %d" % (e, int(g["readable"].sum()), len(g)) for e, g in f.groupby("endpoint")]
    return "readable: " + ", ".join(parts) + " (column readable)"


# Item, analysis, status (source), family and adjustment (source), how it is read (callable or (text, source)),
# report section (report, heading text), review prefix.
ROWS = (
    ("E1a", "R0 and the builder checks", ("planned", ("bioinf2027/plans/E1_design.md", 295, 295, "E1a: diagnostics and read-outs, fixed in advance")),
     ("none, unadjusted", UNADJUSTED), ("by the claim bounds", ("bioinf2027/reports/E1a_R0.md", 102, 102, "What may be claimed")),
     ("bioinf2027/reports/E1a_R0.md", "## Results$"), "E1a_R0"),
    ("E1a", "labels of every grid cell; read-outs R1 to R9", ("planned", ("bioinf2027/plans/E1_design.md", 295, 295, "E1a: diagnostics and read-outs, fixed in advance")),
     ("none, unadjusted", UNADJUSTED), ("by the fixed rules, then the claim bounds", ("bioinf2027/reports/E1a.md", 100, 100, "What follows from the fixed rules")),
     ("bioinf2027/reports/E1a.md", "## Results$"), "E1a"),
    ("E1b", "first family: T, L, R, R⁻", ("planned", ("bioinf2027/plans/E1_design.md", 533, 534, "Sixty statistics, Holm-adjusted together")),
     ("Holm at 0.05 over 60", ("bioinf2027/plans/E1_design.md", 533, 534, "family-wise level of 0.05")), read_e1b,
     ("bioinf2027/reports/E1b.md", "## Results$"), "E1b"),
    ("E1b", "second family: 8 equivalence statements", ("planned", ("bioinf2027/plans/E1_design.md", 1043, 1045, "The 8 statements (4 arms by 2 screens) form a second family")),
     ("Holm at 0.05 over 8; with the first family, 0.10 overall", ("bioinf2027/plans/E1_design.md", 1043, 1045, "a claim from either family is controlled at 0.10 overall")), read_e1b_second,
     ("bioinf2027/reports/E1b.md", "## Results$"), "E1b"),
    ("E1b", "VCC-test entry checks", ("planned", ("bioinf2027/plans/E1_design.md", 547, 549, "Precondition per arm and cell")),
     ("unadjusted", UNADJUSTED), ("the precondition of T and L; an arm that fails has no transfer statement in that cell",
                                 ("bioinf2027/plans/E1_design.md", 547, 549, "has no transfer statement in that cell")),
     ("bioinf2027/reports/E1b.md", "## Results$"), "E1b"),
    ("E1b", "post hoc note: the submitted margin against the trivial predictors and across labels",
     ("post hoc", ("bioinf2027/plans/E1b_posthoc.md", 3, 3, "Post hoc relative to the E1b design")),
     ("unadjusted, no family", ("bioinf2027/plans/E1b_posthoc.md", 5, 5, "Every quantity here is secondary and unadjusted")),
     ("by the claim bounds", ("bioinf2027/reports/E1b.md", 535, 535, "What may be claimed")),
     ("bioinf2027/reports/E1b.md", "### Post hoc analyses (gate G3, item 2"), "E1b_posthoc"),
    ("E1b", "attribution analysis (with D2's part (e)) and its amendment",
     ("post hoc", ("bioinf2027/plans/E1b_attribution.md", 3, 3, "Post hoc relative to the E1b design")),
     ("unadjusted, no family", ("bioinf2027/plans/E1b_attribution.md", 5, 6, "unadjusted; none enters a family or a reading rule of E1b or D2")),
     ("readings fixed in the plan", ("bioinf2027/plans/E1b_attribution.md", 335, 335, "Quantities and readings, fixed now")),
     ("bioinf2027/reports/E1b.md", "## Attribution analysis (gate G2, item 12"), ("E1b_attribution", "E1b_attribution_amendment")),
    ("D1", "the endpoint against the target's own shift; the target among the input features",
     ("diagnostic of the submitted version", ("bioinf2027/reports/D1.md", 1, 1, "D1 report: what the endpoint and the largest coordinate measure")),
     ("none: no family is defined for it (the families are those of E1, E4, D2, E6, E2 and E5)", ("bioinf2027/factsheets/FS0_data_methods.md", 394, 394, "Family-wise error across families")), ("by the claim bounds and the author's decision at G6", ("bioinf2027/reports/D1.md", 162, 162, "The author's decision at gate G6 for D1")),
     ("bioinf2027/reports/D1.md", "## Results$"), "D1"),
    ("D1b", "control cells and the target coordinate", ("diagnostic of the submitted version", ("bioinf2027/reports/D1b.md", 1, 1, "D1b report")),
     ("none: no family is defined for it (the families are those of E1, E4, D2, E6, E2 and E5)", ("bioinf2027/factsheets/FS0_data_methods.md", 394, 394, "Family-wise error across families")), ("by the claim bounds", ("bioinf2027/reports/D1b.md", 115, 115, "What may be claimed")),
     ("bioinf2027/reports/D1b.md", "## Results$"), "D1b"),
    ("D2", "family: the target's own token removed", ("planned", ("bioinf2027/plans/D2.md", 134, 135, "**The family**, Holm at 0.05")),
     ("Holm at 0.05 over 18", ("bioinf2027/plans/D2.md", 134, 135, "9 per cell, 18 in all")), read_d2,
     ("bioinf2027/reports/D2.md", "## Results$"), "D2"),
    ("D2", "pre-specified secondaries", ("planned; computed after the family was seen", ("bioinf2027/reports/D2.md", 169, 170, "They are computed now, after the family was seen")),
     ("unadjusted", ("bioinf2027/reports/D2.md", 169, 170, "they are secondary and unadjusted")),
     ("by the claim bounds", ("bioinf2027/reports/D2.md", 401, 401, "What may be claimed")),
     ("bioinf2027/reports/D2.md", "### Supplement: the pre-specified secondaries"), "D2"),
    ("D2", "post hoc analyses asked for by the review", ("post hoc", ("bioinf2027/reports/D2.md", 272, 272, "post hoc analyses asked for by the review")),
     ("outside D2's family of 18, which the plan lists", ("bioinf2027/plans/D2.md", 134, 135, "9 per cell, 18 in all")),
     ("by the claim bounds", ("bioinf2027/reports/D2.md", 401, 401, "What may be claimed")),
     ("bioinf2027/reports/D2.md", "### Supplement: post hoc analyses asked for by the review"), "D2"),
    ("E4", "family F, tiers 1 and 2 (the pass of G2)", ("planned", ("bioinf2027/plans/E4.md", 117, 118, "Family: F for z, x, m and [x; m]")),
     ("Holm at 0.05 over 16", ("bioinf2027/plans/E4.md", 117, 118, "16\n  statistics, Holm-adjusted at 0.05")), read_e4,
     ("bioinf2027/reports/E4.md", "## Results$"), "E4"),
    ("E4", "tier 3 and the full pass (the family again, and the secondary analyses)",
     ("planned; done 2026-11-18", ("bioinf2027/reports/E4.md", 345, 345, "Full pass over tiers 1 to 3 (2026-11-18)")),
     ("the family as above; every other value secondary and unadjusted",
      ("bioinf2027/reports/E4.md", 349, 349, "Everything in this section other than the family is secondary and")), read_e4_full,
     ("bioinf2027/reports/E4.md", "## Full pass over tiers 1 to 3"), "E4"),
    ("E6", "family: transfer between the two Replogle screens", ("planned", ("bioinf2027/plans/E6.md", 209, 210, "24 statistics at the primary endpoint")),
     ("Holm at 0.05 over 24", ("bioinf2027/plans/E6.md", 209, 210, "family-wise level of 0.05")), read_e6,
     ("bioinf2027/reports/E6.md", "## Results$"), "E6"),
    ("E6", "secondary analyses and the supplements of the family pass and the full pass",
     ("planned; supplements decided at G2 and G3", ("bioinf2027/plans/E6.md", 262, 262, "Secondary analyses, every one unadjusted")),
     ("unadjusted", ("bioinf2027/plans/E6.md", 262, 262, "every one unadjusted")), ("by the claim bounds", ("bioinf2027/reports/E6.md", 301, 301, "What may be claimed")),
     ("bioinf2027/reports/E6.md", "### Secondary, unadjusted"), "E6"),
    ("E2", "family: tests (ii) and (iii)", ("planned", ("bioinf2027/plans/E2.md", 192, 194, "**E2's family**, Holm at family-wise 0.05")),
     ("Holm at 0.05 over 30", ("bioinf2027/plans/E2.md", 192, 194, "30 statistics")), read_e2,
     ("bioinf2027/reports/E2.md", "## Results$"), "E2"),
    ("E2", "test (i) and the secondary quantities (P-a, P-r, P-d, P-c, N0, positive control, B heads, other heads)",
     ("planned", ("bioinf2027/plans/E2.md", 196, 198, "**Secondary, unadjusted:**")), ("unadjusted", ("bioinf2027/plans/E2.md", 196, 198, "Secondary, unadjusted")),
     ("by the claim bounds", ("bioinf2027/reports/E2.md", 658, 658, "What may be claimed")), ("bioinf2027/reports/E2.md", "## Results$"), "E2"),
    ("E2", "supplement (a), (b), (c), (e)", ("post hoc, after the results review (G3, item 5)", ("bioinf2027/reports/E2.md", 200, 200, "Supplement added after the results review")),
     ("unadjusted", ("bioinf2027/reports/E2.md", 204, 204, "Every value is secondary and unadjusted")),
     ("by the claim bounds", ("bioinf2027/reports/E2.md", 658, 658, "What may be claimed")),
     ("bioinf2027/reports/E2.md", "## Supplement added after the results review"), "E2_supplement"),
    ("E2", "(d) the external size-matched null", ("post hoc", ("bioinf2027/plans/E2_null_external.md", 10, 12, "It is post hoc")),
     ("unadjusted, not part of E2's family", ("bioinf2027/plans/E2_null_external.md", 10, 12, "is not part of E2's family")),
     ("readings fixed before any value; no statement about why the external values are negative (G7, item 6)",
      ("bioinf2027/plans/E2_null_external.md", 72, 72, "Reading, fixed before any value")),
     ("bioinf2027/reports/E2.md", "## (d) External size-matched null"), "E2_null_external"),
    ("E2", "(d) follow-ups (1) to (3c)", ("post hoc (G7, item 2)", ("bioinf2027/plans/E2_null_external_followup.md", 8, 8, "post hoc, unadjusted, not part of E2's family")),
     ("unadjusted, not part of E2's family", ("bioinf2027/plans/E2_null_external_followup.md", 8, 8, "unadjusted, not part of E2's family")),
     ("readings fixed in the plan", ("bioinf2027/plans/E2_null_external_followup.md", 31, 31, "Analyses (1) to (3), from saved files")),
     ("bioinf2027/reports/E2.md", "## Follow-ups of (d)"), "E2_null_external_followup"),
    ("E2", "(d) follow-up (4): the size-matched null with heads trained on the all-cells label",
     ("post hoc (G7, item 2)", ("bioinf2027/plans/E2_null_external_followup.md", 8, 8, "post hoc, unadjusted")),
     ("unadjusted, not part of E2's family", ("bioinf2027/plans/E2_null_external_followup.md", 8, 8, "not part of E2's family")),
     ("not an external positive control (results review, blocking 1)", ("bioinf2027/reports/E2.md", 439, 439, "What (4) is (review, blocking 1)")),
     ("bioinf2027/reports/E2.md", "**What (4) is (review, blocking 1)"), "E2_null_external_followup"),
    ("E2", "analyses 1 and 3 of the review of (4)", ("post hoc (the stop of 06:50, item 4)", ("bioinf2027/plans/E2_null_external_followup.md", 116, 119, "post hoc, unadjusted")),
     ("unadjusted; one audit, no results review", ("bioinf2027/plans/E2_null_external_followup.md", 116, 119, "one audit, no results review")),
     ("nothing read about why the external values are negative", ("bioinf2027/plans/E2_null_external_followup.md", 137, 137, "Anything about why the external values are negative")),
     ("bioinf2027/reports/E2.md", "**Analyses 1 and 3 of that review, run"), "E2_null_external_followup_s5"),
    ("E5", "family: the identity vector beside the control-cell profile", ("planned", ("bioinf2027/plans/E5.md", 185, 186, "**E5's family**, Holm at family-wise 0.05, 24 statistics")),
     ("Holm at 0.05 over 24", ("bioinf2027/plans/E5.md", 185, 186, "24 statistics")), read_e5,
     ("bioinf2027/reports/E5.md", "## Results$"), "E5"),
    ("E5", "secondary analyses", ("planned", ("bioinf2027/plans/E5.md", 200, 202, "**Secondary, unadjusted:**")), ("unadjusted", ("bioinf2027/plans/E5.md", 200, 202, "Secondary, unadjusted")),
     ("by the claim bounds", ("bioinf2027/reports/E5.md", 1067, 1067, "What may be claimed")), ("bioinf2027/reports/E5.md", "## Results$"), "E5"),
    ("E5", "supplement: analyses 1 to 8 of the results review", ("post hoc (G7, item 3)", ("bioinf2027/plans/E5_supplement.md", 7, 7, "post hoc, unadjusted")),
     ("unadjusted, not part of E5's family", ("bioinf2027/plans/E5_supplement.md", 7, 7, "not part of E5's family of 24")),
     ("readings fixed in the plan; diagnostics without a reading", ("bioinf2027/plans/E5_supplement.md", 142, 142, "This is a diagnostic")),
     ("bioinf2027/reports/E5.md", "## Supplement: the review's eight analyses"), "E5_supplement"),
    ("E5", "supplement 2: five analyses the first supplement's review finds missing",
     ("post hoc (the stop of 06:50, item 7)", ("bioinf2027/plans/E5_supplement_2.md", 8, 8, "post hoc, unadjusted")),
     ("unadjusted, not part of E5's family", ("bioinf2027/plans/E5_supplement_2.md", 8, 8, "not part of E5's family of 24")),
     ("readings fixed in the plan; the same-row context, the route and the B0 diagnostics without a reading",
      ("bioinf2027/plans/E5_supplement_2.md", 49, 49, "have no reading of their own")),
     ("bioinf2027/reports/E5.md", "## Supplement 2:"), "E5_supplement_2"),
    ("S1 supplement", "(a) to (d): the all-cells label and cell number on VCC and on the Replogle screens",
     ("post hoc (the stop of 06:50, new item)", ("bioinf2027/plans/S1_supplement.md", 8, 8, "post hoc, unadjusted, not part of any family")),
     ("unadjusted, no family", ("bioinf2027/plans/S1_supplement.md", 8, 8, "not part of any family")),
     ("readings fixed in the plan; the curve and the encoders' rows of (d) without a reading", ("bioinf2027/plans/S1_supplement.md", 109, 109, "reported with its state and no reading")),
     ("bioinf2027/reports/S1_supplement.md", "## Results$"), "S1_supplement"),
    ("Closing round", "S1 supplement's missing 4 and 6; E5 supplement 2's missing 2, 6 and 8 (then the lists are closed)",
     ("post hoc (the stop of 17:10, item 9)", ("bioinf2027/plans/missing_analyses_closing.md", 10, 10, "post hoc, unadjusted")),
     ("unadjusted, no family", ("bioinf2027/plans/missing_analyses_closing.md", 10, 10, "not part of any family")),
     ("diagnostics without a reading; 2 and 6 limitations; audited, no results review by the author's decision",
      ("bioinf2027/plans/missing_analyses_closing.md", 10, 10, "no reading string")),
     ("bioinf2027/reports/missing_analyses_closing.md", "## Results$"), "closing"),
)


def main() -> None:
    rows, cited = [], set()
    for item, analysis, (status, s_src), (family, f_src), read, (report, head), prefix in ROWS:
        sources = [cite(s_src), cite(f_src)]
        if callable(read):
            how = read()
        else:
            how = read[0]
            sources.append(cite(read[1]))
        audit, review = verdicts(prefix)
        cited.update(s.split(":")[0] for s in sources)
        rows.append({"Item": item, "Analysis": analysis, "Planned or post hoc": status, "Family and adjustment": family,
                     "Read": how, "Reported in": heading(report, head), "Audit rounds": audit, "Results review rounds": review,
                     "Status from": "; ".join(dict.fromkeys(sources))})
        cited.add(report)
    frame = pd.DataFrame(rows)
    inputs = sorted({REPO / c for c in cited} | set(READ_FILES))
    tablelib.save("tableS_i_analysis_status", frame, inputs, Path(__file__).resolve(), align="l" * len(frame.columns),
                  extra={"e4_status_of_every_row": "computed and reported, not read",
                         "not_listed": "D0 (number provenance of the submitted text), C1 (content of the earlier version), D3 (cell counts): not analyses",
                         "verdicts": "every saved round, as the first line of each review file gives it; a FAIL or REVISE round is followed by the "
                                     "item's report and the author's decisions in bioinf2027/GATE.md"})


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: tableS_i_analysis_status.py")
    main()
