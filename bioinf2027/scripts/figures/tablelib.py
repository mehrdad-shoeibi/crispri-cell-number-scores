"""Shared helpers of the table scripts (the stop of 2026-11-17 06:50, figure and table rules; bioinf2027/GATE.md).

Every table is printed from saved files only, one command each. Outputs go to bioinf2027/tables/: <name>.csv (the
values), <name>.tex (a booktabs tabular, no caption or title: captions are the author's) and <name>.json (the inputs
with their SHA-256 and how each was verified, the code's SHA-256 and the outputs). Before a table is written, each
input is checked against the SHA-256 its item's committed record gives (figure and table rules,
bioinf2027/factsheets/figure_table_plan.md), or, if no record lists it, required to be a tracked file without
uncommitted changes. No value is computed here beyond formatting.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bioinf2027" / "src"))

from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

TABLES = REPO / "bioinf2027" / "tables"


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO))


RECORD_DIRS = ("results", "endpoints", "diag")
_INDEX: dict[str, set[tuple[str, str]]] | None = None


def _records() -> dict[str, set[tuple[str, str]]]:
    """Every (SHA-256, record file) that a committed record under bioinf2027/results, endpoints or diag gives for a
    repository path (entries holding "path" and "sha256"); untracked records are not used."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}

        def walk(node, record):
            if isinstance(node, dict):
                if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str) and node["path"].startswith("bioinf2027/"):
                    _INDEX.setdefault(node["path"], set()).add((node["sha256"], record))
                for value in node.values():
                    walk(value, record)
            elif isinstance(node, list):
                for value in node:
                    walk(value, record)

        tracked = set(subprocess.run(["git", "ls-files", "--", *("bioinf2027/%s" % f for f in RECORD_DIRS)], cwd=REPO,
                                     capture_output=True, text=True, check=True).stdout.split())
        for folder in RECORD_DIRS:
            for path in sorted((REPO / "bioinf2027" / folder).rglob("*.json")):
                if rel(path) not in tracked:
                    continue
                try:
                    walk(json.loads(path.read_text()), rel(path))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    return _INDEX


def environment() -> dict:
    import numpy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pd.__version__,
            "requirements": file_record(REPO / "bioinf2027" / "requirements.txt")}


def verify_inputs(inputs: list[Path]) -> dict[str, str]:
    """Each input must equal a SHA-256 that a committed record gives for it; an input no record lists must be a
    tracked file without uncommitted changes. Stops on a mismatch; returns how each input was verified."""
    out = {}
    for path in inputs:
        name = rel(path)
        recorded = _records().get(name, set())
        if recorded:
            digest = sha256(Path(path))
            hits = sorted(r for d, r in recorded if d == digest)
            if not hits:
                raise SystemExit("%s matches none of its recorded SHA-256 (%s)" % (name, ", ".join(sorted(r for _, r in recorded))))
            out[name] = "equal to its record in %s" % hits[0]
        else:
            tracked = subprocess.run(["git", "ls-files", "--error-unmatch", name], cwd=REPO, capture_output=True).returncode == 0
            clean = subprocess.run(["git", "status", "--porcelain", "--", name], cwd=REPO, capture_output=True, text=True).stdout == ""
            if not (tracked and clean):
                raise SystemExit("%s is listed in no record and is not a tracked file without changes" % name)
            out[name] = "no record lists it; a tracked file without uncommitted changes"
    return out


def latex(frame: pd.DataFrame, align: str | None = None) -> str:
    cols = list(frame.columns)
    align = align or "l" + "r" * (len(cols) - 1)
    def escape(value) -> str:
        s = str(value).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
        s = s.replace("~", r"\textasciitilde{}").replace("`", "")
        return re.sub(r"(?<![\w.])-(?=\d)", "\u2212", s)  # a minus sign, as the CSV files print it elsewhere
    lines = [r"\begin{tabular}{%s}" % align, r"\toprule", " & ".join(escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in frame.iterrows():
        lines.append(" & ".join(escape(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def save(name: str, frame: pd.DataFrame, inputs: list[Path], script: Path, extra: dict | None = None, align: str | None = None) -> None:
    verified = verify_inputs(inputs)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.tex").write_text(latex(frame, align))
    record = {"table": name, "code": {rel(script): sha256(script), rel(Path(__file__)): sha256(Path(__file__))},
              "inputs": {rel(p): file_record(p) for p in inputs}, "inputs_verified": verified, "environment": environment(),
              "tex": "booktabs; Unicode symbols (−, ρ, →, ½), so a Unicode TeX engine (XeLaTeX or LuaLaTeX)", **(extra or {}),
              "outputs": {f"{name}.csv": file_record(TABLES / f"{name}.csv"), f"{name}.tex": file_record(TABLES / f"{name}.tex")}}
    (TABLES / f"{name}.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[%s] %d rows written to %s" % (name, len(frame), rel(TABLES)), flush=True)
