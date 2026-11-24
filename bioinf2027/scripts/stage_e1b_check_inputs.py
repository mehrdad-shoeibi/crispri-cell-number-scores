"""Stage the git-ignored inputs of E1b's pre-run checks under the data root and record their provenance.

bioinf2027/scripts/make_e1b_checks.py (the harness check, bioinf2027/plans/E1_design.md, Section 25 and
following; run recorded in bioinf2027/runs/E1.json, label e1b-checks, commit dd13c55) read ten files
that lie in the git-ignored part of the repository: the stored VCC-test predictions of the seven
submitted arms and the stored VCC-test label (earlier_revision/phase1/preds/), and the stored external
magnitude blocks of the two screens (earlier_revision/phase4/m_tilde/). The first E1b audit
(bioinf2027/reviews/E1b_audit_r1.md, hygiene findings) failed them as inputs outside the data root.

This script copies each file, byte for byte, to $IGA_STORAGE_ROOT/bioinf2027/cache/e1b_check_inputs/
under its repository-relative path, leaves the repository copies in place, and checks that the source,
the copy and every earlier record of the file carry the same size and SHA-256:

- bioinf2027/audit/D0_input_manifest.csv (nine of the ten files, all but m_tilde_xcell_K562.npz; the
  only earlier record of the seven prediction arrays);
- bioinf2027/results/E1/input_records.json (the label and the two magnitude blocks).

The checks are not rerun: a rerun would repeat the timing probe that fixed the refits of the fit run
(bioinf2027/results/E1/checks.json). The staged copies are the archive of record for the files the
checks read.

Output: the staged copies and the committed record bioinf2027/results/E1/check_inputs_staged.json.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TARGET = config.storage_root() / "bioinf2027" / "cache" / "e1b_check_inputs"
RECORD = REPO / "bioinf2027" / "results" / "E1" / "check_inputs_staged.json"
D0_MANIFEST = REPO / "bioinf2027" / "audit" / "D0_input_manifest.csv"
INPUT_RECORDS = REPO / "bioinf2027" / "results" / "E1" / "input_records.json"
PREDICTIONS = [f"earlier_revision/phase1/preds/test_{arm}.npy" for arm in ("z", "zm", "C1", "C2", "C3", "C4", "C5")]
FILES = PREDICTIONS + ["earlier_revision/phase1/preds/y_test.npy",
                       "earlier_revision/phase4/m_tilde/m_tilde_xcell_RPE1.npz",
                       "earlier_revision/phase4/m_tilde/m_tilde_xcell_K562.npz"]
READ_AT = {**{p: "bioinf2027/scripts/make_e1b_checks.py, harness_summary (stored VCC-test predictions)" for p in PREDICTIONS},
           "earlier_revision/phase1/preds/y_test.npy": "bioinf2027/scripts/make_e1b_checks.py, harness_summary (stored VCC-test label)",
           "earlier_revision/phase4/m_tilde/m_tilde_xcell_RPE1.npz": "bioinf2027/src/bioinf2027/e1inputs.py, paths(), RPE1_submitted_magnitude, "
                                                                       "read by make_e1b_checks.py for the external [z; m] blocks",
           "earlier_revision/phase4/m_tilde/m_tilde_xcell_K562.npz": "bioinf2027/src/bioinf2027/e1inputs.py, paths(), K562-essential_submitted_magnitude, "
                                                                       "read by make_e1b_checks.py for the external [z; m] blocks"}


def d0_records() -> dict:
    with open(D0_MANIFEST, newline="") as handle:
        return {row["path"]: {"bytes": int(row["bytes"]), "sha256": row["sha256"], "line": i + 2}
                for i, row in enumerate(csv.DictReader(handle)) if row["kind"] == "file"}


def e1_records() -> dict:
    out = {}
    for name, entry in json.loads(INPUT_RECORDS.read_text())["inputs"].items():
        if isinstance(entry, dict) and entry.get("path") in FILES:
            out[entry["path"]] = {"bytes": int(entry["bytes"]), "sha256": entry["sha256"], "key": f"inputs.{name}"}
    return out


def stage(relative: str, earlier: list[tuple[str, dict]]) -> dict:
    source = REPO / relative
    target = TARGET / relative
    digest = sha256(source)
    if target.exists():
        if sha256(target) != digest:
            raise RuntimeError("%s exists under the data root with other contents" % relative)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    staged = file_record(target)
    if staged["sha256"] != digest:
        raise RuntimeError("the staged copy of %s differs from its source" % relative)
    matches = []
    for where, record in earlier:
        if record["sha256"] != digest or record["bytes"] != source.stat().st_size:
            raise RuntimeError("%s differs from its record in %s" % (relative, where))
        matches.append({"record": where, "bytes": record["bytes"], "sha256": record["sha256"]})
    if not matches:
        raise RuntimeError("%s has no earlier record" % relative)
    return {"source": file_record(source), "staged": staged, "read_by": READ_AT[relative], "earlier_records_matched": matches}


def main() -> None:
    d0, e1 = d0_records(), e1_records()
    files = {}
    for relative in FILES:
        earlier = []
        if relative in d0:
            earlier.append((f"{config.portable(D0_MANIFEST)}, line {d0[relative]['line']}", d0[relative]))
        if relative in e1:
            earlier.append((f"{config.portable(INPUT_RECORDS)}, {e1[relative]['key']}", e1[relative]))
        files[relative] = stage(relative, earlier)
    record = {"purpose": "inputs of the E1b pre-run checks, staged under the data root",
              "plan": "bioinf2027/plans/E1_design.md, Sections 25 to 29; bioinf2027/reviews/E1b_audit_r1.md, hygiene findings",
              "checks_run": "bioinf2027/runs/E1.json, label e1b-checks (commit dd13c55), which read the repository copies; "
                            "every copy staged here carries the same size and SHA-256, so it is the file the checks read",
              "checks_not_rerun": "a rerun would repeat the timing probe that fixed the refits of the fit run (bioinf2027/results/E1/checks.json)",
              "earlier_records": {"D0": file_record(D0_MANIFEST), "E1": file_record(INPUT_RECORDS)},
              "files": files}
    RECORD.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[stage] %d files staged under %s; each equals its source and every earlier record" % (len(files), config.portable(TARGET)), flush=True)


if __name__ == "__main__":
    main()
