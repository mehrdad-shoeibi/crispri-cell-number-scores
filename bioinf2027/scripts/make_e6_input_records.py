"""E6, the data-root inputs of the fits and the statistics, with size and SHA-256, checked against their records.

Design: bioinf2027/plans/E6.md, Section 3.4 and stop rule 1 (the stored z is hashed and the run stops on any
difference); bioinf2027/reviews/E6_audit_r1.md (check 5: E6's records did not list these inputs, and the fits
did not hash the stored z).

The fits (bioinf2027/scripts/make_e6_fits.py, load_inputs and shared_gene_inputs) and the statistics
(bioinf2027/scripts/make_e6_stats.py, non_shared_causes) read 15 files under the data root through
bioinf2027.e1inputs, bioinf2027.features and bioinf2027.config. Each is resolved here through the same
functions, hashed, and compared with the record that already holds it: bioinf2027/results/E1/input_records.json
for 13 of them and bioinf2027/endpoints/R0_provenance.json for the two eligibility tables. The check is made
after E6's runs, so the file's modification time is also compared with the commit of that record and with the
start of E6's first run: a file whose digest equals its record and that was last modified before both was the
same file in every E6 run listed in bioinf2027/runs/E6.json.

Output: bioinf2027/results/E6/input_records.json. Exit code 1 if any input differs (stop rule 1).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, e1inputs  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E6" / "input_records.json"
E1_RECORDS = REPO / "bioinf2027" / "results" / "E1" / "input_records.json"
R0_RECORDS = REPO / "bioinf2027" / "endpoints" / "R0_provenance.json"
RUNS = REPO / "bioinf2027" / "runs" / "E6.json"
PLAN = "bioinf2027/plans/E6.md, Section 3.4 and stop rule 1; bioinf2027/reviews/E6_audit_r1.md"

# name of the input in E6's loaders -> (record file, key in that record's inputs)
E1_KEYS = {
    "RPE1_z": "rpe1_z",
    "RPE1_z_index": "rpe1_z_index",
    "K562-essential_z": "k562_z",
    "K562-essential_z_index": "k562_z_index",
    "RPE1_native": "rpe1_native_delta",
    "RPE1_mapped_with_ids": "rpe1_mapped_delta",
    "RPE1_submitted_x": "rpe1_mapped_delta_1932",
    "K562-essential_native": "k562_native_delta",
    "K562-essential_submitted_x": "k562_mapped_delta_1903",
    "RPE1_single_cell": "rpe1_single_cell",
    "K562-essential_single_cell": "k562_single_cell",
}
VCC_KEYS = {"var_gene_names": "vcc_gene_names", "selected_genes": "vcc_selected_genes"}
R0_KEYS = {"RPE1": "RPE1_eligibility", "K562-essential": "K562-essential_eligibility"}
READ_BY = {
    "z": "make_e6_fits.py load_inputs, through e1inputs.load_screen_inputs (_ordered_z)",
    "x": "make_e6_fits.py load_inputs, through e1inputs.load_screen_inputs",
    "single_cell": "make_e6_fits.py (e1inputs.native_symbols, shared_gene_inputs) and make_e6_stats.py non_shared_causes; var annotations only",
    "vcc": "features.hvg_symbols, in make_e6_fits.py load_inputs and e1inputs.load_screen_inputs",
    "eligibility": "make_e6_stats.py non_shared_causes",
}


def commit_time(path: Path) -> str:
    """Committer time of the last commit that touched a tracked file (ISO 8601)."""
    out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", str(path.relative_to(REPO))], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.strip()
    if not out:
        raise RuntimeError("%s has no commit" % path.relative_to(REPO))
    return out


def e6_runs() -> list[dict]:
    """Every recorded E6 run of the fit or statistics script (these are the runs that read the inputs)."""
    runs = json.loads(RUNS.read_text())
    runs = runs if isinstance(runs, list) else runs["runs"]
    return [{"label": r["label"], "command": r["command"], "started": r["started"], "finished": r["finished"]}
            for r in runs if "make_e6_fits.py" in r["command"] or "make_e6_stats.py" in r["command"]]


def listed() -> list[tuple[str, Path, str, dict, str]]:
    """(name, path as the loaders resolve it, record file, record entry, read by)."""
    e1 = json.loads(E1_RECORDS.read_text())["inputs"]
    r0 = json.loads(R0_RECORDS.read_text())["other_inputs"]
    paths = e1inputs.paths()
    out = []
    for name, key in E1_KEYS.items():
        kind = "z" if name.endswith(("_z", "_z_index")) else "single_cell" if name.endswith("_single_cell") else "x"
        out.append((name, paths[name], "bioinf2027/results/E1/input_records.json", e1[key], READ_BY[kind]))
    for name, key in VCC_KEYS.items():
        out.append((f"vcc_{name}", config.vcc_cache()[name], "bioinf2027/results/E1/input_records.json", e1[key], READ_BY["vcc"]))
    screens = config.screen_paths()
    for screen, key in R0_KEYS.items():
        out.append((f"{screen}_eligibility", screens[screen]["eligibility"], "bioinf2027/endpoints/R0_provenance.json", r0[key], READ_BY["eligibility"]))
    return out


def main() -> None:
    started = time.time()
    runs = e6_runs()
    first_run = min(datetime.fromisoformat(r["started"]) for r in runs)
    record_commits = {"bioinf2027/results/E1/input_records.json": commit_time(E1_RECORDS),
                      "bioinf2027/endpoints/R0_provenance.json": commit_time(R0_RECORDS)}
    inputs, differing = {}, []
    for name, path, record_file, entry, read_by in listed():
        portable = config.portable(path)
        if portable != entry["path"]:
            raise RuntimeError("%s resolves to %s, but %s records %s" % (name, portable, record_file, entry["path"]))
        stat = os.stat(path)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
        record_time = datetime.fromisoformat(record_commits[record_file])
        digest = sha256(path)
        result = {
            "path": portable,
            "bytes": stat.st_size,
            "sha256": digest,
            "modified": modified.isoformat(timespec="seconds"),
            "read_by": read_by,
            "record": record_file,
            "recorded_bytes": entry["bytes"],
            "recorded_sha256": entry["sha256"],
            "equal_to_record": bool(digest == entry["sha256"] and stat.st_size == entry["bytes"]),
            "modified_before_the_record_was_committed": bool(modified < record_time),
            "modified_before_the_first_e6_run": bool(modified < first_run),
        }
        inputs[name] = result
        if not (result["equal_to_record"] and result["modified_before_the_record_was_committed"] and result["modified_before_the_first_e6_run"]):
            differing.append(name)
        print("[e6 inputs] %s: %s" % (name, "equal" if name not in differing else "DIFFERS"), flush=True)
    report = {
        "purpose": "the data-root inputs of E6's fits and statistics, hashed after the runs and checked against their records",
        "plan": PLAN,
        "references": {"bioinf2027/results/E1/input_records.json": {**file_record(E1_RECORDS), "committed": record_commits["bioinf2027/results/E1/input_records.json"]},
                       "bioinf2027/endpoints/R0_provenance.json": {**file_record(R0_RECORDS), "committed": record_commits["bioinf2027/endpoints/R0_provenance.json"]}},
        "runs_covered": runs,
        "first_e6_run_started": first_run.isoformat(),
        "inputs": inputs,
        "differing": differing,
        "passed": not differing,
        "seconds": round(time.time() - started, 1),
    }
    OUT.write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
    print("[e6 inputs] %d inputs, %d differ; %d E6 runs covered, the first started %s; %.0f s"
          % (len(inputs), len(differing), len(runs), first_run.isoformat(), time.time() - started), flush=True)
    if differing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
