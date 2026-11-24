"""E2, check 1 of the plan (bioinf2027/plans/E2.md, Section 9), applied after the runs (gate G3, item 2, the author's
answer of 2026-11-16, bioinf2027/GATE.md): the SHA-256 of the five single-cell files and of every file of
Geneformer's cached VCC control embeddings, computed now and compared with their records. Read only.

- The five single-cell files (VCC training, validation and test; RPE1; K562-essential), against
  bioinf2027/plans/E1_input_records.json, as make_e1a_extract.verify_single_cell_inputs compares them (digest and
  size), and beside it the copies of the Replogle records in bioinf2027/results/D2/input_records.json.
- Geneformer's control embeddings (the folder whose nt__batch_*.npy files make_e2_features.geneformer_null reads),
  file by file and by the digest of the sorted listing, against the cached store vcc_control_embeddings of
  bioinf2027/results/D2/input_records.json.

Output: bioinf2027/results/E2/checks/input_digests.json, written whether or not every digest matches; the script
exits with status 1 if any differs (hard limit 9: stop and report).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, sha256, strict_json  # noqa: E402

import make_e1a_extract as first  # noqa: E402
import make_e2_features as e2feat  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "checks" / "input_digests.json"
D2_RECORDS = REPO / "bioinf2027" / "results" / "D2" / "input_records.json"
CONTROL_STORE = "vcc_control_embeddings"
D2_KEYS = {"RPE1": "rpe1_single_cell", "K562-essential": "k562_single_cell"}


def control_folder() -> Path:
    """The folder of make_e2_features.geneformer_null, under the data root."""
    return config.storage_root() / "MADA" / "cache" / "foundation_models" / "geneformer" / "full_feature_extraction_v2" / "nt_embeddings"


def single_cell() -> dict:
    recorded = json.loads(first.INPUT_RECORDS.read_text())["files"]
    d2 = json.loads(D2_RECORDS.read_text())["single_cell_files"]
    out = {}
    for key, path in {**first.vcc_files(), **first.screen_files()}.items():
        started = time.time()
        record = file_record(path)
        expected = recorded[first.RECORD_KEYS[key]]
        record["recorded"] = {"sha256": expected["sha256"], "bytes": expected["bytes"]}
        record["matches_input_records"] = bool(record.get("sha256") == expected["sha256"] and record.get("bytes") == expected["bytes"])
        if key in D2_KEYS:
            other = d2[D2_KEYS[key]]
            record["matches_d2_input_records"] = bool(record.get("sha256") == other["sha256"] and record.get("bytes") == other["bytes"])
        record["seconds"] = round(time.time() - started, 1)
        print("[e2 digests] %s: %s (%.0f s)" % (key, "equal" if record["matches_input_records"] else "DIFFERENT", record["seconds"]), flush=True)
        out[key] = record
    return out


def control_embeddings() -> dict:
    expected = json.loads(D2_RECORDS.read_text())["cached_embedding_stores"][CONTROL_STORE]["listing"]
    folder = control_folder()
    now = folder_record([p for p in folder.rglob("*") if p.is_file()], folder)
    recorded = {e["path"]: e for e in expected["entries"]}
    current = {e["path"]: e for e in now["entries"]}
    differ = sorted(p for p in set(recorded) & set(current) if (recorded[p]["sha256"], recorded[p]["bytes"]) != (current[p]["sha256"], current[p]["bytes"]))
    missing = sorted(set(recorded) - set(current))
    extra = sorted(set(current) - set(recorded))
    read = sorted(p for p in current if p.startswith("nt__batch_") and p.endswith(".npy") and not p.endswith("_index.npy"))
    result = {"path": now["path"], "files_now": now["files"], "files_recorded": int(expected["files"]),
              "total_bytes_now": now["total_bytes"], "total_bytes_recorded": int(expected["total_bytes"]),
              "sha256_of_listing_now": now["sha256_of_listing"], "sha256_of_listing_recorded": expected["sha256_of_listing"],
              "files_that_differ": differ, "files_missing": missing, "files_not_recorded": extra,
              "embedding_files_read_by_the_null_features": len(read),
              "equal": bool(not differ and not missing and not extra and now["sha256_of_listing"] == expected["sha256_of_listing"])}
    print("[e2 digests] Geneformer's control embeddings: %d files, %s" % (now["files"], "equal" if result["equal"] else "DIFFERENT"), flush=True)
    return result


def main() -> None:
    started = time.time()
    cells = single_cell()
    controls = control_embeddings()
    passed = all(r["matches_input_records"] and r.get("matches_d2_input_records", True) for r in cells.values()) and controls["equal"]
    record = {"check": "check 1 of bioinf2027/plans/E2.md, Section 9, applied after the runs (gate G3, item 2; bioinf2027/GATE.md)",
              "rule": "SHA-256 and size of each file now, against its record; read only",
              "single_cell_files": cells, "geneformer_control_embeddings": controls, "passed": passed,
              "records": {"single_cell": file_record(first.INPUT_RECORDS), "d2": file_record(D2_RECORDS)},
              "code": {config.portable(Path(m.__file__).resolve()): sha256(Path(m.__file__)) for m in (sys.modules[__name__], first, e2feat)},
              "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - started, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 digests] %s in %.0f s" % ("every digest equal to its record" if passed else "A DIGEST DIFFERS", time.time() - started), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: check_e2_input_digests.py")
    main()
