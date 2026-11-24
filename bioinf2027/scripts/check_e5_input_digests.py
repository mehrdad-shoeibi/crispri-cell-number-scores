"""E5, check 1 of the plan (bioinf2027/plans/E5.md, Section 11: every input equals its recorded digest at load),
applied after the runs to the inputs that E5's code reads without listing them or checking them at load (the audit,
bioinf2027/reviews/E5_audit_r1.md, note 2). Read only; it changes no value and no E5 record.

- The 13 data-root inputs of the audit's note 2, at the paths E5's code reads them from: the two endpoint tables
  (bioinf2027.config.screen_paths, read through features.read_endpoint), the two files of the 2,000 features
  (config.vcc_cache, read through features.hvg_symbols) and the nine screen inputs of e1inputs.load_screen_inputs
  (e1inputs.paths). Each is compared, by size and SHA-256, with every record of the same path in
  bioinf2027/results/E1/input_records.json, bioinf2027/diag/D1_external_provenance.json and
  bioinf2027/results/E6/family_pass/supplement/supplement.json; it passes when at least one record holds it and every
  record that holds it agrees.
- The E1a label cache's vcc_train.npz, which the two identity scripts read, against the cache's manifest, and the
  manifest against its record in bioinf2027/results/E4/features.json.

Output: bioinf2027/results/E5/checks/input_digests.json, written whether or not every digest matches; the script
exits with status 1 if any differs or has no record (hard limit 9: stop and report).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1inputs  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e1a_label_extract as second  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E5" / "checks" / "input_digests.json"
RECORD_FILES = {
    "E1": REPO / "bioinf2027" / "results" / "E1" / "input_records.json",
    "D1": REPO / "bioinf2027" / "diag" / "D1_external_provenance.json",
    "E6": REPO / "bioinf2027" / "results" / "E6" / "family_pass" / "supplement" / "supplement.json",
}
E4_FEATURES = REPO / "bioinf2027" / "results" / "E4" / "features.json"
SCREEN_INPUTS = ("RPE1_native", "RPE1_mapped_with_ids", "RPE1_submitted_x", "K562-essential_native",
                 "K562-essential_submitted_x", "RPE1_z", "RPE1_z_index", "K562-essential_z", "K562-essential_z_index")


def inputs() -> dict[str, Path]:
    """The 13 inputs at the paths E5's code reads them from."""
    screens = config.screen_paths()
    out = {"%s_endpoint" % s: screens[s]["endpoint"] for s in ("RPE1", "K562-essential")}
    out.update({"vcc_%s" % k: v for k, v in config.vcc_cache().items()})
    paths = e1inputs.paths()
    out.update({k: paths[k] for k in SCREEN_INPUTS})
    return out


def records_by_path() -> dict[str, list[dict]]:
    """Every file record (a dict with a path, bytes and sha256) of the three record files, by portable path."""
    found: dict[str, list[dict]] = {}

    def walk(node, where: str, source: str) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and "sha256" in node and "bytes" in node:
                found.setdefault(node["path"], []).append({"source": source, "key": where, "bytes": int(node["bytes"]), "sha256": node["sha256"]})
                return
            for key, value in node.items():
                walk(value, where + "/" + str(key), source)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, "%s[%d]" % (where, i), source)

    for source, path in RECORD_FILES.items():
        walk(json.loads(path.read_text()), "", config.portable(path))
    return found


def check_inputs() -> dict:
    recorded = records_by_path()
    out = {}
    for name, path in inputs().items():
        started = time.time()
        now = file_record(path)
        held = recorded.get(now["path"], [])
        for r in held:
            r["equal"] = bool(now.get("sha256") == r["sha256"] and now.get("bytes") == r["bytes"])
        now["records"] = held
        now["equal_to_every_record"] = bool(held) and all(r["equal"] for r in held)
        now["seconds"] = round(time.time() - started, 1)
        print("[e5 digests] %s: %s (%d records, %.0f s)" % (name, "equal" if now["equal_to_every_record"] else "DIFFERENT OR UNRECORDED", len(held), now["seconds"]), flush=True)
        out[name] = now
    return out


def check_label_cache() -> dict:
    folder = second.cache_dir()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["cache"]["vcc_train"]
    now = file_record(folder / "vcc_train.npz")
    manifest_now = file_record(manifest_path)
    e4 = json.loads(E4_FEATURES.read_text())["inputs"]["label_cache"]
    result = {"vcc_train": now, "manifest_entry_sha256": expected["sha256"],
              "vcc_train_equal_to_manifest": bool(now.get("sha256") == expected["sha256"]),
              "manifest": manifest_now, "manifest_record_in_e4_features": e4,
              "manifest_equal_to_record": bool(manifest_now.get("sha256") == e4["sha256"] and manifest_now.get("bytes") == e4["bytes"])}
    result["equal"] = result["vcc_train_equal_to_manifest"] and result["manifest_equal_to_record"]
    print("[e5 digests] E1a label cache vcc_train.npz: %s" % ("equal" if result["equal"] else "DIFFERENT"), flush=True)
    return result


def main() -> None:
    started = time.time()
    found = check_inputs()
    cache = check_label_cache()
    passed = all(r["equal_to_every_record"] for r in found.values()) and cache["equal"]
    record = {"check": "check 1 of bioinf2027/plans/E5.md, Section 11, applied after the runs to the inputs of note 2 of bioinf2027/reviews/E5_audit_r1.md",
              "rule": "SHA-256 and size of each file now, against every record of its path; read only",
              "inputs": found, "E1a_label_cache": cache, "passed": passed,
              "records": {k: file_record(v) for k, v in {**RECORD_FILES, "E4_features": E4_FEATURES}.items()},
              "code": {config.portable(Path(m.__file__).resolve()): sha256(Path(m.__file__)) for m in (sys.modules[__name__], e1inputs, config, second)},
              "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - started, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e5 digests] %s in %.0f s" % ("every digest equal to its records" if passed else "A DIGEST DIFFERS OR HAS NO RECORD", time.time() - started), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: check_e5_input_digests.py")
    main()
