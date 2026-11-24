"""Check of the author's change of 2026-11-15 to E2's fit batch (one process per fit, so that workers can be shed
one at a time, and a cache of the loaded inputs; bioinf2027/GATE.md): it must change no value.

Compares the smoke fits saved by the new code with those saved by the code before the change (make_e2_fits.py at
7f36b1d; for test (iii), make_e2_fits_e6.py before its --slots), file by file: the same files; in each, the same specification, the same record apart from the fit's
seconds, and the same prediction arrays (names, dtype, shape and bytes). Both folders are under
$IGA_STORAGE_ROOT/bioinf2027/scratch/; the result goes to bioinf2027/results/E2/checks/slots_cache_<name>.json.
With --subset, every file of the new folder must be in the reference and equal, but not every reference file in the
new folder (a partial run checked against a whole one). With --reference-files, every file of the reference must be in
the new folder and equal; files of other encoders that only the new folder holds are counted and listed.
Usage: check_e2_slots_cache.py <reference folder> <new folder> <name> [--subset | --reference-files]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bioinf2027" / "src"))
import numpy as np  # noqa: E402

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record  # noqa: E402


def fits(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*.npz")) if ".tmp" not in p.name}


def compare(a: Path, b: Path) -> list[str]:
    out = []
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        if sorted(x.files) != sorted(y.files):
            return ["arrays %s, new %s" % (sorted(x.files), sorted(y.files))]
        for name in x.files:
            u, v = x[name], y[name]
            if name == "record":
                ru, rv = json.loads(str(u)), json.loads(str(v))
                ru.pop("seconds", None)
                rv.pop("seconds", None)
                if ru != rv:
                    out.append("record differs: %s / %s" % (ru, rv))
            elif u.dtype != v.dtype or u.shape != v.shape or u.tobytes() != v.tobytes():
                out.append("%s differs" % name)
    return out


def main() -> int:
    root = config.storage_root() / "bioinf2027" / "scratch"
    ref, new, name = root / sys.argv[1], root / sys.argv[2], sys.argv[3]
    subset = "--subset" in sys.argv[4:]
    reference_files = "--reference-files" in sys.argv[4:]
    a, b = fits(ref / "predictions"), fits(new / "predictions")
    found = {}
    for key in sorted(set(b) if subset else set(a) if reference_files else set(a) | set(b)):
        if key not in a or key not in b:
            found[key] = ["only in the %s folder" % ("reference" if key in a else "new")]
            continue
        d = compare(a[key], b[key])
        if d:
            found[key] = d
    def pick(folder: Path, pattern: str) -> Path | None:
        """The file of the batch the reference holds (same name), else the folder's first."""
        names = [p.name for p in sorted(ref.glob(pattern))]
        found_ = [folder / n for n in names if (folder / n).is_file()] or sorted(folder.glob(pattern))
        return found_[0] if found_ else None

    chosen = pick(new, "protocol_*.json")
    protocol = json.loads(chosen.read_text()) if chosen else {}
    reference_protocol = pick(ref, "protocol_*.json")
    manifests = {k: file_record(m) for k, m in (("reference", pick(ref, "predictions_manifest_*.json")), ("new", pick(new, "predictions_manifest_*.json"))) if m}
    result = {"check": "smoke fits of the new code equal those of the code before the change; the scripts are named under reference_script "
                       "and new_script (bioinf2027/GATE.md, change of 2026-11-15)",
              "reference": config.portable(ref), "new": config.portable(new), "subset": subset, "reference_files": reference_files, "files_reference": len(a), "files_new": len(b),
              "files_only_in_new_not_compared": sorted(set(b) - set(a)) if reference_files else [],
              "files_compared_equal": len(set(a) & set(b)) - len([k for k in found if k in a and k in b]), "differences": found,
              "manifests": manifests, "reference_script": json.loads(reference_protocol.read_text()).get("script") if reference_protocol else None,
              "new_script": protocol.get("script"), "new_runner": protocol.get("runner"), "new_input_cache": protocol.get("input_cache"), "new_check_E0": protocol.get("check_E0"),
              "passed": not found and len(b) > 0 and (subset or reference_files or len(a) == len(b)), "made": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    out = REPO / "bioinf2027" / "results" / "E2" / "checks" / ("slots_cache_%s.json" % name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("%s: %d reference files, %d new, %d equal, %d with differences; passed %s"
          % (name, len(a), len(b), result["files_compared_equal"], len(found), result["passed"]))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
