"""C1: the values behind the decision items of the comparison with the earlier version.

Two facts the comparison relies on are recomputed and saved here: the spread
of the VCC label in each split, which the earlier version printed as the test
spread, and the per-run training times recorded by the reproduction gate of
the submitted version. Nothing is modelled.

Output: bioinf2027/audit/C1_checks.json
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, strict_json  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "audit" / "C1_checks.json"


def label_spread() -> dict:
    shards = config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "variants" / "delta_hvg" / "shards"
    out = {"shards": folder_record(glob.glob(str(shards / "*" / "*.npz")), shards)}
    for split in ("train", "val", "test"):
        y = np.concatenate(
            [np.load(p, allow_pickle=True)["y"].astype(float) for p in sorted(glob.glob(str(shards / split / "*.npz")))]
        )
        out[split] = {"rows": int(y.size), "sd_ddof1": float(y.std(ddof=1)), "sd_ddof0": float(y.std(ddof=0))}
    return out


def recorded_run_times() -> dict:
    log = config.repo_root() / "earlier_revision" / "phase1" / "gate_final.log"
    runs = []
    for number, line in enumerate(log.read_text().splitlines(), start=1):
        match = re.match(r"\[(\w+) s(\d)\] R2=([+-]?\d+\.\d+) \((\d+)s\)", line.strip())
        if match:
            runs.append({"line": number, "model": match.group(1), "seed": int(match.group(2)),
                         "test_r2": float(match.group(3)), "seconds": int(match.group(4))})
    return {"log": file_record(log), "runs": runs}


def main() -> None:
    checks = {"label_spread": label_spread(), "reproduction_gate_run_times": recorded_run_times()}
    OUT.write_text(json.dumps(strict_json(checks), indent=1, allow_nan=False) + "\n")
    spread = checks["label_spread"]
    print({k: round(v["sd_ddof1"], 4) for k, v in spread.items() if k != "shards"})
    print([(r["model"], r["seed"], r["seconds"]) for r in checks["reproduction_gate_run_times"]["runs"]])
    print("wrote", config.portable(OUT))


if __name__ == "__main__":
    main()
