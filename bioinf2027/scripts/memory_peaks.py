"""Other users' memory on the shared server by hour of day, from the memory governor's one-minute samples
(bioinf2027/scripts/memory_governor.py; $IGA_STORAGE_ROOT/bioinf2027/logs/governor/samples-*.jsonl).

Prints two lines for the status lines of bioinf2027/PROGRESS.md (gate G4, item 4, and the author's change of
2026-11-15, item 4; bioinf2027/GATE.md):
- per hour of day (local time), the other users' peak (the largest one-minute total of their resident memory, in
  GB), the machine's lowest available memory and the minutes past the hour at which a spike began (a one-minute
  sample with the other users above 30 GB after one at or below it), with the number of samples and their span;
- E2's fits saved in the last six hours (prediction files under $IGA_STORAGE_ROOT/bioinf2027/E2/predictions/ by
  modification time; a reused fit is read, not written again) and the files in all.
Other users appear as one total only (gate G4, item 4: sizes, no names).
Usage: memory_peaks.py [--json]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402


def samples() -> list[dict]:
    out = []
    for path in sorted((config.storage_root() / "bioinf2027" / "logs" / "governor").glob("samples-*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


SPIKE_GB = 30.0


def by_hour(rows: list[dict]) -> dict:
    hours: dict = {}
    before = None
    for r in rows:
        hour = int(r["time"][11:13])
        h = hours.setdefault(hour, {"n": 0, "other_users_peak_gb": 0.0, "lowest_available_gb": float("inf"), "spikes_at_minute": []})
        h["n"] += 1
        h["other_users_peak_gb"] = max(h["other_users_peak_gb"], r["other_users_gb"])
        h["lowest_available_gb"] = min(h["lowest_available_gb"], r["available_gb"])
        if r["other_users_gb"] > SPIKE_GB and before is not None and before <= SPIKE_GB:
            minute = int(r["time"][14:16])
            if minute not in h["spikes_at_minute"]:
                h["spikes_at_minute"].append(minute)
        before = r["other_users_gb"]
    for h in hours.values():
        h["spikes_at_minute"].sort()
    return dict(sorted(hours.items()))


def e2_recent(hours: float = 6.0) -> dict:
    folder = config.storage_root() / "bioinf2027" / "E2" / "predictions"
    now = time.time()
    times = [p.stat().st_mtime for p in folder.rglob("*.npz") if ".tmp" not in p.name] if folder.is_dir() else []
    return {"saved_last_%g_hours" % hours: sum(1 for t in times if now - t <= hours * 3600), "files": len(times),
            "newest": time.strftime("%Y-%m-%d %H:%M", time.localtime(max(times))) if times else None}


def line(rows: list[dict]) -> str:
    if not rows:
        return "other users' memory by hour of day: no samples yet"
    hours = by_hour(rows)
    parts = ["%02dh %.0f/%.0f%s" % (h, v["other_users_peak_gb"], v["lowest_available_gb"],
                                    " spikes at %s" % " ".join(":%02d" % m for m in v["spikes_at_minute"]) if v["spikes_at_minute"] else "")
             for h, v in hours.items()]
    return ("other users' memory by hour of day (their peak / the machine's lowest available, GB, and the minutes past the hour at which "
            "they rose above %.0f GB; %d one-minute samples, %s to %s, `$IGA_STORAGE_ROOT/bioinf2027/logs/governor/samples-*.jsonl`): %s"
            % (SPIKE_GB, len(rows), rows[0]["time"][:16], rows[-1]["time"][:16], ", ".join(parts)))


def e2_line(recent: dict) -> str:
    return ("E2's fits saved in the last six hours: %d (prediction files under `$IGA_STORAGE_ROOT/bioinf2027/E2/predictions/`: %d in all, "
            "newest %s)" % (recent["saved_last_6_hours"], recent["files"], recent["newest"]))


if __name__ == "__main__":
    rows = samples()
    recent = e2_recent()
    if "--json" in sys.argv[1:]:
        print(json.dumps({"samples": len(rows), "by_hour": by_hour(rows), "e2": recent, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    else:
        print(line(rows))
        print(e2_line(recent))
