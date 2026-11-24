"""Resident memory of one job: the peak of its main process, the peak per worker and the peak of the job's total.

Gate G3, item 5 (bioinf2027/GATE.md, top section): all jobs together stay under 40 GB of resident memory, and
worker counts are set from the measured peak per worker of each job type. Follows the process tree of the given
main process until it ends (or for --seconds), sampling /proc every --interval seconds. Resident memory is summed
over processes (VmRSS, which counts pages shared after fork once per process, so it overstates the job) and, beside
it, as PSS (shared pages divided among the processes that map them). Output bioinf2027/runs/memory/<label>.json,
rewritten after every sample.

    python bioinf2027/scripts/memory_watch.py --label e2-fits --pid <main pid> [--interval 5] [--seconds N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "runs" / "memory"
GB = 1024.0 ** 2  # /proc reports kB


def status(pid: int) -> dict | None:
    try:
        fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
        pss = 0
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                pss = int(line.split()[1])
        return {"rss": int(fields["VmRSS"].split()[0]), "hwm": int(fields["VmHWM"].split()[0]), "pss": pss}
    except (FileNotFoundError, ProcessLookupError, KeyError, PermissionError, ValueError):
        return None


def children(pid: int) -> list[int]:
    out = []
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            out += [int(c) for c in (task / "children").read_text().split()]
        except (FileNotFoundError, ProcessLookupError):
            pass
    return out


def tree(root: int) -> list[int]:
    seen, stack = [], [root]
    while stack:
        pid = stack.pop()
        if pid not in seen:
            seen.append(pid)
            stack += children(pid)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--seconds", type=float, default=None)
    a = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.label}.json"
    started = time.time()
    peak_main = peak_worker = peak_total_rss = peak_total_pss = 0
    workers_seen: set[int] = set()
    samples = 0
    max_workers_at_once = 0
    while True:
        root = status(a.pid)
        if root is None:
            break
        peak_main = max(peak_main, root["hwm"])
        total_rss, total_pss, live_workers = root["rss"], root["pss"], 0
        for pid in tree(a.pid)[1:]:
            s = status(pid)
            if s is None or s["rss"] < 50 * 1024:  # helper processes under 50 MB (shells, the resource tracker) are not workers
                if s is not None:
                    total_rss += s["rss"]
                    total_pss += s["pss"]
                continue
            workers_seen.add(pid)
            live_workers += 1
            peak_worker = max(peak_worker, s["hwm"])
            total_rss += s["rss"]
            total_pss += s["pss"]
        max_workers_at_once = max(max_workers_at_once, live_workers)
        peak_total_rss, peak_total_pss = max(peak_total_rss, total_rss), max(peak_total_pss, total_pss)
        samples += 1
        record = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
                  "last_sample": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "samples": samples, "interval_s": a.interval,
                  "workers_seen": len(workers_seen), "workers_at_once_max": max_workers_at_once,
                  "main_peak_gb": round(peak_main / GB, 3), "worker_peak_gb": round(peak_worker / GB, 3),
                  "total_rss_peak_gb": round(peak_total_rss / GB, 3), "total_pss_peak_gb": round(peak_total_pss / GB, 3),
                  "note": "peaks are VmHWM per process; totals are sums at one sample; RSS counts shared pages once per process"}
        path.write_text(json.dumps(record, indent=1) + "\n")
        if a.seconds is not None and time.time() - started > a.seconds:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
