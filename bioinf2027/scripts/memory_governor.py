"""Memory rules for the analysis jobs on the shared server (gate G4, item 4, and the standing rule for stops as changed
by the author, all of 2026-11-15; bioinf2027/GATE.md).

Every second it reads the machine's available memory (MemAvailable) and the jobs registered in a folder, and:
- below 8 GB available, acts on the running job of the highest rank (a filling job 5, smoke runs 4, E4 3, E2 2, E5 1): a job that
  runs one process per fit (shed in its job file) loses one worker at a time, newest fit first, with a pause of 3 s
  so that the freed memory shows before the next decision, and its main process is stopped only when the machine is
  still below 8 GB with no worker left; any other job is stopped, with a pause of 15 s (the author's change of
  2026-11-15 to how E2 meets the spikes);
- gives a job its shed workers back once the machine has shown at least 15 GB available for 5 minutes since the
  last shed, within the 40 GB; a shed is not a stop (no back-off, and E2's 30 minutes run on);
- stops E4 as soon as the machine shows less than 30 GB available ("E4 tier 3 runs only while the machine shows at
  least 30 GB available");
- stops the job of the highest rank when my jobs' summed resident memory exceeds 40 GB (gate G3, item 5);
- widens a job that reads its worker count from a file (E2's fits, 3 to 6) once the machine has shown at least
  30 GB available for ten minutes and the 40 GB allow it;
- starts and restarts the jobs described by spec files, one at a time in their order (E2 before E5 before E4):
  E2 and E5 once the machine has shown at least 15 GB available for ten minutes, E4 on as many workers as leave
  30 GB available after it starts (judged on the lowest reading of the last ten minutes), each within the 40 GB;
- runs E2 alone first: any other job starts only after the current run of every E2 batch that has not ended has gone
  30 minutes without a stop and the machine still shows at least 20 GB available;
- backs off after every stop of a whole job: a stopped job starts again no earlier than 30 minutes after the stop,
  60 minutes after a third stop within six hours, and then only under the rules above; no job is ever held;
- wakes the operator, who goes to the author, when a job saves no new fit or task for eight hours, or when a run is
  ended from outside (no last line from its launcher, a SIGKILL it did not send, or a broken worker pool);
- writes every action to an event log, the events that need the operator also to a wake file, and once a minute a
  sample (available memory, mine, other users' total) to a samples file beside the event log.

A filling job (a spec with fill; E4 tier 3 under the author's instruction of 2026-11-16, bioinf2027/GATE.md) runs on
whatever cores and memory the other jobs leave: it starts on as many workers as the free cores (the machine's
physical cores less the one-minute load average of all users together, its own workers not counted) and the memory allow, the memory being the 40 GB less mine
and less the estimates of the jobs still waiting to start, with the machine left at 15 GB or more after them (judged
on the lowest reading of the last ten minutes); it gains one worker at a time by the same rule, at most every two
minutes and not within five minutes of a shed; the 30 GB rule of E4 does not apply to it; above the 40 GB budget it
loses one worker instead of a job being stopped; and when a job of E2 or E5 is due to start but held back by the
budget, it gives back one worker at a time. The cores are physical cores: the two threads of one core share it, and
nice does not keep a thread from slowing the other thread of its core, so while the load average exceeds the physical
cores by half a thread or more, the filling job gives back one worker at a time, at most every three minutes (E2's
and E5's jobs keep priority; the author's instruction of 2026-11-16). Shed workers come back only through this rule.
Its eight hours without a saved fit run from the newest saved file or from the spec's progress_from (its release),
whichever is later.

A job is stopped by SIGTERM to its script process and the script's descendants (SIGKILL after 5 s), so the launcher
still writes the run record with the return code. A run whose processes end without the launcher's last line was
stopped from outside (the session's memory guard or the kernel); it counts as a stop. Other users' processes are
read as sizes only (no names, no command lines). Run it with the system interpreter; it imports the standard
library only.

Usage: memory_governor.py --registry DIR --events FILE --launcher PATH --log-dir DIR
- DIR/<label>.job (written by the launcher): label, kind (e2, e4, e5 or smoke), rank, pid and start of the script
  process (start time in clock ticks), log, base (the spec's name, if any), and for a widening job workers_file,
  target_workers, per_worker_gb; shed (true) for a job whose fits run one process each, the script process's
  children (it kills its newest fit's process when the count in workers_file falls; the governor does so itself
  after 5 s, writing the process ids to <workers_file>.killed first, so that the script queues those fits again);
  the governor adds restore_to and last_shed while workers are shed.
- DIR/<base>.spec: base, kind, rank, order, item, script, args, nice, oom_adj, estimate_gb (memory at start, for
  the 40 GB), and either workers {file, initial, target, per_worker_gb} or sized {per_worker_gb, parent_gb,
  max_workers} (the worker count chosen at each start and passed as --workers); shed (true) passed to the job.
  Optionally progress: folders whose newest saved .npz or .pkl file shows the job's last saved fit or task, and
  progress_from (epoch): the eight hours run from no earlier than this. Optionally fill {max_workers, per_worker_gb,
  parent_gb} with workers {file, initial, target 0, per_worker_gb} and shed (true): a filling job (above).
- DIR/<base>.state (kept here): status (pending, running, stopping, done, failed), runs, stops (times), not_before
  (the end of the back-off), first_seen, sheds (times of the last 24 hours).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

LOW_GB = 8.0
E4_FLOOR_GB = 30.0
RESTART_FLOOR_GB = 15.0
BUDGET_GB = 40.0
WINDOW_S = 600
PAUSE_S = 15
SHED_PAUSE_S = 3
SHED_WAIT_S = 5
RESTORE_S = 300
RESTORE_FLOOR_GB = 15.0
GRACE_S = 5
STALL_S = 900
START_GAP_S = 60
BACKOFF_S = 1800
BACKOFF_LONG_S = 3600  # after a third stop within six hours
THIRD_WINDOW_S = 6 * 3600
E2_ALONE_S = 1800
E2_ALONE_FLOOR_GB = 20.0
FILL_GAP_S = 120  # a filling job gains at most one worker every two minutes
CPU_OVER = 0.5  # a filling job gives back a worker while the load average exceeds the physical cores by this much
CPU_SHED_GAP_S = 180  # at most one such worker every three minutes: the one-minute load average then shows the last one
NO_PROGRESS_S = 8 * 3600
PROGRESS_EVERY_S = 600
OUTSIDE = re.compile(r"return code -9\b|BrokenProcessPool|terminated abruptly|Killed")
PAGE = os.sysconf("SC_PAGE_SIZE")
UID = os.getuid()
MINE = "venvs/bioinf2027"  # my analysis processes: the pinned environment's interpreter
FAILURE = re.compile(r"Traceback|FAILED|Killed|has not passed|check failed|MemoryError")
EXIT = re.compile(r"^exit (-?\d+)$", re.M)


def available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable missing")


def processes() -> dict:
    """pid -> ppid, uid, state, rss (GB), start (ticks), cpu (ticks) and, for my processes only, the command line."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                stat = f.read()
            rest = stat[stat.rindex(")") + 2:].split()
            uid = os.stat(f"/proc/{name}").st_uid
            cmd = ""
            if uid == UID:
                with open(f"/proc/{name}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
            out[int(name)] = {"ppid": int(rest[1]), "uid": uid, "state": rest[0], "rss": int(rest[21]) * PAGE / 2**30,
                              "start": int(rest[19]), "cpu": int(rest[11]) + int(rest[12]), "cmd": cmd}
        except (OSError, ValueError, IndexError):
            continue
    return out


def tree(pid: int, procs: dict) -> list[int]:
    children: dict = {}
    for p, v in procs.items():
        children.setdefault(v["ppid"], []).append(p)
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))
    return out


def physical_cores(root: Path = Path("/sys/devices/system/cpu")) -> int:
    """The machine's physical cores (distinct package and core identifiers), or the logical CPU count where the
    topology cannot be read."""
    seen = set()
    for topology in root.glob("cpu[0-9]*/topology"):
        try:
            seen.add((topology.joinpath("physical_package_id").read_text().strip(), topology.joinpath("core_id").read_text().strip()))
        except OSError:
            continue
    return len(seen) or os.cpu_count() or 1


CORES = physical_cores()


def load_average() -> float:
    """The machine's one-minute load average (all users together; no process is named)."""
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return float(CORES)


def stamp(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def read_text(path) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


class Governor:
    def __init__(self, registry: Path, events: Path, launcher: str = "", log_dir: str = "") -> None:
        self.registry, self.events, self.launcher, self.log_dir = registry, events, launcher, log_dir
        self.wake = registry / "wake.log"
        self.history: deque = deque()  # (time, available GB)
        self.last_stop = 0.0
        self.next_action = 0.0  # no shed or stop before this time (3 s after a shed, 15 s after a stop)
        self.shed_waits: dict = {}  # label -> (deadline, workers allowed): the script sheds itself before the deadline
        self.last_start = 0.0
        self.last_low_note = 0.0
        self.last_sample = 0.0
        self.last_progress = 0.0
        self.samples = events.with_name(events.stem.replace("events", "samples") + ".jsonl")
        self.killing: list = []  # (deadline, pid, start)
        self.cpu: dict = {}  # label -> (cpu ticks, time it last changed, warned)
        self.flagged: set = set()
        self.children: list = []  # launchers started here, reaped when they end
        self.last_fill: dict = {}  # label -> time a filling job last gained a worker
        self.fill_jobs: list = []  # the filling jobs alive at this tick
        self.procs: dict = {}

    # ------------------------------------------------------------------ records

    def log(self, text: str, wake: bool = False) -> None:
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S %Z"), text)
        with open(self.events, "a") as f:
            f.write(line + "\n")
        if wake:
            with open(self.wake, "a") as f:
                f.write(line + "\n")

    def specs(self) -> list[dict]:
        out = []
        for path in self.registry.glob("*.spec"):
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda s: s["order"])

    def state(self, base: str) -> dict:
        path = self.registry / (base + ".state")
        if path.is_file():
            return json.loads(path.read_text())
        return {"status": "pending", "runs": [], "stops": []}

    def save_state(self, base: str, state: dict) -> None:
        path = self.registry / (base + ".state")
        path.with_suffix(".tmp").write_text(json.dumps(state, indent=1))
        os.replace(path.with_suffix(".tmp"), path)

    def jobs(self, procs: dict) -> list[dict]:
        out = []
        for path in sorted(self.registry.glob("*.job")):
            try:
                job = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            job["path"] = path
            p = procs.get(job["pid"])
            job["alive"] = p is not None and p["start"] == job["start"] and p["state"] != "Z"
            out.append(job)
        return out

    # ------------------------------------------------------------------ stops

    def count_stop(self, base: str | None, reason: str, final: bool) -> None:
        """Counts a stop of a spec job. Until its processes have ended (final) the job is "stopping" and not started."""
        if not base:
            return
        state = self.state(base)
        now = time.time()
        state["stops"] = [t for t in state["stops"] if now - t <= THIRD_WINDOW_S] + [now]
        if state["runs"]:
            state["runs"][-1]["stopped"] = {"time": stamp(now), "reason": reason}
        wait = BACKOFF_LONG_S if len(state["stops"]) >= 3 else BACKOFF_S
        state["not_before"] = now + wait
        self.log("%s: stop %d within six hours; not started again before %s" % (base, len(state["stops"]), stamp(now + wait)))
        state["status"] = "pending" if final else "stopping"
        self.save_state(base, state)

    def stop(self, job: dict, procs: dict, reason: str, wake: bool) -> None:
        pids = tree(job["pid"], procs)
        for p in pids:
            try:
                os.kill(p, signal.SIGTERM)
            except OSError:
                pass
            if p in procs:
                self.killing.append((time.time() + GRACE_S, p, procs[p]["start"]))
        rss = sum(procs[p]["rss"] for p in pids if p in procs)
        job_file = dict(json.loads(job["path"].read_text()), stopped={"time": stamp(), "reason": reason})
        job["path"].write_text(json.dumps(job_file))
        self.last_stop = time.time()
        self.next_action = self.last_stop + PAUSE_S
        self.count_stop(job.get("base"), reason, final=False)
        self.log("stopped %s (%d processes, %.1f GB): %s" % (job["label"], len(pids), rss, reason), wake=wake)

    # ------------------------------------------------------------------ sheds

    @staticmethod
    def workers_of(job: dict, procs: dict) -> list[int]:
        """The live processes of a shedding job's fits in progress: its script process's children, oldest first."""
        out = [p for p, v in procs.items() if v["ppid"] == job["pid"] and v["state"] != "Z" and v["uid"] == UID]
        return sorted(out, key=lambda p: (procs[p]["start"], p))

    @staticmethod
    def update_job(job: dict, **fields) -> None:
        try:
            current = json.loads(job["path"].read_text())
        except (OSError, ValueError):
            return
        for k, v in fields.items():
            if v is None:
                current.pop(k, None)
            else:
                current[k] = v
        temporary = job["path"].with_name(job["path"].name + ".tmp")
        temporary.write_text(json.dumps(current))
        os.replace(temporary, job["path"])
        job.update(fields)

    def shed(self, job: dict, procs: dict, avail: float, mine: float, reason: str | None = None) -> None:
        """One worker fewer (newest fit first); the main process is stopped only with no worker left. A filling job's
        shed workers come back through the filling rule, not by restore_to."""
        workers = self.workers_of(job, procs)
        try:
            current = int(Path(job["workers_file"]).read_text().strip())
        except (OSError, ValueError):
            current = len(workers)
        now = time.time()
        if not workers and current <= 0:
            self.stop(job, procs, "machine at %.1f GB available (below %.0f GB) with no worker left; mine %.1f GB" % (avail, LOW_GB, mine),
                      wake=job["kind"] != "e4")
            return
        allowed = max(0, min(current, len(workers)) - 1) if workers else 0
        Path(job["workers_file"]).write_text("%d\n" % allowed)
        if "restore_to" not in job and not job.get("fill"):
            self.update_job(job, restore_to=current)
        self.update_job(job, last_shed=now)
        self.next_action = now + SHED_PAUSE_S
        if workers:
            self.shed_waits[job["label"]] = (now + SHED_WAIT_S, allowed)
        if job.get("base"):
            state = self.state(job["base"])
            state["sheds"] = [t for t in state.get("sheds", []) if now - t <= 86400] + [now]
            self.save_state(job["base"], state)
        self.log("shed %s: %d to %d workers (%s)" % (job["label"], len(workers), allowed, reason or "machine at %.1f GB available, below %.0f GB; mine %.1f GB"
                                                                                     % (avail, LOW_GB, mine)))

    def finish_sheds(self, jobs: list[dict], procs: dict) -> None:
        """A script that has not shed within 5 s: the governor kills its newest fit's process itself."""
        now = time.time()
        for job in jobs:
            wait = self.shed_waits.get(job["label"])
            if wait is None or now < wait[0]:
                continue
            del self.shed_waits[job["label"]]
            if not job["alive"]:
                continue
            workers = self.workers_of(job, procs)
            if len(workers) > wait[1]:  # recorded first, so that the script queues these fits again
                with open(job["workers_file"] + ".killed", "a") as f:
                    f.write("".join("%d\n" % p for p in workers[wait[1]:]))
            for p in workers[wait[1]:][::-1]:
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass
            if len(workers) > wait[1]:
                self.log("%s had not shed within %d s: %d of its fit processes killed here, newest first" % (job["label"], SHED_WAIT_S, len(workers) - wait[1]),
                         wake=True)

    def restore(self, alive: list[dict], mine: float, now: float) -> bool:
        """Shed workers back once the machine has shown 15 GB for 5 minutes since the last shed; True while any wait."""
        waiting = False
        recent = [a for t, a in self.history if now - t <= RESTORE_S]
        covered = bool(self.history) and now - self.history[0][0] >= RESTORE_S - 5
        for j in alive:
            if not j.get("shed") or "restore_to" not in j:
                continue
            try:
                current = int(Path(j["workers_file"]).read_text().strip())
            except (OSError, ValueError):
                continue
            if current >= j["restore_to"]:
                self.update_job(j, restore_to=None, last_shed=None)
                continue
            waiting = True
            extra = (j["restore_to"] - current) * j.get("per_worker_gb", 2.0)
            if now - j.get("last_shed", 0) >= RESTORE_S and covered and min(recent) >= RESTORE_FLOOR_GB and mine + extra <= BUDGET_GB:
                Path(j["workers_file"]).write_text("%d\n" % j["restore_to"])
                self.log("%s: workers back from %d to %d (at least %.0f GB available for %d minutes; mine %.1f GB)"
                         % (j["label"], current, j["restore_to"], min(recent), RESTORE_S // 60, mine))
                self.update_job(j, restore_to=None, last_shed=None)
        return waiting

    def finish_kills(self, procs: dict) -> None:
        now, keep = time.time(), []
        for deadline, p, start in self.killing:
            if p not in procs or procs[p]["start"] != start:
                continue
            if now >= deadline:
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass
            else:
                keep.append((deadline, p, start))
        self.killing = keep

    # ------------------------------------------------------------------ ends of runs

    def settle(self, specs: list[dict], jobs: list[dict]) -> None:
        """Ends of the spec runs: done, failed, stopped here (already counted) or stopped from outside."""
        for spec in specs:
            state = self.state(spec["base"])
            if state["status"] not in ("running", "stopping") or not state["runs"]:
                continue
            run = state["runs"][-1]
            if time.time() - run["started_epoch"] < 90:  # the launcher registers the script within a minute
                continue
            text = read_text(run["log"])
            found = EXIT.findall(text)
            alive = any(j["alive"] and j["label"] == run["label"] for j in jobs)
            if found:
                code = int(found[-1])
                run.update(ended=stamp(), exit=code)
                if "stopped" in run:  # stopped here; counted when stopped
                    state["status"] = "pending"
                elif code == 0:
                    state["status"] = "done"
                    self.log("%s ended (exit 0)" % run["label"], wake=True)
                else:
                    state["status"] = "failed"
                    outside = OUTSIDE.search(text)
                    self.log("%s failed (exit %d); not started again%s" % (run["label"], code, "; %r in its log: possibly killed by the kernel, "
                             "to the author" % outside.group(0) if outside else ""), wake=True)
                self.save_state(spec["base"], state)
            elif not alive and time.time() - run.get("seen_dead", time.time()) > 30:
                run.update(ended=stamp(), exit=None)
                if "stopped" in run:  # stopped here, and the launcher wrote no last line
                    state["status"] = "pending"
                    self.save_state(spec["base"], state)
                    continue
                self.save_state(spec["base"], state)
                self.count_stop(spec["base"], "stopped from outside (the session's memory guard or the kernel)", final=True)
                self.log("%s stopped from outside (no exit line): killed by the kernel or the session's guard, to the author" % run["label"], wake=True)
            elif not alive and "seen_dead" not in run:
                run["seen_dead"] = time.time()
                self.save_state(spec["base"], state)

    # ------------------------------------------------------------------ starts

    def start(self, spec: dict, workers: int | None, low: float, mine: float, initial: int | None = None) -> None:
        base = spec["base"]
        label = "%s-%s" % (base, time.strftime("%Y%m%dT%H%M")) + ("-%dw" % workers if workers else "")
        log = str(Path(self.log_dir) / (label + ".out"))
        env = dict(os.environ, NICE=str(spec.get("nice", 0)), OOM_ADJ=str(spec["oom_adj"]), RANK=str(spec["rank"]), KIND=spec["kind"],
                   BASE=base, REGISTRY=str(self.registry))
        args = list(spec["args"])
        if spec.get("shed"):
            env["SHED"] = "1"
        if spec.get("workers"):
            w = spec["workers"]
            Path(w["file"]).parent.mkdir(parents=True, exist_ok=True)
            Path(w["file"]).write_text("%d\n" % (w["initial"] if initial is None else initial))
            env.update(WORKERS_FILE=w["file"], TARGET_WORKERS=str(w["target"]), PER_WORKER_GB=str(w["per_worker_gb"]))
        if workers:
            args += ["--workers", str(workers)]
        self.children.append(subprocess.Popen([self.launcher, spec["item"], label, log, spec["script"], *args], env=env,
                                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        state = self.state(base)
        state["status"] = "running"
        state["runs"].append({"label": label, "log": log, "started": stamp(), "started_epoch": time.time(), "workers": workers})
        self.save_state(base, state)
        self.last_start = time.time()
        if spec.get("fill"):  # a start counts as an added worker: the next one comes two minutes later at the earliest
            self.last_fill[label] = self.last_start
        count = workers or initial
        self.log("started %s%s (lowest available reading of the last ten minutes %.1f GB; mine %.1f GB)"
                 % (label, " on %d worker%s" % (count, "" if count == 1 else "s") if count else "", low, mine), wake=spec["kind"] != "e4")

    @staticmethod
    def e2_alone_done(state: dict, avail: float, now: float) -> bool:
        """E2 first: its current run has gone 30 minutes without a stop and the machine still shows 20 GB, or E2 ended."""
        if state["status"] in ("done", "failed"):
            return True
        if state["status"] != "running" or not state["runs"]:
            return False
        run = state["runs"][-1]
        return "stopped" not in run and now - run["started_epoch"] >= E2_ALONE_S and avail >= E2_ALONE_FLOOR_GB

    def reserve(self, specs: list[dict]) -> float:
        """The memory that the jobs waiting to start (not filling) will need: the sum of their estimates."""
        return sum(s.get("estimate_gb", 0.0) for s in specs if not s.get("fill") and self.state(s["base"])["status"] == "pending")

    @staticmethod
    def fill_room(fill: dict, current: int, low: float, mine: float, reserve: float, load: float, parent: bool) -> int:
        """The workers a filling job may have: within the free cores (the physical cores less the load average of
        everything but the job's own workers), the 40 GB less mine and the reserve, and the machine left at 15 GB or
        more after them."""
        per = fill["per_worker_gb"]
        base = fill.get("parent_gb", 0.0) if parent else 0.0
        by_cores = int(max(0.0, CORES - max(0.0, load - current)))
        by_budget = current + int(max(0.0, BUDGET_GB - mine - reserve - base) // per)
        by_floor = current + int(max(0.0, low - RESTART_FLOOR_GB - reserve - base) // per)
        return max(0, min(fill["max_workers"], by_cores, by_budget, by_floor))

    def make_room(self, spec: dict, avail: float, mine: float) -> None:
        """A job of E2 or E5 due to start and held back by the budget: a filling job gives back one worker."""
        if time.time() < self.next_action:
            return
        for job in self.fill_jobs:
            if self.workers_of(job, self.procs):
                self.shed(job, self.procs, avail, mine, reason="for %s to start within the %.0f GB budget; mine %.1f GB" % (spec["base"], BUDGET_GB, mine))
                return

    def fill(self, specs: list[dict], low: float | None, avail: float, mine: float, now: float) -> None:
        """A filling job gives back one worker (at most every three minutes) while the load average exceeds the physical
        cores by half a thread or more; otherwise it gains one worker when the free cores and memory allow (at most
        every two minutes, not within five minutes of a shed)."""
        load = None
        for job in self.fill_jobs:
            try:
                current = int(Path(job["workers_file"]).read_text().strip())
            except (OSError, ValueError):
                continue
            load = load_average() if load is None else load
            if current >= 1 and load >= CORES + CPU_OVER:
                if now - max(job.get("last_shed") or 0, self.last_fill.get(job["label"], 0)) >= CPU_SHED_GAP_S:
                    self.shed(job, self.procs, avail, mine, reason="load average %.1f on %d physical cores; the jobs of E2 and E5 keep their cores"
                                                                   % (load, CORES))
                continue
            if low is None or now - (job.get("last_shed") or 0) < RESTORE_S or now - self.last_fill.get(job["label"], 0) < FILL_GAP_S:
                continue
            reserve = self.reserve(specs)
            if self.fill_room(job["fill"], current, low, mine, reserve, load, parent=False) > current:
                Path(job["workers_file"]).write_text("%d\n" % (current + 1))
                self.last_fill[job["label"]] = now
                self.log("%s: one worker added, %d to %d (load %.1f of %d physical cores; lowest available reading of the last ten minutes %.1f GB; "
                         "mine %.1f GB; %.1f GB kept for jobs waiting to start)" % (job["label"], current, current + 1, load, CORES, low, mine, reserve))

    def starts(self, specs: list[dict], low: float | None, avail: float, mine: float, short: bool) -> None:
        now = time.time()
        if low is None or now - self.last_start < START_GAP_S:
            return
        e2 = [self.state(s["base"]) for s in specs if s["kind"] == "e2"]
        for spec in specs:
            state = self.state(spec["base"])
            if state["status"] != "pending":  # running, stopping, done or failed
                continue
            if now < state.get("not_before", 0):  # backing off after a stop; the later jobs wait too
                return
            if spec["kind"] != "e2" and not all(self.e2_alone_done(st, avail, now) for st in e2):  # every E2 batch that has not ended
                return
            if spec.get("fill"):
                if low >= RESTART_FLOOR_GB:
                    workers = self.fill_room(spec["fill"], 0, low, mine, self.reserve(specs), load_average(), parent=True)
                    if workers >= 1:
                        self.start(spec, None, low, mine, initial=workers)
                return
            if spec["kind"] == "e4":
                sized = spec["sized"]
                if short:
                    return
                by_floor = int((low - E4_FLOOR_GB - sized["parent_gb"]) // sized["per_worker_gb"])
                by_budget = int((BUDGET_GB - mine - sized["parent_gb"]) // sized["per_worker_gb"])
                workers = min(by_floor, by_budget, sized["max_workers"])
                if workers >= 1:
                    self.start(spec, workers, low, mine)
                return
            if low >= RESTART_FLOOR_GB and mine + spec["estimate_gb"] <= BUDGET_GB:
                self.start(spec, None, low, mine)
            elif low >= RESTART_FLOOR_GB:  # held back by the budget only: a filling job gives way
                self.make_room(spec, avail, mine)
            return  # one start per minute, in order: a later job waits for the earlier ones

    # ------------------------------------------------------------------ the tick

    def window_low(self) -> float | None:
        """The lowest available memory of the last ten minutes, or None before ten minutes of readings."""
        if not self.history or time.time() - self.history[0][0] < WINDOW_S - 5:
            return None
        return min(a for _, a in self.history)

    def tick(self) -> None:
        now = time.time()
        avail = available_gb()
        self.history.append((now, avail))
        while self.history and now - self.history[0][0] > WINDOW_S:
            self.history.popleft()
        self.children = [c for c in self.children if c.poll() is None]
        procs = processes()
        self.finish_kills(procs)
        mine = sum(v["rss"] for v in procs.values() if v["uid"] == UID and MINE in v["cmd"])
        jobs = self.jobs(procs)
        specs = self.specs()
        fills = {s["base"]: s["fill"] for s in specs if s.get("fill")}
        for j in jobs:
            if j.get("base") in fills:
                j["fill"] = fills[j["base"]]
        self.procs = procs
        self.settle(specs, jobs)
        for j in jobs:
            if not j["alive"]:
                note = "stopped here" if "stopped" in j else "processes ended"
                tail = " | ".join(read_text(j["log"]).strip().splitlines()[-2:])[:300]
                self.log("%s %s; log: %s" % (j["label"], note, tail), wake=("stopped" not in j and not j.get("base")))
                j["path"].rename(j["path"].with_suffix(".ended"))
        alive = [j for j in jobs if j["alive"] and "stopped" not in j]
        self.fill_jobs = [j for j in alive if j.get("fill") and j.get("shed") and j.get("workers_file")]
        for j in alive:  # failure lines and stalls
            if j["label"] not in self.flagged:
                text = read_text(j["log"])
                if FAILURE.search(text):
                    self.flagged.add(j["label"])
                    self.log("failure line in %s: %s" % (j["label"], [x for x in text.splitlines() if FAILURE.search(x)][-1][:300]), wake=True)
            cpu = sum(procs[p]["cpu"] for p in tree(j["pid"], procs) if p in procs)
            last = self.cpu.get(j["label"])
            if last is None or cpu != last[0]:
                self.cpu[j["label"]] = (cpu, now, False)
            elif now - last[1] > STALL_S and not last[2]:
                self.cpu[j["label"]] = (cpu, last[1], True)
                self.log("%s used no CPU for %d s" % (j["label"], now - last[1]), wake=True)
        ranked = sorted(alive, key=lambda j: -j["rank"])
        self.finish_sheds(jobs, procs)
        if avail < LOW_GB and now >= self.next_action and ranked:
            if ranked[0].get("shed") and ranked[0].get("workers_file"):
                self.shed(ranked[0], procs, avail, mine)
            else:
                self.stop(ranked[0], procs, "machine at %.1f GB available (below %.0f GB); mine %.1f GB" % (avail, LOW_GB, mine),
                          wake=ranked[0]["kind"] != "e4")
            return
        if avail < LOW_GB:
            return
        for j in alive:
            if j["kind"] == "e4" and not j.get("fill") and avail < E4_FLOOR_GB:
                self.stop(j, procs, "machine at %.1f GB available (E4 runs only at %.0f GB or more)" % (avail, E4_FLOOR_GB), wake=False)
                return
        if mine > BUDGET_GB and now >= self.next_action and ranked:
            if ranked[0].get("fill") and ranked[0] in self.fill_jobs and self.workers_of(ranked[0], procs):
                self.shed(ranked[0], procs, avail, mine, reason="my jobs at %.1f GB summed RSS, above the %.0f GB budget" % (mine, BUDGET_GB))
            else:
                self.stop(ranked[0], procs, "my jobs at %.1f GB summed RSS (budget %.0f GB)" % (mine, BUDGET_GB), wake=True)
            return
        if now - self.last_sample >= 60:
            self.last_sample = now
            others = sum(v["rss"] for v in procs.values() if v["uid"] != UID)
            with open(self.samples, "a") as f:
                f.write(json.dumps({"time": stamp(now), "available_gb": round(avail, 2), "mine_gb": round(mine, 2),
                                    "other_users_gb": round(others, 2), "running": sorted(j["label"] for j in alive)}) + "\n")
        if avail < 12 and now - self.last_low_note > 60:
            self.last_low_note = now
            others = sum(v["rss"] for v in procs.values() if v["uid"] != UID)
            self.log("low: machine %.1f GB available; mine %.1f GB; other users %.1f GB" % (avail, mine, others))
        low = self.window_low()
        short = self.restore(alive, mine, now)
        for j in alive:  # widening
            if j.get("workers_file") and j.get("target_workers"):
                try:
                    current = int(Path(j["workers_file"]).read_text().strip())
                except (OSError, ValueError):
                    continue
                if current >= j["target_workers"]:
                    continue
                short = True
                extra = (j["target_workers"] - current) * j.get("per_worker_gb", 2.0)
                if low is not None and low >= E4_FLOOR_GB and mine + extra <= BUDGET_GB:
                    Path(j["workers_file"]).write_text("%d\n" % j["target_workers"])
                    if "restore_to" in j:
                        self.update_job(j, restore_to=None, last_shed=None)
                    self.log("%s widened from %d to %d workers (lowest available reading of the last ten minutes %.1f GB; mine %.1f GB)"
                             % (j["label"], current, j["target_workers"], low, mine), wake=True)
        self.fill(specs, low, avail, mine, now)
        self.starts(specs, low, avail, mine, short)
        if now - self.last_progress >= PROGRESS_EVERY_S:
            self.last_progress = now
            self.check_progress(specs)

    def check_progress(self, specs: list[dict]) -> None:
        """Wakes the operator when a job that has not ended has saved no new fit or task for eight hours."""
        now = time.time()
        for spec in specs:
            if not spec.get("progress"):
                continue
            state = self.state(spec["base"])
            if state["status"] in ("done", "failed"):
                continue
            first_seen = state.setdefault("first_seen", now)
            newest = None  # the newest saved fit or task file; the time the job was first seen only when it has none
            for folder in spec["progress"]:
                for root, _, files in os.walk(folder):
                    for name in files:
                        if name.endswith((".npz", ".pkl")) and ".tmp" not in name:
                            try:
                                mtime = os.stat(os.path.join(root, name)).st_mtime
                            except OSError:
                                continue
                            newest = mtime if newest is None else max(newest, mtime)
            newest = max(first_seen if newest is None else newest, spec.get("progress_from", 0.0))
            state["last_saved"] = stamp(newest)
            if now - newest > NO_PROGRESS_S and now - state.get("progress_warned", 0) > NO_PROGRESS_S:
                state["progress_warned"] = now
                self.log("%s has saved no new fit or task for %.1f h: to the author" % (spec["base"], (now - newest) / 3600), wake=True)
            self.save_state(spec["base"], state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    options = parser.parse_args()
    options.registry.mkdir(parents=True, exist_ok=True)
    options.events.parent.mkdir(parents=True, exist_ok=True)
    governor = Governor(options.registry, options.events, options.launcher, options.log_dir)
    governor.log("governor started (pid %d): below %.0f GB shed one worker of the highest rank at a time (%d s apart) or stop it; "
                 "workers back after %d minutes at %.0f GB; E4 only at %.0f GB or more; restarts after ten minutes at %.0f GB (E2, E5) "
                 "or %.0f GB (E4), %d or %d minutes after a stop; others after %d minutes of E2 at %.0f GB; budget %.0f GB"
                 % (os.getpid(), LOW_GB, SHED_PAUSE_S, RESTORE_S // 60, RESTORE_FLOOR_GB, E4_FLOOR_GB, RESTART_FLOOR_GB, E4_FLOOR_GB,
                    BACKOFF_S // 60, BACKOFF_LONG_S // 60, E2_ALONE_S // 60, E2_ALONE_FLOOR_GB, BUDGET_GB))
    while True:
        try:
            governor.tick()
        except Exception as error:  # a reading that fails once must not end the rules
            governor.log("tick failed: %r" % (error,))
        time.sleep(options.interval)


if __name__ == "__main__":
    main()
