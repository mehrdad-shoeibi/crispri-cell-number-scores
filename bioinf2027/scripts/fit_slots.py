"""One process per task, so that the memory rules can take one worker away at a time (the author's change of
2026-11-15 to how E2 and E5 meet other users' memory spikes, bioinf2027/GATE.md).

A pool of worker processes cannot lose one worker alone (a killed worker breaks the pool and every task in it), so
each task runs in a process of its own, forked from the calling process when the task starts. The forked process
holds the inputs the caller loaded before the first task, as a pool worker does; the task's result is a function of
the task and of those inputs only, so the way it is run changes no value.

The number of tasks in progress is read from `limit()` before each wait (at most `poll` seconds apart). When it falls
below the number in progress, the process of the newest task is killed and the task goes back to the front of the
queue: it is computed again from its start, and whatever the task saves is written only when it ends. The memory
governor kills the newest task's process itself when the caller has not done so within a few seconds, and writes
the process id to the file `killed` first; a task whose process ends without a result and is listed there goes
back to the queue too. Any other end without a result is a failure, reported as "terminated abruptly".
"""
from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import time
from pathlib import Path


def limit_from(path: Path | None, default: int) -> int:
    """The integer in the file (0 allowed: no task in progress), or the default when there is no file or no integer."""
    if path is not None:
        try:
            return max(0, int(Path(path).read_text().strip()))
        except (OSError, ValueError):
            pass
    return default


def _child(function, task, connection) -> None:
    try:
        result = ("ok", function(task))
    except BaseException as error:  # reported to the caller as the task's failure
        result = ("error", repr(error))
    try:
        connection.send(result)
    finally:
        connection.close()


def taken_by_governor(killed: Path | None, pid: int) -> bool:
    if killed is None:
        return False
    try:
        return str(pid) in Path(killed).read_text().split()
    except OSError:
        return False


def run_slots(listed: list, function, name_of, limit, tag: str, started: float, poll: float = 1.0, every: int = 25,
              killed: Path | None = None) -> tuple[list, list, int]:
    """Runs every task once to its end, in the listed order, each in a process of its own, with at most limit() in
    progress. Returns the results, the failures (each named by name_of) and the number of tasks taken away and
    queued again."""
    context = multiprocessing.get_context("fork")
    queue, running, done, failed = list(listed), [], [], []
    shed, sequence = 0, 0

    def collect() -> None:
        nonlocal shed
        for r in list(running):
            if not r["connection"].poll():  # no result yet and the process still holds its end of the pipe
                continue
            try:
                result = r["connection"].recv()
            except Exception:  # the process ended without a (whole) result
                result = None
            r["process"].join()
            r["connection"].close()
            running.remove(r)
            if result is None:
                code = r["process"].exitcode
                if taken_by_governor(killed, r["process"].pid):
                    queue.insert(0, r["task"])
                    shed += 1
                    print("[%s] worker taken away (exit code %s): %d in progress; %s back in the queue"
                          % (tag, code, len(running), name_of(r["task"])), flush=True)
                else:
                    failed.append({"fit": name_of(r["task"]), "error": "worker terminated abruptly (exit code %s)" % code})
                    print("[%s] FAILED %s: worker terminated abruptly (exit code %s)" % (tag, name_of(r["task"]), code), flush=True)
                continue
            status, value = result
            if status == "ok":
                done.append(value)
                if len(done) % every == 0 or not (queue or running):
                    print("[%s] %d of %d done, %d failed, %.0f s" % (tag, len(done), len(listed), len(failed), time.time() - started), flush=True)
            else:
                failed.append({"fit": name_of(r["task"]), "error": value})
                print("[%s] FAILED %s: %s" % (tag, name_of(r["task"]), value), flush=True)

    try:
        while queue or running:
            collect()
            allowed = limit()
            while len(running) > allowed:  # the newest task's process goes first
                newest = max(running, key=lambda r: r["sequence"])
                newest["process"].kill()
                newest["process"].join()
                newest["connection"].close()
                running.remove(newest)
                queue.insert(0, newest["task"])
                shed += 1
                print("[%s] worker shed: %d in progress (limit %d); %s back in the queue" % (tag, len(running), allowed, name_of(newest["task"])), flush=True)
            while queue and len(running) < allowed:
                task = queue.pop(0)
                receive, send = context.Pipe(duplex=False)
                process = context.Process(target=_child, args=(function, task, send))
                process.start()
                send.close()
                sequence += 1
                running.append({"task": task, "process": process, "connection": receive, "sequence": sequence})
            if running:
                multiprocessing.connection.wait([r["connection"] for r in running] + [r["process"].sentinel for r in running], timeout=poll)
            elif queue:
                time.sleep(poll)
    finally:
        for r in running:
            r["process"].kill()
            r["process"].join()
    return done, failed, shed
