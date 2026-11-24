"""One process per task (bioinf2027/scripts/fit_slots.py): every task ends once with its own value when the worker
count falls, rises, reaches 0, or a task's process is killed from outside."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fit_slots as fs  # noqa: E402

MARK: dict = {}


def _square(x):
    time.sleep(0.2)
    if x == 7:
        raise ValueError("seven")
    if x == 11 and not Path(MARK["path"]).exists():  # killed from outside on its first run
        Path(MARK["path"]).write_text("1")
        if MARK.get("killed"):  # as the governor does: the process id recorded first
            Path(MARK["killed"]).write_text("1\n%d\n" % os.getpid())
        os.kill(os.getpid(), signal.SIGKILL)
    return {"task": x, "value": x * x}


def test_the_limit_comes_from_the_file_and_0_is_allowed(tmp_path):
    path = tmp_path / "workers"
    assert fs.limit_from(None, 3) == 3
    assert fs.limit_from(path, 3) == 3
    path.write_text("0\n")
    assert fs.limit_from(path, 3) == 0
    path.write_text("x")
    assert fs.limit_from(path, 3) == 3


def _slow(x):
    time.sleep(0.6)
    if x == 7:
        raise ValueError("seven")
    return {"task": x, "value": x * x}


def by_time(plan: list):
    """A limit that follows the clock: plan is [(seconds from the first reading, limit), ...]."""
    first = []

    def limit():
        if not first:
            first.append(time.time())
        elapsed = time.time() - first[0]
        return [n for t, n in plan if elapsed >= t][-1]
    return limit


def test_every_task_ends_once_with_its_value_while_the_limit_falls_to_0_and_rises(capsys):
    limit = by_time([(0, 4), (0.4, 2), (1.0, 0), (1.6, 1), (2.2, 4)])
    done, failed, shed = fs.run_slots(list(range(16)), _slow, lambda t: "task %d" % t, limit, "test", time.time(), poll=0.05)
    assert sorted(d["task"] for d in done) == [x for x in range(16) if x != 7]
    assert all(d["value"] == d["task"] ** 2 for d in done)
    assert failed == [{"fit": "task 7", "error": "ValueError('seven')"}]
    out = capsys.readouterr().out
    assert shed >= 2 and "worker shed" in out  # tasks in progress were taken away and run again


def test_the_newest_task_is_shed_first(capsys):
    done, failed, shed = fs.run_slots([0, 1, 2], _slow, str, by_time([(0, 3), (0.4, 2)]), "test", time.time(), poll=0.05)
    assert sorted(d["task"] for d in done) == [0, 1, 2] and shed == 1
    assert "2 back in the queue" in capsys.readouterr().out


def test_a_process_killed_from_outside_is_a_failure_unless_the_governor_recorded_the_kill(tmp_path, capsys):
    MARK["path"], MARK["killed"] = str(tmp_path / "mark"), None
    killed = tmp_path / "workers.killed"
    killed.write_text("1\n")  # another process id only
    done, failed, shed = fs.run_slots([10, 11, 12], _square, lambda t: "task %d" % t, lambda: 3, "test", time.time(), poll=0.05, killed=killed)
    assert [d["task"] for d in sorted(done, key=lambda d: d["task"])] == [10, 12]
    assert failed == [{"fit": "task 11", "error": "worker terminated abruptly (exit code -9)"}]
    assert "FAILED task 11: worker terminated abruptly" in capsys.readouterr().out
    os.remove(MARK["path"])
    MARK["killed"] = str(killed)
    done, failed, shed = fs.run_slots([10, 11, 12], _square, lambda t: "task %d" % t, lambda: 3, "test", time.time(), poll=0.05, killed=killed)
    assert sorted(d["task"] for d in done) == [10, 11, 12] and failed == [] and shed == 1
    assert "worker taken away (exit code -9)" in capsys.readouterr().out
