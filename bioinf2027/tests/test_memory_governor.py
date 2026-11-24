"""The memory rules of gate G4, item 4 (bioinf2027/scripts/memory_governor.py), on stand-in jobs (sleep processes)
with the machine's available memory set by the test."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import memory_governor as mg  # noqa: E402


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    """No test reads the machine's own load average; a test that needs a load sets it."""
    monkeypatch.setattr(mg, "load_average", lambda: 0.0)


def register(folder: Path, label: str, kind: str, rank: int, **extra) -> subprocess.Popen:
    process = subprocess.Popen(["sleep", "120"])
    start = mg.processes()[process.pid]["start"]
    log = folder / (label + ".out")
    log.write_text("running\n")
    (folder / (label + ".job")).write_text(json.dumps({"label": label, "kind": kind, "rank": rank, "pid": process.pid, "start": start,
                                                      "log": str(log), **extra}))
    return process


def governor(tmp_path: Path, monkeypatch, available: list) -> mg.Governor:
    monkeypatch.setattr(mg, "available_gb", lambda: available[0])
    return mg.Governor(tmp_path, tmp_path / "events.log")


def gone(process: subprocess.Popen) -> bool:
    try:
        process.wait(timeout=10)
        return True
    except subprocess.TimeoutExpired:
        return False


def test_below_8_gb_the_highest_rank_stops_first_and_one_at_a_time(tmp_path, monkeypatch):
    available = [6.0]
    g = governor(tmp_path, monkeypatch, available)
    e5, e2 = register(tmp_path, "e5", "e5", 1), register(tmp_path, "e2", "e2", 2)
    try:
        g.tick()
        assert gone(e2) and e5.poll() is None
        assert "stopped e2" in (tmp_path / "wake.log").read_text()
        g.tick()  # within the pause: E5 keeps running
        assert e5.poll() is None
        g.next_action -= mg.PAUSE_S
        g.tick()
        assert gone(e5)
    finally:
        for p in (e5, e2):
            p.kill()


def test_e4_stops_below_30_gb_and_the_others_keep_running(tmp_path, monkeypatch):
    available = [29.0]
    g = governor(tmp_path, monkeypatch, available)
    e2, e4 = register(tmp_path, "e2", "e2", 2), register(tmp_path, "e4", "e4", 3)
    try:
        g.tick()
        assert gone(e4) and e2.poll() is None
        assert "stopped e4" in (tmp_path / "events.log").read_text()
        assert not (tmp_path / "wake.log").exists()  # a routine stop of E4 does not wake the operator
    finally:
        for p in (e2, e4):
            p.kill()


def test_widening_waits_for_ten_minutes_at_30_gb_or_more(tmp_path, monkeypatch):
    available = [35.0]
    g = governor(tmp_path, monkeypatch, available)
    workers = tmp_path / "workers"
    workers.write_text("3\n")
    e2 = register(tmp_path, "e2", "e2", 2, workers_file=str(workers), target_workers=6, per_worker_gb=1.7)
    try:
        g.tick()
        assert workers.read_text().strip() == "3"  # no ten minutes of readings yet
        g.history.appendleft((time.time() - mg.WINDOW_S + 1, 29.0))
        g.tick()
        assert workers.read_text().strip() == "3"  # a reading below 30 GB inside the window
        g.history.clear()
        g.history.append((time.time() - mg.WINDOW_S + 1, 31.0))
        g.tick()
        assert workers.read_text().strip() == "6"
    finally:
        e2.kill()


class FakePopen:
    calls: list = []

    def __init__(self, args, **kwargs):
        FakePopen.calls.append(args)

    def poll(self):
        return None


def write_specs(folder: Path, workers: Path) -> None:
    common = {"item": "X", "script": "s.py", "nice": 0}
    (folder / "e2.spec").write_text(json.dumps({**common, "base": "e2", "kind": "e2", "rank": 2, "order": 1, "args": ["a"], "oom_adj": 900,
                                                "estimate_gb": 7.0, "workers": {"file": str(workers), "initial": 3, "target": 6,
                                                                                "per_worker_gb": 1.7}}))
    (folder / "e4.spec").write_text(json.dumps({**common, "base": "e4", "kind": "e4", "rank": 3, "order": 9, "args": ["--tier", "3"],
                                                "oom_adj": 1000, "estimate_gb": 0.0,
                                                "sized": {"per_worker_gb": 6.9, "parent_gb": 0.7, "max_workers": 8}}))


def test_starts_go_in_order_and_e4_waits_for_the_widening(tmp_path, monkeypatch):
    available = [50.0]
    g = governor(tmp_path, monkeypatch, available)
    FakePopen.calls = []
    monkeypatch.setattr(mg.subprocess, "Popen", FakePopen)
    workers = tmp_path / "control" / "workers"
    write_specs(tmp_path, workers)
    g.tick()
    assert not FakePopen.calls  # no ten minutes of readings yet
    g.history.appendleft((time.time() - mg.WINDOW_S + 1, 50.0))
    g.tick()
    assert len(FakePopen.calls) == 1 and FakePopen.calls[0][-1] == "a"  # E2 first, without --workers
    assert workers.read_text().strip() == "3"
    assert json.loads((tmp_path / "e2.state").read_text())["status"] == "running"
    # E2 is registered as running and short of its target (pid 1 stands in; nothing here signals it)
    (tmp_path / "e2-run.job").write_text(json.dumps({"label": json.loads((tmp_path / "e2.state").read_text())["runs"][-1]["label"],
                                                     "kind": "e2", "rank": 2, "pid": 1, "start": mg.processes()[1]["start"], "base": "e2",
                                                     "log": str(tmp_path / "none.out"), "workers_file": str(workers), "target_workers": 6,
                                                     "per_worker_gb": 1.7}))
    g.last_start -= mg.START_GAP_S
    g.tick()  # widens E2; E4 waits while E2 is short
    assert workers.read_text().strip() == "6" and len(FakePopen.calls) == 1
    g.tick()
    assert len(FakePopen.calls) == 1  # E2 alone first: its run has not yet gone 30 minutes
    s = json.loads((tmp_path / "e2.state").read_text())
    s["runs"][-1]["started_epoch"] -= mg.E2_ALONE_S
    (tmp_path / "e2.state").write_text(json.dumps(s))
    g.tick()
    assert len(FakePopen.calls) == 2 and FakePopen.calls[1][-2:] == ["--workers", "2"]  # (50 - 30 - 0.7) // 6.9 = 2


def test_e5_waits_for_30_minutes_of_e2_without_a_stop_and_20_gb(tmp_path, monkeypatch):
    available = [19.0]
    g = governor(tmp_path, monkeypatch, available)
    FakePopen.calls = []
    monkeypatch.setattr(mg.subprocess, "Popen", FakePopen)
    (tmp_path / "e5.spec").write_text(json.dumps({"base": "e5", "kind": "e5", "rank": 1, "order": 2, "item": "E5", "script": "s.py", "args": ["m"],
                                                "nice": 0, "oom_adj": 800, "estimate_gb": 2.5}))
    (tmp_path / "e2.spec").write_text(json.dumps({"base": "e2", "kind": "e2", "rank": 2, "order": 1, "item": "E2", "script": "s.py", "args": ["a"],
                                                "nice": 0, "oom_adj": 900, "estimate_gb": 7.0}))
    log = tmp_path / "e2.out"
    log.write_text("running\n")
    g.save_state("e2", {"status": "running", "stops": [], "runs": [{"label": "e2-r", "log": str(log), "started": "",
                                                                   "started_epoch": time.time() - mg.E2_ALONE_S - 1}]})
    (tmp_path / "e2-r.job").write_text(json.dumps({"label": "e2-r", "kind": "e2", "rank": 2, "pid": 1, "start": mg.processes()[1]["start"],
                                                   "base": "e2", "log": str(log)}))
    g.history.appendleft((time.time() - mg.WINDOW_S + 1, 19.0))
    g.tick()
    assert not FakePopen.calls  # 19 GB: below the 20 GB that E5 needs beside E2
    available[0] = 21.0
    g.tick()
    assert len(FakePopen.calls) == 1 and FakePopen.calls[0][-1] == "m"


def test_e5_also_waits_for_a_second_e2_batch_after_the_first_has_ended(tmp_path, monkeypatch):
    available = [25.0]
    g = governor(tmp_path, monkeypatch, available)
    FakePopen.calls = []
    monkeypatch.setattr(mg.subprocess, "Popen", FakePopen)
    for base, kind, order in (("e2", "e2", 1), ("e2b", "e2", 2), ("e5", "e5", 3)):
        (tmp_path / (base + ".spec")).write_text(json.dumps({"base": base, "kind": kind, "rank": 2 if kind == "e2" else 1, "order": order,
                                                             "item": "E2", "script": "s.py", "args": [base], "nice": 0, "oom_adj": 900,
                                                             "estimate_gb": 2.0}))
    g.save_state("e2", {"status": "done", "stops": [], "runs": []})
    log = tmp_path / "e2b.out"
    log.write_text("running\n")
    g.save_state("e2b", {"status": "running", "stops": [], "runs": [{"label": "e2b-r", "log": str(log), "started": "",
                                                                    "started_epoch": time.time() - 60}]})
    (tmp_path / "e2b-r.job").write_text(json.dumps({"label": "e2b-r", "kind": "e2", "rank": 2, "pid": 1, "start": mg.processes()[1]["start"],
                                                    "base": "e2b", "log": str(log)}))
    g.history.appendleft((time.time() - mg.WINDOW_S + 1, 25.0))
    g.tick()
    assert not FakePopen.calls  # the second E2 batch has run for one minute only
    state = g.state("e2b")
    state["runs"][-1]["started_epoch"] = time.time() - mg.E2_ALONE_S - 1
    g.save_state("e2b", state)
    g.tick()
    assert len(FakePopen.calls) == 1 and FakePopen.calls[0][-1] == "e5"


def test_every_stop_backs_off_and_no_job_is_held(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    log = tmp_path / "e2-r.out"
    log.write_text("running\n")
    state = {"status": "running", "runs": [{"label": "e2-r", "log": str(log), "started": "", "started_epoch": time.time() - 100}],
             "stops": [time.time() - 7 * 3600, time.time() - 3600]}
    g.save_state("e2", state)
    spec = [{"base": "e2", "order": 1}]
    g.settle(spec, [])  # processes gone, no exit line: wait 30 s before calling it an outside stop
    assert g.state("e2")["status"] == "running"
    s = g.state("e2")
    s["runs"][-1]["seen_dead"] -= 31
    g.save_state("e2", s)
    g.settle(spec, [])
    s = g.state("e2")
    assert s["status"] == "pending" and len(s["stops"]) == 2  # the stop of seven hours ago no longer counts
    assert 1790 < s["not_before"] - time.time() <= 1800  # a second stop within six hours: 30 minutes
    assert "killed by the kernel or the session's guard, to the author" in (tmp_path / "wake.log").read_text()
    s["status"] = "running"
    s["runs"].append({"label": "e2-r2", "log": str(log), "started": "", "started_epoch": time.time() - 100})
    g.save_state("e2", s)
    g.count_stop("e2", "test", final=False)
    assert g.state("e2")["status"] == "stopping"
    log.write_text("running\nexit 241\n")
    g.settle(spec, [])
    s = g.state("e2")
    assert s["status"] == "pending" and len(s["stops"]) == 3
    assert 3590 < s["not_before"] - time.time() <= 3600  # a third stop within six hours: 60 minutes
    FakePopen.calls = []
    monkeypatch.setattr(mg.subprocess, "Popen", FakePopen)
    (tmp_path / "e2.spec").write_text(json.dumps({"base": "e2", "kind": "e2", "rank": 2, "order": 1, "item": "E2", "script": "s.py", "args": [],
                                                "nice": 0, "oom_adj": 900, "estimate_gb": 7.0}))
    g.history.appendleft((time.time() - mg.WINDOW_S + 1, 50.0))
    g.tick()
    assert not FakePopen.calls  # backing off
    s["not_before"] = time.time() - 1
    g.save_state("e2", s)
    g.tick()
    assert len(FakePopen.calls) == 1


def test_no_saved_fit_for_eight_hours_wakes_the_operator(tmp_path, monkeypatch):
    import os
    g = governor(tmp_path, monkeypatch, [50.0])
    fits = tmp_path / "fits"
    fits.mkdir()
    (fits / "a.npz").write_bytes(b"x")
    spec = [{"base": "e4", "order": 9, "progress": [str(fits)]}]
    g.save_state("e4", {"status": "pending", "runs": [], "stops": [], "first_seen": time.time() - 9 * 3600})
    g.check_progress(spec)
    assert not (tmp_path / "wake.log").exists()  # a fit saved just now
    old = time.time() - 8 * 3600 - 60
    os.utime(fits / "a.npz", (old, old))
    g.check_progress(spec)
    assert "saved no new fit or task for 8.0 h" in (tmp_path / "wake.log").read_text()


def test_the_eight_hours_run_from_the_newest_saved_file_even_when_the_job_was_first_seen_later(tmp_path, monkeypatch):
    import os
    g = governor(tmp_path, monkeypatch, [50.0])
    fits = tmp_path / "fits"
    fits.mkdir()
    old = time.time() - 9 * 3600
    (fits / "a.npz").write_bytes(b"x")
    os.utime(fits / "a.npz", (old, old))
    spec = [{"base": "e4", "order": 9, "progress": [str(fits)]}]
    g.save_state("e4", {"status": "pending", "runs": [], "stops": [], "first_seen": time.time() - 60})
    g.check_progress(spec)
    assert "saved no new fit or task for 9.0 h" in (tmp_path / "wake.log").read_text()


def test_a_job_with_no_saved_file_counts_from_when_it_was_first_seen(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    fits = tmp_path / "fits"
    fits.mkdir()
    spec = [{"base": "e4", "order": 9, "progress": [str(fits)]}]
    g.save_state("e4", {"status": "pending", "runs": [], "stops": [], "first_seen": time.time() - 7 * 3600})
    g.check_progress(spec)
    assert not (tmp_path / "wake.log").exists()
    g.save_state("e4", {"status": "pending", "runs": [], "stops": [], "first_seen": time.time() - 8 * 3600 - 60})
    g.check_progress(spec)
    assert "saved no new fit or task for 8.0 h" in (tmp_path / "wake.log").read_text()


# --------------------------------------------------------------------------- shedding (the author's change of 2026-11-15)

PARENT = """
import subprocess, sys, time
path, n, comply = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
children = []
for _ in range(n):
    children.append(subprocess.Popen(["sleep", "120"]))
    time.sleep(0.05)
while True:
    if comply:
        try:
            allowed = int(open(path).read())
        except (OSError, ValueError):
            allowed = n
        live = [c for c in children if c.poll() is None]
        while len(live) > allowed:
            live[-1].kill()
            live[-1].wait()
            live = live[:-1]
    time.sleep(0.2)
"""


def register_shedding(folder: Path, label: str, workers: Path, n: int, comply: bool, **extra) -> subprocess.Popen:
    """A stand-in for a script that runs one process per fit: n children, the newest killed when the file falls."""
    workers.write_text("%d\n" % n)
    process = subprocess.Popen([sys.executable, "-c", PARENT, str(workers), str(n), "1" if comply else "0"])
    for _ in range(100):
        procs = mg.processes()
        if len([p for p, v in procs.items() if v["ppid"] == process.pid]) == n:
            break
        time.sleep(0.05)
    (folder / (label + ".job")).write_text(json.dumps({"label": label, "kind": "e2", "rank": 2, "pid": process.pid, "start": procs[process.pid]["start"],
                                                      "log": str(folder / (label + ".out")), "workers_file": str(workers), "target_workers": n,
                                                      "per_worker_gb": 1.7, "shed": True, **extra}))
    (folder / (label + ".out")).write_text("running\n")
    return process


def children(pid: int) -> list[int]:
    procs = mg.processes()
    return sorted((p for p, v in procs.items() if v["ppid"] == pid and v["state"] != "Z"), key=lambda p: (procs[p]["start"], p))


def settle_children(pid: int, n: int) -> list[int]:
    for _ in range(50):
        now = children(pid)
        if len(now) == n:
            return now
        time.sleep(0.1)
    return children(pid)


def test_below_8_gb_a_shedding_job_loses_one_worker_at_a_time_newest_first_and_is_stopped_only_with_none_left(tmp_path, monkeypatch):
    available = [6.0]
    g = governor(tmp_path, monkeypatch, available)
    workers = tmp_path / "workers"
    e2 = register_shedding(tmp_path, "e2", workers, 3, comply=True, base="e2")
    try:
        first = children(e2.pid)
        assert len(first) == 3
        g.tick()
        assert workers.read_text().strip() == "2"
        assert settle_children(e2.pid, 2) == first[:2]  # the newest fit's process went first
        g.tick()  # within the 3 s pause: nothing more
        assert workers.read_text().strip() == "2"
        for left in (1, 0):
            g.next_action = 0
            g.tick()
            assert workers.read_text().strip() == str(left)
            assert settle_children(e2.pid, left) == first[:left]
        assert e2.poll() is None  # the main process runs on while a worker was left to shed
        state = json.loads((tmp_path / "e2.state").read_text())
        assert len(state["sheds"]) == 3 and state["stops"] == [] and "not_before" not in state  # a shed is not a stop
        assert not (tmp_path / "wake.log").exists()
        g.next_action = 0
        g.tick()  # still below 8 GB with no worker left: the main process is stopped
        assert gone(e2)
        state = json.loads((tmp_path / "e2.state").read_text())
        assert len(state["stops"]) == 1 and "not_before" in state  # a whole-job stop backs off
        assert "with no worker left" in (tmp_path / "wake.log").read_text()
    finally:
        e2.kill()
        for p in children(e2.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_a_script_that_does_not_shed_within_5_s_has_its_newest_fit_killed_by_the_governor(tmp_path, monkeypatch):
    available = [6.0]
    g = governor(tmp_path, monkeypatch, available)
    workers = tmp_path / "workers"
    e2 = register_shedding(tmp_path, "e2", workers, 2, comply=False)
    try:
        first = children(e2.pid)
        g.tick()
        assert workers.read_text().strip() == "1"
        time.sleep(0.5)
        assert children(e2.pid) == first  # the script has not shed
        g.shed_waits["e2"] = (0.0, g.shed_waits["e2"][1])  # the 5 s have passed
        available[0] = 9.0
        g.tick()
        assert settle_children(e2.pid, 1) == first[:1]
        assert "had not shed" in (tmp_path / "wake.log").read_text()
        assert (tmp_path / "workers.killed").read_text().split() == [str(first[1])]  # recorded for the script
    finally:
        e2.kill()
        for p in children(e2.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_shed_workers_come_back_after_5_minutes_at_15_gb(tmp_path, monkeypatch):
    available = [6.0]
    g = governor(tmp_path, monkeypatch, available)
    workers = tmp_path / "workers"
    e2 = register_shedding(tmp_path, "e2", workers, 3, comply=True)
    try:
        g.tick()
        assert workers.read_text().strip() == "2"
        available[0] = 20.0
        now = time.time()
        g.history.clear()
        g.history.extend((now - 300 + i, 20.0) for i in range(300))
        g.tick()
        assert workers.read_text().strip() == "2"  # the shed was less than 5 minutes ago
        job = json.loads((tmp_path / "e2.job").read_text())
        job["last_shed"] = now - 301
        (tmp_path / "e2.job").write_text(json.dumps(job))
        g.history.append((now - 100, 14.0))  # a reading below 15 GB within the 5 minutes
        g.tick()
        assert workers.read_text().strip() == "2"
        g.history.clear()
        g.history.extend((now - 300 + i, 20.0) for i in range(301))
        g.tick()
        assert workers.read_text().strip() == "3"
        assert "workers back from 2 to 3" in (tmp_path / "events.log").read_text()
        assert "restore_to" not in json.loads((tmp_path / "e2.job").read_text())
    finally:
        e2.kill()
        for p in children(e2.pid):
            subprocess.run(["kill", "-9", str(p)])


# --------------------------------------------------------------------------- a filling job (the author's instruction of 2026-11-16)

FILL = {"max_workers": 12, "per_worker_gb": 3.0, "parent_gb": 2.0}


def fill_spec(folder: Path, workers: Path, **extra) -> None:
    (folder / "e4f.spec").write_text(json.dumps({"base": "e4f", "kind": "e4", "rank": 5, "order": 9, "item": "E4", "script": "s.py", "args": ["--tier", "3"],
                                                 "nice": 19, "oom_adj": 1000, "estimate_gb": 0.0, "shed": True, "fill": FILL,
                                                 "workers": {"file": str(workers), "initial": 1, "target": 0, "per_worker_gb": 3.0}, **extra}))


def window(g: mg.Governor, gb: float) -> None:
    g.history.clear()
    g.history.extend((time.time() - mg.WINDOW_S + 5 + i, gb) for i in range(mg.WINDOW_S - 5))


def test_the_room_of_a_filling_job_is_the_least_of_the_cores_the_budget_and_the_15_gb_floor(monkeypatch):
    monkeypatch.setattr(mg, "CORES", 16)
    room = mg.Governor.fill_room
    assert room(FILL, 0, 50.0, 0.0, 0.0, 10.0, parent=True) == 6  # cores: 16 - 10
    assert room(FILL, 0, 50.0, 30.0, 0.0, 0.0, parent=True) == 2  # budget: (40 - 30 - 2) // 3
    assert room(FILL, 0, 25.0, 0.0, 0.0, 0.0, parent=True) == 2  # floor: (25 - 15 - 2) // 3
    assert room(FILL, 0, 50.0, 20.0, 6.0, 0.0, parent=True) == 4  # budget less the reserve: (40 - 20 - 6 - 2) // 3
    assert room(FILL, 4, 50.0, 0.0, 0.0, 15.5, parent=False) == 4  # no free core: no worker more
    assert room(FILL, 0, 80.0, 0.0, 0.0, 0.0, parent=True) == 12  # at most max_workers


def test_a_filling_job_starts_on_the_workers_the_cores_and_memory_allow(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    FakePopen.calls = []
    monkeypatch.setattr(mg.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mg, "CORES", 16)
    monkeypatch.setattr(mg, "load_average", lambda: 10.0)
    workers = tmp_path / "control" / "workers"
    fill_spec(tmp_path, workers)
    window(g, 50.0)
    g.tick()
    assert len(FakePopen.calls) == 1 and "--workers" not in FakePopen.calls[0]
    assert workers.read_text().strip() == "6"
    assert "on 6 workers" in (tmp_path / "events.log").read_text()
    label = json.loads((tmp_path / "e4f.state").read_text())["runs"][-1]["label"]
    assert time.time() - g.last_fill[label] < 5  # the start counts as an added worker


def test_a_filling_job_gains_no_worker_within_two_minutes_of_its_start(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    monkeypatch.setattr(mg, "CORES", 16)
    monkeypatch.setattr(mg, "load_average", lambda: 0.0)
    workers = tmp_path / "workers"
    fill_spec(tmp_path, workers)
    g.save_state("e4f", {"status": "running", "runs": [], "stops": []})
    e4 = register_shedding(tmp_path, "e4f-run", workers, 1, comply=True, base="e4f", kind="e4", rank=5, target_workers=0, per_worker_gb=3.0)
    try:
        window(g, 50.0)
        g.last_fill["e4f-run"] = time.time()  # as start() records it
        g.tick()
        assert workers.read_text().strip() == "1"
        g.last_fill["e4f-run"] -= mg.FILL_GAP_S
        g.tick()
        assert workers.read_text().strip() == "2"
    finally:
        e4.kill()
        for p in children(e4.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_a_filling_job_waits_while_a_job_before_it_waits_and_keeps_room_for_it(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    FakePopen.calls = []
    monkeypatch.setattr(mg.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(mg, "CORES", 16)
    monkeypatch.setattr(mg, "load_average", lambda: 0.0)
    workers = tmp_path / "control" / "workers"
    fill_spec(tmp_path, workers)
    common = {"item": "X", "script": "s.py", "nice": 0, "args": ["b"], "oom_adj": 800}
    (tmp_path / "e5.spec").write_text(json.dumps({**common, "base": "e5", "kind": "e5", "rank": 1, "order": 2, "estimate_gb": 9.0}))
    g.save_state("e5", {"status": "pending", "runs": [], "stops": [], "not_before": time.time() + 600})  # backing off
    window(g, 50.0)
    g.tick()
    assert not FakePopen.calls  # E5 backs off, and the jobs after it wait
    g.save_state("e5", {"status": "pending", "runs": [], "stops": []})
    assert g.reserve(g.specs()) == 9.0
    assert g.fill_room(FILL, 0, 50.0, 0.0, g.reserve(g.specs()), 0.0, parent=True) == 8  # floor: (50 - 15 - 9 - 2) // 3
    g.tick()
    assert len(FakePopen.calls) == 1 and FakePopen.calls[0][-1] == "b"  # E5 first


def test_a_filling_job_gains_one_worker_at_a_time_every_two_minutes_and_not_soon_after_a_shed(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    monkeypatch.setattr(mg, "CORES", 16)
    monkeypatch.setattr(mg, "load_average", lambda: 2.0)
    workers = tmp_path / "workers"
    fill_spec(tmp_path, workers)
    g.save_state("e4f", {"status": "running", "runs": [], "stops": []})
    e4 = register_shedding(tmp_path, "e4f-run", workers, 2, comply=True, base="e4f", kind="e4", rank=5, target_workers=0, per_worker_gb=3.0)
    try:
        window(g, 50.0)
        g.tick()
        assert workers.read_text().strip() == "3"
        assert "one worker added, 2 to 3" in (tmp_path / "events.log").read_text()
        g.tick()
        assert workers.read_text().strip() == "3"  # within two minutes
        g.last_fill["e4f-run"] -= mg.FILL_GAP_S
        job = json.loads((tmp_path / "e4f-run.job").read_text())
        job["last_shed"] = time.time() - 60
        (tmp_path / "e4f-run.job").write_text(json.dumps(job))
        g.tick()
        assert workers.read_text().strip() == "3"  # a shed a minute ago
        job["last_shed"] = time.time() - mg.RESTORE_S - 1
        (tmp_path / "e4f-run.job").write_text(json.dumps(job))
        g.tick()
        assert workers.read_text().strip() == "4"
    finally:
        e4.kill()
        for p in children(e4.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_a_filling_job_is_shed_first_has_no_30_gb_rule_and_keeps_no_restore_to(tmp_path, monkeypatch):
    available = [20.0]
    g = governor(tmp_path, monkeypatch, available)
    workers = tmp_path / "workers"
    fill_spec(tmp_path, workers)
    g.save_state("e4f", {"status": "running", "runs": [], "stops": []})
    e4 = register_shedding(tmp_path, "e4f-run", workers, 2, comply=True, base="e4f", kind="e4", rank=5, target_workers=0, per_worker_gb=3.0)
    smoke = register(tmp_path, "smoke", "smoke", 4)
    try:
        g.tick()
        assert e4.poll() is None and workers.read_text().strip() == "2"  # 20 GB: E4's 30 GB rule does not apply
        available[0] = 6.0
        g.tick()
        assert workers.read_text().strip() == "1" and smoke.poll() is None  # shed before the smoke run
        assert "restore_to" not in json.loads((tmp_path / "e4f-run.job").read_text())
    finally:
        for p in (e4, smoke):
            p.kill()
        for p in children(e4.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_above_the_budget_a_filling_job_loses_a_worker_instead_of_a_stop(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    monkeypatch.setattr(mg, "BUDGET_GB", -1.0)  # mine (0 GB here) is above it
    workers = tmp_path / "workers"
    fill_spec(tmp_path, workers)
    g.save_state("e4f", {"status": "running", "runs": [], "stops": []})
    e4 = register_shedding(tmp_path, "e4f-run", workers, 2, comply=True, base="e4f", kind="e4", rank=5, target_workers=0, per_worker_gb=3.0)
    try:
        g.tick()
        assert workers.read_text().strip() == "1" and e4.poll() is None
        assert "above the -1 GB budget" in (tmp_path / "events.log").read_text()
    finally:
        e4.kill()
        for p in children(e4.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_a_job_held_back_by_the_budget_makes_the_filling_job_give_back_a_worker(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    workers = tmp_path / "workers"
    fill_spec(tmp_path, workers)
    g.save_state("e4f", {"status": "running", "runs": [], "stops": []})
    common = {"item": "X", "script": "s.py", "nice": 0, "args": ["b"], "oom_adj": 800}
    (tmp_path / "e5.spec").write_text(json.dumps({**common, "base": "e5", "kind": "e5", "rank": 1, "order": 2, "estimate_gb": 41.0}))
    e4 = register_shedding(tmp_path, "e4f-run", workers, 2, comply=True, base="e4f", kind="e4", rank=5, target_workers=0, per_worker_gb=3.0)
    FakePopen.calls = []
    monkeypatch.setattr(mg.subprocess, "Popen", FakePopen)
    try:
        window(g, 50.0)
        g.tick()
        assert not FakePopen.calls and workers.read_text().strip() == "1"
        assert "for e5 to start within the 40 GB budget" in (tmp_path / "events.log").read_text()
    finally:
        e4.kill()
        for p in children(e4.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_the_eight_hours_run_from_no_earlier_than_progress_from(tmp_path, monkeypatch):
    import os
    g = governor(tmp_path, monkeypatch, [50.0])
    fits = tmp_path / "fits"
    fits.mkdir()
    old = time.time() - 30 * 3600
    (fits / "a.npz").write_bytes(b"x")
    os.utime(fits / "a.npz", (old, old))
    spec = [{"base": "e4f", "order": 9, "progress": [str(fits)], "progress_from": time.time() - 3600}]
    g.save_state("e4f", {"status": "running", "runs": [], "stops": [], "first_seen": time.time() - 60})
    g.check_progress(spec)
    assert not (tmp_path / "wake.log").exists()  # released an hour ago
    spec[0]["progress_from"] = time.time() - 8 * 3600 - 60
    g.check_progress(spec)
    assert "saved no new fit or task for 8.0 h" in (tmp_path / "wake.log").read_text()


# --------------------------------------------------------------------------- physical cores (E2's and E5's jobs keep their cores)

def test_the_cores_are_the_physical_cores(tmp_path):
    for cpu, (package, core) in enumerate([(0, 0), (0, 1), (0, 0), (0, 1), (1, 0)]):
        topology = tmp_path / ("cpu%d" % cpu) / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("%d\n" % package)
        (topology / "core_id").write_text("%d\n" % core)
    (tmp_path / "cpufreq").mkdir()  # not a CPU
    assert mg.physical_cores(tmp_path) == 3
    assert mg.physical_cores(tmp_path / "missing") == (mg.os.cpu_count() or 1)


def test_the_room_by_cores_is_the_physical_cores_less_the_load_of_the_other_jobs(monkeypatch):
    monkeypatch.setattr(mg, "CORES", 8)
    room = mg.Governor.fill_room
    assert room(FILL, 8, 80.0, 0.0, 0.0, 16.1, parent=False) == 0  # eight workers of E5 beside it: no core free
    assert room(FILL, 3, 80.0, 0.0, 0.0, 9.3, parent=False) == 1  # 8 - (9.3 - 3)
    assert room(FILL, 0, 80.0, 0.0, 0.0, 6.3, parent=True) == 1  # a start: 8 - 6.3
    assert room(FILL, 2, 80.0, 0.0, 0.0, 1.0, parent=False) == 8  # the load average still behind the job's own workers


def test_a_filling_job_gives_back_a_worker_every_three_minutes_while_the_load_exceeds_the_physical_cores(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    monkeypatch.setattr(mg, "CORES", 8)
    load = [10.2]
    monkeypatch.setattr(mg, "load_average", lambda: load[0])
    workers = tmp_path / "workers"
    fill_spec(tmp_path, workers)
    g.save_state("e4f", {"status": "running", "runs": [], "stops": []})
    e4 = register_shedding(tmp_path, "e4f-run", workers, 2, comply=True, base="e4f", kind="e4", rank=5, target_workers=0, per_worker_gb=3.0)
    try:
        g.tick()  # no ten minutes of readings yet: the cores are judged all the same
        assert workers.read_text().strip() == "1"
        assert "shed e4f-run: 2 to 1 workers (load average 10.2 on 8 physical cores" in (tmp_path / "events.log").read_text()
        assert len(settle_children(e4.pid, 1)) == 1  # the newest fit's process
        g.tick()
        assert workers.read_text().strip() == "1"  # within three minutes
        job = json.loads((tmp_path / "e4f-run.job").read_text())
        job["last_shed"] = time.time() - mg.CPU_SHED_GAP_S - 1
        (tmp_path / "e4f-run.job").write_text(json.dumps(job))
        load[0] = 9.2
        g.tick()
        assert workers.read_text().strip() == "0" and e4.poll() is None  # no worker left, and the job is not stopped
        job = json.loads((tmp_path / "e4f-run.job").read_text())
        job["last_shed"] = time.time() - mg.RESTORE_S - 1
        (tmp_path / "e4f-run.job").write_text(json.dumps(job))
        load[0] = 8.3  # E5's eight workers alone
        window(g, 50.0)
        g.tick()
        assert workers.read_text().strip() == "0" and e4.poll() is None  # 8 - 8.3: no core free
        load[0] = 6.3  # two of E5's workers ended
        g.tick()
        assert workers.read_text().strip() == "1"
        assert "one worker added, 0 to 1 (load 6.3 of 8 physical cores" in (tmp_path / "events.log").read_text()
    finally:
        e4.kill()
        for p in children(e4.pid):
            subprocess.run(["kill", "-9", str(p)])


def test_a_filling_job_keeps_its_workers_while_the_load_is_within_half_a_thread_of_the_physical_cores(tmp_path, monkeypatch):
    g = governor(tmp_path, monkeypatch, [50.0])
    monkeypatch.setattr(mg, "CORES", 8)
    monkeypatch.setattr(mg, "load_average", lambda: 8.4)
    workers = tmp_path / "workers"
    fill_spec(tmp_path, workers)
    g.save_state("e4f", {"status": "running", "runs": [], "stops": []})
    e4 = register_shedding(tmp_path, "e4f-run", workers, 2, comply=True, base="e4f", kind="e4", rank=5, target_workers=0, per_worker_gb=3.0)
    try:
        window(g, 50.0)
        g.tick()
        assert workers.read_text().strip() == "2"  # neither shed nor widened: 8 - (8.4 - 2) leaves one core, not two more
    finally:
        e4.kill()
        for p in children(e4.pid):
            subprocess.run(["kill", "-9", str(p)])
