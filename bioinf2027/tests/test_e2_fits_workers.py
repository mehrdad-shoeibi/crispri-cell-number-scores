"""The worker count of E2's fit script (bioinf2027/scripts/make_e2_fits.py): read from a file, widened and narrowed
between fits by adding and closing pools; no data-root input is read and no fit is made."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e2_fits as mf  # noqa: E402


def test_the_worker_count_comes_from_the_file_when_it_holds_an_integer(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "WORKERS", 3)
    monkeypatch.setattr(mf, "WORKERS_FILE", None)
    assert mf.worker_limit() == 3
    path = tmp_path / "workers"
    monkeypatch.setattr(mf, "WORKERS_FILE", path)
    assert mf.worker_limit() == 3  # no file yet
    path.write_text("6\n")
    assert mf.worker_limit() == 6
    path.write_text("six")
    assert mf.worker_limit() == 3  # not an integer: the --workers value
    path.write_text("0")
    assert mf.worker_limit() == 1


def test_pools_are_added_when_the_limit_rises_and_closed_when_idle_and_not_needed():
    pools = mf.resize([], 3, True)
    try:
        assert [p["size"] for p in pools] == [3]
        pools = mf.resize(pools, 6, True)
        assert [p["size"] for p in pools] == [3, 3]
        assert pools[1]["pool"].submit(abs, -4).result() == 4  # the added pool's workers run tasks
        pools = mf.resize(pools, 6, False)  # no fit left to submit: nothing is added
        assert [p["size"] for p in pools] == [3, 3]
        pools[1]["busy"] = 1
        pools = mf.resize(pools, 3, True)  # the newest pool is busy: kept until its fit ends
        assert [p["size"] for p in pools] == [3, 3]
        pools[1]["busy"] = 0
        pools = mf.resize(pools, 3, True)
        assert [p["size"] for p in pools] == [3]
        pools = mf.resize(pools, 1, True)  # the first pool is never closed
        assert [p["size"] for p in pools] == [3]
    finally:
        for p in pools:
            p["pool"].shutdown(wait=True)


def _square(x):
    import time as _time
    _time.sleep(0.05)
    if x == 7:
        raise ValueError("seven")
    return {"task": x, "value": x * x}


def test_every_task_runs_once_and_the_worker_count_can_rise_during_the_run(tmp_path, monkeypatch, capsys):
    path = tmp_path / "workers"
    path.write_text("2\n")
    monkeypatch.setattr(mf, "WORKERS", 2)
    monkeypatch.setattr(mf, "WORKERS_FILE", path)
    original = mf.worker_limit
    calls = []

    def limit():  # the file is changed from 2 to 4 after the first readings, as the memory rules do it
        calls.append(1)
        if len(calls) == 4:
            path.write_text("4\n")
        return original()
    monkeypatch.setattr(mf, "worker_limit", limit)
    done, failed = mf.run_tasks(list(range(30)), _square, lambda t: "task %d" % t, 0.0)
    assert sorted(d["task"] for d in done) == [x for x in range(30) if x != 7]
    assert all(d["value"] == d["task"] ** 2 for d in done)
    assert failed == [{"fit": "task 7", "error": "ValueError('seven')"}]
    out = capsys.readouterr().out
    assert "workers: 2 (a pool of 2 added)" in out and "workers: 4 (a pool of 2 added)" in out
