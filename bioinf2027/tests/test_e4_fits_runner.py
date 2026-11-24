"""E4: the fits run one process each (--slots) give what the pool gives, a failed fit is reported by both runners,
and --smallest-first orders the fits by ascending price, ties in their earlier order (the author's instructions of
2026-11-15 and 2026-11-16, bioinf2027/GATE.md; no value may change)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e4_fits as m  # noqa: E402


class Spec:
    def __init__(self, name: str) -> None:
        self.name = name

    def key(self) -> str:
        return self.name


def toy_worker(description: dict, spec: Spec, index: int) -> dict:
    if spec.name == "fails":
        raise ValueError("no fit")
    return {"spec": {"name": spec.name}, "value": sum(ord(c) for c in spec.name) * (index + 1)}


def tasks(names):
    return [({"name": "context %s" % n}, Spec(n)) for n in names]


def results(monkeypatch, slots: bool, names, workers_file=None):
    monkeypatch.setattr(m, "worker", toy_worker)
    monkeypatch.setattr(m, "SLOTS", slots)
    monkeypatch.setattr(m, "WORKERS", 3)
    monkeypatch.setattr(m, "WORKERS_FILE", workers_file)
    listed = tasks(names)
    index = {d["name"]: i for i, (d, _) in enumerate(listed)}
    return m.run_fits(listed, index, time.time())


def test_slots_give_what_the_pool_gives(monkeypatch, tmp_path):
    names = ["a%d" % i for i in range(9)]
    pool_done, pool_failed, pool_shed = results(monkeypatch, False, names)
    count = tmp_path / "workers"
    count.write_text("2\n")
    slot_done, slot_failed, slot_shed = results(monkeypatch, True, names, count)
    key = lambda r: r["spec"]["name"]  # noqa: E731
    assert sorted(slot_done, key=key) == sorted(pool_done, key=key) and len(slot_done) == len(names)
    assert pool_failed == slot_failed == [] and pool_shed is None and slot_shed == 0


def test_a_failed_fit_is_reported_by_both_runners(monkeypatch):
    for slots in (False, True):
        done, failed, _ = results(monkeypatch, slots, ["a", "fails", "b"])
        assert len(done) == 2 and [f["fit"] for f in failed] == ["fails"] and "no fit" in failed[0]["error"]


def test_smallest_first_orders_by_price_and_keeps_ties_in_order():
    listed = tasks(["p", "q", "r", "s", "t"])
    ordered = m.smallest_first(listed, [3.0, 1.0, 2.0, 1.0, 0.5])
    assert [s.name for _, s in ordered] == ["t", "q", "s", "r", "p"]
    assert sorted(s.name for _, s in ordered) == sorted(s.name for _, s in listed)
