"""The repeated expression pass against the accumulator that made the recorded one.

A small synthetic screen is run through the original accumulator, chunk by
chunk, to produce a mid-pass state and a final delta matrix. The repeated pass
must reproduce both exactly, and each of its three checks must fail when the
thing it guards is broken.

The screen contains the cases that matter for the real file: control cells
that also carry a listed construct, cells of an unlisted construct inside the
pass's row span, an empty cell, and a construct whose cells all come after the
checkpoint.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bioinf2027 import config, singlecell  # noqa: E402

import make_d1b_rpe1_control_mean as rpe1  # noqa: E402

ORIGINAL = config.repo_root() / "earlier_revision" / "pipeline" / "x_cell_accumulator.py"
CHUNK = 50
SNAPSHOT = 3


def _original_accumulator():
    if not ORIGINAL.exists():
        pytest.skip("the original accumulator is not part of this checkout")
    spec = importlib.util.spec_from_file_location("x_cell_accumulator_original", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    # The original lives in the historical tree, which must not be written to,
    # so no bytecode file is left next to it.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture()
def screen(monkeypatch):
    """A synthetic screen, the original accumulator's outputs for it, and patched constants."""
    rng = np.random.default_rng(20270301)
    n_cells, n_genes = 420, 9
    constructs = np.array(sorted(f"c{index:02d}_GENE{index}" for index in range(6)))
    early, late = constructs[:-1], constructs[-1]
    labels = np.array(list(early) + ["non-targeting_g1", "unlisted_construct"])
    transcript = rng.choice(labels, size=n_cells, p=[0.13] * 5 + [0.20, 0.15]).astype("<U40")
    gene = np.where(np.char.startswith(transcript, "non-targeting"), "non-targeting", "some-gene")
    # Three control cells that also carry a listed construct: they count as controls.
    for row in (7, 61, 140):
        gene[row], transcript[row] = "non-targeting", early[row % len(early)]
    # One construct whose cells all come after the checkpoint.
    transcript[-25:] = late
    gene[-25:] = "some-gene"
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    counts[5] = 0.0  # an empty cell exercises the zero-total guard

    stored_control = np.flatnonzero(gene == "non-targeting")
    in_pass = (gene == "non-targeting") | np.isin(transcript, constructs)
    rows = np.flatnonzero(in_pass)
    n_chunks = -(-rows.size // CHUNK)
    assert n_chunks > SNAPSHOT + 1

    original = _original_accumulator()
    accumulator = original.XCellAccumulator(n_genes, checkpoint_dir=None)
    mid_state = None
    for k, start in enumerate(range(0, rows.size, CHUNK)):
        chunk_rows = rows[start : start + CHUNK]
        is_control = gene[chunk_rows] == "non-targeting"
        ids = np.where(is_control, "__NT_POOL__", transcript[chunk_rows]).astype(object)
        accumulator.update(counts[chunk_rows], ids, is_control, chunk_id=k)
        if k + 1 == SNAPSHOT:
            mid_state = copy.deepcopy(accumulator.get_state())
    assert late not in mid_state["count"]
    final = accumulator.finalize()
    persisted = np.stack([final["x_cell"][name] for name in constructs]).astype(np.float64)

    values = {
        "control_cells": int(stored_control.size),
        "pass_cells": int(rows.size),
        "pass_chunks": int(n_chunks),
        "constructs": int(constructs.size),
        "snapshot_chunks": SNAPSHOT,
        "snapshot_cells": int(mid_state["total_cells"]),
        "snapshot_control_cells": int(mid_state["nt_count"]),
    }
    monkeypatch.setattr(
        rpe1, "RECORDED", {k: {"value": v, "source": "synthetic"} for k, v in values.items()}
    )
    monkeypatch.setattr(rpe1, "CHUNK", CHUNK)
    return {
        "counts": counts,
        "gene": gene,
        "transcript": transcript,
        "constructs": constructs,
        "stored_control": stored_control,
        "mid_state": mid_state,
        "persisted": persisted,
        "original_control_mean": final["mu_nt"],
    }


def _repeat(screen, stored=None):
    rows, codes, check = rpe1.pass_definition(
        screen["gene"],
        screen["transcript"],
        screen["constructs"],
        screen["stored_control"] if stored is None else stored,
    )
    state, snapshot = singlecell.accumulate_group_sums(
        screen["counts"],
        rows,
        codes,
        n_groups=screen["constructs"].size + 1,
        chunk=CHUNK,
        snapshot_after=SNAPSHOT,
        read_dtype=rpe1.READ_DTYPE,
    )
    return check, state, snapshot


def _prefix(screen, snapshot, state=None):
    return rpe1.checkpoint_prefix_check(
        snapshot, screen["mid_state"] if state is None else state, screen["constructs"]
    )


def test_repeated_pass_reproduces_the_original_accumulator_exactly(screen):
    stored, state, snapshot = _repeat(screen)
    assert stored["passed"]
    assert stored["control_cells_also_carrying_a_listed_construct"] == 3

    prefix = _prefix(screen, snapshot)
    assert prefix["passed"]
    assert prefix["max_abs_difference_of_sums"] == 0.0

    control_mean, identity = rpe1.identity_check(
        state, screen["constructs"], screen["persisted"], screen["constructs"]
    )
    assert identity["passed"]
    assert identity["identity_max_abs_diff"] == 0.0
    assert np.array_equal(control_mean, screen["original_control_mean"])


def test_stored_index_check_fails_when_the_stored_array_misses_a_control_cell(screen):
    _, _, check = rpe1.pass_definition(
        screen["gene"], screen["transcript"], screen["constructs"], screen["stored_control"][:-1]
    )
    assert not check["stored_equals_recorded_rule"]
    assert not check["passed"]


def test_stored_index_check_fails_on_a_row_outside_the_file(screen):
    bad = np.append(screen["stored_control"], len(screen["gene"]) + 5)
    rows, codes, check = rpe1.pass_definition(
        screen["gene"], screen["transcript"], screen["constructs"], bad
    )
    assert not check["passed"] and rows.size == 0 and codes.size == 0


def test_stored_index_check_fails_on_a_different_pass_size(screen, monkeypatch):
    changed = dict(rpe1.RECORDED)
    changed["pass_cells"] = {"value": changed["pass_cells"]["value"] + 1, "source": "synthetic"}
    monkeypatch.setattr(rpe1, "RECORDED", changed)
    check, _, _ = _repeat(screen)
    assert not check["passed"]


def test_stored_index_check_fails_when_the_endpoint_rule_finds_other_controls(screen):
    gene = screen["gene"].copy()
    gene[3] = "Non-Targeting control"  # the endpoint rule finds it, the recorded rule does not
    _, _, check = rpe1.pass_definition(
        gene, screen["transcript"], screen["constructs"], screen["stored_control"]
    )
    assert check["stored_equals_recorded_rule"]
    assert not check["stored_equals_endpoint_rule"]
    assert not check["passed"]


@pytest.mark.parametrize(
    "damage",
    [
        "shift_control_sum",
        "shift_one_construct_sum",
        "nan_in_one_construct_sum",
        "change_one_construct_count",
        "drop_one_construct",
        "add_one_construct",
        "change_control_count",
        "change_total_cells",
    ],
)
def test_checkpoint_prefix_check_fails_on_every_kind_of_damage(screen, damage):
    _, _, snapshot = _repeat(screen)
    state = copy.deepcopy(screen["mid_state"])
    name = sorted(state["count"])[0]
    if damage == "shift_control_sum":
        state["nt_sum"] = state["nt_sum"] + 2e-9
    elif damage == "shift_one_construct_sum":
        state["sum"][name] = state["sum"][name] + 2e-9
    elif damage == "nan_in_one_construct_sum":
        state["sum"][name] = state["sum"][name].copy()
        state["sum"][name][0] = np.nan
    elif damage == "change_one_construct_count":
        state["count"][name] += 1
    elif damage == "drop_one_construct":
        del state["count"][name]
        del state["sum"][name]
    elif damage == "add_one_construct":
        late = screen["constructs"][-1]
        state["count"][late] = 1
        state["sum"][late] = np.zeros_like(state["nt_sum"])
    elif damage == "change_control_count":
        state["nt_count"] -= 1
    elif damage == "change_total_cells":
        state["total_cells"] += 1
    assert not _prefix(screen, snapshot, state)["passed"]


def test_checkpoint_prefix_check_fails_without_a_snapshot(screen):
    assert not rpe1.checkpoint_prefix_check(None, screen["mid_state"], screen["constructs"])["passed"]


def test_identity_check_fails_on_a_shifted_matrix_or_reordered_constructs(screen):
    _, state, _ = _repeat(screen)
    shifted = screen["persisted"] + 2e-9
    assert not rpe1.identity_check(state, screen["constructs"], shifted, screen["constructs"])[1][
        "passed"
    ]
    reordered = screen["constructs"][::-1].copy()
    assert not rpe1.identity_check(state, reordered, screen["persisted"], screen["constructs"])[1][
        "passed"
    ]


def test_identity_check_fails_on_nan_or_a_wrong_shape_without_raising(screen):
    _, state, _ = _repeat(screen)
    with_nan = screen["persisted"].copy()
    with_nan[2, 4] = np.nan
    _, check = rpe1.identity_check(state, screen["constructs"], with_nan, screen["constructs"])
    assert not check["passed"]
    _, check = rpe1.identity_check(
        state, screen["constructs"], screen["persisted"][:, :-1], screen["constructs"]
    )
    assert not check["passed"] and not check["persisted_shape_matches"]


def test_identity_check_fails_on_a_different_control_count(screen, monkeypatch):
    _, state, _ = _repeat(screen)
    changed = dict(rpe1.RECORDED)
    changed["control_cells"] = {"value": changed["control_cells"]["value"] + 1, "source": "synthetic"}
    monkeypatch.setattr(rpe1, "RECORDED", changed)
    assert not rpe1.identity_check(
        state, screen["constructs"], screen["persisted"], screen["constructs"]
    )[1]["passed"]


def test_strict_json_writes_non_finite_numbers_as_null():
    out = rpe1.strict_json({"a": float("nan"), "b": [np.float64(np.inf), 1.5], "c": np.int64(3)})
    assert out == {"a": None, "b": [None, 1.5], "c": 3}


def test_manifest_comparison_has_three_outcomes():
    assert rpe1.manifest_comparison({"input_arm_list_sha256": "x"}, "x") == "equal"
    assert rpe1.manifest_comparison({"input_arm_list_sha256": "y"}, "x") == "different"
    assert rpe1.manifest_comparison({}, "x") == "key absent"


def test_checkpoint_prefix_tolerance_is_relative_on_large_sums(screen):
    _, _, snapshot = _repeat(screen)
    state = copy.deepcopy(screen["mid_state"])
    name = sorted(state["count"])[0]
    # A difference of 5e-9 on a sum near 1e4 is a rounding difference, inside 1e-9 + 1e-12 * 1e4.
    big = copy.deepcopy(state)
    big["sum"][name] = np.full_like(state["sum"][name], 1e4)
    repeated = copy.deepcopy(snapshot)
    group = list(screen["constructs"]).index(name) + 1
    repeated.sums[group] = np.full_like(repeated.sums[group], 1e4 + 5e-9)
    assert rpe1.checkpoint_prefix_check(repeated, big, screen["constructs"])["passed"]
    repeated.sums[group] = np.full_like(repeated.sums[group], 1e4 + 5e-8)
    assert not rpe1.checkpoint_prefix_check(repeated, big, screen["constructs"])["passed"]
