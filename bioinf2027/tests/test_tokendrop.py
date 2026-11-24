"""Tests of the drop check and the truncation of the D2 token-drop arm."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import tokendrop  # noqa: E402


def test_truncate_adds_the_special_tokens_and_keeps_4094_genes():
    ranking = np.arange(10, 5010)
    model_input = tokendrop.truncate(ranking, cls=2, eos=3)
    assert model_input.size == tokendrop.MODEL_INPUT_SIZE
    assert model_input[0] == 2 and model_input[-1] == 3
    assert np.array_equal(model_input[1:-1], ranking[: tokendrop.GENES_KEPT])


def test_drop_inside_the_input_moves_one_gene_up():
    undropped = np.arange(100, 5100)
    token = int(undropped[17])
    dropped = undropped[undropped != token]
    check = tokendrop.check_drop(undropped, dropped, token)
    assert check["passed"] and check["target_in_input"] and check["appended"] == 1


def test_drop_outside_the_input_changes_nothing_in_it():
    undropped = np.arange(100, 5100)
    token = int(undropped[4500])
    dropped = undropped[undropped != token]
    check = tokendrop.check_drop(undropped, dropped, token)
    assert check["passed"] and not check["target_in_input"] and check["appended"] == 0


def test_any_other_change_fails_the_check():
    undropped = np.arange(100, 5100)
    token = int(undropped[17])
    dropped = undropped[undropped != token].copy()
    dropped[[3, 4]] = dropped[[4, 3]]  # two tied genes swapped
    assert not tokendrop.check_drop(undropped, dropped, token)["passed"]
    assert not tokendrop.check_drop(undropped, undropped, token)["passed"]  # the target was not removed


def test_a_cell_without_the_target_or_without_a_token():
    undropped = np.arange(100, 300)
    assert tokendrop.check_drop(undropped, undropped, None)["passed"]
    assert tokendrop.check_drop(undropped, undropped, 999)["passed"]
