"""Tests of the E1b input helpers on synthetic arrays."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import e1inputs  # noqa: E402


def _means():
    per_batch = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    return e1inputs.ControlMeans(per_batch=per_batch, batch_order=np.array(["b1", "b2"]), selected=np.array([1, 3]), counts={"b1": 1, "b2": 3})


def test_pooled_reference_weights_batches_by_their_control_cells():
    means = _means()
    assert means.pooled().tolist() == pytest.approx([2.5, 3.5, 4.5, 5.5])
    shift = means.shift()
    assert shift["b1"].tolist() == pytest.approx([2.0 - 3.5, 4.0 - 5.5])
    assert shift["b2"].tolist() == pytest.approx([4.0 - 3.5, 6.0 - 5.5])
    embedding = means.pooled_embedding({"b1": np.array([0.0, 4.0], np.float32), "b2": np.array([4.0, 0.0], np.float32)})
    assert embedding.tolist() == pytest.approx([3.0, 1.0])


def test_reference_shift_turns_a_batch_delta_into_a_pooled_delta():
    # x = mean_pert - mean_b; x + mean_b - mean_pooled = mean_pert - mean_pooled
    means = _means()
    mean_pert = np.array([10.0, 20.0])
    for batch, row in (("b1", 0), ("b2", 1)):
        x = mean_pert - means.per_batch[row, means.selected]
        assert (x + means.shift()[batch]).tolist() == pytest.approx((mean_pert - means.pooled()[means.selected]).tolist())


def test_gene_level_means_are_weighted_by_cells():
    modelled = pd.DataFrame({"split": ["train", "train", "train", "val"], "target_gene": ["A", "A", "B", "A"]})
    values = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 5.0], [7.0, 7.0]])
    weights = np.array([1, 3, 2, 4])
    frame = pd.DataFrame({"split": ["train", "val", "train"], "target_gene": ["A", "A", "B"]})
    out = e1inputs.gene_level(frame, modelled, values, weights)
    assert out[0].tolist() == pytest.approx([(1 * 1 + 3 * 3) / 4, (0 + 3 * 2) / 4])
    assert out[1].tolist() == pytest.approx([7.0, 7.0])
    assert out[2].tolist() == pytest.approx([5.0, 5.0])
