"""The safe-pickle helper refuses code and loads plain data (bioinf2027/scripts/safe_pickle.py)."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import safe_pickle  # noqa: E402


class _Payload:
    def __reduce__(self):
        return (os.system, ("echo should-not-run",))


def test_plain_data_needs_no_global_and_loads(tmp_path):
    path = tmp_path / "plain.pkl"
    path.write_bytes(pickle.dumps(["TP53", "MYC", {"A": [1.0, 2.0]}], protocol=4))
    assert not {g for g in safe_pickle.globals_in(path) if g[0] != "<opcode>"}
    assert safe_pickle.load(path) == ["TP53", "MYC", {"A": [1.0, 2.0]}]


def test_code_is_listed_and_refused_without_running(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(_Payload(), protocol=4))
    found = safe_pickle.globals_in(path)
    assert any(name == "system" for _, name in found)
    with pytest.raises(pickle.UnpicklingError):
        safe_pickle.load(path)


def test_numpy_arrays_load_with_the_numpy_allowlist(tmp_path):
    path = tmp_path / "arrays.pkl"
    path.write_bytes(pickle.dumps({"TP53": np.arange(3.0)}, protocol=4))
    with pytest.raises(pickle.UnpicklingError):
        safe_pickle.load(path)
    out = safe_pickle.load(path, safe_pickle.NUMPY_ARRAYS)
    assert np.array_equal(out["TP53"], np.arange(3.0))
