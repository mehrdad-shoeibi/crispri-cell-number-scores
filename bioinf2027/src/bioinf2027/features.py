"""Loaders for the frozen feature namespaces and the external screen artifacts.

Nothing here recomputes a label or an embedding. Every array is read from a
persisted artifact resolved through :mod:`bioinf2027.config`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

__all__ = ["hvg_symbols", "ScreenData", "load_screen", "read_endpoint", "CLIP_LIMIT"]

# The feature convention of the source pipeline: unavailable coordinates are
# zero filled and the delta is clipped to this absolute limit.
CLIP_LIMIT = 10.0


def hvg_symbols() -> np.ndarray:
    """The 2,000 selected highly variable genes, as symbols, in feature order."""
    paths = config.vcc_cache()
    var_names = np.load(paths["var_gene_names"], allow_pickle=True).astype(str)
    selected = np.load(paths["selected_genes"])
    symbols = var_names[selected]
    if len(set(symbols.tolist())) != len(symbols):
        raise RuntimeError("highly variable gene symbols are not unique")
    return symbols


@dataclass(frozen=True)
class ScreenData:
    """Per-construct artifacts for one external screen."""

    name: str
    endpoint: pd.DataFrame
    constructs: np.ndarray
    x_native: np.ndarray
    x_mapped: np.ndarray

    def __post_init__(self) -> None:
        if len(self.endpoint) != len(self.constructs):
            raise RuntimeError("endpoint rows and construct ids disagree")
        if self.x_native.shape[0] != len(self.constructs):
            raise RuntimeError("native matrix rows and construct ids disagree")
        if self.x_mapped.shape[0] != len(self.constructs):
            raise RuntimeError("mapped matrix rows and construct ids disagree")
        if self.x_mapped.shape[1] != 2000:
            raise RuntimeError("mapped matrix must have 2,000 columns")


def read_endpoint(path: Path) -> pd.DataFrame:
    """An endpoint table sorted by construct.

    Numbers are parsed with round_trip precision, which returns the exact double
    each decimal in the file stands for; the default parser can be one unit in
    the last place away.
    """
    table = pd.read_csv(path, float_precision="round_trip")
    return table.sort_values("construct").reset_index(drop=True)


def _load_mapped(path: Path, n_rows: int) -> tuple[np.ndarray, np.ndarray | None]:
    """The mapped matrix and its construct identifiers, or None where none are stored."""
    if path.suffix == ".npz":
        store = np.load(path, allow_pickle=True)
        return np.asarray(store["x"]), np.asarray(store["ids"]).astype(str)
    matrix = np.load(path)
    if matrix.shape[0] != n_rows:
        raise RuntimeError("mapped matrix row count does not match the endpoint")
    return matrix, None


def load_screen(name: str) -> ScreenData:
    """Load one screen's endpoint and cell-level delta matrices.

    The mapped matrix for one screen is stored without identifiers. Its row
    order is verified against the native matrix by checking that the target
    coordinate carries the same value on both sides, which only holds when the
    rows correspond.
    """
    paths = config.screen_paths()[name]
    endpoint = read_endpoint(paths["endpoint"])

    native_store = np.load(paths["x_native"], allow_pickle=True)
    native_ids = np.asarray(native_store["ids"]).astype(str)
    x_native = np.asarray(native_store["x"])
    order = np.argsort(native_ids)
    native_ids, x_native = native_ids[order], x_native[order]

    if not np.array_equal(native_ids, endpoint["construct"].to_numpy().astype(str)):
        raise RuntimeError("native construct ids do not match the endpoint constructs")

    x_mapped, mapped_ids = _load_mapped(paths["x_mapped"], len(endpoint))
    if mapped_ids is not None:
        order = np.argsort(mapped_ids)
        mapped_ids, x_mapped = mapped_ids[order], x_mapped[order]
        if not np.array_equal(mapped_ids, native_ids):
            raise RuntimeError("mapped construct ids do not match the native ids")

    _verify_mapped_alignment(endpoint, x_native, x_mapped)
    return ScreenData(
        name=name,
        endpoint=endpoint,
        constructs=native_ids,
        x_native=x_native,
        x_mapped=np.asarray(x_mapped),
    )


def _verify_mapped_alignment(
    endpoint: pd.DataFrame, x_native: np.ndarray, x_mapped: np.ndarray
) -> None:
    """Confirm the mapped rows correspond to the native rows.

    For every construct whose target gene is one of the selected highly
    variable genes, the mapped vector must carry the clipped native value at
    the target's feature position. Checking this on all such constructs is a
    strong row-order test that needs no stored identifiers.
    """
    symbols = hvg_symbols()
    position = {symbol: index for index, symbol in enumerate(symbols.tolist())}
    checked = 0
    for row, (symbol, column) in enumerate(
        zip(endpoint["target_symbol"].astype(str), endpoint["target_col"].astype(int))
    ):
        index = position.get(symbol)
        if index is None:
            continue
        expected = float(np.clip(x_native[row, column], -CLIP_LIMIT, CLIP_LIMIT))
        actual = float(x_mapped[row, index])
        if not np.isclose(expected, actual, rtol=1e-5, atol=1e-5):
            raise RuntimeError(
                "mapped and native disagree at the target coordinate for row %d" % row
            )
        checked += 1
    if checked == 0:
        raise RuntimeError("row alignment could not be verified: no target gene is a feature")
