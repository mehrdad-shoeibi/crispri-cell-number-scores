"""E2, scGPT: the input of given cells as make_e2_embed_scgpt.cell_input builds it, in whichever environment runs
this script (gate G3, item 5 (c); the results review, round 2, non-blocking point 3: the embedding ran under numpy
2.2.6, the rebuild under 2.4.4).

Input (an .npz file written by make_e2_scgpt_target_input.py): each cell's expressed vocabulary-gene values
(concatenated, with offsets), its generator seed and scGPT's pad value. Output (an .npz file): per cell, the
positions kept (0-based, into the cell's expressed vocabulary genes; at most 1,199) and the binned values given to
the model, concatenated with offsets, and the numpy and Python versions. Reads and writes only the two files named.

Usage: e2_scgpt_draws.py INPUT.npz OUTPUT.npz
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_e2_embed_scgpt as emb  # noqa: E402


def draws(values: np.ndarray, offsets: np.ndarray, seeds: np.ndarray, pad_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept, binned, bounds = [], [], [0]
    for k in range(seeds.size):
        cell = values[offsets[k]: offsets[k + 1]]
        genes, expr = emb.cell_input(cell, np.arange(1, cell.size + 1, dtype=np.int64), np.random.default_rng(int(seeds[k])), 0, pad_value)
        kept.append(genes[1:] - 1)
        binned.append(expr[1:])
        bounds.append(bounds[-1] + genes.size - 1)
    return np.concatenate(kept).astype(np.int64), np.concatenate(binned).astype(np.float32), np.asarray(bounds, dtype=np.int64)


def main(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=False) as store:
        values, offsets, seeds, pad_value = store["values"], store["offsets"], store["seeds"], float(store["pad_value"])
    kept, binned, bounds = draws(values, offsets, seeds, pad_value)
    np.savez(target, kept=kept, binned=binned, bounds=bounds, numpy=np.array(np.__version__), python=np.array(platform.python_version()))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: e2_scgpt_draws.py INPUT.npz OUTPUT.npz")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
