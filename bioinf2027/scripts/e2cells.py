"""E2's cells, parts and Stack units (bioinf2027/plans/E2.md, Sections 3 and 16), and chunked reads of their counts.

The 894,402 cells of E1b, identified by (file, row): the perturbed cells of E1's modelled VCC rows in each VCC
file, the 38,176 VCC control cells of the training file, and each Replogle screen's eligible perturbed cells and
control cells (the E1a label cache, checked against its manifest). Stack's units follow rule (a): (Flex batch,
split) on VCC, with each batch's control cells divided among the splits in proportion to their perturbed cells, and
gem groups on Replogle. Runs in the encoder environments (D2's packages) and in the pipeline environment.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import cellpass, config, singlecell  # noqa: E402
from bioinf2027.provenance import sha256  # noqa: E402

import make_e1a_extract as first  # noqa: E402
import make_e1a_label_extract as second  # noqa: E402

SPLITS = ("train", "val", "test")
SCREENS = ("RPE1", "K562-essential")
PROBE_SEED = 20271120
PROBE_CELLS = 2000


def seed_of(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def largest_remainder(weights, total: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if total == 0 or w.sum() <= 0:
        return np.zeros(w.size, dtype=np.int64)
    exact = w / w.sum() * total
    out = np.floor(exact).astype(np.int64)
    rest = total - int(out.sum())
    if rest:
        order = np.lexsort((np.arange(w.size), -(exact - out)))
        out[order[:rest]] += 1
    return out


def _cache(name: str) -> dict:
    manifest = json.loads((second.cache_dir() / "manifest.json").read_text())
    path = second.cache_dir() / f"{name}.npz"
    if sha256(path) != manifest["cache"][name]["sha256"]:
        raise SystemExit("the E1a label cache file %s does not match its manifest" % name)
    with np.load(path, allow_pickle=False) as store:
        return {k: store[k] for k in store.files}


def parts() -> dict:
    """Every part: its file, file rows (ascending), batch or gem group per cell, symbols, layout."""
    import h5py

    import make_e1a_labels as e1labels

    out = {}
    caches = {f"vcc_{s}": _cache(f"vcc_{s}") for s in SPLITS}
    rows = e1labels.vcc_row_table(caches)
    files = first.vcc_files()
    for split in SPLITS:
        positions = np.unique(np.concatenate(rows.loc[rows["split"] == split, "positions"].tolist()))
        cache = caches[f"vcc_{split}"]
        with h5py.File(files[split], "r") as handle:
            symbols = np.asarray(singlecell.read_annotation(handle["var"], "_index")).astype(str)
        out[f"vcc_perturbed_{split}"] = {"file": files[split], "file_name": f"vcc_{split}", "rows": cache["pert_rows"][positions],
                                         "group": cache["pert_batch"][positions].astype(str), "screen": "VCC", "split": split,
                                         "sparse": True, "symbols": symbols}
    train = caches["vcc_train"]
    with h5py.File(files["train"], "r") as handle:
        symbols = np.asarray(singlecell.read_annotation(handle["var"], "_index")).astype(str)
    out["vcc_control"] = {"file": files["train"], "file_name": "vcc_train", "rows": train["nt_rows"], "group": train["nt_batch"].astype(str),
                          "screen": "VCC", "split": None, "sparse": True, "symbols": symbols}
    for screen in SCREENS:
        cache = _cache(f"screen_{screen}")
        path = first.screen_files()[screen]
        with h5py.File(path, "r") as handle:
            gem = np.asarray(singlecell.read_annotation(handle["obs"], "gem_group")).astype(str)
            symbols = np.asarray(singlecell.read_annotation(handle["var"], "gene_name")).astype(str)
        for kind, key in (("perturbed", "pert_rows"), ("control", "nt_rows")):
            r = np.sort(cache[key])
            out[f"{screen}_{kind}"] = {"file": path, "file_name": screen, "rows": r, "group": gem[r], "screen": screen, "split": None,
                                       "sparse": False, "symbols": symbols}
    for p in out.values():
        if not np.all(np.diff(p["rows"]) > 0):
            raise SystemExit("rows of a part are not strictly ascending")
    return out


def cell_list(P: dict) -> list[tuple[str, int]]:
    """(part, index within part) for every cell, parts in a fixed order."""
    return [(name, i) for name in sorted(P) for i in range(P[name]["rows"].size)]


def probe_cells(P: dict) -> list[tuple[str, int]]:
    cells = cell_list(P)
    pick = np.sort(np.random.default_rng(PROBE_SEED).choice(len(cells), size=PROBE_CELLS, replace=False))
    return [cells[i] for i in pick]


def stack_units(P: dict) -> dict:
    """Rule (a): unit name -> list of (part, index); VCC controls divided among the splits per batch."""
    units: dict = {}
    for split in SPLITS:
        p = P[f"vcc_perturbed_{split}"]
        for i, b in enumerate(p["group"].tolist()):
            units.setdefault(f"VCC | {b} | {split}", []).append((f"vcc_perturbed_{split}", i))
    control = P["vcc_control"]
    for b in sorted(set(control["group"].tolist())):
        members = np.flatnonzero(control["group"] == b)
        members = members[np.random.default_rng(seed_of(f"E2 | Stack | controls | {b}")).permutation(members.size)]
        counts = [len(units.get(f"VCC | {b} | {s}", [])) for s in SPLITS]
        share = largest_remainder(counts, members.size)
        start = 0
        for s, n in zip(SPLITS, share):
            units.setdefault(f"VCC | {b} | {s}", []).extend(("vcc_control", int(i)) for i in members[start:start + n])
            start += n
    for screen in SCREENS:
        for kind in ("perturbed", "control"):
            p = P[f"{screen}_{kind}"]
            for i, g in enumerate(p["group"].tolist()):
                units.setdefault(f"{screen} | gem {g}", []).append((f"{screen}_{kind}", i))
    return units


def ordered(unit: str, members: list, variant: str = "order") -> list:
    order = np.random.default_rng(seed_of(f"E2 | Stack | {variant} | {unit}")).permutation(len(members))
    return [members[i] for i in order]


def read_rows(part: dict, wanted: np.ndarray, chunk_rows: int = 4000):
    """Yield (indices into the part, CSR counts of those cells over every gene of the file), in file order."""
    import h5py
    import scipy.sparse as sp

    rows = part["rows"][np.asarray(wanted)]
    order = np.argsort(rows, kind="stable")
    rows_sorted, wanted_sorted = rows[order], np.asarray(wanted)[order]
    with h5py.File(part["file"], "r") as handle:
        chunks = cellpass.csr_chunks(handle, chunk_rows) if part["sparse"] else cellpass.dense_chunks(handle, chunk_rows)
        for chunk in chunks:
            lo, hi = np.searchsorted(rows_sorted, [chunk.start, chunk.stop])
            if hi <= lo:
                continue
            local = rows_sorted[lo:hi] - chunk.start
            block = chunk.matrix[local]
            yield wanted_sorted[lo:hi], (block.tocsr() if sp.issparse(block) else sp.csr_matrix(block))


def gpu0_free_mb() -> int:
    import subprocess

    text = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    for line in text.strip().splitlines():
        index, free = [x.strip() for x in line.split(",")]
        if index == "0":
            return int(free)
    raise SystemExit("GPU 0 not found")


def verify_inputs() -> dict:
    return first.verify_single_cell_inputs()


def portable(path) -> str:
    return config.portable(Path(path))
