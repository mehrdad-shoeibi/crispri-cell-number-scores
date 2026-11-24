"""E2, Stack-Large cell embeddings under set rule (a) (bioinf2027/plans/E2.md, Sections 3, 15 and 16), GPU 0 only.

Each unit ((Flex batch, split) on VCC, gem group on Replogle; `e2cells.stack_units`) is passed in its seeded order
as an in-memory AnnData of raw counts (var index: the file's gene symbols) to Stack's `get_latent_representation`,
which cuts consecutive sets of the model's size and fills the last one with its own resampling (fixed seed),
keeping each cell's first copy. The checkpoint is read with torch.load(weights_only=True): its pickle holds only an
OrderedDict and tensors (inspected without running it), and the model is rebuilt from its `model_config` with
strict key matching, as Stack's own loader does. Run with the Stack environment (bioinf2027/requirements_stack.txt).

Modes:
  --probe A | B       eight sets: the first set of each of eight units drawn with seed 20271120; A and B in
                      separate runs (determinism); timing and peak GPU memory
  --compare           A against B; writes bioinf2027/results/E2/probe_stack.json
  --units PREFIX ...  full embeddings of the units whose names start with a prefix ("VCC", "RPE1", "K562-essential")
  --order2 PREFIX ... the order-sensitivity run (second order seed) of those units
  --stage PREFIX ...  one pass over each file writing every unit's raw counts to the data root (scratch)
Outputs under $IGA_STORAGE_ROOT/bioinf2027/E2/ (probe/stack/, embeddings/stack/).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if os.environ["CUDA_VISIBLE_DEVICES"] != "0":
    raise SystemExit("GPU 0 only")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import e2cells  # noqa: E402
import safe_pickle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WAIT_SECONDS = 600
PROBE_FREE_MB = 10000
PROBE_UNITS = 8
SETS_PER_BATCH = 1


def out_root() -> Path:
    return config.storage_root() / "bioinf2027" / "E2"


def downloaded(file: str) -> Path:
    record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "downloads.json").read_text())["downloads"]["stack_large"]["files"][file]
    path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["sha256"]:
        raise SystemExit("%s does not match its download record" % file)
    return path


def load_model():
    import torch

    from stack.models.core import StateICLModel

    gene_list = downloaded("basecount_1000per_15000max.pkl")
    allowed = {("numpy", "dtype"), ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar")}
    found = {g for g in safe_pickle.globals_in(gene_list) if g[0] != "<opcode>"}
    if not found <= allowed:
        raise SystemExit("the gene list would import %s" % sorted(found - allowed))
    checkpoint = torch.load(str(downloaded("bc_large.ckpt")), map_location="cpu", weights_only=True)
    cfg = checkpoint["hyper_parameters"]["model_config"]
    model = StateICLModel(**cfg)
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to("cuda").eval()
    record = {"model_config": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}, "parameters": len(state),
              "set_size": int(model.n_cells), "gene_list_globals": sorted(found)}
    return model, gene_list, record


def unit_adata(P: dict, members: list):
    """The unit's cells, in the given order, as an AnnData of raw counts over the file's genes."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    files = {P[p]["file_name"] for p, _ in members}
    symbols = {P[p]["file_name"]: P[p]["symbols"] for p, _ in members}
    reference = next(iter(symbols.values()))
    if any(not np.array_equal(s, reference) for s in symbols.values()):
        raise SystemExit("a unit mixes files with different genes: %s" % sorted(files))
    blocks, position = [], {}
    for part in sorted({p for p, _ in members}):
        idx = np.array(sorted({i for p, i in members if p == part}), dtype=np.int64)
        for got, block in e2cells.read_rows(P[part], idx):
            offset = sum(x.shape[0] for x in blocks)
            for j, i in enumerate(got.tolist()):
                position[(part, i)] = offset + j
            blocks.append(block)
    everything = sp.vstack(blocks).tocsr()
    matrix = everything[[position[m] for m in members]].tocsr().astype(np.float32)
    obs = pd.DataFrame({"organism": ["Homo sapiens"] * len(members)}, index=[f"{p}|{i}" for p, i in members])
    var = pd.DataFrame(index=pd.Index(reference, name="gene_symbol"))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ad.AnnData(X=matrix, obs=obs, var=var)


def safe_name(unit: str) -> str:
    return unit.replace(" | ", "__").replace(" ", "_")


def stage_root() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E2_stack_stage"


def stage(prefixes: list[str]) -> None:
    """One pass over each part's file: every unit's raw counts written to the data root, so a unit loads directly."""
    P = e2cells.parts()
    units = e2cells.stack_units(P)
    names = sorted(u for u in units if u.startswith(tuple(prefixes)))
    owner: dict = {}
    for u in names:
        for p, i in units[u]:
            owner.setdefault(p, {})[i] = u
    root = stage_root()
    started = time.time()
    for part, own in sorted(owner.items()):
        idx = np.array(sorted(own), dtype=np.int64)
        for got, block in e2cells.read_rows(P[part], idx):
            us = np.array([own[i] for i in got.tolist()])
            for u in np.unique(us):
                sel = np.flatnonzero(us == u)
                sub = block[sel].tocsr()
                folder = root / safe_name(str(u))
                folder.mkdir(parents=True, exist_ok=True)
                np.savez(folder / f"{part}__{int(got[sel][0])}.npz", data=sub.data.astype(np.float32), indices=sub.indices, indptr=sub.indptr,
                         shape=np.array(sub.shape), index=got[sel])
        print("[stack stage] %s done, %.0f s" % (part, time.time() - started), flush=True)
    complete = {u: len(units[u]) for u in names}
    (root / f"staged_{'_'.join(prefixes)}.json").write_text(json.dumps(complete, indent=1) + "\n")


def unit_adata_staged(P: dict, members: list, unit: str):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    import warnings

    blocks, position = [], {}
    for path in sorted((stage_root() / safe_name(unit)).glob("*.npz")):
        part = path.name.split("__")[0]
        with np.load(path, allow_pickle=False) as z:
            m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            offset = sum(x.shape[0] for x in blocks)
            for j, i in enumerate(z["index"].tolist()):
                position[(part, int(i))] = offset + j
        blocks.append(m)
    missing = [m for m in members if m not in position]
    if missing:
        raise SystemExit("unit %s: %d cells not staged" % (unit, len(missing)))
    matrix = sp.vstack(blocks).tocsr()[[position[m] for m in members]].tocsr().astype(np.float32)
    reference = P[members[0][0]]["symbols"]
    if any(not np.array_equal(P[p]["symbols"], reference) for p in {p for p, _ in members}):
        raise SystemExit("a unit mixes files with different genes")
    obs = pd.DataFrame({"organism": ["Homo sapiens"] * len(members)}, index=[f"{p}|{i}" for p, i in members])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(reference, name="gene_symbol")))


def embed_unit(model, gene_list: Path, adata) -> tuple[np.ndarray, float]:
    import torch

    started = time.time()
    with torch.no_grad():
        vectors, _ = model.get_latent_representation(adata, str(gene_list), batch_size=SETS_PER_BATCH, show_progress=False, num_workers=0)
    return np.asarray(vectors, dtype=np.float32), time.time() - started


def wait_for_gpu(needed_mb: float) -> dict:
    waited = 0
    while True:
        free = e2cells.gpu0_free_mb()
        if free >= needed_mb:
            return {"gpu0_free_mb_at_start": free, "needed_mb": needed_mb, "waited_seconds": waited}
        print("[stack] GPU 0 has %d MB free, %d needed; waiting" % (free, needed_mb), flush=True)
        time.sleep(WAIT_SECONDS)
        waited += WAIT_SECONDS


def probe_units(P: dict, units: dict, set_size: int) -> list:
    names = sorted(u for u, m in units.items() if len(m) >= set_size)
    pick = np.random.default_rng(e2cells.PROBE_SEED).choice(len(names), size=PROBE_UNITS, replace=False)
    return [(names[i], e2cells.ordered(names[i], units[names[i]])[:set_size]) for i in sorted(pick)]


def probe(mode: str) -> None:
    import torch

    gate = wait_for_gpu(PROBE_FREE_MB)
    P = e2cells.parts()
    units = e2cells.stack_units(P)
    model, gene_list, load_record = load_model()
    torch.cuda.reset_peak_memory_stats()
    vectors, ids, seconds = [], [], 0.0
    for name, members in probe_units(P, units, load_record["set_size"]):
        v, s = embed_unit(model, gene_list, unit_adata(P, members))
        vectors.append(v)
        ids += [f"{name}|{p}|{i}" for p, i in members]
        seconds += s
    folder = out_root() / "probe" / "stack"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"run_{mode}.npz", vectors=np.vstack(vectors), ids=np.array(ids))
    record = {"mode": mode, "gpu": gate, "model": load_record, "sets": PROBE_UNITS, "cells": len(ids), "gpu_seconds": round(seconds, 2),
              "seconds_per_set": round(seconds / PROBE_UNITS, 3), "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    (folder / f"run_{mode}.json").write_text(json.dumps(strict_json(record), indent=1) + "\n")
    print("[stack] probe %s: %d sets, %.2f s per set, peak %.0f MB" % (mode, PROBE_UNITS, record["seconds_per_set"], record["peak_gpu_memory_mb"]), flush=True)


def compare() -> None:
    folder = out_root() / "probe" / "stack"
    a, b = (np.load(folder / f"run_{m}.npz", allow_pickle=False) for m in ("A", "B"))
    if not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit("the probe runs hold different cells or orders")
    sd = a["vectors"].std(axis=0)
    rel = np.abs(a["vectors"] - b["vectors"]) / np.where(sd > 0, sd, np.inf)
    records = [json.loads((folder / f"run_{m}.json").read_text()) for m in ("A", "B")]
    sets = 3619
    per_set = max(r["seconds_per_set"] for r in records)
    out = {"plan": "bioinf2027/plans/E2.md, Sections 9 and 16", "cells": int(a["vectors"].shape[0]), "width": int(a["vectors"].shape[1]),
           "determinism": {"largest_difference_over_sd": float(rel.max()), "tolerance": 1e-4, "passed": bool(rel.max() <= 1e-4)},
           "seconds_per_set": per_set, "peak_gpu_memory_mb": max(r["peak_gpu_memory_mb"] for r in records),
           "sets_in_the_full_job": sets, "projected_gpu_hours": round(sets * per_set / 3600, 2), "model": records[0]["model"],
           "gpu_at_start": [r["gpu"] for r in records]}
    path = REPO / "bioinf2027" / "results" / "E2" / "probe_stack.json"
    path.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print("[stack] determinism %.2e of SD (passed %s); %.2f s per set; peak %.0f MB; projected %.2f GPU h for %d sets" % (
        rel.max(), out["determinism"]["passed"], per_set, out["peak_gpu_memory_mb"], out["projected_gpu_hours"], sets), flush=True)


def full(prefixes: list[str], variant: str = "order") -> None:
    probe_record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "probe_stack.json").read_text())
    if not probe_record["determinism"]["passed"]:
        raise SystemExit("the determinism check has not passed; no full embedding")
    if probe_record["projected_gpu_hours"] > 24:
        raise SystemExit("the projected job is above 24 GPU hours; back to the author")
    gate = wait_for_gpu(probe_record["peak_gpu_memory_mb"] + 1024)
    P = e2cells.parts()
    units = e2cells.stack_units(P)
    model, gene_list, load_record = load_model()
    folder = out_root() / "embeddings" / ("stack" if variant == "order" else "stack_order2") / "units"
    folder.mkdir(parents=True, exist_ok=True)
    for name in sorted(u for u in units if u.startswith(tuple(prefixes))):
        safe = safe_name(name)
        path = folder / f"{safe}.npz"
        if path.exists():
            continue
        members = e2cells.ordered(name, units[name], variant)
        staged = (stage_root() / safe).exists()
        vectors, seconds = embed_unit(model, gene_list, unit_adata_staged(P, members, name) if staged else unit_adata(P, members))
        if vectors.shape[0] != len(members) or not np.isfinite(vectors).all():
            raise SystemExit("unit %s: %d vectors for %d cells, or a non-finite value" % (name, vectors.shape[0], len(members)))
        temporary = folder / f"{safe}.tmp.npz"
        np.savez(temporary, vectors=vectors, part=np.array([p for p, _ in members]), index=np.array([i for _, i in members], dtype=np.int64))
        os.replace(temporary, path)
        print("[stack] %s: %d cells, %.1f s" % (name, len(members), seconds), flush=True)
    (folder.parent / f"run_{'_'.join(prefixes)}.json").write_text(json.dumps(strict_json({"gpu": gate, "model": load_record, "variant": variant}), indent=1) + "\n")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--probe"] and len(arguments) == 2 and arguments[1] in ("A", "B"):
        probe(arguments[1])
    elif arguments == ["--compare"]:
        compare()
    elif arguments[:1] == ["--units"] and len(arguments) > 1:
        full(arguments[1:])
    elif arguments[:1] == ["--stage"] and len(arguments) > 1:
        stage(arguments[1:])
    elif arguments[:1] == ["--order2"] and len(arguments) > 1:
        full(arguments[1:], variant="order 2")
    else:
        raise SystemExit(__doc__)
