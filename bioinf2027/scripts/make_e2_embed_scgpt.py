"""E2, scGPT whole-human cell embeddings (bioinf2027/plans/E2.md, Sections 3, 15 and 16), GPU 0 only.

As `scgpt.tasks.embed_data` (scgpt 0.2.4) with the recorded departures of Section 16: one generator per cell for
the binning tie-break and the gene subset of a cell with more than 1,199 expressed genes (seed from SHA-256 of
"E2 | scGPT | <file> | <row>"), the checkpoint read with weights_only=True and every encoder parameter required to
load from it, four loader-free preprocessing passes in the main process. Unchanged: the vocabulary match, 51 bins,
max_length 1,200, batches of 64 in part order, fp16 autocast, the CLS vector L2-normalised.

Modes (run with the scGPT environment, bioinf2027/requirements_scgpt.txt):
  --probe A | B | R   the 2,000 probe cells (seed 20271120): A and B in the same order (determinism, separate runs),
                      R reversed (other batch compositions); timing and peak GPU memory
  --compare           A against B and against R; writes bioinf2027/results/E2/probe_scgpt.json
  --part NAME ...     full embeddings of the named parts
Outputs under $IGA_STORAGE_ROOT/bioinf2027/E2/ (probe/scgpt/, embeddings/scgpt/).
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

REPO = Path(__file__).resolve().parents[2]
MAX_LENGTH = 1200
N_BINS = 51
BATCH = 64
WAIT_SECONDS = 600
PROBE_FREE_MB = 6000


def out_root() -> Path:
    return config.storage_root() / "bioinf2027" / "E2"


def downloaded(file: str) -> Path:
    record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "downloads.json").read_text())["downloads"]["scgpt_whole_human"]["files"][file]
    path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["sha256"]:
        raise SystemExit("%s does not match its download record" % file)
    return path


# --------------------------------------------------------------------------- model


def load_model():
    import torch

    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    vocab = GeneVocab.from_file(downloaded("vocab.json"))
    for s in ("<pad>", "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    configs = json.loads(downloaded("args.json").read_text())
    model = TransformerModel(ntoken=len(vocab), d_model=configs["embsize"], nhead=configs["nheads"], d_hid=configs["d_hid"],
                             nlayers=configs["nlayers"], nlayers_cls=configs["n_layers_cls"], n_cls=1, vocab=vocab, dropout=configs["dropout"],
                             pad_token=configs["pad_token"], pad_value=configs["pad_value"], do_mvc=True, do_dab=False, use_batch_labels=False,
                             domain_spec_batchnorm=False, explicit_zero_prob=False, use_fast_transformer=False, fast_transformer_backend="flash",
                             pre_norm=False)
    state = torch.load(str(downloaded("best_model.pt")), map_location="cpu", weights_only=True)
    state = {k.replace("Wqkv.", "in_proj_"): v for k, v in state.items()}  # scGPT's own mapping without flash-attention
    own = model.state_dict()
    loaded = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    missing = sorted(k for k in own if k not in loaded)
    needed = [k for k in missing if k.startswith(("encoder.", "value_encoder.", "transformer_encoder."))]
    if needed:
        raise SystemExit("encoder parameters not in the checkpoint: %s" % needed[:5])
    own.update(loaded)
    model.load_state_dict(own)
    model.to("cuda").eval()
    record = {"parameters_loaded": len(loaded), "parameters_in_model": len(own), "not_loaded": missing,
              "checkpoint_keys_unused": sorted(k for k in state if k not in loaded), "configs": {k: configs[k] for k in ("embsize", "nlayers", "nheads", "d_hid", "pad_value", "n_bins") if k in configs}}
    return model, vocab, configs, record


# --------------------------------------------------------------------------- preprocessing


def _digitize(x: np.ndarray, bins: np.ndarray, rng) -> np.ndarray:
    left = np.digitize(x, bins)
    right = np.digitize(x, bins, right=True)
    rands = rng.random(len(x))  # replaces np.random.rand (Section 16)
    return np.ceil(rands * (right - left) + left).astype(np.int64)


def cell_input(values: np.ndarray, genes: np.ndarray, rng, cls_id: int, pad_value: float):
    """One cell as scGPT's dataset and collator build it, with the cell's own generator."""
    genes = np.insert(genes, 0, cls_id).astype(np.int64)
    expr = np.insert(values.astype(np.float32), 0, np.float32(pad_value))
    row = expr[1:]
    if row.size and row.max() > 0:
        if row.min() <= 0:
            nz = row.nonzero()
            binned = np.zeros_like(row, dtype=np.int64)
            binned[nz] = _digitize(row[nz], np.quantile(row[nz], np.linspace(0, 1, N_BINS - 1)), rng)
        else:
            binned = _digitize(row, np.quantile(row, np.linspace(0, 1, N_BINS - 1)), rng)
        expr[1:] = binned.astype(np.float32)
    if genes.size > MAX_LENGTH:
        keep = np.concatenate([[0], rng.permutation(genes.size - 1)[: MAX_LENGTH - 1] + 1])
        genes, expr = genes[keep], expr[keep]
    return genes, expr


def embed(model, vocab, configs, part: dict, part_name: str, indices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Embeddings of the given cells of one part, in the order of ``indices``."""
    import torch

    gene_ids = np.array([vocab[g] if g in vocab else -1 for g in part["symbols"].tolist()], dtype=np.int64)
    keep = np.flatnonzero(gene_ids >= 0)
    cls_id, pad_id, pad_value = vocab["<cls>"], vocab[configs["pad_token"]], configs["pad_value"]
    position = {int(v): i for i, v in enumerate(indices.tolist())}
    inputs: list = [None] * len(indices)
    expressed = np.zeros(len(indices), dtype=np.int64)
    seen_share = np.zeros(len(indices))
    for idx, block in e2cells.read_rows(part, indices):
        totals = np.asarray(block.sum(axis=1)).ravel()
        sub = block[:, keep].tocsr()
        for j, i in enumerate(idx.tolist()):
            lo, hi = sub.indptr[j], sub.indptr[j + 1]
            values, genes = sub.data[lo:hi], gene_ids[keep[sub.indices[lo:hi]]]
            row = int(part["rows"][i])
            rng = np.random.default_rng(e2cells.seed_of(f"E2 | scGPT | {part['file_name']} | {row}"))
            k = position[i]
            inputs[k] = cell_input(values, genes, rng, cls_id, pad_value)
            expressed[k] = hi - lo
            seen_share[k] = float(values.sum() / totals[j]) if totals[j] > 0 else 0.0
    out = np.zeros((len(indices), configs["embsize"]), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for start in range(0, len(indices), BATCH):
            batch = inputs[start:start + BATCH]
            length = min(MAX_LENGTH, max(g.size for g, _ in batch))
            genes = np.full((len(batch), length), pad_id, dtype=np.int64)
            expr = np.full((len(batch), length), np.float32(pad_value), dtype=np.float32)
            for r, (g, e) in enumerate(batch):
                genes[r, : g.size], expr[r, : e.size] = g, e
            g_t = torch.from_numpy(genes).to("cuda")
            emb = model._encode(g_t, torch.from_numpy(expr).to("cuda"), src_key_padding_mask=g_t.eq(pad_id))
            out[start:start + len(batch)] = emb[:, 0, :].float().cpu().numpy()
    seconds = time.time() - started
    out = out / np.linalg.norm(out, axis=1, keepdims=True)
    return out, {"cells": int(len(indices)), "gpu_seconds": round(seconds, 2), "cells_per_second": round(len(indices) / seconds, 1),
                 "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "genes_matched": int(keep.size),
                 "genes_in_file": int(gene_ids.size), "median_expressed_genes": float(np.median(expressed)),
                 "share_above_1199_expressed": float((expressed > MAX_LENGTH - 1).mean()), "median_share_of_counts_seen": float(np.median(seen_share))}


def wait_for_gpu(needed_mb: float) -> dict:
    waited = 0
    while True:
        free = e2cells.gpu0_free_mb()
        if free >= needed_mb:
            return {"gpu0_free_mb_at_start": free, "needed_mb": needed_mb, "waited_seconds": waited}
        print("[scgpt] GPU 0 has %d MB free, %d needed; waiting" % (free, needed_mb), flush=True)
        time.sleep(WAIT_SECONDS)
        waited += WAIT_SECONDS


# --------------------------------------------------------------------------- modes


def probe(mode: str) -> None:
    gate = wait_for_gpu(PROBE_FREE_MB)
    P = e2cells.parts()
    cells = e2cells.probe_cells(P)
    if mode == "R":
        cells = cells[::-1]
    model, vocab, configs, load_record = load_model()
    vectors, ids, records = [], [], []
    for name in sorted({p for p, _ in cells}):
        idx = np.array([i for p, i in cells if p == name], dtype=np.int64)
        v, rec = embed(model, vocab, configs, P[name], name, idx)
        vectors.append(v)
        ids += [(name, int(i)) for i in idx]
        records.append({"part": name, **rec})
    folder = out_root() / "probe" / "scgpt"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"run_{mode}.npz", vectors=np.vstack(vectors), part=np.array([p for p, _ in ids]), index=np.array([i for _, i in ids]))
    (folder / f"run_{mode}.json").write_text(json.dumps(strict_json({"mode": mode, "gpu": gate, "model": load_record, "parts": records}), indent=1) + "\n")
    print("[scgpt] probe %s: %d cells, %s" % (mode, len(ids), json.dumps({r["part"]: r["cells_per_second"] for r in records})), flush=True)


def compare() -> None:
    folder = out_root() / "probe" / "scgpt"
    runs = {m: np.load(folder / f"run_{m}.npz", allow_pickle=False) for m in ("A", "B", "R")}
    keys = {m: [f"{p}|{i}" for p, i in zip(r["part"], r["index"])] for m, r in runs.items()}
    order = {m: np.argsort(keys[m]) for m in runs}
    v = {m: runs[m]["vectors"][order[m]] for m in runs}
    if not (sorted(keys["A"]) == sorted(keys["B"]) == sorted(keys["R"])):
        raise SystemExit("the probe runs hold different cells")
    sd = v["A"].std(axis=0)
    rel_b = np.abs(v["A"] - v["B"]) / sd
    rel_r = np.abs(v["A"] - v["R"]) / sd
    records = [json.loads((folder / f"run_{m}.json").read_text()) for m in ("A", "B", "R")]
    cells = sum(r["cells"] for r in records[0]["parts"])
    seconds = sum(r["gpu_seconds"] for r in records[0]["parts"])
    peak = max(r["peak_gpu_memory_mb"] for rec in records for r in rec["parts"])
    out = {"plan": "bioinf2027/plans/E2.md, Sections 9 and 16", "cells": cells,
           "determinism": {"largest_difference_over_sd": float(rel_b.max()), "tolerance": 1e-4, "passed": bool(rel_b.max() <= 1e-4)},
           "reverse_order": {"largest_difference_over_sd": float(rel_r.max()), "median": float(np.median(rel_r)), "reported_only": True},
           "cells_per_second_probe": round(cells / seconds, 1), "peak_gpu_memory_mb": peak,
           "projected_gpu_hours_for_894402_cells": round(894402 / (cells / seconds) / 3600, 2),
           "per_part": records[0]["parts"], "model": records[0]["model"], "gpu_at_start": [r["gpu"] for r in records]}
    path = REPO / "bioinf2027" / "results" / "E2" / "probe_scgpt.json"
    path.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print("[scgpt] determinism %.2e of SD (passed %s); reverse order %.2e; %.0f cells/s; peak %.0f MB; projected %.2f GPU h" % (
        rel_b.max(), out["determinism"]["passed"], rel_r.max(), out["cells_per_second_probe"], peak, out["projected_gpu_hours_for_894402_cells"]), flush=True)


def full(names: list[str]) -> None:
    probe_record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "probe_scgpt.json").read_text())
    if not probe_record["determinism"]["passed"]:
        raise SystemExit("the determinism check has not passed; no full embedding")
    if probe_record["projected_gpu_hours_for_894402_cells"] > 24:
        raise SystemExit("the projected job is above 24 GPU hours; back to the author")
    gate = wait_for_gpu(probe_record["peak_gpu_memory_mb"] + 1024)
    P = e2cells.parts()
    model, vocab, configs, load_record = load_model()
    folder = out_root() / "embeddings" / "scgpt"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = folder / f"{name}.npy"
        if path.exists():
            print("[scgpt] %s exists; kept" % name, flush=True)
            continue
        idx = np.arange(P[name]["rows"].size)
        vectors, rec = embed(model, vocab, configs, P[name], name, idx)
        if not np.isfinite(vectors).all():
            raise SystemExit("non-finite embedding in %s" % name)
        temporary = folder / f"{name}.tmp.npy"
        np.save(temporary, vectors)
        os.replace(temporary, path)
        np.save(folder / f"{name}_rows.npy", P[name]["rows"])
        meta = {"part": name, "file": e2cells.portable(P[name]["file"]), "gpu": gate, "model": load_record, **rec,
                "embeddings": file_record(path), "rows": file_record(folder / f"{name}_rows.npy")}
        (folder / f"{name}.json").write_text(json.dumps(strict_json(meta), indent=1) + "\n")
        print("[scgpt] %s: %d cells, %.0f cells/s, peak %.0f MB" % (name, rec["cells"], rec["cells_per_second"], rec["peak_gpu_memory_mb"]), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--probe"] and len(arguments) == 2 and arguments[1] in ("A", "B", "R"):
        probe(arguments[1])
    elif arguments == ["--compare"]:
        compare()
    elif arguments[:1] == ["--part"] and len(arguments) > 1:
        full(arguments[1:])
    else:
        raise SystemExit(__doc__)
