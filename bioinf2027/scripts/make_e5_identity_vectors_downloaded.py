"""E5, the gene-identity vectors from the downloads: scGPT's gene embedding and GenePT's text embedding.

Design: bioinf2027/plans/E5.md, Sections 7 and 18 (gate G4, item 7). Each downloaded file is checked against the
SHA-256 recorded at download (bioinf2027/results/E2/downloads.json) before it is read.
- scGPT whole-human: `model.encoder`, an embedding then LayerNorm (eps 1e-5), applied to each gene's vocabulary id;
  the checkpoint is read with `torch.load(weights_only=True)` (tensors only, no code); keys checked.
- GenePT v2: `GenePT_gene_embedding_ada_text.pickle` (text-embedding-ada-002 of the gene's NCBI summary), upper-case
  symbols; its opcodes hold no global (bioinf2027/scripts/safe_pickle.py), and it is loaded with no global allowed.
Genes are keyed by Ensembl ID through each screen's own symbol (the endpoint table's target_symbol, which equals the
file's gene_name at the target column) and, on VCC, through the VCC file's gene_id. Run with D2's environment.

Outputs: $IGA_STORAGE_ROOT/bioinf2027/E5/identity/{scgpt_gene,genept}.npz and
bioinf2027/results/E5/identity/{scgpt_gene,genept}.json (digests, coverage per screen and on VCC).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import safe_pickle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E5.md, Sections 7 and 18"


def downloaded(name: str, file: str) -> Path:
    record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "downloads.json").read_text())["downloads"][name]["files"][file]
    path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["sha256"]:
        raise SystemExit("%s/%s does not match its download record" % (name, file))
    return path


def targets() -> dict:
    """Per screen: Ensembl ID -> symbol of the eligible target genes; on VCC: Ensembl ID -> symbol."""
    import h5py

    from bioinf2027 import singlecell

    import make_e1a_extract as first

    out = {}
    for screen in ("RPE1", "K562-essential"):
        table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        table = table[table["n_perturbed_pool"] >= 27]
        out[screen] = dict(zip(table["target_ensg"].astype(str), table["target_symbol"].astype(str)))
    with h5py.File(first.vcc_files()["train"], "r") as handle:
        names = np.asarray(singlecell.read_annotation(handle["var"], "_index")).astype(str)
        ids = np.asarray(singlecell.read_annotation(handle["var"], "gene_id")).astype(str)
    with np.load(config.storage_root() / "bioinf2027" / "cache" / "E1a_labels" / "vcc_train.npz", allow_pickle=False) as store:
        vcc = set(store["targets"].astype(str).tolist())
    out["VCC"] = {i: n for n, i in zip(names.tolist(), ids.tolist()) if n in vcc}
    return out


def write(name: str, by_ensembl: dict, genes: dict, record: dict) -> None:
    folder = config.storage_root() / "bioinf2027" / "E5" / "identity"
    folder.mkdir(parents=True, exist_ok=True)
    ensembl = sorted(by_ensembl)
    path = folder / f"{name}.npz"
    np.savez(path, ensembl=np.array(ensembl), vectors=np.vstack([by_ensembl[g] for g in ensembl]).astype(np.float32))
    coverage = {k: {"genes": len(v), "covered": sum(g in by_ensembl for g in v)} for k, v in genes.items()}
    out = {"plan": PLAN, **record, "coverage": coverage, "output": file_record(path)}
    results = REPO / "bioinf2027" / "results" / "E5" / "identity"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{name}.json").write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print("[e5 identity] %s: %s" % (name, json.dumps(coverage)), flush=True)


def scgpt(genes: dict) -> None:
    import torch

    model = downloaded("scgpt_whole_human", "best_model.pt")
    vocab_path = downloaded("scgpt_whole_human", "vocab.json")
    state = torch.load(str(model), map_location="cpu", weights_only=True)
    keys = ("encoder.embedding.weight", "encoder.enc_norm.weight", "encoder.enc_norm.bias")
    missing = [k for k in keys if k not in state]
    if missing:
        raise SystemExit("the scGPT checkpoint has no %s" % missing)
    table = state[keys[0]].float()
    vectors = torch.nn.functional.layer_norm(table, (table.shape[1],), state[keys[1]].float(), state[keys[2]].float(), eps=1e-5).numpy()
    vocab = json.loads(vocab_path.read_text())
    by_ensembl = {}
    for part in genes.values():
        for ensembl, symbol in part.items():
            if symbol in vocab and ensembl not in by_ensembl:
                by_ensembl[ensembl] = vectors[vocab[symbol]]
    write("scgpt_gene", by_ensembl, genes, {"inputs": {"best_model.pt": file_record(model), "vocab.json": file_record(vocab_path)},
                                            "vectors": {"key": keys[0], "shape": list(table.shape), "layer_norm_eps": 1e-5}})


def genept(genes: dict) -> None:
    zip_record = downloaded("genept_v2", "GenePT_emebdding_v2.zip")
    path = config.storage_root() / "bioinf2027" / "downloads" / "genept_v2_extracted" / "GenePT_emebdding_v2" / "GenePT_gene_embedding_ada_text.pickle"
    found = {g for g in safe_pickle.globals_in(path) if g[0] != "<opcode>"} | {g for g in safe_pickle.globals_in(path) if g[0] == "<opcode>"}
    if found:
        raise SystemExit("the GenePT pickle would import or build objects: %s" % sorted(found)[:5])
    data = safe_pickle.load(path)
    table = {str(k).upper(): np.asarray(v, dtype=np.float64) for k, v in data.items()}
    widths = {v.size for v in table.values()}
    if widths != {1536}:
        raise SystemExit("GenePT vectors have widths %s" % sorted(widths))
    by_ensembl = {}
    for part in genes.values():
        for ensembl, symbol in part.items():
            if symbol.upper() in table and ensembl not in by_ensembl:
                by_ensembl[ensembl] = table[symbol.upper()]
    write("genept", by_ensembl, genes, {"inputs": {"zip": file_record(zip_record), "pickle": file_record(path),
                                                   "pickle_globals": "none (opcodes read without running, bioinf2027/scripts/safe_pickle.py)"},
                                        "vectors": {"genes_in_file": len(table), "width": 1536}})


if __name__ == "__main__":
    started = time.time()
    genes = targets()
    parts = sys.argv[1:] or ["scgpt", "genept"]
    if "scgpt" in parts:
        scgpt(genes)
    if "genept" in parts:
        genept(genes)
    print("[e5 identity] done in %.0f s" % (time.time() - started), flush=True)
