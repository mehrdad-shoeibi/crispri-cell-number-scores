"""E5, the gene-identity vectors: Geneformer V2-104M's input (token) embedding of each target gene.

Design: bioinf2027/plans/E5.md, Sections 7 and 18. The 20,275 x 768 word-embedding matrix of the checkpoint already
held (`model.safetensors`, read directly, its digest and the token dictionaries' digests checked against D2's input
records) and D2's gene-to-token mapping (`bioinf2027.tokendrop.Encoder.token_of`, unchanged). A gene without a token
is reported as not covered. Run with D2's environment (Python 3.10, safetensors). scGPT's and GenePT's vectors are
added by later parts of this script after their downloads (gate G4).

Output: $IGA_STORAGE_ROOT/bioinf2027/E5/identity/geneformer_token.npz (Ensembl IDs, vectors, tokens) and
bioinf2027/results/E5/identity/geneformer_token.json (digests, coverage per screen and on VCC).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, features, tokendrop  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E5.md, Sections 7 and 18"
KEY = "bert.embeddings.word_embeddings.weight"
PINNED = ("checkpoint", "model_config", "token_dictionary", "ensembl_mapping_dictionary", "gene_name_id_dictionary")


def target_genes() -> dict:
    """Target Ensembl IDs per screen (eligible constructs) and VCC's target symbols with their IDs."""
    import h5py

    from bioinf2027 import singlecell

    import make_e1a_extract as first

    out = {}
    for screen in ("RPE1", "K562-essential"):
        table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        out[screen] = sorted(set(table.loc[table["n_perturbed_pool"] >= 27, "target_ensg"].astype(str)))
    with h5py.File(first.vcc_files()["train"], "r") as handle:
        names = np.asarray(singlecell.read_annotation(handle["var"], "_index")).astype(str)
        ids = np.asarray(singlecell.read_annotation(handle["var"], "gene_id")).astype(str)
    id_of = dict(zip(names.tolist(), ids.tolist()))
    with np.load(config.storage_root() / "bioinf2027" / "cache" / "E1a_labels" / "vcc_train.npz", allow_pickle=False) as store:
        vcc = sorted(set(store["targets"].astype(str).tolist()))
    out["VCC"] = sorted(id_of[s] for s in vcc if s in id_of)
    out["VCC symbols without an ID"] = [s for s in vcc if s not in id_of]
    return out


def main() -> None:
    from safetensors.numpy import load_file

    started = time.time()
    records = json.loads((REPO / "bioinf2027" / "results" / "D2" / "input_records.json").read_text())["encoder"]
    checked = {}
    for name in PINNED:
        rec = records[name]
        path = config.storage_root() / rec["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != rec["sha256"]:
            raise SystemExit("%s does not match D2's input record" % name)
        checked[name] = rec
    repository = config.storage_root() / "MADA" / "cache" / "foundation_models" / "geneformer-repo"
    encoder = tokendrop.Encoder(repository)
    weights = load_file(str(repository / "Geneformer-V2-104M" / "model.safetensors"))
    matrix = np.asarray(weights[KEY], dtype=np.float32)
    genes = target_genes()
    union = sorted(set(genes["RPE1"]) | set(genes["K562-essential"]) | set(genes["VCC"]))
    tokens = {g: encoder.token_of(g) for g in union}
    covered = [g for g in union if tokens[g] is not None]
    out_dir = config.storage_root() / "bioinf2027" / "E5" / "identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "geneformer_token.npz"
    np.savez(path, ensembl=np.array(covered), tokens=np.array([tokens[g] for g in covered], dtype=np.int64),
             vectors=matrix[[tokens[g] for g in covered]])
    coverage = {k: {"genes": len(v), "covered": sum(tokens.get(g) is not None for g in v)} for k, v in genes.items() if k in ("RPE1", "K562-essential", "VCC")}
    record = {"plan": PLAN, "inputs": checked, "matrix": {"key": KEY, "shape": list(matrix.shape)}, "coverage": coverage,
              "VCC symbols without an ID": genes["VCC symbols without an ID"], "output": file_record(path),
              "seconds": round(time.time() - started, 1)}
    folder = REPO / "bioinf2027" / "results" / "E5" / "identity"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "geneformer_token.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e5 identity] Geneformer tokens: %s" % json.dumps(coverage), flush=True)


if __name__ == "__main__":
    main()
