"""D2, input records: the encoder files and the cached per-cell embedding stores, with their layout.

Design: bioinf2027/plans/D2.md, Sections 3, 4, 10.2 (items 3, 4 and 6) and 12. The checkpoint, the
token dictionary, the gene-median dictionary and the Ensembl mapping dictionary are checked against
the digests the lock records (earlier_revision/env/geneformer_env.txt); the cached per-cell
embedding stores of VCC, RPE1 (chunks and top-up) and K562-essential are recorded with their indexes'
columns, so that the identity check and the token-drop pass can refuse anything that differs.
Nothing is embedded here.

Output: bioinf2027/results/D2/input_records.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, strict_json  # noqa: E402

import make_e1b_input_records as records  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "D2" / "input_records.json"
LOCKED = {
    "checkpoint": "fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f",
    "token_dictionary": "67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f",
    "gene_median_dictionary": "a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5",
    "ensembl_mapping_dictionary": "0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c",
}
COMMIT = "ad8f66dfcda3ebbd148d916c01f31339c5b95a15"


def encoder_paths() -> dict[str, Path]:
    repo = config.storage_root() / "MADA" / "cache" / "foundation_models" / "geneformer-repo"
    package = repo / "geneformer"
    return {
        "repository": repo,
        "checkpoint": repo / "Geneformer-V2-104M" / "model.safetensors",
        "model_config": repo / "Geneformer-V2-104M" / "config.json",
        "token_dictionary": package / "token_dictionary_gc104M.pkl",
        "gene_median_dictionary": package / "gene_median_dictionary_gc104M.pkl",
        "ensembl_mapping_dictionary": package / "ensembl_mapping_dict_gc104M.pkl",
        "gene_name_id_dictionary": package / "gene_name_id_dict_gc104M.pkl",
        "tokenizer_code": package / "tokenizer.py",
        "extractor_code": package / "emb_extractor.py",
        "perturber_utils_code": package / "perturber_utils.py",
    }


def store_paths() -> dict[str, Path]:
    root = config.storage_root()
    vcc = root / "MADA" / "cache" / "foundation_models" / "geneformer" / "full_feature_extraction_v2"
    rpe1 = root / "MADA_data" / "processed" / "phasec" / "rpe1_features" / "full"
    gf = root / "IGA_GENEFORMER_EXT"
    return {
        "vcc_cell_index": vcc / "cell_embedding_index.parquet",
        "vcc_perturbed_embeddings": vcc / "perturbed_embeddings",
        "vcc_control_embeddings": vcc / "nt_embeddings",
        "rpe1_master_cell_index": rpe1 / "master_cell_index_v1.parquet",
        "rpe1_chunks": rpe1 / "geneformer_chunks",
        "rpe1_topup": gf / "rpe1_topup",
        "k562_units": gf / "k562_essential" / "emb_units",
    }


def git_head(repo: Path) -> dict:
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return {"git_folder": False}
    text = head.read_text().strip()
    if text.startswith("ref:"):
        ref = repo / ".git" / text.split(" ", 1)[1]
        commit = ref.read_text().strip() if ref.exists() else None
    else:
        commit = text
    return {"git_folder": True, "head": commit, "equals_the_locked_commit": commit == COMMIT}


def folder_summary(folder: Path, sample: int = 3) -> dict:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    by_suffix: dict = {}
    for p in files:
        by_suffix[p.suffix] = by_suffix.get(p.suffix, 0) + 1
    out = {"path": config.portable(folder), "files": len(files), "by_suffix": by_suffix,
           "first_files": [p.relative_to(folder).as_posix() for p in files[:sample]],
           "last_files": [p.relative_to(folder).as_posix() for p in files[-sample:]]}
    layouts = {}
    seen = set()
    for p in files:
        if p.suffix in (".npy", ".npz", ".parquet", ".json", ".csv") and p.suffix not in seen:
            seen.add(p.suffix)
            try:
                layouts[p.relative_to(folder).as_posix()] = records.layout(p)
            except Exception as error:  # recorded, not fatal: the record says what could not be read
                layouts[p.relative_to(folder).as_posix()] = {"unreadable": repr(error)}
    out["layout_of_first_file_per_suffix"] = layouts
    out["listing"] = folder_record(files, folder)
    return out


def main() -> None:
    encoder = {}
    for name, path in encoder_paths().items():
        if name == "repository":
            encoder[name] = {"path": config.portable(path), **git_head(path)}
            continue
        record = file_record(path)
        if name in LOCKED:
            record["locked_sha256"] = LOCKED[name]
            record["equals_the_lock"] = record.get("sha256") == LOCKED[name]
        encoder[name] = record
        print("[d2 inputs] %s" % name, flush=True)
    stores = {}
    for name, path in store_paths().items():
        stores[name] = folder_summary(path) if path.is_dir() else {**file_record(path), **records.layout(path)}
        print("[d2 inputs] %s" % name, flush=True)
    out = {
        "plan": "bioinf2027/plans/D2.md, Sections 3, 4, 10.2 and 12",
        "encoder": encoder,
        "all_locked_digests_equal": bool(all(encoder[k].get("equals_the_lock") for k in LOCKED)),
        "cached_embedding_stores": stores,
        "single_cell_files": {k: file_record(v) for k, v in records.inputs().items() if k.endswith("single_cell")},
        "vcc_single_cell_files": {split: file_record(path) for split, path in {
            "train": config.storage_root() / "IGA_MAIN" / "data" / "vcc" / "train" / "adata_Training.h5ad",
            "val": config.storage_root() / "IGA_MAIN" / "data" / "vcc" / "validation" / "adata_Validation.h5ad",
            "test": config.storage_root() / "IGA_MAIN" / "data" / "vcc" / "test" / "adata_Test.h5ad"}.items()},
        "environment_pins": file_record(REPO / "bioinf2027" / "requirements_d2.txt"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print("[d2 inputs] written; locked digests equal: %s" % out["all_locked_digests_equal"], flush=True)


if __name__ == "__main__":
    main()
