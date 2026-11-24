"""The token-drop arm of D2: Geneformer cell embeddings with the target gene's token removed.

Design: `bioinf2027/plans/D2.md`, Sections 3, 5, 10, 11 and 12. Everything here repeats the settings
of the submitted embeddings (`earlier_revision/experiment_specs/option_b_geneformer_experiment_lock.md`,
Section 5; `earlier_revision/smoke/adapter.py`; `earlier_revision/pipeline/phase3_runner.py:99-121, 189`):
Geneformer-V2-104M, frozen, CLS token of layer quant_layers + (-1), fp32, the tokenizer of the pinned
repository with model_input_size 4,096 and model_version V2, n_counts the cell's total over every gene of
its count matrix, forward batch size 16.

The drop (Section 10.1, item 2, and Section 11.2, item 4): every input column whose collapsed Ensembl
identifier equals the target's is set to zero while the cell's total stays the original one, so the
other genes' normalised values are unchanged; the tokenizer then ranks the remaining genes and keeps
the top 4,094. For every cell the dropped token sequence is checked to equal the undropped full
ranking with the target removed, cut to 4,094 genes, which is the undropped sequence with the target
removed plus at most one gene appended at position 4,094.

Functions needing torch or the Geneformer package import them on call; the module itself imports
neither, so the pipeline environment can import it for tests.
"""
from __future__ import annotations

import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "MODEL_INPUT_SIZE",
    "GENES_KEPT",
    "Encoder",
    "expected_after_drop",
    "check_drop",
    "truncate",
]

MODEL_INPUT_SIZE = 4096
GENES_KEPT = MODEL_INPUT_SIZE - 2  # room for the CLS and EOS tokens
FORWARD_BATCH = 16


def truncate(full_ranking: np.ndarray, cls: int, eos: int) -> np.ndarray:
    """The model input of a full rank encoding: CLS, the first 4,094 genes, EOS."""
    return np.concatenate([[cls], np.asarray(full_ranking)[:GENES_KEPT], [eos]]).astype(np.int64)


def expected_after_drop(full_ranking: np.ndarray, token: int | None) -> np.ndarray:
    """The full ranking with every occurrence of the target's token removed."""
    full_ranking = np.asarray(full_ranking)
    if token is None:
        return full_ranking
    return full_ranking[full_ranking != token]


def check_drop(undropped_full: np.ndarray, dropped_full: np.ndarray, token: int | None) -> dict:
    """The assertion of Section 11.2, item 4, on the model inputs of one cell.

    The dropped input must equal the undropped input with the target's token removed, plus at most
    one gene appended at the last gene position (the gene that moves up into the 4,094).
    """
    expected = expected_after_drop(undropped_full, token)
    full_equal = bool(np.array_equal(np.asarray(dropped_full), expected))
    undropped_cut = np.asarray(undropped_full)[:GENES_KEPT]
    dropped_cut = np.asarray(dropped_full)[:GENES_KEPT]
    held = bool(token is not None and token in set(undropped_cut.tolist()))
    without = undropped_cut[undropped_cut != token] if token is not None else undropped_cut
    head_equal = bool(np.array_equal(dropped_cut[: without.size], without))
    appended = int(dropped_cut.size - without.size)
    return {"full_ranking_equal": full_equal, "input_equal_with_target_removed": head_equal,
            "appended": appended, "target_in_input": held, "passed": bool(full_equal and head_equal and appended <= 1)}


@dataclass
class Encoder:
    """The pinned Geneformer repository, model and dictionaries, loaded once per process."""

    repository: Path
    device: str = "cuda"

    def __post_init__(self) -> None:
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))
        package = self.repository / "geneformer"
        with open(package / "token_dictionary_gc104M.pkl", "rb") as handle:
            self.token_dict = pickle.load(handle)
        with open(package / "ensembl_mapping_dict_gc104M.pkl", "rb") as handle:
            self.mapping = pickle.load(handle)
        self.token_gene = {v: k for k, v in self.token_dict.items()}
        self.pad = self.token_dict["<pad>"]
        self.cls = self.token_dict["<cls>"]
        self.eos = self.token_dict["<eos>"]
        self.model = None
        self.layer = None

    # ------------------------------------------------------------------ tokens

    def collapsed(self, ensembl: str) -> str | None:
        """The collapsed identifier the tokenizer uses for an input Ensembl identifier."""
        return self.mapping.get(str(ensembl).upper())

    def token_of(self, ensembl: str) -> int | None:
        collapsed = self.collapsed(ensembl)
        if collapsed is None:
            return None
        return self.token_dict.get(collapsed)

    def columns_of(self, var_ensembl: np.ndarray, ensembl: str) -> np.ndarray:
        """Input columns whose collapsed identifier equals the target's (the tokenizer sums them)."""
        target = self.collapsed(ensembl)
        if target is None:
            return np.array([], dtype=np.int64)
        collapsed = np.array([self.mapping.get(str(e).upper()) for e in var_ensembl], dtype=object)
        return np.flatnonzero(collapsed == target)

    def full_rankings(self, counts, var_ensembl: np.ndarray, cell_ids, n_counts: np.ndarray, work: Path) -> dict:
        """The tokenizer's full rank encoding of each cell (before truncation), keyed by cell id.

        The counts are written to an h5ad file with obs n_counts as given (the cell's original total)
        and var ensembl_id, and tokenized by the pinned TranscriptomeTokenizer.
        """
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp
        from geneformer import TranscriptomeTokenizer

        work.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=work))
        try:
            matrix = sp.csr_matrix(np.asarray(counts, dtype=np.float32)) if not sp.issparse(counts) else counts.astype(np.float32)
            obs = pd.DataFrame({"cell_id": [str(c) for c in cell_ids]})
            data = ad.AnnData(X=matrix, obs=obs)
            data.var_names = [str(e) for e in var_ensembl]
            data.var["ensembl_id"] = [str(e) for e in var_ensembl]
            # Single precision, as the submitted adapter and runner set it from the float32 row sum, so that
            # the tokenizer's first step (counts / n_counts * 10,000) rounds as it did for the cached embeddings.
            data.obs["n_counts"] = np.asarray(n_counts, dtype=np.float32)
            data.obs_names = data.obs["cell_id"].astype(str)
            path = folder / "unit.h5ad"
            data.write_h5ad(path)
            tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={"cell_id": "cell_id"}, nproc=1, chunk_size=512,
                                               model_input_size=MODEL_INPUT_SIZE, model_version="V2")
            cells, metadata, _ = tokenizer.tokenize_anndata(path)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return {str(c): np.asarray(t, dtype=np.int64) for c, t in zip(metadata["cell_id"], cells)}

    # ------------------------------------------------------------------ embeddings

    def load(self) -> None:
        if self.model is not None:
            return
        from geneformer import perturber_utils as pu

        self.model = pu.load_model("Pretrained", 0, str(self.repository / "Geneformer-V2-104M"), "eval")
        self.layer = pu.quant_layers(self.model) + (-1)

    def embed(self, inputs: dict, work: Path) -> dict:
        """CLS embeddings of model inputs (cell id -> token array), as the submitted runner computed them."""
        import torch
        from datasets import Dataset
        from geneformer import perturber_utils as pu
        from geneformer.emb_extractor import get_embs

        self.load()
        ids = list(inputs)
        dataset = Dataset.from_dict({"input_ids": [inputs[c].tolist() for c in ids], "length": [int(inputs[c].size) for c in ids], "cell_id": ids})
        work.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=work))
        try:
            dataset.save_to_disk(str(folder / "unit.dataset"))
            loaded = pu.load_and_filter(None, 1, str(folder / "unit.dataset"))
            loaded = pu.downsample_and_sort(loaded, len(ids) + 10)
            embs = get_embs(self.model, loaded, "cls", self.layer, self.pad, FORWARD_BATCH, self.token_gene, silent=True)
            torch.cuda.synchronize()
            array = embs.cpu().float().numpy().astype(np.float32)
            order = [str(c) for c in loaded["cell_id"]]
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return {c: array[i] for i, c in enumerate(order)}


def environment_record() -> dict:
    """Versions and settings of the embedding environment (Section 10.2, item 3)."""
    import datasets
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "forward_batch_size": FORWARD_BATCH,
        "length_sorting": "perturber_utils.downsample_and_sort, longest first",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
