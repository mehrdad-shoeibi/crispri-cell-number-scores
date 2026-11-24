"""E2, scGPT: the share of a cell's counts in the genes scGPT is given, after its cut to 1,199 genes (E2's results
review, round 1, blocking issue 1: bioinf2027/reviews/E2_results_r1.md).

The field "median_share_of_counts_seen" of scGPT in bioinf2027/results/E2/coverage.json (and in the part records of
make_e2_embed_scgpt.py) is the share of a cell's counts in genes of scGPT's vocabulary, before the cut that keeps
a random 1,199 of a cell's expressed vocabulary genes (max_length 1,200 with the CLS token). This script rebuilds,
for every cell of every part and without the GPU, the input that make_e2_embed_scgpt.py gave the model: the same
reading (e2cells.read_rows), the same vocabulary match, the same generator per cell (seed from SHA-256 of
"E2 | scGPT | <file> | <row>") and the same function (make_e2_embed_scgpt.cell_input), so that the binning
tie-break draws come before the gene subset as they did. Per cell: the share of its counts in vocabulary genes
(before the cut), the share in the genes kept for the model (after the cut) and the number of expressed vocabulary
genes. The medians before the cut and the share of cells above 1,199 expressed vocabulary genes must equal the part
records, or nothing is written (hard limit 9). Medians and quartiles per part and per screen.

Output: bioinf2027/results/E2/coverage_scgpt_input.json. Pipeline environment; reads only; --workers N (default 4)
reads parts in parallel and changes no value.
"""
from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import e2cells  # noqa: E402
import make_e2_embed_scgpt as emb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "coverage_scgpt_input.json"
RECORDS = config.storage_root() / "bioinf2027" / "E2" / "embeddings" / "scgpt"
WORKERS = 4
SCREENS = {"VCC": ("vcc_perturbed_train", "vcc_perturbed_val", "vcc_perturbed_test", "vcc_control"),
           "RPE1": ("RPE1_perturbed", "RPE1_control"), "K562-essential": ("K562-essential_perturbed", "K562-essential_control")}
DATA: dict = {}


class Stop(RuntimeError):
    pass


def kept_positions(values: np.ndarray, rng, pad_value: float) -> np.ndarray:
    """The positions (0-based, into the cell's expressed vocabulary genes) that make_e2_embed_scgpt.cell_input keeps.
    The positions stand in for the gene identifiers, which cell_input only carries along."""
    genes, _ = emb.cell_input(values, np.arange(1, values.size + 1, dtype=np.int64), rng, 0, pad_value)
    return genes[1:] - 1


def cell_shares(values: np.ndarray, total: float, rng, pad_value: float) -> tuple[float, float]:
    """(share before the cut, share after it), as make_e2_embed_scgpt.embed computes the first."""
    if total <= 0:
        return 0.0, 0.0
    kept = kept_positions(values, rng, pad_value)
    return float(values.sum() / total), float(values[kept].sum() / total)


def part_shares(name: str) -> dict:
    part, vocab, pad_value = DATA["parts"][name], DATA["vocab"], DATA["pad_value"]
    keep = np.flatnonzero(np.array([s in vocab for s in part["symbols"].tolist()]))
    n = part["rows"].size
    before, after, expressed = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64)
    started = time.time()
    for idx, block in e2cells.read_rows(part, np.arange(n)):
        totals = np.asarray(block.sum(axis=1)).ravel()
        sub = block[:, keep].tocsr()
        for j, i in enumerate(idx.tolist()):
            lo, hi = sub.indptr[j], sub.indptr[j + 1]
            rng = np.random.default_rng(e2cells.seed_of(f"E2 | scGPT | {part['file_name']} | {int(part['rows'][i])}"))
            before[i], after[i] = cell_shares(sub.data[lo:hi], totals[j], rng, pad_value)
            expressed[i] = hi - lo
    print("[e2 scgpt input] %s: %d cells, %.0f s" % (name, n, time.time() - started), flush=True)
    return {"name": name, "before": before, "after": after, "expressed": expressed, "genes_matched": int(keep.size), "genes_in_file": int(part["symbols"].size)}


def summary(before: np.ndarray, after: np.ndarray, expressed: np.ndarray) -> dict:
    q = lambda a: [float(v) for v in np.quantile(a, [0.25, 0.5, 0.75])]  # noqa: E731
    return {"cells": int(before.size), "median_expressed_vocabulary_genes": float(np.median(expressed)),
            "share_above_1199_expressed": float((expressed > emb.MAX_LENGTH - 1).mean()),
            "share_of_counts_in_vocabulary_genes_before_the_cut": {"median": float(np.median(before)), "quartiles": q(before)},
            "share_of_counts_in_the_genes_input_after_the_cut": {"median": float(np.median(after)), "quartiles": q(after)}}


def main() -> None:
    started = time.time()
    DATA["parts"] = e2cells.parts()
    DATA["vocab"] = set(json.loads(emb.downloaded("vocab.json").read_text()))
    DATA["pad_value"] = json.loads(emb.downloaded("args.json").read_text())["pad_value"]
    names = [p for parts in SCREENS.values() for p in parts]
    if sorted(names) != sorted(DATA["parts"]):
        raise Stop("the parts differ from the screens listed here: %s" % sorted(DATA["parts"]))
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        results = {r["name"]: r for r in pool.map(part_shares, names)}
    out, problems = {"parts": {}, "screens": {}}, []
    for name in names:
        r, record = results[name], json.loads((RECORDS / f"{name}.json").read_text())
        s = summary(r["before"], r["after"], r["expressed"])
        s.update(genes_matched=r["genes_matched"], genes_in_file=r["genes_in_file"],
                 part_record={k: record[k] for k in ("cells", "median_share_of_counts_seen", "share_above_1199_expressed", "median_expressed_genes", "genes_matched")})
        same = (record["cells"] == s["cells"] and record["genes_matched"] == s["genes_matched"]
                and record["median_share_of_counts_seen"] == s["share_of_counts_in_vocabulary_genes_before_the_cut"]["median"]
                and record["share_above_1199_expressed"] == s["share_above_1199_expressed"]
                and record["median_expressed_genes"] == s["median_expressed_vocabulary_genes"])
        s["equal_to_the_part_record"] = same
        if not same:
            problems.append(name)
        out["parts"][name] = s
    if problems:
        raise Stop("the rebuilt inputs differ from the part records of %s: %s" % (problems, json.dumps({p: out["parts"][p] for p in problems})))
    for screen, parts in SCREENS.items():
        out["screens"][screen] = summary(*(np.concatenate([results[p][k] for p in parts]) for k in ("before", "after", "expressed")))
    record = {"check": "E2 results review, round 1, blocking issue 1 (bioinf2027/reviews/E2_results_r1.md)",
              "rule": "scGPT's input rebuilt cell by cell as make_e2_embed_scgpt.py built it (cell_input, the same generator per cell); "
                      "'before the cut' is the field median_share_of_counts_seen of coverage.json and of the part records; "
                      "'after the cut' is the share of the cell's counts in the genes given to the model (at most 1,199)",
              **out, "inputs": {"part_records": {p: file_record(RECORDS / f"{p}.json") for p in names},
                                "downloads": file_record(REPO / "bioinf2027" / "results" / "E2" / "downloads.json")},
              "code": {config.portable(Path(m.__file__).resolve()): sha256(Path(m.__file__)) for m in (sys.modules[__name__], emb, e2cells)},
              "seconds": round(time.time() - started, 1)}
    OUT.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 scgpt input] done in %.0f s; median share after the cut: %s" % (time.time() - started, {
        k: round(v["share_of_counts_in_the_genes_input_after_the_cut"]["median"], 4) for k, v in out["screens"].items()}), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--workers"] and len(arguments) == 2 and arguments[1].isdigit():
        WORKERS = int(arguments[1])
    elif arguments:
        raise SystemExit("usage: make_e2_scgpt_input_share.py [--workers N]")
    main()
