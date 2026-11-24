"""E2, coverage of each encoder's gene mapping (bioinf2027/plans/E2.md, Section 9 check 3, and Sections 16 and 17).

Per encoder (scGPT: its vocabulary `vocab.json`; Stack: its gene list `basecount_1000per_15000max.pkl`, read with the
restricted unpickler of `safe_pickle.py`, both checked against their download records) and per file (VCC, RPE1,
K562-essential): the file's genes the encoder matches (exact symbol); the share of the 2,000 feature genes
(`features.hvg_symbols()`) and of the file's target genes it matches; and per part the median over the part's
embedded cells of the share of a cell's counts on matched genes (scGPT: from its embedding records, computed during
embedding; Stack: computed here from the counts of every embedded cell). Check 3: a share of the feature genes below
one half in a screen is reported as a stop for that encoder and screen.
Output: bioinf2027/results/E2/coverage.json.
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

import e2cells  # noqa: E402
import make_e2_features as e2feat  # noqa: E402
import safe_pickle  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "coverage.json"
PLAN = "bioinf2027/plans/E2.md, Section 9 (check 3) and Sections 16 and 17"


def downloaded(entry: str, file: str) -> Path:
    record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "downloads.json").read_text())["downloads"][entry]["files"][file]
    path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["sha256"]:
        raise SystemExit("%s does not match its download record" % file)
    return path


def vocabularies() -> dict:
    scgpt = set(json.loads(downloaded("scgpt_whole_human", "vocab.json").read_text()))
    path = downloaded("stack_large", "basecount_1000per_15000max.pkl")
    found = {g for g in safe_pickle.globals_in(path) if g[0] != "<opcode>"}
    if not found <= safe_pickle.NUMPY_ARRAYS:
        raise SystemExit("Stack's gene list holds globals outside numpy's array set: %s" % sorted(found - safe_pickle.NUMPY_ARRAYS))
    genes = safe_pickle.load(path, found)
    genes = list(genes.keys()) if isinstance(genes, dict) else np.asarray(genes).astype(str).ravel().tolist()
    return {"scgpt": scgpt, "stack": set(str(g) for g in genes)}


def target_genes(P: dict, rows) -> dict:
    out = {"VCC": sorted(set(rows["target_gene"].astype(str)))}
    for screen in e2cells.SCREENS:
        table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        out[screen] = sorted(set(table["target_symbol"].astype(str)))
    return out


def stack_share(part: dict, vocabulary: set) -> float:
    keep = np.flatnonzero(np.isin(part["symbols"], sorted(vocabulary)))
    shares = []
    for _, block in e2cells.read_rows(part, np.arange(part["rows"].size)):
        totals = np.asarray(block.sum(axis=1)).ravel()
        seen = np.asarray(block[:, keep].sum(axis=1)).ravel()
        shares.append(np.where(totals > 0, seen / np.where(totals > 0, totals, 1), 0.0))
    return float(np.median(np.concatenate(shares)))


def main() -> None:
    started = time.time()
    P = e2cells.parts()
    rows, _ = e2feat.row_table(P)
    vocab = vocabularies()
    hvg = np.asarray(features.hvg_symbols()).astype(str)
    targets = target_genes(P, rows)
    scgpt_records = config.storage_root() / "bioinf2027" / "E2" / "embeddings" / "scgpt"
    files = {"VCC": [p for p in P if p.startswith("vcc_")], **{s: [p for p in P if p.startswith(s)] for s in e2cells.SCREENS}}
    out: dict = {}
    for encoder, vocabulary in vocab.items():
        out[encoder] = {"vocabulary_size": len(vocabulary)}
        for name, parts in files.items():
            symbols = P[parts[0]]["symbols"].astype(str)
            if any(not np.array_equal(P[p]["symbols"], P[parts[0]]["symbols"]) for p in parts):
                raise SystemExit("the parts of %s list other genes" % name)
            share_features = float(np.isin(hvg, sorted(vocabulary)).mean())
            entry = {"genes_in_file": int(symbols.size), "genes_matched": int(np.isin(symbols, sorted(vocabulary)).sum()),
                     "feature_genes": int(hvg.size), "share_of_feature_genes_matched": share_features,
                     "feature_genes_in_file_and_matched": int(np.isin(hvg, sorted(set(symbols.tolist()) & vocabulary)).sum()),
                     "target_genes": len(targets[name]), "share_of_target_genes_matched": float(np.isin(targets[name], sorted(vocabulary)).mean()),
                     "check_3_stop": bool(share_features < 0.5), "median_share_of_counts_seen": {}}
            for p in parts:
                if encoder == "scgpt":
                    record = json.loads((scgpt_records / f"{p}.json").read_text())
                    entry["median_share_of_counts_seen"][p] = float(record["median_share_of_counts_seen"])
                else:
                    entry["median_share_of_counts_seen"][p] = stack_share(P[p], vocabulary)
                    print("[e2 coverage] stack %s: %.4f (%.0f s)" % (p, entry["median_share_of_counts_seen"][p], time.time() - started), flush=True)
            out[encoder][name] = entry
    record = {"plan": PLAN, "coverage": out,
              "inputs": {"downloads": file_record(REPO / "bioinf2027" / "results" / "E2" / "downloads.json"),
                         "scgpt_part_records": {p: file_record(scgpt_records / f"{p}.json") for p in P}},
              "code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (sys.modules[__name__], e2feat, e2cells, safe_pickle)},
              "seconds": round(time.time() - started, 1)}
    OUT.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    stops = [(e, n) for e in out for n, v in out[e].items() if isinstance(v, dict) and v["check_3_stop"]]
    print("[e2 coverage] done in %.0f s; check 3 stops: %s" % (time.time() - started, stops or "none"), flush=True)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: make_e2_coverage.py")
    main()
