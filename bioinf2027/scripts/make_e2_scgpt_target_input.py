"""E2, scGPT: the share of perturbed cells whose target gene is among the 1,199 genes given to scGPT, per screen
(gate G3, item 5 (c), the author's answer of 2026-11-16, bioinf2027/GATE.md; added after the results review).

scGPT was given, per cell, a random 1,199 of the cell's expressed vocabulary genes (make_e2_embed_scgpt.cell_input,
max_length 1,200 with the CLS token; bioinf2027/results/E2/coverage_scgpt_input.json). For every perturbed cell of
E2's parts (the cells of E1's modelled VCC rows in the three VCC files, and each Replogle screen's eligible perturbed
cells) this script rebuilds that input as make_e2_scgpt_input_share.py does (the same reading, vocabulary match by
exact symbol, generator per cell from "E2 | scGPT | <file> | <row>" and cell_input) and records whether the cell's
target gene is (1) in scGPT's vocabulary, (2) expressed in the cell (a stored count above zero), and (3) among the
genes given to the model. The target gene of a VCC cell is its pert_target in the E1a label cache, matched to the
file's gene symbols; of a Replogle cell, the column of its construct's target gene in the E1a label cache
(construct_col), as E1a's label reads it.

Two checks first, or nothing is written (hard limit 9):
- Across environments (the results review, round 2, point 3): the embedding ran under numpy 2.2.6 (scGPT's
  environment), this script under the pipeline's numpy. For E2's 2,000 probe cells (e2cells.probe_cells) the kept
  genes and binned values are computed by e2_scgpt_draws.py in both environments and must be equal, cell by cell.
  scGPT's environment is the one beside the pipeline environment, its folder named with "-scgpt" (its location is
  not written in any committed file; its numpy and Python versions are recorded).
- Per part, the cell count, the genes matched and the median number of expressed vocabulary genes equal the part's
  embedding record.

Output: bioinf2027/results/E2/supplement/scgpt_target_input.json (per part and per screen). Pipeline environment;
reads only (the probe cells' inputs and outputs of the cross-environment check under
$IGA_STORAGE_ROOT/bioinf2027/scratch/E2_scgpt_draws/). --workers N (default 4) reads parts in parallel and changes
no value.
"""
from __future__ import annotations

import json
import multiprocessing
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import e2_scgpt_draws as draws_script  # noqa: E402
import e2cells  # noqa: E402
import make_e2_embed_scgpt as emb  # noqa: E402
import make_e2_scgpt_input_share as share  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "supplement" / "scgpt_target_input.json"
SCRATCH = config.storage_root() / "bioinf2027" / "scratch" / "E2_scgpt_draws"
SCREENS = {"VCC": ("vcc_perturbed_train", "vcc_perturbed_val", "vcc_perturbed_test"), "RPE1": ("RPE1_perturbed",),
           "K562-essential": ("K562-essential_perturbed",)}
ADDED = "added after the results review (gate G3, item 5 (c); bioinf2027/GATE.md)"
WORKERS = 4
DATA: dict = {}


class Stop(RuntimeError):
    pass


def vocabulary_columns(part: dict) -> tuple[np.ndarray, np.ndarray]:
    """keep (the file's columns in scGPT's vocabulary, as embed() builds it) and each column's position in keep (-1 if none)."""
    vocab = DATA["vocab"]
    keep = np.flatnonzero(np.array([s in vocab for s in part["symbols"].tolist()]))
    where = np.full(part["symbols"].size, -1, dtype=np.int64)
    where[keep] = np.arange(keep.size)
    return keep, where


def target_columns(name: str, part: dict) -> tuple[np.ndarray, dict]:
    """The file column of each cell's target gene (-1 if the file lists no such gene), in part order."""
    if name.startswith("vcc_perturbed_"):
        cache = e2cells._cache("vcc_" + name.rsplit("_", 1)[1])
        index = np.searchsorted(cache["pert_rows"], part["rows"])
        if not (np.all(np.diff(cache["pert_rows"]) > 0) and np.array_equal(cache["pert_rows"][index], part["rows"])):
            raise Stop("%s: the cells do not match the label cache" % name)
        column = {s: i for i, s in reversed(list(enumerate(part["symbols"].tolist())))}  # the first column of a symbol
        targets = cache["pert_target"][index].astype(str)
        cols = np.array([column.get(t, -1) for t in targets.tolist()], dtype=np.int64)
        return cols, {"targets_not_in_the_file": int((cols < 0).sum())}
    screen = name.rsplit("_", 1)[0]
    cache = e2cells._cache(f"screen_{screen}")
    order = np.argsort(cache["pert_rows"], kind="stable")
    if not np.array_equal(cache["pert_rows"][order], part["rows"]):
        raise Stop("%s: the cells do not match the label cache" % name)
    codes = cache["pert_code"][order]
    cols = cache["construct_col"][codes].astype(np.int64)
    named = np.array([c.split("_")[1] for c in cache["constructs"].astype(str).tolist()])[codes]
    return cols, {"targets_not_in_the_file": int((cols < 0).sum()),
                  "cells_whose_target_column_symbol_differs_from_the_construct_name": int((part["symbols"][cols] != named).sum())}


def part_targets(name: str) -> dict:
    part, pad_value = DATA["parts"][name], DATA["pad_value"]
    keep, where = vocabulary_columns(part)
    cols, notes = target_columns(name, part)
    n = part["rows"].size
    in_vocab = np.zeros(n, dtype=bool)
    expressed, given, n_expressed = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=np.int64)
    started = time.time()
    for idx, block in e2cells.read_rows(part, np.arange(n)):
        sub = block[:, keep].tocsr()
        for j, i in enumerate(idx.tolist()):
            lo, hi = sub.indptr[j], sub.indptr[j + 1]
            values, local = sub.data[lo:hi], sub.indices[lo:hi]
            n_expressed[i] = hi - lo
            target = where[cols[i]] if cols[i] >= 0 else -1
            in_vocab[i] = target >= 0
            if target < 0:
                continue
            at = np.flatnonzero(local == target)
            expressed[i] = bool(at.size and values[at[0]] > 0)
            rng = np.random.default_rng(e2cells.seed_of(f"E2 | scGPT | {part['file_name']} | {int(part['rows'][i])}"))
            kept = share.kept_positions(values, rng, pad_value)
            given[i] = bool(at.size and np.isin(at[0], kept))
    print("[e2 scgpt target] %s: %d cells, %.0f s" % (name, n, time.time() - started), flush=True)
    return {"name": name, "in_vocab": in_vocab, "expressed": expressed, "given": given, "n_expressed": n_expressed,
            "genes_matched": int(keep.size), "notes": notes}


def shares(in_vocab: np.ndarray, expressed: np.ndarray, given: np.ndarray) -> dict:
    n = int(in_vocab.size)
    return {"perturbed_cells": n, "share_target_in_scgpt_vocabulary": float(in_vocab.mean()),
            "share_target_expressed_in_vocabulary": float(expressed.mean()),
            "share_target_among_the_genes_given_to_scgpt": float(given.mean()),
            "share_given_among_cells_with_the_target_expressed": float(given[expressed].mean()) if expressed.any() else float("nan"),
            "cells_target_given": int(given.sum()), "cells_target_expressed": int(expressed.sum())}


# --------------------------------------------------------------------------- the check across environments


def probe_inputs() -> Path:
    """The 2,000 probe cells' expressed vocabulary-gene values and generator seeds, for e2_scgpt_draws.py."""
    cells = e2cells.probe_cells(DATA["parts"])
    values, offsets, seeds = [], [0], []
    by_part: dict = {}
    for k, (name, i) in enumerate(cells):
        by_part.setdefault(name, []).append((k, i))
    found: dict = {}
    for name, members in by_part.items():
        part = DATA["parts"][name]
        keep, _ = vocabulary_columns(part)
        wanted = np.array([i for _, i in members])
        position = {i: k for k, i in members}
        for idx, block in e2cells.read_rows(part, wanted):
            sub = block[:, keep].tocsr()
            for j, i in enumerate(idx.tolist()):
                found[position[i]] = (sub.data[sub.indptr[j]: sub.indptr[j + 1]].astype(np.float32),
                                      e2cells.seed_of(f"E2 | scGPT | {part['file_name']} | {int(part['rows'][i])}"))
    for k in range(len(cells)):
        v, s = found[k]
        values.append(v)
        offsets.append(offsets[-1] + v.size)
        seeds.append(s)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "probe_inputs.npz"
    np.savez(path, values=np.concatenate(values), offsets=np.asarray(offsets, dtype=np.int64), seeds=np.asarray(seeds, dtype=np.uint64),
             pad_value=np.array(DATA["pad_value"], dtype=np.float64))
    return path


def scgpt_python() -> Path:
    """The interpreter of scGPT's environment, beside the pipeline environment (its folder name with "-scgpt")."""
    candidate = Path(sys.prefix).with_name(Path(sys.prefix).name + "-scgpt") / "bin" / "python"
    if not candidate.exists():
        raise Stop("scGPT's environment is not beside the pipeline environment")
    return candidate


def across_environments() -> dict:
    source = probe_inputs()
    outputs = {}
    for env, python in (("pipeline", Path(sys.executable)), ("scgpt", scgpt_python())):
        target = SCRATCH / f"probe_draws_{env}.npz"
        subprocess.run([str(python), "-B", str(Path(draws_script.__file__).resolve()), str(source), str(target)], check=True)
        with np.load(target, allow_pickle=False) as store:
            outputs[env] = {k: store[k] for k in store.files}
    a, b = outputs["pipeline"], outputs["scgpt"]
    equal = {k: bool(np.array_equal(a[k], b[k])) for k in ("kept", "binned", "bounds")}
    with np.load(source, allow_pickle=False) as store:
        cells = int(store["seeds"].size)
        above = int((np.diff(store["offsets"]) > emb.MAX_LENGTH - 1).sum())
    return {"probe_cells": cells, "probe_cells_above_1199_expressed": above, "equal": equal, "passed": all(equal.values()),
            "environments": {env: {"numpy": str(o["numpy"]), "python": str(o["python"])} for env, o in outputs.items()},
            "inputs": file_record(source), "outputs": {env: file_record(SCRATCH / f"probe_draws_{env}.npz") for env in outputs}}


# --------------------------------------------------------------------------- main


def main() -> None:
    started = time.time()
    DATA["parts"] = e2cells.parts()
    DATA["vocab"] = set(json.loads(emb.downloaded("vocab.json").read_text()))
    DATA["pad_value"] = json.loads(emb.downloaded("args.json").read_text())["pad_value"]
    check = across_environments()
    print("[e2 scgpt target] across environments: %s (%s), %.0f s" % ("equal" if check["passed"] else "DIFFERENT", check["environments"], time.time() - started), flush=True)
    if not check["passed"]:
        raise Stop("the inputs rebuilt in the two environments differ: %s" % json.dumps(check["equal"]))
    names = [p for parts in SCREENS.values() for p in parts]
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        results = {r["name"]: r for r in pool.map(part_targets, names)}
    out, problems = {"parts": {}, "screens": {}}, []
    for name in names:
        r, record = results[name], json.loads((share.RECORDS / f"{name}.json").read_text())
        same = (record["cells"] == r["in_vocab"].size and record["genes_matched"] == r["genes_matched"]
                and record["median_expressed_genes"] == float(np.median(r["n_expressed"])))
        out["parts"][name] = {**shares(r["in_vocab"], r["expressed"], r["given"]), **r["notes"], "equal_to_the_part_record": same,
                              "part_record": {k: record[k] for k in ("cells", "genes_matched", "median_expressed_genes")}}
        if not same:
            problems.append(name)
    if problems:
        raise Stop("the rebuilt parts differ from their embedding records: %s" % problems)
    for screen, parts in SCREENS.items():
        out["screens"][screen] = shares(*(np.concatenate([results[p][k] for p in parts]) for k in ("in_vocab", "expressed", "given")))
    record = {"item": "E2, supplement (c): scGPT's input and the target gene (gate G3, item 5 (c); bioinf2027/GATE.md)", "status": ADDED,
              "rule": "per perturbed cell: the target gene in scGPT's vocabulary (exact symbol); expressed (stored count above 0); among "
                      "the at most 1,199 genes given to the model (make_e2_embed_scgpt.cell_input, the generator per cell)",
              "check_across_environments": check, **out,
              "inputs": {"part_records": {p: file_record(share.RECORDS / f"{p}.json") for p in names},
                         "downloads": file_record(REPO / "bioinf2027" / "results" / "E2" / "downloads.json")},
              "environment": {"numpy": np.__version__, "python": platform.python_version()},
              "code": {config.portable(Path(m.__file__).resolve()): sha256(Path(m.__file__)) for m in (sys.modules[__name__], draws_script, share, emb, e2cells)},
              "seconds": round(time.time() - started, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 scgpt target] done in %.0f s; share of perturbed cells with the target among scGPT's input: %s" % (time.time() - started, {
        k: round(v["share_target_among_the_genes_given_to_scgpt"], 4) for k, v in out["screens"].items()}), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] == ["--workers"] and len(arguments) == 2 and arguments[1].isdigit():
        WORKERS = int(arguments[1])
    elif arguments:
        raise SystemExit("usage: make_e2_scgpt_target_input.py [--workers N]")
    main()
