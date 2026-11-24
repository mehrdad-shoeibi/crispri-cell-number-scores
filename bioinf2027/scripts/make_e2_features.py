"""E2, the arm features from the per-cell embeddings, the size-matched nulls and the PCA basis.

Design: bioinf2027/plans/E2.md, Sections 3, 4 and 16. For an encoder e (scGPT, Stack):
- a VCC row (split, batch, target gene of E1's modelled rows): the mean of e over the row's perturbed cells minus the
  mean over the batch's control cells (batch-local, as Geneformer's z in G5 and G6);
- an external construct: the mean over its eligible perturbed cells minus the mean over the screen's control cells;
- N0ₑ for a VCC row: the mean of e over n control cells of the row's batch drawn with
  `numpy.random.default_rng([20271122, k])` (k the row's position in the sorted row table; without replacement, or
  with it where the batch holds fewer than n), n the row's perturbed cells, minus the batch's control mean. N0 of
  Geneformer the same way from its cached control embeddings (one file per batch).
PCA: the first 768 axes W of the 38,176 VCC control cells' expression on the 2,000 features (normalised to 10,000
counts, log1p, centred), the arm being W^T x on E1b's x; checked for 50 VCC rows and 50 constructs per screen
against the mean of the cells' scores minus the control mean, to 1e-4 of the norm (Section 3).
N0 of PCA-50 (E2.md Section 17, `pca50_null`): W^T over the first 50 axes of the mean of the control values of the
row's n drawn control cells minus the batch's control mean, with the draws of N0 above.
Outputs: $IGA_STORAGE_ROOT/bioinf2027/E2/features/ and bioinf2027/results/E2/features.json.
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

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E2.md, Sections 3, 4 and 16"
N0_SEED = 20271122
PCA_WIDTH = 768
CHECK_ROWS = 50


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "E2" / "features"


def row_table(P: dict):
    import make_e1a_labels as e1labels

    caches = {f"vcc_{s}": e2cells._cache(f"vcc_{s}") for s in e2cells.SPLITS}
    rows = e1labels.vcc_row_table(caches).sort_values(["split", "batch", "target_gene"]).reset_index(drop=True)
    index_of = {}
    for split in e2cells.SPLITS:
        positions = np.unique(np.concatenate(rows.loc[rows["split"] == split, "positions"].tolist()))
        index_of[split] = positions
    rows["part_index"] = [np.searchsorted(index_of[r.split], r.positions) for r in rows.itertuples()]
    return rows, caches


def vectors_of(encoder: str, P: dict) -> dict:
    folder = config.storage_root() / "bioinf2027" / "E2" / "embeddings"
    out = {}
    if encoder == "scgpt":
        for name, part in P.items():
            v = np.load(folder / "scgpt" / f"{name}.npy", allow_pickle=False)
            rows = np.load(folder / "scgpt" / f"{name}_rows.npy", allow_pickle=False)
            if not np.array_equal(rows, part["rows"]) or v.shape[0] != rows.size:
                raise SystemExit("scGPT %s does not cover its part" % name)
            out[name] = v.astype(np.float64)
        return out
    if encoder == "stack":
        filled = {}
        for path in sorted((folder / "stack" / "units").glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                for p in np.unique(z["part"]):
                    sel = z["part"] == p
                    if p not in out:
                        out[p] = np.full((P[p]["rows"].size, z["vectors"].shape[1]), np.nan)
                        filled[p] = np.zeros(P[p]["rows"].size, dtype=bool)
                    out[p][z["index"][sel]] = z["vectors"][sel]
                    filled[p][z["index"][sel]] = True
        if set(out) != set(P) or not all(f.all() for f in filled.values()):
            raise SystemExit("Stack's units do not cover every cell")
        return out
    raise SystemExit("unknown encoder %s" % encoder)


def vcc_features(vectors: dict, P: dict, rows) -> tuple[np.ndarray, np.ndarray, dict]:
    control = P["vcc_control"]
    batch_members = {b: np.flatnonzero(control["group"] == b) for b in sorted(set(control["group"].tolist()))}
    control_mean = {b: vectors["vcc_control"][m].mean(axis=0) for b, m in batch_members.items()}
    arm = np.zeros((len(rows), vectors["vcc_control"].shape[1]))
    null = np.zeros_like(arm)
    replaced = 0
    for k, r in enumerate(rows.itertuples()):
        cells = vectors[f"vcc_perturbed_{r.split}"][r.part_index]
        arm[k] = cells.mean(axis=0) - control_mean[r.batch]
        members = batch_members[r.batch]
        n = len(r.part_index)
        with_replacement = n > members.size
        replaced += int(with_replacement)
        pick = np.random.default_rng([N0_SEED, k]).choice(members, size=n, replace=with_replacement)
        null[k] = vectors["vcc_control"][pick].mean(axis=0) - control_mean[r.batch]
    return arm.astype(np.float32), null.astype(np.float32), {"rows_with_replacement": replaced}


def external_features(vectors: dict, P: dict) -> dict:
    out = {}
    for screen in e2cells.SCREENS:
        cache = e2cells._cache(f"screen_{screen}")
        table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        order = np.argsort(cache["pert_code"], kind="stable")
        bounds = np.searchsorted(cache["pert_code"][order], np.arange(cache["constructs"].size + 1))
        by_construct = {c: i for i, c in enumerate(cache["constructs"].tolist())}
        part_rows = P[f"{screen}_perturbed"]["rows"]
        control = vectors[f"{screen}_control"].mean(axis=0)
        arm = np.full((len(table), control.size), np.nan)
        for t, construct in enumerate(table["construct"].astype(str)):
            i = by_construct[construct]
            cells = cache["pert_rows"][order[bounds[i]: bounds[i + 1]]]
            if cells.size:
                arm[t] = vectors[f"{screen}_perturbed"][np.searchsorted(part_rows, cells)].mean(axis=0) - control
        out[screen] = {"constructs": np.array(table["construct"].astype(str).tolist(), dtype=str), "arm": arm.astype(np.float32)}  # text, not objects
    return out


def geneformer_null(rows) -> np.ndarray:
    folder = config.storage_root() / "MADA" / "cache" / "foundation_models" / "geneformer" / "full_feature_extraction_v2" / "nt_embeddings"
    records = json.loads((REPO / "bioinf2027" / "results" / "D2" / "input_records.json").read_text())
    cells = {}
    for b in sorted(set(rows["batch"])):
        path = folder / f"nt__batch_{b}.npy"
        cells[b] = np.load(path, allow_pickle=False).astype(np.float64)
    out = np.zeros((len(rows), 768))
    for k, r in enumerate(rows.itertuples()):
        members = np.arange(cells[r.batch].shape[0])
        n = len(r.part_index)
        pick = np.random.default_rng([N0_SEED, k]).choice(members, size=n, replace=n > members.size)
        out[k] = cells[r.batch][pick].mean(axis=0) - cells[r.batch].mean(axis=0)
    return out.astype(np.float32), {"store": config.portable(folder), "record": records.get("cached_embedding_stores", {}).get("vcc_control_embeddings")}


def _e4(name: str) -> dict:
    manifest = json.loads((config.storage_root() / "bioinf2027" / "cache" / "E4" / "manifest.json").read_text())
    path = config.storage_root() / "bioinf2027" / "cache" / "E4" / f"{name}.npz"
    if sha256(path) != manifest["cache"][name]["sha256"]:
        raise SystemExit("the E4 cache's %s does not match its manifest" % name)
    with np.load(path, allow_pickle=False) as store:
        return {k: store[k] for k in store.files}


def _values(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    totals = totals.astype(np.float64)
    return np.log1p(counts.astype(np.float64) / np.where(totals < 1, 1, totals)[:, None] * 1e4)


def pca_basis(P: dict, rows) -> dict:
    train = _e4("vcc_train")
    lab_train = e2cells._cache("vcc_train")
    control_values = _values(train["nt_counts"], lab_train["nt_total"])
    mean = control_values.mean(axis=0)
    centred = control_values - mean
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred / (control_values.shape[0] - 1))
    order = np.argsort(eigenvalues)[::-1][:PCA_WIDTH]
    W = eigenvectors[:, order]
    return {"W": W.astype(np.float32), "mean": mean.astype(np.float32), "explained": eigenvalues[order] / eigenvalues.sum(),
            "check": pca_check(W, rows, control_values, lab_train)}


def pca_check(W: np.ndarray, rows, control_values: np.ndarray, lab_train: dict) -> dict:
    """W^T x against the mean of the cells' scores minus the control mean, for 50 VCC rows (G6 test) and 50 constructs per screen."""
    from bioinf2027 import e1cells, e1inputs

    out = {}
    vrows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    cell = e1cells.build_cell("G6", vrows, means)
    test = cell.vcc[cell.split("test")].reset_index()
    pick = np.sort(np.random.default_rng(e2cells.PROBE_SEED).choice(len(test), size=CHECK_ROWS, replace=False))
    key = {f"{r.split}|{r.batch}|{r.target_gene}": k for k, r in enumerate(rows.itertuples())}
    caches = {s: (_e4(f"vcc_{s}"), e2cells._cache(f"vcc_{s}")) for s in e2cells.SPLITS}
    worst = 0.0
    for j in pick:
        r = test.iloc[j]
        k = key[f"{r['split']}|{r['batch']}|{r['target_gene']}"]
        e4, lab = caches[r["split"]]
        positions = rows.iloc[k]["positions"]
        cells = _values(e4["pert_counts"][positions], lab["pert_total"][positions]).mean(axis=0)
        batch = lab_train["nt_batch"].astype(str) == str(r["batch"])
        delta = cells - control_values[batch].mean(axis=0)
        x = cell.x[int(r["index"])].astype(np.float64)
        a, b = W.T @ delta, W.T @ x
        worst = max(worst, float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    out["vcc_G6_test_rows"] = {"rows": CHECK_ROWS, "largest_relative_difference": worst}
    for screen in e2cells.SCREENS:
        table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        inputs = e1inputs.load_screen_inputs(screen, table)
        e4, lab = _e4(f"screen_{screen}"), e2cells._cache(f"screen_{screen}")
        if not np.array_equal(e4["pert_rows"], lab["pert_rows"]):
            raise SystemExit("the E4 and E1a caches list other %s cells" % screen)
        order = np.argsort(lab["pert_code"], kind="stable")
        bounds = np.searchsorted(lab["pert_code"][order], np.arange(lab["constructs"].size + 1))
        by_construct = {c: i for i, c in enumerate(lab["constructs"].tolist())}
        control = _values(e4["nt_counts"], lab["nt_total"]).mean(axis=0)
        picks = np.sort(np.random.default_rng(e2cells.PROBE_SEED).choice(len(table), size=CHECK_ROWS, replace=False))
        worst = 0.0
        for t in picks:
            i = by_construct[str(table["construct"].iloc[t])]
            cells = order[bounds[i]: bounds[i + 1]]
            delta = _values(e4["pert_counts"][cells], lab["pert_total"][cells]).mean(axis=0) - control
            x = inputs.x["full_pass"][t].astype(np.float64)
            a, b = W.T @ delta, W.T @ x
            worst = max(worst, float(np.linalg.norm(a - b) / np.linalg.norm(b)))
        out[screen] = {"constructs": CHECK_ROWS, "largest_relative_difference": worst}
    out["tolerance"] = 1e-4
    out["passed"] = all(v["largest_relative_difference"] <= 1e-4 for k, v in out.items() if isinstance(v, dict))
    return out


def pca50_null(P: dict, rows) -> tuple[np.ndarray, dict]:
    """N0 of PCA-50: the draws of `vcc_features` (same seed rule, same control part) on the cells' PCA-50 scores."""
    train, lab = _e4("vcc_train"), e2cells._cache("vcc_train")
    if not (np.array_equal(train["nt_rows"], lab["nt_rows"]) and np.array_equal(lab["nt_rows"], P["vcc_control"]["rows"])):
        raise SystemExit("the E4 cache, the E1a cache and the control part list other VCC control cells")
    recorded = json.loads((REPO / "bioinf2027" / "results" / "E2" / "features.json").read_text())["outputs"]["pca"]
    basis_path = out_dir() / "pca_basis.npz"
    if sha256(basis_path) != recorded["sha256"]:
        raise SystemExit("pca_basis.npz does not match features.json")
    with np.load(basis_path, allow_pickle=False) as store:
        W = store["W"][:, :50].astype(np.float64)
    scores = _values(train["nt_counts"], lab["nt_total"]) @ W
    control = P["vcc_control"]
    batch_members = {b: np.flatnonzero(control["group"] == b) for b in sorted(set(control["group"].tolist()))}
    control_mean = {b: scores[m].mean(axis=0) for b, m in batch_members.items()}
    null = np.zeros((len(rows), W.shape[1]))
    replaced = 0
    for k, r in enumerate(rows.itertuples()):
        members = batch_members[r.batch]
        n = len(r.part_index)
        with_replacement = n > members.size
        replaced += int(with_replacement)
        pick = np.random.default_rng([N0_SEED, k]).choice(members, size=n, replace=with_replacement)
        null[k] = scores[pick].mean(axis=0) - control_mean[r.batch]
    return null.astype(np.float32), {"rows_with_replacement": replaced, "basis": file_record(basis_path), "width": int(W.shape[1])}


def main(encoders: list[str]) -> None:
    started = time.time()
    P = e2cells.parts()
    rows, _ = row_table(P)
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    record = {"plan": PLAN, "rows": int(len(rows)), "outputs": {}, "n0_seed": N0_SEED}
    keys = np.array([f"{r.split}|{r.batch}|{r.target_gene}" for r in rows.itertuples()])
    for encoder in encoders:
        if encoder == "geneformer_null":
            null, info = geneformer_null(rows)
            path = folder / "geneformer_null.npz"
            np.savez(path, keys=keys, null=null)
            record["outputs"]["geneformer_null"] = {**file_record(path), **{"store": info["store"]}}
            continue
        if encoder == "pca50_null":
            null, info = pca50_null(P, rows)
            path = folder / "pca50_null.npz"
            np.savez(path, keys=keys, null=null)
            record["outputs"]["pca50_null"] = {**file_record(path), **info}
            print("[e2 features] PCA-50 N0: %d rows, %d with replacement" % (len(rows), info["rows_with_replacement"]), flush=True)
            continue
        if encoder == "pca":
            basis = pca_basis(P, rows)
            path = folder / "pca_basis.npz"
            np.savez(path, W=basis["W"], mean=basis["mean"], explained=basis["explained"])
            record["outputs"]["pca"] = {**file_record(path), "explained_by_50": float(basis["explained"][:50].sum()),
                                        "explained_by_768": float(basis["explained"].sum()), "check": basis["check"]}
            print("[e2 features] PCA check: %s" % json.dumps(basis["check"]), flush=True)
            continue
        vectors = vectors_of(encoder, P)
        arm, null, info = vcc_features(vectors, P, rows)
        external = external_features(vectors, P)
        path = folder / f"{encoder}.npz"
        np.savez(path, keys=keys, vcc=arm, vcc_null=null, **{f"{s}_constructs": external[s]["constructs"] for s in external},
                 **{f"{s}_arm": external[s]["arm"] for s in external})
        record["outputs"][encoder] = {**file_record(path), "width": int(arm.shape[1]), **info,
                                      "constructs_without_cells": {s: int(np.isnan(external[s]["arm"]).any(axis=1).sum()) for s in external}}
        print("[e2 features] %s: %d rows, width %d" % (encoder, len(rows), arm.shape[1]), flush=True)
    record["seconds"] = round(time.time() - started, 1)
    target = REPO / "bioinf2027" / "results" / "E2" / "features.json"
    previous = json.loads(target.read_text()) if target.exists() else {"outputs": {}}
    previous["outputs"].update(record["outputs"])
    record["outputs"] = previous["outputs"]
    target.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    wanted = sys.argv[1:]
    if not wanted or any(w not in ("scgpt", "stack", "geneformer_null", "pca", "pca50_null") for w in wanted):
        raise SystemExit("usage: make_e2_features.py {scgpt|stack|geneformer_null|pca|pca50_null} ...")
    main(wanted)
