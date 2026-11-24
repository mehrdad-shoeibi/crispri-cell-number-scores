"""D2, aggregation: z- and the noise-matched z~ of every VCC row and external construct, with the checks of the drop.

Design: bioinf2027/plans/D2.md, Sections 3, 10, 11 and 13. Inputs: the token-drop embeddings of
bioinf2027/scripts/make_d2_embeddings.py (every unit refused unless its digest matches its record), the
cell lists (bioinf2027/results/D2/units.json), the cached embeddings of the same cells, the encoder's
per-row means (VCC) and the stored external z.

- delta_g, per target gene: the mean over its sampled control cells of the embedding with g's token
  removed minus the same cell's cached embedding; on VCC also per batch (10 cells), and in G1 and G2
  pooled as sum_b (n_b / sum n_b) delta_g,b (Section 11.2, item 2).
- z- of a row or construct: the mean over its perturbed cells of the dropped embeddings minus (the cached
  control mean of its reference plus delta_g); the reference is the row's batch in G5 and G6, the pooled
  controls in G1 and G2 and on the screens. A target without a token keeps z (Section 10.2, item 5).
- z~ = z + (delta_g^A - delta_g^B) / 2 from the two halves of the delta sample (Section 10.1, item 3).
- Reported: h_c, h_p, the share of truncated inputs, ||z - z-|| / ||z|| and ||z - z~|| / ||z||, the precision of
  delta_g (bootstrap over its cells, 1,000 replicates, seed 20271107) against the between-gene sd of z,
  delta_g per Flex run and the batch deviations, the identity check's mean error against ||z|| and
  ||z - z-||, and the uninformative-cell rule (Section 10.1, item 1).

Outputs: $IGA_STORAGE_ROOT/bioinf2027/D2/arms/arms.npz; bioinf2027/results/D2/aggregate.json and
manipulation.csv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, e1inputs  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_d2_embeddings as d2emb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "D2"
PLAN = "bioinf2027/plans/D2.md, Sections 3, 10, 11 and 13"
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20271107
PRECISION_LIMIT = 0.1
SCREENS = e1inputs.SCREENS
PARTS = {"VCC": ("vcc_perturbed", "vcc_delta"), "RPE1": ("rpe1_perturbed", "rpe1_delta"), "K562-essential": ("k562_perturbed", "k562_delta")}


def load_part(part: str) -> pd.DataFrame:
    """The part's cell list joined to its dropped embeddings and per-cell records, every unit checked."""
    units = d2emb.load_parts([part])[part]
    folder = d2emb.d2_root() / "embeddings" / part
    frames, embeddings = [], []
    for unit, cells in units.groupby("unit", sort=True):
        record_path = folder / f"unit_{int(unit):05d}.json"
        array_path = folder / f"unit_{int(unit):05d}.npz"
        if not record_path.exists():
            raise RuntimeError("%s unit %d has not been embedded" % (part, unit))
        record = json.loads(record_path.read_text())
        if sha256(array_path) != record["file"]["sha256"]:
            raise RuntimeError("%s unit %d differs from its record" % (part, unit))
        with np.load(array_path, allow_pickle=False) as store:
            keys = store["key"].astype(str)
            frame = pd.DataFrame({"key": keys, "target_in_input": store["target_in_input"], "truncated": store["truncated"]})
            embeddings.append(store["embedding"])
        cells = cells.assign(key=d2emb.cell_keys(cells))
        merged = cells.merge(frame, on="key", how="left", validate="one_to_one")
        if merged["target_in_input"].isna().any():
            raise RuntimeError("%s unit %d lacks a cell of its list" % (part, unit))
        order = {k: i for i, k in enumerate(keys.tolist())}
        merged["_row"] = [order[k] for k in merged["key"]]
        merged["_unit"] = int(unit)
        frames.append(merged)
    frame = pd.concat(frames, ignore_index=True)
    stack = {int(u): e for u, e in zip(sorted(units["unit"].unique()), embeddings)}
    dropped = np.stack([stack[u][r] for u, r in zip(frame["_unit"], frame["_row"])]).astype(np.float64)
    return frame, dropped


def cached(frame: pd.DataFrame) -> np.ndarray:
    """The cached embedding of every cell of a frame (store, cached file and row)."""
    out = np.empty((len(frame), 768), dtype=np.float64)
    loaded: dict = {}
    for (store, name), group in frame.groupby(["store", "cached_file"]):
        path = d2emb.cached_path(store, name)
        if path not in loaded:
            loaded[path] = np.load(path, mmap_mode="r")
        out[group.index.to_numpy()] = np.asarray(loaded[path][group["cached_row"].to_numpy(dtype=np.int64)], dtype=np.float64)
    return out


def bootstrap_se(effects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap standard error of the mean over cells, per dimension."""
    n = effects.shape[0]
    means = np.empty((BOOTSTRAP, effects.shape[1]))
    for b in range(BOOTSTRAP):
        means[b] = effects[rng.integers(0, n, n)].mean(axis=0)
    return means.std(axis=0, ddof=1)


def main() -> None:
    started = time.time()
    rows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    g6 = e1cells.load_tables("G6")["vcc"]
    report: dict = {"plan": PLAN, "inputs": {"units": file_record(OUT / "units.json")}}
    arms: dict = {}
    manipulation = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    precision: dict = {}
    identity = json.loads((OUT / f"{d2emb.IDENTITY_OF_RECORD}.json").read_text())["identity"]
    mean_error_norm = identity["mean_error_vector_norm"]

    # ---------------------------------------------------------------- VCC
    pert, dropped = load_part("vcc_perturbed")
    pert_effect = dropped - cached(pert)
    delta_frame, delta_dropped = load_part("vcc_delta")
    delta_effect = delta_dropped - cached(delta_frame)
    counts = means.counts
    batches = sorted(counts)
    weight = np.array([counts[b] for b in batches], dtype=np.float64)
    weight = weight / weight.sum()
    genes = sorted(set(delta_frame["row_gene"]))
    delta = {}
    for gene in genes:
        part = delta_frame["row_gene"] == gene
        effects = delta_effect[part.to_numpy()]
        frame = delta_frame[part]
        per_batch = np.stack([effects[(frame["batch"] == b).to_numpy()].mean(axis=0) for b in batches])
        halves = {}
        for half in ("A", "B"):
            sel = (frame["half"] == half).to_numpy()
            hb = np.stack([effects[sel & (frame["batch"] == b).to_numpy()].mean(axis=0) for b in batches])
            halves[half] = {"equal": effects[sel].mean(axis=0), "pooled": (weight[:, None] * hb).sum(axis=0)}
        delta[gene] = {"equal": effects.mean(axis=0), "pooled": (weight[:, None] * per_batch).sum(axis=0), "per_batch": per_batch,
                       "halves": halves, "se": bootstrap_se(effects, rng), "h_c": float(frame["target_in_input"].astype(bool).mean()),
                       "no_token": bool((frame["drop_token"] < 0).all())}
    keys = list(zip(g6["split"], g6["batch"], g6["target_gene"]))
    index = rows.index(g6)
    z_batch = rows.z[index].astype(np.float64)
    pooled_nt = means.pooled_embedding(rows.batch_mu_nt)
    z_pooled = rows.mu_pert[index].astype(np.float64) - pooled_nt[None, :]
    groups = pert.groupby(["row_split", "row_batch", "row_gene"]).indices
    out = {name: np.empty((len(keys), 768)) for name in ("z_minus_batch", "z_minus_pooled", "z_tilde_batch", "z_tilde_pooled")}
    for i, key in enumerate(keys):
        members = groups[key]
        gene = key[2]
        d = delta[gene]
        shift = pert_effect[members].mean(axis=0)
        if d["no_token"]:
            out["z_minus_batch"][i], out["z_minus_pooled"][i] = z_batch[i], z_pooled[i]
        else:
            # mean of the dropped perturbed embeddings minus (the cached reference mean plus delta_g)
            out["z_minus_batch"][i] = z_batch[i] + shift - d["equal"]
            out["z_minus_pooled"][i] = z_pooled[i] + shift - d["pooled"]
        out["z_tilde_batch"][i] = z_batch[i] + (d["halves"]["A"]["equal"] - d["halves"]["B"]["equal"]) / 2.0
        out["z_tilde_pooled"][i] = z_pooled[i] + (d["halves"]["A"]["pooled"] - d["halves"]["B"]["pooled"]) / 2.0
        h_p = float(pert.iloc[members]["target_in_input"].astype(bool).mean())
        manipulation.append({"side": "VCC", "split": key[0], "batch": key[1], "gene": gene, "h_p": h_p, "h_c": d["h_c"],
                             "truncated_share": float(pert.iloc[members]["truncated"].astype(bool).mean()),
                             "relative_change_batch": float(np.linalg.norm(z_batch[i] - out["z_minus_batch"][i]) / np.linalg.norm(z_batch[i])),
                             "relative_change_noise_batch": float(np.linalg.norm(z_batch[i] - out["z_tilde_batch"][i]) / np.linalg.norm(z_batch[i])),
                             "relative_change_pooled": float(np.linalg.norm(z_pooled[i] - out["z_minus_pooled"][i]) / np.linalg.norm(z_pooled[i]))})
    arms["VCC_keys"] = np.array(["|".join(k) for k in keys])
    for name, values in out.items():
        arms[f"VCC_{name}"] = values.astype(np.float32)
    gene_of_row = np.array([k[2] for k in keys])
    gene_means = np.stack([z_batch[gene_of_row == g].mean(axis=0) for g in genes])
    sd = gene_means.std(axis=0, ddof=1)
    ratio = np.stack([delta[g]["se"] for g in genes]) / sd[None, :]
    precision["VCC"] = {"median_ratio": float(np.median(ratio)), "limit": PRECISION_LIMIT, "passed": bool(np.median(ratio) <= PRECISION_LIMIT),
                        "genes_with_median_ratio_above_the_limit": [g for g, r in zip(genes, np.median(ratio, axis=1)) if r > PRECISION_LIMIT]}
    runs = sorted({b.split("_")[1] for b in batches})
    per_run = []
    for gene in genes:
        d = delta[gene]
        change = np.median([m["relative_change_batch"] for m in manipulation if m["gene"] == gene]) if any(m["gene"] == gene for m in manipulation) else float("nan")
        rows_change = np.median([np.linalg.norm(z_batch[i] - out["z_minus_batch"][i]) for i in np.flatnonzero(gene_of_row == gene)])
        entry = {"gene": gene, "delta_norm": float(np.linalg.norm(d["equal"])), "median_row_norm_of_z_minus_z_minus": float(rows_change),
                 "batch_deviation_norm_median": float(np.median(np.linalg.norm(d["per_batch"] - d["equal"][None, :], axis=1)))}
        for run in runs:
            sel = np.array([b.split("_")[1] == run for b in batches])
            entry[f"delta_norm_flex_run_{run}"] = float(np.linalg.norm(d["per_batch"][sel].mean(axis=0)))
        entry["median_relative_change"] = float(change)
        per_run.append(entry)
    pd.DataFrame(per_run).to_csv(OUT / "delta_per_flex_run.csv", index=False)

    # ---------------------------------------------------------------- external screens
    for screen in SCREENS:
        table = e1cells.load_tables("G1")[screen]
        inputs = e1inputs.load_screen_inputs(screen, table)
        pert, dropped = load_part(PARTS[screen][0])
        pert_effect = dropped - cached(pert)
        delta_frame, delta_dropped = load_part(PARTS[screen][1])
        delta_effect = delta_dropped - cached(delta_frame)
        delta = {}
        for gene, frame in delta_frame.groupby("drop_ensembl"):
            effects = delta_effect[frame.index.to_numpy()]
            a = effects[(frame["half"] == "A").to_numpy()].mean(axis=0)
            b = effects[(frame["half"] == "B").to_numpy()].mean(axis=0)
            delta[gene] = {"mean": effects.mean(axis=0), "A": a, "B": b, "se": bootstrap_se(effects, rng),
                           "h_c": float(frame["target_in_input"].astype(bool).mean()), "no_token": bool((frame["drop_token"] < 0).all())}
        groups = pert.groupby("construct").indices
        z = inputs.z.astype(np.float64)
        z_minus = np.empty_like(z)
        z_tilde = np.empty_like(z)
        for i, (construct, gene) in enumerate(zip(table["construct"].astype(str), table["target_ensg"].astype(str))):
            members = groups[construct]
            d = delta[gene]
            if d["no_token"] or (pert.iloc[members]["drop_token"] < 0).all():
                z_minus[i] = z[i]
            else:
                z_minus[i] = z[i] + pert_effect[members].mean(axis=0) - d["mean"]
            z_tilde[i] = z[i] + (d["A"] - d["B"]) / 2.0
            manipulation.append({"side": screen, "construct": construct, "gene": gene,
                                 "h_p": float(pert.iloc[members]["target_in_input"].astype(bool).mean()), "h_c": d["h_c"],
                                 "truncated_share": float(pert.iloc[members]["truncated"].astype(bool).mean()),
                                 "relative_change": float(np.linalg.norm(z[i] - z_minus[i]) / np.linalg.norm(z[i])),
                                 "relative_change_noise": float(np.linalg.norm(z[i] - z_tilde[i]) / np.linalg.norm(z[i]))})
        # a plain string array: pandas 3 returns object arrays, which np.savez would pickle
        arms[f"{screen}_constructs"] = np.array(table["construct"].astype(str).tolist(), dtype=str)
        arms[f"{screen}_z_minus"] = z_minus.astype(np.float32)
        arms[f"{screen}_z_tilde"] = z_tilde.astype(np.float32)
        gene_list = sorted(delta)
        gm = pd.DataFrame(z).groupby(table["target_ensg"].astype(str).to_numpy()).mean()
        sd = gm.to_numpy().std(axis=0, ddof=1)
        ratio = np.stack([delta[g]["se"] for g in gene_list]) / sd[None, :]
        precision[screen] = {"median_ratio": float(np.median(ratio)), "limit": PRECISION_LIMIT, "passed": bool(np.median(ratio) <= PRECISION_LIMIT)}

    frame = pd.DataFrame(manipulation)
    frame.to_csv(OUT / "manipulation.csv", index=False)
    uninformative = {}
    for cell in ("G5", "G6", "G1", "G2"):
        cell_rows = e1cells.load_tables(cell)["vcc"]
        keyset = set(zip(cell_rows["split"], cell_rows["batch"], cell_rows["target_gene"]))
        part = frame[(frame["side"] == "VCC")]
        part = part[[k in keyset for k in zip(part["split"], part["batch"], part["gene"])]]
        change = part["relative_change_batch" if cell in ("G5", "G6") else "relative_change_pooled"]
        low_tokens = float(((part["h_c"] < 0.05) & (part["h_p"] < 0.05)).mean())
        uninformative[cell] = {"rows": int(len(part)), "share_with_h_c_and_h_p_below_0.05": low_tokens,
                               "median_relative_change": float(change.median()),
                               "uninformative": bool(low_tokens > 0.5 or change.median() < 0.01)}
    folder = d2emb.d2_root() / "arms"
    folder.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(v).dtype == object for v in arms.values()):
        raise RuntimeError("an arm would be saved as an object array")
    np.savez(folder / "arms.npz", **arms)
    vcc = frame[frame["side"] == "VCC"]
    report.update({
        "arms": file_record(folder / "arms.npz"),
        "manipulation": file_record(OUT / "manipulation.csv"),
        "delta_per_flex_run": file_record(OUT / "delta_per_flex_run.csv"),
        "precision_of_delta": precision,
        "uninformative_cells": uninformative,
        "identity_mean_error_norm": mean_error_norm,
        "median_norm_of_z": float(np.median(np.linalg.norm(z_batch, axis=1))),
        "median_norm_of_z_minus_z_minus": float(np.median(np.linalg.norm(z_batch - out["z_minus_batch"], axis=1))),
        "truncated_share": {"VCC": float(vcc["truncated_share"].mean()), **{s: float(frame[frame["side"] == s]["truncated_share"].mean()) for s in SCREENS}},
        "settings": {"bootstrap": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "precision_limit": PRECISION_LIMIT},
        "seconds": round(time.time() - started, 1),
    })
    (OUT / "aggregate.json").write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
    print("[d2 aggregate] done in %.0f s; precision %s; uninformative %s" % (time.time() - started, precision, uninformative), flush=True)


if __name__ == "__main__":
    main()
