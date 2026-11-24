"""D1, source side: descriptive statistics of the current VCC label against the target's own delta.

For each VCC split this reports how often the target gene is one of the 2,000
selected features, how often the target's own coordinate is the largest
absolute coordinate of the row (over the rows whose target is a feature and
over all rows), and the rank association between the stored VCC label and the
target's own delta on the rows whose target is a feature, at the row level,
between target genes (gene means) and within target genes (both centred on
their gene means).

These values describe the current label on a stated subset. They are not the
quantity the external part reports and are not comparable with it: the VCC
label is not size-fixed, the rows are the feature-selected subset of their
split, a row is a (target gene, batch) pair, the own delta is a batch-local
pseudobulk read from the frozen feature cache, and validation and test rest on
21 and 33 target genes. No difference between the VCC and external values is
attributed to any cause here; the matched comparison is read-out R3 of E1a.

The own delta is negative in nearly every VCC row whose target is a feature,
so the negated and the absolute own delta give nearly the same association;
both are kept to match the external table, where the sign varies more.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, features, stats  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, strict_json  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "diag"
SPLITS = ("train", "val", "test")
TOP_K = (1, 2, 5, 10)
# Intervals over fewer target genes than this are marked descriptive, the
# convention of bioinf2027/plans/D1b.md (Section 7.5).
DESCRIPTIVE_BELOW_GENES = 50
# Within-gene associations use target genes with at least this many rows.
MIN_ROWS_WITHIN_GENE = 3


def by_level(split: str, y: np.ndarray, own: np.ndarray, target: np.ndarray) -> list[dict]:
    """Between-gene and within-gene association of the label with the negated own delta."""
    frame = pd.DataFrame({"y": y, "x": -own, "gene": target})
    means = frame.groupby("gene", sort=True)[["y", "x"]].mean()
    counts = frame.groupby("gene")["y"].transform("size").to_numpy()
    keep = counts >= MIN_ROWS_WITHIN_GENE
    centred = frame.loc[keep, ["y", "x"]] - frame.loc[keep].groupby("gene")[["y", "x"]].transform("mean")
    out = []
    for level, x, label, groups, note in (
        (
            "between_gene",
            means["x"].to_numpy(),
            means["y"].to_numpy(),
            means.index.to_numpy(),
            "one value per target gene: the mean over its rows of the label and of the negated own delta",
        ),
        (
            "within_gene",
            centred["x"].to_numpy(),
            centred["y"].to_numpy(),
            frame.loc[keep, "gene"].to_numpy(),
            "label and negated own delta centred on their target-gene means; genes with at least %d rows"
            % MIN_ROWS_WITHIN_GENE,
        ),
    ):
        estimate = stats.table_row(stats.cluster_bootstrap_corr(x, label, groups))
        out.append(
            {
                "split": split,
                "predictor": "negated_own_delta",
                "outcome": "y_log1p_AD",
                "subset": "rows_with_target_feature",
                "level": level,
                "definition": note,
                **estimate,
                "descriptive": bool(estimate["n_target_genes"] < DESCRIPTIVE_BELOW_GENES),
            }
        )
    return out


def shard_root() -> Path:
    return config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "variants" / "delta_hvg" / "shards"


def load_split(split: str) -> list[dict]:
    shards = sorted(glob.glob(str(shard_root() / split / "*.npz")))
    if not shards:
        raise RuntimeError("no cache shards found for split %s" % split)

    frames = []
    for path in shards:
        store = np.load(path, allow_pickle=True)
        frames.append(
            {
                "x": np.asarray(store["x"]),
                "y": np.asarray(store["y"], dtype=np.float64),
                "target_gene": np.asarray(store["sample_ids"]).astype(str),
                "batch": np.repeat(np.asarray(store["batch_name"]).astype(str), len(store["y"])),
            }
        )
    return frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = features.hvg_symbols()
    position = {symbol: index for index, symbol in enumerate(symbols.tolist())}

    rows, assoc, levels = [], [], []
    for split in SPLITS:
        frames = load_split(split)
        y = np.concatenate([f["y"] for f in frames])
        target = np.concatenate([f["target_gene"] for f in frames])
        batch = np.concatenate([f["batch"] for f in frames])
        column = np.array([position.get(symbol, -1) for symbol in target])
        present = column >= 0

        own = np.full(len(y), np.nan)
        rank = np.full(len(y), np.nan)
        for frame, offset in zip(frames, np.cumsum([0] + [len(f["y"]) for f in frames])[:-1]):
            local = slice(offset, offset + len(frame["y"]))
            local_cols = column[local]
            mask = local_cols >= 0
            if not mask.any():
                continue
            matrix = frame["x"][mask]
            cols = local_cols[mask]
            magnitude = np.abs(matrix)
            own_value = magnitude[np.arange(matrix.shape[0]), cols]
            greater = (magnitude > own_value[:, None]).sum(axis=1)
            equal = (magnitude == own_value[:, None]).sum(axis=1)
            idx = np.flatnonzero(mask) + offset
            own[idx] = matrix[np.arange(matrix.shape[0]), cols]
            rank[idx] = greater + (equal + 1) / 2.0

        record = {
            "split": split,
            "n_rows": int(len(y)),
            "n_targets": int(len(set(target.tolist()))),
            "n_batches": int(len(set(batch.tolist()))),
            "rows_with_target_feature": int(present.sum()),
            "rows_with_target_feature_fraction": float(present.mean()),
            "targets_with_feature": int(len(set(target[present].tolist()))),
            "own_delta_median": float(np.nanmedian(own)),
            "own_delta_fraction_negative": float(np.nanmean(own[present] < 0)),
        }
        # The conditional fraction is over the rows whose target is a feature;
        # the unconditional one over all rows of the split, since a row whose
        # target is not a feature never has the target as its largest coordinate.
        ranks = rank[present]
        for k in TOP_K:
            label = "argmax" if k == 1 else f"top{k}"
            hits = int(np.sum(ranks <= k))
            record[f"{label}_fraction_of_rows_with_target_feature"] = (
                float(hits / ranks.size) if ranks.size else float("nan")
            )
            record[f"{label}_rows"] = hits
            record[f"{label}_fraction_of_all_rows"] = float(hits / len(y))
        record["median_rank"] = float(np.median(ranks)) if ranks.size else float("nan")
        rows.append(record)
        levels.extend(by_level(split, y[present], own[present], target[present]))

        for name, values in (
            ("negated_own_delta", -own),
            ("abs_own_delta", np.abs(own)),
        ):
            estimate = stats.table_row(
                stats.cluster_bootstrap_corr(values[present], y[present], target[present])
            )
            assoc.append(
                {
                    "split": split,
                    "predictor": name,
                    "outcome": "y_log1p_AD",
                    "subset": "rows_with_target_feature",
                    "row_unit": "(target gene, batch) row",
                    "n_rows_in_split": record["n_rows"],
                    "n_target_genes_in_split": record["n_targets"],
                    **estimate,
                    "descriptive": bool(estimate["n_target_genes"] < DESCRIPTIVE_BELOW_GENES),
                }
            )
        print(f"[{split}] {record['n_rows']} rows, {record['rows_with_target_feature']} with the target as a feature", flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "D1_target_in_features_vcc.csv", index=False)
    pd.DataFrame(assoc).to_csv(OUT_DIR / "D1_label_vs_own_delta_vcc.csv", index=False)
    pd.DataFrame(levels).to_csv(OUT_DIR / "D1_label_vs_own_delta_vcc_by_level.csv", index=False)
    (OUT_DIR / "D1_vcc_provenance.json").write_text(
        json.dumps(
            strict_json({
                "paths_relative_to": "$" + config.STORAGE_VARIABLE,
                "inputs": {
                    "feature_shards": folder_record(
                        glob.glob(str(shard_root() / "*" / "*.npz")), shard_root()
                    ),
                    "vcc_gene_names": file_record(config.vcc_cache()["var_gene_names"]),
                    "vcc_selected_genes": file_record(config.vcc_cache()["selected_genes"]),
                },
                "row_unit": "(target gene, batch) pseudobulk",
                "reading": "descriptive statistics of the current VCC label on the rows whose target is a feature; not comparable with the external values",
                "bootstrap": {
                    "b": stats.BOOTSTRAP_B,
                    "seed": stats.BOOTSTRAP_SEED,
                    "unit": "target gene",
                    "descriptive_below_target_genes": DESCRIPTIVE_BELOW_GENES,
                },
                "within_gene_min_rows": MIN_ROWS_WITHIN_GENE,
                "summaries": rows,
            }),
            indent=1,
            allow_nan=False,
        )
        + "\n"
    )
    print("wrote", config.portable(OUT_DIR / "D1_target_in_features_vcc.csv"))


if __name__ == "__main__":
    main()
