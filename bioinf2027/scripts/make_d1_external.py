"""D1, external part: what does the largest-magnitude coordinate measure?

For the two external screens this reports, per screen:
  (i)   how often the target gene is one of the 2,000 selected features,
  (ii)  how often the largest absolute coordinate of the response vector is
        the target gene's own coordinate, with the rank distribution,
  (iii) the association between the endpoint and the target gene's own delta.

Two feature spaces are reported throughout. The native space is the screen's
own gene set, which always contains the target. The mapped space is the
2,000-feature space the heads actually consume, where unavailable coordinates
are zero filled, so the target is present only when it is a selected feature.
In the mapped space the argmax and top-k shares are given both over the
constructs whose target is a feature and over all constructs.

The label and the own delta share cells: each label draw is a subset of the
perturbed cells the own delta averages, and of the pooled control cells. A
sensitivity without the constructs whose label sits at the floor of zero is
written separately.

Writes four CSV files and one JSON provenance file under bioinf2027/diag/; the
provenance file records the path, size and digest of every input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, features, stats  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "diag"
SCREENS = ("RPE1", "K562-essential")
TOP_K = (1, 2, 5, 10)
# Intervals over fewer target genes than this are marked descriptive, the
# convention of bioinf2027/plans/D1b.md (Section 7.5).
DESCRIPTIVE_BELOW_GENES = 50


def rank_of_target(matrix: np.ndarray, column_index: np.ndarray) -> np.ndarray:
    """Rank of the target coordinate by descending absolute value, 1 is largest.

    Ties take the average rank, so a coordinate tied with others does not get
    credit for being the single largest.
    """
    magnitude = np.abs(matrix)
    own = magnitude[np.arange(matrix.shape[0]), column_index]
    greater = (magnitude > own[:, None]).sum(axis=1)
    equal = (magnitude == own[:, None]).sum(axis=1)
    return greater + (equal + 1) / 2.0


def per_construct_table(screen: features.ScreenData, symbols: np.ndarray) -> pd.DataFrame:
    endpoint = screen.endpoint
    position = {symbol: index for index, symbol in enumerate(symbols.tolist())}

    target_symbol = endpoint["target_symbol"].astype(str).to_numpy()
    native_col = endpoint["target_col"].astype(int).to_numpy()
    in_features = np.array([symbol in position for symbol in target_symbol])
    mapped_col = np.array([position.get(symbol, -1) for symbol in target_symbol])

    native_rank = rank_of_target(screen.x_native, native_col)
    own_delta_native = screen.x_native[np.arange(len(endpoint)), native_col]

    mapped_rank = np.full(len(endpoint), np.nan)
    own_delta_mapped = np.full(len(endpoint), np.nan)
    if in_features.any():
        rows = np.flatnonzero(in_features)
        sub_rank = rank_of_target(screen.x_mapped[rows], mapped_col[rows])
        mapped_rank[rows] = sub_rank
        own_delta_mapped[rows] = screen.x_mapped[rows, mapped_col[rows]]

    return pd.DataFrame(
        {
            "screen": screen.name,
            "construct": screen.constructs,
            "target_symbol": target_symbol,
            "target_ensg": endpoint["target_ensg"].astype(str).to_numpy(),
            "y_primary": endpoint["y_primary"].to_numpy(),
            "n_perturbed_pool": endpoint["n_perturbed_pool"].to_numpy(),
            "target_is_selected_feature": in_features,
            "target_col_native": native_col,
            "target_col_mapped": mapped_col,
            "own_delta_native": own_delta_native,
            "own_delta_mapped": own_delta_mapped,
            "target_rank_native": native_rank,
            "target_rank_mapped": mapped_rank,
            "n_genes_native": screen.x_native.shape[1],
        }
    )


def summarise(table: pd.DataFrame) -> dict:
    screen = table["screen"].iloc[0]
    n = len(table)
    selected = table["target_is_selected_feature"].to_numpy()
    out = {
        "screen": screen,
        "n_constructs": int(n),
        "n_unique_targets": int(table["target_ensg"].nunique()),
        "n_genes_native": int(table["n_genes_native"].iloc[0]),
        "n_features_mapped": 2000,
        "target_is_selected_feature_constructs": int(selected.sum()),
        "target_is_selected_feature_fraction": float(selected.mean()),
        "target_is_selected_feature_unique_targets": int(
            table.loc[selected, "target_ensg"].nunique()
        ),
    }

    for space, column, mask in (
        ("native", "target_rank_native", np.ones(n, dtype=bool)),
        ("mapped", "target_rank_mapped", selected),
    ):
        ranks = table.loc[mask, column].to_numpy()
        out[f"{space}_n_rows_scored"] = int(mask.sum())
        for k in TOP_K:
            label = "argmax" if k == 1 else f"top{k}"
            hits = int(np.sum(ranks <= k))
            if space == "native":
                # Every construct is scored in the native space.
                out[f"{space}_{label}_fraction"] = float(hits / ranks.size) if ranks.size else float("nan")
            else:
                # Conditional on the target being a feature, and over all constructs:
                # a construct whose target is not a feature never has it as its largest coordinate.
                out[f"{space}_{label}_fraction_of_constructs_with_target_feature"] = (
                    float(hits / ranks.size) if ranks.size else float("nan")
                )
                out[f"{space}_{label}_constructs"] = hits
                out[f"{space}_{label}_fraction_of_all_constructs"] = float(hits / n)
        out[f"{space}_median_rank"] = float(np.median(ranks)) if ranks.size else float("nan")

    deltas = table["own_delta_native"].to_numpy()
    out["own_delta_native_median"] = float(np.median(deltas))
    out["own_delta_native_fraction_negative"] = float(np.mean(deltas < 0))
    out["label_at_zero_constructs"] = int(np.sum(table["y_primary"].to_numpy() == 0))
    return out


def label_floor_sensitivity(table: pd.DataFrame) -> dict:
    """The label against the negated own delta without the constructs whose label is exactly zero."""
    y = table["y_primary"].to_numpy()
    keep = y != 0
    estimate = stats.table_row(
        stats.cluster_bootstrap_corr(
            -table["own_delta_native"].to_numpy()[keep], y[keep], table["target_ensg"].to_numpy()[keep]
        )
    )
    return {
        "screen": table["screen"].iloc[0],
        "predictor": "negated_own_delta_native",
        "outcome": "y_primary",
        "subset": "label_above_zero",
        "row_unit": "construct",
        "n_rows_in_screen": int(len(table)),
        "n_excluded_label_at_zero": int((~keep).sum()),
        **estimate,
        "descriptive": bool(estimate["n_target_genes"] < DESCRIPTIVE_BELOW_GENES),
    }


def associations(table: pd.DataFrame) -> list[dict]:
    rows = []
    groups = table["target_ensg"].to_numpy()
    y = table["y_primary"].to_numpy()
    specs = [
        ("own_delta_native", table["own_delta_native"].to_numpy(), "all"),
        ("negated_own_delta_native", -table["own_delta_native"].to_numpy(), "all"),
        ("abs_own_delta_native", np.abs(table["own_delta_native"].to_numpy()), "all"),
    ]
    selected = table["target_is_selected_feature"].to_numpy()
    if selected.any():
        specs.append(
            (
                "negated_own_delta_mapped",
                -table["own_delta_mapped"].to_numpy(),
                "target_is_selected_feature",
            )
        )
    for name, values, subset in specs:
        mask = selected if subset != "all" else np.ones(len(table), dtype=bool)
        estimate = stats.table_row(stats.cluster_bootstrap_corr(values[mask], y[mask], groups[mask]))
        rows.append(
            {
                "screen": table["screen"].iloc[0],
                "predictor": name,
                "outcome": "y_primary",
                "subset": subset,
                "row_unit": "construct",
                "n_rows_in_screen": int(len(table)),
                "n_target_genes_in_screen": int(table["target_ensg"].nunique()),
                **estimate,
                "descriptive": bool(estimate["n_target_genes"] < DESCRIPTIVE_BELOW_GENES),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = features.hvg_symbols()

    tables, summaries, assoc, floor = [], [], [], []
    for name in SCREENS:
        screen = features.load_screen(name)
        table = per_construct_table(screen, symbols)
        tables.append(table)
        summaries.append(summarise(table))
        assoc.extend(associations(table))
        floor.append(label_floor_sensitivity(table))
        print(f"[{name}] {len(table)} constructs scored", flush=True)

    per_construct = pd.concat(tables, ignore_index=True)
    per_construct.to_csv(OUT_DIR / "D1_target_in_features_external_per_construct.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "D1_target_in_features_external.csv", index=False)
    pd.DataFrame(assoc).to_csv(OUT_DIR / "D1_label_vs_own_delta_external.csv", index=False)
    pd.DataFrame(floor).to_csv(OUT_DIR / "D1_label_vs_own_delta_external_label_floor.csv", index=False)

    paths = config.screen_paths()
    used = ("endpoint", "x_native", "x_mapped")
    vcc = config.vcc_cache()
    provenance = {
        "paths_relative_to": "$" + config.STORAGE_VARIABLE,
        "bootstrap": {
            "b": stats.BOOTSTRAP_B,
            "seed": stats.BOOTSTRAP_SEED,
            "unit": "target gene",
            "descriptive_below_target_genes": DESCRIPTIVE_BELOW_GENES,
        },
        "shared_cells": "each label draw is a subset of the perturbed and control cells behind the own delta",
        "feature_space_mapped": "2,000 selected genes, unavailable coordinates zero filled, delta clipped to +/- 10",
        "inputs": {
            **{f"{name}_{key}": file_record(paths[name][key]) for name in SCREENS for key in used},
            "vcc_gene_names": file_record(vcc["var_gene_names"]),
            "vcc_selected_genes": file_record(vcc["selected_genes"]),
        },
        "summaries": summaries,
    }
    (OUT_DIR / "D1_external_provenance.json").write_text(
        json.dumps(strict_json(provenance), indent=1, allow_nan=False) + "\n"
    )
    print("wrote", config.portable(OUT_DIR / "D1_target_in_features_external.csv"))


if __name__ == "__main__":
    main()
