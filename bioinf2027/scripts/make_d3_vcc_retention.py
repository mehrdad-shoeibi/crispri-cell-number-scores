"""D3 supplement: how many VCC rows and target genes a perturbed-cell floor retains.

A size-controlled source label needs a fixed number of perturbed cells per row,
so every row with fewer cells is lost. This table counts, per split and for
each candidate floor, the rows and the target genes that remain, and how many
of the remaining rows have their target gene among the 2,000 features.

Only cell counts and row keys are read. No label is computed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, features  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "diag"
SPLITS = {"train": "training", "val": "validation", "test": "test"}
MODELLED_FLOOR = 10
FLOORS = (10, 27)


def retention_table(counts: pd.DataFrame, feature_symbols: set[str]) -> pd.DataFrame:
    """Rows and target genes retained per split and floor.

    ``counts`` holds one line per (split, target gene, batch) group with its
    number of perturbed cells. The modelled rows are the groups with at least
    ten cells; every share is taken over them.
    """
    rows = []
    for split, stored_name in SPLITS.items():
        groups = counts[counts["split"] == stored_name]
        modelled = groups[groups["n_perturbed_cells"] >= MODELLED_FLOOR]
        for floor in FLOORS:
            kept = modelled[modelled["n_perturbed_cells"] >= floor]
            if kept.empty:
                raise RuntimeError("no row of split %s reaches %d cells" % (split, floor))
            cells = kept["n_perturbed_cells"].to_numpy(dtype=np.float64)
            in_features = kept["target_gene"].astype(str).isin(feature_symbols)
            rows.append(
                {
                    "split": split,
                    "perturbed_cell_floor": floor,
                    "groups_with_any_cells": int(len(groups)),
                    "rows_modelled": int(len(modelled)),
                    "rows_retained": int(len(kept)),
                    "rows_retained_share_of_modelled": float(len(kept) / len(modelled)),
                    "rows_retained_share_of_all_groups": float(len(kept) / len(groups)),
                    "target_genes_modelled": int(modelled["target_gene"].nunique()),
                    "target_genes_retained": int(kept["target_gene"].nunique()),
                    "rows_retained_target_in_features": int(in_features.sum()),
                    "target_genes_retained_in_features": int(
                        kept.loc[in_features, "target_gene"].nunique()
                    ),
                    "cells_per_retained_row_min": float(cells.min()),
                    "cells_per_retained_row_q25": float(np.quantile(cells, 0.25)),
                    "cells_per_retained_row_median": float(np.median(cells)),
                    "cells_per_retained_row_q75": float(np.quantile(cells, 0.75)),
                    "cells_per_retained_row_max": float(cells.max()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = config.storage_root() / "MADA" / "data" / "metadata" / "vcc" / "final_labels.parquet"
    counts = pd.read_parquet(source, columns=["split", "target_gene", "batch", "n_perturbed_cells"])
    table = retention_table(counts, set(features.hvg_symbols().tolist()))
    table.to_csv(OUT_DIR / "D3_vcc_retention.csv", index=False)
    (OUT_DIR / "D3_vcc_retention_provenance.json").write_text(
        json.dumps(
            {
                "paths_relative_to": "$" + config.STORAGE_VARIABLE,
                "source": config.portable(source),
                "columns_read": ["split", "target_gene", "batch", "n_perturbed_cells"],
                "modelled_rows": "groups with at least %d perturbed cells" % MODELLED_FLOOR,
                "floors": list(FLOORS),
                "feature_set": "the 2,000 selected genes of the source feature cache",
                "note": "cell counts and row keys only; the label columns of the source file are not read",
            },
            indent=1,
        )
        + "\n"
    )
    print(table.to_string(index=False))
    print("\nwrote", config.portable(OUT_DIR / "D3_vcc_retention.csv"))


if __name__ == "__main__":
    main()
