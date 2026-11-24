"""D3: per-screen cell-count support, control pools and batch structure.

Every value is read from a persisted artifact and the artifact path is recorded
alongside it, so the resulting table can be checked line by line. Nothing is
recomputed from raw expression data.

Two cell-count conventions exist for the external screens and both are
reported, because they differ: the pooled eligible count used by the endpoint
construction, and the count on the cell set used by the encoder feature
accumulator. The difference is a documented cell-set difference, not an error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "diag"
QUANTILES = (0.25, 0.5, 0.75)


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    q25, median, q75 = np.quantile(values, QUANTILES)
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def vcc_rows() -> tuple[list[dict], dict]:
    root = config.storage_root() / "IGA_MAIN"
    sizes_path = root / "results_v4" / "harmonized_endpoint_v1" / "subsample_sizes.json"
    sizes = json.loads(sizes_path.read_text())

    nt_per_batch = np.array(list(sizes["nt_per_batch"].values()), dtype=np.float64)
    rows = []
    for split in ("train", "val", "test"):
        summary = sizes["per_split_perturbed"][split]["summary"]
        rows.append(
            {
                "screen": "VCC",
                "split": split,
                "row_unit": "(target gene, batch) pseudobulk",
                "n_rows": int(summary["n"]),
                "perturbed_min": summary["min"],
                "perturbed_q25": summary["q25"],
                "perturbed_median": summary["median"],
                "perturbed_q75": summary["q75"],
                "perturbed_max": summary["max"],
                "perturbed_mean": summary["mean"],
                "control_scheme": "batch local, all cells, no subsampling",
                "control_pool_total": int(nt_per_batch.sum()),
                "control_per_batch_min": float(nt_per_batch.min()),
                "control_per_batch_median": float(np.median(nt_per_batch)),
                "control_per_batch_max": float(nt_per_batch.max()),
                "n_batches": int(sizes["n_vcc_batches"]),
                "batch_local_controls_feasible": int(sizes["n_vcc_batches"]),
                "endpoint_size_controlled": False,
                "source": config.portable(sizes_path),
            }
        )
    return rows, {"vcc_subsample_sizes": config.portable(sizes_path)}


def screen_rows() -> tuple[list[dict], dict]:
    paths = config.screen_paths()
    results = config.storage_root() / "IGA_MAIN" / "results_v4"
    coverage_files = {
        "RPE1": results / "harmonized_endpoint_v1" / "coverage_summary.json",
        "K562-essential": results / "harmonized_endpoint_k562_essential_v1" / "coverage_summary.json",
    }
    coverage_keys = {"RPE1": "rpe1", "K562-essential": "k562_essential"}
    gemgroup_nt = results / "k562_essential_coverage_gate_v1" / "gemgroup_nt.csv"

    rows, sources = [], {}
    for name in ("RPE1", "K562-essential"):
        endpoint = pd.read_csv(paths[name]["endpoint"])
        eligibility = pd.read_csv(paths[name]["eligibility"])
        coverage = json.loads(coverage_files[name].read_text())[coverage_keys[name]]

        pooled = describe(endpoint["n_perturbed_pool"].to_numpy())
        all_non_nt = describe(eligibility["n_perturbed_cells"].to_numpy())
        eligible_gemgroups = endpoint["n_eligible_gemgroups"].fillna(0).to_numpy()

        if name == "K562-essential":
            table = pd.read_csv(gemgroup_nt)
            n_batches = int(len(table))
            batch_source = config.portable(gemgroup_nt)
        else:
            n_batches = int(eligible_gemgroups.max())
            batch_source = "lower bound from n_eligible_gemgroups; screen total not in a results artifact"

        rows.append(
            {
                "screen": name,
                "split": "external test",
                "row_unit": "construct, cells pooled over batches",
                "n_rows": int(len(endpoint)),
                "perturbed_min": pooled["min"],
                "perturbed_q25": pooled["q25"],
                "perturbed_median": pooled["median"],
                "perturbed_q75": pooled["q75"],
                "perturbed_max": pooled["max"],
                "perturbed_mean": pooled["mean"],
                "control_scheme": "pooled across batches, fixed subsample of 788 per draw",
                "control_pool_total": int(endpoint["n_nt_pool"].iloc[0]),
                "control_per_batch_min": float("nan"),
                "control_per_batch_median": float("nan"),
                "control_per_batch_max": float("nan"),
                "n_batches": n_batches,
                "batch_local_controls_feasible": int((eligible_gemgroups >= 3).sum()),
                "endpoint_size_controlled": True,
                "source": config.portable(paths[name]["endpoint"]),
                "denom1_raw_non_nt": coverage["denom1_raw_nonNT"],
                "denom2_post_join": coverage["denom2_post_join"],
                "denom3_transfer": coverage.get("denom3", coverage["denom2_post_join"]),
                "n_eligible": coverage["n_eligible"],
                "coverage_denom3": coverage.get("coverage_denom3"),
                "distinct_target_genes": coverage["distinct_eligible_target_cols"],
                "excluded_target_not_in_var": coverage["exclusion_reason_counts"].get(
                    "target_ensg_not_in_var"
                ),
                "excluded_too_few_cells": coverage["exclusion_reason_counts"].get(
                    "perturbed_cells<27"
                ),
                "excluded_both": coverage["exclusion_reason_counts"].get(
                    "target_ensg_not_in_var;perturbed_cells<27"
                ),
                "all_non_nt_perturbed_median": all_non_nt["median"],
                "all_non_nt_perturbed_min": all_non_nt["min"],
                "batch_count_source": batch_source,
            }
        )
        sources[f"{name}_endpoint"] = config.portable(paths[name]["endpoint"])
        sources[f"{name}_eligibility"] = config.portable(paths[name]["eligibility"])
        sources[f"{name}_coverage"] = config.portable(coverage_files[name])
    sources["k562_gemgroup_nt"] = config.portable(gemgroup_nt)
    return rows, sources


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vcc, vcc_sources = vcc_rows()
    external, external_sources = screen_rows()

    table = pd.DataFrame(vcc + external)
    table.to_csv(OUT_DIR / "D3_cell_counts.csv", index=False)

    provenance = {
        "paths_relative_to": "$" + config.STORAGE_VARIABLE,
        "sources": {**vcc_sources, **external_sources},
        "input_records": {
            name: file_record(config.storage_root() / path[len("$" + config.STORAGE_VARIABLE) + 1:])
            for name, path in {**vcc_sources, **external_sources}.items()
        },
        "conventions": {
            "VCC": "endpoint uses all perturbed cells of the row against the batch local control cells; not size controlled",
            "external": "endpoint draws 27 perturbed and 788 control cells per draw, 25 draws, median across draws",
            "batch_local_controls_feasible": "constructs with at least three batches meeting ten perturbed and fifty control cells",
        },
        "known_cell_set_difference": (
            "the encoder feature accumulator used a slightly different RPE1 cell set from the "
            "endpoint construction, so the RPE1 perturbed median differs between the two "
            "conventions; both are recorded in the source specifications"
        ),
    }
    (OUT_DIR / "D3_cell_counts_provenance.json").write_text(json.dumps(provenance, indent=1))
    print(table.to_string(index=False, max_colwidth=28))
    print("\nwrote", config.portable(OUT_DIR / "D3_cell_counts.csv"))


if __name__ == "__main__":
    main()
