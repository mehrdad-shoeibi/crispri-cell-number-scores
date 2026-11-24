"""D0: recompute the values behind the number-provenance table and recount it.

The table bioinf2027/audit/number_provenance.csv traces every number of the
submitted main text to an artifact. Where its classification rests on a value
recomputed from retained files, this script recomputes that value, so that the
table does not depend on an unsaved calculation. The keys of D0_checks.json:

  phase1                     paired R-squared margins of the encoder over controls
                             C1 and C2 on the VCC test split, retained paired bootstrap
  phase1_arms                every seed of every phase-1 arm against the stored test table
  phase4                     the external arm correlations and paired contrasts, retained
                             cluster bootstrap keyed on the target gene identifier
  seeds                      three-seed means and standard deviations of the deep models,
                             and the later reproduction of the same configuration
  ablation_table             the seed-0 ablation table against its retained file
  legacy_transfer_table      the legacy transfer table against its retained files
  feature_overlap            the gene totals behind the feature-dimension entry
  rpe1_accumulator_state     the retained mid-pass RPE1 state, and the complete-pass artifacts
  k562_accumulator_state     the retained K562-essential state
  model_parameters           the encoder's parameter count from its weights header, and the
                             two locations of its token dictionary
  strong_baselines_per_seed  the strong baselines from their per-seed file
  vcc_rows                   VCC rows and target genes per split
  coverage                   the coverage denominators of both screens
  mag_anchor_seeds           the five MAG-Anchor seeds
  body_line_coverage         which digit-bearing lines of the main body a row cites

D0_counts.json recounts the table's classes, agreements and recomputation tags,
and the changes since the table went to its first audit. D0_input_manifest.csv
lists, with size and digest, every path the table's artifact_path and
producer_script columns cite and every file this script reads; a folder is
listed with its file count and one digest over its files. Nothing is modelled
or refitted.

Outputs: bioinf2027/audit/D0_checks.json, bioinf2027/audit/D0_counts.json,
bioinf2027/audit/D0_input_manifest.csv.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit"
READ: set = set()  # every file this script reads, for the manifest
TABLE = AUDIT_DIR / "number_provenance.csv"
MAIN_TEX_LAST_BODY_LINE = 791
TEX = "earlier_revision/v7/main.tex"


# --------------------------------------------------------------------------- helpers


def repo() -> Path:
    return config.repo_root()


def note(path) -> Path:
    """Register a file this script reads and return it."""
    READ.add(Path(path).resolve())
    return Path(path)


def load_retained_module(relative: str, name: str):
    """Import a retained module of the earlier protocol without writing bytecode next to it.

    The module is registered as a file this script reads, so the manifest lists it.
    """
    path = note(repo() / relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value):
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def r2_pooled(y, yh) -> float:
    """earlier_revision/pipeline/phase1_harness.py:91-94, the scoring of the submitted harness."""
    y = np.asarray(y, float)
    yh = np.asarray(yh, float)
    sst = ((y - y.mean()) ** 2).sum()
    return float("nan") if sst == 0 else float(1.0 - ((y - yh) ** 2).sum() / sst)


def numbers_in(text: str) -> list[float]:
    """Signed decimal numbers in a line of a table, in order."""
    return [float(v) for v in re.findall(r"[+-]?\d+\.\d+", text.replace("$", " "))]


def read_numbers(path) -> pd.DataFrame:
    """A retained CSV file, every decimal parsed to the exact double it stands for."""
    return pd.read_csv(path, float_precision="round_trip")


# --------------------------------------------------------------------------- phase1


def phase1_margins() -> dict:
    preds = repo() / "earlier_revision" / "phase1" / "preds"
    y = np.load(note(preds / "y_test.npy"))
    stored = json.loads(note(repo() / "earlier_revision" / "phase1" / "test_table.json").read_text())
    bw = load_retained_module("earlier_revision/pipeline/bootstrap_wiring.py", "bootstrap_wiring_retained")
    out = {
        "inputs": {
            "y_test": {"dtype": str(y.dtype), "shape": list(y.shape)},
            "prediction_dtype": str(np.load(note(preds / "test_z.npy")).dtype),
        }
    }
    for other in ("C1", "C2"):
        z = np.load(note(preds / "test_z.npy"))
        c = np.load(note(preds / f"test_{other}.npy"))
        per_seed_z = [r2_pooled(y, z[s]) for s in range(z.shape[0])]
        per_seed_c = [r2_pooled(y, c[s]) for s in range(c.shape[0])]
        boot = bw.vcc_paired_delta_r2(y, list(z), list(c), B=bw.VCC_B, seed=bw.VCC_SEED)
        out[f"z_minus_{other}"] = {
            "point_seed_averaged": float(np.mean(per_seed_z) - np.mean(per_seed_c)),
            "point_from_test_table_json": float(stored["z"]["r2_mean"] - stored[other]["r2_mean"]),
            "mean_prediction_variant": r2_pooled(y, z.mean(axis=0)) - r2_pooled(y, c.mean(axis=0)),
            "bootstrap_median": boot["median"],
            "bootstrap_ci": [boot["ci_lo"], boot["ci_hi"]],
            "bootstrap_b": bw.VCC_B,
            "bootstrap_seed": bw.VCC_SEED,
            "procedure": "earlier_revision/pipeline/bootstrap_wiring.py vcc_paired_delta_r2, rows resampled, per-seed R-squared averaged inside each replicate",
        }
    return out


# --------------------------------------------------------------------------- phase4


def phase4_primary() -> dict:
    store = np.load(note(repo() / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz"), allow_pickle=True)
    stored = json.loads(note(repo() / "earlier_revision" / "phase4" / "phase4_primary.json").read_text())
    bw = load_retained_module("earlier_revision/pipeline/bootstrap_wiring.py", "bootstrap_wiring_retained")
    arms = np.asarray(store["arms"]).astype(str).tolist()
    endpoints = {
        "RPE1": repo() / "earlier_revision" / "endpoint" / "rpe1_endpoint_by_construct.csv",
        "K562": repo() / "earlier_revision" / "endpoint" / "k562_essential_endpoint_by_construct.csv",
    }
    out = {"cluster_key": "target_ensg", "screens": {}}
    for screen, path in endpoints.items():
        endpoint = read_numbers(note(path)).set_index("construct")
        constructs = np.asarray(store[f"{screen}_constructs"]).astype(str)
        predictions = np.asarray(store[f"{screen}_P"], dtype=float)
        y = endpoint.loc[constructs, "y_primary"].to_numpy(dtype=float)
        clusters = endpoint.loc[constructs, "target_ensg"].astype(str).to_numpy()
        rho_boot, contrast = bw.external_paired_delta_rho(
            predictions, y, clusters, {arm: [j] for j, arm in enumerate(arms)},
            B=bw.EXT_B, seed=bw.EXT_SEED[screen],
        )
        recomputed, worst = {}, 0.0
        for j, arm in enumerate(arms):
            rho = float(spearmanr(predictions[:, j], y).statistic)
            lo, hi = np.percentile(rho_boot[:, j][np.isfinite(rho_boot[:, j])], [2.5, 97.5])
            recomputed[arm] = {"rho": rho, "ci_lo": float(lo), "ci_hi": float(hi)}
            record = stored["screens"][screen]["arm_rho"][arm]
            worst = max(worst, abs(rho - record["rho"]), abs(lo - record["ci_lo"]), abs(hi - record["ci_hi"]))
        contrasts = {}
        for name, record in stored["screens"][screen]["contrasts"].items():
            first, second = name.split(" ")[-1].split("-")
            mine = contrast(first, second)
            contrasts[name] = mine
            for key in ("ci_low", "ci_high", "ci_median"):
                worst = max(worst, abs(mine[key] - record[key]))
        out["screens"][screen] = {
            "n_constructs": int(len(y)),
            "n_clusters": int(len(set(clusters.tolist()))),
            "bootstrap_seed": bw.EXT_SEED[screen],
            "arms": recomputed,
            "contrasts": contrasts,
            "max_abs_difference_from_phase4_primary_json": worst,
        }
    return out


# --------------------------------------------------------------------------- seeds


def three_seed_statistics() -> dict:
    results = config.storage_root() / "IGA_MAIN" / "results"
    table = read_numbers(note(results / "phase9c_lite" / "results_table_3seed.csv"))
    test = table[table["split"] == "test"]
    out = {"source": config.portable(results / "phase9c_lite" / "results_table_3seed.csv"), "families": {}}
    for family, rows in test.groupby("family"):
        values = rows.sort_values("seed")["R2"].to_numpy(dtype=float)
        out["families"][family] = {
            "seeds": rows.sort_values("seed")["seed"].astype(int).tolist(),
            "per_seed_r2": values.tolist(),
            "mean": float(values.mean()),
            "sd_ddof1": float(values.std(ddof=1)),
            "sd_ddof0": float(values.std(ddof=0)),
        }
    gate = json.loads(note(repo() / "earlier_revision" / "phase1" / "repro_gate_final.json").read_text())
    out["reproduction_gate"] = {
        "source": "earlier_revision/phase1/repro_gate_final.json",
        "SPSM": {"mean": gate["SPSM"]["test_r2_mean"], "sd": gate["SPSM"]["test_r2_sd"], "per_seed": gate["SPSM"].get("per_seed")},
        "MAG_SPSM": {"mean": gate["MAG_SPSM"]["test_r2_mean"], "sd": gate["MAG_SPSM"]["test_r2_sd"], "per_seed": gate["MAG_SPSM"].get("per_seed")},
        "RF_X_MTILDE": {"mean": gate["RF_X_MTILDE"]["test_r2_mean"], "sd": gate["RF_X_MTILDE"]["test_r2_sd"]},
    }
    strong = read_numbers(note(results / "phase11_strong_baselines" / "summary.csv"))
    out["strong_baselines"] = {
        "source": config.portable(results / "phase11_strong_baselines" / "summary.csv"),
        "rows": strong[["method", "feature_set", "n_seeds", "mean_test_R2", "std_test_R2"]].to_dict("records"),
    }
    return out


# --------------------------------------------------------------------------- tables


def ablation_table() -> dict:
    """The seed-0 ablation table against the retained ablation results."""
    tex = note(repo() / "earlier_revision" / "v7" / "tables" / "table4_ablation.tex").read_text().splitlines()
    results = config.storage_root() / "IGA_MAIN" / "results" / "phase9d_ablation" / "magnitude_ablation_results.csv"
    frame = read_numbers(note(results))
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    printed_rows = [(i + 1, numbers_in(line)) for i, line in enumerate(tex) if line.count("&") >= 4 and numbers_in(line)]
    out = {"source": config.portable(results), "rows": []}
    for (line, printed), (_, row) in zip(printed_rows, test.iterrows()):
        out["rows"].append({"tex_line": line, "variant": row["variant"], "printed": printed,
                            "retained_test_r2": float(row["R2"]), "retained_spearman": float(row["Spearman"])})
    out["printed_rows"] = len(printed_rows)
    out["retained_test_rows"] = int(len(test))
    out["note"] = "rows paired in file order; every printed value is listed beside the retained R-squared and Spearman of its row"
    return out


def legacy_transfer_table() -> dict:
    """The legacy transfer table against the retained per-object metrics."""
    tex = note(repo() / "earlier_revision" / "v7" / "tables" / "table_harmonized_transfer.tex").read_text().splitlines()
    results = config.storage_root() / "IGA_MAIN" / "results_v4"
    files = {
        "RPE1": results / "arm1_rpe1_harmonized_v2" / "per_object_metrics.csv",
        "K562-essential": results / "arm2_k562_harmonized_v1" / "per_object_metrics.csv",
    }
    retained = {screen: read_numbers(note(path)) for screen, path in files.items()}
    order = [
        "Ridge__x_only", "Ridge__mag_only", "Ridge__x_plus_mag",
        "HistGradientBoosting__x_only", "HistGradientBoosting__mag_only", "HistGradientBoosting__x_plus_mag",
        "RandomForest__x_only", "RandomForest__mag_only", "RandomForest__x_plus_mag",
        "SPSM baseline", "MAG-SPSM", "MAG-LINEAR",
    ]
    lines = [(i + 1, numbers_in(text)) for i, text in enumerate(tex) if re.search(r"(Pos|Neg|Inc)\s*\\\\", text)]
    out = {"sources": {k: config.portable(v) for k, v in files.items()}, "rows": [], "objects_in_files": {}}
    for screen, frame in retained.items():
        out["objects_in_files"][screen] = frame["object"].astype(str).tolist()
    worst = 0.0
    for (line, printed), name in zip(lines, order):
        entry = {"tex_line": line, "object": name, "printed": printed}
        for k, screen in enumerate(("RPE1", "K562-essential")):
            frame = retained[screen]
            match = frame[frame["object"].astype(str) == name]
            value = float(match["spearman_harmonized"].iloc[0]) if len(match) == 1 else float("nan")
            entry[screen] = value
            if k < len(printed) and math.isfinite(value):
                worst = max(worst, abs(round(value, 3) - printed[k]))
        out["rows"].append(entry)
    out["printed_rows"] = len(lines)
    out["values_compared"] = int(sum(
        1 for r in out["rows"] for k, screen in enumerate(("RPE1", "K562-essential"))
        if k < len(r["printed"]) and math.isfinite(r[screen])
    ))
    out["max_abs_difference_at_three_decimals"] = worst
    if out["values_compared"] != 2 * len(order):
        raise RuntimeError("the legacy transfer table was not matched in full: %s" % out["values_compared"])
    return out


def feature_overlap() -> dict:
    results = config.storage_root() / "IGA_MAIN" / "results" / "phase10" / "replogle_vcc_feature_overlap.csv"
    row = read_numbers(note(results)).iloc[0].to_dict()
    native = config.storage_root() / "IGA_GENEFORMER_EXT" / "k562_essential" / "x_cell" / "x_cell_k562_native.npz"
    store = np.load(note(native), allow_pickle=True)
    return {
        "source": config.portable(results),
        "overlap_file": {k: (int(v) if float(v).is_integer() else float(v)) for k, v in row.items()},
        "k562_essential_native_matrix_genes": int(np.asarray(store["x"]).shape[1]),
        "k562_essential_native_matrix": config.portable(native),
    }


def rpe1_accumulator_state() -> dict:
    """What the retained RPE1 artifacts come from: the complete pass or the checkpoint."""
    import pickle

    encoder = config.storage_root() / "IGA_GENEFORMER_EXT"
    log = json.loads(note(encoder / "logs" / "phase2_embed_xcell.json").read_text())
    with open(note(encoder / "checkpoints" / "rpe1_xcell" / "xcell_state.pkl"), "rb") as handle:
        state = pickle.load(handle)
    counts = np.array(list(state["count"].values()), dtype=float)
    out = {
        "pass_log": {
            "path": config.portable(encoder / "logs" / "phase2_embed_xcell.json"),
            "chunks": log["chunks"],
            "total_cells": log["acc_trace"]["total_cells"],
            "control_cells": log["acc_trace"]["nt_count"],
            "checkpoint_reloaded_ok": log["checkpoint_reloaded_ok"],
        },
        "retained_state": {
            "path": config.portable(encoder / "checkpoints" / "rpe1_xcell" / "xcell_state.pkl"),
            "chunks_consumed": len(state["consumed"]),
            "updates": int(state["updates"]),
            "total_cells": int(state["total_cells"]),
            "control_cells": int(state["nt_count"]),
            "constructs": int(len(counts)),
            "perturbed_cells_per_construct": {
                "q25": float(np.percentile(counts, 25)),
                "median": float(np.percentile(counts, 50)),
                "q75": float(np.percentile(counts, 75)),
                "min": float(counts.min()),
                "max": float(counts.max()),
            },
        },
    }
    del state
    mapped = np.load(note(encoder / "x_cell" / "x_cell_rpe1_mapped.npz"), allow_pickle=True)
    ids, x = np.asarray(mapped["ids"]).astype(str), np.asarray(mapped["x"])
    copy = np.load(note(encoder / "x_cell" / "x_cell_rpe1_1932_mapped.npy"))
    magnitude = np.load(note(repo() / "earlier_revision" / "phase4" / "m_tilde" / "m_tilde_xcell_RPE1.npz"))
    xd = x.astype(np.float64)
    mine = np.column_stack([np.abs(xd).mean(1), np.sqrt((xd * xd).sum(1)), np.abs(xd).max(1), xd.std(1)])
    out["complete_pass_artifacts"] = {
        "mapped_files_identical": bool(np.array_equal(x, copy)),
        "mapped_ids_sorted": bool(np.all(ids[:-1] < ids[1:])),
        "magnitude_summaries_from_mapped_matrix_max_abs_difference": float(
            np.max(np.abs(np.asarray(magnitude["m_raw"], dtype=np.float64) - mine))
        ),
        "files": [
            config.portable(encoder / "x_cell" / "x_cell_rpe1_mapped.npz"),
            config.portable(encoder / "x_cell" / "x_cell_rpe1_1932_mapped.npy"),
            "earlier_revision/phase4/m_tilde/m_tilde_xcell_RPE1.npz",
        ],
        "magnitude_definition": "[mean |x|, ||x||_2, max |x|, sd(x)] per row, earlier_revision/pipeline/m_tilde.py:3",
    }
    return out


def phase1_arms() -> dict:
    """Per-seed test R-squared and Spearman of every phase-1 arm, against the stored test table."""
    preds = repo() / "earlier_revision" / "phase1" / "preds"
    y = np.load(note(preds / "y_test.npy"))
    stored = json.loads(note(repo() / "earlier_revision" / "phase1" / "test_table.json").read_text())
    out, worst = {}, 0.0
    for arm in ("z", "zm", "C1", "C2", "C3", "C4", "C5"):
        p = np.load(note(preds / f"test_{arm}.npy"))
        r2 = [r2_pooled(y, p[s]) for s in range(p.shape[0])]
        rho = [float(spearmanr(np.asarray(y, float), p[s]).statistic) for s in range(p.shape[0])]
        out[arm] = {"r2_per_seed": r2, "spearman_per_seed": rho, "r2_mean": float(np.mean(r2)),
                    "r2_sd_ddof1": float(np.std(r2, ddof=1))}
        worst = max(worst, max(abs(a - b) for a, b in zip(r2, stored[arm]["r2_per_seed"])),
                    max(abs(a - b) for a, b in zip(rho, stored[arm]["spearman_per_seed"])))
    out["max_abs_difference_from_test_table_json"] = worst
    out["scoring"] = "pooled R-squared and Spearman in double precision, earlier_revision/pipeline/phase1_harness.py:91-98"
    return out


def k562_accumulator_state() -> dict:
    import pickle

    path = config.storage_root() / "IGA_GENEFORMER_EXT" / "k562_essential" / "checkpoints" / "xcell_state.pkl"
    with open(note(path), "rb") as handle:
        state = pickle.load(handle)
    counts = np.array(list(state["count"].values()), dtype=float)
    out = {
        "path": config.portable(path),
        "control_cells": int(state["nt_count"]),
        "total_cells": int(state["total_cells"]),
        "constructs": int(len(counts)),
        "perturbed_cells_per_construct": {
            "min": float(counts.min()), "q25": float(np.percentile(counts, 25)),
            "median": float(np.percentile(counts, 50)), "q75": float(np.percentile(counts, 75)),
            "max": float(counts.max()),
        },
    }
    del state
    return out


def model_parameters() -> dict:
    """Parameter count of the frozen encoder from the tensor shapes in its weights header.

    Also records both copies of the token dictionary in the data root, with size
    and digest: the one at the relative path the experiment lock gives
    (geneformer/geneformer/, under a model folder outside the data root) and
    the one next to the weights.
    """
    models = config.storage_root() / "MADA" / "cache" / "foundation_models"
    path = models / "geneformer-repo" / "Geneformer-V2-104M" / "model.safetensors"
    with open(note(path), "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    total = sum(int(np.prod(v["shape"])) for k, v in header.items() if k != "__metadata__")
    dictionaries = {}
    for name, folder in (("at_the_lock_relative_path", "geneformer"), ("next_to_the_weights", "geneformer-repo")):
        location = models / folder / "geneformer" / "token_dictionary_gc104M.pkl"
        dictionaries[name] = {"path": config.portable(location), "bytes": note(location).stat().st_size, "sha256": sha256(location)}
    return {
        "path": config.portable(path),
        "tensors": len(header) - ("__metadata__" in header),
        "parameters": total,
        "token_dictionaries": dictionaries,
        "token_dictionaries_identical": dictionaries["at_the_lock_relative_path"]["sha256"] == dictionaries["next_to_the_weights"]["sha256"],
    }


def strong_baselines_per_seed() -> dict:
    results = config.storage_root() / "IGA_MAIN" / "results" / "phase11_strong_baselines"
    per_seed = read_numbers(note(results / "per_seed.csv"))
    summary = read_numbers(note(results / "summary.csv"))
    rows, worst = [], 0.0
    for (method, feature_set), group in per_seed.groupby(["method", "feature_set"]):
        values = group["test_R2"].to_numpy(dtype=float)
        stored = summary[(summary["method"] == method) & (summary["feature_set"] == feature_set)]
        entry = {"method": method, "feature_set": feature_set, "seeds": int(len(values)),
                 "mean": float(values.mean()),
                 "sd_ddof1": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                 "sd_ddof0": float(values.std(ddof=0))}
        if len(stored) == 1:
            entry["summary_mean"] = float(stored["mean_test_R2"].iloc[0])
            entry["summary_sd"] = float(stored["std_test_R2"].iloc[0])
            worst = max(worst, abs(entry["mean"] - entry["summary_mean"]))
        rows.append(entry)
    return {"source": config.portable(results / "per_seed.csv"), "rows": rows,
            "max_abs_difference_of_means_from_summary": worst}


def vcc_rows() -> dict:
    import glob as _glob

    shards = config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "variants" / "delta_hvg" / "shards"
    out = {"path": config.portable(shards)}
    for split in ("train", "val", "test"):
        n, genes = 0, set()
        for path in sorted(_glob.glob(str(shards / split / "*.npz"))):
            store = np.load(note(path), allow_pickle=True)
            n += int(np.asarray(store["y"]).shape[0])
            genes.update(np.asarray(store["sample_ids"]).astype(str).tolist())
        out[split] = {"rows": n, "target_genes": len(genes)}
    return out


def coverage_summaries() -> dict:
    results = config.storage_root() / "IGA_MAIN" / "results_v4"
    out = {}
    for name, rel, key in (("RPE1", "harmonized_endpoint_v1", "rpe1"), ("K562-essential", "harmonized_endpoint_k562_essential_v1", "k562_essential")):
        data = json.loads(note(results / rel / "coverage_summary.json").read_text())[key]
        out[name] = {k: data.get(k) for k in ("denom1_raw_nonNT", "denom2_post_join", "n_eligible", "coverage_denom3")}
        out[name]["path"] = config.portable(results / rel / "coverage_summary.json")
    return out


def mag_anchor_seeds() -> dict:
    path = config.storage_root() / "IGA_MAIN" / "results" / "mag_anchor" / "followup_xmag" / "base_xmag_5seed_vcc.csv"
    frame = read_numbers(note(path))
    values = frame["vcc_r2"].to_numpy(dtype=float)
    return {
        "path": config.portable(path),
        "seeds": frame["seed"].astype(int).tolist(),
        "vcc_r2": values.tolist(),
        "mean": float(values.mean()),
        "sd_ddof1": float(values.std(ddof=1)),
    }


def body_line_coverage(table: pd.DataFrame) -> dict:
    """Digit-bearing lines of the main body and whether a table row covers each."""
    lines = note(repo() / TEX).read_text().splitlines()[:MAIN_TEX_LAST_BODY_LINE]
    covered = set(
        pd.to_numeric(table.loc[table["source_file"] == TEX, "tex_line"], errors="coerce").dropna().astype(int)
    )
    digit_lines = [
        i + 1 for i, text in enumerate(lines)
        if re.search(r"\d", text) and not text.lstrip().startswith("%")
    ]
    uncovered = [i for i in digit_lines if i not in covered]
    return {
        "digit_bearing_lines": len(digit_lines),
        "covered_by_a_row": len(digit_lines) - len(uncovered),
        "not_covered": uncovered,
        "note": "a line is covered when a row cites it; an uncovered line is not necessarily an omission, since many carry only references, labels, typesetting or restated numbers",
    }


# --------------------------------------------------------------------------- table


BEFORE_AUDIT = "a5a7f76"  # the table as committed before the first reproducibility audit


def changes_since(table: pd.DataFrame, revision: str = BEFORE_AUDIT) -> dict:
    """Rows changed since the table went to its first audit, per column, and the dated reasons."""
    import io
    import subprocess

    text = subprocess.run(
        ["git", "show", f"{revision}:bioinf2027/audit/number_provenance.csv"],
        cwd=repo(), check=True, capture_output=True, text=True,
    ).stdout
    before = pd.read_csv(io.StringIO(text), keep_default_na=False, dtype=str)
    shared = [c for c in before.columns if c in table.columns]
    changed = {c: int((before[c].to_numpy() != table[c].astype(str).to_numpy()).sum()) for c in shared}
    reasons = table["correction_reason"].astype(str)
    rounds = {
        "first": reasons.str.contains("after the reproducibility audit"),
        **{name: reasons.str.contains(f"after the {name} reproducibility audit") for name in ("second", "third", "fourth")},
    }
    any_round = np.logical_or.reduce([mask.to_numpy() for mask in rounds.values()])
    return {
        "compared_with": revision,
        "rows_changed_per_column": changed,
        "columns_added": [c for c in table.columns if c not in before.columns],
        **{f"rows_with_a_dated_reason_after_the_{name}_audit": int(mask.sum()) for name, mask in rounds.items()},
        "rows_with_any_dated_reason": int(any_round.sum()),
        "rows_whose_class_differs_from_the_first_pass": int((table["classification"] != table["classification_as_extracted"]).sum()),
        "rows_the_first_pass_missed": int(table["classification_as_extracted"].astype(str).str.contains("missed", case=False).sum()),
    }


def counts(table: pd.DataFrame) -> dict:
    body = (table["source_file"] == TEX) & (pd.to_numeric(table["tex_line"], errors="coerce") <= MAIN_TEX_LAST_BODY_LINE)
    return {
        "rows": int(len(table)),
        "by_scope": table["scope"].value_counts().to_dict(),
        "by_classification": table["classification"].value_counts().to_dict(),
        "by_agreement": table["agrees"].value_counts().to_dict(),
        "main_body_by_line_range": {
            "definition": f"source {TEX}, lines 1 to {MAIN_TEX_LAST_BODY_LINE}",
            "rows": int(body.sum()),
            "by_classification": table.loc[body, "classification"].value_counts().to_dict(),
        },
        "scope_main_body_rows_outside_that_range": int(((table["scope"] == "main_body") & ~body).sum()),
        "classification_by_agreement": pd.crosstab(table["classification"], table["agrees"]).to_dict(),
        "by_recomputed_in": table["recomputed_in"].value_counts().to_dict() if "recomputed_in" in table else {},
        "changes": changes_since(table),
    }


_PATH_CHARS = r"(?:[^\s;,()\"'\]{}]|\{[^{}\s]*\})+"
PATH_PATTERN = re.compile(
    r"(\$IGA_STORAGE_ROOT/" + _PATH_CHARS
    + r"|(?<![\w/.$])(?:earlier_revision|scripts|src|docs|bioinf2027|configs|tests)/" + _PATH_CHARS + r")"
)


def expand(token: str) -> list[str]:
    match = re.search(r"\{([^{}]*)\}", token)
    if not match:
        return [token]
    head, tail = token[: match.start()], token[match.end():]
    return [p for option in match.group(1).split(",") for p in expand(head + option + tail)]


def cited_paths(table: pd.DataFrame) -> list[str]:
    found = set()
    for column in ("artifact_path", "producer_script"):
        for text in table[column].fillna("").astype(str):
            for token in PATH_PATTERN.findall(text):
                token = token.split("::", 1)[0]  # a function named after the file
                token = re.sub(r"(:\d[\d,\-]*)+$", "", token.rstrip(".:"))
                for path in expand(token):
                    found.add(path.rstrip("/.:"))
    return sorted(found)


def manifest(paths: list[str], read: set) -> pd.DataFrame:
    rows = []
    cited = set(paths)
    extra = sorted(config.portable(p) for p in read)
    for text in sorted(cited | set(extra)):
        if text.startswith("$IGA_STORAGE_ROOT/"):
            path = config.storage_root() / text[len("$IGA_STORAGE_ROOT/"):]
            root = "data_root"
        else:
            path = repo() / text
            root = "repository"
        entry = {"path": text, "root": root, "exists": path.exists(), "kind": "", "bytes": "", "sha256": "", "files": "",
                 "cited_by_the_table": text in cited, "read_by_this_script": text in extra}
        if path.is_file():
            entry.update({"kind": "file", "bytes": path.stat().st_size, "sha256": sha256(path)})
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            listing = "\n".join("%s %d %s" % (p.relative_to(path).as_posix(), p.stat().st_size, sha256(p)) for p in files)
            entry.update({"kind": "folder", "files": len(files), "sha256": hashlib.sha256(listing.encode()).hexdigest()})
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    table = pd.read_csv(TABLE, keep_default_na=False)
    checks = {
        "table": config.portable(TABLE),
        "table_sha256": sha256(TABLE),
        "phase1": phase1_margins(),
        "phase4": phase4_primary(),
        "seeds": three_seed_statistics(),
        "ablation_table": ablation_table(),
        "legacy_transfer_table": legacy_transfer_table(),
        "feature_overlap": feature_overlap(),
        "rpe1_accumulator_state": rpe1_accumulator_state(),
        "k562_accumulator_state": k562_accumulator_state(),
        "phase1_arms": phase1_arms(),
        "model_parameters": model_parameters(),
        "strong_baselines_per_seed": strong_baselines_per_seed(),
        "vcc_rows": vcc_rows(),
        "coverage": coverage_summaries(),
        "mag_anchor_seeds": mag_anchor_seeds(),
        "body_line_coverage": body_line_coverage(table),
    }
    paths = cited_paths(table)
    READ.add(TABLE.resolve())
    files = manifest(paths, READ)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "D0_checks.json").write_text(json.dumps(strict_json(checks), indent=1, allow_nan=False) + "\n")
    (AUDIT_DIR / "D0_counts.json").write_text(json.dumps(strict_json(counts(table)), indent=1, allow_nan=False) + "\n")
    files.to_csv(AUDIT_DIR / "D0_input_manifest.csv", index=False)
    print("phase1 z-C2:", {k: checks["phase1"]["z_minus_C2"][k] for k in ("point_seed_averaged", "bootstrap_median")})
    for screen, entry in checks["phase4"]["screens"].items():
        print("phase4", screen, "max abs difference from the stored file:", entry["max_abs_difference_from_phase4_primary_json"])
    print("legacy transfer table, max abs difference at 3 dp:", checks["legacy_transfer_table"]["max_abs_difference_at_three_decimals"])
    print("manifest: %d cited paths, %d missing" % (len(files), int((~files["exists"]).sum())))
    print("RPE1 state:", checks["rpe1_accumulator_state"]["retained_state"])
    print("RPE1 complete-pass artifacts:", checks["rpe1_accumulator_state"]["complete_pass_artifacts"]["mapped_files_identical"],
          checks["rpe1_accumulator_state"]["complete_pass_artifacts"]["magnitude_summaries_from_mapped_matrix_max_abs_difference"])
    print("MAG-Anchor five seeds:", checks["mag_anchor_seeds"]["mean"], checks["mag_anchor_seeds"]["sd_ddof1"])
    coverage = checks["body_line_coverage"]
    print("main-body digit lines:", coverage["digit_bearing_lines"], "covered:", coverage["covered_by_a_row"])
    print("wrote", config.portable(AUDIT_DIR / "D0_checks.json"))


if __name__ == "__main__":
    main()
