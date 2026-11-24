"""D1b: is the endpoint readable without perturbation data?

Three parts, all on persisted artifacts, no retraining:
  (a) source association under adjustment for sampling depth,
  (b) how much of the endpoint control cells alone explain,
  (c) the archived transfer results split by whether the target gene is
      inside the feature space.

Two checks must pass before any value is computed. The first reproduces the
rank-residual partial correlations of the four magnitude summaries recorded by
the source analyses. The second confirms, for each external screen, that the
control mean reconstructs the persisted delta matrix exactly. Where the final
accumulator state was retained the control mean is recomputed from it in every
run. Where only a mid-pass state was retained it comes from a repeated pass
over the single-cell file, made by make_d1b_rpe1_control_mean.py, and is used
only if all three checks of that pass succeeded and the array matches the
digest the checks file records.

Every interval resamples target genes, 10,000 replicates. Part (c) uses one
joint draw per replicate for every object, baseline and subset, which also
gives the in-minus-out difference and the family-level mean difference. The
point estimates of the first run must come out unchanged; the script stops
before writing anything if one does not.

Part (c) also holds the masking contrast of Section 10.1: plain max |x| minus
max |x| with the target coordinate set to zero, in the same joint draw. Its
primary form, one per screen (cell-level input, all eligible constructs, all
constructs of the set), was fixed before its values were computed (Section
12.1); every other part (c) statistic is post hoc.

See bioinf2027/plans/D1b.md for the fixed design, Section 7 for the addendum
and Sections 8 to 12 for the changes after the four plan reviews.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, features, stats  # noqa: E402
from bioinf2027.partial import cluster_bootstrap_partial, partial_spearman  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, sha256, strict_json  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "diag"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
SPLITS = {"train": "training", "val": "validation", "test": "test"}
SPLIT_ROWS = {"train": 5740, "val": 2052, "test": 3972}
SCREENS = ("RPE1", "K562-essential")
RECORDED_PARTIAL = (-0.095, -0.010, 0.379, -0.010)
PARTIAL_TOLERANCE = 0.005
CONTROL_MEAN_FLOOR = 0.01
IDENTITY_TOLERANCE = 1e-9
N_QUINTILES = 5
FEW_CLUSTERS = 50
POWER_FACTOR = (1.959963984540054 + 0.8416212335729143) / 1.959963984540054
ENCODER_TOLERANCE = 1e-9
MAX_LINEAR_TOLERANCE = 5e-4

# The commit whose run of make_d1b_rpe1_control_mean.py wrote the RPE1 checks file
# (bioinf2027/runs/D1b.json); that script is left unchanged after its run (Section 12.2).
RPE1_PASS_COMMIT = "8e5f3fc"
# D1's count of constructs whose target is the strict, unique largest mapped coordinate.
D1_EXTERNAL_SUMMARY = OUT_DIR / "D1_target_in_features_external.csv"
CLIP = features.CLIP_LIMIT

FIRST_RUN_FILE = OUT_DIR / "D1b_first_run_point_estimates.csv"
FIRST_RUN_TOLERANCE = 1e-12
FIRST_RUN_COUNTS = {"a": 6, "a_quintiles": 15, "b": 42, "c": 72}
FIRST_RUN_RATIO = "negated_knockdown_fraction"

ROW_UNIT_SOURCE = "(target gene, batch) row"
ROW_UNIT_EXTERNAL = "construct"
SUBSET_NAMES = {"all": "all", "in": "target_in_features", "out": "target_not_in_features"}
RATIO = "negated_own_delta_over_control_mean"
VCC_RATIO_ALL = "all_rows_with_own_delta"

# Screens whose final accumulator state was retained.
ACCUMULATOR_STATE = {
    "K562-essential": ("IGA_GENEFORMER_EXT", "k562_essential/checkpoints/xcell_state.pkl"),
}
# Screens whose control mean comes from a repeated pass over the single-cell file.
REPEATED_PASS = {
    "RPE1": {
        "script": "bioinf2027/scripts/make_d1b_rpe1_control_mean.py",
        "array": "mu_nt_rpe1.npy",
        "checks": "D1b_rpe1_control_mean_checks.json",
    },
}

FIRST_RUN_OBJECTS = (
    "Ridge__x_only",
    "Ridge__mag_only",
    "Ridge__x_plus_mag",
    "HistGradientBoosting__x_only",
    "HistGradientBoosting__mag_only",
    "HistGradientBoosting__x_plus_mag",
    "RandomForest__x_only",
    "RandomForest__mag_only",
    "RandomForest__x_plus_mag",
    "SPSM baseline",
    "MAG-SPSM",
    "MAG-LINEAR",
)
DEEP_COLUMNS = {
    "SPSM baseline": "yhat_SPSM_baseline",
    "MAG-SPSM": "yhat_MAG_SPSM",
    "MAG-LINEAR": "yhat_magnitude_only_LR",
}
ENCODER_KEY = {"RPE1": "RPE1", "K562-essential": "K562"}
ENCODER_ARMS = {
    "z": ("frozen encoder z, forest head", "not_availability"),
    "zm": ("frozen encoder with magnitude [z; m], forest head", "partly_availability"),
    "C1": ("control C1, Gaussian features, boosting head", "null"),
    "C2": ("control C2, random projection of x, forest head", "availability"),
    "C3": ("control C3, row-shuffled z, boosting head", "not_availability"),
    "C4": ("control C4, [z; row-shuffled m], forest head", "not_availability"),
    "C5": ("control C5, [z; Gaussian scalars], forest head", "not_availability"),
}
INPUT_TYPE = {
    "Ridge__x_only": "expression only",
    "HistGradientBoosting__x_only": "expression only",
    "RandomForest__x_only": "expression only",
    "SPSM baseline": "expression only",
    "Ridge__mag_only": "magnitude summaries only",
    "HistGradientBoosting__mag_only": "magnitude summaries only",
    "RandomForest__mag_only": "magnitude summaries only",
    "MAG-LINEAR": "magnitude summaries only",
    "Ridge__x_plus_mag": "expression and magnitude summaries",
    "HistGradientBoosting__x_plus_mag": "expression and magnitude summaries",
    "RandomForest__x_plus_mag": "expression and magnitude summaries",
    "MAG-SPSM": "expression and magnitude summaries",
}
B2_RELEASE = "B2 negated own delta, release pseudobulk, 2,000 features"
B2_CELL = "B2 negated own delta, cell-level delta, 2,000 features"
MAX_FAMILIES = {
    "max |x|, release pseudobulk": ("release", False),
    "max |x| with the target coordinate set to zero, release pseudobulk": ("release", True),
    "max |x|, cell-level delta": ("cell", False),
    "max |x| with the target coordinate set to zero, cell-level delta": ("cell", True),
}
SPLIT_MEANING = {
    "availability": "the split is by whether the model's input contains the target coordinate",
    "summary_availability": "the input is four summaries of the 2,000 features; the target enters only "
    "through them, most directly through max |x|",
    "positive_control_release": "positive control for the twelve first-run objects: the target coordinate "
    "of the release pseudobulk they were scored on; zero where the target is not a feature, and its "
    "statistic set to 0 there by convention",
    "positive_control_cell": "positive control for the arms built on the cell-level delta (C2 and "
    "[z; m]): the target coordinate of that input; zero where the target is not a feature, and its "
    "statistic set to 0 there by convention",
    "partly_availability": "the magnitude part of the input is four summaries of the 2,000 features, so "
    "the target enters it only through them and its closer reference is the cell-level max |x| row; "
    "the encoder part sees most genes, so the split is only partly an availability split",
    "not_availability": "not an availability split: the encoder input contains most genes, "
    "including the target, whether or not it is one of the 2,000 features",
    "null": "the representation carries no information; a difference here is noise",
    "baseline": "trivial predictor computed without any model",
}


# Readings of the masking contrast and of the masked score, fixed before their values were
# computed (bioinf2027/plans/D1b.md, Section 12.1).
MASKING_CALLS = {
    "masking lowers the score": "part of the max-response secondary's external rank correlation comes "
    "from the target's own coordinate",
    "masking raises it": "where the target coordinate is the largest, it ranks constructs worse than the "
    "next-largest coordinate does; the secondary's external score does not rest on the target coordinate",
    "no change resolved": "no change resolved; the interval bounds the change; no statement that the "
    "score does not use the coordinate",
}
MASKED_SCORE_CALLS = {
    "positive": "max |x| ranks constructs above chance without the target coordinate; whether that is "
    "response or sampling depth is read with the partial given log pool size",
    "inconclusive": "no external rank association is resolved without the target coordinate",
    "negative": "a negative association without the target coordinate, reported as such",
    "undefined": "undefined",
}
# Readings of the RPE1 minus K562-essential difference of part (b) (Section 12.2, item 7).
RPE1_K562_CALLS = {
    "positive": "on shared target genes the control-cell mean rank-predicts the label more strongly on "
    "RPE1; the paper reports the cell lines separately",
    "negative": "on shared target genes the control-cell mean rank-predicts the label more weakly on "
    "RPE1; the paper reports the cell lines separately",
    "inconclusive": "no difference between the cell lines is resolved and the interval bounds it; no "
    "statement that they agree",
    "undefined": "undefined",
}
NO_MECHANISM_CLAIM = (
    "no D1b result tests mechanism for the twelve first-run objects, the encoder arm z, [z; m] or the "
    "controls C1 to C5; the masked max |x| score equals a masked MAX-LINEAR score for the stored fit "
    "and for any refit with the target zeroed whose slope stays positive, and such a refit belongs to D2"
)
COMPOSITION = (
    "no D1b result excludes a difference in composition between the subsets; B1, B2 native, the "
    "target-zeroed max |x|, the control-mean tertiles and the reduced construct sets are context, "
    "with no rule attached"
)


def seed_convention(meta: dict) -> str:
    """How the Spearman of a family is formed from its archived seeds."""
    group = meta["group"]
    if group == "archived object, first run":
        return "mean over archived seeds of the per-seed Spearman" if meta["n_seeds"] > 1 else "single archived run"
    if group == "archived object, single seed":
        return "single archived seed"
    if group.startswith("archived encoder arm"):
        return "Spearman of the prediction averaged over three seeds"
    return "no model"


VCC_BATCHES = 48
VCC_CONTROL_SHAPES: dict = {}
RELEASE_CHECKS: dict = {}
CELL_REVERSE_CHECKS: dict = {}


class ControlMeanUnavailable(RuntimeError):
    """Raised when a screen's control mean is not available from a checked source."""

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------- helpers


def few_clusters(n_target_genes) -> bool:
    try:
        return bool(int(n_target_genes) < FEW_CLUSTERS)
    except (TypeError, ValueError):
        return True


def _verdict(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    return "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"


def shard_folder() -> Path:
    return config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "variants" / "delta_hvg" / "shards"


def cell_count_file() -> Path:
    return config.storage_root() / "MADA" / "data" / "metadata" / "vcc" / "final_labels.parquet"


def prediction_files() -> dict[str, tuple[Path, Path]]:
    results = config.storage_root() / "IGA_MAIN" / "results"
    return {
        "RPE1": (
            results / "phase11b_classical_transfer" / "replogle_predictions_RPE1.csv",
            results / "phase10" / "transfer_run_RPE1" / "replogle_predictions.csv",
        ),
        "K562-essential": (
            results / "phase11b_classical_transfer" / "replogle_predictions.csv",
            results / "phase10" / "transfer_run" / "replogle_predictions.csv",
        ),
    }


def encoder_prediction_file() -> Path:
    return config.repo_root() / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz"


def release_bulk_file(name: str) -> Path:
    raw = config.storage_root() / "IGA_MAIN" / "data" / "replogle" / "raw"
    return raw / {"RPE1": "rpe1_normalized_bulk_01.h5ad", "K562-essential": "K562_essential_normalized_bulk_01.h5ad"}[name]


def vcc_annotation_file() -> Path:
    return config.storage_root() / "IGA_MAIN" / "data" / "vcc" / "train" / "adata_Training.h5ad"


def phase4_file(kind: str) -> Path:
    return config.repo_root() / "earlier_revision" / "phase4" / f"phase4_{kind}.json"


def largest_coordinate_flags(block: np.ndarray, feature: np.ndarray, selected: np.ndarray) -> dict:
    """max |x| with and without the target coordinate, and where the target stands at the maximum.

    ``feature`` holds each row's target position among the columns, used only
    where ``selected`` is true. The target is the unique largest coordinate when
    its absolute value is the maximum and no other coordinate reaches it; it ties
    when another coordinate reaches the same maximum. Zeroing the target changes
    max |x| exactly where it is the unique largest.
    """
    block = np.asarray(block, dtype=np.float64)
    rows = np.flatnonzero(selected)
    magnitude = np.abs(block)
    top = magnitude.max(axis=1)
    at_top = (magnitude == top[:, None]).sum(axis=1)
    target = np.full(block.shape[0], -np.inf)
    target[rows] = magnitude[rows, feature[rows]]
    on_top = selected & (target == top)
    zeroed = block.copy()
    zeroed[rows, feature[rows]] = 0.0
    zeroed_top = np.abs(zeroed).max(axis=1)
    changed = zeroed_top != top
    unique = on_top & (at_top == 1)
    if not np.array_equal(changed, unique):
        raise RuntimeError("max |x| changes on constructs where the target is not the unique largest coordinate")
    return {"max_abs": top, "max_abs_target_zeroed": zeroed_top, "unique_largest": unique,
            "ties_largest": on_top & (at_top > 1), "changed": changed}


def single_cell_file(name: str) -> Path:
    data = config.storage_root() / "IGA_MAIN" / "data" / "replogle" / "raw_singlecell"
    return data / {"RPE1": "rpe1_raw_singlecell_01.h5ad", "K562-essential": "K562_essential_raw_singlecell_01.h5ad"}[name]


def cell_level_reverse_check(name: str, constructs, target_symbol, target_ensg, selected) -> tuple[np.ndarray, dict]:
    """The reverse availability check on the cell-level input, from the screen's gene annotation.

    The cell-level input maps each of the 2,000 feature symbols to the first
    gene of the screen with that symbol
    (earlier_revision/pipeline/x_cell_accumulator.py:259-276). A construct
    outside the features is a case when its target identifier is the identifier
    of one of the mapped genes. Only the annotation is read, no expression value.
    """
    import h5py

    from bioinf2027 import singlecell

    with h5py.File(single_cell_file(name), "r") as handle:
        var = handle["var"]
        index_name = var.attrs["_index"]
        index_name = index_name.decode() if isinstance(index_name, bytes) else str(index_name)
        identifiers = singlecell.read_annotation(var, index_name)
        gene_name = singlecell.read_annotation(var, "gene_name")
    first: dict = {}
    for position, symbol in enumerate(gene_name.tolist()):
        first.setdefault(symbol, position)
    symbols = features.hvg_symbols()
    mapped_ids = {str(identifiers[first[s]]): str(s) for s in symbols.tolist() if s in first}
    cases = [
        {"construct": str(c), "target_symbol": str(t), "target_ensg": str(e), "mapped_as": mapped_ids[str(e)]}
        for c, t, e, sel in zip(constructs, target_symbol, target_ensg, selected)
        if not sel and str(e) in mapped_ids
    ]
    flag = np.isin(np.asarray(constructs).astype(str), [c["construct"] for c in cases])
    return flag, {
        "annotation": config.portable(single_cell_file(name)),
        "features_mapped": len(mapped_ids),
        "constructs_outside_the_features_with_the_target_mapped_under_another_symbol": len(cases),
        "reverse_cases": cases[:20],
        "mapping": "earlier_revision/pipeline/x_cell_accumulator.py:259-276",
    }


def release_inputs(name: str, constructs: np.ndarray, target_symbol: np.ndarray, target_ensg: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """The release pseudobulk in the 2,000-feature space, as the first-run objects saw it.

    The mapping repeats scripts/phase11b_classical_transfer/run.py:87-120: rows
    whose identifier contains 'non-targeting' are dropped, each selected gene is
    taken from the first column with its symbol, missing genes are zero filled,
    non-finite values set to 0 or +/-10, and every value clipped to +/-10. For
    each construct whose target gene is a feature, the column chosen for the
    target symbol must carry the target's own Ensembl identifier.
    """
    import h5py

    from bioinf2027 import singlecell

    symbols = features.hvg_symbols()
    with h5py.File(release_bulk_file(name), "r") as handle:
        obs, var = handle["obs"], handle["var"]
        ids = singlecell.read_annotation(obs, obs.attrs["_index"])
        gene_id = singlecell.read_annotation(var, var.attrs["_index"])
        gene_name = singlecell.read_annotation(var, "gene_name")
        matrix = np.asarray(handle["X"][()], dtype=np.float32)
    keep = np.char.find(ids.astype(str), "non-targeting") < 0
    ids, matrix = ids[keep], matrix[keep]
    first = {}
    for index, symbol in enumerate(gene_name.tolist()):
        first.setdefault(symbol, index)
    source = np.array([first.get(symbol, -1) for symbol in symbols.tolist()])
    mapped = np.zeros((matrix.shape[0], len(symbols)), dtype=np.float32)
    present = source >= 0
    mapped[:, present] = matrix[:, source[present]]
    mapped = np.clip(np.nan_to_num(mapped, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

    row = {c: i for i, c in enumerate(ids.tolist())}
    missing = [c for c in constructs.tolist() if c not in row]
    if missing:
        raise RuntimeError("%d endpoint constructs of %s are absent from the release file" % (len(missing), name))
    rows = np.array([row[c] for c in constructs.tolist()])
    position = {symbol: index for index, symbol in enumerate(symbols.tolist())}
    feature = np.array([position.get(s, -1) for s in target_symbol.tolist()])
    selected = feature >= 0
    own = np.zeros(len(constructs))
    own[selected] = mapped[rows[selected], feature[selected]]
    disagree = int(sum(
        1 for k in np.flatnonzero(selected)
        if gene_id[source[feature[k]]] != target_ensg[k]
    ))
    block = mapped[rows].astype(np.float64)
    flags = largest_coordinate_flags(block, feature, selected)
    # Reverse check (Section 10.2, item 3): a construct outside the features whose target
    # identifier is the identifier of the column mapped to one of the 2,000 features.
    mapped_ids = {str(gene_id[source[j]]): str(symbols[j]) for j in np.flatnonzero(present)}
    reverse = [
        {"construct": str(c), "target_symbol": str(t), "target_ensg": str(e), "mapped_as": mapped_ids[str(e)]}
        for c, t, e, sel in zip(constructs, target_symbol, target_ensg, selected)
        if not sel and str(e) in mapped_ids
    ]
    frame = pd.DataFrame(
        {
            "release_own_delta_zero_filled": own,
            "release_max_abs": flags["max_abs"],
            "release_max_abs_target_zeroed": flags["max_abs_target_zeroed"],
            "release_target_unique_largest": flags["unique_largest"],
            "release_target_ties_largest": flags["ties_largest"],
            "release_max_abs_at_clip": flags["max_abs"] >= CLIP,
            "release_target_mapped_under_another_symbol": np.isin(constructs, [r["construct"] for r in reverse]),
        }
    )
    check = {
        "file": config.portable(release_bulk_file(name)),
        "release_rows_without_controls": int(len(ids)),
        "selected_genes_mapped": int(present.sum()),
        "constructs_with_target_feature": int(selected.sum()),
        "target_column_with_another_ensembl_identifier": disagree,
        "constructs_outside_the_features_with_the_target_mapped_under_another_symbol": len(reverse),
        "reverse_cases": reverse[:20],
        "constructs_whose_max_abs_changes_when_the_target_is_zeroed": int(flags["changed"].sum()),
        "constructs_whose_target_is_the_unique_largest_coordinate": int(flags["unique_largest"].sum()),
        "constructs_whose_target_ties_another_coordinate_at_the_largest": int(flags["ties_largest"].sum()),
        "constructs_with_max_abs_at_the_clip": int((flags["max_abs"] >= CLIP).sum()),
        "values_at_the_clip_over_all_constructs": int((np.abs(block) >= CLIP).sum()),
        "passed": bool(disagree == 0),
        "mapping": "scripts/phase11b_classical_transfer/run.py:87-120",
    }
    if not check["passed"]:
        raise RuntimeError("release mapping places another gene at the target position on %s: %s" % (name, check))
    return frame, check


def vcc_target_identifiers(table: pd.DataFrame) -> tuple[dict, dict]:
    """Ensembl identifiers of the VCC target genes per split, from the VCC gene annotation."""
    import h5py

    from bioinf2027 import singlecell

    with h5py.File(vcc_annotation_file(), "r") as handle:
        var = handle["var"]
        symbols = singlecell.read_annotation(var, var.attrs["_index"])
        gene_id = singlecell.read_annotation(var, "gene_id")
    lookup = dict(zip(symbols.tolist(), gene_id.tolist()))
    out, unmatched = {}, {}
    for split in SPLITS:
        targets = sorted(set(table.loc[table["split"] == split, "target_gene"]))
        out[split] = {lookup[t] for t in targets if t in lookup}
        unmatched[split] = [t for t in targets if t not in lookup]
    report = {
        "annotation": config.portable(vcc_annotation_file()),
        "targets_per_split": {k: int(table.loc[table["split"] == k, "target_gene"].nunique()) for k in SPLITS},
        "targets_without_an_identifier": unmatched,
    }
    return out, report


# --------------------------------------------------------------------------- loaders


def vcc_rows() -> pd.DataFrame:
    """Every modelled VCC row with its label, keys, own delta and cell count."""
    symbols = features.hvg_symbols()
    position = {symbol: index for index, symbol in enumerate(symbols.tolist())}

    frames = []
    for split in SPLITS:
        for path in sorted(glob.glob(str(shard_folder() / split / "*.npz"))):
            store = np.load(path, allow_pickle=True)
            label = np.asarray(store["y"], dtype=np.float64)
            target = np.asarray(store["sample_ids"]).astype(str)
            batch = str(np.asarray(store["batch_name"]).astype(str)[0])
            column = np.array([position.get(symbol, -1) for symbol in target])
            own = np.full(len(label), np.nan)
            present = column >= 0
            if present.any():
                matrix = np.asarray(store["x"])[present]
                own[present] = matrix[np.arange(matrix.shape[0]), column[present]]
            frames.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "batch": batch,
                        "target_gene": target,
                        "label": label,
                        "own_delta": own,
                        "target_is_selected_feature": present,
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True)

    counts = pd.read_parquet(
        cell_count_file(), columns=["split", "target_gene", "batch", "n_perturbed_cells"]
    )
    counts = counts.assign(split=counts["split"].map({v: k for k, v in SPLITS.items()}))
    table = table.merge(
        counts, on=["split", "target_gene", "batch"], how="left", validate="many_to_one"
    )
    if table["n_perturbed_cells"].isna().any():
        raise RuntimeError(
            "cell-count join left %d rows unmatched" % int(table["n_perturbed_cells"].isna().sum())
        )
    observed = table.groupby("split").size().to_dict()
    if observed != SPLIT_ROWS:
        raise RuntimeError("modelled rows per split %s differ from %s" % (observed, SPLIT_ROWS))
    return table


def vcc_control_mean(table: pd.DataFrame) -> np.ndarray:
    """Control-cell mean expression of each row's target gene, in its own batch."""
    cache = config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "nt_baseline"
    per_batch = np.load(cache / "mean_log1p.npy", allow_pickle=True)
    batch_order = np.load(cache / "batch_order.npy", allow_pickle=True).astype(str)
    var_names = np.load(config.vcc_cache()["var_gene_names"], allow_pickle=True).astype(str)
    VCC_CONTROL_SHAPES.update(
        {
            "mean_log1p": list(per_batch.shape),
            "batch_order": int(batch_order.size),
            "var_gene_names": int(var_names.size),
        }
    )
    if per_batch.shape != (VCC_BATCHES, var_names.size) or batch_order.size != VCC_BATCHES:
        raise RuntimeError("the VCC control-mean matrix does not match the gene names and the batch count: %s" % VCC_CONTROL_SHAPES)

    batch_index = {name: index for index, name in enumerate(batch_order.tolist())}
    gene_index = {name: index for index, name in enumerate(var_names.tolist())}

    out = np.full(len(table), np.nan)
    for row, (batch, gene) in enumerate(zip(table["batch"], table["target_gene"])):
        bi = batch_index.get(batch)
        gi = gene_index.get(gene)
        if bi is not None and gi is not None:
            out[row] = per_batch[bi, gi]
    return out


def vcc_control_mean_noise(table: pd.DataFrame) -> dict:
    """How far a batch's control mean of a target gene varies, against the spread across target genes.

    From the per-batch control means (cache/vcc/nt_baseline/mean_log1p.npy): per target gene, the
    standard deviation over the 48 batches of its control mean, which bounds the sampling noise of a
    batch mean from above, against the standard deviation across target genes of the mean over
    batches (bioinf2027/plans/D1b.md, Section 12.2, item 12).
    """
    cache = config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "nt_baseline"
    per_batch = np.load(cache / "mean_log1p.npy", allow_pickle=True).astype(np.float64)
    var_names = np.load(config.vcc_cache()["var_gene_names"], allow_pickle=True).astype(str)
    gene_index = {name: index for index, name in enumerate(var_names.tolist())}
    out = {}
    for scope, genes in [("all splits", sorted(set(table["target_gene"])))] + [
        (split, sorted(set(table.loc[table["split"] == split, "target_gene"]))) for split in SPLITS
    ]:
        columns = [gene_index[g] for g in genes if g in gene_index]
        between = per_batch[:, columns].std(axis=0, ddof=1)
        across = float(per_batch[:, columns].mean(axis=0).std(ddof=1))
        out[scope] = {
            "target_genes": len(columns),
            "between_batch_sd_median": float(np.median(between)),
            "between_batch_sd_p90": float(np.quantile(between, 0.9)),
            "between_batch_sd_max": float(between.max()),
            "sd_across_target_genes": across,
            "median_between_batch_sd_over_across_gene_sd": float(np.median(between)) / across if across > 0 else float("nan"),
        }
    out["external"] = (
        "assumption, not computed: the pooled control sets hold 11,485 (RPE1) and 10,691 (K562-essential) "
        "cells, so the sampling noise of a pooled control mean is small against the spread across genes"
    )
    return out


def external_control_mean(name: str, screen: features.ScreenData) -> tuple[np.ndarray, dict]:
    """Pooled control mean per gene of an external screen, with the report of its checks."""
    if name in REPEATED_PASS:
        return _control_mean_from_repeated_pass(name, screen)
    return _control_mean_from_state(name, screen)


def _control_mean_from_repeated_pass(
    name: str, screen: features.ScreenData
) -> tuple[np.ndarray, dict]:
    """Control mean written by the repeated pass, accepted only if every check passed.

    The checks file decides. When it is absent the pass has not been run. When it
    exists its verdicts and observed differences are always copied into the
    report; the array is used only if all three checks passed and its digest
    equals the one the checks file records.
    """
    route = REPEATED_PASS[name]
    array_path = CACHE_DIR / route["array"]
    checks_path = OUT_DIR / route["checks"]
    report = {
        "screen": name,
        "route": "repeated pass over the single-cell file",
        "made_by": route["script"],
        "checks_file": config.portable(checks_path),
        "endpoint_control_pool": int(screen.endpoint["n_nt_pool"].iloc[0]),
        "identity_tolerance": IDENTITY_TOLERANCE,
        "identity_passed": False,
    }
    if not checks_path.exists():
        report["blocked_reason"] = "the repeated pass has not been run; run %s first" % route["script"]
        raise ControlMeanUnavailable(report["blocked_reason"], report)

    checks = json.loads(checks_path.read_text())
    parts = checks.get("checks", {})
    stored = parts.get("stored_index", {})
    prefix = parts.get("checkpoint_prefix", {})
    identity = parts.get("identity", {})
    report.update(
        {
            "checks_file_sha256": sha256(checks_path),
            "stored_index_passed": bool(stored.get("passed", False)),
            "checkpoint_prefix_passed": bool(prefix.get("passed", False)),
            "checkpoint_prefix_max_abs_difference_of_sums": prefix.get("max_abs_difference_of_sums"),
            "identity_passed": bool(identity.get("passed", False)),
            "identity_max_abs_diff": identity.get("identity_max_abs_diff"),
            "control_cells": identity.get("control_cells"),
            "n_genes": identity.get("n_genes"),
            "pass_error": checks.get("error"),
        }
    )
    every_check = (
        checks.get("all_passed", False)
        and report["stored_index_passed"]
        and report["checkpoint_prefix_passed"]
        and report["identity_passed"]
    )
    if not every_check:
        report["identity_passed"] = False
        report["blocked_reason"] = "a check of the repeated pass failed or did not complete, see the checks file"
        raise ControlMeanUnavailable(report["blocked_reason"], report)

    record = checks.get("control_mean_array") or {}
    if not array_path.exists() or "sha256" not in record:
        report["identity_passed"] = False
        report["blocked_reason"] = "the checks passed but the cached array is absent; run the pass again"
        raise ControlMeanUnavailable(report["blocked_reason"], report)
    digest = sha256(array_path)
    report["array_sha256"] = digest
    if digest != record["sha256"]:
        report["identity_passed"] = False
        report["blocked_reason"] = "the cached array does not match the digest in the checks file"
        raise ControlMeanUnavailable(report["blocked_reason"], report)

    mu_nt = np.load(array_path)
    if mu_nt.shape[0] != screen.x_native.shape[1] or not np.isfinite(mu_nt).all():
        raise RuntimeError("the cached control mean does not match the gene space of %s" % name)
    if report["control_cells"] != report["endpoint_control_pool"]:
        raise RuntimeError("the repeated pass and the endpoint disagree on the control pool size")
    return mu_nt, report


def _cells_against_pool(screen: features.ScreenData, counts: np.ndarray, source: str) -> dict:
    """Perturbed cells behind each construct's delta against the endpoint's n_perturbed_pool."""
    pool = screen.endpoint["n_perturbed_pool"].to_numpy(dtype=int)
    counts = np.asarray(counts, dtype=int)
    mismatch = np.flatnonzero(counts != pool)
    return {
        "source": source,
        "constructs": int(pool.size),
        "constructs_that_differ": int(mismatch.size),
        "first_differences": [
            {"construct": str(screen.constructs[i]), "cells": int(counts[i]), "n_perturbed_pool": int(pool[i])}
            for i in mismatch[:20]
        ],
        "rule": "exact equality construct by construct; a difference is reported and the definition of the external own delta is replaced by the measured one (bioinf2027/plans/D1b.md, Section 12.2, item 8)",
    }


def rpe1_pass_module():
    """The repeated-pass script, imported unchanged for its input paths and its group definition."""
    folder = str(Path(__file__).resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import make_d1b_rpe1_control_mean

    return make_d1b_rpe1_control_mean


def rpe1_cells_per_construct(screen: features.ScreenData) -> dict:
    """RPE1 perturbed cells per construct, by the recorded pass's group definition, from the annotations only.

    No expression value is read, so this is not a repetition of the pass
    (bioinf2027/plans/D1b.md, Section 12.2, item 8).
    """
    import h5py

    from bioinf2027 import singlecell

    module = rpe1_pass_module()
    paths = module.input_paths()
    constructs = np.array(sorted(pd.read_csv(paths["construct_list"])["construct"].astype(str)))
    stored_control = np.load(paths["stored_control_index"]).astype(np.int64)
    with h5py.File(paths["single_cell_counts"], "r") as handle:
        gene = singlecell.read_annotation(handle["obs"], "gene")
        transcript = singlecell.read_annotation(handle["obs"], "gene_transcript")
    _, codes, definition = module.pass_definition(gene, transcript, constructs, stored_control)
    counts = np.bincount(codes[codes > 0] - 1, minlength=constructs.size)
    by_construct = dict(zip(constructs.tolist(), counts.tolist()))
    report = _cells_against_pool(
        screen, np.array([by_construct.get(c, -1) for c in screen.constructs]),
        "gene and gene_transcript annotations with the pass's group definition; no expression value read",
    )
    report["pass_definition_check_passed"] = bool(definition.get("passed", False))
    return report


def _control_mean_from_state(name: str, screen: features.ScreenData) -> tuple[np.ndarray, dict]:
    """Pooled control mean per gene, recomputed from the retained accumulator state.

    The identity check runs in every run: the per-construct mean minus this
    control mean must reproduce the persisted native delta matrix. Every
    difference is collected and the largest is taken with numpy, so a single
    non-finite difference, a missing construct or a zero count fails the check.
    """
    sub, rel = ACCUMULATOR_STATE[name]
    state_path = config.storage_root() / sub / rel
    with open(state_path, "rb") as handle:
        state = pickle.load(handle)
    mu_nt = np.asarray(state["nt_sum"], dtype=np.float64) / float(state["nt_count"])

    differences, missing = [], []
    for row, construct in enumerate(screen.constructs):
        count = float(state["count"].get(construct, 0))
        if construct not in state["sum"] or count <= 0:
            missing.append(construct)
            differences.append(np.inf)
            continue
        delta = np.asarray(state["sum"][construct], dtype=np.float64) / count - mu_nt
        differences.append(np.max(np.abs(delta - screen.x_native[row])))
    worst = float(np.max(np.asarray(differences, dtype=np.float64)))
    state_counts = np.array([int(state["count"].get(c, -1)) for c in screen.constructs])
    cells_check = _cells_against_pool(screen, state_counts, "final accumulator state")
    report = {
        "screen": name,
        "route": "retained accumulator state, recomputed in this run",
        "state_path": config.portable(state_path),
        "state_nt_count": int(state["nt_count"]),
        "endpoint_control_pool": int(screen.endpoint["n_nt_pool"].iloc[0]),
        "state_total_cells": int(state["total_cells"]),
        "state_updates": int(state["updates"]),
        "n_genes": int(mu_nt.size),
        "constructs_missing_from_state": missing[:10],
        "identity_max_abs_diff": worst,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "formula": "mu_nt = nt_sum / nt_count, the plain mean, matching finalize(use_compensated=False)",
        "cells_per_construct_against_n_perturbed_pool": cells_check,
    }
    del state
    report["identity_passed"] = bool(
        np.isfinite(worst)
        and worst <= IDENTITY_TOLERANCE
        and not missing
        and np.isfinite(mu_nt).all()
        and report["state_nt_count"] == report["endpoint_control_pool"]
    )
    if not report["identity_passed"]:
        report["blocked_reason"] = (
            "the retained accumulator state does not reproduce the persisted delta matrix: its "
            "control pool holds %d cells against the %d the endpoint used"
            % (report["state_nt_count"], report["endpoint_control_pool"])
        )
        raise ControlMeanUnavailable(report["blocked_reason"], report)
    return mu_nt, report


# --------------------------------------------------------------------------- checks


def check_partial_reproduction() -> dict:
    """Reproduce the recorded rank-residual partial correlations."""
    matrices, labels, covariates = [], [], []
    counts = pd.read_parquet(
        cell_count_file(), columns=["split", "target_gene", "batch", "n_perturbed_cells"]
    )
    for split in ("train", "val"):
        store = np.load(
            config.repo_root() / "earlier_revision" / "phase0" / "m_tilde" / f"m_tilde_VCC_{split}.npz",
            allow_pickle=True,
        )
        matrices.append(np.asarray(store["m_raw"], dtype=np.float64))
        label, target, batch = [], [], []
        for path in sorted(glob.glob(str(shard_folder() / split / "*.npz"))):
            shard = np.load(path, allow_pickle=True)
            label.append(np.asarray(shard["y"], dtype=np.float64))
            target.append(np.asarray(shard["sample_ids"]).astype(str))
            batch.append(np.repeat(np.asarray(shard["batch_name"]).astype(str), len(shard["y"])))
        label = np.concatenate(label)
        keys = pd.DataFrame({"target_gene": np.concatenate(target), "batch": np.concatenate(batch)})
        sub = counts[counts["split"] == SPLITS[split]].set_index(["target_gene", "batch"])[
            "n_perturbed_cells"
        ]
        n_cells = np.array(
            [sub.get((g, b), np.nan) for g, b in zip(keys["target_gene"], keys["batch"])],
            dtype=np.float64,
        )
        labels.append(label)
        covariates.append(np.log1p(n_cells))

    matrix = np.vstack(matrices)
    label = np.concatenate(labels)
    covariate = np.concatenate(covariates)
    values = [partial_spearman(matrix[:, j], label, covariate) for j in range(matrix.shape[1])]
    passed = all(
        abs(got - expected) <= PARTIAL_TOLERANCE for got, expected in zip(values, RECORDED_PARTIAL)
    )
    report = {
        "recomputed": [round(v, 4) for v in values],
        "recorded": list(RECORDED_PARTIAL),
        "source_of_record": "earlier_revision/experiment_specs/phase4_prep.md:448 and :524",
        "tolerance": PARTIAL_TOLERANCE,
        "passed": bool(passed),
        "n_rows_pooled": int(matrix.shape[0]),
    }
    if not passed:
        raise RuntimeError("partial-correlation reproduction check failed: %s" % report)
    return report


def first_run_key(table: str, key: str) -> str:
    """The key a first-run value has in this run's tables.

    In part (b) the predictor called the knockdown fraction in the first run is
    renamed, and on VCC its 'all' row, which only ever covered the rows with an
    own delta, is relabelled accordingly.
    """
    if table != "b" or FIRST_RUN_RATIO not in key:
        return key
    scope, _predictor, subset, statistic = key.split("|")
    if scope.startswith("VCC") and subset == "all":
        subset = VCC_RATIO_ALL
    return "|".join((scope, RATIO, subset, statistic))


def compare_with_first_run(table: str, values: dict[str, float], allowed_new=None) -> dict:
    """Stop if a point estimate of the first run does not come out unchanged.

    The reference must hold the expected number of values for the table, every
    one of them must be present in this run, finite, and equal within the
    tolerance, and every value the first run did not have must be one of the
    declared additions.
    """
    reference = pd.read_csv(FIRST_RUN_FILE, float_precision="round_trip")
    reference = reference[reference["table"] == table]
    expected = {first_run_key(table, k): v for k, v in zip(reference["key"], reference["value"])}
    missing = sorted(key for key in expected if key not in values)
    differences = [abs(values[key] - expected[key]) for key in expected if key in values]
    not_finite = sum(1 for difference in differences if not np.isfinite(difference))
    worst = max((d for d in differences if np.isfinite(d)), default=0.0)
    new = sorted(key for key in values if key not in expected)
    undeclared = [key for key in new if allowed_new is None or not allowed_new(key)]
    report = {
        "table": table,
        "reference_values": len(expected),
        "reference_values_expected": FIRST_RUN_COUNTS.get(table),
        "values_compared": len(expected) - len(missing),
        "values_missing": len(missing),
        "values_not_finite": int(not_finite),
        "values_new": len(new),
        "values_new_undeclared": len(undeclared),
        "max_abs_difference": float(worst),
        "tolerance": FIRST_RUN_TOLERANCE,
    }
    report["passed"] = bool(
        len(expected) == FIRST_RUN_COUNTS.get(table)
        and not missing
        and not not_finite
        and not undeclared
        and worst <= FIRST_RUN_TOLERANCE
    )
    if not report["passed"]:
        raise RuntimeError(
            "point estimates differ from the first run for table %s: %s, missing %s, undeclared %s"
            % (table, report, missing[:5], undeclared[:5])
        )
    return report


# --------------------------------------------------------------------------- part (a)


def subsample_sizes_file() -> Path:
    return config.storage_root() / "IGA_MAIN" / "results_v4" / "harmonized_endpoint_v1" / "subsample_sizes.json"


def vcc_control_cells(table: pd.DataFrame) -> np.ndarray:
    """Control cells of each row's batch, the control side of its label."""
    per_batch = json.loads(subsample_sizes_file().read_text())["nt_per_batch"]
    counts = table["batch"].map(per_batch)
    if counts.isna().any():
        raise RuntimeError("a VCC batch has no recorded control count")
    return counts.to_numpy(dtype=float)


def part_a(
    table: pd.DataFrame, b: int = stats.BOOTSTRAP_B, control_cells: np.ndarray | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source association of the label with the own delta, raw and depth adjusted.

    ``b`` is the number of bootstrap replicates of every interval, the planned
    10,000 unless a test asks for fewer. With ``control_cells`` the table also
    holds a sensitivity adjusted for the log of the effective sample size
    n_p * n_c / (n_p + n_c) of the two-sample statistic.
    """
    rows, quintiles = [], []
    for split in SPLITS:
        whole = table[table["split"] == split]
        sub = whole[whole["target_is_selected_feature"]].copy()
        totals = {
            "n_rows_in_split": int(len(whole)),
            "n_target_genes_in_split": int(whole["target_gene"].nunique()),
        }
        predictor = -sub["own_delta"].to_numpy()
        label = sub["label"].to_numpy()
        groups = sub["target_gene"].to_numpy()
        covariate = np.log1p(sub["n_perturbed_cells"].to_numpy())

        plain = stats.cluster_bootstrap_corr(predictor, label, groups, b=b)
        adjusted = cluster_bootstrap_partial(predictor, label, covariate, groups, b=b)
        sensitivity = {}
        if control_cells is not None:
            n_p = sub["n_perturbed_cells"].to_numpy(dtype=float)
            n_c = control_cells[sub.index.to_numpy()]
            effective = cluster_bootstrap_partial(
                predictor, label, np.log(n_p * n_c / (n_p + n_c)), groups, b=b
            )
            sensitivity = {
                "partial_spearman_given_log_effective_size": effective.partial_spearman,
                "effective_size_lo": effective.partial_lo,
                "effective_size_hi": effective.partial_hi,
                "effective_size_n_skipped": effective.n_skipped,
            }
        rows.append(
            {
                "split": split,
                "subset": "rows_with_target_feature",
                "row_unit": ROW_UNIT_SOURCE,
                "n_rows": plain.n,
                "n_target_genes": plain.n_groups,
                **totals,
                "spearman": plain.spearman,
                "spearman_lo": plain.spearman_lo,
                "spearman_hi": plain.spearman_hi,
                "spearman_n_skipped": plain.n_skipped,
                "partial_spearman_given_log1p_n_cells": adjusted.partial_spearman,
                "partial_lo": adjusted.partial_lo,
                "partial_hi": adjusted.partial_hi,
                "partial_n_skipped": adjusted.n_skipped,
                **sensitivity,
                "few_clusters_descriptive": few_clusters(plain.n_groups),
                "bootstrap_b": plain.bootstrap_b,
                "bootstrap_seed": plain.bootstrap_seed,
            }
        )

        counts = sub["n_perturbed_cells"].to_numpy()
        edges = np.quantile(counts, np.linspace(0, 1, N_QUINTILES + 1))
        edges[-1] = np.inf
        for q in range(N_QUINTILES):
            mask = (counts >= edges[q]) & (counts < edges[q + 1])
            base = {
                "split": split,
                "quintile": q + 1,
                "cells_from": float(edges[q]),
                "cells_to": float(edges[q + 1]) if np.isfinite(edges[q + 1]) else float(counts.max()),
                "row_unit": ROW_UNIT_SOURCE,
                "n_rows": int(mask.sum()),
                "n_target_genes": int(len(set(groups[mask].tolist()))),
                **totals,
            }
            if mask.sum() < 10:
                quintiles.append({**base, "spearman": np.nan, "few_clusters_descriptive": True})
                continue
            estimate = stats.cluster_bootstrap_corr(predictor[mask], label[mask], groups[mask], b=b)
            quintiles.append(
                {
                    **base,
                    "spearman": estimate.spearman,
                    "spearman_lo": estimate.spearman_lo,
                    "spearman_hi": estimate.spearman_hi,
                    "spearman_n_skipped": estimate.n_skipped,
                    "few_clusters_descriptive": few_clusters(estimate.n_groups),
                    "bootstrap_b": estimate.bootstrap_b,
                    "bootstrap_seed": estimate.bootstrap_seed,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(quintiles)


# --------------------------------------------------------------------------- part (b)


def _readability_rows(
    scope: str,
    row_unit: str,
    label: np.ndarray,
    control_mean: np.ndarray,
    own_delta: np.ndarray,
    groups: np.ndarray,
    selected: np.ndarray,
    b: int = stats.BOOTSTRAP_B,
) -> list[dict]:
    """Control-cell mean and the ratio of own delta to it, against the label.

    The ratio is the own delta divided by the control mean, both on the log1p
    scale, and enters negated so that a larger drop gives a larger value. It is
    a ratio of log1p means, not the fraction of transcript lost.
    """
    totals = {
        "n_rows_in_scope": int(len(label)),
        "n_target_genes_in_scope": int(len(set(groups.tolist()))),
    }
    is_source = scope.startswith("VCC")
    comparability = (
        "not comparable with the rows of another scope: the VCC label is not size-controlled and the "
        "external label is; no cross-screen statement is drawn between them"
    )
    out = []
    for subset_name, mask in (
        ("all", np.ones(len(label), dtype=bool)),
        ("target_not_in_features", ~selected),
        ("target_in_features", selected),
    ):
        if mask.sum() < 10:
            continue
        has_mean = mask & np.isfinite(control_mean)
        if has_mean.sum() >= 10:
            estimate = stats.cluster_bootstrap_corr(
                control_mean[has_mean], label[has_mean], groups[has_mean], b=b
            )
            out.append(
                {
                    "scope": scope,
                    "predictor": "control_mean_expression",
                    "subset": subset_name,
                    "row_unit": row_unit,
                    "n_excluded_missing_control_mean": int(mask.sum() - has_mean.sum()),
                    "n_excluded_no_own_delta": 0,
                    "n_excluded_low_control_mean": 0,
                    "comparability": comparability,
                    **totals,
                    **stats.table_row(estimate),
                    "few_clusters_descriptive": few_clusters(estimate.n_groups),
                }
            )

        # On the source side the own delta exists only where the target gene is a
        # feature, since that cache holds the 2,000 selected columns only; there the
        # 'all' row therefore covers the rows with an own delta and is labelled so.
        has_delta = has_mean & np.isfinite(own_delta)
        usable = has_delta & (control_mean > CONTROL_MEAN_FLOOR)
        if usable.sum() >= 10:
            ratio = own_delta[usable] / control_mean[usable]
            estimate = stats.cluster_bootstrap_corr(-ratio, label[usable], groups[usable], b=b)
            out.append(
                {
                    "scope": scope,
                    "predictor": RATIO,
                    "subset": VCC_RATIO_ALL if (is_source and subset_name == "all") else subset_name,
                    "row_unit": row_unit,
                    "n_excluded_missing_control_mean": int(mask.sum() - has_mean.sum()),
                    "n_excluded_no_own_delta": int(has_mean.sum() - has_delta.sum()),
                    "n_excluded_low_control_mean": int(has_delta.sum() - usable.sum()),
                    "comparability": comparability,
                    **totals,
                    **stats.table_row(estimate),
                    "few_clusters_descriptive": few_clusters(estimate.n_groups),
                }
            )
    return out


def external_arrays(name: str, screen: features.ScreenData, mu_nt) -> pd.DataFrame:
    """Per-construct quantities of one external screen used by parts (b) and (c)."""
    endpoint = screen.endpoint
    symbols = features.hvg_symbols()
    position = {symbol: index for index, symbol in enumerate(symbols.tolist())}
    target_symbol = endpoint["target_symbol"].astype(str).to_numpy()
    column = endpoint["target_col"].astype(int).to_numpy()
    mapped_col = np.array([position.get(s, -1) for s in target_symbol])
    selected = mapped_col >= 0
    rows = np.arange(len(endpoint))
    own_mapped = np.zeros(len(endpoint))
    own_mapped[selected] = screen.x_mapped[rows[selected], mapped_col[selected]]
    cell = np.asarray(screen.x_mapped, dtype=np.float64)
    flags = largest_coordinate_flags(cell, mapped_col, selected)
    target_ensg = endpoint["target_ensg"].astype(str).to_numpy()
    release, RELEASE_CHECKS[name] = release_inputs(name, screen.constructs, target_symbol, target_ensg)
    reverse_flag, CELL_REVERSE_CHECKS[name] = cell_level_reverse_check(
        name, screen.constructs, target_symbol, target_ensg, selected
    )
    CELL_REVERSE_CHECKS[name].update(
        {
            "constructs_whose_max_abs_changes_when_the_target_is_zeroed": int(flags["changed"].sum()),
            "constructs_whose_target_ties_another_coordinate_at_the_largest": int(flags["ties_largest"].sum()),
            "constructs_with_max_abs_at_the_clip": int((flags["max_abs"] >= CLIP).sum()),
        }
    )
    frame = pd.DataFrame(
        {
            "screen": name,
            "construct": screen.constructs,
            "target_symbol": target_symbol,
            "target_ensg": endpoint["target_ensg"].astype(str).to_numpy(),
            "target_is_selected_feature": selected,
            "y_primary": endpoint["y_primary"].to_numpy(dtype=np.float64),
            "n_perturbed_pool": endpoint["n_perturbed_pool"].to_numpy(dtype=np.float64),
            "control_mean_target_gene": (mu_nt[column] if mu_nt is not None else np.nan),
            "own_delta_native": screen.x_native[rows, column].astype(np.float64),
            "own_delta_mapped_zero_filled": own_mapped,
            "cell_max_abs": flags["max_abs"],
            "cell_max_abs_target_zeroed": flags["max_abs_target_zeroed"],
            "cell_target_unique_largest": flags["unique_largest"],
            "cell_target_ties_largest": flags["ties_largest"],
            "cell_target_mapped_under_another_symbol": reverse_flag,
        }
    )
    return pd.concat([frame, release.reset_index(drop=True)], axis=1)


def part_b(table, control, screens, control_means, b=stats.BOOTSTRAP_B) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        mask = (table["split"] == split).to_numpy()
        rows.extend(
            _readability_rows(
                f"VCC {split}",
                ROW_UNIT_SOURCE,
                table["label"].to_numpy()[mask],
                control[mask],
                table["own_delta"].to_numpy()[mask],
                table["target_gene"].to_numpy()[mask],
                table["target_is_selected_feature"].to_numpy()[mask],
                b=b,
            )
        )
    for name in SCREENS:
        mu_nt = control_means[name]
        if mu_nt is None:
            continue
        arrays = external_arrays(name, screens[name], mu_nt)
        rows.extend(
            _readability_rows(
                name,
                ROW_UNIT_EXTERNAL,
                arrays["y_primary"].to_numpy(),
                arrays["control_mean_target_gene"].to_numpy(),
                arrays["own_delta_native"].to_numpy(),
                arrays["target_ensg"].to_numpy(),
                arrays["target_is_selected_feature"].to_numpy(),
                b=b,
            )
        )
    return pd.DataFrame(rows)


def part_b_vcc_depth_adjusted(table, control, b=stats.BOOTSTRAP_B) -> pd.DataFrame:
    """Control-cell mean against the label on VCC, adjusted for cell count and by quintile."""
    rows = []
    for split in SPLITS:
        sub = table[table["split"] == split]
        mean = control[(table["split"] == split).to_numpy()]
        keep = np.isfinite(mean)
        label = sub["label"].to_numpy()[keep]
        groups = sub["target_gene"].to_numpy()[keep]
        counts = sub["n_perturbed_cells"].to_numpy()[keep]
        mean = mean[keep]
        adjusted = cluster_bootstrap_partial(mean, label, np.log1p(counts), groups, b=b)
        rows.append(
            {
                "split": split,
                "statistic": "partial_spearman_given_log1p_n_cells",
                "quintile": "all",
                "row_unit": ROW_UNIT_SOURCE,
                "n_rows": adjusted.n,
                "n_target_genes": adjusted.n_groups,
                "estimate": adjusted.partial_spearman,
                "lo": adjusted.partial_lo,
                "hi": adjusted.partial_hi,
                "n_skipped": adjusted.n_skipped,
                "few_clusters_descriptive": few_clusters(adjusted.n_groups),
            }
        )
        edges = np.quantile(counts, np.linspace(0, 1, N_QUINTILES + 1))
        edges[-1] = np.inf
        for q in range(N_QUINTILES):
            m = (counts >= edges[q]) & (counts < edges[q + 1])
            if m.sum() < 10:
                continue
            e = stats.cluster_bootstrap_corr(mean[m], label[m], groups[m], b=b)
            rows.append(
                {
                    "split": split,
                    "statistic": "spearman_within_cell_count_quintile",
                    "quintile": q + 1,
                    "cells_from": float(edges[q]),
                    "cells_to": float(edges[q + 1]) if np.isfinite(edges[q + 1]) else float(counts.max()),
                    "row_unit": ROW_UNIT_SOURCE,
                    "n_rows": e.n,
                    "n_target_genes": e.n_groups,
                    "estimate": e.spearman,
                    "lo": e.spearman_lo,
                    "hi": e.spearman_hi,
                    "n_skipped": e.n_skipped,
                    "few_clusters_descriptive": few_clusters(e.n_groups),
                }
            )
    return pd.DataFrame(rows)


def part_b_external_own_delta(per_screen: dict, b=stats.BOOTSTRAP_B) -> pd.DataFrame:
    """What perturbed cells add over control cells on the external screens."""
    rows = []
    for name, arrays in per_screen.items():
        if not np.isfinite(arrays["control_mean_target_gene"]).all():
            continue
        y = arrays["y_primary"].to_numpy()
        mean = arrays["control_mean_target_gene"].to_numpy()
        own = -arrays["own_delta_native"].to_numpy()
        groups = arrays["target_ensg"].to_numpy()
        selected = arrays["target_is_selected_feature"].to_numpy()
        for subset, mask in (("all", np.ones(len(y), dtype=bool)), ("target_not_in_features", ~selected)):
            specs = [
                (
                    "spearman_control_mean_vs_negated_own_delta",
                    stats.cluster_bootstrap_corr(mean[mask], own[mask], groups[mask], b=b),
                ),
                (
                    "partial_spearman_label_negated_own_delta_given_control_mean",
                    cluster_bootstrap_partial(own[mask], y[mask], mean[mask], groups[mask], b=b),
                ),
                (
                    "partial_spearman_label_control_mean_given_negated_own_delta",
                    cluster_bootstrap_partial(mean[mask], y[mask], own[mask], groups[mask], b=b),
                ),
            ]
            for statistic, e in specs:
                is_partial = hasattr(e, "partial_spearman")
                rows.append(
                    {
                        "screen": name,
                        "subset": subset,
                        "statistic": statistic,
                        "row_unit": ROW_UNIT_EXTERNAL,
                        "n_rows": e.n,
                        "n_target_genes": e.n_groups,
                        "estimate": e.partial_spearman if is_partial else e.spearman,
                        "lo": e.partial_lo if is_partial else e.spearman_lo,
                        "hi": e.partial_hi if is_partial else e.spearman_hi,
                        "n_skipped": e.n_skipped,
                    }
                )
    return pd.DataFrame(rows)


def cross_screen(per_screen: dict, b=stats.BOOTSTRAP_B) -> pd.DataFrame:
    """Gene-level checks on the target genes shared by the two external screens."""
    if set(per_screen) != set(SCREENS):
        return pd.DataFrame()
    genes = {}
    for name, arrays in per_screen.items():
        genes[name] = arrays.groupby("target_ensg").agg(
            label=("y_primary", "median"), control_mean=("control_mean_target_gene", "first")
        )
    shared = genes["RPE1"].join(genes["K562-essential"], lsuffix="_rpe1", rsuffix="_k562", how="inner")
    pairs = [
        ("label RPE1 against label K562-essential", "label_rpe1", "label_k562"),
        ("control mean RPE1 against label K562-essential", "control_mean_rpe1", "label_k562"),
        ("control mean K562-essential against label RPE1", "control_mean_k562", "label_rpe1"),
        ("control mean RPE1 against control mean K562-essential", "control_mean_rpe1", "control_mean_k562"),
        ("control mean RPE1 against label RPE1, shared genes", "control_mean_rpe1", "label_rpe1"),
        ("control mean K562-essential against label K562-essential, shared genes", "control_mean_k562", "label_k562"),
    ]
    rows = []
    groups = shared.index.to_numpy().astype(str)
    for name, first, second in pairs:
        x, y = shared[first].to_numpy(dtype=float), shared[second].to_numpy(dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        e = stats.cluster_bootstrap_corr(x, y, groups, b=b)
        rows.append(
            {
                "pair": name,
                "row_unit": "target gene, label as the median over its constructs",
                **stats.table_row(e),
            }
        )
    return pd.DataFrame(rows)


def rpe1_against_k562(per_screen: dict, b=stats.BOOTSTRAP_B) -> dict:
    """The control-mean association on RPE1 against K562-essential, on shared target genes."""
    if set(per_screen) != set(SCREENS):
        return {"computed": False, "reason": "a control mean is not available for both screens"}
    inputs = []
    for name in SCREENS:
        a = per_screen[name]
        inputs.append(
            (
                a["control_mean_target_gene"].to_numpy(),
                a["y_primary"].to_numpy(),
                a["target_ensg"].to_numpy(),
            )
        )
    result = stats.paired_screen_difference(inputs[0], inputs[1], b=b)
    difference = result["first_minus_second"]
    call = _verdict(difference["lo"], difference["hi"])
    result.update(
        {
            "computed": True,
            "first": {**result["first"], "screen": "RPE1"},
            "second": {**result["second"], "screen": "K562-essential"},
            "unit": "constructs of the target genes shared by both screens; one joint draw of shared genes",
            "call_sign": call,
            "reading_by_call": RPE1_K562_CALLS,
            "reading_for_this_call": RPE1_K562_CALLS[call],
            "reading": "a replication in a second cell line whose direction was foreseeable from the "
            "K562-essential value and from RPE1's own-delta association; not an independent confirmation",
        }
    )
    return result


def vcc_batch_sensitivity(table, control, b=stats.BOOTSTRAP_B) -> pd.DataFrame:
    """Parts (a) and (b) on VCC with whole batches resampled instead of target genes."""
    rows = []
    for split in SPLITS:
        whole = table[table["split"] == split]
        mean = control[(table["split"] == split).to_numpy()]
        feature_rows = whole["target_is_selected_feature"].to_numpy()
        predictor = -whole["own_delta"].to_numpy()[feature_rows]
        label_f = whole["label"].to_numpy()[feature_rows]
        batch_f = whole["batch"].to_numpy()[feature_rows]
        cov_f = np.log1p(whole["n_perturbed_cells"].to_numpy()[feature_rows])
        keep = np.isfinite(mean)
        specs = [
            (
                "a", "spearman_label_negated_own_delta", "rows_with_target_feature",
                stats.cluster_bootstrap_corr(predictor, label_f, batch_f, b=b),
            ),
            (
                "a", "partial_spearman_given_log1p_n_cells", "rows_with_target_feature",
                cluster_bootstrap_partial(predictor, label_f, cov_f, batch_f, b=b),
            ),
            (
                "b", "spearman_label_control_mean", "all",
                stats.cluster_bootstrap_corr(
                    mean[keep], whole["label"].to_numpy()[keep], whole["batch"].to_numpy()[keep], b=b
                ),
            ),
        ]
        for part, statistic, subset, e in specs:
            is_partial = hasattr(e, "partial_spearman")
            rows.append(
                {
                    "split": split,
                    "part": part,
                    "statistic": statistic,
                    "subset": subset,
                    "resampling_unit": "batch",
                    "n_rows": e.n,
                    "n_batches": e.n_groups,
                    "estimate": e.partial_spearman if is_partial else e.spearman,
                    "lo": e.partial_lo if is_partial else e.spearman_lo,
                    "hi": e.partial_hi if is_partial else e.spearman_hi,
                    "n_skipped": e.n_skipped,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- part (c)


def external_objects(name: str, arrays: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Prediction columns and the families built on them, for one external screen.

    Families: the twelve archived objects of the first run, their single seeds,
    the archived encoder arm and its controls, and the trivial baselines. A
    family with several columns is scored by the mean of its per-column statistic.
    """
    classical_path, deep_path = prediction_files()[name]
    keys = arrays[["construct"]]
    classical = pd.read_csv(classical_path)
    deep = pd.read_csv(deep_path)
    merged = keys.merge(
        classical, left_on="construct", right_on="obs_index", how="left", validate="one_to_one"
    ).merge(
        deep, left_on="construct", right_on="obs_index", how="left", suffixes=("", "_deep"),
        validate="one_to_one",
    )
    if len(merged) != len(arrays):
        raise RuntimeError("prediction join changed the row count for %s" % name)

    columns: list[np.ndarray] = []
    families: dict[str, dict] = {}

    def add(values) -> int:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(arrays),) or not np.isfinite(values).all():
            raise RuntimeError("a prediction column of %s is missing or not finite" % name)
        columns.append(values)
        return len(columns) - 1

    for object_name in FIRST_RUN_OBJECTS:
        if object_name in DEEP_COLUMNS:
            index = [add(merged[DEEP_COLUMNS[object_name]])]
            seed_names = ["single archived run"]
        else:
            seed_columns = sorted(c for c in merged.columns if c.startswith(object_name + "__seed"))
            if not seed_columns:
                raise RuntimeError("no archived predictions for %s on %s" % (object_name, name))
            index = [add(merged[c]) for c in seed_columns]
            seed_names = [c.split("__")[-1] for c in seed_columns]
        meaning = "summary_availability" if INPUT_TYPE[object_name] == "magnitude summaries only" else "availability"
        families[object_name] = {
            "group": "archived object, first run",
            "columns": index,
            "n_seeds": len(index),
            "split_meaning": meaning,
            "input": "release pseudobulk",
        }
        if len(index) > 1:
            for column, seed_name in zip(index, seed_names):
                families[f"{object_name} | {seed_name}"] = {
                    "group": "archived object, single seed",
                    "columns": [column],
                    "n_seeds": 1,
                    "split_meaning": meaning,
                    "input": "release pseudobulk",
                }

    store = np.load(encoder_prediction_file(), allow_pickle=True)
    key = ENCODER_KEY[name]
    stored_constructs = np.asarray(store[f"{key}_constructs"]).astype(str)
    matrix = np.asarray(store[f"{key}_P"], dtype=np.float64)
    arms = np.asarray(store["arms"]).astype(str)
    order = {c: i for i, c in enumerate(stored_constructs.tolist())}
    if set(order) != set(arrays["construct"]):
        raise RuntimeError("encoder-arm constructs differ from the endpoint constructs on %s" % name)
    rows = np.array([order[c] for c in arrays["construct"]])
    for j, arm in enumerate(arms.tolist()):
        title, meaning = ENCODER_ARMS[arm]
        families[title] = {
            "group": "archived encoder arm or control, seed-averaged",
            "columns": [add(matrix[rows, j])],
            "n_seeds": 3,
            "split_meaning": meaning,
            "input": "cell-level delta and frozen encoder",
            "arm": arm,
        }

    baseline_values = {
        "B0 cells per construct": (arrays["n_perturbed_pool"].to_numpy(), "baseline", "endpoint"),
        "B1 control-cell mean of the target gene": (
            arrays["control_mean_target_gene"].to_numpy(), "baseline", "control cells"),
        "B2 negated own delta, native space": (
            -arrays["own_delta_native"].to_numpy(), "baseline", "cell-level delta, native genes"),
        B2_RELEASE: (-arrays["release_own_delta_zero_filled"].to_numpy(), "positive_control_release", "release pseudobulk"),
        B2_CELL: (-arrays["own_delta_mapped_zero_filled"].to_numpy(), "positive_control_cell", "cell-level delta"),
    }
    for title, (source, zeroed) in MAX_FAMILIES.items():
        column = f"{source}_max_abs" + ("_target_zeroed" if zeroed else "")
        baseline_values[title] = (
            arrays[column].to_numpy(),
            "summary_availability",
            "release pseudobulk" if source == "release" else "cell-level delta",
        )
    for title, (values, meaning, source) in baseline_values.items():
        if not np.isfinite(values).all():
            continue
        families[title] = {
            "group": "trivial baseline" if not title.startswith("max") else "predictor without a model",
            "columns": [add(values)],
            "n_seeds": 1,
            "split_meaning": meaning,
            "input": source,
        }
    return np.column_stack(columns), families


def _joint(matrix, arrays, families, mask, covariate=None, b=stats.BOOTSTRAP_B, in_subset=None):
    names = list(families)
    if in_subset is None:
        in_subset = arrays["target_is_selected_feature"].to_numpy()
    return names, stats.joint_cluster_bootstrap_subsets(
        matrix[mask],
        arrays["y_primary"].to_numpy()[mask],
        arrays["target_ensg"].to_numpy()[mask],
        np.asarray(in_subset, dtype=bool)[mask],
        {n: families[n]["columns"] for n in names},
        b=b,
        return_draws=True,
        covariate=None if covariate is None else covariate[mask],
        zero_if_constant={n for n in names if n in (B2_RELEASE, B2_CELL)},
    )


def _family_mean(names, point, draws, members, column=3) -> dict:
    """Mean of a statistic over a set of families, inside every replicate."""
    index = [names.index(n) for n in members]
    values = draws[:, index, column]
    complete = np.isfinite(values).all(axis=1)
    mean = np.where(complete, values.mean(axis=1), np.nan)
    lo, hi, skipped = stats.percentile_interval(mean)
    return {"estimate": float(point[index, column].mean()), "lo": lo, "hi": hi, "n_skipped": skipped}


def _difference_row(name, set_name, family, meta, inside, outside, d, role="secondary"):
    half = (d.hi - d.lo) / 2 if np.isfinite(d.hi - d.lo) else np.nan
    return {
        "screen": name,
        "construct_set": set_name,
        "object": family,
        "group": meta["group"],
        "input": meta.get("input", ""),
        "n_seeds": meta["n_seeds"],
        "seed_convention": seed_convention(meta),
        "role": role,
        "row_unit": ROW_UNIT_EXTERNAL,
        "n_rows_in": inside.n,
        "n_target_genes_in": inside.n_groups,
        "n_rows_out": outside.n,
        "n_target_genes_out": outside.n_groups,
        "spearman_in": inside.estimate,
        "spearman_out": outside.estimate,
        "in_minus_out": d.estimate,
        "in_minus_out_lo": d.lo,
        "in_minus_out_hi": d.hi,
        "interval_half_width": half,
        "difference_detectable_at_80_percent_power": half * POWER_FACTOR if np.isfinite(half) else np.nan,
        "verdict": _verdict(d.lo, d.hi),
        "split_meaning": SPLIT_MEANING[meta["split_meaning"]],
        "bootstrap_b": d.bootstrap_b,
        "bootstrap_seed": d.bootstrap_seed,
        "n_skipped": d.n_skipped,
    }


def _summary_rows(name, set_name, names, point, draws, by_key, b, label=""):
    """Post-hoc means of the in-minus-out difference over the twelve and by input type."""
    groups = {"mean of the twelve first-run objects": list(FIRST_RUN_OBJECTS)}
    for kind in ("expression only", "magnitude summaries only", "expression and magnitude summaries"):
        members = [o for o in FIRST_RUN_OBJECTS if INPUT_TYPE.get(o) == kind]
        if members:
            groups[f"mean of the {len(members)} first-run objects on {kind}"] = members
    rows = []
    first = FIRST_RUN_OBJECTS[0]
    for title, members in groups.items():
        mean = _family_mean(names, point, draws, members)
        half = (mean["hi"] - mean["lo"]) / 2
        rows.append(
            {
                "screen": name,
                "construct_set": set_name,
                "object": title + label,
                "group": "post-hoc summary",
                "input": "release pseudobulk",
                "n_seeds": np.nan,
                "role": "post-hoc summary",
                "row_unit": ROW_UNIT_EXTERNAL,
                "n_rows_in": by_key[(first, "in")].n,
                "n_target_genes_in": by_key[(first, "in")].n_groups,
                "n_rows_out": by_key[(first, "out")].n,
                "n_target_genes_out": by_key[(first, "out")].n_groups,
                "spearman_in": _family_mean(names, point, draws, members, 1)["estimate"],
                "spearman_out": _family_mean(names, point, draws, members, 2)["estimate"],
                "in_minus_out": mean["estimate"],
                "in_minus_out_lo": mean["lo"],
                "in_minus_out_hi": mean["hi"],
                "interval_half_width": half,
                "difference_detectable_at_80_percent_power": half * POWER_FACTOR,
                "verdict": _verdict(mean["lo"], mean["hi"]),
                "split_meaning": "mean over objects scored on the release pseudobulk; post hoc, its point value follows from the first-run table",
                "bootstrap_b": b,
                "bootstrap_seed": stats.BOOTSTRAP_SEED,
                "n_skipped": mean["n_skipped"],
            }
        )
    return rows


def d1_changed_constructs(name: str) -> int:
    """D1's count of constructs whose target is the strict, unique largest mapped coordinate."""
    table = pd.read_csv(D1_EXTERNAL_SUMMARY)
    return int(table.loc[table["screen"] == name, "mapped_argmax_constructs"].iloc[0])


def masking_contrast(name, arrays, matrix, families, names, point, draws, b=stats.BOOTSTRAP_B) -> list[dict]:
    """Plain minus target-zeroed max |x|, for each input, on all and on in-feature constructs.

    The contrast is taken inside the replicates of the joint draw of all eligible
    constructs, so it is paired with every other statistic of that draw
    (bioinf2027/plans/D1b.md, Sections 10.1 and 12.1). The partial Spearman of
    each max |x| with the label given log(n_perturbed_pool) comes from a joint
    draw with the same seed over the same constructs.
    """
    log_pool = np.log(arrays["n_perturbed_pool"].to_numpy(dtype=float))
    max_families = {title: families[title] for title in MAX_FAMILIES}
    _, (partial_results, _, _) = _joint(
        matrix, arrays, max_families, np.ones(len(arrays), dtype=bool), covariate=log_pool, b=b
    )
    partial = {(r.family, r.subset): r for r in partial_results}
    expected = d1_changed_constructs(name)
    rows = []
    for source, input_name in (("cell", "cell-level delta"), ("release", "release pseudobulk")):
        plain = next(t for t, (src, zeroed) in MAX_FAMILIES.items() if src == source and not zeroed)
        masked = next(t for t, (src, zeroed) in MAX_FAMILIES.items() if src == source and zeroed)
        ip, im = names.index(plain), names.index(masked)
        changed = int((arrays[f"{source}_max_abs"] != arrays[f"{source}_max_abs_target_zeroed"]).sum())
        if source == "cell" and changed != expected:
            raise RuntimeError(
                "%s: %d constructs change max |x| when the target is zeroed, D1 counts %d with the target as the "
                "unique largest coordinate" % (name, changed, expected)
            )
        for subset, column in (("all", 0), ("target_in_features", 1)):
            difference = draws[:, ip, column] - draws[:, im, column]
            lo, hi, skipped = stats.percentile_interval(difference)
            m_lo, m_hi, _ = stats.percentile_interval(draws[:, im, column])
            p_lo, p_hi, _ = stats.percentile_interval(draws[:, ip, column])
            call = "masking lowers the score" if lo > 0 else "masking raises it" if hi < 0 else "no change resolved"
            primary = source == "cell" and subset == "all"
            half = (hi - lo) / 2
            subset_key = "all" if subset == "all" else "in"
            row = {
                "screen": name,
                "input": input_name,
                "construct_set": "all eligible constructs",
                "subset": subset,
                "role": "primary, fixed before its values were computed" if primary else "secondary, not adjusted for multiplicity",
                "n_constructs": int(len(arrays)) if subset == "all" else int(arrays["target_is_selected_feature"].sum()),
                "n_target_genes": int(arrays["target_ensg"].nunique()) if subset == "all"
                else int(arrays.loc[arrays["target_is_selected_feature"], "target_ensg"].nunique()),
                "spearman_max_abs": float(point[ip, column]),
                "spearman_max_abs_lo": p_lo,
                "spearman_max_abs_hi": p_hi,
                "spearman_max_abs_target_zeroed": float(point[im, column]),
                "spearman_max_abs_target_zeroed_lo": m_lo,
                "spearman_max_abs_target_zeroed_hi": m_hi,
                "plain_minus_target_zeroed": float(point[ip, column] - point[im, column]),
                "difference_lo": lo,
                "difference_hi": hi,
                "interval_half_width": half,
                "difference_detectable_at_80_percent_power": half * POWER_FACTOR,
                "call": call,
                "reading": MASKING_CALLS[call],
                "partial_max_abs_given_log_pool": partial[(plain, subset_key)].estimate,
                "partial_max_abs_given_log_pool_lo": partial[(plain, subset_key)].lo,
                "partial_max_abs_given_log_pool_hi": partial[(plain, subset_key)].hi,
                "partial_target_zeroed_given_log_pool": partial[(masked, subset_key)].estimate,
                "partial_target_zeroed_given_log_pool_lo": partial[(masked, subset_key)].lo,
                "partial_target_zeroed_given_log_pool_hi": partial[(masked, subset_key)].hi,
                "constructs_whose_max_abs_changes": changed,
                "constructs_whose_target_ties_the_largest": int(arrays[f"{source}_target_ties_largest"].sum()),
                "constructs_with_max_abs_at_the_clip": int(arrays[f"{source}_max_abs"].ge(CLIP).sum()) if source == "release" else 0,
                "d1_count_of_unique_largest_targets": expected if source == "cell" else np.nan,
                "bootstrap_b": int(b),
                "bootstrap_seed": stats.BOOTSTRAP_SEED,
                "n_skipped": skipped,
            }
            if primary:
                score_call = _verdict(m_lo, m_hi)
                row["masked_score_call"] = score_call
                row["masked_score_reading"] = MASKED_SCORE_CALLS[score_call]
                partial_call = _verdict(partial[(masked, "all")].lo, partial[(masked, "all")].hi)
                row["masked_score_partial_call"] = partial_call
            rows.append(row)
    return rows


def part_c(per_screen, vcc_ids, vcc_symbols=None, b=stats.BOOTSTRAP_B):
    """Transfer by target availability, with baselines, post-hoc summaries and the extra tables."""
    estimates, differences, partials, composition, overlap, tertiles = [], [], [], [], [], []
    masking, by_identifier = [], []
    for name in SCREENS:
        arrays = per_screen[name]
        matrix, families = external_objects(name, arrays)
        selected = arrays["target_is_selected_feature"].to_numpy()
        ensg = arrays["target_ensg"].to_numpy()
        in_train = np.isin(ensg, sorted(vcc_ids["train"]))
        in_val = np.isin(ensg, sorted(vcc_ids["val"]))
        if not in_train.any():
            raise RuntimeError("no target of %s matches a VCC training target by identifier" % name)
        construct_sets = {
            "all eligible constructs": np.ones(len(arrays), dtype=bool),
            "without VCC training targets": ~in_train,
            "without VCC training or validation targets": ~(in_train | in_val),
        }
        for set_name, mask in construct_sets.items():
            names, (results, point, draws) = _joint(matrix, arrays, families, mask, b=b)
            by_key = {(r.family, r.subset): r for r in results}
            if set_name == "all eligible constructs":
                masking.extend(masking_contrast(name, arrays, matrix, families, names, point, draws, b=b))
            for family in names:
                meta = families[family]
                for subset in ("all", "in", "out"):
                    e = by_key[(family, subset)]
                    estimates.append(
                        {
                            "screen": name,
                            "construct_set": set_name,
                            "object": family,
                            "group": meta["group"],
                            "input": meta.get("input", ""),
                            "n_seeds": meta["n_seeds"],
                            "seed_convention": seed_convention(meta),
                            "subset": SUBSET_NAMES[subset],
                            "row_unit": ROW_UNIT_EXTERNAL,
                            "n_rows": e.n,
                            "n_target_genes": e.n_groups,
                            "spearman": e.estimate,
                            "spearman_lo": e.lo,
                            "spearman_hi": e.hi,
                            "set_to_zero_where_constant": family in (B2_RELEASE, B2_CELL),
                            "verdict": _verdict(e.lo, e.hi),
                            "split_meaning": SPLIT_MEANING[meta["split_meaning"]],
                            "bootstrap_b": e.bootstrap_b,
                            "bootstrap_seed": e.bootstrap_seed,
                            "n_skipped": e.n_skipped,
                        }
                    )
                differences.append(
                    _difference_row(
                        name, set_name, family, meta,
                        by_key[(family, "in")], by_key[(family, "out")], by_key[(family, "in_minus_out")],
                    )
                )
            differences.extend(_summary_rows(name, set_name, names, point, draws, by_key, b))

        # Declared addition (Section 12.2, item 10): where the reverse check finds a target mapped
        # into the 2,000 under another symbol, the split by identifier beside the split by symbol.
        for source, column in (("release pseudobulk", "release_target_mapped_under_another_symbol"),
                               ("cell-level delta", "cell_target_mapped_under_another_symbol")):
            extra = arrays[column].to_numpy(dtype=bool)
            if not extra.any():
                continue
            kept = {k: v for k, v in families.items() if v["group"] != "archived object, single seed"
                    and (v.get("input", "").startswith("release") == source.startswith("release"))}
            names_i, (results_i, _, _) = _joint(matrix, arrays, kept, np.ones(len(arrays), dtype=bool),
                                                b=b, in_subset=selected | extra)
            for r in results_i:
                by_identifier.append({"screen": name, "split_by": "identifier, " + source, "object": r.family,
                                      "subset": SUBSET_NAMES.get(r.subset, r.subset), "n_rows": r.n,
                                      "n_target_genes": r.n_groups, "spearman": r.estimate, "lo": r.lo,
                                      "hi": r.hi, "verdict": _verdict(r.lo, r.hi), "n_skipped": r.n_skipped,
                                      "role": "declared addition; put to the author under gate G3"})

        control = arrays["control_mean_target_gene"].to_numpy()
        if np.isfinite(control).all():
            partial_families = {
                k: v
                for k, v in families.items()
                if k != "B1 control-cell mean of the target gene"
                and v["group"] != "archived object, single seed"
                and k not in (B2_RELEASE, B2_CELL)
            }
            names, (results, _, _) = _joint(
                matrix, arrays, partial_families, np.ones(len(arrays), dtype=bool), covariate=control, b=b
            )
            for r in results:
                partials.append(
                    {
                        "screen": name,
                        "object": r.family,
                        "group": partial_families[r.family]["group"],
                        "input": partial_families[r.family].get("input", ""),
                        "subset": SUBSET_NAMES.get(r.subset, r.subset),
                        "statistic": "partial_spearman_given_control_mean",
                        "n_rows": r.n,
                        "n_target_genes": r.n_groups,
                        "estimate": r.estimate,
                        "lo": r.lo,
                        "hi": r.hi,
                        "verdict": _verdict(r.lo, r.hi),
                        "n_skipped": r.n_skipped,
                    }
                )
            # Post hoc: the split inside tertiles of the control-cell mean, a property of the
            # gene, so that a difference in label range between the subsets is held apart.
            edges = np.quantile(control, [1 / 3, 2 / 3])
            tertile = np.digitize(control, edges)
            tertile_families = {k: v for k, v in families.items() if v["group"] != "archived object, single seed"}
            for t in range(3):
                mask = tertile == t
                try:
                    names, (results, point, draws) = _joint(matrix, arrays, tertile_families, mask, b=b)
                except ValueError as error:
                    tertiles.append({"screen": name, "control_mean_tertile": t + 1, "error": str(error)})
                    continue
                by_key = {(r.family, r.subset): r for r in results}
                for family in names:
                    row = _difference_row(
                        name, f"control-mean tertile {t + 1}", family, tertile_families[family],
                        by_key[(family, "in")], by_key[(family, "out")], by_key[(family, "in_minus_out")],
                        role="post hoc",
                    )
                    row["control_mean_tertile"] = t + 1
                    row["control_mean_from"] = float(control[mask].min())
                    row["control_mean_to"] = float(control[mask].max())
                    tertiles.append(row)
                for row in _summary_rows(name, f"control-mean tertile {t + 1}", names, point, draws, by_key, b):
                    row["control_mean_tertile"] = t + 1
                    tertiles.append(row)

        subsets = (
            ("all", np.ones(len(arrays), dtype=bool)),
            ("target_in_features", selected),
            ("target_not_in_features", ~selected),
        )
        for subset, mask in subsets:
            part = arrays[mask]
            entry = {
                "screen": name,
                "subset": subset,
                "n_constructs": int(len(part)),
                "n_target_genes": int(part["target_ensg"].nunique()),
            }
            for column in (
                "y_primary", "control_mean_target_gene", "n_perturbed_pool", "own_delta_native",
                "release_max_abs", "cell_max_abs",
            ):
                values = part[column].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                for q, label in ((0.25, "q25"), (0.5, "median"), (0.75, "q75")):
                    entry[f"{column}_{label}"] = float(np.quantile(values, q)) if values.size else np.nan
            composition.append(entry)

            ids = part["target_ensg"].to_numpy()
            membership = {
                "vcc_training_target": np.isin(ids, sorted(vcc_ids["train"])),
                "vcc_validation_target": np.isin(ids, sorted(vcc_ids["val"])),
                "vcc_test_target": np.isin(ids, sorted(vcc_ids["test"])),
            }
            membership["not_a_vcc_target"] = ~(
                membership["vcc_training_target"]
                | membership["vcc_validation_target"]
                | membership["vcc_test_target"]
            )
            row = {
                "screen": name,
                "subset": subset,
                "matched_on": "Ensembl identifier",
                "n_constructs": int(len(part)),
                "n_target_genes": int(part["target_ensg"].nunique()),
            }
            for key, flags in membership.items():
                row[f"{key}_constructs"] = int(flags.sum())
                row[f"{key}_target_genes"] = int(part.loc[flags, "target_ensg"].nunique())
            if vcc_symbols is not None:
                symbol = part["target_symbol"].to_numpy().astype(str)
                for key, split in (("vcc_training_target", "train"), ("vcc_validation_target", "val"), ("vcc_test_target", "test")):
                    hit = np.isin(symbol, sorted(vcc_symbols[split]))
                    row[f"{key}_constructs_by_symbol"] = int(hit.sum())
                    row[f"{key}_target_genes_by_symbol"] = int(part.loc[hit, "target_ensg"].nunique())
            overlap.append(row)
    return (
        pd.DataFrame(estimates),
        pd.DataFrame(differences),
        pd.DataFrame(partials),
        pd.DataFrame(composition),
        pd.DataFrame(overlap),
        pd.DataFrame(tertiles),
        pd.DataFrame(masking),
        pd.DataFrame(by_identifier),
    )


def max_response_by_argmax(per_screen, b=stats.BOOTSTRAP_B) -> pd.DataFrame:
    """max |x| against the label among in-feature constructs, by whether the target is the argmax."""
    rows = []
    for name, arrays in per_screen.items():
        selected = arrays["target_is_selected_feature"].to_numpy()
        for source in ("release", "cell"):
            unique = arrays[f"{source}_target_unique_largest"].to_numpy()
            ties = arrays[f"{source}_target_ties_largest"].to_numpy()
            for label, mask in (("argmax is the target coordinate alone", selected & unique),
                                ("the target ties another coordinate at the largest", selected & ties),
                                ("argmax is another coordinate", selected & ~unique & ~ties)):
                if mask.sum() < 3:
                    rows.append({"screen": name, "input": "release pseudobulk" if source == "release" else "cell-level delta",
                                 "subset": "target_in_features, " + label, "row_unit": ROW_UNIT_EXTERNAL,
                                 "n_rows": int(mask.sum()), "role": "post hoc, descriptive; too few constructs for a correlation"})
                    continue
                e = stats.cluster_bootstrap_corr(
                    arrays[f"{source}_max_abs"].to_numpy()[mask],
                    arrays["y_primary"].to_numpy()[mask],
                    arrays["target_ensg"].to_numpy()[mask],
                    b=b,
                )
                rows.append(
                    {
                        "screen": name,
                        "input": "release pseudobulk" if source == "release" else "cell-level delta",
                        "subset": "target_in_features, " + label,
                        "row_unit": ROW_UNIT_EXTERNAL,
                        **stats.table_row(e),
                        "role": "post hoc, descriptive; not paired with the other subset",
                    }
                )
    return pd.DataFrame(rows)


def check_archived_reproduction(per_screen) -> dict:
    """Before any split: the encoder arms and the max-response secondary reproduce their stored values."""
    primary = json.loads(phase4_file("primary").read_text())
    secondary = json.loads(phase4_file("secondary").read_text())
    report = {"encoder_arms": {}, "max_linear": {}}
    worst = 0.0
    for name in SCREENS:
        arrays = per_screen[name]
        matrix, families = external_objects(name, arrays)
        y = arrays["y_primary"].to_numpy()
        key = ENCODER_KEY[name]
        for family, meta in families.items():
            arm = meta.get("arm")
            if arm is None:
                continue
            rho = float(sps.spearmanr(matrix[:, meta["columns"][0]], y).statistic)
            stored = float(primary["screens"][key]["arm_rho"][arm]["rho"])
            report["encoder_arms"][f"{name}|{arm}"] = {"recomputed": rho, "stored": stored}
            worst = max(worst, abs(rho - stored))
        rho = float(sps.spearmanr(arrays["cell_max_abs"].to_numpy(), y).statistic)
        stored = float(secondary["screens"][key]["rho"]["MAX_LINEAR"]["rho"])
        report["max_linear"][name] = {"recomputed_spearman_of_max_abs_cell_level": rho, "stored_max_linear": stored}
        if abs(rho - stored) > MAX_LINEAR_TOLERANCE:
            raise RuntimeError("max |x| does not reproduce the stored max-response secondary on %s: %s" % (name, report["max_linear"][name]))
    report["encoder_max_abs_difference"] = worst
    report["encoder_tolerance"] = ENCODER_TOLERANCE
    report["max_linear_tolerance"] = MAX_LINEAR_TOLERANCE
    report["sources"] = [config.portable(phase4_file("primary")), config.portable(phase4_file("secondary"))]
    report["passed"] = bool(worst <= ENCODER_TOLERANCE)
    if not report["passed"]:
        raise RuntimeError("encoder arms do not reproduce the stored primary: %s" % report)
    return report


# --------------------------------------------------------------------------- main


def inventory(screens_paths: dict) -> dict:
    """Portable path, size and digest of every input this script reads."""
    root = config.storage_root()
    vcc = root / "IGA_MAIN" / "cache" / "vcc"
    repo = config.repo_root()
    records = {
        "vcc_feature_shards": folder_record(glob.glob(str(shard_folder() / "*" / "*.npz")), shard_folder()),
        "vcc_cell_counts": file_record(cell_count_file()),
        "vcc_control_means_per_batch": file_record(vcc / "nt_baseline" / "mean_log1p.npy"),
        "vcc_control_batch_order": file_record(vcc / "nt_baseline" / "batch_order.npy"),
        "vcc_gene_names": file_record(config.vcc_cache()["var_gene_names"]),
        "vcc_selected_genes": file_record(config.vcc_cache()["selected_genes"]),
        "vcc_magnitude_train": file_record(repo / "earlier_revision" / "phase0" / "m_tilde" / "m_tilde_VCC_train.npz"),
        "vcc_magnitude_val": file_record(repo / "earlier_revision" / "phase0" / "m_tilde" / "m_tilde_VCC_val.npz"),
        "encoder_arm_predictions": file_record(encoder_prediction_file()),
        "phase4_primary": file_record(phase4_file("primary")),
        "phase4_secondary": file_record(phase4_file("secondary")),
        "vcc_gene_annotation": file_record(vcc_annotation_file()),
        "vcc_control_cells_per_batch": file_record(subsample_sizes_file()),
        "first_run_reference": file_record(FIRST_RUN_FILE),
        # D1's count of constructs whose target is the unique largest mapped coordinate,
        # which gates the masking contrast (Section 12.1).
        "d1_external_summary": file_record(D1_EXTERNAL_SUMMARY),
    }
    for name, paths in screens_paths.items():
        for key in ("endpoint", "x_native", "x_mapped"):
            records[f"{name}_{key}"] = file_record(paths[key])
        records[f"{name}_release_pseudobulk"] = file_record(release_bulk_file(name))
        classical, deep = prediction_files()[name]
        records[f"{name}_classical_predictions"] = file_record(classical)
        records[f"{name}_deep_predictions"] = file_record(deep)
    for name, (sub, rel) in ACCUMULATOR_STATE.items():
        records[f"{name}_accumulator_state"] = file_record(root / sub / rel)
    for name, route in REPEATED_PASS.items():
        records[f"{name}_control_mean_checks"] = file_record(OUT_DIR / route["checks"])
        records[f"{name}_control_mean_checks"]["written_at_commit"] = RPE1_PASS_COMMIT
        records[f"{name}_control_mean_array"] = file_record(CACHE_DIR / route["array"])
    # The six inputs of the repeated RPE1 pass (Section 7.2), each compared with the digest its
    # checks file records, since that file is not rewritten (Section 12.2, item 9).
    recorded = json.loads((OUT_DIR / REPEATED_PASS["RPE1"]["checks"]).read_text())["input_digests"]
    for key, path in rpe1_pass_module().input_paths().items():
        record = file_record(path)
        record["digest_in_the_checks_file"] = recorded.get(f"{key}_sha256")
        record["matches_the_checks_file"] = bool(record.get("sha256") == record["digest_in_the_checks_file"])
        if not record["matches_the_checks_file"]:
            raise RuntimeError("the RPE1 pass input %s differs from the digest in its checks file" % key)
        records[f"RPE1_pass_{key}"] = record
    records["K562-essential_single_cell_counts"] = file_record(single_cell_file("K562-essential"))
    return records


def write_outputs(outputs: dict, provenance: dict) -> None:
    """Write every table through a temporary name, then the provenance last, with every table's digest."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, frame in outputs.items():
        temporary = OUT_DIR / (name + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, OUT_DIR / name)
        records[name] = file_record(OUT_DIR / name)
    provenance["output_records"] = records
    temporary = OUT_DIR / "D1b_provenance.json.tmp"
    temporary.write_text(json.dumps(strict_json(provenance), indent=1, allow_nan=False) + "\n")
    os.replace(temporary, OUT_DIR / "D1b_provenance.json")


def main() -> None:
    print("check 1: reproducing the recorded partial correlations", flush=True)
    partial_report = check_partial_reproduction()
    print("  recomputed", partial_report["recomputed"], "PASS", flush=True)

    table = vcc_rows()
    screens = {name: features.load_screen(name) for name in SCREENS}
    inputs = inventory(config.screen_paths())

    print("check 2: control means of the external screens", flush=True)
    control_means, identity_reports = {}, {}
    for name in SCREENS:
        try:
            control_means[name], identity_reports[name] = external_control_mean(name, screens[name])
        except ControlMeanUnavailable as error:
            control_means[name], identity_reports[name] = None, error.report
            print(f"  {name}: BLOCKED, {error}", flush=True)
            continue
        report = identity_reports[name]
        print(
            f"  {name}: {report['route']}, identity max abs diff "
            f"{report['identity_max_abs_diff']:.2e} PASS",
            flush=True,
        )

    per_screen = {name: external_arrays(name, screens[name], control_means[name]) for name in SCREENS}
    available = {n: a for n, a in per_screen.items() if control_means[n] is not None}
    vcc_ids, vcc_id_report = vcc_target_identifiers(table)
    vcc_symbols = {split: set(table.loc[table["split"] == split, "target_gene"]) for split in SPLITS}
    control = vcc_control_mean(table)
    noise = vcc_control_mean_noise(table)
    cells_check = {
        "RPE1": rpe1_cells_per_construct(screens["RPE1"]),
        "K562-essential": identity_reports["K562-essential"].get("cells_per_construct_against_n_perturbed_pool"),
    }
    unchanged = []

    print("check 3: archived encoder arms and the max-response secondary", flush=True)
    reproduction = check_archived_reproduction(per_screen)
    print("  encoder arms max abs difference %.2e PASS" % reproduction["encoder_max_abs_difference"], flush=True)

    print("part (a)", flush=True)
    a_rows, a_quintiles = part_a(table, control_cells=vcc_control_cells(table))
    values = {}
    for row in a_rows.itertuples():
        values[f"{row.split}|spearman"] = row.spearman
        values[f"{row.split}|partial_spearman_given_log1p_n_cells"] = row.partial_spearman_given_log1p_n_cells
    unchanged.append(compare_with_first_run("a", values))
    unchanged.append(
        compare_with_first_run(
            "a_quintiles",
            {f"{r.split}|{int(r.quintile)}|spearman": r.spearman for r in a_quintiles.itertuples()},
        )
    )

    print("part (b)", flush=True)
    b_rows = part_b(table, control, screens, control_means)
    values = {}
    for row in b_rows.itertuples():
        values[f"{row.scope}|{row.predictor}|{row.subset}|spearman"] = row.spearman
        values[f"{row.scope}|{row.predictor}|{row.subset}|pearson"] = row.pearson
    unchanged.append(
        compare_with_first_run("b", values, allowed_new=lambda key: key.startswith("RPE1|"))
    )
    b_vcc_adjusted = part_b_vcc_depth_adjusted(table, control)
    b_external = part_b_external_own_delta(available)
    b_cross = cross_screen(available)
    b_rpe1 = rpe1_against_k562(available)
    batch_sensitivity = vcc_batch_sensitivity(table, control)

    print("part (c)", flush=True)
    (c_rows, c_differences, c_partials, c_composition, c_overlap, c_tertiles, c_masking,
     c_identifier) = part_c(per_screen, vcc_ids, vcc_symbols)
    c_argmax = max_response_by_argmax(per_screen)
    first_set = c_rows[
        (c_rows["construct_set"] == "all eligible constructs") & c_rows["object"].isin(FIRST_RUN_OBJECTS)
    ]
    unchanged.append(
        compare_with_first_run(
            "c",
            {f"{r.screen}|{r.object}|{r.subset}|spearman": r.spearman for r in first_set.itertuples()},
        )
    )

    outputs = {
        "D1b_a_vcc_depth_adjusted.csv": a_rows,
        "D1b_a_quintiles.csv": a_quintiles,
        "D1b_b_control_readability.csv": b_rows,
        "D1b_b_vcc_depth_adjusted.csv": b_vcc_adjusted,
        "D1b_b_external_own_delta_and_control_mean.csv": b_external,
        "D1b_b_cross_screen.csv": b_cross,
        "D1b_b_external_per_construct.csv": pd.concat(per_screen.values(), ignore_index=True),
        "D1b_c_transfer_by_availability.csv": c_rows,
        "D1b_c_in_minus_out.csv": c_differences,
        "D1b_c_partial_given_control_mean.csv": c_partials,
        "D1b_c_subset_composition.csv": c_composition,
        "D1b_c_vcc_target_overlap.csv": c_overlap,
        "D1b_c_in_minus_out_by_control_mean_tertile.csv": c_tertiles,
        "D1b_c_max_response_by_argmax.csv": c_argmax,
        "D1b_vcc_batch_cluster_sensitivity.csv": batch_sensitivity,
        "D1b_c_masking_contrast.csv": c_masking,
    }
    if len(c_identifier):
        outputs["D1b_c_transfer_by_identifier_availability.csv"] = c_identifier
    own_delta_external = (
        "native cell-level delta at the target gene's column: mean log1p over all of the construct's "
        "eligible cells minus the mean log1p over the whole pooled control set, not the per-draw "
        "subsample the label uses"
    )
    differing = {n: c["constructs_that_differ"] for n, c in cells_check.items() if c and c["constructs_that_differ"]}
    if differing:
        own_delta_external += "; measured: the cells behind the delta differ from n_perturbed_pool for %s constructs (checks, cells_per_construct)" % differing
    provenance = {
        "plan": "bioinf2027/plans/D1b.md, Sections 1 to 12: the addendum of Section 7 and the changes after the four plan reviews, Sections 8 to 12",
        "paths_relative_to": "$" + config.STORAGE_VARIABLE + " for data-root files; the repository root for every other path",
        "inputs": inputs,
        "outputs": sorted(outputs) + ["D1b_provenance.json"],
        "checks": {
            "partial_correlation_reproduction": partial_report,
            "control_mean_identity": identity_reports,
            "archived_values_reproduced_before_any_split": reproduction,
            "release_mapping": RELEASE_CHECKS,
            "cell_level_reverse_availability": CELL_REVERSE_CHECKS,
            "reverse_availability_cases_put_to_the_author_under_G3": {
                n: {"release": RELEASE_CHECKS[n]["constructs_outside_the_features_with_the_target_mapped_under_another_symbol"],
                    "cell_level": CELL_REVERSE_CHECKS[n]["constructs_outside_the_features_with_the_target_mapped_under_another_symbol"]}
                for n in SCREENS
            },
            "cells_per_construct": cells_check,
            "vcc_control_mean_shapes": VCC_CONTROL_SHAPES,
            "vcc_target_identifiers": vcc_id_report,
            "point_estimates_unchanged_from_first_run": {
                "reference": config.portable(FIRST_RUN_FILE),
                "first_run_commit": "bd78b5e",
                "renamed_in_this_run": {
                    FIRST_RUN_RATIO: RATIO,
                    "VCC rows labelled 'all' for that predictor": VCC_RATIO_ALL,
                },
                "tables": unchanged,
            },
        },
        "rpe1_against_k562": b_rpe1,
        "rpe1_pass": {
            "script": REPEATED_PASS["RPE1"]["script"],
            "checks_file_written_at_commit": RPE1_PASS_COMMIT,
            "script_left_unchanged_after_its_run": True,
            "run_time": "bioinf2027/runs/D1b.json, the run of that script",
        },
        "vcc_control_mean_noise": noise,
        "masking_contrast": {
            "primary": "plain minus target-zeroed max |x|, cell-level input, all eligible constructs, subset all, one per screen; fixed before its values were computed (Section 12.1)",
            "secondary": "the release input and the in-feature subset, not adjusted for multiplicity",
            "readings": MASKING_CALLS,
            "masked_score_readings": MASKED_SCORE_CALLS,
            "changed_constructs_check": "on the cell-level input the constructs whose max |x| changes must equal D1's mapped_argmax_constructs (bioinf2027/diag/D1_target_in_features_external.csv); the run stops otherwise",
        },
        "mechanism": NO_MECHANISM_CLAIM,
        "composition": COMPOSITION,
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
        "thresholds": {
            "control_mean_floor": CONTROL_MEAN_FLOOR,
            "identity_tolerance": IDENTITY_TOLERANCE,
            "partial_tolerance": PARTIAL_TOLERANCE,
            "first_run_tolerance": FIRST_RUN_TOLERANCE,
            "few_clusters_descriptive_below_target_genes": FEW_CLUSTERS,
            "power_factor_for_the_detectable_difference": POWER_FACTOR,
            "encoder_tolerance": ENCODER_TOLERANCE,
            "max_linear_tolerance": MAX_LINEAR_TOLERANCE,
        },
        "bootstrap": {
            "b": stats.BOOTSTRAP_B,
            "seed": stats.BOOTSTRAP_SEED,
            "unit": "target gene, except the batch sensitivity table, which resamples batches",
            "part_c": "one joint draw per replicate for every object, baseline and subset; a family "
            "with several archived seeds is scored by the mean of its per-seed Spearman inside the "
            "replicate; the family-level statistic is the mean in-minus-out difference over the "
            "twelve first-run objects inside the replicate",
            "multiplicity": "intervals are not adjusted for the number of comparisons; part (c) is post hoc, "
            "including its family-level means, whose point values follow from the first-run table, except the "
            "primary masking contrast, one per screen, fixed before its values were computed (Section 12.1)",
            "constant_predictor_convention": "a predictor that is constant on a subset of a replicate "
            "(the zero-filled target coordinate outside the features) has its statistic set to 0 there",
        },
        "sample_sizes": {
            "n_rows": "rows used by the statistic; a row is named in row_unit",
            "n_target_genes": "distinct target genes among those rows",
            "totals": "n_rows_in_split, n_rows_in_scope and their target-gene counterparts give "
            "the size of the whole split or screen",
            "selection": "on VCC the rows with an own delta are those whose target gene is one of the "
            "2,000 features; the features were chosen by variance across training-row deltas, so the "
            "training rows with a feature target are selected partly on the size of their own delta, "
            "and the three splits do not estimate one quantity",
        },
        "definitions": {
            "control_mean": {
                "VCC": "mean log1p expression of the target gene over the control cells of the row's batch, the cells the label uses",
                "external": "mean log1p expression of the target gene over the whole pooled control set, the population the label's control draws are taken from, not any one draw",
            },
            RATIO: "own delta divided by the control mean, both on the log1p scale, negated; a ratio of log1p means, not the fraction of transcript lost",
            "own_delta": {
                "VCC": "delta-HVG cache value at the target gene's feature position: mean log1p over the row's perturbed cells minus the mean log1p over that batch's control cells, same cells as the label; defined only where the target gene is one of the 2,000 features",
                "external": own_delta_external,
            },
            "split_meaning": SPLIT_MEANING,
            "seed_convention": "named per row; no in-minus-out difference is compared across conventions",
            "archived_objects": "all were trained on the all-cells, batch-local source label of the submitted work; intervals are conditional on the archived predictions",
        },
    }
    write_outputs(outputs, provenance)
    print("\nwrote %d tables and the provenance file to %s" % (len(outputs), config.portable(OUT_DIR)))


if __name__ == "__main__":
    main()
