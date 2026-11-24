"""Path and configuration resolution for the revision analyses.

The data root is resolved in priority order:
  1. environment variable IGA_STORAGE_ROOT
  2. a file ``storage_root.txt`` at the repository root

No path is hard coded. Resolution fails loudly rather than guessing, so a run
can never silently read a different storage tree.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "storage_root",
    "portable",
    "screen_paths",
    "vcc_cache",
    "BOOTSTRAP_B",
    "BOOTSTRAP_SEED",
    "STORAGE_VARIABLE",
]

# Every cluster-bootstrap interval of the revision analyses: replicates and seed.
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20270301

# Name of the variable that holds the data root. Recorded paths are written
# relative to it, never as a location on one machine.
STORAGE_VARIABLE = "IGA_STORAGE_ROOT"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "storage_root.txt").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("repository root not found above %s" % here)


def storage_root() -> Path:
    value = os.environ.get("IGA_STORAGE_ROOT")
    if not value:
        marker = repo_root() / "storage_root.txt"
        if marker.exists():
            value = marker.read_text().strip()
    if not value:
        raise RuntimeError(
            "IGA_STORAGE_ROOT is unset and storage_root.txt was not found. "
            "Set one of them to the absolute path of the project data root."
        )
    root = Path(value).resolve()
    if not root.is_dir():
        raise RuntimeError("configured storage root does not exist: %s" % root)
    return root


def portable(path: str | os.PathLike) -> str:
    """Express a path so that it can be recorded in a tracked file.

    A path inside the data root is written as ``$IGA_STORAGE_ROOT/<relative>``
    and a path inside the repository as its repository-relative form. Any
    other location is refused, so a provenance record can never carry a
    machine-specific path.
    """
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(storage_root())
    except ValueError:
        pass
    else:
        if relative == Path("."):
            return "$%s" % STORAGE_VARIABLE
        return "$%s/%s" % (STORAGE_VARIABLE, relative.as_posix())
    try:
        return resolved.relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        raise ValueError(
            "path is outside both the data root and the repository; refusing to record it"
        ) from None


def vcc_cache() -> dict[str, Path]:
    """Frozen VCC feature-namespace artifacts."""
    cache = storage_root() / "IGA_MAIN" / "cache" / "vcc"
    return {
        "var_gene_names": cache / "var_gene_names.npy",
        "selected_genes": cache / "variants" / "delta_hvg" / "selected_genes.npy",
    }


def screen_paths() -> dict[str, dict[str, Path]]:
    """Per-screen endpoint, eligibility and cell-level expression artifacts."""
    root = storage_root()
    results = root / "IGA_MAIN" / "results_v4"
    encoder = root / "IGA_GENEFORMER_EXT"
    return {
        "RPE1": {
            "endpoint": results / "harmonized_endpoint_v1" / "rpe1_endpoint_by_construct.csv",
            "eligibility": results / "harmonized_endpoint_v1" / "rpe1_eligibility.csv",
            "x_native": encoder / "x_cell" / "x_cell_rpe1_native.npz",
            "x_mapped": encoder / "x_cell" / "x_cell_rpe1_mapped.npz",
        },
        "K562-essential": {
            "endpoint": results / "harmonized_endpoint_k562_essential_v1" / "endpoint_by_construct.csv",
            "eligibility": results / "harmonized_endpoint_k562_essential_v1" / "eligibility.csv",
            "x_native": encoder / "k562_essential" / "x_cell" / "x_cell_k562_native.npz",
            "x_mapped": encoder / "k562_essential" / "x_cell" / "x_cell_k562_1903_mapped.npy",
        },
    }
