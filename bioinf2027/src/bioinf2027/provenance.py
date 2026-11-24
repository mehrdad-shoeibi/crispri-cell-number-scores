"""Provenance records: what each output was computed from, byte for byte.

Every input is recorded with its portable path (see
:func:`bioinf2027.config.portable`), its size and its SHA-256 digest. A folder
of inputs is recorded file by file, each file with its path relative to the
folder, its size and its digest, together with the total size and one digest
over the sorted listing.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from . import config

__all__ = ["file_record", "folder_record", "sha256", "strict_json"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    """Portable path, size and digest of one file, or a note that it is absent."""
    path = Path(path)
    if not path.is_file():
        return {"path": config.portable(path), "exists": False}
    return {"path": config.portable(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def folder_record(paths, folder: Path) -> dict:
    """Every file of a folder input, and one digest over the sorted listing.

    Each entry holds the file's path relative to the folder, its size and its
    digest. The listing digest is taken over the same entries written one per
    line as "relative path, bytes, sha256", so it can be recomputed from them.
    """
    folder = Path(folder)
    entries = []
    for path in sorted(Path(p) for p in paths):
        entries.append(
            {"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    lines = ["%s %d %s" % (entry["path"], entry["bytes"], entry["sha256"]) for entry in entries]
    return {
        "path": config.portable(folder),
        "files": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "sha256_of_listing": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
        "listing_format": "relative path, bytes, sha256, one file per line, sorted",
        "entries": entries,
    }


def strict_json(value):
    """The same structure with every non-finite number written as null."""
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
