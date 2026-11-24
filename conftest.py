"""Repository-level test settings.

Two tests read the VCC training file from the data root. They are skipped when
that file is absent, so the suite runs on a copy without the single-cell data.
"""
import os
from pathlib import Path

import pytest

NEEDS_VCC_TRAINING_FILE = {
    "test_a_recurring_control_cell_is_read_once_and_dropped_per_row",
    "test_perturbed_cells_are_tokenized_as_before",
}


def _vcc_training_file_present() -> bool:
    root = os.environ.get("IGA_STORAGE_ROOT")
    if not root:
        marker = Path(__file__).parent / "storage_root.txt"
        root = marker.read_text().strip() if marker.exists() else ""
    return bool(root) and (Path(root) / "IGA_MAIN" / "data" / "vcc" / "train" / "adata_Training.h5ad").is_file()


def pytest_collection_modifyitems(config, items):
    if _vcc_training_file_present():
        return
    skip = pytest.mark.skip(reason="needs the VCC training file under the data root")
    for item in items:
        if item.path.name == "test_d2_cell_keys.py" and item.name in NEEDS_VCC_TRAINING_FILE:
            item.add_marker(skip)
