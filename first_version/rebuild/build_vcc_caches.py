"""Build the Virtual Cell Challenge caches with the unchanged loader of the first version.

Calls load_vcc_pseudobulk for the three splits with feature_transform="delta"
and gene_selection="hvg" (defaults n_hvg=2000, min_cells_per_group=10, seed=42).
The first call writes the control-cell baselines, the gene names, the gene
selection and the training per-batch files; the second and third write the
validation and test per-batch files.

usage: IGA_STORAGE_ROOT=<data root> python -B build_vcc_caches.py <folder that holds iga_paths.py>
"""
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path

clone = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(clone))

import iga_paths  # noqa: E402  (the resolver of that folder; reads IGA_STORAGE_ROOT)

if iga_paths.STORAGE != os.path.abspath(os.environ["IGA_STORAGE_ROOT"]):
    raise SystemExit("data root is not the one in IGA_STORAGE_ROOT: %s" % iga_paths.STORAGE)

from src.data.vcc.loader import load_vcc_pseudobulk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
DATA_VCC = iga_paths.IGA_MAIN + "/data/vcc"
for split in ("train", "val", "test"):
    t0 = time.time()
    shards = load_vcc_pseudobulk(DATA_VCC, split, feature_transform="delta", gene_selection="hvg")
    print(json.dumps({
        "split": split,
        "shards": len(shards),
        "groups": int(sum(s.y.shape[0] for s in shards)),
        "seconds": round(time.time() - t0, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2),
    }), flush=True)
