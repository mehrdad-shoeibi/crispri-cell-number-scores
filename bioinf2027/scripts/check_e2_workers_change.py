"""Check of the worker-count change of make_e2_fits.py (gate G4, item 4): it must change no value.

Refits, in this one process and without any pool, the smoke's N0[pca50] fits (E2.md, Section 17) that the pooled
smoke run computed, and compares every saved prediction array exactly. Reads the smoke's predictions under
$IGA_STORAGE_ROOT/bioinf2027/scratch/E2_fits_smoke/; writes nothing.
"""
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bioinf2027" / "src"))
sys.path.insert(0, str(REPO / "bioinf2027" / "scripts"))
import numpy as np  # noqa: E402

import make_e2_fits as mf  # noqa: E402
from bioinf2027 import e1cells, heads  # noqa: E402

mf.SMOKE = True
encoders = ["stack", "pca50", "pca768"]
mf.e1f.load_inputs(list(mf.CELLS))
order = list(e1cells.CELLS)
mf.DATA["contexts"] = {cell: mf.e1f.build_context(cell, order.index(cell), {}) for cell in mf.CELLS}
mf.DATA["screens"] = {s: mf.e1f.DATA["screens"][s].constructs.tolist() for s in mf.e1f.DATA["screens"]}
mf.DATA["features"] = mf.load_features(tuple(encoders))
heads.FOREST["n_estimators"] = 5
heads.BOOSTING["max_iter"] = 20
mf.DATA["sides"] = {}
for cell in mf.CELLS:
    mf.DATA["sides"][(cell, "pca50", True)] = mf.encoder_sides(cell, "pca50", null=True)
tasks = [t for t in mf.tasks(["pca50"]) if t[2] == "N0"]
compared = 0
for cell, encoder, r, spec in tasks:
    path = mf.predictions_root() / spec.context.replace(" | ", "_") / spec.file_name()
    out, _ = mf.fit(cell, encoder, spec, r)
    with np.load(path, allow_pickle=False) as store:
        assert str(store["spec"]) == json.dumps(spec.as_dict(), sort_keys=True)
        for name, values in out.items():
            saved = store[name.replace("|", "__")]
            assert saved.shape == values.shape and np.array_equal(saved, values, equal_nan=True), (spec.key(), name)
            compared += 1
print("[e2 workers check] %d fits refitted, %d prediction arrays equal exactly" % (len(tasks), compared))
