"""E5, the SVD fallback (bioinf2027/plans/E5.md, Section 19): where LAPACK gesdd does not converge, and what gesvd gives.

Replays the family's [Bc; e] model of GenePT, endpoint (b), within RPE1 (the task on which the family smoke trace
stopped) and, for every training matrix the ridge decomposes, tries gesdd at 1, 2, 4, 8 and 16 BLAS threads. For a
matrix on which gesdd fails at any thread count: its rank, gesvd's reconstruction and orthogonality errors, and the
largest difference between the ridge predictions through gesvd and through gesdd at each thread count where gesdd
converges. Output bioinf2027/results/E5/svd_check.json. Diagnostic only; no E5 value is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import e5fit  # noqa: E402
import make_e5_fits as m  # noqa: E402
from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E5" / "svd_check.json"
THREADS = (1, 2, 4, 8, 16)
TASK = ("GenePT", "b", "within RPE1")


def ridge_by(x_train, y_train, x_test, alphas, driver: str):
    x_train, x_test = np.asarray(x_train, dtype=np.float64), np.asarray(x_test, dtype=np.float64)
    mean, sd = x_train.mean(axis=0), x_train.std(axis=0)
    keep = sd > 0
    a, b = (x_train[:, keep] - mean[keep]) / sd[keep], (x_test[:, keep] - mean[keep]) / sd[keep]
    y_mean = float(np.mean(y_train))
    if driver == "gesdd":
        u, s, vt = np.linalg.svd(a, full_matrices=False)
    else:
        u, s, vt = scipy.linalg.svd(a, full_matrices=False, lapack_driver="gesvd")
    shrink = s[:, None] / (s[:, None] ** 2 + np.asarray(alphas, dtype=np.float64)[None, :])
    return (b @ vt.T) @ (shrink * (u.T @ (np.asarray(y_train, dtype=np.float64) - y_mean))[:, None]) + y_mean, a


def main() -> None:
    m.DATA.update(m.load())
    m.DATA["embeddings"] = m.load_embeddings()
    m.DATA["covered"] = m.covered()
    original = e5fit.ridge_path
    matrices, failing = [], []

    def hook(x_train, y_train, x_test, alphas=e5fit.RIDGE_GRID):
        converged, predictions = {}, {}
        for threads in THREADS:
            with threadpool_limits(threads):
                try:
                    predictions[threads], _ = ridge_by(x_train, y_train, x_test, alphas, "gesdd")
                    converged[threads] = True
                except np.linalg.LinAlgError:
                    converged[threads] = False
        record = {"call": len(matrices) + 1, "rows": int(np.shape(x_train)[0]), "columns": int(np.shape(x_train)[1]),
                  "penalties": len(alphas), "gesdd_converged": {str(t): c for t, c in converged.items()}}
        matrices.append(record)
        if not all(converged.values()):
            with threadpool_limits(1):
                through_gesvd, a = ridge_by(x_train, y_train, x_test, alphas, "gesvd")
                u, s, vt = scipy.linalg.svd(a, full_matrices=False, lapack_driver="gesvd")
            failing.append({**record, "rank_at_1e-10": int((s > s[0] * 1e-10).sum()), "smallest_singular_value": float(s[-1]),
                            "largest_singular_value": float(s[0]),
                            "gesvd_reconstruction_max_abs_error": float(np.abs((u * s) @ vt - a).max()),
                            "gesvd_orthogonality_max_abs_error": float(max(np.abs(u.T @ u - np.eye(s.size)).max(), np.abs(vt @ vt.T - np.eye(s.size)).max())),
                            "predictions": int(through_gesvd.size), "prediction_max_abs": float(np.abs(through_gesvd).max()),
                            "max_abs_difference_gesvd_minus_gesdd": {str(t): float(np.abs(through_gesvd - p).max()) for t, p in predictions.items()}})
        return original(x_train, y_train, x_test, alphas)

    e5fit.ridge_path = hook
    name, endpoint, setting = TASK
    vectors = m.DATA["embeddings"][name]
    m.run_setting(setting, endpoint, {"[Bc; e]": m.embedding_provider(name, vectors)}, m.setting_genes(setting))
    out = {"plan": "bioinf2027/plans/E5.md, Section 19", "task": {"embedding": name, "endpoint": endpoint, "setting": setting, "model": "[Bc; e]"},
           "threads": list(THREADS), "matrices": len(matrices), "matrices_with_a_gesdd_failure": len(failing), "failing": failing,
           "numpy": np.__version__, "scipy": scipy.__version__,
           "blas": [{k: v for k, v in i.items() if k in ("internal_api", "version", "architecture")} for i in threadpool_info()],
           "script": {config.portable(Path(__file__)): sha256(Path(__file__))}}
    OUT.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print("[e5 svd check] %d matrices, %d with a gesdd failure" % (len(matrices), len(failing)), flush=True)


if __name__ == "__main__":
    main()
