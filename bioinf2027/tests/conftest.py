"""Test settings: one OpenMP and BLAS thread.

On the shared server the fits run at a higher priority on every core. A histogram gradient boosting
fit inside a test then starts one OpenMP thread per core, the threads are scheduled so rarely that a
fit of 400 rows did not finish in an hour, and the suite stalled (2026-11-10). With one thread the
whole suite takes about 20 seconds. The fit scripts set the same limits; this file sets them before
any test module loads numpy or scikit-learn, unless the caller has set them.
"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
