#!/usr/bin/env python
"""SESSION 11 shared helpers for the K562-essential harmonized endpoint.

Exact replication of scripts/v4/harmonized_endpoint_v1/common.py (Session 6):
ad_log1p, obs_codes, and cluster_bootstrap_corr are COPIED VERBATIM so the numerics
are byte-for-byte the Session 6 procedure. Only OUT (new output dir), the DATASETS
entry (k562_essential inputs), and a frozen-size loader differ. The AD call matches
VCC loader.py:139-171 exactly. Read-only inputs.

n_pert / n_nt are NOT re-derived here. They are the FROZEN matched-size values
(n_pert=27, n_nt=788) read from the Session 6 subsample_sizes.json, which was
derived from the VCC distribution (prereg 3.1.1). Re-deriving them from the K562
distribution would destroy cross-dataset comparability and is prohibited.
"""
from __future__ import annotations
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import json, warnings
import numpy as np
from scipy.stats import anderson_ksamp

ROOT = iga_paths.IGA_MAIN
OUT = f"{ROOT}/results_v4/harmonized_endpoint_k562_essential_v1"
REPL = f"{ROOT}/data/replogle"
PREREG_COMMIT = "220aed06ab309641266c6b3939c1ff39037771f5"
SEEDS = list(range(20260801, 20260826))   # 25 literal integers, frozen (same as RPE1/gwps)
R = 25
QMETH = "linear"

# Frozen matched-size values live here (Session 6 output, read-only).
FROZEN_SUBSAMPLE_SIZES = f"{ROOT}/results_v4/harmonized_endpoint_v1/subsample_sizes.json"

DATASETS = {
    "k562_essential": dict(
        bulk=f"{REPL}/raw/K562_essential_normalized_bulk_01.h5ad",
        sc=f"{REPL}/raw_singlecell/K562_essential_raw_singlecell_01.h5ad",
        # transfer set == this same essential bulk non-NT (the K562 transfer evaluation set),
        # single-cell == this same essential screen -> same-screen, so denom(3)=denom(2)
        # (RPE1-style branch, matching Session 10's coverage gate).
        transfer_n=2176,
        runtime_limit_hours=6.0,
    ),
}


def load_frozen_sizes():
    """Return (n_pert, n_nt, full_frozen_dict) from the frozen Session 6 file. No re-derivation."""
    with open(FROZEN_SUBSAMPLE_SIZES) as fh:
        ss = json.load(fh)
    return int(ss["n_pert"]), int(ss["n_nt"]), ss


def ad_log1p(a, b):
    """log1p(max(0, anderson_ksamp([a,b]).statistic)) -- EXACTLY VCC loader.py:139-171.
    Returns np.nan on degenerate/error, matching VCC (a.size<2|b.size<2 -> NaN; non-finite
    or ValueError/RuntimeError -> NaN; scipy p-value warnings suppressed; default midrank=True)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat = anderson_ksamp([a, b]).statistic
        if not np.isfinite(stat):
            return np.nan
        return float(np.log1p(max(0.0, stat)))
    except (ValueError, RuntimeError):
        return np.nan


def obs_codes(series):
    """Return (codes int32/int64, categories np.ndarray[str]) using categorical codes when
    available to avoid materializing millions of python strings."""
    import pandas as pd
    if hasattr(series, "cat"):
        cats = series.cat.categories.astype(str).to_numpy()
        codes = series.cat.codes.to_numpy()
        return codes, cats
    codes, uniques = pd.factorize(series.astype(str).to_numpy(), sort=False)
    return codes.astype(np.int64), np.asarray(uniques, dtype=object).astype(str)


def cluster_bootstrap_corr(x, y, groups, B=10000, seed=20260801):
    """Cluster bootstrap over `groups` (target genes). Resample unique groups with replacement,
    pool their member rows, recompute Pearson & Spearman. Returns dict with point + 95% CI."""
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float); groups = np.asarray(groups)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[m], y[m], groups[m]
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    pe0 = float(pearsonr(x, y)[0]) if x.size > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    sp0 = float(spearmanr(x, y)[0]) if x.size > 2 else np.nan
    rng = np.random.default_rng(seed)
    pes, sps = [], []
    ng = ug.size
    for _ in range(B):
        pick = ug[rng.integers(0, ng, ng)]
        rows = np.concatenate([idx_by_g[g] for g in pick])
        xb, yb = x[rows], y[rows]
        if xb.size > 2 and np.std(xb) > 0 and np.std(yb) > 0:
            pes.append(pearsonr(xb, yb)[0]); sps.append(spearmanr(xb, yb)[0])
    def ci(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        return [float(np.quantile(v, 0.025, method=QMETH)), float(np.quantile(v, 0.975, method=QMETH))] if v.size else [None, None]
    return dict(pearson=pe0, spearman=sp0, n=int(x.size), n_groups=int(ng), B=int(B), seed=int(seed),
                pearson_ci95=ci(pes), spearman_ci95=ci(sps))
