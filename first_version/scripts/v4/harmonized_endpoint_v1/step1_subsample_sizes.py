#!/usr/bin/env python
"""SESSION 6 STEP 1 (prereg §3.1.1): derive fixed subsample sizes n_pert, n_nt from
existing VCC artifacts only. No relabeling, no refitting.

Exact-cell provenance:
  - Frozen VCC modeled rows = (sample_ids x batch_name) stored in the delta_hvg cache
    shards ($ROOT/cache/vcc/variants/delta_hvg/shards/{train,val,test}/*.npz). Each shard
    is ONE batch; its `sample_ids` are exactly the target genes whose (target,batch) group
    survived min_cells_per_group=10 + finite-AD, i.e. the rows the frozen VCC label was
    computed on (loader.py:385,394-399). These reconcile to 5740/2052/3972 (DATA_LINEAGE §4).
  - Perturbed cells per row = count of cells with (target_gene==g, batch==b) among perturbed
    cells (target_gene!='non-targeting') in the corresponding split h5ad obs. This is the
    SAME grouping loader.py applies before computing AD (§3.5/§5) -> the exact cells used.
  - NT cells per batch = count of cells with target_gene=='non-targeting' per batch in the
    Training h5ad (loader loads NT only from Training; canonical 38,176-cell pool; §1).

n_pert = floor(median perturbed cells per VCC row); n_nt = floor(median NT cells per VCC batch).
Quantiles: numpy.quantile(method='linear'). Read-only; obs only (never .X). Writes ONE file:
  results_v4/harmonized_endpoint_v1/subsample_sizes.json
"""
from __future__ import annotations
import os as _o, sys as _s
_r = _o.path.dirname(_o.path.abspath(__file__))
while _r != _o.path.dirname(_r) and not _o.path.exists(_o.path.join(_r, "iga_paths.py")):
    _r = _o.path.dirname(_r)
if _r not in _s.path:
    _s.path.insert(0, _r)
import iga_paths
import os, json, glob
import numpy as np, pandas as pd
import anndata as ad

ROOT = iga_paths.IGA_MAIN
OUT = f"{ROOT}/results_v4/harmonized_endpoint_v1"
SHARDS = f"{ROOT}/cache/vcc/variants/delta_hvg/shards"
SPLIT_H5 = {
    "train": f"{ROOT}/data/vcc/train/adata_Training.h5ad",
    "val":   f"{ROOT}/data/vcc/validation/adata_Validation.h5ad",
    "test":  f"{ROOT}/data/vcc/test/adata_Test.h5ad",
}
QMETH = "linear"


def qsummary(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(n=int(a.size), min=float(a.min()), q25=float(np.quantile(a, 0.25, method=QMETH)),
                median=float(np.quantile(a, 0.50, method=QMETH)),
                q75=float(np.quantile(a, 0.75, method=QMETH)), max=float(a.max()),
                mean=float(a.mean()))


def frozen_rows(split):
    """Return set of (target_gene, batch) frozen modeled rows for a split, from shards."""
    rows = set()
    per_batch = {}
    for f in sorted(glob.glob(f"{SHARDS}/{split}/*.npz")):
        z = np.load(f, allow_pickle=True)
        b = str(z["batch_name"][0])
        sids = [str(s) for s in z["sample_ids"].tolist()]
        per_batch[b] = len(sids)
        for tg in sids:
            rows.add((tg, b))
    return rows, per_batch


print("=== STEP 1: VCC subsample-size provenance ===", flush=True)
print(f"[artifact] frozen modeled rows  : {SHARDS}/{{train,val,test}}/*.npz  (fields: sample_ids, batch_name)")
print(f"[artifact] perturbed cell counts: {{Training,Validation,Test}} h5ad obs (fields: target_gene, batch)")
print(f"[artifact] NT cell counts       : Training h5ad obs (target_gene=='non-targeting', per batch)")

pert_per_row = {}       # (tg,batch) -> perturbed cell count, over all frozen rows
per_split_rowcounts = {}
per_split_pert = {}
nt_per_batch = None

for split, h5 in SPLIT_H5.items():
    fr, per_batch = frozen_rows(split)
    per_split_rowcounts[split] = dict(n_frozen_rows=len(fr), n_batches=len(per_batch),
                                      n_targets=len({t for t, _ in fr}))
    a = ad.read_h5ad(h5, backed="r")
    o = a.obs
    tg = o["target_gene"].astype(str).to_numpy()
    bt = o["batch"].astype(str).to_numpy()
    a.file.close()
    is_nt = (tg == "non-targeting")
    # perturbed group sizes for this split
    dfp = pd.DataFrame({"tg": tg[~is_nt], "b": bt[~is_nt]})
    grp = dfp.groupby(["tg", "b"], observed=True).size()
    vals = []
    missing = 0
    below10 = 0
    for (tg_g, b_g) in fr:
        c = int(grp.get((tg_g, b_g), 0))
        if c == 0:
            missing += 1
        if c < 10:
            below10 += 1
        pert_per_row[(tg_g, b_g)] = c
        vals.append(c)
    per_split_pert[split] = dict(summary=qsummary(vals), n_missing_in_obs=missing,
                                 n_below_10=below10)
    print(f"[{split}] frozen rows={len(fr)} targets={per_split_rowcounts[split]['n_targets']} "
          f"batches={len(per_batch)} | perturbed/row median={per_split_pert[split]['summary']['median']:.1f} "
          f"min={per_split_pert[split]['summary']['min']:.0f} max={per_split_pert[split]['summary']['max']:.0f} "
          f"| missing_in_obs={missing} below10={below10}", flush=True)
    if split == "train":
        dfn = pd.DataFrame({"b": bt[is_nt]})
        nt_counts = dfn.groupby("b", observed=True).size()
        nt_per_batch = {str(k): int(v) for k, v in nt_counts.items()}
        print(f"[train] NT total={int(is_nt.sum())} across {len(nt_per_batch)} batches "
              f"(median/batch={np.median(list(nt_per_batch.values())):.0f})", flush=True)

# pooled distribution across ALL VCC frozen rows
all_pert = np.array(list(pert_per_row.values()), dtype=np.float64)
nt_vals = np.array(list(nt_per_batch.values()), dtype=np.float64)
pert_summary = qsummary(all_pert)
nt_summary = qsummary(nt_vals)

n_pert = int(np.floor(pert_summary["median"]))
n_nt = int(np.floor(nt_summary["median"]))

n_rows_total = int(all_pert.size)
n_batches_total = int(nt_vals.size)

# QC
qc = dict(
    total_frozen_rows=n_rows_total,
    expected_frozen_rows=5740 + 2052 + 3972,
    frozen_rows_match=(n_rows_total == 5740 + 2052 + 3972),
    per_split_rowcounts=per_split_rowcounts,
    any_row_missing_in_obs=int(sum(s["n_missing_in_obs"] for s in per_split_pert.values())),
    any_row_below_10_cells=int(sum(s["n_below_10"] for s in per_split_pert.values())),
    nt_total_cells=int(nt_vals.sum()),
    nt_expected_total=38176,
    nt_total_match=(int(nt_vals.sum()) == 38176),
    n_nt_batches=n_batches_total,
)

print("\n=== provenance summary ===")
print(f"VCC rows represented: {n_rows_total} (expected 11764; match={qc['frozen_rows_match']})")
print(f"VCC batches represented (NT): {n_batches_total}; NT total={qc['nt_total_cells']} (match 38176={qc['nt_total_match']})")
print(f"rows missing in obs: {qc['any_row_missing_in_obs']}; rows <10 cells: {qc['any_row_below_10_cells']} (must be 0)")
print("\nPerturbed cells per VCC (target,batch) row:")
print(f"  min={pert_summary['min']:.0f} q25={pert_summary['q25']:.1f} median={pert_summary['median']:.1f} "
      f"q75={pert_summary['q75']:.1f} max={pert_summary['max']:.0f}")
print("NT cells per VCC batch:")
print(f"  min={nt_summary['min']:.0f} q25={nt_summary['q25']:.1f} median={nt_summary['median']:.1f} "
      f"q75={nt_summary['q75']:.1f} max={nt_summary['max']:.0f}")
print(f"\n>>> n_pert = floor(median perturbed/row) = floor({pert_summary['median']:.1f}) = {n_pert}")
print(f">>> n_nt   = floor(median NT/batch)      = floor({nt_summary['median']:.1f}) = {n_nt}")

out = dict(
    prereg_commit="220aed06ab309641266c6b3939c1ff39037771f5",
    derivation="prereg §3.1.1: n_pert=floor(median perturbed cells per VCC row); n_nt=floor(median NT cells per VCC batch)",
    quantile_method=QMETH,
    numpy_version=np.__version__,
    provenance=dict(
        frozen_rows_source=f"{SHARDS}/{{train,val,test}}/*.npz (sample_ids x batch_name)",
        perturbed_count_source="split h5ad obs groupby(target_gene,batch), perturbed only",
        nt_count_source="Training h5ad obs, target_gene=='non-targeting', per batch",
        why_exact=("frozen rows taken directly from the cache the VCC label was written to; "
                   "cell counts are the size of the identical (target_gene,batch) groups the "
                   "loader forms before computing AD; NT per batch is the canonical Training pool "
                   "used as the control sample for every row"),
    ),
    n_vcc_rows=n_rows_total,
    n_vcc_batches=n_batches_total,
    perturbed_cells_per_row=pert_summary,
    nt_cells_per_batch=nt_summary,
    per_split_perturbed=per_split_pert,
    nt_per_batch=nt_per_batch,
    n_pert=n_pert,
    n_nt=n_nt,
    qc=qc,
)
with open(f"{OUT}/subsample_sizes.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\n[written] {OUT}/subsample_sizes.json")
