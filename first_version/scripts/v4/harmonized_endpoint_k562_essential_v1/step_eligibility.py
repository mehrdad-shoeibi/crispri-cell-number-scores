#!/usr/bin/env python
"""SESSION 11 STEP 1/3: eligibility & coverage (prereg 6). OBS ONLY, no .X materialized.
Exact replication of Session 6 step3_eligibility.py, K562-essential (same-screen) branch:
the transfer set is the essential bulk itself and the single-cell is the essential screen,
so denom(3) = denom(2) and coverage = n_eligible/denom2 (identical to Session 10's gate).

Writes eligibility.csv and coverage_summary.json to the new output dir, prints all three
denominators and coverage, and asserts the counts match Session 10 (2176/1970/1903).
This computes NO endpoint value (no AD statistic, no subsampling).
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd
import anndata as ad
from common import OUT, DATASETS, obs_codes, load_frozen_sizes

DS = "k562_essential"

# Session 10 reference counts (docs/SESSION10_K562_ESSENTIAL_COVERAGE_GATE.md).
S10 = dict(denom1=2176, denom2=1970, denom3=1970, eligible=1903)


def target_ensg(obs_index_arr):
    return np.array([s.rsplit("_", 1)[-1] for s in obs_index_arr])


def main():
    cfg = DATASETS[DS]
    n_pert, n_nt, _ = load_frozen_sizes()
    MIN_PERT = max(20, n_pert); MIN_NT = max(200, n_nt)
    print(f"=== [{DS}] eligibility  (MIN_PERT={MIN_PERT}, MIN_NT={MIN_NT}) ===", flush=True)

    # ---- bulk (small) ----
    bu = ad.read_h5ad(cfg["bulk"], backed="r")
    bidx = bu.obs.index.astype(str).to_numpy()
    is_nt_b = pd.Series(bidx).str.contains("non-targeting", case=False).to_numpy()
    tens = target_ensg(bidx)
    bu.file.close()
    assert len(bidx) == len(set(bidx)), "bulk obs_index not unique -> one-to-one join violated"
    nonnt = ~is_nt_b
    denom1 = int(nonnt.sum())

    # ---- single-cell obs (codes, memory-safe) ----
    sc = ad.read_h5ad(cfg["sc"], backed="r")
    gt_codes, gt_cats = obs_codes(sc.obs["gene_transcript"])
    gene_codes, gene_cats = obs_codes(sc.obs["gene"])
    var_ens = sc.var.index.astype(str).to_numpy()
    sc.file.close()
    ens2col = {e: i for i, e in enumerate(var_ens.tolist())}
    nt_gene_mask = np.array(["non-targeting" in c.lower() for c in gene_cats])
    is_nt_cell = nt_gene_mask[gene_codes]
    nt_available = int(is_nt_cell.sum())
    pert_cell = ~is_nt_cell
    pert_counts = np.bincount(gt_codes[pert_cell], minlength=len(gt_cats))
    gtcat2idx = {c: i for i, c in enumerate(gt_cats.tolist())}

    print(f"[{DS}] bulk non-NT={denom1}  NT_available(single-cell)={nt_available}  "
          f"gene_transcript cats={len(gt_cats)} var genes={len(var_ens)}", flush=True)

    # ---- per-construct eligibility ----
    rows = []
    for i in np.where(nonnt)[0]:
        con = bidx[i]; ens = tens[i]
        gidx = gtcat2idx.get(con, -1)
        npc = int(pert_counts[gidx]) if gidx >= 0 else 0
        join_ok = gidx >= 0 and npc >= 1
        col = ens2col.get(ens, -1)
        map_ok = col >= 0
        reasons = []
        if not join_ok: reasons.append("no_single_cell_join")
        if not map_ok: reasons.append("target_ensg_not_in_var")
        if join_ok and npc < MIN_PERT: reasons.append(f"perturbed_cells<{MIN_PERT}")
        if nt_available < MIN_NT: reasons.append(f"nt_available<{MIN_NT}")
        eligible = (join_ok and map_ok and npc >= MIN_PERT and nt_available >= MIN_NT)
        rows.append(dict(construct=con, target_symbol=con.split("_")[1] if "_" in con else "",
                         target_ensg=ens, target_col=col, n_perturbed_cells=npc,
                         n_nt_available=nt_available, join_status=bool(join_ok),
                         unique_map_status=bool(map_ok), eligible=bool(eligible),
                         exclusion_reason="" if eligible else ";".join(reasons)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/eligibility.csv", index=False)

    denom2 = int((df.join_status & df.unique_map_status).sum())
    elig = df[df.eligible].copy()
    n_elig = len(elig)
    elig_cols = sorted(set(int(c) for c in elig.target_col.tolist()))
    C = len(elig_cols)

    # same-screen: transfer set == essential bulk itself; denom(3) = denom(2)
    denom3 = denom2
    coverage = n_elig / denom3 if denom3 else None

    cov = dict(dataset=DS, MIN_PERT=MIN_PERT, MIN_NT=MIN_NT,
               denom1_raw_nonNT=denom1, denom2_post_join=denom2,
               denom3_definition=("K562-essential transfer set = this same essential bulk (2176 non-NT); "
                                  "single-cell = this same essential screen -> denom(3)=denom(2)=constructs "
                                  "satisfying (2)."),
               denom3=denom3,
               n_eligible=n_elig, distinct_eligible_target_cols=C,
               nt_available=nt_available,
               coverage_denom3=coverage,
               gate_coverage_used_for_60pct=coverage,
               passes_60pct=(coverage is not None and coverage >= 0.60))
    fail = df[~df.eligible].exclusion_reason.value_counts().to_dict()
    cov["exclusion_reason_counts"] = {str(k): int(v) for k, v in fail.items()}

    # cross-check against Session 10
    s10ok = (denom1 == S10["denom1"] and denom2 == S10["denom2"]
             and denom3 == S10["denom3"] and n_elig == S10["eligible"])
    cov["session10_reference"] = S10
    cov["matches_session10"] = bool(s10ok)

    path = f"{OUT}/coverage_summary.json"
    with open(path, "w") as fh:
        json.dump({DS: cov}, fh, indent=2)

    print(f"[{DS}] denom(1)={denom1}  denom(2)={denom2}  denom(3)={denom3}  eligible={n_elig}  cols C={C}")
    print(f"[{DS}] exclusion reasons: {cov['exclusion_reason_counts']}")
    print(f"[{DS}] COVERAGE vs denom(3) = {coverage:.4f}  ->  passes 60% = {cov['passes_60pct']}")
    print(f"[{DS}] MATCHES SESSION 10 (2176/1970/1970/1903): {s10ok}")
    if not s10ok:
        print(f"STOP: eligibility counts differ from Session 10 {S10}. "
              f"got denom1={denom1} denom2={denom2} denom3={denom3} eligible={n_elig}")
        sys.exit(4)
    nt_mem_gb = nt_available * C * 4 / 1e9
    print(f"[{DS}] projected NT_targ memory (NT x C x 4B) = {nt_mem_gb:.3f} GB "
          f"({'SINGLE PASS OK' if nt_mem_gb < 1.5 else 'NEEDS COLUMN-BLOCKING'})")


if __name__ == "__main__":
    main()
