#!/usr/bin/env python
"""SESSION 6 STEP 3.1 / 4.1: eligibility & coverage (prereg §6). OBS ONLY, no .X materialized.
Usage: python step3_eligibility.py {rpe1|gwps}
Writes results_v4/harmonized_endpoint_v1/{ds}_eligibility.csv and merges a coverage block into
coverage_summary.json. Prints all three denominators and coverage vs denominator (3)."""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd
import anndata as ad
from common import OUT, DATASETS, obs_codes


def target_ensg(obs_index_arr):
    return np.array([s.rsplit("_", 1)[-1] for s in obs_index_arr])


def main(ds):
    cfg = DATASETS[ds]
    with open(f"{OUT}/subsample_sizes.json") as fh:
        ss = json.load(fh)
    n_pert, n_nt = ss["n_pert"], ss["n_nt"]
    MIN_PERT = max(20, n_pert); MIN_NT = max(200, n_nt)
    print(f"=== [{ds}] eligibility  (MIN_PERT={MIN_PERT}, MIN_NT={MIN_NT}) ===", flush=True)

    # ---- bulk (small) ----
    bu = ad.read_h5ad(cfg["bulk"], backed="r")
    bidx = bu.obs.index.astype(str).to_numpy()
    stored_ad = bu.obs["anderson_darling_counts"].to_numpy().astype(float)
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
    # NT cells: gene category contains 'non-targeting'
    nt_cat_mask = np.array(["non-targeting" in c.lower() for c in gt_cats])  # gene_transcript-level NT
    nt_gene_mask = np.array(["non-targeting" in c.lower() for c in gene_cats])
    is_nt_cell = nt_gene_mask[gene_codes]
    nt_available = int(is_nt_cell.sum())
    # perturbed cells per gene_transcript
    pert_cell = ~is_nt_cell
    pert_counts = np.bincount(gt_codes[pert_cell], minlength=len(gt_cats))
    gtcat2idx = {c: i for i, c in enumerate(gt_cats.tolist())}

    print(f"[{ds}] bulk non-NT={denom1}  NT_available(single-cell)={nt_available}  "
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
    df.to_csv(f"{OUT}/{ds}_eligibility.csv", index=False)

    denom2 = int((df.join_status & df.unique_map_status).sum())
    elig = df[df.eligible].copy()
    n_elig = len(elig)
    elig_cols = sorted(set(int(c) for c in elig.target_col.tolist()))
    C = len(elig_cols)

    # ---- denominator (3) ----
    cov = dict(dataset=ds, MIN_PERT=MIN_PERT, MIN_NT=MIN_NT,
               denom1_raw_nonNT=denom1, denom2_post_join=denom2,
               n_eligible=n_elig, distinct_eligible_target_cols=C,
               nt_available=nt_available)
    fail = df[~df.eligible].exclusion_reason.value_counts().to_dict()
    cov["exclusion_reason_counts"] = {str(k): int(v) for k, v in fail.items()}

    if ds == "rpe1":
        # transfer set == all bulk non-NT (2549); denom(3) = those satisfying (2)
        cov["denom3_definition"] = "RPE1 transfer set = all 2549 bulk non-NT; denom(3)=constructs satisfying (2)"
        cov["denom3"] = denom2
        cov["coverage_denom3"] = n_elig / denom2 if denom2 else None
        gate = cov["coverage_denom3"]
    else:
        # gwps: 'K562 transfer eval' = ESSENTIAL 2176 constructs -> link by TARGET ENSG
        eb = ad.read_h5ad(cfg["essential_bulk"], backed="r")
        eidx = eb.obs.index.astype(str).to_numpy()
        e_nt = pd.Series(eidx).str.contains("non-targeting", case=False).to_numpy()
        eb.file.close()
        ess_constructs = eidx[~e_nt]
        ess_targets = set(target_ensg(ess_constructs).tolist())
        elig_targets = set(elig.target_ensg.tolist())
        join_targets = set(df[df.join_status & df.unique_map_status].target_ensg.tolist())
        covered = ess_targets & elig_targets
        cov["denom3_definition"] = ("gwps screen != essential screen (distinct construct sets/ids). "
            "K562 transfer eval used ESSENTIAL bulk (2176 non-NT constructs). Linked by TARGET ENSG: "
            "coverage(3)=fraction of essential-transfer target genes for which an ELIGIBLE gwps construct exists.")
        cov["essential_transfer_constructs"] = int(len(ess_constructs))
        cov["essential_transfer_unique_targets"] = int(len(ess_targets))
        cov["essential_targets_with_gwps_join"] = int(len(ess_targets & join_targets))
        cov["essential_targets_covered_by_eligible_gwps"] = int(len(covered))
        cov["coverage_denom3_targetlinked"] = len(covered) / len(ess_targets) if ess_targets else None
        # gwps-native transparency figure
        cov["coverage_gwps_native_elig_over_denom2"] = n_elig / denom2 if denom2 else None
        gate = cov["coverage_denom3_targetlinked"]

    cov["gate_coverage_used_for_60pct"] = gate
    cov["passes_60pct"] = (gate is not None and gate >= 0.60)

    # merge into coverage_summary.json
    path = f"{OUT}/coverage_summary.json"
    allcov = {}
    if os.path.exists(path):
        with open(path) as fh: allcov = json.load(fh)
    allcov[ds] = cov
    with open(path, "w") as fh: json.dump(allcov, fh, indent=2)

    print(f"[{ds}] denom(1)={denom1}  denom(2)={denom2}  eligible={n_elig}  distinct target cols C={C}")
    print(f"[{ds}] exclusion reasons: {cov['exclusion_reason_counts']}")
    print(f"[{ds}] COVERAGE vs denom(3) = {gate:.4f}  ->  passes 60% = {cov['passes_60pct']}")
    if ds == "gwps":
        print(f"[gwps] essential targets={cov['essential_transfer_unique_targets']} "
              f"covered-by-eligible-gwps={cov['essential_targets_covered_by_eligible_gwps']} "
              f"| gwps-native elig/denom2={cov['coverage_gwps_native_elig_over_denom2']:.4f}")
    # NT_targ memory projection for the endpoint pass
    nt_mem_gb = nt_available * C * 4 / 1e9
    print(f"[{ds}] projected NT_targ memory (NT x C x 4B) = {nt_mem_gb:.2f} GB "
          f"({'SINGLE PASS OK' if nt_mem_gb < 1.5 else 'NEEDS COLUMN-BLOCKING'})")
    return cov


if __name__ == "__main__":
    main(sys.argv[1])
