#!/usr/bin/env python
"""SESSION 11: harmonized primary endpoint for K562-essential.

EXACT replication of scripts/v4/harmonized_endpoint_v1/run_replogle_endpoint.py (Session 6),
retargeted to the K562-essential single-cell/bulk and the Session 11 output filenames. The
streaming layout, VCC normalization, resample(), gemgroup_stat(), AD call, and every numeric
choice are copied verbatim from Session 6.

Primary endpoint (prereg 3.1): for each eligible construct c targeting gene g, for each of the
25 literal seeds 20260801..20260825: draw n_pert=27 perturbed cells (without replacement from c's
pooled eligible cells) and n_nt=788 NT cells (without replacement from the pooled eligible NT set);
s = log1p(max(0, anderson_ksamp([pert_sub[:,g], nt_sub[:,g]]).statistic)) on VCC-normalized values
(per-cell normalize_total(1e4)+log1p, matching VCC loader.py:123-136); y_primary = median of 25.

Memory: dense X read in sequential row-chunks (<=~0.30 GB each); NT held at eligible target columns
in a single block (NT x C x 4B = ~0.08 GB here); full matrix never densified. Reads X exactly once.
Reading a full-gene row-chunk (<0.30 GB, justified below the 2 GB limit) is Session 6's mandated
layout: it extracts every needed target column in ONE sequential pass, rather than re-reading the
10.66 GB contiguous dataset once per gene column (which for a contiguous h5py layout would re-scan
the whole file ~1811 times). Each target column is thus materialized exactly once.
"""
from __future__ import annotations
import os, sys, json, time, resource, gc, hashlib, warnings, platform
import numpy as np, pandas as pd
import h5py, anndata as ad
import scipy
from common import OUT, DATASETS, SEEDS, R, QMETH, ad_log1p, obs_codes, cluster_bootstrap_corr, load_frozen_sizes

DS = "k562_essential"
RUNTIME_LIMIT_HOURS = DATASETS[DS]["runtime_limit_hours"]
CAPTURED_WARNINGS = []


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def q(a, p):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.quantile(a, p, method=QMETH)) if a.size else float("nan")


def sha256_file(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    cfg = DATASETS[DS]
    t_start = time.time()
    start_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start))
    n_pert, n_nt, _ = load_frozen_sizes()
    print(f"=== [{DS}] harmonized endpoint  n_pert={n_pert} n_nt={n_nt} R={R} ===", flush=True)

    # ---- eligible constructs (from step_eligibility) ----
    el = pd.read_csv(f"{OUT}/eligibility.csv")
    el = el[el.eligible].reset_index(drop=True)
    constructs = el.construct.to_numpy()
    con_col = el.target_col.to_numpy().astype(int)
    con_ensg = el.target_ensg.to_numpy()
    con_sym = el.target_symbol.astype(str).to_numpy()
    n_con = len(constructs)

    # ---- stored AD from bulk ----
    bu = ad.read_h5ad(cfg["bulk"], backed="r")
    bidx = bu.obs.index.astype(str).to_numpy()
    stored_all = bu.obs["anderson_darling_counts"].to_numpy().astype(float)
    bu.file.close()
    b2ad = {b: v for b, v in zip(bidx.tolist(), stored_all.tolist())}
    con_stored = np.array([b2ad[c] for c in constructs.tolist()])

    # ---- single-cell obs (codes; no X) ----
    sc = ad.read_h5ad(cfg["sc"], backed="r")
    gt_codes, gt_cats = obs_codes(sc.obs["gene_transcript"])
    gene_codes, gene_cats = obs_codes(sc.obs["gene"])
    gem_codes, gem_cats = obs_codes(sc.obs["gem_group"])
    n_cells = sc.n_obs; n_genes = sc.n_vars
    sc.file.close(); del sc; gc.collect()
    gt_codes = gt_codes.astype(np.int32); gem_codes = gem_codes.astype(np.int32)
    nt_gene_cat = np.array(["non-targeting" in c.lower() for c in gene_cats])
    is_nt_cell = nt_gene_cat[gene_codes]
    del gene_codes
    con_of_cat = np.full(len(gt_cats), -1, dtype=np.int32)
    for ci, c in enumerate(constructs.tolist()):
        j = np.where(gt_cats == c)[0]
        if j.size: con_of_cat[j[0]] = ci
    ecidx_full = np.where(is_nt_cell, -1, con_of_cat[gt_codes]).astype(np.int32)
    del gt_codes; gc.collect()
    owncol_full = np.where(ecidx_full >= 0, con_col[np.clip(ecidx_full, 0, n_con - 1)], -1).astype(np.int32)
    elig_pert_mask = ecidx_full >= 0

    nt_global = np.where(is_nt_cell)[0]
    n_NT = nt_global.size
    nt_gg = gem_codes[nt_global]
    nt_by_gem = {int(g): np.where(nt_gg == g)[0] for g in np.unique(nt_gg)}
    print(f"[{DS}] cells={n_cells} genes={n_genes} eligible constructs={n_con} NT={n_NT} "
          f"elig-perturbed cells={int(elig_pert_mask.sum())}", flush=True)

    elig_cols = np.array(sorted(set(int(c) for c in con_col.tolist())))
    max_block = max(1, int(0.70e9 / (n_NT * 4)))
    blocks = [elig_cols[i:i + max_block] for i in range(0, len(elig_cols), max_block)]
    CHUNK = max(2000, int(0.30e9 / (n_genes * 4)))
    print(f"[{DS}] distinct target cols={len(elig_cols)} -> {len(blocks)} column-block(s); "
          f"row-chunk={CHUNK}", flush=True)

    pv_by_con = [None] * n_con
    pg_by_con = [None] * n_con
    endpoint_rows = [None] * n_con

    def stream_fill(block_cols, gather_perturbed):
        col_lookup = np.full(n_genes, -1, dtype=np.int64)
        col_lookup[block_cols] = np.arange(len(block_cols))
        NT_targ = np.empty((n_NT, len(block_cols)), dtype=np.float32)
        nt_ptr = 0
        fcon, fval, fgem = [], [], []
        h = h5py.File(cfg["sc"], "r"); X = h["X"]
        t0 = time.time()
        for i in range(0, n_cells, CHUNK):
            j = min(i + CHUNK, n_cells)
            Xc = X[i:j, :]
            rs = Xc.sum(axis=1); rs = np.maximum(rs, 1.0)
            seg_nt = is_nt_cell[i:j]
            if seg_nt.any():
                loc = np.where(seg_nt)[0]
                sub = Xc[loc][:, block_cols].astype(np.float64)
                sub = np.log1p(sub / rs[loc, None] * 1e4).astype(np.float32)
                NT_targ[nt_ptr:nt_ptr + loc.size] = sub
                nt_ptr += loc.size
            if gather_perturbed:
                seg_p = elig_pert_mask[i:j]
                if seg_p.any():
                    loc = np.where(seg_p)[0]
                    oc = owncol_full[i + loc]
                    vals = np.log1p(Xc[loc, oc].astype(np.float64) / rs[loc] * 1e4).astype(np.float32)
                    fcon.append(ecidx_full[i + loc].astype(np.int64))
                    fval.append(vals); fgem.append(gem_codes[i + loc].astype(np.int64))
            if i == 0:
                print(f"    [chunk 0] {j} rows in {time.time()-t0:.2f}s peakRSS={rss_gb():.2f}GB", flush=True)
        h.close()
        assert nt_ptr == n_NT, f"NT fill {nt_ptr} != {n_NT}"
        print(f"    stream pass over {n_cells} rows: {time.time()-t0:.1f}s peakRSS={rss_gb():.2f}GB", flush=True)
        if gather_perturbed:
            return NT_targ, (np.concatenate(fcon), np.concatenate(fval), np.concatenate(fgem))
        return NT_targ, None

    def resample(pool_vals, NT_g):
        draws = np.empty(R, dtype=np.float64)
        npool = pool_vals.size
        for k, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed)
            pi = rng.choice(npool, size=n_pert, replace=False)
            ni = rng.choice(n_NT, size=n_nt, replace=False)
            draws[k] = ad_log1p(pool_vals[pi], NT_g[ni])
        return draws

    def gemgroup_stat(pvals, pgem, NT_g):
        s = []
        for g in np.unique(pgem):
            pm = pgem == g
            if pm.sum() < 10: continue
            idx = nt_by_gem.get(int(g))
            if idx is None or idx.size < 50: continue
            v = ad_log1p(pvals[pm], NT_g[idx])
            if np.isfinite(v): s.append(v)
        if len(s) >= 3:
            return float(np.mean(s)), len(s)
        return float("nan"), len(s)

    probe_info = {}
    one_probed = False
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        for bnum, block_cols in enumerate(blocks):
            need = (bnum == 0)
            NT_targ, flat = stream_fill(block_cols, gather_perturbed=need)
            if need:
                fcon, fval, fgem = flat
                order = np.argsort(fcon, kind="stable")
                fcon, fval, fgem = fcon[order], fval[order], fgem[order]
                bounds = np.searchsorted(fcon, np.arange(n_con + 1))
                for ci in range(n_con):
                    pv_by_con[ci] = fval[bounds[ci]:bounds[ci + 1]]
                    pg_by_con[ci] = fgem[bounds[ci]:bounds[ci + 1]]
                del fcon, fval, fgem, flat; gc.collect()
            col_local = {int(c): k for k, c in enumerate(block_cols.tolist())}
            members = [ci for ci in range(n_con) if int(con_col[ci]) in col_local]
            if not one_probed and members:
                ci = members[0]; NT_g = NT_targ[:, col_local[int(con_col[ci])]]
                t1 = time.time(); d = resample(pv_by_con[ci], NT_g)
                _ = ad_log1p(pv_by_con[ci], NT_g); _ = gemgroup_stat(pv_by_con[ci], pg_by_con[ci], NT_g)
                per = time.time() - t1
                proj_h = (per * n_con) / 3600
                probe_info = dict(per_construct_sec=per, projected_total_sec=per * n_con,
                                  projected_total_hours=proj_h, limit_hours=RUNTIME_LIMIT_HOURS,
                                  probe_construct=str(constructs[ci]))
                print(f"[{DS}] ONE-CONSTRUCT probe: {per*1000:.0f} ms; project {n_con} constructs = "
                      f"{per*n_con:.0f}s ({proj_h:.3f}h); limit={RUNTIME_LIMIT_HOURS}h", flush=True)
                if proj_h > RUNTIME_LIMIT_HOURS:
                    print(f"PILOT STOP -- APPROVAL REQUIRED: projected {proj_h:.2f}h exceeds "
                          f"{RUNTIME_LIMIT_HOURS}h limit.")
                    json.dump(probe_info, open(f"{OUT}/PILOT_STOP.json", "w"), indent=2)
                    sys.exit(3)
                one_probed = True
            for ci in members:
                NT_g = NT_targ[:, col_local[int(con_col[ci])]]
                pool = pv_by_con[ci]
                draws = resample(pool, NT_g)
                valid = draws[np.isfinite(draws)]
                y_primary = float(np.median(valid)) if valid.size else float("nan")
                y_pooled = ad_log1p(pool, NT_g)
                y_gg, n_elig_gg = gemgroup_stat(pool, pg_by_con[ci], NT_g)
                endpoint_rows[ci] = dict(
                    construct=constructs[ci], target_symbol=con_sym[ci], target_ensg=con_ensg[ci],
                    target_col=int(con_col[ci]), n_perturbed_pool=int(pool.size), n_nt_pool=int(n_NT),
                    y_primary=y_primary,
                    y_mean=float(np.mean(valid)) if valid.size else float("nan"),
                    y_std=float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan"),
                    y_p5=q(valid, 0.05), y_p95=q(valid, 0.95),
                    y_iqr=(q(valid, 0.75) - q(valid, 0.25)) if valid.size else float("nan"),
                    n_valid_draws=int(valid.size), y_pooled=float(y_pooled),
                    y_gemgroup=y_gg, n_eligible_gemgroups=int(n_elig_gg),
                    stored_ad=float(con_stored[ci]), stored_ad_log1p=float(np.log1p(max(0.0, con_stored[ci]))),
                )
            NT_g = None; pool = None
            del NT_targ; gc.collect()
        for w in wlist:
            CAPTURED_WARNINGS.append(f"{w.category.__name__}: {str(w.message)[:200]}")

    df = pd.DataFrame([r for r in endpoint_rows if r is not None])
    df.to_csv(f"{OUT}/endpoint_by_construct.csv", index=False)
    print(f"[{DS}] endpoint rows written: {len(df)}  NaN y_primary: {int(df.y_primary.isna().sum())}", flush=True)

    # ---------- resampling spread ----------
    iqr = df.y_iqr.to_numpy()
    yprim = df.y_primary.to_numpy()
    std_between = float(np.nanstd(yprim, ddof=1))
    ratio_con = float(np.nanmedian(iqr) / std_between) if std_between > 0 else None
    tdf = df.dropna(subset=["y_primary"]).groupby("target_ensg")
    y_target = tdf.y_primary.median()
    iqr_target = tdf.y_iqr.median()
    std_between_t = float(np.nanstd(y_target.to_numpy(), ddof=1))
    ratio_tgt = float(np.nanmedian(iqr_target.to_numpy()) / std_between_t) if std_between_t > 0 else None
    spread = dict(
        definition="resampling-noise ratio = median_c(IQR of y_c across 25 draws) / std_c(y_primary); "
                   "target level uses median-per-target IQR / std across target medians (prereg 3.3/3.4). "
                   "No pass/fail threshold applied (none pre-registered).",
        construct_level_ratio=ratio_con, target_level_ratio=ratio_tgt,
        median_construct_iqr=float(np.nanmedian(iqr)), std_between_construct=std_between,
        n_targets=int(len(y_target)), std_between_target=std_between_t,
        per_construct_iqr_summary=dict(min=q(iqr, 0.0), q25=q(iqr, 0.25), median=q(iqr, 0.5),
                                       q75=q(iqr, 0.75), p95=q(iqr, 0.95), max=q(iqr, 1.0)),
        reference_session6=dict(rpe1_construct=0.174, rpe1_target=0.184, gwps_construct=0.331, gwps_target=0.353),
    )
    json.dump(spread, open(f"{OUT}/resampling_spread.json", "w"), indent=2)

    # ---------- pooled-scale vs matched ----------
    m = np.isfinite(df.y_primary.to_numpy()) & np.isfinite(df.y_pooled.to_numpy())
    pvm = cluster_bootstrap_corr(df.y_pooled.to_numpy()[m], df.y_primary.to_numpy()[m],
                                 df.target_ensg.to_numpy()[m], B=10000, seed=20260801)
    pvm["comparison"] = "pooled-scale (all eligible cells vs full NT) vs matched-size primary (median of 25)"
    pvm["n_constructs"] = int(m.sum())
    pvm["reference_session6"] = dict(rpe1_spearman=0.884, gwps_spearman=0.885)
    json.dump(pvm, open(f"{OUT}/pooled_vs_matched.json", "w"), indent=2)

    # ---------- gem_group-matched sensitivity ----------
    gsub = df.dropna(subset=["y_gemgroup"])
    gsub = gsub[gsub.n_eligible_gemgroups >= 3]
    gcorr = cluster_bootstrap_corr(gsub.y_gemgroup.to_numpy(), gsub.y_primary.to_numpy(),
                                   gsub.target_ensg.to_numpy(), B=10000, seed=20260801) if len(gsub) > 2 else {}
    df[["construct", "target_symbol", "target_ensg", "n_eligible_gemgroups", "y_gemgroup",
        "y_primary"]].to_csv(f"{OUT}/gemgroup_sensitivity.csv", index=False)
    gj = dict(n_qualifying_constructs=int(len(gsub)),
              correlation_with_pooled_nt_primary=gcorr,
              reference_session6=dict(rpe1_n=43, gwps_n=4),
              note="per-gem_group statistic = log1p(max(0,AD)) on same-gem_group NT (>=10 pert & >=50 NT per gg), "
                   "unweighted mean across >=3 eligible groups (prereg 3.5/5). Not resampled.")

    # ---------- within-screen cross-endpoint ----------
    ce = cluster_bootstrap_corr(df.y_primary.to_numpy(), df.stored_ad.to_numpy(),
                                df.target_ensg.to_numpy(), B=10000, seed=20260801)
    ce_log = cluster_bootstrap_corr(df.y_primary.to_numpy(), df.stored_ad_log1p.to_numpy(),
                                    df.target_ensg.to_numpy(), B=10000, seed=20260801)
    cross = dict(construct_level=dict(vs_stored_raw=ce, vs_stored_log1p=ce_log),
                 reference_session6=dict(rpe1_spearman=0.165, gwps_spearman=0.396),
                 note="within-K562-essential association between matched-size primary endpoint and stored "
                      "anderson_darling_counts; cluster bootstrap over target genes; descriptive (prereg 3.3/3.6). "
                      "Not a transfer metric -- no prediction vector involved.")
    json.dump(cross, open(f"{OUT}/cross_endpoint.json", "w"), indent=2)

    dur = time.time() - t_start
    end_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time()))
    peak = rss_gb()
    print(f"[{DS}] DONE in {dur:.1f}s  peakRSS={peak:.2f}GB")
    print(f"[{DS}] median y_primary={np.nanmedian(yprim):.6f}  spread ratio con={ratio_con:.4f} tgt={ratio_tgt:.4f}")
    print(f"[{DS}] pooled-vs-matched Spearman={pvm['spearman']:.4f} Pearson={pvm['pearson']:.4f}")
    print(f"[{DS}] gemgroup qualifying={len(gsub)} corr Spearman={gcorr.get('spearman') if gcorr else None}")
    print(f"[{DS}] cross-endpoint (raw) Spearman={ce['spearman']:.4f} Pearson={ce['pearson']:.4f}")

    # ---------- run_metadata.json ----------
    def statmeta(p):
        st = os.stat(p)
        return dict(path=p, size_bytes=int(st.st_size),
                    mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)))
    sc_meta = statmeta(cfg["sc"]); sc_meta["sha256"] = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
    bu_meta = statmeta(cfg["bulk"]); bu_meta["sha256"] = sha256_file(cfg["bulk"])
    meta = dict(
        dataset=DS, session="SESSION 11",
        start_timestamp=start_wall, end_timestamp=end_wall, wall_clock_sec=dur,
        inputs=dict(single_cell=sc_meta, bulk=bu_meta),
        exact_command="PYTHONDONTWRITEBYTECODE=1 python -B run_endpoint.py",
        package_versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__, scipy=scipy.__version__,
                              anndata=ad.__version__, h5py=h5py.__version__),
        peak_rss_gb=peak, n_eligible=n_con, n_blocks=len(blocks), chunk=CHUNK,
        n_NT=int(n_NT), n_genes=int(n_genes), n_cells=int(n_cells),
        distinct_target_cols=int(len(elig_cols)),
        resampling_seeds=[int(s) for s in SEEDS],
        one_construct_probe=probe_info,
        median_y_primary=float(np.nanmedian(yprim)),
        n_nan_y_primary=int(df.y_primary.isna().sum()),
        gemgroup_summary=gj,
        warnings=CAPTURED_WARNINGS[:50], n_warnings=len(CAPTURED_WARNINGS),
        deviations="none",
        hostname=platform.node(),
    )
    json.dump(meta, open(f"{OUT}/run_metadata.json", "w"), indent=2)
    print(f"[{DS}] run_metadata.json + gemgroup summary written; warnings={len(CAPTURED_WARNINGS)}")


if __name__ == "__main__":
    main()
