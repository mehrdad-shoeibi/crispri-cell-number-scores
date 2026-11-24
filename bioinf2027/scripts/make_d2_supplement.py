"""D2 supplement after the first results review (bioinf2027/reviews/D2_results_r1.md), from saved files only.

No new fit and no GPU time. Every value is secondary and unadjusted; none enters D2's family of 18 or
changes a reading of bioinf2027/results/D2/readings.csv.

Pre-specified in bioinf2027/plans/D2.md and fitted, not computed by make_d2_stats.py (review, blocking 1;
non-blocking 7):
- existing_label.csv: the existing (all-cells) label in G5 and G6 (Sections 10.1, item 4 and 10.3, item
  5): R2 of z, z-, z~, C1 (forest) and C1 (boosting, the submitted comparator); dR2_token, the noise check,
  dR2_rest with either C1, the share lost, and R2(z-) - R2(C1 boosting) beside E1b's post hoc (a0),
  R2(z) - R2(C1 boosting), which in G6 is the submitted statistic; batch-resampled intervals beside the
  gene-clustered ones; externally drho_token, the noise check, T(z) and T(z-). No reading table was
  fixed for this label: any reading of these values is post hoc.
- heads.csv: dR2_token, dR2_rest, drho_token and T(z-) under boosting (seeds 0 to 2) and ridge (seed 0),
  rebuilt and existing labels (Section 10.2, item 12).
- refits.csv: the 20 refits of z- paired with E1b's refits of z on the same training genes; the share
  of refits with the sign of the family's dR2_token and drho_token (Section 10.2, item 12).
- batch_resampled.csv: the family's VCC statistics and both preconditions with batch-clustered
  replicates (E1b's seed 20270302; Section 10.2, item 12).
- absolute.csv: VCC-test R2 and Spearman of z, z-, z~, C1 and TR, and the Spearman of the raw B0 to B4;
  external Spearman of z, z-, z~, C1, TR and the raw B0 to B4 (Sections 6 and 10.2, item 12).
- availability.csv: drho_token, rho(z), rho(z-), rho(C1) and T(z-) on the constructs whose target is and
  is not one of the 2,000 features (Section 6; for z and z- the split is not an availability split).
- manipulation_summary.csv: medians and quartiles of h_p, h_c, the truncated share, ||z - z-|| / ||z||
  and ||z - z~|| / ||z|| per side and split (Sections 10.1, item 2 and 10.3, item 2; review, blocking 2).
- reach.csv: Spearman of h_p, h_c and the truncated share against B1 and B3, per side (Sections 10.2,
  item 13 and 11.3, item 3).
- delta_batch.csv: per VCC target gene, the median batch deviation of delta_g (from
  delta_per_flex_run.csv, Section 11.2, item 3) beside a sampling reference, the same median with the
  gene's 480 delta cells' batch labels permuted (20 permutations, seed 20271108). The reference is post
  hoc (review, non-blocking 7).

Post hoc, asked for by the review (missing analyses 1, 2 and 4; non-blocking 4, 6 and 8):
- positive_control_absolute.csv: VCC-test R2 of z and z- predicting B2 native and B3, G5 and G6.
- agreement.csv: per cell and screen, the Spearman between the seed-averaged z and z- predictions, and of
  each with ||z|| and ||z-||; per screen the Spearman between ||z|| and ||z-|| (point values).
- c2_paired.csv: drho_token - (rho(C2) - rho(C2-)) from the joint draw, per cell and screen.
- identity_population.csv: per part of the embedding run, the largest |dropped - cached| over the cells
  whose input does not hold the target's token, where the two must agree under the job code.
- family_read_status.csv: the read status of each family row, with reading (f) on the VCC rows, and the
  Monte Carlo standard error of each p, sqrt(p (1 - p) / B).

Outputs under bioinf2027/results/D2/supplement/ with supplement.json. With --smoke: 20 replicates, two
permutations, the identity check on the first two units of each part, outputs to
$IGA_STORAGE_ROOT/bioinf2027/scratch/D2_supplement_smoke/, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, resample  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import make_d2_stats as d2s  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
D2 = REPO / "bioinf2027" / "results" / "D2"
OUT_DIR = D2 / "supplement"
PLAN = "bioinf2027/plans/D2.md, Sections 6, 10 and 11; bioinf2027/reviews/D2_results_r1.md"
BATCH_SEED = 20270302  # E1b's batch-clustered VCC replicates
PERMUTATION_SEED = 20271108
PERMUTATIONS = 20
SMOKE = False
PRIMARY = tuple(e1cells.PRIMARY)
HEAD_SEEDS = {"forest": (0, 1, 2), "boosting": (0, 1, 2), "ridge": (0,)}
PARTS = ("vcc_perturbed", "vcc_delta", "rpe1_perturbed", "rpe1_delta", "k562_perturbed", "k562_delta")
SECONDARY = "secondary, unadjusted, pre-specified"
POST_HOC = "secondary, unadjusted, post hoc (results review r1)"
STATE: dict = {}


# --------------------------------------------------------------------------- predictions


def fits(cell, arm, label="rebuilt", head="forest", side="test", seeds=None, variant="all training rows"):
    """Per-seed predictions: z, C1, TR and C2 from E1b; z-, z~ and C2- from D2. None if any is missing."""
    seeds = HEAD_SEEDS[head] if seeds is None else seeds
    out = []
    for seed in seeds:
        if arm in ("z", "C1", "TR", "C2"):
            values = d2s.predictions(d2s.DATA["e1b"], FitSpec(cell, label, arm, head, seed, variant))
        elif arm == "C2-":
            values = d2s.predictions(d2s.DATA["d2"], FitSpec(f"D2 | {cell} | x-", label, "C2", head, seed, variant))
        elif arm in ("z-", "z~"):
            values = d2s.predictions(d2s.DATA["d2"], FitSpec(f"D2 | {cell} | {arm}", label, "z", head, seed, variant))
        else:
            raise ValueError(arm)
        if values is None or side not in values:
            return None
        out.append(values[side])
    return out


def vcc_label(cell: str, label: str) -> np.ndarray:
    column = {"rebuilt": "y", "existing": "existing_label"}[label]
    return d2s.DATA["test_order"][cell][column].to_numpy(dtype=np.float64)


def batch_draw(cell: str) -> dict:
    key = ("batch", cell)
    if key not in STATE:
        groups = d2s.DATA["test_order"][cell]["batch"].astype(str).to_numpy()
        STATE[key] = {"draw": resample.cluster_draw(groups, d2s.B, BATCH_SEED), "point": d2s.ones(groups.size)}
    return STATE[key]


def r2_with(draw: dict, vectors, label):
    if vectors is None:
        return None
    columns = np.column_stack(vectors)
    return float(resample.weighted_r2(label, columns, draw["point"])[0].mean()), resample.weighted_r2(label, columns, draw["draw"]).mean(axis=1)


def spearman_with(draw: dict, vectors, label, mask=None):
    if vectors is None:
        return None
    columns = np.column_stack(vectors)
    return (float(resample.weighted_spearman(columns, label, draw["point"], mask=mask)[0].mean()),
            resample.weighted_spearman(columns, label, draw["draw"], mask=mask).mean(axis=1))


def gene_draw(cell: str) -> dict:
    return d2s.vcc_draw(cell)


def row(cell, scope, quantity, value, adjustment, **extra) -> dict:
    return {"cell": cell, "scope": scope, "quantity": quantity, **d2s.summary(value), "adjustment": adjustment, **extra}


def point_rho(screen: str, vectors, label) -> float:
    columns = np.column_stack(vectors)
    return float(resample.weighted_spearman(columns, label, d2s.DATA["draws"][screen]["point"])[0].mean())


def point_r2(cell: str, vectors, label) -> float:
    columns = np.column_stack(vectors)
    return float(resample.weighted_r2(label, columns, gene_draw(cell)["point"])[0].mean())


# --------------------------------------------------------------------------- pre-specified secondaries


def existing_label_rows() -> list:
    rows = []
    for cell in PRIMARY:
        y = vcc_label(cell, "existing")
        note = "the submitted statistic's configuration (all modelled rows, batch-local features)" if cell == "G6" else "not the submitted statistic"
        for kind, draw in (("target-gene clusters", gene_draw(cell)), ("batch clusters", batch_draw(cell))):
            z, zm, zt, c1 = (r2_with(draw, fits(cell, a, "existing"), y) for a in ("z", "z-", "z~", "C1"))
            c1b = r2_with(draw, fits(cell, "C1", "existing", "boosting"), y)
            extra = {"replicates": kind, "note": note}
            for name, value in (("R2(z)", z), ("R2(z-)", zm), ("R2(z~)", zt), ("R2(C1)", c1), ("R2(C1, boosting)", c1b),
                                ("dR2_token = R2(z) - R2(z-)", d2s.diff(z, zm)), ("noise check R2(z~) - R2(z-)", d2s.diff(zt, zm)),
                                ("dR2_rest = R2(z-) - R2(C1)", d2s.diff(zm, c1)), ("R2(z) - R2(C1)", d2s.diff(z, c1)),
                                ("(a0) R2(z) - R2(C1, boosting)", d2s.diff(z, c1b)), ("R2(z-) - R2(C1, boosting)", d2s.diff(zm, c1b))):
                rows.append(row(cell, "VCC test, existing label", name, value, SECONDARY, **extra))
            if None not in (z, zm, c1):
                rows.append({"cell": cell, "scope": "VCC test, existing label", "quantity": "share lost (R2(z) - R2(z-)) / (R2(z) - R2(C1))",
                             **d2s.share_lost(z, zm, c1), "adjustment": SECONDARY, **extra})
        for screen in d2s.SCREENS:
            label, _, _ = d2s.external(cell, screen)
            rz, rzm, rzt, rc1 = (d2s.rho(screen, fits(cell, a, "existing", side=screen), label) for a in ("z", "z-", "z~", "C1"))
            for name, value in (("drho_token = rho(z) - rho(z-)", d2s.diff(rz, rzm)), ("noise check rho(z~) - rho(z-)", d2s.diff(rzt, rzm)),
                                ("T(z)", d2s.diff(rz, rc1)), ("T(z-)", d2s.diff(rzm, rc1))):
                rows.append(row(cell, f"{screen}, existing label", name, value, SECONDARY, replicates="target-gene clusters", note=note))
    return rows


def head_rows() -> list:
    rows = []
    for cell in PRIMARY:
        for label in ("rebuilt", "existing"):
            y = vcc_label(cell, label)
            for head in ("boosting", "ridge"):
                z, zm, c1 = (d2s.r2(cell, fits(cell, a, label, head), y) for a in ("z", "z-", "C1"))
                extra = {"head": head, "label": label, "seeds": len(HEAD_SEEDS[head])}
                rows.append(row(cell, "VCC test", "dR2_token", d2s.diff(z, zm), SECONDARY, **extra))
                rows.append(row(cell, "VCC test", "dR2_rest", d2s.diff(zm, c1), SECONDARY, **extra))
                rows.append(row(cell, "VCC test", "R2(z) - R2(C1)", d2s.diff(z, c1), SECONDARY, **extra))
                for screen in d2s.SCREENS:
                    lab, _, _ = d2s.external(cell, screen)
                    rz, rzm, rc1 = (d2s.rho(screen, fits(cell, a, label, head, side=screen), lab) for a in ("z", "z-", "C1"))
                    rows.append(row(cell, screen, "drho_token", d2s.diff(rz, rzm), SECONDARY, **extra))
                    rows.append(row(cell, screen, "T(z-)", d2s.diff(rzm, rc1), SECONDARY, **extra))
    return rows


def refit_rows(family: pd.DataFrame) -> list:
    rows = []
    estimate = {(r.cell, r.scope, r.statistic): r.estimate for r in family.itertuples()}
    refits = json.loads((D2 / "fit_protocol.json").read_text())["refits"]
    for cell in PRIMARY:
        y = vcc_label(cell, "rebuilt")
        scopes = [("VCC test", "dR2_token")] + [(s, "drho_token") for s in d2s.SCREENS]
        for scope, statistic in scopes:
            values = []
            for r in range(refits):
                side = "test" if scope == "VCC test" else scope
                a = fits(cell, "z", seeds=(0,), variant=f"refit {r}", side=side)
                b = fits(cell, "z-", seeds=(0,), variant=f"refit {r}", side=side)
                if a is None or b is None:
                    continue
                if scope == "VCC test":
                    values.append(point_r2(cell, a, y) - point_r2(cell, b, y))
                else:
                    label, _, _ = d2s.external(cell, scope)
                    values.append(point_rho(scope, a, label) - point_rho(scope, b, label))
            values = np.array(values)
            e = estimate[(cell, scope, statistic)]
            rows.append({"cell": cell, "scope": scope, "statistic": statistic, "family_estimate": e, "refits": int(values.size),
                         "refit_p2.5": float(np.percentile(values, 2.5)) if values.size else float("nan"),
                         "refit_median": float(np.median(values)) if values.size else float("nan"),
                         "refit_p97.5": float(np.percentile(values, 97.5)) if values.size else float("nan"),
                         "share_with_the_sign_of_the_family_estimate": float((np.sign(values) == np.sign(e)).mean()) if values.size else float("nan"),
                         "pairing": "refit r of z- and of z drew the same training genes (E1b's refit seed 20270414 and context index)",
                         "adjustment": SECONDARY})
    return rows


def batch_rows() -> list:
    rows = []
    for cell in PRIMARY:
        y = vcc_label(cell, "rebuilt")
        draw = batch_draw(cell)
        z, zm, zt, c1 = (r2_with(draw, fits(cell, a), y) for a in ("z", "z-", "z~", "C1"))
        for name, value in (("dR2_token", d2s.diff(z, zm)), ("dR2_rest", d2s.diff(zm, c1)), ("VCC noise check R2(z~) - R2(z-)", d2s.diff(zt, zm)),
                            ("precondition of z: R2(z) - R2(C1)", d2s.diff(z, c1)), ("precondition of z-: R2(z-) - R2(C1)", d2s.diff(zm, c1))):
            rows.append(row(cell, "VCC test", name, value, SECONDARY, replicates="batch clusters, seed 20270302"))
    return rows


def absolute_rows() -> list:
    rows = []
    for cell in d2s.CELLS:
        y = vcc_label(cell, "rebuilt")
        draw = gene_draw(cell)
        test = d2s.DATA["test_order"][cell]
        for arm in ("z", "z-", "z~", "C1", "TR"):
            vectors = fits(cell, arm)
            rows.append(row(cell, "VCC test", f"R2({arm})", r2_with(draw, vectors, y), SECONDARY))
            rows.append(row(cell, "VCC test", f"Spearman({arm})", spearman_with(draw, vectors, y), SECONDARY))
        for b in e1cells.TRIVIAL:
            rows.append(row(cell, "VCC test", f"Spearman(raw {b})", spearman_with(draw, [test[b].to_numpy(dtype=np.float64)], y), SECONDARY))
        for screen in d2s.SCREENS:
            label, cov, _ = d2s.external(cell, screen)
            ext = d2s.DATA["draws"][screen]
            for arm in ("z", "z-", "z~", "C1", "TR"):
                rows.append(row(cell, screen, f"Spearman({arm})", spearman_with(ext, fits(cell, arm, side=screen), label), SECONDARY))
            for k, b in enumerate(e1cells.TRIVIAL):
                rows.append(row(cell, screen, f"Spearman(raw {b})", spearman_with(ext, [cov[:, k]], label), SECONDARY))
    return rows


def availability_rows() -> list:
    rows = []
    for cell in PRIMARY:
        for screen in d2s.SCREENS:
            label, _, _ = d2s.external(cell, screen)
            feature = d2s.DATA["tables"][cell][screen]["target_is_feature"].to_numpy(dtype=bool)
            for subset, mask in (("target is a feature", feature), ("target is not a feature", ~feature)):
                rz, rzm, rc1 = (d2s.rho(screen, fits(cell, a, side=screen), label, mask) for a in ("z", "z-", "C1"))
                extra = {"subset": subset, "constructs": int(mask.sum()),
                         "note": "for z and z- the split is not an availability split (bioinf2027/plans/D2.md, Section 6)"}
                for name, value in (("drho_token", d2s.diff(rz, rzm)), ("Spearman(z)", rz), ("Spearman(z-)", rzm), ("Spearman(C1)", rc1),
                                    ("T(z-)", d2s.diff(rzm, rc1))):
                    rows.append(row(cell, screen, name, value, SECONDARY, **extra))
    return rows


# --------------------------------------------------------------------------- the manipulation


def manipulation_table() -> pd.DataFrame:
    record = json.loads((D2 / "aggregate.json").read_text())["manipulation"]
    if sha256(D2 / "manipulation.csv") != record["sha256"]:
        raise RuntimeError("manipulation.csv does not match aggregate.json")
    return pd.read_csv(D2 / "manipulation.csv", float_precision="round_trip")


def manipulation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = frame.assign(split=frame["split"].fillna("all constructs"))
    groups = list(frame.groupby(["side", "split"], sort=True)) + [((side, "all rows"), part) for side, part in frame.groupby("side", sort=True) if side == "VCC"]
    for (side, split), part in groups:
        entry = {"side": side, "split": split, "rows": int(len(part)),
                 "share_with_h_c_and_h_p_below_0.05": float(((part["h_c"] < 0.05) & (part["h_p"] < 0.05)).mean())}
        columns = (("h_p", "h_p"), ("h_c", "h_c"), ("truncated_share", "truncated_share"))
        columns += ((("relative_change", "relative_change_batch"), ("relative_change_noise", "relative_change_noise_batch"), ("relative_change_pooled", "relative_change_pooled"))
                    if side == "VCC" else (("relative_change", "relative_change"), ("relative_change_noise", "relative_change_noise")))
        for name, column in columns:
            values = part[column].to_numpy(dtype=np.float64)
            entry[f"{name} median"] = float(np.median(values))
            entry[f"{name} p25"] = float(np.percentile(values, 25))
            entry[f"{name} p75"] = float(np.percentile(values, 75))
        entry["ratio of the medians, relative_change / relative_change_noise"] = entry["relative_change median"] / entry["relative_change_noise median"]
        entry["reference"] = "batch reference (G5, G6)" if side == "VCC" else "pooled controls"
        rows.append(entry)
    return pd.DataFrame(rows)


def reach_rows(frame: pd.DataFrame) -> list:
    rows = []
    vcc = d2s.DATA["tables"]["G6"]["vcc"]
    m = frame[frame["side"] == "VCC"].merge(vcc[["split", "batch", "target_gene", "B1", "B3"]], left_on=["split", "batch", "gene"],
                                              right_on=["split", "batch", "target_gene"], how="left", validate="one_to_one")
    sides = [("VCC", m)]
    for screen in d2s.SCREENS:
        table = d2s.DATA["tables"]["G6"][screen][["construct", "B1", "B3"]].astype({"construct": str})
        part = frame[frame["side"] == screen].astype({"construct": str})
        sides.append((screen, part.merge(table, on="construct", how="left", validate="one_to_one")))
    for side, part in sides:
        if part[["B1", "B3"]].isna().any().any():
            raise RuntimeError("a %s row of manipulation.csv has no B1 or B3" % side)
        for quantity in ("h_p", "h_c", "truncated_share"):
            for b in ("B1", "B3"):
                x = part[quantity].to_numpy(dtype=np.float64)
                constant = np.unique(x).size == 1
                value = float("nan") if constant else float(sps.spearmanr(x, part[b].to_numpy(dtype=np.float64)).statistic)
                rows.append({"side": side, "quantity": quantity, "against": b, "spearman": value, "rows": int(len(part)),
                             "note": "constant: every value is %g" % x[0] if constant else "", "adjustment": SECONDARY})
    return rows


def delta_batch_rows() -> list:
    import make_d2_aggregate as d2agg

    delta_frame, delta_dropped = d2agg.load_part("vcc_delta")
    effects = delta_dropped - d2agg.cached(delta_frame)
    saved = pd.read_csv(D2 / "delta_per_flex_run.csv", float_precision="round_trip").set_index("gene")
    genes = sorted(set(delta_frame["row_gene"]))
    permutations = 2 if SMOKE else PERMUTATIONS
    rows = []
    for g_index, gene in enumerate(genes):
        sel = (delta_frame["row_gene"] == gene).to_numpy()
        e = effects[sel]
        batches = delta_frame.loc[sel, "batch"].astype(str).to_numpy()
        labels = sorted(set(batches))

        def median_deviation(b):
            mean = e.mean(axis=0)
            return float(np.median([np.linalg.norm(e[b == lab].mean(axis=0) - mean) for lab in labels]))

        observed = median_deviation(batches)
        reference = [median_deviation(np.random.default_rng([PERMUTATION_SEED, g_index, k]).permutation(batches)) for k in range(permutations)]
        rows.append({"gene": gene, "cells": int(sel.sum()), "batches": len(labels), "delta_norm": float(saved.loc[gene, "delta_norm"]),
                     "median_row_norm_of_z_minus_z_minus": float(saved.loc[gene, "median_row_norm_of_z_minus_z_minus"]),
                     "batch_deviation_norm_median": observed, "saved_batch_deviation_norm_median": float(saved.loc[gene, "batch_deviation_norm_median"]),
                     "permuted_reference_mean": float(np.mean(reference)), "permuted_reference_p95": float(np.percentile(reference, 95)),
                     # a gene whose delta cells all lack its token has every effect 0 and no ratio
                     "ratio_to_reference": observed / float(np.mean(reference)) if np.mean(reference) > 0 else float("nan"),
                     "permutations": permutations,
                     "adjustment": "secondary, unadjusted; the permutation reference is post hoc (results review r1, non-blocking 7)"})
    return rows


# --------------------------------------------------------------------------- post hoc, asked for by the review


def positive_control_rows() -> list:
    rows = []
    for cell in PRIMARY:
        for name in ("B2", "B3"):
            label = d2s.test_label(cell, name)
            for arm in ("z", "z-"):
                rows.append(row(cell, "VCC test", f"R2({arm}) predicting {name}{' native' if name == 'B2' else ''}",
                                d2s.r2(cell, d2s.seeds_of_own(cell, arm, name), label), POST_HOC))
    return rows


def agreement_rows() -> list:
    import make_e1b_fits as e1b

    record = json.loads((D2 / "aggregate.json").read_text())["arms"]
    path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["sha256"]:
        raise RuntimeError("arms.npz does not match aggregate.json")
    rows = []
    with np.load(path, allow_pickle=False) as arms:
        for screen in d2s.SCREENS:
            inputs = e1b.DATA["screens"][screen]
            if not np.array_equal(arms[f"{screen}_constructs"].astype(str), np.asarray(inputs.constructs).astype(str)):
                raise RuntimeError("the arms list the %s constructs in another order" % screen)
            norm_z = np.linalg.norm(np.asarray(inputs.z, dtype=np.float64), axis=1)
            norm_zm = np.linalg.norm(arms[f"{screen}_z_minus"].astype(np.float64), axis=1)
            rows.append({"cell": "all", "screen": screen, "quantity": "Spearman(||z||, ||z-||) over constructs",
                         "value": float(sps.spearmanr(norm_z, norm_zm).statistic), "constructs": int(norm_z.size), "adjustment": POST_HOC})
            for cell in PRIMARY:
                table = d2s.DATA["tables"][cell][screen]
                if not np.array_equal(table["construct"].astype(str).to_numpy(), np.asarray(inputs.constructs).astype(str)):
                    raise RuntimeError("the %s table of %s lists other constructs" % (screen, cell))
                pz = np.mean(fits(cell, "z", side=screen), axis=0)
                pzm = np.mean(fits(cell, "z-", side=screen), axis=0)
                for name, a, b in (("Spearman(z prediction, z- prediction)", pz, pzm), ("Spearman(z prediction, ||z||)", pz, norm_z),
                                   ("Spearman(z- prediction, ||z-||)", pzm, norm_zm), ("Spearman(z- prediction, ||z||)", pzm, norm_z)):
                    rows.append({"cell": cell, "screen": screen, "quantity": name, "value": float(sps.spearmanr(a, b).statistic),
                                 "constructs": int(a.size), "adjustment": POST_HOC})
    return rows


def c2_paired_rows() -> list:
    rows = []
    for cell in PRIMARY:
        for screen in d2s.SCREENS:
            label, _, _ = d2s.external(cell, screen)
            rz, rzm, rc2, rc2m = (d2s.rho(screen, fits(cell, a, side=screen), label) for a in ("z", "z-", "C2", "C2-"))
            value = d2s.diff(d2s.diff(rz, rzm), d2s.diff(rc2, rc2m))
            rows.append(row(cell, screen, "drho_token - (rho(C2) - rho(C2-))", value, POST_HOC))
    return rows


def identity_rows() -> list:
    import make_d2_aggregate as d2agg
    import make_d2_embeddings as d2emb

    rows = []
    for part in PARTS:
        if SMOKE:
            units = d2emb.load_parts([part])[part]
            keep = sorted(units["unit"].unique())[:2]
            original = d2emb.load_parts

            def limited(parts, _keep=keep, _part=part, _original=original):
                out = _original(parts)
                out[_part] = out[_part][out[_part]["unit"].isin(_keep)]
                return out

            d2emb.load_parts = limited
            try:
                frame, dropped = d2agg.load_part(part)
            finally:
                d2emb.load_parts = original
        else:
            frame, dropped = d2agg.load_part(part)
        absent = ~frame["target_in_input"].astype(bool).to_numpy()
        difference = np.abs(dropped[absent] - d2agg.cached(frame[absent].reset_index(drop=True)))
        per_cell = difference.max(axis=1) if difference.size else np.array([])
        rows.append({"part": part, "cells": int(len(frame)), "cells_without_the_target_in_input": int(absent.sum()),
                     "max_abs_difference": float(per_cell.max()) if per_cell.size else float("nan"),
                     "cells_above_1e-4": int((per_cell > 1e-4).sum()), "cells_above_0": int((per_cell > 0).sum()),
                     "check": "dropped equals cached where the input does not hold the target's token (results review r1, non-blocking 8)",
                     "adjustment": POST_HOC})
        del frame, dropped, difference
    return rows


def read_status(family: pd.DataFrame) -> pd.DataFrame:
    pre = pd.read_csv(REPO / "bioinf2027" / "results" / "E1" / "preconditions.csv")
    passes = {c: bool(p) for c, a, lab, p in zip(pre["cell"], pre["arm"], pre["label"], pre["passes"]) if a == "z" and lab == "rebuilt"}
    out = family[["cell", "scope", "statistic", "estimate", "p", "holm_threshold", "state"]].copy()
    status = []
    for r in family.itertuples():
        if r.scope == "VCC test" and not passes[r.cell]:
            status.append("not read: reading (f), z fails its VCC precondition in this cell")
        elif r.statistic == "R(z-)" and not r.r5_gate_passed:
            status.append("not read: E1's R5 gate fails on this screen")
        elif r.uninformative_cell:
            status.append("not read: uninformative cell")
        else:
            status.append("read: %s; external statement in readings.csv" % r.state)
    out["read_status"] = status
    out["monte_carlo_se_of_p"] = np.sqrt(out["p"] * (1.0 - out["p"]) / d2s.B)
    out["p_minus_threshold_in_monte_carlo_se"] = (out["p"] - out["holm_threshold"]) / out["monte_carlo_se_of_p"]
    return out


# --------------------------------------------------------------------------- main


def setup() -> dict:
    import make_e1b_fits as e1b

    d2s.DATA["e1b"] = d2s.manifest(REPO / "bioinf2027" / "results" / "E1" / "predictions_manifest.json")
    d2s.DATA["d2"] = d2s.manifest(D2 / "predictions_manifest.json")
    d2s.DATA["tables"] = {cell: e1cells.load_tables(cell) for cell in d2s.CELLS}
    d2s.DATA["vcc_draws"] = {}
    harness = e1b.load_inputs(list(d2s.CELLS))
    d2s.DATA["test_order"] = {cell: e1b.DATA["cells"][cell].vcc[e1b.DATA["cells"][cell].split("test")] for cell in d2s.CELLS}
    d2s.DATA["draws"] = {}
    for screen in d2s.SCREENS:
        genes = d2s.DATA["tables"]["G1"][screen]["target_ensg"].astype(str).to_numpy()
        d2s.DATA["draws"][screen] = {"draw": resample.cluster_draw(genes, d2s.B, d2s.EXT_SEED[screen]), "point": d2s.ones(genes.size)}
    return harness


def counts() -> dict:
    out = {}
    for cell in d2s.CELLS:
        test = d2s.DATA["test_order"][cell]
        out[f"{cell} VCC test"] = {"rows": int(len(test)), "target_genes": int(test["target_gene"].nunique())}
        for screen in d2s.SCREENS:
            table = d2s.DATA["tables"][cell][screen]
            out[f"{cell} {screen}"] = {"constructs": int(len(table)), "target_genes": int(table["target_ensg"].nunique())}
    return out


def main() -> None:
    started = time.time()
    harness = setup()
    family = pd.read_csv(D2 / "family.csv", float_precision="round_trip")
    frame = manipulation_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "family_read_status.csv": lambda: read_status(family),
        "existing_label.csv": lambda: pd.DataFrame(existing_label_rows()),
        "heads.csv": lambda: pd.DataFrame(head_rows()),
        "refits.csv": lambda: pd.DataFrame(refit_rows(family)),
        "batch_resampled.csv": lambda: pd.DataFrame(batch_rows()),
        "absolute.csv": lambda: pd.DataFrame(absolute_rows()),
        "availability.csv": lambda: pd.DataFrame(availability_rows()),
        "manipulation_summary.csv": lambda: manipulation_summary(frame),
        "reach.csv": lambda: pd.DataFrame(reach_rows(frame)),
        "positive_control_absolute.csv": lambda: pd.DataFrame(positive_control_rows()),
        "agreement.csv": lambda: pd.DataFrame(agreement_rows()),
        "c2_paired.csv": lambda: pd.DataFrame(c2_paired_rows()),
        "delta_batch.csv": lambda: pd.DataFrame(delta_batch_rows()),
        "identity_population.csv": lambda: pd.DataFrame(identity_rows()),
    }
    outputs, seconds = {}, {}
    for name, build in tables.items():
        t = time.time()
        build().to_csv(OUT_DIR / name, index=False)
        outputs[name] = file_record(OUT_DIR / name)
        seconds[name] = round(time.time() - t, 1)
        print("[d2 supplement] %s in %.0f s" % (name, seconds[name]), flush=True)
    batch = pd.read_csv(OUT_DIR / "delta_batch.csv", float_precision="round_trip")
    ratio = batch["ratio_to_reference"].dropna()
    delta_batch_summary = {"genes": int(len(batch)), "genes_with_a_reference": int(ratio.size),
                           "genes_without_a_reference (every delta effect 0)": int(batch["ratio_to_reference"].isna().sum()),
                           "ratio_to_reference median": float(ratio.median()), "ratio_to_reference p10": float(ratio.quantile(0.1)),
                           "ratio_to_reference p90": float(ratio.quantile(0.9)),
                           "genes above the reference's 95th percentile": int((batch["batch_deviation_norm_median"] > batch["permuted_reference_p95"]).sum()),
                           "largest |observed - saved| batch deviation": float((batch["batch_deviation_norm_median"] - batch["saved_batch_deviation_norm_median"]).abs().max())}
    stats = {"plan": PLAN, "replicates": d2s.B, "smoke": SMOKE, "delta_batch_summary": delta_batch_summary,
             "seeds": {"external": d2s.EXT_SEED, "vcc genes": d2s.VCC_SEED, "vcc batches": BATCH_SEED, "permutations": PERMUTATION_SEED},
             "permutations": 2 if SMOKE else PERMUTATIONS,
             "replicate_index_sha256": {s: d2s.DATA["draws"][s]["draw"].index_sha256 for s in d2s.SCREENS},
             "n": counts(),
             "inputs": {"family": file_record(D2 / "family.csv"), "e1b_predictions": file_record(REPO / "bioinf2027" / "results" / "E1" / "predictions_manifest.json"),
                        "d2_predictions": file_record(D2 / "predictions_manifest.json"), "aggregate": file_record(D2 / "aggregate.json"),
                        "manipulation": file_record(D2 / "manipulation.csv"), "delta_per_flex_run": file_record(D2 / "delta_per_flex_run.csv"),
                        "units": file_record(D2 / "units.json"), "fit_protocol": file_record(D2 / "fit_protocol.json"),
                        "preconditions": file_record(REPO / "bioinf2027" / "results" / "E1" / "preconditions.csv"), "e1b_harness": harness},
             "outputs": outputs, "seconds_per_table": seconds, "seconds": round(time.time() - started, 1)}
    (OUT_DIR / "supplement.json").write_text(json.dumps(strict_json(stats), indent=1, allow_nan=False) + "\n")
    print("[d2 supplement] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["--smoke"]:
        SMOKE = True
        d2s.B = 20
        OUT_DIR = config.storage_root() / "bioinf2027" / "scratch" / "D2_supplement_smoke"
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    main()
