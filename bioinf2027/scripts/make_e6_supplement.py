"""E6 supplement to the family pass, from saved predictions only (no fit).

Decided by the author at gate G2 (bioinf2027/GATE.md, top section, item 11): one supplement from saved
predictions with one extra audit, before the full pass. It computes the analyses that the results review
(bioinf2027/reviews/E6_results_r1.md, non-blocking 1, 2, 3, 5 and missing analyses 1 to 6) asked for and that
the family pass did not write. Every value is secondary and unadjusted, and none changes a family state or a
reading of bioinf2027/results/E6/family_pass/. The refit check of departure 1 (missing analysis 6, new fits)
is not run: the author accepted departure 1.

Outputs under bioinf2027/results/E6/family_pass/supplement/:
- absolute.csv: the external Spearman of each arm (z, x, m, [x; m], the masked arms), C1 and C2 at 27/788
  under the forest (head seeds 0 to 2), and of the target screen's raw B1 and B2, with intervals (review,
  non-blocking 1; missing 1).
- folds.csv: the Spearman within each gene fold and the pooled Spearman with predictions centred within fold,
  for C1 and each family arm (plan Section 13.1, item 8; missing 2). Point values, as the full pass writes them.
- availability.csv: T (arm minus C1) on the constructs whose target is and is not one of the features of x,
  for x, m, [x; m], C2 and the masked arms (read for expression-derived arms only) and for z and C3 (reported;
  not an availability split for them) (plan Section 6, item 3; E1_design.md, Section 10; review, non-blocking 5;
  missing 3).
- gene_fold.csv: T and R of each family arm under the primary gene-fold assignment at head seed 0 and under
  the two other assignments (seeds 20271207 and 20271208) at head seed 0, each state set against the Holm
  threshold of the statistic's step in the primary ordering, and the marker "not robust to the gene-fold
  assignment" under the plan's rule beside the family pass's (plan Section 13.1, item 2; departure 7; review,
  non-blocking 2; missing 4). The rule is bioinf2027/scripts/make_e6_stats.py, gene_fold_value and
  assignment_state, which the full pass uses.
- one_sided.csv: the one-sided p-values of plan Section 4.1 (R >= 0.10; the [TR; arm] increment over TR >=
  0.03) where T is Holm-positive, R is not and the R5 gate passes, with no statement attached (departure 8;
  review, non-blocking 3; missing 5); make_e6_stats.one_sided_rows, which the full pass uses. The [TR; arm]
  fits are tier-3 fits.
- contrasts.csv: z minus x, z minus [x; m] and z minus C2 in external Spearman, per direction (plan Section 6,
  item 7; missing 6).
- supplement.json: inputs, the checks below, replicate digests and seconds.

Checks before any value is written (stop on failure; hard limit 9):
1. every data-root input equals its record (make_e6_stats.verify_inputs, stop rule 1);
2. every prediction of tiers 1 and 2 is listed in the tier-3 manifest with the same digest, so the family
   pass's predictions are the ones read here;
3. T and R of the four family arms in both directions (estimate, lo, hi, p) equal family_pass/family.csv to
   1e-12;
4. the replicate index digests equal E1b's.

With --smoke: 20 replicates, outputs to $IGA_STORAGE_ROOT/bioinf2027/scratch/E6_supplement_smoke/, check 3 on
point values only, never reported.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, resample  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

import make_e6_fits as fits  # noqa: E402
import make_e6_stats as e6s  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E6 = REPO / "bioinf2027" / "results" / "E6"
FAMILY_PASS = E6 / "family_pass"
OUT_DIR = FAMILY_PASS / "supplement"
PLAN = "bioinf2027/plans/E6.md, Sections 4.1, 6 and 13.1; bioinf2027/reviews/E6_results_r1.md; bioinf2027/GATE.md, top section, item 11"
SECONDARY = "secondary, unadjusted (bioinf2027/plans/E6.md, %s)"
TOLERANCE = 1e-12
SMOKE = False


def row(direction: str, quantity: str, value, adjustment: str, **extra) -> dict:
    return {"direction": direction, "quantity": quantity, **e6s.summary(value), "adjustment": adjustment, **extra}


def absolute_rows() -> list:
    rows = []
    for source in e6s.SCREENS:
        target = fits.other(source)
        direction = f"{source} to {target}"
        for arm in e6s.FAMILY + e6s.MASKED + ("C1", "C2"):
            rows.append(row(direction, f"Spearman of {arm}", e6s.arm_rho(source, fits.PRIMARY, arm), SECONDARY % "Section 6; review, non-blocking 1",
                            arm=arm, head="forest", seeds=len(e6s.SEEDS)))
        for column in ("B1", "B2"):
            rows.append(row(direction, f"Spearman of the target screen's raw {column}", e6s.raw(target, column),
                            SECONDARY % "Section 3.5; review, non-blocking 1", arm=f"raw {column}", head="none (untrained)", seeds=0))
    return rows


def availability_rows() -> list:
    rows = []
    for source in e6s.SCREENS:
        target = fits.other(source)
        direction = f"{source} to {target}"
        feature = e6s.DATA["target_is_feature"][target]
        for subset, mask in (("target is a feature", feature), ("target is not a feature", ~feature)):
            c1 = e6s.arm_rho(source, fits.PRIMARY, "C1", mask=mask)
            for arm in ("x", "m", "[x; m]", "C2") + e6s.MASKED + ("z", "C3"):
                read = arm not in ("z", "C3")
                rows.append(row(direction, f"T of {arm}, {subset}", e6s.contrast(e6s.arm_rho(source, fits.PRIMARY, arm, mask=mask), c1),
                                SECONDARY % "Section 6, item 3", arm=arm, subset=subset, constructs=int(mask.sum()),
                                split=("availability split, read for expression-derived arms only" if read
                                       else "reported; not an availability split for z and C3 (E1_design.md, Section 10)")))
    return rows


def gene_fold_rows(family: pd.DataFrame) -> list:
    rows = []
    for r in family.itertuples():
        if not (r.statistic == "T" or (r.statistic == "R" and r.arm in e6s.FAMILY)):
            continue
        source = r.direction.split(" to ")[0]
        primary = e6s.gene_fold_value(source, r.statistic, r.arm, fits.FOLD_SEED)
        primary_state = e6s.assignment_state(primary, r.holm_threshold)
        entry = {"direction": r.direction, "statistic": r.statistic, "arm": r.arm, "family_state": r.state, "holm_threshold": r.holm_threshold,
                 "primary_assignment_head_seed_0": e6s.summary(primary)["estimate"], "primary_assignment_head_seed_0_lo": e6s.summary(primary)["lo"],
                 "primary_assignment_head_seed_0_hi": e6s.summary(primary)["hi"], "primary_assignment_head_seed_0_p": e6s.summary(primary)["p"],
                 "primary_assignment_head_seed_0_state": primary_state}
        states = []
        for assignment in fits.FOLD_SENSITIVITY:
            value = e6s.gene_fold_value(source, r.statistic, r.arm, assignment)
            s = e6s.summary(value)
            state = e6s.assignment_state(value, r.holm_threshold)
            states.append(state)
            entry.update({f"assignment_{assignment}_estimate": s["estimate"], f"assignment_{assignment}_lo": s["lo"],
                          f"assignment_{assignment}_hi": s["hi"], f"assignment_{assignment}_p": s["p"], f"assignment_{assignment}_state": state})
        entry["marker_under_the_plan_rule"] = (r.state != "inconclusive" and primary_state != "not fitted"
                                               and any(s not in (primary_state, "not fitted") for s in states))
        entry["marker_in_the_family_pass"] = bool(r.not_robust_to_the_gene_fold_assignment)
        entry["adjustment"] = SECONDARY % "Section 13.1, item 2; departure 7 of bioinf2027/reports/E6.md"
        rows.append(entry)
    return rows


def contrast_rows() -> list:
    rows = []
    for source in e6s.SCREENS:
        direction = f"{source} to {fits.other(source)}"
        z = e6s.arm_rho(source, fits.PRIMARY, "z")
        for other in ("x", "[x; m]", "C2"):
            rows.append(row(direction, f"z minus {other}", e6s.contrast(z, e6s.arm_rho(source, fits.PRIMARY, other)), SECONDARY % "Section 6, item 7"))
    return rows


# --------------------------------------------------------------------------- checks


def manifests_check() -> dict:
    tier2 = json.loads((E6 / "predictions_manifest_tier2.json").read_text())["files"]
    tier3 = json.loads((E6 / "predictions_manifest_tier3.json").read_text())["files"]
    missing = [k for k in tier2 if k not in tier3]
    differing = [k for k in tier2 if k in tier3 and tier3[k]["sha256"] != tier2[k]["sha256"]]
    out = {"tier2_fits": len(tier2), "tier3_fits": len(tier3), "tier2_missing_from_tier3": len(missing), "tier2_with_another_digest": len(differing),
           "passed": not missing and not differing}
    if not out["passed"]:
        raise SystemExit("the tier-3 manifest does not hold the family pass's predictions unchanged: %s" % json.dumps(out))
    return out


def family_check(family: pd.DataFrame) -> dict:
    largest, compared = 0.0, 0
    for r in family.itertuples():
        if r.arm not in e6s.FAMILY or r.statistic not in ("T", "R"):
            continue
        source = r.direction.split(" to ")[0]
        if r.statistic == "T":
            value = e6s.contrast(e6s.arm_rho(source, fits.PRIMARY, r.arm), e6s.arm_rho(source, fits.PRIMARY, "C1"))
        else:
            value = e6s.arm_r(source, fits.PRIMARY, r.arm)
        s = e6s.summary(value)
        names = ("estimate",) if SMOKE else ("estimate", "lo", "hi", "p")
        for name in names:
            difference = abs(float(s[name]) - float(getattr(r, name)))
            largest = max(largest, difference)
            compared += 1
            if not difference <= TOLERANCE:
                raise SystemExit("%s %s %s: %s is %r here and %r in family_pass/family.csv; nothing is written"
                                 % (r.direction, r.statistic, r.arm, name, s[name], getattr(r, name)))
    return {"values_compared": compared, "largest_difference": largest, "tolerance": TOLERANCE, "passed": True}


def digest_check() -> dict:
    theirs = json.loads((REPO / "bioinf2027" / "results" / "E1" / "stats.json").read_text())["replicate_index_sha256"]
    out = {s: {"E6": e6s.DATA["draws"][s]["draw"].index_sha256, "E1b": theirs.get(s)} for s in e6s.SCREENS}
    for s, v in out.items():
        v["equal"] = v["E6"] == v["E1b"]
    if not SMOKE and not all(v["equal"] for v in out.values()):
        raise SystemExit("the replicate index differs from E1b's: %s" % json.dumps(out))
    return out


def main() -> None:
    started = time.time()
    checked_inputs = e6s.verify_inputs()
    manifests = manifests_check()
    e6s.PASS = "full"  # the tier-3 manifest, which holds the family pass's predictions unchanged (check 2)
    e6s.load_predictions()
    inputs = e6s.load()
    family = pd.read_csv(FAMILY_PASS / "family.csv")
    checks = {"data_root_inputs": "equal to their records" if checked_inputs else None, "manifests": manifests,
              "family_values": family_check(family), "replicate_index_digests": digest_check()}
    tables = {"absolute.csv": pd.DataFrame(absolute_rows()), "folds.csv": e6s.fold_rows(), "availability.csv": pd.DataFrame(availability_rows()),
              "gene_fold.csv": pd.DataFrame(gene_fold_rows(family)), "one_sided.csv": e6s.one_sided_rows(family),
              "contrasts.csv": pd.DataFrame(contrast_rows())}
    tables["folds.csv"]["adjustment"] = "secondary, unadjusted; point values, diagnostics (bioinf2027/plans/E6.md, Section 13.1, item 8)"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, frame in tables.items():
        frame.to_csv(OUT_DIR / name, index=False)
        outputs[name] = file_record(OUT_DIR / name)
    record = {"plan": PLAN, "replicates": e6s.B, "seeds": e6s.SEED, "nuisance_fold_seed_R": e6s.FOLD_SEED_R,
              "gene_fold_assignments": {"primary": fits.FOLD_SEED, "others": list(fits.FOLD_SENSITIVITY)},
              "margins": {"R": e6s.MARGIN_R, "increment": e6s.MARGIN_INCREMENT},
              "inputs": {**inputs, "predictions_manifest": e6s.DATA["manifest"], "family_pass_family": file_record(FAMILY_PASS / "family.csv"),
                         "data_root_inputs": checked_inputs},
              "checks": checks, "outputs": outputs, "smoke": SMOKE, "seconds": round(time.time() - started, 1)}
    (OUT_DIR / "supplement.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e6 supplement] done in %.0f s; %s" % (time.time() - started, ", ".join("%s %d rows" % (k, len(v)) for k, v in tables.items())), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["--smoke"]:
        SMOKE = True
        e6s.B = 20
        resample.BLOCK = 20
        OUT_DIR = config.storage_root() / "bioinf2027" / "scratch" / "E6_supplement_smoke"
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    main()
