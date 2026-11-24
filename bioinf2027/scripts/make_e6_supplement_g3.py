"""E6, the supplement of the full pass: departures (a) to (f) of results review r2, from saved predictions.

Design: bioinf2027/plans/E6.md, Section 17 (gate G3, item 1; bioinf2027/GATE.md, top section). No new fit. Imports
bioinf2027/scripts/make_e6_stats.py unchanged and uses its predictions (predictions_manifest_tier3.json), joint
target-gene draws (10,000 replicates, seeds 20260901 and 20260902), R (nuisance fold seed 20270409) and summaries.
Every row is secondary and unadjusted and names its statistic; every R or R- row carries its target screen's R5
gate and is marked not read where the gate fails.

- (a) a_oracle_linear_partial.csv: the oracle's linear partial given the two control means only.
- (b) b_mean_label.csv: R and R- under the mean-across-draws label; R(z-) not fitted under that label.
- (c) c_gene_identity.csv: whole-screen minus gene-disjoint T, R and R-.
- (d) d_namespace.csv: E1b's x and the shared-gene x, Spearman, T, R, R- and arm minus masked arm by subset.
- (e) e_secondary_heads_whole_screen.csv: ridge, boosting and the monotone label positive control on the whole
  screen, all four endpoints.
- (f) f_cell_disjoint.csv: T of the cell-disjoint features and the paired cell-disjoint minus primary difference.
- secondary_statistics.csv: the statistic of each row of secondary.csv (that file is not changed).

Checks before any value: prediction digests (make_e6_stats.prediction), the draws' index digests against
bioinf2027/results/E6/stats.json, and T and R of z recomputed here against family.csv, exactly.
Outputs: bioinf2027/results/E6/supplement_g3/ with supplement.json. With --smoke: 20 replicates, the smoke
predictions, outputs to the scratch folder, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, resample, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e6_fits as fits  # noqa: E402
import make_e6_stats as st  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E6.md, Section 17 (gate G3, item 1)"
OUT_DIR = st.OUT / "supplement_g3"
SMOKE = False
PRIMARY = fits.PRIMARY
FAMILY = st.FAMILY
MASKED = st.MASKED
PAIRS = (("x-", "x"), ("m-", "m"), ("[x; m]-", "[x; m]"))
NOTE_WHOLE = "the whole-screen fit trains on every source construct, the gene-disjoint fit on about four fifths of them"


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def gate(target: str) -> bool:
    return bool(st.DATA["r5"][f"G1 | {target}"]["passed"])


def row(item: str, source: str, statistic: str, quantity: str, value, r_type: bool = False, **extra) -> dict:
    target = fits.other(source)
    out = {"item": item, "direction": f"{source} to {target}", "statistic": statistic, "quantity": quantity, **st.summary(value),
           "fitted": value is not None, "adjustment": "unadjusted", "role": "secondary"}
    if r_type:
        out["r5_gate_passed"] = gate(target)
        out["read"] = gate(target)
    out.update(extra)
    return out


def seeds(head: str = "forest", endpoint: str = PRIMARY) -> tuple:
    return st.seeds_of(head, endpoint)


# --------------------------------------------------------------------------- (a)


def ols_rank_residual(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    y = sps.rankdata(values) / (values.size + 1.0)
    design = np.column_stack([np.ones(values.size), covariates])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def item_a() -> pd.DataFrame:
    rows = []
    for source in st.SCREENS:
        target = fits.other(source)
        s_table, t_table = st.DATA["tables"][PRIMARY][source], st.DATA["tables"][PRIMARY][target]
        by_gene = s_table.groupby("target_ensg")
        s_label, s_b1 = by_gene["y"].median(), by_gene["B1"].median()
        genes = t_table["target_ensg"].astype(str)
        held = genes.isin(s_label.index).to_numpy()
        label = t_table["y"].to_numpy(dtype=np.float64)
        source_label = genes.map(s_label).fillna(0).to_numpy(dtype=np.float64)
        covariates = np.column_stack([t_table["B1"].to_numpy(dtype=np.float64), genes.map(s_b1).fillna(0).to_numpy(dtype=np.float64)])
        e_t, e_s = np.zeros(label.size), np.zeros(label.size)
        e_t[held] = ols_rank_residual(label[held], covariates[held])
        e_s[held] = ols_rank_residual(source_label[held], covariates[held])
        draws = st.DATA["draws"][target]
        point = float(resample.weighted_pearson(e_s, e_t, draws["point"], mask=held)[0, 0])
        replicates = resample.weighted_pearson(e_s, e_t, draws["draw"], mask=held)[:, 0]
        rows.append(row("a", source, "linear partial correlation",
                        "the two labels (ranks) given the two control means (target B1, source gene-median B1), ordinary least squares",
                        (point, replicates), r_type=True, constructs=int(held.sum()),
                        note="the source label shares the guide design with the target construct; not a ceiling for a representation"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- (b)


def item_b() -> pd.DataFrame:
    rows = []
    for source in st.SCREENS:
        target = fits.other(source)
        label = st.label_of(target, PRIMARY, "y_mean")
        covariates, genes = st.covariates_of(target, PRIMARY), st.DATA["genes"][target]
        for arm in FAMILY + MASKED:
            vectors = [st.assembled(source, PRIMARY, arm, "forest", s, label="mean") for s in seeds()]
            value = st.r_stat(target, vectors, label, covariates, genes)
            rows.append(row("b", source, "R-" if arm in MASKED else "R", f"{'R-' if arm in MASKED else 'R'} of {arm}, mean-across-draws label",
                            value, r_type=True, arm=arm))
        rows.append(row("b", source, "R(z-)", "R(z-), mean-across-draws label", None, r_type=True, arm="z-",
                        note="not fitted: D2's z- arm was fitted on the rebuilt label only; computing it would need a new fit"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- (c)


def whole(source: str, model: str, head: str = "forest", endpoint: str = PRIMARY) -> list:
    return [st.whole_screen(source, endpoint, model, head, s) for s in seeds(head, endpoint)]


def item_c() -> pd.DataFrame:
    rows = []
    for source in st.SCREENS:
        target = fits.other(source)
        label, covariates, genes = st.label_of(target, PRIMARY), st.covariates_of(target, PRIMARY), st.DATA["genes"][target]
        c1_gd, c1_whole = st.arm_rho(source, PRIMARY, "C1"), st.rho(target, whole(source, "C1"), label)
        for arm in FAMILY:
            t_gd = st.contrast(st.arm_rho(source, PRIMARY, arm), c1_gd)
            t_whole = st.contrast(st.rho(target, whole(source, arm), label), c1_whole)
            rows.append(row("c", source, "T", f"T of {arm}, whole screen", t_whole, arm=arm, note=NOTE_WHOLE))
            rows.append(row("c", source, "difference of T", f"T of {arm}, whole screen minus gene-disjoint", st.contrast(t_whole, t_gd), arm=arm, note=NOTE_WHOLE))
        for arm in FAMILY + MASKED:
            statistic = "R-" if arm in MASKED else "R"
            r_gd = st.arm_r(source, PRIMARY, arm)
            r_whole = st.r_stat(target, whole(source, arm), label, covariates, genes)
            rows.append(row("c", source, statistic, f"{statistic} of {arm}, whole screen", r_whole, r_type=True, arm=arm, note=NOTE_WHOLE))
            rows.append(row("c", source, f"difference of {statistic}", f"{statistic} of {arm}, whole screen minus gene-disjoint",
                            st.contrast(r_whole, r_gd), r_type=True, arm=arm, note=NOTE_WHOLE))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- (d)


def item_d() -> pd.DataFrame:
    rows = []
    for source in st.SCREENS:
        target = fits.other(source)
        c1 = st.arm_rho(source, PRIMARY, "C1")
        for basis in ("e1b", "shared genes"):
            feature = np.asarray(fits.DATA[("x", target)][basis][1]) >= 0
            for arm in ("x", "m", "[x; m]") + tuple(m for m, _ in PAIRS):
                a = st.arm_rho(source, PRIMARY, arm, basis=basis)
                statistic = "R-" if arm.endswith("-") else "R"
                rows.append(row("d", source, "Spearman", f"Spearman of {arm}, basis {basis}", a, arm=arm, basis=basis))
                rows.append(row("d", source, "T", f"T of {arm}, basis {basis} (C1 does not depend on x)", st.contrast(a, c1), arm=arm, basis=basis))
                rows.append(row("d", source, statistic, f"{statistic} of {arm}, basis {basis}", st.arm_r(source, PRIMARY, arm, basis=basis),
                                r_type=True, arm=arm, basis=basis))
            for masked, arm in PAIRS:
                for subset, mask in (("all", None), ("target is a feature of the basis", feature), ("target is not a feature of the basis", ~feature)):
                    value = st.contrast(st.arm_rho(source, PRIMARY, arm, mask=mask, basis=basis), st.arm_rho(source, PRIMARY, masked, mask=mask, basis=basis))
                    rows.append(row("d", source, "difference of Spearman", f"{arm} minus {masked}, basis {basis}, {subset}", value, arm=arm, basis=basis,
                                    subset=subset, constructs=int(feature.size if mask is None else mask.sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- (e)


def whole_screen_models() -> dict:
    """(endpoint, head) -> models with whole-screen fits in the manifest."""
    out: dict = {}
    for spec in st.DATA["files"]:
        context, _, model, head, _, _, _ = spec.rsplit("|", 6)
        if "| whole screen |" in context and head in ("ridge-cv", "boosting", "monotone"):
            out.setdefault((context.split("|")[2].strip(), head), set()).add(model)
    return {k: sorted(v) for k, v in out.items()}


def item_e() -> pd.DataFrame:
    rows = []
    listed = whole_screen_models()
    for (endpoint, head), models in sorted(listed.items()):
        for source in st.SCREENS:
            target = fits.other(source)
            label = st.label_of(target, endpoint)
            c1_whole = st.rho(target, whole(source, "C1", head, endpoint), label) if "C1" in models else None
            for model in models:
                w = st.rho(target, whole(source, model, head, endpoint), label)
                gd = st.arm_rho(source, endpoint, model, head)
                common = {"endpoint": endpoint, "head": head, "arm": model, "note": NOTE_WHOLE}
                rows.append(row("e", source, "Spearman", f"Spearman of {model}, {head}, {endpoint}, whole screen", w, **common))
                rows.append(row("e", source, "difference of Spearman", f"Spearman of {model}, {head}, {endpoint}, whole screen minus gene-disjoint",
                                st.contrast(w, gd), **common))
                if model not in ("C1", "label positive control") and c1_whole is not None:
                    rows.append(row("e", source, "T", f"T of {model}, {head}, {endpoint}, whole screen", st.contrast(w, c1_whole), **common))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- (f)


def item_f() -> pd.DataFrame:
    rows = []
    for source in st.SCREENS:
        target = fits.other(source)
        if target not in st.DATA.get("disjoint_label", {}):
            continue
        half_label, _, kept = st.DATA["disjoint_label"][target]

        def on_kept(model: str, cell_disjoint: bool):
            suffix = " cell-disjoint" if cell_disjoint else ""
            return st.rho(target, [st.assembled(source, PRIMARY, model, "forest", s, side_suffix=suffix) for s in seeds()], half_label, kept)

        c1 = on_kept("C1", True)
        for arm in FAMILY:
            disjoint, primary = on_kept(arm, True), on_kept(arm, False)
            common = {"arm": arm, "constructs": int(kept.sum()), "label": "half A of the G1 external halves"}
            rows.append(row("f", source, "T", f"T of {arm}, cell-disjoint features (half B), C1 on the same constructs", st.contrast(disjoint, c1), **common))
            rows.append(row("f", source, "Spearman", f"Spearman of {arm}, primary features, on the cell-disjoint constructs", primary, **common))
            rows.append(row("f", source, "difference of Spearman", f"{arm}: cell-disjoint minus primary features, same constructs and label",
                            st.contrast(disjoint, primary), **common))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the statistic of each row of secondary.csv

NAME_RULES = (
    (r" minus the target screen's raw B[12]$", "difference of Spearman"),
    (r" minus the B1 model trained on the source$", "difference of Spearman"),
    (r" minus TR$", "difference of Spearman"),
    (r", basis (e1b|shared genes)$", "Spearman"),
    (r"^T of .*, mean-across-draws label$", "T"),
    (r"^T of .*, E4 matched features draw \d$", "T"),
    (r", cell-disjoint \(half B features, half A label\)$", "Spearman"),
    (r"^R of .*, cell-disjoint$", "R"),
    (r" minus (x|m|\[x; m\])-, (all|target is a feature|target is not a feature)$", "difference of Spearman"),
    (r"^z minus (x|C2|\[x; m\])$", "difference of Spearman"),
    (r"^R of .*-, nuisance fold seed \d+$", "R-"),
    (r"^R of [^,]*[^-], nuisance fold seed \d+$", "R"),
    (r"^C2 of the shared-gene basis$", "Spearman"),
)


def secondary_statistics() -> pd.DataFrame:
    frame = pd.read_csv(st.OUT / "secondary.csv")
    names = []
    for contrast in frame["contrast"]:
        found = [statistic for pattern, statistic in NAME_RULES if re.search(pattern, contrast)]
        if not found:
            raise Stop("secondary.csv: no statistic name for %r" % contrast)
        names.append(found[0])
    return pd.DataFrame({"direction": frame["direction"], "contrast": frame["contrast"], "statistic": names})


# --------------------------------------------------------------------------- checks and main


def checks() -> dict:
    out = {}
    if SMOKE:
        return {"note": "smoke: no comparison with the full pass"}
    recorded = json.loads((st.OUT / "stats.json").read_text())["replicate_index_digests"]
    out["replicate_index_digests"] = {s: {"here": st.DATA["draws"][s]["draw"].index_sha256, "stats.json": recorded[s]["E6"],
                                          "equal": st.DATA["draws"][s]["draw"].index_sha256 == recorded[s]["E6"]} for s in st.SCREENS}
    family = pd.read_csv(st.OUT / "family.csv", float_precision="round_trip")
    compared = []
    for source in st.SCREENS:
        direction = f"{source} to {fits.other(source)}"
        for statistic, value in (("T", st.contrast(st.arm_rho(source, PRIMARY, "z"), st.arm_rho(source, PRIMARY, "C1"))), ("R", st.arm_r(source, PRIMARY, "z"))):
            saved = family[(family.direction == direction) & (family.statistic == statistic) & (family.arm == "z")].iloc[0]
            here = st.summary(value)
            equal = {k: bool(here[k] == saved[k]) for k in ("estimate", "lo", "hi", "p")}
            compared.append({"direction": direction, "statistic": statistic, "arm": "z", "equal": equal})
    out["family_z_recomputed"] = compared
    failed = [s for s, v in out["replicate_index_digests"].items() if not v["equal"]] + [c for c in compared if not all(c["equal"].values())]
    out["passed"] = not failed
    if failed:
        raise Stop("check failed: %s" % json.dumps(strict_json(out)))
    return out


def main() -> None:
    started = time.time()
    checked_inputs = None if SMOKE else st.verify_inputs()
    st.load_predictions()
    inputs = st.load()
    record_checks = checks()
    print("[e6 supplement] checks passed", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, make in (("a_oracle_linear_partial.csv", item_a), ("b_mean_label.csv", item_b), ("c_gene_identity.csv", item_c),
                       ("d_namespace.csv", item_d), ("e_secondary_heads_whole_screen.csv", item_e), ("f_cell_disjoint.csv", item_f),
                       ("secondary_statistics.csv", secondary_statistics)):
        frame = make()
        frame.to_csv(OUT_DIR / name, index=False)
        outputs[name] = {**file_record(OUT_DIR / name), "rows": int(len(frame))}
        print("[e6 supplement] %s: %d rows, %.0f s" % (name, len(frame), time.time() - started), flush=True)
    record = {"plan": PLAN, "smoke": SMOKE, "replicates": st.B, "seeds": st.SEED, "nuisance_fold_seed_R": st.FOLD_SEED_R,
              "checks": record_checks,
              "inputs": {**inputs, "predictions_manifest": st.DATA["manifest"], "data_root_inputs": checked_inputs,
                         "readouts": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json"),
                         **({} if SMOKE else {"stats.json": file_record(st.OUT / "stats.json"), "family.csv": file_record(st.OUT / "family.csv"),
                                               "secondary.csv": file_record(st.OUT / "secondary.csv")})},
              "code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (sys.modules[__name__], st, fits, transfer, resample)},
              "outputs": outputs, "seconds": round(time.time() - started, 1)}
    (OUT_DIR / "supplement.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e6 supplement] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["--smoke"]:
        SMOKE = True
        st.SMOKE = True
        fits.SMOKE = True
        st.SEEDS = (0,)
        st.B = 20
        resample.BLOCK = 20
        OUT_DIR = config.storage_root() / "bioinf2027" / "scratch" / "E6_supplement_g3_smoke"
    elif arguments:
        raise SystemExit("usage: make_e6_supplement.py [--smoke]")
    main()
