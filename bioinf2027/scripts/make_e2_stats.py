"""E2, the statistics, from the saved predictions of make_e2_fits.py (tests i and ii) and make_e2_fits_e6.py (test iii).

Design: bioinf2027/plans/E2.md, Sections 2, 4 to 6, 9 and 12 to 16, with the readings it cites:
bioinf2027/plans/E1b_attribution.md, Sections 2 and 11; bioinf2027/plans/E1_design.md, Sections 10 and 11;
bioinf2027/plans/E6.md, Sections 4 and 7. Nothing is refitted.

- Encoders. scGPT (512), Stack (1,600) and PCA-50 are read. PCA-768 is a width-matched reference with no reading
  (its C1 and [B0; C1] are E1b's C1 and the attribution note's fits). Geneformer's z, from E1b's fits, is the
  reference for S2, with its size-matched null N0_z; E1b's x and C2 are printed beside PCA.
- (i) VCC test, cells G5 and G6 (G6 the cell of record, G5 beside it with no reading); forest (primary), boosting
  and ridge. The target-gene cluster bootstrap of E1b's precondition (10,000 replicates, seed 20270301) through
  make_e1b_stats.vcc_r2, unchanged: per-seed R-squared and Spearman averaged over the head seeds inside each
  replicate, every contrast inside each replicate. Beside P-a and P-d the batch-resampled interval (seed
  20270302), a sensitivity analysis. P-a, P-d (with its MSE version and the B0, B1, B2 and TR reference drops of
  the post hoc note), P-r and P-c; the attribution rows beside cell count; the size-matched null; the positive
  control at each width; e - z; PCA-50 beside its width-50 random projection.
- (ii) External: E1b's joint target-gene draws (seeds 20260901 and 20260902). T and L per encoder, cell and
  screen, read only for an encoder that passes E1b's two-cell precondition. No R externally (gate G2).
- (iii) E6's T at 27/788: the gene-disjoint predictions of the five folds, E6's draws (the same replicates as
  E1b's, built from the same constructs in the same order; their index digests are compared).
- E2's family: T and L of scGPT, Stack and PCA-50 on both screens in G5 and G6 (24) and E6's T in both
  directions (6), 30 statistics, Holm at family-wise 0.05; every one computed, adjusted and reported. Everything
  else is secondary and unadjusted, and every table says so.

Check 7 of the plan (Section 9), before any E2 value is computed: Geneformer's values, computed by this
script's code, reproduce to 1e-12 E1b's precondition values for z, the post hoc note's rows, the attribution
note's rows and its amendment's part (g), E1b's T and L of z and E6's T of z, and the replicate index digests
equal those on file. Every prediction file is checked against its manifest's digest; every planned fit must be
present, and no fit may have failed. A failed check raises Stop and nothing is written.

Outputs under bioinf2027/results/E2/stats/: family.csv, pattern.csv, vcc.csv, label_scale.csv,
attribution_s2.csv, external.csv, readings_external.csv, e6.csv, readings_e6.csv, power.csv and stats.json.

With --smoke: 20 replicates, the smoke predictions (head seed 0), outputs to
$IGA_STORAGE_ROOT/bioinf2027/scratch/E2_stats_smoke/, never reported; a planned fit missing from the smoke runs is
listed, not fatal, and the reproduction compares point values only. --workers N sets the processes of the
external scoring (default 1; it changes no value).
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, e1inputs, resample, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e1b_attribution as att  # noqa: E402
import make_e1b_attribution_amendment as am  # noqa: E402
import make_e1b_posthoc as pho  # noqa: E402
import make_e1b_stats as e1s  # noqa: E402
import make_e2_fits as e2f  # noqa: E402
import make_e2_fits_e6 as e2e6  # noqa: E402
import make_e6_fits as e6f  # noqa: E402
import make_e6_stats as e6s  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "bioinf2027" / "results"
E1 = RESULTS / "E1"
E6 = RESULTS / "E6"
E2 = RESULTS / "E2"
FITS = E2 / "fits"
OUT = E2 / "stats"
ORDER_RECORD = E2 / "order_sensitivity.json"
PLAN = "bioinf2027/plans/E2.md, Sections 2, 4 to 6, 9 and 12 to 16"
STATUS = "secondary, unadjusted (bioinf2027/plans/E2.md, Section 6)"
FAMILY_NOTE = ("E2, 30 statistics, Holm at family-wise 0.05; across E1 (0.10), E4, D2, E6 and E2 (0.05 each) the family-wise "
               "error is at most 0.30")
READ_ENCODERS = ("scgpt", "stack", "pca50")
REFERENCE_ENCODER = "pca768"
GENEFORMER = "geneformer"
EXPRESSION = ("x", "C2")
WIDTHS = dict(e2f.ENCODERS)
CELLS = tuple(e1cells.PRIMARY)
RECORD_CELL = "G6"
LABELS = ("existing", "rebuilt")
PAIRED = "existing minus rebuilt"
L_LABEL = "rebuilt minus existing"
HEADS = dict(e2f.HEAD_SEEDS)
SCREENS = tuple(e1s.SCREENS)
SOURCES = tuple(e6f.SCREENS)
ALPHA = 0.05
TOLERANCE = 1e-12
POSITIVE_LIMIT = 0.80
ORDER_THRESHOLD = 0.10
WORKERS = 1
SMOKE = False
NOT_RECORD = pho.NOT_RECORD
PLANNED_DETECTABLE = {("G6", "P-a"): 0.235, ("G6", "P-d"): 0.19, ("G6", "P-r"): 0.13, ("G6", "P-c"): 0.23, ("G5", "P-r"): 0.64}
STACK_REPLOGLE = ("Stack: the Replogle screens are possibly in Stack's pre-training data (bioinf2027/results/E2/pretraining_check.json); "
                  "no comparison between encoders is made on them")
STACK_VCC = ("Stack: set-context embeddings; the VCC data are not shown to be in Stack's pre-training data (inconclusive, "
             "bioinf2027/results/E2/pretraining_check.json)")
NO_R = "R is not computed externally (the author's decision at gate G2), so no statement beyond the trivial predictors is made"
RESTRICTED_INPUTS = ("the restricted-input sensitivity (E2.md, Section 2, (ii)) is not computed: it needs embeddings of the VCC cells "
                     "restricted to each screen's measured genes and their fits, which no E2 script makes")
E6_MARKERS_ABSENT = ("E6's markers for the gene-fold assignment and head fitting are not computed: E2's fits hold the primary folds "
                     "and head seeds 0 to 2 only")

DATA: dict = {}
CACHE: dict = {}


class Stop(RuntimeError):
    """A check failed or an input is missing: nothing is written (hard limit 9)."""


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E2_stats_smoke" if SMOKE else OUT


def fits_dir() -> Path:
    """Where make_e2_fits.py writes its manifests; make_e2_fits_e6.py writes to bioinf2027/results/E2/fits/ in both modes."""
    return config.storage_root() / "bioinf2027" / "scratch" / "E2_fits_smoke" if SMOKE else FITS


def resolve(portable: str) -> Path:
    prefix = "$" + config.STORAGE_VARIABLE + "/"
    if portable.startswith(prefix):
        return config.storage_root() / portable[len(prefix):]
    return REPO / portable


def missing(what: str) -> None:
    """A planned fit or value is absent: fatal in the real run, listed in the smoke run."""
    if not SMOKE:
        raise Stop("a planned fit is missing: %s" % what)
    DATA.setdefault("missing", []).append(what)


# --------------------------------------------------------------------------- representations and the lookup


GENEFORMER_NAMES = {"e": "z", "C1": "C1", "C3": "C3", "[B0; e]": "[B0; z]", "[B0; C1]": "[B0; C1]", "[B0; C3]": "[B0; C3]",
                    "N0": "N0[geneformer]", "[e; B2]": "[z; B2]", "[C1; B2]": "[C1; B2]"}
FIXED = ("B0", "B1", "B2", "TR", "z")
POSITIVE_ROLES = ("[e; B2]", "[C1; B2]")


def roles(encoder: str) -> dict:
    """The encoder's representation names by role, as the fitting scripts name them; only the roles fitted."""
    if encoder == GENEFORMER:
        return dict(GENEFORMER_NAMES)
    if encoder in EXPRESSION:
        return {"e": encoder, "C1": "C1"}
    out = {r: e2f.representation_name(encoder, r) for r in e2f.representations(encoder)}
    if encoder == REFERENCE_ENCODER:
        out.update({"C1": "C1", "[B0; C1]": "[B0; C1]"})  # E1b's C1 and the attribution note's [B0; C1] (E2.md, Section 4)
    else:
        out.update({r: e2f.representation_name(encoder, r) for r in POSITIVE_ROLES})
    return out


def role_name(encoder: str, role: str) -> str | None:
    return role if role in FIXED else roles(encoder).get(role)


def available(encoder: str, role: str, cell: str) -> bool:
    if role in FIXED:
        return True
    if role in POSITIVE_ROLES and cell != RECORD_CELL:
        return False  # the positive control is fitted in G6 only (E2.md, Section 5)
    return role in roles(encoder)


def e2_names() -> dict:
    """Every representation name of E2's fits, with its encoder (context "E2 | <encoder> | <cell>", make_e2_fits.py)."""
    out = {}
    for encoder in WIDTHS:
        for role, name in roles(encoder).items():
            if not (encoder == REFERENCE_ENCODER and role in ("C1", "[B0; C1]")):
                out[name] = encoder
    out[GENEFORMER_NAMES["N0"]] = GENEFORMER
    clash = set(out) & (set(transfer.REPRESENTATIONS) | set(att.NEW) | set(am.NEW))
    if clash:
        raise Stop("E2's representation names clash with E1b's: %s" % sorted(clash))
    return out


def width(encoder: str) -> int:
    return {GENEFORMER: 768, "x": 2000, "C2": 768}.get(encoder, WIDTHS.get(encoder, 0))


class Lookup:
    """One lookup over E1b's, the attribution note's and E2's manifests; make_e1b_stats.vcc_r2 reads DATA["predictions"].

    A representation that E2 names is found in E2's manifests under the context "E2 | <encoder> | <cell>"; every
    other one in E1b's under the cell. Each file is checked against its manifest's digest on first use."""

    def __init__(self, e1b: list, e2: list, encoder_of: dict) -> None:
        sources = list(e1b) + list(e2)
        for i, a in enumerate(sources):
            for b in sources[i + 1:]:
                shared = set(a.files) & set(b.files)
                if shared:
                    raise Stop("two manifests share %d keys, e.g. %s" % (len(shared), sorted(shared)[0]))
        self.e1b, self.e2, self.encoder_of = list(e1b), list(e2), dict(encoder_of)

    def is_e2(self, representation: str) -> bool:
        return representation in self.encoder_of

    def get(self, context, label, representation, head="forest", seed=0, variant=e1s.ALL_ROWS, realisation=0):
        encoder = self.encoder_of.get(representation)
        sources = self.e1b if encoder is None else self.e2
        if encoder is not None:
            context = f"E2 | {encoder} | {context}"
        for source in sources:
            try:
                values = source.get(context, label, representation, head, seed, variant, realisation)
            except RuntimeError as error:  # make_e1b_stats.Predictions: a file that does not match its manifest
                raise Stop(str(error)) from error
            if values is not None:
                return values
        return None

    def seeds(self, context, label, representation, head="forest", variant=e1s.ALL_ROWS, realisation=0, seeds=e1s.SEEDS):
        out = []
        for seed in seeds:
            values = self.get(context, label, representation, head, seed, variant, realisation)
            if values is not None:
                out.append(values)
        return out


def wanted_seeds(representation: str, head: str) -> tuple:
    seeds = tuple(HEADS[head])
    if SMOKE and DATA["lookup"].is_e2(representation):
        return seeds[:1]  # the smoke fits hold head seed 0 only
    return seeds


def heads_for(encoder: str) -> tuple:
    return ("forest",) if encoder in EXPRESSION else tuple(HEADS)


# --------------------------------------------------------------------------- manifests and planned fits


def manifest_paths(first_folder: Path, third_folder: Path, smoke: bool) -> dict:
    """The manifests of tests (i) and (ii) and of test (iii) for this mode."""
    first = sorted(p for p in first_folder.glob("predictions_manifest_*.json")
                   if not p.name.startswith("predictions_manifest_e6_") and not p.name.endswith("_smoke.json"))
    third = sorted(p for p in third_folder.glob("predictions_manifest_e6_*.json") if p.name.endswith("_smoke.json") == smoke)
    return {"i_ii": first, "iii": third}


def check_manifest(path: Path, kind: str) -> dict:
    """A manifest is used only if every fit completed, none failed, its protocol matches and the protocol's check passed."""
    manifest = json.loads(path.read_text())
    problems = []
    if manifest.get("failed"):
        problems.append("%d fits failed" % len(manifest["failed"]))
    if not (manifest.get("completed") == manifest.get("fits") == len(manifest.get("files", {}))):
        problems.append("completed %s of %s fits, %d files" % (manifest.get("completed"), manifest.get("fits"), len(manifest.get("files", {}))))
    protocol_path = resolve(manifest["protocol"]["path"])
    if not protocol_path.is_file() or sha256(protocol_path) != manifest["protocol"]["sha256"]:
        problems.append("the protocol does not match the manifest's record (a run in progress, or rewritten)")
        protocol = {}
    else:
        protocol = json.loads(protocol_path.read_text())
        if bool(protocol.get("smoke")) != SMOKE:
            problems.append("the protocol's smoke flag is %s" % protocol.get("smoke"))
        check = "E0" if kind == "i_ii" else "E6-0"
        results = protocol.get("checks", {}).get(check)
        if not results or not all(r.get("passed") for r in results):
            problems.append("check %s did not pass" % check)
    if problems:
        raise Stop("%s: %s" % (path.name, "; ".join(problems)))
    return {"manifest": file_record(path), "protocol": file_record(protocol_path), "fits": int(manifest["fits"]),
            "script": protocol.get("script"), "fitting_code": protocol.get("fitting_code")}


def planned_fits(smoke: bool) -> tuple[dict, dict]:
    """Every fit the fitting scripts plan, keyed as their manifests key them: (i) and (ii) with N0_z, and (iii)."""
    e2f.SMOKE, e2e6.SMOKE = smoke, smoke
    try:
        first = {t[3].key(): t for t in e2f.tasks(list(WIDTHS)) + e2f.geneformer_null_tasks()}
        third = {t[4].key(): t for t in e2e6.tasks(list(e2e6.WIDTHS))}
    finally:
        e2f.SMOKE, e2e6.SMOKE = False, False
    return first, third


def load_predictions() -> dict:
    """The lookups of (i) and (ii) and of (iii), each manifest checked; every planned fit present (real run)."""
    paths = manifest_paths(fits_dir(), FITS, SMOKE)
    first_planned, third_planned = planned_fits(SMOKE)
    records = {"i_ii": [check_manifest(p, "i_ii") for p in paths["i_ii"]], "iii": [check_manifest(p, "iii") for p in paths["iii"]]}
    e2_first = [e1s.Predictions(p) for p in paths["i_ii"]]
    e2_third = [e1s.Predictions(p) for p in paths["iii"]]
    e1b = [e1s.Predictions(E1 / "predictions_manifest.json"), e1s.Predictions(E1 / "attribution" / "predictions_manifest.json")]
    DATA["lookup"] = Lookup(e1b, e2_first, e2_names())
    e1s.DATA["predictions"] = DATA["lookup"]
    for i, a in enumerate(e2_third):
        for b in e2_third[i + 1:]:
            if set(a.files) & set(b.files):
                raise Stop("two manifests of test (iii) share keys")
    DATA["e6_sources"] = e2_third
    DATA["e6_specs"] = {(t[0], t[1], t[2], t[3], t[4].seed): t[4] for t in third_planned.values()}
    present_first = set().union(*[set(p.files) for p in e2_first]) if e2_first else set()
    present_third = set().union(*[set(p.files) for p in e2_third]) if e2_third else set()
    absent_first, absent_third = sorted(set(first_planned) - present_first), sorted(set(third_planned) - present_third)
    if absent_first or absent_third:
        missing("%d planned fits of (i) and (ii) and %d of (iii) are in no manifest, e.g. %s"
                % (len(absent_first), len(absent_third), (absent_first or absent_third)[0]))
    return {"manifests": records, "planned": {"i_ii": len(first_planned), "iii": len(third_planned)},
            "absent": {"i_ii": len(absent_first), "iii": len(absent_third)},
            "not_planned": {"i_ii": sorted(present_first - set(first_planned)), "iii": sorted(present_third - set(third_planned))},
            "e1b_manifests": [file_record(E1 / "predictions_manifest.json"), file_record(E1 / "attribution" / "predictions_manifest.json")]}


# --------------------------------------------------------------------------- inputs


def load_cells() -> dict:
    """E1b's G5 and G6 (VCC rows in the loader's order, labels, external tables), and E1b's joint external draws."""
    rows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    e1s.DATA["cells"], e1s.DATA["external"], e1s.DATA["external_source"] = {}, {}, {}
    for cell in CELLS:
        if e1cells.CELLS[cell]["unit"] != "row":
            raise Stop("%s is not a row-level cell" % cell)
        data = e1cells.build_cell(cell, rows, means)
        e1s.DATA["cells"][cell] = data
        for screen in data.spec["screens"]:
            e1s.DATA["external"][(screen, data.external_source[screen])] = data.external[screen]
        e1s.DATA["external_source"][cell] = data.external_source[data.spec["screens"][0]]
    del rows, means
    for screen in SCREENS:
        constructs = None
        for cell in CELLS:
            table = e1s.DATA["external"][(screen, e1s.DATA["cells"][cell].external_source[screen])]
            if constructs is None:
                constructs = table["construct"].astype(str).to_numpy()
            elif not np.array_equal(constructs, table["construct"].astype(str).to_numpy()):
                raise Stop("the external constructs of %s differ between G5 and G6" % screen)
    e1s.STATE["draws"] = e1s.build_draws()
    e1s.STATE["vcc_draws"] = {}
    return {"cell_manifests": {c: file_record(REPO / "bioinf2027" / "endpoints" / c / "manifest.json") for c in CELLS + ("G1",)},
            "e1_input_records": file_record(E1 / "input_records.json")}


def load_e6() -> dict:
    """E6's tables at 27/788 (G1), its primary gene folds (seed 20271204), its draws and E6's own saved predictions."""
    tables = e1cells.load_tables(e6f.ENDPOINTS[e6f.PRIMARY])
    e6s.DATA["tables"] = {e6f.PRIMARY: {s: tables[s] for s in SCREENS}}
    e6s.DATA["n"] = {s: len(tables[s]) for s in SCREENS}
    e6s.DATA["genes"] = {s: tables[s]["target_ensg"].astype(str).to_numpy() for s in SCREENS}
    union = sorted(set(e6s.DATA["genes"][SCREENS[0]].tolist()) | set(e6s.DATA["genes"][SCREENS[1]].tolist()))
    e6s.DATA["folds"] = {e6f.FOLD_SEED: e6f.fold_of(union, e6f.FOLD_SEED)}
    e6s.DATA["draws"], shared = {}, {}
    for screen in SCREENS:
        mine = e1s.STATE["draws"][screen]
        shared[screen] = bool(np.array_equal(e6s.DATA["genes"][screen], mine["groups"]))
        if shared[screen]:
            e6s.DATA["draws"][screen] = mine  # the same clusters in the same order and seed: the same replicates
        else:
            draw = resample.cluster_draw(e6s.DATA["genes"][screen], e1s.B, e6s.SEED[screen])
            e6s.DATA["draws"][screen] = {"draw": draw, "point": e1s._ones(e6s.DATA["n"][screen])}
    manifest = E6 / "predictions_manifest_tier3.json"
    e6s.DATA["files"] = json.loads(manifest.read_text())["files"]
    e6s.DATA["cache"], e6s.DATA["r_cache"] = {}, {}
    return {"e6_predictions_manifest": file_record(manifest), "e6_draw_is_e1b_draw": shared}


def validity() -> dict:
    """E1b's label positive control and [C1; B2] per cell and screen; E6's monotone control per direction at 27/788."""
    pc = pd.read_csv(E1 / "positive_controls.csv", float_precision="round_trip")
    out = {"excluded": {}, "head_limited": {}, "e6_label_ok": {}}
    for cell in CELLS:
        for screen in SCREENS:
            part = pc[(pc.cell == cell) & (pc.screen == screen) & (pc.subset == "all")]
            label = part[part.control.str.startswith("label construction")]
            pipeline = part[part.control == "pipeline [C1; B2]"]
            out["excluded"][(cell, screen)] = bool(label.empty or label["below_0.80"].astype(bool).iloc[0])
            out["head_limited"][(cell, screen)] = bool(not pipeline.empty and pipeline["below_0.80"].astype(bool).iloc[0])
    q = pd.read_csv(E6 / "positive_controls.csv", float_precision="round_trip")
    for source in SOURCES:
        direction = f"{source} to {e6f.other(source)}"
        part = q[(q.direction == direction) & (q.endpoint == e6f.PRIMARY) & (q.control == "label construction")]
        out["e6_label_ok"][direction] = bool(not part.empty and not part["below_0.80"].astype(bool).iloc[0])
    return out


def order_marker() -> str:
    """The Stack set-order flag (E2.md, Section 3): 1 - median correlation above 0.10, read from its record if it exists."""
    if not ORDER_RECORD.exists():
        return "Stack: the set-order sensitivity is not computed (no bioinf2027/results/E2/order_sensitivity.json)"
    results = json.loads(ORDER_RECORD.read_text())["results"]  # per set: the VCC test rows and the K562-essential constructs
    over = {name: float(r["one_minus_median_correlation"]) for name, r in results.items() if float(r["one_minus_median_correlation"]) > ORDER_THRESHOLD}
    return ("Stack: depends on the set order (%s above %.2f)" % ("; ".join("%s %.3f" % kv for kv in over.items()), ORDER_THRESHOLD)) if over else ""


def markers(encoder: str, dataset: str) -> str:
    if encoder != "stack":
        return ""
    parts = [STACK_VCC if dataset == "VCC" else STACK_REPLOGLE, DATA.get("order_marker", "")]
    return "; ".join(p for p in parts if p)


# --------------------------------------------------------------------------- VCC


def score(cell: str, label: str, representation: str, head: str = "forest", kind: str = "genes") -> dict | None:
    """make_e1b_stats.vcc_r2, unchanged, once every planned head seed is found."""
    key = (cell, label, representation, head, kind)
    if key not in CACHE:
        wanted = wanted_seeds(representation, head)
        found = DATA["lookup"].seeds(cell, label, representation, head, seeds=wanted)
        if len(found) != len(wanted):
            missing("VCC %s | %s | %s | %s: %d of %d head seeds" % (cell, label, representation, head, len(found), len(wanted)))
            CACHE[key] = None
        else:
            CACHE[key] = e1s.vcc_r2(cell, label, representation, head, kind=kind)
    return CACHE[key]


minus = e1s.contrast


def half(value):
    return None if value is None else att.half(value)


def _two(a, b):
    return lambda v, k, label: minus(v(a, label, k), v(b, label, k))


def _drop(a, b):
    return lambda v, k, label: minus(minus(v(a, "existing", k), v(b, "existing", k)), minus(v(a, "rebuilt", k), v(b, "rebuilt", k)))


def _delta(control):
    return lambda v, k, label: minus(minus(v("e", label, k), v(control, label, k)), minus(v("[B0; e]", label, k), v(f"[B0; {control}]", label, k)))


def _half(control):
    return lambda v, k, label: minus(minus(v("[B0; e]", label, k), v(f"[B0; {control}]", label, k)), half(minus(v("e", label, k), v(control, label, k))))


# name, roles used, "label" (one value per label) or "paired" (existing minus rebuilt, one joint draw), role, formula, description
QUANTITIES = (
    ("e - C1", ("e", "C1"), "label", "margin", _two("e", "C1"), "e minus C1 of e's width (P-a on the existing label, P-r on the rebuilt label)"),
    ("P-d", ("e", "C1"), "paired", "margin", _drop("e", "C1"), "e minus C1 on the existing label minus the same on the rebuilt label, one joint draw"),
    ("e - B0", ("e", "B0"), "label", "trivial heads", _two("e", "B0"), "e minus the B0 head (cell count; P-c on the existing label)"),
    ("e - B1", ("e", "B1"), "label", "trivial heads", _two("e", "B1"), "e minus the B1 head (control-cell mean of the target gene)"),
    ("e - B2", ("e", "B2"), "label", "trivial heads", _two("e", "B2"), "e minus the B2 head (own shift)"),
    ("drop B0", ("B0", "C1"), "paired", "reference drop beside P-d", _drop("B0", "C1"), "B0 minus C1 of e's width, existing minus rebuilt label; reference for P-d"),
    ("drop B1", ("B1", "C1"), "paired", "reference drop beside P-d", _drop("B1", "C1"), "B1 minus C1 of e's width, existing minus rebuilt label; reference for P-d"),
    ("drop B2", ("B2", "C1"), "paired", "reference drop beside P-d", _drop("B2", "C1"), "B2 minus C1 of e's width, existing minus rebuilt label; reference for P-d"),
    ("drop TR", ("TR", "C1"), "paired", "reference drop beside P-d", _drop("TR", "C1"), "TR minus C1 of e's width, existing minus rebuilt label; reference for P-d"),
    ("e - C3", ("e", "C3"), "label", "reference, unread", _two("e", "C3"), "e minus C3 (e with its rows shuffled); M' of the attribution amendment, an unread row"),
    ("N0 - C1", ("N0", "C1"), "label", "size-matched null", _two("N0", "C1"), "the size-matched null minus C1 of e's width"),
    ("e - N0", ("e", "N0"), "label", "size-matched null", _two("e", "N0"), "e minus its size-matched null"),
    ("d1", ("[B0; e]", "B0"), "label", "attribution", _two("[B0; e]", "B0"), "[B0; e] minus B0"),
    ("d2", ("[B0; C1]", "B0"), "label", "attribution", _two("[B0; C1]", "B0"), "[B0; C1] minus B0 (control)"),
    ("d2'", ("[B0; C3]", "B0"), "label", "attribution", _two("[B0; C3]", "B0"), "[B0; C3] minus B0 (control, e's columns with the rows shuffled)"),
    ("d3", ("[B0; e]", "[B0; C1]"), "label", "attribution", _two("[B0; e]", "[B0; C1]"), "[B0; e] minus [B0; C1]"),
    ("d3'", ("[B0; e]", "[B0; C3]"), "label", "attribution", _two("[B0; e]", "[B0; C3]"), "[B0; e] minus [B0; C3]"),
    ("Delta", ("e", "C1", "[B0; e]", "[B0; C1]"), "label", "attribution", _delta("C1"),
     "Delta = [e minus C1] minus [[B0; e] minus [B0; C1]], inside each replicate"),
    ("D1/2", ("e", "C1", "[B0; e]", "[B0; C1]"), "label", "attribution", _half("C1"),
     "D1/2 = [[B0; e] minus [B0; C1]] minus half of [e minus C1], inside each replicate"),
    ("Delta'", ("e", "C3", "[B0; e]", "[B0; C3]"), "label", "attribution, structure-matched (E1b_attribution.md, Section 11)", _delta("C3"),
     "Delta' = [e minus C3] minus [[B0; e] minus [B0; C3]], inside each replicate"),
    ("D1/2'", ("e", "C3", "[B0; e]", "[B0; C3]"), "label", "attribution, structure-matched (E1b_attribution.md, Section 11)", _half("C3"),
     "D1/2' = [[B0; e] minus [B0; C3]] minus half of [e minus C3], inside each replicate"),
    ("[C1; B2] - B2", ("[C1; B2]", "B2"), "label", "positive control at e's width", _two("[C1; B2]", "B2"),
     "[C1; B2] minus the B2 head: how this head responds to a block of e's width beside one strong column"),
    ("[e; B2] - [C1; B2]", ("[e; B2]", "[C1; B2]"), "label", "positive control at e's width", _two("[e; B2]", "[C1; B2]"),
     "[e; B2] minus [C1; B2]: whether e adds to it (read as (a3) of the post hoc note)"),
    ("e - z", ("e", "z"), "label", "between encoders, descriptive", _two("e", "z"), "e minus Geneformer's z, paired from one draw"),
    ("e - C2-50", ("e", "C2-50"), "label", "PCA-50 beside its random projection", _two("e", "C2-50"), "PCA-50 minus a random projection of x of width 50"),
    ("C2-50 - C1", ("C2-50", "C1"), "label", "PCA-50 beside its random projection", _two("C2-50", "C1"), "the width-50 random projection minus C1 of width 50"),
)
EXPRESSION_QUANTITIES = ("e - C1", "P-d")
BATCH = {"e - C1": CELLS, "P-d": CELLS, "d3": (RECORD_CELL,), "d3'": (RECORD_CELL,), "Delta": (RECORD_CELL,), "Delta'": (RECORD_CELL,)}
ATTRIBUTION_READ = ("d1", "d2", "d2'", "d3", "d3'", "Delta", "D1/2", "Delta'", "D1/2'")
N0_READINGS = {
    "N0 - C1": ("averaging noise alone, at the row's cell count, gives a margin over the Gaussian control of {e}'s width",
                "averaging noise alone is not shown to give a margin over the Gaussian control", "the size-matched null falls below the Gaussian control"),
    "e - N0": ("{e} exceeds its size-matched null", "{e} is not shown to exceed its size-matched null", "the size-matched null exceeds {e}"),
}
POSITIVE_READINGS = {
    "[C1; B2] - B2": ("a Gaussian block of {e}'s width raises this head's R-squared beside the own shift: an effect of the head",
                      "no effect of a Gaussian block of {e}'s width beside the own shift is resolved",
                      "a Gaussian block of {e}'s width lowers this head's R-squared beside the own shift (dilution)"),
    "[e; B2] - [C1; B2]": tuple(re.sub(r"\bz\b", "{e}", t) for t in pho.READINGS["a3"]),
}


def named(text: str, encoder: str) -> str:
    """A reading of the attribution note, written for z at width 768, for encoder e."""
    text = re.sub(r"\bz\b", encoder, text)
    return text.replace("768 columns", "{:,} columns".format(width(encoder)))


def wanted_quantity(encoder: str, quantity: str, roles_used: tuple, cell: str) -> bool:
    if encoder in EXPRESSION and quantity not in EXPRESSION_QUANTITIES:
        return False
    if quantity == "e - z" and encoder not in WIDTHS:
        return False
    return all(available(encoder, r, cell) for r in roles_used)


def vcc_entry(encoder, cell, label, head, quantity, role, description, r2, rho, first=None, second=None, n=None) -> dict:
    s, sp = e1s.summarise(r2), e1s.summarise(rho)
    entry = {"encoder": encoder, "width": width(encoder), "cell": cell, "label": label, "head": head, "quantity": quantity,
             "fact": fact(quantity, label), "role": role, "description": description,
             "n_test_rows": n["n_rows"] if n else np.nan, "n_test_genes": n["n_genes"] if n else np.nan,
             "r2": s["estimate"], "r2_lo": s["lo"], "r2_hi": s["hi"], "r2_p": s["p"], "r2_se": s["se"], "r2_detectable_2.80se": s["detectable_2.80se"],
             "spearman": sp["estimate"], "spearman_lo": sp["lo"], "spearman_hi": sp["hi"], "spearman_p": sp["p"],
             "first": first[0] if first else "", "r2_first": first[1] if first else np.nan,
             "second": second[0] if second else "", "r2_second": second[1] if second else np.nan,
             "state": att.state(s["lo"], s["hi"]), "batch_resampled_lo": np.nan, "batch_resampled_hi": np.nan}
    return entry


def fact(quantity: str, label: str) -> str:
    if quantity == "e - C1":
        return "P-a" if label == "existing" else "P-r"
    if quantity == "P-d":
        return "P-d"
    if quantity == "e - B0" and label == "existing":
        return "P-c"
    return ""


def vcc_rows(encoder: str) -> list[dict]:
    rows = []
    names = roles(encoder)
    for cell in CELLS:
        for head in heads_for(encoder):
            def v(role, label, k, kind="genes"):
                s = score(cell, label, role_name(encoder, role), head, kind)
                return None if s is None else s[k]

            for quantity, used, kind, role, formula, description in QUANTITIES:
                if not wanted_quantity(encoder, quantity, used, cell):
                    continue
                for label in (LABELS if kind == "label" else (PAIRED,)):
                    base = "existing" if label == PAIRED else label
                    n = score(cell, base, role_name(encoder, used[0]), head)
                    first = second = None
                    if kind == "label" and len(used) == 2:
                        a, b = (score(cell, label, role_name(encoder, r), head) for r in used)
                        first = (role_name(encoder, used[0]), np.nan if a is None else a["r2"][0])
                        second = (role_name(encoder, used[1]), np.nan if b is None else b["r2"][0])
                    entry = vcc_entry(encoder, cell, label, head, quantity, role, description, formula(v, "r2", label), formula(v, "spearman", label),
                                      first, second, n)
                    if head == "forest" and cell in BATCH.get(quantity, ()):
                        value = formula(lambda r, lab, k: v(r, lab, k, "batches"), "r2", label)
                        if value is not None:
                            lo, hi, _ = e1s.resample_interval(value[1])
                            entry.update({"batch_resampled_lo": lo, "batch_resampled_hi": hi})
                    rows.append(entry)
            for r, name in names.items():
                if not available(encoder, r, cell):
                    continue
                for label in LABELS:
                    s = score(cell, label, name, head)
                    rows.append(vcc_entry(encoder, cell, label, head, f"R2({name})", "reference R-squared", "R-squared of one representation",
                                          None if s is None else s["r2"], None if s is None else s["spearman"], n=s))
                    rows[-1]["state"] = ""
    return rows


def trivial_rows() -> list[dict]:
    rows = []
    for cell in CELLS:
        for head in HEADS:
            for name in ("B0", "B1", "B2", "TR"):
                for label in LABELS:
                    s = score(cell, label, name, head)
                    rows.append(vcc_entry("trivial heads (E1b)", cell, label, head, f"R2({name})", "reference R-squared", "R-squared of a trivial head",
                                          None if s is None else s["r2"], None if s is None else s["spearman"], n=s))
                    rows[-1]["state"] = ""
    return rows


def vcc_frame(encoders: tuple, with_trivial: bool = False) -> pd.DataFrame:
    rows = []
    for encoder in encoders:
        rows += vcc_rows(encoder)
    if with_trivial:
        rows += trivial_rows()
    frame = pd.DataFrame(rows)
    frame["reading"] = ""
    frame["boosting_agrees_with_forest"] = ""
    frame["descriptive_holm_g6"] = ""
    for encoder in READ_ENCODERS:
        if encoder in encoders:
            add_readings(frame, encoder)
    frame["note"] = [note(r.encoder, r.cell, r.head, r.label, r.role) for r in frame.itertuples()]
    frame["markers"] = [markers(e, "VCC") for e in frame.encoder]
    frame["adjustment"] = STATUS
    return frame


def note(encoder, cell, head, label, role) -> str:
    parts = []
    if cell != RECORD_CELL:
        parts.append(NOT_RECORD)
    if head == "ridge":
        parts.append("ridge, descriptive")
    elif head == "boosting":
        parts.append("boosting, beside the forest; no reading")
    if encoder == GENEFORMER:
        parts.append("Geneformer's z, E1b's fits: the reference for S2, not an E2 reading")
    elif encoder == REFERENCE_ENCODER:
        parts.append("PCA-768, width-matched reference, no reading")
    elif encoder in EXPRESSION:
        parts.append("E1b's %s, printed beside PCA" % encoder)
    elif role.startswith("attribution") and label != "existing" and cell == RECORD_CELL and head == "forest":
        parts.append("rebuilt label, no reading")
    return "; ".join(parts)


def add_readings(frame: pd.DataFrame, encoder: str) -> None:
    """Readings in G6 under the forest: the attribution rows and the size-matched null on the existing label (E1b_attribution.md,
    Sections 2 and 11, E2.md, Section 4), the positive control on both labels; the pattern facts' states."""
    record = (frame.encoder == encoder) & (frame.cell == RECORD_CELL) & (frame["head"] == "forest")
    existing = record & (frame.label == "existing")
    d2 = frame.loc[existing & (frame.quantity == "d2"), "state"]
    d2_state = d2.iloc[0] if len(d2) else "undefined"
    for index in frame.index[existing & frame.quantity.isin(ATTRIBUTION_READ)]:
        q, s = frame.at[index, "quantity"], frame.at[index, "state"]
        text = att.reading(q, s, d2_state) if q in att.READINGS else am.reading(q, s, ("", ""))
        frame.at[index, "reading"] = named(text, encoder)
    for table, mask in ((N0_READINGS, existing), (POSITIVE_READINGS, record)):
        for index in frame.index[mask & frame.quantity.isin(tuple(table))]:
            above, contains, below = table[frame.at[index, "quantity"]]
            text = {"above zero": above, "contains zero": contains, "below zero": below}.get(frame.at[index, "state"], "undefined")
            frame.at[index, "reading"] = text.format(e=encoder)
    for index in frame.index[record & (frame.fact != "")]:
        held = frame.at[index, "r2_lo"] > 0
        frame.at[index, "reading"] = "%s: %s (read in pattern.csv)" % (frame.at[index, "fact"], "resolved above zero" if held else "not resolved above zero")
    for label in LABELS:
        for name in ATTRIBUTION_READ:
            f = frame.index[record & (frame.label == label) & (frame.quantity == name)]
            b = frame.index[(frame.encoder == encoder) & (frame.cell == RECORD_CELL) & (frame["head"] == "boosting") & (frame.label == label)
                            & (frame.quantity == name)]
            if len(f) and len(b):
                frame.loc[f, "boosting_agrees_with_forest"] = str(bool(frame.at[f[0], "state"] == frame.at[b[0], "state"]))
    read = frame.index[existing & frame.quantity.isin(ATTRIBUTION_READ)]
    rejected, thresholds = resample.holm(frame.loc[read, "r2_p"].to_numpy(dtype=float), ALPHA)
    for index, r, t in zip(read, rejected, thresholds):
        frame.at[index, "descriptive_holm_g6"] = "%s at threshold %.4f (descriptive, not a decision rule)" % ("rejected" if r else "not rejected", t)


def label_scale_frame(encoders: tuple) -> pd.DataFrame:
    """P-d's MSE version: [MSE(e) - MSE(C1)] on each label and existing minus rebuilt (forest), with the labels' variances."""
    rows = []
    for cell in CELLS:
        draws = e1s.vcc_draw(cell, "genes")
        for label in LABELS:
            y = e1s.vcc_label(cell, label)
            rows.append({"encoder": "both labels", "cell": cell, "quantity": f"variance of the {label} label over the test rows", "estimate": float(np.var(y)),
                         "lo": np.nan, "hi": np.nan, "p": np.nan, "se": np.nan, "detectable_2.80se": np.nan, "n_test_rows": int(y.size)})
        for encoder in encoders:
            names, values = roles(encoder), {}
            for label in LABELS:
                y = e1s.vcc_label(cell, label)
                fits = {r: DATA["lookup"].seeds(cell, label, names[r], "forest", seeds=wanted_seeds(names[r], "forest")) for r in ("e", "C1")}
                if any(len(f) != len(wanted_seeds(names[r], "forest")) for r, f in fits.items()):
                    missing("VCC MSE %s | %s | %s" % (cell, label, encoder))
                    values[label] = None
                else:
                    pe = np.column_stack([f["test"] for f in fits["e"]])
                    pc = np.column_stack([f["test"] for f in fits["C1"]])
                    point = pho.weighted_mse(y, pe, draws["point"])[0] - pho.weighted_mse(y, pc, draws["point"])[0]
                    values[label] = (float(point), pho.weighted_mse(y, pe, draws["draw"]) - pho.weighted_mse(y, pc, draws["draw"]))
                rows.append({"encoder": encoder, "cell": cell, "quantity": f"MSE of e minus MSE of C1 (forest), {label} label",
                             **summary(values[label]), "n_test_rows": int(y.size)})
            rows.append({"encoder": encoder, "cell": cell, "quantity": "MSE of e minus MSE of C1, existing label minus rebuilt label, one joint draw",
                         **summary(minus(values["existing"], values["rebuilt"])), "n_test_rows": int(y.size)})
    frame = pd.DataFrame(rows)
    frame["role"] = ["beside P-d" if c == RECORD_CELL else NOT_RECORD for c in frame.cell]
    frame["markers"] = [markers(e, "VCC") for e in frame.encoder]
    frame["adjustment"] = STATUS
    return frame


def summary(value) -> dict:
    s = e1s.summarise(value)
    return {k: s[k] for k in ("estimate", "lo", "hi", "p", "se", "detectable_2.80se")}


# --------------------------------------------------------------------------- the pattern (G6, forest)


OUTCOMES = {
    "pattern": ("P-a, P-d, and not P-r",
                "{e} shows S2's pattern: its margin on the all-cells label is resolved, is smaller on the size-fixed label by P-d [interval], "
                "and is not shown on the size-fixed label", "S2's finding holds for {e} as a bounded paired difference"),
    "margin on the size-fixed label": ("P-a and P-r", "{e} carries a margin on the size-fixed label in G6",
                                       "S2 does not hold for {e} in G6. Reported with {e} - B2 head and [{e}; B2] - [C1; B2], and qualified as \"may rest "
                                       "on the target's own expression\" (the size-fixed label tracks the own shift at +0.89 to +0.98, and {e} sees the "
                                       "target gene). (ii) is read only if the two-cell precondition also passes"),
    "no change shown": ("P-a, not P-d, not P-r", "{e}'s margin is not shown to change with label construction",
                        "S2's drop is not shown for {e}; nothing is read from the rebuilt-label value alone"),
    "no margin": ("not P-a", "{e} has no resolved margin on the all-cells label", "the pattern does not apply to {e}; its rebuilt-label value is reported"),
    "undefined": ("a fact is not computed", "undefined", "undefined"),
}


def pattern_outcome(pa_lo: float, pd_lo: float, pr_lo: float) -> str:
    if not all(np.isfinite(v) for v in (pa_lo, pd_lo, pr_lo)):
        return "undefined"
    a, d, r = pa_lo > 0, pd_lo > 0, pr_lo > 0
    if a and r:
        return "margin on the size-fixed label"
    if a and d:
        return "pattern"
    if a:
        return "no change shown"
    return "no margin"


def pick(frame: pd.DataFrame, encoder, cell, label, quantity, head="forest") -> pd.Series | None:
    part = frame[(frame.encoder == encoder) & (frame.cell == cell) & (frame.label == label) & (frame["head"] == head) & (frame.quantity == quantity)]
    if len(part) > 1:
        raise Stop("%d rows for %s | %s | %s | %s | %s" % (len(part), encoder, cell, label, quantity, head))
    return part.iloc[0] if len(part) else None


def precondition(frame: pd.DataFrame, encoder: str) -> dict:
    """E1b's precondition per cell: e - C1 on the rebuilt label, VCC test, forest, interval above zero; passed in both cells."""
    out = {}
    for cell in CELLS:
        row = pick(frame, encoder, cell, "rebuilt", "e - C1")
        out[cell] = bool(row is not None and row.r2_lo > 0)
    out["both cells"] = all(out[c] for c in CELLS)
    return out


def pattern_frame(vcc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for encoder in (GENEFORMER,) + READ_ENCODERS + (REFERENCE_ENCODER,):
        facts = {}
        entry = {"encoder": encoder, "width": width(encoder), "cell": RECORD_CELL, "head": "forest"}
        for name, cell, label, quantity in (("P-a", RECORD_CELL, "existing", "e - C1"), ("P-d", RECORD_CELL, PAIRED, "P-d"),
                                            ("P-r", RECORD_CELL, "rebuilt", "e - C1"), ("P-c", RECORD_CELL, "existing", "e - B0"),
                                            ("G5 P-a", "G5", "existing", "e - C1"), ("G5 P-d", "G5", PAIRED, "P-d"),
                                            ("G5 P-r", "G5", "rebuilt", "e - C1"), ("G5 P-c", "G5", "existing", "e - B0")):
            row = pick(vcc, encoder, cell, label, quantity)
            keys = ("r2", "r2_lo", "r2_hi", "r2_p", "r2_se", "r2_detectable_2.80se") if not name.startswith("G5") else ("r2", "r2_lo", "r2_hi")
            for k in keys:
                entry[f"{name}_{k.replace('r2_', '').replace('r2', 'estimate')}"] = np.nan if row is None else row[k]
            if name in ("P-a", "P-d"):
                entry[f"{name}_batch_resampled_lo"] = np.nan if row is None else row.batch_resampled_lo
                entry[f"{name}_batch_resampled_hi"] = np.nan if row is None else row.batch_resampled_hi
            facts[name] = np.nan if row is None else row.r2_lo
            if name in ("P-a", "P-d", "P-r"):
                entry[f"{name}_holds"] = bool(np.isfinite(facts[name]) and facts[name] > 0)
        for name, label, quantity in (("e - B2, existing", "existing", "e - B2"), ("e - B2, rebuilt", "rebuilt", "e - B2"),
                                      ("[e; B2] - [C1; B2], existing", "existing", "[e; B2] - [C1; B2]"),
                                      ("[e; B2] - [C1; B2], rebuilt", "rebuilt", "[e; B2] - [C1; B2]"),
                                      ("N0 - C1, existing", "existing", "N0 - C1"), ("e - N0, existing", "existing", "e - N0")):
            row = pick(vcc, encoder, RECORD_CELL, label, quantity)
            for k in ("r2", "r2_lo", "r2_hi"):
                entry[f"{name}_{k.replace('r2_', '').replace('r2', 'estimate')}"] = np.nan if row is None else row[k]
        outcome = pattern_outcome(facts["P-a"], facts["P-d"], facts["P-r"])
        condition, reading, meaning = OUTCOMES[outcome]
        pre = precondition(vcc, encoder)
        entry.update({"precondition_G5": pre["G5"], "precondition_G6": pre["G6"], "precondition_passed_in_both_cells": pre["both cells"],
                      "outcome": outcome, "condition": condition})
        if encoder in READ_ENCODERS:
            entry.update({"reading": reading.format(e=encoder), "for_S2": meaning.format(e=encoder),
                          "claims": "per encoder, unadjusted (the pattern facts are secondary)"})
        else:
            entry.update({"reading": "reference, no reading" + (" (Geneformer's z, E1b's fits)" if encoder == GENEFORMER else " (PCA-768)"),
                          "for_S2": "", "claims": ""})
        entry["markers"] = markers(encoder, "VCC")
        rows.append(entry)
    frame = pd.DataFrame(rows)
    read = frame.encoder.isin(READ_ENCODERS)
    p = np.concatenate([frame.loc[read, "P-a_p"].to_numpy(dtype=float), frame.loc[read, "P-d_p"].to_numpy(dtype=float)])
    rejected, thresholds = resample.holm(p, ALPHA)
    n = int(read.sum())
    for column, offset in (("P-a", 0), ("P-d", n)):
        frame[f"{column}_descriptive_holm"] = ""
        for j, index in enumerate(frame.index[read]):
            frame.at[index, f"{column}_descriptive_holm"] = "%s at threshold %.4f (descriptive, not a decision rule)" % (
                "rejected" if rejected[offset + j] else "not rejected", thresholds[offset + j])
    shown = [r.encoder for r in frame[read].itertuples() if r.outcome == "pattern"]
    any_margin = any(r["P-a_holds"] for _, r in frame[read].iterrows())
    summary_row = {"encoder": "all read encoders", "cell": RECORD_CELL, "head": "forest"}
    if not any_margin:
        summary_row.update({"outcome": "not P-a for every encoder", "reading": "no new encoder has the submitted kind of margin",
                            "for_S2": "S2 stays a Geneformer result; the paper says the other encoders show no margin to lose"})
    else:
        summary_row.update({"outcome": "pattern shown by: " + (", ".join(shown) if shown else "none"),
                            "reading": "a sentence about \"the encoders\" is made only for encoders that each show the pattern, naming them",
                            "for_S2": ""})
    frame = pd.concat([frame, pd.DataFrame([summary_row])], ignore_index=True)
    frame["adjustment"] = "secondary, unadjusted; the Holm columns are descriptive, not a decision rule (E2.md, Section 2)"
    return frame


S2_ROWS = (
    ("A", "(d3) and (d3') both resolved above zero",
     "beside the perturbed-cell count, {e} gives this {head} more than an independent Gaussian block and more than its own row-shuffled columns of "
     "the same width; consistent with {e} carrying the control-cell mean or the own shift"),
    ("A'", "(d3) resolved above zero, (d3') not",
     "beside cell count, {e} gives this {head} more than an independent Gaussian block of the same width; not separated from a difference in block structure"),
    ("B", "Delta resolved above zero", "beside cell count, {e}'s advantage over a Gaussian block of the same width is smaller than without it"),
    ("C", "D1/2 upper bound below zero",
     "beside cell count, {e}'s advantage over the Gaussian block is less than half of its {head} margin over C1 without it"),
    ("D", "none of A, B and C", "not determined"),
    ("B'", "Delta' resolved above zero", "beside cell count, {e}'s advantage over its row-shuffled copy of the same width is smaller than without it"),
    ("C'", "D1/2' upper bound below zero",
     "beside cell count, {e}'s advantage over its row-shuffled copy is less than half of its {head} margin over C3 without it"),
)


def s2_holds(get: dict) -> dict:
    def st(q):
        return get[q].state if q in get else "undefined"

    def hi(q):
        return get[q].r2_hi if q in get else np.nan

    a = st("d3") == "above zero" and st("d3'") == "above zero"
    a1 = st("d3") == "above zero" and st("d3'") != "above zero"
    b = st("Delta") == "above zero"
    c = bool(np.isfinite(hi("D1/2")) and hi("D1/2") < 0)
    return {"A": a, "A'": a1, "B": b, "C": c, "D": not (a or b or c), "B'": st("Delta'") == "above zero",
            "C'": bool(np.isfinite(hi("D1/2'")) and hi("D1/2'") < 0)}


def attribution_s2_frame(vcc: pd.DataFrame) -> pd.DataFrame:
    """Rows A to D (E1b_attribution.md, Section 2) and B', C' (Section 11), per read encoder, G6, existing label."""
    rows = []
    for encoder in READ_ENCODERS:
        holds = {}
        for head in ("forest", "boosting"):
            part = vcc[(vcc.encoder == encoder) & (vcc.cell == RECORD_CELL) & (vcc.label == "existing") & (vcc["head"] == head)]
            holds[head] = s2_holds({r.quantity: r for r in part.itertuples()})
        for head in ("forest", "boosting"):
            for row, condition, text in S2_ROWS:
                rows.append({"encoder": encoder, "width": width(encoder), "cell": RECORD_CELL, "label": "existing", "head": head, "row": row,
                             "condition": condition, "holds": bool(holds[head][row]), "reading": text.format(e=encoder, head=head),
                             "role": "the S2 reading (forest)" if head == "forest" else "beside the forest: the S2 reading is specific to the forest unless boosting agrees",
                             "boosting_agrees": bool(holds["forest"][row] == holds["boosting"][row]),
                             "note": "rows B and C name the Gaussian control; B' and C' the row-shuffled control (attribution results review r1)",
                             "markers": markers(encoder, "VCC")})
    frame = pd.DataFrame(rows)
    frame["adjustment"] = STATUS
    return frame


def power_frame(vcc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, name), planned in PLANNED_DETECTABLE.items():
        label, quantity = {"P-a": ("existing", "e - C1"), "P-d": (PAIRED, "P-d"), "P-r": ("rebuilt", "e - C1"), "P-c": ("existing", "e - B0")}[name]
        for encoder in (GENEFORMER,) + READ_ENCODERS:
            row = pick(vcc, encoder, cell, label, quantity)
            rows.append({"fact": name, "cell": cell, "encoder": encoder, "se": np.nan if row is None else row.r2_se,
                         "detectable_2.80se": np.nan if row is None else row["r2_detectable_2.80se"],
                         "planned_detectable_from_geneformer": planned,
                         "note": "size detectable with 80 % power at 2.80 SE (E2.md, Section 6); at Geneformer's own margin an encoder with the "
                                 "same true margin would reach P-a about half the time" if name == "P-a" and encoder == GENEFORMER else ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- external (ii)


EXTERNAL_ROLES = ("e", "C1", "C3", "C2-50", "[e; B2]", "[C1; B2]")


def external_wanted(encoder: str, role: str, cell: str, head: str) -> bool:
    if not available(encoder, role, cell) or role in FIXED:
        return False
    return role in ("e", "C1") or head == "forest"


def external_objects(encoders: tuple) -> list[tuple]:
    out = set()
    for encoder in encoders:
        for cell in CELLS:
            for screen in SCREENS:
                for label in LABELS:
                    for head in heads_for(encoder):
                        for role in EXTERNAL_ROLES:
                            if external_wanted(encoder, role, cell, head):
                                out.add((cell, screen, label, role_name(encoder, role), head))
    return sorted(out)


def request_external(scorer, encoders: tuple) -> int:
    objects = external_objects(encoders)
    for cell, screen, label, representation, head in objects:
        endpoint = (e1s.DATA["cells"][cell].external_source[screen], "y")
        e1s.request_object(scorer, screen, endpoint, "all", cell, label, representation, head, e1s.ALL_ROWS, 0, wanted_seeds(representation, head))
    return len(objects)


def rho_ext(cell, screen, label, representation, head="forest"):
    """make_e1b_stats.object_stat on all constructs: the mean over head seeds of the external Spearman, point and replicates."""
    endpoint = (e1s.DATA["cells"][cell].external_source[screen], "y")
    wanted = wanted_seeds(representation, head)
    value = e1s.object_stat(e1s.STATE["scorer"], screen, endpoint, "all", cell, label, representation, head, e1s.ALL_ROWS, 0, wanted)
    if value is None or value[2] != len(wanted):
        missing("external %s | %s | %s | %s | %s" % (cell, screen, label, representation, head))
        return None
    return value[0], value[1]


def ext_entry(encoder, cell, screen, label, head, quantity, value, role) -> dict:
    return {"encoder": encoder, "width": width(encoder), "cell": cell, "screen": screen, "label": label, "head": head, "quantity": quantity,
            **summary(value), "role": role}


def external_frame(encoders: tuple) -> pd.DataFrame:
    rows = []
    for encoder in encoders:
        names = roles(encoder)
        for cell in CELLS:
            for screen in SCREENS:
                for head in heads_for(encoder):
                    t = {}
                    for label in LABELS:
                        values = {r: rho_ext(cell, screen, label, names[r], head) for r in EXTERNAL_ROLES if external_wanted(encoder, r, cell, head)}
                        for r, value in values.items():
                            rows.append(ext_entry(encoder, cell, screen, label, head, f"rho({names[r]})", value, "Spearman with the external label"))
                        t[label] = minus(values.get("e"), values.get("C1"))
                        family = encoder in READ_ENCODERS and head == "forest" and label == "rebuilt"
                        rows.append(ext_entry(encoder, cell, screen, label, head, "T", t[label],
                                              "family" if family else ("component of L" if encoder in READ_ENCODERS and head == "forest" else STATUS)))
                        if "C3" in values:
                            rows.append(ext_entry(encoder, cell, screen, label, head, "C3 - C1", minus(values["C3"], values["C1"]), STATUS))
                        if "C2-50" in values:
                            rows.append(ext_entry(encoder, cell, screen, label, head, "e - C2-50", minus(values["e"], values["C2-50"]), STATUS))
                            rows.append(ext_entry(encoder, cell, screen, label, head, "C2-50 - C1", minus(values["C2-50"], values["C1"]), STATUS))
                        if head == "forest" and encoder in WIDTHS and encoder != "stack":  # E2.md Sections 2 and 17: no comparison between encoders on the Replogle screens for Stack
                            rows.append(ext_entry(encoder, cell, screen, label, head, "e - z", minus(values["e"], rho_ext(cell, screen, label, "z")),
                                                  "between encoders, descriptive"))
                    family = encoder in READ_ENCODERS and head == "forest"
                    rows.append(ext_entry(encoder, cell, screen, L_LABEL, head, "L", minus(t["rebuilt"], t["existing"]), "family" if family else STATUS))
    frame = pd.DataFrame(rows)
    frame["adjustment"] = ["Holm in E2's family (family.csv)" if r == "family" else STATUS for r in frame.role]
    frame["markers"] = [markers(e, "Replogle") for e in frame.encoder]
    frame["note"] = ["" if e in READ_ENCODERS else ("reference, no reading" if e != GENEFORMER else "Geneformer's z, E1b's fits; reference")
                     for e in frame.encoder]
    return frame


# --------------------------------------------------------------------------- E6 (iii)


E6_ROLES = ("e", "C1", "[e; B2]", "[C1; B2]")


def e6_lookup(spec) -> dict | None:
    for source in DATA["e6_sources"]:
        try:
            values = source.get(spec.context, spec.label, spec.representation, spec.head, spec.seed)
        except RuntimeError as error:
            raise Stop(str(error)) from error
        if values is not None:
            return values
    return None


def assembled_e2(encoder: str, source: str, role: str, seed: int) -> np.ndarray | None:
    """As make_e6_stats.assembled: each target construct takes the prediction of the fold that holds its gene."""
    target = e6f.other(source)
    genes = e6s.DATA["genes"][target]
    fold = e6s.folds_of(genes, e6s.DATA["folds"][e6f.FOLD_SEED])
    out = np.full(genes.size, np.nan)
    for k in range(5):
        spec = DATA["e6_specs"].get((encoder, source, k, role, seed))
        values = None if spec is None else e6_lookup(spec)
        if values is None or target not in values:
            return None
        rows = np.flatnonzero(fold == k)
        if values[target].shape != (rows.size,):
            raise Stop("%s: %d predictions for %d constructs of fold %d" % (spec.key(), values[target].size, rows.size, k))
        out[rows] = values[target]
    return out


def e6_available(encoder: str, role: str) -> bool:
    if encoder in (REFERENCE_ENCODER,) + EXPRESSION:
        return role in ("e", "C1")
    return role in E6_ROLES


def e6_vectors(encoder: str, source: str, role: str) -> list | None:
    if encoder == GENEFORMER or encoder in EXPRESSION or (encoder == REFERENCE_ENCODER and role == "C1"):
        model = {GENEFORMER: GENEFORMER_NAMES}.get(encoder, {"e": encoder})[role] if role != "C1" else "C1"
        return [e6s.assembled(source, e6f.PRIMARY, model, "forest", s) for s in e6f.SEEDS]
    seeds = e6f.SEEDS[:1] if SMOKE else e6f.SEEDS
    return [assembled_e2(encoder, source, role, s) for s in seeds]


def e6_rho(encoder: str, source: str, role: str):
    key = ("e6", encoder, source, role)
    if key not in CACHE:
        try:
            vectors = e6_vectors(encoder, source, role)
        except Stop:
            raise
        except RuntimeError as error:  # make_e6_stats.prediction: a file that does not match E6's manifest
            raise Stop(str(error)) from error
        if vectors is None or any(v is None for v in vectors):
            missing("(iii) %s | %s | %s" % (encoder, source, role))
            CACHE[key] = None
        else:
            target = e6f.other(source)
            CACHE[key] = e6s.rho(target, vectors, e6s.label_of(target, e6f.PRIMARY))
    return CACHE[key]


def e6_frame(encoders: tuple) -> pd.DataFrame:
    rows = []
    for encoder in encoders:
        names = roles(encoder)
        for source in SOURCES:
            target = e6f.other(source)
            direction = f"{source} to {target}"
            values = {r: e6_rho(encoder, source, r) for r in E6_ROLES if e6_available(encoder, r)}

            def add(quantity, value, role):
                rows.append({"encoder": encoder, "width": width(encoder), "direction": direction, "endpoint": e6f.PRIMARY, "head": "forest",
                             "quantity": quantity, **summary(value), "role": role})

            for r, value in values.items():
                add(f"rho({names[r]})", value, "Spearman with the target label, gene-disjoint predictions")
            add("T", minus(values.get("e"), values.get("C1")), "family" if encoder in READ_ENCODERS else STATUS)
            for b in ("B1", "B2"):
                add(f"e minus the target screen's raw {b}", minus(values.get("e"), e6s.raw(target, b)), STATUS)
            if encoder in WIDTHS and encoder != "stack":  # E2.md Sections 2 and 17: not on the Replogle screens for Stack
                add("e - z", minus(values.get("e"), e6_rho(GENEFORMER, source, "e")), "between encoders, descriptive")
    frame = pd.DataFrame(rows)
    frame["adjustment"] = ["Holm in E2's family (family.csv)" if r == "family" else STATUS for r in frame.role]
    frame["markers"] = [markers(e, "Replogle") for e in frame.encoder]
    frame["note"] = ["" if e in READ_ENCODERS else ("reference, no reading" if e != GENEFORMER else "E6's z; reference") for e in frame.encoder]
    return frame


# --------------------------------------------------------------------------- the family and the readings


def stat(frame: pd.DataFrame, **where) -> dict:
    part = frame
    for k, v in where.items():
        part = part[part[k] == v]
    if len(part) != 1:
        raise Stop("%d rows where one was expected: %s" % (len(part), where))
    row = part.iloc[0]
    return {k: row[k] for k in ("estimate", "lo", "hi", "p", "se", "detectable_2.80se")}


def family_frame(external: pd.DataFrame, e6: pd.DataFrame, pre: dict, valid: dict) -> pd.DataFrame:
    rows = []
    for encoder in READ_ENCODERS:
        for cell in CELLS:
            for screen in SCREENS:
                for statistic, label in (("T", "rebuilt"), ("L", L_LABEL)):
                    values = stat(external, encoder=encoder, cell=cell, screen=screen, label=label, head="forest", quantity=statistic)
                    excluded = any(valid["excluded"][(c, screen)] for c in CELLS)
                    read = pre[encoder]["both cells"] and not excluded
                    why = "" if read else ("fails E1b's two-cell precondition" if not pre[encoder]["both cells"] else
                                           "a cell of the pair is excluded on this screen (label positive control below +0.80)")
                    rows.append({"test": "ii", "encoder": encoder, "width": width(encoder), "cell": cell, "screen": screen, "direction": "",
                                 "statistic": statistic, **values, "read": read, "why_not_read": why, "markers": markers(encoder, "Replogle")})
        for source in SOURCES:
            direction = f"{source} to {e6f.other(source)}"
            values = stat(e6, encoder=encoder, direction=direction, quantity="T")
            read = valid["e6_label_ok"][direction]
            rows.append({"test": "iii", "encoder": encoder, "width": width(encoder), "cell": "", "screen": e6f.other(source), "direction": direction,
                         "statistic": "T", **values, "read": read,
                         "why_not_read": "" if read else "E6's monotone positive control is below +0.80 (or missing) in this direction",
                         "markers": markers(encoder, "Replogle")})
    frame = pd.DataFrame(rows)
    if len(frame) != 30:
        raise Stop("the family holds %d statistics, not 30" % len(frame))
    rejected, thresholds = resample.holm(frame["p"].to_numpy(dtype=float), ALPHA)
    frame["holm_threshold"] = thresholds
    frame["rejected"] = rejected
    frame["state"] = [resample.holm_state(e, r) for e, r in zip(frame["estimate"], rejected)]
    frame["family"] = FAMILY_NOTE
    return frame


def external_readings(family: pd.DataFrame, external: pd.DataFrame, pre: dict, valid: dict) -> pd.DataFrame:
    """E1_design.md, Section 11, per read encoder and screen, with T and L of E2's family; no R."""
    rows = []
    state = {(r.encoder, r.cell, r.screen, r.statistic): r.state for r in family[family.test == "ii"].itertuples()}
    for encoder in READ_ENCODERS:
        for screen in SCREENS:
            excluded = [c for c in CELLS if valid["excluded"][(c, screen)]]
            t = [state[(encoder, c, screen, "T")] for c in CELLS]
            l_states = [state[(encoder, c, screen, "L")] for c in CELLS]
            head_limited = any(valid["head_limited"][(c, screen)] for c in CELLS)
            if excluded:
                statement = "a cell of the primary pair is excluded on this screen (label positive control below +0.80)"
            elif not pre[encoder]["both cells"]:
                statement = "uninformative about transfer (fails the precondition in a cell of the pair); values reported, not read"
            elif t[0] != t[1]:
                statement = "undetermined: the two cells of the pair differ"
            elif t[0] == "Holm-negative":
                statement = "read as limited by the head ([C1; B2 native] below +0.80)" if head_limited else "transfer fails after label matching"
            elif t[0] == "inconclusive":
                statement = "neither a transfer nor a failure is resolved"
            else:
                statement = "transfer resolved; " + NO_R
            if not pre[encoder]["both cells"] or excluded:
                l_reading = "not read"
            elif l_states[0] != l_states[1]:
                l_reading = "undetermined"
            elif l_states[0] == "Holm-positive":
                l_reading = "label matching raises the arm's contrast over C1"
            elif l_states[0] == "Holm-negative":
                l_reading = "label matching lowers the arm's contrast over C1"
            else:
                l_reading = "no change from label matching is resolved" + (
                    "; the failure is not measurably changed by label matching" if t[0] in ("Holm-negative", "inconclusive") else "")
            own = pick_ext(external, encoder, screen)
            rows.append({"encoder": encoder, "width": width(encoder), "screen": screen, "T_states": "; ".join(t), "L_states": "; ".join(l_states),
                         "precondition_passed_in_both_cells": pre[encoder]["both cells"], "excluded_cells": ";".join(excluded),
                         "head_limited_by_E1b_[C1; B2]": head_limited, "own_width_[C1; B2]_G6_rebuilt": own,
                         "statement": statement, "L_reading": l_reading,
                         "restricted_input_sensitivity": RESTRICTED_INPUTS if pre[encoder]["both cells"] else "",
                         "markers": markers(encoder, "Replogle")})
    frame = pd.DataFrame(rows)
    by_encoder = frame.groupby("encoder")["statement"].nunique()
    frame["screens_agree"] = [bool(by_encoder[e] == 1) for e in frame.encoder]
    frame["note"] = ["" if a else "the screens disagree: the encoder's contribution is undetermined (E1_design.md, Section 11)" for a in frame.screens_agree]
    return frame


def pick_ext(external: pd.DataFrame, encoder: str, screen: str) -> float:
    part = external[(external.encoder == encoder) & (external.cell == RECORD_CELL) & (external.screen == screen) & (external.label == "rebuilt")
                    & (external["head"] == "forest") & (external.quantity == f"rho({roles(encoder)['[C1; B2]']})")]
    return float(part["estimate"].iloc[0]) if len(part) else np.nan


def e6_readings(family: pd.DataFrame, e6: pd.DataFrame, valid: dict) -> pd.DataFrame:
    """E6.md, Section 7, for T alone (no R), with E6's qualifiers (E2.md, Section 2, (iii))."""
    rows = []
    state = {(r.encoder, r.direction): r.state for r in family[family.test == "iii"].itertuples()}
    for encoder in READ_ENCODERS:
        for source in SOURCES:
            direction = f"{source} to {e6f.other(source)}"

            def est(quantity):
                part = e6[(e6.encoder == encoder) & (e6.direction == direction) & (e6.quantity == quantity)]
                return float(part["estimate"].iloc[0]) if len(part) else np.nan

            names = roles(encoder)
            c1b2, eb2 = est(f"rho({names['[C1; B2]']})"), est(f"rho({names['[e; B2]']})")
            head_limited = bool(np.isfinite(c1b2) and c1b2 < POSITIVE_LIMIT)
            masks = bool(np.isfinite(eb2) and eb2 < POSITIVE_LIMIT and not head_limited)
            t = state[(encoder, direction)]
            if not valid["e6_label_ok"][direction]:
                statement = "no reading: the monotone positive control is below +0.80 (or missing) in this direction"
            elif t == "Holm-negative":
                statement = "read as limited by the head ([C1; B2 native] below +0.80)" if head_limited else "the arm ranks the target constructs below the Gaussian control"
            elif t == "inconclusive":
                statement = "neither transfer nor its failure is resolved; the interval bounds it" + (
                    "; read as limited by the head ([C1; B2 native] below +0.80)" if head_limited else "")
            else:
                statement = ("trained on one Replogle screen, %s ranks the other screen's constructs of untrained target genes above a Gaussian "
                             "control of its width; %s" % (encoder, NO_R))
            notes = ["the own coordinate is not excluded (no masked arm for this encoder)"]
            if masks:
                notes.append("[%s; B2 native] is below +0.80 while [C1; B2 native] is not: %s masks the signal under this head" % (encoder, encoder))
            for b in ("B1", "B2"):
                value = est(f"e minus the target screen's raw {b}")
                if np.isfinite(value) and value <= 0:
                    notes.append(f"the untrained {b} of the target screen is at least as high: no 'transfer works' wording")
            notes.append(E6_MARKERS_ABSENT)
            rows.append({"encoder": encoder, "width": width(encoder), "direction": direction, "T": t, "label_positive_control_passed": valid["e6_label_ok"][direction],
                         "own_width_[C1; B2]": c1b2, "[e; B2]": eb2, "head_limited": head_limited, "statement": statement,
                         "qualifiers": "; ".join(notes), "markers": markers(encoder, "Replogle"), "representation": names["e"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- check 7: reproduction


def compare(checks: list, where: str, mine, theirs) -> None:
    a, b = float(mine), float(theirs)
    difference = 0.0 if (np.isnan(a) and np.isnan(b)) else abs(a - b)
    checks.append({"where": where, "recomputed": a, "file": b, "difference": difference})


def one(frame: pd.DataFrame, where: str) -> pd.Series:
    if len(frame) != 1:
        raise Stop("%s: %d rows where one was expected" % (where, len(frame)))
    return frame.iloc[0]


def reproduction(vcc: pd.DataFrame, scale: pd.DataFrame, external: pd.DataFrame, e6: pd.DataFrame) -> dict:
    """Geneformer's values from this script's code against E1b's, the post hoc note's, the attribution note's and E6's files."""
    checks: list = []
    full = not SMOKE
    read = lambda path: pd.read_csv(path, float_precision="round_trip")  # noqa: E731
    pre = read(E1 / "preconditions.csv")
    for cell in CELLS:
        for label in LABELS:
            row = one(pre[(pre.cell == cell) & (pre.label == label) & (pre.arm == "z")], f"preconditions.csv {cell} {label}")
            mine = pick(vcc, GENEFORMER, cell, label, "e - C1")
            pairs = [("r2_arm", "r2_first"), ("r2_C1", "r2_second"), ("r2_difference_estimate", "r2"), ("spearman_difference", "spearman")]
            if full:
                pairs += [("r2_difference_lo", "r2_lo"), ("r2_difference_hi", "r2_hi"), ("r2_difference_p", "r2_p"), ("r2_difference_se", "r2_se"),
                          ("spearman_difference_lo", "spearman_lo"), ("spearman_difference_hi", "spearman_hi")]
                if label == "rebuilt":
                    pairs += [("batch_resampled_lo", "batch_resampled_lo"), ("batch_resampled_hi", "batch_resampled_hi")]
            for theirs, ours in pairs:
                compare(checks, f"preconditions.csv, {cell}, {label}, z, {theirs}", mine[ours], row[theirs])
    columns = [("r2_difference", "r2"), ("spearman_difference", "spearman")] + (
        [("r2_lo", "r2_lo"), ("r2_hi", "r2_hi"), ("r2_p", "r2_p"), ("r2_se", "r2_se"), ("spearman_lo", "spearman_lo"), ("spearman_hi", "spearman_hi")] if full else [])
    posthoc = read(E1 / "posthoc" / "posthoc_vcc.csv")
    names = {"a1": ("existing", "e - B0"), "a1b": ("existing", "e - B1"), "a1c": ("existing", "e - B2"), "b": (PAIRED, "P-d"),
             "b-B0": (PAIRED, "drop B0"), "b-B1": (PAIRED, "drop B1"), "b-B2": (PAIRED, "drop B2"), "b-TR": (PAIRED, "drop TR"),
             "a3": ("existing", "[e; B2] - [C1; B2]")}
    for cell in CELLS:
        for name, (label, quantity) in names.items():
            if name == "a3" and cell != RECORD_CELL:
                continue
            row = one(posthoc[(posthoc.cell == cell) & (posthoc.quantity == name)], f"posthoc_vcc.csv {cell} {name}")
            mine = pick(vcc, GENEFORMER, cell, label, quantity)
            for theirs, ours in columns:
                compare(checks, f"posthoc_vcc.csv, {cell}, {name}, {theirs}", mine[ours], row[theirs])
    label_scale = read(E1 / "posthoc" / "posthoc_label_scale.csv")
    texts = {"MSE of z minus MSE of C1 (forest), existing label": "MSE of e minus MSE of C1 (forest), existing label",
             "MSE of z minus MSE of C1 (forest), rebuilt label": "MSE of e minus MSE of C1 (forest), rebuilt label",
             "MSE of z minus MSE of C1, existing label minus rebuilt label, one joint draw":
                 "MSE of e minus MSE of C1, existing label minus rebuilt label, one joint draw"}
    for cell in CELLS:
        for theirs_name, ours_name in texts.items():
            row = one(label_scale[(label_scale.cell == cell) & (label_scale.quantity == theirs_name)], f"posthoc_label_scale.csv {cell}")
            mine = one(scale[(scale.encoder == GENEFORMER) & (scale.cell == cell) & (scale.quantity == ours_name)], "label_scale")
            for k in (("estimate", "lo", "hi") if full else ("estimate",)):
                compare(checks, f"posthoc_label_scale.csv, {cell}, {theirs_name}, {k}", mine[k], row[k])
        for label in LABELS:
            q = f"variance of the {label} label over the test rows"
            row = one(label_scale[(label_scale.cell == cell) & (label_scale.quantity == q)], "posthoc_label_scale.csv")
            mine = one(scale[(scale.encoder == "both labels") & (scale.cell == cell) & (scale.quantity == q)], "label_scale")
            compare(checks, f"posthoc_label_scale.csv, {cell}, {q}", mine["estimate"], row["estimate"])
    attribution_columns = [("r2", "r2"), ("spearman", "spearman")] + (
        [("r2_lo", "r2_lo"), ("r2_hi", "r2_hi"), ("r2_p", "r2_p"), ("r2_se", "r2_se"), ("spearman_lo", "spearman_lo"), ("spearman_hi", "spearman_hi")] if full else [])
    batch_columns = [("batch_resampled_lo", "batch_resampled_lo"), ("batch_resampled_hi", "batch_resampled_hi")] if full else []
    for path, quantities in ((E1 / "attribution" / "attribution_vcc.csv", {"d1": "d1", "d2": "d2", "d2'": "d2'", "d3": "d3", "d3'": "d3'", "Delta": "Delta",
                                                                         "D1/2": "D1/2", "M": "e - C1", "z - C3": "e - C3", "[C1; B2] - B2": "[C1; B2] - B2"}),
                             (E1 / "attribution" / "amendment" / "amendment_vcc.csv", {"Delta'": "Delta'", "D1/2'": "D1/2'", "M'": "e - C3"})):
        theirs_frame = read(path)
        for _, row in theirs_frame[theirs_frame.quantity.isin(tuple(quantities))].iterrows():
            mine = pick(vcc, GENEFORMER, row["cell"], row["label"], quantities[row["quantity"]], row["head"])
            if mine is None:
                if row["quantity"] == "[C1; B2] - B2" and row["cell"] != RECORD_CELL:
                    continue  # computed in G6 only here
                raise Stop("%s: no value here for %s" % (path.name, dict(row[["cell", "label", "head", "quantity"]])))
            where = "%s, %s, %s, %s, %s" % (path.name, row["cell"], row["label"], row["head"], row["quantity"])
            for theirs, ours in attribution_columns:
                compare(checks, f"{where}, {theirs}", mine[ours], row[theirs])
            if np.isfinite(row.get("batch_resampled_lo", np.nan)):
                for theirs, ours in batch_columns:
                    compare(checks, f"{where}, {theirs}", mine[ours], row[theirs])
    family = read(E1 / "primary_family.csv")
    for cell in CELLS:
        for screen in SCREENS:
            for statistic, label in (("T", "rebuilt"), ("L", L_LABEL)):
                row = one(family[(family.cell == cell) & (family.screen == screen) & (family.statistic == statistic) & (family.arm == "z")], "primary_family.csv")
                mine = stat(external, encoder=GENEFORMER, cell=cell, screen=screen, label=label, head="forest", quantity=statistic)
                for k in (("estimate", "lo", "hi", "p", "se") if full else ("estimate",)):
                    compare(checks, f"primary_family.csv, {cell}, {screen}, {statistic}, z, {k}", mine[k], row[k])
    e6_family = read(E6 / "family.csv")
    for source in SOURCES:
        direction = f"{source} to {e6f.other(source)}"
        row = one(e6_family[(e6_family.direction == direction) & (e6_family.statistic == "T") & (e6_family.arm == "z")], "E6 family.csv")
        mine = stat(e6, encoder=GENEFORMER, direction=direction, quantity="T")
        for k in (("estimate", "lo", "hi", "p", "se") if full else ("estimate",)):
            compare(checks, f"E6 family.csv, {direction}, T, z, {k}", mine[k], row[k])
    digests = replicate_digests()
    failed = [c for c in checks if not c["difference"] <= TOLERANCE]
    if failed:
        raise Stop("check 7: %d values differ by more than %g, the first: %s" % (len(failed), TOLERANCE, failed[0]))
    if full and not all(d["equal"] for d in digests.values()):
        raise Stop("check 7: a replicate index differs from the record: %s" % {k: v for k, v in digests.items() if not v["equal"]})
    return {"tolerance": TOLERANCE, "values_compared": len(checks), "largest_difference": max(c["difference"] for c in checks),
            "intervals_compared": full, "replicate_index_digests": digests, "checks": checks}


def replicate_digests() -> dict:
    out = {}
    posthoc = json.loads((E1 / "posthoc" / "posthoc.json").read_text())["vcc_replicate_index_sha256"]
    for cell in CELLS:
        for kind in ("genes", "batches"):
            ours = e1s.vcc_draw(cell, kind)["draw"].index_sha256
            out[f"VCC {cell} | {kind}"] = {"here": ours, "record": posthoc.get(f"{cell} | {kind}"), "equal": ours == posthoc.get(f"{cell} | {kind}")}
    e1_stats = json.loads((E1 / "stats.json").read_text())["replicate_index_sha256"]
    e6_stats = json.loads((E6 / "stats.json").read_text())["replicate_index_digests"]
    for screen in SCREENS:
        ours = e1s.STATE["draws"][screen]["draw"].index_sha256
        out[f"external {screen}"] = {"here": ours, "record": e1_stats.get(screen), "equal": ours == e1_stats.get(screen)}
        e6_ours = e6s.DATA["draws"][screen]["draw"].index_sha256
        theirs = e6_stats.get(screen, {}).get("E6")
        out[f"E6 {screen}"] = {"here": e6_ours, "record": theirs, "equal": e6_ours == theirs}
    return out


# --------------------------------------------------------------------------- main


def code_digests() -> dict:
    modules = [sys.modules[__name__], e1s, pho, att, am, e2f, e2e6, e6f, e6s, sys.modules["make_e1b_fits"]]
    modules += [sys.modules[f"bioinf2027.{m}"] for m in ("config", "e1cells", "e1inputs", "resample", "stats", "transfer", "provenance", "heads", "representations")
                if f"bioinf2027.{m}" in sys.modules]
    return {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in modules}


def encoder_records() -> dict:
    out = {"pretraining_check": file_record(E2 / "pretraining_check.json"), "features": file_record(E2 / "features.json")}
    for name in ("probe_scgpt.json", "probe_stack.json"):
        path = E2 / name
        record = json.loads(path.read_text()) if path.exists() else {}
        out[name] = {**file_record(path), "determinism_passed": record.get("determinism", {}).get("passed")}
    out["order_sensitivity"] = {**file_record(ORDER_RECORD), "marker": DATA.get("order_marker", "")}
    out["coverage"] = "no coverage record is read by this script (E2.md, Section 16, outputs)"
    return out


def write(frames: dict) -> dict:
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, frame in frames.items():
        frame.to_csv(folder / name, index=False)
        outputs[name] = file_record(folder / name)
    return outputs


def main() -> None:
    started = time.time()
    e1s.WORKERS = WORKERS
    DATA["missing"] = []
    DATA["order_marker"] = order_marker()
    predictions = load_predictions()
    inputs = {**load_cells(), **load_e6()}
    valid = validity()
    scorer = e1s.Scorer()
    e1s.STATE["scorer"] = scorer
    # Geneformer's z and E1b's x and C2 first: check 7 before any E2 value is computed
    references = (GENEFORMER,) + EXPRESSION
    objects = request_external(scorer, references)
    scorer.run()
    vcc_reference = vcc_frame(references)
    scale_reference = label_scale_frame((GENEFORMER,))
    external_reference = external_frame(references)
    e6_reference = e6_frame(references)
    checks = reproduction(vcc_reference, scale_reference, external_reference, e6_reference)
    print("[e2 stats] check 7 passed: %d values within %g (largest %.2e), %.0f s"
          % (checks["values_compared"], TOLERANCE, checks["largest_difference"], time.time() - started), flush=True)
    encoders = READ_ENCODERS + (REFERENCE_ENCODER,)
    objects += request_external(scorer, encoders)
    scorer.run()
    vcc = pd.concat([vcc_reference, vcc_frame(encoders, with_trivial=True)], ignore_index=True)
    scale = pd.concat([scale_reference, label_scale_frame(encoders)], ignore_index=True)
    scale = scale.drop_duplicates(subset=["encoder", "cell", "quantity"], keep="first").reset_index(drop=True)
    external = pd.concat([external_reference, external_frame(encoders)], ignore_index=True)
    e6 = pd.concat([e6_reference, e6_frame(encoders)], ignore_index=True)
    pre = {e: precondition(vcc, e) for e in (GENEFORMER,) + encoders + EXPRESSION}
    family = family_frame(external, e6, pre, valid)
    frames = {"family.csv": family, "pattern.csv": pattern_frame(vcc), "vcc.csv": vcc, "label_scale.csv": scale,
              "attribution_s2.csv": attribution_s2_frame(vcc), "external.csv": external,
              "readings_external.csv": external_readings(family, external, pre, valid), "e6.csv": e6,
              "readings_e6.csv": e6_readings(family, e6, valid), "power.csv": power_frame(vcc)}
    outputs = write(frames)
    record = {
        "plan": PLAN, "status": "the family is in family.csv; every other value is " + STATUS, "smoke": SMOKE,
        "settings": {"replicates": e1s.B, "vcc_seed": e1s.VCC_SEED, "batch_seed": e1s.BATCH_SEED, "external_seeds": e1s.EXT_SEED,
                     "e6_seeds": e6s.SEED, "e6_fold_seed": e6f.FOLD_SEED, "e6_endpoint": e6f.PRIMARY, "head_seeds": HEADS, "alpha": ALPHA,
                     "tolerance": TOLERANCE, "positive_control_limit": POSITIVE_LIMIT, "order_threshold": ORDER_THRESHOLD, "workers": WORKERS},
        "family": FAMILY_NOTE, "cell_of_record": RECORD_CELL,
        "encoders": {"read": {e: WIDTHS[e] for e in READ_ENCODERS}, "reference": {REFERENCE_ENCODER: WIDTHS[REFERENCE_ENCODER]},
                     "geneformer": "E1b's z and C1; N0_z from make_e2_fits.py --geneformer-null"},
        "preconditions": {e: v for e, v in pre.items()},
        "vcc_replicate_index_sha256": {f"{c} | {k}": v["draw"].index_sha256 for (c, k), v in e1s.STATE["vcc_draws"].items()},
        "external_replicate_index_sha256": {s: e1s.STATE["draws"][s]["draw"].index_sha256 for s in SCREENS},
        "inputs": {**inputs, "predictions": predictions,
                   **{f"E1/{n}": file_record(E1 / n) for n in ("preconditions.csv", "primary_family.csv", "positive_controls.csv", "stats.json")},
                   **{f"E1/{n}": file_record(E1 / n) for n in ("posthoc/posthoc_vcc.csv", "posthoc/posthoc_label_scale.csv", "posthoc/posthoc.json",
                                                              "attribution/attribution_vcc.csv", "attribution/amendment/amendment_vcc.csv")},
                   **{f"E6/{n}": file_record(E6 / n) for n in ("family.csv", "positive_controls.csv", "stats.json")}},
        "encoder_records": encoder_records(), "code": code_digests(), "check_7": checks,
        "external_objects_scored": objects, "missing_in_this_run": DATA["missing"],
        "outputs": outputs, "seconds": round(time.time() - started, 1),
    }
    (out_dir() / "stats.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 stats] done in %.0f s; family: %s; %d values missing%s" % (
        time.time() - started, ", ".join("%s %s" % (s, (family.state == s).sum()) for s in ("Holm-positive", "Holm-negative", "inconclusive")),
        len(DATA["missing"]), " (smoke run)" if SMOKE else ""), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--workers" in arguments:
        i = arguments.index("--workers")
        if i + 1 >= len(arguments) or not arguments[i + 1].isdigit() or int(arguments[i + 1]) < 1:
            raise SystemExit("--workers takes a positive integer")
        WORKERS = int(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if arguments == ["--smoke"]:
        SMOKE = True
        e1s.B = 20
        resample.BLOCK = 20
        PLAN = PLAN + "; smoke run, 20 replicates, never reported"
    elif arguments:
        raise SystemExit("usage: make_e2_stats.py [--smoke] [--workers N]")
    main()
