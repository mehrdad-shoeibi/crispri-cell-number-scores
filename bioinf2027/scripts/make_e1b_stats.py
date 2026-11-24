"""E1b, the statistics of the evaluation, from the saved predictions of bioinf2027/scripts/make_e1b_fits.py.

Design: bioinf2027/plans/E1_design.md, Sections 10 and 11, as amended by Sections 21 to 28.

- One joint target-gene cluster bootstrap per screen (10,000 replicates; seeds 20260901 for RPE1 and
  20260902 for K562-essential; the draw of earlier_revision/pipeline/bootstrap_wiring.py:84-117)
  supplies every external statistic of that screen. Per-seed statistics are averaged over the forest's
  seeds 0 to 2 inside each replicate.
- VCC test: a target-gene cluster bootstrap of the test rows (10,000 replicates, seed 20270301), and the
  batch-resampled sensitivity (seed 20270302).
- The primary family: T, L, R and R- for z, x, m, [x; m] and the masked arms in G5 and G6 on both
  screens, 60 statistics, Holm at family-wise 0.05; the "reduces to" family of 8 (Section 23.2).
- Everything else is secondary and unadjusted, and every table says so.

Outputs under bioinf2027/results/E1/: primary_family.csv, reduces_to_family.csv, preconditions.csv,
transfer.csv, secondary_contrasts.csv, baselines.csv, positive_controls.csv, null_band.csv,
refits.csv, bridge.csv, composition.csv, secondary_heads.csv, ridge_penalties.csv, oracle.csv,
readings.csv and stats.json (settings, inputs, replicate digests).

With --smoke: 20 replicates, the smoke predictions, outputs to the scratch folder, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, e1cells, e1inputs, heads, resample, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E1"
RESULTS = REPO / "bioinf2027" / "results" / "E1"
PLAN = "bioinf2027/plans/E1_design.md, Sections 10 and 11, as amended by Sections 21 to 28"
B = 10000
EXT_SEED = {"RPE1": 20260901, "K562-essential": 20260902}
VCC_SEED = 20270301
BATCH_SEED = 20270302
FOLD_SEEDS = (20270409, 20270410, 20270411, 20270412, 20270413)
ALPHA = 0.05
MARGIN_R, MARGIN_INCREMENT = 0.10, 0.03
POSITIVE_LIMIT = 0.80
GATE = 0.2
NEAR_CEILING = 0.1
IN_SUPPORT = "B0 inside the VCC training range"
DESCRIPTIVE_GENES = 50
DETECTABLE = 2.80
SEEDS = (0, 1, 2)
WORKERS = 14
SCREENS = e1inputs.SCREENS
FAMILY = transfer.FAMILY
MASKED = transfer.MASKED
UNMASKED_OF = {"x-": "x", "m-": "m", "[x; m]-": "[x; m]"}
ALL_ROWS = "all training rows"
SMOKE = False

DATA: dict = {}
STATE: dict = {}


# --------------------------------------------------------------------------- predictions


class Predictions:
    """The saved predictions, refused unless each file matches the manifest's digest."""

    def __init__(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        self.manifest = manifest
        self.files = manifest["files"]
        self.checked: set = set()
        self.cache: dict = {}

    def get(self, context, label, representation, head, seed, variant=ALL_ROWS, realisation=0) -> dict | None:
        key = transfer.FitSpec(context, label, representation, head, int(seed), variant, int(realisation)).key()
        if key in self.cache:
            return self.cache[key]
        record = self.files.get(key)
        if record is None:
            return None
        path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if key not in self.checked:
            if sha256(path) != record["sha256"]:
                raise RuntimeError("prediction file of %s does not match the manifest" % key)
            self.checked.add(key)
        _, fit_record, values = transfer.load_predictions(path)
        values["_record"] = fit_record
        self.cache[key] = values
        return values

    def seeds(self, context, label, representation, head="forest", variant=ALL_ROWS, realisation=0, seeds=SEEDS) -> list[dict]:
        out = []
        for seed in seeds:
            values = self.get(context, label, representation, head, seed, variant, realisation)
            if values is not None:
                out.append(values)
        return out


# --------------------------------------------------------------------------- draws


def _rank01(values: np.ndarray) -> np.ndarray:
    return sps.rankdata(values) / (values.size + 1.0)


def _ones(n: int) -> resample.ClusterDraw:
    return resample.ClusterDraw(weights=np.ones((1, n)), clusters=np.array([]), seed=-1, index_sha256="")


def build_draws() -> dict:
    out = {}
    for screen in SCREENS:
        table = DATA["external"][(screen, "G5")]
        groups = table["target_ensg"].astype(str).to_numpy()
        out[screen] = {"draw": resample.cluster_draw(groups, B, EXT_SEED[screen]), "point": _ones(groups.size), "groups": groups}
    return out


def _spearman_task(screen: str, columns: np.ndarray, label: np.ndarray, mask) -> tuple[np.ndarray, np.ndarray]:
    draws = STATE["draws"][screen]
    point = resample.weighted_spearman(columns, label, draws["point"], mask=mask)[0]
    replicates = resample.weighted_spearman(columns, label, draws["draw"], mask=mask)
    return point, replicates


def parallel(function, argument_list: list[tuple]) -> list:
    if not argument_list:
        return []
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = [pool.submit(function, *args) for args in argument_list]
        return [f.result() for f in futures]


class Scorer:
    """External Spearman of prediction vectors against an endpoint, in one batch per (screen, label, mask)."""

    def __init__(self) -> None:
        self.requests: dict = {}
        self.results: dict = {}

    def request(self, screen: str, endpoint: tuple, mask_name: str, key: str, vector: np.ndarray) -> None:
        self.requests.setdefault((screen, endpoint, mask_name), {})[key] = np.asarray(vector, dtype=np.float64)

    def run(self) -> None:
        tasks, index = [], []
        for (screen, endpoint, mask_name), vectors in self.requests.items():
            label = endpoint_label(screen, endpoint)
            mask = None if mask_name == "all" else DATA["mask_arrays"][(screen, mask_name)]
            keys = sorted(vectors)
            for start in range(0, len(keys), 8):
                chunk = keys[start : start + 8]
                tasks.append((screen, np.column_stack([vectors[k] for k in chunk]), label, mask))
                index.append(((screen, endpoint, mask_name), chunk))
        results = parallel(_spearman_task, tasks)
        for (group, chunk), (point, replicates) in zip(index, results):
            for j, key in enumerate(chunk):
                self.results[group + (key,)] = (float(point[j]), replicates[:, j])
        self.requests = {}

    def get(self, screen, endpoint, mask_name, key):
        return self.results[(screen, endpoint, mask_name, key)]


def endpoint_label(screen: str, endpoint: tuple) -> np.ndarray:
    source, column = endpoint
    return DATA["external"][(screen, source)][column].to_numpy(dtype=np.float64)


def endpoint_of(cell: str, column: str = "y") -> tuple:
    return (e1cells_external_source(cell), column)


def e1cells_external_source(cell: str) -> str:
    return DATA["external_source"][cell]


# --------------------------------------------------------------------------- objects


def object_key(context, label, representation, head="forest", variant=ALL_ROWS, realisation=0) -> str:
    return "|".join([context, label, representation, head, variant, str(realisation)])


def request_object(scorer: Scorer, screen: str, endpoint: tuple, mask_name: str, context, label, representation, head="forest",
                   variant=ALL_ROWS, realisation=0, seeds=SEEDS, basis=None) -> bool:
    side = screen if basis is None else f"{screen}|{basis}"
    requested = False
    for seed in seeds:
        values = DATA["predictions"].get(context, label, representation, head, seed, variant, realisation)
        if values is None or side not in values:
            continue
        scorer.request(screen, endpoint, mask_name, object_key(context, label, representation, head, variant, realisation) + f"|{seed}|{basis}", values[side])
        requested = True
    return requested


def object_stat(scorer: Scorer, screen, endpoint, mask_name, context, label, representation, head="forest", variant=ALL_ROWS,
                realisation=0, seeds=SEEDS, basis=None):
    """Mean over seeds of the external Spearman, point and replicates; None if not fitted."""
    base = object_key(context, label, representation, head, variant, realisation)
    points, draws = [], []
    for seed in seeds:
        key = base + f"|{seed}|{basis}"
        if (screen, endpoint, mask_name, key) not in scorer.results:
            continue
        p, d = scorer.get(screen, endpoint, mask_name, key)
        points.append(p)
        draws.append(d)
    if not points:
        return None
    return float(np.mean(points)), np.mean(draws, axis=0), len(points)


def summarise(value) -> dict:
    """summary() of a (point, replicates) pair, or missing values where the statistic was not fitted."""
    if value is None:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "se": float("nan"),
                "detectable_2.80se": float("nan"), "replicates_undefined": None}
    return summary(value[0], value[1])


def summary(point: float, draws: np.ndarray, *_) -> dict:
    # a third value (the number of seeds of get_rho) is accepted and ignored
    lo, hi, skipped = resample_interval(draws)
    se = float(np.nanstd(draws, ddof=1)) if np.isfinite(draws).sum() > 1 else float("nan")
    return {"estimate": point, "lo": lo, "hi": hi, "p": resample.p_value(draws), "se": se, "detectable_2.80se": DETECTABLE * se,
            "replicates_undefined": skipped}


def resample_interval(draws: np.ndarray):
    from bioinf2027.stats import percentile_interval

    return percentile_interval(draws)


# --------------------------------------------------------------------------- residuals for R


def oof_residuals(values: np.ndarray, covariates: np.ndarray, genes: np.ndarray, fold_seed: int, kind: str = "boosting") -> np.ndarray:
    """Out-of-fold residuals of rank-scaled values on the covariates, five folds grouped by gene."""
    from sklearn.linear_model import LinearRegression

    y = _rank01(np.asarray(values, dtype=np.float64))
    unique = np.array(sorted(set(genes.tolist())))
    fold_of = dict(zip(unique[np.random.default_rng(fold_seed).permutation(unique.size)], np.arange(unique.size) % 5))
    folds = np.array([fold_of[g] for g in genes])
    out = np.full(y.size, np.nan)
    with heads.single_thread():
        for f in range(5):
            test = folds == f
            model = heads.nuisance() if kind == "boosting" else LinearRegression()
            model.fit(covariates[~test], y[~test])
            out[test] = y[test] - model.predict(covariates[test])
    return out


def linear_rank_residuals(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Residual of the ranks of values on the ranks of the covariates (with intercept), in sample."""
    ranks = sps.rankdata(values)
    design = np.column_stack([np.ones(values.size)] + [sps.rankdata(covariates[:, j]) for j in range(covariates.shape[1])])
    coefficients, *_ = np.linalg.lstsq(design, ranks, rcond=None)
    return ranks - design @ coefficients


def _residual_task(screen: str, endpoint: tuple, fold_seed: int, kind: str, columns, vector: np.ndarray) -> np.ndarray:
    table = DATA["external"][(screen, endpoint[0])]
    covariates = table[list(columns)].to_numpy(dtype=np.float64)
    genes = table["target_ensg"].astype(str).to_numpy()
    if kind == "linear":
        return linear_rank_residuals(vector, covariates)
    return oof_residuals(vector, covariates, genes, fold_seed, "boosting")


class Residualiser:
    """R and its relatives: residual correlations of label and prediction given B0 to B4."""

    def __init__(self) -> None:
        self.requests: dict = {}
        self.results: dict = {}

    def request(self, screen, endpoint, fold_seed, kind, columns, key, vector):
        self.requests[(screen, endpoint, fold_seed, kind, tuple(columns), key)] = np.asarray(vector, dtype=np.float64)

    def run(self) -> None:
        # the label of every endpoint enters once per fold seed and kind
        labels = {}
        for (screen, endpoint, fold_seed, kind, columns, _), _vector in self.requests.items():
            labels[(screen, endpoint, fold_seed, kind, columns, "__label__")] = endpoint_label(screen, endpoint)
        items = list({**labels, **self.requests}.items())
        results = parallel(_residual_task, [(k[0], k[1], k[2], k[3], k[4], v) for k, v in items])
        for (key, _), residual in zip(items, results):
            self.results[key] = residual
        self.requests = {}

    def correlation(self, screen, endpoint, fold_seed, kind, columns, key, mask=None):
        """Residual correlation over all constructs, or within a subset (the residuals stay those of the fits on all constructs)."""
        e_p = self.results[(screen, endpoint, fold_seed, kind, tuple(columns), key)]
        e_y = self.results[(screen, endpoint, fold_seed, kind, tuple(columns), "__label__")]
        draws = STATE["draws"][screen]
        point = resample.weighted_pearson(e_p, e_y, draws["point"], mask=mask)[0, 0]
        replicates = resample.weighted_pearson(e_p, e_y, draws["draw"], mask=mask)[:, 0]
        return float(point), replicates


def request_r(res: Residualiser, screen, endpoint, context, label, representation, fold_seeds=(FOLD_SEEDS[0],), kind="boosting",
              columns=e1cells.TRIVIAL, head="forest", variant=ALL_ROWS):
    count = 0
    for seed in SEEDS:
        values = DATA["predictions"].get(context, label, representation, head, seed, variant)
        if values is None:
            continue
        count += 1
        for fold_seed in fold_seeds:
            res.request(screen, endpoint, fold_seed, kind, columns, object_key(context, label, representation, head, variant) + f"|{seed}", _rank_input(values[screen]))
    return count


def _rank_input(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float64)


def r_stat(res: Residualiser, screen, endpoint, context, label, representation, fold_seed=FOLD_SEEDS[0], kind="boosting",
           columns=e1cells.TRIVIAL, head="forest", variant=ALL_ROWS, mask=None):
    base = object_key(context, label, representation, head, variant)
    points, draws = [], []
    for seed in SEEDS:
        if (screen, endpoint, fold_seed, kind, tuple(columns), base + f"|{seed}") not in res.results:
            continue
        p, d = res.correlation(screen, endpoint, fold_seed, kind, columns, base + f"|{seed}", mask=mask)
        points.append(p)
        draws.append(d)
    if not points:
        return None
    return float(np.mean(points)), np.mean(draws, axis=0)


# --------------------------------------------------------------------------- VCC side


def vcc_draw(context: str, kind: str = "genes") -> dict:
    key = (context, kind)
    if key in STATE["vcc_draws"]:
        return STATE["vcc_draws"][key]
    data = DATA["cells"][context]
    test = data.vcc[data.split("test")]
    groups = test["target_gene"].astype(str).to_numpy() if kind == "genes" else test["batch"].astype(str).to_numpy()
    seed = VCC_SEED if kind == "genes" else BATCH_SEED
    draw = resample.cluster_draw(groups, B, seed)
    STATE["vcc_draws"][key] = {"draw": draw, "point": _ones(groups.size), "groups": groups}
    return STATE["vcc_draws"][key]


def vcc_label(context: str, label: str) -> np.ndarray:
    data = DATA["cells"][context]
    test = data.vcc[data.split("test")]
    if label.startswith("rebuilt:"):
        other = DATA["cells"][label.split(":", 1)[1]].vcc.set_index(["split", "batch", "target_gene"])["y"]
        return other.loc[list(zip(test["split"], test["batch"], test["target_gene"]))].to_numpy(dtype=np.float64)
    column = {"rebuilt": "y", "source": "src_y", "mean": "y_mean", "existing": "existing_label"}[label]
    return test[column].to_numpy(dtype=np.float64)


def vcc_r2(context, label, representation, head="forest", variant=ALL_ROWS, kind="genes", label_for_scoring=None):
    """Mean over seeds of the VCC test R-squared, point and replicates; and the same for Spearman."""
    fits = DATA["predictions"].seeds(context, label, representation, head, variant)
    if not fits:
        return None
    y = vcc_label(context, label_for_scoring or label)
    draws = vcc_draw(context, kind)
    predictions = np.column_stack([f["test"] for f in fits])
    r2_point = resample.weighted_r2(y, predictions, draws["point"])[0].mean()
    r2 = resample.weighted_r2(y, predictions, draws["draw"]).mean(axis=1)
    rho_point = resample.weighted_spearman(predictions, y, draws["point"])[0].mean()
    rho = resample.weighted_spearman(predictions, y, draws["draw"]).mean(axis=1)
    return {"r2": (float(r2_point), r2), "spearman": (float(rho_point), rho), "n_rows": int(y.size), "n_genes": int(np.unique(draws["groups"]).size)}


def vcc_gene_level(context, label, representation, head="forest"):
    """Gene-level R-squared and Spearman on VCC test: label and prediction averaged per target gene."""
    fits = DATA["predictions"].seeds(context, label, representation, head)
    if not fits:
        return None
    y = vcc_label(context, label)
    draws = vcc_draw(context, "genes")
    groups = draws["groups"]
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)

    def per_gene(v):
        return np.bincount(inverse, weights=v) / counts

    gy = per_gene(y)
    gp = np.column_stack([per_gene(f["test"]) for f in fits])
    first_row = np.array([np.flatnonzero(inverse == g)[0] for g in range(unique.size)])
    gene_draw = resample.ClusterDraw(weights=draws["draw"].weights[:, first_row], clusters=unique, seed=VCC_SEED, index_sha256=draws["draw"].index_sha256)
    point = _ones(unique.size)
    return {
        "r2": (float(resample.weighted_r2(gy, gp, point)[0].mean()), resample.weighted_r2(gy, gp, gene_draw).mean(axis=1)),
        "spearman": (float(resample.weighted_spearman(gp, gy, point)[0].mean()), resample.weighted_spearman(gp, gy, gene_draw).mean(axis=1)),
        "n_genes": int(unique.size),
    }


# --------------------------------------------------------------------------- inputs


def load_inputs() -> dict:
    cells = list(e1cells.CELLS)
    rows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    modelled = e1cells.load_tables("G2")["vcc"]
    DATA["cells"] = {cell: e1cells.build_cell(cell, rows, means, modelled=modelled) for cell in cells}
    DATA["external"], DATA["external_source"] = {}, {}
    for cell, data in DATA["cells"].items():
        for screen in data.spec["screens"]:
            DATA["external"][(screen, data.external_source[screen])] = data.external[screen]
        DATA["external_source"][cell] = data.external_source[data.spec["screens"][0]]
    # the endpoint of a derived cell on its own screen
    supplement = json.loads((REPO / "bioinf2027" / "endpoints" / "readouts" / "supplement" / "supplement.json").read_text())
    ceilings = {}
    for name, record in supplement["per_row_ceilings"].items():
        path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != record["sha256"]:
            raise RuntimeError("%s does not match the supplement's record" % name)
        ceilings[name] = pd.read_csv(path, float_precision="round_trip")
    DATA["ceilings"] = ceilings
    DATA["r5_decisions"] = json.loads((REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json").read_text())["r5_gate"]
    # every VCC target gene, all splits: the non-VCC-target subsets use the training and validation
    # targets, the oracle reference every target
    genes = set(DATA["cells"]["G2"].vcc["target_gene"].astype(str))
    DATA["vcc_ensembl"], DATA["vcc_unmatched"] = e1inputs.vcc_target_ensembl(genes)
    return {"supplement": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "supplement" / "supplement.json"),
            "r5_gate": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json"),
            "vcc_annotation": file_record(e1inputs.vcc_annotation_file()),
            "cell_manifests": {c: file_record(REPO / "bioinf2027" / "endpoints" / c / "manifest.json") for c in cells}}


def build_masks() -> None:
    """Construct subsets per screen and cell: target in or out of the features, outside the VCC targets, away from the ceiling."""
    masks = {}
    for (screen, source), table in DATA["external"].items():
        feature = table["target_is_feature"].to_numpy(dtype=bool)
        masks[(screen, source, "target is a feature")] = feature
        masks[(screen, source, "target is not a feature")] = ~feature
    for cell, data in DATA["cells"].items():
        train = set(data.vcc.loc[data.vcc["split"] == "train", "target_gene"].astype(str))
        val = set(DATA["cells"]["G2"].vcc.loc[DATA["cells"]["G2"].vcc["split"] == "val", "target_gene"].astype(str))
        excluded = {DATA["vcc_ensembl"][g] for g in train | val if g in DATA["vcc_ensembl"]}
        for screen in data.spec["screens"]:
            source = data.external_source[screen]
            table = DATA["external"][(screen, source)]
            masks[(screen, cell, "target not a VCC training or validation target")] = ~table["target_ensg"].astype(str).isin(excluded).to_numpy()
            train_b0 = data.vcc.loc[data.vcc["split"] == "train", "B0"].to_numpy(dtype=np.float64)
            b0 = table["B0"].to_numpy(dtype=np.float64)
            inside = (b0 >= train_b0.min()) & (b0 <= train_b0.max())
            masks[(screen, cell, IN_SUPPORT)] = inside
            DATA.setdefault("support_share", {})[(cell, screen)] = {"constructs": int(b0.size), "inside": int(inside.sum()),
                                                                     "share_inside": float(inside.mean()),
                                                                     "vcc_training_b0_min": float(train_b0.min()), "vcc_training_b0_max": float(train_b0.max())}
            name = f"ceiling_{cell}_{screen}.csv"
            if name in DATA["ceilings"]:
                ceiling = DATA["ceilings"][name]
                if not np.array_equal(ceiling["construct"].astype(str).to_numpy(), table["construct"].astype(str).to_numpy()):
                    raise RuntimeError("the ceiling table of %s on %s lists other constructs" % (cell, screen))
                gap = ceiling["ceiling"].to_numpy(float) - table["y"].to_numpy(float)
                masks[(screen, cell, "further than 0.1 below the ceiling")] = gap > NEAR_CEILING
                DATA.setdefault("ceiling_share", {})[(cell, screen)] = float((gap <= NEAR_CEILING).mean())
    DATA["masks"] = masks


def mask_key(cell: str, screen: str, name: str):
    """Masks are keyed by the external source for availability and by the cell for the cell-specific subsets."""
    if name in ("target is a feature", "target is not a feature"):
        return DATA["cells"][cell].external_source[screen]
    return cell


def mask_id(cell: str, screen: str, name: str) -> str | None:
    """The identifier of a subset's array, registered on first use; None if the subset does not exist."""
    if name == "all":
        return "all"
    owner = mask_key(cell, screen, name)
    if (screen, owner, name) not in DATA["masks"]:
        return None
    identifier = f"{name}@{owner}"
    DATA.setdefault("mask_arrays", {})[(screen, identifier)] = DATA["masks"][(screen, owner, name)]
    return identifier


# --------------------------------------------------------------------------- the family


def family_and_secondary(inputs: dict) -> dict:
    preds = DATA["predictions"]
    scorer = Scorer()
    res = Residualiser()
    contexts = list(e1cells.CELLS)
    model_list = list(transfer.REPRESENTATIONS)

    # every scored object: (cell, screen, endpoint column, mask, context, label, representation, head, variant, realisation)
    plan = []

    def want(cell, screen, context, label, representation, head="forest", variant=ALL_ROWS, realisation=0, column="y", mask="all", basis=None, seeds=SEEDS):
        endpoint = (DATA["cells"][cell].external_source[screen], column)
        identifier = mask_id(cell, screen, mask)
        if identifier is None:
            return
        if request_object(scorer, screen, endpoint, identifier, context, label, representation, head, variant, realisation, seeds, basis):
            plan.append((cell, screen, column, identifier, context, label, representation, head, variant, realisation, basis))

    def existing_context(cell):
        spec = e1cells.CELLS[cell]
        return spec.get("base", cell)

    subsets = ("all", "target is a feature", "target is not a feature", "target not a VCC training or validation target", "further than 0.1 below the ceiling", IN_SUPPORT)
    for cell in contexts:
        for screen in DATA["cells"][cell].spec["screens"]:
            for r in model_list:
                head_list = ("monotone",) if r == "label positive control" else ("forest", "boosting", "ridge")
                for head in head_list:
                    seeds = SEEDS if head in ("forest", "boosting") else (0,)
                    for mask in (subsets if head == "forest" else ("all", IN_SUPPORT)):
                        want(cell, screen, cell, "rebuilt", r, head, mask=mask, seeds=seeds)
                        want(cell, screen, existing_context(cell), "existing", r, head, mask=mask, seeds=seeds)
                    if head == "forest":
                        want(cell, screen, cell, "rebuilt", r, head, basis="release", seeds=seeds)
            if cell in e1cells.PRIMARY:
                for label in ("source", "mean"):
                    for r in FAMILY + MASKED + ("C1",):
                        want(cell, screen, cell, label, r, column="y_mean" if label == "mean" else "y")
                for r in FAMILY + ("C1",):
                    want(cell, screen, cell, "existing", r, column="y_mean")
                for control, count in (("C1", 100), ("C3", 100), ("C2", 20)):
                    for k in range(1, count):
                        want(cell, screen, cell, "rebuilt", control, realisation=k, seeds=(0,))
                for r in FAMILY + ("C1",):
                    for refit in range(DATA["refits"]):
                        want(cell, screen, cell, "rebuilt", r, variant=f"refit {refit}", seeds=(0,))
                    for variant in ("target is a feature", "target is not a feature", "single-guide rows"):
                        want(cell, screen, cell, "rebuilt", r, variant=variant)
                    want(cell, screen, cell, "existing", r, variant="single-guide rows")
                    want(cell, screen, cell, "rebuilt", r, seeds=(0,))
            if cell in ("G1", "G2"):
                bl = {"G1": "G5", "G2": "G6"}[cell]
                for r in FAMILY + ("C1",):
                    want(cell, screen, bl, f"rebuilt:{cell}", r)
                    want(cell, screen, bl, "existing", r)
    # the bridge: the submitted configuration and the stored 27/788 endpoint
    for screen in SCREENS:
        for context in ("G6", "G5", "G1", "G2"):
            for r in FAMILY + ("C1", "C2", "C3"):
                want("G1", screen, context, "existing", r, column="stored_y_primary")
    print("[stats] scoring %d objects" % len(plan), flush=True)
    scorer.run()

    # residualisation for R, R- and their relatives
    for cell in contexts:
        for screen in DATA["cells"][cell].spec["screens"]:
            endpoint = (DATA["cells"][cell].external_source[screen], "y")
            fold_seeds = FOLD_SEEDS if cell in e1cells.PRIMARY else (FOLD_SEEDS[0],)
            for r in FAMILY + MASKED:
                request_r(res, screen, endpoint, cell, "rebuilt", r, fold_seeds=fold_seeds)
                if cell in e1cells.PRIMARY:
                    request_r(res, screen, endpoint, cell, "source", r)
                    request_r(res, screen, (endpoint[0], "y_mean"), cell, "mean", r)
                    request_r(res, screen, endpoint, cell, "rebuilt", r, kind="linear", columns=("B0", "B1", "B2"))
                    request_r(res, screen, endpoint, cell, "rebuilt", r, kind="linear", columns=e1cells.TRIVIAL)
    print("[stats] residualising", flush=True)
    res.run()
    STATE["scorer"], STATE["res"] = scorer, res
    return {"objects": len(plan)}


def get_rho(cell, screen, context, label, representation, head="forest", variant=ALL_ROWS, realisation=0, column="y", mask="all", basis=None):
    scorer = STATE["scorer"]
    source = DATA["cells"][cell].external_source[screen]
    identifier = mask_id(cell, screen, mask)
    if identifier is None:
        return None
    return object_stat(scorer, screen, (source, column), identifier, context, label, representation, head, variant, realisation, basis=basis,
                       seeds=SEEDS if head in ("forest", "boosting") else (0,))


def contrast(a, b):
    if a is None or b is None:
        return None
    return a[0] - b[0], a[1] - b[1]


def t_stat(cell, screen, arm, label="rebuilt", context=None, **kw):
    context = context or cell
    return contrast(get_rho(cell, screen, context, label, arm, **kw), get_rho(cell, screen, context, label, "C1", **kw))


def l_stat(cell, screen, arm, rebuilt_label="rebuilt", **kw):
    existing_context = e1cells.CELLS[cell].get("base", cell)
    return contrast(t_stat(cell, screen, arm, rebuilt_label, **kw), t_stat(cell, screen, arm, "existing", context=existing_context, **kw))


def r_value(cell, screen, arm, label="rebuilt", fold_seed=FOLD_SEEDS[0], kind="boosting", columns=e1cells.TRIVIAL, column="y", subset=None):
    endpoint = (DATA["cells"][cell].external_source[screen], column)
    mask = None if subset is None else DATA["masks"][(screen, mask_key(cell, screen, subset), subset)]
    return r_stat(STATE["res"], screen, endpoint, cell, label, arm, fold_seed, kind, columns, mask=mask)


def primary_family() -> tuple[pd.DataFrame, dict]:
    rows = []
    for cell in e1cells.PRIMARY:
        for screen in SCREENS:
            for arm in FAMILY:
                for name, value in (("T", t_stat(cell, screen, arm)), ("L", l_stat(cell, screen, arm)), ("R", r_value(cell, screen, arm))):
                    rows.append({"cell": cell, "screen": screen, "statistic": name, "arm": arm, **summarise(value)})
            for arm in MASKED:
                rows.append({"cell": cell, "screen": screen, "statistic": "R-", "arm": arm, **summarise(r_value(cell, screen, arm))})
    frame = pd.DataFrame(rows)
    rejected, thresholds = resample.holm(frame["p"].to_numpy(), ALPHA)
    frame["holm_threshold"] = thresholds
    frame["rejected"] = rejected
    frame["state"] = [resample.holm_state(e, r) for e, r in zip(frame["estimate"], rejected)]
    gate = r5_gate()
    frame["r5_gate_passed"] = [gate[(c, s)] for c, s in zip(frame["cell"], frame["screen"])]
    frame["read"] = [not (st in ("R", "R-") and not g) for st, g in zip(frame["statistic"], frame["r5_gate_passed"])]

    # robustness: label mode, mean label, fold assignment, head fitting
    def state_at(estimate, p, threshold):
        return resample.holm_state(estimate, bool(np.isfinite(p) and p <= threshold))

    source_state, mean_state, fold_state, refit_share = [], [], [], []
    for row in frame.itertuples():
        kw = {}
        if row.statistic == "T":
            s = t_stat(row.cell, row.screen, row.arm, "source")
            m = t_stat(row.cell, row.screen, row.arm, "mean", column="y_mean")
        elif row.statistic == "L":
            s = l_stat(row.cell, row.screen, row.arm, "source")
            m = contrast(t_stat(row.cell, row.screen, row.arm, "mean", column="y_mean"), t_stat(row.cell, row.screen, row.arm, "existing", column="y_mean"))
        else:
            s = r_value(row.cell, row.screen, row.arm, "source")
            m = r_value(row.cell, row.screen, row.arm, "mean", column="y_mean")
        source_state.append(state_at(s[0], resample.p_value(s[1]), row.holm_threshold) if s is not None else "not fitted")
        mean_state.append(state_at(m[0], resample.p_value(m[1]), row.holm_threshold) if m is not None else "not fitted")
        if row.statistic in ("R", "R-"):
            states = []
            for seed in FOLD_SEEDS[1:]:
                v = r_value(row.cell, row.screen, row.arm, fold_seed=seed)
                states.append(state_at(v[0], resample.p_value(v[1]), row.holm_threshold) if v is not None else "not fitted")
            fold_state.append(";".join(states))
            refit_share.append(float("nan"))
        else:
            fold_state.append("")
            refit_share.append(refit_sign_share(row.cell, row.screen, row.arm, row.estimate) if row.statistic == "T" else float("nan"))
        _ = kw
    frame["state_source_mode_labels"] = source_state
    frame["state_mean_label"] = mean_state
    frame["states_other_fold_assignments"] = fold_state
    frame["share_of_refits_with_the_sign"] = refit_share
    frame["not_robust_to_label_mode"] = [r.state != "inconclusive" and r.state != s for r, s in zip(frame.itertuples(), source_state)]
    frame["not_robust_to_fold_assignment"] = [r.state != "inconclusive" and fs != "" and any(x != r.state for x in fs.split(";")) for r, fs in zip(frame.itertuples(), fold_state)]
    frame["not_robust_to_head_fitting"] = [r.state != "inconclusive" and np.isfinite(sh) and sh < 0.8 for r, sh in zip(frame.itertuples(), refit_share)]
    frame["family"] = "primary, 60 statistics, Holm at family-wise 0.05 (with the second family, 0.10 together)"
    return frame, gate


def in_support_family(family: pd.DataFrame) -> pd.DataFrame:
    """The 60 statistics on the in-support subset (Section 29, item 4): secondary and unadjusted."""
    rows = []
    for row in family.itertuples():
        if row.statistic == "T":
            value = t_stat(row.cell, row.screen, row.arm, mask=IN_SUPPORT)
        elif row.statistic == "L":
            value = l_stat(row.cell, row.screen, row.arm, mask=IN_SUPPORT)
        else:
            value = r_value(row.cell, row.screen, row.arm, subset=IN_SUPPORT)
        share = DATA["support_share"][(row.cell, row.screen)]
        rows.append({"cell": row.cell, "screen": row.screen, "statistic": row.statistic, "arm": row.arm, "subset": IN_SUPPORT,
                     **summarise(value), "constructs_in_subset": share["inside"], "share_of_constructs": share["share_inside"],
                     "vcc_training_b0_range": "%.3f to %.3f" % (share["vcc_training_b0_min"], share["vcc_training_b0_max"]),
                     "all_constructs_estimate": row.estimate, "all_constructs_state": row.state,
                     "adjustment": "secondary, unadjusted; support table bioinf2027/results/E1/support.csv"})
    return pd.DataFrame(rows)


def refit_values(cell, screen, arm) -> np.ndarray:
    out = []
    for r in range(DATA["refits"]):
        a = get_rho(cell, screen, cell, "rebuilt", arm, variant=f"refit {r}")
        c = get_rho(cell, screen, cell, "rebuilt", "C1", variant=f"refit {r}")
        if a is not None and c is not None:
            out.append(a[0] - c[0])
    return np.array(out)


def refit_sign_share(cell, screen, arm, estimate) -> float:
    values = refit_values(cell, screen, arm)
    if values.size == 0 or not np.isfinite(estimate) or estimate == 0:
        return float("nan")
    return float((np.sign(values) == np.sign(estimate)).mean())


def r5_gate() -> dict:
    """Passed for a cell on a screen when both the screen's and the VCC training rows' gates pass (Section 9, R5)."""
    gate = DATA["r5_decisions"]
    out = {}
    for cell in list(e1cells.PRIMARY) + ["G1", "G2"]:
        for screen in SCREENS:
            out[(cell, screen)] = bool(gate[f"{cell} | VCC"]["passed"] and gate[f"{cell} | {screen}"]["passed"])
    return out


def reduces_to(frame: pd.DataFrame, gate: dict) -> pd.DataFrame:
    rows = []
    state = {(r.cell, r.screen, r.statistic, r.arm): r.state for r in frame.itertuples()}
    for screen in SCREENS:
        for arm in FAMILY:
            t_pos = all(state.get((c, screen, "T", arm)) == "Holm-positive" for c in e1cells.PRIMARY)
            r_not_pos = all(state.get((c, screen, "R", arm)) != "Holm-positive" for c in e1cells.PRIMARY)
            gate_ok = all(gate[(c, screen)] for c in e1cells.PRIMARY)
            considered = t_pos and r_not_pos and gate_ok
            p = 1.0
            components = {}
            if considered:
                ps = []
                for cell in e1cells.PRIMARY:
                    r = r_value(cell, screen, arm)
                    inc = contrast(get_rho(cell, screen, cell, "rebuilt", f"[TR; {arm}]"), get_rho(cell, screen, cell, "rebuilt", "TR"))
                    if r is None or inc is None:
                        ps.append(1.0)
                        continue
                    pr = (1 + int((r[1] >= MARGIN_R).sum())) / (B + 1)
                    pi = (1 + int((inc[1] >= MARGIN_INCREMENT).sum())) / (B + 1)
                    components[cell] = {"p_R_at_least_0.10": pr, "p_increment_at_least_0.03": pi}
                    ps += [pr, pi]
                p = max(ps)
            rows.append({"screen": screen, "arm": arm, "considered": considered,
                         "why_not_considered": "" if considered else ("the R5 gate fails" if not gate_ok else "T not Holm-positive in both cells or R Holm-positive"),
                         "p": p, "components": json.dumps(components)})
    out = pd.DataFrame(rows)
    rejected, thresholds = resample.holm(out["p"].to_numpy(), ALPHA)
    out["holm_threshold"] = thresholds
    out["rejected"] = rejected
    out["statement"] = ["reduces to the specified trivial predictors" if r else "not made" for r in rejected]
    out["family"] = "second family, 8 statements, Holm at family-wise 0.05 (0.10 together with the primary family)"
    return out


# --------------------------------------------------------------------------- secondary tables


def precondition_rows() -> pd.DataFrame:
    rows = []
    for cell in e1cells.CELLS:
        contexts = [(cell, "rebuilt")]
        if cell in e1cells.PRIMARY or "base" not in e1cells.CELLS[cell]:
            contexts.append((cell, "existing"))
        for context, label in contexts:
            c1 = vcc_r2(context, label, "C1")
            for arm in FAMILY + MASKED + ("C2", "C3", "TR"):
                a = vcc_r2(context, label, arm)
                if a is None or c1 is None:
                    continue
                d = contrast(a["r2"], c1["r2"])
                ds = contrast(a["spearman"], c1["spearman"])
                entry = {"cell": cell, "label": label, "arm": arm, "n_test_rows": a["n_rows"], "n_test_genes": a["n_genes"],
                         "r2_arm": a["r2"][0], "r2_C1": c1["r2"][0], "spearman_arm": a["spearman"][0],
                         **{f"r2_difference_{k}": v for k, v in summary(*d).items()},
                         "spearman_difference": ds[0], "spearman_difference_lo": resample_interval(ds[1])[0], "spearman_difference_hi": resample_interval(ds[1])[1],
                         "passes": bool(resample_interval(d[1])[0] > 0), "descriptive": bool(a["n_genes"] < DESCRIPTIVE_GENES),
                         "adjustment": "unadjusted"}
                if e1cells.CELLS[cell]["unit"] == "row" and label == "rebuilt":
                    ab = vcc_r2(context, label, arm, kind="batches")
                    cb = vcc_r2(context, label, "C1", kind="batches")
                    db = contrast(ab["r2"], cb["r2"])
                    lo, hi, _ = resample_interval(db[1])
                    entry.update({"batch_resampled_lo": lo, "batch_resampled_hi": hi})
                rows.append(entry)
    return pd.DataFrame(rows)


def vcc_test_rows() -> pd.DataFrame:
    """VCC test R-squared and Spearman, row level and target-gene level, every representation and head."""
    rows = []
    for cell in e1cells.CELLS:
        labels = [("rebuilt", cell)] + ([("existing", cell)] if (cell in e1cells.PRIMARY or "base" not in e1cells.CELLS[cell]) else [])
        for label, context in labels:
            for r in transfer.REPRESENTATIONS:
                for head in (("monotone",) if r == "label positive control" else ("forest", "boosting", "ridge")):
                    v = vcc_r2(context, label, r, head)
                    if v is None or r == "B2 feature space":
                        continue
                    entry = {"cell": cell, "label": label, "representation": r, "head": head, "n_test_rows": v["n_rows"], "n_test_genes": v["n_genes"],
                             "descriptive": bool(v["n_genes"] < DESCRIPTIVE_GENES)}
                    for name in ("r2", "spearman"):
                        s = summary(*v[name])
                        entry.update({f"row_{name}": s["estimate"], f"row_{name}_lo": s["lo"], f"row_{name}_hi": s["hi"]})
                    if head == "forest":
                        g = vcc_gene_level(context, label, r, head)
                        for name in ("r2", "spearman"):
                            s = summary(*g[name])
                            entry.update({f"gene_{name}": s["estimate"], f"gene_{name}_lo": s["lo"], f"gene_{name}_hi": s["hi"]})
                    rows.append(entry)
    return pd.DataFrame(rows)


def transfer_rows() -> pd.DataFrame:
    rows = []
    subsets = ("all", "target is a feature", "target is not a feature", "target not a VCC training or validation target", "further than 0.1 below the ceiling", IN_SUPPORT)
    for cell in e1cells.CELLS:
        for screen in DATA["cells"][cell].spec["screens"]:
            for label, context in (("rebuilt", cell), ("existing", e1cells.CELLS[cell].get("base", cell))):
                for r in transfer.REPRESENTATIONS:
                    for head in (("monotone",) if r == "label positive control" else ("forest", "boosting", "ridge")):
                        for mask in (subsets if head == "forest" else ("all", IN_SUPPORT)):
                            v = get_rho(cell, screen, context, label, r, head, mask=mask)
                            if v is None:
                                continue
                            c1 = get_rho(cell, screen, context, label, "C1", head if head != "monotone" else "forest", mask=mask)
                            entry = {"cell": cell, "screen": screen, "label": label, "representation": r, "head": head, "subset": mask,
                                     **{f"spearman_{k}": x for k, x in summary(v[0], v[1]).items() if k in ("estimate", "lo", "hi")}}
                            if c1 is not None and r != "C1":
                                for k, x in summary(*contrast(v, c1)).items():
                                    entry[f"minus_C1_{k}"] = x
                            entry["role"] = "family" if (cell in e1cells.PRIMARY and r in FAMILY + MASKED + ("C1",) and head == "forest" and mask == "all") else "secondary, unadjusted"
                            if head == "forest" and mask == "all":
                                rb = get_rho(cell, screen, context, label, r, head, basis="release")
                                if rb is not None:
                                    entry["release_basis_spearman"] = rb[0]
                            rows.append(entry)
    return pd.DataFrame(rows)


def secondary_contrast_rows() -> pd.DataFrame:
    rows = []

    def add(cell, screen, name, value, **extra):
        if value is None:
            return
        rows.append({"cell": cell, "screen": screen, "contrast": name, **summary(*value), "adjustment": "unadjusted", **extra})

    for cell in e1cells.CELLS:
        for screen in DATA["cells"][cell].spec["screens"]:
            source = DATA["cells"][cell].external_source[screen]
            for arm in FAMILY:
                a = get_rho(cell, screen, cell, "rebuilt", arm)
                for raw in ("B1", "B2"):
                    add(cell, screen, f"{arm} minus raw {raw}", contrast(a, raw_baseline(cell, screen, raw)))
                add(cell, screen, f"{arm} minus the B1 head", contrast(a, get_rho(cell, screen, cell, "rebuilt", "B1")))
                add(cell, screen, f"{arm} minus TR", contrast(a, get_rho(cell, screen, cell, "rebuilt", "TR")))
                add(cell, screen, f"[TR; {arm}] minus TR (never read as beyond the trivial predictors)",
                    contrast(get_rho(cell, screen, cell, "rebuilt", f"[TR; {arm}]"), get_rho(cell, screen, cell, "rebuilt", "TR")))
                if cell not in e1cells.PRIMARY:
                    add(cell, screen, f"T of {arm}", t_stat(cell, screen, arm))
                    add(cell, screen, f"L of {arm}", l_stat(cell, screen, arm))
                if (screen, cell, "further than 0.1 below the ceiling") in DATA["masks"]:
                    add(cell, screen, f"T of {arm}, constructs further than 0.1 below their ceiling", t_stat(cell, screen, arm, mask="further than 0.1 below the ceiling"),
                        share_at_ceiling=DATA["ceiling_share"][(cell, screen)])
                    add(cell, screen, f"L of {arm}, constructs further than 0.1 below their ceiling", l_stat(cell, screen, arm, mask="further than 0.1 below the ceiling"),
                        share_at_ceiling=DATA["ceiling_share"][(cell, screen)])
                for mask in ("target is a feature", "target is not a feature", "target not a VCC training or validation target", IN_SUPPORT):
                    add(cell, screen, f"T of {arm}, {mask}", t_stat(cell, screen, arm, mask=mask))
                add(cell, screen, f"L of {arm}, {IN_SUPPORT}", l_stat(cell, screen, arm, mask=IN_SUPPORT))
                add(cell, screen, f"{arm} minus raw B1, {IN_SUPPORT}", contrast(get_rho(cell, screen, cell, "rebuilt", arm, mask=IN_SUPPORT), raw_baseline(cell, screen, "B1", mask=IN_SUPPORT)))
                add(cell, screen, f"{arm} minus TR, {IN_SUPPORT}", contrast(get_rho(cell, screen, cell, "rebuilt", arm, mask=IN_SUPPORT), get_rho(cell, screen, cell, "rebuilt", "TR", mask=IN_SUPPORT)))
            for masked, arm in UNMASKED_OF.items():
                for mask in ("all", "target is a feature", "target is not a feature"):
                    add(cell, screen, f"{arm} minus {masked}, {mask}", contrast(get_rho(cell, screen, cell, "rebuilt", arm, mask=mask), get_rho(cell, screen, cell, "rebuilt", masked, mask=mask)))
            for other in ("x", "C2", "[x; m]"):
                add(cell, screen, f"z minus {other}", contrast(get_rho(cell, screen, cell, "rebuilt", "z"), get_rho(cell, screen, cell, "rebuilt", other)))
            if cell in e1cells.PRIMARY:
                for arm in FAMILY + MASKED:
                    for kind, columns, name in (("linear", ("B0", "B1", "B2"), "linear rank-residual partial on B0 to B2"), ("linear", e1cells.TRIVIAL, "linear rank-residual partial on B0 to B4")):
                        add(cell, screen, f"{name}, {arm} (never read as beyond the trivial predictors)", r_value(cell, screen, arm, kind=kind, columns=columns))
                    add(cell, screen, f"R of {arm}, source-mode labels", r_value(cell, screen, arm, "source"))
                    add(cell, screen, f"R of {arm}, mean-across-draws label", r_value(cell, screen, arm, "mean", column="y_mean"))
                    for seed in FOLD_SEEDS[1:]:
                        add(cell, screen, f"R of {arm}, fold seed {seed}", r_value(cell, screen, arm, fold_seed=seed))
            if cell not in e1cells.PRIMARY and "base" not in e1cells.CELLS[cell]:
                for arm in FAMILY + MASKED:
                    add(cell, screen, f"R of {arm}", r_value(cell, screen, arm))
            _ = source
        if cell in ("G1", "G2"):
            bl = {"G1": "G5", "G2": "G6"}[cell]
            for screen in SCREENS:
                for arm in FAMILY:
                    rebuilt_bl = t_stat(cell, screen, arm, f"rebuilt:{cell}", context=bl)
                    existing_bl = t_stat(cell, screen, arm, "existing", context=bl)
                    add(cell, screen, f"T of {arm}, rebuilt label, batch-local features (2 x 2)", rebuilt_bl)
                    add(cell, screen, f"T of {arm}, existing label, batch-local features (2 x 2)", existing_bl)
                    add(cell, screen, f"L of {arm}, batch-local features (2 x 2)", contrast(rebuilt_bl, existing_bl))
    return pd.DataFrame(rows)


def raw_baseline(cell: str, screen: str, name: str, mask: str | None = None):
    """The screen's own value of a trivial predictor against its label (no fitting), on all constructs or a subset."""
    table = DATA["external"][(screen, DATA["cells"][cell].external_source[screen])]
    values = table[name].to_numpy(dtype=np.float64)
    label = table["y"].to_numpy(dtype=np.float64)
    draws = STATE["draws"][screen]
    if name == "B2 feature space":
        raise ValueError(name)
    subset = None if mask is None else DATA["masks"][(screen, mask_key(cell, screen, mask), mask)]
    point = resample.weighted_spearman(values, label, draws["point"], mask=subset)[0, 0]
    replicates = resample.weighted_spearman(values, label, draws["draw"], mask=subset)[:, 0]
    return float(point), replicates


def baseline_rows() -> pd.DataFrame:
    rows = []
    for cell in e1cells.CELLS:
        for screen in DATA["cells"][cell].spec["screens"]:
            for name in ("B0", "B1", "B2", "B3", "B4"):
                v = raw_baseline(cell, screen, name)
                rows.append({"cell": cell, "screen": screen, "baseline": name, "kind": "raw, the screen's own value", "subset": "all", **summary(*v)})
                v = raw_baseline(cell, screen, name, mask=IN_SUPPORT)
                rows.append({"cell": cell, "screen": screen, "baseline": name, "kind": "raw, the screen's own value", "subset": IN_SUPPORT, **summary(*v)})
            for name in transfer.ONE_VARIABLE + ("TR",):
                v = get_rho(cell, screen, cell, "rebuilt", name)
                if v is not None:
                    rows.append({"cell": cell, "screen": screen, "baseline": name, "kind": "forest fitted on VCC training rows, seeds 0 to 2", **summary(*v)})
    return pd.DataFrame(rows)


def positive_control_rows() -> pd.DataFrame:
    rows = []
    for cell in e1cells.CELLS:
        for screen in DATA["cells"][cell].spec["screens"]:
            table = DATA["external"][(screen, DATA["cells"][cell].external_source[screen])]
            label = table["y"].to_numpy(dtype=np.float64)
            fit = DATA["predictions"].get(cell, "rebuilt", "label positive control", "monotone", 0)
            if fit is not None:
                v = get_rho(cell, screen, cell, "rebuilt", "label positive control", "monotone")
                prediction = fit[screen]
                slope, intercept, r_value_, _, _ = sps.linregress(prediction, label)
                rows.append({"cell": cell, "screen": screen, "control": "label construction (monotone in B2 native)", "subset": "all", **summary(*v),
                             "calibration_slope": slope, "calibration_intercept": intercept, "calibration_r2": r_value_ ** 2,
                             "below_0.80": bool(v[0] < POSITIVE_LIMIT), "expected": "at least +0.90"})
            for name in ("[C1; B2]", "[z; B2]"):
                v = get_rho(cell, screen, cell, "rebuilt", name)
                if v is not None:
                    rows.append({"cell": cell, "screen": screen, "control": f"pipeline {name}", "subset": "all", **summary(*v), "below_0.80": bool(v[0] < POSITIVE_LIMIT)})
                v = get_rho(cell, screen, cell, "rebuilt", name, mask=IN_SUPPORT)
                if v is not None:
                    rows.append({"cell": cell, "screen": screen, "control": f"pipeline {name}", "subset": IN_SUPPORT, **summary(*v),
                                 "below_0.80": bool(v[0] < POSITIVE_LIMIT), "note": "secondary; validity is read on all constructs"})
            v = get_rho(cell, screen, cell, "rebuilt", "label positive control", "monotone", mask=IN_SUPPORT)
            if v is not None:
                rows.append({"cell": cell, "screen": screen, "control": "label construction (monotone in B2 native)", "subset": IN_SUPPORT, **summary(*v),
                             "below_0.80": bool(v[0] < POSITIVE_LIMIT), "note": "secondary; validity is read on all constructs"})
    return pd.DataFrame(rows)


def null_band_rows() -> pd.DataFrame:
    rows = []
    for cell in e1cells.PRIMARY:
        for screen in SCREENS:
            for control, count in (("C1", 100), ("C3", 100), ("C2", 20)):
                values = np.array([v for v in (single_seed_rho(cell, screen, control, k) for k in range(count)) if v is not None], dtype=float)
                if values.size == 0:
                    continue
                entry = {"cell": cell, "screen": screen, "control": control, "realisations": int(values.size), "head_seed": 0}
                if control == "C2":
                    entry.update({"min": float(values.min()), "max": float(values.max()), "reading": "range of a random projection, not a null"})
                else:
                    entry.update({"p2.5": float(np.percentile(values, 2.5)), "p97.5": float(np.percentile(values, 97.5)), "reading": "null band"})
                for arm in FAMILY:
                    a = get_rho(cell, screen, cell, "rebuilt", arm)
                    if a is not None:
                        entry[f"{arm}_external_spearman"] = a[0]
                rows.append(entry)
    return pd.DataFrame(rows)


def single_seed_rho(cell, screen, representation, realisation):
    """External Spearman of head seed 0 of one realisation; None if not fitted."""
    scorer = STATE["scorer"]
    source = DATA["cells"][cell].external_source[screen]
    key = (screen, (source, "y"), "all", object_key(cell, "rebuilt", representation, "forest", ALL_ROWS, realisation) + "|0|None")
    if key not in scorer.results:
        return None
    return scorer.results[key][0]


def refit_rows(family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in e1cells.PRIMARY:
        for screen in SCREENS:
            for arm in FAMILY:
                values = refit_values(cell, screen, arm)
                estimate = family[(family.cell == cell) & (family.screen == screen) & (family.statistic == "T") & (family.arm == arm)]["estimate"].iloc[0]
                if values.size:
                    rows.append({"cell": cell, "screen": screen, "arm": arm, "refits": int(values.size), "T": estimate,
                                 "p2.5": float(np.percentile(values, 2.5)), "p97.5": float(np.percentile(values, 97.5)),
                                 "share_with_the_sign_of_T": float((np.sign(values) == np.sign(estimate)).mean())})
    return pd.DataFrame(rows)


def composition_rows() -> pd.DataFrame:
    rows = []
    for cell in e1cells.PRIMARY:
        for screen in SCREENS:
            for arm in FAMILY:
                for variant in ("target is a feature", "target is not a feature", "single-guide rows"):
                    v = t_stat(cell, screen, arm, variant=variant)
                    if v is not None:
                        rows.append({"cell": cell, "screen": screen, "arm": arm, "training_rows": variant, "statistic": "T", **summary(*v), "adjustment": "unadjusted"})
                v = contrast(t_stat(cell, screen, arm, variant="single-guide rows"), t_stat(cell, screen, arm, "existing", variant="single-guide rows"))
                if v is not None:
                    rows.append({"cell": cell, "screen": screen, "arm": arm, "training_rows": "single-guide rows", "statistic": "L", **summary(*v), "adjustment": "unadjusted"})
    return pd.DataFrame(rows)


def secondary_head_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ridge = [], []
    for cell in e1cells.CELLS:
        for screen in DATA["cells"][cell].spec["screens"]:
            for head in ("boosting", "ridge"):
                for arm in FAMILY + MASKED:
                    v = t_stat(cell, screen, arm, head=head)
                    if v is not None:
                        rows.append({"cell": cell, "screen": screen, "head": head, "arm": arm, "statistic": "T", **summary(*v),
                                     "note": "descriptive" if (cell == "G5" and head == "ridge") else "secondary, unadjusted"})
        for label in ("rebuilt", "existing"):
            for r in transfer.REPRESENTATIONS:
                fit = DATA["predictions"].get(cell, label, r, "ridge", 0)
                if fit is not None:
                    choice = fit["_record"]["ridge"]
                    ridge.append({"cell": cell, "label": label, "representation": r, "chosen_penalty": choice["chosen"], "at_grid_edge": choice["at_grid_edge"],
                                  **{f"validation_r2_at_{k}": v for k, v in choice["validation_r2"].items()}})
    return pd.DataFrame(rows), pd.DataFrame(ridge)


def bridge_rows() -> pd.DataFrame:
    """The steps from the submitted configuration to each cell's rebuilt label (Section 21.1, item 4)."""
    stored = {screen: DATA["external"][(screen, "G1")]["stored_y_primary"].to_numpy(dtype=np.float64) for screen in SCREENS}
    rows = []
    draws = STATE["draws"]
    phase4 = REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz"
    predictions = np.load(phase4, allow_pickle=True)
    arms = list(np.asarray(predictions["arms"]).astype(str))
    short = {"RPE1": "RPE1", "K562-essential": "K562"}

    def stored_rho(arm, screen):
        """The submitted value: Spearman of the stored seed-mean prediction (Section 29, item 2)."""
        constructs = np.asarray(predictions[f"{short[screen]}_constructs"]).astype(str)
        if not np.array_equal(constructs, DATA["external"][(screen, "G1")]["construct"].astype(str).to_numpy()):
            raise RuntimeError("the stored %s predictions are not in construct order" % screen)
        column = np.asarray(predictions[f"{short[screen]}_P"])[:, arms.index(arm)]
        point = resample.weighted_spearman(column, stored[screen], draws[screen]["point"])[0, 0]
        replicates = resample.weighted_spearman(column, stored[screen], draws[screen]["draw"])[:, 0]
        return float(point), replicates

    steps = {}
    for screen in SCREENS:
        for arm in ("z", "C1", "C2", "C3"):
            steps[(screen, arm, 0)] = stored_rho(arm, screen)
            # the external x arrays are equal (checks.json), so step 1 changes no value
            steps[(screen, arm, 1)] = steps[(screen, arm, 0)]
    for cell, (all_rows, own_rows, own_reference) in {"G5": ("G6", "G5", "G5"), "G6": ("G6", "G6", "G6"), "G1": ("G6", "G5", "G1"), "G2": ("G6", "G6", "G2")}.items():
        for screen in SCREENS:
            for arm in FAMILY + ("C1", "C2", "C3"):
                sequence = []
                if arm in ("z", "C1", "C2", "C3"):
                    sequence.append(("0 submitted configuration (stored phase-4 predictions)", steps[(screen, arm, 0)]))
                    sequence.append(("1 external x array (the arrays are equal; no change)", steps[(screen, arm, 1)]))
                sequence.append(("2 primary forest for every arm, all modelled rows", get_rho("G1", screen, all_rows, "existing", arm, column="stored_y_primary")))
                sequence.append(("3 the cell's rows", get_rho("G1", screen, own_rows, "existing", arm, column="stored_y_primary")))
                sequence.append(("4 the cell's feature reference", get_rho("G1", screen, own_reference, "existing", arm, column="stored_y_primary")))
                sequence.append(("5 the cell's external endpoint", get_rho(cell, screen, own_reference, "existing", arm)))
                sequence.append(("6 the rebuilt label", get_rho(cell, screen, cell, "rebuilt", arm)))
                previous = None
                for name, value in sequence:
                    if value is None:
                        continue
                    entry = {"cell": cell, "screen": screen, "arm": arm, "step": name, **{f"spearman_{k}": v for k, v in summary(value[0], value[1]).items() if k in ("estimate", "lo", "hi")}}
                    if previous is not None:
                        d = contrast(value, previous)
                        entry.update({f"change_{k}": v for k, v in summary(*d).items() if k in ("estimate", "lo", "hi")})
                    rows.append(entry)
                    previous = value
    return pd.DataFrame(rows)


def oracle_rows() -> pd.DataFrame:
    """External constructs whose target is a VCC target: the VCC gene-level label and control mean as predictors."""
    rows = []
    for cell, gene_cell in (("G5", "G3"), ("G6", "G4"), ("G1", "G3"), ("G2", "G4")):
        gene_table = DATA["cells"][gene_cell].vcc
        by_ensembl = {}
        for row in gene_table.itertuples():
            ens = DATA["vcc_ensembl"].get(str(row.target_gene))
            if ens is not None:
                by_ensembl.setdefault(ens, []).append((row.y, row.B1))
        for screen in SCREENS:
            table = DATA["external"][(screen, DATA["cells"][cell].external_source[screen])]
            ens = table["target_ensg"].astype(str).to_numpy()
            mask = np.array([e in by_ensembl for e in ens])
            if mask.sum() < 3:
                continue
            label = table["y"].to_numpy(dtype=np.float64)
            for name, j in (("VCC gene-level rebuilt label", 0), ("VCC control mean of the gene", 1)):
                predictor = np.array([np.mean([v[j] for v in by_ensembl[e]]) if e in by_ensembl else 0.0 for e in ens])
                draws = STATE["draws"][screen]
                point = resample.weighted_spearman(predictor, label, draws["point"], mask=mask)[0, 0]
                replicates = resample.weighted_spearman(predictor, label, draws["draw"], mask=mask)[:, 0]
                rows.append({"cell": cell, "screen": screen, "predictor": name, "gene_level_cell": gene_cell, "constructs": int(mask.sum()),
                             **summary(float(point), replicates), "reading": "descriptive; bounds what gene identity and baseline expression reach without a representation"})
    return pd.DataFrame(rows)


def reading_rows(family: pd.DataFrame, positive: pd.DataFrame, preconditions: pd.DataFrame) -> pd.DataFrame:
    """Section 11, read in order: validity, agreement, the table per arm, and L."""
    rows = []
    state = {(r.cell, r.screen, r.statistic, r.arm): r.state for r in family.itertuples()}
    gate = {(r.cell, r.screen): r.r5_gate_passed for r in family.itertuples()}
    for screen in SCREENS:
        excluded = []
        for cell in e1cells.PRIMARY:
            pc = positive[(positive.cell == cell) & (positive.screen == screen) & positive.control.str.startswith("label construction") & (positive.subset == "all")]
            if not pc.empty and bool(pc["below_0.80"].iloc[0]):
                excluded.append(cell)
        for arm in FAMILY:
            pre = preconditions[(preconditions.cell.isin(e1cells.PRIMARY)) & (preconditions.label == "rebuilt") & (preconditions.arm == arm)]
            precondition_ok = bool(len(pre) == 2 and pre["passes"].all())
            t = [state[(c, screen, "T", arm)] for c in e1cells.PRIMARY]
            r = [state[(c, screen, "R", arm)] for c in e1cells.PRIMARY]
            l_states = [state[(c, screen, "L", arm)] for c in e1cells.PRIMARY]
            answerable = all(gate[(c, screen)] for c in e1cells.PRIMARY)
            head_limited = []
            for c in e1cells.PRIMARY:
                pp = positive[(positive.cell == c) & (positive.screen == screen) & (positive.control == "pipeline [C1; B2]") & (positive.subset == "all")]
                head_limited.append(bool(not pp.empty and pp["below_0.80"].iloc[0]))
            if excluded:
                statement, contribution = "a cell of the primary pair is excluded on this screen (label positive control below +0.80)", "undetermined"
            elif not precondition_ok:
                statement, contribution = "uninformative about transfer (fails the precondition in a cell of the pair)", "undetermined"
            elif t[0] != t[1]:
                statement, contribution = "undetermined: the two cells of the pair differ", "undetermined"
            elif t[0] == "Holm-negative":
                if arm == "z" and any(head_limited):
                    statement, contribution = "read as limited by the head ([C1; B2 native] below +0.80)", "undetermined"
                else:
                    statement, contribution = "transfer fails after label matching", "toward branch A"
            elif t[0] == "inconclusive":
                statement, contribution = "neither a transfer nor a failure is resolved", "undetermined"
            elif not answerable:
                statement, contribution = "transfer resolved; whether it goes beyond the trivial predictors is not answerable on this endpoint", "neither branch"
            elif r[0] != r[1]:
                statement, contribution = "transfer resolved; beyond the trivial predictors undetermined", "neither branch"
            elif r[0] == "Holm-positive":
                statement, contribution = "information beyond the specified trivial predictors (qualifiers of Section 11 apply)", "toward branch B"
            elif r[0] == "Holm-negative":
                statement, contribution = "transfer resolved; the prediction runs against the label once the trivial predictors are removed", "neither branch"
            else:
                statement, contribution = "transfer resolved; not shown to exceed the specified trivial predictors", "neither branch"
            if l_states[0] != l_states[1]:
                l_reading = "undetermined"
            elif l_states[0] == "Holm-positive":
                l_reading = "label matching raises the arm's contrast over C1"
            elif l_states[0] == "Holm-negative":
                l_reading = "label matching lowers the arm's contrast over C1"
            else:
                l_reading = "no change from label matching is resolved" + ("; the failure is not measurably changed by label matching" if t[0] in ("Holm-negative", "inconclusive") else "")
            rows.append({"screen": screen, "arm": arm, "T_states": "; ".join(t), "R_states": "; ".join(r), "L_states": "; ".join(l_states),
                         "precondition_passed_in_both_cells": precondition_ok, "excluded_cells": ";".join(excluded),
                         "r5_gate_passed_in_both_cells": answerable, "statement": statement, "contribution": contribution, "L_reading": l_reading})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main


def write(frame: pd.DataFrame, name: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)
    return file_record(OUT / name)


def main() -> None:
    started = time.time()
    manifest_path = (config.storage_root() / "bioinf2027" / "scratch" / "E1b_fits_smoke" / "predictions_manifest.json") if SMOKE else (OUT / "predictions_manifest.json")
    DATA["predictions"] = Predictions(manifest_path)
    protocol = json.loads((manifest_path.parent / "protocol.json").read_text()) if SMOKE else json.loads((OUT / "protocol.json").read_text())
    DATA["refits"] = 2 if SMOKE else int(protocol["refits"])
    inputs = load_inputs()
    build_masks()
    STATE["draws"] = build_draws()
    STATE["vcc_draws"] = {}
    info = family_and_secondary(inputs)
    family, gate = primary_family()
    outputs = {"primary_family.csv": write(family, "primary_family.csv")}
    outputs["reduces_to_family.csv"] = write(reduces_to(family, gate), "reduces_to_family.csv")
    outputs["primary_family_in_support.csv"] = write(in_support_family(family), "primary_family_in_support.csv")
    preconditions = precondition_rows()
    outputs["preconditions.csv"] = write(preconditions, "preconditions.csv")
    positive = positive_control_rows()
    outputs["positive_controls.csv"] = write(positive, "positive_controls.csv")
    outputs["readings.csv"] = write(reading_rows(family, positive, preconditions), "readings.csv")
    outputs["transfer.csv"] = write(transfer_rows(), "transfer.csv")
    outputs["vcc_test.csv"] = write(vcc_test_rows(), "vcc_test.csv")
    outputs["secondary_contrasts.csv"] = write(secondary_contrast_rows(), "secondary_contrasts.csv")
    outputs["baselines.csv"] = write(baseline_rows(), "baselines.csv")
    outputs["null_band.csv"] = write(null_band_rows(), "null_band.csv")
    outputs["refits.csv"] = write(refit_rows(family), "refits.csv")
    outputs["composition.csv"] = write(composition_rows(), "composition.csv")
    heads_frame, ridge_frame = secondary_head_rows()
    outputs["secondary_heads.csv"] = write(heads_frame, "secondary_heads.csv")
    outputs["ridge_penalties.csv"] = write(ridge_frame, "ridge_penalties.csv")
    outputs["bridge.csv"] = write(bridge_rows(), "bridge.csv")
    outputs["oracle.csv"] = write(oracle_rows(), "oracle.csv")
    stats = {
        "plan": PLAN,
        "settings": {"replicates": B, "external_seeds": EXT_SEED, "vcc_seed": VCC_SEED, "batch_seed": BATCH_SEED, "fold_seeds": list(FOLD_SEEDS),
                     "alpha": ALPHA, "margins": {"R": MARGIN_R, "increment": MARGIN_INCREMENT}, "positive_control_limit": POSITIVE_LIMIT,
                     "near_ceiling": NEAR_CEILING, "descriptive_below_target_genes": DESCRIPTIVE_GENES, "refits": DATA["refits"]},
        "replicate_index_sha256": {s: STATE["draws"][s]["draw"].index_sha256 for s in SCREENS},
        "cluster_identifier": "target_ensg, constructs in construct order",
        "vcc_targets_without_an_ensembl_identifier": DATA["vcc_unmatched"],
        "inputs": {**inputs, "predictions_manifest": file_record(manifest_path), "protocol": file_record(manifest_path.parent / "protocol.json") if SMOKE else file_record(OUT / "protocol.json")},
        "outputs": outputs,
        "objects_scored": info["objects"],
        "in_support_subset": {f"{c} | {s}": v for (c, s), v in DATA["support_share"].items()},
        "stored_phase4_predictions": file_record(REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz"),
        "seconds": round(time.time() - started, 1),
    }
    (OUT / "stats.json").write_text(json.dumps(strict_json(stats), indent=1, allow_nan=False) + "\n")
    print("[stats] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        SMOKE = True
        B = 20
        resample.BLOCK = 20
        OUT = config.storage_root() / "bioinf2027" / "scratch" / "E1b_stats_smoke"
        PLAN = PLAN + "; smoke run, 20 replicates, never reported"
    elif sys.argv[1:]:
        raise SystemExit("unknown arguments: %s" % sys.argv[1:])
    main()
