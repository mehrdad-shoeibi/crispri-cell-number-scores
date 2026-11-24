"""The fitting harness of the transfer analyses: representation blocks, one fit, saved predictions.

A fit is defined by a context (the rows, labels, trivial predictors and raw feature blocks of one
training set and of every side it predicts), a label, a representation, a head, a seed and a
training variant. Every fitted preprocessing step uses the fit's own training rows:

- z, x, the masked x and the Gaussian and projection controls are standardised by a scaler fitted on
  the training rows (``earlier_revision/pipeline/fm_preprocess.py``, extended from z, C1, C2 and C3
  to every block); the shuffled control C3 uses the scaler of z, as submitted;
- the magnitude summaries are z-scored with the training rows' mean and population sd
  (``src/training/magnitude_augment.py``), the masked ones from the masked x;
- the trivial predictors B0 to B4 enter unscaled; ridge standardises its whole input again.

Controls follow the submitted generators (`bioinf2027/src/bioinf2027/representations.py`): within a
VCC split, C1 row i and the C3 permutation are those of the split's rows in the loader's order; on a
screen, of its constructs in construct order. Realisation k adds 1,000 k to the base seed.

The predictions of a fit, one array per side, are saved with the fit's specification, and a fit
whose file exists with the same specification is not repeated.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import heads, representations as rep
from .provenance import file_record

__all__ = [
    "FAMILY",
    "MASKED",
    "CONTROLS",
    "PIPELINE",
    "TRIVIAL_MODELS",
    "ONE_VARIABLE",
    "REPRESENTATIONS",
    "USES_B",
    "USES_X",
    "Side",
    "Context",
    "FitSpec",
    "run_fit",
]

FAMILY = ("z", "x", "m", "[x; m]")
MASKED = ("x-", "m-", "[x; m]-")
CONTROLS = ("C1", "C2", "C3")
PIPELINE = ("[C1; B2]", "[z; B2]")
TRIVIAL_MODELS = ("TR", "[TR; z]", "[TR; x]", "[TR; m]", "[TR; [x; m]]")
ONE_VARIABLE = ("B0", "B1", "B2", "B3", "B4", "B2 feature space")
POSITIVE = ("label positive control",)
REPRESENTATIONS = FAMILY + MASKED + CONTROLS + PIPELINE + TRIVIAL_MODELS + ONE_VARIABLE + POSITIVE
USES_B = set(PIPELINE + TRIVIAL_MODELS + ONE_VARIABLE + POSITIVE)
USES_X = {"x", "m", "[x; m]", "x-", "m-", "[x; m]-", "C2", "[TR; x]", "[TR; m]", "[TR; [x; m]]", "B2 feature space"}
B_COLUMNS = ("B0", "B1", "B2", "B3", "B4")


@dataclass
class Side:
    """Raw blocks of one set of rows: a VCC split of the context, or an external screen.

    ``control_split`` names the stream of the submitted controls (train, val, test, RPE1 or
    K562-essential); ``x`` maps a basis name to the expression delta in that basis.

    A side that holds part of a screen (a fold of E6) sets ``control_rows``, its rows' positions among
    the screen's constructs, and ``z_full``, the screen's z in construct order: its controls are then
    the screen's controls, generated once for every construct, at those positions.
    """

    name: str
    control_split: str
    z: np.ndarray
    x: dict
    b: np.ndarray
    target_column: np.ndarray
    control_rows: np.ndarray | None = None
    z_full: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.z.shape[0])

    @property
    def control_total(self) -> int:
        return self.n if self.control_rows is None else int(self.z_full.shape[0])


@dataclass
class Context:
    """The training rows (with their labels) and every side a fit of this context predicts."""

    name: str
    train: Side
    labels: dict
    sides: dict
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FitSpec:
    context: str
    label: str
    representation: str
    head: str
    seed: int
    variant: str = "all training rows"
    realisation: int = 0

    def key(self) -> str:
        return "|".join([self.context, self.label, self.representation, self.head, str(self.seed), self.variant, str(self.realisation)])

    def as_dict(self) -> dict:
        return {"context": self.context, "label": self.label, "representation": self.representation, "head": self.head,
                "seed": self.seed, "variant": self.variant, "realisation": self.realisation}

    def file_name(self) -> str:
        import hashlib

        return hashlib.sha1(self.key().encode()).hexdigest()[:20] + ".npz"


# --------------------------------------------------------------------------- blocks


def _controls(side: Side, realisation: int, kind: str, z_scaler=None, x_raw: np.ndarray | None = None) -> np.ndarray:
    """C1, C2 or C3 of a side's rows; C3 needs the scaler of z, C2 the side's x."""
    if kind == "C1":
        full = rep.c1_features(side.control_total, side.control_split, realisation)
        return full if side.control_rows is None else full[side.control_rows]
    if kind == "C2":
        return rep.c2_features(x_raw, rep.c2_projection(realisation))
    if kind == "C3":
        source = side.z if side.control_rows is None else side.z_full
        shuffled = z_scaler(source)[rep.c3_permutation(side.control_total, side.control_split, realisation)]
        return shuffled if side.control_rows is None else shuffled[side.control_rows]
    raise ValueError(kind)


class Blocks:
    """The fitted transforms of one fit and the matrices they give for any side."""

    def __init__(self, spec: FitSpec, train: Side, train_index: np.ndarray, basis: str = "main") -> None:
        self.spec = spec
        self.train_side = train
        self.index = train_index
        self.basis = basis
        rows = train_index
        self.fitted = {}
        r = spec.representation
        k = spec.realisation
        if r in ("z", "C3", "[z; B2]", "[TR; z]"):
            self.fitted["z"] = rep.BlockScaler(train.z[rows])
        if r in ("x", "[x; m]", "[TR; x]", "[TR; [x; m]]"):
            self.fitted["x"] = rep.BlockScaler(train.x["main"][rows])
        if r in ("m", "[x; m]", "[TR; m]", "[TR; [x; m]]"):
            self.fitted["m"] = rep.fit_magnitude(train.x["main"][rows])
        if r in ("x-", "[x; m]-"):
            masked = rep.mask_target(train.x["main"], train.target_column)
            self.fitted["x-"] = rep.BlockScaler(masked[rows])
        if r in ("m-", "[x; m]-"):
            masked = rep.mask_target(train.x["main"], train.target_column)
            self.fitted["m-"] = rep.fit_magnitude(masked[rows])
        if r in ("C1", "[C1; B2]"):
            self.fitted["C1"] = rep.BlockScaler(_controls(train, k, "C1")[rows])
        if r == "C2":
            self.fitted["C2"] = rep.BlockScaler(_controls(train, k, "C2", x_raw=train.x["main"])[rows])

    def matrix(self, side: Side, basis: str = "main") -> np.ndarray:
        r = self.spec.representation
        k = self.spec.realisation
        x = side.x.get(basis) if r in USES_X else None
        if r in USES_X and x is None:
            raise KeyError("side %s has no basis %s" % (side.name, basis))
        b = side.b

        def z_block():
            return self.fitted["z"](side.z)

        def x_block():
            return self.fitted["x"](x)

        def m_block():
            return self.fitted["m"].transform(rep.magnitude(x))

        def masked_x():
            return rep.mask_target(x, side.target_column)

        if r == "z":
            return z_block()
        if r == "x":
            return x_block()
        if r == "m":
            return m_block()
        if r == "[x; m]":
            return np.concatenate([x_block(), m_block()], axis=1)
        if r == "x-":
            return self.fitted["x-"](masked_x())
        if r == "m-":
            return self.fitted["m-"].transform(rep.magnitude(masked_x()))
        if r == "[x; m]-":
            mx = masked_x()
            return np.concatenate([self.fitted["x-"](mx), self.fitted["m-"].transform(rep.magnitude(mx))], axis=1)
        if r == "C1":
            return self.fitted["C1"](_controls(side, k, "C1"))
        if r == "C2":
            return self.fitted["C2"](_controls(side, k, "C2", x_raw=x))
        if r == "C3":
            return _controls(side, k, "C3", z_scaler=self.fitted["z"])
        if r == "[C1; B2]":
            return np.concatenate([self.fitted["C1"](_controls(side, k, "C1")), b[:, [2]]], axis=1).astype(np.float32)
        if r == "[z; B2]":
            return np.concatenate([z_block(), b[:, [2]]], axis=1).astype(np.float32)
        if r == "TR":
            return b[:, :3].astype(np.float32)
        if r == "[TR; z]":
            return np.concatenate([b[:, :3], z_block()], axis=1).astype(np.float32)
        if r == "[TR; x]":
            return np.concatenate([b[:, :3], x_block()], axis=1).astype(np.float32)
        if r == "[TR; m]":
            return np.concatenate([b[:, :3], m_block()], axis=1).astype(np.float32)
        if r == "[TR; [x; m]]":
            return np.concatenate([b[:, :3], x_block(), m_block()], axis=1).astype(np.float32)
        if r in ("B0", "B1", "B2", "B3", "B4"):
            return b[:, [B_COLUMNS.index(r)]].astype(np.float32)
        if r == "B2 feature space":
            out = np.full((side.n, 1), np.nan, dtype=np.float32)
            has = side.target_column >= 0
            out[has, 0] = -x[has, side.target_column[has]]
            return out
        if r == "label positive control":
            return b[:, [1, 2]].astype(np.float32)
        raise ValueError("unknown representation %r" % r)


# --------------------------------------------------------------------------- one fit


def training_rows(context: Context, spec: FitSpec, refit_seed: int) -> np.ndarray:
    """Positions of the training rows of a fit, repeated where a refit draws a gene twice."""
    meta = context.meta
    n = context.train.n
    rows = np.arange(n)
    if spec.representation == "B2 feature space":
        rows = rows[context.train.target_column >= 0]
    variant = spec.variant
    if variant == "all training rows":
        return rows
    if variant == "target is a feature":
        return rows[context.train.target_column[rows] >= 0]
    if variant == "target is not a feature":
        return rows[context.train.target_column[rows] < 0]
    if variant == "single-guide rows":
        return rows[meta["n_guides"][rows] == 1]
    if variant.startswith("refit "):
        r = int(variant.split(" ", 1)[1])
        genes = meta["genes"][rows]
        unique = np.unique(genes)
        rng = np.random.default_rng([refit_seed, r, meta["context_index"]])
        drawn = unique[rng.integers(0, unique.size, unique.size)]
        members = {g: rows[genes == g] for g in unique}
        return np.concatenate([members[g] for g in drawn])
    raise ValueError("unknown variant %r" % variant)


def _head(spec: FitSpec):
    if spec.head == "forest":
        return heads.forest(spec.seed)
    if spec.head == "boosting":
        return heads.boosting(spec.seed)
    if spec.head == "monotone":
        return heads.monotone([0, 1])
    raise ValueError(spec.head)


def run_fit(context: Context, spec: FitSpec, folder: Path, refit_seed: int, external_bases: dict) -> dict:
    """Fit one model and save its predictions on every side; skip a fit already saved."""
    path = folder / spec.file_name()
    spec_text = json.dumps(spec.as_dict(), sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as store:
            if str(store["spec"]) == spec_text:
                return {"spec": spec.as_dict(), "file": file_record(path), "reused": True}
    rows = training_rows(context, spec, refit_seed)
    y = context.labels[spec.label][rows]
    if not np.isfinite(y).all():
        raise RuntimeError("%s: a training label is missing" % spec.key())
    blocks = Blocks(spec, context.train, rows)
    start = time.time()
    record = {}
    out = {}
    with heads.single_thread():
        x_train = blocks.matrix(context.train)[rows]
        if spec.head == "ridge":
            val = context.sides["val"]
            val_mask = np.isfinite(context.labels[spec.label + "@val"])
            x_val = blocks.matrix(val)
            if spec.representation == "B2 feature space":
                val_mask &= val.target_column >= 0
            model, scaler, choice = heads.fit_ridge_by_validation(x_train, y, x_val[val_mask], context.labels[spec.label + "@val"][val_mask])
            record["ridge"] = choice

            def predict(matrix):
                return model.predict(scaler.transform(np.asarray(matrix, dtype=np.float64)))
        else:
            model = _head(spec).fit(x_train, y)

            def predict(matrix):
                return model.predict(matrix)

        for name, side in context.sides.items():
            bases = ["main"] + ([b for b in external_bases.get(name, ()) if b in side.x] if spec.representation in USES_X else [])
            for basis in bases:
                matrix = blocks.matrix(side, basis)
                values = np.full(side.n, np.nan)
                usable = np.isfinite(matrix).all(axis=1)
                if usable.any():
                    values[usable] = predict(matrix[usable])
                out[name if basis == "main" else f"{name}|{basis}"] = values
    record["seconds"] = time.time() - start
    record["training_rows"] = int(rows.size)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, spec=np.array(spec_text), record=np.array(json.dumps(record, sort_keys=True)),
             **{name.replace("|", "__"): values for name, values in out.items()})
    os.replace(temporary, path)
    return {"spec": spec.as_dict(), "file": file_record(path), "reused": False, "seconds": record["seconds"]}


def load_predictions(path: Path) -> tuple[dict, dict, dict]:
    with np.load(path, allow_pickle=False) as store:
        spec = json.loads(str(store["spec"]))
        record = json.loads(str(store["record"]))
        values = {k.replace("__", "|"): np.asarray(store[k]) for k in store.files if k not in ("spec", "record")}
    return spec, record, values
