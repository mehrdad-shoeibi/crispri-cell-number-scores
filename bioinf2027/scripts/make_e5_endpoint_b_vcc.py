"""E5, endpoint (b) on VCC, descriptive: the target-excluded energy distance at 27 (G3), 10 (G4) and 50 cells.

Design: bioinf2027/plans/E5.md, Sections 3, 4, 16.1 and 17 (VCC). Per (split, target gene) of E1's gene-level
cells, the gene's perturbed cells pooled over its modelled rows in the split (E1's `vcc_row_table` and
`gene_rows`, checked against G3's and G4's pool sizes), against the 38,176 VCC control cells, on the first 50
control-cell components of the 2,000 features, the target removed from the features and from every cell's
total, with E1's draws; halves with E1's keys, k the gene's position among the split's genes with at least 10
pooled cells (sorted by symbol); the composition null and the matched versions per batch. The 10 kb sensitivity
is not available: the VCC files carry no gene positions. The construct-level computation is
`make_e5_endpoint_b.construct_values`, imported unchanged. VCC values are descriptive and enter no family.

Outputs: bioinf2027/results/E5/endpoint_b/vcc_genes.csv and endpoint_b_vcc.json. With --smoke: 12 genes per
split, outputs to $IGA_STORAGE_ROOT/bioinf2027/scratch/E5_endpoint_b_smoke/.
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
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, endpoint, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import e5lib  # noqa: E402
import make_e1a_extract as first  # noqa: E402
import make_e1a_label_extract as second  # noqa: E402
import make_e1a_labels as e1labels  # noqa: E402
import make_e5_endpoint_b as eb  # noqa: E402

PLAN = "bioinf2027/plans/E5.md, Sections 3, 4, 16.1 and 17 (VCC)"
SPLITS = ("train", "val", "test")
LEVELS = (27, 10, 50)
HALF_FLOOR = {27: 54, 10: 20, 50: 100}
WORKERS = 4
SMOKE = False
SMOKE_GENES = 12
CHECK_GENES = 10


def load() -> dict:
    m1 = json.loads((second.cache_dir() / "manifest.json").read_text())
    m4 = json.loads((eb.e4_dir() / "manifest.json").read_text())
    lab, e4 = {}, {}
    for split in SPLITS:
        keys = ["pert_rows", "pert_target", "pert_batch", "pert_guide", "pert_count", "pert_total"]
        if split == "train":
            keys += ["nt_rows", "nt_batch", "targets", "nt_counts", "nt_total"]
        with eb.checked(second.cache_dir(), "vcc_" + split, m1["cache"]["vcc_" + split]) as store:
            lab[split] = {k: store[k] for k in keys}
        with eb.checked(eb.e4_dir(), "vcc_" + split, m4["cache"]["vcc_" + split]) as store:
            if not np.array_equal(store["pert_rows"], lab[split]["pert_rows"]):
                raise eb.Stop("check (v): the E4 cache's VCC %s rows differ from the E1a label cache's" % split)
            e4[split] = {"pert_f": store["pert_counts"], "feature_columns": store["feature_columns"]}
            if split == "train":
                if not np.array_equal(store["nt_rows"], lab[split]["nt_rows"]):
                    raise eb.Stop("check (v): the E4 cache's VCC control rows differ from the E1a label cache's")
                e4[split]["nt_f"] = store["nt_counts"]
    rows = e1labels.vcc_row_table({"vcc_" + s: lab[s] for s in SPLITS})
    genes = {n: e1labels.gene_rows(rows, n) for n in LEVELS}
    for n, cell in ((27, "G3"), (10, "G4")):
        path = config.storage_root() / "bioinf2027" / "endpoints" / cell / "vcc_labels.csv"
        recorded = json.loads((eb.REPO / "bioinf2027" / "endpoints" / cell / "manifest.json").read_text())["tables"]["vcc_labels.csv"]["sha256"]
        if sha256(path) != recorded:
            raise eb.Stop("check (i): %s's vcc_labels.csv does not match its manifest" % cell)
        table = pd.read_csv(path, float_precision="round_trip")
        ours = genes[n].set_index(["split", "target_gene"])["n_pool"]
        theirs = table.set_index(["split", "target_gene"])["n_pool"]
        if not ours.sort_index().equals(theirs.sort_index()):
            raise eb.Stop("the pooled genes at %d cells differ from %s's" % (n, cell))
        genes[cell] = table.set_index(["split", "target_gene"])
    return {"lab": lab, "e4": e4, "genes": genes}


def screen_like(data: dict, split: str) -> dict:
    """The fields make_e5_endpoint_b.construct_values reads, for one VCC split."""
    lab, e4, train = data["lab"][split], data["e4"][split], data["lab"]["train"]
    part = data["genes"][10][data["genes"][10]["split"] == split].sort_values("target_gene").reset_index(drop=True)
    if SMOKE:
        part = part.iloc[:SMOKE_GENES].reset_index(drop=True)
    symbols = features.hvg_symbols()
    position = {s: i for i, s in enumerate(symbols.tolist())}
    slot_of = {t: i for i, t in enumerate(train["targets"].tolist())}
    nt_gem = train["nt_batch"].astype(str)
    S = {
        "screen": f"VCC {split}", "table": pd.DataFrame({"construct": part.target_gene, "target_ensg": part.target_gene}),
        "pools": [np.asarray(p) for p in part.positions], "slot": np.array([slot_of[g] for g in part.target_gene]),
        "pert_f": e4["pert_f"], "nt_f": data["e4"]["train"]["nt_f"], "pert_total": lab["pert_total"], "nt_total": train["nt_total"],
        "pert_count": lab["pert_count"], "nt_counts": train["nt_counts"], "pert_gem": lab["pert_batch"].astype(str), "nt_gem": nt_gem,
        "ctrl_by_gem": {g: np.flatnonzero(nt_gem == g) for g in sorted(set(nt_gem.tolist()))}, "decile_members": None,
        "target_symbol": part.target_gene.to_numpy().astype(str),
        "target_feature": np.array([position.get(g, -1) for g in part.target_gene], dtype=np.int64),
        "neighbours": [np.empty(0, dtype=np.int64) for _ in range(len(part))],
        "symbol_mismatches": 0,
        "control_halves": endpoint.half_split(train["nt_rows"].size, e1labels.HALF_CONTROLS_SEED),
        "deciles": np.quantile(train["nt_total"].astype(np.float64), np.arange(1, 10) / 10.0),
    }
    eb.add_deciles(S)
    return S


def main() -> None:
    started = time.time()
    eb.LEVELS, eb.HALF_FLOOR, eb.WORKERS = LEVELS, HALF_FLOOR, WORKERS
    # 38,176 controls: Gram blocks of 1,024 rows keep each worker near 1 GB (the values do not depend on the block)
    e5lib.mean_within.__defaults__ = (1024,)
    e5lib.mean_cross.__defaults__ = (1024,)
    single_cell = first.verify_single_cell_inputs()
    data = load()
    if not all(np.all(data["e4"][s]["feature_columns"] >= 0) for s in SPLITS):
        raise eb.Stop("a VCC feature gene is missing from a split's file")
    folder = eb.out_dir() if not SMOKE else config.storage_root() / "bioinf2027" / "scratch" / "E5_endpoint_b_smoke"
    folder.mkdir(parents=True, exist_ok=True)
    nt_f, nt_total = data["e4"]["train"]["nt_f"], data["lab"]["train"]["nt_total"]
    eb.DATA["basis0"] = e5lib.basis(e5lib.values(nt_f, nt_total), eb.N_PC)
    frames, checks = [], {}
    for split in SPLITS:
        S = screen_like(data, split)
        eb.DATA["screen"] = S
        k_all = list(range(len(S["pools"])))
        g3 = data["genes"]["G3"]
        probe = [k for k in k_all if S["pools"][k].size >= 27][:CHECK_GENES]
        largest = 0.0
        for k in probe:
            ours = eb.check_iii(k)
            theirs = g3.loc[(split, S["table"].construct.iloc[k])]
            for key in ("y", "y_draws_1_12", "y_draws_13_25"):
                a_, b_ = ours[key], float(theirs[key])
                d = 0.0 if (np.isnan(a_) and np.isnan(b_)) else abs(a_ - b_)
                largest = max(largest, d)
                if d != 0.0:
                    raise eb.Stop("check (iii): VCC %s %s %s differs from G3 by %r" % (split, S["table"].construct.iloc[k], key, d))
        checks[f"iii {split}"] = {"genes": len(probe), "largest_difference": largest, "passed": True}
        results = eb.run_pool(eb.construct_values, k_all)
        table = pd.DataFrame(results)
        largest = 0.0
        for k in probe[:3]:
            again = eb.construct_values(k)
            saved = table[table.position == k].iloc[0]
            for key, value in again.items():
                if isinstance(value, float):
                    other = float(saved[key])
                    if np.isnan(value) != np.isnan(other):
                        raise eb.Stop("check (iv): VCC %s %s is NaN in one computation only" % (again["construct"], key))
                    d = 0.0 if np.isnan(value) else abs(value - other)
                    largest = max(largest, d)
                    if d > 1e-10:
                        raise eb.Stop("check (iv): VCC %s %s differs by %r" % (again["construct"], key, d))
        checks[f"iv {split}"] = {"genes": len(probe[:3]), "largest_difference": largest, "tolerance": 1e-10, "passed": True}
        table.insert(0, "split", split)
        frames.append(table)
        print("[e5 b vcc] %s: %d genes, %.0f s" % (split, len(table), time.time() - started), flush=True)
    genes = pd.concat(frames, ignore_index=True).rename(columns={"construct": "target_gene"}).drop(columns=["target_ensg", "n_neighbour_features"])
    genes = genes.drop(columns=[c for c in genes.columns if c.endswith("_10kb") or c.endswith("_10kb_valid")])
    gate_values = {}
    for split in SPLITS:
        part = genes[genes.split == split].reset_index(drop=True)
        folds = part.target_gene.map(e5lib.within_screen_folds(part.target_gene)).to_numpy()
        bc = np.column_stack([e5lib.ranks01(part[c].to_numpy(dtype=np.float64)) for c in ("B1", "detection_rate", "control_sd")])
        gate_values[split] = {}
        for n in LEVELS:
            if f"b{n}_half_A" not in part:
                continue
            a, b = part[f"b{n}_half_A"].to_numpy(dtype=np.float64), part[f"b{n}_half_B"].to_numpy(dtype=np.float64)
            both = np.isfinite(a) & np.isfinite(b)
            r = float(spearmanr(a[both], b[both])[0]) if both.sum() > 2 else float("nan")
            ra = e5lib.out_of_fold_residuals(bc, np.where(both, a, np.nan), folds)
            rb = e5lib.out_of_fold_residuals(bc, np.where(both, b, np.nan), folds)
            ok = np.isfinite(ra) & np.isfinite(rb)
            gate_values[split][str(n)] = {"genes_with_halves": int(both.sum()), "reliability_spearman": r,
                                          "beyond_bc_spearman": float(spearmanr(ra[ok], rb[ok])[0]) if ok.sum() > 2 else float("nan")}
    genes.to_csv(folder / "vcc_genes.csv", index=False)
    record = {"plan": PLAN, "smoke": SMOKE, "single_cell_inputs": single_cell,
              "caches": {"E1a_labels": file_record(second.cache_dir() / "manifest.json"), "E4": file_record(eb.e4_dir() / "manifest.json")},
              "shards": config.portable(first.shard_root()), "levels": list(LEVELS), "half_floors": HALF_FLOOR,
              "half_key": "[20270407, position of the gene among the split's genes with at least 10 pooled cells, sorted by symbol]",
              "controls": int(nt_total.size), "features": int(nt_f.shape[1]), "components": eb.N_PC,
              "ten_kb": "not available: the VCC files carry no gene positions", "gates_descriptive": gate_values, "checks": checks,
              "genes": {s: int((genes.split == s).sum()) for s in SPLITS},
              "outputs": {"vcc_genes.csv": file_record(folder / "vcc_genes.csv")}, "seconds": round(time.time() - started, 1)}
    (folder / "endpoint_b_vcc.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e5 b vcc] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
    if "--workers" in arguments:  # changes no value; set from the measured memory per worker (gate G3, item 5)
        i = arguments.index("--workers")
        WORKERS = int(arguments[i + 1])
        arguments = arguments[:i] + arguments[i + 2:]
    if arguments:
        raise SystemExit("usage: make_e5_endpoint_b_vcc.py [--smoke] [--workers N]")
    main()
