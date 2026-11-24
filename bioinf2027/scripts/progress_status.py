"""One status line for bioinf2027/PROGRESS.md, from saved files only, with finish estimates.

Read-only: it fits nothing, writes nothing and touches no running job. It prints one Markdown table row
for PROGRESS.md and, after it, the JSON of the values the row uses. Sources:
- E1b: one prediction file per finished fit under $IGA_STORAGE_ROOT/bioinf2027/E1b/predictions/<cell>/
  (the folder harness/ holds the pre-run harness check and is not counted), against the fits of
  bioinf2027/results/E1/protocol.json. Each file records its fit's seconds and training rows.
- D2: one record per finished unit, $IGA_STORAGE_ROOT/bioinf2027/D2/embeddings/<part>/unit_NNNNN.json,
  against the units per part of bioinf2027/results/D2/units.json.
- E4 and E6: their records under bioinf2027/results/E4 and bioinf2027/results/E6, and whether their
  scripts are running.

Estimates (labelled as such; none enters a result):
- E1b: a remaining fit takes the median recorded seconds of the finished fits of its kind (cell, head,
  representation); a kind not yet seen takes k x training rows x columns, with k the head's seconds per
  row and column over its finished fits, or, for a head with fewer than three, the forest's k times the
  head's ratio to the forest in the timing probe (bioinf2027/results/E1/checks.json), else times E1b's
  cost weight. The sum is divided by E1b's 12 workers; the fits in progress count as not started.
- D2: the remaining cells of each part times the recorded seconds per cell (tokenizing and embedding) of
  the part's finished units, or of all finished units of the six parts if the part has none; one GPU,
  the time between jobs not included.
- E4, E6 and D2's head fits in the decided order (bioinf2027/plans/E4.md and E6.md, Section 16): every
  fit of tiers 1 and 2 of E4 and E6 and D2's head fits ("G2's fits") from the end of E1b (D2's fits not
  before the end of D2's embeddings), then tier 3; the fits not yet saved priced as E1b's unseen kinds on
  12 workers (E4 without the fits the amendment drops; E6 with 20 refits and head seeds 0 to 2 until its
  check fixes them, E6 at 300 training rows for the size-matched draws and 1,500 otherwise, scaled to
  E6's own projection once its check exists). Fits saved are counted per tier from the prediction files.
  Statistics, audits and reviews are not included.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells  # noqa: E402
from bioinf2027.provenance import sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "bioinf2027" / "results"
WORKERS = 12
COLUMNS = {"z": 768, "x": 2000, "[x; m]": 2004, "x-": 2000, "[x; m]-": 2004, "C1": 768, "C2": 768, "C3": 768,
           "[C1; B2]": 769, "[z; B2]": 769, "[TR; z]": 771, "[TR; x]": 2003, "[TR; [x; m]]": 2007}
COST_WEIGHT = {"forest": 1.0, "boosting": 0.05, "ridge": 0.02, "ridge-cv": 0.1, "monotone": 0.001}
D2_PARTS = ("vcc_perturbed", "vcc_delta", "rpe1_perturbed", "rpe1_delta", "k562_perturbed", "k562_delta")


def columns(representation: str) -> int:
    return COLUMNS.get(representation, 5)


def key(spec: dict) -> tuple:
    return tuple(sorted(spec.items()))


def running(script: str) -> bool:
    """Whether a Python process of this user runs the script (its own command line, not a shell's)."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if argv and b"python" in Path(argv[0].decode(errors="replace")).name.encode() and any(a.endswith(script.encode()) for a in argv[1:]):
            return True
    return False


def train_rows() -> dict:
    root = config.storage_root() / "bioinf2027" / "endpoints"
    return {cell: int((pd.read_csv(root / cell / "vcc_labels.csv", usecols=["split"])["split"] == "train").sum()) for cell in e1cells.CELLS}


# --------------------------------------------------------------------------- E1b


def e1b_finished() -> list[dict]:
    root = config.storage_root() / "bioinf2027" / "E1b" / "predictions"
    out = []
    for path in root.rglob("*.npz"):
        if path.parent.name == "harness":
            continue
        with np.load(path, allow_pickle=False) as store:
            out.append({"spec": json.loads(str(store["spec"])), **json.loads(str(store["record"])), "mtime": path.stat().st_mtime})
    return out


def pricing(finished: list[dict], rows: dict) -> dict:
    """Seconds per kind from finished fits, and seconds per row and column per head."""
    frame = pd.DataFrame([{**f["spec"], "seconds": f["seconds"], "rows": f["training_rows"]} for f in finished])
    by_kind = frame.groupby(["context", "head", "representation"])["seconds"].median().to_dict() if len(frame) else {}
    probe = json.loads((RESULTS / "E1" / "checks.json").read_text())["timing_probe"]["fits"]
    probe_k = {}
    for head in {p["head"] for p in probe}:
        chosen = [p for p in probe if p["head"] == head and p["columns"] > 5]
        probe_k[head] = sum(p["seconds"] for p in chosen) / sum(p["rows"] * p["columns"] for p in chosen)
    k = {}
    for head, group in frame.groupby("head") if len(frame) else []:
        group = group[group["representation"].map(columns) > 5]
        if len(group) >= 3:
            k[head] = float(group["seconds"].sum() / (group["rows"] * group["representation"].map(columns)).sum())
    forest = k.get("forest", probe_k["forest"])

    def head_k(head: str) -> float:
        if head in k:
            return k[head]
        if head in probe_k:
            return forest * probe_k[head] / probe_k["forest"]
        return forest * COST_WEIGHT[head]

    return {"by_kind": by_kind, "head_k": head_k, "k": k, "probe_k": probe_k}


def price(context: str, head: str, representation: str, n_rows: int, model: dict, by_kind: bool = True) -> float:
    if by_kind and (context, head, representation) in model["by_kind"]:
        return float(model["by_kind"][(context, head, representation)])
    return model["head_k"](head) * n_rows * columns(representation)


def e1b_status(rows: dict) -> dict:
    import make_e1b_fits as e1b

    protocol = json.loads((RESULTS / "E1" / "protocol.json").read_text())
    e1b.DATA["train_rows"] = rows
    tasks = e1b.all_tasks(protocol["refits"])
    finished = e1b_finished()
    done = {key(f["spec"]) for f in finished}
    model = pricing(finished, rows)
    remaining = [t for t in tasks if key(t.as_dict()) not in done]
    seconds = sum(price(t.context, t.head, t.representation, rows[t.context], model) for t in remaining)
    return {"fits": protocol["fits"], "listed": len(tasks), "finished": len(finished), "matched": len(done & {key(t.as_dict()) for t in tasks}),
            "remaining_cpu_hours": seconds / 3600, "remaining_wall_hours": seconds / WORKERS / 3600, "model": model,
            "running": running("make_e1b_fits.py"),
            "source": "prediction files under `$IGA_STORAGE_ROOT/bioinf2027/E1b/predictions`, harness/ excluded"}


# --------------------------------------------------------------------------- D2


def d2_status() -> dict:
    record = json.loads((RESULTS / "D2" / "units.json").read_text())
    root = config.storage_root() / "bioinf2027" / "D2" / "embeddings"
    units, cells, finished = {}, {}, []
    for part in D2_PARTS:
        units[part] = int(record["files"][part]["units"])
        cells[part] = int(record["files"][part]["cells"])
        for path in sorted((root / part).glob("unit_*.json")) if (root / part).exists() else []:
            finished.append(json.loads(path.read_text()))
    frame = pd.DataFrame(finished)
    if len(frame):
        frame["seconds_per_cell"] = (frame["tokenize_seconds"] + frame["embed_seconds"]) / frame["cells"]
        overall = float(frame["seconds_per_cell"].median())
    else:
        overall = float("nan")
    seconds, per_part = 0.0, {}
    for part in D2_PARTS:
        own = frame[frame["part"] == part] if len(frame) else frame
        done_cells = int(own["cells"].sum()) if len(own) else 0
        rate = float(own["seconds_per_cell"].median()) if len(own) else overall
        seconds += (cells[part] - done_cells) * rate
        per_part[part] = {"units_done": int(len(own)), "units": units[part], "seconds_per_cell": rate}
    return {"units": sum(units.values()), "finished": int(len(frame)), "per_part": per_part, "remaining_gpu_hours": seconds / 3600,
            "running": running("make_d2_embeddings.py"),
            "source": "unit records under `$IGA_STORAGE_ROOT/bioinf2027/D2/embeddings`, against `bioinf2027/results/D2/units.json`"}


# --------------------------------------------------------------------------- E4 and E6


def saved(item: str, tasks: list) -> list:
    """Whether each fit's prediction file exists under the item's prediction folder."""
    root = config.storage_root() / "bioinf2027" / item / "predictions"
    return [(root / s.file_name()[:2] / s.file_name()).exists() for _, s in tasks]


def per_tier(tiers: list, done: list, seconds: list) -> dict:
    out = {}
    for tier in (1, 2, 3):
        idx = [i for i, t in enumerate(tiers) if t == tier]
        out[str(tier)] = {"fits": len(idx), "saved": int(sum(done[i] for i in idx)),
                          "remaining_cpu_hours": sum(seconds[i] for i in idx if not done[i]) / 3600}
    return out


def e4_status(rows: dict, model: dict) -> dict:
    import fit_tiers
    import make_e4_fits as e4

    main = RESULTS / "E4" / "features.json"
    thinned = RESULTS / "E4" / "features_thinned.json"
    state = {"builder_running": running("make_e4_features.py"), "fits_running": running("make_e4_fits.py"),
             "stats_running": running("make_e4_stats.py")}
    if main.exists():
        record = json.loads(main.read_text())
        state["features_checks_passed"] = bool(record.get("checks_passed"))
        state["features_construct_names_as_text"] = "second_run" in record
    if thinned.exists() and main.exists():
        state["thinned_on_the_current_features"] = json.loads(thinned.read_text())["inputs"]["main_features"]["sha256"] == sha256(main)
    state["manifests"] = sorted(p.name for p in (RESULTS / "E4").glob("predictions_manifest_tier*.json"))
    state["family_pass_written"] = (RESULTS / "E4" / "family_pass" / "stats.json").exists()
    state["stats_written"] = (RESULTS / "E4" / "stats.json").exists()
    e4.DATA["thinned"] = thinned.exists()  # the depth-thinned fits are listed once their features exist
    tasks = e4.run_list(json.loads((RESULTS / "E1" / "protocol.json").read_text())["refits"], 3)
    seconds = [price(d["cell"], s.head, s.representation, rows.get(d["cell"], 3000), model, by_kind=False) for d, s in tasks]
    state["tiers"] = per_tier([fit_tiers.e4_tier(d, s) for d, s in tasks], saved("E4", tasks), seconds)
    state["fits"] = len(tasks)
    return state


def e6_status(model: dict) -> dict:
    import fit_tiers
    import make_e6_fits as e6
    from bioinf2027.transfer import FitSpec

    # R(z-) of the family: z- of D2 in each fold of both directions, head seeds 0 to 2 (make_e6_fits.z_minus_tasks)
    r_z_minus = [({"source": s, "endpoint": e6.PRIMARY, "fit_set": f"fold {k}", "z_minus": True},
                  FitSpec(e6.context_name(s, e6.PRIMARY, f"fold {k}") + " | z- of D2", "rebuilt", "z", "forest", seed))
                 for s in e6.SCREENS for k in range(5) for seed in e6.SEEDS]
    state = {"fits_running": running("make_e6_fits.py"), "stats_running": running("make_e6_stats.py"),
             "check_written": (RESULTS / "E6" / "checks.json").exists(), "stats_written": (RESULTS / "E6" / "stats.json").exists(),
             "family_pass_written": (RESULTS / "E6" / "family_pass" / "stats.json").exists(),
             "manifests": sorted(p.name for p in (RESULTS / "E6").glob("predictions_manifest_tier*.json"))}
    refits, seeds, projected = 20, (0, 1, 2), None
    if state["check_written"]:
        check = json.loads((RESULTS / "E6" / "checks.json").read_text())
        refits = int(check["refits"])
        seeds = tuple(check["seeds_of_the_700_control_endpoints"])
        projected = float(check["projected_cpu_hours"])
    e6.DATA["e4"] = (RESULTS / "E4" / "features.json").exists()  # the fits on E4's matched features

    def priced(listed: list) -> list:
        # E6's own training sizes before its probe: 300 rows for the draw endpoints, 1,500 otherwise (make_e6_fits.cost)
        return [price("", s.head, s.representation, 300 if "draw" in d else 1500, model, by_kind=False) for d, s in listed]

    tasks = e6.task_list(refits, seeds) + e6.shared_c2_tasks() + r_z_minus
    seconds = priced(tasks)
    if projected is None:
        # and its rule: above CPU_LIMIT_HOURS the 700-control endpoints run with head seed 0 only
        if sum(seconds) / 3600 > e6.CPU_LIMIT_HOURS:
            seeds = (0,)
            tasks = e6.task_list(refits, seeds) + e6.shared_c2_tasks() + r_z_minus
            seconds = priced(tasks)
        state["projection"] = "priced as E1b's unseen kinds, with E6's seed rule"
    else:
        scale = projected * 3600 / sum(seconds[:-len(r_z_minus)]) if sum(seconds[:-len(r_z_minus)]) else 1.0
        seconds = [x * scale for x in seconds]
        state["projection"] = "scaled to E6's own, bioinf2027/results/E6/checks.json"
    state["tiers"] = per_tier([fit_tiers.e6_tier(d, s) for d, s in tasks], saved("E6", tasks), seconds)
    state.update({"fits": len(tasks), "refits": refits, "seeds_700": list(seeds)})
    return state


def d2_fit_hours(rows: dict, model: dict) -> float:
    import make_d2_fits as d2

    if (RESULTS / "D2" / "predictions_manifest.json").exists():
        return 0.0
    refits = json.loads((RESULTS / "E1" / "protocol.json").read_text())["refits"]
    return sum(price(d.get("cell", ""), s.head, s.representation, rows.get(d.get("cell", ""), 3000), model, by_kind=False)
               for d, s in d2.task_list(refits)) / 3600


# --------------------------------------------------------------------------- the row


def when(hours: float, start: datetime) -> str:
    return (start + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    now = datetime.now()
    rows = train_rows()
    e1 = e1b_status(rows)
    d2 = d2_status()
    e4s = e4_status(rows, e1["model"])
    e6s = e6_status(e1["model"])
    d2_fits = d2_fit_hours(rows, e1["model"])
    e1_end = now + timedelta(hours=e1["remaining_wall_hours"])
    d2_end = now + timedelta(hours=d2["remaining_gpu_hours"])
    zone = time.strftime("%Z")

    def tiers(s: dict) -> str:
        return ", ".join("tier %s %d of %d" % (k, v["saved"], v["fits"]) for k, v in s["tiers"].items())

    e4_text = ("builder %s; features %s, thinned features %s; fits saved: %s%s; family pass %s; statistics %s" % (
        "running" if e4s["builder_running"] else "not running",
        "checks passed" + (" (names as text)" if e4s.get("features_construct_names_as_text") else " (names as objects)") if e4s.get("features_checks_passed") else "not passed",
        "current" if e4s.get("thinned_on_the_current_features") else "not on the current features",
        tiers(e4s), " (running)" if e4s["fits_running"] else "",
        "written" if e4s["family_pass_written"] else "not yet", "written" if e4s["stats_written"] else "not yet"))
    e6_text = ("check %s; fits saved: %s%s; family pass %s; statistics %s" % (
        "written" if e6s["check_written"] else "waits for E1b's predictions", tiers(e6s), " (running)" if e6s["fits_running"] else "",
        "written" if e6s["family_pass_written"] else "not yet", "written" if e6s["stats_written"] else "not yet"))
    what = ("E1b: %d of %d fits (%s). D2: %d of %d units (%s), GPU 0 only. E4: %s. E6: %s. From `bioinf2027/scripts/progress_status.py`." % (
        e1["finished"], e1["fits"], e1["source"], d2["finished"], d2["units"], d2["source"], e4_text, e6_text))
    start = max(now, e1_end)
    first = sum(s["tiers"][k]["remaining_cpu_hours"] for s in (e4s, e6s) for k in ("1", "2"))
    third = sum(s["tiers"]["3"]["remaining_cpu_hours"] for s in (e4s, e6s))
    g2_end = max(start + timedelta(hours=(first + d2_fits) / WORKERS), max(d2_end, start) + timedelta(hours=d2_fits / WORKERS))
    last = g2_end + timedelta(hours=third / WORKERS)
    estimates = ("Estimates (%s): E1b about %s (%.0f h on 12 workers); D2 embeddings about %s (%.0f GPU h left, restarts not included); "
                 "G2's fits (tiers 1 and 2 of E4 and E6, %d fits not yet saved, and D2's head fits) about %s; E4's and E6's last fit (tier 3) about %s; "
                 "statistics, audits and reviews not included" % (
                     zone, when(e1["remaining_wall_hours"], now), e1["remaining_wall_hours"], when(d2["remaining_gpu_hours"], now),
                     d2["remaining_gpu_hours"],
                     sum(s["tiers"][k]["fits"] - s["tiers"][k]["saved"] for s in (e4s, e6s) for k in ("1", "2")),
                     g2_end.strftime("%Y-%m-%d %H:%M"), last.strftime("%Y-%m-%d %H:%M")))
    print("| %s %s | status | %s | | %s |" % (now.strftime("%Y-%m-%d %H:%M"), zone, what, estimates))
    model = e1.pop("model")
    print(json.dumps({"E1b": {**e1, "k_per_head": model["k"], "probe_k_per_head": model["probe_k"]}, "D2": d2, "D2_fit_cpu_hours": d2_fits,
                      "E4": e4s, "E6": e6s}, indent=1, default=str))


if __name__ == "__main__":
    main()
