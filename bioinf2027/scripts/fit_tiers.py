"""The tiers of the E4, E6 and D2 head fits, as the author decided them on 2026-11-11.

Recorded in bioinf2027/plans/E4.md, Section 16, bioinf2027/plans/E6.md, Section 16, and
bioinf2027/GATE.md (decided block). The categories are the ones priced in
bioinf2027/diag/E4_E6_runtime.json and bioinf2027/diag/E4_E6_runtime_categories.csv (commit 39fd098);
bioinf2027/scripts/estimate_e4_e6_runtime.py and the fit scripts take them from here.

Tier 1, the families: E4's matched forests of z, x, m and [x; m] in G5 and G6 (draws 1 to 3, head seeds
0 to 2) and the corner z fits there; E6's family at 27/788 (gene-disjoint, forest, seeds 0 to 2, with
its validity controls and the monotone positive control) and R(z-); D2's head fits.
Tier 2: fits whose markers attach to family statements (E4: source-mode labels and refits of the family
arms in G5 and G6; E6: gene-fold sensitivity and refits), E6's Section 6, item 6 complete (the paired
inputs) and E4's z - C2 (matched C2 in G5 and G6).
Tier 3: every other listed fit. E4's unread fits (the amendment, E4 plan, Section 16, item 3) are not
run at all.
"""
from __future__ import annotations

FAMILY_ARMS = ("z", "x", "m", "[x; m]")
E6_FAMILY_MODELS = ("z", "x", "m", "[x; m]", "x-", "m-", "[x; m]-", "C1", "[C1; B2]", "[z; B2]")
TIERS = (1, 2, 3)


# --------------------------------------------------------------------------- E4


def e4_unread(d: dict, s) -> str | None:
    """Fits that bioinf2027/scripts/make_e4_stats.py never reads (each traced in GATE.md, Section 3)."""
    in_pair = d["cell"] in ("G5", "G6")
    if s.variant.startswith("refit"):
        if not in_pair:
            return "unread: refits in G1 and G2 (E1b refits G5 and G6 only)"
        if s.representation not in FAMILY_ARMS:
            return "unread: C1 refits"
    if d["kind"] == "matched" and s.label == "source" and (not in_pair or s.representation not in FAMILY_ARMS):
        return "unread: source-mode fits in G1 and G2 and of the masked arms and C1"
    if d["kind"] == "matched" and s.label == "existing" and s.representation not in FAMILY_ARMS:
        return "unread: existing-label C1"
    if d["kind"] == "matched" and s.label == "rebuilt" and s.representation in ("[C1; B2]", "[z; B2]"):
        return "unread: matched [C1; B2] and [z; B2]"
    if d["kind"] == "matched" and s.head == "monotone":
        return "unread: matched label positive control"
    if d["kind"] == "matched" and s.head in ("boosting", "ridge") and s.representation not in FAMILY_ARMS:
        return "unread: secondary heads of arms outside the family"
    return None


def e4_category(d: dict, s) -> str:
    in_pair = d["cell"] in ("G5", "G6")
    unread = e4_unread(d, s)
    if unread:
        return unread
    if d["kind"] == "matched" and in_pair and s.label == "rebuilt" and s.head == "forest" and s.representation in FAMILY_ARMS \
            and s.variant == "all training rows":
        return "family: matched forests"
    if d["kind"] == "corner" and in_pair and s.representation == "z":
        return "family: corner z"
    if s.variant.startswith("refit"):
        return "marker: refits of the family arms"
    if d["kind"] == "matched" and s.label == "source":
        return "marker: source-mode labels of the family arms"
    if s.head != "forest":
        return "secondary heads"
    if d["kind"] == "matched":
        if s.label == "existing":
            return "fourth corner (existing label)"
        if in_pair and s.representation == "C2":
            return "z - C2: matched C2, G5 and G6"
        return "matched, other arms, G5 and G6" if in_pair else "matched, G1 and G2"
    return {"corner": "corner, other arms and G1, G2", "disjoint": "cell-disjoint", "draw-level B2": "draw-level B2",
            "size factor": "size factor", "thinned": "depth-thinned"}[d["kind"]]


def e4_tier(d: dict, s) -> int | None:
    """1, 2 or 3; None for a fit the amendment drops."""
    category = e4_category(d, s)
    if category.startswith("unread"):
        return None
    if category.startswith("family"):
        return 1
    if category.startswith("marker") or category.startswith("z - C2"):
        return 2
    return 3


# --------------------------------------------------------------------------- E6


def e6_category(d: dict, s) -> str:
    plain = not any(k in d for k in ("basis", "assignment", "matched", "draw"))
    fold = d["fit_set"].startswith("fold")
    if d.get("z_minus"):
        return "family: R(z-)"
    if d["endpoint"] == "27/788" and fold and plain and s.label == "rebuilt" and s.variant == "all training rows" and (
            (s.head == "forest" and s.representation in E6_FAMILY_MODELS) or s.head == "monotone"):
        return "family: T, R, R- and validity controls"
    if "assignment" in d:
        return "marker: gene-fold sensitivity"
    if s.variant.startswith("refit"):
        return "marker: refits"
    if "draw" in d:
        return "paired: size-matched arm"
    if "matched" in d:
        return "E4 matched features"
    if d.get("basis") == "e1b" and s.representation in ("x", "m", "[x; m]", "C2"):
        return "paired: E1b's x at 27/788"
    if "basis" in d:
        return "namespace sensitivities, other"
    if s.head != "forest":
        return "secondary heads"
    if d["fit_set"] == "whole screen":
        return "whole-screen fit"
    if s.label == "mean":
        return "mean-across-draws label"
    if fold and s.label == "rebuilt" and s.variant == "all training rows":
        if s.representation == "C2":
            return "paired: C2 at the four endpoints"
        if d["endpoint"] != "27/788" and s.representation in FAMILY_ARMS + ("C1",):
            return "paired: the arms and C1 at the other endpoints"
    return "other endpoints and models"


def e6_tier(d: dict, s) -> int:
    category = e6_category(d, s)
    if category.startswith("family"):
        return 1
    if category.startswith("marker") or category.startswith("paired"):
        return 2
    return 3


# --------------------------------------------------------------------------- selection


def parse_tier(arguments: list) -> tuple[int | None, list]:
    """The value of --tier N (1, 2 or 3) and the remaining arguments; None when absent."""
    if "--tier" not in arguments:
        return None, list(arguments)
    i = arguments.index("--tier")
    if i + 1 >= len(arguments) or arguments[i + 1] not in ("1", "2", "3"):
        raise SystemExit("--tier takes 1, 2 or 3")
    return int(arguments[i + 1]), arguments[:i] + arguments[i + 2:]


def parse_workers(arguments: list, default: int) -> tuple[int, list]:
    """The value of --workers N (1 to 16) and the remaining arguments; the default when absent.

    The number of workers changes no prediction: each fit runs single-threaded with its own seed."""
    if "--workers" not in arguments:
        return default, list(arguments)
    i = arguments.index("--workers")
    if i + 1 >= len(arguments) or not arguments[i + 1].isdigit() or not 1 <= int(arguments[i + 1]) <= 16:
        raise SystemExit("--workers takes 1 to 16")
    return int(arguments[i + 1]), arguments[:i] + arguments[i + 2:]


def comparable(protocol: dict) -> dict:
    """The parts of a fit run's protocol that its predictions depend on (inputs, heads, seeds, code)."""
    keys = ("inputs", "heads", "seeds", "refits", "seeds_of_the_700_control_endpoints", "fitting_code", "smoke")
    return {k: protocol.get(k) for k in keys}


def differences(earlier: dict, current: dict, path: str = "") -> list:
    """Keys whose values differ between two protocols; a key absent from the earlier one is no difference
    (no earlier fit used it, e.g. D2's arms before they existed)."""
    out = []
    for key, value in earlier.items():
        where = f"{path}.{key}" if path else str(key)
        if value is None:
            continue
        if key not in current or current[key] is None:
            out.append(where)
        elif isinstance(value, dict) and isinstance(current[key], dict):
            out.extend(differences(value, current[key], where))
        elif value != current[key]:
            out.append(where)
    return out


# --------------------------------------------------------------------------- the family pass against the full pass


def compare_passes(family_folder, full_folder, pairs: dict) -> dict:
    """Every value of the family pass's tables against the same rows of the full pass's (E4 and E6 plans, Section 16).

    pairs: {family table: (full table, key columns, excluded columns)}. Numbers must be equal exactly (both
    passes compute them from the same predictions with the same bootstrap draws); missing values match
    missing values, numbers and text alike; other columns are compared as text.
    """
    import numpy as np
    import pandas as pd

    out = {"files": {}, "passed": True}
    for name, (full_name, keys, excluded) in pairs.items():
        family = pd.read_csv(family_folder / name)
        full = pd.read_csv(full_folder / full_name)
        columns = [c for c in family.columns if c not in excluded]
        wanted = set(map(tuple, family[keys].astype(str).to_numpy().tolist()))
        right = full[[tuple(k) in wanted for k in full[keys].astype(str).to_numpy().tolist()]]
        repeated = int(right.duplicated(keys).sum())
        merged = family[columns].merge(right[[c for c in columns if c in right.columns]].drop_duplicates(keys), on=keys, how="left",
                                       suffixes=("", " (full)"), indicator=True)
        missing = int((merged["_merge"] != "both").sum()) + repeated
        absent = [c for c in columns if c not in keys and c not in right.columns]
        differing = []
        for c in columns:
            if c in keys or c in absent:
                continue
            a, b = merged[c], merged[f"{c} (full)"]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                x, y = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
                same = (x == y) | (np.isnan(x) & np.isnan(y))
            else:
                same = np.array([(pd.isna(x) and pd.isna(y)) or (not pd.isna(x) and not pd.isna(y) and str(x) == str(y))
                                 for x, y in zip(a.astype(object), b.astype(object))], dtype=bool)
            if not same.all():
                differing.append({"column": c, "rows": int((~same).sum())})
        passed = missing == 0 and not differing and not absent and len(family) > 0
        out["files"][name] = {"rows": int(len(family)), "compared_with": full_name, "rows_missing_from_the_full_pass": missing,
                              "columns_compared": len(columns) - len(keys) - len(absent), "columns_absent_from_the_full_pass": absent,
                              "differing": differing, "excluded": list(excluded), "passed": passed}
        out["passed"] = out["passed"] and passed
    return out
