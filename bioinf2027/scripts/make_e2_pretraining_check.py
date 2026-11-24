"""E2, Stack's pre-training check (bioinf2027/plans/E2.md, Sections 3 and 15; gate G4, item 4).

Are the Replogle 2022 screens (BioProject PRJNA831566) or the VCC data in scBaseCount, Stack-Large's training data?
Both human sample-metadata releases (2025-02-25 and 2026-01-12; their digests checked against
bioinf2027/results/E2/downloads.json) are read as data. (1) The SRX accessions of PRJNA831566 are fetched from NCBI's
public E-utilities (no key, a generic tool name) and intersected with each release's srx_accession. (2) A text search
of cell_line, perturbation and czi_collection_name for K562, RPE1, H1, CRISPRi, Perturb-seq, Replogle and the Virtual
Cell Challenge. Stack-Large's paper reports 19,978 SRX samples and 149 million cells, which neither release matches, so
both are reported. Output: bioinf2027/results/E2/pretraining_check.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BIOPROJECT = "PRJNA831566"
PATTERNS = {"K562": r"\bK-?562\b", "RPE1": r"\bRPE-?1\b|hTERT[- ]?RPE", "H1": r"\bH1\b(?!-?[0-9])", "CRISPRi": r"CRISPRi|dCas9-KRAB|CRISPR interference",
            "Perturb-seq": r"perturb-?seq", "Replogle": r"replogle", "Virtual Cell Challenge": r"virtual cell challenge|\bVCC\b"}


def get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "research-check/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def bioproject_srx() -> dict:
    query = {"db": "sra", "term": f"{BIOPROJECT}[BioProject]", "retmax": "100000", "retmode": "json", "tool": "research-check"}
    found = json.loads(get(EUTILS + "esearch.fcgi?" + urllib.parse.urlencode(query)))
    ids = found["esearchresult"]["idlist"]
    accessions = set()
    for start in range(0, len(ids), 200):
        part = ids[start:start + 200]
        text = get(EUTILS + "efetch.fcgi?" + urllib.parse.urlencode({"db": "sra", "id": ",".join(part), "rettype": "runinfo", "retmode": "text", "tool": "research-check"}))
        lines = [line for line in text.splitlines() if line.strip()]
        header = lines[0].split(",") if lines else []
        if "Experiment" in header:
            column = header.index("Experiment")
            accessions |= {line.split(",")[column] for line in lines[1:] if not line.startswith("Run,")}
        time.sleep(0.4)
    return {"query": f"{BIOPROJECT}[BioProject] in db sra", "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sra_records": len(ids), "srx": sorted(accessions)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    downloads = json.loads((REPO / "bioinf2027" / "results" / "E2" / "downloads.json").read_text())["downloads"]
    project = bioproject_srx()
    out = {"plan": "bioinf2027/plans/E2.md, Sections 3 and 15 (gate G4, item 4)", "bioproject": {k: v for k, v in project.items() if k != "srx"},
           "bioproject_srx_count": len(project["srx"]), "releases": {}}
    srx = set(project["srx"])
    for release in ("2025-02-25", "2026-01-12"):
        record = downloads[f"scbasecount_{release}"]["files"]["sample_metadata.parquet"]
        path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != record["sha256"]:
            raise SystemExit("the %s sample metadata does not match its download record" % release)
        meta = pd.read_parquet(path)
        hits = meta[meta["srx_accession"].astype(str).isin(srx)]
        fields = [meta[c].astype(object).where(meta[c].notna(), "").astype(str) for c in ("cell_line", "perturbation", "czi_collection_name")]
        text = fields[0] + " | " + fields[1] + " | " + fields[2]  # a missing field must not make the whole row missing
        matches = {}
        for name, pattern in PATTERNS.items():
            m = text.str.contains(pattern, flags=re.IGNORECASE, regex=True)
            matches[name] = {"samples": int(m.sum()), "cells": int(meta.loc[m, "obs_count"].sum()),
                             "examples": meta.loc[m, ["srx_accession", "cell_line", "perturbation"]].head(5).astype(str).values.tolist()}
        both = text.str.contains(PATTERNS["K562"], flags=re.IGNORECASE, regex=True) & text.str.contains(PATTERNS["CRISPRi"] + "|" + PATTERNS["Perturb-seq"], flags=re.IGNORECASE, regex=True)
        out["releases"][release] = {"samples": int(len(meta)), "cells": int(meta["obs_count"].sum()),
                                    "PRJNA831566_srx_present": int(len(hits)), "PRJNA831566_cells_present": int(hits["obs_count"].sum()) if len(hits) else 0,
                                    "PRJNA831566_examples": hits[["srx_accession", "cell_line", "perturbation"]].head(10).astype(str).values.tolist(),
                                    "text_matches": matches, "K562_with_CRISPRi_or_Perturb_seq": int(both.sum())}
    path = REPO / "bioinf2027" / "results" / "E2" / "pretraining_check.json"
    path.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print(json.dumps({r: {k: v for k, v in d.items() if k in ("samples", "cells", "PRJNA831566_srx_present", "K562_with_CRISPRi_or_Perturb_seq")}
                      for r, d in out["releases"].items()} | {"bioproject_srx": len(srx)}), flush=True)


if __name__ == "__main__":
    main()
