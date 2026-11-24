"""E2 and E5, the downloads approved at gate G4 (bioinf2027/GATE.md, decided section, items 4 and 7).

Each source goes into its own new folder under $IGA_STORAGE_ROOT/bioinf2027/downloads/; a folder that exists is used
only if every file in it already matches its recorded digest. Source, date (UTC), size, SHA-256 and MD5 are
recorded at download in bioinf2027/results/E2/downloads.json; a published checksum (Hugging Face SHA-256,
Zenodo MD5, Google Cloud Storage MD5) must match or the download stops. Nothing downloaded is opened here: no
archive is extracted and no pickle or checkpoint is loaded.

- scGPT whole-human checkpoint (best_model.pt, vocab.json, args.json; public Google Drive folder
  1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y, no published checksum; sizes compared with bioinf2027/diag/E2_E5_sources.md).
- Stack-Large (bc_large.ckpt and basecount_1000per_15000max.pkl; Hugging Face arcinstitute/Stack-Large).
- scBaseCount human sample metadata, releases 2025-02-25 and 2026-01-12 (GeneFull_Ex50pAS), public bucket.
- GenePT v2 (Zenodo record 10833191, GenePT_emebdding_v2.zip, CC BY 4.0).
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "bioinf2027" / "results" / "E2" / "downloads.json"
HF = "https://huggingface.co/arcinstitute/Stack-Large/resolve/main/"
GCS = "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/scbasecount/{release}/metadata/GeneFull_Ex50pAS/Homo_sapiens/sample_metadata.parquet"
DRIVE = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

SOURCES = {
    "scgpt_whole_human": [
        {"file": "best_model.pt", "url": DRIVE.format(id="14AebJfGOUF047Eg40hk57HCtrb0fyDTm"), "expected_bytes_about": 195.9e6},
        {"file": "vocab.json", "url": DRIVE.format(id="1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC"), "expected_bytes_about": 1.3e6},
        {"file": "args.json", "url": DRIVE.format(id="1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1"), "expected_bytes_about": 1e3},
    ],
    "stack_large": [
        {"file": "bc_large.ckpt", "url": HF + "bc_large.ckpt", "bytes": 2610004146,
         "sha256": "d388c75891fb26618e6e36054c9819b6a82d6f5bc608baac75994e33ddc8bb81"},
        {"file": "basecount_1000per_15000max.pkl", "url": HF + "basecount_1000per_15000max.pkl", "bytes": 925039,
         "sha256": "d8761dfda955b9897d3251798b72361ddd0171ef707119eaf65381bed2d85dcc"},
        {"file": "MODEL_LICENSE.md", "url": HF + "MODEL_LICENSE.md", "bytes": 9863},
    ],
    "scbasecount_2025-02-25": [
        {"file": "sample_metadata.parquet", "url": GCS.format(release="2025-02-25"), "bytes": 532899, "md5_base64": "eVMSjmHyuZsS4u1aFBjrNA=="},
    ],
    "scbasecount_2026-01-12": [
        {"file": "sample_metadata.parquet", "url": GCS.format(release="2026-01-12"), "bytes": 3205158, "md5_base64": "I9joKl5rRCadnbhckS7E3A=="},
    ],
    "genept_v2": [
        {"file": "GenePT_emebdding_v2.zip", "url": "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1",
         "bytes": 574395233, "md5": "3f6ce4317e3a0091978ae5cb8fbf05a3"},
    ],
}


def digests(path: Path) -> tuple[str, str, int]:
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            sha.update(block)
            md5.update(block)
            n += len(block)
    return sha.hexdigest(), md5.hexdigest(), n


def fetch(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "research-download/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as handle:
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            handle.write(block)
    partial.replace(target)


def check(entry: dict, sha: str, md5: str, n: int) -> dict:
    out = {"bytes_match": None, "sha256_match": None, "md5_match": None}
    if "bytes" in entry:
        out["bytes_match"] = n == entry["bytes"]
    if "expected_bytes_about" in entry:
        out["bytes_within_5_percent"] = abs(n - entry["expected_bytes_about"]) <= 0.05 * entry["expected_bytes_about"]
    if "sha256" in entry:
        out["sha256_match"] = sha == entry["sha256"]
    if "md5" in entry:
        out["md5_match"] = md5 == entry["md5"]
    if "md5_base64" in entry:
        out["md5_match"] = base64.b64encode(bytes.fromhex(md5)).decode() == entry["md5_base64"]
    return out


def main(names: list[str]) -> None:
    root = config.storage_root() / "bioinf2027" / "downloads"
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {"gate": "G4, bioinf2027/GATE.md, decided section, items 4 and 7", "downloads": {}}
    for name in names:
        folder = root / name
        known = record["downloads"].get(name, {}).get("files", {})
        if folder.exists() and any(folder.iterdir()):
            if not known:
                raise SystemExit("%s exists, is not empty and has no record; nothing is downloaded" % folder)
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for entry in SOURCES[name]:
            target = folder / entry["file"]
            if target.exists() and entry["file"] in known:
                sha, md5, n = digests(target)
                if sha != known[entry["file"]]["sha256"]:
                    raise SystemExit("%s/%s no longer matches its recorded digest" % (name, entry["file"]))
                files[entry["file"]] = known[entry["file"]]
                continue
            started = time.time()
            fetch(entry["url"], target)
            sha, md5, n = digests(target)
            checks = check(entry, sha, md5, n)
            files[entry["file"]] = {"source": entry["url"], "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bytes": n,
                                    "sha256": sha, "md5": md5, "checks": checks, "seconds": round(time.time() - started, 1),
                                    "path": config.portable(target)}
            failed = [k for k, v in checks.items() if v is False]
            print("[download] %s/%s: %d bytes, sha256 %s, %s" % (name, entry["file"], n, sha[:12], "FAILED " + ", ".join(failed) if failed else "checks passed"), flush=True)
            if failed:
                record["downloads"][name] = {"files": files, "complete": False}
                RECORD.parent.mkdir(parents=True, exist_ok=True)
                RECORD.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
                raise SystemExit("%s/%s failed its check (%s); stopped" % (name, entry["file"], ", ".join(failed)))
        record["downloads"][name] = {"files": files, "complete": True}
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SOURCES)
    unknown = [w for w in wanted if w not in SOURCES]
    if unknown:
        raise SystemExit("unknown source: %s" % unknown)
    main(wanted)
