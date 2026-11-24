"""E5 supplement, analysis 2: the DepMap download decided at G4 on 2026-11-17 (bioinf2027/GATE.md, top section;
bioinf2027/plans/E5_supplement.md, Section 2).

From the official figshare+ record of "DepMap 24Q4 Public" (article 27993248, DOI 10.25452/figshare.plus.27993248.v1,
CC BY 4.0), two files: CRISPRGeneEffect.csv and Model.csv (the cell-line identifiers, for the K562 value). Nothing from
the DepMap portal. 24Q4 is not the current DepMap release (26Q3, 2026-10-05); the record says so.

The record's metadata is read from the figshare API at run time and must show the expected title, DOI and licence. Each
file is fetched from the record's own download link into a new folder, $IGA_STORAGE_ROOT/bioinf2027/downloads/depmap_24Q4/,
and must be under 1 GB and match the size and MD5 that the record states, or the download stops. The record is written
to bioinf2027/results/E5/supplement/depmap_download.json: source, release, date (UTC), size, SHA-256 and MD5 per file.
Nothing downloaded is opened here.
"""
from __future__ import annotations

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
RECORD = REPO / "bioinf2027" / "results" / "E5" / "supplement" / "depmap_download.json"
ARTICLE = 27993248
API = f"https://api.figshare.com/v2/articles/{ARTICLE}"
EXPECTED = {"title": "DepMap 24Q4 Public", "doi": "10.25452/figshare.plus.27993248.v1", "license": "CC BY 4.0"}
FILES = ("CRISPRGeneEffect.csv", "Model.csv")
LIMIT_BYTES = 1_000_000_000
FOLDER = "depmap_24Q4"
AGENT = {"User-Agent": "research-download/1.0"}


def get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as response:
        return json.loads(response.read().decode())


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
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as response, open(partial, "wb") as handle:
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            handle.write(block)
    partial.replace(target)


def main() -> None:
    article = get_json(API)
    meta = {"title": article.get("title"), "doi": article.get("doi"), "license": (article.get("license") or {}).get("name"),
            "license_url": (article.get("license") or {}).get("url"), "version": article.get("version"),
            "published_date": article.get("published_date"), "url": article.get("url_public_html"), "api": API}
    wrong = {k: (meta[k], v) for k, v in EXPECTED.items() if meta[k] != v}
    if wrong:
        raise SystemExit("the figshare record is not the expected one: %s" % wrong)
    listed = {f["name"]: f for f in article["files"] if f["name"] in FILES}
    if sorted(listed) != sorted(FILES):
        raise SystemExit("the record does not list %s" % sorted(set(FILES) - set(listed)))
    for name, f in listed.items():
        if f["size"] >= LIMIT_BYTES or f.get("is_link_only"):
            raise SystemExit("%s: %d bytes or a link only; not downloaded" % (name, f["size"]))
    folder = config.storage_root() / "bioinf2027" / "downloads" / FOLDER
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit("%s exists and is not empty; nothing is downloaded" % config.portable(folder))
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in FILES:
        f = listed[name]
        started = time.time()
        target = folder / name
        fetch(f["download_url"], target)
        sha, md5, n = digests(target)
        checks = {"bytes_match": n == f["size"], "md5_match": md5 == (f.get("supplied_md5") or f.get("computed_md5")), "under_1_GB": n < LIMIT_BYTES}
        files[name] = {"source": f["download_url"], "figshare_file_id": f["id"], "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bytes": n,
                       "sha256": sha, "md5": md5, "record_md5": f.get("supplied_md5") or f.get("computed_md5"), "checks": checks,
                       "seconds": round(time.time() - started, 1), "path": config.portable(target)}
        failed = [k for k, v in checks.items() if not v]
        print("[depmap] %s: %d bytes, sha256 %s, %s" % (name, n, sha[:12], "FAILED " + ", ".join(failed) if failed else "checks passed"), flush=True)
        if failed:
            write({"record": meta, "files": files, "complete": False})
            raise SystemExit("%s failed its check (%s); stopped" % (name, ", ".join(failed)))
    write({"record": meta, "files": files, "complete": True})


def write(body: dict) -> None:
    record = {"gate": "G4, bioinf2027/GATE.md, top section, decided 2026-11-17 (DepMap release)", "plan": "bioinf2027/plans/E5_supplement.md, Section 2",
              "release": "DepMap 24Q4 Public", "not_the_current_release": "the current DepMap release is 26Q3 (announced 2026-10-05); 24Q4 is the newest "
              "release with an official figshare record, chosen by the author", "portal_used": False, **body}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: make_e5_depmap_download.py")
    main()
