"""E5 supplement 2, Section 5: the HGNC download approved in the answer of 2026-11-17 to the stop of 06:50, item 7 (9)
(bioinf2027/GATE.md, top section; bioinf2027/plans/E5_supplement_2.md, Section 5).

The HGNC complete gene set (hgnc_complete_set.txt) from the official HGNC site: the link on genenames.org's download
pages, which points to the HGNC's public Google Storage bucket. Before anything is downloaded, the licence page is read
and must state the Creative Commons Public Domain (CC0) licence, and the download page must link the file; otherwise
nothing is downloaded and (9) is a limitation. The file goes to a new folder, $IGA_STORAGE_ROOT/bioinf2027/downloads/hgnc/,
and the record to bioinf2027/results/E5/supplement_2/hgnc_download.json: source, the linking page, date (UTC), the
server's Last-Modified, size, SHA-256 and the licence text as read. Nothing downloaded is opened here.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "bioinf2027" / "results" / "E5" / "supplement_2" / "hgnc_download.json"
LICENCE_PAGE = "https://www.genenames.org/about/license/"
DOWNLOAD_PAGE = "https://www.genenames.org/download/archive/"
FILE = "hgnc_complete_set.txt"
SOURCE = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/" + FILE
LICENCE_PHRASE = "Creative Commons Public Domain (CC0) License"
LIMIT_BYTES = 1_000_000_000
FOLDER = "hgnc"
AGENT = {"User-Agent": "research-download/1.0"}


def text_of(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=120) as response:
        page = response.read().decode("utf-8", errors="replace")
    page = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S)
    return page


def plain(page: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page))).strip()


def main() -> None:
    licence = plain(text_of(LICENCE_PAGE))
    at = licence.find(LICENCE_PHRASE)
    if at < 0:
        raise SystemExit("the licence page does not state %r; nothing is downloaded (record (9) as a limitation)" % LICENCE_PHRASE)
    sentence_start = licence.rfind(".", 0, at) + 1
    sentence_end = licence.find(".", at + len(LICENCE_PHRASE)) + 1
    licence_text = licence[sentence_start:sentence_end].strip()
    download_page = text_of(DOWNLOAD_PAGE)
    if ("//storage.googleapis.com/public-download-files/hgnc/tsv/tsv/" + FILE) not in download_page:
        raise SystemExit("the download page does not link %s; nothing is downloaded" % SOURCE)
    folder = config.storage_root() / "bioinf2027" / "downloads" / FOLDER
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit("%s exists and is not empty; nothing is downloaded" % config.portable(folder))
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    target = folder / FILE
    partial = target.with_name(target.name + ".part")
    sha, n = hashlib.sha256(), 0
    with urllib.request.urlopen(urllib.request.Request(SOURCE, headers=AGENT), timeout=120) as response, open(partial, "wb") as handle:
        last_modified = response.headers.get("Last-Modified")
        stated = response.headers.get("Content-Length")
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            handle.write(block)
            sha.update(block)
            n += len(block)
            if n >= LIMIT_BYTES:
                raise SystemExit("more than 1 GB; stopped")
    if stated is not None and int(stated) != n:
        raise SystemExit("received %d bytes, the server stated %s; stopped" % (n, stated))
    partial.replace(target)
    record = {"decision": "the answer of 2026-11-17 to the stop of 06:50, item 7 (9), bioinf2027/GATE.md, top section",
              "plan": "bioinf2027/plans/E5_supplement_2.md, Section 5",
              "source": SOURCE, "linked_from": DOWNLOAD_PAGE, "licence_page": LICENCE_PAGE, "licence_as_stated": licence_text,
              "file": {"name": FILE, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "last_modified": last_modified, "bytes": n,
                       "sha256": sha.hexdigest(), "seconds": round(time.time() - started, 1), "path": config.portable(target)},
              "complete": True}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[hgnc] %s: %d bytes, sha256 %s, last modified %s; licence: %s" % (FILE, n, sha.hexdigest()[:12], last_modified, licence_text), flush=True)


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("usage: make_e5_hgnc_download.py")
    main()
