"""Compare the tables and figure data with the checksums stored in docs/OUTPUT_CHECKSUMS.sha256.

Run from the repository root after rebuilding the tables and figures:
  python tools/check_outputs.py
The script exits with code 1 when a file is missing or differs.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "docs" / "OUTPUT_CHECKSUMS.sha256"


def main() -> int:
    bad = []
    lines = [line for line in LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        expected, name = line.split(maxsplit=1)
        path = ROOT / name
        if not path.is_file():
            bad.append(f"missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            bad.append(f"differs: {name}")
    for message in bad:
        print(message)
    print(f"{len(lines) - len(bad)} of {len(lines)} files identical to the stored versions")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
