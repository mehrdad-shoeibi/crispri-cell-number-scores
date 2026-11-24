"""Launch one analysis script and record the run.

Usage, from the repository root, with the pinned environment's interpreter:

    python bioinf2027/scripts/run_item.py --item D1b --label regenerate \
        bioinf2027/scripts/make_d1b.py [script arguments]

The output is streamed to a log under the data root and one record is appended
to bioinf2027/runs/<item>.json: command, commit, environment, start and end
time, return code and the log location.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import runlog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--item", required=True, help="item identifier, for example D1b")
    parser.add_argument("--label", required=True, help="short label for this run")
    parser.add_argument("script", help="repository script to run")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="arguments passed through")
    options = parser.parse_args()

    record = runlog.execute(options.item, options.label, Path(options.script), options.arguments)
    print(
        "\nrun recorded: %s, return code %d, log %s"
        % (record["command"], record["returncode"], record["log"])
    )
    return int(record["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
