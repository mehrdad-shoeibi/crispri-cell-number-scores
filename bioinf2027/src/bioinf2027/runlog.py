"""Run records: what was executed, from which commit, in which environment, when.

Every analysis run is launched through :func:`execute`, which streams the
output to a log file under the data root and appends one record to
``bioinf2027/runs/<item>.json``. The record holds the digest of the entry
script and of every module of the pipeline package, taken when the run
starts, and names any of them that changed before it ended. Records hold no
machine-specific location:
paths are written in the portable form of :func:`bioinf2027.config.portable`
and the interpreter is recorded as ``python``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from . import config

__all__ = [
    "DISTRIBUTIONS",
    "append_record",
    "code_digests",
    "describe_command",
    "environment",
    "execute",
    "git_state",
    "pinned_versions",
]

# Distributions whose versions are recorded with every run.
DISTRIBUTIONS = ("numpy", "scipy", "pandas", "pyarrow", "h5py", "scikit-learn", "pytest")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pinned_versions(requirements: Path) -> dict[str, str]:
    """Exact pins from a requirements file, keyed by lower-case distribution name."""
    pins = {}
    for line in requirements.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def environment(requirements: Path | None = None) -> dict:
    """Interpreter and package versions, and whether they match the pinned file."""
    packages = {}
    for name in DISTRIBUTIONS:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    record = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }
    if requirements is not None and requirements.exists():
        pins = pinned_versions(requirements)
        installed = {dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions()}
        mismatched = sorted(
            name for name, version in pins.items() if installed.get(name) != version
        )
        record["requirements_file"] = config.portable(requirements)
        record["requirements_sha256"] = hashlib.sha256(requirements.read_bytes()).hexdigest()
        record["matches_requirements"] = not mismatched
        record["mismatched_distributions"] = mismatched
    return record


def git_state(repo: Path) -> dict:
    """Commit, branch and whether tracked files differ from the commit."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout

    # A status line is two status characters, a space, then the path. The output
    # is not stripped as a whole, because the first status character may be a space.
    status = git("status", "--porcelain", "--untracked-files=no")
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    # Untracked files are listed for the revision folder only, so a run of a script
    # that is not yet committed shows as such in its record.
    loose = git("status", "--porcelain", "--untracked-files=all", "--", "bioinf2027")
    untracked = [line[3:] for line in loose.splitlines() if line.startswith("??")]
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "tracked_files_modified": changed,
        "untracked_files_in_bioinf2027": untracked,
    }


def code_digests(repo: Path) -> dict[str, str]:
    """SHA-256 of every module of the pipeline package, keyed by repository-relative path."""
    package = repo / "bioinf2027" / "src" / "bioinf2027"
    return {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.glob("*.py"))
    }


def describe_command(script: Path, arguments: list[str]) -> str:
    """The command in a form that can be recorded and repeated from the repository root."""
    return " ".join(["python", config.portable(script), *arguments])


def append_record(runs_dir: Path, item: str, record: dict) -> Path:
    """Append one run record to the item's run file, creating it when absent."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{item}.json"
    if path.exists():
        document = json.loads(path.read_text())
    else:
        document = {"item": item, "runs": []}
    document["runs"].append(record)
    path.write_text(json.dumps(document, indent=1) + "\n")
    return path


def execute(item: str, label: str, script: Path, arguments: list[str]) -> dict:
    """Run a repository script, log its output under the data root and record the run."""
    repo = config.repo_root().resolve()
    script = Path(script).resolve()
    script.relative_to(repo)  # only repository scripts are run and recorded

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_dir = config.storage_root() / "bioinf2027" / "logs" / item
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stamp}_{label}.log"

    # The code is fingerprinted before the run starts, so the record describes
    # what was executed even if a file is edited while the run is in progress.
    state = git_state(repo)
    script_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    modules = code_digests(repo)
    started = _now()
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), *arguments],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    finished = _now()

    after = code_digests(repo)
    changed = sorted(path for path in set(modules) | set(after) if modules.get(path) != after.get(path))
    if hashlib.sha256(script.read_bytes()).hexdigest() != script_digest:
        changed.append(config.portable(script))

    record = {
        "label": label,
        "command": describe_command(script, arguments),
        "script_sha256": script_digest,
        "module_sha256": modules,
        "code_changed_during_run": changed,
        "working_directory": "repository root",
        **state,
        "environment": environment(repo / "bioinf2027" / "requirements.txt"),
        "started": started,
        "finished": finished,
        "returncode": returncode,
        "log": config.portable(log_path),
    }
    append_record(repo / "bioinf2027" / "runs", item, record)
    return record
