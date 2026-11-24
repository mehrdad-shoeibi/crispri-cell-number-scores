"""Tests of the run-record helpers."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, runlog  # noqa: E402


def test_pinned_versions_reads_exact_pins_and_ignores_comments(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# pinned\nnumpy==2.4.4\nSciPy==1.17.1  # exact\n\nloose>=1\n")
    assert runlog.pinned_versions(requirements) == {"numpy": "2.4.4", "scipy": "1.17.1"}


def test_environment_reports_versions_without_machine_locations():
    record = runlog.environment()
    assert record["python"].count(".") == 2
    assert "numpy" in record["packages"]
    assert "/" not in json.dumps(record)


def test_environment_flags_a_mismatch_with_the_pinned_file(tmp_path, monkeypatch):
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(root))
    requirements = root / "requirements.txt"
    requirements.write_text("numpy==0.0.0\n")
    record = runlog.environment(requirements)
    assert record["matches_requirements"] is False
    assert record["mismatched_distributions"] == ["numpy"]
    assert record["requirements_file"] == "$IGA_STORAGE_ROOT/requirements.txt"


def test_describe_command_uses_a_repository_relative_script():
    script = config.repo_root() / "bioinf2027" / "scripts" / "make_d1b.py"
    command = runlog.describe_command(script, ["--flag", "1"])
    assert command == "python bioinf2027/scripts/make_d1b.py --flag 1"


def test_append_record_creates_then_extends_the_run_file(tmp_path):
    first = runlog.append_record(tmp_path, "D9", {"label": "a"})
    runlog.append_record(tmp_path, "D9", {"label": "b"})
    document = json.loads(first.read_text())
    assert document["item"] == "D9"
    assert [run["label"] for run in document["runs"]] == ["a", "b"]


def test_execute_refuses_a_script_outside_the_repository(tmp_path, monkeypatch):
    # Both roots are replaced, so the refusal holds wherever the temporary
    # directory lies and nothing can reach the real run records or logs.
    fake_repo = tmp_path / "repo"
    store = tmp_path / "store"
    fake_repo.mkdir()
    store.mkdir()
    monkeypatch.setattr(runlog.config, "repo_root", lambda: fake_repo)
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(store))
    outside = tmp_path / "elsewhere" / "script.py"
    outside.parent.mkdir()
    outside.write_text("print('x')\n")
    with pytest.raises(ValueError):
        runlog.execute("D9", "outside", outside, [])
    assert not (fake_repo / "bioinf2027" / "runs").exists()
    assert not (store / "bioinf2027").exists()


def test_git_state_lists_modified_tracked_files_with_their_full_paths(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@example.org", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "first.txt").write_text("one\n")
    (tmp_path / "second.txt").write_text("two\n")
    git("add", ".")
    git("commit", "-q", "-m", "initial")
    (tmp_path / "folder" / "first.txt").write_text("changed\n")
    (tmp_path / "second.txt").write_text("changed\n")
    (tmp_path / "untracked.txt").write_text("new\n")

    state = runlog.git_state(tmp_path)
    assert state["tracked_files_modified"] == ["folder/first.txt", "second.txt"]
    assert len(state["commit"]) == 40


def _repository_with_package(root: Path, script_text: str) -> tuple[Path, Path]:
    """A committed repository holding one package module and one script."""
    repo = root / "repo"
    package = repo / "bioinf2027" / "src" / "bioinf2027"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n")
    script = repo / "bioinf2027" / "scripts" / "job.py"
    script.parent.mkdir(parents=True)
    script.write_text(script_text)
    for args in (("init", "-q"), ("add", "."), ("commit", "-q", "-m", "initial")):
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@example.org", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )
    return repo, script


def test_execute_records_the_script_and_module_digests_taken_at_launch(tmp_path, monkeypatch):
    repo, script = _repository_with_package(tmp_path, "print('ran')\n")
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(runlog.config, "repo_root", lambda: repo)
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(store))

    record = runlog.execute("D9", "digests", script, [])

    assert record["returncode"] == 0
    assert record["script_sha256"] == hashlib.sha256(b"print('ran')\n").hexdigest()
    assert record["module_sha256"] == {
        "bioinf2027/src/bioinf2027/module.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()
    }
    assert record["code_changed_during_run"] == []
    saved = json.loads((repo / "bioinf2027" / "runs" / "D9.json").read_text())
    assert saved["runs"][0]["module_sha256"] == record["module_sha256"]


def test_execute_names_a_module_edited_while_the_run_was_in_progress(tmp_path, monkeypatch):
    edit = (
        "from pathlib import Path\n"
        "module = Path(__file__).resolve().parents[1] / 'src' / 'bioinf2027' / 'module.py'\n"
        "module.write_text('VALUE = 2\\n')\n"
    )
    repo, script = _repository_with_package(tmp_path, edit)
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(runlog.config, "repo_root", lambda: repo)
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(store))

    record = runlog.execute("D9", "edited", script, [])

    assert record["module_sha256"]["bioinf2027/src/bioinf2027/module.py"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert record["code_changed_during_run"] == ["bioinf2027/src/bioinf2027/module.py"]


def test_git_state_lists_untracked_files_of_the_revision_folder_only(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@example.org", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "bioinf2027" / "scripts").mkdir(parents=True)
    (tmp_path / "bioinf2027" / "tracked.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "initial")
    (tmp_path / "bioinf2027" / "scripts" / "new.py").write_text("y = 2\n")
    (tmp_path / "elsewhere.txt").write_text("z\n")

    state = runlog.git_state(tmp_path)
    assert state["untracked_files_in_bioinf2027"] == ["bioinf2027/scripts/new.py"]
