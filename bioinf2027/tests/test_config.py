"""Tests of path resolution and of the portable form used in provenance records."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "store"
    (root / "screen").mkdir(parents=True)
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(root))
    return root


def test_portable_writes_data_root_paths_relative_to_the_variable(data_root):
    recorded = config.portable(data_root / "screen" / "endpoint.csv")
    assert recorded == "$IGA_STORAGE_ROOT/screen/endpoint.csv"


def test_portable_names_the_data_root_itself_by_the_variable(data_root):
    assert config.portable(data_root) == "$IGA_STORAGE_ROOT"


def test_portable_writes_repository_paths_relative_to_the_repository(data_root):
    inside = config.repo_root() / "bioinf2027" / "README.md"
    assert config.portable(inside) == "bioinf2027/README.md"


def test_portable_refuses_a_path_outside_both_roots(data_root, tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr(config, "repo_root", lambda: fake_repo)
    with pytest.raises(ValueError):
        config.portable(tmp_path / "elsewhere" / "file.csv")


def test_portable_never_returns_an_absolute_path(data_root):
    for path in (data_root / "a" / "b.npy", config.repo_root() / "bioinf2027"):
        assert not config.portable(path).startswith("/")


def test_storage_root_fails_loudly_when_the_directory_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(config.STORAGE_VARIABLE, str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        config.storage_root()
