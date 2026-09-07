# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys

import pytest

import kernelforge.loop.editable_repo as editable_repo  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    editable_repo._editable_roots_cached.cache_clear()
    yield
    editable_repo._editable_roots_cached.cache_clear()


@pytest.fixture
def only_these_search_dirs(tmp_path, monkeypatch):
    """Confine the scan to given directories, off the host's real install.

    The scan deliberately looks beyond ``sys.path`` -- a venv's site-packages
    often is not on it -- so isolating it means answering for the prefixes and
    the interpreter path too.
    """

    def _apply(*directories: str) -> None:
        empty = tmp_path / "empty-prefix"
        empty.mkdir(exist_ok=True)
        monkeypatch.setattr(editable_repo.sys, "path", list(directories))
        monkeypatch.setattr(editable_repo.sys, "prefix", str(empty))
        monkeypatch.setattr(editable_repo.sys, "exec_prefix", str(empty))
        monkeypatch.setattr(editable_repo.sys, "base_prefix", str(empty))
        monkeypatch.setattr(editable_repo.sys, "executable", str(empty / "bin" / "python3"))
        monkeypatch.setattr(editable_repo.site, "getsitepackages", lambda: [])
        monkeypatch.setattr(editable_repo.site, "getusersitepackages", lambda: "")
        for name in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            monkeypatch.delenv(name, raising=False)

    return _apply


def test_the_scan_runs_once_however_often_it_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint probe asks this every second for the length of a campaign."""
    calls = {"n": 0}

    def counted() -> tuple[str, ...]:
        calls["n"] += 1
        return ("/one-root",)

    monkeypatch.setattr(editable_repo, "_scan_editable_roots", counted)

    answers = [editable_repo.editable_roots() for _ in range(50)]

    assert calls["n"] == 1
    assert answers == [["/one-root"]] * 50


def test_a_caller_cannot_disturb_the_cached_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(editable_repo, "_scan_editable_roots", lambda: ("/one-root",))

    editable_repo.editable_roots().append("/not-a-real-root")

    assert editable_repo.editable_roots() == ["/one-root"]


def test_needs_inplace_reads_through_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(editable_repo, "_scan_editable_roots", lambda: ("/sgl-workspace/aiter",))

    assert editable_repo.needs_inplace("/sgl-workspace/aiter") is True
    assert editable_repo.needs_inplace("/sgl-workspace/aiter/aiter/ops") is True
    assert editable_repo.needs_inplace("/elsewhere") is False
    assert editable_repo.needs_inplace("") is False


def test_the_scan_finds_a_quoted_path_and_a_finder_mapping(tmp_path, only_these_search_dirs) -> None:
    """The two layouts setuptools writes, plus the bare-path one pip leaves."""
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    bare = tmp_path / "bare-root"
    quoted = tmp_path / "quoted-root"
    mapped = tmp_path / "mapped-root"
    for root in (bare, quoted, mapped):
        root.mkdir()
    (site_dir / "__editable__.bare-1.0.pth").write_text(f"{bare}\n", encoding="utf-8")
    (site_dir / "__editable__.quoted-1.0.pth").write_text(f'x = "{quoted}"\n', encoding="utf-8")
    (site_dir / "__editable__.mapped-1.0.pth").write_text("import __editable___mapped_1_0_finder\n", encoding="utf-8")
    (site_dir / "__editable___mapped_1_0_finder.py").write_text(
        f'MAPPING = {{"mapped": "{mapped}"}}\n', encoding="utf-8"
    )
    (site_dir / "unrelated.pth").write_text(f"{tmp_path / 'ignored'}\n", encoding="utf-8")
    only_these_search_dirs(str(site_dir))

    found = editable_repo._scan_editable_roots()

    assert set(found) == {str(bare.resolve()), str(quoted.resolve()), str(mapped.resolve())}


def test_an_unreadable_search_directory_is_skipped(tmp_path, only_these_search_dirs) -> None:
    """A path entry that is not a directory, or cannot be listed, is not fatal."""
    only_these_search_dirs("", str(tmp_path / "absent"), str(tmp_path))

    assert editable_repo._scan_editable_roots() == ()


def test_a_lock_is_exclusive_and_release_frees_it(tmp_path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    held = editable_repo.acquire_repo_lock(str(repo))
    contender = editable_repo.acquire_repo_lock(str(repo))

    assert held is not None
    assert contender is None
    editable_repo.release_repo_lock(held)
    again = editable_repo.acquire_repo_lock(str(repo))
    assert again is not None
    editable_repo.release_repo_lock(again)


def test_a_repository_with_no_git_directory_cannot_be_locked(tmp_path) -> None:
    """The lock lives inside .git, so a non-checkout has nowhere to keep one."""
    assert editable_repo.acquire_repo_lock(str(tmp_path / "not-a-repo")) is None


def test_releasing_nothing_is_allowed() -> None:
    assert editable_repo.release_repo_lock(None) is None


def test_a_virtualenv_site_packages_off_the_path_is_still_scanned(
    tmp_path,
    only_these_search_dirs,
    monkeypatch,
) -> None:
    """A venv's site-packages routinely is not on sys.path."""
    venv = tmp_path / "venv"
    version = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    site_dir = venv / "lib" / version / "site-packages"
    site_dir.mkdir(parents=True)
    root = tmp_path / "venv-root"
    root.mkdir()
    (site_dir / "__editable__.venv_pkg-1.0.pth").write_text(f"{root}\n", encoding="utf-8")
    only_these_search_dirs()
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))

    assert editable_repo._scan_editable_roots() == (str(root.resolve()),)


def test_releasing_a_lock_twice_over_is_reported_by_the_caller(tmp_path) -> None:
    """Documents today's behaviour: the second release is the caller's error.

    Unreachable from the lanes, which release once in a ``finally``, and left
    as it is rather than swallowed so a double release stays visible.
    """
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    lock = editable_repo.acquire_repo_lock(str(repo))
    assert lock is not None
    editable_repo.release_repo_lock(lock)

    with pytest.raises(ValueError, match="closed file"):
        editable_repo.release_repo_lock(lock)
