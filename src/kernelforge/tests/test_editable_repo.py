# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who must be edited in place, and who owns them while it happens.

This module decides whether a campaign gets a private checkout or the user's
live repository, force-checked-out onto a branch and restored path by path
afterwards, so it has the widest blast radius in the rewrite controller. It
also has the quietest failure: every probe below swallows its errors, and a
swallowed error means ``needs_inplace`` answers ``False``, which yields a
private checkout, which is the exact defect the module exists to prevent -- the
campaign edits a copy that is never imported and reports no improvement.
Indistinguishable, from the outside, from an operator with nothing to gain.

The tests build finder files rather than mocking the scan, because the three
layouts are the contract with setuptools and a rewrite that reads only one of
them would still pass an interface test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kernelforge.loop import editable_repo
from kernelforge.loop.editable_repo import (
    acquire_repo_lock,
    editable_roots,
    needs_inplace,
    release_repo_lock,
)


@pytest.fixture
def site_dir(tmp_path: Path, monkeypatch):
    """A site-packages directory that is the only place the scan will look.

    Every source the scan consults has to be redirected, not just ``sys.path``:
    it also derives candidate site-packages from the prefixes and from the
    interpreter's own location, which is how this very checkout's editable
    install would otherwise answer these tests.
    """
    directory = tmp_path / "site-packages"
    directory.mkdir()
    empty = tmp_path / "empty-prefix"
    empty.mkdir()
    # The scan is memoized for the life of the process, so without this every
    # test after the first would be answered from the first one's layout.
    editable_repo._editable_roots_cached.cache_clear()
    monkeypatch.setattr(editable_repo.sys, "path", [str(directory)])
    monkeypatch.setattr(editable_repo.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(editable_repo.site, "getusersitepackages", lambda: "")
    for attribute in ("prefix", "exec_prefix", "base_prefix"):
        monkeypatch.setattr(editable_repo.sys, attribute, str(empty))
    monkeypatch.setattr(editable_repo.sys, "executable", str(empty / "bin" / "python3"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    yield directory
    editable_repo._editable_roots_cached.cache_clear()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    checkout = tmp_path / "live-repo"
    (checkout / ".git").mkdir(parents=True)
    return checkout


# --- the three finder layouts -------------------------------------------------


def test_a_bare_path_pth_is_read(site_dir: Path, repo: Path) -> None:
    (site_dir / "__editable__.demo.pth").write_text(f"{repo}\n", encoding="utf-8")

    assert str(repo) in editable_roots()


def test_a_quoted_path_pth_is_read(site_dir: Path, repo: Path) -> None:
    (site_dir / "__editable__.demo.pth").write_text(f"import sys; sys.path.append('{repo}')\n", encoding="utf-8")

    assert str(repo) in editable_roots()


def test_a_finder_module_mapping_is_read(site_dir: Path, repo: Path) -> None:
    """The setuptools layout: the .pth only imports, the paths are next door."""
    (site_dir / "__editable__.demo.pth").write_text("import __editable___demo_finder\n", encoding="utf-8")
    (site_dir / "__editable___demo_finder.py").write_text(
        f"MAPPING = {{'demo': '{repo}'}}\n",
        encoding="utf-8",
    )

    assert str(repo) in editable_roots()


def test_a_path_that_no_longer_exists_is_not_a_root(site_dir: Path, tmp_path: Path) -> None:
    """A stale finder outlives the checkout it points at."""
    (site_dir / "__editable__.gone.pth").write_text(f"{tmp_path / 'removed'}\n", encoding="utf-8")

    assert editable_roots() == []


def test_an_ordinary_pth_is_ignored(site_dir: Path, repo: Path) -> None:
    """Only PEP 660 finders pin an import to a live directory."""
    (site_dir / "regular.pth").write_text(f"{repo}\n", encoding="utf-8")

    assert editable_roots() == []


def test_an_unreadable_finder_does_not_abort_the_scan(site_dir: Path, repo: Path) -> None:
    """One bad file must not cost the roots that come after it.

    The scan swallows the error, which is right -- but only if what it swallows
    is that one file. A directory wearing a finder's name is the least
    contrived way to make the read fail for real.
    """
    (site_dir / "__editable__.broken.pth").mkdir()
    (site_dir / "__editable__.good.pth").write_text(f"{repo}\n", encoding="utf-8")

    assert str(repo) in editable_roots()


# --- what the verdict covers --------------------------------------------------


def test_the_repository_itself_must_be_edited_in_place(site_dir: Path, repo: Path) -> None:
    (site_dir / "__editable__.demo.pth").write_text(f"{repo}\n", encoding="utf-8")

    assert needs_inplace(str(repo)) is True


def test_a_monorepo_containing_an_editable_package_is_borrowed_whole(site_dir: Path, tmp_path: Path) -> None:
    """The consequence the docstring now states: the scope is the whole tree."""
    monorepo = tmp_path / "monorepo"
    package = monorepo / "packages" / "demo"
    package.mkdir(parents=True)
    (site_dir / "__editable__.demo.pth").write_text(f"{package}\n", encoding="utf-8")

    assert needs_inplace(str(monorepo)) is True


def test_a_package_under_an_editable_root_is_also_in_place(site_dir: Path, tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "sub" / "pkg"
    nested.mkdir(parents=True)
    (site_dir / "__editable__.demo.pth").write_text(f"{root}\n", encoding="utf-8")

    assert needs_inplace(str(nested)) is True


def test_an_unrelated_repository_gets_a_private_checkout(site_dir: Path, repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    (site_dir / "__editable__.demo.pth").write_text(f"{repo}\n", encoding="utf-8")

    assert needs_inplace(str(other)) is False


def test_a_sibling_sharing_a_name_prefix_is_not_a_match(site_dir: Path, tmp_path: Path) -> None:
    """``/a/repo-2`` must not match ``/a/repo``; the separator is load-bearing."""
    root = tmp_path / "repo"
    root.mkdir()
    sibling = tmp_path / "repo-2"
    sibling.mkdir()
    (site_dir / "__editable__.demo.pth").write_text(f"{root}\n", encoding="utf-8")

    assert needs_inplace(str(sibling)) is False


def test_no_repository_named_is_not_in_place(site_dir: Path) -> None:
    assert needs_inplace("") is False


# --- the lock -----------------------------------------------------------------


def test_the_lock_is_exclusive_while_held(repo: Path) -> None:
    """Two campaigns editing one live repository would race, so only one may."""
    first = acquire_repo_lock(str(repo))
    try:
        assert first is not None
        assert acquire_repo_lock(str(repo)) is None
    finally:
        release_repo_lock(first)


def test_the_lock_is_available_again_after_release(repo: Path) -> None:
    release_repo_lock(acquire_repo_lock(str(repo)))

    second = acquire_repo_lock(str(repo))
    try:
        assert second is not None
    finally:
        release_repo_lock(second)


def test_the_lock_file_is_not_world_readable(repo: Path) -> None:
    lock = acquire_repo_lock(str(repo))
    try:
        mode = (repo / ".git" / "forge_inplace.lock").stat().st_mode
        assert mode & 0o077 == 0
    finally:
        release_repo_lock(lock)


def test_a_repository_with_no_git_directory_cannot_be_locked(tmp_path: Path) -> None:
    """The lock lives under .git, so there is nowhere to put it."""
    assert acquire_repo_lock(str(tmp_path / "not-a-checkout")) is None


def test_releasing_nothing_is_allowed(repo: Path) -> None:
    """Callers release in a finally that may not have acquired."""
    release_repo_lock(None)


def test_two_repositories_do_not_share_a_lock(tmp_path: Path) -> None:
    first_repo = tmp_path / "one"
    second_repo = tmp_path / "two"
    for checkout in (first_repo, second_repo):
        (checkout / ".git").mkdir(parents=True)

    first = acquire_repo_lock(str(first_repo))
    second = acquire_repo_lock(str(second_repo))
    try:
        assert first is not None
        assert second is not None
    finally:
        release_repo_lock(first)
        release_repo_lock(second)


def test_the_lock_survives_being_keyed_on_a_relative_path(repo: Path, monkeypatch) -> None:
    """One repository must not yield two locks, which would serialize nothing."""
    monkeypatch.chdir(repo.parent)

    absolute = acquire_repo_lock(str(repo))
    try:
        assert absolute is not None
        assert acquire_repo_lock(os.path.relpath(repo)) is None
    finally:
        release_repo_lock(absolute)
