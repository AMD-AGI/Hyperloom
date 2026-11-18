# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A wheel-installed kernel library is sealed, borrowed and returned like a checkout.

The tests build a site-packages tree and answer ``find_spec`` from it, so the
resolver and the history it lends both run for real; only the interpreter's
view of what is installed is replaced.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import hyperloom.orchestrator.kernel.campaign_baseline as campaign_baseline
import hyperloom.orchestrator.kernel.wheel_repository as wheel_repository
from hyperloom.orchestrator.kernel.campaign_baseline import seal_campaign_baseline
from hyperloom.orchestrator.kernel.wheel_repository import detach_wheel_histories, lent_work_trees
from kernelforge.workflows.rewrite_controller import CAMPAIGN_BRANCH_PREFIX, reclaim_campaign_branch
from kernelforge.workflows.rewrite_controller.worktree import record_campaign_baseline, untracked_paths

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "wheel-test",
    "GIT_AUTHOR_EMAIL": "wheel-test@local",
    "GIT_COMMITTER_NAME": "wheel-test",
    "GIT_COMMITTER_EMAIL": "wheel-test@local",
}

_RESOLVER = campaign_baseline._package_repository
_INSTALLED = wheel_repository._installed_package_dir


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        env={**os.environ, **_GIT_IDENTITY},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _package(site: Path, name: str) -> Path:
    package = site / name
    (package / "ops").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "ops" / "moe.py").write_text("BLOCK_M = 64\n", encoding="utf-8")
    (package / "module_moe.so").write_bytes(b"\x7fELF")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "moe.cpython-312.pyc").write_bytes(b"\0")
    return package


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    packages = {name: _package(site, name) for name in ("aiter", "vllm")}
    monkeypatch.setattr(campaign_baseline, "_package_repository", _RESOLVER)
    monkeypatch.setattr(wheel_repository, "_installed_package_dir", _INSTALLED)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(packages[name] / "__init__.py")) if name in packages else None,
    )
    return packages


def _seal(session_dir: Path, session_id: str = "s1") -> dict:
    return seal_campaign_baseline(
        SimpleNamespace(framework_repo_path=""),
        session_dir=session_dir,
        session_id=session_id,
        macro_cycle=0,
    )


def test_a_wheel_installed_kernel_library_is_sealed_from_a_session_history(
    installed: dict[str, Path], tmp_path: Path
) -> None:
    """The package holds a pointer; the history, and only its sources, live in the session."""
    aiter = installed["aiter"]
    session = tmp_path / "session"

    pins = _seal(session)

    assert (aiter / ".git").is_file()
    assert _git(aiter, "rev-parse", "--absolute-git-dir") == str(
        (session / "wheel_repositories" / "aiter.git").resolve()
    )
    assert _git(aiter, "show", f"{pins[str(aiter)].commit}:ops/moe.py") == "BLOCK_M = 64"
    tracked = _git(aiter, "ls-files").splitlines()
    assert "ops/moe.py" in tracked
    assert "module_moe.so" not in tracked
    assert not [path for path in tracked if "__pycache__" in path]


def test_a_wheel_installed_framework_is_lent_a_history_too(installed: dict[str, Path], tmp_path: Path) -> None:
    vllm = installed["vllm"]

    pins = _seal(tmp_path / "session")

    assert _git(vllm, "show", f"{pins[str(vllm)].commit}:ops/moe.py") == "BLOCK_M = 64"


def test_a_withdrawn_history_still_names_the_tree_it_was_lent(installed: dict[str, Path], tmp_path: Path) -> None:
    """Integration lands on a lent tree after the pointer is gone, and has to know which trees those are."""
    session = tmp_path / "session"
    _seal(session)

    detach_wheel_histories(session)

    assert lent_work_trees(session) == frozenset(installed.values())
    assert not [package for package in installed.values() if (package / ".git").exists()]


def test_the_next_entry_carries_the_session_forward_after_the_pointer_is_withdrawn(
    installed: dict[str, Path], tmp_path: Path
) -> None:
    """What the session kept between entries is sealed on top of the first snapshot."""
    aiter = installed["aiter"]
    session = tmp_path / "session"
    first = _seal(session)[str(aiter)].commit

    detach_wheel_histories(session)
    assert not (aiter / ".git").exists()
    (aiter / "ops" / "moe.py").write_text("BLOCK_M = 128\n", encoding="utf-8")
    second = _seal(session)[str(aiter)].commit

    assert _git(aiter, "show", f"{second}:ops/moe.py") == "BLOCK_M = 128"
    _git(aiter, "merge-base", "--is-ancestor", first, second)


def test_a_pointer_another_session_left_is_not_withdrawn(installed: dict[str, Path], tmp_path: Path) -> None:
    aiter = installed["aiter"]
    _seal(tmp_path / "session-a")

    detach_wheel_histories(tmp_path / "session-b")

    assert (aiter / ".git").is_file()


def test_a_campaign_an_earlier_session_was_killed_in_is_returned_before_the_snapshot(
    installed: dict[str, Path], tmp_path: Path
) -> None:
    """Snapshotting a tree mid-campaign would make an unmeasured rewrite the new baseline."""
    aiter = installed["aiter"]
    base = _seal(tmp_path / "session-a", "a")[str(aiter)].commit
    record_campaign_baseline(
        aiter, base, untracked_paths(aiter), origin_ref=_git(aiter, "rev-parse", "--abbrev-ref", "HEAD")
    )
    _git(aiter, "checkout", "-b", f"{CAMPAIGN_BRANCH_PREFIX}dead")
    (aiter / "ops" / "moe.py").write_text("REWRITTEN_BY_A_DEAD_CAMPAIGN = 1\n", encoding="utf-8")
    _git(aiter, "commit", "-am", "campaign rewrite")

    pins = _seal(tmp_path / "session-b", "b")

    assert _git(aiter, "show", f"{pins[str(aiter)].commit}:ops/moe.py") == "BLOCK_M = 64"
    assert (aiter / "ops" / "moe.py").read_text(encoding="utf-8") == "BLOCK_M = 64\n"
    assert _git(aiter, "rev-parse", "--absolute-git-dir") == str(
        (tmp_path / "session-b" / "wheel_repositories" / "aiter.git").resolve()
    )


def test_a_pointer_to_a_history_that_is_gone_is_replaced(installed: dict[str, Path], tmp_path: Path) -> None:
    aiter = installed["aiter"]
    _seal(tmp_path / "session-a", "a")
    shutil.rmtree(tmp_path / "session-a")

    pins = _seal(tmp_path / "session-b", "b")

    assert str(aiter) in pins
    assert _git(aiter, "show", f"{pins[str(aiter)].commit}:ops/moe.py") == "BLOCK_M = 64"


def test_a_borrowed_package_is_handed_back_through_its_pointer(installed: dict[str, Path], tmp_path: Path) -> None:
    """The borrow record and the lock sit in the history, since ``.git`` here is a file."""
    aiter = installed["aiter"]
    base = _seal(tmp_path / "session")[str(aiter)].commit
    record_campaign_baseline(
        aiter, base, untracked_paths(aiter), origin_ref=_git(aiter, "rev-parse", "--abbrev-ref", "HEAD")
    )
    _git(aiter, "checkout", "-b", f"{CAMPAIGN_BRANCH_PREFIX}op")
    (aiter / "ops" / "moe.py").write_text("REWRITTEN = 1\n", encoding="utf-8")
    _git(aiter, "commit", "-am", "campaign rewrite")

    assert reclaim_campaign_branch(aiter, base) == f"{CAMPAIGN_BRANCH_PREFIX}op"
    assert _git(aiter, "rev-parse", "HEAD").lower() == base
