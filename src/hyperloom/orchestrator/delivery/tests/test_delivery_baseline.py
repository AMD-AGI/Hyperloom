# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pre-image is on record before the tree moves, and the record is what is checked."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.bringup import trees


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture()
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("base\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def test_a_wheel_under_an_unrelated_checkout_is_not_that_checkout(checkout: Path) -> None:
    nested = checkout / "site-packages" / "sglang"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text("x\n", encoding="utf-8")

    pinned = trees.resolve_trees([str(nested)])
    # Borrowing the enclosing repo's commit would offer a diff base that never
    # contained these files.
    assert pinned[0].head_commit == ""


def test_a_checkout_records_the_commit_it_was_pinned_at(checkout: Path) -> None:
    pinned = trees.resolve_trees([str(checkout)])[0]
    assert len(pinned.head_commit) == 40
