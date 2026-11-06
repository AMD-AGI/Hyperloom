# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The multi-branch app's git over a tree the sandboxes write starts no program the tree or its repository's config
names: every case of ``hyperloom_common.git_safety``'s, each a config key class git starts a program from, run through
a worktree's git (:mod:`~hyperloom_multibranch.repo.isolated_git`) instead of ``app_git``, with the same proof that
plain git runs the plant; and the branch worktree's own commands over a signed head, a recursing submodule and a
repository nested in the tree."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom_common.tests import test_git_safety as cases
from hyperloom_common.tests.test_git_safety import unauthorized  # noqa: F401  (a fixture)
from hyperloom_multibranch.specialist import worktree as worktree_module


def _through_the_worktree_s_git(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.CompletedProcess(args, 0, worktree_module._git(tree, *args), "")
    except worktree_module.WorktreeError as error:
        return subprocess.CompletedProcess(args, 1, "", str(error))


@pytest.mark.parametrize(("plant", "expected"), cases._PLANTS)
def test_no_program_the_repository_configures_runs_through_the_worktree_s_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plant: cases._Plant, expected: str | None
) -> None:
    monkeypatch.setattr(cases, "app_git", _through_the_worktree_s_git)

    done = cases._proves(tmp_path, plant)

    if expected is not None:
        assert done.returncode == 0, done.stderr
        assert expected in done.stdout


@pytest.mark.parametrize("key", ["credential.helper", "core.askPass"])
def test_an_http_fetch_through_the_worktree_s_git_starts_no_credential_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unauthorized: str, key: str
) -> None:
    monkeypatch.setattr(cases, "app_git", _through_the_worktree_s_git)

    done = cases._proves(tmp_path, lambda tree: cases._lazy_fetch(tree, unauthorized, (key, str(tree.program))))

    assert done.returncode != 0
