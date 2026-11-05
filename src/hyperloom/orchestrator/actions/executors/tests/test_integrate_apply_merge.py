# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch authored against a file an earlier KEEP has since changed is merged in, not refused."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import patch_conflict_merge
from hyperloom.orchestrator.actions.executors._integrate_apply import _apply_patch_series
from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.executors.patch_conflict_merge import STRATEGY_LLM, STRATEGY_THREE_WAY

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "merge-test",
    "GIT_AUTHOR_EMAIL": "merge-test@local",
    "GIT_COMMITTER_NAME": "merge-test",
    "GIT_COMMITTER_EMAIL": "merge-test@local",
}

_MODULE = "serving_base.py"

_BASE = """def alpha():
    return 1

def beta():
    return 2
"""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, env={**os.environ, **_GIT_IDENTITY}, capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "framework"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / _MODULE).write_text(_BASE, encoding="utf-8")
    _git(repo, "add", _MODULE)
    _git(repo, "commit", "-m", "base")
    return repo


def _authored(repo: Path, name: str, text: str) -> Path:
    """Diff ``text`` against the original file, as a specialist on its own worktree would."""
    (repo / _MODULE).write_text(text, encoding="utf-8")
    patch = repo.parent / f"{name}.patch"
    patch.write_text(_git(repo, "diff"), encoding="utf-8")
    _git(repo, "checkout", "--", _MODULE)
    return patch


def _keep(repo: Path, patch: Path) -> None:
    _git(repo, "apply", str(patch))
    _git(repo, "commit", "-am", f"keep {patch.stem}")


async def _series(repo: Path, patch: Path) -> tuple[IntegrateAttempt, object]:
    attempt = IntegrateAttempt(task_id="integrate-b", specialist_task_id="spec-b", framework_root=repo)
    head = _git(repo, "rev-parse", "HEAD").strip()
    series = await _apply_patch_series(attempt, [patch], git_head=head, backup_root=repo.parent / "backups")
    return attempt, series


def _resolver_returning(text: str):
    async def resolve(*_: object) -> str:
        return text

    return lambda: resolve


async def test_a_patch_staled_by_a_kept_neighbour_lands_by_three_way(repo: Path) -> None:
    landed = _authored(repo, "a", _BASE.replace("return 1", "return 10"))
    incoming = _authored(repo, "b", _BASE.replace("return 2", "return 20"))
    _keep(repo, landed)
    assert subprocess.run(["git", "-C", str(repo), "apply", "--check", str(incoming)], check=False).returncode != 0

    attempt, series = await _series(repo, incoming)

    assert series.errors == []
    assert attempt.applied == [incoming]
    assert [(row["patch"], row["strategy"]) for row in attempt.merged_patches] == [(str(incoming), STRATEGY_THREE_WAY)]
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    assert "return 10" in merged and "return 20" in merged


async def test_a_resolution_dropping_the_kept_side_is_refused(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    landed = _authored(repo, "a", _BASE.replace("return 1", "return alpha_landed()"))
    incoming = _authored(repo, "b", _BASE.replace("return 1", "return alpha_incoming()"))
    _keep(repo, landed)
    kept = (repo / _MODULE).read_text(encoding="utf-8")
    monkeypatch.setattr(
        patch_conflict_merge, "_conflict_resolver", _resolver_returning("    return alpha_incoming()\n")
    )

    attempt, series = await _series(repo, incoming)

    assert attempt.applied == []
    assert attempt.merged_patches == []
    assert [row["patch"] for row in series.errors] == [str(incoming)]
    stderr = series.errors[0]["stderr"]
    assert "patch does not apply" in stderr
    assert "merge dropped 1 added line(s)" in stderr
    assert series.feedback
    assert (repo / _MODULE).read_text(encoding="utf-8") == kept
    assert _git(repo, "status", "--porcelain") == ""


async def test_a_resolution_keeping_both_sides_lands(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    landed = _authored(repo, "a", _BASE.replace("return 1", "return alpha_landed()"))
    incoming = _authored(repo, "b", _BASE.replace("return 1", "return alpha_incoming()"))
    _keep(repo, landed)
    both = "    if beta():\n        return alpha_incoming()\n    return alpha_landed()\n"
    monkeypatch.setattr(patch_conflict_merge, "_conflict_resolver", _resolver_returning(both))

    attempt, series = await _series(repo, incoming)

    assert series.errors == []
    assert attempt.applied == [incoming]
    assert [row["strategy"] for row in attempt.merged_patches] == [STRATEGY_LLM]
    merged = (repo / _MODULE).read_text(encoding="utf-8")
    assert "return alpha_landed()" in merged and "return alpha_incoming()" in merged
    compile(merged, _MODULE, "exec")


def test_the_integrate_result_names_the_merged_patches(tmp_path: Path) -> None:
    attempt = IntegrateAttempt(task_id="integrate-b")
    attempt.merged_patches.append({"patch": "b.patch", "strategy": STRATEGY_THREE_WAY, "steps": []})

    result = IntegratePatchExecutor(session_dir=tmp_path)._finish_attempt(attempt, {"status": "kept"})

    assert result["patches_merged"] == attempt.merged_patches
