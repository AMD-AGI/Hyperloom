# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A patch authored against a file an earlier KEEP has since changed is merged in, not refused."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import patch_conflict_merge
from hyperloom.orchestrator.actions.executors._integrate_apply import _apply_patch_series
from hyperloom.orchestrator.actions.executors._integrate_attempt import IntegrateAttempt
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.executors.patch_conflict_merge import STRATEGY_THREE_WAY
from hyperloom.orchestrator.tests._helpers import diff_against_head, git, init_git_repo

_MODULE = "serving_base.py"

_BASE = """def alpha():
    return 1

def beta():
    return 2
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    init_git_repo(tmp_path / "framework", seed_file=_MODULE, seed_text=_BASE)
    return tmp_path / "framework"


def _authored(repo: Path, name: str, text: str) -> Path:
    """Diff ``text`` against the original file, as a specialist on its own worktree would."""
    patch = repo.parent / f"{name}.patch"
    patch.write_text(diff_against_head(repo, _MODULE, text), encoding="utf-8")
    return patch


def _keep(repo: Path, patch: Path) -> None:
    git(repo, "apply", str(patch))
    git(repo, "commit", "-am", f"keep {patch.stem}")


async def _series(repo: Path, patch: Path) -> tuple[IntegrateAttempt, object]:
    attempt = IntegrateAttempt(task_id="integrate-b", specialist_task_id="spec-b", framework_root=repo)
    head = git(repo, "rev-parse", "HEAD")
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


async def test_an_overlapping_edit_goes_back_to_its_author(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    landed = _authored(repo, "a", _BASE.replace("return 1", "return alpha_landed()"))
    incoming = _authored(repo, "b", _BASE.replace("return 1", "return alpha_incoming()"))
    _keep(repo, landed)
    kept = (repo / _MODULE).read_text(encoding="utf-8")
    calls: list[object] = []

    def resolver() -> object:
        calls.append("resolver")
        return _resolver_returning("    return alpha_incoming()\n    return alpha_landed()\n")()

    monkeypatch.setattr(patch_conflict_merge, "_conflict_resolver", resolver)

    attempt, series = await _series(repo, incoming)

    assert calls == []
    assert attempt.applied == []
    assert attempt.merged_patches == []
    assert [row["patch"] for row in series.errors] == [str(incoming)]
    stderr = series.errors[0]["stderr"]
    assert "patch does not apply" in stderr
    assert "three-way left conflicts" in stderr
    assert series.feedback
    assert (repo / _MODULE).read_text(encoding="utf-8") == kept
    assert git(repo, "status", "--porcelain") == ""


def test_the_integrate_result_names_the_merged_patches(tmp_path: Path) -> None:
    attempt = IntegrateAttempt(task_id="integrate-b")
    attempt.merged_patches.append({"patch": "b.patch", "strategy": STRATEGY_THREE_WAY, "steps": []})

    result = IntegratePatchExecutor(session_dir=tmp_path)._finish_attempt(attempt, {"status": "kept"})

    assert result["patches_merged"] == attempt.merged_patches
