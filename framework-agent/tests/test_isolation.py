"""Tests for framework_agent.isolation (disk preflight + cleanup)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from framework_agent.isolation import (
    DiskPreflightError,
    WorkspacePaths,
    cleanup_workspace,
    disk_preflight,
)


def test_disk_preflight_passes_when_enough_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 1GB floor is easily satisfied by a normal scratch mount."""
    monkeypatch.delenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", raising=False)
    disk_preflight(tmp_path / "wd", n_candidates=2, min_free_gb=0.001)


def test_disk_preflight_raises_when_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Floor higher than available free GB triggers DiskPreflightError."""
    monkeypatch.delenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", raising=False)
    usage = shutil.disk_usage(str(tmp_path))
    impossible_gb = (usage.free / (1024 ** 3)) + 10_000
    with pytest.raises(DiskPreflightError) as exc:
        disk_preflight(tmp_path / "wd", n_candidates=1, min_free_gb=impossible_gb)
    assert "insufficient disk" in str(exc.value)
    assert "required" in str(exc.value)


def test_disk_preflight_n_candidates_scales_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """required = max(floor, n * per_candidate) — large n breaches a low floor."""
    monkeypatch.delenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", raising=False)
    usage = shutil.disk_usage(str(tmp_path))
    free_gb = usage.free / (1024 ** 3)
    n = int(free_gb / 1.5) + 50
    with pytest.raises(DiskPreflightError):
        disk_preflight(
            tmp_path / "wd",
            n_candidates=n,
            min_free_gb=0.001,
            per_candidate_gb=1.5,
        )


def test_disk_preflight_env_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FRAMEWORK_EXPLORER_DISK_MIN_GB is consulted when explicit is None."""
    monkeypatch.setenv("FRAMEWORK_EXPLORER_DISK_MIN_GB", "999999")
    with pytest.raises(DiskPreflightError):
        disk_preflight(tmp_path / "wd", n_candidates=1)


def test_cleanup_workspace_removes_loser(tmp_path: Path) -> None:
    """Non-winner with keep_winner_only=True has worktree+venv removed."""
    candidate_dir = tmp_path / "candidates" / "01_pr1"
    worktree = candidate_dir / "worktree"
    venv = candidate_dir / "venv"
    worktree.mkdir(parents=True)
    venv.mkdir(parents=True)
    (candidate_dir / "pr.patches").write_text("placeholder")
    (worktree / "src.py").write_text("x = 1")
    ws = WorkspacePaths(candidate_dir, worktree, venv)
    cleanup_workspace(ws, is_winner=False, keep_winner_only=True)
    assert not worktree.exists()
    assert not venv.exists()
    # Audit material stays on disk for reviewers.
    assert (candidate_dir / "pr.patches").exists()


def test_cleanup_workspace_keeps_winner(tmp_path: Path) -> None:
    """Winner is preserved even with keep_winner_only=True."""
    candidate_dir = tmp_path / "candidates" / "01_pr1"
    worktree = candidate_dir / "worktree"
    venv = candidate_dir / "venv"
    worktree.mkdir(parents=True)
    venv.mkdir(parents=True)
    ws = WorkspacePaths(candidate_dir, worktree, venv)
    cleanup_workspace(ws, is_winner=True, keep_winner_only=True)
    assert worktree.exists()
    assert venv.exists()


def test_cleanup_workspace_noop_when_policy_off(tmp_path: Path) -> None:
    """keep_winner_only=False is a no-op even for losers (legacy behaviour)."""
    candidate_dir = tmp_path / "candidates" / "01_pr1"
    worktree = candidate_dir / "worktree"
    venv = candidate_dir / "venv"
    worktree.mkdir(parents=True)
    venv.mkdir(parents=True)
    ws = WorkspacePaths(candidate_dir, worktree, venv)
    cleanup_workspace(ws, is_winner=False, keep_winner_only=False)
    assert worktree.exists()
    assert venv.exists()


def test_cleanup_workspace_swallows_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cleanup is best-effort: shutil.rmtree failure is logged, not raised."""
    candidate_dir = tmp_path / "candidates" / "01_pr1"
    worktree = candidate_dir / "worktree"
    venv = candidate_dir / "venv"
    worktree.mkdir(parents=True)
    venv.mkdir(parents=True)

    def boom(*a, **kw):
        raise OSError("simulated disk error")

    monkeypatch.setattr("framework_agent.isolation.shutil.rmtree", boom)
    ws = WorkspacePaths(candidate_dir, worktree, venv)
    cleanup_workspace(ws, is_winner=False, keep_winner_only=True)
    # No exception escapes; dirs still exist because rmtree was stubbed.
    assert worktree.exists()
