"""End-to-end tests for ranking_mode + keep_winner_only + build_concurrency.

Hermetic — every test stubs out the heavy paths (workspace, commands,
metric evaluation) so we exercise the explore() main-loop control flow
without git / venv / GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import framework_agent.explorer as ex
from framework_agent.isolation import WorkspacePaths
from framework_agent.models import (
    Candidate,
    CommandResult,
    ExploreRequest,
)


def _request(
    work_dir: Path,
    *,
    ranking_mode: bool = False,
    keep_winner_only: bool = False,
    build_concurrency: int = 1,
    disk_min_free_gb: float | None = 0.0,
    min_throughput_ratio: float = 1.05,
) -> ExploreRequest:
    """Build a minimal request that exercises only the explore() control flow."""
    return ExploreRequest.from_dict({
        "framework": "sglang",
        "repo_url": "https://github.com/x/y.git",
        "work_dir": str(work_dir),
        "baseline": {"throughput": 100.0, "accuracy": 0.9, "completed": "1/1"},
        "thresholds": {
            "min_throughput_ratio": min_throughput_ratio,
            "max_accuracy_drop": 0.05,
        },
        "candidate_refs": ["main"],
        "prepare_candidate_env": False,
        "commands": {
            "build": {"command": "true", "timeout_sec": 5, "required": True},
        },
        "ranking_mode": ranking_mode,
        "keep_winner_only": keep_winner_only,
        "build_concurrency": build_concurrency,
        "disk_min_free_gb": disk_min_free_gb,
    })


def _stub_explore(
    monkeypatch: pytest.MonkeyPatch,
    work_dir: Path,
    *,
    metrics_table: dict[str, tuple[float, float, str]],
    n_candidates: int = 3,
) -> list[Candidate]:
    """Wire up workspace + commands + metric stubs.

    ``metrics_table`` maps candidate.ref to (throughput, accuracy, completed)
    so each test can choose how many winners exist.
    """
    candidates = [
        Candidate(ref=f"PR:{i}", repo="r", source="explicit")
        for i in range(1, n_candidates + 1)
    ]

    def fake_enumerate(_req: ExploreRequest) -> tuple[list[Candidate], list[Any]]:
        return list(candidates), []

    def fake_prepare(
        req: ExploreRequest,
        candidate: Candidate,
        *,
        index: int,
        execute: bool,
    ) -> tuple[WorkspacePaths, dict[str, str]]:
        candidate_dir = work_dir / "candidates" / f"{index:02d}_{candidate.slug}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        worktree = candidate_dir / "worktree"
        venv = candidate_dir / "venv"
        worktree.mkdir(parents=True, exist_ok=True)
        venv.mkdir(parents=True, exist_ok=True)
        ws = WorkspacePaths(candidate_dir, worktree, venv)
        return ws, {}

    def fake_run_command(name: str, command: str, *, cwd: Path, timeout_sec: int) -> CommandResult:
        return CommandResult(name=name, command=command, returncode=0)

    def fake_evaluate(req: ExploreRequest, variables: dict[str, str]) -> tuple[float, float, str]:
        ref = variables["candidate_ref"]
        return metrics_table.get(ref, (0.0, 0.0, ""))

    monkeypatch.setattr(ex, "_enumerate_with_skipped", fake_enumerate)
    monkeypatch.setattr(ex, "_prepare_candidate_workspace_with_artifacts", fake_prepare)
    monkeypatch.setattr(ex, "run_command", fake_run_command)
    monkeypatch.setattr(ex, "_evaluate_candidate", fake_evaluate)
    return candidates


# ranking_mode --------------------------------------------------------------


def test_ranking_mode_runs_every_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ranking_mode=True does NOT short-circuit on first winner."""
    metrics = {
        "PR:1": (200.0, 0.9, "1/1"),
        "PR:2": (300.0, 0.9, "1/1"),
        "PR:3": (250.0, 0.9, "1/1"),
    }
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics)
    req = _request(tmp_path, ranking_mode=True)
    summary = ex.explore(req, execute=True)
    assert len(summary["candidates"]) == 3
    # Highest throughput (PR:2 -> ratio 3.0) ranks first.
    assert summary["candidates"][0]["candidate"]["ref"] == "PR:2"
    assert summary["candidates"][1]["candidate"]["ref"] == "PR:3"
    assert summary["candidates"][2]["candidate"]["ref"] == "PR:1"


def test_legacy_mode_short_circuits_on_first_winner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ranking_mode=False (default) stops after the first winner."""
    metrics = {
        "PR:1": (200.0, 0.9, "1/1"),
        "PR:2": (300.0, 0.9, "1/1"),
        "PR:3": (250.0, 0.9, "1/1"),
    }
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics)
    req = _request(tmp_path, ranking_mode=False)
    summary = ex.explore(req, execute=True)
    # PR:1 wins on the first iteration; PR:2/PR:3 never run.
    assert len(summary["candidates"]) == 1
    assert summary["winner_ref"] == "PR:1"


def test_ranking_mode_failed_candidates_sort_to_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed-bench candidate scores 0 and ranks below survivors."""
    metrics = {
        "PR:1": (50.0, 0.9, "1/1"),
        "PR:2": (None, None, ""),
        "PR:3": (200.0, 0.9, "1/1"),
    }
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics)
    req = _request(tmp_path, ranking_mode=True)
    summary = ex.explore(req, execute=True)
    refs = [c["candidate"]["ref"] for c in summary["candidates"]]
    assert refs[0] == "PR:3"  # highest score
    assert refs[-1] == "PR:2"  # zero score sorts last


# build_concurrency --------------------------------------------------------


def test_build_concurrency_runs_async_fanout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ranking_mode=True + build_concurrency>1 takes the asyncio.gather path."""
    metrics = {
        f"PR:{i}": (100.0 + i, 0.9, "1/1") for i in range(1, 5)
    }
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics, n_candidates=4)
    req = _request(tmp_path, ranking_mode=True, build_concurrency=3)
    summary = ex.explore(req, execute=True)
    assert len(summary["candidates"]) == 4
    assert summary["build_concurrency"] == 3


# keep_winner_only ---------------------------------------------------------


def test_keep_winner_only_removes_losers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Loser worktree+venv removed; winner kept; audit kept everywhere."""
    metrics = {
        "PR:1": (60.0, 0.9, "1/1"),     # loser: ratio 0.6 < 1.05
        "PR:2": (200.0, 0.9, "1/1"),    # winner
        "PR:3": (70.0, 0.9, "1/1"),     # loser
    }
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics)
    monkeypatch.setattr(ex, "prepare_repo_cache", lambda req: tmp_path / "_repos")
    req = _request(tmp_path, ranking_mode=True, keep_winner_only=True)
    summary = ex.explore(req, execute=True)
    assert summary["winner_ref"] == "PR:2"
    for cand in summary["candidates"]:
        worktree = Path(cand["worktree_dir"])
        venv = Path(cand["venv_dir"])
        if cand["winner"]:
            assert worktree.exists()
            assert venv.exists()
        else:
            assert not worktree.exists(), f"loser {cand['candidate']['ref']} worktree should be gone"
            assert not venv.exists()
        # candidate_dir always kept regardless
        assert Path(cand["candidate_dir"]).exists()


def test_keep_winner_only_off_keeps_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default keep_winner_only=False preserves every workspace (legacy)."""
    metrics = {
        "PR:1": (60.0, 0.9, "1/1"),
        "PR:2": (200.0, 0.9, "1/1"),
    }
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics, n_candidates=2)
    req = _request(tmp_path, ranking_mode=True, keep_winner_only=False)
    summary = ex.explore(req, execute=True)
    for cand in summary["candidates"]:
        assert Path(cand["worktree_dir"]).exists()
        assert Path(cand["venv_dir"]).exists()


# disk_preflight integration ------------------------------------------------


def test_disk_preflight_skipped_when_threshold_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """disk_min_free_gb=0 bypasses the preflight even in execute mode."""
    called: dict[str, bool] = {"ran": False}

    def watch(*a, **kw):
        called["ran"] = True

    monkeypatch.setattr(ex, "disk_preflight", watch)
    metrics = {"PR:1": (200.0, 0.9, "1/1")}
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics, n_candidates=1)
    req = _request(tmp_path, disk_min_free_gb=0.0)
    ex.explore(req, execute=True)
    assert called["ran"] is False


def test_disk_preflight_runs_when_threshold_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-zero disk_min_free_gb triggers preflight in execute mode."""
    called: dict[str, Any] = {"args": None}

    def watch(work_dir, n_candidates, *, min_free_gb=None):
        called["args"] = (work_dir, n_candidates, min_free_gb)

    monkeypatch.setattr(ex, "disk_preflight", watch)
    metrics = {"PR:1": (200.0, 0.9, "1/1")}
    _stub_explore(monkeypatch, tmp_path, metrics_table=metrics, n_candidates=1)
    req = _request(tmp_path, disk_min_free_gb=0.001)
    ex.explore(req, execute=True)
    work_dir, n, threshold = called["args"]
    assert n == 1
    assert threshold == 0.001
