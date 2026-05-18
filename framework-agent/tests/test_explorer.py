"""Tests for framework_agent.explorer pure logic + plan-mode path.

Hermetic - no GPU / git / network. The plan-mode test stubs out
``sources.enumerate_candidates`` and the per-candidate audit-material
writer so it exercises explore() control flow without I/O to primus.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import framework_agent.explorer as ex
from framework_agent.models import (
    Baseline,
    Candidate,
    ExploreRequest,
    PrFilter,
    Thresholds,
)


# _winner_decision -------------------------------------------------------


def _req_for_gate(threshold_ratio: float = 1.05, max_drop: float = 0.05) -> ExploreRequest:
    """Build a minimal request usable by _winner_decision tests."""
    return ExploreRequest.from_dict({
        "framework": "sglang",
        "repo_url": "https://github.com/x/y.git",
        "work_dir": "/tmp/x",
        "baseline": {"throughput": 100.0, "accuracy": 0.9, "completed": "1/1"},
        "thresholds": {
            "min_throughput_ratio": threshold_ratio,
            "max_accuracy_drop": max_drop,
        },
    })


def test_winner_decision_pass() -> None:
    """Both gates satisfied -> winner=True."""
    req = _req_for_gate()
    winner, reason = ex._winner_decision(req, throughput=200.0, accuracy=0.9, completed="1/1")
    assert winner is True
    assert "gates passed" in reason


def test_winner_decision_throughput_too_low() -> None:
    """Throughput below baseline*ratio -> rejected with ratio reason."""
    req = _req_for_gate()
    winner, reason = ex._winner_decision(req, throughput=104.0, accuracy=0.9, completed="1/1")
    assert winner is False
    assert "throughput ratio" in reason


def test_winner_decision_accuracy_drop_too_large() -> None:
    """Accuracy drop above the max -> rejected."""
    req = _req_for_gate()
    winner, reason = ex._winner_decision(req, throughput=200.0, accuracy=0.5, completed="1/1")
    assert winner is False
    assert "accuracy drop" in reason


def test_winner_decision_incomplete_benchmark() -> None:
    """completed=='50/100' triggers an incomplete-run rejection."""
    req = _req_for_gate()
    winner, reason = ex._winner_decision(req, throughput=200.0, accuracy=0.9, completed="50/100")
    assert winner is False
    assert "incomplete" in reason


def test_winner_decision_missing_throughput() -> None:
    """Missing throughput is rejected immediately."""
    req = _req_for_gate()
    winner, reason = ex._winner_decision(req, throughput=None, accuracy=0.9, completed="1/1")
    assert winner is False
    assert "missing throughput" in reason


# _passes_filter ---------------------------------------------------------


def test_passes_filter_empty_filter_passes() -> None:
    """An empty PrFilter is a no-op."""
    cand = Candidate(ref="PR:1", repo="r", source="x")
    ok, reason = ex._passes_filter(cand, PrFilter())
    assert ok is True
    assert reason == ""


def test_passes_filter_require_labels() -> None:
    """require_labels rejects candidates missing one of the labels."""
    cand = Candidate(ref="PR:1", repo="r", source="x", labels=("perf",))
    f = PrFilter.from_dict({"require_labels": ["rocm"]})
    ok, reason = ex._passes_filter(cand, f)
    assert ok is False
    assert "required label" in reason


def test_passes_filter_path_filter_needs_changed_files() -> None:
    """include_paths without changed_files metadata fails informatively."""
    cand = Candidate(ref="PR:1", repo="r", source="x", changed_files=())
    f = PrFilter.from_dict({"include_paths": ["python/"]})
    ok, reason = ex._passes_filter(cand, f)
    assert ok is False
    assert "changed_files metadata" in reason


def test_passes_filter_include_paths_hit() -> None:
    """include_paths passes when any changed_file matches a prefix."""
    cand = Candidate(
        ref="PR:1", repo="r", source="x",
        changed_files=("python/sglang/foo.py", "docs/x.md"),
    )
    f = PrFilter.from_dict({"include_paths": ["python/"]})
    ok, _ = ex._passes_filter(cand, f)
    assert ok is True


# _metric_float ----------------------------------------------------------


def test_metric_float_returns_first_numeric_key() -> None:
    """_metric_float returns the first numeric value found among keys."""
    assert ex._metric_float({"a": "no", "b": 12}, "a", "b") == 12.0
    assert ex._metric_float({"a": "no"}, "a", "missing") is None


# explore plan mode ------------------------------------------------------


def test_explore_plan_writes_summary(monkeypatch, tmp_path: Path) -> None:
    """Plan mode returns mode=plan and never invokes shell commands."""
    work_dir = tmp_path / "work"
    req = ExploreRequest.from_dict({
        "framework": "sglang",
        "repo_url": "https://github.com/sgl-project/sglang.git",
        "work_dir": str(work_dir),
        "baseline": {"throughput": 1.0, "accuracy": 0.9, "completed": "1/1"},
        "candidate_refs": ["main"],
        "search_perf_prs": False,
        "prepare_candidate_env": False,
    })

    # Stub: enumerate returns one explicit candidate, no skipped.
    def fake_enum(r):
        return [Candidate(ref="main", repo=r.repo_url, source="explicit")], []

    # Stub: no primus configured so artifact writer is a no-op anyway,
    # but make it explicit to avoid accidental disk I/O.
    monkeypatch.setattr(ex, "_enumerate_with_skipped", fake_enum)
    monkeypatch.setattr(ex, "_write_pr_artifacts", lambda *a, **k: {})

    summary = ex.explore(req, execute=False)
    assert summary["mode"] == "plan"
    assert summary["winner_ref"] is None
    assert summary["promotion_policy"] == "manual_only"
    assert len(summary["candidates"]) == 1
    assert summary["candidates"][0]["status"] == "planned"
