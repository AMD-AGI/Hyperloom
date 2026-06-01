"""Tests for the kernel-optimization attempt summary aggregator.

Pins the contract for ``build_kernel_optimization_summary``:

* Categories: INTEGRATED / KEEP_PENDING / ATTEMPTED_REJECTED /
  IN_FLIGHT / UNATTEMPTED.
* Unattempted sub-reason classification (no_source_file /
  not_reusable_native_kernel / no_recommended_backend /
  below_priority_cutoff).
* Backend ladder harvesting from ``kernel-agent/runs/<session_id>/
  results/<kid>.json`` — present, missing dir, missing file, malformed.
* ``failure_reason_breakdown`` aggregation (ladder_all_failed,
  ladder_partial_no_artifact, ladder_unavailable, etc).
* ``top_takeaways`` highlights the highest-impact missed kernel.
* Deterministic, no LLM, no SVG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.kernel_attempt_summary import (
    CATEGORY_ATTEMPTED_REJECTED,
    CATEGORY_INTEGRATED,
    CATEGORY_IN_FLIGHT,
    CATEGORY_KEEP_PENDING,
    CATEGORY_UNATTEMPTED,
    UNATTEMPTED_NOT_REUSABLE,
    UNATTEMPTED_NO_BACKEND,
    UNATTEMPTED_NO_SOURCE,
    build_kernel_optimization_summary,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _top15_entry(
    kid: str,
    *,
    name: str = "aten::test",
    source_file: str = "",
    gpu_pct: float = 1.0,
    efficiency_pct: float = 50.0,
    bound_type: str = "memory-bound",
    reusable: bool = True,
    backends: list[str] | None = None,
    arithmetic_intensity: float = 1.0,
) -> dict[str, Any]:
    return {
        "kernel_id": kid,
        "name": name,
        "source_file": source_file,
        "gpu_pct": gpu_pct,
        "efficiency_percent": efficiency_pct,
        "bound_type": bound_type,
        "arithmetic_intensity": arithmetic_intensity,
        "reusable_native_kernel": reusable,
        "recommended_backends": (
            list(backends)
            if backends is not None
            else ["geak", "claude", "codex"]
        ),
        "kernel_category": "test",
    }


def _attempt_entry(
    *,
    decision: str = "REVERT",
    status: str = "ok",
    micro: float = 0.0,
    attempts: int = 1,
    partial: int = 0,
    failure: int = 0,
    rejected_reason: str = "",
    source_file: str = "",
    compile_passed: bool | None = None,
    correctness_passed: bool | None = None,
) -> dict[str, Any]:
    return {
        "attempts": attempts,
        "partial_count": partial,
        "failure_count": failure,
        "last_decision": decision,
        "last_status": status,
        "last_micro_speedup": micro,
        "last_source_file": source_file,
        "last_ts": "2026-05-29T12:00:00+00:00",
        "rejected_reason": rejected_reason,
        "compile_passed": compile_passed,
        "correctness_passed": correctness_passed,
        "history": [
            {"decision": decision, "micro": micro, "status": status,
             "ts": "2026-05-29T12:00:00+00:00"},
        ],
    }


def _make_state(
    *,
    session_id: str = "20260529T104050Z",
    top15: list[dict[str, Any]] | None = None,
    attempts: dict[str, dict[str, Any]] | None = None,
    rejected_ids: list[str] | None = None,
    integrated_kids: list[str] | None = None,
    last_kernel_opt: dict[str, Any] | None = None,
) -> SharedState:
    state = SharedState(session_id=session_id, model_name="test/model")
    state.last_trace_analyze = {
        "kernel_roofline_top15": top15 or [],
        "analysis_md_path": "/tmp/analysis.md",
    }
    state.kernel_opt_attempts = dict(attempts or {})
    state.rejected_kernel_ids = list(rejected_ids or [])
    state.optimization_stack = [
        {"action": "integrate", "kernel_id": kid, "ts": "2026-05-29T12:00:00+00:00"}
        for kid in (integrated_kids or [])
    ]
    if last_kernel_opt is not None:
        state.last_kernel_opt = last_kernel_opt
    return state


def _write_backend_results(
    session_dir: Path,
    session_id: str,
    kernel_id: str,
    *,
    backends: list[dict[str, Any]],
) -> None:
    """Write a kernel-agent ``results/<kid>.json`` with given backend rows."""
    results_dir = (
        session_dir / "kernel-agent" / "runs" / session_id / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{kernel_id}.json").write_text(
        json.dumps({"kernel_id": kernel_id, "attempts": backends}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
def test_unattempted_no_source_classifies_vendor_lib_ops(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", name="aten::mm", source_file="",
                             reusable=False, backends=[])],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["unattempted"] == 1
    assert out["totals"]["attempted"] == 0
    row = out["by_kernel"][0]
    assert row["category"] == CATEGORY_UNATTEMPTED
    assert row["unattempted_reason"] == UNATTEMPTED_NO_SOURCE
    assert "vendor-library" in row["unattempted_detail"].lower() \
        or "vendor" in row["unattempted_detail"].lower()


def test_unattempted_not_reusable_distinct_from_no_source(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", source_file="/some/file.py",
                             reusable=False, backends=[])],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["by_kernel"][0]["unattempted_reason"] == UNATTEMPTED_NOT_REUSABLE


def test_unattempted_no_backend_when_reusable_but_empty_recs(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", source_file="/some/file.py",
                             reusable=True, backends=[])],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["by_kernel"][0]["unattempted_reason"] == UNATTEMPTED_NO_BACKEND


def test_attempted_rejected_revert_classifies_correctly(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", gpu_pct=43.9, efficiency_pct=48.4)],
        attempts={"k001": _attempt_entry(
            decision="REVERT", rejected_reason="revert_decision",
            compile_passed=False, correctness_passed=False,
        )},
        rejected_ids=["k001"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["attempted"] == 1
    assert out["totals"]["rejected"] == 1
    row = out["by_kernel"][0]
    assert row["category"] == CATEGORY_ATTEMPTED_REJECTED
    assert row["rejected_reason"] == "revert_decision"
    assert out["rejection_breakdown"]["revert_decision"] == 1


def test_integrated_kernel_classifies_correctly(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(decision="KEEP", micro=1.25)},
        integrated_kids=["k001"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["integrated"] == 1
    assert out["by_kernel"][0]["category"] == CATEGORY_INTEGRATED
    assert "integrated" in out["by_kernel"][0]["summary"].lower()


def test_keep_pending_classifies_correctly(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(decision="KEEP", micro=1.20)},
        last_kernel_opt={"kernel_id": "k001", "decision": "KEEP",
                         "micro_speedup": 1.20, "compile_passed": True,
                         "correctness_passed": True},
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["keep_pending"] == 1
    row = out["by_kernel"][0]
    assert row["category"] == CATEGORY_KEEP_PENDING
    assert row["verification"]["compile_passed"] is True


def test_in_flight_classifies_correctly(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(
            decision="PARTIAL", partial=1, rejected_reason="",
        )},
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["in_flight"] == 1
    assert out["by_kernel"][0]["category"] == CATEGORY_IN_FLIGHT


# ---------------------------------------------------------------------------
# Backend ladder harvesting
# ---------------------------------------------------------------------------
def test_backend_ladder_loaded_from_kernel_agent_results(tmp_path: Path) -> None:
    session_dir = tmp_path
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(
            decision="REVERT", rejected_reason="revert_decision",
            compile_passed=False,
        )},
        rejected_ids=["k001"],
    )
    _write_backend_results(
        session_dir, "sid1", "k001",
        backends=[
            {"backend": "geak", "status": "failed",
             "attempt_id": "geak-1", "optimized_path": ""},
            {"backend": "claude", "status": "failed",
             "attempt_id": "claude-1", "optimized_path": ""},
            {"backend": "codex", "status": "failed",
             "attempt_id": "codex-1", "optimized_path": ""},
        ],
    )
    out = build_kernel_optimization_summary(state, session_dir)
    row = out["by_kernel"][0]
    assert len(row["backend_ladder"]) == 3
    assert all(b["status"] == "failed" for b in row["backend_ladder"])
    assert all(b["produced_artifact"] is False for b in row["backend_ladder"])
    assert row["backend_ladder_unavailable_reason"] == ""
    assert "kernel-agent" in row["kernel_agent_result_path"]
    assert out["failure_reason_breakdown"]["ladder_all_failed"] == 1


def test_backend_ladder_missing_dir_marks_unavailable(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(
            decision="REVERT", rejected_reason="revert_decision",
            compile_passed=False,
        )},
        rejected_ids=["k001"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    row = out["by_kernel"][0]
    assert row["backend_ladder"] == []
    assert row["backend_ladder_unavailable_reason"] == (
        "kernel_agent_results_dir_missing"
    )
    assert out["failure_reason_breakdown"]["ladder_unavailable"] == 1


def test_backend_ladder_malformed_json_falls_back_safely(tmp_path: Path) -> None:
    results_dir = tmp_path / "kernel-agent" / "runs" / "sid1" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "k001.json").write_text("{not valid json", encoding="utf-8")
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(
            decision="REVERT", rejected_reason="revert_decision",
            compile_passed=False,
        )},
        rejected_ids=["k001"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    row = out["by_kernel"][0]
    assert row["backend_ladder"] == []
    assert row["backend_ladder_unavailable_reason"] == "parse_error"


def test_backend_ladder_with_artifact_marks_partial(tmp_path: Path) -> None:
    """One backend produced an artifact but verification still rejected
    (e.g. correctness failed). Should not bucket as ladder_all_failed."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(
            decision="REVERT", rejected_reason="revert_decision",
            compile_passed=True, correctness_passed=False, micro=1.05,
        )},
        rejected_ids=["k001"],
    )
    _write_backend_results(
        tmp_path, "sid1", "k001",
        backends=[
            {"backend": "geak", "status": "completed",
             "attempt_id": "geak-1",
             "optimized_path": "/tmp/optimized.cu"},
            {"backend": "claude", "status": "failed",
             "attempt_id": "claude-1", "optimized_path": ""},
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    breakdown = out["failure_reason_breakdown"]
    assert breakdown["ladder_all_failed"] == 0
    assert (
        breakdown["correctness_failed"] == 1
        or breakdown["speedup_below_threshold"] == 1
    )


# ---------------------------------------------------------------------------
# Top takeaways + glossary
# ---------------------------------------------------------------------------
def test_top_takeaways_highlight_highest_gpu_pct_missed(tmp_path: Path) -> None:
    state = _make_state(
        session_id="sid1",
        top15=[
            _top15_entry("k001", name="aiter::ck_moe_stage1",
                          gpu_pct=43.9, efficiency_pct=48.4),
            _top15_entry("k002", name="aiter::rmsnorm",
                          gpu_pct=1.4, efficiency_pct=15.5),
        ],
        attempts={
            "k001": _attempt_entry(decision="REVERT",
                                    rejected_reason="revert_decision",
                                    compile_passed=False),
            "k002": _attempt_entry(decision="REVERT",
                                    rejected_reason="revert_decision",
                                    compile_passed=False),
        },
        rejected_ids=["k001", "k002"],
    )
    _write_backend_results(
        tmp_path, "sid1", "k001",
        backends=[{"backend": "geak", "status": "failed",
                   "attempt_id": "geak-1", "optimized_path": ""}],
    )
    _write_backend_results(
        tmp_path, "sid1", "k002",
        backends=[{"backend": "geak", "status": "failed",
                   "attempt_id": "geak-1", "optimized_path": ""}],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    joined = " ".join(out["top_takeaways"])
    assert "aiter::ck_moe_stage1" in joined
    assert "43.9" in joined


def test_glossary_present_and_documents_efficiency_pct(tmp_path: Path) -> None:
    state = _make_state(top15=[_top15_entry("k001")])
    out = build_kernel_optimization_summary(state, tmp_path)
    assert "field_glossary" in out
    assert "efficiency_pct" in out["field_glossary"]
    assert "gpu_pct" in out["field_glossary"]
    assert "backend_ladder" in out["field_glossary"]


def test_zero_attempts_session_does_not_crash(tmp_path: Path) -> None:
    state = _make_state(top15=[])
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"] == {
        "top_candidates": 0, "attempted": 0, "integrated": 0,
        "keep_pending": 0, "rejected": 0, "in_flight": 0, "unattempted": 0,
    }
    assert out["by_kernel"] == []
    assert out["top_takeaways"][0].startswith("No kernels were attempted")


def test_attempt_without_top15_still_listed(tmp_path: Path) -> None:
    """Kernels with an attempts ledger but absent from the current
    top15 (e.g. dropped after a roofline refresh) are still surfaced."""
    state = _make_state(
        top15=[],
        attempts={"k_obsolete": _attempt_entry(decision="REVERT",
                                                rejected_reason="revert_decision",
                                                compile_passed=False)},
        rejected_ids=["k_obsolete"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["attempted"] == 1
    assert any(r["kernel_id"] == "k_obsolete" for r in out["by_kernel"])
