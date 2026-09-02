# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the kernel-optimization attempt summary aggregator ``build_kernel_optimization_summary``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from hyperloom.orchestrator.kernel.attempt_summary import (
    CATEGORY_ATTEMPTED_REJECTED,
    CATEGORY_INTEGRATED,
    CATEGORY_IN_FLIGHT,
    CATEGORY_KEEP_PENDING,
    CATEGORY_UNATTEMPTED,
    OUTCOME_SKIP,
    UNATTEMPTED_BELOW_MIN_GPU_PCT,
    UNATTEMPTED_NOT_REUSABLE,
    UNATTEMPTED_NO_BACKEND,
    UNATTEMPTED_NO_SOURCE,
    build_kernel_optimization_summary,
)
from hyperloom.orchestrator.state.shared_state import SharedState


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
        "recommended_backends": (list(backends) if backends is not None else ["geak"]),
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
            {"decision": decision, "micro": micro, "status": status, "ts": "2026-05-29T12:00:00+00:00"},
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
        {"action": "integrate", "kernel_id": kid, "ts": "2026-05-29T12:00:00+00:00"} for kid in (integrated_kids or [])
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
    """Write a kernel-agent ``results/<kid>.json`` with the given backend rows."""
    results_dir = session_dir / "kernel-agent" / "runs" / session_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{kernel_id}.json").write_text(
        json.dumps({"kernel_id": kernel_id, "attempts": backends}),
        encoding="utf-8",
    )


def test_unattempted_no_source_classifies_vendor_lib_ops(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", name="aten::mm", source_file="", reusable=False, backends=[])],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["unattempted"] == 1
    assert out["totals"]["attempted"] == 0
    row = out["by_kernel"][0]
    assert row["category"] == CATEGORY_UNATTEMPTED
    assert row["unattempted_reason"] == UNATTEMPTED_NO_SOURCE
    assert "vendor-library" in row["unattempted_detail"].lower() or "vendor" in row["unattempted_detail"].lower()


def test_unattempted_not_reusable_distinct_from_no_source(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", source_file="/some/file.py", reusable=False, backends=[])],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["by_kernel"][0]["unattempted_reason"] == UNATTEMPTED_NOT_REUSABLE


def test_unattempted_no_backend_when_reusable_but_empty_recs(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", source_file="/some/file.py", reusable=True, backends=[])],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["by_kernel"][0]["unattempted_reason"] == UNATTEMPTED_NO_BACKEND


def test_a_kernel_the_filter_skipped_reads_as_unattempted(tmp_path: Path) -> None:
    """A dispatch gate is not a failure, and the summary must not call it one.

    The classification keys on the presence of a ledger row, so a row written
    for a kernel no backend ever saw lands in ``attempted`` with no decision --
    IN_FLIGHT, whose empty backend ladder rolls up to ``fail``. Recording the
    skip is what has to be avoided; this pins the report the absence produces.
    """
    from hyperloom.orchestrator.kernel.request_handlers import record_kernel_opt

    state = _make_state(
        top15=[_top15_entry("k001", source_file="/pkg/aiter/gqa.py", gpu_pct=1.0)],
    )
    record_kernel_opt(
        state,
        {"status": "skipped", "reason": "below_min_gpu_pct=5.0", "kernel_id": "k001"},
    )

    out = build_kernel_optimization_summary(state, tmp_path)

    assert out["totals"]["attempted"] == 0
    assert out["totals"]["unattempted"] == 1
    row = out["by_kernel"][0]
    assert row["category"] == CATEGORY_UNATTEMPTED
    assert row["unattempted_reason"] == UNATTEMPTED_BELOW_MIN_GPU_PCT
    assert row["outcome_class"] == OUTCOME_SKIP


def test_attempted_rejected_revert_classifies_correctly(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001", gpu_pct=43.9, efficiency_pct=48.4)],
        attempts={
            "k001": _attempt_entry(
                decision="REVERT",
                rejected_reason="revert_decision",
                compile_passed=False,
                correctness_passed=False,
            )
        },
        rejected_ids=["k001"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["attempted"] == 1
    assert out["totals"]["rejected"] == 1
    row = out["by_kernel"][0]
    assert row["category"] == CATEGORY_ATTEMPTED_REJECTED
    assert row["rejected_reason"] == "revert_decision"
    assert out["rejection_breakdown"]["revert_decision"] == 1


def test_ledger_only_rejection_reaches_the_breakdown(tmp_path: Path) -> None:
    """A rejected kernel that never made top15 must be in both the total and the
    per-reason split, or ``totals.rejected`` and ``rejection_breakdown`` disagree."""
    state = _make_state(
        top15=[_top15_entry("k001")],
        attempts={
            "k001": _attempt_entry(decision="REVERT", rejected_reason="revert_decision"),
            # Never surfaced in top15: only the ledger knows about it.
            "k002": _attempt_entry(decision="REVERT", rejected_reason="max_failures_3"),
        },
        rejected_ids=["k001", "k002"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["rejected"] == 2
    assert sum(out["rejection_breakdown"].values()) == out["totals"]["rejected"]
    assert out["rejection_breakdown"]["revert_decision"] == 1
    assert out["rejection_breakdown"]["max_failures_without_keep"] == 1


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
        last_kernel_opt={
            "kernel_id": "k001",
            "decision": "KEEP",
            "micro_speedup": 1.20,
            "compile_passed": True,
            "correctness_passed": True,
        },
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["keep_pending"] == 1
    row = out["by_kernel"][0]
    assert row["category"] == CATEGORY_KEEP_PENDING
    assert row["verification"]["compile_passed"] is True


def test_in_flight_classifies_correctly(tmp_path: Path) -> None:
    state = _make_state(
        top15=[_top15_entry("k001")],
        attempts={
            "k001": _attempt_entry(
                decision="PARTIAL",
                partial=1,
                rejected_reason="",
            )
        },
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["in_flight"] == 1
    assert out["by_kernel"][0]["category"] == CATEGORY_IN_FLIGHT


def test_backend_ladder_loaded_from_kernel_agent_results(tmp_path: Path) -> None:
    session_dir = tmp_path
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={
            "k001": _attempt_entry(
                decision="REVERT",
                rejected_reason="revert_decision",
                compile_passed=False,
            )
        },
        rejected_ids=["k001"],
    )
    _write_backend_results(
        session_dir,
        "sid1",
        "k001",
        backends=[
            {"backend": "forge", "status": "failed", "attempt_id": "forge-1", "optimized_path": ""},
            {"backend": "claude", "status": "failed", "attempt_id": "claude-1", "optimized_path": ""},
            {"backend": "codex", "status": "failed", "attempt_id": "codex-1", "optimized_path": ""},
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
        attempts={
            "k001": _attempt_entry(
                decision="REVERT",
                rejected_reason="revert_decision",
                compile_passed=False,
            )
        },
        rejected_ids=["k001"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    row = out["by_kernel"][0]
    assert row["backend_ladder"] == []
    assert row["backend_ladder_unavailable_reason"] == ("kernel_agent_results_dir_missing")
    assert out["failure_reason_breakdown"]["ladder_unavailable"] == 1


def test_backend_ladder_malformed_json_falls_back_safely(tmp_path: Path) -> None:
    results_dir = tmp_path / "kernel-agent" / "runs" / "sid1" / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "k001.json").write_text("{not valid json", encoding="utf-8")
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={
            "k001": _attempt_entry(
                decision="REVERT",
                rejected_reason="revert_decision",
                compile_passed=False,
            )
        },
        rejected_ids=["k001"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    row = out["by_kernel"][0]
    assert row["backend_ladder"] == []
    assert row["backend_ladder_unavailable_reason"] == "parse_error"


def test_backend_ladder_with_artifact_marks_partial(tmp_path: Path) -> None:
    """One backend produced an artifact but verification rejected; must not bucket as ladder_all_failed."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={
            "k001": _attempt_entry(
                decision="REVERT",
                rejected_reason="revert_decision",
                compile_passed=True,
                correctness_passed=False,
                micro=1.05,
            )
        },
        rejected_ids=["k001"],
    )
    _write_backend_results(
        tmp_path,
        "sid1",
        "k001",
        backends=[
            {"backend": "forge", "status": "completed", "attempt_id": "forge-1", "optimized_path": "/tmp/optimized.cu"},
            {"backend": "claude", "status": "failed", "attempt_id": "claude-1", "optimized_path": ""},
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    breakdown = out["failure_reason_breakdown"]
    assert breakdown["ladder_all_failed"] == 0
    assert breakdown["correctness_failed"] == 1 or breakdown["speedup_below_threshold"] == 1


def test_top_takeaways_highlight_highest_gpu_pct_missed(tmp_path: Path) -> None:
    state = _make_state(
        session_id="sid1",
        top15=[
            _top15_entry("k001", name="aiter::ck_moe_stage1", gpu_pct=43.9, efficiency_pct=48.4),
            _top15_entry("k002", name="aiter::rmsnorm", gpu_pct=1.4, efficiency_pct=15.5),
        ],
        attempts={
            "k001": _attempt_entry(decision="REVERT", rejected_reason="revert_decision", compile_passed=False),
            "k002": _attempt_entry(decision="REVERT", rejected_reason="revert_decision", compile_passed=False),
        },
        rejected_ids=["k001", "k002"],
    )
    _write_backend_results(
        tmp_path,
        "sid1",
        "k001",
        backends=[{"backend": "forge", "status": "failed", "attempt_id": "forge-1", "optimized_path": ""}],
    )
    _write_backend_results(
        tmp_path,
        "sid1",
        "k002",
        backends=[{"backend": "forge", "status": "failed", "attempt_id": "forge-1", "optimized_path": ""}],
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


def test_dispatch_skip_reason_surfaced_when_present(tmp_path: Path) -> None:
    """A no_eligible_kernels dispatch skip rides through to the summary."""
    state = _make_state(top15=[_top15_entry("k001")])
    state.last_kernel_opt_dispatch_skip = {
        "reason": "no_eligible_kernels",
        "kernels_considered": 5,
        "message": "no eligible kernels to optimize (...)",
        "ts": "2026-06-22T00:00:00+00:00",
    }
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["dispatch_skip_reason"]["reason"] == "no_eligible_kernels"
    assert out["dispatch_skip_reason"]["kernels_considered"] == 5


def test_dispatch_skip_reason_empty_by_default(tmp_path: Path) -> None:
    """No dispatch skip => empty dict, never a failure marker."""
    state = _make_state(top15=[_top15_entry("k001")])
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["dispatch_skip_reason"] == {}


def test_takeaway_names_the_dispatch_skip_instead_of_guessing(tmp_path: Path) -> None:
    """With nothing attempted, the takeaway must state the recorded reason.

    The old sentence offered two guesses -- disabled, or nothing qualified --
    and so could not express the case that actually happened: the candidate
    table was never produced, so kernel_opt was never asked for. A reader
    holding six zero buckets and that sentence concluded the workload had no
    headroom.
    """
    state = _make_state(top15=[])
    state.last_kernel_opt_dispatch_skip = {
        "reason": "no_candidate_table",
        "trace_analyze_empty": True,
        "roofline_failure_streak": 3,
        "ts": "2026-09-01T00:00:00+00:00",
    }
    out = build_kernel_optimization_summary(state, tmp_path)
    joined = " ".join(out["top_takeaways"])
    assert "no_candidate_table" in joined
    assert "no candidates qualified" not in joined


def test_takeaway_does_not_claim_a_dispatch_never_happened(tmp_path: Path) -> None:
    """``no_eligible_kernels`` is written by a dispatch that did happen.

    The two writers of this field disagree about what the skip was: the phase
    records declining to ask, while this reason records a dispatch whose queue
    came back empty. A sentence built around "never dispatched" is false for
    the second, and a confident false statement is worse than the hedge it
    replaced.
    """
    state = _make_state(top15=[])
    state.last_kernel_opt_dispatch_skip = {
        "reason": "no_eligible_kernels",
        "kernels_considered": 7,
        "ts": "2026-09-01T00:00:00+00:00",
    }
    out = build_kernel_optimization_summary(state, tmp_path)
    joined = " ".join(out["top_takeaways"])
    assert "no_eligible_kernels" in joined
    assert "never dispatched" not in joined


def test_takeaway_keeps_the_generic_line_without_a_recorded_reason(tmp_path: Path) -> None:
    """No breadcrumb => the original wording stays (nothing to name)."""
    state = _make_state(top15=[])
    out = build_kernel_optimization_summary(state, tmp_path)
    joined = " ".join(out["top_takeaways"])
    assert "No kernels were attempted" in joined
    assert "no candidates qualified" in joined


def test_zero_attempts_session_does_not_crash(tmp_path: Path) -> None:
    state = _make_state(top15=[])
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"] == {
        "top_candidates": 0,
        "attempted": 0,
        "integrated": 0,
        "keep_pending": 0,
        "rejected": 0,
        "in_flight": 0,
        "unattempted": 0,
    }
    assert out["by_kernel"] == []
    assert out["top_takeaways"][0].startswith("No kernels were attempted")


def test_attempt_without_top15_still_listed(tmp_path: Path) -> None:
    """Kernels with an attempts ledger but absent from the current top15 are still surfaced."""
    state = _make_state(
        top15=[],
        attempts={
            "k_obsolete": _attempt_entry(decision="REVERT", rejected_reason="revert_decision", compile_passed=False)
        },
        rejected_ids=["k_obsolete"],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    assert out["totals"]["attempted"] == 1
    assert any(r["kernel_id"] == "k_obsolete" for r in out["by_kernel"])


def _write_full_kernel_result(
    session_dir: Path,
    session_id: str,
    kernel_id: str,
    *,
    attempts: list[dict[str, Any]],
    verification: dict[str, Any] | None = None,
) -> None:
    """Like _write_backend_results but also supports the top-level verification block."""
    results_dir = session_dir / "kernel-agent" / "runs" / session_id / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {"kernel_id": kernel_id, "attempts": attempts}
    if verification is not None:
        body["verification"] = verification
    (results_dir / f"{kernel_id}.json").write_text(
        json.dumps(body),
        encoding="utf-8",
    )


def test_produced_artifact_excludes_stdout_log_path(tmp_path: Path) -> None:
    """A *_stdout.log optimized_path is a log dump, not an artifact; produced_artifact must be False."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001", gpu_pct=50.0)],
        attempts={"k001": _attempt_entry(decision="REVERT", rejected_reason="max_failures", compile_passed=False)},
        rejected_ids=["k001"],
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k001",
        attempts=[
            {
                "backend": "forge",
                "status": "failed",
                "attempt_id": "g1",
                "optimized_path": "/workspace/optimized/forge-78bd_stdout.log",
                "returncode": 1,
                "elapsed_s": 213.5,
            }
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    ladder = out["by_kernel"][0]["backend_ladder"]
    assert ladder[0]["produced_artifact"] is False, f"stdout log should not count as artifact: {ladder[0]}"


def test_backend_ladder_elapsed_uses_elapsed_s_field(tmp_path: Path) -> None:
    """The loader reads ``elapsed_s`` so elapsed_sec lands in the ladder row."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(decision="REVERT", rejected_reason="max_failures", compile_passed=False)},
        rejected_ids=["k001"],
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k001",
        attempts=[
            {
                "backend": "forge",
                "status": "failed",
                "attempt_id": "g1",
                "optimized_path": "/workspace/optimized/v1.cu",
                "returncode": 1,
                "elapsed_s": 213.5,
            }
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    ladder = out["by_kernel"][0]["backend_ladder"]
    assert ladder[0].get("elapsed_sec") == 213.5


def test_backend_ladder_classifies_timeout(tmp_path: Path) -> None:
    """A 'Timed out after 480s' failure surfaces error_class='timeout' + a concise error_message."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(decision="REVERT", rejected_reason="max_failures", compile_passed=False)},
        rejected_ids=["k001"],
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k001",
        attempts=[
            {
                "backend": "claude",
                "status": "failed",
                "attempt_id": "c1",
                "optimized_path": "/workspace/optimized/c1_stdout.log",
                "returncode": 1,
                "elapsed_s": 483.4,
                "stdout_tail": (
                    "status: running -> failed\n"
                    '{"task_id": "abc",\n  "status": "failed",\n'
                    '  "error_message": "Timed out after 480s",\n'
                    '  "partial_outputs": []}\n'
                ),
            }
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    row = out["by_kernel"][0]["backend_ladder"][0]
    assert row.get("error_class") == "timeout"
    assert "480s" in (row.get("error_message") or "")


def test_backend_ladder_classifies_preprocess_failed(tmp_path: Path) -> None:
    """'success=False, errors=N' maps to error_class='preprocess_failed'."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(decision="REVERT", rejected_reason="max_failures", compile_passed=False)},
        rejected_ids=["k001"],
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k001",
        attempts=[
            {
                "backend": "forge",
                "status": "failed",
                "attempt_id": "g1",
                "optimized_path": "/workspace/optimized/g1_stdout.log",
                "returncode": 1,
                "elapsed_s": 213.5,
                "stdout_tail": (
                    "minisweagent.run.preprocess_v3.adapter: INFO: v3 "
                    "preprocess completed in 184.0s (success=False, errors=1)\n"
                    "minisweagent.run.mini: INFO:  starting\n"
                    "minisweagent.run.mini: INFO:  completed: ran\n"
                ),
            }
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    row = out["by_kernel"][0]["backend_ladder"][0]
    assert row.get("error_class") == "preprocess_failed"
    assert "preprocess" in (row.get("error_message") or "").lower()


def test_backend_ladder_preprocess_failed_with_line_wrapped_stdout(
    tmp_path: Path,
) -> None:
    """The classifier's regex tolerates line-wrapped stdout where the preprocess fragments span lines."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={"k001": _attempt_entry(decision="REVERT", rejected_reason="max_failures", compile_passed=False)},
        rejected_ids=["k001"],
    )
    wrapped_stdout = (
        "  > minisweagent.run.preprocess_v3.adapter: INFO: v3 "
        "preprocess completed in 108.7s \n"
        "  > (success=False, errors=1)                          \n"
        "  > minisweagent.run.mini: INFO:  starting\n"
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k001",
        attempts=[
            {
                "backend": "forge",
                "status": "failed",
                "attempt_id": "g1",
                "optimized_path": "/workspace/optimized/g1_stdout.log",
                "returncode": 1,
                "elapsed_s": 108.7,
                "stdout_tail": wrapped_stdout,
            }
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    row = out["by_kernel"][0]["backend_ladder"][0]
    assert row.get("error_class") == "preprocess_failed", f"wrapped log should still classify: {row}"


def test_render_attempted_row_pulls_verification_from_result_file(
    tmp_path: Path,
) -> None:
    """compile_passed / correctness_passed come from the detail file's verification block when the ledger lacks them."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001")],
        attempts={
            "k001": _attempt_entry(
                decision="REVERT", rejected_reason="max_failures", compile_passed=None, correctness_passed=None
            )
        },
        rejected_ids=["k001"],
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k001",
        attempts=[
            {
                "backend": "codex",
                "status": "partial",
                "attempt_id": "x1",
                "optimized_path": "/workspace/optimized/v1.cu",
                "returncode": 1,
                "elapsed_s": 483.3,
            }
        ],
        verification={
            "compile_passed": True,
            "correctness_passed": False,
            "correctness_source": "missing",
            "micro_speedup": 1.0,
            "verification_status": "deferred",
        },
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    v = out["by_kernel"][0]["verification"]
    assert v.get("compile_passed") is True
    assert v.get("correctness_passed") is False
    assert v.get("correctness_source") == "missing"
    assert v.get("verification_status") == "deferred"


def test_failure_breakdown_classifies_by_error_class(tmp_path: Path) -> None:
    """failure_reason_breakdown uses error_class buckets so root causes aren't buried in 'other'."""
    state = _make_state(
        session_id="sid1",
        top15=[_top15_entry("k001", gpu_pct=10.0), _top15_entry("k002", gpu_pct=20.0)],
        attempts={
            "k001": _attempt_entry(decision="REVERT", rejected_reason="max_failures", compile_passed=False),
            "k002": _attempt_entry(decision="REVERT", rejected_reason="max_failures", compile_passed=False),
        },
        rejected_ids=["k001", "k002"],
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k001",
        attempts=[
            {
                "backend": "forge",
                "status": "failed",
                "attempt_id": "g1",
                "optimized_path": "/workspace/optimized/g1_stdout.log",
                "returncode": 1,
                "stdout_tail": ("v3 preprocess completed in 100s (success=False, errors=2)"),
            }
        ],
    )
    _write_full_kernel_result(
        tmp_path,
        "sid1",
        "k002",
        attempts=[
            {
                "backend": "claude",
                "status": "failed",
                "attempt_id": "c1",
                "optimized_path": "/workspace/optimized/c1_stdout.log",
                "returncode": 1,
                "stdout_tail": '"error_message": "Timed out after 480s"',
            },
            {
                "backend": "codex",
                "status": "failed",
                "attempt_id": "x1",
                "optimized_path": "/workspace/optimized/x1_stdout.log",
                "returncode": 1,
                "stdout_tail": '"error_message": "Timed out after 480s"',
            },
        ],
    )
    out = build_kernel_optimization_summary(state, tmp_path)
    breakdown = out["failure_reason_breakdown"]
    assert breakdown.get("preprocess_failed") == 1, breakdown
    assert breakdown.get("timeout") == 1, breakdown
    assert breakdown.get("other", 0) == 0, f"new buckets should absorb root causes, leaving other empty: {breakdown}"
