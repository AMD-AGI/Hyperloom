# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""#266 method-1 wiring: lifecycle events are actually emitted at the
phase/step boundaries.

Covers the emit points added on top of the lifecycle schema (Task B):

* ``_lifecycle_paths`` extracts only present, non-empty path-like fields.
* ``Coordinator._emit_lifecycle`` records AND persists to state.json.
* ``Coordinator._handle_request`` brackets a programmatic kernel step with
  START + END events carrying the input/output artifact paths + duration,
  and emits a lone END (no START) for the cache-hit / rejected-patch
  short-circuits.
* ``Coordinator._advance_phase_if_needed`` emits an ENTER phase-boundary
  marker (not a paired START).
* ``Coordinator._on_enter_close`` emits the final report END event.
* ``RooflineExecutor`` emits a TraceLens END event for the auto-roofline
  path (which never passes through ``_handle_request``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator.action_executors.roofline import (
    RooflineExecutor,
)
from inference_optimizer.orchestrator.backends import MockBackend, ScriptedPlan
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _lifecycle_paths,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.paths import make_session_dir
from inference_optimizer.protocol.intent import Intent, IntentType


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


# ===========================================================================
# _lifecycle_paths pure helper
# ===========================================================================
def test_lifecycle_paths_extracts_present_path_keys():
    payload = {
        "trace_input": "/tmp/trace.json.gz",
        "candidates_path": "/tmp/kc.json",
        "empty": "",
        "missing": None,
        "not_a_path_key": "ignored",
        "best_artifact_path": "/tmp/best.py",
    }
    out = _lifecycle_paths(payload)
    assert out == {
        "trace_input": "/tmp/trace.json.gz",
        "candidates_path": "/tmp/kc.json",
        "best_artifact_path": "/tmp/best.py",
    }


def test_lifecycle_paths_handles_non_dict():
    assert _lifecycle_paths(None) == {}
    assert _lifecycle_paths("nope") == {}
    assert _lifecycle_paths([1, 2]) == {}


# ===========================================================================
# Coordinator._emit_lifecycle records + persists
# ===========================================================================
@pytest.mark.asyncio
async def test_emit_lifecycle_records_and_persists(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c._emit_lifecycle(
            step="report",
            status="END",
            artifacts={"md_path": "/x/final.md", "json_path": "/x/final.json"},
            detail="close_phase_entry",
        )
        ev = c.shared_state.lifecycle[-1]
        assert ev["step"] == "report"
        assert ev["label"] == "Report"
        assert ev["status"] == "END"
        assert ev["artifacts"]["md_path"] == "/x/final.md"

        # Persisted so a launcher poll sees it.
        reloaded = SharedState.load_or_init(session_dir)
        assert reloaded.lifecycle[-1]["step"] == "report"
        assert reloaded.lifecycle[-1]["artifacts"]["json_path"] == "/x/final.json"
    finally:
        await c.stop()


# ===========================================================================
# Coordinator._handle_request brackets a kernel step with START + END
# ===========================================================================
@pytest.mark.asyncio
async def test_handle_request_emits_start_and_end(session_dir, monkeypatch, tmp_path):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from inference_optimizer.orchestrator import kernel_request_handlers
        candidates_path = tmp_path / "kernel_candidates.json"
        candidates_path.write_text("{}", encoding="utf-8")

        async def fake_handler(payload, *, session_dir):
            return {
                "status": "ok",
                "candidates_path": str(candidates_path),
                "hot_kernels": [],
            }
        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "trace_analyze", fake_handler,
        )

        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "trace_analyze",
                     "params": {"trace_input": "/tmp/trace-A.json.gz"}},
        )
        await c._handle_intent("orchestration", intent)

        ta_events = [e for e in c.shared_state.lifecycle if e["step"] == "trace_analyze"]
        assert len(ta_events) == 2, f"expected START + END, got {ta_events}"
        start, end = ta_events[0], ta_events[1]

        # START carries the input trace path; no duration yet.
        assert start["status"] == "START"
        assert start["label"] == "TraceLens"
        assert start["artifacts"]["trace_input"] == "/tmp/trace-A.json.gz"
        assert "duration_s" not in start

        # END carries the produced artifact + a measured duration.
        assert end["status"] == "END"
        assert end["artifacts"]["candidates_path"] == str(candidates_path)
        assert "duration_s" in end and end["duration_s"] >= 0.0
        # Monotonic ordering preserved.
        assert end["seq"] > start["seq"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_request_failed_handler_emits_error(session_dir, monkeypatch):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from inference_optimizer.orchestrator import kernel_request_handlers

        async def boom_handler(payload, *, session_dir):
            raise RuntimeError("kaboom")
        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "trace_analyze", boom_handler,
        )

        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "trace_analyze",
                     "params": {"trace_input": "/tmp/t.json.gz"}},
        )
        await c._handle_intent("orchestration", intent)

        ta_events = [e for e in c.shared_state.lifecycle if e["step"] == "trace_analyze"]
        assert [e["status"] for e in ta_events] == ["START", "ERROR"]
    finally:
        await c.stop()


# ===========================================================================
# RooflineExecutor emits a TraceLens END event (auto-roofline path)
# ===========================================================================
def _roofline_ctx(tmp_path: Path) -> RunnerContext:
    task = Task(
        task_id="t-roofline-1", kind="roofline", state="running",
        params={"base_extra_args": "--mem-fraction-static=0.92"},
        idempotency_key="roofline:t-1",
        requires_lanes=["profile_lane"],
    )
    return RunnerContext(task=task, lease=None, extra={"session_dir": str(tmp_path)})


@pytest.mark.asyncio
async def test_roofline_executor_emits_lifecycle_end(tmp_path):
    state = SharedState()
    state.baseline_tput = 100.0
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\nCompute 51%\n", encoding="utf-8")

    async def fake_profile(ctx):
        return {"status": "succeeded", "main_trace_path": "/tmp/trace.gz",
                "workspace": "/tmp/workspace"}

    async def fake_ta(payload, *, session_dir):
        return {"status": "ok", "candidates_path": "/tmp/kc.json",
                "trace_report_path": str(md), "hot_kernels": []}

    p1 = patch(
        "inference_optimizer.orchestrator.action_executors.profile.profile_executor",
        new=fake_profile,
    )
    p2 = patch(
        "inference_optimizer.orchestrator.kernel_request_handlers.trace_analyze_handler",
        new=fake_ta,
    )
    executor = RooflineExecutor(shared_state=state)
    with p1, p2:
        result = await executor(_roofline_ctx(tmp_path))

    assert result["status"] == "succeeded"
    rf_events = [e for e in state.lifecycle if e["step"] == "roofline"]
    assert len(rf_events) == 1
    ev = rf_events[0]
    assert ev["status"] == "END"
    assert ev["label"] == "TraceLens"
    assert ev["artifacts"]["trace_input"] == "/tmp/trace.gz"
    assert ev["artifacts"]["analysis_md_path"] == str(md)
    assert "duration_s" in ev


# ===========================================================================
# _handle_request short-circuits: cache hit / rejected patch emit a lone END
# ===========================================================================
@pytest.mark.asyncio
async def test_handle_request_cache_hit_emits_lone_end(session_dir, monkeypatch):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        cached = {"status": "ok", "candidates_path": "/tmp/cached_kc.json"}
        monkeypatch.setattr(
            c, "_cached_kernel_request", lambda kind, payload: cached,
        )

        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "trace_analyze",
                     "params": {"trace_input": "/tmp/trace.json.gz"}},
        )
        await c._handle_intent("orchestration", intent)

        ta = [e for e in c.shared_state.lifecycle if e["step"] == "trace_analyze"]
        # A cache hit never runs the handler: exactly one END, no START.
        assert [e["status"] for e in ta] == ["END"], f"want lone END, got {ta}"
        assert ta[0]["detail"] == "cache_hit"
        assert ta[0]["label"] == "TraceLens"
        assert ta[0]["artifacts"]["candidates_path"] == "/tmp/cached_kc.json"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_handle_request_rejected_integrate_emits_lone_end(
    session_dir, monkeypatch,
):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        # Bypass the execution-order gate: an integrate request is denied in
        # the initial phase, which would return before reaching the emit.
        monkeypatch.setattr(
            c, "_sequence_denial_for_request", lambda target, kind: None,
        )
        monkeypatch.setattr(
            c, "_cached_kernel_request", lambda kind, payload: None,
        )
        rejection = {
            "kernel_id": "tg001",
            "patch_path": "/tmp/p.patch",
            "target_file": "/tmp/t.py",
            "attempt_count": 3,
        }
        monkeypatch.setattr(
            c.shared_state, "find_rejected_kernel_patch",
            lambda payload: rejection,
        )

        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "integrate",
                     "params": {"patch_path": "/tmp/p.patch"}},
        )
        await c._handle_intent("orchestration", intent)

        ig = [e for e in c.shared_state.lifecycle if e["step"] == "integrate"]
        assert [e["status"] for e in ig] == ["END"], f"want lone END, got {ig}"
        assert ig[0]["detail"] == "rejected"
        assert ig[0]["label"] == "Integrate"
        assert ig[0]["artifacts"]["patch_path"] == "/tmp/p.patch"
    finally:
        await c.stop()


# ===========================================================================
# _advance_phase_if_needed emits an ENTER phase-boundary marker
# ===========================================================================
@pytest.mark.asyncio
async def test_advance_phase_emits_enter_marker(session_dir, monkeypatch):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from inference_optimizer.orchestrator import phase_state as _ps

        c.shared_state.phase = _ps.PHASE_PRELUDE
        # Force a single PRELUDE -> KERNEL transition.
        monkeypatch.setattr(
            _ps, "compute_next_phase",
            lambda *a, **k: (_ps.PHASE_KERNEL, "kernel_ready",
                             {"evidence": "test"}),
        )

        # Isolate the emit from per-phase entry side effects.
        async def _noop(**kwargs):
            return None
        monkeypatch.setattr(c, "_on_phase_entered", _noop)

        await c._advance_phase_if_needed()

        enter = [e for e in c.shared_state.lifecycle if e["status"] == "ENTER"]
        assert len(enter) == 1, f"want one ENTER, got {c.shared_state.lifecycle}"
        ev = enter[0]
        # ENTER is a point-in-time marker: step == the phase name, no END.
        assert ev["phase"] == _ps.PHASE_KERNEL.upper()
        assert ev["step"] == _ps.PHASE_KERNEL
        assert ev["label"] == "Kernel optimization"
        assert "reason=kernel_ready" in ev["detail"]
        assert "duration_s" not in ev
    finally:
        await c.stop()


# ===========================================================================
# _on_enter_close emits the final report END event
# ===========================================================================
@pytest.mark.asyncio
async def test_on_enter_close_emits_report_end(session_dir, monkeypatch):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        report_task = Task(
            task_id="rpt-1", kind="report", state="running", params={},
            idempotency_key="internal-report-close", requires_lanes=[],
        )
        bd_task = Task(
            task_id="bd-1", kind="session_breakdown", state="running",
            params={}, idempotency_key="internal-breakdown-close",
            requires_lanes=[],
        )

        async def fake_enqueue_report(*, reason):
            return report_task

        async def fake_enqueue_breakdown(*, reason):
            return bd_task

        class _Res:
            state = "succeeded"

        async def fake_run_task(task):
            return _Res()

        monkeypatch.setattr(
            c, "_enqueue_internal_report_task", fake_enqueue_report,
        )
        monkeypatch.setattr(
            c, "_enqueue_internal_session_breakdown_task",
            fake_enqueue_breakdown,
        )
        monkeypatch.setattr(c.sub, "run_task", fake_run_task)
        monkeypatch.setattr(
            c, "cortex_finalize_recipe_and_journal", lambda: None,
        )

        await c._on_enter_close(from_phase="SWEEP")

        rpt = [e for e in c.shared_state.lifecycle
               if e["step"] == "report" and e["status"] == "END"]
        assert rpt, f"want a report END, got {c.shared_state.lifecycle}"
        ev = rpt[-1]
        assert ev["label"] == "Report"
        assert ev["detail"] == "close_phase_entry"
        assert ev["artifacts"]["md_path"].endswith("final.md")
        assert ev["artifacts"]["json_path"].endswith("final.json")
    finally:
        await c.stop()
