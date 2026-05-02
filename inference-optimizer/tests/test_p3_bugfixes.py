"""P3 bug-fix regression tests.

Locks the three P3 fixes uncovered by the resume3 1h validation
(see PR #130 description):

* **Bug A** ``ReportExecutor`` failed with ``could not resolve session_dir``
  because Conductor never threaded the active session_dir into the
  executor context. Fix: cli.py exports ``$INFERENCE_OPTIMIZER_SESSION_DIR``
  and report.py picks it up before falling back to the most-recent-session
  heuristic.

* **Bug B** Codex Kernel LLM emitted a duplicate (hallucinated) RESPONSE
  to the same REQUEST that the programmatic_handler had already answered.
  Fix: Conductor advances the target_agent cursor past ``request_msg.seq``
  immediately after the handler responds, so the next reactor pass for
  that agent doesn't see the request in its inbox.

* **Bug C** Orchestration fabricated trace paths for ``select_kernels``
  REQUESTs because SharedState never exposed the trace path produced by
  ``ProfileExecutor``. Fix: ``Conductor._promote_to_shared_state`` writes
  ``main_trace_path`` to ``shared_state.last_profile_trace`` on profile
  succeeded; ``to_prompt_summary`` shows it; cli.py's _DEFAULT_ORCH_PROMPT
  tells Orch to use it verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.action_executors.report import (
    ReportExecutor,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.conductor import Conductor
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.message_bus import Message
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.task_registry import Task
from inference_optimizer.paths import make_session_dir


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
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_ROOT", str(tmp_path))
    return make_session_dir("p3-test")


# ===========================================================================
# Bug A — ReportExecutor session_dir resolution
# ===========================================================================
@pytest.mark.asyncio
async def test_report_resolves_session_dir_from_env(tmp_path, monkeypatch):
    """When CLI exports INFERENCE_OPTIMIZER_SESSION_DIR, ReportExecutor
    uses it without needing task.params or ctx.extra to carry it."""
    sd = tmp_path / "real-session"
    sd.mkdir()
    state = SharedState(session_id=sd.name, model_name="qwen3-8b",
                        baseline_tput=800.0, cumulative_gain=2.5)
    state.save(sd)
    # Initialise the conductor.db with the schema (ensure_schema runs
    # automatically when SqliteConnection opens).
    from inference_optimizer.storage.connection import SqliteConnection
    storage_dir = sd / "storage"
    storage_dir.mkdir()
    SqliteConnection(storage_dir / "conductor.db").close()

    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(sd))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SESSION_ROOT", raising=False)

    class _Ctx:
        task = Task(task_id="t-1", kind="report", params={},
                    requires_lanes=(), state="running",
                    idempotency_key="report-test-1")
        lease = None
        extra: dict = {}

    out = await ReportExecutor()(_Ctx())
    assert out["status"] == "succeeded", out
    assert "json_path" in out and Path(out["json_path"]).exists()
    assert "md_path" in out and Path(out["md_path"]).exists()


@pytest.mark.asyncio
async def test_report_prefers_ctx_extra_over_env(tmp_path, monkeypatch):
    """ctx.extra['session_dir'] beats env var (in-process Conductor wins)."""
    sd = tmp_path / "ctx-session"
    sd.mkdir()
    SharedState(session_id=sd.name, baseline_tput=600.0).save(sd)
    from inference_optimizer.storage.connection import SqliteConnection
    (sd / "storage").mkdir()
    SqliteConnection(sd / "storage" / "conductor.db").close()

    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(tmp_path / "wrong"))

    class _Ctx:
        task = Task(task_id="t-2", kind="report", params={},
                    requires_lanes=(), state="running",
                    idempotency_key="report-test-2")
        lease = None
        extra = {"session_dir": str(sd)}

    out = await ReportExecutor()(_Ctx())
    assert out["status"] == "succeeded", out
    summary = json.loads(Path(out["json_path"]).read_text())
    assert summary["session_id"] == "ctx-session"
    assert summary["baseline_tput"] == 600.0


# ===========================================================================
# Bug B — programmatic_handler advances target cursor
# ===========================================================================
@pytest.mark.asyncio
async def test_programmatic_handler_advances_target_cursor(session_dir, monkeypatch):
    """After Conductor handles a 'select_kernels' request inline, the
    kernel agent's cursor should be past the request seq so its next
    compose_prompt won't include the already-handled request."""
    c = Conductor(session_dir, backends=_silent_backends())
    try:
        # Stub the real handler so we don't shell out to kernel-agent.
        from inference_optimizer.orchestrator import kernel_request_handlers
        async def fake_handler(payload, *, session_dir):
            return {"status": "ok", "selected_kernels": [{"rank": 1, "name": "x"}]}
        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "select_kernels", fake_handler,
        )

        # Cursor for kernel starts at 0
        cur_before = await c.cursors.load("kernel")
        assert cur_before.last_processed_seq == 0

        # Dispatch a REQUEST from orchestration → kernel
        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "select_kernels",
                     "params": {"trace_input": "/tmp/trace.json"}},
        )
        await c._handle_intent("orchestration", intent)

        # The request was written + the response was written
        msgs = await c.bus.tail(n=10)
        topics = [m.topic for m in msgs]
        assert "request" in topics
        assert "response" in topics

        # Kernel cursor is past the request — kernel won't see it next tick
        cur_after = await c.cursors.load("kernel")
        request_msg = next(m for m in msgs if m.topic == "request")
        assert cur_after.last_processed_seq >= request_msg.seq, (
            f"kernel cursor {cur_after.last_processed_seq} must be past "
            f"request seq {request_msg.seq} to suppress duplicate response"
        )
        # Verify replay_for excludes the request from kernel's next inbox
        leftover = await c.bus.replay_for(
            "kernel", after_seq=cur_after.last_processed_seq,
        )
        assert not any(m.topic == "request" and m.msg_id == request_msg.msg_id
                       for m in leftover)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_select_kernels_caches_result_to_shared_state(session_dir, monkeypatch, tmp_path):
    """Successful select_kernels writes a cache entry; the next identical
    request short-circuits without invoking the handler."""
    c = Conductor(session_dir, backends=_silent_backends())
    try:
        from inference_optimizer.orchestrator import kernel_request_handlers
        candidates_path = tmp_path / "kernel_candidates.json"
        candidates_path.write_text("{}", encoding="utf-8")
        call_count = {"n": 0}

        async def fake_handler(payload, *, session_dir):
            call_count["n"] += 1
            return {
                "status": "ok",
                "candidates_path": str(candidates_path),
                "hot_kernels": [
                    {"kernel_id": "k001", "name": "kA", "gpu_pct": 30.0,
                     "source_file": "/sgl-workspace/aiter/csrc/x.cu",
                     "reusable_native_kernel": True},
                ],
            }
        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "select_kernels", fake_handler,
        )

        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "select_kernels",
                     "params": {"trace_input": "/tmp/trace-A.json.gz"}},
        )
        await c._handle_intent("orchestration", intent)
        await c._handle_intent("orchestration", intent)

        assert call_count["n"] == 1, "second identical request must hit the cache"
        cached = c.shared_state.last_select_kernels
        assert cached["trace_input"] == "/tmp/trace-A.json.gz"
        assert cached["candidates_path"] == str(candidates_path)
        assert cached["hot_kernels_top5"][0]["kernel_id"] == "k001"

        # Different trace_input must NOT hit the cache.
        await c._handle_intent("orchestration", Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "select_kernels",
                     "params": {"trace_input": "/tmp/trace-B.json.gz"}},
        ))
        assert call_count["n"] == 2

        # to_prompt_summary surfaces the cached state for Orchestration.
        assert "last_select_kernels=" in c.shared_state.to_prompt_summary()
    finally:
        await c.stop()


# ===========================================================================
# Bug C — profile result promotes main_trace_path to SharedState
# ===========================================================================
@pytest.mark.asyncio
async def test_profile_promotion_writes_last_profile_trace(session_dir):
    c = Conductor(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.save(session_dir)

        result = {
            "status": "succeeded",
            "output_throughput": 805.0,  # +0.6%, below 1% so no current_best
            "trace_dir": "/tmp/ws/torch_trace",
            "main_trace_path": "/tmp/ws/torch_trace/main.trace.json.gz",
            "trace_files": ["/tmp/ws/torch_trace/main.trace.json.gz"],
            "workspace": "/tmp/ws",
        }
        await c._promote_to_shared_state("profile", result)

        assert c.shared_state.last_profile_trace == \
            "/tmp/ws/torch_trace/main.trace.json.gz"
        # below-threshold tput shouldn't move current_best
        assert (c.shared_state.current_best or {}).get("action") != "profile"

        # On reload, the field survives.
        reloaded = SharedState.load_or_init(session_dir)
        assert reloaded.last_profile_trace == \
            "/tmp/ws/torch_trace/main.trace.json.gz"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_trace_appears_in_prompt_summary(session_dir):
    c = Conductor(session_dir, backends=_silent_backends())
    try:
        c.shared_state.last_profile_trace = "/abs/path/to/trace.json.gz"
        summary = c.shared_state.to_prompt_summary()
        assert "last_profile_trace=/abs/path/to/trace.json.gz" in summary
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_trace_falls_back_to_trace_dir_when_no_main(session_dir):
    """If profile result only has trace_dir (no .json.gz files yet),
    SharedState should still get a usable hint instead of staying empty."""
    c = Conductor(session_dir, backends=_silent_backends())
    try:
        result = {
            "status": "succeeded",
            "trace_dir": "/tmp/ws/torch_trace",
            "trace_files": [],
            "workspace": "/tmp/ws",
        }
        await c._promote_to_shared_state("profile", result)
        assert c.shared_state.last_profile_trace == "/tmp/ws/torch_trace"
    finally:
        await c.stop()
