# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""P3 bug-fix regression tests (PR #130).

Bug A: ReportExecutor session_dir resolution via $USER_DATA_PATH.
Bug B: programmatic handler advances the target-agent cursor to suppress
duplicate RESPONSEs.
Bug C: profile success promotes ``main_trace_path`` to
``shared_state.last_profile_trace``.
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
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
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
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


# Bug A — ReportExecutor session_dir resolution
@pytest.mark.asyncio
async def test_report_resolves_session_dir_from_env(tmp_path, monkeypatch):
    """ReportExecutor resolves session_dir from $USER_DATA_PATH."""
    sd = tmp_path / "real-session"
    sd.mkdir()
    state = SharedState(session_id=sd.name, model_name="qwen3-8b",
                        baseline_tput=800.0, cumulative_gain=2.5)
    state.save(sd)
    # Initialise the coordinator.db schema.
    from inference_optimizer.storage.connection import SqliteConnection
    storage_dir = sd / "storage"
    storage_dir.mkdir()
    SqliteConnection(storage_dir / "coordinator.db").close()

    monkeypatch.setenv("USER_DATA_PATH", str(sd))

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
    """ctx.extra['session_dir'] beats env var (in-process Coordinator wins)."""
    sd = tmp_path / "ctx-session"
    sd.mkdir()
    SharedState(session_id=sd.name, baseline_tput=600.0).save(sd)
    from inference_optimizer.storage.connection import SqliteConnection
    (sd / "storage").mkdir()
    SqliteConnection(sd / "storage" / "coordinator.db").close()

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "wrong"))

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


# Bug B — programmatic_handler advances target cursor
@pytest.mark.asyncio
async def test_programmatic_handler_advances_target_cursor(session_dir, monkeypatch):
    """After an inline 'trace_analyze' handler, the kernel cursor moves past the request."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from inference_optimizer.orchestrator import kernel_request_handlers
        async def fake_handler(payload, *, session_dir):
            return {"status": "ok", "selected_kernels": [{"rank": 1, "name": "x"}]}
        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "trace_analyze", fake_handler,
        )

        cur_before = await c.cursors.load("kernel")
        assert cur_before.last_processed_seq == 0

        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "trace_analyze",
                     "params": {"trace_input": "/tmp/trace.json"}},
        )
        await c._handle_intent("orchestration", intent)

        msgs = await c.bus.tail(n=10)
        topics = [m.topic for m in msgs]
        assert "request" in topics
        assert "response" in topics

        cur_after = await c.cursors.load("kernel")
        request_msg = next(m for m in msgs if m.topic == "request")
        assert cur_after.last_processed_seq >= request_msg.seq, (
            f"kernel cursor {cur_after.last_processed_seq} must be past "
            f"request seq {request_msg.seq} to suppress duplicate response"
        )
        leftover = await c.bus.replay_for(
            "kernel", after_seq=cur_after.last_processed_seq,
        )
        assert not any(m.topic == "request" and m.msg_id == request_msg.msg_id
                       for m in leftover)
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_trace_analyze_caches_result_to_shared_state(session_dir, monkeypatch, tmp_path):
    """Successful trace_analyze caches; an identical request short-circuits the handler."""
    c = Coordinator(session_dir, backends=_silent_backends())
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
            "trace_analyze", fake_handler,
        )

        intent = Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "trace_analyze",
                     "params": {"trace_input": "/tmp/trace-A.json.gz"}},
        )
        await c._handle_intent("orchestration", intent)
        await c._handle_intent("orchestration", intent)

        assert call_count["n"] == 1, "second identical request must hit the cache"
        cached = c.shared_state.last_trace_analyze
        assert cached["trace_input"] == "/tmp/trace-A.json.gz"
        assert cached["candidates_path"] == str(candidates_path)
        assert cached["hot_kernels_top15"][0]["kernel_id"] == "k001"
        assert "k001" in cached["reusable_native_kernel_ids"]

        # Different trace_input must NOT hit the cache.
        await c._handle_intent("orchestration", Intent(
            type=IntentType.REQUEST,
            payload={"target_agent": "kernel", "kind": "trace_analyze",
                     "params": {"trace_input": "/tmp/trace-B.json.gz"}},
        ))
        assert call_count["n"] == 2

        assert "last_trace_analyze=" in c.shared_state.to_prompt_summary()
    finally:
        await c.stop()


# Bug C — profile result promotes main_trace_path to SharedState
@pytest.mark.asyncio
async def test_profile_promotion_records_args_and_clears_select_cache(session_dir):
    """A new profile clears the trace_analyze cache and stamps the server config."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from inference_optimizer.orchestrator.task_registry import Task
        c.shared_state.last_trace_analyze = {
            "trace_input": "/old/trace.json.gz",
            "candidates_path": "/old/k.json",
            "hot_kernels_top15": [],
            "reusable_native_kernel_ids": [],
        }
        task = Task(
            task_id="t-profile-1",
            kind="profile",
            params={"base_extra_args": "--cuda-graph-max-bs 8"},
            requires_lanes=(),
            state="running",
            idempotency_key="profile-test-1",
        )
        result = {
            "status": "succeeded",
            "main_trace_path": "/new/trace.json.gz",
            "trace_files": ["/new/trace.json.gz"],
            "trace_dir": "/new/torch_trace",
        }
        await c._promote_to_shared_state("profile", result, task=task)
        assert c.shared_state.last_profile_trace == "/new/trace.json.gz"
        assert c.shared_state.last_profile_args == "--cuda-graph-max-bs 8"
        assert c.shared_state.last_trace_analyze == {}
        summary = c.shared_state.to_prompt_summary()
        assert "last_profile_args='--cuda-graph-max-bs 8'" in summary
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_promotion_writes_last_profile_trace(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.save(session_dir)

        result = {
            "status": "succeeded",
            "output_throughput": 805.0,  # +0.6%, below 1% threshold
            "trace_dir": "/tmp/ws/torch_trace",
            "main_trace_path": "/tmp/ws/torch_trace/main.trace.json.gz",
            "trace_files": ["/tmp/ws/torch_trace/main.trace.json.gz"],
            "workspace": "/tmp/ws",
        }
        await c._promote_to_shared_state("profile", result)

        assert c.shared_state.last_profile_trace == \
            "/tmp/ws/torch_trace/main.trace.json.gz"
        assert (c.shared_state.current_best or {}).get("action") != "profile"

        reloaded = SharedState.load_or_init(session_dir)
        assert reloaded.last_profile_trace == \
            "/tmp/ws/torch_trace/main.trace.json.gz"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_trace_appears_in_prompt_summary(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.last_profile_trace = "/abs/path/to/trace.json.gz"
        summary = c.shared_state.to_prompt_summary()
        assert "last_profile_trace=/abs/path/to/trace.json.gz" in summary
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_trace_dir_without_json_not_promoted(session_dir):
    """Empty trace_dir without .trace.json.gz must NOT be promoted."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        await c._promote_to_shared_state(
            "profile",
            {
                "status": "succeeded",
                "trace_dir": "/tmp/ws/torch_trace",
                "trace_files": [],
                "workspace": "/tmp/ws",
            },
        )
        assert c.shared_state.last_profile_trace == ""
    finally:
        await c.stop()
