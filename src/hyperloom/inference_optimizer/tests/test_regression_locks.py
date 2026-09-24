# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bug-fix regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors.report import (
    ReportExecutor,
)
from hyperloom.orchestrator.roles import (
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.roles.base import BackendError
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType, NoIntentEmitted
from hyperloom.orchestrator.bus.message_bus import Message
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.mark.asyncio
async def test_report_resolves_session_dir_from_env(tmp_path, monkeypatch):
    """ReportExecutor resolves session_dir from $USER_DATA_PATH."""
    sd = tmp_path / "real-session"
    sd.mkdir()
    state = SharedState(session_id=sd.name, model_name="qwen3-8b", baseline_tput=800.0, cumulative_gain_validated=2.5)
    state.save(sd)
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    storage_dir = sd / "storage"
    storage_dir.mkdir()
    SqliteConnection(storage_dir / "coordinator.db").close()

    monkeypatch.setenv("USER_DATA_PATH", str(sd))

    class _Ctx:
        task = Task(
            task_id="t-1", kind="report", params={}, requires_lanes=(), state="running", idempotency_key="report-test-1"
        )
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
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    (sd / "storage").mkdir()
    SqliteConnection(sd / "storage" / "coordinator.db").close()

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "wrong"))

    class _Ctx:
        task = Task(
            task_id="t-2", kind="report", params={}, requires_lanes=(), state="running", idempotency_key="report-test-2"
        )
        lease = None
        extra = {"session_dir": str(sd)}

    out = await ReportExecutor()(_Ctx())
    assert out["status"] == "succeeded", out
    summary = json.loads(Path(out["json_path"]).read_text())
    assert summary["session_id"] == "ctx-session"
    assert summary["baseline_tput"] == 600.0


@pytest.mark.asyncio
async def test_trace_analyze_caches_result_to_shared_state(session_dir, monkeypatch, tmp_path):
    """Successful trace_analyze caches; an identical request short-circuits the handler."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

        candidates_path = tmp_path / "kernel_candidates.json"
        candidates_path.write_text("{}", encoding="utf-8")
        call_count = {"n": 0}

        async def fake_handler(payload, *, session_dir):
            call_count["n"] += 1
            return {
                "status": "ok",
                "candidates_path": str(candidates_path),
                "hot_kernels": [
                    {
                        "kernel_id": "k001",
                        "name": "kA",
                        "gpu_pct": 30.0,
                        "source_file": "/sgl-workspace/aiter/csrc/x.cu",
                        "reusable_native_kernel": True,
                    },
                ],
            }

        monkeypatch.setitem(
            kernel_request_handlers.KERNEL_REQUEST_HANDLERS,
            "trace_analyze",
            fake_handler,
        )

        intent = Intent(
            type=IntentType.REQUEST,
            payload={
                "target_agent": "kernel_agent",
                "kind": "trace_analyze",
                "params": {"trace_input": "/tmp/trace-A.json.gz"},
            },
        )
        await c._handle_intent("orchestration", intent)
        await c._handle_intent("orchestration", intent)

        assert call_count["n"] == 1, "second identical request must hit the cache"
        cached = c.shared_state.last_trace_analyze
        assert cached["trace_input"] == "/tmp/trace-A.json.gz"
        assert cached["candidates_path"] == str(candidates_path)
        assert cached["hot_kernels_top15"][0]["kernel_id"] == "k001"
        assert "k001" in cached["reusable_native_kernel_ids"]

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel_agent",
                    "kind": "trace_analyze",
                    "params": {"trace_input": "/tmp/trace-B.json.gz"},
                },
            ),
        )
        assert call_count["n"] == 2

        assert "last_trace_analyze=" in c.shared_state.to_prompt_summary()
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_profile_promotion_records_args_and_clears_select_cache(session_dir):
    """A new profile clears the trace_analyze cache and stamps the server config."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        from hyperloom.orchestrator.state.task_registry import Task

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
            "output_throughput": 805.0,
            "trace_dir": "/tmp/ws/torch_trace",
            "main_trace_path": "/tmp/ws/torch_trace/main.trace.json.gz",
            "trace_files": ["/tmp/ws/torch_trace/main.trace.json.gz"],
            "workspace": "/tmp/ws",
        }
        await c._promote_to_shared_state("profile", result)

        assert c.shared_state.last_profile_trace == "/tmp/ws/torch_trace/main.trace.json.gz"
        assert (c.shared_state.current_best or {}).get("action") != "profile"

        reloaded = SharedState.load_or_init(session_dir)
        assert reloaded.last_profile_trace == "/tmp/ws/torch_trace/main.trace.json.gz"
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


def _orchestration_turn(turn: MockTurn) -> dict[str, object]:
    backends = _silent_backends()
    backends["orchestration"] = MockBackend(ScriptedPlan(turns=[turn]), name="o")
    return backends


@pytest.mark.asyncio
async def test_request_response_visible_in_next_prompt(session_dir, monkeypatch):
    """The response to a kernel request reaches the requester's next prompt."""
    from hyperloom.orchestrator.kernel import request_handlers as krh

    async def fake_handler(payload, *, session_dir):
        return {"status": "ok", "selected_kernels": [{"rank": 1, "name": "x"}]}

    monkeypatch.setitem(krh.KERNEL_REQUEST_HANDLERS, "trace_analyze", fake_handler)
    request = Intent(
        type=IntentType.REQUEST,
        payload={"target_agent": "kernel_agent", "kind": "trace_analyze", "params": {"trace_input": "/t.json"}},
    )
    c = Coordinator(session_dir, backends=_orchestration_turn(MockTurn(intents=[request])))
    try:
        await c._reactor_pass("orchestration")
        assert "trace_analyze_done" in await c._compose_prompt("orchestration")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_no_intent_turn_advances_cursor(session_dir):
    """A reply carrying no parseable intent still consumes what its prompt rendered."""
    c = Coordinator(session_dir, backends=_orchestration_turn(MockTurn(raise_error=NoIntentEmitted("no envelope"))))
    try:
        alert = Message.new("robustness", "*", "alert", {"kind": "stall_warning"})
        await c.bus.append_and_seq(alert)
        await c._reactor_pass("orchestration")

        cur = await c.cursors.load("orchestration")
        assert cur.last_processed_seq >= alert.seq
        assert "stall_warning" not in await c._compose_prompt("orchestration")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_backend_error_turn_does_not_advance_cursor(session_dir):
    """A failed backend call leaves the messages its prompt rendered unread."""
    c = Coordinator(session_dir, backends=_orchestration_turn(MockTurn(raise_error=BackendError("gateway down"))))
    try:
        alert = Message.new("robustness", "*", "alert", {"kind": "stall_warning"})
        await c.bus.append_and_seq(alert)
        await c._reactor_pass("orchestration")

        cur = await c.cursors.load("orchestration")
        assert cur.last_processed_seq == 0
        assert "stall_warning" in await c._compose_prompt("orchestration")
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_failed_kernel_request_recorded_in_last_action_failures(session_dir):
    """A failed kernel request lands in the log the FAILURE RECOVERY block reads."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel_agent", "kind": "no_such_kind"},
            ),
        )
        assert c.shared_state.last_action_failures[-1]["error_class"] == "unknown_kernel_kind"
        assert "unknown_kernel_kind" in c.shared_state.to_prompt_summary()
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_integrate_deferred_when_lanes_busy(session_dir, monkeypatch):
    """An integrate request returns deferred (not failed) when the benchmark lanes are held."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

    c = Coordinator(session_dir, backends=_silent_backends())
    c.shared_state.baseline_tput = 800.0  # satisfy the execution_order gate
    try:
        # Hold the integrate lanes with an external holder.
        lanes = list(ACTION_CATALOGUE["integrate"].requires_lanes)
        held = await c.locks.try_acquire_many(
            lanes,
            holder_id="blocker",
            task_id="blocker",
            action="test_holder",
            ttl_sec=60,
        )
        assert held is not None

        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel_agent",
                    "kind": "integrate",
                    "patch_path": "/tmp/fake.patch",
                },
            ),
        )

        # Response message should carry status=deferred.
        msgs = await c.bus.tail(topic="response", n=100)
        responses = [m for m in msgs if m.payload.get("kind") == "integrate_done"]
        assert responses, "expected integrate_done response"
        assert responses[-1].payload["status"] == "deferred"
        # The failed-requests ledger must NOT record this as a failure.
        assert not any(
            f.get("error_class") == "handler_exception" or f.get("kind") == "integrate"
            for f in (c.shared_state.last_action_failures or [])
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_integrate_executes_and_releases_lanes_when_free(session_dir, monkeypatch, tmp_path):
    """An integrate request acquires lanes, runs, and releases them."""
    from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

    c = Coordinator(session_dir, backends=_silent_backends())
    c.shared_state.baseline_tput = 800.0  # satisfy the execution_order gate
    try:
        lanes = list(ACTION_CATALOGUE["integrate"].requires_lanes)

        # Verify lanes are free before the request.
        holders_before = await c.locks.lane_holders()
        for lane in lanes:
            assert int(holders_before.get(lane, 0)) == 0

        # A minimal integrate that reverts immediately (no patch file exists).
        await c._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel_agent",
                    "kind": "integrate",
                    "patch_path": str(tmp_path / "nonexistent.patch"),
                    "kernel_id": "k-test",
                },
            ),
        )

        # Lanes must be free after the handler returned.
        holders_after = await c.locks.lane_holders()
        for lane in lanes:
            assert int(holders_after.get(lane, 0)) == 0, f"lane {lane!r} still held after integrate"
    finally:
        await c.stop()
