"""P2-2 tests: ProfileExecutor + kernel REQUEST programmatic handlers."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.orchestrator import kernel_request_handlers as krh
from inference_optimizer.orchestrator.action_executors.profile import (
    PROFILE_DEFAULT_CONFIG,
    ProfileExecutor,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
    MockTurn,
)
from inference_optimizer.orchestrator.conductor import Conductor
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.task_registry import TaskRegistry
from inference_optimizer.orchestrator.resource_lock import (
    ResourceLockManager, SqliteLeaseBackend,
)
from inference_optimizer.orchestrator.sub_agent_runner import (
    ExecutorContext, SubAgentRunner,
)
from inference_optimizer.paths import make_session_dir
from inference_optimizer.storage import SqliteConnection


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_ROOT", str(tmp_path))
    return make_session_dir("p2-2-test")


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _backends_silent() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {n: MockBackend(silent, name=n)
            for n in ("orchestration", "kernel", "critic", "robustness")}


# ===========================================================================
# ProfileExecutor
# ===========================================================================
def test_profile_default_config_path_is_in_assets():
    assert "profile_qwen3_8b_sglang.yaml" in str(PROFILE_DEFAULT_CONFIG)
    assert PROFILE_DEFAULT_CONFIG.exists(), \
        "profile YAML must ship as a package asset"


def test_profile_yaml_has_torch_profiler_enabled():
    """The whole point of the profile config is profiler ON."""
    import yaml
    with PROFILE_DEFAULT_CONFIG.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["benchmark"]["profiler"]["torch_profiler"]["enabled"] is True


@pytest.mark.asyncio
async def test_profile_executor_extracts_trace_dir(tmp_path):
    """When the workspace contains torch_trace/*.trace.json.gz, the
    executor surfaces them in the result so downstream consumers can
    feed them into tracelens_analysis.py."""
    db = SqliteConnection(tmp_path / "x.db")
    locks = ResourceLockManager(SqliteLeaseBackend(db))
    tr = TaskRegistry(db)
    sub = SubAgentRunner(locks, tr)

    # Build a fake workspace dir matching what Magpie would create.
    output_dir = tmp_path / "out"
    workspace = output_dir / "benchmark_sglang_20260501_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(json.dumps({
        "success": True,
        "framework": "sglang",
        "model": "/wekafs/models/Qwen-Qwen3-8B",
        "throughput": {
            "request_throughput": 3.2, "output_throughput": 800.0,
            "total_token_throughput": 1600.0, "completed_requests": 80,
            "duration_seconds": 25.0,
        },
        "latency": {"ttft": {"mean_ms": 140, "p99_ms": 158},
                    "e2el": {"mean_ms": 2500, "p99_ms": 2580}},
    }))
    trace_dir = workspace / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "TP-0_main.trace.json.gz").write_bytes(b"fake-trace")
    (trace_dir / "TP-0_aux.trace.json.gz").write_bytes(b"fake-trace")

    # Stub subprocess.run so we don't actually launch sglang.
    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ok", stderr="",
    )
    def _fake_run(*args, **kwargs):
        return fake_completed

    pe = ProfileExecutor(default_output_root=tmp_path / "ignored_root")
    task = await tr.create(
        kind="profile",
        params={"output_dir": str(output_dir), "config_path": str(PROFILE_DEFAULT_CONFIG)},
        idempotency_key="prof-1",
    )
    sub.register_executor("profile", pe)
    with patch("subprocess.run", side_effect=_fake_run):
        res = await sub.run_task(task)
    assert res.state == "succeeded"
    assert res.result["framework"] == "sglang"
    assert res.result["trace_dir"] == str(trace_dir)
    assert len(res.result["trace_files"]) == 2
    assert "main_trace_path" in res.result
    db.close()


# ===========================================================================
# kernel_request_handlers — direct unit
# ===========================================================================
@pytest.mark.asyncio
async def test_select_kernels_handler_dry_run_returns_structured_result(session_dir):
    """Tracelens tool always emits structured JSON (even on validation
    failure). Our handler must surface it verbatim — including ``status``
    + run_id + session_id — so callers can debug without parsing logs."""
    fake_trace = session_dir / "fake_trace_dir"
    fake_trace.mkdir()
    payload = {
        "trace_input": str(fake_trace),
        "session_id": session_dir.name,
        "model_name": "Qwen3-8B",
        "framework": "sglang",
        "top_k": 5,
        "dry_run": True,
        "budget_minutes": 1,
    }
    res = await krh.select_kernels_handler(payload, session_dir=session_dir)
    # The tool will return failed because the dir has no trace files,
    # but the response must be structured (not generic returncode-only).
    assert res["status"] in ("ok", "succeeded", "failed")
    assert "tool" in res or "run_id" in res or "error" in res
    assert res.get("session_id") == session_dir.name or "run_id" in res


@pytest.mark.asyncio
async def test_select_kernels_handler_surfaces_candidates_path(session_dir, monkeypatch):
    async def fake_run_subprocess(cmd, *, timeout_sec):
        payload = {
            "status": "ok",
            "hot_kernels": [],
            "artifact_paths": {
                "kernel_candidates": "/tmp/kernel_candidates.json",
            },
        }
        return 0, json.dumps(payload), ""

    monkeypatch.setattr(krh, "_run_subprocess", fake_run_subprocess)
    res = await krh.select_kernels_handler(
        {"trace_input": str(session_dir), "dry_run": True},
        session_dir=session_dir,
    )
    assert res["candidates_path"] == "/tmp/kernel_candidates.json"


@pytest.mark.asyncio
async def test_select_kernels_handler_missing_trace_input(session_dir):
    res = await krh.select_kernels_handler({}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "trace_input" in res["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_missing_kernel_id(session_dir):
    res = await krh.run_optimization_handler({}, session_dir=session_dir)
    assert res["status"] == "failed"
    assert "kernel_id" in res["error"]


@pytest.mark.asyncio
async def test_run_optimization_handler_dry_run(session_dir):
    payload = {
        "kernel_id": "fake_kernel_1",
        "session_id": session_dir.name,
        "dry_run": True,
        "budget_minutes": 1,
    }
    res = await krh.run_optimization_handler(payload, session_dir=session_dir)
    assert res.get("status") in ("ok", "succeeded", "failed")  # dry-run may still fail validation


def test_handlers_dispatch_table():
    """P2-2 only registered select_kernels + run_optimization. P2-4
    added apply_patch + integrate (covered in test_p2_4_integrate_report)."""
    assert krh.has_handler("select_kernels")
    assert krh.has_handler("run_optimization")
    assert not krh.has_handler("totally_unknown_kind")


# ===========================================================================
# Conductor — REQUEST programmatic handler integration
# ===========================================================================
@pytest.mark.asyncio
async def test_conductor_request_select_kernels_uses_handler(session_dir):
    """When Orchestration emits REQUEST{kind=select_kernels}, the Conductor
    should run the registered handler programmatically and emit RESPONSE
    on the bus *without* waiting for the Kernel LLM."""
    c = Conductor(session_dir, backends=_backends_silent())

    captured: dict = {}

    async def fake_handler(payload, *, session_dir):
        captured["payload"] = payload
        captured["session_dir"] = session_dir
        return {"status": "ok", "hot_kernels": ["kernel_a", "kernel_b"]}

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"select_kernels": fake_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel",
                    "kind": "select_kernels",
                    "params": {"trace_input": "/tmp/fake-trace.json.gz"},
                },
            ))
            req_msgs = await c.bus.tail(topic="request", to_agent="kernel")
            assert req_msgs, "request must be mirrored to kernel inbox"
            req_id = req_msgs[0].msg_id

            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs, "handler must emit RESPONSE without LLM"
            r = resp_msgs[0]
            assert r.from_agent == "kernel"
            assert r.payload["kind"] == "select_kernels_done"
            assert r.payload["status"] == "ok"
            assert r.payload["result"]["hot_kernels"] == ["kernel_a", "kernel_b"]
            assert r.payload["in_reply_to"] == req_id
            assert r.payload["source"] == "programmatic_handler"

            # And the handler did receive merged payload (params flattened in).
            assert captured["payload"].get("trace_input") == "/tmp/fake-trace.json.gz"
            assert captured["session_dir"] == session_dir
        finally:
            await c.stop()


@pytest.mark.asyncio
async def test_conductor_request_unknown_kind_routes_to_llm(session_dir):
    """REQUEST whose kind has no handler is mirrored to kernel inbox
    (LLM responder path) — no auto-RESPONSE."""
    c = Conductor(session_dir, backends=_backends_silent())
    try:
        await c._handle_intent("orchestration", Intent(
            type=IntentType.REQUEST,
            payload={
                "target_agent": "kernel",
                "kind": "invent_brand_new_kind",  # NOT in registry
            },
        ))
        req_msgs = await c.bus.tail(topic="request", to_agent="kernel")
        assert req_msgs, "request must be mirrored even when no handler"
        # No auto-response should have been emitted.
        resp_msgs = await c.bus.tail(topic="response")
        assert not resp_msgs
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_conductor_request_handler_exception_recorded(session_dir):
    """Handler crashes → RESPONSE.status='failed' + error_class set."""
    c = Conductor(session_dir, backends=_backends_silent())

    async def bad_handler(payload, *, session_dir):
        raise RuntimeError("boom")

    with patch.dict(krh.KERNEL_REQUEST_HANDLERS,
                     {"select_kernels": bad_handler}):
        try:
            await c._handle_intent("orchestration", Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel", "kind": "select_kernels"},
            ))
            resp_msgs = await c.bus.tail(topic="response", to_agent="orchestration")
            assert resp_msgs
            r = resp_msgs[0]
            assert r.payload["status"] == "failed"
            assert r.payload["result"]["error_class"] == "handler_exception"
            assert "boom" in r.payload["result"]["error"]
        finally:
            await c.stop()
