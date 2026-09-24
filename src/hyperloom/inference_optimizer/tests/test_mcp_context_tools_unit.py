# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the in-process context-pull MCP tool surface."""

from __future__ import annotations

import asyncio
import contextvars
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.roles import mcp_context_tools as mct


def _shared_state():
    """Build a fake SharedState exposing the to_*_summary projections."""
    return SimpleNamespace(
        to_mission_summary=lambda: "mission",
        to_prompt_summary=lambda: "prompt",
        to_gaps_summary=lambda max_attempts=0: "gaps",
        to_warm_start_summary=lambda: "warm",
        to_proposal_scores_summary=lambda: "scores",
        to_intervention_mix_summary=lambda: "mix",
        to_policy_denial_summary=lambda top_k=6: f"denials({top_k})",
        failures=[],
        find_failure=lambda fid: None,
        failures_for_task=lambda tid: [],
    )


def test_qualified():
    assert mct._qualified("foo") == "mcp__inference_optimizer_context__foo"


def test_provider_projections():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert p.mission_status() == "mission"
    assert p.shared_state_summary() == "prompt"
    assert p.gaps() == "gaps"
    assert p.warm_start() == "warm"
    assert p.proposal_scores() == "scores"
    assert p.intervention_mix() == "mix"


def test_safe_handles_exception():
    def boom():
        raise RuntimeError("x")

    p = mct.ContextProvider(shared_state=SimpleNamespace())
    out = p._safe(boom, "lbl")
    assert "unavailable" in out


def test_safe_empty_marker():
    p = mct.ContextProvider(shared_state=SimpleNamespace())
    assert p._safe(lambda: "", "lbl") == "(lbl: empty)"
    assert p._safe(lambda: None, "lbl") == "(lbl: empty)"


def test_why_denied_via_shared_state():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert p.why_denied(top_k=3) == "denials(3)"


def test_why_denied_via_reader():
    p = mct.ContextProvider(
        shared_state=_shared_state(),
        denial_reader=lambda k: f"reader({k})",
    )
    assert p.why_denied(top_k=2) == "reader(2)"


def test_analysis_md_not_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.analysis_md()


def test_analysis_md_wired():
    p = mct.ContextProvider(shared_state=_shared_state(), analysis_reader=lambda: "md")
    assert p.analysis_md() == "md"


def test_inbox_not_wired_and_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.inbox()
    p2 = mct.ContextProvider(shared_state=_shared_state(), inbox_reader=lambda s: f"inbox({s})")
    assert p2.inbox(5) == "inbox(5)"


def test_recent_outcomes_not_wired_and_wired():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in p.recent_outcomes()
    p2 = mct.ContextProvider(
        shared_state=_shared_state(),
        recent_outcomes_reader=lambda k: f"out({k})",
    )
    assert p2.recent_outcomes(4) == "out(4)"


async def test_run_action_now_not_wired_and_wired():
    async def run(name, params):
        return f"ran {name} {params}"

    p = mct.ContextProvider(shared_state=_shared_state())
    assert "not wired" in await p.run_action_now("a")
    p2 = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    assert await p2.run_action_now("act", {"k": 1}) == "ran act {'k': 1}"


def test_tool_name_tuples():
    assert "get_mission_status" in mct.CONTEXT_TOOL_NAMES
    assert mct.CONTEXT_TOOL_QUALIFIED_NAMES[0].startswith("mcp__")
    assert len(mct.CONTEXT_TOOL_NAMES) == len(mct.CONTEXT_TOOL_SPECS)


# ---- _resolve_sdk ----


def test_resolve_sdk_explicit():
    sentinel = object()
    assert mct._resolve_sdk(sentinel) is sentinel


def test_resolve_sdk_import_error(monkeypatch):
    def boom(_name):
        raise ImportError("no sdk")

    monkeypatch.setattr(mct.importlib, "import_module", boom)
    assert mct._resolve_sdk(None) is None


# ---- _make_handler ----


async def test_make_handler_success():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "mission_status")
    out = await handler({})
    assert out["content"][0]["text"] == "mission"


async def test_make_handler_forwards_kwargs():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "why_denied")
    out = await handler({"top_k": 2})
    assert out["content"][0]["text"] == "denials(2)"


async def test_make_handler_forwards_since_seq():
    p = mct.ContextProvider(shared_state=_shared_state(), inbox_reader=lambda s: f"inbox({s})")
    handler = mct._make_handler(p, "inbox")
    out = await handler({"since_seq": 9})
    assert out["content"][0]["text"] == "inbox(9)"


async def test_make_handler_forwards_action_args():
    async def run(name, params):
        return f"{name}:{params}"

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    handler = mct._make_handler(p, "run_action_now")
    out = await handler({"action_name": "act", "params": {"k": 1}})
    assert out["content"][0]["text"] == "act:{'k': 1}"


async def test_make_handler_non_str_result():
    provider = SimpleNamespace(foo=lambda **k: {"a": 1})
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert json.loads(out["content"][0]["text"]) == {"a": 1}


async def test_make_handler_exception():
    def boom(**k):
        raise RuntimeError("nope")

    provider = SimpleNamespace(foo=boom)
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert out["is_error"] is True


async def test_action_handler_awaits_on_loop_and_preserves_contextvars():
    trace = contextvars.ContextVar("mcp_test_trace", default="missing")
    owner_thread = threading.get_ident()
    seen = []

    async def execute(name, params):
        seen.append((threading.get_ident(), trace.get(), name, params))
        trace.set("action-only")
        return "done"

    async def run(name, params):
        return await asyncio.create_task(execute(name, params))

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    token = trace.set("caller-trace")
    try:
        out = await mct._make_handler(p, "run_action_now")({"action_name": "probe", "params": {"k": 1}})
        assert seen == [(owner_thread, "caller-trace", "probe", {"k": 1})]
        assert trace.get() == "caller-trace"
        assert out == {"content": [{"type": "text", "text": "done"}]}
    finally:
        trace.reset(token)


async def test_read_projection_stays_on_the_calling_thread():
    owner_thread = threading.get_ident()
    seen = []

    def read():
        seen.append(threading.get_ident())
        return "mission"

    p = mct.ContextProvider(shared_state=SimpleNamespace(to_mission_summary=read))
    out = await mct._make_handler(p, "mission_status")({})
    assert seen == [owner_thread]
    assert out == {"content": [{"type": "text", "text": "mission"}]}


@pytest.mark.parametrize("params", [{"k": 1}, None, "not an object"])
async def test_action_handler_keeps_argument_normalization(params):
    seen = []

    async def run(name, arguments):
        seen.append((name, arguments))
        return "done"

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    await mct._make_handler(p, "run_action_now")({"action_name": 123, "params": params, "ignored": True})
    assert seen == [("123", params if isinstance(params, dict) else {})]
    if isinstance(params, dict):
        assert seen[0][1] is not params


async def test_action_handler_keeps_provider_error_envelope():
    async def run(_name, _params):
        raise ValueError("action failed")

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "probe"})
    assert out == {
        "content": [{"type": "text", "text": "(context tool run_action_now unavailable: ValueError('action failed'))"}]
    }


@pytest.mark.parametrize("value", ["", None, {"not": "text"}])
async def test_action_handler_keeps_empty_marker(value):
    async def run(_name, _params):
        return value

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "probe"})
    assert out == {"content": [{"type": "text", "text": "(run_action_now: empty)"}]}


async def test_action_handler_keeps_handler_error_envelope(monkeypatch):
    async def run(**_kwargs):
        raise RuntimeError("bridge failed")

    p = mct.ContextProvider(shared_state=_shared_state())
    monkeypatch.setattr(p, "run_action_now", run)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "probe"})
    assert out == {"content": [{"type": "text", "text": "error: RuntimeError('bridge failed')"}], "is_error": True}


async def test_action_handler_propagates_provider_cancellation():
    async def run(_name, _params):
        raise asyncio.CancelledError("action cancelled")

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    with pytest.raises(asyncio.CancelledError, match="action cancelled"):
        await mct._make_handler(p, "run_action_now")({"action_name": "probe"})


@pytest.fixture
async def loop_action():
    """Await a real scheduled Future without lending it the caller's cancellation."""
    loop = asyncio.get_running_loop()
    probe = SimpleNamespace(
        started=asyncio.Event(),
        release=asyncio.Event(),
        finished=asyncio.Event(),
        submitted=[],
        calls=[],
        owner_thread=threading.get_ident(),
        bridge_threads=[],
    )

    async def execute(name, params):
        probe.calls.append((name, params, threading.get_ident()))
        probe.started.set()
        await probe.release.wait()
        probe.finished.set()
        return "action complete"

    async def run(name, params):
        probe.bridge_threads.append(threading.get_ident())
        future = asyncio.run_coroutine_threadsafe(execute(name, params), loop)
        probe.submitted.append(future)
        return await asyncio.shield(asyncio.wrap_future(future))

    probe.provider = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    try:
        yield probe
    finally:
        probe.release.set()
        await asyncio.wait_for(
            asyncio.gather(*(asyncio.wrap_future(future) for future in probe.submitted), return_exceptions=True),
            2.0,
        )


async def _sdk_request(server, request):
    """Exercise the request handler registered by the SDK on MCP 1.x or 2.x."""
    if hasattr(server, "get_request_handler"):
        entry = server.get_request_handler(request.method)
        assert entry is not None
        params = entry.params_type.model_validate(request.params.model_dump() if request.params else {})
        return await entry.handler(None, params)
    result = await server.request_handlers[type(request)](request)
    return result.root


@pytest.fixture(params=["direct", "sdk"])
async def action_handler(request, loop_action):
    if request.param == "direct":
        handler = mct._make_handler(loop_action.provider, "run_action_now")

        async def call(arguments):
            result = await handler(arguments)
            assert not result.get("is_error")
            return result["content"][0]["text"]

        return call

    sdk = pytest.importorskip("claude_agent_sdk")
    from mcp import types

    config = mct.build_context_tools_server(loop_action.provider, sdk_module=sdk)
    assert config["type"] == "sdk"
    assert config["name"] == mct.MCP_SERVER_NAME
    server = config["instance"]
    listed = await _sdk_request(server, types.ListToolsRequest())
    assert {tool.name for tool in listed.tools} == set(mct.CONTEXT_TOOL_NAMES)
    action = next(tool for tool in listed.tools if tool.name == "run_action_now")
    assert action.model_dump(by_alias=True)["inputSchema"] == mct._RUN_ACTION_SCHEMA

    async def call(arguments):
        result = await _sdk_request(
            server,
            types.CallToolRequest(
                method="tools/call", params=types.CallToolRequestParams(name="run_action_now", arguments=arguments)
            ),
        )
        assert not result.model_dump(by_alias=True)["isError"]
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        return result.content[0].text

    return call


async def test_action_handler_keeps_loop_timers_moving_before_action_completes(loop_action, action_handler):
    probe = loop_action
    call = asyncio.create_task(action_handler({"action_name": "probe", "params": {"k": 1}}))
    timer = asyncio.Event()
    handle = None
    try:
        await asyncio.wait_for(probe.started.wait(), 2.0)
        handle = asyncio.get_running_loop().call_later(0.01, timer.set)
        await asyncio.wait_for(timer.wait(), 0.5)
        assert not call.done(), "the handler must still be waiting while the loop timer advances"
        assert not probe.finished.is_set()
        assert probe.calls == [("probe", {"k": 1}, probe.owner_thread)]
        assert probe.bridge_threads == [probe.owner_thread]
        probe.release.set()
        out = await asyncio.wait_for(call, 1.0)
        assert out == "action complete"
    finally:
        if handle is not None:
            handle.cancel()
        probe.release.set()
        await asyncio.gather(call, return_exceptions=True)


async def test_action_handler_cancellation_does_not_cancel_submitted_action(loop_action, action_handler):
    probe = loop_action
    call = asyncio.create_task(action_handler({"action_name": "probe"}))
    try:
        await asyncio.wait_for(probe.started.wait(), 2.0)
        assert not call.done(), "the handler must yield before the bridge wait expires"
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(call, 0.5)
        assert not probe.finished.is_set()
        assert not probe.submitted[0].cancelled()
        probe.release.set()
        result = await asyncio.wait_for(asyncio.wrap_future(probe.submitted[0]), 1.0)
        assert result == "action complete"
        assert probe.calls == [("probe", {}, probe.owner_thread)]
    finally:
        probe.release.set()
        await asyncio.gather(call, return_exceptions=True)


# ---- build_context_tools_server ----


def test_build_server_unavailable_without_factories():
    p = mct.ContextProvider(shared_state=_shared_state())
    assert mct.build_context_tools_server(p, sdk_module=object()) is None


def test_build_server_with_fake_factories():
    p = mct.ContextProvider(shared_state=_shared_state())
    created = {}

    def tool_factory(name, desc, schema):
        def decorator(handler):
            return (name, handler)

        return decorator

    def server_factory(name, version, tools):
        created["name"] = name
        created["tools"] = tools
        return "SERVER"

    out = mct.build_context_tools_server(
        p,
        tool_factory=tool_factory,
        server_factory=server_factory,
    )
    assert out == "SERVER"
    assert created["name"] == mct.MCP_SERVER_NAME
    assert len(created["tools"]) == len(mct.CONTEXT_TOOL_SPECS)


async def test_action_provider_awaits_async_callback():
    async def run(name, params):
        await asyncio.sleep(0)
        return f"{name}:{params}"

    p = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    assert await p.run_action_now("probe", {"k": 1}) == "probe:{'k': 1}"


async def _run_saturation_probe(db_path, use_sdk):
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    db = SqliteConnection(db_path)
    executed, submitted = [], []
    all_started, worker_entered = asyncio.Event(), asyncio.Event()
    release_worker = threading.Event()
    started = 0

    def hold_worker():
        loop.call_soon_threadsafe(worker_entered.set)
        release_worker.wait(5)

    async def execute(name, params):
        nonlocal started
        started += 1
        if started == 20:
            all_started.set()
        row = await db.fetchone("SELECT ? AS value", (params["index"],))
        executed.append(row["value"])
        return f"{name}:{row['value']}"

    async def run(name, params):
        future = asyncio.run_coroutine_threadsafe(execute(name, params), loop)
        submitted.append(future)
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), 3.0)

    provider = mct.ContextProvider(shared_state=_shared_state(), action_runner=run)
    handler = mct._make_handler(provider, "run_action_now")
    if use_sdk:
        import claude_agent_sdk
        from mcp import types

        server = mct.build_context_tools_server(provider, sdk_module=claude_agent_sdk)["instance"]

        async def call(index):
            result = await _sdk_request(
                server,
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(
                        name="run_action_now", arguments={"action_name": "probe", "params": {"index": index}}
                    ),
                ),
            )
            assert not result.model_dump(by_alias=True)["isError"]
            return result.content[0].text

    else:

        async def call(index):
            result = await handler({"action_name": "probe", "params": {"index": index}})
            assert not result.get("is_error")
            return result["content"][0]["text"]

    blocker = asyncio.create_task(asyncio.to_thread(hold_worker))
    calls = []
    try:
        await asyncio.wait_for(worker_entered.wait(), 1.0)
        calls = [asyncio.create_task(call(index)) for index in range(20)]
        await asyncio.wait_for(all_started.wait(), 1.0)
        assert not blocker.done()
        assert all(not task.done() for task in calls)
        release_worker.set()
        results = await asyncio.wait_for(asyncio.gather(*calls), 2.0)
        assert results == [f"probe:{index}" for index in range(20)]
        assert sorted(executed) == list(range(20))
    finally:
        release_worker.set()
        await asyncio.gather(blocker, *calls, return_exceptions=True)
        await asyncio.gather(*(asyncio.wrap_future(future) for future in submitted), return_exceptions=True)
        db.close()


@pytest.mark.parametrize("use_sdk", [False, True], ids=["direct", "sdk"])
def test_concurrent_actions_leave_default_executor_available(tmp_path, use_sdk):
    if use_sdk:
        pytest.importorskip("claude_agent_sdk")
    source_root = Path(mct.__file__).resolve().parents[3]
    command = (
        "import asyncio,runpy,sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        f"module=runpy.run_path({str(Path(__file__).resolve())!r}); "
        f"asyncio.run(module['_run_saturation_probe']({str(tmp_path / 'probe.db')!r}, {use_sdk!r}))"
    )
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


# ---- real dispatcher lifecycle (POSIX coordinator dependencies) ----


@pytest.fixture
async def inline_coordinator(session_dir, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("Coordinator imports POSIX-only resource/fcntl modules")
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan

    monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S", "1")
    coord = Coordinator(
        session_dir,
        backends={name: MockBackend(ScriptedPlan(turns=[]), name=name) for name in ("orchestration", "critic")},
    )
    coord._coordinator_loop = asyncio.get_running_loop()
    try:
        yield coord
    finally:
        await coord.stop()


async def test_action_handler_rejects_baseline_with_default_whitelist(inline_coordinator):
    coord = inline_coordinator
    assert "baseline" not in coord._inline_action_whitelist()
    p = mct.ContextProvider(shared_state=coord.shared_state, action_runner=coord.dispatcher._run_action_now_wait)
    out = await mct._make_handler(p, "run_action_now")({"action_name": "baseline"})
    assert "not inline-eligible" in out["content"][0]["text"]
    assert await coord.tasks.queued() == []
    assert await coord.tasks.running() == []
    assert not coord.dispatcher._executions
    assert not coord.dispatcher._inflight_actions


async def test_sdk_advertises_dispatcher_inline_actions_and_rejects_baseline(inline_coordinator, monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    from mcp import types
    from hyperloom.inference_optimizer.cli.executors import _register_executors

    coord = inline_coordinator
    providers = []
    monkeypatch.setattr(coord.backends["orchestration"], "set_context_provider", providers.append, raising=False)
    _register_executors(coord, session_dir=coord.session_dir)
    server = mct.build_context_tools_server(providers[0], sdk_module=sdk)["instance"]
    listed = await _sdk_request(server, types.ListToolsRequest(method="tools/list"))
    action = next(tool for tool in listed.tools if tool.name == "run_action_now")
    choices = action.inputSchema["properties"]["action_name"]["enum"]
    assert choices == sorted(coord._inline_action_whitelist())
    assert "target_analysis" in choices
    assert "baseline" not in choices
    for name in ("baseline", "sweep", "profile"):
        result = await _sdk_request(
            server,
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(name="run_action_now", arguments={"action_name": name}),
            ),
        )
        assert result.model_dump(by_alias=True)["isError"] is True
        assert name in result.content[0].text
    assert not await coord.tasks.queued()
    assert not await coord.tasks.running()
    assert not coord.dispatcher._executions
    assert "enum" not in mct._RUN_ACTION_SCHEMA["properties"]["action_name"]

    async def target_analysis(ctx):
        return {"status": "ok"}

    coord.sub.register_executor("target_analysis", target_analysis)
    accepted = await _sdk_request(
        server,
        types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="run_action_now", arguments={"action_name": "target_analysis"}),
        ),
    )
    assert accepted.model_dump(by_alias=True)["isError"] is False
    assert "inline run complete" in accepted.content[0].text
    outcomes = await coord.bus.tail(topic="delegated_result")
    assert len(outcomes) == 1 and outcomes[0].payload["kind"] == "target_analysis"


async def test_sdk_omits_inline_tool_when_no_actions_are_eligible():
    sdk = pytest.importorskip("claude_agent_sdk")
    from mcp import types

    provider = mct.ContextProvider(shared_state=_shared_state(), inline_action_names=frozenset)
    server = mct.build_context_tools_server(provider, sdk_module=sdk)["instance"]
    listed = await _sdk_request(server, types.ListToolsRequest(method="tools/list"))
    assert "run_action_now" not in {tool.name for tool in listed.tools}


@pytest.mark.parametrize("intent_type", ["propose_action", "delegate"])
async def test_baseline_intent_runs_through_async_scheduler(inline_coordinator, intent_type):
    from hyperloom.inference_optimizer.protocol.intent import validate_envelope
    from hyperloom.orchestrator.roles.mcp_emit_intent import _emit_intent_handler

    coord = inline_coordinator
    coord.shared_state.phase = "PRELUDE"
    coord.shared_state.max_minutes = 180
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def baseline(ctx):
        calls.append(ctx.task.task_id)
        started.set()
        await release.wait()
        return {"status": "ok"}

    coord.sub.register_executor("baseline", baseline)
    envelope = {
        "intent_type": intent_type,
        "payload": {"action_name": "baseline", "predicted_gain_pct": 0, "params": {}},
    }
    ack = await _emit_intent_handler(envelope)
    assert not ack.get("is_error")
    intent = validate_envelope({"intents": [envelope]})[0]
    await coord._handle_intent("orchestration", intent)
    assert not calls
    if intent_type == "propose_action":
        assert not await coord.tasks.queued()
        proposal_id = next(iter(coord.state.pending_proposals))
        approval = validate_envelope(
            {
                "intents": [
                    {
                        "intent_type": "review_verdict",
                        "payload": {"target_proposal_msg_id": proposal_id, "verdict": "approve"},
                    }
                ]
            }
        )[0]
        await coord._handle_intent("critic", approval)
    queued = await coord.tasks.queued()
    assert len(queued) == 1 and queued[0].kind == "baseline"
    task = queued[0]
    assert "benchmark_lane" in task.requires_lanes
    assert not calls
    pump = asyncio.create_task(coord._pump_dispatcher_once())
    try:
        await asyncio.wait_for(started.wait(), 3)
        assert (await coord.tasks.get(task.task_id)).state == "running"
        assert await coord.locks.lane_holders()
        assert not pump.done()
        assert coord.shared_state.baseline_tput == 0
    finally:
        release.set()
        await asyncio.wait_for(pump, 3)
    assert calls == [task.task_id]
    assert (await coord.tasks.get(task.task_id)).state == "succeeded"
    outcomes = await coord.bus.tail(topic="delegated_result")
    assert len(outcomes) == 1 and outcomes[0].payload["task_id"] == task.task_id
    assert not outcomes[0].payload.get("inline")
    assert not await coord.locks.lane_holders()
    assert coord.shared_state.baseline_tput == 0


@pytest.mark.parametrize("abandon", ["timeout", "cancel"])
async def test_action_handler_keeps_real_dispatcher_execution_exactly_once(inline_coordinator, monkeypatch, abandon):
    from hyperloom.orchestrator.actions.cancel_channel import stop_was_asked_for

    coord = inline_coordinator
    started, release = asyncio.Event(), asyncio.Event()
    trace = contextvars.ContextVar("mcp_dispatch_trace", default="missing")
    calls, bridge_threads = [], []
    owner_thread = threading.get_ident()
    cancelled = []

    async def execute(ctx):
        calls.append((ctx.task.task_id, threading.get_ident(), trace.get()))
        trace.set("action-only")
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        assert not stop_was_asked_for()
        return {"status": "ok", "gain_pct": 1.5}

    async def run(name, params):
        bridge_threads.append(threading.get_ident())
        return await coord.dispatcher._run_action_now_wait(name, params)

    coord.sub.register_executor("target_analysis", execute)
    assert "target_analysis" in coord._inline_action_whitelist()
    monkeypatch.setattr(coord.policy, "validate_intent", lambda *_args: None)
    monkeypatch.setattr(coord.dispatcher, "_admission_denial_for_action", lambda _action: None)
    if abandon == "timeout":
        monkeypatch.setenv("INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S", "0.05")
    p = mct.ContextProvider(shared_state=coord.shared_state, action_runner=run)
    handler = mct._make_handler(p, "run_action_now")
    arguments = {"action_name": "target_analysis", "params": {"probe": "same action"}}
    token = trace.set("request-trace")
    call = asyncio.create_task(handler(arguments))
    execution_handle = None
    try:
        await asyncio.wait_for(started.wait(), 2.0)
        task_id = calls[0][0]
        execution_handle = coord.dispatcher._inflight_actions[task_id].atask
        if abandon == "timeout":
            out = await asyncio.wait_for(call, 1.0)
            assert "still running after" in out["content"][0]["text"]
        else:
            assert not call.done()
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(call, 0.5)
        assert not cancelled
        assert (await coord.tasks.get(task_id)).state == "running"
        assert not execution_handle.done()
        assert coord.dispatcher._executions
        repeated = await handler(arguments)
        assert "already 'running'" in repeated["content"][0]["text"]
        release.set()
        await asyncio.wait_for(asyncio.shield(execution_handle), 2.0)
        completed = await handler(arguments)
        assert "inline run complete" in completed["content"][0]["text"]
        assert "gain=1.5" in completed["content"][0]["text"]
        assert calls == [(task_id, owner_thread, "request-trace")]
        assert all(tid == owner_thread for tid in bridge_threads)
        assert trace.get() == "request-trace"
        assert (await coord.tasks.get(task_id)).state == "succeeded"
        outcomes = await coord.bus.tail(topic="delegated_result")
        assert len(outcomes) == 1
        assert outcomes[0].payload["task_id"] == task_id
        assert outcomes[0].payload["inline"] is True
        assert not coord.dispatcher._executions
        assert not coord.dispatcher._inflight_actions
        assert not await coord.locks.lane_holders()
    finally:
        release.set()
        await asyncio.gather(call, return_exceptions=True)
        if execution_handle is not None:
            await asyncio.wait_for(asyncio.shield(execution_handle), 2.0)
        trace.reset(token)


# ---- new tools: get_failure / get_variant_failures ----


def test_spec_methods_all_exist_on_provider():
    """Every CONTEXT_TOOL_SPECS entry must name a real ContextProvider method."""
    for _, _, _, method_name in mct.CONTEXT_TOOL_SPECS:
        assert callable(getattr(mct.ContextProvider, method_name, None)), (
            f"ContextProvider missing method {method_name!r}"
        )


def test_get_failure_not_found_points_at_the_disk_mirror():
    """An evicted packet is still reachable, so the miss must say where."""
    p = mct.ContextProvider(shared_state=_shared_state())
    out = p.get_failure("fail.t1.abc")
    assert "reports/failures/fail.t1.abc.json" in out


def test_get_failure_returns_json():
    import json

    fe = {"failure_id": "fail.t1.abc", "task_id": "t1", "error_class": "x"}
    ss = _shared_state()
    ss.find_failure = lambda fid: fe if fid == "fail.t1.abc" else None
    p = mct.ContextProvider(shared_state=ss)
    out = p.get_failure("fail.t1.abc")
    data = json.loads(out)
    assert data["failure_id"] == "fail.t1.abc"


def test_get_failure_requires_failure_id():
    p = mct.ContextProvider(shared_state=_shared_state())
    out = p.get_failure("")
    assert "required" in out


def test_get_variant_failures_empty():
    p = mct.ContextProvider(shared_state=_shared_state())
    out = p.get_variant_failures()
    assert "no failure" in out


def test_get_variant_failures_returns_entries():
    import json

    fe1 = {"failure_id": "fail.t1.a", "task_id": "t1"}
    fe2 = {"failure_id": "fail.t1.b", "task_id": "t1"}
    ss = _shared_state()
    ss.failures = [fe1, fe2]
    ss.failures_for_task = lambda tid: [fe for fe in [fe1, fe2] if fe["task_id"] == tid]
    p = mct.ContextProvider(shared_state=ss)
    out = p.get_variant_failures(task_id="t1")
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 2
    ids = {json.loads(l)["failure_id"] for l in lines}
    assert ids == {"fail.t1.a", "fail.t1.b"}


async def test_make_handler_forwards_failure_id():
    p = mct.ContextProvider(shared_state=_shared_state())
    handler = mct._make_handler(p, "get_failure")
    out = await handler({"failure_id": "fail.t1.abc"})
    assert "fail.t1.abc" in out["content"][0]["text"]
