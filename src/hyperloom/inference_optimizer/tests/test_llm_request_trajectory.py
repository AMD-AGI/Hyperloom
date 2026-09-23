# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-request LLM usage/timing on the trajectory ledger, and the canonical uncached-input semantics."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest

from hyperloom.common.token_usage import uncached_input_tokens
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.roles import claude as cl
from hyperloom.orchestrator.roles.claude_requests import (
    TIMING_ASSISTANT_MESSAGE,
    TIMING_STREAM,
    ClaudeRequestTracker,
)
from hyperloom.orchestrator.roles.critic_agent import CriticAgentBackend
from hyperloom.orchestrator.trace import parse_usage as pu
from hyperloom.orchestrator.trace import trajectory_trace as tt

sdk_types = pytest.importorskip("claude_agent_sdk.types")


def _clock(*ticks: float):
    it = iter(ticks)
    return lambda: next(it)


def _stream(event: dict, parent: str | None = None):
    return sdk_types.StreamEvent(uuid="u", session_id="s", event=event, parent_tool_use_id=parent)


def _assistant(message_id: str, **over):
    fields = {"content": [], "model": "claude-x", "message_id": message_id}
    fields.update(over)
    return sdk_types.AssistantMessage(**fields)


def _tool_result():
    return sdk_types.UserMessage(content=[sdk_types.ToolResultBlock(tool_use_id="t1", content="ok")])


def _request_events(message_id: str, *, input_tokens: int, output_tokens: int) -> list:
    return [
        _stream(
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "model": "claude-x",
                    "usage": {"input_tokens": input_tokens, "cache_read_input_tokens": 900, "output_tokens": 1},
                },
            }
        ),
        _stream({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "a"}}),
        _stream({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "b"}}),
        _stream(
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": output_tokens}}
        ),
        _stream({"type": "message_stop"}),
    ]


def test_stream_events_time_each_request_from_its_boundary():
    # t=0 open; request 1 events at 1..5; tool result at 7; request 2 events at 8..12.
    clock = _clock(0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0)
    tracker = ClaudeRequestTracker(clock=clock)
    for message in _request_events("m1", input_tokens=10, output_tokens=40):
        tracker.observe(message)
    tracker.observe(_tool_result())
    for message in _request_events("m2", input_tokens=3, output_tokens=7):
        tracker.observe(message)

    first, second = tracker.records()
    assert (first["message_id"], first["timing_source"], first["complete"]) == ("m1", TIMING_STREAM, True)
    assert (first["ttft_ms"], first["latency_ms"]) == (1500, 5000)
    assert (first["input_tokens"], first["output_tokens"], first["cache_read_input_tokens"]) == (10, 40, 900)
    assert first["stop_reason"] == "tool_use"
    assert (second["ttft_ms"], second["latency_ms"]) == (1500, 3000)
    assert second["output_tokens"] == 7


def test_assistant_messages_are_the_fallback_deduped_by_message_id():
    clock = _clock(0.0, 2.0, 3.0, 6.0, 9.0)
    tracker = ClaudeRequestTracker(clock=clock)
    tracker.observe(_assistant("m1", usage={"input_tokens": 5, "output_tokens": 1}))
    tracker.observe(_assistant("m1", usage={"input_tokens": 5, "output_tokens": 30}, stop_reason="tool_use"))
    tracker.observe(_tool_result())
    tracker.observe(_assistant("m2", usage={"input_tokens": 2, "output_tokens": 4}))

    first, second = tracker.records()
    assert first["timing_source"] == TIMING_ASSISTANT_MESSAGE
    assert (first["output_tokens"], first["stop_reason"], first["latency_ms"], first["ttft_ms"]) == (
        30,
        "tool_use",
        3000,
        None,
    )
    assert second["latency_ms"] == 3000


def test_a_cut_off_request_is_recorded_failed_under_the_open_call(tmp_path):
    clock = _clock(0.0, 1.0, 2.0, 4.0)
    tracker = ClaudeRequestTracker(clock=clock)
    for message in _request_events("m1", input_tokens=10, output_tokens=40)[:2]:
        tracker.observe(message)
    with tt.trajectory_scope(session_dir=tmp_path, component="orchestration", call_id="c-1"):
        with tt.trajectory_span(tt.EVENT_LLM_CALL) as span:
            tracker.record_trajectory(attempt=2, fallback_model="fallback")

    (request,) = [r for r in tt.load_events(tmp_path) if r["event_type"] == tt.EVENT_LLM_REQUEST]
    assert request["status"] == tt.STATUS_FAILED
    assert request["parent_span_id"] == span.span_id
    assert request["call_id"] == "c-1"
    assert request["attributes"]["complete"] is False
    assert request["attributes"]["attempt"] == 2
    assert request["attributes"]["model"] == "claude-x"
    assert request["start_ts"] < request["ts"]


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _emit_block():
    return sdk_types.ToolUseBlock(
        id="tu1",
        name=cl.EMIT_INTENT_TOOL_QUALIFIED,
        input={"intent_type": "send_message", "payload": {"topic": "t", "body_md": "ok"}},
    )


@pytest.mark.asyncio
async def test_claude_backend_streams_partials_and_shares_the_scope_call_id(tmp_path):
    seen_options: list = []

    async def _query(*, prompt, options):
        seen_options.append(options)
        for message in _request_events("m1", input_tokens=10, output_tokens=40):
            yield message
        yield _assistant("m1", content=[_emit_block()])
        yield sdk_types.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            usage={"input_tokens": 10, "output_tokens": 40},
        )

    backend = cl.ClaudeBackend(
        sdk_query_factory=_query,
        sdk_options_cls=_FakeOptions,
        api_key_env="UNSET_KEY_ENV_FOR_TEST",
        model="claude-x",
        capture_turn_diagnostics=True,
    )
    with tt.trajectory_scope(session_dir=tmp_path, component="orchestration", call_id="c-7"):
        result = await backend.run("hello")

    assert seen_options[0].kwargs["include_partial_messages"] is True
    assert result.metadata["call_id"] == "c-7"
    requests = [r for r in tt.load_events(tmp_path) if r["event_type"] == tt.EVENT_LLM_REQUEST]
    assert [(r["status"], r["attributes"]["output_tokens"]) for r in requests] == [(tt.STATUS_COMPLETED, 40)]
    diag_types = {m["type"] for m in backend.get_turn_diagnostic()["messages"]}
    assert diag_types == {"AssistantMessage", "ResultMessage"}


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


@pytest.mark.asyncio
async def test_each_reactor_turn_is_an_llm_call_span_under_the_session(session_dir):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    c = Coordinator(
        session_dir,
        backends={"orchestration": MockBackend(silent, name="o"), "critic": MockBackend(silent, name="c")},
    )
    try:
        await c.run(max_ticks=1)
    finally:
        await c.stop()
    rows = tt.load_events(session_dir)
    (session,) = [r for r in rows if r["event_type"] == tt.EVENT_SESSION and r["status"] == tt.STATUS_STARTED]
    calls = [r for r in rows if r["event_type"] == tt.EVENT_LLM_CALL]
    started = [r for r in calls if r["status"] == tt.STATUS_STARTED]
    closed = [r for r in calls if r["status"] == tt.STATUS_COMPLETED]
    assert {r["agent"] for r in started} == {"orchestration", "critic"}
    assert {r["parent_span_id"] for r in started} == {session["span_id"]}
    assert sorted(r["span_id"] for r in started) == sorted(r["span_id"] for r in closed)
    assert all(r["call_id"] for r in started)


def test_uncached_input_subtracts_the_cached_prefix():
    assert uncached_input_tokens(100, 30) == 70
    assert uncached_input_tokens(100, None) == 100
    assert uncached_input_tokens(10, 30) == 0
    assert uncached_input_tokens(None, 5) is None


def test_critic_openai_usage_reports_cached_tokens_separately():
    acc = {"input_tokens": 0, "output_tokens": 0}
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        prompt_tokens_details=SimpleNamespace(cached_tokens=800),
    )
    CriticAgentBackend._accumulate_usage(acc, usage)
    CriticAgentBackend._accumulate_usage(acc, SimpleNamespace(prompt_tokens=10, completion_tokens=5))
    assert acc["input_tokens"] == 200 + 10
    assert acc["cache_read_input_tokens"] == 800
    assert acc["output_tokens"] == 55


def test_claude_stream_json_usage_names_the_serving_model(tmp_path):
    log = tmp_path / "process.log"
    rows = [
        {"type": "system", "subtype": "init", "model": "claude-init"},
        {"type": "assistant", "message": {"id": "m1", "model": "<synthetic>", "usage": {"input_tokens": 1}}},
        {"type": "assistant", "message": {"id": "m2", "model": "claude-real", "usage": {"input_tokens": 2}}},
        {"type": "result", "usage": {"input_tokens": 3, "output_tokens": 4}},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    usage = pu.parse_claude_stream_json_usage(log)
    assert usage is not None
    assert usage["model"] == "claude-real"
    assert usage["input_tokens"] == 3


def test_request_rows_count_is_bounded_by_requests_not_stream_events():
    ticks = itertools.count()
    tracker = ClaudeRequestTracker(clock=lambda: float(next(ticks)))
    for message in _request_events("m1", input_tokens=1, output_tokens=1):
        tracker.observe(message)
    tracker.observe(_assistant("m1"))
    assert len(tracker.records()) == 1
