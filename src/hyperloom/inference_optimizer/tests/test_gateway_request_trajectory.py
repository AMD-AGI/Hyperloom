# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``llm.request`` rows for the single HTTP requests the ``llm_config`` helpers issue, and for specialist logs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hyperloom.common import llm_config
from hyperloom.common import llm_request_hooks as hooks
from hyperloom.inference_optimizer.trace import request_events as re_
from hyperloom.inference_optimizer.trace import trajectory_trace as tt


def _requests(session_dir) -> list[dict]:
    return [r for r in tt.load_events(session_dir) if r["event_type"] == tt.EVENT_LLM_REQUEST]


def _chat_response(**overrides):
    fields = {
        "id": "chatcmpl-abc",
        "model": "gpt-x",
        "choices": [SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        "usage": SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=7,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        ),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _AsyncChatClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **_params):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_the_trajectory_observer_is_registered_on_import():
    assert tt.record_gateway_request in hooks._OBSERVERS


@pytest.mark.asyncio
async def test_a_chat_completion_is_one_request_under_the_ambient_call(tmp_path):
    client = _AsyncChatClient(_chat_response())
    with tt.trajectory_scope(session_dir=tmp_path, component="coordinator", call_id="c-1", parent_span_id="p-1"):
        result = await llm_config.achat_completion(client, component="critic", operation="review", model="gpt-x")

    assert result.text == "ok"
    (row,) = _requests(tmp_path)
    assert row["status"] == tt.STATUS_COMPLETED
    assert (row["component"], row["call_id"], row["parent_span_id"]) == ("critic", "c-1", "p-1")
    attributes = row["attributes"]
    assert attributes["message_id"] == "chatcmpl-abc"
    assert attributes["operation"] == "review"
    assert attributes["protocol"] == hooks.PROTOCOL_OPENAI_CHAT
    assert attributes["timing_source"] == tt.TIMING_HTTP_RESPONSE
    assert (attributes["input_tokens"], attributes["cache_read_input_tokens"]) == (60, 40)
    assert attributes["output_tokens"] == 7
    assert attributes["stop_reason"] == "stop"
    assert row["start_ts"] <= row["ts"]


@pytest.mark.asyncio
async def test_a_failed_request_is_a_failed_row_and_still_raises(tmp_path):
    client = _AsyncChatClient(RuntimeError("gateway 502"))
    with tt.trajectory_scope(session_dir=tmp_path, component="coordinator"):
        with pytest.raises(RuntimeError, match="gateway 502"):
            await llm_config.achat_completion(client, component="forge", model="m")

    (row,) = _requests(tmp_path)
    assert row["status"] == tt.STATUS_FAILED
    assert row["component"] == "forge"
    assert row["attributes"]["error_type"] == "RuntimeError"
    assert row["attributes"]["complete"] is False
    assert row["attributes"]["message_id"] is None


@pytest.mark.asyncio
async def test_an_unknown_call_site_component_keeps_the_ambient_one(tmp_path):
    client = _AsyncChatClient(_chat_response())
    with tt.trajectory_scope(session_dir=tmp_path, component="coordinator"):
        await llm_config.achat_completion(client, component="not_a_component", model="m")

    (row,) = _requests(tmp_path)
    assert row["component"] == "coordinator"


def test_a_streamed_completion_reads_id_and_usage_off_its_chunks(tmp_path):
    def chunk(content=None, finish=None, usage=None):
        choices = []
        if content is not None or finish is not None:
            choices = [SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)]
        return SimpleNamespace(id="chatcmpl-s", model="gpt-s", choices=choices, usage=usage)

    chunks = [
        chunk(content="he"),
        chunk(content="llo", finish="stop"),
        chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2)),
    ]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: iter(chunks))))
    with tt.trajectory_scope(session_dir=tmp_path, component="coordinator"):
        text, _usage = llm_config.stream_chat_completion_text(client, component="proposal_scorer", model="gpt-s")

    assert text == "hello"
    (row,) = _requests(tmp_path)
    attributes = row["attributes"]
    assert row["component"] == "proposal_scorer"
    assert attributes["message_id"] == "chatcmpl-s"
    assert attributes["timing_source"] == tt.TIMING_HTTP_STREAM
    assert attributes["ttft_ms"] is not None
    assert (attributes["input_tokens"], attributes["output_tokens"]) == (10, 2)
    assert attributes["stop_reason"] == "stop"


def test_an_anthropic_messages_post_is_one_request(tmp_path):
    body = {
        "id": "msg_01",
        "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3, "cache_read_input_tokens": 90},
    }
    resp = SimpleNamespace(status_code=200, json=lambda: body)
    client = SimpleNamespace(post=lambda *_a, **_k: resp)
    with tt.trajectory_scope(session_dir=tmp_path, component="coordinator"):
        result = llm_config.anthropic_messages(client, component="breakdown", model="claude-x", max_tokens=8)

    assert result.text == "ok"
    (row,) = _requests(tmp_path)
    attributes = row["attributes"]
    assert attributes["message_id"] == "msg_01"
    assert attributes["protocol"] == hooks.PROTOCOL_ANTHROPIC_MESSAGES
    assert (attributes["input_tokens"], attributes["cache_read_input_tokens"]) == (5, 90)
    assert attributes["cache_creation_input_tokens"] == 0


def test_an_observer_fault_never_reaches_the_request(tmp_path):
    def broken(_record):
        raise ValueError("observer bug")

    hooks.add_llm_request_observer(broken)
    try:
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: _chat_response())))
        assert llm_config.chat_completion(client, component="critic", model="m").text == "ok"
    finally:
        hooks.remove_llm_request_observer(broken)


def test_requests_outside_a_session_write_nothing(tmp_path):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: _chat_response())))
    llm_config.chat_completion(client, component="critic", model="m")
    assert not _requests(tmp_path)


def _write_log(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{torn", encoding="utf-8")


def _assistant(message_id: str, *, output: int, model: str = "claude-x", parent=None) -> dict:
    return {
        "type": "assistant",
        "parent_tool_use_id": parent,
        "message": {
            "id": message_id,
            "model": model,
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": 3, "cache_read_input_tokens": 1000, "output_tokens": output},
        },
    }


def test_specialist_log_requests_are_one_row_per_message_id(tmp_path):
    log = tmp_path / "process.log"
    _write_log(
        log,
        [
            {"type": "system", "subtype": "init", "model": "claude-x"},
            _assistant("m1", output=10),
            _assistant("m1", output=10),
            _assistant("m2", output=5, parent="tu-1"),
            _assistant("m3", output=1, model="<synthetic>"),
            {"type": "result", "usage": {"output_tokens": 15}},
        ],
    )
    with tt.trajectory_scope(session_dir=tmp_path, component="specialist", task_id="t-1", parent_span_id="call-1"):
        assert re_.record_stream_json_requests(log) == 2

    first, second = _requests(tmp_path)
    assert [r["attributes"]["message_id"] for r in (first, second)] == ["m1", "m2"]
    assert {r["task_id"] for r in (first, second)} == {"t-1"}
    assert {r["parent_span_id"] for r in (first, second)} == {"call-1"}
    assert first["attributes"]["cache_read_input_tokens"] == 1000
    assert first["attributes"]["output_tokens_exact"] is True
    assert second["attributes"]["parent_tool_use_id"] == "tu-1"
    assert first["attributes"]["timing_source"] == "none"


def test_specialist_log_output_is_flagged_inexact_when_it_does_not_add_up(tmp_path):
    log = tmp_path / "process.log"
    _write_log(
        log,
        [_assistant("m1", output=1), _assistant("m2", output=1), {"type": "result", "usage": {"output_tokens": 900}}],
    )
    rows = re_.stream_json_requests(log)
    assert [r["output_tokens_exact"] for r in rows] == [False, False]
    assert re_.stream_json_requests(tmp_path / "missing.log") == []
