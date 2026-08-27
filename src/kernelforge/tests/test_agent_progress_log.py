"""The progress sink must survive a cancelled agent run.

A prep attempt that hits its wall-clock cap is cancelled mid-stream, so whatever
the backend accumulated in locals is lost — a real run left nothing behind but
``{"status": "timeout", "elapsed_s": 900.132}`` for 900 seconds of agent work.
``AgentRunSpec.progress_log`` is owned by the caller, so it still holds what the
agent was doing after the cancellation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends import claude as claude_backend
from kernelforge.loop.task_preparer import summarize_agent_progress


class _TextBlock:
    def __init__(self, text):
        self.text = text


class ToolUseBlock:  # name matters: the backend dispatches on __class__.__name__
    def __init__(self, name, payload):
        self.name = name
        self.input = payload


def _msg(*blocks):
    return SimpleNamespace(content=list(blocks))


def test_records_text_and_tool_calls():
    sink: list[str] = []
    claude_backend._record_progress(sink, _msg(_TextBlock("Let me read the spec first")))
    claude_backend._record_progress(sink, _msg(ToolUseBlock("Read", {"file_path": "/ws/.forge_driver.py"})))
    claude_backend._record_progress(sink, _msg(ToolUseBlock("Bash", {"command": "python driver.py --smoke"})))

    assert sink == [
        "say: Let me read the spec first",
        "tool: Read /ws/.forge_driver.py",
        "tool: Bash python driver.py --smoke",
    ]


def test_sink_is_bounded():
    sink: list[str] = []
    for index in range(claude_backend._PROGRESS_MAX_ENTRIES + 50):
        claude_backend._record_progress(sink, _msg(_TextBlock(f"step {index}")))

    assert len(sink) == claude_backend._PROGRESS_MAX_ENTRIES
    # Oldest entries are the ones dropped.
    assert sink[-1].endswith(f"step {claude_backend._PROGRESS_MAX_ENTRIES + 49}")


def test_none_sink_and_malformed_messages_are_ignored():
    claude_backend._record_progress(None, _msg(_TextBlock("dropped")))

    sink: list[str] = []
    claude_backend._record_progress(sink, SimpleNamespace())  # no content attribute
    claude_backend._record_progress(sink, SimpleNamespace(content=None))
    assert sink == []


def test_sink_survives_cancellation_of_the_streaming_run():
    """This is the whole point: the caller keeps the record, not the backend."""
    sink: list[str] = []

    async def stream_forever():
        claude_backend._record_progress(sink, _msg(ToolUseBlock("Read", {"file_path": "spec.json"})))
        claude_backend._record_progress(sink, _msg(ToolUseBlock("Grep", {"pattern": "paged_attention"})))
        await asyncio.sleep(60)

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(stream_forever(), timeout=0.05)

    asyncio.run(main())

    assert sink == ["tool: Read spec.json", "tool: Grep paged_attention"]


def test_summary_counts_tools_and_keeps_the_tail():
    summary = summarize_agent_progress(
        [
            "tool: Read a.py",
            "tool: Read b.py",
            "tool: Grep foo",
            "say: thinking about it",
            "tool: Read c.py",
        ]
    )

    assert "Readx3" in summary
    assert "Grepx1" in summary
    assert "last steps:" in summary
    assert "tool: Read c.py" in summary


def test_summary_calls_out_an_agent_that_did_nothing():
    assert "no tool activity" in summarize_agent_progress([])
    assert "no tool calls at all" in summarize_agent_progress(["say: hmm"])
