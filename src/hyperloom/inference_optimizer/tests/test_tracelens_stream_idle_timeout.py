# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The TraceLens SDK stream watchdog must not kill a running tool call.

Session 20260803T091144Z lost its roofline to this. The transcript shows the
agent announcing the trace size, issuing a ``Bash`` ``ToolUseBlock`` for
``TraceLens_generate_perf_report_pytorch``, then a ``TaskStartedMessage`` — and
nothing more, because the SDK is silent while a tool runs. The 300s
between-messages bound fired exactly 300s after the tool started and the failure
was reported as ``(gateway stall)``, though the gateway answered every probe and
the same command run by hand was still working 25 minutes later.

The fakes below mirror only what the runner reads: the message/block class names
and ``message.content``.
"""

from __future__ import annotations

from hyperloom.agents.kernel.tools import tracelens_skill_runner as tr


class ToolUseBlock:
    def __init__(self, name: str = "Bash") -> None:
        self.name = name


class ToolResultBlock:
    pass


class TextBlock:
    def __init__(self, text: str = "") -> None:
        self.text = text


class AssistantMessage:
    def __init__(self, *blocks: object) -> None:
        self.content = list(blocks)


class TaskStartedMessage:
    pass


class ResultMessage:
    def __init__(self, result: str = "done") -> None:
        self.result = result


def _drive(messages: list[object]) -> bool:
    """Replay messages through the runner's transition rule; return in-flight."""
    in_flight = False
    for msg in messages:
        transition = tr._tool_call_transition(msg)
        if transition == "start":
            in_flight = True
        elif transition == "end":
            in_flight = False
    return in_flight


class TestToolCallTransition:
    def test_tool_use_block_starts_a_call(self):
        assert tr._tool_call_transition(AssistantMessage(ToolUseBlock())) == "start"

    def test_tool_result_block_ends_a_call(self):
        assert tr._tool_call_transition(AssistantMessage(ToolResultBlock())) == "end"

    def test_plain_text_is_neutral(self):
        assert tr._tool_call_transition(AssistantMessage(TextBlock("thinking"))) is None

    def test_task_started_starts_a_call(self):
        assert tr._tool_call_transition(TaskStartedMessage()) == "start"

    def test_result_message_ends_a_call(self):
        assert tr._tool_call_transition(ResultMessage()) == "end"

    def test_the_session_sequence_leaves_a_tool_in_flight(self):
        """Announce -> ToolUse -> TaskStarted -> silence: the tool is running."""
        assert _drive(
            [
                AssistantMessage(TextBlock("The trace is ~896MB so this may take a while.")),
                AssistantMessage(ToolUseBlock("Bash")),
                TaskStartedMessage(),
            ]
        )

    def test_a_finished_tool_restores_the_tight_bound(self):
        assert not _drive(
            [
                AssistantMessage(ToolUseBlock("Bash")),
                TaskStartedMessage(),
                AssistantMessage(ToolResultBlock()),
            ]
        )

    def test_silence_before_any_tool_stays_tightly_bounded(self):
        """A real gateway stall must still be caught by the tight bound."""
        assert not _drive([AssistantMessage(TextBlock("reading the skill file"))])


class TestToolIdleTimeoutResolution:
    def test_defaults_above_the_between_message_bound(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC", raising=False)
        idle = tr._DEFAULT_STREAM_IDLE_TIMEOUT_SEC

        assert tr._resolve_tool_idle_timeout_sec(idle) > idle

    def test_never_tighter_than_the_idle_bound(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC", raising=False)

        # Widening only the idle bound must not leave the tool bound behind it.
        assert tr._resolve_tool_idle_timeout_sec(7200.0) >= 7200.0

    def test_zero_disables_the_bound(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC", "0")

        assert tr._resolve_tool_idle_timeout_sec(300.0) == 0.0

    def test_floored_at_thirty_seconds(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC", "1")

        assert tr._resolve_tool_idle_timeout_sec(0.0) == 30.0

    def test_garbage_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC", "soon")

        assert tr._resolve_tool_idle_timeout_sec(0.0) == tr._DEFAULT_TOOL_IDLE_TIMEOUT_SEC

    def test_session_default_would_have_survived_the_kill(self, monkeypatch):
        """The killed run needed >25 min; the in-flight default must cover it."""
        monkeypatch.delenv("HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC", raising=False)

        assert tr._resolve_tool_idle_timeout_sec(tr._DEFAULT_STREAM_IDLE_TIMEOUT_SEC) >= 25 * 60
