# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The TraceLens SDK stream watchdog must not kill a running tool call."""

from __future__ import annotations

import asyncio

from hyperloom.orchestrator.trace_analysis import tracelens_skill_runner as tr


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
    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id


class TaskNotificationMessage:
    def __init__(self, task_id: str, status: str = "completed") -> None:
        self.task_id = task_id
        self.status = status


class TaskUpdatedMessage:
    def __init__(self, task_id: str, status: str) -> None:
        self.task_id = task_id
        self.patch = {"status": status}


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


def _busy_after(messages: list[object]) -> bool:
    in_flight = tr._InFlight()
    for msg in messages:
        in_flight.observe(msg)
    return in_flight.busy


class TestBackgroundTasks:
    def test_async_agents_still_running_keep_the_run_busy(self):
        """Launch returns at once while the analyzer keeps running; the killed roofline sat exactly here."""
        assert _busy_after(
            [
                AssistantMessage(ToolUseBlock("Agent")),
                TaskStartedMessage("gemm"),
                AssistantMessage(ToolResultBlock()),
                AssistantMessage(TextBlock("10 of 11 subagents complete. Only the gemm analyzer is still running.")),
            ]
        )

    def test_notification_releases_the_task(self):
        assert not _busy_after(
            [
                AssistantMessage(ToolUseBlock("Agent")),
                TaskStartedMessage("gemm"),
                AssistantMessage(ToolResultBlock()),
                TaskNotificationMessage("gemm", "completed"),
            ]
        )

    def test_terminal_update_releases_the_task(self):
        assert not _busy_after(
            [TaskStartedMessage("gemm"), AssistantMessage(ToolResultBlock()), TaskUpdatedMessage("gemm", "completed")]
        )

    def test_non_terminal_update_keeps_the_task(self):
        assert _busy_after(
            [TaskStartedMessage("gemm"), AssistantMessage(ToolResultBlock()), TaskUpdatedMessage("gemm", "running")]
        )

    def test_one_of_two_tasks_finishing_stays_busy(self):
        assert _busy_after(
            [
                TaskStartedMessage("a"),
                TaskStartedMessage("b"),
                AssistantMessage(ToolResultBlock()),
                TaskNotificationMessage("a"),
            ]
        )

    def test_a_turn_ending_on_a_background_bash_keeps_the_run_busy(self):
        """The agent ends its turn to wait on a background Bash; the session stays open until it reports."""
        assert _busy_after(
            [
                AssistantMessage(ToolUseBlock("Bash")),
                TaskStartedMessage("perf_report"),
                AssistantMessage(ToolResultBlock()),
                AssistantMessage(TextBlock("Perf report generation (Step 1) is still running in the background.")),
                ResultMessage(),
            ]
        )

    def test_the_background_task_reporting_after_the_turn_releases_it(self):
        assert not _busy_after(
            [
                TaskStartedMessage("perf_report"),
                AssistantMessage(ToolResultBlock()),
                ResultMessage(),
                TaskNotificationMessage("perf_report", "completed"),
            ]
        )

    def test_silence_with_nothing_in_flight_stays_tightly_bounded(self):
        assert not _busy_after([AssistantMessage(TextBlock("reading the skill file"))])

    def test_a_background_shell_outlives_its_turn_result(self):
        """The CLI ends the turn while the shell it backgrounded still runs, and notifies it after."""
        assert _busy_after(
            [
                AssistantMessage(ToolUseBlock("Bash")),
                TaskStartedMessage("shell"),
                AssistantMessage(ToolResultBlock()),
                AssistantMessage(TextBlock("Waiting for the command to finish.")),
                ResultMessage(),
            ]
        )

    def test_notification_after_the_turn_result_releases_the_task(self):
        assert not _busy_after(
            [
                TaskStartedMessage("shell"),
                AssistantMessage(ToolResultBlock()),
                ResultMessage(),
                TaskNotificationMessage("shell"),
            ]
        )


async def _replay(messages: list[object], silence_after: int, silence_sec: float):
    for index, message in enumerate(messages):
        if index == silence_after:
            await asyncio.sleep(silence_sec)
        yield message


def _stream_error(messages: list[object], *, silence_after: int) -> str:
    """Drive the watchdog over ``messages`` with a silence longer than the message bound before ``silence_after``."""
    bounds = tr._StreamIdleBounds(message=0.05, tool=5.0)
    stream = _replay(messages, silence_after, silence_sec=0.3)
    return asyncio.run(tr._drive_sdk_stream(stream, bounds, observe=lambda _message: None, log=None))


class TestStreamWatchdog:
    def test_waits_through_a_background_shell_past_the_turn_result(self):
        """The killed roofline: the report generator outran the CLI's foreground limit and ran on in the background."""
        messages = [
            AssistantMessage(ToolUseBlock("Bash")),
            TaskStartedMessage("shell"),
            AssistantMessage(ToolResultBlock()),
            ResultMessage(),
            TaskNotificationMessage("shell"),
            ResultMessage(),
        ]

        assert _stream_error(messages, silence_after=4) == ""

    def test_silence_after_a_turn_with_nothing_running_still_times_out(self):
        messages = [AssistantMessage(TextBlock("done")), ResultMessage(), ResultMessage()]

        assert "with no tool call in flight" in _stream_error(messages, silence_after=2)


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
