# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A DeepSeek Harness session is continued until the artifact it owes exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.common import agent_sdk_runner
from hyperloom.common.deepseek_session import DeepSeekSessionResult


class _FakeSession:
    """Record every prompt; run a scripted side effect per turn."""

    instances: list["_FakeSession"] = []

    def __init__(self, *, effects: list[Any], **_kwargs: Any) -> None:
        self.prompts: list[str] = []
        self.turn_timeouts: list[float] = []
        self.effects = effects
        self.request_timeout_sec = 60.0
        _FakeSession.instances.append(self)

    async def start(self) -> None:
        return None

    async def turn(self, prompt: str, *, timeout_sec: float) -> DeepSeekSessionResult:
        self.prompts.append(prompt)
        self.turn_timeouts.append(timeout_sec)
        effect = self.effects[len(self.prompts) - 1] if len(self.prompts) <= len(self.effects) else None
        if effect is not None:
            effect()
        return DeepSeekSessionResult(text=f"turn {len(self.prompts)}", session_id="s1")

    async def aclose(self) -> None:
        return None


def _install(monkeypatch: pytest.MonkeyPatch, effects: list[Any]) -> None:
    _FakeSession.instances.clear()
    monkeypatch.setattr(agent_sdk_runner, "DeepSeekSession", lambda **kw: _FakeSession(effects=effects, **kw))


async def _drive(tmp_path: Path, report: Path, **kwargs: Any) -> Any:
    return await agent_sdk_runner.drive_agent_session(
        prompt="run the skill",
        system_prompt="",
        cwd=tmp_path,
        allowed_tools=["Read", "Write"],
        backend="deepseek",
        done=report.exists,
        continuation_prompt=f"Write {report}.",
        **kwargs,
    )


async def test_a_session_that_stopped_short_is_continued_until_its_report_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live TraceLens run ended on a tool call written as text, and analysis.md was never written."""
    report = tmp_path / "analysis.md"
    _install(monkeypatch, [None, lambda: report.write_text("done")])

    result = await _drive(tmp_path, report)

    session = _FakeSession.instances[0]
    assert len(session.prompts) == 2
    assert session.prompts[1] == f"Write {report}."
    assert report.exists()
    assert "turn 2" in result.text


async def test_continuations_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rule the agent cannot satisfy must not spend the rest of the budget failing in place."""
    report = tmp_path / "analysis.md"
    _install(monkeypatch, [])

    await _drive(tmp_path, report, max_continuations=2)

    assert len(_FakeSession.instances[0].prompts) == 3


async def test_a_session_that_finished_is_not_continued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "analysis.md"
    _install(monkeypatch, [lambda: report.write_text("done")])

    await _drive(tmp_path, report)

    assert len(_FakeSession.instances[0].prompts) == 1


async def test_the_caller_budget_bounds_every_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A TraceLens budget of a few minutes must not leave a turn running for the harness default of two hours."""
    report = tmp_path / "analysis.md"
    _install(monkeypatch, [])

    await _drive(tmp_path, report, max_continuations=1, timeout_sec=30.0)

    timeouts = _FakeSession.instances[0].turn_timeouts
    assert len(timeouts) == 2
    assert all(0.0 < t <= 30.0 for t in timeouts)


async def test_a_spent_budget_stops_the_continuations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "analysis.md"
    clock = iter([0.0, 0.0, 31.0, 31.0])
    monkeypatch.setattr(agent_sdk_runner, "monotonic", lambda: next(clock))
    _install(monkeypatch, [])

    result = await _drive(tmp_path, report, max_continuations=3, timeout_sec=30.0)

    assert len(_FakeSession.instances[0].prompts) == 1
    assert "budget" in result.error
