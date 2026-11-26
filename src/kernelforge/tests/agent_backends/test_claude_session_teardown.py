# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whatever a Claude session started ends with it, however the session ended."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends import claude as claude_mod
from kernelforge.utils.process_reaping import ReapReport, SessionMark
from kernelforge.tests.agent_backends._claude_fakes import _backend, _guarded_workspace, _result_message, _spec


@pytest.fixture
def reaps(monkeypatch):
    """Record every reap the backend asks for, in order, and answer with whatever the test queued."""
    calls: list[tuple] = []
    answers = {"session": ReapReport(), "workspace": ReapReport()}

    async def _session(mark, cwd):
        calls.append(("session", mark, cwd))
        return answers["session"]

    async def _workspace(cwd):
        calls.append(("workspace", cwd))
        return answers["workspace"]

    monkeypatch.setattr(claude_mod, "_reap_session_processes", _session)
    monkeypatch.setattr(claude_mod, "_reap_workspace_processes", _workspace)
    return calls, answers


def test_a_session_that_ends_on_its_own_still_has_its_processes_ended(reaps):
    calls, _answers = reaps
    captured: dict = {}
    backend = _backend([_result_message(session_id="sess-done")], captured)

    result = asyncio.run(backend.run(_spec()))

    assert result.end_reason != "timeout"
    assert [call[0] for call in calls] == ["session"]
    _kind, mark, cwd = calls[0]
    assert cwd == _guarded_workspace()
    # The mark that is reaped by is the one the CLI was started with, so everything the CLI starts carries it.
    assert isinstance(mark, SessionMark)
    assert mark.environment.items() <= captured["options"].kwargs["env"].items()
    assert result.workspace_contention == ""


def test_every_session_is_marked_apart_from_the_others(reaps):
    calls, _answers = reaps
    backend = _backend([_result_message(session_id="sess-done")], {})

    asyncio.run(backend.run(_spec()))
    asyncio.run(backend.run(_spec()))

    first, second = (call[1] for call in calls)
    assert first.token != second.token


def test_a_leftover_that_would_not_die_is_reported_even_when_the_session_ended_on_its_own(reaps):
    """The measurement that follows a session does not care why the device is still held."""
    calls, answers = reaps
    answers["session"] = ReapReport(directory=_guarded_workspace(), unkillable=(4321,), holding_device=(4321,))
    backend = _backend([_result_message(session_id="sess-done")], {})

    result = asyncio.run(backend.run(_spec()))

    assert "4321" in result.workspace_contention
    assert "survived SIGKILL" in result.workspace_contention


def test_a_session_whose_stream_failed_has_its_processes_ended_too(reaps):
    calls, _answers = reaps
    backend = _backend(
        [_result_message(session_id="sess-broken", subtype="init")], {}, stream_error=RuntimeError("stream broke")
    )

    result = asyncio.run(backend.run(_spec()))

    assert result.session_id == "sess-broken"
    assert [call[0] for call in calls] == ["session"]
