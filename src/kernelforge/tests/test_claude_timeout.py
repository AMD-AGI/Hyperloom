"""The Claude backend must honour ``AgentRunSpec.timeout_sec``.

The SDK stream is an unbounded ``async for``: the model keeps the turn until it
answers, hits its turn cap, or the transport fails. Nothing here bounds the wall
clock, so a session that neither answers nor caps runs until something outside
kills it -- the exact gap that let long implementer sessions burn the campaign's
clock (the turn cap fired 2.2% of the time and could not bound time at all).

These tests are GPU-free and SDK-free: ``query`` is a fake async generator that
either hangs past the deadline or completes normally, and the subprocess reaper
is replaced with a recorder so no real ``/proc`` scan runs. They pin the
contract the resume/orchestrator layers depend on: a session that outlives its
budget is stopped, its handle preserved, its leftovers reaped, and its end
reason reported as the terminal ``timeout`` -- not the retryable ``sdk_error``.

That the reaper itself works is a separate question, answered against real
processes in ``tests/test_process_reaping.py``.
"""

from __future__ import annotations

import asyncio
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends import claude as claude_mod
from kernelforge.agent_backends.base import AgentRunSpec, AgentToolPolicy
from kernelforge.agent_backends.claude import ClaudeBackend
from kernelforge.llm.git import git
from kernelforge.llm.process_reaping import ReapReport


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _message(**fields):
    return SimpleNamespace(content=[], **fields)


@lru_cache(maxsize=1)
def _guarded_workspace() -> str:
    """A real worktree: a writable session is judged against one."""
    root = Path(tempfile.mkdtemp(prefix="forge-claude-timeout-"))
    (root / "kernel.py").write_text("VALUE = 0\n")
    git("init", "--quiet", cwd=root)
    git("config", "user.email", "t@test", cwd=root)
    git("config", "user.name", "t", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-m", "baseline", cwd=root)
    return str(root)


def _spec(**overrides) -> AgentRunSpec:
    base = dict(
        system_prompt="implementer system prompt",
        user_prompt="optimize the kernel",
        cwd=_guarded_workspace(),
        model="fake-model",
        timeout_sec=0.05,
        tool_policy=AgentToolPolicy(read=True, write=True, shell=True, max_turns=100),
    )
    base.update(overrides)
    return AgentRunSpec(**base)


def _hanging_backend(messages, captured):
    """A ClaudeBackend whose SDK yields ``messages`` then never completes."""
    backend = ClaudeBackend.__new__(ClaudeBackend)
    backend.runtime = SimpleNamespace(
        provider="claude",
        model="fake-model",
        executable="",
        timeout_sec=60,
        reasoning_effort="high",
        options={},
    )
    backend.fallback_reason = ""

    async def fake_query(prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        for message in messages:
            yield message
        # The model is still "thinking": no ResultMessage, no error, ever.
        await asyncio.sleep(3600)
        yield _message(subtype="never")  # pragma: no cover - unreachable

    backend._query = fake_query
    backend._options_type = _FakeOptions
    return backend


def _run_bounded(coro, *, guard_sec=5.0):
    """Run ``coro`` under an outer guard that trips only if nothing bounds it.

    Before the backend honours its own deadline the fake stream hangs forever,
    so this guard is what turns "the bug is present" into a clean failure rather
    than a wedged worker. Once the deadline is honoured the coroutine returns
    well inside the guard and the guard never fires.
    """

    async def _guarded():
        return await asyncio.wait_for(coro, timeout=guard_sec)

    return asyncio.run(_guarded())


def test_timeout_after_session_id_returns_a_resumable_terminal_result(monkeypatch):
    reaped: list[str] = []

    async def _record_reap(cwd):
        reaped.append(cwd)
        return ReapReport(directory=cwd)

    monkeypatch.setattr(claude_mod, "_reap_workspace_processes", _record_reap)
    captured: dict = {}
    backend = _hanging_backend(
        [
            _message(subtype="init", session_id="sess-timeout"),
            SimpleNamespace(content=[SimpleNamespace(text="partial work")]),
        ],
        captured,
    )

    result = _run_bounded(backend.run(_spec()))

    # A deadline is an answer, not weather: terminal, never retried.
    assert result.end_reason == "timeout"
    assert result.session_id == "sess-timeout"
    assert "partial work" in result.text
    assert "timed out" in result.stderr_tail.lower()
    # The CLI's GPU-holding leftovers must be reaped from the session's cwd, or
    # they corrupt the canonical measurement that follows.
    assert reaped == [_guarded_workspace()]
    # A reap that cleared the workspace leaves nothing for the loop to act on.
    assert result.workspace_contention == ""


def test_a_workspace_the_reaper_could_not_clear_is_reported_on_the_result(
    monkeypatch,
):
    """The reaper is best effort; the measurement that follows it is not.

    A leftover that survived SIGKILL, or one that belongs to someone else and is
    therefore not ours to kill, is still holding the device. The backend is the
    only place that knows, and the loop is the only place that can decline to
    benchmark, so the finding has to travel on the result.
    """

    async def _contended_reap(cwd):
        return ReapReport(directory=cwd, unkillable=(4321,), holding_device=(4321,))

    monkeypatch.setattr(claude_mod, "_reap_workspace_processes", _contended_reap)
    captured: dict = {}
    backend = _hanging_backend([_message(subtype="init", session_id="sess-timeout")], captured)

    result = _run_bounded(backend.run(_spec()))

    assert result.end_reason == "timeout"
    assert "4321" in result.workspace_contention
    assert "survived SIGKILL" in result.workspace_contention


def test_timeout_reports_a_terminal_reason_not_sdk_error(monkeypatch):
    """``timeout`` is in TERMINAL_END_REASONS; ``sdk_error`` would be retried."""
    from kernelforge.agent_backends.session_resume import (
        TERMINAL_END_REASONS,
        is_api_failure,
    )

    async def _noop_reap(cwd):
        return ReapReport(directory=cwd)

    monkeypatch.setattr(claude_mod, "_reap_workspace_processes", _noop_reap)
    captured: dict = {}
    backend = _hanging_backend([_message(subtype="init", session_id="sess-timeout")], captured)

    result = _run_bounded(backend.run(_spec()))

    assert result.end_reason in TERMINAL_END_REASONS
    assert is_api_failure(result) is False


def test_timeout_before_any_session_id_is_raised(monkeypatch):
    async def _noop_reap(cwd):
        return ReapReport(directory=cwd)

    monkeypatch.setattr(claude_mod, "_reap_workspace_processes", _noop_reap)
    captured: dict = {}
    backend = _hanging_backend([], captured)

    with pytest.raises(Exception) as excinfo:
        _run_bounded(backend.run(_spec()))
    # Nothing was established, so there is no handle to resume; the failure
    # precedes the session and must unwind like the pre-init stream error does.
    assert "timed out" in str(excinfo.value).lower()
