"""An API failure must resume the session; a limit the caller set must not.

A candidate Session is expensive — by the time the gateway drops it, the agent
has usually read the kernel, edited it, and paid for a build and a benchmark.
These pin that such a session is continued rather than abandoned, that a turn
cap or a deadline is left alone, and that a session the API killed is never
reported as an agent that decided to change nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kernelforge.agent_backends.base import (
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
)
from kernelforge.agent_backends.session_resume import (
    EXHAUSTED_END_REASON,
    is_api_failure,
    is_retryable_api_error,
    resumable_session_id,
    run_session_with_api_resume,
)


class _Backend:
    """A backend replaying scripted run/resume outcomes (result or exception)."""

    name = "fake"

    def __init__(self, runs, resumes=(), *, resumable=True):
        self.capabilities = SimpleNamespace(resumable=resumable)
        self._runs = list(runs)
        self._resumes = list(resumes)
        self.run_calls = 0
        self.resume_calls: list[tuple[str, str]] = []

    async def run(self, spec, usage=None):
        self.run_calls += 1
        return _replay(self._runs, self.run_calls - 1)

    async def resume(self, spec, session_id, feedback, usage=None):
        self.resume_calls.append((session_id, feedback))
        return _replay(self._resumes, len(self.resume_calls) - 1)


class _NoResumeBackend:
    """A provider that declares itself resumable but implements no resume()."""

    name = "no-resume"
    capabilities = SimpleNamespace(resumable=True)

    def __init__(self, runs):
        self._runs = list(runs)
        self.resume_calls: list[tuple[str, str]] = []

    async def run(self, spec, usage=None):
        return _replay(self._runs, 0)


def _replay(scripted, index):
    item = scripted[min(index, len(scripted) - 1)]
    if isinstance(item, Exception):
        raise item
    return item


def _api_failure(session_id="sess-1", detail="gateway 529"):
    return AgentRunResult(
        text="[session ended with SDK error: overloaded]",
        subtype="error",
        end_reason="sdk_error",
        session_id=session_id,
        stderr_tail=detail,
    )


def _spec():
    return AgentRunSpec(system_prompt="s", user_prompt="u", cwd="/tmp")


def _run(backend, **kwargs):
    kwargs.setdefault("sleep", _no_sleep)
    kwargs.setdefault("rng", lambda: 1.0)
    return asyncio.run(run_session_with_api_resume(backend, _spec(), **kwargs))


async def _no_sleep(_seconds):
    return None


# ── what counts as an API failure ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("end_reason", "expected"),
    [
        ("sdk_error", True),
        ("sdk_error_during_execution", True),
        ("api_error", True),
        ("turn_cap", False),
        ("timeout", False),
        ("agent_stopped", False),
        ("candidate_submitted", False),
        ("budget_exhausted", False),
    ],
)
def test_only_provider_failures_count_as_api_failures(end_reason, expected):
    assert is_api_failure(AgentRunResult(end_reason=end_reason)) is expected


class _FakeSafetyError(RuntimeError):
    """Stands in for WorkspaceSafetyError without importing the optional backend."""


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # Transport and gateway weather: the request never got an answer, and the
        # next one might.
        (ConnectionError("connection reset"), True),
        (RuntimeError("429 Too Many Requests"), True),
        (RuntimeError("503 Service Unavailable"), True),
        (RuntimeError("upstream connect error: connection reset by peer"), True),
        # Decisions, not weather. Each of these fails again identically.
        (RuntimeError("Error code: 400 - Bad Request"), False),
        (RuntimeError("401 unauthorized"), False),
        (RuntimeError("missing subscription key"), False),
        (_FakeSafetyError("Codex changed HEAD or the active branch"), False),
        (AgentProviderUnavailableError("claude-agent-sdk is not installed"), False),
        # A bare timeout type is the transport's; the local turn deadline is a
        # plain backend error saying the model was answering and ran out of clock.
        (asyncio.TimeoutError(), False),
        (RuntimeError("Codex timed out after 1800s"), False),
    ],
)
def test_only_transient_transport_failures_are_retried(error, expected):
    """The policy is an allowlist: retrying is the exception, not the default.

    It used to be a denylist of credentials plus ``TimeoutError``, which made a
    safety stop and a 1800s turn timeout retryable -- one wedged session could
    burn two hours re-running the same timeout four times.
    """
    assert is_retryable_api_error(error) is expected


def test_a_wrapped_transport_error_is_read_through_the_cause_chain():
    """Backends flatten the transport error into a message of their own, so the
    type that decides retryability is the ``__cause__``."""
    cause = ConnectionResetError("connection reset by peer")
    wrapped = RuntimeError("Codex SDK execution failed: [Errno 104]")
    wrapped.__cause__ = cause

    assert is_retryable_api_error(wrapped) is True


def test_a_safety_stop_stays_terminal_even_when_wrapped():
    """A rollback-triggering safety stop must never be retried, however it is
    reported: retrying it re-runs the session that violated the workspace."""
    wrapped = RuntimeError("connection reset")  # would otherwise look transient
    wrapped.__cause__ = _FakeSafetyError("Codex read-only resume changed the workspace")

    assert is_retryable_api_error(wrapped) is False


# ── a raise that still holds a live session handle ────────────────────────────


class _ErrorWithSession(RuntimeError):
    """Stands in for CodexExecutionError, which carries the thread it established."""

    def __init__(self, message, session_id=""):
        super().__init__(message)
        self.session_id = session_id


def test_a_raise_after_the_thread_exists_resumes_it_instead_of_restarting():
    """The expensive case: the transport dropped, but the thread holds the turns.

    Codex raises ``CodexExecutionError`` for a transport failure, so a session
    that had already read, built and benchmarked came back as a bare exception and
    the retry opened a BRAND NEW thread -- the exact opposite of resuming.
    """
    finished = AgentRunResult(text="PLAN: fused the rmsnorm", end_reason="agent_stopped")
    backend = _Backend(
        [_ErrorWithSession("Codex SDK execution failed: connection reset", "thread-9")],
        [finished],
    )

    result = _run(backend)

    assert backend.run_calls == 1, "a live thread must not be thrown away"
    assert backend.resume_calls[0][0] == "thread-9"
    assert result.text == "PLAN: fused the rmsnorm"
    assert result.end_reason == "agent_stopped"


def test_a_terminal_raise_is_not_resumed_even_with_a_handle():
    """A safety stop or a spent clock still ends the session, handle or not."""
    backend = _Backend([_ErrorWithSession("Codex timed out after 1800s", "thread-9")])

    with pytest.raises(_ErrorWithSession):
        _run(backend)

    assert backend.resume_calls == []


def test_the_handle_is_read_through_the_cause_chain():
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = _ErrorWithSession("inner", "thread-3")

    assert resumable_session_id(wrapped) == "thread-3"
    assert resumable_session_id(RuntimeError("nothing to resume")) == ""


# ── the retry chain has a clock, not just a count ─────────────────────────────


def test_the_resume_chain_stops_at_its_deadline():
    """The resume budget does not bound wall clock: each attempt may spend a full
    turn timeout, so an outage outliving the budget would hold the campaign."""
    # First read anchors the start; every later read is past the deadline.
    reads = iter([0.0])
    clock = lambda: next(reads, 5000.0)  # noqa: E731 - one-line fake clock
    backend = _Backend([_api_failure()], [_api_failure()])

    result = _run(
        backend,
        max_resumes=3,
        deadline_sec=600.0,
        monotonic=clock,
    )

    assert backend.resume_calls == [], "past the deadline, stop rather than retry"
    assert result.end_reason == EXHAUSTED_END_REASON


def test_deadline_zero_lifts_the_bound():
    finished = AgentRunResult(text="done", end_reason="agent_stopped")
    backend = _Backend([_api_failure()], [finished])

    result = _run(backend, deadline_sec=0.0, monotonic=lambda: 10**9)

    assert backend.resume_calls[0][0] == "sess-1"
    assert result.end_reason == "agent_stopped"


# ── resume on API failure ─────────────────────────────────────────────────────


def test_an_api_failure_resumes_the_same_session():
    finished = AgentRunResult(text="PLAN: fused the rmsnorm", end_reason="agent_stopped")
    backend = _Backend([_api_failure()], [finished])

    result = _run(backend)

    assert backend.run_calls == 1, "the session must be continued, not restarted"
    assert backend.resume_calls[0][0] == "sess-1"
    assert result.text == "PLAN: fused the rmsnorm"
    assert result.end_reason == "agent_stopped"


def test_resume_keeps_the_work_done_before_the_failure():
    before = _api_failure()
    before.tool_calls = [("Edit", {"file_path": "kernel.py"})]
    before.findings = ["baseline measured"]
    before.num_turns = 12
    after = AgentRunResult(
        text="done",
        end_reason="agent_stopped",
        num_turns=3,
        tool_calls=[("Bash", {"command": "pytest"})],
        findings=["parity ok"],
    )
    backend = _Backend([before], [after])

    result = _run(backend)

    assert [name for name, _ in result.tool_calls] == ["Edit", "Bash"]
    assert result.findings == ["baseline measured", "parity ok"]
    assert result.num_turns == 15
    assert result.session_id == "sess-1"


def test_repeated_api_failures_keep_resuming_within_budget():
    backend = _Backend(
        [_api_failure()],
        [_api_failure(), _api_failure(), AgentRunResult(end_reason="agent_stopped")],
    )

    result = _run(backend, max_resumes=3)

    assert len(backend.resume_calls) == 3
    assert result.end_reason == "agent_stopped"


def test_a_resume_that_itself_fails_is_retried():
    backend = _Backend(
        [_api_failure()],
        [ConnectionError("gateway dropped"), AgentRunResult(end_reason="agent_stopped")],
    )

    result = _run(backend, max_resumes=3)

    assert len(backend.resume_calls) == 2
    assert result.end_reason == "agent_stopped"


def test_a_resume_that_fails_on_credentials_propagates():
    backend = _Backend([_api_failure()], [RuntimeError("401 unauthorized")])
    with pytest.raises(RuntimeError, match="401"):
        _run(backend, max_resumes=3)


# ── limits the caller chose are never retried ─────────────────────────────────


@pytest.mark.parametrize("end_reason", ["turn_cap", "timeout", "agent_stopped"])
def test_a_session_that_answered_is_never_resumed(end_reason):
    backend = _Backend([AgentRunResult(end_reason=end_reason, session_id="s")])

    result = _run(backend)

    assert backend.resume_calls == []
    assert backend.run_calls == 1
    assert result.end_reason == end_reason


# ── giving up ─────────────────────────────────────────────────────────────────


def test_an_exhausted_chain_reports_api_error_not_silence():
    """Downstream reads an empty diff; only this end reason says why it is empty."""
    backend = _Backend([_api_failure()], [_api_failure()])

    result = _run(backend, max_resumes=2)

    assert len(backend.resume_calls) == 2
    assert result.end_reason == EXHAUSTED_END_REASON


def test_a_provider_that_cannot_resume_still_reports_api_error():
    backend = _Backend([_api_failure()], resumable=False)

    result = _run(backend)

    assert backend.resume_calls == []
    assert result.end_reason == EXHAUSTED_END_REASON


def test_a_failure_without_a_session_handle_reports_api_error():
    backend = _Backend([_api_failure(session_id="")])

    result = _run(backend)

    assert backend.resume_calls == []
    assert result.end_reason == EXHAUSTED_END_REASON


def test_a_backend_missing_resume_reports_api_error():
    backend = _NoResumeBackend([_api_failure()])

    result = _run(backend)

    assert result.end_reason == EXHAUSTED_END_REASON


# ── failures that precede the session ─────────────────────────────────────────


def test_a_start_that_never_reached_the_model_is_retried_fresh():
    """No session id exists yet, so a plain re-run loses nothing."""
    backend = _Backend([ConnectionError("gateway down"), AgentRunResult(end_reason="agent_stopped")])

    result = _run(backend, max_resumes=2)

    assert backend.run_calls == 2
    assert result.end_reason == "agent_stopped"


def test_a_start_that_keeps_failing_raises():
    backend = _Backend([ConnectionError("gateway down")])
    with pytest.raises(ConnectionError):
        _run(backend, max_resumes=2)
    assert backend.run_calls == 3


def test_a_start_that_fails_on_credentials_raises_immediately():
    backend = _Backend([RuntimeError("invalid api key")])
    with pytest.raises(RuntimeError, match="invalid api key"):
        _run(backend, max_resumes=3)
    assert backend.run_calls == 1


# ── operator overrides ────────────────────────────────────────────────────────


def test_the_retry_budget_is_tunable_without_a_redeploy(monkeypatch):
    monkeypatch.setenv("FORGE_AGENT_API_MAX_RESUMES", "1")
    backend = _Backend([_api_failure()], [_api_failure()])

    result = _run(backend)

    assert len(backend.resume_calls) == 1
    assert result.end_reason == EXHAUSTED_END_REASON


def test_backoff_grows_between_resume_attempts():
    slept: list[float] = []

    async def record(seconds):
        slept.append(seconds)

    backend = _Backend([_api_failure()], [_api_failure(), _api_failure(), _api_failure()])
    asyncio.run(
        run_session_with_api_resume(
            backend,
            _spec(),
            max_resumes=3,
            base_delay_sec=5.0,
            max_delay_sec=120.0,
            sleep=record,
            rng=lambda: 1.0,
        )
    )
    assert slept == [5.0, 15.0, 45.0]
