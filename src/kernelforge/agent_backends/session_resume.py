# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Continue an agent session that the LLM API cut short.

A session stops for one of two very different reasons. Either the model
answered — it finished, it hit its turn cap, or its deadline expired — or the
API never answered at all, because the gateway returned an error or the stream
dropped mid-turn. Only the second kind deserves another attempt, and that
attempt has to be a RESUME: by the time the gateway fails, the session has
usually already read the kernel, built it, and benchmarked it, and a fresh
session throws every one of those turns away.

Retrying the first kind is always wrong. A turn cap and a deadline are limits
the caller chose, and a finished session is an answer; re-running either one
buys nothing and spends the campaign's budget twice.

The distinction also has to survive into the result. A session the API killed
produced no candidate, which looks exactly like a session that produced no
candidate on purpose — so an exhausted retry chain reports ``api_error``
rather than leaving the caller to record "the agent changed nothing".
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Callable

from kernelforge.agent_backends.base import (
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
)

log = logging.getLogger(__name__)

# Backends prefix a provider-side failure with ``sdk_`` (see ClaudeBackend).
API_FAILURE_PREFIX = "sdk_"
# The session reached a limit or an answer. Never retried, whatever it cost.
TERMINAL_END_REASONS = frozenset(
    {
        "agent_stopped",
        "budget_exhausted",
        "candidate_submitted",
        "gate_met",
        "timeout",
        "turn_cap",
    }
)
# End reason for a session the API killed and that could not be recovered. It
# is deliberately NOT one of the reasons above: nothing was measured, so no
# downstream reader may treat it as a verdict about the kernel.
EXHAUSTED_END_REASON = "api_error"

DEFAULT_MAX_RESUMES = 3
DEFAULT_BASE_DELAY_SEC = 5.0
DEFAULT_MAX_DELAY_SEC = 120.0
_DELAY_FACTOR = 3.0
# Ceiling on the wall clock the retry/resume chain may add. A session that keeps
# failing must give the campaign its remaining budget back rather than spending
# hours discovering the same outage; 0 lifts the bound.
DEFAULT_DEADLINE_SEC = 3600.0

# Exception types that mean "this request never got an answer, and the next one
# might". Matched by name because the SDKs wrap httpx/aiohttp/openai errors and
# importing those to isinstance-check them would make optional deps mandatory.
_TRANSIENT_TYPE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ClientConnectorError",
        "ClientOSError",
        "ClientPayloadError",
        "ConnectError",
        "ConnectTimeout",
        "IncompleteRead",
        "InternalServerError",
        "OverloadedError",
        "PoolTimeout",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "ServerDisconnectedError",
        "ServiceUnavailableError",
        "WriteError",
        "WriteTimeout",
    }
)
# Substrings that identify the same failures once a backend has flattened them
# into a message (CodexExecutionError does this), keyed to what gateways and
# proxies actually emit. A local turn timeout is deliberately absent: it means
# the model was answering and ran out of clock, so re-running it just burns the
# same clock again.
_TRANSIENT_MARKERS = (
    "429",
    "500 internal",
    "502",
    "503",
    "504",
    "bad gateway",
    "broken pipe",
    "connection aborted",
    "connection refused",
    "connection reset",
    "eof occurred",
    "gateway time-out",
    "gateway timeout",
    "incomplete chunked read",
    "internal server error",
    "overloaded",
    "rate limit",
    "rate_limit",
    "remote end closed",
    "server disconnected",
    "service unavailable",
    "temporarily unavailable",
    "too many requests",
)

RESUME_PROMPT = (
    "The previous turn was cut short by an API error on our side, not by "
    "anything you did, and not by any limit you reached. You are still in the "
    "SAME session and your earlier work is intact. Re-check the current state "
    "of the files you were editing, continue exactly where you stopped, and "
    "finish the turn normally."
)


def _env_number(name: str, default: float, *, cast: Callable[[str], Any]) -> Any:
    """Read one operator override, ignoring anything unparseable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        log.warning("ignoring unparseable %s=%r", name, raw)
        return default
    return value if value >= 0 else default


def is_api_failure(result: Any) -> bool:
    """Whether this session ended because the API failed, not because it answered."""
    reason = str(getattr(result, "end_reason", "") or "").strip()
    if reason in TERMINAL_END_REASONS:
        return False
    return reason.startswith(API_FAILURE_PREFIX) or reason == EXHAUSTED_END_REASON


def _error_chain(error: BaseException) -> list[BaseException]:
    """The exception and everything it was raised from.

    Backends flatten the transport error into a message of their own
    (``CodexExecutionError(f"Codex SDK execution failed: {exc}")``), so the type
    that says whether a retry can help is usually the ``__cause__``, not the
    exception the caller sees.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_retryable_api_error(error: BaseException) -> bool:
    """Whether a failed session is worth another attempt.

    Retrying is the exception, not the default. Only a transport or gateway
    failure — a reset connection, a 5xx, a rate limit — stops happening on its
    own; a rejected request, a workspace-safety stop, and an expired clock all
    fail again identically, and each retry spends the campaign's budget and
    (for a timeout) its wall clock to learn nothing.

    That is why the check is an allowlist. It used to be a denylist of
    credentials plus ``TimeoutError``, which made ``WorkspaceSafetyError`` and
    ``CodexExecutionError("Codex timed out after 1800s")`` retryable: a 1800s
    turn ran four times, so a single wedged session could eat two hours.
    """
    chain = _error_chain(error)
    for exc in chain:
        # A provider that is not installed or configured, and a session stopped
        # for touching what it must not, are both decisions -- never weather.
        if isinstance(exc, AgentProviderUnavailableError):
            return False
        if "safety" in type(exc).__name__.lower():
            return False
    for exc in chain:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            # A real timeout type from the transport is transient; the local turn
            # deadline is raised as a plain backend error and stays terminal.
            return type(exc).__name__ in _TRANSIENT_TYPE_NAMES
        if type(exc).__name__ in _TRANSIENT_TYPE_NAMES:
            return True
        if isinstance(exc, ConnectionError):
            return True
        lowered = str(exc).lower()
        if any(marker in lowered for marker in _TRANSIENT_MARKERS):
            return True
    return False


def _delay_for(
    attempt: int,
    *,
    base_sec: float,
    max_sec: float,
    rng: Callable[[], float],
) -> float:
    """Exponential backoff with full jitter, for a 1-based attempt number.

    The gateway degradations this recovers from last minutes, not seconds, so
    the ceiling grows fast; the jitter keeps a whole batch of pods from
    retrying in lockstep and re-degrading the gateway they are waiting on.
    """
    ceiling = min(max_sec, base_sec * (_DELAY_FACTOR ** max(0, attempt - 1)))
    return ceiling * (0.5 + 0.5 * rng())


def _supports_resume(backend: Any) -> bool:
    """Whether this provider can continue an existing session."""
    return bool(getattr(backend.capabilities, "resumable", False) and hasattr(backend, "resume"))


def _merged(
    previous: AgentRunResult,
    resumed: AgentRunResult,
    session_id: str,
) -> AgentRunResult:
    """Fold a resumed turn into the session it continued.

    The resumed turn's text is the session's answer — the previous text is the
    truncated fragment plus the SDK's error line — but the tool calls and
    findings from before the failure are real work and stay attributed.

    Workspace contention carries forward the same way, and for a stronger
    reason: the turn that hit the deadline is exactly the turn that leaves a
    benchmark running, and a clean resume afterwards does not free the device
    that leftover is still holding.
    """
    resumed.tool_calls = [*previous.tool_calls, *resumed.tool_calls]
    resumed.findings = [*previous.findings, *resumed.findings]
    if not resumed.workspace_contention:
        resumed.workspace_contention = previous.workspace_contention
    if not resumed.session_id:
        resumed.session_id = session_id
    if isinstance(previous.num_turns, int) and isinstance(resumed.num_turns, int):
        resumed.num_turns = previous.num_turns + resumed.num_turns
    return resumed


def resumable_session_id(error: BaseException) -> str:
    """The session handle a raising backend managed to establish, if any.

    A transport error after ``thread_start`` succeeded is the expensive case: the
    session already holds every turn it spent reading, building and benchmarking.
    Backends attach the handle to the exception so this layer can continue that
    session instead of opening a new one and paying for all of it again.
    """
    for exc in _error_chain(error):
        session_id = str(getattr(exc, "session_id", "") or "").strip()
        if session_id:
            return session_id
    return ""


def _interrupted_result(error: BaseException, session_id: str) -> AgentRunResult:
    """Present a session that died holding a live handle as a resumable result.

    Returning it (rather than raising) hands it to the resume loop, which already
    knows how to continue a session, count the attempt against the budget, and
    merge the recovered turn back in.
    """
    return AgentRunResult(
        text=f"[session ended with SDK error: {error}]",
        subtype="error",
        num_turns=0,
        end_reason=f"{API_FAILURE_PREFIX}error",
        session_id=session_id,
        stderr_tail=str(error)[:2000],
    )


async def _start(
    backend: Any,
    spec: AgentRunSpec,
    *,
    usage: Any,
    max_retries: int,
    sleep: Callable,
    delay: Callable[[int], float],
) -> AgentRunResult:
    """Open the session, retrying a start that never reached the model.

    A failure with no session handle raises instead of returning a result:
    nothing was established, so there is no context to preserve and a plain
    re-run loses nothing. A failure that DOES carry a handle is returned as a
    resumable result so the caller continues that session.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await backend.run(spec, usage=usage)
        except Exception as exc:  # noqa: BLE001 — classified below
            if not is_retryable_api_error(exc):
                raise
            session_id = resumable_session_id(exc)
            if session_id and _supports_resume(backend):
                log.warning(
                    "agent session %s failed after its handle existed (%s: %s); resuming it instead of starting over",
                    session_id,
                    type(exc).__name__,
                    exc,
                )
                return _interrupted_result(exc, session_id)
            if attempt > max_retries:
                raise
            log.warning(
                "agent session failed before it started (attempt %d/%d): %s: %s",
                attempt,
                max_retries,
                type(exc).__name__,
                exc,
            )
            await sleep(delay(attempt))


async def run_session_with_api_resume(
    backend: Any,
    spec: AgentRunSpec,
    *,
    usage: Any = None,
    max_resumes: int | None = None,
    base_delay_sec: float | None = None,
    max_delay_sec: float | None = None,
    deadline_sec: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> AgentRunResult:
    """Run one agent session, resuming it whenever the API — not the agent — fails.

    Returns the session's result. When the API keeps failing, the last result is
    returned with ``end_reason`` set to :data:`EXHAUSTED_END_REASON` so the
    caller can tell an outage apart from an agent that decided to change
    nothing. Only a failure that precedes the session (nothing to resume, and
    therefore nothing lost) can still raise.

    ``deadline_sec`` bounds the wall clock this whole chain may consume, because
    the resume budget alone does not: every attempt can spend a full turn
    timeout, so an outage that outlives the budget would hold the campaign for
    hours. Passing 0 lifts the bound.
    """
    resume_budget = int(
        max_resumes
        if max_resumes is not None
        else _env_number("FORGE_AGENT_API_MAX_RESUMES", DEFAULT_MAX_RESUMES, cast=int)
    )
    base = float(
        base_delay_sec
        if base_delay_sec is not None
        else _env_number("FORGE_AGENT_API_RETRY_BASE_SEC", DEFAULT_BASE_DELAY_SEC, cast=float)
    )
    ceiling = float(
        max_delay_sec
        if max_delay_sec is not None
        else _env_number("FORGE_AGENT_API_RETRY_MAX_SEC", DEFAULT_MAX_DELAY_SEC, cast=float)
    )

    budget_sec = float(
        deadline_sec
        if deadline_sec is not None
        else _env_number("FORGE_AGENT_API_RETRY_DEADLINE_SEC", DEFAULT_DEADLINE_SEC, cast=float)
    )
    started_at = monotonic()

    def out_of_time() -> bool:
        return budget_sec > 0 and (monotonic() - started_at) >= budget_sec

    def delay(attempt: int) -> float:
        return _delay_for(attempt, base_sec=base, max_sec=ceiling, rng=rng)

    result = await _start(
        backend,
        spec,
        usage=usage,
        max_retries=resume_budget,
        sleep=sleep,
        delay=delay,
    )

    attempt = 0
    while is_api_failure(result) and attempt < resume_budget:
        session_id = str(result.session_id or "").strip()
        if not session_id or not _supports_resume(backend):
            log.error(
                "agent session ended on an API failure with no resumable handle (provider=%s session=%r): %s",
                getattr(backend, "name", "?"),
                session_id,
                result.stderr_tail or result.end_reason,
            )
            break
        if out_of_time():
            log.error(
                "agent session %s still failing after %.0fs of retrying; "
                "stopping so the campaign keeps the rest of its clock",
                session_id,
                monotonic() - started_at,
            )
            break
        attempt += 1
        await sleep(delay(attempt))
        log.warning(
            "resuming session %s after an API failure (attempt %d/%d): %s",
            session_id,
            attempt,
            resume_budget,
            result.stderr_tail or result.end_reason,
        )
        try:
            resumed = await backend.resume(spec, session_id, RESUME_PROMPT, usage=usage)
        except Exception as exc:  # noqa: BLE001 — classified below
            if not is_retryable_api_error(exc):
                raise
            log.warning(
                "resume of session %s failed (%s: %s); will retry while budget remains",
                session_id,
                type(exc).__name__,
                exc,
            )
            # Keep the newest handle: a resume that got as far as re-opening the
            # thread may report a different id, and that is the one to continue.
            result.session_id = resumable_session_id(exc) or session_id
            continue
        result = _merged(result, resumed, session_id)

    if is_api_failure(result):
        log.error(
            "agent session gave up after %d API failure(s); reporting %s so no "
            "caller records this as an agent decision: %s",
            attempt,
            EXHAUSTED_END_REASON,
            result.stderr_tail or result.end_reason,
        )
        result.end_reason = EXHAUSTED_END_REASON
    return result


__all__ = [
    "DEFAULT_DEADLINE_SEC",
    "EXHAUSTED_END_REASON",
    "RESUME_PROMPT",
    "TERMINAL_END_REASONS",
    "is_api_failure",
    "is_retryable_api_error",
    "resumable_session_id",
    "run_session_with_api_resume",
]
