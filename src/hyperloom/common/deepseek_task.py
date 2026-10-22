# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run one task to completion on a DeepSeek Harness session, continuing it until its owed artifact exists."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from hyperloom.common.deepseek_session import DEEPSEEK_RUNTIME_MAPPING, DeepSeekSession, DeepSeekSessionError
from hyperloom.common.reasoning_effort import DEFAULT_REASONING_EFFORT, harness_reasoning_effort

_LOG_PREFIX = "[deepseek-harness]"

#: Wall-clock limit of one turn when the task sets no budget of its own.
TURN_TIMEOUT_SEC = 7200.0

#: How many times a session that stopped short of its owed artifact is asked to continue.
DEFAULT_MAX_CONTINUATIONS = 3


@dataclass
class DeepSeekTaskResult:
    """Text and error of one harness task."""

    text: str = ""
    error: str = ""


@dataclass(frozen=True)
class DeepSeekTaskRequest:
    """What one harness task is asked to do, and where."""

    prompt: str
    system_prompt: str
    cwd: Path
    writable: bool = False
    child_env: Mapping[str, str] | None = None
    log: Callable[[str], None] | None = None
    timeout_sec: float | None = None
    """Wall-clock budget of the whole task, continuations included."""


@dataclass(frozen=True)
class OwedArtifact:
    """The artifact a task owes, and how it is asked for again.

    The DeepSeek Harness ends a turn whenever the model replies without a tool
    call -- including when it writes one out as text -- so a session that
    stopped before ``done()`` holds is continued with ``continuation_prompt``,
    at most ``max_continuations`` times.
    """

    done: Callable[[], bool]
    continuation_prompt: str
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS


def _note(log: Callable[[str], None] | None, text: str) -> None:
    if log:
        log(f"{_LOG_PREFIX} {text}")


def _session_for(request: DeepSeekTaskRequest) -> DeepSeekSession:
    instructions = "\n\n".join(
        part for part in (DEEPSEEK_RUNTIME_MAPPING.strip(), request.system_prompt.strip()) if part
    )
    return DeepSeekSession(
        cwd=request.cwd,
        instructions=instructions,
        reasoning_effort=harness_reasoning_effort(DEFAULT_REASONING_EFFORT),
        permission_mode="danger-full-access" if request.writable else "read-only",
        env=dict(request.child_env) if request.child_env is not None else None,
        request_timeout_sec=TURN_TIMEOUT_SEC,
    )


def _turn_timeout(session: DeepSeekSession, deadline: float | None) -> float | None:
    """The next turn's timeout, or ``None`` once the task budget is spent."""
    if deadline is None:
        return session.request_timeout_sec
    remaining = deadline - monotonic()
    return min(session.request_timeout_sec, remaining) if remaining > 0 else None


async def _run_until_owed(
    session: DeepSeekSession,
    request: DeepSeekTaskRequest,
    owed: OwedArtifact | None,
    chunks: list[str],
) -> str:
    """Run turns until the owed artifact exists; return the error that stopped them, if any."""
    deadline = monotonic() + request.timeout_sec if request.timeout_sec else None
    turn_prompt = request.prompt
    continuations = owed.max_continuations if owed else 0
    for continuation in range(continuations + 1):
        if continuation:
            _note(request.log, f"continuing: the owed artifact is not written (continuation {continuation})")
        turn_timeout = _turn_timeout(session, deadline)
        if turn_timeout is None:
            return f"the {request.timeout_sec:.0f}s task budget ran out before the owed artifact was written"
        result = await session.turn(turn_prompt, timeout_sec=turn_timeout)
        if result.text:
            chunks.append(result.text)
            _note(request.log, result.text[:1000])
        if result.error or owed is None or owed.done():
            return result.error
        turn_prompt = owed.continuation_prompt
    return f"the owed artifact was still not written after {continuations} continuations"


async def run_deepseek_task(request: DeepSeekTaskRequest, owed: OwedArtifact | None = None) -> DeepSeekTaskResult:
    """Run the task on a fresh harness session and return the accumulated assistant text."""
    session = _session_for(request)
    chunks: list[str] = []
    try:
        await session.start()
        error = await _run_until_owed(session, request, owed, chunks)
    except DeepSeekSessionError as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await session.aclose()
    if error:
        _note(request.log, f"WARNING: {error}")
    return DeepSeekTaskResult(text="\n".join(chunks), error=error)


__all__ = [
    "DEFAULT_MAX_CONTINUATIONS",
    "TURN_TIMEOUT_SEC",
    "DeepSeekTaskRequest",
    "DeepSeekTaskResult",
    "OwedArtifact",
    "run_deepseek_task",
]
