# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive one multi-turn agent session on Claude Code or the DeepSeek Harness."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from hyperloom.common import llm_config
from hyperloom.common.claude_oneshot import message_text
from hyperloom.common.deepseek_session import DeepSeekSession, DeepSeekSessionError
from hyperloom.common.reasoning_effort import DEFAULT_REASONING_EFFORT, harness_reasoning_effort

_RUNTIME_MAPPING = """\
## DeepSeek Harness runtime mapping
Use the read, glob and grep tools to inspect and search files, and bash to run
commands. References below to Read, Edit, Write, Grep, Glob, or Bash describe
equivalent capabilities, not literal tool names.
"""

_DEEPSEEK_LOG_PREFIX = "[deepseek-harness]"
_CLAUDE_LOG_PREFIX = "[claude-sdk]"


@dataclass
class AgentSdkTurnResult:
    """Normalized text output of one agent harness session."""

    text: str = ""
    error: str = ""
    chunks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentSessionRequest:
    """What one harness session is asked to do, and where."""

    prompt: str
    system_prompt: str
    cwd: Path
    model: str | None = None
    max_turns: int = 12
    allowed_tools: list[str] | None = None
    child_env: Mapping[str, str] | None = None
    log: Callable[[str], None] | None = None
    backend: str | None = None
    timeout_sec: float | None = None
    """Wall-clock budget of the whole DeepSeek session, continuations included."""


@dataclass(frozen=True)
class OwedArtifact:
    """The artifact a DeepSeek session owes, and how it is asked for again.

    The DeepSeek Harness ends a turn whenever the model replies without a tool
    call -- including when it writes one out as text -- so a session that
    stopped before ``done()`` holds is continued with ``continuation_prompt``,
    at most ``max_continuations`` times. Claude's own loop runs to its turn cap
    and takes no continuation.
    """

    done: Callable[[], bool]
    continuation_prompt: str
    max_continuations: int = 3


def _import_claude_sdk() -> tuple[Any, Any]:
    try:
        import claude_agent_sdk as sdk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("claude_agent_sdk not installed") from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")
    return sdk.query, sdk.ClaudeAgentOptions


def _permission_mode(writable: bool) -> str:
    return "danger-full-access" if writable else "read-only"


def _note(log: Callable[[str], None] | None, prefix: str, text: str) -> None:
    if log:
        log(f"{prefix} {text}")


async def drive_agent_session(request: AgentSessionRequest, owed: OwedArtifact | None = None) -> AgentSdkTurnResult:
    """Run one harness session and return the accumulated assistant text."""
    resolved = (request.backend or llm_config.resolve_agent_backend()).strip().lower()
    if resolved == llm_config.AGENT_BACKEND_DEEPSEEK:
        return await _drive_deepseek(request, owed)
    return await _drive_claude(request)


def _claude_options(request: AgentSessionRequest, options_cls: Any) -> Any:
    log = request.log
    kwargs: dict[str, Any] = {
        "max_turns": request.max_turns,
        "system_prompt": request.system_prompt,
        "allowed_tools": request.allowed_tools or [],
        "stderr": (lambda line: log(f"{_CLAUDE_LOG_PREFIX} {line.rstrip()}")) if log else None,
    }
    if request.model:
        kwargs["model"] = request.model
    if request.child_env is not None:
        kwargs["env"] = dict(request.child_env)
    kwargs["cwd"] = str(request.cwd)
    try:
        return options_cls(**kwargs)
    except TypeError:
        kwargs.pop("cwd", None)
        return options_cls(**kwargs)


async def _drive_claude(request: AgentSessionRequest) -> AgentSdkTurnResult:
    query, options_cls = _import_claude_sdk()
    options = _claude_options(request, options_cls)
    chunks: list[str] = []
    error = ""
    try:
        async for message in query(prompt=request.prompt, options=options):
            for text in _iter_message_text(message):
                chunks.append(text)
                _note(request.log, _CLAUDE_LOG_PREFIX, text[:1000])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        _note(request.log, _CLAUDE_LOG_PREFIX, f"WARNING: {error}")
    return AgentSdkTurnResult(text="\n".join(chunks), error=error, chunks=chunks)


def _deepseek_session(request: AgentSessionRequest) -> DeepSeekSession:
    instructions = "\n\n".join(part for part in (_RUNTIME_MAPPING.strip(), request.system_prompt.strip()) if part)
    return DeepSeekSession(
        cwd=request.cwd,
        model=request.model or llm_config.resolve_forge_llm_model(llm_config.AGENT_BACKEND_DEEPSEEK),
        instructions=instructions,
        reasoning_effort=harness_reasoning_effort(DEFAULT_REASONING_EFFORT),
        permission_mode=_permission_mode(bool(request.allowed_tools)),
        env=dict(request.child_env) if request.child_env is not None else None,
        request_timeout_sec=7200.0,
    )


def _turn_timeout(session: DeepSeekSession, deadline: float | None) -> float | None:
    """The next turn's timeout, or ``None`` once the session budget is spent."""
    if deadline is None:
        return session.request_timeout_sec
    remaining = deadline - monotonic()
    return min(session.request_timeout_sec, remaining) if remaining > 0 else None


async def _run_until_owed(
    session: DeepSeekSession,
    request: AgentSessionRequest,
    owed: OwedArtifact | None,
    chunks: list[str],
) -> str:
    """Run turns until the owed artifact exists; return the error that stopped them, if any."""
    deadline = monotonic() + request.timeout_sec if request.timeout_sec else None
    turn_prompt = request.prompt
    for continuation in range((owed.max_continuations if owed else 0) + 1):
        if continuation:
            _note(
                request.log,
                _DEEPSEEK_LOG_PREFIX,
                f"continuing: the owed artifact is not written (continuation {continuation})",
            )
        turn_timeout = _turn_timeout(session, deadline)
        if turn_timeout is None:
            return f"the {request.timeout_sec:.0f}s session budget ran out before the owed artifact was written"
        result = await session.turn(turn_prompt, timeout_sec=turn_timeout)
        if result.text:
            chunks.append(result.text)
            _note(request.log, _DEEPSEEK_LOG_PREFIX, result.text[:1000])
        if result.error or owed is None or owed.done():
            return result.error
        turn_prompt = owed.continuation_prompt
    return ""


async def _drive_deepseek(request: AgentSessionRequest, owed: OwedArtifact | None) -> AgentSdkTurnResult:
    session = _deepseek_session(request)
    chunks: list[str] = []
    try:
        await session.start()
        error = await _run_until_owed(session, request, owed, chunks)
    except DeepSeekSessionError as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await session.aclose()
    if error:
        _note(request.log, _DEEPSEEK_LOG_PREFIX, f"WARNING: {error}")
    return AgentSdkTurnResult(text="\n".join(chunks), error=error, chunks=chunks)


def _iter_message_text(message: Any) -> Iterable[str]:
    yield from (fragment for fragment in message_text(message) if fragment)


__all__ = [
    "AgentSdkTurnResult",
    "AgentSessionRequest",
    "OwedArtifact",
    "drive_agent_session",
]
