# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive one multi-turn agent session on Claude Code or the DeepSeek Harness."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class AgentSdkTurnResult:
    """Normalized text output of one agent harness session."""

    text: str = ""
    error: str = ""
    chunks: list[str] = field(default_factory=list)


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


async def drive_agent_session(
    *,
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str | None = None,
    max_turns: int = 12,
    allowed_tools: list[str] | None = None,
    child_env: Mapping[str, str] | None = None,
    log: Callable[[str], None] | None = None,
    backend: str | None = None,
    done: Callable[[], bool] | None = None,
    continuation_prompt: str = "",
    max_continuations: int = 3,
) -> AgentSdkTurnResult:
    """Run one harness session and return the accumulated assistant text.

    ``done`` names the artifact the session owes. The DeepSeek Harness ends a
    turn whenever the model replies without a tool call -- including when it
    writes one out as text -- so a session that stopped before ``done()`` holds
    is continued with ``continuation_prompt``, at most ``max_continuations``
    times. Claude's own loop runs to its turn cap and takes no continuation.
    """
    resolved = (backend or llm_config.resolve_agent_backend()).strip().lower()
    if resolved == llm_config.AGENT_BACKEND_DEEPSEEK:
        return await _drive_deepseek(
            prompt=prompt,
            system_prompt=system_prompt,
            cwd=cwd,
            model=model,
            allowed_tools=allowed_tools,
            child_env=child_env,
            log=log,
            done=done,
            continuation_prompt=continuation_prompt,
            max_continuations=max_continuations,
        )
    return await _drive_claude(
        prompt=prompt,
        system_prompt=system_prompt,
        cwd=cwd,
        model=model,
        max_turns=max_turns,
        allowed_tools=allowed_tools,
        child_env=child_env,
        log=log,
    )


async def _drive_claude(
    *,
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str | None,
    max_turns: int,
    allowed_tools: list[str] | None,
    child_env: Mapping[str, str] | None,
    log: Callable[[str], None] | None,
) -> AgentSdkTurnResult:
    query, options_cls = _import_claude_sdk()
    kwargs: dict[str, Any] = {
        "max_turns": max_turns,
        "system_prompt": system_prompt,
        "allowed_tools": allowed_tools or [],
        "stderr": (lambda line: log(f"[claude-sdk] {line.rstrip()}")) if log else None,
    }
    if model:
        kwargs["model"] = model
    if child_env is not None:
        kwargs["env"] = dict(child_env)
    kwargs["cwd"] = str(cwd)
    try:
        options = options_cls(**kwargs)
    except TypeError:
        kwargs.pop("cwd", None)
        options = options_cls(**kwargs)

    chunks: list[str] = []
    error = ""
    try:
        async for message in query(prompt=prompt, options=options):
            for text in _iter_message_text(message):
                chunks.append(text)
                if log:
                    log(f"[claude-sdk] {text[:1000]}")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[claude-sdk] WARNING: {error}")
    return AgentSdkTurnResult(text="\n".join(chunks), error=error, chunks=chunks)


async def _drive_deepseek(
    *,
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str | None,
    allowed_tools: list[str] | None,
    child_env: Mapping[str, str] | None,
    log: Callable[[str], None] | None,
    done: Callable[[], bool] | None = None,
    continuation_prompt: str = "",
    max_continuations: int = 0,
) -> AgentSdkTurnResult:
    writable = bool(allowed_tools)
    instructions = "\n\n".join(part for part in (_RUNTIME_MAPPING.strip(), system_prompt.strip()) if part)
    session = DeepSeekSession(
        cwd=cwd,
        model=model or llm_config.resolve_forge_llm_model(llm_config.AGENT_BACKEND_DEEPSEEK),
        instructions=instructions,
        reasoning_effort=harness_reasoning_effort(DEFAULT_REASONING_EFFORT),
        permission_mode=_permission_mode(writable),
        env=dict(child_env) if child_env is not None else None,
        request_timeout_sec=7200.0,
    )
    chunks: list[str] = []
    error = ""
    try:
        await session.start()
        turn_prompt = prompt
        for continuation in range(max_continuations + 1):
            if continuation and log:
                log(f"[deepseek-harness] continuing: the owed artifact is not written (continuation {continuation})")
            result = await session.turn(turn_prompt, timeout_sec=session.request_timeout_sec)
            text = result.text or ""
            if text:
                chunks.append(text)
                if log:
                    log(f"[deepseek-harness] {text[:1000]}")
            if result.error:
                error = result.error
                if log:
                    log(f"[deepseek-harness] WARNING: {error}")
                break
            if done is None or done():
                break
            turn_prompt = continuation_prompt
    except DeepSeekSessionError as exc:
        error = f"{type(exc).__name__}: {exc}"
        if log:
            log(f"[deepseek-harness] WARNING: {error}")
    finally:
        await session.aclose()
    return AgentSdkTurnResult(text="\n".join(chunks), error=error, chunks=chunks)


def _iter_message_text(message: Any) -> Iterable[str]:
    yield from (fragment for fragment in message_text(message) if fragment)


def run_agent_session_sync(**kwargs: Any) -> AgentSdkTurnResult:
    """Synchronous wrapper around :func:`drive_agent_session`."""
    return asyncio.run(drive_agent_session(**kwargs))


__all__ = ["AgentSdkTurnResult", "drive_agent_session", "run_agent_session_sync"]
