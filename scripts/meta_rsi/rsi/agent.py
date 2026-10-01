# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claude Code runs for the rsi judgment steps, through the Claude Agent SDK.

One call is one ``claude_agent_sdk.query`` session with a prompt, a tool allow-list, a working
directory, a turn cap and a dollar cap. A PreToolUse hook refuses pushes, package installs and
file edits outside the step's write root (shell commands can still write; the root is a guard,
not a sandbox). Each call appends one row in the ``llm_calls.jsonl`` field layout to the round's
``agent_calls.jsonl``, so the round's own spend reads like a session's.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WRITE_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
READ_TOOLS = ("Read", "Glob", "Grep")
DENY_BASH = re.compile(
    r"\bgit\s+push\b|\bpip3?\s+install\b|\buv\s+pip\b|\bsudo\b|\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b|\brm\s+-rf\s+/(\s|$)"
)
JSON_FENCE = re.compile(r"```json\s*\n(.*?)\n```", re.S)

QueryFn = Callable[..., AsyncIterator[Any]]


@dataclass(frozen=True)
class AgentSpec:
    """One agent session. ``write_root`` is the only tree its edit tools may touch (None: none)."""

    name: str
    prompt: str
    system_prompt: str
    cwd: Path
    tools: tuple[str, ...]
    model: str
    budget_usd: float
    max_turns: int
    write_root: Path | None = None
    add_dirs: tuple[Path, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    transcript: Path | None = None


@dataclass(frozen=True)
class AgentResult:
    text: str
    is_error: bool
    turns: int
    cost_usd: float
    usage: dict
    error: str = ""
    session_id: str = ""


def role_of(name: str) -> str:
    """The step a session belongs to: ``implement:trim-prompt#2`` -> ``implement``."""
    return re.split(r"[:#]", name, maxsplit=1)[0]


def guard_reason(tool: str, tool_input: dict, write_root: Path | None) -> str:
    """Why a tool call is refused, or an empty string when it may run."""
    if tool == "Bash":
        command = str(tool_input.get("command") or "")
        return (
            "this step may not push, install packages, use sudo or pipe downloads into a shell"
            if DENY_BASH.search(command)
            else ""
        )
    if tool not in WRITE_TOOLS:
        return ""
    if write_root is None:
        return "this step is read-only"
    target = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    resolved = Path(str(target)).expanduser().resolve()
    root = write_root.resolve()
    if resolved != root and root not in resolved.parents:
        return f"edits are limited to {root}"
    return ""


def pre_tool_use_hook(write_root: Path | None) -> Callable[..., Any]:
    """SDK PreToolUse callback applying ``guard_reason``."""

    async def hook(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        reason = guard_reason(str(input_data.get("tool_name") or ""), input_data.get("tool_input") or {}, write_root)
        if not reason:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return hook


def build_options(spec: AgentSpec) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from hyperloom.common.llm_config import claude_sdk_env_options

    env_options = claude_sdk_env_options(
        model=spec.model, env=spec.env or None, component="meta_rsi", operation=role_of(spec.name)
    )
    return ClaudeAgentOptions(
        **env_options,
        model=spec.model,
        cwd=str(spec.cwd),
        system_prompt=spec.system_prompt,
        tools=list(spec.tools),
        allowed_tools=list(spec.tools),
        max_turns=spec.max_turns,
        max_budget_usd=spec.budget_usd,
        add_dirs=[str(d) for d in spec.add_dirs],
        hooks={"PreToolUse": [HookMatcher(hooks=[pre_tool_use_hook(spec.write_root)])]},
    )


def _jsonable(message: Any) -> dict:
    body = dataclasses.asdict(message) if dataclasses.is_dataclass(message) else {"repr": repr(message)}
    return {"type": type(message).__name__, **body}


def _result(final: Any, texts: list[str], error: str) -> AgentResult:
    if final is None:
        return AgentResult(
            text="\n".join(texts), is_error=True, turns=0, cost_usd=0.0, usage={}, error=error or "no result"
        )
    return AgentResult(
        text=str(getattr(final, "result", None) or "\n".join(texts)),
        is_error=bool(getattr(final, "is_error", False)),
        turns=int(getattr(final, "num_turns", 0) or 0),
        cost_usd=float(getattr(final, "total_cost_usd", 0.0) or 0.0),
        usage=dict(getattr(final, "usage", None) or {}),
        error=error or (str(getattr(final, "subtype", "")) if getattr(final, "is_error", False) else ""),
        session_id=str(getattr(final, "session_id", "") or ""),
    )


async def _drive(spec: AgentSpec, query_fn: QueryFn) -> AgentResult:
    from claude_agent_sdk import ClaudeSDKError

    final, texts, error = None, [], ""
    with open(spec.transcript, "a") if spec.transcript else contextlib.nullcontext() as transcript:
        try:
            async for message in query_fn(prompt=spec.prompt, options=build_options(spec)):
                if transcript:
                    transcript.write(json.dumps(_jsonable(message), default=str) + "\n")
                if hasattr(message, "total_cost_usd"):
                    final = message
                for block in getattr(message, "content", None) or []:
                    if isinstance(getattr(block, "text", None), str):
                        texts.append(block.text)
        except ClaudeSDKError as exc:
            error = f"{type(exc).__name__}: {exc}"
    return _result(final, texts, error)


def ledger_row(spec: AgentSpec, result: AgentResult, latency_ms: int) -> dict:
    """One ``llm_calls.jsonl``-shaped row for the call."""
    u = result.usage
    return {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "component": "meta_rsi",
        "role": role_of(spec.name),
        "task_id": spec.name,
        "call_id": result.session_id,
        "model": spec.model,
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "latency_ms": latency_ms,
        "status": "error" if result.is_error else "ok",
        "error_message": result.error,
        "cost_usd": result.cost_usd,
        "turns": result.turns,
    }


def run_agent(spec: AgentSpec, ledger: Path, query_fn: QueryFn | None = None) -> AgentResult:
    """Run one agent session to completion and append its ledger row."""
    if query_fn is None:
        from claude_agent_sdk import query as query_fn
    started = time.monotonic()
    result = asyncio.run(_drive(spec, query_fn))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as fh:
        fh.write(json.dumps(ledger_row(spec, result, int(1000 * (time.monotonic() - started)))) + "\n")
    return result


def extract_json(text: str) -> Any:
    """The JSON value an agent ended its reply with: the last ```json fence, else the last parsable object."""
    fences = JSON_FENCE.findall(text)
    if fences:
        return json.loads(fences[-1])
    decoder, found, i = json.JSONDecoder(), [], text.find("{")
    while i >= 0:
        try:
            value, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        found.append(value)
        i = text.find("{", end)
    if not found:
        raise ValueError("the reply holds no JSON object")
    return found[-1]
