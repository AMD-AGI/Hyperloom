# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single-shot, tool-free Claude completions through the Claude Agent SDK.

The Anthropic Messages HTTP path in :mod:`hyperloom.common.llm_config`
authenticates with ``x-api-key``, a channel that rejects a Claude Max/Pro
subscription token outright. Driving single-shot Anthropic inference through the
SDK hands credential resolution back to the Claude CLI, which reads every
supported form -- API key, gateway bearer token, and ``CLAUDE_CODE_OAUTH_TOKEN``
-- so one transport serves all of them.

Callers keep the :class:`~hyperloom.common.llm_config.AnthropicMessageResult`
shape they already consume, so token accounting and stop-reason handling are
unchanged from the HTTP path.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .llm_config import AnthropicMessageResult, claude_sdk_env_options

__all__ = [
    "DISALLOWED_TOOLS",
    "ClaudeOneShotClient",
    "ensure_available",
    "message_text",
]

# Claude Code must behave as a plain completion client here. Denying every
# built-in tool keeps a single-shot request from reading beyond its prompt.
DISALLOWED_TOOLS: tuple[str, ...] = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "Agent",
    "Task",
    "TaskOutput",
    "TaskStop",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "Skill",
    "SlashCommand",
)

# ClaudeAgentOptions has no max_tokens field; the CLI reads the cap from its own
# environment instead, which is the only way to keep an output budget here.
OUTPUT_TOKEN_CAP_ENV = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"

_DEFAULT_TIMEOUT_SEC = 60.0


def _load_sdk() -> Any:
    """Import ``claude_agent_sdk`` and check the attributes this module uses.

    Returns:
        The imported ``claude_agent_sdk`` module.

    Raises:
        RuntimeError: If the package is missing or lacks ``query`` /
            ``ClaudeAgentOptions``.
    """
    try:
        import claude_agent_sdk as sdk  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("claude_agent_sdk is not installed") from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise RuntimeError("claude_agent_sdk missing query / ClaudeAgentOptions")
    return sdk


def ensure_available() -> None:
    """Fail now if the SDK is missing, so callers can degrade at build time.

    Raises:
        RuntimeError: If ``claude_agent_sdk`` cannot be used.
    """
    _load_sdk()


def message_text(message: Any) -> list[str]:
    """Extract text fragments from one Claude SDK message.

    Args:
        message: A message yielded by ``claude_agent_sdk.query``.

    Returns:
        Every text fragment the message carries, in order; empty when it
        carries none.
    """
    if isinstance(message, str):
        return [message]
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return [text]
    parts: list[str] = []
    content = getattr(message, "content", None)
    if isinstance(content, list):
        for block in content:
            block_text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(block_text, str):
                parts.append(block_text)
    return parts


def _prompt_from_messages(messages: Sequence[Mapping[str, Any]] | None) -> str:
    """Flatten Anthropic-style turns into the single prompt the SDK accepts.

    Args:
        messages: The Messages-API ``messages`` array.

    Returns:
        The concatenated turn content.

    Raises:
        ValueError: If no turn carries content, which would send an empty
            prompt and bill a call for nothing.
    """
    parts: list[str] = []
    for turn in messages or ():
        content = turn.get("content") if isinstance(turn, Mapping) else None
        if isinstance(content, str) and content.strip():
            parts.append(content)
            continue
        if isinstance(content, Iterable) and not isinstance(content, (str, bytes)):
            for block in content:
                block_text = block.get("text") if isinstance(block, Mapping) else getattr(block, "text", None)
                if isinstance(block_text, str) and block_text.strip():
                    parts.append(block_text)
    prompt = "\n\n".join(parts).strip()
    if not prompt:
        raise ValueError("claude one-shot completion requires at least one non-empty message")
    return prompt


def _build_options(
    sdk: Any,
    *,
    model: str,
    system: str | None,
    max_tokens: int | None,
) -> Any:
    """Assemble the tool-free, single-turn options for one completion."""
    kwargs: dict[str, Any] = dict(claude_sdk_env_options(model=model))
    if max_tokens:
        # claude_sdk_env_options returns {} when no provider signal is set; fall
        # back to the ambient environment so the cap is the only addition.
        env = dict(kwargs.get("env") or os.environ)
        env[OUTPUT_TOKEN_CAP_ENV] = str(int(max_tokens))
        kwargs["env"] = env
    kwargs.update(
        {
            "model": model or None,
            "system_prompt": system or None,
            "tools": [],
            "setting_sources": [],
            "skills": [],
            "strict_mcp_config": True,
            "mcp_servers": {},
            "plugins": [],
            "max_turns": 1,
            "allowed_tools": [],
            "disallowed_tools": list(DISALLOWED_TOOLS),
        }
    )
    return sdk.ClaudeAgentOptions(**kwargs)


async def _drive(sdk: Any, prompt: str, options: Any) -> AnthropicMessageResult:
    """Consume one ``query`` stream and flatten it onto the shared result type."""
    final = ""
    chunks: list[str] = []
    usage: Any = None
    stop_reason: str | None = None
    async for message in sdk.query(prompt=prompt, options=options):
        message_usage = getattr(message, "usage", None)
        if isinstance(message_usage, Mapping):
            usage = dict(message_usage)
        reason = getattr(message, "stop_reason", None)
        if isinstance(reason, str) and reason:
            stop_reason = reason
        result = getattr(message, "result", None)
        if isinstance(result, str) and result.strip():
            final = result
            continue
        chunks.extend(message_text(message))
    return AnthropicMessageResult(
        text=final.strip() or "".join(chunks).strip(),
        stop_reason=stop_reason,
        usage=usage,
    )


@dataclass
class ClaudeOneShotClient:
    """Tool-free Claude completion client shaped like the Anthropic HTTP one.

    Attributes:
        timeout_s: Wall-clock budget for one completion, CLI startup included.
    """

    timeout_s: float = _DEFAULT_TIMEOUT_SEC

    async def amessages(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> AnthropicMessageResult:
        """Run one completion and return its text, stop reason and usage.

        Args:
            model: The Claude model id.
            messages: Anthropic-style turns; flattened into one prompt.
            system: The system instruction, or ``None``.
            max_tokens: Output-token cap, applied through the CLI environment.

        Returns:
            The flattened :class:`AnthropicMessageResult`.

        Raises:
            RuntimeError: If the SDK is unavailable.
            asyncio.TimeoutError: If the completion outruns :attr:`timeout_s`.
        """
        sdk = _load_sdk()
        options = _build_options(sdk, model=model, system=system, max_tokens=max_tokens)
        return await asyncio.wait_for(
            _drive(sdk, _prompt_from_messages(messages), options),
            timeout=max(0.1, float(self.timeout_s)),
        )

    def messages(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> AnthropicMessageResult:
        """Synchronous twin of :meth:`amessages`; see it for the full contract.

        Raises:
            RuntimeError: If called from inside a running event loop, where the
                caller must await :meth:`amessages` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.amessages(model=model, messages=messages, system=system, max_tokens=max_tokens)
            )
        raise RuntimeError("ClaudeOneShotClient.messages cannot run inside an active event loop")

    async def aclose(self) -> None:
        """Release transport state; the SDK owns none, so this is a no-op."""
        return None
