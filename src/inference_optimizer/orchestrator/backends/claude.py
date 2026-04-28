"""ClaudeBackend — uses ``claude-agent-sdk`` to drive Claude (DESIGN §6.1, §10.5.4).

v0.6 (F4) implementation strategy
=================================

We register the real ``emit_intent`` MCP tool with the Claude SDK whenever
the SDK supports it (any modern ``claude-agent-sdk``), so Claude calls
``emit_intent`` directly via ``ToolUseBlock`` and the trajectory parser
collects the structured input. JSON-in-text remains a *fallback*:

1. ``__post_init__`` builds an in-process MCP server (DESIGN §10.5.4)
   exposing ``emit_intent`` and records its qualified tool name in
   :attr:`mcp_tool_name` (e.g. ``"mcp__inference_optimizer__emit_intent"``).
2. ``_build_options`` injects ``mcp_servers={...}`` and ensures the
   qualified tool name appears in ``allowed_tools``.
3. The prompt suffix prefers the tool path (``call emit_intent(...)``) and
   keeps the JSON envelope shape as a fallback for runtimes where the SDK
   couldn't register custom tools.
4. ``parse_claude_trajectory(trajectory, fallback_text=...)`` picks up the
   tool_use blocks first; only when none are present it falls back to
   parsing fenced JSON in the assistant's text reply.

Lazy SDK import — if ``claude-agent-sdk`` isn't installed, we raise a
clear ``BackendError`` at construction time so the CLI can surface the
exact pip command. If the SDK is installed but lacks
``create_sdk_mcp_server``, we silently degrade to JSON-in-text mode.

Auto-bootstrap of Node + ``claude`` CLI lives in
:mod:`inference_optimizer.bootstrap`; ClaudeBackend itself does **not**
trigger installs — that's the CLI's job at startup, before this class
is constructed.
"""
from __future__ import annotations

import asyncio
import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..intent_parser import (
    EMIT_INTENT_TOOL_SCHEMA,
    Intent,
    IntentValidationError,
    NoIntentEmitted,
    parse_claude_trajectory,
)
from .base import Backend, BackendError
from .mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    MCP_SERVER_NAME,
    build_emit_intent_server,
)


# Two prompt suffixes — when the MCP ``emit_intent`` tool is registered we
# prefer the tool path; when the SDK lacks custom-tool support (or the user
# disables it) we fall back to fenced-JSON output.
_OUTPUT_INSTRUCTIONS_TOOL = f"""
==== OUTPUT FORMAT (REQUIRED) ====
Use the ``{EMIT_INTENT_TOOL_NAME}`` tool to communicate. Each call carries
exactly one intent; call multiple times to emit multiple intents in the
same turn. Free-text replies are ignored — they are only displayed for
debugging.

Tool input shape:

  {{
    "intent_type": "<one of send_message|delegate|propose_action|"
                  "objection|vote|update_state|update_persona|"
                  "answer|ask_question|alert>",
    "payload": {{ /* per-intent fields */ }}
  }}

If you have nothing to say, call the tool once with intent_type=send_message
and payload={{"topic":"heartbeat","body_md":"no-op tick"}}.
==== END OUTPUT FORMAT ====
""".strip()

_OUTPUT_INSTRUCTIONS_TEXT = """
==== OUTPUT FORMAT (REQUIRED, fallback) ====
Reply with a single JSON object that matches this schema:

{
  "intents": [
    { "intent_type": "<one of send_message|delegate|propose_action|"
                     "objection|vote|update_state|update_persona|"
                     "answer|ask_question|alert>",
      "payload": { /* per-intent fields */ } }
  ]
}

Rules:
- Free text outside the JSON is ignored.
- Wrap the JSON in a ```json fenced block.
- Always emit at least one intent. If you have nothing to say, emit
  {"intent_type": "send_message", "payload": {"topic": "heartbeat",
  "body_md": "no-op tick"}}.
==== END OUTPUT FORMAT ====
""".strip()


def _import_sdk() -> tuple[Any, Any, Callable[..., Any], Any]:
    """Return ``(query, ClaudeAgentOptions, text_extractor, sdk_module)``.

    Tries ``claude_agent_sdk`` first (current name), falls back to
    ``claude_code_sdk`` (legacy). The returned ``sdk_module`` is forwarded
    to :func:`build_emit_intent_server` so the MCP setup can use the same
    in-process server helpers.
    """
    last_err: Exception | None = None
    for mod_name in ("claude_agent_sdk", "claude_code_sdk"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            last_err = exc
            continue
        # Both packages expose ``query`` and an options dataclass.
        opts_cls = getattr(mod, "ClaudeAgentOptions", None) or getattr(
            mod, "ClaudeCodeOptions", None
        )
        if not (hasattr(mod, "query") and opts_cls is not None):
            last_err = RuntimeError(
                f"{mod_name} loaded but missing query/options"
            )
            continue
        return mod.query, opts_cls, _extract_text_for(mod), mod
    raise BackendError(
        "claude-agent-sdk not installed; run `pip install claude-agent-sdk` "
        "(or `claude-code-sdk` for legacy)."
    ) from last_err


def _extract_text_for(mod: Any) -> Callable[[Any], str]:
    """Return a function that pulls plain text out of an SDK message.

    The SDK packages expose a class hierarchy roughly like ``Message ->
    AssistantMessage -> content[List[TextBlock|ToolUseBlock|...]]`` and
    ``ResultMessage`` with a ``.result`` string. We accept all flavours.
    """
    text_block_cls = getattr(mod, "TextBlock", None)
    result_message_cls = getattr(mod, "ResultMessage", None)

    def extract(message: Any) -> str:
        if result_message_cls is not None and isinstance(message, result_message_cls):
            return getattr(message, "result", "") or ""
        chunks: list[str] = []
        for block in getattr(message, "content", []) or []:
            if text_block_cls is not None and isinstance(block, text_block_cls):
                chunks.append(getattr(block, "text", "") or "")
            elif isinstance(block, dict) and block.get("type") == "text":
                chunks.append(block.get("text", ""))
            else:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    chunks.append(t)
        return "".join(chunks)

    return extract


# ---------------------------------------------------------------------------
@dataclass
class ClaudeBackend(Backend):
    """Production backend backed by ``claude-agent-sdk``.

    Args:
        model: Claude model id, e.g. ``"claude-opus-4-7"``. Defaults to
            ``ANTHROPIC_MODEL`` env or library default.
        api_key_env: Env var that must hold the API key. The SDK reads
            ``ANTHROPIC_API_KEY`` itself; we just check up-front.
        allowed_tools_default: Tool list to forward to the SDK when the
            caller doesn't override.
        max_turns_default: Default agent-loop budget.
        repair_attempts: When parsing fails, how many extra calls we'll
            send with a "repair" prompt (DESIGN §10.5.5). 0 = no repair.
        sdk_query_factory: Test seam — replace the ``query`` import.
        sdk_options_cls: Test seam — replace ``ClaudeAgentOptions``.
    """

    model: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    allowed_tools_default: tuple[str, ...] = ("Read",)
    max_turns_default: int = 10
    repair_attempts: int = 1
    enable_mcp_emit_intent: bool = True

    sdk_query_factory: Callable[..., Any] | None = None
    sdk_options_cls: Any | None = None
    sdk_extract_text: Callable[[Any], str] | None = None
    sdk_module: Any | None = None
    mcp_server_factory: Callable[..., Any] | None = None
    mcp_tool_factory: Callable[..., Any] | None = None

    calls: list[dict[str, Any]] = field(default_factory=list)
    mcp_server_config: Any | None = field(default=None, init=False)
    mcp_tool_name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        sdk_module = self.sdk_module
        if (
            self.sdk_query_factory is None
            or self.sdk_options_cls is None
            or sdk_module is None
        ):
            try:
                query, opts_cls, extractor, mod = _import_sdk()
            except BackendError:
                if (
                    self.sdk_query_factory is None
                    or self.sdk_options_cls is None
                ):
                    raise
                mod = None
            else:
                if self.sdk_query_factory is None:
                    self.sdk_query_factory = query
                if self.sdk_options_cls is None:
                    self.sdk_options_cls = opts_cls
                if self.sdk_extract_text is None:
                    self.sdk_extract_text = extractor
                if sdk_module is None:
                    sdk_module = mod
        self.sdk_module = sdk_module
        if self.sdk_extract_text is None:
            self.sdk_extract_text = lambda m: getattr(m, "result", "") or ""
        if not os.environ.get(self.api_key_env):
            # Soft warning; SDK may still work via Bedrock/Vertex/Foundry.
            self.calls.append(
                {"warn": f"{self.api_key_env} not set in env"}
            )
        if self.enable_mcp_emit_intent:
            try:
                cfg = build_emit_intent_server(
                    sdk_module=self.sdk_module,
                    tool_factory=self.mcp_tool_factory,
                    server_factory=self.mcp_server_factory,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort, log + degrade
                self.calls.append({"warn": f"emit_intent MCP setup failed: {exc!r}"})
                cfg = None
            if cfg is not None:
                self.mcp_server_config = cfg
                self.mcp_tool_name = EMIT_INTENT_TOOL_QUALIFIED

    # ------------------------------------------------------------------
    @property
    def has_emit_intent_tool(self) -> bool:
        """True iff the in-process MCP ``emit_intent`` tool is registered."""

        return self.mcp_server_config is not None and self.mcp_tool_name is not None

    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        agent_name: str,
        allowed_tools: Sequence[str] = (),
        max_turns: int = 0,
        extra: dict | None = None,
    ) -> list[Intent]:
        full_prompt = self._compose_prompt(prompt)
        tools = tuple(allowed_tools) or self.allowed_tools_default
        turns = max_turns or self.max_turns_default
        options = self._build_options(tools, turns, extra or {})

        last_error: Exception | None = None
        for attempt in range(self.repair_attempts + 1):
            text, trajectory = await self._invoke_sdk(
                full_prompt if attempt == 0 else self._repair_prompt(full_prompt, last_error),
                options=options,
            )
            self.calls.append(
                {
                    "agent": agent_name,
                    "attempt": attempt,
                    "text_chars": len(text),
                    "tool_blocks": sum(
                        1 for _ in self._iter_tool_use_names(trajectory)
                    ),
                }
            )
            try:
                return parse_claude_trajectory(trajectory, fallback_text=text)
            except (IntentValidationError, NoIntentEmitted) as exc:
                last_error = exc
                if attempt == self.repair_attempts:
                    raise BackendError(
                        f"ClaudeBackend: failed to parse intents after "
                        f"{attempt + 1} attempt(s): {exc}"
                    ) from exc
        # unreachable
        raise BackendError("ClaudeBackend: exhausted retries unexpectedly")

    # ------------------------------------------------------------------
    def _compose_prompt(self, prompt: str) -> str:
        if self.has_emit_intent_tool:
            return (
                f"{prompt.rstrip()}\n\n"
                f"{_OUTPUT_INSTRUCTIONS_TOOL}\n\n"
                f"{_OUTPUT_INSTRUCTIONS_TEXT}\n"
            )
        return f"{prompt.rstrip()}\n\n{_OUTPUT_INSTRUCTIONS_TEXT}\n"

    def _repair_prompt(self, original: str, error: Exception | None) -> str:
        reason = str(error) if error else "unknown parse failure"
        return (
            f"{original}\n\n"
            f"Your previous reply did not validate ({reason}). "
            "Please retry — return ONLY the JSON envelope inside a "
            "```json fenced block."
        )

    def _build_options(
        self, allowed_tools: Sequence[str], max_turns: int, extra: dict
    ) -> Any:
        kwargs = dict(extra)
        merged_tools = list(allowed_tools)
        if self.has_emit_intent_tool and self.mcp_tool_name not in merged_tools:
            # ``allowed_tools`` here is the caller-supplied list; we always
            # want the MCP emit_intent tool available regardless of the
            # caller's preferences.
            merged_tools = [self.mcp_tool_name, *merged_tools]
        kwargs.setdefault("allowed_tools", merged_tools)
        kwargs.setdefault("max_turns", max_turns)
        if self.has_emit_intent_tool:
            servers = dict(kwargs.get("mcp_servers") or {})
            servers.setdefault(MCP_SERVER_NAME, self.mcp_server_config)
            kwargs["mcp_servers"] = servers
        if self.model:
            kwargs.setdefault("model", self.model)
        return self.sdk_options_cls(**kwargs)  # type: ignore[misc]

    async def _invoke_sdk(
        self, prompt: str, *, options: Any
    ) -> tuple[str, list[Any]]:
        """Run ``query()`` and return ``(joined_text, trajectory_list)``."""
        text_chunks: list[str] = []
        trajectory: list[Any] = []
        try:
            iterator = self.sdk_query_factory(prompt=prompt, options=options)  # type: ignore[misc]
            async for message in _aiter(iterator):
                trajectory.append(message)
                t = self.sdk_extract_text(message) or ""
                if t:
                    text_chunks.append(t)
        except BackendError:
            raise
        except Exception as exc:  # SDK exceptions vary across versions
            raise BackendError(f"ClaudeBackend: SDK call failed: {exc}") from exc
        return "".join(text_chunks), trajectory

    @staticmethod
    def _iter_tool_use_names(trajectory: Sequence[Any]):
        for msg in trajectory:
            for block in getattr(msg, "content", []) or []:
                name = getattr(block, "name", None)
                if name:
                    yield name


async def _aiter(possibly_async: Any):
    """Iterate over either an async iterator or a sync iterable."""
    if hasattr(possibly_async, "__aiter__"):
        async for item in possibly_async:
            yield item
    else:
        for item in possibly_async:
            yield item
        if False:  # pragma: no cover — keep this an async generator
            yield None


# Re-export ``EMIT_INTENT_TOOL_SCHEMA`` for callers that want to feed it
# into a future MCP custom-tool registration.
__all__ = [
    "ClaudeBackend",
    "EMIT_INTENT_TOOL_QUALIFIED",
    "EMIT_INTENT_TOOL_SCHEMA",
    "MCP_SERVER_NAME",
]
