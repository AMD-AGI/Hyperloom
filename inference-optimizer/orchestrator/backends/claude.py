"""ClaudeBackend — uses ``claude-agent-sdk`` to drive Claude (DESIGN v0.6 §14.2).

P1-5 implementation:

* In-process MCP server registers ``emit_intent`` so Claude calls it as a
  real tool (DESIGN §14.2). Each ``ToolUseBlock`` becomes one validated
  :class:`Intent` downstream.
* Lazy SDK import — if ``claude-agent-sdk`` isn't installed we raise a
  clear :class:`BackendError` at construction time so the CLI surfaces
  the exact pip command.
* Test seam — ``sdk_query_factory`` / ``sdk_options_cls`` /
  ``mcp_*_factory`` can be injected so unit tests bypass the real SDK
  + the network entirely.

Out of scope for P1-5:

* JSON-in-text fallback (silently degrades to NoIntentEmitted error)
* Repair-prompt retry on parse failure (let the Conductor surface the
  policy_denied / observation event so the agent self-corrects)
* Codex backend — ships in a follow-up commit; the Critic role still
  uses MockCriticBackend until then
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..intent_parser import (
    Intent,
    IntentType,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from .base import Backend, BackendError, BackendTurnResult
from .mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    MCP_SERVER_NAME,
    build_emit_intent_server,
)


log = logging.getLogger(__name__)


# Prompt suffix injected into every Claude turn so the model knows the
# tool contract. Conductor.compose_prompt() runs first; this is appended.
_OUTPUT_INSTRUCTIONS = f"""
==== OUTPUT FORMAT (REQUIRED) ====
You MUST communicate with the system by calling the `{EMIT_INTENT_TOOL_NAME}`
tool. Each call carries exactly one intent; call multiple times to emit
several intents in the same turn. Free-text replies are dropped.

Tool input shape:

  {{
    "intent_type": "<one of send_message|delegate|propose_action|request|"
                   "response|review_verdict|update_state|update_persona|"
                   "ask_question|answer|alert|kill_task|force_dispatch|"
                   "prune_branch|escalate_strategy_change>",
    "payload": {{ /* per-intent fields — see tool description */ }}
  }}

If you have nothing to say, call once with intent_type=send_message and
payload={{"topic":"heartbeat","body_md":"ok"}}.
==== END OUTPUT FORMAT ====
""".strip()


def _import_sdk() -> tuple[Any, Any, Any]:
    """Return ``(query, ClaudeAgentOptions, sdk_module)`` or raise.

    Only ``claude_agent_sdk`` is supported in v0.6 — legacy
    ``claude_code_sdk`` was deprecated upstream.
    """
    try:
        sdk = importlib.import_module("claude_agent_sdk")
    except ImportError as exc:
        raise BackendError(
            "claude-agent-sdk not installed; run "
            "`pip install claude-agent-sdk` (>= 0.1.65)."
        ) from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise BackendError(
            "claude_agent_sdk loaded but missing query / ClaudeAgentOptions"
        )
    return sdk.query, sdk.ClaudeAgentOptions, sdk


@dataclass
class ClaudeBackend:
    """Production Claude backend (DESIGN v0.6 §14.2). Implements :class:`Backend`.

    Args:
        model: Claude model id (e.g. ``"claude-opus-4-7"``); defaults to
            ``ANTHROPIC_MODEL`` env or library default.
        api_key_env: env var checked at construction (``ANTHROPIC_API_KEY``
            by default). Missing key is recorded as a soft warning — SDK
            may still authenticate via Bedrock / Vertex.
        max_turns_default: agent-loop budget when caller doesn't override.
        enable_mcp_emit_intent: if True (default), registers the
            in-process MCP ``emit_intent`` tool.
    """

    model: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_turns_default: int = 5
    enable_mcp_emit_intent: bool = True

    # Test seams — set these to bypass SDK import / network calls.
    sdk_query_factory: Callable[..., Any] | None = None
    sdk_options_cls: Any | None = None
    sdk_module: Any | None = None
    mcp_server_factory: Callable[..., Any] | None = None
    mcp_tool_factory: Callable[..., Any] | None = None

    name: str = "claude"
    calls: list[dict[str, Any]] = field(default_factory=list)
    mcp_server_config: Any | None = field(default=None, init=False)
    mcp_tool_name: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.sdk_query_factory is None or self.sdk_options_cls is None:
            try:
                query, opts_cls, mod = _import_sdk()
            except BackendError:
                if self.sdk_query_factory is None or self.sdk_options_cls is None:
                    raise
                mod = None
            else:
                if self.sdk_query_factory is None:
                    self.sdk_query_factory = query
                if self.sdk_options_cls is None:
                    self.sdk_options_cls = opts_cls
                if self.sdk_module is None:
                    self.sdk_module = mod
        if not os.environ.get(self.api_key_env):
            self.calls.append({"warn": f"{self.api_key_env} not set in env"})
        if self.enable_mcp_emit_intent:
            try:
                cfg = build_emit_intent_server(
                    sdk_module=self.sdk_module,
                    tool_factory=self.mcp_tool_factory,
                    server_factory=self.mcp_server_factory,
                )
            except Exception as exc:  # noqa: BLE001
                self.calls.append({"warn": f"emit_intent MCP setup failed: {exc!r}"})
                cfg = None
            if cfg is not None:
                self.mcp_server_config = cfg
                self.mcp_tool_name = EMIT_INTENT_TOOL_QUALIFIED

    @property
    def has_emit_intent_tool(self) -> bool:
        return self.mcp_server_config is not None and self.mcp_tool_name is not None

    # ------------------------------------------------------------------
    # Backend protocol
    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        full_prompt = self._compose_prompt(prompt)
        max_turns_use = max_turns or self.max_turns_default
        options = self._build_options(
            tools=tools or [],
            max_turns=max_turns_use,
            system_prompt=system_prompt,
        )
        intents, raw_text, tool_block_count = await self._invoke_and_collect(
            full_prompt, options
        )
        self.calls.append({
            "prompt_chars": len(full_prompt),
            "tool_blocks": tool_block_count,
            "intents": len(intents),
            "max_turns": max_turns_use,
        })
        if not intents:
            raise NoIntentEmitted(
                f"claude reply contained no parseable emit_intent tool_use "
                f"blocks (raw_text_len={len(raw_text)}, tool_blocks={tool_block_count})"
            )
        return BackendTurnResult(
            intents=intents, raw_text=raw_text,
            metadata={"tool_blocks": tool_block_count, "model": self.model},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compose_prompt(self, prompt: str) -> str:
        return f"{prompt}\n\n{_OUTPUT_INSTRUCTIONS}"

    def _build_options(
        self,
        *,
        tools: list[str],
        max_turns: int,
        system_prompt: str | None,
    ) -> Any:
        kwargs: dict[str, Any] = {"max_turns": max_turns}
        if self.model:
            kwargs["model"] = self.model
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        # Allowed tools = caller-provided + our MCP-qualified emit_intent.
        allowed = list(tools)
        if self.mcp_tool_name and self.mcp_tool_name not in allowed:
            allowed.append(self.mcp_tool_name)
        if allowed:
            kwargs["allowed_tools"] = allowed
        if self.mcp_server_config is not None:
            kwargs["mcp_servers"] = {MCP_SERVER_NAME: self.mcp_server_config}
        return self.sdk_options_cls(**kwargs)

    async def _invoke_and_collect(
        self, prompt: str, options: Any
    ) -> tuple[list[Intent], str, int]:
        """Stream messages from the SDK, collect intents + raw text + tool counts."""
        intents: list[Intent] = []
        text_chunks: list[str] = []
        tool_block_count = 0
        async for message in self.sdk_query_factory(prompt=prompt, options=options):
            for block in self._iter_blocks(message):
                if self._is_tool_use_for_emit_intent(block):
                    tool_block_count += 1
                    intent = self._parse_tool_use_block(block)
                    if intent is not None:
                        intents.append(intent)
                else:
                    txt = self._extract_text(block)
                    if txt:
                        text_chunks.append(txt)
            # ResultMessage carries .result text in many SDKs
            result_text = getattr(message, "result", None)
            if isinstance(result_text, str):
                text_chunks.append(result_text)
        return intents, "".join(text_chunks), tool_block_count

    @staticmethod
    def _iter_blocks(message: Any):
        return list(getattr(message, "content", None) or [])

    def _is_tool_use_for_emit_intent(self, block: Any) -> bool:
        # Match by class name so we don't depend on SDK internals.
        cls_name = type(block).__name__
        if cls_name not in ("ToolUseBlock", "ServerToolUseBlock"):
            return False
        name = getattr(block, "name", "")
        return name in (EMIT_INTENT_TOOL_NAME, EMIT_INTENT_TOOL_QUALIFIED)

    def _parse_tool_use_block(self, block: Any) -> Intent | None:
        raw_input = getattr(block, "input", None) or {}
        try:
            envelope = {"intents": [{
                "intent_type": raw_input.get("intent_type"),
                "payload": raw_input.get("payload") or {},
            }]}
            validated = validate_envelope(envelope)
        except IntentValidationError as exc:
            log.info("claude tool_use validation failed: %s", exc)
            return None
        return validated[0] if validated else None

    @staticmethod
    def _extract_text(block: Any) -> str:
        if type(block).__name__ == "TextBlock":
            return getattr(block, "text", "") or ""
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "") or ""
        t = getattr(block, "text", None)
        return t if isinstance(t, str) else ""


__all__ = ["ClaudeBackend"]
