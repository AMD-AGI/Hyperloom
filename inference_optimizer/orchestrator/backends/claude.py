"""ClaudeBackend — uses ``claude-agent-sdk`` to drive Claude ().

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
* Repair-prompt retry on parse failure (let the Coordinator surface the
  policy_denied / observation event so the agent self-corrects)
* Codex backend — ships in a follow-up commit; the Critic role still
  uses MockCriticBackend until then
"""

from __future__ import annotations

import asyncio
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
# tool contract. Coordinator.compose_prompt() runs first; this is appended.
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


# Claude Code built-in tools disallowed in raw_completion mode so the
# model produces exactly one text turn (no agentic tool loop).
_RAW_COMPLETION_DISALLOWED_TOOLS: tuple[str, ...] = (
    "Bash", "BashOutput", "KillShell", "Read", "Write", "Edit",
    "NotebookEdit", "Glob", "Grep", "Task", "WebFetch", "WebSearch",
    "TodoWrite", "ExitPlanMode", "SlashCommand",
)


def _import_sdk() -> tuple[Any, Any, Any]:
    """Return ``(query, ClaudeAgentOptions, sdk_module)`` or raise.

    Only ``claude_agent_sdk`` is supported in the legacy release — legacy
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
    """Production Claude backend (). Implements :class:`Backend`.

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
    # Default of 4 covers the typical reactor-tick sequence:
    # tool_use → tool_result → final assistant text (3 messages).
    # Larger = more retries on validation failure but more tokens.
    max_turns_default: int = 4
    enable_mcp_emit_intent: bool = True
    # Raw single-shot completion mode for callers that drive their own
    # loop + parse the model's plain text themselves (e.g. the
    # dynamic_action ReAct runner). When True: the emit_intent MCP
    # server + output-format suffix are skipped, all tools are
    # disallowed so the model produces exactly one text turn, and
    # ``run`` returns ``raw_text`` without requiring an emitted intent.
    raw_completion: bool = False
    # Wall-clock cap for one ``run()`` call. The claude-agent-sdk shells
    # out to the ``claude`` CLI which talks to the AMD primus-safe
    # gateway; if the gateway is unreachable the subprocess can hang
    # on TCP for minutes, stalling the orchestrator reactor. 120s is
    # well above a normal turn (~10–30s) but bounds the worst case.
    call_timeout_s: float = 120.0

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
        if self.raw_completion:
            self.enable_mcp_emit_intent = False
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
        if self.raw_completion:
            # Claude Code counts the single assistant text message as a
            # turn and errors when max_turns is reached, so a literal
            # max_turns=1 trips even on a clean one-shot answer. All
            # tools are disallowed in raw mode, so the model cannot
            # loop — generous headroom guarantees the text turn returns.
            max_turns_use = max(max_turns_use, 8)
        options = self._build_options(
            tools=tools or [],
            max_turns=max_turns_use,
            system_prompt=system_prompt,
        )
        # Combine N6 (cache metric extraction via 4-tuple from
        # _invoke_and_collect) with main's timeout guard (#243 area):
        # wrap the SDK call in asyncio.wait_for so an upstream proxy
        # stall doesn't park the reactor indefinitely.
        try:
            intents, raw_text, tool_block_count, usage = await asyncio.wait_for(
                self._invoke_and_collect(full_prompt, options),
                timeout=self.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            self.calls.append({
                "warn": (
                    f"claude SDK call timed out after {self.call_timeout_s:.0f}s; "
                    "treating as no-intent so the reactor pass can proceed"
                ),
            })
            raise BackendError(
                f"Claude backend timed out after {self.call_timeout_s:.0f}s "
                "(likely upstream proxy stall)"
            ) from exc
        # N6: stash the per-tick cache metric on backend.calls so the
        # audit scripts (N7) can compute session-level cache_hit_rate
        # without needing a separate Coordinator wiring path.
        cache_creation = self._safe_int(
            usage.get("cache_creation_input_tokens") if usage else None
        )
        cache_read = self._safe_int(
            usage.get("cache_read_input_tokens") if usage else None
        )
        input_tokens = self._safe_int(usage.get("input_tokens") if usage else None)
        output_tokens = self._safe_int(usage.get("output_tokens") if usage else None)
        self.calls.append({
            "prompt_chars": len(full_prompt),
            "tool_blocks": tool_block_count,
            "intents": len(intents),
            "max_turns": max_turns_use,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
        if not intents and not self.raw_completion:
            raise NoIntentEmitted(
                f"claude reply contained no parseable emit_intent tool_use "
                f"blocks (raw_text_len={len(raw_text)}, tool_blocks={tool_block_count})"
            )
        # N6: expose cache metrics on metadata too so a Coordinator-side
        # post-tick hook (future / N7+) can read them off the
        # BackendTurnResult without scanning backend.calls.
        return BackendTurnResult(
            intents=intents, raw_text=raw_text,
            metadata={
                "tool_blocks": tool_block_count,
                "model": self.model,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compose_prompt(self, prompt: str) -> str:
        if self.raw_completion:
            return prompt
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
        if self.raw_completion:
            # Single text turn: no MCP tools, and every Claude Code
            # built-in tool disallowed. The caller parses the model's
            # plain text itself, so any tool use would only burn a
            # turn and trip the max_turns cap.
            kwargs["allowed_tools"] = []
            kwargs["disallowed_tools"] = list(_RAW_COMPLETION_DISALLOWED_TOOLS)
            kwargs["stderr"] = self._stderr_sink
            return self.sdk_options_cls(**kwargs)
        # Allowed tools = caller-provided + our MCP-qualified emit_intent.
        # Drop the unqualified short name "emit_intent" — Claude CLI rejects
        # bare tool names that don't match a real registered tool. The MCP
        # qualified form "mcp__inference_optimizer__emit_intent" is what
        # actually wires into the SDK tool registry.
        allowed = [t for t in tools if t != EMIT_INTENT_TOOL_NAME]
        if self.mcp_tool_name and self.mcp_tool_name not in allowed:
            allowed.append(self.mcp_tool_name)
        if allowed:
            kwargs["allowed_tools"] = allowed
        if self.mcp_server_config is not None:
            kwargs["mcp_servers"] = {MCP_SERVER_NAME: self.mcp_server_config}
        # Capture CLI stderr so failures are diagnosable instead of opaque
        # "Command failed with exit code 1".
        kwargs["stderr"] = self._stderr_sink
        return self.sdk_options_cls(**kwargs)

    def _stderr_sink(self, line: str) -> None:
        """Default stderr handler — append to ``self.calls`` for postmortems."""
        text = line.strip()
        if text:
            self.calls.append({"stderr": text})

    async def _invoke_and_collect(
        self, prompt: str, options: Any
    ) -> tuple[list[Intent], str, int, dict[str, Any]]:
        """Stream messages from the SDK, collect intents + raw text +
        tool counts + the most recent `ResultMessage.usage` dict.

        Roofline-v2 N6: extract `usage` so the Coordinator (or audit
        scripts via `backend.calls`) can read
        `cache_creation_input_tokens` / `cache_read_input_tokens`
        and measure how effective Claude Code's automatic prompt
        caching is at hitting our SECTION-A/B stable-prefix design
        (§5.1, §8.8).

        `usage` mirrors what task_manager.py in Primus-Claw/OOB reads
        (lines 152-153) — the field shape is fixed by the Anthropic
        Messages API response and surfaces here because Claude Code
        forwards it on its terminal `ResultMessage`.
        """
        intents: list[Intent] = []
        text_chunks: list[str] = []
        result_chunks: list[str] = []
        tool_block_count = 0
        last_usage: dict[str, Any] = {}
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
            # ResultMessage.result is the consolidated final assistant
            # text — the SAME content already streamed as TextBlocks.
            # Keep it separate so we don't double-count: the joined
            # stream and the result would otherwise concatenate into a
            # duplicated payload (breaks raw_completion JSON parsing).
            result_text = getattr(message, "result", None)
            if isinstance(result_text, str) and result_text:
                result_chunks.append(result_text)
            # N6: ResultMessage carries .usage on terminal messages
            # (Anthropic Messages API response schema). The SDK
            # propagates this dict verbatim. We overwrite (not
            # accumulate) because the last message of a multi-turn
            # session reports the cumulative session usage.
            msg_usage = getattr(message, "usage", None)
            if isinstance(msg_usage, dict) and msg_usage:
                last_usage = dict(msg_usage)
        # Prefer the consolidated ResultMessage text; fall back to the
        # streamed TextBlocks only when no result was emitted.
        raw_text = "".join(result_chunks) or "".join(text_chunks)
        return intents, raw_text, tool_block_count, last_usage

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
