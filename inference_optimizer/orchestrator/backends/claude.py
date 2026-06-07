# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ClaudeBackend — uses ``claude-agent-sdk`` to drive Claude.

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
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ...protocol.intent import (
    Intent,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from .base import BackendError, BackendTurnResult, parse_call_timeout_env
from .mcp_context_tools import (
    CONTEXT_TOOL_QUALIFIED_NAMES,
    MCP_SERVER_NAME as CONTEXT_MCP_SERVER_NAME,
    ContextProvider,
    build_context_tools_server,
)
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


# Conversational-mode floors (plan Step 1). A persistent ReAct turn may
# pull several read-only context tools before it emits an intent, so it
# needs more agentic turns and a longer wall-clock budget than a single
# stateless propose-and-go turn.
_CONVERSATIONAL_MIN_MAX_TURNS: int = 12
_CONVERSATIONAL_DEFAULT_TIMEOUT_SEC: float = 300.0


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
    """Production Claude backend. Implements :class:`Backend`.

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
    #
    # Conversational mode (see ``conversational`` below) raises this floor:
    # a persistent ReAct turn can call several read-only context tools
    # before emitting an intent, so it needs more headroom than a single
    # stateless propose-and-go turn.
    max_turns_default: int = 4
    # Persistent-conversation mode for the Orchestration reactor (plan
    # Step 1). When True, ``run()`` continues the SAME Claude session
    # across ticks via the SDK ``resume=<session_id>`` option instead of
    # starting a fresh stateless conversation each turn. The per-tick
    # prompt is then a *delta* (new inbox events + phase line) appended as
    # the next user turn, and the model's chain-of-thought / plan persists
    # in the conversation history rather than being re-derived from a full
    # state dump every tick. ``session_id`` is captured from the SDK
    # message stream and fed back on the next call. kernel / critic /
    # robustness keep the default stateless mode.
    conversational: bool = False
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
    #
    # Env-var override ``INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC`` lets
    # operators bump this when opus-class models with a heavy orchestrator
    # system prompt (~22 KB) and a 4-turn agentic loop consistently exceed
    # 120 s on the AMD gateway under load. Invalid values fall back to the
    # 120s default rather than crashing backend construction; backend
    # boot-time refuses to die for a malformed env-var.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC",
            default=120.0,
        )
    )

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
    # Conversational-mode runtime state (plan Step 1).
    # ``_session_id`` is the SDK session token captured from the last
    # turn; when set and ``conversational`` is True, the next ``run()``
    # passes ``resume=<session_id>`` so the model continues the same
    # conversation. ``reset_conversation()`` clears it so the Coordinator
    # can start a fresh session after a checkpoint/compaction (plan
    # Step 4) or on resume rebuild.
    _session_id: str | None = field(default=None, init=False)
    # Read-only context-pull MCP server config (plan Step 2). Built when
    # the Coordinator calls ``set_context_provider``; merged into the SDK
    # options alongside the emit_intent server so a conversational turn
    # can pull mission status / gaps / analysis.md / etc. on demand
    # instead of receiving a full state dump every tick.
    _context_server_config: Any | None = field(default=None, init=False)

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
        if self.conversational:
            # A persistent ReAct turn may call several read-only context
            # tools (plan Step 2) before emitting an intent, so give the
            # in-tick agentic loop more turns and a longer wall-clock
            # budget than the stateless default. Operators can still
            # override the timeout via the env var below.
            if self.max_turns_default < _CONVERSATIONAL_MIN_MAX_TURNS:
                self.max_turns_default = _CONVERSATIONAL_MIN_MAX_TURNS
            if os.environ.get(
                "INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC", "",
            ).strip() == "":
                # No explicit operator override -> raise the floor so the
                # extra tool round-trips don't trip the 120s wall.
                self.call_timeout_s = max(
                    self.call_timeout_s, _CONVERSATIONAL_DEFAULT_TIMEOUT_SEC,
                )
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
        allow_no_intent: bool = False,
    ) -> BackendTurnResult:
        # ``allow_no_intent`` (plan Step 4): a conversational summary /
        # checkpoint turn deliberately asks the model for a plain-text
        # (JSON) reply rather than an ``emit_intent`` tool call, so the
        # usual "no emit_intent ⇒ NoIntentEmitted" guard must be relaxed
        # for those turns. Normal reactor turns leave it False.
        full_prompt = self._compose_prompt(prompt)
        max_turns_use = max_turns or self.max_turns_default
        if self.raw_completion:
            # Claude Code counts the single assistant text message as a
            # turn and errors when max_turns is reached, so a literal
            # max_turns=1 trips even on a clean one-shot answer. All
            # tools are disallowed in raw mode, so the model cannot
            # loop — generous headroom guarantees the text turn returns.
            max_turns_use = max(max_turns_use, 8)
        # Conversational mode: continue the same SDK session across ticks
        # by resuming the captured session_id. The first turn (no captured
        # id yet) starts a fresh session and we capture its id below.
        resume_session = self._session_id if self.conversational else None
        options = self._build_options(
            tools=tools or [],
            max_turns=max_turns_use,
            system_prompt=system_prompt,
            resume_session_id=resume_session,
        )
        # Cache-metric extraction (5-tuple from _invoke_and_collect)
        # plus a timeout guard: wrap the SDK call in asyncio.wait_for so
        # an upstream proxy stall doesn't park the reactor indefinitely.
        try:
            (
                intents, raw_text, tool_block_count, usage, session_id,
            ) = await asyncio.wait_for(
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
        # Stash the per-tick cache metric on backend.calls so the audit
        # scripts can compute session-level cache_hit_rate without
        # needing a separate Coordinator wiring path.
        # Capture the SDK session token so the next conversational turn
        # resumes the same conversation. Only update when we actually got
        # an id back; a missing id leaves the previous one intact so a
        # transient stream without a terminal ResultMessage doesn't drop
        # the conversation thread.
        if self.conversational:
            # Observability for the persistent-conversation design (plan
            # Steps 1-3): make resume continuity + on-demand tool usage
            # visible in the run log. ``resumed`` True means this turn
            # continued the prior SDK session; a changing session_id while
            # resumed=False would indicate resume is NOT working.
            log.info(
                "claude[conv] turn: resumed=%s prev_session=%s "
                "new_session=%s tool_blocks=%d intents=%d prompt_chars=%d",
                bool(resume_session),
                (resume_session or "")[-12:],
                (session_id or "")[-12:],
                tool_block_count,
                len(intents),
                len(full_prompt),
            )
            if session_id:
                self._session_id = session_id
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
        if not intents and not self.raw_completion and not allow_no_intent:
            raise NoIntentEmitted(
                f"claude reply contained no parseable emit_intent tool_use "
                f"blocks (raw_text_len={len(raw_text)}, tool_blocks={tool_block_count})"
            )
        # Expose cache metrics on metadata too so a Coordinator-side
        # post-tick hook can read them off the BackendTurnResult without
        # scanning backend.calls.
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

    def set_context_provider(self, provider: ContextProvider | None) -> None:
        """Attach (or clear) the read-only context-pull MCP server.

        The Coordinator calls this once after construction with a
        :class:`ContextProvider` bound to its live ``SharedState`` so the
        conversational Orchestration turn can pull context on demand
        (plan Step 2). Passing ``None`` detaches the server. Best-effort:
        a build failure is recorded as a soft warning and leaves the
        backend usable without the pull tools.
        """
        if provider is None:
            self._context_server_config = None
            return
        try:
            self._context_server_config = build_context_tools_server(
                provider,
                sdk_module=self.sdk_module,
                tool_factory=self.mcp_tool_factory,
                server_factory=self.mcp_server_factory,
            )
        except Exception as exc:  # noqa: BLE001
            self.calls.append({"warn": f"context tools MCP setup failed: {exc!r}"})
            self._context_server_config = None

    @property
    def has_context_tools(self) -> bool:
        return self._context_server_config is not None

    def reset_conversation(self) -> None:
        """Drop the captured session so the next ``run`` starts fresh.

        Used by the Coordinator after a checkpoint/compaction (plan
        Step 4) or when rebuilding the conversation on resume: the next
        conversational turn re-seeds a new SDK session with the compacted
        memory instead of resuming an unbounded transcript.
        """
        self._session_id = None

    @property
    def conversation_session_id(self) -> str | None:
        """Current SDK session token (conversational mode), or None."""
        return self._session_id if self.conversational else None

    def _build_options(
        self,
        *,
        tools: list[str],
        max_turns: int,
        system_prompt: str | None,
        resume_session_id: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"max_turns": max_turns}
        if self.model:
            kwargs["model"] = self.model
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if resume_session_id:
            # SDK ClaudeAgentOptions.resume continues an existing session
            # by id (claude-agent-sdk >= 0.2). Falls back gracefully when
            # the options class doesn't accept it (older SDK) — see
            # _build_options caller, which only sets this in conversational
            # mode after a session_id was captured.
            kwargs["resume"] = resume_session_id
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
        # Read-only context-pull tools (plan Step 2). When the context
        # server is attached, allow-list its qualified tool names so the
        # conversational turn can call them. The bare names are already
        # allow-listed via PolicyGate.allowed_tools_for_agent, but the SDK
        # needs the MCP-qualified form to wire into the tool registry.
        if self._context_server_config is not None:
            for qname in CONTEXT_TOOL_QUALIFIED_NAMES:
                if qname not in allowed:
                    allowed.append(qname)
        if allowed:
            kwargs["allowed_tools"] = allowed
        mcp_servers: dict[str, Any] = {}
        if self.mcp_server_config is not None:
            mcp_servers[MCP_SERVER_NAME] = self.mcp_server_config
        if self._context_server_config is not None:
            mcp_servers[CONTEXT_MCP_SERVER_NAME] = self._context_server_config
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers
        # Capture CLI stderr so failures are diagnosable instead of opaque
        # "Command failed with exit code 1".
        kwargs["stderr"] = self._stderr_sink
        return self._instantiate_options(kwargs)

    def _instantiate_options(self, kwargs: dict[str, Any]) -> Any:
        """Build options, dropping ``resume`` if the SDK can't accept it.

        Older ``claude-agent-sdk`` builds may not expose ``resume`` on
        ``ClaudeAgentOptions``. Rather than crash the reactor, fall back
        to a stateless turn (no resume) and record a one-time warning so
        the operator knows conversational continuity degraded.
        """
        try:
            return self.sdk_options_cls(**kwargs)
        except TypeError as exc:
            if "resume" in kwargs:
                kwargs.pop("resume", None)
                self.calls.append({
                    "warn": (
                        "SDK ClaudeAgentOptions rejected resume= "
                        f"({exc!r}); falling back to stateless turn"
                    ),
                })
                return self.sdk_options_cls(**kwargs)
            raise

    def _stderr_sink(self, line: str) -> None:
        """Default stderr handler — append to ``self.calls`` for postmortems."""
        text = line.strip()
        if text:
            self.calls.append({"stderr": text})

    async def _invoke_and_collect(
        self, prompt: str, options: Any
    ) -> tuple[list[Intent], str, int, dict[str, Any], str | None]:
        """Stream messages from the SDK, collect intents + raw text +
        tool counts + the most recent `ResultMessage.usage` dict + the
        SDK ``session_id`` (for conversational resume, plan Step 1).

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
        session_id: str | None = None
        try:
            async for message in self.sdk_query_factory(prompt=prompt, options=options):
                # Capture the session token from any message that carries
                # it (AssistantMessage / ResultMessage both expose it on
                # claude-agent-sdk >= 0.2). The last seen wins.
                msg_session = getattr(message, "session_id", None)
                if isinstance(msg_session, str) and msg_session:
                    session_id = msg_session
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
                # ResultMessage carries .usage on terminal messages
                # (Anthropic Messages API response schema). The SDK
                # propagates this dict verbatim. We overwrite (not
                # accumulate) because the last message of a multi-turn
                # session reports the cumulative session usage.
                msg_usage = getattr(message, "usage", None)
                if isinstance(msg_usage, dict) and msg_usage:
                    last_usage = dict(msg_usage)
        except Exception as exc:
            # SDK ≥ 0.2.82 / CLI ≥ 2.1.123 may raise "Claude Code returned
            # an error result: success" when the CLI exits after emitting a
            # valid result with is_error=True + subtype='success' (max-turns
            # reached). If we already collected intents, return them rather
            # than losing the entire turn.
            err_str = str(exc)
            if "error result: success" in err_str:
                if intents:
                    log.warning(
                        "claude SDK raised '%s' but %d intents already collected; "
                        "returning partial results",
                        err_str, len(intents),
                    )
                else:
                    log.warning(
                        "claude SDK raised '%s' with no intents collected; "
                        "treating as no-intent turn (will retry next tick)",
                        err_str,
                    )
            else:
                raise
        # Prefer the consolidated ResultMessage text; fall back to the
        # streamed TextBlocks only when no result was emitted.
        raw_text = "".join(result_chunks) or "".join(text_chunks)
        return intents, raw_text, tool_block_count, last_usage, session_id

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
