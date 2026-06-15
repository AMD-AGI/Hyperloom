# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ClaudeBackend — uses ``claude-agent-sdk`` to drive Claude.

An in-process MCP server registers ``emit_intent`` (DESIGN §14.2) so each
``ToolUseBlock`` becomes one validated :class:`Intent`. SDK import is lazy
(clear :class:`BackendError` if missing); ``sdk_query_factory`` /
``sdk_options_cls`` / ``mcp_*_factory`` are test seams that bypass the real
SDK + network.
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
from .base import (
    BackendError,
    BackendTurnResult,
    RetryPolicy,
    parse_call_timeout_env,
    retry_with_backoff,
)
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


# Prompt suffix appended to every Claude turn so the model knows the tool contract.
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


# Conversational-mode floors (plan Step 1): a persistent ReAct turn pulls
# context tools before emitting, so it needs more turns + wall-clock budget.
_CONVERSATIONAL_MIN_MAX_TURNS: int = 12
_CONVERSATIONAL_DEFAULT_TIMEOUT_SEC: float = 300.0


# Built-in tools disallowed in raw_completion mode so the model produces
# exactly one text turn (no agentic tool loop).
_RAW_COMPLETION_DISALLOWED_TOOLS: tuple[str, ...] = (
    "Bash", "BashOutput", "KillShell", "Read", "Write", "Edit",
    "NotebookEdit", "Glob", "Grep", "Task", "WebFetch", "WebSearch",
    "TodoWrite", "ExitPlanMode", "SlashCommand",
)


def _import_sdk() -> tuple[Any, Any, Any]:
    """Return ``(query, ClaudeAgentOptions, sdk_module)`` or raise.

    Only ``claude_agent_sdk`` is supported (``claude_code_sdk`` deprecated).
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
    # Default 4 covers tool_use → tool_result → final text; conversational
    # mode raises this floor (more context-pull headroom per turn).
    max_turns_default: int = 4
    # Persistent-conversation mode (plan Step 1): resume the SAME SDK session
    # across ticks (``resume=<session_id>``) feeding only a per-tick delta,
    # instead of a fresh stateless conversation. kernel / critic / robustness
    # stay stateless.
    conversational: bool = False
    enable_mcp_emit_intent: bool = True
    # Raw single-shot completion mode: skips the emit_intent server + suffix,
    # disallows all tools, and returns ``raw_text`` without requiring an
    # emitted intent.
    raw_completion: bool = False
    # Wall-clock cap for one ``run()`` call; bounds a hung ``claude`` CLI /
    # unreachable gateway. Env override: ``INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC``.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC",
            default=120.0,
        )
    )
    # Bounded transient-failure retry/backoff (R6): a multi-day run must absorb
    # gateway stalls / 5xx / connection blips instead of failing the turn.
    # Env override via INFERENCE_OPTIMIZER_LLM_RETRY_* (ATTEMPTS=1 disables).
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.from_env)

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
    # SDK session token captured last turn; replayed via ``resume`` in
    # conversational mode. ``reset_conversation()`` clears it (plan Step 1/4).
    _session_id: str | None = field(default=None, init=False)
    # Read-only context-pull MCP server config (plan Step 2), set via
    # ``set_context_provider`` and merged into the SDK options.
    _context_server_config: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Resolve the SDK and optionally register the ``emit_intent`` tool.

        Imports the real SDK unless test seams already supply ``query`` /
        options, records a soft warning when the API-key env var is unset,
        disables MCP tooling in ``raw_completion`` mode, and otherwise builds
        the in-process ``emit_intent`` MCP server config.

        Raises:
            BackendError: If the SDK cannot be imported and no test factories
                were provided to substitute for it.
        """
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
        """Whether the ``emit_intent`` MCP tool is wired up and usable.

        Returns:
            bool: ``True`` when both the MCP server config and qualified tool
            name are present.
        """
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
        """Run a single backend turn against Claude and parse the result.

        Args:
            prompt: User prompt for this turn.
            system_prompt: Optional system prompt override.
            tools: Tool names to enable for the turn.
            max_turns: Maximum agent turns; falls back to the backend
                default when falsy.
            allow_no_intent: When ``True``, relax the guard that requires
                an ``emit_intent`` (e.g. for summary/checkpoint turns).

        Returns:
            The parsed :class:`BackendTurnResult` for the turn.
        """
        # ``allow_no_intent`` (plan Step 4): a summary/checkpoint turn asks
        # for plain-text instead of emit_intent, so relax the no-intent guard.
        full_prompt = self._compose_prompt(prompt)
        max_turns_use = max_turns or self.max_turns_default
        if self.raw_completion:
            # Claude Code counts the single text message as a turn, so a
            # literal max_turns=1 trips; give headroom (tools are disallowed).
            max_turns_use = max(max_turns_use, 8)
        resume_session = self._session_id if self.conversational else None
        options = self._build_options(
            tools=tools or [],
            max_turns=max_turns_use,
            system_prompt=system_prompt,
            resume_session_id=resume_session,
        )
        # Timeout guard: an upstream proxy stall must not park the reactor.
        # Bounded retry/backoff (R6) absorbs transient stalls / blips across a
        # multi-day run; a per-attempt wall-clock cap still bounds each try.
        async def _one_attempt() -> tuple[Any, ...]:
            return await asyncio.wait_for(
                self._invoke_and_collect(full_prompt, options),
                timeout=self.call_timeout_s,
            )

        def _note_retry(attempt: int, exc: BaseException, delay: float) -> None:
            self.calls.append({
                "warn": (
                    f"claude SDK transient failure (attempt {attempt}): {exc!r}; "
                    f"retrying in {delay:.2f}s"
                ),
            })

        try:
            (
                intents, raw_text, tool_block_count, usage, session_id,
            ) = await retry_with_backoff(
                _one_attempt,
                policy=self.retry_policy,
                retry_on=(
                    asyncio.TimeoutError, BackendError, ConnectionError, OSError,
                ),
                on_retry=_note_retry,
            )
        except asyncio.TimeoutError as exc:
            self.calls.append({
                "warn": (
                    f"claude SDK call timed out after {self.call_timeout_s:.0f}s "
                    f"(retries exhausted); treating as no-intent so the reactor "
                    "pass can proceed"
                ),
            })
            raise BackendError(
                f"Claude backend timed out after {self.call_timeout_s:.0f}s "
                "(likely upstream proxy stall)"
            ) from exc
        # Capture the SDK session token for the next conversational resume;
        # only overwrite on a non-empty id so a stream without a terminal
        # ResultMessage doesn't drop the conversation thread.
        if self.conversational:
            # Observability: surface resume continuity + tool usage per turn.
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
        return BackendTurnResult(
            intents=intents, raw_text=raw_text,
            metadata={
                "tool_blocks": tool_block_count,
                "model": self.model,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                # Full conversation text so the caller (which holds the
                # session_dir / component / tick context the stateless
                # backend lacks) can persist it to conversations.jsonl.
                # The composed prompt carries the user turn; the system
                # prompt is snapshotted once under agents/<role>/.
                "prompt": full_prompt,
                "response": raw_text,
            },
        )

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Coerce a possibly-missing usage value to a non-negative int.

        Args:
            value (Any): A token-count value that may be ``None`` or
                non-numeric.

        Returns:
            int: The integer value, or ``0`` when it is falsy or not coercible.
        """
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compose_prompt(self, prompt: str) -> str:
        """Append the emit_intent output-format suffix unless in raw mode.

        Args:
            prompt (str): The base turn prompt.

        Returns:
            str: The prompt unchanged in ``raw_completion`` mode, otherwise the
            prompt with the required output-format instructions appended.
        """
        if self.raw_completion:
            return prompt
        return f"{prompt}\n\n{_OUTPUT_INSTRUCTIONS}"

    def set_context_provider(self, provider: ContextProvider | None) -> None:
        """Attach (or clear) the read-only context-pull MCP server (plan Step 2).

        ``None`` detaches it. Best-effort: a build failure is recorded as a
        soft warning and leaves the backend usable without the pull tools.
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
        """Whether the context-tools MCP server is configured.

        Returns:
            ``True`` if a context-server config was set up successfully.
        """
        return self._context_server_config is not None

    def reset_conversation(self) -> None:
        """Drop the captured session so the next ``run`` starts fresh.

        Used after a checkpoint/compaction or resume rebuild (plan Step 4).
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
        """Build the SDK options object for one turn.

        In ``raw_completion`` mode all tools are disallowed so the model emits a
        single text turn; otherwise the qualified ``emit_intent`` tool and MCP
        server are wired into the allow-list. CLI stderr is captured for
        postmortems in both modes.

        Args:
            tools (list[str]): Caller-provided allowed tool names.
            max_turns (int): Agent-loop budget to pass through to the SDK.
            system_prompt (str | None): Optional system prompt for the turn.

        Returns:
            Any: A constructed ``ClaudeAgentOptions`` instance.
        """
        kwargs: dict[str, Any] = {"max_turns": max_turns}
        if self.model:
            kwargs["model"] = self.model
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if resume_session_id:
            # Resume an existing session by id (claude-agent-sdk >= 0.2).
            kwargs["resume"] = resume_session_id
        if self.raw_completion:
            # Single text turn: no MCP tools, all built-ins disallowed.
            kwargs["allowed_tools"] = []
            kwargs["disallowed_tools"] = list(_RAW_COMPLETION_DISALLOWED_TOOLS)
            kwargs["stderr"] = self._stderr_sink
            return self.sdk_options_cls(**kwargs)
        # Drop the bare "emit_intent" name (CLI rejects unregistered names);
        # the MCP-qualified form is what wires into the SDK tool registry.
        allowed = [t for t in tools if t != EMIT_INTENT_TOOL_NAME]
        if self.mcp_tool_name and self.mcp_tool_name not in allowed:
            allowed.append(self.mcp_tool_name)
        # Allow-list the context-pull tools' qualified names (plan Step 2);
        # the SDK needs the qualified form even though bare names are gated.
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
        # Capture CLI stderr so failures are diagnosable.
        kwargs["stderr"] = self._stderr_sink
        return self._instantiate_options(kwargs)

    def _instantiate_options(self, kwargs: dict[str, Any]) -> Any:
        """Build options, dropping ``resume`` if the SDK can't accept it.

        Older SDK builds lack ``resume``; fall back to a stateless turn
        (with a one-time warning) rather than crashing the reactor.
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
        """Default stderr handler — append to ``self.calls`` for postmortems.

        Args:
            line (str): One line of CLI stderr output; blank lines are dropped.
        """
        text = line.strip()
        if text:
            self.calls.append({"stderr": text})

    async def _invoke_and_collect(
        self, prompt: str, options: Any
    ) -> tuple[list[Intent], str, int, dict[str, Any], str | None]:
        """Stream SDK messages, collecting intents, raw text, tool counts,
        the latest `ResultMessage.usage` dict, and the SDK ``session_id``.

        `usage` (cache_creation/read_input_tokens) measures prompt-cache
        effectiveness against the SECTION-A/B stable-prefix design (§5.1, §8.8).
        """
        intents: list[Intent] = []
        text_chunks: list[str] = []
        result_chunks: list[str] = []
        tool_block_count = 0
        last_usage: dict[str, Any] = {}
        session_id: str | None = None
        try:
            async for message in self.sdk_query_factory(prompt=prompt, options=options):
                # Capture the session token from any message; last seen wins.
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
                # ResultMessage.result duplicates the streamed TextBlocks;
                # keep it separate to avoid double-counting (would break
                # raw_completion JSON parsing).
                result_text = getattr(message, "result", None)
                if isinstance(result_text, str) and result_text:
                    result_chunks.append(result_text)
                # Overwrite (not accumulate) usage: the terminal message
                # reports cumulative session usage.
                msg_usage = getattr(message, "usage", None)
                if isinstance(msg_usage, dict) and msg_usage:
                    last_usage = dict(msg_usage)
        except Exception as exc:
            # SDK ≥ 0.2.82 / CLI ≥ 2.1.123 may raise "error result: success"
            # on max-turns exit; keep any intents already collected.
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
        # Prefer the consolidated ResultMessage text; fall back to TextBlocks.
        raw_text = "".join(result_chunks) or "".join(text_chunks)
        return intents, raw_text, tool_block_count, last_usage, session_id

    @staticmethod
    def _iter_blocks(message: Any):
        """Return the content blocks of an SDK message as a list.

        Args:
            message (Any): An SDK message that may carry a ``content`` list.

        Returns:
            list: The message's content blocks, or an empty list when absent.
        """
        return list(getattr(message, "content", None) or [])

    def _is_tool_use_for_emit_intent(self, block: Any) -> bool:
        """Whether a content block is an ``emit_intent`` tool-use call.

        Matches by class name (``ToolUseBlock`` / ``ServerToolUseBlock``) to
        avoid depending on SDK internals, then checks the tool name.

        Args:
            block (Any): A single SDK content block.

        Returns:
            bool: ``True`` if the block is a tool-use for the (qualified or
            bare) ``emit_intent`` tool.
        """
        # Match by class name so we don't depend on SDK internals.
        cls_name = type(block).__name__
        if cls_name not in ("ToolUseBlock", "ServerToolUseBlock"):
            return False
        name = getattr(block, "name", "")
        return name in (EMIT_INTENT_TOOL_NAME, EMIT_INTENT_TOOL_QUALIFIED)

    def _parse_tool_use_block(self, block: Any) -> Intent | None:
        """Validate one ``emit_intent`` tool-use block into an :class:`Intent`.

        Args:
            block (Any): A tool-use block whose ``input`` carries
                ``intent_type`` and ``payload``.

        Returns:
            Intent | None: The validated intent, or ``None`` if validation
            fails (the failure is logged, not raised).
        """
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
        """Extract plain text from a content block across block shapes.

        Handles SDK ``TextBlock`` objects, ``{"type": "text"}`` dicts, and any
        object exposing a string ``text`` attribute.

        Args:
            block (Any): A single content block.

        Returns:
            str: The block's text, or an empty string when none is present.
        """
        if type(block).__name__ == "TextBlock":
            return getattr(block, "text", "") or ""
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "") or ""
        t = getattr(block, "text", None)
        return t if isinstance(t, str) else ""


__all__ = ["ClaudeBackend"]
