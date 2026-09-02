# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ClaudeBackend — uses ``claude-agent-sdk`` to drive Claude.

An in-process MCP server registers ``emit_intent`` so each
``ToolUseBlock`` becomes one validated :class:`Intent`. SDK import is lazy
(clear :class:`BackendError` if missing); ``sdk_query_factory`` /
``sdk_options_cls`` / ``mcp_*_factory`` are test seams that bypass the real
SDK + network.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

from hyperloom.common.llm_config import claude_sdk_env_options
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from ..prompts.transport import TRANSPORT_TOOLS
from ..trace.llm_trace import new_call_id
from .base import (
    BackendError,
    BackendTurnResult,
    LLMCallFailed,
    RetryPolicy,
    parse_call_timeout_env,
    retry_with_backoff,
    safe_int,
)
from .mcp_context_tools import (
    CONTEXT_TOOL_QUALIFIED_NAMES,
    MCP_SERVER_NAME as CONTEXT_MCP_SERVER_NAME,
    ContextProvider,
    build_context_tools_server,
)
from hyperloom.inference_optimizer.protocol.intent import IntentType
from .mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    EMIT_INTENT_TOOL_INPUT_SCHEMA,
    MCP_SERVER_NAME,
    build_emit_intent_server,
    constraints_sentence,
    payload_contract,
)


log = logging.getLogger(__name__)


def _input_side_total(usage: dict[str, Any]) -> int:
    """Total input-side tokens (fresh + cache read + cache creation) of a usage dict.

    Args:
        usage (dict[str, Any]): An Anthropic-shaped usage dict.

    Returns:
        int: The summed input side; ``0`` when no counters are present.
    """
    return (
        safe_int(usage.get("input_tokens"))
        + safe_int(usage.get("cache_read_input_tokens"))
        + safe_int(usage.get("cache_creation_input_tokens"))
    )


def _context_tokens_estimate(usage: dict[str, Any], *, num_turns: int) -> int:
    """Mean per-request context size implied by call-cumulative usage.

    Args:
        usage (dict[str, Any]): The call-cumulative usage dict.
        num_turns (int): Internal turns the call took; ``<= 1`` means the sum
            already describes a single request.

    Returns:
        int: The per-request estimate, or ``0`` when usage carries no counters.
    """
    total = _input_side_total(usage)
    return total // num_turns if num_turns > 1 else total


def _build_output_instructions(allowed_intents: frozenset[IntentType]) -> str:
    """Render the output-format suffix for a role's allowed intent set."""
    import json as _json

    contract = payload_contract(allowed_intents)
    constraints = constraints_sentence(allowed_intents)
    constraints_line = f"\n-{constraints}" if constraints else ""
    heartbeat = _json.dumps({"topic": "heartbeat", "body_md": "ok"})
    return f"""
==== OUTPUT FORMAT (REQUIRED) ====
You MUST communicate with the system by calling the `{EMIT_INTENT_TOOL_NAME}`
tool. Each call carries exactly one intent; call multiple times to emit
several intents in the same turn. Free-text replies are dropped.

Tool input shape:

  {{
    "intent_type": "<one of {", ".join(sorted(t.value for t in allowed_intents))}>",
    "payload": {{ /* per-intent fields — see tool description */ }}
  }}

- Required keys per intent_type: {contract}.{constraints_line}
- If you have nothing to say, call once with intent_type=send_message and
  payload={heartbeat}.

Keep payload bodies focused on NEW information. Do not restate context already
in SharedState, your inbox, or analysis.md — reference it and summarize only
what changed. Match length to substance.
==== END OUTPUT FORMAT ====
""".strip()


# Conversational-mode floors: a persistent ReAct turn pulls context tools before
# emitting, so it needs more turns + wall-clock budget.
_CONVERSATIONAL_MIN_MAX_TURNS: int = 12
_CONVERSATIONAL_DEFAULT_TIMEOUT_SEC: float = 300.0
# Global floor, applied by ``run()`` to every mode: Claude Code counts the
# model's own text message as a turn, so a low max_turns trips before any output.
_RAW_COMPLETION_MIN_MAX_TURNS: int = 8

# Retried timeouts get a progressively larger idle budget so a genuinely slow
# gateway is not re-killed at the same wall.
_RETRY_IDLE_TIMEOUT_MULTIPLIER: float = 2.0

# Claude Code stores a tool JSON string the streaming parser did not promote
# to an object. Native Claude input already has ``intent_type`` at the top
# level and is left unchanged.
_UNPARSED_TOOL_INPUT_KEY = "__unparsedToolInput"


def _is_unparsed_tool_wrapper(raw_input: Any) -> bool:
    """Whether ``input`` is Claude Code's wrapped unparsed tool JSON object."""
    if not isinstance(raw_input, dict):
        return False
    if "intent_type" in raw_input:
        return False
    wrapped = raw_input.get(_UNPARSED_TOOL_INPUT_KEY)
    return isinstance(wrapped, dict) and isinstance(wrapped.get("raw"), str)


def _coerce_emit_intent_input(raw_input: Any) -> dict[str, Any]:
    """Return native emit_intent input, decoding Claude Code's wrapper when needed.

    A dict that already carries ``intent_type`` is returned as-is. Only the
    ``__unparsedToolInput.raw`` shape is decoded; malformed JSON leaves the
    original dict so existing validation still fails.
    """
    if not isinstance(raw_input, dict):
        return {}
    if "intent_type" in raw_input:
        return raw_input
    wrapped = raw_input.get(_UNPARSED_TOOL_INPUT_KEY)
    if not isinstance(wrapped, dict):
        return raw_input
    raw = wrapped.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return raw_input
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw_input
    return parsed if isinstance(parsed, dict) else raw_input


def _intent_fingerprint(intent: Intent) -> str:
    """Stable key for one validated intent used to drop fallback retries."""
    return json.dumps(
        {"intent_type": intent.type.value, "payload": intent.payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


# Built-in tools disallowed in raw_completion mode so the model produces
# exactly one text turn (no agentic tool loop).
_RAW_COMPLETION_DISALLOWED_TOOLS: tuple[str, ...] = (
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Write",
    "Edit",
    "NotebookEdit",
    "Glob",
    "Grep",
    "Task",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "ExitPlanMode",
    "SlashCommand",
)

# Env-driven reasoning effort / extended thinking. A per-role override wins over
# the shared override; kernel defaults to ``low`` and orchestration to ``medium``.
_EFFORT_ENV: str = "INFERENCE_OPTIMIZER_CLAUDE_EFFORT"
_EFFORT_ENV_ORCH: str = "INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT"
_EFFORT_ENV_KERNEL: str = "INFERENCE_OPTIMIZER_CLAUDE_KERNEL_EFFORT"
_THINKING_ENV: str = "INFERENCE_OPTIMIZER_CLAUDE_THINKING"
_CLI_PATH_ENV: str = "HYPERLOOM_CLAUDE_CLI_PATH"
_VALID_EFFORT: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})


def _import_sdk() -> tuple[Any, Any, Any]:
    """Return ``(query, ClaudeAgentOptions, sdk_module)`` or raise.

    Returns:
        A tuple of the SDK ``query`` callable, the ``ClaudeAgentOptions``
        class, and the imported SDK module.

    Raises:
        BackendError: If ``claude_agent_sdk`` is not installed or is missing
            the required ``query`` / ``ClaudeAgentOptions`` attributes.
    """
    try:
        sdk = importlib.import_module("claude_agent_sdk")
    except ImportError as exc:
        raise BackendError("claude-agent-sdk not installed; run `pip install claude-agent-sdk` (>= 0.1.65).") from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise BackendError("claude_agent_sdk loaded but missing query / ClaudeAgentOptions")
    return sdk.query, sdk.ClaudeAgentOptions, sdk


@dataclass
class ClaudeBackend:
    """Production Claude backend. Implements :class:`Backend`.

    Args:
        model: Claude model id; defaults to ``ANTHROPIC_MODEL`` env or library default.
        api_key_env: Env var checked at construction (``ANTHROPIC_API_KEY`` by default).
        max_turns_default: Agent-loop budget when caller doesn't override; floored at 8.
        enable_mcp_emit_intent: Registers the in-process MCP ``emit_intent`` tool.
        capture_turn_diagnostics: Capture full turn diagnostics for orchestration tracing.
        allowed_intents: Role's permitted intent set; drives the output-format suffix.
    """

    model: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    # Nominal budget only: run() floors every mode at _RAW_COMPLETION_MIN_MAX_TURNS
    # (8), so values below 8 have no effect; conversational mode raises it to 12.
    max_turns_default: int = 4
    # Persistent-conversation mode: resume the SAME SDK session across ticks,
    # feeding only a per-tick delta. kernel / critic / robustness stay stateless.
    conversational: bool = False
    enable_mcp_emit_intent: bool = True
    capture_turn_diagnostics: bool = False
    # Raw single-shot completion mode: skips the emit_intent server + suffix,
    # disallows all tools, and returns ``raw_text`` without an emitted intent.
    raw_completion: bool = False
    # Role's allowed intent set for the output-format suffix. None = all IntentType values.
    allowed_intents: frozenset[IntentType] | None = None
    # Attribution labels for the spend this backend's turns produce. One class
    # serves several roles -- ``executors`` reuses it for in-process specialists
    # -- so a fixed label would file their spend under orchestration and leave
    # the rollup unable to tell the two apart. Defaults keep the orchestrator's.
    attribution_component: str = "orchestration"
    attribution_operation: str = "orchestrate_turn"
    # Idle timeout for one ``run()`` call: max wall-clock gap allowed BETWEEN
    # streamed SDK messages before the turn is aborted. Env override:
    # ``INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC``.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC",
            default=120.0,
        )
    )
    # Bounded transient-failure retry/backoff. Env override via
    # INFERENCE_OPTIMIZER_LLM_RETRY_* (ATTEMPTS=1 disables).
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.from_env)

    # Test seams — set these to bypass SDK import / network calls.
    sdk_query_factory: Callable[..., Any] | None = None
    sdk_options_cls: Any | None = None
    sdk_module: Any | None = None
    mcp_server_factory: Callable[..., Any] | None = None
    mcp_tool_factory: Callable[..., Any] | None = None

    name: str = "claude"
    # Which prompt modules describe a surface this backend actually has. Read
    # by the prompt builder, which cannot infer it from the role.
    transport = TRANSPORT_TOOLS
    calls: list[dict[str, Any]] = field(default_factory=list)
    mcp_server_config: Any | None = field(default=None, init=False)
    mcp_tool_name: str | None = field(default=None, init=False)
    # SDK session token captured last turn; replayed via ``resume`` in
    # conversational mode. ``reset_conversation()`` clears it.
    _session_id: str | None = field(default=None, init=False)
    # Read-only context-pull MCP server config, set via
    # ``set_context_provider`` and merged into the SDK options.
    _context_server_config: Any | None = field(default=None, init=False)
    _mcp_setup_error: str | None = field(default=None, init=False)
    _active_turn_diagnostic: dict[str, Any] | None = field(default=None, init=False)
    _last_turn_diagnostic: dict[str, Any] = field(default_factory=dict, init=False)
    _active_stderr: list[str] = field(default_factory=list, init=False)

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
            # Persistent ReAct turns need more turns + a longer idle budget than
            # the stateless default; operator env override still wins.
            if self.max_turns_default < _CONVERSATIONAL_MIN_MAX_TURNS:
                self.max_turns_default = _CONVERSATIONAL_MIN_MAX_TURNS
            if (
                os.environ.get(
                    "INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC",
                    "",
                ).strip()
                == ""
            ):
                # No operator override -> raise the idle-timeout floor.
                self.call_timeout_s = max(
                    self.call_timeout_s,
                    _CONVERSATIONAL_DEFAULT_TIMEOUT_SEC,
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
                self._mcp_setup_error = f"{type(exc).__name__}: {exc}"
                cfg = None
            if cfg is not None:
                self.mcp_server_config = cfg
                self.mcp_tool_name = EMIT_INTENT_TOOL_QUALIFIED

    # ------------------------------------------------------------------
    # Backend protocol
    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        max_turns: int = 1,
        allow_no_intent: bool = False,
    ) -> BackendTurnResult:
        """Run a single backend turn against Claude and parse the result.

        Args:
            prompt: User prompt for this turn.
            system_prompt: Optional system prompt override.
            tools: Tool names to enable for the turn.
            disallowed_tools: Tool names to remove from the available set.
                Applied additively on top of any ``raw_completion`` denylist.
            max_turns: Maximum agent turns; falls back to the backend
                default when falsy.
            allow_no_intent: When ``True``, relax the guard that requires
                an ``emit_intent`` (e.g. for summary/checkpoint turns).

        Returns:
            The parsed :class:`BackendTurnResult` for the turn.
        """
        full_prompt = self._compose_prompt(prompt)
        self._begin_turn_diagnostic(
            prompt=full_prompt,
            system_prompt=system_prompt,
            tools=tools or [],
        )
        max_turns_use = max_turns or self.max_turns_default
        # Claude Code counts the model's own text/tool messages as turns, so a
        # literal max_turns=1 trips ("Reached maximum number of turns (1)")
        # before the model can emit any tool call or intent — newer bundled CLI
        # builds raise this as an error rather than returning a partial result.
        # Callers that pass max_turns=1 to mean "one agentic step" (e.g. the
        # specialist runner's per-turn loop) therefore need headroom. Apply the
        # raw-completion floor to every mode, not just raw_completion.
        max_turns_use = max(max_turns_use, _RAW_COMPLETION_MIN_MAX_TURNS)
        resume_session = self._session_id if self.conversational else None
        try:
            options = self._build_options(
                tools=tools or [],
                disallowed_tools=disallowed_tools or [],
                max_turns=max_turns_use,
                system_prompt=system_prompt,
                resume_session_id=resume_session,
            )
        except BaseException as exc:
            self._finish_turn_diagnostic(outcome="backend_error", error=exc)
            raise
        self._update_turn_options(
            options,
            max_turns=max_turns_use,
            resume_session=resume_session,
        )

        # Each attempt bounds the gap BETWEEN streamed SDK messages (silence
        # budget), not the total turn; each retry amplifies the idle budget.
        attempt_state = {"n": 0}

        async def _one_attempt() -> tuple[Any, ...]:
            """Run one SDK invocation under an amplified per-attempt idle timeout.

            Returns:
                The collected ``_invoke_and_collect`` result tuple.

            Raises:
                asyncio.TimeoutError: If the stream stays idle (no new message)
                    for longer than this attempt's amplified idle budget.
            """
            attempt_state["n"] += 1
            idle_timeout_s = self.call_timeout_s * (_RETRY_IDLE_TIMEOUT_MULTIPLIER ** (attempt_state["n"] - 1))
            return await self._invoke_and_collect(full_prompt, options, idle_timeout_s=idle_timeout_s)

        def _note_retry(attempt: int, exc: BaseException, delay: float) -> None:
            """Record a transient-failure retry warning into the call log.

            Args:
                attempt: The 1-based attempt number that failed.
                exc: The transient exception raised by the attempt.
                delay: Seconds to wait before the next retry.
            """
            self.calls.append(
                {
                    "warn": (f"claude SDK transient failure (attempt {attempt}): {exc!r}; retrying in {delay:.2f}s"),
                }
            )

        try:
            (
                intents,
                raw_text,
                tool_block_count,
                usage,
                session_id,
                stop_reason,
            ) = await retry_with_backoff(
                _one_attempt,
                policy=self.retry_policy,
                retry_on=(
                    asyncio.TimeoutError,
                    BackendError,
                    ConnectionError,
                    OSError,
                ),
                on_retry=_note_retry,
            )
        except asyncio.TimeoutError as exc:
            self.calls.append(
                {
                    "warn": (
                        f"claude SDK stream idle / timed out (no new message for "
                        f">{self.call_timeout_s:.0f}s, retries exhausted); treating "
                        "as no-intent so the reactor pass can proceed"
                    ),
                }
            )
            error = LLMCallFailed(
                f"Claude backend timed out: stream idle for >{self.call_timeout_s:.0f}s (likely upstream proxy stall)"
            )
            self._finish_turn_diagnostic(outcome="backend_error", error=error)
            raise error from exc
        except BaseException as exc:
            self._finish_turn_diagnostic(outcome="backend_error", error=exc)
            # Once retries are exhausted, anything the SDK stream raised is a
            # provider call that produced nothing usable — including the gateway
            # 400s (``litellm.BadRequestError: AnthropicException``) this
            # telemetry exists to count. Marking it here matches what Codex
            # already does for its own API errors, and keeps the failure out of
            # the Coordinator's "unexpected crash" path. ``NoIntentEmitted`` is
            # raised below, outside this block, so a call that succeeded with
            # unusable output stays distinct. Cancellation is a BaseException
            # and is re-raised untouched.
            if isinstance(exc, Exception) and not isinstance(exc, LLMCallFailed):
                raise LLMCallFailed(f"Claude backend call failed: {exc!r}") from exc
            raise
        if self._active_turn_diagnostic is not None:
            self._active_turn_diagnostic["session_id_hash"] = self._session_hash(session_id)
            self._active_turn_diagnostic["new_session"] = bool(session_id and not resume_session)
        # Capture the SDK session token for the next conversational resume;
        # only overwrite on a non-empty id.
        if self.conversational:
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
        cache_creation = safe_int(usage.get("cache_creation_input_tokens") if usage else None)
        cache_read = safe_int(usage.get("cache_read_input_tokens") if usage else None)
        input_tokens = safe_int(usage.get("input_tokens") if usage else None)
        output_tokens = safe_int(usage.get("output_tokens") if usage else None)
        if self._active_turn_diagnostic is not None:
            self._active_turn_diagnostic["usage"] = {
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        self.calls.append(
            {
                "prompt_chars": len(full_prompt),
                "tool_blocks": tool_block_count,
                "intents": len(intents),
                "max_turns": max_turns_use,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )
        if not intents and not self.raw_completion and not allow_no_intent:
            error = NoIntentEmitted(
                f"claude reply contained no parseable emit_intent tool_use "
                f"blocks (raw_text_len={len(raw_text)}, tool_blocks={tool_block_count})"
            )
            self._finish_turn_diagnostic(outcome="no_intent", error=error)
            raise error
        self._finish_turn_diagnostic(outcome="succeeded")
        return BackendTurnResult(
            intents=intents,
            raw_text=raw_text,
            metadata={
                "tool_blocks": tool_block_count,
                "model": self.model,
                # Pairs this turn's token row with its conversation row; both
                # halves are written from this one metadata dict.
                "call_id": new_call_id(),
                # Why the model stopped ("end_turn" / "max_tokens" / ...).
                # Without it a truncated reply looks like a malformed one.
                "stop_reason": stop_reason,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                # Per-request context size; the counters above sum the call's
                # internal turns and are spend, not size.
                "context_tokens_peak": safe_int(usage.get("context_tokens_peak") if usage else None),
                # Full conversation text so the caller (which holds the
                # session_dir / component / tick context the stateless
                # backend lacks) can persist it to conversations.jsonl.
                # The composed prompt carries the user turn; the system
                # prompt is snapshotted per scope under agents/<role>/.
                "prompt": full_prompt,
                "response": raw_text,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compose_prompt(self, prompt: str) -> str:
        """Append the emit_intent output-format suffix unless in raw mode.

        The suffix is rendered from :attr:`allowed_intents` so only the
        intents this role may emit are listed.

        Args:
            prompt (str): The base turn prompt.

        Returns:
            str: The prompt unchanged in ``raw_completion`` mode, otherwise the
            prompt with the required output-format instructions appended.
        """
        if self.raw_completion:
            return prompt
        intents = self.allowed_intents if self.allowed_intents is not None else frozenset(IntentType)
        return f"{prompt}\n\n{_build_output_instructions(intents)}"

    def set_context_provider(self, provider: ContextProvider | None) -> None:
        """Attach (or clear) the read-only context-pull MCP server.

        ``None`` detaches it. Best-effort: a build failure is recorded as a
        soft warning and leaves the backend usable without the pull tools.

        Args:
            provider: The context provider to back the pull tools, or ``None``
                to detach the context-tools server.
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
    def context_tools_mounted(self) -> bool:
        """Whether the read-only context-pull tools are live this turn.

        ``set_context_provider`` degrades to a soft warning when the MCP build
        fails, so a caller must not infer availability from having called it.
        """
        return self._context_server_config is not None

    def reset_conversation(self) -> None:
        """Drop the captured session so the next ``run`` starts fresh.

        Used after a checkpoint/compaction or resume rebuild.
        """
        self._session_id = None

    def get_turn_diagnostic(self) -> dict[str, Any]:
        """Return the most recently completed turn diagnostic."""
        return dict(self._last_turn_diagnostic)

    def get_mcp_setup_diagnostic(self) -> dict[str, Any]:
        """Return the current MCP setup snapshot."""
        schema = json.dumps(EMIT_INTENT_TOOL_INPUT_SCHEMA, sort_keys=True, separators=(",", ":"))
        diag = self.get_turn_diagnostic()
        return {
            "backend": type(self).__name__,
            "model": self.model,
            "sdk_name": getattr(self.sdk_module, "__name__", None),
            "sdk_version": getattr(self.sdk_module, "__version__", None),
            "cli_version": os.environ.get("CLAUDE_CODE_VERSION") or None,
            "gateway_endpoint": self._gateway_endpoint_identifier(),
            "mcp_servers": diag.get("mcp_servers", []),
            "emit_intent": {
                "qualified_name": EMIT_INTENT_TOOL_QUALIFIED,
                "registered": bool(self.mcp_server_config is not None and self.mcp_tool_name),
                "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
                "setup_error": self._mcp_setup_error,
            },
            "allowed_tools": diag.get("allowed_tools", []),
        }

    def _begin_turn_diagnostic(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        tools: list[str],
    ) -> None:
        if not self.capture_turn_diagnostics:
            return
        previous_session = self._session_id if self.conversational else None
        self._active_stderr = []
        self._active_turn_diagnostic = {
            "backend": type(self).__name__,
            "model": self.model,
            "sdk_name": getattr(self.sdk_module, "__name__", None),
            "sdk_version": getattr(self.sdk_module, "__version__", None),
            "cli_version": os.environ.get("CLAUDE_CODE_VERSION") or None,
            "gateway_endpoint": self._gateway_endpoint_identifier(),
            "resume_requested": bool(previous_session),
            "previous_session_id_hash": self._session_hash(previous_session),
            "session_id_hash": None,
            "new_session": None,
            "max_turns": None,
            "timeout_sec": self.call_timeout_s,
            "reasoning_effort": None,
            "thinking": None,
            "prompt": prompt,
            "system_prompt": system_prompt or "",
            "allowed_tools": list(tools),
            "mcp_servers": [],
            "emit_intent_registered": bool(self.mcp_server_config is not None and self.mcp_tool_name),
            "messages": [],
            "result": "",
            "raw_text": "",
            "tool_blocks": [],
            "parse_errors": [],
            "usage": {},
            "stderr_tail": [],
        }

    def _update_turn_options(self, options: Any, *, max_turns: int, resume_session: str | None) -> None:
        diag = self._active_turn_diagnostic
        if diag is None:
            return
        kwargs = getattr(options, "kwargs", None)
        if not isinstance(kwargs, dict):
            kwargs = {}
        allowed = kwargs.get("allowed_tools", getattr(options, "allowed_tools", diag["allowed_tools"]))
        servers = kwargs.get("mcp_servers", getattr(options, "mcp_servers", {}))
        if not kwargs:
            if self.raw_completion:
                allowed = []
                servers = {}
            else:
                allowed = [tool for tool in diag["allowed_tools"] if tool != EMIT_INTENT_TOOL_NAME]
                if self.mcp_tool_name and self.mcp_tool_name not in allowed:
                    allowed.append(self.mcp_tool_name)
                if self._context_server_config is not None:
                    allowed.extend(tool for tool in CONTEXT_TOOL_QUALIFIED_NAMES if tool not in allowed)
                servers = {}
                if self.mcp_server_config is not None:
                    servers[MCP_SERVER_NAME] = self.mcp_server_config
                if self._context_server_config is not None:
                    servers[CONTEXT_MCP_SERVER_NAME] = self._context_server_config
        diag["max_turns"] = max_turns
        diag["resume_requested"] = bool(resume_session)
        diag["allowed_tools"] = [str(tool) for tool in allowed or []]
        diag["mcp_servers"] = sorted(str(name) for name in (servers or {}))
        role_env = _EFFORT_ENV_ORCH if self.conversational else _EFFORT_ENV_KERNEL
        default_effort = "medium" if self.conversational else "low"
        diag["reasoning_effort"] = kwargs.get(
            "effort",
            getattr(
                options, "effort", (os.environ.get(role_env) or os.environ.get(_EFFORT_ENV) or default_effort).strip()
            ),
        )
        diag["thinking"] = kwargs.get(
            "thinking",
            getattr(options, "thinking", {"type": (os.environ.get(_THINKING_ENV) or "adaptive").strip().lower()}),
        )

    def _finish_turn_diagnostic(self, *, outcome: str, error: BaseException | None = None) -> None:
        diag = self._active_turn_diagnostic
        if diag is None:
            return
        diag["outcome"] = outcome
        diag["stderr_tail"] = self._active_stderr[-50:]
        if error is not None:
            diag["error_type"] = type(error).__name__
            diag["error_message"] = str(error)
        self._last_turn_diagnostic = diag
        self._active_turn_diagnostic = None

    def _gateway_endpoint_identifier(self) -> str | None:
        raw = (os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not raw:
            return None
        parts = urlsplit(raw)
        # hostname, not netloc: netloc carries any ``user:secret@`` userinfo.
        return parts.hostname or "configured"

    @staticmethod
    def _session_hash(session_id: str | None) -> str | None:
        if not session_id:
            return None
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def _build_options(
        self,
        *,
        tools: list[str],
        disallowed_tools: list[str] | None = None,
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
            disallowed_tools (list[str] | None): Additional tool names to deny.
                Merged with the ``raw_completion`` denylist when both apply.
            max_turns (int): Agent-loop budget to pass through to the SDK.
            system_prompt (str | None): Optional system prompt for the turn.
            resume_session_id (str | None): SDK session token to resume; set
                by the caller only in conversational mode, ``None`` starts a
                fresh session.

        Returns:
            Any: A constructed ``ClaudeAgentOptions`` instance.
        """
        kwargs: dict[str, Any] = {"max_turns": max_turns}
        if self.model:
            kwargs["model"] = self.model
        cli_path = os.environ.get(_CLI_PATH_ENV, "").strip()
        if cli_path:
            kwargs["cli_path"] = cli_path
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if resume_session_id:
            kwargs["resume"] = resume_session_id
        self._apply_sdk_env_options(kwargs)
        self._apply_effort_options(kwargs)
        if self.raw_completion:
            # Single text turn: no MCP tools, all built-ins disallowed.
            kwargs["allowed_tools"] = []
            deny = list(_RAW_COMPLETION_DISALLOWED_TOOLS)
            if disallowed_tools:
                deny = list(dict.fromkeys(deny + disallowed_tools))
            kwargs["disallowed_tools"] = deny
            kwargs["stderr"] = self._stderr_sink
            return self._instantiate_options(kwargs)
        # Drop the bare "emit_intent" name; the MCP-qualified form wires into the
        # SDK tool registry.
        allowed = [t for t in tools if t != EMIT_INTENT_TOOL_NAME]
        if self.mcp_tool_name and self.mcp_tool_name not in allowed:
            allowed.append(self.mcp_tool_name)
        # Allow-list the context-pull tools' qualified names.
        if self._context_server_config is not None:
            for qname in CONTEXT_TOOL_QUALIFIED_NAMES:
                if qname not in allowed:
                    allowed.append(qname)
        if allowed:
            kwargs["allowed_tools"] = allowed
        if disallowed_tools:
            kwargs["disallowed_tools"] = disallowed_tools
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

    def _apply_sdk_env_options(self, kwargs: dict[str, Any]) -> None:
        """Pin Claude Code subprocess auth to the current Hyperloom env.

        When a Hyperloom process has explicit Anthropic/gateway env, pass that
        env to the SDK subprocess and disable settings sources so the run is
        hermetic (Claude Code otherwise reads ``~/.claude`` config).
        """
        kwargs.update(
            claude_sdk_env_options(
                model=self.model,
                component=self.attribution_component,
                operation=self.attribution_operation,
            )
        )

    def _apply_effort_options(self, kwargs: dict[str, Any]) -> None:
        """Add env-driven reasoning effort + adaptive thinking to the options.

        Role is inferred from ``conversational`` (True => orchestration).
        """
        role_env = _EFFORT_ENV_ORCH if self.conversational else _EFFORT_ENV_KERNEL
        default = "medium" if self.conversational else "low"
        effort = (os.environ.get(role_env) or os.environ.get(_EFFORT_ENV) or default).strip().lower()
        if effort in _VALID_EFFORT:
            kwargs["effort"] = effort
        thinking = (os.environ.get(_THINKING_ENV) or "adaptive").strip().lower()
        if thinking and thinking != "off":
            kwargs["thinking"] = {"type": thinking}

    def _instantiate_options(self, kwargs: dict[str, Any]) -> Any:
        """Build the SDK options.

        Args:
            kwargs: Keyword arguments to pass to the SDK options constructor.

        Returns:
            A constructed SDK options instance.
        """
        return self.sdk_options_cls(**kwargs)

    def _stderr_sink(self, line: str) -> None:
        """Default stderr handler — append to ``self.calls`` for postmortems.

        Args:
            line (str): One line of CLI stderr output; blank lines are dropped.
        """
        text = line.strip()
        if text:
            self.calls.append({"stderr": text})
            if self._active_turn_diagnostic is not None:
                self._active_stderr.append(text)

    async def _invoke_and_collect(
        self, prompt: str, options: Any, *, idle_timeout_s: float | None = None
    ) -> tuple[list[Intent], str, int, dict[str, Any], str | None, str | None]:
        """Stream SDK messages, collecting intents, raw text, tool counts,
        the latest `ResultMessage.usage` dict, the SDK ``session_id`` and the
        model's ``stop_reason``.

        Args:
            prompt: The composed prompt to stream to the SDK.
            options: The SDK options object configuring this turn.
            idle_timeout_s: When set, the max wall-clock allowed to wait for the
                *next* streamed message before raising ``asyncio.TimeoutError``.
                Bounds a stalled gateway without capping total turn time, so a
                slow-but-live reasoning model is never killed (issue #679).

        Returns:
            A tuple ``(intents, raw_text, tool_block_count, usage, session_id,
            stop_reason)`` where ``usage`` is the latest cumulative usage dict
            plus a per-request ``context_tokens_peak``, ``session_id`` is the
            SDK session token (or ``None``), and ``stop_reason`` is the last
            reason the model reported (or ``None``).
        """
        intents: list[Intent] = []
        text_chunks: list[str] = []
        result_chunks: list[str] = []
        tool_block_count = 0
        # Claude Code may retry the same wrapped tool JSON many times in one
        # bounded turn; keep the first decoded fallback envelope only.
        seen_fallback_intents: set[str] = set()
        # Every usage dict the stream reports, in order: the last is cumulative
        # over the call, the ones before it describe single requests.
        usages: list[dict[str, Any]] = []
        num_turns = 0
        session_id: str | None = None
        stop_reason: str | None = None
        stream = self.sdk_query_factory(prompt=prompt, options=options)
        try:
            stream_iter = stream.__aiter__()
            while True:
                # Idle timeout: bound only the wait for the NEXT message; a
                # fully silent gateway trips ``asyncio.TimeoutError``.
                try:
                    if idle_timeout_s is not None:
                        message = await asyncio.wait_for(stream_iter.__anext__(), timeout=idle_timeout_s)
                    else:
                        message = await stream_iter.__anext__()
                except StopAsyncIteration:
                    break
                # Capture the session token from any message; last seen wins.
                msg_session = getattr(message, "session_id", None)
                if isinstance(msg_session, str) and msg_session:
                    session_id = msg_session
                self._record_message_diagnostic(message)
                for block in self._iter_blocks(message):
                    self._record_tool_block_diagnostic(block)
                    if self._is_tool_use_for_emit_intent(block):
                        tool_block_count += 1
                        intent = self._parse_tool_use_block(block)
                        if intent is not None:
                            if _is_unparsed_tool_wrapper(getattr(block, "input", None)):
                                fingerprint = _intent_fingerprint(intent)
                                if fingerprint in seen_fallback_intents:
                                    continue
                                seen_fallback_intents.add(fingerprint)
                            intents.append(intent)
                    else:
                        txt = self._extract_text(block)
                        if txt:
                            text_chunks.append(txt)
                # ResultMessage.result duplicates the streamed TextBlocks; keep
                # it separate to avoid double-counting.
                result_text = getattr(message, "result", None)
                if isinstance(result_text, str) and result_text:
                    result_chunks.append(result_text)
                msg_turns = getattr(message, "num_turns", None)
                if isinstance(msg_turns, int) and msg_turns > 0:
                    num_turns = msg_turns
                # Both AssistantMessage and the terminal ResultMessage carry it;
                # last seen wins, so the call's own reason ends up reported.
                msg_stop = getattr(message, "stop_reason", None)
                if isinstance(msg_stop, str) and msg_stop:
                    stop_reason = msg_stop
                msg_usage = getattr(message, "usage", None)
                if isinstance(msg_usage, dict) and msg_usage:
                    usages.append(dict(msg_usage))
        except Exception as exc:
            # The SDK raises on a terminal ResultMessage with is_error=True. Two
            # such subtypes are NON-fatal turn boundaries, not real failures, and
            # any intents/text already streamed must be kept:
            #   * "error result: success"                    (older max-turns-on-success)
            #   * "Reached maximum number of turns (N)"       (newer CLI: hitting the
            #     per-call max_turns cap mid-agentic-loop — expected when a caller
            #     runs a bounded step; the collected partial turn is still usable)
            err_str = str(exc)
            _non_fatal = "error result: success" in err_str or "maximum number of turns" in err_str
            if _non_fatal:
                if self._active_turn_diagnostic is not None:
                    self._active_turn_diagnostic["sdk_boundary_error"] = err_str
                if intents:
                    log.warning(
                        "claude SDK raised '%s' but %d intents already collected; returning partial results",
                        err_str,
                        len(intents),
                    )
                else:
                    log.warning(
                        "claude SDK raised '%s' with no intents collected; "
                        "treating as no-intent turn (will retry next tick)",
                        err_str,
                    )
            else:
                raise
        finally:
            # Best-effort: close the (async-gen) stream so an idle-timeout abort
            # doesn't leak a half-consumed generator. Errors here are never fatal.
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 — cleanup must not mask the turn result
                    pass
        # Prefer the consolidated ResultMessage text; fall back to TextBlocks.
        raw_text = "".join(result_chunks) or "".join(text_chunks)
        if self._active_turn_diagnostic is not None:
            self._active_turn_diagnostic["result"] = "".join(result_chunks)
            self._active_turn_diagnostic["raw_text"] = raw_text
        last_usage: dict[str, Any] = dict(usages[-1]) if usages else {}
        if last_usage:
            peak = max((_input_side_total(u) for u in usages[:-1]), default=0)
            last_usage["context_tokens_peak"] = peak or _context_tokens_estimate(
                last_usage,
                num_turns=num_turns,
            )
        return intents, raw_text, tool_block_count, last_usage, session_id, stop_reason

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

        Matches by class name, then checks the tool name.

        Args:
            block (Any): A single SDK content block.

        Returns:
            bool: ``True`` if the block is a tool-use for the (qualified or
            bare) ``emit_intent`` tool.
        """
        cls_name = type(block).__name__
        if cls_name not in ("ToolUseBlock", "ServerToolUseBlock"):
            return False
        name = getattr(block, "name", "")
        return name in (EMIT_INTENT_TOOL_NAME, EMIT_INTENT_TOOL_QUALIFIED)

    def _parse_tool_use_block(self, block: Any) -> Intent | None:
        """Validate one ``emit_intent`` tool-use block into an :class:`Intent`.

        Args:
            block (Any): A tool-use block whose ``input`` carries
                ``intent_type`` and ``payload``, or Claude Code's
                ``__unparsedToolInput.raw`` wrapper around that JSON.

        Returns:
            Intent | None: The validated intent, or ``None`` if validation
            fails (the failure is logged, not raised).
        """
        raw_input = _coerce_emit_intent_input(getattr(block, "input", None) or {})
        try:
            envelope = {
                "intents": [
                    {
                        "intent_type": raw_input.get("intent_type"),
                        "payload": raw_input.get("payload") or {},
                    }
                ]
            }
            validated = validate_envelope(envelope)
        except IntentValidationError as exc:
            log.info("claude tool_use validation failed: %s", exc)
            if self._active_turn_diagnostic is not None:
                self._active_turn_diagnostic["parse_errors"].append(str(exc))
            return None
        return validated[0] if validated else None

    def _record_message_diagnostic(self, message: Any) -> None:
        diag = self._active_turn_diagnostic
        if diag is None:
            return
        summary: dict[str, Any] = {"type": type(message).__name__}
        for name in ("is_error", "subtype", "request_id"):
            value = getattr(message, name, None)
            if value is not None:
                summary[name] = value
                if name == "request_id" and not diag.get("request_id"):
                    diag["request_id"] = str(value)
        result = getattr(message, "result", None)
        if isinstance(result, str):
            summary["result"] = result
        diag["messages"].append(summary)

    def _record_tool_block_diagnostic(self, block: Any) -> None:
        diag = self._active_turn_diagnostic
        if diag is None:
            return
        summary: dict[str, Any] = {"type": type(block).__name__}
        name = getattr(block, "name", None)
        if isinstance(name, str) and name:
            summary["name"] = name
        raw_input = getattr(block, "input", None)
        if isinstance(raw_input, dict):
            summary["input_keys"] = sorted(str(key) for key in raw_input)
            coerced = _coerce_emit_intent_input(raw_input)
            intent_type = coerced.get("intent_type")
            if isinstance(intent_type, str):
                summary["intent_type"] = intent_type
        diag["tool_blocks"].append(summary)

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
