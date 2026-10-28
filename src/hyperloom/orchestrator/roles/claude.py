# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ClaudeBackend — uses ``claude-agent-sdk`` to drive Claude."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import signal
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from hyperloom.common.agent_cli import resolve_claude_executable
from hyperloom.common.llm_config import claude_sdk_env_options
from hyperloom.common.reasoning_effort import REASONING_EFFORT_RANK
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentValidationError,
    validate_envelope,
)
from ..prompts.transport import TRANSPORT_TOOLS
from hyperloom.inference_optimizer.trace.llm_trace import new_call_id
from hyperloom.inference_optimizer.trace.trajectory_trace import current_context
from hyperloom.common.coerce import to_int
from .base import (
    BackendError,
    BackendTurnResult,
    LLMCallFailed,
    RetryPolicy,
    parse_call_timeout_env,
    retry_with_backoff,
)
from .claude_requests import ClaudeRequestTracker
from .mcp_context_tools import (
    CONTEXT_TOOL_NAMES,
    CONTEXT_TOOL_QUALIFIED_NAMES,
    MCP_SERVER_NAME as CONTEXT_MCP_SERVER_NAME,
    ContextProvider,
    build_context_tools_server,
)
from hyperloom.inference_optimizer.protocol.intent import IntentType
from .mcp_emit_intent import (
    EMIT_INTENT_TOOL_NAME,
    EMIT_INTENT_TOOL_QUALIFIED,
    MCP_SERVER_NAME,
    build_emit_intent_server,
    build_emit_intent_tool_input_schema,
    constraints_sentence,
    decode_emit_intent_input,
    is_unparsed_tool_wrapper,
    payload_contract,
)


log = logging.getLogger(__name__)


def _input_side_total(usage: dict[str, Any]) -> int:
    """Total input-side tokens (fresh + cache read + cache creation) of a usage dict."""
    return (
        to_int(usage.get("input_tokens"), 0)
        + to_int(usage.get("cache_read_input_tokens"), 0)
        + to_int(usage.get("cache_creation_input_tokens"), 0)
    )


def _context_tokens_estimate(usage: dict[str, Any], *, num_turns: int) -> int:
    """Mean per-request context size implied by call-cumulative usage."""
    total = _input_side_total(usage)
    return total // num_turns if num_turns > 1 else total


def _build_output_instructions(allowed_intents: frozenset[IntentType]) -> str:
    """Render the output-format suffix for a role's allowed intent set."""
    contract = payload_contract(allowed_intents)
    constraints = constraints_sentence(allowed_intents)
    constraints_line = f"\n-{constraints}" if constraints else ""
    return f"""
==== OUTPUT FORMAT (REQUIRED) ====
You MUST communicate with the system by calling the
`{EMIT_INTENT_TOOL_QUALIFIED}` tool, under that exact name including its
`mcp__` prefix; it is the only tool that carries an intent. Each call carries
exactly one intent; call multiple times to emit several intents in the same
turn. Free-text replies are dropped.

Tool input shape — these two keys only, nothing beside them:

  {{
    "intent_type": "<one of {", ".join(sorted(t.value for t in allowed_intents))}>",
    "payload": {{ /* per-intent fields — see tool description */ }}
  }}

- Required keys per intent_type: {contract}.{constraints_line}
- The `name{{key='value'}}` shorthand used elsewhere in these instructions
  names an intent, not a tool: call the tool above with
  intent_type="name" and the braced fields inside `payload`, never as
  top-level tool input keys and never as a tool of their own.

Keep payload bodies focused on NEW information. Do not restate context already
in SharedState, your inbox, or analysis.md — reference it and summarize only
what changed. Match length to substance.
==== END OUTPUT FORMAT ====
""".strip()


# Floor applied by ``run()``: Claude Code counts the model's own text message as a turn, so a low max_turns trips
# before any output.
_MIN_MAX_TURNS: int = 8

# Retried timeouts get a progressively larger idle budget so a genuinely slow gateway is not re-killed at the same
# wall.
_RETRY_IDLE_TIMEOUT_MULTIPLIER: float = 2.0


# Wall-clock bound on one ``run()`` call: every attempt and the backoff between them; tearing the CLI down afterwards
# adds at most ``_TURN_CLEANUP_GRACE_SEC``. The idle timeout cannot be that bound: the CLI emits a ``system``/
# ``api_retry`` message each time one of its own API requests times out (10 min each by default, retried up to 10
# times), and every message resets the idle timer, so a stalled gateway can hold a turn open for well over an hour. The default sits below the coordinator's 1800 s reactor-stage
# bound (``INFERENCE_OPTIMIZER_REACTOR_TURN_TIMEOUT_SEC``), the longest turn this runtime already accepts, with room
# for the teardown below, so a reactor turn fails here as a backend error rather than being cancelled from outside;
# and it also bounds the turns no stage bound covers, such as the macro-cycle memory capture.
_TURN_TIMEOUT_ENV: str = "INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC"
DEFAULT_TURN_TIMEOUT_SEC: float = 1500.0
# How long a turn past its bound waits for the SDK to close its CLI (the SDK's own stdin-EOF / SIGTERM / SIGKILL
# escalation is bounded at ~20 s) before abandoning the close and killing the processes itself.
_TURN_CLEANUP_GRACE_SEC: float = 30.0
# Set in the CLI's environment so every process of one turn (the CLI and anything it spawns) can be found and killed.
_TURN_MARKER_ENV: str = "HYPERLOOM_CLAUDE_TURN_ID"


@dataclass
class _TurnStop:
    """Set once one ``run()`` call is being stopped; its SDK task checks it at every message and before every attempt.

    The ``cancel()`` that stops the task is not enough on its own: before Python 3.12 ``asyncio.wait_for`` swallows a
    cancellation that arrives just as the awaited message does, so a turn whose stream keeps delivering (a CLI
    retrying its API request) can lose it and run on, and its retry would start a fresh CLI. Checking this flag at
    each message turns a lost cancellation into one the task raises itself, without re-cancelling (and so cutting
    short) the SDK's own close of the CLI.
    """

    requested: bool = False

    def raise_if_requested(self) -> None:
        """Raise the stop as a cancellation once it has been requested."""
        if self.requested:
            raise asyncio.CancelledError


def _kill_turn_processes(marker: str, *, proc_root: Path = Path("/proc")) -> list[int]:
    """SIGKILL every process whose environment carries ``_TURN_MARKER_ENV=marker``; return their pids.

    The marker is inherited, so this reaches the CLI and every process it spawned, including ones re-parented after
    the CLI died. A process that leads its own process group (a shell the CLI started for a tool) is killed with its
    whole group. Best-effort: a process that exits or cannot be read mid-scan is skipped.
    """
    needle = f"{_TURN_MARKER_ENV}={marker}".encode()
    own_pid = os.getpid()
    try:
        own_pgid = os.getpgid(0)
    except OSError:
        own_pgid = -1
    killed: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return killed
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if needle not in environ.split(b"\0"):
            continue
        try:
            pgid = os.getpgid(pid)
            if pgid == pid and pgid != own_pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue
        killed.append(pid)
    return killed


def _intent_fingerprint(intent: Intent) -> str:
    """Stable key for one validated intent used to drop fallback retries."""
    return json.dumps(
        {"intent_type": intent.type.value, "payload": intent.payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _is_sdk_boundary_error(exc: Exception) -> bool:
    """Whether the SDK raised this to end a turn whose collected output is still usable.

    It raises on a terminal ``ResultMessage`` with ``is_error=True``, which is also how it reports a turn that merely
    ran out of turns or carried an error-shaped success envelope.
    """
    err_str = str(exc)
    return "error result: success" in err_str or "maximum number of turns" in err_str


@dataclass
class _StreamedTurn:
    """What one SDK stream produced: the turn's intents and text, and the telemetry the call reported alongside them.

    ``usages`` holds every usage dict the stream reported, in order: the last is cumulative over the call, the ones
    before it describe single requests.
    """

    intents: list[Intent] = field(default_factory=list)
    text_chunks: list[str] = field(default_factory=list)
    result_chunks: list[str] = field(default_factory=list)
    num_turns: int = 0
    stop_reason: str | None = None
    usages: list[dict[str, Any]] = field(default_factory=list)
    # Defense in depth: identical wrapper envelopes in one query are the parser retry storm.
    seen_fallback_intents: set[str] = field(default_factory=set)

    @property
    def result_text(self) -> str:
        """The call's consolidated ``ResultMessage`` text."""
        return "".join(self.result_chunks)

    @property
    def raw_text(self) -> str:
        """The turn's text: the consolidated ``ResultMessage`` text, falling back to the streamed text blocks."""
        return self.result_text or "".join(self.text_chunks)

    @property
    def usage(self) -> dict[str, Any]:
        """The call-cumulative usage, carrying the largest single request's input-side total as the context peak."""
        if not self.usages:
            return {}
        cumulative = dict(self.usages[-1])
        peak = max((_input_side_total(u) for u in self.usages[:-1]), default=0)
        cumulative["context_tokens_peak"] = peak or _context_tokens_estimate(cumulative, num_turns=self.num_turns)
        return cumulative

    def absorb(self, message: Any) -> None:
        """Take the fields one streamed message reports about the call as a whole; the last non-empty value wins.

        ``ResultMessage.result`` duplicates the streamed text blocks, so it is kept apart from them to avoid
        double-counting. Both ``AssistantMessage`` and the terminal ``ResultMessage`` carry ``stop_reason``, so the
        call's own reason is the one that ends up reported.
        """
        result_text = getattr(message, "result", None)
        if isinstance(result_text, str) and result_text:
            self.result_chunks.append(result_text)
        num_turns = getattr(message, "num_turns", None)
        if isinstance(num_turns, int) and num_turns > 0:
            self.num_turns = num_turns
        stop_reason = getattr(message, "stop_reason", None)
        if isinstance(stop_reason, str) and stop_reason:
            self.stop_reason = stop_reason
        usage = getattr(message, "usage", None)
        if isinstance(usage, dict) and usage:
            self.usages.append(dict(usage))


# Env-driven reasoning effort / extended thinking.
_EFFORT_ENV: str = "INFERENCE_OPTIMIZER_CLAUDE_EFFORT"
_EFFORT_ENV_ORCH: str = "INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT"
_THINKING_ENV: str = "INFERENCE_OPTIMIZER_CLAUDE_THINKING"


def _import_sdk() -> tuple[Any, Any, Any]:
    """Return ``(query, ClaudeAgentOptions, sdk_module)`` or raise."""
    try:
        sdk = importlib.import_module("claude_agent_sdk")
    except ImportError as exc:
        raise BackendError("claude-agent-sdk not installed; run `pip install claude-agent-sdk` (>= 0.1.65).") from exc
    if not (hasattr(sdk, "query") and hasattr(sdk, "ClaudeAgentOptions")):
        raise BackendError("claude_agent_sdk loaded but missing query / ClaudeAgentOptions")
    return sdk.query, sdk.ClaudeAgentOptions, sdk


@dataclass
class ClaudeBackend:
    """Production Claude backend. Implements :class:`Backend`."""

    model: str | None = None
    max_turns_default: int = 12
    # Role's allowed intent set for the emit_intent tool and the output-format suffix. None = all IntentType values.
    allowed_intents: frozenset[IntentType] | None = None
    # Idle timeout for one ``run()`` call: max wall-clock gap allowed BETWEEN streamed SDK messages before the turn is
    # aborted.
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC",
            default=120.0,
        )
    )
    # Wall-clock bound on one ``run()`` call, retries included; see ``DEFAULT_TURN_TIMEOUT_SEC``.
    turn_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(_TURN_TIMEOUT_ENV, default=DEFAULT_TURN_TIMEOUT_SEC)
    )
    # Bounded transient-failure retry/backoff.
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.from_env)

    # Test seams — set these to bypass SDK import / network calls.
    sdk_query_factory: Callable[..., Any] | None = None
    sdk_options_cls: Any | None = None
    sdk_module: Any | None = None
    mcp_server_factory: Callable[..., Any] | None = None
    mcp_tool_factory: Callable[..., Any] | None = None

    name: str = "claude"
    # Which prompt modules describe a surface this backend actually has.
    transport = TRANSPORT_TOOLS
    # The system prompt is cached across turns, so the stable session context is sent in it.
    system_prompt_cached = True
    mcp_server_config: Any | None = field(default=None, init=False)
    # Read-only context-pull MCP server config, set via
    # ``set_context_provider`` and merged into the SDK options.
    _context_server_config: Any | None = field(default=None, init=False)
    _mcp_setup_error: str | None = field(default=None, init=False)
    # The tool palette (``allowed_tools``, ``mcp_servers``) of the latest turn's options, for the MCP setup snapshot.
    _palette_kwargs: dict[str, Any] = field(default_factory=dict, init=False)
    # The CLI's own account from its latest init message: its version, the MCP tools it offered the model, and each
    # MCP server's connection status.
    _cli_init: dict[str, Any] = field(default_factory=dict, init=False)
    # SDK calls abandoned past their bound whose close is still running; held so the task is not collected mid-close.
    _abandoned_turns: set[asyncio.Task[Any]] = field(default_factory=set, init=False)
    # Stops of turns whose caller was cancelled: they outlive that caller, so they are held here until they finish.
    _turn_stoppers: set[asyncio.Task[Any]] = field(default_factory=set, init=False)
    # System prompt files: the CLI's command-line arguments are capped at 128 KiB.
    _system_prompt_dir: tempfile.TemporaryDirectory[str] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Resolve the SDK and register the ``emit_intent`` tool."""
        if self.sdk_query_factory is None or self.sdk_options_cls is None:
            query, opts_cls, mod = _import_sdk()
            if self.sdk_query_factory is None:
                self.sdk_query_factory = query
            if self.sdk_options_cls is None:
                self.sdk_options_cls = opts_cls
            if self.sdk_module is None:
                self.sdk_module = mod
        try:
            self.mcp_server_config = build_emit_intent_server(
                sdk_module=self.sdk_module,
                tool_factory=self.mcp_tool_factory,
                server_factory=self.mcp_server_factory,
                allowed_intents=self._offered_intents,
            )
        except Exception as exc:  # isolation boundary: the emit_intent MCP server, recorded as setup_error
            log.warning("emit_intent MCP setup failed: %s", exc, exc_info=True)
            self._mcp_setup_error = f"{type(exc).__name__}: {exc}"

    # Backend protocol
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """Run a single backend turn against Claude and parse the result."""
        full_prompt = f"{prompt}\n\n{_build_output_instructions(self._offered_intents)}"
        max_turns_use = max(max_turns or self.max_turns_default, _MIN_MAX_TURNS)
        turn_marker = uuid.uuid4().hex
        turn_stop = _TurnStop()
        options = self._build_options(
            tools=tools or [],
            max_turns=max_turns_use,
            system_prompt=system_prompt,
            turn_marker=turn_marker,
        )

        # Each attempt bounds the gap BETWEEN streamed SDK messages (silence budget), not the total turn; each retry
        # amplifies the idle budget.
        attempt_state = {"n": 0}

        async def _one_attempt() -> _StreamedTurn:
            """Run one SDK invocation under an amplified per-attempt idle timeout."""
            # The turn is being stopped: never start another CLI for it.
            turn_stop.raise_if_requested()
            attempt_state["n"] += 1
            idle_timeout_s = self.call_timeout_s * (_RETRY_IDLE_TIMEOUT_MULTIPLIER ** (attempt_state["n"] - 1))
            return await self._invoke_and_collect(
                full_prompt,
                options,
                idle_timeout_s=idle_timeout_s,
                attempt=attempt_state["n"],
                turn_stop=turn_stop,
            )

        try:
            collected = await self._within_turn_bound(
                retry_with_backoff(
                    _one_attempt,
                    policy=self.retry_policy,
                    retry_on=(
                        asyncio.TimeoutError,
                        BackendError,
                        ConnectionError,
                        OSError,
                    ),
                ),
                turn_marker=turn_marker,
                turn_stop=turn_stop,
            )
        except asyncio.TimeoutError as exc:
            log.warning(
                "claude SDK turn timed out: stream idle for >%.0fs on every attempt (retries exhausted); "
                "treating as a failed turn",
                self.call_timeout_s,
            )
            error = LLMCallFailed(
                f"Claude backend timed out: stream idle for >{self.call_timeout_s:.0f}s (likely upstream proxy stall)"
            )
            raise error from exc
        except LLMCallFailed:
            raise
        except Exception as exc:
            # Once retries are exhausted, anything the SDK stream raised is a provider call that produced nothing
            # usable — including the gateway 400s (``litellm.BadRequestError: AnthropicException``) this telemetry
            # exists to count.
            raise LLMCallFailed(f"Claude backend call failed: {exc!r}") from exc
        raw_text = collected.raw_text
        usage = collected.usage
        cache_creation = to_int(usage.get("cache_creation_input_tokens") if usage else None, 0)
        cache_read = to_int(usage.get("cache_read_input_tokens") if usage else None, 0)
        input_tokens = to_int(usage.get("input_tokens") if usage else None, 0)
        output_tokens = to_int(usage.get("output_tokens") if usage else None, 0)
        return BackendTurnResult(
            intents=collected.intents,
            raw_text=raw_text,
            metadata={
                "model": self.model,
                # Pairs this turn's token row with its conversation row; both halves are written from this one
                # metadata dict. A caller that opened an ``llm.call`` trajectory span owns the id.
                "call_id": current_context().call_id or new_call_id(),
                # Why the model stopped ("end_turn" / "max_tokens" / ...).
                "stop_reason": collected.stop_reason,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                # Per-request context size; the counters above sum the call's internal turns and are spend, not size.
                "context_tokens_peak": to_int(usage.get("context_tokens_peak") if usage else None, 0),
                # Full conversation text so the caller (which holds the session_dir / component / tick context the
                # stateless backend lacks) can persist it to conversations.jsonl.
                "prompt": full_prompt,
                "response": raw_text,
            },
        )

    # Internals
    @property
    def _offered_intents(self) -> frozenset[IntentType]:
        """The intent types the agent is offered: the role's set, or every type when none is given."""
        return self.allowed_intents if self.allowed_intents is not None else frozenset(IntentType)

    def set_context_provider(self, provider: ContextProvider | None) -> None:
        """Attach (or clear) the read-only context-pull MCP server."""
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
        except Exception as exc:  # isolation boundary: the context-tools MCP server, absent from the MCP setup snapshot
            log.warning("context tools MCP setup failed: %s", exc, exc_info=True)
            self._context_server_config = None

    def get_mcp_setup_diagnostic(self) -> dict[str, Any]:
        """Return the current MCP setup snapshot: what Hyperloom configured and what the CLI offered the model."""
        schema = json.dumps(
            build_emit_intent_tool_input_schema(self._offered_intents), sort_keys=True, separators=(",", ":")
        )
        mcp_tools = self._cli_init.get("mcp_tools")
        return {
            "backend": type(self).__name__,
            "model": self.model,
            "sdk_name": getattr(self.sdk_module, "__name__", None),
            "sdk_version": getattr(self.sdk_module, "__version__", None),
            "cli_version": self._cli_init.get("cli_version"),
            "gateway_endpoint": self._gateway_endpoint_identifier(),
            "mcp_servers": sorted(self._palette_kwargs.get("mcp_servers", {})),
            "mcp_server_status": self._cli_init.get("mcp_server_status"),
            "mcp_tools": mcp_tools,
            "emit_intent": {
                "qualified_name": EMIT_INTENT_TOOL_QUALIFIED,
                "registered": self.mcp_server_config is not None,
                "offered": None if mcp_tools is None else EMIT_INTENT_TOOL_QUALIFIED in mcp_tools,
                "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
                "setup_error": self._mcp_setup_error,
            },
            "allowed_tools": list(self._palette_kwargs.get("allowed_tools", [])),
        }

    def _gateway_endpoint_identifier(self) -> str | None:
        raw = (os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not raw:
            return None
        parts = urlsplit(raw)
        # hostname, not netloc: netloc carries any ``user:secret@`` userinfo.
        return parts.hostname or "configured"

    def _build_options(
        self,
        *,
        tools: list[str],
        max_turns: int,
        system_prompt: str | None,
        turn_marker: str | None = None,
    ) -> Any:
        """Build the SDK options object for one turn; ``turn_marker`` tags the CLI's processes for cleanup."""
        kwargs = self._turn_option_kwargs(
            max_turns=max_turns,
            system_prompt=system_prompt,
            turn_marker=turn_marker,
        )
        self._palette_kwargs = self._tool_palette_option_kwargs(tools)
        kwargs.update(self._palette_kwargs)
        # A stderr callback makes the SDK pipe the CLI's stderr instead of letting it reach the operator console.
        kwargs["stderr"] = self._stderr_sink
        return self._instantiate_options(kwargs)

    def _turn_option_kwargs(
        self,
        *,
        max_turns: int,
        system_prompt: str | None,
        turn_marker: str | None,
    ) -> dict[str, Any]:
        """The SDK options every turn carries, whatever it is allowed to do: model, CLI, prompt, environment, effort."""
        # Stream events are what time each model request inside the call (see ``claude_requests``).
        kwargs: dict[str, Any] = {"max_turns": max_turns, "include_partial_messages": True}
        if self.model:
            kwargs["model"] = self.model
        cli_path = resolve_claude_executable()
        if cli_path:
            kwargs["cli_path"] = cli_path
        if system_prompt:
            kwargs["system_prompt"] = self._system_prompt_file(system_prompt)
        self._apply_sdk_env_options(kwargs)
        if turn_marker:
            # options.env is merged over the inherited environment, so a marker-only dict changes nothing else.
            kwargs["env"] = {**dict(kwargs.get("env") or {}), _TURN_MARKER_ENV: turn_marker}
        self._apply_effort_options(kwargs)
        return kwargs

    def _system_prompt_file(self, system_prompt: str) -> dict[str, str]:
        """The SDK option that hands the CLI ``system_prompt`` as a file, written once per distinct text."""
        if self._system_prompt_dir is None:
            self._system_prompt_dir = tempfile.TemporaryDirectory(prefix="hyperloom-system-prompt-")
        path = Path(self._system_prompt_dir.name) / f"{hashlib.sha256(system_prompt.encode()).hexdigest()}.md"
        if not path.exists():
            path.write_text(system_prompt, encoding="utf-8")
        return {"type": "file", "path": str(path)}

    def _tool_palette_option_kwargs(self, tools: list[str]) -> dict[str, Any]:
        """The options naming this turn's tool palette and the MCP servers that serve it."""
        # Drop the bare "emit_intent" name; the MCP-qualified form wires into the SDK tool registry.
        allowed = [t for t in tools if t != EMIT_INTENT_TOOL_NAME]
        if self.mcp_server_config is not None and EMIT_INTENT_TOOL_QUALIFIED not in allowed:
            allowed.append(EMIT_INTENT_TOOL_QUALIFIED)
        # Allow-list the context-pull tools' qualified names.
        if self._context_server_config is not None:
            for qname in CONTEXT_TOOL_QUALIFIED_NAMES:
                if qname not in allowed:
                    allowed.append(qname)
        kwargs: dict[str, Any] = {}
        if allowed:
            kwargs["allowed_tools"] = allowed
        kwargs["tools"] = [t for t in allowed if t not in CONTEXT_TOOL_NAMES and not t.startswith("mcp__")]
        mcp_servers: dict[str, Any] = {}
        if self.mcp_server_config is not None:
            mcp_servers[MCP_SERVER_NAME] = self.mcp_server_config
        if self._context_server_config is not None:
            mcp_servers[CONTEXT_MCP_SERVER_NAME] = self._context_server_config
        if mcp_servers:
            kwargs["mcp_servers"] = mcp_servers
        return kwargs

    def _apply_sdk_env_options(self, kwargs: dict[str, Any]) -> None:
        """Pin Claude Code subprocess auth to the current Hyperloom env."""
        kwargs.update(
            claude_sdk_env_options(
                model=self.model,
                component="orchestration",
                operation="orchestrate_turn",
            )
        )

    def _resolve_effort(self) -> str:
        """Reasoning-effort tier: the orchestration override, else the shared override, else ``medium``."""
        return (os.environ.get(_EFFORT_ENV_ORCH) or os.environ.get(_EFFORT_ENV) or "medium").strip().lower()

    def _apply_effort_options(self, kwargs: dict[str, Any]) -> None:
        """Add env-driven reasoning effort + adaptive thinking to the options."""
        effort = self._resolve_effort()
        if effort in REASONING_EFFORT_RANK:
            kwargs["effort"] = effort
        thinking = (os.environ.get(_THINKING_ENV) or "adaptive").strip().lower()
        if thinking and thinking != "off":
            kwargs["thinking"] = {"type": thinking}

    def _instantiate_options(self, kwargs: dict[str, Any]) -> Any:
        """Build the SDK options."""
        return self.sdk_options_cls(**kwargs)

    def _stderr_sink(self, line: str) -> None:
        """Log one line of the CLI's stderr at debug level."""
        text = line.strip()
        if text:
            log.debug("claude CLI stderr: %s", text)

    async def _within_turn_bound(self, coro: Awaitable[Any], *, turn_marker: str, turn_stop: _TurnStop) -> Any:
        """Await ``coro`` for at most :attr:`turn_timeout_s`; past it, stop the call and kill its CLI.

        The SDK call runs in its own task so that a close which never finishes cannot hold the caller: the task is
        stopped (see :meth:`_stop_turn`), given ``_TURN_CLEANUP_GRACE_SEC`` to finish, and abandoned if it does not.
        Either way every process tagged with ``turn_marker`` is then killed. A caller cancelled mid-turn returns at
        once; the same stop then runs in the background. A call that fails on its own also has its processes killed.

        Raises:
            LLMCallFailed: The bound elapsed before ``coro`` finished.
        """
        task = asyncio.ensure_future(coro)
        try:
            done, _ = await asyncio.wait({task}, timeout=self.turn_timeout_s)
        except BaseException:
            # The caller itself was cancelled: take the SDK call and its CLI down without holding the caller.
            stopper = asyncio.ensure_future(self._stop_turn(task, turn_stop, turn_marker))
            self._turn_stoppers.add(stopper)
            stopper.add_done_callback(self._turn_stoppers.discard)
            stopper.add_done_callback(lambda t: t.cancelled() or t.exception())
            raise
        if done:
            try:
                return task.result()
            except Exception:
                # A CLI that crashed (SIGABRT included) or whose call raised may have left children behind.
                killed = _kill_turn_processes(turn_marker)
                if killed:
                    log.warning("claude SDK call failed; killed %d leftover CLI process(es)", len(killed))
                raise
        closed, elapsed, killed = await self._stop_turn(task, turn_stop, turn_marker)
        if closed:
            cleanup = f"SDK closed the CLI in {elapsed:.1f}s"
        else:
            cleanup = f"SDK close still running after {_TURN_CLEANUP_GRACE_SEC:g}s, abandoned"
        message = (
            f"turn exceeded its {self.turn_timeout_s:g}s wall-clock bound ({_TURN_TIMEOUT_ENV}); "
            f"{cleanup}; killed {len(killed)} leftover CLI process(es)"
        )
        log.warning("claude SDK %s; treating as a failed turn", message)
        raise LLMCallFailed(f"Claude backend timed out: {message}")

    async def _stop_turn(
        self, task: asyncio.Task[Any], stop: _TurnStop, turn_marker: str
    ) -> tuple[bool, float, list[int]]:
        """Stop ``task`` (cancel it and set ``stop``, see :class:`_TurnStop`), wait out its close, then kill the turn's
        processes.

        A task still running after ``_TURN_CLEANUP_GRACE_SEC`` is abandoned: held in :attr:`_abandoned_turns` until
        it finishes, and its processes killed again then.

        Returns:
            Whether the task finished within the grace, the seconds the stop took, and the pids killed.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        stop.requested = True
        task.cancel()
        await asyncio.wait({task}, timeout=_TURN_CLEANUP_GRACE_SEC)
        elapsed = loop.time() - started
        closed = task.done()
        if closed:
            if not task.cancelled():
                task.exception()  # Mark retrieved: the stop decides this turn's outcome, not the late result.
        else:
            self._abandoned_turns.add(task)
            task.add_done_callback(self._abandoned_turns.discard)
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
            # Whatever the late close leaves behind (or starts) once it does finish.
            task.add_done_callback(lambda _t: _kill_turn_processes(turn_marker))
        return closed, elapsed, _kill_turn_processes(turn_marker)

    async def _invoke_and_collect(
        self,
        prompt: str,
        options: Any,
        *,
        idle_timeout_s: float,
        attempt: int,
        turn_stop: _TurnStop,
    ) -> _StreamedTurn:
        """Stream the SDK's messages for one attempt and collect what they carry.

        Each model request of the attempt, finished or cut off, lands on the trajectory ledger as an ``llm.request``
        row.
        """
        requests = ClaudeRequestTracker()
        collected = _StreamedTurn()
        stream = self.sdk_query_factory(prompt=prompt, options=options)
        try:
            stream_iter = stream.__aiter__()
            while True:
                # Idle timeout: bound only the wait for the NEXT message; a fully silent gateway trips
                # ``asyncio.TimeoutError``.
                try:
                    message = await asyncio.wait_for(stream_iter.__anext__(), timeout=idle_timeout_s)
                except StopAsyncIteration:
                    break
                # A stop whose cancellation was lost in the wait above (see ``_TurnStop``) is raised here.
                turn_stop.raise_if_requested()
                requests.observe(message)
                self._record_cli_status(message)
                self._collect_message_blocks(message, collected)
                collected.absorb(message)
        except Exception as exc:
            if not _is_sdk_boundary_error(exc):
                raise
            self._record_sdk_boundary_error(exc, collected)
        finally:
            requests.record_trajectory(attempt=attempt, fallback_model=self.model)
            await self._close_stream(stream)
        return collected

    def _record_cli_status(self, message: Any) -> None:
        """Record what a streamed message says about the CLI itself: its session banner and the API retries it makes."""
        subtype = getattr(message, "subtype", None)
        if subtype == "init":
            self._record_cli_init(getattr(message, "data", None) or {})
        if subtype == "api_retry":
            # Each of these resets the stream's idle timer; only the turn's wall-clock bound ends a retry loop.
            retry = getattr(message, "data", None) or {}
            log.warning(
                "claude CLI retrying its API request (attempt %s/%s, status=%s, error=%s)",
                retry.get("attempt"),
                retry.get("max_retries"),
                retry.get("error_status"),
                retry.get("error"),
            )

    def _collect_message_blocks(self, message: Any, collected: _StreamedTurn) -> None:
        """Add one message's content blocks to the turn: emit_intent calls as intents, everything else as text."""
        for block in self._iter_blocks(message):
            if not self._is_tool_use_for_emit_intent(block):
                txt = self._extract_text(block)
                if txt:
                    collected.text_chunks.append(txt)
                continue
            intent = self._parse_tool_use_block(block)
            if intent is None:
                continue
            if is_unparsed_tool_wrapper(getattr(block, "input", None)):
                fingerprint = _intent_fingerprint(intent)
                if fingerprint in collected.seen_fallback_intents:
                    log.info(
                        "claude fallback intent deduped (fingerprint=%s)",
                        hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12],
                    )
                    continue
                collected.seen_fallback_intents.add(fingerprint)
            collected.intents.append(intent)

    def _record_sdk_boundary_error(self, exc: Exception, collected: _StreamedTurn) -> None:
        """Log that the SDK ended the stream by raising, and what survived it."""
        err_str = str(exc)
        if collected.intents:
            log.warning(
                "claude SDK raised '%s' but %d intents already collected; returning partial results",
                err_str,
                len(collected.intents),
            )
        else:
            log.warning(
                "claude SDK raised '%s' with no intents collected; treating as no-intent turn (will retry next tick)",
                err_str,
            )

    @staticmethod
    async def _close_stream(stream: Any) -> None:
        """Close the SDK's stream so an aborted turn does not leak a half-consumed async generator."""
        aclose = getattr(stream, "aclose", None)
        if aclose is None:
            return
        try:
            await aclose()
        except Exception:  # outermost finalizer: stream cleanup, which must not mask the turn result
            log.debug("claude SDK stream aclose failed", exc_info=True)

    @staticmethod
    def _iter_blocks(message: Any):
        """Return the content blocks of an SDK message as a list."""
        return list(getattr(message, "content", None) or [])

    def _is_tool_use_for_emit_intent(self, block: Any) -> bool:
        """Whether a content block is an ``emit_intent`` tool-use call."""
        cls_name = type(block).__name__
        if cls_name not in ("ToolUseBlock", "ServerToolUseBlock"):
            return False
        name = getattr(block, "name", "")
        return name in (EMIT_INTENT_TOOL_NAME, EMIT_INTENT_TOOL_QUALIFIED)

    def _parse_tool_use_block(self, block: Any) -> Intent | None:
        """Validate one ``emit_intent`` tool-use block into an :class:`Intent`."""
        # Hand the raw block input to the decoder: it accepts a native object,
        # Claude Code's __unparsedToolInput wrapper, and the JSON-string
        # envelope that OpenAI-compatible / litellm proxies emit. Passing the
        # value through untouched matters — coercing a non-dict (or an empty
        # string, which ``or {}`` would swallow) to {} decodes to an empty
        # intent with no error, and the real cause surfaces later and wrongly
        # as "intent_type None not in allowed set".
        raw_input, decode_error = decode_emit_intent_input(getattr(block, "input", None))
        if decode_error is not None:
            log.info("claude tool_use decode failed: %s", decode_error)
            return None
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
            return None
        return validated[0] if validated else None

    def _record_cli_init(self, data: dict[str, Any]) -> None:
        """Keep what the CLI's init message says it offered the model; warn on the first one lacking emit_intent."""
        tools = data.get("tools")
        mcp_tools = sorted(tool for tool in tools if tool.startswith("mcp__")) if isinstance(tools, list) else None
        if (
            not self._cli_init
            and self.mcp_server_config is not None
            and mcp_tools is not None
            and EMIT_INTENT_TOOL_QUALIFIED not in mcp_tools
        ):
            log.warning(
                "Claude Code %s did not offer %s to the model; MCP tools offered: %s",
                data.get("claude_code_version"),
                EMIT_INTENT_TOOL_QUALIFIED,
                mcp_tools,
            )
        self._cli_init = {
            "cli_version": data.get("claude_code_version"),
            "mcp_tools": mcp_tools,
            "mcp_server_status": {server.get("name"): server.get("status") for server in data.get("mcp_servers") or []},
        }

    @staticmethod
    def _extract_text(block: Any) -> str:
        """Extract plain text from a content block across block shapes."""
        if type(block).__name__ == "TextBlock":
            return getattr(block, "text", "") or ""
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "") or ""
        t = getattr(block, "text", None)
        return t if isinstance(t, str) else ""


__all__ = ["ClaudeBackend"]
