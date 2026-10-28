# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claude Agent SDK execution backend."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL
from hyperloom.common.llm_stability_env import apply_llm_stability_env
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentHook,
    AgentHooks,
    AgentProviderError,
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
)
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard
from hyperloom.common.llm_headers import format_custom_headers
from kernelforge.agent_backends.gateway import (
    normalize_anthropic_base_url,
    resolve_anthropic_gateway,
)
from kernelforge.tools.process_reaping import (
    ReapReport,
    install_child_subreaper,
    reap_processes_under,
)

log = logging.getLogger(__name__)


class _DeadlineBackport:
    """``asyncio.timeout`` stand-in for Python 3.10 (added to stdlib in 3.11)."""

    def __init__(self, delay: float | None) -> None:
        self._delay = delay
        self._task: asyncio.Task | None = None
        self._handle: asyncio.TimerHandle | None = None
        self._expired = False

    def expired(self) -> bool:
        return self._expired

    def _on_timeout(self) -> None:
        self._expired = True
        if self._task is not None:
            self._task.cancel()

    async def __aenter__(self) -> "_DeadlineBackport":
        if self._delay is not None:
            loop = asyncio.get_running_loop()
            self._task = asyncio.current_task()
            self._handle = loop.call_at(loop.time() + self._delay, self._on_timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.cancel()
        # Our cancellation surfaces as CancelledError; convert it to the same TimeoutError asyncio.timeout raises so
        # the caller's ``except Exception`` catches it (CancelledError is a BaseException on 3.10).
        if self._expired and exc_type is not None and issubclass(exc_type, asyncio.CancelledError):
            raise asyncio.TimeoutError from exc
        return False


def _session_deadline(delay: float | None):
    """Return a timeout context manager that works on both 3.10 and 3.11+."""
    if sys.version_info >= (3, 11):
        return asyncio.timeout(delay)
    return _DeadlineBackport(delay)


def _supports_adaptive_thinking(model: str) -> bool:
    """Resolve Claude thinking capability by model family, not default alias."""
    normalized = model.strip().lower()
    if not normalized:
        return False
    family = re.search(
        # ``[`` terminates the family: an operator who spells a windowed id
        # by hand still names ``claude-opus-5``, and reading the bracket as part
        # of the version would drop it out of the family it belongs to.
        r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d+))?(?:[-._\[]|$)",
        normalized,
    )
    if family:
        major = int(family.group(1))
        minor = int(family.group(2)) if family.group(2) is not None else None
        if major > 4:
            return True
        if major < 4 or minor is None:
            return False
        return minor >= 6
    if re.search(r"claude-3(?:[-._]|$)", normalized):
        return False
    # Gateway aliases generally track current models.
    return True


def _is_turn_cap_error(error: Exception) -> bool:
    """Whether an SDK stream error represents the configured turn ceiling."""
    lowered = str(error).lower()
    return "maximum number of turns" in lowered or "max_turns" in lowered


async def _reap_workspace_processes(cwd: str) -> ReapReport:
    """Kill whatever a timed-out session left running inside its workspace."""
    return await reap_processes_under(cwd, description=f"left running by a timed-out session in {cwd}")


class ClaudeBackendError(AgentProviderError):
    """Base error for Claude backend failures."""


class ClaudeUnavailableError(
    ClaudeBackendError,
    AgentProviderUnavailableError,
):
    """Report an unavailable optional Claude SDK dependency."""


class ClaudeTimeoutError(ClaudeBackendError):
    """The session outran its wall-clock budget before it could be resumed."""


#: ``--version`` verdicts keyed by (path, mtime, size), so a session does not relaunch the CLI to re-ask.
_claude_cli_verdicts: dict[tuple[str, int, int], bool] = {}
_ignored_agent_clis: set[str] = set()


def _is_claude_cli(path: str) -> bool:
    """Whether the executable at ``path`` reports itself as the Claude Code CLI."""
    try:
        status = os.stat(path)
    except OSError:
        return False
    key = (path, status.st_mtime_ns, status.st_size)
    if key not in _claude_cli_verdicts:
        try:
            completed = subprocess.run([path, "--version"], capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            _claude_cli_verdicts[key] = False
        else:
            text = b"\n".join([completed.stdout, completed.stderr]).decode(errors="replace")
            _claude_cli_verdicts[key] = completed.returncode == 0 and "claude" in text.lower()
    return _claude_cli_verdicts[key]


def _bundled_claude_cli() -> str:
    """The CLI claude-agent-sdk ships inside its package, which it would run itself if no path were given."""
    spec = importlib.util.find_spec("claude_agent_sdk")
    if spec is None or not spec.origin:
        return ""
    name = "claude.exe" if sys.platform == "win32" else "claude"
    bundled = Path(spec.origin).parent / "_bundled" / name
    return str(bundled) if bundled.is_file() and os.access(bundled, os.X_OK) else ""


def resolve_claude_cli(explicit: str = "") -> str:
    """Locate the Claude CLI for SDK subprocess execution."""
    if explicit.strip():
        return explicit.strip()
    candidate = os.environ.get("FORGE_AGENT_CLI", "").strip()
    if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        # The variable names one provider's executable while both run in one process.
        if _is_claude_cli(candidate):
            return candidate
        if candidate not in _ignored_agent_clis:
            _ignored_agent_clis.add(candidate)
            log.warning("ignoring FORGE_AGENT_CLI=%s for the Claude backend: it is not the Claude CLI", candidate)
    found = shutil.which("claude")
    if found:
        return found
    candidates = [
        "/usr/local/bin/claude",
        "/usr/bin/claude",
        str(Path.home() / ".local/bin/claude"),
        str(Path.home() / ".npm-global/bin/claude"),
        "/usr/local/lib/node_modules/.bin/claude",
        "/opt/node/bin/claude",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return _bundled_claude_cli() or "claude"


def _hook_matcher(hook: AgentHook, hook_type: Any) -> Any:
    """Translate one generic lifecycle callback into an SDK matcher."""
    kwargs: dict[str, Any] = {"hooks": [hook.callback]}
    if hook.matcher:
        kwargs["matcher"] = hook.matcher
    if hook.timeout_sec is not None:
        kwargs["timeout"] = hook.timeout_sec
    return hook_type(**kwargs)


def _sdk_hooks(hooks: AgentHooks, hook_type: Any) -> dict[str, list[Any]]:
    """Translate generic hook groups into Claude SDK hook names."""
    translated: dict[str, list[Any]] = {}
    groups = (
        ("PreToolUse", hooks.pre_tool_use),
        ("PostToolUse", hooks.post_tool_use),
        ("Stop", hooks.stop),
    )
    for name, entries in groups:
        if entries:
            translated[name] = [_hook_matcher(entry, hook_type) for entry in entries]
    return translated


def _load_claude_sdk() -> tuple[Any, Any]:
    """Load the optional Claude SDK or raise a provider-level error."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError as exc:
        raise ClaudeUnavailableError("claude-agent-sdk is not installed; install the 'claude' extra") from exc
    return query, ClaudeAgentOptions


_PROGRESS_MAX_ENTRIES = 400
_PROGRESS_TEXT_CHARS = 160


def _tool_argument_digest(payload: Any) -> str:
    """One short, human-scannable line for a tool call's arguments."""
    if not isinstance(payload, dict):
        return ""
    for key in ("file_path", "path", "pattern", "command", "notebook_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_PROGRESS_TEXT_CHARS]
    return ""


def _record_progress(sink: list[str] | None, message: Any) -> None:
    """Append what this streamed message shows the agent doing."""
    if sink is None:
        return
    try:
        for block in getattr(message, "content", None) or ():
            if hasattr(block, _TEXT):
                text = " ".join(str(block.text).split())
                if text:
                    sink.append(f"say: {text[:_PROGRESS_TEXT_CHARS]}")
            elif block.__class__.__name__ == "ToolUseBlock":
                name = getattr(block, "name", "?")
                detail = _tool_argument_digest(getattr(block, "input", {}))
                sink.append(f"tool: {name}{f' {detail}' if detail else ''}")
        if hasattr(message, "total_cost_usd"):
            sink.append(f"end: subtype={getattr(message, 'subtype', '') or '?'}")
        overflow = len(sink) - _PROGRESS_MAX_ENTRIES
        if overflow > 0:
            del sink[:overflow]
    except Exception:  # never let telemetry break the session
        log.debug("agent progress recording failed", exc_info=True)


def _prepare_claude_environment() -> None:
    """Apply Claude CLI environment compatibility only when selected."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")
    gateway = resolve_anthropic_gateway()
    if gateway.has_endpoint:
        configured = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        normalized = normalize_anthropic_base_url(gateway.base_url)
        if normalized != configured:
            # Say so: this edits a process-wide variable the operator set, and a silent rewrite is the thing that
            # makes an endpoint problem hard to trace in the first place.
            log.info(
                "ANTHROPIC_BASE_URL %s -> %s (the CLI appends /v1/messages itself)",
                configured,
                normalized,
            )
            os.environ["ANTHROPIC_BASE_URL"] = normalized
    if gateway.headers:
        os.environ["ANTHROPIC_CUSTOM_HEADERS"] = format_custom_headers(gateway.headers)


def _session_env_defaults() -> dict[str, str]:
    """The shared LLM-transport stability env, less every variable the operator's environment already sets.

    It turns off Claude Code's non-essential traffic, which includes a session-title request that resends the first
    prompt, uncached, at the session model's price; no Forge session reads that title.
    """
    defaults: dict[str, str] = {}
    apply_llm_stability_env(defaults)
    return {key: value for key, value in defaults.items() if key not in os.environ}


async def _refuse_described_only_tool(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
    """Deny a tool the session describes only because the conversation it continues used it."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "This turn may only read; the tool is listed for the conversation it continues.",
        }
    }


# The content-block type of plain text, and the attribute an SDK text block carries it in.
_TEXT = "text"


def _builtin_tools(names: list[str]) -> list[str]:
    """Narrow a permission list to the built-in tools it names.

    ``--tools`` selects from the CLI's built-in set and rejects anything else, so
    an MCP tool -- which reaches the session through ``allowed_tools`` and its own
    server declaration -- must not be forwarded here.
    """
    return [name for name in dict.fromkeys(names) if not name.startswith("mcp__")]


def _shared_prefix_prompt(spec: AgentRunSpec):
    """The first user message as two text blocks, the shared prefix closed by a cache breakpoint of its own.

    The CLI caches a prompt at the end of the system prompt and of the last message only, so a prefix several
    sessions share inside one text block is written again by every one of them.
    """
    prefix = spec.shared_prompt_prefix
    if not spec.user_prompt.startswith(prefix):
        raise ClaudeBackendError("shared_prompt_prefix must be a leading part of user_prompt")
    parts = [prefix, spec.user_prompt[len(prefix) :]]
    blocks: list[dict[str, Any]] = [{"type": _TEXT, _TEXT: part} for part in parts if part]
    blocks[0]["cache_control"] = {"type": "ephemeral"}
    message = {
        "type": "user",
        "message": {"role": "user", "content": blocks},
        "parent_tool_use_id": None,
        "session_id": "default",
    }

    async def single_message():
        yield message

    return single_message()


def _first_prompt(spec: AgentRunSpec):
    """What a fresh session sends first: the user prompt, split around its shared prefix when it has one."""
    return _shared_prefix_prompt(spec) if spec.shared_prompt_prefix else spec.user_prompt


def _with_response_start(stream, callback):
    """``stream``, calling ``callback`` once at the first sign the model is answering: a partial event, or failing
    that a whole message."""
    if callback is None:
        return stream

    async def reporting():
        pending = callback
        try:
            async for message in stream:
                if pending is not None and message.__class__.__name__ in ("StreamEvent", "AssistantMessage"):
                    pending, notify = None, pending
                    notify()
                yield message
        finally:
            # Closing this wrapper must still tear down the CLI behind the SDK stream.
            await stream.aclose()

    return reporting()


class _StreamTranscript:
    """What one SDK message stream said, in the fields a run result reports."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.subtype = ""
        self.num_turns: int | None = None
        self.session_id = ""

    def absorb(self, message: Any) -> None:
        """Fold one message into the transcript."""
        # The init SystemMessage and the final ResultMessage both carry the session id; keep the
        # latest non-empty one so a caller can resume this exact conversation later.
        candidate_session = getattr(message, "session_id", "") or ""
        if isinstance(candidate_session, str) and candidate_session:
            self.session_id = candidate_session
        if hasattr(message, "total_cost_usd"):
            self.subtype = getattr(message, "subtype", "") or ""
            turns = getattr(message, "num_turns", None)
            if isinstance(turns, int):
                self.num_turns = turns
        if not hasattr(message, "content"):
            return
        for block in message.content:
            if hasattr(block, _TEXT):
                self.text_parts.append(block.text)
            elif block.__class__.__name__ == "ToolUseBlock":
                self.tool_calls.append((getattr(block, "name", "?"), getattr(block, "input", {}) or {}))


def _session_end_reason(*, timed_out: bool, stream_error: Exception | None, subtype: str) -> tuple[str, str, str]:
    """The subtype, end reason and closing note for a session that has stopped."""
    if timed_out:
        return subtype or "error_timeout", "timeout", f"[session ended: {stream_error}]"
    if stream_error is not None:
        note = f"[session ended with SDK error: {stream_error}]"
        if _is_turn_cap_error(stream_error):
            return subtype or "error_max_turns", "turn_cap", note
        return subtype or "error", "sdk_error", note
    if "max_turns" in subtype:
        return subtype, "turn_cap", ""
    if subtype and subtype != "success":
        return subtype, f"sdk_{subtype}", ""
    if subtype == "success":
        return subtype, "agent_stopped", ""
    # The stream ended without the ResultMessage that carries the subtype, so the CLI never said how the
    # session finished; reading that as a voluntary stop hands the caller whatever the session left behind
    # as the agent's answer.
    return subtype, "sdk_no_result", ""


def _verified_file_changes(guard: WorkspaceGuard):
    """What the guard says the session changed, rolling the workspace back when it cannot say.

    ``verify()`` restores the baseline itself before raising, so this covers the
    paths that fail earlier and must not mask them.
    """
    try:
        return guard.verify()
    except Exception:
        with suppress(Exception):  # broad-suppress: rollback must not shadow the verify error
            guard.rollback()
        raise


class ClaudeBackend:
    """Execute Forge sessions through the Claude Agent SDK."""

    name = "claude"
    capabilities = AgentCapabilities(
        writable=True,
        resumable=True,
        stop_hooks=True,
        native_subagents=True,
        mcp=True,
        probe=True,
        session_env=True,
        workspace_guard=True,
        shared_prompt_cache=True,
    )

    def __init__(
        self,
        runtime: AgentRuntimeConfig | None = None,
    ) -> None:
        """Resolve SDK symbols when the backend is selected."""
        self.runtime = runtime or AgentRuntimeConfig(
            provider=self.name,
            model=DEFAULT_CLAUDE_MODEL,
        )
        _prepare_claude_environment()
        self._query, self._options_type = _load_claude_sdk()
        self.fallback_reason = ""

    def preflight(self) -> None:
        """Validate that an explicitly configured executable is Claude CLI."""
        if not self.runtime.executable.strip():
            return
        self.validate_runtime(self.runtime)

    @staticmethod
    def validate_runtime(runtime: AgentRuntimeConfig) -> None:
        """Check the selected CLI with a 10-second --version call, without loading the SDK."""
        selected = resolve_claude_cli(runtime.executable)
        candidate = Path(selected).expanduser()
        executable = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else shutil.which(selected)
        if not executable:
            raise ClaudeUnavailableError(f"Claude CLI is not executable: {selected}")
        try:
            version = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ClaudeUnavailableError(f"Claude CLI version check failed: {exc}") from exc
        version_text = b"\n".join([version.stdout, version.stderr]).decode(errors="replace").strip()
        if version.returncode != 0 or "claude" not in version_text.lower():
            raise ClaudeUnavailableError(
                f"configured CLI does not appear to be Claude: {selected}; --version returned {version_text!r}"
            )

    def probe(
        self,
        *,
        cwd: str,
        model: str = "",
        reasoning_effort: str = "",
        timeout_sec: int | None = None,
        usage: Any = None,
    ) -> AgentRunResult:
        """Make one tool-free request to verify URL/key/model compatibility."""
        del usage  # Availability probes are not part of campaign accounting.
        self.preflight()
        selected_model = model.strip() or self.runtime.model
        timeout = timeout_sec or min(60, self.runtime.timeout_sec)
        command = [
            resolve_claude_cli(self.runtime.executable),
            "--print",
            "Reply with exactly OK. Do not inspect files or run tools.",
            "--output-format",
            "json",
            "--model",
            selected_model,
            "--effort",
            # The probe answers "will the campaign's configuration work", so it
            # has to ask under that configuration: pinning ``low`` here made the
            # probe pass on deployments where the configured effort is the thing
            # the gateway rejects.
            reasoning_effort.strip() or self.runtime.reasoning_effort.strip() or "low",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--max-turns",
            "1",
            "--no-session-persistence",
        ]
        # The probe must reach the endpoint the sessions will, and sessions apply ``options["env"]`` over the process.
        runtime_env = self.runtime.options.get("env") or {}
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env={**os.environ, **runtime_env} if runtime_env else None,
            )
        except Exception as error:
            raise ClaudeUnavailableError(f"Claude model probe failed for {selected_model!r}: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-1200:]
            raise ClaudeUnavailableError(f"Claude model probe failed for {selected_model!r}: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ClaudeUnavailableError(f"Claude model probe returned invalid JSON for {selected_model!r}") from error
        text = str(payload.get("result") or "").strip()
        if text != "OK":
            raise ClaudeUnavailableError(
                f"Claude model probe returned an unexpected response for {selected_model!r}: {text[:200]!r}"
            )
        return AgentRunResult(text=text, end_reason="agent_stopped")

    def _tool_policy_options(self, policy, spec: AgentRunSpec) -> dict[str, Any]:
        """The SDK options one generic tool policy asks for."""
        allowed_tools = self._role_tools(policy)
        declared_tools = self._role_tools(spec.declared_tool_policy) if spec.declared_tool_policy else allowed_tools
        options: dict[str, Any] = {
            "allowed_tools": allowed_tools,
            # ``allowed_tools`` is only a permission list: the CLI still loads
            # and describes every built-in tool, and those schemas sit in the
            # cached prefix that is re-read on every turn of the session, so
            # naming the base set here is a per-turn saving rather than a
            # one-off. Four campaigns differing in this line alone, all
            # deferring the knowledge maps, put an analysis session's
            # first-turn prefix at 35,555 / 35,507 tokens with the default
            # set against 8,487 / 8,439 naming the six an implementer uses:
            # -27,068, or -76.2%, with each arm's two replicates within 50
            # tokens of each other. The implementer lanes of those runs
            # split by the same amount, 44.9-47.1k against 17.4-19.6k.
            # The saving assumes the knowledge maps are deferred: inlined,
            # they dominate the prefix and this line saves only 5,317 tokens.
            # MCP tools are not part of the built-in set and are carried by
            # ``allowed_tools`` alone.
            "tools": _builtin_tools(declared_tools),
            "permission_mode": (policy.permission_mode or os.environ.get("FORGE_PERMISSION_MODE", "acceptEdits")),
        }
        if policy.max_turns is not None:
            options["max_turns"] = policy.max_turns
        if _supports_adaptive_thinking(spec.model):
            # Claude 4.6+ uses adaptive thinking. Without a summarized display the stream carries no bytes until the
            # thinking ends, and a gateway that drops a silent stream cuts a long planning turn before its first token;
            # every retry then thinks from the start. Billing counts the full thinking either way.
            options["thinking"] = {"type": "adaptive", "display": "summarized"}
        elif policy.thinking_budget_tokens > 0:
            options["thinking"] = {"type": "enabled", "budget_tokens": policy.thinking_budget_tokens}
        if policy.bare and spec.hooks is None:
            options["extra_args"] = {"bare": None}
        return options

    def _apply_subagents(self, options: dict[str, Any], subagents) -> None:
        """Declare the spec's subagents, and make sure this session can reach them."""
        from claude_agent_sdk import AgentDefinition

        options["agents"] = {
            name: AgentDefinition(
                description=role.description,
                prompt=role.instructions,
                tools=(
                    self._role_tools(role.tool_policy) if role.tool_policy is not None else ["Read", "Grep", "Glob"]
                ),
                model=role.model or None,
            )
            for name, role in subagents.items()
        }
        allowed = options.setdefault("allowed_tools", [])
        if "Task" not in allowed:
            allowed.append("Task")
        if "tools" in options and "Task" not in options["tools"]:
            # Task is built-in, so a restricted base set has to name it or the
            # subagents declared just above are unreachable.
            options["tools"] = [*options["tools"], "Task"]

    @staticmethod
    def _apply_mcp_servers(options: dict[str, Any], mcp_servers) -> None:
        """Declare the spec's MCP servers and allow the tools they carry."""
        options["mcp_servers"] = {
            name: {
                "type": "stdio",
                "command": server.command,
                "args": list(server.args),
                **({"env": server.env} if server.env else {}),
            }
            for name, server in mcp_servers.items()
        }
        allowed = options.setdefault("allowed_tools", [])
        for server in mcp_servers.values():
            for tool in server.tools:
                if tool not in allowed:
                    allowed.append(tool)

    def _provider_options(self, spec: AgentRunSpec) -> dict[str, Any]:
        """Adapt a generic run specification into Claude SDK options."""
        spec = self._with_described_only_tools_denied(spec)
        options: dict[str, Any] = {
            "model": spec.model,
            "cwd": spec.cwd,
            "system_prompt": spec.system_prompt,
            "cli_path": resolve_claude_cli(self.runtime.executable),
        }
        if spec.reasoning_effort:
            options["effort"] = spec.reasoning_effort
        if spec.additional_directories:
            options["add_dirs"] = list(spec.additional_directories)
        if spec.tool_policy is not None:
            options.update(self._tool_policy_options(spec.tool_policy, spec))
        if spec.hooks is not None:
            from claude_agent_sdk import HookMatcher

            options["hooks"] = _sdk_hooks(spec.hooks, HookMatcher)
            options.pop("extra_args", None)
        if spec.subagents:
            self._apply_subagents(options, spec.subagents)
        if spec.mcp_servers:
            self._apply_mcp_servers(options, spec.mcp_servers)
        if spec.on_response_started is not None:
            # Partial events report the response starting; whole messages only arrive as each block completes.
            options["include_partial_messages"] = True
        options.update(self.runtime.options)
        options.update(spec.provider_options)
        if spec.env:
            # The SDK spawns the CLI with the inherited process environment and applies this over it.
            options["env"] = {**options.get("env", {}), **spec.env}
        if defaults := _session_env_defaults():
            options["env"] = {**defaults, **options.get("env", {})}
        return options

    def _with_described_only_tools_denied(self, spec: AgentRunSpec) -> AgentRunSpec:
        """``spec`` with a PreToolUse hook denying every built-in tool it describes but its tool policy does not allow.

        A deny hook runs before any permission mode or settings allow rule can approve the tool, and unlike
        ``disallowed_tools`` it leaves the tool definitions, which open the cached prefix, as they are. A bare session
        runs no hooks and has a prefix of its own, so one describes only the tools it may use.
        """
        if spec.declared_tool_policy is None or spec.tool_policy is None:
            return spec
        allowed = set(self._role_tools(spec.tool_policy))
        described_only = [
            name for name in _builtin_tools(self._role_tools(spec.declared_tool_policy)) if name not in allowed
        ]
        if not described_only:
            return spec
        if spec.hooks is None and spec.tool_policy.bare:
            return replace(spec, declared_tool_policy=None)
        denial = AgentHook(matcher="|".join(described_only), callback=_refuse_described_only_tool)
        hooks = spec.hooks or AgentHooks()
        return replace(spec, hooks=replace(hooks, pre_tool_use=[denial, *hooks.pre_tool_use]))

    @staticmethod
    def _role_tools(policy) -> list[str]:
        """Translate one generic role tool policy into Claude tool names."""
        tools: list[str] = []
        if policy.read:
            tools.append("Read")
        if policy.search:
            tools.extend(["Grep", "Glob"])
        if policy.write:
            tools.extend(["Edit", "Write"])
        if policy.shell:
            tools.append("Bash")
        tools.extend(policy.extra_tools)
        return list(dict.fromkeys(tools))

    async def run(self, spec: AgentRunSpec, usage: Any = None) -> AgentRunResult:
        """Run one Claude SDK query and normalize its streamed messages."""
        spec = spec.resolved(self.runtime)
        return await self._query_once(spec, _first_prompt(spec), usage=usage)

    async def resume(
        self,
        spec: AgentRunSpec,
        session_id: str,
        feedback: str,
        usage: Any = None,
    ) -> AgentRunResult:
        """Continue an exact prior SDK session with a new prompt."""
        if not session_id.strip():
            raise ClaudeBackendError("Claude resume requires a session ID")
        spec = spec.resolved(self.runtime)
        result = await self._query_once(
            spec,
            feedback,
            usage=usage,
            resume_session_id=session_id.strip(),
        )
        if not result.session_id:
            result.session_id = session_id.strip()
        return result

    async def _query_once(
        self,
        spec: AgentRunSpec,
        prompt: str,
        *,
        usage: Any = None,
        resume_session_id: str = "",
    ) -> AgentRunResult:
        """Run one guarded SDK query (fresh or resumed) and normalize its messages."""
        # Armed before the CLI starts rather than when the reaper runs: the ownership tag is only inherited by
        # children exec'd after it is set, and the subreaper flag is what keeps a detached benchmark traceable to this
        # session once the shell that started it has exited.
        install_child_subreaper()
        # These worktrees carry the loop's own ledger and build output, so a clean-HEAD demand would refuse every
        # session before it started.
        guard = WorkspaceGuard(spec, dirty_baseline_default=True)
        guard.prepare()
        provider_options = self._provider_options(spec)
        if resume_session_id:
            provider_options["resume"] = resume_session_id
        options = self._options_type(**provider_options)
        transcript = _StreamTranscript()

        # The SDK RAISES on the turn cap (and some other mid-session failures) rather than yielding a final
        # ResultMessage, and its stream is an unbounded ``async for`` -- nothing here stops a session that neither
        # answers nor caps.
        stream_error: Exception | None = None
        timed_out = False
        # Independent of ``timed_out``: the CLI subprocess and any detached benchmark children exist the moment the
        # deadline fires, even before a session id is established, so they must be torn down on that path too --
        # otherwise a hung init leaks the process group and keeps the GPU.
        reap_on_exit = False
        # Set on the paths that leave without a result.
        rollback_on_exit = False
        # What the reap could not clear out of the workspace.
        contention = ""
        agen = _with_response_start(self._query(prompt=prompt, options=options), spec.on_response_started)
        # ``asyncio.timeout(None)`` applies no bound, so a spec without a budget keeps the previous unbounded
        # behaviour; ``expired()`` tells our own deadline apart from a bare TimeoutError surfacing from the transport.
        deadline = _session_deadline(spec.timeout_sec)
        try:
            try:
                async with deadline:
                    async for message in agen:
                        if usage is not None:
                            usage.add_from_message(message, role=spec.role)
                        _record_progress(spec.progress_log, message)
                        transcript.absorb(message)
            except Exception as exc:
                if deadline.expired():
                    reap_on_exit = True
                    if not transcript.session_id:
                        rollback_on_exit = True
                        raise ClaudeTimeoutError(
                            f"Claude session timed out after {spec.timeout_sec}s before it established a session"
                        ) from exc
                    timed_out = True
                    stream_error = TimeoutError(f"Claude session timed out after {spec.timeout_sec}s")
                    log.warning(
                        "Claude session %s timed out after %ss; preserving the resume handle and reaping its leftovers",
                        transcript.session_id,
                        spec.timeout_sec,
                    )
                    if spec.progress_log is not None:
                        spec.progress_log.append(f"end: timeout {spec.timeout_sec}s")
                else:
                    if not transcript.session_id:
                        rollback_on_exit = True
                        raise
                    stream_error = exc
                    if not _is_turn_cap_error(exc):
                        log.warning(
                            "Claude SDK stream failed after session %s; preserving the resume handle (%s: %s)",
                            transcript.session_id,
                            type(exc).__name__,
                            exc,
                        )
                    if spec.progress_log is not None:
                        spec.progress_log.append(f"end: sdk-error {str(exc)[:_PROGRESS_TEXT_CHARS]}")
        except asyncio.CancelledError:
            # The subprocess and any detached benchmark children are running regardless of whether a session id was
            # established; reap them so they do not hold the GPU past this point.
            reap_on_exit = True
            if not transcript.session_id:
                rollback_on_exit = True
            raise
        finally:
            if reap_on_exit:
                # ``async for`` does not close its iterator on exit (PEP 533 was deferred), so close it to tear the
                # CLI down, then reap any detached benchmark child that outlived it -- an orphan holding the GPU
                # corrupts the canonical measurement that follows.
                with suppress(Exception):  # broad-suppress: SDK generator close; the reap below is what matters
                    await agen.aclose()
                report = await _reap_workspace_processes(spec.cwd)
                if report.contended:
                    contention = report.describe()
            if rollback_on_exit:
                guard.rollback()

        subtype, end_reason, closing_note = _session_end_reason(
            timed_out=timed_out,
            stream_error=stream_error,
            subtype=transcript.subtype,
        )
        if closing_note:
            transcript.text_parts.append(closing_note)

        result = AgentRunResult(
            text="\n".join(transcript.text_parts).strip(),
            subtype=subtype,
            num_turns=transcript.num_turns,
            end_reason=end_reason,
            session_id=transcript.session_id,
            tool_calls=transcript.tool_calls,
            stderr_tail=(str(stream_error)[:2000] if stream_error else ""),
            workspace_contention=contention,
        )
        result.file_changes = _verified_file_changes(guard)
        result.target_edit_count = guard.count_target_edits()
        result.edit_count = result.target_edit_count
        return result


__all__ = [
    "ClaudeBackend",
    "ClaudeBackendError",
    "ClaudeTimeoutError",
    "ClaudeUnavailableError",
    "resolve_claude_cli",
]
