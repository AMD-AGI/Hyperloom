# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Claude Agent SDK execution backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

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
from kernelforge.llm import (
    format_custom_headers,
    normalize_anthropic_base_url,
    resolve_anthropic_gateway,
)
from kernelforge.llm.process_reaping import (
    ReapReport,
    install_child_subreaper,
    reap_processes_under,
)

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
FALLBACK_CLAUDE_MODEL = "claude-opus-4-8"
log = logging.getLogger(__name__)


class _DeadlineBackport:
    """``asyncio.timeout`` stand-in for Python 3.10 (added to stdlib in 3.11).

    Cancels the running task once the delay elapses and surfaces the same
    ``TimeoutError`` the 3.11+ context manager would, while ``expired()`` tells
    our own deadline apart from a transport ``TimeoutError``. A delay of None
    applies no bound, matching ``asyncio.timeout(None)``.
    """

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
        # Our cancellation surfaces as CancelledError; convert it to the same
        # TimeoutError asyncio.timeout raises so the caller's ``except Exception``
        # catches it (CancelledError is a BaseException on 3.10).
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
        r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d+))?(?:[-._]|$)",
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
    # Gateway aliases generally track current models. Prefer the modern API and
    # let callers targeting a known legacy model use its canonical family name.
    return True


def _is_turn_cap_error(error: Exception) -> bool:
    """Whether an SDK stream error represents the configured turn ceiling."""
    lowered = str(error).lower()
    return "maximum number of turns" in lowered or "max_turns" in lowered


async def _reap_workspace_processes(cwd: str) -> ReapReport:
    """Kill whatever a timed-out session left running inside its workspace.

    A benchmark still running when the deadline expires outlives the CLI and
    keeps the device busy through the canonical measurement that follows, so the
    workspace has to be clear before this returns. Whatever could not be cleared
    -- a process of ours that survived SIGKILL, or one that is not this
    campaign's to kill at all -- comes back in the report, because the caller is
    the one that can decline to measure.
    """
    return await reap_processes_under(cwd, description=f"left running by a timed-out session in {cwd}")


class ClaudeBackendError(AgentProviderError):
    """Base error for Claude backend failures."""


class ClaudeUnavailableError(
    ClaudeBackendError,
    AgentProviderUnavailableError,
):
    """Report an unavailable optional Claude SDK dependency."""


class ClaudeTimeoutError(ClaudeBackendError):
    """The session outran its wall-clock budget before it could be resumed.

    Raised only when the deadline expired before any session id existed: nothing
    was established, so there is no handle to preserve and the failure precedes
    the session. A local deadline is a limit the caller chose, never transport
    weather, so :mod:`~kernelforge.agent_backends.session_resume` must not retry
    it -- a re-run would burn the same clock to reach the same deadline.
    """


def resolve_claude_cli(explicit: str = "") -> str:
    """Locate the Claude CLI for SDK subprocess execution."""
    if explicit.strip():
        return explicit.strip()
    candidate = os.environ.get("FORGE_AGENT_CLI", "").strip()
    if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
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
    return "claude"


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
    """Append what this streamed message shows the agent doing.

    Best-effort by construction: observability must never be able to fail a run,
    so any surprise in the SDK's message shape is swallowed.
    """
    if sink is None:
        return
    try:
        for block in getattr(message, "content", None) or ():
            if hasattr(block, "text"):
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
    except Exception:  # noqa: BLE001 — never let telemetry break the session
        pass


def _prepare_claude_environment() -> None:
    """Apply Claude CLI environment compatibility only when selected.

    ``ANTHROPIC_BASE_URL`` keeps the operator's route but loses a duplicated
    ``/v1`` tail, because the CLI appends its own. A LiteLLM proxy publishes its
    base that way, and left as configured the CLI answers "There's an issue with
    the selected model ... it may not exist or you may not have access to it" --
    a 404 on the doubled path, reported as a model and permission problem.

    ``ANTHROPIC_CUSTOM_HEADERS`` is consumed by the CLI rather than passed in, so
    it is normalized in place through the same parser the OpenAI line uses:
    ``${VAR}`` references are resolved, and a JSON object is rewritten as the
    newline-delimited form the CLI understands. Missing or unparseable input is
    left alone rather than replaced with an empty value.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.environ.setdefault("IS_SANDBOX", "1")
    gateway = resolve_anthropic_gateway()
    if gateway.has_endpoint:
        configured = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        normalized = normalize_anthropic_base_url(gateway.base_url)
        if normalized != configured:
            # Say so: this edits a process-wide variable the operator set, and a
            # silent rewrite is the thing that makes an endpoint problem hard to
            # trace in the first place.
            log.info(
                "ANTHROPIC_BASE_URL %s -> %s (the CLI appends /v1/messages itself)",
                configured,
                normalized,
            )
            os.environ["ANTHROPIC_BASE_URL"] = normalized
    if gateway.headers:
        os.environ["ANTHROPIC_CUSTOM_HEADERS"] = format_custom_headers(gateway.headers)


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
    )

    def __init__(
        self,
        runtime: AgentRuntimeConfig | None = None,
    ) -> None:
        """Resolve SDK symbols when the backend is selected."""
        self.runtime = runtime or AgentRuntimeConfig(
            provider=self.name,
            model=DEFAULT_CLAUDE_MODEL,
            fallback_model=FALLBACK_CLAUDE_MODEL,
        )
        _prepare_claude_environment()
        self._query, self._options_type = _load_claude_sdk()
        self.fallback_reason = ""

    def preflight(self) -> None:
        """Validate that an explicitly configured executable is Claude CLI."""
        explicit = self.runtime.executable.strip()
        if not explicit:
            return
        candidate = Path(explicit).expanduser()
        executable = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else shutil.which(explicit)
        if not executable:
            raise ClaudeUnavailableError(f"Claude CLI is not executable: {explicit}")
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
                f"configured CLI does not appear to be Claude: {explicit}; --version returned {version_text!r}"
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
            reasoning_effort.strip() or "low",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--max-turns",
            "1",
            "--no-session-persistence",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
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

    def _provider_options(self, spec: AgentRunSpec) -> dict[str, Any]:
        """Adapt a generic run specification into Claude SDK options."""
        options: dict[str, Any] = {
            "model": spec.model,
            "cwd": spec.cwd,
            "system_prompt": spec.system_prompt,
            "cli_path": resolve_claude_cli(self.runtime.executable),
        }
        if spec.reasoning_effort:
            options["effort"] = spec.reasoning_effort
        fallback_model = getattr(self.runtime, "fallback_model", "").strip()
        if fallback_model and fallback_model != spec.model:
            options["fallback_model"] = fallback_model
        if spec.additional_directories:
            options["add_dirs"] = list(spec.additional_directories)
        policy = spec.tool_policy
        if policy is not None:
            allowed_tools: list[str] = []
            if policy.read:
                allowed_tools.append("Read")
            if policy.search:
                allowed_tools.extend(["Grep", "Glob"])
            if policy.write:
                allowed_tools.extend(["Edit", "Write"])
            if policy.shell:
                allowed_tools.append("Bash")
            allowed_tools.extend(policy.extra_tools)
            options.update(
                allowed_tools=list(dict.fromkeys(allowed_tools)),
                permission_mode=(policy.permission_mode or os.environ.get("FORGE_PERMISSION_MODE", "acceptEdits")),
            )
            if policy.max_turns is not None:
                options["max_turns"] = policy.max_turns
            if _supports_adaptive_thinking(spec.model):
                # Claude 4.6+ uses adaptive thinking. Claude 4.7+ rejects fixed
                # budget_tokens entirely, so capability must follow the model
                # family rather than whichever alias is currently the default.
                options["thinking"] = {"type": "adaptive"}
            elif policy.thinking_budget_tokens > 0:
                options["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": policy.thinking_budget_tokens,
                }
            if policy.bare and spec.hooks is None:
                options["extra_args"] = {"bare": None}
        if spec.hooks is not None:
            from claude_agent_sdk import HookMatcher

            options["hooks"] = _sdk_hooks(spec.hooks, HookMatcher)
            options.pop("extra_args", None)
        if spec.subagents:
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
                for name, role in spec.subagents.items()
            }
            allowed = options.setdefault("allowed_tools", [])
            if "Task" not in allowed:
                allowed.append("Task")
        if spec.mcp_servers:
            options["mcp_servers"] = {
                name: {
                    "type": "stdio",
                    "command": server.command,
                    "args": list(server.args),
                    **({"env": server.env} if server.env else {}),
                }
                for name, server in spec.mcp_servers.items()
            }
            allowed = options.setdefault("allowed_tools", [])
            for server in spec.mcp_servers.values():
                for tool in server.tools:
                    if tool not in allowed:
                        allowed.append(tool)
        options.update(self.runtime.options)
        options.update(spec.provider_options)
        if spec.env:
            # The SDK spawns the CLI with the inherited process environment and
            # applies this over it. Merged last rather than assigned earlier: a
            # provider option carrying its own env would otherwise drop the
            # session's, and that is what keeps concurrent sessions out of each
            # other's build cache.
            options["env"] = {**options.get("env", {}), **spec.env}
        return options

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
        return await self._query_once(spec, spec.user_prompt, usage=usage)

    async def resume(
        self,
        spec: AgentRunSpec,
        session_id: str,
        feedback: str,
        usage: Any = None,
    ) -> AgentRunResult:
        """Continue an exact prior SDK session with a new prompt.

        The SDK reloads that session's full conversation, so the model answers
        with the earlier turns in context. ``spec`` still governs THIS turn's
        tools, hooks, and system prompt, which lets a caller resume a writable
        implementer session under a read-only policy (see the lesson summarizer).
        """
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
        # Armed before the CLI starts rather than when the reaper runs: the
        # ownership tag is only inherited by children exec'd after it is set,
        # and the subreaper flag is what keeps a detached benchmark traceable to
        # this session once the shell that started it has exited.
        install_child_subreaper()
        # These worktrees carry the loop's own ledger and build output, so a
        # clean-HEAD demand would refuse every session before it started.
        guard = WorkspaceGuard(spec, dirty_baseline_default=True)
        guard.prepare()
        provider_options = self._provider_options(spec)
        if resume_session_id:
            provider_options["resume"] = resume_session_id
        options = self._options_type(**provider_options)
        text_parts: list[str] = []
        tool_calls: list[tuple[str, dict[str, Any]]] = []
        subtype = ""
        num_turns: int | None = None
        session_id = ""

        # The SDK RAISES on the turn cap (and some other mid-session failures)
        # rather than yielding a final ResultMessage, and its stream is an
        # unbounded ``async for`` -- nothing here stops a session that neither
        # answers nor caps. Both are handled the same way: bound the stream with
        # the spec's wall-clock budget, and if that budget or the turn cap trips
        # after a session id exists, capture it instead of unwinding. The caller
        # registers the resume handle only AFTER run() returns (see
        # orchestrator.agent), so an exception that escapes this method loses the
        # handle and the session can never be resumed to write a full lesson --
        # the exact "provider cannot resume" path that produced outcome-only
        # lessons. The init message carries the session id, so it is set before
        # any mid-session failure; return a normal result carrying it and let
        # the outer loop validate the on-disk candidate AND resume THIS session.
        # Only a failure that preceded the session (no id yet, nothing to
        # resume) still raises.
        stream_error: Exception | None = None
        timed_out = False
        # Independent of ``timed_out``: the CLI subprocess and any detached
        # benchmark children exist the moment the deadline fires, even before a
        # session id is established, so they must be torn down on that path too
        # -- otherwise a hung init leaks the process group and keeps the GPU.
        reap_on_exit = False
        # Set on the paths that leave without a result. The guard puts the
        # workspace back, but only after the reap below: restoring files while a
        # detached child is still writing them would undo the restore.
        rollback_on_exit = False
        # What the reap could not clear out of the workspace. Non-empty means
        # something is still holding the device, so the measurement that follows
        # this session would be measuring it too.
        contention = ""
        agen = self._query(prompt=prompt, options=options)
        # ``asyncio.timeout(None)`` applies no bound, so a spec without a budget
        # keeps the previous unbounded behaviour; ``expired()`` tells our own
        # deadline apart from a bare TimeoutError surfacing from the transport.
        deadline = _session_deadline(spec.timeout_sec)
        try:
            try:
                async with deadline:
                    async for message in agen:
                        if usage is not None:
                            usage.add_from_message(message)
                        _record_progress(spec.progress_log, message)
                        # The init SystemMessage and the final ResultMessage both
                        # carry the session id; keep the latest non-empty one so a
                        # caller can resume this exact conversation later.
                        candidate_session = getattr(message, "session_id", "") or ""
                        if isinstance(candidate_session, str) and candidate_session:
                            session_id = candidate_session
                        if hasattr(message, "total_cost_usd"):
                            subtype = getattr(message, "subtype", "") or ""
                            turns = getattr(message, "num_turns", None)
                            if isinstance(turns, int):
                                num_turns = turns
                        if hasattr(message, "content"):
                            for block in message.content:
                                if hasattr(block, "text"):
                                    text_parts.append(block.text)
                                elif block.__class__.__name__ == "ToolUseBlock":
                                    tool_calls.append(
                                        (
                                            getattr(block, "name", "?"),
                                            getattr(block, "input", {}) or {},
                                        )
                                    )
            except Exception as exc:  # noqa: BLE001 - convert to a resumable result
                if deadline.expired():
                    reap_on_exit = True
                    if not session_id:
                        rollback_on_exit = True
                        raise ClaudeTimeoutError(
                            f"Claude session timed out after {spec.timeout_sec}s before it established a session"
                        ) from exc
                    timed_out = True
                    stream_error = TimeoutError(f"Claude session timed out after {spec.timeout_sec}s")
                    log.warning(
                        "Claude session %s timed out after %ss; preserving the resume handle and reaping its leftovers",
                        session_id,
                        spec.timeout_sec,
                    )
                    if spec.progress_log is not None:
                        spec.progress_log.append(f"end: timeout {spec.timeout_sec}s")
                else:
                    if not session_id:
                        rollback_on_exit = True
                        raise
                    stream_error = exc
                    if not _is_turn_cap_error(exc):
                        log.warning(
                            "Claude SDK stream failed after session %s; preserving the resume handle (%s: %s)",
                            session_id,
                            type(exc).__name__,
                            exc,
                        )
                    if spec.progress_log is not None:
                        spec.progress_log.append(f"end: sdk-error {str(exc)[:_PROGRESS_TEXT_CHARS]}")
        except asyncio.CancelledError:
            # The subprocess and any detached benchmark children are running
            # regardless of whether a session id was established; reap them so
            # they do not hold the GPU past this point. The caller receives the
            # exception, but any workspace writes and the resume handle survive.
            reap_on_exit = True
            if not session_id:
                rollback_on_exit = True
            raise
        finally:
            if reap_on_exit:
                # ``async for`` does not close its iterator on exit (PEP 533 was
                # deferred), so close it to tear the CLI down, then reap any
                # detached benchmark child that outlived it -- an orphan holding
                # the GPU corrupts the canonical measurement that follows. This
                # runs whether or not a session id was established: a deadline
                # that fires during a hung init still left a process group.
                with suppress(Exception):
                    await agen.aclose()
                report = await _reap_workspace_processes(spec.cwd)
                if report.contended:
                    contention = report.describe()
            if rollback_on_exit:
                guard.rollback()

        if timed_out:
            subtype = subtype or "error_timeout"
            end_reason = "timeout"
            text_parts.append(f"[session ended: {stream_error}]")
        elif stream_error is not None:
            if _is_turn_cap_error(stream_error):
                subtype = subtype or "error_max_turns"
                end_reason = "turn_cap"
            else:
                subtype = subtype or "error"
                end_reason = "sdk_error"
            text_parts.append(f"[session ended with SDK error: {stream_error}]")
        elif "max_turns" in subtype:
            end_reason = "turn_cap"
        elif subtype and subtype != "success":
            end_reason = f"sdk_{subtype}"
        else:
            end_reason = "agent_stopped"

        result = AgentRunResult(
            text="\n".join(text_parts).strip(),
            subtype=subtype,
            num_turns=num_turns,
            end_reason=end_reason,
            session_id=session_id,
            tool_calls=tool_calls,
            stderr_tail=(str(stream_error)[:2000] if stream_error else ""),
            workspace_contention=contention,
        )
        try:
            result.file_changes = guard.verify()
        except Exception:
            # verify() restores the baseline itself before raising, so this
            # covers the paths that fail earlier and must not mask them.
            with suppress(Exception):
                guard.rollback()
            raise
        result.target_edit_count = guard.count_target_edits()
        result.edit_count = result.target_edit_count
        return result


__all__ = [
    "ClaudeBackend",
    "ClaudeBackendError",
    "ClaudeTimeoutError",
    "ClaudeUnavailableError",
    "DEFAULT_CLAUDE_MODEL",
    "FALLBACK_CLAUDE_MODEL",
    "resolve_claude_cli",
]
