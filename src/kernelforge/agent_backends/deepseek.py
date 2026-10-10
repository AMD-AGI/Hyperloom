# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DeepSeek Harness Python SDK backend for Forge implementer sessions."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from hyperloom.common.reasoning_effort import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_LEVELS,
    harness_reasoning_effort,
)
from kernelforge.agent_backends.base import (
    AgentCapabilities,
    AgentProviderError,
    AgentProviderUnavailableError,
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
)
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard
from kernelforge.llm import LlmGateway, resolve_openai_gateway

log = logging.getLogger(__name__)

#: The route the harness mounts for DeepSeek models in its own default plugin
#: composition. Naming any other provider needs a composition that mounts an
#: adapter owning it, which initialization refuses to guess at.
DEEPSEEK_PROVIDER_ROUTE = "deepseek-official"

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


class DeepSeekHarnessBackendError(AgentProviderError):
    """Base error for DeepSeek Harness backend failures."""


class DeepSeekHarnessUnavailableError(
    DeepSeekHarnessBackendError,
    AgentProviderUnavailableError,
):
    """Report a missing SDK or incomplete gateway configuration."""


class DeepSeekHarnessExecutionError(DeepSeekHarnessBackendError):
    """Report a failed or timed-out DeepSeek Harness session."""

    def __init__(self, *args: Any, session_id: str = "") -> None:
        super().__init__(*args)
        self.session_id = str(session_id or "")


def _load_harness_sdk() -> Any:
    """Load the optional DeepSeek Harness SDK only when this provider is selected."""
    try:
        import deepseek_harness
    except ImportError as exc:
        raise DeepSeekHarnessUnavailableError(
            "DeepSeek Harness Python SDK is not installed; install kernelforge[deepseek]"
        ) from exc
    return deepseek_harness


def resolve_deepseek_model(explicit: str = "") -> str:
    """Resolve a DeepSeek model from generic runtime input or built-in default."""
    return explicit.strip() or DEFAULT_DEEPSEEK_MODEL


def resolve_deepseek_reasoning_effort(explicit: str = "") -> str:
    """Settle the effort a harness session runs at, refusing anything off-ladder."""
    effort = harness_reasoning_effort(explicit or DEFAULT_REASONING_EFFORT)
    if not effort:
        raise DeepSeekHarnessExecutionError(
            f"unsupported DeepSeek Harness reasoning effort: {explicit!r}; "
            f"expected one of {', '.join(REASONING_EFFORT_LEVELS)}"
        )
    return effort


def _harness_instructions(spec: AgentRunSpec) -> str:
    """Adapt shared system instructions to DeepSeek Harness capabilities.

    The harness owns its system prompt through its plugin composition and its
    SDK exposes no per-session field for one, so the caller's instructions ride
    the first message of the session instead. Its own persona template
    interpolates ``{{name}}`` references and fails assembly on an unknown one,
    which a kernel prompt carrying brace-heavy source would trip; a message
    body is not interpolated at all.
    """
    write_guidance = (
        "Use the write and edit tools for the files the current request explicitly allows you to change."
        if spec.writable
        else "This is a read-only session. Do not edit, create, delete, or rename any file."
    )
    runtime = f"""\
## DeepSeek Harness runtime mapping
Use the read, glob and grep tools to inspect and search files, and bash to run
commands. References below to Read, Edit, Write, Grep, Glob, or Bash describe
equivalent capabilities, not literal tool names. {write_guidance}

Do not run git commands that change repository state. Do not commit, reset,
checkout, stash, clean, or alter branches. The Forge loop owns all git state.
Follow the task-specific write scope and output format below exactly. Stop when
that task is complete; do not continue open-ended exploration.
"""
    return f"{runtime}\n## System instructions\n{spec.system_prompt}"


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum the per-step token counts the harness reports into Forge fields.

    Only ``inputTokens`` and ``outputTokens`` are added up. The harness reports
    ``totalTokens`` as a running total for the whole turn, so summing that field
    across steps would count the same prefix once per step.
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    wire_names = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "cacheReadTokens": "cache_read_input_tokens",
    }
    seen = False
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "assistant/message":
            continue
        data = event.get("data")
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            continue
        seen = True
        for wire, field in wire_names.items():
            value = usage.get(wire)
            if isinstance(value, bool):
                continue
            with contextlib.suppress(TypeError, ValueError):
                parsed = int(value)
                if parsed >= 0:
                    totals[field] += parsed
    return totals if seen else {}


def _normalize_run_result(result: Any, session_id: str) -> AgentRunResult:
    """Normalize one completed harness activity interval into the shared result."""
    text = str(getattr(result, "final_response", "") or "").strip()
    events = [event for event in (getattr(result, "events", ()) or ()) if isinstance(event, dict)]
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    for event in events:
        if event.get("type") != "tool/call":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        tool_calls.append(
            (
                str(data.get("name") or "tool"),
                {key: data[key] for key in ("callId", "step") if key in data},
            )
        )

    findings: list[str] = []
    stderr_tail = ""
    subtype = "success"
    end_reason = "agent_stopped"
    # ``finish_reason`` is the kind of the last turn/end in the interval: the harness reports a provider failure or a
    # spent output cap there while the run itself completes, so an unexamined result reads as a session that simply
    # had nothing to say.
    finish_reason = str(getattr(result, "finish_reason", "") or "")
    if finish_reason == "max-tokens":
        subtype = "error_max_turns"
        end_reason = "turn_cap"
    elif finish_reason and finish_reason != "completed":
        message = _turn_end_failure(events) or f"session ended with turn/end {finish_reason!r}"
        findings.append(message)
        stderr_tail = message[:2000]
        subtype = "error"
        end_reason = "sdk_error"
        text = text or f"[session ended with harness error: {message}]"

    return AgentRunResult(
        text=text,
        subtype=subtype,
        num_turns=1,
        end_reason=end_reason,
        session_id=str(getattr(result, "session_id", "") or session_id),
        tool_calls=tool_calls,
        usage=_usage_from_events(events),
        findings=findings,
        stderr_tail=stderr_tail,
    )


def _turn_end_failure(events: list[dict[str, Any]]) -> str:
    """Return the message of the last failing ``turn/end``, if it carries one."""
    for event in reversed(events):
        if event.get("type") != "turn/end":
            continue
        data = event.get("data")
        reason = data.get("reason") if isinstance(data, dict) else None
        error = reason.get("error") if isinstance(reason, dict) else None
        if isinstance(error, dict):
            code = str(error.get("code") or "").strip()
            message = str(error.get("message") or "").strip()
            return f"{code}: {message}" if code and message else message or code
        break
    return ""


class DeepSeekHarnessBackend:
    """Execute Forge sessions through the DeepSeek Harness Python SDK."""

    name = "deepseek"
    capabilities = AgentCapabilities(
        writable=True,
        resumable=True,
        sandbox=True,
        probe=True,
        requires_workspace_cwd=True,
        session_env=True,
        workspace_guard=True,
    )

    def __init__(
        self,
        gateway: LlmGateway | None = None,
        runtime: AgentRuntimeConfig | None = None,
    ) -> None:
        """Capture transport overrides while deferring checks until execution."""
        self.runtime = runtime or AgentRuntimeConfig(
            provider=self.name,
            model=DEFAULT_DEEPSEEK_MODEL,
        )
        self.gateway = gateway
        self._preflight_done = False
        self._runtimes: dict[str, Any] = {}
        # Which runtime ran each session: only that process can resume it, whatever spec the resume carries.
        self._session_runtimes: dict[str, Any] = {}
        self._runtimes_lock = threading.Lock()

    def _effective_gateway(self) -> LlmGateway:
        """Return the explicit override, else what the environment configures.

        The OpenAI-compatible line, not a ``DEEPSEEK_*`` one: that spelling is
        retired in this repository and the preflight rewrites it into the two
        standard lines, so reading it here would resurrect a variable an
        operator is being migrated off. DeepSeek's own wire format is
        OpenAI-compatible chat completions, so the line that already carries a
        base URL and key for that protocol is the one this route speaks.
        """
        if self.gateway is not None and self.gateway.is_complete():
            return self.gateway
        return resolve_openai_gateway()

    def _require_gateway(self) -> LlmGateway:
        """Return a complete gateway or explain which half is missing."""
        gateway = self._effective_gateway()
        if not gateway.is_complete():
            raise DeepSeekHarnessUnavailableError(
                "DeepSeek Harness gateway is not configured; it speaks the "
                "OpenAI-compatible protocol and needs both OPENAI_BASE_URL and "
                "OPENAI_API_KEY. The ANTHROPIC_* line belongs to Claude and is "
                "not a substitute."
            )
        return gateway

    def _harness_home(self) -> Path:
        """Resolve the harness home holding profiles, plugins and session logs.

        The SDK refuses to discover ``~/.dsh`` and requires this to be named, so
        that a campaign's durable session logs cannot land in whatever home an
        unrelated interactive install left behind.
        """
        configured = str(self.runtime.options.get("home", "")).strip()
        home = (
            Path(configured).expanduser()
            if configured
            else Path.home() / ".cache" / "kernelforge" / "deepseek_harness_home"
        )
        home.mkdir(parents=True, exist_ok=True)
        return home.resolve()

    def _permission_mode(self, spec: AgentRunSpec) -> str:
        """Map the generic runtime sandbox onto a harness permission preset.

        A writable session gets ``danger-full-access`` rather than
        ``workspace-write`` because the harness pairs the latter with an
        ``ask`` approval policy, and an SDK client registers no approval
        answerer: every write would fail closed on a request nobody can answer.
        Confinement for a writable session is :class:`WorkspaceGuard`'s to
        enforce, which judges the edits against the spec's declared targets and
        restores the baseline when they deviate.
        """
        if not spec.writable or self.runtime.sandbox_mode.strip().lower() == "read-only":
            return "read-only"
        return "danger-full-access"

    def _child_environment(self, spec: AgentRunSpec) -> dict[str, str]:
        """Build the environment the harness runtime subprocess is spawned with.

        ``DEEPSEEK_*`` is what the harness's own DeepSeek adapter reads, so it
        is written here for the subprocess only. It is never read back from the
        parent environment: see :meth:`_effective_gateway`.
        """
        gateway = self._require_gateway()
        env = os.environ.copy()
        env.update(spec.env)
        env["DEEPSEEK_BASE_URL"] = gateway.base_url
        env["DEEPSEEK_API_KEY"] = os.environ.get(gateway.key_env, "")
        env["DSH_PERMISSION_MODE"] = self._permission_mode(spec)
        # These outrank -C and cwd, so a caller that set them to reach one repository would answer every git command
        # the session runs anywhere.
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)
        return env

    def _runtime_key(self, spec: AgentRunSpec) -> str:
        """Key one harness process by everything fixed at its initialization.

        The workspace, the model, the effort and the permission mode are all
        sent during the JSON-RPC handshake or read from the subprocess
        environment, so two sessions differing in any of them cannot share one
        runtime. The system prompt is deliberately absent: it rides each
        session's first message, which is what lets the roles of one campaign
        share a process instead of each paying for its own Node runtime.
        """
        env = "\x00".join(f"{name}={value}" for name, value in sorted(spec.env.items()))
        return "\x00".join(
            (
                str(Path(spec.cwd).resolve()),
                resolve_deepseek_model(spec.model),
                resolve_deepseek_reasoning_effort(spec.reasoning_effort),
                self._permission_mode(spec),
                env,
            )
        )

    def _harness_for(self, spec: AgentRunSpec) -> Any:
        """Return the harness runtime serving ``spec``, starting one if needed.

        Held for the backend's lifetime rather than per session because the SDK
        cannot adopt a persisted session from a second process: a later
        ``resume`` against an id this backend already ran is answered with
        ``session "<id>" already exists``. One live runtime per key is what
        makes :meth:`resume` continue the durable conversation instead of
        starting a fresh one that has never seen the work it is judging.
        """
        sdk = _load_harness_sdk()
        key = self._runtime_key(spec)
        with self._runtimes_lock:
            harness = self._runtimes.get(key)
            if harness is not None:
                return harness
            # A deleted workspace (a finished lane round) can host no resume, so its runtime only holds a process.
            orphaned = [k for k in self._runtimes if not Path(k.split("\x00", 1)[0]).is_dir()]
            stale = [self._runtimes.pop(k) for k in orphaned]
            self._forget_sessions_of(stale)
            workspace = str(Path(spec.cwd).resolve())
            harness = sdk.DeepSeekHarness(
                dsh_home=str(self._harness_home()),
                cwd=workspace,
                # The harness reads its sandbox root from the subprocess working directory, so a runtime rooted
                # anywhere else would confine a read-only session to the wrong tree.
                runtime_cwd=workspace,
                provider=DEEPSEEK_PROVIDER_ROUTE,
                model=resolve_deepseek_model(spec.model),
                reasoning_effort=resolve_deepseek_reasoning_effort(spec.reasoning_effort),
                env=self._child_environment(spec),
                initialize_timeout_seconds=float(self.runtime.options.get("initialize_timeout_sec", 120)),
                request_timeout_seconds=float(spec.timeout_sec or self.runtime.timeout_sec),
            )
            self._runtimes[key] = harness
        for runtime in stale:
            with contextlib.suppress(Exception):
                runtime.close()
        return harness

    def preflight(self) -> None:
        """Validate SDK and gateway prerequisites without spending tokens."""
        if self._preflight_done:
            return
        _load_harness_sdk()
        self._require_gateway()
        self._preflight_done = True

    def probe(self, *, cwd: str, usage: Any = None) -> None:
        """Start one harness runtime so a misconfigured route fails before a campaign.

        Initialization is where the harness rejects a missing adapter, an
        unavailable model, or an effort the selected route does not support, so
        this turns those into a launch-time error instead of one raised hours in
        by the first session that happened to need them.
        """
        del usage
        self.preflight()
        spec = AgentRunSpec(system_prompt="", user_prompt="", cwd=cwd)
        try:
            self._harness_for(spec.resolved(self.runtime))
        except DeepSeekHarnessBackendError:
            raise
        except Exception as exc:
            raise DeepSeekHarnessUnavailableError(f"DeepSeek Harness initialization failed: {exc}") from exc

    def _discard_harness(self, spec: AgentRunSpec, harness: Any) -> None:
        """Close one runtime and stop serving anything from it."""
        del spec
        with self._runtimes_lock:
            for key in [k for k, runtime in self._runtimes.items() if runtime is harness]:
                del self._runtimes[key]
            self._forget_sessions_of([harness])
        with contextlib.suppress(Exception):
            harness.close()

    def _forget_sessions_of(self, runtimes: list[Any]) -> None:
        """Drop the sessions owned by runtimes being closed; the caller holds the lock."""
        for session_id in [s for s, runtime in self._session_runtimes.items() if any(runtime is r for r in runtimes)]:
            del self._session_runtimes[session_id]

    def _owning_harness(self, session_id: str) -> Any:
        """The live runtime that already ran ``session_id``, if any."""
        with self._runtimes_lock:
            return self._session_runtimes.get(session_id)

    def close(self) -> None:
        """Shut down every harness runtime this backend started."""
        with self._runtimes_lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
            self._session_runtimes.clear()
        for harness in runtimes:
            with contextlib.suppress(Exception):
                harness.close()

    async def _execute(
        self,
        *,
        spec: AgentRunSpec,
        prompt: str,
        session_id: str,
        usage: Any = None,
    ) -> AgentRunResult:
        """Execute one guarded harness turn and normalize its result."""
        self.preflight()
        guard = WorkspaceGuard(spec)
        guard.prepare()
        try:
            harness = self._owning_harness(session_id) or self._harness_for(spec)
            with self._runtimes_lock:
                self._session_runtimes[session_id] = harness
            run = asyncio.to_thread(harness.run, prompt, session_id=session_id)
            try:
                sdk_result = await asyncio.wait_for(run, timeout=spec.timeout_sec)
            except asyncio.TimeoutError as exc:
                # The SDK has no per-turn cancel, so the turn is still editing the workspace the guard is about to
                # restore. Closing its runtime is the only way to stop it first.
                await asyncio.to_thread(self._discard_harness, spec, harness)
                raise DeepSeekHarnessExecutionError(
                    f"DeepSeek Harness timed out after {spec.timeout_sec}s",
                    session_id=session_id,
                ) from exc
        except asyncio.CancelledError:
            guard.rollback()
            raise
        except DeepSeekHarnessBackendError as exc:
            guard.rollback()
            if not getattr(exc, "session_id", ""):
                with contextlib.suppress(AttributeError):
                    exc.session_id = session_id
            raise
        except Exception as exc:
            guard.rollback()
            raise DeepSeekHarnessExecutionError(
                f"DeepSeek Harness SDK execution failed: {exc}",
                session_id=session_id,
            ) from exc

        result = _normalize_run_result(sdk_result, session_id)
        if usage is not None:
            usage.add_usage(result.usage, role=spec.role)
        try:
            actual_changes = guard.verify()
        except Exception:
            # verify() restores the baseline itself before raising a rejection, so this second call only covers the
            # paths that fail before it gets there.
            with contextlib.suppress(Exception):
                guard.rollback()
            raise
        result.file_changes = actual_changes
        result.target_edit_count = guard.count_target_edits()
        result.edit_count = max(result.edit_count, result.target_edit_count)
        return result

    async def run(self, spec: AgentRunSpec, usage: Any = None) -> AgentRunResult:
        """Run one new DeepSeek Harness session."""
        spec = spec.resolved(self.runtime)
        if spec.progress_log is not None and not spec.progress_log:
            spec.progress_log.append("progress: not supported by deepseek backend")
        return await self._execute(
            spec=spec,
            prompt=_harness_instructions(spec) + "\n\n## Task\n" + spec.user_prompt,
            session_id=f"forge-{uuid.uuid4()}",
            usage=usage,
        )

    async def resume(
        self,
        spec: AgentRunSpec,
        session_id: str,
        feedback: str,
        usage: Any = None,
    ) -> AgentRunResult:
        """Resume an exact harness session with canonical Forge gate feedback."""
        if not session_id.strip():
            raise DeepSeekHarnessExecutionError("DeepSeek Harness resume requires a session ID")
        spec = spec.resolved(self.runtime)
        result = await self._execute(
            spec=spec,
            prompt=feedback,
            session_id=session_id,
            usage=usage,
        )
        if not result.session_id:
            result.session_id = session_id
        return result


__all__ = [
    "DEEPSEEK_PROVIDER_ROUTE",
    "DEFAULT_DEEPSEEK_MODEL",
    "DeepSeekHarnessBackend",
    "DeepSeekHarnessBackendError",
    "DeepSeekHarnessExecutionError",
    "DeepSeekHarnessUnavailableError",
    "resolve_deepseek_model",
    "resolve_deepseek_reasoning_effort",
]
