# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DeepSeek Harness Python SDK backend for Forge implementer sessions.

The runtime layer (SDK loading, gateway environment, handshake, result
normalization) is :mod:`hyperloom_common.llm.deepseek_session`; this module adds
what Forge needs on top of it: the workspace guard, resumable sessions and a
cache of live runtimes.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hyperloom_common.llm.deepseek_session import (
    DEEPSEEK_RUNTIME_MAPPING,
    DeepSeekSessionUnavailableError,
    HarnessLaunch,
    child_environment,
    load_harness_sdk,
    normalize_run_result,
    sdk_failures,
    start_harness,
)
from hyperloom_common.llm.llm_config import DEFAULT_DEEPSEEK_HARNESS_MODEL
from hyperloom_common.llm.reasoning_effort import (
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


_READ_ONLY = "read-only"


@contextlib.contextmanager
def _as_forge_unavailable() -> Iterator[None]:
    """Re-raise the shared layer's unavailability as this provider's own error."""
    try:
        yield
    except DeepSeekSessionUnavailableError as exc:
        raise DeepSeekHarnessUnavailableError(str(exc)) from exc


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
    runtime = (
        f"{DEEPSEEK_RUNTIME_MAPPING}\n{write_guidance} The Forge loop owns all git state.\n"
        "Follow the task-specific write scope and output format below exactly. Stop when\n"
        "that task is complete; do not continue open-ended exploration.\n"
    )
    return f"{runtime}\n## System instructions\n{spec.system_prompt}"


def _normalize_run_result(result: Any, session_id: str) -> AgentRunResult:
    """Fold one completed harness activity interval into the shared Forge result."""
    shared = normalize_run_result(result, session_id)
    events = [event for event in (getattr(result, "events", ()) or ()) if isinstance(event, dict)]
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    for event in events:
        data = event.get("data")
        if event.get("type") != "tool/call" or not isinstance(data, dict):
            continue
        tool_calls.append(
            (
                str(data.get("name") or "tool"),
                {key: data[key] for key in ("callId", "step") if key in data},
            )
        )

    text = shared.text
    findings: list[str] = []
    stderr_tail = ""
    subtype = "success"
    end_reason = "agent_stopped"
    if shared.finish_reason == "max-tokens":
        subtype = "error_max_turns"
        end_reason = "turn_cap"
    elif shared.error:
        findings.append(shared.error)
        stderr_tail = shared.error[:2000]
        subtype = "error"
        end_reason = "sdk_error"
        text = text or f"[session ended with harness error: {shared.error}]"

    return AgentRunResult(
        text=text,
        subtype=subtype,
        num_turns=1,
        end_reason=end_reason,
        session_id=shared.session_id,
        tool_calls=tool_calls,
        usage=shared.usage,
        findings=findings,
        stderr_tail=stderr_tail,
    )


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

    def __init__(self, runtime: AgentRuntimeConfig | None = None) -> None:
        """Capture runtime overrides while deferring checks until execution."""
        self.runtime = runtime or AgentRuntimeConfig(
            provider=self.name,
            model=DEFAULT_DEEPSEEK_HARNESS_MODEL,
        )
        self._preflight_done = False
        self._runtimes: dict[str, Any] = {}
        # Which runtime ran each session: only that process can resume it, whatever spec the resume carries.
        self._session_runtimes: dict[str, Any] = {}
        self._runtimes_lock = threading.Lock()

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
        if not spec.writable or self.runtime.sandbox_mode.strip().lower() == _READ_ONLY:
            return _READ_ONLY
        return "danger-full-access"

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
                spec.model,
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
        key = self._runtime_key(spec)
        with self._runtimes_lock:
            harness = self._runtimes.get(key)
            if harness is not None:
                return harness
            # A deleted workspace (a finished lane round) can host no resume, so its runtime only holds a process.
            orphaned = [k for k in self._runtimes if not Path(k.split("\x00", 1)[0]).is_dir()]
            stale = [self._runtimes.pop(k) for k in orphaned]
            self._forget_sessions_of(stale)
        for runtime in stale:
            with contextlib.suppress(*sdk_failures()):
                runtime.close()
        started = self._start_harness(spec)
        with self._runtimes_lock:
            winner = self._runtimes.setdefault(key, started)
        if winner is not started:
            with contextlib.suppress(*sdk_failures()):
                started.close()
        return winner

    def _start_harness(self, spec: AgentRunSpec) -> Any:
        """Spawn one runtime for ``spec`` and run its handshake."""
        with _as_forge_unavailable():
            sdk = load_harness_sdk()
            env = child_environment(spec.env, self._permission_mode(spec))
        try:
            launch = HarnessLaunch(
                dsh_home=self._harness_home(),
                cwd=Path(spec.cwd).resolve(),
                model=spec.model,
                reasoning_effort=resolve_deepseek_reasoning_effort(spec.reasoning_effort),
                env=env,
                request_timeout_sec=float(spec.timeout_sec or self.runtime.timeout_sec),
                initialize_timeout_sec=float(self.runtime.options.get("initialize_timeout_sec", 120)),
            )
            return start_harness(sdk, launch)
        except sdk_failures() as exc:
            raise DeepSeekHarnessUnavailableError(f"DeepSeek Harness initialization failed: {exc}") from exc

    def preflight(self) -> None:
        """Validate SDK and gateway prerequisites without spending tokens."""
        if self._preflight_done:
            return
        with _as_forge_unavailable():
            load_harness_sdk()
            child_environment(None, _READ_ONLY)
        self._preflight_done = True

    def probe(self, *, cwd: str, usage: Any = None) -> None:
        """Start one harness runtime and run its handshake, so a misconfigured route fails before a campaign.

        The handshake is where the harness rejects a missing binary, an
        unavailable model, or an effort the selected route does not support, so
        this turns those into a launch-time error instead of one raised hours in
        by the first session that happened to need them.
        """
        del usage
        self.preflight()
        spec = AgentRunSpec(system_prompt="", user_prompt="", cwd=cwd)
        self._harness_for(spec.resolved(self.runtime))

    def close(self) -> None:
        """Shut down every runtime this backend started; a later session starts a fresh one."""
        with self._runtimes_lock:
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
            self._session_runtimes.clear()
        for runtime in runtimes:
            with contextlib.suppress(*sdk_failures()):
                runtime.close()

    def _discard_harness(self, harness: Any) -> None:
        """Close one runtime and stop serving anything from it."""
        with self._runtimes_lock:
            for key in [k for k, runtime in self._runtimes.items() if runtime is harness]:
                del self._runtimes[key]
            self._forget_sessions_of([harness])
        with contextlib.suppress(*sdk_failures()):
            harness.close()

    def _forget_sessions_of(self, runtimes: list[Any]) -> None:
        """Drop the sessions owned by runtimes being closed; the caller holds the lock."""
        for session_id in [s for s, runtime in self._session_runtimes.items() if any(runtime is r for r in runtimes)]:
            del self._session_runtimes[session_id]

    def _owning_harness(self, session_id: str) -> Any:
        """The live runtime that already ran ``session_id``, if any."""
        with self._runtimes_lock:
            return self._session_runtimes.get(session_id)

    async def _run_turn(self, harness: Any, prompt: str, session_id: str, spec: AgentRunSpec) -> Any:
        """Run one turn, closing the runtime when it ends in anything but a result.

        The SDK has no per-turn cancel, so a turn that is abandoned keeps
        editing the workspace the guard is about to restore. Closing its runtime
        is the only way to stop it, and a runtime that failed once is never
        served to a later session.
        """
        run = asyncio.to_thread(harness.run, prompt, session_id=session_id)
        try:
            return await asyncio.wait_for(run, timeout=spec.timeout_sec)
        except asyncio.TimeoutError as exc:
            await asyncio.to_thread(self._discard_harness, harness)
            raise DeepSeekHarnessExecutionError(
                f"DeepSeek Harness timed out after {spec.timeout_sec}s",
                session_id=session_id,
            ) from exc
        except asyncio.CancelledError:
            await asyncio.to_thread(self._discard_harness, harness)
            raise
        except Exception as exc:
            await asyncio.to_thread(self._discard_harness, harness)
            raise DeepSeekHarnessExecutionError(
                f"DeepSeek Harness SDK execution failed: {exc}",
                session_id=session_id,
            ) from exc

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
            harness = self._owning_harness(session_id) or await asyncio.to_thread(self._harness_for, spec)
            with self._runtimes_lock:
                self._session_runtimes[session_id] = harness
            sdk_result = await self._run_turn(harness, prompt, session_id, spec)
        except BaseException:
            guard.rollback()
            raise

        result = _normalize_run_result(sdk_result, session_id)
        if usage is not None:
            usage.add_usage(result.usage, role=spec.role)
        try:
            actual_changes = guard.verify()
        except Exception:
            # verify() restores the baseline itself before raising a rejection, so this second call only covers the
            # paths that fail before it gets there.
            with contextlib.suppress(Exception):  # broad-suppress: rollback must not shadow the verify error
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
        return await self._execute(
            spec=spec.resolved(self.runtime),
            prompt=feedback,
            session_id=session_id,
            usage=usage,
        )


__all__ = [
    "DeepSeekHarnessBackend",
    "DeepSeekHarnessBackendError",
    "DeepSeekHarnessExecutionError",
    "DeepSeekHarnessUnavailableError",
    "resolve_deepseek_reasoning_effort",
]
