# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Provider-neutral contracts for Forge agent execution backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


#: Environment overlay applied to every session started inside the current
#: context. Held in a ``ContextVar`` rather than in ``os.environ`` because
#: several sessions run concurrently in one process: each asyncio task carries
#: its own copy of the context, so one task's overlay is invisible to its
#: siblings, while an ``os.environ`` write would be the last writer's for all of
#: them. Read by :meth:`AgentRunSpec.resolved`.
_session_environment: ContextVar[Mapping[str, str]] = ContextVar(
    "forge_agent_session_environment",
    default={},
)


@contextmanager
def session_environment(overlay: Mapping[str, str]) -> Iterator[None]:
    """Give the sessions started in this context their own environment overlay.

    For callers that own a session but do not build its :class:`AgentRunSpec`:
    the Implementer lanes are constructed by the shared implementer factory, so
    a lane can reach ``AgentRunSpec.env`` no other way. Nested scopes replace
    rather than merge, and the spec's own ``env`` wins over the overlay.
    """
    token = _session_environment.set(dict(overlay))
    try:
        yield
    finally:
        _session_environment.reset(token)


#: Attribute a provider sets to ``True`` on an error that is a VERDICT about
#: what a session did to the workspace, as opposed to the provider failing at its
#: own bookkeeping. Callers classify by this attribute rather than by class name,
#: because a provider raises one class for both: a snapshot it could not read or
#: a Git query that timed out on NFS says nothing about the session and recovers
#: on its own, while matching on the name made such a failure abandon the work.
AGENT_SAFETY_REJECTION_ATTR = "agent_safety_rejection"


class AgentProviderError(RuntimeError):
    """Base error raised by a registered Agent provider.

    A provider that can reject a session for what it did to the workspace must
    mark that error with :data:`AGENT_SAFETY_REJECTION_ATTR` set to ``True``, and
    must leave it unset (or ``False``) on errors that merely report the provider
    failing at its own bookkeeping. Callers abandon the work on the first and
    retry the second. An error that carries neither is read as retryable, which
    is the recoverable mistake: retrying a genuine rejection costs one attempt,
    while abandoning a recipe over a transient failure discards work that would
    have finished.
    """


class AgentProviderUnavailableError(AgentProviderError):
    """Report a provider that cannot run in the current environment."""


@dataclass(frozen=True)
class AgentCapabilities:
    """Declare optional features implemented by one Agent provider."""

    writable: bool = True
    resumable: bool = False
    # Whether the provider runs the callbacks in ``AgentRunSpec.hooks``. Named
    # after one of the three groups but deciding all of them: the Claude backend
    # translates PreToolUse, PostToolUse and Stop through a single path keyed on
    # ``spec.hooks is not None``, so a provider either runs the whole hook
    # mechanism or none of it, and no caller can ask for one group by itself.
    # The name is therefore narrower than what the flag decides.
    stop_hooks: bool = False
    native_subagents: bool = False
    # Whether the provider judges what the session did to the workspace: edits
    # outside its targets, a moved HEAD, a changed protected measurement file.
    workspace_guard: bool = False
    mcp: bool = False
    sandbox: bool = False
    probe: bool = False
    requires_workspace_cwd: bool = False
    # Whether the provider applies ``AgentRunSpec.env`` over the environment it
    # spawns the session with. Several sessions can run side by side in one
    # Forge process, where a per-session value cannot be routed through
    # ``os.environ`` -- the last write would be every session's -- so this is
    # the only way two concurrent sessions get different values for the same
    # variable. A provider that ignores ``env`` puts every Implementer lane back
    # into one AITER build cache, where aiter imports a module by name and a
    # lane can measure a binary a sibling compiled.
    session_env: bool = False


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Hold provider-neutral runtime configuration for one selected Agent CLI."""

    provider: str
    model: str
    fallback_model: str = ""
    executable: str = ""
    timeout_sec: int = 1800
    reasoning_effort: str = "high"
    sandbox_mode: str = "bypass"
    precheck: bool = True
    fallback_provider: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate generic runtime values without imposing provider semantics."""
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")


def with_writable_sandbox(runtime: AgentRuntimeConfig) -> AgentRuntimeConfig:
    """Return ``runtime`` permitted to write, without loosening it any further.

    A turn that authors a file cannot run under ``read-only``. Assigning
    ``workspace-write`` to say so also *lowers* ``bypass``, which is not a
    weaker form of the same permission but the operator's statement that this
    process is already isolated externally and that no OS-level sandbox is to be
    built. Lowering it demands a bubblewrap sandbox on hosts deliberately run
    without one, where the turn keeps its write permission and loses every
    filesystem tool instead.
    """
    if runtime.sandbox_mode.strip().lower() != "read-only":
        return runtime
    return replace(runtime, sandbox_mode="workspace-write")


@dataclass(frozen=True)
class AgentToolPolicy:
    """Describe provider-neutral tools and turn limits for one session."""

    read: bool = True
    search: bool = True
    write: bool = False
    shell: bool = False
    # None delegates session termination entirely to AgentRunSpec.timeout_sec.
    max_turns: int | None = 1
    permission_mode: str = ""
    bare: bool = True
    thinking_budget_tokens: int = 0
    extra_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentHook:
    """Bind one provider-neutral lifecycle callback to a tool matcher."""

    matcher: str
    callback: Any
    timeout_sec: int | None = None


@dataclass
class AgentHooks:
    """Collect generic callbacks that hook-capable providers may expose."""

    pre_tool_use: list[AgentHook] = field(default_factory=list)
    post_tool_use: list[AgentHook] = field(default_factory=list)
    stop: list[AgentHook] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRole:
    """Describe one provider-neutral read-only or writable subagent role."""

    description: str
    instructions: str
    model: str = ""
    reasoning_effort: str = ""
    writable: bool = False
    tool_policy: AgentToolPolicy | None = None


@dataclass(frozen=True)
class StdioMcpServer:
    """Describe one provider-neutral stdio MCP server."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout_sec: int | None = None
    tool_timeout_sec: int | None = None
    tools: tuple[str, ...] = ()


@dataclass
class AgentRunSpec:
    """Describe one backend agent session."""

    system_prompt: str
    user_prompt: str
    cwd: str
    model: str = ""
    writable: bool = True
    timeout_sec: int | None = None
    reasoning_effort: str = ""
    additional_directories: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    driver_script: str = ""
    protected_globs: list[str] = field(default_factory=list)
    allow_dirty_targets: bool = False
    allow_untracked: bool = False
    # A resumed, read-only follow-up may need to inspect a workspace after the
    # implementer has left staged or non-target changes behind. Providers may accept
    # that pre-existing state only when they can prove the turn is read-only and
    # verify that the complete Git-visible state is unchanged afterwards.
    read_only_resume: bool = False
    tool_policy: AgentToolPolicy | None = None
    hooks: AgentHooks | None = None
    subagents: dict[str, AgentRole] = field(default_factory=dict)
    mcp_servers: dict[str, StdioMcpServer] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
    # Append-only observability sink. A backend that streams appends one short
    # line per assistant turn / tool call as it goes, so a caller whose
    # asyncio.wait_for cancels the run still has a record of what the agent was
    # doing — without it, a timed-out session leaves nothing behind but its
    # elapsed time. Shared by reference across ``resolved()``; optional, and
    # backends that cannot stream simply leave it alone.
    #
    # Backends that do NOT support streaming should append a single
    # "progress: not supported by <backend>" entry at the start of run()
    # so callers can distinguish "silent backend" from "agent did nothing".
    progress_log: list[str] | None = None
    # A WRITABLE turn may equally have to start from a worktree the caller already
    # left dirty in ways the turn never touches — a long serving campaign leaves
    # framework runtime files modified and staged. Judging such a turn against a
    # clean HEAD rejects the inherited state before the agent even starts, so a
    # provider that honours this flag snapshots the pre-run state instead and
    # holds the turn responsible only for deviations from that snapshot. It is
    # orthogonal to ``read_only_resume``, which additionally forbids any deviation
    # at all; this flag says nothing about what the turn is allowed to change.
    # ``None`` leaves the choice to the provider, whose default reflects the
    # worktrees it actually runs in.
    allow_dirty_baseline: bool | None = None
    # Exact protected measurement paths that are not necessarily the primary
    # driver.
    protected_paths: list[str] = field(default_factory=list)
    # Environment variables applied over the inherited process environment when
    # the provider spawns this session, so that two sessions running side by
    # side in one Forge process can be given different values for the same
    # variable.
    env: dict[str, str] = field(default_factory=dict)
    # Untracked paths a tool is known to drop in the workspace on its own, as
    # fnmatch patterns relative to the workspace root. Narrower than
    # ``allow_untracked``, which forgives every untracked path and so stops the
    # guard doing its job: this forgives only what the caller can name up front.
    # Empty by default -- a caller that names nothing gets the unchanged rule.
    # Appended, not inserted: the field order above is a published contract that
    # positional callers bind against (tests/test_agent_run_spec_contract.py).
    ignored_untracked_globs: list[str] = field(default_factory=list)

    def resolved(self, runtime: AgentRuntimeConfig) -> AgentRunSpec:
        """Fill omitted per-run values from the runtime and the session scope."""
        return replace(
            self,
            model=self.model.strip() or runtime.model,
            timeout_sec=(self.timeout_sec if self.timeout_sec is not None else runtime.timeout_sec),
            reasoning_effort=(self.reasoning_effort.strip() or runtime.reasoning_effort),
            env={**_session_environment.get(), **self.env},
        )


#: Extra time an outer watchdog adds over ``AgentRunSpec.timeout_sec``. A watchdog
#: set to the same budget races the backend's own deadline, and the cancel it
#: delivers destroys the outputs that deadline was about to preserve.
AGENT_WATCHDOG_GRACE_SEC: int = 300


def watchdog_timeout_sec(session_timeout: float | int) -> float:
    """Return the outer-watchdog budget for a session bounded by ``session_timeout``.

    Args:
        session_timeout: The value passed to ``AgentRunSpec.timeout_sec``.

    Returns:
        A timeout strictly greater than ``session_timeout``, so the backend's own
        graceful-deadline path always fires first.
    """
    return float(session_timeout) + AGENT_WATCHDOG_GRACE_SEC


@dataclass
class AgentRunResult:
    """Normalize one backend session result for the Forge loop."""

    text: str = ""
    subtype: str = ""
    num_turns: int | None = None
    end_reason: str = "agent_stopped"
    session_id: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    edit_count: int = 0
    target_edit_count: int | None = None
    stderr_tail: str = ""
    # Set when the session's workspace could not be cleared of leftover
    # processes: one of ours survived SIGKILL, or one that is not ours to kill
    # is holding a device node. Whatever the loop measures next would be
    # measuring that too, so this is a reason to skip the measurement rather
    # than a detail about how the session ended.
    workspace_contention: str = ""


class AgentBackend(Protocol):
    """Run one Forge agent session through a concrete provider."""

    name: str
    capabilities: AgentCapabilities
    runtime: AgentRuntimeConfig

    async def run(self, spec: AgentRunSpec, usage: Any = None) -> AgentRunResult:
        """Execute one agent session and return a normalized result."""
        raise NotImplementedError


class ResumableAgentBackend(AgentBackend, Protocol):
    """Extend an agent backend with explicit session continuation."""

    async def resume(
        self,
        spec: AgentRunSpec,
        session_id: str,
        feedback: str,
        usage: Any = None,
    ) -> AgentRunResult:
        """Continue one prior session with deterministic gate feedback."""
        raise NotImplementedError
