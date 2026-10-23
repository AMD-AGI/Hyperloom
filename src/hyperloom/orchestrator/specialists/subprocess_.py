# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Subprocess-based specialist dispatcher.

Per-task git worktree under ``runs/specialist/<task_id>/worktree/``, an agent
CLI subprocess scoped via ``--add-dir``, and a ``specialist_done.json``
(+ ``worktree/patches/``) exit signal harvested into the final
:class:`SpecialistRunResult`.

Three agent CLIs can drive that contract, and the deployment picks one
(:func:`hyperloom.common.llm_config.resolve_agent_backend`):
``claude --print --output-format stream-json`` authenticates against the
Anthropic side, ``codex exec --json`` against the OpenAI side and
``dsh --profile headless`` through the DeepSeek Harness. An OpenAI-only deployment has no
Anthropic credential at all, so spawning the Claude CLI there produced a
``Not logged in`` exit on every specialist task and silently cost the session
those domains. Everything around the spawn — worktree setup, the reap loop,
heartbeat/staleness, patch discovery, done-file harvesting and the Ray
GPU-specialist actor path — is backend-agnostic and shared; only the argv and
the ``process.log`` parsers differ.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hyperloom.common.agent_cli import resolve_codex_executable
from hyperloom.common.codex_session import (
    CodexSessionError,
    probe_codex_sandbox_capability,
    resolve_codex_provider_config,
    resolve_codex_sandbox_mode,
)
from hyperloom.common.deadline import Deadline
from hyperloom.common.llm_attribution import inject_env as inject_attribution_env
from hyperloom.common.llm_config import (
    AGENT_BACKEND_CLAUDE,
    AGENT_BACKEND_CODEX,
    AGENT_BACKEND_DEEPSEEK,
    resolve_agent_backend,
    resolve_forge_llm_model,
)
from hyperloom.common.env_safety import (
    redact_file_in_place,
)
from hyperloom.common.fs_utils import (
    create_nofollow,
    read_text_nofollow,
    regular_files_nofollow,
    write_text_nofollow,
)
from hyperloom.common.env_safety import RunAs, RunAsError
from hyperloom.common.proctree import REAP_MARK_ENV, collect_tree, kill_tree, marked_pids, process_uids

from .agent_cli_config import (
    SpecialistAgentUnavailableError,
    _build_specialist_env,
    _codex_mcp_config,
    _codex_provider_resolver_overlay,
    _write_private_codex_config,
)
from .launch_env import apply_gpu_env, prepare_demote, resolve_launch_user, stage_mcp_config
from .dsh_launch import DshLaunchRequest, build_dsh_launch, remove_dsh_profiles, resolve_dsh_executable
from .reap_watch import PROCESS_LOG_NAME, ReapWatch, newest_activity, specialist_activity_paths, tool_in_flight
from .trace_harvest import harvest_agent_trace
from ..actions.cancel_channel import CancelScope, cancel_scope_listener, current_cancel_scope
from hyperloom.common.git_safety import app_git, worktree_head
from ..delivery.deliverable import declared_targets
from ..loop.sub_agent_runner import ExecutionCleanupUnconfirmed
from .wall_budget import clear_wall_budget_extension, wall_budget_extension
from hyperloom.inference_optimizer.trace.agent_trace import AgentTrace


log = logging.getLogger(__name__)

#: How long the reaper waits on a specialist nobody bounded, matching the
#: largest deadline the dispatcher hands down.
UNBOUNDED_REAP_CAP_SEC: float = 4 * 60 * 60.0


# Configuration
@dataclass(frozen=True)
class SpecialistSubprocessConfig:
    """Static config for spawning agent-CLI subprocesses per specialist.

    Captured once at CLI boot and reused for every dispatch; per-task state is
    passed at run time via :meth:`SpecialistSubprocessDispatcher.run`.
    """

    agent_backend: str = ""
    """Which agent CLI to spawn: ``"claude"``, ``"codex"``, ``"deepseek"``, or ``""``.

    Empty resolves the deployment's backend per dispatch via
    :func:`resolve_agent_backend`. The CLI pins it explicitly at boot
    so the backend cannot disagree with the executable and model chosen next to
    it; leaving it empty is for callers that construct a config directly.
    """

    claude_executable: str = "claude"
    """Path / name of the claude CLI binary. Default looks it up on $PATH."""

    codex_executable: str = ""
    """Path / name of the codex CLI binary. Empty resolves it per
    :func:`resolve_codex_executable`."""

    dsh_executable: str = ""
    """Path / name of the DeepSeek Harness ``dsh`` CLI. Empty resolves it per
    :func:`resolve_dsh_executable`."""

    model: str = ""
    """Model id for the selected agent CLI. Empty = that CLI's own default."""

    framework_source_roots: tuple[str, ...] = ()
    """Roots used to seed ``git worktree add`` and as ``--add-dir`` parents.

    The first existing root becomes the worktree base; the rest are exposed
    to the CLI as additional ``--add-dir`` entries.
    """

    mcp_config_path: str | None = None
    """Optional path to a JSON file holding ``{"mcpServers": {...}}``."""

    poll_interval_seconds: float = 5.0
    """How often the reaper polls done.json / process exit / heartbeat."""

    heartbeat_stale_seconds: float = 900.0
    """Liveness window for activity in any of :func:`specialist_activity_paths`.

    Longer than the agent CLI's own API request timeout (600 s for Claude
    Code), after which it logs a retry, so a CLI waiting on its model still
    writes within it; a CLI waiting on a tool is covered by :func:`tool_in_flight`.
    A process that never creates any of them is reaped after this window.
    After one has appeared, its mtime can refresh the liveness clock
    for one window before the subsequent stale window expires, yielding about
    two windows of silence before reap.
    """


# Result
@dataclass
class SpecialistSubprocessResult:
    """Outcome of one specialist subprocess invocation.

    The SpecialistRunner translates this into its own
    :class:`SpecialistRunResult`.
    """

    done_payload: dict[str, Any] | None = None
    """Parsed ``specialist_done.json`` content, or None when the file never
    appeared (the runner then falls back to ``build_empty_specialist_done``)."""

    exit_code: int | None = None
    """Subprocess exit code (None when killed before exit)."""

    elapsed_seconds: float = 0.0

    timed_out: bool = False
    """True when the dispatcher killed the subprocess at its deadline."""

    stale_heartbeat: bool = False
    """True when the heartbeat went stale and the dispatcher killed
    the subprocess."""

    process_log_path: str = ""
    patches: list[str] = field(default_factory=list)
    """Full filesystem paths of patch files discovered under
    ``<worktree>/patches/`` and ``<workspace>/patches/`` (absolute, since both
    roots are session-absolute). Consumers read and sandbox-check these
    directly — do not join them onto a base."""

    patch_roots: dict[str, str] = field(default_factory=dict)
    """Apply root per patch path, for patches harvested from a worktree.

    Empty for patches merely found on disk, whose target tree is not knowable at
    collection time and must therefore still be resolved from their contents."""

    trace: AgentTrace = field(default_factory=AgentTrace)
    """What the agent CLI's own log recorded: token usage (re-entering the
    unified ledger the production specialist's spend), the reply text, the
    intel/tool calls and the per-turn usage. Empty when the log was missing or
    truncated."""

    error: str = ""


#: Directories a specialist writes for its own use inside the worktree, never
#: part of a deliverable.
_SPECIALIST_SCRATCH_DIRS: tuple[str, ...] = ("patches", "artifacts", ".hyperloom")


# Worktree management
def _setup_worktree(
    base: Path,
    worktree_path: Path,
    branch: str,
) -> tuple[Path | None, str]:
    """Create a fresh git worktree at ``worktree_path`` branched off
    ``base``'s HEAD.

    Best-effort: on git error returns ``(None, err)`` so the caller can
    proceed without isolation or hard-fail.

    Args:
        base: Git checkout the worktree is branched off of.
        worktree_path: Destination path for the new worktree.
        branch: Branch name to create for the worktree.

    Returns:
        A ``(worktree_path, "")`` tuple on success, or ``(None, error)`` on
        git failure.
    """
    if worktree_path.exists():
        # Resume / retry: reuse an existing worktree.
        log.warning(
            "specialist worktree already exists at %s; reusing",
            worktree_path,
        )
        return worktree_path, ""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "-C",
        str(base),
        "worktree",
        "add",
        "-b",
        branch,
        str(worktree_path),
    ]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
            # The ref, gitfile and admin dir git makes here must stay readable by a demoted specialist,
            # whatever umask a launch set on the app.
            umask=0o022,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"git worktree add failed to spawn: {exc!r}"
    if cp.returncode != 0:
        return None, (f"git worktree add rc={cp.returncode}: stderr={cp.stderr.strip()[:400]!r}")
    return worktree_path, ""


# How often to poll a PENDING GPU-specialist Ray actor for its pid.
_RAY_PENDING_POLL_INTERVAL_SEC: float = 1.0


def _raise_if_cancelled(scope: CancelScope | None) -> None:
    """Raise ``CancelledError`` with the scope's reason once the action's cancel scope is cancelled."""
    if scope is not None and scope.cancelled:
        raise CancelledError(scope.reason)


def _ray_specialist_pending_deadline_sec() -> float:
    """Max seconds to wait for a GPU-specialist actor to schedule before failing.

    Reads ``INFERENCE_OPTIMIZER_RAY_SPECIALIST_SCHED_TIMEOUT_SEC``. A pending
    request that exceeds this becomes a structured task failure rather than an
    unbounded stall; pending time is bounded and tracked separately from the
    running wall budget.
    """
    try:
        return float(os.environ.get("INFERENCE_OPTIMIZER_RAY_SPECIALIST_SCHED_TIMEOUT_SEC", "300"))
    except (TypeError, ValueError):
        return 300.0


class _RayLeaseProcess:
    """``Popen``-like adapter for a specialist that runs inside a Ray actor."""

    def __init__(self, lease: Any, pid: int) -> None:
        self._lease = lease
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        """Return ``None`` while running; exit code once done.

        When the actor is unreachable (dead) and ``exit_code()`` is ``None``,
        latches :data:`_RAY_ACTOR_DIED_RC` so the reap loop treats it as a
        real failure immediately rather than looping until the wall-clock cap.
        """
        from hyperloom.orchestrator.actions.executors._ray_serving import (
            _RAY_ACTOR_DIED_RC,
        )

        if self.returncode is not None:
            return self.returncode
        if self._lease.is_alive():
            return None
        rc = self._lease.exit_code()
        if rc is None:
            rc = _RAY_ACTOR_DIED_RC
        self.returncode = rc
        return self.returncode

    def reap(self) -> bool:
        """Confirm teardown, consuming one late ACK before abandoning the actor."""
        confirmed = self._lease.stop() is True
        # close() destroys the actor, so retain its exit status while it is observable.
        self.returncode = self._lease.exit_code()
        return confirmed or self._lease.close() is True


def _local_tree_pgid(proc: Any) -> int | None:
    """The process group to name in an operator's log for this specialist, or None.

    Nothing probes this number. A served process is setsid'd by design, so it
    leaves the group its spawn created, and every attempt to decide from such an
    identity whether a lane was free was refuted in review. What the number is
    still worth is a starting point for the human who has to go looking for
    work a task may have left running.

    A local specialist is spawned with ``start_new_session=True``, so its root
    pid is also the id of the group and session it leads, and that number keeps
    naming the group once the root itself has exited.

    A group id means something only inside the PID namespace that issued it, and
    a Ray actor's ids come from whichever node Ray placed the actor on. Printing
    one of those would point the operator at a process on a different host, so
    an actor names nothing here.

    Args:
        proc: The specialist's process handle, which may be absent when the
            cleanup that failed never spawned one.

    Returns:
        int | None: A local process group to record for an operator's benefit,
        or None when there is none to name. Either way it decides nothing:
        no lane is held or released on the strength of this number.
    """
    if proc is None or isinstance(proc, _RayLeaseProcess):
        return None
    pid = getattr(proc, "pid", None)
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None


#: Prints the pids whose environment holds the entry in ``$1``; exits 3 when ``grep`` cannot read NUL-separated input.
_LIST_MARKED_SCRIPT = (
    'grep -qzxF -- x /dev/null; [ "$?" -eq 1 ] || exit 3; '
    'for p in /proc/[0-9]*; do grep -qzxF -- "$1" "$p/environ" 2>/dev/null && echo "${p#/proc/}"; done; exit 0'
)
#: Bounds that scan; walking a few thousand /proc entries takes well under a second.
_MARK_SCAN_TIMEOUT_SEC: float = 30.0


def _marked_pids_as(run_as: RunAs, mark: str) -> list[int]:
    """The pids carrying ``mark``, read as ``run_as``: the app's uid cannot read a demoted child's environ.

    The scan runs as a shell under that account, because it cannot import Hyperloom from the app's tree. Any
    process of that account can write into the scan's stdout, so only pids the account owns are believed.
    """
    try:
        cp = subprocess.run(
            run_as.command(["sh", "-c", _LIST_MARKED_SCRIPT, "sh", f"{REAP_MARK_ENV}={mark}"]),
            capture_output=True,
            text=True,
            timeout=_MARK_SCAN_TIMEOUT_SEC,
            check=False,
            env={"PATH": os.defpath},
            cwd="/",
            **run_as.popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("specialist reap: cannot list marked processes as %s: %s", run_as.user, exc)
        return []
    if cp.returncode != 0:
        log.warning("specialist reap: listing marked processes as %s exited %d", run_as.user, cp.returncode)
    listed = (int(pid) for pid in cp.stdout.split() if pid.isdigit())
    return [pid for pid in listed if process_uids(pid) == (run_as.uid, run_as.uid)]


@dataclass(frozen=True)
class SpecialistSpawn:
    """One specialist subprocess to spawn and reap.

    Attributes:
        task_id: Task identifier used for logging / workspace layout.
        workspace: ``runs/specialist/<task_id>/`` -- where prompt.md,
            process.log, heartbeat.json, and specialist_done.json live.
        worktree: Per-task git worktree (None when worktree setup failed; the
            dispatcher still spawns the agent but it has no write-isolated
            tree, only ``--add-dir <workspace>``).
        worktree_base: Base checkout the worktree was branched off; the clean
            base patch harvesting diffs against.
        system_prompt: System prompt assembled by
            :func:`specialist_prompt_builder.build_specialist_prompts`.
        user_prompt: User prompt from the same builder.
        disallowed_tools: Tool names to deny in the agent CLI subprocess.
            Passed as ``--disallowedTools`` to the Claude CLI; has no Codex or
            dsh equivalent (Codex containment uses ``--sandbox``).
        gpu_ids: GPU ids to expose to the subprocess.
        deadline: Absolute instant the reaper kills at. ``None`` means the
            caller set no bound and the reaper falls back to
            :data:`UNBOUNDED_REAP_CAP_SEC`. A deadline already in the past
            kills on the first poll.
        gpu_lease: When set (Ray-managed GPU execution), a started-on-demand
            ``GpuSpecialistLease``; the whole subprocess runs inside its actor
            holding ``num_gpus`` (Ray sets the visible devices, so any GPU
            command the specialist issues stays within its lease). ``None``
            keeps the local ``Popen`` path, with ``gpu_ids`` pinned into
            ``*_VISIBLE_DEVICES`` as before.
        progress_cb: Optional async callback invoked with each new partial
            checkpoint the specialist writes while it is still alive.
            Exceptions from it never affect the run.
    """

    task_id: str
    workspace: Path
    worktree: Path | None
    worktree_base: Path | None
    system_prompt: str
    user_prompt: str
    disallowed_tools: frozenset[str] | set[str] | tuple[str, ...] = frozenset()
    gpu_ids: tuple[int, ...] = ()
    deadline: Deadline | None = None
    gpu_lease: Any = None
    progress_cb: Any = None


# Dispatcher
class SpecialistSubprocessDispatcher:
    """Spawn + reap one agent-CLI subprocess for a specialist task.

    Reusable across many specialist tasks; owns no per-task state.
    """

    def __init__(self, config: SpecialistSubprocessConfig):
        """Store the static spawn config for reuse across dispatches.

        Args:
            config (SpecialistSubprocessConfig): Session-wide config
                captured at CLI boot; reused for every :meth:`run` call.
        """
        self.config = config

    # Public entry point
    async def run(self, request: SpecialistSpawn) -> SpecialistSubprocessResult:
        """Spawn an agent-CLI subprocess, reap it, return the parsed result.

        The CLI is the one the deployment's credentials can drive (module
        docstring); the spawn, reap and harvest contract is identical either way.

        Args:
            request (SpecialistSpawn): The task, its workspace, prompts, tool
                denials, turn budget, GPU ids or lease, reap deadline and
                progress callback.

        Returns:
            SpecialistSubprocessResult: Parsed outcome — done payload (if
                any), exit code, timing, timeout / stale-heartbeat flags,
                process log path, and discovered patches.
        """
        scope = current_cancel_scope()
        _raise_if_cancelled(scope)
        # Drop any extension left over from a prior run of this task id.
        clear_wall_budget_extension(request.task_id)
        # The pre-image the harvest diffs against, so it must be read before
        # the specialist can commit anything.
        worktree_base_commit = worktree_head(request.worktree)
        request.workspace.mkdir(parents=True, exist_ok=True)
        prompt_file = request.workspace / "prompt.md"
        # A reused workspace may hold a link the previous run's account planted at this name.
        write_text_nofollow(request.workspace, (prompt_file.name,), request.user_prompt)
        process_log = request.workspace / PROCESS_LOG_NAME

        # A requested demote that cannot be honoured fails the task: running the
        # specialist as the app user would defeat the point of asking.
        try:
            run_as = resolve_launch_user(request.gpu_lease)
        except RunAsError as exc:
            return SpecialistSubprocessResult(
                done_payload=None,
                exit_code=None,
                elapsed_seconds=0.0,
                process_log_path=str(process_log),
                error=f"specialist user refused: {exc}",
            )

        # Compose a minimal env. Provider credentials are inherited by default
        # for compatibility with deployments that authenticate the agent CLI
        # via env; set HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=0 to disable.
        env = _build_specialist_env()
        # Keep the spawned CLI off its non-essential traffic and auto-updater.
        from hyperloom.common.llm_stability_env import apply_llm_stability_env

        apply_llm_stability_env(env)
        # Unique to this launch, so every process the CLI starts can be found after it detaches.
        reap_mark = uuid.uuid4().hex
        env[REAP_MARK_ENV] = reap_mark
        # The child spends against the gateway, so tag it or its spend lands
        # under no component at all. The task tag is what attributes the
        # requests of a child that dies before its result row -- the only
        # place such a child's token usage survives is the gateway's log.
        inject_attribution_env(env, component="specialist", operation="run_agent", task=request.task_id)

        backend = ""
        try:
            backend = self._agent_backend()
            cmd = self._build_launch(backend, request, env, run_as)
            watch = self._reap_watch(request)
        except SpecialistAgentUnavailableError as exc:
            # No runtime for this deployment's credential shape. Report it as the
            # task's failure rather than degrading to a CLI that cannot auth.
            return SpecialistSubprocessResult(
                done_payload=None,
                exit_code=None,
                elapsed_seconds=0.0,
                process_log_path=str(process_log),
                error=f"specialist agent runtime unavailable (backend={backend or 'unresolved'}): {exc}",
            )

        apply_gpu_env(env, request.workspace, request.gpu_ids, request.gpu_lease)

        with cancel_scope_listener() as scope:
            log_fh: Any = None
            proc: Any = None
            proc_started: float
            try:
                _raise_if_cancelled(scope)
                if request.gpu_lease is not None:
                    # Non-blocking start: submit the actor launch, then poll for
                    # the pid with ``asyncio.sleep`` between polls. This keeps the
                    # Coordinator event loop responsive while Ray schedules the actor
                    # (no blocking ``ray.get``), and — combined with the timing split
                    # below — excludes Ray *pending* time from the specialist's running
                    # wall budget. A bounded pending deadline turns a permanently
                    # unschedulable request into a structured task failure instead of an
                    # unbounded stall. The actor opens process.log inside its worker
                    # (same host on single-node), so the reaper below reads it directly.
                    from hyperloom.orchestrator.actions.executors._ray_serving import ActorLaunch

                    try:
                        request.gpu_lease.start_async(
                            ActorLaunch(
                                cmd,
                                env=env,
                                cwd=str(request.worktree or request.workspace),
                                log_path=str(process_log),
                                stdin_path=str(prompt_file),
                            )
                        )
                    except Exception as exc:  # isolation boundary: one specialist's GPU actor submit
                        log.warning("specialist GPU actor submit failed: %s", exc, exc_info=True)
                        return SpecialistSubprocessResult(
                            done_payload=None,
                            exit_code=None,
                            elapsed_seconds=0.0,
                            process_log_path=str(process_log),
                            error=f"failed to submit specialist GPU actor: {exc!r}",
                        )
                    pending_deadline_sec = _ray_specialist_pending_deadline_sec()
                    pending_start = time.monotonic()
                    pid: int | None = None
                    while True:
                        _raise_if_cancelled(scope)
                        try:
                            pid = request.gpu_lease.poll_started()
                        except Exception as exc:  # isolation boundary: one specialist's GPU actor start
                            log.warning("specialist GPU actor start failed: %s", exc, exc_info=True)
                            request.gpu_lease.close()
                            return SpecialistSubprocessResult(
                                done_payload=None,
                                exit_code=None,
                                elapsed_seconds=0.0,
                                process_log_path=str(process_log),
                                error=f"specialist GPU actor start failed: {exc!r}",
                            )
                        if pid is not None:
                            break
                        pending_elapsed = time.monotonic() - pending_start
                        if pending_elapsed >= pending_deadline_sec:
                            request.gpu_lease.close()
                            return SpecialistSubprocessResult(
                                done_payload=None,
                                exit_code=None,
                                elapsed_seconds=0.0,
                                process_log_path=str(process_log),
                                error=(
                                    f"specialist GPU actor did not schedule within "
                                    f"{pending_deadline_sec:.0f}s (Ray pending deadline); "
                                    "cluster fully occupied"
                                ),
                            )
                        await asyncio.sleep(_RAY_PENDING_POLL_INTERVAL_SEC)
                    proc = _RayLeaseProcess(request.gpu_lease, pid)
                    # Timing split: the wall-budget clock starts
                    # only now that a real pid exists — the Ray pending time above is
                    # excluded so a slow-to-schedule actor is never mis-reaped.
                    proc_started = time.monotonic()
                else:
                    proc_started = time.monotonic()
                    # Created fresh, never through a name the previous run's account planted; owner-only.
                    log_fh = create_nofollow(request.workspace, (process_log.name,))
                    stdin_fh: Any = None
                    try:
                        demote = await asyncio.to_thread(
                            prepare_demote, run_as, request.workspace, request.worktree, env, (prompt_file, process_log)
                        )
                        stdin_fh = prompt_file.open("rb")
                        proc = subprocess.Popen(
                            cmd if run_as is None else run_as.command(cmd),
                            stdin=stdin_fh,
                            stdout=log_fh,
                            stderr=subprocess.STDOUT,
                            env=env,
                            cwd=str(request.worktree or request.workspace),
                            start_new_session=True,
                            **demote,
                        )
                    except (FileNotFoundError, OSError) as exc:
                        log_fh.close()
                        return SpecialistSubprocessResult(
                            done_payload=None,
                            exit_code=None,
                            elapsed_seconds=0.0,
                            process_log_path=str(process_log),
                            error=f"failed to spawn {backend} subprocess: {exc!r}",
                        )
                    finally:
                        # Popen duplicates the descriptor before returning. Close the
                        # parent's copy on success and every spawn failure.
                        if stdin_fh is not None:
                            try:
                                stdin_fh.close()
                            except OSError:
                                log.warning("failed to close specialist prompt stdin", exc_info=True)

                # Reap loop — poll done-file / exit / heartbeat staleness / deadline.
                outcome = await self._reap_loop(proc, watch, deadline=request.deadline, started=proc_started)
            except (CancelledError, asyncio.CancelledError):
                self._confirm_cancel_teardown(request, proc)
                raise
            finally:
                await asyncio.to_thread(self._reap_marked, reap_mark, run_as)
                if log_fh is not None:
                    log_fh.close()
                try:
                    await asyncio.to_thread(redact_file_in_place, process_log, mode=0o600)
                finally:
                    self._release_task(backend, request)

        return self._harvest_result(request, watch, backend, outcome, worktree_base_commit=worktree_base_commit)

    def _harvest_result(
        self,
        request: SpecialistSpawn,
        watch: ReapWatch,
        backend: str,
        outcome: dict[str, Any],
        *,
        worktree_base_commit: str,
    ) -> SpecialistSubprocessResult:
        """Read the reaped specialist's done payload, patches and agent trace into its result."""
        process_log = watch.workspace / PROCESS_LOG_NAME
        # Parse done.json (best-effort) — first existing candidate.
        done_payload = None
        for cand in watch.done_files:
            done_payload = self._read_done(cand)
            if done_payload is not None:
                break

        # No final done.json — fall back to the most recent incremental partial
        # so a killed-but-productive specialist still surfaces its findings.
        if done_payload is None:
            for cand in watch.partial_files:
                partial = self._read_done(cand)
                if partial is not None:
                    partial["_recovered_from_partial"] = True
                    done_payload = partial
                    if not outcome.get("error"):
                        outcome["error"] = "recovered_from_partial"
                    break

        # Patches: harvest from the worktree via git diff first, else scan
        # disk. Runs after the done payload, whose targets scope the harvest.
        patches, collected_patch_roots = self._collect_patches(
            request.worktree,
            request.workspace,
            request.worktree_base,
            worktree_base_commit=worktree_base_commit,
            targets=declared_targets(done_payload),
        )

        trace = harvest_agent_trace(backend, request.workspace, process_log)
        if trace.error and (outcome["exit_code"] not in (None, 0) or done_payload is None):
            prior_error = str(outcome.get("error") or "").strip()
            outcome["error"] = f"{prior_error}; {trace.error}" if prior_error else trace.error

        return SpecialistSubprocessResult(
            done_payload=done_payload,
            exit_code=outcome["exit_code"],
            elapsed_seconds=outcome["elapsed"],
            timed_out=outcome["timed_out"],
            stale_heartbeat=outcome["stale_heartbeat"],
            process_log_path=str(process_log),
            patches=patches,
            patch_roots=collected_patch_roots,
            trace=trace,
            error=outcome["error"],
        )

    @staticmethod
    def _reap_marked(mark: str, run_as: RunAs | None) -> None:
        """End every process that inherited this launch's reap mark, including one re-parented after the CLI exited."""
        pids = marked_pids(REAP_MARK_ENV, mark) if run_as is None else _marked_pids_as(run_as, mark)
        if pids and not kill_tree(collect_tree(pids)):
            log.warning("specialist reap: marked processes %s outlived SIGKILL", pids)

    def _confirm_cancel_teardown(self, request: SpecialistSpawn, proc: Any) -> None:
        """Tear a cancelled specialist down, raising ``ExecutionCleanupUnconfirmed`` unless teardown is confirmed."""
        try:
            if request.gpu_lease is not None:
                confirmed = request.gpu_lease.close() is True or request.gpu_lease.close() is True
            else:
                confirmed = self._kill(proc) if proc is not None else True
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExecutionCleanupUnconfirmed(
                f"task={request.task_id}: specialist cleanup failed: {exc}", tree_pgid=_local_tree_pgid(proc)
            ) from exc
        if confirmed is not True:
            raise ExecutionCleanupUnconfirmed(
                f"task={request.task_id}: specialist cleanup unconfirmed", tree_pgid=_local_tree_pgid(proc)
            )

    # Internals
    def _build_launch(
        self, backend: str, request: SpecialistSpawn, env: dict[str, str], run_as: RunAs | None
    ) -> list[str]:
        """Build the CLI argv for ``backend``; child-only values go into ``env``."""
        if backend == AGENT_BACKEND_DEEPSEEK:
            dsh_request = DshLaunchRequest(
                executable=resolve_dsh_executable(self.config.dsh_executable),
                model=resolve_forge_llm_model(AGENT_BACKEND_DEEPSEEK, explicit=self.config.model),
                mcp_config_path=self.config.mcp_config_path,
                workspace=request.workspace,
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
            )
            launch = build_dsh_launch(dsh_request, env)
            env.update(launch.env_additions)
            return launch.cmd
        if backend == AGENT_BACKEND_CODEX:
            # Codex's higher-priority developer instructions live in private task-local config.
            cmd, launch_env_additions = self._build_codex_launch(
                workspace=request.workspace,
                worktree=request.worktree,
                system_prompt=request.system_prompt,
                base_env=env,
            )
            env.update(launch_env_additions)
            # Per-task CODEX_HOME so concurrent specialists and the operator's
            # own Codex state stay independent. It lives in the task workspace
            # rather than a temp dir because Codex refuses to create its helper
            # binaries under /tmp and would run without its PATH aliases.
            env["CODEX_HOME"] = str(request.workspace / ".codex")
            return cmd
        cmd = self._build_claude_cmd(
            system_prompt_file=request.workspace / "system_prompt.md",
            system_prompt=request.system_prompt,
            workspace=request.workspace,
            worktree=request.worktree,
            disallowed_tools=frozenset(request.disallowed_tools),
        )
        stage_mcp_config(run_as, request.workspace, cmd)
        return cmd

    @staticmethod
    def _reap_watch(request: SpecialistSpawn) -> ReapWatch:
        """What the reaper polls for ``request``: each file in the worktree first (the prompt-advertised path)."""
        roots = (request.worktree, request.workspace) if request.worktree is not None else (request.workspace,)
        return ReapWatch(
            workspace=request.workspace,
            done_files=tuple(root / "specialist_done.json" for root in roots),
            # Recovered as best-so-far on a budget kill; never a reap signal.
            partial_files=tuple(root / "specialist_done.partial.json" for root in roots),
            progress_cb=request.progress_cb,
            task_id=request.task_id,
        )

    @staticmethod
    def _release_task(backend: str, request: SpecialistSpawn) -> None:
        """Drop what a task holds once its subprocess is gone, however it ended."""
        clear_wall_budget_extension(request.task_id)
        if backend == AGENT_BACKEND_DEEPSEEK:
            remove_dsh_profiles(request.workspace)

    def _agent_backend(self) -> str:
        """Return the agent CLI this dispatch should spawn.

        An explicitly configured backend wins; otherwise the deployment's
        selection decides (:func:`resolve_agent_backend`).

        Returns:
            str: :data:`AGENT_BACKEND_CLAUDE`, :data:`AGENT_BACKEND_CODEX` or :data:`AGENT_BACKEND_DEEPSEEK`.

        Raises:
            SpecialistAgentUnavailableError: If the selected backend is not
                one this dispatcher can spawn.
        """
        pinned = (self.config.agent_backend or "").strip().lower()
        backend = pinned or resolve_agent_backend()
        if backend not in (AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX, AGENT_BACKEND_DEEPSEEK):
            raise SpecialistAgentUnavailableError(
                f"agent_backend={self.config.agent_backend!r} is not one of "
                f"{AGENT_BACKEND_CLAUDE!r} / {AGENT_BACKEND_CODEX!r} / {AGENT_BACKEND_DEEPSEEK!r}"
            )
        return backend

    def _writable_dirs(self, workspace: Path, worktree: Path | None) -> list[str]:
        """Return the dirs an agent CLI may write, in precedence order.

        Worktree first (where patches are authored), then the workspace (where
        ``specialist_done.json`` lands). The framework source trees are absent:
        ``integrate_patch`` is their only writer. Reads are unaffected.

        Args:
            workspace (Path): Task workspace.
            worktree (Path | None): Per-task worktree, when present.

        Returns:
            list[str]: Directory paths, de-duplicated, in order.
        """
        dirs: list[str] = []
        if worktree is not None:
            dirs.append(str(worktree))
        dirs.append(str(workspace))
        return dirs

    def _build_codex_launch(
        self,
        *,
        workspace: Path,
        worktree: Path | None,
        system_prompt: str,
        base_env: Mapping[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        """Build secure Codex argv, private config, and child-only env values.

        Codex 0.144.4 accepts ``developer_instructions`` in ``config.toml`` as a
        real developer-role message. The same schema declares MCP servers under
        ``mcp_servers``. The user prompt is represented only by the positional
        ``-`` marker and is supplied through file-backed stdin by both local and
        Ray launchers.
        """
        cfg = self.config
        executable = resolve_codex_executable(cfg.codex_executable)
        if not executable:
            raise SpecialistAgentUnavailableError(
                "no codex CLI found: set SpecialistSubprocessConfig.codex_executable, "
                "install `codex` on $PATH, or install the Codex SDK runtime "
                "(pip install 'hyperloom-inference_optimizer[llm]')"
            )
        resolver_env = _codex_provider_resolver_overlay(base_env)
        try:
            provider_config = resolve_codex_provider_config(env=resolver_env)
        except CodexSessionError as exc:
            raise SpecialistAgentUnavailableError(f"codex gateway is not configured: {exc}") from exc
        try:
            sandbox_mode = resolve_codex_sandbox_mode(env=dict(base_env))
        except CodexSessionError as exc:
            raise SpecialistAgentUnavailableError(f"codex sandbox policy is not usable: {exc}") from exc
        if sandbox_mode == "read-only":
            raise SpecialistAgentUnavailableError(
                "Codex sandbox mode 'read-only' cannot run a specialist: "
                "the specialist must write its result and may need to author patches"
            )
        if sandbox_mode != "bypass" and not probe_codex_sandbox_capability(env=dict(base_env)):
            raise SpecialistAgentUnavailableError(
                f"Codex sandbox mode {sandbox_mode!r} requires a working bubblewrap "
                "sandbox, but the capability probe failed; refusing to fall back to bypass"
            )

        provider_env_additions = dict(provider_config.env_additions)
        child_env_before_mcp = dict(base_env)
        child_env_before_mcp.update(provider_env_additions)
        effective_source = os.environ.copy()
        effective_source.update(resolver_env)
        effective_source.update(provider_env_additions)
        mcp_lines, mcp_env_additions = _codex_mcp_config(
            cfg.mcp_config_path,
            source=effective_source,
            child_env=child_env_before_mcp,
            protected_env_names=frozenset(provider_env_additions),
        )
        env_additions = dict(provider_env_additions)
        env_additions.update(mcp_env_additions)
        codex_home = workspace / ".codex"
        try:
            _write_private_codex_config(
                codex_home=codex_home,
                developer_instructions=system_prompt,
                mcp_lines=mcp_lines,
            )
        except OSError as exc:
            raise SpecialistAgentUnavailableError(
                f"cannot prepare private Codex config under {codex_home}: {exc}"
            ) from exc

        writable_dirs = self._writable_dirs(workspace, worktree)
        cmd: list[str] = [
            executable,
            "exec",
            "--json",
            "--strict-config",
            "--skip-git-repo-check",
        ]
        if sandbox_mode == "bypass":
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", sandbox_mode])
        # Primary working root; the reap loop spawns with the same cwd.
        cmd.extend(["-C", str(worktree or workspace)])
        for extra_dir in writable_dirs[1:]:
            cmd.extend(["--add-dir", extra_dir])
        # ``features.memories=false`` matches the SDK session: a specialist must
        # not carry state between tasks.
        for override in ("features.memories=false", *provider_config.overrides):
            cmd.extend(["-c", override])
        if cfg.model:
            cmd.extend(["-m", cfg.model])
        cmd.append("-")
        return cmd, env_additions

    def _build_claude_cmd(
        self,
        *,
        system_prompt_file: Path,
        system_prompt: str,
        workspace: Path,
        worktree: Path | None,
        disallowed_tools: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Assemble the ``claude`` CLI argv for a specialist subprocess.

        System and user prompts travel through separate channels: system via
        ``--system-prompt-file``, user via stdin.

        Args:
            system_prompt_file (Path): Destination for the written system prompt;
                passed to ``--system-prompt-file``.
            system_prompt (str): The system prompt text to write.
            workspace (Path): Task workspace surfaced as an ``--add-dir``.
            worktree (Path | None): Write-isolated worktree surfaced as the
                first ``--add-dir`` when present.
            disallowed_tools: Tool names to remove from the available set.

        Returns:
            list[str]: The full command argv to spawn.
        """
        write_text_nofollow(system_prompt_file.parent, (system_prompt_file.name,), system_prompt, mode=0o600)

        cfg = self.config
        cmd: list[str] = [
            cfg.claude_executable,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            # Unattended: a permission prompt waits with no output and no tool running, so the run is reaped as stale.
            # Worktrees and the tool denylist scope behaviour but are not filesystem containment.
            "--permission-mode",
            "bypassPermissions",
            # HOME and CLAUDE_CONFIG_DIR are forwarded for credentials; the operator's settings and hooks are not
            # the specialist's.
            "--setting-sources=",
            "--system-prompt-file",
            str(system_prompt_file),
        ]
        if cfg.model:
            cmd.extend(["--model", cfg.model])
        if disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(sorted(disallowed_tools))])
        from .leaf import build_leaf_agents_json

        cmd.extend(["--agents", build_leaf_agents_json()])
        if cfg.mcp_config_path:
            cmd.extend(["--mcp-config", cfg.mcp_config_path])
        for d in self._writable_dirs(workspace, worktree):
            cmd.extend(["--add-dir", d])
        return cmd

    async def _publish_partial_progress(
        self,
        *,
        partial_files: tuple[Path, ...],
        since_mtime: float,
        elapsed: float,
        progress_cb: Any,
    ) -> float:
        """Forward a freshly-rewritten partial checkpoint to ``progress_cb``.

        Publishes at most one file per call: worktree first, then workspace.

        Args:
            partial_files: Candidate checkpoint paths, worktree first.
            since_mtime: Newest mtime already published.
            elapsed: Seconds since spawn, passed to the callback.
            progress_cb: Async callback receiving ``(payload, elapsed)``.

        Returns:
            The newest mtime seen, so the caller can skip unchanged files.
        """
        newest = since_mtime
        for cand in partial_files:
            try:
                mtime = cand.stat().st_mtime
            except OSError:
                continue
            if mtime <= since_mtime:
                continue
            payload = self._read_done(cand)
            if payload is None:
                continue
            newest = max(newest, mtime)
            try:
                await progress_cb(payload, elapsed)
            except Exception:
                log.exception("specialist progress callback raised")
            break
        return newest

    async def _reap_loop(
        self,
        proc: Any,
        watch: ReapWatch,
        *,
        deadline: Deadline | None,
        started: float,
    ) -> dict[str, Any]:
        """Poll the subprocess until it finishes, stalls, or times out.

        Each tick checks (in order): a done-file at any candidate path
        (graceful exit with a short grace window), natural process exit,
        activity staleness (process.log or the dsh session log, see
        :func:`specialist_activity_paths`, unless :func:`tool_in_flight`), and the hard
        wall-clock cap. Stale / timed-out
        runs are killed via :meth:`_kill`. Partial checkpoints written along
        the way are forwarded to ``progress_cb`` as they change.

        Args:
            proc (Any): The running agent subprocess — a ``subprocess.Popen``
                (local) or a :class:`_RayLeaseProcess` (Ray GPU-specialist
                actor). Only ``poll`` / ``returncode`` / ``pid`` are used.
            watch (ReapWatch): The task workspace, done and partial-checkpoint
                candidates, heartbeat file, progress callback and task id.
            deadline (Deadline | None): Absolute instant to kill at; ``None``
                falls back to :data:`UNBOUNDED_REAP_CAP_SEC` from spawn.
            started (float): ``time.monotonic()`` value at spawn time.

        Returns:
            dict[str, Any]: Outcome with ``exit_code``, ``elapsed``,
                ``timed_out``, ``stale_heartbeat``, and ``error`` keys.
        """
        cfg = self.config
        outcome: dict[str, Any] = {
            "exit_code": None,
            "elapsed": 0.0,
            "timed_out": False,
            "stale_heartbeat": False,
            "error": "",
        }
        # An unbounded caller still gets a finite instant.
        hard_stop = deadline if deadline is not None else Deadline.after(UNBOUNDED_REAP_CAP_SEC, now=started)
        last_heartbeat_seen: float = started
        last_partial_mtime: float = 0.0
        scope = current_cancel_scope()

        while True:
            _raise_if_cancelled(scope)
            await asyncio.sleep(cfg.poll_interval_seconds)
            _raise_if_cancelled(scope)
            now = time.monotonic()
            elapsed = now - started
            outcome["elapsed"] = elapsed

            # done.json appeared — graceful exit with up to 30s grace.
            if any(p.exists() for p in watch.done_files):
                grace_until = now + 30.0
                while time.monotonic() < grace_until and proc.poll() is None:
                    _raise_if_cancelled(scope)
                    await asyncio.sleep(2.0)
                _raise_if_cancelled(scope)
                # Still alive after grace — reap it so no orphaned subprocess leaks.
                if proc.poll() is None:
                    self._kill(proc)
                outcome["exit_code"] = proc.returncode
                outcome["elapsed"] = time.monotonic() - started
                break

            # Process exited on its own.
            if proc.poll() is not None:
                outcome["exit_code"] = proc.returncode
                outcome["elapsed"] = elapsed
                break

            # Still running: republish any checkpoint written since the last tick.
            if watch.progress_cb is not None:
                last_partial_mtime = await self._publish_partial_progress(
                    partial_files=watch.partial_files,
                    since_mtime=last_partial_mtime,
                    elapsed=elapsed,
                    progress_cb=watch.progress_cb,
                )

            # Liveness check: alive while the CLI's output changed within the window, or while it waits on a
            # tool it started after that output. The hard wall-clock cap below still bounds a hung subprocess.
            a_mtime = newest_activity(specialist_activity_paths(watch.workspace))
            if a_mtime is not None and (
                max(0.0, time.time() - a_mtime) <= cfg.heartbeat_stale_seconds
                or await asyncio.to_thread(tool_in_flight, proc.pid, a_mtime)
            ):
                last_heartbeat_seen = now

            if (now - last_heartbeat_seen) > cfg.heartbeat_stale_seconds:
                outcome["stale_heartbeat"] = True
                outcome["error"] = (
                    f"heartbeat stale for {now - last_heartbeat_seen:.0f}s "
                    f"(> {cfg.heartbeat_stale_seconds:.0f}s threshold)"
                )
                self._kill(proc)
                outcome["exit_code"] = proc.returncode
                outcome["elapsed"] = time.monotonic() - started
                break

            # Hard stop — the extension is re-read every poll so an
            # ``extend_lease`` granted mid-run actually moves the kill instant.
            kill_at = hard_stop.at + wall_budget_extension(watch.task_id)
            if now >= kill_at:
                outcome["timed_out"] = True
                outcome["error"] = f"specialist subprocess killed {elapsed:.0f}s in, at its deadline"
                self._kill(proc)
                outcome["exit_code"] = proc.returncode
                outcome["elapsed"] = time.monotonic() - started
                break

        return outcome

    @staticmethod
    def _kill(proc: Any) -> bool:
        """Confirm teardown of the live specialist tree before releasing ownership."""
        if isinstance(proc, _RayLeaseProcess):
            if proc.reap():
                return True
            # No ``tree_pgid``: see :func:`_local_tree_pgid`.
            raise ExecutionCleanupUnconfirmed(f"specialist pid={proc.pid}: actor cleanup unconfirmed")
        if proc.poll() is not None:
            # A re-parented descendant can outlive its root and old process group.
            raise ExecutionCleanupUnconfirmed(
                f"specialist pid={proc.pid}: exited root has no verifiable tree", tree_pgid=proc.pid
            )
        try:
            tree = collect_tree([proc.pid])
            if not any(pid == proc.pid for pid, _ in tree.members) or not kill_tree(tree):
                raise ExecutionCleanupUnconfirmed(
                    f"specialist pid={proc.pid}: tree cleanup unconfirmed", tree_pgid=proc.pid
                )
            proc.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutionCleanupUnconfirmed(
                f"specialist pid={proc.pid}: tree cleanup failed: {exc}", tree_pgid=proc.pid
            ) from exc
        return True

    @staticmethod
    def _harvest_pathspec(targets: Sequence[str] = ()) -> list[str]:
        """Return the ``git diff`` / ``git add`` pathspec that scopes a harvest.

        With no target declared the whole worktree is in scope, minus
        :data:`_SPECIALIST_SCRATCH_DIRS`, whose whole-file copies would
        otherwise be harvested as file creations, and minus the work artifacts
        ``vet_patches`` refuses -- task-owned done files and bytecode caches.
        Excluding exactly what vetting rejects keeps the harvest from authoring
        a patch that is guaranteed to be dropped downstream.
        """
        from .patch_safety import SPECIALIST_WORK_ARTIFACT_PATHSPECS

        declared = [str(t).strip().lstrip("/") for t in targets if str(t).strip()]
        if declared:
            return declared
        return [
            ".",
            *(f":(exclude){name}" for name in _SPECIALIST_SCRATCH_DIRS),
            *SPECIALIST_WORK_ARTIFACT_PATHSPECS,
        ]

    @staticmethod
    def _harvest_worktree_diff(worktree: Path, *, base: str = "HEAD", targets: Sequence[str] = ()) -> str:
        """Return the worktree's edits as one ``-p1`` diff, or ``""``.

        The comparison is ``base``-against-working-tree, since a specialist is
        not required to commit. Intent-to-add stages untracked paths so they
        render as creations, ``git diff`` being blind to them otherwise, which
        is what keeps "edited a file and added one" from harvesting a patch
        that silently omits the addition. Proven Python comment-only edits
        produce no installable patch at all.

        Args:
            worktree: Per-task worktree holding a ``.git`` marker.
            base: The recorded pre-round commit to diff against.
            targets: Worktree-relative paths the round declared.

        Returns:
            str: The diff text, or ``""`` when there is nothing to harvest or
                git could not be run.
        """

        def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
            try:
                return app_git(worktree, *args)
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("specialist: git %s in %s failed: %r", args[0], worktree, exc)
                return None

        from .patch_safety import patch_is_annotation_only

        pathspec = SpecialistSubprocessDispatcher._harvest_pathspec(targets)
        _git("add", "-A", "-N", "--", *pathspec)
        diff = _git("diff", "--no-ext-diff", "--no-textconv", base, "--", *pathspec)
        if diff is None or diff.returncode != 0 or not diff.stdout.strip():
            return ""
        if patch_is_annotation_only(diff.stdout, worktree, reverse=True):
            return ""
        return diff.stdout

    @staticmethod
    def _collect_patches(
        worktree: Path | None,
        workspace: Path,
        worktree_base: Path | None = None,
        *,
        worktree_base_commit: str = "",
        targets: Sequence[str] = (),
    ) -> tuple[list[str], dict[str, str]]:
        """Harvest the specialist's edits, else collect the patch files it wrote.

        Editing the worktree is the primary contract: ``git diff HEAD`` renders
        those edits as one canonical ``-p1`` patch and names its apply root, so
        nothing downstream has to infer which tree it belongs to. Scanning
        ``patches/`` remains for dispatches that never got a worktree, and for
        any git failure here -- a harvest that raises would otherwise discard
        the whole specialist result, done-file included.

        Args:
            worktree: Per-task worktree, or None.
            workspace: Task workspace.
            worktree_base: The tree the worktree stands for -- the checkout it
                was branched off, or the directory it holds a snapshot of --
                which is the apply root of anything harvested from it.
            worktree_base_commit: The commit recorded when the worktree was
                created, so the harvest stays anchored to the pre-round state
                even if the specialist committed.
            targets: Worktree-relative paths the round declared, which scope the
                harvest pathspec.

        Returns:
            ``(patch_paths, patch_roots)``; the latter is empty for scanned files.
        """
        if worktree is not None and (worktree / ".git").exists():
            harvested_diff = SpecialistSubprocessDispatcher._harvest_worktree_diff(
                worktree,
                base=worktree_base_commit or "HEAD",
                targets=targets,
            )
            if harvested_diff:
                try:
                    harvested = write_text_nofollow(worktree, ("patches", "_worktree_diff.patch"), harvested_diff)
                except OSError as exc:
                    log.warning("specialist: refusing to write the harvested patch under %s: %r", worktree, exc)
                else:
                    return [str(harvested)], {str(harvested): str(worktree_base or worktree)}

        out: list[str] = []
        for base in (worktree, workspace):
            if base is None:
                continue
            for ext in (".patch", ".diff"):
                out.extend(
                    str(p) for p in regular_files_nofollow(base, "patches", (ext,)) if p.name != "_worktree_diff.patch"
                )
        return out, {}

    @staticmethod
    def _read_done(done_file: Path) -> dict[str, Any] | None:
        """Parse a ``specialist_done.json`` file, unwrapping intent envelopes.

        Tolerates missing files and parse errors (logged, returns None).
        When the file holds a ``specialist_done`` intent envelope, the
        inner ``payload`` is merged with the outer keys so callers always
        see a flat dict.

        Args:
            done_file (Path): Path to the candidate done-file.

        Returns:
            dict[str, Any] | None: The parsed (and possibly unwrapped)
                payload, or None when missing / unparseable / not a dict.
        """
        if not done_file.exists():
            return None
        try:
            data = json.loads(read_text_nofollow(done_file))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "specialist_done.json parse failed at %s: %r",
                done_file,
                exc,
            )
            return None
        if not isinstance(data, dict):
            log.warning(
                "specialist_done.json at %s is not a dict (%r); ignoring",
                done_file,
                type(data).__name__,
            )
            return None
        if str(data.get("intent_type") or "") == "specialist_done" and isinstance(data.get("payload"), dict):
            inner = data["payload"]
            merged: dict[str, Any] = {}
            for k, v in data.items():
                if k in ("intent_type", "payload"):
                    continue
                merged[k] = v
            for k, v in inner.items():
                merged[k] = v
            log.info(
                "_read_done: unwrapped specialist_done intent envelope at %s (proposal_set_len=%d)",
                done_file,
                len(inner.get("proposal_set") or []) if isinstance(inner.get("proposal_set"), list) else 0,
            )
            return merged
        return data


__all__ = [
    "UNBOUNDED_REAP_CAP_SEC",
    "SpecialistSubprocessConfig",
    "SpecialistSubprocessDispatcher",
    "SpecialistSubprocessResult",
    "_setup_worktree",
]
