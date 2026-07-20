# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Off-loop targeted-build runner (Rung 5, S2).

Spawns a build command as a **detached** subprocess (its own process group) so a
multi-hour compile never blocks the coordinator tick loop, then polls it across
ticks against a monotonic wall-clock deadline. On timeout it tears the whole
process group down non-blocking (SIGTERM, then SIGKILL after a grace window so a
poll never sleeps).

S2 exercises this with a *fake* ``build_command`` argv; the isolation worktree,
preflight, real recipes, and artifact/symbol verification are added in S3-S6.
The build command is argv-only (never a shell string). Each build gets a
per-attempt ``INFERENCE_OPTIMIZER_AITER_JIT_DIR`` so it never shares the
node-global aiter JIT cache (L3).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .build_actions import BuildResult, FrameworkRuntime, TargetedBuildAction

log = logging.getLogger(__name__)

# Seconds between SIGTERM and the escalation to SIGKILL on the process group.
_KILL_GRACE_SEC = 5.0

# Default per-component build budgets (upper bounds, §11 / D4), in seconds.
_DEFAULT_BUDGET_SEC: dict[str, int] = {
    "aiter": 40 * 60,
    "sgl_kernel": 60 * 60,
    "vllm_source": 90 * 60,
    "framework_ext": 90 * 60,
}


def default_budget_sec(component: str) -> int:
    """Per-component wall-clock budget upper bound (§11)."""
    return _DEFAULT_BUDGET_SEC.get(component, 40 * 60)


@dataclass
class BuildHandle:
    """In-memory handle for one in-flight detached build.

    Not persisted directly; the durable copy is ``pending_targeted_build`` in
    shared state. ``sigterm_at`` tracks the two-phase kill across poll calls so
    the reaper never blocks the tick.
    """

    action: TargetedBuildAction
    attempt_root: str
    aiter_jit_dir: str
    build_log_path: str
    proc: Any
    pid: int
    pgid: int
    deadline: float
    sigterm_at: float = 0.0

    def to_sentinel(self, task_id: str) -> dict[str, Any]:
        """Project onto the ``pending_targeted_build`` sentinel dict."""
        return {
            "task_id": task_id,
            "pid": self.pid,
            "pgid": self.pgid,
            "attempt_root": self.attempt_root,
            "aiter_jit_dir": self.aiter_jit_dir,
            "deadline": self.deadline,
            "action": self.action.to_state(),
            "build_log_path": self.build_log_path,
            "ts": time.time(),
        }


def _resolve_budget_sec(action: TargetedBuildAction) -> int:
    budget = int(action.build_budget_sec or 0)
    return budget if budget > 0 else default_budget_sec(action.component)


def spawn_build(
    action: TargetedBuildAction,
    *,
    attempt_root: str,
    run: Callable[..., Any] = subprocess.Popen,
    now: Callable[[], float] = time.monotonic,
) -> BuildHandle:
    """Spawn ``action.build_command`` as a detached process group.

    Creates ``attempt_root`` and a per-attempt ``aiter_jit`` dir, exports
    ``INFERENCE_OPTIMIZER_AITER_JIT_DIR`` for it, and starts the argv command in
    a new session (own process group) with output redirected to ``build.log``.

    Args:
        action: The build to run; ``build_command`` must be a non-empty argv.
        attempt_root: Directory anchoring this attempt's logs and JIT cache.
        run: Injectable process spawner (defaults to ``subprocess.Popen``).
        now: Injectable monotonic clock (defaults to ``time.monotonic``).

    Returns:
        BuildHandle: The in-flight handle (pid/pgid/deadline/log path).

    Raises:
        ValueError: If ``build_command`` is empty.
    """
    argv = list(action.build_command)
    if not argv:
        raise ValueError("targeted_build: build_command must be a non-empty argv")

    root = Path(attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    jit_dir = root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    build_log = root / "build.log"

    env = dict(os.environ)
    env["INFERENCE_OPTIMIZER_AITER_JIT_DIR"] = str(jit_dir)
    for k, v in dict(action.envs).items():
        env[str(k)] = str(v)

    log_fh = build_log.open("w", encoding="utf-8")
    try:
        proc = run(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(root),
            start_new_session=True,
        )
    except Exception:
        log_fh.close()
        raise

    pid = int(getattr(proc, "pid", 0) or 0)
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = pid

    deadline = now() + float(_resolve_budget_sec(action))
    log.info(
        "targeted_build: spawned %s build pid=%d pgid=%d budget=%ds root=%s",
        action.component,
        pid,
        pgid,
        _resolve_budget_sec(action),
        attempt_root,
    )
    return BuildHandle(
        action=action,
        attempt_root=str(root),
        aiter_jit_dir=str(jit_dir),
        build_log_path=str(build_log),
        proc=proc,
        pid=pid,
        pgid=pgid,
        deadline=deadline,
    )


def kill_build_pgroup(pgid: int, *, sig: int = signal.SIGTERM) -> None:
    """Signal an entire build process group (best-effort, non-blocking)."""
    if pgid <= 0:
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError, OverflowError, ValueError):
        pass


def _build_runtime(handle: BuildHandle) -> FrameworkRuntime:
    """The runtime a KEEP would promote (S4-S6 enrich with real prefixes)."""
    return FrameworkRuntime(
        source_root=handle.attempt_root,
        attempt_root=handle.attempt_root,
        runtime_env={"INFERENCE_OPTIMIZER_AITER_JIT_DIR": handle.aiter_jit_dir},
    )


def _finalize(handle: BuildHandle, *, ok: bool, failure_class: str, summary: str) -> BuildResult:
    return BuildResult(
        ok=ok,
        attempt_root=handle.attempt_root,
        runtime=_build_runtime(handle) if ok else FrameworkRuntime(),
        build_log_path=handle.build_log_path,
        failure_class="ok" if ok else failure_class,
        failure_summary=summary,
        error="" if ok else summary,
    )


def poll_build(
    handle: BuildHandle,
    *,
    now: Callable[[], float] = time.monotonic,
) -> BuildResult | None:
    """Poll a build once; return ``None`` while still running.

    Non-blocking: on deadline it sends SIGTERM and records ``sigterm_at``, then
    on a later poll past the grace window escalates to SIGKILL, so the reaper
    never sleeps inside a tick.

    Args:
        handle: The in-flight build handle.
        now: Injectable monotonic clock.

    Returns:
        BuildResult when terminal (exited / timed out+dead), else ``None``.
    """
    rc = handle.proc.poll()
    if rc is not None:
        if handle.sigterm_at > 0.0:
            return _finalize(
                handle,
                ok=False,
                failure_class="timeout",
                summary=f"build exceeded wall-clock budget and was terminated (rc={rc})",
            )
        if int(rc) == 0:
            return _finalize(handle, ok=True, failure_class="ok", summary="")
        return _finalize(
            handle,
            ok=False,
            failure_class="compile_error",
            summary=f"build command exited {int(rc)}",
        )

    t = now()
    if handle.sigterm_at > 0.0:
        # Already asked to stop; escalate to SIGKILL once the grace elapses.
        if t - handle.sigterm_at >= _KILL_GRACE_SEC:
            kill_build_pgroup(handle.pgid, sig=signal.SIGKILL)
        return None
    if t >= handle.deadline:
        kill_build_pgroup(handle.pgid, sig=signal.SIGTERM)
        handle.sigterm_at = t
        return None
    return None


__all__ = [
    "BuildHandle",
    "default_budget_sec",
    "kill_build_pgroup",
    "poll_build",
    "spawn_build",
]
