# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Long-lived Ray actors that hold GPU/serving process lifecycles (P0).

Implements ray_modify.plan.md §4.2's make-or-break invariant: a GPU-holding
process (sglang/vLLM/Magpie server, or a needs_gpu specialist) must live inside
a Ray actor/task that owns the GPU lease. The process must never outlive its
actor — no detached/``nohup`` GPU processes are allowed to escape the lease.

Design:
- :class:`ManagedServerProcess` is a plain (Ray-free) supervisor: it launches a
  subprocess in its own POSIX session, best-effort arms ``PR_SET_PDEATHSIG`` so
  the OS reaps it if the owning process dies unexpectedly, and reaps the whole
  process tree on ``stop()``. This is unit-testable without a Ray cluster.
- :class:`ServingActor` / :class:`GpuSpecialistActor` are thin ``ray.remote``
  wrappers that hold ``num_gpus`` (and, for serving, a ``serving_slot`` custom
  resource) and delegate lifecycle to a :class:`ManagedServerProcess`. Ray owns
  ``*_VISIBLE_DEVICES``; the subprocess inherits them.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


def _pdeathsig_preexec() -> None:
    """Best-effort: ask the OS to SIGKILL this child if its parent dies.

    Linux-only (``PR_SET_PDEATHSIG``). Combined with an explicit ``stop()`` this
    guarantees no detached GPU process survives its owning Ray actor. A no-op
    where prctl is unavailable.
    """
    try:
        import ctypes  # noqa: PLC0415

        # PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)
    except Exception:  # noqa: BLE001 — best-effort hardening only
        pass


@dataclass
class ManagedServerProcess:
    """Supervise a single GPU/serving subprocess tied to this object's lifetime.

    The process is launched in a new POSIX session (distinct pgid) so the whole
    tree can be reaped atomically, and PR_SET_PDEATHSIG is armed so an
    unexpected owner death still kills it.
    """

    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _cmd: list[str] = field(default_factory=list, init=False, repr=False)

    def start(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        log_path: str | None = None,
    ) -> int:
        """Launch the subprocess and return its pid.

        Args:
            cmd: Command to launch.
            env: Environment (Ray-set ``*_VISIBLE_DEVICES`` already present in
                the actor's ``os.environ``; pass a merged env if overlaying).
            cwd: Working directory.
            log_path: Optional path to redirect stdout/stderr.

        Returns:
            The launched process pid.

        Raises:
            RuntimeError: If a process is already running under this supervisor.
        """
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("ManagedServerProcess already running")
        self._cmd = list(cmd)
        stdout: Any = subprocess.DEVNULL
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            stdout = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 — closed on stop
        if os.name == "posix":
            # New session (distinct pgid) so the whole tree reaps atomically;
            # PR_SET_PDEATHSIG so an unexpected owner death still kills the child.
            self._proc = subprocess.Popen(  # noqa: S603 — cmd is caller's responsibility
                cmd,
                env=env,
                cwd=cwd,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_pdeathsig_preexec,
            )
        else:  # pragma: no cover - non-posix fallback
            self._proc = subprocess.Popen(  # noqa: S603
                cmd,
                env=env,
                cwd=cwd,
                stdout=stdout,
                stderr=subprocess.STDOUT,
            )
        return self._proc.pid

    def pid(self) -> int | None:
        """Return the running pid, or ``None`` when not running.

        Returns:
            The pid, or ``None`` when no live process is supervised.
        """
        if self._proc is None or self._proc.poll() is not None:
            return None
        return self._proc.pid

    def is_alive(self) -> bool:
        """Return whether the supervised process is still running.

        Returns:
            ``True`` when the process is live.
        """
        return self._proc is not None and self._proc.poll() is None

    def stop(self, *, grace_seconds: float = 5.0) -> None:
        """Reap the whole process tree (SIGTERM → grace → SIGKILL). Idempotent.

        Args:
            grace_seconds: Seconds to wait after SIGTERM before SIGKILL.
        """
        from ._subprocess_kill import kill_my_spawned_server

        kill_my_spawned_server(self._proc, grace_seconds=grace_seconds)
        self._proc = None


def _serving_actor_body() -> Any:
    """Build the ServingActor class (imports ray lazily so import is cheap).

    Returns:
        A ``ray.remote``-decorated actor class holding a serving process.
    """
    import ray  # noqa: PLC0415

    @ray.remote
    class ServingActor:
        """Ray actor owning one serving process for its whole lifetime.

        Holds ``num_gpus`` (+ optional ``serving_slot``) via the ``.options()``
        the submitter sets. The server lives exactly as long as the actor.
        """

        def __init__(self) -> None:
            self._mgr = ManagedServerProcess()

        def start(self, cmd, *, env=None, cwd=None, log_path=None) -> int:
            """Launch the serving subprocess; Ray has set visible devices.

            Returns:
                The launched pid.
            """
            merged = dict(os.environ)
            for key, value in (env or {}).items():
                if key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
                    continue
                merged[key] = value
            return self._mgr.start(cmd, env=merged, cwd=cwd, log_path=log_path)

        def is_alive(self) -> bool:
            """Return whether the serving process is still up.

            Returns:
                ``True`` when alive.
            """
            return self._mgr.is_alive()

        def pid(self) -> int | None:
            """Return the serving pid, or ``None``.

            Returns:
                The pid or ``None``.
            """
            return self._mgr.pid()

        def stop(self) -> None:
            """Reap the serving process tree."""
            self._mgr.stop()

        def __ray_terminate__(self) -> None:  # pragma: no cover - Ray teardown hook
            """Reap the serving process when Ray tears the actor down."""
            try:
                self._mgr.stop()
            except Exception:  # noqa: BLE001
                pass

    return ServingActor


def make_serving_actor(num_gpus: float, *, serving_slot: bool = True):
    """Create a ServingActor handle holding ``num_gpus`` (+ optional slot).

    Args:
        num_gpus: GPUs the actor leases.
        serving_slot: Whether to also hold the ``serving_slot=1`` custom
            resource (whole-machine serving mutex, see §4.5).

    Returns:
        A Ray actor handle for the ServingActor.
    """
    actor_cls: Any = _serving_actor_body()
    resources = {"serving_slot": 1} if serving_slot else None
    return actor_cls.options(num_gpus=num_gpus, resources=resources).remote()


def make_gpu_specialist_actor(num_gpus: float):
    """Create a GpuSpecialistActor handle holding ``num_gpus`` (serving-disjoint).

    Args:
        num_gpus: GPUs the specialist actor leases (does not take
            ``serving_slot`` — it is disjoint from serving, see §4.5).

    Returns:
        A Ray actor handle for the GpuSpecialistActor (a ServingActor without
        the serving slot).
    """
    actor_cls: Any = _serving_actor_body()
    return actor_cls.options(num_gpus=num_gpus).remote()


__all__ = [
    "ManagedServerProcess",
    "make_gpu_specialist_actor",
    "make_serving_actor",
]
