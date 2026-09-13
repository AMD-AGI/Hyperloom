# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Long-lived Ray actors that hold GPU/serving process lifecycles."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.env_safety import scrub_benchmark_process_env
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS

from ._subprocess_kill import COOPERATIVE_REAP_BUDGET_SEC

log = logging.getLogger(__name__)

# Ray-side sentinel returncodes, allocated out of the same space as ``_subprocess_kill``'s -- read the note there
# before claiming a new one.
_ACTOR_TIMEOUT_RC: int = -916
_RAY_ACTOR_DIED_RC: int = -913

# Timeout for ray.get probes on specialist actor methods (is_alive/exit_code/stop).
_LEASE_PROBE_TIMEOUT_SEC: float = 30.0

# How often the submitter of a round looks up from ``ray.wait`` to see whether the action it belongs to has been
# cancelled.
_CANCEL_POLL_SEC: float = 0.25

# How long the submitter waits for a cancelled round to come back on its own before killing the actor out from under
# it.
CANCEL_ROUND_GRACE_SEC: float = COOPERATIVE_REAP_BUDGET_SEC + _CANCEL_POLL_SEC

# How long releasing a lease waits for the actor to reap its served process before killing the actor anyway.
CLOSE_STOP_TIMEOUT_SEC: float = 10.0

# Method slots the serving actor runs at once: the round, plus room for the cancel that has to reach it.
_SERVING_ACTOR_CONCURRENCY: int = 2

#: The masks Ray owns for its serving children. Single definition lives in
#: ``hyperloom.common.visible_devices``.
_VISIBLE_DEVICE_ENV_KEYS: tuple[str, ...] = COUNTING_VISIBLE_DEVICE_VARS


class RayInfeasibleError(RuntimeError):
    """Raised when the cluster can never satisfy the requested resources."""


def _assert_cluster_feasible(*, num_gpus: float, serving_slot: bool) -> None:
    """Raise :exc:`RayInfeasibleError` when the cluster cannot satisfy the request."""
    import ray  # noqa: PLC0415

    totals = ray.cluster_resources()
    cluster_gpus = float(totals.get("GPU", 0))
    if cluster_gpus < num_gpus:
        raise RayInfeasibleError(
            f"cluster has {cluster_gpus} GPU(s), {num_gpus} requested; set INFERENCE_OPTIMIZER_RAY_EXEC=0 or add GPUs"
        )
    if serving_slot and "serving_slot" not in totals:
        raise RayInfeasibleError(
            "existing Ray head has no serving_slot resource; "
            "restart with --resources='{\"serving_slot\":1}' or set INFERENCE_OPTIMIZER_RAY_EXEC=0"
        )


def _pdeathsig_preexec() -> None:
    """Ask the OS to SIGTERM this child if its parent dies (Linux ``PR_SET_PDEATHSIG``).

    The signal must be trappable. This child is the benchmark wrapper, and the wrapper -- not us -- owns the server:
    the server is ``setsid``'d into its own process group, so the only in-band teardown that can reach it is the
    wrapper's own ``trap cleanup EXIT INT TERM``. SIGKILL cannot be trapped, so arming it here killed the one process
    that knew how to stop the server and orphaned a multi-GPU vLLM tree. A no-op where prctl is unavailable, and no
    guarantee either way -- the durable backstop is the pidfile scanned by
    :func:`._server_lifecycle.reap_orphaned_servers`.
    """
    try:
        import ctypes  # noqa: PLC0415

        # PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)
    except Exception:  # noqa: BLE001 — best-effort hardening only
        pass


@dataclass
class ManagedServerProcess:
    """Supervise a single GPU/serving subprocess tied to this object's lifetime.

    Launched in a new POSIX session (distinct pgid) so the tree can be reaped atomically; PR_SET_PDEATHSIG is armed so
    an unexpected owner death still triggers the child's own cleanup (see :func:`_pdeathsig_preexec`).
    """

    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def start(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        log_path: str | None = None,
        stdin_path: str | None = None,
    ) -> int:
        """Launch the subprocess and return its pid."""
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("ManagedServerProcess already running")
        stdin: Any = subprocess.DEVNULL
        stdout: Any = subprocess.DEVNULL
        stdin_fh: Any = None
        stdout_fh: Any = None
        try:
            if stdin_path:
                stdin_fh = open(stdin_path, "rb")
                stdin = stdin_fh
            if log_path:
                os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                stdout_fh = open(log_path, "w", encoding="utf-8")
                stdout = stdout_fh
            if os.name == "posix":
                # New session (distinct pgid) so the whole tree reaps atomically; PR_SET_PDEATHSIG so an unexpected
                # owner death still kills the child.
                self._proc = subprocess.Popen(  # noqa: S603 — cmd is caller's responsibility
                    cmd,
                    env=env,
                    cwd=cwd,
                    stdin=stdin,
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
                    stdin=stdin,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                )
        finally:
            # Popen has transferred the descriptors to the child before it returns.
            for fh in (stdin_fh, stdout_fh):
                if fh is not None:
                    try:
                        fh.close()
                    except OSError:
                        log.warning("failed to close parent subprocess file handle", exc_info=True)
        return self._proc.pid

    def pid(self) -> int | None:
        """Return the running pid, or ``None`` when not running."""
        if self._proc is None or self._proc.poll() is not None:
            return None
        return self._proc.pid

    def is_alive(self) -> bool:
        """Return whether the supervised process is still running."""
        return self._proc is not None and self._proc.poll() is None

    def exit_code(self) -> int | None:
        """Return the process exit code, or ``None`` while running / never started."""
        if self._proc is None:
            return None
        return self._proc.poll()

    def stop(self, *, grace_seconds: float = 5.0) -> None:
        """Reap the whole process tree (SIGTERM → grace → SIGKILL). Idempotent."""
        from ._subprocess_kill import kill_my_spawned_server

        kill_my_spawned_server(self._proc, grace_seconds=grace_seconds)
        self._proc = None


def _serving_actor_body() -> Any:
    """Build the ServingActor class (imports ray lazily so import is cheap)."""
    import ray  # noqa: PLC0415

    @ray.remote
    class ServingActor:
        """Ray actor owning one serving process for its whole lifetime."""

        def __init__(self) -> None:
            self._mgr = ManagedServerProcess()
            # The cancel scope of the round currently in flight, if any.
            self._round_scope: Any = None

        def start(
            self,
            cmd,
            *,
            env=None,
            cwd=None,
            log_path=None,
            scrub_benchmark_env=False,
            env_mode="merge",
            stdin_path=None,
        ) -> int:
            """Launch the serving subprocess; Ray has set visible devices."""
            if env_mode == "merge":
                child_env = dict(os.environ)
                for key, value in (env or {}).items():
                    if key in _VISIBLE_DEVICE_ENV_KEYS:
                        continue
                    child_env[key] = value
            elif env_mode == "replace":
                child_env = dict(env or {})
                for key in _VISIBLE_DEVICE_ENV_KEYS:
                    if key in os.environ:
                        child_env[key] = os.environ[key]
            else:
                raise ValueError(f"unsupported env_mode {env_mode!r}; expected 'merge' or 'replace'")
            if scrub_benchmark_env:
                scrub_benchmark_process_env(child_env)
            start_kwargs = {
                "env": child_env,
                "cwd": cwd,
                "log_path": log_path,
            }
            if stdin_path is not None:
                start_kwargs["stdin_path"] = stdin_path
            return self._mgr.start(cmd, **start_kwargs)

        def run_blocking(
            self,
            cmd,
            *,
            env=None,
            cwd=None,
            timeout=None,
            soft_deadline_sec=None,
            server_log_path=None,
            server_already_ready=False,
            session_remaining_sec=None,
        ):
            """Run one benchmark round to completion; return ``(rc, stdout, stderr)``."""
            import subprocess as _sp  # noqa: PLC0415

            from ..cancel_channel import CancelScope, use_cancel_scope  # noqa: PLC0415
            from ._ray_backend import _run_subprocess_worker  # noqa: PLC0415

            scope = CancelScope()
            self._round_scope = scope
            try:
                with use_cancel_scope(scope):
                    return _run_subprocess_worker(
                        cmd=cmd,
                        env=env,
                        cwd=cwd,
                        timeout_s=timeout,
                        soft_deadline_sec=soft_deadline_sec,
                        server_log_path=server_log_path,
                        server_already_ready=server_already_ready,
                        session_remaining_sec=session_remaining_sec,
                    )
            except _sp.TimeoutExpired as exc:
                return _ACTOR_TIMEOUT_RC, "", f"TimeoutExpired: {exc}"
            finally:
                self._round_scope = None

        def cancel_round(self, reason: str) -> bool:
            """Ask the round in flight to stop itself; return whether there was one."""
            scope = self._round_scope
            if scope is None:
                return False
            scope.cancel(reason=reason)
            return True

        def is_alive(self) -> bool:
            """Return whether the serving process is still up."""
            return self._mgr.is_alive()

        def pid(self) -> int | None:
            """Return the serving pid, or ``None``."""
            return self._mgr.pid()

        def exit_code(self) -> int | None:
            """Return the supervised process exit code, or ``None`` while running."""
            return self._mgr.exit_code()

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
    """Create a ServingActor handle holding ``num_gpus`` (+ optional ``serving_slot``)."""
    actor_cls: Any = _serving_actor_body()
    resources = {"serving_slot": 1} if serving_slot else None
    return actor_cls.options(
        num_gpus=num_gpus,
        resources=resources,
        max_concurrency=_SERVING_ACTOR_CONCURRENCY,
    ).remote()


def make_gpu_specialist_actor(num_gpus: float, *, serving_slot: bool = False):
    """Create a ServingActor handle for a GPU specialist, holding ``num_gpus`` (+ optional ``serving_slot``)."""
    actor_cls: Any = _serving_actor_body()
    resources = {"serving_slot": 1} if serving_slot else None
    return actor_cls.options(num_gpus=num_gpus, resources=resources).remote()


class ServingLease:
    """A held Ray GPU lease spanning every round that shares one server."""

    def __init__(
        self,
        *,
        num_gpus: float,
        serving_slot: bool = True,
        ensure_log_path: Any = None,
    ) -> None:
        self._num_gpus = float(num_gpus)
        self._serving_slot = bool(serving_slot)
        self._ensure_log_path = ensure_log_path
        self._actor: Any = None

    def ensure(self) -> None:
        """Ensure the Ray cluster is up and the serving actor is created."""
        if self._actor is not None:
            return
        from ._ray_backend import get_ray_backend  # noqa: PLC0415

        get_ray_backend().ensure(log_path=self._ensure_log_path)
        _assert_cluster_feasible(num_gpus=self._num_gpus, serving_slot=self._serving_slot)
        self._actor = make_serving_actor(self._num_gpus, serving_slot=self._serving_slot)

    def run_session_kill(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: int | float | None = None,
        soft_deadline_sec: float | None = None,
        server_log_path: str | None = None,
        server_already_ready: bool = False,
        session_remaining_sec: float | None = None,
    ) -> tuple[int, str, str]:
        """Run one benchmark round inside the lease's actor; return ``(rc, stdout, stderr)``."""
        from ..cancel_channel import cancel_scope_listener  # noqa: PLC0415

        try:
            self.ensure()
        except (RayInfeasibleError, RuntimeError) as exc:
            log.warning("ServingLease.run_session_kill: cluster ensure failed: %r", exc)
            return 1, "", f"ray_ensure_error: {exc}"[:2000]
        # Registered before the round is submitted, so a cancel that arrives while Ray is still scheduling it is one
        # this call is counted as able to hear -- the same window the local path opens around its spawn.
        with cancel_scope_listener() as cancel_scope:
            ref = self._actor.run_blocking.remote(
                cmd,
                env=env,
                cwd=cwd,
                timeout=timeout,
                soft_deadline_sec=soft_deadline_sec,
                server_log_path=server_log_path,
                server_already_ready=server_already_ready,
                session_remaining_sec=session_remaining_sec,
            )
            return self._collect_round(ref, cmd=cmd, timeout=timeout, cancel_scope=cancel_scope)

    def _collect_round(
        self,
        ref: Any,
        *,
        cmd: list[str],
        timeout: int | float | None,
        cancel_scope: Any,
    ) -> tuple[int, str, str]:
        """Wait for a submitted round, forwarding a cancel to the actor if one comes."""
        import subprocess as _sp  # noqa: PLC0415

        import ray  # noqa: PLC0415

        # Resolve Ray's exception classes defensively.
        _ray_exc = getattr(ray, "exceptions", None)
        _actor_err: Any = getattr(_ray_exc, "RayActorError", ()) if _ray_exc else ()
        _task_err: Any = getattr(_ray_exc, "RayTaskError", ()) if _ray_exc else ()
        try:
            if cancel_scope is None:
                rc, out, err = ray.get(ref)
            else:
                rc, out, err = self._await_or_cancel(ref, cancel_scope=cancel_scope)
        except _actor_err as exc:  # type: ignore[misc]
            # The actor (worker) itself died — e.g. its server OOM-killed the worker, or raylet reaped it. Drop the
            # dead handle so the next round re-creates a fresh actor via ``ensure()`` and this round surfaces as a
            # benchmark failure instead of cascading. Dropping it also makes ``stop()``/``close()`` no-ops, so nothing
            # here can still reach the server tree the dead actor spawned; the shutdown pidfile reap frees those GPUs.
            log.warning(
                "ServingLease.run_session_kill: ray actor died: %r; its server tree (if any) "
                "is left to the pidfile reaper",
                exc,
            )
            self._actor = None
            try:
                from ._ray_backend import mark_ray_backend_unhealthy  # noqa: PLC0415

                mark_ray_backend_unhealthy()
            except Exception:  # noqa: BLE001 - failure recovery must not raise
                pass
            return 1, "", f"ray_actor_error: {exc}"[:2000]
        except _task_err as exc:  # type: ignore[misc]
            # Worker crash / unexpected error: surface as a benchmark failure so the caller's existing rc!=0 handling
            # runs, not a session crash.
            log.warning("ServingLease.run_session_kill: ray worker error: %r", exc)
            return 1, "", f"ray_worker_error: {exc}"[:2000]
        if rc == _ACTOR_TIMEOUT_RC:
            raise _sp.TimeoutExpired(cmd, timeout or 0, output=out or None, stderr=err or None)
        return rc, out, err

    def _await_or_cancel(self, ref: Any, *, cancel_scope: Any) -> tuple[int, str, str]:
        """Block on a round, asking the actor to stop it if the scope is cancelled."""
        import ray  # noqa: PLC0415

        from ._subprocess_kill import ORCHESTRATOR_CANCELLED_RETURNCODE  # noqa: PLC0415

        asked_at: float | None = None
        while True:
            ready, _ = ray.wait([ref], num_returns=1, timeout=_CANCEL_POLL_SEC)
            if ready:
                return ray.get(ref)
            if asked_at is None:
                if not cancel_scope.cancelled:
                    continue
                reason = cancel_scope.reason or "orchestrator_cancelled"
                asked_at = time.monotonic()
                log.warning(
                    "ServingLease: asking the actor to stop the round in flight (%s)",
                    reason,
                )
                if not self._ask_actor_to_cancel(reason):
                    # The actor never took the round, or cannot be reached to be told about it.
                    asked_at -= CANCEL_ROUND_GRACE_SEC
            elif time.monotonic() - asked_at >= CANCEL_ROUND_GRACE_SEC:
                log.warning(
                    "ServingLease: the actor did not return its cancelled round within %.0fs; "
                    "killing it to release the lease",
                    CANCEL_ROUND_GRACE_SEC,
                )
                # Straight to the kill: an actor that has not answered is not going to answer a graceful stop either,
                # and waiting for one would spend the rest of the window the caller is owed.
                self._kill_actor()
                return (
                    ORCHESTRATOR_CANCELLED_RETURNCODE,
                    "",
                    "the orchestrator cancelled this action; its Ray actor was killed after "
                    f"{CANCEL_ROUND_GRACE_SEC:.0f}s without returning the round",
                )

    def _ask_actor_to_cancel(self, reason: str) -> bool:
        """Tell the actor to stop the round it is running. Never raises."""
        import ray  # noqa: PLC0415

        actor = self._actor
        if actor is None:
            return False
        try:
            return bool(ray.get(actor.cancel_round.remote(reason), timeout=_LEASE_PROBE_TIMEOUT_SEC))
        except Exception as exc:  # noqa: BLE001 — an unreachable actor gets killed instead
            log.warning("ServingLease: could not reach the actor to cancel its round: %r", exc)
            return False

    def close(self) -> None:
        """Release the GPU lease: stop the server, then kill the actor. Idempotent."""
        if self._actor is None:
            return
        try:
            import ray  # noqa: PLC0415

            ray.get(self._actor.stop.remote(), timeout=CLOSE_STOP_TIMEOUT_SEC)
        except Exception as exc:  # noqa: BLE001 — the kill below is the backstop
            log.warning("ServingLease.close: the actor did not stop its server: %r", exc)
        self._kill_actor()

    def _kill_actor(self) -> None:
        """Kill the actor handle without waiting for it. Idempotent, never raises."""
        if self._actor is None:
            return
        try:
            import ray  # noqa: PLC0415

            ray.kill(self._actor)
        except Exception:  # noqa: BLE001 — teardown must not raise
            pass
        self._actor = None

    def __enter__(self) -> ServingLease:
        """Ensure the lease on context entry."""
        self.ensure()
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Release the lease on context exit."""
        self.close()
        return False


def maybe_serving_lease(
    *,
    num_gpus: float,
    serving_slot: bool = True,
    ensure_log_path: Any = None,
) -> ServingLease | None:
    """Return a :class:`ServingLease` when single-node Ray execution is active."""
    from ._multi_node_env import is_multi_node  # noqa: PLC0415
    from ._ray_backend import _should_use_ray_backend  # noqa: PLC0415

    if not _should_use_ray_backend() or is_multi_node():
        return None
    return ServingLease(
        num_gpus=num_gpus,
        serving_slot=serving_slot,
        ensure_log_path=ensure_log_path,
    )


class GpuSpecialistLease:
    """A held Ray GPU lease that runs a ``needs_gpu`` specialist subprocess."""

    def __init__(
        self,
        *,
        num_gpus: float,
        serving_slot: bool = False,
        ensure_log_path: Any = None,
    ) -> None:
        self._num_gpus = float(num_gpus)
        self._serving_slot = bool(serving_slot)
        self._ensure_log_path = ensure_log_path
        self._actor: Any = None
        self._pid: int | None = None
        # §3.3 non-blocking start: the pending ObjectRef for the actor's ``start`` remote call.
        self._start_ref: Any = None

    def start_async(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        log_path: str | None = None,
        env_mode: str = "merge",
        stdin_path: str | None = None,
    ) -> None:
        """Create the actor and SUBMIT the subprocess launch without blocking."""
        from ._ray_backend import get_ray_backend  # noqa: PLC0415

        get_ray_backend().ensure(log_path=self._ensure_log_path)
        _assert_cluster_feasible(num_gpus=self._num_gpus, serving_slot=self._serving_slot)
        self._actor = make_gpu_specialist_actor(self._num_gpus, serving_slot=self._serving_slot)
        self._start_ref = self._actor.start.remote(
            cmd,
            env=env,
            cwd=cwd,
            log_path=log_path,
            env_mode=env_mode,
            stdin_path=stdin_path,
        )

    def poll_started(self) -> int | None:
        """Non-blocking poll for the launched pid."""
        if self._pid is not None:
            return self._pid
        if self._start_ref is None:
            return None
        import ray  # noqa: PLC0415

        ready, _ = ray.wait([self._start_ref], num_returns=1, timeout=0)
        if not ready:
            return None
        self._pid = int(ray.get(self._start_ref))
        self._start_ref = None
        return self._pid

    def pid(self) -> int | None:
        """Return the launched pid, or ``None`` before it has been resolved."""
        return self._pid

    def is_alive(self) -> bool:
        """Return whether the specialist subprocess is still running."""
        if self._actor is None:
            return False
        import ray  # noqa: PLC0415

        try:
            return bool(ray.get(self._actor.is_alive.remote(), timeout=_LEASE_PROBE_TIMEOUT_SEC))
        except ray.exceptions.GetTimeoutError:
            return True  # still-alive assumption on timeout (avoid premature kill)
        except Exception:  # noqa: BLE001 — dead actor reads as not-alive
            return False

    def exit_code(self) -> int | None:
        """Return the subprocess exit code, or ``None`` while running / actor dead."""
        if self._actor is None:
            return None
        import ray  # noqa: PLC0415

        try:
            return ray.get(self._actor.exit_code.remote(), timeout=_LEASE_PROBE_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            return None

    def stop(self) -> None:
        """Reap the specialist subprocess tree (keeps the actor/lease alive). Never raises."""
        if self._actor is None:
            return
        import ray  # noqa: PLC0415

        try:
            ray.get(self._actor.stop.remote(), timeout=_LEASE_PROBE_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001 — teardown must not raise
            pass

    def close(self) -> None:
        """Stop the specialist, then kill the actor to release the GPU lease.

        The stop comes first for the same reason it does in
        :meth:`ServingLease.close`: ``ray.kill`` skips ``__ray_terminate__``,
        so killing the actor first leaves the specialist's process tree with no
        one to reap it. Idempotent, never raises.
        """
        if self._actor is None:
            return
        self.stop()
        try:
            import ray  # noqa: PLC0415

            ray.kill(self._actor)
        except Exception:  # noqa: BLE001 — teardown must not raise
            pass
        self._actor = None


def maybe_gpu_specialist_lease(
    *,
    num_gpus: float,
    serving_slot: bool = False,
    ensure_log_path: Any = None,
) -> GpuSpecialistLease | None:
    """Return a :class:`GpuSpecialistLease` when single-node Ray execution is active."""
    if num_gpus <= 0:
        return None
    from ._multi_node_env import is_multi_node  # noqa: PLC0415
    from ._ray_backend import _should_use_ray_backend  # noqa: PLC0415

    if not _should_use_ray_backend() or is_multi_node():
        return None
    return GpuSpecialistLease(
        num_gpus=num_gpus,
        serving_slot=serving_slot,
        ensure_log_path=ensure_log_path,
    )


__all__ = [
    "GpuSpecialistLease",
    "ManagedServerProcess",
    "RayInfeasibleError",
    "ServingLease",
    "make_gpu_specialist_actor",
    "make_serving_actor",
    "maybe_gpu_specialist_lease",
    "maybe_serving_lease",
]
