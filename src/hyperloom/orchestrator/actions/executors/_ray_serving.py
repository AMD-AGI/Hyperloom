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

# Ray-side sentinel returncodes, allocated out of the same space as
# ``_subprocess_kill``'s -- read the note there before claiming a new one. Both
# of these leave ``_grid_runner._run_magpie`` through the very return channel
# that carries ``AGENTX_PREFLIGHT_RETURNCODE``, so an overlap would have an
# actor timeout recorded as a failed AgentX preflight.
_ACTOR_TIMEOUT_RC: int = -916
_RAY_ACTOR_DIED_RC: int = -913

# Timeout for ray.get probes on specialist actor methods (is_alive/exit_code/stop).
_LEASE_PROBE_TIMEOUT_SEC: float = 30.0

# How often the submitter of a round looks up from ``ray.wait`` to see whether
# the action it belongs to has been cancelled. Short enough that the hop through
# Ray costs the cancel almost nothing on top of what stopping the round costs
# anyway, long enough not to spin.
_CANCEL_POLL_SEC: float = 0.25

# How long the submitter waits for a cancelled round to come back on its own
# before killing the actor out from under it. The round in the actor stops itself
# exactly the way a local child does, so this is that cost plus the poll the
# answer is seen at -- derived, not picked, because a grace even slightly short of
# the reap expires every single time and throws away the sentinel the round was
# about to hand back. The kill is what a wedged actor gets, not what a working one
# gets for being ordinary.
CANCEL_ROUND_GRACE_SEC: float = COOPERATIVE_REAP_BUDGET_SEC + _CANCEL_POLL_SEC

# How long releasing a lease waits for the actor to reap its served process
# before killing the actor anyway. Sized on what that reap costs -- SIGTERM, the
# grace, SIGKILL -- and deliberately short: teardown often runs inside the
# closing window, which is reserved for the report, not for waiting on a server.
CLOSE_STOP_TIMEOUT_SEC: float = 10.0

# Method slots the serving actor runs at once: the round, plus room for the
# cancel that has to reach it. A single-slot actor would queue the cancel behind
# the very round it is meant to stop.
_SERVING_ACTOR_CONCURRENCY: int = 2

#: The masks Ray owns for its serving children. Single definition lives in
#: ``hyperloom.common.visible_devices``.
_VISIBLE_DEVICE_ENV_KEYS: tuple[str, ...] = COUNTING_VISIBLE_DEVICE_VARS


class RayInfeasibleError(RuntimeError):
    """Raised when the cluster can never satisfy the requested resources."""


def _assert_cluster_feasible(*, num_gpus: float, serving_slot: bool) -> None:
    """Raise :exc:`RayInfeasibleError` when the cluster cannot satisfy the request.

    Reads ``ray.cluster_resources()`` (totals, not available) so legitimate
    contention still queues — only permanently infeasible configurations fail fast.
    """
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
        stdin_path: str | None = None,
    ) -> int:
        """Launch the subprocess and return its pid.

        Args:
            cmd: Command to launch.
            env: Environment (Ray-set ``*_VISIBLE_DEVICES`` already present in
                the actor's ``os.environ``; pass a merged env if overlaying).
            cwd: Working directory.
            log_path: Optional path to redirect stdout/stderr.
            stdin_path: Optional read-only file to use as stdin. Defaults to
                ``subprocess.DEVNULL`` so actor stdin is never inherited.

        Returns:
            The launched process pid.

        Raises:
            RuntimeError: If a process is already running under this supervisor.
        """
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("ManagedServerProcess already running")
        self._cmd = list(cmd)
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
                # New session (distinct pgid) so the whole tree reaps atomically;
                # PR_SET_PDEATHSIG so an unexpected owner death still kills the child.
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
            # Popen has transferred the descriptors to the child before it
            # returns. Close the parent's copies immediately, including when
            # spawning fails, so neither long-running actors nor retries leak fds.
            for fh in (stdin_fh, stdout_fh):
                if fh is not None:
                    try:
                        fh.close()
                    except OSError:
                        log.warning("failed to close parent subprocess file handle", exc_info=True)
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

    def exit_code(self) -> int | None:
        """Return the process exit code, or ``None`` while running / never started.

        Returns:
            The exit code once the process has terminated, else ``None``.
        """
        if self._proc is None:
            return None
        return self._proc.poll()

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
            # The cancel scope of the round currently in flight, if any. The
            # dispatcher's scope is a ContextVar in the submitter's process and
            # cannot cross into this one, so the actor keeps its own and
            # :meth:`cancel_round` is the wire between them.
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
            """Launch the serving subprocess; Ray has set visible devices.

            GPU specialist actors retain control-plane credentials by default;
            benchmark serving ranks opt into scrubbing at their call site.
            Specialists may request ``env_mode="replace"`` to start from their
            filtered mapping without reintroducing actor credentials.

            Returns:
                The launched pid.
            """
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
            """Run one benchmark round to completion; return ``(rc, stdout, stderr)``.

            ``session_remaining_sec`` is a duration, not the submitter's absolute
            session deadline: this actor is a separate process with its own
            ``time.monotonic()`` origin, so only a duration survives the trip.

            The round runs under a cancel scope published in this process, which
            is what gives :meth:`cancel_round` something to raise: the reaper
            inside ``run_with_session_kill`` then stops the tree and names the
            stop exactly as it does on the local path.
            """
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
            """Ask the round in flight to stop itself; return whether there was one.

            Runs in a second method slot (see ``_SERVING_ACTOR_CONCURRENCY``) so
            it is not queued behind the round it is cancelling. Returns as soon
            as the flag is raised: the round is what decides how to stop, and the
            submitter waits for it to come back.

            Args:
                reason: Short cause from the canceller, carried into the stopped
                    round's message.

            Returns:
                ``True`` when a round was asked to stop, ``False`` when the actor
                was idle -- which the submitter reads as "nothing to wait for".
            """
            scope = self._round_scope
            if scope is None:
                return False
            scope.cancel(reason=reason)
            return True

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

        def exit_code(self) -> int | None:
            """Return the supervised process exit code, or ``None`` while running.

            Returns:
                The exit code once the process has terminated, else ``None``.
            """
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
    """Create a ServingActor handle holding ``num_gpus`` (+ optional ``serving_slot``).

    Given more than one method slot so ``cancel_round`` can reach a round that is
    already running; with the default single slot it would wait for the round to
    finish, which is the one thing a cancel cannot do.
    """
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
    """A held Ray GPU lease spanning every round that shares one server.

    Wraps a long-lived :class:`ServingActor` holding ``num_gpus`` (+ optional
    ``serving_slot`` whole-machine mutex). The actor is created on first use and
    stays alive until :meth:`close`. Single-node only.
    """

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
        """Ensure the Ray cluster is up and the serving actor is created.

        Idempotent — a second call is a no-op once the actor exists.
        Raises :exc:`RayInfeasibleError` when the cluster cannot satisfy the lease.
        """
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
        """Run one benchmark round inside the lease's actor; return ``(rc, stdout, stderr)``.

        Drop-in for ``run_with_session_kill``; re-raises ``subprocess.TimeoutExpired``
        on hard timeout. Cluster-ensure failures and Ray worker errors degrade to
        a benchmark failure (rc=1) rather than crashing the session.

        The cancel scope published by the dispatcher is watched for as long as
        the round is in flight, the same as on the local path -- the difference
        is that the round is in another process, so the scope cannot be read
        there and the cancel is forwarded to the actor instead.

        Args:
            cmd: The benchmark command to run inside the actor.
            env: Caller env for the subprocess.
            cwd: Working directory for the subprocess.
            timeout: Hard timeout in seconds.
            soft_deadline_sec: Overtime soft deadline.
            server_log_path: Path to the server log for the watchdogs.
            server_already_ready: Warm reuse round (soft clock from spawn).
            session_remaining_sec: Seconds left on the session budget, as
                produced by ``session_deadline_to_remaining_sec``. The absolute
                deadline the local path uses cannot cross into the actor: it is
                a ``time.monotonic()`` instant, and the actor's clock has its own
                origin. The actor re-anchors this duration onto its own clock.

        Returns:
            ``(returncode, stdout, stderr)`` from the round.
        """
        from ..cancel_channel import cancel_scope_listener  # noqa: PLC0415

        try:
            self.ensure()
        except (RayInfeasibleError, RuntimeError) as exc:
            log.warning("ServingLease.run_session_kill: cluster ensure failed: %r", exc)
            return 1, "", f"ray_ensure_error: {exc}"[:2000]
        # Registered before the round is submitted, so a cancel that arrives
        # while Ray is still scheduling it is one this call is counted as able
        # to hear -- the same window the local path opens around its spawn.
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
        """Wait for a submitted round, forwarding a cancel to the actor if one comes.

        Args:
            ref: The ``ObjectRef`` for the round in flight.
            cmd: The round's command, for the ``TimeoutExpired`` it may raise.
            timeout: The round's hard timeout, for the same reason.
            cancel_scope: The scope to watch, or ``None`` when the caller is not
                running under one, in which case this is a plain blocking wait.

        Returns:
            ``(returncode, stdout, stderr)`` from the round.

        Raises:
            subprocess.TimeoutExpired: When the actor reports a hard timeout.
        """
        import subprocess as _sp  # noqa: PLC0415

        import ray  # noqa: PLC0415

        # Resolve Ray's exception classes defensively. Real ray always exposes
        # both, but this is a failure hot-path: a partial test double or a future
        # ray rename must never turn a benchmark failure into an AttributeError
        # *while handling* the original error. An empty-tuple fallback simply
        # catches nothing, so an unknown error still propagates unchanged.
        _ray_exc = getattr(ray, "exceptions", None)
        _actor_err: Any = getattr(_ray_exc, "RayActorError", ()) if _ray_exc else ()
        _task_err: Any = getattr(_ray_exc, "RayTaskError", ()) if _ray_exc else ()
        try:
            if cancel_scope is None:
                rc, out, err = ray.get(ref)
            else:
                rc, out, err = self._await_or_cancel(ref, cancel_scope=cancel_scope)
        except _actor_err as exc:  # type: ignore[misc]
            # The actor (worker) itself died — e.g. its server OOM-killed the
            # worker, or raylet reaped it. Drop the dead handle so the NEXT round
            # / variant re-creates a fresh actor via ``ensure()``, and surface
            # this round as a benchmark failure (rc=1) instead of letting the
            # actor error propagate and crash the session. Matters now that one
            # lease is reused across a whole round: a mid-round actor death must
            # self-heal rather than cascade to every remaining variant.
            log.warning("ServingLease.run_session_kill: ray actor died: %r", exc)
            self._actor = None
            try:
                from ._ray_backend import mark_ray_backend_unhealthy  # noqa: PLC0415

                mark_ray_backend_unhealthy()
            except Exception:  # noqa: BLE001 - failure recovery must not raise
                pass
            return 1, "", f"ray_actor_error: {exc}"[:2000]
        except _task_err as exc:  # type: ignore[misc]
            # Worker crash / unexpected error: surface as a benchmark failure so
            # the caller's existing rc!=0 handling runs, not a session crash.
            log.warning("ServingLease.run_session_kill: ray worker error: %r", exc)
            return 1, "", f"ray_worker_error: {exc}"[:2000]
        if rc == _ACTOR_TIMEOUT_RC:
            raise _sp.TimeoutExpired(cmd, timeout or 0, output=out or None, stderr=err or None)
        return rc, out, err

    def _await_or_cancel(self, ref: Any, *, cancel_scope: Any) -> tuple[int, str, str]:
        """Block on a round, asking the actor to stop it if the scope is cancelled.

        The round attributes its own stop, exactly as the local path does, so
        the sentinel this returns is the actor's whenever the actor answers.
        Only a wedged actor -- one that has not come back within
        ``CANCEL_ROUND_GRACE_SEC`` of being asked -- is killed, and only then
        does the submitter attribute the stop on its behalf, because otherwise
        an unattributed failure is what the ledger would read.

        Args:
            ref: The ``ObjectRef`` for the round in flight.
            cancel_scope: The scope this call is listening on.

        Returns:
            ``(returncode, stdout, stderr)`` from the round, or an
            ``ORCHESTRATOR_CANCELLED_RETURNCODE`` triple when the actor had to
            be killed.
        """
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
                    # The actor never took the round, or cannot be reached to be
                    # told about it. Either way nothing in there will stop on its
                    # own, so go straight to the kill.
                    asked_at -= CANCEL_ROUND_GRACE_SEC
            elif time.monotonic() - asked_at >= CANCEL_ROUND_GRACE_SEC:
                log.warning(
                    "ServingLease: the actor did not return its cancelled round within %.0fs; "
                    "killing it to release the lease",
                    CANCEL_ROUND_GRACE_SEC,
                )
                # Straight to the kill: an actor that has not answered is not
                # going to answer a graceful stop either, and waiting for one
                # would spend the rest of the window the caller is owed.
                self._kill_actor()
                return (
                    ORCHESTRATOR_CANCELLED_RETURNCODE,
                    "",
                    "the orchestrator cancelled this action; its Ray actor was killed after "
                    f"{CANCEL_ROUND_GRACE_SEC:.0f}s without returning the round",
                )

    def _ask_actor_to_cancel(self, reason: str) -> bool:
        """Tell the actor to stop the round it is running. Never raises.

        Args:
            reason: Short cause, carried into the stopped round's message.

        Returns:
            ``True`` when the actor confirmed it had a round to stop.
        """
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
        """Release the GPU lease: stop the server, then kill the actor. Idempotent.

        The stop comes first because ``ray.kill`` skips ``__ray_terminate__``,
        so the actor's own reaper never runs on that path; the served process is
        deliberately in its own POSIX session, which is exactly what a
        process-group teardown does not reach. Never raises, and the kill still
        happens when the stop does not.
        """
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
        """Ensure the lease on context entry.

        Returns:
            This lease.
        """
        self.ensure()
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Release the lease on context exit.

        Returns:
            ``False`` so exceptions propagate.
        """
        self.close()
        return False


def maybe_serving_lease(
    *,
    num_gpus: float,
    serving_slot: bool = True,
    ensure_log_path: Any = None,
) -> ServingLease | None:
    """Return a :class:`ServingLease` when single-node Ray execution is active.

    The single seam executors use to opt a benchmark unit onto Ray: it returns
    a (not-yet-ensured) lease only when the Ray backend is explicitly enabled
    for a single-node run, else ``None`` (default local path, multi-node,
    ``INFERENCE_OPTIMIZER_RAY_EXEC`` off, or the pytest default). Callers pass
    the result straight into
    ``run_grid(..., serving_lease=lease)`` / ``run_session_kill`` — ``None``
    transparently keeps the existing local-subprocess path — and MUST
    :meth:`ServingLease.close` a non-``None`` lease (typically in a ``finally``)
    to release the GPU lease.

    Args:
        num_gpus: GPUs the lease holds (typically the serving ``TP``).
        serving_slot: Whether to also hold the whole-machine ``serving_slot``
            resource. Defaults ``True`` — every serving-family caller
            (baseline / conc_sweep / sweep / explore) is mutually exclusive on
            the node (§12 T6).
        ensure_log_path: Optional path forwarded to the cluster ensure.

    Returns:
        A lease to route benchmark rounds through, or ``None`` to run locally.
    """
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
    """A held Ray GPU lease that runs a ``needs_gpu`` specialist subprocess.

    Wraps a :func:`make_gpu_specialist_actor` actor holding ``num_gpus``. The
    specialist's entire subprocess runs inside the actor so all GPU commands land
    within Ray's assigned visible devices. Exposes ``start_async`` /
    ``poll_started`` / ``is_alive`` / ``exit_code`` / ``stop`` / ``close``
    (mirroring Popen for the reap loop).
    Single-node only.
    """

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
        # §3.3 non-blocking start: the pending ObjectRef for the actor's
        # ``start`` remote call, and the monotonic clock at submit time so the
        # caller can measure Ray *pending* time separately from the subprocess's
        # *running* wall budget.
        self._start_ref: Any = None
        self._pending_started_monotonic: float | None = None

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
        """Create the actor and SUBMIT the subprocess launch without blocking.

        Stores the pending ObjectRef; the caller must poll :meth:`poll_started`
        until it returns a pid (non-``None``). This is the §3.3 non-blocking
        start: Ray scheduling time is neither charged to the specialist's wall
        budget nor allowed to block the Coordinator's event loop.

        ``env_mode="merge"`` preserves the existing actor-environment overlay
        behavior. ``env_mode="replace"`` uses the supplied filtered environment
        plus Ray's actor-assigned visible-device variables. ``stdin_path`` is
        forwarded as file-backed stdin; when omitted, stdin is ``DEVNULL``.

        Raises :exc:`RayInfeasibleError` for a permanently-unschedulable request
        (caller turns this into a structured task failure).
        """
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
        self._pending_started_monotonic = time.monotonic()

    def poll_started(self) -> int | None:
        """Non-blocking poll for the launched pid.

        Returns the pid once Ray has scheduled the actor and the subprocess has
        launched, else ``None`` while still pending. Never blocks (uses
        ``ray.wait(..., timeout=0)``), so a caller can interleave it with
        ``asyncio.sleep`` and keep the event loop responsive (§3.3).
        """
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
        """Return the launched pid, or ``None`` before it has been resolved.

        Returns:
            The pid, or ``None``.
        """
        return self._pid

    def is_alive(self) -> bool:
        """Return whether the specialist subprocess is still running.

        Returns ``False`` when the actor is unreachable or the probe times out
        (treated as transient; caller retries on the next poll tick).
        """
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
        """Kill the actor, releasing the GPU lease. Idempotent, never raises."""
        if self._actor is None:
            return
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
    """Return a :class:`GpuSpecialistLease` when single-node Ray execution is active.

    Mirrors :func:`maybe_serving_lease`: the single seam the dispatcher uses to
    route a ``needs_gpu`` specialist's whole subprocess into a Ray ``num_gpus``
    actor (§12 T4). Returns ``None`` (keep the legacy SQLite-gpu-id device path)
    for multi-node, ``INFERENCE_OPTIMIZER_RAY_EXEC`` off, the pytest default, or
    a non-positive ``num_gpus``.

    Args:
        num_gpus: GPUs the specialist would lease.
        serving_slot: Whether the specialist also holds the whole-machine
            ``serving_slot`` (whole-machine / bench-capable ``gpu_research_lane``
            specialists — mutually exclusive with serving, §12 T6). ``False``
            (default) for the serving-disjoint pool.
        ensure_log_path: Optional path forwarded to the cluster ensure.

    Returns:
        A lease to run the specialist subprocess through, or ``None`` (local).
    """
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


# ── P4 (skeleton) — multi-node serving via placement group + rank actors ──────
# Gated OFF by default; wired in only when INFERENCE_OPTIMIZER_RAY_MN_SERVING is set.


def _mn_serving_ray_enabled() -> bool:
    """Return whether the P4 Ray multi-node serving skeleton is opted into.

    Off by default: multi-node is deferred (decisions 4/5). ``True`` only when
    ``INFERENCE_OPTIMIZER_RAY_MN_SERVING`` is explicitly truthy.

    Returns:
        ``True`` when the multi-node Ray serving path is enabled.
    """
    return os.environ.get("INFERENCE_OPTIMIZER_RAY_MN_SERVING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _make_serving_placement_group(nodes: int, gpus_per_node: float, *, serving_slot: bool):
    """Reserve one whole-node bundle per serving rank (STRICT_SPREAD).

    Each bundle asks for ``gpus_per_node`` GPUs (+ optional ``serving_slot``) and
    STRICT_SPREAD forces one bundle per distinct node, so the rank actors below
    land one-per-node and collectively hold every serving card until the group
    is torn down. Blocks until the group is scheduled.

    Args:
        nodes: Number of serving nodes (bundles).
        gpus_per_node: GPUs each node's rank holds.
        serving_slot: Whether each bundle also reserves the node's
            ``serving_slot`` (declared on GPU worker pods by the multi-node
            maintainer, §4.5).

    Returns:
        The ready Ray ``PlacementGroup``.
    """
    import ray  # noqa: PLC0415
    from ray.util.placement_group import placement_group  # noqa: PLC0415

    bundle: dict[str, float] = {"GPU": float(gpus_per_node)}
    if serving_slot:
        bundle["serving_slot"] = 1
    pg = placement_group([dict(bundle) for _ in range(int(nodes))], strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    return pg


def _remove_serving_placement_group(pg: Any) -> None:
    """Release a serving placement group's reserved bundles.

    Args:
        pg: The placement group to remove.
    """
    from ray.util.placement_group import remove_placement_group  # noqa: PLC0415

    remove_placement_group(pg)


def _make_rank_actor(pg: Any, bundle_index: int, num_gpus: float, *, serving_slot: bool):
    """Create one serving rank actor pinned to ``pg``'s ``bundle_index``.

    Args:
        pg: The serving placement group.
        bundle_index: Bundle (node) this rank is pinned to.
        num_gpus: GPUs this rank holds.
        serving_slot: Whether the rank also holds ``serving_slot``.

    Returns:
        A Ray actor handle for the rank (a ServingActor pinned to the bundle).
    """
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy  # noqa: PLC0415

    actor_cls: Any = _serving_actor_body()
    resources = {"serving_slot": 1} if serving_slot else None
    return actor_cls.options(
        num_gpus=num_gpus,
        resources=resources,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=int(bundle_index),
        ),
    ).remote()


class ServingGroupManager:
    """Multi-node serving held by a Ray placement group + per-node rank actors.

    **P4 SKELETON — gated OFF by default.** Reserves a STRICT_SPREAD placement
    group and launches one :class:`ServingActor` rank per bundle. Interface
    mirrors :class:`ServingLease` for eventual single-node parity.
    """

    def __init__(
        self,
        *,
        nodes: int,
        gpus_per_node: float,
        serving_slot: bool = True,
        ensure_log_path: Any = None,
    ) -> None:
        self._nodes = int(nodes)
        self._gpus_per_node = float(gpus_per_node)
        self._serving_slot = bool(serving_slot)
        self._ensure_log_path = ensure_log_path
        self._pg: Any = None
        self._ranks: list[Any] = []
        self._pids: list[int] = []

    def start(
        self,
        rank_cmds: list[list[str]],
        *,
        envs: list[dict[str, str] | None] | None = None,
        cwds: list[str | None] | None = None,
        log_paths: list[str | None] | None = None,
    ) -> list[int]:
        """Reserve the placement group and launch one server rank per node.

        Args:
            rank_cmds: One command per rank (``len == nodes``).
            envs: Optional per-rank env overlays.
            cwds: Optional per-rank working directories.
            log_paths: Optional per-rank stdout/stderr log paths.

        Returns:
            The launched rank pids (one per node).

        Raises:
            ValueError: If ``len(rank_cmds)`` does not match ``nodes``.
        """
        if len(rank_cmds) != self._nodes:
            raise ValueError(f"expected {self._nodes} rank_cmds, got {len(rank_cmds)}")
        import ray  # noqa: PLC0415

        from ._ray_backend import get_ray_backend  # noqa: PLC0415

        get_ray_backend().ensure(log_path=self._ensure_log_path)
        self._pg = _make_serving_placement_group(self._nodes, self._gpus_per_node, serving_slot=self._serving_slot)
        self._ranks = []
        self._pids = []
        for i, cmd in enumerate(rank_cmds):
            actor = _make_rank_actor(self._pg, i, self._gpus_per_node, serving_slot=self._serving_slot)
            self._ranks.append(actor)
            pid = int(
                ray.get(
                    actor.start.remote(
                        cmd,
                        env=(envs[i] if envs else None),
                        cwd=(cwds[i] if cwds else None),
                        log_path=(log_paths[i] if log_paths else None),
                        scrub_benchmark_env=True,
                    )
                )
            )
            self._pids.append(pid)
        return list(self._pids)

    def pids(self) -> list[int]:
        """Return the launched rank pids.

        Returns:
            The per-node rank pids.
        """
        return list(self._pids)

    def ranks_alive(self) -> list[bool]:
        """Return per-rank liveness (``False`` for a rank whose actor is gone).

        Returns:
            One bool per rank.
        """
        if not self._ranks:
            return []
        import ray  # noqa: PLC0415

        out: list[bool] = []
        for actor in self._ranks:
            try:
                out.append(bool(ray.get(actor.is_alive.remote())))
            except Exception:  # noqa: BLE001 — a dead rank reads as not-alive
                out.append(False)
        return out

    def is_alive(self) -> bool:
        """Return whether every rank server is still running.

        Returns:
            ``True`` when all ranks are alive (and at least one exists).
        """
        alive = self.ranks_alive()
        return bool(alive) and all(alive)

    def stop(self) -> None:
        """Reap every rank's server subprocess tree (keeps actors/PG alive)."""
        if not self._ranks:
            return
        import ray  # noqa: PLC0415

        for actor in self._ranks:
            try:
                ray.get(actor.stop.remote())
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass

    def close(self) -> None:
        """Kill all rank actors and remove the placement group. Idempotent."""
        import ray  # noqa: PLC0415

        for actor in self._ranks:
            try:
                ray.kill(actor)
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
        self._ranks = []
        self._pids = []
        if self._pg is not None:
            try:
                _remove_serving_placement_group(self._pg)
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
            self._pg = None


def maybe_serving_group_manager(
    *,
    nodes: int,
    gpus_per_node: float,
    serving_slot: bool = True,
    ensure_log_path: Any = None,
) -> ServingGroupManager | None:
    """Return a :class:`ServingGroupManager` when the P4 MN-serving path is opted in.

    **Off by default (decisions 4/5).** Returns ``None`` unless the run is
    multi-node AND ``INFERENCE_OPTIMIZER_RAY_MN_SERVING`` is set — so the live
    detached ``restart_server_for_round`` path is completely unaffected until a
    multi-node maintainer opts in and wires it.

    Args:
        nodes: Number of serving nodes.
        gpus_per_node: GPUs each rank holds.
        serving_slot: Whether each rank reserves the node ``serving_slot``.
        ensure_log_path: Optional path forwarded to the cluster ensure.

    Returns:
        A group manager to start rank servers through, or ``None`` (legacy path).
    """
    if nodes <= 0 or gpus_per_node <= 0:
        return None
    from ._multi_node_env import is_multi_node  # noqa: PLC0415

    if not is_multi_node() or not _mn_serving_ray_enabled():
        return None
    return ServingGroupManager(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        serving_slot=serving_slot,
        ensure_log_path=ensure_log_path,
    )


__all__ = [
    "GpuSpecialistLease",
    "ManagedServerProcess",
    "RayInfeasibleError",
    "ServingGroupManager",
    "ServingLease",
    "make_gpu_specialist_actor",
    "make_serving_actor",
    "maybe_gpu_specialist_lease",
    "maybe_serving_group_manager",
    "maybe_serving_lease",
]
