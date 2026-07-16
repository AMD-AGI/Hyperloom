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


# Sentinel returncode a serving actor returns when the benchmark subprocess hit
# its hard timeout. Ray cannot re-raise ``subprocess.TimeoutExpired`` across the
# worker boundary reliably (its constructor needs args), so the actor reports
# the timeout as this returncode and :meth:`ServingLease.run_session_kill`
# re-raises the real ``TimeoutExpired`` for callers that already handle it.
# Distinct from the -909/-910/-911 watchdog sentinels in ``_subprocess_kill``.
_ACTOR_TIMEOUT_RC: int = -912


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
        ):
            """Run one benchmark round to completion inside this actor's lease.

            The actor holds ``num_gpus`` for its whole lifetime, so every round
            submitted here (boot, warmup, reuse, measure) runs on the same
            worker under the same GPU lease — a detached server booted by an
            earlier round is still covered by the lease when a later round
            re-attaches (plan §4.2 / §12 T1). Ray has already set
            ``*_VISIBLE_DEVICES`` on this worker; the subprocess inherits them
            (the caller env's device vars are dropped by
            :func:`_run_subprocess_worker`).

            Returns:
                ``(returncode, stdout, stderr)``. A hard-timeout is surfaced as
                the :data:`_ACTOR_TIMEOUT_RC` returncode (never a raised
                exception, which Ray cannot faithfully reconstruct).
            """
            import subprocess as _sp  # noqa: PLC0415

            from ._ray_backend import _run_subprocess_worker  # noqa: PLC0415

            try:
                return _run_subprocess_worker(
                    cmd=cmd,
                    env=env,
                    cwd=cwd,
                    timeout_s=timeout,
                    soft_deadline_sec=soft_deadline_sec,
                    server_log_path=server_log_path,
                    server_already_ready=server_already_ready,
                )
            except _sp.TimeoutExpired as exc:
                return _ACTOR_TIMEOUT_RC, "", f"TimeoutExpired: {exc}"

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


def make_gpu_specialist_actor(num_gpus: float, *, serving_slot: bool = False):
    """Create a GpuSpecialistActor handle holding ``num_gpus``.

    Args:
        num_gpus: GPUs the specialist actor leases.
        serving_slot: Whether to also hold the whole-machine ``serving_slot``
            resource. ``False`` (default) for the serving-disjoint
            ``gpu_specialist_pool`` — it runs on cards disjoint from serving.
            ``True`` for the whole-machine / bench-capable ``gpu_research_lane``
            specialists that are mutually exclusive with serving (§4.5 / §12 T6).

    Returns:
        A Ray actor handle for the GpuSpecialistActor (a ServingActor,
        optionally without the serving slot).
    """
    actor_cls: Any = _serving_actor_body()
    resources = {"serving_slot": 1} if serving_slot else None
    return actor_cls.options(num_gpus=num_gpus, resources=resources).remote()


class ServingLease:
    """A held Ray GPU lease spanning every round that shares one server (§12 T1).

    Wraps a long-lived :class:`ServingActor` holding ``num_gpus``. The actor is
    created on first use and stays alive — keeping the GPU lease — until
    :meth:`close`, so Ray never reassigns those cards while the shared server is
    up. All rounds that reuse one persistent server (conc_sweep arm boot+reuse,
    baseline/explore warmup+measure, ``run_grid`` auto-warmup) run through the
    same lease, satisfying the hard invariant that a GPU process must live
    inside a Ray lease that covers its whole lifetime.

    Single-node only (the caller gates on ``not is_multi_node()``). Since P3 the
    lease also holds the whole-machine ``serving_slot`` custom resource (declared
    on the head by ``ensure_ray_cluster``, §12 T6): it is the authoritative
    physical mutex making serving ⊥ serving ⊥ profile ⊥ benchmark ⊥ gpu_research
    — a second serving-family lease PENDs on ``serving_slot`` until this one is
    released. GPU specialists request ``num_gpus`` only (serving-disjoint) and do
    not take the slot.
    """

    def __init__(
        self,
        *,
        num_gpus: float,
        serving_slot: bool = True,
        ensure_log_path: Any = None,
    ) -> None:
        """Configure the lease (the actor is created lazily on first use).

        Args:
            num_gpus: GPUs the lease holds (typically the serving ``TP``).
            serving_slot: Whether to also hold the ``serving_slot`` custom
                resource. Defaults to ``True`` (serving-family whole-machine
                mutex, §12 T6); ``ensure_ray_cluster`` declares the resource.
            ensure_log_path: Optional path forwarded to the cluster ensure.
        """
        self._num_gpus = float(num_gpus)
        self._serving_slot = bool(serving_slot)
        self._ensure_log_path = ensure_log_path
        self._actor: Any = None

    def ensure(self) -> None:
        """Ensure the Ray cluster is up and the serving actor is created.

        Idempotent — a second call is a no-op once the actor exists.
        """
        if self._actor is not None:
            return
        from ._ray_backend import get_ray_backend  # noqa: PLC0415

        get_ray_backend().ensure(log_path=self._ensure_log_path)
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
    ) -> tuple[int, str, str]:
        """Run one benchmark round inside the lease's actor; return the triple.

        Drop-in for a local ``run_with_session_kill(...)`` call: returns
        ``(returncode, stdout, stderr)`` and re-raises ``subprocess.TimeoutExpired``
        on a hard timeout so existing caller-side timeout handling is unchanged.
        A worker-side Ray failure degrades to a benchmark failure (non-zero rc)
        rather than crashing the session.

        Args:
            cmd: Benchmark command to run.
            env: Caller env (device vars are dropped inside the worker).
            cwd: Working directory.
            timeout: Hard timeout in seconds.
            soft_deadline_sec: Overtime soft deadline.
            server_log_path: Server log path for the watchdogs.
            server_already_ready: Warm-reuse soft-clock semantics.

        Returns:
            ``(returncode, stdout, stderr)``.

        Raises:
            subprocess.TimeoutExpired: When the round hit its hard timeout.
        """
        import subprocess as _sp  # noqa: PLC0415

        import ray  # noqa: PLC0415

        self.ensure()
        ref = self._actor.run_blocking.remote(
            cmd,
            env=env,
            cwd=cwd,
            timeout=timeout,
            soft_deadline_sec=soft_deadline_sec,
            server_log_path=server_log_path,
            server_already_ready=server_already_ready,
        )
        try:
            rc, out, err = ray.get(ref)
        except ray.exceptions.RayTaskError as exc:
            # Worker crash / unexpected error: surface as a benchmark failure so
            # the caller's existing rc!=0 handling runs, not a session crash.
            log.warning("ServingLease.run_session_kill: ray worker error: %r", exc)
            return 1, "", f"ray_worker_error: {exc}"[:2000]
        if rc == _ACTOR_TIMEOUT_RC:
            raise _sp.TimeoutExpired(cmd, timeout or 0, output=out or None, stderr=err or None)
        return rc, out, err

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
    a (not-yet-ensured) lease when the Ray backend should run this work and the
    run is single-node, else ``None`` (multi-node, ``INFERENCE_OPTIMIZER_RAY_EXEC``
    off, or the pytest default). Callers pass the result straight into
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
    """A held Ray GPU lease that runs a ``needs_gpu`` specialist subprocess (§12 T4).

    Wraps a :func:`make_gpu_specialist_actor` actor holding ``num_gpus``. The
    specialist's *entire* subprocess (the agent runtime and every Bash / GPU
    command it spawns) runs inside the actor, so any GPU command lands within
    Ray's assigned visible devices — the specialist can no longer reach a card
    outside its lease (ray_modify.plan.md §4.2 / §8). Unlike a benchmark round
    the process is long-lived and reaper-polled, so this exposes ``start`` /
    ``is_alive`` / ``exit_code`` / ``stop`` (mirroring a ``Popen`` for the reap
    loop) rather than a single blocking call; ``close`` releases the lease.

    Serving-disjoint (``gpu_specialist_pool``) specialists request ``num_gpus``
    only and run on cards disjoint from serving; whole-machine / bench-capable
    (``gpu_research_lane``) specialists set ``serving_slot=True`` so they are
    mutually exclusive with serving on the node's ``serving_slot`` (§12 T6).

    Single-node only (callers gate via :func:`maybe_gpu_specialist_lease`). The
    physical card assignment is Ray's (``num_gpus``); the Coordinator's SQLite
    pool is now only capacity/TTL accounting (decision 3 / §12 T5).
    """

    def __init__(
        self,
        *,
        num_gpus: float,
        serving_slot: bool = False,
        ensure_log_path: Any = None,
    ) -> None:
        """Configure the lease (the actor is created on :meth:`start`).

        Args:
            num_gpus: GPUs the specialist actor leases.
            serving_slot: Whether to also hold the whole-machine ``serving_slot``
                resource — ``True`` for whole-machine / bench-capable
                ``gpu_research_lane`` specialists (mutually exclusive with
                serving, §12 T6); ``False`` (default) for the serving-disjoint
                pool.
            ensure_log_path: Optional path forwarded to the cluster ensure.
        """
        self._num_gpus = float(num_gpus)
        self._serving_slot = bool(serving_slot)
        self._ensure_log_path = ensure_log_path
        self._actor: Any = None
        self._pid: int | None = None

    def start(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        log_path: str | None = None,
    ) -> int:
        """Ensure the cluster, create the actor, and launch the subprocess.

        Ray sets ``*_VISIBLE_DEVICES`` in the actor's worker; the subprocess
        inherits them (the actor drops the caller env's device vars).

        Args:
            cmd: The specialist command to launch.
            env: Caller env (device vars are dropped inside the actor).
            cwd: Working directory for the subprocess.
            log_path: Path the subprocess's stdout/stderr are written to (read
                by the caller's reaper — single-node, same host).

        Returns:
            The launched subprocess pid.
        """
        import ray  # noqa: PLC0415

        from ._ray_backend import get_ray_backend  # noqa: PLC0415

        get_ray_backend().ensure(log_path=self._ensure_log_path)
        self._actor = make_gpu_specialist_actor(
            self._num_gpus, serving_slot=self._serving_slot
        )
        self._pid = int(
            ray.get(self._actor.start.remote(cmd, env=env, cwd=cwd, log_path=log_path))
        )
        return self._pid

    def pid(self) -> int | None:
        """Return the launched pid, or ``None`` before :meth:`start`.

        Returns:
            The pid, or ``None``.
        """
        return self._pid

    def is_alive(self) -> bool:
        """Return whether the specialist subprocess is still running.

        Returns:
            ``True`` when the actor reports the process alive; ``False`` when it
            has exited, was never started, or the actor is unreachable.
        """
        if self._actor is None:
            return False
        import ray  # noqa: PLC0415

        try:
            return bool(ray.get(self._actor.is_alive.remote()))
        except Exception:  # noqa: BLE001 — a dead actor reads as not-alive
            return False

    def exit_code(self) -> int | None:
        """Return the subprocess exit code, or ``None`` while running.

        Returns:
            The exit code once the process terminates, else ``None``.
        """
        if self._actor is None:
            return None
        import ray  # noqa: PLC0415

        try:
            return ray.get(self._actor.exit_code.remote())
        except Exception:  # noqa: BLE001
            return None

    def stop(self) -> None:
        """Reap the specialist subprocess tree (keeps the actor/lease alive).

        Used by the reaper on wall-budget / stall kills; the lease itself is
        released later by :meth:`close`. Never raises.
        """
        if self._actor is None:
            return
        import ray  # noqa: PLC0415

        try:
            ray.get(self._actor.stop.remote())
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
#
# ray_modify.plan.md §4.2 / §6 P4. This is a SKELETON building block only,
# gated OFF by default (decisions 4/5 defer multi-node out of this round). The
# live multi-node serving path stays the detached
# ``_multi_node_server_lifecycle.restart_server_for_round`` (SSH / RayJob
# Dashboard) — this scaffolds the eventual replacement so a serving process's
# whole lifetime is held by Ray rank actors (no detached RayJob server escaping
# the lease), mirroring how P0's ServingActor/ManagedServerProcess did for
# single-node. Wiring it into ``restart_server_for_round`` and downgrading
# ``magpie_remote_env()`` to a placement strategy is left to the multi-node
# maintainer (§12 footer). Nothing here runs unless a caller explicitly opts in
# via :func:`maybe_serving_group_manager` (multi-node + the env flag below).


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

    **P4 SKELETON — gated OFF by default (decisions 4/5).** Reserves a
    STRICT_SPREAD placement group of ``nodes`` bundles (each ``gpus_per_node``
    GPUs, optionally + ``serving_slot``) and launches one :class:`ServingActor`
    rank per bundle. Each rank runs its server-rank subprocess via
    ``ManagedServerProcess`` (new POSIX session + ``PR_SET_PDEATHSIG`` + tree
    reap), so a rank server dies with its actor — no detached RayJob server
    escapes the lease, and rank actors hold their GPUs until :meth:`stop` /
    :meth:`close` (§4.2). The rank *commands* are supplied by the caller; this
    skeleton does not invent the distributed sglang/vLLM bootstrap (rendezvous /
    head-vs-worker roles / KV transport) — that, plus wiring into
    ``restart_server_for_round`` and the ``magpie_remote_env`` placement
    strategy, is left to the multi-node maintainer (§12 footer).

    Interface mirrors :class:`ServingLease` (``start`` / ``is_alive`` / ``stop``
    / ``close``) so the eventual executor wiring matches the single-node path.
    """

    def __init__(
        self,
        *,
        nodes: int,
        gpus_per_node: float,
        serving_slot: bool = True,
        ensure_log_path: Any = None,
    ) -> None:
        """Configure the group (the PG + rank actors are created on :meth:`start`).

        Args:
            nodes: Number of serving nodes (rank actors / bundles).
            gpus_per_node: GPUs each rank holds.
            serving_slot: Whether each rank reserves the node ``serving_slot``.
            ensure_log_path: Optional path forwarded to the cluster ensure.
        """
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
        self._pg = _make_serving_placement_group(
            self._nodes, self._gpus_per_node, serving_slot=self._serving_slot
        )
        self._ranks = []
        self._pids = []
        for i, cmd in enumerate(rank_cmds):
            actor = _make_rank_actor(
                self._pg, i, self._gpus_per_node, serving_slot=self._serving_slot
            )
            self._ranks.append(actor)
            pid = int(
                ray.get(
                    actor.start.remote(
                        cmd,
                        env=(envs[i] if envs else None),
                        cwd=(cwds[i] if cwds else None),
                        log_path=(log_paths[i] if log_paths else None),
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
    "ServingGroupManager",
    "ServingLease",
    "make_gpu_specialist_actor",
    "make_serving_actor",
    "maybe_gpu_specialist_lease",
    "maybe_serving_group_manager",
    "maybe_serving_lease",
]
