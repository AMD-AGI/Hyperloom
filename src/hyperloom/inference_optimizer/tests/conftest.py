# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest hooks and shared helpers for the inference_optimizer tests package."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.session.paths import make_session_dir


@pytest.fixture(autouse=True)
def _isolate_session_layout_env(monkeypatch, tmp_path_factory):
    """Drop the session-dir pin and point MULTI_NODE_STATE_FILE at a missing sentinel so tests run single-node."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    mn_state_sentinel = tmp_path_factory.mktemp("mn_state") / "missing_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(mn_state_sentinel))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)


@pytest.fixture(autouse=True)
def _restore_recording_session_binding():
    """Undo any recording-session binding a test leaves behind.

    ``Coordinator.__init__`` binds the session and drops the token, so a test
    that constructs one leaves every later test in the worker running with a
    session bound. Restoring a token is the only way back -- no public call
    unbinds to "nothing" -- hence the reach for the module's ContextVar.
    """
    from hyperloom.inference_optimizer.session import session_binding

    token = session_binding._CURRENT_SESSION.set(session_binding._CURRENT_SESSION.get())
    try:
        yield
    finally:
        session_binding._CURRENT_SESSION.reset(token)


def _bootstrap_kernel_agent_env() -> None:
    """Point HYPERLOOM_KERNEL_AGENT_ROOT at the in-repo kernel-agent checkout."""
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    repo = Path(__file__).resolve().parents[4]
    kernel_agent = repo / "src" / "hyperloom" / "agents" / "kernel"
    if kernel_agent.is_dir():
        os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] = str(kernel_agent)


_bootstrap_kernel_agent_env()


def enable_multi_node(monkeypatch, nodes: int = 2) -> None:
    """Put the executors in multi-node mode with a no-op per-round server restart."""
    from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as mnl

    async def _no_restart(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", str(nodes))
    monkeypatch.setattr(mnl, "restart_server_for_round", _no_restart)


def launches_by_round_slot(recorded: list[dict]) -> dict[str, dict]:
    """Index recorded benchmark launches by the output slot each round ran in."""
    return {launch["round_slot"]: launch for launch in recorded}


def seed_target_analysis_marker(session_dir: Path) -> Path:
    """Write a ``no_target_gpu_configured`` marker JSON at the session dir."""
    from hyperloom.inference_optimizer.session.session_paths import target_baseline_json

    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "skipped",
                "reason": "no_target_gpu_configured",
                "warning": "compare_against_gpu is empty",
            }
        ),
        encoding="utf-8",
    )
    return path


def seed_kernel_keep(
    state,
    kernel_id: str,
    *,
    decision: str = "KEEP",
    micro: float = 1.5,
    source_file: str = "",
    artifact: str = "",
    task_group_key: str = "",
) -> str:
    """Put one attempt row and, for a KEEP, its queued patch into ``state``."""
    from hyperloom.orchestrator.kernel._kernel_decisions import (
        _queue_kernel_keep,
        _stable_kernel_task_key,
    )

    task_key = _stable_kernel_task_key(
        task_group_key=task_group_key,
        kernel_id=kernel_id,
        source_file=source_file,
    )
    entry = {
        "current_kernel_id": kernel_id,
        "task_group_key": task_group_key,
        "last_decision": decision,
        "last_status": "ok",
        "last_micro_speedup": micro,
        "last_source_file": source_file,
        "last_artifact_path": artifact,
        "attempts": 1,
        "last_ts": f"2026-01-01T00:00:{len(state.kernel_opt_task_attempts):02d}+00:00",
    }
    state.kernel_opt_task_attempts[task_key] = entry
    _queue_kernel_keep(state, task_key=task_key, kernel_id=kernel_id, entry=entry)
    return task_key


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    """A fresh session dir under an isolated ``USER_DATA_PATH``, seeded with the ``no_target_gpu_configured`` target-analysis marker."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    seed_target_analysis_marker(sd)
    return sd


def init_git_repo(
    path: Path,
    *,
    seed_file: str = "src.py",
    seed_text: str = "def f():\n    return 1\n",
) -> None:
    """Initialise a minimal git repo with one commit under ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Hyperloom Test"
    env["GIT_AUTHOR_EMAIL"] = "hyperloom@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True,
        capture_output=True,
        env=env,
    )
    (path / seed_file).write_text(seed_text, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init"],
        check=True,
        capture_output=True,
        env=env,
    )


def git_commit_all(path: Path, message: str) -> None:
    """Stage everything under ``path`` and commit with a fixed non-interactive identity."""
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Hyperloom Test"
    env["GIT_AUTHOR_EMAIL"] = "hyperloom@test.local"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", message],
        check=True,
        capture_output=True,
        env=env,
    )


class _BuildFakeCoordinator:
    """Minimal coordinator surface for off-loop targeted-build tests."""

    def __init__(self, session_dir: Path, db) -> None:
        from hyperloom.orchestrator.bus.resource_lock import (
            ResourceLockManager,
            SqliteLeaseBackend,
        )
        from hyperloom.orchestrator.state.shared_state import SharedState
        from hyperloom.orchestrator.state.task_registry import TaskRegistry

        self.session_dir = session_dir
        self.tasks = TaskRegistry(db)
        self.locks = ResourceLockManager(SqliteLeaseBackend(db))
        self.shared_state = SharedState()


@pytest.fixture
def build_coord(tmp_path):
    """Fake coordinator backed by a temp DB for targeted-build lifecycle tests."""
    from hyperloom.orchestrator.bus.storage import SqliteConnection
    from hyperloom.orchestrator.bus.storage.schema import ensure_schema

    db = SqliteConnection(tmp_path / "coordinator.db")
    ensure_schema(db.raw)
    fc = _BuildFakeCoordinator(tmp_path, db)
    yield fc
    db.close()


@pytest.fixture
def build_lifecycle(build_coord):
    """``BuildLifecycleCollaborator`` bound to the ``build_coord`` fixture."""
    from hyperloom.orchestrator.loop.build_lifecycle import BuildLifecycleCollaborator

    return BuildLifecycleCollaborator(build_coord)


def patch_integrate_patch_roots(monkeypatch, tmp_path: Path) -> None:
    """Register common tmp_path framework repos as integrate_patch search roots."""
    from hyperloom.orchestrator.actions.executors import integrate_patch as ip
    from hyperloom.orchestrator.framework import paths as fp

    real = fp.resolve_kernel_search_roots

    def _merged() -> tuple[str, ...]:
        # The tmp repos lead: a relative artifact target must land in the tree
        # the test built, not in whatever framework happens to be installed on
        # the host running the suite.
        merged: list[str] = []
        for name in ("fw", "repo", "framework"):
            cand = tmp_path / name
            if cand.is_dir():
                merged.append(str(cand.resolve()))
        for root in real():
            if root not in merged:
                merged.append(root)
        return tuple(merged)

    monkeypatch.setattr(ip, "resolve_kernel_search_roots", _merged)


# Progress cadence: how long a long-running path may go unreported


# Production seconds per real test second.
PROGRESS_TIME_SCALE: float = 600.0


class ProgressCadence:
    """Records when a path reported progress, on a simulated production clock."""

    def __init__(self, scale: float = PROGRESS_TIME_SCALE, clock=None) -> None:
        """Start the clock at zero.

        Args:
            scale (float): Production seconds per real second.
            clock: A ``VirtualClock`` to charge the simulated work to instead of
                a private counter, for a test that also drives a scripted round
                and needs both to agree about when things happened.
        """
        self.scale = scale
        self.notes: list[dict] = []
        self.reported_at: list[float] = []
        self._clock = clock
        self._elapsed = 0.0

    def now(self) -> float:
        """Production seconds of simulated work done so far."""
        return self._clock.elapsed if self._clock is not None else self._elapsed

    def sleep(self, simulated_s: float) -> None:
        """Charge ``simulated_s`` production seconds, blocking the real time they map to.

        The real block is what gives the heartbeat driver — ticking on the same
        compressed timescale — its chance to notice the output and report.

        Args:
            simulated_s (float): Production seconds the simulated child spent.
        """
        if self._clock is not None:
            self._clock.advance(simulated_s)
        else:
            self._elapsed += simulated_s
        time.sleep(simulated_s / self.scale)

    def sink(self):
        """Return the ambient progress sink to pass to ``progress_scope``."""

        async def _sink(**note) -> None:
            self.notes.append(note)
            self.reported_at.append(self.now())

        return _sink

    def widest_silence(self) -> float:
        """Longest unreported stretch, in production seconds."""
        marks = [0.0, *self.reported_at, self.now()]
        return max(later - earlier for earlier, later in zip(marks, marks[1:]))


@pytest.fixture
def progress_cadence(monkeypatch) -> "ProgressCadence":
    """A :class:`ProgressCadence` with the heartbeat tick on the same timescale."""
    from hyperloom.orchestrator.trace import task_progress

    monkeypatch.setattr(
        task_progress,
        "_OUTPUT_HEARTBEAT_INTERVAL_S",
        task_progress._OUTPUT_HEARTBEAT_INTERVAL_S / PROGRESS_TIME_SCALE,
    )
    return ProgressCadence()


def chatty_child(cadence: ProgressCadence, inner, *, blocks_for_s: float, line_every_s: float):
    """Wrap a fake ``run_with_session_kill`` so its child talks while it blocks."""

    def _run(cmd, *args, on_output=None, **kwargs):
        for _ in range(int(blocks_for_s / line_every_s)):
            cadence.sleep(line_every_s)
            if on_output is not None:
                on_output()
        return inner(cmd, *args, **kwargs)

    return _run


def suppression_window_s() -> float:
    """The silence past which robustness accuses an agent of stalling."""
    from hyperloom.agents.robustness.signals.stall import StallConfig

    return StallConfig().stall_timeout_s


class _RayDoubleActorClass:
    """The ``@ray.remote`` class: ``.options(...)`` then ``.remote()`` for a handle."""

    def __init__(self, cls: type) -> None:
        self._cls = cls
        self._options: dict = {}

    def options(self, **opts):
        self._options = dict(opts)
        return self

    def remote(self, *args, **kwargs) -> "_RayDoubleActorHandle":
        return _RayDoubleActorHandle(self._cls(*args, **kwargs), self._options)


class _RayDoubleActorHandle:
    """An actor handle whose methods run in a pool sized like the real actor's."""

    def __init__(self, obj, options: dict) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._obj = obj
        self._pool = ThreadPoolExecutor(max_workers=int(options.get("max_concurrency", 1) or 1))
        self.killed = False

    def __getattr__(self, name: str):
        from types import SimpleNamespace

        method = getattr(self._obj, name)
        return SimpleNamespace(remote=lambda *a, **kw: self._pool.submit(method, *a, **kw))


class RayDouble:
    """A ``ray`` module stand-in that runs actor methods in real threads."""

    class exceptions:  # noqa: N801 — mirrors the ray.exceptions namespace
        class RayActorError(Exception):
            pass

        class RayTaskError(Exception):
            pass

        class GetTimeoutError(Exception):
            pass

    def __init__(self) -> None:
        self.killed: list = []

    def remote(self, cls: type) -> _RayDoubleActorClass:
        return _RayDoubleActorClass(cls)

    def cluster_resources(self) -> dict:
        return {"CPU": 8.0, "GPU": 8.0, "serving_slot": 1.0}

    def get(self, ref, timeout: float | None = None):
        return ref.result(timeout)

    def wait(self, refs: list, *, num_returns: int = 1, timeout: float | None = None):
        from concurrent.futures import FIRST_COMPLETED
        from concurrent.futures import wait as futures_wait

        done, not_done = futures_wait(refs, timeout=timeout, return_when=FIRST_COMPLETED)
        return list(done)[:num_returns], list(not_done)

    def kill(self, actor) -> None:
        actor.killed = True
        self.killed.append(actor)


@pytest.fixture
def serving_lease_on_a_ray_double(monkeypatch):
    """A real :class:`ServingLease` over :class:`RayDouble`, closed on teardown."""
    import sys
    from types import SimpleNamespace

    from hyperloom.orchestrator.actions.executors import _ray_backend as rb
    from hyperloom.orchestrator.actions.executors import _ray_serving as rs

    monkeypatch.setitem(sys.modules, "ray", RayDouble())
    monkeypatch.setattr(rb, "get_ray_backend", lambda: SimpleNamespace(ensure=lambda **_kw: None))
    with rs.ServingLease(num_gpus=1) as lease:
        yield lease


@pytest.fixture
def virtual_clock():
    """A :class:`VirtualClock` for tests whose subject is a deadline.

    :class:`ProgressCadence` measures one path's reporting gaps and blocks for
    real (scaled) time so a heartbeat driver gets to run; that is the right
    instrument for cadence and the wrong one for a multi-tick round, where
    nothing may block at all and the readings production takes have to be the
    clock's. This one is that clock, and can be handed to ``ProgressCadence``
    so a test using both keeps one timeline.
    """
    from hyperloom.orchestrator.rehearsal import VirtualClock

    return VirtualClock()


class NoLaunchBackendInstalled(BaseException):
    """A test launched a subprocess before installing a launch backend.

    A ``BaseException`` on purpose. Several production launch sites sit inside
    a bare ``except Exception`` -- the Magpie interpreter probe answers "this
    interpreter cannot import Magpie" that way -- so an ordinary exception here
    would be swallowed and the test would carry on against a wrong answer
    instead of stopping.
    """


@pytest.fixture
def launch_backend(monkeypatch):
    """Install a scripted stand-in for ``run_with_session_kill``.

    Call it with any object exposing that function's signature as ``run``. Both
    the definition and the names the eager importers bound are patched, so a
    launch made on a worker thread the test never sees is covered too.
    """
    from hyperloom.orchestrator.actions.executors import _grid_runner, _subprocess_kill, baseline

    installed: list = []

    def _run(cmd, **kwargs):
        if not installed:
            raise NoLaunchBackendInstalled(f"no launch backend is installed for this test; cmd={list(cmd)[:3]}")
        return installed[-1].run(cmd, **kwargs)

    for module in (_subprocess_kill, _grid_runner, baseline):
        monkeypatch.setattr(module, "run_with_session_kill", _run)

    def _install(backend):
        installed.append(backend)
        return backend

    return _install
