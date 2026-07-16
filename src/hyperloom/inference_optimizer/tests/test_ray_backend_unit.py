# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Ray-managed GPU execution backend (P0).

Covers the flag gate, the visible-device merge invariant, the run_subprocess
contract (via an inline fake ``ray``), and the ManagedServerProcess reap
invariant (with a real subprocess, no Ray cluster required).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _ray_backend as rb
from hyperloom.orchestrator.actions.executors import _ray_serving as rs
from hyperloom.orchestrator.actions.executors._ray_serving import (
    ManagedServerProcess,
    ServingLease,
    maybe_serving_lease,
)


# ── flag gate ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
def test_ray_exec_enabled_true(monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", val)
    assert rb.ray_exec_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off"])
def test_ray_exec_enabled_explicit_off(monkeypatch: pytest.MonkeyPatch, val: str):
    """Explicit off wins even on single-node (emergency escape valve)."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", val)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    assert rb.ray_exec_enabled() is False


def test_ray_exec_forced_on_single_node_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Decision 2+4: unset env -> ON for single-node."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rb.ray_exec_enabled() is True


def test_ray_exec_off_on_multi_node_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Decision 4: unset env -> OFF for multi-node (out of scope this round)."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rb.ray_exec_enabled() is False


# ── visible-device merge invariant ───────────────────────────────────────────
def test_merge_worker_env_preserves_ray_visible_devices(monkeypatch: pytest.MonkeyPatch):
    """Ray owns *_VISIBLE_DEVICES; the caller must never override them."""
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    merged = rb._merge_worker_env(
        {
            "ROCR_VISIBLE_DEVICES": "0",  # must be ignored
            "HIP_VISIBLE_DEVICES": "0",  # must be ignored
            "CUDA_VISIBLE_DEVICES": "0",  # must be ignored
            "MY_FLAG": "1",  # must be applied
        }
    )
    assert merged["ROCR_VISIBLE_DEVICES"] == "2,3"
    assert merged["HIP_VISIBLE_DEVICES"] == "2,3"
    assert merged["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert merged["MY_FLAG"] == "1"


def test_merge_worker_env_none():
    merged = rb._merge_worker_env(None)
    assert isinstance(merged, dict)
    assert merged.get("PATH") == os.environ.get("PATH")


# ── run_subprocess contract (inline fake ray) ────────────────────────────────
class _FakeWorker:
    def __init__(self, fn, num_gpus, resources):
        self.fn = fn
        self.num_gpus = num_gpus
        self.resources = resources

    def remote(self, **kw):
        # Execute the worker body inline so the real run_with_session_kill runs.
        return {
            "result": self.fn(**kw),
            "num_gpus": self.num_gpus,
            "resources": self.resources,
        }


class _FakeRay:
    def __init__(self):
        self.last_ref = None

    def remote(self, **opts):
        def _deco(fn):
            return _FakeWorker(fn, opts.get("num_gpus"), opts.get("resources"))

        return _deco

    def get(self, ref):
        self.last_ref = ref
        return ref["result"]


def test_run_subprocess_passthrough_and_exec(monkeypatch: pytest.MonkeyPatch):
    """num_gpus/resources are forwarded to ray.remote and the subprocess runs."""
    fake = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    backend = rb.RayExecutionBackend()
    backend._ensured = True  # skip real cluster ensure

    result = asyncio.run(
        backend.run_subprocess(
            ["echo", "hello-ray"],
            num_gpus=2,
            resources={"serving_slot": 1},
            timeout_s=30,
        )
    )
    assert isinstance(result, rb.SubprocessResult)
    assert result.returncode == 0
    assert "hello-ray" in result.stdout
    # ray.remote received the requested lease.
    assert fake.last_ref["num_gpus"] == 2
    assert fake.last_ref["resources"] == {"serving_slot": 1}


def test_run_subprocess_nonzero_rc(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    backend = rb.RayExecutionBackend()
    backend._ensured = True
    result = asyncio.run(backend.run_subprocess(["false"], num_gpus=0))
    assert result.returncode != 0


# ── ManagedServerProcess reap invariant (real subprocess) ────────────────────
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_managed_process_start_and_reap():
    """A supervised process is reaped on stop() — no detached GPU-proc escape."""
    mgr = ManagedServerProcess()
    pid = mgr.start(["sleep", "30"])
    try:
        assert mgr.is_alive()
        assert mgr.pid() == pid
        assert _pid_alive(pid)
    finally:
        mgr.stop()
    # Give the OS a moment to finish reaping the group.
    deadline = time.time() + 5.0
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.05)
    assert not mgr.is_alive()
    assert not _pid_alive(pid), "supervised process must not survive stop()"


def test_managed_process_stop_idempotent():
    mgr = ManagedServerProcess()
    mgr.start(["sleep", "5"])
    mgr.stop()
    mgr.stop()  # must not raise
    assert not mgr.is_alive()


def test_managed_process_double_start_rejected():
    mgr = ManagedServerProcess()
    mgr.start(["sleep", "5"])
    try:
        with pytest.raises(RuntimeError):
            mgr.start(["sleep", "5"])
    finally:
        mgr.stop()


# ── shared artifact root ─────────────────────────────────────────────────────
def test_resolve_shared_artifact_root_single_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rb.resolve_shared_artifact_root(tmp_path) == tmp_path


def test_get_ray_backend_singleton():
    a = rb.get_ray_backend()
    b = rb.get_ray_backend()
    assert a is b


# ── _should_use_ray_backend (execution-route gate) ───────────────────────────
def test_should_use_ray_backend_pytest_default_off(monkeypatch: pytest.MonkeyPatch):
    """Under pytest with env unset, the route defaults OFF (hermetic tests)."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    # PYTEST_CURRENT_TEST is set by pytest during the test.
    assert rb._should_use_ray_backend() is False


def test_should_use_ray_backend_explicit_on(monkeypatch: pytest.MonkeyPatch):
    """Explicit RAY_EXEC=1 opts a test into the Ray route even under pytest."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    assert rb._should_use_ray_backend() is True


# ── _run_magpie routing (P1/T1) ──────────────────────────────────────────────
def test_num_gpus_for_config_reads_tp(tmp_path: Path):
    from hyperloom.orchestrator.actions.executors import _grid_runner as gr

    cfg = tmp_path / "c.yaml"
    cfg.write_text("benchmark:\n  envs:\n    TP: 4\n", encoding="utf-8")
    assert gr._num_gpus_for_config(cfg) == 4.0


def test_num_gpus_for_config_defaults_to_one(tmp_path: Path):
    from hyperloom.orchestrator.actions.executors import _grid_runner as gr

    cfg = tmp_path / "c.yaml"
    cfg.write_text("benchmark:\n  envs: {}\n", encoding="utf-8")
    assert gr._num_gpus_for_config(cfg) == 1.0


# ── T2: strip *_VISIBLE_DEVICES from the benchmark config ────────────────────
def test_strip_visible_devices_from_config(tmp_path: Path):
    """Ray sets visible devices in the worker; the YAML list must be dropped."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  envs:\n"
        "    TP: 2\n"
        "    ROCR_VISIBLE_DEVICES: '0,1'\n"
        "    HIP_VISIBLE_DEVICES: '0,1'\n"
        "    CUDA_VISIBLE_DEVICES: '0,1'\n"
        "    FOO: bar\n",
        encoding="utf-8",
    )
    out = rb.strip_visible_devices_from_config(cfg)
    assert out != cfg
    assert out.name.endswith(".ray.yaml")
    envs = yaml.safe_load(out.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert "ROCR_VISIBLE_DEVICES" not in envs
    assert "HIP_VISIBLE_DEVICES" not in envs
    assert "CUDA_VISIBLE_DEVICES" not in envs
    # Non-device envs are preserved verbatim.
    assert envs["TP"] == 2
    assert envs["FOO"] == "bar"


def test_strip_visible_devices_noop_when_absent(tmp_path: Path):
    """No device vars => the original path is returned unchanged (no rewrite)."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("benchmark:\n  envs:\n    TP: 1\n", encoding="utf-8")
    assert rb.strip_visible_devices_from_config(cfg) == cfg


# ── ServingLease + maybe_serving_lease (fake ray, no cluster) ─────────────────
class _FakeMethod:
    def __init__(self, ret):
        self._ret = ret

    def remote(self, *_a, **_k):
        return self._ret


class _FakeActor:
    def __init__(self, ret):
        self.run_blocking = _FakeMethod(ret)


class _LeaseFakeRay:
    """Minimal fake ``ray`` for ServingLease: get() unwraps refs, kill() records."""

    class exceptions:  # noqa: N801 — mirror ray.exceptions namespace
        class RayTaskError(Exception):
            pass

    def __init__(self):
        self.killed: list = []

    def get(self, ref):
        if isinstance(ref, _LeaseFakeRay.exceptions.RayTaskError):
            raise ref
        return ref

    def kill(self, actor):
        self.killed.append(actor)


def test_serving_lease_run_session_kill_success(monkeypatch: pytest.MonkeyPatch):
    fake = _LeaseFakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    lease = ServingLease(num_gpus=1)
    lease._actor = _FakeActor((0, "hi", ""))  # pre-set so ensure() is a no-op
    rc, out, err = lease.run_session_kill(["echo", "hi"], timeout=5)
    assert (rc, out, err) == (0, "hi", "")


def test_serving_lease_run_session_kill_timeout_reraises(monkeypatch: pytest.MonkeyPatch):
    """A hard-timeout sentinel from the actor is re-raised as TimeoutExpired."""
    fake = _LeaseFakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    lease = ServingLease(num_gpus=1)
    lease._actor = _FakeActor((rs._ACTOR_TIMEOUT_RC, "", "TimeoutExpired: 5s"))
    with pytest.raises(subprocess.TimeoutExpired):
        lease.run_session_kill(["sleep", "99"], timeout=5)


def test_serving_lease_run_session_kill_ray_error_degrades(monkeypatch: pytest.MonkeyPatch):
    """A worker-side Ray failure becomes a benchmark failure, not a crash."""
    fake = _LeaseFakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    lease = ServingLease(num_gpus=1)
    lease._actor = _FakeActor(fake.exceptions.RayTaskError("boom"))
    rc, out, err = lease.run_session_kill(["x"], timeout=5)
    assert rc == 1
    assert "ray_worker_error" in err


def test_serving_lease_close_idempotent(monkeypatch: pytest.MonkeyPatch):
    fake = _LeaseFakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    lease = ServingLease(num_gpus=1)
    actor = _FakeActor((0, "", ""))
    lease._actor = actor
    lease.close()
    assert actor in fake.killed
    assert lease._actor is None
    lease.close()  # must not raise


def test_maybe_serving_lease_pytest_default_none(monkeypatch: pytest.MonkeyPatch):
    """Under pytest with env unset the seam returns None (hermetic local path)."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    assert maybe_serving_lease(num_gpus=1) is None


def test_maybe_serving_lease_explicit_on_single_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    lease = maybe_serving_lease(num_gpus=2)
    assert isinstance(lease, ServingLease)
    assert lease._num_gpus == 2.0


def test_maybe_serving_lease_multi_node_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Multi-node is out of scope this round: no lease even with RAY_EXEC=1."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert maybe_serving_lease(num_gpus=2) is None


# ── _run_magpie routing (P1/T1 + T2) ─────────────────────────────────────────
class _RecordingLease:
    """Stand-in ServingLease that records the round it was asked to run."""

    def __init__(self, result=(0, "ok", "")):
        self.result = result
        self.calls: list[dict] = []

    def run_session_kill(self, cmd, **kw):
        self.calls.append({"cmd": cmd, **kw})
        return self.result


def test_run_magpie_routes_through_lease_and_strips_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With a lease, _run_magpie runs in its actor on a device-stripped config."""
    from hyperloom.orchestrator.actions.executors import _grid_runner as gr

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  envs:\n"
        "    TP: 1\n"
        "    ROCR_VISIBLE_DEVICES: '0'\n"
        "    FOO: bar\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    seen: dict = {}

    def _fake_build(*, python_exe, config_path, output_dir):
        seen["config_path"] = Path(config_path)
        return ["magpie", "-m", "Magpie", "benchmark", str(config_path)]

    monkeypatch.setattr(gr, "build_benchmark_command", _fake_build)

    lease = _RecordingLease()
    rc, out, err = gr._run_magpie(
        magpie_python="python3",
        config_path=cfg,
        output_dir=out_dir,
        timeout_sec=10,
        cwd=str(tmp_path),
        serving_lease=lease,
    )
    assert (rc, out, err) == (0, "ok", "")
    assert lease.calls, "the round must run inside the lease's actor"
    # T2: the config handed to Magpie has the device list stripped.
    used_cfg = seen["config_path"]
    assert used_cfg.name.endswith(".ray.yaml")
    envs = yaml.safe_load(used_cfg.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert "ROCR_VISIBLE_DEVICES" not in envs
    assert envs["TP"] == 1 and envs["FOO"] == "bar"
    # server.log is pinned into the task slot for the watchdogs.
    assert lease.calls[0]["server_log_path"] == str(out_dir / "server.log")
    assert lease.calls[0]["timeout"] == 10


# ── P2: ManagedServerProcess.exit_code (real subprocess) ─────────────────────
def test_managed_process_exit_code_none_then_latched():
    """exit_code is None before start / while alive, then the real return code."""
    mgr = ManagedServerProcess()
    assert mgr.exit_code() is None  # never started
    mgr.start(["sh", "-c", "sleep 0.2; exit 3"])
    assert mgr.exit_code() is None  # still running
    deadline = time.time() + 5.0
    while time.time() < deadline and mgr.is_alive():
        time.sleep(0.05)
    assert mgr.exit_code() == 3
    mgr.stop()


# ── P2: GpuSpecialistLease (fake ray + fake actor) ───────────────────────────
class _FakeActorMethodP2:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        # Defer the call to fake ray.get, mirroring Ray's ObjectRef.
        return ("call", self._fn, a, k)


class _FakeGpuActor:
    def __init__(self):
        self._alive = True
        self._exit: int | None = None
        self.stopped = False
        self.started_with: dict | None = None
        self.start = _FakeActorMethodP2(self._start)
        self.is_alive = _FakeActorMethodP2(lambda: self._alive)
        self.exit_code = _FakeActorMethodP2(lambda: self._exit)
        self.stop = _FakeActorMethodP2(self._stop)

    def _start(self, cmd, env=None, cwd=None, log_path=None):
        self.started_with = {"cmd": cmd, "env": env, "cwd": cwd, "log_path": log_path}
        return 4242

    def _stop(self):
        self.stopped = True
        self._alive = False
        self._exit = -15
        return None


class _FakeRayP2:
    class exceptions:  # noqa: N801 — mirror ray.exceptions namespace
        class RayTaskError(Exception):
            pass

    def __init__(self):
        self.killed: list = []

    def get(self, ref):
        _tag, fn, a, k = ref
        return fn(*a, **k)

    def kill(self, actor):
        self.killed.append(actor)


class _StubBackendP2:
    def ensure(self, *a, **k):
        return None


def test_gpu_specialist_lease_lifecycle(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeRayP2()
    monkeypatch.setitem(sys.modules, "ray", fake)
    actor = _FakeGpuActor()
    monkeypatch.setattr(rs, "make_gpu_specialist_actor", lambda n, *, serving_slot=False: actor)
    monkeypatch.setattr(rb, "get_ray_backend", lambda: _StubBackendP2())

    lease = rs.GpuSpecialistLease(num_gpus=2)
    pid = lease.start(["claude"], env={"A": "1"}, cwd="/tmp", log_path="/tmp/p.log")
    assert pid == 4242
    assert lease.pid() == 4242
    assert actor.started_with["log_path"] == "/tmp/p.log"
    assert lease.is_alive() is True
    assert lease.exit_code() is None
    lease.stop()
    assert actor.stopped is True
    assert lease.is_alive() is False
    lease.close()
    assert actor in fake.killed
    lease.close()  # idempotent


def test_gpu_specialist_lease_is_alive_false_before_start():
    lease = rs.GpuSpecialistLease(num_gpus=1)
    assert lease.is_alive() is False
    assert lease.exit_code() is None
    lease.close()  # no-op, no actor


def test_maybe_gpu_specialist_lease_pytest_default_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    assert rs.maybe_gpu_specialist_lease(num_gpus=2) is None


def test_maybe_gpu_specialist_lease_zero_gpus_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    assert rs.maybe_gpu_specialist_lease(num_gpus=0) is None


def test_maybe_gpu_specialist_lease_single_node_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    lease = rs.maybe_gpu_specialist_lease(num_gpus=2)
    assert isinstance(lease, rs.GpuSpecialistLease)


def test_maybe_gpu_specialist_lease_multi_node_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rs.maybe_gpu_specialist_lease(num_gpus=2) is None


# ── P3: serving_slot custom resource (T6) ────────────────────────────────────
def test_serving_slot_declared_in_ray_start_args():
    """ensure_ray_cluster's head declares the serving_slot custom resource."""
    from hyperloom.agents.kernel.tools.backends import ray_runtime as rr

    args = rr._resources_start_args()
    assert args[0] == "--resources"
    import json as _json

    assert _json.loads(args[1]) == {"serving_slot": 1}


def test_maybe_serving_lease_holds_serving_slot_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Serving-family leases hold the whole-machine serving_slot (§12 T6)."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    lease = rs.maybe_serving_lease(num_gpus=2)
    assert isinstance(lease, ServingLease)
    assert lease._serving_slot is True


def test_maybe_gpu_specialist_lease_serving_slot_passthrough(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Serving-disjoint pool takes no slot; whole-machine specialists take it."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    disjoint = rs.maybe_gpu_specialist_lease(num_gpus=2)
    assert disjoint is not None and disjoint._serving_slot is False
    whole = rs.maybe_gpu_specialist_lease(num_gpus=8, serving_slot=True)
    assert whole is not None and whole._serving_slot is True


def test_gpu_specialist_lease_start_passes_serving_slot(monkeypatch: pytest.MonkeyPatch):
    """The lease forwards serving_slot to the actor factory on start."""
    fake = _FakeRayP2()
    monkeypatch.setitem(sys.modules, "ray", fake)
    actor = _FakeGpuActor()
    seen: dict = {}

    def _fake_make(n, *, serving_slot=False):
        seen["num_gpus"] = n
        seen["serving_slot"] = serving_slot
        return actor

    monkeypatch.setattr(rs, "make_gpu_specialist_actor", _fake_make)
    monkeypatch.setattr(rb, "get_ray_backend", lambda: _StubBackendP2())

    lease = rs.GpuSpecialistLease(num_gpus=8, serving_slot=True)
    assert lease.start(["claude"]) == 4242
    assert seen == {"num_gpus": 8.0, "serving_slot": True}


# ── P4 (skeleton): ServingGroupManager — placement group + rank actors ───────
def test_serving_group_manager_lifecycle(monkeypatch: pytest.MonkeyPatch):
    """start reserves a PG + one rank actor per node; stop/close reap them."""
    fake = _FakeRayP2()
    monkeypatch.setitem(sys.modules, "ray", fake)
    monkeypatch.setattr(rb, "get_ray_backend", lambda: _StubBackendP2())

    fake_pg = object()
    pg_calls: dict = {}

    def _fake_make_pg(nodes, gpus, *, serving_slot):
        pg_calls.update(nodes=nodes, gpus=gpus, serving_slot=serving_slot)
        return fake_pg

    monkeypatch.setattr(rs, "_make_serving_placement_group", _fake_make_pg)

    made: list = []

    def _fake_make_rank(pg, idx, num_gpus, *, serving_slot):
        assert pg is fake_pg
        actor = _FakeGpuActor()
        made.append((idx, num_gpus, serving_slot, actor))
        return actor

    monkeypatch.setattr(rs, "_make_rank_actor", _fake_make_rank)
    removed: dict = {"pg": None}
    monkeypatch.setattr(
        rs, "_remove_serving_placement_group", lambda pg: removed.__setitem__("pg", pg)
    )

    sgm = rs.ServingGroupManager(nodes=2, gpus_per_node=8, serving_slot=True)
    pids = sgm.start([["srv", "rank0"], ["srv", "rank1"]])
    assert pids == [4242, 4242]
    assert pg_calls == {"nodes": 2, "gpus": 8.0, "serving_slot": True}
    assert [m[0] for m in made] == [0, 1]  # one rank pinned per bundle index
    assert all(m[1] == 8.0 for m in made)  # num_gpus per rank
    assert sgm.ranks_alive() == [True, True]
    assert sgm.is_alive() is True

    sgm.stop()
    assert all(m[3].stopped for m in made)
    assert sgm.is_alive() is False

    sgm.close()
    assert len(fake.killed) == 2  # both rank actors killed
    assert removed["pg"] is fake_pg
    sgm.close()  # idempotent


def test_serving_group_manager_start_arity_mismatch():
    """A rank_cmds count that doesn't match nodes fails fast (before any Ray)."""
    sgm = rs.ServingGroupManager(nodes=2, gpus_per_node=8)
    with pytest.raises(ValueError):
        sgm.start([["only-one-rank"]])


def test_maybe_serving_group_manager_default_none(monkeypatch: pytest.MonkeyPatch):
    """P4 is deferred: off by default even multi-node (needs the explicit flag)."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_MN_SERVING", raising=False)
    assert rs.maybe_serving_group_manager(nodes=2, gpus_per_node=8) is None


def test_maybe_serving_group_manager_flag_on_single_node_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_MN_SERVING", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rs.maybe_serving_group_manager(nodes=2, gpus_per_node=8) is None


def test_maybe_serving_group_manager_flag_on_multi_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_MN_SERVING", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    sgm = rs.maybe_serving_group_manager(nodes=2, gpus_per_node=8)
    assert isinstance(sgm, rs.ServingGroupManager)


def test_maybe_serving_group_manager_zero_nodes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_MN_SERVING", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rs.maybe_serving_group_manager(nodes=0, gpus_per_node=8) is None


def test_run_magpie_local_path_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """serving_lease=None keeps the local run_with_session_kill path + config."""
    from hyperloom.orchestrator.actions.executors import _grid_runner as gr

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "benchmark:\n  framework: sglang\n  envs:\n    TP: 1\n"
        "    ROCR_VISIBLE_DEVICES: '0'\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    seen: dict = {}

    def _fake_build(*, python_exe, config_path, output_dir):
        seen["config_path"] = Path(config_path)
        return ["magpie", str(config_path)]

    class _Proc:
        returncode = 0
        stdout = "local"
        stderr = ""

    def _fake_run(cmd, **kw):
        seen["ran_local"] = True
        return _Proc()

    monkeypatch.setattr(gr, "build_benchmark_command", _fake_build)
    monkeypatch.setattr(gr, "run_with_session_kill", _fake_run)

    rc, out, err = gr._run_magpie(
        magpie_python="python3",
        config_path=cfg,
        output_dir=out_dir,
        timeout_sec=10,
        cwd=str(tmp_path),
        serving_lease=None,
    )
    assert (rc, out) == (0, "local")
    assert seen.get("ran_local") is True
    # The local path uses the ORIGINAL config (no .ray.yaml rewrite).
    assert seen["config_path"] == cfg
    assert not (tmp_path / "config.ray.yaml").exists()


# ── coverage: backend sync submit + shared-root MN + strip fallbacks ─────────
def test_run_subprocess_sync_passthrough(monkeypatch: pytest.MonkeyPatch):
    """run_subprocess_sync submits via ray and returns the SubprocessResult."""
    fake = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    backend = rb.RayExecutionBackend()
    backend._ensured = True
    result = backend.run_subprocess_sync(
        ["echo", "sync-ray"], num_gpus=1, resources={"serving_slot": 1}, timeout_s=30
    )
    assert isinstance(result, rb.SubprocessResult)
    assert result.returncode == 0
    assert "sync-ray" in result.stdout
    assert fake.last_ref["num_gpus"] == 1


def test_resolve_shared_artifact_root_multi_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Multi-node + HYPERLOOM_MN_PROFILE_TRACE_DIR -> the shared root wins."""
    mn_root = tmp_path / "shared"
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", str(mn_root))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rb.resolve_shared_artifact_root(tmp_path / "sess") == mn_root


def test_strip_visible_devices_unparseable_yaml_returns_src(tmp_path: Path):
    """A YAML parse error returns the original path unchanged (fail-soft)."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("benchmark: [unbalanced\n", encoding="utf-8")
    assert rb.strip_visible_devices_from_config(cfg) == cfg


def test_strip_visible_devices_no_envs_dict_returns_src(tmp_path: Path):
    """When benchmark.envs is not a dict, return the original path."""
    cfg = tmp_path / "noenvs.yaml"
    cfg.write_text("benchmark:\n  envs: not_a_dict\n", encoding="utf-8")
    assert rb.strip_visible_devices_from_config(cfg) == cfg


def test_should_use_ray_backend_explicit_off(monkeypatch: pytest.MonkeyPatch):
    """Explicit RAY_EXEC=0 forces the local path even outside pytest gating."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "0")
    assert rb._should_use_ray_backend() is False


# ── coverage: GpuSpecialistLease exception branches (dead actor) ─────────────
class _RaisingActor:
    """Fake actor whose method .remote() refs make fake ray.get/kill raise."""

    class _M:
        def remote(self, *a, **k):
            return "boom-ref"

    def __init__(self):
        self.is_alive = self._M()
        self.exit_code = self._M()
        self.stop = self._M()


class _RaisingRay:
    class exceptions:  # noqa: N801
        class RayTaskError(Exception):
            pass

    def get(self, ref):
        raise RuntimeError("actor dead")

    def kill(self, actor):
        raise RuntimeError("kill failed")


def test_gpu_specialist_lease_dead_actor_degrades(monkeypatch: pytest.MonkeyPatch):
    """is_alive/exit_code/stop/close swallow a dead-actor error (never raise)."""
    monkeypatch.setitem(sys.modules, "ray", _RaisingRay())
    lease = rs.GpuSpecialistLease(num_gpus=1)
    lease._actor = _RaisingActor()  # force the ray.get/kill paths
    assert lease.is_alive() is False       # 598-600
    assert lease.exit_code() is None       # 613-615
    lease.stop()                           # 628-630 (no raise)
    lease.close()                          # 639-641 (kill raises, swallowed)
    assert lease._actor is None


# ── coverage: ServingGroupManager empty + exception branches ─────────────────
def test_serving_group_manager_empty_before_start():
    """A never-started SGM: ranks_alive=[] / is_alive False / stop no-op."""
    sgm = rs.ServingGroupManager(nodes=2, gpus_per_node=8)
    assert sgm.pids() == []
    assert sgm.ranks_alive() == []   # 892-893
    assert sgm.is_alive() is False
    sgm.stop()                       # 915-916 (no ranks -> return)
    sgm.close()                      # no pg / no ranks -> clean


def test_serving_group_manager_rank_errors_degrade(monkeypatch: pytest.MonkeyPatch):
    """A rank actor that raises reads as not-alive; stop/close swallow errors."""
    monkeypatch.setitem(sys.modules, "ray", _RaisingRay())
    sgm = rs.ServingGroupManager(nodes=2, gpus_per_node=8)
    sgm._ranks = [_RaisingActor(), _RaisingActor()]
    sgm._pids = [1, 2]
    sgm._pg = object()
    removed: dict = {"hit": False}
    monkeypatch.setattr(
        rs,
        "_remove_serving_placement_group",
        lambda pg: removed.__setitem__("hit", (_ for _ in ()).throw(RuntimeError("pg gone"))),
    )
    assert sgm.ranks_alive() == [False, False]  # 899-901
    assert sgm.is_alive() is False
    sgm.stop()                                  # 921-923 swallowed
    sgm.close()                                 # 931-933 + 939-940 swallowed
    assert sgm._ranks == [] and sgm._pg is None


# ── coverage: ManagedServerProcess pid/exit_code before start ────────────────
def test_managed_process_pid_exit_code_before_start():
    mgr = ManagedServerProcess()
    assert mgr.pid() is None
    assert mgr.exit_code() is None
    assert mgr.is_alive() is False


# ── coverage: ServingActor class body via a pass-through fake ray.remote ─────
class _PassthroughRay:
    """Fake ray whose @remote is an identity decorator, so the ServingActor
    class body runs as plain Python (no cluster) for coverage of start /
    run_blocking / is_alive / pid / exit_code / stop."""

    def remote(self, *dargs, **dkw):
        # Support both @ray.remote and @ray.remote(...) forms.
        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]

        def _deco(cls):
            return cls

        return _deco


def test_serving_actor_body_methods_drive_real_subprocess(monkeypatch: pytest.MonkeyPatch):
    """Instantiate the ServingActor class directly and drive its lifecycle on a
    real short-lived subprocess (covers _serving_actor_body's method bodies)."""
    monkeypatch.setitem(sys.modules, "ray", _PassthroughRay())
    actor_cls = rs._serving_actor_body()
    actor = actor_cls()  # plain instance (identity-decorated)

    # start(): device vars in caller env are dropped; non-device vars flow.
    pid = actor.start(["sh", "-c", "sleep 0.3; exit 0"], env={"ROCR_VISIBLE_DEVICES": "9", "FOO": "bar"})
    assert isinstance(pid, int) and pid > 0
    assert actor.pid() == pid
    assert actor.is_alive() is True
    assert actor.exit_code() is None
    deadline = time.time() + 5.0
    while time.time() < deadline and actor.is_alive():
        time.sleep(0.05)
    assert actor.exit_code() == 0
    actor.stop()  # idempotent reap

    # run_blocking(): fresh actor runs one round to completion, returns triple.
    actor2 = actor_cls()
    rc, out, err = actor2.run_blocking(["echo", "hello-actor"], timeout=30)
    assert rc == 0 and "hello-actor" in out
    actor2.stop()


def test_serving_actor_run_blocking_timeout_sentinel(monkeypatch: pytest.MonkeyPatch):
    """A hard timeout in run_blocking is reported as the sentinel returncode."""
    monkeypatch.setitem(sys.modules, "ray", _PassthroughRay())
    actor = rs._serving_actor_body()()
    rc, _out, err = actor.run_blocking(["sleep", "30"], timeout=1)
    assert rc == rs._ACTOR_TIMEOUT_RC
    assert "TimeoutExpired" in err
    actor.stop()


# ── coverage: RayExecutionBackend.ensure (reuses kernel ray_runtime) ─────────
def test_backend_ensure_reuses_kernel_runtime(monkeypatch: pytest.MonkeyPatch):
    """ensure() calls ensure_ray_cluster + quiet_ray_init once, then is idempotent."""
    from hyperloom.agents.kernel.tools.backends import ray_runtime as rr

    calls: dict = {"ensure": 0, "init": 0, "num_gpus": None}

    def _fake_ensure(*, num_gpus=None, log_path=None):
        calls["ensure"] += 1
        calls["num_gpus"] = num_gpus
        return True

    def _fake_init(*, num_gpus=None, log_path=None):
        calls["init"] += 1

    monkeypatch.setattr(rr, "ensure_ray_cluster", _fake_ensure)
    monkeypatch.setattr(rr, "quiet_ray_init", _fake_init)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_NUM_GPUS", "3")

    backend = rb.RayExecutionBackend()
    backend.ensure()
    assert calls == {"ensure": 1, "init": 1, "num_gpus": 3}
    backend.ensure()  # idempotent: no second call
    assert calls["ensure"] == 1


# ── coverage: PG + rank-actor factory helpers (fake ray.util modules) ────────
def test_make_serving_placement_group_and_rank_actor(monkeypatch: pytest.MonkeyPatch):
    """Cover the placement-group + rank-actor factories via fake ray.util modules."""
    import types

    seen: dict = {}

    class _FakePG:
        def ready(self):
            return "ready-ref"

    def _fake_placement_group(bundles, strategy):
        seen["bundles"] = bundles
        seen["strategy"] = strategy
        return _FakePG()

    class _FakePassRay:
        def remote(self, *da, **dk):
            if len(da) == 1 and callable(da[0]) and not dk:
                return da[0]
            return lambda cls: cls

        def get(self, ref):
            return ref

    # Fake ray + ray.util.placement_group + ray.util.scheduling_strategies.
    monkeypatch.setitem(sys.modules, "ray", _FakePassRay())
    pg_mod = types.ModuleType("ray.util.placement_group")
    pg_mod.placement_group = _fake_placement_group
    pg_mod.remove_placement_group = lambda pg: seen.__setitem__("removed", pg)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", pg_mod)

    class _FakeSchedStrat:
        def __init__(self, *, placement_group, placement_group_bundle_index):
            seen["bundle_index"] = placement_group_bundle_index

    ss_mod = types.ModuleType("ray.util.scheduling_strategies")
    ss_mod.PlacementGroupSchedulingStrategy = _FakeSchedStrat
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", ss_mod)

    pg = rs._make_serving_placement_group(2, 8, serving_slot=True)
    assert isinstance(pg, _FakePG)
    assert seen["strategy"] == "STRICT_SPREAD"
    assert seen["bundles"][0] == {"GPU": 8.0, "serving_slot": 1}
    assert len(seen["bundles"]) == 2

    # rank actor: options()(...).remote() — the ServingActor class is identity
    # under _FakePassRay.remote, so .options must exist. Wrap it.
    actor_cls = rs._serving_actor_body()

    class _Opts:
        def options(self, **kw):
            seen["opts"] = kw
            return self

        def remote(self):
            return "rank-actor"

    monkeypatch.setattr(rs, "_serving_actor_body", lambda: _Opts())
    handle = rs._make_rank_actor(pg, 1, 8, serving_slot=True)
    assert handle == "rank-actor"
    assert seen["bundle_index"] == 1
    assert seen["opts"]["resources"] == {"serving_slot": 1}

    rs._remove_serving_placement_group(pg)
    assert seen["removed"] is pg
    # Defensive: drop the injected fake ray.util.* submodules so a later test's
    # lazy ``import ray`` never sees this test's fakes (belt-and-suspenders on
    # top of monkeypatch's own setitem teardown).
    for mod in ("ray.util.scheduling_strategies", "ray.util.placement_group"):
        sys.modules.pop(mod, None)


# ── coverage: ServingLease ensure/context-manager/close + make_serving_actor ─
def test_serving_lease_context_manager_and_ensure(monkeypatch: pytest.MonkeyPatch):
    """ensure() creates the actor once; __enter__/__exit__ ensure+close it."""
    fake = _LeaseFakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    made: list = []
    monkeypatch.setattr(
        rs, "make_serving_actor",
        lambda n, *, serving_slot=True: made.append((n, serving_slot)) or _FakeActor((0, "", "")),
    )
    monkeypatch.setattr(rb, "get_ray_backend", lambda: _StubBackendP2())

    lease = rs.ServingLease(num_gpus=2, serving_slot=True)
    lease.ensure()
    assert made == [(2.0, True)]
    lease.ensure()  # idempotent — actor already set, no second make
    assert len(made) == 1
    actor = lease._actor
    lease.close()  # ray.kill path (actor present)
    assert actor in fake.killed
    assert lease._actor is None

    # Context manager: __enter__ ensures, __exit__ closes.
    with rs.ServingLease(num_gpus=1) as l2:
        assert l2._actor is not None
    assert l2._actor is None  # __exit__ closed it


def test_make_serving_actor_slot_modes(monkeypatch: pytest.MonkeyPatch):
    """make_serving_actor toggles the serving_slot resource on/off."""
    captured: dict = {}

    class _Opts:
        def options(self, **kw):
            captured.update(kw)
            return self

        def remote(self):
            return "actor"

    monkeypatch.setattr(rs, "_serving_actor_body", lambda: _Opts())
    rs.make_serving_actor(4, serving_slot=True)
    assert captured["num_gpus"] == 4 and captured["resources"] == {"serving_slot": 1}
    rs.make_serving_actor(2, serving_slot=False)
    assert captured["resources"] is None
    rs.make_gpu_specialist_actor(1)  # default serving_slot=False
    assert captured["resources"] is None


def test_gpu_specialist_lease_stop_no_actor_noop():
    """stop()/close() on a never-started GpuSpecialistLease are safe no-ops."""
    lease = rs.GpuSpecialistLease(num_gpus=1)
    lease.stop()   # 623-624: no actor -> return
    lease.close()  # no actor -> return
    assert lease.pid() is None


def test_gpu_specialist_lease_close_kills_live_actor(monkeypatch: pytest.MonkeyPatch):
    """close() ray.kill()s a live actor and clears the handle (normal path)."""
    fake = _LeaseFakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    lease = rs.GpuSpecialistLease(num_gpus=1)
    actor = _FakeGpuActor()
    lease._actor = actor
    lease.close()                      # 432-435: ray.kill path
    assert actor in fake.killed
    assert lease._actor is None


def test_should_use_ray_backend_unset_single_node_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Env unset + not-under-pytest + single-node -> True (production default)."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_RAY_EXEC", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)  # bypass the pytest gate
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(tmp_path / "nope.json"))
    assert rb._should_use_ray_backend() is True  # 80-82


def test_strip_visible_devices_write_error_returns_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A write failure on the .ray.yaml sibling returns the original path."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "benchmark:\n  envs:\n    TP: 1\n    ROCR_VISIBLE_DEVICES: '0'\n",
        encoding="utf-8",
    )
    real_open = Path.open

    def _boom_open(self, *a, **k):
        if self.name.endswith(".ray.yaml") and (a[:1] == ("w",) or k.get("mode") == "w"):
            raise OSError("disk full")
        return real_open(self, *a, **k)

    monkeypatch.setattr(Path, "open", _boom_open)
    assert rb.strip_visible_devices_from_config(cfg) == cfg  # 395-396


def test_serving_lease_close_swallows_kill_error(monkeypatch: pytest.MonkeyPatch):
    """ServingLease.close swallows a ray.kill error and still clears the handle."""
    monkeypatch.setitem(sys.modules, "ray", _RaisingRay())  # kill() raises
    lease = rs.ServingLease(num_gpus=1)
    lease._actor = object()
    lease.close()  # 432-435: except path swallowed
    assert lease._actor is None


def test_managed_process_start_with_log_path(tmp_path: Path):
    """start(log_path=...) opens the log file (covers the log_path branch)."""
    log = tmp_path / "nested" / "server.log"
    mgr = ManagedServerProcess()
    pid = mgr.start(["sh", "-c", "echo hi; sleep 0.2"], log_path=str(log))  # 98-100
    try:
        assert pid > 0
        assert log.parent.is_dir()      # os.makedirs ran
    finally:
        mgr.stop()
    assert log.exists()
