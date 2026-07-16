# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Ray-managed GPU execution backend (P0).

Covers the flag gate, the visible-device merge invariant, the run_subprocess
contract (via an inline fake ``ray``), and the ManagedServerProcess reap
invariant (with a real subprocess, no Ray cluster required).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _ray_backend as rb
from hyperloom.orchestrator.actions.executors._ray_serving import ManagedServerProcess


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
