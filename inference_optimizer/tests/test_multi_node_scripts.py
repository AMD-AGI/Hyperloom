"""Unit tests for ``multi_node/scripts`` launch and kill helpers.

The scripts depend on ``ray`` at import time (in-cluster runtime). Tests
install a tiny ``sys.modules`` stub so CI can import the modules without
installing Ray, then exercise pure helpers and a few side-effect paths
with mocks / ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
import types
from pathlib import Path


def _install_min_ray_stub() -> None:
    """Minimal ``ray`` package graph so script modules can import."""
    if "ray" in sys.modules:
        existing = sys.modules["ray"]
        if getattr(existing, "util", None) is not None:
            return
        if "ray.util" not in sys.modules:
            ray_util_fix = types.ModuleType("ray.util")
            ray_util_fix.get_node_ip_address = lambda: "127.0.0.1"
            sys.modules["ray.util"] = ray_util_fix
        existing.util = sys.modules["ray.util"]
        return

    def _remote_decorator(**_kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    ray_mod = types.ModuleType("ray")
    ray_mod.init = lambda **kwargs: None
    ray_mod.nodes = lambda: []
    ray_mod.remote = _remote_decorator
    ray_mod.get = lambda ref, timeout=None: ref

    ray_util = types.ModuleType("ray.util")
    ray_util.get_node_ip_address = lambda: "127.0.0.1"

    sched = types.ModuleType("ray.util.scheduling_strategies")

    class NodeAffinitySchedulingStrategy:
        def __init__(self, node_id: str, soft: bool = False) -> None:
            self.node_id = node_id
            self.soft = soft

    sched.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy

    ray_mod.util = ray_util
    sys.modules["ray"] = ray_mod
    sys.modules["ray.util"] = ray_util
    sys.modules["ray.util.scheduling_strategies"] = sched


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_script_module(unique_name: str, script_name: str):
    _install_min_ray_stub()
    path = _repo_root() / "multi_node" / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_kill_remote_missing_pid_dir():
    km = _load_script_module("km_test_missing", "kill_multinode.py")
    out = km._kill_remote("/no/such/dir/exists", grace_sec=1)
    assert out == {"killed": [], "stale": [], "missing": []}


def test_kill_remote_non_digit_pid_file_removed(tmp_path):
    km = _load_script_module("km_test_nondigit", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_0.pid").write_text("not-a-number", encoding="utf-8")
    out = km._kill_remote(str(d), grace_sec=0)
    assert "rank_0.pid" in out["stale"]
    assert not (d / "rank_0.pid").exists()


def test_kill_remote_sentinel_zero_pid_removed(tmp_path):
    km = _load_script_module("km_test_zero", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_1.pid").write_text("0", encoding="utf-8")
    out = km._kill_remote(str(d), grace_sec=0)
    assert any("rank_1.pid" in s for s in out["stale"])
    assert not (d / "rank_1.pid").exists()


def test_kill_remote_dead_pid_stale(tmp_path, monkeypatch):
    km = _load_script_module("km_test_dead", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_2.pid").write_text("99999", encoding="utf-8")

    def _kill(pid, sig):
        if sig == 0:
            raise ProcessLookupError()

    monkeypatch.setattr("os.kill", _kill)
    out = km._kill_remote(str(d), grace_sec=0)
    assert any("rank_2.pid" in s for s in out["stale"])
    assert not (d / "rank_2.pid").exists()


def test_kill_remote_sigterms_then_process_exits(tmp_path, monkeypatch):
    km = _load_script_module("km_test_term", "kill_multinode.py")
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_0.pid").write_text("4242", encoding="utf-8")

    after_term = {"done": False}

    def _kill(pid, sig):
        if sig == 0:
            if after_term["done"]:
                raise ProcessLookupError()
            return None
        assert pid == 4242

    def _getpgid(pid):
        assert pid == 4242
        return 424200

    def _killpg(pgid, sig):
        assert pgid == 424200
        assert sig in (signal.SIGTERM, signal.SIGKILL)
        if sig == signal.SIGTERM:
            after_term["done"] = True

    monkeypatch.setattr("os.kill", _kill)
    monkeypatch.setattr("os.getpgid", _getpgid)
    monkeypatch.setattr("os.killpg", _killpg)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    out = km._kill_remote(str(d), grace_sec=0)
    assert any(x.startswith("rank_0.pid:") for x in out["killed"])
    assert not (d / "rank_0.pid").exists()


def test_build_sglang_cmd_head_has_host_port():
    lm = _load_script_module("lm_test_sglang_head", "launch_multinode.py")
    cmd = lm._build_sglang_cmd(
        model="/m",
        tp=2,
        nnodes=2,
        node_rank=0,
        dist_init_addr="10.0.0.1:5000",
        extra_args=["--x"],
    )
    assert "--host" in cmd and "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "8888"
    assert "--x" in cmd


def test_build_sglang_cmd_worker_omits_http_bind():
    lm = _load_script_module("lm_test_sglang_worker", "launch_multinode.py")
    cmd = lm._build_sglang_cmd(
        model="/m",
        tp=2,
        nnodes=2,
        node_rank=1,
        dist_init_addr="10.0.0.1:5000",
        extra_args=[],
    )
    assert "--host" not in cmd
    assert "--port" not in cmd


def test_build_vllm_cmd_includes_ray_backend():
    lm = _load_script_module("lm_test_vllm", "launch_multinode.py")
    cmd = lm._build_vllm_cmd(model="/m", tp=16, extra_args=["--enforce-eager"])
    assert cmd[0] == "vllm"
    assert "--distributed-executor-backend" in cmd
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "16"
    assert "--enforce-eager" in cmd


def test_subprocess_env_prepends_venv_bin(monkeypatch):
    lm = _load_script_module("lm_test_env", "launch_multinode.py")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = lm._subprocess_env()
    assert env["PATH"].startswith("/opt/venv/bin:")


def test_subprocess_env_idempotent_if_already_first(monkeypatch):
    lm = _load_script_module("lm_test_env2", "launch_multinode.py")
    monkeypatch.setenv("PATH", "/opt/venv/bin:/usr/bin")
    env = lm._subprocess_env()
    assert env["PATH"].startswith("/opt/venv/bin")
    assert env["PATH"].count("/opt/venv/bin") == 1


def test_detach_framework_launch_starts_sleep(tmp_path):
    lm = _load_script_module("lm_test_detach_sleep", "launch_multinode.py")
    log_f = tmp_path / "r0.log"
    pid_f = tmp_path / "r0.pid"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    pid = lm._detach_framework_launch(
        ["sleep", "60"],
        log_f,
        pid_f,
        env,
        node_rank=0,
    )
    assert int(pid_f.read_text(encoding="utf-8").strip()) == pid
    os.kill(pid, 0)
    os.kill(pid, signal.SIGTERM)


def test_detach_framework_launch_raises_on_immediate_exit(tmp_path):
    lm = _load_script_module("lm_test_detach_false", "launch_multinode.py")
    log_f = tmp_path / "r0.log"
    pid_f = tmp_path / "r0.pid"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        lm._detach_framework_launch(
            ["sh", "-c", "exit 42"],
            log_f,
            pid_f,
            env,
            node_rank=0,
        )
    except RuntimeError as exc:
        assert "immediately" in str(exc).lower() or "not alive" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_pick_head_first_orders_local_first(monkeypatch):
    lm = _load_script_module("lm_test_pick", "launch_multinode.py")
    monkeypatch.setattr(
        lm.ray.util, "get_node_ip_address", lambda: "10.0.0.1",
    )
    nodes = [
        {"NodeManagerAddress": "10.0.0.2", "NodeID": "b"},
        {"NodeManagerAddress": "10.0.0.1", "NodeID": "a"},
    ]
    ordered = lm._pick_head_first(nodes)
    assert ordered[0]["NodeManagerAddress"] == "10.0.0.1"
    assert ordered[1]["NodeManagerAddress"] == "10.0.0.2"


def test_pick_head_first_fallback_sorts_when_local_missing(monkeypatch):
    lm = _load_script_module("lm_test_pick2", "launch_multinode.py")
    monkeypatch.setattr(
        lm.ray.util, "get_node_ip_address", lambda: "192.168.99.99",
    )
    nodes = [
        {"NodeManagerAddress": "10.0.0.2", "NodeID": "b"},
        {"NodeManagerAddress": "10.0.0.1", "NodeID": "a"},
    ]
    ordered = lm._pick_head_first(nodes)
    addrs = [n["NodeManagerAddress"] for n in ordered]
    assert addrs == sorted(addrs)


def test_wait_for_nodes_returns_when_enough_gpu_nodes(monkeypatch):
    lm = _load_script_module("lm_test_wait_ok", "launch_multinode.py")
    sample = [
        {"Alive": True, "Resources": {"GPU": 1.0}},
        {"Alive": True, "Resources": {"GPU": 1.0}},
    ]
    monkeypatch.setattr(lm.ray, "nodes", lambda: sample)
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    got = lm._wait_for_nodes(2, timeout_s=10)
    assert len(got) == 2


def test_wait_for_nodes_times_out(monkeypatch):
    lm = _load_script_module("lm_test_wait_to", "launch_multinode.py")
    one = [{"Alive": True, "Resources": {"GPU": 1.0}}]
    monkeypatch.setattr(lm.ray, "nodes", lambda: one)
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    times = iter([0.0, 0.0, 200.0])

    def _mono():
        return next(times)

    monkeypatch.setattr(lm.time, "monotonic", _mono)
    try:
        lm._wait_for_nodes(3, timeout_s=120)
    except RuntimeError as exc:
        assert "only 1/3" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_launch_main_rejects_nnodes_below_two(monkeypatch):
    lm = _load_script_module("lm_test_main_nnodes", "launch_multinode.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_multinode.py",
            "--framework",
            "sglang",
            "--model",
            "/m",
            "--tp",
            "1",
            "--nnodes",
            "1",
            "--pid-dir",
            "/tmp",
            "--log-dir",
            "/tmp",
        ],
    )
    assert lm.main() == 2


def test_spawn_remote_vllm_worker_writes_sentinel_pid(tmp_path):
    lm = _load_script_module("lm_test_spawn_vllm", "launch_multinode.py")
    pid_dir = tmp_path / "p"
    log_dir = tmp_path / "l"
    rc = lm._spawn_remote(
        framework="vllm",
        model="/m",
        tp=8,
        nnodes=2,
        node_rank=1,
        head_ip="127.0.0.1",
        dist_init_port=5000,
        pid_dir=str(pid_dir),
        log_dir=str(log_dir),
        extra_args=[],
    )
    assert rc == 0
    assert (pid_dir / "rank_1.pid").read_text(encoding="utf-8") == "0"


def test_wait_health_true_on_200(monkeypatch):
    lm = _load_script_module("lm_test_health", "launch_multinode.py")
    calls = {"n": 0}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _urlopen(*_a, **_k):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        _urlopen,
    )
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    assert lm._wait_health(timeout_s=30) is True
    assert calls["n"] == 1


def test_wait_health_false_on_timeout(monkeypatch):
    lm = _load_script_module("lm_test_health2", "launch_multinode.py")

    def _urlopen(*_a, **_k):
        raise OSError("down")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr(lm.time, "sleep", lambda _s: None)
    times = iter([0.0, 100.0])

    def _mono():
        return next(times)

    monkeypatch.setattr(lm.time, "monotonic", _mono)
    assert lm._wait_health(timeout_s=5) is False


def test_find_head_pod_ip_prefers_kuberay_head_suffix():
    """SaFE may list submitter as resourceId 0; real Ray head podId contains '-head-'."""
    from inference_optimizer.multi_node import cli as mn_cli

    wl = {
        "pods": [
            {
                "podId": "primus-claw-40290b670f98c7-twr9t-khlfj",
                "resourceId": 0,
                "podIP": "172.16.152.122",
            },
            {
                "podId": "primus-claw-40290b670f98c7-twr9t-x6fkf-head-ddtvg",
                "resourceId": 1,
                "podIP": "10.245.131.77",
            },
        ],
    }
    assert mn_cli._find_head_pod_ip(wl) == "10.245.131.77"


def test_find_head_pod_ip_fallback_resource_id_zero():
    from inference_optimizer.multi_node import cli as mn_cli

    wl = {
        "pods": [
            {"podId": "legacy-pod", "resourceId": 0, "podIP": "10.0.0.1"},
        ],
    }
    assert mn_cli._find_head_pod_ip(wl) == "10.0.0.1"


def test_build_rayjob_entrypoints_empty_submitter_tail():
    import base64

    from inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        workspace="core42-hyperloom",
        display_name="t",
        image="harbor.example/sglang:1",
        nodes=2,
        gpus_per_node=8,
        cpus_per_node=96,
        mem_gi_per_node=1024,
        ephemeral_gi_per_node=400,
    )
    assert b["entryPoints"] == ["", ""]
    dec = base64.b64decode(b["env"]["RAY_JOB_ENTRYPOINT"]).decode().strip()
    assert dec == "tail -f /dev/null"


# ===========================================================================
# (formerly test_multi_node_env_ray.py)
# ===========================================================================
"""Tests for multi-node Ray address helpers."""

import json
import os

import pytest

from inference_optimizer.orchestrator.action_executors import _multi_node_env as mne


def test_ray_gcs_address_from_state_prefers_ray_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps({"ray_address": "10.1.2.3:6379", "head_pod_ip": "10.9.9.9"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.3:6379"


def test_ray_gcs_address_from_state_fallback_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.1.2.4"}), encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.4:6379"


def test_export_ray_address_to_os(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.0.0.5"}), encoding="utf-8")
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    mne.export_ray_address_to_os()
    assert os.environ.get("RAY_ADDRESS") == "10.0.0.5:6379"
