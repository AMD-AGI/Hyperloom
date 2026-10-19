# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``multi_node/scripts/launch_infera_node.py``."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import types

import pytest

from hyperloom.inference_optimizer.multi_node import cli as mn_cli


def _bundle() -> str:
    return mn_cli._read_bundled_pod_python_script("launch_infera_node.py", mn_cli._LAUNCHER_DEPS)


def _load_module():
    mod = types.ModuleType("launch_infera_node")
    exec(compile(_bundle(), "launch_infera_node_bundle.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize(
    ("main", "deps"),
    [
        ("launch_infera_node.py", mn_cli._LAUNCHER_DEPS),
        ("kernel_node_ops.py", mn_cli._KERNEL_NODE_OPS_DEPS),
    ],
)
def test_bundled_ssh_pod_script_runs_standalone(tmp_path, main, deps):
    script = tmp_path / "pod_script"
    script.write_text(mn_cli._read_bundled_pod_python_script(main, deps), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-I", str(script), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_build_sglang_cmd_uses_infera_engine():
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {"model": "/models/x", "tp": 8, "nnodes": 2, "dist_init_port": 5000, "ep": 8, "extra_args": ""},
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=1, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert cmd[0:3] == ["python3", "-m", "infera.engine.sglang"]
    assert "--discovery-backend" in cmd and "kubernetes" in cmd
    assert "--advertise-host" in cmd and "10.0.0.2" in cmd


def test_build_vllm_cmd_uses_infera_engine():
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {"model": "/models/x", "tp": 8, "ep": 1, "extra_args": ""},
    )()
    cmd = mod._build_vllm_cmd(ns, advertise_host="10.0.0.3")
    assert cmd[0:3] == ["python3", "-m", "infera.engine.vllm"]
    assert "--model-path" in cmd
    assert "--advertise-host" in cmd and "10.0.0.3" in cmd


def test_build_sglang_cmd_autofills_dp_size_for_dp_attention():
    # --enable-dp-attention without --dp-size => sglang would disable it at dp_size==1; the launcher must inject
    # --dp-size=tp so it takes effect.
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--enable-dp-attention --enable-dp-lm-head",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert "--enable-dp-attention" in cmd
    assert cmd[cmd.index("--dp-size") + 1] == "8"


def test_build_sglang_cmd_respects_explicit_dp_size():
    # An explicit --dp-size must not be overridden by the auto-fill.
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--enable-dp-attention --dp-size 4",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert cmd.count("--dp-size") == 1
    assert cmd[cmd.index("--dp-size") + 1] == "4"


def test_build_sglang_cmd_injects_skip_server_warmup_for_pd_leg():
    # PD warmup can hang until SGLANG_WARMUP_TIMEOUT; the launcher skips it for PD-disaggregated legs.
    mod = _load_module()
    pd_ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--disaggregation-mode decode",
        },
    )()
    pd_cmd = mod._build_sglang_cmd(pd_ns, node_rank=0, leader="l", advertise_host="h")
    assert pd_cmd.count("--skip-server-warmup") == 1

    agg_ns = type(
        "NS",
        (),
        {"model": "/models/x", "tp": 8, "nnodes": 1, "dist_init_port": 5000, "ep": 8, "extra_args": ""},
    )()
    agg_cmd = mod._build_sglang_cmd(agg_ns, node_rank=0, leader="l", advertise_host="h")
    assert "--skip-server-warmup" not in agg_cmd


def test_build_sglang_cmd_skip_warmup_not_duplicated():
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--skip-server-warmup --mem-fraction-static 0.8",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="l", advertise_host="h")
    assert cmd.count("--skip-server-warmup") == 1


def test_build_sglang_cmd_no_dp_size_without_dp_attention():
    # No DP-attention flag => no --dp-size injection.
    mod = _load_module()
    ns = type(
        "NS",
        (),
        {
            "model": "/models/x",
            "tp": 8,
            "nnodes": 1,
            "dist_init_port": 5000,
            "ep": 8,
            "extra_args": "--mem-fraction-static 0.8",
        },
    )()
    cmd = mod._build_sglang_cmd(ns, node_rank=0, leader="10.0.0.1", advertise_host="10.0.0.2")
    assert "--dp-size" not in cmd


def test_ray_start_runs_detached_from_the_launcher(monkeypatch):
    """``ray start`` gets its own session and no pipe of the launcher's, so its daemons neither pin the launcher's
    process group nor hold a pipe a reader waits on."""
    mod = _load_module()
    seen: list[tuple[list[str], dict]] = []

    def _fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        kwargs["stderr"].write("started head\n")
        return subprocess.CompletedProcess(cmd, 0)

    logged: list[str] = []
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod, "_log", logged.append)

    mod._ray_start("head", "10.0.0.1", {"PATH": "/usr/bin"})

    ((cmd, kwargs),) = seen
    assert cmd[:2] == ["/bin/bash", "-lc"] and cmd[2].startswith("ray start --head ")
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert "capture_output" not in kwargs
    assert kwargs["stdout"] is not subprocess.PIPE and kwargs["stderr"] is not subprocess.PIPE
    assert logged == ["ray start (head) rc=0 started head"]


def test_ray_start_real_daemon_leaves_the_launcher_group(monkeypatch, tmp_path):
    """With the launcher's real kwargs, a stand-in daemon lands outside its group and does not block the call."""
    import os
    import signal
    import threading

    mod = _load_module()
    pid_file = tmp_path / "daemon.pid"
    real_run = subprocess.run

    def _stand_in(cmd, **kwargs):
        # Same kwargs the launcher passes; only the command is a harmless stand-in for `ray start`.
        script = f"sleep 60 & echo $! > {pid_file}; echo stand-in >&2"
        return real_run(["/bin/bash", "-c", script], **kwargs)

    monkeypatch.setattr(mod.subprocess, "run", _stand_in)
    monkeypatch.setattr(mod, "_log", lambda msg: None)
    worker = threading.Thread(target=mod._ray_start, args=("head", "10.0.0.1", dict(os.environ)), daemon=True)
    worker.start()
    try:
        worker.join(timeout=20)
        assert not worker.is_alive(), "_ray_start blocked on a pipe the daemon inherited"
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        assert os.getpgid(pid) != os.getpgid(0)
        assert os.getsid(pid) != os.getsid(0)
    finally:
        if pid_file.is_file():
            # The stand-in daemon may already have exited; only a live one needs killing.
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8").strip()), signal.SIGKILL)
        worker.join(timeout=5)
