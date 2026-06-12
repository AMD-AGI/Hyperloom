# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``multi_node/scripts/launch_dynamo_node.py``.

The pod-side Dynamo (idle-pod SSH) launcher is Ray-free / stdlib-only, so it is
imported directly via importlib. These guard the command construction (notably
the single-node distributed-flag omission that fixed the 0-output-token decode
regression), the pid1 env recovery (incl. the KUBERNETES_* discovery fix), and
the routable-pod-IP resolution used for cross-pod PD KV handshakes.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(unique_name: str):
    path = _repo_root() / "multi_node" / "scripts" / "launch_dynamo_node.py"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ns(**overrides):
    base = dict(
        framework="sglang", model="/m/x", tp=8, ep=1, nnodes=1,
        dist_init_port=5000, extra_args="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_sglang_cmd_single_node_omits_distributed_flags():
    # REGRESSION GUARD: nnodes==1 (single-pod PD role) must NOT pass
    # --nnodes/--node-rank/--dist-init-addr. Passing them made decode emit 0
    # output tokens (finish_reason=stop); the SaFE native launch omits them.
    lm = _load("ldn_sglang_single")
    cmd = lm._build_sglang_cmd(_ns(nnodes=1), node_rank=0, leader="10.0.0.1")
    assert "python3" in cmd and "dynamo.sglang" in cmd
    assert cmd[cmd.index("--model-path") + 1] == "/m/x"
    assert cmd[cmd.index("--tp-size") + 1] == "8"
    assert "--nnodes" not in cmd
    assert "--node-rank" not in cmd
    assert "--dist-init-addr" not in cmd


def test_sglang_cmd_multi_node_includes_distributed_flags():
    lm = _load("ldn_sglang_multi")
    cmd = lm._build_sglang_cmd(
        _ns(nnodes=2, dist_init_port=5000), node_rank=1, leader="10.0.0.9",
    )
    assert cmd[cmd.index("--nnodes") + 1] == "2"
    assert cmd[cmd.index("--node-rank") + 1] == "1"
    assert cmd[cmd.index("--dist-init-addr") + 1] == "10.0.0.9:5000"


def test_sglang_cmd_ep_and_extra_args():
    lm = _load("ldn_sglang_ep")
    cmd = lm._build_sglang_cmd(
        _ns(ep=8, extra_args="--mem-fraction-static 0.7"),
        node_rank=0, leader="x",
    )
    assert cmd[cmd.index("--ep-size") + 1] == "8"
    assert "--mem-fraction-static" in cmd and "0.7" in cmd


def test_sglang_cmd_ep_one_omits_ep_size():
    lm = _load("ldn_sglang_ep1")
    cmd = lm._build_sglang_cmd(_ns(ep=1), node_rank=0, leader="x")
    assert "--ep-size" not in cmd


def test_vllm_cmd_enables_expert_parallel_only_when_ep_gt_1():
    lm = _load("ldn_vllm")
    cmd_ep = lm._build_vllm_cmd(_ns(framework="vllm", tp=16, ep=8))
    assert cmd_ep[0:3] == ["python3", "-m", "dynamo.vllm"]
    assert cmd_ep[cmd_ep.index("--tensor-parallel-size") + 1] == "16"
    assert "--enable-expert-parallel" in cmd_ep
    cmd_noep = lm._build_vllm_cmd(_ns(framework="vllm", tp=8, ep=1))
    assert "--enable-expert-parallel" not in cmd_noep


def test_recover_container_env_overlays_pid1_and_prepends_venv(monkeypatch):
    # sshd sessions start with a bare env; the launcher overlays the LWS/NATS/
    # KUBERNETES_ keys from pid1 and guarantees /opt/venv/bin leads PATH.
    lm = _load("ldn_env")
    pid1 = (
        b"LWS_WORKER_INDEX=1\0LWS_LEADER_ADDRESS=10.0.0.1\0"
        b"KUBERNETES_SERVICE_HOST=10.96.0.1\0NATS_SERVER=nats://x:4222\0"
        b"IGNORED_VAR=should_not_copy\0PATH=/pid1bin\0"
    )
    monkeypatch.setattr(lm.Path, "read_bytes", lambda self: pid1)
    monkeypatch.setenv("PATH", "/usr/bin")
    env = lm._recover_container_env()
    assert env["LWS_WORKER_INDEX"] == "1"
    assert env["LWS_LEADER_ADDRESS"] == "10.0.0.1"
    assert env["KUBERNETES_SERVICE_HOST"] == "10.96.0.1"  # k8s discovery fix
    assert env["NATS_SERVER"] == "nats://x:4222"
    assert "IGNORED_VAR" not in env  # not in recover prefixes/names
    assert env["PATH"].startswith("/opt/venv/bin:")


def test_recover_container_env_survives_unreadable_proc(monkeypatch):
    # When /proc/1/environ is unreadable the launcher must not crash; it falls
    # back to the (bare) sshd session env rather than raising.
    lm = _load("ldn_env2")

    def _boom(self):
        raise OSError("no /proc/1/environ")

    monkeypatch.setattr(lm.Path, "read_bytes", _boom)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("MN_PROBE_KEY", "present")
    env = lm._recover_container_env()
    # Returns the session env (the early OSError return path predates the PATH
    # venv-prepend, so PATH is left as-is on this fallback).
    assert env["MN_PROBE_KEY"] == "present"
    assert env["PATH"] == "/usr/bin"


def test_resolve_pod_ip_prefers_pod_ip_downward_api():
    lm = _load("ldn_ip")
    assert lm._resolve_pod_ip({"POD_IP": "10.245.1.2"}) == "10.245.1.2"


def test_resolve_pod_ip_rejects_loopback_and_uses_egress_probe(monkeypatch):
    # A 127.* POD_IP must be rejected (cross-pod KV handshake would otherwise
    # dial its own localhost); the egress-route probe supplies the routable IP.
    lm = _load("ldn_ip2")
    import socket as _socket

    class _FakeSock:
        def connect(self, _addr):
            pass

        def getsockname(self):
            return ("10.245.9.9", 12345)

        def close(self):
            pass

    monkeypatch.setattr(_socket, "socket", lambda *a, **kw: _FakeSock())
    assert lm._resolve_pod_ip({"POD_IP": "127.0.0.1"}) == "10.245.9.9"
