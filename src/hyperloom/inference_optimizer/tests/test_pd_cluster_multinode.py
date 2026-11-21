# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomesh and sglangrouter multi-node state and launch commands."""

from __future__ import annotations

import socket
import types
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node._internal import pd_cluster as pd_internal
from hyperloom.inference_optimizer.multi_node._internal import topology_file as topo
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import engine_command
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import LaunchSpec
from hyperloom.inference_optimizer.multi_node.state_paths import bind_state_file_to_session


def _spec(**overrides: object) -> LaunchSpec:
    base = dict(
        model="/models/Qwen3-30B-A3B",
        framework="sglang",
        prefill_tp=8,
        decode_tp=8,
        ep=1,
        prefill_ep=1,
        decode_ep=1,
        kv="mooncake",
        ib="",
        server_args="",
        prefill_extra="",
        decode_extra="",
        prefill_url="http://10.0.1.11:8000",
        decode_url="http://10.0.2.21:8000",
    )
    base.update(overrides)
    return LaunchSpec(**base)  # type: ignore[arg-type]


def test_engine_commands_match_the_platform_entrypoints() -> None:
    spec = _spec()
    prefill = engine_command("prefill", spec)
    decode = engine_command("decode", spec)
    assert "python3 -m sglang.launch_server" in prefill
    assert "--disaggregation-mode prefill" in prefill
    assert "--disaggregation-bootstrap-port 8998" in prefill
    assert "--disaggregation-mode decode" in decode
    assert "--disaggregation-bootstrap-port 9098" in decode


def test_engine_listens_on_the_port_of_its_role_url() -> None:
    spec = _spec(
        prefill_url="http://w-prefill.ns.svc.cluster.local:8010",
        decode_url="http://w-decode.ns.svc.cluster.local:8020",
    )
    assert "--port 8010 " in engine_command("prefill", spec)
    assert "--port 8020 " in engine_command("decode", spec)
    no_port = _spec(prefill_url="http://w-prefill", decode_url="")
    assert "--port 8000 " in engine_command("prefill", no_port)
    assert "--port 8000 " in engine_command("decode", no_port)


def test_shared_args_replace_the_roles_value_of_a_flag_both_set() -> None:
    spec = _spec(
        server_args="--mem-fraction-static=0.8 --max-running-requests 64",
        decode_extra="--mem-fraction-static 0.85 --chunked-prefill-size 8192",
    )
    words = engine_command("decode", spec).split()
    assert [word for word in words if word.startswith("--mem-fraction-static")] == ["--mem-fraction-static=0.8"]
    assert "0.85" not in words
    assert words[-5:] == ["--chunked-prefill-size", "8192", "--mem-fraction-static=0.8", "--max-running-requests", "64"]


def test_engine_ep_size_follows_the_role_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EP", "PD_PREFILL_EP", "PD_DECODE_EP"):
        monkeypatch.delenv(name, raising=False)
    state = {"backend": "atomesh", "prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"]}
    restart = pd_internal.launch_spec(state, types.SimpleNamespace(model="/m", ep=1, pd_decode_ep=8))
    assert "--ep-size" not in engine_command("prefill", restart)
    assert "--ep-size 8" in engine_command("decode", restart)

    monkeypatch.setenv("EP", "8")
    monkeypatch.setenv("PD_PREFILL_EP", "1")
    handed_off = pd_internal.launch_spec(state)
    assert "--ep-size" not in engine_command("prefill", handed_off)
    assert "--ep-size 8" in engine_command("decode", handed_off)


def test_pods_record_the_engine_port_of_their_role() -> None:
    state = {"prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"], "ssh_port": 29000}
    pd_internal.set_pods(state, _spec(prefill_url="http://p-svc:8010", decode_url="http://d-svc:8020"))
    ports = {pod["role"]: pod.get("enginePort") for pod in state["prefill_pods"] + state["decode_pods"]}
    assert ports == {"prefill": 8010, "decode": 8020}


def test_platform_router_is_recorded_from_the_service_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def _resolve(host: str) -> str:
        if host != "router-svc.ns":
            raise socket.gaierror(host)
        return "192.168.0.5"

    monkeypatch.setattr(pd_internal.socket, "gethostbyname", _resolve)
    base = {"prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"], "ssh_port": 29000}
    state = {**base, "service_url": "http://router-svc.ns:8000"}
    pd_internal.set_pods(state, _spec())
    assert [len(state[f"{role}_pods"]) for role in ("router", "prefill", "decode")] == [1, 1, 1]
    router = state["router_pods"][0]
    assert router["podIP"] == "192.168.0.5"
    assert router["managed_by"] == "platform"
    assert "launch_command" not in router
    unresolved = {**base, "service_url": "http://unknown-host:8000"}
    pd_internal.set_pods(unresolved, _spec())
    assert unresolved["router_pods"][0]["podIP"] == "unknown-host"


def test_platform_router_ip_is_listed_and_stays_platform_managed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd_internal.socket, "gethostbyname", lambda host: "192.168.0.5")
    env = {"HYPERLOOM_MN_EXT_SSH_PORT": "29000"}
    state = {
        "backend": "sglangrouter",
        "service_url": "http://router-svc.ns:8000",
        "prefill_pod_ips": ["10.0.0.1"],
        "decode_pod_ips": ["10.0.0.2"],
    }
    pd_internal.stamp_pd_cluster(state, lambda name: env.get(name, ""))
    assert state["router_pod_ips"] == ["192.168.0.5"]
    assert state["router_pods"][0]["managed_by"] == "platform"

    pd_internal.set_pods(state, _spec())
    assert state["router_pod_ips"] == ["192.168.0.5"]
    assert [pod.get("managed_by") for pod in state["router_pods"]] == ["platform"]
    assert [pod["role"] for pod in pd_internal.role_engine_pods(state, "decode")] == ["decode"]


def test_atomesh_handoff_records_the_cluster_and_uses_one_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = tmp_path / "session"
    monkeypatch.delenv(topo.TOPOLOGY_FILE_ENV, raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "4")
    monkeypatch.delenv("INFERENCE_OPTIMIZER_GPUS_PER_NODE", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "atomesh")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://10.0.0.10:8000")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.setenv("PD_MODE", "disaggregated")
    monkeypatch.setenv("MODEL_PATH", "/models/Qwen3-30B-A3B")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_KEY", "/workspace/hyperloom/runtime/mn_ssh/id_ed25519")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "2222")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.11,10.0.1.12")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_IPS", "10.0.2.21,10.0.2.22")
    bind_state_file_to_session(session)
    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)

    state = ext.build_external_state_from_env()

    assert state["backend"] == "atomesh"
    assert state["external"] is True
    assert state["pd_mode"] == "disaggregated"
    assert state["nodes"] == 4
    assert state["gpus_per_node"] == 8
    assert state["ssh_port"] == 2222
    assert state["ssh_key_path"].endswith("id_ed25519")
    assert state["service_url"] == "http://10.0.0.10:8000"
    assert [len(state[f"{role}_pods"]) for role in ("router", "prefill", "decode")] == [1, 2, 2]
    assert {pod["sshPort"] for pod in state["prefill_pods"] + state["decode_pods"]} == {2222}
    assert state["router_pods"][0]["managed_by"] == "platform"
    assert "sglang.launch_server" in state["prefill_pods"][0]["launch_command"]
    assert "--disaggregation-mode prefill" in state["prefill_pods"][0]["launch_command"]
    assert "--disaggregation-mode decode" in state["decode_pods"][0]["launch_command"]
    # Infera's decode stride must not apply.
    assert all(pod["sshPort"] == 2222 for pod in state["decode_pods"])
