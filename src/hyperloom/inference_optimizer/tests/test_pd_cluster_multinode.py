# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atom and sglangrouter multi-node state, launch commands, and in-place exec."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node._internal import topology_file as topo
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import (
    engine_command,
    record_profile_paths,
    router_command,
)
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import LaunchSpec
from hyperloom.inference_optimizer.multi_node._internal.process_image import replace_process_image
from hyperloom.inference_optimizer.multi_node.state_paths import bind_state_file_to_session


def _spec(**overrides: object) -> LaunchSpec:
    base = dict(
        backend="atom",
        model="/models/Qwen3-30B-A3B",
        framework="sglang",
        prefill_tp=8,
        decode_tp=8,
        ep=1,
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


def test_router_and_engine_commands_match_the_platform_entrypoints() -> None:
    spec = _spec()
    router = router_command(spec)
    assert router.startswith("atomesh launch ")
    assert "--prefill http://10.0.1.11:8000 8998" in router
    assert "--decode http://10.0.2.21:8000" in router
    assert "--backend sglang" in router
    prefill = engine_command("prefill", spec)
    decode = engine_command("decode", _spec(backend="sglangrouter"))
    assert "python3 -m sglang.launch_server" in prefill
    assert "--disaggregation-mode prefill" in prefill
    assert "--disaggregation-bootstrap-port 8998" in prefill
    assert "--disaggregation-mode decode" in decode
    assert "--disaggregation-bootstrap-port 9098" in decode
    sglang_router = router_command(_spec(backend="sglangrouter"))
    assert sglang_router.startswith("python3 -m sglang_router.launch_router ")
    assert "atomesh" not in sglang_router


def test_atom_handoff_keeps_ssh_env_and_uses_one_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = {
        "schema": "topology-handoff/v1",
        "backend": "atom",
        "nodes": 5,
        "gpus_per_node": 8,
        "service_url": "http://10.0.0.10:8000",
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    session = tmp_path / "session"
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_GPUS_PER_NODE", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_MN_BACKEND", raising=False)
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.setenv("PD_MODE", "disaggregated")
    monkeypatch.setenv("MODEL_PATH", "/models/Qwen3-30B-A3B")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_KEY", "/workspace/hyperloom/runtime/mn_ssh/id_ed25519")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "2222")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_ROUTER_IPS", "10.0.0.10")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.11,10.0.1.12")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_IPS", "10.0.2.21,10.0.2.22")
    state_file = bind_state_file_to_session(session)
    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)

    state = ext.build_external_state_from_env()

    assert os.environ["MULTI_NODE_STATE"] == str(state_file)
    assert state["backend"] == "atom"
    assert state["external"] is True
    assert state["pd_mode"] == "disaggregated"
    assert state["nodes"] == 5
    assert state["gpus_per_node"] == 8
    assert state["ssh_port"] == 2222
    assert state["ssh_key_path"].endswith("id_ed25519")
    assert state["service_url"] == "http://10.0.0.10:8000"
    assert state["profile_trace"] == str(session / "runs" / "profile" / "torch_trace")
    assert state["steady_state_trace"] == ""
    assert [pod["role"] for pod in state["pods"]] == ["router", "prefill", "prefill", "decode", "decode"]
    assert {pod["sshPort"] for pod in state["pods"]} == {2222}
    assert state["pods"][0]["launch_command"].startswith("atomesh launch ")
    assert "sglang.launch_server" in state["pods"][1]["launch_command"]
    assert "--disaggregation-mode prefill" in state["pods"][1]["launch_command"]
    assert "--disaggregation-mode decode" in state["pods"][3]["launch_command"]
    # Infera's decode stride must not apply.
    assert all(pod["sshPort"] == 2222 for pod in state["decode_pods"])


def test_sglangrouter_file_does_not_require_an_infera_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "topology.json"
    path.write_text(
        json.dumps(
            {
                "schema": "topology-handoff/v1",
                "backend": "sglangrouter",
                "nodes": 3,
                "gpus_per_node": 8,
                "service_url": "http://10.0.0.10:8000",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    overlay = topo.topology_file_overlay()
    assert overlay is not None
    assert overlay["INFERENCE_OPTIMIZER_MN_BACKEND"] == "sglangrouter"
    assert "HYPERLOOM_MN_EXT_SSH_PORT" not in overlay


def test_record_profile_paths_updates_the_state_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = tmp_path / "session"
    path = bind_state_file_to_session(session)
    path.write_text(
        json.dumps({"backend": "atom", "profile_trace": "", "steady_state_trace": ""}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE", str(path))
    record_profile_paths(
        profile_trace=str(session / "runs" / "profile" / "torch_trace"),
        steady_state_trace=str(session / "runs" / "profile" / "torch_trace" / "rank0.trace.json.gz"),
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["profile_trace"].endswith("torch_trace")
    assert saved["steady_state_trace"].endswith("rank0.trace.json.gz")


def test_replace_process_image_keeps_the_pid_and_applies_env(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    pid = os.fork()
    if pid == 0:
        time.sleep(30)
        os._exit(0)
    try:
        replace_process_image(
            pid,
            ["/bin/sh", "-c", f"printf %s \"$HL_FROM_PARENT\" > {marker}"],
            env_overrides={"HL_FROM_PARENT": "kept"},
        )
        os.waitpid(pid, 0)
    finally:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    assert marker.read_text(encoding="utf-8") == "kept"
