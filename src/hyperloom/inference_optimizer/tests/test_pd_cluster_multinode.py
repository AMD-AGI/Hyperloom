# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atom and sglangrouter multi-node state, launch commands, and in-place exec."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node._internal import pd_cluster as pd_internal
from hyperloom.inference_optimizer.multi_node._internal import topology_file as topo
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import (
    drop_unsupported_baseline_flags,
    engine_command,
    pods_for,
    record_profile_paths,
    router_command,
)
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import LaunchSpec
from hyperloom.inference_optimizer.multi_node._internal.process_image import replace_process_image
from hyperloom.inference_optimizer.multi_node.commands import pd_cluster as pd_cmds
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


def test_engine_listens_on_the_port_of_its_role_url() -> None:
    spec = _spec(
        prefill_url="http://w-prefill.ns.svc.cluster.local:8010",
        decode_url="http://w-decode.ns.svc.cluster.local:8020",
    )
    assert "--port 8010 " in engine_command("prefill", spec)
    assert "--port 8020 " in engine_command("decode", spec)
    assert "--port 8000 " in router_command(spec)
    no_port = _spec(prefill_url="http://w-prefill", decode_url="")
    assert "--port 8000 " in engine_command("prefill", no_port)
    assert "--port 8000 " in engine_command("decode", no_port)


_SPLIT_FLAGS = frozenset(
    {"--cuda-graph-bs-decode", "--cuda-graph-bs-prefill", "--cuda-graph-max-bs-decode", "--cuda-graph-max-bs-prefill"}
)


def test_engine_uses_split_cuda_graph_flags_when_the_engine_has_no_legacy_flag() -> None:
    spec = _spec(
        prefill_extra="--cuda-graph-bs 1 2 4 --mem-fraction-static 0.8",
        decode_extra="--cuda-graph-bs 1 2 --cuda-graph-max-bs=64",
        engine_flags=_SPLIT_FLAGS,
    )
    prefill = engine_command("prefill", spec)
    decode = engine_command("decode", spec)
    assert "--cuda-graph-bs" not in prefill
    assert "--cuda-graph-max-bs" not in prefill
    assert "--mem-fraction-static 0.8" in prefill
    assert "--cuda-graph-bs-decode 1 2 " in decode
    assert "--cuda-graph-max-bs-decode=64" in decode
    for flags in (frozenset(), _SPLIT_FLAGS | {"--cuda-graph-bs", "--cuda-graph-max-bs"}):
        legacy = _spec(decode_extra="--cuda-graph-bs 1 2", prefill_extra="--cuda-graph-bs 1 2", engine_flags=flags)
        assert "--cuda-graph-bs 1 2" in engine_command("prefill", legacy)
        assert "--cuda-graph-bs 1 2" in engine_command("decode", legacy)


def test_baseline_flags_the_engine_lacks_are_dropped_and_variant_flags_kept() -> None:
    words = "--load-balance-method round_robin --prefill-round-robin-balance --cuda-graph-bs 1 2 --bogus x".split()
    baseline = "--prefill-round-robin-balance --load-balance-method round_robin --cuda-graph-bs 1 2".split()
    engine = frozenset({"--load-balance-method", "--cuda-graph-bs-decode"})
    kept, dropped = drop_unsupported_baseline_flags(words, baseline, engine)
    assert kept == "--load-balance-method round_robin --cuda-graph-bs 1 2 --bogus x".split()
    assert dropped == ["--prefill-round-robin-balance"]
    assert drop_unsupported_baseline_flags(words, baseline, frozenset()) == (words, [])


def test_pods_record_the_engine_port_of_their_role() -> None:
    spec = _spec(prefill_url="http://p-svc:8010", decode_url="http://d-svc:8020")
    pods = pods_for(["10.0.0.9"], ["10.0.0.1"], ["10.0.0.2"], 29000, spec)
    ports = {pod["role"]: pod.get("enginePort") for pod in pods}
    assert ports == {"router": None, "prefill": 8010, "decode": 8020}


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
            ["/bin/sh", "-c", f'printf %s "$HL_FROM_PARENT" > {marker}'],
            env_overrides={"HL_FROM_PARENT": "kept"},
        )
        os.waitpid(pid, 0)
    finally:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    assert marker.read_text(encoding="utf-8") == "kept"


class _FakeCli:
    """Records SSH scripts per IP and answers with a scripted stdout."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str]] = []

    def _infera_ssh_bash_with_env(self, state, ip, script, *, timeout, port=None):
        self.calls.append((ip, script))
        return subprocess.CompletedProcess(["ssh"], 0, stdout=self.replies.get(ip, ""), stderr="")


_PD_STATE = {
    "backend": "sglangrouter",
    "ssh_port": 29000,
    "prefill_pods": [{"role": "prefill", "podIP": "10.0.0.1", "sshPort": 29000}],
    "decode_pods": [{"role": "decode", "podIP": "10.0.0.2", "sshPort": 29000}],
}


def test_engines_not_running_names_pods_whose_launch_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "HL_ALIVE\n", "10.0.0.2": "HL_GONE\n"})
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    assert pd_cmds.engines_not_running(dict(_PD_STATE)) == ["decode 10.0.0.2"]
    assert "/tmp/hyperloom-mn-decode.pid" in dict(fake.calls)["10.0.0.2"]


def test_engine_flags_are_probed_once_and_only_for_legacy_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "--cuda-graph-bs-decode\n--cuda-graph-bs-prefill\n--port\n"})
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    state = dict(_PD_STATE)
    assert pd_cmds.probe_engine_flags(state, _spec(), timeout=30) == frozenset()
    assert fake.calls == []
    spec = _spec(decode_extra="--cuda-graph-bs 1 2")
    flags = pd_cmds.probe_engine_flags(state, spec, timeout=30)
    assert flags == frozenset({"--cuda-graph-bs-decode", "--cuda-graph-bs-prefill", "--port"})
    assert pd_cmds.probe_engine_flags(state, spec, timeout=30) == flags
    assert len(fake.calls) == 1
    inferencex = dict(_PD_STATE, baseline_args_source="inferencex", baseline_decode_args="--port 1")
    assert pd_cmds.probe_engine_flags(inferencex, _spec(), timeout=30) == flags
    assert len(fake.calls) == 2


def test_engine_flag_probe_runs_with_the_container_environment(tmp_path: Path) -> None:
    """The probe resolves python3 from PID 1's PATH, not the SSH session's."""
    bin_dir = tmp_path / "venv-bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text("#!/bin/sh\necho 'usage: [--from-container-env]'\n", encoding="utf-8")
    fake_python.chmod(0o755)
    environ = tmp_path / "environ"
    environ.write_bytes(f"PATH={bin_dir}:/usr/bin:/bin\0HOME=/root\0".encode())
    script = pd_cmds._flag_probe_script(str(environ))
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert out.stdout.split() == ["--from-container-env"]


def test_restart_spec_drops_inferencex_baseline_flags_the_engine_lacks() -> None:
    state = dict(
        _PD_STATE,
        baseline_args_source="inferencex",
        baseline_decode_args="--prefill-round-robin-balance --load-balance-method round_robin",
        baseline_prefill_args="--load-balance-method round_robin",
    )
    spec = _spec(
        decode_extra="--prefill-round-robin-balance --load-balance-method round_robin",
        prefill_extra="--load-balance-method round_robin",
        engine_flags=frozenset({"--load-balance-method"}),
    )
    out = pd_cmds.without_unsupported_baseline_flags(state, spec)
    assert out.decode_extra == "--load-balance-method round_robin"
    assert out.prefill_extra == "--load-balance-method round_robin"
    manual = pd_cmds.without_unsupported_baseline_flags(dict(state, baseline_args_source="manual"), spec)
    assert manual == spec


def _run_launch(tmp_path: Path, command: str, env: dict[str, str], *, log_ip: str = "") -> subprocess.CompletedProcess:
    script = tmp_path / "launch.sh"
    script.write_text(pd_internal.launch_script("decode", command, env, log_ip=log_ip), encoding="utf-8")
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, SGLANG_HOST_IP="127.0.0.1"),
    )


def test_engine_launch_tees_its_output_to_the_shared_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(tmp_path / "no-gpu"),))
    logs = tmp_path / "logs"
    done = _run_launch(tmp_path, "echo engine-says-hi", {"HYPERLOOM_MN_SERVER_LOG_DIR": str(logs)}, log_ip="10.0.0.2")
    assert done.returncode == 0, done.stderr
    assert "engine-says-hi" in done.stdout
    log_file = logs / "mn_infera_server_10.0.0.2_r0.log"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and "engine-says-hi" not in (log_file.read_text() if log_file.exists() else ""):
        time.sleep(0.05)
    assert "engine-says-hi" in log_file.read_text()


def test_engine_launch_waits_for_the_gpu_to_be_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device = tmp_path / "kfd"
    device.write_text("")
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(device),))
    marker = tmp_path / "holder-exited"
    holder = subprocess.Popen(
        ["python3", "-c", f"import time; f = open({str(device)!r}); time.sleep(1.5); open({str(marker)!r}, 'w')"]
    )
    try:
        time.sleep(0.3)
        done = _run_launch(tmp_path, f"test -f {marker}", {})
    finally:
        holder.wait(timeout=10)
    assert done.returncode == 0, done.stderr


def test_engine_launch_gives_up_waiting_on_a_held_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device = tmp_path / "kfd"
    device.write_text("")
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(device),))
    monkeypatch.setattr(pd_internal, "GPU_RELEASE_WAIT_S", 1)
    holder = subprocess.Popen(["python3", "-c", f"import time; f = open({str(device)!r}); time.sleep(30)"])
    try:
        time.sleep(0.3)
        done = _run_launch(tmp_path, "echo launched", {})
    finally:
        holder.kill()
        holder.wait(timeout=10)
    assert done.returncode == 0
    assert "launched" in done.stdout
    assert f"GPU still held by pid(s) {holder.pid}" in done.stderr


def test_engine_launch_ignores_gpu_handles_it_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU handle the replaced engine left open in this process is not another holder."""
    device = tmp_path / "kfd"
    device.write_text("")
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(device),))
    monkeypatch.setattr(pd_internal, "GPU_RELEASE_WAIT_S", 3)
    script = tmp_path / "launch.sh"
    logs = tmp_path / "logs"
    script.write_text(
        pd_internal.launch_script("decode", "echo launched", {"HYPERLOOM_MN_SERVER_LOG_DIR": str(logs)}, log_ip="ip"),
        encoding="utf-8",
    )
    with open(device, encoding="utf-8") as held:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=(held.fileno(),),
            env=dict(os.environ, SGLANG_HOST_IP="127.0.0.1"),
        )
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, err
    assert "GPU still held" not in out + err
