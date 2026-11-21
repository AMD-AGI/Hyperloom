# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomesh and sglangrouter multi-node state, launch commands, and in-place exec."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import time
import types
from pathlib import Path

import pytest

from inference_suite.multinode import launch as suite_launch
from inference_suite.multinode import remote_driver

from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node._internal import pd_cluster as pd_internal
from hyperloom.inference_optimizer.multi_node._internal import topology_file as topo
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import (
    engine_command,
    launch_script,
    record_profile_paths,
)
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import LaunchSpec
from hyperloom.inference_optimizer.multi_node.commands import infera
from hyperloom.inference_optimizer.multi_node.commands import pd_cluster as pd_cmds
from hyperloom.inference_optimizer.multi_node.state_paths import bind_state_file_to_session, multi_node_state_env


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


def test_engine_launch_defaults_the_disaggregation_timeouts_without_overriding_a_set_one() -> None:
    script = launch_script("true", {"SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "5"})
    assert 'SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-100000}"' in script
    assert "export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5" in script
    assert "SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-" not in script


def test_mooncake_knobs_are_forwarded_to_the_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.multi_node.commands import infera

    monkeypatch.setenv("MC_ENABLE_DEST_DEVICE_AFFINITY", "1")
    assert infera._collect_forward_env(infera.POD_FORWARD_ENV_PREFIXES)["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "1"


def test_missing_baseline_names_the_manual_args_and_the_session_record() -> None:
    from hyperloom.inference_optimizer.multi_node._internal import baseline_args as ba

    with pytest.raises(ba.BaselineArgsMissing) as info:
        ba.resolve_baseline_args({"MODEL_PATH": "/models/M", "INFERENCEX_PATH": "/ix"})

    message = str(info.value)
    assert "--server-args / --pd-prefill-extra-args / --pd-decode-extra-args" in message
    assert "previous launch recorded them" in message
    assert "INFERENCEX" not in message and "Magpie" not in message


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


def test_pods_record_the_engine_port_of_their_role() -> None:
    state = {"prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"], "ssh_port": 29000}
    pd_internal.set_pods(state, _spec(prefill_url="http://p-svc:8010", decode_url="http://d-svc:8020"))
    ports = {pod["role"]: pod.get("enginePort") for pod in state["prefill_pods"] + state["decode_pods"]}
    assert ports == {"prefill": 8010, "decode": 8020}


def test_previous_baseline_args_are_reused_only_for_the_same_model_and_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.multi_node._internal import baseline_args as ba

    recorded: dict[str, object] = {}
    monkeypatch.setenv("MODEL_PATH", "/models/DeepSeek-R1")
    monkeypatch.delenv("EP", raising=False)
    ba.record_baseline_args(recorded, ba.BaselineArgs(ba.SOURCE_MANUAL, "--context-length 8192", "--x"))
    same = {"MODEL_PATH": "/models/DeepSeek-R1"}
    assert ba.resolve_baseline_args(env=same, previous=recorded).prefill == "--context-length 8192"
    for env in ({"MODEL_PATH": "/models/Qwen3"}, {"MODEL_PATH": "/models/DeepSeek-R1", "EP": "8"}):
        with pytest.raises(ba.BaselineArgsMissing):
            ba.resolve_baseline_args(env=env, previous=recorded)
    legacy = {k: v for k, v in recorded.items() if k != "baseline_identity"}
    with pytest.raises(ba.BaselineArgsMissing):
        ba.resolve_baseline_args(env=same, previous=legacy)


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
    assert [pod["role"] for pod in pd_cmds._engine_pods(state)] == ["prefill", "decode"]


def test_stamp_keeps_the_pinned_profile_dir_and_restores_recorded_traces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pinned dir is where this session's engines write; a recorded dir from disk must not replace it."""
    pinned = tmp_path / "new" / "torch_trace"
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", str(pinned))
    path = bind_state_file_to_session(tmp_path / "session")
    path.write_text(
        json.dumps(
            {
                "backend": "sglangrouter",
                "service_url": "http://router-svc.ns:8000",
                "prefill_pod_ips": ["10.0.0.1"],
                "profile_trace": str(tmp_path / "old" / "torch_trace"),
                "decode_trace": "/old/decode/rank0.trace.json.gz",
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    monkeypatch.setattr(pd_internal.socket, "gethostbyname", lambda host: "192.168.0.5")
    env = {"HYPERLOOM_MN_EXT_PREFILL_IPS": "10.0.0.1", "HYPERLOOM_MN_EXT_DECODE_IPS": "10.0.0.2"}
    state = {"backend": "sglangrouter", "service_url": "http://router-svc.ns:8000"}
    pd_internal.stamp_pd_cluster(state, lambda name: env.get(name, ""))
    assert state["profile_trace"] == str(pinned)
    assert state["decode_trace"] == "/old/decode/rank0.trace.json.gz"


def test_state_records_the_user_data_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.session.paths import default_workspace_root

    monkeypatch.delenv(topo.TOPOLOGY_FILE_ENV, raising=False)
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://10.0.0.10:8000")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "user"))
    assert ext.build_external_state_from_env()["user_data_path"] == str(tmp_path / "user")
    monkeypatch.delenv("USER_DATA_PATH")
    assert ext.build_external_state_from_env()["user_data_path"] == str(default_workspace_root())


def test_multi_node_state_env_names_the_bound_state_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MULTI_NODE_STATE", raising=False)
    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    assert multi_node_state_env() == {}
    path = bind_state_file_to_session(tmp_path / "session")
    assert "MULTI_NODE_STATE" not in os.environ
    assert multi_node_state_env() == {}
    path.write_text("{}", encoding="utf-8")
    assert multi_node_state_env() == {"MULTI_NODE_STATE": str(path), "MULTI_NODE_STATE_FILE": str(path)}


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
    assert state["profile_trace"] == str(session / "runs" / "profile" / "torch_trace")
    assert state["steady_state_trace"] == ""
    assert [len(state[f"{role}_pods"]) for role in ("router", "prefill", "decode")] == [1, 2, 2]
    assert {pod["sshPort"] for pod in state["prefill_pods"] + state["decode_pods"]} == {2222}
    assert state["router_pods"][0]["managed_by"] == "platform"
    assert "sglang.launch_server" in state["prefill_pods"][0]["launch_command"]
    assert "--disaggregation-mode prefill" in state["prefill_pods"][0]["launch_command"]
    assert "--disaggregation-mode decode" in state["decode_pods"][0]["launch_command"]
    # Infera's decode stride must not apply.
    assert all(pod["sshPort"] == 2222 for pod in state["decode_pods"])


def test_record_profile_paths_updates_the_state_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = tmp_path / "session"
    path = bind_state_file_to_session(session)
    path.write_text(
        json.dumps({"backend": "atomesh", "profile_trace": "", "steady_state_trace": ""}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    decode = str(session / "runs" / "profile" / "torch_trace" / "decode" / "rank0.trace.json.gz")
    prefill = str(session / "runs" / "profile" / "torch_trace" / "prefill" / "rank0.trace.json.gz")
    record_profile_paths(
        profile_trace=str(session / "runs" / "profile" / "torch_trace"),
        steady_state_trace=decode,
        role_traces={"decode": decode, "prefill": prefill},
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["profile_trace"].endswith("torch_trace")
    assert saved["steady_state_trace"] == decode
    assert saved["decode_trace"] == decode
    assert saved["prefill_trace"] == prefill


def test_replace_process_image_keeps_the_pid(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    pid = os.fork()
    if pid == 0:
        time.sleep(30)
        os._exit(0)
    try:
        remote_driver.replace_process_image(
            pid, ["/bin/sh", "-c", f"printf kept > {marker}"], str(tmp_path / "engine.environ")
        )
        os.waitpid(pid, 0)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, 9)
    assert marker.read_text(encoding="utf-8") == "kept"


def _wait_for_cmdline(pid: int, prefix: bytes) -> None:
    """Poll until ``pid`` runs a command line starting with ``prefix``."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if Path(f"/proc/{pid}/cmdline").read_bytes().startswith(prefix):
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} never ran {prefix!r}")


def test_replace_process_image_drops_env_exported_by_the_previous_image(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    snapshot = str(tmp_path / "engine.environ")
    pid = os.fork()
    if pid == 0:
        os.execve("/bin/sleep", ["/bin/sleep", "30"], {**os.environ, "HL_ORIGINAL": "orig"})
    try:
        _wait_for_cmdline(pid, b"/bin/sleep\x0030")
        remote_driver.replace_process_image(pid, ["/bin/sh", "-c", "export HL_LEAKED=1; exec /bin/sleep 30"], snapshot)
        _wait_for_cmdline(pid, b"/bin/sleep\x0030")
        remote_driver.replace_process_image(
            pid, ["/bin/sh", "-c", f'printf "%s|%s" "${{HL_ORIGINAL:-}}" "${{HL_LEAKED:-}}" > {marker}'], snapshot
        )
        os.waitpid(pid, 0)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, 9)
    assert marker.read_text(encoding="utf-8") == "orig|"


def test_exec_strings_write_undecodable_environment_bytes_back_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[bytes] = []
    monkeypatch.setattr(remote_driver, "_poke", lambda _pid, _addr, blob: written.append(blob))
    raw = b"HL_RAW=\xff\xfe"

    remote_driver._write_strings(4242, 0x1000, [raw.decode("utf-8", "surrogateescape")])

    assert written[0] == raw + b"\0"


class _FakeCli:
    """Records SSH scripts per IP and answers with a scripted stdout."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str]] = []

    def _infera_ssh_bash_with_env(self, state, ip, script, *, timeout, port=None):
        self.calls.append((ip, script))
        return subprocess.CompletedProcess(["ssh"], 0, stdout=self.replies.get(ip, ""), stderr="")

    def _extract_pod_json(self, logs: str) -> dict | None:
        return None


_PD_STATE = {
    "backend": "sglangrouter",
    "ssh_port": 29000,
    "prefill_pods": [{"role": "prefill", "podIP": "10.0.0.1", "sshPort": 29000, "launch_command": "true"}],
    "decode_pods": [{"role": "decode", "podIP": "10.0.0.2", "sshPort": 29000, "launch_command": "true"}],
}


def _route_pd_ssh(monkeypatch: pytest.MonkeyPatch, bash_with_env) -> None:
    """Point the multi-node CLI at a PD state whose engine-pod SSH goes to ``bash_with_env``."""
    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    state = {**_PD_STATE, "pd_mode": "disaggregated", "ssh_key_path": "/key"}
    for target in (mn_cli, infera._mn_cli):
        monkeypatch.setattr(target, "_load_state", lambda: dict(state))
        monkeypatch.setattr(target, "_read_bundled_pod_python_script", lambda *_args: "print('patched')")
    monkeypatch.setattr(infera, "_infera_ssh_bash_with_env", bash_with_env)


def test_tracelens_patch_failure_reports_the_pod_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import argparse

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    pod_error = "cannot derive per-version patches subdir from version '0.5.23.dev1+g1'"
    stdout = json.dumps({"status": "failed", "per_pod": [{"status": "failed", "error": pod_error}]})
    _route_pd_ssh(
        monkeypatch,
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ssh"], 1, stdout=stdout, stderr="bash: warning: setlocale: LC_ALL: cannot change locale\n"
        ),
    )

    ns = argparse.Namespace(tracelens_root="/tl", sglang_version_pin=None, patch_set="roofline")
    assert mn_cli.cmd_apply_tracelens_patch(ns) == 1
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{") :])
    assert [failure["error"] for failure in report["failures"]] == [pod_error, pod_error]


def _probe_state(tmp_path: Path) -> dict:
    return {**_PD_STATE, "ssh_key_path": "/key", "profile_trace": str(tmp_path)}


def _probe_cli(reply) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(_infera_ssh_bash_with_env=lambda state, ip, script, *, timeout, port=None: reply(ip, script))


def _run_here(_ip: str, script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _decode_unreachable(ip: str, script: str) -> subprocess.CompletedProcess:
    if ip == "10.0.0.2":
        raise subprocess.TimeoutExpired("ssh", 30)
    return _run_here(ip, script)


def test_profile_dir_probe_passes_when_every_engine_reads_the_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_pod_ssh(monkeypatch, _probe_cli(_run_here))
    state = _probe_state(tmp_path)

    assert pd_cmds.record_profile_dir_share(state) == ""
    assert state["profile_trace_shared"] is True
    assert "profile_trace_share_error" not in state
    assert list(tmp_path.iterdir()) == []


def test_an_explicit_profile_dir_not_created_yet_is_probed_as_shared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace_dir = tmp_path / "traces-run42"
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", str(trace_dir))
    _route_pod_ssh(monkeypatch, _probe_cli(_run_here))
    state = {**_PD_STATE, "ssh_key_path": "/key"}

    pd_internal._pin_profile_dir(state)

    assert state["profile_trace"] == str(trace_dir)
    assert pd_cmds.record_profile_dir_share(state) == ""


@pytest.mark.parametrize(
    "reply",
    [
        lambda ip, _s: subprocess.CompletedProcess(["ssh"], 1, stdout="", stderr="No such file or directory"),
        lambda ip, _s: subprocess.CompletedProcess(["ssh"], 0, stdout="other-token\n", stderr=""),
        _decode_unreachable,
    ],
    ids=["missing", "different-file", "ssh-timeout"],
)
def test_profile_dir_probe_records_why_the_dir_is_not_shared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply
) -> None:
    _route_pod_ssh(monkeypatch, _probe_cli(reply))
    state = _probe_state(tmp_path)

    reason = pd_cmds.record_profile_dir_share(state)

    assert reason and "pod 10.0.0." in reason
    assert state["profile_trace_shared"] is False
    assert state["profile_trace_share_error"] == reason
    assert list(tmp_path.iterdir()) == []


def test_tracelens_patch_runs_under_the_container_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python3").write_text('#!/bin/sh\necho "venv-python $*"\n', encoding="utf-8")
    (venv_bin / "python3").chmod(0o755)
    environ = tmp_path / "environ"
    environ.write_bytes(f"PATH={venv_bin}:/usr/bin:/bin\0HOME=/root\0".encode())
    scripts: list[str] = []

    def _bash_with_env(state, ip, script, *, timeout, port=None):
        scripts.append(script)
        return subprocess.CompletedProcess(["ssh"], 0, stdout='{"status": "applied", "per_pod": [{}]}', stderr="")

    _route_pd_ssh(monkeypatch, _bash_with_env)
    monkeypatch.delenv("HYPERLOOM_MN_POD_PYTHON", raising=False)
    monkeypatch.setattr(infera, "CONTAINER_ENVIRON_PATH", str(environ))

    ns = argparse.Namespace(tracelens_root="/tl", sglang_version_pin=None, patch_set="roofline")
    assert mn_cli.cmd_apply_tracelens_patch(ns) == 0
    assert len(scripts) == 2
    ran = subprocess.run(
        ["bash", "-c", scripts[0]], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=True
    )
    assert ran.stdout.startswith("venv-python ")
    assert "--local --tracelens-root /tl --patch-set roofline" in ran.stdout


def test_kernel_patch_and_geak_install_reach_every_engine_pod(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli

    monkeypatch.setenv("HYPERLOOM_GEAK_SRC", str(tmp_path / "geak"))
    calls: list[tuple[str, str]] = []

    def _bash_with_env(_state, ip, script, *, timeout, port=None):
        calls.append((ip, script))
        return subprocess.CompletedProcess(["ssh"], 0, stdout='{"status": "ok"}', stderr="")

    _route_pd_ssh(monkeypatch, _bash_with_env)
    patch_file = tmp_path / "kernel.patch"
    patch_file.write_text("--- a\n+++ b\n", encoding="utf-8")
    ns = argparse.Namespace(
        patch_file=str(patch_file),
        target_path="/opt/sglang/k.py",
        backup_dir="/tmp/bk",
        kernel_id="k0",
        jit_build_dir="",
        timeout_sec=30,
    )
    assert mn_cli.cmd_apply_patch(ns) == 0
    mn_cli.install_geak_on_pods_best_effort()
    assert [ip for ip, _ in calls] == ["10.0.0.1", "10.0.0.2", "10.0.0.1", "10.0.0.2"]
    assert all(infera.CONTAINER_ENVIRON_PATH in script for _, script in calls)


def test_engine_restart_gives_each_pod_its_roles_profiler_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({})
    _route_pod_ssh(monkeypatch, fake)
    env = {"SGLANG_TORCH_PROFILER_DIR": "/shared/torch_trace"}
    second_decode = {"role": "decode", "podIP": "10.0.0.4", "sshPort": 29000, "launch_command": "true"}
    state = dict(_PD_STATE, decode_pods=[*_PD_STATE["decode_pods"], second_decode])
    for pod in [*state["prefill_pods"], *state["decode_pods"]]:
        pd_cmds._ssh_pod(state, pod, extra_env=env, timeout=5)
    scripts = dict(fake.calls)
    assert "export SGLANG_TORCH_PROFILER_DIR=/shared/torch_trace/prefill\n" in scripts["10.0.0.1"]
    assert "export SGLANG_TORCH_PROFILER_DIR=/shared/torch_trace/decode\n" in scripts["10.0.0.2"]
    assert "export SGLANG_TORCH_PROFILER_DIR=/shared/torch_trace/decode\n" in scripts["10.0.0.4"]
    assert env == {"SGLANG_TORCH_PROFILER_DIR": "/shared/torch_trace"}


def test_engines_not_running_names_pods_whose_launch_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "HL_ALIVE\n", "10.0.0.2": "HL_GONE\n"})
    _route_pod_ssh(monkeypatch, fake)
    assert pd_cmds.engines_not_running(dict(_PD_STATE)) == ["decode 10.0.0.2"]
    assert "/tmp/hyperloom-mn-decode.pid" in dict(fake.calls)["10.0.0.2"]


def test_remote_driver_finds_the_server_by_its_needle(tmp_path: Path) -> None:
    token = f"hl-needle-{os.getpid()}-{time.monotonic_ns()}"
    named = subprocess.Popen(["python3", "-c", "import time; time.sleep(30)", token])
    try:
        time.sleep(0.2)
        assert remote_driver.find_server(str(tmp_path / "none.pid"), token) == named.pid
        assert remote_driver.find_server(str(tmp_path / "none.pid"), f"absent-{token}") == 0
    finally:
        named.kill()
        named.wait()


def _fake_driver(monkeypatch: pytest.MonkeyPatch, *, held: bool, calls: list) -> None:
    """Replace the driver's process operations with recorders."""
    monkeypatch.setattr(remote_driver, "_held_by_launcher", lambda pid: held)
    monkeypatch.setattr(
        remote_driver, "replace_process_image", lambda pid, argv, snapshot: calls.append(("exec", pid, argv[0]))
    )
    monkeypatch.setattr(remote_driver, "descendant_pids", lambda pid: [4243])
    monkeypatch.setattr(
        remote_driver, "baseline_environ", lambda pid, snapshot: calls.append(("env", pid, snapshot)) or [f"FROM={pid}"]
    )
    monkeypatch.setattr(
        remote_driver, "_spawn", lambda launch, env: calls.append(("spawn", launch, tuple(env.items()))) or 5151
    )
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append(("kill", pid, sig)))
    monkeypatch.delenv("HL_ENV_SOURCE", raising=False)


@pytest.mark.parametrize("held", [True, False])
def test_remote_driver_execs_in_place_only_for_the_launcher_waited_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, held: bool
) -> None:
    calls: list = []
    _fake_driver(monkeypatch, held=held, calls=calls)
    pidfile = str(tmp_path / "role.pid")
    action, _pid = remote_driver._restart(4242, pidfile, str(tmp_path / "launch.sh"))
    if held:
        assert action == "exec"
        assert [c[0] for c in calls] == ["exec"]
    else:
        assert action == "respawn"
        assert ("kill", 4242, 9) in calls and ("kill", 4243, 9) in calls
        assert not [c for c in calls if c[0] == "exec"]
        assert calls.index(("env", 4242, pidfile + ".environ")) < calls.index(("kill", 4242, 9))
        assert [c[2] for c in calls if c[0] == "spawn"] == [(("FROM", "4242"),)]


def test_spawn_without_a_server_takes_the_container_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list = []
    _fake_driver(monkeypatch, held=False, calls=calls)
    pidfile = str(tmp_path / "role.pid")
    assert remote_driver._restart(0, pidfile, str(tmp_path / "launch.sh"))[0] == "spawn"
    assert ("env", 1, pidfile + ".environ") in calls
    assert [c[2] for c in calls if c[0] == "spawn"] == [(("FROM", "1"),)]


def test_respawn_takes_the_container_environment_when_the_server_environment_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list = []
    _fake_driver(monkeypatch, held=False, calls=calls)
    monkeypatch.setattr(
        remote_driver, "baseline_environ", lambda pid, snapshot: calls.append(("env", pid, snapshot)) or []
    )
    monkeypatch.setattr(remote_driver, "read_environ", lambda pid: ["PATH=/opt/venv/bin:/usr/bin", f"FROM={pid}"])
    assert remote_driver._restart(4242, str(tmp_path / "role.pid"), str(tmp_path / "launch.sh"))[0] == "respawn"
    assert [c[2] for c in calls if c[0] == "spawn"] == [(("PATH", "/opt/venv/bin:/usr/bin"), ("FROM", "1"))]


def test_detached_spawn_runs_with_the_given_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HL_CONSOLE", str(tmp_path / "none"))
    launch = tmp_path / "launch.sh"
    launch.write_text('#!/bin/bash\necho "hl-env-$HL_TEST_MARK"\n', encoding="utf-8")
    pid = remote_driver._spawn(str(launch), {"PATH": os.environ["PATH"], "HL_TEST_MARK": "from-container"})
    os.waitpid(pid, 0)
    assert "hl-env-from-container" in (tmp_path / "launch.sh.log").read_text(encoding="utf-8")


def test_detached_spawn_logs_to_file_and_console(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    console = tmp_path / "console"
    console.write_text("", encoding="utf-8")
    monkeypatch.setenv("HL_CONSOLE", str(console))
    launch = tmp_path / "launch.sh"
    launch.write_text("#!/bin/bash\necho hl-spawned-$$\n", encoding="utf-8")
    pid = remote_driver._spawn(str(launch), {"PATH": os.environ["PATH"]})
    os.waitpid(pid, 0)
    log_file = tmp_path / "launch.sh.log"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not all(
        path.exists() and f"hl-spawned-{pid}" in path.read_text(encoding="utf-8") for path in (log_file, console)
    ):
        time.sleep(0.05)
    assert f"hl-spawned-{pid}" in log_file.read_text(encoding="utf-8")
    assert f"hl-spawned-{pid}" in console.read_text(encoding="utf-8")


def test_liveness_probe_greps_the_engine_command(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "HL_ALIVE\n", "10.0.0.2": "HL_ALIVE\n"})
    _route_pod_ssh(monkeypatch, fake)
    assert pd_cmds.engines_not_running(dict(_PD_STATE)) == []
    script = dict(fake.calls)["10.0.0.1"]
    assert "grep -qF -e sglang.launch_server" in script


def test_engine_flags_are_probed_once_and_only_for_legacy_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "--cuda-graph-bs-decode\n--cuda-graph-bs-prefill\n--port\n"})
    _route_pod_ssh(monkeypatch, fake)
    state = dict(_PD_STATE)
    assert pd_cmds.probe_engine_flags(state, _spec(), timeout=30) == frozenset()
    assert fake.calls == []
    spec = _spec(decode_extra="--cuda-graph-bs 1 2")
    flags = pd_cmds.probe_engine_flags(state, spec, timeout=30)
    assert flags == frozenset({"--cuda-graph-bs-decode", "--cuda-graph-bs-prefill", "--port"})
    assert pd_cmds.probe_engine_flags(state, spec, timeout=30) == flags
    assert len(fake.calls) == 1


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


def _run_launch(tmp_path: Path, command: str, env: dict[str, str], *, log_ip: str = "") -> subprocess.CompletedProcess:
    script = tmp_path / "launch.sh"
    script.write_text(pd_internal.launch_script(command, env, log_ip=log_ip), encoding="utf-8")
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, SGLANG_HOST_IP="127.0.0.1"),
    )


def test_engine_launch_tees_its_output_to_the_shared_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(suite_launch, "GPU_DEVICE_PATHS", (str(tmp_path / "no-gpu"),))
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
    monkeypatch.setattr(suite_launch, "GPU_DEVICE_PATHS", (str(device),))
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
    monkeypatch.setattr(suite_launch, "GPU_DEVICE_PATHS", (str(device),))
    monkeypatch.setattr(suite_launch, "GPU_RELEASE_WAIT_S", 1)
    holder = subprocess.Popen(["python3", "-c", f"import time; f = open({str(device)!r}); time.sleep(30)"])
    try:
        time.sleep(0.3)
        done = _run_launch(tmp_path, "echo launched", {})
    finally:
        holder.kill()
        holder.wait(timeout=10)
    assert done.returncode == 0
    assert "launched" in done.stdout
    assert f"GPU still held by {holder.pid}" in done.stderr


def test_engine_launch_ignores_gpu_handles_it_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU handle the replaced engine left open in this process is not another holder."""
    device = tmp_path / "kfd"
    device.write_text("")
    monkeypatch.setattr(suite_launch, "GPU_DEVICE_PATHS", (str(device),))
    monkeypatch.setattr(suite_launch, "GPU_RELEASE_WAIT_S", 3)
    script = tmp_path / "launch.sh"
    logs = tmp_path / "logs"
    script.write_text(
        pd_internal.launch_script("echo launched", {"HYPERLOOM_MN_SERVER_LOG_DIR": str(logs)}, log_ip="ip"),
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


def test_engine_launch_closes_listening_sockets_it_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine can bind its port when the replaced engine left that port's listening socket open in this process."""
    monkeypatch.setattr(suite_launch, "GPU_DEVICE_PATHS", (str(tmp_path / "no-gpu"),))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    script = tmp_path / "launch.sh"
    bind = f"python3 -c \"import socket; socket.socket().bind(('127.0.0.1', {port})); print('bound')\""
    script.write_text(pd_internal.launch_script(bind, {}), encoding="utf-8")
    with listener:
        proc = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=(listener.fileno(),),
            env=dict(os.environ, SGLANG_HOST_IP="127.0.0.1"),
        )
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0, err
    assert "bound" in out


def _route_pod_ssh(monkeypatch: pytest.MonkeyPatch, fake: object) -> None:
    """Point the pod hand-off commands' state and SSH helpers at the ones ``fake`` defines."""
    for name in ("_load_state", "_save_state", "_extract_pod_json", "_infera_ssh_bash_with_env"):
        if hasattr(fake, name):
            monkeypatch.setattr(pd_cmds, name, getattr(fake, name))


class _RestartCli:
    """Multi-node CLI stand-in that records restart SSH calls and the saved state."""

    EXIT_CONFIG_ERROR = 3
    ConfigurationError = RuntimeError

    def __init__(self, state: dict, replies: dict | None = None) -> None:
        self.state = state
        self.replies = replies or {}
        self.calls: list[tuple[str, int, str]] = []
        self.saved: dict | None = None

    def _load_state(self) -> dict:
        return dict(self.state)

    def _save_state(self, state: dict) -> None:
        self.saved = state

    def _normalize_extra_args(self, raw: str) -> str:
        return str(raw or "").strip()

    def _extract_pod_json(self, logs: str) -> dict | None:
        return None

    def _infera_ssh_bash_with_env(self, state, ip, script, *, timeout, port=None):
        self.calls.append((ip, port, script))
        reply = self.replies.get(ip, 0)
        if reply == "timeout":
            raise subprocess.TimeoutExpired("ssh", timeout)
        return subprocess.CompletedProcess(["ssh"], reply, stdout="", stderr="engine failed" if reply else "")


_RESTART_STATE = {
    "backend": "atomesh",
    "ssh_key_path": "/keys/id",
    "ssh_port": 29000,
    "prefill_pod_ips": ["10.0.0.1"],
    "decode_pod_ips": ["10.0.0.2"],
    "gpus_per_node": 8,
}


def _restart_args(**overrides: object) -> types.SimpleNamespace:
    base = dict(
        model="/models/m",
        framework="sglang",
        tp=8,
        ep=1,
        extra_args="--mem-fraction-static 0.8",
        pd_prefill_extra_args="",
        pd_decode_extra_args="",
        poll_timeout=60,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_restart_relaunches_every_engine_pod_and_records_the_round(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cli = _RestartCli(_RESTART_STATE)
    _route_pod_ssh(monkeypatch, cli)

    assert pd_cmds.restart_pd_cluster(_restart_args()) == 0

    assert [(ip, port) for ip, port, _script in cli.calls] == [("10.0.0.1", 29000), ("10.0.0.2", 29000)]
    assert "--disaggregation-mode prefill" in cli.calls[0][2]
    assert "--disaggregation-mode decode" in cli.calls[1][2]
    assert cli.saved is not None
    assert cli.saved["last_restart_model"] == "/models/m"
    assert cli.saved["last_restart_extra_args"] == "--mem-fraction-static 0.8"
    assert json.loads(capsys.readouterr().out)["rc"] == 0


def test_restart_reports_failed_and_timed_out_pods(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cli = _RestartCli(_RESTART_STATE, replies={"10.0.0.1": "timeout", "10.0.0.2": 1})
    _route_pod_ssh(monkeypatch, cli)

    assert pd_cmds.restart_pd_cluster(_restart_args()) == 1

    results = {item["podIP"]: item for item in json.loads(capsys.readouterr().out)["results"]}
    assert results["10.0.0.1"]["rc"] == 124
    assert results["10.0.0.2"]["stderr"] == "engine failed"


def test_restart_refuses_denied_server_args_and_an_empty_model(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _RestartCli(_RESTART_STATE)
    _route_pod_ssh(monkeypatch, cli)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    assert pd_cmds.restart_pd_cluster(_restart_args(extra_args="--model-path /other")) == cli.EXIT_CONFIG_ERROR
    assert pd_cmds.restart_pd_cluster(_restart_args(model="")) == cli.EXIT_CONFIG_ERROR
    assert cli.calls == []


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ({"backend": "infera"}, "not atomesh or sglangrouter"),
        ({"ssh_key_path": ""}, "HYPERLOOM_MN_EXT_SSH_KEY"),
        ({"ssh_port": 0}, "HYPERLOOM_MN_EXT_SSH_PORT"),
        ({"decode_pod_ips": []}, "HYPERLOOM_MN_EXT_DECODE_IPS"),
    ],
)
def test_restart_requires_a_complete_pd_handoff(monkeypatch: pytest.MonkeyPatch, missing: dict, message: str) -> None:
    cli = _RestartCli({**_RESTART_STATE, **missing})
    _route_pod_ssh(monkeypatch, cli)
    with pytest.raises(RuntimeError, match=message):
        pd_cmds.restart_pd_cluster(_restart_args())


def test_pod_handoff_round_restart_applies_only_the_candidate_over_each_roles_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The session base and the multi-node sglang default stay off the engines; the candidate wins a shared flag."""
    import asyncio

    from hyperloom.inference_optimizer.multi_node import cli as mn_cli
    from hyperloom.orchestrator.actions.executors import _multi_node_env
    from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as life
    from hyperloom.orchestrator.actions.executors.benchmark_run.launch import _BenchmarkLaunchMixin
    from hyperloom.orchestrator.state import base_launch
    from hyperloom.orchestrator.state.base_launch import ServerBase

    state = {**_RESTART_STATE, "nodes": 2}
    cli = _RestartCli(state)
    _route_pod_ssh(monkeypatch, cli)
    monkeypatch.setattr(mn_cli, "cmd_restart_server", pd_cmds.restart_pd_cluster)
    for module in (life, _multi_node_env):
        monkeypatch.setattr(module, "_read_state", lambda: state)
        monkeypatch.setattr(module, "is_multi_node", lambda: True)
    monkeypatch.setattr(ext, "external_has_server_control", lambda: True)
    session_base = ServerBase(args="--mem-fraction-static 0.8 --chunked-prefill-size 196608")
    monkeypatch.setattr(base_launch, "session_server_base", lambda: session_base)

    async def _ready(*_args: object, **_kwargs: object) -> None:
        return None

    for wait in (
        "_wait_for_workers_ready_async",
        "_wait_for_server_health_async",
        "_wait_for_published_service_ready_async",
    ):
        monkeypatch.setattr(life, wait, _ready)
    for name, value in {
        "FRAMEWORK": "sglang",
        "TP": "8",
        "PD_MODE": "disaggregated",
        "PD_PREFILL_EXTRA_ARGS": "--chunked-prefill-size 262144",
        "PD_DECODE_EXTRA_ARGS": "--mem-fraction-static 0.85 --chunked-prefill-size 262144",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SERVER_ARGS", raising=False)

    round_ = types.SimpleNamespace(ctx_extra={}, output_dir=tmp_path)
    restarted = _BenchmarkLaunchMixin._restart_multi_node_server(
        None, round_, server_args="--chunked-prefill-size 4096", resolved_model="/models/m"
    )
    assert asyncio.run(restarted) is None

    assert cli.saved is not None
    for role, mem_fraction in (("prefill", []), ("decode", ["--mem-fraction-static", "0.85"])):
        words = cli.saved[f"{role}_pods"][0]["launch_command"].split()
        tail = words[words.index("9098" if role == "decode" else "8998") + 1 :]
        assert tail == [*mem_fraction, "--chunked-prefill-size", "4096"]
