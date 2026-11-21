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
)
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import LaunchSpec
from hyperloom.inference_optimizer.multi_node.commands import pd_cluster as pd_cmds
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


def test_engine_launch_defaults_the_disaggregation_timeouts_without_overriding_a_set_one() -> None:
    script = launch_script("true", {"SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "5"})
    assert 'SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-100000}"' in script
    assert "export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5" in script
    assert "SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-" not in script


def test_mooncake_knobs_are_forwarded_to_the_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.multi_node.commands import infera

    monkeypatch.setenv("MC_ENABLE_DEST_DEVICE_AFFINITY", "1")
    assert infera._collect_forward_env(infera.POD_FORWARD_ENV_PREFIXES)["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "1"


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
    assert [pod["role"] for pod in pd_cmds._engine_pods(state)] == ["prefill", "decode"]


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


_PD_STATE = {
    "backend": "sglangrouter",
    "ssh_port": 29000,
    "prefill_pods": [{"role": "prefill", "podIP": "10.0.0.1", "sshPort": 29000}],
    "decode_pods": [{"role": "decode", "podIP": "10.0.0.2", "sshPort": 29000}],
}


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
