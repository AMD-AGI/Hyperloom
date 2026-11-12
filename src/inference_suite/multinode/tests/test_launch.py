# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
from dataclasses import replace

import pytest

from inference_suite.frameworks import Framework
from inference_suite.multinode.commands import HostCommand
from inference_suite.multinode.launch import (
    Cluster,
    LaunchError,
    LaunchOptions,
    check_env,
    fit_to_engine,
    launch_script,
    replace_flags,
    restart,
    start,
    stop,
    with_overrides,
)
from inference_suite.multinode.nodes import SshAccess, parse_node_list
from inference_suite.multinode.record import Frontend, FrontendType, PdLaunch, PdSource, PdSourceKind, RoleLaunch
from inference_suite.multinode.ssh import SshClient, SshResult

_FLAGS = "--model-path\n--mem-fraction-static\n--cuda-graph-bs-decode\n--chunked-prefill-size\n"


class FakeHosts:
    """SSH runner that answers each script kind and records what every host ran."""

    def __init__(self, gone=(), driver_rc=0, host_key_failures=0):
        self.calls = []
        self.gone = set(gone)
        self.driver_rc = driver_rc
        self.host_key_failures = host_key_failures

    def __call__(self, argv, stdin, timeout):
        if argv[0] == "ssh-keyscan":
            self.calls.append(("keyscan", argv[-1], ""))
            return SshResult(0, f"{argv[-1]} ssh-ed25519 AAAA\n", "")
        host = argv[-3].split("@", 1)[1]
        self.calls.append(("ssh", host, stdin))
        if self.host_key_failures:
            self.host_key_failures -= 1
            return SshResult(255, "", "Host key verification failed.")
        if "launch_server --help" in stdin:
            return SshResult(0, _FLAGS, "")
        if "HL_ACTION=" in stdin:
            return SshResult(self.driver_rc, '{"status": "ok", "action": "spawn", "pid": 7}\n', "denied")
        if "HL_ALIVE" in stdin:
            return SshResult(0, "HL_GONE\n" if host in self.gone else "HL_ALIVE\n", "")
        if stdin.lstrip().startswith("set -uo pipefail\ntail"):
            return SshResult(0, "RuntimeError: out of memory\n", "")
        return SshResult(0, "", "")

    def driven(self, action):
        return [host for kind, host, stdin in self.calls if kind == "ssh" and f"HL_ACTION={action}" in stdin]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _nodes(**fields):
    document = {
        "framework": "sglang",
        "model": "/models/M",
        "service_url": "http://svc:8000",
        "router": {"host": "r0"},
        "prefill": [{"host": "p0"}],
        "decode": [{"host": "d0"}],
        "gpus_per_node": 8,
        "ssh": {"port": 22, "user": "root", "key_file": "/keys/id"},
        "launch_mode": "in_place",
    }
    document.update(fields)
    return parse_node_list(document)


def _launch(kind=PdSourceKind.INFERENCEX_MASTER):
    return PdLaunch(
        source=PdSource(kind),
        framework=Framework.SGLANG,
        model="/models/M",
        served_model_name="org/M",
        context_length=2048,
        transfer_backend="mori",
        prefill=RoleLaunch(("p0",), 1, 8, args=("--mem-fraction-static", "0.8", "--cuda-graph-bs", "1", "2")),
        decode=RoleLaunch(("d0",), 1, 8, args=("--cuda-graph-bs", "1", "2", "--retired-flag", "x")),
        frontend=Frontend(FrontendType.SGLANG_ROUTER, "r0", ()),
    )


def _cluster(tmp_path, hosts, healthy=lambda url: True):
    clock = Clock()
    client = SshClient(SshAccess(22, "root", "/keys/id"), tmp_path / "known_hosts", runner=hosts)
    return Cluster(_nodes(), client, probe=healthy, sleep=clock.sleep, clock=clock)


_OPTIONS = LaunchOptions(ready_timeout_s=60, poll_interval_s=10)


def test_start_runs_engines_before_the_router(tmp_path):
    hosts = FakeHosts()
    probed = []
    ran = start(_launch(), _cluster(tmp_path, hosts, lambda url: probed.append(url) or True), _OPTIONS)
    assert hosts.driven("restart") == ["p0", "d0", "r0"]
    assert probed == [
        "http://p0:8000/health",
        "http://d0:8000/health",
        "http://r0:8000/health",
        "http://svc:8000/health",
    ]
    assert ran.decode.args == ("--cuda-graph-bs-decode", "1", "2")
    assert ran.decode.stripped == ("unsupported --retired-flag",)
    assert ran.prefill.args == ("--mem-fraction-static", "0.8")
    assert ran.prefill.stripped == ("unsupported --cuda-graph-bs",)


def test_operator_flags_are_kept_when_the_engine_does_not_list_them():
    fitted = fit_to_engine(_launch(PdSourceKind.OPERATOR), frozenset({"cuda-graph-bs-decode"}))
    assert "--retired-flag" in fitted.decode.args


def test_dead_engine_fails_the_start_with_its_log(tmp_path):
    hosts = FakeHosts(gone={"d0"})
    with pytest.raises(LaunchError, match="decode on d0 exited during startup:\nRuntimeError: out of memory"):
        start(_launch(), _cluster(tmp_path, hosts, lambda url: "d0" not in url), _OPTIONS)
    assert "r0" not in hosts.driven("restart")


def test_engines_that_never_answer_time_out(tmp_path):
    with pytest.raises(LaunchError, match="not ready after 60s"):
        start(_launch(), _cluster(tmp_path, FakeHosts(), lambda url: "p0" not in url), _OPTIONS)


def test_a_host_that_refuses_fails_the_start(tmp_path):
    with pytest.raises(LaunchError, match="restart of prefill on p0 failed: denied"):
        start(_launch(), _cluster(tmp_path, FakeHosts(driver_rc=1)), _OPTIONS)


def test_changed_host_key_is_rescanned_once(tmp_path):
    hosts = FakeHosts(host_key_failures=1)
    start(_launch(), _cluster(tmp_path, hosts), _OPTIONS)
    assert hosts.calls[1][:2] == ("keyscan", "p0")
    assert (tmp_path / "known_hosts").read_text() == "p0 ssh-ed25519 AAAA\n"


def test_stop_parks_the_router_first(tmp_path):
    hosts = FakeHosts()
    stop(_launch(), _cluster(tmp_path, hosts), _OPTIONS)
    assert hosts.driven("stop") == ["r0", "p0", "d0"]


def test_restart_replaces_same_named_flags(tmp_path):
    hosts = FakeHosts()
    ran = restart(_launch(), {"prefill": ["--mem-fraction-static=0.7"]}, _cluster(tmp_path, hosts), _OPTIONS)
    assert ran.prefill.args == ("--mem-fraction-static=0.7",)


def test_overrides_cannot_set_launcher_flags():
    with pytest.raises(LaunchError, match="the launcher sets"):
        with_overrides(_launch(), {"decode": ["--port", "1"]})
    with pytest.raises(LaunchError, match="no role"):
        with_overrides(_launch(), {"worker": ["--a"]})


def test_replace_flags_keeps_the_others():
    assert replace_flags(("--a", "1", "--b", "2", "3"), ("--b", "4")) == ("--a", "1", "--b", "4")


def test_env_keys_must_be_shell_names():
    assert check_env({"SGLANG_TORCH_PROFILER_DIR": "/t"}) == {"SGLANG_TORCH_PROFILER_DIR": "/t"}
    for key in ("A-B", "HL_PIDFILE", "X;rm"):
        with pytest.raises(LaunchError):
            check_env({key: "v"})


def test_engine_script_waits_for_released_gpus_and_runs_under_bash():
    command = HostCommand("decode", "d0", 0, 0, "echo $HL_IB_DEVICE")
    script = launch_script(command, {"K": "a b"}, Framework.SGLANG, "/logs")
    assert "export K='a b'" in script and "hl_gpu_holders" in script and "/logs/pd_decode_d0.log" in script
    assert script.rstrip().endswith("exec echo $HL_IB_DEVICE")
    environ = "/nonexistent"
    probe = script.split("for hl_fd")[0].replace("/proc/1/environ", environ)
    done = subprocess.run(
        ["bash", "-c", probe + 'printf "%s" "$HL_IB_DEVICE"'],
        env={"PATH": "/usr/bin:/bin", "NCCL_IB_HCA": "bnxt_re0:1,bnxt_re1:1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == "bnxt_re0,bnxt_re1"


def test_router_script_skips_engine_steps():
    script = launch_script(HostCommand("router", "r0", 0, 0, "atomesh launch"), {}, Framework.SGLANG, "/logs")
    assert "hl_gpu_holders" not in script and "HL_IB_DEVICE" not in script


def test_atom_launch_does_not_probe_sglang(tmp_path):
    hosts = FakeHosts()
    launch = replace(_launch(), framework=Framework.ATOM, frontend=Frontend(FrontendType.ATOMESH, "r0", ()))
    cluster = replace(_cluster(tmp_path, hosts), nodes=_nodes(framework="atom"))
    ran = start(launch, cluster, _OPTIONS)
    assert ran.decode.args == launch.decode.args
    assert not any("launch_server --help" in stdin for _kind, _host, stdin in hosts.calls)
