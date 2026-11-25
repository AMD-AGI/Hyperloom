# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD sessions on a node-list cluster: intake, the frozen launch, and per-round restarts."""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from inference_suite.contracts.base_launch import LaunchQuery, Workload
from inference_suite.multinode import RECORD_NAME, Cluster, PdMode, Role, SshClient, read_record, write_record
from inference_suite.multinode.commands import CommandError
from inference_suite.multinode.ssh import SshResult
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.inference_optimizer.cli import multi_node as cli_mn
from hyperloom.inference_optimizer.multi_node import TopologyHandoffError
from hyperloom.inference_optimizer.multi_node import cli as mn_cli
from hyperloom.inference_optimizer.multi_node import pd_session
from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.session.base_launch import read_base_launch

_NODES = {
    "framework": "sglang",
    "model": "/models/M",
    "service_url": "http://svc:8000",
    "router": {"host": "r0"},
    "prefill": [{"host": "p0"}],
    "decode": [{"host": "d0"}],
    "gpus_per_node": 8,
    "ssh": {"port": 22, "user": "root", "key_file": "/keys/id"},
    "frontend": "sglang-router",
    "config": {
        "prefill_args": "--tp-size 8 --mem-fraction-static 0.7",
        "decode_args": "--tp-size 8 --mem-fraction-static 0.85",
    },
}
_SESSION = LaunchQuery("org/M", "sglang", "mi355x", Workload.FIXED, concurrencies=(64,), isl=1024, osl=1024)


def _run_here(_argv, stdin: str, timeout: int) -> SshResult:
    """Run an SSH payload locally, as a host sharing this filesystem would."""
    done = subprocess.run(["bash", "-s"], input=stdin, capture_output=True, text=True, timeout=timeout, check=False)
    return SshResult(done.returncode, done.stdout, done.stderr)


def _ssh_through(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    """Route the session cluster's SSH through ``runner``."""

    def cluster_of(nodes, record):
        return Cluster(nodes=nodes, client=SshClient(nodes.ssh, record.parent / "known_hosts", runner=runner))

    monkeypatch.setattr(pd_session, "cluster_of", cluster_of)


@pytest.fixture()
def session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A PD node-list hand-off whose hosts share this filesystem, and an open session directory."""
    for name in ("HYPERLOOM_MN_EXT_SERVICE_URL", "TOPOLOGY_HANDOFF_FILE", "INFERENCE_OPTIMIZER_MN_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(pd_session.NODE_LIST_ENV, json.dumps(_NODES))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("PD_MODE", "disaggregated")
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_TRACE_DIR", str(tmp_path / "traces"))
    _ssh_through(monkeypatch, _run_here)
    directory = tmp_path / "session"
    directory.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(directory))
    return directory


def _args() -> argparse.Namespace:
    return argparse.Namespace(nodes=2, mn_backend=None, no_kernel=True)


def _adopt(session: Path) -> dict:
    pd_session.freeze_launch(session, _SESSION, pd_session.handed_node_list())
    cli_mn._prepare_multi_node_state(_args(), session)
    return ext.load_multi_node_state()


def test_the_node_list_is_the_hand_off(session: Path) -> None:
    state = ext.build_external_state_from_env()
    assert ext.external_service_url() == "http://svc:8000"
    assert ext.external_has_ssh_control()
    assert state["backend"] == "sglangrouter"
    assert (state["router_pod_ip"], state["prefill_pod_ips"], state["decode_pod_ips"]) == ("r0", ["p0"], ["d0"])
    assert state["nodes"] == 2
    assert cli_mn._resolve_mn_backend(_args()) == "sglangrouter"


@pytest.mark.parametrize("user", ["root", None, "ubuntu"])
def test_a_node_list_must_hand_over_root_ssh(session: Path, monkeypatch, user) -> None:
    ssh = {key: value for key, value in _NODES["ssh"].items() if key != "user"}
    if user is not None:
        ssh["user"] = user
    monkeypatch.setenv(pd_session.NODE_LIST_ENV, json.dumps({**_NODES, "ssh": ssh}))
    if user == "ubuntu":
        with pytest.raises(TopologyHandoffError, match="ubuntu.*root"):
            pd_session.handed_node_list()
    else:
        assert pd_session.handed_node_list().ssh.user == "root"


def test_a_pd_session_freezes_its_launch_and_no_single_node_base(session: Path, monkeypatch) -> None:
    frozen = []
    monkeypatch.setattr(bootstrap, "server_session_query", lambda *a, **k: (_SESSION, ""))
    monkeypatch.setattr(cli_mn, "freeze_pd_session_launch", lambda *a, **k: frozen.append(a))
    monkeypatch.setattr(bootstrap, "resolve_base_launch", lambda *a, **k: pytest.fail("resolved a single-node base"))
    args = argparse.Namespace(nodes=2, mn_backend=None, conc=64, enable_conc_sweep=False)
    assert bootstrap.resolve_session_base(session, args, framework="sglang") is None
    assert frozen[0][2] is _SESSION


def test_a_single_node_session_still_resolves_its_base(session: Path, monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "resolve_base_launch", lambda *a, **k: "base")
    args = argparse.Namespace(nodes=1, mn_backend=None, conc=64, enable_conc_sweep=False)
    assert bootstrap.resolve_session_base(session, args, framework="sglang") == "base"


def test_a_frozen_pd_session_gives_its_rounds_a_client_base(session: Path, monkeypatch) -> None:
    for name in ("TP", "EP"):
        monkeypatch.setenv(name, "1")
    base = cli_mn.freeze_pd_session_launch(session, _args(), _SESSION)
    assert read_base_launch(session) == base
    assert (base.framework, base.topology.tp, base.args, dict(base.env)) == ("sglang", 8, (), {})
    assert base.client is not None


def test_a_node_list_session_has_server_control_whatever_backend_it_names(session: Path, monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "sglangrouter")
    assert ext.external_has_server_control()


def test_adoption_records_the_frozen_launch_and_its_digest(session: Path) -> None:
    state = _adopt(session)
    launch = read_record(session / RECORD_NAME)
    assert state[pd_session.PD_LAUNCH_KEY] == str(session / RECORD_NAME)
    assert state[pd_session.PD_DIGEST_KEY] == launch.digest()


def test_adoption_gives_every_host_a_pod_with_the_command_it_serves(session: Path) -> None:
    state = _adopt(session)
    pods = {pod["role"]: pod for pod in state["pods"]}
    assert sorted(pods) == ["decode", "prefill", "router"]
    assert {role: pod["podIP"] for role, pod in pods.items()} == {"router": "r0", "prefill": "p0", "decode": "d0"}
    assert all(pod["sshPort"] == 22 for pod in pods.values())
    for role in ("prefill", "decode"):
        assert "sglang.launch_server" in pods[role]["launch_command"]
        assert f"--disaggregation-mode {role}" in pods[role]["launch_command"]
    assert "sglang_router.launch_router" in pods["router"]["launch_command"]
    assert "--port 8000" in pods["router"]["launch_command"]
    assert state["user_data_path"]


def test_resume_refuses_a_launch_that_is_not_the_one_frozen(session: Path) -> None:
    _adopt(session)
    record = session / RECORD_NAME
    launch = read_record(record)
    write_record(dataclasses.replace(launch, context_length=4096), record)
    with pytest.raises(SystemExit) as raised:
        cli_mn._prepare_multi_node_state(_args(), session)
    assert raised.value.code == 2


def test_round_overrides_share_the_round_and_keep_each_roles_own() -> None:
    mode = PdMode.DISAGGREGATED
    overrides = pd_session._round_overrides("--chunked-prefill-size 4096", "--chunked-prefill-size 8192", "", mode=mode)
    assert overrides == {
        "prefill": ["--chunked-prefill-size", "8192"],
        "decode": ["--chunked-prefill-size", "4096"],
    }
    assert pd_session._round_overrides("", "", "", mode=mode) == {}


def _fake_restart(seen: dict):
    from inference_suite.multinode.launch import with_overrides

    def restart(launch, overrides, cluster, options):
        seen.update(overrides=overrides, env=options.extra_env)
        return with_overrides(launch, overrides)

    return restart


def test_a_malformed_round_or_command_is_a_session_error(session: Path, monkeypatch) -> None:
    state = _adopt(session)

    def refuse(*_args, **_kwargs):
        raise CommandError("router r0 serves on 8000")

    monkeypatch.setattr(pd_session, "restart", refuse)
    monkeypatch.setattr(pd_session, "stop", refuse)
    with pytest.raises(pd_session.PdSessionError, match="r0"):
        pd_session.restart_round(state, "", pd_session.LaunchOptions())
    with pytest.raises(pd_session.PdSessionError, match="r0"):
        pd_session.stop_cluster(state)
    monkeypatch.setenv(pd_session.PREFILL_EXTRA_ENV, "--x 'unbalanced")
    with pytest.raises(pd_session.PdSessionError):
        pd_session.restart_round(state, "", pd_session.LaunchOptions())


def test_kill_inference_stops_the_cluster(session: Path, monkeypatch) -> None:
    _adopt(session)
    stopped = []
    monkeypatch.setattr(
        pd_session, "stop", lambda launch, cluster, options: stopped.append(launch.role(Role.DECODE).hosts)
    )
    assert mn_cli.cmd_kill_inference(argparse.Namespace(pid_file=None)) == 0
    assert stopped == [("d0",)]


_INFERA_NODES = {
    "framework": "sglang",
    "model": "/models/M",
    "service_url": "http://frontend:30080",
    "mode": "aggregated",
    "worker": [{"host": "w0", "ssh_port": 2240}, {"host": "w1", "ssh_port": 2241}],
    "gpus_per_node": 8,
    "ssh": {"port": 2240, "user": "root", "key_file": "/keys/id"},
    "frontend": "infera",
    "topology": {"worker": {"tp": 16}},
    "config": {"worker_args": "--mem-fraction-static 0.8"},
}


@pytest.fixture()
def infera_session(session: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An aggregated Infera node-list hand-off."""
    monkeypatch.setenv(pd_session.NODE_LIST_ENV, json.dumps(_INFERA_NODES))
    monkeypatch.setenv("PD_MODE", "aggregated")
    return session


def test_an_infera_node_list_is_a_suite_session(infera_session: Path) -> None:
    state = _adopt(infera_session)
    assert state["backend"] == "infera" and pd_session.is_pd_state(state)
    assert (state["pd_mode"], state["worker_pod_ips"], state["prefill_pod_ips"]) == ("aggregated", ["w0", "w1"], [])
    assert [pod["sshPort"] for pod in state["worker_pods"]] == [2240, 2241]
    assert cli_mn.is_pd_session(_args())
    assert read_record(infera_session / RECORD_NAME).mode.value == "aggregated"


def test_an_infera_state_without_a_node_list_keeps_the_legacy_path() -> None:
    assert not pd_session.is_pd_state({"backend": "infera", "pd_mode": "aggregated"})


def test_an_infera_hand_off_without_a_node_list_is_no_suite_session(session: Path, monkeypatch) -> None:
    monkeypatch.delenv(pd_session.NODE_LIST_ENV)
    assert not cli_mn.is_pd_session(argparse.Namespace(nodes=2, mn_backend="infera"))


def test_the_session_mode_must_be_the_node_lists(infera_session: Path, monkeypatch) -> None:
    monkeypatch.setenv("PD_MODE", "disaggregated")
    with pytest.raises(SystemExit) as raised:
        cli_mn.freeze_pd_session_launch(infera_session, _args(), _SESSION)
    assert raised.value.code == 2


def test_aggregated_rounds_override_the_workers() -> None:
    mode = PdMode.AGGREGATED
    assert pd_session._round_overrides("--chunked-prefill-size 4096", "--x", "", mode=mode) == {
        "worker": ["--chunked-prefill-size", "4096"]
    }
    assert pd_session._round_overrides("", "", "", mode=mode) == {}


def test_kill_inference_stops_the_infera_workers(infera_session: Path, monkeypatch) -> None:
    _adopt(infera_session)
    stopped = []
    monkeypatch.setattr(pd_session, "stop", lambda launch, cluster, options: stopped.append(launch.mode.value))
    assert mn_cli.cmd_kill_inference(argparse.Namespace(pid_file=None)) == 0
    assert stopped == ["aggregated"]


def test_adoption_records_that_the_engine_hosts_share_the_trace_root(session: Path, tmp_path: Path) -> None:
    state = _adopt(session)
    assert state[pd_session.TRACE_SHARED_KEY] is True
    assert pd_session.TRACE_SHARE_ERROR_KEY not in state
    assert list((tmp_path / "traces").iterdir()) == []


@pytest.mark.parametrize(
    "reply",
    [
        lambda _argv, _stdin, _timeout: SshResult(1, "", "cat: No such file or directory"),
        lambda _argv, _stdin, _timeout: SshResult(0, "other-token\n", ""),
        lambda _argv, _stdin, timeout: SshResult(124, "", f"timed out after {timeout}s"),
    ],
    ids=["missing", "different-file", "ssh-timeout"],
)
def test_adoption_records_why_the_trace_root_is_not_shared(
    session: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply
) -> None:
    _ssh_through(monkeypatch, reply)
    state = _adopt(session)
    assert state[pd_session.TRACE_SHARED_KEY] is False
    assert state[pd_session.TRACE_SHARE_ERROR_KEY].startswith("engine host p0:")
    assert list((tmp_path / "traces").iterdir()) == []


def test_a_profiling_round_gives_each_engine_role_its_own_profiler_dir() -> None:
    options = pd_session.launch_options({pd_session.PROFILER_DIR_ENV: "/t/", "SGLANG_X": "1"}, 60)
    assert options.extra_env == {"SGLANG_X": "1"}
    assert options.role_env == {
        "prefill": {pd_session.PROFILER_DIR_ENV: "/t/prefill"},
        "decode": {pd_session.PROFILER_DIR_ENV: "/t/decode"},
    }
    assert pd_session.launch_options({"SGLANG_X": "1"}, 60).role_env == {}


@pytest.mark.parametrize("backend", ["atomesh", "sglangrouter"])
def test_a_node_list_cluster_installs_geak_over_ssh(monkeypatch, backend):
    monkeypatch.setattr(mn_cli, "_load_state", lambda: {"backend": backend, "node_list": True})
    monkeypatch.setattr(mn_cli, "cmd_install_geak", lambda _args: 5)
    assert mn_cli.install_geak_on_pods_best_effort() == 5


@pytest.mark.parametrize(
    ("command", "infera_command"),
    [
        ("cmd_apply_patch", "_infera_apply_patch"),
        ("cmd_revert_patch", "_infera_revert_patch"),
        ("cmd_finalize_patch", "_infera_finalize_patch"),
        ("cmd_apply_tracelens_patch", "_infera_apply_tracelens_patch"),
    ],
)
@pytest.mark.parametrize("backend", ["atomesh", "sglangrouter"])
def test_a_node_list_cluster_takes_the_ssh_patch_path(monkeypatch, command, infera_command, backend):
    monkeypatch.setattr(mn_cli, "_load_state", lambda: {"backend": backend, "node_list": True})
    monkeypatch.setattr(mn_cli, infera_command, lambda _args: 7)
    assert getattr(mn_cli, command)(argparse.Namespace()) == 7
