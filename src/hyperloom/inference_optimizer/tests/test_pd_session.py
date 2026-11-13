# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD sessions on a node-list cluster: intake, the frozen launch, and per-round restarts."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import subprocess
import types
from pathlib import Path

import httpx
import pytest

from inference_suite.contracts.base_launch import LaunchQuery, Workload
from inference_suite.multinode import RECORD_NAME, Cluster, SshClient, read_record, write_record
from inference_suite.multinode.ssh import SshResult
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.inference_optimizer.cli import multi_node as cli_mn
from hyperloom.inference_optimizer.multi_node import cli as mn_cli
from hyperloom.inference_optimizer.multi_node import pd_session
from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle as lifecycle

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


def test_adoption_records_the_frozen_launch_and_its_digest(session: Path) -> None:
    state = _adopt(session)
    launch = read_record(session / RECORD_NAME)
    assert state[pd_session.PD_LAUNCH_KEY] == str(session / RECORD_NAME)
    assert state[pd_session.PD_DIGEST_KEY] == launch.digest()


def test_resume_refuses_a_launch_that_is_not_the_one_frozen(session: Path) -> None:
    _adopt(session)
    record = session / RECORD_NAME
    launch = read_record(record)
    write_record(dataclasses.replace(launch, context_length=4096), record)
    with pytest.raises(SystemExit) as raised:
        cli_mn._prepare_multi_node_state(_args(), session)
    assert raised.value.code == 2


def test_round_overrides_share_the_round_and_keep_each_roles_own() -> None:
    overrides = pd_session.round_overrides("--chunked-prefill-size 4096", "--chunked-prefill-size 8192", "")
    assert overrides == {
        "prefill": ["--chunked-prefill-size", "8192"],
        "decode": ["--chunked-prefill-size", "4096"],
    }
    assert pd_session.round_overrides("") == {}


def _fake_restart(seen: dict):
    from inference_suite.multinode.launch import with_overrides

    def restart(launch, overrides, cluster, options):
        seen.update(overrides=overrides, env=options.extra_env)
        return with_overrides(launch, overrides)

    return restart


def test_a_round_restarts_over_the_frozen_launch_and_decode_keeps_its_mem_fraction(session: Path, monkeypatch) -> None:
    _adopt(session)
    seen: dict = {}
    monkeypatch.setattr(pd_session, "restart", _fake_restart(seen))
    asyncio.run(lifecycle.restart_server_for_round(lifecycle.RoundServer("--chunked-prefill-size 4096")))
    ran = read_record(session / pd_session.RAN_RECORD_NAME)
    assert seen["overrides"]["decode"] == ["--chunked-prefill-size", "4096"]
    assert "0.85" in ran.decode.args and "0.75" not in ran.decode.args
    assert "4096" in ran.prefill.args and "0.7" in ran.prefill.args
    assert read_record(session / RECORD_NAME).digest() == ext.load_multi_node_state()[pd_session.PD_DIGEST_KEY]


def test_a_failed_restart_is_a_failed_round(session: Path, monkeypatch) -> None:
    _adopt(session)

    def refuse(*_args, **_kwargs):
        raise pd_session.LaunchError("decode engine on d0 exited")

    monkeypatch.setattr(pd_session, "restart", refuse)
    with pytest.raises(lifecycle.ServerRestartFailed, match="d0 exited"):
        asyncio.run(lifecycle.restart_server_for_round(lifecycle.RoundServer("")))


def test_kill_inference_stops_the_cluster(session: Path, monkeypatch) -> None:
    _adopt(session)
    stopped = []
    monkeypatch.setattr(pd_session, "stop", lambda launch, cluster, options: stopped.append(launch.decode.hosts))
    assert mn_cli.cmd_kill_inference(argparse.Namespace(pid_file=None)) == 0
    assert stopped == [("d0",)]


def test_a_gone_engine_is_named(session: Path, monkeypatch) -> None:
    _adopt(session)
    gone = [types.SimpleNamespace(role="decode", host="d0")]
    monkeypatch.setattr(pd_session, "engines_gone", lambda launch, cluster: gone)
    assert lifecycle.pd_engines_lost_reason() == "decode engine on d0 is gone"


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


def _record_posts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Answer every engine POST with 200 and return what was posted."""
    seen: list[tuple[str, dict]] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content or b"{}")))
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    return seen


def test_pd_engine_profile_reaches_every_engine_of_the_frozen_launch(
    session: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _adopt(session)
    nodes = {**_NODES, "prefill_url": "http://pre:8010", "decode_url": "http://dec:8020"}
    monkeypatch.setenv(pd_session.NODE_LIST_ENV, json.dumps(nodes))
    monkeypatch.delenv("HYPERLOOM_MN_PROFILE_WITH_STACK", raising=False)
    seen = _record_posts(monkeypatch)
    out = tmp_path / "round"

    asyncio.run(lifecycle.trigger_pd_engine_profile("start", {"output_dir": str(out)}))
    asyncio.run(lifecycle.trigger_pd_engine_profile("stop"))

    assert seen == [
        ("http://p0:8010/start_profile", {"output_dir": str(out / "prefill"), "with_stack": False}),
        ("http://d0:8020/start_profile", {"output_dir": str(out / "decode"), "with_stack": False}),
        ("http://p0:8010/stop_profile", {}),
        ("http://d0:8020/stop_profile", {}),
    ]
    assert (out / "prefill").is_dir() and (out / "decode").is_dir()


def test_pd_engine_profile_with_stack_follows_env(session: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _adopt(session)
    monkeypatch.setenv("HYPERLOOM_MN_PROFILE_WITH_STACK", "1")
    seen = _record_posts(monkeypatch)
    asyncio.run(lifecycle.trigger_pd_engine_profile("start"))
    assert seen == [
        ("http://p0:8000/start_profile", {"with_stack": True}),
        ("http://d0:8000/start_profile", {"with_stack": True}),
    ]


def test_pd_engine_profile_leaves_other_backends_alone(session: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "_read_state", lambda: {"backend": "infera"})
    seen = _record_posts(monkeypatch)
    asyncio.run(lifecycle.trigger_pd_engine_profile("start"))
    assert seen == []
