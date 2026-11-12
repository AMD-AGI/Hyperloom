# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PD sessions on a node-list cluster: intake, the frozen launch, and per-round restarts."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import types
from pathlib import Path

import pytest

from inference_suite.contracts.base_launch import LaunchQuery, Workload
from inference_suite.multinode import RECORD_NAME, read_record, write_record
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


@pytest.fixture()
def session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A PD node-list hand-off and an open session directory."""
    for name in ("HYPERLOOM_MN_EXT_SERVICE_URL", "TOPOLOGY_HANDOFF_FILE", "INFERENCE_OPTIMIZER_MN_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(pd_session.NODE_LIST_ENV, json.dumps(_NODES))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setenv("PD_MODE", "disaggregated")
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
