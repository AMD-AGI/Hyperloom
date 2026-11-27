# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import json
from dataclasses import replace

import pytest

from inference_suite.contracts.base_launch import LaunchQuery, Workload
from inference_suite.multinode import cli
from inference_suite.multinode import launch as launch_module
from inference_suite.multinode.launch import LaunchError
from inference_suite.multinode.nodes import NODE_LIST_ENV
from inference_suite.multinode.record import (
    RAN_RECORD_NAME,
    Frontend,
    FrontendType,
    Role,
    RoleLaunch,
    read_record,
    write_record,
)
from inference_suite.multinode.tests.builders import SSH, node_document, pd_launch

_NODES = node_document(ssh=SSH)
_LAUNCH = pd_launch(
    {Role.PREFILL: RoleLaunch(("old-p",), 1, 8, args=("--a", "1")), Role.DECODE: RoleLaunch(("old-d",), 1, 8)},
    frontend=Frontend(FrontendType.SGLANG_ROUTER, "old-r", ()),
)


def main(argv):
    parser = argparse.ArgumentParser()
    cli.add_commands(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return args.handler(args)


@pytest.fixture
def record(tmp_path, monkeypatch):
    monkeypatch.setenv(NODE_LIST_ENV, json.dumps(_NODES))
    path = tmp_path / "pd_launch.json"
    write_record(_LAUNCH, path)
    return path


def test_resolve_replays_a_record(record, tmp_path, capsys):
    query = tmp_path / "query.json"
    session = LaunchQuery("org/M", "sglang", "mi355x", Workload.FIXED, concurrencies=(64,), isl=1024, osl=1024)
    query.write_text(json.dumps(session.to_document()))
    out = tmp_path / "again.json"
    assert main(["multinode", "resolve", "--query", str(query), "--replay", str(record), "--out", str(out)]) == 0
    replayed = read_record(out)
    assert capsys.readouterr().out.strip() == replayed.digest() == _LAUNCH.digest()
    assert replayed.role(Role.PREFILL).hosts == ("p0",)


def test_start_runs_on_the_node_list_hosts_and_records_what_ran(record, monkeypatch, capsys):
    seen = {}

    def fake_start(launch, cluster, options):
        seen.update(launch=launch, known=cluster.client.known_hosts, env=options.extra_env)
        return launch.with_roles({Role.PREFILL: replace(launch.role(Role.PREFILL), stripped=("unsupported --x",))})

    monkeypatch.setattr(cli, "start", fake_start)
    assert main(["multinode", "start", "--record", str(record), "--env", "A=b=c"]) == 0
    assert seen["launch"].role(Role.PREFILL).hosts == ("p0",) and seen["launch"].frontend.host == "r0"
    assert seen["known"] == record.parent / "known_hosts" and seen["env"] == {"A": "b=c"}
    ran = read_record(record.with_name(RAN_RECORD_NAME))
    assert ran.role(Role.PREFILL).stripped == ("unsupported --x",)
    assert capsys.readouterr().out.strip() == ran.digest()


def test_restart_passes_role_overrides_and_writes_what_ran_beside_the_frozen_record(record, monkeypatch):
    frozen = record.read_bytes()
    seen = {}

    def fake_restart(launch, overrides, cluster, options):
        seen.update(o=overrides)
        return launch.with_roles({Role.DECODE: replace(launch.role(Role.DECODE), args=("--b", "2 3"))})

    monkeypatch.setattr(cli, "restart", fake_restart)
    assert main(["multinode", "restart", "--record", str(record), "--decode-args", "--b '2 3'"]) == 0
    assert seen["o"] == {"decode": ["--b", "2 3"]}
    assert record.read_bytes() == frozen
    assert read_record(record.with_name(RAN_RECORD_NAME)).role(Role.DECODE).args == ("--b", "2 3")


def test_a_cluster_that_does_not_come_up_exits_1(record, monkeypatch):
    def fail(*_args):
        raise LaunchError("not ready")

    monkeypatch.setattr(cli, "stop", fail)
    assert main(["multinode", "stop", "--record", str(record)]) == 1


def test_node_list_without_ssh_is_refused(record, monkeypatch, capsys):
    monkeypatch.setenv(NODE_LIST_ENV, json.dumps({key: value for key, value in _NODES.items() if key != "ssh"}))
    assert main(["multinode", "start", "--record", str(record)]) == 2
    assert "SSH" in capsys.readouterr().err


def test_record_defaults_to_the_session_dir(tmp_path):
    assert cli.default_record({cli.SESSION_DIR_ENV: str(tmp_path)}) == tmp_path / "pd_launch.json"


@pytest.mark.parametrize(
    ("flag", "value", "reason"),
    [("--decode-args", "--port 9000", "launcher sets"), ("--worker-args", "--x 1", "overrides name no role")],
)
def test_an_override_refused_before_any_host_is_contacted_exits_2(record, monkeypatch, flag, value, reason, capsys):
    def contacted(*_args):
        raise AssertionError("a host was contacted")

    monkeypatch.setattr(launch_module, "probe_engine_flags", contacted)
    assert main(["multinode", "restart", "--record", str(record), flag, value]) == 2
    assert reason in capsys.readouterr().err
