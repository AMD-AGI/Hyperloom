# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace

import pytest

from inference_suite.cli import main
from inference_suite.contracts.base_launch import LaunchQuery, Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode import cli
from inference_suite.multinode.launch import LaunchError
from inference_suite.multinode.nodes import NODE_LIST_ENV
from inference_suite.multinode.record import (
    Frontend,
    FrontendType,
    PdLaunch,
    PdSource,
    PdSourceKind,
    RoleLaunch,
    read_record,
    write_record,
)

_NODES = {
    "framework": "sglang",
    "model": "/models/M",
    "service_url": "http://svc:8000",
    "router": {"host": "r0"},
    "prefill": [{"host": "p0"}],
    "decode": [{"host": "d0"}],
    "gpus_per_node": 8,
    "ssh": {"port": 22, "user": "root", "key_file": "/keys/id"},
}
_LAUNCH = PdLaunch(
    source=PdSource(PdSourceKind.PD_TABLE),
    framework=Framework.SGLANG,
    model="/models/M",
    served_model_name="org/M",
    context_length=2048,
    transfer_backend="mori",
    prefill=RoleLaunch(("old-p",), 1, 8, args=("--a", "1")),
    decode=RoleLaunch(("old-d",), 1, 8),
    frontend=Frontend(FrontendType.SGLANG_ROUTER, "old-r", ()),
)


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
    assert replayed.prefill.hosts == ("p0",)


def test_start_runs_on_the_node_list_hosts_and_records_what_ran(record, monkeypatch, capsys):
    seen = {}

    def fake_start(launch, cluster, options):
        seen.update(launch=launch, known=cluster.client.known_hosts, env=options.extra_env)
        return replace(launch, prefill=replace(launch.prefill, stripped=("unsupported --x",)))

    monkeypatch.setattr(cli, "start", fake_start)
    assert main(["multinode", "start", "--record", str(record), "--env", "A=b=c"]) == 0
    assert seen["launch"].prefill.hosts == ("p0",) and seen["launch"].frontend.host == "r0"
    assert seen["known"] == record.parent / "known_hosts" and seen["env"] == {"A": "b=c"}
    assert read_record(record).prefill.stripped == ("unsupported --x",)
    assert capsys.readouterr().out.strip() == read_record(record).digest()


def test_restart_passes_role_overrides(record, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "restart", lambda launch, overrides, cluster, options: seen.update(o=overrides) or launch)
    assert main(["multinode", "restart", "--record", str(record), "--decode-args", "--b '2 3'"]) == 0
    assert seen["o"] == {"decode": ["--b", "2 3"]}


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
