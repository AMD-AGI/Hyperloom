# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import shlex
from dataclasses import replace

import pytest

from inference_suite.frameworks import Framework
from inference_suite.multinode.commands import CommandError, cluster_commands, engine_urls, router_command
from inference_suite.multinode.nodes import LaunchMode, parse_node_list
from inference_suite.multinode.record import Frontend, FrontendType, PdLaunch, PdSource, PdSourceKind, RoleLaunch


def _nodes(**fields):
    document = {
        "framework": "sglang",
        "model": "/models/M",
        "service_url": "http://svc:8000",
        "router": {"host": "r0"},
        "prefill": [{"host": "p0"}],
        "decode": [{"host": "d0"}, {"host": "d1"}],
        "gpus_per_node": 8,
        "launch_mode": "direct",
    }
    document.update(fields)
    return parse_node_list(document)


def _launch(framework=Framework.SGLANG, frontend=FrontendType.SGLANG_ROUTER, decode_tp=16, backend="mori"):
    return PdLaunch(
        source=PdSource(PdSourceKind.PD_TABLE),
        framework=framework,
        model="/models/M",
        served_model_name="org/M",
        context_length=2048,
        transfer_backend=backend,
        prefill=RoleLaunch(("p0",), 1, 8, args=("--mem-fraction-static", "0.8")),
        decode=RoleLaunch(("d0", "d1"), 1, decode_tp, args=("--a",)),
        frontend=Frontend(frontend, "r0", ("--policy", "random")),
    )


def _argv(command):
    return shlex.split(command.command)


def test_sglang_worker_spanning_hosts_joins_them():
    commands = cluster_commands(_launch(), _nodes())
    assert [(c.role, c.host, c.worker, c.rank) for c in commands] == [
        ("prefill", "p0", 0, 0),
        ("decode", "d0", 0, 0),
        ("decode", "d1", 0, 1),
        ("router", "r0", 0, 0),
    ]
    decode_rank1 = _argv(commands[2])
    assert decode_rank1[decode_rank1.index("--dist-init-addr") + 1] == "d0:20000"
    assert decode_rank1[decode_rank1.index("--node-rank") + 1] == "1"
    assert decode_rank1[decode_rank1.index("--tp-size") + 1] == "16"
    assert commands[2].command.endswith(' --a ${HL_IB_DEVICE:+--disaggregation-ib-device "$HL_IB_DEVICE"}')
    prefill = _argv(commands[0])
    assert "--dist-init-addr" not in prefill
    assert prefill[prefill.index("--context-length") + 1] == "2048"
    assert prefill[prefill.index("--served-model-name") + 1] == "org/M"
    assert prefill[prefill.index("--disaggregation-transfer-backend") + 1] == "mori"


def test_sglang_router_reaches_worker_heads_in_direct_mode():
    argv = _argv(router_command(_launch(), _nodes()))
    assert argv[:4] == ["python3", "-m", "sglang_router.launch_router", "--pd-disaggregation"]
    assert argv[argv.index("--prefill") : argv.index("--prefill") + 3] == ["--prefill", "http://p0:8000", "8998"]
    assert argv[argv.index("--decode") : argv.index("--decode") + 2] == ["--decode", "http://d0:8000"]
    assert argv[-2:] == ["--policy", "random"]


def test_in_place_router_reaches_role_services():
    nodes = _nodes(launch_mode="in_place", prefill_url="http://pre:8010", decode_url="http://dec:8020")
    assert nodes.launch_mode is LaunchMode.IN_PLACE
    commands = cluster_commands(_launch(), nodes)
    argv = _argv(commands[-1])
    assert "http://pre:8010" in argv and "http://dec:8020" in argv
    assert _argv(commands[0])[_argv(commands[0]).index("--port") + 1] == "8010"


def test_engine_urls_name_each_worker_head_on_its_role_port():
    nodes = _nodes(launch_mode="in_place", prefill_url="http://pre:8010", decode_url="http://dec:8020")
    assert engine_urls(_launch(), nodes) == {"prefill": ["http://p0:8010"], "decode": ["http://d0:8020"]}
    two_workers = replace(_launch(), decode=RoleLaunch(("d0", "d1"), 2, 8))
    assert engine_urls(two_workers, _nodes())["decode"] == ["http://d0:8000", "http://d1:8000"]


def test_atom_spans_hosts_with_data_parallelism():
    launch = _launch(Framework.ATOM, FrontendType.ATOMESH, decode_tp=8, backend="mooncake")
    commands = cluster_commands(launch, _nodes(framework="atom"))
    decode = _argv(commands[2])
    assert decode[decode.index("--server-port") + 1] == "8020"
    assert decode[decode.index("--data-parallel-size") + 1] == "2"
    assert decode[decode.index("--data-parallel-size-local") + 1] == "1"
    assert decode[decode.index("--data-parallel-rank") + 1] == "1"
    assert decode[decode.index("--data-parallel-master-ip") + 1] == "d0"
    kv = json.loads(decode[decode.index("--kv-transfer-config") + 1])
    assert kv == {"kv_role": "kv_consumer", "kv_connector": "mooncake", "proxy_ip": "d1", "handshake_port": 6301}
    assert decode[decode.index("--max-model-len") + 1] == "2048"
    router = _argv(commands[-1])
    assert router[:5] == ["atomesh", "launch", "--pd-disaggregation", "--backend", "atom"]
    assert router[router.index("--prefill") + 2] == "--decode"


def test_atom_tp_above_a_host_cannot_span_hosts():
    launch = _launch(Framework.ATOM, FrontendType.ATOMESH, decode_tp=16)
    with pytest.raises(CommandError, match="cannot exceed"):
        cluster_commands(launch, _nodes(framework="atom"))


def test_hosts_must_split_into_workers():
    launch = _launch()
    launch = replace(launch, decode=replace(launch.decode, workers=3))
    with pytest.raises(CommandError, match="split"):
        cluster_commands(launch, _nodes())


def test_engine_context_is_left_to_the_engine_when_unset():
    commands = cluster_commands(replace(_launch(), context_length=None), _nodes())
    assert "--context-length" not in _argv(commands[0])
