# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

import pytest

from inference_suite.contracts.base_launch import LaunchQuery, Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode.nodes import (
    NODE_LIST_ENV,
    LaunchMode,
    NodeListError,
    parse_node_list,
    read_node_list,
)
from inference_suite.multinode.record import FrontendType, PdMode, Role


def _document(**fields):
    document = {
        "framework": "sglang",
        "model": "/models/DeepSeek-R1",
        "service_url": "http://msg1.ns.svc.cluster.local:8000",
        "prefill_url": "http://msg1-prefill.ns.svc.cluster.local:8010",
        "decode_url": "http://msg1-decode.ns.svc.cluster.local:8020",
        "router": {"host": "10.0.0.1"},
        "prefill": [{"host": "10.0.0.2"}],
        "decode": [{"host": "10.0.0.3"}, {"host": "10.0.0.4"}],
        "gpus_per_node": 8,
        "ssh": {"port": 25032, "user": "root", "key_file": "/tmp/key"},
        "transfer_backend": "mori",
        "topology": {"prefill": {"tp": 8, "ep": None, "dp_attention": None}, "decode": {"tp": None}},
        "launch_mode": "in_place",
        "config": {"prefill_args": "--a 1", "decode_args": "--b '2 3'", "router_args": ""},
    }
    document.update(fields)
    return {key: value for key, value in document.items() if value is not None}


def test_full_node_list_parses():
    nodes = parse_node_list(_document())
    assert nodes.framework is Framework.SGLANG
    assert (nodes.router, nodes.prefill, nodes.decode) == ("10.0.0.1", ("10.0.0.2",), ("10.0.0.3", "10.0.0.4"))
    assert nodes.ssh.port == 25032 and nodes.ssh.key_file == "/tmp/key"
    assert nodes.launch_mode is LaunchMode.IN_PLACE
    assert nodes.topology[Role.PREFILL].tp == 8 and nodes.topology[Role.DECODE].tp is None
    assert nodes.operator.prefill == ("--a", "1") and nodes.operator.decode == ("--b", "2 3")
    assert nodes.prefill_url.endswith(":8010")


def test_minimal_node_list_takes_defaults():
    nodes = parse_node_list(
        _document(ssh=None, topology=None, config=None, launch_mode=None, transfer_backend=None, prefill_url=None)
    )
    assert nodes.ssh is None and nodes.topology == {} and not nodes.operator.given
    assert nodes.launch_mode is LaunchMode.DIRECT and nodes.transfer_backend == "" and nodes.prefill_url == ""


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"extra": 1}, "unknown keys"),
        ({"framework": "vllm"}, "framework"),
        ({"service_url": "msg1:8000"}, "service_url"),
        ({"decode": [{"host": "10.0.0.2"}]}, "share hosts"),
        ({"decode": [{"host": "10.0.0.3"}, {"host": "10.0.0.3"}]}, "twice"),
        ({"decode": []}, "non-empty list"),
        ({"ssh": {"key_file": "key"}}, "absolute"),
        ({"ssh": {"key_file": "/k", "port": 70000}}, "ssh.port"),
        ({"gpus_per_node": 0}, "gpus_per_node"),
        ({"launch_mode": "remote"}, "launch_mode"),
        ({"topology": {"router": {}}}, "unknown keys"),
        ({"topology": {"decode": {"dp_attention": "yes"}}}, "dp_attention"),
        ({"config": {"prefill_args": "--a '1"}}, "tokenizable"),
    ],
)
def test_invalid_node_list_is_refused(fields, match):
    with pytest.raises(NodeListError, match=match):
        parse_node_list(_document(**fields))


def test_missing_required_key_is_refused():
    document = _document()
    del document["router"]
    with pytest.raises(NodeListError, match="lacks"):
        parse_node_list(document)


def test_read_from_environment():
    assert read_node_list({NODE_LIST_ENV: json.dumps(_document())}).router == "10.0.0.1"
    with pytest.raises(NodeListError, match="unset"):
        read_node_list({})
    with pytest.raises(NodeListError, match="not JSON"):
        read_node_list({NODE_LIST_ENV: "{"})


def _session(framework="sglang"):
    return LaunchQuery(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        framework=framework,
        board="mi355x",
        workload=Workload.FIXED,
        concurrencies=(64,),
        isl=1024,
        osl=1024,
    )


def test_query_combines_session_and_node_list():
    query = parse_node_list(_document()).query(_session())
    assert query.model_path == "/models/DeepSeek-R1" and query.model_id == "deepseek-ai/DeepSeek-R1-0528"
    assert query.prefill.tp == 8 and query.decode.tp is None and query.decode.hosts == ("10.0.0.3", "10.0.0.4")
    assert query.router_host == "10.0.0.1" and query.transfer_backend == "mori"
    with pytest.raises(NodeListError, match="framework|runs"):
        parse_node_list(_document()).query(_session("atom"))


def test_node_list_names_the_router_it_runs():
    nodes = parse_node_list(_document(frontend="atomesh"))
    assert nodes.frontend is FrontendType.ATOMESH
    assert nodes.query(_session()).served_frontend is FrontendType.ATOMESH
    assert parse_node_list(_document()).query(_session()).served_frontend is FrontendType.SGLANG_ROUTER
    with pytest.raises(NodeListError, match="frontend"):
        parse_node_list(_document(frontend="nginx"))


def _aggregated(**fields):
    aggregated = {
        "mode": "aggregated",
        "prefill": None,
        "decode": None,
        "prefill_url": None,
        "decode_url": None,
        "worker": [{"host": "10.0.0.5"}, {"host": "10.0.0.6"}],
        "topology": {"worker": {"tp": 16}},
        "config": {"worker_args": "--w 1"},
    }
    return _document(**{**aggregated, **fields})


def test_aggregated_node_list_names_worker_hosts():
    nodes = parse_node_list(_aggregated())
    assert nodes.mode is PdMode.AGGREGATED and nodes.roles == (Role.WORKER,)
    assert nodes.worker == ("10.0.0.5", "10.0.0.6") and nodes.prefill == () and nodes.decode == ()
    assert nodes.operator.worker == ("--w", "1") and nodes.topology[Role.WORKER].tp == 16
    query = nodes.query(_session())
    assert query.mode is PdMode.AGGREGATED and query.roles[Role.WORKER].hosts == nodes.worker


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"prefill": [{"host": "10.0.0.2"}]}, "aggregated node list names no prefill"),
        ({"worker": None}, "lacks"),
        ({"config": {"prefill_args": "--a"}}, "unknown keys"),
        ({"topology": {"decode": {}}}, "unknown keys"),
        ({"worker": [{"host": "10.0.0.5"}, {"host": "10.0.0.5"}]}, "twice"),
        ({"mode": "mixed"}, "mode"),
    ],
)
def test_invalid_aggregated_node_list_is_refused(fields, match):
    with pytest.raises(NodeListError, match=match):
        parse_node_list(_aggregated(**fields))


def test_a_disaggregated_node_list_names_no_workers():
    with pytest.raises(NodeListError, match="disaggregated node list names no worker"):
        parse_node_list(_document(worker=[{"host": "10.0.0.9"}]))
    with pytest.raises(NodeListError, match="unknown keys"):
        parse_node_list(_document(config={"worker_args": "--w"}))


def test_infera_frontend_needs_no_router_host():
    nodes = parse_node_list(_aggregated(frontend="infera", router=None))
    assert nodes.frontend is FrontendType.INFERA and nodes.router == ""
    with pytest.raises(NodeListError, match="lacks"):
        parse_node_list(_aggregated(router=None))
    with pytest.raises(NodeListError, match="does not serve through the infera frontend"):
        parse_node_list(_aggregated(frontend="infera", framework="atom"))


def test_a_host_entry_names_its_own_ssh_port():
    nodes = parse_node_list(_document(decode=[{"host": "10.0.0.3", "ssh_port": 25042}, {"host": "10.0.0.4"}]))
    assert nodes.ssh.port_of("10.0.0.3") == 25042 and nodes.ssh.port_of("10.0.0.4") == 25032
    with pytest.raises(NodeListError, match="ssh_port"):
        parse_node_list(_document(decode=[{"host": "10.0.0.3", "ssh_port": 0}]))
