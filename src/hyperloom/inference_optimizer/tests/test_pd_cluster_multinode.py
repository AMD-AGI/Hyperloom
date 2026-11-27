# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atom and sglangrouter multi-node state, launch commands, and in-place exec."""

from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node._internal import pd_cluster as pd_internal
from hyperloom.inference_optimizer.multi_node._internal import process_image
from hyperloom.inference_optimizer.multi_node._internal import topology_file as topo
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import (
    drop_unsupported_baseline_flags,
    engine_command,
    launch_script,
    pods_for,
    record_profile_paths,
    router_command,
)
from hyperloom.inference_optimizer.multi_node._internal.pd_cluster import LaunchSpec
from hyperloom.inference_optimizer.multi_node._internal.process_image import replace_process_image
from hyperloom.inference_optimizer.multi_node.commands import pd_cluster as pd_cmds
from hyperloom.inference_optimizer.multi_node._internal import pd_remote
from hyperloom.inference_optimizer.multi_node.state_paths import bind_state_file_to_session, multi_node_state_env


def _spec(**overrides: object) -> LaunchSpec:
    base = dict(
        backend="atomesh",
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


def test_router_and_engine_commands_match_the_platform_entrypoints() -> None:
    spec = _spec()
    router = router_command(spec)
    assert router.startswith("atomesh launch ")
    assert "--prefill http://10.0.1.11:8000 8998" in router
    assert "--decode http://10.0.2.21:8000" in router
    assert "--backend sglang" in router
    prefill = engine_command("prefill", spec)
    decode = engine_command("decode", _spec(backend="sglangrouter"))
    assert "python3 -m sglang.launch_server" in prefill
    assert "--disaggregation-mode prefill" in prefill
    assert "--disaggregation-bootstrap-port 8998" in prefill
    assert "--disaggregation-mode decode" in decode
    assert "--disaggregation-bootstrap-port 9098" in decode
    sglang_router = router_command(_spec(backend="sglangrouter"))
    assert sglang_router.startswith("python3 -m sglang_router.launch_router ")
    assert "atomesh" not in sglang_router


def test_sglang_router_tolerates_a_busy_prefill_unless_the_operator_sets_its_health_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shlex

    monkeypatch.delenv("HYPERLOOM_MN_SGLANG_ROUTER_ARGS", raising=False)
    words = shlex.split(router_command(_spec(backend="sglangrouter")))
    assert "--disable-circuit-breaker" in words
    assert words[words.index("--health-check-timeout-secs") + 1] == "600"
    assert words[words.index("--health-failure-threshold") + 1] == "100"
    assert words[words.index("--request-timeout-secs") + 1] == "14400"
    assert "--health-check-timeout-secs" not in router_command(_spec())

    monkeypatch.setenv("HYPERLOOM_MN_SGLANG_ROUTER_ARGS", "--health-check-timeout-secs 30")
    words = shlex.split(router_command(_spec(backend="sglangrouter")))
    assert "--disable-circuit-breaker" not in words
    assert words[words.index("--health-check-timeout-secs") + 1] == "30"


def test_engine_launch_defaults_the_disaggregation_timeouts_without_overriding_a_set_one() -> None:
    script = launch_script("prefill", "true", {"SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "5"})
    assert 'SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-100000}"' in script
    assert "export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=5" in script
    assert "SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-" not in script
    assert "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT" not in launch_script("router", "true", {})


def test_mooncake_knobs_are_forwarded_to_the_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.multi_node.commands import infera

    monkeypatch.setenv("MC_ENABLE_DEST_DEVICE_AFFINITY", "1")
    assert infera._collect_forward_env()["MC_ENABLE_DEST_DEVICE_AFFINITY"] == "1"


@pytest.mark.parametrize(
    ("base_flags", "env", "disabled"),
    [
        ("--trust-remote-code", {}, True),
        ("--trust-remote-code", {"HYPERLOOM_AGENTX": "1"}, False),
        ("--trust-remote-code --enable-hierarchical-cache", {}, False),
    ],
    ids=["fixed-length", "agentic", "hicache"],
)
def test_an_inferencex_baseline_disables_the_prefill_prefix_cache_only_on_a_fixed_length_run_without_hicache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, base_flags: str, env: dict[str, str], disabled: bool
) -> None:
    import shlex

    import yaml

    from hyperloom.inference_optimizer.multi_node._internal import baseline_args as ba

    models = tmp_path / ba._MODELS_YAML
    models.parent.mkdir(parents=True)
    entry = {"base_flags": base_flags, "prefill": {"disable_radix_cache": True}, "decode": {}}
    models.write_text(yaml.safe_dump({"M": entry}), encoding="utf-8")
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)

    choice = ba._inferencex({"MODEL_PATH": "/models/M", "INFERENCEX_PATH": str(tmp_path), **env})

    assert choice is not None
    assert ("--disable-radix-cache" in shlex.split(choice.prefill)) is disabled


@pytest.mark.parametrize(
    ("keys", "env", "picked"),
    [
        (["M", "M-AgentX", "M-DI"], {"HYPERLOOM_AGENTX": "1"}, "M-AgentX"),
        (["M", "M-AgentX", "M-DI"], {}, "M"),
        (["M", "M-DI"], {"HYPERLOOM_AGENTX": "1"}, "M"),
        (["M-DI", "M-AgentX"], {}, "M-DI"),
        (["m-di"], {"HYPERLOOM_AGENTX": "1"}, "m-di"),
    ],
)
def test_inferencex_entry_lookup_prefers_the_agentx_then_bare_then_di_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, keys: list[str], env: dict[str, str], picked: str
) -> None:
    import yaml

    from hyperloom.inference_optimizer.multi_node._internal import baseline_args as ba

    models = tmp_path / ba._MODELS_YAML
    models.parent.mkdir(parents=True)
    models.write_text(yaml.safe_dump({key: {"base_flags": "", "prefill": {}, "decode": {}} for key in keys}))
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)

    choice = ba._inferencex({"MODEL_PATH": "/models/M", "INFERENCEX_PATH": str(tmp_path), **env})

    assert choice is not None and choice.detail == f"{models} entry {picked}"


@pytest.mark.parametrize("layout", ["unset", "no_models_yaml", "no_entry"])
def test_missing_inferencex_baseline_names_the_actual_gap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, layout: str
) -> None:
    import yaml

    from hyperloom.inference_optimizer.multi_node._internal import baseline_args as ba

    models = tmp_path / ba._MODELS_YAML
    env = {"MODEL_PATH": "/models/M"}
    if layout != "unset":
        env["INFERENCEX_PATH"] = str(tmp_path)
    if layout == "no_entry":
        models.parent.mkdir(parents=True)
        models.write_text(yaml.safe_dump({"Other": {}}))
    expected = {
        "unset": "INFERENCEX_PATH is unset",
        "no_models_yaml": f"{models} does not exist",
        "no_entry": f"{models} has no entry for MODEL_PATH=/models/M",
    }[layout]

    with pytest.raises(ba.BaselineArgsMissing) as info:
        ba.resolve_baseline_args(env)

    assert expected in str(info.value)


def test_engine_listens_on_the_port_of_its_role_url() -> None:
    spec = _spec(
        prefill_url="http://w-prefill.ns.svc.cluster.local:8010",
        decode_url="http://w-decode.ns.svc.cluster.local:8020",
    )
    assert "--port 8010 " in engine_command("prefill", spec)
    assert "--port 8020 " in engine_command("decode", spec)
    assert "--port 8000 " in router_command(spec)
    no_port = _spec(prefill_url="http://w-prefill", decode_url="")
    assert "--port 8000 " in engine_command("prefill", no_port)
    assert "--port 8000 " in engine_command("decode", no_port)


def test_engine_ep_size_follows_the_role_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("EP", "PD_PREFILL_EP", "PD_DECODE_EP"):
        monkeypatch.delenv(name, raising=False)
    state = {"backend": "atomesh", "prefill_pod_ips": ["10.0.0.1"], "decode_pod_ips": ["10.0.0.2"]}
    restart = pd_internal.spec_from_restart(types.SimpleNamespace(model="/m", ep=1, pd_decode_ep=8), state)
    assert "--ep-size" not in engine_command("prefill", restart)
    assert "--ep-size 8" in engine_command("decode", restart)

    monkeypatch.setenv("EP", "8")
    monkeypatch.setenv("PD_PREFILL_EP", "1")
    handed_off = pd_internal.spec_from_env(state)
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


def test_baseline_flags_the_engine_lacks_are_dropped_and_variant_flags_kept() -> None:
    words = "--load-balance-method round_robin --prefill-round-robin-balance --cuda-graph-bs 1 2 --bogus x".split()
    baseline = "--prefill-round-robin-balance --load-balance-method round_robin --cuda-graph-bs 1 2".split()
    engine = frozenset({"--load-balance-method", "--cuda-graph-bs-decode"})
    kept, dropped = drop_unsupported_baseline_flags(words, baseline, engine)
    assert kept == "--load-balance-method round_robin --cuda-graph-bs 1 2 --bogus x".split()
    assert dropped == ["--prefill-round-robin-balance"]
    assert drop_unsupported_baseline_flags(words, baseline, frozenset()) == (words, [])


def test_pods_record_the_engine_port_of_their_role() -> None:
    spec = _spec(prefill_url="http://p-svc:8010", decode_url="http://d-svc:8020")
    pods = pods_for(["10.0.0.9"], ["10.0.0.1"], ["10.0.0.2"], 29000, spec)
    ports = {pod["role"]: pod.get("enginePort") for pod in pods}
    assert ports == {"router": None, "prefill": 8010, "decode": 8020}


def test_previous_baseline_args_are_reused_only_for_the_same_model_and_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.multi_node._internal import baseline_args as ba

    recorded: dict[str, object] = {}
    monkeypatch.setenv("MODEL_PATH", "/models/DeepSeek-R1")
    monkeypatch.delenv("EP", raising=False)
    ba.record_baseline_args(recorded, ba.BaselineArgs(ba.SOURCE_INFERENCEX, "--context-length 8192", "--x"))
    same = {"MODEL_PATH": "/models/DeepSeek-R1"}
    assert ba.resolve_baseline_args(env=same, previous=recorded).prefill == "--context-length 8192"
    for env in ({"MODEL_PATH": "/models/Qwen3"}, {"MODEL_PATH": "/models/DeepSeek-R1", "EP": "8"}):
        with pytest.raises(ba.BaselineArgsMissing):
            ba.resolve_baseline_args(env=env, previous=recorded)
    legacy = {k: v for k, v in recorded.items() if k != "baseline_identity"}
    with pytest.raises(ba.BaselineArgsMissing):
        ba.resolve_baseline_args(env=same, previous=legacy)


def test_router_listens_on_the_service_url_port_when_the_url_names_its_pod() -> None:
    spec = _spec(backend="sglangrouter")
    direct = pods_for(["10.0.0.9"], ["10.0.0.1"], ["10.0.0.2"], 29000, spec, service_url="http://10.0.0.9:30000")
    assert "--port 30000" in direct[0]["launch_command"]
    assert direct[0]["routerPort"] == 30000
    service = pods_for(["10.0.0.9"], ["10.0.0.1"], ["10.0.0.2"], 29000, spec, service_url="http://router-svc:30000")
    assert "--port 8000" in service[0]["launch_command"]
    assert service[0]["routerPort"] == 8000
    assert "export ATOM_SERVER_PORT=30000\n" in pd_internal.launch_script("router", "true", {}, server_port=30000)


def test_platform_router_is_recorded_from_the_service_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def _resolve(host: str) -> str:
        if host != "router-svc.ns":
            raise socket.gaierror(host)
        return "192.168.0.5"

    monkeypatch.setattr(pd_internal.socket, "gethostbyname", _resolve)
    spec = _spec(backend="sglangrouter")
    pods = pods_for([], ["10.0.0.1"], ["10.0.0.2"], 29000, spec, service_url="http://router-svc.ns:8000")
    assert [pod["role"] for pod in pods] == ["router", "prefill", "decode"]
    router = pods[0]
    assert router["podIP"] == "192.168.0.5"
    assert router["managed_by"] == "platform"
    assert router["sshPort"] == 29000
    assert router["launch_command"].startswith("python3 -m sglang_router.launch_router ")
    unresolved = pods_for([], ["10.0.0.1"], ["10.0.0.2"], 29000, spec, service_url="http://unknown-host:8000")
    assert unresolved[0]["podIP"] == "unknown-host"
    handed_off = pods_for(
        ["10.0.0.9"], ["10.0.0.1"], ["10.0.0.2"], 29000, spec, service_url="http://router-svc.ns:8000"
    )
    assert [pod["podIP"] for pod in handed_off if pod["role"] == "router"] == ["10.0.0.9"]
    assert "managed_by" not in handed_off[0]


def test_platform_router_ip_is_listed_and_stays_platform_managed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd_internal.socket, "gethostbyname", lambda host: "192.168.0.5")
    env = {
        "HYPERLOOM_MN_EXT_SSH_KEY": "/keys/id",
        "HYPERLOOM_MN_EXT_SSH_PORT": "29000",
        "HYPERLOOM_MN_EXT_PREFILL_IPS": "10.0.0.1",
        "HYPERLOOM_MN_EXT_DECODE_IPS": "10.0.0.2",
    }
    state = {"backend": "sglangrouter", "service_url": "http://router-svc.ns:8000"}
    pd_internal.stamp_pd_cluster(state, lambda name: env.get(name, ""))
    assert state["router_pod_ips"] == ["192.168.0.5"]
    assert state["router_pods"][0]["managed_by"] == "platform"

    pd_cmds._refresh_pods(state, _spec(backend="sglangrouter"))
    assert state["router_pod_ips"] == ["192.168.0.5"]
    assert [pod.get("managed_by") for pod in state["router_pods"]] == ["platform"]
    assert [pod["role"] for pod in pd_cmds._ordered_pods(state)] == ["prefill", "decode"]


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


def test_restart_and_stop_skip_the_platform_router() -> None:
    state = {
        "pods": [
            {"role": "router", "podIP": "192.168.0.5", "managed_by": "platform"},
            {"role": "prefill", "podIP": "10.0.0.1"},
            {"role": "decode", "podIP": "10.0.0.2"},
        ]
    }
    assert [pod["role"] for pod in pd_cmds._ordered_pods(state)] == ["prefill", "decode"]


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


@pytest.mark.parametrize("file_backend", ["atom", "atomesh"])
def test_handoff_file_names_atomesh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, file_backend: str) -> None:
    doc = {
        "schema": "topology-handoff/v1",
        "backend": file_backend,
        "nodes": 3,
        "gpus_per_node": 8,
        "service_url": "http://10.0.0.10:8000",
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    overlay = topo.topology_file_overlay()
    assert overlay is not None
    assert overlay["INFERENCE_OPTIMIZER_MN_BACKEND"] == "atomesh"


def test_atom_handoff_records_atomesh_and_uses_one_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = {
        "schema": "topology-handoff/v1",
        "backend": "atom",
        "nodes": 5,
        "gpus_per_node": 8,
        "service_url": "http://10.0.0.10:8000",
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    session = tmp_path / "session"
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_GPUS_PER_NODE", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_MN_BACKEND", raising=False)
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.setenv("PD_MODE", "disaggregated")
    monkeypatch.setenv("MODEL_PATH", "/models/Qwen3-30B-A3B")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_KEY", "/workspace/hyperloom/runtime/mn_ssh/id_ed25519")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "2222")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_ROUTER_IPS", "10.0.0.10")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_PREFILL_IPS", "10.0.1.11,10.0.1.12")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_DECODE_IPS", "10.0.2.21,10.0.2.22")
    state_file = bind_state_file_to_session(session)
    monkeypatch.delenv("MULTI_NODE_STATE_FILE", raising=False)

    state = ext.build_external_state_from_env()

    assert state["backend"] == "atomesh"
    assert state["external"] is True
    assert state["pd_mode"] == "disaggregated"
    assert state["nodes"] == 5
    assert state["gpus_per_node"] == 8
    assert state["ssh_port"] == 2222
    assert state["ssh_key_path"].endswith("id_ed25519")
    assert state["service_url"] == "http://10.0.0.10:8000"
    assert state["profile_trace"] == str(session / "runs" / "profile" / "torch_trace")
    assert state["steady_state_trace"] == ""
    assert [pod["role"] for pod in state["pods"]] == ["router", "prefill", "prefill", "decode", "decode"]
    assert {pod["sshPort"] for pod in state["pods"]} == {2222}
    assert state["pods"][0]["launch_command"].startswith("atomesh launch ")
    assert "sglang.launch_server" in state["pods"][1]["launch_command"]
    assert "--disaggregation-mode prefill" in state["pods"][1]["launch_command"]
    assert "--disaggregation-mode decode" in state["pods"][3]["launch_command"]
    # Infera's decode stride must not apply.
    assert all(pod["sshPort"] == 2222 for pod in state["decode_pods"])


def test_sglangrouter_file_does_not_require_an_infera_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "topology.json"
    path.write_text(
        json.dumps(
            {
                "schema": "topology-handoff/v1",
                "backend": "sglangrouter",
                "nodes": 3,
                "gpus_per_node": 8,
                "service_url": "http://10.0.0.10:8000",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    overlay = topo.topology_file_overlay()
    assert overlay is not None
    assert overlay["INFERENCE_OPTIMIZER_MN_BACKEND"] == "sglangrouter"
    assert "HYPERLOOM_MN_EXT_SSH_PORT" not in overlay


def test_record_profile_paths_updates_the_state_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = tmp_path / "session"
    path = bind_state_file_to_session(session)
    path.write_text(
        json.dumps({"backend": "atomesh", "profile_trace": "", "steady_state_trace": ""}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    monkeypatch.setenv("MULTI_NODE_STATE", str(path))
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


def test_replace_process_image_keeps_the_pid_and_applies_env(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    pid = os.fork()
    if pid == 0:
        time.sleep(30)
        os._exit(0)
    try:
        replace_process_image(
            pid,
            ["/bin/sh", "-c", f'printf %s "$HL_FROM_PARENT" > {marker}'],
            env_overrides={"HL_FROM_PARENT": "kept"},
        )
        os.waitpid(pid, 0)
    finally:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
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
        replace_process_image(pid, ["/bin/sh", "-c", "export HL_LEAKED=1; exec /bin/sleep 30"], env_snapshot=snapshot)
        _wait_for_cmdline(pid, b"/bin/sleep\x0030")
        replace_process_image(
            pid,
            ["/bin/sh", "-c", f'printf "%s|%s" "${{HL_ORIGINAL:-}}" "${{HL_LEAKED:-}}" > {marker}'],
            env_snapshot=snapshot,
        )
        os.waitpid(pid, 0)
    finally:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    assert marker.read_text(encoding="utf-8") == "orig|"


def test_replace_process_image_refuses_a_missing_executable_and_leaves_the_process_running() -> None:
    server = subprocess.Popen(["/bin/sh", "-c", "/bin/sleep 30 & wait"])
    try:
        deadline = time.monotonic() + 10
        while not process_image.descendant_pids(server.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        children = process_image.descendant_pids(server.pid)
        assert children
        with pytest.raises(OSError, match="not an executable"):
            replace_process_image(server.pid, ["/nonexistent/hl-no-such-binary"])
        time.sleep(0.3)
        assert server.poll() is None
        assert process_image.descendant_pids(server.pid) == children
    finally:
        for child in process_image.descendant_pids(server.pid):
            os.kill(child, 9)
        server.kill()
        server.wait()


@pytest.mark.parametrize(
    ("failing", "killed"),
    [("_syscall_gadget", False), ("_stack_slot", False), ("_check_exec", True)],
)
def test_replace_process_image_kills_only_once_the_exec_is_under_way(
    monkeypatch: pytest.MonkeyPatch, failing: str, killed: bool
) -> None:
    kills: list[tuple[int, int]] = []
    stopped_by_trap = (5 << 8) | 0x7F

    def _fail(*_args: object, **_kwargs: object) -> int:
        raise OSError(errno.ENOENT, "boom")

    monkeypatch.setattr(process_image, "_ptrace", lambda *_args: 0)
    monkeypatch.setattr(process_image, "_libc", types.SimpleNamespace(ptrace=lambda *_args: 0))
    monkeypatch.setattr(process_image, "_syscall_gadget", lambda _pid: 0x1000)
    monkeypatch.setattr(process_image, "_stack_slot", lambda _pid, _rsp, _nbytes: 0x2000)
    monkeypatch.setattr(process_image, "_check_exec", lambda *_args: None)
    monkeypatch.setattr(process_image, "merged_environ", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(process_image, "descendant_pids", lambda _pid: [4243])
    monkeypatch.setattr(process_image, failing, _fail)
    monkeypatch.setattr(os, "waitpid", lambda pid, _flags: (pid, stopped_by_trap))
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))

    with pytest.raises(OSError):
        process_image.replace_process_image(4242, ["/bin/true"])

    assert kills == ([(4243, 9), (4242, 9)] if killed else [])


def test_cstr_table_writes_undecodable_environment_bytes_back_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[bytes] = []
    monkeypatch.setattr(process_image, "_poke", lambda _pid, _addr, blob: written.append(blob))
    raw = b"HL_RAW=\xff\xfe"

    process_image._write_cstr_table(4242, 0x1000, [raw.decode("utf-8", "surrogateescape")])

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


def test_tracelens_patch_failure_reports_the_pod_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import argparse
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.multi_node import cli as real_cli

    pod_error = "cannot derive per-version patches subdir from version '0.5.23.dev1+g1'"
    stdout = json.dumps({"status": "failed", "per_pod": [{"status": "failed", "error": pod_error}]})
    fake = SimpleNamespace(
        EXIT_CONFIG_ERROR=real_cli.EXIT_CONFIG_ERROR,
        _poll_timeout_from_args=lambda _args: 30,
        _infera_ssh_bash_with_env=lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ssh"], 1, stdout=stdout, stderr="bash: warning: setlocale: LC_ALL: cannot change locale\n"
        ),
    )
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    monkeypatch.setattr(pd_cmds, "read_bundled_pod_python_script", lambda *_args: "print()")
    monkeypatch.setattr(pd_cmds, "_require_state", lambda: dict(_PD_STATE))

    ns = argparse.Namespace(tracelens_root="/tl", sglang_version_pin=None, patch_set="roofline")
    assert pd_cmds.apply_tracelens_patch(ns) == 1
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
    monkeypatch.setattr(pd_cmds, "_cli", lambda: _probe_cli(_run_here))
    state = _probe_state(tmp_path)

    assert pd_cmds.record_profile_dir_share(state) == ""
    assert state["profile_trace_shared"] is True
    assert "profile_trace_share_error" not in state
    assert list(tmp_path.iterdir()) == []


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
    monkeypatch.setattr(pd_cmds, "_cli", lambda: _probe_cli(reply))
    state = _probe_state(tmp_path)

    reason = pd_cmds.record_profile_dir_share(state)

    assert reason and "pod 10.0.0." in reason
    assert state["profile_trace_shared"] is False
    assert state["profile_trace_share_error"] == reason
    assert list(tmp_path.iterdir()) == []


def test_tracelens_patch_runs_under_the_container_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse

    from hyperloom.inference_optimizer.multi_node import cli as real_cli

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python3").write_text('#!/bin/sh\necho "venv-python $*"\n', encoding="utf-8")
    (venv_bin / "python3").chmod(0o755)
    environ = tmp_path / "environ"
    environ.write_bytes(f"PATH={venv_bin}:/usr/bin:/bin\0HOME=/root\0".encode())
    scripts: list[str] = []
    fake = _FakeCli({})
    fake.EXIT_CONFIG_ERROR = real_cli.EXIT_CONFIG_ERROR
    fake._poll_timeout_from_args = lambda _args: 30

    def _bash_with_env(state, ip, script, *, timeout, port=None):
        scripts.append(script)
        return subprocess.CompletedProcess(["ssh"], 0, stdout='{"status": "applied", "per_pod": [{}]}', stderr="")

    fake._infera_ssh_bash_with_env = _bash_with_env
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    monkeypatch.setattr(pd_cmds, "_require_state", lambda: dict(_PD_STATE))
    monkeypatch.setattr(pd_cmds, "CONTAINER_ENVIRON_PATH", str(environ))
    monkeypatch.setattr(pd_cmds, "read_bundled_pod_python_script", lambda *_args: "print('patched')")

    ns = argparse.Namespace(tracelens_root="/tl", sglang_version_pin=None, patch_set="roofline")
    assert pd_cmds.apply_tracelens_patch(ns) == 0
    assert len(scripts) == 2
    ran = subprocess.run(
        ["bash", "-c", scripts[0]], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, check=True
    )
    assert ran.stdout.startswith("venv-python ")
    assert "--local --tracelens-root /tl --patch-set roofline" in ran.stdout


def test_engine_restart_gives_each_pod_its_own_profiler_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({})
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    monkeypatch.setattr(pd_cmds, "extract_pod_json", lambda _stdout: {})
    env = {"SGLANG_TORCH_PROFILER_DIR": "/shared/torch_trace"}
    router = {"role": "router", "podIP": "10.0.0.3", "sshPort": 29000}
    second_decode = {"role": "decode", "podIP": "10.0.0.4", "sshPort": 29000}
    state = dict(_PD_STATE, decode_pods=[*_PD_STATE["decode_pods"], second_decode])
    for pod in [*state["prefill_pods"], *state["decode_pods"], router]:
        pd_cmds._ssh_pod(state, pod, action="restart", extra_env=env, timeout=5)
    scripts = dict(fake.calls)
    assert "export SGLANG_TORCH_PROFILER_DIR=/shared/torch_trace/prefill/0\n" in scripts["10.0.0.1"]
    assert "export SGLANG_TORCH_PROFILER_DIR=/shared/torch_trace/decode/0\n" in scripts["10.0.0.2"]
    assert "export SGLANG_TORCH_PROFILER_DIR=/shared/torch_trace/decode/1\n" in scripts["10.0.0.4"]
    assert "SGLANG_TORCH_PROFILER_DIR" not in scripts["10.0.0.3"]
    assert env == {"SGLANG_TORCH_PROFILER_DIR": "/shared/torch_trace"}


@pytest.mark.parametrize(
    ("service_url", "port"), [("http://svc.ns.svc.cluster.local:80", 8000), ("http://10.0.0.9:9000", 9000)]
)
def test_infera_frontend_listens_on_the_router_pod_port(
    monkeypatch: pytest.MonkeyPatch, service_url: str, port: int
) -> None:
    from hyperloom.inference_optimizer.multi_node.commands import infera

    fake = _FakeCli({})
    monkeypatch.setattr(infera, "_mn_cli", fake)
    monkeypatch.setattr(infera, "extract_pod_json", lambda _stdout: {})
    state = {
        "service_url": service_url,
        "ssh_port": 29000,
        "router_pods": [{"role": "router", "podIP": "10.0.0.9", "sshPort": 29000}],
    }

    infera._infera_ensure_frontend(state, "/m", 5)

    [(_ip, script)] = fake.calls
    assert f"--port {port} --router-tokenizer-path" in script


def test_engines_not_running_names_pods_whose_launch_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "HL_ALIVE\n", "10.0.0.2": "HL_GONE\n"})
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    assert pd_cmds.engines_not_running(dict(_PD_STATE)) == ["decode 10.0.0.2"]
    assert "/tmp/hyperloom-mn-decode.pid" in dict(fake.calls)["10.0.0.2"]


class _LocalCli:
    """Runs the SSH script on this host instead of a pod."""

    def _infera_ssh_bash_with_env(self, state, ip, script, *, timeout, port=None):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout)


def test_engines_not_running_counts_an_engine_waiting_in_its_launch_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The launch script holds the pid while it waits for the GPU; that engine is starting, not gone."""
    launch = tmp_path / "hyperloom-mn-decode.sh"
    launch.write_text("#!/bin/bash\nsleep 30\n", encoding="utf-8")
    launch.chmod(0o755)
    pidfile = tmp_path / "decode.pid"
    monkeypatch.setattr(pd_cmds, "_cli", lambda: _LocalCli())
    monkeypatch.setattr(pd_cmds, "pidfile_path", lambda role: str(pidfile))
    monkeypatch.setattr(pd_cmds, "launch_path", lambda role: str(launch))
    state = dict(_PD_STATE, prefill_pods=[])
    waiting = subprocess.Popen([str(launch)])
    other = subprocess.Popen(["sleep", "30"])
    try:
        time.sleep(0.2)
        pidfile.write_text(str(waiting.pid), encoding="utf-8")
        assert pd_cmds.engines_not_running(state) == []
        pidfile.write_text(str(other.pid), encoding="utf-8")
        assert pd_cmds.engines_not_running(state) == ["decode 10.0.0.2"]
    finally:
        for proc in (waiting, other):
            proc.kill()
            proc.wait()


def test_sglangrouter_router_needle_matches_the_renamed_process() -> None:
    needle = pd_internal.cmdline_needle("router", "sglangrouter")
    assert "sglang::router" in needle.split("|")
    assert "sglang_router.launch_router" in needle.split("|")


def test_remote_driver_finds_the_server_by_any_needle(tmp_path: Path) -> None:
    driver = pd_remote.REMOTE_DRIVER.replace("\nrestart_remote_main()\n", "\n")
    scope: dict[str, object] = {"os": os}
    exec(driver, scope)
    token = f"hl-needle-{os.getpid()}-{time.monotonic_ns()}"
    named = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", token])
    try:
        time.sleep(0.2)
        find = scope["_find_server"]
        assert find(str(tmp_path / "none.pid"), f"absent-{token}|{token}") == named.pid
        assert find(str(tmp_path / "none.pid"), f"absent-{token}") == 0
    finally:
        named.kill()
        named.wait()


def test_remote_driver_ignores_a_pidfile_naming_an_unrelated_process(tmp_path: Path) -> None:
    driver = pd_remote.REMOTE_DRIVER.replace("\nrestart_remote_main()\n", "\n")
    scope: dict[str, object] = {"os": os}
    exec(driver, scope)
    unrelated = subprocess.Popen(["sleep", "30"])
    pidfile = tmp_path / "decode.pid"
    pidfile.write_text(str(unrelated.pid), encoding="utf-8")
    try:
        time.sleep(0.2)
        assert scope["_find_server"](str(pidfile), "sglang.launch_server") != unrelated.pid
    finally:
        unrelated.kill()
        unrelated.wait()


def test_remote_driver_takes_over_a_parked_launcher_child(tmp_path: Path) -> None:
    driver = pd_remote.REMOTE_DRIVER.replace("\nrestart_remote_main()\n", "\n")
    scope: dict[str, object] = {"os": os}
    exec(driver, scope)
    parked = subprocess.Popen(["sleep", "infinity"])
    other = subprocess.Popen(["sleep", "infinity"])
    try:
        time.sleep(0.2)
        real_ppid = scope["_ppid"]
        scope["_ppid"] = lambda pid: 1 if pid in (parked.pid, other.pid) else real_ppid(pid)
        scope["_sid"] = lambda pid: 1 if pid in (1, parked.pid) else 2
        find = scope["_find_server"]
        assert find(str(tmp_path / "none.pid"), f"absent-{os.getpid()}-{time.monotonic_ns()}") == parked.pid
    finally:
        for p in (parked, other):
            p.kill()
            p.wait()


def _driver_scope(monkeypatch: pytest.MonkeyPatch, *, held: bool, calls: list) -> dict:
    driver = pd_remote.REMOTE_DRIVER.replace("\nrestart_remote_main()\n", "\n")
    scope: dict = {"os": os}
    exec(driver, scope)
    scope["_find_server"] = lambda _pidfile, _needle: 4242
    scope["_ppid"] = lambda pid: 1
    scope["_sid"] = lambda pid: 1 if (pid == 1 or held) else pid
    scope["replace_process_image"] = lambda pid, argv, env_snapshot="": calls.append(("exec", pid, argv[0]))
    scope["descendant_pids"] = lambda pid: [4243]
    scope["baseline_environ"] = lambda pid, snapshot: calls.append(("env", pid, snapshot)) or [f"FROM={pid}"]
    scope["_spawn_detached"] = lambda launch, env: calls.append(("spawn", launch, tuple(env))) or 5151
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append(("kill", pid, sig)))
    return scope


@pytest.mark.parametrize("held", [True, False])
@pytest.mark.parametrize("action", ["restart", "stop"])
def test_remote_driver_execs_in_place_only_for_the_launcher_waited_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], held: bool, action: str
) -> None:
    calls: list = []
    scope = _driver_scope(monkeypatch, held=held, calls=calls)
    monkeypatch.setenv("HL_PIDFILE", str(tmp_path / "role.pid"))
    monkeypatch.setenv("HL_LAUNCH_PATH", str(tmp_path / "launch.sh"))
    monkeypatch.setenv("HL_ACTION", action)
    scope["restart_remote_main"]()
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    if held:
        assert result["action"] == ("exec" if action == "restart" else "stop-exec")
        assert [c[0] for c in calls] == ["exec"]
    else:
        assert result["action"] == ("respawn" if action == "restart" else "stop-kill")
        assert ("kill", 4242, 9) in calls and ("kill", 4243, 9) in calls
        assert not [c for c in calls if c[0] == "exec"]
        assert [c[0] for c in calls if c[0] == "spawn"] == (["spawn"] if action == "restart" else [])
        if action == "restart":
            assert calls.index(("env", 4242, str(tmp_path / "role.pid") + ".environ")) < calls.index(("kill", 4242, 9))
            assert [c[2] for c in calls if c[0] == "spawn"] == [("FROM=4242",)]


def test_stop_parks_the_launcher_pid_without_its_inherited_sockets() -> None:
    """A parked server must not keep the old server's listening socket or GPU handle open."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.set_inheritable(True)
    parked = subprocess.Popen(["/bin/bash", "-c", pd_remote.park_script()], pass_fds=(sock.fileno(),))
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            cmd = open(f"/proc/{parked.pid}/cmdline", "rb").read().split(b"\0")
            if os.path.basename(cmd[0].decode()) == "sleep":
                break
            time.sleep(0.05)
        assert [part.decode() for part in cmd if part] == ["/bin/sleep", "infinity"]
        links = [os.readlink(f"/proc/{parked.pid}/fd/{fd}") for fd in os.listdir(f"/proc/{parked.pid}/fd")]
        assert not [link for link in links if link.startswith("socket:")]
    finally:
        parked.kill()
        parked.wait()
        sock.close()


def test_spawn_without_a_server_takes_the_container_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []
    scope = _driver_scope(monkeypatch, held=False, calls=calls)
    scope["_find_server"] = lambda _pidfile, _needle: 0
    monkeypatch.setenv("HL_PIDFILE", str(tmp_path / "role.pid"))
    monkeypatch.setenv("HL_LAUNCH_PATH", str(tmp_path / "launch.sh"))
    monkeypatch.setenv("HL_ACTION", "restart")
    scope["restart_remote_main"]()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["action"] == "spawn"
    assert ("env", 1, str(tmp_path / "role.pid") + ".environ") in calls
    assert [c[2] for c in calls if c[0] == "spawn"] == [("FROM=1",)]


def test_respawn_takes_the_container_environment_when_the_server_environment_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list = []
    scope = _driver_scope(monkeypatch, held=False, calls=calls)
    scope["baseline_environ"] = lambda pid, snapshot: calls.append(("env", pid, snapshot)) or []
    scope["read_environ"] = lambda pid: ["PATH=/opt/venv/bin:/usr/bin", f"FROM={pid}"]
    monkeypatch.setenv("HL_PIDFILE", str(tmp_path / "role.pid"))
    monkeypatch.setenv("HL_LAUNCH_PATH", str(tmp_path / "launch.sh"))
    monkeypatch.setenv("HL_ACTION", "restart")
    scope["restart_remote_main"]()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["action"] == "respawn"
    assert [c[2] for c in calls if c[0] == "spawn"] == [("PATH=/opt/venv/bin:/usr/bin", "FROM=1")]


@pytest.mark.parametrize(("found", "expected"), [(4242, "running"), (0, "spawn")])
def test_ensure_starts_the_server_only_when_none_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], found: int, expected: str
) -> None:
    calls: list = []
    scope = _driver_scope(monkeypatch, held=False, calls=calls)
    scope["_find_server"] = lambda _pidfile, _needle: found
    monkeypatch.setenv("HL_PIDFILE", str(tmp_path / "role.pid"))
    monkeypatch.setenv("HL_LAUNCH_PATH", str(tmp_path / "launch.sh"))
    monkeypatch.setenv("HL_ACTION", "ensure")
    scope["restart_remote_main"]()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["action"] == expected
    assert not [c for c in calls if c[0] in ("kill", "exec")]
    assert [c[2] for c in calls if c[0] == "spawn"] == ([("FROM=1",)] if found == 0 else [])


def test_detached_spawn_runs_with_the_given_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    driver = pd_remote.REMOTE_DRIVER.replace("\nrestart_remote_main()\n", "\n")
    scope: dict = {"os": os}
    exec(driver, scope)
    monkeypatch.setenv("HL_CONSOLE", str(tmp_path / "none"))
    launch = tmp_path / "launch.sh"
    launch.write_text('#!/bin/bash\necho "hl-env-$HL_TEST_MARK"\n', encoding="utf-8")
    env = [f"PATH={os.environ['PATH']}", "HL_TEST_MARK=from-container"]
    pid = scope["_spawn_detached"](str(launch), env)
    os.waitpid(pid, 0)
    assert "hl-env-from-container" in (tmp_path / "launch.sh.log").read_text(encoding="utf-8")


def test_detached_spawn_logs_to_file_and_console(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    driver = pd_remote.REMOTE_DRIVER.replace("\nrestart_remote_main()\n", "\n")
    scope: dict = {"os": os}
    exec(driver, scope)
    console = tmp_path / "console"
    console.write_text("", encoding="utf-8")
    monkeypatch.setenv("HL_CONSOLE", str(console))
    launch = tmp_path / "launch.sh"
    launch.write_text("#!/bin/bash\necho hl-spawned-$$\n", encoding="utf-8")
    pid = scope["_spawn_detached"](str(launch), [f"PATH={os.environ['PATH']}"])
    os.waitpid(pid, 0)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and f"hl-spawned-{pid}" not in console.read_text(encoding="utf-8"):
        time.sleep(0.05)
    assert f"hl-spawned-{pid}" in (tmp_path / "launch.sh.log").read_text(encoding="utf-8")
    assert f"hl-spawned-{pid}" in console.read_text(encoding="utf-8")


def test_liveness_probe_greps_every_needle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "HL_ALIVE\n", "10.0.0.2": "HL_ALIVE\n"})
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    monkeypatch.setattr(pd_cmds, "cmdline_needle", lambda _role, _backend: "first|second")
    assert pd_cmds.engines_not_running(dict(_PD_STATE)) == []
    script = dict(fake.calls)["10.0.0.1"]
    assert "grep -qF -e first -e second" in script


def test_engine_flags_are_probed_once_and_only_for_legacy_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCli({"10.0.0.1": "--cuda-graph-bs-decode\n--cuda-graph-bs-prefill\n--port\n"})
    monkeypatch.setattr(pd_cmds, "_cli", lambda: fake)
    state = dict(_PD_STATE)
    assert pd_cmds.probe_engine_flags(state, _spec(), timeout=30) == frozenset()
    assert fake.calls == []
    spec = _spec(decode_extra="--cuda-graph-bs 1 2")
    flags = pd_cmds.probe_engine_flags(state, spec, timeout=30)
    assert flags == frozenset({"--cuda-graph-bs-decode", "--cuda-graph-bs-prefill", "--port"})
    assert pd_cmds.probe_engine_flags(state, spec, timeout=30) == flags
    assert len(fake.calls) == 1
    inferencex = dict(_PD_STATE, baseline_args_source="inferencex", baseline_decode_args="--port 1")
    assert pd_cmds.probe_engine_flags(inferencex, _spec(), timeout=30) == flags
    assert len(fake.calls) == 2


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


def test_restart_spec_drops_inferencex_baseline_flags_the_engine_lacks() -> None:
    state = dict(
        _PD_STATE,
        baseline_args_source="inferencex",
        baseline_decode_args="--prefill-round-robin-balance --load-balance-method round_robin",
        baseline_prefill_args="--load-balance-method round_robin",
    )
    spec = _spec(
        decode_extra="--prefill-round-robin-balance --load-balance-method round_robin",
        prefill_extra="--load-balance-method round_robin",
        engine_flags=frozenset({"--load-balance-method"}),
    )
    out = pd_cmds.without_unsupported_baseline_flags(state, spec)
    assert out.decode_extra == "--load-balance-method round_robin"
    assert out.prefill_extra == "--load-balance-method round_robin"
    manual = pd_cmds.without_unsupported_baseline_flags(dict(state, baseline_args_source="manual"), spec)
    assert manual == spec


def _run_launch(tmp_path: Path, command: str, env: dict[str, str], *, log_ip: str = "") -> subprocess.CompletedProcess:
    script = tmp_path / "launch.sh"
    script.write_text(pd_internal.launch_script("decode", command, env, log_ip=log_ip), encoding="utf-8")
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, SGLANG_HOST_IP="127.0.0.1"),
    )


def test_engine_launch_tees_its_output_to_the_shared_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(tmp_path / "no-gpu"),))
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
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(device),))
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
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(device),))
    monkeypatch.setattr(pd_internal, "GPU_RELEASE_WAIT_S", 1)
    holder = subprocess.Popen(["python3", "-c", f"import time; f = open({str(device)!r}); time.sleep(30)"])
    try:
        time.sleep(0.3)
        done = _run_launch(tmp_path, "echo launched", {})
    finally:
        holder.kill()
        holder.wait(timeout=10)
    assert done.returncode == 0
    assert "launched" in done.stdout
    assert f"GPU still held by pid(s) {holder.pid}" in done.stderr


def test_engine_launch_ignores_gpu_handles_it_inherited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU handle the replaced engine left open in this process is not another holder."""
    device = tmp_path / "kfd"
    device.write_text("")
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(device),))
    monkeypatch.setattr(pd_internal, "GPU_RELEASE_WAIT_S", 3)
    script = tmp_path / "launch.sh"
    logs = tmp_path / "logs"
    script.write_text(
        pd_internal.launch_script("decode", "echo launched", {"HYPERLOOM_MN_SERVER_LOG_DIR": str(logs)}, log_ip="ip"),
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
    monkeypatch.setattr(pd_internal, "GPU_DEVICE_PATHS", (str(tmp_path / "no-gpu"),))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    script = tmp_path / "launch.sh"
    bind = f"python3 -c \"import socket; socket.socket().bind(('127.0.0.1', {port})); print('bound')\""
    script.write_text(pd_internal.launch_script("decode", bind, {}), encoding="utf-8")
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
