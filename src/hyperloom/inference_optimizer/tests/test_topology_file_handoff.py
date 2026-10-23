# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cluster hand-off through TOPOLOGY_HANDOFF_FILE (TopologyHandoff v1) versus the legacy variables."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.cli.multi_node import _prepare_multi_node_state, _resolve_mn_backend
from hyperloom.inference_optimizer.multi_node._internal import external_state as ext
from hyperloom.inference_optimizer.multi_node._internal import topology_file as topo
from hyperloom.inference_optimizer.multi_node.state_paths import resolve_state_file

_LEGACY_NAMES = topo.COVERED_NAMES + ("PD_MODE", "PD_PREFILL_NODES", "PD_DECODE_NODES")


@pytest.fixture(autouse=True)
def _clean_handoff_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAFE_API_URL", raising=False)
    monkeypatch.delenv(topo.TOPOLOGY_FILE_ENV, raising=False)
    for name in _LEGACY_NAMES:
        monkeypatch.delenv(name, raising=False)


def _write(path: Path, doc: dict[str, Any]) -> Path:
    """Write a document the way the writer does: ``JSON.stringify(handoff, null, 2)`` plus a newline."""
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _rayjob_doc(tmp_path: Path) -> dict[str, Any]:
    token = tmp_path / "ray-token"
    token.write_text("tok-abc123", encoding="utf-8")
    return {
        "schema": "topology-handoff/v1",
        "backend": "rayjob",
        "nodes": 2,
        "gpus_per_node": 8,
        "service_url": "http://10.0.0.12:30000",
        "ray": {"head_ip": "10.0.0.12", "dashboard_token_file": str(token)},
    }


def _infera_doc(tmp_path: Path, *, prefill: list[str], decode: list[str], worker: list[str]) -> dict[str, Any]:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("", encoding="utf-8")
    return {
        "schema": "topology-handoff/v1",
        "backend": "infera",
        "nodes": len(prefill) + len(decode) or len(worker),
        "gpus_per_node": 4,
        "service_url": "http://10.0.1.9:8000",
        "infera": {
            "ssh_key_file": "/workspace/.ssh/id_ed25519",
            "ssh_port_base": 2244,
            "known_hosts_file": str(known_hosts),
            "prefill_ips": prefill,
            "decode_ips": decode,
            "worker_ips": worker,
        },
    }


def _legacy_env(doc: dict[str, Any], tmp_path: Path) -> dict[str, str]:
    """The variables the writer's legacy path sets for the same topology (see applyMultiNodeExternalEnv)."""
    env = {
        "INFERENCE_OPTIMIZER_MN_BACKEND": doc["backend"],
        "INFERENCE_OPTIMIZER_NODES": str(doc["nodes"]),
        "INFERENCE_OPTIMIZER_GPUS_PER_NODE": str(doc["gpus_per_node"]),
        "HYPERLOOM_MN_EXT_SERVICE_URL": doc["service_url"],
    }
    if doc["backend"] == "rayjob":
        env["HYPERLOOM_MN_EXT_HEAD_IP"] = doc["ray"]["head_ip"]
        env["HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN"] = (tmp_path / "ray-token").read_text(encoding="utf-8")
        return env
    infera = doc["infera"]
    env["HYPERLOOM_MN_EXT_SSH_KEY"] = infera["ssh_key_file"]
    env["HYPERLOOM_MN_EXT_SSH_PORT"] = str(infera["ssh_port_base"])
    env["HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS"] = infera["known_hosts_file"]
    for role in ("prefill", "decode", "worker"):
        if infera[f"{role}_ips"]:
            env[f"HYPERLOOM_MN_EXT_{role.upper()}_IPS"] = ",".join(infera[f"{role}_ips"])
    return env


def _state_from_file(monkeypatch: pytest.MonkeyPatch, path: Path, pd_mode: str = "") -> dict[str, Any]:
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    if pd_mode:
        monkeypatch.setenv("PD_MODE", pd_mode)
    return ext.build_external_state_from_env()


def _state_from_legacy(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], pd_mode: str = "") -> dict[str, Any]:
    monkeypatch.delenv(topo.TOPOLOGY_FILE_ENV, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if pd_mode:
        monkeypatch.setenv("PD_MODE", pd_mode)
    return ext.build_external_state_from_env()


def _assert_same_state_item_by_item(from_file: dict[str, Any], from_legacy: dict[str, Any]) -> None:
    assert from_legacy, "the legacy variables produced no state"
    assert sorted(from_file) == sorted(from_legacy)
    for key in from_legacy:
        assert from_file[key] == from_legacy[key], f"state[{key!r}] from the file differs from the legacy variables"


_TOPOLOGIES = {
    "rayjob": ("rayjob", ""),
    "infera_aggregated": ("infera", ""),
    "infera_pd": ("infera", "disaggregated"),
}


@pytest.mark.parametrize("name", sorted(_TOPOLOGIES))
def test_file_resolves_to_what_the_legacy_variables_produce(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend, pd_mode = _TOPOLOGIES[name]
    if backend == "rayjob":
        doc = _rayjob_doc(tmp_path)
    elif pd_mode:
        doc = _infera_doc(tmp_path, prefill=["10.0.1.1", "10.0.1.2"], decode=["10.0.1.3"], worker=[])
    else:
        doc = _infera_doc(tmp_path, prefill=[], decode=[], worker=["10.0.1.4", "10.0.1.5"])
    path = _write(tmp_path / "topology.json", doc)

    from_file = _state_from_file(monkeypatch, path, pd_mode)
    monkeypatch.delenv(topo.TOPOLOGY_FILE_ENV)
    from_legacy = _state_from_legacy(monkeypatch, _legacy_env(doc, tmp_path), pd_mode)

    _assert_same_state_item_by_item(from_file, from_legacy)
    assert from_file["backend"] == backend
    assert from_file["nodes"] == doc["nodes"]
    assert from_file["gpus_per_node"] == doc["gpus_per_node"]
    assert from_file["service_url"] == doc["service_url"]


def test_rayjob_file_fields_land_in_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = _state_from_file(monkeypatch, _write(tmp_path / "t.json", _rayjob_doc(tmp_path)))
    assert state["head_pod_ip"] == "10.0.0.12"
    assert state["ray_address"] == "10.0.0.12:6379"
    assert state["ray_dashboard_token"] == "tok-abc123"
    assert ext.external_has_server_control() is True
    assert ext.external_has_ssh_control() is False


def test_infera_file_fields_land_in_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = _infera_doc(tmp_path, prefill=[], decode=[], worker=["10.0.1.4", "10.0.1.5"])
    state = _state_from_file(monkeypatch, _write(tmp_path / "t.json", doc))
    assert state["ssh_key_path"] == "/workspace/.ssh/id_ed25519"
    assert state["ssh_port"] == 2244
    assert state["ssh_known_hosts"] == str(tmp_path / "known_hosts")
    assert state["worker_pod_ips"] == ["10.0.1.4", "10.0.1.5"]
    assert ext.external_has_ssh_control() is True


def test_file_ignores_legacy_variables_it_does_not_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A stale legacy variable must not leak into a cluster the file describes."""
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_KEY", "/stale/key")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_WORKER_IPS", "9.9.9.9")
    state = _state_from_file(monkeypatch, _write(tmp_path / "t.json", _rayjob_doc(tmp_path)))
    assert "ssh_key_path" not in state
    assert state["worker_pod_ips"] == []
    assert ext.external_has_ssh_control() is False


def test_without_the_file_variable_the_legacy_variables_are_read_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Behavioural pin for today's launches; the whole of test_external_multi_node.py is the wider one."""
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://frontend:8000")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.0.7")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "3")
    assert topo.topology_file_overlay() is None
    state = ext.build_external_state_from_env()
    assert (state["backend"], state["nodes"], state["head_pod_ip"], state["gpus_per_node"]) == (
        "rayjob",
        3,
        "10.0.0.7",
        8,
    )
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, "   ")
    assert topo.topology_file_overlay() is None
    assert ext.build_external_state_from_env() == state


def _set_valid_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://frontend:8000")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.0.0.7")


def test_missing_file_is_an_error_naming_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_valid_legacy_env(monkeypatch)
    missing = tmp_path / "nope.json"
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(missing))
    with pytest.raises(topo.TopologyHandoffError, match="cannot read the file") as caught:
        ext.build_external_state_from_env()
    assert str(missing) in str(caught.value)
    with pytest.raises(topo.TopologyHandoffError):
        ext.external_service_url()


def test_bad_json_is_an_error_naming_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_valid_legacy_env(monkeypatch)
    path = tmp_path / "topology.json"
    path.write_text('{"schema": "topology-handoff/v1", ', encoding="utf-8")
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    with pytest.raises(topo.TopologyHandoffError, match="not valid JSON") as caught:
        ext.build_external_state_from_env()
    assert str(path) in str(caught.value)


@pytest.mark.parametrize("schema", ["topology-handoff/v2", "topology-handoff/v0", "other/v1", "vendor.topology-handoff/v1", None])
def test_unknown_schema_is_refused(schema: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_valid_legacy_env(monkeypatch)
    doc = _rayjob_doc(tmp_path)
    if schema is None:
        del doc["schema"]
    else:
        doc["schema"] = schema
    path = _write(tmp_path / "topology.json", doc)
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    expected = "unsupported schema major" if schema and schema.startswith("topology-handoff/") else "is not a topology-handoff"
    with pytest.raises(topo.TopologyHandoffError, match=expected) as caught:
        ext.external_service_url()
    assert str(path) in str(caught.value)


def test_a_minor_of_the_supported_major_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    doc = _rayjob_doc(tmp_path)
    doc["schema"] = "topology-handoff/v1.2"
    doc["a_future_field"] = True
    assert _state_from_file(monkeypatch, _write(tmp_path / "t.json", doc))["backend"] == "rayjob"


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda d: d.update(backend="slurm"), "backend 'slurm'"),
        (lambda d: d.update(nodes=0), "nodes must be an integer"),
        (lambda d: d.update(nodes=True), "nodes must be an integer"),
        (lambda d: d.update(gpus_per_node="8"), "gpus_per_node must be an integer"),
        (lambda d: d.update(service_url="frontend:8000"), "service_url must be an http"),
        (lambda d: d.update(ray=["x"]), "ray must be an object"),
        (lambda d: d["ray"].update(head_ip=""), "ray.head_ip must be a non-empty string"),
        (lambda d: d["ray"].update(dashboard_token_file="/nonexistent/ray-token"), "cannot be read"),
    ],
)
def test_a_malformed_field_is_refused_not_half_used(
    mutate: Any, problem: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_valid_legacy_env(monkeypatch)
    doc = _rayjob_doc(tmp_path)
    mutate(doc)
    path = _write(tmp_path / "topology.json", doc)
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(path))
    with pytest.raises(topo.TopologyHandoffError, match=problem) as caught:
        ext.build_external_state_from_env()
    assert str(path) in str(caught.value)


def test_infera_without_its_block_or_with_a_bad_port_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    doc = _infera_doc(tmp_path, prefill=[], decode=[], worker=["10.0.1.4"])
    doc["infera"]["ssh_port_base"] = 70000
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(_write(tmp_path / "a.json", doc)))
    with pytest.raises(topo.TopologyHandoffError, match="infera.ssh_port_base"):
        ext.build_external_state_from_env()
    del doc["infera"]
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(_write(tmp_path / "b.json", doc)))
    with pytest.raises(topo.TopologyHandoffError, match="needs an 'infera' block"):
        ext.build_external_state_from_env()


def test_disagreement_with_legacy_variables_warns_naming_the_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    doc = _rayjob_doc(tmp_path)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "4")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_HEAD_IP", "10.9.9.9")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", doc["service_url"])  # agrees: must not be named
    monkeypatch.setenv("HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN", "a-different-secret")
    with caplog.at_level(logging.WARNING, logger=topo.log.name):
        state = _state_from_file(monkeypatch, _write(tmp_path / "t.json", doc))
    messages = [r.getMessage() for r in caplog.records if r.name == topo.log.name]
    assert len(messages) == 1
    assert "nodes (INFERENCE_OPTIMIZER_NODES)" in messages[0]
    assert "ray.head_ip (HYPERLOOM_MN_EXT_HEAD_IP)" in messages[0]
    assert "ray.dashboard_token_file (HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN)" in messages[0]
    assert "service_url" not in messages[0]
    assert "a-different-secret" not in messages[0]
    assert "tok-abc123" not in messages[0]
    assert (state["nodes"], state["head_pod_ip"]) == (2, "10.0.0.12"), "the file must win"

    caplog.clear()
    ext.build_external_state_from_env()
    assert not [r for r in caplog.records if r.name == topo.log.name], "the same disagreement warns once"


def test_agreeing_legacy_variables_do_not_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    doc = _rayjob_doc(tmp_path)
    for name, value in _legacy_env(doc, tmp_path).items():
        monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING, logger=topo.log.name):
        _state_from_file(monkeypatch, _write(tmp_path / "t.json", doc))
    assert not [r for r in caplog.records if r.name == topo.log.name]


def _optimize_args(**overrides: Any) -> argparse.Namespace:
    values = {"nodes": 2, "mn_backend": None, "mn_image": None, "model": "/models/test", "no_kernel": True}
    return argparse.Namespace(**{**values, **overrides})


def _session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session))
    monkeypatch.setenv("BENCHMARK_BASE_URL", "")  # restored by monkeypatch; the launch sets it
    monkeypatch.setenv("MAGPIE_RUN_PHASE", "")


def test_the_files_backend_is_the_launch_backend_when_no_flag_or_variable_says_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """optimize --nodes 2 without --mn-backend must not turn an infera file into rayjob."""
    doc = _infera_doc(tmp_path, prefill=[], decode=[], worker=["10.0.1.4", "10.0.1.5"])
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(_write(tmp_path / "t.json", doc)))
    assert _resolve_mn_backend(_optimize_args()) == "infera"
    assert _resolve_mn_backend(_optimize_args(mn_backend="rayjob")) == "rayjob"  # an explicit flag still wins

    _session(monkeypatch, tmp_path)
    _prepare_multi_node_state(_optimize_args())
    saved = json.loads(resolve_state_file().read_text(encoding="utf-8"))
    assert (saved["backend"], saved["worker_pod_ips"], saved["external"]) == ("infera", ["10.0.1.4", "10.0.1.5"], True)


def test_launch_with_an_unusable_topology_file_exits_2_naming_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_valid_legacy_env(monkeypatch)
    _session(monkeypatch, tmp_path)
    bad = tmp_path / "topology.json"
    bad.write_text("{", encoding="utf-8")
    monkeypatch.setenv(topo.TOPOLOGY_FILE_ENV, str(bad))
    with pytest.raises(SystemExit) as caught:
        _prepare_multi_node_state(_optimize_args())
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert str(bad) in err
    assert "not valid JSON" in err
    with pytest.raises(SystemExit) as caught_backend:
        _resolve_mn_backend(_optimize_args())
    assert caught_backend.value.code == 2
