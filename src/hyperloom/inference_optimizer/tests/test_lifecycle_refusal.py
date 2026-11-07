# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""server_lifecycle eligibility: only scriptable workloads, multi-node and profiling runs are refused."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _multi_node_env
from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

_PORT = 8888


@pytest.fixture(autouse=True)
def _fixed_port(monkeypatch):
    monkeypatch.setattr(sl, "pick_server_port", lambda: _PORT)


def _resolve(tmp_path: Path, bench: dict) -> dict:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"benchmark": {"envs": {"PORT": _PORT}, **bench}}), encoding="utf-8")
    return sl.resolve_lifecycle_params(cfg_path)


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
@pytest.mark.parametrize("script", ["", "custom_thing.sh", "vllm_gfx12.sh"])
def test_serving_framework_is_eligible_whatever_its_script(tmp_path, framework, script):
    info = _resolve(tmp_path, {"framework": framework, "benchmark_script": script})
    assert info["eligible"] is True, info["reason"]
    assert info["framework"] == framework
    assert info["port"] == _PORT
    assert info["reason"] == ""


@pytest.mark.parametrize(
    "bench",
    [
        {"framework": "xdit"},
        {"framework": "custom"},
        {"framework": "sglang", "workload_kind": "Scriptable"},
    ],
)
def test_scriptable_workload_is_refused(tmp_path, bench):
    info = _resolve(tmp_path, bench)
    assert info["eligible"] is False
    assert "scriptable" in info["reason"]


def test_multi_node_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(_multi_node_env, "is_multi_node", lambda: True)
    info = _resolve(tmp_path, {"framework": "vllm"})
    assert info["eligible"] is False
    assert "multi-node" in info["reason"]


def test_torch_profiler_is_refused(tmp_path):
    info = _resolve(tmp_path, {"framework": "vllm", "profiler": {"torch_profiler": {"enabled": True}}})
    assert info["eligible"] is False
    assert "torch_profiler" in info["reason"]


def test_refusal_does_not_read_the_benchmark_backend():
    assert "benchmark_backend" not in Path(sl.__file__).read_text(encoding="utf-8")


def test_lifecycle_port_is_below_the_ephemeral_range(tmp_path, monkeypatch):
    """The persistent server binds minutes after its port is picked; an ephemeral one is taken by then."""
    from hyperloom.common import ports

    monkeypatch.setattr(sl, "pick_server_port", ports.pick_server_port)
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text(encoding="utf-8").split()[0])

    port = _resolve(tmp_path, {"framework": "vllm", "benchmark_script": "vllm_mi300x.sh"})["port"]

    assert port < ephemeral_start and port not in ports.DEV_SANDBOX_PORTS and port not in ports.RAY_WORKER_PORTS
