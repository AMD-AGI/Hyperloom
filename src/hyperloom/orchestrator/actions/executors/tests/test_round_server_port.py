# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every server a round boots on this host listens on a free port, profile rounds included."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.common.benchmark_config import read_benchmark_point
from hyperloom.orchestrator.actions.executors import _server_lifecycle as sl

_FREE_PORT = 23456


def _config(path: Path, **benchmark: object) -> Path:
    bench = {"framework": "vllm", "benchmark_script": "vllm_mi300x.sh", "envs": {}, **benchmark}
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _free_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sl, "pick_server_port", lambda: _FREE_PORT)
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", lambda: False)


def test_a_profile_round_boots_on_a_free_port_not_the_launcher_default(tmp_path: Path) -> None:
    # A torch-profiler round cannot reuse a server, so it used to skip the port pick and boot on 8888 -- which a
    # co-tenant on a shared host already held, failing the round with port_in_use before the profiler ever ran.
    path = _config(tmp_path / "profile.yaml", profiler={"torch_profiler": {"enabled": True}})

    info = sl.resolve_lifecycle_params(path)
    sl.pin_server_port(path, info["port"])

    assert info["eligible"] is False
    assert info["local_server"] is True
    assert int(read_benchmark_point(path).envs["PORT"]) == _FREE_PORT


def test_a_reusable_round_keeps_its_free_port(tmp_path: Path) -> None:
    info = sl.resolve_lifecycle_params(_config(tmp_path / "plain.yaml"))

    assert info["eligible"] is True
    assert info["local_server"] is True
    assert info["port"] == _FREE_PORT


def test_a_round_that_boots_no_local_server_gets_no_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._multi_node_env.is_multi_node", lambda: True)

    info = sl.resolve_lifecycle_params(_config(tmp_path / "multi-node.yaml"))

    assert info["local_server"] is False
    assert info["port"] == sl.REUSE_PORT_DEFAULT
