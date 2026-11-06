# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The port local measurement gives a test's server: below the ephemeral range, never in a dev sandbox's block."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.common.ports import DEV_SANDBOX_PORTS, RAY_WORKER_PORTS
from hyperloom.measure import local


def _point(tmp_path: Path) -> Path:
    path = tmp_path / "point.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"TP": "1"}}}), encoding="utf-8")
    return path


def test_a_round_without_a_lifecycle_serves_below_the_ephemeral_range(tmp_path: Path) -> None:
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text(encoding="utf-8").split()[0])
    with local._free_port_for(_point(tmp_path), None) as pinned:
        port = yaml.safe_load(pinned.read_text(encoding="utf-8"))["benchmark"]["envs"]["PORT"]
    assert port < ephemeral_start and port not in DEV_SANDBOX_PORTS and port not in RAY_WORKER_PORTS


def _lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, picked: int) -> Any:
    info = {"eligible": True, "port": picked, "framework": "sglang"}
    monkeypatch.setattr(local, "resolve_lifecycle_params", lambda _point: info)
    call = SimpleNamespace(output_dir=str(tmp_path), points=[_point(tmp_path)])
    return local._Lifecycle(SimpleNamespace(call=call, phases=None))


def test_a_lifecycle_whose_port_fell_in_the_reserved_block_serves_on_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local, "pick_server_port", lambda: 25678)
    lifecycle = _lifecycle(tmp_path, monkeypatch, picked=32105)
    assert (lifecycle.port, lifecycle.settings(cleanup=True)["port"]) == (25678, 25678)


def test_a_lifecycle_port_outside_the_block_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "pick_server_port", lambda: pytest.fail("a port outside the block is not replaced"))
    assert _lifecycle(tmp_path, monkeypatch, picked=8123).port == 8123
