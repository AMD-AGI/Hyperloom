# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The port local measurement gives a test's server: below the ephemeral range, never in a dev sandbox's block."""

from __future__ import annotations

from pathlib import Path

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
