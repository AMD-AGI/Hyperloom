# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The port a round's server gets when its point names none: below the ephemeral range, never in a dev sandbox's
block. Every local measurement round reads its point through ``_suite.rounds._round_point``."""

from __future__ import annotations

from pathlib import Path

import yaml

from hyperloom.common.ports import DEV_SANDBOX_PORTS
from hyperloom.orchestrator.actions.executors._suite import rounds as suite_rounds


def _point(tmp_path: Path) -> Path:
    path = tmp_path / "point.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"TP": "1"}}}), encoding="utf-8")
    return path


def test_a_round_without_a_lifecycle_serves_below_the_ephemeral_range(tmp_path: Path) -> None:
    ephemeral_start = int(Path("/proc/sys/net/ipv4/ip_local_port_range").read_text(encoding="utf-8").split()[0])

    point = suite_rounds._round_point(_point(tmp_path), 1)

    port = int(point["benchmark"]["envs"]["PORT"])
    assert port < ephemeral_start and port not in DEV_SANDBOX_PORTS
