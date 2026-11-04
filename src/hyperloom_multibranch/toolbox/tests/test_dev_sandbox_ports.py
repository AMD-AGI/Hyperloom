# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one thing the dev sandboxes and local measurement must agree on: the block of ports that is the sandboxes'."""

from __future__ import annotations

from hyperloom.measure.ports import DEV_SANDBOX_PORTS
from hyperloom_multibranch.specialist.dev_sandbox import DevLimits


def test_the_ports_a_dev_sandbox_slot_serves_on_are_the_block_measurement_leaves_alone() -> None:
    limits = DevLimits()
    assert limits.port == DEV_SANDBOX_PORTS.start, "slot 0 starts where the reserved block does"
    assert limits.port + 100 * limits.ports == DEV_SANDBOX_PORTS.stop, "a hundred slots fill the block"
