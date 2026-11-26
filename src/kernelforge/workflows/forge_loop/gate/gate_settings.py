# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an in-session gate measures and what it protects, independent of the gate that enforces it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB, CaseAggregate

# Max times the Stop hook will BLOCK a stop for an unrestored protected-harness change before it gives up, allows the
# stop, and hands off to the outer loop's force-REVERT.
MAX_STOP_BLOCKS = 3


@dataclass(frozen=True)
class DeviceAccess:
    """How a session that shares its device with other lanes reaches it.

    The session sees no device of its own: its commands go through ``runner``, which queues at ``queue_root``. The
    gate's stop check queues there too, and runs the driver under ``environment`` -- the lane's own build caches --
    because the check runs in the loop's process rather than the session's. A command that still holds the device
    after ``hold_limit_sec`` is stopped by the runner as hung.
    """

    runner: str
    queue_root: str
    hold_limit_sec: int
    usage_log: str = ""
    lane: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GateMeasurement:
    """How the gate judges a candidate: its driver, threshold, budgets and timeouts, and how it reaches the device."""

    driver_script: str | None = None
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB
    max_blocks: int = 10
    stage_timeout_sec: int = 1800
    bench_timeout_sec: int = 300
    bench_repeat: int = 1
    case_aggregate: CaseAggregate = CaseAggregate.ARITHMETIC
    correctness_only: bool = False
    max_stop_blocks: int = MAX_STOP_BLOCKS
    # Set for a lane, whose device is shared; None for a session that has the device to itself.
    device: DeviceAccess | None = None
    # Whether a stop needs a candidate that beats the current best. A lane whose round has other lanes stops on a
    # correct candidate: what it gained on some cases is for the merge to use, whatever its score.
    gain_required: bool = True


@dataclass(frozen=True)
class GateScope:
    """What one session works on and what the gate refuses to let it touch beyond the driver."""

    kernel_file: str = ""
    target_files: list[str] | None = None
    extra_protected_globs: list[str] | None = None
    extra_protected_paths: list[str] | None = None
    workspace: str | Path | None = None
