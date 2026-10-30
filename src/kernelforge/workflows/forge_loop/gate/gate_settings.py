# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an in-session gate measures and what it protects, independent of the gate that enforces it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB

# Max times the Stop hook will BLOCK a stop for an unrestored protected-harness change before it gives up, allows the
# stop, and hands off to the outer loop's force-REVERT.
MAX_STOP_BLOCKS = 3


@dataclass(frozen=True)
class GateMeasurement:
    """How the gate judges a candidate: its driver (and any interposed wrapper), threshold, budgets and timeouts."""

    driver_script: str | None = None
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB
    max_blocks: int = 10
    stage_timeout_sec: int = 1800
    bench_timeout_sec: int = 300
    bench_repeat: int = 1
    correctness_only: bool = False
    interposed_driver_path: str | None = None
    max_stop_blocks: int = MAX_STOP_BLOCKS


@dataclass(frozen=True)
class GateScope:
    """What one session works on and what the gate refuses to let it touch beyond the driver."""

    kernel_file: str = ""
    target_files: list[str] | None = None
    extra_protected_globs: list[str] | None = None
    extra_protected_paths: list[str] | None = None
    workspace: str | Path | None = None
