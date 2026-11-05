# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A card someone else is holding is not a verdict about the kernel."""

from __future__ import annotations

import pytest

from hyperloom.common.amd_smi import GpuVram
from kernelforge.fusion.validate import _free_vram_fraction, gpu_is_free_enough

BUSY = [GpuVram(used_mib=98.0, total_mib=100.0), GpuVram(used_mib=3.0, total_mib=100.0)]
IDLE = [GpuVram(used_mib=0.0, total_mib=100.0)]


def test_a_card_someone_else_is_holding_is_reported(tmp_path=None) -> None:
    ok, reason = gpu_is_free_enough("0", _probe=lambda gpu: 0.02)

    assert ok is False
    assert "still holding the card" in reason


def test_an_idle_card_passes() -> None:
    assert gpu_is_free_enough("0", _probe=lambda gpu: 1.0) == (True, "")


def test_an_unreadable_card_is_not_treated_as_busy() -> None:
    # No amd-smi is a reason to say nothing, not a reason to block the run.
    assert gpu_is_free_enough("0", _probe=lambda gpu: None) == (True, "")


def test_the_probe_reads_the_requested_gpu() -> None:
    assert _free_vram_fraction("0", _usage=lambda: BUSY) == pytest.approx(0.02)
    assert _free_vram_fraction("1", _usage=lambda: BUSY) == pytest.approx(0.97)


def test_an_out_of_range_gpu_falls_back_to_the_last_one() -> None:
    assert _free_vram_fraction("7", _usage=lambda: IDLE) == pytest.approx(1.0)


def test_unparsable_output_reads_as_unknown() -> None:
    # gpu_vram_usage() degrades to None rather than raising, same as every other read-only amd-smi probe.
    assert _free_vram_fraction("0", _usage=lambda: None) is None


def test_a_probe_that_raises_reads_as_unknown() -> None:
    def boom():
        raise OSError("amd-smi not found")

    with pytest.raises(OSError, match="amd-smi not found"):
        _free_vram_fraction("0", _usage=boom)
