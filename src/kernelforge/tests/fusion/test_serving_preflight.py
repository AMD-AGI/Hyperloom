# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A card someone else is holding is not a verdict about the kernel.

Four smoke failures across three runs were allocator errors raised before the
server finished starting -- 283 of 288 GiB were already allocated by a process
from an earlier stage. Each was recorded as "KERNEL OK but SERVING CRASHED
(CUDA-graph-ON decode)" and spent the run's remaining attempts re-authoring a
kernel that had never run. One of the three was the model with the most
predicted headroom in the whole set.
"""

from __future__ import annotations

import pytest

from kernelforge.fusion import validate
from kernelforge.fusion.validate import _free_vram_fraction, gpu_is_free_enough

BUSY = """
GPU[0]		: GPU Memory Allocated (VRAM%): 98
GPU[1]		: GPU Memory Allocated (VRAM%): 3
"""

IDLE = """
GPU[0]		: GPU Memory Allocated (VRAM%): 0
"""


class _Out:
    def __init__(self, text: str) -> None:
        self.stdout = text


def test_a_card_someone_else_is_holding_is_reported(tmp_path=None) -> None:
    ok, reason = gpu_is_free_enough("0", _probe=lambda gpu: 0.02)

    assert ok is False
    assert "still holding the card" in reason


def test_an_idle_card_passes() -> None:
    assert gpu_is_free_enough("0", _probe=lambda gpu: 1.0) == (True, "")


def test_an_unreadable_card_is_not_treated_as_busy() -> None:
    # No rocm-smi is a reason to say nothing, not a reason to block the run.
    assert gpu_is_free_enough("0", _probe=lambda gpu: None) == (True, "")


def test_the_probe_reads_the_requested_gpu() -> None:
    assert _free_vram_fraction("0", _run=lambda cmd: _Out(BUSY)) == pytest.approx(0.02)
    assert _free_vram_fraction("1", _run=lambda cmd: _Out(BUSY)) == pytest.approx(0.97)


def test_an_out_of_range_gpu_falls_back_to_the_last_one() -> None:
    assert _free_vram_fraction("7", _run=lambda cmd: _Out(IDLE)) == pytest.approx(1.0)


def test_unparsable_output_reads_as_unknown() -> None:
    assert _free_vram_fraction("0", _run=lambda cmd: _Out("no such tool")) is None


def test_a_probe_that_raises_reads_as_unknown() -> None:
    def boom(cmd):
        raise OSError("rocm-smi not found")

    assert _free_vram_fraction("0", _run=boom) is None


def test_the_cleanup_kills_only_this_users_engines(monkeypatch) -> None:
    """``VLLM::EngineCore`` names an engine, not a run.

    The card is checked because someone else's process can be holding it. The
    same reasoning applies to the cleanup that runs moments later: on a shared
    validation host an unrestricted pkill against that pattern reaps the very
    colleague whose run the preflight was there to notice.
    """
    if not hasattr(validate.os, "getuid"):
        pytest.skip("no POSIX uid on this platform")
    seen: list[str] = []
    monkeypatch.setattr(
        validate.subprocess,
        "run",
        lambda cmd, **kw: seen.append(cmd) or _Out(""),
    )

    validate._pkill("VLLM::EngineCore")

    assert seen == ["pkill -9 -u %d -f 'VLLM::EngineCore'" % validate.os.getuid()]
