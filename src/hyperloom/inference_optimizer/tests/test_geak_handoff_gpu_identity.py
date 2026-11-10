# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The GEAK handoff carries the explicit R9700 product/ISA pair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.actions.executors._gpu_pin import _resolve_handoff_gpu_identity
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.parametrize(
    ("gpu_type", "expected"),
    [
        ("r9700", {"gpu_type": "r9700", "expected_target": "r9700", "expected_gfx": "gfx1201"}),
        ("mi300x", {"gpu_type": "mi300x"}),
        ("mi355x", {"gpu_type": "mi355x"}),
        ("gfx1201", {"gpu_type": "gfx1201"}),
    ],
)
def test_handoff_gpu_identity(gpu_type, expected) -> None:
    assert _resolve_handoff_gpu_identity(gpu_type) == expected


async def _write_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: SharedState) -> dict:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = state
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.delenv("GPU_TYPE", raising=False)

    def stop_after_handoff() -> None:
        raise RuntimeError("stop after handoff write")

    # The timeline is the first thing the GEAK phase touches once handoff.json is on disk.
    monkeypatch.setattr(coord.phase_kernel, "timeline", stop_after_handoff)
    with pytest.raises(RuntimeError, match="stop after handoff write"):
        await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")
    return json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_r9700_handoff_carries_product_and_isa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handoff = await _write_handoff(tmp_path, monkeypatch, SharedState(model_path="/models/m", gpu_type="r9700"))
    assert handoff["gpu_type"] == "r9700"
    assert handoff["expected_target"] == "r9700"
    assert handoff["expected_gfx"] == "gfx1201"
