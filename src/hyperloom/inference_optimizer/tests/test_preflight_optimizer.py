# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the launcher-side preflight tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from hyperloom.common import rocm_smi
from hyperloom.common.rocm_smi import GpuVram

_TOTAL_MIB = 288 * 1024.0


@pytest.fixture
def preflight() -> ModuleType:
    """Load the tool by path; it ships as an operator script, not as a module."""
    path = Path(__file__).resolve().parents[1] / "tools" / "preflight_optimizer.py"
    spec = importlib.util.spec_from_file_location("_preflight_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _usage(*used_fractions: float) -> list[GpuVram]:
    return [GpuVram(_TOTAL_MIB * f, _TOTAL_MIB) for f in used_fractions]


def test_occupancy_idle_below_fraction(preflight: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every GPU under the fraction reports idle."""
    monkeypatch.setattr(rocm_smi, "gpu_vram_usage", lambda: _usage(0.005, 0.001))
    assert preflight._check_gpu_occupancy() is True


def test_occupancy_busy_when_one_gpu_exceeds_fraction(preflight: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """One GPU over the fraction fails the whole check."""
    monkeypatch.setattr(rocm_smi, "gpu_vram_usage", lambda: _usage(0.001, 0.05))
    assert preflight._check_gpu_occupancy() is False


def test_occupancy_fails_when_unreadable(preflight: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown GPU state is a failure, not a pass."""
    monkeypatch.setattr(rocm_smi, "gpu_vram_usage", lambda: None)
    assert preflight._check_gpu_occupancy() is False


def test_main_propagates_busy_gpu_to_exit_code(
    preflight: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A busy GPU must reach the exit code, not just stdout."""
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(rocm_smi, "gpu_vram_usage", lambda: _usage(0.05))
    monkeypatch.setattr(preflight, "_print_torch_visibility", lambda: True)
    monkeypatch.setattr(preflight, "_find_stale_processes", lambda: [])
    monkeypatch.setattr(sys, "argv", ["preflight_optimizer", str(model)])
    assert preflight.main() == 2


def test_main_returns_zero_when_every_check_passes(
    preflight: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Model path present, torch visible, GPUs idle, no stale processes."""
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(rocm_smi, "gpu_vram_usage", lambda: _usage(0.002))
    monkeypatch.setattr(preflight, "_print_torch_visibility", lambda: True)
    monkeypatch.setattr(preflight, "_find_stale_processes", lambda: [])
    monkeypatch.setattr(sys, "argv", ["preflight_optimizer", str(model)])
    assert preflight.main() == 0
