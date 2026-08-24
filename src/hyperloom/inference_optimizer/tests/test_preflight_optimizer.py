# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for inference_optimizer/tools/preflight_optimizer.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hyperloom.common.rocm_smi import GpuVram


def _load_preflight(unique_name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / "preflight_optimizer.py"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_check_gpu_occupancy_passes_when_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU below 1% threshold -> occupancy check returns True."""
    import hyperloom.common.rocm_smi as rsm

    pf = _load_preflight("pf_ok")
    total = 288 * 1024.0
    used = total * 0.005
    monkeypatch.setattr(rsm, "gpu_vram_usage", lambda **_k: [GpuVram(used_mib=used, total_mib=total)])
    assert pf._check_gpu_occupancy() is True


def test_check_gpu_occupancy_fails_when_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU above 1% threshold -> occupancy check returns False."""
    import hyperloom.common.rocm_smi as rsm

    pf = _load_preflight("pf_busy")
    total = 288 * 1024.0
    used = total * 0.05

    monkeypatch.setattr(rsm, "gpu_vram_usage", lambda **_k: [GpuVram(used_mib=used, total_mib=total)])
    result = pf._check_gpu_occupancy()
    assert result is False


def test_check_gpu_occupancy_fails_when_rocm_smi_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rocm-smi unavailable (None) -> occupancy check returns False (fail closed)."""
    import hyperloom.common.rocm_smi as rsm

    pf = _load_preflight("pf_unavail")
    monkeypatch.setattr(rsm, "gpu_vram_usage", lambda **_k: None)
    assert pf._check_gpu_occupancy() is False


def test_main_exits_2_when_gpu_busy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main() returns 2 when GPU occupancy check fails."""
    import hyperloom.common.rocm_smi as rsm

    pf = _load_preflight("pf_main_busy")
    model = tmp_path / "model"
    model.mkdir()

    total = 288 * 1024.0
    used = total * 0.05
    monkeypatch.setattr(rsm, "gpu_vram_usage", lambda **_k: [GpuVram(used_mib=used, total_mib=total)])
    monkeypatch.setattr(pf, "_print_torch_visibility", lambda: True)
    monkeypatch.setattr(pf, "_find_stale_processes", lambda: [])
    import sys as _sys

    monkeypatch.setattr(_sys, "argv", ["preflight_optimizer", str(model)])
    assert pf.main() == 2


def test_main_exits_0_when_all_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main() returns 0 when model path exists, torch ok, GPU clean, no stale PIDs."""
    import hyperloom.common.rocm_smi as rsm

    pf = _load_preflight("pf_main_ok")
    model = tmp_path / "model"
    model.mkdir()

    total = 288 * 1024.0
    used = total * 0.002
    monkeypatch.setattr(rsm, "gpu_vram_usage", lambda **_k: [GpuVram(used_mib=used, total_mib=total)])
    monkeypatch.setattr(pf, "_print_torch_visibility", lambda: True)
    monkeypatch.setattr(pf, "_find_stale_processes", lambda: [])
    import sys as _sys

    monkeypatch.setattr(_sys, "argv", ["preflight_optimizer", str(model)])
    assert pf.main() == 0


def test_main_exits_2_when_rocm_smi_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main() returns 2 when GPU state is unknown (rocm-smi unavailable)."""
    import hyperloom.common.rocm_smi as rsm

    pf = _load_preflight("pf_main_unavail")
    model = tmp_path / "model"
    model.mkdir()

    monkeypatch.setattr(rsm, "gpu_vram_usage", lambda **_k: None)
    monkeypatch.setattr(pf, "_print_torch_visibility", lambda: True)
    monkeypatch.setattr(pf, "_find_stale_processes", lambda: [])
    import sys as _sys

    monkeypatch.setattr(_sys, "argv", ["preflight_optimizer", str(model)])
    assert pf.main() == 2
