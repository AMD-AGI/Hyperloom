# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared rocm-smi VRAM reader."""

from __future__ import annotations

import json

import pytest

from hyperloom.common import rocm_smi
from hyperloom.common.rocm_smi import GpuVram, gpu_vram_usage

_TOTAL_B = 288 * 1024**3
_TOTAL_MIB = 288 * 1024.0


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


@pytest.fixture(autouse=True)
def _binary_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rocm_smi.shutil, "which", lambda _n: "/opt/rocm/bin/rocm-smi")


def _stub_output(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(rocm_smi.subprocess, "run", lambda *a, **k: _FakeProc(0, text))


def _card(used_b: int) -> dict[str, str]:
    return {
        "VRAM Total Memory (B)": str(_TOTAL_B),
        "VRAM Total Used Memory (B)": str(used_b),
    }


def test_parses_used_and_total_per_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte fields become MiB per card; rows without vram keys are skipped."""
    _stub_output(
        monkeypatch,
        {
            "card0": _card(100 * 1024**2),
            "card1": _card(250 * 1024**2),
            "system": {"Driver version": "6.1.4"},
        },
    )
    assert gpu_vram_usage() == [
        GpuVram(100.0, _TOTAL_MIB),
        GpuVram(250.0, _TOTAL_MIB),
    ]


def test_none_when_a_card_omits_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """Used without total cannot answer a ratio gate, so the probe is unknown."""
    _stub_output(monkeypatch, {"card0": {"VRAM Total Used Memory (B)": str(512 * 1024**2)}})
    assert gpu_vram_usage() is None


def test_none_when_total_is_not_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero total would make the ratio undefined."""
    _stub_output(monkeypatch, {"card0": {"VRAM Total Memory (B)": "0", "VRAM Total Used Memory (B)": "0"}})
    assert gpu_vram_usage() is None


def test_none_when_binary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No rocm-smi on PATH."""
    monkeypatch.setattr(rocm_smi.shutil, "which", lambda _n: None)
    assert gpu_vram_usage() is None


def test_none_on_unusable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-zero exit, blank stdout, bad JSON, non-object JSON, and exec errors."""
    monkeypatch.setattr(rocm_smi.subprocess, "run", lambda *a, **k: _FakeProc(1, "boom"))
    assert gpu_vram_usage() is None

    _stub_output(monkeypatch, "   ")
    assert gpu_vram_usage() is None

    _stub_output(monkeypatch, "{not json")
    assert gpu_vram_usage() is None

    _stub_output(monkeypatch, "[]")
    assert gpu_vram_usage() is None

    monkeypatch.setattr(rocm_smi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no device")))
    assert gpu_vram_usage() is None
