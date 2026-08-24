# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hyperloom.common.rocm_smi."""

from __future__ import annotations

import json

import pytest

from hyperloom.common.rocm_smi import GpuVram, gpu_vram_usage


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


_CARD0_BOTH = {
    "VRAM Total Memory (B)": str(int(288 * 1024**3)),
    "VRAM Total Used Memory (B)": str(100 * 1024 * 1024),
}
_CARD1_USED_ONLY = {
    "VRAM Total Used Memory (B)": str(250 * 1024 * 1024),
}
_SYSTEM_ROW = {"Driver version": "6.1.4"}


def test_gpu_vram_usage_parses_used_and_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both used and total are parsed; non-card top-level dicts are ignored."""
    import hyperloom.common.rocm_smi as mod

    payload = {
        "card0": _CARD0_BOTH,
        "card1": _CARD1_USED_ONLY,
        "system": _SYSTEM_ROW,
    }
    monkeypatch.setattr(mod.shutil, "which", lambda _n: "/opt/rocm/bin/rocm-smi")
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: _FakeProc(0, json.dumps(payload))
    )

    result = gpu_vram_usage()
    assert result is not None
    assert len(result) == 2

    assert result[0].used_mib == pytest.approx(100.0)
    assert result[0].total_mib == pytest.approx(288 * 1024.0)

    assert result[1].used_mib == pytest.approx(250.0)
    assert result[1].total_mib is None


def test_gpu_vram_usage_total_absent_does_not_block_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card with no total still yields a GpuVram with total_mib=None."""
    import hyperloom.common.rocm_smi as mod

    payload = {"card0": {"VRAM Total Used Memory (B)": str(512 * 1024 * 1024)}}
    monkeypatch.setattr(mod.shutil, "which", lambda _n: "/opt/rocm/bin/rocm-smi")
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: _FakeProc(0, json.dumps(payload))
    )

    result = gpu_vram_usage()
    assert result is not None
    assert result[0] == GpuVram(used_mib=512.0, total_mib=None)


def test_gpu_vram_usage_none_when_rocm_smi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing rocm-smi binary returns None."""
    import hyperloom.common.rocm_smi as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _n: None)
    assert gpu_vram_usage() is None


def test_gpu_vram_usage_none_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-zero rocm-smi exit code returns None."""
    import hyperloom.common.rocm_smi as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _n: "/opt/rocm/bin/rocm-smi")
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: _FakeProc(1, "GPU error")
    )
    assert gpu_vram_usage() is None


def test_gpu_vram_usage_none_on_empty_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty rocm-smi stdout returns None."""
    import hyperloom.common.rocm_smi as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _n: "/opt/rocm/bin/rocm-smi")
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: _FakeProc(0, "   ")
    )
    assert gpu_vram_usage() is None


def test_gpu_vram_usage_none_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unparseable JSON returns None."""
    import hyperloom.common.rocm_smi as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _n: "/opt/rocm/bin/rocm-smi")
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: _FakeProc(0, "{not valid json")
    )
    assert gpu_vram_usage() is None


def test_gpu_vram_usage_none_on_exec_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OSError during subprocess.run returns None."""
    import hyperloom.common.rocm_smi as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _n: "/opt/rocm/bin/rocm-smi")
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no device"))
    )
    assert gpu_vram_usage() is None
