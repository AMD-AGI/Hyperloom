# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight GPU-visibility check external-mode skip.

``_check_gpu_visibility`` reads ``amd-smi`` through ``hyperloom.common.amd_smi`` (``amd_smi_available`` /
``amd_smi_json``), not via a subprocess call of its own, so every scenario here stubs that module rather than
``preflight.subprocess`` -- that stub stopped being on the call path when the check was ported off ``rocm-smi``, and
a version of this file that still patched it was silently not exercising the function at all (two of its three cases
asserted on output that can only appear if the real ``amd-smi`` on the host happened to be invoked).
"""

from __future__ import annotations

import pytest

from hyperloom.common import amd_smi
from hyperloom.inference_optimizer.cli import preflight


@pytest.fixture(autouse=True)
def _clear_safe_and_ext(monkeypatch):
    # Each test sets the hand-off itself; SAFE_API_* are cleared only to keep a developer's real credentials out of
    # the run, they do not affect the mode.
    for key in (
        "SAFE_API_URL",
        "HYPERLOOM_MN_EXT_SERVICE_URL",
        "INFERENCE_OPTIMIZER_NODES",
    ):
        monkeypatch.delenv(key, raising=False)


def _refuse_probe(monkeypatch):
    """Make any amd-smi call an error, so a skip can be proven."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("amd-smi must not run here")

    monkeypatch.setattr(amd_smi, "amd_smi_available", _boom)
    monkeypatch.setattr(amd_smi, "amd_smi_json", _boom)


def _stub_amd_smi(monkeypatch, gpu_count: int) -> None:
    """Make amd-smi report ``gpu_count`` devices to ``amd-smi list``."""
    monkeypatch.setattr(amd_smi, "amd_smi_available", lambda: True)
    monkeypatch.setattr(amd_smi, "amd_smi_json", lambda *_a, **_k: [{"gpu": i} for i in range(gpu_count)])


def test_external_mode_skips_local_gpu_probe(monkeypatch, capsys):
    # In external multi-node mode the GPUs live on remote pods, so the local amd-smi probe must be skipped instead of
    # warning "0 GPUs".
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://claw-rayjob:8000")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    _refuse_probe(monkeypatch)

    result = preflight._check_gpu_visibility()

    out = capsys.readouterr().out
    assert "external multi-node mode" in out
    assert "benchmark will fail" not in out
    assert result["skip_reason"] == "external_multi_node"


def test_single_node_still_probes_despite_a_stray_handoff_url(monkeypatch, capsys):
    """A leaked HYPERLOOM_MN_EXT_* must not disarm the check on a single-node run."""
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://claw-rayjob:8000")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "1")
    _stub_amd_smi(monkeypatch, gpu_count=0)

    preflight._check_gpu_visibility()

    out = capsys.readouterr().out
    assert "amd-smi sees 0 GPUs" in out
    assert "external multi-node mode" not in out


def test_local_mode_still_warns_on_zero_gpus(monkeypatch, capsys):
    # Without an external URL the probe runs; a 0-GPU result keeps warning.
    _stub_amd_smi(monkeypatch, gpu_count=0)

    result = preflight._check_gpu_visibility()

    out = capsys.readouterr().out
    assert "amd-smi sees 0 GPUs" in out
    assert result == {
        "status": "warned",
        "skip_reason": None,
        "detail": {"visible": 0, "tp_requested": 1, "warn": "amd-smi sees 0 GPUs; benchmark will fail"},
    }


def test_tp_larger_than_visible_warns_without_claiming_zero(monkeypatch, capsys):
    """A real but insufficient GPU count gets its own warning, distinct from the 0-GPU case."""
    monkeypatch.setenv("TP", "4")
    _stub_amd_smi(monkeypatch, gpu_count=1)

    result = preflight._check_gpu_visibility()

    out = capsys.readouterr().out
    assert "TP=4 but amd-smi sees 1 GPU(s)" in out
    assert "amd-smi sees 0 GPUs" not in out
    assert result["status"] == "warned"
    assert result["detail"] == {"visible": 1, "tp_requested": 4, "warn": result["detail"]["warn"]}


def test_tp_within_visible_count_applies_quietly(monkeypatch, capsys):
    """Enough GPUs for the requested TP prints nothing and reports "applied"."""
    monkeypatch.setenv("TP", "2")
    _stub_amd_smi(monkeypatch, gpu_count=2)

    result = preflight._check_gpu_visibility()

    out = capsys.readouterr().out
    assert out == ""
    assert result == {
        "status": "applied",
        "skip_reason": None,
        "detail": {"visible": 2, "tp_requested": 2, "warn": None},
    }


def test_amd_smi_unavailable_skips_silently(monkeypatch, capsys):
    """No amd-smi on PATH degrades to a silent skip, per the function's own contract."""
    monkeypatch.setattr(amd_smi, "amd_smi_available", lambda: False)

    def _boom(*_a, **_k):
        raise AssertionError("amd_smi_json must not run when amd-smi is not on PATH")

    monkeypatch.setattr(amd_smi, "amd_smi_json", _boom)

    result = preflight._check_gpu_visibility()

    assert capsys.readouterr().out == ""
    assert result["skip_reason"] == "amd_smi_unavailable"


def test_amd_smi_failure_skips_silently(monkeypatch, capsys):
    """A probe that is on PATH but fails (driver absent, timeout, bad JSON) also degrades silently, not loudly."""
    monkeypatch.setattr(amd_smi, "amd_smi_available", lambda: True)

    def _raise(*_a, **_k):
        raise amd_smi.AmdSmiError("amd-smi list exited 1")

    monkeypatch.setattr(amd_smi, "amd_smi_json", _raise)

    result = preflight._check_gpu_visibility()

    assert capsys.readouterr().out == ""
    assert result["skip_reason"] == "amd_smi_failed"
