# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight GPU-visibility check external-mode skip."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import preflight


@pytest.fixture(autouse=True)
def _clear_safe_and_ext(monkeypatch):
    # Each test sets the hand-off itself; SAFE_API_* are cleared only to keep a
    # developer's real credentials out of the run, they do not affect the mode.
    for key in ("SAFE_API_URL", "SAFE_API_KEY", "HYPERLOOM_MN_EXT_SERVICE_URL"):
        monkeypatch.delenv(key, raising=False)


def test_external_mode_skips_local_gpu_probe(monkeypatch, capsys):
    # In external multi-node mode the GPUs live on remote pods, so the local
    # rocm-smi probe must be skipped instead of warning "0 GPUs".
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SERVICE_URL", "http://claw-rayjob:8000")

    def _boom(*_args, **_kwargs):
        raise AssertionError("rocm-smi must not run in external mode")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)

    preflight._check_gpu_visibility()

    out = capsys.readouterr().out
    assert "external multi-node mode" in out
    assert "benchmark will fail" not in out


def test_local_mode_still_warns_on_zero_gpus(monkeypatch, capsys):
    # Without an external URL the probe runs; a 0-GPU result keeps warning.
    class _Proc:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, **_k: _Proc())

    preflight._check_gpu_visibility()

    out = capsys.readouterr().out
    assert "rocm-smi sees 0 GPUs" in out
