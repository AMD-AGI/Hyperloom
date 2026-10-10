# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight WARN-only check on the kernel phase's GEMM tuning CLIs."""

from __future__ import annotations

from hyperloom.inference_optimizer.cli import preflight


def test_missing_tuning_clis_warn_and_are_recorded(monkeypatch, capsys):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/x" if name == "ckProfiler" else None)

    outcome = preflight._check_kernel_tuning_clis()

    assert outcome["status"] == "warned"
    assert outcome["detail"]["missing"] == ["hipblaslt-bench"]
    out = capsys.readouterr().out
    assert "WARNING — hipblaslt-bench not on PATH" in out
    assert "ckProfiler" not in out


def test_present_tuning_clis_pass_quietly(monkeypatch, capsys):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")

    outcome = preflight._check_kernel_tuning_clis()

    assert outcome == {"status": "applied", "skip_reason": None, "detail": {"missing": []}}
    assert capsys.readouterr().out == ""
