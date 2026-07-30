# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight lm_eval accuracy-gate dependency ensure."""

from __future__ import annotations

import subprocess

from hyperloom.inference_optimizer.cli import preflight


def _record_runner(monkeypatch, *, lm_eval_present: bool):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))

        class _R:
            returncode = 0

        r = _R()
        # The probe imports lm_eval; return non-zero when it should be missing.
        if cmd[1:3] == ["-c", "import lm_eval"]:
            r.returncode = 0 if lm_eval_present else 1
        return r

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    return calls


def test_lm_eval_installed_when_missing_and_eval_enabled(monkeypatch):
    monkeypatch.delenv("RUN_EVAL", raising=False)  # default => enabled
    calls = _record_runner(monkeypatch, lm_eval_present=False)
    preflight._ensure_lm_eval_dep("py", ["--break-system-packages"])
    assert any(c[:4] == ["py", "-m", "pip", "install"] and "lm_eval" in c for c in calls)


def test_lm_eval_skipped_when_present(monkeypatch):
    monkeypatch.delenv("RUN_EVAL", raising=False)
    calls = _record_runner(monkeypatch, lm_eval_present=True)
    preflight._ensure_lm_eval_dep("py", [])
    assert not any("pip" in c for c in calls)  # probe only, no install


def test_lm_eval_skipped_when_run_eval_disabled(monkeypatch):
    monkeypatch.setenv("RUN_EVAL", "false")

    def _boom(*_a, **_k):
        raise AssertionError("must not probe/install when RUN_EVAL is disabled")

    monkeypatch.setattr(preflight.subprocess, "run", _boom)
    preflight._ensure_lm_eval_dep("py", [])
