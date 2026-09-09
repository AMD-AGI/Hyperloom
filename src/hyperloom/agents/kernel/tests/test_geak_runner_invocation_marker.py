# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK is told who invoked it, because it also runs standalone."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hyperloom.agents.kernel.tools.backends import geak_runner


def _fake_popen(captured: dict) -> object:
    def popen(cmd, **kwargs):  # noqa: ANN001, ANN003
        captured["env"] = kwargs.get("env") or {}
        return SimpleNamespace(
            pid=1,
            returncode=0,
            communicate=lambda timeout=None: ("", ""),
        )

    return popen


def test_geak_is_told_that_hyperloom_invoked_it(monkeypatch, tmp_path: Path):
    """GEAK names its end-of-run report after the harness that drove the run.

    Standalone GEAK and GEAK-as-KERNEL_AGENT answer different questions and their
    numbers are not comparable, so the two must not produce identically-named
    reports in a directory someone later reads without context.
    """
    runner = tmp_path / "interface" / "run_e2e.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("")
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.delenv("GEAK_INVOKED_BY", raising=False)

    captured: dict = {}
    monkeypatch.setattr(geak_runner.subprocess, "Popen", _fake_popen(captured))
    (tmp_path / "result.json").write_text(json.dumps({"ok": True}))

    geak_runner.call_geak({"model_path": "/m"}, tmp_path, timeout_s=120)

    assert captured["env"]["GEAK_INVOKED_BY"] == "hyperloom"
