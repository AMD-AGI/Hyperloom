# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scriptable benchmark output must reach disk while the child runs.

Buffering it in memory until the child exits loses every byte when the runner
itself is killed — which is exactly when the log is the only forensic evidence.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import bypass_scriptable as bs


def _script(tmp_path: Path, body: str) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "xdit_mi300x.sh").write_text(f"#!/bin/bash\n{body}", encoding="utf-8")
    return scripts


def _run(tmp_path: Path, monkeypatch, body: str, *, timeout_s: float) -> tuple[int, Path]:
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(_script(tmp_path, body)))
    monkeypatch.delenv("MAGPIE_PATH", raising=False)
    workspace = tmp_path / "ws"
    rc, error = bs.run_scriptable(
        framework="xdit",
        runner_type="mi300x",
        inferencex_root=str(tmp_path / "InferenceX"),
        bench={"model": "/models/flux"},
        workspace=workspace,
        timeout_s=timeout_s,
    )
    assert error is None
    return rc, workspace


def test_scriptable_logs_survive_a_timeout_kill(tmp_path, monkeypatch):
    """Output written before the kill is on disk, and the marker is appended."""
    rc, workspace = _run(
        tmp_path,
        monkeypatch,
        'echo "alive-on-stdout"\necho "alive-on-stderr" >&2\nsleep 30\n',
        timeout_s=2.0,
    )

    assert rc == 124
    stdout_log = (workspace / "scriptable_stdout.log").read_text(encoding="utf-8")
    stderr_log = (workspace / "scriptable_stderr.log").read_text(encoding="utf-8")
    assert "alive-on-stdout" in stdout_log
    assert "alive-on-stderr" in stderr_log
    assert "scriptable benchmark timed out" in stderr_log


def test_scriptable_logs_written_on_clean_exit(tmp_path, monkeypatch):
    rc, workspace = _run(
        tmp_path,
        monkeypatch,
        'echo "done"\necho "warn" >&2\nexit 3\n',
        timeout_s=30.0,
    )

    assert rc == 3
    assert (workspace / "scriptable_stdout.log").read_text(encoding="utf-8").strip() == "done"
    assert (workspace / "scriptable_stderr.log").read_text(encoding="utf-8").strip() == "warn"
