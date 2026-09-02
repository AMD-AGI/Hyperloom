# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hyperloom.orchestrator.kernel import controller_submit


def test_build_controller_command_contains_only_the_three_contract_arguments(tmp_path: Path) -> None:
    command = controller_submit.build_controller_command(
        handoff_dir=tmp_path / "handoff",
        output_dir=tmp_path / "cycle",
        budget_minutes=120,
    )

    assert command[:4] == [
        sys.executable,
        "-m",
        "kernelforge.cli",
        "kernel-rewrite-controller",
    ]
    assert command[4:] == [
        "--handoff-dir",
        str((tmp_path / "handoff").resolve()),
        "--budget-minutes",
        "120",
        "--output-dir",
        str((tmp_path / "cycle").resolve()),
    ]


def test_timeout_result_recovers_complete_patches_without_terminal_state(tmp_path: Path) -> None:
    output = tmp_path / "cycle"
    patch = output / "result" / "patches" / "operator"
    patch.mkdir(parents=True)
    (patch / "change.patch").write_text("diff\n", encoding="utf-8")
    (patch / "report.md").write_text("# Report\n", encoding="utf-8")
    (patch / "publication.json").write_text("{}\n", encoding="utf-8")
    state = output / "controller" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"status": "running", "task_count": 1}), encoding="utf-8")

    result = controller_submit.read_controller_result(
        output_dir=output,
        returncode=-1,
        timed_out=True,
    )

    assert result["status"] == "partial"
    assert result["patch_count"] == 1
    assert result["killed_by_hyperloom"] is True
    assert result["patches_root"] == str(output.resolve() / "result" / "patches")


def test_run_controller_subprocess_normalizes_a_successful_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "cycle"

    def _run(command, **_kwargs):
        state = output / "controller" / "state.json"
        state.parent.mkdir(parents=True)
        state.write_text(
            json.dumps({"status": "no_opportunity", "task_count": 0}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "done", "")

    monkeypatch.setattr(controller_submit, "run_with_session_kill", _run)

    result = controller_submit.run_controller_subprocess(
        handoff_dir=tmp_path / "handoff",
        output_dir=output,
        budget_minutes=60,
        hard_timeout_sec=3600,
    )

    assert result["status"] == "no_opportunity"
    assert result["returncode"] == 0
    assert result["timed_out"] is False


def test_run_controller_subprocess_recovers_after_hard_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["controller"], 1, stderr="timed out")

    monkeypatch.setattr(controller_submit, "run_with_session_kill", _timeout)

    result = controller_submit.run_controller_subprocess(
        handoff_dir=tmp_path / "handoff",
        output_dir=tmp_path / "cycle",
        budget_minutes=1,
        hard_timeout_sec=1,
    )

    assert result["status"] == "failed"
    assert result["timed_out"] is True
    assert result["stderr_tail"] == "timed out"
