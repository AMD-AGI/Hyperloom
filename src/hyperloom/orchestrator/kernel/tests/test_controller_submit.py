# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
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


def test_controller_environment_prioritizes_this_checkout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(("/other/src", "/another/src")))

    env = controller_submit._controller_environment()

    source_root = str(Path(controller_submit.__file__).resolve().parents[3])
    assert env["PYTHONPATH"].split(os.pathsep)[0] == source_root
    assert env["PYTHONPATH"].split(os.pathsep).count(source_root) == 1


def test_business_failure_is_preserved_even_with_zero_returncode(tmp_path: Path) -> None:
    output = tmp_path / "cycle"
    state = output / "controller" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "opportunity analysis failed",
                "task_count": 0,
            }
        ),
        encoding="utf-8",
    )

    result = controller_submit.read_controller_result(
        output_dir=output,
        returncode=0,
        timed_out=False,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "opportunity analysis failed"
    assert result["returncode"] == 0


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


def test_forge_loop_spend_reaches_the_llm_ledger(tmp_path: Path) -> None:
    # The Controller is a child process and cannot append to this ledger while it
    # runs, so a campaign's model spend is only accounted for if what it recorded
    # is filed after the child exits.
    from hyperloom.orchestrator.trace.llm_trace import llm_calls_path

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    appended = controller_submit.record_controller_llm_usage(
        result={
            "forge_llm_usage": [
                {
                    "operator_id": "kernel:forge-loop:moe:sglang:0.5.16:flydsl:mi355x",
                    "model": "claude-opus-5",
                    "input_tokens": 1200,
                    "output_tokens": 340,
                    "cache_creation_input_tokens": 80,
                    "cache_read_input_tokens": 5000,
                    "calls": 7,
                }
            ]
        },
        session_dir=session_dir,
    )

    assert appended == 1
    rows = [json.loads(line) for line in llm_calls_path(session_dir).read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["component"] == "forge"
    assert rows[0]["task_id"] == "kernel:forge-loop:moe:sglang:0.5.16:flydsl:mi355x"
    assert rows[0]["model"] == "claude-opus-5"
    assert rows[0]["input_tokens"] == 1200
    assert rows[0]["cache_read_input_tokens"] == 5000


def test_a_controller_result_without_usage_files_nothing(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    assert controller_submit.record_controller_llm_usage(result={}, session_dir=session_dir) == 0
