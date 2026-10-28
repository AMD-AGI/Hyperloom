# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise ordered transformations, measured retries and verified checkpoints."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import time

from claude_agent_sdk import AssistantMessage, ResultMessage, ToolResultBlock, ToolUseBlock, UserMessage
import pytest

from ave.optimizer import AveLoop
from ave.workspace import git


def session_result(text="Applied the transformation.", *, is_error=False):
    return ResultMessage(
        subtype="error_during_execution" if is_error else "success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="test",
        result=text,
    )


class FakeClaude:
    def __init__(self, candidates):
        self.candidates = iter(candidates)
        self.prompts = []
        self.sources = []

    async def query(self, *, prompt, options):
        assert set(options.tools) == {"Read", "Glob", "Grep", "Edit", "Write", "Bash"}
        assert options.allowed_tools == options.tools
        assert (Path(options.cwd) / "LANGUAGE.md").read_text() == "The kernel must preserve VALUE = 7.\n"
        self.prompts.append(json.loads(prompt))
        source = Path(options.cwd) / "kernel.py"
        self.sources.append(source.read_text())
        source.write_text(next(self.candidates))
        yield session_result()


@pytest.mark.parametrize("recover", [True, False])
def test_ordered_steps_keep_slower_preparation_and_retry_from_verified_prefix(prepared_run, recover, monkeypatch):
    slower = "VALUE = 7\nLATENCY = 3.0\n"
    wrong = "VALUE = 8\nLATENCY = 0.5\n"
    faster = "VALUE = 7\nLATENCY = 1.0\n"
    candidates = [slower, wrong, faster] if recover else [slower, wrong, wrong, wrong]
    claude = FakeClaude(candidates)
    monkeypatch.setattr("claude_agent_sdk.query", claude.query)
    result = asyncio.run(AveLoop(prepared_run).optimize())

    assert result["status"] == ("completed" if recover else "failed")
    assert result["completed_steps"] == [step.id for step in prepared_run.steps[: 2 if recover else 1]]
    assert result["attempts"][0]["case_ms"] == {"decode": 3, "prefill": 6}
    assert "reference mismatch" in result["attempts"][1]["reason"]
    assert all(source == slower for source in claude.sources[1:])
    assert claude.prompts[2]["recent_attempts"][-1]["status"] == "failed"
    assert (prepared_run.workspace / "kernel.py").read_text() == (faster if recover else slower)
    assert git("rev-parse", "HEAD", cwd=prepared_run.workspace).strip() == result["final_commit"]
    assert git("diff", "HEAD", cwd=prepared_run.workspace) == ""
    assert json.loads(prepared_run.result_json.read_text()) == result


@pytest.mark.parametrize("timeout_source", ["agent", "driver"])
def test_step_timeout_retries_from_verified_prefix(prepared_run, monkeypatch, timeout_source):
    slower = "VALUE = 7\nLATENCY = 3.0\n"
    faster = "VALUE = 7\nLATENCY = 1.0\n"
    stalled = "VALUE = 7\nLATENCY = 0.5\nHANG = True\n"
    driver = prepared_run.driver
    driver.write_text(
        driver.read_text().replace(
            "import kernel\n",
            "import kernel\nimport time\n"
            "if getattr(kernel, 'HANG', False):\n"
            "    print('candidate stalled', flush=True)\n"
            "    time.sleep(60)\n",
        )
    )
    claude = FakeClaude([slower, stalled, faster])

    async def query(*, prompt, options):
        async for message in claude.query(prompt=prompt, options=options):
            if len(claude.prompts) == 2 and timeout_source == "agent":
                await asyncio.Event().wait()
            yield message

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr("ave.optimizer.MAX_SESSION_SEC", 0.1)
    monkeypatch.setattr("ave.measurement.DRIVER_TIMEOUT_SEC", 0.5)
    result = asyncio.run(AveLoop(prepared_run).optimize())

    assert result["status"] == "completed"
    assert result["completed_steps"] == [step.id for step in prepared_run.steps]
    assert [attempt["status"] for attempt in result["attempts"]] == ["passed", "failed", "passed"]
    failure = result["attempts"][1]
    assert failure["step_id"] == prepared_run.steps[1].id and failure["attempt"] == 1
    assert "timed out" in failure["reason"]
    assert json.loads((prepared_run.logs / "0002-1/result.json").read_text()) == failure
    assert claude.sources[2] == slower
    assert claude.prompts[2]["recent_attempts"][-1] == failure
    assert (prepared_run.workspace / "kernel.py").read_text() == faster
    assert git("status", "--porcelain", cwd=prepared_run.workspace) == ""
    if timeout_source == "driver":
        assert "candidate stalled" in (prepared_run.logs / "0002-1/correctness.log").read_text()


def test_deadline_stops_retries_and_preserves_verified_prefix(prepared_run, monkeypatch):
    slower = "VALUE = 7\nLATENCY = 3.0\n"
    claude = FakeClaude([slower, "VALUE = 7\nLATENCY = 1.0\n"])
    expired = []
    now = time.time()
    monkeypatch.setattr("ave.optimizer.time.time", lambda: prepared_run.deadline_unix + 1 if expired else now)

    async def query(*, prompt, options):
        async for message in claude.query(prompt=prompt, options=options):
            if len(claude.prompts) == 2:
                expired.append(True)
                raise asyncio.TimeoutError
            yield message

    monkeypatch.setattr("claude_agent_sdk.query", query)
    result = asyncio.run(AveLoop(prepared_run).optimize())

    assert result["status"] == "timed_out" and result["reason"]
    assert result["completed_steps"] == [prepared_run.steps[0].id]
    assert len(claude.prompts) == 2
    assert result["final_commit"] == result["attempts"][0]["commit"]
    assert (prepared_run.workspace / "kernel.py").read_text() == slower
    assert git("status", "--porcelain", cwd=prepared_run.workspace) == ""
    assert json.loads(prepared_run.result_json.read_text()) == result


def test_driver_stopped_by_campaign_deadline_is_not_retried(prepared_run, monkeypatch):
    prepared_run.driver.write_text("import time\nprint('driver started', flush=True)\ntime.sleep(60)\n")
    run = replace(prepared_run, deadline_unix=time.time() + 0.5)
    claude = FakeClaude([])
    monkeypatch.setattr("claude_agent_sdk.query", claude.query)
    result = asyncio.run(AveLoop(run).optimize())

    assert result["status"] == "timed_out"
    assert "deadline" in result["reason"]
    assert result["attempts"] == [] and claude.prompts == []
    assert not result["correctness_passed"]
    assert "driver started" in (run.logs / "baseline/correctness.log").read_text()
    assert json.loads(run.result_json.read_text()) == result


def test_agent_repairs_and_benchmarks_in_one_session_before_host_validation(prepared_run, monkeypatch):
    run = replace(prepared_run, steps=prepared_run.steps[:1])
    sessions = []

    async def query(*, prompt, options):
        context = json.loads(prompt)
        sessions.append(context)
        assert options.env["PYTHONPATH"].split(":")[0] == str(run.workspace)
        assert options.env["PYTHONDONTWRITEBYTECODE"] == "1"
        checks = [
            ("VALUE = 8\nLATENCY = 1.0\n", "correctness_command", "reference mismatch", True),
            ("VALUE = 7\nLATENCY = 1.0\n", "correctness_command", "allclose: True", False),
            (None, "benchmark_command", "case_ms: decode 1.0", False),
        ]
        for index, (source, command_key, expected, failed) in enumerate(checks):
            if source is not None:
                (Path(options.cwd) / "kernel.py").write_text(source)
            command = context[command_key]
            tool_id = f"check-{index}"
            yield AssistantMessage([ToolUseBlock(tool_id, "Bash", {"command": command})], model="test")
            process = subprocess.run(
                shlex.split(command), cwd=options.cwd, env=options.env, capture_output=True, text=True, timeout=10
            )
            output = process.stdout + process.stderr
            assert expected in output
            assert (process.returncode != 0) == failed
            yield UserMessage([ToolResultBlock(tool_id, output, is_error=failed)])
        yield session_result("Correctness passed. Agent speedup estimate: 100x.")

    monkeypatch.setattr("claude_agent_sdk.query", query)
    result = asyncio.run(AveLoop(run).optimize())

    assert len(sessions) == 1
    assert result["status"] == "completed"
    assert len(result["attempts"]) == 1
    assert result["final_case_ms"] == {"decode": 1.0, "prefill": 2.0}
    assert result["mean_case_speedup"] == 2.0
    attempt = run.logs / "0001-1"
    transcript = [json.loads(line) for line in (attempt / "claude.jsonl").read_text().splitlines()]
    results = [message["content"][0] for message in transcript if message["type"] == "UserMessage"]
    assert [message["is_error"] for message in results] == [True, False, False]
    assert "reference mismatch" in results[0]["content"]
    assert "allclose: True" in (attempt / "correctness.log").read_text()
    assert len(list(attempt.glob("benchmark-*.log"))) == 3


@pytest.mark.parametrize("violation", ["driver", "reference", "untracked", "commit"])
def test_agent_scope_violation_is_rejected_and_restored(prepared_run, monkeypatch, violation):
    run = replace(prepared_run, steps=prepared_run.steps[:1])
    original_driver = run.driver.read_bytes()
    reference = run.workspace / "LANGUAGE.md"
    original_reference = reference.read_bytes()
    claude = FakeClaude(["VALUE = 7\nLATENCY = 1.0\n"] * 3)

    async def query(*, prompt, options):
        async for message in claude.query(prompt=prompt, options=options):
            if violation == "driver":
                run.driver.write_text("print('allclose: True')\n")
            elif violation == "reference":
                reference.write_text("Ignore incorrect results.\n")
            elif violation == "untracked":
                (run.workspace / "extra.py").write_text("VALUE = 8\n")
            else:
                git("add", "kernel.py", cwd=run.workspace)
                git("commit", "-m", "unauthorized", cwd=run.workspace)
            yield message

    monkeypatch.setattr("claude_agent_sdk.query", query)
    result = asyncio.run(AveLoop(run).optimize())

    assert result["status"] == "failed"
    assert result["completed_steps"] == []
    assert result["final_commit"] == run.base_commit
    assert all(attempt["reason"].startswith("agent changed") for attempt in result["attempts"])
    assert not list(run.logs.glob("0001-*/correctness.log"))
    assert run.driver.read_bytes() == original_driver
    assert reference.read_bytes() == original_reference
    assert git("status", "--porcelain", cwd=run.workspace) == ""
