# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise ordered transformations, measured retries and verified checkpoints."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import time
from unittest.mock import AsyncMock, Mock

from claude_agent_sdk import AssistantMessage, ResultMessage, ToolResultBlock, ToolUseBlock, UserMessage
import pytest

from ave.optimizer import AveLoop, optimize
from ave.preparation import PreparedKernel
from ave.task import DeadlineExceeded
from hyperloom_common.git_safety import app_git, repo_git


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
    assert repo_git(prepared_run.workspace, "rev-parse", "HEAD").stdout.strip() == result["final_commit"]
    assert repo_git(prepared_run.workspace, "diff", "HEAD").stdout == ""
    assert json.loads(prepared_run.result_json.read_text()) == result


@pytest.mark.parametrize("failure", ["sdk", "git"])
@pytest.mark.parametrize("recover", [True, False])
def test_unexpected_step_errors_preserve_verified_prefix(prepared_run, monkeypatch, caplog, failure, recover):
    slower = "VALUE = 7\nLATENCY = 3.0\n"
    faster = "VALUE = 7\nLATENCY = 1.0\n"
    claude = FakeClaude([slower, faster, faster, faster])

    def should_fail():
        return len(claude.prompts) == 2 or (not recover and len(claude.prompts) > 1)

    async def query(*, prompt, options):
        async for message in claude.query(prompt=prompt, options=options):
            if failure == "sdk" and should_fail():
                raise Exception("SDK stream failed")
            yield message

    def host_git(worktree, *args, **kwargs):
        if failure == "git" and "commit" in args and should_fail():
            raise subprocess.CalledProcessError(1, ["git", *args])
        return app_git(worktree, *args, **kwargs)

    monkeypatch.setattr("claude_agent_sdk.query", query)
    monkeypatch.setattr("ave.optimizer.app_git", host_git)
    result = asyncio.run(AveLoop(prepared_run).optimize())

    assert result["status"] == ("completed" if recover else "failed")
    assert result["completed_steps"] == [step.id for step in prepared_run.steps[: 2 if recover else 1]]
    assert [item["status"] for item in result["attempts"]] == (
        ["passed", "failed", "passed"] if recover else ["passed", "failed", "failed", "failed"]
    )
    assert all(source == slower for source in claude.sources[1:])
    assert claude.prompts[2]["recent_attempts"][-1] == result["attempts"][1]
    assert json.loads((prepared_run.logs / "0002-1/result.json").read_text()) == result["attempts"][1]
    assert any(record.exc_info for record in caplog.records if "Ave step" in record.message)
    assert (prepared_run.workspace / "kernel.py").read_text() == (faster if recover else slower)
    assert repo_git(prepared_run.workspace, "status", "--porcelain").stdout == ""
    assert json.loads(prepared_run.result_json.read_text()) == result


@pytest.mark.parametrize("failure", ["baseline", "restore"])
def test_fatal_loop_errors_write_failed_checkpoint(prepared_run, monkeypatch, failure):
    claude = FakeClaude(["VALUE = 7\nLATENCY = 1.0\n"])
    monkeypatch.setattr("claude_agent_sdk.query", claude.query)
    target = "ave.optimizer.measure" if failure == "baseline" else "ave.workspace.StepWorkspace.restore"
    monkeypatch.setattr(target, Mock(side_effect=Exception("fatal loop error")))

    with pytest.raises(Exception, match="fatal loop error"):
        asyncio.run(AveLoop(prepared_run).optimize())

    result = json.loads(prepared_run.result_json.read_text())
    assert result["status"] == "failed" and result["reason"] == "fatal loop error"
    assert len(claude.prompts) == (0 if failure == "baseline" else 1)


@pytest.mark.parametrize("stop", [DeadlineExceeded, asyncio.CancelledError, CancelledError])
def test_deadline_and_cancellation_do_not_retry(prepared_run, monkeypatch, stop):
    agent = AsyncMock(side_effect=stop("stopped"))
    monkeypatch.setattr(AveLoop, "run_step_agent", agent)
    loop = AveLoop(prepared_run)
    if stop is DeadlineExceeded:
        asyncio.run(loop.optimize())
    else:
        with pytest.raises(stop):
            asyncio.run(loop.optimize())

    result = json.loads(prepared_run.result_json.read_text())
    assert result["status"] == ("timed_out" if stop is DeadlineExceeded else "cancelled")
    assert result["attempts"] == []
    agent.assert_awaited_once()


@pytest.mark.parametrize("source", ["repo_root", "workspace"])
def test_host_git_ignores_hooks_signing_and_ambient_config(prepared_run, monkeypatch, source):
    repo = getattr(prepared_run, source)
    output = prepared_run.result_json.parent / "campaign"
    output.mkdir()
    marker = output / "executed"
    hooks = output / "hooks"
    hooks.mkdir()
    for name in ("pre-commit", "post-checkout"):
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$0\" >> {shlex.quote(str(marker))}\nexit 1\n")
        hook.chmod(0o755)
    for name, value in (
        ("core.hooksPath", hooks),
        ("core.fsmonitor", hook),
        ("commit.gpgsign", "true"),
        ("gpg.program", hook),
        ("diff.external", hook),
        ("diff.probe.command", hook),
        ("diff.probe.textconv", hook),
    ):
        repo_git(repo, "config", name, str(value))
    admin = prepared_run.repo_root / ".git"
    (admin / "info/attributes").write_text("kernel.py diff=probe\n")
    config = (admin / "config").read_bytes()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(admin / "config"))
    monkeypatch.setenv("GIT_DIR", str(output / "missing-git-dir"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(output / "foreign-index"))
    original = "VALUE = 7\nLATENCY = 2.0\n# Existing serving edit\n"
    (repo / "kernel.py").write_text(original)
    claude = FakeClaude(["VALUE = 7\nLATENCY = 1.0\n"])
    monkeypatch.setattr("claude_agent_sdk.query", claude.query)
    kernel = PreparedKernel(repo, "kernel.py", ("kernel.py",), "kernel", prepared_run.driver.read_text(), "test")

    async def worker(run):
        assert (run.workspace / "kernel.py").read_text() == original
        result = await AveLoop(run).optimize()
        assert result["status"] == "completed" and len(result["attempts"]) == 1
        return 0

    result = asyncio.run(
        optimize(kernel, steps=prepared_run.steps[:1], output=output, budget_sec=120, run_worker=worker)
    )

    assert result["status"] == "completed"
    assert "+LATENCY = 1.0" in Path(result["integrate_params"]["patch_path"]).read_text()
    assert not marker.exists() and not (output / "foreign-index").exists()
    assert not (output / "workspace").exists()
    assert (admin / "config").read_bytes() == config
    assert (repo / "kernel.py").read_text() == original
    assert repo_git(repo, "rev-parse", "HEAD").stdout.strip() == prepared_run.base_commit


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
    assert repo_git(prepared_run.workspace, "status", "--porcelain").stdout == ""
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
    assert repo_git(prepared_run.workspace, "status", "--porcelain").stdout == ""
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
                repo_git(run.workspace, "add", "kernel.py")
                repo_git(run.workspace, "commit", "-m", "unauthorized")
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
    assert repo_git(run.workspace, "status", "--porcelain").stdout == ""
