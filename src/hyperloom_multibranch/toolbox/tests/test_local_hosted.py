# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one-call factory: the shipped runner in a local sandbox, only the GPU measurement stubbed."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SENT, MeasurementJournal, Record
from hyperloom.measure.runner.tools import ToolError, ensure_inferencex
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.test_hosted_real_runner import _git_repo, _point
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
    last_output,
    max_parallel_tests,
)
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner


@pytest.mark.parametrize(("cards", "tp", "expected"), [(8, 1, 8), (8, 2, 4), (8, 8, 1), (16, 2, 4), (2, 1, 2)])
def test_at_most_eight_over_tp_tests_run_at_once(cards: int, tp: int, expected: int) -> None:
    assert max_parallel_tests(cards, tp) == expected


def test_a_test_that_needs_more_cards_than_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot run a test of TP=4"):
        max_parallel_tests(2, 4)


def _write(path: Path, text: str, written: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (written, written))


def test_the_live_output_is_the_last_line_of_the_live_log_written_last(tmp_path: Path) -> None:
    variant = tmp_path / "variant_00_test"
    _write(variant / "warmup_round" / "benchmark_sglang_1" / "server.log", "boot\nready\n", 1000.0)
    _write(variant / "benchmark_sglang_1" / "benchmark_stderr.log", "\x1b[32mINFO\x1b[0m\t42/100\r\x07\n\n", 1100.0)
    _write(variant / "notes.log", "written last, but no live log\n", 1200.0)
    assert last_output(tmp_path, 1130.0) == (30.0, "INFO 42/100")
    _write(variant / "benchmark_sglang_1" / "benchmark_stdout.log", "x" * 1000 + "END\n", 1300.0)
    seconds, line = last_output(tmp_path, 1300.0) or (None, "")
    assert seconds == 0.0 and len(line) == 300 and line.endswith("END")
    assert last_output(tmp_path / "missing", 1300.0) is None


def test_the_live_output_is_read_from_the_sandbox_of_the_unfinished_attempt(tmp_path: Path) -> None:
    toolbox = LocalHostedToolbox.__new__(LocalHostedToolbox)
    journal = MeasurementJournal(tmp_path / "hosted")
    toolbox.backend, toolbox.sandboxes = SimpleNamespace(journal=journal), tmp_path / "sandboxes"  # type: ignore[assignment]
    _write(
        tmp_path / "sandboxes" / "lsb-1" / "fs" / "out" / "files" / "v" / "server.log", "decode batch\n", time.time()
    )
    assert toolbox._live_log("r1") is None
    journal.append(Record("r1", 1, STATE_SENT, "lsb-1", "tests/r1/attempt-1", "r1", {}))
    output = toolbox._live_log("r1")
    assert output is not None and output[1] == "decode batch" and 0.0 <= output[0] < 60.0
    journal.append(Record("r1", 1, STATE_FINISHED, "lsb-1", "tests/r1/attempt-1", "r1", {}))
    assert toolbox._live_log("r1") is None


def _require_tool_checkout(repo: Path, commit: str, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The precondition of the end-to-end test: the runner can check the stand-in repository out as it runs it.

    The runner fetches the tools by hash under an empty home, like a sandbox. Where that fails (no usable ``git``, a
    refused local fetch) the test would otherwise show only ``measurement_failed``, so it fails here and says why.
    """
    monkeypatch.setenv("INFERENCEX_REPO", str(repo))
    monkeypatch.setenv("HOME", str(home))
    try:
        ensure_inferencex(commit, home / "cache")
    except ToolError as error:
        pytest.fail(
            f"this machine cannot check the stand-in InferenceX repository out, so no runner can start: {error}"
        )


def _toolbox_of(
    notebook: LocalNotebook, tmp_path: Path, session: Path, commit: str, **extra: Any
) -> LocalHostedToolbox:
    return build_local_toolbox(
        LocalSession(session, "run-1", range(8)),
        notebook,
        AcceptanceSetup(FakeRepo(), PointPlanner(_point(tmp_path)), IMAGE, WorkloadKind.AGENTX),
        spec_context(
            tmp_path, inferencex_commit=commit, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")
        ),
        options=LocalOptions(
            policy=QueuePolicy(per_branch_k=2, global_concurrency=64), poll_interval_sec=0.1, timeout_sec=120, **extra
        ),
    )


async def test_a_runner_that_cannot_start_says_why_and_leaves_no_log_traceback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The runner fails before it writes ``out/files``: the result carries its message and saving logs stays quiet."""
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    session = tmp_path / "session"
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    envs = {"INFERENCEX_REPO": str(repo) + "-missing"}
    toolbox = _toolbox_of(notebook, tmp_path, session, commit, runner_envs=envs, max_infra_retries=0)
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        with caplog.at_level(logging.DEBUG):
            await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry.outcome
        result = TestResult.from_document(entry.outcome["result"])
        assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
        assert result.failure_reason and "runner_start_failed" in result.failure_reason
        assert "git fetch failed" in result.failure_reason, result.failure_reason
        assert "could not be saved" not in caplog.text, "a runner with no out/files has no logs: that is not a warning"
    finally:
        await toolbox.close()


async def test_the_factory_builds_a_toolbox_that_runs_a_test_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    session = tmp_path / "session"
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox_of(notebook, tmp_path, session, commit, runner_envs={"INFERENCEX_REPO": str(repo)})
    try:
        assert toolbox._queue.policy.global_concurrency == 8, "the policy is held to 8 / TP"
        await toolbox.start()
        request_id = await toolbox.submit_test(req())
        await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry.outcome
        result = TestResult.from_document(entry.outcome["result"])
        # No GPU here, so the shipped runner cannot serve a model: what this proves is the wiring. The real runner
        # started from the local template, signed a bundle, the backend verified it, and the toolbox took the cause
        # the server log names (no sglang module) as the candidate's failure, did not rerun it and wrote it to the
        # notebook.
        assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
        assert result.failure_reason == "import_error"
        assert len(result.attempts) == 1 and result.manifest_sha256, result
        sandboxes = {rec.sandbox_id for rec in toolbox.backend.journal.records() if rec.request_id == request_id}
        assert len(sandboxes) == 1
        record = await notebook.record(request_id)
        assert record.result is not None and record.result.manifest_sha256 == result.manifest_sha256
        assert toolbox.backend.journal.latest(request_id).extra["verified"] is True  # type: ignore[union-attr]
    finally:
        await toolbox.close()
