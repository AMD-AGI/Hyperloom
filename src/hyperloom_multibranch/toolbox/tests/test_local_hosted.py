# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one-call factory: the shipped runner in a local sandbox, only the GPU measurement stubbed."""

from __future__ import annotations

import asyncio
import logging
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
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
    max_parallel_tests,
)
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner


_GPU_TOOLS = ("amd-smi", "rocm-smi")
_UNREADABLE_TOOL = "#!/bin/sh\nexit 1\n"


@pytest.fixture(autouse=True)
def _machine_without_readable_gpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the tests independent of the host's GPUs.

    The shipped runner reads the model from ``amd-smi`` and the VRAM in use from ``rocm-smi``. On a GPU host the real
    tools answer and the runner compares them with the stand-in sheet, so these tests would see a hardware mismatch
    instead of the no-GPU outcome they assert. Tools that exit non-zero ahead of the real ones on ``PATH`` give every
    host the same answer: nothing readable, so no hardware check is made.
    """
    shadow = tmp_path / "gpu-tools-shadow"
    shadow.mkdir()
    for name in _GPU_TOOLS:
        tool = shadow / name
        tool.write_text(_UNREADABLE_TOOL)
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{shadow}{os.pathsep}{os.environ.get('PATH', '')}")
    for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(("cards", "tp", "expected"), [(8, 1, 8), (8, 2, 4), (8, 8, 1), (16, 2, 4), (2, 1, 2)])
def test_at_most_eight_over_tp_tests_run_at_once(cards: int, tp: int, expected: int) -> None:
    assert max_parallel_tests(cards, tp) == expected


def test_a_test_that_needs_more_cards_than_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot run a test of TP=4"):
        max_parallel_tests(2, 4)


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
        # started from the local template, signed a bundle, the backend verified it, and the toolbox applied its
        # unclear-failure rule (one rerun in a new sandbox) and wrote the final failure to the notebook.
        assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.UNCLEAR
        assert len(result.attempts) == 2 and result.manifest_sha256, result
        sandboxes = {rec.sandbox_id for rec in toolbox.backend.journal.records() if rec.request_id == request_id}
        assert len(sandboxes) == 2
        record = await notebook.record(request_id)
        assert record.result is not None and record.result.manifest_sha256 == result.manifest_sha256
        assert toolbox.backend.journal.latest(request_id).extra["verified"] is True  # type: ignore[union-attr]
    finally:
        await toolbox.close()
