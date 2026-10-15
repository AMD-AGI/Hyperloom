# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The one-call factory: the shipped runner in a local sandbox, only the GPU measurement stubbed."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.test_hosted_real_runner import _git_repo, _point
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.local_hosted import build_local_toolbox, max_parallel_tests
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner


@pytest.mark.parametrize(("cards", "tp", "expected"), [(8, 1, 8), (8, 2, 4), (8, 8, 1), (16, 2, 4), (2, 1, 2)])
def test_at_most_eight_over_tp_tests_run_at_once(cards: int, tp: int, expected: int) -> None:
    assert max_parallel_tests(cards, tp) == expected


def test_a_test_that_needs_more_cards_than_the_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot run a test of TP=4"):
        max_parallel_tests(2, 4)


async def test_the_factory_builds_a_toolbox_that_runs_a_test_end_to_end(tmp_path: Path) -> None:
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    session = tmp_path / "session"
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = build_local_toolbox(
        session_dir=session,
        run_id="run-1",
        cards=range(8),
        notebook=notebook,
        repo=FakeRepo(),
        planner=PointPlanner(_point(tmp_path)),
        spec=spec_context(
            tmp_path, inferencex_commit=commit, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")
        ),
        active_image_digest=IMAGE,
        workload_kind=WorkloadKind.AGENTX,
        policy=QueuePolicy(per_branch_k=2, global_concurrency=64),
        runner_envs={"INFERENCEX_REPO": str(repo)},
        poll_interval_sec=0.1,
        timeout_sec=120,
    )
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
