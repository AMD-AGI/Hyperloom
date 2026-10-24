# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The warnings ``run_grid`` writes reach the notebook's stored result of a test run by the local factory (CPU only).

The shipped runner and the real local backend run in a local sandbox; only ``run_grid`` (a stand-in that measures and
warns) and the probes of the machine (a machine whose cards nothing else holds, so that no host adds a warning of its own)
are replaced. What this proves is the carriage: run_grid, the local backend, the signed rounds file, the verified bundle,
the executor, the result in the notebook, and the verification again.
"""

from __future__ import annotations

import asyncio
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.test_hosted_real_runner import _git_repo, _point
from hyperloom.measure.tests.warnings_runner_entry import WARNINGS
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.local_hosted import LocalOptions, LocalSession, build_local_toolbox
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.tests.test_local_hosted import _require_tool_checkout
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup


def _runner_python(tmp_path: Path) -> str:
    script = tmp_path / "warn-python"
    script.write_text(f"#!/bin/sh\nexec {sys.executable} -m hyperloom.measure.tests.warnings_runner_entry\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


async def test_the_warnings_run_grid_wrote_are_on_the_stored_result_and_an_edited_list_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    session = tmp_path / "session"
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = build_local_toolbox(
        LocalSession(session, "run-1", range(2)),
        notebook,
        AcceptanceSetup(FakeRepo(), PointPlanner(_point(tmp_path)), IMAGE, WorkloadKind.AGENTX),
        spec_context(
            tmp_path, inferencex_commit=commit, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")
        ),
        options=LocalOptions(
            policy=QueuePolicy(per_branch_k=2, global_concurrency=64),
            runner_envs={"INFERENCEX_REPO": str(repo)},
            python=_runner_python(tmp_path),
            poll_interval_sec=0.1,
            timeout_sec=120,
        ),
    )
    try:
        await toolbox.start()
        request_id = await toolbox.submit_test(req(accuracy=False))
        await asyncio.wait_for(toolbox.drain(), 120)
        entry = toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry
        result = TestResult.from_document(entry.outcome["result"])
        assert result.validity is Validity.SUCCEEDED, result.failure_reason
        assert result.nonfatal_warnings == tuple(WARNINGS)
        stored = (await notebook.record(request_id)).result
        assert stored is not None and stored.nonfatal_warnings == tuple(WARNINGS)
        executor = toolbox._executor
        await executor.reverify(result)  # type: ignore[attr-defined]
        with pytest.raises(ResultUnverified, match="nonfatal_warnings: recorded"):
            await executor.reverify(replace(result, nonfatal_warnings=result.nonfatal_warnings[:1]))  # type: ignore[attr-defined]
    finally:
        await toolbox.close()
