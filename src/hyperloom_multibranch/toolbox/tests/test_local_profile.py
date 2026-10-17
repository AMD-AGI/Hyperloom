# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A profile test on a toolbox built by the local factory, then TraceLens on its trace (CPU only).

The shipped runner and the real local backend run in a local sandbox; only ``run_grid`` is replaced (no GPU), by a
stand-in that leaves the trace a profiled round leaves. What this proves is the wiring: acceptance accepts the kind,
the point turns the profiler on, the trace lands in the signed bundle and in the session, the result names it and the
analysis route resolves it.
"""

from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.test_hosted_real_runner import _git_repo
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.local_hosted import (
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req
from hyperloom_multibranch.toolbox.tests.test_analysis_route import _Analysis
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.tests.test_local_hosted import _require_tool_checkout
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup
from hyperloom_multibranch.toolbox.tools import op_shares as op_shares_module

PROFILE_POINT = """\
benchmark:
  model: /nonexistent/model
  framework: sglang
  envs:
    TP: '1'
    RUN_EVAL: 'false'
    PROFILE: '1'
  profiler:
    torch_profiler:
      enabled: true
"""


def _runner_python(tmp_path: Path) -> str:
    """An interpreter whose ``-m hyperloom.measure.runner`` is the runner with ``run_grid`` replaced."""
    script = tmp_path / "profile-python"
    script.write_text(f"#!/bin/sh\nexec {sys.executable} -m hyperloom.measure.tests.profile_runner_entry\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _toolbox(tmp_path: Path, notebook: LocalNotebook, commit: str, repo: Path, point: str) -> LocalHostedToolbox:
    session = tmp_path / "session"
    point_path = tmp_path / "profile-point.yaml"
    point_path.write_text(point)
    return build_local_toolbox(
        LocalSession(session, "run-1", range(2)),
        notebook,
        AcceptanceSetup(FakeRepo(), PointPlanner(point_path), IMAGE, WorkloadKind.AGENTX),
        spec_context(
            tmp_path, inferencex_commit=commit, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")
        ),
        options=LocalOptions(
            policy=QueuePolicy(per_branch_k=2, global_concurrency=64),
            runner_envs={"INFERENCEX_REPO": str(repo)},
            python=_runner_python(tmp_path),
            poll_interval_sec=0.1,
            timeout_sec=120,
            tools=[_Analysis("tracelens")],
        ),
    )


async def _profile(toolbox: LocalHostedToolbox) -> TestResult:
    request_id = await toolbox.submit_test(req(kind=TestKind.PROFILE, purpose=Purpose.PROFILE))
    await asyncio.wait_for(toolbox.drain(), 120)
    entry = toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry
    return TestResult.from_document(entry.outcome["result"])


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    _require_tool_checkout(repo, commit, tmp_path / "home", monkeypatch)
    return repo, commit


async def test_a_profile_test_leaves_a_verified_trace_that_tracelens_is_started_on(
    tmp_path: Path, checkout: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(op_shares_module, "read_op_shares", lambda result: {"gemm": 0.5})
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox(tmp_path, notebook, checkout[1], checkout[0], PROFILE_POINT)
    try:
        await toolbox.start()
        result = await _profile(toolbox)
        assert result.validity is Validity.SUCCEEDED and result.rounds == (), result
        session = tmp_path / "session"
        trace_dir = session / result.profile_location
        assert (trace_dir / "torch_trace" / "rank0.pt.trace.json.gz").read_bytes() == b"stub trace bytes"
        await toolbox._executor.reverify(result)  # type: ignore[attr-defined]

        job = await toolbox.submit_analysis("b1", "tracelens", {}, profile_request_id=result.request_id)
        await toolbox._tools.wait(job)  # type: ignore[union-attr]
        (request,) = toolbox._tools._tools["tracelens"].requests  # type: ignore[union-attr]
        assert request.params["trace_input"] == str(trace_dir.resolve())
    finally:
        await toolbox.close()


async def test_a_profile_point_that_does_not_turn_the_profiler_on_is_not_run_as_a_profile(
    tmp_path: Path, checkout: tuple[Path, str]
) -> None:
    point = PROFILE_POINT.replace("enabled: true", "enabled: false")
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _toolbox(tmp_path, notebook, checkout[1], checkout[0], point)
    try:
        await toolbox.start()
        result = await _profile(toolbox)
        assert result.validity is Validity.MEASUREMENT_FAILED and result.profile_location == "", result
        assert "does not turn the torch profiler on" in (result.failure_reason or ""), result.failure_reason
    finally:
        await toolbox.close()
