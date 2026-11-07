# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An AgentX baseline served with the prefix cache off does not end ``succeeded``."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Purpose, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from inference_suite.contracts import RunLayout

SGLANG_OFF = "server_args=ServerArgs(model_path='m', disable_radix_cache=True, tp_size=8)\n"
SGLANG_ON = "server_args=ServerArgs(model_path='m', disable_radix_cache=False, tp_size=8)\n"
POLICY = QueuePolicy(per_branch_k=1, global_concurrency=1)


class _LoggingBackend:
    """The real backend; every round the test landed also gets the server log of the launch that ran it."""

    name = "hosted"

    def __init__(self, inner: object, log: str, report: str | None = None) -> None:
        self.inner, self.log, self.report = inner, log, report

    @property
    def journal(self):  # type: ignore[no-untyped-def]
        return self.inner.journal  # type: ignore[attr-defined]

    def run_test(self, call):  # type: ignore[no-untyped-def]
        outcome = self.inner.run_test(call)  # type: ignore[attr-defined]
        for item in outcome.rounds:
            (Path(item.output_dir) / "server.log").write_text(self.log, encoding="utf-8")
            if self.report is not None:
                RunLayout(Path(item.output_dir)).report.write_text(self.report, encoding="utf-8")
        return outcome


@pytest.fixture
def rig(tmp_path: Path):
    made = make_rig(tmp_path)
    yield made
    made.backend.close()
    made.provider.close()


async def _run(
    tmp_path: Path, r: Rig, log: str, kind: WorkloadKind, purpose: Purpose, report: str | None = None
) -> TestResult:
    point = tmp_path / "planned.yaml"
    point.write_text("benchmark:\n  framework: sglang\n  envs:\n    TP: '1'\n    RUN_EVAL: 'false'\nfake_mode: ok\n")
    executor = HostedExecutor(_LoggingBackend(r.backend, log, report), tmp_path / "session", workload_kind=kind)  # type: ignore[arg-type]
    tb = build(tmp_path, executor=executor, planner=PointPlanner(point), policy=POLICY, workload_kind=kind)
    await tb.toolbox.start()
    request_id = await tb.toolbox.submit_test(req(purpose=purpose))
    await asyncio.wait_for(tb.toolbox.drain(), 60)
    entry = tb.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None
    return TestResult.from_document(entry.outcome["result"])


async def test_an_agentx_baseline_on_a_cacheless_server_is_refused(tmp_path: Path, rig) -> None:
    result = await _run(tmp_path, rig, SGLANG_OFF, WorkloadKind.AGENTX, Purpose.BASELINE)
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
    assert (result.failure_reason or "").startswith("agentx_baseline_prefix_cache_off: ")
    assert "disable_radix_cache=True" in (result.failure_reason or "")
    assert result.rounds == ()


async def test_an_agentx_baseline_with_the_cache_on_succeeds(tmp_path: Path, rig) -> None:
    result = await _run(tmp_path, rig, SGLANG_ON, WorkloadKind.AGENTX, Purpose.BASELINE)
    assert result.validity is Validity.SUCCEEDED and result.failure_class is None


async def test_an_agentx_explore_on_a_cacheless_server_is_not_a_baseline_and_is_left_alone(tmp_path: Path, rig) -> None:
    result = await _run(tmp_path, rig, SGLANG_OFF, WorkloadKind.AGENTX, Purpose.EXPLORE)
    assert result.validity is Validity.SUCCEEDED


async def test_an_inferencex_baseline_on_a_cacheless_server_is_left_alone(tmp_path: Path, rig) -> None:
    # An InferenceX baseline carries accuracy, which the fake runner has no round for: the test ends elsewhere, never
    # as the cache refusal.
    result = await _run(tmp_path, rig, SGLANG_OFF, WorkloadKind.INFERENCEX, Purpose.BASELINE)
    assert "prefix_cache_off" not in (result.failure_reason or "")
    assert result.failure_class is not FailureClass.CANDIDATE


async def test_an_off_contract_suite_report_of_an_agentx_baseline_is_a_refused_measurement(tmp_path: Path, rig) -> None:
    result = await _run(tmp_path, rig, SGLANG_ON, WorkloadKind.AGENTX, Purpose.BASELINE, report="{")
    assert result.validity is Validity.MEASUREMENT_FAILED
    assert "measurement refused: the verified baseline round is unreadable" in (result.failure_reason or "")
