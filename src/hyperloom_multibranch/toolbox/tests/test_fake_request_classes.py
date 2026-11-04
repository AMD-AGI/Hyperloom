# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fake toolbox's result of an AgentX test counts the failed requests of its measured rounds by class, as a hosted one's.

A consumer that refuses a result which says nothing of its failed requests (the classes are missing, so it cannot tell
a server that refused from one that answered with nothing) must not refuse the fake's: the fake derives them from the
figures it already gives each round, the requests and the ones that failed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hyperloom.common.perf_metric import GRADED_ERROR_RATE
from hyperloom.contracts.measurement import Purpose, RequestClasses, TestKind, TestRequest, TestResult, Validity
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom_multibranch.contracts.gate import ResultUnverified
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.fake import FakeToolbox, FakeWiring
from hyperloom_multibranch.toolbox.fake_model import DEFAULT_BASELINE, Fate, FakeToolboxConfig

DELTA = RecipeDelta(config=ConfigDelta(server_args="--max-num-seqs 256"))
ROUND_REQUESTS = 1000


def _request(**overrides: object) -> TestRequest:
    fields: dict[str, object] = {
        "kind": TestKind.TEST,
        "purpose": Purpose.EXPLORE,
        "branch_id": "b1",
        "parent_commit": "c0",
        "delta": DELTA,
    }
    return TestRequest(**{**fields, **overrides})


def _toolbox(workload: WorkloadKind, *, error_rate: float = 0.0, **config: object) -> FakeToolbox:
    baseline = {**DEFAULT_BASELINE, GRADED_ERROR_RATE: error_rate}
    return FakeToolbox(
        FakeToolboxConfig(baseline=baseline, **config), wiring=FakeWiring(workload_kind=workload), autorun=False
    )


async def _run(box: FakeToolbox, request: TestRequest) -> TestResult:
    request_id = await box.submit_test(request)
    await box.drain()
    return box.results[request_id]


async def test_each_measured_round_of_an_agentx_test_counts_its_failed_requests_from_the_rounds_own_figures() -> None:
    box = _toolbox(WorkloadKind.AGENTX, error_rate=2.5, measure_rounds=2)
    result = await _run(box, _request(concurrency=(4, 8)))

    assert result.validity is Validity.SUCCEEDED
    assert [(c.point, c.concurrency) for c in result.request_classes] == [(0, 4), (0, 4), (1, 8), (1, 8)]
    for classes in result.request_classes:
        assert classes == RequestClasses(
            classes.point,
            concurrency=classes.concurrency,
            requests_total=ROUND_REQUESTS,
            failed_total=25,
            failed_transport=25,
            failed_no_content=0,
            failed_empty=0,
            failed_short=0,
        ), "the fake's 2.5 % error rate is 25 failed requests of a round's 1000, every one a transport failure"
    measured = [r for r in result.rounds if r.phase == "measure"]
    assert [(c.requests_total, c.failed_total) for c in result.request_classes] == [
        (r.completed_requests, r.failed_requests) for r in measured
    ], "the classes are the rounds' own figures, no second set"


async def test_a_round_the_fake_gave_no_failed_request_counts_zero_failed_and_not_an_unknown() -> None:
    result = await _run(_toolbox(WorkloadKind.AGENTX), _request())
    classes = result.request_classes[0]
    assert (classes.failed_total, classes.failed_transport, classes.failed_no_content) == (0, 0, 0)
    assert classes.reason == "" and classes.failed_empty == 0 and classes.failed_short == 0


async def test_the_classes_survive_the_log_the_toolbox_keeps_its_results_in() -> None:
    box = _toolbox(WorkloadKind.AGENTX, error_rate=1.0)
    result = await _run(box, _request())
    assert result.request_classes, "the result read back from the submit log still has them"
    assert TestResult.from_document(result.to_document()).request_classes == result.request_classes


async def test_a_result_of_an_inferencex_campaign_has_none() -> None:
    assert (await _run(_toolbox(WorkloadKind.INFERENCEX, error_rate=2.5), _request())).request_classes == ()


async def test_a_test_that_did_not_succeed_has_none() -> None:
    box = _toolbox(WorkloadKind.AGENTX)
    box.script(lambda request: True, [Fate.CANDIDATE])
    result = await _run(box, _request())
    assert result.validity is Validity.CANDIDATE_FAILED and result.request_classes == ()


async def test_a_profile_test_has_none_because_it_holds_no_measured_round() -> None:
    result = await _run(_toolbox(WorkloadKind.AGENTX), _request(kind=TestKind.PROFILE, purpose=Purpose.PROFILE))
    assert result.validity is Validity.SUCCEEDED and result.request_classes == ()


async def test_a_result_whose_classes_were_edited_is_not_what_the_executor_produced() -> None:
    box = _toolbox(WorkloadKind.AGENTX, error_rate=2.5)
    result = await _run(box, _request())
    await box.executor.reverify(result)

    first = result.request_classes[0]
    edited = replace(first, failed_transport=0, failed_no_content=25, failed_empty=25)
    with pytest.raises(ResultUnverified, match="request_classes: differs from what the fake executor produced"):
        await box.executor.reverify(replace(result, request_classes=(edited, *result.request_classes[1:])))
