# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``profile_failing``: once a profile failed in a way every later one repeats, further profiles are refused.

A profile whose profiler hung or killed the server fails the same way in every sandbox of the image, so each repeat
only holds a card for a cold start and a profile window. The toolbox reads the failure from its submit log and refuses
the next PROFILE at acceptance, naming the failed one; a failure the candidate caused, or one the platform's capacity
caused, refuses nothing. All of it goes through ``RealToolbox.submit_test``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Purpose, TestKind, TestRequest, TestResult
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.failures import HF_CACHE_PERMISSION, PROFILER_DIED, PROFILER_HUNG, SERVER_STALLED_REASON
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected
from hyperloom_multibranch.toolbox.acceptance import PROFILE_FAILING_CODE
from hyperloom_multibranch.toolbox.attempts import TestJob
from hyperloom_multibranch.toolbox.tests.helpers import PARENT, Rig, ScriptedExecutor, build, delta, failed, ok, req


def profile(branch: str = "b1", change: RecipeDelta | None = None) -> TestRequest:
    request = req(branch, kind=TestKind.PROFILE, purpose=Purpose.PROFILE, change=change or RecipeDelta())
    return replace(request, parent_commit=PARENT)


def profiles_fail(failure: FailureClass, reason: str) -> ScriptedExecutor:
    """Every attempt of a profile fails with ``failure`` and ``reason``; every other test measures."""

    def run(job: TestJob) -> TestResult:
        return failed(job, failure, reason=reason) if job.request.kind is TestKind.PROFILE else ok(job)

    return ScriptedExecutor(script=[run])


async def ended(rig: Rig, request: TestRequest) -> str:
    request_id = await rig.toolbox.submit_test(request)
    await rig.toolbox.drain()
    return request_id


async def refused(rig: Rig, request: TestRequest) -> RequestRejected:
    with pytest.raises(RequestRejected) as error:
        await rig.toolbox.submit_test(request)
    assert error.value.code == PROFILE_FAILING_CODE
    return error.value


def test_profile_failing_is_in_the_closed_refusal_vocabulary() -> None:
    assert PROFILE_FAILING_CODE in REJECTION_CODES


async def test_a_profile_whose_profiler_hung_refuses_the_next_one_and_names_it(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=profiles_fail(FailureClass.INFRA, f"{PROFILER_HUNG}: last line"))
    first = await ended(rig, profile("b1"))
    assert len(rig.executor.jobs) == 1, "the hang is not retried"

    rejection = await refused(rig, profile("b2", delta("--other")))

    assert first in rejection.reason and PROFILER_HUNG in rejection.reason
    assert "read the code" in rejection.reason and "dispatch path" in rejection.reason
    assert rejection.existing_request_id is None, "there is nothing to wait for"
    assert [entry.request_id for entry in rig.toolbox._log.entries()] == [first], "the refused one is not logged"
    assert await rig.toolbox.submit_test(req("b2")), "a test that is no profile is accepted"
    await rig.toolbox.close()


async def test_a_profile_whose_server_died_in_its_window_twice_refuses_the_next_one(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=profiles_fail(FailureClass.INFRA, f"{PROFILER_DIED}: RuntimeError"))
    first = await ended(rig, profile("b1"))
    assert len(rig.executor.jobs) == 2, "the death in the window is retried once"

    assert first in (await refused(rig, profile("b2", delta("--other")))).reason
    await rig.toolbox.close()


async def test_a_restarted_toolbox_still_refuses(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=profiles_fail(FailureClass.INFRA, f"{PROFILER_HUNG}: last line"))
    first = await ended(rig, profile("b1"))
    await rig.toolbox.close()

    again = build(tmp_path)
    await again.toolbox.start()

    assert first in (await refused(again, profile("b2"))).reason
    await again.toolbox.close()


async def test_profiles_the_candidate_failed_refuse_nothing(tmp_path: Path) -> None:
    """A candidate that breaks its server fails its profile too, with any reason; it says nothing of the profiler."""
    rig = build(tmp_path, executor=profiles_fail(FailureClass.CANDIDATE, f"{PROFILER_HUNG}: last line"))
    await ended(rig, profile("b1", delta("--one")))
    await ended(rig, profile("b2", delta("--two")))

    assert await rig.toolbox.submit_test(profile("b3", delta("--three")))
    await rig.toolbox.close()


async def test_profiles_of_two_recipes_failing_the_same_way_refuse_the_next_one(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=profiles_fail(FailureClass.UNCLEAR, f"{SERVER_STALLED_REASON}: silent"))
    await ended(rig, profile("b1", delta("--one")))
    assert await rig.toolbox.submit_test(profile("b1", delta("--one"))), "one recipe failing again trips nothing"
    await rig.toolbox.drain()
    assert await rig.toolbox.submit_test(profile("b2", delta("--two"))), "the first failure of another recipe neither"
    await rig.toolbox.drain()

    rejection = await refused(rig, profile("b3", delta("--three")))
    assert SERVER_STALLED_REASON in rejection.reason
    await rig.toolbox.close()


async def test_a_refused_measurement_is_judged_by_its_own_reason(tmp_path: Path) -> None:
    rig = build(tmp_path, executor=ScriptedExecutor(refuse="profile_without_steps: capture traces only"))
    await ended(rig, profile("b1", delta("--one")))
    second = await ended(rig, profile("b2", delta("--two")))

    rejection = await refused(rig, profile("b3", delta("--three")))
    assert second in rejection.reason and "profile_without_steps" in rejection.reason
    await rig.toolbox.close()


@pytest.mark.parametrize(
    "reason",
    [f"{HF_CACHE_PERMISSION}: /cache", "boom"],
    ids=["waited_out", "retries_exhausted"],
)
async def test_failures_of_the_platform_refuse_nothing(tmp_path: Path, reason: str) -> None:
    rig = build(tmp_path, executor=profiles_fail(FailureClass.INFRA, reason))
    await ended(rig, profile("b1", delta("--one")))
    await ended(rig, profile("b2", delta("--two")))

    assert await rig.toolbox.submit_test(profile("b3", delta("--three")))
    await rig.toolbox.close()
