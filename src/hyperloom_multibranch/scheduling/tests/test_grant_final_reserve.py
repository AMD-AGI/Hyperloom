# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A hosted session keeps FINAL enough of its grant: EXPLORE ends a FINAL reserve before the grant's stop when that
comes before the end the session's own budget gives it, and FINAL's validation, retried after infrastructure failures
a retry waits out, still ends before the grant does. The grant counts from the run's creation, before the session
starts, so a session's own deadline can leave FINAL less of the grant than it needs."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, Purpose, TestResult, Validity
from hyperloom.hosted.host_info import HostInfo, read_host_info
from hyperloom.measure.failures import FOREIGN_PROCESS_ON_CARD
from hyperloom_multibranch.blackboard.tests.factories import FakeClock
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS, SessionBudget
from hyperloom_multibranch.scheduling.hosted_run import (
    FINAL_ATTEMPTS,
    FINAL_COLD_START_SECONDS,
    FINAL_RESERVE_FLOOR_SECONDS,
    FINAL_WIND_DOWN_SECONDS,
    GRANT_MARGIN_SECONDS,
    GrantEnd,
    final_reserve_seconds,
    grant_end,
    grant_explore_cap,
)
from hyperloom_multibranch.toolbox.attempts import INFRA_BACKOFF_SECONDS, TestJob
from hyperloom_multibranch.toolbox.tests.helpers import ScriptedExecutor, build, failed, ok, req

CREATED = datetime(2026, 11, 10, 16, 58, tzinfo=timezone.utc)
#: The grant's runtime: six and a half hours.
RUNTIME_SEC = 23400
#: The session's budget: six hours, from its start three minutes after the run was created.
BUDGET_SEC = 6 * 3600.0
STARTED_AFTER_CREATE_SEC = 180.0
#: The session's longest test.
TEST_SEC = 1500.0


def _host(tmp_path: Path, **grant: int) -> HostInfo:
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    env = {
        "HYPERLOOM_INPUT_FILE": str(spec),
        "HYPERLOOM_RUN_ID": "run-1",
        "HYPERLOOM_RUN_CREATED_AT": CREATED.isoformat(),
        "HYPERLOOM_STATE_DIR": str(tmp_path),
    }
    if grant:
        env["HYPERLOOM_GRANT_LIMITS"] = json.dumps(grant)
    return read_host_info(env, require=())


def test_the_grant_ends_its_runtime_after_the_run_was_created_and_its_tests_end_a_close_grace_before(
    tmp_path: Path,
) -> None:
    end = grant_end(_host(tmp_path, max_runtime_sec=RUNTIME_SEC))

    assert end == GrantEnd(CREATED.timestamp() + RUNTIME_SEC)
    assert end.tests_end_by == end.at - MB_CLOSE_GRACE_CAP_SECONDS
    cap = grant_explore_cap(_host(tmp_path, max_runtime_sec=RUNTIME_SEC))
    assert cap is not None and cap(TEST_SEC) == end.explore_cap(TEST_SEC)


@pytest.mark.parametrize("grant", [{}, {"max_gpus": 8}], ids=["no_grant", "no_runtime"])
def test_a_run_whose_grant_states_no_runtime_has_no_grant_end_and_no_explore_cap(
    tmp_path: Path, grant: dict[str, int]
) -> None:
    assert grant_end(_host(tmp_path, **grant)) is None
    assert grant_explore_cap(_host(tmp_path, **grant)) is None


def test_the_reserve_covers_the_wind_down_three_attempts_their_backoff_and_a_cold_start() -> None:
    backoff = INFRA_BACKOFF_SECONDS[0] + INFRA_BACKOFF_SECONDS[1]

    assert FINAL_ATTEMPTS == 3
    assert final_reserve_seconds(TEST_SEC) == pytest.approx(
        FINAL_WIND_DOWN_SECONDS + 3 * TEST_SEC + backoff + FINAL_COLD_START_SECONDS
    )
    assert final_reserve_seconds(0.0) == FINAL_RESERVE_FLOOR_SECONDS, "nothing measured yet: the floor"
    assert final_reserve_seconds(60.0) == FINAL_RESERVE_FLOOR_SECONDS


def test_a_grant_that_ends_after_the_session_s_own_explore_end_leaves_it_as_it_was() -> None:
    budget = SessionBudget(1_000_000.0, BUDGET_SEC)
    roomy = GrantEnd(budget.deadline + 10 * 3600)

    own = budget.explore_end(budget.started_at, TEST_SEC)
    assert budget.explore_end(budget.started_at, TEST_SEC, roomy.explore_cap(TEST_SEC)) == own
    assert budget.explore_end(budget.started_at, TEST_SEC, math.inf) == own


def _validation_attempts(clock: FakeClock, cold_start: float) -> ScriptedExecutor:
    """Every test takes :data:`TEST_SEC`; the validation test's first attempt also ``cold_start``, and its first two
    attempts end as cards a process outside the test held."""

    def attempt(job: TestJob) -> TestResult:
        started = clock.now
        validation = job.request.branch_id == INTEGRATION_BRANCH
        clock.advance(TEST_SEC + (cold_start if validation and job.attempt == 1 else 0.0))
        if validation and job.attempt <= 2:
            result = failed(job, FailureClass.INFRA, f"n{job.attempt}", f"{FOREIGN_PROCESS_ON_CARD}: card 0 held")
        else:
            result = ok(job, f"n{job.attempt}")
        return replace(result, started_at=started, finished_at=clock.now)

    return ScriptedExecutor([attempt])


async def _final_from(tmp_path: Path, explore_end: float, grant: GrantEnd) -> tuple[TestResult, list[float]]:
    """FINAL from ``explore_end``: one EXPLORE test finished as EXPLORE ended, then, one wind-down later, the
    validation test on the integration branch, in a toolbox whose tests must end by the grant's stop. Its result, and
    the waits before its retries."""

    clock, slept = FakeClock(explore_end - TEST_SEC), []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.advance(seconds)

    rig = build(
        tmp_path,
        executor=_validation_attempts(clock, FINAL_COLD_START_SECONDS),
        clock=clock,
        infra_backoff=INFRA_BACKOFF_SECONDS,
        sleep=sleep,
        tests_end_by=grant.tests_end_by,
    )
    await rig.toolbox.submit_test(req("b1"))
    await rig.toolbox.drain()
    clock.advance(FINAL_WIND_DOWN_SECONDS)
    validation = await rig.toolbox.submit_test(req(INTEGRATION_BRANCH, purpose=Purpose.FINAL))
    await rig.toolbox.drain()
    record = await rig.blackboard.record(validation)
    assert record.result is not None
    return record.result, slept


async def test_a_grant_that_ends_before_the_session_s_deadline_ends_explore_so_final_with_two_infra_retries_ends_in_it(
    tmp_path: Path,
) -> None:
    grant = GrantEnd(CREATED.timestamp() + RUNTIME_SEC)
    budget = SessionBudget(CREATED.timestamp() + STARTED_AFTER_CREATE_SEC, BUDGET_SEC)
    assert budget.hard_stop_at < grant.at, "the session's own deadline and grace fit the grant"

    explore_end = budget.explore_end(budget.started_at, TEST_SEC, grant.explore_cap(TEST_SEC))
    result, slept = await _final_from(tmp_path / "capped", explore_end, grant)

    assert explore_end == grant.explore_cap(TEST_SEC) < budget.explore_end(budget.started_at, TEST_SEC)
    assert result.validity is Validity.SUCCEEDED and len(result.attempts) == 3
    assert slept == pytest.approx(list(INFRA_BACKOFF_SECONDS[:2]), abs=0.1), "the whole backoff fitted"
    assert result.finished_at is not None
    # The reserve is spent to the second, less the millisecond the fake clock moves on every reading.
    assert result.finished_at == pytest.approx(grant.tests_end_by - GRANT_MARGIN_SECONDS, abs=1.0)
    assert result.finished_at < grant.tests_end_by, "FINAL ended before the grant's stop"

    # The same FINAL from the end the session's own deadline gives: its last attempt ends past the grant's stop.
    own_end = budget.explore_end(budget.started_at, TEST_SEC)
    late, _ = await _final_from(tmp_path / "own", own_end, grant)
    assert late.finished_at is not None and late.finished_at > grant.tests_end_by


async def test_final_s_infra_retries_shorten_their_backoff_so_the_last_attempt_still_ends_by_the_grant_s_stop(
    tmp_path: Path,
) -> None:
    """A FINAL that starts late (a restart that finishes a session, an EXPLORE that wound down slowly) does not sleep
    its full backoff past the time its last attempt can still end by the grant's stop."""

    explore_end = 1_000_000.0
    final_start = explore_end + FINAL_WIND_DOWN_SECONDS
    # Its third attempt ends by the stop only if it starts 30 s after the second ended, not the 120 s backoff.
    stop = final_start + (TEST_SEC + FINAL_COLD_START_SECONDS) + 60.0 + TEST_SEC + 30.0 + TEST_SEC
    grant = GrantEnd(stop + MB_CLOSE_GRACE_CAP_SECONDS)
    result, slept = await _final_from(tmp_path, explore_end, grant)

    assert result.validity is Validity.SUCCEEDED and len(result.attempts) == 3
    assert slept == pytest.approx([60.0, 30.0], abs=0.1)
    assert result.finished_at is not None
    assert result.finished_at == pytest.approx(grant.tests_end_by, abs=1.0), "it ends by the stop, to the second"
