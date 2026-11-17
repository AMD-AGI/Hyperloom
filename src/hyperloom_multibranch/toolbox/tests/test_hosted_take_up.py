# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A later leg of a hosted run takes over the session an earlier leg left: the toolbox it starts recovers the earlier
leg's unfinished tests from the submit log and the session withdraws them (``Session._take_up``). The sandboxes that leg
left measuring in the grant are the run's, whichever container or account leased them, and are ended by that withdrawal
rather than left to hold the grant's slots until they time out.

The service is the local sandbox provider, with the fake runner in each ``measurement`` sandbox; a leg is a
:class:`~hyperloom_multibranch.toolbox.hosted_toolbox.HostedToolbox` over a backend of its own, as a restarted
application is.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.measure.hosted import HostedMeasurementBackend
from hyperloom.measure.hosted_log import STATE_SENT
from hyperloom.measure.sandboxes import MEASUREMENT_ROLE, RUN_ID_KEY
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolbox
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import Rig as ToolboxRig, build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom_multibranch.toolbox.toolbox import RealToolbox

RUN = "run-1"


class _Crash(BaseException):
    """What a killed application looks like to the code under test: it unwinds without cleaning up."""


@pytest.fixture
def service(tmp_path: Path):
    made: list[Rig] = []

    def make() -> Rig:
        rig = make_rig(tmp_path)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.backend.close()
        rig.provider.close()


def _leg(tmp_path: Path, backend: HostedMeasurementBackend, rig: Rig, *, parallel: int) -> ToolboxRig:
    """A leg's hosted toolbox over ``backend``, every test planned to a point the fake runner never finishes."""

    point = point_yaml(tmp_path, "planned.yaml", fake_mode="hang")
    leg = build(
        tmp_path,
        executor=HostedExecutor(backend, tmp_path / "session"),  # type: ignore[arg-type]
        planner=PointPlanner(point),
        policy=QueuePolicy(per_branch_k=parallel, global_concurrency=parallel),
        workload_kind=WorkloadKind.AGENTX,
        kind=HostedToolbox,
    )
    toolbox = leg.toolbox
    assert isinstance(toolbox, HostedToolbox)
    toolbox.backend, toolbox.provider, toolbox.leases, toolbox.run_id = backend, rig.provider, None, RUN
    return leg


def _measuring(rig: Rig) -> list[str]:
    return [info.sandbox_id for info in rig.provider.find({RUN_ID_KEY: RUN}, role=MEASUREMENT_ROLE)]


async def _earlier_leg_dies_measuring(tmp_path: Path, rig: Rig) -> list[str]:
    """Two tests of an earlier leg in flight when its container was killed, and a create of that leg the service made
    after it died; the sandboxes it left measuring."""

    dying = rig.another_backend()
    dying._pool.release = lambda runner: None  # type: ignore[method-assign]  # a dead process closes nothing

    def die(attempt: object) -> None:
        raise _Crash

    dying._wait = die  # type: ignore[method-assign]
    first = _leg(tmp_path, dying, rig, parallel=2)
    await first.toolbox.start()
    sent = [await first.toolbox.submit_test(req("b1")), await first.toolbox.submit_test(req("b2"))]
    await asyncio.wait_for(first.toolbox.drain(), 30)
    for request_id in sent:
        record = rig.backend.journal.latest(request_id)
        assert record is not None and record.state == STATE_SENT, "the leg died while its runner measured"
    await RealToolbox.close(first.toolbox)  # its tasks die with it; the dead leg's backend ends no sandbox
    unjournalled = rig.provider.create("runner-1", role=MEASUREMENT_ROLE, timeout_sec=600, metadata={RUN_ID_KEY: RUN})
    left = _measuring(rig)
    assert unjournalled.sandbox_id in left and len(left) == 3
    return left


async def test_the_take_up_of_a_later_leg_ends_the_sandboxes_the_earlier_leg_left_measuring(
    tmp_path: Path, service
) -> None:
    rig = service()
    elsewhere = rig.provider.create("runner-1", role=MEASUREMENT_ROLE, timeout_sec=600, metadata={RUN_ID_KEY: "run-2"})
    await _earlier_leg_dies_measuring(tmp_path, rig)
    created: list[object] = []
    real_create = rig.provider.create
    rig.provider.create = lambda *a, **k: created.append(k) or real_create(*a, **k)  # type: ignore[method-assign]
    later = _leg(tmp_path, rig.another_backend(), rig, parallel=1)  # one recovered test runs, the other waits

    await later.toolbox.start()
    withdrawn = await later.toolbox.withdraw_all("infra_failure")

    assert len(withdrawn) == 2, "both of the earlier leg's tests are withdrawn"
    assert _measuring(rig) == [], "no sandbox of the earlier leg is left holding the grant"
    assert created == [], "nothing is measured again"
    assert [info.sandbox_id for info in rig.provider.find({}, role=MEASUREMENT_ROLE)] == [elsewhere.sandbox_id]
    await later.toolbox.close()


async def test_only_the_first_withdrawal_ends_the_earlier_legs_sandboxes(tmp_path: Path, service) -> None:
    rig = service()
    left = await _earlier_leg_dies_measuring(tmp_path, rig)
    later = _leg(tmp_path, rig.another_backend(), rig, parallel=2)
    await later.toolbox.start()
    await later.toolbox.withdraw_all("infra_failure")
    assert not set(left) & set(_measuring(rig))
    ended: list[str] = []
    real_discard = later.toolbox.backend.pool.discard
    later.toolbox.backend.pool.discard = lambda sandbox_id: ended.append(sandbox_id) or real_discard(sandbox_id)  # type: ignore[method-assign]

    request_id = await later.toolbox.submit_test(req("b3"))
    await _until(lambda: len(_measuring(rig)) == 1, "the leg's own test to measure")
    await later.toolbox.withdraw_all("session stopped")

    assert ended == [], "the earlier leg's sandboxes were ended once; the leg's own is ended by its withdrawn test"
    assert later.toolbox._log.get(request_id).outcome == {"withdrawal": "session stopped"}  # type: ignore[union-attr]
    await _until(lambda: _measuring(rig) == [], "the withdrawn test's sandbox to end")
    await later.toolbox.close()


async def _until(condition, what: str, seconds: float = 30.0) -> None:  # type: ignore[no-untyped-def]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while not condition():
        assert loop.time() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.02)
