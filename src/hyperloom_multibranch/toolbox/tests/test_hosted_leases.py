# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch's sandbox lease on the hosted backend, against the E2B stand-in: real envd processes behind a simulated
control plane that caps the live sandboxes of a role (429 ``grant_quota_exceeded``) and can be made to fail a call.

The tests skip where the stand-in cannot run (it needs root, ``unshare`` and an envd binary, ``HYPERLOOM_E2B_ENVD``).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.sandboxes import BRANCH_ID_KEY, PURPOSE_KEY
from hyperloom.sandbox.e2b import E2BSandboxProvider, RetryPolicy
from hyperloom.sandbox.socket import (
    IDEMPOTENCY_KEY,
    CreateOptions,
    Role,
    Sandbox,
    SandboxDenied,
    SandboxGone,
    SandboxInfo,
    SandboxQuotaExceeded,
    SandboxUnavailable,
)
from hyperloom.sandbox.tests.e2b_standin import ENVD_ENV, StandIn, StandInTemplate, unavailable_reason
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import LeaseCharge
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, BranchLease, MeasurementCapacity, RequestRejected
from hyperloom_multibranch.toolbox.hosted_leases import (
    LEASE_PURPOSE,
    LIFETIME_MARGIN_SEC,
    RUN_ID_KEY,
    DevSession,
    GrantLimits,
    HostedBranchLeases,
    LeaseTuning,
)

#: The ``dev`` templates by card count; the names are opaque, the cards are the keys.
_BY_CARDS = {1: "run-dev-a1", 2: "run-dev-b7", 4: "run-dev-c3", 8: "run-dev-d0"}
_NAMES = tuple(_BY_CARDS.values())
_RETRY = RetryPolicy(budget_sec=3.0, first_delay_sec=0.02, max_delay_sec=0.1)
#: Quick enough for a test, slow enough that the second attempt is not the first.
_FAST = LeaseTuning(wait_sec=20.0, first_backoff_sec=0.02, max_backoff_sec=0.05)
_QUOTA = "grant_quota_exceeded: the grant has no room for another sandbox"


class _Recorder:
    """A provider that remembers the metadata of each create it was asked for, and how long it was told to wait."""

    name = "recorder"

    def __init__(self, inner: E2BSandboxProvider) -> None:
        self._inner = inner
        self.creates: list[tuple[str, Mapping[str, str]]] = []
        self.waits: list[float] = []
        self.refused = 0

    def create(
        self,
        template: str,
        *,
        role: Role,
        timeout_sec: int,
        metadata: Mapping[str, str],
        options: CreateOptions = CreateOptions(),
    ) -> Sandbox:
        self.creates.append((template, dict(metadata)))
        self.waits.append(options.wait_sec)
        try:
            return self._inner.create(template, role=role, timeout_sec=timeout_sec, metadata=metadata, options=options)
        except SandboxQuotaExceeded:
            self.refused += 1
            raise

    def connect(self, sandbox_id: str, *, role: Role) -> Sandbox:
        return self._inner.connect(sandbox_id, role=role)

    def find(self, metadata: Mapping[str, str], *, role: Role) -> list[SandboxInfo]:
        return self._inner.find(metadata, role=role)


@dataclass
class Hosted:
    standin: StandIn
    provider: _Recorder
    leases: HostedBranchLeases

    def running(self) -> list[str]:
        return [info.sandbox_id for info in self.provider.find({}, role="dev")]


Make = Callable[..., Hosted]


@pytest.fixture
def make(tmp_path: Path) -> Iterator[Make]:
    """``make(gpus=8, tp=2, sandboxes=8, quota=None, tuning=..., templates=..., retry=..., runtime=600)``: leases on a fresh
    stand-in whose ``dev`` role holds at most ``quota`` live sandboxes."""
    reason = unavailable_reason(os.environ.get(ENVD_ENV))
    if reason is not None:
        pytest.skip(reason)
    stack = ExitStack()
    made: list[Hosted] = []

    def build(
        gpus: int = 8,
        tp: int = 2,
        sandboxes: int = 8,
        quota: int | None = None,
        tuning: LeaseTuning = _FAST,
        templates: Mapping[int, str] | None = None,
        retry: RetryPolicy = _RETRY,
        runtime: int = 600,
    ) -> Hosted:
        workdir = tmp_path / f"standin-{len(made)}"
        workdir.mkdir()
        known = [StandInTemplate(name, "dev") for name in _NAMES]
        caps = {} if quota is None else {"dev": quota}
        standin = stack.enter_context(StandIn(Path(os.environ[ENVD_ENV]), workdir, known, max_sandboxes=caps))
        provider = _Recorder(
            E2BSandboxProvider(standin.keys, api_url=standin.api_url, sandbox_url=standin.sandbox_url, retry=retry)
        )
        session = DevSession("run-1", tp, templates or _BY_CARDS)
        leases = HostedBranchLeases(provider, GrantLimits(sandboxes, gpus, 400.0, runtime), session, tuning)
        made.append(Hosted(standin, provider, leases))
        return made[-1]

    yield build
    for hosted in reversed(made):
        hosted.leases.close()
    stack.close()


async def _until(condition: Callable[[], bool], what: str, seconds: float = 15.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.02)


# -- what a lease is -------------------------------------------------------------------------------------------------


async def test_the_lease_is_a_dev_sandbox_of_the_sessions_cards(make: Make) -> None:
    hosted = make(tp=4)
    lease = await hosted.leases.lease("b02", 4, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert isinstance(lease, BranchLease)
    assert (lease.branch_id, lease.gpu_ids, lease.root, dict(lease.env)) == ("b02", (0, 1, 2, 3), Path("/app/work"), {})
    assert lease.template == _BY_CARDS[4] and lease.remaining_cards == 4
    info = lease.sandbox.info()
    assert info.template == _BY_CARDS[4]
    assert info.metadata[PURPOSE_KEY] == LEASE_PURPOSE and info.metadata[BRANCH_ID_KEY] == "b02"
    assert info.metadata[RUN_ID_KEY] == "run-1"
    assert info.metadata[IDEMPOTENCY_KEY].startswith("run-1:")
    assert hosted.provider.creates[0][0] == _BY_CARDS[4]


def test_the_default_working_directory_of_a_lease_is_the_one_under_the_envd_file_root() -> None:
    assert LeaseTuning().workdir == "/app/work"


async def test_the_lease_hands_over_a_sandbox_that_runs_commands_and_holds_files(make: Make) -> None:
    lease = await make().leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert lease.sandbox.run("echo hello").stdout == "hello\n"
    lease.sandbox.write_file(f"{lease.root}/note.txt", b"data")
    assert lease.sandbox.read_file(f"{lease.root}/note.txt") == b"data"
    assert lease.sandbox.list_dir(str(lease.root)) == ["note.txt"]
    # The file API and the processes agree on the working directory: a process sees the file the API wrote.
    assert lease.sandbox.run(f"cat {lease.root}/note.txt").stdout == "data"


@pytest.mark.parametrize("asked", [0, -1, 1, 3, 4])
async def test_a_lease_has_the_sessions_cards_and_nothing_is_created_for_another_number(make: Make, asked: int) -> None:
    hosted = make(tp=2)
    with pytest.raises(RequestRejected) as refused:
        await hosted.leases.lease("b02", asked, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert refused.value.code == "invalid_request" and "invalid_request" in REJECTION_CODES
    assert hosted.provider.creates == [] and hosted.running() == []
    assert hosted.leases.capacity().leased_cards == 0


# -- accounting ------------------------------------------------------------------------------------------------------


async def test_a_lease_counts_as_leased_cards_and_release_gives_them_back_and_ends_the_sandbox(make: Make) -> None:
    hosted = make(gpus=8, tp=2)
    assert hosted.leases.tests_at_once() == 4
    lease = await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert hosted.leases.capacity() == MeasurementCapacity(8, 8, leased_cards=2)
    assert lease.remaining_cards == 6 and hosted.leases.tests_at_once() == 3
    await hosted.leases.release(lease.lease_id)
    assert hosted.leases.capacity() == MeasurementCapacity(8, 8) and hosted.running() == []
    with pytest.raises(SandboxGone):
        lease.sandbox.info()
    await hosted.leases.release(lease.lease_id)
    await hosted.leases.release("a lease that never was")
    assert (
        await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    ).sandbox.sandbox_id != lease.sandbox.sandbox_id


async def test_two_releases_at_once_end_the_sandbox_once(make: Make) -> None:
    hosted = make()
    lease = await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    ended: list[str] = []
    kill = lease.sandbox.kill

    def slow_kill() -> None:
        ended.append(lease.sandbox.sandbox_id)
        time.sleep(0.2)
        kill()

    lease.sandbox.kill = slow_kill  # type: ignore[method-assign]
    await asyncio.gather(hosted.leases.release(lease.lease_id), hosted.leases.release(lease.lease_id))
    assert ended == [lease.sandbox.sandbox_id] and hosted.running() == []


async def test_a_branch_that_holds_a_lease_cannot_take_a_second(make: Make) -> None:
    hosted = make(gpus=8)
    await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    with pytest.raises(RequestRejected) as refused:
        await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert refused.value.code == "lease_exists" and "lease_exists" in REJECTION_CODES
    assert len(hosted.running()) == 1 and hosted.leases.capacity().leased_cards == 2


async def test_a_lease_that_would_leave_less_than_one_test_needs_is_refused_before_anything_is_created(
    make: Make,
) -> None:
    hosted = make(gpus=4, tp=2)
    first = await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    with pytest.raises(RequestRejected) as refused:
        await hosted.leases.lease("b03", 2, charge=LeaseCharge(LeaseKind.DEV, "b03"))
    assert refused.value.code == "lease_leaves_no_measurement" and "lease_leaves_no_measurement" in REJECTION_CODES
    assert len(hosted.provider.creates) == 1 and hosted.leases.capacity().leased_cards == 2
    await hosted.leases.release(first.lease_id)
    assert (await hosted.leases.lease("b03", 2, charge=LeaseCharge(LeaseKind.DEV, "b03"))).remaining_cards == 2


async def test_a_lease_that_would_leave_no_sandbox_for_a_test_is_refused(make: Make) -> None:
    hosted = make(gpus=8, tp=1, sandboxes=1)
    with pytest.raises(RequestRejected) as refused:
        await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert refused.value.code == "lease_leaves_no_measurement" and hosted.provider.creates == []
    hosted = make(gpus=8, tp=1, sandboxes=3)
    await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert hosted.leases.tests_at_once() == 2
    await hosted.leases.lease("b03", 1, charge=LeaseCharge(LeaseKind.DEV, "b03"))
    assert hosted.leases.tests_at_once() == 1
    with pytest.raises(RequestRejected) as refused:
        await hosted.leases.lease("b04", 1, charge=LeaseCharge(LeaseKind.DEV, "b04"))
    assert refused.value.code == "lease_leaves_no_measurement"


async def test_the_sandbox_lives_as_long_as_its_holder_needs_or_the_grant_lets(make: Make) -> None:
    hosted = make(runtime=3600)
    short = await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"), lifetime_sec=100)
    held = await hosted.leases.lease("b03", 2, charge=LeaseCharge(LeaseKind.AUTONOMOUS, "b03"))
    lifetimes = [hosted.standin.plane.lookup(lease.sandbox.sandbox_id).timeout_sec for lease in (short, held)]
    assert lifetimes == [100 + LIFETIME_MARGIN_SEC, 3600 - LIFETIME_MARGIN_SEC]
    assert (short.lifetime_sec, held.lifetime_sec) == tuple(lifetimes)


async def test_the_toolbox_is_told_whenever_cards_come_back(make: Make) -> None:
    hosted = make(gpus=8, tp=2, templates={2: "not-a-template"})
    told: list[str] = []
    hosted.leases.on_release = lambda: told.append("back")
    with pytest.raises(RequestRejected):
        await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert told == [], "a refusal held nothing"
    with pytest.raises(SandboxDenied):
        await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert told == ["back"], "the promise of a lease that failed was given back"
    ok = make(gpus=8, tp=2)
    ok.leases.on_release = lambda: told.append("released")
    lease = await ok.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert told == ["back"]
    await ok.leases.release(lease.lease_id)
    await ok.leases.release(lease.lease_id)
    assert told == ["back", "released"], "once for the release, none for the repeat"
    await asyncio.to_thread(ok.leases.close)
    assert told == ["back", "released", "released"]


# -- a full quota ----------------------------------------------------------------------------------------------------


async def test_a_create_the_quota_refuses_is_repeated_with_one_idempotency_key(make: Make) -> None:
    hosted = make()
    hosted.standin.plane.fail_next("create", 429, times=3, message=_QUOTA)
    lease = await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert (
        len(hosted.provider.creates) == 4 and len({meta[IDEMPOTENCY_KEY] for _, meta in hosted.provider.creates}) == 1
    )
    assert hosted.running() == [lease.sandbox.sandbox_id]
    other = await hosted.leases.lease("b03", 2, charge=LeaseCharge(LeaseKind.DEV, "b03"))
    assert hosted.provider.creates[-1][1][IDEMPOTENCY_KEY] != hosted.provider.creates[0][1][IDEMPOTENCY_KEY]
    assert other.sandbox.sandbox_id != lease.sandbox.sandbox_id


async def test_a_lease_waits_for_the_quota_while_its_cards_stay_promised_and_proceeds_when_it_clears(
    make: Make,
) -> None:
    hosted = make(gpus=8, tp=1, quota=1)
    first = await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    waiting = asyncio.ensure_future(hosted.leases.lease("b03", 1, charge=LeaseCharge(LeaseKind.DEV, "b03")))
    await _until(lambda: len(hosted.provider.creates) >= 3, "the second create to be refused and repeated")
    assert not waiting.done() and hosted.leases.capacity().leased_cards == 2, "the waiting lease promised its card"
    await hosted.leases.release(first.lease_id)
    second = await asyncio.wait_for(waiting, 30)
    assert hosted.running() == [second.sandbox.sandbox_id] and second.branch_id == "b03"


async def test_a_lease_that_the_quota_does_not_clear_for_ends_as_unavailable_and_gives_its_promise_back(
    make: Make,
) -> None:
    hosted = make(gpus=8, tp=1, quota=1, tuning=LeaseTuning(wait_sec=0.3, first_backoff_sec=0.02, max_backoff_sec=0.05))
    first = await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    with pytest.raises(SandboxUnavailable, match="no room in the grant") as failed:
        await asyncio.wait_for(hosted.leases.lease("b03", 1, charge=LeaseCharge(LeaseKind.DEV, "b03")), 10)
    assert not isinstance(failed.value, SandboxQuotaExceeded)
    assert "already has 1 sandboxes" in str(failed.value), "the service's refusal names the limit it counted"
    assert hosted.leases.capacity().leased_cards == 1
    await hosted.leases.release(first.lease_id)
    assert (await hosted.leases.lease("b03", 1, charge=LeaseCharge(LeaseKind.DEV, "b03"))).branch_id == "b03"


# -- how long a lease waits ------------------------------------------------------------------------------------------

#: Sleeps five seconds between creates a full quota refuses, so a lease that waits at all is seen to.
_SLOW_BACKOFF = LeaseTuning(wait_sec=60.0, first_backoff_sec=5.0, max_backoff_sec=5.0)


async def test_a_lease_that_waits_for_nothing_is_refused_at_once_by_a_full_quota(make: Make) -> None:
    hosted = make(gpus=8, tp=1, quota=1, tuning=_SLOW_BACKOFF)
    first = await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable, match="no room in the grant within 0 s") as failed:
        await asyncio.wait_for(hosted.leases.lease("b03", 1, wait_sec=0, charge=LeaseCharge(LeaseKind.DEV, "b03")), 20)
    assert time.monotonic() - started < 3, "a lease that waits for nothing waited out the quota"
    assert not isinstance(failed.value, SandboxQuotaExceeded)
    assert len(hosted.provider.creates) == 2 and hosted.provider.refused == 1, "one create, refused once, not repeated"
    assert hosted.leases.capacity().leased_cards == 1, "the refused lease gave its promise back"
    await hosted.leases.release(first.lease_id)
    assert (
        await hosted.leases.lease("b03", 1, wait_sec=0, charge=LeaseCharge(LeaseKind.DEV, "b03"))
    ).branch_id == "b03"


async def test_a_lease_that_waits_for_nothing_still_gives_the_sandbox_time_to_come_up(make: Make) -> None:
    hosted = make(tuning=LeaseTuning(wait_sec=20.0, first_backoff_sec=0.02, max_backoff_sec=0.05))
    lease = await hosted.leases.lease("b02", 2, wait_sec=0, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert hosted.running() == [lease.sandbox.sandbox_id]
    assert hosted.provider.waits and min(hosted.provider.waits) >= 19.0, "the create was given the tuning's time"


async def test_a_positive_wait_bounds_how_long_a_lease_waits_out_the_quota(make: Make) -> None:
    hosted = make(
        gpus=8, tp=1, quota=1, tuning=LeaseTuning(wait_sec=60.0, first_backoff_sec=0.02, max_backoff_sec=0.05)
    )
    await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable, match="no room in the grant within 0.3 s"):
        await asyncio.wait_for(
            hosted.leases.lease("b03", 1, wait_sec=0.3, charge=LeaseCharge(LeaseKind.DEV, "b03")), 20
        )
    waited = time.monotonic() - started
    assert 0.2 <= waited < 10, f"waited {waited:.2f} s for a wait of 0.3 s"
    assert hosted.provider.refused >= 2, "the quota was retried while the wait lasted"
    assert hosted.leases.capacity().leased_cards == 1


async def test_a_wait_longer_than_the_tuning_waits_that_long_for_the_quota(make: Make) -> None:
    hosted = make(gpus=8, tp=1, quota=1, tuning=LeaseTuning(wait_sec=0.3, first_backoff_sec=0.02, max_backoff_sec=0.05))
    first = await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    waiting = asyncio.ensure_future(
        hosted.leases.lease("b03", 1, wait_sec=30, charge=LeaseCharge(LeaseKind.DEV, "b03"))
    )
    await _until(lambda: hosted.provider.refused >= 3, "the second create to be refused and repeated")
    await asyncio.sleep(0.5)  # longer than the tuning's own wait, which would have ended it
    assert not waiting.done()
    await hosted.leases.release(first.lease_id)
    assert (await asyncio.wait_for(waiting, 30)).branch_id == "b03"


@pytest.mark.parametrize("wait", [-1.0, float("nan"), float("inf")])
async def test_a_wait_that_is_not_a_number_of_seconds_is_refused_before_anything_is_created(
    make: Make, wait: float
) -> None:
    hosted = make()
    with pytest.raises(RequestRejected) as refused:
        await hosted.leases.lease("b02", 2, wait_sec=wait, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert refused.value.code == "invalid_request" and "wait_sec" in refused.value.reason
    assert hosted.provider.creates == [] and hosted.leases.capacity().leased_cards == 0


# -- every exit path -------------------------------------------------------------------------------------------------


async def test_a_create_the_service_refuses_gives_the_promise_back(make: Make) -> None:
    hosted = make(templates={2: "not-a-template"})
    with pytest.raises(SandboxDenied):
        await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    assert hosted.leases.capacity().leased_cards == 0 and hosted.running() == []
    with pytest.raises(SandboxDenied):
        await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))


async def test_a_lease_cancelled_while_it_waits_for_the_quota_gives_its_promise_back(make: Make) -> None:
    hosted = make(gpus=8, tp=1, quota=1)
    await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    waiting = asyncio.ensure_future(hosted.leases.lease("b03", 1, charge=LeaseCharge(LeaseKind.DEV, "b03")))
    await _until(lambda: hosted.leases.capacity().leased_cards == 2, "the second lease to promise its card")
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert hosted.leases.capacity().leased_cards == 1


async def test_a_lease_cancelled_while_the_sandbox_is_being_made_ends_the_sandbox_that_comes_up(make: Make) -> None:
    hosted = make()
    hosted.standin.plane.create_delay_sec = 1.0
    waiting = asyncio.ensure_future(hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02")))
    await _until(lambda: hosted.standin.plane.starting() == 1, "the service to start the sandbox")
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert hosted.leases.capacity().leased_cards == 0
    await _until(lambda: hosted.standin.plane.starting() == 0, "the service to finish making the sandbox")
    assert len(hosted.standin.host.pids) == 1, "the service made the sandbox the lease abandoned"
    await _until(lambda: hosted.standin.plane.live_sandboxes() == [] and hosted.running() == [], "the sandbox to end")


async def test_a_sandbox_that_comes_up_after_the_toolbox_closed_is_ended_and_not_leased(make: Make) -> None:
    hosted = make()
    hosted.standin.plane.create_delay_sec = 0.8
    waiting = asyncio.ensure_future(hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02")))
    await _until(lambda: hosted.standin.plane.starting() == 1, "the service to start the sandbox")
    await asyncio.to_thread(hosted.leases.close)
    with pytest.raises(SandboxUnavailable, match="closing"):
        await asyncio.wait_for(waiting, 30)
    assert len(hosted.standin.host.pids) == 1 and hosted.running() == []
    assert hosted.leases.capacity().leased_cards == 0


async def test_a_release_the_service_cannot_carry_out_still_gives_the_cards_back(make: Make) -> None:
    hosted = make(retry=RetryPolicy(budget_sec=0.2, first_delay_sec=0.02, max_delay_sec=0.05))
    lease = await hosted.leases.lease("b02", 2, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    hosted.standin.plane.fail_next("kill", 500, times=50)
    await hosted.leases.release(lease.lease_id)
    assert hosted.leases.capacity().leased_cards == 0
    assert hosted.running() == [lease.sandbox.sandbox_id], "the sandbox is left to its own timeout"


async def test_close_ends_every_lease_and_the_sandboxes_of_this_run_no_lease_holds_and_refuses_new_ones(
    make: Make,
) -> None:
    hosted = make(gpus=8, tp=1)
    a, b = (
        await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02")),
        await hosted.leases.lease("b03", 1, charge=LeaseCharge(LeaseKind.DEV, "b03")),
    )
    ours = {PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: "run-1", BRANCH_ID_KEY: "gone"}
    stray = hosted.provider.create(_BY_CARDS[1], role="dev", timeout_sec=60, metadata=ours)
    others = [
        hosted.provider.create(_BY_CARDS[1], role="dev", timeout_sec=60, metadata=metadata)
        for metadata in ({**ours, RUN_ID_KEY: "run-2"}, {PURPOSE_KEY: "other", RUN_ID_KEY: "run-1"})
    ]
    await asyncio.to_thread(hosted.leases.close)
    assert sorted(hosted.running()) == sorted(other.sandbox_id for other in others), "another run's are left alone"
    assert {a.sandbox.sandbox_id, b.sandbox.sandbox_id, stray.sandbox_id}.isdisjoint(hosted.running())
    assert hosted.leases.capacity().leased_cards == 0
    with pytest.raises(SandboxUnavailable, match="closing"):
        await hosted.leases.lease("b04", 1, charge=LeaseCharge(LeaseKind.DEV, "b04"))


async def test_end_leftovers_ends_this_runs_unheld_sandboxes_and_keeps_its_own_leases(make: Make) -> None:
    hosted = make(gpus=8, tp=1)
    held = await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    ours = {PURPOSE_KEY: LEASE_PURPOSE, RUN_ID_KEY: "run-1", BRANCH_ID_KEY: "of-a-dead-process"}
    leftover = hosted.provider.create(_BY_CARDS[1], role="dev", timeout_sec=60, metadata=ours)
    others = [
        hosted.provider.create(_BY_CARDS[1], role="dev", timeout_sec=60, metadata=metadata)
        for metadata in ({**ours, RUN_ID_KEY: "run-2"}, {PURPOSE_KEY: "other", RUN_ID_KEY: "run-1"})
    ]

    await asyncio.to_thread(hosted.leases.end_leftovers)

    expected = sorted([held.sandbox.sandbox_id, *(other.sandbox_id for other in others)])
    assert sorted(hosted.running()) == expected, "only the unheld sandbox of this run's leases is ended"
    assert leftover.sandbox_id not in hosted.running()
    assert hosted.leases.capacity().leased_cards == 1


async def test_a_lease_still_waiting_when_the_toolbox_closes_is_refused(make: Make) -> None:
    hosted = make(gpus=8, tp=1, quota=1, tuning=LeaseTuning(wait_sec=60.0, first_backoff_sec=5.0, max_backoff_sec=5.0))
    await hosted.leases.lease("b02", 1, charge=LeaseCharge(LeaseKind.DEV, "b02"))
    waiting = asyncio.ensure_future(hosted.leases.lease("b03", 1, charge=LeaseCharge(LeaseKind.DEV, "b03")))
    await _until(lambda: hosted.provider.refused == 1, "the second create to be refused")
    await asyncio.sleep(0.3)  # the lease is now asleep for its first backoff, five seconds
    started = time.monotonic()
    await asyncio.to_thread(hosted.leases.close)
    with pytest.raises(SandboxUnavailable, match="closing"):
        await asyncio.wait_for(waiting, 30)
    assert time.monotonic() - started < 4, "the wait did not notice the close"


# -- the pieces ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"max_sandboxes": 0},
        {"max_gpus": 0},
        {"max_runtime_sec": 0},
        {"gpu_hours": 0},
    ],
)
def test_a_grant_has_something_to_run_on(bad: dict[str, Any]) -> None:
    fields: dict[str, Any] = {"max_sandboxes": 4, "max_gpus": 8, "gpu_hours": 10.0, "max_runtime_sec": 60}
    GrantLimits(**fields)
    with pytest.raises(ValueError, match="at least one sandbox"):
        GrantLimits(**{**fields, **bad})


def test_a_sandbox_lives_its_holders_term_and_a_margin_within_the_grants_runtime() -> None:
    leases = HostedBranchLeases(object(), GrantLimits(4, 8, 10.0, 7200), DevSession("run-1", 2, _BY_CARDS))  # type: ignore[arg-type]
    assert leases.lifetime_sec(None) == 7200 - LIFETIME_MARGIN_SEC
    assert leases.lifetime_sec(1799.5) == 1800 + LIFETIME_MARGIN_SEC
    assert leases.lifetime_sec(6 * 3600) == 7200 - LIFETIME_MARGIN_SEC


def test_the_session_needs_a_template_of_its_own_card_count_and_a_run() -> None:
    with pytest.raises(ValueError, match="no dev template has the session's 4 cards"):
        DevSession("run-1", 4, {1: "a", 2: "b"})
    with pytest.raises(ValueError, match="run id"):
        DevSession("", 1, {1: "a"})
    assert DevSession("run-1", 2, {1: "a", 2: "b"}).template == "b"
