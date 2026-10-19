# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Same-state deduplication and the life of a claim, as a specialist runtime and the toolbox see them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.notebook import (
    AlreadyTested,
    CeilingTooLow,
    ClaimHeld,
    ClaimKey,
    ClaimLost,
    ConflictingWrite,
    Role,
    UnknownRequest,
    WriterNotAllowed,
)
from hyperloom_multibranch.contracts.notebook.records import BaselineRecord, ToolResultPointer
from hyperloom_multibranch.contracts.notebook.vocab import ClaimStatus, DedupStatus
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.local.claims import LeaseTaken
from hyperloom_multibranch.notebook.local.segments import EntryKind
from hyperloom_multibranch.notebook.tests.factories import TTL, FakeClock, delta, intent, request, result

Open = Callable[..., LocalNotebook]
ACTION = delta("--max-num-seqs 256").fingerprint()
OTHER_ACTION = delta("--max-num-seqs 512").fingerprint()


async def _measured(open_as: Open, branch: str, parent: str, request_id: str, change_args: str) -> None:
    specialist, toolbox = open_as(Role.SPECIALIST_RUNTIME, branch), open_as(Role.TOOLBOX)
    target = intent(branch, parent)
    change = delta(change_args)
    claim = await specialist.acquire(target, change.fingerprint(), ttl_seconds=TTL)
    await toolbox.write_request(request(request_id, target, change))
    await toolbox.attach_request(claim.key, request_id, ttl_seconds=TTL)
    await toolbox.write_result(result(request_id))


async def test_a_claimed_intent_is_in_progress_for_every_other_branch_on_that_state(open_as: Open) -> None:
    claim = await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(intent("s1", "c0"), ACTION, ttl_seconds=TTL)

    with pytest.raises(ClaimHeld) as held:
        await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(intent("s2", "c0"), OTHER_ACTION, ttl_seconds=TTL)

    assert (claim.status, claim.holder, claim.generation) == (ClaimStatus.HELD, "s1", 1)
    assert held.value.answer.status is DedupStatus.IN_PROGRESS
    assert held.value.answer.claim is not None and held.value.answer.claim.holder == "s1"


async def test_the_same_action_in_progress_under_another_intent_is_held_too(open_as: Open) -> None:
    await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(intent("s1", "c0"), ACTION, ttl_seconds=TTL)

    with pytest.raises(ClaimHeld):
        await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(
            intent("s2", "c0", "decode batch", "larger batch"), ACTION, ttl_seconds=TTL
        )


async def test_claims_made_before_their_change_is_written_meet_only_by_intent(open_as: Open) -> None:
    no_change = RecipeDelta().fingerprint()
    await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(intent("s1", "c0"), no_change, ttl_seconds=TTL)
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")

    other = await s2.acquire(intent("s2", "c0", "decode attention", "page size 16"), no_change, ttl_seconds=TTL)
    with pytest.raises(ClaimHeld):
        await s2.acquire(intent("s2", "c0"), no_change, ttl_seconds=TTL)

    assert other.holder == "s2"


async def test_a_suspected_duplicate_needs_a_reason_and_the_decision_is_logged_for_q6(open_as: Open) -> None:
    await _measured(open_as, "s1", "c0", "r1", "--max-num-seqs 256")
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")

    with pytest.raises(AlreadyTested) as refused:
        await s2.acquire(intent("s2", "c0"), OTHER_ACTION, ttl_seconds=TTL)
    claim = await s2.acquire(
        intent("s2", "c0"), OTHER_ACTION, ttl_seconds=TTL, reason="512 seqs fits after c0's kv change"
    )

    assert refused.value.answer.status is DedupStatus.SUSPECTED
    assert [record.request_id for record in refused.value.answer.records] == ["r1"]
    assert claim.holder == "s2"
    decisions = [entry.detail for entry in s2._state().entries if entry.kind is EntryKind.CLAIM]
    assert [(row["found"], row["acquired"], row["reason"]) for row in decisions[-2:]] == [
        ("suspected", False, ""),
        ("suspected", True, "512 seqs fits after c0's kv change"),
    ]


async def test_the_same_action_measured_on_that_state_is_refused_whatever_the_reason(open_as: Open) -> None:
    await _measured(open_as, "s1", "c0", "r1", "--max-num-seqs 256")

    with pytest.raises(AlreadyTested) as refused:
        await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(
            intent("s2", "c0", "other", "framing"), ACTION, ttl_seconds=TTL, reason="please"
        )

    assert refused.value.answer.status is DedupStatus.TESTED


async def test_the_same_intent_on_another_state_is_only_a_reference(open_as: Open) -> None:
    await _measured(open_as, "s1", "c0", "r1", "--max-num-seqs 256")
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")
    target = intent("s2", "c9")

    answer = await s2.lookup(ClaimKey("c9", target.fingerprint), ACTION)
    claim = await s2.acquire(target, ACTION, ttl_seconds=TTL)

    assert answer.status is DedupStatus.FREE
    assert [record.request_id for record in answer.elsewhere] == ["r1"]
    assert claim.key.parent_commit == "c9"


async def test_an_expired_claim_can_be_taken_over_and_its_old_holder_loses_it(open_as: Open, clock: FakeClock) -> None:
    s1 = open_as(Role.SPECIALIST_RUNTIME, "s1")
    first = await s1.acquire(intent("s1", "c0"), ACTION, ttl_seconds=TTL)
    clock.advance(2 * TTL)

    taken = await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(intent("s2", "c0"), ACTION, ttl_seconds=TTL)

    with pytest.raises(ClaimLost):
        await s1.renew(first.key, ttl_seconds=TTL)
    assert (taken.holder, taken.generation) == ("s2", first.generation + 1)


async def test_a_held_claim_nobody_renewed_reads_as_expired_until_renewed(open_as: Open, clock: FakeClock) -> None:
    s1 = open_as(Role.SPECIALIST_RUNTIME, "s1")
    claim = await s1.acquire(intent("s1", "c0"), ACTION, ttl_seconds=TTL)
    clock.advance(2 * TTL)

    expired = await s1.lookup(claim.key, ACTION)
    renewed = await s1.renew(claim.key, ttl_seconds=TTL)

    assert expired.status is DedupStatus.FREE
    assert renewed.status is ClaimStatus.HELD and renewed.expires_at > clock.now


async def test_a_restarted_runtime_keeps_its_claims_and_the_toolbox_carries_an_attached_one(open_as: Open) -> None:
    target = intent("s1", "c0")
    claim = await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(target, ACTION, ttl_seconds=TTL)
    toolbox = open_as(Role.TOOLBOX)

    with pytest.raises(ClaimLost):
        await toolbox.renew(claim.key, ttl_seconds=TTL)
    restarted = await open_as(Role.SPECIALIST_RUNTIME, "s1").renew(claim.key, ttl_seconds=TTL)
    await toolbox.write_request(request("r1", target))
    long_ttl = 10 * TTL
    attached = await toolbox.attach_request(claim.key, "r1", ttl_seconds=long_ttl)
    carried = await toolbox.renew(claim.key, ttl_seconds=long_ttl)

    assert restarted.holder == "s1"
    assert (attached.request_id, carried.request_id) == ("r1", "r1")
    assert carried.expires_at > attached.expires_at


async def test_release_frees_the_key_and_stopping_a_branch_releases_all_it_holds(open_as: Open) -> None:
    s1 = open_as(Role.SPECIALIST_RUNTIME, "s1")
    one = await s1.acquire(intent("s1", "c0"), ACTION, ttl_seconds=TTL)
    await s1.acquire(intent("s1", "c0", "attention", "split kv"), OTHER_ACTION, ttl_seconds=TTL)
    await s1.acquire(intent("s1", "c1", "attention", "split kv"), OTHER_ACTION, ttl_seconds=TTL)

    with pytest.raises(ClaimLost):
        await open_as(Role.SPECIALIST_RUNTIME, "s2").release(one.key)
    released = await s1.release(one.key)
    stopped = await open_as(Role.SCHEDULER).release_branch("s1")
    again = await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(intent("s2", "c0"), ACTION, ttl_seconds=TTL)

    assert released.status is ClaimStatus.RELEASED
    assert sorted(claim.key.parent_commit for claim in stopped) == ["c0", "c1"]
    assert all(claim.status is ClaimStatus.RELEASED for claim in stopped)
    assert again.holder == "s2"


async def test_a_runtime_claims_only_for_its_own_branch_and_only_runtimes_claim(open_as: Open) -> None:
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(intent("s2", "c0"), ACTION, ttl_seconds=TTL)
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.TOOLBOX).acquire(intent("s1", "c0"), ACTION, ttl_seconds=TTL)
    with pytest.raises(WriterNotAllowed):
        await open_as(Role.GATE).release_branch("s1")


async def test_attaching_a_request_checks_it_measures_that_claim(open_as: Open) -> None:
    target = intent("s1", "c0")
    claim = await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(target, ACTION, ttl_seconds=TTL)
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request("r-other", intent("s1", "c0", "attention", "split kv")))

    with pytest.raises(ConflictingWrite):
        await toolbox.attach_request(claim.key, "r-other", ttl_seconds=TTL)
    with pytest.raises(UnknownRequest):
        await toolbox.attach_request(claim.key, "r-missing", ttl_seconds=TTL)


async def test_a_finished_or_released_claim_cannot_be_released_or_attached_again(open_as: Open) -> None:
    await _measured(open_as, "s1", "c0", "r1", "--max-num-seqs 256")
    s1, toolbox = open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.TOOLBOX)
    done = ClaimKey("c0", intent("s1", "c0").fingerprint)
    other = intent("s1", "c0", "attention", "split kv")
    released = await s1.acquire(other, OTHER_ACTION, ttl_seconds=TTL)
    await s1.release(released.key)
    await toolbox.write_request(request("r2", other))

    with pytest.raises(ClaimLost):
        await open_as(Role.SCHEDULER).release(done)
    with pytest.raises(ClaimLost):
        await toolbox.attach_request(released.key, "r2", ttl_seconds=TTL)


async def test_losing_the_race_for_a_free_claim_is_refused_as_held_and_logged(
    open_as: Open, monkeypatch: pytest.MonkeyPatch
) -> None:
    s1, s2 = open_as(Role.SPECIALIST_RUNTIME, "s1"), open_as(Role.SPECIALIST_RUNTIME, "s2")
    winner = await s1.acquire(intent("s1", "c0"), ACTION, ttl_seconds=TTL)

    def lose(claim: object, now: float) -> None:
        raise LeaseTaken(winner)

    monkeypatch.setattr(s2._leases, "acquire", lose)
    monkeypatch.setattr(s2._leases, "all_current", list)
    with pytest.raises(ClaimHeld) as held:
        await s2.acquire(intent("s2", "c0"), ACTION, ttl_seconds=TTL)

    decision = [entry.detail for entry in s2._state().entries if entry.kind is EntryKind.CLAIM][-1]
    assert held.value.answer.claim == winner
    assert (decision["found"], decision["acquired"]) == ("in_progress", False)


async def _bar_of_three_pct(open_as: Open) -> None:
    await open_as(Role.GATE).write_baseline(BaselineRecord("c0", {"output_tps": 1000.0}, {}, {"output_tps": 3.0}, ()))


async def test_an_intent_whose_ceiling_is_below_the_bar_is_refused_and_logged(open_as: Open) -> None:
    s1 = open_as(Role.SPECIALIST_RUNTIME, "s1")
    await _bar_of_three_pct(open_as)
    shares = ToolResultPointer("s1", "job-1", "/a/analysis.json", "trace analysed", "c0", "p1", {"moe_gemm": 2.0})
    await open_as(Role.TOOLBOX).notify_tool_result(shares)

    with pytest.raises(CeilingTooLow) as low:
        await s1.acquire(replace(intent("s1", "c0"), target_share_pct=2.0), ACTION, ttl_seconds=TTL)
    granted = await s1.acquire(replace(intent("s1", "c0"), target_share_pct=20.0), ACTION, ttl_seconds=TTL)

    refused, kept = [entry for entry in s1._state().entries if entry.kind is EntryKind.CLAIM]
    assert (round(low.value.ceiling_pct, 2), low.value.bar_pct) == (2.04, 3.0)
    assert low.value.profiled is not None and low.value.profiled.job_id == "job-1"
    assert (refused.detail["acquired"], round(refused.detail["ceiling_pct"], 2), refused.detail["bar_pct"]) == (
        False,
        2.04,
        3.0,
    )
    assert "ceiling 2.04% below the bar of 3.00%" in refused.summary
    assert granted.status is ClaimStatus.HELD and kept.detail["ceiling_pct"] == 25.0


async def test_no_intent_is_refused_for_its_ceiling_before_a_baseline_or_without_a_share(open_as: Open) -> None:
    s1 = open_as(Role.SPECIALIST_RUNTIME, "s1")

    early = await s1.acquire(replace(intent("s1", "c0"), target_share_pct=0.5), ACTION, ttl_seconds=TTL)
    await _bar_of_three_pct(open_as)
    unstated = await s1.acquire(intent("s1", "c1"), ACTION, ttl_seconds=TTL)

    assert early.status is unstated.status is ClaimStatus.HELD


async def test_op_shares_are_the_latest_a_tool_reported_on_the_commit(open_as: Open) -> None:
    toolbox = open_as(Role.TOOLBOX)
    for job, share in (("job-1", 30.0), ("job-2", 41.5)):
        await toolbox.notify_tool_result(
            ToolResultPointer("s1", job, "/a", "trace analysed", "c1", "", {"gemm": share})
        )
    await toolbox.notify_tool_result(ToolResultPointer("s1", "job-3", "/build.log", "build done"))

    latest = await toolbox.op_shares("c1")

    assert latest is not None and (latest.job_id, dict(latest.shares)) == ("job-2", {"gemm": 41.5})
    assert await toolbox.op_shares("c0") is None
    assert (await toolbox.snapshot()).op_shares == {"c1": latest}


async def test_a_late_result_does_not_complete_the_claim_another_branch_took_over(
    open_as: Open, clock: FakeClock
) -> None:
    target = intent("s1", "c0")
    await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(target, ACTION, ttl_seconds=TTL)
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request("r1", target))
    clock.advance(2 * TTL)
    taken = await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(intent("s2", "c0"), OTHER_ACTION, ttl_seconds=TTL)

    await toolbox.write_result(result("r1"))

    current = toolbox._leases.current(taken.key)
    assert current is not None
    assert (current.holder, current.status, current.request_id) == ("s2", ClaimStatus.HELD, None)
