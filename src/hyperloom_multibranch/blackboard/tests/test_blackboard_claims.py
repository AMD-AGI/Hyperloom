# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deduplication on a state and across branches, and the life of a claim, as a specialist runtime and the toolbox see
them; a branch refused while another branch tests the same intent hears how that test ended."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.blackboard import (
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
from hyperloom_multibranch.contracts.blackboard.peers import Intent, IntentLayer, Purpose, TestKind, Validity
from hyperloom_multibranch.contracts.blackboard.records import (
    REPLAY_TARGET,
    BaselineRecord,
    InboxItem,
    ToolResultPointer,
    Withdrawal,
)
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus, Conclusion, DedupStatus, InboxKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.blackboard.local.claims import LeaseTaken
from hyperloom_multibranch.blackboard.local.segments import EntryKind
from hyperloom_multibranch.blackboard.tests.factories import TTL, FakeClock, delta, intent, judgement, request, result

Open = Callable[..., LocalBlackboard]
ACTION = delta("--max-num-seqs 256").fingerprint()
OTHER_ACTION = delta("--max-num-seqs 512").fingerprint()
REPLAY_ACTION = delta("--enable-chunked-prefill").fingerprint()


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


async def test_a_test_measured_without_a_change_answers_only_for_its_own_intent(open_as: Open) -> None:
    no_change = RecipeDelta().fingerprint()
    tried = intent("s1", "c0")
    claim = await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(tried, no_change, ttl_seconds=TTL)
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request("r1", tried, RecipeDelta()))
    await toolbox.attach_request(claim.key, "r1", ttl_seconds=TTL)
    await toolbox.write_result(result("r1"))
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")

    other = await s2.lookup(ClaimKey("c0", intent("s2", "c0", "attention", "fused rope").fingerprint), no_change)
    same = await s2.lookup(ClaimKey("c0", tried.fingerprint), no_change)

    assert (other.status, same.status) == (DedupStatus.FREE, DedupStatus.SUSPECTED)


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


async def _in_flight(open_as: Open, branch: str, parent: str, request_id: str) -> None:
    tried = intent(branch, parent)
    claim = await open_as(Role.SPECIALIST_RUNTIME, branch).acquire(tried, ACTION, ttl_seconds=TTL)
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request(request_id, tried, delta("--max-num-seqs 256")))
    await toolbox.attach_request(claim.key, request_id, ttl_seconds=TTL)


def _decisions(handle: LocalBlackboard) -> list[dict[str, object]]:
    return [entry.detail for entry in handle._state().entries if entry.kind is EntryKind.CLAIM]


async def test_an_intent_another_branch_is_testing_is_refused_naming_the_branch_and_its_test(open_as: Open) -> None:
    await _in_flight(open_as, "s1", "c1", "r1")
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")

    looked = await s2.lookup(ClaimKey("c2", intent("s2", "c2").fingerprint), OTHER_ACTION)
    with pytest.raises(ClaimHeld) as held:
        await s2.acquire(intent("s2", "c2"), OTHER_ACTION, ttl_seconds=TTL, reason="c2 is another head")

    answer = held.value.answer
    assert (looked.status, answer.status) == (DedupStatus.TESTING_ELSEWHERE, DedupStatus.TESTING_ELSEWHERE)
    assert answer.claim is not None and (answer.claim.holder, answer.claim.key.parent_commit) == ("s1", "c1")
    assert [record.request_id for record in answer.records] == ["r1"]
    (decision,) = _decisions(s2)[-1:]
    assert (decision["found"], decision["acquired"], decision["matched"], decision["holders"]) == (
        "testing_elsewhere",
        False,
        ["r1"],
        ["s1"],
    )


async def test_the_intent_is_free_again_once_the_other_branchs_test_has_its_result(open_as: Open) -> None:
    await _in_flight(open_as, "s1", "c1", "r1")
    await open_as(Role.TOOLBOX).write_result(result("r1"))

    claim = await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(intent("s2", "c2"), OTHER_ACTION, ttl_seconds=TTL)

    assert claim.holder == "s2"


async def test_a_claim_held_before_its_test_is_written_refuses_the_intent_on_other_heads_too(open_as: Open) -> None:
    await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(intent("s1", "c1"), ACTION, ttl_seconds=TTL)
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")

    with pytest.raises(ClaimHeld) as held:
        await s2.acquire(intent("s2", "c2"), OTHER_ACTION, ttl_seconds=TTL)

    assert (held.value.answer.status, held.value.answer.records) == (DedupStatus.TESTING_ELSEWHERE, ())
    (decision,) = _decisions(s2)[-1:]
    assert (decision["matched"], decision["holders"]) == ([], ["s1"])


async def test_neither_a_replay_nor_the_branchs_own_test_on_an_older_head_refuses_an_intent(open_as: Open) -> None:
    replayed = intent("s1", "c1", REPLAY_TARGET, "inference:champion", IntentLayer.CONFIG)
    await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(replayed, REPLAY_ACTION, ttl_seconds=TTL)
    await _in_flight(open_as, "s1", "c1", "r1")

    again = await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(
        replace(replayed, branch="s2", parent_commit="c2"), REPLAY_ACTION, ttl_seconds=TTL
    )
    own = await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(intent("s1", "c3"), OTHER_ACTION, ttl_seconds=TTL)

    assert (again.holder, own.holder) == ("s2", "s1")


async def _end(open_as: Open, request_id: str, ending: str) -> None:
    toolbox = open_as(Role.TOOLBOX)
    if ending == "withdrawn":
        await toolbox.write_withdrawal(Withdrawal(request_id, "branch stopped"))
    elif ending == "unmeasured":
        await toolbox.write_result(result(request_id, validity=Validity.MEASUREMENT_FAILED, failure_reason="node lost"))
    else:
        await toolbox.write_result(result(request_id))
        await open_as(Role.GATE).write_judgement(request_id, judgement(4.0, (request_id,)), Conclusion.RECORDED_ONLY)


def _heard(items: tuple[InboxItem, ...]) -> list[InboxItem]:
    return [item for item in items if item.kind is InboxKind.PEER_OUTCOME]


@pytest.mark.parametrize(
    ("ending", "says"),
    [
        ("judged", "recorded_only: improved +4.00% on r1"),
        ("withdrawn", "withdrawn r1 on s1: branch stopped"),
        ("unmeasured", "measurement_failed r1 on s1"),
    ],
)
async def test_a_branch_refused_for_another_branchs_test_hears_once_how_it_ended(
    open_as: Open, ending: str, says: str
) -> None:
    await _in_flight(open_as, "s1", "c1", "r1")
    s1, s2, s3 = (open_as(Role.SPECIALIST_RUNTIME, branch) for branch in ("s1", "s2", "s3"))
    with pytest.raises(ClaimHeld):
        await s2.acquire(intent("s2", "c2"), OTHER_ACTION, ttl_seconds=TTL)
    before = await s2.poll("s2")

    await _end(open_as, "r1", ending)
    page = await s2.poll("s2", before.cursor)

    (heard,) = _heard(page.items)
    assert (heard.branch, heard.pointer) == ("s2", "r1")
    assert heard.summary.startswith("s1's test of moe gemm / tuned kernel, which refused your claim, ended: ")
    assert says in heard.summary and "blackboard_replay with request_id r1" in heard.summary
    assert (await s2.poll("s2", page.cursor)).items == ()
    assert _heard((await s1.poll("s1")).items) == [] and _heard((await s3.poll("s3")).items) == []


async def test_a_branch_refused_before_the_holder_wrote_its_test_hears_how_that_test_ended(open_as: Open) -> None:
    held = await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(intent("s1", "c1"), ACTION, ttl_seconds=TTL)
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")
    with pytest.raises(ClaimHeld):
        await s2.acquire(intent("s2", "c2"), OTHER_ACTION, ttl_seconds=TTL)
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request("r1", intent("s1", "c1"), delta("--max-num-seqs 256")))
    await toolbox.attach_request(held.key, "r1", ttl_seconds=TTL)

    await _end(open_as, "r1", "judged")

    assert [item.pointer for item in _heard((await s2.poll("s2")).items)] == ["r1"]


NO_ACTION = RecipeDelta().fingerprint()
MORE_SEQUENCES = intent("s2", "c0", "decode batch", "more sequences")


async def _claimed_then_written(open_as: Open, of: Intent, request_id: str) -> ClaimKey:
    """``of``'s branch claims it before writing its change, as the role asks; the toolbox writes the request, whose
    change ``ACTION`` names, and has not attached it to the claim yet."""

    claim = await open_as(Role.SPECIALIST_RUNTIME, of.branch).acquire(of, NO_ACTION, ttl_seconds=TTL)
    await open_as(Role.TOOLBOX).write_request(request(request_id, of, delta("--max-num-seqs 256")))
    return claim.key


async def test_a_change_another_branch_is_testing_on_the_same_state_is_refused_whatever_the_intent(
    open_as: Open,
) -> None:
    key = await _claimed_then_written(open_as, intent("s1", "c0"), "r1")
    await open_as(Role.TOOLBOX).attach_request(key, "r1", ttl_seconds=TTL)
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")

    with pytest.raises(ClaimHeld) as held:
        await s2.acquire(MORE_SEQUENCES, ACTION, ttl_seconds=TTL)
    other = await s2.acquire(MORE_SEQUENCES, OTHER_ACTION, ttl_seconds=TTL)

    answer = held.value.answer
    assert (answer.status, [record.request_id for record in answer.records]) == (DedupStatus.TESTING_ELSEWHERE, ["r1"])
    assert other.holder == "s2"


async def test_a_change_written_after_the_claim_is_refused_at_submit_and_hears_how_the_first_test_ended(
    open_as: Open,
) -> None:
    first = await _claimed_then_written(open_as, intent("s1", "c0"), "r1")
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.attach_request(first, "r1", ttl_seconds=TTL)
    second = await _claimed_then_written(open_as, MORE_SEQUENCES, "r2")
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")

    with pytest.raises(ClaimLost, match="s1 is testing the same change on c0 as r1; its outcome comes to your inbox"):
        await toolbox.attach_request(second, "r2", ttl_seconds=TTL)
    await toolbox.write_withdrawal(Withdrawal("r2", "claim lost"))
    again = await s2.acquire(MORE_SEQUENCES, NO_ACTION, ttl_seconds=TTL)
    await _end(open_as, "r1", "judged")

    (decision,) = _decisions(s2)[-2:-1]
    assert (decision["found"], decision["acquired"], decision["matched"], decision["holders"]) == (
        "testing_elsewhere",
        False,
        ["r1"],
        ["s1"],
    )
    assert again.holder == "s2"
    (heard,) = _heard((await s2.poll("s2")).items)
    assert heard.summary.startswith("s1's test of the same change on c0, which refused your claim, ended: ")
    assert heard.summary.endswith("so there is nothing to replay")


async def test_of_two_branches_submitting_one_change_on_one_state_only_the_later_is_refused(open_as: Open) -> None:
    first = await _claimed_then_written(open_as, intent("s1", "c0"), "r1")
    second = await _claimed_then_written(open_as, MORE_SEQUENCES, "r2")
    toolbox = open_as(Role.TOOLBOX)

    with pytest.raises(ClaimLost):
        await toolbox.attach_request(second, "r2", ttl_seconds=TTL)
    attached = await toolbox.attach_request(first, "r1", ttl_seconds=TTL)

    assert attached.request_id == "r1"


async def test_a_change_measured_on_the_state_is_refused_at_submit_but_not_a_profile_or_the_branchs_own_test(
    open_as: Open,
) -> None:
    first = await _claimed_then_written(open_as, intent("s1", "c0"), "r1")
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.attach_request(first, "r1", ttl_seconds=TTL)
    await toolbox.write_result(result("r1"))
    second = await _claimed_then_written(open_as, MORE_SEQUENCES, "r2")
    own = await _claimed_then_written(open_as, intent("s1", "c0", "scheduler", "chunked prefill"), "r3")
    profiled = request(
        "p2", MORE_SEQUENCES, delta("--max-num-seqs 256"), kind=TestKind.PROFILE, purpose=Purpose.PROFILE
    )
    await toolbox.write_request(profiled)

    with pytest.raises(ClaimLost, match="s1 already measured the same change on c0 as r1"):
        await toolbox.attach_request(second, "r2", ttl_seconds=TTL)
    await toolbox.write_withdrawal(Withdrawal("r2", "claim lost"))
    again = await open_as(Role.SPECIALIST_RUNTIME, "s2").acquire(MORE_SEQUENCES, NO_ACTION, ttl_seconds=TTL)
    kept = await toolbox.attach_request(own, "r3", ttl_seconds=TTL)
    profile = await toolbox.attach_request(again.key, "p2", ttl_seconds=TTL)

    assert (kept.request_id, profile.request_id) == ("r3", "p2")
    refused = [row for row in _decisions(toolbox) if not row["acquired"]]
    assert (refused[-1]["found"], refused[-1]["matched"], refused[-1]["holders"]) == ("tested", ["r1"], [])


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
