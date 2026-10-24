# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brief items: the inbox asks a specialist to read its brief again when its branch stalls, or when another branch
learns a pitfall in a direction it holds a claim on."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, ClaimHeld, Role, Writer
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import InboxItem, Learning, LineageEvent
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, InboxKind, LearningKind, LineageKind
from hyperloom_multibranch.notebook import STALL_TESTS, BriefTriggers, LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import TTL, FakeClock, delta, intent, judgement, request, result

Open = Callable[..., LocalNotebook]
FP8_KV = intent("s1", "c0", "kv cache", "fp8 kv cache", IntentLayer.CONFIG)
FP8_PITFALL = "fp8 KV cache fails the accuracy gate on this model"


@pytest.fixture
async def open_as(open_as: Open) -> Open:
    """The notebook with the root ``c0`` and the branches ``s1`` and ``s2`` open on it."""

    scheduler = open_as(Role.SCHEDULER)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    for branch in ("s1", "s2"):
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit="c0"))
    return open_as


def _briefed(root: Path, clock: FakeClock, branch: str, triggers: BriefTriggers) -> LocalNotebook:
    return LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, branch, branch), clock=clock, briefs=triggers)


async def _judged(open_as: Open, request_id: str, conclusion: Conclusion, parent: str = "c0", commit: str = "") -> None:
    toolbox = open_as(Role.TOOLBOX)
    await toolbox.write_request(request(request_id, intent("s1", parent, target=request_id), delta()))
    await toolbox.write_result(result(request_id))
    await open_as(Role.GATE).write_judgement(request_id, judgement(0.1, (request_id,)), conclusion, commit or None)


def _briefs(items: tuple[InboxItem, ...]) -> list[tuple[str, str]]:
    return [(item.pointer, item.summary.split(":")[0]) for item in items if item.kind is InboxKind.BRIEF]


async def test_a_stall_briefs_the_branch_once_when_its_last_commit_is_that_many_judged_tests_back(
    open_as: Open, root: Path, clock: FakeClock
) -> None:
    s1 = _briefed(root, clock, "s1", BriefTriggers(stall_tests=2))
    await _judged(open_as, "r1", Conclusion.RECORDED_ONLY)
    await _judged(open_as, "r2", Conclusion.UNVERIFIED)
    first = await s1.poll("s1")
    await _judged(open_as, "r3", Conclusion.RECORDED_ONLY)
    second = await s1.poll("s1", first.cursor)
    await _judged(open_as, "r4", Conclusion.RECORDED_ONLY)
    third = await s1.poll("s1", second.cursor)
    await _judged(open_as, "r5", Conclusion.COMMITTED, commit="c1")
    for request_id in ("r6", "r7"):
        await _judged(open_as, request_id, Conclusion.RECORDED_ONLY, parent="c1")
    fourth = await s1.poll("s1", third.cursor)
    (row,) = [row for row in await s1.progress() if row.branch == "s1"]

    assert (_briefs(first.items), _briefs(second.items), _briefs(third.items)) == ([], [("c0", "stall")], [])
    assert _briefs(fourth.items) == [("c1", "stall")]
    assert "notebook_brief from c1" in next(item.summary for item in fourth.items if item.kind is InboxKind.BRIEF)
    assert row.judged_since_commit == 2
    assert _briefs((await s1.poll("s1")).items) == [("c1", "stall"), ("c1", "stall")]


async def test_another_branchs_pitfall_briefs_a_branch_holding_a_claim_in_that_direction(
    open_as: Open, root: Path, clock: FakeClock
) -> None:
    s1 = _briefed(root, clock, "s1", BriefTriggers())
    await s1.acquire(FP8_KV, delta("--kv-cache-dtype fp8").fingerprint(), ttl_seconds=TTL)
    s2 = open_as(Role.SPECIALIST_RUNTIME, "s2")
    await s2.append_learning(Learning("s2", LearningKind.PITFALL, FP8_PITFALL, "kv cache", "fp8 kv cache"))
    await s2.append_learning(Learning("s2", LearningKind.PITFALL, "split kv regresses decode here", "attention"))
    await s2.append_learning(Learning("s2", LearningKind.INSIGHT, "the kv cache is half the memory", "kv cache"))
    await s1.append_learning(
        Learning("s1", LearningKind.PITFALL, "fp8 kv cache needs scales on this model", "kv cache")
    )

    page = await s1.poll("s1")
    unheld = await _briefed(root, clock, "s2", BriefTriggers()).poll("s2")
    quiet = await _briefed(root, clock, "s1", BriefTriggers(pitfalls=False)).poll("s1")

    (brief,) = [item for item in page.items if item.kind is InboxKind.BRIEF]
    assert brief.pointer == "c0"
    assert brief.summary.startswith(f"pitfall: s2 learnt, in a direction this branch holds: {FP8_PITFALL};")
    assert _briefs(unheld.items) == [] and _briefs(quiet.items) == []


async def test_claims_refused_while_another_branch_tests_the_intent_do_not_count_toward_a_stall(
    open_as: Open, root: Path, clock: FakeClock
) -> None:
    s2 = _briefed(root, clock, "s2", BriefTriggers(stall_tests=1))
    await open_as(Role.SPECIALIST_RUNTIME, "s1").acquire(
        FP8_KV, delta("--kv-cache-dtype fp8").fingerprint(), ttl_seconds=TTL
    )
    for head in ("c2", "c3", "c4"):
        with pytest.raises(ClaimHeld):
            await s2.acquire(
                replace(FP8_KV, branch="s2", parent_commit=head),
                delta("--kv-cache-dtype fp8_e5m2").fingerprint(),
                ttl_seconds=TTL,
            )

    page = await s2.poll("s2")
    (row,) = [row for row in await s2.progress() if row.branch == "s2"]

    assert _briefs(page.items) == [] and row.judged_since_commit == 0


async def test_without_triggers_the_inbox_has_no_brief_items(open_as: Open) -> None:
    for request_id in ("r1", "r2", "r3", "r4"):
        await _judged(open_as, request_id, Conclusion.RECORDED_ONLY)

    page = await open_as(Role.SPECIALIST_RUNTIME, "s1").poll("s1")

    assert [item.kind for item in page.items].count(InboxKind.JUDGEMENT) == STALL_TESTS
    assert _briefs(page.items) == []


def test_a_stall_takes_at_least_one_judged_test() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BriefTriggers(stall_tests=0)
