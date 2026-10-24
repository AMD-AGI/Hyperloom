# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom_multibranch.contracts.toolbox import Backpressure
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.toolbox.queue import QueuedTest, QueuePolicy, WorkQueue, exempt_from_k, sandbox_priority
from hyperloom_multibranch.toolbox.tests.helpers import req


def _item(name: str, branch: str = "b1", **fields: object) -> QueuedTest:
    request = req(branch, **fields).accepted(name)  # type: ignore[arg-type]
    return QueuedTest(request, QueuePolicy().priority_of(request), 0)


def _queue(**policy: object) -> WorkQueue:
    return WorkQueue(QueuePolicy(**policy))  # type: ignore[arg-type]


def test_default_priorities_follow_the_purpose() -> None:
    policy = QueuePolicy()
    confirm = req(purpose=Purpose.CONFIRM, retest_of="x")
    repro = req(purpose=Purpose.REPRO, kind=TestKind.REPRO, attachments=_attachment())
    got = [
        policy.priority_of(r)
        for r in (confirm, repro, req(purpose=Purpose.INTEGRATION), req(purpose=Purpose.PROFILE), req())
    ]
    assert got == [0, 1, 2, 3, 4]


def test_an_explicit_priority_wins_and_the_scheduler_override_applies_to_the_rest() -> None:
    policy = QueuePolicy(priority_override={"explore": 1, "baseline": 0})
    assert policy.priority_of(req(priority=3)) == 3
    assert policy.priority_of(req()) == 1
    assert policy.priority_of(req(purpose=Purpose.BASELINE)) == 0


def test_the_queue_hands_out_the_highest_priority_first_then_arrival_order() -> None:
    queue = _queue(per_branch_k=9, global_concurrency=9)
    for name, priority in [("a", 4), ("b", 2), ("c", 4), ("d", 0), ("e", 2)]:
        item = _item(name, priority=priority)
        queue.push(item)
    order = []
    while (item := queue.take()) is not None:
        order.append(item.request_id)
    assert order == ["d", "b", "e", "a", "c"]


def test_the_cards_of_the_toolbox_lower_the_global_concurrency_but_never_raise_it() -> None:
    queue = _queue(per_branch_k=9, global_concurrency=3)
    for name in "abcd":
        queue.push(_item(name))
    assert [queue.take(parallel_limit=2).request_id, queue.take(parallel_limit=2).request_id] == ["a", "b"]  # type: ignore[union-attr]
    assert queue.take(parallel_limit=2) is None, "two run and the cards run two"
    assert queue.take(parallel_limit=9).request_id == "c"  # type: ignore[union-attr]
    assert queue.take(parallel_limit=9) is None, "three run and the policy allows three"


def test_a_branch_never_runs_more_than_k() -> None:
    queue = _queue(per_branch_k=2, global_concurrency=9)
    for name in "abcd":
        queue.push(_item(name))
    assert [queue.take().request_id, queue.take().request_id] == ["a", "b"]  # type: ignore[union-attr]
    assert queue.take() is None
    queue.push(_item("other", "b2"))
    assert queue.take().request_id == "other"  # type: ignore[union-attr]


def test_a_confirm_retest_is_not_blocked_by_a_full_k() -> None:
    queue = _queue(per_branch_k=1, global_concurrency=9)
    queue.push(_item("explore"))
    assert queue.take().request_id == "explore"  # type: ignore[union-attr]
    queue.push(_item("confirm", purpose=Purpose.CONFIRM, retest_of="explore"))
    queue.push(_item("another-explore"))
    assert queue.take().request_id == "confirm"  # type: ignore[union-attr]
    assert queue.take() is None, "the exempt test must not have freed or used a slot of K"


def test_a_running_confirm_does_not_use_a_slot_of_k() -> None:
    queue = _queue(per_branch_k=1, global_concurrency=9)
    queue.push(_item("confirm", purpose=Purpose.CONFIRM, retest_of="x"))
    assert queue.take().request_id == "confirm"  # type: ignore[union-attr]
    queue.push(_item("explore"))
    assert queue.take().request_id == "explore"  # type: ignore[union-attr]


def test_the_global_limit_applies_to_exempt_tests_too() -> None:
    queue = _queue(per_branch_k=5, global_concurrency=1)
    queue.push(_item("confirm", purpose=Purpose.CONFIRM, retest_of="x"))
    queue.push(_item("explore"))
    assert queue.take().request_id == "confirm"  # type: ignore[union-attr]
    assert queue.take() is None


def test_repro_by_kind_or_purpose_is_exempt_and_explore_is_not() -> None:
    assert exempt_from_k(req(purpose=Purpose.REPRO, kind=TestKind.REPRO, attachments=_attachment()))
    assert not exempt_from_k(req())


def test_session_measurements_are_exempt_and_integration_and_profile_tests_are_not() -> None:
    for purpose in (Purpose.BASELINE, Purpose.HEALTH, Purpose.FINAL):
        assert exempt_from_k(req(purpose=purpose)), purpose
    for purpose in (Purpose.INTEGRATION, Purpose.PROFILE, Purpose.ACCURACY, Purpose.EXPLORE):
        assert not exempt_from_k(req(purpose=purpose)), purpose


def test_several_baseline_tests_of_one_branch_run_at_once_under_a_k_of_one() -> None:
    queue = _queue(per_branch_k=1, global_concurrency=9)
    for name in "abc":
        queue.push(_item(name, purpose=Purpose.BASELINE))
    assert sorted(queue.take().request_id for _ in range(3)) == ["a", "b", "c"]  # type: ignore[union-attr]


def _attachment():
    from hyperloom.contracts.measurement import AttachmentRef

    return (AttachmentRef("case", "file", "0" * 64),)


def test_backpressure_counts_queued_tests_of_the_branch_and_spares_exempt_ones() -> None:
    queue = _queue(per_branch_k=1, max_queued_per_branch=2)
    for name in "ab":
        queue.check_backpressure(req())
        queue.push(_item(name))
    with pytest.raises(Backpressure) as refused:
        queue.check_backpressure(req())
    assert (refused.value.branch_id, refused.value.queued, refused.value.limit) == ("b1", 2, 2)
    queue.check_backpressure(req("b2"))
    queue.check_backpressure(req(purpose=Purpose.CONFIRM, retest_of="a"))


def test_the_queue_state_counts_by_priority_branch_and_running() -> None:
    queue = _queue(per_branch_k=1, global_concurrency=9)
    queue.push(_item("a", priority=0))
    queue.push(_item("b", priority=4))
    queue.take()
    state = queue.state()
    assert dict(state.queued_by_priority) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}
    assert dict(state.running_by_branch) == {"b1": 1} and dict(state.queued_by_branch) == {"b1": 1}


def test_a_new_policy_reprioritises_only_queued_tests_without_an_explicit_priority() -> None:
    queue = _queue()
    queue.push(_item("default"))
    queue.push(_item("explicit", priority=4))
    queue.set_policy(QueuePolicy(priority_override={"explore": 0}))
    queue.reprioritise()
    assert (queue.get("default").priority, queue.get("explicit").priority) == (0, 4)  # type: ignore[union-attr]


def test_sandbox_priority_has_three_tiers() -> None:
    assert [sandbox_priority(p) for p in range(5)] == [0, 0, 1, 1, 2]


def test_a_policy_outside_the_contract_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        QueuePolicy(per_branch_k=0)
    with pytest.raises(ValueError, match="outside"):
        QueuePolicy(priority_override={"explore": 7})


def test_k_is_a_default_with_per_branch_overrides() -> None:
    policy = QueuePolicy(per_branch_k=2, branch_k={"slow": 1, "wide": 5})
    assert [policy.k_for(b) for b in ("other", "slow", "wide")] == [2, 1, 5]
    draft = QueuePolicy(per_branch_k={"*": 3, "slow": 1})
    assert (draft.per_branch_k, draft.k_for("slow"), draft.k_for("x")) == (3, 1, 3)
    assert QueuePolicy(per_branch_k={"a": 4}).k_for("b") == 2, "a mapping without a default keeps the default of 2"
    with pytest.raises(ValueError, match="at least 1"):
        QueuePolicy(branch_k={"a": 0})


def test_the_queue_applies_each_branchs_own_k_and_backpressure_limit() -> None:
    queue = _queue(per_branch_k=1, branch_k={"wide": 2}, global_concurrency=9)
    for name in ("a", "b", "c"):
        queue.push(_item(name, "wide"))
        queue.push(_item(f"n{name}", "narrow"))
    while queue.take() is not None:
        pass
    running = {"wide": 0, "narrow": 0}
    for item in queue.items():
        running[item.branch] += item.state.value == "running"
    assert running == {"wide": 2, "narrow": 1}
    assert queue.policy.queue_limit_for("wide") == 8 and queue.policy.queue_limit_for("narrow") == 4


def test_a_listed_only_policy_refuses_a_branch_that_has_no_quota() -> None:
    queue = _queue(branch_k={"b1": 2}, listed_branches_only=True)
    queue.check_branch_listed(req("b1"))
    with pytest.raises(RequestRejected) as refused:
        queue.check_branch_listed(req("b2"))
    assert refused.value.code == "branch_stopped"


def test_without_the_listed_only_flag_an_unlisted_branch_runs_at_the_default_k() -> None:
    _queue(branch_k={"b1": 2}).check_branch_listed(req("b2"))
