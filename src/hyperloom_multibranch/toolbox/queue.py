# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's queue: five priorities, a per-branch K, a global limit and backpressure (design 4.17).

The queue is plain in-memory state with no I/O, so the fake executor and the real one share it. Confirm-retests and
reproduction checks do not count against their branch's K (otherwise K exploration tests would starve the retest that
decides them), but they do count against the global limit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from hyperloom.contracts.measurement import (
    PRIORITY_HIGHEST,
    PRIORITY_LOWEST,
    Purpose,
    TestKind,
    TestRequest,
)
from hyperloom_multibranch.contracts.toolbox import Backpressure, QueueState

_LOWEST_PRIORITY = 4

#: Priority of a request that names none, by purpose (design 4.17). Baseline, health and final are the scheduler's to
#: give through QueuePolicy.priority_override; until it does they take the fallbacks below.
DEFAULT_PRIORITY: Mapping[Purpose, int] = {
    Purpose.CONFIRM: 0,
    Purpose.REPRO: 1,
    Purpose.INTEGRATION: 2,
    Purpose.PROFILE: 3,
    Purpose.ACCURACY: 3,
    Purpose.EXPLORE: _LOWEST_PRIORITY,
    Purpose.BASELINE: 2,
    Purpose.FINAL: 2,
    Purpose.HEALTH: _LOWEST_PRIORITY,
}
#: A queued test's slack, per K, before the branch's next submit is refused with backpressure.
DEFAULT_QUEUE_FACTOR = 4


@dataclass(frozen=True)
class QueuePolicy:
    """What the scheduler sets: tests a branch runs at once, tests in flight overall, a priority per purpose.

    priority_override maps a purpose's value to a priority and applies to requests that name none.
    max_queued_per_branch is the backpressure limit; None means DEFAULT_QUEUE_FACTOR times K.
    """

    per_branch_k: int = 2
    global_concurrency: int = 8
    priority_override: Mapping[str, int] = field(default_factory=dict)
    max_queued_per_branch: int | None = None

    def __post_init__(self) -> None:
        if self.per_branch_k < 1 or self.global_concurrency < 1:
            raise ValueError("per_branch_k and global_concurrency must be at least 1")
        for purpose, priority in self.priority_override.items():
            Purpose(purpose)
            if not PRIORITY_HIGHEST <= priority <= PRIORITY_LOWEST:
                raise ValueError(
                    f"priority {priority} for {purpose!r} is outside {PRIORITY_HIGHEST}..{PRIORITY_LOWEST}"
                )

    @property
    def queue_limit(self) -> int:
        if self.max_queued_per_branch is not None:
            return self.max_queued_per_branch
        return DEFAULT_QUEUE_FACTOR * self.per_branch_k

    def priority_of(self, request: TestRequest) -> int:
        if request.priority is not None:
            return request.priority
        override = self.priority_override.get(request.purpose.value)
        return override if override is not None else DEFAULT_PRIORITY[request.purpose]


def sandbox_priority(priority: int) -> int:
    """The three claw.io/priority tiers: 0 and 1 map to 0, 2 and 3 to 1, 4 to 2."""

    return {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}[priority]


def exempt_from_k(request: TestRequest) -> bool:
    """Confirm-retests and reproduction checks do not count against their branch's K."""

    return request.purpose in (Purpose.CONFIRM, Purpose.REPRO) or request.kind is TestKind.REPRO


class State(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"


@dataclass
class QueuedTest:
    """One accepted test; ``seq`` is the queue's arrival order and breaks ties inside a priority."""

    request: TestRequest
    priority: int
    seq: int
    environment: object = None
    queued_at: float = 0.0
    state: State = State.QUEUED

    @property
    def request_id(self) -> str:
        return self.request.request_id or ""

    @property
    def branch(self) -> str:
        return self.request.branch_id


class WorkQueue:
    """Priority queue with K accounting; ``take`` hands out the best test the limits allow."""

    def __init__(self, policy: QueuePolicy) -> None:
        self.policy = policy
        self._items: dict[str, QueuedTest] = {}
        self._arrivals = 0
        self.cold_start_seconds: float | None = None

    def set_policy(self, policy: QueuePolicy) -> None:
        self.policy = policy

    def get(self, request_id: str) -> QueuedTest | None:
        return self._items.get(request_id)

    def items(self) -> list[QueuedTest]:
        return list(self._items.values())

    def queued_counted(self, branch: str) -> int:
        """Tests of ``branch`` waiting that count against K; the backpressure number."""

        return sum(
            1
            for item in self._items.values()
            if item.state is State.QUEUED and item.branch == branch and not exempt_from_k(item.request)
        )

    def check_backpressure(self, request: TestRequest) -> None:
        if exempt_from_k(request):
            return
        queued, limit = self.queued_counted(request.branch_id), self.policy.queue_limit
        if queued >= limit:
            raise Backpressure(request.branch_id, queued, limit)

    def push(self, item: QueuedTest) -> None:
        self._arrivals += 1
        item.seq = self._arrivals
        item.state = State.QUEUED
        self._items[item.request_id] = item

    def _running_counted(self, branch: str) -> int:
        return sum(
            1
            for item in self._items.values()
            if item.state is State.RUNNING and item.branch == branch and not exempt_from_k(item.request)
        )

    def take(self) -> QueuedTest | None:
        """Mark and return the next test to run, or ``None`` when none may start now."""

        running = sum(1 for item in self._items.values() if item.state is State.RUNNING)
        if running >= self.policy.global_concurrency:
            return None
        ready = [
            item
            for item in self._items.values()
            if item.state is State.QUEUED
            and (exempt_from_k(item.request) or self._running_counted(item.branch) < self.policy.per_branch_k)
        ]
        if not ready:
            return None
        best = min(ready, key=lambda item: (item.priority, item.seq))
        best.state = State.RUNNING
        return best

    def remove(self, request_id: str) -> QueuedTest | None:
        return self._items.pop(request_id, None)

    def reprioritise(self) -> None:
        """Apply a changed policy to queued tests whose request named no priority."""

        for item in self._items.values():
            if item.state is State.QUEUED and item.request.priority is None:
                item.priority = self.policy.priority_of(item.request)

    def state(self) -> QueueState:
        by_priority = dict.fromkeys(range(PRIORITY_HIGHEST, PRIORITY_LOWEST + 1), 0)
        running: dict[str, int] = {}
        queued: dict[str, int] = {}
        for item in self._items.values():
            if item.state is State.QUEUED:
                by_priority[item.priority] += 1
                queued[item.branch] = queued.get(item.branch, 0) + 1
            else:
                running[item.branch] = running.get(item.branch, 0) + 1
        return QueueState(by_priority, running, queued, self.cold_start_seconds)
