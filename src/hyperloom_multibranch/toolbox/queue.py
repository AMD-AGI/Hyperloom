# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The toolbox's queue: five priorities, a per-branch K, a global limit and backpressure (design 4.17).

The queue is plain in-memory state with no I/O, so the fake executor and the real one share it. Confirm-retests,
reproduction checks and session measurements do not count against their branch's K (otherwise K exploration tests
would starve the retest that decides them), but they do count against the global limit.
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
from hyperloom_multibranch.contracts.toolbox import Backpressure, QueueState, RequestRejected

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
DEFAULT_K = 2
DEFAULT_K_KEY = "*"
BRANCH_STOPPED = "branch_stopped"


@dataclass(frozen=True)
class QueuePolicy:
    """What the scheduler sets: tests a branch runs at once, tests in flight overall, a priority per purpose.

    ``per_branch_k`` is the default K, an int; ``branch_k`` overrides it per branch. A mapping given as
    ``per_branch_k`` is the board draft's form: its ``"*"`` entry is the default, the rest are overrides.

    priority_override maps a purpose's value to a priority and applies to requests that name none.
    max_queued_per_branch is the backpressure limit; None means DEFAULT_QUEUE_FACTOR times K.

    ``branch_deadline`` is the Unix time (the toolbox clock's base) each branch's exploration ends, as scheduling gives
    ``BranchTask.deadline``; a test whose native builds and one run of its kind would end past it is refused as
    ``build_after_deadline``. No entry, or 0, means no check.

    ``listed_branches_only`` makes ``branch_k`` the list of branches that may submit at all: a request of a branch
    not in it is refused as ``branch_stopped`` (the scheduler gives every running branch a K and takes it away when it
    stops the branch). The default False keeps ``per_branch_k`` as the K of every branch not listed.
    """

    per_branch_k: int | Mapping[str, int] = 2
    global_concurrency: int = 8
    priority_override: Mapping[str, int] = field(default_factory=dict)
    max_queued_per_branch: int | None = None
    branch_k: Mapping[str, int] = field(default_factory=dict)
    listed_branches_only: bool = False
    branch_deadline: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.per_branch_k, Mapping):
            # The board draft: a K per branch, the default under ``DEFAULT_K_KEY``.
            overrides = {k: v for k, v in self.per_branch_k.items() if k != DEFAULT_K_KEY}
            default = self.per_branch_k.get(DEFAULT_K_KEY, DEFAULT_K)
            object.__setattr__(self, "per_branch_k", default)
            object.__setattr__(self, "branch_k", {**overrides, **self.branch_k})
        if self.per_branch_k < 1 or self.global_concurrency < 1 or any(k < 1 for k in self.branch_k.values()):
            raise ValueError("every K and global_concurrency must be at least 1")
        for purpose, priority in self.priority_override.items():
            Purpose(purpose)
            if not PRIORITY_HIGHEST <= priority <= PRIORITY_LOWEST:
                raise ValueError(
                    f"priority {priority} for {purpose!r} is outside {PRIORITY_HIGHEST}..{PRIORITY_LOWEST}"
                )

    def k_for(self, branch: str) -> int:
        """The K of ``branch``: its override, else the default."""

        return self.branch_k.get(branch, self.per_branch_k)  # type: ignore[arg-type]

    def queue_limit_for(self, branch: str) -> int:
        if self.max_queued_per_branch is not None:
            return self.max_queued_per_branch
        return DEFAULT_QUEUE_FACTOR * self.k_for(branch)

    def priority_of(self, request: TestRequest) -> int:
        if request.priority is not None:
            return request.priority
        override = self.priority_override.get(request.purpose.value)
        return override if override is not None else DEFAULT_PRIORITY[request.purpose]


def sandbox_priority(priority: int) -> int:
    """The three agent-sandbox.io/priority tiers: 0 and 1 map to 0, 2 and 3 to 1, 4 to 2."""

    return {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}[priority]


#: Purposes that never take a slot of their branch's K: confirm-retests and reproduction checks (the retest that
#: decides a result must not wait behind K exploration tests) and the session measurements (baseline, health check and
#: final validation), which run while no specialist branch does.
EXEMPT_PURPOSES = frozenset({Purpose.CONFIRM, Purpose.REPRO, Purpose.BASELINE, Purpose.HEALTH, Purpose.FINAL})


def exempt_from_k(request: TestRequest) -> bool:
    """Whether the request takes no slot of its branch's K (it still counts against the global limit)."""

    return request.purpose in EXEMPT_PURPOSES or request.kind is TestKind.REPRO


class State(str, Enum):
    """``PENDING_BUILD``: accepted, waiting for its native builds; it holds no card but takes a slot of K's backlog."""

    PENDING_BUILD = "pending_build"
    QUEUED = "queued"
    RUNNING = "running"


#: The states of a test that has not started: both count against its branch's backlog.
WAITING_STATES = frozenset({State.PENDING_BUILD, State.QUEUED})


@dataclass
class QueuedTest:
    """One accepted test; ``seq`` is the queue's arrival order and breaks ties inside a priority.

    ``started_at`` is when the toolbox took it from the queue to run, ``None`` while it waits.
    """

    request: TestRequest
    priority: int
    seq: int
    environment: object = None
    queued_at: float = 0.0
    state: State = State.QUEUED
    started_at: float | None = None

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
        """Tests of ``branch`` waiting, for their builds or for a card, that count against K; the backpressure number."""

        return sum(
            1
            for item in self._items.values()
            if item.state in WAITING_STATES and item.branch == branch and not exempt_from_k(item.request)
        )

    def check_branch_listed(self, request: TestRequest) -> None:
        if self.policy.listed_branches_only and request.branch_id not in self.policy.branch_k:
            raise RequestRejected(f"branch {request.branch_id!r} has no quota", code=BRANCH_STOPPED)

    def check_backpressure(self, request: TestRequest) -> None:
        if exempt_from_k(request):
            return
        queued, limit = self.queued_counted(request.branch_id), self.policy.queue_limit_for(request.branch_id)
        if queued >= limit:
            raise Backpressure(request.branch_id, queued, limit)

    def push(self, item: QueuedTest) -> None:
        """Add an accepted test in the state it was made in: ``QUEUED``, or ``PENDING_BUILD`` until :meth:`release`."""

        if item.state not in WAITING_STATES:
            raise ValueError(f"a test enters the queue waiting, not {item.state.value}")
        self._arrivals += 1
        item.seq = self._arrivals
        self._items[item.request_id] = item

    def release(self, request_id: str, now: float) -> bool:
        """Move a test whose builds are all ready to ``QUEUED``; its queue time starts ``now``. False when it is not
        waiting for builds (taken back meanwhile)."""

        item = self._items.get(request_id)
        if item is None or item.state is not State.PENDING_BUILD:
            return False
        item.state, item.queued_at = State.QUEUED, now
        return True

    def _running_counted(self, branch: str) -> int:
        return sum(
            1
            for item in self._items.values()
            if item.state is State.RUNNING and item.branch == branch and not exempt_from_k(item.request)
        )

    def take(self, parallel_limit: int | None = None) -> QueuedTest | None:
        """Mark and return the next test to run, or ``None`` when none may start now.

        ``parallel_limit`` is what the toolbox's cards run at once; it lowers the policy's global concurrency.
        """

        running = sum(1 for item in self._items.values() if item.state is State.RUNNING)
        cap = self.policy.global_concurrency
        if running >= (cap if parallel_limit is None else min(cap, parallel_limit)):
            return None
        ready = [
            item
            for item in self._items.values()
            if item.state is State.QUEUED
            and (exempt_from_k(item.request) or self._running_counted(item.branch) < self.policy.k_for(item.branch))
        ]
        if not ready:
            return None
        best = min(ready, key=lambda item: (item.priority, item.seq))
        best.state = State.RUNNING
        return best

    def queued_ahead(self, request_id: str) -> int:
        """How many waiting tests ``take`` prefers to ``request_id`` by priority and arrival, whatever their K."""

        item = self._items[request_id]
        return sum(
            1
            for other in self._items.values()
            if other.state is State.QUEUED and (other.priority, other.seq) < (item.priority, item.seq)
        )

    def remove(self, request_id: str) -> QueuedTest | None:
        return self._items.pop(request_id, None)

    def reprioritise(self) -> None:
        """Apply a changed policy to queued tests whose request named no priority."""

        for item in self._items.values():
            if item.state in WAITING_STATES and item.request.priority is None:
                item.priority = self.policy.priority_of(item.request)

    def state(self) -> QueueState:
        by_priority = dict.fromkeys(range(PRIORITY_HIGHEST, PRIORITY_LOWEST + 1), 0)
        counts: dict[State, dict[str, int]] = {state: {} for state in State}
        for item in self._items.values():
            if item.state is State.QUEUED:
                by_priority[item.priority] += 1
            per_branch = counts[item.state]
            per_branch[item.branch] = per_branch.get(item.branch, 0) + 1
        return QueueState(
            by_priority,
            counts[State.RUNNING],
            counts[State.QUEUED],
            self.cold_start_seconds,
            pending_build_by_branch=counts[State.PENDING_BUILD],
        )
