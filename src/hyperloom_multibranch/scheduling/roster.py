# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The branch roster: which branches the session has, where each started, its K, and why it stopped.

The roster says what is running now; the history of the branch tree is the notebook's lineage, which the roster does
not copy. The branch cap counts specialist branches that are admitted or running, never the integration branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

from hyperloom.common.io import atomic_write_json
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.toolbox.queue import QueuePolicy

#: Names how :meth:`Roster.quota_policy` hands out K; recorded in the manifest fingerprint.
QUOTA_POLICY_VERSION = "mb1-quota-3"


class BranchKind(str, Enum):
    SPECIALIST = "specialist"
    INTEGRATION = "integration"


class BranchOrigin(str, Enum):
    """Where a branch's start comes from: the root commit (integration), the opening batch, a later batch that refills
    freed slots, or a rule-gate request: a fork for a correctness fix or a repair of a blocked gain."""

    ROOT = "root"
    FIRST_BATCH = "first_batch"
    REFILL = "refill"
    FORK = "fork"
    REPAIR = "repair"

    @property
    def cut_by_scheduler(self) -> bool:
        """Whether the scheduler cuts the branch in the repository; the rule gate cuts a fork and lands its fix."""

        return self is not BranchOrigin.FORK


class BranchStatus(str, Enum):
    """``ADMITTED`` is a fork or repair the scheduler accepted whose specialist has not started yet."""

    ADMITTED = "admitted"
    RUNNING = "running"
    STOPPED = "stopped"


class StopCause(str, Enum):
    """Why a branch stopped: the judge stopped it, the session ended, its specialist did not start, or it was a repair
    branch that used up its time."""

    JUDGE = "judge"
    SESSION_END = "session_end"
    START_FAILED = "start_failed"
    REPAIR_TIME_LIMIT = "repair_time_limit"


@dataclass(frozen=True)
class BranchStop:
    cause: StopCause
    reason: str
    at: float

    @property
    def summary(self) -> str:
        """``cause: reason``, as the ``STOP`` lineage event and the judge's view state it."""

        return f"{self.cause.value}: {self.reason}"


@dataclass(frozen=True)
class BranchEntry:
    """One branch of the roster; ``start_commit`` is empty while an admitted fork or repair waits to start."""

    branch: str
    kind: BranchKind
    origin: BranchOrigin
    start_commit: str
    k: int
    opened_at: float
    from_branch: str = ""
    stop: BranchStop | None = None

    @property
    def status(self) -> BranchStatus:
        if self.stop is not None:
            return BranchStatus.STOPPED
        return BranchStatus.RUNNING if self.start_commit else BranchStatus.ADMITTED

    @property
    def counts_toward_cap(self) -> bool:
        return self.kind is BranchKind.SPECIALIST and self.stop is None


@dataclass(frozen=True)
class RosterCaps:
    """The hard limits the judge's decisions are cut to."""

    max_branches: int
    max_k_per_branch: int
    max_parallel_tests: int

    def __post_init__(self) -> None:
        for name in ("max_branches", "max_k_per_branch", "max_parallel_tests"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")

    def clamp_k(self, k: int) -> int:
        """``k`` cut to ``[1, max_k_per_branch]``; a branch is brought to no tests by stopping it."""

        return min(max(k, 1), self.max_k_per_branch)


class RosterFull(RuntimeError):
    """No branch may be opened: every slot is taken, or the session is stopping."""


class BranchStopped(RuntimeError):
    """The branch is stopped; it is never started or given a new K again."""


class Roster:
    """The live roster, written to ``snapshot_file`` after every change so the dashboard and a post-mortem see it.

    Branch ids are ``b01``, ``b02``, ... in the order they are handed out, and an id is never handed out twice.
    """

    def __init__(self, caps: RosterCaps, snapshot_file: Path, clock: Clock) -> None:
        self.caps = caps
        self._snapshot_file = snapshot_file
        self._clock = clock
        self._entries: dict[str, BranchEntry] = {}
        self._issued = 0
        self._closed = False
        self._persist()

    @property
    def closed(self) -> bool:
        return self._closed

    def entries(self) -> tuple[BranchEntry, ...]:
        return tuple(self._entries.values())

    def entry(self, branch: str) -> BranchEntry:
        return self._entries[branch]

    def free_slots(self) -> int:
        if self._closed:
            return 0
        taken = sum(1 for entry in self._entries.values() if entry.counts_toward_cap)
        return max(self.caps.max_branches - taken, 0)

    def open_refusal(self) -> str:
        """Why no branch may be opened now, or ``""`` when one may; :meth:`open` raises :class:`RosterFull` with it."""

        if self._closed:
            return "the session is stopping"
        if self.free_slots() == 0:
            return f"all {self.caps.max_branches} branch slots are taken"
        return ""

    def next_branch_id(self) -> str:
        self._issued += 1
        self._persist()
        return f"b{self._issued:02d}"

    def close(self) -> None:
        """Nothing is opened or admitted after this; the branches already on the roster are stopped by their owner."""

        self._closed = True
        self._persist()

    def add_integration(self, root_commit: str, k: int) -> BranchEntry:
        entry = BranchEntry(
            INTEGRATION_BRANCH,
            BranchKind.INTEGRATION,
            BranchOrigin.ROOT,
            root_commit,
            self.caps.clamp_k(k),
            self._clock.now(),
        )
        return self._add(entry)

    def open(
        self, branch: str, *, origin: BranchOrigin, start_commit: str, k: int, from_branch: str = ""
    ) -> BranchEntry:
        """A specialist branch; an admitted fork passes an empty ``start_commit`` and is started by :meth:`started`."""

        refusal = self.open_refusal()
        if refusal:
            raise RosterFull(f"cannot open {branch}: {refusal}")
        entry = BranchEntry(
            branch,
            BranchKind.SPECIALIST,
            origin,
            start_commit,
            self.caps.clamp_k(k),
            self._clock.now(),
            from_branch=from_branch,
        )
        return self._add(entry)

    def started(self, branch: str, start_commit: str) -> BranchEntry:
        if not start_commit:
            raise ValueError(f"{branch} cannot start without a commit")
        return self._update(self._live(branch), start_commit=start_commit)

    def set_k(self, branch: str, k: int) -> BranchEntry:
        return self._update(self._live(branch), k=self.caps.clamp_k(k))

    def stop(self, branch: str, stop: BranchStop) -> BranchEntry:
        """Stops ``branch``; stopping it again keeps the first stop."""

        entry = self._entries[branch]
        if entry.stop is not None:
            return entry
        return self._update(entry, stop=stop)

    def quota_policy(self) -> QueuePolicy:
        """The toolbox's queue policy: every running branch's K under the global cap on running tests.

        Only the running branches may submit: the toolbox refuses a test of a branch that is stopped or not started.
        The session measurements of PRELUDE and FINAL take no slot of the integration branch's K.
        """

        ks = {entry.branch: entry.k for entry in self._entries.values() if entry.status is BranchStatus.RUNNING}
        return QueuePolicy(global_concurrency=self.caps.max_parallel_tests, branch_k=ks, listed_branches_only=True)

    def _live(self, branch: str) -> BranchEntry:
        entry = self._entries[branch]
        if entry.stop is not None:
            raise BranchStopped(f"{branch} is stopped ({entry.stop.summary})")
        return entry

    def _add(self, entry: BranchEntry) -> BranchEntry:
        if entry.branch in self._entries:
            raise ValueError(f"{entry.branch} is already on the roster")
        self._entries[entry.branch] = entry
        self._persist()
        return entry

    def _update(self, entry: BranchEntry, **changes: object) -> BranchEntry:
        updated = replace(entry, **changes)
        self._entries[entry.branch] = updated
        self._persist()
        return updated

    def _persist(self) -> None:
        atomic_write_json(
            self._snapshot_file,
            {
                "caps": asdict(self.caps),
                "closed": self._closed,
                "issued_branch_ids": self._issued,
                "entries": [_entry_record(entry) for entry in self._entries.values()],
            },
        )


def _entry_record(entry: BranchEntry) -> dict[str, object]:
    stop = entry.stop
    return {
        "branch": entry.branch,
        "kind": entry.kind.value,
        "origin": entry.origin.value,
        "status": entry.status.value,
        "start_commit": entry.start_commit,
        "k": entry.k,
        "opened_at": entry.opened_at,
        "from_branch": entry.from_branch,
        "stop": None if stop is None else {"cause": stop.cause.value, "reason": stop.reason, "at": stop.at},
    }


__all__ = [
    "QUOTA_POLICY_VERSION",
    "BranchEntry",
    "BranchKind",
    "BranchOrigin",
    "BranchStatus",
    "BranchStop",
    "BranchStopped",
    "Roster",
    "RosterCaps",
    "RosterFull",
    "StopCause",
]
