# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The search tree a tree-strategy campaign grows: nodes, directions, attempts and the round in progress."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, replace
from enum import Enum

from kernelforge.evaluation.scoring import keep_score


class SlotType(str, Enum):
    """What one expansion does with its start node."""

    DEEPEN = "DEEPEN"
    BRANCH = "BRANCH"
    PORT = "PORT"


class DirectionStatus(str, Enum):
    """Whether a direction may still be deepened."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class AttemptOutcome(str, Enum):
    """What one slot's candidate came to."""

    NODE = "NODE"
    INCORRECT = "INCORRECT"
    NO_CHANGES = "NO_CHANGES"
    ERROR = "ERROR"


class NodeSource(str, Enum):
    """How a node's code came to exist."""

    BASELINE = "baseline"
    WARM_START = "warm_start"
    ITERATION = "iteration"
    COMPOSE = "compose"
    PORT = "port"


class SlotStatus(str, Enum):
    """How far one slot of the round in progress has got."""

    PLANNED = "PLANNED"
    DISPATCHED = "DISPATCHED"
    MEASURED = "MEASURED"


@dataclass(frozen=True)
class NodeMeasurement:
    """What one measurement of a node's code adds to its record: a score and each case's time for every bench run."""

    samples: tuple[float, ...]
    case_runs: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class TreeNode:
    """One validated and measured kernel version, held as a git commit."""

    commit: str
    parent: str
    direction: str
    # None for the root and for a node the framework composed rather than a slot produced.
    slot_type: SlotType | None
    iteration: int
    slot: int
    samples: tuple[float, ...]
    wall_ms: float | None
    case_times: dict[str, float]
    # Each case's run-to-run spread over the node's own measurements, the noise its per-case wins are read against.
    case_spreads: dict[str, float]
    snr_db: float | None
    plan: str
    source: NodeSource
    provenance: tuple[str, ...] = ()
    # How the round planner had the slot's lane differ from the start's other lanes, and the cases it aimed at; empty
    # for a node no partition lane produced.
    approach: str = ""
    focus_cases: tuple[str, ...] = ()
    # The Agent profile whose session wrote the node; empty for the root, a mechanical merge, and a session without one.
    agent_profile_id: str = ""
    # Each case's time in every bench run of every measurement of the node: ``case_times`` is its median and
    # ``case_spreads`` its standard deviation. Empty for a node a tree recorded without it.
    case_samples: dict[str, tuple[float, ...]] = field(default_factory=dict)
    # The directory recording the change that made the node: its slot's or its merge's; empty for the root.
    records: str = ""

    @property
    def score(self) -> float:
        """The mean of every pristine-relative score measured for this node."""
        return keep_score(list(self.samples)) or 0.0

    @property
    def is_root(self) -> bool:
        return not self.parent

    def with_measurement(self, measurement: NodeMeasurement) -> "TreeNode":
        """The node with one more measurement folded into its samples and its per-case statistics."""
        runs = {case: list(times) for case, times in self.case_samples.items()}
        for run in measurement.case_runs:
            for case, ms in run.items():
                runs.setdefault(case, []).append(ms)
        case_samples = {case: tuple(times) for case, times in runs.items()}
        return replace(
            self,
            samples=(*self.samples, *measurement.samples),
            case_samples=case_samples,
            case_times={case: statistics.median(times) for case, times in case_samples.items()},
            case_spreads={case: statistics.stdev(times) for case, times in case_samples.items() if len(times) >= 2},
        )


@dataclass(frozen=True)
class Direction:
    """One named optimization route, opened by a BRANCH slot at its origin node."""

    direction_id: str
    name: str
    hypothesis: str
    origin: str
    branch: str
    expected_ceiling: str
    failure_criterion: str
    created_iteration: int
    status: DirectionStatus = DirectionStatus.OPEN
    status_reason: str = ""
    hint_rebuttals: tuple[str, ...] = ()


@dataclass(frozen=True)
class Slot:
    """One expansion the selector asked for in the round in progress."""

    index: int
    slot_type: SlotType
    start: str
    reason: str
    direction: str = ""
    source: str = ""
    hint: str = ""
    gains: str = ""
    status: SlotStatus = SlotStatus.PLANNED
    # The Agent profile the slot is planned and implemented on; empty when the session has none.
    agent_profile_id: str = ""


@dataclass(frozen=True)
class Attempt:
    """One finished slot: where it started, what it was for, and what it came to."""

    iteration: int
    slot: int
    slot_type: SlotType
    start: str
    direction: str
    outcome: AttemptOutcome
    score: float | None
    node: str
    archive_path: str
    reason: str
    # Why the slot came to nothing without being measured: never planned, lost, or a copy of an existing node.
    detail: str = ""
    # The Agent profile the slot ran on.
    agent_profile_id: str = ""
    # The plan the slot's Implementer was given; empty for a slot that was never planned.
    plan_path: str = ""


@dataclass(frozen=True)
class MergeFailure:
    """A merge of one round's gains that did not land: the node whose change was to be carried, onto which, and why."""

    source: str
    base: str
    reason: str


@dataclass(frozen=True)
class Line:
    """The history one start node is planned from: the path from the root to it, and the attempts of that path."""

    nodes: tuple[TreeNode, ...]
    attempts: tuple[Attempt, ...]


@dataclass(frozen=True)
class Round:
    """The slots and requests of the iteration in progress."""

    iteration: int
    slots: tuple[Slot, ...]
    remeasure: tuple[str, ...] = ()
    refresh_analysis: tuple[str, ...] = ()


@dataclass
class SearchTree:
    """Every node, direction and attempt of one tree-strategy campaign."""

    root_branch: str
    nodes: list[TreeNode]
    published_best: str
    # The campaign-branch commit that publishes ``published_best``: the same tree, committed on the published line.
    published_head: str
    directions: list[Direction] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    round: Round | None = None
    next_direction_seq: int = 1

    @classmethod
    def create(cls, root: TreeNode, *, root_branch: str) -> "SearchTree":
        """A tree holding only the node the campaign starts from, which is also what it publishes."""
        if not root.is_root:
            raise ValueError(f"the root node {root.commit} names a parent")
        return cls(root_branch=root_branch, nodes=[root], published_best=root.commit, published_head=root.commit)

    @property
    def root(self) -> TreeNode:
        return self.nodes[0]

    def node(self, commit: str) -> TreeNode | None:
        return next((item for item in self.nodes if item.commit == commit), None)

    def require_node(self, commit: str) -> TreeNode:
        found = self.node(commit)
        if found is None:
            raise ValueError(f"{commit or '(empty)'} is not a node of the search tree")
        return found

    def direction(self, direction_id: str) -> Direction | None:
        return next((item for item in self.directions if item.direction_id == direction_id), None)

    def resolve(self, token: str) -> str:
        """The node a full commit or an unambiguous prefix of at least seven characters names, or ""."""
        candidate = token.strip().lower()
        if len(candidate) < 7:
            return ""
        matches = [item.commit for item in self.nodes if item.commit.startswith(candidate)]
        return matches[0] if len(matches) == 1 else ""

    def children(self, commit: str) -> list[TreeNode]:
        return [item for item in self.nodes if item.parent == commit]

    def ancestors(self, commit: str) -> list[TreeNode]:
        """The path from the root to ``commit``, both included."""
        path: list[TreeNode] = []
        current = self.node(commit)
        while current is not None:
            path.append(current)
            current = self.node(current.parent) if current.parent else None
        return list(reversed(path))

    def direction_nodes(self, direction_id: str) -> list[TreeNode]:
        return [item for item in self.nodes if item.direction == direction_id]

    def direction_best(self, direction_id: str) -> TreeNode | None:
        """The highest-scoring node of a direction; the earliest one wins a tie."""
        members = self.direction_nodes(direction_id)
        return max(members, key=_rank) if members else None

    def directions_from(self, commit: str) -> list[Direction]:
        """The directions opened at ``commit``."""
        return [item for item in self.directions if item.origin == commit]

    def best_measured(self) -> TreeNode:
        """The highest-scoring node, published or not; the earliest one wins a tie."""
        return max(self.nodes, key=_rank)

    def attempts_from(self, commit: str) -> list[Attempt]:
        return [item for item in self.attempts if item.start == commit]

    def line(self, start: str) -> Line:
        """The line ending at ``start``: the path to it, and every attempt made from a node on that path.

        An attempt from the path belongs to the line when it produced no node, when its node is on the path, or when it
        started at ``start`` itself; a node made from the path but not on it, and what grew from it, is another line.
        """
        path = self.ancestors(start)
        on_path = {item.commit for item in path}
        attempts = [
            item
            for item in self.attempts
            if item.start in on_path
            and (item.start == start or item.outcome is not AttemptOutcome.NODE or item.node in on_path)
        ]
        return Line(tuple(path), tuple(sorted(attempts, key=lambda item: (item.iteration, item.slot))))

    def allocate_direction(self, proposal: "DirectionProposal", *, origin: str, iteration: int) -> Direction:
        """Name, record and return a new direction opened at ``origin``."""
        self.require_node(origin)
        direction_id = f"d{self.next_direction_seq:03d}"
        self.next_direction_seq += 1
        direction = Direction(
            direction_id=direction_id,
            name=proposal.name,
            hypothesis=proposal.hypothesis,
            origin=origin,
            branch=f"{self.root_branch}-{direction_id}",
            expected_ceiling=proposal.expected_ceiling,
            failure_criterion=proposal.failure_criterion,
            created_iteration=iteration,
        )
        self.directions.append(direction)
        return direction

    def add_node(self, node: TreeNode) -> None:
        """Add a node grown from one already in the tree, on a direction the tree knows."""
        if self.node(node.commit) is not None:
            raise ValueError(f"the search tree already holds {node.commit}")
        self.require_node(node.parent)
        if self.direction(node.direction) is None:
            raise ValueError(f"node {node.commit} names an unknown direction {node.direction!r}")
        self.nodes.append(node)

    def add_measurement(self, commit: str, measurement: NodeMeasurement) -> TreeNode:
        """Fold a later measurement of a node into its record."""
        updated = self.require_node(commit).with_measurement(measurement)
        self.nodes = [updated if item.commit == commit else item for item in self.nodes]
        return updated

    def set_direction_status(self, direction_id: str, status: DirectionStatus, reason: str) -> None:
        if self.direction(direction_id) is None:
            raise ValueError(f"unknown direction {direction_id!r}")
        self.directions = [
            replace(item, status=status, status_reason=reason) if item.direction_id == direction_id else item
            for item in self.directions
        ]

    def record_hint_rebuttal(self, direction_id: str, rebuttal: str) -> None:
        self.directions = [
            replace(item, hint_rebuttals=(*item.hint_rebuttals, rebuttal))
            if item.direction_id == direction_id
            else item
            for item in self.directions
        ]

    def publish(self, commit: str, *, head: str) -> None:
        """Record ``commit`` as the published best, published on the campaign branch as ``head``."""
        self.require_node(commit)
        self.published_best = commit
        self.published_head = head

    def open_round(self, round_: Round) -> None:
        if self.round is not None:
            raise ValueError(f"iteration {self.round.iteration} still holds the round")
        for slot in round_.slots:
            self.require_node(slot.start)
        self.round = round_

    def mark_slot(self, index: int, status: SlotStatus) -> None:
        round_ = self._open_round()
        slots = tuple(replace(item, status=status) if item.index == index else item for item in round_.slots)
        self.round = replace(round_, slots=slots)

    def close_round(self, attempts: list[Attempt]) -> None:
        """Record the round's attempts and clear it."""
        iteration = self._open_round().iteration
        if any(item.iteration != iteration for item in attempts):
            raise ValueError(f"an attempt does not belong to iteration {iteration}")
        self.attempts.extend(attempts)
        self.round = None

    def close_interrupted_round(self, reason: str) -> None:
        """Close the round a session left open: a slot whose node was admitted keeps it, every other slot is lost."""
        round_ = self._open_round()
        admitted = {item.slot: item for item in self.nodes if item.iteration == round_.iteration and item.slot}
        self.close_round(
            [_interrupted_attempt(round_.iteration, slot, admitted.get(slot.index), reason) for slot in round_.slots]
        )

    def _open_round(self) -> Round:
        if self.round is None:
            raise ValueError("no round is open")
        return self.round


@dataclass(frozen=True)
class DirectionProposal:
    """What a planner wrote down when it opened a direction."""

    name: str
    hypothesis: str
    expected_ceiling: str
    failure_criterion: str


def _rank(node: TreeNode) -> tuple[float, int]:
    return (node.score, -node.iteration)


def lost_attempt(iteration: int, slot: Slot, reason: str) -> Attempt:
    """The attempt of a slot that came to nothing measured, and why."""
    return Attempt(
        iteration=iteration,
        slot=slot.index,
        slot_type=slot.slot_type,
        start=slot.start,
        direction=slot.direction,
        outcome=AttemptOutcome.ERROR,
        score=None,
        node="",
        archive_path="",
        reason=slot.reason,
        detail=reason,
        agent_profile_id=slot.agent_profile_id,
    )


def _interrupted_attempt(iteration: int, slot: Slot, node: TreeNode | None, reason: str) -> Attempt:
    lost = lost_attempt(iteration, slot, reason)
    if node is None:
        return lost
    return replace(lost, outcome=AttemptOutcome.NODE, score=node.score, node=node.commit, detail="")


__all__ = [
    "Attempt",
    "AttemptOutcome",
    "Direction",
    "DirectionProposal",
    "DirectionStatus",
    "Line",
    "MergeFailure",
    "NodeMeasurement",
    "NodeSource",
    "Round",
    "SearchTree",
    "Slot",
    "SlotStatus",
    "SlotType",
    "TreeNode",
    "lost_attempt",
]
