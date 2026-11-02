# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The search tree a tree-strategy campaign grows: nodes, directions, attempts and the round in progress."""

from __future__ import annotations

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

    @property
    def score(self) -> float:
        """The mean of every pristine-relative score measured for this node."""
        return keep_score(list(self.samples)) or 0.0

    @property
    def is_root(self) -> bool:
        return not self.parent


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

    def history_scope(self, start: str) -> frozenset[int]:
        """Iterations whose history belongs to the line ending at ``start``.

        The nodes on the path, the attempts from any of them that produced no node, and every attempt made from the
        start itself; a node made from an ancestor that is not on the path belongs to another line.
        """
        path = self.ancestors(start)
        on_path = {item.commit for item in path}
        scope = {item.iteration for item in path if item.iteration > 0}
        for attempt in self.attempts:
            produced_nothing = attempt.outcome is not AttemptOutcome.NODE
            if attempt.start == start or (attempt.start in on_path and produced_nothing):
                scope.add(attempt.iteration)
        return frozenset(scope)

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

    def add_samples(self, commit: str, samples: tuple[float, ...]) -> TreeNode:
        """Fold a later measurement of a node into its record."""
        updated = replace(self.require_node(commit), samples=(*self.require_node(commit).samples, *samples))
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
        if self.round is None:
            raise ValueError("no round is open")
        slots = tuple(replace(item, status=status) if item.index == index else item for item in self.round.slots)
        self.round = replace(self.round, slots=slots)

    def close_round(self, attempts: list[Attempt]) -> None:
        """Record the round's attempts and clear it."""
        if self.round is None:
            raise ValueError("no round is open")
        if any(item.iteration != self.round.iteration for item in attempts):
            raise ValueError(f"an attempt does not belong to iteration {self.round.iteration}")
        self.attempts.extend(attempts)
        self.round = None


@dataclass(frozen=True)
class DirectionProposal:
    """What a planner wrote down when it opened a direction."""

    name: str
    hypothesis: str
    expected_ceiling: str
    failure_criterion: str


def _rank(node: TreeNode) -> tuple[float, int]:
    return (node.score, -node.iteration)


__all__ = [
    "Attempt",
    "AttemptOutcome",
    "Direction",
    "DirectionProposal",
    "DirectionStatus",
    "NodeSource",
    "Round",
    "SearchTree",
    "Slot",
    "SlotStatus",
    "SlotType",
    "TreeNode",
]
