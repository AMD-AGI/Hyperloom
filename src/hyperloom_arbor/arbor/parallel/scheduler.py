"""Which lanes start now, pure: at most ``max_nodes`` nodes, in (level, rank, id) order.

The cap counts nodes, not lanes or GPUs (decision 0003): a lane measures on a
whole node, and nothing runs on its other GPUs. Every lane that has a job,
queued or running, holds its nodes against the cap, so the lanes a tick
submits never add up to more than the cap once they all start.

The queue is not overtaken: when the lane at its head needs more nodes than
are free, nothing behind it starts either, or a wide lane could wait forever
behind a stream of narrow ones. A lane wider than the cap waits for the whole
cap. Multi-node lanes run on one node until phase 4 (kinds.py); the cap still
reserves what their plan asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Lane states that hold their nodes against the cap.
HOLDING = ("SUBMITTING", "PENDING", "RUNNING", "FAILED")


@dataclass(frozen=True)
class Slot:
    """What the scheduler reads of a lane."""
    id: str
    state: str
    level: int = 0
    rank: int = 999
    nodes: int = 1
    depends_on: tuple = field(default_factory=tuple)    # lane IDs on this round

    def key(self) -> tuple:
        return (self.level, self.rank, self.id)


def promote(slots: list[Slot]) -> list[str]:
    """WAITING_DEPS lanes whose every dependency has CLOSED: they join the queue."""
    closed = {s.id for s in slots if s.state == "CLOSED"}
    return sorted((s.id for s in slots if s.state == "WAITING_DEPS" and set(s.depends_on) <= closed),
                  key=lambda i: next(s.key() for s in slots if s.id == i))


def busy(slots: list[Slot], max_nodes: int) -> int:
    return sum(min(s.nodes, max_nodes) for s in slots if s.state in HOLDING)


def plan_starts(slots: list[Slot], max_nodes: int) -> list[str]:
    """The QUEUED lanes to submit now, in order, while their nodes fit."""
    free = max_nodes - busy(slots, max_nodes)
    out = []
    for s in sorted((s for s in slots if s.state == "QUEUED"), key=Slot.key):
        need = min(s.nodes, max_nodes)
        if need > free:
            break
        out.append(s.id)
        free -= need
    return out
