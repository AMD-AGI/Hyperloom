# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The open areas the judge reads every round: where a new branch could open, and whether one should open at all.

:func:`open_areas` lists the ops of the integration head's profile, or of the nearest commit it builds on that was
profiled, with the largest Amdahl ceilings, leaving out the ops a running branch works on (its focus, or an op of the
family of a claim it holds or of its last explore test, which is all a branch opened without a focus shows) and the
spent ones: those the playbook's ``failed_op_limit`` branches opened on the integration head as it stands were focused
on and stopped without a commit. A spent op opens again once the integration head moves. With too little of EXPLORE
left for a new branch's first commit, the playbook's ``last_open_minutes`` or else the median time a new branch took to
it, it says that no branch should open. The direction families' odds are the specialists' menu (:mod:`rsi.menu`), not
the judge's: a family has too few tests to rank ops by.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot, BranchNode, amdahl_ceiling_pct
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus
from hyperloom_multibranch.blackboard.families import family_of, op_family
from hyperloom_multibranch.blackboard.rsi.menu import profiled_commit, record_family
from hyperloom_multibranch.blackboard.rsi.playbook import H0, Playbook

#: How many ops the open areas list at most.
OPEN_OPS = 5
_MINUTE = 60.0


@dataclass(frozen=True)
class OpenOp:
    """An op of the profile: its op family, ``None`` for one of none, its share of GPU time and its Amdahl ceiling."""

    op: str
    family: str | None
    share_pct: float

    @property
    def ceiling_pct(self) -> float:
        return amdahl_ceiling_pct(self.share_pct)


@dataclass(frozen=True)
class FocusHistory:
    """The branches this session opened on one op as their focus: how many, how many of them stopped without a
    commit, the commits they made and the best gain among those."""

    opened: int
    ended_without_commit: int
    commits: int = 0
    best_gain_pct: float | None = None

    def render(self) -> str:
        made = "no commit" if self.best_gain_pct is None else f"commits {self.commits}, best {self.best_gain_pct:+.1f}%"
        return f"opened {self.opened}, ended without a commit {self.ended_without_commit}, {made}"


@dataclass(frozen=True)
class OpenAreas:
    """Where a new branch could open: the ``ops`` of the profile of ``profiled`` left open, the ops left out because
    running branches work on them (``covered``, by op, the branches) or they are spent (``spent``, by op, the
    history of the branches focused on it), the minutes left of EXPLORE and those a new branch takes to its first
    commit."""

    profiled: str = ""
    ops: tuple[OpenOp, ...] = ()
    covered: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    spent: Mapping[str, FocusHistory] = field(default_factory=dict)
    minutes_left: float | None = None
    first_commit_minutes: float | None = None

    @property
    def too_late(self) -> bool:
        """Whether a branch opened now would not reach its first commit before EXPLORE ends."""

        left, needed = self.minutes_left, self.first_commit_minutes
        return left is not None and needed is not None and left < needed

    def render(self) -> str:
        if self.profiled:
            lines = [f"Open areas for a new branch, from the profile of {self.profiled}:"]
            lines += [
                f"- {item.op} ({item.family or 'no op family'}): {item.share_pct:.1f}% of GPU time, at most "
                f"+{item.ceiling_pct:.1f}%"
                for item in self.ops
            ] or ["- none: every large op is worked on or spent"]
        else:
            lines = ["Open areas for a new branch: neither the integration head nor a commit it builds on is profiled."]
        if self.covered:
            worked = "; ".join(f"{op}, worked on by {', '.join(branches)}" for op, branches in self.covered.items())
            lines.append(f"Left out: {worked}.")
        if self.spent:
            lines.append(
                "Spent focus, left out until the integration head moves: the branches focused on it here ended "
                "without a commit. Open one again only on an angle they did not take:"
            )
            lines += [f"- {op}: {history.render()}" for op, history in self.spent.items()]
        if self.too_late:
            lines.append(
                f"Time left: {self.minutes_left:.0f} min, short of the {self.first_commit_minutes:.0f} min a new "
                "branch takes to its first commit: open no new branch."
            )
        return "\n".join(lines)


def _working(snapshot: BlackboardSnapshot) -> dict[str, set[str]]:
    """The families the running specialist branches work on, each with the branches: a claim held, or the last
    explore test."""

    running = {row.branch for row in snapshot.progress if row.running and row.branch != INTEGRATION_BRANCH}
    working: dict[str, set[str]] = {}
    for claim in snapshot.claims:
        if claim.status is ClaimStatus.HELD and claim.holder in running:
            intent = claim.intent
            working.setdefault(family_of(intent.target, intent.mechanism, intent.layer).target, set()).add(claim.holder)
    latest = {}
    for record in sorted(snapshot.records, key=lambda item: item.requested_at):
        if record.request.branch in running and (family := record_family(snapshot, record)) is not None:
            latest[record.request.branch] = family.target
    for branch, target in latest.items():
        working.setdefault(target, set()).add(branch)
    return working


def _focused(snapshot: BlackboardSnapshot) -> dict[str, list[BranchNode]]:
    """The specialist branches this session opened with a focus, by the op they focus on."""

    focused: dict[str, list[BranchNode]] = {}
    for node in snapshot.lineage.branches:
        if node.focus and node.branch != INTEGRATION_BRANCH:
            focused.setdefault(node.focus, []).append(node)
    return focused


def _spent(snapshot: BlackboardSnapshot, focused: Sequence[BranchNode], head: str, limit: int) -> FocusHistory | None:
    """The history of the ``focused`` branches of an op when ``limit`` of those opened on ``head`` stopped without a
    commit, else ``None``."""

    committed = {commit.branch for commit in snapshot.committed}
    bare = [node for node in focused if node.stopped_at is not None and node.branch not in committed]
    if sum(1 for node in bare if node.start_commit == head) < limit:
        return None
    branches = {node.branch for node in focused}
    gains = [commit.gain_pct for commit in snapshot.committed if commit.branch in branches]
    return FocusHistory(len(focused), len(bare), len(gains), max(gains, default=None))


def open_areas(
    snapshot: BlackboardSnapshot,
    *,
    seconds_left: float,
    first_commit_minutes: float | None,
    playbook: Playbook = H0,
) -> OpenAreas:
    """The open areas of the session as it stands, with ``seconds_left`` of EXPLORE: ``first_commit_minutes`` is the
    median time a new branch took to its first commit, which the ``playbook``'s ``last_open_minutes`` overrides when it
    sets one."""

    numbers = playbook.numbers
    integration = next((node for node in snapshot.lineage.branches if node.branch == INTEGRATION_BRANCH), None)
    head = integration.head_commit if integration is not None else ""
    profiled = profiled_commit(snapshot, head) if head else ""
    shares = snapshot.op_shares[profiled].largest(len(snapshot.op_shares[profiled].shares)) if profiled else ()
    ops = [OpenOp(op, op_family(op), share) for op, share in shares if share > 0]
    focused = _focused(snapshot)
    working = _working(snapshot)
    covered: dict[str, tuple[str, ...]] = {}
    spent: dict[str, FocusHistory] = {}
    for item in ops:
        nodes = focused.get(item.op, [])
        running = {node.branch for node in nodes if node.stopped_at is None} | working.get(item.family or "", set())
        if running:
            covered[item.op] = tuple(sorted(running))
        elif (history := _spent(snapshot, nodes, head, numbers.failed_op_limit)) is not None:
            spent[item.op] = history
    left = max(0.0, seconds_left) / _MINUTE
    needed = numbers.last_open_minutes if numbers.last_open_minutes is not None else first_commit_minutes
    open_ops = tuple(item for item in ops if item.op not in covered and item.op not in spent)[:OPEN_OPS]
    return OpenAreas(profiled, open_ops, covered, spent, left, needed)


__all__ = ["OPEN_OPS", "FocusHistory", "OpenAreas", "OpenOp", "open_areas"]
