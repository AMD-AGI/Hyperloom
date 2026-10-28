# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The open areas the judge reads every round: where a new branch could open, and whether one should open at all.

:func:`open_areas` lists the ops of the integration head's profile, or of the nearest commit it builds on that was
profiled, with the largest Amdahl ceilings, leaving out the op families a running branch works on (a claim it holds, or
its last explore test) and those this session's tests failed the playbook's ``failed_op_limit`` times or more. With too
little of EXPLORE left for a new branch's first commit, the playbook's ``last_open_minutes`` or else the median time a
new branch took to it, it says that no branch should open. The direction families' odds are the specialists' menu
(:mod:`rsi.menu`), not the judge's: a family has too few tests to rank ops by.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.blackboard.records import BlackboardSnapshot, amdahl_ceiling_pct
from hyperloom_multibranch.contracts.blackboard.vocab import ClaimStatus
from hyperloom_multibranch.blackboard.families import OP_FAMILIES, family_of, op_family
from hyperloom_multibranch.blackboard.rsi.menu import explored, record_family
from hyperloom_multibranch.blackboard.rsi.playbook import H0, Playbook

#: How many ops the open areas list at most.
OPEN_OPS = 5
_MINUTE = 60.0
_OP_FAMILIES = frozenset(name for name, _ in OP_FAMILIES)


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
class OpenAreas:
    """Where a new branch could open: the ``ops`` of the profile of ``profiled`` left open, the op families left out
    because running branches work on them (``covered``, by family, the branches) or this session failed them
    (``failed``, by family, how often), the minutes left of EXPLORE and those a new branch takes to its first
    commit."""

    profiled: str = ""
    ops: tuple[OpenOp, ...] = ()
    covered: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    failed: Mapping[str, int] = field(default_factory=dict)
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
            ] or ["- none: every large op is worked on or failed this session"]
        else:
            lines = ["Open areas for a new branch: neither the integration head nor a commit it builds on is profiled."]
        out = [f"{family}, worked on by {', '.join(branches)}" for family, branches in sorted(self.covered.items())]
        out += [f"{family}, failed {count} times this session" for family, count in sorted(self.failed.items())]
        if out:
            lines.append(f"Left out: {'; '.join(out)}.")
        if self.too_late:
            lines.append(
                f"Time left: {self.minutes_left:.0f} min, short of the {self.first_commit_minutes:.0f} min a new "
                "branch takes to its first commit: open no new branch."
            )
        return "\n".join(lines)


def _profiled(snapshot: BlackboardSnapshot, head: str) -> str:
    return next((commit for commit in (head, *snapshot.lineage.ancestry(head)) if commit in snapshot.op_shares), "")


def _covered(snapshot: BlackboardSnapshot) -> dict[str, tuple[str, ...]]:
    """The op families the running specialist branches work on, each with the branches: a claim held, or the last
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
    return {target: tuple(sorted(branches)) for target, branches in working.items()}


def _failed(snapshot: BlackboardSnapshot) -> Counter[str]:
    """How often this session's explore tests of each op family did not commit."""

    found: Counter[str] = Counter()
    for record in snapshot.records:
        family = record_family(snapshot, record) if explored(record) else None
        if family is not None and record.commit_id is None and family.target in _OP_FAMILIES:
            found[family.target] += 1
    return found


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
    profiled = _profiled(snapshot, integration.head_commit) if integration is not None else ""
    covered = _covered(snapshot)
    failed = {family: count for family, count in _failed(snapshot).items() if count >= numbers.failed_op_limit}
    shares = snapshot.op_shares[profiled].largest(len(snapshot.op_shares[profiled].shares)) if profiled else ()
    ops = [OpenOp(op, op_family(op), share) for op, share in shares if share > 0]
    held = {*covered, *failed}
    left = max(0.0, seconds_left) / _MINUTE
    needed = numbers.last_open_minutes if numbers.last_open_minutes is not None else first_commit_minutes
    shown_covered = {family: branches for family, branches in covered.items() if any(op.family == family for op in ops)}
    return OpenAreas(
        profiled,
        tuple(op for op in ops if op.family not in held)[:OPEN_OPS],
        shown_covered,
        failed,
        left,
        needed,
    )


__all__ = ["OPEN_OPS", "OpenAreas", "OpenOp", "open_areas"]
