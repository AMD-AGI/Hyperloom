# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch against the integration head, where the session's gains stack: what the integration head holds that the
branch head lacks, and the branch moved onto it.

A change the integration head holds is measured already: :func:`integration_holdings` names its launch and the
specialist commits merged into it, marks those the branch head lacks, and lists the branch's own commits not on it
with how their merges went, so that the specialist neither tests nor replays what the integration already took in.

A specialist asks for its branch to follow the integration head with a branch-log entry opening with :data:`REROOT`.
Its session moves the branch (:func:`follow`) only once no candidate test of the branch is in flight: a gain lands only
on the head it was measured against. The move is recorded in the blackboard too, so the branch's head there is the new
one and the judge sees what waits on it. The branch's change the integration head lacks comes back as one delta on the
new head, the patches that no longer apply there written out to port by hand; the old head stays a merge candidate.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hyperloom.common.fs_utils import write_text_nofollow
from hyperloom.contracts.recipe import Recipe, RecipeError, diff
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry
from hyperloom_multibranch.contracts.blackboard.interfaces import ConflictingWrite, LineageSink
from hyperloom_multibranch.contracts.blackboard.records import (
    CommitNode,
    CommittedCommit,
    LineageEvent,
    BlackboardSnapshot,
)
from hyperloom_multibranch.contracts.blackboard.vocab import INTEGRATION_BRANCH, CommitOrigin, LineageKind, MergeOutcome
from hyperloom_multibranch.contracts.repo import BranchMoved, ExperimentRepo, RepoMerge
from hyperloom_multibranch.blackboard.changes import describe_change

#: Opens the branch-log entry with which a specialist asks for its branch to follow the integration head.
REROOT = "REROOT"
#: Merged commits a listing names, the latest last: a long session merges a few dozen, and older ones are what every
#: branch opened since holds anyway.
MERGED_SHOWN = 40
SHORT = 12
_SPECIALIST_ORIGINS = frozenset({CommitOrigin.COMMIT, CommitOrigin.SPLIT})


def reroot_requested(content: str) -> bool:
    """Whether the branch-log entry ``content`` asks for the branch to follow the integration head."""
    return content.lstrip().startswith(REROOT)


async def integration_holdings(repo: ExperimentRepo, snapshot: BlackboardSnapshot, head: str) -> str:
    """What the integration head holds, and what of it the branch head ``head`` lacks; the branch's own commits that
    are not on it, with their merges."""

    onto = await repo.head(INTEGRATION_BRANCH)
    integration, own = await repo.read_recipe(onto), await repo.read_recipe(head)
    on_head = set(snapshot.lineage.ancestry(head))
    on_integration = set(snapshot.lineage.ancestry(onto))
    lines = [f"The integration head is {onto}: {_launch(integration)}."]
    merged = _specialist_commits(snapshot, on_integration)[-MERGED_SHOWN:]
    if merged:
        lines.append("Merged into it, oldest first; never test or replay any of these:")
        lines += [_merged(snapshot, node, node.commit_id in on_head) for node in merged]
    lacks = _lacks(own, integration)
    lines.append(f"Your head {head} lacks of it: {lacks}." if lacks else f"Your head {head} holds all of it.")
    pending = _specialist_commits(snapshot, on_head - on_integration)
    if pending:
        committed = {item.commit_id: item for item in snapshot.committed}
        lines.append("Commits of your head that are not on it:")
        lines += [f"- {_named(snapshot, node)}: {_merge(committed.get(node.commit_id))}" for node in pending]
    return "\n".join(lines)


@dataclass(frozen=True)
class Followed:
    """What a request to follow the integration head did: ``notice`` for the branch log, ``detail`` for the
    specialist's next message (empty when the branch did not move)."""

    notice: str
    detail: str = ""


async def follow(
    repo: ExperimentRepo, blackboard: LineageSink, snapshot: BlackboardSnapshot, asked: BranchLogEntry, notes: Path
) -> Followed | None:
    """Move the branch that ``asked`` to follow the integration head onto the integration head ``snapshot`` records,
    carrying its change the integration head lacks onto it as a delta, and record the move through ``blackboard``, the
    branch's runtime handle; the patches of that change that do not apply there are written under ``notes``. ``None``
    while the blackboard cannot record a move yet: the branch has not started there, or the integration branch has no
    head.

    The integration head is the blackboard's, not the repository's: the integrator lands a merge before it writes the
    judgement, and the blackboard refuses a re-root onto a commit it has not recorded. A move the blackboard refuses all the
    same, the branch stopped meanwhile, is taken back."""

    nodes = {node.branch: node for node in snapshot.lineage.branches}
    own, integration = nodes.get(asked.branch), nodes.get(INTEGRATION_BRANCH)
    if own is None or integration is None:
        return None
    if own.stop_reason is not None:
        return Followed(f"your branch was not re-rooted: it was stopped ({own.stop_reason})")
    branch, head, onto = asked.branch, await repo.head(asked.branch), integration.head_commit
    try:
        carried = await repo.merge(head, onto)
        await repo.reroot(branch, head, onto)
    except RecipeError as error:
        return Followed(
            f"your branch was not re-rooted: its change cannot be carried onto the integration head ({error})"
        )
    except BranchMoved:
        return Followed(f"your branch was not re-rooted: it moved off {head} meanwhile; ask again once it settles")
    except ValueError:
        return Followed(f"your head {head} holds the integration head {onto} already; your branch stays where it is")
    try:
        await blackboard.record_lineage(
            LineageEvent(LineageKind.REROOT, branch, commit_id=onto, from_commit=head, reason=asked.content)
        )
    except ConflictingWrite as error:
        await repo.rewind(branch, onto, head)
        return Followed(f"your branch was not re-rooted: the blackboard refused the move ({error})")
    ported = await _write_conflicts(repo, head, carried, notes)
    notice = f"your branch was re-rooted from {head} onto the integration head {onto}"
    return Followed(notice, _carried(snapshot, head, onto, carried, ported))


def _launch(recipe: Recipe) -> str:
    envs = ", ".join(f"{name}={value}" for name, value in sorted(recipe.launch.envs.items()))
    patches = {item.component: len(item.patches) for item in recipe.components if item.patches}
    parts = {
        "server arguments": recipe.launch.server_args,
        "environment": envs,
        "patches": ", ".join(f"{count} to {name}" for name, count in sorted(patches.items())),
        "kernel tables": ", ".join(sorted(item.target for item in recipe.kernels)),
    }
    return "; ".join(f"{name} {value or 'none'}" for name, value in parts.items())


def _lacks(own: Recipe, integration: Recipe) -> str:
    """What ``integration`` sets that ``own`` does not, each setting with the value ``own`` has instead."""
    change = diff(own, integration)
    settings = [
        f"{item.name} {item.after}".rstrip() + (f" (yours: {item.before or 'set'})" if item.before is not None else "")
        for item in (*change.args, *change.envs)
        if item.after is not None
    ]
    if change.patches_only_in_second:
        settings.append(f"{len(change.patches_only_in_second)} patches")
    if change.kernels_only_in_second:
        settings.append(f"{len(change.kernels_only_in_second)} kernel tables")
    return "; ".join(settings)


def _specialist_commits(snapshot: BlackboardSnapshot, among: set[str]) -> list[CommitNode]:
    return sorted(
        (node for node in snapshot.lineage.commits if node.commit_id in among and node.origin in _SPECIALIST_ORIGINS),
        key=lambda node: node.at,
    )


def _named(snapshot: BlackboardSnapshot, node: CommitNode) -> str:
    """The commit, the intent it was claimed for and what its test changed."""
    record = next((item for item in snapshot.records if item.request_id == node.request_id), None)
    if record is None:
        return node.commit_id[:SHORT]
    intent = snapshot.intents.get(record.request.intent_fingerprint or "")
    claimed = f"{intent.target} / {intent.mechanism}: " if intent is not None else ""
    gain = f" ({record.judgement.gain_pct:+.2f}%)" if record.judgement is not None else ""
    return f"{node.commit_id[:SHORT]} of {node.branch}{gain}, {claimed}{describe_change(record.request.delta)}"


def _merged(snapshot: BlackboardSnapshot, node: CommitNode, on_head: bool) -> str:
    return f"- {_named(snapshot, node)}" + ("" if on_head else " [not on your head]")


def _merge(commit: CommittedCommit | None) -> str:
    last = commit.last_merge if commit is not None else None
    if last is None:
        return "the integrator has not tried it yet; merging it is the integrator's job"
    if last.outcome is MergeOutcome.PENDING:
        return f"the integrator is testing its merge onto {last.integration_head}"
    if last.outcome is MergeOutcome.CONFLICT_SKIPPED:
        return f"its merge onto {last.integration_head} conflicted, untested ({last.reason})"
    return f"{last.outcome.value} on {last.integration_head} ({last.reason})"


async def _write_conflicts(repo: ExperimentRepo, head: str, carried: RepoMerge, notes: Path) -> list[Path]:
    """Write each patch of ``carried`` that does not apply on the new head under ``notes/<head>``, readable by the
    specialist; ``notes`` sits in its workspace, so nothing on the way is followed through a link it left."""
    written = []
    for conflict in carried.patch_conflicts:
        text = await repo.patch_text(head, conflict.path)
        rel = (notes.name, head[:SHORT], f"{conflict.component}-{PurePosixPath(conflict.path).name}")
        written.append(await asyncio.to_thread(write_text_nofollow, notes.parent, rel, text, mode=0o644))
    return written


def _carried(snapshot: BlackboardSnapshot, head: str, onto: str, carried: RepoMerge, ported: Sequence[Path]) -> str:
    lines = [
        f"Your branch now starts at the integration head {onto}; your old head {head} stays a merge candidate. "
        "Call worktree_sync before your next change."
    ]
    if not carried.carried and not ported:
        lines.append("Every commit of your old head is on the integration head: nothing of it is left out.")
        return "\n".join(lines)
    committed = {item.commit_id: item for item in snapshot.committed}
    by_id = {node.commit_id: node for node in snapshot.lineage.commits}
    lines.append("Your commits the integration head lacks:")
    lines += [
        f"- {_named(snapshot, by_id[commit]) if commit in by_id else commit[:SHORT]}: {_merge(committed.get(commit))}"
        for commit in carried.carried
    ]
    lines.append(
        "Their change carried onto the new head, as one delta: "
        + json.dumps(carried.delta.to_document(), sort_keys=True)
    )
    lines += [
        f"In it, {item.name} takes your {item.after} over the integration head's {item.before}."
        for item in carried.config_conflicts
    ]
    if ported:
        lines.append(
            "These patches of it do not apply on the new head and are left out of the delta; port them by hand: "
            + ", ".join(map(str, ported))
        )
    return "\n".join(lines)


__all__ = ["MERGED_SHOWN", "REROOT", "Followed", "follow", "integration_holdings", "reroot_requested"]
