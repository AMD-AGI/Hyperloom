# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A branch against the integration head, where the session's gains stack: what the integration head holds that the
branch head lacks.

A change the integration head holds is measured already: :func:`integration_holdings` names its launch and the
specialist commits merged into it, marks those the branch head lacks, and lists the branch's own commits not on it
with how their merges went, so that the specialist neither tests nor replays what the integration already took in.
"""

from __future__ import annotations

from hyperloom.contracts.recipe import Recipe, diff
from hyperloom_multibranch.contracts.notebook.records import CommitNode, CommittedCommit, NotebookSnapshot
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, CommitOrigin, MergeOutcome
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.notebook.changes import describe_change

#: Merged commits a listing names, the latest last: a long session merges a few dozen, and older ones are what every
#: branch opened since holds anyway.
MERGED_SHOWN = 40
SHORT = 12
_SPECIALIST_ORIGINS = frozenset({CommitOrigin.COMMIT, CommitOrigin.SPLIT})


async def integration_holdings(repo: ExperimentRepo, snapshot: NotebookSnapshot, head: str) -> str:
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


def _specialist_commits(snapshot: NotebookSnapshot, among: set[str]) -> list[CommitNode]:
    return sorted(
        (node for node in snapshot.lineage.commits if node.commit_id in among and node.origin in _SPECIALIST_ORIGINS),
        key=lambda node: node.at,
    )


def _named(snapshot: NotebookSnapshot, node: CommitNode) -> str:
    """The commit, the intent it was claimed for and what its test changed."""
    record = next((item for item in snapshot.records if item.request_id == node.request_id), None)
    if record is None:
        return node.commit_id[:SHORT]
    intent = snapshot.intents.get(record.request.intent_fingerprint or "")
    claimed = f"{intent.target} / {intent.mechanism}: " if intent is not None else ""
    gain = f" ({record.judgement.gain_pct:+.2f}%)" if record.judgement is not None else ""
    return f"{node.commit_id[:SHORT]} of {node.branch}{gain}, {claimed}{describe_change(record.request.delta)}"


def _merged(snapshot: NotebookSnapshot, node: CommitNode, on_head: bool) -> str:
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


__all__ = ["MERGED_SHOWN", "integration_holdings"]
