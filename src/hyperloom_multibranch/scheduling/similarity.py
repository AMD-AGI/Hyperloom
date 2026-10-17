# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How alike the running specialist branches are, for the judge to weigh: their own changes, compared pair by pair.

A branch's own change is what ``hyperloom.contracts.recipe.diff`` finds from the recipe of its start commit to that of
its head: each server argument and environment variable at its new value, each patch and kernel added or dropped by
content hash, each component's new upstream ref, each item newly or no longer unexpressed, and a changed image. It is
read from the branch's own start, not the root, because two branches started from the same integration head share
every integration change they inherited, and from the root they would look alike for it. A branch without a commit of
its own has no change and is not compared, nor is the integration branch.

Two branches are listed as similar when the Jaccard index of their changes reaches :data:`MB_SIMILARITY_THRESHOLD`,
and as identical, with a score of 1.0, when their heads hold the same recipe. Nothing here stops a branch: the judge
reads the pairs and decides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from hyperloom.contracts.recipe import Recipe, ValueChange, diff
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.scheduling.roster import BranchKind, BranchStatus, Roster

#: The least Jaccard index of two branches' own changes at which the judge is shown them as similar.
MB_SIMILARITY_THRESHOLD = 0.8


@dataclass(frozen=True)
class SimilarBranch:
    """Another branch whose own change overlaps by ``score``; ``identical`` when both heads hold the same recipe."""

    branch: str
    score: float
    identical: bool


@dataclass(frozen=True)
class BranchChange:
    """A branch with a commit of its own: the digest of its head's recipe and its own change since its start."""

    branch: str
    head_digest: str
    change: frozenset[str]


def _values(kind: str, changes: Sequence[ValueChange]) -> set[str]:
    return {f"{kind} {item.name}" + (" unset" if item.after is None else f"={item.after}") for item in changes}


def _hashes(kind: str, added: frozenset[str], dropped: frozenset[str]) -> set[str]:
    return {f"{kind} +{value}" for value in added} | {f"{kind} -{value}" for value in dropped}


def own_change(start: Recipe, head: Recipe) -> frozenset[str]:
    """What ``head`` changes from ``start``, one comparable item per setting, file and ref it sets or drops."""

    change = diff(start, head)
    items = _values("arg", change.args) | _values("env", change.envs)
    items |= {
        f"ref {item.component}" + (" dropped" if item.after is None else f"@{item.after}") for item in change.refs
    }
    items |= _hashes("patch", change.patches_only_in_second, change.patches_only_in_first)
    items |= _hashes("kernel", change.kernels_only_in_second, change.kernels_only_in_first)
    items |= _hashes("unexpressed", change.unexpressed_only_in_second, change.unexpressed_only_in_first)
    if change.base_image_changed:
        items.add(f"image {head.base_image_digest} {head.runtime_snapshot}")
    return frozenset(items)


def _similarity(first: BranchChange, second: BranchChange) -> SimilarBranch | None:
    """How ``second`` is similar to ``first``; ``None`` below the threshold."""

    if first.head_digest == second.head_digest:
        return SimilarBranch(second.branch, 1.0, identical=True)
    union = first.change | second.change
    score = len(first.change & second.change) / len(union) if union else 0.0
    return SimilarBranch(second.branch, round(score, 3), identical=False) if score >= MB_SIMILARITY_THRESHOLD else None


def similar_pairs(branches: Sequence[BranchChange]) -> dict[str, tuple[SimilarBranch, ...]]:
    """Each branch's similar branches, both ways round; a branch similar to none is left out."""

    found: dict[str, list[SimilarBranch]] = {}
    for first, second in combinations(branches, 2):
        pair = _similarity(first, second)
        if pair is not None:
            found.setdefault(first.branch, []).append(pair)
            found.setdefault(second.branch, []).append(SimilarBranch(first.branch, pair.score, pair.identical))
    return {branch: tuple(similar) for branch, similar in found.items()}


class BranchSimilarity:
    """Reads the running specialist branches off the roster and their recipes out of the experiment repository."""

    def __init__(self, repo: ExperimentRepo, roster: Roster) -> None:
        self._repo = repo
        self._roster = roster

    async def similar_to(self) -> Mapping[str, tuple[SimilarBranch, ...]]:
        """Each running specialist branch's similar branches; a branch with no commit of its own is in no pair."""

        changed = []
        for entry in self._roster.entries():
            if entry.kind is not BranchKind.SPECIALIST or entry.status is not BranchStatus.RUNNING:
                continue
            head = await self._repo.head(entry.branch)
            if head == entry.start_commit:
                continue
            start, recipe = await self._repo.read_recipe(entry.start_commit), await self._repo.read_recipe(head)
            changed.append(BranchChange(entry.branch, recipe.digest(), own_change(start, recipe)))
        return similar_pairs(changed)


__all__ = [
    "MB_SIMILARITY_THRESHOLD",
    "BranchChange",
    "BranchSimilarity",
    "SimilarBranch",
    "own_change",
    "similar_pairs",
]
