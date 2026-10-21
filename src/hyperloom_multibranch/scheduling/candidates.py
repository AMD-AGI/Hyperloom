# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which committed specialist commits the integrator offers on an integration head.

A commit is offered once per head, until it is merged or found already on the head, except where the answer it
already has still holds. ``NOT_STACKABLE`` on a commit's latest merge is a measured answer about that commit's own
change against what the head held: a settled merge test the rule gate judged no improvement, or a regression set
that failed on the merge. The integration head only grows, so that answer stands until a later merge changes one of
the things the candidate's own change names -- a server argument, an environment variable, a component, a kernel
artifact or the base image -- and then the candidate is offered again. A merge test that did not measure writes no
judgement, so its candidate is ``PENDING`` and comes back on the next head whatever it named.

The asymmetry with a merged or already-on-head commit is deliberate: its change is on the head, and a later merge
that overrode a value it set was itself judged an improvement over a head that held the old value, so the old value
has been measured and beaten.

:func:`touched` names what a change sets, not what it sets it to, because a candidate competes with whatever value
the head now holds for a setting it names; ``similarity.own_change`` compares the same recipes by value, which
answers how alike two branches' changes are.
"""

from __future__ import annotations

from collections.abc import Mapping

from hyperloom.contracts.recipe import Recipe, diff
from hyperloom_multibranch.contracts.notebook.interfaces import CommitQuery
from hyperloom_multibranch.contracts.notebook.records import CommittedCommit
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome
from hyperloom_multibranch.contracts.repo import ExperimentRepo

#: What :func:`touched` calls a moved base image or runtime snapshot.
IMAGE = "image"


def _components(recipe: Recipe) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {
        pinned.component: (pinned.upstream_ref, tuple(patch.sha256 for patch in pinned.patches))
        for pinned in recipe.components
    }


def _kernels(recipe: Recipe) -> dict[str, str]:
    return {f"{artifact.kind.value} {artifact.target}": artifact.sha256 for artifact in recipe.kernels}


def _differing(before: Mapping[str, object], after: Mapping[str, object]) -> set[str]:
    return {key for key in {*before, *after} if before.get(key) != after.get(key)}


def touched(before: Recipe, after: Recipe) -> frozenset[str]:
    """What a change from ``before`` to ``after`` names: each argument, variable, component, kernel slot and image
    it sets, drops or moves, whatever the value it gives them."""

    change = diff(before, after)
    names = {f"arg {item.name}" for item in change.args}
    names |= {f"env {item.name}" for item in change.envs}
    names |= {f"component {name}" for name in _differing(_components(before), _components(after))}
    names |= {f"kernel {slot}" for slot in _differing(_kernels(before), _kernels(after))}
    if change.base_image_changed:
        names.add(IMAGE)
    return frozenset(names)


async def _touched_between(repo: ExperimentRepo, before: str, after: str) -> frozenset[str]:
    return touched(await repo.read_recipe(before), await repo.read_recipe(after))


async def _offered(commit: CommittedCommit, repo: ExperimentRepo, head: str) -> bool:
    last = commit.last_merge
    if commit.merged_into is not None:
        return False
    if last is None:
        return True
    if last.integration_head == head or last.outcome is MergeOutcome.ALREADY_ON_HEAD:
        return False
    if last.outcome is not MergeOutcome.NOT_STACKABLE:
        return True
    own = await _touched_between(repo, commit.parent_commit, commit.commit_id)
    return bool(own & await _touched_between(repo, last.integration_head, head))


async def offered_on(notebook: CommitQuery, repo: ExperimentRepo, head: str) -> tuple[CommittedCommit, ...]:
    """The committed commits to test on ``head``, in the order the notebook gives them."""

    offered = [commit for commit in await notebook.committed() if await _offered(commit, repo, head)]
    return tuple(offered)


__all__ = ["IMAGE", "offered_on", "touched"]
