# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which committed specialist commits the integrator offers on an integration head.

A commit is offered once per head, until it is merged or found already on the head, except where the answer it
already has still holds. An answer holds when it is about the candidate's own change rather than about the head the
change met: ``NOT_STACKABLE`` is the rule gate's measured one, and a merge test the candidate failed is another --
the merged recipe did not run, and a head that changes nothing the change names builds the same recipe again. The
integration head only grows, so such an answer stands until a later merge changes one of the things the candidate's
own change names -- a server argument, an environment variable, a component or a kernel artifact -- and then the
candidate is offered again. A merge attempt that settled nothing comes back on the next head whatever it named: a
test whose measurement failed, a regression set that did not run clean, a conflict, and a candidate another of the
round's merges superseded. The session's base image is fixed, and
:func:`~hyperloom_multibranch.repo.merge.carry` refuses a candidate that moves it, so an image is never one of the
things a change names here.

The asymmetry with a merged or already-on-head commit is deliberate: its change is on the head, and a later merge
that overrode a value it set was itself judged an improvement over a head that held the old value, so the old value
has been measured and beaten.

:func:`touched` names what a change sets, not what it sets it to, because a candidate competes with whatever value
the head now holds for a setting it names; ``similarity.own_change`` compares the same recipes by value, which
answers how alike two branches' changes are.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from hyperloom.contracts.measurement import Validity
from hyperloom.contracts.recipe import Recipe, diff
from hyperloom_multibranch.contracts.notebook.interfaces import CommitQuery, RecordSearch
from hyperloom_multibranch.contracts.notebook.records import CommittedCommit, MergeState
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome
from hyperloom_multibranch.contracts.repo import ExperimentRepo


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
    """What a change from ``before`` to ``after`` names: each argument, variable, component and kernel slot it sets,
    drops or moves, whatever the value it gives them."""

    change = diff(before, after)
    names = {f"arg {item.name}" for item in change.args}
    names |= {f"env {item.name}" for item in change.envs}
    names |= {f"component {name}" for name in _differing(_components(before), _components(after))}
    names |= {f"kernel {slot}" for slot in _differing(_kernels(before), _kernels(after))}
    return frozenset(names)


class CandidateNotebook(CommitQuery, RecordSearch, Protocol):
    """What :class:`Candidates` reads: the committed commits, and how a candidate's last merge test ended."""


class Candidates:
    """The committed commits the integrator offers, and the recipes it has read to decide.

    A commit's recipe never changes, so each one is read from ``repo`` once: the idle merge loop asks on every tick,
    and both sides of the overlap are changes between commits already read.
    """

    def __init__(self, notebook: CandidateNotebook, repo: ExperimentRepo) -> None:
        self._notebook = notebook
        self._repo = repo
        self._recipes: dict[str, Recipe] = {}

    async def offered_on(self, head: str) -> tuple[CommittedCommit, ...]:
        """The committed commits to test on ``head``, in the order the notebook gives them."""

        offered = []
        for commit in await self._notebook.committed():
            if await self._offered(commit, head):
                offered.append(commit)
        return tuple(offered)

    async def _offered(self, commit: CommittedCommit, head: str) -> bool:
        last = commit.last_merge
        if commit.merged_into is not None:
            return False
        if last is None:
            return True
        if last.integration_head == head or last.outcome is MergeOutcome.ALREADY_ON_HEAD:
            return False
        if not await self._answered_the_change(last):
            return True
        own = await self._touched_between(commit.parent_commit, commit.commit_id)
        return bool(own & await self._touched_between(last.integration_head, head))

    async def _answered_the_change(self, last: MergeState) -> bool:
        """Whether ``last`` settled something about the candidate's own change rather than about the head it met.

        A merge test the candidate failed is such an answer: what the toolbox could not run it does not run again on
        a head that changes nothing the change names. A measurement that failed is not, and neither is a round that
        superseded the candidate or a regression set that did not run clean -- they leave the candidate ``PENDING``
        too.
        """

        if last.outcome is MergeOutcome.NOT_STACKABLE:
            return True
        if last.outcome is not MergeOutcome.PENDING:
            return False
        result = (await self._notebook.record(str(last.request_id))).result
        return result is not None and result.validity is Validity.CANDIDATE_FAILED

    async def _touched_between(self, before: str, after: str) -> frozenset[str]:
        return touched(await self._recipe(before), await self._recipe(after))

    async def _recipe(self, commit: str) -> Recipe:
        if commit not in self._recipes:
            self._recipes[commit] = await self._repo.read_recipe(commit)
        return self._recipes[commit]


__all__ = ["CandidateNotebook", "Candidates", "touched"]
