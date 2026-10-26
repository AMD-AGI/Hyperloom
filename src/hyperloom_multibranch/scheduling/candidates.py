# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which committed specialist commits the integrator offers on an integration head.

A commit is offered once per head, until it is merged or found already on the head, except where the answer it
already has still holds. An answer holds when it is about the candidate's own change rather than about the head the
change met: ``NOT_STACKABLE`` is the rule gate's measured one, and a merge test the candidate failed is another --
the merged recipe did not run, and a head that changes nothing the change names builds the same recipe again. The
change is the one a merge carries: from the candidate's merge base with the head, so it holds the change of every
parent of the candidate that was never merged. The integration head only grows, so such an answer stands until a
later merge changes one of the things that change names -- a server argument, an environment variable, a kernel
artifact, a component's upstream pin or one of the files a component's patches touch -- and then the candidate is
offered again. Two candidates that patch the same component in different files name nothing in common and do not
release each other. A merge attempt that settled nothing comes back on the next head whatever it named: a regression
set that did not run clean, a conflict, a candidate another of the round's merges superseded, and a test that ended
as an infrastructure failure -- the node's, not the candidate's -- up to :data:`MERGE_INFRA_RETRIES` times per
candidate. Past that such a test holds as an answer, as one the candidate failed does: a merge that ends that way on
every head costs a test each time. The session's base image is fixed, and
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

from hyperloom.common.unified_diff import touched_paths
from hyperloom.contracts.measurement import FailureClass, Validity
from hyperloom.contracts.recipe import PatchRef, Recipe, diff
from hyperloom_multibranch.contracts.notebook.interfaces import CommitQuery, RecordSearch
from hyperloom_multibranch.contracts.notebook.records import CommittedCommit, MergeState
from hyperloom_multibranch.contracts.notebook.vocab import MergeOutcome
from hyperloom_multibranch.contracts.repo import ExperimentRepo

#: How many times a candidate is offered again after a merge test that ended as an infrastructure failure.
MERGE_INFRA_RETRIES = 2


def _upstreams(recipe: Recipe) -> dict[str, str]:
    return {pinned.component: pinned.upstream_ref for pinned in recipe.components}


def _patches(recipe: Recipe) -> dict[str, frozenset[str]]:
    return {pinned.component: frozenset(patch.sha256 for patch in pinned.patches) for pinned in recipe.components}


def _kernels(recipe: Recipe) -> dict[str, str]:
    return {f"{artifact.kind.value} {artifact.target}": artifact.sha256 for artifact in recipe.kernels}


def _differing(before: Mapping[str, object], after: Mapping[str, object]) -> set[str]:
    return {key for key in {*before, *after} if before.get(key) != after.get(key)}


def _rebased(before: Recipe, after: Recipe) -> set[str]:
    """The components both recipes pin, at different upstream commits: every file they build moved with the pin."""

    old, new = _upstreams(before), _upstreams(after)
    return {name for name in old.keys() & new.keys() if old[name] != new[name]}


def _restacked(before: Recipe, after: Recipe) -> dict[str, frozenset[str]]:
    """Per component whose upstream pin held still, the sha256 of the patches one side stacks and the other does not.

    A component only one side names is one its patches create out of the installed package, so its appearance is a
    change of those patches rather than a move of the pin.
    """

    old, new = _patches(before), _patches(after)
    stood = {*old, *new} - _rebased(before, after)
    changed = ((name, old.get(name, frozenset()) ^ new.get(name, frozenset())) for name in stood)
    return {name: shas for name, shas in changed if shas}


def touched(before: Recipe, after: Recipe, patched: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """What a change from ``before`` to ``after`` names: each argument, variable, kernel slot, upstream pin and
    patched component file it sets, drops or moves, whatever the value it gives them.

    ``patched`` gives the files a patch touches, by sha256, for every patch the two recipes do not both stack.
    """

    change = diff(before, after)
    names = {f"arg {item.name}" for item in change.args}
    names |= {f"env {item.name}" for item in change.envs}
    names |= {f"component {name}" for name in _rebased(before, after)}
    names |= {f"kernel {slot}" for slot in _differing(_kernels(before), _kernels(after))}
    names |= {
        f"component {name} file {path}"
        for name, shas in _restacked(before, after).items()
        for sha in shas
        for path in patched[sha]
    }
    return frozenset(names)


def _merge_base(commit: CommittedCommit, unmerged: Mapping[str, CommittedCommit]) -> str:
    """The commit a merge of ``commit`` carries its change from: its nearest ancestor that is merged or no specialist
    commit, of ``unmerged`` the specialist commits not merged.

    The notebook answers for the repository's merge base because the integrator records every commit a landed merge
    carries as merged: one the notebook does not record as merged is on no integration head, and one it does is on
    every head since its merge.
    """

    base = commit.parent_commit
    while base in unmerged:
        base = unmerged[base].parent_commit
    return base


class CandidateNotebook(CommitQuery, RecordSearch, Protocol):
    """What :class:`Candidates` reads: the committed commits, and how a candidate's last merge test ended."""


class Candidates:
    """The committed commits the integrator offers, and the recipes and patches it has read to decide.

    A commit's recipe never changes and a patch's text is fixed by its sha256, so each is read from ``repo`` once:
    the idle merge loop asks on every tick, and both sides of the overlap are changes between commits already read.
    The notebook keeps only a candidate's latest merge, so the merge tests that ended as infrastructure failures are
    tallied here as each is read; every retry is offered from a read of the one before it.
    """

    def __init__(self, notebook: CandidateNotebook, repo: ExperimentRepo) -> None:
        self._notebook = notebook
        self._repo = repo
        self._recipes: dict[str, Recipe] = {}
        self._patched: dict[str, frozenset[str]] = {}
        self._infra_ended: dict[str, set[str]] = {}

    async def offered_on(self, head: str) -> tuple[CommittedCommit, ...]:
        """The committed commits to test on ``head``, in the order the notebook gives them."""

        committed = await self._notebook.committed()
        unmerged = {commit.commit_id: commit for commit in committed if commit.merged_into is None}
        offered = []
        for commit in committed:
            if await self._offered(commit, head, unmerged):
                offered.append(commit)
        return tuple(offered)

    async def _offered(self, commit: CommittedCommit, head: str, unmerged: Mapping[str, CommittedCommit]) -> bool:
        last = commit.last_merge
        if commit.merged_into is not None:
            return False
        if last is None:
            return True
        if last.integration_head == head or last.outcome is MergeOutcome.ALREADY_ON_HEAD:
            return False
        if not await self._answered_the_change(commit.commit_id, last):
            return True
        own = await self._touched_between(_merge_base(commit, unmerged), commit.commit_id)
        return bool(own & await self._touched_between(last.integration_head, head))

    async def _answered_the_change(self, candidate: str, last: MergeState) -> bool:
        """Whether ``last`` settled something about ``candidate``'s own change rather than about the head it met.

        A merge test the candidate failed is such an answer: what the toolbox could not run it does not run again on
        a head that changes nothing the change names. A test that ended as an infrastructure failure is not, until the
        candidate had more than :data:`MERGE_INFRA_RETRIES` of them, and neither is a round that superseded the
        candidate or a regression set that did not run clean -- they leave the candidate ``PENDING`` too.
        """

        if last.outcome is MergeOutcome.NOT_STACKABLE:
            return True
        if last.outcome is not MergeOutcome.PENDING:
            return False
        request_id = str(last.request_id)
        result = (await self._notebook.record(request_id)).result
        if result is None:
            return False
        if result.failure_class is FailureClass.INFRA:
            return self._infra_endings(candidate, request_id) > MERGE_INFRA_RETRIES
        return result.validity is Validity.CANDIDATE_FAILED

    def _infra_endings(self, candidate: str, request_id: str) -> int:
        """How many of ``candidate``'s merge tests ended as infrastructure failures, ``request_id`` among them."""

        ended = self._infra_ended.setdefault(candidate, set())
        ended.add(request_id)
        return len(ended)

    async def _touched_between(self, before: str, after: str) -> frozenset[str]:
        old, new = await self._recipe(before), await self._recipe(after)
        wanted = {sha for shas in _restacked(old, new).values() for sha in shas}
        patched = {}
        for commit, recipe in ((before, old), (after, new)):
            for patch in (stacked for component in recipe.components for stacked in component.patches):
                if patch.sha256 in wanted:
                    patched[patch.sha256] = await self._patch_files(commit, patch)
        return touched(old, new, patched)

    async def _patch_files(self, commit: str, patch: PatchRef) -> frozenset[str]:
        """The files ``patch`` touches, read from ``commit``, whose recipe stacks it."""

        if patch.sha256 not in self._patched:
            self._patched[patch.sha256] = frozenset(touched_paths(await self._repo.patch_text(commit, patch.path)))
        return self._patched[patch.sha256]

    async def _recipe(self, commit: str) -> Recipe:
        if commit not in self._recipes:
            self._recipes[commit] = await self._repo.read_recipe(commit)
        return self._recipes[commit]


__all__ = ["MERGE_INFRA_RETRIES", "CandidateNotebook", "Candidates", "touched"]
