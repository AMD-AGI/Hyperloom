# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MB1 integrator: carries specialist commits onto the integration branch, in rounds of up to K merge tests.

Candidates are the notebook's committed specialist commits, in the order it gives them (foundation fixes first, then
by their own gain), whatever became of the branch that made them. Each is tried once per integration head. A round
tests up to the integration branch's K candidates at once on the same head; the first of them, in that order, that
improves it is merged, and every other improvement is superseded: it gets no judgement and is tried again on the new
head. A merge whose recipe the recipe contract refuses, or whose test the toolbox refuses, is skipped and recorded
like an unresolved conflict.

The experiment repository does the merge. Its rule for a configuration conflict is that the candidate's value wins,
and it replays the head's patches when the candidate moves a component to another upstream ref; the integrator
records every overridden value in the decision log. Patches that still do not apply go to :class:`LlmPatchResolver`;
a merge it cannot resolve is skipped and recorded. The merged change is tested from the integration head before
anything is written, so only an improvement reaches the integration branch and nothing on it is ever reverted.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace

from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import ComponentChange, PatchRef, Recipe, RecipeDelta, RecipeError, ValueChange
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Notebook
from hyperloom_multibranch.contracts.notebook.peers import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.records import CommittedCommit, MergeAttempt
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, MergeOutcome
from hyperloom_multibranch.contracts.scheduling import ExperimentRepo, PatchConflict, RepoMerge, RuleGate
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.llm import Completion, CompletionFailed
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_regression_passes
from hyperloom_multibranch.scheduling.requests import SessionTests, TestWithdrawn, integration_test_request

logger = logging.getLogger(__name__)

SUPERSEDED = "superseded: another candidate of the round was merged onto this head first"
#: The JSON key the resolver's reply carries the rewritten patch under; ``null`` means it cannot be combined.
PATCH_KEY = "patch"
RESOLVER_SYSTEM_PROMPT = (
    "You resolve patch conflicts while merging two optimizations of an LLM inference server. "
    "One patch of a component did not apply where the merge puts it. Rewrite that patch so it applies on top of "
    "the other patches of the same component and keeps its intent, without undoing what the other patches do. "
    f'Answer with one JSON object {{"{PATCH_KEY}": "<the rewritten patch as a unified diff>"}}, or '
    f'{{"{PATCH_KEY}": null}} when the two changes cannot be combined.'
)


@dataclass(frozen=True)
class MergeReport:
    """What came of one candidate on integration head ``onto``; ``config_conflicts`` took the candidate's value."""

    candidate: str
    onto: str
    outcome: MergeOutcome
    reason: str
    request_id: str | None = None
    merged_commit: str | None = None
    config_conflicts: tuple[ValueChange, ...] = ()


class LlmPatchResolver:
    """Asks a model to rewrite every patch a merge could not apply; one it cannot rewrite leaves the merge unresolved.

    A rewritten patch is not applied here: the merge test builds the merged recipe, and a patch that still does not
    apply fails that test as the candidate's failure.
    """

    def __init__(self, completion: Completion, repo: ExperimentRepo) -> None:
        self._completion = completion
        self._repo = repo

    async def resolve(self, candidate: str, merge: RepoMerge) -> RecipeDelta | None:
        """``merge.delta`` with a rewritten patch added for each conflict, or ``None`` when one stays unresolved."""

        delta = merge.delta
        for conflict in merge.patch_conflicts:
            text = await self._rewrite(candidate, merge.onto, conflict)
            if text is None:
                return None
            delta = _with_patch(delta, conflict.component, await self._repo.store_patch(conflict.component, text))
        return delta

    async def _rewrite(self, candidate: str, onto: str, conflict: PatchConflict) -> str | None:
        prompt = await self._prompt(candidate, onto, conflict)
        try:
            reply = await self._completion(RESOLVER_SYSTEM_PROMPT, prompt)
        except CompletionFailed as exc:
            logger.warning("patch resolver call for %s failed: %s", conflict.path, exc)
            return None
        answer = extract_last_json_with_key(reply, PATCH_KEY)
        patch = answer.get(PATCH_KEY) if answer is not None else None
        return patch if isinstance(patch, str) and patch.strip() else None

    async def _prompt(self, candidate: str, onto: str, conflict: PatchConflict) -> str:
        theirs = _patches_of(await self._repo.recipe_at(candidate), conflict.component)
        ours = _patches_of(await self._repo.recipe_at(onto), conflict.component)
        owner, other, other_stack = (candidate, onto, ours) if conflict.path in theirs else (onto, candidate, theirs)
        sections = [
            f"Component: {conflict.component}",
            f"Conflicting patch {conflict.path}, from commit {owner}:",
            await self._repo.patch_text(owner, conflict.path),
            f"What applying it said:\n{conflict.detail}",
            f"The other patches of {conflict.component} at commit {other}, in order:",
        ]
        for path in other_stack:
            if path != conflict.path:
                sections.append(f"--- {path} ---\n{await self._repo.patch_text(other, path)}")
        return "\n\n".join(sections)


def _patches_of(recipe: Recipe, component: str) -> tuple[str, ...]:
    pinned = recipe.component(component)
    return tuple(patch.path for patch in pinned.patches) if pinned is not None else ()


def _with_patch(delta: RecipeDelta, component: str, patch: PatchRef) -> RecipeDelta:
    changes = list(delta.components)
    index = next((i for i, change in enumerate(changes) if change.component == component), None)
    if index is None:
        changes.append(ComponentChange(component, add_patches=(patch,)))
    else:
        changes[index] = replace(changes[index], add_patches=(*changes[index].add_patches, patch))
    return replace(delta, components=tuple(changes))


def _untried(commit: CommittedCommit, head: str) -> bool:
    last = commit.last_merge
    return commit.merged_into is None and (last is None or last.integration_head != head)


def not_stackable_reason(candidate: str, judgement: Judgement) -> str:
    """Why the merge test of ``candidate`` keeps it off the integration branch; empty when it may be merged."""

    if judgement.verdict is not Verdict.IMPROVED:
        return f"{judgement.verdict.value} {judgement.gain_pct:+.2f}% on the integration head"
    if not judgement.accuracy_ok:
        return f"improvement of {judgement.gain_pct:+.2f}% failed the accuracy gate"
    if not mb_fake_regression_passes(candidate):
        return "the regression set failed"
    return ""


@dataclass(frozen=True)
class _TestedMerge:
    """A merge test that produced a judgement; ``report`` is still ``PENDING``."""

    report: MergeReport
    delta: RecipeDelta
    judgement: Judgement


class Integrator:
    """Merges committed specialist commits onto the integration branch; the caller runs one round at a time.

    A cancelled round takes its merge tests back; a caller that cancels one then awaits :meth:`abandon`, which leaves
    the repository and the notebook agreeing on the integration head.
    """

    def __init__(
        self,
        notebook: Notebook,
        repo: ExperimentRepo,
        tests: SessionTests,
        gate: RuleGate,
        resolver: LlmPatchResolver,
        decisions: DecisionLog,
    ) -> None:
        self._notebook = notebook
        self._repo = repo
        self._tests = tests
        self._gate = gate
        self._resolver = resolver
        self._decisions = decisions
        self._recording: asyncio.Future[str] | None = None

    async def integrate_next(self, width: int) -> tuple[MergeReport, ...]:
        """Test up to ``width`` commits not merged and not yet tried on the current head, at once; ``()`` when none is.

        At most one of them is merged, so the head moves once per round.
        """

        head = await self._repo.head(INTEGRATION_BRANCH)
        untried = [commit.commit_id for commit in await self._notebook.committed() if _untried(commit, head)]
        tested = await asyncio.gather(*(self._test(candidate, head) for candidate in untried[:width]))
        reports = []
        merged = False
        for item in tested:
            report = item if isinstance(item, MergeReport) else await self._conclude(item, merged)
            merged = merged or report.outcome is MergeOutcome.MERGED
            self._decisions.record(_decision(report))
            reports.append(report)
        return tuple(reports)

    async def _test(self, candidate: str, head: str) -> MergeReport | _TestedMerge:
        try:
            merge = await self._repo.merge(candidate, head)
            delta = merge.delta if merge.clean else await self._resolver.resolve(candidate, merge)
        except RecipeError as exc:
            return await self._skip(candidate, head, f"the merged recipe is invalid: {exc}")
        if delta is None:
            conflicts = ", ".join(f"{c.component} {c.path}" for c in merge.patch_conflicts)
            return await self._skip(candidate, head, f"unresolved patch conflict: {conflicts}")
        request = integration_test_request(TestKind.TEST, Purpose.INTEGRATION, head, delta)
        withdrawn: TestWithdrawn | None = None
        try:
            request_id = (await self._tests.run(request)).request_id
        except RequestRejected as exc:
            return await self._skip(candidate, head, f"the toolbox refused the merge test: {exc}")
        except TestWithdrawn as exc:
            request_id, withdrawn = exc.request_id, exc
        await self._notebook.record_merge_attempt(MergeAttempt((candidate,), head, request_id))
        tested = MergeReport(
            candidate, head, MergeOutcome.PENDING, "", request_id, config_conflicts=merge.config_conflicts
        )
        if withdrawn is not None:
            return replace(tested, reason="the merge test was withdrawn")
        return _TestedMerge(tested, delta, await self._gate.compare(request_id))

    async def _skip(self, candidate: str, head: str, reason: str) -> MergeReport:
        await self._notebook.record_merge_attempt(MergeAttempt((candidate,), head, skipped_reason=reason))
        return MergeReport(candidate, head, MergeOutcome.CONFLICT_SKIPPED, reason)

    async def _conclude(self, tested: _TestedMerge, superseded: bool) -> MergeReport:
        """A superseded improvement gets no integration judgement: it stays pending and is tried on the new head."""

        report, judgement = tested.report, tested.judgement
        reason = not_stackable_reason(report.candidate, judgement)
        if reason:
            await self._notebook.write_judgement(str(report.request_id), judgement, Conclusion.NOT_STACKABLE)
            return replace(report, outcome=MergeOutcome.NOT_STACKABLE, reason=reason)
        if superseded:
            return replace(report, reason=SUPERSEDED)
        self._recording = asyncio.ensure_future(self._record_merge(report, tested.delta, judgement))
        merged = await asyncio.shield(self._recording)
        reason = f"improved {judgement.gain_pct:+.2f}% on the integration head"
        return replace(report, outcome=MergeOutcome.MERGED, reason=reason, merged_commit=merged)

    async def _record_merge(self, tested: MergeReport, delta: RecipeDelta, judgement: Judgement) -> str:
        """Commit the merge and record it; runs shielded, so a cancelled merge never moves only one of the two."""

        merged = await self._repo.commit_integration(tested.onto, delta, (tested.candidate,))
        digest = (await self._repo.recipe_at(merged)).digest()
        await self._notebook.write_judgement(str(tested.request_id), judgement, Conclusion.MERGED, merged, digest)
        return merged

    async def abandon(self) -> None:
        """After the round was cancelled: finish recording a merge already committed."""

        if self._recording is not None:
            await self._recording


def _decision(report: MergeReport) -> Decision:
    detail: dict[str, object] = {
        "candidate": report.candidate,
        "onto": report.onto,
        "outcome": report.outcome.value,
        "request_id": report.request_id,
        "merged_commit": report.merged_commit,
        "config_conflicts": [
            {"name": change.name, "before": change.before, "after": change.after} for change in report.config_conflicts
        ],
    }
    return Decision(DecisionKind.MERGE, report.reason, INTEGRATION_BRANCH, detail)


__all__ = ["Integrator", "LlmPatchResolver", "MergeReport", "not_stackable_reason"]
