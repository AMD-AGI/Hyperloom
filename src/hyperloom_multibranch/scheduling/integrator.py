# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MB1 integrator: carries specialist commits onto the integration branch, in rounds of up to K merge tests.

Candidates are the notebook's committed specialist commits, in the order it gives them (foundation fixes first, then
by their own gain), whatever became of the branch that made them. Each is tried once per integration head. A round
tests up to the integration branch's K candidates at once on the same head; the first of them, in that order, that
improves it is merged, and every other improvement is superseded: it gets no judgement and is tried again on the new
head. A merge whose recipe the recipe contract refuses, or whose test the toolbox refuses, is skipped and recorded
like an unresolved conflict.

A merge carries the change the candidate's committing test measured onto the integration head's recipe. Where the two
set a server argument or environment variable differently the candidate's value wins, and the integrator records every
overridden value in the decision log; when the candidate moves a component to another upstream ref, the head's patches
of it stay. A patch that does not apply there is found when the toolbox refuses the merge test; the patches the
candidate adds then go to :class:`LlmPatchResolver`, and a merge it cannot resolve is skipped and recorded. The merged
change is tested, accuracy included, from the integration head before anything is written, so only an improvement
reaches the integration branch and nothing on it is ever reverted.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace

from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import PatchRef, Recipe, RecipeDelta, RecipeError, ValueChange, apply, diff
from hyperloom.measure.compile import BlobError
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Notebook
from hyperloom_multibranch.contracts.notebook.records import CommittedCommit, MergeAttempt, TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, MergeOutcome
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.llm import Completion, CompletionFailed
from hyperloom_multibranch.scheduling.mb_fakes import mb_fake_regression_passes
from hyperloom_multibranch.scheduling.patches import PatchStore
from hyperloom_multibranch.scheduling.requests import SessionTests, TestWithdrawn, integration_test_request
from hyperloom_multibranch.scheduling.rule_judge import MbFakeRuleJudge, NotMeasured

logger = logging.getLogger(__name__)

SUPERSEDED = "superseded: another candidate of the round was merged onto this head first"
#: The toolbox's refusal code for a merged change whose patch does not apply; it sends the merge to the resolver.
PATCH_NOT_APPLICABLE = "delta_not_applicable"
#: The JSON key the resolver's reply carries the rewritten patch under; ``null`` means it cannot be combined.
PATCH_KEY = "patch"
RESOLVER_SYSTEM_PROMPT = (
    "You resolve patch conflicts while merging two optimizations of an LLM inference server. "
    "A patch of a component did not apply on top of the other patches of that component. Rewrite it so it applies "
    "on top of them and keeps its intent, without undoing what the other patches do. "
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
    """Asks a model to rewrite every patch a merged change adds so it applies on the integration head's patches.

    A rewritten patch is not applied here: the merge test builds the merged recipe, and the toolbox refuses it again
    when a patch still does not apply.
    """

    def __init__(self, completion: Completion, patches: PatchStore) -> None:
        self._completion = completion
        self._patches = patches

    async def resolve(self, delta: RecipeDelta, head: Recipe, refusal: str) -> RecipeDelta | None:
        """``delta`` with each patch it adds rewritten for ``head``; ``None`` when it adds none or one stays
        unresolved. ``refusal`` is what the toolbox said when it refused the merge."""

        if not any(change.add_patches for change in delta.components):
            return None
        changes = []
        for change in delta.components:
            rewritten = []
            for patch in change.add_patches:
                text = await self._rewrite(change.component, patch, head, refusal)
                if text is None:
                    return None
                rewritten.append(await asyncio.to_thread(self._patches.store, change.component, text))
            changes.append(replace(change, add_patches=tuple(rewritten)))
        return replace(delta, components=tuple(changes))

    async def _rewrite(self, component: str, patch: PatchRef, head: Recipe, refusal: str) -> str | None:
        prompt = await asyncio.to_thread(self._prompt, component, patch, head, refusal)
        try:
            reply = await self._completion(RESOLVER_SYSTEM_PROMPT, prompt)
        except CompletionFailed as exc:
            logger.warning("patch resolver call for %s failed: %s", patch.path, exc)
            return None
        answer = extract_last_json_with_key(reply, PATCH_KEY)
        rewritten = answer.get(PATCH_KEY) if answer is not None else None
        return rewritten if isinstance(rewritten, str) and rewritten.strip() else None

    def _prompt(self, component: str, patch: PatchRef, head: Recipe, refusal: str) -> str:
        pinned = head.component(component)
        stack = pinned.patches if pinned is not None else ()
        sections = [
            f"Component: {component}",
            f"Patch {patch.path} to rewrite:",
            self._patches.text(patch),
            f"Why the merge was refused:\n{refusal}",
            f"The patches of {component} it has to apply on top of, in order:",
            *(f"--- {applied.path} ---\n{self._patches.text(applied)}" for applied in stack),
        ]
        return "\n\n".join(sections)


def _untried(commit: CommittedCommit, head: str) -> bool:
    last = commit.last_merge
    return commit.merged_into is None and (last is None or last.integration_head != head)


def not_stackable_reason(candidate: str, judgement: Judgement) -> str:
    """Why the merge test of ``candidate`` keeps it off the integration branch; empty when it may be merged.

    Only a confirmed gain that passed the accuracy gate is ``IMPROVED``.
    """

    if judgement.verdict is not Verdict.IMPROVED:
        return f"{judgement.verdict.value} {judgement.gain_pct:+.2f}% on the integration head"
    if not mb_fake_regression_passes(candidate):
        return "the regression set failed"
    return ""


@dataclass(frozen=True)
class _Merge:
    """A candidate's change carried onto integration head ``onto``: the ``delta`` tested and the ``recipe`` written."""

    candidate: str
    onto: str
    head: Recipe
    delta: RecipeDelta
    recipe: Recipe
    config_conflicts: tuple[ValueChange, ...]


def _merged(candidate: str, onto: str, head: Recipe, delta: RecipeDelta) -> _Merge:
    """Raises :class:`RecipeError` when the change does not fit the head's recipe."""

    recipe = apply(head, delta)
    return _Merge(candidate, onto, head, delta, recipe, diff(head, recipe).config_conflicts)


class _Skipped(Exception):
    """A merge that cannot be tested; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Measured:
    """The merge test of ``merge``: its settled record, or its withdrawal."""

    merge: _Merge
    outcome: TestRecord | TestWithdrawn


@dataclass(frozen=True)
class _TestedMerge:
    """A merge test that produced a judgement; ``report`` is still ``PENDING``."""

    report: MergeReport
    recipe: Recipe
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
        judge: MbFakeRuleJudge,
        resolver: LlmPatchResolver,
        decisions: DecisionLog,
    ) -> None:
        self._notebook = notebook
        self._repo = repo
        self._tests = tests
        self._judge = judge
        self._resolver = resolver
        self._decisions = decisions
        self._recording: asyncio.Future[str] | None = None

    async def integrate_next(self, width: int) -> tuple[MergeReport, ...]:
        """Test up to ``width`` commits not merged and not yet tried on the current head, at once; ``()`` when none is.

        At most one of them is merged, so the head moves once per round.
        """

        head = await self._repo.head(INTEGRATION_BRANCH)
        untried = [commit for commit in await self._notebook.committed() if _untried(commit, head)]
        tested = await asyncio.gather(*(self._test(candidate, head) for candidate in untried[:width]))
        reports = []
        merged = False
        for item in tested:
            report = item if isinstance(item, MergeReport) else await self._conclude(item, merged)
            merged = merged or report.outcome is MergeOutcome.MERGED
            self._decisions.record(_decision(report))
            reports.append(report)
        return tuple(reports)

    async def _test(self, candidate: CommittedCommit, head: str) -> MergeReport | _TestedMerge:
        change = (await self._notebook.record(candidate.request_id)).request.delta
        try:
            merge = _merged(candidate.commit_id, head, await self._repo.read_recipe(head), change)
            measured = await self._measure(merge)
        except RecipeError as exc:
            return await self._skip(candidate.commit_id, head, f"the merged recipe is invalid: {exc}")
        except _Skipped as exc:
            return await self._skip(candidate.commit_id, head, exc.reason)
        outcome = measured.outcome
        await self._notebook.record_merge_attempt(MergeAttempt((candidate.commit_id,), head, outcome.request_id))
        conflicts = measured.merge.config_conflicts
        tested = MergeReport(candidate.commit_id, head, MergeOutcome.PENDING, "", outcome.request_id, None, conflicts)
        if isinstance(outcome, TestWithdrawn):
            return replace(tested, reason="the merge test was withdrawn")
        try:
            return _TestedMerge(tested, measured.merge.recipe, await self._judge.judge(outcome))
        except TestWithdrawn:
            return replace(tested, reason="the confirming retest was withdrawn")
        except NotMeasured as exc:
            return replace(tested, outcome=MergeOutcome.NOT_STACKABLE, reason=f"the merge test failed: {exc}")

    async def _measure(self, merge: _Merge) -> _Measured:
        """The merge test's outcome, after the patches the toolbox refused were resolved; raises :class:`_Skipped`
        when the merge cannot be tested."""

        try:
            return _Measured(merge, await self._run(merge))
        except RequestRejected as exc:
            if exc.code != PATCH_NOT_APPLICABLE:
                raise _Skipped(f"the toolbox refused the merge test: {exc}") from exc
            refusal = exc.reason
        try:
            delta = await self._resolver.resolve(merge.delta, merge.head, refusal)
        except BlobError as exc:
            raise _Skipped(f"unresolved patch conflict: {refusal}: {exc}") from exc
        if delta is None:
            raise _Skipped(f"unresolved patch conflict: {refusal}")
        try:
            resolved = _merged(merge.candidate, merge.onto, merge.head, delta)
            return _Measured(resolved, await self._run(resolved))
        except RecipeError as exc:
            raise _Skipped(f"the resolved merge is invalid: {exc}") from exc
        except RequestRejected as exc:
            raise _Skipped(f"the toolbox refused the resolved merge test: {exc}") from exc

    async def _run(self, merge: _Merge) -> TestRecord | TestWithdrawn:
        request = integration_test_request(TestKind.TEST, Purpose.INTEGRATION, merge.onto, merge.delta, accuracy=True)
        try:
            return await self._tests.run(request)
        except TestWithdrawn as exc:
            return exc

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
        self._recording = asyncio.ensure_future(self._record_merge(report, tested.recipe, judgement))
        merged = await asyncio.shield(self._recording)
        reason = f"improved {judgement.gain_pct:+.2f}% on the integration head"
        return replace(report, outcome=MergeOutcome.MERGED, reason=reason, merged_commit=merged)

    async def _record_merge(self, tested: MergeReport, recipe: Recipe, judgement: Judgement) -> str:
        """Commit the merge and record it; runs shielded, so a cancelled merge never moves only one of the two.

        A commit whose judgement could not be written is rewound: the notebook would not know the commit the
        integration head is at.
        """

        message = f"merge {tested.candidate} onto {tested.onto}: {tested.request_id} {judgement.gain_pct:+.2f}%"
        merged = await self._repo.commit(INTEGRATION_BRANCH, tested.onto, recipe, message)
        written = False
        try:
            await self._notebook.write_judgement(
                str(tested.request_id), judgement, Conclusion.MERGED, merged, recipe.digest()
            )
            written = True
        finally:
            if not written:
                await self._repo.rewind(INTEGRATION_BRANCH, merged, tested.onto)
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
