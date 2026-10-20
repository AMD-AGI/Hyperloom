# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MB1 integrator: carries specialist commits onto the integration branch, in rounds of up to K merge tests.

Candidates are the notebook's committed specialist commits, in the order it gives them (foundation fixes first, then
by their own gain), whatever became of the branch that made them. Each is tried once per integration head. A round
tests up to the integration branch's K candidates at once on the same head; the first of them, in that order, that
may be merged is merged, and every other one that may is superseded: it gets no judgement and is tried again on the
new head. A merge the experiment repository cannot combine, or whose test the toolbox refuses, is skipped and recorded
like an unresolved conflict, and so is one that leaves the head's recipe as it is, untested: another branch's commit
already brought the same change.

The experiment repository carries the candidate's own change onto the integration head and reports what conflicts:
where the two set a server argument or environment variable differently the candidate's value wins, and the
integrator records every overridden value in the decision log; a patch that does not apply where the merge puts it
goes to :class:`LlmPatchResolver`, and a merge it cannot resolve is skipped and recorded. The merged change is tested
from the integration head and judged by the rule gate, and a candidate about to be merged runs the regression set,
before anything is written, so only a candidate that keeps the regression set passing reaches the integration branch
and nothing on it is ever reverted. That candidate is an improvement, or a foundation fix the accuracy gate passed
whatever its gain: a fix the other branches build on is worth merging even when it costs throughput.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from functools import partial

from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import PatchRef, Recipe, RecipeDelta, RecipeError, ValueChange, apply
from hyperloom.measure.compile import BlobError
from hyperloom_multibranch.contracts.gate import NotMeasured
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Notebook
from hyperloom_multibranch.contracts.notebook.records import CommittedCommit, MergeAttempt
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, MergeOutcome
from hyperloom_multibranch.contracts.repo import ExperimentRepo, PatchConflict, RepoMerge
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.rulegate.agentx import P50_BELOW_THRESHOLD
from hyperloom_multibranch.rulegate.inferencex import GAIN_BELOW_THRESHOLD, GAIN_NOT_CONFIRMED
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.scheduling.llm import Completion, CompletionFailed
from hyperloom_multibranch.scheduling.requests import SessionChecks, integration_test_request

logger = logging.getLogger(__name__)

SUPERSEDED = "superseded: another candidate of the round was merged onto this head first"
EXPLORATION_ENDED = "exploration ended"
ALREADY_ON_HEAD = "its change is already on the integration head"
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
    """What came of one candidate on integration head ``onto``; ``config_conflicts`` took the candidate's value.

    A merged candidate carries the gain its merge test was judged to make and the bar it was judged against;
    ``foundation_fix`` says the candidate is a foundation fix, which merges without a gain.
    """

    candidate: str
    onto: str
    outcome: MergeOutcome
    reason: str
    request_id: str | None = None
    merged_commit: str | None = None
    config_conflicts: tuple[ValueChange, ...] = ()
    gain_pct: float | None = None
    threshold_pct: float | None = None
    foundation_fix: bool = False


class LlmPatchResolver:
    """Asks a model to rewrite every patch of a merge that does not apply where the merge puts it.

    A rewritten patch is not applied here: the merge test builds the merged recipe, and the toolbox refuses it when a
    patch still does not apply.
    """

    def __init__(self, completion: Completion, repo: ExperimentRepo) -> None:
        self._completion = completion
        self._repo = repo

    async def resolve(self, merge: RepoMerge, candidate: str) -> RecipeDelta | None:
        """``merge``'s delta with every conflicting patch rewritten and added; ``None`` when one stays unresolved."""

        merged = apply(await self._repo.read_recipe(merge.onto), merge.delta)
        rewritten: dict[str, list[PatchRef]] = {}
        for conflict in merge.patch_conflicts:
            text = await self._rewrite(conflict, merged, candidate, merge.onto)
            if text is None:
                return None
            rewritten.setdefault(conflict.component, []).append(await self._repo.store_patch(conflict.component, text))
        changes = tuple(
            replace(change, add_patches=(*change.add_patches, *rewritten.get(change.component, ())))
            for change in merge.delta.components
        )
        return replace(merge.delta, components=changes)

    async def _rewrite(self, conflict: PatchConflict, merged: Recipe, candidate: str, onto: str) -> str | None:
        prompt = await self._prompt(conflict, merged, candidate, onto)
        try:
            reply = (await self._completion(RESOLVER_SYSTEM_PROMPT, prompt)).text
        except CompletionFailed as exc:
            logger.warning("patch resolver call for %s failed: %s", conflict.path, exc)
            return None
        answer = extract_last_json_with_key(reply, PATCH_KEY)
        rewritten = answer.get(PATCH_KEY) if answer is not None else None
        return rewritten if isinstance(rewritten, str) and rewritten.strip() else None

    async def _prompt(self, conflict: PatchConflict, merged: Recipe, candidate: str, onto: str) -> str:
        """``merged`` is the merge without the patches that conflict: what the rewritten patch has to apply on."""

        pinned = merged.component(conflict.component)
        stack = pinned.patches if pinned is not None else ()
        sections = [
            f"Component: {conflict.component}",
            f"Patch {conflict.path} to rewrite:",
            await self._text(conflict.path, candidate, onto),
            f"Why it does not apply:\n{conflict.detail}",
            f"The patches of {conflict.component} it has to apply on top of, in order:",
            *[f"--- {patch.path} ---\n{await self._text(patch.path, candidate, onto)}" for patch in stack],
        ]
        return "\n\n".join(sections)

    async def _text(self, path: str, candidate: str, onto: str) -> str:
        """A patch of the merge comes from the candidate's recipe or from the integration head's."""

        try:
            return await self._repo.patch_text(candidate, path)
        except KeyError:
            return await self._repo.patch_text(onto, path)


def _untried(commit: CommittedCommit, head: str) -> bool:
    last = commit.last_merge
    return commit.merged_into is None and (last is None or last.integration_head != head)


#: The verdicts a foundation fix that passed the accuracy gate merges on: a settled measurement, whatever its gain.
FOUNDATION_FIX_VERDICTS = frozenset({Verdict.IMPROVED, Verdict.FLAT, Verdict.REGRESSED})
# A settled verdict refused only for want of a gain; any other refusal (a broken guard, extra failed requests, a pair
# that did not measure the same work) is a defect a foundation fix may not carry onto the integration branch.
GAIN_ONLY_REFUSALS = frozenset({GAIN_BELOW_THRESHOLD, GAIN_NOT_CONFIRMED, P50_BELOW_THRESHOLD})


def defects_beyond_gain(judgement: Judgement) -> tuple[str, ...]:
    """The reasons ``judgement`` was refused for other than a missing gain, in the order the rule gate gave them."""

    return tuple(reason for reason in judgement.refused_by if reason not in GAIN_ONLY_REFUSALS)


def not_stackable_reason(judgement: Judgement, *, foundation_fix: bool) -> str:
    """Why a merge test judged ``judgement`` keeps its candidate off the integration branch; empty when it may be
    merged once the regression set passes.

    Only a confirmed gain that passed the accuracy gate is ``IMPROVED``. A foundation fix needs no gain: it may be
    merged on any settled verdict once it passed the accuracy gate, as long as nothing but the missing gain refused it.
    """

    if foundation_fix and judgement.accuracy_ok and judgement.verdict in FOUNDATION_FIX_VERDICTS:
        defects = defects_beyond_gain(judgement)
        if not defects:
            return ""
        return f"a foundation fix, {judgement.verdict.value} on the integration head, refused by {', '.join(defects)}"
    if judgement.verdict is not Verdict.IMPROVED:
        return f"{judgement.verdict.value} {judgement.gain_pct:+.2f}% on the integration head"
    return ""


class _Skipped(Exception):
    """A merge that cannot be tested; ``reason`` says why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _TestedMerge:
    """A merge test the rule gate judged; ``report`` is still ``PENDING`` and ``delta`` is what was tested."""

    report: MergeReport
    delta: RecipeDelta
    judgement: Judgement


@dataclass(frozen=True)
class MergeRecords:
    """Where the integrator records its rounds: its notebook handle, which writes as the integrator, and the merge log."""

    notebook: Notebook
    merges: DecisionLog


class Integrator:
    """Merges committed specialist commits onto the integration branch; the caller runs one round at a time.

    A cancelled round takes its merge tests back; a caller that cancels one then awaits :meth:`abandon`, which takes
    back the retests the gate asked for too and leaves the repository and the notebook agreeing on the integration
    head.
    """

    def __init__(
        self,
        records: MergeRecords,
        repo: ExperimentRepo,
        checks: SessionChecks,
        resolver: LlmPatchResolver,
    ) -> None:
        self._notebook = records.notebook
        self._repo = repo
        self._tests = checks.tests
        self._gate = checks.gate
        self._regression = checks.regression
        self._resolver = resolver
        self._decisions = records.merges
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

    async def _test(self, commit: CommittedCommit, head: str) -> MergeReport | _TestedMerge:
        candidate = commit.commit_id
        try:
            merge = await self._repo.merge(candidate, head)
            delta = await self._resolved(merge, candidate)
            onto = await self._repo.read_recipe(head)
            if apply(onto, delta).digest() == onto.digest():
                return await self._skip(commit, head, ALREADY_ON_HEAD)
            request = integration_test_request(TestKind.TEST, Purpose.INTEGRATION, head, delta)
            return await self._tests.judged(request, partial(self._judge, merge, delta, commit))
        except RecipeError as exc:
            return await self._skip(commit, head, f"the merge does not combine: {exc}")
        except RequestRejected as exc:
            return await self._skip(commit, head, f"the toolbox refused the merge test: {exc}")
        except _Skipped as exc:
            return await self._skip(commit, head, exc.reason)

    async def _judge(
        self, merge: RepoMerge, delta: RecipeDelta, commit: CommittedCommit, request_id: str
    ) -> MergeReport | _TestedMerge:
        candidate = commit.commit_id
        await self._notebook.record_merge_attempt(MergeAttempt((candidate,), merge.onto, request_id))
        tested = MergeReport(
            candidate,
            merge.onto,
            MergeOutcome.PENDING,
            "",
            request_id,
            None,
            merge.config_conflicts,
            foundation_fix=commit.foundation_fix,
        )
        try:
            judgement = await self._gate.compare(request_id)
        except NotMeasured as exc:
            return replace(tested, outcome=MergeOutcome.NOT_STACKABLE, reason=f"the merge test failed: {exc}")
        return _TestedMerge(tested, delta, judgement)

    async def _resolved(self, merge: RepoMerge, candidate: str) -> RecipeDelta:
        """The delta to test; raises :class:`_Skipped` when a conflicting patch stays unresolved."""

        if merge.clean:
            return merge.delta
        conflicts = "; ".join(f"{conflict.path}: {conflict.detail}" for conflict in merge.patch_conflicts)
        try:
            delta = await self._resolver.resolve(merge, candidate)
        except BlobError as exc:
            raise _Skipped(f"unresolved patch conflict: {conflicts}: {exc}") from exc
        if delta is None:
            raise _Skipped(f"unresolved patch conflict: {conflicts}")
        return delta

    async def _skip(self, commit: CommittedCommit, head: str, reason: str) -> MergeReport:
        await self._notebook.record_merge_attempt(MergeAttempt((commit.commit_id,), head, skipped_reason=reason))
        return MergeReport(
            commit.commit_id, head, MergeOutcome.CONFLICT_SKIPPED, reason, foundation_fix=commit.foundation_fix
        )

    async def _conclude(self, tested: _TestedMerge, superseded: bool) -> MergeReport:
        """A superseded candidate gets no integration judgement and runs no regression set: it stays pending and is
        tried on the new head. Any other candidate that may be merged is merged once the regression set passes on the
        merge."""

        report, judgement = tested.report, tested.judgement
        reason = not_stackable_reason(judgement, foundation_fix=report.foundation_fix)
        if not reason and not superseded:
            reason = await self._regression_failure(report, tested.delta)
        if reason:
            await self._notebook.write_judgement(str(report.request_id), judgement, Conclusion.NOT_STACKABLE)
            return replace(report, outcome=MergeOutcome.NOT_STACKABLE, reason=reason)
        if superseded:
            return replace(report, reason=SUPERSEDED)
        self._recording = asyncio.ensure_future(self._record_merge(report, tested.delta, judgement))
        merged = await asyncio.shield(self._recording)
        reason = f"{judgement.verdict.value} {judgement.gain_pct:+.2f}% on the integration head"
        if report.foundation_fix:
            reason = f"a foundation fix, {reason}"
        return replace(
            report,
            outcome=MergeOutcome.MERGED,
            reason=reason,
            merged_commit=merged,
            gain_pct=judgement.gain_pct,
            threshold_pct=judgement.threshold_pct,
        )

    async def _regression_failure(self, report: MergeReport, delta: RecipeDelta) -> str:
        outcome = await self._regression.check(report.onto, delta, sources=(report.candidate,))
        return "" if outcome.passed else f"the regression set failed: {outcome.reason}"

    async def _record_merge(self, tested: MergeReport, delta: RecipeDelta, judgement: Judgement) -> str:
        """Commit the merge and record it; runs shielded, so a cancelled merge never moves only one of the two.

        A commit whose judgement could not be written is rewound: the notebook would not know the commit the
        integration head is at.
        """

        merged = await self._repo.commit_integration(tested.onto, delta, (tested.candidate,))
        written = False
        try:
            digest = (await self._repo.read_recipe(merged)).digest()
            await self._notebook.write_judgement(str(tested.request_id), judgement, Conclusion.MERGED, merged, digest)
            written = True
        finally:
            if not written:
                await self._repo.rewind(INTEGRATION_BRANCH, merged, tested.onto)
        return merged

    async def abandon(self) -> None:
        """After the round was cancelled: take back every integration test still running, the gate's retests
        included, and finish recording a merge already committed."""

        await self._tests.withdraw_integration(EXPLORATION_ENDED)
        if self._recording is not None:
            await self._recording


def _decision(report: MergeReport) -> Decision:
    detail: dict[str, object] = {
        "candidate": report.candidate,
        "onto": report.onto,
        "outcome": report.outcome.value,
        "request_id": report.request_id,
        "merged_commit": report.merged_commit,
        "foundation_fix": report.foundation_fix,
        "gain_pct": report.gain_pct,
        "threshold_pct": report.threshold_pct,
        "config_conflicts": [
            {"name": change.name, "before": change.before, "after": change.after} for change in report.config_conflicts
        ],
    }
    return Decision(DecisionKind.MERGE, report.reason, INTEGRATION_BRANCH, detail)


__all__ = [
    "FOUNDATION_FIX_VERDICTS",
    "GAIN_ONLY_REFUSALS",
    "Integrator",
    "LlmPatchResolver",
    "MergeRecords",
    "MergeReport",
    "defects_beyond_gain",
    "not_stackable_reason",
]
