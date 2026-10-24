# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MB1 integrator: carries specialist commits onto the integration branch, in rounds of up to K merge tests.

Candidates are the notebook's committed specialist commits, in the order it gives them (foundation fixes first, then
by their own gain), whatever became of the branch that made them. Each is tried once per integration head, until it
is merged or found already on the head, except that one the rule gate measured as not stackable waits until a later
merge changes something its own change names: see :mod:`hyperloom_multibranch.scheduling.candidates`. A round
tests up to the integration branch's K candidates at once on the same head, and a slot it had nothing to fill takes
the first commit committed while it runs rather than standing idle until the round ends; the round ends its tests in
the order it started them, and the first of them that
may be merged is merged, and every other one that may is superseded: it gets no judgement and is tried again on the
new head. So is a merge whose test did not measure, and one whose regression set did not run clean: the set says a
case did not come back clean, not whether it reproduced or never ran, so it settles nothing about the candidate. A
merge the experiment repository cannot combine, or whose test the toolbox refuses, is skipped and recorded
like an unresolved conflict. One that leaves the head's recipe as it is is skipped untested as already on the head:
another branch's commit already brought the same change, and the head keeps it, so it is not tried again, as a merged
commit is not.

The experiment repository carries the candidate's own change onto the integration head and reports what conflicts:
where the two set a server argument or environment variable differently the candidate's value wins, and the
integrator records every overridden value in the decision log; where both tuned one kernel slot the merge unites the
two tables, and that is recorded there too, since the merged recipe then loads a table neither branch measured and
the merge test's gain is measured on it; a patch that does not apply where the merge puts it
goes to :class:`LlmPatchResolver`, and a merge it cannot resolve is skipped and recorded. The merged change is tested
from the integration head and judged by the rule gate, and a candidate about to be merged runs the regression set,
before anything is written, so only a candidate that keeps the regression set passing reaches the integration branch
and nothing on it is ever reverted. That candidate is an improvement, or a foundation fix the accuracy gate passed
whatever its gain: a fix the other branches build on is worth merging even when it costs throughput.

Before EXPLORE opens, :meth:`Integrator.replay_champion` carries the Recipe KB champion PRELUDE prepared onto the
root the same way and by the same rule. It is nobody's candidate, so no merge attempt records it and the regression
set it would run is still empty; what it leaves the integration branch at is the commit EXPLORE opens on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence, Set
from dataclasses import dataclass, field, replace
from functools import partial

from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.contracts.recipe import ComponentPatches, PatchRef, RecipeDelta, RecipeError, ValueChange, apply
from hyperloom.measure.compile import BlobError
from hyperloom_multibranch.contracts.gate import NotMeasured
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Notebook
from hyperloom_multibranch.contracts.notebook.records import CommittedCommit, MergeAttempt, WarmReplay
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, MergeOutcome
from hyperloom_multibranch.contracts.repo import ExperimentRepo, PatchConflict, RepoMerge, UnitedTable
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook.knowledge.warm_start import PreparedReplay
from hyperloom_multibranch.rulegate.agentx import P50_BELOW_THRESHOLD
from hyperloom_multibranch.rulegate.inferencex import GAIN_BELOW_THRESHOLD, GAIN_NOT_CONFIRMED
from hyperloom_multibranch.scheduling.candidates import Candidates
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import Decision, DecisionKind, DecisionLog
from hyperloom_multibranch.contracts.llm import Completion, CompletionFailed
from hyperloom_multibranch.scheduling.requests import SessionChecks, integration_test_request

SUPERSEDED = "superseded: another candidate of the round was merged onto this head first"
EXPLORATION_ENDED = "exploration ended"
ALREADY_ON_HEAD = "its change is already on the integration head"
#: A patch the three-way rebase could not settle and the resolver's answer did not resolve either.
CONFLICT_UNRESOLVED = "unresolved patch conflict"
#: The same conflict, lost before any answer: the resolver could not be reached. Counted apart from
#: :data:`CONFLICT_UNRESOLVED`, because a timed-out call says nothing about whether the patches can be combined.
RESOLVER_UNREACHED = "the patch resolver could not be asked"
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

    ``united_tables`` are the kernel slots the merge filled with a union of both sides' tables: the tested recipe
    loads a table neither branch measured, and the gain below was measured on it.

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
    united_tables: tuple[UnitedTable, ...] = ()
    gain_pct: float | None = None
    threshold_pct: float | None = None
    foundation_fix: bool = False


class PatchUnresolved(Exception):
    """A conflicting patch the resolver got no rewrite of that applies; the message says why."""


@dataclass(frozen=True)
class _Sources:
    """Where the text of a patch of the merge is read: a rewrite this resolution wrote, else the candidate's recipe,
    else the integration head's."""

    candidate: str
    onto: str
    written: dict[str, str] = field(default_factory=dict)


class LlmPatchResolver:
    """Asks a model to rewrite every patch of a merge that does not apply where the merge puts it.

    A rewrite is applied on top of the patches of its component it has to apply on before it is kept, so one that does
    not apply costs no merge test; the toolbox still builds the merged recipe once the merge is tested.
    """

    def __init__(self, completion: Completion, repo: ExperimentRepo) -> None:
        self._completion = completion
        self._repo = repo

    async def resolve(self, merge: RepoMerge, candidate: str) -> RecipeDelta:
        """``merge``'s delta with every conflicting patch rewritten and added; :class:`PatchUnresolved` when one stays
        unresolved. Each rewrite applies on top of the rewrites before it.

        A call that fails raises :class:`CompletionFailed`: a model that could not be asked is a different loss from
        a model that answered, and the merge log names them apart.
        """

        merged = apply(await self._repo.read_recipe(merge.onto), merge.delta)
        stacks = {pinned.component: pinned for pinned in merged.components}
        sources = _Sources(candidate, merge.onto)
        rewritten: dict[str, list[PatchRef]] = {}
        for conflict in merge.patch_conflicts:
            stack = stacks[conflict.component]
            text = await self._rewrite(conflict, stack, await self._prompt(conflict, stack, sources))
            patch = await self._repo.store_patch(conflict.component, text)
            sources.written[patch.path] = text
            stacks[conflict.component] = replace(stack, patches=(*stack.patches, patch))
            rewritten.setdefault(conflict.component, []).append(patch)
        changes = tuple(
            replace(change, add_patches=(*change.add_patches, *rewritten.get(change.component, ())))
            for change in merge.delta.components
        )
        return replace(merge.delta, components=changes)

    async def _rewrite(self, conflict: PatchConflict, stack: ComponentPatches, prompt: str) -> str:
        """The model's rewrite of ``conflict``, once it applies on top of ``stack``."""

        reply = (await self._completion(RESOLVER_SYSTEM_PROMPT, prompt)).text
        answer = extract_last_json_with_key(reply, PATCH_KEY)
        text = answer.get(PATCH_KEY) if answer is not None else None
        if not isinstance(text, str) or not text.strip():
            raise PatchUnresolved(f"the model gave no rewrite of {conflict.path}")
        error = await self._repo.apply_error(stack, text)
        if error is not None:
            raise PatchUnresolved(f"the rewrite of {conflict.path} does not apply: {error}")
        return text

    async def _prompt(self, conflict: PatchConflict, stack: ComponentPatches, sources: _Sources) -> str:
        """``stack`` is what the rewritten patch has to apply on: the merge without the patches that conflict."""

        sections = [
            f"Component: {conflict.component}",
            f"Patch {conflict.path} to rewrite:",
            await self._text(conflict.path, sources),
            f"Why it does not apply:\n{conflict.detail}",
            f"The patches of {conflict.component} it has to apply on top of, in order:",
            *[f"--- {patch.path} ---\n{await self._text(patch.path, sources)}" for patch in stack.patches],
        ]
        return "\n\n".join(sections)

    async def _text(self, path: str, sources: _Sources) -> str:
        if path in sources.written:
            return sources.written[path]
        try:
            return await self._repo.patch_text(sources.candidate, path)
        except KeyError:
            return await self._repo.patch_text(sources.onto, path)


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
        return _verdict_on_head(judgement)
    return ""


def _verdict_on_head(judgement: Judgement) -> str:
    """How a merge test was judged, as the merge log names it."""

    return f"{judgement.verdict.value} {judgement.gain_pct:+.2f}% on the integration head"


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


#: What a merge test ends as: a report of its own, or a judged merge the round has still to conclude.
_Tested = MergeReport | _TestedMerge


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
        clock: Clock,
    ) -> None:
        self._clock = clock
        self._notebook = records.notebook
        self._repo = repo
        self._candidates = Candidates(records.notebook, repo)
        self._tests = checks.tests
        self._gate = checks.gate
        self._regression = checks.regression
        self._resolver = resolver
        self._decisions = records.merges
        self._recording: asyncio.Future[str] | None = None

    async def integrate_next(self, width: int, tick: float) -> tuple[MergeReport, ...]:
        """Test the commits offered on the current head, ``width`` of them at once; ``()`` when none is offered.

        A slot the round has nothing to fill takes a commit committed while it runs, looked for every ``tick`` clock
        seconds, instead of standing idle until the round ends. The head stays where the round found it, so that
        commit is tested on the same head as the rest. At most one commit is merged, and once one is the round starts
        no more: the head has moved under the rest, which are offered again on the head it moved to.

        A test that ends waits for the tests the round started before it, so which commit a round merges follows the
        order the notebook offers them in rather than how long each one took to measure.
        """

        head = await self._repo.head(INTEGRATION_BRANCH)
        running: list[asyncio.Task[_Tested]] = []
        started: set[str] = set()
        reports: list[MergeReport] = []
        merged = False
        try:
            while True:
                for commit in await self._to_test(head, 0 if merged else width - len(started), started):
                    started.add(commit.commit_id)
                    running.append(asyncio.create_task(self._test(commit, head)))
                while running and running[0].done():
                    item = running.pop(0).result()
                    report = item if isinstance(item, MergeReport) else await self._conclude(item, merged)
                    merged = merged or report.outcome is MergeOutcome.MERGED
                    self._decisions.record(_decision(report))
                    reports.append(report)
                if not running:
                    return tuple(reports)
                await self._until_one_ends([task for task in running if not task.done()], tick)
        finally:
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    async def _until_one_ends(self, running: Sequence[asyncio.Task[_Tested]], tick: float) -> None:
        """Wait until one of ``running`` has ended, or ``tick`` clock seconds have passed, whichever comes first."""

        timer = asyncio.ensure_future(self._clock.sleep(tick))
        try:
            await asyncio.wait([*running, timer], return_when=asyncio.FIRST_COMPLETED)
        finally:
            timer.cancel()

    async def _to_test(self, head: str, free: int, started: Set[str]) -> tuple[CommittedCommit, ...]:
        """The commits to start now on ``head``: those it offers, less the ones this round has already started.

        A test records its merge attempt once the toolbox has taken it, which is what stops the notebook offering it
        again; until then it still reads as offered, so the round remembers what it started rather than racing that
        record.
        """

        if free <= 0:
            return ()
        offered = await self._candidates.offered_on(head)
        return tuple(commit for commit in offered if commit.commit_id not in started)[:free]

    async def _test(self, commit: CommittedCommit, head: str) -> _Tested:
        candidate = commit.commit_id
        try:
            merge = await self._repo.merge(candidate, head, await self._foundation_fixes())
            delta = await self._resolved(merge, candidate)
            onto = await self._repo.read_recipe(head)
            if apply(onto, delta).digest() == onto.digest():
                return await self._skip(
                    commit, head, ALREADY_ON_HEAD, MergeOutcome.ALREADY_ON_HEAD, merge.united_tables
                )
            request = integration_test_request(TestKind.TEST, Purpose.INTEGRATION, head, delta)
            return await self._tests.judged(request, partial(self._judge, merge, delta, commit))
        except RecipeError as exc:
            return await self._skip(commit, head, f"the merge does not combine: {exc}", MergeOutcome.CONFLICT_SKIPPED)
        except RequestRejected as exc:
            return await self._skip(
                commit, head, f"the toolbox refused the merge test: {exc}", MergeOutcome.CONFLICT_SKIPPED
            )
        except _Skipped as exc:
            return await self._skip(commit, head, exc.reason, MergeOutcome.CONFLICT_SKIPPED)

    async def _foundation_fixes(self) -> frozenset[str]:
        """The foundation fixes merged onto the integration branch, as the notebook records them: no later merge may
        replace a patch they brought, which other branches build on."""

        committed = await self._notebook.committed()
        return frozenset(
            commit.commit_id for commit in committed if commit.foundation_fix and commit.merged_into is not None
        )

    async def _judge(self, merge: RepoMerge, delta: RecipeDelta, commit: CommittedCommit, request_id: str) -> _Tested:
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
            merge.united_tables,
            foundation_fix=commit.foundation_fix,
        )
        try:
            judgement = await self._gate.compare(request_id)
        except NotMeasured as exc:
            return replace(tested, reason=f"the merge test failed: {exc}")
        return _TestedMerge(tested, delta, judgement)

    async def _resolved(self, merge: RepoMerge, candidate: str) -> RecipeDelta:
        """The delta to test; raises :class:`_Skipped` when a conflicting patch stays unresolved."""

        if merge.clean:
            return merge.delta
        conflicts = "; ".join(f"{conflict.path}: {conflict.detail}" for conflict in merge.patch_conflicts)
        try:
            return await self._resolver.resolve(merge, candidate)
        except CompletionFailed as exc:
            raise _Skipped(f"{RESOLVER_UNREACHED}: {conflicts}: {exc}") from exc
        except (BlobError, PatchUnresolved) as exc:
            raise _Skipped(f"{CONFLICT_UNRESOLVED}: {conflicts}: {exc}") from exc

    async def _skip(
        self,
        commit: CommittedCommit,
        head: str,
        reason: str,
        outcome: MergeOutcome,
        united: tuple[UnitedTable, ...] = (),
    ) -> MergeReport:
        """A skipped candidate's report. ``united`` is what the merge united before the skip: a candidate whose every
        tuned row the head already had reads as already on the head, which is true of the slot and says nothing about
        the rows it cost, so the union is recorded even where nothing was tested."""

        attempt = MergeAttempt((commit.commit_id,), head, skipped_reason=reason, skipped_outcome=outcome)
        await self._notebook.record_merge_attempt(attempt)
        return MergeReport(
            commit.commit_id, head, outcome, reason, united_tables=united, foundation_fix=commit.foundation_fix
        )

    async def _conclude(self, tested: _TestedMerge, superseded: bool) -> MergeReport:
        """A superseded candidate gets no integration judgement and runs no regression set: it stays pending and is
        tried on the new head, as does one whose regression set did not run clean. Any other candidate that may be
        merged is merged once the regression set passes on the merge."""

        report, judgement = tested.report, tested.judgement
        reason = not_stackable_reason(judgement, foundation_fix=report.foundation_fix)
        if reason:
            await self._notebook.write_judgement(str(report.request_id), judgement, Conclusion.NOT_STACKABLE)
            return replace(report, outcome=MergeOutcome.NOT_STACKABLE, reason=reason)
        if superseded:
            return replace(report, reason=SUPERSEDED)
        failed = await self._regression_failure(report, tested.delta)
        if failed:
            return replace(report, reason=failed)
        self._recording = asyncio.ensure_future(self._record_merge(report, tested.delta, judgement))
        merged = await asyncio.shield(self._recording)
        reason = _verdict_on_head(judgement)
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

    async def replay_champion(self, champion: PreparedReplay, root: str) -> str:
        """Put the Recipe KB's ``champion`` on the integration branch at ``root``; the commit EXPLORE opens on.

        The champion is judged by the rule every candidate is judged by, so EXPLORE opens on it only where it
        improved on the baseline ``root`` just anchored, and every later gain is still measured from ``root``. It is
        no specialist's commit: nothing offered it, so no merge attempt records it, and the regression set it would
        run is empty before EXPLORE. A champion whose recipe ``root`` carries already, whose test did not measure,
        that the toolbox refuses or that the rule gate does not stack leaves the head at ``root``: the session
        explores from the baseline it anchored rather than ending before EXPLORE for a commit nothing of it needs.
        """

        replay = champion.on(await self._repo.read_recipe(root), root)
        if replay is None:
            return self._record_champion(root, champion.canonical_id, ALREADY_ON_HEAD)
        request = integration_test_request(TestKind.TEST, Purpose.INTEGRATION, root, replay.delta)
        try:
            return await self._tests.judged(request, partial(self._stack_champion, replay))
        except RequestRejected as exc:
            return self._record_champion(root, champion.canonical_id, f"the toolbox refused the replay: {exc.reason}")

    async def _stack_champion(self, replay: WarmReplay, request_id: str) -> str:
        root = replay.parent_commit
        name = replay.canonical_id
        try:
            judgement = await self._gate.compare(request_id)
        except NotMeasured as exc:
            return self._record_champion(root, name, f"the replay did not measure: {exc}", request_id=request_id)
        reason = not_stackable_reason(judgement, foundation_fix=False)
        if reason:
            await self._notebook.write_judgement(request_id, judgement, Conclusion.NOT_STACKABLE)
            return self._record_champion(root, name, reason, request_id=request_id)
        recipe = apply(await self._repo.read_recipe(root), replay.delta)
        landed = await self._repo.commit(INTEGRATION_BRANCH, root, recipe, f"replay the Recipe KB champion {name}")
        await self._notebook.write_judgement(request_id, judgement, Conclusion.MERGED, landed, recipe.digest())
        return self._record_champion(root, name, _verdict_on_head(judgement), request_id=request_id, landed=landed)

    def _record_champion(
        self, onto: str, canonical_id: str, reason: str, *, request_id: str = "", landed: str = ""
    ) -> str:
        """Record what came of the champion in the merge log; the head it leaves the integration branch at."""

        detail = {"champion": canonical_id, "onto": onto, "request_id": request_id, "merged_commit": landed}
        self._decisions.record(Decision(DecisionKind.MERGE, reason, INTEGRATION_BRANCH, detail))
        return landed or onto

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
        "united_tables": [
            {
                "kind": united.table.kind.value,
                "target": united.table.target,
                "theirs": united.theirs.sha256,
                "ours": united.ours.sha256,
                "table": united.table.sha256,
                "kept_head_shapes": [dict(shape) for shape in united.kept_head_shapes],
            }
            for united in report.united_tables
        ],
    }
    return Decision(DecisionKind.MERGE, report.reason, INTEGRATION_BRANCH, detail)


__all__ = [
    "CONFLICT_UNRESOLVED",
    "FOUNDATION_FIX_VERDICTS",
    "GAIN_ONLY_REFUSALS",
    "RESOLVER_UNREACHED",
    "Integrator",
    "LlmPatchResolver",
    "MergeRecords",
    "MergeReport",
    "PatchUnresolved",
    "defects_beyond_gain",
    "not_stackable_reason",
]
