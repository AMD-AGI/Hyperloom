# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The full startup brief: what a new specialist reads before its first action.

:class:`CrossSessionBriefing` is the second ``StartupBriefProvider``; the local notebook's own brief stays the fake.
Its brief has three parts:

* tried: what specialists tried on the start commit and on every state it builds on, from the notebook;
* lessons: this session's repeated failures in one direction on those states (from the lineage analysis), then what
  this session's specialists learnt about ``focus``, then Experiences recalled from other sessions, labelled
  ``prior`` when measured under this session's conditions and ``to re-verify`` otherwise, then what specialists of
  past sessions learnt about it, then the pitfalls and failures the warm-start records list;
* leads: the Recipe KB champions the warm start read, and recalled priors with a positive gain, taking turns by the
  layer they change (configuration, source), so that one kind of change does not fill them all. The Experience KB's
  schema has no kernel family, so no recalled lead is a kernel change.

A champion the start commit is already, or builds on, is no lead: the lessons name it first, as :data:`ON_HEAD`, and
``notebook_replay`` refuses to replay it, since its change would undo what the head did since.

A peer or warm replay that would swap the head's kernel table out stacks the two tables when ``tables`` is set: the
wider of them, or a union of their rows, using the same primitives a merge uses (#109 item 7). When neither fits, the
swap stays and the head's artifact is named in ``replaces``.

At the root before anything was tried, the brief is the first batch's: cross-session content only. Lessons and leads
together stay under ``max_chars``, the most specific first. Where the brief sits in the prompt and how long it may be
is xiaofei's call, so the cap is a parameter.

:class:`SessionBriefs` is the session's one source of briefs, so a branch's brief can be rebuilt at every launch of its
specialist, on its head then: the Experience KB is read once per focus and branch, and only the notebook's part is read
again. It keeps the focus each branch was given.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from typing import Protocol

from hyperloom.contracts.recipe import Recipe, RecipeDelta, RecipeError, apply
from hyperloom_multibranch.contracts.notebook.interfaces import Notebook, StartupBriefProvider
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import (
    LearningRecord,
    NotebookSnapshot,
    PeerReplay,
    StartupBrief,
    WarmReplay,
)
from hyperloom_multibranch.notebook.knowledge.learning_recall import LearningRecall, PastLearnings
from hyperloom_multibranch.notebook.knowledge.replay_tables import TableStack, stack_added_kernels
from hyperloom_multibranch.notebook.knowledge.lessons import MAX_ITEMS, clipped, listed
from hyperloom_multibranch.notebook.knowledge.recall import (
    ExperienceReader,
    IdentityIndex,
    Recall,
    RecalledExperience,
    RecallQuery,
    RecallTier,
    recall_experiences,
)
from hyperloom_multibranch.notebook.knowledge.warm_start import PreparedReplay, WarmLead, delta_layer
from hyperloom_multibranch.notebook.learning_search import (
    Retrieval,
    RetrievalSource,
    ShownLead,
    render_session_learning,
    session_matches,
)
from hyperloom_multibranch.notebook.rsi.analysis import FailureTier, failures_at, repeated_failures
from hyperloom_multibranch.notebook.tools import render_brief

log = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 4000
KEPT = "keep"
#: How many learnings of this session, and how many of past sessions, a brief lists at most.
LEARNINGS = 5
#: What a brief says, before the champion, of one its start commit is already or builds on.
ON_HEAD = "already on your head, nothing to replay"


class RecipeReads(Protocol):
    """What a replay reads of the experiment repository, as ``contracts.repo.ExperimentRepo`` gives it."""

    async def head(self, branch: str) -> str: ...

    async def read_recipe(self, commit: str) -> Recipe: ...


@dataclass(frozen=True)
class _Item:
    lesson: bool
    text: str
    lead: ShownLead | None = None


@dataclass(frozen=True)
class _Recalled:
    """Recalled priors with a positive gain (leads), and the other priors and those to re-verify (lessons)."""

    leads: tuple[_Item, ...] = ()
    lessons: tuple[str, ...] = ()


def _by_layer(items: Sequence[RecalledExperience]) -> list[RecalledExperience]:
    """``items`` taking turns by the layer they change, each layer in its own order, the first layer seen first."""

    layers: dict[IntentLayer | None, list[RecalledExperience]] = {}
    for item in items:
        layers.setdefault(item.layer, []).append(item)
    return [item for turn in zip_longest(*layers.values()) for item in turn if item is not None]


def _lead(item: RecalledExperience) -> _Item:
    layer = f" ({item.layer.value})" if item.layer is not None else ""
    return _Item(False, f"prior{layer}: {item.describe()}", ShownLead(item.experience_id, item.change_summary))


def _recipe(lead: WarmLead) -> _Item:
    envs = (f"{name}={value}" for name, value in sorted(lead.envs.items()))
    change = " ".join(part for part in (lead.server_args, *envs) if part)
    return _Item(False, f"recipe: {lead.describe()}", ShownLead(lead.canonical_id, change))


def _listed(items: Iterable[_Item]) -> list[_Item]:
    """``items`` as :func:`listed` keeps texts: without repeats, each clipped, at most :data:`MAX_ITEMS` of them."""

    kept: dict[str, _Item] = {}
    for item in items:
        kept.setdefault(clipped(item.text), replace(item, text=clipped(item.text)))
    return list(kept.values())[:MAX_ITEMS]


def _recalled(recall: Recall | None) -> _Recalled:
    if recall is None or not recall.available:
        return _Recalled()
    priors = recall.of(RecallTier.PRIOR)
    positive = [item for item in priors if (item.gain_pct or 0.0) > 0 and item.decision == KEPT]
    return _Recalled(
        leads=tuple(_lead(item) for item in _by_layer(positive)),
        lessons=(
            *(f"prior: {item.describe()}" for item in priors if item not in positive),
            *(f"to re-verify: {item.describe()}" for item in recall.of(RecallTier.REVERIFY)),
        ),
    )


def _fit(items: Iterable[_Item], max_chars: int) -> list[_Item]:
    """Lessons and leads, in priority order, up to the first that would pass ``max_chars``."""

    kept: list[_Item] = []
    used = 0
    for item in items:
        if used + len(item.text) > max_chars:
            break
        kept.append(item)
        used += len(item.text)
    return kept


class RecallCache:
    """The completed Experience KB reads of one session, by focus, branch and baseline; a read that did not complete
    is not kept, so the next brief tries again."""

    def __init__(self) -> None:
        self._reads: dict[tuple[str, str, float | None], Recall] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str, float | None]) -> Recall | None:
        with self._lock:
            return self._reads.get(key)

    def keep(self, key: tuple[str, str, float | None], recall: Recall) -> None:
        if recall.available:
            with self._lock:
                self._reads[key] = recall


@dataclass(frozen=True)
class CrossSessionBriefing(StartupBriefProvider):
    """A startup brief with this session's lessons and what other sessions left.

    ``warm`` are the leads PRELUDE read; ``experiences`` the Experience KB, read for ``focus`` (the bottleneck the
    specialist attacks) on ``branch``; ``metric`` the throughput metric whose baseline the read carries.
    ``identities`` looks up the conditions a read's summaries leave out; by default one is built on ``experiences``,
    and passing one index to every briefing of a session reads the export once. ``learnings`` is the session's one
    recall of the specialists' learnings; without it the brief holds this session's only. ``recalls`` keeps the
    completed reads, so a brief built again reads the Experience KB no more. ``champions`` keeps the champions each
    commit read is, by commit, so each commit's recipe is read once a session.
    """

    notebook: Notebook
    warm: tuple[WarmLead, ...] = ()
    experiences: ExperienceReader | None = None
    metric: str = ""
    focus: str = ""
    branch: str = ""
    max_chars: int = DEFAULT_MAX_CHARS
    identities: IdentityIndex | None = None
    learnings: LearningRecall | None = None
    recalls: RecallCache | None = None
    repo: RecipeReads | None = None
    champions: dict[str, frozenset[str]] | None = None
    tables: TableStack | None = None

    def __post_init__(self) -> None:
        if self.identities is None and self.experiences is not None:
            object.__setattr__(self, "identities", IdentityIndex.of(self.experiences))

    async def warm_replay(self, canonical_id: str) -> WarmReplay | None:
        """The change that replays the champion ``canonical_id`` on this branch's head, a gain landing only on the head
        it was measured against; ``None`` when PRELUDE prepared no replay of it, or it would change nothing there.
        ``ValueError`` when the head is the champion's recipe already or builds on a commit that is."""

        prepared = next((lead.replay for lead in self.warm if lead.canonical_id == canonical_id), None)
        if prepared is None or self.repo is None or not self.branch:
            return None
        head = await self.repo.head(self.branch)
        commit = (await self._on_line(await self.notebook.snapshot(), head)).get(canonical_id)
        if commit is not None:
            where = "is its recipe" if commit == head else f"builds on {commit}, which is its recipe"
            raise ValueError(f"{canonical_id} is on your head already: your head {head} {where}; build on it")
        return prepared.on(await self.repo.read_recipe(head), head, self.tables)

    async def _on_line(self, snapshot: NotebookSnapshot, commit: str) -> dict[str, str]:
        """The champions PRELUDE prepared a replay of that ``commit`` is already or builds on, each with the first commit
        found to be it: ``commit`` itself, then the commits its lineage builds on."""

        prepared = [lead.replay for lead in self.warm if lead.replay is not None]
        if self.repo is None or not prepared:
            return {}
        found: dict[str, str] = {}
        for item in dict.fromkeys((commit, *snapshot.lineage.ancestry(commit))):
            for canonical_id in await self._champions_of(self.repo, item, prepared):
                found.setdefault(canonical_id, item)
            if len(found) == len(prepared):
                break
        return found

    async def _champions_of(self, repo: RecipeReads, commit: str, prepared: Sequence[PreparedReplay]) -> frozenset[str]:
        """The champions of ``prepared`` whose recipe ``commit``'s is; none when it cannot be read, so they stay leads."""

        kept = self.champions.get(commit) if self.champions is not None else None
        if kept is not None:
            return kept
        try:
            recipe = await repo.read_recipe(commit)
            kept = frozenset(item.canonical_id for item in prepared if item.is_on(recipe))
        except (KeyError, RuntimeError, ValueError) as exc:
            log.warning("could not read %s to find the warm-start champions it is: %s", commit, exc)
            return frozenset()
        if self.champions is not None:
            self.champions[commit] = kept
        return kept

    async def peer_replay(self, request_id: str) -> PeerReplay:
        """The change another branch's test ``request_id`` tried, to test on this branch's head, with the head's kernel
        artifacts it still swaps out with loss; ``ValueError`` says why there is none: the test is this branch's own or
        changed nothing, was made on this head already, or this head has or cannot take it.

        A component the head does not have takes the upstream it had on the commit the test was made on, which the
        test's change does not name when that commit had the component already. When ``tables`` is set, each kernel the
        change adds in a slot the head already fills is stacked over the head's table instead of replacing it."""

        request = (await self.notebook.record(request_id)).request
        if request.branch == self.branch:
            raise ValueError(f"{request_id} is this branch's own test")
        if request.delta == RecipeDelta():
            raise ValueError(f"{request_id} was measured without a change; there is nothing to replay")
        if self.repo is None or not self.branch:
            raise ValueError("this brief reads no experiment repository, so it has no head to replay on")
        head = await self.repo.head(self.branch)
        if request.parent_commit == head:
            raise ValueError(f"{request_id} tested this change on your head {head}; there is nothing to replay")
        recipe = await self.repo.read_recipe(head)
        delta = await self._with_upstreams(self.repo, request.delta, recipe, request.parent_commit)
        delta, replaces = stack_added_kernels(delta, recipe, self.tables)
        try:
            replayed = apply(recipe, delta)
        except RecipeError as exc:
            raise ValueError(f"the change {request_id} tested no longer applies to your head {head}: {exc}") from exc
        if replayed == recipe:
            raise ValueError(f"your head {head} already has the change {request_id} tested")
        return PeerReplay(request_id, request.branch, head, delta, delta_layer(delta), replaces)

    @staticmethod
    async def _with_upstreams(repo: RecipeReads, delta: RecipeDelta, head: Recipe, tested_on: str) -> RecipeDelta:
        """``delta`` with each component ``head`` does not have, and the change names no upstream for, given the one
        it has in ``tested_on``'s recipe; ``delta`` itself when there is none, or that recipe cannot be read."""

        unnamed = {
            change.component
            for change in delta.components
            if head.component(change.component) is None and None in (change.upstream_repo, change.upstream_ref)
        }
        if not unnamed:
            return delta
        try:
            source = await repo.read_recipe(tested_on)
        except (KeyError, RuntimeError, ValueError) as exc:
            log.warning("could not read %s for the upstream of %s: %s", tested_on, ", ".join(sorted(unnamed)), exc)
            return delta
        changes = []
        for change in delta.components:
            known = source.component(change.component) if change.component in unnamed else None
            if known is not None:
                change = replace(
                    change,
                    upstream_repo=change.upstream_repo or known.upstream_repo,
                    upstream_ref=change.upstream_ref or known.upstream_ref,
                )
            changes.append(change)
        return replace(delta, components=tuple(changes))

    async def _recall(self, snapshot: NotebookSnapshot) -> Recall | None:
        if self.experiences is None or snapshot.profile is None:
            return None
        baseline = snapshot.baseline.values.get(self.metric) if snapshot.baseline is not None else None
        key = (self.focus, self.branch, baseline)
        if self.recalls is not None and (kept := self.recalls.get(key)) is not None:
            return kept
        recall = await asyncio.to_thread(
            recall_experiences,
            self.experiences,
            snapshot.profile,
            baseline,
            RecallQuery(self.focus, self.branch),
            identities=self.identities,
        )
        if self.recalls is not None:
            self.recalls.keep(key, recall)
        return recall

    async def _learnt(self, snapshot: NotebookSnapshot) -> tuple[tuple[LearningRecord, ...], PastLearnings | None]:
        """This session's learnings about ``focus``, and the read of past sessions' when there is a recall."""

        here = session_matches(snapshot, self.focus, limit=LEARNINGS)
        past = None
        if self.learnings is not None:
            past = await asyncio.to_thread(self.learnings.past_for, snapshot, self.focus, limit=LEARNINGS)
        return tuple(here), past

    async def startup_brief(self, start_commit: str) -> StartupBrief:
        """The brief from ``start_commit``; the recall logs the learnings and the leads it shows."""

        tried = await self.notebook.startup_brief(start_commit)
        snapshot = await self.notebook.snapshot()
        on_head = await self._on_line(snapshot, start_commit)
        on_line = [item for item in failures_at(snapshot, start_commit) if item.tier is not FailureTier.ELSEWHERE]
        repeated = [
            f"this session: {direction} failed {count} times on the states this one builds on"
            for direction, count in repeated_failures(on_line)
        ]
        recalled = _recalled(await self._recall(snapshot))
        here, past = await self._learnt(snapshot)
        learnt_before = [f"learnt before: {item.render()[2:]}" for item in (past.items if past is not None else ())]
        kb_lessons = [
            f"{lead.canonical_id}: {text}" for lead in self.warm for text in (*lead.pitfalls, *lead.what_failed)
        ]
        ordered = [
            *(_Item(True, f"{ON_HEAD}: recipe {lead.summary()}") for lead in self.warm if lead.canonical_id in on_head),
            *(_Item(True, text) for text in listed(repeated)),
            *(_Item(True, f"learnt here: {render_session_learning(record, snapshot)[2:]}") for record in here),
            *(_recipe(lead) for lead in self.warm if lead.canonical_id not in on_head),
            *_listed(recalled.leads),
            *(_Item(True, text) for text in listed([*recalled.lessons, *learnt_before, *kb_lessons])),
        ]
        kept = _fit(ordered, self.max_chars)
        if self.learnings is not None:
            shown = tuple(item.lead for item in kept if item.lead is not None)
            self.learnings.record(
                snapshot, Retrieval(self.branch, RetrievalSource.BRIEF, self.focus, shown), here, past
            )
        return replace(
            tried,
            lessons=tuple(item.text for item in kept if item.lesson),
            leads=tuple(item.text for item in kept if not item.lesson),
            focus=self.focus,
        )


@dataclass
class SessionBriefs:
    """The session's briefs for every branch, each built on demand from the notebook as it stands then.

    The fields are :class:`CrossSessionBriefing`'s for the whole session. One identity index and one cache of the
    Experience KB reads serve every branch. A branch keeps the ``focus`` it was last given, so a later call without
    one, at a restart of its specialist, briefs it on the same bottleneck.
    """

    notebook: Notebook
    warm: tuple[WarmLead, ...] = ()
    experiences: ExperienceReader | None = None
    metric: str = ""
    learnings: LearningRecall | None = None
    max_chars: int = DEFAULT_MAX_CHARS
    repo: RecipeReads | None = None
    tables: TableStack | None = None
    _identities: IdentityIndex | None = field(init=False, default=None)
    _recalls: RecallCache = field(init=False, default_factory=RecallCache)
    _champions: dict[str, frozenset[str]] = field(init=False, default_factory=dict)
    _focus: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._identities = IdentityIndex.of(self.experiences) if self.experiences is not None else None

    def for_branch(self, branch: str, focus: str | None = None) -> CrossSessionBriefing:
        """The briefing of ``branch``, on ``focus`` when given and else on the focus it was last given."""

        if focus is not None:
            self._focus[branch] = focus
        return CrossSessionBriefing(
            self.notebook,
            warm=self.warm,
            experiences=self.experiences,
            metric=self.metric,
            focus=self._focus.get(branch, ""),
            branch=branch,
            max_chars=self.max_chars,
            identities=self._identities,
            learnings=self.learnings,
            recalls=self._recalls,
            repo=self.repo,
            champions=self._champions,
            tables=self.tables,
        )

    async def text(self, branch: str, commit: str, focus: str | None = None) -> str:
        """The brief of ``branch`` starting from ``commit``, rendered as the prompt shows it."""

        return render_brief(await self.for_branch(branch, focus).startup_brief(commit))


__all__ = [
    "DEFAULT_MAX_CHARS",
    "ON_HEAD",
    "CrossSessionBriefing",
    "RecallCache",
    "RecipeReads",
    "SessionBriefs",
    "TableStack",
]
