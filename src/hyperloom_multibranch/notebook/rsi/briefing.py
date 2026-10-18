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

At the root before anything was tried, the brief is the first batch's: cross-session content only. Lessons and leads
together stay under ``max_chars``, the most specific first. Where the brief sits in the prompt and how long it may be
is xiaofei's call, so the cap is a parameter.

:class:`SessionBriefs` is the session's one source of briefs, so a branch's brief can be rebuilt at every launch of its
specialist, on its head then: the Experience KB is read once per focus and branch, and only the notebook's part is read
again. It keeps the focus each branch was given.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from typing import Protocol

from hyperloom.contracts.recipe import Recipe
from hyperloom_multibranch.contracts.notebook.interfaces import Notebook, StartupBriefProvider
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import (
    LearningRecord,
    NotebookSnapshot,
    StartupBrief,
    WarmReplay,
)
from hyperloom_multibranch.notebook.knowledge.learning_recall import LearningRecall, PastLearnings
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
from hyperloom_multibranch.notebook.knowledge.warm_start import WarmLead
from hyperloom_multibranch.notebook.learning_search import (
    Retrieval,
    RetrievalSource,
    ShownLead,
    render_session_learning,
    session_matches,
)
from hyperloom_multibranch.notebook.rsi.analysis import FailureTier, failures_at, repeated_failures
from hyperloom_multibranch.notebook.tools import render_brief

DEFAULT_MAX_CHARS = 4000
KEPT = "keep"
#: How many learnings of this session, and how many of past sessions, a brief lists at most.
LEARNINGS = 5


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
    completed reads, so a brief built again reads the Experience KB no more.
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

    def __post_init__(self) -> None:
        if self.identities is None and self.experiences is not None:
            object.__setattr__(self, "identities", IdentityIndex.of(self.experiences))

    async def warm_replay(self, canonical_id: str) -> WarmReplay | None:
        """The change that replays the champion ``canonical_id`` on this branch's head, a gain landing only on the head
        it was measured against; ``None`` when PRELUDE prepared no replay of it, or it would change nothing there."""

        prepared = next((lead.replay for lead in self.warm if lead.canonical_id == canonical_id), None)
        if prepared is None or self.repo is None or not self.branch:
            return None
        head = await self.repo.head(self.branch)
        return prepared.on(await self.repo.read_recipe(head), head)

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
        if self.learnings is not None and snapshot.profile is not None:
            past = await asyncio.to_thread(self.learnings.past, snapshot.profile, self.focus, limit=LEARNINGS)
        return tuple(here), past

    async def startup_brief(self, start_commit: str) -> StartupBrief:
        """The brief from ``start_commit``; the recall logs the learnings and the leads it shows."""

        tried = await self.notebook.startup_brief(start_commit)
        snapshot = await self.notebook.snapshot()
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
            *(_Item(True, text) for text in listed(repeated)),
            *(_Item(True, f"learnt here: {render_session_learning(record, snapshot)[2:]}") for record in here),
            *(_recipe(lead) for lead in self.warm),
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
    _identities: IdentityIndex | None = field(init=False, default=None)
    _recalls: RecallCache = field(init=False, default_factory=RecallCache)
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
        )

    async def text(self, branch: str, commit: str, focus: str | None = None) -> str:
        """The brief of ``branch`` starting from ``commit``, rendered as the prompt shows it."""

        return render_brief(await self.for_branch(branch, focus).startup_brief(commit))


__all__ = ["DEFAULT_MAX_CHARS", "CrossSessionBriefing", "RecallCache", "RecipeReads", "SessionBriefs"]
