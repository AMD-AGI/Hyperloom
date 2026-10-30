# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The full startup brief: what a new specialist reads before its first action.

:class:`CrossSessionBriefing` is the second ``StartupBriefProvider``; the local blackboard's own brief stays the fake.
Its brief has three parts:

* tried: what specialists tried on the start commit and on every state it builds on, from the blackboard;
* lessons: this session's repeated failures in one direction on those states (from the lineage analysis), then the
  notes about ``focus``, this session's learnings and past sessions' lessons in one list, its family first, then
  Experiences recalled from other sessions, labelled ``prior`` when measured under this session's conditions and ``to
  re-verify`` otherwise, then the pitfalls and failures the warm-start records list;
* leads: the Recipe KB champions the warm start read, and recalled priors with a positive gain, taking turns by the
  layer they change (configuration, source), so that one kind of change does not fill them all. The Experience KB's
  schema has no kernel family, so no recalled lead is a kernel change;
* the menu: the direction families ranked for the branch on its start commit (:mod:`rsi.menu`), each with its best note
  under it, none for a branch of the menu's control. A claim's reply says the same of the claimed direction's family
  (:meth:`direction_note`).

A champion the start commit is already, or builds on, is no lead: the lessons name it first, as :data:`ON_HEAD`, and
``blackboard_replay`` refuses to replay it, since its change would undo what the head did since.

A peer or warm replay that would swap the head's kernel table out stacks the two tables when ``stack_slot`` is set: the
wider of them, or a union of their rows, decided as a merge decides it (#109 item 7). When neither fits, the swap stays
and the head's artifact is named in ``replaces``.

At the root before anything was tried, the brief is the first batch's: cross-session content only. Lessons and leads
together stay under ``max_chars``, the most specific first. Where the brief sits in the prompt and how long it may be
is xiaofei's call, so the cap is a parameter.

:class:`SessionBriefs` is the session's one source of briefs, so a branch's brief can be rebuilt at every launch of its
specialist, on its head then: the Experience KB is read once per focus and branch, and only the blackboard's part is read
again. It keeps the focus each branch was given, and each head ``blackboard_replay`` refused to replay an unfinished lead
on, because its change no longer applies there or the head has it already: the lead leaves the briefs from that head
and from every commit that builds on it, for the rest of the session.
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
from hyperloom_multibranch.contracts.blackboard.interfaces import Blackboard, StartupBriefProvider
from hyperloom_multibranch.contracts.blackboard.peers import Intent, IntentLayer
from hyperloom_multibranch.contracts.blackboard.records import (
    BlackboardSnapshot,
    PeerReplay,
    StartupBrief,
    UnfinishedLead,
    WarmReplay,
)
from hyperloom_multibranch.blackboard.knowledge.learning_recall import LearningRecall, Notes
from hyperloom_multibranch.blackboard.knowledge.replay_tables import StackSlot, stack_added_kernels
from hyperloom_multibranch.blackboard.knowledge.lessons import MAX_ITEMS, clipped, listed
from hyperloom_multibranch.blackboard.knowledge.recall import (
    ExperienceReader,
    IdentityIndex,
    Recall,
    RecalledExperience,
    RecallQuery,
    RecallTier,
    recall_experiences,
)
from hyperloom_multibranch.blackboard.knowledge.reference_leads import ReferenceLead
from hyperloom_multibranch.blackboard.knowledge.warm_start import PreparedReplay, WarmLead, delta_layer
from hyperloom_multibranch.blackboard.learning_search import (
    Retrieval,
    RetrievalSource,
    ShownLead,
    query_family,
    render_session_learning,
    session_matches,
)
from hyperloom_multibranch.blackboard.rsi.analysis import FailureTier, failures_at, repeated_failures
from hyperloom_multibranch.blackboard.rsi.cases import CasePool, known_defects
from hyperloom_multibranch.blackboard.rsi.menu import CODE, SETTINGS, DirectionMenus, MenuItem
from hyperloom_multibranch.blackboard.tools import render_brief

log = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 4000
KEPT = "keep"
#: How many notes a brief lists about its focus at most, this session's and past sessions' in one list.
LEARNINGS = 5
#: How a family's note sits under its line of the menu.
MENU_NOTE = "  note "
#: What a brief says, before the champion, of one its start commit is already or builds on.
ON_HEAD = "already on your head, nothing to replay"
#: What a brief says, before a past session and its fix, of a pooled case that reproduced on the root.
KNOWN_DEFECT = "known defect of the root, a case from"


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

    ``warm`` are the leads PRELUDE read; ``references`` the reference launches the run was given, listed after them and
    never replayed; ``experiences`` the Experience KB, read for ``focus`` (the bottleneck the specialist attacks) on
    ``branch``; ``metric`` the throughput metric whose baseline the read carries.
    ``identities`` looks up the conditions a read's summaries leave out; by default one is built on ``experiences``,
    and passing one index to every briefing of a session reads the export once. ``learnings`` is the session's one
    recall of the specialists' learnings; without it the brief holds this session's only. ``recalls`` keeps the
    completed reads, so a brief built again reads the Experience KB no more. ``champions`` keeps the champions each
    commit read is, by commit, so each commit's recipe is read once a session. ``refused`` keeps, by test, the heads
    :meth:`peer_replay` found its change no longer applies to or already in, which the briefs from them leave out.
    ``menus`` draws the branch's two lists of directions (:mod:`rsi.menu`); without it the brief has neither.
    ``cases`` is the case pool (:mod:`rsi.cases`): a pooled case that reproduced on the root is a defect the root
    carries, which the brief names first among its lessons.
    """

    blackboard: Blackboard
    warm: tuple[WarmLead, ...] = ()
    references: tuple[ReferenceLead, ...] = ()
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
    stack_slot: StackSlot | None = None
    refused: dict[str, set[str]] | None = None
    menus: DirectionMenus | None = None
    cases: CasePool | None = None

    def __post_init__(self) -> None:
        if self.identities is None and self.experiences is not None:
            object.__setattr__(self, "identities", IdentityIndex.of(self.experiences))

    async def warm_replay(self, canonical_id: str) -> WarmReplay | None:
        """The change that replays the champion ``canonical_id`` on this branch's head, a gain landing only on the head
        it was measured against; ``None`` when PRELUDE prepared no replay of it, or it would change nothing there.
        ``ValueError`` when PRELUDE's check of acceptance refused the replay, or the head is the champion's recipe
        already or builds on a commit that is."""

        lead = next((lead for lead in self.warm if lead.canonical_id == canonical_id), None)
        if lead is not None and lead.refusal is not None:
            raise ValueError(f"{canonical_id} is {lead.refusal.describe()}")
        prepared = lead.replay if lead is not None else None
        if prepared is None or self.repo is None or not self.branch:
            return None
        head = await self.repo.head(self.branch)
        commit = (await self._on_line(await self.blackboard.snapshot(), head)).get(canonical_id)
        if commit is not None:
            where = "is its recipe" if commit == head else f"builds on {commit}, which is its recipe"
            raise ValueError(f"{canonical_id} is on your head already: your head {head} {where}; build on it")
        return await asyncio.to_thread(prepared.on, await self.repo.read_recipe(head), head, self.stack_slot)

    async def warm_reference(self, canonical_id: str, ref: str) -> str | None:
        """The text of the patch ``ref`` the replay PRELUDE prepared of the champion ``canonical_id`` leaves out, the
        same on any head; ``None`` when that replay leaves out no such patch."""

        lead = next((lead for lead in self.warm if lead.canonical_id == canonical_id), None)
        prepared = lead.replay if lead is not None else None
        return prepared.reference_text(ref) if prepared is not None else None

    async def _on_line(self, snapshot: BlackboardSnapshot, commit: str) -> dict[str, str]:
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
        test's change does not name when that commit had the component already. With ``stack_slot``, each kernel the
        change adds in a slot the head already fills is stacked over the head's table instead of replacing it."""

        request = (await self.blackboard.record(request_id)).request
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
        delta, replaces = await asyncio.to_thread(stack_added_kernels, delta, recipe, self.stack_slot)
        try:
            replayed = apply(recipe, delta)
        except RecipeError as exc:
            self._refuse(request_id, head)
            raise ValueError(f"the change {request_id} tested no longer applies to your head {head}: {exc}") from exc
        if replayed == recipe:
            self._refuse(request_id, head)
            raise ValueError(f"your head {head} already has the change {request_id} tested")
        return PeerReplay(request_id, request.branch, head, delta, delta_layer(delta), replaces)

    def _refuse(self, request_id: str, head: str) -> None:
        if self.refused is not None:
            self.refused.setdefault(request_id, set()).add(head)

    def _unrefused(
        self, leads: Sequence[UnfinishedLead], snapshot: BlackboardSnapshot, start_commit: str
    ) -> tuple[UnfinishedLead, ...]:
        """``leads`` less those :meth:`peer_replay` refused on ``start_commit`` or on a commit it builds on."""

        if not self.refused:
            return tuple(leads)
        builds_on = {start_commit, *snapshot.lineage.ancestry(start_commit)}
        return tuple(lead for lead in leads if not self.refused.get(lead.request_id, set()) & builds_on)

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

    async def _recall(self, snapshot: BlackboardSnapshot) -> Recall | None:
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

    async def _learnt(self, snapshot: BlackboardSnapshot) -> Notes:
        """The notes about ``focus`` in one list of :data:`LEARNINGS`: this session's learnings, and with a recall
        past sessions' too."""

        family = query_family(self.focus)
        if self.learnings is not None:
            return await asyncio.to_thread(self.learnings.notes, snapshot, self.focus, limit=LEARNINGS, family=family)
        here = session_matches(snapshot, self.focus, limit=LEARNINGS, family=family)
        lines = tuple(render_session_learning(record, snapshot) for record in here)
        return Notes(lines, here, (), (True,) * len(here))

    async def _menu(self, snapshot: BlackboardSnapshot, start_commit: str) -> tuple[dict[str, tuple[str, ...]], Notes]:
        """The branch's two lists, each family's best note under its line, and those notes; none without menus or for
        a branch of the menu's control."""

        menu = self.menus.menu(snapshot, self.branch, start_commit) if self.menus is not None and self.branch else None
        if menu is None or menu.control:
            return {}, Notes()
        lists: dict[str, tuple[str, ...]] = {}
        shown = Notes()
        for name, items in menu.lists.items():
            lines: list[str] = []
            for item in items:
                found = await self._family_note(snapshot, item)
                lines += [*item.lines(), *(f"{MENU_NOTE}{line[2:]}" for line in found.lines)]
                shown += found
            lists[name] = tuple(lines)
        return lists, shown

    async def _family_note(self, snapshot: BlackboardSnapshot, item: MenuItem) -> Notes:
        """The best note of a family ``item`` stands for: of the first of a code line's layers that has one."""

        if self.learnings is None:
            return Notes()
        for family in item.families:
            found = await asyncio.to_thread(self.learnings.family_note, snapshot, family)
            if found.lines:
                return found
        return Notes()

    async def _known_defects(self, snapshot: BlackboardSnapshot) -> list[_Item]:
        """A lesson for each defect the root carries: each pooled case that reproduced on it, by its diagnosis."""

        if self.cases is None:
            return []
        found = await asyncio.to_thread(known_defects, snapshot, self.cases)
        return [
            _Item(True, f"{KNOWN_DEFECT} {item.origin}: {item.diagnosis.strip().splitlines()[0]}") for item in found
        ]

    async def startup_brief(self, start_commit: str) -> StartupBrief:
        """The brief from ``start_commit``; the recall logs the learnings and the leads it shows."""

        tried = await self.blackboard.startup_brief(start_commit)
        snapshot = await self.blackboard.snapshot()
        on_head = await self._on_line(snapshot, start_commit)
        on_line = [item for item in failures_at(snapshot, start_commit) if item.tier is not FailureTier.ELSEWHERE]
        repeated = [
            f"this session: {direction} failed {count} times on the states this one builds on"
            for direction, count in repeated_failures(on_line)
        ]
        recalled = _recalled(await self._recall(snapshot))
        notes = await self._learnt(snapshot)
        learnt = [
            f"{'learnt here' if mine else 'learnt before'}: {line[2:]}" for line, mine in zip(notes.lines, notes.mine)
        ]
        kb_lessons = [
            f"{lead.canonical_id}: {text}" for lead in self.warm for text in (*lead.pitfalls, *lead.what_failed)
        ]
        ordered = [
            *await self._known_defects(snapshot),
            *(_Item(True, f"{ON_HEAD}: recipe {lead.summary()}") for lead in self.warm if lead.canonical_id in on_head),
            *(_Item(True, text) for text in listed(repeated)),
            *(_Item(True, text) for text in learnt),
            *(_recipe(lead) for lead in self.warm if lead.canonical_id not in on_head),
            *(_Item(False, f"reference: {lead.describe()}") for lead in self.references),
            *_listed(recalled.leads),
            *(_Item(True, text) for text in listed([*recalled.lessons, *kb_lessons])),
        ]
        kept = _fit(ordered, self.max_chars)
        lists, noted = await self._menu(snapshot, start_commit)
        if self.learnings is not None:
            shown = tuple(item.lead for item in kept if item.lead is not None)
            logged = notes + noted
            self.learnings.record(
                snapshot, Retrieval(self.branch, RetrievalSource.BRIEF, self.focus, shown), logged.here, logged.logged()
            )
        return replace(
            tried,
            lessons=tuple(item.text for item in kept if item.lesson),
            leads=tuple(item.text for item in kept if not item.lesson),
            unfinished=self._unrefused(tried.unfinished, snapshot, start_commit),
            focus=self.focus,
            settings_menu=lists.get(SETTINGS, ()),
            code_menu=lists.get(CODE, ()),
        )

    async def direction_note(self, intent: Intent, delta: RecipeDelta) -> str:
        """What the archive and this session say of the family a claim of ``intent`` with ``delta`` falls in."""

        if self.menus is None:
            return ""
        return self.menus.note(await self.blackboard.snapshot(), intent, delta)


@dataclass
class SessionBriefs:
    """The session's briefs for every branch, each built on demand from the blackboard as it stands then.

    The fields are :class:`CrossSessionBriefing`'s for the whole session. One identity index and one cache of the
    Experience KB reads serve every branch. A branch keeps the ``focus`` it was last given, so a later call without
    one, at a restart of its specialist, briefs it on the same bottleneck.
    """

    blackboard: Blackboard
    warm: tuple[WarmLead, ...] = ()
    references: tuple[ReferenceLead, ...] = ()
    experiences: ExperienceReader | None = None
    metric: str = ""
    learnings: LearningRecall | None = None
    max_chars: int = DEFAULT_MAX_CHARS
    repo: RecipeReads | None = None
    stack_slot: StackSlot | None = None
    menus: DirectionMenus | None = None
    cases: CasePool | None = None
    _identities: IdentityIndex | None = field(init=False, default=None)
    _recalls: RecallCache = field(init=False, default_factory=RecallCache)
    _champions: dict[str, frozenset[str]] = field(init=False, default_factory=dict)
    _focus: dict[str, str] = field(init=False, default_factory=dict)
    _refused: dict[str, set[str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._identities = IdentityIndex.of(self.experiences) if self.experiences is not None else None

    def for_branch(self, branch: str, focus: str | None = None) -> CrossSessionBriefing:
        """The briefing of ``branch``, on ``focus`` when given and else on the focus it was last given."""

        if focus is not None:
            self._focus[branch] = focus
        return CrossSessionBriefing(
            self.blackboard,
            warm=self.warm,
            references=self.references,
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
            stack_slot=self.stack_slot,
            refused=self._refused,
            menus=self.menus,
            cases=self.cases,
        )

    async def text(self, branch: str, commit: str, focus: str | None = None) -> str:
        """The brief of ``branch`` starting from ``commit``, rendered as the prompt shows it."""

        return render_brief(await self.for_branch(branch, focus).startup_brief(commit))


__all__ = [
    "DEFAULT_MAX_CHARS",
    "KNOWN_DEFECT",
    "ON_HEAD",
    "CrossSessionBriefing",
    "RecallCache",
    "RecipeReads",
    "SessionBriefs",
]
