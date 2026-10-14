# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The full startup brief: what a new specialist reads before its first action.

:class:`CrossSessionBriefing` is the second ``StartupBriefProvider``; the local notebook's own brief stays the fake.
Its brief has three parts:

* tried: what specialists tried on the start commit and on every state it builds on, from the notebook;
* lessons: this session's repeated failures in one direction on those states (from the lineage analysis), then
  Experiences recalled from other sessions, labelled ``prior`` when measured under this session's conditions and
  ``to re-verify`` otherwise, then the pitfalls and failures the warm-start records list;
* leads: the Recipe KB champions the warm start read, and recalled priors with a positive gain.

At the root before anything was tried, the brief is the first batch's: cross-session content only. Lessons and leads
together stay under ``max_chars``, the most specific first. Where the brief sits in the prompt and how long it may be
is xiaofei's call, so the cap is a parameter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, replace

from hyperloom_multibranch.contracts.notebook.interfaces import Notebook, StartupBriefProvider
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, StartupBrief
from hyperloom_multibranch.notebook.knowledge.lessons import listed
from hyperloom_multibranch.notebook.knowledge.recall import ExperienceReader, Recall, RecallTier, recall_experiences
from hyperloom_multibranch.notebook.knowledge.warm_start import WarmLead
from hyperloom_multibranch.notebook.rsi.analysis import FailureTier, failures_at, repeated_failures

DEFAULT_MAX_CHARS = 4000
KEPT = "keep"


@dataclass(frozen=True)
class _Item:
    lesson: bool
    text: str


@dataclass(frozen=True)
class _Recalled:
    """Recalled priors with a positive gain (leads), and the other priors and those to re-verify (lessons)."""

    leads: tuple[str, ...] = ()
    lessons: tuple[str, ...] = ()


def _recalled(recall: Recall | None) -> _Recalled:
    if recall is None or not recall.available:
        return _Recalled()
    priors = recall.of(RecallTier.PRIOR)
    positive = [item for item in priors if (item.gain_pct or 0.0) > 0 and item.decision == KEPT]
    return _Recalled(
        leads=tuple(f"prior: {item.describe()}" for item in positive),
        lessons=(
            *(f"prior: {item.describe()}" for item in priors if item not in positive),
            *(f"to re-verify: {item.describe()}" for item in recall.of(RecallTier.REVERIFY)),
        ),
    )


def _fit(items: Iterable[_Item], max_chars: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Lessons and leads, in priority order, up to the first that would pass ``max_chars``."""

    kept: list[_Item] = []
    used = 0
    for item in items:
        if used + len(item.text) > max_chars:
            break
        kept.append(item)
        used += len(item.text)
    return tuple(item.text for item in kept if item.lesson), tuple(item.text for item in kept if not item.lesson)


@dataclass(frozen=True)
class CrossSessionBriefing(StartupBriefProvider):
    """A startup brief with this session's lessons and what other sessions left.

    ``warm`` are the leads PRELUDE read; ``experiences`` the Experience KB, read for ``focus`` (the bottleneck the
    specialist attacks) on ``branch``; ``metric`` the throughput metric whose baseline the read carries.
    """

    notebook: Notebook
    warm: tuple[WarmLead, ...] = ()
    experiences: ExperienceReader | None = None
    metric: str = ""
    focus: str = ""
    branch: str = ""
    max_chars: int = DEFAULT_MAX_CHARS

    async def _recall(self, snapshot: NotebookSnapshot) -> Recall | None:
        if self.experiences is None or snapshot.profile is None:
            return None
        baseline = snapshot.baseline.values.get(self.metric) if snapshot.baseline is not None else None
        return await asyncio.to_thread(
            recall_experiences,
            self.experiences,
            snapshot.profile,
            baseline,
            bottleneck=self.focus,
            branch=self.branch,
        )

    async def startup_brief(self, start_commit: str) -> StartupBrief:
        tried = await self.notebook.startup_brief(start_commit)
        snapshot = await self.notebook.snapshot()
        on_line = [item for item in failures_at(snapshot, start_commit) if item.tier is not FailureTier.ELSEWHERE]
        repeated = [
            f"this session: {direction} failed {count} times on the states this one builds on"
            for direction, count in repeated_failures(on_line)
        ]
        recalled = _recalled(await self._recall(snapshot))
        kb_lessons = [
            f"{lead.canonical_id}: {text}" for lead in self.warm for text in (*lead.pitfalls, *lead.what_failed)
        ]
        ordered = [
            *(_Item(True, text) for text in listed(repeated)),
            *(_Item(False, f"recipe: {lead.describe()}") for lead in self.warm),
            *(_Item(False, text) for text in listed(recalled.leads)),
            *(_Item(True, text) for text in listed([*recalled.lessons, *kb_lessons])),
        ]
        lessons, leads = _fit(ordered, self.max_chars)
        return replace(tried, lessons=lessons, leads=leads)


__all__ = ["DEFAULT_MAX_CHARS", "CrossSessionBriefing"]
