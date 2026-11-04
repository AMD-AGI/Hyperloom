# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Past sessions' learnings from the Experience KB, after this session's, for a branch's next decision.

:class:`LearningRecall` is the session's one ``LearningFinder`` with a knowledge base: this session's learnings first
(``blackboard.learning_search``), then a read of the Experience KB's ``multibranch-learning-v1`` schema with this
session's conditions. Each past learning says whether it was learnt under every one of them (model, GPU, framework,
TP, concurrency, sequence lengths), whether tests backed it, and whether it was learnt in the other kind of session:
an AgentX session queries no sequence lengths, so only that label tells it an InferenceX session's learning apart. Each query is read once a session, since a read plans
with an LLM; until some session has published learnings the service does not know the schema, and the read says it is
unavailable. Given a ``RetrievalLog``, each retrieval is one line of it, this session's and past sessions' learnings
together.
"""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hyperloom_kb import ReadStatus, RemoteExperienceKB
from hyperloom_kb.collect import load_mapping
from hyperloom_multibranch.contracts.blackboard.records import LearningRecord, BlackboardSnapshot, SessionProfile
from hyperloom_multibranch.blackboard.knowledge.experience import experience_identity
from hyperloom_multibranch.blackboard.knowledge.learnings import MAPPING_PATH
from hyperloom_multibranch.blackboard.knowledge.recall import WORKLOAD_KEYS, ExperienceReader, IdentityIndex, classify
from hyperloom_multibranch.blackboard.families import Family
from hyperloom_multibranch.blackboard.learning_search import (
    SAME_FAMILY,
    SESSION_HEADING,
    Retrieval,
    RetrievalLog,
    RetrievalSource,
    SessionLearnings,
    render_session_learning,
    session_matches,
    session_scored,
    words,
)

LEARNING_DECISION = "Choose the next step of one branch of a multi-branch LLM inference optimization session."
LEARNING_OBJECTIVE = "specialist_learning@v1"
LEARNING_CONDITIONS = ("model", "gpu", "framework", "tp", "conc", "isl", "osl")
PAST_HEADING = "Past sessions' learnings:"
MEASURED = "measured"
CACHE_SIZE = 64
AGENTX_SESSION = "AgentX"
INFERENCEX_SESSION = "InferenceX"


@dataclass(frozen=True)
class PastLearning:
    """A past session's learning; ``workload`` names its kind of session when that is not this session's, and
    ``confirmations`` how often later sessions bore it out."""

    experience_id: str
    content: str
    measured: bool
    source_run_id: str
    differing: tuple[str, ...]
    workload: str = ""
    confirmations: int = 0

    def render(self) -> str:
        where = [f"{', '.join(self.differing)} differ"] if self.differing else []
        if self.workload:
            where.append(f"learnt in an {self.workload} session")
        backed = MEASURED if self.measured else "unmeasured"
        borne = f", borne out {self.confirmations} times" if self.confirmations else ""
        said = f"{', '.join(where) or 'same conditions'}, {backed}{borne}"
        return f"- [{self.source_run_id}, {said}] {self.content} [{self.experience_id}]"


def _session_kind(identity: Mapping[str, Any]) -> str:
    """The kind of session conditions were recorded in: an AgentX trace replay has no input length."""

    return INFERENCEX_SESSION if "isl" in identity else AGENTX_SESSION


@dataclass(frozen=True)
class PastLearnings:
    """One read of past sessions' learnings; ``status`` is the read's, ``disabled`` without a KB."""

    status: str
    read_id: str = ""
    items: tuple[PastLearning, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status == ReadStatus.COMPLETED.value


def learning_conditions(profile: SessionProfile) -> dict[str, object]:
    """The conditions a past learning must share with this session to count as learnt under the same ones."""

    identity = experience_identity(profile)
    return {key: identity[key] for key in LEARNING_CONDITIONS if key in identity}


def learning_context(profile: SessionProfile, query: str) -> dict[str, Any]:
    """The read context. ``multibranch-learning-v1`` declares the sizes under identity and the Framework phase's
    shape keeps them under workload; the read planner may source them from either, so both hold them."""

    identity = experience_identity(profile)
    return {
        "identity": dict(identity),
        "workload": {key: value for key, value in identity.items() if key in WORKLOAD_KEYS},
        "objective": {"id": LEARNING_OBJECTIVE},
        "observations": {"query": query} if query else {},
    }


def read_past_learnings(
    reader: ExperienceReader | None,
    profile: SessionProfile,
    query: str,
    *,
    limit: int,
    identities: IdentityIndex | None = None,
) -> PastLearnings:
    """Past sessions' learnings for ``query``, each marked by the conditions it differs in and, learnt in the other
    kind of session (AgentX or InferenceX), by that kind; this session's left out."""

    if reader is None:
        return PastLearnings(ReadStatus.DISABLED.value)
    result = reader.read(LEARNING_DECISION, learning_context(profile, query), limit=limit)
    conditions = learning_conditions(profile)
    ours = _session_kind(experience_identity(profile))
    items = []
    for item in result.experiences:
        recalled = classify(item, conditions, identities)
        if recalled.source_run_id == profile.session_id:
            continue
        identity = item.get("identity") or (identities.identity(recalled.experience_id) if identities else {})
        theirs = _session_kind(identity) if identity else ours
        items.append(
            PastLearning(
                recalled.experience_id,
                recalled.change_summary,
                recalled.decision == MEASURED,
                recalled.source_run_id,
                recalled.differing,
                theirs if theirs != ours else "",
            )
        )
    return PastLearnings(
        str(result.status), str(result.read_id), tuple(items), tuple(str(warning) for warning in result.warnings)
    )


@dataclass(frozen=True)
class Notes:
    """The notes that bear on a query in one list, best first, as ``lines``, each ``mine`` when it is this session's;
    which of them are this session's learnings (``here``) and which past sessions' (``past``)."""

    lines: tuple[str, ...] = ()
    here: tuple[LearningRecord, ...] = ()
    past: tuple[PastLearning, ...] = ()
    mine: tuple[bool, ...] = ()

    def logged(self) -> PastLearnings | None:
        """The past sessions' notes as a read the retrieval log records; ``None`` for none."""

        return PastLearnings(ReadStatus.COMPLETED.value, items=self.past) if self.past else None

    def __add__(self, other: Notes) -> Notes:
        return Notes(
            (*self.lines, *other.lines), (*self.here, *other.here), (*self.past, *other.past), (*self.mine, *other.mine)
        )


class LearningRecall(SessionLearnings):
    """One session's retrieval over the learnings, shared by every branch's tools and briefs.

    ``reader`` reads the learning schema of the Experience KB, ``None`` without one; ``identities`` looks up the
    conditions the read summaries leave out, by default on ``reader``'s service. Each query is read once. ``log`` keeps
    what each retrieval showed, this session's and past sessions' learnings on one line.
    """

    def __init__(
        self,
        reader: ExperienceReader | None = None,
        *,
        identities: IdentityIndex | None = None,
        log: RetrievalLog | None = None,
    ) -> None:
        super().__init__(log)
        self._reader = reader
        self._identities = identities if identities is not None else IdentityIndex.of(reader)
        self._read: OrderedDict[tuple[str, int], PastLearnings] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, log: RetrievalLog | None = None) -> LearningRecall:
        """The recall on the Experience KB ``HYPERLOOM_KB_URL`` names; this session's learnings only without one."""

        return cls(RemoteExperienceKB.from_env(load_mapping(MAPPING_PATH).declaration, env), log=log)

    def record(
        self,
        snapshot: BlackboardSnapshot,
        retrieval: Retrieval,
        here: Sequence[LearningRecord],
        past: PastLearnings | None,
    ) -> None:
        """Log one retrieval: this session's learnings ``here`` and the read of past sessions'; nothing without a log."""

        if self.log is None:
            return
        found = past.items if past is not None else ()
        self.log.record(
            snapshot,
            retrieval,
            here,
            past=[item.experience_id for item in found],
            past_status=past.status if past is not None else "",
        )

    def past_revision(self) -> str:
        """A digest of the past sessions' learnings this recall reads as a whole, for the session to record; empty, as
        an Experience KB read is planned query by query and holds no snapshot to name."""

        return ""

    def past_for(self, snapshot: BlackboardSnapshot, query: str, *, limit: int) -> PastLearnings | None:
        """Past sessions' learnings for ``query`` as the session of ``snapshot`` reads them; ``None`` before its
        profile, or without a knowledge base to read."""

        if snapshot.profile is None or self._reader is None:
            return None
        return self.past(snapshot.profile, query, limit=limit)

    def past(self, profile: SessionProfile, query: str, *, limit: int) -> PastLearnings:
        key = (" ".join(sorted(words(query))), limit)
        with self._lock:
            if key in self._read:
                self._read.move_to_end(key)
                return self._read[key]
        found = read_past_learnings(self._reader, profile, query, limit=limit, identities=self._identities)
        if found.available:
            with self._lock:
                self._read[key] = found
                while len(self._read) > CACHE_SIZE:
                    self._read.popitem(last=False)
        return found

    def notes(self, snapshot: BlackboardSnapshot, query: str, *, limit: int, family: Family | None = None) -> Notes:
        """The ``limit`` notes that bear on ``query``, about ``family`` first: this session's learnings, then the
        Experience KB's past learnings, which name no family. It reads the KB, so call it off the event loop."""

        here = session_matches(snapshot, query, limit=limit, family=family)
        read = self.past_for(snapshot, query, limit=limit)
        past = tuple(read.items[: limit - len(here)]) if read is not None and read.available else ()
        rendered = (*(render_session_learning(record, snapshot) for record in here), *(item.render() for item in past))
        return Notes(rendered, here, past, (True,) * len(here) + (False,) * len(past))

    def family_note(self, snapshot: BlackboardSnapshot, family: Family) -> Notes:
        """This session's newest learning about ``family`` itself, for its menu line; past learnings name no family."""

        same = [record for match, record in session_scored(snapshot, "", family) if match.kin == SAME_FAMILY]
        newest = max(same, key=lambda record: record.at, default=None)
        if newest is None:
            return Notes()
        return Notes((render_session_learning(newest, snapshot),), (newest,), (), (True,))

    async def lines(self, snapshot: BlackboardSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
        """This session's matches, then past sessions', each under its heading; empty when nothing matched."""

        here = session_matches(snapshot, query, limit=limit)
        past = await asyncio.to_thread(self.past_for, snapshot, query, limit=limit)
        self.record(snapshot, Retrieval(branch, RetrievalSource.TOOL, query), here, past)
        rendered = [render_session_learning(record, snapshot) for record in here]
        before = _rendered_past(past)
        return [*([SESSION_HEADING, *rendered] if rendered else []), *([PAST_HEADING, *before] if before else [])]


def _rendered_past(past: PastLearnings | None) -> list[str]:
    if past is None:
        return []
    if not past.available:
        return [f"- unavailable ({'; '.join(past.warnings) or past.status})"]
    return [item.render() for item in past.items]


__all__ = [
    "AGENTX_SESSION",
    "CACHE_SIZE",
    "INFERENCEX_SESSION",
    "LEARNING_CONDITIONS",
    "LEARNING_DECISION",
    "LEARNING_OBJECTIVE",
    "PAST_HEADING",
    "LearningRecall",
    "Notes",
    "PastLearning",
    "PastLearnings",
    "learning_conditions",
    "learning_context",
    "read_past_learnings",
]
