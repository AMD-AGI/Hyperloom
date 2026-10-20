# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Past sessions' learnings from the Experience KB, after this session's, for a branch's next decision.

:class:`LearningRecall` is the session's one ``LearningFinder`` with a knowledge base: this session's learnings first
(``notebook.learning_search``), then a read of the Experience KB's ``multibranch-learning-v1`` schema with this
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
from hyperloom_multibranch.contracts.notebook.records import LearningRecord, NotebookSnapshot, SessionProfile
from hyperloom_multibranch.notebook.knowledge.experience import experience_identity
from hyperloom_multibranch.notebook.knowledge.learnings import MAPPING_PATH
from hyperloom_multibranch.notebook.knowledge.recall import WORKLOAD_KEYS, ExperienceReader, IdentityIndex, classify
from hyperloom_multibranch.notebook.learning_search import (
    SESSION_HEADING,
    Retrieval,
    RetrievalLog,
    RetrievalSource,
    SessionLearnings,
    render_session_learning,
    session_matches,
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
    """A past session's learning; ``workload`` names its kind of session when that is not this session's."""

    experience_id: str
    content: str
    measured: bool
    source_run_id: str
    differing: tuple[str, ...]
    workload: str = ""

    def render(self) -> str:
        where = [f"{', '.join(self.differing)} differ"] if self.differing else []
        if self.workload:
            where.append(f"learnt in an {self.workload} session")
        backed = MEASURED if self.measured else "unmeasured"
        return f"- [{self.source_run_id}, {', '.join(where) or 'same conditions'}, {backed}] {self.content} [{self.experience_id}]"


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
        snapshot: NotebookSnapshot,
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

    async def lines(self, snapshot: NotebookSnapshot, query: str, *, limit: int, branch: str = "") -> list[str]:
        """This session's matches, then past sessions', each under its heading; empty when nothing matched."""

        here = session_matches(snapshot, query, limit=limit)
        past = None
        if snapshot.profile is not None and self._reader is not None:
            past = await asyncio.to_thread(self.past, snapshot.profile, query, limit=limit)
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
    "PastLearning",
    "PastLearnings",
    "learning_conditions",
    "learning_context",
    "read_past_learnings",
]
