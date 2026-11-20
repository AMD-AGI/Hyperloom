# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recall what other sessions published to the Experience KB, sorted by whether it holds here.

One read per bottleneck, with the signals the single-lineage Framework phase sends (the session's conditions, its
objective and the bottleneck), weighed as that phase weighs them. A recalled Experience measured under every condition this
session queried with (model, GPU, framework, TP, concurrency, bottleneck, sequence lengths, prompt dataset) is a prior;
one measured under another condition is only a lead to re-verify, and the briefing names the conditions that differ.
Sending no dataset is a condition too: an Experience measured on one is a lead to a session on random prompts. The
mapping refuses Experiences without evidence, so nothing recalled lacks one.

Sorting reads each summary's ``identity``. The service's read summaries carry none yet, so an :class:`IdentityIndex`
looks the identity up by id in the service's export. An Experience whose identity is found nowhere matches no
condition, so it is one to re-verify, never a prior. The summaries carry no change family either; the index keeps
it from the same export, so a recalled Experience names the layer it changed.

An AgentX session asks for the objective it is graded on, interactivity or total throughput, an InferenceX session for
output throughput, and a read may return any of a schema's Experiences. What one kind of session learnt may hold for
another, a crash, a failed accuracy gate or a faster kernel, but its gain is on another objective. So an Experience the
index finds recorded on another objective is one to re-verify, never a prior, after those on the session's own, and
it names its objective; one whose objective is found nowhere is one to re-verify as well.

A session whose tests are published under more than one schema reads them through :class:`SchemaReaders`, one read
per schema, their Experiences in one list, and its index pages every schema's export.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import zip_longest
from typing import Any, Protocol, cast

from hyperloom.common.perf_metric import TOTAL_TOKEN_THROUGHPUT
from hyperloom.inference_optimizer.experience_kb import read_signals
from hyperloom_kb import QuerySignal, ReadStatus, RemoteClientError, RemoteReadResult
from hyperloom_multibranch.contracts.blackboard.peers import IntentLayer
from hyperloom_multibranch.contracts.blackboard.records import SessionProfile
from hyperloom_multibranch.blackboard.knowledge.experience import (
    DATASET_KEY,
    experience_identity,
)

log = logging.getLogger(__name__)

CONDITION_KEYS = ("model", "gpu", "framework", "tp", "conc", "bottleneck", "isl", "osl", DATASET_KEY)
#: The layer each recorded change family changes; ``inference-recipe-v1`` declares the first two.
LAYER_OF_FAMILY = {
    "config_variant": IntentLayer.CONFIG,
    "source_patch": IntentLayer.SOURCE,
    "kernel_artifact": IntentLayer.KERNEL,
}
THROUGHPUT_OBJECTIVE = "e2e_throughput@v1"
INTERACTIVITY_OBJECTIVE = "e2e_interactivity@v1"
TOTAL_THROUGHPUT_OBJECTIVE = "e2e_total_throughput@v1"
EXPORT_PAGE = 100
RECALL_LIMIT = 10


class RecallTier(str, Enum):
    PRIOR = "prior"
    REVERIFY = "reverify"


class ExperienceReader(Protocol):
    """What recall needs of an Experience KB client, such as ``hyperloom_kb.RemoteExperienceKB``."""

    def read(self, signals: Sequence[QuerySignal], *, limit: int | None = None) -> Any:
        """Rank Experiences by ``signals``; the result has ``status``, ``read_id``, ``experiences``,
        ``rendered_refs`` and ``warnings``."""


class ExperienceExport(Protocol):
    """What looking identities up needs of a service client, such as ``hyperloom_kb.RemoteClient``."""

    def export_page(self, *, after: int = 0, limit: int = 100, schema_ref: str | None = None) -> Any:
        """One page of complete records in write order; the page has ``items``, ``next_cursor`` and ``has_more``."""


@dataclass(frozen=True)
class SchemaReaders:
    """One read over several schemas of a service: each reader is asked in turn and the completed reads answer
    together, so one schema the service does not know yet leaves the others' Experiences; with none completed, the
    first read's status answers. Their Experiences take turns by schema, at most ``limit`` in all, so no schema's are
    always the ones cut. Every read's warnings are kept."""

    readers: tuple[ExperienceReader, ...]

    def read(self, signals: Sequence[QuerySignal], *, limit: int | None = None) -> RemoteReadResult:
        results = [reader.read(signals, limit=limit) for reader in self.readers]
        answering = [result for result in results if _status(result) == ReadStatus.COMPLETED.value] or results[:1]
        turns = zip_longest(*(result.experiences for result in answering))
        experiences = tuple(item for turn in turns for item in turn if item is not None)[:limit]
        kept = {item.get("experience_id") for item in experiences}
        return RemoteReadResult(
            read_id=", ".join(str(result.read_id) for result in answering if result.read_id),
            status=_status(answering[0]),
            prompt_block="\n".join(str(result.prompt_block) for result in answering if result.prompt_block),
            rendered_refs=tuple(ref for result in answering for ref in result.rendered_refs if ref.id in kept),
            warnings=tuple(str(warning) for result in results for warning in result.warnings),
            experiences=experiences,
        )


def _status(result: Any) -> str:
    return str(getattr(result.status, "value", result.status))


@dataclass
class _ExportCursor:
    export: ExperienceExport
    schema_ref: str | None
    after: int = 0


class IdentityIndex:
    """The conditions each Experience of one schema was measured under, and its change family, by id, read from the
    service's export.

    The export is paged from the start the first time an id is missing, and later only from where it stopped, so
    Experiences written since are found too. One index serves every recall of a session; a failed export is logged
    and leaves the identity and the family unknown.
    """

    def __init__(self, export: ExperienceExport, schema_ref: str | None = None) -> None:
        self._exports = [_ExportCursor(export, schema_ref)]
        self._known: dict[str, dict[str, Any]] = {}
        self._families: dict[str, str] = {}
        self._objectives: dict[str, str] = {}
        self._lock = threading.Lock()

    @classmethod
    def of(cls, reader: object) -> IdentityIndex | None:
        """The index of ``reader``'s schema when its service exports, as a ``RemoteExperienceKB``'s does, and of each
        schema a :class:`SchemaReaders` reads."""

        cursors: list[_ExportCursor] = []
        for part in reader.readers if isinstance(reader, SchemaReaders) else (reader,):
            client = getattr(part, "client", None)
            if callable(getattr(client, "export_page", None)):
                cursors.append(_ExportCursor(cast(ExperienceExport, client), getattr(part, "schema_ref", None)))
        if not cursors:
            return None
        index = cls(cursors[0].export, cursors[0].schema_ref)
        index._exports.extend(cursors[1:])
        return index

    def identity(self, experience_id: str) -> Mapping[str, Any]:
        with self._lock:
            self._find(experience_id)
            return self._known.get(experience_id, {})

    def change_family(self, experience_id: str) -> str:
        """The family of the change the Experience recorded, such as ``config_variant``; empty when unknown."""

        with self._lock:
            self._find(experience_id)
            return self._families.get(experience_id, "")

    def objective(self, experience_id: str) -> str:
        """The objective the Experience was recorded on, such as ``e2e_throughput@v1``; empty when unknown."""

        with self._lock:
            self._find(experience_id)
            return self._objectives.get(experience_id, "")

    def _find(self, experience_id: str) -> None:
        if experience_id not in self._known:
            self._advance()

    def _advance(self) -> None:
        for cursor in self._exports:
            self._advance_export(cursor)

    def _advance_export(self, cursor: _ExportCursor) -> None:
        try:
            while True:
                page = cursor.export.export_page(after=cursor.after, limit=EXPORT_PAGE, schema_ref=cursor.schema_ref)
                for item in page.items:
                    self._remember(item.get("experience"))
                cursor.after = page.next_cursor
                if not page.has_more:
                    return
        except (RemoteClientError, OSError) as exc:
            log.warning("could not look Experience identities up in the export: %s", exc)

    def _remember(self, stored: object) -> None:
        """Keep one exported Experience's identity, change family and objective; one that names no id is skipped."""

        if not isinstance(stored, Mapping) or not stored.get("id"):
            return
        experience_id = str(stored["id"])
        self._known[experience_id] = dict(stored.get("identity") or {})
        change = stored.get("change")
        family = change.get("change_family") if isinstance(change, Mapping) else None
        if isinstance(family, str) and family:
            self._families[experience_id] = family
        objective = stored.get("objective")
        if isinstance(objective, str) and objective:
            self._objectives[experience_id] = objective


@dataclass(frozen=True)
class RecalledExperience:
    """One recalled Experience; ``objective`` is the one it was recorded on when that is not the session's."""

    experience_id: str
    tier: RecallTier
    differing: tuple[str, ...]
    change_summary: str
    decision: str
    baseline_value: float | None
    outcome_value: float | None
    source_run_id: str
    change_family: str = ""
    objective: str = ""

    @property
    def layer(self) -> IntentLayer | None:
        """The layer the recorded change acts on; ``None`` when its family is unknown."""

        return LAYER_OF_FAMILY.get(self.change_family)

    @property
    def gain_pct(self) -> float | None:
        """The recorded gain over the Experience's own baseline."""

        if not self.baseline_value or self.outcome_value is None:
            return None
        return (self.outcome_value / self.baseline_value - 1.0) * 100.0

    def describe(self) -> str:
        gain = "n/a" if self.gain_pct is None else f"{self.gain_pct:+.2f}%"
        on = f" on {self.objective}" if self.objective else ""
        where = f" ({', '.join(self.differing)} differ)" if self.differing else ""
        return f"{self.change_summary}: {self.decision} {gain}{on}{where} [{self.experience_id}]"

    def render(self) -> str:
        return f"- {self.describe()}"


@dataclass(frozen=True)
class Recall:
    """One Experience KB read for a branch, its Experiences already sorted into tiers."""

    status: str
    read_id: str
    experiences: tuple[RecalledExperience, ...]
    rendered_refs: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.status == ReadStatus.COMPLETED.value

    def of(self, tier: RecallTier) -> tuple[RecalledExperience, ...]:
        return tuple(experience for experience in self.experiences if experience.tier is tier)

    def render(self) -> list[str]:
        if not self.available:
            reason = f": {'; '.join(self.warnings)}" if self.warnings else ""
            return [f"- unavailable ({self.status}{reason})"]
        lines = [f"Experience KB read {self.read_id}; cite these ids when a claim relies on one."]
        for tier, title in (
            (RecallTier.PRIOR, "priors, measured under this session's conditions:"),
            (RecallTier.REVERIFY, "to re-verify, measured under other conditions:"),
        ):
            rendered = [experience.render() for experience in self.of(tier)]
            if rendered:
                lines += [title, *rendered]
        return lines if len(lines) > 1 else [*lines, "- nothing matched"]


def recall_conditions(profile: SessionProfile, bottleneck: str) -> dict[str, object]:
    """The conditions a recalled Experience must share with this session to count as a prior."""

    merged = {**experience_identity(profile), "bottleneck": bottleneck}
    return {key: merged[key] for key in CONDITION_KEYS if merged.get(key) not in (None, "")}


def recall_objective(profile: SessionProfile, metric: str = "") -> str:
    """The objective a read asks for. An AgentX trace replay, which has no input length, asks for the objective of
    ``metric``, the one it is graded on: total throughput, else interactivity."""

    if profile.isl is not None:
        return THROUGHPUT_OBJECTIVE
    return TOTAL_THROUGHPUT_OBJECTIVE if metric == TOTAL_TOKEN_THROUGHPUT else INTERACTIVITY_OBJECTIVE


def _same(left: Any, right: Any) -> bool:
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (left, right)):
        return float(left) == float(right)
    return str(left).strip().lower() == str(right).strip().lower()


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def classify(
    item: Mapping[str, Any], query: Mapping[str, Any], identities: IdentityIndex | None = None
) -> RecalledExperience:
    """Sort one read summary by whether its identity, or the one ``identities`` holds for it, matches every queried
    condition, and names no prompt dataset the query does not; its change family comes from the summary, else from
    ``identities``."""

    experience_id = str(item["experience_id"])
    identity = item.get("identity") or (identities.identity(experience_id) if identities else {})
    family = item.get("change_family") or (identities.change_family(experience_id) if identities else "")
    differing = tuple(key for key, value in query.items() if key not in identity or not _same(identity[key], value))
    if identity.get(DATASET_KEY) and DATASET_KEY not in query:
        differing += (DATASET_KEY,)
    return RecalledExperience(
        experience_id=experience_id,
        tier=RecallTier.REVERIFY if differing else RecallTier.PRIOR,
        differing=differing,
        change_summary=str(item.get("change_summary") or ""),
        decision=str(item.get("decision") or ""),
        baseline_value=_number(item.get("baseline_value")),
        outcome_value=_number(item.get("outcome_value")),
        source_run_id=str(item.get("source_run_id") or ""),
        change_family=str(family),
    )


@dataclass(frozen=True)
class RecallQuery:
    """What one recall asks for: Experiences on ``bottleneck``, at most :data:`RECALL_LIMIT`, on the objective of
    ``metric``, the one the session is graded on (:func:`recall_objective`)."""

    bottleneck: str
    metric: str = ""


def recall_experiences(
    reader: ExperienceReader | None,
    profile: SessionProfile,
    wanted: RecallQuery,
    *,
    identities: IdentityIndex | None = None,
) -> Recall:
    """Read the Experiences relevant to ``wanted`` on this session's task; ``reader`` is ``None`` without a KB.

    A summary without an identity is looked up in ``identities``, and so is every Experience's objective: one recorded
    on another objective than the session's, or on one not found, is one to re-verify, and those on another objective
    come after the rest, each naming its objective.
    """

    if reader is None:
        return Recall(ReadStatus.DISABLED.value, "", (), (), ())
    objective = recall_objective(profile, wanted.metric)
    signals = read_signals(experience_identity(profile), objective, {"bottleneck": wanted.bottleneck})
    result = reader.read(signals, limit=RECALL_LIMIT)
    query = recall_conditions(profile, wanted.bottleneck)
    recalled = [_on_objective(classify(item, query, identities), objective, identities) for item in result.experiences]
    return Recall(
        status=str(result.status),
        read_id=str(result.read_id),
        experiences=tuple(sorted(recalled, key=lambda found: bool(found.objective))),
        rendered_refs=tuple(ref.id for ref in result.rendered_refs),
        warnings=tuple(str(warning) for warning in result.warnings),
    )


def _on_objective(found: RecalledExperience, objective: str, identities: IdentityIndex | None) -> RecalledExperience:
    recorded = identities.objective(found.experience_id) if identities is not None else objective
    if recorded == objective:
        return found
    return replace(found, tier=RecallTier.REVERIFY, differing=(*found.differing, "objective"), objective=recorded)


__all__ = [
    "CONDITION_KEYS",
    "INTERACTIVITY_OBJECTIVE",
    "LAYER_OF_FAMILY",
    "THROUGHPUT_OBJECTIVE",
    "TOTAL_THROUGHPUT_OBJECTIVE",
    "ExperienceExport",
    "ExperienceReader",
    "IdentityIndex",
    "Recall",
    "RecallQuery",
    "RecallTier",
    "RecalledExperience",
    "SchemaReaders",
    "classify",
    "recall_conditions",
    "recall_objective",
    "recall_experiences",
]
