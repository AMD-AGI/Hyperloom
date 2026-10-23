# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recall what other sessions published to the Experience KB, sorted by whether it holds here.

One read per branch, with the context the single-lineage Framework phase sends (identity, workload, objective,
baseline, observations), so one planner serves both flows. A recalled Experience measured under every condition this
session queried with (model, GPU, framework, TP, concurrency, bottleneck, sequence lengths) is a prior; one measured
under another condition is only a lead to re-verify, and the briefing names the conditions that differ. The mapping
refuses Experiences without evidence, so nothing recalled lacks one.

Sorting reads each summary's ``identity``. The service's read summaries carry none yet, so an :class:`IdentityIndex`
looks the identity up by id in the service's export. An Experience whose identity is found nowhere matches no
condition, so it is one to re-verify, never a prior. The summaries carry no change family either; the index keeps
it from the same export, so a recalled Experience names the layer it changed.

The session's Experiences live under two declarations, the shared ``inference-recipe-v1`` and the notebook's own
``multibranch-recipe-v1`` for kernel changes and AgentX sessions; :class:`ExperienceReaders` reads both as one. An
AgentX session asks for the interactivity objective, an InferenceX session for output throughput, and a read may
return any of a schema's Experiences. What one kind of session learnt may hold for the other, a crash, a failed
accuracy gate or a faster kernel, but its gain is on another objective and another workload. So an Experience the
index finds recorded on another objective is one to re-verify, never a prior, after those on the session's own, and
it names its objective; one whose objective is found nowhere is one to re-verify as well.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol, cast

from hyperloom_kb import ReadStatus, RemoteClientError, RemoteExperienceKB, RemoteReadResult
from hyperloom_kb.collect import load_mapping
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import SessionProfile
from hyperloom_multibranch.notebook.knowledge.experience import (
    MAPPING_PATH,
    MULTIBRANCH_MAPPING_PATH,
    experience_identity,
)

log = logging.getLogger(__name__)

CONDITION_KEYS = ("model", "gpu", "framework", "tp", "conc", "bottleneck", "isl", "osl")
#: The layer each recorded change family changes; ``inference-recipe-v1`` declares the first two.
LAYER_OF_FAMILY = {
    "config_variant": IntentLayer.CONFIG,
    "source_patch": IntentLayer.SOURCE,
    "kernel_artifact": IntentLayer.KERNEL,
}
THROUGHPUT_OBJECTIVE = "e2e_throughput@v1"
INTERACTIVITY_OBJECTIVE = "e2e_interactivity@v1"
EXPORT_PAGE = 100
WORKLOAD_KEYS = frozenset({"tp", "ep", "conc", "isl", "osl", "max_model_len", "compute_partition_mode", "partitions"})
DECISION = "Choose the next actions for one branch of a multi-branch inference optimization session."
DEFAULT_LIMIT = 10


class RecallTier(str, Enum):
    PRIOR = "prior"
    REVERIFY = "reverify"


class ExperienceReader(Protocol):
    """What recall needs of an Experience KB client, such as ``hyperloom_kb.RemoteExperienceKB``."""

    def read(self, decision: str, context: Mapping[str, Any], *, limit: int | None = None) -> Any:
        """Rank Experiences for ``decision``; the result has ``status``, ``read_id``, ``experiences``,
        ``rendered_refs`` and ``warnings``."""


class ExperienceExport(Protocol):
    """What looking identities up needs of a service client, such as ``hyperloom_kb.RemoteClient``."""

    def export_page(self, *, after: int = 0, limit: int = 100, schema_ref: str | None = None) -> Any:
        """One page of complete records in write order; the page has ``items``, ``next_cursor`` and ``has_more``."""


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
        """The index of ``reader``'s schema when its service exports, as a ``RemoteExperienceKB``'s does; of every
        schema of :class:`ExperienceReaders` whose service exports."""

        index: IdentityIndex | None = None
        for each in reader.readers if isinstance(reader, ExperienceReaders) else (reader,):
            client = getattr(each, "client", None)
            if not callable(getattr(client, "export_page", None)):
                continue
            export, schema_ref = cast(ExperienceExport, client), getattr(each, "schema_ref", None)
            if index is None:
                index = cls(export, schema_ref)
            else:
                index._exports.append(_ExportCursor(export, schema_ref))
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
class ExperienceReaders:
    """Several Experience KB readers read as one, such as one per declaration the session's Experiences live under.

    Every reader is asked with the same context. The completed answers are merged in the readers' order, each at most
    ``limit`` Experiences; the warnings are every answer's. With none completed, the first answer is the result.
    """

    readers: tuple[ExperienceReader, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ExperienceReaders | None:
        """One reader per declaration on the Experience KB ``HYPERLOOM_KB_URL`` names; ``None`` without one."""

        readers = tuple(
            reader
            for path in (MAPPING_PATH, MULTIBRANCH_MAPPING_PATH)
            if (reader := RemoteExperienceKB.from_env(load_mapping(path).declaration, env)) is not None
        )
        return cls(readers) if readers else None

    def read(self, decision: str, context: Mapping[str, Any], *, limit: int | None = None) -> RemoteReadResult:
        answers = [reader.read(decision, context, limit=limit) for reader in self.readers]
        completed = [answer for answer in answers if str(answer.status) == ReadStatus.COMPLETED.value]
        if not completed:
            return answers[0] if answers else RemoteReadResult("", ReadStatus.DISABLED.value, "", (), ())
        return RemoteReadResult(
            read_id=", ".join(answer.read_id for answer in completed if answer.read_id),
            status=ReadStatus.COMPLETED.value,
            prompt_block="\n\n".join(answer.prompt_block for answer in completed if answer.prompt_block),
            rendered_refs=tuple(ref for answer in completed for ref in answer.rendered_refs),
            warnings=tuple(str(warning) for answer in answers for warning in answer.warnings),
            experiences=tuple(item for answer in completed for item in answer.experiences),
        )


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


def recall_context(
    profile: SessionProfile, baseline_value: float | None, *, bottleneck: str, branch: str = "", intent: str = ""
) -> dict[str, Any]:
    """The read context, in the shape the Framework phase's reads use; an AgentX trace replay, which has no input
    length, asks for the interactivity objective."""

    identity = experience_identity(profile)
    observations = {
        name: text for name, text in (("bottleneck", bottleneck), ("branch", branch), ("intent", intent)) if text
    }
    replays_traces = profile.isl is None
    return {
        "identity": {key: value for key, value in identity.items() if key not in WORKLOAD_KEYS},
        "workload": {key: value for key, value in identity.items() if key in WORKLOAD_KEYS},
        "objective": {
            "id": INTERACTIVITY_OBJECTIVE if replays_traces else THROUGHPUT_OBJECTIVE,
            "direction": "higher_is_better",
        },
        "benchmark_baseline": {
            "interactivity" if replays_traces else "throughput": baseline_value,
            "source": "prelude_baseline",
        },
        "observations": observations,
    }


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
    condition; its change family comes from the summary, else from ``identities``."""

    experience_id = str(item["experience_id"])
    identity = item.get("identity") or (identities.identity(experience_id) if identities else {})
    family = item.get("change_family") or (identities.change_family(experience_id) if identities else "")
    differing = tuple(key for key, value in query.items() if key not in identity or not _same(identity[key], value))
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
    """What one recall asks for: Experiences on ``bottleneck``, for ``branch`` about to try ``intent``, at most
    ``limit``."""

    bottleneck: str
    branch: str = ""
    intent: str = ""
    limit: int = DEFAULT_LIMIT


def recall_experiences(
    reader: ExperienceReader | None,
    profile: SessionProfile,
    baseline_value: float | None,
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
    context = recall_context(
        profile, baseline_value, bottleneck=wanted.bottleneck, branch=wanted.branch, intent=wanted.intent
    )
    result = reader.read(DECISION, context, limit=wanted.limit)
    query = recall_conditions(profile, wanted.bottleneck)
    objective = str(context["objective"]["id"])
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
    "DECISION",
    "INTERACTIVITY_OBJECTIVE",
    "LAYER_OF_FAMILY",
    "THROUGHPUT_OBJECTIVE",
    "ExperienceExport",
    "ExperienceReader",
    "ExperienceReaders",
    "IdentityIndex",
    "Recall",
    "RecallQuery",
    "RecallTier",
    "RecalledExperience",
    "classify",
    "recall_conditions",
    "recall_context",
    "recall_experiences",
]
