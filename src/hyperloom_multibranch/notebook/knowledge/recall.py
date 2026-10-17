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
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, cast

from hyperloom_kb import ReadStatus, RemoteClientError
from hyperloom_multibranch.contracts.notebook.peers import IntentLayer
from hyperloom_multibranch.contracts.notebook.records import SessionProfile
from hyperloom_multibranch.notebook.knowledge.experience import experience_identity

log = logging.getLogger(__name__)

CONDITION_KEYS = ("model", "gpu", "framework", "tp", "conc", "bottleneck", "isl", "osl")
#: The layer each change family of ``inference-recipe-v1`` changes; the schema declares no kernel family.
LAYER_OF_FAMILY = {"config_variant": IntentLayer.CONFIG, "source_patch": IntentLayer.SOURCE}
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


class IdentityIndex:
    """The conditions each Experience of one schema was measured under, and its change family, by id, read from the
    service's export.

    The export is paged from the start the first time an id is missing, and later only from where it stopped, so
    Experiences written since are found too. One index serves every recall of a session; a failed export is logged
    and leaves the identity and the family unknown.
    """

    def __init__(self, export: ExperienceExport, schema_ref: str | None = None) -> None:
        self._export = export
        self._schema_ref = schema_ref
        self._cursor = 0
        self._known: dict[str, dict[str, Any]] = {}
        self._families: dict[str, str] = {}
        self._lock = threading.Lock()

    @classmethod
    def of(cls, reader: object) -> IdentityIndex | None:
        """The index of ``reader``'s schema when its service exports, as a ``RemoteExperienceKB``'s does."""

        client = getattr(reader, "client", None)
        if not callable(getattr(client, "export_page", None)):
            return None
        return cls(cast(ExperienceExport, client), getattr(reader, "schema_ref", None))

    def identity(self, experience_id: str) -> Mapping[str, Any]:
        with self._lock:
            if experience_id not in self._known:
                self._advance()
            return self._known.get(experience_id, {})

    def change_family(self, experience_id: str) -> str:
        """The family of the change the Experience recorded, such as ``config_variant``; empty when unknown."""

        with self._lock:
            if experience_id not in self._known:
                self._advance()
            return self._families.get(experience_id, "")

    def _advance(self) -> None:
        try:
            while True:
                page = self._export.export_page(after=self._cursor, limit=EXPORT_PAGE, schema_ref=self._schema_ref)
                for item in page.items:
                    stored = item.get("experience")
                    if isinstance(stored, Mapping) and stored.get("id"):
                        experience_id = str(stored["id"])
                        self._known[experience_id] = dict(stored.get("identity") or {})
                        change = stored.get("change")
                        family = change.get("change_family") if isinstance(change, Mapping) else None
                        if isinstance(family, str) and family:
                            self._families[experience_id] = family
                self._cursor = page.next_cursor
                if not page.has_more:
                    return
        except (RemoteClientError, OSError) as exc:
            log.warning("could not look Experience identities up in the export: %s", exc)


@dataclass(frozen=True)
class RecalledExperience:
    experience_id: str
    tier: RecallTier
    differing: tuple[str, ...]
    change_summary: str
    decision: str
    baseline_value: float | None
    outcome_value: float | None
    source_run_id: str
    change_family: str = ""

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
        where = f" ({', '.join(self.differing)} differ)" if self.differing else ""
        return f"{self.change_summary}: {self.decision} {gain}{where} [{self.experience_id}]"

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
    """The read context, in the shape the Framework phase's reads use."""

    identity = experience_identity(profile)
    observations = {
        name: text for name, text in (("bottleneck", bottleneck), ("branch", branch), ("intent", intent)) if text
    }
    return {
        "identity": {key: value for key, value in identity.items() if key not in WORKLOAD_KEYS},
        "workload": {key: value for key, value in identity.items() if key in WORKLOAD_KEYS},
        "objective": {"id": "e2e_throughput@v1", "direction": "higher_is_better"},
        "benchmark_baseline": {"throughput": baseline_value, "source": "prelude_baseline"},
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

    A summary without an identity is looked up in ``identities``.
    """

    if reader is None:
        return Recall(ReadStatus.DISABLED.value, "", (), (), ())
    context = recall_context(
        profile, baseline_value, bottleneck=wanted.bottleneck, branch=wanted.branch, intent=wanted.intent
    )
    result = reader.read(DECISION, context, limit=wanted.limit)
    query = recall_conditions(profile, wanted.bottleneck)
    return Recall(
        status=str(result.status),
        read_id=str(result.read_id),
        experiences=tuple(classify(item, query, identities) for item in result.experiences),
        rendered_refs=tuple(ref.id for ref in result.rendered_refs),
        warnings=tuple(str(warning) for warning in result.warnings),
    )


__all__ = [
    "CONDITION_KEYS",
    "DECISION",
    "LAYER_OF_FAMILY",
    "ExperienceExport",
    "ExperienceReader",
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
