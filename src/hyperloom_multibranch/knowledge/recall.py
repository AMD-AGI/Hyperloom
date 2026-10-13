# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recall what other sessions published to the Experience KB, sorted by whether it holds here.

One read per branch, with the context the single-lineage Framework phase sends (identity, workload, objective,
baseline, observations), so one planner serves both flows. A recalled Experience measured under every condition this
session queried with is a prior; one measured under another condition is only a lead to re-verify, and the briefing
names the conditions that differ. The mapping refuses Experiences without evidence, so nothing recalled lacks one.

Sorting reads each summary's ``identity``, which the Experience KB's read summaries carry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from hyperloom_kb import ReadStatus
from hyperloom_multibranch.knowledge.experience import experience_identity
from hyperloom_multibranch.notebook.records import SessionProfile

CONDITION_KEYS = ("gpu", "framework", "bottleneck", "isl", "osl")
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

    @property
    def gain_pct(self) -> float | None:
        """The recorded gain over the Experience's own baseline."""

        if not self.baseline_value or self.outcome_value is None:
            return None
        return (self.outcome_value / self.baseline_value - 1.0) * 100.0

    def render(self) -> str:
        gain = "n/a" if self.gain_pct is None else f"{self.gain_pct:+.2f}%"
        where = f" ({', '.join(self.differing)} differ)" if self.differing else ""
        return f"- {self.change_summary}: {self.decision} {gain}{where} [{self.experience_id}]"


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


def classify(item: Mapping[str, Any], query: Mapping[str, Any]) -> RecalledExperience:
    """Sort one read summary by whether its identity matches every queried condition."""

    identity = item.get("identity") or {}
    differing = tuple(key for key, value in query.items() if key not in identity or not _same(identity[key], value))
    return RecalledExperience(
        experience_id=str(item["experience_id"]),
        tier=RecallTier.REVERIFY if differing else RecallTier.PRIOR,
        differing=differing,
        change_summary=str(item.get("change_summary") or ""),
        decision=str(item.get("decision") or ""),
        baseline_value=_number(item.get("baseline_value")),
        outcome_value=_number(item.get("outcome_value")),
        source_run_id=str(item.get("source_run_id") or ""),
    )


def recall_experiences(
    reader: ExperienceReader | None,
    profile: SessionProfile,
    baseline_value: float | None,
    *,
    bottleneck: str,
    branch: str = "",
    intent: str = "",
    limit: int = DEFAULT_LIMIT,
) -> Recall:
    """Read the Experiences relevant to ``bottleneck`` on this session's task; ``reader`` is ``None`` without a KB."""

    if reader is None:
        return Recall(ReadStatus.DISABLED.value, "", (), (), ())
    context = recall_context(profile, baseline_value, bottleneck=bottleneck, branch=branch, intent=intent)
    result = reader.read(DECISION, context, limit=limit)
    query = recall_conditions(profile, bottleneck)
    return Recall(
        status=str(result.status),
        read_id=str(result.read_id),
        experiences=tuple(classify(item, query) for item in result.experiences),
        rendered_refs=tuple(ref.id for ref in result.rendered_refs),
        warnings=tuple(str(warning) for warning in result.warnings),
    )


__all__ = [
    "CONDITION_KEYS",
    "DECISION",
    "ExperienceReader",
    "Recall",
    "RecallTier",
    "RecalledExperience",
    "classify",
    "recall_conditions",
    "recall_context",
    "recall_experiences",
]
