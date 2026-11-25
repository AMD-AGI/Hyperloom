"""The typed evidence the Analysis Agent publishes about a canonical commit, and its standing against the tree."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _identifier(value: Any, label: str) -> str:
    normalized = _text(value, label)
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase identifier")
    return normalized


def _optional_number(
    value: Any,
    label: str,
    *,
    positive: bool = False,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{label} must be greater than zero")
    return number


def calculate_evidence_gain(
    evidence_mean_case_speedup: float | None,
    current_mean_case_speedup: float | None,
) -> float | None:
    """Return canonical gain since evidence collection, when scores are valid."""
    if evidence_mean_case_speedup is None or current_mean_case_speedup is None:
        return None
    evidence = float(evidence_mean_case_speedup)
    current = float(current_mean_case_speedup)
    if not math.isfinite(evidence) or not math.isfinite(current) or evidence <= 0 or current <= 0:
        return None
    return current / evidence - 1.0


@dataclass(frozen=True)
class EvidenceRef:
    """Reference one immutable evidence artifact."""

    kind: str
    path: str
    summary: str = ""

    def __post_init__(self) -> None:
        _identifier(self.kind, "evidence.kind")
        _text(self.path, "evidence.path")
        _text(self.summary, "evidence.summary", allow_empty=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class CaseRoofline:
    """One case's estimated roofline ceiling, and how much of it the incumbent reaches.

    ``attainment`` is ``ceiling_ms / incumbent_ms``. It is absent, with the
    reason in ``excluded``, when the two cannot be divided into a meaningful
    figure -- most often a ceiling above the latency already measured, which is
    the estimate contradicting itself rather than a kernel beyond its limit.
    """

    ceiling_ms: float
    incumbent_ms: float | None = None
    attainment: float | None = None
    excluded: str = ""

    def __post_init__(self) -> None:
        if self.ceiling_ms is None:
            raise ValueError("case.roofline.ceiling_ms is required")
        _optional_number(self.ceiling_ms, "case.roofline.ceiling_ms", positive=True)
        _optional_number(self.incumbent_ms, "case.roofline.incumbent_ms", positive=True)
        _optional_number(self.attainment, "case.roofline.attainment", positive=True)
        _text(self.excluded, "case.roofline.excluded", allow_empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ceiling_ms": self.ceiling_ms,
            "incumbent_ms": self.incumbent_ms,
            "attainment": self.attainment,
            "excluded": self.excluded,
        }


#: How a reviewing agent is to read the roofline figures. Carried in
#: the payload beside them, so it reaches every agent that sees them and no
#: campaign without a ceiling has its prompt changed.
ROOFLINE_HOW_TO_READ = (
    "Each case's roofline.attainment is ceiling_ms / incumbent_ms: the fraction of that shape's estimated best "
    "achievable latency the current best kernel delivers, 1.0 being at the ceiling. The objective is the "
    "equal-weight mean across scored cases, so the case with the lowest attainment is where the next round's "
    "effort buys the most, and a case near 1.0 has little headroom left. A case whose roofline.excluded is set has "
    "no usable figure, most often because its ceiling sits above the latency already measured: that ceiling is "
    "wrong, and the case must not be read as finished. mean_attainment is the objective only when covered_cases "
    "equals scored_cases. When target is set, the campaign stops once mean_attainment reaches it over every scored "
    "case. The ceiling is an agent's estimate, not a measurement: use it to choose where to work, never to accept "
    "or reject a candidate."
)


@dataclass(frozen=True)
class CampaignRoofline:
    """The campaign's standing against its estimated roofline ceiling.

    ``mean_attainment`` is the equal-weight mean of the per-case attainment over
    ``covered_cases`` of the ``scored_cases`` the objective scores, or ``None``
    when no case has a figure. ``target`` is the mean attainment the campaign
    stops at, or ``None`` when it was given none.
    """

    scored_cases: int
    covered_cases: int
    mean_attainment: float | None = None
    target: float | None = None

    def __post_init__(self) -> None:
        if self.scored_cases < 0 or not 0 <= self.covered_cases <= self.scored_cases:
            raise ValueError("roofline.covered_cases must lie between 0 and roofline.scored_cases")
        _optional_number(self.mean_attainment, "roofline.mean_attainment", positive=True)
        _optional_number(self.target, "roofline.target", positive=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_attainment": self.mean_attainment,
            "target": self.target,
            "covered_cases": self.covered_cases,
            "scored_cases": self.scored_cases,
            "how_to_read": ROOFLINE_HOW_TO_READ,
        }


@dataclass(frozen=True)
class CaseEvidence:
    """Normalized case evidence exposed to read-only reviewing agents."""

    case_id: str
    shape: str = ""
    dtype: str = ""
    latency_ms: float | None = None
    bottleneck: str = ""
    profile_summary_path: str = ""
    flags: tuple[str, ...] = ()
    # Present only when the campaign has a roofline ceiling: an estimate, which decides no KEEP.
    roofline: CaseRoofline | None = None

    def __post_init__(self) -> None:
        _text(self.case_id, "case.case_id")
        _text(self.shape, "case.shape", allow_empty=True)
        _text(self.dtype, "case.dtype", allow_empty=True)
        _optional_number(self.latency_ms, "case.latency_ms", positive=True)
        _text(self.bottleneck, "case.bottleneck", allow_empty=True)
        _text(
            self.profile_summary_path,
            "case.profile_summary_path",
            allow_empty=True,
        )
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("case.flags must not contain duplicates")
        if self.roofline is not None and not isinstance(self.roofline, CaseRoofline):
            raise ValueError("case.roofline must be a CaseRoofline or absent")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "latency_ms": self.latency_ms,
            "bottleneck": self.bottleneck,
            "profile_summary_path": self.profile_summary_path,
            "flags": list(self.flags),
        }
        # Omitted rather than null, so a campaign without a ceiling plans from exactly the evidence it always did.
        if self.roofline is not None:
            payload["roofline"] = self.roofline.to_dict()
        return payload


@dataclass(frozen=True)
class EvidenceContext:
    """The evidence about one canonical commit that the Analysis Agent gathers and the round's sessions read."""

    analysis_commit: str
    workspace: str
    gpu_target: str
    objective: str
    program_context: str
    source_map_path: str
    cases: tuple[CaseEvidence, ...]
    # Every file the campaign declared as its own source set, in campaign order (entry 0 is the primary kernel path).
    editable_sources: tuple[str, ...] = ()
    knowledge_index: str = ""
    supervisor_guidance: str = ""
    search_mode: str = "EXPLOIT"
    search_reason_codes: tuple[str, ...] = ()
    search_objective: str = "CANONICAL_GAIN"
    evidence_refs: tuple[EvidenceRef, ...] = ()
    # ``analysis_commit`` remains the canonical commit for compatibility.
    canonical_commit: str = ""
    evidence_commit: str = ""
    evidence_stale: bool = False
    evidence_status: str = ""
    evidence_mean_case_speedup: float | None = None
    current_mean_case_speedup: float | None = None
    cumulative_diff_path: str = ""
    cumulative_diff_error: str = ""
    # Present only when the campaign has a roofline ceiling; each case's own figure rides on ``CaseEvidence.roofline``.
    roofline: CampaignRoofline | None = None

    def __post_init__(self) -> None:
        _text(self.analysis_commit, "context.analysis_commit")
        _text(self.workspace, "context.workspace")
        _text(self.gpu_target, "context.gpu_target")
        _text(self.objective, "context.objective")
        _text(self.program_context, "context.program_context")
        _text(self.source_map_path, "context.source_map_path")
        for index, source in enumerate(self.editable_sources):
            _text(source, f"context.editable_sources[{index}]")
        if len(set(self.editable_sources)) != len(self.editable_sources):
            raise ValueError("context.editable_sources must not contain duplicates")
        _text(self.knowledge_index, "context.knowledge_index", allow_empty=True)
        _text(
            self.supervisor_guidance,
            "context.supervisor_guidance",
            allow_empty=True,
        )
        if self.search_mode not in {"EXPLOIT", "DIVERSIFY"}:
            raise ValueError("context.search_mode is unsupported")
        _text(self.search_objective, "context.search_objective")
        self._validate_evidence_standing()
        case_ids = [case.case_id for case in self.cases]
        if not case_ids:
            raise ValueError("context.cases must not be empty")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("context.cases must have unique case_id values")
        if self.roofline is not None and not isinstance(self.roofline, CampaignRoofline):
            raise ValueError("context.roofline must be a CampaignRoofline or absent")

    def _validate_evidence_standing(self) -> None:
        """Validate the fields that place the Analysis evidence against the canonical commit."""
        _text(
            self.canonical_commit,
            "context.canonical_commit",
            allow_empty=True,
        )
        _text(
            self.evidence_commit,
            "context.evidence_commit",
            allow_empty=True,
        )
        _text(
            self.evidence_status,
            "context.evidence_status",
            allow_empty=True,
        )
        _optional_number(
            self.evidence_mean_case_speedup,
            "context.evidence_mean_case_speedup",
            positive=True,
        )
        _optional_number(
            self.current_mean_case_speedup,
            "context.current_mean_case_speedup",
            positive=True,
        )
        _text(
            self.cumulative_diff_path,
            "context.cumulative_diff_path",
            allow_empty=True,
        )
        _text(
            self.cumulative_diff_error,
            "context.cumulative_diff_error",
            allow_empty=True,
        )

    @property
    def case_ids(self) -> frozenset[str]:
        return frozenset(case.case_id for case in self.cases)

    def to_prompt_dict(
        self,
        *,
        case_ids: tuple[str, ...] | None = None,
        evidence_refs: tuple[EvidenceRef, ...] | None = None,
    ) -> dict[str, Any]:
        selected = self.case_ids if case_ids is None else frozenset(case_ids)
        canonical_commit = self.canonical_commit or self.analysis_commit
        evidence_commit = self.evidence_commit or self.analysis_commit
        gain_since_evidence = calculate_evidence_gain(
            self.evidence_mean_case_speedup,
            self.current_mean_case_speedup,
        )
        payload: dict[str, Any] = {
            "analysis_commit": self.analysis_commit,
            "canonical_commit": canonical_commit,
            "analysis_evidence": {
                "commit": evidence_commit,
                "status": self.evidence_status or "current",
                "stale": self.evidence_stale,
                "mean_case_speedup_at_collection": (self.evidence_mean_case_speedup),
                "current_mean_case_speedup": self.current_mean_case_speedup,
                "gain_since_collection": gain_since_evidence,
                "cumulative_diff_path": self.cumulative_diff_path,
                "cumulative_diff_error": self.cumulative_diff_error,
                "path_policy": "absolute_workspace_paths",
            },
            "workspace": self.workspace,
            "gpu_target": self.gpu_target,
            "objective": self.objective,
            "program_context": self.program_context,
            "source_map_path": self.source_map_path,
            "editable_sources": list(self.editable_sources),
            "knowledge_index": self.knowledge_index,
            "supervisor_guidance": self.supervisor_guidance,
            "search_policy": {
                "mode": self.search_mode,
                "reason_codes": list(self.search_reason_codes),
                "objective_kind": self.search_objective,
            },
            "cases": [case.to_dict() for case in self.cases if case.case_id in selected],
            "evidence_refs": [
                ref.to_dict() for ref in (self.evidence_refs if evidence_refs is None else evidence_refs)
            ],
        }
        # Omitted rather than null, like ``CaseEvidence.roofline``.
        if self.roofline is not None:
            payload["roofline"] = self.roofline.to_dict()
        return payload
