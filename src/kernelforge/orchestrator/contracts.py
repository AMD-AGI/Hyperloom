"""Typed contracts for forge-loop orchestration and specialist analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_FAILURE_KINDS = frozenset(
    {
        "unknown_role",
        "timeout",
        "backend_failure",
        "backend_error",
        "empty_output",
    }
)


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
class CaseEvidence:
    """Normalized case evidence exposed to read-only planning agents."""

    case_id: str
    shape: str = ""
    dtype: str = ""
    latency_ms: float | None = None
    bottleneck: str = ""
    profile_summary_path: str = ""
    flags: tuple[str, ...] = ()

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "latency_ms": self.latency_ms,
            "bottleneck": self.bottleneck,
            "profile_summary_path": self.profile_summary_path,
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class OrchestrationContext:
    """Immutable evidence bundle shared by orchestration and specialists."""

    analysis_commit: str
    workspace: str
    gpu_target: str
    objective: str
    program_context: str
    source_map_path: str
    cases: tuple[CaseEvidence, ...]
    # Every file the campaign declared as its own source set, in campaign order
    # (entry 0 is the primary kernel path). This is the declared FLOOR of the
    # edit surface, never its ceiling -- the hard boundary is the protected
    # measurement surface. A planner that is never told it may edit the tuned
    # CSV or the sibling module that holds the dispatch constant will reason as
    # though only the anchor file exists and price whole directions out on that
    # mistake. Data and config files belong here exactly as much as .py sources.
    editable_sources: tuple[str, ...] = ()
    knowledge_index: str = ""
    supervisor_guidance: str = ""
    search_mode: str = "EXPLOIT"
    search_reason_codes: tuple[str, ...] = ()
    search_objective: str = "IMMEDIATE_CANONICAL_GAIN"
    search_mode_residence_remaining: int = 0
    evidence_refs: tuple[EvidenceRef, ...] = ()
    # ``analysis_commit`` remains the canonical commit for compatibility.
    # These fields separate the code being planned from the commit that
    # produced the active Analysis/profiling evidence.
    canonical_commit: str = ""
    evidence_commit: str = ""
    evidence_stale: bool = False
    evidence_status: str = ""
    evidence_mean_case_speedup: float | None = None
    current_mean_case_speedup: float | None = None
    cumulative_diff_path: str = ""
    cumulative_diff_error: str = ""
    # The previous iteration's Plan Critic ruling. Carried because a critic can
    # only rule on a plan that already exists, so a verdict that the route
    # itself is dominated cannot change the round it was passed on -- it can
    # only change the next one.
    last_critic_verdict: str = ""
    last_critic_review: str = ""

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
        if self.last_critic_verdict and self.last_critic_verdict not in {
            "ACCEPT",
            "REVISE",
            "REPLACE",
        }:
            raise ValueError("context.last_critic_verdict is unsupported")
        _text(
            self.last_critic_review,
            "context.last_critic_review",
            allow_empty=True,
        )
        if self.search_mode not in {"EXPLOIT", "DIVERSIFY"}:
            raise ValueError("context.search_mode is unsupported")
        _text(self.search_objective, "context.search_objective")
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
        if self.search_mode_residence_remaining < 0:
            raise ValueError("context.search_mode_residence_remaining must be non-negative")
        case_ids = [case.case_id for case in self.cases]
        if not case_ids:
            raise ValueError("context.cases must not be empty")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("context.cases must have unique case_id values")

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
        return {
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
            "last_plan_critic": {
                "verdict": self.last_critic_verdict,
                "review": self.last_critic_review,
            },
            "search_policy": {
                "mode": self.search_mode,
                "reason_codes": list(self.search_reason_codes),
                "objective_kind": self.search_objective,
                "residence_iterations_remaining": (self.search_mode_residence_remaining),
            },
            "cases": [case.to_dict() for case in self.cases if case.case_id in selected],
            "evidence_refs": [
                ref.to_dict() for ref in (self.evidence_refs if evidence_refs is None else evidence_refs)
            ],
        }


@dataclass(frozen=True)
class SpecialistDefinition:
    """Describe one registered read-only specialist role."""

    role_id: str
    description: str
    instructions: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.role_id, "specialist.role_id")
        _text(self.description, "specialist.description")
        _text(self.instructions, "specialist.instructions")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("specialist.capabilities must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "description": self.description,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class DispatchIntent:
    """Minimal probabilistic role/case intent returned by orchestration."""

    role_id: str
    target_case_ids: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class SpecialistAssignment:
    """Assign one specialist to an evidence-scoped analysis task."""

    assignment_id: str
    role_id: str
    target_case_ids: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    reason: str

    def __post_init__(self) -> None:
        _identifier(self.assignment_id, "assignment.assignment_id")
        _identifier(self.role_id, "assignment.role_id")
        if not self.target_case_ids:
            raise ValueError("assignment.target_case_ids must not be empty")
        if len(set(self.target_case_ids)) != len(self.target_case_ids):
            raise ValueError("assignment.target_case_ids must not contain duplicates")
        _text(self.reason, "assignment.reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "role_id": self.role_id,
            "target_case_ids": list(self.target_case_ids),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DispatchPlan:
    """Framework-bound specialist assignments and normalization notes."""

    analysis_commit: str
    assignments: tuple[SpecialistAssignment, ...]
    normalization_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_commit": self.analysis_commit,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "normalization_notes": list(self.normalization_notes),
        }


@dataclass(frozen=True)
class SpecialistFailure:
    """Normalize one isolated specialist failure."""

    kind: str
    message: str

    def __post_init__(self) -> None:
        if self.kind not in _FAILURE_KINDS:
            raise ValueError(f"unsupported specialist failure kind: {self.kind}")
        _text(self.message, "specialist failure.message")


@dataclass(frozen=True)
class SpecialistOutcome:
    """Hold one free-form specialist analysis or an isolated failure."""

    assignment_id: str
    role_id: str
    duration_sec: float
    content: str | None = None
    failure: SpecialistFailure | None = None

    def __post_init__(self) -> None:
        _identifier(self.assignment_id, "specialist outcome.assignment_id")
        _identifier(self.role_id, "specialist outcome.role_id")
        _optional_number(
            self.duration_sec,
            "specialist outcome.duration_sec",
        )
        if self.duration_sec < 0:
            raise ValueError("specialist outcome.duration_sec must not be negative")
        if (self.content is None) == (self.failure is None):
            raise ValueError("specialist outcome must contain exactly one analysis or failure")
        if self.content is not None:
            _text(self.content, "specialist outcome.content")

    @property
    def succeeded(self) -> bool:
        return self.content is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "role_id": self.role_id,
            "duration_sec": self.duration_sec,
            "content": self.content,
            "failure": (
                {
                    "kind": self.failure.kind,
                    "message": self.failure.message,
                }
                if self.failure
                else None
            ),
        }


@dataclass(frozen=True)
class LaneDrop:
    """One lane the review judged not worth its Implementer session.

    The reason is required and travels with the lane_id, because a lane is
    dropped for something the review found -- a ground the evidence does not
    support, another lane's change in different words -- and a round that
    published fewer lanes than it planned without saying why cannot be audited
    afterwards. Whether the drop is obeyed is not decided here: the round's
    width belongs to whoever holds the lanes.
    """

    lane_id: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.lane_id, int) or isinstance(self.lane_id, bool):
            raise ValueError("lane drop.lane_id must be an integer")
        if self.lane_id < 1:
            raise ValueError("lane drop.lane_id must be positive")
        _text(self.reason, "lane drop.reason")

    def to_dict(self) -> dict[str, Any]:
        return {"lane_id": self.lane_id, "reason": self.reason}


@dataclass(frozen=True)
class PlanCriticOutcome:
    """One free-form plan review, its routing verdict, and its width ruling.

    The verdict routes the round's one implementation route; ``lane_drops``
    rules on how much of the round is worth running. They are separate because
    a round is one route divided into several lanes: "this route needs
    correcting" and "this lane is not worth a session" are different findings,
    and a vocabulary that only carries the first leaves the second with no
    outlet.
    """

    verdict: str
    review: str = ""
    error: str = ""
    duration_sec: float = 0.0
    verdict_source: str = "explicit"
    lane_drops: tuple[LaneDrop, ...] = ()
    # What reading the review's width block found: a block that was not there,
    # an entry that named no lane. Named rather than dropped, so "the review
    # wanted every lane" and "the review wanted something nobody could read"
    # never reach the round as one answer. Each note stays an observation and
    # leaves the outcome to ``narrowing_status`` and ``lane_drops``, because a
    # note that concluded anything would be concluding it before the repair
    # pass and the round have had their say.
    narrowing_notes: tuple[str, ...] = ()
    # How the width ruling above was arrived at, which no count of drops can
    # say: an empty ``lane_drops`` is the answer to "run every lane", to "the
    # review never answered", and to "the block was there and unusable" alike.
    # ``not_asked`` covers a one-plan round, which is never held to a block, and
    # a review that never ran.
    narrowing_status: str = "not_asked"

    def __post_init__(self) -> None:
        if self.verdict not in {"ACCEPT", "REVISE", "REPLACE"}:
            raise ValueError("plan critic verdict is unsupported")
        _text(self.review, "plan critic review", allow_empty=True)
        _text(self.error, "plan critic error", allow_empty=True)
        if self.duration_sec < 0:
            raise ValueError("plan critic duration_sec must be non-negative")
        if self.verdict_source not in {"explicit", "inferred", "error"}:
            raise ValueError("plan critic verdict_source is unsupported")
        lane_ids = [drop.lane_id for drop in self.lane_drops]
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("plan critic lane_drops must name each lane once")
        for note in self.narrowing_notes:
            _text(note, "plan critic narrowing note")
        if self.narrowing_status not in {
            "not_asked",
            "answered",
            "repaired",
            "absent",
            "malformed",
        }:
            raise ValueError("plan critic narrowing_status is unsupported")

    @property
    def fail_open(self) -> bool:
        """Whether the draft bypassed enforcement because review failed."""
        return bool(self.error)

    @property
    def requires_revision(self) -> bool:
        return not self.error and self.verdict in {
            "REVISE",
            "REPLACE",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "CRITIC_ERROR" if self.error else "reviewed",
            "verdict": self.verdict,
            "error": self.error,
            "fail_open": self.fail_open,
            "duration_sec": self.duration_sec,
            "verdict_source": self.verdict_source,
            "lane_drops": [drop.to_dict() for drop in self.lane_drops],
            "narrowing_notes": list(self.narrowing_notes),
            "narrowing_status": self.narrowing_status,
        }

    def render_artifact(self) -> str:
        if self.error:
            return f"STATUS: CRITIC_ERROR\n\nERROR: {self.error}\n\nThe draft plan was used without critic enforcement."
        return self.review.strip()


@dataclass(frozen=True)
class LaneGround:
    """The ground one lane of a round owns, in the terms an edit lands in.

    A round is partitioned so two lanes never spend two Implementer sessions on
    the same change. What decides that is the code each lane will edit, not the
    specialist role its evidence came from: the roles are three readings of one
    kernel, so dividing by role divides nothing. ``ground`` therefore names
    files, functions and mechanisms.

    Only what a lane owns is recorded. What it must stay off is every other
    lane's ``ground``, derived at the point of use, so the two can never be
    written down as disagreeing descriptions of one boundary.

    ``joint`` marks the one shape that buys a wider ground than a region:
    a change and the launch configuration it invalidates, or a move that spans
    what no single region contains. It widens one lane and does not repeal the
    rule above: a launch site two bodies share is named in exactly one lane's
    ``ground``, and every other lane derives it as ground it does not own. Such
    a lane returns a gain that cannot
    be decomposed, so it pays for the width with ``fallback`` -- the smaller
    change inside the same ground that its Implementer lands if the joint step
    does not converge, so a lane that risks more cannot also risk measuring
    nothing.
    """

    lane_id: int
    ground: str
    reason: str = ""
    joint: bool = False
    fallback: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.lane_id, int) or isinstance(self.lane_id, bool):
            raise ValueError("lane ground.lane_id must be an integer")
        if self.lane_id < 1:
            raise ValueError("lane ground.lane_id must be positive")
        _text(self.ground, "lane ground.ground")
        _text(self.reason, "lane ground.reason", allow_empty=True)
        if not isinstance(self.joint, bool):
            raise ValueError("lane ground.joint must be a boolean")
        _text(self.fallback, "lane ground.fallback", allow_empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "ground": self.ground,
            "reason": self.reason,
            "joint": self.joint,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class SynthesizedPlan:
    """One generated plan, the ground it was planned on, and its session.

    ``ground`` is empty for a single-lane round, which is planned over the whole
    kernel and has no sibling to be bounded away from.

    ``joint`` and ``fallback`` are the lane's from :class:`LaneGround`, carried
    here because the steps that rule on a drafted lane -- the review's width
    ruling above all -- hold drafts and not grounds. Without them the review
    decided whether a lane was worth a session while blind to the fact that the
    lane was deliberately widened and to the smaller change it falls back to,
    and the round could not name, afterwards, which widened ground it had
    published nowhere.
    """

    text: str
    session_id: str = ""
    ground: str = ""
    joint: bool = False
    fallback: str = ""

    def __post_init__(self) -> None:
        _text(self.text, "synthesized plan")
        _text(
            self.session_id,
            "synthesized plan session_id",
            allow_empty=True,
        )
        _text(self.ground, "synthesized plan ground", allow_empty=True)
        if not isinstance(self.joint, bool):
            raise ValueError("synthesized plan joint must be a boolean")
        _text(self.fallback, "synthesized plan fallback", allow_empty=True)


@dataclass(frozen=True)
class OrchestrationRunResult:
    """Aggregate one best-effort planning cycle."""

    dispatch_plan: DispatchPlan
    specialist_outcomes: tuple[SpecialistOutcome, ...] = ()
    # Every lane's plan for this round, in lane order. A single-lane round
    # carries exactly one, which is the ordinary path.
    optimization_plans: tuple[str, ...] = ()
    structured_output_diagnostics: dict[str, Any] | None = None
    optimization_plan_executable: bool = True
    optimization_plan_draft: str = ""
    plan_critic: PlanCriticOutcome | None = None
    plan_revised: bool = False

    def __post_init__(self) -> None:
        for plan in self.optimization_plans or ("",):
            _text(plan, "orchestration optimization_plan")
        _text(
            self.optimization_plan_draft,
            "orchestration optimization_plan_draft",
            allow_empty=True,
        )
        if not isinstance(self.optimization_plan_executable, bool):
            raise ValueError("orchestration optimization_plan_executable must be a boolean")
        if not isinstance(self.plan_revised, bool):
            raise ValueError("orchestration plan_revised must be a boolean")

    @property
    def optimization_plan(self) -> str:
        """Lane 1's plan, which is the whole round on the single-lane path.

        Derived rather than stored: it was a second field holding a copy of
        ``optimization_plans[0]``, and the only thing two fields for one value
        can add is the chance of disagreeing.
        """
        return self.optimization_plans[0]
