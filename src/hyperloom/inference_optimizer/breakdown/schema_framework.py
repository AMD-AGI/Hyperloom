# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema (TypedDict shape) of the FRAMEWORK_AGENT timeline event in ``session_breakdown.json``."""

from __future__ import annotations

from typing import Any, TypedDict

from .schema import V6Failure


class V6FrameworkPolicyConfigArm(TypedDict, total=False):
    """The configuration arm's own thresholds."""

    keep_gain_threshold_pct: float | None
    empty_streak_threshold: int | None
    lookback: int | None


class V6FrameworkPolicySourceArm(TypedDict, total=False):
    """The source arm's own thresholds."""

    no_keep_streak_threshold: int | None
    discovery_retry_limit: int | None


class V6FrameworkPolicy(TypedDict, total=False):
    """The thresholds one entry ran under, as it resolved them.

    Recorded rather than cited, because a threshold read back at export is the
    one the session ended on and not the one this entry acted under."""

    keep_threshold_pct: float | None
    config: V6FrameworkPolicyConfigArm
    source: V6FrameworkPolicySourceArm


class V6FrameworkPlateauReading(TypedDict, total=False):
    """One plateau evaluation, beside the values it ruled on.

    Appended rather than keyed: two readings that agree are still two readings.
    ``path`` names the reader -- the advisory asks whether to switch arms and
    the exit whether the phase may leave -- and ``triggered`` of ``None`` is a
    reading that could not rule, which the phase treats as a live arm."""

    arm: str
    path: str
    evaluated_at: str
    triggered: bool | None
    inputs: dict[str, Any]
    thresholds: dict[str, Any]


class V6FrameworkRun(TypedDict, total=False):
    """One dispatch this entry made, and what it came back with.

    ``role`` is not derivable from ``arm``: the source arm dispatches twice per
    candidate, once to discover candidates and once to author a patch from one.
    ``produced_ids`` is projected at assembly from the proposals naming this
    run, since a run's own row does not know what it yielded."""

    run_id: str
    role: str
    arm: str
    status: str
    domain: str
    scope: str
    gap_canonical_id: str
    reason: str
    dispatched_at: str
    completed_at: str
    worktree: str
    summary: str
    tags: list[str]
    transcripts: list[str]
    new_findings: list[str]
    residual_questions: list[str]
    notes: list[str]
    parallelism: int | None
    proposals_total: int | None
    empty: bool | None
    confidence: float | None
    confidence_avg: float | None
    ensemble_scores: dict[str, Any]
    produced_ids: list[str]


class V6FrameworkCriticVariantRuling(TypedDict, total=False):
    """The Critic's ruling on one variant of a grid reviewed as a whole.

    A rejected variant reaches no bench, so no attempt row carries its
    ruling and this is the only place it is readable."""

    variant_name: str
    verdict: str
    effective_verdict: str
    held_to_rule: str
    reason: str
    failure_reason_code: str


class V6FrameworkCriticReviewOutcome(TypedDict, total=False):
    """What the loop did with a ruling.

    An ``advise`` that materialised and an ``advise`` held at the patch gate
    are the same ruling with opposite outcomes. ``patch_verdict_key`` is the
    subject the patch gate consults the ruling under, which connects a blocked
    ``integrate_patch`` back to the review that blocked it."""

    materialized: bool
    denied: bool
    reauthored: bool
    patch_verdict_key: str


class V6FrameworkCriticReview(TypedDict, total=False):
    """The Critic's ruling on one proposal, inline on the proposal.

    Both verdicts are kept: a reject the loop held to a rule that only declared
    ``advise`` is two facts, and either alone misreads the round. ``reviewer``
    separates a ruling the Critic authored from one it never got to make.
    The Critic's advisory keys are merged in as authored, so a consumer may
    meet fields beyond the ones named here."""

    verdict: str
    effective_verdict: str
    held_to_rule: str
    reviewer: str
    reason: str
    confidence: float | None
    failure_reason_code: str
    concerns: list[str]
    reviewed_at: str
    variants: list[V6FrameworkCriticVariantRuling]
    outcome: V6FrameworkCriticReviewOutcome
    artifacts: dict[str, Any]
    kb: dict[str, Any]


class V6FrameworkProposalTerminal(TypedDict, total=False):
    """Where a proposal ended up.

    ``dropped`` covers every way it never reached a measurement and ``reason``
    says which. A proposal the phase never resolved carries none."""

    disposition: str
    reason: str
    settled_at: str


class V6FrameworkLifecycleStep(TypedDict, total=False):
    """One step a proposal moved through, recorded as it happened.

    Recorded rather than derived from counters, so a candidate re-authored
    twice and then retried once reads as three steps."""

    step: str
    ts: str
    run_ref: str
    outcome: str
    reason: str


class V6FrameworkRenderedRef(TypedDict, total=False):
    """One Experience a KB read placed in the prompt that raised a proposal.

    Exposure, not evidence of use: the read rendered it, which says nothing
    about whether the model relied on it."""

    id: str
    purpose: str


class V6FrameworkPatch(TypedDict, total=False):
    """One session-local patch an authored attempt applied or reverted, verbatim."""

    path: str
    sha256: str
    content: str


class V6ExperienceCitation(TypedDict, total=False):
    """An Experience the deciding agent was shown and says shaped this attempt.

    ``stance`` is ``adopt``, ``adapt``, ``avoid``, or ``contrast``; ``claim`` is its reason, as it wrote it."""

    id: str
    stance: str
    claim: str


class V6FrameworkProposal(TypedDict, total=False):
    """One thing this entry pursued, whichever producer raised it.

    ``run_ref`` is absent rather than empty on a proposal with no dispatch
    behind it, which is the load-bearing fact for the producers that have no
    parent run at all: the orchestration agent and the seed grid.
    ``attempt_refs`` is projected at assembly from the attempts naming this
    proposal, because a second stored copy of the link is a second thing that
    can disagree."""

    proposal_id: str
    arm: str
    producer: str
    producer_ref: str
    run_ref: str
    domain: str
    scope: str
    lever_kind: str
    gap_canonical_id: str
    source_ref: str
    repo: str
    title: str
    verdict: str
    route: str
    changed_files: list[str]
    confidence: float | None
    reasoning: str
    kb_read_id: str
    rendered_refs: list[V6FrameworkRenderedRef]
    critic_review: V6FrameworkCriticReview
    terminal: V6FrameworkProposalTerminal
    lifecycle: list[V6FrameworkLifecycleStep]
    attempt_refs: list[str]


class V6FrameworkStack(TypedDict, total=False):
    """The configuration an attempt was measured on top of.

    Recorded rather than referenced: every KEEP advances the session's stack,
    so what the session serves now is not what this attempt was judged
    against. Both arms have one -- a source patch sits on whatever is being
    served, exactly as a config variant does."""

    throughput: float | None
    accuracy: float | None
    extra_server_args: str
    extra_envs: dict[str, Any]
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str | None


class V6FrameworkConfigDelta(TypedDict, total=False):
    """What one configuration variant changed about the serving command."""

    extra_server_args: str
    extra_envs: dict[str, Any]
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str | None


class V6FrameworkMeasurement(TypedDict, total=False):
    """Both ends of the throughput pair, and the runtime that produced them.

    The pair rather than the percentage alone, because the anchor advances on
    every KEEP and a gain without its denominator adds to nothing."""

    before_tput: float | None
    after_tput: float | None
    gain_pct: float | None
    runtime_sec: float | None
    estimated_output_throughput: float | None


class V6FrameworkAccuracy(TypedDict, total=False):
    """The accuracy gate's inputs and verdict.

    ``required`` of ``None`` is a gate that never ran, which is not the same as
    one that ran and failed."""

    required: bool | None
    reference: float | None
    value: float | None
    passed: bool | None


class V6FrameworkAttemptFailure(TypedDict, total=False):
    """Why one attempt yielded no measurement to judge."""

    error_class: str
    error_excerpt: str
    attribution: str


class V6FrameworkArtifacts(TypedDict, total=False):
    """Where one attempt's evidence was written."""

    workspace: str
    server_log_path: str
    raw_result_path: str


class V6FrameworkGate(TypedDict, total=False):
    """One gate's verdict on one attempt, as it was evaluated.

    A gate never reached writes no row, which is how a reader tells "did not
    pass" from "did not apply". ``passed`` of ``None`` is a gate that ran and
    could not rule. Order is the order of evaluation, not of the clock: a whole
    gating sequence fits inside one tick."""

    gate: str
    passed: bool | None
    reason: str
    observed: float | None
    threshold: float | None
    ts: str


class V6FrameworkAttempt(TypedDict, total=False):
    """One measured attempt, from either arm.

    One uniform row per thing measured, discriminated by ``arm``, so the
    adoption ledger walks both arms with one reader. Which fields carry
    still follows the arm -- a variant has a ``fingerprint`` and a
    ``config_delta``, an authored patch has a ``patch_path`` and the files it
    touched, and an authored deliverable of server args or envs has those as
    its ``config_delta`` -- but the lifecycle and the verdict are the same shape for both.
    ``blocked_by`` is projected at assembly as the first gate that did not
    pass."""

    attempt_id: str
    arm: str
    ts: str
    round_id: str
    task_id: str
    proposal_ref: str
    provenance: str
    outcome: str
    reason: str
    stage: str
    decision: str
    adopted: bool | None
    #: Whether the attempt reached the bench, whatever came of it; absent on a
    #: row recorded before the field existed.
    benched: bool | None
    attribution_eligible: bool | None
    validation_basis: str
    fingerprint: str
    variant_name: str
    candidate_id: str
    source_ref: str
    route: str
    patch_source: str
    patch_path: str
    patches_applied: list[str]
    patches_reverted: list[str]
    patch_material: list[V6FrameworkPatch]
    reasoning: str
    reasoning_origin: str
    experience_citations: list[V6ExperienceCitation]
    target_files: list[str]
    measured_against: V6FrameworkStack
    config_delta: V6FrameworkConfigDelta
    measurement: V6FrameworkMeasurement
    accuracy: V6FrameworkAccuracy
    failure: V6FrameworkAttemptFailure
    artifacts: V6FrameworkArtifacts
    gates: list[V6FrameworkGate]
    blocked_by: str | None


class V6FrameworkExit(TypedDict, total=False):
    """Why the entry left. Not a failure -- every entry that closes has one."""

    reason: str
    trigger: str
    hint: str
    switch_bottleneck: bool | None


class V6FrameworkExt(TypedDict, total=False):
    """``ext`` of the V6 ``framework_agent`` timeline event.

    Both OPTIMIZE arms in one event: the configuration arm searches server args
    and env vars, the source arm lands upstream patches, and the phase leaves
    only when both have run dry. The shape follows the progression a proposal
    moves along rather than the arm it belongs to -- ``proposals`` is the main
    line, ``runs`` holds the dispatch facts those rows reference, and
    ``attempts`` is one uniform row per thing measured -- because the arms
    differ in content and not in lifecycle. ``plateau`` sits off that
    progression.

    ``failure`` is absent on an entry that did not fail as a whole, which is
    not the same as one that failed at nothing: a failed attempt is on its
    attempt row, and a failed dispatch on its run row."""

    macro_cycle: int
    duration_sec: float | None
    policy: V6FrameworkPolicy
    plateau: list[V6FrameworkPlateauReading]
    runs: list[V6FrameworkRun]
    proposals: list[V6FrameworkProposal]
    attempts: list[V6FrameworkAttempt]
    exit: V6FrameworkExit
    failure: V6Failure | None


__all__ = [
    "V6ExperienceCitation",
    "V6FrameworkAccuracy",
    "V6FrameworkArtifacts",
    "V6FrameworkAttempt",
    "V6FrameworkAttemptFailure",
    "V6FrameworkConfigDelta",
    "V6FrameworkCriticReview",
    "V6FrameworkCriticReviewOutcome",
    "V6FrameworkCriticVariantRuling",
    "V6FrameworkExit",
    "V6FrameworkExt",
    "V6FrameworkGate",
    "V6FrameworkLifecycleStep",
    "V6FrameworkMeasurement",
    "V6FrameworkPatch",
    "V6FrameworkPlateauReading",
    "V6FrameworkPolicy",
    "V6FrameworkPolicyConfigArm",
    "V6FrameworkPolicySourceArm",
    "V6FrameworkProposal",
    "V6FrameworkProposalTerminal",
    "V6FrameworkRenderedRef",
    "V6FrameworkRun",
    "V6FrameworkStack",
]
