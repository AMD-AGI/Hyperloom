"""Schema (TypedDict shape) for ``session_breakdown.json``.

Single contract between hyperloom (producer) and any downstream consumer
(dashboards, stats services, notebooks, KB ingest).

Design notes:
* All fields are NotRequired-by-convention: collectors return None/[]/{}
  when underlying artifacts are absent. Consumers treat missing data as
  "not available" — never fabricate values.
* JSON-serializable only; no dataclasses/enums in the wire shape.
* schema_version bumped on breaking changes only. New optional fields
  are additive.
"""

from __future__ import annotations

from typing import Any, TypedDict

SCHEMA_VERSION = "hyperloom.session_breakdown.v2"


# ---------------------------------------------------------------------------
# §1 Session metadata
# ---------------------------------------------------------------------------


class SessionMeta(TypedDict, total=False):
    session_id: str
    external_session_id: str | None
    sandbox_user_id: str | None
    created_at_utc: str
    ended_at_utc: str
    stop_reason: str
    max_minutes: int
    elapsed_minutes: float
    host: str
    code_revision: str
    pid: int
    session_dir: str
    tick_count: int
    agent_count: int
    image: str | None


# ---------------------------------------------------------------------------
# §2 Workload configuration
# ---------------------------------------------------------------------------


class WorkloadObjective(TypedDict, total=False):
    kind: str
    value: Any


class Workload(TypedDict, total=False):
    framework: str
    framework_version: str
    model_name: str
    model_path: str
    model_class: str
    gpu_type: str
    tp: int | None
    concurrency: int | None
    isl: int | None
    osl: int | None
    max_model_len: int | None
    precision: str
    objective: WorkloadObjective


# ---------------------------------------------------------------------------
# §3 Baseline
# ---------------------------------------------------------------------------


class BaselineAttemptSummary(TypedDict, total=False):
    ts: str
    task_id: str
    status: str
    decision: str
    key_metric: float | None
    workspace: str | None
    error_class: str | None


class BenchmarkInvocation(TypedDict, total=False):
    """Replayable record of how a benchmark variant was launched."""

    framework_args: str
    framework_args_source: str
    extra_envs: dict[str, str]
    config_path: str | None
    server_log_path: str | None


class Baseline(TypedDict, total=False):
    throughput_tok_s_per_gpu: float
    accuracy: float
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    ttft_e2el_source: str
    config_path: str | None
    benchmark_report_path: str | None
    attempts_history: list[BaselineAttemptSummary]
    failure_streak: int
    invocation: BenchmarkInvocation


# ---------------------------------------------------------------------------
# §4 Final state
# ---------------------------------------------------------------------------


class Final(TypedDict, total=False):
    throughput_tok_s_per_gpu: float | None
    cumulative_gain_pct_validated: float
    cumulative_gain_pct_per_round_sum: float
    validated_at_stack_len: int
    validated_ts: str
    stack_changed_after_validation: bool
    extra_args: str
    extra_envs: dict[str, Any]
    action_path: list[str]
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    ttft_e2el_source: str
    invocation: BenchmarkInvocation
    closing_phase_entered: bool
    closing_started_unix: float
    closing_report_task_id: str


# ---------------------------------------------------------------------------
# §5 Phase timeline
# ---------------------------------------------------------------------------


class PhaseEvent(TypedDict, total=False):
    ts: str
    action: str
    task_id: str
    kernel_id: str | None
    status: str
    decision: str
    key_metric: float | None
    key_metric_kind: str | None
    workspace: str | None
    error_class: str | None
    extras: dict[str, Any]


class PhaseSegment(TypedDict, total=False):
    phase: str
    from_phase: str
    entered_ts: str
    entered_unix: float | None
    exit_ts: str
    exit_reason: str
    evidence: dict[str, Any]
    actions: list[PhaseEvent]
    elapsed_seconds: float | None


# ---------------------------------------------------------------------------
# §6 Capability summary
# ---------------------------------------------------------------------------


class CapabilityEntry(TypedDict, total=False):
    status: str
    attempts: int
    keeps: int
    tested: int
    best_gain_pct: float | None
    reason: str
    keep_unstable_count: int
    winners_history: int


class CapabilitySummary(TypedDict, total=False):
    geak: CapabilityEntry
    oob: CapabilityEntry
    explore: CapabilityEntry
    backends: CapabilityEntry
    params: CapabilityEntry
    sweep: CapabilityEntry
    validate_stack: CapabilityEntry
    specialist: CapabilityEntry


# ---------------------------------------------------------------------------
# §7 Kernel invocations (GEAK / OOB)
# ---------------------------------------------------------------------------


class KernelMetadata(TypedDict, total=False):
    name: str
    source_file: str
    shapes: list[dict[str, Any]]
    gpu_pct: float | None
    arithmetic_intensity: float | None


class Invocation(TypedDict, total=False):
    kernel_id: str
    attempt_id: str
    run_id: str
    ts: str
    backend: str
    model: str | None
    kernel_metadata: KernelMetadata
    prompt_path: str | None
    optimized_files: list[str]
    result_path: str | None
    verification_path: str | None
    decision: str
    micro_speedup: float | None
    compile_passed: bool | None
    correctness_passed: bool | None
    best_artifact_path: str | None
    error: str | None
    cli_log_path: str | None


# ---------------------------------------------------------------------------
# §8 Kernel lifecycle
# ---------------------------------------------------------------------------


class DetectedKernel(TypedDict, total=False):
    kernel_id: str
    name: str
    gpu_pct: float | None
    time_ms: float | None
    bottleneck: str
    arithmetic_intensity: float | None
    reusable_native_kernel: bool
    source_file: str | None
    detected_from_task: str
    benchmark_report_path: str


class RecommendedKernel(TypedDict, total=False):
    kernel_id: str
    name: str
    gpu_pct: float | None
    recommended_backends: list[str]
    recommended_actions: list[str]
    bottleneck: str
    reusable_native_kernel: bool


class OptimizedKernel(TypedDict, total=False):
    kernel_id: str
    backend: str
    total_attempts: int
    successful_attempts: int
    best_micro_speedup: float | None
    last_decision: str
    best_artifact_path: str | None


class AdoptedKernel(TypedDict, total=False):
    kernel_id: str
    patch_path: str
    e2e_gain_pct: float | None
    validated: bool
    adopted_at: str


class RejectedKernel(TypedDict, total=False):
    kernel_id: str
    reason: str
    attempt_count: int
    best_gain_pct: float | None


class KernelLifecycle(TypedDict, total=False):
    detected: list[DetectedKernel]
    recommended: list[RecommendedKernel]
    optimized: list[OptimizedKernel]
    adopted: list[AdoptedKernel]
    rejected: list[RejectedKernel]


# ---------------------------------------------------------------------------
# §9 Explore / param search
# ---------------------------------------------------------------------------


class ParamSearchEntry(TypedDict, total=False):
    name: str
    fingerprint: str
    extra_args: str
    extra_envs: dict[str, Any]
    output_throughput: float | None
    gain_pct: float | None
    ts: str
    status: str


class ParamSearchLedger(TypedDict, total=False):
    schema_version: int
    tested_count: int
    accepted: list[ParamSearchEntry]
    rejected: list[ParamSearchEntry]
    top_by_gain: list[ParamSearchEntry]
    winner_history: list[dict[str, Any]]
    no_promote_streak: int


class ParamSearch(TypedDict, total=False):
    params: ParamSearchLedger
    backends: ParamSearchLedger
    synergy_attempted: list[str]
    discovered_flags: dict[str, Any]
    backend_winners_history: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# §10 Sweep
# ---------------------------------------------------------------------------


class SweepPoint(TypedDict, total=False):
    variant_name: str
    concurrency: int | None
    isl: int | None
    osl: int | None
    throughput_tok_s: float | None
    latency_mean_ms: float | None
    status: str


class Sweep(TypedDict, total=False):
    grid_size: int
    best_overall: dict[str, Any]
    pareto_front: list[dict[str, Any]]
    all_variants: list[SweepPoint]


# ---------------------------------------------------------------------------
# §11 Critic & robustness
# ---------------------------------------------------------------------------


class CriticIteration(TypedDict, total=False):
    iter: int
    ts: str
    topic: str
    verdict: str
    summary: str
    request_path: str
    judge_bundle_path: str
    emit_path: str
    review_path: str


class RobustnessSignal(TypedDict, total=False):
    ts: str
    signal: str
    action: str
    workdir: str


class CriticKBWritesSummary(TypedDict, total=False):
    total: int
    by_verdict: dict[str, int]


class CriticRobustness(TypedDict, total=False):
    critic_iterations: list[CriticIteration]
    robustness_signals: list[RobustnessSignal]
    kb_writes_summary: CriticKBWritesSummary


# ---------------------------------------------------------------------------
# §12 Telemetry
# ---------------------------------------------------------------------------


class GpuMonitorAggregate(TypedDict, total=False):
    samples: int
    avg_power_w: float
    max_power_w: float
    avg_temp_c: float
    max_temp_c: float
    avg_clock_mhz: float


class LaneTimelineEntry(TypedDict, total=False):
    lane: str
    capacity: int
    live_holders: int
    lease_expired_count: int


class Telemetry(TypedDict, total=False):
    gpu_count: int
    gpu_type: str
    baseline_report_path: str | None
    profile_report_paths: list[str]
    torch_trace_paths: list[str]
    system_profile_paths: list[str]
    server_log_paths: list[str]
    gpu_monitor_aggregate: GpuMonitorAggregate
    lane_timeline: list[LaneTimelineEntry]


# ---------------------------------------------------------------------------
# §13 Attribution
# ---------------------------------------------------------------------------


class StackGainEntry(TypedDict, total=False):
    ts: str
    action: str
    variant_name: str | None
    delta_pct: float | None
    cum_gain_after: float


class SourceBreakdown(TypedDict, total=False):
    geak_pct_of_total: float
    oob_pct_of_total: float
    framework_pct_of_total: float
    config_pct_of_total: float


class PhaseBreakdownExplore(TypedDict, total=False):
    gain_pct: float
    actions_count: int
    keeps_count: int


class PhaseBreakdownKernel(TypedDict, total=False):
    gain_pct: float
    actions_count: int
    keeps_count: int


class PhaseBreakdown(TypedDict, total=False):
    explore: PhaseBreakdownExplore
    kernel: PhaseBreakdownKernel
    unattributed: PhaseBreakdownExplore


class Attribution(TypedDict, total=False):
    gain_per_stack_entry: list[StackGainEntry]
    method: str
    source_breakdown: SourceBreakdown
    phase_breakdown: PhaseBreakdown
    notes: list[str]


# ---------------------------------------------------------------------------
# §14 KB Provenance
# ---------------------------------------------------------------------------


class KBPendingEdge(TypedDict, total=False):
    proposal_msg_id: str
    edge_id: str
    action: str
    ts: str


class KBQueueStats(TypedDict, total=False):
    pending_lines: int
    flushed_bookmarks: int
    dead_letter_lines: int


class KBCommitSummary(TypedDict, total=False):
    status: str
    promoted_edges: list[str]
    derived_summary_id: str


class KBPointCreated(TypedDict, total=False):
    canonical_id: str
    kind: str
    authority: str
    source: str
    status: str
    ts: str


class KBFlusherStatus(TypedDict, total=False):
    enabled: bool
    spawned: bool
    alive: bool
    pid: int | None
    cortex_kb_url: str | None
    interval_sec: float
    batch_size: int
    reason: str
    ts: str
    pid_path: str


class WarmReplayOutcome(TypedDict, total=False):
    status: str
    expected_gain_pct: float
    actual_gain_pct: float
    throughput_after: float
    warm_recipe_tier: str
    warm_recipe_conf: float
    replay_task_id: str
    error_class: str
    reason: str


class KBProvenance(TypedDict, total=False):
    cortex_session_id: str
    warm_start_ts: str
    warm_start_recipe_seen: bool
    warm_start_recipe_tier: str
    warm_start_pitfall_count: int
    warm_start_lesson_count: int
    warm_replay: WarmReplayOutcome
    warm_replay_attempted: bool
    warm_history_injected: bool
    stack_fingerprint: dict[str, str]
    pending_edges: list[KBPendingEdge]
    queue: KBQueueStats
    audit_tail_count: int
    audit_status_counts: dict[str, int]
    points_created: list[KBPointCreated]
    points_by_kind: dict[str, int]
    commit_summary: KBCommitSummary
    flusher_status: KBFlusherStatus
    kb_degraded_reason: str
    pr_degraded_reason: str


# ---------------------------------------------------------------------------
# §15 Specialist runs
# ---------------------------------------------------------------------------


class SpecialistDomainBreakdown(TypedDict, total=False):
    dispatched: int
    proposals_total: int
    proposals_kept: int
    proposals_rejected: int


class SpecialistTranscriptRef(TypedDict, total=False):
    task_id: str
    domain: str
    path: str
    body: str


class SpecialistRound(TypedDict, total=False):
    round_id: int
    dispatched_at: str
    completed_at: str
    domains: list[str]
    parallelism: int
    proposals_total: int
    proposals_kept: int
    proposals_rejected: int
    proposals_skipped: int
    kb_edge_ids: list[str]
    confidence_avg: float | None
    domain_breakdown: dict[str, SpecialistDomainBreakdown]
    transcripts: list[SpecialistTranscriptRef]
    notes: list[str]


# ---------------------------------------------------------------------------
# §16 Source files
# ---------------------------------------------------------------------------


class SourceFiles(TypedDict, total=False):
    session_dir: str
    manifest: str
    state: str
    baseline_report: str | None
    profile_reports: list[str]
    sweep_reports: list[str]
    kernel_attempts: list[str]
    agent_logs: list[str]
    patches: list[str]
    benchmark_reports: list[str]
    critic_workdir: str | None
    robustness_workdir: str | None


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------


class SessionBreakdown(TypedDict, total=False):
    schema_version: str
    exported_at_utc: str
    exporter_version: str

    session: SessionMeta
    workload: Workload
    baseline: Baseline
    final: Final

    phase_timeline: list[PhaseEvent]
    phase_segments: list[PhaseSegment]
    action_timeline: list[PhaseEvent]

    capability_summary: CapabilitySummary
    geak_invocations: list[Invocation]
    oob_invocations: list[Invocation]
    kernel_lifecycle: KernelLifecycle

    param_search: ParamSearch
    explore_search: ParamSearch
    sweep: Sweep

    critic_robustness: CriticRobustness
    telemetry: Telemetry
    attribution: Attribution
    kb_provenance: KBProvenance
    specialist_runs: list[SpecialistRound]

    warnings: list[str]
    source_files: SourceFiles


__all__ = [
    "SCHEMA_VERSION",
    "AdoptedKernel",
    "Attribution",
    "Baseline",
    "BaselineAttemptSummary",
    "BenchmarkInvocation",
    "CapabilityEntry",
    "CapabilitySummary",
    "CriticIteration",
    "CriticKBWritesSummary",
    "CriticRobustness",
    "DetectedKernel",
    "Final",
    "GpuMonitorAggregate",
    "Invocation",
    "KBCommitSummary",
    "KBFlusherStatus",
    "KBPendingEdge",
    "KBPointCreated",
    "KBProvenance",
    "KBQueueStats",
    "KernelLifecycle",
    "KernelMetadata",
    "LaneTimelineEntry",
    "OptimizedKernel",
    "ParamSearch",
    "ParamSearchEntry",
    "ParamSearchLedger",
    "PhaseBreakdown",
    "PhaseBreakdownExplore",
    "PhaseBreakdownKernel",
    "PhaseEvent",
    "PhaseSegment",
    "RecommendedKernel",
    "RejectedKernel",
    "RobustnessSignal",
    "SessionBreakdown",
    "SessionMeta",
    "SourceBreakdown",
    "SourceFiles",
    "SpecialistDomainBreakdown",
    "SpecialistRound",
    "SpecialistTranscriptRef",
    "StackGainEntry",
    "Sweep",
    "SweepPoint",
    "Telemetry",
    "WarmReplayOutcome",
    "Workload",
    "WorkloadObjective",
]
