# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema (TypedDict shape) for ``session_breakdown.json``."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from ..session.sbd_v6 import SCHEMA_VERSION_V6

#: Unified optimization schema. This is a breaking wire-shape cutover: adopted
#: optimizations are emitted only through ``optimizations``, and that section
#: is built exclusively from recorder fragments -- never reconstructed from
#: session business files.
SCHEMA_VERSION_V5 = "hyperloom.session_breakdown.v5.0"

#: Current breakdown schema version. V6 stamps the document once the timeline
#: is recorded by the actions themselves rather than projected out of their
#: artefacts afterwards, which is what makes an event's start time its real one.
#: ``enablement`` gained its ledger-sourced round fields inside this version:
#: they add a section to the block rather than reshape the document.
SCHEMA_VERSION = SCHEMA_VERSION_V6


# Session metadata
class Recovery(TypedDict, total=False):
    """Crash / interruption / resume history for one optimization session."""

    recovered: bool
    crash_count: int
    crash_timestamps: list[str]
    degraded_mode: bool
    resume_pending_revalidation: bool
    last_tick_exception: dict[str, Any] | None


class SessionMeta(TypedDict, total=False):
    """Identity, timing, and host context for one optimization session."""

    session_id: str  # hyperloom internal id (manifest.session_id)
    claw_session_id: str | None  # SaFE / Claw session id (env CLAW_SESSION_ID)
    sandbox_user_id: str | None
    created_at_utc: str
    start_ts: str  # budget anchor; re-anchored only by a resume after a crash or a recorded stop
    ended_at_utc: str
    stop_reason: str  # target_reached / time_exhausted / global_converged / max_ticks / baseline_failed / ...
    max_minutes: int
    elapsed_minutes: float
    host: str
    code_revision: str
    pid: int
    session_dir: str
    user_data_path: str  # USER_DATA_PATH root the run wrote under
    tick_count: int
    image: str | None  # container image fully-qualified (or None if not configured)
    recovery: Recovery  # crash / interruption / resume history


# Workload configuration
class WorkloadObjective(TypedDict, total=False):
    """Optimization goal the session was asked to pursue."""

    kind: str  # gain_pct / tput / baseline / roofline_pct / time_only
    value: Any  # float or str (target_baseline_dir) or None
    objectives: list[dict[str, Any]]


class Workload(TypedDict, total=False):
    """Model, framework name, and serving configuration under optimization."""

    framework_name: str  # sglang / vllm / atom
    framework_version: str
    model_name: str
    model_path: str
    model_class: str
    gpu_type: str  # mi300x / mi325x / mi355x
    tp: int | None
    conc: int | None
    isl: int | None
    osl: int | None
    max_model_len: int | None
    precision: str
    objective: WorkloadObjective


# Model basics — architecture/scale summary parsed from the served model's config.json.
class ModelInfo(TypedDict, total=False):
    """Structural summary of the served model (architecture / scale / attention)."""

    model_family: str
    model_type: str
    architectures: list[str]
    attention_type: str
    is_moe: bool
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    vocab_size: int
    torch_dtype: str
    kv_cache_dtype: str
    quantization: str
    num_experts: int
    num_experts_per_tok: int


# Baseline
class BaselineAttemptSummary(TypedDict, total=False):
    """One recorded attempt to establish the baseline measurement."""

    ts: str
    task_id: str
    status: str
    decision: str
    key_metric: float | None
    workspace: str | None
    error_class: str | None
    # Real failure text from the executor; None on success / reconstruction.
    error_excerpt: str | None
    stderr_tail: str | None
    stderr_log_path: str | None
    extras: dict[str, Any]


class BenchmarkInvocation(TypedDict, total=False):
    """Replayable record of how a benchmark variant was launched (server cmd + envs + config)."""

    framework_args: str  # e.g. "python -m sglang.launch_server --model ... --tp 8"
    framework_args_source: str
    # vocab: log_non_default_args / log_args_line / log_python_cmd / yaml_cmd / yaml_benchmark / unknown.
    extra_envs: dict[str, str]  # allowlisted env vars only (no secrets)
    config_path: str | None  # baseline_config.with_envs.yaml or variant config
    server_log_path: str | None  # for debug


class Baseline(TypedDict, total=False):
    """Pre-optimization reference performance for the workload."""

    throughput_tok_s_per_gpu: float
    throughput_unit: str  # "tok/s" (serving) or "img/s" (scriptable xDiT)
    accuracy: float
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    ttft_e2el_source: str  # state_workspace / runs_baseline_disk / unavailable
    config_path: str | None
    benchmark_report_path: str | None
    attempts_history: list[BaselineAttemptSummary]
    failure_streak: int
    total_failures: int
    invocation: BenchmarkInvocation
    roofline_ceiling: dict[str, Any]


# Final state — SaFE contract core
class Final(TypedDict, total=False):
    """Final validated optimization state — the SaFE contract core."""

    throughput_tok_s_per_gpu: float | None
    throughput_unit: str  # "tok/s" (serving) or "img/s" (scriptable xDiT)
    cumulative_gain_pct_validated: float
    validated_at_stack_len: int
    validated_ts: str
    stack_changed_after_validation: bool
    extra_server_args: str
    extra_envs: dict[str, Any]
    action_path: list[str]  # ordered list of action:variant labels from optimization_stack
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    ttft_e2el_source: str  # current_best / current_best_disk / stack_top_disk / unavailable
    invocation: BenchmarkInvocation
    closing_phase_entered: bool
    closing_started_unix: float
    closing_report_task_id: str


# Phase timeline — chronological events
class PhaseEvent(TypedDict, total=False):
    """One chronological event in the optimization timeline."""

    ts: str
    action: str  # baseline / profile / explore / roofline / sweep / kernel_opt / integrate (+ archived: backends / params / validate_stack)
    task_id: str
    kernel_id: str | None  # only for kernel_agent-owned actions
    status: str  # succeeded / failed
    decision: str  # promoted / discarded / salvaged / no_promote / skipped / error / KEEP / PARTIAL / REVERT
    key_metric: float | None
    key_metric_kind: str | None
    workspace: str | None
    error_class: str | None
    phase: str  # declared phase (journal-sourced events); "" otherwise
    change: str  # human-readable change summary (journal) or action key
    extras: dict[str, Any]


# Capability summary — Capability cards in UI
class CapabilityEntry(TypedDict, total=False):
    status: str  # kept / reverted / tried / attempted / not_attempted / not_configured / failed / completed
    attempts: int  # invocation rows, NOT distinct kernels: how many tries
    keeps: int  # distinct kernels adopted at integrate (NOT micro-only KEEP)
    micro_only_keeps: int  # micro-KEPT kernels that never reached integrate
    pending_integrate: int  # micro-KEPT kernels whose integrate verdict is undecided
    reverts: int  # micro-KEPT kernels reverted at integrate (e2e regressed)
    e2e_gain_pct: float | None  # best end-to-end integrate gain for this lane's kernel
    tested: int  # for backends/params/explore: distinct variants tested
    best_gain_pct: float | None
    reason: str  # human readable, e.g. "geak backend only this run"
    # explore-specific:
    keep_unstable_count: int  # Pre-removal sessions only: KEEP'd variants a confirmation round evicted
    winners_history: int  # cumulative explore_search.winners_history length
    # specialist-row only — per-domain split keyed by SpecialistDomain.key; every catalogue domain is seeded
    # not_attempted for presence-free iteration.
    by_specialist: dict[str, "CapabilityEntry"]


class CapabilitySummary(TypedDict, total=False):
    """Per-capability roll-up powering the dashboard capability cards."""

    geak: CapabilityEntry
    forge: CapabilityEntry
    # primary explore row; backends/params/validate_stack are compat aliases.
    explore: CapabilityEntry
    backends: CapabilityEntry
    params: CapabilityEntry
    validate_stack: CapabilityEntry
    specialist: CapabilityEntry


# Kernel backend invocations
class KernelMetadata(TypedDict, total=False):
    """Descriptive metadata for a kernel targeted by a backend invocation."""

    name: str
    source_file: str
    shapes: list[dict[str, Any]]
    gpu_pct: float | None
    arithmetic_intensity: float | None


class Invocation(TypedDict, total=False):
    """One backend invocation for one kernel; ``backend`` identifies the engine."""

    kernel_id: str
    attempt_id: str
    run_id: str
    ts: str
    backend: str  # forge / geak / backend name
    model: str | None
    kernel_metadata: KernelMetadata
    prompt_path: str | None
    optimized_files: list[str]
    result_path: str | None
    verification_path: str | None
    decision: str  # KEEP / PARTIAL / REVERT / FAILED
    micro_speedup: float | None
    compile_passed: bool | None
    correctness_passed: bool | None
    best_artifact_path: str | None
    error: str | None
    cli_log_path: str | None


# Kernel lifecycle
class DetectedKernel(TypedDict, total=False):
    """A hot kernel surfaced by profiling (stage 1 of the kernel lifecycle)."""

    kernel_id: str
    name: str
    gpu_pct: float | None
    time_ms: float | None
    bottleneck: str  # compute / memory / comm
    arithmetic_intensity: float | None
    reusable_native_kernel: bool
    source_file: str | None
    detected_from_task: str  # which profile task_id surfaced it
    benchmark_report_path: str
    # lifecycle stamps (added by _collect_detected_kernels)
    selected_for_optimization: bool
    geak: dict[str, Any] | None  # {attempts, best_speedup, decision, last_status}
    adopted_by: str | None  # geak / forge / kernel_agent / None
    final_decision: str  # kept / reverted / rejected / attempted / not_optimized
    integrate_gain_pct: float | None  # e2e (integrate) gain; negative => regressed -> reverted


class RecommendedKernel(TypedDict, total=False):
    """A kernel recommended for optimization (stage 2 of the lifecycle)."""

    kernel_id: str
    name: str
    gpu_pct: float | None
    recommended_backends: list[str]
    recommended_actions: list[str]
    bottleneck: str
    reusable_native_kernel: bool


class OptimizedKernel(TypedDict, total=False):
    """A kernel that went through optimization (stage 3 of the lifecycle)."""

    kernel_id: str
    backend: str  # forge / geak / backend name (best-of)
    total_attempts: int
    successful_attempts: int
    best_micro_speedup: float | None
    last_decision: str
    best_artifact_path: str | None
    attempts_summary: list[dict[str, Any]]


class AdoptedKernel(TypedDict, total=False):
    """A kernel optimization adopted into the stack (stage 4 of the lifecycle)."""

    kernel_id: str
    patch_path: str
    target_file: str
    extra_server_args: str
    e2e_gain_pct: float | None
    validated: bool
    last_status: str
    adopted_at: str
    attempt_count: int
    basis: str
    alignment_status: str


class RejectedKernel(TypedDict, total=False):
    """A kernel optimization that was tried but not adopted (the +1 stage)."""

    kernel_id: str
    reason: str
    patch_path: str | None
    target_file: str | None
    attempt_count: int
    best_gain_pct: float | None
    ts: str


class KernelLifecycle(TypedDict, total=False):
    """Kernels grouped by lifecycle stage (4+1 stages)."""

    detected: list[DetectedKernel]
    recommended: list[RecommendedKernel]
    optimized: list[OptimizedKernel]
    adopted: list[AdoptedKernel]
    rejected: list[RejectedKernel]


# Kernel journey — kernel-major unified lifecycle view
class KernelToolMetadata(TypedDict, total=False):
    """Provenance for an external kernel tool (tracelens / geak / forge / kernel_agent)."""

    tool: str
    root_dir: str
    commit: str
    version: str


class DiscoveredHotKernel(TypedDict, total=False):
    """One hot kernel surfaced by a discovery run (projected onto the journey)."""

    kernel_id: str
    name: str
    gpu_pct: float | None
    time_ms: float | None
    bound_type: str
    arithmetic_intensity: float | None
    flops_per_byte: float | None
    efficiency_percent: float | None
    reusable_native_kernel: bool
    source_file: str | None
    recommended_backends: list[str]
    selected_for_optimization: bool


class KernelDiscoveryRun(TypedDict, total=False):
    """One hot-kernel discovery invocation (stage 1)."""

    source: str
    status: str
    ts: str
    duration_sec: float | None
    scan: dict[str, Any]
    hot_kernel_count: int
    hot_kernels: list[DiscoveredHotKernel]
    error: str | None


class KernelDispatch(TypedDict, total=False):
    """The dispatch decision for one kernel (stage 2)."""

    kernel_id: str
    dispatched: bool
    backends: list[str]
    skip_reason: str
    orchestration_commit: str
    task_group: str | None
    ts: str


class KernelBackendAttempt(TypedDict, total=False):
    """One backend attempt for one kernel (stage 3)."""

    kernel_id: str
    attempt_id: str
    run_id: str
    backend: str
    model: str | None
    ts: str
    status: str
    decision: str
    micro_speedup: float | None
    compile_passed: bool | None
    correctness_passed: bool | None
    correctness_source: str | None
    best_artifact_path: str
    optimized_files: list[str]
    error: str | None
    error_class: str | None
    duration_sec: float | None
    pre_dispatch_failure: bool


class KernelE2E(TypedDict, total=False):
    """The end-to-end integrate outcome for one kernel (stage 4)."""

    kernel_id: str
    integrated: bool
    e2e_gain_pct: float | None
    validated: bool | None
    decision: str
    patch_path: str | None
    target_file: str | None
    extra_server_args: str
    kernel_repo: str | None
    self_reported_e2e_gain_pct: float | None
    revalidation_measured_tput: float
    revalidation_current_best_tput: float
    revalidation_provenance: str
    rejection_reason: str
    ts: str


class KernelJourneyEntry(TypedDict, total=False):
    """One kernel's full lifecycle, joined across the four stages."""

    kernel_id: str
    name: str
    gpu_pct: float | None
    bound_type: str
    source_file: str | None
    micro_speedup: float | None
    discovery: DiscoveredHotKernel
    dispatch: KernelDispatch
    backend_attempts: list[KernelBackendAttempt]
    e2e: KernelE2E
    roofline: dict[str, Any]
    outcome: str


class KernelJourney(TypedDict, total=False):
    """Kernel-major unified lifecycle view."""

    discovery_runs: list[KernelDiscoveryRun]
    kernels: list[KernelJourneyEntry]


# Param search
class ParamSearchEntry(TypedDict, total=False):
    """One candidate variant from ``explore_search.{tested,accepted,rejected}``."""

    name: str
    fingerprint: str
    extra_server_args: str
    extra_envs: dict[str, Any]
    output_throughput: float | None
    gain_pct: float | None
    ts: str
    status: str  # accepted / rejected / tested
    operation_kind: str  # backend / param / env
    provenance: str
    proposer: str
    scope: str


class ParamSearchLedger(TypedDict, total=False):
    """Ledger of one explore family's tested/accepted/rejected variants."""

    schema_version: int
    tested_count: int
    accepted: list[ParamSearchEntry]
    rejected: list[ParamSearchEntry]
    top_by_gain: list[ParamSearchEntry]
    no_promote_streak: int


class ParamSearch(TypedDict, total=False):
    """Merged explore-search results across the param and backend families."""

    params: ParamSearchLedger
    backends: ParamSearchLedger
    synergy_attempted: list[str]
    discovered_flags: dict[str, Any]


class Geak(TypedDict, total=False):
    """GEAK e2e KERNEL-phase breakdown section."""

    engaged: bool
    status: str
    error_class: str | None
    error: str | None
    returncode: int | None
    # Same-harness adjudication, kept on the result because it is terminal state: ``geak_pending`` is cleared when the
    # verdict lands, and the final report still has to say why a measured candidate was dropped.
    revalidation_status: str | None
    revalidation_error_class: str | None
    revalidation_error: str | None
    recovered_from_disk: bool
    handoff: dict[str, Any] | None
    exp_root: str | None
    stages_reached: list[str]
    kernels_attempted: list[Any]
    opbench_results: list[Any]
    runner_log_tails: dict[str, str]
    likely_cause: str | None
    flushed_result_status: str | None
    last_artifact_ts: str | None
    baseline_throughput_tok_s: float | None
    final_throughput_tok_s: float | None
    throughput_speedup: float | None
    gain_pct: float | None
    metric_basis: str | None
    bench_client: str | None
    ttft_mean_ms: float | None
    tpot_mean_ms: float | None
    output_parity: str | None
    accepted_kernels: list[Any]
    accepted_kernels_source: str | None
    accepted_kernels_kind_sources: dict[str, int]
    accepted_heads: list[Any]
    kernels_optimized: int
    accepted_config: dict[str, Any]
    validated_regimes: list[Any]
    eval_dir: str | None
    report_path: str | None
    final_launch_script: str | None
    bench_script: str | None
    final_patch: str | None
    runner_timeout_s: int | None
    kill_timeout_s: int | None


# Critic / Robustness
class CriticIteration(TypedDict, total=False):
    """One critic-agent review pass over a proposed change."""

    iteration_id: str
    iter: int
    ts: str
    topic: str  # what was reviewed (kernel_opt:k001, backends:flag_X, ...)
    verdict: str  # approve / reject / redirect / advise / needs_review
    summary: str
    request_path: str
    judge_bundle_path: str
    emit_path: str
    review_path: str
    phase: str
    macro_cycle: int
    framework_reviews: list[dict[str, Any]]


class RobustnessSignal(TypedDict, total=False):
    """A fault/recovery event handled during the session."""

    ts: str
    signal: str  # crash / stall / disk_full / cluster_fault / ...
    action: str  # what was done
    workdir: str


class CriticRobustness(TypedDict, total=False):
    """Critic-review iterations and robustness signals for the session."""

    critic_iterations: list[CriticIteration]
    robustness_signals: list[RobustnessSignal]
    # KB writes proxied through the critic's ``commit-review`` protocol.
    kb_writes_summary: "CriticKBWritesSummary"


# Telemetry
class GpuMonitorAggregate(TypedDict, total=False):
    """Aggregated GPU power/thermal/clock telemetry over the session."""

    samples: int
    avg_power_w: float
    max_power_w: float
    avg_temp_c: float
    max_temp_c: float
    avg_clock_mhz: float


class LaneTimelineEntry(TypedDict, total=False):
    """One row of the lane occupancy summary (resource_lock capacity / live holders / expired leases)."""

    lane: str
    capacity: int
    live_holders: int
    lease_expired_count: int


class OrchestrationContext(TypedDict, total=False):
    """Orchestration turn count for this session.

    Attributes:
        tick_count (int): Coordinator ticks executed.
    """

    tick_count: int


class Telemetry(TypedDict, total=False):
    """Pointers to telemetry artifacts and aggregated hardware metrics."""

    baseline_report_path: str | None
    profile_report_paths: list[str]
    torch_trace_paths: list[str]
    system_profile_paths: list[str]
    server_log_paths: list[str]
    gpu_monitor_aggregate: GpuMonitorAggregate
    # per-lane capacity / occupancy summary.
    lane_timeline: list[LaneTimelineEntry]
    orchestration_context: OrchestrationContext


# Attribution
class StackGainEntry(TypedDict, total=False):
    """One KEEP/validation event with its incremental gain contribution."""

    ts: str
    stack_len_before: int
    stack_len_after: int
    action: str  # backends / params / kernel_opt:<kid> / validate_stack
    variant_name: str | None
    cum_gain_before: float
    cum_gain_after: float
    delta_pct: float | None  # None when validate_stack re-baselined
    extra_server_args: str


class SourceBreakdown(TypedDict, total=False):
    """Validated total gain split by contributing source/family."""

    geak_pct_of_total: float
    # primary explore family bucket.
    explore_pct_of_total: float
    # REPLAY_WARM_RECIPE (warm-recipe / recipe KB best_config replay) contribution.
    replay_warm_recipe_pct_of_total: float
    # FRAMEWORK_AGENT phase contribution (upstream-PR bake-ins).
    framework_pct_of_total: float
    # GEMM_TUNING (deterministic FP8 GEMM tuner) gain; always emitted (0.0 when skipped / no KEEP).
    gemm_tuning_pct_of_total: float
    # Kernel-lane gain with no backend KEEP evidence; unattributed on purpose.
    kernel_unattributed_pct_of_total: float
    # Gain whose owning source could not be resolved.
    unattributed_pct_of_total: float
    backends_pct_of_total: float
    params_pct_of_total: float
    sweep_pct_of_total: float
    validated_total_pct: float


class PhaseBreakdownExplore(TypedDict, total=False):
    """Explore-phase gain split by specialist domain."""

    total_gain_pct: float
    by_domain: dict[str, float]
    by_scope: dict[str, float]


class PhaseBreakdownKernel(TypedDict, total=False):
    """Kernel-phase gain split by ``kernel_id``."""

    total_gain_pct: float
    by_kernel_id: dict[str, float]


class PhaseBreakdownFramework(TypedDict, total=False):
    """FRAMEWORK_AGENT phase gain split by adopted PR; ``by_pr`` keys on ``variant_name`` (``"?"`` when empty)."""

    total_gain_pct: float
    by_pr: dict[str, float]


class PhaseBreakdownGemmTuning(TypedDict, total=False):
    """KERNEL-entry FP8 GEMM tuning gain split by ``tuned_file`` (falls back to ``variant_name`` then ``"?"``)."""

    total_gain_pct: float
    by_tuned_file: dict[str, float]


class PhaseBreakdownGeak(TypedDict, total=False):
    """GEAK e2e gain, split by what was actually running when it was measured."""

    total_gain_pct: float
    by_contribution: dict[str, float]
    by_kernel_id: dict[str, float]


class PhaseBreakdown(TypedDict, total=False):
    """Per-phase gain attribution."""

    prelude: PhaseBreakdownExplore  # always 0 by definition
    enablement: PhaseBreakdownExplore  # always 0 (graded on runnability, not throughput)
    framework: PhaseBreakdownFramework
    explore: PhaseBreakdownExplore
    kernel_agent: PhaseBreakdownKernel
    gemm_tuning: PhaseBreakdownGemmTuning
    geak: PhaseBreakdownGeak
    sweep: PhaseBreakdownExplore  # usually 0 (sweep is measurement)
    close: PhaseBreakdownExplore  # usually 0
    unattributed: PhaseBreakdownExplore  # gain whose phase couldn't be inferred


class Attribution(TypedDict, total=False):
    """Gain attribution across stack entries, sources, and phases."""

    gain_per_stack_entry: list[StackGainEntry]
    # validated / single_source / reconstructed / missing
    method: str
    source_breakdown: SourceBreakdown
    phase_breakdown: PhaseBreakdown
    lever_breakdown: dict[str, float]
    notes: list[str]  # human-readable caveats


# Phase segments — phase state machine
class PhaseSegment(TypedDict, total=False):
    """One contiguous segment of the phase state machine."""

    phase: str  # PRELUDE / FRAMEWORK_AGENT / KERNEL_AGENT / SWEEP / CLOSE
    from_phase: str  # previous phase (empty for first segment)
    entered_ts: str  # iso UTC of entry
    entered_unix: float | None
    exit_ts: str  # iso UTC of next transition; "" for current segment
    exit_unix: float | None  # unix epoch of next transition; None for current segment
    exit_reason: str  # transition reason vocab entry; "" for current segment
    evidence: dict[str, Any]  # entry evidence (snapshot at transition time)
    events: list[dict[str, Any]]  # non-transition sub-events folded into this phase
    actions: list[PhaseEvent]  # phase_timeline events attributed to this phase
    elapsed_seconds: float | None


# specialist_runs section
class SpecialistDomainBreakdown(TypedDict, total=False):
    """Per-domain attribution for one ``specialist_rounds`` entry."""

    dispatched: int
    proposals_total: int
    proposals_kept: int
    proposals_rejected: int


class SpecialistTranscriptRef(TypedDict, total=False):
    """Reference to a specialist transcript on disk (path only by default; ``body`` inlined when the flag is set)."""

    task_id: str
    domain: str
    path: str
    body: str  # only set when CLI flag enabled


class SpecialistRound(TypedDict, total=False):
    """One element of ``specialist_runs``."""

    # round_id: numeric counter / "explore-NNN" / task-id hash, coerced numeric when possible.
    round_id: int | str
    dispatched_at: str
    completed_at: str
    domains: list[str]
    parallelism: int
    proposals_total: int
    proposals_kept: int
    proposals_rejected: int
    proposals_skipped: int
    confidence_avg: float | None
    domain_breakdown: dict[str, SpecialistDomainBreakdown]
    transcripts: list[SpecialistTranscriptRef]
    notes: list[str]


# critic_robustness.kb_writes_summary sub-block
class CriticKBWritesSummary(TypedDict, total=False):
    """Summary of critic-agent ``commit-review`` outputs."""

    total: int
    by_verdict: dict[str, int]  # APPROVE / REJECT / REDIRECT / ADVISE / NEEDS_REVIEW (upper-cased critic verdicts)


# Top-level shape
class SourceFiles(TypedDict, total=False):
    """Paths to the on-disk artifacts the breakdown was assembled from."""

    manifest: str
    state: str
    baseline_report: str | None
    profile_reports: list[str]
    sweep_reports: list[str]
    kernel_attempts: list[str]
    critic_workdir: str | None
    robustness_workdir: str | None


# Roofline — optimization-progress curve for the dashboard: a stepped line from baseline through every KEEP against
# ceiling/target reference lines, all derived from ``state.json``.
class RooflineTrajectoryPoint(TypedDict, total=False):
    """One x/y/tooltip on the optimization-progress curve (first point is ``baseline``, rest from the KEEP stack)."""

    ts: str  # iso UTC, x value
    tput: float  # tok/s, y value
    label: str  # "baseline" / variant_name
    action: str  # "baseline" / "explore" / "kernel_opt" / ...
    gain_pct: float  # cumulative gain vs baseline at this point
    flags: str  # candidate_extra_server_args
    extra_envs: dict[str, str]  # KEY=value pairs the variant set


class RooflineSnapshot(TypedDict, total=False):
    """One ``state.roofline_snapshots[]`` entry mirrored verbatim (on-disk shape, so new fields flow through)."""

    snapshot_id: int
    ts: str
    achieved_tok_per_sec: float
    theoretical_peak_tok_per_sec: float  # ceiling, vendor peak (unreachable)
    within_roofline_pct: float  # achieved / peak * 100, capped at 100
    gap_to_roofline_pct: float
    within_roofline_pct_uncapped: float | None  # uncapped ratio; >100 = wrong ceiling
    roofline_ceiling_exceeded: bool
    compute_pct: float
    idle_pct: float
    comm_pct: float
    top_bottleneck: str  # "MoE_unfused" etc
    top_kernel: dict[str, Any]  # {name, bound_type, efficiency_pct, gpu_pct}
    analysis_md_path: str
    kernel_roofline_path: str
    trace_input: str


class RooflineProgress(TypedDict, total=False):
    """Top-level ``roofline_progress`` section."""

    # Reference lines (only set when snapshots[] is non-empty)
    ceiling_tok_per_sec: float | None
    target_tok_per_sec: float | None
    ceiling_ratio_target: float  # default 0.70
    ceiling_available: bool
    snapshot_top_bottleneck: str  # tooltip on the ceiling line
    snapshot_within_roofline_pct: float
    snapshot_gap_to_roofline_pct: float

    # Trajectory
    trajectory: list[RooflineTrajectoryPoint]

    # Headline numbers surfaced so the dashboard's "current" callout needn't recompute them.
    baseline_tput: float
    current_best_tput: float
    cumulative_gain_pct: float
    current_best_pct_of_ceiling: float | None  # tput/ceiling*100, None when no ceiling
    current_best_pct_of_target: float | None  # tput/target*100, None when no ceiling

    # Audit / staleness
    roofline_failure_streak: int  # consecutive watermark roofline failures
    snapshots: list[RooflineSnapshot]


# Optimization stack — raw KEEP ledger passthrough
class OptimizationStackEntry(TypedDict, total=False):
    """One KEEP from ``state.optimization_stack[]`` exposed verbatim."""

    action: str
    variant_name: str
    candidate_extra_server_args: str
    extra_envs: dict[str, str]
    tput: float | None
    ts: str
    workspace: str | None
    validated: bool
    # gemm_tuning evidence; engine is the tuning provenance ("geak" / "forge").
    engine: str
    tuned_file: str
    final_report_path: str
    source: str
    # generic optionals
    gain_pct: float | None
    kernel_id: str
    fingerprint: str
    provenance: str
    task_id: str
    source_phase: str
    # filter label for the kind of optimization (backend / param / env on explore KEEPs); specialist dial.
    operation_kind: str
    scope: str
    accepted_heads: list[Any]
    extra_server_args_is_invariant: bool
    candidate_flags: Any


# Canonical optimizations — single downstream read model
OptimizationSource = Literal[
    "warm_replay",
    "explore",
    "framework_agent",
    "kernel_agent",
    "unattributed",
]
OptimizationSourceMethod = Literal[
    "recorded",
    "phase_history_ts",
    "action_family",
    "unknown",
]
KernelOptimizationBackend = Literal["geak", "forge"]
KernelExecutionMode = Literal["whole_pipeline", "per_kernel"]


class OptimizationArtifact(TypedDict, total=False):
    """One artifact attached to a canonical optimization entry."""

    kind: str
    path: str


class OptimizationConfiguration(TypedDict, total=False):
    """Effective serving configuration carried by one optimization."""

    extra_server_args: str
    extra_envs: dict[str, str]


class OptimizationEntry(TypedDict, total=False):
    """One adopted optimization's contribution to the session's reported gain."""

    id: str
    stack_index: int
    adopted_attempt_id: str | None
    adoption_id: str | None
    source: OptimizationSource
    source_method: OptimizationSourceMethod
    optimization_kind: str
    name: str
    backend: KernelOptimizationBackend | None
    # Gain against the session baseline.
    gain_pct: float | None
    gain_method: str
    # The adopting executor's own measurement, relative to its starting point.
    local_gain_pct: float | None
    cumulative_gain_pct: float | None
    throughput_after: float | None
    validated: bool
    ts: str
    execution_mode: KernelExecutionMode | None
    kernel_id: str | None
    action: str
    variant_name: str
    fingerprint: str
    scope: str
    source_phase: str
    accepted_heads: list[Any]
    extra_server_args_is_invariant: bool | None
    candidate_flags: Any
    throughput_before: float | None
    task_id: str
    provenance: str
    configuration: OptimizationConfiguration
    artifacts: list[OptimizationArtifact]


class OptimizationSourceSummary(TypedDict, total=False):
    """Validated KEEP count and gain for one canonical source."""

    keeps: int
    total_gain_pct: float
    by_backend: dict[str, "OptimizationSourceSummary"]


class OptimizationBackendAttempt(TypedDict, total=False):
    """One ordered backend attempt, including non-adopted outcomes."""

    attempt_id: str
    run_id: str
    kernel_id: str
    backend: str
    decision: str
    sequence: int
    ts: str
    duration_sec: float | None
    micro_speedup: float | None
    compile_passed: bool | None
    correctness_passed: bool | None
    error_class: str | None
    error: str | None
    result_path: str | None
    verification_path: str | None


class OptimizationValidation(TypedDict, total=False):
    """Attribution trust, reconciliation, and diagnostic metadata."""

    method: str
    validated_at_stack_len: int
    validated_total_gain_pct: float | None
    attributed_total_gain_pct: float
    # Gain the session really moved that no adopted step accounts for, most often a KEEP that never reached the
    # ledger.
    unattributed_gain_pct: float
    attribution_gap_pct: float | None
    notes: list[str]
    source_breakdown: dict[str, float]
    phase_breakdown: dict[str, Any]
    domain_attribution: dict[str, Any]


class OptimizationAttemptGate(TypedDict, total=False):
    """One gate the attempt had to clear, as evaluated at author time."""

    kind: str
    name: str
    status: str
    decision: str
    reason: str


class OptimizationAttempt(TypedDict, total=False):
    """One attempt at making the workload faster, adopted or not."""

    attempt_id: str
    agent: AgentBucket
    # ``recorded`` when the producer stamped the owner, ``derived`` when it was reconstructed for a session recorded
    # before that field existed.
    agent_method: str
    producer: str
    kind: str
    name: str
    subject: dict[str, str]
    kernel_id: str | None
    backend: str
    phase: str
    macro_cycle: int | None
    started_at: str
    ended_at: str
    duration_sec: float | None
    status: str
    decision: str
    decision_reason: str
    keep_threshold_pct: float | None
    adopted: bool
    attribution_eligible: bool | None
    # Measured against this attempt's own starting point, not the session baseline.
    local_gain_pct: float | None
    throughput_before: float | None
    throughput_after: float | None
    adoption_id: str | None
    gates: list[OptimizationAttemptGate]
    backend_attempts: list[OptimizationBackendAttempt]
    # Each row carries ``occurrence``, its position among this operation's readings of that metric name, oldest first,
    # along with ``occurrences_of_name`` for how many there are in total.
    measurements: list[dict[str, Any]]
    # ``adoption_pinned`` when the adoption named the readings it was decided on, ``latest_occurrence`` when the
    # newest reading of each metric was used for want of one, ``adoption_pinned_stale`` when the pinned readings were
    # overwritten by a later re-measure and no longer match the frozen decision values.
    measurement_source: str
    measurement_occurrences: int
    artifacts: list[dict[str, str]]


class OptimizationAgentSummary(TypedDict, total=False):
    """Per-agent rollup: the top layer of the optimization report."""

    attempts: int
    keeps: int
    reverts: int
    attributable_gain_pct: float
    non_attributable_keeps: int
    by_kind: dict[str, dict[str, Any]]


class Optimizations(TypedDict, total=False):
    """Canonical downstream optimization API."""

    schema_version: int
    # ``recorder`` when projected from author-time records, ``state`` when rebuilt from business state for a session
    # that predates the recorder.
    source_of_truth: str
    attempts: list[OptimizationAttempt]
    entries: list[OptimizationEntry]
    backend_attempts: list[OptimizationBackendAttempt]
    summary_by_agent: dict[AgentBucket, OptimizationAgentSummary]
    summary_by_source: dict[OptimizationSource, OptimizationSourceSummary]
    summary_by_kind: dict[str, OptimizationSourceSummary]
    validation: OptimizationValidation
    gemm_tuning_runs: list["GemmTuningRun"]


# GEMM tuning — fixed FP8 block-scale GEMM tuning stage that runs at KERNEL entry.
class GemmTuningRun(TypedDict, total=False):
    """One GEMM-tuning run, keyed by the produced ``tuned_file`` CSV."""

    engine: str
    status: str
    decision: str
    source: str
    ts: str
    duration_sec: float | None
    error_class: str
    error: str
    precision: str
    framework: str
    gpu_type: str
    tp: int
    conc: int
    isl: int
    osl: int
    libtype: str
    baseline_tput: float | None
    best_speedup: float | None
    gain_pct: float | None
    tuned_tput: float | None
    tuned_file: str
    final_report_path: str
    workspace: str
    adopted: bool
    summary: dict[str, Any]
    parameters: dict[str, Any]
    candidates: list[dict[str, Any]]
    shapes: list[dict[str, Any]]


class GemmTuning(TypedDict, total=False):
    """Top-level GEMM-tuning section envelope."""

    runs: list[GemmTuningRun]
    adopted_engine: str
    adopted_tuned_file: str
    total_gain_pct: float


# Kernel Roofline — hot-kernel table mirroring reports/kernel_roofline.json.
class KernelRooflineEntry(TypedDict, total=False):
    """One hot-kernel row (on-disk shape passed through verbatim)."""

    kernel_id: str  # zero-padded rank ordinal, ``k001``.. (one per candidate; pool size = --top-k, default 100)
    name: str  # ``aiter::ck_moe_stage1`` etc
    source_file: str  # absolute path; "" when unknown
    kernel_category: str  # ``MoE`` / ``LayerNorm`` / ``unknown``
    bound_type: str  # ``memory-bound`` / ``compute-bound``
    arithmetic_intensity: float
    flops_per_byte: float
    efficiency_percent: float  # kernel-self efficiency 0..100
    gpu_pct: float  # share of overall GPU time 0..100
    call_count: int
    duration_us: float
    reusable_native_kernel: bool  # True ⇒ GEAK can swap in a custom kernel


class KernelRoofline(TypedDict, total=False):
    """Top-level ``kernel_roofline`` section (loaded from the report; empty ``{}`` on missing/malformed)."""

    schema_version: int  # tracelens output schema (currently 1)
    source: str  # provenance label, e.g. ``tracelens_analysis``
    analysis_md_path: str  # absolute path to the human-readable analysis
    kernel_candidates_path: str  # absolute path to kernel_candidates.json
    trace_input: str  # absolute path to the trace dir
    trace_input_type: str  # ``capture_dir`` / ``trace_file`` / ...
    kernels: list[KernelRooflineEntry]


# Kernel Optimization Summary — mirror of ``reports/kernel_optimization_summary.json``, passed through verbatim;
# ``by_kernel[]`` rows stay loose.
class KernelOptimizationSummary(TypedDict, total=False):
    schema_version: int  # producer schema (currently 1; int, unlike conc_sweep's str)
    session_id: str  # global id ``{model}_{ts}_{short_uuid}``
    model_name: str
    cumulative_gain_validated_pct: float
    totals: dict[str, int]  # {attempted, integrated, keep_pending, rejected, in_flight}
    rejection_breakdown: dict[str, int]
    failure_reason_breakdown: dict[str, int]
    field_glossary: dict[str, str]  # {field_name: explanation} for tooltips
    top_takeaways: list[str]  # 2-4 deterministic (non-LLM) sentences
    by_kernel: list[dict[str, Any]]  # one row per top kernel, sorted gpu_pct desc
    report_path: str  # rel-to-session path to the mirrored source report


# Conc Sweep Summary — mirror of ``reports/conc_sweep_summary.json``, a baseline-vs-current_best curve across a CONC
# ladder.
class ConcSweepSummary(TypedDict, total=False):
    schema_version: str  # producer schema (currently "1.0"; str, unlike kernel summary's int)
    status: str  # succeeded / failed / skipped
    skip_reason: str  # only when status="skipped"
    session_id: str
    isl: int
    osl: int
    tp: int
    benchmark_mode: str  # "agentx" / "synthetic"; names the axis pair the points carry
    concs_requested: list[int]
    # {extra_server_args, extra_envs, points[]}. A point carries the pair its mode is plotted on:
    # output_throughput + e2el_mean_ms synthetic, total_token_throughput + e2e_norm_intvty_p90 agentic.
    baseline: dict[str, Any]
    optimized: dict[str, Any]
    comparison: list[dict[str, Any]]  # per-CONC paired rows (feeds the dual curve + speedup bars)
    # {metric, successful_pairs, failed_pairs, best_conc, best_speedup, median_speedup, mean_speedup}.
    summary: dict[str, Any]
    workspace: str
    elapsed_sec: float
    total_budget_sec: int  # None when budget gate disabled
    budget_exhausted: bool
    budget_skip_reason: str  # why budget-gated variants were skipped, when budget_exhausted=true
    budget_remaining_sec: float
    report_json_path: str
    report_csv_path: str  # for the "download CSV" button
    roofline_ceiling: dict[str, Any]  # per-CONC theoretical peak + MBU%; may be absent on old products
    report_path: str  # rel-to-session path to the mirrored source report


# Full-trace: unified token + decision timeline
class TokenBucket(TypedDict, total=False):
    """Aggregated token counters for one grouping (phase / component / total)."""

    total_in: int
    total_out: int
    total_cache_creation: int
    total_cache_read: int
    total_reasoning_out: int
    calls: int


class DecisionTokens(TypedDict, total=False):
    by_component: dict[str, TokenBucket]  # component -> its token bucket for this decision
    total_in: int
    total_out: int
    total_cache: int  # cache_creation + cache_read
    total_reasoning_out: int  # hidden reasoning output, not part of total_out
    calls: int


class DecisionTraceEntry(TypedDict, total=False):
    phase: str  # phase active at the decision (declared or ts-window backfill)
    tick: int | None  # orchestrator tick (None when the producer didn't stamp one)
    ts: str  # ISO ...Z of the decision
    # ``decision`` carries proposer attribution + a filter label: {component (resolved proposer: specialist:<domain> /
    # grid / orchestration), operation_kind (backend / param / env / kernel_opt / kernel_integrate / ...),
    # change/event/verdict, outcome, gain_pct, task_id/dyn_id, kind, provenance, scope, fingerprint, metrics}
    decision: dict[str, Any]
    tokens: DecisionTokens


class TokenRollup(TypedDict, total=False):
    by_phase: dict[str, TokenBucket]  # phase -> aggregate token bucket
    by_component: dict[str, TokenBucket]  # component -> aggregate token bucket
    session_total: TokenBucket  # whole-session token total


class DecisionTrace(TypedDict, total=False):
    """The joined token+decision timeline plus its rollups."""

    decision_trace: list[DecisionTraceEntry]
    token_rollup: TokenRollup
    unattributed_tokens: TokenBucket
    # Inherently cross-decision LLM spend (orchestration / critic / robustness reactor turns) with no single owning
    # decision — kept separate from ``unattributed_tokens`` (a genuine attribution gap).
    overhead_tokens: TokenBucket


# Token usage — promoted, discoverable top-level rollup of LLM token spend.
class TokenUsageBucket(TypedDict, total=False):
    """A token bucket plus two convenience totals for at-a-glance reading."""

    total_in: int
    total_out: int
    total_cache_creation: int
    total_cache_read: int
    total_cache: int
    total_reasoning_out: int
    calls: int
    total_in_out: int
    grand_total: int
    # cache_read / (cache_creation + cache_read); 0.0 when no split cache data.
    cache_hit_rate: float


class TokenUsageAttribution(TypedDict, total=False):
    """How much of the session token spend ties back to a decision."""

    attributed_to_decisions: TokenUsageBucket
    overhead: TokenUsageBucket
    unattributed: TokenUsageBucket
    attributed_calls_pct: float
    overhead_calls_pct: float


class TokenUsageTimelineEntry(TypedDict, total=False):
    """One ``action_timeline`` row annotated with the tokens tied to it."""

    task_id: str | None
    action: str
    phase: str
    decision: str
    ts: str
    tokens: TokenUsageBucket | None


class TokenUsage(TypedDict, total=False):
    """Top-level, discoverable LLM-token-spend summary for the session."""

    session_total: TokenUsageBucket
    by_component: dict[str, TokenUsageBucket]
    by_phase: dict[str, TokenUsageBucket]
    attribution: TokenUsageAttribution
    timeline: list[TokenUsageTimelineEntry]
    source: str
    correlation: str


# Langfuse push receipt — was the trace mirrored live to Langfuse?
class LangfuseConfig(TypedDict, total=False):
    """Redacted Langfuse connection config that was in effect this session."""

    enable_flag: bool
    host: str | None
    public_key_set: bool
    secret_key_set: bool
    sdk_available: bool


class LangfusePushCounts(TypedDict, total=False):
    """How many observations the live push actually emitted this session."""

    generations_sent: int
    generations_paired: int
    generations_text_only: int
    generations_token_only: int
    scores_sent: int
    spans_opened: int
    errors: int


class LangfusePush(TypedDict, total=False):
    """Receipt of whether/where/how much the session was pushed to Langfuse."""

    enabled: bool
    disabled_reason: str | None
    config: LangfuseConfig
    trace_id: str | None
    session_id: str | None
    correlated_on: str
    counts: LangfusePushCounts
    counts_final: bool
    receipt_source: str


class EnablementRoundSummary(TypedDict, total=False):
    """One bring-up round, as the durable round ledger recorded it.

    Attributes:
        round_id: Identity of the round.
        state: ``open`` while a holder has it, ``settled`` once it ended.
        outcome: How it ended -- booted / failed / abandoned, or one of the two
            expiries. Empty while it is open.
        holder_task_id: The task holding it.
        fence: The holder's token; only a handoff advances it.
        opened_unix: When the round was acquired.
        settled_unix: When it ended, or ``None`` while it is open.
    """

    round_id: str
    state: str
    outcome: str
    holder_task_id: str
    fence: int
    opened_unix: float
    settled_unix: float | None


class EnablementStackActionSummary(TypedDict, total=False):
    """One attempt-runtime stack action considered/applied."""

    kind: str
    framework: str
    capability: str
    acquisition_method: str
    repo_url: str
    ref: str
    index_url: str
    reason: str


class EnablementAttemptRuntime(TypedDict, total=False):
    """One provisioned attempt runtime (promoted or discarded)."""

    venv_root: str
    bin_path: str
    python_path: str
    installed_versions: dict[str, str]
    promoted: bool


class TargetedBuildAttemptSummary(TypedDict, total=False):
    """One targeted-build attempt (AITER / sgl-kernel / vLLM-source)."""

    component: str
    ref: str
    gpu_arch: str
    max_jobs: int
    ok: bool
    failure_class: str
    failure_summary: str
    installed_versions: dict[str, str]
    build_probes: list[str]
    build_log_path: str
    attempt_root: str


class EnablementBreakdown(TypedDict, total=False):
    """Enablement subsystem observability section."""

    mode: str
    engaged: bool
    origin: str
    attempts: int
    dispatched: bool
    succeeded: bool
    pending: bool
    validation_pending: bool
    round_id: str
    round_holder_task_id: str
    rounds: list[EnablementRoundSummary]
    round_count: int
    round_outcomes: dict[str, int]
    last_specialist_task_id: str
    revalidation_task_id: str
    revalidation_generation: int
    launch_log_excerpt: str
    trigger_evidence_excerpt: str
    kept_patches: list[str]
    kept_artifacts: list[dict[str, Any]]
    framework_root: str
    kept_stack_action: EnablementStackActionSummary
    candidate_refs: list[str]
    setup_commands: list[str]
    localization_manifest: list[str]
    build_novelty: list[str]
    human_review_count: int
    active_runtime: EnablementAttemptRuntime
    attempt_runtimes: list[EnablementAttemptRuntime]
    failure_kind: str
    build_attempts: list[TargetedBuildAttemptSummary]
    last_build_failure: dict[str, str]
    build_attempt_count: int
    trigger_kind: str
    observed_accuracy: float
    accuracy_floor: float
    observed_task: str
    observed_metric: str
    probe_config_path: str
    accepted_config_path: str
    accepted_config: dict[str, Any]
    eval_contract_fingerprint: str
    setting_script: str


# Session Breakdown v4 canonical author-time schema
class SubjectRef(TypedDict, total=False):
    """Stable reference to a subject participating in an operation."""

    subject_id: str
    subject_type: str
    role: str
    name: str
    attributes: dict[str, Any]


class OperationRelation(TypedDict, total=False):
    """Typed relation from an operation to another operation or subject."""

    relation_id: str
    relation_type: str
    operation_id: str
    target_operation_id: str
    subject: SubjectRef
    metadata: dict[str, Any]


class OperationAttempt(TypedDict, total=False):
    """One execution attempt belonging to an operation."""

    attempt_id: str
    status: str
    producer: str
    backend: str
    started_at: str
    ended_at: str
    sequence: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: dict[str, Any] | str | None
    measurements: list[str]
    artifacts: list[str]
    metadata: dict[str, Any]


class OperationSubstep(TypedDict, total=False):
    """One stable substep nested under an operation."""

    substep_id: str
    kind: str
    name: str
    status: str
    started_at: str
    ended_at: str
    sequence: int
    attempts: list[OperationAttempt]
    measurements: list[str]
    artifacts: list[str]
    metadata: dict[str, Any]


class OperationGate(TypedDict, total=False):
    """A gate evaluated while deciding whether an operation may proceed."""

    gate_id: str
    kind: str
    name: str
    status: str
    decision: str
    reason: str
    evaluated_at: str
    inputs: dict[str, Any]
    evidence: dict[str, Any]
    metadata: dict[str, Any]


class OperationDecision(TypedDict, total=False):
    """An author-time decision made within an operation."""

    decision_id: str
    kind: str
    verdict: str
    reason: str
    component: str
    confidence: float
    decided_at: str
    evidence: dict[str, Any]
    metadata: dict[str, Any]


ExecutorClass = Literal["llm_agent", "llm_tool", "deterministic"]
IntegrityStatus = Literal["exact", "derived", "partial", "unavailable"]

# Which agent owns a unit of work.
AgentBucket = Literal[
    "kernel_agent",
    "framework_agent",
    "explore",
    "warm_replay",
    "coordinator",
    "critic",
    "robustness",
    "unattributed",
]


class Operation(TypedDict, total=False):
    """Canonical unit of work, incrementally upserted by stable id."""

    operation_id: str
    kind: str
    name: str
    phase: str
    status: str
    producer: str
    sequence: int
    started_at: str
    ended_at: str
    parent_operation_id: str
    root_operation_id: str
    macro_cycle: int
    source: str
    executor_class: ExecutorClass
    purpose: str
    scope: str
    # Canonical owning agent, stamped by the producer at author time so the exporter never has to infer ownership from
    # phase timestamps.
    agent: AgentBucket
    strategy_group: str
    strategy: str
    subject: SubjectRef
    subjects: list[SubjectRef]
    relations: list[OperationRelation]
    attempts: list[OperationAttempt]
    substeps: list[OperationSubstep]
    gates: list[OperationGate]
    decisions: list[OperationDecision]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: dict[str, Any] | str | None
    measurement_refs: list[str]
    artifact_refs: list[str]
    adoption_refs: list[str]
    extensions: dict[str, Any]
    metadata: dict[str, Any]


class Measurement(TypedDict, total=False):
    """Canonical measured value authored at the measurement site."""

    measurement_id: str
    operation_id: str
    subject: SubjectRef
    kind: str
    name: str
    value: Any
    unit: str
    status: str
    measured_at: str
    sequence: int
    producer: str
    dimensions: dict[str, Any]
    statistics: dict[str, Any]
    source: dict[str, Any] | str
    metric_basis: str
    harness: dict[str, Any] | str
    workload: dict[str, Any]
    samples: list[Any]
    aggregation: dict[str, Any] | str
    metadata: dict[str, Any]


class ArtifactRef(TypedDict, total=False):
    """Canonical reference to an artifact without reading its contents."""

    artifact_id: str
    operation_id: str
    subject: SubjectRef
    kind: str
    name: str
    path: str
    uri: str
    digest: str
    mime_type: str
    size_bytes: int
    status: str
    present: bool
    created_at: str
    producer: str
    producer_operation_id: str
    consumers: list[str]
    coverage: dict[str, Any] | str
    retention: dict[str, Any] | str
    metadata: dict[str, Any]


class Adoption(TypedDict, total=False):
    """Canonical adoption of an operation result into the accepted state."""

    adoption_id: str
    operation_id: str
    subject: SubjectRef
    artifact_ids: list[str]
    measurement_ids: list[str]
    kind: str
    status: str
    decision: str
    reason: str
    adopted_at: str
    validated: bool
    gain_pct: float | None
    # Frozen at adoption time.
    throughput_before: float | None
    throughput_after: float | None
    configuration: dict[str, Any]
    producer: str
    # Mirrors ``Operation.agent`` so an adoption can be bucketed without a join.
    agent: AgentBucket
    # False for pre-baseline enablement work: real, adopted, and deliberately excluded from reported gain.
    attribution_eligible: bool
    metadata: dict[str, Any]


class IntegrityFieldStatus(TypedDict, total=False):
    """Availability and provenance for one canonical v4 field."""

    status: IntegrityStatus
    source: str
    reason: str
    record_count: int
    producers: list[str]
    warnings: list[str]


class Integrity(TypedDict, total=False):
    """Completeness declaration for the v4 canonical envelope."""

    status: IntegrityStatus
    canonical_source: str
    fields: dict[str, IntegrityFieldStatus]
    warnings: list[str]
    conflicts: list[dict[str, Any]]


class V6MetadataVersions(TypedDict, total=False):
    """Version identifiers projected into V6 metadata."""

    schema_version: str
    hyperloom: str
    framework: str | None
    framework_version: str | None
    tools: dict[str, str | None]


class V6MetadataSession(TypedDict, total=False):
    """Session identity and lifecycle fields exposed by V6 metadata."""

    session_id: str
    claw_session_id: str | None
    sandbox_user_id: str | None
    created_at_utc: str
    start_ts: str
    ended_at_utc: str
    host: str
    session_dir: str
    user_data_path: str
    code_revision: str
    pid: int
    max_minutes: int
    elapsed_minutes: float
    tick_count: int
    recovery: dict[str, Any]


class V6TaskConfig(TypedDict, total=False):
    """Launch-time workload and model architecture projected into V6."""

    model_name: str
    model_path: str
    framework_name: str
    framework_version: str
    gpu_type: str
    tp: int | None
    conc: int | None
    isl: int | None
    osl: int | None
    precision: str
    max_model_len: int | None
    objective: dict[str, Any]
    launch_env: dict[str, str]
    launch_server_args: str
    architecture: dict[str, Any]


class V6Metadata(TypedDict, total=False):
    """V6 task identity, configuration, versions, and trace entrypoint."""

    exported_at_utc: str
    versions: V6MetadataVersions
    session: V6MetadataSession
    task_config: V6TaskConfig
    langfuse: dict[str, Any]
    warnings: list[str]


class V6OutcomeGainBucket(TypedDict, total=False):
    """Additive, session-baseline-relative gain for one V6 source bucket."""

    total_gain_pct: float | None
    keep_count: int
    non_attributable_keep_count: int


class V6OutcomeKernelAttribution(V6OutcomeGainBucket, total=False):
    """Kernel gain with its authoritative GEAK and Forge backend split."""

    by_backend: dict[str, V6OutcomeGainBucket]


class V6OutcomeAttributionBySource(TypedDict, total=False):
    """Canonical ledger gain projected onto the V6 stage vocabulary."""

    warm_replay: V6OutcomeGainBucket
    framework_agent: V6OutcomeGainBucket
    kernel: V6OutcomeKernelAttribution


class V6OutcomeAttribution(TypedDict, total=False):
    """Availability and additive gain attribution from the canonical ledger."""

    available: bool
    by_source: V6OutcomeAttributionBySource


class V6OutcomeValidation(TypedDict, total=False):
    """Reconciliation of final measured gain with canonical KEEP entries."""

    attributed_gain_pct: float
    unattributed_gain_pct: float
    reconciliation_gap_pct: float | None
    attribution: V6OutcomeAttribution
    notes: list[str]


class V6Outcome(TypedDict, total=False):
    """V6 session result projection for downstream consumers."""

    stop_reason: str
    status: Literal["completed", "failed", "aborted"]
    stage_reached: str
    baseline: dict[str, Any]
    final: dict[str, Any]
    validation: V6OutcomeValidation


class V6TimelineEvent(TypedDict, total=False):
    """One ordered V6 business-stage event; CLOSE is intentionally excluded."""

    type: str
    kind: str
    status: str
    start_time: str
    end_time: str
    id: str
    ext: dict[str, Any]


class V6WarmStartMatched(TypedDict, total=False):
    """The Recipe the PRELUDE KB lookup selected."""

    match_type: Literal["exact", "degraded"]
    tier: str
    confidence: float | None
    source: str
    canonical_id: str
    scope: dict[str, Any]
    optimized_throughput: float | None
    validated_gain_pct: float | None
    expected_gain_pct: float | None
    replayable: bool | None
    replay_disabled_reason: str | None
    replay_material_available: bool | None
    view_source: str | None
    origin: dict[str, Any]
    experience: dict[str, Any]


class V6WarmStartReads(TypedDict, total=False):
    """``timeline[type=warm_start].ext.reads`` — Recipe KB read attribution."""

    count: int
    hits: int
    by_resolution: dict[str, int]
    by_remote: dict[str, int]
    by_source: dict[str, int]
    best_config_by_source: dict[str, int]
    tail: list[dict[str, Any]]


class V6WarmStartExt(TypedDict, total=False):
    """``timeline[type=warm_start].ext`` — what was asked for, what came back."""

    requested: dict[str, Any]
    match_status: str
    matched: V6WarmStartMatched
    reads: V6WarmStartReads


class V6WarmReplayApplied(TypedDict, total=False):
    """What was running when a warm replay reproduced its gain."""

    config: dict[str, Any]
    patch: list[str]
    kernel: dict[str, Any]


class V6WarmReplayExt(TypedDict, total=False):
    """``timeline[type=warm_replay].ext`` — did the record reproduce, and why not."""

    raw_status: str
    result_type: str
    raw_reason: str | None
    tier: str | None
    confidence: float | None
    config_source: str | None
    config_donor_tier: str | None
    donor: dict[str, Any]
    before_tput: float | None
    after_tput: float | None
    gain_pct: float | None
    expected_gain_pct: float | None
    keep_threshold_pct: float | None
    historical_reproduce_bar_pct: float | None
    below_historical_reproduce: bool | None
    accuracy: dict[str, Any]
    applied: V6WarmReplayApplied
    active_framework_root: str
    rollback: dict[str, Any]
    failure: dict[str, Any]


class V6KBWriteBackExt(TypedDict, total=False):
    """``timeline[type=kb_write_back].ext`` — did this session's Recipe land."""

    result_type: str
    raw_reason: str | None
    backend: str | None
    canonical_id: str | None
    session_id: str | None
    scope: dict[str, Any]
    optimized_throughput: float | None
    validated_gain_pct: float | None
    attempts: int | None
    source: str | None
    queue: dict[str, Any]
    failure: dict[str, Any]


class V6Close(TypedDict, total=False):
    """V6 session finalization result exposed outside the business timeline."""

    status: Literal["succeeded", "failed", "degraded"]
    start_time: str
    end_time: str
    close_sequence_done: bool
    steps: list[dict[str, Any]]
    robustness: dict[str, Any]
    artifacts: dict[str, Any]


# V6 KERNEL timeline event
class V6KernelEntry(TypedDict, total=False):
    """What the KERNEL entry hook decided and what it inherited."""

    route: str
    route_reason: str
    resumed: bool
    code_revision: str | None
    stack_depth_in: int | None
    budget_remaining_sec: float | None
    roofline_snapshot_id: int | None
    roofline_snapshot_ts: str | None
    roofline_baseline_gain_at_snapshot: float | None
    snapshot_staleness: float | None


class V6KernelReprofile(TypedDict, total=False):
    """The entry re-profile that decides whether cached analysis is stale."""

    ran: bool
    task_kind: str | None
    trigger: str | None
    skipped_reason: str | None
    idempotency_reason: str | None
    snapshot_landed: bool
    snapshot_id_before: int | None
    snapshot_id_after: int | None


class V6RowScope(TypedDict, total=False):
    """Recording-side bookkeeping every V6 row fragment carries."""

    event_id: str
    ordinal: int


class V6KernelAnalysisArtifacts(TypedDict, total=False):
    """Paths one trace-analysis run produced."""

    trace_report_path: str
    analysis_report_path: str
    candidates_path: str
    kernel_roofline_path: str
    tracelens_summary_path: str
    cli_log_path: str


class V6KernelAnalysisDetail(TypedDict, total=False):
    """Trace-analysis metadata shared by the roofline and kernel events."""

    route: str
    tool: str
    tool_run_id: str
    steady_state: dict[str, Any]
    preflight: dict[str, Any]
    split: dict[str, Any]
    selection: dict[str, Any]
    steps: Any
    route_ext: dict[str, Any]
    hot_kernels: dict[str, Any]
    warnings: list[dict[str, Any]]
    artifacts: V6KernelAnalysisArtifacts


class V6KernelTraceAnalyzeRun(V6KernelAnalysisDetail, V6RowScope, total=False):
    """One analysis the KERNEL phase requested for itself."""

    run_id: str
    trigger: str | None
    requested_by: str | None
    request_msg_id: str | None
    ts: str
    status: str
    cache_hit: bool
    trace_input: str | None
    top_k: int | None
    roofline_snapshot_id: int | None
    roofline_baseline_gain_at_snapshot: float | None
    steady_state_trace: str | None
    analysis_md_path: str | None
    reusable_native_kernel_ids: list[str]
    trace_validate_ref: str | None


class V6KernelLaneRun(V6RowScope, total=False):
    """Fields every forge candidate row carries, whichever lane produced it."""

    lane: str
    source_kind: str
    run_id: str
    status: str
    started_at: str | None
    ended_at: str | None
    duration_sec: float | None
    micro_decision: str | None
    rebench_ref: str | None
    outcome: str
    failure_reason: str | None


class V6KernelRewriteE2E(TypedDict, total=False):
    """End-to-end integration sub-result of one kernel rewrite."""

    integrated: bool
    e2e_gain_pct: float | None
    validated: bool | None
    decision: str | None
    patch_path: str | None
    target_file: str | None


class V6KernelRewriteRun(V6KernelLaneRun, total=False):
    """One forge source-level kernel rewrite."""

    kernel_id: str
    kernel_name: str | None
    dispatched: bool
    backends_tried: list[str]
    adopted_backend: str | None
    skip_reason: str | None
    task_group: str | None
    speedup: float | None
    baseline_us: float | None
    candidate_us: float | None
    compile_status: str | None
    correctness: bool | None
    artifact_path: str | None
    trace_analyze_ref: str | None
    e2e: V6KernelRewriteE2E | None


class V6KernelFusionRun(V6KernelLaneRun, total=False):
    """One forge-fusion run."""

    pattern: str | None
    target_module: str | None
    applied: bool
    gain_pct: float | None
    patch_path: str | None


class V6KernelGemmTuningRun(V6KernelLaneRun, total=False):
    """One GEMM shape-table tuning run."""

    shapes_total: int | None
    shapes_tuned: int | None
    config_path: str | None
    gain_pct: float | None
    tuner: str | None


class V6KernelForgeLanes(TypedDict, total=False):
    """The forge candidate lanes, split back out at assembly."""

    kernel_rewrites: list[V6KernelRewriteRun]
    fusion_runs: list[V6KernelFusionRun]
    gemm_tuning_runs: list[V6KernelGemmTuningRun]


class V6KernelRebenchEngagement(TypedDict, total=False):
    """Whether the configuration under test actually took effect."""

    config_matched: bool | None
    overlay_loaded: bool | None
    expected_cfg_hash: str | None
    observed_cfg_hash: str | None
    expected_overlay_digest: str | None
    observed_overlay_digest: str | None


class V6KernelRebenchAttempt(V6RowScope, total=False):
    """One end-to-end re-measurement of a candidate."""

    attempt_id: str
    source_kind: str
    source_ref: str | None
    idempotency_key: str | None
    task_id: str | None
    dispatched_at: str | None
    settled_at: str | None
    base_tput: float | None
    measured_tput: float | None
    decision: str | None
    decision_reason: str | None
    status: str | None
    engagement: V6KernelRebenchEngagement


class V6KernelForge(TypedDict, total=False):
    """The forge route's work for one visit."""

    engaged: bool
    reprofile: V6KernelReprofile | None
    trace_analyze_runs: list[V6KernelTraceAnalyzeRun]
    lanes: V6KernelForgeLanes
    rebench_ledger: list[V6KernelRebenchAttempt]


class V6KernelGeakHandoff(TypedDict, total=False):
    """The conditions GEAK was asked to work under."""

    schema_version: int | None
    model_path: str | None
    framework: str | None
    gpu_type: str | None
    tp: int | None
    workload: dict[str, Any]
    baseline_flags: str | None
    baseline_envs: str | dict[str, Any] | None
    baseline_env_spec_present: bool
    launch_recipe: str | None
    raw_baseline_tput: float | None
    orchestrator_best_tput_same_config: float | None
    max_model_len: int | None
    mem_fraction: float | None
    bench_client: str | None
    e2e_metric: str | None
    bench_protocol_present: bool
    gpu_ids: str | None
    exp_root: str | None
    eval_dir: str | None


class V6KernelGeakDelegation(TypedDict, total=False):
    """How the delegated GEAK runner process itself ended."""

    runner_status: str
    started_at: str | None
    ended_at: str | None
    duration_sec: float | None
    error_class: str | None
    error: str | None
    returncode: int | None
    runner_timeout_sec: int | None
    kill_timeout_sec: int | None
    exp_root: str | None
    eval_dir: str | None
    report_path: str | None
    versions: dict[str, Any]
    recovered_from_disk: bool
    stages_reached: list[str]


class V6KernelGeakDiscoveryRun(V6RowScope, total=False):
    """One hot-kernel discovery run GEAK performed for itself."""

    source: str | None
    status: str | None
    hot_kernel_count: int
    scan: dict[str, Any]


class V6KernelGeakBackendResult(TypedDict, total=False):
    """What one backend measured for one kernel GEAK attempted."""

    backend: str | None
    status: str | None
    speedup: float | None
    baseline_us: float | None
    candidate_us: float | None
    compile_status: str | None
    correctness: bool | None
    artifact_path: str | None
    error_class: str | None


class V6KernelGeakAttempt(V6RowScope, total=False):
    """One kernel GEAK considered, replayed from its conclusion file."""

    kernel_id: str
    dispatched: bool
    backends: list[str]
    skip_reason: str | None
    task_group: str | None
    backend_result: V6KernelGeakBackendResult | None
    e2e: V6KernelRewriteE2E | None


class V6KernelGeakAttemptCounts(TypedDict, total=False):
    """Assembly-derived tally of the kernels GEAK attempted."""

    discovered: int
    dispatched: int
    skipped: int
    backend_ok: int
    backend_fail: int
    integrated: int


class V6KernelGeakAttempts(TypedDict, total=False):
    """What GEAK tried, assembled from its replayed conclusion file."""

    discovery_runs: list[V6KernelGeakDiscoveryRun]
    kernels: list[V6KernelGeakAttempt]
    counts: V6KernelGeakAttemptCounts


class V6KernelGeakAuthoredKernel(V6RowScope, total=False):
    """One kernel GEAK authored and accepted."""

    short_name: str | None
    kernel_id: str | None
    cand_tag: str | None
    name_source: str
    op_kind: str | None
    lane: str | None
    e2e_delta_pct: float | None
    alias_collapsed: bool


class V6KernelGeakEnvSelection(V6RowScope, total=False):
    """One environment / flag selection GEAK accepted."""

    selection: str
    op_kind: str | None
    lane: str | None
    e2e_delta_pct: float | None


class V6KernelGeakClaim(TypedDict, total=False):
    """What GEAK reported about itself, before any re-measurement."""

    verified: bool
    self_reported_tput: float | None
    self_reported_speedup: float | None
    self_reported_gain_pct: float | None
    self_reported_basis: str | None
    geak_status: str | None
    baseline_alignment_status: str | None
    authored_kernels: list[V6KernelGeakAuthoredKernel]
    env_selections: list[V6KernelGeakEnvSelection]
    kernels_optimized: int
    accepted_heads_count: int
    validated_regimes: list[Any]


class V6KernelGeakProduct(TypedDict, total=False):
    """The reproducible configuration GEAK handed back."""

    accepted_flags: str | list[str] | None
    accepted_envs: dict[str, Any]
    accepted_config: dict[str, Any]
    cfg_hash: str | None
    final_overlay: str | None
    final_overlay_digest: str | None
    final_launch_script: str | None
    bench_script: str | None
    final_patch: str | None


class V6KernelGeakRebench(TypedDict, total=False):
    """The orchestrator's own re-measurement campaign for GEAK's candidate."""

    required: bool
    max_attempts: int | None
    attempts_used: int
    attempts: list[V6KernelRebenchAttempt]
    final_status: str | None
    final_error_class: str | None
    final_error: str | None
    conflicting_decisions: list[str]


class V6KernelGeak(TypedDict, total=False):
    """The GEAK route's work for one visit, in causal order."""

    engaged: bool
    handoff: V6KernelGeakHandoff | None
    delegation: V6KernelGeakDelegation | None
    attempts: V6KernelGeakAttempts | None
    claim: V6KernelGeakClaim | None
    product: V6KernelGeakProduct | None
    rebench: V6KernelGeakRebench


class V6KernelAdoptedRow(TypedDict, total=False):
    """One candidate a settled rebench validated."""

    source_kind: str
    ref: str
    gain_pct: float | None
    rebench_ref: str


class V6KernelPendingRow(TypedDict, total=False):
    """One candidate no settled rebench concluded on."""

    source_kind: str
    ref: str
    why: str


class V6KernelSourceCounters(TypedDict, total=False):
    """Assembly-derived per-source candidate tally."""

    attempted: int
    adopted: int
    needs_review: int
    rejected: int


class V6KernelStackDelta(TypedDict, total=False):
    """Optimization-stack entries this visit added and removed."""

    added: list[dict[str, Any]]
    removed: list[dict[str, Any]]


class V6KernelOutcome(TypedDict, total=False):
    """What the visit concluded, settled against the rebench evidence."""

    route: str
    verdict: str | None
    exit_reason: str | None
    tput_before: float | None
    tput_after: float | None
    net_gain_pct: float | None
    session_baseline_tput: float | None
    cumulative_gain_validated_out: float | None
    stack_depth_out: int | None
    adopted: list[V6KernelAdoptedRow]
    pending_review: list[V6KernelPendingRow]
    by_source: dict[str, V6KernelSourceCounters]
    stack_delta: V6KernelStackDelta


class V6KernelFailure(TypedDict, total=False):
    """The stage that failed, when the visit ended on a miss."""

    phase: str
    error_class: str
    message: str


class V6KernelExt(TypedDict, total=False):
    """``ext`` of the V6 ``kernel`` timeline event."""

    macro_cycle: int
    in_flight_stage: str | None
    duration_sec: float | None
    entry: V6KernelEntry
    geak: V6KernelGeak | None
    forge: V6KernelForge | None
    outcome: V6KernelOutcome
    failure: V6KernelFailure | None


class V6KernelEvent(TypedDict, total=False):
    """Recorder fragment for one KERNEL event (section ``kernel_event``)."""

    event_id: str
    macro_cycle: int
    timeline_sequence: int | None
    in_flight_stage: str | None
    start_time: str
    end_time: str | None
    status: str | None
    entry: V6KernelEntry
    route: str
    forge_engaged: bool
    reprofile: V6KernelReprofile | None
    geak_engaged: bool
    geak_handoff: V6KernelGeakHandoff | None
    geak_delegation: V6KernelGeakDelegation | None
    geak_claim: V6KernelGeakClaim | None
    geak_product: V6KernelGeakProduct | None
    geak_rebench: V6KernelGeakRebench | None
    outcome: V6KernelOutcome
    failure: V6KernelFailure | None


class SessionBreakdown(TypedDict, total=False):
    """Top-level wire shape of ``session_breakdown.json``."""

    schema_version: str
    exported_at_utc: str
    exporter_version: str

    session: SessionMeta
    workload: Workload
    # Structural model summary parsed from config.json (state.model_info mirror); {} when absent.
    model_info: ModelInfo
    baseline: Baseline
    final: Final
    # flat per-action timeline (v1 compat); ``phase_segments`` is the boundary view.
    phase_timeline: list[PhaseEvent]
    phase_segments: list[PhaseSegment]
    # flat-list alias for older readers.
    action_timeline: list[PhaseEvent]
    capability_summary: CapabilitySummary
    geak: Geak
    kernel_lifecycle: KernelLifecycle
    # explore_search is the native merged ledger; param_search is a v1 alias.
    param_search: ParamSearch
    explore_search: ParamSearch
    critic_robustness: CriticRobustness
    telemetry: Telemetry
    # Single downstream read model for every formally adopted optimization.
    optimizations: Optimizations
    specialist_runs: list[SpecialistRound]
    # Hot-kernel table, mirror of ``reports/kernel_roofline.json``.
    kernel_roofline: KernelRoofline
    # Kernel-agent attempt outcome summary; empty → dashboard hides Block 1.
    kernel_optimization_summary: KernelOptimizationSummary
    # Post-optimization concurrency sweep; empty → dashboard hides Block 2.
    conc_sweep_summary: ConcSweepSummary
    # Per-snapshot roofline comparison list driving the markdown ``## Roofline`` section.
    roofline: list[dict[str, Any]]
    # Optimization-progress curve (coexists with the list-shaped ``roofline`` above).
    roofline_progress: RooflineProgress
    # Full-trace token + decision timeline; {} on pre-trace-subsystem sessions.
    decision_trace: DecisionTrace
    # Promoted token-spend rollup derived from decision_trace.token_rollup.
    token_usage: TokenUsage
    # Live-Langfuse push receipt; ``enabled`` False (with ``disabled_reason``) when the push is off.
    langfuse: LangfusePush
    # Kernel-major unified lifecycle view (discovery -> dispatch -> backend attempts -> e2e); {} when absent.
    kernel_journey: KernelJourney
    # Authoritative external-tool versions keyed by tool name; {} when absent.
    versions: dict[str, KernelToolMetadata]
    # Enablement attempt-runtime observability; {} → dashboard hides the block.
    enablement: EnablementBreakdown
    metadata: V6Metadata
    outcome: V6Outcome
    timeline: list[V6TimelineEvent]
    close: V6Close

    warnings: list[str]
    source_files: SourceFiles


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V5",
    "SCHEMA_VERSION_V6",
    "Adoption",
    "AdoptedKernel",
    "ArtifactRef",
    "Attribution",
    "Baseline",
    "BaselineAttemptSummary",
    "BenchmarkInvocation",
    "CapabilityEntry",
    "CapabilitySummary",
    "ConcSweepSummary",
    "CriticIteration",
    "CriticKBWritesSummary",
    "CriticRobustness",
    "DecisionTokens",
    "DecisionTrace",
    "DecisionTraceEntry",
    "DetectedKernel",
    "DiscoveredHotKernel",
    "EnablementAttemptRuntime",
    "EnablementBreakdown",
    "EnablementRoundSummary",
    "EnablementStackActionSummary",
    "ExecutorClass",
    "KernelBackendAttempt",
    "KernelDiscoveryRun",
    "KernelDispatch",
    "KernelE2E",
    "KernelJourney",
    "KernelJourneyEntry",
    "KernelToolMetadata",
    "LangfuseConfig",
    "LangfusePush",
    "LangfusePushCounts",
    "Measurement",
    "ModelInfo",
    "Operation",
    "OperationAttempt",
    "OperationDecision",
    "OperationGate",
    "OperationRelation",
    "OperationSubstep",
    "OptimizationArtifact",
    "OptimizationBackendAttempt",
    "OptimizationConfiguration",
    "OptimizationEntry",
    "Optimizations",
    "OptimizationSource",
    "OptimizationSourceMethod",
    "OptimizationSourceSummary",
    "OptimizationValidation",
    "OrchestrationContext",
    "Final",
    "GpuMonitorAggregate",
    "Invocation",
    "Integrity",
    "IntegrityFieldStatus",
    "IntegrityStatus",
    "LaneTimelineEntry",
    "KernelLifecycle",
    "KernelMetadata",
    "KernelOptimizationSummary",
    "KernelExecutionMode",
    "KernelOptimizationBackend",
    "OptimizedKernel",
    "ParamSearch",
    "ParamSearchEntry",
    "ParamSearchLedger",
    "PhaseEvent",
    "PhaseSegment",
    "RecommendedKernel",
    "RejectedKernel",
    "RobustnessSignal",
    "SessionBreakdown",
    "SessionMeta",
    "SpecialistDomainBreakdown",
    "SpecialistRound",
    "SpecialistTranscriptRef",
    "SubjectRef",
    "PhaseBreakdown",
    "PhaseBreakdownExplore",
    "PhaseBreakdownKernel",
    "SourceBreakdown",
    "SourceFiles",
    "StackGainEntry",
    "Telemetry",
    "TokenBucket",
    "TokenRollup",
    "TokenUsage",
    "TokenUsageAttribution",
    "TokenUsageBucket",
    "TokenUsageTimelineEntry",
    "V6Metadata",
    "V6MetadataSession",
    "V6MetadataVersions",
    "V6Close",
    "V6KBWriteBackExt",
    "V6KernelAdoptedRow",
    "V6KernelAnalysisArtifacts",
    "V6KernelAnalysisDetail",
    "V6KernelEntry",
    "V6KernelEvent",
    "V6KernelExt",
    "V6KernelFailure",
    "V6KernelForge",
    "V6KernelForgeLanes",
    "V6KernelFusionRun",
    "V6KernelGeak",
    "V6KernelGeakAttempt",
    "V6KernelGeakAttemptCounts",
    "V6KernelGeakAttempts",
    "V6KernelGeakAuthoredKernel",
    "V6KernelGeakBackendResult",
    "V6KernelGeakClaim",
    "V6KernelGeakDelegation",
    "V6KernelGeakDiscoveryRun",
    "V6KernelGeakEnvSelection",
    "V6KernelGeakHandoff",
    "V6KernelGeakProduct",
    "V6KernelGeakRebench",
    "V6KernelGemmTuningRun",
    "V6KernelLaneRun",
    "V6KernelOutcome",
    "V6KernelPendingRow",
    "V6KernelRebenchAttempt",
    "V6KernelRebenchEngagement",
    "V6KernelReprofile",
    "V6KernelRewriteE2E",
    "V6KernelRewriteRun",
    "V6KernelSourceCounters",
    "V6KernelStackDelta",
    "V6KernelTraceAnalyzeRun",
    "V6OutcomeAttribution",
    "V6OutcomeAttributionBySource",
    "V6OutcomeGainBucket",
    "V6OutcomeKernelAttribution",
    "V6Outcome",
    "V6OutcomeValidation",
    "V6RowScope",
    "V6TaskConfig",
    "V6TimelineEvent",
    "V6WarmReplayApplied",
    "V6WarmReplayExt",
    "V6WarmStartExt",
    "V6WarmStartMatched",
    "V6WarmStartReads",
    "Workload",
    "WorkloadObjective",
]
