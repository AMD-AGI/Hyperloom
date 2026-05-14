"""Schema (TypedDict shape) for ``session_breakdown.json``.

This is the single contract between ``inference_optimizer`` (producer)
and any downstream consumer (``claw-stats-service``, results service,
notebooks).

Design notes
------------
* All fields are ``NotRequired``-by-convention: collectors may return
  ``None`` / ``[]`` / ``{}`` when the underlying artifacts are not
  present. Consumers MUST treat missing data as "not available" — never
  fabricate values.
* Schema is JSON-serializable; no dataclasses, no enums in the wire
  shape. Status strings are documented in their respective TypedDicts.
* Versioning: ``schema_version`` is bumped on any breaking change. Add
  new optional fields freely without bumping.
"""

from __future__ import annotations

from typing import Any, TypedDict

SCHEMA_VERSION = "hyperloom.session_breakdown.v1"


# ---------------------------------------------------------------------------
# §1 Session metadata
# ---------------------------------------------------------------------------
class SessionMeta(TypedDict, total=False):
    session_id: str               # hyperloom internal id (manifest.session_id)
    claw_session_id: str | None   # SaFE / Claw session id (env CLAW_SESSION_ID)
    sandbox_user_id: str | None
    created_at_utc: str
    ended_at_utc: str
    stop_reason: str              # target_reached / time_exhausted / no_more_leverage / max_ticks / baseline_failed / ...
    max_minutes: int
    elapsed_minutes: float
    host: str
    code_revision: str
    pid: int
    session_dir: str
    tick_count: int


# ---------------------------------------------------------------------------
# §2 Workload configuration
# ---------------------------------------------------------------------------
class WorkloadObjective(TypedDict, total=False):
    kind: str                     # gain_pct / tput / baseline / time_only
    value: Any                    # float or str (target_baseline_dir) or None


class Workload(TypedDict, total=False):
    framework: str                # sglang / vllm
    framework_version: str
    model_name: str
    model_path: str
    model_class: str
    gpu_type: str                 # mi300x / mi325x / mi355x
    tp: int | None
    conc: int | None
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


class Baseline(TypedDict, total=False):
    throughput_tok_s_per_gpu: float
    accuracy: float
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    config_path: str | None
    benchmark_report_path: str | None
    attempts_history: list[BaselineAttemptSummary]
    failure_streak: int


# ---------------------------------------------------------------------------
# §4 Final state — SaFE contract core
# ---------------------------------------------------------------------------
class Final(TypedDict, total=False):
    throughput_tok_s_per_gpu: float | None
    cumulative_gain_pct_validated: float
    cumulative_gain_pct_per_round_sum: float
    validated_at_stack_len: int
    validated_ts: str
    stack_changed_after_validation: bool
    extra_sglang_args: str
    extra_envs: dict[str, Any]
    action_path: list[str]        # ordered list of action:variant labels from optimization_stack
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None


# ---------------------------------------------------------------------------
# §5 Phase timeline — chronological events
# ---------------------------------------------------------------------------
class PhaseEvent(TypedDict, total=False):
    ts: str
    action: str                   # baseline / profile / backends / params / sweep / validate_stack / kernel_opt / select_kernels / integrate
    task_id: str
    kernel_id: str | None         # only for kernel-owned actions
    status: str                   # succeeded / failed
    decision: str                 # promoted / discarded / salvaged / no_promote / error / KEEP / PARTIAL / REVERT
    key_metric: float | None
    key_metric_kind: str | None
    workspace: str | None
    error_class: str | None
    extras: dict[str, Any]


# ---------------------------------------------------------------------------
# §6 Capability summary — Capability cards in UI
# ---------------------------------------------------------------------------
class CapabilityEntry(TypedDict, total=False):
    status: str                   # kept / tried / attempted / not_attempted / not_configured / failed / completed
    attempts: int
    keeps: int
    tested: int                   # for backends/params: distinct variants tested
    best_gain_pct: float | None
    reason: str                   # human readable, e.g. "kernel-claude only this run"


class CapabilitySummary(TypedDict, total=False):
    geak: CapabilityEntry
    oob: CapabilityEntry
    backends: CapabilityEntry
    params: CapabilityEntry
    sweep: CapabilityEntry
    validate_stack: CapabilityEntry


# ---------------------------------------------------------------------------
# §7 / §8 GEAK / OOB invocations
# ---------------------------------------------------------------------------
class KernelMetadata(TypedDict, total=False):
    name: str
    source_file: str
    shapes: list[dict[str, Any]]
    gpu_pct: float | None
    arithmetic_intensity: float | None


class Invocation(TypedDict, total=False):
    """One backend invocation for one kernel.

    Same shape for GEAK and OOB; ``backend`` distinguishes them.
    """
    kernel_id: str
    attempt_id: str
    run_id: str
    ts: str
    backend: str                  # geak / claude / codex
    model: str | None
    kernel_metadata: KernelMetadata
    prompt_path: str | None
    optimized_files: list[str]
    result_path: str | None
    verification_path: str | None
    decision: str                 # KEEP / PARTIAL / REVERT / FAILED
    micro_speedup: float | None
    compile_passed: bool | None
    correctness_passed: bool | None
    best_artifact_path: str | None
    error: str | None
    cli_log_path: str | None


# ---------------------------------------------------------------------------
# §9 Kernel lifecycle (4+1 stages)
# ---------------------------------------------------------------------------
class DetectedKernel(TypedDict, total=False):
    kernel_id: str
    name: str
    gpu_pct: float | None
    time_ms: float | None
    bottleneck: str               # compute / memory / comm
    arithmetic_intensity: float | None
    reusable_native_kernel: bool
    source_file: str | None
    detected_from_task: str       # which profile task_id surfaced it
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
    backend: str                  # geak / claude / codex (best-of)
    total_attempts: int
    successful_attempts: int
    best_micro_speedup: float | None
    last_decision: str
    best_artifact_path: str | None
    attempts_summary: list[dict[str, Any]]


class AdoptedKernel(TypedDict, total=False):
    kernel_id: str
    patch_path: str
    target_file: str
    extra_sglang_args: str
    e2e_gain_pct: float | None
    validated: bool
    last_status: str
    adopted_at: str
    attempt_count: int


class RejectedKernel(TypedDict, total=False):
    kernel_id: str
    reason: str
    patch_path: str | None
    target_file: str | None
    attempt_count: int
    best_gain_pct: float | None
    ts: str


class KernelLifecycle(TypedDict, total=False):
    detected: list[DetectedKernel]
    recommended: list[RecommendedKernel]
    optimized: list[OptimizedKernel]
    adopted: list[AdoptedKernel]
    rejected: list[RejectedKernel]


# ---------------------------------------------------------------------------
# §10 Param search
# ---------------------------------------------------------------------------
class ParamSearchEntry(TypedDict, total=False):
    """One row from params_search.{tested,accepted,rejected}."""
    name: str
    fingerprint: str
    extra_sglang_args: str
    extra_envs: dict[str, Any]
    output_throughput: float | None
    gain_pct: float | None
    ts: str
    status: str                   # accepted / rejected / tested


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
# §11 Sweep
# ---------------------------------------------------------------------------
class SweepPoint(TypedDict, total=False):
    variant_name: str
    conc: int | None
    isl: int | None
    osl: int | None
    output_throughput_tok_s: float | None
    ttft_mean_ms: float | None
    tpot_mean_ms: float | None
    e2el_mean_ms: float | None
    status: str                   # ok / skipped / failed
    benchmark_report_path: str | None


class Sweep(TypedDict, total=False):
    grid_size: int
    best_overall: dict[str, Any]
    best_for_each_conc: list[dict[str, Any]]
    pareto_front: list[dict[str, Any]]
    all_variants: list[SweepPoint]
    config_path: str | None


# ---------------------------------------------------------------------------
# §12 Critic / Robustness
# ---------------------------------------------------------------------------
class CriticIteration(TypedDict, total=False):
    iter: int
    ts: str
    topic: str                    # what was reviewed (kernel_opt:k001, backends:flag_X, ...)
    verdict: str                  # approve / reject / redirect / advise / needs_review
    summary: str
    request_path: str
    judge_bundle_path: str
    emit_path: str
    review_path: str


class RobustnessSignal(TypedDict, total=False):
    ts: str
    signal: str                   # crash / stall / disk_full / cluster_fault / ...
    action: str                   # what was done
    workdir: str


class CriticRobustness(TypedDict, total=False):
    critic_iterations: list[CriticIteration]
    robustness_signals: list[RobustnessSignal]


# ---------------------------------------------------------------------------
# §13 Telemetry
# ---------------------------------------------------------------------------
class GpuMonitorAggregate(TypedDict, total=False):
    samples: int
    avg_power_w: float
    max_power_w: float
    avg_temp_c: float
    max_temp_c: float
    avg_clock_mhz: float


class Telemetry(TypedDict, total=False):
    baseline_report_path: str | None
    profile_report_paths: list[str]
    torch_trace_paths: list[str]
    system_profile_paths: list[str]
    server_log_paths: list[str]
    gpu_monitor_aggregate: GpuMonitorAggregate


# ---------------------------------------------------------------------------
# §14 Attribution
# ---------------------------------------------------------------------------
class StackGainEntry(TypedDict, total=False):
    """One KEEP/validation event with its incremental contribution."""
    ts: str
    stack_len_before: int
    stack_len_after: int
    action: str                   # backends / params / kernel_opt:<kid> / validate_stack
    variant_name: str | None
    cum_gain_before: float
    cum_gain_after: float
    delta_pct: float | None       # None when validate_stack re-baselined
    extra_sglang_args: str


class SourceBreakdown(TypedDict, total=False):
    geak_pct_of_total: float
    oob_pct_of_total: float
    backends_pct_of_total: float
    params_pct_of_total: float
    sweep_pct_of_total: float
    validated_total_pct: float


class Attribution(TypedDict, total=False):
    gain_per_stack_entry: list[StackGainEntry]
    source_breakdown: SourceBreakdown
    notes: list[str]              # human-readable caveats


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------
class SourceFiles(TypedDict, total=False):
    manifest: str
    state: str
    baseline_report: str | None
    profile_reports: list[str]
    sweep_reports: list[str]
    kernel_attempts: list[str]
    critic_workdir: str | None
    robustness_workdir: str | None


class SessionBreakdown(TypedDict, total=False):
    schema_version: str
    exported_at_utc: str
    exporter_version: str

    session: SessionMeta
    workload: Workload
    baseline: Baseline
    final: Final
    phase_timeline: list[PhaseEvent]
    capability_summary: CapabilitySummary
    geak_invocations: list[Invocation]
    oob_invocations: list[Invocation]
    kernel_lifecycle: KernelLifecycle
    param_search: ParamSearch
    sweep: Sweep
    critic_robustness: CriticRobustness
    telemetry: Telemetry
    attribution: Attribution

    warnings: list[str]
    source_files: SourceFiles


__all__ = [
    "SCHEMA_VERSION",
    "AdoptedKernel",
    "Attribution",
    "Baseline",
    "BaselineAttemptSummary",
    "CapabilityEntry",
    "CapabilitySummary",
    "CriticIteration",
    "CriticRobustness",
    "DetectedKernel",
    "Final",
    "GpuMonitorAggregate",
    "Invocation",
    "KernelLifecycle",
    "KernelMetadata",
    "OptimizedKernel",
    "ParamSearch",
    "ParamSearchEntry",
    "ParamSearchLedger",
    "PhaseEvent",
    "RecommendedKernel",
    "RejectedKernel",
    "RobustnessSignal",
    "SessionBreakdown",
    "SessionMeta",
    "SourceBreakdown",
    "SourceFiles",
    "StackGainEntry",
    "Sweep",
    "SweepPoint",
    "Telemetry",
    "Workload",
    "WorkloadObjective",
]
