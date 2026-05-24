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

SCHEMA_VERSION = "hyperloom.session_breakdown.v1.1"


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
    image: str | None             # container image fully-qualified (or None if not configured)
    # v1.1 additions — image fingerprint + real session lifecycle timestamps.
    # ``image_id`` is the short / human-friendly image tag suffix (e.g. ``rocm/sglang:6.4``)
    # when the manifest carries a fully-qualified ``image`` plus a separate id; ``image_digest``
    # is the immutable content digest (``sha256:...``) when known. Both default to None.
    image_id: str | None
    image_digest: str | None
    # Real session start / end timestamps and duration, distinct from the
    # legacy ``ended_at_utc`` field which records dump (export) time for
    # backward compatibility with existing consumers. When the
    # orchestrator wrote ``state.start_ts`` we propagate it verbatim into
    # ``session_started_at_utc``; ``session_ended_at_utc`` is derived from
    # ``state.closing_started_unix`` (preferred) or the latest
    # phase_timeline event end, and ``session_duration_seconds`` is the
    # arithmetic difference (rounded to 1s). All three are None when the
    # underlying signal is missing.
    session_started_at_utc: str | None
    session_ended_at_utc: str | None
    session_duration_seconds: float | None


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
    invocation: BenchmarkInvocation


class BenchmarkInvocation(TypedDict, total=False):
    """Replayable record of how a benchmark variant was launched.

    Allows operators to rerun the exact same workload (server command +
    env vars + config) when investigating a regression. ``extra_envs`` is
    allowlist-filtered in the collector to keep secrets out of the
    breakdown JSON.
    """
    framework_args: str           # e.g. "python -m sglang.launch_server --model ... --tp 8"
    framework_args_source: str
    # log_non_default_args (vllm/sglang parsed-args echo, most authoritative)
    # log_args_line (Server arguments: / Args: Namespace(...) header)
    # log_python_cmd (literal python/vllm/sglang launch line)
    # yaml_cmd (cmd/command/launch field in materialized config yaml)
    # yaml_benchmark (synthesized from magpie benchmark.* fields)
    # unknown (none of the above; warning emitted)
    extra_envs: dict[str, str]    # allowlisted env vars only (no secrets)
    config_path: str | None       # baseline_config.with_envs.yaml or variant config
    server_log_path: str | None   # for debug


class WorkloadDims(TypedDict, total=False):
    conc: int | None
    isl: int | None
    osl: int | None
    tp: int | None
    precision: str


class Baseline(TypedDict, total=False):
    throughput_tok_s_per_gpu: float
    accuracy: float
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    ttft_e2el_source: str         # state_workspace / runs_baseline_disk / unavailable
    config_path: str | None
    benchmark_report_path: str | None
    attempts_history: list[BaselineAttemptSummary]
    failure_streak: int
    invocation: BenchmarkInvocation
    workload_dims: WorkloadDims


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
    ttft_e2el_source: str         # current_best / validate_stack_disk / stack_top_disk / unavailable
    invocation: BenchmarkInvocation
    closing_phase_entered: bool
    closing_started_unix: float
    closing_report_task_id: str


# ---------------------------------------------------------------------------
# §5 Phase timeline — chronological events
# ---------------------------------------------------------------------------
class PhaseEvent(TypedDict, total=False):
    ts: str
    action: str                   # baseline / profile / backends / params / sweep / validate_stack / kernel_opt / select_kernels / integrate / tracelens_analysis / closing
    task_id: str
    kernel_id: str | None         # only for kernel-owned actions
    status: str                   # succeeded / failed
    decision: str                 # promoted / discarded / salvaged / no_promote / error / KEEP / PARTIAL / REVERT
    key_metric: float | None
    key_metric_kind: str | None
    workspace: str | None
    error_class: str | None
    extras: dict[str, Any]
    # v1.1 additions — per-event timing so the timeline conveys total
    # wall-clock cost without consumers having to walk benchmark_report.json
    # on disk. Both fields are optional and default to None when unknown.
    duration_seconds: float | None  # workload wall-clock seconds
    ended_ts_utc: str | None        # ts + duration_seconds (iso8601, UTC)


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


class VerificationSummary(TypedDict, total=False):
    micro_speedup: float | None
    compile_passed: bool | None
    correctness_passed: bool | None
    best_artifact_path: str | None


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
    status: str                   # succeeded / failed / error (per-attempt)
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
    proposal_reasons: list[str]
    verification_summary: VerificationSummary
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
    # v1.1 additions — TraceLens roofline merge (set when category_data
    # operations can be matched by name; left absent when no roofline
    # signal is available for this kernel).
    efficiency_percent: float | None
    bound_type: str | None
    tflops_achieved: float | None
    flops_per_byte: float | None
    library: str | None


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
    # validated / single_source / reconstructed / missing
    method: str
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


# ---------------------------------------------------------------------------
# v1.1 — decision journal + kernel profiling
# ---------------------------------------------------------------------------
class VariantDecision(TypedDict, total=False):
    name: str
    fingerprint: str
    extra_sglang_args: str
    extra_envs: dict[str, str]
    status: str                   # succeeded / failed / skipped
    output_throughput: float | None
    gain_pct_vs_base: float | None
    gain_pct_vs_current_best: float | None
    outcome: str                  # tested / round_winner / promoted / rejected
    reject_reason: str | None     # not_keep / combo_conflict / ...
    benchmark_report_path: str | None
    invocation: BenchmarkInvocation
    # v1.1 additions — per-variant duration (from the variant's own
    # benchmark_report.json) and the human-readable note recorded by
    # the round winner (e.g. "retry_alt_value_larger", "new_family_…").
    duration_seconds: float | None
    decision_note: str | None


class RoundDecision(TypedDict, total=False):
    outcome: str                    # promoted / discarded
    best_variant_name: str | None
    gain_vs_cb_pct: float | None
    best_gain_pct_vs_base: float | None
    promotion_rule: str | None      # single_shot / cross_round_consistent / accuracy_blocked / below_threshold
    promotion_rule_detail: str | None
    keep_threshold_pct: float | None
    accuracy_gate_passed: bool | None
    variants_tested_count: int | None


class DecisionJournalEntry(TypedDict, total=False):
    ts: str
    phase: str                      # params / backends
    round_id: str | None
    task_id: str | None
    workspace: str | None
    baseline_ref_tput: float | None
    current_best_tput: float | None
    keep_threshold_pct: float | None
    variants: list[VariantDecision]
    round_decision: RoundDecision


class KernelProfilingLaunch(TypedDict, total=False):
    framework_args: str
    framework_args_source: str
    extra_envs: dict[str, str]
    tracelens_patched: bool | None


class KernelProfilingArtifacts(TypedDict, total=False):
    benchmark_report_path: str | None
    trace_paths: list[str]
    kernel_summary_csv: str | None
    kernel_candidates_json: str | None
    tracelens_status_json: str | None
    tracelens_log: str | None


class KernelProfilingOutputs(TypedDict, total=False):
    tool: str                       # tracelens_analysis / magpie_torch_profiler
    # Each ``top_kernels`` entry is a free-form ``dict[str, Any]`` (deliberately
    # unconstrained so the schema doesn't force every collector branch into a
    # uniform shape). v1.1 collectors emit at minimum:
    #   ``kernel_id``, ``name``, ``gpu_pct``, ``duration_us``, ``bottleneck``,
    # plus when TraceLens roofline data is available:
    #   ``efficiency_percent``, ``bound_type``, ``flops_per_byte``,
    #   ``tflops_achieved``, ``percent_of_total``, ``arithmetic_intensity``,
    #   ``library``, ``operation_count``.
    # Consumers MUST treat any subset of these keys as optional.
    top_kernels: list[dict[str, Any]]
    analysis_summary: str | None


class KernelProfilingRun(TypedDict, total=False):
    run_id: str
    ts: str
    task_id: str
    framework: str | None
    profile_config_path: str | None
    launch: KernelProfilingLaunch
    artifacts: KernelProfilingArtifacts
    outputs: KernelProfilingOutputs
    # v1.1 P2-3 addition — derived end-of-run wall-clock (ISO8601 UTC)
    # and total seconds. Populated when the underlying status JSON
    # carries ``ended_at`` / ``duration_seconds`` (new kernel-agent
    # runs). Left absent on historical sessions.
    ended_ts_utc: str | None
    duration_seconds: float | None


# ---------------------------------------------------------------------------
# v1.1 P2-1 — kernel decision path (per-kid causal chain across
#             select_kernels → kernel_opt → integrate → validate_stack)
# ---------------------------------------------------------------------------
class KernelDecisionStep(TypedDict, total=False):
    kid: str                       # kernel id (orchestrator alias, e.g. k001)
    kernel_name: str               # human-readable name when known
    step: str                      # "select" | "kernel_opt" | "integrate" | "validate"
    backend: str | None            # geak / oob — only meaningful for kernel_opt
    ts: str                        # ISO8601 UTC, "" if unknown
    duration_seconds: float | None
    ended_ts_utc: str | None
    task_id: str
    workspace: str | None
    # Free-form decision label coming straight from the underlying
    # audit entry: ``promoted`` / ``discarded`` / ``rejected`` /
    # ``skipped`` / ``KEEP`` / ``PARTIAL`` / ``REVERT`` / …
    outcome: str
    decision_note: str
    gain_pct: float | None
    speedup: float | None
    extras: dict[str, Any]


class KernelDecisionPathSummary(TypedDict, total=False):
    total_steps: int
    backends_attempted: list[str]   # e.g. ["geak", "oob"]
    final_outcome: str              # last step's outcome
    total_duration_seconds: float | None


class KernelDecisionPathEntry(TypedDict, total=False):
    kid: str
    kernel_name: str
    steps: list[KernelDecisionStep]
    summary: KernelDecisionPathSummary


# ---------------------------------------------------------------------------
# v1.1 — data provenance (per-section source artifact probes)
# ---------------------------------------------------------------------------
# ``data_provenance`` lets consumers answer the "why is this section
# empty?" question without having to walk the session_dir themselves.
# Each entry records, for one logical section of the breakdown,
# (a) whether the section emitted any data, and
# (b) for each source artifact the collector relies on, whether that
#     artifact actually exists on disk (or whether the relevant env
#     var was configured). Probes are stat / glob only — no file
#     contents are read, so producing the provenance block is cheap.
#
# Status values:
#   * ``complete`` — every ``required`` probe found a hit. The
#     section may still be ``populated=False`` (e.g. ``sweep`` never
#     ran), which the consumer reads as "data sources are present
#     but the session never produced this kind of activity".
#   * ``partial``  — at least one ``required`` probe missing AND the
#     section produced some output. The section is partially
#     reconstructible from whatever was available.
#   * ``empty``    — at least one ``required`` probe missing AND the
#     section is empty. ``missing_required`` enumerates exactly
#     which source roles are absent so the operator knows what
#     would have to be re-collected.
class FileSourceProbe(TypedDict, total=False):
    path: str                    # relative-to-session_dir path or glob pattern;
                                 # env vars use ``env:<NAME>`` form.
    role: str                    # human-readable description of what the
                                 # artifact provides (e.g. "Magpie yaml").
    required: bool               # True if the section's collector cannot
                                 # function without this source.
    found: bool                  # glob hit count > 0 (or env var set).
    found_count: int             # number of paths matched (1 for env probes).
    representative_path: str | None  # first hit's relative path (or env value
                                     # for env probes), None when not found.
    note: str | None             # extra context (e.g. "permission denied",
                                 # "value masked", ...).


class SectionProvenance(TypedDict, total=False):
    section: str                 # section identifier (matches the breakdown
                                 # key, e.g. ``baseline`` / ``decision_journal``
                                 # / ``kernel_profiling`` / ``roofline`` / ...).
    status: str                  # ``complete`` / ``partial`` / ``empty``.
    populated: bool              # True iff the corresponding breakdown section
                                 # carries non-trivial data (list with items,
                                 # dict with at least one meaningful key, ...).
    sources: list[FileSourceProbe]
    missing_required: list[str]  # simplified list of ``role`` strings whose
                                 # required probe missed; empty when all
                                 # required sources are present.
    notes: list[str]             # free-form explanations (e.g. "baseline run
                                 # failed → benchmark_report.json absent").


class SessionBreakdown(TypedDict, total=False):
    schema_version: str
    exported_at_utc: str
    exporter_version: str
    detail_level: str               # standard / verbose
    # ``coverage`` records which of the two canonical input files
    # (``state.json`` + ``manifest.json``) were available when the
    # breakdown was built. Consumers can use this to distinguish a real
    # session run (``full``) from a post-orchestrator output directory
    # that lacks the in-flight session state (``shell_only``):
    #   ``full``        — both state.json and manifest.json present
    #   ``partial``     — exactly one of the two was present
    #   ``shell_only``  — neither present; emitted payload is best-effort
    #                     file-system walk only (no kernel lifecycle,
    #                     no decision journal, no attribution, ...).
    # Field is optional for backwards compatibility — older breakdowns
    # produced by exporters < this revision will simply not carry it.
    coverage: str

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

    decision_journal: list[DecisionJournalEntry]
    kernel_profiling: list[KernelProfilingRun]
    # v1.1 P2-1 addition — per-kid causal chain. Empty list when the
    # session ran no kernel selection / optimization / integration.
    kernel_decision_path: list[KernelDecisionPathEntry]
    # v1.1 addition — per-section source-artifact provenance. Each entry
    # explains why a section is empty (or partial) by listing the
    # required / optional source files (or env vars) the collector
    # consulted and whether each one was actually present. Optional for
    # backwards compatibility — older breakdowns will not carry the
    # field, and downstream consumers MUST treat its absence as "no
    # provenance information available" rather than an error.
    data_provenance: list[SectionProvenance]

    warnings: list[str]
    source_files: SourceFiles


__all__ = [
    "SCHEMA_VERSION",
    "AdoptedKernel",
    "DecisionJournalEntry",
    "Attribution",
    "Baseline",
    "BaselineAttemptSummary",
    "BenchmarkInvocation",
    "CapabilityEntry",
    "CapabilitySummary",
    "CriticIteration",
    "CriticRobustness",
    "DetectedKernel",
    "FileSourceProbe",
    "Final",
    "GpuMonitorAggregate",
    "Invocation",
    "KernelLifecycle",
    "KernelDecisionPathEntry",
    "KernelDecisionPathSummary",
    "KernelDecisionStep",
    "KernelProfilingArtifacts",
    "KernelProfilingLaunch",
    "KernelProfilingOutputs",
    "KernelProfilingRun",
    "KernelMetadata",
    "OptimizedKernel",
    "ParamSearch",
    "ParamSearchEntry",
    "ParamSearchLedger",
    "PhaseEvent",
    "RecommendedKernel",
    "RejectedKernel",
    "RobustnessSignal",
    "RoundDecision",
    "SectionProvenance",
    "VariantDecision",
    "VerificationSummary",
    "WorkloadDims",
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
