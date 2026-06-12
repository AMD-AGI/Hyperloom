# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Schema (TypedDict shape) for ``session_breakdown.json``.

The single contract between ``inference_optimizer`` and downstream
consumers. All fields are optional-by-convention (consumers treat missing
data as "not available", never fabricate); the wire shape is plain JSON;
``schema_version`` bumps only on breaking changes, not additive fields.
"""

from __future__ import annotations

from typing import Any, TypedDict

#: breakdown schema version. v2 adds specialist_runs, the specialist
#: capability row, kb_writes_summary, the action_timeline alias, the
#: explore_search ledger, and the kernel_optimization_summary /
#: conc_sweep_summary sections. Additive-only, so v1 readers still parse v2.
SCHEMA_VERSION = "hyperloom.session_breakdown.v2"


# §1 Session metadata
class SessionMeta(TypedDict, total=False):
    """Identity, timing, and host context for one optimization session.

    Captures the metadata describing the run that produced the breakdown: its
    identifiers, lifecycle timestamps, stop reason, and runtime environment.

    Attributes:
        session_id (str): Hyperloom internal id (``manifest.session_id``).
        claw_session_id (str | None): SaFE / Claw session id (env ``CLAW_SESSION_ID``).
        sandbox_user_id (str | None): Sandbox user identifier, if any.
        created_at_utc (str): ISO UTC timestamp when the session started.
        ended_at_utc (str): ISO UTC timestamp when the session ended.
        stop_reason (str): Why the run stopped (``target_reached`` /
            ``time_exhausted`` / ``no_more_leverage`` / ``max_ticks`` /
            ``baseline_failed`` / ...).
        max_minutes (int): Configured time budget in minutes.
        elapsed_minutes (float): Wall-clock minutes the session ran.
        host (str): Hostname the session executed on.
        code_revision (str): Source revision of the optimizer.
        pid (int): Process id of the optimizer.
        session_dir (str): Absolute path to the session working directory.
        tick_count (int): Number of orchestration ticks executed.
        image (str | None): Fully-qualified container image, or None if unset.
    """
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


# §2 Workload configuration
class WorkloadObjective(TypedDict, total=False):
    """Optimization goal the session was asked to pursue.

    Attributes:
        kind (str): Objective type (``gain_pct`` / ``tput`` / ``baseline`` /
            ``time_only``).
        value (Any): Goal value — a float target, a string (e.g.
            ``target_baseline_dir``), or None when not applicable.
    """
    kind: str                     # gain_pct / tput / baseline / time_only
    value: Any                    # float or str (target_baseline_dir) or None


class Workload(TypedDict, total=False):
    """Model, framework, and serving configuration under optimization.

    Describes the inference workload (model + framework + parallelism + shape)
    plus the objective that defines success for the run.

    Attributes:
        framework (str): Serving framework (``sglang`` / ``vllm`` / ``atom``).
        framework_version (str): Version string of the framework.
        model_name (str): Human-readable model name.
        model_path (str): Filesystem or registry path to the model weights.
        model_class (str): Model architecture class.
        gpu_type (str): GPU SKU (``mi300x`` / ``mi325x`` / ``mi355x``).
        tp (int | None): Tensor-parallel degree, or None if unset.
        conc (int | None): Request concurrency, or None if unset.
        isl (int | None): Input sequence length, or None if unset.
        osl (int | None): Output sequence length, or None if unset.
        max_model_len (int | None): Max context length, or None if unset.
        precision (str): Numeric precision of the served model.
        objective (WorkloadObjective): The optimization goal for the run.
    """
    framework: str                # sglang / vllm / atom
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


# §3 Baseline
class BaselineAttemptSummary(TypedDict, total=False):
    """One recorded attempt to establish the baseline measurement.

    Attributes:
        ts (str): ISO UTC timestamp of the attempt.
        task_id (str): Orchestrator task id for the attempt.
        status (str): Outcome status of the attempt.
        decision (str): Decision taken (e.g. promoted / discarded).
        key_metric (float | None): Headline metric value, or None if absent.
        workspace (str | None): Benchmark workspace path, or None.
        error_class (str | None): Error classification on failure, or None.
    """
    ts: str
    task_id: str
    status: str
    decision: str
    key_metric: float | None
    workspace: str | None
    error_class: str | None
    # Real failure text from the executor; lets RCA read the cause
    # without crawling server logs. None on success / reconstruction.
    error_excerpt: str | None
    stderr_tail: str | None
    stderr_log_path: str | None


class BenchmarkInvocation(TypedDict, total=False):
    """Replayable record of how a benchmark variant was launched (server cmd + envs + config).

    ``extra_envs`` is allowlist-filtered to keep secrets out of the JSON.
    """
    framework_args: str           # e.g. "python -m sglang.launch_server --model ... --tp 8"
    framework_args_source: str
    # framework_args_source vocab: log_non_default_args / log_args_line /
    # log_python_cmd / yaml_cmd / yaml_benchmark / unknown.
    extra_envs: dict[str, str]    # allowlisted env vars only (no secrets)
    config_path: str | None       # baseline_config.with_envs.yaml or variant config
    server_log_path: str | None   # for debug


class Baseline(TypedDict, total=False):
    """Pre-optimization reference performance for the workload.

    The baseline against which all gains are computed, including latency
    sub-metrics, attempt history, and the replayable launch invocation.

    Attributes:
        throughput_tok_s_per_gpu (float): Baseline throughput (tok/s per GPU).
        accuracy (float): Baseline accuracy score.
        ttft_mean_ms (float | None): Mean time-to-first-token (ms), or None.
        e2el_mean_ms (float | None): Mean end-to-end latency (ms), or None.
        ttft_e2el_source (str): Provenance of the latency metrics
            (``state_workspace`` / ``runs_baseline_disk`` / ``unavailable``).
        config_path (str | None): Path to the baseline config, or None.
        benchmark_report_path (str | None): Path to the benchmark report, or None.
        attempts_history (list[BaselineAttemptSummary]): Recorded baseline attempts.
        failure_streak (int): Consecutive baseline failures.
        invocation (BenchmarkInvocation): Replayable launch record.
    """
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


# §4 Final state — SaFE contract core
class Final(TypedDict, total=False):
    """Final validated optimization state — the SaFE contract core.

    Records the best validated result of the session: throughput, cumulative
    gain, the applied server-arg/env stack, and closing-phase bookkeeping.

    Attributes:
        throughput_tok_s_per_gpu (float | None): Final throughput (tok/s/GPU), or None.
        cumulative_gain_pct_validated (float): Validated cumulative gain percent.
        cumulative_gain_pct_per_round_sum (float): Sum of per-round gain percents.
        validated_at_stack_len (int): Stack depth at which validation occurred.
        validated_ts (str): ISO UTC timestamp of the validation.
        stack_changed_after_validation (bool): Whether the stack changed post-validation.
        extra_server_args (str): Final extra server-arg CLI fragment.
        extra_envs (dict[str, Any]): Final extra env vars applied.
        action_path (list[str]): Ordered ``action:variant`` labels from the stack.
        ttft_mean_ms (float | None): Mean time-to-first-token (ms), or None.
        e2el_mean_ms (float | None): Mean end-to-end latency (ms), or None.
        ttft_e2el_source (str): Provenance of the latency metrics (``current_best`` /
            ``validate_stack_disk`` / ``stack_top_disk`` / ``unavailable``).
        invocation (BenchmarkInvocation): Replayable launch record for the final state.
        closing_phase_entered (bool): Whether the closing phase was entered.
        closing_started_unix (float): Unix time the closing phase started.
        closing_report_task_id (str): Task id of the closing report.
    """
    throughput_tok_s_per_gpu: float | None
    cumulative_gain_pct_validated: float
    cumulative_gain_pct_per_round_sum: float
    validated_at_stack_len: int
    validated_ts: str
    stack_changed_after_validation: bool
    extra_server_args: str
    extra_envs: dict[str, Any]
    action_path: list[str]        # ordered list of action:variant labels from optimization_stack
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    ttft_e2el_source: str         # current_best / validate_stack_disk / stack_top_disk / unavailable
    invocation: BenchmarkInvocation
    closing_phase_entered: bool
    closing_started_unix: float
    closing_report_task_id: str


# §5 Phase timeline — chronological events
class PhaseEvent(TypedDict, total=False):
    """One chronological event in the optimization timeline.

    A single action attempt (profile, backend trial, kernel opt, validation,
    etc.) with its outcome and optional contextual extras.

    Attributes:
        ts (str): ISO UTC timestamp of the event.
        action (str): Action kind (``baseline`` / ``profile`` / ``backends`` /
            ``params`` / ``sweep`` / ``validate_stack`` / ``kernel_opt`` /
            ``trace_analyze`` / ``integrate``).
        task_id (str): Orchestrator task id.
        kernel_id (str | None): Kernel id for kernel-owned actions, else None.
        status (str): Outcome (``succeeded`` / ``failed``).
        decision (str): Decision label (``promoted`` / ``discarded`` /
            ``salvaged`` / ``no_promote`` / ``error`` / ``KEEP`` / ``PARTIAL`` /
            ``REVERT``).
        key_metric (float | None): Headline metric value, or None.
        key_metric_kind (str | None): Type/label of the key metric, or None.
        workspace (str | None): Benchmark workspace path, or None.
        error_class (str | None): Error classification on failure, or None.
        extras (dict[str, Any]): Action-specific extra fields.
    """
    ts: str
    action: str                   # baseline / profile / backends / params / sweep / validate_stack / kernel_opt / trace_analyze / integrate
    task_id: str
    kernel_id: str | None         # only for kernel-owned actions
    status: str                   # succeeded / failed
    decision: str                 # promoted / discarded / salvaged / no_promote / error / KEEP / PARTIAL / REVERT
    key_metric: float | None
    key_metric_kind: str | None
    workspace: str | None
    error_class: str | None
    phase: str                    # declared phase (journal-sourced events); "" otherwise
    change: str                   # human-readable change summary (journal) or action key
    extras: dict[str, Any]


# §6 Capability summary — Capability cards in UI
class CapabilityEntry(TypedDict, total=False):
    status: str                   # kept / reverted / tried / attempted / not_attempted / not_configured / failed / completed
    attempts: int
    keeps: int                    # geak/oob: kernels adopted at integrate (NOT micro-only KEEP)
    reverts: int                  # geak/oob: micro-KEPT kernels reverted at integrate (e2e regressed)
    e2e_gain_pct: float | None    # geak/oob: best end-to-end integrate gain for this lane's kernel
    tested: int                   # for backends/params/explore: distinct variants tested
    best_gain_pct: float | None
    reason: str                   # human readable, e.g. "kernel-claude only this run"
    # explore-specific:
    keep_unstable_count: int      # KEEP'd variants evicted by inlined stack rebench
    winners_history: int          # cumulative explore_search.winners_history length
    # specialist-row only — per-domain split keyed by SpecialistDomain.key;
    # every catalogue domain is seeded not_attempted for presence-free iteration.
    by_specialist: dict[str, "CapabilityEntry"]


class CapabilitySummary(TypedDict, total=False):
    """Per-capability roll-up powering the dashboard capability cards.

    Holds one :class:`CapabilityEntry` per capability family. ``backends`` /
    ``params`` / ``validate_stack`` are retained as compatibility aliases of
    the primary ``explore`` row.

    Attributes:
        geak (CapabilityEntry): GEAK kernel-generation capability.
        oob (CapabilityEntry): Out-of-box kernel backend capability.
        explore (CapabilityEntry): Primary explore (param/backend search) row.
        backends (CapabilityEntry): Compatibility alias for backend exploration.
        params (CapabilityEntry): Compatibility alias for param exploration.
        sweep (CapabilityEntry): Concurrency/shape sweep capability.
        validate_stack (CapabilityEntry): Compatibility alias for stack validation.
        specialist (CapabilityEntry): Specialist sub-agent capability; ``tested`` =
            total proposals across rounds, ``keeps`` = proposals kept,
            ``attempts`` = number of dispatch rounds.
    """
    geak: CapabilityEntry
    oob: CapabilityEntry
    # primary explore row; backends/params/validate_stack are compat aliases.
    explore: CapabilityEntry
    backends: CapabilityEntry
    params: CapabilityEntry
    sweep: CapabilityEntry
    validate_stack: CapabilityEntry
    # specialist row: tested=proposals_total, keeps=proposals_kept, attempts=rounds.
    specialist: CapabilityEntry


# §7 / §8 GEAK / OOB invocations
class KernelMetadata(TypedDict, total=False):
    """Descriptive metadata for a kernel targeted by a backend invocation.

    Attributes:
        name (str): Kernel name.
        source_file (str): Source file the kernel lives in.
        shapes (list[dict[str, Any]]): Input/output shape descriptors.
        gpu_pct (float | None): Share of total GPU time (0..100), or None.
        arithmetic_intensity (float | None): FLOPs per byte, or None.
    """
    name: str
    source_file: str
    shapes: list[dict[str, Any]]
    gpu_pct: float | None
    arithmetic_intensity: float | None


class Invocation(TypedDict, total=False):
    """One backend invocation for one kernel (same shape for GEAK/OOB; ``backend`` distinguishes)."""
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


# §9 Kernel lifecycle (4+1 stages)
class DetectedKernel(TypedDict, total=False):
    """A hot kernel surfaced by profiling (stage 1 of the kernel lifecycle).

    Attributes:
        kernel_id (str): Kernel identifier.
        name (str): Kernel name.
        gpu_pct (float | None): Share of total GPU time (0..100), or None.
        time_ms (float | None): Kernel duration in milliseconds, or None.
        bottleneck (str): Bottleneck class (``compute`` / ``memory`` / ``comm``).
        arithmetic_intensity (float | None): FLOPs per byte, or None.
        reusable_native_kernel (bool): Whether a native kernel can be swapped in.
        source_file (str | None): Source file of the kernel, or None.
        detected_from_task (str): Profile task id that surfaced the kernel.
        benchmark_report_path (str): Path to the benchmark report.
    """
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
    # lifecycle stamps (added by _collect_detected_kernels)
    selected_for_optimization: bool
    geak: dict[str, Any] | None   # {attempts, best_speedup, decision, last_status}
    oob: dict[str, Any] | None
    adopted_by: str | None        # geak / oob / kernel_agent / None
    final_decision: str           # kept / reverted / rejected / attempted / not_optimized
    integrate_gain_pct: float | None  # e2e (integrate) gain; negative => regressed -> reverted


class RecommendedKernel(TypedDict, total=False):
    """A kernel recommended for optimization (stage 2 of the lifecycle).

    Attributes:
        kernel_id (str): Kernel identifier.
        name (str): Kernel name.
        gpu_pct (float | None): Share of total GPU time (0..100), or None.
        recommended_backends (list[str]): Suggested optimization backends.
        recommended_actions (list[str]): Suggested optimization actions.
        bottleneck (str): Bottleneck class (compute / memory / comm).
        reusable_native_kernel (bool): Whether a native kernel can be swapped in.
    """
    kernel_id: str
    name: str
    gpu_pct: float | None
    recommended_backends: list[str]
    recommended_actions: list[str]
    bottleneck: str
    reusable_native_kernel: bool


class OptimizedKernel(TypedDict, total=False):
    """A kernel that went through optimization (stage 3 of the lifecycle).

    Attributes:
        kernel_id (str): Kernel identifier.
        backend (str): Winning backend (``geak`` / ``claude`` / ``codex``, best-of).
        total_attempts (int): Total optimization attempts.
        successful_attempts (int): Attempts that succeeded.
        best_micro_speedup (float | None): Best micro-benchmark speedup, or None.
        last_decision (str): Decision of the last attempt.
        best_artifact_path (str | None): Path to the best artifact, or None.
        attempts_summary (list[dict[str, Any]]): Per-attempt summary rows.
    """
    kernel_id: str
    backend: str                  # geak / claude / codex (best-of)
    total_attempts: int
    successful_attempts: int
    best_micro_speedup: float | None
    last_decision: str
    best_artifact_path: str | None
    attempts_summary: list[dict[str, Any]]


class AdoptedKernel(TypedDict, total=False):
    """A kernel optimization adopted into the stack (stage 4 of the lifecycle).

    Attributes:
        kernel_id (str): Kernel identifier.
        patch_path (str): Path to the adopted patch.
        target_file (str): File the patch applies to.
        extra_server_args (str): Server-arg fragment introduced by the adoption.
        e2e_gain_pct (float | None): End-to-end gain percent, or None.
        validated (bool): Whether the adoption was validated.
        last_status (str): Last recorded status.
        adopted_at (str): ISO UTC timestamp of adoption.
        attempt_count (int): Number of attempts before adoption.
    """
    kernel_id: str
    patch_path: str
    target_file: str
    extra_server_args: str
    e2e_gain_pct: float | None
    validated: bool
    last_status: str
    adopted_at: str
    attempt_count: int


class RejectedKernel(TypedDict, total=False):
    """A kernel optimization that was tried but not adopted (the +1 stage).

    Attributes:
        kernel_id (str): Kernel identifier.
        reason (str): Why the kernel optimization was rejected.
        patch_path (str | None): Path to the rejected patch, or None.
        target_file (str | None): File the patch targeted, or None.
        attempt_count (int): Number of attempts made.
        best_gain_pct (float | None): Best gain percent observed, or None.
        ts (str): ISO UTC timestamp of rejection.
    """
    kernel_id: str
    reason: str
    patch_path: str | None
    target_file: str | None
    attempt_count: int
    best_gain_pct: float | None
    ts: str


class KernelLifecycle(TypedDict, total=False):
    """Kernels grouped by lifecycle stage (4+1 stages).

    Tracks kernels as they move from detection through recommendation,
    optimization, and finally adoption or rejection.

    Attributes:
        detected (list[DetectedKernel]): Hot kernels surfaced by profiling.
        recommended (list[RecommendedKernel]): Kernels recommended for optimization.
        optimized (list[OptimizedKernel]): Kernels that were optimized.
        adopted (list[AdoptedKernel]): Optimizations adopted into the stack.
        rejected (list[RejectedKernel]): Optimizations tried but not adopted.
    """
    detected: list[DetectedKernel]
    recommended: list[RecommendedKernel]
    optimized: list[OptimizedKernel]
    adopted: list[AdoptedKernel]
    rejected: list[RejectedKernel]


# §10 Param search
class ParamSearchEntry(TypedDict, total=False):
    """One candidate variant from ``explore_search.{tested,accepted,rejected}``.

    Records a single param/backend variant that was evaluated, its launch
    fingerprint, the measured throughput, and the resulting gain.

    Attributes:
        name (str): Variant name.
        fingerprint (str): Content-hash deduplication key.
        extra_server_args (str): Server-arg CLI fragment for the variant.
        extra_envs (dict[str, Any]): Env vars set by the variant.
        output_throughput (float | None): Measured throughput, or None.
        gain_pct (float | None): Gain percent vs current best, or None.
        ts (str): ISO UTC timestamp of evaluation.
        status (str): Outcome (``accepted`` / ``rejected`` / ``tested``).
    """
    name: str
    fingerprint: str
    extra_server_args: str
    extra_envs: dict[str, Any]
    output_throughput: float | None
    gain_pct: float | None
    ts: str
    status: str                   # accepted / rejected / tested


class ParamSearchLedger(TypedDict, total=False):
    """Ledger of one explore family's tested/accepted/rejected variants.

    Attributes:
        schema_version (int): Ledger schema version.
        tested_count (int): Total number of variants tested.
        accepted (list[ParamSearchEntry]): Variants that were accepted.
        rejected (list[ParamSearchEntry]): Variants that were rejected.
        top_by_gain (list[ParamSearchEntry]): Best variants ordered by gain.
        winner_history (list[dict[str, Any]]): History of winning variants.
        no_promote_streak (int): Consecutive evaluations without a promotion.
    """
    schema_version: int
    tested_count: int
    accepted: list[ParamSearchEntry]
    rejected: list[ParamSearchEntry]
    top_by_gain: list[ParamSearchEntry]
    winner_history: list[dict[str, Any]]
    no_promote_streak: int


class ParamSearch(TypedDict, total=False):
    """Merged explore-search results across the param and backend families.

    Attributes:
        params (ParamSearchLedger): Ledger for the param-tuning family.
        backends (ParamSearchLedger): Ledger for the backend-tuning family.
        synergy_attempted (list[str]): Synergy combinations that were attempted.
        discovered_flags (dict[str, Any]): Flags discovered during search.
        backend_winners_history (list[dict[str, Any]]): History of backend winners.
    """
    params: ParamSearchLedger
    backends: ParamSearchLedger
    synergy_attempted: list[str]
    discovered_flags: dict[str, Any]
    backend_winners_history: list[dict[str, Any]]


# §11 Sweep
class SweepPoint(TypedDict, total=False):
    """One measured point in a concurrency/shape sweep grid.

    Attributes:
        variant_name (str): Name of the swept variant.
        conc (int | None): Concurrency at this point, or None.
        isl (int | None): Input sequence length, or None.
        osl (int | None): Output sequence length, or None.
        output_throughput_tok_s (float | None): Throughput (tok/s), or None.
        ttft_mean_ms (float | None): Mean time-to-first-token (ms), or None.
        tpot_mean_ms (float | None): Mean time-per-output-token (ms), or None.
        e2el_mean_ms (float | None): Mean end-to-end latency (ms), or None.
        status (str): Point status (``ok`` / ``skipped`` / ``failed``).
        benchmark_report_path (str | None): Path to the benchmark report, or None.
    """
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
    """Results of the concurrency/shape sweep across the variant grid.

    Attributes:
        grid_size (int): Number of points in the sweep grid.
        best_overall (dict[str, Any]): Best-performing point overall.
        best_for_each_conc (list[dict[str, Any]]): Best point per concurrency level.
        pareto_front (list[dict[str, Any]]): Pareto-optimal sweep points.
        all_variants (list[SweepPoint]): All measured sweep points.
        config_path (str | None): Path to the sweep config, or None.
    """
    grid_size: int
    best_overall: dict[str, Any]
    best_for_each_conc: list[dict[str, Any]]
    pareto_front: list[dict[str, Any]]
    all_variants: list[SweepPoint]
    config_path: str | None


# §12 Critic / Robustness
class CriticIteration(TypedDict, total=False):
    """One critic-agent review pass over a proposed change.

    Attributes:
        iter (int): Iteration index.
        ts (str): ISO UTC timestamp of the review.
        topic (str): What was reviewed (e.g. ``kernel_opt:k001`` / ``backends:flag_X``).
        verdict (str): Review verdict (``approve`` / ``reject`` / ``redirect`` /
            ``advise`` / ``needs_review``).
        summary (str): Human-readable review summary.
        request_path (str): Path to the review request artifact.
        judge_bundle_path (str): Path to the judge bundle.
        emit_path (str): Path to the emitted review output.
        review_path (str): Path to the review record.
    """
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
    """A fault/recovery event handled during the session.

    Attributes:
        ts (str): ISO UTC timestamp of the signal.
        signal (str): Signal type (``crash`` / ``stall`` / ``disk_full`` /
            ``cluster_fault`` / ...).
        action (str): Recovery action taken in response.
        workdir (str): Working directory associated with the signal.
    """
    ts: str
    signal: str                   # crash / stall / disk_full / cluster_fault / ...
    action: str                   # what was done
    workdir: str


class CriticRobustness(TypedDict, total=False):
    """Critic-review iterations and robustness signals for the session.

    Attributes:
        critic_iterations (list[CriticIteration]): Critic-agent review passes.
        robustness_signals (list[RobustnessSignal]): Fault/recovery events handled.
        kb_writes_summary (CriticKBWritesSummary): Counts of KB writes proxied
            through the critic's ``commit-review`` protocol (the Coordinator
            performs the writes; the critic only authors them).
    """
    critic_iterations: list[CriticIteration]
    robustness_signals: list[RobustnessSignal]
    # KB writes proxied through the critic's ``commit-review`` protocol.
    kb_writes_summary: "CriticKBWritesSummary"


# §13 Telemetry
class GpuMonitorAggregate(TypedDict, total=False):
    """Aggregated GPU power/thermal/clock telemetry over the session.

    Attributes:
        samples (int): Number of telemetry samples aggregated.
        avg_power_w (float): Average power draw (watts).
        max_power_w (float): Peak power draw (watts).
        avg_temp_c (float): Average temperature (Celsius).
        max_temp_c (float): Peak temperature (Celsius).
        avg_clock_mhz (float): Average clock frequency (MHz).
    """
    samples: int
    avg_power_w: float
    max_power_w: float
    avg_temp_c: float
    max_temp_c: float
    avg_clock_mhz: float


class LaneTimelineEntry(TypedDict, total=False):
    """One row of the legacy M6 lane occupancy summary (resource_lock capacity / holders / expired leases)."""
    lane: str
    capacity: int
    live_holders: int
    lease_expired_count: int


class Telemetry(TypedDict, total=False):
    """Pointers to telemetry artifacts and aggregated hardware metrics.

    Attributes:
        baseline_report_path (str | None): Path to the baseline report, or None.
        profile_report_paths (list[str]): Paths to profile reports.
        torch_trace_paths (list[str]): Paths to torch traces.
        system_profile_paths (list[str]): Paths to system profiles.
        server_log_paths (list[str]): Paths to server logs.
        gpu_monitor_aggregate (GpuMonitorAggregate): Aggregated GPU telemetry.
        lane_timeline (list[LaneTimelineEntry]): Per-lane capacity/occupancy summary.
    """
    baseline_report_path: str | None
    profile_report_paths: list[str]
    torch_trace_paths: list[str]
    system_profile_paths: list[str]
    server_log_paths: list[str]
    gpu_monitor_aggregate: GpuMonitorAggregate
    # per-lane capacity / occupancy summary.
    lane_timeline: list[LaneTimelineEntry]


# §14 Attribution
class StackGainEntry(TypedDict, total=False):
    """One KEEP/validation event with its incremental gain contribution.

    Records how a single stack change moved the cumulative gain, used to
    attribute total gain across the optimization stack.

    Attributes:
        ts (str): ISO UTC timestamp of the event.
        stack_len_before (int): Stack depth before the change.
        stack_len_after (int): Stack depth after the change.
        action (str): Action kind (``backends`` / ``params`` /
            ``kernel_opt:<kid>`` / ``validate_stack``).
        variant_name (str | None): Variant label, or None.
        cum_gain_before (float): Cumulative gain percent before the change.
        cum_gain_after (float): Cumulative gain percent after the change.
        delta_pct (float | None): Incremental gain percent; None when
            ``validate_stack`` re-baselined.
        extra_server_args (str): Server-arg fragment associated with the change.
    """
    ts: str
    stack_len_before: int
    stack_len_after: int
    action: str                   # backends / params / kernel_opt:<kid> / validate_stack
    variant_name: str | None
    cum_gain_before: float
    cum_gain_after: float
    delta_pct: float | None       # None when validate_stack re-baselined
    extra_server_args: str


class SourceBreakdown(TypedDict, total=False):
    """Validated total gain split by contributing source/family.

    Each ``*_pct_of_total`` field is the share of the validated total gain
    attributed to that source. The per-source values reconcile against
    ``validated_total_pct``.

    Attributes:
        geak_pct_of_total (float): Gain share from GEAK kernel rewrites.
        oob_pct_of_total (float): Gain share from out-of-box backends.
        explore_pct_of_total (float): Gain share from the primary explore family.
        framework_pr_pct_of_total (float): Gain share from FRAMEWORK_PR bake-ins.
        gemm_tuning_pct_of_total (float): Gain share from the FP8 GEMM tuner
            (0.0 on non-FP8 workloads or when the tuner produced no KEEP).
        backends_pct_of_total (float): Gain share from backend exploration.
        params_pct_of_total (float): Gain share from param exploration.
        sweep_pct_of_total (float): Gain share attributed to the sweep.
        validated_total_pct (float): Total validated gain percent.
    """
    geak_pct_of_total: float
    oob_pct_of_total: float
    # primary explore family bucket.
    explore_pct_of_total: float
    # FRAMEWORK_PR phase contribution (upstream-PR bake-ins), bucketed
    # separately so per-source totals reconcile against validated_total_pct.
    framework_pr_pct_of_total: float
    # GEMM_TUNING (deterministic FP8 GEMM tuner) gain, split from the
    # kernel family; always emitted (0.0 when it skipped / no KEEP).
    gemm_tuning_pct_of_total: float
    backends_pct_of_total: float
    params_pct_of_total: float
    sweep_pct_of_total: float
    validated_total_pct: float


class PhaseBreakdownExplore(TypedDict, total=False):
    """v0.8 M7 (KB_design §3.12 §4.6) — explore-phase gain split by specialist domain.

    ``by_domain`` keys are normalized to the bare SpecialistDomain.key;
    non-specialist provenance is ``default_grid`` / ``llm_direct``, v1
    resumes are ``legacy_<action>``, empty falls back to ``unknown``.

    ``by_scope`` is the additive specialist-dial split (``domain`` /
    ``domains`` / ``freeform``); sessions that never recorded a ``scope``
    collapse into ``unspecified``. Omitted on pre-scope breakdowns.
    """
    total_gain_pct: float
    by_domain: dict[str, float]
    by_scope: dict[str, float]


class PhaseBreakdownKernel(TypedDict, total=False):
    """v0.8 M7 — kernel-phase gain split by ``kernel_id`` (KB_design §3.12 §4.6)."""
    total_gain_pct: float
    by_kernel_id: dict[str, float]


class PhaseBreakdownFrameworkPr(TypedDict, total=False):
    """FRAMEWORK_PR phase gain split by adopted PR; ``by_pr`` keys on ``variant_name`` (``"?"`` when empty)."""
    total_gain_pct: float
    by_pr: dict[str, float]


class PhaseBreakdownGemmTuning(TypedDict, total=False):
    """KERNEL-entry FP8 GEMM tuning gain split by ``tuned_file`` (falls back to ``variant_name`` then ``"?"``)."""
    total_gain_pct: float
    by_tuned_file: dict[str, float]


class PhaseBreakdown(TypedDict, total=False):
    """v0.8 M7 per-phase gain attribution (KB_design §3.13 M7 §6).

    Splits the validated total gain across the phase state machine, with each
    phase carrying its own per-sub-bucket breakdown.

    Attributes:
        prelude (PhaseBreakdownExplore): PRELUDE phase gain (always 0 by definition).
        framework_pr (PhaseBreakdownFrameworkPr): FRAMEWORK_PR phase gain.
        explore (PhaseBreakdownExplore): EXPLORE phase gain by domain.
        kernel (PhaseBreakdownKernel): KERNEL phase gain by ``kernel_id``.
        gemm_tuning (PhaseBreakdownGemmTuning): KERNEL-entry GEMM-tuning gain,
            bucketed separately from source-level kernel rewrites.
        sweep (PhaseBreakdownExplore): SWEEP phase gain (usually 0; measurement).
        close (PhaseBreakdownExplore): CLOSE phase gain (usually 0).
        unattributed (PhaseBreakdownExplore): Gain whose phase could not be inferred.
    """
    prelude: PhaseBreakdownExplore         # always 0 by definition
    framework_pr: PhaseBreakdownFrameworkPr  # PRELUDE → FRAMEWORK_PR → EXPLORE
    explore: PhaseBreakdownExplore
    kernel:  PhaseBreakdownKernel
    # GEMM_TUNING bucketed separately from source-level GEAK/OOB rewrite gain.
    gemm_tuning: PhaseBreakdownGemmTuning
    sweep:   PhaseBreakdownExplore         # usually 0 (sweep is measurement)
    close:   PhaseBreakdownExplore         # usually 0
    unattributed: PhaseBreakdownExplore    # gain whose phase couldn't be inferred


class Attribution(TypedDict, total=False):
    """Gain attribution across stack entries, sources, and phases.

    Attributes:
        gain_per_stack_entry (list[StackGainEntry]): Per-KEEP incremental gains.
        method (str): How attribution was computed (``validated`` /
            ``single_source`` / ``reconstructed`` / ``missing``).
        source_breakdown (SourceBreakdown): Gain split by contributing source.
        phase_breakdown (PhaseBreakdown): Gain split per optimization phase.
        notes (list[str]): Human-readable caveats about the attribution.
    """
    gain_per_stack_entry: list[StackGainEntry]
    # validated / single_source / reconstructed / missing
    method: str
    source_breakdown: SourceBreakdown
    phase_breakdown: PhaseBreakdown
    notes: list[str]              # human-readable caveats


# §16 Phase segments — phase state machine
class PhaseSegment(TypedDict, total=False):
    """One contiguous segment of the v0.8 M2 phase state machine.

    Captures a phase the session occupied between two transitions, including
    the entry evidence and the events that fell within the segment window.

    Attributes:
        phase (str): Phase name (``PRELUDE`` / ``FRAMEWORK_PR`` / ``EXPLORE`` /
            ``KERNEL`` / ``SWEEP`` / ``CLOSE``).
        from_phase (str): Previous phase (empty for the first segment).
        entered_ts (str): ISO UTC timestamp of entry.
        entered_unix (float | None): Unix time of entry, or None.
        exit_ts (str): ISO UTC timestamp of the next transition; "" if current.
        exit_reason (str): Transition reason; "" for the current segment.
        evidence (dict[str, Any]): Entry evidence snapshot at transition time.
        actions (list[PhaseEvent]): Timeline events with ts in [entered, exit).
        elapsed_seconds (float | None): Segment duration in seconds, or None.
    """
    phase: str                 # PRELUDE / FRAMEWORK_PR / EXPLORE / KERNEL / SWEEP / CLOSE
    from_phase: str            # previous phase (empty for first segment)
    entered_ts: str            # iso UTC of entry
    entered_unix: float | None
    exit_ts: str               # iso UTC of next transition; "" for current segment
    exit_unix: float | None    # unix epoch of next transition; None for current segment
    exit_reason: str           # transition reason vocab entry; "" for current segment
    evidence: dict[str, Any]   # entry evidence (snapshot at transition time)
    events: list[dict[str, Any]]  # non-transition sub-events folded into this phase
    actions: list[PhaseEvent]  # phase_timeline events attributed to this phase
    elapsed_seconds: float | None


# §15 KB Provenance — RecipeKB / PR Monitor integration
class KBQueueStats(TypedDict, total=False):
    """Depth statistics for the on-disk KB write queues.

    Attributes:
        pending_lines (int): Current depth of ``.kb_pending.ndjson``.
        flushed_bookmarks (int): Drain-bookmark rows in ``.kb_flushed.ndjson``.
        dead_letter_lines (int): Rows in ``.kb_dead_letter.ndjson``.
    """
    pending_lines: int             # current depth of .kb_pending.ndjson
    flushed_bookmarks: int         # rows in .kb_flushed.ndjson (drain bookmarks)
    dead_letter_lines: int         # rows in .kb_dead_letter.ndjson


class KBFlusherStatus(TypedDict, total=False):
    """``kb_provenance.flusher_status`` (v0.8 KB_gaps/Dead-E): boot marker merged with a live pid probe."""
    enabled: bool                  # cli flag (false when --no-kb-flusher or --degraded-kb)
    spawned: bool                  # daemon was actually subprocess.Popen'd this boot
    alive: bool                    # live pid probe at breakdown emit time
    pid: int | None
    cortex_kb_url: str | None
    interval_sec: float
    batch_size: int
    reason: str                    # boot-time spawn decision text
    ts: str                        # iso UTC of the boot marker
    pid_path: str                  # absolute path to .kb_flusher.pid


class WarmReplayOutcome(TypedDict, total=False):
    """GAP 1 — warm-recipe replay result. Empty {} when it never fired; else ``status`` + per-status fields."""
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
    """Cortex KB integration audit for the session.

    Covers warm-start context seeded from the KB, the warm-replay outcome,
    pending/created KB points, queue depth, and the flusher daemon status.

    Attributes:
        cortex_session_id (str): Cortex KB session id.
        warm_start_ts (str): ISO UTC timestamp of warm start.
        warm_start_recipe_seen (bool): Whether a warm recipe was seen.
        warm_start_recipe_tier (str): Tier of the seen warm recipe.
        warm_start_pitfall_count (int): Number of pitfalls injected at warm start.
        warm_start_lesson_count (int): Number of lessons injected at warm start.
        warm_replay (WarmReplayOutcome): Operator-visible warm-replay summary.
        warm_replay_attempted (bool): Whether a warm replay was attempted.
        warm_history_injected (bool): Whether warm history was injected.
        stack_fingerprint (dict[str, str]): Fingerprint of the optimization stack.
        pending_edges (list[KBPendingEdge]): Edges queued but not committed.
        queue (KBQueueStats): Depth stats for the KB write queues.
        audit_tail_count (int): Number of audit-tail entries.
        audit_status_counts (dict[str, int]): Audit entries counted by status.
        points_created (list[KBPointCreated]): KB points created this session.
        points_by_kind (dict[str, int]): Created-point counts by kind.
        commit_summary (KBCommitSummary): Outcome of committing the edges.
        flusher_status (KBFlusherStatus): KB flusher daemon lifecycle marker.
        kb_degraded_reason (str): KB soft-degrade reason (None / ``explicit_flag`` /
            ``ir3_auto``).
        pr_degraded_reason (str): PR Monitor soft-degrade reason (None /
            ``explicit_flag`` / ``ir3_auto``).
    """
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
    queue: KBQueueStats
    audit_tail_count: int
    audit_status_counts: dict[str, int]
    flusher_status: KBFlusherStatus
    # Soft-degrade audit: None / "explicit_flag" / "ir3_auto".
    kb_degraded_reason: str
    pr_degraded_reason: str


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
    body: str   # only set when CLI flag enabled


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
    # Retired field, kept (always empty) for claw-stats-service readers.
    kb_edge_ids: list[str]
    confidence_avg: float | None
    domain_breakdown: dict[str, SpecialistDomainBreakdown]
    transcripts: list[SpecialistTranscriptRef]
    notes: list[str]


# critic_robustness.kb_writes_summary sub-block
class CriticKBWritesSummary(TypedDict, total=False):
    """Summary of critic-agent ``commit-review`` outputs (Coordinator proxies these into ``kb_provenance``)."""
    total: int
    by_verdict: dict[str, int]   # KEEP / REVERT / NEEDS_INFO / ...


# Top-level shape
class SourceFiles(TypedDict, total=False):
    """Paths to the on-disk artifacts the breakdown was assembled from.

    Attributes:
        manifest (str): Path to the session manifest.
        state (str): Path to the session state file.
        baseline_report (str | None): Path to the baseline report, or None.
        profile_reports (list[str]): Paths to profile reports.
        sweep_reports (list[str]): Paths to sweep reports.
        kernel_attempts (list[str]): Paths to kernel attempt artifacts.
        critic_workdir (str | None): Critic working directory, or None.
        robustness_workdir (str | None): Robustness working directory, or None.
    """
    manifest: str
    state: str
    baseline_report: str | None
    profile_reports: list[str]
    sweep_reports: list[str]
    kernel_attempts: list[str]
    critic_workdir: str | None
    robustness_workdir: str | None


# Roofline — optimization-progress curve for the dashboard (spec §2):
# a stepped line from baseline through every KEEP against ceiling/target
# reference lines, all derived from ``state.json``.
class RooflineTrajectoryPoint(TypedDict, total=False):
    """One x/y/tooltip on the optimization-progress curve (first point is ``baseline``, rest from the KEEP stack)."""
    ts: str                          # iso UTC, x value
    tput: float                      # tok/s, y value
    label: str                       # "baseline" / variant_name
    action: str                      # "baseline" / "explore" / "kernel_opt" / ...
    gain_pct: float                  # cumulative gain vs baseline at this point
    flags: str                       # candidate_extra_server_args
    extra_envs: dict[str, str]       # KEY=value pairs the variant set


class RooflineSnapshot(TypedDict, total=False):
    """One ``state.roofline_snapshots[]`` entry mirrored verbatim (on-disk shape, so new fields flow through)."""
    snapshot_id: int
    ts: str
    achieved_tok_per_sec: float
    theoretical_peak_tok_per_sec: float       # ceiling, vendor peak (unreachable)
    within_roofline_pct: float                # achieved / peak * 100
    gap_to_roofline_pct: float
    compute_pct: float
    idle_pct: float
    comm_pct: float
    top_bottleneck: str                       # "MoE_unfused" etc
    top_kernel: dict[str, Any]                # {name, bound_type, efficiency_pct, gpu_pct}
    analysis_md_path: str
    kernel_roofline_path: str
    trace_input: str


class RooflineProgress(TypedDict, total=False):
    """Top-level ``roofline_progress`` section (renamed from ``roofline`` to avoid the markdown-renderer key clash).

    Carries reference lines (ceiling = vendor peak, target = ceiling ×
    ratio, default 0.70 per spec §2.1), the ``trajectory[]`` stepped line
    (baseline + KEEPs), and raw ``snapshots[]``. Edge cases (spec §5): no
    snapshot → ``ceiling_available = False``; no KEEP → trajectory is just
    the baseline point.
    """
    # Reference lines (only set when snapshots[] is non-empty)
    ceiling_tok_per_sec: float | None
    target_tok_per_sec: float | None
    ceiling_ratio_target: float                # default 0.70
    ceiling_available: bool
    snapshot_top_bottleneck: str               # tooltip on the ceiling line
    snapshot_within_roofline_pct: float
    snapshot_gap_to_roofline_pct: float

    # Trajectory
    trajectory: list[RooflineTrajectoryPoint]

    # Headline numbers surfaced so the dashboard's "current" callout needn't recompute them.
    baseline_tput: float
    current_best_tput: float
    cumulative_gain_pct: float
    current_best_pct_of_ceiling: float | None  # tput/ceiling*100, None when no ceiling
    current_best_pct_of_target: float | None   # tput/target*100, None when no ceiling

    # Audit / staleness
    roofline_failure_streak: int               # consecutive watermark roofline failures
    snapshots: list[RooflineSnapshot]


# Optimization stack — raw KEEP ledger passthrough
class OptimizationStackEntry(TypedDict, total=False):
    """One KEEP from ``state.optimization_stack[]`` exposed verbatim.

    Always-present fields: ``action`` / ``variant_name`` /
    ``candidate_extra_server_args`` / ``extra_envs`` / ``tput`` / ``ts`` /
    ``workspace``. GEMM-tuning entries add ``tuned_file`` /
    ``final_report_path`` / ``source``. Other optionals: ``gain_pct`` /
    ``kernel_id`` / ``fingerprint`` / ``provenance`` / ``task_id`` /
    ``validated`` (within the last full-stack rebench).
    """
    action: str
    variant_name: str
    candidate_extra_server_args: str
    extra_envs: dict[str, str]
    tput: float | None
    ts: str
    workspace: str | None
    validated: bool
    # gemm_tuning evidence
    tuned_file: str
    final_report_path: str
    source: str
    # generic optionals
    gain_pct: float | None
    kernel_id: str
    fingerprint: str
    provenance: str
    task_id: str


# Kernel Roofline — hot-kernel table for the dashboard, mirroring
# ``<session_dir>/reports/kernel_roofline.json``.
class KernelRooflineEntry(TypedDict, total=False):
    """One hot-kernel row (on-disk shape passed through verbatim; fields per spec §1)."""
    kernel_id: str                 # ``k001``..``k010``
    name: str                      # ``aiter::ck_moe_stage1`` etc
    source_file: str               # absolute path; "" when unknown
    kernel_category: str           # ``MoE`` / ``LayerNorm`` / ``unknown``
    bound_type: str                # ``memory-bound`` / ``compute-bound``
    arithmetic_intensity: float
    flops_per_byte: float
    efficiency_percent: float      # kernel-self efficiency 0..100
    gpu_pct: float                 # share of overall GPU time 0..100
    call_count: int
    duration_us: float
    reusable_native_kernel: bool   # True ⇒ GEAK can swap in a custom kernel


class KernelRoofline(TypedDict, total=False):
    """Top-level ``kernel_roofline`` section (loaded from the report; empty ``{}`` on missing/malformed)."""
    schema_version: int                    # tracelens output schema (currently 1)
    source: str                            # provenance label, e.g. ``tracelens_analysis``
    analysis_md_path: str                  # absolute path to the human-readable analysis
    kernel_candidates_path: str            # absolute path to kernel_candidates.json
    trace_input: str                       # absolute path to the trace dir
    trace_input_type: str                  # ``capture_dir`` / ``trace_file`` / ...
    kernels: list[KernelRooflineEntry]


# Kernel Optimization Summary (spec A1; PR #399 lishuoshuo): mirror of
# ``reports/kernel_optimization_summary.json``, passed through verbatim so
# new producer fields ride through; ``by_kernel[]`` rows stay loose (spec §A1.4).
class KernelOptimizationSummary(TypedDict, total=False):
    schema_version: int                    # producer schema (currently 1; int, unlike conc_sweep's str)
    session_id: str                        # global id ``{model}_{ts}_{short_uuid}``
    model_name: str
    cumulative_gain_validated_pct: float
    totals: dict[str, int]                 # {top_candidates, attempted, integrated, keep_pending, rejected, in_flight, unattempted}
    rejection_breakdown: dict[str, int]
    unattempted_reason_breakdown: dict[str, int]
    failure_reason_breakdown: dict[str, int]
    field_glossary: dict[str, str]         # {field_name: explanation} for tooltips
    top_takeaways: list[str]               # 2-4 deterministic (non-LLM) sentences
    by_kernel: list[dict[str, Any]]        # one row per top kernel, sorted gpu_pct desc; shape per §A1.4
    report_path: str                       # rel-to-session path to the mirrored source report


# Conc Sweep Summary (spec A2; PR #399 lishuoshuo): mirror of
# ``reports/conc_sweep_summary.json``, a baseline-vs-current_best curve
# across a CONC ladder. When ``status="skipped"`` the baseline/optimized/
# comparison/summary blocks are omitted — read ``status`` first.
class ConcSweepSummary(TypedDict, total=False):
    schema_version: str                    # producer schema (currently "1.0"; str, unlike kernel summary's int)
    status: str                            # succeeded / failed / skipped
    skip_reason: str                       # only when status="skipped"
    session_id: str
    isl: int
    osl: int
    tp: int
    concs_requested: list[int]
    baseline: dict[str, Any]               # {extra_server_args, extra_envs, points[]}
    optimized: dict[str, Any]              # {extra_server_args, extra_envs, points[]}
    comparison: list[dict[str, Any]]       # per-CONC paired rows (feeds the dual curve + speedup bars)
    summary: dict[str, Any]                # {successful_pairs, failed_pairs, best_conc, best_speedup, median_speedup, mean_speedup}
    workspace: str
    elapsed_sec: float
    total_budget_sec: int                  # None when budget gate disabled
    budget_exhausted: bool
    report_json_path: str
    report_csv_path: str                   # for the "download CSV" button
    roofline_ceiling: dict[str, Any]       # per-CONC theoretical peak + MBU% (§A2.9); may be absent on old products
    report_path: str                       # rel-to-session path to the mirrored source report


# ---------------------------------------------------------------------------
# Full-trace: unified token + decision timeline (FULL_TRACE_DESIGN §6)
# ---------------------------------------------------------------------------
class TokenBucket(TypedDict, total=False):
    """Aggregated token counters for one grouping (phase / component / total).

    ``total_cache`` (in the per-decision view) is the sum of cache-creation
    and cache-read tokens; the rollup view keeps them split. ``calls`` is
    the number of LLM calls folded into this bucket.
    """
    total_in: int
    total_out: int
    total_cache_creation: int
    total_cache_read: int
    calls: int


class DecisionTokens(TypedDict, total=False):
    by_component: dict[str, TokenBucket]   # component -> its token bucket for this decision
    total_in: int
    total_out: int
    total_cache: int                       # cache_creation + cache_read
    calls: int


class DecisionTraceEntry(TypedDict, total=False):
    phase: str                             # phase active at the decision (declared or ts-window backfill)
    tick: int | None                       # orchestrator tick (None when the producer didn't stamp one)
    ts: str                                # ISO ...Z of the decision
    decision: dict[str, Any]               # {component, change/event/verdict, outcome, gain_pct, task_id/dyn_id}
    tokens: DecisionTokens


class TokenRollup(TypedDict, total=False):
    by_phase: dict[str, TokenBucket]       # phase -> aggregate token bucket
    by_component: dict[str, TokenBucket]   # component -> aggregate token bucket
    session_total: TokenBucket             # whole-session token total


class DecisionTrace(TypedDict, total=False):
    """The joined token+decision timeline plus its rollups.

    ``decision_trace`` is one entry per decision (KEEP/REVERT journal row +
    dynamic_action dispatch event) with the LLM calls attributed to it.
    ``token_rollup`` summarises every call by phase / component / total.
    ``unattributed_tokens`` is the bucket of calls that matched no decision
    (kept so the per-decision sums + this reconcile to ``session_total``).
    """
    decision_trace: list[DecisionTraceEntry]
    token_rollup: TokenRollup
    unattributed_tokens: TokenBucket


# ---------------------------------------------------------------------------
# Token usage — promoted, discoverable top-level rollup of LLM token spend.
# ---------------------------------------------------------------------------
class TokenUsageBucket(TypedDict, total=False):
    """A token bucket plus two convenience totals for at-a-glance reading.

    Same counters as :class:`TokenBucket` (``total_cache`` appears in the
    per-action view where creation/read are pre-summed; the rollup view keeps
    them split). Adds:

    Attributes:
        total_in_out (int): ``total_in + total_out`` — the non-cache prompt +
            completion tokens (what most "how many tokens" questions mean).
        grand_total (int): ``total_in + total_out`` + all cache tokens
            (creation + read) — the all-in figure.
    """
    total_in: int
    total_out: int
    total_cache_creation: int
    total_cache_read: int
    total_cache: int
    calls: int
    total_in_out: int
    grand_total: int


class TokenUsageAttribution(TypedDict, total=False):
    """How much of the session token spend ties back to a decision.

    Attributes:
        attributed_to_decisions (TokenUsageBucket): Tokens whose call carried a
            ``task_id`` / ``dyn_id`` that joined to a KEEP/REVERT or
            dynamic_action decision (e.g. specialist subprocess turns).
        unattributed (TokenUsageBucket): Tokens from calls with no decision key
            (orchestration / kernel / critic / proposal_scorer turns — these
            are LLM-internal, not bound to a single tracked change).
        attributed_calls_pct (float): Percentage of calls that were attributed.
    """
    attributed_to_decisions: TokenUsageBucket
    unattributed: TokenUsageBucket
    attributed_calls_pct: float


class TokenUsageTimelineEntry(TypedDict, total=False):
    """One ``action_timeline`` row annotated with the tokens tied to it.

    Tokens join on ``task_id``; rows whose action carries no LLM token spend
    (most config-exploration actions) get ``tokens: null`` rather than a zero
    bucket, to make the (intentional) sparsity visible.

    Attributes:
        task_id (str | None): The action's task id (join key into the ledger).
        action (str): The action / change label (mirrors action_timeline).
        phase (str): Phase the action ran in.
        decision (str): KEEP / REVERT / ... outcome.
        ts (str): ISO timestamp of the action.
        tokens (TokenUsageBucket | None): Tokens attributed to this task_id, or
            None when no LLM call tied to it.
    """
    task_id: str | None
    action: str
    phase: str
    decision: str
    ts: str
    tokens: TokenUsageBucket | None


class TokenUsage(TypedDict, total=False):
    """Top-level, discoverable LLM-token-spend summary for the session.

    A promoted view over ``decision_trace.token_rollup`` (the full per-call
    ledger ``reports/trace/llm_calls.jsonl`` + ``ext/*.jsonl``) plus a
    timeline correlation. Purely derived — no new disk read — so it always
    reconciles with ``decision_trace``.

    Attributes:
        session_total (TokenUsageBucket): Whole-session total across every call.
        by_component (dict[str, TokenUsageBucket]): Per-agent breakdown
            (orchestration / kernel / critic / specialist / proposal_scorer / ...).
        by_phase (dict[str, TokenUsageBucket]): Per-phase breakdown
            (PRELUDE / FRAMEWORK_PR / EXPLORE / SWEEP / ...).
        attribution (TokenUsageAttribution): Decision-attributed vs unattributed.
        timeline (list[TokenUsageTimelineEntry]): ``action_timeline`` rows with
            their token spend joined on ``task_id``.
        source (str): The ledger files the totals derive from.
        correlation (str): How ``timeline`` joins to ``action_timeline``.
    """
    session_total: TokenUsageBucket
    by_component: dict[str, TokenUsageBucket]
    by_phase: dict[str, TokenUsageBucket]
    attribution: TokenUsageAttribution
    timeline: list[TokenUsageTimelineEntry]
    source: str
    correlation: str


# ---------------------------------------------------------------------------
# Langfuse push receipt — was the trace mirrored live to Langfuse?
# ---------------------------------------------------------------------------
class LangfuseConfig(TypedDict, total=False):
    """Redacted Langfuse connection config that was in effect this session.

    Credentials are never recorded verbatim: only the host URL (not a secret)
    and presence booleans for the public/secret keys.

    Attributes:
        enable_flag (bool): Whether ``HYPERLOOM_LANGFUSE_ENABLE`` was on.
        host (str | None): ``LANGFUSE_HOST`` URL, or None if unset.
        public_key_set (bool): Whether ``LANGFUSE_PUBLIC_KEY`` was present.
        secret_key_set (bool): Whether ``LANGFUSE_SECRET_KEY`` was present.
        sdk_available (bool): Whether the optional ``langfuse`` SDK importable.
    """
    enable_flag: bool
    host: str | None
    public_key_set: bool
    secret_key_set: bool
    sdk_available: bool


class LangfusePushCounts(TypedDict, total=False):
    """How many observations the live push actually emitted this session.

    Attributes:
        generations_sent (int): Generations successfully started.
        generations_paired (int): Of those, ones that had both a token row and
            conversation text (vs token-only / text-only).
        generations_text_only (int): Generations from a conversation row only.
        generations_token_only (int): Generations from a token row only
            (the typical out-of-process child case).
        scores_sent (int): Decision Scores created (span- + trace-level).
        spans_opened (int): Phase + agent spans created.
        ext_shards_read (int): Out-of-process ``ext/*.jsonl`` shards swept at flush.
        errors (int): Swallowed send failures (a Langfuse outage never breaks
            the optimization loop).
    """
    generations_sent: int
    generations_paired: int
    generations_text_only: int
    generations_token_only: int
    scores_sent: int
    spans_opened: int
    ext_shards_read: int
    errors: int


class LangfusePush(TypedDict, total=False):
    """Receipt of whether/where/how much the session was pushed to Langfuse.

    The local ``reports/trace/*.jsonl`` ledger is always written; this section
    records the *optional* second sink (live Langfuse push, default off). When
    disabled it still reports the config + ``disabled_reason`` so an operator
    can see why nothing was sent.

    Attributes:
        enabled (bool): Whether the live push was active (all gates passed).
        disabled_reason (str | None): Which gate tripped when not enabled
            (``disabled`` / ``no_credentials`` / ``sdk_missing`` /
            ``init_failed``); None when enabled.
        config (LangfuseConfig): Redacted connection config in effect.
        trace_id (str | None): Langfuse trace id (derived from the correlation
            id), or None when disabled.
        session_id (str | None): Langfuse ``session_id`` grouping value.
        correlated_on (str): Which id seeded the trace
            (``claw_session_id`` / ``internal_session_id``).
        counts (LangfusePushCounts): What was actually emitted.
        counts_final (bool): True once the session-end flush ran (counts then
            include out-of-process ext shards + decision scores); False when
            the breakdown was assembled before flush (in-process counts only).
        receipt_source (str): Where the collector read this from
            (``receipt_file`` / ``live_emitter`` / ``config_only``).
    """
    enabled: bool
    disabled_reason: str | None
    config: LangfuseConfig
    trace_id: str | None
    session_id: str | None
    correlated_on: str
    counts: LangfusePushCounts
    counts_final: bool
    receipt_source: str


class SessionBreakdown(TypedDict, total=False):
    """Top-level wire shape of ``session_breakdown.json``.

    The complete contract between the producer (``inference_optimizer``) and
    downstream consumers, aggregating every section of the breakdown. Several
    keys are intentional v1-reader compatibility aliases (``phase_timeline`` /
    ``action_timeline``, ``param_search`` / ``explore_search``) that carry the
    same data under both names.

    Attributes:
        schema_version (str): Schema version string (see ``SCHEMA_VERSION``).
        exported_at_utc (str): ISO UTC timestamp the file was exported.
        exporter_version (str): Version of the exporter that produced the file.
        session (SessionMeta): Session identity, timing, and host context.
        workload (Workload): Model/framework/serving configuration.
        baseline (Baseline): Pre-optimization reference performance.
        final (Final): Final validated optimization state.
        phase_timeline (list[PhaseEvent]): Flat per-action timeline (v1-reader compat).
        phase_segments (list[PhaseSegment]): Phase-boundary view (M2).
        action_timeline (list[PhaseEvent]): v2 canonical flat per-action timeline.
        capability_summary (CapabilitySummary): Per-capability roll-up.
        geak_invocations (list[Invocation]): GEAK backend invocations.
        oob_invocations (list[Invocation]): Out-of-box backend invocations.
        kernel_lifecycle (KernelLifecycle): Kernels grouped by lifecycle stage.
        param_search (ParamSearch): v1-reader compat alias for ``explore_search``.
        explore_search (ParamSearch): Merged explore-search ledger.
        sweep (Sweep): Concurrency/shape sweep results.
        critic_robustness (CriticRobustness): Critic reviews and robustness signals.
        telemetry (Telemetry): Telemetry artifacts and aggregated metrics.
        attribution (Attribution): Gain attribution across stack/source/phase.
        kb_provenance (KBProvenance): Cortex KB integration audit.
        specialist_runs (list[SpecialistRound]): Specialist sub-agent dispatch records.
        optimization_stack (list[OptimizationStackEntry]): Raw KEEP ledger passthrough.
        kernel_roofline (KernelRoofline): Hot-kernel table for the dashboard.
        roofline (list[dict[str, Any]]): Per-snapshot roofline comparison list for
            the markdown report's ``## Roofline`` section.
        roofline_progress (RooflineProgress): Optimization-progress curve for the dashboard.
        langfuse (LangfusePush): Live-Langfuse push receipt (enabled? / redacted
            config / counts); the local trace jsonl is always written regardless.
        warnings (list[str]): Collector warnings emitted while assembling the file.
        source_files (SourceFiles): Paths to the source artifacts used.
    """
    schema_version: str
    exported_at_utc: str
    exporter_version: str

    session: SessionMeta
    workload: Workload
    baseline: Baseline
    final: Final
    # flat per-action timeline (v1 compat); ``phase_segments`` is the boundary view.
    phase_timeline: list[PhaseEvent]
    phase_segments: list[PhaseSegment]
    # flat-list alias for older readers.
    action_timeline: list[PhaseEvent]
    capability_summary: CapabilitySummary
    geak_invocations: list[Invocation]
    oob_invocations: list[Invocation]
    kernel_lifecycle: KernelLifecycle
    # explore_search is the native merged ledger; param_search is a v1 alias.
    param_search: ParamSearch
    explore_search: ParamSearch
    sweep: Sweep
    critic_robustness: CriticRobustness
    telemetry: Telemetry
    attribution: Attribution
    kb_provenance: KBProvenance      # Cortex KB audit
    specialist_runs: list[SpecialistRound]
    # Raw KEEP ledger passthrough mirroring ``state.optimization_stack[]``
    # with full per-entry evidence; the stack-derived sections summarise it.
    optimization_stack: list["OptimizationStackEntry"]
    # Hot-kernel table (spec §1), mirror of ``reports/kernel_roofline.json``.
    kernel_roofline: KernelRoofline
    # Kernel-agent attempt outcome summary (spec §A1); empty → dashboard hides Block 1.
    kernel_optimization_summary: KernelOptimizationSummary
    # Post-optimization concurrency sweep (spec §A2); empty → dashboard hides Block 2.
    conc_sweep_summary: ConcSweepSummary
    # Per-snapshot roofline comparison list driving the markdown ``## Roofline`` section.
    roofline: list[dict[str, Any]]
    # Optimization-progress curve (spec §2); renamed from ``roofline`` to
    # coexist with the list-shaped ``roofline`` above.
    roofline_progress: RooflineProgress
    # Full-trace token + decision timeline (FULL_TRACE_DESIGN §6). New
    # optional section: a v1/v2 reader that doesn't know about it simply
    # ignores it. Empty dict on sessions that ran before the trace
    # subsystem landed (no reports/trace/ files).
    decision_trace: DecisionTrace
    # Promoted, discoverable token-spend rollup (full total + by component /
    # phase + decision attribution + action_timeline correlation). Derived
    # from decision_trace.token_rollup, so always reconciles with it. Additive
    # optional section; v1 readers ignore it.
    token_usage: TokenUsage
    # Live-Langfuse push receipt: enabled?/redacted config/how much was sent.
    # Additive optional section; the local trace jsonl is always written
    # regardless. ``enabled`` is False (with a ``disabled_reason``) on the
    # default path where HYPERLOOM_LANGFUSE_ENABLE is off.
    langfuse: LangfusePush

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
    "ConcSweepSummary",
    "CriticIteration",
    "CriticKBWritesSummary",
    "CriticRobustness",
    "DecisionTokens",
    "DecisionTrace",
    "DecisionTraceEntry",
    "DetectedKernel",
    "LangfuseConfig",
    "LangfusePush",
    "LangfusePushCounts",
    "Final",
    "GpuMonitorAggregate",
    "Invocation",
    "KBProvenance",
    "KBQueueStats",
    "LaneTimelineEntry",
    "KernelLifecycle",
    "KernelMetadata",
    "KernelOptimizationSummary",
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
    "PhaseBreakdown",
    "PhaseBreakdownExplore",
    "PhaseBreakdownKernel",
    "SourceBreakdown",
    "SourceFiles",
    "StackGainEntry",
    "Sweep",
    "SweepPoint",
    "Telemetry",
    "TokenBucket",
    "TokenRollup",
    "TokenUsage",
    "TokenUsageAttribution",
    "TokenUsageBucket",
    "TokenUsageTimelineEntry",
    "Workload",
    "WorkloadObjective",
]
