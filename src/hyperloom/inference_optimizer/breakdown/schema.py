# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema (TypedDict shape) for ``session_breakdown.json``.

The single contract between ``inference_optimizer`` and downstream
consumers. All fields are optional-by-convention (consumers treat missing
data as "not available", never fabricate); the wire shape is plain JSON;
``schema_version`` bumps only on breaking changes, not additive fields.
"""

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
SCHEMA_VERSION = SCHEMA_VERSION_V6


# Session metadata
class Recovery(TypedDict, total=False):
    """Crash / interruption / resume history for one optimization session.

    Records the recovery-relevant signals SharedState tracks so the breakdown
    captures when a run was interrupted and continued.

    Attributes:
        recovered (bool): True when the run crashed and/or was continued after
            an interruption (any of the signals below fired).
        crash_count (int): Monotonic total of Coordinator tick/agent crashes.
        crash_timestamps (list[str]): ISO UTC timestamps of recent crashes
            (bounded tail).
        degraded_mode (bool): Whether the run entered degraded operation.
        resume_pending_revalidation (bool): Accepted stack awaits post-resume
            revalidation (validated gain not yet re-trusted).
        last_tick_exception (dict[str, Any] | None): Compact summary of the last
            Coordinator tick exception (tick / stage / type / message), traceback
            omitted.
    """

    recovered: bool
    crash_count: int
    crash_timestamps: list[str]
    degraded_mode: bool
    resume_pending_revalidation: bool
    last_tick_exception: dict[str, Any] | None


class SessionMeta(TypedDict, total=False):
    """Identity, timing, and host context for one optimization session.

    Captures the metadata describing the run that produced the breakdown: its
    identifiers, lifecycle timestamps, stop reason, and runtime environment.

    Attributes:
        session_id (str): Hyperloom internal id (``manifest.session_id``).
        claw_session_id (str | None): SaFE / Claw session id (env ``CLAW_SESSION_ID``).
        sandbox_user_id (str | None): Sandbox user identifier, if any.
        created_at_utc (str): ISO UTC timestamp when the session was first
            created; unchanged by a resume.
        start_ts (str): ISO UTC timestamp the wall-clock budget is counted
            from. A resume re-anchors it on the new leg only when the previous
            one crashed or stopped for a recorded reason; after a clean stop it
            stays at the original start, because ``--max-hours`` keeps counting
            from there.
        ended_at_utc (str): ISO UTC timestamp when the session ended.
        stop_reason (str): Why the run stopped (``target_reached`` /
            ``time_exhausted`` / ``global_converged`` / ``max_ticks`` /
            ``baseline_failed`` / ...).
        max_minutes (int): Configured time budget in minutes.
        elapsed_minutes (float): Wall-clock minutes from ``start_ts`` to the
            end, or to now while still running, so it stays comparable with
            ``max_minutes``. When a resume kept ``start_ts`` this spans the
            gap between the legs as well, which is the span the budget is
            charged for too.
        host (str): Hostname the session executed on.
        code_revision (str): Source revision of the optimizer.
        pid (int): Process id of the optimizer.
        session_dir (str): Absolute path to the session working directory.
        user_data_path (str): ``USER_DATA_PATH`` root the run wrote under
            (``session_dir`` nests beneath it in per_model_ts layout); empty
            when unset. Lets a trace-based consumer locate the on-disk
            artifacts without re-deriving the path.
        tick_count (int): Number of orchestration ticks executed.
        image (str | None): Fully-qualified container image, or None if unset.
        recovery (Recovery): Crash / interruption / resume history for the run.
    """

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
    """Optimization goal the session was asked to pursue.

    Attributes:
        kind (str): Objective type (``gain_pct`` / ``tput`` / ``baseline`` /
            ``roofline_pct`` / ``time_only``).
        value (Any): Goal value — a float target, a string (e.g.
            ``target_baseline_dir``), or None when not applicable.
        objectives (list): Every target the run carried, present only when it
            carried more than one; ``kind`` / ``value`` name the first.
    """

    kind: str  # gain_pct / tput / baseline / roofline_pct / time_only
    value: Any  # float or str (target_baseline_dir) or None
    objectives: list[dict[str, Any]]


class Workload(TypedDict, total=False):
    """Model, framework name, and serving configuration under optimization.

    Describes the inference workload (model + framework name + parallelism + shape)
    plus the objective that defines success for the run.

    Attributes:
        framework_name (str): Serving framework name (``sglang`` / ``vllm`` / ``atom``).
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
    """Structural summary of the served model (architecture / scale / attention).

    Best-effort parse of the model's ``config.json``; every field is
    optional-by-convention so consumers must null-check. ``{}`` when the model
    is not a transformers checkpoint (diffusion etc.) or the session predates
    the field.

    Attributes:
        model_family (str): Base family with generation (``llama3`` / ``qwen3`` /
            ``deepseek_v3``).
        model_type (str): HuggingFace ``model_type`` (``qwen2`` / ``llama``).
        architectures (list[str]): Architecture class names
            (``["Qwen3ForCausalLM"]``).
        attention_type (str): Inferred attention variant (``MHA`` / ``GQA`` /
            ``MQA`` / ``MLA``).
        is_moe (bool): Whether the model is a Mixture-of-Experts model.
        num_hidden_layers (int): Number of transformer layers.
        hidden_size (int): Model hidden dimension.
        intermediate_size (int): FFN intermediate dimension.
        num_attention_heads (int): Number of attention heads.
        num_key_value_heads (int): Number of KV heads (GQA groups).
        head_dim (int): Per-head dimension.
        max_position_embeddings (int): Native context length.
        vocab_size (int): Vocabulary size.
        torch_dtype (str): Declared weight dtype (``bfloat16`` / ...).
        kv_cache_dtype (str): KV cache dtype when declared.
        quantization (str): Weight quant method (``fp8`` / ...); '' when
            unquantized.
        num_experts (int): Expert count (MoE only).
        num_experts_per_tok (int): Activated experts per token (MoE only).
    """

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
    """One recorded attempt to establish the baseline measurement.

    Attributes:
        ts (str): ISO UTC timestamp of the attempt.
        task_id (str): Orchestrator task id for the attempt.
        status (str): Outcome status of the attempt.
        decision (str): Decision taken (e.g. promoted / discarded).
        key_metric (float | None): Headline metric value, or None if absent.
        workspace (str | None): Benchmark workspace path, or None.
        error_class (str | None): Error classification on failure, or None.
        extras (dict[str, Any]): Attempt-specific fields from the writeback
            audit: ``fingerprint``, ``anchor_kept_tput``, and ``eval_probe``
            (why an accuracy of ~0 was a runaway generation, not wrong answers).
    """

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
    """Replayable record of how a benchmark variant was launched (server cmd + envs + config).

    ``extra_envs`` is allowlist-filtered to keep secrets out of the JSON.
    """

    framework_args: str  # e.g. "python -m sglang.launch_server --model ... --tp 8"
    framework_args_source: str
    # vocab: log_non_default_args / log_args_line / log_python_cmd / yaml_cmd / yaml_benchmark / unknown.
    extra_envs: dict[str, str]  # allowlisted env vars only (no secrets)
    config_path: str | None  # baseline_config.with_envs.yaml or variant config
    server_log_path: str | None  # for debug


class Baseline(TypedDict, total=False):
    """Pre-optimization reference performance for the workload.

    The baseline against which all gains are computed, including latency
    sub-metrics, attempt history, and the replayable launch invocation.

    Attributes:
        throughput_tok_s_per_gpu (float): Baseline throughput, whole-server
            total in ``throughput_unit``. The key name is a misnomer kept for
            wire compatibility; do not divide it by a GPU count.
        accuracy (float): Baseline accuracy score.
        ttft_mean_ms (float | None): Mean time-to-first-token (ms), or None.
        e2el_mean_ms (float | None): Mean end-to-end latency (ms), or None.
        ttft_e2el_source (str): Provenance of the latency metrics
            (``state_workspace`` / ``runs_baseline_disk`` / ``unavailable``).
        config_path (str | None): Path to the baseline config, or None.
        benchmark_report_path (str | None): Path to the benchmark report, or None.
        attempts_history (list[BaselineAttemptSummary]): Recorded baseline attempts.
        failure_streak (int): Consecutive baseline failures.
        total_failures (int): Combined backstop count of ALL baseline failures
            (any error_class); fast-fails when per-class streaks each stay below
            threshold but the total reaches it.
        invocation (BenchmarkInvocation): Replayable launch record.
        roofline_ceiling (dict[str, Any]): Standalone baseline-arm roofline
            ceiling backup (theoretical peak + mem/cmp + perfmodel breakdown);
            frontend ceiling fallback when the roofline step failed. {} when absent.
    """

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
    """Final validated optimization state — the SaFE contract core.

    Records the best validated result of the session: throughput, cumulative
    gain, the applied server-arg/env stack, and closing-phase bookkeeping.

    Attributes:
        throughput_tok_s_per_gpu (float | None): Final throughput, whole-server
            total in ``throughput_unit`` (same misnomer as
            ``BaselineSummary``), or None.
        cumulative_gain_pct_validated (float): Validated cumulative gain percent.
        validated_at_stack_len (int): Stack depth at which validation occurred.
        validated_ts (str): ISO UTC timestamp of the validation.
        stack_changed_after_validation (bool): Whether the stack changed post-validation.
        extra_server_args (str): Final extra server-arg CLI fragment.
        extra_envs (dict[str, Any]): Final extra env vars applied.
        action_path (list[str]): Ordered ``action:variant`` labels from the stack.
        ttft_mean_ms (float | None): Mean time-to-first-token (ms), or None.
        e2el_mean_ms (float | None): Mean end-to-end latency (ms), or None.
        ttft_e2el_source (str): Provenance of the latency metrics (``current_best`` /
            ``current_best_disk`` / ``stack_top_disk`` / ``unavailable``).
        invocation (BenchmarkInvocation): Replayable launch record for the final state.
        closing_phase_entered (bool): Whether the closing phase was entered.
        closing_started_unix (float): Unix time the closing phase started.
        closing_report_task_id (str): Task id of the closing report.
    """

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
    """One chronological event in the optimization timeline.

    A single action attempt (profile, backend trial, kernel opt, validation,
    etc.) with its outcome and optional contextual extras.

    Attributes:
        ts (str): ISO UTC timestamp of the event.
        action (str): Action kind (``baseline`` / ``profile`` / ``explore`` /
            ``roofline`` / ``sweep`` / ``kernel_opt`` / ``integrate``).
            ``backends`` / ``params`` / ``validate_stack`` appear only when
            reading archived sessions (see :class:`CapabilitySummary`).
        task_id (str): Orchestrator task id.
        kernel_id (str | None): Kernel id for kernel_agent-owned actions, else None.
        status (str): Outcome (``succeeded`` / ``failed``).
        decision (str): Decision label (``promoted`` / ``discarded`` /
            ``salvaged`` / ``no_promote`` / ``skipped`` / ``error`` / ``KEEP`` /
            ``PARTIAL`` / ``REVERT``).
        key_metric (float | None): Headline metric value, or None.
        key_metric_kind (str | None): Type/label of the key metric, or None.
        workspace (str | None): Benchmark workspace path, or None.
        error_class (str | None): Error classification on failure, or None.
        phase (str): The phase that ordered the dispatch, recorded at the
            attempt. Empty only for rows read from an archived session whose
            writer dropped it, which is why ``phase_segments`` still falls back
            to a timestamp window.
        macro_cycle (int): The macro cycle the dispatch was ordered in.
        extras (dict[str, Any]): Action-specific extra fields. For journal-sourced
            events this carries proposer attribution and a filter label:
            ``provenance`` (raw explore label), ``proposer`` (resolved component:
            ``specialist:<domain>`` / ``grid`` / ``orchestration``), ``scope``,
            ``fingerprint``, ``operation_kind`` (``backend`` / ``param`` / ``env``
            / ``kernel_opt`` / ``kernel_integrate`` / ``baseline`` / ...), and
            ``metrics`` (per-variant measurement detail).
    """

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
    phase: str  # recorded at the attempt; "" only for archived sessions
    macro_cycle: int
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
        forge (CapabilityEntry): Forge kernel-generation capability.
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
    forge: CapabilityEntry
    # primary explore row; backends/params/validate_stack are compat aliases.
    explore: CapabilityEntry
    backends: CapabilityEntry
    params: CapabilityEntry
    validate_stack: CapabilityEntry
    specialist: CapabilityEntry


# Kernel backend invocations
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
        backend (str): Winning backend (for example ``forge`` / ``geak``, best-of).
        total_attempts (int): Total optimization attempts.
        successful_attempts (int): Attempts that succeeded.
        best_micro_speedup (float | None): Best micro-benchmark speedup, or None.
        last_decision (str): Decision of the last attempt.
        best_artifact_path (str | None): Path to the best artifact, or None.
        attempts_summary (list[dict[str, Any]]): Per-attempt summary rows.
    """

    kernel_id: str
    backend: str  # forge / geak / backend name (best-of)
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
        basis (str): Throughput basis the gain was measured on ("hot" / "cold"
            / "" when the writer did not record one). A gain is meaningless
            without the baseline it was measured against.
        alignment_status (str): Whether the producer's baseline agreed with the
            orchestrator's ("" when not recorded).
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
    basis: str
    alignment_status: str


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


# Kernel journey — kernel-major unified lifecycle view
class KernelToolMetadata(TypedDict, total=False):
    """Provenance for an external kernel tool (tracelens / geak / forge / kernel_agent).

    Attributes:
        tool (str): Tool/backend name (``tracelens`` / ``geak`` / ``claude`` / ...).
        root_dir (str): Resolved tool root directory ("" when not resolvable).
        commit (str): Short git commit of ``root_dir`` ("" when not a repo).
        version (str): Tool-reported version string ("" when the tool did not
            surface one).
    """

    tool: str
    root_dir: str
    commit: str
    version: str


class DiscoveredHotKernel(TypedDict, total=False):
    """One hot kernel surfaced by a discovery run (projected onto the journey).

    The roofline numeric fields (``arithmetic_intensity`` / ``flops_per_byte``
    / ``efficiency_percent``) are backfilled at export from ``kernel_roofline``
    when discovery surfaced them empty (roofline enrichment runs after the
    discovery record is written).
    """

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
    """One hot-kernel discovery invocation (stage 1).

    Attributes:
        source (str): Discovery source (``tracelens`` / ``roofline`` / ...).
        status (str): Run status (``success`` / ``failed``).
        ts (str): ISO UTC timestamp of the run.
        duration_sec (float | None): Wall-clock seconds the discovery run took
            (source efficiency), or None.
        scan (dict[str, Any]): Scan inputs/outputs (``splitter_mode`` /
            ``trace_dir`` / ``candidates_path`` / ``trace_report_path``).
        hot_kernel_count (int): Number of hot kernels surfaced.
        hot_kernels (list[DiscoveredHotKernel]): The surfaced hot kernels.
        error (str | None): Failure text, or None on success.

    The discovery tool's authoritative version is not inlined here; it lives in
    the top-level ``versions`` map keyed by ``source``.
    """

    source: str
    status: str
    ts: str
    duration_sec: float | None
    scan: dict[str, Any]
    hot_kernel_count: int
    hot_kernels: list[DiscoveredHotKernel]
    error: str | None


class KernelDispatch(TypedDict, total=False):
    """The dispatch decision for one kernel (stage 2).

    Attributes:
        kernel_id (str): Kernel identifier.
        dispatched (bool): Whether any backend was dispatched.
        backends (list[str]): Backends dispatched to.
        skip_reason (str): Gate reason when not dispatched.
        orchestration_commit (str): Orchestrator commit at dispatch time.
        task_group (str | None): Task-group/correlation id, or None.
        ts (str): ISO UTC timestamp of the decision.
    """

    kernel_id: str
    dispatched: bool
    backends: list[str]
    skip_reason: str
    orchestration_commit: str
    task_group: str | None
    ts: str


class KernelBackendAttempt(TypedDict, total=False):
    """One backend attempt for one kernel (stage 3).

    Attributes:
        kernel_id (str): Kernel identifier.
        attempt_id (str): Attempt identifier (dedupe key across retries).
        run_id (str): Kernel-agent run id the attempt belonged to.
        backend (str): Backend that ran (``geak`` / ``claude`` / ``codex``).
        model (str | None): Model used by the backend, or None.
        ts (str): ISO UTC timestamp of the attempt.
        status (str): Attempt status.
        decision (str): KEEP / PARTIAL / REVERT / NEEDS_REVIEW / FAILED. The
            kernel-level verdict on the adopted attempt, the attempt's own
            otherwise.
        micro_speedup (float | None): Micro-benchmark speedup, or None.
        compile_passed (bool | None): Whether compilation passed, or None.
        correctness_passed (bool | None): Whether correctness passed, or None.
        correctness_source (str | None): What the correctness verdict was read
            from (``forge_rewrite_reference`` / ``report_scan`` /
            ``cli_override`` / ...), or None when nothing recorded one.
        best_artifact_path (str): The rewritten source the kernel was carried
            to integrate with. Written on the adopted attempt only; ``""``
            elsewhere. Distinct from ``optimized_files``, which is the
            attempt's own output (a stdout log for a real backend run).
        optimized_files (list[str]): Optimized artifact paths.
        error (str | None): Failure text, or None.
        error_class (str | None): Failure classification (pre-dispatch markers).
        duration_sec (float | None): Wall-clock seconds the attempt ran, or None.
        pre_dispatch_failure (bool): True for a synthetic marker recorded when a
            backend failed before running any real attempt (e.g. geak rejecting
            an empty/non-reusable kernel shape).

    The backend tool's authoritative version is not inlined here; it lives in
    the top-level ``versions`` map keyed by ``backend``.
    """

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
    """The end-to-end integrate outcome for one kernel (stage 4).

    Attributes:
        kernel_id (str): Kernel identifier.
        integrated (bool): Whether the optimization was integrated into the stack.
        e2e_gain_pct (float | None): Validated end-to-end gain percent (negative
            => regressed and reverted), or None.
        validated (bool | None): Whether the integrate was validated, or None.
        decision (str): KEEP / REVERT / REJECTED.
        patch_path (str | None): Adopted patch path, or None.
        target_file (str | None): File the patch applies to, or None.
        extra_server_args (str): Server-arg fragment introduced by the adoption.
        kernel_repo (str | None): Absolute deploy/apply root the integrated
            kernel landed in (the kernel analogue of the framework apply root),
            or None for an env-only adoption that named no repo.
        ts (str): ISO UTC timestamp of the integrate decision.
    """

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
    """One kernel's full lifecycle, joined across the four stages.

    Attributes:
        kernel_id (str): Kernel identifier.
        name (str): Kernel name (from discovery).
        gpu_pct (float | None): Share of total GPU time (from discovery), or None.
        bound_type (str): Bottleneck class (from discovery).
        source_file (str | None): Source file (from discovery), or None.
        micro_speedup (float | None): Best achieved micro-benchmark speedup
            across attempts (kernel-level), or None. Pair with
            ``e2e.e2e_gain_pct`` for the speedup-vs-e2e correlation.
        discovery (DiscoveredHotKernel): Discovery snapshot.
        dispatch (KernelDispatch): Dispatch decision.
        backend_attempts (list[KernelBackendAttempt]): Backend attempts.
        e2e (KernelE2E): End-to-end integrate outcome.
        roofline (dict[str, Any]): A copy of the matching ``kernel_roofline``
            entry (arithmetic intensity / efficiency / bound type / rocprof
            roofline), attached at export. Absent when no roofline ran.
        outcome (str): Coarse rollup (``adopted`` / ``reverted`` / ``attempted``
            / ``dispatched`` / ``skipped`` / ``discovered``).
    """

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
    """Kernel-major unified lifecycle view.

    Consolidates what was previously scattered across ``kernel_roofline``,
    ``geak_invocations`` / ``forge_invocations``, ``kernel_lifecycle`` and the
    attribution sections into a single per-kernel record threading discovery ->
    dispatch -> backend attempts -> end-to-end integrate. Composed at assembly
    from four recorder substreams; empty/absent on sessions that predate them.

    Attributes:
        discovery_runs (list[KernelDiscoveryRun]): Every discovery invocation,
            with tool provenance and the hot kernels each surfaced.
        kernels (list[KernelJourneyEntry]): Per-kernel lifecycle, sorted by
            ``gpu_pct`` descending.
    """

    discovery_runs: list[KernelDiscoveryRun]
    kernels: list[KernelJourneyEntry]


# Param search
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
        operation_kind (str): Filter label for the variant's change type
            (``backend`` / ``param`` / ``env``).
        provenance (str): Raw explore proposer label (``llm_direct`` /
            ``default_grid`` / ``specialist:<domain>``).
        proposer (str): Resolved proposer/component (``specialist:<domain>`` /
            ``grid`` / ``orchestration``).
        scope (str): Specialist dial (``domain`` / ``domains`` / ``freeform``).
    """

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
    """Ledger of one explore family's tested/accepted/rejected variants.

    Attributes:
        schema_version (int): Ledger schema version.
        tested_count (int): Total number of variants tested.
        accepted (list[ParamSearchEntry]): Variants that were accepted.
        rejected (list[ParamSearchEntry]): Variants that were rejected.
        top_by_gain (list[ParamSearchEntry]): Best variants ordered by gain.
        no_promote_streak (int): Consecutive evaluations without a promotion.
    """

    schema_version: int
    tested_count: int
    accepted: list[ParamSearchEntry]
    rejected: list[ParamSearchEntry]
    top_by_gain: list[ParamSearchEntry]
    no_promote_streak: int


class ParamSearch(TypedDict, total=False):
    """Merged explore-search results across the param and backend families.

    Attributes:
        params (ParamSearchLedger): Ledger for the param-tuning family.
        backends (ParamSearchLedger): Ledger for the backend-tuning family.
        synergy_attempted (list[str]): Synergy combinations that were attempted.
        discovered_flags (dict[str, Any]): Flags discovered during search.
    """

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
    # Same-harness adjudication, kept on the result because it is terminal
    # state: ``geak_pending`` is cleared when the verdict lands, and the final
    # report still has to say why a measured candidate was dropped.
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
    """One critic-agent review pass over a proposed change.

    Attributes:
        iteration_id (str): Stable session-unique identity for this persisted
            review pass, including resume-time reuse of ``iter``.
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
        phase (str): Coordinator phase captured with the critic request.
        macro_cycle (int): Coordinator macro cycle captured with the request.
        framework_reviews (list[dict[str, Any]]): Durable normalized V6
            Framework review rows.
    """

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


class CriticRobustness(TypedDict, total=False):
    """Critic-review iterations for the session.

    What the robustness agent raised is carried by the top-level
    :class:`V6Robustness` instead.

    Attributes:
        critic_iterations (list[CriticIteration]): Critic-agent review passes.
        kb_writes_summary (CriticKBWritesSummary): Tally of the critic
            iterations' verdicts (``total`` plus ``by_verdict``).
    """

    critic_iterations: list[CriticIteration]
    # KB writes proxied through the critic's ``commit-review`` protocol.
    kb_writes_summary: "CriticKBWritesSummary"


# Telemetry
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
    """One row of the lane occupancy summary (resource_lock capacity / live holders / expired leases)."""

    lane: str
    capacity: int
    live_holders: int
    lease_expired_count: int


class OrchestrationContext(TypedDict, total=False):
    """Health of the orchestration conversation's compaction loop.

    Attributes:
        seed_prompts (int): Full state pushes to the orchestration backend.
        delta_prompts (int): Thin delta pushes.
        compactions (int): ``orchestration_checkpoint`` events recorded.
        degenerate_compactions (int): Compactions skipped on an unusable summary.
        tick_count (int): Ticks executed, for the per-tick rates below.
        compactions_per_tick (float): ``compactions / tick_count``; near 1.0
            means the conversation is re-seeded every tick.
        delta_ratio (float): ``delta_prompts / (seed + delta)``; near 0 means
            the persistent conversation is buying nothing.
        context_tokens_at_compaction (dict[str, int]): ``min`` / ``median`` /
            ``max`` water level recorded on the compaction events. A ``min``
            above the soft budget means compacting cannot un-trip the trigger.
    """

    seed_prompts: int
    delta_prompts: int
    compactions: int
    degenerate_compactions: int
    tick_count: int
    compactions_per_tick: float
    delta_ratio: float
    context_tokens_at_compaction: dict[str, int]


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
        orchestration_context (OrchestrationContext): Compaction-loop health.
    """

    baseline_report_path: str | None
    profile_report_paths: list[str]
    torch_trace_paths: list[str]
    system_profile_paths: list[str]
    server_log_paths: list[str]
    gpu_monitor_aggregate: GpuMonitorAggregate
    # per-lane capacity / occupancy summary.
    lane_timeline: list[LaneTimelineEntry]
    # SEED/DELTA census + compaction rate for the orchestration conversation.
    orchestration_context: OrchestrationContext


# Attribution
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
    action: str  # backends / params / kernel_opt:<kid> / validate_stack
    variant_name: str | None
    cum_gain_before: float
    cum_gain_after: float
    delta_pct: float | None  # None when validate_stack re-baselined
    extra_server_args: str


class SourceBreakdown(TypedDict, total=False):
    """Validated total gain split by contributing source/family.

    Each ``*_pct_of_total`` field is the share of the validated total gain
    attributed to that source. The per-source values reconcile against
    ``validated_total_pct``.

    Attributes:
        forge_pct_of_total (float): Gain share from Forge kernel rewrites,
            credited only on Forge KEEP evidence. Emitted first by the
            collector; not yet declared on this TypedDict.
        geak_pct_of_total (float): Gain share from GEAK kernel rewrites.
        explore_pct_of_total (float): Gain share from the primary explore family.
        replay_warm_recipe_pct_of_total (float): Gain share from warm-recipe
            replay (recipe KB best_config replay); 0.0 when none was adopted.
        framework_pct_of_total (float): Gain share from FRAMEWORK bake-ins.
        gemm_tuning_pct_of_total (float): Gain share from the FP8 GEMM tuner
            (0.0 on non-FP8 workloads or when the tuner produced no KEEP).
        kernel_unattributed_pct_of_total (float): Kernel-lane gain that could
            not be tied to any backend KEEP (e.g. no Forge/GEAK KEEP evidence);
            kept unattributed rather than credited to a backend.
        unattributed_pct_of_total (float): Gain whose owning source could not
            be resolved from explicit ownership, recorded phase, or phase
            history.
        backends_pct_of_total (float): Gain share from backend exploration.
        params_pct_of_total (float): Gain share from param exploration.
        sweep_pct_of_total (float): Gain share attributed to the sweep.
        validated_total_pct (float): Total validated gain percent.
    """

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
    """Explore-phase gain split by specialist domain.

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
    """GEAK e2e gain, split by what was actually running when it was measured.

    ``by_contribution`` keys on ``config`` / ``kernel`` / ``joint``. A GEAK
    revalidation benchmarks server arguments, env and the authored overlay
    together against one baseline, so a ``joint`` row's gain cannot be divided
    between them; it is reported whole under ``joint`` rather than split by
    guess. ``by_kernel_id`` names the authored kernels that were loaded, and is
    empty for a config-only gain.
    """

    total_gain_pct: float
    by_contribution: dict[str, float]
    by_kernel_id: dict[str, float]


class PhaseBreakdown(TypedDict, total=False):
    """Per-phase gain attribution.

    Splits the validated total gain across the phase state machine, with each
    phase carrying its own per-sub-bucket breakdown.

    Attributes:
        prelude (PhaseBreakdownExplore): PRELUDE phase gain (always 0 by definition).
        framework (PhaseBreakdownFramework): FRAMEWORK_AGENT phase gain.
        explore (PhaseBreakdownExplore): Configuration-lever gain by domain.
        kernel_agent (PhaseBreakdownKernel): KERNEL_AGENT phase gain by
            ``kernel_id``. Unlike ``framework``, which the producer normalizes
            down from ``FRAMEWORK_AGENT``, this bucket keeps the phase name.
        gemm_tuning (PhaseBreakdownGemmTuning): KERNEL-entry GEMM-tuning gain,
            bucketed separately from source-level kernel rewrites.
        geak (PhaseBreakdownGeak): GEAK e2e gain, split by whether a config, a
            kernel, or both were running when it was measured.
        sweep (PhaseBreakdownExplore): SWEEP phase gain (usually 0; measurement).
        close (PhaseBreakdownExplore): CLOSE phase gain (usually 0).
        unattributed (PhaseBreakdownExplore): Gain whose phase could not be inferred.
    """

    prelude: PhaseBreakdownExplore  # always 0 by definition
    framework: PhaseBreakdownFramework
    explore: PhaseBreakdownExplore
    kernel_agent: PhaseBreakdownKernel
    gemm_tuning: PhaseBreakdownGemmTuning
    geak: PhaseBreakdownGeak
    sweep: PhaseBreakdownExplore  # usually 0 (sweep is measurement)
    close: PhaseBreakdownExplore  # usually 0
    unattributed: PhaseBreakdownExplore  # gain whose phase couldn't be inferred


class Attribution(TypedDict, total=False):
    """Gain attribution across stack entries, sources, and phases.

    Attributes:
        gain_per_stack_entry (list[StackGainEntry]): Per-KEEP incremental gains.
        method (str): How attribution was computed (``validated`` /
            ``single_source`` / ``reconstructed`` / ``missing``).
        source_breakdown (SourceBreakdown): Gain split by contributing source.
        phase_breakdown (PhaseBreakdown): Gain split per optimization phase.
        lever_breakdown (dict[str, float]): Gain split by ``lever_kind``:
            ``config`` (server args / envs), ``source_patch`` (a diff a
            specialist wrote), ``upstream_pr`` (a diff fetched from a PR),
            ``enablement`` (graded on runnability, not throughput) and
            ``kernel`` (a tuned or authored kernel). ``unattributed`` collects
            gain no stamp claimed. Two levers share the optimisation phase, so
            the lever -- not the phase that was live -- says which earned it.
        notes (list[str]): Human-readable caveats about the attribution.
    """

    gain_per_stack_entry: list[StackGainEntry]
    # validated / single_source / reconstructed / missing
    method: str
    source_breakdown: SourceBreakdown
    phase_breakdown: PhaseBreakdown
    lever_breakdown: dict[str, float]
    notes: list[str]  # human-readable caveats


# Phase segments — phase state machine
class PhaseSegment(TypedDict, total=False):
    """One contiguous segment of the phase state machine.

    Captures a phase the session occupied between two transitions, including
    the entry evidence and the events that fell within the segment window.

    Attributes:
        phase (str): Phase name (``PRELUDE`` / ``FRAMEWORK_AGENT`` /
            ``FRAMEWORK_AGENT`` / ``KERNEL_AGENT`` / ``SWEEP`` / ``CLOSE``).
        from_phase (str): Previous phase (empty for the first segment).
        entered_ts (str): ISO UTC timestamp of entry.
        entered_unix (float | None): Unix time of entry, or None.
        exit_ts (str): ISO UTC timestamp of the next transition; "" if current.
        exit_reason (str): Transition reason; "" for the current segment.
        evidence (dict[str, Any]): Entry evidence snapshot at transition time.
        actions (list[PhaseEvent]): Timeline events with ts in [entered, exit).
        elapsed_seconds (float | None): Segment duration in seconds, or None.
    """

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


# Roofline — optimization-progress curve for the dashboard:
# a stepped line from baseline through every KEEP against ceiling/target
# reference lines, all derived from ``state.json``.
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
    """Top-level ``roofline_progress`` section.

    Carries reference lines (ceiling = vendor peak, target = ceiling ×
    ratio, default 0.70), the ``trajectory[]`` stepped line
    (baseline + KEEPs), and raw ``snapshots[]``. Edge cases: no KEEP →
    trajectory is just the baseline point.

    Two ceiling domains exist. Decode workloads use the throughput fields
    below. Scriptable / diffusion (xDiT) workloads have no tok/s decode
    ceiling and instead report a latency-domain ceiling — the collector emits
    ``ceiling_kind`` (``"throughput"`` / ``"latency"`` / ``"none"``),
    ``latency_ceiling_ms`` (ideal per-image compute floor),
    ``achieved_latency_ms`` (measured e2e), ``latency_ceiling_available`` and
    ``current_best_pct_of_latency_ceiling`` (ideal/measured, so higher is
    nearer the floor); none of the five are declared below yet. Consumers
    should branch on ``ceiling_kind``: the latency branch leaves
    ``ceiling_available`` False while setting ``latency_ceiling_available``
    True, so ``ceiling_available = False`` does not mean "no ceiling".
    """

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
    """One KEEP from ``state.optimization_stack[]`` exposed verbatim.

    Always-present fields: ``action`` / ``variant_name`` /
    ``candidate_extra_server_args`` / ``extra_envs`` / ``tput`` / ``ts`` /
    ``workspace``. GEMM-tuning entries add ``engine`` / ``tuned_file`` /
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
    # filter label for the kind of optimization (backend / param / env on
    # explore KEEPs); specialist dial.
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
    """One adopted optimization's contribution to the session's reported gain.

    On the recorder path this is a ledger row over ``Optimizations.attempts``:
    it carries the chain arithmetic and enough identity to read, and defers
    everything descriptive to the attempt named by ``adopted_attempt_id``.
    The remaining fields are emitted only by the legacy state-rebuilt path,
    which has no attempts to point at.
    """

    id: str
    stack_index: int
    adopted_attempt_id: str | None
    adoption_id: str | None
    source: OptimizationSource
    source_method: OptimizationSourceMethod
    optimization_kind: str
    name: str
    backend: KernelOptimizationBackend | None
    # Gain against the session baseline. This is the only gain figure in the
    # report that may be summed across rows.
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
    # Gain the session really moved that no adopted step accounts for, most
    # often a KEEP that never reached the ledger. Held out of every entry so it
    # cannot be read as the next step's contribution.
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
    """One attempt at making the workload faster, adopted or not.

    This is the per-attempt layer of the optimization report: who proposed it,
    what it touched, what it measured, whether it was kept, why, and what it
    left behind. Rejected attempts appear here exactly like adopted ones.
    """

    attempt_id: str
    agent: AgentBucket
    # ``recorded`` when the producer stamped the owner, ``derived`` when it was
    # reconstructed for a session recorded before that field existed.
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
    # Measured against this attempt's own starting point, not the session
    # baseline. Never sum these; use ``OptimizationEntry.gain_pct`` instead.
    local_gain_pct: float | None
    throughput_before: float | None
    throughput_after: float | None
    adoption_id: str | None
    gates: list[OptimizationAttemptGate]
    backend_attempts: list[OptimizationBackendAttempt]
    # Each row carries ``occurrence``, its position among this operation's
    # readings of that metric name, oldest first, along with
    # ``occurrences_of_name`` for how many there are in total. Two readings
    # that agree are two readings: repeatability is the evidence, so it is
    # counted rather than inferred from the values.
    measurements: list[dict[str, Any]]
    # ``adoption_pinned`` when the adoption named the readings it was decided
    # on, ``latest_occurrence`` when the newest reading of each metric was used
    # for want of one, ``adoption_pinned_stale`` when the pinned readings were
    # overwritten by a later re-measure and no longer match the frozen decision
    # values. ``measurement_occurrences`` counts every reading the operation
    # kept, so a re-measured subject is visibly re-measured.
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
    # ``recorder`` when projected from author-time records, ``state`` when
    # rebuilt from business state for a session that predates the recorder.
    source_of_truth: str
    attempts: list[OptimizationAttempt]
    entries: list[OptimizationEntry]
    backend_attempts: list[OptimizationBackendAttempt]
    summary_by_agent: dict[AgentBucket, OptimizationAgentSummary]
    summary_by_source: dict[OptimizationSource, OptimizationSourceSummary]
    summary_by_kind: dict[str, OptimizationSourceSummary]
    validation: OptimizationValidation
    gemm_tuning_runs: list["GemmTuningRun"]


# GEMM tuning — fixed FP8 block-scale GEMM tuning stage that runs at KERNEL
# entry. Engine-tagged so GEAK ("geak") and a forge-backed tuner ("forge")
# share one home; gain is mirrored here while ``attribution`` stays authoritative.
class GemmTuningRun(TypedDict, total=False):
    """One GEMM-tuning run, keyed by the produced ``tuned_file`` CSV.

    A run is a config search over many GEMM shapes (not a single-kernel
    rewrite); its artifact is a dispatch CSV consumed via the
    ``AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`` env, not a kernel patch.

    Attributes:
        engine (str): Tuning engine provenance — ``"geak"`` today; a
            forge-backed tuner records ``"forge"``.
        status (str): Tool status (``ok`` / ``complete`` / ``skipped`` / ...).
        decision (str): ``KEEP`` / ``REVERT``.
        source (str): What triggered the run (``kernel_entry_auto`` / ...).
        ts (str): ISO UTC timestamp of the run record.
        precision (str): Workload precision (``fp8``).
        framework (str): Serving framework (``sglang``).
        gpu_type (str): Target GPU (``mi355x`` / ...).
        tp (int): Tensor-parallel degree (locked workload knob).
        conc (int): Concurrency (locked workload knob).
        isl (int): Input sequence length (locked workload knob).
        osl (int): Output sequence length (locked workload knob).
        libtype (str): Tuner library family (``ck`` / ``cktile`` / ``all``).
        baseline_tput (float | None): Pre-tuning throughput reference.
        best_speedup (float | None): Tuned / baseline throughput ratio.
        gain_pct (float | None): ``(best_speedup - 1) * 100``; mirrors
            the optimization-stack KEEP gain for this run.
        tuned_tput (float | None): ``baseline_tput * best_speedup``.
        tuned_file (str): Absolute path to the produced dispatch CSV.
        final_report_path (str): Absolute path to ``final_report.json``.
        workspace (str): Run workspace directory.
        adopted (bool): Whether this run's ``tuned_file`` landed as a KEEP
            in ``optimization_stack``.
        summary (dict[str, Any]): Tool-reported summary passthrough.
        shapes (list[dict[str, Any]]): Optional per-shape CSV rows
            (``M`` / ``N`` / ``K`` / ``libtype`` / ``kernelId`` / ``splitK`` /
            ``us`` / ``tflops``); empty until a producer emits them.
    """

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
    """Top-level GEMM-tuning section envelope.

    Attributes:
        runs (list[GemmTuningRun]): Every GEMM-tuning run this session
            recorded, newest-last, across engines.
        adopted_engine (str): Engine of the KEEP that won (``""`` if none).
        adopted_tuned_file (str): ``tuned_file`` of the adopted KEEP.
        total_gain_pct (float): Summed gain of adopted runs; mirrors
            ``attribution.phase_breakdown.gemm_tuning`` for convenience.
    """

    runs: list[GemmTuningRun]
    adopted_engine: str
    adopted_tuned_file: str
    total_gain_pct: float


# Collective — multi-rank communication campaigns run at KERNEL entry. Kept as
# its own section because the E2E gate, not the microbenchmark, decides
# adoption: a campaign that wins its micro run and loses the gate is absent
# from ``optimizations`` and would otherwise leave no trace in the breakdown.
class CollectiveAttempt(TypedDict, total=False):
    """One collective campaign, from candidate selection through the E2E gate.

    Attributes:
        collective_attempt_id (str): Stable campaign identity; deduplicates a
            resumed or salvaged run so it is not double-counted.
        experiment_id (str): KernelForge experiment identity for the campaign.
        kernel_id (str): Roofline ordinal of the targeted kernel (``k038``).
        kernel_name (str): Mangled device symbol that was optimized.
        collective_op (str): ``all_reduce`` / ``reduce_scatter`` / ``all_gather``.
        world_size (int | None): Rank count the campaign benchmarked against.
        engine (str): Producing engine (``forge_collective``).
        status (str): Tool status (``ok`` / ``skipped`` / ...).
        decision (str): Microbenchmark verdict (``KEEP`` / ``REVERT``).
        kept (bool): Whether the microbenchmark verdict was a KEEP.
        salvaged (bool): Whether the record was recovered from a partial run.
        requires_e2e_validation (bool): Whether a KEEP still owes an E2E gate.
        iterations (int | None): Campaign iterations forge-loop completed.
        kernel_speedup (float | None): Mean case speedup from the microbenchmark.
        gpu_pct (float | None): Share of GPU time the targeted collective held.
        duration_sec (float | None): Campaign wall time.
        ts (str): ISO UTC timestamp of the campaign record.
        source_file (str): Source the patch rewrote.
        kernel_repo (str): Repo root the patch applies to.
        workspace (str): Campaign workspace directory.
        patch_path (str): Absolute path to the produced patch.
        error_class (str): Machine-readable failure class (``""`` when none).
        error (str): Human-readable failure detail.
        integration_decision (str): E2E gate verdict — the one that decides
            adoption (``KEEP`` / ``REVERT`` / ``NEEDS_REVIEW``).
        integration_gain_pct (float | None): End-to-end throughput delta.
        integration_base_tput (float | None): Pre-patch throughput reference.
        integration_new_tput (float | None): Post-patch measured throughput.
        bandwidth (dict[str, Any]): Per-case measured bandwidth keyed by case
            name (``bytes`` / ``algbw_gbps`` / ``busbw_gbps``).
        artifact_files (list[str]): Repo-relative files the patch touched.
    """

    collective_attempt_id: str
    experiment_id: str
    kernel_id: str
    kernel_name: str
    collective_op: str
    world_size: int | None
    engine: str
    status: str
    decision: str
    kept: bool
    salvaged: bool
    requires_e2e_validation: bool
    iterations: int | None
    kernel_speedup: float | None
    gpu_pct: float | None
    duration_sec: float | None
    ts: str
    source_file: str
    kernel_repo: str
    workspace: str
    patch_path: str
    error_class: str
    error: str
    integration_id: str
    integration_decision: str
    patch_cleanup_status: str
    integration_result_status: str
    integration_revert_status: str
    integration_finalize_status: str
    integration_recovery_action: str
    integration_error_class: str
    integration_error: str
    integration_report_path: str
    integration_workspace: str
    integration_ts: str
    integration_gain_pct: float | None
    integration_base_tput: float | None
    integration_new_tput: float | None
    bandwidth: dict[str, Any]
    artifact_files: list[str]


class Collective(TypedDict, total=False):
    """Top-level collective-lane section envelope.

    Attributes:
        only_mode (bool): Mirrors ``HYPERLOOM_COLLECTIVE_ONLY`` — distinguishes
            a collective-only session from one where the lane merely ran.
        attempts (list[CollectiveAttempt]): One row per logical campaign,
            deduplicated by ``collective_attempt_id``, newest-last.
        last (CollectiveAttempt): The most recent campaign record, carrying the
            measurement evidence (``bandwidth``) the ledger rows omit.
    """

    only_mode: bool
    attempts: list[CollectiveAttempt]
    last: CollectiveAttempt


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


# Kernel Optimization Summary — mirror of
# ``reports/kernel_optimization_summary.json``, passed through verbatim;
# ``by_kernel[]`` rows stay loose.
class KernelOptimizationSummary(TypedDict, total=False):
    schema_version: int  # producer schema (currently 1; int, unlike conc_sweep's str)
    session_id: str  # global id ``{model}_{ts}_{short_uuid}``
    model_name: str
    cumulative_gain_validated_pct: float
    totals: dict[str, int]  # {top_candidates, attempted, integrated, keep_pending, rejected, in_flight, unattempted}
    rejection_breakdown: dict[str, int]
    unattempted_reason_breakdown: dict[str, int]
    failure_reason_breakdown: dict[str, int]
    dispatch_skip_reason: dict[
        str, Any
    ]  # {} or {reason, kernels_considered, message, ts} when a dispatch found no eligible kernels
    field_glossary: dict[str, str]  # {field_name: explanation} for tooltips
    top_takeaways: list[str]  # 2-4 deterministic (non-LLM) sentences
    by_kernel: list[dict[str, Any]]  # one row per top kernel, sorted gpu_pct desc
    report_path: str  # rel-to-session path to the mirrored source report


# Conc Sweep Summary — mirror of ``reports/conc_sweep_summary.json``, a
# baseline-vs-current_best curve across a CONC ladder. When ``status="skipped"``
# the baseline/optimized/comparison/summary blocks are omitted — read ``status`` first.
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
    # {extra_server_args, extra_envs, points[]}. A point carries the pair its
    # mode is plotted on: output_throughput + e2el_mean_ms synthetic,
    # total_token_throughput + intvty_p90 agentic.
    baseline: dict[str, Any]
    optimized: dict[str, Any]
    comparison: list[dict[str, Any]]  # per-CONC paired rows (feeds the dual curve + speedup bars)
    # {metric, successful_pairs, failed_pairs, best_conc, best_speedup,
    # median_speedup, mean_speedup}. ``metric`` names the axis the speedups
    # were taken on, which is the one that mode's chart is drawn on.
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


# ---------------------------------------------------------------------------
# Full-trace: unified token + decision timeline
# ---------------------------------------------------------------------------
class TokenBucket(TypedDict, total=False):
    """Aggregated token counters for one grouping (phase / component / total).

    ``total_cache`` (in the per-decision view) is the sum of cache-creation
    and cache-read tokens; the rollup view keeps them split.
    ``total_reasoning_out`` is hidden reasoning output, billed but absent from
    ``total_out`` (which counts the visible reply). ``calls`` is the number of
    LLM calls folded into this bucket.
    """

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
    # ``decision`` carries proposer attribution + a filter label:
    # {component (resolved proposer: specialist:<domain> / grid / orchestration),
    #  operation_kind (backend / param / env / kernel_opt / kernel_integrate / ...),
    #  change/event/verdict, outcome, gain_pct, task_id/dyn_id,
    #  kind, provenance, scope, fingerprint, metrics}
    decision: dict[str, Any]
    tokens: DecisionTokens


class TokenRollup(TypedDict, total=False):
    by_phase: dict[str, TokenBucket]  # phase -> aggregate token bucket
    by_component: dict[str, TokenBucket]  # component -> aggregate token bucket
    session_total: TokenBucket  # whole-session token total


class DecisionTrace(TypedDict, total=False):
    """The joined token+decision timeline plus its rollups.

    ``decision_trace`` is one entry per decision (KEEP/REVERT journal row +
    dynamic_action dispatch event) with the LLM calls attributed to it.
    ``token_rollup`` summarises every call by phase / component / total.
    ``unattributed_tokens`` + ``overhead_tokens`` are the buckets of calls that
    matched no decision (overhead = expected cross-decision spend; unattributed
    = a real gap), kept so per-decision sums + these reconcile to
    ``session_total``.
    """

    decision_trace: list[DecisionTraceEntry]
    token_rollup: TokenRollup
    unattributed_tokens: TokenBucket
    # Inherently cross-decision LLM spend (orchestration / critic / robustness
    # reactor turns) with no single owning decision — kept separate from
    # ``unattributed_tokens`` (a genuine attribution gap). Additive/optional.
    overhead_tokens: TokenBucket


# ---------------------------------------------------------------------------
# Token usage — promoted, discoverable top-level rollup of LLM token spend.
# ---------------------------------------------------------------------------
class TokenUsageBucket(TypedDict, total=False):
    """A token bucket plus two convenience totals for at-a-glance reading.

    Same counters as :class:`TokenBucket` (``total_cache`` appears in the
    per-action view where creation/read are pre-summed; the rollup view keeps
    them split). Adds:

    Attributes:
        total_in_out (int): ``total_in + total_out`` — the visible, non-cache
            prompt + completion tokens (what most "how many tokens" questions
            mean).
        grand_total (int): ``total_in + total_out`` + all cache tokens
            (creation + read) + ``total_reasoning_out`` — the all-in figure.
    """

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
    """How much of the session token spend ties back to a decision.

    Attributes:
        attributed_to_decisions (TokenUsageBucket): Tokens whose call carried a
            ``task_id`` / ``dyn_id`` that joined to a KEEP/REVERT or
            dynamic_action decision (e.g. specialist subprocess turns, scorer
            rounds keyed by their specialist task).
        overhead (TokenUsageBucket): Inherently cross-decision spend
            (orchestration / critic / robustness reactor turns) with no single
            owning decision — expected shared cost, not an attribution gap.
        unattributed (TokenUsageBucket): Tokens from calls that carried no
            decision key and are not recognised overhead — a real attribution
            gap to chase.
        attributed_calls_pct (float): Percentage of calls that were attributed.
        overhead_calls_pct (float): Percentage of calls classed as overhead.
    """

    attributed_to_decisions: TokenUsageBucket
    overhead: TokenUsageBucket
    unattributed: TokenUsageBucket
    attributed_calls_pct: float
    overhead_calls_pct: float


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
    ledger ``reports/trace/llm_calls.jsonl``) plus a timeline correlation.
    Purely derived — no new disk read — so it always reconciles with
    ``decision_trace``.

    Attributes:
        session_total (TokenUsageBucket): Whole-session total across every call.
        by_component (dict[str, TokenUsageBucket]): Per-agent breakdown
            (orchestration / kernel / critic / specialist / proposal_scorer / ...).
        by_phase (dict[str, TokenUsageBucket]): Per-phase breakdown
            (PRELUDE / FRAMEWORK_AGENT / KERNEL_AGENT / SWEEP / ...).
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
            (an unpaired token half flushed at session end).
        scores_sent (int): Decision Scores created (span- + trace-level).
        spans_opened (int): Phase + agent spans created.
        errors (int): Swallowed send failures (a Langfuse outage never breaks
            the optimization loop).
    """

    generations_sent: int
    generations_paired: int
    generations_text_only: int
    generations_token_only: int
    scores_sent: int
    spans_opened: int
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


class EnablementStackActionSummary(TypedDict, total=False):
    """One attempt-runtime stack action considered/applied.

    Attributes:
        kind: Stack-action kind (``runtime_candidate`` / ...).
        framework: Target framework.
        capability: Missing capability being repaired.
        acquisition_method: ``wheel`` / ``editable_ref`` / ...
        repo_url: Origin git URL (source acquisition), or "".
        ref: Pinned ref (source acquisition), or "".
        index_url: Pip index (wheel acquisition), or "".
        reason: Human-readable justification.
    """

    kind: str
    framework: str
    capability: str
    acquisition_method: str
    repo_url: str
    ref: str
    index_url: str
    reason: str


class EnablementAttemptRuntime(TypedDict, total=False):
    """One provisioned attempt runtime (promoted or discarded).

    Attributes:
        venv_root: Attempt venv root (``$SESSION_DIR/enablement/stacks/...``).
        bin_path: Attempt bin dir prepended to the materialized-YAML PATH.
        python_path: Attempt interpreter.
        installed_versions: Package -> version installed into the attempt venv.
        promoted: True when this runtime was KEPT (survives rearm).
    """

    venv_root: str
    bin_path: str
    python_path: str
    installed_versions: dict[str, str]
    promoted: bool


class TargetedBuildAttemptSummary(TypedDict, total=False):
    """One targeted-build attempt (AITER / sgl-kernel / vLLM-source).

    Attributes:
        component: ``aiter`` / ``sgl_kernel`` / ``vllm_source`` / ``framework_ext``.
        ref: Git ref / tag used for the build.
        gpu_arch: Explicit target arch (``gfx942`` / ``gfx950`` / ...).
        max_jobs: Parallelism cap passed to the compile.
        ok: Whether the build probe and install succeeded.
        failure_class: One of the ``FAILURE_CLASSES`` values, or ``"ok"``.
        failure_summary: Human-readable reason (agent decision input).
        installed_versions: torch/ref/sha/arch recorded after a successful build;
            includes ``source_pr_url`` when a discovered PR ref drove the build.
        build_probes: Post-build probe descriptors (e.g. ``"import aiter: ok"``).
        build_log_path: Path to the compile log inside the attempt dir.
        attempt_root: Attempt directory anchoring the build.
    """

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
    """Enablement subsystem observability section.

    Emitted when the lane did something or was explicitly turned off; ``all`` is
    the default, so an armed lane that was never needed stays hidden. A
    boot-origin round repaired by a plain source patch provisions no runtime and
    builds nothing, so admission and round lifecycle are reported independently
    of those artifacts.

    Attributes:
        mode: Admitted lane from ``--enablement``: launch / eval / all / off.
        engaged: True once a round was dispatched, attempted, or landed a patch.
        origin: Trigger origin: "boot" (cannot launch) or "eval" (accuracy).
        attempts: Number of authoring rounds dispatched this session.
        dispatched: True while an authoring round is in flight.
        succeeded: True once a round was KEPT (eval-origin additionally requires
            the revalidation baseline to promote at or above the floor).
        pending: True while a trigger is captured but unconsumed.
        validation_pending: True while an eval-origin KEEP awaits baseline
            revalidation.
        stall_streak: Consecutive no-progress rounds toward ``enablement_stalled``.
        inflight_task_id: Specialist task id of the in-flight round.
        last_specialist_task_id: Specialist task id of the most recent round.
        revalidation_task_id: TaskRegistry id of the tracked revalidation task.
        revalidation_generation: Revalidation window counter (idempotency).
        launch_log_excerpt: Tail of the captured boot failure text that triggered
            the round.
        trigger_evidence_excerpt: Tail of the captured eval-failure evidence.
        kept_patches: Session-relative paths of patches landed by enablement.
        framework_root: Source tree ``kept_patches`` were applied against.
        kept_stack_action: The stack action behind the KEPT attempt runtime.
        candidate_refs: Bridging candidate refs considered for rotation.
        setup_commands: Setup commands the specialist requested.
        localization_manifest: Files the localization pass identified.
        build_novelty: Novelty keys of the targeted builds requested.
        human_review_count: Number of logs parked for human review.
        active_runtime: The currently-promoted attempt runtime, or {} when none.
        attempt_runtimes: Retained attempt-runtime records (capped).
        failure_kind: Last classified enablement failure kind.
        build_attempts: Targeted-build attempt history (newest last).
        last_build_failure: ``{failure_class, failure_summary}`` from the most
            recent failed build attempt (framework-channel decision input).
        build_attempt_count: Total number of targeted-build rows attempted.
        trigger_kind: Eval trigger kind (eval_runtime_failure /
            accuracy_below_floor / accuracy_unavailable) when origin is "eval".
        observed_accuracy: Baseline accuracy observed at the eval trigger.
        accuracy_floor: Effective accuracy floor for the trigger + KEEP gate.
        observed_task: Eval task name observed at the trigger.
        observed_metric: Eval metric observed at the trigger.
        probe_config_path: Materialized config re-run to reproduce the contract.
        accepted_config_path: Base YAML from the KEEP'd candidate bench, used as
            the revalidation baseline config.
        accepted_config: Server args / envs that bench also launched with, which
            the YAML does not carry; the revalidation replays them on top.
        eval_contract_fingerprint: Fingerprint of the captured eval contract.
        setting_script: Session-relative path to the generated
            ``enablement_setting.sh`` artifact, when it was produced.
        kept_artifacts: Whole-file installs landed by enablement, as
            ``target`` / ``rel_target`` / ``kind`` per entry.
    """

    mode: str
    engaged: bool
    origin: str
    attempts: int
    dispatched: bool
    succeeded: bool
    pending: bool
    validation_pending: bool
    stall_streak: int
    inflight_task_id: str
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


# ---------------------------------------------------------------------------
# Session Breakdown v4 canonical author-time schema
# ---------------------------------------------------------------------------
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

# Which agent owns a unit of work. Recorded by the producer at author time;
# ``unattributed`` means the producer genuinely could not name an owner, never
# that the exporter failed to guess one.
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
    # Canonical owning agent, stamped by the producer at author time so the
    # exporter never has to infer ownership from phase timestamps.
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
    # Frozen at adoption time. Measurement ids are stable per subject, so a
    # later attempt on the same subject overwrites the referenced measurements;
    # these two carry the numbers this adoption was actually decided on.
    throughput_before: float | None
    throughput_after: float | None
    configuration: dict[str, Any]
    producer: str
    # Mirrors ``Operation.agent`` so an adoption can be bucketed without a join.
    agent: AgentBucket
    # False for pre-baseline enablement work: real, adopted, and deliberately
    # excluded from reported gain.
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


class V6ToolVersion(TypedDict, total=False):
    """One external tool's recorded provenance."""

    tool: str
    root_dir: str
    commit: str
    version: str


class V6MetadataVersions(TypedDict, total=False):
    """Version identifiers projected into V6 metadata.

    The breakdown's own schema version and the optimizer's revision are not
    here: they are the envelope's ``schema_version`` and
    ``metadata.session.code_revision``.
    """

    framework: str | None
    framework_version: str | None
    tools: dict[str, V6ToolVersion]


class V6MetadataRecovery(TypedDict, total=False):
    """Crash / interruption / resume history for the session."""

    recovered: bool
    crash_count: int
    crash_timestamps: list[str]
    degraded_mode: bool
    resume_pending_revalidation: bool
    last_tick_exception: dict[str, Any] | None


class V6MetadataSession(TypedDict, total=False):
    """Session identity and lifecycle fields exposed by V6 metadata.

    ``elapsed_minutes`` is how long this run leg ran and
    ``total_elapsed_minutes`` how long every leg of the session ran, so
    neither counts the gap between two legs the way the wall-clock budget
    does. Both are snapshotted while the run is going, which is why a
    re-export of a finished session reports the same figures rather than the
    span since its anchor.
    """

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
    image: str | None
    image_id: str | None
    max_minutes: int
    elapsed_minutes: float
    total_elapsed_minutes: float
    tick_count: int
    recovery: V6MetadataRecovery


class V6ModelArchitecture(TypedDict, total=False):
    """Structural model summary parsed from the model's own config."""

    model_class: str
    model_family: str
    model_type: str
    architectures: list[str]
    attention_type: str
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    hidden_size: int | None
    intermediate_size: int | None
    max_position_embeddings: int | None
    vocab_size: int | None
    torch_dtype: str
    kv_cache_dtype: str
    quantization: str
    is_moe: bool | None
    num_experts: int | None
    num_experts_per_tok: int | None
    has_shared_expert: bool | None
    num_shared_experts: int | None


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
    architecture: V6ModelArchitecture


class V6MetadataLangfuse(TypedDict, total=False):
    """Live-Langfuse trace entrypoint and push receipt."""

    enabled: bool
    disabled_reason: str | None
    trace_id: str | None
    session_id: str | None
    trace_url: str | None
    counts: dict[str, int]


class V6Metadata(TypedDict, total=False):
    """V6 task identity, configuration, versions, and trace entrypoint."""

    exported_at_utc: str
    versions: V6MetadataVersions
    session: V6MetadataSession
    task_config: V6TaskConfig
    langfuse: V6MetadataLangfuse
    warnings: list[str]


class V6OutcomeGainBucket(TypedDict, total=False):
    """Additive, session-baseline-relative gain for one V6 source bucket."""

    total_gain_pct: float | None
    keep_count: int
    #: Adoptions in the bucket with no measurable contribution. Zero of these
    #: is what makes ``total_gain_pct`` a complete account rather than a floor.
    unmeasured_keep_count: int


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
    """Reconciliation of the stack ledger's parts against the measured whole.

    Read off the ``stack`` timeline event, whose rows the orchestrator recorded
    as each adoption was accepted. ``attributed_gain_pct`` is the sum of the
    per-adoption contributions, all measured against the session baseline;
    ``chain_total_gain_pct`` is the last adoption's own reading against that
    same baseline. The two differ by ``unattributed_gain_pct``, which is
    throughput the chain gained between one adoption's measurement and the
    next one's. ``validated_total_gain_pct`` is the independent figure measured
    on the whole stack, and ``reconciliation_gap_pct`` is its distance from the
    chain -- the number worth alerting on, since the parts and the whole
    disagreeing means one of them is wrong.
    """

    attributed_gain_pct: float
    unattributed_gain_pct: float
    chain_total_gain_pct: float | None
    validated_total_gain_pct: float | None
    reconciliation_gap_pct: float | None
    attribution: V6OutcomeAttribution
    #: ``unmeasured`` and ``chain_breaks``; see :mod:`.recorder.stack_event`.
    guards: dict[str, int]
    #: One entry per finding the ledger's own figures support. Empty is the
    #: meaningful case: the ledger reconciles.
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
    """One ordered V6 business-stage event; CLOSE is intentionally excluded.

    Every field whose meaning is the same for all event types belongs here
    rather than being redeclared inside each ``ext``: one semantic stored per
    type is one semantic that drifts per type.

    ``id`` is the event id, ``{phase}:{macro_cycle}:{component}``. The rows the
    event was assembled from name the same value ``event_id``, and the
    asymmetry is deliberate -- here it is this object's own identity, on a row
    it is a reference to the event the row belongs to.
    """

    type: str
    kind: str
    status: str
    start_time: str
    end_time: str
    id: str
    ext: dict[str, Any]


class V6WarmStartMatched(TypedDict, total=False):
    """The Recipe the PRELUDE KB lookup selected.

    Present when the lookup found a record, whether or not it turned out to be
    executable -- a ``seed_only`` match is described here too. ``tier`` and
    ``confidence`` name the
    rung of the seven-tuple degradation ladder the hit came from, which is what
    separates an exact identity match from one that relaxed hardware or
    framework version to find anything at all. ``origin`` points back at the
    session that wrote the record, so a replay result can be compared against
    the run it came from.

    Attributes:
        match_type (str): ``exact`` when the tier is ``exact``, else ``degraded``.
        tier (str): Ladder rung — ``exact`` / ``same_arch_class`` /
            ``same_gpu_isa`` / ``compatible_framework_version``.
        confidence (float | None): Transfer confidence for that rung; gates the
            replay through ``--warm-replay-min-confidence``.
        source (str): Store the record came from (``kb-store`` / local).
        canonical_id (str): The seven-tuple actually matched.
        scope (dict[str, Any]): The matched record's own workload shape.
        optimized_throughput (float | None): Throughput the record validated.
        validated_gain_pct (float | None): Gain the record validated.
        expected_gain_pct (float | None): Gain the replay is expected to reproduce.
        replayable (bool | None): Whether the record may be replayed at all.
        replay_disabled_reason (str | None): Why it may not.
        replay_material_available (bool | None): Whether the three columns hold
            anything to replay — separates a hit on an empty record from a hit
            on a usable one.
        view_source (str | None): Which Recipe View the record was read through.
        origin (dict[str, Any]): ``{session_id, gain_pct}`` of the writing session.
        experience (dict[str, Any]): Counts of the lessons/pitfalls carried over.
    """

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
    """``timeline[type=warm_start].ext.reads`` — the KB reads T0 made.

    One row per read, recorded as the KB serves it, plus tallies over exactly
    those rows. Omitted when T0 made no read.

    Only T0's own reads are here. ``_kb_amend_recipe`` consults the same store
    through the same audit hook in the middle of the session; those reads are
    real but they are not the anchor's, and this block would misreport the
    lookup if it counted them.

    Attributes:
        count (int): How many reads T0 made.
        hits (int): How many returned a record.
        by_resolution (dict[str, int]): Reads counted by resolution outcome.
        by_method (dict[str, int]): Reads counted by the KB method that served
            them (``get_recipe`` / ``search`` / ``get_authoritative_recipe``),
            which is what separates the exact-identity probe from the
            degradation cascade.
        rows (list[dict[str, Any]]): The reads themselves, in service order, so
            a tally that looks wrong can be checked against what it counted.
    """

    count: int
    hits: int
    by_resolution: dict[str, int]
    by_method: dict[str, int]
    rows: list[dict[str, Any]]


class V6WarmStartExt(TypedDict, total=False):
    """``timeline[type=warm_start].ext`` — what was asked for, what came back.

    Whether the lookup ran and whether it found anything are separate facts:
    the event's ``status`` carries the first and ``match_status`` the second.
    Every finding is a ``succeeded`` lookup, because the KB was asked and it
    answered; only the machinery breaking is a ``failed`` one. A first-ever
    session for a workload matches the bare anchor row T0 stamped moments
    earlier and reports ``seed_only``, which is normal and not a fault.

    Attributes:
        requested (dict[str, Any]): ``{canonical_id, scope}`` this session asked
            for, recorded by T0 as it asks. The hardware dimension is
            topology-aware, so rebuilding the identity later could disagree
            with what the run actually queried.
        match_status (str): What was found — ``hit`` / ``seed_only`` / ``miss``.
            ``seed_only`` is a record that was found and cannot be executed,
            which is neither a usable match nor the absence of one.
        matched (V6WarmStartMatched | None): The matched record's facts; absent
            on a miss.
        reads (V6WarmStartReads | None): T0's own KB reads; absent when it made
            none.
        failure (dict[str, Any]): ``{error_class}``, present only when the
            lookup itself failed rather than completing with no match.
    """

    requested: dict[str, Any]
    match_status: str
    matched: V6WarmStartMatched | None
    reads: V6WarmStartReads | None
    failure: dict[str, Any]


class V6WarmReplayApplied(TypedDict, total=False):
    """What was running when a warm replay reproduced its gain.

    Recorded only on ``reproduced``. The columns are applied together and
    measured together, so one merged configuration is reported rather than a
    per-column split that would have to guess which column earned the gain. A
    replay that did not reproduce records its reason instead — its material has
    already been rolled back, so there is no running configuration to describe.

    Attributes:
        config (dict[str, Any]): The effective ``extra_server_args`` and
            ``extra_envs``, recipe and kernel columns already merged.
        patch (list[str]): Overlay refs that applied successfully. The
            lexicographic order of a ref is its replay order; the separate
            ``patch_timeline`` column is retired.
        kernel (dict[str, Any]): ``{status, total, kept, reverted, columns}``
            for the kernel column.
    """

    config: dict[str, Any]
    patch: list[str]
    kernel: dict[str, Any]


class V6WarmReplayExt(TypedDict, total=False):
    """``timeline[type=warm_replay].ext`` — did the record reproduce, and why not.

    Attributes:
        raw_status (str): The runtime status before it was collapsed onto the
            five published outcomes (it also spells ``rollback_failed``,
            ``enqueue_failed``, ``quality_failed``, ``accuracy_failed``,
            ``promotion_failed``, ``kernel_preparation_failed`` and
            ``reproduced_but_no_params``).
        result_type (str): Stable reason code; omitted on a clean reproduce.
        raw_reason (str | None): The unmapped reason, so normalization cannot
            silently drop detail.
        tier (str | None): Ladder rung of the replayed record.
        confidence (float | None): Transfer confidence of that rung.
        config_source (str | None): Identity that owned the replayed config.
        config_donor_tier (str | None): ``self`` when the identity owned it.
        donor (dict[str, Any]): Borrowed donor's identity, session and gain.
        before_tput (float | None): Baseline the replay was judged against.
        after_tput (float | None): Measured HOT-round throughput.
        gain_pct (float | None): Measured gain against ``before_tput``.
        expected_gain_pct (float | None): Gain the record claimed.
        keep_threshold_pct (float | None): Threshold this replay had to clear.
        historical_reproduce_bar_pct (float | None): ``expected_gain`` scaled by
            the minimum reproduce ratio.
        below_historical_reproduce (bool | None): Positive gain that still fell
            short of that bar — reproduced, but materially degraded.
        accuracy (dict[str, Any]): ``{eval_ran, baseline, replay, passed}``.
        applied (V6WarmReplayApplied): Present only on ``reproduced``.
        active_framework_root (str): Checkout promoted after a reproduce.
        rollback (dict[str, Any]): ``{ok, errors}`` when material was reverted.
        failure (dict[str, Any]): ``{error_class, error}``.
    """

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
    """``close.kb_write_back`` — did this session's Recipe land.

    Lives under ``close`` rather than as a timeline event of its own. The
    session publishes on its way out no matter what, so whether it did is a
    question always worth answering, and a timeline event that did not happen
    is simply absent — there would be nowhere to answer it. Absence of this
    key therefore means the publication was never attempted at all.

    The published Recipe body is deliberately not mirrored here: it is the KB
    Store's record, and duplicating three columns of overlay refs into every
    breakdown would grow the export without answering a question the identity
    and the throughput do not already answer.

    Attributes:
        status (str): The arc's terminal verdict. ``pending`` means an attempt
            was opened and never settled, i.e. the process died mid-publish —
            not that the store refused anything.
        result_type (str): Stable reason code, recorded by the publisher at
            whichever exit it took rather than recovered afterwards by matching
            substrings against ``raw_reason``.
        raw_reason (str | None): The publisher's own reason, verbatim.
        backend (str | None): ``kb-store`` / ``local`` / ``disabled``.
        canonical_id (str | None): Identity written to.
        session_id (str | None): Session id recorded on the KB side.
        scope (dict[str, Any]): Workload dimensions the Champion is keyed by.
        optimized_throughput (float | None): Throughput submitted, and the value
            compared against the incumbent Champion.
        validated_gain_pct (float | None): Session's cumulative validated gain.
        attempts (list[dict[str, Any]]): One row per attempt, each carrying its
            own ``source`` / ``status`` / ``result_type``. A list rather than a
            count because the publication is retried from two different seams
            and the row is what says which one settled it.
        queue (dict[str, Any]): Local write-queue depths, snapshotted when the
            attempt settled rather than counted at export.
        failure (dict[str, Any]): ``{error_class, error}``. ``error_class`` is
            the exception class name and is kept apart from ``raw_reason``,
            which the two used to share.
    """

    status: str
    result_type: str
    raw_reason: str | None
    backend: str | None
    canonical_id: str | None
    session_id: str | None
    scope: dict[str, Any]
    optimized_throughput: float | None
    validated_gain_pct: float | None
    attempts: list[dict[str, Any]]
    queue: dict[str, Any]
    failure: dict[str, Any]


class V6RooflineKernel(TypedDict, total=False):
    """One kernel in a roofline action's own per-kernel table.

    Recorded by reading the sidecar the analyzer wrote, at the moment the action
    that produced it settles. The table is not truncated by GPU share, because
    the kernel worth optimizing is often a cheap one running at low efficiency,
    which is exactly what a top-N-by-cost cut removes.

    The two analysis routes agree on identity and cost and diverge on
    provenance: the bypass route measures attainment against a real rocprof
    ceiling and sets ``roofline_measured`` / ``roofline_attainment_pct``, where
    TraceLens has only its analytical model. The absent field is itself the
    answer to "was this number measured".

    Attributes:
        kernel_id (str): Stable id the candidate list joins on.
        name (str): Kernel name as the trace reported it.
        kernel_category (str): TraceLens category bucket.
        source_file (str | None): Source file the kernel was attributed to.
        gpu_pct (float | None): Share of GPU time.
        duration_us (float | None): Total GPU time.
        call_count (int | None): Invocations in the traced window.
        bottleneck (str | None): The route's own bottleneck verdict.
        bound_type (str | None): ``memory`` / ``compute``, the question the
            roofline exists to answer.
        arithmetic_intensity (float | None): FLOPs per byte moved.
        flops_per_byte (float | None): As reported, before the fallback above.
        efficiency_percent (float | None): Achieved share of the kernel's own
            ceiling.
        compute_utilization_pct (float | None): Share of peak FLOPs.
        bandwidth_utilization_pct (float | None): Share of peak bandwidth.
        roofline_attainment_pct (float | None): Attainment against a measured
            ceiling; absent on the analytical route.
        roofline_name (str | None): The roofline model applied.
        roofline_source (str): ``analytical`` / ``measured`` / ``placeholder``.
        roofline_measured (bool): Whether the numbers came from rocprof.
        suggestion (str): The route's own optimization suggestion.
        recommended_actions (list[str]): Actions the route recommends.
        reusable_native_kernel (bool): Whether a native kernel could replace it.
        rocprof_roofline (dict[str, Any] | None): The rocprof sidecar's own
            measurements, when it produced any.
    """

    kernel_id: str
    name: str
    kernel_category: str
    source_file: str | None
    gpu_pct: float | None
    duration_us: float | None
    call_count: int | None
    bottleneck: str | None
    bound_type: str | None
    arithmetic_intensity: float | None
    flops_per_byte: float | None
    efficiency_percent: float | None
    compute_utilization_pct: float | None
    bandwidth_utilization_pct: float | None
    roofline_attainment_pct: float | None
    roofline_name: str | None
    roofline_source: str
    roofline_measured: bool
    suggestion: str
    recommended_actions: list[str]
    reusable_native_kernel: bool
    rocprof_roofline: dict[str, Any] | None


class V6RooflineKernelTable(TypedDict, total=False):
    """A roofline action's per-kernel table with the provenance of its source.

    Attributes:
        schema_version (str | None): Sidecar schema version, absent on the
            bypass route.
        source (str): ``tracelens_analysis`` / ``bypass``.
        trace_input (str): The trace the table was computed from.
        trace_input_type (str): Whether that input was a file or a directory.
        analysis_md_path (str): The analysis report the table accompanies.
        kernel_candidates_path (str): The candidate list built from it.
        path (str): The sidecar the table was read from.
        kernel_count (int): Kernels the table held, before the cap. Compare
            against ``len(kernels)`` to see how many the cap dropped.
        truncated (bool): Whether the cap dropped rows.
        kernels (list[V6RooflineKernel]): The rows, by descending GPU share.
    """

    schema_version: str | None
    source: str
    trace_input: str
    trace_input_type: str
    analysis_md_path: str
    kernel_candidates_path: str
    path: str
    kernel_count: int
    truncated: bool
    kernels: list[V6RooflineKernel]


class V6RooflineEventSnapshot(TypedDict, total=False):
    """A roofline run's own quantitative conclusion, recorded on its event.

    The event used to record only ``snapshot_id``, which made the conclusion
    reachable solely by joining against a capped session-state history that
    later runs evict entries from.

    Attributes:
        snapshot_id (int | None): The snapshot this run appended.
        ts (str): When it was taken.
        framework (str): The serving framework measured.
        macro_cycle (int | None): The cycle it was taken in.
        throughput_unit (str): Unit the throughput fields are in.
        achieved_tok_per_sec (float | None): Measured throughput.
        theoretical_peak_tok_per_sec (float | None): The binding ceiling.
        roofline_mem_ceiling_tok_per_sec (float | None): Memory-bound ceiling.
        roofline_cmp_ceiling_tok_per_sec (float | None): Compute-bound ceiling.
        roofline_bound_kind (str): Which of the two binds.
        e2e_mean_ms (float | None): Measured end-to-end latency.
        roofline_ideal_ms (float | None): Ideal latency the model implies.
        within_roofline_pct (float | None): Achieved share of the ceiling.
        within_roofline_pct_uncapped (float | None): The same, uncapped, so a
            measurement above the modelled ceiling stays visible.
        gap_to_roofline_pct (float | None): Headroom remaining.
        roofline_ceiling_exceeded (bool): Whether the measurement beat the
            model, which indicts the model rather than the measurement.
        ceiling_arm (str): ``baseline`` when the snapshot fixed the baseline
            ceiling.
        compute_pct (float | None): Share of wall time computing.
        idle_pct (float | None): Share idle.
        comm_pct (float | None): Share communicating.
        top_bottleneck (str): The analysis's headline bottleneck.
        top_kernel (dict[str, Any] | None): The costliest kernel, with its own
            share, efficiency and bound type.
        roofline_provenance (dict[str, Any] | None): The inputs the ceiling was
            computed from, so a suspicious ceiling can be audited.
        perfmodel_breakdown (dict[str, Any]): The per-operator analytical
            model, with ``op_count`` alongside a bounded ``ops``.
    """

    snapshot_id: int | None
    ts: str
    framework: str
    macro_cycle: int | None
    throughput_unit: str
    achieved_tok_per_sec: float | None
    theoretical_peak_tok_per_sec: float | None
    roofline_mem_ceiling_tok_per_sec: float | None
    roofline_cmp_ceiling_tok_per_sec: float | None
    roofline_bound_kind: str
    e2e_mean_ms: float | None
    roofline_ideal_ms: float | None
    within_roofline_pct: float | None
    within_roofline_pct_uncapped: float | None
    gap_to_roofline_pct: float | None
    roofline_ceiling_exceeded: bool
    ceiling_arm: str
    compute_pct: float | None
    idle_pct: float | None
    comm_pct: float | None
    top_bottleneck: str
    top_kernel: dict[str, Any] | None
    roofline_provenance: dict[str, Any] | None
    perfmodel_breakdown: dict[str, Any]


class V6RooflineTrajectoryPoint(TypedDict, total=False):
    """One measured step on the session's throughput curve.

    Attributes:
        ts (str): When the step was measured.
        tput (float): Throughput at that step.
        label (str): The variant name, falling back to the action.
        action (str): The action that produced the step.
        gain_pct (float): Gain over the baseline.
        flags (str): Extra server args the variant carried.
        extra_envs (dict[str, Any]): Extra environment the variant carried.
    """

    ts: str
    tput: float
    label: str
    action: str
    gain_pct: float
    flags: str
    extra_envs: dict[str, Any]


class V6RooflineProgress(TypedDict, total=False):
    """How far the session got against its roofline ceiling, snapshotted at close.

    This is the one roofline fact that is not a property of a single roofline
    run, which is why it sits under ``close`` rather than in a timeline event:
    the ceiling comes from the last analysis, the curve from every promotion
    between them, and the streak from the runs that failed.

    ``ceiling_kind`` discriminates two domains that cannot share fields.
    Token-decoding models are bounded in throughput and report the ``tok/sec``
    fields; scriptable/diffusion models decode no tokens and are bounded in
    latency, reporting the ``ms`` fields instead. A reader that ignores
    ``ceiling_kind`` will read the absent domain's nulls as a failed analysis.

    The snapshot history is deliberately absent: each roofline event carries
    its own snapshot in full, and a second copy here would be free to disagree.
    ``latest_snapshot_id`` is the join back to the event that set the ceiling.

    Attributes:
        ceiling_kind (str): ``throughput`` / ``latency`` / ``none``.
        ceiling_tok_per_sec (float | None): Theoretical peak from the last
            analysis. ``None``, never zero, when nothing was measured: zero
            would read as a ceiling of zero.
        target_tok_per_sec (float | None): ``ceiling_tok_per_sec`` scaled by
            ``ceiling_ratio_target``, a roofline ceiling being unreachable in
            practice.
        ceiling_ratio_target (float): The share of the ceiling aimed at.
        ceiling_available (bool): Whether a throughput ceiling was measured.
        latency_ceiling_ms (float | None): Ideal per-image compute floor.
        achieved_latency_ms (float | None): Measured end-to-end latency.
        latency_ceiling_available (bool): Whether a latency ceiling was
            measured.
        current_best_pct_of_latency_ceiling (float | None): Ideal over
            measured, so higher is nearer the floor.
        trajectory (list[V6RooflineTrajectoryPoint]): Baseline plus one point
            per promotion, oldest first.
        baseline_tput (float): Throughput the curve starts from.
        current_best_tput (float): The curve's own tail.
        cumulative_gain_pct (float): Validated cumulative gain.
        current_best_pct_of_ceiling (float | None): Curve tail over ceiling.
        current_best_pct_of_target (float | None): Curve tail over target.
        roofline_failure_streak (int): Consecutive failed roofline runs at
            close.
        latest_snapshot_id (int | None): The snapshot the ceiling came from.
        trajectory_incomplete (bool): True when the curve's tail disagrees with
            the session's own headline throughput, which is what a resume
            interrupted mid-promotion leaves behind.
        current_best_tput_declared (float): The session's headline throughput,
            present only when it disagrees with the curve.
    """

    ceiling_kind: Literal["throughput", "latency", "none"]
    ceiling_tok_per_sec: float | None
    target_tok_per_sec: float | None
    ceiling_ratio_target: float
    ceiling_available: bool
    latency_ceiling_ms: float | None
    achieved_latency_ms: float | None
    latency_ceiling_available: bool
    current_best_pct_of_latency_ceiling: float | None
    trajectory: list[V6RooflineTrajectoryPoint]
    baseline_tput: float
    current_best_tput: float
    cumulative_gain_pct: float
    current_best_pct_of_ceiling: float | None
    current_best_pct_of_target: float | None
    roofline_failure_streak: int
    latest_snapshot_id: int | None
    trajectory_incomplete: bool
    current_best_tput_declared: float


class V6BaselineProgress(TypedDict, total=False):
    """The session's final tally of baseline failures, snapshotted at close.

    Under ``close`` rather than in the baseline event for the same reason the
    roofline curve is: no single measurement can hold it. Each baseline event
    closes when its own measurement ends and the counters are advanced by the
    write-back that accounts for it afterwards, so an event records the count
    it was dispatched under (``request.failure_streak_before``) while the
    session total is only final here.

    Attributes:
        failure_streak (int): Consecutive baseline failures still standing at
            the close. Non-zero on a finished session means the last baseline
            it tried never landed.
        total_failures (int): Every baseline failure the session had.
        arg_error_streak (int): Consecutive failures rooted in a rejected
            server arg, counted apart because a bad flag is a configuration
            error the session can correct and a dying server is not.
    """

    failure_streak: int
    total_failures: int
    arg_error_streak: int


class V6ConcSweepPoint(TypedDict, total=False):
    """One rung of one arm's concurrency curve, as the sweep recorded it.

    The measurement half is the same flattening the sweep's own report writes,
    so the recorded curve and the written one cannot differ. The process half
    -- everything from ``stage`` down -- is what the report has no place for:
    a rung that produced no throughput number is otherwise indistinguishable
    from one the budget refused, one the server would not boot at, and one the
    benchmark simply failed.

    Attributes:
        arm (str): ``baseline`` or ``optimized``.
        conc (int): The rung's concurrency.
        status (str): The rung's outcome.
        output_throughput (float | None): Output tokens per second, the axis a
            synthetic sweep is graded on.
        request_throughput (float | None): Requests per second.
        total_token_throughput (float | None): Input plus output tokens per
            second, the axis an agentic sweep is graded on.
        input_throughput (float | None): Input tokens per second.
        intvty_p90 (float | None): The p90 interactivity an agentic run is
            plotted against; null on a synthetic run.
        tpot_p90_ms (float | None): p90 time per output token.
        ttft_mean_ms (float | None): Mean time to first token.
        e2el_mean_ms (float | None): Mean end-to-end latency.
        duration_seconds (float | None): The benchmark's own measured window,
            which is not the rung's wall clock.
        completed_requests (int | None): Requests the rung completed.
        error (str | None): What went wrong, when something did.
        error_class (str | None): The failure's class.
        killed_overtime (bool | None): Whether the rung was killed for
            exceeding its cap.
        estimated_output_throughput (float | None): The throughput estimated
            for a rung that did not finish.
        workspace (str): The rung's own workspace.
        report_path (str): The rung's benchmark report.
        stage (str): How the rung came to run -- ``boot``, ``reuse``,
            ``server_restart``, ``boot_attempt`` or ``budget_skip``.
        num_prompts (int | None): The load the rung carried, derived from its
            concurrency and never otherwise written down.
        start_time (str): When the rung started.
        end_time (str): When it ended.
        wall_duration_sec (float | None): How long it occupied, wall clock.
        granted_cap_sec (float | None): The cap it was granted, which under
            AgentX is raised above the declared per-rung timeout.
        budget_remaining_sec (float | None): What the budget had left when the
            rung was admitted.
    """

    arm: str
    conc: int
    status: str
    output_throughput: float | None
    request_throughput: float | None
    total_token_throughput: float | None
    input_throughput: float | None
    intvty_p90: float | None
    tpot_p90_ms: float | None
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    duration_seconds: float | None
    completed_requests: int | None
    error: str | None
    error_class: str | None
    killed_overtime: bool | None
    estimated_output_throughput: float | None
    workspace: str
    report_path: str
    stage: str
    num_prompts: int | None
    start_time: str
    end_time: str
    wall_duration_sec: float | None
    granted_cap_sec: float | None
    budget_remaining_sec: float | None


class V6ConcSweepArm(TypedDict, total=False):
    """One arm of a concurrency sweep: a whole ladder under one configuration.

    The two arms differ only by the server args and envs they add, so the
    curve is only readable alongside the decisions the ladder was run under.
    The reuse path boots one server at the most demanding rung and reuses it
    down; the restart path pays a server start per rung. They fail
    differently, and which one ran is not recoverable after the fact.

    Attributes:
        arm (str): ``baseline`` or ``optimized``.
        status (str): The arm's outcome. ``degraded`` when it measured some
            rungs and lost others, which is the ordinary partial result.
        start_time (str): When the arm started its ladder.
        end_time (str): When it finished.
        extra_server_args (str): The args defining this arm. ``""`` on the
            baseline arm, whose defining property is that it adds none -- a
            fact ``None`` would not carry.
        extra_envs (dict[str, str]): The environment defining this arm.
        strategy (str): ``single_server_reuse``, ``server_restart``, or
            ``refused`` for an arm the budget turned away before it built
            anything.
        strategy_reason (str | None): Why that strategy, when it was not the
            intended one -- an ineligible framework, or every boot failing.
        lifecycle (dict[str, Any]): ``{eligible, reason, port, framework}``
            from resolving whether a server can be kept across rungs.
        serving_lease_held (bool | None): Whether a Ray serving lease covered
            this arm's server for its whole lifetime.
        refused (dict[str, Any] | None): ``{reason, remaining_sec}`` for an arm
            the budget gate refused; ``None`` for an arm that ran.
        grid (list[dict[str, Any]]): The rungs planned, each
            ``{name, conc, num_prompts}``.
        boot (dict[str, Any]): The boot-retry-descend outcome --
            ``{succeeded, booted_conc, attempted_concs, failed_concs,
            attempts[]}``. ``failed_concs`` is the capacity finding a sweep
            produces for free: the concurrencies this configuration could not
            bring a server up at.
        points (list[V6ConcSweepPoint]): The curve, ascending by concurrency.
    """

    arm: str
    status: str
    start_time: str
    end_time: str
    extra_server_args: str
    extra_envs: dict[str, str]
    strategy: str
    strategy_reason: str | None
    lifecycle: dict[str, Any]
    serving_lease_held: bool | None
    refused: dict[str, Any] | None
    grid: list[dict[str, Any]]
    boot: dict[str, Any]
    points: list[V6ConcSweepPoint]


class V6ConcSweepPair(TypedDict, total=False):
    """The two arms joined at one concurrency.

    Attributes:
        conc (int): The concurrency both arms were measured at.
        baseline_throughput (float | None): The baseline arm's number on the
            graded axis.
        optimized_throughput (float | None): The optimized arm's number.
        speedup (float | None): Their ratio, ``None`` when either side is
            missing.
        delta_pct (float | None): The same gain as a percentage.
        baseline_status (str): The baseline point's status.
        optimized_status (str): The optimized point's status.
        error (str | None): Why the pair produced no speedup, named by the arm
            that did not succeed. Settled when the pair was computed rather
            than at export, because only the arm that broke can say why.
    """

    conc: int
    baseline_throughput: float | None
    optimized_throughput: float | None
    speedup: float | None
    delta_pct: float | None
    baseline_status: str
    optimized_status: str
    error: str | None


class V6ConcSweepExt(TypedDict, total=False):
    """The ``conc_sweep`` timeline event's ``ext``, recorded as the sweep runs.

    The sweep runs the CONC ladder twice -- once on the session's optimized
    server args, once on none -- and pairs the curves into a speedup per
    concurrency. What the event adds over the report on disk is the run rather
    than the result: the ladder and where it came from, the budget each arm
    was admitted under, the strategy each ladder ran with, and every rung
    including the ones that were refused or would not boot.

    Attributes:
        schema_version (str): The producer's report schema.
        request (dict[str, Any]): The dispatch -- ``{task_id, task_kind,
            reason, requested_concs, requested_variant_timeout_sec,
            requested_total_budget_sec}``.
        input_anchor (dict[str, Any]): The configuration the sweep was asked
            to compare -- ``{base_variant_id, base_action,
            input_throughput_tok_s_per_gpu, anchor_tput, baseline_tput,
            extra_server_args, extra_envs}``. A sweep dispatched two cycles
            later compares a different configuration under the same event
            type, which is why the event names it.
        workload (dict[str, Any]): ``{session_id, isl, osl, tp,
            benchmark_mode}``. ``benchmark_mode`` names the axis pair the
            points are drawn on, so a reader never infers it from whether
            ``intvty_p90`` happens to be null.
        plan (dict[str, Any]): ``{grid_source, concs_requested,
            concs_ordered, num_prompts_factor, variant_timeout_sec,
            arms_order}``. ``grid_source`` says whether the ladder was handed
            to the sweep or picked for the workload; ``arms_order`` matters to
            a reader of the budget, since the optimized arm runs first and a
            budget that runs out takes the baseline arm with it.
        budget (dict[str, Any]): ``{declared_total_sec, granted_total_sec,
            rung_cost_sec, raised, gate_active, deadline,
            session_soft_deadline_sec}``. Both totals are kept because they
            disagree: the sweep raises its own default when that default
            cannot fund one rung at the cap the grid runner will grant.
        environment (dict[str, Any]): ``{sweep_task_id, workspace,
            model_path, gpu_type, base_config_path}``. ``sweep_task_id`` is
            the sweep's own minted id, which names its workspace and is
            distinct from the dispatched task id.
        arms (dict[str, V6ConcSweepArm]): The two arms, keyed by name.
        comparison (list[V6ConcSweepPair]): The pair table, by concurrency.
        result (dict[str, Any]): ``{status, metric, best_conc, best_speedup,
            successful_pairs, failed_pairs, median_speedup, mean_speedup,
            skip_reason, was_skipped, budget_exhausted, declined}``.
            ``declined`` marks a sweep that refused before running anything,
            which is a different outcome from a ladder that ran and produced
            no usable pair.
        roofline_ceiling (dict[str, Any] | None): The per-concurrency
            theoretical peak and MBU the sweep computes once every point is
            in; ``None`` when model meta or the GPU spec was unavailable.
        runtime (dict[str, Any]): ``{workspace, elapsed_sec, duration_sec,
            budget_remaining_sec, budget_skip_reason}``.
        artifacts (dict[str, Any]): ``{report_json_path, report_csv_path}``.
        failure (dict[str, Any]): ``{stop_reason, message, error_class}``.
        superseded_sweeps (list[str]): Present only if one phase and cycle
            somehow held more than one sweep; the event reports the newest and
            names the rest rather than dropping them silently.
    """

    schema_version: str
    request: dict[str, Any]
    input_anchor: dict[str, Any]
    workload: dict[str, Any]
    plan: dict[str, Any]
    budget: dict[str, Any]
    environment: dict[str, Any]
    arms: dict[str, V6ConcSweepArm]
    comparison: list[V6ConcSweepPair]
    result: dict[str, Any]
    roofline_ceiling: dict[str, Any] | None
    runtime: dict[str, Any]
    artifacts: dict[str, Any]
    failure: dict[str, Any]
    superseded_sweeps: list[str]


class V6EnablementAttempt(TypedDict, total=False):
    """One authoring round of the enablement lane, keyed by its specialist.

    The dispatch and the settlement are recorded onto the same row from
    different ticks, so a round the session was killed between the two is on
    the timeline as a round that was dispatched and never ruled -- which the
    counters it used to be folded into could not express at all.

    The gap a round faced and the gap it revealed are separate fields.
    ``launch_log_excerpt`` is what the round was pointed at;
    ``next_launch_log_excerpt`` is what its patch uncovered underneath, which
    is the gap the *next* round will be pointed at. The projection kept one
    ``launch_log`` for the whole lane and every advance overwrote it, so the
    export published the newest gap as the reason the lane had opened.

    Attributes:
        attempt (int): The round's ordinal, 1-based.
        task_id (str): The authoring specialist's task id.
        failure_kind (str): The classified signature the round targeted. Read
            off the specialist params the dispatch built; the projection read a
            ``failure_kind`` state field that does not exist, so it was always
            empty.
        dispatched_at (str): ISO UTC timestamp the round was dispatched.
        launch_log_excerpt (str | None): The tail of the log it was dispatched
            against.
        candidate_refs (list[str]): The candidate refs the mandate carried.
        settled_at (str): ISO UTC timestamp the round was ruled.
        status (str): The integrate gate's verdict -- ``kept`` / ``advanced``
            / ``reverted``, in the gate's own words.
        advanced (bool): Whether the patch cleared its gap without making the
            combo runnable, which is progress on a serial enablement.
        reason (str): The gate's stated reason.
        landed (bool): Whether the lane reached terminal success on this round.
        validation_pending (bool): Whether an eval-origin KEEP opened a
            revalidation window instead of landing. A KEEP here is provisional.
        stall_streak_after (int): The no-progress streak after this round was
            scored, against a cap of five.
        patches_applied (list[str]): The patches this round contributed.
        artifacts_applied (list[dict[str, Any]]): ``{target, rel_target,
            kind}`` per whole-file artifact it installed.
        setup_commands_applied (list[str]): The env-setup commands it ran.
        patches_dropped_by_grounding (list[str]): Patches dropped for naming
            targets absent from the tree, which is what the next mandate is
            told so it stops writing diffs that cannot apply.
        patches_span_multiple_roots (bool): Whether its patches targeted more
            than one source tree.
        framework_root (str | None): The tree the patches apply against.
        accepted_config_path (str | None): The materialized config the KEEP'd
            bench ran.
        effective_config (dict[str, Any] | None): ``{extra_server_args,
            extra_envs}`` the bench launched with.
        stack_action (dict[str, Any] | None): The capability acquisition this
            round made -- ``{kind, framework, capability,
            acquisition_method, repo_url, ref, index_url, reason}``.
        runtime (dict[str, Any] | None): The framework runtime it provisioned.
        localization_manifest (dict[str, Any] | None): The localized closure
            it recorded, so the next round does not re-fetch it.
        next_launch_log_excerpt (str | None): The gap this round revealed.
    """

    attempt: int
    task_id: str
    failure_kind: str
    dispatched_at: str
    launch_log_excerpt: str | None
    candidate_refs: list[str]
    settled_at: str
    status: str
    advanced: bool
    reason: str
    landed: bool
    validation_pending: bool
    stall_streak_after: int
    patches_applied: list[str]
    artifacts_applied: list[dict[str, Any]]
    setup_commands_applied: list[str]
    patches_dropped_by_grounding: list[str]
    patches_span_multiple_roots: bool
    framework_root: str | None
    accepted_config_path: str | None
    effective_config: dict[str, Any] | None
    stack_action: dict[str, Any] | None
    runtime: dict[str, Any] | None
    localization_manifest: dict[str, Any] | None
    next_launch_log_excerpt: str | None


class V6EnablementBuild(TypedDict, total=False):
    """One targeted build the enablement lane ran, keyed by its task id.

    ``ok`` is absent, rather than false, on a row that only records an enqueue:
    a build with no verdict yet has not failed.

    Attributes:
        task_id (str): The build task id.
        recorded_at (str): ISO UTC timestamp the row was written.
        component (str): The component built -- ``aiter`` / ``vllm`` /
            ``sgl_kernel``.
        ref (str): The ref it built, as the installed versions report it.
        gpu_arch (str): The architecture it targeted.
        max_jobs (int): The parallelism it was given.
        installed_versions (dict[str, str]): What the build left installed.
        build_probes (list[str]): The import probes that verified it.
        build_log_path (str | None): Its log.
        attempt_root (str | None): Its build tree.
        novelty_key (str): The key the enqueue was idempotent on, which is what
            stops the lane rebuilding the same thing every round.
        ok (bool): Whether it succeeded.
        failure_class (str): The failure class, ``ok`` when it succeeded.
        failure_summary (str): The failure, clipped.
    """

    task_id: str
    recorded_at: str
    component: str
    ref: str
    gpu_arch: str
    max_jobs: int
    installed_versions: dict[str, str]
    build_probes: list[str]
    build_log_path: str | None
    attempt_root: str | None
    novelty_key: str
    ok: bool
    failure_class: str
    failure_summary: str


class V6EnablementRevalidation(TypedDict, total=False):
    """One eval-origin revalidation window, keyed by its generation.

    An eval-origin KEEP is provisional: the patch passed the gate's own bench,
    but accuracy is only official once a genuine baseline re-measures it under
    the frozen eval contract. The window is what holds the lane open until that
    happens, and its generation is what keeps a fresh enqueue from resolving to
    a spent task row.

    A window the run stopped is recorded with a ``reason`` and no
    ``error_class``: it measured nothing, so it is not a failed revalidation
    and the lane is not charged a stall for it.

    Attributes:
        generation (int): The window's generation, 1-based.
        opened_at (str): ISO UTC timestamp the window opened.
        task_id (str): The revalidation baseline's task id.
        config_path (str): The config it ran -- the accepted one from the
            KEEP'd bench, or the original probe config as a fallback.
        reason (str): Why the window opened, or why it did not promote.
        closed_at (str): ISO UTC timestamp the window closed.
        promoted (bool): Whether a genuine baseline promoted and cleared it.
        accuracy (float | None): The accuracy it measured.
        accuracy_floor (float | None): The floor it was graded against.
        error_class (str): The failure class, when it failed rather than
            measuring under the floor.
    """

    generation: int
    opened_at: str
    task_id: str
    config_path: str
    reason: str
    closed_at: str
    promoted: bool
    accuracy: float | None
    accuracy_floor: float | None
    error_class: str


class V6EnablementExt(TypedDict, total=False):
    """The ``enablement`` timeline event's ``ext``: one repair lane per session.

    Enablement repairs a (model, backend) combo that cannot be benched at all
    -- it will not boot, or it boots and fails its accuracy eval. The lane
    dispatches an authoring specialist, applies its patch, optionally compiles
    a component, benches the result, and either lands or rearms against the
    gap the patch revealed underneath. That is a sequence of dispatched actions
    with outcomes, which is what this event holds.

    The event spans the whole session rather than a phase, because the lane
    does. Its pump is phase-independent by design: a combo that cannot boot
    never leaves PRELUDE, and the round that repairs it is ruled in
    FRAMEWORK_AGENT. A phase-scoped event would hold the trigger in one half
    and the outcome in the other.

    Attributes:
        mode (str): The admitted ``--enablement`` mode -- ``off`` / ``launch``
            / ``eval`` / ``all``. A lane that never opened because the operator
            opted out is why a run with no baseline tried nothing.
        origin (str): ``boot`` or ``eval``, recorded when the lane opened.
            The projection derived this from two fields with different
            lifetimes -- ``origin`` is cleared on success, ``baseline_eval_kind``
            is not -- because nothing recorded it.
        engaged (bool): Always true. The lane is on the timeline because it was
            triggered; the projection needed this field to separate "did
            something" from "was armed and never needed", which an event that
            exists at all already answers.
        trigger (dict[str, Any] | None): What opened the lane -- ``{kind,
            recorded_at, evidence_excerpt, observed_accuracy, accuracy_floor,
            observed_task, observed_metric, eval_contract_fingerprint,
            probe_config_path}``. The first trigger wins: a later failure of
            the same gap belongs to the round that faced it, and letting the
            newest overwrite the oldest is how an eval-less re-baseline could
            downgrade a measured ``accuracy_below_floor`` to an empty
            ``accuracy_unavailable``.
        attempts (dict[str, Any]): ``{count, settled, landed, advanced,
            rows}`` over :class:`V6EnablementAttempt`. ``count`` counts rounds
            dispatched and ``settled`` counts rounds ruled, which differ by the
            one still in flight -- a distinction the projection's single
            ``attempts`` counter was read as making and did not.
        builds (dict[str, Any]): ``{count, failed, rows}`` over
            :class:`V6EnablementBuild`.
        revalidations (dict[str, Any]): ``{count, promoted, rows}`` over
            :class:`V6EnablementRevalidation`.
        human_review (dict[str, Any]): ``{count, rows}`` over the launch
            failures the lane could not classify well enough to dispatch a
            round for -- ``{digest, failure_kind, reason, signature,
            recorded_at}``. These are the rounds that never happened, and a
            lane that spent a session declining to dispatch reads, from
            counters alone, exactly like one that was never triggered.
        result (dict[str, Any] | None): The terminal -- ``{outcome, reason,
            stall_streak, kept_patches, kept_artifacts, setup_commands,
            accepted_config, accepted_config_path, setting_script,
            framework_root, active_runtime, attempt_runtimes}``. ``outcome`` is
            ``succeeded`` or ``stalled``; ``None`` on an event finalize
            recovered, which is a lane nothing judged rather than one that
            failed.
    """

    mode: str
    origin: str
    engaged: bool
    trigger: dict[str, Any] | None
    attempts: dict[str, Any]
    builds: dict[str, Any]
    revalidations: dict[str, Any]
    human_review: dict[str, Any]
    result: dict[str, Any] | None


class V6PhaseSegment(TypedDict, total=False):
    """One entry into a phase, with the exit that ended it.

    A row rather than a whole event because a phase re-entered inside one macro
    cycle cannot be given a second event id: the id's three segments are
    ``(phase, macro_cycle, component)`` and every one of them must be
    recomputable from persisted state, so there is nothing left to tell two
    entries apart. The event therefore covers all of a phase's time in a cycle
    and each entry is a segment on it.

    Both endpoints are recorded by the transition that produced them. The
    legacy ``phase_segments`` key derived them instead, by pairing
    ``phase_history`` rows off two at a time -- which cannot describe the
    segment a session ends in, because that one has no successor row to be
    closed by, and so published it with an empty exit and no duration.

    Attributes:
        sequence (int): The entering transition's position in ``phase_history``,
            which is this row's identity.
        from_phase (str): The phase the run came from; empty at the first entry.
        entered_at (str): The ISO timestamp of the entering transition.
        entered_unix (float | None): The matching Unix epoch, which the exit
            measures the segment against.
        entered_reason (str): The entering transition's reason.
        entered_evidence (dict[str, Any]): Its structured evidence.
        to_phase (str): The phase the run left for; absent while still here.
        exited_at (str): The ISO timestamp of the leaving transition; absent on
            the segment the session ended in.
        exited_unix (float | None): The matching Unix epoch.
        exit_reason (str): Why the run left, from ``PHASE_EXIT_REASONS``.
        exit_evidence (dict[str, Any]): The leaving transition's evidence.
        duration_sec (float | None): Measured from this segment's own two
            endpoints. ``None`` means the segment never closed, which is a
            different fact from zero.
    """

    sequence: int
    from_phase: str
    entered_at: str
    entered_unix: float | None
    entered_reason: str
    entered_evidence: dict[str, Any]
    to_phase: str
    exited_at: str
    exited_unix: float | None
    exit_reason: str
    exit_evidence: dict[str, Any]
    duration_sec: float | None


class V6PhaseAction(TypedDict, total=False):
    """One dispatched action, charged to the phase that ordered it.

    Deliberately thin. The per-dispatch detail belongs to the stage events --
    ``baseline.ext.actions[]`` carries the discarded cold-warmup rounds,
    ``framework_agent.ext.attempts[]`` carries the arm and provenance, and
    neither could be expressed by a flat row -- so restating any of it here
    would put one semantic in two places. The detail is reached by joining on
    :attr:`task_id`, which those rows already carry. For ``report``,
    ``recover``, ``session_breakdown`` and ``target_analysis``, which no stage
    event covers, this row is the only record and the join finds nothing.

    :attr:`phase` and :attr:`macro_cycle` are recorded at the dispatch, not at
    the settle. An action can outlive the phase that ordered it -- a baseline
    settling after a plateau exit -- so reading the phase when the result lands
    charges the wrong one. That is what the legacy ``phase_timeline`` did: the
    writer was handed both fields and dropped them, leaving export to attribute
    each action by testing its timestamp against the phase windows.

    Attributes:
        action (str): The action kind, as dispatched.
        task_id (str): The task id, which is this row's identity and the join
            key to whichever stage event holds the detail.
        phase (str): The phase that ordered the dispatch.
        macro_cycle (int): The macro cycle it was ordered in.
        tick (int): The coordinator tick, which orders dispatches within a phase
            more finely than a seconds-resolution timestamp can.
        dispatched_at (str): When the runner started it.
        dispatched_unix (float | None): The matching Unix epoch.
        status (str): The state it settled on; absent while in flight.
        decision (str): ``promoted`` or ``no_promote``, the dispatcher's own
            promotability verdict.
        error_class (str | None): The failure class, when it failed.
        workspace (str | None): The workspace it ran in.
        settled_at (str): When the verdict landed; absent while in flight.
        settled_unix (float | None): The matching Unix epoch.
        duration_sec (float | None): Measured from this row's two endpoints.
            ``None`` for an action that never settled, which is how a dispatch
            killed at shutdown stays legible as one that ran and was cut off.
    """

    action: str
    task_id: str
    phase: str
    macro_cycle: int
    tick: int
    dispatched_at: str
    dispatched_unix: float | None
    status: str
    decision: str
    error_class: str | None
    workspace: str | None
    settled_at: str
    settled_unix: float | None
    duration_sec: float | None


class V6PhaseMarker(TypedDict, total=False):
    """One non-transition ``phase_history`` marker, in the phase that raised it.

    Attributes:
        sequence (int): The marker's position in ``phase_history``.
        reason (str): What it marks.
        evidence (dict[str, Any]): Its structured payload.
        ts (str): When it was raised.
    """

    sequence: int
    reason: str
    evidence: dict[str, Any]
    ts: str


class V6PhaseExt(TypedDict, total=False):
    """The ``phase`` timeline event's ``ext``: the run's time in one phase.

    The one event that is about the run rather than about work. Every other
    event's id is scoped by a phase, and until this event existed the timeline
    held no record of the phases themselves -- a reader could see a baseline
    event tagged ``framework_agent`` and had no way to learn when that phase was
    entered, why the run left it, or how long it had.

    One event per ``(phase, macro_cycle)``, covering every entry into that phase
    in that cycle. See :class:`V6PhaseSegment` for why a re-entry is a row here
    rather than an event of its own.

    Attributes:
        phase (str): The phase this event is about.
        macro_cycle (int): The macro cycle it ran in.
        entered_at (str): When the phase was first entered in this cycle.
        exited_at (str): When it was last left; empty when it never was.
        exit_reason (str): The reason it was last left on.
        entries (int): How many times the phase was entered in this cycle.
        duration_sec (float | None): Summed over the entries, not measured from
            the first to the last: a phase re-entered inside one cycle did not
            own the time the run spent elsewhere in between, and charging it
            that time is how a budget guard comes to believe a phase overran.
        open (bool): True when an entry has no exit -- the phase the run was in
            when it stopped.
        segments (list[V6PhaseSegment]): One row per entry.
        actions (dict[str, Any]): ``{count, settled, kinds, rows}`` over
            :class:`V6PhaseAction`. ``count`` minus ``settled`` is the
            dispatches still in flight or killed mid-flight, which the flat
            projection could not express because it only ever held settled rows.
        markers (dict[str, Any]): ``{count, rows}`` over
            :class:`V6PhaseMarker`.
    """

    phase: str
    macro_cycle: int
    entered_at: str
    exited_at: str
    exit_reason: str
    entries: int
    duration_sec: float | None
    open: bool
    segments: list[V6PhaseSegment]
    actions: dict[str, Any]
    markers: dict[str, Any]


class V6StackAdoption(TypedDict, total=False):
    """One adoption onto the optimization stack, recorded as it was accepted.

    Attributes:
        stack_index (int): Position in the stack, which keys the row and indexes
            ``optimization_stack`` directly.
        action (str): The action kind that produced the winner.
        source (str): The attribution bucket, from
            :func:`.recorder.stack_event.source_for`. Recorded alongside the
            raw ``action`` so a row bucketed wrongly can still be re-bucketed.
        variant_name (str | None): The winning variant's name.
        lever_kind (str | None): Which lever the winner moved.
        operation_kind (str | None): The stable "what kind of optimization"
            label the stack can be sliced by.
        backend (str | None): For kernel adoptions, ``geak`` or ``forge``.
        source_phase (str | None): The phase that authored the winner, which is
            not always the phase live at writeback time.
        task_id (str | None): The dispatch that produced it, joining this row to
            its stage event.
        throughput_before (float | None): The anchor this adoption beat. The
            lift refuses a winner that does not beat exactly this number, and
            the ``current_best`` write immediately after overwrites it -- so
            this is the only moment it can be recorded.
        throughput_after (float | None): What the winner measured.
        baseline_tput (float | None): The session baseline.
        contribution_pct (float | None): ``(after - before) / baseline``. On the
            baseline rather than on ``before``, because contributions on one
            denominator sum to the chain total exactly and contributions on
            their own anchors do not.
        local_gain_pct (float | None): ``(after - before) / before`` -- the
            step's gain over the anchor it beat, which is what the promotion
            decision was actually made on.
        cumulative_gain_pct (float | None): ``(after - baseline) / baseline``.
        objective (str): The axis both readings were taken on.
        degrade_reason (str): Why the run's requested axis did not apply.
        attribution_eligible (bool | None): ``None`` when the producer never
            ruled, which is different from ruling it ineligible.
    """

    stack_index: int
    recorded_at: str
    ts: str
    action: str
    source: str
    variant_name: str | None
    lever_kind: str | None
    operation_kind: str | None
    scope: str | None
    backend: str | None
    source_phase: str | None
    task_id: str | None
    kernel_id: str | None
    fingerprint: str | None
    provenance: str | None
    gap_canonical_id: str | None
    objective: str
    degrade_reason: str
    throughput_before: float | None
    throughput_after: float | None
    baseline_tput: float | None
    contribution_pct: float | None
    local_gain_pct: float | None
    cumulative_gain_pct: float | None
    accuracy: float | None
    attribution_eligible: bool | None
    accepted_kernels: list[str]


class V6StackValidation(TypedDict, total=False):
    """One measurement of the whole stack's gain, keyed by the length it covers.

    The ledger's only independent check on itself. Without one of these the
    session total is the sum of the very steps it is meant to be checking.

    Attributes:
        stack_len (int): The stack length this figure validates. A later
            validation at one length supersedes the earlier one, which is the
            right reading: a re-measurement replaces the figure it revises.
        measurement_basis (str): ``e2e_rebench`` for a full-stack
            revalidation, ``e2e_decision_round`` for the round a variant was
            graded on.
    """

    stack_len: int
    ts: str
    baseline_tput: float | None
    validated_tput: float | None
    validated_gain_pct: float | None
    source: str
    measurement_basis: str


class V6StackExt(TypedDict, total=False):
    """The ``stack`` timeline event's ``ext``: what the session actually kept.

    One event per session, because there is one stack. Its adoptions arrive from
    PRELUDE warm replay, EXPLORE, FRAMEWORK_AGENT and KERNEL_AGENT and form a
    single ordered chain; scoping the event by phase would cut that chain at
    every phase boundary, which is exactly where its before / after pairs have
    to line up for the reconciliation to mean anything.

    Attributes:
        baseline_tput (float | None): The denominator every contribution shares.
        adoptions (dict[str, Any]): ``{count, by_source, rows}`` over
            :class:`V6StackAdoption`. Each ``by_source`` bucket carries its
            adoption count, summed contribution and unmeasured tally, and every
            bucket is present even when empty -- so a reader can tell "this
            subsystem earned nothing" from "this subsystem is not reported".
        validations (dict[str, Any]): ``{count, rows, settled, at_head}`` over
            :class:`V6StackValidation`. ``at_head`` is false when the last
            validation predates the final adoptions, meaning the session total
            is a claim about a shorter stack than the one that shipped.
        attributed_gain_pct (float): Summed contributions.
        chain_total_gain_pct (float | None): The last adoption's own reading
            against the baseline. One measurement, not a sum -- which is how the
            ledger gets to check itself.
        unattributed_gain_pct (float | None): ``chain_total - attributed``:
            throughput the chain gained that no adoption claims. An identity,
            equal to the sum of the chain breaks below.
        validated_total_gain_pct (float | None): The settled whole-stack figure.
        reconciliation_gap_pct (float | None): Its distance from the chain.
        guards (dict[str, int]): ``unmeasured`` and ``chain_breaks``. The eight
            guard counts the legacy ledger published were each a check on
            whether three v4 streams agreed; a fact recorded once at the moment
            it becomes true has nothing to disagree with, so only these two --
            which are about the measurements rather than the bookkeeping --
            have a referent here.
    """

    baseline_tput: float | None
    objective: str
    adoptions: dict[str, Any]
    validations: dict[str, Any]
    attributed_gain_pct: float
    chain_total_gain_pct: float | None
    unattributed_gain_pct: float | None
    validated_total_gain_pct: float | None
    reconciliation_gap_pct: float | None
    guards: dict[str, int]


class V6CriticReviewVariant(TypedDict, total=False):
    """One variant's ruling from a grid the Critic reviewed per variant.

    A rejected variant never reaches a bench, so there is no attempt row for
    its ruling to live on and this is the only record that it was judged. The
    map is kept per variant rather than collapsed because the collapse is
    deliberately lossy: a grid proceeds on its approved subset, and the
    proposal's summary verdict does not say which variants that was.

    Attributes:
        variant_name (str): The variant ruled on.
        verdict (str): What the Critic wrote for it.
        effective_verdict (str): What the loop acted on, which differs when a
            reject was held to a rule that declared a lesser verdict.
        held_to_rule (str): The rule the reject was held to, when one was.
        reason (str): The Critic's rationale for this variant.
        failure_reason_code (str): The rule it cited, when it cited one.
    """

    variant_name: str
    verdict: str
    effective_verdict: str
    held_to_rule: str
    reason: str
    failure_reason_code: str


class V6CriticReview(TypedDict, total=False):
    """The Critic's ruling on one proposal, recorded on the proposal itself.

    The review is a sub-structure of its subject rather than a stream of its
    own. On the bus a proposal and its verdict are two messages about one
    thing, and the Critic reviews proposals raised by every phase -- so a
    ruling attached to the proposal follows it wherever it was raised, needs no
    per-phase home, and leaves nothing to reconcile between a review list and a
    proposal list.

    Both verdicts are kept. A reject the loop held to a rule that only declared
    ``advise`` is two facts, and reporting either alone misreads the round: the
    authored verdict alone says a proposal was refused that in fact ran, the
    effective verdict alone says one was approved that the Critic refused.

    Attributes:
        verdict (str): The ruling the Critic wrote -- ``approve``, ``reject``,
            ``redirect``, ``advise`` or ``needs_review``.
        effective_verdict (str): The ruling the loop acted on.
        held_to_rule (str): The reason code a reject was held to, empty when
            nothing held it.
        reviewer (str): ``critic``, or ``critic_unavailable`` for a ruling the
            Critic could not ground -- which is not the same fact as one it
            examined and refused.
        iteration (int | None): Which review round, for a proposal re-submitted
            after ``needs_review``.
        reason (str): Why it ruled that way.
        confidence (float | None): How sure it was.
        failure_reason_code (str): The rule it cited, when it cited one.
        concerns (list[str]): The concerns it raised.
        reviewed_at (str): When the ruling was recorded.
        required_evidence (list[Any]): What it wants measured before approving.
        risks (list[Any]): The risks it named, each ``{severity, risk}``.
        notes (list[Any]): Remediation text.
        kb_evidence (list[Any]): The KB entries it cited.
        packet_evidence (list[Any]): The evidence packet rows it cited.
        advice_text (str): Its advice, on an ``advise`` verdict.
        alternative_action (str): What it would rather have run.
        variants (list[V6CriticReviewVariant]): Per-variant rulings, present
            only when the grid was reviewed by ``verdict_map``.
        outcome (dict[str, Any]): What the loop did with the ruling --
            ``{materialized, denied, reauthored, patch_verdict_key}``. Recorded
            onto the ruling because the consequence is usually what a reader is
            after and on its own does not say what it was the outcome of.
            ``patch_verdict_key`` names the subject the patch gate consults
            this ruling under, which is what connects a blocked
            ``integrate_patch`` to the review that blocked it.
    """

    verdict: str
    effective_verdict: str
    held_to_rule: str
    reviewer: str
    iteration: int | None
    reason: str
    confidence: float | None
    failure_reason_code: str
    concerns: list[str]
    reviewed_at: str
    required_evidence: list[Any]
    risks: list[Any]
    notes: list[Any]
    kb_evidence: list[Any]
    packet_evidence: list[Any]
    advice_text: str
    alternative_action: str
    variants: list[V6CriticReviewVariant]
    outcome: dict[str, Any]


class V6RobustnessIntent(TypedDict, total=False):
    """One intent the robustness agent raised on a turn.

    Attributes:
        type (str): The intent type the agent emitted.
        severity (str): How urgent the agent called it, when it said.
        topic (str): What the intent is about, when it said.
        payload (dict[str, Any]): The intent payload, verbatim.
    """

    type: str
    severity: str
    topic: str
    payload: dict[str, Any]


class V6RobustnessTurn(TypedDict, total=False):
    """The robustness agent's account of one turn.

    ``outcome`` distinguishes a turn that produced intents from one the agent
    could not complete, which is the distinction the section it replaces could
    not express: a mute agent and a silent session looked identical.

    Attributes:
        turn_idx (int): The agent turn this row describes.
        outcome (Literal): ``intents`` when the envelope validated,
            ``invalid_envelope`` when it failed validation, ``no_envelope``
            when the agent emitted none.
        ts (str): ISO UTC timestamp of the turn.
        tick_index (int): The optimizer tick the agent reported.
        intents (list[V6RobustnessIntent]): What the agent raised.
        parse_warnings (list[str]): Parse problems the agent reported.
        workdir (str): The turn's workdir, as a provenance pointer.
        detail (str): Why a turn without intents ended that way.
    """

    turn_idx: int
    outcome: Literal["intents", "invalid_envelope", "no_envelope"]
    ts: str
    tick_index: int
    intents: list[V6RobustnessIntent]
    parse_warnings: list[str]
    workdir: str
    detail: str


class V6Robustness(TypedDict, total=False):
    """What the robustness agent raised, outside the business timeline.

    The agent watches the session from the side, so its turns belong to no
    phase or macro cycle and it keeps a fixed top-level place instead. An
    empty ``turns`` is an answer, not a gap: the agent never completed a turn.

    Attributes:
        turns (list[V6RobustnessTurn]): One row per agent turn, by turn order.
    """

    turns: list[V6RobustnessTurn]


class V6Close(TypedDict, total=False):
    """V6 session finalization result exposed outside the business timeline.

    ``status`` is recorded by the CLOSE sequencer, not derived from ``steps``.
    ``running`` means no verdict was ever recorded, so the process died partway
    through its own close-out; ``degraded`` means the sequence finished with at
    least one step reporting a failure. The two used to be the same word, which
    made a healthy session indistinguishable from a damaged one.
    """

    status: Literal["running", "succeeded", "failed", "degraded"]
    start_time: str
    end_time: str
    close_sequence_done: bool
    steps: list[dict[str, Any]]
    # The close-out's verdict about robustness (escalation + stop reason). What
    # the agent itself raised is in the top-level ``robustness`` key.
    robustness: dict[str, Any]
    artifacts: dict[str, Any]
    # Absent when the session never attempted to publish its Recipe.
    kb_write_back: V6KBWriteBackExt
    # Absent when the close-out never got far enough to snapshot it.
    roofline_progress: V6RooflineProgress
    # Same rule: absent when the close-out never reached the snapshot.
    baseline_progress: V6BaselineProgress


# V6 KERNEL timeline event
#
# The ``kernel`` timeline event's ``ext`` shape, plus the recorder fragment
# shapes it is assembled from. Two rules govern the split between the two:
#
#   * A row that is updated incrementally owns an item section of its own, one
#     fragment per row keyed by its real id. It must not be recorded as a
#     nested list inside another fragment unless it carries one of
#     ``recorder._ENTITY_ID_FIELDS``, because the upsert list merge only merges
#     in place by those fields and otherwise appends.
#   * A value the assembler can compute is not recorded. Counts, ``delta_pct``
#     and the per-source counters are derived at assembly from the row
#     fragments, so an aggregate can never drift from its own detail.
class V6KernelEntry(TypedDict, total=False):
    """What the KERNEL entry hook decided and what it inherited.

    Attributes:
        route (str): The dispatch route the entry hook selected.
        route_reason (str): Why that route was selected.
        resumed (bool): Whether the phase was entered by a resume.
        code_revision (str | None): Orchestration commit the entry ran.
        stack_depth_in (int | None): Optimization-stack depth on entry.
        budget_remaining_sec (float | None): Session budget left on entry.
        roofline_snapshot_id (int | None): Analysis snapshot the entry read.
        roofline_snapshot_ts (str | None): When that snapshot was taken.
        roofline_baseline_gain_at_snapshot (float | None): Validated gain the
            snapshot was taken against.
        snapshot_staleness (float | None): Gain drift since the snapshot, which
            is what the re-profile trigger tests.
    """

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
    """The entry re-profile that decides whether cached analysis is stale.

    Attributes:
        ran (bool): Whether a re-profile was actually dispatched.
        task_kind (str | None): ``roofline`` (carries its own analysis and
            refreshes the cache) or ``profile`` (invalidates it, forcing the
            phase to request analysis of its own).
        trigger (str | None): ``gain`` / ``config_changed`` /
            ``workload_changed``.
        skipped_reason (str | None): Why it was skipped, when it was.
        idempotency_reason (str | None): The dispatch reason tag.
        snapshot_landed (bool): Whether a new snapshot actually landed.
        snapshot_id_before (int | None): Snapshot counter before the attempt.
        snapshot_id_after (int | None): Snapshot counter after the attempt.
    """

    ran: bool
    task_kind: str | None
    trigger: str | None
    skipped_reason: str | None
    idempotency_reason: str | None
    snapshot_landed: bool
    snapshot_id_before: int | None
    snapshot_id_after: int | None


class V6RowScope(TypedDict, total=False):
    """Recording-side bookkeeping every V6 row fragment carries.

    Shared by every event type, not just KERNEL: any type assembled from row
    fragments needs both fields for the same reasons.

    The spool directory is per session, not per event, so a row has to name its
    event twice over. The fragment key is prefixed with the event id, so two
    events holding a row with the same natural id cannot upsert into one file
    and lose the first event's row. And the payload repeats it as a field, so
    assembly can select the rows of the event it is closing. Omitting either
    half breaks something, but different things: the first loses data, the
    second mixes events together.

    ``ordinal`` orders rows that carry no timestamp of their own. The GEAK rows
    are replayed from ``kernel_journey.json``, which records no per-kernel
    time, so their only ordering information is their position in that file and
    it has to be captured explicitly. The fragment envelope cannot supply it:
    every upsert redraws the envelope ``seq`` and refreshes its ``ts``, so both
    name the last update rather than the first write, and ``seq`` is a
    per-recorder in-memory counter that restarts at 1 in a resumed process.

    Neither field reaches the timeline event -- ``event_id`` has done its job
    once the rows are filtered, and ``ordinal`` is superseded by the row's
    position once the array is sorted -- so assembly filters on the first,
    sorts on the second, and drops both.

    Attributes:
        event_id (str): The event this row belongs to,
            ``{phase}:{macro_cycle}:{component}``.
        ordinal (int): Position in the source that produced the row, for rows
            with no timestamp to sort by.
    """

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
    """Trace-analysis metadata shared by the roofline and kernel events.

    ``route`` records the routing policy and ``tool`` records the implementation
    that served it. Oversized sub-blocks are replaced by an omission marker
    rather than dropped, so a consumer can tell a bounded block from a missing
    one.

    Attributes:
        route (str): ``agent`` / ``bypass``.
        tool (str): ``tracelens`` / ``bypass``.
        tool_run_id (str): The analysis tool's own run id.
        steady_state (dict[str, Any]): Steady-state window selection.
        preflight (dict[str, Any]): Preflight checks.
        split (dict[str, Any]): Trace splitting.
        selection (dict[str, Any]): Hot-kernel selection.
        steps (Any): Per-step trace of the run.
        route_ext (dict[str, Any]): Route-specific extras.
        hot_kernels (dict[str, Any]): Bounded hot-kernel ranking summary.
        warnings (list[dict[str, Any]]): Trace-health warnings.
        artifacts (V6KernelAnalysisArtifacts): Paths the run produced.
    """

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
    """One analysis the KERNEL phase requested for itself.

    Normally absent: the entry re-profile dispatches a ``roofline`` task by
    default, which analyses the trace it just captured, so the phase's own
    request is served from cache. A present run therefore marks the case where
    the analysis behind a rewrite has no roofline event of its own.

    ``reusable_native_kernel_ids`` is recorded because it is the only legal
    source of a ``kernel_id``: the hot-kernel ranking includes vendor binaries
    that dispatch rejects as ``non_reusable_kernel``, so without the admitted
    set there is no way to check afterwards whether the kernel the phase went
    on to rewrite was ever a legitimate target.

    Attributes:
        run_id (str): Entry-stable identifier for this analysis.
        trigger (str | None): ``pre_run_optimization`` or ``llm_explicit``.
        requested_by (str | None): The role that requested it.
        request_msg_id (str | None): The bus request message id.
        ts (str): ISO UTC timestamp of the run.
        status (str): ``ok`` or ``failed``.
        cache_hit (bool): Whether a cached result served the request.
        trace_input (str | None): The trace the run analysed.
        top_k (int | None): The requested ranking depth.
        roofline_snapshot_id (int | None): Snapshot counter the run produced.
        roofline_baseline_gain_at_snapshot (float | None): Validated gain the
            produced snapshot was taken against.
        steady_state_trace (str | None): The steady-state trace selected.
        analysis_md_path (str | None): The human-readable analysis.
        reusable_native_kernel_ids (list[str]): Kernels dispatch would admit.
        trace_validate_ref (str | None): The trace validation that gated it.
    """

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
    """Fields every forge candidate row carries, whichever lane produced it.

    The two verdict fields are deliberately separate. ``micro_decision`` is the
    candidate layer's verdict on its own output; ``outcome`` is end-to-end
    adoption and is derived at assembly, never accepted from a caller. A row
    with no ``rebench_ref`` has nothing that re-measured it end to end, so it
    can only be ``needs_review`` however confident its own ``micro_decision``
    was.

    Attributes:
        lane (str): Which lane the row belongs to, discriminating the shared
            ``kernel_lane_run`` section.
        source_kind (str): The producer this candidate came from.
        run_id (str): The candidate's real attempt id.
        status (str): How the candidate's own run ended.
        started_at (str | None): ISO timestamp the candidate started.
        ended_at (str | None): ISO timestamp the candidate ended.
        duration_sec (float | None): Wall-clock seconds the candidate took.
        micro_decision (str | None): The candidate layer's own verdict.
        rebench_ref (str | None): The rebench attempt id that re-measured it.
        outcome (str): End-to-end adoption verdict, derived at assembly.
        failure_reason (str | None): Normalized failure reason.
    """

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
    """One forge source-level kernel rewrite.

    ``adopted_backend`` and ``run_id`` are stated rather than derived: the
    superseded projection had to guess the backend from a speedup plus an
    artifact path, and to synthesize an identifier from
    ``kernel_id:backend:sequence`` whenever the real attempt id had been lost.

    Attributes:
        kernel_id (str): The kernel the rewrite targeted.
        kernel_name (str | None): Human-readable kernel name.
        dispatched (bool): Whether a backend was actually dispatched.
        backends_tried (list[str]): The backends attempted.
        adopted_backend (str | None): The backend whose output was taken.
        skip_reason (str | None): Why dispatch was skipped, when it was.
        task_group (str | None): The dispatch task group.
        speedup (float | None): Micro-benchmark speedup.
        baseline_us (float | None): Micro-benchmark baseline microseconds.
        candidate_us (float | None): Micro-benchmark candidate microseconds.
        compile_status (str | None): Compilation outcome.
        correctness (bool | None): Correctness verdict.
        artifact_path (str | None): The produced artifact.
        trace_analyze_ref (str | None): The analysis that nominated the kernel.
        e2e (V6KernelRewriteE2E | None): Integration sub-result.
    """

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
    """One forge-fusion run.

    Attributes:
        pattern (str | None): The fusion pattern attempted.
        target_module (str | None): The module the fusion targeted.
        applied (bool): Whether the fusion was applied.
        gain_pct (float | None): The gain the run claimed.
        patch_path (str | None): The produced patch.
    """

    pattern: str | None
    target_module: str | None
    applied: bool
    gain_pct: float | None
    patch_path: str | None


class V6KernelGemmTuningRun(V6KernelLaneRun, total=False):
    """One GEMM shape-table tuning run.

    Attributes:
        shapes_total (int | None): Shapes the run considered.
        shapes_tuned (int | None): Shapes the run tuned.
        config_path (str | None): The produced shape-table.
        gain_pct (float | None): The gain the run claimed.
        tuner (str | None): The tuner that ran.
    """

    shapes_total: int | None
    shapes_tuned: int | None
    config_path: str | None
    gain_pct: float | None
    tuner: str | None


class V6KernelCollectiveRun(V6KernelLaneRun, total=False):
    """One collective-tuning run.

    Attributes:
        op (str | None): The collective operation tuned.
        algo (str | None): The algorithm selected.
        size_bytes (int | None): The message size tuned for.
        world_size (int | None): The participating rank count.
        gain_pct (float | None): The gain the run claimed.
        withheld (bool): Whether the candidate was withheld from adoption.
        withhold_reason (str | None): Why it was withheld.
    """

    op: str | None
    algo: str | None
    size_bytes: int | None
    world_size: int | None
    gain_pct: float | None
    withheld: bool
    withhold_reason: str | None


class V6KernelForgeLanes(TypedDict, total=False):
    """The four forge candidate lanes, split back out at assembly."""

    kernel_rewrites: list[V6KernelRewriteRun]
    fusion_runs: list[V6KernelFusionRun]
    gemm_tuning_runs: list[V6KernelGemmTuningRun]
    collective_runs: list[V6KernelCollectiveRun]


class V6KernelRebenchEngagement(TypedDict, total=False):
    """Whether the configuration under test actually took effect.

    This is the part the orchestrator already computed but never persisted: the
    GEAK verdict path compares the config fingerprint and the overlay digest to
    decide ``validated`` versus ``fallback``, then dropped both booleans once
    the decision was made. Without them a ``fallback`` cannot be told from a
    genuine regression, because a rebench whose config never engaged measured
    the baseline rather than the candidate.

    Attributes:
        config_matched (bool | None): Whether the observed config fingerprint
            matched the expected one.
        overlay_loaded (bool | None): Whether the expected overlay was loaded.
        expected_cfg_hash (str | None): The fingerprint the attempt asked for.
        observed_cfg_hash (str | None): The fingerprint the server reported.
        expected_overlay_digest (str | None): The overlay digest asked for.
        observed_overlay_digest (str | None): The overlay digest reported.
    """

    config_matched: bool | None
    overlay_loaded: bool | None
    expected_cfg_hash: str | None
    observed_cfg_hash: str | None
    expected_overlay_digest: str | None
    observed_overlay_digest: str | None


class V6KernelRebenchAttempt(V6RowScope, total=False):
    """One end-to-end re-measurement of a candidate.

    One section holds both the forge and the GEAK attempts, discriminated by
    ``source_kind``, because adoption is settled by looking them up together:
    an attempt id resolves to a verdict regardless of which producer's
    candidate it re-measured. Assembly routes them back to their two wire
    locations.

    Attributes:
        attempt_id (str): Ledger-stable identifier for this attempt.
        source_kind (str): The producer whose candidate this re-measured.
        source_ref (str | None): The candidate's ``run_id``.
        idempotency_key (str | None): The dispatch idempotency key.
        task_id (str | None): The dispatched task id.
        dispatched_at (str | None): ISO timestamp the attempt was dispatched.
        settled_at (str | None): ISO timestamp the verdict landed.
        base_tput (float | None): The throughput the attempt measured against.
        measured_tput (float | None): The throughput the attempt measured.
        decision (str | None): The verdict, absent while unsettled.
        decision_reason (str | None): Why the verdict landed that way.
        status (str | None): The attempt's own lifecycle status.
        engagement (V6KernelRebenchEngagement): Config / overlay verification.
    """

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


class V6KernelDiscoveredKernel(TypedDict, total=False):
    """One kernel the trace attributed, with profiling fields the summary drops.

    Attributes:
        kernel_id (str): Stable kernel identity.
        name (str): Kernel name as reported by the trace.
        snapshot_id (int | None): The analysis snapshot this row came from.
        provenance (str): Why this snapshot was recorded.
        gpu_pct (float | None): Share of GPU time.
        duration_us (float | None): Total GPU time in microseconds.
        call_count (int | None): Invocations in the traced window.
        kernel_category (str): TraceLens category bucket.
        bottleneck (str | None): Bottleneck verdict from the analyzer.
        bound_type (str | None): Memory/compute bound classification.
        arithmetic_intensity (float | None): FLOPs per byte moved.
        flops_per_byte (float | None): As reported before the fallback above.
        efficiency_percent (float | None): Achieved share of the kernel ceiling.
        bandwidth_util_pct (float | None): Share of peak bandwidth.
        compute_util_pct (float | None): Share of peak compute.
        source_file (str | None): Source file the kernel was attributed to.
        optimization_notes (str): Analyzer notes or suggestions.
        recommended_backends (list[str]): Backends the analyzer recommends.
        recommended_actions (list[str]): Actions the analyzer recommends.
        reusable_native_kernel (bool): Whether a native rewrite is in play.
        selected (bool): Whether the visit considered this kernel a target.
    """

    kernel_id: str
    name: str
    snapshot_id: int | None
    provenance: str
    gpu_pct: float | None
    duration_us: float | None
    call_count: int | None
    kernel_category: str
    bottleneck: str | None
    bound_type: str | None
    arithmetic_intensity: float | None
    flops_per_byte: float | None
    efficiency_percent: float | None
    bandwidth_util_pct: float | None
    compute_util_pct: float | None
    source_file: str | None
    optimization_notes: str
    recommended_backends: list[str]
    recommended_actions: list[str]
    reusable_native_kernel: bool
    selected: bool


class V6KernelForge(TypedDict, total=False):
    """The forge route's work for one visit.

    Attributes:
        engaged (bool): Whether the forge route ran at all.
        reprofile (V6KernelReprofile | None): The entry re-profile.
        trace_analyze_runs (list[V6KernelTraceAnalyzeRun]): Analyses the phase
            requested for itself.
        discovered_kernels (list[V6KernelDiscoveredKernel]): Profiling-rich
            kernel table this visit targeted.
        recommended_kernels (list[V6KernelDiscoveredKernel]): Subset marked as
            optimization targets.
        lanes (V6KernelForgeLanes): The four candidate lanes.
        rebench_ledger (list[V6KernelRebenchAttempt]): Forge re-measurements.
    """

    engaged: bool
    reprofile: V6KernelReprofile | None
    trace_analyze_runs: list[V6KernelTraceAnalyzeRun]
    discovered_kernels: list[V6KernelDiscoveredKernel]
    recommended_kernels: list[V6KernelDiscoveredKernel]
    lanes: V6KernelForgeLanes
    rebench_ledger: list[V6KernelRebenchAttempt]


class V6KernelGeakHandoff(TypedDict, total=False):
    """The conditions GEAK was asked to work under.

    ``baseline_flags`` and ``baseline_envs`` are the orchestrator's current
    best, meaning GEAK's *starting* point, while the accepted flags GEAK later
    reports in :class:`V6KernelGeakProduct` are what it *produced*. The two are
    named apart because one ``config`` block holding both would be read
    backwards, and their difference is the configuration surface this
    delegation actually moved.

    Attributes:
        schema_version (int | None): Handoff schema version.
        model_path (str | None): The served model.
        framework (str | None): The serving framework.
        gpu_type (str | None): The GPU the run targeted.
        tp (int | None): Tensor-parallel width.
        workload (dict[str, Any]): The workload GEAK optimized against.
        baseline_flags (str | None): Server flags GEAK started from.
        baseline_envs (str | dict[str, Any] | None): Environment GEAK started
            from.
        baseline_env_spec_present (bool): Whether a structured env spec was
            handed over.
        launch_recipe (str | None): The launch recipe handed over.
        raw_baseline_tput (float | None): Unadjusted baseline throughput.
        orchestrator_best_tput_same_config (float | None): The orchestrator's
            own best throughput at the same configuration.
        max_model_len (int | None): Context length handed over.
        mem_fraction (float | None): Memory fraction handed over.
        bench_client (str | None): The benchmark client to use.
        e2e_metric (str | None): The metric to optimize.
        bench_protocol_present (bool): Whether a bench protocol was handed over.
        gpu_ids (str | None): The GPUs made available.
        exp_root (str | None): The runner's experiment root.
        eval_dir (str | None): The macro-cycle-scoped eval dir.
    """

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
    """How the delegated GEAK runner process itself ended.

    Separate from what GEAK claimed and from what the rebench measured: a
    runner can exit non-zero having still produced an adoptable candidate, and
    a clean exit is not evidence of a gain.

    Attributes:
        runner_status (str): The runner's own status.
        started_at (str | None): ISO timestamp the runner started.
        ended_at (str | None): ISO timestamp the runner ended.
        duration_sec (float | None): Wall-clock seconds the runner took.
        error_class (str | None): The failure class, on a miss.
        error (str | None): The failure message, on a miss.
        returncode (int | None): The runner's exit code.
        runner_timeout_sec (int | None): The runner's budget.
        kill_timeout_sec (int | None): The runner's hard-kill budget.
        exp_root (str | None): The runner's experiment root.
        eval_dir (str | None): The macro-cycle-scoped eval dir.
        report_path (str | None): The human report the runner wrote.
        versions (dict[str, Any]): Tool version provenance.
        recovered_from_disk (bool): Whether the result was reconstructed from
            disk after the runner died without reporting.
        stages_reached (list[str]): Stages a crashed run got through.
    """

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
    """One hot-kernel discovery run GEAK performed for itself.

    Attributes:
        source (str | None): Discovery source.
        status (str | None): Run status.
        hot_kernel_count (int): Hot kernels surfaced.
        scan (dict[str, Any]): Scan inputs and outputs.
    """

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
    """One kernel GEAK considered, replayed from its conclusion file.

    GEAK's ``kernel_journey.json`` names every kernel it considered, which
    backends it dispatched and what each measured, not just the acceptances
    that survived. These rows are shaped like forge's because the orchestrator
    replays them through the same field helpers, which is exactly why they are
    stored under GEAK and tagged with their producer rather than merged into
    the forge lane the superseded projection appended them to.

    Attributes:
        kernel_id (str): Kernel identifier.
        dispatched (bool): Whether any backend was dispatched.
        backends (list[str]): Backends dispatched to.
        skip_reason (str | None): Gate reason when not dispatched.
        task_group (str | None): The dispatch task group.
        backend_result (V6KernelGeakBackendResult | None): Backend measurement.
        e2e (V6KernelRewriteE2E | None): Integration sub-result.
    """

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
    """One kernel GEAK authored and accepted.

    GEAK routes an acceptance to its kernel queue or its head queue purely by
    which queue proposed it, and both lanes carry the same parity-checked
    ``e2e_delta_pct``; reading only the first drops most of the campaign.

    Attributes:
        short_name (str | None): The kernel's symbol name.
        kernel_id (str | None): Kernel identifier.
        cand_tag (str | None): The candidate slot that proposed it.
        name_source (str): ``symbol`` or ``cand_tag``, naming which of the two
            identified the row, so a tag-named acceptance cannot be mistaken
            for a symbol-named one.
        op_kind (str | None): The operation kind.
        lane (str | None): The queue that proposed it.
        e2e_delta_pct (float | None): GEAK's own end-to-end delta.
        alias_collapsed (bool): Whether an alias twin was folded into this row.
    """

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
    """What GEAK reported about itself, before any re-measurement.

    Every number here is the optimizer's own account of its run. ``verified``
    is a constant ``False`` so a consumer cannot mistake this block for a
    conclusion: nothing in it has been re-measured by the orchestrator's own
    harness, and the adoption verdict rests solely on the rebench.

    Attributes:
        verified (bool): Always ``False``.
        self_reported_tput (float | None): Throughput GEAK claimed.
        self_reported_speedup (float | None): Speedup GEAK claimed.
        self_reported_gain_pct (float | None): Gain GEAK claimed.
        self_reported_basis (str | None): What the claim was measured against.
        geak_status (str | None): GEAK's own terminal status.
        baseline_alignment_status (str | None): Whether GEAK's baseline agreed
            with the orchestrator's.
        authored_kernels (list[V6KernelGeakAuthoredKernel]): Kernels authored.
        env_selections (list[V6KernelGeakEnvSelection]): Environment picks.
        kernels_optimized (int): Assembly-derived count of authored kernels.
        accepted_heads_count (int): Assembly-derived head-queue acceptances.
        validated_regimes (list[Any]): Regimes GEAK says it validated in.
    """

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
    """The reproducible configuration GEAK handed back.

    ``cfg_hash`` and ``final_overlay_digest`` are the expected side of the
    engagement check in :class:`V6KernelRebenchEngagement`: without them a
    rebench cannot prove the configuration it measured was the one GEAK
    produced.

    Attributes:
        accepted_flags (str | list[str] | None): Server flags GEAK accepted.
        accepted_envs (dict[str, Any]): Environment GEAK accepted.
        accepted_config (dict[str, Any]): The runner's accepted-config block.
        cfg_hash (str | None): Canonical fingerprint of flags and envs.
        final_overlay (str | None): The overlay PYTHONPATH produced.
        final_overlay_digest (str | None): Digest of that overlay.
        final_launch_script (str | None): The optimized launch script.
        bench_script (str | None): The benchmark script GEAK measured with.
        final_patch (str | None): The aggregate source patch.
    """

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
    """The orchestrator's own re-measurement campaign for GEAK's candidate.

    GEAK may rebench the same candidate up to its per-cycle ceiling, so unlike
    a forge lane it can end a visit holding several settled verdicts. Two that
    disagree is a fact worth seeing rather than one to resolve by recency:
    ``conflicting_decisions`` is populated and neither verdict is honoured.

    Attributes:
        required (bool): Whether a rebench was required at all.
        max_attempts (int | None): The per-cycle attempt ceiling.
        attempts_used (int): Assembly-derived count of attempts made.
        attempts (list[V6KernelRebenchAttempt]): The attempts.
        final_status (str | None): The revalidation status stamped on the
            result.
        final_error_class (str | None): The revalidation failure class.
        final_error (str | None): The revalidation failure message.
        conflicting_decisions (list[str]): The disagreeing verdicts, when
            settled attempts did not agree.
    """

    required: bool
    max_attempts: int | None
    attempts_used: int
    attempts: list[V6KernelRebenchAttempt]
    final_status: str | None
    final_error_class: str | None
    final_error: str | None
    conflicting_decisions: list[str]


class V6KernelGeak(TypedDict, total=False):
    """The GEAK route's work for one visit, in causal order.

    The five blocks are the five distinct things a consumer conflates at its
    peril: what GEAK was asked to do, how its process ended, what it tried,
    what it claimed, what it produced, and what the orchestrator measured.
    """

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
    """One candidate no settled rebench concluded on.

    ``why`` separates the reasons that are routinely conflated: ``no_rebench``
    means nothing re-measured it, ``rebench_inconclusive`` means something did
    and concluded nothing (a rebench whose configuration never engaged measured
    the baseline, not the candidate), and ``rebench_conflict`` means two
    settled verdicts disagreed.
    """

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
    """What the visit concluded, settled against the rebench evidence.

    ``verdict`` is left absent when nothing was adopted, so a visit that
    adopted nothing cannot read as having concluded something about a
    candidate. ``net_gain_pct`` is computed against ``tput_before`` rather than
    the session baseline, because a visit is answerable for the change it made,
    not for the gains that preceded it.

    Attributes:
        route (str): The route the visit ran.
        verdict (str | None): The entry's conclusion, absent when nothing was
            adopted.
        exit_reason (str | None): The phase's own exit reason.
        tput_before (float | None): Throughput the visit started from.
        tput_after (float | None): Throughput the visit exited on.
        net_gain_pct (float | None): Change across the visit.
        session_baseline_tput (float | None): The session's baseline.
        cumulative_gain_validated_out (float | None): Validated cumulative gain
            on exit.
        stack_depth_out (int | None): Optimization-stack depth on exit.
        adopted (list[V6KernelAdoptedRow]): Candidates a rebench validated.
        pending_review (list[V6KernelPendingRow]): Candidates nothing settled.
        by_source (dict[str, V6KernelSourceCounters]): Per-source tally.
        stack_delta (V6KernelStackDelta): Stack entries added and removed.
    """

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
    """``ext`` of the V6 ``kernel`` timeline event.

    ``geak`` and ``forge`` are mutually exclusive by construction: the entry
    hook picks one of three routes, so the block that did not run stays absent
    rather than being emitted empty.

    Attributes:
        macro_cycle (int): The macro cycle this visit belongs to. It identifies
            the visit, so it sits here rather than being repeated inside each
            route's block.
        in_flight_stage (str | None): The stage in progress, absent once the
            visit closed. A killed session leaves this set, naming where it
            stopped.
        duration_sec (float | None): Wall-clock seconds the visit took.
        entry (V6KernelEntry): What the entry hook decided.
        geak (V6KernelGeak | None): The GEAK route's work.
        forge (V6KernelForge | None): The forge route's work.
        outcome (V6KernelOutcome): What the visit concluded.
        failure (V6KernelFailure | None): The stage that failed, on a miss.
    """

    macro_cycle: int
    in_flight_stage: str | None
    duration_sec: float | None
    entry: V6KernelEntry
    geak: V6KernelGeak | None
    forge: V6KernelForge | None
    outcome: V6KernelOutcome
    failure: V6KernelFailure | None


class V6KernelEvent(TypedDict, total=False):
    """Recorder fragment for one KERNEL event (section ``kernel_event``).

    Created when the phase is entered and upserted as the run proceeds; the
    row-shaped facts live in their own sections and are folded in at assembly.
    Everything here is a mapping, so the upsert's recursive merge applies and
    a partial update never drops a field an earlier one set.

    ``timeline_sequence`` is the storage sequence the opening timeline write
    returned. It lives here because the fragment is the event's only durable
    identity: the closing write reuses it to update that same event in place,
    and its absence is what tells finalize this run never got an event of its
    own, as opposed to getting one it never closed.

    Attributes:
        event_id (str): The event id this fragment is keyed by.
        macro_cycle (int): The macro cycle this run belongs to.
        timeline_sequence (int | None): Storage sequence of the event.
        in_flight_stage (str | None): The stage in progress.
        start_time (str): ISO UTC timestamp the phase was entered.
        end_time (str | None): ISO UTC timestamp the run closed.
        status (str | None): The closing status, absent while running.
        entry (V6KernelEntry): What the entry hook decided.
        route (str): The route the run took.
        forge_engaged (bool): Whether the forge route ran.
        reprofile (V6KernelReprofile | None): The entry re-profile.
        geak_engaged (bool): Whether the GEAK route ran.
        geak_handoff (V6KernelGeakHandoff | None): Conditions GEAK got.
        geak_delegation (V6KernelGeakDelegation | None): How the runner ended.
        geak_claim (V6KernelGeakClaim | None): What GEAK claimed.
        geak_product (V6KernelGeakProduct | None): What GEAK produced.
        geak_rebench (V6KernelGeakRebench | None): Campaign-level rebench
            facts; the attempts themselves are their own section.
        outcome (V6KernelOutcome): The recorded part of the conclusion; the
            settled rows and counters are derived at assembly.
        failure (V6KernelFailure | None): The stage that failed, on a miss.
    """

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
        metadata (V6Metadata): Task identity: session ids and lifecycle, the
            launch configuration and model architecture, tool versions, and
            the Langfuse entrypoint. Recorded at author time.
        baseline (Baseline): Pre-optimization reference performance.
        final (Final): Final validated optimization state.
        phase_timeline (list[PhaseEvent]): Flat per-action timeline.
        phase_segments (list[PhaseSegment]): Phase-boundary view.
        capability_summary (CapabilitySummary): Per-capability roll-up.
        geak (Geak): GEAK route diagnostics and accepted artifacts.
        kernel_lifecycle (KernelLifecycle): Kernels grouped by lifecycle stage.
        collective (Collective): Collective-lane campaigns and their E2E
            verdicts; empty {} when the lane never ran.
        param_search (ParamSearch): Merged explore-search ledger.
        critic_robustness (CriticRobustness): Critic reviews and robustness signals.
        telemetry (Telemetry): Telemetry artifacts and aggregated metrics.
        optimizations (Optimizations): Canonical adopted-optimization read
            model spanning Warm Replay, Explore, Framework Agent, and Kernel
            Agent.
        specialist_runs (list[SpecialistRound]): Specialist sub-agent dispatch records.
        kernel_roofline (KernelRoofline): Hot-kernel table for the dashboard.
        roofline (list[dict[str, Any]]): Per-snapshot roofline comparison list for
            the markdown report's ``## Roofline`` section.
        roofline_progress (RooflineProgress): Optimization-progress curve for the dashboard.
        warnings (list[str]): Collector warnings emitted while assembling the file.
        source_files (SourceFiles): Paths to the source artifacts used.
    """

    schema_version: str
    exported_at_utc: str
    exporter_version: str

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
    collective: Collective
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
    # Kernel-major unified lifecycle view (discovery -> dispatch -> backend attempts -> e2e); {} when absent.
    kernel_journey: KernelJourney
    # Enablement attempt-runtime observability; {} → dashboard hides the block.
    enablement: EnablementBreakdown
    metadata: V6Metadata
    outcome: V6Outcome
    timeline: list[V6TimelineEvent]
    close: V6Close
    robustness: V6Robustness

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
    "V6MetadataLangfuse",
    "V6MetadataRecovery",
    "V6MetadataSession",
    "V6MetadataVersions",
    "V6ModelArchitecture",
    "V6ToolVersion",
    "V6BaselineProgress",
    "V6Close",
    "V6Robustness",
    "V6RobustnessIntent",
    "V6RobustnessTurn",
    "V6ConcSweepArm",
    "V6ConcSweepExt",
    "V6EnablementAttempt",
    "V6EnablementBuild",
    "V6EnablementExt",
    "V6EnablementRevalidation",
    "V6PhaseAction",
    "V6PhaseExt",
    "V6PhaseMarker",
    "V6PhaseSegment",
    "V6StackAdoption",
    "V6StackExt",
    "V6StackValidation",
    "V6ConcSweepPair",
    "V6ConcSweepPoint",
    "V6CriticReview",
    "V6CriticReviewVariant",
    "V6KBWriteBackExt",
    "V6KernelAdoptedRow",
    "V6KernelAnalysisArtifacts",
    "V6KernelAnalysisDetail",
    "V6KernelCollectiveRun",
    "V6KernelEntry",
    "V6KernelEvent",
    "V6KernelExt",
    "V6KernelFailure",
    "V6KernelDiscoveredKernel",
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
    "V6RooflineEventSnapshot",
    "V6RooflineKernel",
    "V6RooflineKernelTable",
    "V6RooflineProgress",
    "V6RooflineTrajectoryPoint",
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
