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

#: Historical collector-only schema retained for archived-reader identification.
SCHEMA_VERSION_V2 = "hyperloom.session_breakdown.v2"

#: breakdown schema version stamped when the file was assembled from the
#: author-time recorder fragments. Same wire shape as v2 plus recorder-only
#: sections; lets consumers tell a recorder-aggregated breakdown apart from a
#: legacy collector fallback.
SCHEMA_VERSION_V3 = "hyperloom.session_breakdown.v3.0"

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
            ``time_only``).
        value (Any): Goal value — a float target, a string (e.g.
            ``target_baseline_dir``), or None when not applicable.
    """

    kind: str  # gain_pct / tput / baseline / time_only
    value: Any  # float or str (target_baseline_dir) or None


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
        throughput_tok_s_per_gpu (float | None): Final throughput (tok/s/GPU), or None.
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
            ``validate_stack_disk`` / ``stack_top_disk`` / ``unavailable``).
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
    ttft_e2el_source: str  # current_best / validate_stack_disk / stack_top_disk / unavailable
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
    sweep: CapabilityEntry
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


# Sweep
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
        status (str): Point status. ``ok`` when a readable JSON object was
            read and ``success`` was not false; ``failed`` when the report
            says failure, was unreadable, or ``abort_reason.json`` is
            present with no report; ``skipped`` when neither file exists.
        error (str | None): Non-empty failure reason when ``status`` is
            ``failed``, else None. Present on every row so table-shaped
            consumers see a stable key set.
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
    status: str  # ok / skipped / failed
    error: str | None
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
    signal: str  # crash / stall / disk_full / cluster_fault / ...
    action: str  # what was done
    workdir: str


class CriticRobustness(TypedDict, total=False):
    """Critic-review iterations and robustness signals for the session.

    Attributes:
        critic_iterations (list[CriticIteration]): Critic-agent review passes.
        robustness_signals (list[RobustnessSignal]): Fault/recovery events handled.
        kb_writes_summary (CriticKBWritesSummary): Tally of the critic
            iterations' verdicts (``total`` plus ``by_verdict``).
    """

    critic_iterations: list[CriticIteration]
    robustness_signals: list[RobustnessSignal]
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


# KB Provenance — RecipeKB / PR Monitor integration
class KBQueueStats(TypedDict, total=False):
    """Depth statistics for the on-disk KB write queues.

    Attributes:
        pending_lines (int): Current depth of ``.kb_pending.ndjson``.
        flushed_bookmarks (int): Drain-bookmark rows in ``.kb_flushed.ndjson``.
        dead_letter_lines (int): Rows in ``.kb_dead_letter.ndjson``.
    """

    pending_lines: int  # current depth of .kb_pending.ndjson
    flushed_bookmarks: int  # rows in .kb_flushed.ndjson (drain bookmarks)
    dead_letter_lines: int  # rows in .kb_dead_letter.ndjson


class KBFlusherStatus(TypedDict, total=False):
    """``kb_provenance.flusher_status``: boot marker merged with a live pid probe."""

    enabled: bool  # cli flag (false when --no-kb-flusher or --degraded-kb)
    spawned: bool  # daemon was actually subprocess.Popen'd this boot
    alive: bool  # live pid probe at breakdown emit time
    pid: int | None
    interval_sec: float
    batch_size: int
    reason: str  # boot-time spawn decision text
    ts: str  # iso UTC of the boot marker
    pid_path: str  # absolute path to .kb_flusher.pid


class WarmReplayOutcome(TypedDict, total=False):
    """GAP 1 — warm-recipe replay result. Empty {} when it never fired; else ``status`` + per-status fields.

    ``eval_ran`` / ``replay_accuracy`` / ``baseline_accuracy`` are recorded on
    every replay that reached a throughput measurement, not only on rejection:
    a config that was checked and passed is a different record from one that
    was never checked. ``eval_ran`` is what separates "the model scored 0.0"
    from "no score exists", which are otherwise both a null accuracy.

    A measurement that fails never stops the run. The replay is admitted and
    ``eval_error`` carries why no score could be read, so an unjudged promotion
    is visible after the fact rather than silently indistinguishable from a
    judged one.
    """

    status: str
    expected_gain_pct: float
    actual_gain_pct: float
    throughput_after: float
    eval_ran: bool
    eval_error: str | None
    replay_accuracy: float | None
    baseline_accuracy: float | None
    warm_recipe_tier: str
    warm_recipe_conf: float
    config_source: str
    config_donor_tier: str
    donor_canonical_id: str
    donor_model: str
    donor_session_id: str
    donor_family_tags: list[str]
    donor_gain_pct: float
    donor_breakdown_link: str
    replay_task_id: str
    error_class: str
    reason: str


class KBProvenance(TypedDict, total=False):
    """Recipe KB integration audit for the session.

    Covers warm-start context seeded from the KB, the warm-replay outcome,
    queue depth, and the flusher daemon status.

    Attributes:
        recipe_kb_session_id (str): Recipe KB session id.
        warm_start_ts (str): ISO UTC timestamp of warm start.
        warm_start_recipe_seen (bool): Whether a warm recipe was seen.
        warm_start_recipe_tier (str): Tier of the seen warm recipe.
        warm_start_recipe_source (str): KB path that supplied the applied
            warm recipe (e.g. ``kb-store`` / ``recipe_kb``); empty when none.
        warm_start_pitfall_count (int): Number of pitfalls injected at warm start.
        warm_start_lesson_count (int): Number of lessons injected at warm start.
        warm_replay (WarmReplayOutcome): Operator-visible warm-replay summary.
        warm_replay_attempted (bool): Whether a warm replay was attempted.
        warm_history_injected (bool): Whether warm history was injected.
        recipe_finalize (dict[str, Any]): Terminal Recipe publication outcome.
        recipe_finalize_status (str): Persisted publication lifecycle state.
        recipe_finalize_attempts (int): Number of idempotent finalize attempts.
        stack_fingerprint (dict[str, str]): Fingerprint of the optimization stack.
        queue (KBQueueStats): Depth stats for the KB write queues.
        audit_tail_count (int): Number of audit-tail entries.
        audit_status_counts (dict[str, int]): Audit entries counted by status.
        flusher_status (KBFlusherStatus): KB flusher daemon lifecycle marker.
        kb_degraded_reason (str): KB soft-degrade reason (None / ``explicit_flag`` /
            ``ir3_auto``).
        pr_degraded_reason (str): PR Monitor soft-degrade reason (None /
            ``explicit_flag`` / ``ir3_auto``).
    """

    recipe_kb_session_id: str
    warm_start_ts: str
    warm_start_recipe_seen: bool
    warm_start_recipe_tier: str
    # Which KB path (e.g. "kb-store" / "recipe_kb") supplied the applied warm recipe.
    warm_start_recipe_source: str
    warm_start_pitfall_count: int
    warm_start_lesson_count: int
    warm_replay: WarmReplayOutcome
    warm_replay_attempted: bool
    warm_history_injected: bool
    recipe_finalize: dict[str, Any]
    recipe_finalize_status: str
    recipe_finalize_attempts: int
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
    """Summary of critic-agent ``commit-review`` outputs (Coordinator proxies these into ``kb_provenance``)."""

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
    concs_requested: list[int]
    baseline: dict[str, Any]  # {extra_server_args, extra_envs, points[]}
    optimized: dict[str, Any]  # {extra_server_args, extra_envs, points[]}
    comparison: list[dict[str, Any]]  # per-CONC paired rows (feeds the dual curve + speedup bars)
    summary: dict[str, Any]  # {successful_pairs, failed_pairs, best_conc, best_speedup, median_speedup, mean_speedup}
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

    ``route`` and ``tool`` are independent: a request can be answered
    deterministically by the analysis tool, delegated to the agent, or bypassed
    entirely, and the tool that served it is a separate fact from the route
    that chose it. Oversized sub-blocks are replaced by an omission marker
    rather than dropped, so a consumer can tell a bounded block from a missing
    one.

    Attributes:
        route (str): ``agent`` / ``deterministic`` / ``bypass``.
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


class V6KernelForge(TypedDict, total=False):
    """The forge route's work for one visit.

    Attributes:
        engaged (bool): Whether the forge route ran at all.
        reprofile (V6KernelReprofile | None): The entry re-profile.
        trace_analyze_runs (list[V6KernelTraceAnalyzeRun]): Analyses the phase
            requested for itself.
        lanes (V6KernelForgeLanes): The four candidate lanes.
        rebench_ledger (list[V6KernelRebenchAttempt]): Forge re-measurements.
    """

    engaged: bool
    reprofile: V6KernelReprofile | None
    trace_analyze_runs: list[V6KernelTraceAnalyzeRun]
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
        session (SessionMeta): Session identity, timing, and host context.
        workload (Workload): Model/framework/serving configuration.
        model_info (ModelInfo): Structural summary of the served model
            (architecture / scale / attention), parsed from its config.json.
            Empty {} on non-transformers models or pre-field sessions.
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
        sweep (Sweep): Concurrency/shape sweep results.
        critic_robustness (CriticRobustness): Critic reviews and robustness signals.
        telemetry (Telemetry): Telemetry artifacts and aggregated metrics.
        optimizations (Optimizations): Canonical adopted-optimization read
            model spanning Warm Replay, Explore, Framework Agent, and Kernel
            Agent.
        kb_provenance (KBProvenance): Recipe KB integration audit.
        specialist_runs (list[SpecialistRound]): Specialist sub-agent dispatch records.
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
    collective: Collective
    # explore_search is the native merged ledger; param_search is a v1 alias.
    param_search: ParamSearch
    explore_search: ParamSearch
    sweep: Sweep
    critic_robustness: CriticRobustness
    telemetry: Telemetry
    # Single downstream read model for every formally adopted optimization.
    optimizations: Optimizations
    kb_provenance: KBProvenance  # Recipe KB audit
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
    "SCHEMA_VERSION_V2",
    "SCHEMA_VERSION_V3",
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
    "KBProvenance",
    "KBQueueStats",
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
    "Sweep",
    "SweepPoint",
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
    "V6KernelAdoptedRow",
    "V6KernelAnalysisArtifacts",
    "V6KernelAnalysisDetail",
    "V6KernelCollectiveRun",
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
    "Workload",
    "WorkloadObjective",
]
