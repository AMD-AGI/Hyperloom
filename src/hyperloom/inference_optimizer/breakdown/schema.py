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

#: Current breakdown schema version.
SCHEMA_VERSION = SCHEMA_VERSION_V5


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
    ext: dict[str, Any]


class V6WarmStartMatched(TypedDict, total=False):
    """The Recipe the PRELUDE KB lookup selected.

    Present only on a ``matched`` event. ``tier`` and ``confidence`` name the
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
    """``timeline[type=warm_start].ext.reads`` — Recipe KB read attribution.

    Aggregated from the recipe-snapshot audit log (last 50 reads): how the T0
    lookups resolved, which backend served each, and which source supplied the
    champion config. Omitted when the session recorded no readable read.

    Attributes:
        count (int): Number of read rows considered (capped at the last 50).
        hits (int): How many of those reads returned a usable record.
        by_resolution (dict[str, int]): Reads counted by resolution outcome.
        by_remote (dict[str, int]): Reads counted by serving backend
            (e.g. ``kb-store`` vs local ``recipe_kb``).
        by_source (dict[str, int]): Contributing-source counts across the reads.
        best_config_by_source (dict[str, int]): Which source supplied the
            champion config, counted per source.
        tail (list[dict[str, Any]]): The most recent raw audit rows; downstream
            champion-config and donor resolution read the latest hit's own
            ``result`` off these.
    """

    count: int
    hits: int
    by_resolution: dict[str, int]
    by_remote: dict[str, int]
    by_source: dict[str, int]
    best_config_by_source: dict[str, int]
    tail: list[dict[str, Any]]


class V6WarmStartExt(TypedDict, total=False):
    """``timeline[type=warm_start].ext`` — what was asked for, what came back.

    Attributes:
        requested (dict[str, Any]): ``{canonical_id, scope}`` this session asked
            for. ``canonical_id`` is read from ``recipe_finalize`` rather than
            rebuilt, because the hardware dimension is topology-aware.
        match_status (str): The raw ``warm_start_context.status`` when it is not
            a plain hit; ``seed_only`` is a hit that could not be executed and
            would otherwise be indistinguishable from a miss.
        matched (V6WarmStartMatched): Omitted unless the status is ``matched``.
        reads (V6WarmStartReads): Per-source read attribution from the recipe
            snapshot audit; omitted when no read was recorded.
    """

    requested: dict[str, Any]
    match_status: str
    matched: V6WarmStartMatched
    reads: V6WarmStartReads


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
    """``timeline[type=kb_write_back].ext`` — did this session's Recipe land.

    The published Recipe body is deliberately not mirrored here: it is the KB
    Store's record, and duplicating three columns of overlay refs into every
    breakdown would grow the export without answering a question the identity
    and the throughput do not already answer.

    Attributes:
        result_type (str): Stable reason code. The publisher's own vocabulary is
            narrower than it looks — build, transport and upload failures all
            surface as a bare exception class name — so the raw reasons are
            mapped onto a fixed set here.
        raw_reason (str | None): The unmapped reason.
        backend (str | None): ``kb-store`` / ``local`` / ``disabled``.
        canonical_id (str | None): Identity written to.
        session_id (str | None): Session id recorded on the KB side.
        scope (dict[str, Any]): Workload dimensions the Champion is keyed by.
        optimized_throughput (float | None): Throughput submitted, and the value
            compared against the incumbent Champion.
        validated_gain_pct (float | None): Session's cumulative validated gain.
        attempts (int | None): Finalize attempts; above one means it retried.
        source (str | None): ``close`` or the ``t4_fallback`` teardown path.
        queue (dict[str, Any]): Local write-queue depths.
        failure (dict[str, Any]): ``{error_class, error}``.
    """

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
    "V6OutcomeAttribution",
    "V6OutcomeAttributionBySource",
    "V6OutcomeGainBucket",
    "V6OutcomeKernelAttribution",
    "V6Outcome",
    "V6OutcomeValidation",
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
