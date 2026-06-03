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

#: breakdown schema version. v2 adds the
#: ``specialist_runs`` section, ``capability_summary.specialist``
#: row, ``critic_robustness.kb_writes_summary`` sub-block, and the
#: top-level ``action_timeline`` / ``explore_search`` v1-reader
#: aliases. Inv-12.1 guarantees a v0.6 / reader can still
#: consume the file because v2 only *adds* fields.
SCHEMA_VERSION = "hyperloom.session_breakdown.v2"


# ---------------------------------------------------------------------------
# §1 Session metadata
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# §2 Workload configuration
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# §3 Baseline
# ---------------------------------------------------------------------------
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


class BenchmarkInvocation(TypedDict, total=False):
    """Replayable record of how a benchmark variant was launched.

    Allows operators to rerun the exact same workload (server command +
    env vars + config) when investigating a regression. ``extra_envs`` is
    allowlist-filtered in the collector to keep secrets out of the
    breakdown JSON.

    Attributes:
        framework_args (str): Full server launch command line.
        framework_args_source (str): Where the launch line was recovered from
            (``log_non_default_args`` / ``log_args_line`` / ``log_python_cmd`` /
            ``yaml_cmd`` / ``yaml_benchmark`` / ``unknown``).
        extra_envs (dict[str, str]): Allowlisted env vars only (no secrets).
        config_path (str | None): Path to the materialized config yaml, or None.
        server_log_path (str | None): Path to the server log for debug, or None.
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


# ---------------------------------------------------------------------------
# §4 Final state — SaFE contract core
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# §5 Phase timeline — chronological events
# ---------------------------------------------------------------------------
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
    extras: dict[str, Any]


# ---------------------------------------------------------------------------
# §6 Capability summary — Capability cards in UI
# ---------------------------------------------------------------------------
class CapabilityEntry(TypedDict, total=False):
    """Summary of one capability's activity for the UI capability cards.

    Aggregates attempt/keep counts and the best gain for a capability family
    (geak, oob, explore, sweep, specialist, etc.), with explore- and
    specialist-specific extensions.

    Attributes:
        status (str): Capability status (``kept`` / ``tried`` / ``attempted`` /
            ``not_attempted`` / ``not_configured`` / ``failed`` / ``completed``).
        attempts (int): Number of attempts made.
        keeps (int): Number of promoted (kept) variants.
        tested (int): Distinct variants tested (backends/params/explore).
        best_gain_pct (float | None): Best single gain percent seen, or None.
        reason (str): Human-readable explanation of the status.
        keep_unstable_count (int): KEEP'd variants later evicted by stack rebench.
        winners_history (int): Cumulative ``explore_search.winners_history`` length.
        by_specialist (dict[str, CapabilityEntry]): Per-domain split keyed by
            ``SpecialistDomain.key`` (specialist row only); every catalogue
            domain is seeded with a ``not_attempted`` entry.
    """
    status: str                   # kept / tried / attempted / not_attempted / not_configured / failed / completed
    attempts: int
    keeps: int
    tested: int                   # for backends/params/explore: distinct variants tested
    best_gain_pct: float | None
    reason: str                   # human readable, e.g. "kernel-claude only this run"
    # v0.8 M3 explore-specific:
    keep_unstable_count: int      # KEEP'd variants evicted by inlined stack rebench
    winners_history: int          # cumulative explore_search.winners_history length
    # specialist-row only — per-domain split. Keys are
    # SpecialistDomain.key strings (``serving_specialist`` /
    # ``kernel_switch_specialist`` / ``comm_specialist`` /
    # ``compiler_specialist`` / ``system_specialist`` /
    # ``pr_intel_specialist`` / ``session_steward_specialist``). Every
    # catalogue domain is seeded with a not_attempted entry so the
    # dashboard can iterate without presence checks.
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
    # primary explore row; ``backends`` / ``params`` /
    # ``validate_stack`` are kept as compatibility aliases (§3.12 §4.2).
    explore: CapabilityEntry
    backends: CapabilityEntry
    params: CapabilityEntry
    sweep: CapabilityEntry
    validate_stack: CapabilityEntry
    # specialist sub-agent capability row. ``tested``
    # = total proposals_total across all rounds; ``keeps`` =
    # proposals_kept; ``attempts`` = number of dispatch rounds.
    specialist: CapabilityEntry


# ---------------------------------------------------------------------------
# §7 / §8 GEAK / OOB invocations
# ---------------------------------------------------------------------------
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
    """One backend invocation for one kernel.

    Same shape for GEAK and OOB; ``backend`` distinguishes them.

    Attributes:
        kernel_id (str): Id of the kernel being optimized.
        attempt_id (str): Id of this optimization attempt.
        run_id (str): Id of the enclosing run.
        ts (str): ISO UTC timestamp of the invocation.
        backend (str): Backend used (``geak`` / ``claude`` / ``codex``).
        model (str | None): Model that generated the kernel, or None.
        kernel_metadata (KernelMetadata): Metadata of the targeted kernel.
        prompt_path (str | None): Path to the prompt artifact, or None.
        optimized_files (list[str]): Files produced/edited by the attempt.
        result_path (str | None): Path to the result artifact, or None.
        verification_path (str | None): Path to the verification artifact, or None.
        decision (str): Outcome (``KEEP`` / ``PARTIAL`` / ``REVERT`` / ``FAILED``).
        micro_speedup (float | None): Micro-benchmark speedup factor, or None.
        compile_passed (bool | None): Whether compilation passed, or None.
        correctness_passed (bool | None): Whether correctness checks passed, or None.
        best_artifact_path (str | None): Path to the best artifact, or None.
        error (str | None): Error message on failure, or None.
        cli_log_path (str | None): Path to the CLI log, or None.
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


# ---------------------------------------------------------------------------
# §10 Param search
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# §11 Sweep
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# §12 Critic / Robustness
# ---------------------------------------------------------------------------
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
    # counts of KB writes proxied through the
    # critic agent's ``commit-review`` protocol (Coordinator
    # actually performs the writes; the critic only authors them).
    kb_writes_summary: "CriticKBWritesSummary"


# ---------------------------------------------------------------------------
# §13 Telemetry
# ---------------------------------------------------------------------------
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
    """One row of the legacy M6 lane occupancy summary.

    Surfaces resource_lock state (per-lane capacity vs. live holders +
    lifetime expired-lease count) into the breakdown's ``telemetry``
    section so cross-cluster dashboards can chart lane usage alongside
    GPU power / temperature.

    Attributes:
        lane (str): Lane name.
        capacity (int): Configured lane capacity.
        live_holders (int): Number of live lease holders.
        lease_expired_count (int): Lifetime count of expired leases.
    """
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


# ---------------------------------------------------------------------------
# §14 Attribution
# ---------------------------------------------------------------------------
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
    # FRAMEWORK_PR phase contribution (PRELUDE → FRAMEWORK_PR →
    # EXPLORE). Tracks gain from upstream-PR bake-ins as a separate
    # row so the dashboard's per-source totals reconcile against
    # ``validated_total_pct``; previously these KEEPs fell into
    # ``other`` and silently disappeared.
    framework_pr_pct_of_total: float
    # GEMM_TUNING contribution (KERNEL-entry deterministic FP8 GEMM
    # tuner). Bucketed separately from the ``kernel`` family so the
    # dashboard can show "deterministic tuner gain" vs "source-level
    # GEAK / OOB rewrite gain". Always emitted (0.0 on non-FP8
    # workloads / when the tuner skipped or produced no KEEP).
    gemm_tuning_pct_of_total: float
    backends_pct_of_total: float
    params_pct_of_total: float
    sweep_pct_of_total: float
    validated_total_pct: float


class PhaseBreakdownExplore(TypedDict, total=False):
    """v0.8 M7 (KB_design §3.12 §4.6) — explore-phase gain split by
    specialist domain.

    ``by_domain`` keys are normalized — the collector strips
    ``specialist:`` prefixes before bucketing, so consumers see the
    bare SpecialistDomain.key (``serving_specialist`` /
    ``kernel_switch_specialist`` / …). Non-specialist provenance
    appears as ``default_grid`` / ``llm_direct``; resumed-from-v1
    sessions appear as ``legacy_<action>`` (e.g. ``legacy_backends``)
    so they don't masquerade as a real specialist domain. Empty
    provenance falls back to ``unknown``.

    Attributes:
        total_gain_pct (float): Total explore-phase gain percent.
        by_domain (dict[str, float]): Gain percent keyed by normalized domain.
    """
    total_gain_pct: float
    by_domain: dict[str, float]


class PhaseBreakdownKernel(TypedDict, total=False):
    """v0.8 M7 — kernel-phase gain split by ``kernel_id`` (KB_design §3.12 §4.6).

    Attributes:
        total_gain_pct (float): Total kernel-phase gain percent.
        by_kernel_id (dict[str, float]): Gain percent keyed by ``kernel_id``.
    """
    total_gain_pct: float
    by_kernel_id: dict[str, float]


class PhaseBreakdownFrameworkPr(TypedDict, total=False):
    """FRAMEWORK_PR phase gain split by adopted PR
    reference. ``by_pr`` keys are the entry's ``variant_name`` (PR
    label, typically ``PR:<repo>#<num>`` or ``PR:<num>``); empty
    string falls back to ``"?"``.

    Attributes:
        total_gain_pct (float): Total FRAMEWORK_PR phase gain percent.
        by_pr (dict[str, float]): Gain percent keyed by PR label.
    """
    total_gain_pct: float
    by_pr: dict[str, float]


class PhaseBreakdownGemmTuning(TypedDict, total=False):
    """KERNEL-entry FP8 GEMM tuning gain split by tuned-CSV path.
    ``by_tuned_file`` keys on the entry's ``tuned_file`` (absolute
    path to ``a8w8_blockscale_tuned_gemm.csv``); fallbacks: entry's
    ``variant_name`` then ``"?"`` so the key is always a string.

    Attributes:
        total_gain_pct (float): Total GEMM-tuning gain percent.
        by_tuned_file (dict[str, float]): Gain percent keyed by tuned-CSV path.
    """
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
    # GEMM_TUNING is a KERNEL-entry deterministic step; bucketed
    # separately so the dashboard can split tuner gain from
    # source-level GEAK/OOB rewrite gain.
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
    # per-phase gain attribution.
    phase_breakdown: PhaseBreakdown
    notes: list[str]              # human-readable caveats


# ---------------------------------------------------------------------------
# §16 Phase segments — v0.8 M2 phase state machine
# ---------------------------------------------------------------------------
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
    exit_reason: str           # KB_design §3.2 §6 vocab entry; "" for current segment
    evidence: dict[str, Any]   # entry evidence (snapshot at transition time)
    actions: list[PhaseEvent]  # events from phase_timeline whose ts ∈ [entered, exit)
    elapsed_seconds: float | None


# ---------------------------------------------------------------------------
# §15 KB Provenance — Cortex KB integration
# ---------------------------------------------------------------------------
class KBPendingEdge(TypedDict, total=False):
    """A KB edge proposal queued but not yet committed.

    Attributes:
        proposal_msg_id (str): Id of the originating proposal message.
        edge_id (str): Id of the proposed KB edge.
        action (str): Action that produced the proposal.
        ts (str): ISO UTC timestamp of the proposal.
    """
    proposal_msg_id: str
    edge_id: str
    action: str
    ts: str


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


class KBCommitSummary(TypedDict, total=False):
    """Outcome of committing the session's KB edges.

    Attributes:
        status (str): Commit status (``committed`` / ``commit_failed`` /
            ``skip_disabled`` / ...).
        promoted_edges (list[str]): Ids of edges promoted on commit.
        derived_summary_id (str): Id of the derived summary node.
    """
    status: str                    # committed / commit_failed / skip_disabled / ...
    promoted_edges: list[str]
    derived_summary_id: str


class KBPointCreated(TypedDict, total=False):
    """One row in ``kb_provenance.points_created``.

    ``kind`` ∈ {workload_node / issue_node / optimization_node /
    pr_node / attempt_node / ...}. ``pr_node`` rows are the M4
    contribution; everything else came from M1/M3 path.

    Attributes:
        canonical_id (str): Canonical id of the created KB point.
        kind (str): Node kind (``workload_node`` / ``issue_node`` /
            ``optimization_node`` / ``pr_node`` / ``attempt_node`` / ...).
        authority (str): Authority that asserted the point.
        source (str): Source that produced the point.
        status (str): Creation status.
        ts (str): ISO UTC timestamp of creation.
    """
    canonical_id: str
    kind: str
    authority: str
    source: str
    status: str
    ts: str


class KBFlusherStatus(TypedDict, total=False):
    """``kb_provenance.flusher_status`` (v0.8 KB_gaps/Dead-E).

    Merge of the cli boot marker (``.kb_flusher_status.json``) and a
    live ``kill -0 $pid`` probe at breakdown emit time. Populated even
    for ``--degraded-kb`` / ``--no-kb-flusher`` sessions so operators
    can grep a single key.

    Attributes:
        enabled (bool): CLI flag (false when ``--no-kb-flusher`` / ``--degraded-kb``).
        spawned (bool): Whether the daemon was subprocess-spawned this boot.
        alive (bool): Live pid probe result at breakdown emit time.
        pid (int | None): Daemon pid, or None.
        cortex_kb_url (str | None): Cortex KB URL, or None.
        interval_sec (float): Flush interval in seconds.
        batch_size (int): Flush batch size.
        reason (str): Boot-time spawn decision text.
        ts (str): ISO UTC timestamp of the boot marker.
        pid_path (str): Absolute path to ``.kb_flusher.pid``.
    """
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
    """GAP 1 — warm-recipe replay result. Empty {} when the replay
    never fired (``--no-warm-replay`` / low confidence / no recipe);
    otherwise one of ``in_flight`` / ``reproduced`` / ``drift`` /
    ``failed`` / ``skipped`` with the per-status fields populated.

    Attributes:
        status (str): Replay status (``in_flight`` / ``reproduced`` / ``drift`` /
            ``failed`` / ``skipped``).
        expected_gain_pct (float): Gain percent the recipe predicted.
        actual_gain_pct (float): Gain percent actually measured.
        throughput_after (float): Throughput after replay.
        warm_recipe_tier (str): Tier of the warm recipe used.
        warm_recipe_conf (float): Confidence score of the warm recipe.
        replay_task_id (str): Task id of the replay attempt.
        error_class (str): Error classification on failure.
        reason (str): Human-readable explanation of the outcome.
    """
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
    # GAP 1 — operator-visible warm-replay summary.
    warm_replay: WarmReplayOutcome
    warm_replay_attempted: bool
    warm_history_injected: bool
    stack_fingerprint: dict[str, str]
    pending_edges: list[KBPendingEdge]
    queue: KBQueueStats
    audit_tail_count: int
    audit_status_counts: dict[str, int]
    # points created during this session.
    points_created: list[KBPointCreated]
    points_by_kind: dict[str, int]
    commit_summary: KBCommitSummary
    # v0.8 KB_gaps/Dead-E — Cortex KB flusher daemon lifecycle marker.
    flusher_status: KBFlusherStatus
    # IR-3 soft-degrade audit. Values:
    # ``None`` (KB / PR Monitor reachable, no degrade), ``"explicit_flag"``
    # (operator passed ``--degraded-{kb,pr}``), or ``"ir3_auto"`` (IR-3
    # probe failed and cli auto-enabled the corresponding degrade).
    kb_degraded_reason: str
    pr_degraded_reason: str


# ---------------------------------------------------------------------------
# specialist_runs section
# ---------------------------------------------------------------------------
class SpecialistDomainBreakdown(TypedDict, total=False):
    """Per-domain attribution for one ``specialist_rounds`` entry.

    Mirror of ``SharedState.specialist_rounds[i].domain_breakdown[domain]``
    contents.

    Attributes:
        dispatched (int): Number of specialists dispatched for the domain.
        proposals_total (int): Total proposals produced.
        proposals_kept (int): Proposals that were kept.
        proposals_rejected (int): Proposals that were rejected.
    """
    dispatched: int
    proposals_total: int
    proposals_kept: int
    proposals_rejected: int


class SpecialistTranscriptRef(TypedDict, total=False):
    """Reference to a specialist transcript on disk.

    Default behaviour (``--breakdown-include-transcripts=false``) is
    to record only the relative path; ``true`` inlines the raw
    transcript bytes under ``body``.

    Attributes:
        task_id (str): Task id of the specialist run.
        domain (str): Specialist domain key.
        path (str): Relative path to the transcript on disk.
        body (str): Raw transcript bytes; only set when the CLI flag is enabled.
    """
    task_id: str
    domain: str
    path: str
    body: str   # only set when CLI flag enabled


class SpecialistRound(TypedDict, total=False):
    """One dispatch round of specialist sub-agents (element of ``specialist_runs``).

    Records a round in which one or more domain specialists were dispatched in
    parallel, their proposal tallies, per-domain breakdown, and transcripts.

    Attributes:
        round_id (int): Sequential id of the dispatch round.
        dispatched_at (str): ISO UTC timestamp when the round was dispatched.
        completed_at (str): ISO UTC timestamp when the round completed.
        domains (list[str]): Specialist domains dispatched in the round.
        parallelism (int): Number of specialists run concurrently.
        proposals_total (int): Total proposals across the round.
        proposals_kept (int): Proposals that were kept.
        proposals_rejected (int): Proposals that were rejected.
        proposals_skipped (int): Proposals that were skipped.
        kb_edge_ids (list[str]): Retired field (always empty); kept for reader compat.
        confidence_avg (float | None): Average proposal confidence, or None.
        domain_breakdown (dict[str, SpecialistDomainBreakdown]): Per-domain tallies.
        transcripts (list[SpecialistTranscriptRef]): Specialist transcript references.
        notes (list[str]): Human-readable notes about the round.
    """
    round_id: int
    dispatched_at: str
    completed_at: str
    domains: list[str]
    parallelism: int
    proposals_total: int
    proposals_kept: int
    proposals_rejected: int
    proposals_skipped: int
    # Retired field — was populated by the T2 hypothesize hook (now
    # gone). Kept on the schema so claw-stats-service readers that
    # destructure specialist_runs[] don't break; always empty.
    kb_edge_ids: list[str]
    confidence_avg: float | None
    domain_breakdown: dict[str, SpecialistDomainBreakdown]
    transcripts: list[SpecialistTranscriptRef]
    notes: list[str]


# ---------------------------------------------------------------------------
# critic_robustness.kb_writes_summary sub-block
# ---------------------------------------------------------------------------
class CriticKBWritesSummary(TypedDict, total=False):
    """Summary of critic-agent ``commit-review`` outputs.

    The Coordinator proxies these writes into ``kb_provenance``.

    Attributes:
        total (int): Total number of critic KB writes.
        by_verdict (dict[str, int]): Write counts by verdict (``KEEP`` /
            ``REVERT`` / ``NEEDS_INFO`` / ...).
    """
    total: int
    by_verdict: dict[str, int]   # KEEP / REVERT / NEEDS_INFO / ...


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Roofline — optimization-progress curve for the dashboard
# ---------------------------------------------------------------------------
# Drives the "优化进度曲线" panel (Dashboard-Roofline 对接清单 §2): a
# stepped line from baseline through every KEEP, plotted against two
# horizontal reference lines (ceiling = vendor peak, target = ceiling
# × 0.70). All inputs derived from ``state.json`` so the dashboard
# only needs to read ``session_breakdown.json``.
class RooflineTrajectoryPoint(TypedDict, total=False):
    """One x/y/tooltip on the optimization-progress curve.

    ``x`` is an iso UTC timestamp; the dashboard may convert to a step
    index for the horizontal axis. The first point is always
    ``label = "baseline"`` (taken from ``manifest.created_at_utc`` +
    ``state.baseline_tput``); subsequent points come from
    ``state.optimization_stack[]`` in promotion order.

    Attributes:
        ts (str): ISO UTC timestamp (the x value).
        tput (float): Throughput in tok/s (the y value).
        label (str): Point label (``baseline`` / variant_name).
        action (str): Action kind (``baseline`` / ``explore`` / ``kernel_opt`` / ...).
        gain_pct (float): Cumulative gain percent vs baseline at this point.
        flags (str): The variant's ``candidate_extra_server_args``.
        extra_envs (dict[str, str]): KEY=value env pairs the variant set.
    """
    ts: str                          # iso UTC, x value
    tput: float                      # tok/s, y value
    label: str                       # "baseline" / variant_name
    action: str                      # "baseline" / "explore" / "kernel_opt" / ...
    gain_pct: float                  # cumulative gain vs baseline at this point
    flags: str                       # candidate_extra_server_args
    extra_envs: dict[str, str]       # KEY=value pairs the variant set


class RooflineSnapshot(TypedDict, total=False):
    """One ``state.roofline_snapshots[]`` entry mirrored verbatim.

    Kept as a list (the dashboard reads ``snapshots[0]`` for the
    headline ceiling but downstream tooling may want the full
    history). Field shape mirrors the on-disk record so a future
    snapshot field addition flows through transparently.

    Attributes:
        snapshot_id (int): Sequential snapshot id.
        ts (str): ISO UTC timestamp of the snapshot.
        achieved_tok_per_sec (float): Achieved throughput (tok/s).
        theoretical_peak_tok_per_sec (float): Vendor peak ceiling (unreachable).
        within_roofline_pct (float): ``achieved / peak * 100``.
        gap_to_roofline_pct (float): Percent gap to the roofline ceiling.
        compute_pct (float): Compute-bound time share.
        idle_pct (float): Idle time share.
        comm_pct (float): Communication time share.
        top_bottleneck (str): Dominant bottleneck label (e.g. ``MoE_unfused``).
        top_kernel (dict[str, Any]): Top kernel descriptor
            (``{name, bound_type, efficiency_pct, gpu_pct}``).
        analysis_md_path (str): Path to the human-readable analysis markdown.
        kernel_roofline_path (str): Path to the kernel roofline data.
        trace_input (str): Path to the trace input.
    """
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
    """Top-level ``roofline_progress`` section.

    NOTE — this used to be called ``Roofline`` and exported under the
    top-level key ``roofline``, but that collided with the markdown-
    report renderer's pre-existing ``roofline`` list contract (per-
    final.json comparison snapshots, populated by
    ``collect_roofline``). The two surfaces serve different consumers
    and the previous name clash silently broke the markdown report's
    Roofline section. Renamed to ``roofline_progress`` so both
    surfaces coexist.

    Two products in one structure:

    1. **Reference lines** (``ceiling_tok_per_sec`` / ``target_tok_per_sec``):
       horizontal dashed lines on the chart. ``ceiling`` is the
       vendor's theoretical peak (from the latest snapshot);
       ``target = ceiling × ceiling_ratio_target`` (default 0.70 — see
       Dashboard 对接清单 §2.1 for why we don't aim at 100%).

    2. **Trajectory** (``trajectory[]``): the stepped line itself —
       baseline + every KEEP, sorted by ts.

    ``snapshots[]`` carries the raw ``state.roofline_snapshots[]``
    entries verbatim for tooltips / drill-downs; consumers that just
    want to render the chart can ignore it.

    Edge cases (Dashboard-Roofline 对接清单 §5):
    * No snapshot ever taken → ``ceiling_available = False``,
      ``ceiling_tok_per_sec / target_tok_per_sec`` absent. Dashboard
      hides the reference lines.
    * No KEEP yet → ``trajectory`` has the single baseline point only.
      ``current_best_tput == baseline_tput`` and
      ``cumulative_gain_pct == 0.0``.

    Attributes:
        ceiling_tok_per_sec (float | None): Vendor peak ceiling (tok/s); set only
            when ``snapshots[]`` is non-empty.
        target_tok_per_sec (float | None): Target line (``ceiling × ceiling_ratio_target``).
        ceiling_ratio_target (float): Target ratio of the ceiling (default 0.70).
        ceiling_available (bool): Whether a ceiling could be derived.
        snapshot_top_bottleneck (str): Top bottleneck tooltip for the ceiling line.
        snapshot_within_roofline_pct (float): Latest snapshot's within-roofline percent.
        snapshot_gap_to_roofline_pct (float): Latest snapshot's gap-to-roofline percent.
        trajectory (list[RooflineTrajectoryPoint]): Stepped baseline + KEEP points.
        baseline_tput (float): Baseline throughput (tok/s).
        current_best_tput (float): Current best throughput (tok/s).
        cumulative_gain_pct (float): Cumulative gain percent vs baseline.
        current_best_pct_of_ceiling (float | None): ``tput / ceiling * 100``; None
            when no ceiling.
        current_best_pct_of_target (float | None): ``tput / target * 100``; None
            when no ceiling.
        roofline_failure_streak (int): Consecutive watermark roofline failures.
        snapshots (list[RooflineSnapshot]): Raw snapshot history for drill-downs.
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

    # Headline numbers (also derivable from trajectory[-1] /
    # state.cumulative_gain — surfaced here so the dashboard's "current"
    # callout doesn't need to compute them).
    baseline_tput: float
    current_best_tput: float
    cumulative_gain_pct: float
    current_best_pct_of_ceiling: float | None  # tput/ceiling*100, None when no ceiling
    current_best_pct_of_target: float | None   # tput/target*100, None when no ceiling

    # Audit / staleness
    roofline_failure_streak: int               # consecutive watermark roofline failures
    snapshots: list[RooflineSnapshot]


# ---------------------------------------------------------------------------
# Optimization stack — raw KEEP ledger passthrough
# ---------------------------------------------------------------------------
class OptimizationStackEntry(TypedDict, total=False):
    """One KEEP from ``state.optimization_stack[]`` exposed verbatim.

    Required-shape fields (always present on writers' entries):

    * ``action`` — ``baseline`` / ``params`` / ``backends`` /
      ``explore`` / ``kernel_opt`` / ``integrate`` / ``gemm_tuning``
      / ``framework_pr`` / ``validate_stack``.
    * ``variant_name`` — human-readable label (``vllm_kv_cache_fp8`` /
      kid for kernel_opt / PR ref for framework_pr / etc.).
    * ``candidate_extra_server_args`` — full CLI fragment patched into
      the server launch.
    * ``extra_envs`` — env-var dict patched into the launch.
    * ``tput`` — measured throughput (tok/s/GPU) at this stack depth.
    * ``ts`` — iso UTC promotion timestamp.
    * ``workspace`` — absolute path to the benchmark workspace dir
      (or None for synthetic / kernel-only entries).

    GEMM-tuning-specific fields (optional, populated by the
    Coordinator's ``_promote_gemm_tuning_keep`` path):

    * ``tuned_file`` — absolute path to the produced
      ``a8w8_blockscale_tuned_gemm.csv``.
    * ``final_report_path`` — absolute path to ``final_report.json``.
    * ``source`` — provenance label (e.g. ``kernel_entry_auto``).

    Additional optional fields surface from individual writers:

    * ``gain_pct`` — single-step % gain (kernel_opt promotions).
    * ``kernel_id`` — kid for kernel-owned entries.
    * ``fingerprint`` — content-hash deduplication key for explore.
    * ``provenance`` — ``specialist:<domain>`` / ``default_grid`` /
      ``llm_direct`` / ``legacy:<action>`` for explore winners.
    * ``task_id`` — orchestrator task id (link to specialist_runs etc).
    """
    action: str
    variant_name: str
    candidate_extra_server_args: str
    extra_envs: dict[str, str]
    tput: float | None
    ts: str
    workspace: str | None
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


# ---------------------------------------------------------------------------
# Kernel Roofline — hot-kernel table for the dashboard
# ---------------------------------------------------------------------------
# Mirrors ``<session_dir>/reports/kernel_roofline.json`` produced by the
# kernel-agent's tracelens roofline pipeline. The dashboard renders one row
# per kernel (default sort = ``gpu_pct`` desc); each entry is self-contained
# so the consumer doesn't need to read the original report file.
class KernelRooflineEntry(TypedDict, total=False):
    """One hot-kernel row in the dashboard's kernel roofline table.

    Keys mirror the on-disk shape; collector passes them through
    verbatim (with type coercion) to keep the schema loose-coupled to
    the tracelens output format. Fields documented in
    ``Dashboard-Roofline 对接清单.md`` §1.

    Attributes:
        kernel_id (str): Kernel id (``k001``..``k010``).
        name (str): Kernel name (e.g. ``aiter::ck_moe_stage1``).
        source_file (str): Absolute source path; "" when unknown.
        kernel_category (str): Category (``MoE`` / ``LayerNorm`` / ``unknown``).
        bound_type (str): Bound type (``memory-bound`` / ``compute-bound``).
        arithmetic_intensity (float): FLOPs per byte.
        flops_per_byte (float): Arithmetic intensity expressed as FLOPs/byte.
        efficiency_percent (float): Kernel-self efficiency (0..100).
        gpu_pct (float): Share of overall GPU time (0..100).
        call_count (int): Number of kernel invocations.
        duration_us (float): Total kernel duration in microseconds.
        reusable_native_kernel (bool): True if GEAK can swap in a custom kernel.
    """
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
    """Top-level ``kernel_roofline`` section.

    Loaded from ``<session_dir>/reports/kernel_roofline.json`` when
    present; left empty (``{}``) on missing / malformed file (collector
    appends a warning instead of raising).

    Attributes:
        schema_version (int): Tracelens output schema version (currently 1).
        source (str): Provenance label (e.g. ``tracelens_analysis``).
        analysis_md_path (str): Absolute path to the human-readable analysis.
        kernel_candidates_path (str): Absolute path to ``kernel_candidates.json``.
        trace_input (str): Absolute path to the trace dir.
        trace_input_type (str): Trace input type (``capture_dir`` / ``trace_file`` / ...).
        kernels (list[KernelRooflineEntry]): Hot-kernel rows for the table.
    """
    schema_version: int                    # tracelens output schema (currently 1)
    source: str                            # provenance label, e.g. ``tracelens_analysis``
    analysis_md_path: str                  # absolute path to the human-readable analysis
    kernel_candidates_path: str            # absolute path to kernel_candidates.json
    trace_input: str                       # absolute path to the trace dir
    trace_input_type: str                  # ``capture_dir`` / ``trace_file`` / ...
    kernels: list[KernelRooflineEntry]


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
    # ``phase_timeline`` retained for v1-reader
    # compat as the flat per-action timeline (``action_timeline`` is
    # the canonical v2 name; see below). ``phase_segments`` carries
    # the phase-boundary view (M2).
    phase_timeline: list[PhaseEvent]
    phase_segments: list[PhaseSegment]
    # v0.8 §3.12 §4.2 / §5 — top-level action_timeline alias used by
    # v0.6 readers that still expect a flat per-action list.
    action_timeline: list[PhaseEvent]
    capability_summary: CapabilitySummary
    geak_invocations: list[Invocation]
    oob_invocations: list[Invocation]
    kernel_lifecycle: KernelLifecycle
    # ``param_search`` is the v1-reader compat alias
    # for the merged ``explore_search`` ledger; both fields carry
    # identical data so an old reader doesn't see a missing key.
    param_search: ParamSearch
    explore_search: ParamSearch
    sweep: Sweep
    critic_robustness: CriticRobustness
    telemetry: Telemetry
    attribution: Attribution
    kb_provenance: KBProvenance      # Cortex KB audit
    # specialist sub-agent dispatch records.
    specialist_runs: list[SpecialistRound]
    # Raw KEEP ledger passthrough. Mirrors
    # ``state.optimization_stack[]`` verbatim so downstream tooling
    # (dashboard chart, GEMM-tuning visualization, audit trails) can
    # read full per-entry evidence — including ``tuned_file`` /
    # ``final_report_path`` for gemm_tuning entries and ``workspace``
    # for any KEEP — without round-tripping back to state.json. The
    # other "stack-derived" sections (``final.action_path``,
    # ``attribution.gain_per_stack_entry``,
    # ``roofline_progress.trajectory``) intentionally summarise this
    # list for their respective consumers and don't carry the full
    # per-entry metadata.
    optimization_stack: list["OptimizationStackEntry"]
    # Hot-kernel table for the dashboard (Dashboard-Roofline 对接清单
    # §1). Mirrors ``<sd>/reports/kernel_roofline.json`` so consumers
    # don't have to walk the kernel-agent output tree themselves.
    kernel_roofline: KernelRoofline
    # Per-snapshot roofline comparison list (one entry per
    # ``state.roofline_snapshots`` history pass). Drives the markdown-
    # report ``## Roofline`` section. Each entry has ``source_path /
    # mode / baseline / latest / delta``.
    roofline: list[dict[str, Any]]
    # Optimization-progress curve for the dashboard
    # (Dashboard-Roofline 对接清单 §2). Carries the trajectory
    # (baseline + KEEP points), the ceiling/target reference lines,
    # and the headline current-best numbers. Renamed from ``roofline``
    # to ``roofline_progress`` to coexist with the existing list-
    # shaped ``roofline`` consumed by the markdown renderer.
    roofline_progress: RooflineProgress

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
    "KBPendingEdge",
    "KBPointCreated",
    "KBProvenance",
    "KBQueueStats",
    "LaneTimelineEntry",
    "KernelLifecycle",
    "KernelMetadata",
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
    "Workload",
    "WorkloadObjective",
]
