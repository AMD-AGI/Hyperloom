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
#: row, ``critic_robustness.kb_writes_summary`` sub-block, the
#: top-level ``action_timeline`` / ``explore_search`` v1-reader
#: aliases, and (additively) the ``kernel_optimization_summary`` /
#: ``conc_sweep_summary`` sections mirrored from
#: ``reports/kernel_optimization_summary.json`` and
#: ``reports/conc_sweep_summary.json`` (PR #399 lishuoshuo). Inv-12.1
#: guarantees a v0.6 / v1 reader can still consume the file because v2
#: only *adds* fields — the version string does not bump for additive
#: sections.
SCHEMA_VERSION = "hyperloom.session_breakdown.v2"


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


# ---------------------------------------------------------------------------
# §2 Workload configuration
# ---------------------------------------------------------------------------
class WorkloadObjective(TypedDict, total=False):
    kind: str                     # gain_pct / tput / baseline / time_only
    value: Any                    # float or str (target_baseline_dir) or None


class Workload(TypedDict, total=False):
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
    extra_server_args: str
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
    """One row from explore_search.{tested,accepted,rejected}."""
    name: str
    fingerprint: str
    extra_server_args: str
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
    # counts of KB writes proxied through the
    # critic agent's ``commit-review`` protocol (Coordinator
    # actually performs the writes; the critic only authors them).
    kb_writes_summary: "CriticKBWritesSummary"


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


class LaneTimelineEntry(TypedDict, total=False):
    """One row of the legacy M6 lane occupancy summary.

    Surfaces resource_lock state (per-lane capacity vs. live holders +
    lifetime expired-lease count) into the breakdown's ``telemetry``
    section so cross-cluster dashboards can chart lane usage alongside
    GPU power / temperature.
    """
    lane: str
    capacity: int
    live_holders: int
    lease_expired_count: int


class Telemetry(TypedDict, total=False):
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
    """One KEEP/validation event with its incremental contribution."""
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
    """
    total_gain_pct: float
    by_domain: dict[str, float]


class PhaseBreakdownKernel(TypedDict, total=False):
    """v0.8 M7 — kernel-phase gain split by ``kernel_id`` (KB_design
    §3.12 §4.6)."""
    total_gain_pct: float
    by_kernel_id: dict[str, float]


class PhaseBreakdownFrameworkPr(TypedDict, total=False):
    """FRAMEWORK_PR phase gain split by adopted PR
    reference. ``by_pr`` keys are the entry's ``variant_name`` (PR
    label, typically ``PR:<repo>#<num>`` or ``PR:<num>``); empty
    string falls back to ``"?"``."""
    total_gain_pct: float
    by_pr: dict[str, float]


class PhaseBreakdownGemmTuning(TypedDict, total=False):
    """KERNEL-entry FP8 GEMM tuning gain split by tuned-CSV path.
    ``by_tuned_file`` keys on the entry's ``tuned_file`` (absolute
    path to ``a8w8_blockscale_tuned_gemm.csv``); fallbacks: entry's
    ``variant_name`` then ``"?"`` so the key is always a string."""
    total_gain_pct: float
    by_tuned_file: dict[str, float]


class PhaseBreakdown(TypedDict, total=False):
    """v0.8 M7 per-phase gain attribution (KB_design §3.13 M7 §6)."""
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
    proposal_msg_id: str
    edge_id: str
    action: str
    ts: str


class KBQueueStats(TypedDict, total=False):
    pending_lines: int             # current depth of .kb_pending.ndjson
    flushed_bookmarks: int         # rows in .kb_flushed.ndjson (drain bookmarks)
    dead_letter_lines: int         # rows in .kb_dead_letter.ndjson


class KBCommitSummary(TypedDict, total=False):
    status: str                    # committed / commit_failed / skip_disabled / ...
    promoted_edges: list[str]
    derived_summary_id: str


class KBPointCreated(TypedDict, total=False):
    """One row in ``kb_provenance.points_created``.

    ``kind`` ∈ {workload_node / issue_node / optimization_node /
    pr_node / attempt_node / ...}. ``pr_node`` rows are the M4
    contribution; everything else came from M1/M3 path.
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
    ``failed`` / ``skipped`` with the per-status fields populated."""
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
    """
    task_id: str
    domain: str
    path: str
    body: str   # only set when CLI flag enabled


class SpecialistRound(TypedDict, total=False):
    """One element of ``specialist_runs``."""
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
    """Summary of critic-agent ``commit-review`` outputs (Coordinator
    proxies these into ``kb_provenance``)."""
    total: int
    by_verdict: dict[str, int]   # KEEP / REVERT / NEEDS_INFO / ...


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
    """
    schema_version: int                    # tracelens output schema (currently 1)
    source: str                            # provenance label, e.g. ``tracelens_analysis``
    analysis_md_path: str                  # absolute path to the human-readable analysis
    kernel_candidates_path: str            # absolute path to kernel_candidates.json
    trace_input: str                       # absolute path to the trace dir
    trace_input_type: str                  # ``capture_dir`` / ``trace_file`` / ...
    kernels: list[KernelRooflineEntry]


# ---------------------------------------------------------------------------
# Kernel Optimization Summary (Breakdown 面板对接文档 A1; PR #399 lishuoshuo)
# ---------------------------------------------------------------------------
# Mirror of ``<session_dir>/reports/kernel_optimization_summary.json``
# (produced deterministically by the ``report`` action via
# ``orchestrator/kernel_attempt_summary.build_kernel_optimization_summary``).
# Answers "did each top kernel get optimized by the kernel-agent, and
# why did it fail" without the dashboard walking the kernel-agent tree.
# The collector mirrors the report verbatim (light top-level shape
# guards only) so new producer fields ride through without a schema
# change; the deeply-nested ``by_kernel[]`` rows therefore stay loose
# (``dict``) and are documented in roofline优化对接文档.md §A1.4.
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


# ---------------------------------------------------------------------------
# Conc Sweep Summary (Breakdown 面板对接文档 A2; PR #399 lishuoshuo)
# ---------------------------------------------------------------------------
# Mirror of ``<session_dir>/reports/conc_sweep_summary.json`` (produced
# by the ``conc_sweep`` action during SWEEP). Extends the single-CONC
# headline gain into a baseline-vs-current_best curve across a CONC
# ladder. Absent when conc_sweep never ran (Block hidden). When
# ``status="skipped"`` the producer omits the baseline/optimized/
# comparison/summary blocks — read ``status`` before those keys.
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


class SessionBreakdown(TypedDict, total=False):
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
    # Kernel-agent attempt outcome summary (Breakdown 面板对接文档 §A1).
    # Mirrors ``<sd>/reports/kernel_optimization_summary.json``. Empty
    # dict when the report is absent (session predates PR #399 or the
    # ``report`` action never ran) — the dashboard hides Block 1.
    kernel_optimization_summary: KernelOptimizationSummary
    # Post-optimization concurrency sweep (Breakdown 面板对接文档 §A2).
    # Mirrors ``<sd>/reports/conc_sweep_summary.json``. Empty dict when
    # conc_sweep never ran — the dashboard hides Block 2.
    conc_sweep_summary: ConcSweepSummary
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
    "ConcSweepSummary",
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
    "Workload",
    "WorkloadObjective",
]
