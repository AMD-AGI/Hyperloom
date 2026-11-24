# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema (TypedDict shape) for ``session_breakdown.json``."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


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
    ep: int | None
    compute_partition: dict[str, Any]
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


class V6GradedAxes(TypedDict, total=False):
    """The axes an AgentX measurement is ranked and reported on.

    ``e2e_norm_intvty_p50`` is the objective. Its guards are the tail and ``total_throughput``
    (``hyperloom.common.agentx_objective``). ``output_tput_per_gpu`` is reported and enters no verdict. The latency
    percentiles are the detail view: reported, never graded.

    ``duration_seconds`` and ``request_error_rate`` are the comparability inputs. A pair is graded only when both
    replayed a window of the same length and the candidate's error rate exceeds its anchor's by no more than
    ``ERROR_RATE_SLACK_PCT``, so a verdict published without them cannot be re-derived from this record.

    Every axis is present on every measurement, ``None`` where nothing measured
    it: absent would be indistinguishable from an axis the framework failed to
    report, and zero reads as "measured, and it was zero". A synthetic run
    carries nulls for the AgentX-only axes; ``output_tput_per_gpu`` is set whenever the TP degree is known.
    """

    e2e_norm_intvty_p90: float | None
    e2e_norm_intvty_p50: float | None
    total_throughput: float | None
    output_tput_per_gpu: float | None
    input_throughput: float | None
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    tpot_p50_ms: float | None
    tpot_p90_ms: float | None
    duration_seconds: float | None
    request_error_rate: float | None


class V6GradingTputGuard(TypedDict, total=False):
    """The throughput constraint riding along with the interactivity objective.

    ``noise_pct`` is ``None`` on a session seeded before the band was recorded:
    the band that session applied is unknown, and today's environment is not
    evidence of it.
    """

    enabled: bool
    noise_pct: float | None


class V6Grading(TypedDict, total=False):
    """The grading mode this session was started in.

    ``objective`` is a start-time mode marker (``e2e_norm_intvty_p90`` on an interactivity session), not the
    promotion axis: that is ``outcome.validation.graded_on`` / ``outcome.final.graded_on``, absent when nothing
    was adopted and the adoption axis when nothing was validated. ``prompt_dataset`` is present, and true, on a
    session whose fixed rounds sent a prompt dataset: ``task_config.isl`` and ``osl`` do not describe its requests.
    """

    benchmark_mode: str
    objective: str
    tput_guard: V6GradingTputGuard
    prompt_dataset: bool


class V6WorkflowMetadata(TypedDict, total=False):
    """Identity and frozen action surfaces for deterministic workflow replay."""

    workflow_contract_version: str
    contract_digest: str
    run_flags: dict[str, Any]
    phase_actions: dict[str, list[str]]
    llm_proposable_actions: dict[str, list[str]]
    coordinator_internal_actions: list[str]
    coordinator_reserved_actions: dict[str, list[str]]
    kernel_lane_task_kinds: list[str]


class V6Metadata(TypedDict, total=False):
    """V6 task identity, configuration, versions, and trace entrypoint."""

    exported_at_utc: str
    versions: V6MetadataVersions
    session: V6MetadataSession
    task_config: V6TaskConfig
    grading: V6Grading
    langfuse: V6MetadataLangfuse
    workflow: V6WorkflowMetadata
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

    #: The axis every percentage here shares, read off the row that produced
    #: the settled figure. The reconciliation has to be single-axis: an
    #: attributed figure on one axis against an unattributed figure on another
    #: makes the gap meaningless.
    graded_on: str | None
    #: The settled measurement's own axes, on the same row as the gain they
    #: produced -- a revalidation moves the cumulative figure without
    #: re-promoting the recipe, so ``current_best`` can be a later measurement.
    perf: V6GradedAxes
    #: ``None`` when no adoption could be measured.
    attributed_gain_pct: float | None
    #: ``attributed_gain_pct`` without adoptions kept for a non-gain reason;
    #: ``None`` on a ledger recorded before the field existed.
    attributed_gain_pct_excluding_fixes: float | None
    #: Per ``keep_reason``: ``count``, ``total_gain_pct``, ``unmeasured``.
    by_keep_reason: dict[str, Any]
    #: ``None`` when nothing was measured or no chain total exists.
    unattributed_gain_pct: float | None
    chain_total_gain_pct: float | None
    validated_total_gain_pct: float | None
    reconciliation_gap_pct: float | None
    #: How many adoptions the ledger holds, across every source.
    adoption_count: int
    #: The stack length the settled figure was measured on, and the author-time
    #: stamp and basis of that measurement. ``None`` when nothing was validated.
    validated_at_stack_len: int | None
    validated_ts: str | None
    measurement_basis: str | None
    #: The latency the settled measurement reported, and which field of the
    #: benchmark report the pair came from, as the measurement labelled it.
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    ttft_e2el_source: str | None
    #: The flags the measured server was launched with, and the run directory
    #: the measurement was taken in.
    server_launch_flags: str | None
    workspace: str | None
    #: Adoptions landed after the settled figure was measured, so it describes
    #: a shorter stack than the one that shipped.
    stack_changed_after_validation: bool
    attribution: V6OutcomeAttribution
    #: ``unmeasured`` and ``chain_breaks``; see :mod:`.recorder.stack_event`.
    guards: dict[str, int]
    #: One entry per finding the ledger's own figures support. Empty is the
    #: meaningful case: the ledger reconciles.
    notes: list[str]


class V6OutcomeVariants(TypedDict, total=False):
    """The framework phase's bench runs, counted by how each ended.

    Read off the ``attempts`` of every ``framework_agent`` event, configuration
    variants and source patches alike; an attempt that never reached the bench
    is not counted, and kernel candidates are the kernel lane's own funnel.
    """

    #: ``kept + reverted + failed``.
    bench_runs: int
    #: Distinct changes among the runs, by arm and fingerprint: a change benched
    #: again on a later stack is one change and two runs.
    distinct: int
    #: Runs adopted into the stack.
    kept: int
    #: Runs measured and not adopted.
    reverted: int
    #: Runs that reached the bench and produced no measurement.
    failed: int
    #: ``kept`` per bench run; ``None`` when nothing was benched.
    keep_rate_pct: float | None


class V6Outcome(TypedDict, total=False):
    """V6 session result projection for downstream consumers."""

    stop_reason: str
    status: Literal["completed", "failed", "aborted"]
    stage_reached: str
    baseline: dict[str, Any]
    anchoring_eval: dict[str, Any] | None
    final: dict[str, Any]
    validation: V6OutcomeValidation
    variants: V6OutcomeVariants


class V6TimelineEvent(TypedDict, total=False):
    """One ordered V6 business-stage event; CLOSE is intentionally excluded."""

    type: str
    kind: str
    status: str
    process_status: str
    business_outcome: str | None
    failure: dict[str, Any] | None
    blocked_by: str | None
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
    the run it came from."""

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

    Only T0's own reads are here. ``kb_amend_recipe`` consults the same store
    through the same audit hook in the middle of the session; those reads are
    real but they are not the anchor's, and this block would misreport the
    lookup if it counted them."""

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
    earlier and reports ``seed_only``, which is normal and not a fault."""

    request: dict[str, Any]
    match_status: str
    matched: V6WarmStartMatched | None
    reads: V6WarmStartReads | None
    failure: V6Failure | None


class V6WarmReplayExt(TypedDict, total=False):
    """``timeline[type=warm_replay].ext`` — the replay arc from request to verdict.

    One event per replay attempt. ``blocked_by`` names the first gate that did
    not pass; a successful arc leaves it ``None``. ``failure`` is the canonical
    crash block when the arc raised rather than settling a verdict."""

    request: dict[str, Any]
    measurement: dict[str, Any]
    gates: list[dict[str, Any]]
    blocked_by: str | None
    applied: dict[str, Any] | None
    verdict: dict[str, Any]
    promotion: dict[str, Any] | None
    rollback: dict[str, Any] | None
    skip: dict[str, Any] | None
    failure: V6Failure | None
    duration_sec: float | None


class V6BaselineExt(TypedDict, total=False):
    """``timeline[type=baseline].ext`` — one entry per measurement the event owns.

    ``anchoring_eval`` is present only when one of those actions established
    the session's quality reference."""

    actions: list[dict[str, Any]]
    anchoring_eval: dict[str, Any]


class V6RooflineExt(TypedDict, total=False):
    """``timeline[type=roofline].ext`` — one entry per analysis the event owns."""

    actions: list[dict[str, Any]]


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
    and the throughput do not already answer."""

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
    failure: dict[str, Any]


class V6RooflineKernel(TypedDict, total=False):
    """One kernel in a roofline action's own per-kernel table.

    Recorded by reading the sidecar the analyzer wrote, at the moment the action
    that produced it settles. The table is not truncated by GPU share, because
    the kernel worth optimizing is often a cheap one running at low efficiency,
    which is exactly what a top-N-by-cost cut removes.

    ``roofline_source`` says whether a row's bound came from a model or is a placeholder."""

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
    roofline_name: str | None
    roofline_source: str
    suggestion: str
    recommended_actions: list[str]
    reusable_native_kernel: bool


class V6RooflineKernelTable(TypedDict, total=False):
    """A roofline action's per-kernel table with the provenance of its source."""

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

    Carried on the event rather than only as ``snapshot_id``, because a join
    against the capped session-state history fails once later runs evict the
    entry."""

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
    """One measured step on the session's throughput curve."""

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

    The snapshot history is deliberately absent: each roofline event carries
    its own snapshot in full, and a second copy here would be free to disagree.
    ``latest_snapshot_id`` is the join back to the event that set the ceiling."""

    ceiling_tok_per_sec: float | None
    target_tok_per_sec: float | None
    ceiling_ratio_target: float
    ceiling_available: bool
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
    session total is only final here."""

    failure_streak: int
    total_failures: int
    arg_error_streak: int


class V6GeakCandidate(TypedDict, total=False):
    """Where the GEAK candidate stood when the session wound down.

    Under ``close`` for the same reason the baseline tally is. A candidate's
    attempts belong to the kernel event that ran them, and that event closes
    when the phase is left -- but a slot still awaiting a rebench, or one the
    close drain cancelled, is settled after every kernel event has closed.

    The self-reported numbers travel with the verdict so a reader can say what
    was dropped rather than only that something was. They are the optimizer's
    own account and were never re-measured; a candidate that survived to a
    headline has its measured numbers on the stack instead."""

    revalidation_pending: bool
    status: str
    revalidation_error: str | None
    revalidation_error_class: str | None
    self_reported_gain_pct: float | None
    self_reported_tput: float | None
    self_reported_basis: str


class V6Failure(TypedDict, total=False):
    """The canonical failure row, shared by every event that records one.

    One shape from one producer, so a reader that can parse one event's
    failure can parse them all. ``stage`` names the step it died at -- a
    profiling substep, a baseline round, a phase entry -- rather than the phase
    it died in. Absent on anything that did not fail, which is not the same as
    a block present and empty: an event stopped early by the session's end
    reports that beside the rest of how it ran, because nothing about it
    failed.

    Some events carry their own additions beside these keys, the baseline's
    ``returncode`` and ``stderr_log_path`` among them."""

    stage: str
    error_class: str
    message: str


class CriticIteration(TypedDict, total=False):
    """One critic-agent review pass over a proposed change.

    ``verdict`` reads the iteration's rulings as one line (``2 approve, 1
    reject``) and ``verdict_counts`` carries the same distribution for callers
    that need to count. A review turn that produced no ruling leaves both
    empty."""

    iteration_id: str
    iter: int
    ts: str
    topic: str
    verdict: str
    verdict_counts: dict[str, int]
    summary: str
    request_path: str
    judge_bundle_path: str
    emit_path: str
    review_path: str
    phase: str
    macro_cycle: int
    framework_reviews: list[dict[str, Any]]


class V6Critic(TypedDict, total=False):
    """The critic agent's own run, outside the business timeline.

    The per-proposal verdicts stay with the proposals they judge; this key
    carries the agent's completed review turns -- what it reviewed and what
    it concluded each time."""

    iterations: list[CriticIteration]


class V6FinalRecipe(TypedDict, total=False):
    """The configuration the session ended on, as the close-out settled it."""

    throughput: float | None
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    action_path: list[str]
    extra_server_args: str
    extra_envs: dict[str, str]


class V6Close(TypedDict, total=False):
    """V6 session finalization result exposed outside the business timeline.

    ``status`` is recorded by the CLOSE sequencer, not derived from ``steps``.
    ``running`` means no verdict was ever recorded, so the process died partway
    through its own close-out; ``degraded`` means the sequence finished with at
    least one step reporting a failure. The two are separate words so a healthy
    session stays distinguishable from a damaged one.
    """

    status: Literal["running", "succeeded", "failed", "degraded"]
    start_time: str
    end_time: str
    close_sequence_done: bool
    steps: list[dict[str, Any]]
    artifacts: dict[str, Any]
    # Absent when the session never attempted to publish its Recipe.
    kb_write_back: V6KBWriteBackExt
    # Absent when the close-out never got far enough to snapshot it.
    roofline_progress: V6RooflineProgress
    # Same rule: absent when the close-out never reached the snapshot.
    baseline_progress: V6BaselineProgress
    # Absent when the close drain never ran, which a killed session is.
    geak_candidate: V6GeakCandidate
    # The recipe that shipped, which ``outcome.final`` is projected from.
    # Absent when the close-out never reached the snapshot.
    final_recipe: V6FinalRecipe


class SessionBreakdown(TypedDict, total=False):
    """Top-level wire shape of ``session_breakdown.json``.

    The complete contract between the producer (``inference_optimizer``) and
    downstream consumers. Every section is either recorded at author time or
    projected from what was recorded: the export re-derives nothing.

    How the export itself went is reported once, on ``metadata.warnings``."""

    schema_version: str
    exported_at_utc: str
    exporter_version: str

    metadata: V6Metadata
    outcome: V6Outcome
    timeline: list[V6TimelineEvent]
    close: V6Close
    critic: V6Critic
    # The concurrency sweep's ``reports/conc_sweep_summary.json``, verbatim. Absent until a sweep runs, which is after
    # CLOSE (``--enable-conc-sweep`` or ``hyperloom conc-sweep``); the sweep adds it each time it rewrites its report.
    conc_sweep_summary: dict[str, Any]


__all__ = [
    "CriticIteration",
    "SessionBreakdown",
    "V6BaselineExt",
    "V6BaselineProgress",
    "V6Close",
    "V6Critic",
    "V6Failure",
    "V6GeakCandidate",
    "V6GradedAxes",
    "V6Grading",
    "V6GradingTputGuard",
    "V6KBWriteBackExt",
    "V6Metadata",
    "V6MetadataLangfuse",
    "V6MetadataRecovery",
    "V6MetadataSession",
    "V6MetadataVersions",
    "V6ModelArchitecture",
    "V6Outcome",
    "V6OutcomeAttribution",
    "V6OutcomeAttributionBySource",
    "V6OutcomeGainBucket",
    "V6OutcomeKernelAttribution",
    "V6OutcomeValidation",
    "V6OutcomeVariants",
    "V6RooflineEventSnapshot",
    "V6RooflineExt",
    "V6RooflineKernel",
    "V6RooflineKernelTable",
    "V6RooflineProgress",
    "V6RooflineTrajectoryPoint",
    "V6TaskConfig",
    "V6TimelineEvent",
    "V6ToolVersion",
    "V6WarmReplayExt",
    "V6WarmStartExt",
    "V6WarmStartMatched",
    "V6WarmStartReads",
]
