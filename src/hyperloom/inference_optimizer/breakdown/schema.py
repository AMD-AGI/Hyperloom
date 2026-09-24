# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema (TypedDict shape) for ``session_breakdown.json``."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from ..session.sbd_v6 import SCHEMA_VERSION_V6


#: Current breakdown schema version. V6 stamps the document once the timeline
#: is recorded by the actions themselves rather than projected out of their
#: artefacts afterwards, which is what makes an event's start time its real one.
#: ``enablement`` gained its ledger-sourced round fields inside this version:
#: they add a section to the block rather than reshape the document.
SCHEMA_VERSION = SCHEMA_VERSION_V6


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


class V6GradedAxes(TypedDict, total=False):
    """The canonical AgentX metrics published with a measurement.

    Every axis is present on every measurement, ``None`` where nothing measured
    it: absent would be indistinguishable from an axis the framework failed to
    report, and zero reads as "measured, and it was zero". A synthetic run
    carries seven nulls.
    """

    e2e_intvty_p50: float | None
    e2e_intvty_p90: float | None
    output_tput_per_gpu: float | None
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    tpot_p50_ms: float | None
    tpot_p90_ms: float | None


class V6GradingTputGuard(TypedDict, total=False):
    """The throughput constraint riding along with the interactivity objective.

    ``noise_pct`` is ``None`` on a session seeded before the band was recorded:
    the band that session applied is unknown, and today's environment is not
    evidence of it.
    """

    enabled: bool
    noise_pct: float | None


class V6Grading(TypedDict, total=False):
    """The axis this session was configured to grade on.

    The session-level setting and only that. What a promotion was actually
    decided on is ``outcome.validation.graded_on``, read off the promotion
    itself. On a session that promoted anything the two agree, because a
    comparison that cannot supply the configured axis pair fails rather than
    settling for another axis -- no promotion is ever graded off-objective.
    Neither field resolves the other even so: a session can be configured for
    an axis and promote nothing on it.
    """

    benchmark_mode: str
    objective: str
    tput_guard: V6GradingTputGuard
    policy: dict[str, Any]


class V6Metadata(TypedDict, total=False):
    """V6 task identity, configuration, versions, and trace entrypoint."""

    exported_at_utc: str
    versions: V6MetadataVersions
    session: V6MetadataSession
    task_config: V6TaskConfig
    grading: V6Grading
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

    #: The axis every percentage here shares, read off the row that produced
    #: the settled figure. The reconciliation has to be single-axis: an
    #: attributed figure on one axis against an unattributed figure on another
    #: makes the gap meaningless.
    graded_on: str | None
    #: The settled measurement's own axes, on the same row as the gain they
    #: produced -- a revalidation moves the cumulative figure without
    #: re-promoting the recipe, so ``current_best`` can be a later measurement.
    perf: V6GradedAxes
    agentx_policy: dict[str, Any]
    attributed_gain_pct: float
    unattributed_gain_pct: float
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


class V6Outcome(TypedDict, total=False):
    """V6 session result projection for downstream consumers."""

    stop_reason: str
    status: Literal["completed", "failed", "aborted"]
    stage_reached: str
    baseline: dict[str, Any]
    anchoring_eval: dict[str, Any] | None
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

    Only T0's own reads are here. ``_kb_amend_recipe`` consults the same store
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
    answer to "was this number measured"."""

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

    The event used to record only ``snapshot_id``, which made the conclusion
    reachable solely by joining against a capped session-state history that
    later runs evict entries from."""

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

    ``ceiling_kind`` discriminates two domains that cannot share fields.
    Token-decoding models are bounded in throughput and report the ``tok/sec``
    fields; scriptable/diffusion models decode no tokens and are bounded in
    latency, reporting the ``ms`` fields instead. A reader that ignores
    ``ceiling_kind`` will read the absent domain's nulls as a failed analysis.

    The snapshot history is deliberately absent: each roofline event carries
    its own snapshot in full, and a second copy here would be free to disagree.
    ``latest_snapshot_id`` is the join back to the event that set the ceiling."""

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


class V6ConcSweepRequest(TypedDict, total=False):
    """What the dispatch asked for, before the sweep resolved anything.

    Kept apart from ``plan`` so a ladder the sweep chose is never mistaken for
    one it was handed: a null ``requested_concs`` is the operator declining to
    pick, not an empty ladder."""

    task_id: str
    task_kind: str
    reason: str
    requested_concs: list[int]
    requested_variant_timeout_sec: int | None
    requested_total_budget_sec: int | None


class V6ConcSweepWorkload(TypedDict, total=False):
    """The shape the ladder is swept over.

    ``benchmark_mode`` names the axis pair the points are drawn on, so a reader
    never infers it from whether a latency field happens to be null."""

    session_id: str
    isl: int | None
    osl: int | None
    tp: int | None
    benchmark_mode: str


class V6ConcSweepInputAnchor(TypedDict, total=False):
    """The optimized configuration the sweep was asked to compare.

    The optimized arm is whatever the session's current best was when the sweep
    started, which is a moving target: a sweep two cycles later compares a
    different configuration under the same event type. ``base_action`` is the
    action kind that promoted it."""

    base_variant_id: str | None
    base_action: str | None
    input_throughput_tok_s_per_gpu: float | None
    anchor_tput: float | None
    baseline_tput: float | None
    extra_server_args: str
    extra_envs: dict[str, str]


class V6ConcSweepPlan(TypedDict, total=False):
    """The ladder the sweep resolved and the order it will run it in.

    ``arms_order`` matters to a reader of the budget: the optimized arm runs
    first as the more informative of the two, so a budget that runs out takes
    the baseline arm with it. ``concs_ordered`` is descending because a
    single-server arm boots at the most demanding rung and reuses down."""

    grid_source: str | None
    concs_requested: list[int]
    concs_ordered: list[int]
    num_prompts_factor: int | None
    variant_timeout_sec: int | None
    arms_order: list[str]


class V6ConcSweepBudget(TypedDict, total=False):
    """The budget the ladder was admitted under.

    Both totals are kept because they disagree: the sweep raises its own
    default when that default cannot fund even one rung at the cap the grid
    runner will actually grant. ``rung_cost_sec`` is that granted cap rather
    than the declared timeout, since pricing at the smaller admits a rung the
    budget cannot pay for. ``deadline`` is a wall-clock epoch."""

    declared_total_sec: int | None
    granted_total_sec: int | None
    rung_cost_sec: float | None
    raised: bool
    gate_active: bool
    deadline: float | None
    session_soft_deadline_sec: float | None


class V6ConcSweepEnvironment(TypedDict, total=False):
    """What the sweep resolved to run against.

    ``sweep_task_id`` is the sweep's own minted id, distinct from the dispatched
    task id: it names the ``runs/conc_sweep/`` workspace the rung artifacts live
    under, and nothing else ever wrote it down."""

    sweep_task_id: str
    workspace: str
    model_path: str
    gpu_type: str
    base_config_path: str


class V6ConcSweepArtifacts(TypedDict, total=False):
    """Where the sweep's own report was written."""

    report_json_path: str
    report_csv_path: str


class V6ConcSweepResult(TypedDict, total=False):
    """The roll-up over the pairs, and the sweep's own word on how it ended.

    ``status`` is the sweep's verdict rather than the event's: the event
    degrades a budget-truncated success, and keeping both means the two cannot
    be confused. ``declined`` separates a sweep that refused before running
    anything from ``was_skipped``, which a ladder that ran and produced no
    usable pair also sets. ``best_conc`` is the best rung on the objective
    alone; ``best_conc_guard_holds`` says whether the session's KEEP rule would
    also have accepted it, and ``guard_axis`` names the axis that verdict is
    about. Both are empty off the interactivity objective, where there is no
    second axis to hold."""

    status: str
    metric: str
    guard_axis: str
    best_conc: int | None
    best_speedup: float | None
    best_conc_guard_holds: bool | None
    successful_pairs: int | None
    failed_pairs: int | None
    median_speedup: float | None
    mean_speedup: float | None
    skip_reason: str
    was_skipped: bool
    budget_exhausted: bool | None
    declined: bool


class V6ConcSweepRuntime(TypedDict, total=False):
    """How the run went, as opposed to what it found.

    ``stop_reason`` is the session's, set when its end cut the sweep short; a
    sweep stopped that way measured what it got to and did not fail, which is
    why it is here rather than in ``failure``. ``elapsed_sec`` is the sweep's
    own accounting and ``duration_sec`` the event's."""

    elapsed_sec: float | None
    budget_remaining_sec: float | None
    budget_skip_reason: str
    stop_reason: str
    workspace: str
    duration_sec: float | None


class V6ConcSweepCeilingModel(TypedDict, total=False):
    """The model dimensions the ceiling was computed from.

    Recorded because the ceiling is only auditable against them: the same
    checkpoint served at a different weight precision yields a different
    ceiling from the same curve."""

    weight_bytes: int
    active_weight_bytes: int
    num_experts: int
    experts_per_tok: int
    expert_weight_bytes: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    weight_dtype_bytes: float


class V6ConcSweepCeilingRow(TypedDict, total=False):
    """One concurrency's theoretical ceiling, and how close each arm came.

    ``bound_kind`` says which of the two limits ``t_peak_tok_s`` came from, so
    a rung far below peak can be read as the memory wall or as headroom the
    configuration is not using."""

    conc: int
    t_mem_tok_s: float
    t_cmp_tok_s: float
    t_peak_tok_s: float
    bound_kind: str
    mbu_baseline_pct: float | None
    mbu_optimized_pct: float | None


class V6ConcSweepCeiling(TypedDict, total=False):
    """The decode roofline the curve is read against, one row per rung."""

    schema_version: int
    source: str
    gpu_type: str
    precision: str
    tp: int
    isl: int
    osl: int
    model_meta: V6ConcSweepCeilingModel
    rows: list[V6ConcSweepCeilingRow]


class V6ConcSweepLifecycle(TypedDict, total=False):
    """Whether the framework can hold a server across rungs, and on what port.

    ``eligible`` false is what forces the restart-per-rung path, so a curve
    that cost a server start per point is explained rather than merely slow."""

    eligible: bool | None
    reason: str
    port: int | None
    framework: str


class V6ConcSweepRefused(TypedDict, total=False):
    """The gate that refused an arm before it built anything."""

    reason: str
    remaining_sec: float | None


class V6ConcSweepRung(TypedDict, total=False):
    """One planned rung and the load it carries.

    A rung's ``num_prompts`` is derived from its CONC and never written down
    elsewhere, so a run cannot be reproduced from the report alone."""

    name: str
    conc: int | None
    num_prompts: int | None


class V6ConcSweepBootAttempt(TypedDict, total=False):
    """One rung the boot-retry-descend loop tried.

    ``committed`` false is a concurrency the server would not come up at --
    the capacity finding the sweep produces for free and the report discards."""

    conc: int | None
    committed: bool
    status: str
    error_class: str
    error: str | None
    start_time: str
    wall_duration_sec: float | None


class V6ConcSweepBoot(TypedDict, total=False):
    """How the boot-retry-descend loop resolved.

    ``attempted_concs`` is in the order tried. Absent on an arm refused before
    it built anything."""

    succeeded: bool
    booted_conc: int | None
    attempted_concs: list[int]
    failed_concs: list[int]
    attempts: list[V6ConcSweepBootAttempt]


class V6ConcSweepPoint(TypedDict, total=False):
    """One rung of one arm's concurrency curve, as the sweep recorded it.

    The measurement half is the same flattening the sweep's own report writes,
    so the recorded curve and the written one cannot differ. The process half
    -- everything from ``stage`` down -- is what the report has no place for:
    a rung that produced no throughput number is otherwise indistinguishable
    from one the budget refused, one the server would not boot at, and one the
    benchmark simply failed."""

    arm: str
    conc: int
    status: str
    output_throughput: float | None
    request_throughput: float | None
    total_token_throughput: float | None
    input_throughput: float | None
    output_tput_per_gpu: float | None
    e2e_intvty_p50: float | None
    e2e_intvty_p90: float | None
    e2e_norm_intvty_p90: float | None
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    tpot_p50_ms: float | None
    tpot_p90_ms: float | None
    ttft_mean_ms: float | None
    e2el_mean_ms: float | None
    duration_seconds: float | None
    request_error_rate: float | None
    submission_valid: bool | None
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
    differently, and which one ran is not recoverable after the fact."""

    arm: str
    status: str
    start_time: str
    end_time: str
    extra_server_args: str
    extra_envs: dict[str, str]
    strategy: str
    strategy_reason: str | None
    lifecycle: V6ConcSweepLifecycle
    serving_lease_held: bool | None
    refused: V6ConcSweepRefused | None
    failure: V6Failure | None
    grid: list[V6ConcSweepRung]
    boot: V6ConcSweepBoot
    points: list[V6ConcSweepPoint]


class V6ConcSweepPair(TypedDict, total=False):
    """The two arms joined at one concurrency.

    The pair is ranked on one axis and reports a second. ``*_value`` is on the
    axis ``result.metric`` names -- a slow-tail interactivity percentile
    whenever the session grades on one, which is why these are not named for
    throughput. ``*_guard`` and ``guard_holds`` carry the throughput the
    session would have held a promotion to, reported rather than enforced: a
    sweep exists to draw the interactivity/throughput frontier, so a rung that
    moved along it is a result and not a failure. They are null off the
    interactivity objective, where there is no second axis to hold."""

    conc: int
    baseline_value: float | None
    optimized_value: float | None
    speedup: float | None
    delta_pct: float | None
    baseline_guard: float | None
    optimized_guard: float | None
    guard_holds: bool | None
    baseline_status: str
    optimized_status: str
    error: str | None


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


class V6ConcSweepExt(TypedDict, total=False):
    """The ``conc_sweep`` timeline event's ``ext``, recorded as the sweep runs.

    The sweep runs the CONC ladder twice -- once on the session's optimized
    server args, once on none -- and pairs the curves into a speedup per
    concurrency. What the event adds over the report on disk is the run rather
    than the result: the ladder and where it came from, the budget each arm
    was admitted under, the strategy each ladder ran with, and every rung
    including the ones that were refused or would not boot."""

    schema_version: str
    request: V6ConcSweepRequest
    input_anchor: V6ConcSweepInputAnchor
    workload: V6ConcSweepWorkload
    plan: V6ConcSweepPlan
    budget: V6ConcSweepBudget
    environment: V6ConcSweepEnvironment
    arms: dict[str, V6ConcSweepArm]
    comparison: list[V6ConcSweepPair]
    result: V6ConcSweepResult
    roofline_ceiling: V6ConcSweepCeiling | None
    runtime: V6ConcSweepRuntime
    artifacts: V6ConcSweepArtifacts
    faults: list[V6Failure]
    failure: V6Failure | None
    superseded_sweeps: list[str]


class V6ArchivedFile(TypedDict, total=False):
    """One copy an enablement round's archive holds, and what it is.

    ``role`` is a ``delivery.archive.ROLE_*`` value. It distinguishes a patch
    the round applied from one it refused, which the round's own result cannot.
    """

    path: str
    role: str


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

    ``files`` and ``accepted_config_path`` are session-relative and name only
    copies the round's archive took, which is a weaker claim than "fetchable":
    ``reports/enablement/**`` is packaged outside ``RESERVED_PATHS``, so a
    session that hit the file or byte cap ships a truncated package and a
    consumer must tolerate a miss. ``patches_applied`` and
    ``artifacts_applied[].target`` are the workspace originals the round
    reported, which are never packaged -- they are identity, not a way to
    fetch. Note the asymmetry with ``V6EnablementResult.accepted_config_path``,
    which stays absolute because the revalidation baseline opens that one
    directly."""

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
    files: list[V6ArchivedFile]
    accepted_config_path: str | None
    effective_config: dict[str, Any] | None
    stack_action: dict[str, Any] | None
    runtime: dict[str, Any] | None
    localization_manifest: dict[str, Any] | None
    next_launch_log_excerpt: str | None


class V6EnablementBuild(TypedDict, total=False):
    """One targeted build the enablement lane ran, keyed by its task id.

    ``ok`` is absent, rather than false, on a row that only records an enqueue:
    a build with no verdict yet has not failed."""

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
    and the lane is not charged a stall for it."""

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
    and the outcome in the other."""

    mode: str
    origin: str
    engaged: bool
    trigger: dict[str, Any] | None
    attempts: dict[str, Any]
    builds: dict[str, Any]
    revalidations: dict[str, Any]
    human_review: dict[str, Any]
    result: dict[str, Any] | None
    failure: V6Failure | None
    recipe: dict[str, Any] | None


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
    closed by, and so published it with an empty exit and no duration."""

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
    each action by testing its timestamp against the phase windows."""

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
    """One non-transition ``phase_history`` marker, in the phase that raised it."""

    sequence: int
    reason: str
    evidence: dict[str, Any]
    ts: str


class V6PhaseProposalReview(TypedDict, total=False):
    """The Critic's ruling on one proposal, filed on the proposal."""

    verdict: str
    effective_verdict: str
    held_to_rule: bool
    source: str
    reasoning: str
    confidence: float | None
    failure_reason_code: str | None
    required_evidence: list[str]
    risks: list[dict[str, Any]]
    advice_text: str | None
    alternative_action: str | None
    variants: list[dict[str, Any]]
    reviewed_at: str


class V6PhaseProposalOutcome(TypedDict, total=False):
    """What the loop did with a proposal once it had a ruling."""

    materialized: bool
    denied: bool
    reauthored: bool
    task_id: str | None
    patch_verdict_key: str | None
    settled_at: str


class V6PhaseProposal(TypedDict, total=False):
    """One proposal the phase raised, with the ruling and the result on it.

    Recorded when the proposal is minted rather than when it is acted on,
    because most proposals are never acted on: one the Critic refused, or one
    left pending when the phase exited, produces no dispatch and so has no
    other row anywhere.

    This is the only place a Critic ruling can be filed with the thing it ruled
    on. ``framework_agent`` carries ``proposals[].critic_review`` too, but its
    rows exist for two creation paths inside one phase; a ruling on a KERNEL
    ``kernel_opt``, on a PRELUDE ``baseline``, or on an action no framework arm
    maps to had no subject row anywhere, and was dropped without trace."""

    proposal_msg_id: str
    action: str
    from_agent: str
    phase: str
    macro_cycle: int
    tick: int
    predicted_gain_pct: float | None
    candidate_id: str | None
    variant_name: str | None
    proposed_at: str
    critic_review: V6PhaseProposalReview
    outcome: V6PhaseProposalOutcome


class V6PhaseExt(TypedDict, total=False):
    """The ``phase`` timeline event's ``ext``: the run's time in one phase.

    The one event that is about the run rather than about work. Every other
    event's id is scoped by a phase, and until this event existed the timeline
    held no record of the phases themselves -- a reader could see a baseline
    event tagged ``framework_agent`` and had no way to learn when that phase was
    entered, why the run left it, or how long it had.

    One event per ``(phase, macro_cycle)``, covering every entry into that phase
    in that cycle. See :class:`V6PhaseSegment` for why a re-entry is a row here
    rather than an event of its own."""

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
    proposals: dict[str, Any]


class V6StackAdoption(TypedDict, total=False):
    """One adoption onto the optimization stack, recorded as it was accepted."""

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
    session total is the sum of the very steps it is meant to be checking."""

    stack_len: int
    ts: str
    baseline_tput: float | None
    validated_tput: float | None
    validated_gain_pct: float | None
    source: str
    measurement_basis: str
    #: The axis the figure was graded on, so an intvty-graded gain is not later
    #: read as an output gain. Always the axis the session was configured for:
    #: only a comparison the orchestrator found comparable is recorded here.
    graded_objective: str
    #: The graded axes of the measurement that produced the figure, recorded
    #: beside it because a later revalidation moves the cumulative gain without
    #: re-promoting the recipe.
    perf: V6GradedAxes


class V6StackExt(TypedDict, total=False):
    """The ``stack`` timeline event's ``ext``: what the session actually kept.

    One event per session, because there is one stack. Its adoptions arrive from
    PRELUDE warm replay, EXPLORE, FRAMEWORK_AGENT and KERNEL_AGENT and form a
    single ordered chain; scoping the event by phase would cut that chain at
    every phase boundary, which is exactly where its before / after pairs have
    to line up for the reconciliation to mean anything."""

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
    proposal's summary verdict does not say which variants that was."""

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
    effective verdict alone says one was approved that the Critic refused."""

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


class CriticIteration(TypedDict, total=False):
    """One critic-agent review pass over a proposed change.

    ``verdict`` reads the iteration's rulings as one line (``2 approve, 1
    reject``) and ``verdict_counts`` carries the same distribution for callers
    that need to count. A pass that only spoke -- a heartbeat, a request for
    context -- rules on nothing and leaves both empty."""

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
    kb_priors: dict[str, Any]
    framework_reviews: list[dict[str, Any]]


class V6Critic(TypedDict, total=False):
    """The critic agent's own run, outside the business timeline.

    The per-proposal verdicts stay with the proposals they judge; this key
    carries the agent's session-level run -- how many times it was asked, about
    what, and what it concluded each time."""

    iterations: list[CriticIteration]


class V6RobustnessIntent(TypedDict, total=False):
    """One intent the robustness agent raised on a turn."""

    type: str
    severity: str
    topic: str
    payload: dict[str, Any]


class V6RobustnessTurn(TypedDict, total=False):
    """The robustness agent's account of one turn.

    ``outcome`` distinguishes a turn that produced intents from one the agent
    could not complete, which is the distinction the section it replaces could
    not express: a mute agent and a silent session looked identical."""

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
    empty ``turns`` is an answer, not a gap: the agent never completed a turn."""

    turns: list[V6RobustnessTurn]


class V6RobustnessFinding(TypedDict, total=False):
    """One firing of the robustness ladder, as the ladder itself persisted it.

    Read from the ladder's own JSONL log at close time rather than mirrored
    through a recorder: the log is already durable and complete, so a second
    copy taken as each finding was raised could only be the same rows or fewer."""

    tick_index: int
    timestamp_unix: float
    symptom_name: str
    severity: str
    summary: str
    rca_text: str
    intents: list[str]
    evidence: dict[str, Any]


class V6CloseRobustness(TypedDict, total=False):
    """The close sequencer's final robustness verdict."""

    escalated: bool
    stop_reason: str
    findings: list[V6RobustnessFinding]
    findings_total: int


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
    least one step reporting a failure. The two used to be the same word, which
    made a healthy session indistinguishable from a damaged one.
    """

    status: Literal["running", "succeeded", "failed", "degraded"]
    start_time: str
    end_time: str
    close_sequence_done: bool
    steps: list[dict[str, Any]]
    # The whole robustness account: the close-out's verdict (``escalated`` and
    # the ``stop_reason`` it was drawn from), the agent's own ``turns``, and
    # the ``findings`` the ladder persisted. Together because they are one
    # account of one thing -- the verdict on its own says a session was
    # escalated without saying what for, and the findings of an un-escalated
    # session, the ones judged survivable, had nowhere to be read at all.
    # ``findings`` and ``findings_total`` are absent when the ladder never
    # wrote anything, because an empty list would claim it ran and found none.
    robustness: V6CloseRobustness
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


# V6 FRAMEWORK_AGENT timeline event
class V6FrameworkPolicyConfigArm(TypedDict, total=False):
    """The configuration arm's own thresholds."""

    keep_gain_threshold_pct: float | None
    empty_streak_threshold: int | None
    lookback: int | None


class V6FrameworkPolicySourceArm(TypedDict, total=False):
    """The source arm's own thresholds."""

    no_keep_streak_threshold: int | None
    discovery_retry_limit: int | None
    authoring_enabled: bool | None


class V6FrameworkPolicy(TypedDict, total=False):
    """The thresholds one entry ran under, as it resolved them.

    Recorded rather than cited, because a threshold read back at export is the
    one the session ended on and not the one this entry acted under."""

    keep_threshold_pct: float | None
    variant_timeout_sec: int | None
    overtime_kill_ratio: float | None
    force_exit_budget_pct: float | None
    config: V6FrameworkPolicyConfigArm
    source: V6FrameworkPolicySourceArm


class V6FrameworkPlateauReading(TypedDict, total=False):
    """One plateau evaluation, beside the values it ruled on.

    Appended rather than keyed: two readings that agree are still two readings.
    ``path`` names the reader -- the advisory asks whether to switch arms and
    the exit whether the phase may leave -- and ``triggered`` of ``None`` is a
    reading that could not rule, which the phase treats as a live arm."""

    arm: str
    path: str
    evaluated_at: str
    triggered: bool | None
    inputs: dict[str, Any]
    thresholds: dict[str, Any]


class V6FrameworkRun(TypedDict, total=False):
    """One dispatch this entry made, and what it came back with.

    ``role`` is not derivable from ``arm``: the source arm dispatches twice per
    candidate, once to discover candidates and once to author a patch from one.
    ``produced_ids`` is projected at assembly from the proposals naming this
    run, since a run's own row does not know what it yielded."""

    run_id: str
    role: str
    arm: str
    status: str
    domain: str
    scope: str
    gap_canonical_id: str
    reason: str
    dispatched_at: str
    completed_at: str
    worktree: str
    summary: str
    tags: list[str]
    transcripts: list[str]
    new_findings: list[str]
    residual_questions: list[str]
    notes: list[str]
    parallelism: int | None
    proposals_total: int | None
    empty: bool
    confidence: float | None
    confidence_avg: float | None
    ensemble_scores: dict[str, Any]
    produced_ids: list[str]


class V6FrameworkCriticVariantRuling(TypedDict, total=False):
    """The Critic's ruling on one variant of a grid reviewed as a whole.

    A rejected variant reaches no bench, so no attempt row carries its
    ruling and this is the only place it is readable."""

    variant_name: str
    verdict: str
    effective_verdict: str
    held_to_rule: str
    reason: str
    failure_reason_code: str


class V6FrameworkCriticReviewOutcome(TypedDict, total=False):
    """What the loop did with a ruling.

    An ``advise`` that materialised and an ``advise`` held at the patch gate
    are the same ruling with opposite outcomes. ``patch_verdict_key`` is the
    subject the patch gate consults the ruling under, which connects a blocked
    ``integrate_patch`` back to the review that blocked it."""

    materialized: bool
    denied: bool
    reauthored: bool
    patch_verdict_key: str


class V6FrameworkCriticReview(TypedDict, total=False):
    """The Critic's ruling on one proposal, inline on the proposal.

    Both verdicts are kept: a reject the loop held to a rule that only declared
    ``advise`` is two facts, and either alone misreads the round. ``reviewer``
    separates a ruling the Critic authored from one it never got to make.
    The Critic's advisory keys are merged in as authored, so a consumer may
    meet fields beyond the ones named here."""

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
    variants: list[V6FrameworkCriticVariantRuling]
    outcome: V6FrameworkCriticReviewOutcome
    artifacts: dict[str, Any]
    kb: dict[str, Any]


class V6FrameworkProposalTerminal(TypedDict, total=False):
    """Where a proposal ended up.

    ``dropped`` covers every way it never reached a measurement and ``reason``
    says which; ``pending`` is one the phase never resolved, the honest reading
    of a session killed mid-review."""

    disposition: str
    reason: str
    settled_at: str


class V6FrameworkLifecycleStep(TypedDict, total=False):
    """One step a proposal moved through, recorded as it happened.

    Recorded rather than derived from counters, so a candidate re-authored
    twice and then retried once reads as three steps."""

    step: str
    ts: str
    run_ref: str
    outcome: str
    reason: str


class V6FrameworkProposal(TypedDict, total=False):
    """One thing this entry pursued, whichever producer raised it.

    ``run_ref`` is absent rather than empty on a proposal with no dispatch
    behind it, which is the load-bearing fact for the producers that have no
    parent run at all: the orchestration agent and the seed grid.
    ``attempt_refs`` is projected at assembly from the attempts naming this
    proposal, because a second stored copy of the link is a second thing that
    can disagree."""

    proposal_id: str
    arm: str
    producer: str
    producer_ref: str
    run_ref: str
    domain: str
    scope: str
    lever_kind: str
    gap_canonical_id: str
    source_ref: str
    repo: str
    title: str
    verdict: str
    route: str
    changed_files: list[str]
    confidence: float | None
    critic_review: V6FrameworkCriticReview
    terminal: V6FrameworkProposalTerminal
    lifecycle: list[V6FrameworkLifecycleStep]
    attempt_refs: list[str]


class V6FrameworkStack(TypedDict, total=False):
    """The configuration an attempt was measured on top of.

    Recorded rather than referenced: every KEEP advances the session's stack,
    so what the session serves now is not what this attempt was judged
    against. Both arms have one -- a source patch sits on whatever is being
    served, exactly as a config variant does."""

    throughput: float | None
    accuracy: float | None
    extra_server_args: str
    extra_envs: dict[str, Any]
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str | None


class V6FrameworkConfigDelta(TypedDict, total=False):
    """What one configuration variant changed about the serving command."""

    extra_server_args: str
    extra_envs: dict[str, Any]
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str | None


class V6FrameworkMeasurement(TypedDict, total=False):
    """Both ends of the throughput pair, and the runtime that produced them.

    The pair rather than the percentage alone, because the anchor advances on
    every KEEP and a gain without its denominator adds to nothing."""

    before_tput: float | None
    after_tput: float | None
    gain_pct: float | None
    runtime_sec: float | None
    estimated_output_throughput: float | None
    e2e_intvty_p50: float | None
    e2e_intvty_p90: float | None
    output_tput_per_gpu: float | None
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    tpot_p50_ms: float | None
    tpot_p90_ms: float | None
    duration_s: float | None
    request_error_rate: float | None


class V6FrameworkAccuracy(TypedDict, total=False):
    """The accuracy gate's inputs and verdict.

    ``required`` of ``None`` is a gate that never ran, which is not the same as
    one that ran and failed."""

    required: bool | None
    reference: float | None
    value: float | None
    passed: bool | None


class V6FrameworkAttemptFailure(TypedDict, total=False):
    """Why one attempt yielded no measurement to judge."""

    error_class: str
    error_excerpt: str


class V6FrameworkArtifacts(TypedDict, total=False):
    """Where one attempt's evidence was written."""

    workspace: str
    server_log_path: str
    raw_result_path: str


class V6FrameworkGate(TypedDict, total=False):
    """One gate's verdict on one attempt, as it was evaluated.

    A gate never reached writes no row, which is how a reader tells "did not
    pass" from "did not apply". ``passed`` of ``None`` is a gate that ran and
    could not rule. Order is the order of evaluation, not of the clock: a whole
    gating sequence fits inside one tick."""

    gate: str
    passed: bool | None
    reason: str
    observed: float | None
    threshold: float | None
    ts: str


class V6FrameworkAttempt(TypedDict, total=False):
    """One measured attempt, from either arm.

    One uniform row per thing measured, discriminated by ``arm``, so the
    adoption ledger walks both arms with one reader. Which fields carry
    still follows the arm -- a variant has a ``fingerprint`` and a
    ``config_delta``, an authored patch has a ``patch_path`` and the files it
    touched -- but the lifecycle and the verdict are the same shape for both.
    ``blocked_by`` is projected at assembly as the first gate that did not
    pass."""

    attempt_id: str
    arm: str
    ts: str
    round_id: str
    task_id: str
    proposal_ref: str
    provenance: str
    outcome: str
    reason: str
    stage: str
    decision: str
    adopted: bool | None
    attribution_eligible: bool | None
    validation_basis: str
    fingerprint: str
    variant_name: str
    candidate_id: str
    source_ref: str
    route: str
    patch_source: str
    patch_path: str
    patches_applied: list[str]
    target_files: list[str]
    accepted_kernels: list[str]
    measured_against: V6FrameworkStack
    config_delta: V6FrameworkConfigDelta
    measurement: V6FrameworkMeasurement
    agentx_policy: dict[str, Any]
    submission_valid: bool | None
    submission_invalid_reasons: list[str]
    accuracy_passed: bool | None
    accuracy: V6FrameworkAccuracy
    failure: V6FrameworkAttemptFailure
    artifacts: V6FrameworkArtifacts
    gates: list[V6FrameworkGate]
    blocked_by: str | None


class V6FrameworkExit(TypedDict, total=False):
    """Why the entry left. Not a failure -- every entry that closes has one."""

    reason: str
    trigger: str
    hint: str
    switch_bottleneck: bool | None


class V6FrameworkExt(TypedDict, total=False):
    """``ext`` of the V6 ``framework_agent`` timeline event.

    Both OPTIMIZE arms in one event: the configuration arm searches server args
    and env vars, the source arm lands upstream patches, and the phase leaves
    only when both have run dry. The shape follows the progression a proposal
    moves along rather than the arm it belongs to -- ``proposals`` is the main
    line, ``runs`` holds the dispatch facts those rows reference, and
    ``attempts`` is one uniform row per thing measured -- because the arms
    differ in content and not in lifecycle. ``plateau`` sits off that
    progression.

    ``failure`` is absent on an entry that did not fail as a whole, which is
    not the same as one that failed at nothing: a failed attempt is on its
    attempt row, and a failed dispatch on its run row."""

    macro_cycle: int
    duration_sec: float | None
    policy: V6FrameworkPolicy
    plateau: list[V6FrameworkPlateauReading]
    runs: list[V6FrameworkRun]
    proposals: list[V6FrameworkProposal]
    attempts: list[V6FrameworkAttempt]
    exit: V6FrameworkExit
    failure: V6Failure | None


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


class V6KernelRewriteE2E(TypedDict, total=False):
    """End-to-end integration sub-result of one kernel rewrite.

    Derived at assembly from the event's ``integrate`` rows, never accepted
    from a caller: the gate that produces it runs outside the phase that
    produced the rewrite, so the lane has no verdict of its own to state. The
    rows carry the full account, including the attempts this collapses; this
    is the standing verdict, on the row it rules on.
    """

    integrated: bool
    e2e_gain_pct: float | None
    validated: bool | None
    decision: str | None
    patch_path: str | None
    target_file: str | None


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


class V6KernelDiscoveredKernel(TypedDict, total=False):
    """One kernel the trace attributed, with profiling fields the summary drops."""

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
    """What is peculiar to the forge route for one visit.

    The candidates forge produced are not here: they are on ``ext.attempts``
    beside GEAK's, under one set of names. What remains is the work only forge
    does -- the re-profile it can run on entry, and the trace analysis that
    nominates its targets."""

    engaged: bool
    reprofile: V6KernelReprofile | None
    trace_analyze_runs: list[V6KernelTraceAnalyzeRun]
    discovered_kernels: list[V6KernelDiscoveredKernel]
    recommended_kernels: list[V6KernelDiscoveredKernel]


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
    """One kernel GEAK considered, replayed from its conclusion file.

    GEAK's ``kernel_journey.json`` names every kernel it considered, which
    backends it dispatched and what each measured, not just the acceptances
    that survived. These rows are shaped like forge's because the orchestrator
    replays them through the same field helpers, which is exactly why they are
    stored under GEAK and tagged with their producer rather than merged into
    the forge lane the superseded projection appended them to."""

    kernel_id: str
    name: str
    op_kind: str | None
    gpu_pct: float | None
    micro_speedup: float | None
    dispatched: bool
    backends: list[str]
    skip_reason: str | None
    task_group: str | None
    backend_result: V6KernelGeakBackendResult | None
    e2e: V6KernelRewriteE2E | None


class V6KernelGeakAuthoredKernel(V6RowScope, total=False):
    """One kernel GEAK authored and accepted.

    GEAK routes an acceptance to its kernel queue or its head queue purely by
    which queue proposed it, and both lanes carry the same parity-checked
    ``e2e_delta_pct``; reading only the first drops most of the campaign."""

    short_name: str | None
    kernel_id: str | None
    cand_tag: str | None
    name_source: str
    op_kind: str | None
    lane: str | None
    e2e_delta_pct: float | None
    alias_collapsed: bool
    aliases: list[str]


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
    harness, and the adoption verdict rests solely on the rebench."""

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
    metric_basis: str | None
    bench_client: str | None
    ttft_mean_ms: float | None
    tpot_mean_ms: float | None
    output_parity: Any


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
    """How GEAK's re-measurement campaign was bounded, and how it ended.

    The attempts themselves are on ``ext.rebench``, which the settled attempts
    point at by id. What stays here is what only GEAK's campaign has: the
    per-cycle ceiling it ran under, which attempt the acceptances were settled
    against, and the terminal error when it never reached one."""

    required: bool
    max_attempts: int | None
    attempts_used: int
    settled_against: str
    final_status: str | None
    final_error_class: str | None
    final_error: str | None
    conflicting_decisions: list[str]


class V6KernelGeak(TypedDict, total=False):
    """What is peculiar to the GEAK route for one visit, in causal order.

    The kernels GEAK tried are not here: they are on ``ext.attempts`` beside
    forge's. What remains is the delegation itself -- the conditions GEAK was
    given, how its runner ended, what it claimed and what configuration it
    handed back."""

    engaged: bool
    handoff: V6KernelGeakHandoff | None
    delegation: V6KernelGeakDelegation | None
    discovery_runs: list[V6KernelGeakDiscoveryRun]
    claim: V6KernelGeakClaim | None
    product: V6KernelGeakProduct | None
    rebench: V6KernelGeakRebench


class V6KernelAttempt(TypedDict, total=False):
    """One candidate, from either route, in the shape both routes fill.

    The two routes run different machinery and their producers name the same
    facts differently -- a status lives on the row for forge and inside a
    backend block for GEAK. Normalizing at assembly is what lets one reader
    replay the whole visit in order without knowing which producer wrote a
    given row.

    ``accepted`` is the producer's own verdict on its candidate: whether the
    lane kept it, or GEAK named it in its acceptances. ``outcome`` is what the
    instrument then ruled, and ``settled_by`` names which one ruled it --
    ``rebench`` for a GEAK candidate re-benched end to end, ``integrate`` for
    a patch the gate had already measured by the time this visit closed,
    ``lane`` for a forge candidate on the timing of the lane that produced it.
    A reader who cannot see which one spoke cannot tell an unsettled candidate
    from one whose evidence simply lives elsewhere.

    ``gain_pct`` is that instrument's measurement, normalized to a percentage
    from the ratio a forge lane reports and the delta a rebench computes. It
    is the number the visit's verdict is read off, and it is ``None`` when the
    candidate was kept without anything measuring it.

    ``rebench_ref`` and ``integrate_ref`` are the two evidence pointers, and
    they are symmetric: whichever instrument ruled, the row names the record
    that holds the measurement. Forge fills the second, GEAK the first.

    ``detail`` carries what is specific to the producing lane and has no
    counterpart on the other route: a fusion pattern, a GEMM shape count,
    GEAK's GPU share."""

    attempt_id: str
    route: str
    source_kind: str
    kernel_id: str
    name: str
    status: str
    dispatched: bool
    skip_reason: str
    started_at: str
    ended_at: str
    duration_sec: float | None
    backend: str
    backends_tried: list[str]
    speedup: float | None
    gain_pct: float | None
    compile_status: str
    correctness: bool | None
    artifact_path: str
    error_class: str
    failure_reason: str
    micro_decision: str
    accepted: bool
    rebench_ref: str
    integrate_ref: str
    e2e: V6KernelRewriteE2E | None
    outcome: str
    settled_by: str
    unsettled_reason: str
    detail: dict[str, Any]


class V6KernelDeliveredRow(TypedDict, total=False):
    """One candidate this visit handed to the optimization stack.

    A row here says the candidate was kept, not that it improved the model:
    ``gain_pct`` is ``None`` when nothing measured it, and ``settled_by`` says
    which instrument is behind the number when there is one."""

    route: str
    source_kind: str
    ref: str
    kernel_id: str
    gain_pct: float | None
    settled_by: str


class V6KernelStackDelta(TypedDict, total=False):
    """Optimization-stack entries this visit added and removed."""

    added: list[dict[str, Any]]
    removed: list[dict[str, Any]]


class V6Throughput(TypedDict, total=False):
    """The throughput anchors a stage is judged against, and the gains they yield.

    Three anchors, because a result means different things depending on what it
    is read against: ``before`` is where the stage started -- the running best
    when it was entered -- ``after`` is where it left off, and
    ``session_baseline`` is the session's first measurement, which no stage
    moves. A flat ``tput_before`` beside a bare ``net_gain_pct`` left the
    denominator to be inferred, and left the baseline anchor recorded with
    nothing ever read against it.

    Each gain names the pair it came from: ``gain_pct`` is what this stage
    moved, ``session_gain_pct`` is where the session stands after it. So a
    further comparison is one field here, derived from the anchors already
    beside it, rather than another pair threaded through every outcome.

    Every gain here is arithmetic over those anchors, not an independent
    measurement. A stage's own numbers can outrun what the ledger will
    validate; ``cumulative_gain_validated_out`` on the outcome is the
    validated figure, and the two are meant to be comparable.
    """

    before: float | None
    after: float | None
    session_baseline: float | None
    gain_pct: float | None
    session_gain_pct: float | None


class V6KernelOutcome(TypedDict, total=False):
    """How the visit ran, in one vocabulary both routes reach.

    ``verdict`` is derived from the settled attempts rather than stated by the
    phase, so it cannot contradict the instruments that ruled on them. It is
    one of ``improved`` (a candidate was kept and something measured a gain on
    it), ``no_improvement`` (the visit ran and nothing measured a gain) or
    ``failed`` (the visit raised, or every attempt failed before it could be
    judged). It is empty on a visit that never concluded -- one rebuilt from
    the rows a killed session left behind -- because the rows may add up to a
    gain nobody ever concluded, and the event's ``interrupted`` status is the
    only statement such a visit supports.

    The subject is the visit, not the eventual fate of what it produced. A
    forge patch is gated end to end after this visit exits, and that verdict
    lands on the event of the cycle that ran the gate, so a verdict waiting on
    it could never be stated here at all. Follow ``delivered`` into a later
    event's ``integrate`` rows for that.

    ``reason`` says why whenever the verdict is not an improvement.
    ``error_class`` and ``failed_stage`` name a fault the visit hit, which is
    not the same as the verdict: a raising tick is filed against the session's
    crash count and the loop carries on, so a visit can fault somewhere and
    still deliver a measured candidate. Read together they separate a clean
    empty-handed visit from one that blew up, and a clean win from one that
    was not come by cleanly.

    ``throughput`` carries every anchor the visit is read against and the gain
    against each. ``cumulative_gain_validated_out`` stays outside it because it
    is a different kind of number: the validated ledger's figure for the whole
    stack, stated by the phase rather than derived from this visit's anchors,
    and ``stack_depth_out`` is the stack length it was measured at."""

    route: str
    verdict: str
    reason: str
    error_class: str
    failed_stage: str
    exit_reason: str | None
    throughput: V6Throughput
    cumulative_gain_validated_out: float | None
    stack_depth_out: int | None
    delivered: list[V6KernelDeliveredRow]
    stack_delta: V6KernelStackDelta


class V6KernelIntegrateRun(TypedDict, total=False):
    """One end-to-end integrate gate verdict on one queued patch.

    The gate is the orchestrator's, not a lane's. A KERNEL visit hands a KEEP
    to a queue and exits; a later step drains the queue, measures the patch end
    to end and rules on it. So this is a sibling of the two routes rather than
    a field inside either, and it can settle in a later cycle than the visit
    that produced the patch -- which ``settled_in_macro_cycle`` states, because
    the event this row is attached to is the one that ran the gate.

    ``decision`` and ``rejected_reason`` answer different questions.
    ``REVERT`` is a verdict on the patch; a ``rejected_reason`` is an
    exhausted budget, which drops a patch without ever having ruled against
    it. ``fault_count`` is the same distinction counted: an integration fault
    never measured the patch fairly."""

    integration_id: str
    kernel_id: str
    decision: str
    status: str
    attempt_count: int | None
    fault_count: int | None
    gain_pct: float | None
    accuracy_pass: bool | None
    validation_tier: str
    patch_path: str
    target_file: str
    error_class: str
    rejected_reason: str
    retryable: bool
    settled_at: str
    settled_in_macro_cycle: int | None
    extra_server_args: str
    basis: str
    alignment_status: str
    gain_attributed: bool | None


class V6KernelExt(TypedDict, total=False):
    """``ext`` of the V6 ``kernel`` timeline event.

    Three layers, and which layer a fact belongs in is decided by what kind of
    fact it is rather than by which route produced it. ``attempts`` holds every
    candidate; ``rebench``, ``integrate`` and ``measurements`` hold the evidence
    that ruled on them; ``forge`` and ``geak`` hold only what is peculiar to one
    route; ``outcome`` states what the visit delivered.

    ``geak`` and ``forge`` are mutually exclusive by construction: the entry
    hook picks one of three routes, so the block that did not run stays absent
    rather than being emitted empty."""

    macro_cycle: int
    in_flight_stage: str | None
    duration_sec: float | None
    entry: V6KernelEntry
    attempts: list[V6KernelAttempt]
    rebench: list[V6KernelRebenchAttempt]
    integrate: list[V6KernelIntegrateRun]
    measurements: list[dict[str, Any]]
    geak: V6KernelGeak | None
    forge: V6KernelForge | None
    outcome: V6KernelOutcome


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
    robustness: V6Robustness


__all__ = [
    "CriticIteration",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V6",
    "SessionBreakdown",
    "V6ArchivedFile",
    "V6BaselineExt",
    "V6BaselineProgress",
    "V6Close",
    "V6CloseRobustness",
    "V6ConcSweepArm",
    "V6ConcSweepArtifacts",
    "V6ConcSweepBoot",
    "V6ConcSweepBootAttempt",
    "V6ConcSweepBudget",
    "V6ConcSweepCeiling",
    "V6ConcSweepCeilingModel",
    "V6ConcSweepCeilingRow",
    "V6ConcSweepEnvironment",
    "V6ConcSweepExt",
    "V6ConcSweepInputAnchor",
    "V6ConcSweepLifecycle",
    "V6ConcSweepPair",
    "V6ConcSweepPlan",
    "V6ConcSweepPoint",
    "V6ConcSweepRefused",
    "V6ConcSweepRequest",
    "V6ConcSweepResult",
    "V6ConcSweepRung",
    "V6ConcSweepRuntime",
    "V6ConcSweepWorkload",
    "V6Critic",
    "V6CriticReview",
    "V6CriticReviewVariant",
    "V6EnablementAttempt",
    "V6EnablementBuild",
    "V6EnablementExt",
    "V6EnablementRevalidation",
    "V6Failure",
    "V6FrameworkAccuracy",
    "V6FrameworkArtifacts",
    "V6FrameworkAttempt",
    "V6FrameworkAttemptFailure",
    "V6FrameworkConfigDelta",
    "V6FrameworkCriticReview",
    "V6FrameworkCriticReviewOutcome",
    "V6FrameworkCriticVariantRuling",
    "V6FrameworkExit",
    "V6FrameworkExt",
    "V6FrameworkGate",
    "V6FrameworkLifecycleStep",
    "V6FrameworkMeasurement",
    "V6FrameworkPlateauReading",
    "V6FrameworkPolicy",
    "V6FrameworkPolicyConfigArm",
    "V6FrameworkPolicySourceArm",
    "V6FrameworkProposal",
    "V6FrameworkProposalTerminal",
    "V6FrameworkRun",
    "V6FrameworkStack",
    "V6GeakCandidate",
    "V6GradedAxes",
    "V6Grading",
    "V6GradingTputGuard",
    "V6KBWriteBackExt",
    "V6KernelAnalysisArtifacts",
    "V6KernelAnalysisDetail",
    "V6KernelAttempt",
    "V6KernelDeliveredRow",
    "V6KernelDiscoveredKernel",
    "V6KernelEntry",
    "V6KernelExt",
    "V6KernelForge",
    "V6KernelGeak",
    "V6KernelGeakAttempt",
    "V6KernelGeakAuthoredKernel",
    "V6KernelGeakBackendResult",
    "V6KernelGeakClaim",
    "V6KernelGeakDelegation",
    "V6KernelGeakDiscoveryRun",
    "V6KernelGeakEnvSelection",
    "V6KernelGeakHandoff",
    "V6KernelGeakProduct",
    "V6KernelGeakRebench",
    "V6KernelIntegrateRun",
    "V6KernelOutcome",
    "V6KernelRebenchAttempt",
    "V6KernelRebenchEngagement",
    "V6KernelReprofile",
    "V6KernelRewriteE2E",
    "V6KernelStackDelta",
    "V6KernelTraceAnalyzeRun",
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
    "V6PhaseAction",
    "V6PhaseExt",
    "V6PhaseMarker",
    "V6PhaseProposal",
    "V6PhaseProposalOutcome",
    "V6PhaseProposalReview",
    "V6PhaseSegment",
    "V6Robustness",
    "V6RobustnessFinding",
    "V6RobustnessIntent",
    "V6RobustnessTurn",
    "V6RooflineEventSnapshot",
    "V6RooflineExt",
    "V6RooflineKernel",
    "V6RooflineKernelTable",
    "V6RooflineProgress",
    "V6RooflineTrajectoryPoint",
    "V6RowScope",
    "V6StackAdoption",
    "V6StackExt",
    "V6StackValidation",
    "V6TaskConfig",
    "V6Throughput",
    "V6TimelineEvent",
    "V6ToolVersion",
    "V6WarmReplayExt",
    "V6WarmStartExt",
    "V6WarmStartMatched",
    "V6WarmStartReads",
]
