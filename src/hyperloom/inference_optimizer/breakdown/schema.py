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


#: Current breakdown schema version. V6 stamps the document once the timeline
#: is recorded by the actions themselves rather than projected out of their
#: artefacts afterwards, which is what makes an event's start time its real one.
SCHEMA_VERSION = SCHEMA_VERSION_V6


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


class V6GeakCandidate(TypedDict, total=False):
    """Where the GEAK candidate stood when the session wound down.

    Under ``close`` for the same reason the baseline tally is. A candidate's
    attempts belong to the kernel event that ran them, and that event closes
    when the phase is left -- but a slot still awaiting a rebench, or one the
    close drain cancelled, is settled after every kernel event has closed.

    The self-reported numbers travel with the verdict so a reader can say what
    was dropped rather than only that something was. They are the optimizer's
    own account and were never re-measured; a candidate that survived to a
    headline has its measured numbers on the stack instead.

    Attributes:
        revalidation_pending (bool): Whether the accepted stack still needs a
            recheck against the current tree, as a resume that inherited a
            stack it did not measure itself leaves it.
        status (str): The slot's terminal status -- ``awaiting_rebench`` on a
            session that ended before its rebench landed,
            ``rebench_cancelled`` when the close drain closed it, empty when
            the candidate was adjudicated and the slot released or when there
            was never a candidate.
        revalidation_error (str | None): Why the candidate was dropped, when it
            was.
        revalidation_error_class (str | None): The failure class of that drop.
        self_reported_gain_pct (float | None): The gain GEAK claimed.
        self_reported_tput (float | None): The throughput GEAK claimed.
        self_reported_basis (str): What GEAK measured that claim against.
    """

    revalidation_pending: bool
    status: str
    revalidation_error: str | None
    revalidation_error_class: str | None
    self_reported_gain_pct: float | None
    self_reported_tput: float | None
    self_reported_basis: str


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


class V6PhaseProposalReview(TypedDict, total=False):
    """The Critic's ruling on one proposal, filed on the proposal.

    Attributes:
        verdict (str): What the Critic authored.
        effective_verdict (str): What the loop acted on, which the envelope
            validator can change.
        held_to_rule (bool): Whether the two differ. Its own field because the
            two verdicts alone say that they differ without saying that the
            difference was imposed.
        source (str): ``critic``, or ``critic_unavailable`` when the ruling
            stood in for one the Critic never made.
        reasoning (str): The Critic's own account, clipped.
        confidence (float | None): How sure it was.
        failure_reason_code (str | None): The code behind a refusal.
        required_evidence (list[str]): What it asked to see first.
        risks (list[dict[str, Any]]): The risks it named.
        advice_text (str | None): The advisory attached to the ruling.
        alternative_action (str | None): What it proposed instead.
        variants (list[dict[str, Any]]): Per-variant rulings, when the
            proposal is a grid the Critic ruled on by name.
        reviewed_at (str): When the ruling was filed.
    """

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
    """What the loop did with a proposal once it had a ruling.

    Attributes:
        materialized (bool): Whether it went ahead.
        denied (bool): Whether the framework refused it.
        reauthored (bool): Whether it was sent back to be authored again.
        task_id (str | None): The task it became. This is what joins the
            proposal to the dispatch row beside it on the same event: without
            it, what was asked for and what was run sit on one event with
            nothing connecting them.
        patch_verdict_key (str | None): The patch the ruling was recorded
            against.
        settled_at (str): When the loop acted.
    """

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
    maps to had no subject row anywhere, and was dropped without trace.

    Attributes:
        proposal_msg_id (str): The bus message carrying it, which keys the row.
        action (str): The action proposed, in the proposer's own words --
            including the ones no framework arm maps to.
        from_agent (str): The role that raised it.
        phase (str): The phase in scope when it was raised.
        macro_cycle (int): The macro cycle it was raised in.
        tick (int): The coordinator tick, for ordering within a phase.
        predicted_gain_pct (float | None): The gain the proposer claimed.
        candidate_id (str | None): The upstream candidate, when it names one.
        variant_name (str | None): The variant, when it names one.
        proposed_at (str): When it was raised.
        critic_review (V6PhaseProposalReview): The ruling. Absent when the
            Critic never reached this proposal, which is a different thing
            from having refused it.
        outcome (V6PhaseProposalOutcome): What the loop then did.
    """

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
        proposals (dict[str, Any]): ``{count, reviewed, materialized, rows}``
            over :class:`V6PhaseProposal`. ``count`` minus ``reviewed`` is the
            proposals the Critic never reached -- left pending when the phase
            exited, or filtered as already reviewed -- which is a different
            thing from having been refused.
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
    proposals: dict[str, Any]


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
    topic: str
    verdict: str
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
    carries the agent's session-level run -- how many times it was asked, about
    what, and what it concluded each time.

    Attributes:
        iterations (list[CriticIteration]): One row per review pass, in the
            order the agent ran them.
    """

    iterations: list[CriticIteration]


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


class V6RobustnessFinding(TypedDict, total=False):
    """One firing of the robustness ladder, as the ladder itself persisted it.

    Read from the ladder's own JSONL log at close time rather than mirrored
    through a recorder: the log is already durable and complete, so a second
    copy taken as each finding was raised could only be the same rows or fewer.

    Attributes:
        tick_index (int): The tick the ladder fired on.
        timestamp_unix (float): When it fired.
        symptom_name (str): The symptom that tripped.
        severity (str): How bad the ladder judged it.
        summary (str): The ladder's one-line account.
        rca_text (str): Its root-cause reasoning, when it produced any.
        intents (list[str]): The types of the intents it raised. The payloads
            are the ladder's own working detail and are not carried: one badly
            degraded session's would outweigh the rest of the breakdown.
        evidence (dict[str, Any]): The counters the finding was drawn from.
    """

    tick_index: int
    timestamp_unix: float
    symptom_name: str
    severity: str
    summary: str
    rca_text: str
    intents: list[str]
    evidence: dict[str, Any]


class V6CloseRobustness(TypedDict, total=False):
    """The session's whole robustness account, in one place.

    Attributes:
        escalated (bool): Whether robustness ended the session.
        stop_reason (str): The reason the verdict was drawn from, carried so
            the escalation can be checked against it.
        turns (list[V6RobustnessTurn]): One row per agent turn, by turn order.
        findings (list[V6RobustnessFinding]): The newest firings of the ladder,
            capped; absent when the ladder never wrote any.
        findings_total (int): How many it wrote in all, so a capped list never
            reads as the whole of it.
    """

    escalated: bool
    stop_reason: str
    turns: list[V6RobustnessTurn]
    findings: list[V6RobustnessFinding]
    findings_total: int


class V6FinalRecipe(TypedDict, total=False):
    """The configuration the session ended on, as the close-out settled it.

    Attributes:
        throughput (float | None): The throughput of the configuration that
            shipped.
        ttft_mean_ms (float | None): The latency the same configuration was
            measured at. Carried because a session that never validated its
            whole stack has no other author-time record of it.
        e2el_mean_ms (float | None): As ``ttft_mean_ms``, end to end.
        action_path (list[str]): The layers still on the stack, in promotion
            order, each ``action`` or ``action:variant``. Not the same as the
            ledger's adoptions: a revert leaves its adoption row standing.
        extra_server_args (str): The cumulative server args of the final launch.
        extra_envs (dict[str, str]): The cumulative env overrides of the final
            launch, beside the args because the two are one recipe.
    """

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
        name (str): The kernel's reported name, falling back to its id.
        op_kind (str | None): The operation kind, read from whichever of the
            kernel, dispatch or e2e block the journey resolved it in.
        gpu_pct (float | None): Share of GPU time the kernel held in the
            profile that nominated it.
        micro_speedup (float | None): The isolated speedup, stated on the
            kernel or left to the backend's verification block.
        dispatched (bool): Whether any backend was dispatched.
        backends (list[str]): Backends dispatched to.
        skip_reason (str | None): Gate reason when not dispatched.
        task_group (str | None): The dispatch task group.
        backend_result (V6KernelGeakBackendResult | None): Backend measurement.
        e2e (V6KernelRewriteE2E | None): Integration sub-result.
    """

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
        aliases (list[str]): The other names this acceptance was written under.
            Collapsing a twin without keeping its name would leave the
            surviving row unfindable by the name a reader has in hand.
    """

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
        metric_basis (str | None): Which metric GEAK's headline was taken on.
        bench_client (str | None): The benchmark client that measured it.
        ttft_mean_ms (float | None): Time to first token GEAK measured.
        tpot_mean_ms (float | None): Time per output token GEAK measured.
        output_parity (Any): GEAK's own parity check on the optimized output.
            Recorded from the runner's result rather than the candidate slot,
            so a run that measured but accepted nothing still reports it.
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
    metric_basis: str | None
    bench_client: str | None
    ttft_mean_ms: float | None
    tpot_mean_ms: float | None
    output_parity: Any


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
    """Assembly-derived per-source candidate tally.

    ``adopted`` is what the rebench validated; ``keeps`` is what the E2E
    integrate gate then kept. They are separate counts because a candidate can
    clear the micro benchmark and never be gated at all -- ``micro_only_keeps``
    is exactly that population, and collapsing it into ``adopted`` left a
    reader unable to tell an adoption from a measurement that looked good.

    Attributes:
        attempted (int): Candidates this source produced.
        adopted (int): Candidates the rebench validated.
        needs_review (int): Candidates no rebench settled.
        rejected (int): Candidates the rebench or their own failure rejected.
        keeps (int): Kernels the integrate gate kept, counted once each.
        reverts (int): Kernels the integrate gate ruled against.
        micro_only_keeps (int): Adoptions the integrate gate never ruled on.
        e2e_gain_pct (float | None): Best end-to-end gain the gate measured
            for this source, or ``None`` when it measured none.
    """

    attempted: int
    adopted: int
    needs_review: int
    rejected: int
    keeps: int
    reverts: int
    micro_only_keeps: int
    e2e_gain_pct: float | None


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
    never measured the patch fairly.

    Attributes:
        integration_id (str): The queued patch's id, which keys the row.
        kernel_id (str): The kernel the patch targeted, joining this row to the
            rewrite lane row it rules on.
        decision (str): The gate's verdict (``KEEP`` / ``REVERT``).
        status (str): How the gate's own run ended.
        attempt_count (int | None): Gate attempts this patch has had.
        fault_count (int | None): How many of those never measured it fairly.
        gain_pct (float | None): The best end-to-end gain measured.
        accuracy_pass (bool | None): Whether accuracy held.
        validation_tier (str): How thoroughly the patch was validated.
        patch_path (str): The patch that was integrated.
        target_file (str): The file it was applied to.
        error_class (str): The last failure's classification.
        rejected_reason (str): Why the patch was dropped without a verdict.
        retryable (bool): Whether the gate will try this patch again.
        settled_at (str): ISO timestamp the verdict landed.
        settled_in_macro_cycle (int | None): The cycle that ran the gate.
        extra_server_args (str): Server-arg fragment the adoption introduced.
        basis (str): Throughput basis the gain was measured on (``hot`` /
            ``cold``). A gain is meaningless without the baseline behind it.
        alignment_status (str): Whether the producer's baseline agreed with
            the orchestrator's.
        gain_attributed (bool | None): Whether the measured gain is this one
            kernel's. A rebench carrying several kernels measured all of them
            together, so the gain is real but unattributable -- a different
            fact from the accuracy check in ``accuracy_pass``.
    """

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
        integrate (list[V6KernelIntegrateRun]): End-to-end gate verdicts that
            settled in this cycle, on patches from either route.
        outcome (V6KernelOutcome): What the visit concluded.
        failure (V6KernelFailure | None): The stage that failed, on a miss.
    """

    macro_cycle: int
    in_flight_stage: str | None
    duration_sec: float | None
    entry: V6KernelEntry
    geak: V6KernelGeak | None
    forge: V6KernelForge | None
    integrate: list[V6KernelIntegrateRun]
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
    downstream consumers. Every section is either recorded at author time or
    projected from what was recorded: the export re-derives nothing.

    Attributes:
        schema_version (str): Schema version string (see ``SCHEMA_VERSION``).
        exported_at_utc (str): ISO UTC timestamp the file was exported.
        exporter_version (str): Version of the exporter that produced the file.
        metadata (V6Metadata): Task identity: session ids and lifecycle, the
            launch configuration and model architecture, tool versions, and
            the Langfuse entrypoint. Recorded at author time.
        outcome (V6Outcome): What the session achieved -- the anchoring
            baseline, the adoption ledger and its validation, and the recipe
            that shipped.
        timeline (list[V6TimelineEvent]): One event per unit of work, in the
            order the work ran.
        close (V6Close): The close-out sequence, its verdict and its artifacts,
            as the sequencer recorded them.
        critic (V6Critic): One row per critic iteration.
        robustness (V6Robustness): The robustness agent's turns, escalations
            and findings.
        warnings (list[str]): Collector warnings emitted while assembling the file.
    """

    schema_version: str
    exported_at_utc: str
    exporter_version: str

    metadata: V6Metadata
    outcome: V6Outcome
    timeline: list[V6TimelineEvent]
    close: V6Close
    critic: V6Critic
    robustness: V6Robustness

    warnings: list[str]


__all__ = [
    "CriticIteration",
    "ExecutorClass",
    "IntegrityStatus",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V6",
    "SessionBreakdown",
    "V6BaselineProgress",
    "V6Close",
    "V6CloseRobustness",
    "V6ConcSweepArm",
    "V6ConcSweepExt",
    "V6ConcSweepPair",
    "V6ConcSweepPoint",
    "V6Critic",
    "V6CriticReview",
    "V6CriticReviewVariant",
    "V6EnablementAttempt",
    "V6EnablementBuild",
    "V6EnablementExt",
    "V6EnablementRevalidation",
    "V6GeakCandidate",
    "V6KBWriteBackExt",
    "V6KernelAdoptedRow",
    "V6KernelAnalysisArtifacts",
    "V6KernelAnalysisDetail",
    "V6KernelCollectiveRun",
    "V6KernelDiscoveredKernel",
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
    "V6KernelIntegrateRun",
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
    "V6RooflineKernel",
    "V6RooflineKernelTable",
    "V6RooflineProgress",
    "V6RooflineTrajectoryPoint",
    "V6RowScope",
    "V6StackAdoption",
    "V6StackExt",
    "V6StackValidation",
    "V6TaskConfig",
    "V6TimelineEvent",
    "V6ToolVersion",
    "V6WarmReplayApplied",
    "V6WarmReplayExt",
    "V6WarmStartExt",
    "V6WarmStartMatched",
    "V6WarmStartReads",
]
