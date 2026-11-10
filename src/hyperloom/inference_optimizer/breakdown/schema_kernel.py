# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema (TypedDict shape) of the KERNEL timeline event in ``session_breakdown.json``."""

from __future__ import annotations

from typing import Any, TypedDict


class V6KernelEntry(TypedDict, total=False):
    """What the KERNEL entry hook decided and what it inherited."""

    route: str
    route_reason: str
    resumed: bool
    stack_depth_in: int | None
    roofline_snapshot_id: int | None
    roofline_snapshot_ts: str | None
    roofline_baseline_gain_at_snapshot: float | None
    #: ``"absent"`` when the entry inherited no analysis snapshot, ``"fresh"`` otherwise.
    snapshot_staleness: str | None


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


class V6KernelTraceAnalyzeRun(V6KernelAnalysisDetail, total=False):
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


class V6KernelRewriteE2E(TypedDict, total=False):
    """End-to-end integration sub-result of one kernel rewrite.

    Derived at assembly from the event's ``integrate`` rows (and a GEAK
    candidate's rebench), never accepted from a caller. ``e2e_gain_pct`` is
    Hyperloom-measured; GEAK's own figure is ``self_reported_e2e_gain_pct``.
    """

    integrated: bool
    e2e_gain_pct: float | None
    self_reported_e2e_gain_pct: float | None
    validated: bool | None
    decision: str | None
    patch_path: str | None
    target_file: str | None
    graded_objective: str | None


class V6KernelRebenchEngagement(TypedDict, total=False):
    """Whether the configuration under test actually took effect."""

    config_matched: bool | None
    overlay_loaded: bool | None
    expected_cfg_hash: str | None
    observed_cfg_hash: str | None
    expected_overlay_digest: str | None
    observed_overlay_digest: str | None


class V6KernelRebenchAttempt(TypedDict, total=False):
    """One end-to-end re-measurement of a candidate."""

    attempt_id: str
    source_kind: str
    idempotency_key: str | None
    task_id: str | None
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


class V6KernelGeakDiscoveryRun(TypedDict, total=False):
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


class V6KernelGeakAttempt(TypedDict, total=False):
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


class V6KernelGeakAuthoredKernel(TypedDict, total=False):
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


class V6KernelGeakEnvSelection(TypedDict, total=False):
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
    final_overlay: str | None
    final_launch_script: str | None
    bench_script: str | None
    final_patch: str | None


class V6KernelGeakRebench(TypedDict, total=False):
    """How GEAK's re-measurement campaign was bounded, and how it ended.

    The attempts themselves are on ``ext.rebench``, which the settled attempts
    point at by id. What stays here is what only GEAK's campaign has: how many
    attempts it used, which attempt the acceptances were settled against, and
    the terminal error when it never reached one."""

    required: bool
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
    gain_axis: str | None  # kernel_micro, output_throughput or the gate's graded axis
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
    gain_axis: str | None
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
    failure: dict[str, Any] | None
    attempts: list[V6KernelAttempt]
    rebench: list[V6KernelRebenchAttempt]
    integrate: list[V6KernelIntegrateRun]
    measurements: list[dict[str, Any]]
    geak: V6KernelGeak | None
    forge: V6KernelForge | None
    outcome: V6KernelOutcome


__all__ = [
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
    "V6Throughput",
]
