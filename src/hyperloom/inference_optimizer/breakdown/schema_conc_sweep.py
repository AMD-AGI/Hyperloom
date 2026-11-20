# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Schema (TypedDict shape) of the ``conc_sweep`` timeline event in ``session_breakdown.json``."""

from __future__ import annotations

from typing import TypedDict

from .schema import V6Failure


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

    ``arms_order`` matters to a reader of the budget: past its first rung the first arm may
    spend half the time left, and the second runs only the rungs it reached, so a budget that
    runs out drops whole pairs. ``concs_ordered`` is the session's own CONC first, then the
    rest cheapest first. The plan names no benchmark cap: each rung is granted one sized to
    the workload it runs, and reports it as ``granted_cap_sec``."""

    grid_source: str | None
    concs_requested: list[int]
    concs_ordered: list[int]
    num_prompts_factor: int | None
    arms_order: list[str]


class V6ConcSweepBudget(TypedDict, total=False):
    """The budget the ladder ran under.

    ``gate_active`` says whether a total was declared. ``rung_cost_sec`` is null:
    a rung is held to its arm's deadline rather than priced. ``deadline`` is a
    wall-clock epoch."""

    declared_total_sec: int | None
    rung_cost_sec: float | None
    gate_active: bool
    deadline: float | None


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
    """One rung whose round tried to bring the arm's server up.

    ``committed`` false appears only in older sessions, on an attempt a later
    rung's boot superseded."""

    conc: int | None
    committed: bool
    status: str
    error_class: str
    error: str | None
    start_time: str
    wall_duration_sec: float | None


class V6ConcSweepBoot(TypedDict, total=False):
    """How the arm's boot attempts resolved: each rung boots until one brings the server up.

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
    e2e_norm_intvty_p90: float | None
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
    The reuse path boots one server on its first rung and reuses it for the
    rest; the restart path pays a server start per rung. They fail
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


__all__ = [
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
]
