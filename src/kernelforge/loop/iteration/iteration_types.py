# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The values one iteration of the optimization loop hands to the next step."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

from kernelforge.evaluation.scoring import DEFAULT_SNR_THRESHOLD_DB
from kernelforge.loop.decision_labels import DecisionLabel
from kernelforge.loop.lessons import SummaryOutcome
from kernelforge.tools.new_path_allowlist import normalize_commit_new_paths
from kernelforge.loop.run_state import IterationVerdict
from kernelforge.loop.search_actors import SearchActorPool


# How far a KEEP has to improve a case's measured time before that case counts as one the KEEP's configuration was
# chosen for.
CONFIG_COVERAGE_MIN_MOVE_RATIO = 0.01
CONFIG_COVERAGE_DISPERSION_MULTIPLE = 1.0


@dataclass
class IterationConfig:
    """Configuration for the autonomous iteration loop."""

    # Target kernel file (single-file modification)
    kernel_file: str

    # Test driver for validation
    driver_script: str
    # Immutable digest captured by forge-loop's campaign configuration.
    canonical_driver_sha256: str = ""
    # Original campaign HEAD used to publish a self-contained cumulative best.
    campaign_base_commit: str = ""

    # Build command (if needed)
    build_command: list[str] | None = None
    build_dir: str | None = None

    # Performance targets
    target_wall_ms: float | None = None
    baseline_wall_ms: float | None = None
    baseline_case_times: dict = field(default_factory=dict)
    # Optional pristine baseline for external publication.
    publication_baseline_wall_ms: float | None = None
    pristine_baseline_wall_ms: float | None = None
    # Warm-start measurements are kept separate from the immutable pristine baseline.
    warm_start_wall_ms: float | None = None
    warm_start_mean_case_speedup: float | None = None
    warm_start_bench: dict = field(default_factory=dict)
    preloop_baseline_unscored_cases: list[str] = field(default_factory=list)
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB

    # Budget
    max_time_hours: float = 8.0  # overnight budget
    deadline_unix: float | None = None
    # Held back for finalization: the loop will not START another Agent session once what remains falls below it
    # (``_is_budget_exhausted``).
    budget_reserve_sec: int = 1800
    # Per-step timeouts (seconds): FIXED, sensible ceilings — like bench's 300s — and independent of the campaign
    # budget.
    build_timeout_sec: int = 900  # 15 min compile ceiling
    # Ceiling for the driver-owned complete correctness suite.
    validate_stage_timeout_sec: int = 1800
    bench_timeout_sec: int = 300

    # Git settings
    git_branch: str = "kernel-agent-optimize"
    workspace_dir: str = "."
    # Optional caller-owned ID.
    experiment_id: str = ""

    # Experiment identity persisted by ExperimentTracker.
    backend: str = ""
    kernel_backend: str = ""

    # Task shape awareness (repository / image_kernel vs single-file snippet).
    task_type: str = ""
    # Declared implementation entry points used for orientation, profiling, JIT hints, and KB identity.
    source_files: list[str] = field(default_factory=list)
    # Target kernel/function names the task flagged (host entry + GPU kernels).
    target_functions: list[str] = field(default_factory=list)

    # Supervisor self-supervision (AVO): when the search stalls, a supervisor LLM (a different model family than the
    # implementer — see orchestrator/supervisor.py) reviews the trajectory and injects fresh directions INSTEAD of the
    # loop stopping at the first plateau.
    supervise_after: int = 3  # consecutive no-improvement iters that trigger
    supervise_cooldown: int = 3  # min iterations between interventions
    max_consecutive_orchestration_errors: int = 3
    # Task context handed to the Analysis Agent and planning chain.
    program_md: str = ""
    # References injected into this run's prompt.
    pr_reference_labels: tuple[str, ...] = ()
    pr_reference_context: str = ""
    # PR refresh event and snapshot deferred until campaign initialization, so a rejected invocation leaves the
    # workspace exactly as it found it.
    pr_kb_event: dict = field(default_factory=dict)
    pr_kb_snapshot: dict = field(default_factory=dict)

    # Human-readable caller identity for the profiled operator.
    operator_name: str = ""
    implementation_signature: str = ""
    implementation_identity: dict = field(default_factory=dict)
    warm_start_commit: str = ""
    warm_start_solution_slug: str = ""
    # Result returned by the CLI's pre-loop recovery publication.
    warm_start_publication: dict = field(default_factory=dict)
    # How many times each bench repeats its measurement in-process, reporting the per-case median. 1 selects
    # single-shot behavior (and omits the --repeat flag entirely, so drivers that don't accept it are unaffected).
    bench_repeat: int = 1
    # Implementer lanes run concurrently from one round's analysis, each in its own workspace copy, and each candidate
    # is measured on its own. 1 keeps the single fused plan and single session this loop has always run; fan-out also
    # needs an ``agent_factory``, because a session is bound to its workspace.
    lanes: int = 1
    # Whether a stalled search may spend an iteration measuring two archived rejected gains applied together.
    merge_stacking: bool = True
    # Ranks the driver self-launches (via torchrun) for a collective task. >1 switches profiling to the per-rank
    # backend, because wrapping the outer process would only profile the launcher, which runs no kernel.
    nproc_per_node: int = 1
    # Paths the Implementer may CREATE and still have committed with a KEEP.
    commit_new_paths: list[str] = field(default_factory=list)
    # How large a per-case improvement has to be, relative to the case's own time, before a KEEP counts as having been
    # configured for that case.
    config_coverage_min_move_ratio: float = CONFIG_COVERAGE_MIN_MOVE_RATIO
    # Mean per-case attainment (``ceiling / measured``, equal-weight across scored cases) at which the campaign has
    # nothing left worth buying and stops. Zero disables the gate, which is the default: a ceiling is an estimate
    # with no framework-side check on its arithmetic, so stopping on one is something an operator opts into.
    roofline_target: float = 0.0

    def __post_init__(self) -> None:
        # Validated here rather than at the CLI boundary alone, so a pattern can never reach the commit/delete sites
        # unvalidated -- including via ``dataclasses.replace``.
        self.commit_new_paths = normalize_commit_new_paths(self.commit_new_paths)


@dataclass(frozen=True)
class LoopAgents:
    """The sessions and services the loop drives: implementer, lane factory, analysis, orchestration, supervisor."""

    agent_fn: Any = None
    agent_factory: Any = None
    analysis_service: Any = None
    orchestration_service: Any = None
    supervisor_fn: Any = None
    # Every relay profile's search actor; when set, the loop selects among them and the four above are its start actor.
    search_actors: SearchActorPool | None = None


@dataclass(frozen=True)
class LoopCallbacks:
    """What the caller is told as the loop runs: each iteration, each committed best, each best ready to publish."""

    on_iteration: Any = None
    on_best_committed: Any = None
    on_best_ready: Any = None


def _with_search_actor(agents: LoopAgents, bundle) -> LoopAgents:
    """The loop's agents with the planning and implementation actor taken from one relay profile, if any."""
    if bundle is None:
        return agents
    return replace(
        agents,
        agent_fn=bundle.agent_fn,
        agent_factory=bundle.agent_factory,
        orchestration_service=bundle.orchestration_service,
        supervisor_fn=bundle.supervisor_fn,
    )


@dataclass(frozen=True)
class _RoundPlan:
    """The plan an orchestrated round synthesized, and whether it is executable."""

    path: str = ""
    executable: bool = False


@dataclass(frozen=True)
class _IterationSupply:
    """What an iteration already holds before it buys a session: a device hazard, a queued lane, a fan-out plan."""

    hazard: Any
    queued_lane: Any
    fan_out_plan: HeldRound | None


@dataclass(frozen=True)
class _IterationFrame:
    """One iteration's number, its session hand-back, its staged merge and the best it is measured against."""

    iteration: int
    session_sink: dict
    merge: _StagedMerge
    best_before: float | None = None
    best_mean_case_speedup_before: float | None = None


class _Attempt(NamedTuple):
    """The candidate an iteration will measure, and where it came from."""

    unmeasured_result: IterationResult | None
    commit_hash: str
    rationale: str
    attempt_source: str
    attempt_diff: str
    reusable_benchmark: object | None
    optimization_plan_path: str
    optimization_plan_executable: bool
    lane_record: SummaryOutcome | None = None


class _Ruling(NamedTuple):
    """What ruling on an attempt leaves to finish: the pending KEEP, whether its checkpoint is final, its commit."""

    pending_keep: dict | None
    keep_checkpoint_finalized: bool
    commit_hash: str


class HeldRound(NamedTuple):
    """What a fan-out round hands the iteration that has to finish without it."""

    plan_path: Path | None
    error: str


class _StagedMerge(NamedTuple):
    """A stacked pair staged for measurement, and the diff that stacks it."""

    pair: tuple | None
    diff: str


class CaseConfigCoverage(NamedTuple):
    """Which scored cases a shipped configuration has ever been chosen for."""

    # case id -> the last KEEP iteration that moved its measured time.
    covered: dict[str, int]
    # Scored cases no KEEP has moved. Nothing has been tuned for them.
    fallback: tuple[str, ...]
    # Groups of covered cases that every KEEP has moved together.
    undifferentiated: tuple[tuple[str, ...], ...]
    # The KEEP iterations this ledger was read off, in order.
    keeps: tuple[int, ...]
    # Scored cases no KEEP on record emitted a timing for, so their coverage is unknown rather than absent.
    unmeasured: tuple[str, ...]
    # KEEP iterations that carried no per-case timings at all.
    unreadable: tuple[int, ...]
    # Covered cases no KEEP ever tested the dispersion of, because no KEEP that moved them carried its independent
    # measurements.
    floor_only: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectionEvidence:
    """A decided candidate's measured score, the incumbent's, and the per-case timings behind them."""

    mean_case_speedup: float | None
    best_mean_case_speedup: float | None
    bench_detail: dict | None = None
    incumbent_case_times: dict[str, float] | None = None


@dataclass(frozen=True)
class WindowGain:
    """The exploit-window trend, or the named reason there is not one."""

    ratio: float | None
    unavailable: str | None

    def __post_init__(self) -> None:
        if (self.ratio is None) == (self.unavailable is None):
            raise ValueError(
                "a window gain is either a ratio or a reason, never both or "
                f"neither: {self.ratio!r} / {self.unavailable!r}"
            )


@dataclass
class IterationResult:
    """Result of a single iteration."""

    iteration: int
    duration_sec: float
    validation_passed: bool
    validation_summary: str
    validation_outcome: str = ""
    wall_ms: float | None = None
    mean_case_speedup: float | None = None
    snr_db: float | None = None
    vgpr: int | None = None
    kept: bool = False  # True if change was kept, False if reverted
    commit_hash: str = ""
    agent_rationale: str = ""
    # Real error tail from the first failing validation stage (for the ledger); populated on validation failure so
    # gate-off runs still record true errors.
    error_output: str = ""
    # True when the iteration raised an unexpected exception (build/validate/bench crash) rather than merely failing
    # validation.
    crashed: bool = False
    # Full measurement detail for the candidate archive (not just the scalars above).
    bench_detail: dict = field(default_factory=dict)
    # Why the agent session ended this iteration (from the in-session gate / SDK): converged / block_budget_exhausted
    # / block_cap / turn_cap / gate_error / agent_stopped / sdk_*. "" when no agent ran.
    session_end_reason: str = ""
    turns: int | None = None
    # Independent of session termination: a candidate that changed protected measurement state is rejected before any
    # canonical driver is executed.
    integrity_violation: bool = False
    # Why the workspace could not be cleared of leftover processes.
    workspace_contention: str = ""
    # Exact model-harness profile that produced this candidate.
    agent_profile_id: str = ""
    agent_provider: str = ""
    agent_model: str = ""
    blockage_epoch: int = 0

    def keep_validation_text(self) -> str:
        """The validation evidence a KEEP publishes: the summary, then any error tail."""
        text = self.validation_summary or "canonical validation passed"
        if self.error_output:
            text = f"{text}\n\n{self.error_output}".strip()
        return text

    def error_signature(self) -> str:
        """The last non-empty line of a failed attempt's error, as its journal signature."""
        if self.validation_passed:
            return ""
        blob = self.error_output or self.validation_summary or ""
        err_lines = [line.strip() for line in blob.splitlines() if line.strip()]
        return err_lines[-1][:160] if err_lines else ""


def _decision_label(result: IterationResult) -> str:
    """The canonical keep/revert label for one finished attempt."""
    if result.integrity_violation:
        return DecisionLabel.REVERT_INTEGRITY.value
    if result.workspace_contention:
        return DecisionLabel.REVERT_CONTENDED.value
    if result.crashed:
        return DecisionLabel.CRASH.value
    if not result.validation_passed:
        if (result.validation_summary or "").startswith("BUILD FAILED"):
            return DecisionLabel.BUILD_FAILED.value
        if result.validation_outcome == "timeout":
            return DecisionLabel.REVERT_VALIDATION_TIMEOUT.value
        if result.validation_outcome in {"driver_error", "invalid_result"}:
            return DecisionLabel.REVERT_VALIDATION_ERROR.value
        return DecisionLabel.REVERT_VALIDATION.value
    return DecisionLabel.KEEP.value if result.kept else DecisionLabel.REVERT_PERF.value


def _verdict_of(result: "IterationResult", decision: str, plan: str) -> IterationVerdict:
    """The run-state verdict of one decided iteration."""
    return IterationVerdict(
        result.iteration, decision, result.kept, result.wall_ms, result.commit_hash, plan, result.mean_case_speedup
    )


def _append_finding(session_sink: dict, summary: str) -> None:
    """Add a loop-authored verdict to the session's findings."""
    session_sink["findings"] = "\n---\n".join(
        part for part in (str(session_sink.get("findings") or ""), summary) if part
    )
