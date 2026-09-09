# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable, evidence-driven autonomous kernel optimization loop."""

from __future__ import annotations

import asyncio
import copy
import contextlib
import hashlib
import inspect
import json
import logging
import math
import os
import signal
import tempfile
import textwrap
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

from kernelforge.agent_backends.session_resume import EXHAUSTED_END_REASON
from kernelforge.llm.process_reaping import processes_under
from kernelforge.llm.workspace_policy import is_protected_path
from kernelforge.llm.git import git
from kernelforge.config import Config
from kernelforge.loop.canonical_correctness import accept_candidate
from kernelforge.loop.validation import run_validation_pipeline
from kernelforge.loop.experience import ExperienceLedger
from kernelforge.loop.lessons import (
    SUMMARY_MIN_SECONDS,
    UNDISPROVEN_CLAIM,
    LessonScope,
    LessonStore,
    build_fallback_document,
    cases_named_in,
    format_outcome_line,
    format_scope_line,
    is_claim_disproved,
    parse_disproof_marker,
    parse_held_fixed,
    parse_negatives_marker,
    summarize_iteration,
)
from kernelforge.loop.archive import CandidateArchive, CandidateRecord
from kernelforge.loop.handoffs import HandoffStore, IterationHandoff
from kernelforge.loop.jit_rebuild import (
    force_jit_rebuild,
    tracked_source_changes,
)
from kernelforge.loop.analysis_runtime import AnalysisRuntimeMixin
from kernelforge.loop.search_policy import (
    MARGINAL_GAIN_SCAN_WINDOW,
    MARGINAL_GAIN_WINDOW,
    NO_CHANGES_STREAK_WINDOW,
    SEARCH_MODE_EXPLOIT,
    SearchPolicyDecision,
    SearchPolicyEngine,
)
from kernelforge.loop.round_budget import (
    admit_dispatch,
    admit_round,
    estimate_measurement_sec,
)
from kernelforge.loop.run_state import (
    MAX_PINNED_ITERATIONS,
    ORCHESTRATION_CIRCUIT_OPEN,
    SESSION_COMPLETED,
    SESSION_PAUSED,
    BestRecord,
    CriticRuling,
    LoopStateStore,
    RunState,
    pin_iteration,
    apply_iteration,
    apply_round_cost,
    apply_supervisor_attempt,
    apply_supervisor_intervention,
    begin_orchestration_probe,
    complete_orchestration_probe,
    finish_session,
    is_infrastructure_decision,
    make_event,
    measured_nothing,
    reconcile_stale_running_session,
    should_resume,
    start_session,
)
from kernelforge.orchestrator.orchestration import (
    OrchestrationInfrastructureError,
)
from kernelforge.orchestrator.supervisor import (
    clear_latest_supervisor_ruling,
    latest_supervisor_ruling_path,
    load_latest_supervisor_ruling,
    persist_supervisor_ruling,
)
from kernelforge.loop.prompt_view import (
    MAX_RECENT_ATTEMPT_LINES,
    render_long_horizon_header,
)
from kernelforge.loop.reporting import BestResultPublisher
from kernelforge.rtk import err_wrap, unavailable_warning as rtk_unavailable_warning
from kernelforge.mcp_server.tools.bench import (
    CaseCoverageError,
    calculate_mean_case_speedup,
    calculate_measurement_case_speedups,
    measure_wallclock,
)
from kernelforge.mcp_server.tools._subprocess import communicate_process_group
from kernelforge.loop.new_path_allowlist import (
    matches_commit_new_paths,
    normalize_commit_new_paths,
)
from kernelforge.durable_io import atomic_write_text
from kernelforge.loop.scoring import (
    DEFAULT_SNR_THRESHOLD_DB,
    KEEP_MEASUREMENT_COUNT,
    SIGMA_REMEASURE_BATCH,
    SIGMA_REMEASURE_MAX_ROUNDS,
    attribute_sigma,
    beats_current_best,
    keep_score,
    measurement_sigma,
    passes_keep_threshold,
    required_keep_speedup,
    rescaled_sigma,
)
from kernelforge.loop.baseline_reference import (
    BASELINE_DRIFT_TOLERANCE,
    BASELINE_DRIFT_TOLERANCE_ENV,
    check_baseline_against_reference,
)
from kernelforge.loop.device_hazard import DeviceHazard, DeviceHazardLog
from kernelforge.loop.fanout import LanePlan, LaneResult, run_lanes
from kernelforge.loop.merge_candidates import (
    MERGE_ATTEMPT_STALL_THRESHOLD,
    MERGE_PRECEDENCE_STREAK_LIMIT,
    MergeCandidate,
    attempted_pairs,
    case_spreads,
    cases_beating_reference,
    eligible_candidates,
    merge_plan,
    select_merge_pair,
)
from kernelforge.mcp_server.tools.registers import check_registers
from kernelforge.tracker import ExperimentTracker, Experiment

log = logging.getLogger(__name__)

# How many recent iteration outcomes the long-horizon prompt header is built from.
LONG_HORIZON_OUTCOME_WINDOW = max(
    MAX_PINNED_ITERATIONS,
    MAX_RECENT_ATTEMPT_LINES,
)

# Where a campaign writes its own output inside the workspace.
LOOP_ARTIFACT_ROOT = "forge_experiments"

# How far a KEEP has to improve a case's measured time before that case counts as one the KEEP's configuration was
# chosen for.
CONFIG_COVERAGE_MIN_MOVE_RATIO = 0.01
CONFIG_COVERAGE_DISPERSION_MULTIPLE = 1.0


def _measurement_case_times(
    bench_detail: dict | None,
) -> dict[str, tuple[float, ...]]:
    """Per-case times from each independent measurement of one bench."""
    measurements = (bench_detail or {}).get("measurements")
    if not isinstance(measurements, list):
        return {}
    per_case: dict[str, list[float]] = {}
    for measurement in measurements:
        if not isinstance(measurement, dict):
            continue
        for case_id, value in (measurement.get("case_times") or {}).items():
            try:
                time_ms = float(value)
            except (TypeError, ValueError):
                continue
            if time_ms > 0:
                per_case.setdefault(str(case_id), []).append(time_ms)
    return {case_id: tuple(times) for case_id, times in per_case.items()}


@dataclass(frozen=True)
class SigmaResolution:
    """The sigma the KEEP bar was charged to, and how it was arrived at."""

    sigma: float | None
    measured_sigma: float | None
    dominant_case: str | None
    variance_share: float | None
    wall_share: float | None
    rounds: int
    sample_size: int
    unstable: bool
    detail: str = ""


def _sigma_attribution_note(resolution: SigmaResolution) -> str:
    """The clause the bench line carries when one case set the bar."""
    if resolution.dominant_case is None:
        if not resolution.detail:
            return ""
        return f"sigma not attributed per case ({resolution.detail}); "
    parts = [
        f"sigma attributed to case {resolution.dominant_case!r} "
        f"({resolution.variance_share:.1%} of variance on "
        f"{resolution.wall_share:.1%} of wall time)"
    ]
    if resolution.rounds:
        parts.append(
            f"bought {resolution.rounds} extra bench(es), sigma over "
            f"{resolution.sample_size} samples per case: "
            f"{resolution.measured_sigma:.6f} -> {resolution.sigma:.6f}"
        )
        if resolution.detail:
            parts.append(f"stopped early: {resolution.detail}")
    else:
        parts.append(f"not re-measured ({resolution.detail})")
    if resolution.unstable:
        parts.append(
            "case still dominates after the bound and the larger sample did "
            "not lower its spread, so the bar below is inflated by one case "
            "rather than by this candidate -- read it as a hint, not a "
            "finding: at nine samples that comparison misfires on about half "
            "of the cases that are merely noisy"
        )
    return "; ".join(parts) + "; "


def _bench_failure_detail(bench_result: dict) -> str:
    """The driver's own evidence for why a bench run produced nothing usable."""
    message = str(bench_result.get("message") or "no failure message reported")
    output = str(bench_result.get("output") or "").strip()
    if not output:
        return message
    return f"{message}\n{textwrap.indent(output[-2000:], '    ')}"


def _build_failure_tail(stdout: bytes, stderr: bytes, limit: int) -> str:
    """The tail of a failed build, taken from whichever stream carried it.

    Only stderr used to be read. ninja prints the compiler's own output on
    stdout, so a ninja failure was reported to the agent as ``BUILD FAILED:``
    and nothing else -- the one line that would have told it what to fix went
    to the stream nobody looked at. ``rtk err`` keeps each diagnostic on the
    stream it arrived on, which makes reading both the fix as well as the
    precondition.
    """
    combined = b"\n".join(part.strip() for part in (stdout or b"", stderr or b"") if part.strip())
    text = combined.decode("utf-8", errors="replace").strip()
    return text[-limit:] if text else "no build output"


def llm_spend_lines(usage: dict) -> list[str]:
    """Render a campaign's LLM spend: the total, then the split by role.

    Four token columns, not two. Priced across the 316 recorded end-to-end
    campaigns, ``cache_read`` is 39.5% of the bill and ``cache_creation`` 32.9%,
    against 1.6% for uncached input -- so a summary that reports only ``in`` and
    ``out`` hides roughly three quarters of what was actually paid for. It also
    hides the effect of any change that shrinks the prompt, because what such a
    change moves is exactly these two columns: the campaign whose prefix fell
    83% reported the same ``in``/``out`` line as the one whose prefix did not.

    Counters are read with a default so a usage dict recorded by an older run --
    or a partial one checkpointed mid-campaign -- renders as 0 rather than
    raising while reporting a result that has already been computed.
    """

    def _row(counters: dict, cost_available: bool) -> str:
        cost = f"${counters.get('total_cost_usd', 0.0):.2f}" if cost_available else "cost unavailable"
        return (
            f"{counters.get('input_tokens', 0):,} in / "
            f"{counters.get('output_tokens', 0):,} out / "
            f"{counters.get('cache_creation_input_tokens', 0):,} cache-write / "
            f"{counters.get('cache_read_input_tokens', 0):,} cache-read tokens, "
            f"{cost} ({counters.get('calls', 0)} calls)"
        )

    cost_available = usage.get("cost_available", "total_cost_usd" in usage)
    lines = [f"  LLM spend: {_row(usage, cost_available)}"]
    # The total alone says a campaign was expensive; it never says what was
    # expensive. Print the split so the next cut can be aimed.
    for name, counters in (usage.get("by_role") or {}).items():
        lines.append(f"    {name}: {_row(counters, cost_available)}")
    return lines


def _patch_paths(patch: str, *, cwd: str) -> list[str]:
    """Every workspace path a patch writes, as git itself reads them."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".diff", encoding="utf-8", delete=False)
    try:
        handle.write(patch if patch.endswith("\n") else patch + "\n")
        handle.close()
        completed = git("apply", "--numstat", "-z", handle.name, cwd=cwd, check=False)
    finally:
        os.unlink(handle.name)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or f"git apply --numstat exited {completed.returncode}"
        raise ValueError(f"the paths in the diff could not be read: {detail}")
    paths: list[str] = []
    records = [record for record in completed.stdout.split("\0") if record]
    for record in records:
        fields = record.split("\t")
        if len(fields) < 3:
            raise ValueError(
                f"the paths in the diff could not be read: unreadable git apply --numstat record {record!r}"
            )
        paths.append(fields[2])
    for line in patch.splitlines():
        for prefix in ("rename from ", "copy from "):
            if line.startswith(prefix):
                paths.append(line[len(prefix) :].strip())
    return sorted({path for path in paths if path})


def _lane_prompt(plan: str, *, serialized_driver: Path) -> str:
    """One lane's plan, with the one thing about the wrapper the session cannot see."""
    return (
        f"`python3 {serialized_driver}` takes an exclusive lock on the GPU this "
        "round shares, so it may sit silent before it starts. That wait is "
        "another lane's benchmark, not a hang -- wait for it rather than looking "
        "for another way to run the driver.\n\n"
        f"{plan}"
    )


@contextlib.contextmanager
def _defer_termination_signals(enabled: bool):
    """Delay SIGTERM/SIGINT across best-commit checkpoint publication."""
    if not enabled:
        yield
        return
    pending: list[int] = []
    previous_handlers: dict[int, object] = {}

    def _defer(signum, _frame) -> None:
        pending.append(signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _defer)
    except ValueError:
        # signal.signal is available only on the process main thread.
        yield
        return
    try:
        yield
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if pending:
            os.kill(os.getpid(), pending[0])


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

    def __post_init__(self) -> None:
        # Validated here rather than at the CLI boundary alone, so a pattern can never reach the commit/delete sites
        # unvalidated -- including via ``dataclasses.replace``.
        self.commit_new_paths = normalize_commit_new_paths(self.commit_new_paths)


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


class HeldRound(NamedTuple):
    """What a fan-out round hands the iteration that has to finish without it."""

    plan_path: Path | None
    error: str


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
    pmc_diagnosis: str = ""
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
    pmc_full: str = ""
    # Structured profile metadata (backend, bottleneck, target kernels, roofline dtype/AI/HBM+compute pct, SoL
    # metrics) for the candidate archive's meta.json, so the supervisor / next agent can consume it without parsing
    # prose.
    profile_meta: dict = field(default_factory=dict)
    # Why the agent session ended this iteration (from the in-session gate / SDK): converged / block_budget_exhausted
    # / block_cap / turn_cap / gate_error / agent_stopped / sdk_*. "" when no agent ran.
    session_end_reason: str = ""
    turns: int | None = None
    # Independent of session termination: a candidate that changed protected measurement state is rejected before any
    # canonical driver is executed.
    integrity_violation: bool = False
    # Why the workspace could not be cleared of leftover processes.
    workspace_contention: str = ""


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


class IterationLoop(AnalysisRuntimeMixin):
    """Autonomous kernel optimization loop."""

    def __init__(
        self,
        iter_config: IterationConfig,
        tracker: ExperimentTracker,
        config: Config | None = None,
        resume: bool = False,
    ):
        self.ic = iter_config
        # Declared here so persistence works before the methods that populate them have run.
        self._best_case_times: dict[str, float] = {}
        # Pairs this process selected and could not stage.
        self._declined_merge_pairs: set[frozenset[int]] = set()
        # Last reported per-case bandwidth.
        self.last_case_bandwidth: dict[str, dict[str, float | int]] = {}
        self._scoring_state_restored = False
        self._unscored_cases: set[str] = set()
        # New files the last commit or discard could act on neither way, because no ``commit_new_paths`` entry admits
        # them.
        self._refused_new_paths: list[str] = []
        # Allowlisted new files a discard left on the tree because they were already there when this loop took the
        # workspace over.
        self._retained_new_paths: list[str] = []
        # Why the last new-file enumeration could not be read, "" when it could.
        self._new_paths_unreadable: str = ""
        # Untracked paths present when the current iteration began -- captured again before resume recovery, which
        # also discards and runs before any iteration.
        self._pre_untracked: set[str] | None = None
        # Validation and benchmarking invoke the driver with no arguments, so a multi-rank task has no other way to
        # tell it how many ranks to launch.
        if self.ic.nproc_per_node > 1:
            os.environ["FORGE_NPROC_PER_NODE"] = str(self.ic.nproc_per_node)
        else:
            os.environ.pop("FORGE_NPROC_PER_NODE", None)
        self.tracker = tracker
        self.config = config or Config.from_env()
        self.resume = resume
        self.experiment: Experiment | None = None
        self.results: list[IterationResult] = []
        # FIXED per-case baseline wall times (case_id -> ms), captured once on the pristine kernel and never
        # overwritten.
        self._baseline_case_times: dict = dict(self.ic.baseline_case_times)
        # UsageAccumulator for the run (set in run()); lets the analyst fold its token spend into the run total.
        self._usage = None
        self.best_wall_ms: float | None = None
        self.best_mean_case_speedup: float | None = None
        self.start_time: float = 0
        # Total LLM token spend for the run, populated from the UsageAccumulator passed to run() (empty when no agent
        # / no accumulator).
        self.llm_usage: dict = {}
        self.persistence_degraded = False
        self.persistence_errors: list[str] = []
        self._analysis_bundle = None
        self._last_published_analysis_commit = ""
        self._active_analysis_context = None
        self._analysis_diff_results = {}
        self.search_policy_engine = SearchPolicyEngine()
        self._search_policy_decision: SearchPolicyDecision | None = None
        self._reported_window_gain_faults: set[str] = set()
        self.handoff_store: HandoffStore | None = None
        # A committed KEEP recovered during synchronous resume preflight waits here until the async run can restore
        # its post-KEEP profile/archive.
        self._recovered_pending_keep: tuple[dict, IterationResult] | None = None
        # Why the loop stopped: "gate_met" (target reached), "budget_exhausted" (not enough time to admit another
        # session), or "round_budget_exhausted" (time is left, but not enough to finish even the narrowest round).
        self.termination_reason: str = ""
        # The round currently being timed, opened when an iteration is admitted and closed once the next one starts.
        self._round_started_at: float | None = None
        self._round_iteration = 0
        self._round_lanes = 1
        self._round_planning_sec = 0.0
        # What the round spent in the canonical validation and benchmark, which is what prices the next round's
        # dispatch.
        self._round_measurement_sec = 0.0
        # The instant the CAMPAIGN began, which on a resumed session is before this process did: ``start_time`` less
        # the wall-clock earlier sessions already banked.
        self._campaign_started_at: float = 0.0
        # The refusal that ended the campaign, as the line the operator sees, kept so the run summary and the
        # published report can say a round was priced out rather than let it read as a round that found nothing.
        self._refused_round: str = ""
        # Supervisor self-supervision (AVO): the latest free-form ruling to pass verbatim through planning, and the
        # factual progress monitor that decides when to request a new review.
        self._supervisor_ruling: str = ""
        self._latest_optimization_plan_path = ""
        self._last_orchestration_plan_executable: bool | None = None
        # The previous round's Plan Critic ruling, kept for the next round's partition.
        self._last_critic_verdict = ""
        self._last_critic_review = ""
        self.monitor = None

    def _expire_supervisor_ruling(self) -> None:
        """Stop injecting a ruling after its stall episode ends."""
        self._supervisor_ruling = ""
        clear_latest_supervisor_ruling(self.ic.workspace_dir)

    def _checkpoint_llm_usage(self) -> None:
        """Best-effort checkpoint of the latest cumulative LLM usage."""
        if self._usage is None:
            return
        try:
            self.llm_usage = dict(self._usage.totals())
            has_usage = bool(self.llm_usage.get("calls")) or any(
                self.llm_usage.get(key)
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            if self.experiment is not None and has_usage:
                self.tracker.set_llm_usage(
                    self.experiment.experiment_id,
                    self.llm_usage,
                )
        except Exception:  # noqa: BLE001 - accounting must never break the loop
            log.debug("failed to checkpoint LLM usage", exc_info=True)

    def _git(self, *args: str) -> str:
        """Read the workspace repository, reporting git's answer verbatim."""
        result = git(*args, cwd=self.ic.workspace_dir, check=False)
        return (result.stdout + "\n" + result.stderr).strip()

    def _workspace_path(self, value: str) -> str:
        """Normalize a task path relative to the workspace when possible."""
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.ic.workspace_dir) / path
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(Path(self.ic.workspace_dir).resolve()))
        except ValueError:
            return str(resolved)

    def _task_fingerprint(self) -> str:
        """Stable identity for the task inputs that define a resume campaign."""
        payload = {
            "kernel_path": self._workspace_path(self.ic.kernel_file),
            "driver_path": self._workspace_path(self.ic.driver_script),
            "task_type": self.ic.task_type,
            "source_files": sorted(self._workspace_path(path) for path in self.ic.source_files),
            "target_functions": sorted(self.ic.target_functions),
            "operator_name": self.ic.operator_name,
            "implementation_signature": self.ic.implementation_signature,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _driver_sha256(self) -> str:
        """Hash the exact driver bytes that validation and benchmarking will run."""
        path = Path(self.ic.driver_script)
        if not path.is_file():
            raise ValueError(f"driver integrity check failed: file is missing: {path}")
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"driver integrity check failed: could not read {path}: {error}") from error

    def _validate_driver_integrity(self, state: RunState) -> str:
        """Accept only the canonical campaign driver."""
        canonical = (self.ic.canonical_driver_sha256 or "").strip().lower()
        if not canonical:
            return ""
        current = self._driver_sha256()
        if current == canonical:
            return current
        raise ValueError("driver integrity check failed: workspace driver does not match the campaign canonical digest")

    def _set_state_identity(self, state: RunState) -> None:
        """Stamp the current workspace/task/git identity onto campaign state."""
        state.kernel_path = self._workspace_path(self.ic.kernel_file)
        state.task_fingerprint = self._task_fingerprint()
        state.git_branch = self.ic.git_branch
        state.head_commit = self._git("rev-parse", "HEAD").splitlines()[0]

    @property
    def _pending_keep_path(self) -> Path:
        return Path(self.ic.workspace_dir) / "forge_experiments" / "pending_keep.json"

    def _tracked_diff_from_head(self) -> str:
        """Return all staged and unstaged tracked changes relative to HEAD."""
        return self._git("diff", "HEAD", "--", ".")

    def _persist_pending_keep(self, pending: dict) -> None:
        """Atomically persist a verified candidate before creating its commit."""
        atomic_write_text(
            self._pending_keep_path,
            json.dumps(pending, indent=2, sort_keys=True, default=str) + "\n",
        )

    def _load_pending_keep(self) -> dict | None:
        path = self._pending_keep_path
        if not path.exists():
            return None
        try:
            pending = json.loads(path.read_text())
        except Exception as error:
            raise ValueError(f"invalid pending KEEP metadata: {path}") from error
        if not isinstance(pending, dict) or pending.get("schema_version") != 2:
            raise ValueError(f"invalid pending KEEP metadata: {path}")
        return pending

    def _clear_pending_keep(self) -> None:
        try:
            self._pending_keep_path.unlink()
        except FileNotFoundError:
            return

    def _search_control_snapshot(self) -> dict:
        """Capture decision-critical planning state for pending KEEP recovery."""
        return {
            "diversification_cycle_completed": (self.run_state.diversification_cycle_completed),
        }

    def _apply_iteration_planning_state(
        self,
        *,
        optimization_plan_created: bool,
    ) -> None:
        """Reduce one completed iteration's planning outcome into run_state."""
        self.run_state.diversification_cycle_completed = (
            self.run_state.search_mode == "DIVERSIFY" and optimization_plan_created
        )

    def _build_pending_keep(
        self,
        result: IterationResult,
        *,
        plan: str,
        best_before: float | None,
        rationale: str,
        kernel_source: str,
    ) -> dict:
        """Capture every fact needed to finish a verified KEEP after restart."""
        patch = self._tracked_diff_from_head()
        if not patch:
            raise ValueError("verified KEEP has no tracked candidate diff")
        base_head = self._git("rev-parse", "HEAD").splitlines()[0]
        validation_text = result.validation_summary or "canonical validation passed"
        if result.error_output:
            validation_text = f"{validation_text}\n\n{result.error_output}".strip()
        benchmark = dict(result.bench_detail or {})
        benchmark.setdefault("median_ms", result.wall_ms)
        changed_files = [
            line.strip()
            for line in self._git(
                "diff",
                "--name-only",
                "HEAD",
                "--",
                ".",
            ).splitlines()
            if line.strip()
        ]
        publication_base = self.ic.campaign_base_commit or base_head
        publication_patch = self._git(
            "diff",
            publication_base,
            "--",
            ".",
        )
        publication_changed_files = [
            line.strip()
            for line in self._git(
                "diff",
                "--name-only",
                publication_base,
                "--",
                ".",
            ).splitlines()
            if line.strip()
        ]
        commit_message = f"iter-{result.iteration}: {rationale[:72]}"
        return {
            "schema_version": 2,
            "campaign_id": self.run_state.campaign_id,
            "session_index": self.run_state.session_index,
            "experiment_id": (self.experiment.experiment_id if self.experiment else ""),
            "base_head": base_head,
            "iteration": result.iteration,
            "wall_ms": result.wall_ms,
            "mean_case_speedup": result.mean_case_speedup,
            "snr_db": result.snr_db,
            "vgpr": result.vgpr,
            "plan": (plan or "").strip(),
            "rationale": rationale,
            "validation_text": validation_text,
            "benchmark": benchmark,
            "changed_files": changed_files,
            "patch": patch,
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "publication_base_commit": publication_base,
            "publication_changed_files": publication_changed_files,
            "publication_patch": publication_patch,
            "kernel_source": kernel_source,
            "kernel_file": self.ic.kernel_file,
            "shape": {},
            "baseline_wall_ms": (
                self.ic.publication_baseline_wall_ms or self.ic.baseline_wall_ms or self.run_state.baseline_wall_ms
            ),
            "pristine_baseline_wall_ms": (
                self.ic.pristine_baseline_wall_ms
                if self.ic.pristine_baseline_wall_ms is not None
                else self.ic.baseline_wall_ms
            ),
            "best_wall_ms_before": best_before,
            "best_mean_case_speedup_before": self.best_mean_case_speedup,
            "session_end_reason": result.session_end_reason,
            "turns": result.turns,
            "search_control": self._search_control_snapshot(),
            "commit_message": commit_message,
            "commit_subject": commit_message.splitlines()[0],
            "task_fingerprint": self._task_fingerprint(),
            "git_branch": self.ic.git_branch,
        }

    def _inspect_pending_keep(self, state: RunState, pending: dict) -> str:
        """Classify a pending KEEP as uncommitted or the exact expected child."""
        if state.session_status == SESSION_COMPLETED:
            raise ValueError("completed campaign cannot be resumed")
        base_head = str(pending.get("base_head") or "")
        patch = str(pending.get("patch") or "")
        iteration = int(pending.get("iteration", 0) or 0)
        expected_hash = str(pending.get("patch_sha256") or "")
        if not base_head or not patch or iteration <= 0:
            raise ValueError("pending KEEP metadata is incomplete")
        if hashlib.sha256(patch.encode()).hexdigest() != expected_hash:
            raise ValueError("pending KEEP metadata patch checksum mismatch")
        if pending.get("campaign_id") != state.campaign_id:
            raise ValueError("pending KEEP campaign mismatch")
        current_kernel = self._workspace_path(self.ic.kernel_file)
        if state.kernel_path and state.kernel_path != current_kernel:
            raise ValueError(f"kernel path mismatch: expected {state.kernel_path}, got {current_kernel}")
        if state.task_fingerprint and state.task_fingerprint != pending.get("task_fingerprint"):
            raise ValueError("pending KEEP state task fingerprint mismatch")
        if pending.get("task_fingerprint") != self._task_fingerprint():
            raise ValueError("pending KEEP task fingerprint mismatch")
        if state.git_branch and state.git_branch != pending.get("git_branch"):
            raise ValueError("pending KEEP state branch mismatch")
        if pending.get("git_branch") != self.ic.git_branch:
            raise ValueError("pending KEEP branch mismatch")

        current_branch = self._git("branch", "--show-current").splitlines()[0]
        if current_branch != self.ic.git_branch:
            raise ValueError(
                f"branch mismatch: workspace is on {current_branch or 'detached HEAD'}, expected {self.ic.git_branch}"
            )
        current_head = self._git("rev-parse", "HEAD").splitlines()[0]
        state_anchor = state.best.commit_hash or state.head_commit
        already_finalized = state.best.iteration == iteration and state.best.commit_hash == current_head
        if not already_finalized and state.next_iteration != iteration:
            raise ValueError(f"pending KEEP iteration mismatch: expected {state.next_iteration}, got {iteration}")
        if not already_finalized and base_head != state_anchor:
            raise ValueError(f"pending KEEP base mismatch: expected {state_anchor}, got {base_head}")

        tracked_diff = self._tracked_diff_from_head()
        if current_head == base_head:
            if tracked_diff and hashlib.sha256(tracked_diff.encode()).hexdigest() != expected_hash:
                raise ValueError("pending KEEP working tree mismatch")
            return "uncommitted"

        parents = self._git("rev-list", "--parents", "-n", "1", current_head).split()
        if len(parents) != 2 or parents[1] != base_head:
            raise ValueError(f"pending KEEP HEAD mismatch: {current_head} is not the expected child")
        if tracked_diff:
            raise ValueError("pending KEEP committed child has tracked workspace changes")
        committed_patch = self._git("diff", base_head, current_head, "--", ".")
        if hashlib.sha256(committed_patch.encode()).hexdigest() != expected_hash:
            raise ValueError("pending KEEP committed patch mismatch")
        subject = self._git("show", "-s", "--format=%s", current_head)
        expected_subject = pending.get("commit_subject") or str(pending.get("commit_message") or "").splitlines()[0]
        if subject != expected_subject:
            raise ValueError("pending KEEP commit message mismatch")
        return "committed"

    def _validate_resume_scoring_state(self, state: RunState) -> None:
        """Reject checkpoints that cannot restore the original scoring rules."""
        if state.best.commit_hash and not state.best_case_times:
            raise ValueError("resume state has no incumbent per-case timings; start a fresh campaign")

    def _validate_resume_state(
        self,
        state: RunState,
        *,
        expected_head: str | None = None,
        allow_dirty: bool = False,
    ) -> None:
        """Fail closed before a resumed invocation mutates persistent state."""
        self._validate_resume_scoring_state(state)
        if state.session_status == SESSION_COMPLETED:
            raise ValueError("completed campaign cannot be resumed")
        if state.best.commit_hash and state.best.mean_case_speedup is None:
            raise ValueError(
                "resume state predates mean-case-speedup scoring; start a fresh "
                "campaign so pristine per-case timings can be captured"
            )
        if not state.baseline_case_times:
            raise ValueError(
                "resume state has no pristine per-case timings; start a fresh "
                "campaign so mean case speedup can be computed"
            )

        self._validate_driver_integrity(state)

        current_kernel = self._workspace_path(self.ic.kernel_file)
        if state.kernel_path and state.kernel_path != current_kernel:
            raise ValueError(f"kernel path mismatch: expected {state.kernel_path}, got {current_kernel}")

        if state.task_fingerprint and state.task_fingerprint != self._task_fingerprint():
            raise ValueError("task fingerprint mismatch")

        current_branch = self._git("branch", "--show-current").splitlines()[0]
        if state.git_branch and state.git_branch != self.ic.git_branch:
            raise ValueError(f"branch mismatch: state uses {state.git_branch}, configuration uses {self.ic.git_branch}")
        if current_branch != self.ic.git_branch:
            raise ValueError(
                f"branch mismatch: workspace is on {current_branch or 'detached HEAD'}, expected {self.ic.git_branch}"
            )

        current_head = self._git("rev-parse", "HEAD").splitlines()[0]
        resume_head = expected_head or state.best.commit_hash or state.head_commit
        if not resume_head:
            raise ValueError("resume state has no HEAD anchor")
        if current_head != resume_head:
            raise ValueError(f"HEAD mismatch: expected {resume_head}, got {current_head}")

        dirty = self._git("status", "--porcelain", "--untracked-files=no")
        if dirty and not allow_dirty:
            raise ValueError("workspace has uncommitted tracked changes")

    def validate_resume_preflight(self) -> RunState:
        """Validate a persisted resume checkpoint without mutating campaign files."""
        store = LoopStateStore(self.ic.workspace_dir)
        if not store.state_path.is_file():
            raise ValueError(f"resume state not found: {store.state_path}")
        state = store.load()
        self.state_store = store
        pending = self._load_pending_keep()
        planned, status, _result, _append_keep = self._plan_resume_recovery(
            state,
            pending,
        )
        self._validate_resume_state(
            planned,
            allow_dirty=status == "uncommitted",
        )
        return state

    def _restore_resume_baseline_case_times(self, state: RunState) -> None:
        """Restore the immutable scoring baseline for a validated resume."""
        if not self.resume:
            return
        state_cases = dict(state.baseline_case_times)
        if not state_cases:
            raise ValueError(
                "resume state has no pristine per-case timings; start a fresh "
                "campaign so mean case speedup can be computed"
            )
        if self._baseline_case_times and self._baseline_case_times != state_cases:
            raise ValueError("resume baseline case timings conflict with the persisted campaign")
        self._baseline_case_times = state_cases
        self.ic.baseline_case_times = dict(state_cases)

    def _list_untracked(self) -> list[str]:
        """Every untracked, non-ignored path in the workspace, as git reports it."""
        listed = git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            cwd=self.ic.workspace_dir,
            check=False,
            text=False,
        )
        if listed.returncode != 0:
            detail = (listed.stderr or listed.stdout).decode(
                errors="surrogateescape"
            ).strip() or f"git ls-files exited {listed.returncode}"
            raise RuntimeError(f"could not list new files: {detail}")
        return [item.decode(errors="surrogateescape") for item in listed.stdout.split(b"\0") if item]

    def _untracked_snapshot(self) -> set[str] | None:
        """Snapshot the untracked set an iteration or lane starts from."""
        try:
            return set(self._list_untracked())
        except RuntimeError as error:
            log.warning("could not snapshot untracked files: %s", error)
            return None

    def _new_paths(self) -> tuple[list[str], list[str]]:
        """Split the workspace's new files into the shippable ones and the rest."""
        patterns = list(self.ic.commit_new_paths)
        own_roots = [LOOP_ARTIFACT_ROOT]
        if self.ic.build_dir:
            build_dir = self._workspace_path(self.ic.build_dir)
            if not Path(build_dir).is_absolute():
                own_roots.append(build_dir)
        admitted: list[str] = []
        refused: list[str] = []
        for path in sorted(self._list_untracked()):
            if any(path == root or path.startswith(f"{root}/") for root in own_roots):
                continue
            allowed = matches_commit_new_paths(path, patterns) and not is_protected_path(
                path,
                workspace=self.ic.workspace_dir,
                # The campaign driver carries no protected name of its own.
                exact_paths=(self.ic.driver_script,),
            )
            if allowed:
                admitted.append(path)
            else:
                refused.append(path)
        return admitted, refused

    def _new_paths_best_effort(self) -> tuple[list[str], list[str]] | None:
        """``_new_paths`` for callers that are already recovering from a failure."""
        try:
            listed = self._new_paths()
        except RuntimeError as error:
            log.warning("skipping the new-file clean: %s", error)
            print(f"  [git] could not enumerate new files, skipping the new-file clean: {error}")
            # Both callers return early from here without reporting, so a refusal list left standing would be read as
            # this iteration's.
            self._refused_new_paths = []
            self._retained_new_paths = []
            self._new_paths_unreadable = str(error)
            return None
        return listed

    def _report_refused_new_paths(
        self,
        refused: list[str],
        action: str,
        retained: Sequence[str] = (),
    ) -> None:
        """Record and print the new files this ``action`` could not act on."""
        self._refused_new_paths = list(refused)
        self._retained_new_paths = list(retained)
        self._new_paths_unreadable = ""
        if refused:
            print(f"  [git] {len(refused)} new file(s) outside commit_new_paths, not {action}: " + ", ".join(refused))

    def _new_paths_need_discard(self) -> bool:
        """Whether new files alone make a discard necessary, refusals reported."""
        listed = self._new_paths_best_effort()
        if listed is None:
            return False
        admitted, refused = listed
        self._report_refused_new_paths(refused, "removed")
        return bool(admitted)

    def _render_uncommittable_new_paths(self) -> str:
        """Tell the Implementer what the new-file report says this iteration."""
        allowlist = ", ".join(self.ic.commit_new_paths) or "(empty)"
        blocks: list[str] = []
        if self._new_paths_unreadable:
            blocks.append(
                "\n".join(
                    (
                        "## New files could not be listed",
                        (
                            "This iteration could not enumerate the "
                            "workspace's new files "
                            f"({self._new_paths_unreadable}), so nothing "
                            "below reports on them. A file you created may "
                            "be sitting on the measured tree uncommitted "
                            "and unremoved; treat the absence of a new-file "
                            "report as unknown, not as nothing."
                        ),
                    )
                )
            )
        if self._refused_new_paths:
            blocks.append(
                "\n".join(
                    (
                        "## New files that cannot ship",
                        (
                            "A KEEP commits tracked edits plus new files "
                            f"matching {allowlist}. These new files match "
                            "nothing there, so a KEEP cannot carry them and "
                            "a REVERT cannot remove them, and the measured "
                            "tree is not the committed tree while they "
                            "exist: " + ", ".join(self._refused_new_paths)
                        ),
                        (
                            "Put the change in a tracked file, or state in "
                            "your findings which path the operator has to "
                            "allowlist and why the change cannot live in a "
                            "tracked file."
                        ),
                    )
                )
            )
        if self._retained_new_paths:
            blocks.append(
                "\n".join(
                    (
                        "## Allowlisted new files this loop did not create",
                        (
                            f"These match {allowlist} but were already on "
                            "the workspace before this loop touched it, "
                            "so they are the operator's or an earlier "
                            "round's and a REVERT leaves them in place. They "
                            "are on the measured tree without being part of "
                            "any candidate: " + ", ".join(self._retained_new_paths)
                        ),
                        (
                            "If one of them is a leftover of your own work, "
                            "say so in your findings -- its effect is being "
                            "measured and attributed to nothing."
                        ),
                    )
                )
            )
        return "\n\n".join(blocks)

    def _git_commit(self, message: str) -> str:
        """Stage ALL tracked modifications and commit, raising on failure."""
        before = self._git("rev-parse", "HEAD").strip()
        git("add", "-u", cwd=self.ic.workspace_dir)

        # Fail-fast here on purpose: a KEEP built from a file set that could not be enumerated would ship an unknown
        # tree.
        admitted, refused = self._new_paths()
        if admitted:
            git("add", "--", *admitted, cwd=self.ic.workspace_dir)
        self._report_refused_new_paths(refused, "committed")

        git("commit", "-m", message, cwd=self.ic.workspace_dir)

        after = self._git("rev-parse", "HEAD").strip()
        if not after or after == before:
            raise RuntimeError("git commit did not advance HEAD")
        return after

    def _git_revert_last(self) -> None:
        """Revert the last commit, raising when the candidate stays on the tree."""
        git("revert", "--no-edit", "HEAD", cwd=self.ic.workspace_dir)

    def _git_discard_worktree(self) -> None:
        """Discard staged and unstaged tracked edits in the workspace."""
        self._git_discard_all_tracked_changes()

    def _git_discard_all_tracked_changes(self) -> None:
        """Discard an exact pending candidate from both index and worktree."""
        git(
            "restore",
            "--source=HEAD",
            "--staged",
            "--worktree",
            "--",
            ".",
            cwd=self.ic.workspace_dir,
        )

        listed = self._new_paths_best_effort()
        if listed is None:
            return
        admitted, refused = listed
        if self._pre_untracked is None:
            preexisting = []
            if admitted:
                print(f"  [git] no untracked snapshot; removing every allowlisted new file: {', '.join(admitted)}")
        else:
            preexisting = [path for path in admitted if path in self._pre_untracked]
            if preexisting:
                print("  [git] leaving allowlisted file(s) this loop did not create: " + ", ".join(preexisting))
        admitted = [path for path in admitted if path not in preexisting]
        if admitted:
            clean = git(
                "clean",
                "-f",
                "--",
                *admitted,
                cwd=self.ic.workspace_dir,
                check=False,
            )
            if clean.returncode != 0:
                raise RuntimeError(f"git clean failed: {(clean.stderr or clean.stdout).strip()}")
        self._report_refused_new_paths(refused, "removed", preexisting)

    def _read_source_file(self, path: str) -> str:
        """Read a source file's current content (best-effort)."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = Path(self.ic.workspace_dir) / p
            return p.read_text()
        except Exception as e:
            log.debug("could not read source file %s: %s", path, e)
            return ""

    def _read_kernel_source(self) -> str:
        """Read the anchor kernel file's current on-disk content (best-effort)."""
        return self._read_source_file(self.ic.kernel_file)

    def _kernel_source_for_scope(self) -> list[str | None] | None:
        """Every declared source file's text, None per file that would not read."""
        texts: list[str | None] = []
        unreadable: list[str] = []
        for declared in self._target_source_files():
            path = Path(declared)
            if not path.is_absolute():
                path = Path(self.ic.workspace_dir) / path
            try:
                texts.append(path.read_text())
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                texts.append(None)
                unreadable.append(str(path))
                log.warning(
                    "lessons: could not read %s; it will not be checked for held-fixed premises this round: %s", path, e
                )
        listed = ", ".join(unreadable)
        if any(text is not None for text in texts):
            if unreadable:
                # Printed, not only logged: a premise checked against part of the declared source is a weaker check
                # than the note reads as.
                print(f"  [lesson] source unreadable ({listed}): held-fixed premises checked against the rest only")
            return texts
        print(f"  [lesson] kernel source unreadable ({listed}): held-fixed premises not checked this round")
        return None

    def _target_source_files(self) -> list[str]:
        """Declared implementation hints, anchor first, de-duplicated."""
        files: list[str] = []
        for f in [self.ic.kernel_file, *self.ic.source_files]:
            if f and f not in files:
                files.append(f)
        return files

    def _jit_source_files(self) -> list[str]:
        """Declared hints plus actual tracked edits that may require recompilation."""

        workspace = getattr(
            self.ic,
            "workspace_dir",
            str(Path(self.ic.kernel_file).resolve().parent),
        )
        return list(
            dict.fromkeys(
                [
                    *self._target_source_files(),
                    *tracked_source_changes(workspace),
                ]
            )
        )

    def _full_diff(self, commit_hash: str) -> str:
        """Full unified diff of one iteration's commit (all files it touched)."""
        if not commit_hash:
            return ""
        try:
            return self._git("diff", f"{commit_hash}~1", commit_hash)
        except Exception as e:
            log.debug("could not diff commit %s: %s", commit_hash, e)
            return ""

    def _working_tree_diff(self) -> str:
        """Full diff of the current staged/unstaged candidate relative to HEAD."""
        return git("diff", "HEAD", "--", ".", cwd=self.ic.workspace_dir).stdout

    def _can_reuse_insession_benchmark(
        self,
        measurement: dict | None,
        *,
        attempt_diff: str,
    ) -> bool:
        """Return whether a gate measurement belongs to this exact candidate."""
        if not isinstance(measurement, dict) or not measurement.get("success"):
            return False
        if not attempt_diff.strip():
            return False
        if self.ic.build_command:
            return False
        if measurement.get("measurement_count") != KEEP_MEASUREMENT_COUNT:
            return False
        if len(measurement.get("measurements") or []) != KEEP_MEASUREMENT_COUNT:
            return False
        if measurement.get("bench_repeat") != self.ic.bench_repeat:
            return False
        expected_fingerprint = hashlib.sha256(attempt_diff.encode()).hexdigest()
        if measurement.get("candidate_diff_sha256") != expected_fingerprint:
            return False
        if measurement.get("driver_sha256") != self._driver_sha256():
            return False
        return (
            measurement.get("baseline_case_times") == self._baseline_case_times
            and measurement.get("best_mean_case_speedup") == self.best_mean_case_speedup
        )

    def _diff_summary_from_diff(self, diff: str, max_lines: int = 8) -> str:
        """Compact summary from an already-captured unified diff."""
        if not diff:
            return ""
        import re as _re

        files: list[str] = []
        for ln in diff.splitlines():
            if ln.startswith("diff --git "):
                parts = ln.split()
                if len(parts) >= 4:
                    name = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                    files.append(name)
        stat_lines = [f"{name} | changed" for name in files[:4]]
        signal = _re.compile(
            r"BLOCK_|VEC_|WARP|WAVE|tile|fastmath|const_expr|num_stage|num_warp|"
            r"occupancy|def |return |Vec\(|\.to\(|=|if ",
            _re.IGNORECASE,
        )
        changed: list[str] = []
        for ln in diff.splitlines():
            if ln[:3] in ("+++", "---"):
                continue
            if ln[:1] in "+-":
                content = ln[1:].strip()
                if not content or content.startswith("#"):
                    continue
                if signal.search(content):
                    changed.append(f"{ln[0]} {content[:100]}")
            if len(changed) >= max_lines:
                break
        parts: list[str] = []
        if stat_lines:
            parts.append("files: " + "; ".join(stat_lines[:4]))
        parts.extend(changed[:max_lines])
        return "\n".join(parts)

    def _diff_summary(self, commit_hash: str, max_lines: int = 8) -> str:
        """Mechanical, loop-authored summary of one iteration's NET change."""
        if not commit_hash:
            return ""
        diff = self._git("diff", f"{commit_hash}~1", commit_hash, "--", ".")
        return self._diff_summary_from_diff(diff, max_lines=max_lines)

    def _commit_changed_files(self, commit_hash: str) -> list[str]:
        """Tracked paths changed by one verified KEEP commit."""
        if not commit_hash:
            return []
        output = self._git(
            "diff",
            "--name-only",
            f"{commit_hash}~1",
            commit_hash,
            "--",
            ".",
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _publication_changed_files(self, commit_hash: str) -> list[str]:
        base = self.ic.campaign_base_commit
        if not base:
            return self._commit_changed_files(commit_hash)
        output = self._git(
            "diff",
            "--name-only",
            base,
            commit_hash,
            "--",
            ".",
        )
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _publication_patch(self, commit_hash: str) -> str:
        base = self.ic.campaign_base_commit
        if not base:
            return self._full_diff(commit_hash)
        return self._git("diff", base, commit_hash, "--", ".")

    def _publish_best_result(
        self,
        result: IterationResult,
        *,
        plan: str,
        best_before: float | None,
        pending: dict | None = None,
    ) -> bool:
        """Publish one KEEP before another Agent session may start."""
        if not result.kept or not result.commit_hash:
            return False
        baseline = (
            (pending or {}).get("pristine_baseline_wall_ms")
            or (pending or {}).get("baseline_wall_ms")
            or self.ic.pristine_baseline_wall_ms
            or self.ic.publication_baseline_wall_ms
            or self.ic.baseline_wall_ms
            or self.run_state.baseline_wall_ms
            or best_before
        )
        if baseline is None or result.wall_ms is None or result.mean_case_speedup is None:
            return False
        validation_text = (
            (pending or {}).get("validation_text") or result.validation_summary or "canonical validation passed"
        )
        if result.error_output and not pending:
            validation_text = f"{validation_text}\n\n{result.error_output}".strip()
        benchmark = dict((pending or {}).get("benchmark") or result.bench_detail or {})
        benchmark.setdefault("median_ms", result.wall_ms)
        benchmark.setdefault("mean_case_speedup", result.mean_case_speedup)
        try:
            self.best_publisher.publish(
                campaign_id=self.run_state.campaign_id,
                session_index=int((pending or {}).get("session_index", self.run_state.session_index)),
                experiment_id=(
                    str((pending or {}).get("experiment_id") or "")
                    or (self.experiment.experiment_id if self.experiment else "")
                ),
                iteration=result.iteration,
                commit_hash=result.commit_hash,
                plan=plan,
                baseline_wall_ms=baseline,
                search_start_ms=(self.ic.warm_start_wall_ms or self.ic.baseline_wall_ms),
                best_wall_ms=result.wall_ms,
                mean_case_speedup=result.mean_case_speedup,
                search_start_mean_case_speedup=(self.ic.warm_start_mean_case_speedup or 1.0),
                snr_db=result.snr_db,
                validation_text=validation_text,
                benchmark=benchmark,
                changed_files=(
                    list((pending or {}).get("publication_changed_files") or (pending or {}).get("changed_files") or [])
                    or self._publication_changed_files(result.commit_hash)
                ),
                patch=(
                    str((pending or {}).get("publication_patch") or (pending or {}).get("patch") or "")
                    or self._publication_patch(result.commit_hash)
                ),
                round_budget=self._round_budget_summary(),
            )
            return True
        except Exception as error:  # noqa: BLE001 - keep commit remains authoritative
            first_failure = not self.persistence_degraded
            self.persistence_degraded = True
            self.persistence_errors.append(f"publish best iteration {result.iteration}: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            # The first drop into degraded persistence is the one an operator can still act on; the 12-hour run buried
            # it at debug and the run looked healthy until the final report.
            if first_failure:
                log.warning("failed to publish best result", exc_info=True)
            else:
                log.debug("failed to publish best result", exc_info=True)
            return False

    def _finalize_keep_checkpoint(
        self,
        result: IterationResult,
        *,
        plan: str,
        best_before: float | None,
        pending: dict,
    ) -> None:
        """Durably finalize the compact state and event for one KEEP commit."""
        self._record_iteration_outcome(
            result,
            plan=plan,
            require_durable=True,
            checkpoint_metadata=pending,
        )

    def _archive_pending_keep(
        self,
        pending: dict,
        commit_hash: str,
        *,
        result: IterationResult | None = None,
    ) -> None:
        """Recover the candidate archive when a KEEP was interrupted post-commit."""
        iteration = int(pending["iteration"])
        existing = self.archive.load_meta(iteration)
        if existing:
            if existing.get("decision") != "KEEP" or existing.get("commit_hash") != commit_hash:
                raise ValueError(f"candidate archive conflicts with pending KEEP iteration {iteration}")
            return
        archived = self.archive.record(
            CandidateRecord(
                iteration=iteration,
                commit_hash=commit_hash,
                decision="KEEP",
                kept=True,
                validation_passed=True,
                wall_ms=pending.get("wall_ms"),
                mean_case_speedup=pending.get("mean_case_speedup"),
                bench_detail=pending.get("benchmark") or {},
                snr_db=pending.get("snr_db"),
                vgpr=pending.get("vgpr"),
                pmc_diagnosis=result.pmc_diagnosis if result else "",
                profile_meta=result.profile_meta if result else {},
                baseline_wall_ms=pending.get("baseline_wall_ms"),
                best_wall_ms_before=pending.get("best_wall_ms_before"),
                best_mean_case_speedup_before=pending.get("best_mean_case_speedup_before"),
                plan=str(pending.get("plan") or ""),
                rationale=str(pending.get("rationale") or ""),
                session_end_reason=str(pending.get("session_end_reason") or ""),
                turns=pending.get("turns"),
                kernel_file=str(pending.get("kernel_file") or self.ic.kernel_file),
                shape=pending.get("shape") or {},
                kernel_source=str(pending.get("kernel_source") or ""),
                change_diff=str(pending.get("patch") or ""),
                pmc_full=result.pmc_full if result else "",
                validation_text=str(pending.get("validation_text") or ""),
            )
        )
        if archived != self.archive._iter_dir(iteration):
            raise RuntimeError(f"failed to recover candidate archive for iteration {iteration}")

    def _pending_keep_result(
        self,
        pending: dict,
        commit_hash: str,
    ) -> IterationResult:
        """Rebuild the compact KEEP result represented by its journal."""
        return IterationResult(
            iteration=int(pending["iteration"]),
            duration_sec=0.0,
            validation_passed=True,
            validation_summary=str(pending.get("validation_text") or ""),
            wall_ms=pending.get("wall_ms"),
            mean_case_speedup=pending.get("mean_case_speedup"),
            snr_db=pending.get("snr_db"),
            vgpr=pending.get("vgpr"),
            kept=True,
            commit_hash=commit_hash,
            agent_rationale=str(pending.get("rationale") or ""),
            bench_detail=dict(pending.get("benchmark") or {}),
            session_end_reason=str(pending.get("session_end_reason") or ""),
            turns=pending.get("turns"),
        )

    @staticmethod
    def _require_matching_keep_event(
        event: dict,
        pending: dict,
        commit_hash: str,
    ) -> None:
        """Reject a KEEP event that does not describe the pending journal."""
        expected = {
            "decision": "KEEP",
            "commit_hash": commit_hash,
            "plan": str(pending.get("plan") or "").strip()[:120],
            "wall_ms": pending.get("wall_ms"),
            "mean_case_speedup": pending.get("mean_case_speedup"),
            "snr_db": pending.get("snr_db"),
            "session_end_reason": (str(pending.get("session_end_reason") or "") or None),
            "session_index": int(pending.get("session_index", 0) or 0),
            "experiment_id": str(pending.get("experiment_id") or "") or None,
            "turns": pending.get("turns"),
            "validation_passed": True,
            "is_new_best": True,
        }
        conflicts = [key for key, value in expected.items() if event.get(key) != value]
        if conflicts:
            raise ValueError("pending KEEP event payload mismatch: " + ", ".join(sorted(conflicts)))

    @staticmethod
    def _consecutive_no_changes(events: list[dict]) -> int:
        """Count the trailing run of empty diffs under the latest search mode."""
        streak = 0
        mode: str | None = None
        for event in reversed(events):
            if event.get("type") != "iteration_result":
                continue
            decision = str(event.get("decision") or "").strip().upper()
            if measured_nothing(decision):
                continue
            if decision != "NO_CHANGES":
                break
            recorded_mode = str(event.get("search_mode") or "")
            if mode is None:
                mode = recorded_mode
            elif recorded_mode != mode:
                break
            if not mode:
                break
            streak += 1
        return streak

    @staticmethod
    def _exploit_window_gain(
        events: list[dict],
        *,
        window: int,
        since_iteration: int,
    ) -> WindowGain:
        """Relative incumbent gain over the last full window of EXPLOIT outcomes."""
        scores: list[float] = []
        unavailable = "short_window"
        for event in reversed(events):
            if event.get("type") != "iteration_result":
                continue
            if int(event.get("iter", 0) or 0) <= since_iteration:
                break
            if str(event.get("search_mode") or "") != SEARCH_MODE_EXPLOIT:
                break
            decision = str(event.get("decision") or "").strip().upper()
            if measured_nothing(decision):
                continue
            score = event.get("best_after_mean_case_speedup")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                unavailable = "non_numeric_score"
                break
            score = float(score)
            if not math.isfinite(score):
                unavailable = "non_finite_score"
                break
            if score <= 0:
                unavailable = "non_positive_score"
                break
            scores.append(score)
            if len(scores) > window:
                break
        if len(scores) <= window:
            return WindowGain(ratio=None, unavailable=unavailable)
        anchor = scores[-1]
        return WindowGain(ratio=(scores[0] - anchor) / anchor, unavailable=None)

    async def _fan_out_round(
        self,
        *,
        iteration: int,
        orchestration_service,
        agent_factory,
        lanes: int | None = None,
    ) -> HeldRound | None:
        """Plan one round as lanes and run their sessions concurrently."""
        self._last_lane_plans = []
        # Reset here rather than at the ordinary path's planning call, which the round may now stand in for: a reused
        # plan must report its own executability and not the previous iteration's.
        self._last_orchestration_plan_executable = None
        plan_path: Path | None = None
        recovered = self._recoverable_lane_plans(iteration)
        if recovered is not None:
            planned_iteration, plans = recovered
            print(f"  [lanes] resuming {len(plans)} plans iteration {planned_iteration} paid for and never dispatched")
            self._last_lane_plans = plans
        else:
            width = max(1, int(self.ic.lanes if lanes is None else lanes))
            print(f"  [lanes] planning {width} concurrent Implementer lanes...")
            plan_path, error = await self._plan_round(
                iteration=iteration,
                orchestration_service=orchestration_service,
                lanes=width,
            )
            if plan_path is None:
                print(f"  [lanes] planning unavailable ({error}); falling back")
                return HeldRound(None, error)
        if len(self._last_lane_plans) < 2:
            print("  [lanes] one plan available; running the ordinary session")
            return HeldRound(plan_path, "")
        try:
            if recovered is not None:
                # Republished under the iteration that actually runs them, so this round is as recoverable as the one
                # it inherited from and an iteration's artifacts still describe what it did.
                plan_path = self._persist_lane_plans(
                    iteration,
                    self._last_lane_plans,
                    analysis_commit=self._canonical_commit(),
                )
                self._latest_optimization_plan_path = str(plan_path)
            # The round's plans exist and their cost is spent; what is priced here is only the sessions and the
            # measurement still to come.
            if not self._admit_dispatch(iteration):
                # The plans stay on disk and this iteration records no result, so the next session runs them instead
                # of buying them again.
                return HeldRound(plan_path, "")
            await self._fill_lane_queue(
                iteration=iteration,
                agent_factory=agent_factory,
                lane_plans=self._last_lane_plans,
            )
        except (OSError, RuntimeError) as error:
            self._lane_queue = []
            print(f"  [lanes] fan-out unavailable ({error}); falling back")
        # A recovered round that could not even republish holds nothing, and nothing was spent on it: that iteration
        # plans for itself as usual.
        return None if plan_path is None else HeldRound(plan_path, "")

    async def _fill_lane_queue(self, *, iteration: int = 0, agent_factory, lane_plans) -> None:
        """Run this round's lane sessions concurrently and queue what they wrote."""
        kernel_relative = self._workspace_path(self.ic.kernel_file)

        async def _session(
            lane: LanePlan,
            lane_dir: Path,
            serialized_driver: Path,
        ) -> None:
            # The session edits the lane's own copy, so it is handed that copy's kernel rather than the canonical one.
            await agent_factory(str(lane_dir), str(serialized_driver))(
                str(lane_dir / kernel_relative),
                _lane_prompt(lane.plan, serialized_driver=serialized_driver),
            )

        results = await run_lanes(
            workspace_dir=self.ic.workspace_dir,
            lanes=[LanePlan(lane_id=str(index + 1), plan=plan) for index, plan in enumerate(lane_plans)],
            session=_session,
            # Beside the workspace, not in /tmp: a lane copy carries the build outputs and the whole experiment
            # archive, and /tmp is typically a smaller local filesystem than the one sized for the campaign.
            parent_dir=str(Path(self.ic.workspace_dir).resolve().parent),
            driver=self._workspace_path(self.ic.driver_script),
        )
        for result in results:
            if result.error:
                print(f"  [lane {result.lane_id}] session lost: {result.error}")
        self._lane_queue = [item for item in results if item.produced_candidate]
        print(f"  [lanes] {len(self._lane_queue)} of {len(results)} lanes produced a candidate")
        self._persist_lane_queue()
        # The device is not per-lane.
        contended = [item for item in results if item.contended]
        if contended:
            hazard = self.device_hazard.record(
                iteration=iteration,
                detail="; ".join(f"lane {item.lane_id}: {item.reaped.describe()}" for item in contended),
                pids={pid for item in contended for pid in item.reaped.blockers},
            )
            print(
                f"  [lanes] {len(contended)} of {len(results)} lanes left the "
                f"device contended; this round measures nothing. "
                f"{hazard.describe()}"
            )

    def _unmeasurable_on_a_held_device(
        self,
        *,
        iteration: int,
        detail: str,
        session_sink: dict,
    ) -> IterationResult:
        """File an iteration that refused to measure on a device it does not own."""
        summary = (
            "REVERT (workspace contention): canonical correctness and benchmark "
            "were skipped because the device is still held by processes this "
            f"campaign could not clear. {detail}"
        )
        session_sink["findings"] = "\n---\n".join(
            part for part in (str(session_sink.get("findings") or ""), summary) if part
        )
        print("  [REVERT] Device still contended; nothing planned, run or measured this iteration")
        return IterationResult(
            iteration=iteration,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary=summary,
            kept=False,
            workspace_contention=detail,
        )

    def _lane_rejection(self, patch: str) -> str:
        """Why a lane candidate must not reach the canonical tree, or \"\"."""
        try:
            paths = _patch_paths(patch, cwd=self.ic.workspace_dir)
        except ValueError as error:
            return str(error)
        protected = sorted(
            path
            for path in paths
            if is_protected_path(
                path,
                workspace=self.ic.workspace_dir,
                # The campaign driver carries no protected name of its own.
                exact_paths=(self.ic.driver_script,),
            )
        )
        if protected:
            return "it changes the measurement surface: " + ", ".join(protected)
        return ""

    def _take_lane_candidate(self) -> LaneResult | None:
        """Apply the next queued lane candidate to the canonical tree."""
        try:
            return self._next_lane_candidate()
        finally:
            self._persist_lane_queue()

    def _next_lane_candidate(self) -> LaneResult | None:
        while self._lane_queue:
            lane = self._lane_queue.pop(0)
            rejection = self._lane_rejection(lane.diff)
            if rejection:
                print(f"  [lane {lane.lane_id}] candidate rejected: {rejection}")
                continue
            if not self._git_apply_patch(lane.diff):
                self._git_discard_worktree()
                print(f"  [lane {lane.lane_id}] candidate no longer applies; dropped")
                continue
            try:
                # The driver is the measurement boundary and must remain byte-for-byte canonical.
                self._validate_driver_integrity(self.run_state)
            except ValueError as error:
                # Take the candidate back off the tree before anything else, so the next candidate does not inherit
                # it.
                self._git_discard_worktree()
                self._validate_driver_integrity(self.run_state)
                print(f"  [lane {lane.lane_id}] candidate rejected: {error}")
                continue
            return lane
        return None

    def _git_apply_patch(self, patch: str) -> bool:
        """Apply one archived diff to the working tree, reporting whether it took."""
        if not patch.strip():
            return False
        handle = tempfile.NamedTemporaryFile("w", suffix=".diff", encoding="utf-8", delete=False)

        def _apply(*extra: str) -> bool:
            return (
                git(
                    "apply",
                    *extra,
                    handle.name,
                    cwd=self.ic.workspace_dir,
                    check=False,
                ).returncode
                == 0
            )

        try:
            handle.write(patch if patch.endswith("\n") else patch + "\n")
            handle.close()
            if _apply("--check") and _apply():
                return True
            if _apply("--3way"):
                return True
            self._git_discard_all_tracked_changes()
            return False
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def _select_merge_attempt(
        self,
    ) -> tuple[MergeCandidate, MergeCandidate] | None:
        """Two rejected gains worth measuring stacked, once single patches stall."""
        if not self.ic.merge_stacking:
            return None
        if self.run_state.stall.unresolved_stall_iters < MERGE_ATTEMPT_STALL_THRESHOLD:
            return None
        incumbent_case_times = self._scored_incumbent_case_times()
        if not incumbent_case_times:
            return None
        index = self.archive.load_index()
        metas = []
        for row in index:
            try:
                meta = self.archive.load_meta(int(row.get("iter") or 0))
            except Exception:  # noqa: BLE001 - a damaged record is not a candidate
                continue
            if meta:
                metas.append(meta)
        return select_merge_pair(
            eligible_candidates(metas, incumbent_case_times),
            already_attempted=(
                attempted_pairs([str(row.get("plan") or "") for row in index]) | frozenset(self._declined_merge_pairs)
            ),
        )

    # The one obstacle a later iteration clears on its own, and so the one the caller must not hold against the pair:
    # a tree carrying work is this iteration's accident, not a fact about two archived diffs.
    TREE_ALREADY_DIRTY_OBSTACLE = "the working tree already carried uncommitted work"

    def _merge_attempt_refusal(self) -> str:
        """Why a selected pair may not be measured this iteration, or \"\"."""
        if self._merge_precedence_streak >= MERGE_PRECEDENCE_STREAK_LIMIT:
            return (
                f"{self._merge_precedence_streak} stacked iterations have run "
                "back to back without the queue being reached"
            )
        return ""

    def _stage_merge_attempt(
        self,
        pair: tuple[MergeCandidate, MergeCandidate] | None,
    ) -> tuple[str, str]:
        """Put both candidates' diffs in the tree; the diff, or why there is none."""
        if pair is None:
            return "", ""
        if self._working_tree_diff().strip():
            return "", self.TREE_ALREADY_DIRTY_OBSTACLE
        for candidate in pair:
            try:
                patch = self.archive.read_candidate_file(candidate.iteration, "change.diff")
            except Exception:  # noqa: BLE001 - an unreadable diff is not stackable
                patch = ""
            if not str(patch or "").strip():
                # Reported apart from a conflict because the two ask for opposite responses.
                self._git_discard_worktree()
                return "", (f"iteration {candidate.iteration}'s archived diff is missing or unreadable")
            if not self._git_apply_patch(patch):
                self._git_discard_worktree()
                return "", (f"iteration {candidate.iteration}'s diff would not apply over the other's")
        staged = self._working_tree_diff()
        if not staged.strip():
            self._git_discard_worktree()
            return "", "both diffs applied but changed nothing against HEAD"
        return staged, ""

    def _decline_merge_attempt(
        self,
        iteration: int,
        pair: tuple[MergeCandidate, MergeCandidate],
        obstacle: str,
        *,
        about_the_iteration: bool = False,
    ) -> None:
        """Report a pair that was selected and not measured, and drop it or not."""
        print(f"  [merge] declined: {obstacle}")
        if not about_the_iteration and obstacle != self.TREE_ALREADY_DIRTY_OBSTACLE:
            self._declined_merge_pairs.add(frozenset({pair[0].iteration, pair[1].iteration}))
        self.state_store.append_event(
            make_event(
                "merge_attempt_declined",
                iteration,
                first_iteration=pair[0].iteration,
                second_iteration=pair[1].iteration,
                obstacle=obstacle,
            )
        )

    @staticmethod
    def _record_direction_verdict(
        state: RunState,
        *,
        iteration: int,
        decision_label: str,
        mean_case_speedup: float | None,
        best_mean_case_speedup: float | None,
        bench_detail: dict | None = None,
        incumbent_case_times: dict[str, float] | None = None,
    ) -> None:
        """Pin a rejected candidate that still measured faster than the incumbent."""
        if decision_label != "REVERT_PERF":
            return
        if beats_current_best(
            mean_case_speedup,
            best_mean_case_speedup=best_mean_case_speedup,
        ):
            pin_iteration(state, iteration)
            return
        detail = bench_detail if isinstance(bench_detail, dict) else {}
        if not incumbent_case_times or not detail:
            return
        if cases_beating_reference(
            dict(detail.get("case_times") or {}),
            incumbent_case_times,
            case_spreads(detail.get("measurements")),
        ):
            pin_iteration(state, iteration)

    def _apply_replayed_non_keep(self, state: RunState, event: dict) -> None:
        """Reduce one validated non-KEEP event without persisting state."""
        iteration = int(event["iter"])
        decision = str(event.get("decision") or "")
        if not decision or decision == "KEEP":
            raise ValueError(f"iteration {iteration} is not a replayable non-KEEP event")
        apply_iteration(
            state,
            iteration=iteration,
            decision=decision,
            kept=False,
            wall_ms=event.get("wall_ms"),
            mean_case_speedup=event.get("mean_case_speedup"),
            commit_hash=str(event.get("commit_hash") or ""),
            plan=str(event.get("plan") or ""),
            baseline_wall_ms=state.baseline_wall_ms,
            best_wall_ms=event.get("best_after_ms"),
            best_mean_case_speedup=event.get("best_after_mean_case_speedup"),
            stall_threshold=self.ic.supervise_after,
            orchestration_error_threshold=(self.ic.max_consecutive_orchestration_errors),
        )
        state.diversification_cycle_completed = event.get("diversification_cycle_completed") is True
        self._record_direction_verdict(
            state,
            iteration=iteration,
            decision_label=decision,
            mean_case_speedup=event.get("mean_case_speedup"),
            # A non-KEEP leaves the incumbent untouched, so the recorded post-decision score is the bar this candidate
            # had to clear.
            best_mean_case_speedup=event.get("best_after_mean_case_speedup"),
        )

    def _plan_resume_recovery(
        self,
        state: RunState,
        pending: dict | None,
    ) -> tuple[RunState, str, IterationResult | None, bool]:
        """Validate and reduce the complete contiguous recovery window."""
        planned = copy.deepcopy(state)
        events_by_iteration: dict[int, dict] = {}
        for event in self.state_store.read_events():
            if event.get("type") != "iteration_result":
                continue
            iteration = int(event["iter"])
            if iteration in events_by_iteration:
                raise ValueError(f"duplicate iteration_result events for iteration {iteration}")
            events_by_iteration[iteration] = event

        cursor = planned.next_iteration
        pending_iteration = int(pending.get("iteration", 0) or 0) if pending is not None else None
        if pending is not None and pending_iteration <= 0:
            raise ValueError("pending KEEP metadata is incomplete")

        forward_iterations = sorted(iteration for iteration in events_by_iteration if iteration >= cursor)
        boundary = pending_iteration
        for iteration in forward_iterations:
            if boundary is not None and iteration >= boundary:
                break
            if iteration != cursor:
                raise ValueError(f"iteration_result recovery gap: expected {cursor}, got {iteration}")
            event = events_by_iteration[iteration]
            if event.get("decision") == "KEEP":
                raise ValueError(f"uncheckpointed KEEP iteration {iteration} has no matching pending journal")
            self._apply_replayed_non_keep(planned, event)
            cursor = planned.next_iteration

        if pending is None:
            for iteration in forward_iterations:
                if iteration < cursor:
                    continue
                if iteration != cursor:
                    raise ValueError(f"iteration_result recovery gap: expected {cursor}, got {iteration}")
                event = events_by_iteration[iteration]
                if event.get("decision") == "KEEP":
                    raise ValueError(f"uncheckpointed KEEP iteration {iteration} has no pending journal")
                self._apply_replayed_non_keep(planned, event)
                cursor = planned.next_iteration
            return planned, "", None, False

        assert pending_iteration is not None
        already_applied = pending_iteration < cursor
        if not already_applied and pending_iteration != cursor:
            raise ValueError(
                f"iteration_result recovery gap before pending KEEP: expected {cursor}, got {pending_iteration}"
            )
        later_events = [iteration for iteration in forward_iterations if iteration > pending_iteration]
        if later_events:
            raise ValueError(
                f"iteration_result exists after pending KEEP iteration {pending_iteration}: {later_events[0]}"
            )

        status = self._inspect_pending_keep(planned, pending)
        keep_event = events_by_iteration.get(pending_iteration)
        if keep_event is not None and keep_event.get("decision") != "KEEP":
            raise ValueError(f"pending KEEP conflicts with iteration {pending_iteration} event")
        if status == "uncommitted":
            if keep_event is not None:
                raise ValueError(f"uncommitted pending KEEP iteration {pending_iteration} already has a KEEP event")
            return planned, status, None, False

        current_head = self._git("rev-parse", "HEAD").splitlines()[0]
        result = self._pending_keep_result(pending, current_head)
        if keep_event is not None:
            self._require_matching_keep_event(
                keep_event,
                pending,
                current_head,
            )
        if not already_applied:
            apply_iteration(
                planned,
                iteration=result.iteration,
                decision="KEEP",
                kept=True,
                wall_ms=result.wall_ms,
                mean_case_speedup=result.mean_case_speedup,
                commit_hash=result.commit_hash,
                plan=str(pending.get("plan") or ""),
                baseline_wall_ms=planned.baseline_wall_ms,
                best_wall_ms=result.wall_ms,
                best_mean_case_speedup=result.mean_case_speedup,
                stall_threshold=self.ic.supervise_after,
                orchestration_error_threshold=(self.ic.max_consecutive_orchestration_errors),
            )
            control = pending.get("search_control")
            if isinstance(control, dict):
                planned.diversification_cycle_completed = control.get("diversification_cycle_completed") is True
        elif (
            planned.best.iteration != result.iteration
            or planned.best.commit_hash != result.commit_hash
            or planned.best.wall_ms != result.wall_ms
            or planned.best.mean_case_speedup != result.mean_case_speedup
        ):
            raise ValueError(f"run state conflicts with KEEP iteration {result.iteration}")
        return planned, status, result, keep_event is None

    def _coordinate_resume_recovery(self, on_best_committed=None) -> None:
        """Replay, reconcile, and checkpoint one ordered recovery transaction."""
        pending = self._load_pending_keep()
        planned, pending_status, result, append_keep = self._plan_resume_recovery(self.run_state, pending)
        if pending_status == "uncommitted":
            if self._tracked_diff_from_head():
                self._git_discard_all_tracked_changes()
            if self._tracked_diff_from_head():
                raise RuntimeError("pending KEEP workspace remained dirty after restore")
            self._clear_pending_keep()

        head_out = self._git("rev-parse", "HEAD").strip()
        if head_out:
            planned.head_commit = head_out.splitlines()[0]
        self.run_state = planned
        if append_keep:
            assert pending is not None and result is not None
            self.state_store.append_event(
                self._iteration_result_event(
                    result,
                    plan=str(pending.get("plan") or ""),
                    checkpoint_metadata=pending,
                )
            )
        self.state_store.save(self.run_state)
        persisted = self.state_store.load()
        if persisted.to_dict() != self.run_state.to_dict():
            raise RuntimeError("resume recovery state was not persisted")

        if result is not None and pending is not None:
            self._promote_best(result)
            self.best_mean_case_speedup = result.mean_case_speedup
            if on_best_committed is not None:
                on_best_committed(result)
            self._recovered_pending_keep = (pending, result)

    async def _finish_recovered_pending_keep(self) -> None:
        """Rebuild optional post-KEEP views after critical recovery is safe."""
        recovered = self._recovered_pending_keep
        if recovered is None:
            return
        pending, result = recovered
        try:
            self._archive_pending_keep(
                pending,
                result.commit_hash,
                result=result,
            )
        except Exception as error:  # noqa: BLE001 - derived view is rebuildable
            self.persistence_degraded = True
            self.persistence_errors.append(f"rebuild candidate archive iteration {result.iteration}: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            log.debug("failed to rebuild recovered candidate archive", exc_info=True)
        self._publish_best_result(
            result,
            plan=str(pending.get("plan") or ""),
            best_before=pending.get("best_wall_ms_before"),
            pending=pending,
        )
        self._clear_pending_keep()
        self._recovered_pending_keep = None
        self._publish_optimization_history()
        self._checkpoint_llm_usage()

    def _reconcile_best_publication(self) -> None:
        """Repair manifest and derived best views from the durable run state."""
        best = self.run_state.best
        if not best.commit_hash or best.wall_ms is None or best.mean_case_speedup is None:
            return
        # A resumed session recomputes session_index and experiment_id, which legitimately differ from what the stored
        # manifest was written with, so republishing an already-current best tripped the same-iteration conflict guard
        # and reported persistence_degraded over a bundle that was already correct.
        if self.best_publisher.describes_current_best(
            iteration=best.iteration,
            commit_hash=best.commit_hash,
        ):
            return
        metadata = self.archive.load_meta(best.iteration)
        published: dict = {}
        publication_paths = [
            self.best_publisher.manifest_path,
            (self.best_publisher.best_root / f"iter_{best.iteration:03d}" / "publication.json"),
        ]
        for path in publication_paths:
            try:
                candidate = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            except Exception as error:
                raise ValueError(f"invalid best publication metadata: {path}") from error
            if (
                int(candidate.get("iteration", 0) or 0) == best.iteration
                and candidate.get("commit_hash") == best.commit_hash
            ):
                published = candidate
                break
        validation_text = (
            self.archive.read_candidate_file(
                best.iteration,
                "validation.txt",
            )
            or "canonical validation passed (recovered from run state)"
        )
        benchmark = dict(metadata.get("bench") or {})
        benchmark.setdefault("median_ms", best.wall_ms)
        benchmark.setdefault("mean_case_speedup", best.mean_case_speedup)
        published_patch = ""
        published_patch_path = str(published.get("patch_path") or "")
        if published_patch_path:
            try:
                published_patch = (self.best_publisher.root / published_patch_path).read_text()
            except OSError:
                published_patch = ""
        pending = {
            "session_index": published.get(
                "session_index",
                self.run_state.session_index,
            ),
            "experiment_id": published.get(
                "experiment_id",
                self.run_state.last_experiment_id,
            ),
            "baseline_wall_ms": published.get(
                "baseline_wall_ms",
                self.run_state.baseline_wall_ms,
            ),
            "validation_text": validation_text,
            "benchmark": benchmark,
            "changed_files": (published.get("changed_files") or self._publication_changed_files(best.commit_hash)),
            "patch": (published_patch or self._publication_patch(best.commit_hash)),
        }
        result = IterationResult(
            iteration=best.iteration,
            duration_sec=0.0,
            validation_passed=True,
            validation_summary=validation_text,
            wall_ms=best.wall_ms,
            mean_case_speedup=best.mean_case_speedup,
            snr_db=published.get("snr_db", metadata.get("snr_db")),
            kept=True,
            commit_hash=best.commit_hash,
            bench_detail=benchmark,
        )
        if not self._publish_best_result(
            result,
            plan=str(published.get("plan") or metadata.get("plan") or best.plan),
            best_before=None,
            pending=pending,
        ):
            log.debug(
                "best publication derived views remain unavailable for iteration %s",
                best.iteration,
            )

    def _publish_optimization_history(self) -> None:
        """Regenerate history from durable events and candidate metadata."""
        events = self.state_store.read_events()
        metadata: dict[int, dict] = {}
        for event in events:
            if event.get("type") != "iteration_result":
                continue
            iteration = int(event.get("iter", 0) or 0)
            candidate = self.archive.load_meta(iteration)
            if not candidate:
                continue
            candidate["archive_path"] = f"candidates/iter_{iteration:03d}/"
            candidate["change_diff"] = self.archive.read_candidate_file(
                iteration,
                "change.diff",
            )
            metadata[iteration] = candidate
        try:
            self.best_publisher.publish_history(
                events=events,
                candidate_metadata=metadata,
            )
        except Exception as error:  # noqa: BLE001 - structured history remains durable
            self.persistence_degraded = True
            self.persistence_errors.append(f"publish optimization history: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            log.debug("failed to publish optimization history", exc_info=True)

    def _time_remaining(self) -> float:
        """Seconds remaining in the budget."""
        elapsed = time.time() - self.start_time
        return max(0, self.ic.max_time_hours * 3600 - elapsed)

    def _analysis_deadline_unix(self) -> float:
        """Absolute Analysis deadline preserving iteration/finalization reserve."""
        now = time.time()
        started_at = self.start_time or now
        deadlines = [started_at + self.ic.max_time_hours * 3600 - self.ic.budget_reserve_sec]
        if self.ic.deadline_unix is not None:
            deadlines.append(self.ic.deadline_unix - self.ic.budget_reserve_sec)
        return max(now, min(deadlines))

    def _is_budget_exhausted(self) -> bool:
        """Whether remaining campaign time cannot admit another Agent session."""
        return self._time_remaining() < self.ic.budget_reserve_sec

    def _advance_campaign_clock(self) -> float:
        """Bring the campaign's cumulative wall-clock up to now, and return it."""
        costs = self.run_state.round_costs
        if self._campaign_started_at <= 0:
            return costs.campaign_sec
        costs.campaign_sec = max(
            costs.campaign_sec,
            max(0.0, time.time() - self._campaign_started_at),
        )
        return costs.campaign_sec

    def _open_round(self, iteration: int, *, lanes: int) -> None:
        """Start timing the round the budget has just admitted."""
        self._round_started_at = time.time()
        self._round_iteration = iteration
        self._round_lanes = max(1, int(lanes))
        self._round_planning_sec = 0.0
        self._round_measurement_sec = 0.0

    def _close_round(self) -> None:
        """Record what the open round cost, if it bought any planning."""
        started_at = self._round_started_at
        planning_sec = self._round_planning_sec
        measurement_sec = self._round_measurement_sec
        self._round_started_at = None
        self._round_planning_sec = 0.0
        self._round_measurement_sec = 0.0
        if started_at is None or planning_sec <= 0:
            return
        total_sec = max(planning_sec, time.time() - started_at)
        try:
            apply_round_cost(
                self.run_state,
                iteration=self._round_iteration,
                lanes=self._round_lanes,
                planning_sec=planning_sec,
                total_sec=total_sec,
                measurement_sec=measurement_sec,
                campaign_sec=self._advance_campaign_clock(),
            )
            self.state_store.append_event(
                make_event(
                    "round_cost",
                    self._round_iteration,
                    lanes=self._round_lanes,
                    planning_sec=round(planning_sec, 3),
                    total_sec=round(total_sec, 3),
                    measurement_sec=round(measurement_sec, 3),
                )
            )
            self.state_store.save(self.run_state)
        except Exception:  # noqa: BLE001 - best-effort
            log.debug("run_state: round cost record failed", exc_info=True)

    def _round_budget_summary(self) -> dict:
        """What the campaign's rounds have cost, for the published report."""
        costs = self.run_state.round_costs
        if not costs.rounds and not self._refused_round:
            return {}
        summary = {
            "rounds": costs.rounds,
            "planning_total_sec": round(costs.planning_total_sec, 3),
            "total_sec": round(costs.total_sec, 3),
            "campaign_sec": round(self._advance_campaign_clock(), 3),
        }
        share = costs.planning_share_pct()
        if share is not None:
            summary["planning_share_pct"] = round(share, 1)
        if self._refused_round:
            summary["refused"] = self._refused_round
        return summary

    def _observe_measurement(self, started_at: float) -> None:
        """Charge the open round for one canonical validate-and-benchmark cycle."""
        if self._round_started_at is None:
            return
        self._round_measurement_sec += max(0.0, time.time() - started_at)

    def _measurement_estimate_sec(self) -> float:
        """Wall-clock the canonical validation and benchmark may still take."""
        return estimate_measurement_sec(list(self.run_state.round_costs.recent))

    def _admit_next_round(self, iteration: int) -> int | None:
        """How many lanes the next round may PLAN, or ``None`` if none fit."""
        decision = admit_round(
            remaining_sec=self._time_remaining(),
            requested_lanes=self.ic.lanes,
            history=list(self.run_state.round_costs.recent),
            measurement_sec=self._measurement_estimate_sec(),
        )
        if decision.admitted and not decision.narrowed:
            return decision.lanes
        event_fields = {
            "lanes": decision.lanes,
            "requested_lanes": max(1, int(self.ic.lanes)),
            "remaining_sec": round(decision.remaining_sec, 3),
            "required_sec": round(decision.required_sec, 3),
            "planning_sec": round(decision.planning_sec, 3),
            "execution_sec": round(decision.execution_sec, 3),
        }
        if decision.admitted:
            print(f"  [budget] round narrowed to {decision.lanes} lane(s): {decision.summary()}")
        else:
            self._refuse_round(iteration, decision.summary())
        try:
            self.state_store.append_event(
                make_event(
                    "round_admission",
                    iteration,
                    admitted=decision.admitted,
                    **event_fields,
                )
            )
        except Exception:  # noqa: BLE001 - best-effort
            log.debug("run_state: round admission append failed", exc_info=True)
        return decision.lanes if decision.admitted else None

    def _admit_dispatch(self, iteration: int) -> bool:
        """Whether the round now holding plans may start its session."""
        decision = admit_dispatch(
            remaining_sec=self._time_remaining(),
            measurement_sec=self._measurement_estimate_sec(),
        )
        try:
            self.state_store.append_event(
                make_event(
                    "round_dispatch",
                    iteration,
                    admitted=decision.admitted,
                    remaining_sec=round(decision.remaining_sec, 3),
                    required_sec=round(decision.required_sec, 3),
                    session_sec=round(decision.session_sec, 3),
                    measurement_sec=round(decision.measurement_sec, 3),
                    # Recorded because it is the one case where the parts do not add up to the requirement: this
                    # campaign estimated less than the external-timeout floor and was held at it.
                    floored=decision.floored,
                )
            )
        except Exception:  # noqa: BLE001 - best-effort
            log.debug("run_state: round dispatch append failed", exc_info=True)
        if decision.admitted:
            return True
        self._refuse_round(iteration, decision.summary())
        return False

    def _refuse_round(self, iteration: int, summary: str) -> None:
        """End the campaign on a round the remaining budget cannot pay for."""
        self._refused_round = summary
        self.termination_reason = "round_budget_exhausted"
        print(f"\nROUND REFUSED FOR BUDGET at iteration {iteration}: {summary}")

    def _is_force_stopped(self) -> bool:
        """Whether the operator requested an early stop via <workspace>/.stop."""
        return (Path(self.ic.workspace_dir) / ".stop").exists()

    def _is_gate_met(self) -> bool:
        """Check if performance target is met."""
        if self.ic.target_wall_ms is None or self.best_wall_ms is None:
            return False
        return self.best_wall_ms <= self.ic.target_wall_ms

    async def _measure_baseline(self) -> float | None:
        """Bench the pristine kernel before any agent edit — the speedup anchor."""
        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *err_wrap(list(self.ic.build_command)),
                cwd=self.ic.build_dir or self.ic.workspace_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await communicate_process_group(
                proc,
                timeout=self.ic.build_timeout_sec,
            )
            if proc.returncode != 0:
                print(f"  Baseline build FAILED: {_build_failure_tail(stdout, stderr, 300)}")
                return None
        bench_result = await measure_wallclock(
            driver_script=self.ic.driver_script,
            driver_args=[],
            measurements=KEEP_MEASUREMENT_COUNT,
            timeout_sec=self.ic.bench_timeout_sec,
            repeat=self.ic.bench_repeat,
        )
        if not bench_result.get("success"):
            print("  Baseline bench FAILED: " + _bench_failure_detail(bench_result))
            return None
        baseline_case_times = dict(bench_result.get("case_times") or {})
        if not baseline_case_times:
            print(
                "  Baseline bench FAILED: the driver ran but printed no "
                "'case_ms: <case_id> <ms>' line: " + _bench_failure_detail(bench_result)
            )
            return None
        if bench_result.get("median_ms") is None:
            print(
                "  Baseline bench FAILED: the driver printed per-case timings "
                "but no aggregate 'median_ms:'/'mean_ms:' line: " + _bench_failure_detail(bench_result)
            )
            return None
        self.last_case_bandwidth = dict(bench_result.get("case_bandwidth") or {})
        unscored_cases = set(bench_result.get("unscored_cases") or [])
        try:
            baseline_score = calculate_mean_case_speedup(
                baseline_case_times,
                self._baseline_case_times or baseline_case_times,
                unscored_cases,
            )
        except CaseCoverageError as error:
            print(f"  Baseline bench FAILED: {error}")
            return None
        if baseline_score is None:
            print("  Baseline bench FAILED: mean case speedup is unavailable")
            return None

        if not self._baseline_case_times:
            self._baseline_case_times = dict(baseline_case_times)
            self.ic.baseline_case_times = dict(baseline_case_times)
        self._best_case_times = dict(baseline_case_times)
        self._unscored_cases = set(unscored_cases)
        self._persist_scoring_state()
        return bench_result.get("median_ms")

    def _promote_best(self, result: IterationResult) -> None:
        """Make a kept candidate's aggregate case medians the new incumbent."""
        self.best_wall_ms = result.wall_ms
        detail = result.bench_detail or {}
        cases = detail.get("case_times") or {}
        if cases:
            self._best_case_times = dict(cases)
        self._persist_scoring_state()

    def _set_baseline_case_times(self, case_times: dict | None) -> None:
        """Record (once) the pristine per-case wall times and persist them."""
        if not case_times or self._baseline_case_times:
            return
        self._baseline_case_times = dict(case_times)
        try:
            self.run_state.baseline_case_times = dict(case_times)
            self.state_store.save(self.run_state)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            self.persistence_degraded = True
            self.persistence_errors.append("persist pristine baseline case timings")
            self.persistence_errors = self.persistence_errors[-10:]
            log.warning(
                "run_state: failed to persist pristine baseline case timings",
                exc_info=True,
            )

    def _persist_scoring_state(self) -> None:
        """Checkpoint the state that decides keep/revert."""
        try:
            self.run_state.baseline_case_times = dict(self._baseline_case_times)
            self.run_state.best_case_times = dict(self._best_case_times)
            self.run_state.unscored_cases = sorted(self._unscored_cases)
            self.state_store.save(self.run_state)
        except Exception:  # noqa: BLE001 - persistence is best-effort
            self.persistence_degraded = True
            self.persistence_errors.append("persist scoring state")
            self.persistence_errors = self.persistence_errors[-10:]
            log.warning("run_state: failed to persist scoring state", exc_info=True)

    def _restore_scoring_state(self) -> None:
        """Rehydrate the keep/revert state recorded by a previous session."""
        state = self.run_state
        if state.best_case_times:
            self._best_case_times = dict(state.best_case_times)
        if state.unscored_cases:
            self._unscored_cases = {str(case_id) for case_id in state.unscored_cases}
        self._scoring_state_restored = True
        if state.best_case_times:
            print(f"  [run-state] restored scoring state: {len(self._best_case_times)} case(s)")

    def _apply_mean_case_speedup_metric(self, bench_result: dict | None) -> None:
        """Attach three pristine-relative scores and their mean."""
        if not isinstance(bench_result, dict):
            return
        try:
            measurement_scores = calculate_measurement_case_speedups(
                bench_result,
                self._baseline_case_times,
                expected_measurements=KEEP_MEASUREMENT_COUNT,
            )
        except CaseCoverageError as error:
            bench_result["success"] = False
            bench_result["mean_case_speedup"] = None
            bench_result["measurement_mean_case_speedups"] = []
            bench_result["message"] = f"CASE COVERAGE FAILED: {error}"
            bench_result["case_coverage_complete"] = False
            return
        mean_case_speedup = keep_score(measurement_scores)
        if mean_case_speedup is None:
            bench_result["success"] = False
            bench_result["mean_case_speedup"] = None
            bench_result["message"] = "MEAN CASE SCORING FAILED: pristine per-case timings are unavailable"
            bench_result["case_coverage_complete"] = False
            return
        bench_result["mean_case_speedup"] = mean_case_speedup
        bench_result["measurement_mean_case_speedups"] = measurement_scores
        bench_result["case_coverage_complete"] = True

    def _scored_incumbent_case_times(self) -> dict[str, float]:
        """The incumbent's per-case times, restricted to the scored cases."""
        scored = set(self._scored_case_ids())
        return {
            case_id: float(time_ms)
            for case_id, time_ms in (self._best_case_times or {}).items()
            if case_id in scored and isinstance(time_ms, (int, float)) and float(time_ms) > 0.0
        }

    def _scored_baseline_case_times(self, bench_result: dict) -> dict[str, float]:
        """The pristine per-case times the objective's mean is actually taken over."""
        excluded: set[str] = set()
        for measurement in bench_result.get("measurements") or ():
            if isinstance(measurement, dict):
                excluded.update(str(case_id) for case_id in (measurement.get("unscored_cases") or ()))
        return {
            case_id: float(baseline_ms)
            for case_id, baseline_ms in self._baseline_case_times.items()
            if case_id not in excluded and isinstance(baseline_ms, (int, float)) and float(baseline_ms) > 0.0
        }

    async def _resolve_keep_sigma(
        self,
        bench_result: dict,
        measurement_scores: list[float],
    ) -> SigmaResolution:
        """Estimate the objective's sigma from the case that supplies it."""
        measured = measurement_sigma(measurement_scores)
        idle = SigmaResolution(
            sigma=measured,
            measured_sigma=measured,
            dominant_case=None,
            variance_share=None,
            wall_share=None,
            rounds=0,
            sample_size=len(measurement_scores),
            unstable=False,
        )
        if measured is None or not bench_result.get("success"):
            return idle
        baseline = self._scored_baseline_case_times(bench_result)
        series = {
            case_id: list(times)
            for case_id, times in _measurement_case_times(bench_result).items()
            if case_id in baseline
        }
        base = attribute_sigma(series, baseline)
        if base is None:
            return replace(idle, detail="per-case times resolve no spread to attribute")
        if base.dominant_case is None:
            return idle

        dominant = base.dominant_case
        found = replace(
            idle,
            dominant_case=dominant,
            variance_share=base.variance_shares[dominant],
            wall_share=base.wall_shares[dominant],
            sample_size=base.sample_size,
        )
        incumbent = self.best_mean_case_speedup or 1.0
        if not beats_current_best(
            keep_score(measurement_scores),
            best_mean_case_speedup=incumbent,
        ):
            return replace(found, detail="reverted at every sigma")
        if passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=incumbent,
            sigma=measured,
            sigma_sample_size=base.sample_size,
        ):
            return replace(found, detail="kept at the measured sigma")

        current = base
        rounds = 0
        stopped = ""
        while rounds < SIGMA_REMEASURE_MAX_ROUNDS and current.dominant_case is not None:
            remeasure_started = time.time()
            extra = await measure_wallclock(
                driver_script=self.ic.driver_script,
                driver_args=[],
                measurements=SIGMA_REMEASURE_BATCH,
                timeout_sec=self.ic.bench_timeout_sec,
                repeat=self.ic.bench_repeat,
            )
            self._observe_measurement(remeasure_started)
            rounds += 1
            extra_series = _measurement_case_times(extra if isinstance(extra, dict) else {})
            if not (isinstance(extra, dict) and extra.get("success")):
                stopped = "re-measure bench failed"
                break
            if set(series) - set(extra_series):
                stopped = "re-measure lost a scored case"
                break
            for case_id in series:
                series[case_id].extend(extra_series[case_id])
            refreshed = attribute_sigma(series, baseline)
            if refreshed is None:
                stopped = "re-measure produced no usable spread"
                break
            current = refreshed

        if current is base:
            return replace(found, rounds=rounds, detail=stopped)
        share = current.variance_shares.get(dominant)
        # A case's *share* of the variance is structural -- q61's speedup term is 8x the others', so it holds most of
        # the variance at any sample size -- and re-measuring cannot be expected to move it.
        settled = current.case_sigmas[dominant] < base.case_sigmas[dominant]
        return replace(
            found,
            sigma=rescaled_sigma(measured, base, current),
            variance_share=share if share is not None else found.variance_share,
            wall_share=current.wall_shares.get(dominant, found.wall_share),
            rounds=rounds,
            sample_size=current.sample_size,
            unstable=current.dominant_case is not None and not settled,
            detail=stopped,
        )

    def _case_move_rule(
        self,
        before: float,
        measured: float,
        per_run: tuple[float, ...] | list[float],
    ) -> str | None:
        """Which rule, if any, admits one KEEP's move on one case as real."""
        if before <= 0 or measured <= 0:
            return None
        move = before - measured
        if move / before < self.ic.config_coverage_min_move_ratio:
            return None
        times = [float(value) for value in per_run if float(value) > 0]
        if len(times) < 2:
            return "floor"
        if max(times) >= before:
            return None
        spread = max(times) - min(times)
        if move >= CONFIG_COVERAGE_DISPERSION_MULTIPLE * spread:
            return "dispersion"
        return None

    def _case_config_coverage(self) -> CaseConfigCoverage:
        """Read per-case configuration coverage off this session's KEEPs."""
        scored = self._scored_case_ids()
        previous = dict(self._baseline_case_times)
        moved_by: dict[str, list[int]] = {case_id: [] for case_id in scored}
        # Covered cases whose every admitting KEEP was admitted by the floor alone.
        dispersion_tested: set[str] = set()
        keeps: list[int] = []
        unreadable: list[int] = []
        unmeasured = set(scored)
        for result in self.results:
            if not result.kept:
                continue
            detail = result.bench_detail or {}
            case_times = dict(detail.get("case_times") or {})
            per_run = _measurement_case_times(detail)
            if not case_times:
                # A KEEP replayed from a pending record can arrive without its per-case timings.
                unreadable.append(result.iteration)
                continue
            keeps.append(result.iteration)
            for case_id in scored:
                measured = case_times.get(case_id)
                before = previous[case_id]
                if not measured:
                    continue
                unmeasured.discard(case_id)
                rule = self._case_move_rule(
                    before,
                    float(measured),
                    per_run.get(case_id, ()),
                )
                if rule is not None:
                    moved_by[case_id].append(result.iteration)
                    if rule == "dispersion":
                        dispersion_tested.add(case_id)
            previous.update(case_times)

        if not keeps:
            # Before the first readable KEEP there is nothing to read coverage off.
            return CaseConfigCoverage(
                covered={},
                fallback=(),
                undifferentiated=(),
                keeps=(),
                unmeasured=(),
                unreadable=tuple(unreadable),
                floor_only=(),
            )

        groups: dict[tuple[int, ...], list[str]] = {}
        for case_id, iterations in moved_by.items():
            if iterations:
                groups.setdefault(tuple(iterations), []).append(case_id)
        return CaseConfigCoverage(
            covered={case_id: iterations[-1] for case_id, iterations in moved_by.items() if iterations},
            fallback=tuple(case_id for case_id in scored if not moved_by[case_id] and case_id not in unmeasured),
            undifferentiated=tuple(
                tuple(sorted(members)) for _signature, members in sorted(groups.items()) if len(members) > 1
            ),
            keeps=tuple(keeps),
            unmeasured=tuple(sorted(unmeasured)),
            unreadable=tuple(unreadable),
            floor_only=tuple(
                sorted(
                    case_id
                    for case_id, iterations in moved_by.items()
                    if iterations and case_id not in dispersion_tested
                )
            ),
        )

    def _case_config_coverage_flags(self) -> dict[str, tuple[str, ...]]:
        """Per-case coverage flags for the planning context's case evidence."""
        coverage = self._case_config_coverage()
        flags: dict[str, list[str]] = {}
        for case_id, iteration in coverage.covered.items():
            flags.setdefault(case_id, []).append(f"config_coverage_keep_{iteration}")
        for case_id in coverage.floor_only:
            # Covered, but on the floor ratio alone.
            flags.setdefault(case_id, []).append("config_coverage_floor_only")
        for case_id in coverage.fallback:
            flags.setdefault(case_id, []).append("config_coverage_fallback")
        for case_id in coverage.unmeasured:
            flags.setdefault(case_id, []).append("config_coverage_unmeasured")
        for group in coverage.undifferentiated:
            for case_id in group:
                flags.setdefault(case_id, []).append("config_coverage_undifferentiated")
        if coverage.unreadable:
            # Every other flag on this ledger was read off a partial record, so the planner is told which cases were
            # classified without it rather than being handed the classification alone.
            for case_id in self._scored_case_ids():
                flags.setdefault(case_id, []).append("config_coverage_partial_record")
        return {case_id: tuple(dict.fromkeys(values)) for case_id, values in flags.items()}

    def _with_case_config_coverage(self, context):
        """Attach measured configuration coverage to a planning context."""
        flags = self._case_config_coverage_flags()
        if not flags:
            return context
        return replace(
            context,
            cases=tuple(
                replace(
                    case,
                    flags=tuple(dict.fromkeys([*case.flags, *flags.get(case.case_id, ())])),
                )
                for case in context.cases
            ),
        )

    def _render_case_config_coverage(self) -> str:
        """Render the configuration-coverage ledger for the Implementer."""
        coverage = self._case_config_coverage()
        scored = self._scored_case_ids()
        if not scored:
            return ""
        lines = [
            "## Per-case configuration coverage (measured, this session)",
            (
                "A scored case counts as covered once some KEEP improved its "
                "measured time by at least "
                f"{self.ic.config_coverage_min_move_ratio:.1%}. "
                "What was established beyond that depends on the record, and "
                "each covered case below says which. A case no KEEP has "
                "improved has never had a configuration chosen for it: it is "
                "running on whatever generic path the canonical falls "
                "through to, and the suite mean averages it in at 1.00x "
                "without saying so."
            ),
        ]
        if coverage.unreadable:
            lines.append(
                "INCOMPLETE RECORD: KEEP iteration(s) "
                + ", ".join(str(iteration) for iteration in coverage.unreadable)
                + " carried no per-case timings, so nothing below accounts "
                "for what they changed. Treat every case listed as uncovered "
                "as unconfirmed until a KEEP with per-case timings lands."
            )
        if not coverage.keeps:
            lines.append(
                "No KEEP with per-case timings is on this session's record, "
                "so no scored case has been shown to own a configuration "
                "yet: " + ", ".join(scored)
            )
            return "\n".join(lines)
        lines.append(
            "Read off KEEP iteration(s) "
            + ", ".join(str(iteration) for iteration in coverage.keeps)
            + ". A resumed campaign restarts this record, so earlier "
            "sessions are absent from it rather than counted as uncovered."
        )
        lines.append(
            f"Covered {len(coverage.covered)}/{len(scored)}: "
            + (
                ", ".join(f"{case_id} (iter {iteration})" for case_id, iteration in sorted(coverage.covered.items()))
                or "(none)"
            )
        )
        strongly_covered = [case_id for case_id in sorted(coverage.covered) if case_id not in coverage.floor_only]
        if strongly_covered:
            lines.append(
                "Faster in every independent measurement of the KEEP that "
                "covered them, by more than those measurements disagree "
                "among themselves: " + ", ".join(strongly_covered)
            )
        if coverage.floor_only:
            lines.append(
                "Admitted by the "
                f"{self.ic.config_coverage_min_move_ratio:.1%} floor alone -- "
                "the KEEP that moved them carried no per-measurement detail, "
                "so their run-to-run spread was never tested and only the "
                "size of the move is established: " + ", ".join(coverage.floor_only)
            )
        if coverage.fallback:
            lines.append("No configuration of its own: " + ", ".join(coverage.fallback))
        if coverage.unmeasured:
            lines.append("Coverage unknown -- no KEEP on record timed them: " + ", ".join(coverage.unmeasured))
        for group in coverage.undifferentiated:
            lines.append(
                "Never distinguished by any KEEP, so one configuration currently serves them all: " + ", ".join(group)
            )
        return "\n".join(lines)

    def _seed_and_hydrate_run_state(self) -> None:
        """Seed a fresh session or hydrate an explicitly validated resume."""
        try:
            head_out = self._git("rev-parse", "HEAD").strip()
            head = head_out.splitlines()[0] if head_out else ""
            if self.resume and self.run_state.baseline_case_times:
                self._baseline_case_times = dict(self.run_state.baseline_case_times)
            if self.resume and should_resume(self.run_state, head):
                self.best_wall_ms = self.run_state.best.wall_ms
                self.best_mean_case_speedup = self.run_state.best.mean_case_speedup
                print(
                    f"  [run-state] resumed best from {self.run_state.best.commit_hash[:8]}: "
                    f"mean case speedup={self.run_state.best.mean_case_speedup:.6f}x, "
                    f"raw mean={self.run_state.best.wall_ms} ms "
                    f"(iter {self.run_state.best.iteration})"
                )
            if self.monitor is not None:
                self.monitor.no_improve_streak = self.run_state.stall.no_improvement_iters
                self.monitor.last_intervention_iter = self.run_state.stall.last_supervisor_iter or -10_000
                self.monitor.last_attempt_iter = self.run_state.stall.last_supervisor_attempt_iter or -10_000
                self.monitor.intervention_count = self.run_state.intervention_count
            if self.resume:
                # Without this the resumed session re-derives its own noise floor, incumbents and SNR reference, so it
                # judges candidates by different rules than the session it continues.
                if not self._scoring_state_restored:
                    self._restore_scoring_state()
            if self.ic.baseline_wall_ms is not None:
                self.run_state.baseline_wall_ms = self.ic.baseline_wall_ms
            if self._baseline_case_times:
                self.run_state.baseline_case_times = dict(self._baseline_case_times)
            if self.ic.pristine_baseline_wall_ms is not None:
                self.run_state.pristine_baseline_wall_ms = self.ic.pristine_baseline_wall_ms
            if not self.resume:
                self.state_store.append_event(
                    make_event(
                        "baseline_measured",
                        0,
                        baseline_wall_ms=self.ic.baseline_wall_ms,
                        mean_case_speedup=1.0,
                    )
                )
            self.state_store.save(self.run_state)
        except Exception:  # noqa: BLE001 - best-effort; never break the loop
            log.debug("run_state: seed/hydrate failed", exc_info=True)

    def _validate_pre_published_warm_start(
        self,
        *,
        commit_hash: str,
        baseline_ms: float,
        best_ms: float,
        mean_case_speedup: float,
    ) -> bool:
        """Validate the CLI's kill-recoverable warm-start publication."""
        publication = self.ic.warm_start_publication
        if not publication:
            return False

        try:
            manifest_path = Path(str(publication["best_manifest"]))
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("best manifest is not an object")
            checks = (
                int(publication.get("best_iteration", -1)) == 0,
                str(publication.get("best_commit") or "") == commit_hash,
                float(publication.get("baseline_ms")) == float(baseline_ms),
                float(publication.get("best_ms")) == float(best_ms),
                float(publication.get("mean_case_speedup")) == float(mean_case_speedup),
                int(manifest.get("iteration", -1)) == 0,
                str(manifest.get("commit_hash") or "") == commit_hash,
                float(manifest.get("baseline_wall_ms")) == float(baseline_ms),
                float(manifest.get("best_wall_ms")) == float(best_ms),
                float(manifest.get("mean_case_speedup")) == float(mean_case_speedup),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError("pre-published warm-start best artifact is unreadable") from error
        if not all(checks):
            raise RuntimeError("pre-published warm-start best artifact does not match the validated workspace state")
        return True

    def _stage_validated_warm_start_state(self) -> None:
        """Make iteration-zero warm-start state durable on the first save."""
        if self.resume or not self.ic.warm_start_commit:
            return
        head = self._git("rev-parse", "HEAD").strip()
        if head != self.ic.warm_start_commit:
            raise RuntimeError("validated warm-start commit is not the current workspace HEAD")
        if (
            self.ic.baseline_wall_ms is None
            or self.ic.pristine_baseline_wall_ms is None
            or self.ic.warm_start_wall_ms is None
            or self.ic.warm_start_mean_case_speedup is None
        ):
            raise RuntimeError("validated warm-start is missing performance baselines")
        warm_start_bench = dict(self.ic.warm_start_bench or {})
        self.run_state.best = BestRecord(
            iteration=0,
            wall_ms=self.ic.warm_start_wall_ms,
            mean_case_speedup=self.ic.warm_start_mean_case_speedup,
            commit_hash=head,
            plan=f"KB warm-start {self.ic.warm_start_solution_slug}".strip(),
            source="warm_start",
        )
        self.run_state.head_commit = head
        self.run_state.baseline_wall_ms = self.ic.baseline_wall_ms
        self.run_state.pristine_baseline_wall_ms = self.ic.pristine_baseline_wall_ms
        self.run_state.baseline_case_times = dict(self.ic.baseline_case_times)
        self.run_state.best_case_times = dict(warm_start_bench.get("case_times") or {})
        self.run_state.unscored_cases = [str(case_id) for case_id in (warm_start_bench.get("unscored_cases") or [])]

    def _adopt_validated_warm_start(self) -> None:
        """Persist an applied KB seed as the recoverable local best at iteration 0."""
        if self.resume or not self.ic.warm_start_commit:
            return
        head = self._git("rev-parse", "HEAD").strip()
        if head != self.ic.warm_start_commit:
            raise RuntimeError("validated warm-start commit is not the current workspace HEAD")
        if (
            self.ic.baseline_wall_ms is None
            or self.ic.pristine_baseline_wall_ms is None
            or self.ic.warm_start_wall_ms is None
            or self.ic.warm_start_mean_case_speedup is None
        ):
            raise RuntimeError("validated warm-start is missing performance baselines")
        incumbent_wall_ms = self.ic.warm_start_wall_ms
        incumbent_mean_case_speedup = self.ic.warm_start_mean_case_speedup
        self.best_wall_ms = incumbent_wall_ms
        self.best_mean_case_speedup = incumbent_mean_case_speedup
        warm_start_bench = dict(self.ic.warm_start_bench or {})
        if not self._best_case_times:
            self._best_case_times = dict(warm_start_bench.get("case_times") or {})
        if not self._unscored_cases:
            self._unscored_cases = {str(case_id) for case_id in (warm_start_bench.get("unscored_cases") or [])}
        self.run_state.best = BestRecord(
            iteration=0,
            wall_ms=incumbent_wall_ms,
            mean_case_speedup=incumbent_mean_case_speedup,
            commit_hash=head,
            plan=f"KB warm-start {self.ic.warm_start_solution_slug}".strip(),
            source="warm_start",
        )
        self.run_state.head_commit = head
        self.state_store.append_event(
            make_event(
                "warm_start_adopted",
                0,
                commit_hash=head,
                solution_slug=self.ic.warm_start_solution_slug,
                pristine_baseline_ms=self.ic.pristine_baseline_wall_ms,
                search_start_ms=incumbent_wall_ms,
                mean_case_speedup=incumbent_mean_case_speedup,
            )
        )
        self.state_store.save(self.run_state)
        self._persist_scoring_state()
        persisted = self.state_store.load()
        if (
            persisted.best.commit_hash != head
            or persisted.best.wall_ms != incumbent_wall_ms
            or persisted.best.mean_case_speedup != incumbent_mean_case_speedup
        ):
            raise RuntimeError("validated warm-start best state was not persisted")
        if self._validate_pre_published_warm_start(
            commit_hash=head,
            baseline_ms=self.ic.pristine_baseline_wall_ms,
            best_ms=self.ic.warm_start_wall_ms,
            mean_case_speedup=self.ic.warm_start_mean_case_speedup,
        ):
            return
        result = IterationResult(
            iteration=0,
            duration_sec=0.0,
            validation_passed=True,
            validation_summary="KB warm-start passed canonical correctness and performance gates",
            wall_ms=incumbent_wall_ms,
            mean_case_speedup=incumbent_mean_case_speedup,
            kept=True,
            commit_hash=head,
            bench_detail={
                **warm_start_bench,
                "case_times": dict(self._best_case_times),
                "unscored_cases": sorted(self._unscored_cases),
                "median_ms": incumbent_wall_ms,
                "mean_case_speedup": incumbent_mean_case_speedup,
            },
        )
        if not self._publish_best_result(
            result,
            plan=self.run_state.best.plan,
            best_before=self.ic.pristine_baseline_wall_ms,
        ):
            raise RuntimeError("failed to publish validated warm-start best artifact")

    def _record_iteration_outcome(
        self,
        result: IterationResult,
        *,
        plan: str = "",
        decision_label: str | None = None,
        require_durable: bool = False,
        checkpoint_metadata: dict | None = None,
    ) -> bool:
        """Synchronize one completed attempt into live and durable control state."""
        plan = (plan or "").strip()

        try:
            if decision_label is None:
                decision_label = _decision_label(result)

            error_sig = ""
            if not result.validation_passed:
                blob = getattr(result, "error_output", "") or result.validation_summary or ""
                err_lines = [line.strip() for line in blob.splitlines() if line.strip()]
                error_sig = err_lines[-1][:160] if err_lines else ""

            existing_events = [
                event
                for event in self.state_store.read_events()
                if event.get("type") == "iteration_result" and int(event.get("iter", 0) or 0) == result.iteration
            ]
            if len(existing_events) > 1:
                raise ValueError(f"duplicate iteration_result events for iteration {result.iteration}")
            if existing_events:
                existing = existing_events[0]
                if existing.get("decision") != decision_label or (
                    result.kept and existing.get("commit_hash") != result.commit_hash
                ):
                    raise ValueError(f"iteration_result conflicts with iteration {result.iteration}")

            newly_applied = self.run_state.next_iteration <= result.iteration
            if newly_applied:
                apply_iteration(
                    self.run_state,
                    iteration=result.iteration,
                    decision=decision_label,
                    kept=result.kept,
                    wall_ms=result.wall_ms,
                    mean_case_speedup=result.mean_case_speedup,
                    commit_hash=result.commit_hash,
                    plan=plan,
                    baseline_wall_ms=self.ic.baseline_wall_ms,
                    best_wall_ms=self.best_wall_ms,
                    best_mean_case_speedup=self.best_mean_case_speedup,
                    stall_threshold=self.ic.supervise_after,
                    orchestration_error_threshold=(self.ic.max_consecutive_orchestration_errors),
                )
            elif result.kept and (
                self.run_state.best.iteration != result.iteration
                or self.run_state.best.commit_hash != result.commit_hash
                or self.run_state.best.wall_ms != result.wall_ms
                or self.run_state.best.mean_case_speedup != result.mean_case_speedup
            ):
                raise ValueError(f"run state conflicts with KEEP iteration {result.iteration}")

            # The monitor remains operational even if durable state I/O fails, but an idempotent recovery must not
            # count the outcome twice.
            if newly_applied and self.monitor is not None and not is_infrastructure_decision(decision_label):
                self.monitor.record(kept=result.kept)
            if result.kept:
                self._expire_supervisor_ruling()
            head_out = self._git("rev-parse", "HEAD").strip()
            if head_out:
                self.run_state.head_commit = head_out.splitlines()[0]
            if not existing_events:
                self.state_store.append_event(
                    self._iteration_result_event(
                        result,
                        plan=plan,
                        decision_label=decision_label,
                        error_sig=error_sig,
                        checkpoint_metadata=checkpoint_metadata,
                    )
                )
            if newly_applied:
                self._record_direction_verdict(
                    self.run_state,
                    iteration=result.iteration,
                    decision_label=decision_label,
                    mean_case_speedup=result.mean_case_speedup,
                    best_mean_case_speedup=self.best_mean_case_speedup,
                    bench_detail=result.bench_detail,
                    incumbent_case_times=self._scored_incumbent_case_times(),
                )
            self.state_store.save(self.run_state)

            if require_durable:
                persisted = self.state_store.load()
                durable_events = [
                    event
                    for event in self.state_store.read_events()
                    if event.get("type") == "iteration_result" and int(event.get("iter", 0) or 0) == result.iteration
                ]
                if (
                    persisted.next_iteration <= result.iteration
                    or len(durable_events) != 1
                    or durable_events[0].get("decision") != decision_label
                    or (
                        result.kept
                        and (
                            persisted.best.iteration != result.iteration
                            or persisted.best.commit_hash != result.commit_hash
                            or persisted.best.wall_ms != result.wall_ms
                        )
                    )
                ):
                    raise RuntimeError(f"iteration {result.iteration} checkpoint was not durable")
            return True
        except Exception as error:  # noqa: BLE001 - best-effort unless required
            if require_durable:
                raise RuntimeError(f"failed to finalize iteration {result.iteration} checkpoint") from error
            log.debug("run_state: iteration reduce/save failed", exc_info=True)
            return False

    def _iteration_result_event(
        self,
        result: IterationResult,
        *,
        plan: str,
        decision_label: str | None = None,
        error_sig: str = "",
        checkpoint_metadata: dict | None = None,
    ) -> dict:
        """Build the canonical durable event for one completed iteration."""
        resolved_decision = decision_label or _decision_label(result)
        return make_event(
            "iteration_result",
            result.iteration,
            decision=resolved_decision,
            plan=(plan or "").strip()[:120] or None,
            # The mode this iteration actually ran under, which is the direction identity the empty-diff streak is
            # counted against.
            search_mode=self.run_state.search_mode,
            wall_ms=result.wall_ms,
            mean_case_speedup=result.mean_case_speedup,
            snr_db=result.snr_db,
            error_sig=error_sig or None,
            session_end_reason=result.session_end_reason or None,
            session_index=int(
                (checkpoint_metadata or {}).get(
                    "session_index",
                    self.run_state.session_index,
                )
            ),
            experiment_id=(
                str((checkpoint_metadata or {}).get("experiment_id") or "")
                or (self.experiment.experiment_id if self.experiment else None)
            ),
            turns=result.turns,
            validation_passed=result.validation_passed,
            commit_hash=result.commit_hash or None,
            best_after_ms=(result.wall_ms if result.kept else self.best_wall_ms),
            best_after_mean_case_speedup=(result.mean_case_speedup if result.kept else self.best_mean_case_speedup),
            is_new_best=result.kept,
            diversification_cycle_completed=(self.run_state.diversification_cycle_completed),
        )

    def _record_iteration_handoff(
        self,
        *,
        iteration: int,
        decision: str,
        optimization_plan_path: str,
        session_sink: dict,
        archived_path: Path | None = None,
    ) -> Path | None:
        """Persist one lightweight handoff without duplicating full artifacts."""
        if self.handoff_store is None:
            return None
        try:
            head_lines = self._git("rev-parse", "HEAD").splitlines()
            analysis_commit = (
                self.run_state.best.commit_hash
                or self.run_state.head_commit
                or (head_lines[0] if head_lines else "")
                or self.ic.campaign_base_commit
                or "uncommitted"
            )
            lesson_path = ""
            handoff_plan = str(session_sink.get("plan") or "")
            if getattr(self, "lessons", None) is not None:
                candidate = self.lessons.path(iteration)
                if candidate.is_file():
                    lesson_path = str(candidate.resolve().relative_to(Path(self.ic.workspace_dir).resolve()))
                    # The planner reads handoffs, not lesson documents, and a refutation quoted out of a handoff is
                    # how one sweep at one M became a ban on every case.
                    scope = self.lessons.scope_of(iteration)
                    if scope is not None:
                        line = format_scope_line(scope)
                        handoff_plan = f"{handoff_plan}\n{line}" if handoff_plan else line
            workspace = Path(self.ic.workspace_dir).resolve()
            relative_plan_path = ""
            orchestration_artifacts = ""
            if optimization_plan_path:
                plan_path = Path(optimization_plan_path).resolve()
                if plan_path.is_file():
                    relative_plan_path = str(plan_path.relative_to(workspace))
                    orchestration_artifacts = str(plan_path.parent.relative_to(workspace))
            supervisor_ruling_path = ""
            current_ruling = latest_supervisor_ruling_path(self.ic.workspace_dir)
            if current_ruling.is_file():
                supervisor_ruling_path = str(current_ruling.resolve().relative_to(workspace))
            handoff = IterationHandoff(
                iteration=iteration,
                analysis_commit=analysis_commit,
                canonical_verdict=decision,
                search_mode=self.run_state.search_mode,
                search_reason_codes=tuple(self.run_state.search_reason_codes),
                search_objective=self.run_state.search_objective,
                search_mode_residence_remaining=(self.run_state.search_mode_residence_remaining),
                diversification_cycle_complete=(self.run_state.diversification_cycle_completed),
                optimization_plan_path=relative_plan_path,
                supervisor_ruling_path=supervisor_ruling_path,
                plan=handoff_plan,
                lesson_path=lesson_path,
                orchestration_artifacts=orchestration_artifacts,
                candidate_archive=(
                    str(archived_path.resolve().relative_to(Path(self.ic.workspace_dir).resolve()))
                    if archived_path is not None
                    else ""
                ),
            )
            return self.handoff_store.write(handoff)
        except Exception as error:  # noqa: BLE001 - handoff is best-effort
            self.persistence_degraded = True
            self.persistence_errors.append(f"persist handoff iteration {iteration}: {error}")
            self.persistence_errors = self.persistence_errors[-10:]
            log.debug(
                "failed to persist iteration handoff %s",
                iteration,
                exc_info=True,
            )
            return None

    def _scored_case_ids(self) -> list[str]:
        """The cases the suite actually scores, sorted."""
        return sorted(case_id for case_id in self._baseline_case_times if case_id not in self._unscored_cases)

    def _loop_measured_a_negative(self, decision: str, result: IterationResult, session_sink: dict) -> bool:
        """Whether the LOOP itself saw something come out worse this iteration."""
        label = (decision or "").strip().upper()
        if label not in _LABELS_WITHOUT_A_MEASURED_NEGATIVE:
            return True
        if label != "KEEP" and result.mean_case_speedup is not None:
            return True
        return bool(str(session_sink.get("findings") or "").strip())

    def _carries_measured_negative(
        self,
        decision: str,
        result: IterationResult,
        session_sink: dict,
        *,
        document: str,
        agent_narrative: bool,
    ) -> bool | None:
        """Whether this DOCUMENT records anything that measured worse."""
        if self._loop_measured_a_negative(decision, result, session_sink):
            return True
        if not agent_narrative:
            return False
        return parse_negatives_marker(document)

    def _lesson_scope(
        self,
        store: LessonStore,
        iteration: int,
        session_sink: dict,
        *,
        decision: str,
        result: IterationResult,
        agent_narrative: bool,
    ) -> LessonScope:
        """The conditions this iteration's observations were taken under."""
        scored = tuple(self._scored_case_ids())
        named = cases_named_in(str(session_sink.get("plan") or ""), scored)
        restricted = bool(named) and len(named) < len(scored)
        document = store.read(iteration)
        return LessonScope(
            cases=named if restricted else scored,
            held_fixed=parse_held_fixed(document),
            lane_restricted=restricted,
            carries_negative=self._carries_measured_negative(
                decision,
                result,
                session_sink,
                document=document,
                agent_narrative=agent_narrative,
            ),
            disproof=parse_disproof_marker(document),
        )

    async def _record_lesson(
        self,
        *,
        iteration: int,
        result: IterationResult,
        decision: str,
        session_sink: dict,
        diff_summary: str = "",
    ) -> None:
        """Write this iteration's free-form factual session record."""
        store = getattr(self, "lessons", None)
        if store is None or session_sink.get("session_started") is not True:
            # No agent ran this iteration (e.g. the baseline measurement path): there is no exploration to record.
            return

        has_narrative = False
        # Narrated by the resumed session itself, as opposed to machine-written by the loop below.
        agent_narrative = False
        summary_failure = ""
        if self._time_remaining() < SUMMARY_MIN_SECONDS:
            # Gate on whether there is time to PRODUCE the summary, not on the loop's session-admission reserve: a
            # campaign that runs out of room for another implementer session is resumed later, and that session reads
            # this document.
            print("  [lesson] too little time left — recording outcome only")
            summary_failure = "insufficient campaign time to run summarizer"
        else:
            try:
                outcome = await summarize_iteration(
                    store=store,
                    iteration=iteration,
                    end_reason=result.session_end_reason,
                    summarizer=session_sink.get("summarize"),
                    pr_references=self.ic.pr_reference_labels,
                    pr_reference_context=self.ic.pr_reference_context,
                )
                has_narrative = bool(outcome)
                agent_narrative = has_narrative
                if has_narrative:
                    print(f"  [lesson] recorded iter {iteration}: {len(outcome.text)} chars")
                else:
                    summary_failure = outcome.reason
                    print(
                        f"  [lesson] no summary ({outcome.reason}) — falling back to machine-observed session progress"
                    )
            except Exception as error:  # noqa: BLE001 - never break the loop
                summary_failure = f"{type(error).__name__}: {str(error)[:200]}"
                log.debug("lessons: summarizer step failed", exc_info=True)
                print(f"  [lesson] summarizer step failed ({type(error).__name__}: {error}) — falling back")
            finally:
                self._checkpoint_llm_usage()

        if not has_narrative:
            # No session could describe what was explored, but the gate's block reasons are a real record of what the
            # agent ran into.
            try:
                fallback = build_fallback_document(
                    diff_summary=diff_summary,
                    findings=session_sink.get("findings", ""),
                    end_reason=result.session_end_reason,
                    summary_failure=summary_failure,
                    turns=result.turns,
                    plan=session_sink.get("plan", ""),
                    progress_log=session_sink.get("progress_log"),
                )
                if fallback and store.write(iteration, fallback) is not None:
                    has_narrative = True
                    print(f"  [lesson] machine-recorded iter {iteration} from gate findings: {len(fallback)} chars")
            except Exception:  # noqa: BLE001 - best-effort
                log.debug("lessons: fallback document failed", exc_info=True)

        try:
            scope = self._lesson_scope(
                store,
                iteration,
                session_sink,
                decision=decision,
                result=result,
                agent_narrative=agent_narrative,
            )
            if store.append_scope(iteration, scope):
                print(f"  [lesson] {format_scope_line(scope)}")
                if scope.carries_negative is not False and not scope.held_fixed:
                    print(
                        "  [lesson] no held-fixed constants recorded — "
                        "negatives from this iteration re-open on the next "
                        "change"
                    )
                if is_claim_disproved(scope.disproof):
                    print(
                        "  [lesson] a direction reported unreachable was "
                        "shown reachable by the experiment run against it — "
                        "later iterations are told to re-enter it, not to "
                        "treat this record as closing it"
                    )
                elif scope.disproof == UNDISPROVEN_CLAIM:
                    print(
                        "  [lesson] a direction was reported unreachable "
                        "without running the experiment that would falsify "
                        "that — it stays open for later iterations"
                    )
                elif scope.disproof is None and agent_narrative:
                    print(
                        "  [lesson] the record answered nothing about "
                        "unreachable directions — any 'cannot' in it is "
                        "recorded as unchecked, and closes nothing"
                    )
            else:
                print(
                    f"  [lesson] scope not recorded for iter {iteration}: "
                    f"the document renders unscoped and closes nothing"
                )
        except Exception:  # noqa: BLE001 - best-effort
            log.debug("lessons: scope append failed", exc_info=True)

        try:
            store.append_outcome(
                iteration,
                format_outcome_line(
                    decision=decision,
                    wall_ms=result.wall_ms,
                    best_wall_ms=self.best_wall_ms,
                    mean_case_speedup=result.mean_case_speedup,
                    best_mean_case_speedup=self.best_mean_case_speedup,
                    snr_db=result.snr_db,
                    end_reason=result.session_end_reason,
                    turns=result.turns if not has_narrative else None,
                    summary_failure=(summary_failure if not has_narrative else ""),
                ),
            )
        except Exception:  # noqa: BLE001 - best-effort
            log.debug("lessons: outcome append failed", exc_info=True)

    async def run_one_iteration(
        self,
        iteration: int,
        plan: str = "",
        *,
        benchmark_measurement: dict | None = None,
    ) -> IterationResult:
        """Execute a single build→validate→bench→canonical→decide iteration."""
        iter_start = time.time()
        force_jit_rebuild(self._jit_source_files())

        # Step 1: Build (if configured), through `rtk err` so the tail the agent is
        # handed below is diagnostics rather than progress chatter. Bare `rtk` was
        # wrapped here before and did nothing: rtk ships no ninja/cmake filter and
        # passes an unknown command through (see kernelforge.rtk).
        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *err_wrap(list(self.ic.build_command)),
                cwd=self.ic.build_dir or self.ic.workspace_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await communicate_process_group(
                proc,
                timeout=self.ic.build_timeout_sec,
            )
            if proc.returncode != 0:
                return IterationResult(
                    iteration=iteration,
                    duration_sec=time.time() - iter_start,
                    validation_passed=False,
                    validation_summary=f"BUILD FAILED: {_build_failure_tail(stdout, stderr, 500)}",
                    kept=False,
                )

        # Step 2: driver-owned full correctness suite.
        measurement_started = time.time()
        print("  [validate] Running full correctness suite...")
        report = await run_validation_pipeline(
            driver_script=self.ic.driver_script,
            snr_threshold=self.ic.snr_threshold,
            timeout_per_stage=self.ic.validate_stage_timeout_sec,
        )
        for r in report.results:
            status = (
                "PASS"
                if r.passed
                else "TIMEOUT"
                if r.outcome == "timeout"
                else "ERROR"
                if r.outcome in {"driver_error", "invalid_result"}
                else "FAIL"
            )
            snr_str = f" SNR={r.snr_db:.1f}dB" if r.snr_db is not None else ""
            print(f"  [validate] Stage {r.stage} {r.stage_name}: {status}{snr_str}")

        if not report.all_passed:
            print(f"  [validate] FAILED at stage {report.failed_stage} — skipping bench")
            self._observe_measurement(measurement_started)
            return IterationResult(
                iteration=iteration,
                duration_sec=time.time() - iter_start,
                validation_passed=False,
                validation_summary=report.summary(),
                validation_outcome=report.failed_outcome,
                error_output=report.failed_output,
                kept=False,
            )

        # Step 3: Benchmark (only if validation passed).
        if benchmark_measurement is not None and self._can_reuse_insession_benchmark(
            benchmark_measurement,
            attempt_diff=self._working_tree_diff(),
        ):
            print("  [bench] Reusing in-session three-measurement result...")
            bench_result = dict(benchmark_measurement)
            bench_result["reused_from_insession"] = True
        else:
            print("  [bench] Running three independent benchmark suites...")
            bench_result = await measure_wallclock(
                driver_script=self.ic.driver_script,
                driver_args=[],
                measurements=KEEP_MEASUREMENT_COUNT,
                timeout_sec=self.ic.bench_timeout_sec,
                repeat=self.ic.bench_repeat,
            )
        self._observe_measurement(measurement_started)

        self.last_case_bandwidth = dict(bench_result.get("case_bandwidth") or {})
        selected_raw_mean_ms = bench_result.get("median_ms")
        snr_db = report.results[-1].snr_db if report.results else None
        # A candidate whose bench crashed is reverted for "no speedup" unless the crash itself reaches the agent; the
        # tool's output tail is the only place the traceback exists.
        bench_error_output = (
            "" if bench_result.get("success") else "BENCH FAILED: " + _bench_failure_detail(bench_result)
        )

        # Collapse per-case times into an equal-weight mean of per-case speedups, rather than allowing expensive cases
        # to dominate an aggregate ratio.
        self._apply_mean_case_speedup_metric(bench_result)
        mean_case_speedup = bench_result.get("mean_case_speedup")
        measurement_scores = list(bench_result.get("measurement_mean_case_speedups") or [])
        score_text = f"{mean_case_speedup:.6f}x" if mean_case_speedup is not None else "n/a"
        sigma_resolution = await self._resolve_keep_sigma(bench_result, measurement_scores)
        sigma = sigma_resolution.sigma
        required_score = required_keep_speedup(
            self.best_mean_case_speedup or 1.0,
            measurement_scores,
            sigma=sigma,
            sigma_sample_size=sigma_resolution.sample_size,
        )
        # The bar is a t multiple of the standard error of the mean, so a REVERT is only readable next to the spread
        # that set it and, when one case supplied that spread, next to the case: a weak candidate and a noisy 10 us
        # dispatch print the same mean score.
        sigma_text = f"{sigma:.6f}" if sigma is not None else "n/a"
        print(
            f"  [bench] pristine-relative scores="
            f"{[round(score, 6) for score in measurement_scores]}; "
            f"sigma={sigma_text}; "
            f"{_sigma_attribution_note(sigma_resolution)}"
            f"mean score={score_text}; required={required_score:.6f}x; "
            f"raw mean={selected_raw_mean_ms} ms  "
            f"({bench_result.get('message', '')})"
        )
        if bench_error_output:
            # The scoring verdict above has already overwritten ``message`` with "candidate emitted no per-case
            # timings", which describes the symptom of a crash as if it were a formatting choice.
            print(f"  [bench] {bench_error_output}")

        # Profiling evidence is produced by the commit-bound Analysis Agent.
        pmc_diagnosis = ""
        pmc_full = ""

        # Step 5: Register check (optional — requires build artifacts)
        vgpr = None
        try:
            reg_result = await check_registers(build_dir=self.ic.build_dir)
            vgpr = reg_result.get("vgpr") if reg_result.get("success") else None
            if vgpr:
                print(f"  [registers] VGPR={vgpr}")
        except Exception:
            log.debug("optional register check failed", exc_info=True)

        # Step 6: the mean of the independent pristine-relative scores must clear the current best by the candidate's
        # own measurement noise.
        improved = bool(bench_result.get("success")) and passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=(self.best_mean_case_speedup or 1.0),
            sigma=sigma,
            sigma_sample_size=sigma_resolution.sample_size,
        )

        # Step 7: the arena's own verdict.
        if improved:
            canonical_started = time.time()
            canonical = await accept_candidate(
                self.ic.workspace_dir,
                timeout_cap_sec=self.ic.validate_stage_timeout_sec,
                candidate_label=f"iteration {iteration}",
            )
            # The suite only runs for a candidate the round produced, so it is part of that round's measurement and
            # has to be priced into the next round's admission alongside the validate-and-bench cycle.
            self._observe_measurement(canonical_started)
            if not canonical.passed:
                return IterationResult(
                    iteration=iteration,
                    duration_sec=time.time() - iter_start,
                    validation_passed=False,
                    validation_summary=(
                        f"{report.summary()}\n  Canonical correctness suite: FAILED — {canonical.detail}"
                    ),
                    validation_outcome=(canonical.outcome or "canonical_correctness_failure"),
                    wall_ms=selected_raw_mean_ms,
                    mean_case_speedup=mean_case_speedup,
                    snr_db=snr_db,
                    vgpr=vgpr,
                    error_output=canonical.output,
                    kept=False,
                    bench_detail=(bench_result if isinstance(bench_result, dict) else {}),
                )

        duration = time.time() - iter_start

        result = IterationResult(
            iteration=iteration,
            duration_sec=duration,
            validation_passed=True,
            validation_summary=report.summary(),
            wall_ms=selected_raw_mean_ms,
            mean_case_speedup=mean_case_speedup,
            snr_db=snr_db,
            pmc_diagnosis=pmc_diagnosis,
            vgpr=vgpr,
            kept=improved,
            bench_detail=bench_result if isinstance(bench_result, dict) else {},
            pmc_full=pmc_full,
            error_output=bench_error_output,
        )

        return result

    def _update_search_policy(self, iteration: int) -> SearchPolicyDecision:
        """Derive and persist the search mode before planning an iteration."""
        window_gain = self._exploit_window_gain(
            self.state_store.recent_results(MARGINAL_GAIN_SCAN_WINDOW),
            window=MARGINAL_GAIN_WINDOW,
            since_iteration=self.run_state.stall.last_supervisor_iter,
        )
        decision = self.search_policy_engine.decide(
            best_source=self.run_state.best.source,
            no_improvement_iters=self.run_state.stall.unresolved_stall_iters,
            stall_threshold=self.ic.supervise_after,
            current_mode=self.run_state.search_mode,
            residence_iterations_remaining=(self.run_state.search_mode_residence_remaining),
            diversification_cycle_completed=(self.run_state.diversification_cycle_completed),
            consecutive_no_changes=self._consecutive_no_changes(
                self.state_store.recent_results(NO_CHANGES_STREAK_WINDOW)
            ),
            window_gain_ratio=window_gain.ratio,
        )
        previous_mode = self.run_state.search_mode
        previous_reasons = tuple(self.run_state.search_reason_codes)
        self.run_state.search_mode = decision.mode
        self.run_state.search_reason_codes = list(decision.reason_codes)
        self.run_state.search_objective = decision.objective_kind
        self.run_state.search_mode_residence_remaining = decision.residence_iterations_remaining
        self.run_state.diversification_cycle_completed = False
        self._search_policy_decision = decision
        try:
            self.state_store.append_event(
                make_event(
                    "search_policy_decision",
                    iteration,
                    mode=decision.mode,
                    reason_codes=list(decision.reason_codes),
                    objective_kind=decision.objective_kind,
                    residence_iterations_remaining=(decision.residence_iterations_remaining),
                    # ``make_event`` drops empty fields, so a ratio of None on its own would leave the event silent
                    # about a trigger that could not be evaluated at all -- indistinguishable from a young campaign.
                    window_gain_ratio=window_gain.ratio,
                    window_gain_unavailable=window_gain.unavailable,
                    mode_changed=(decision.mode != previous_mode),
                )
            )
            self.state_store.save(self.run_state)
        except Exception:  # noqa: BLE001 - policy remains available in memory
            log.debug("search policy persistence failed", exc_info=True)
        # A window that has not filled yet is the ordinary state of a young campaign.
        fault = window_gain.unavailable
        if fault is not None and fault != "short_window" and fault not in self._reported_window_gain_faults:
            self._reported_window_gain_faults.add(fault)
            print(f"  [search-policy] diminishing-returns trigger unavailable: {fault}")
        if decision.mode != previous_mode or decision.reason_codes != previous_reasons:
            print(f"  [search-policy] {decision.mode}: " + ", ".join(decision.reason_codes))
        return decision

    async def _plan_round(
        self,
        *,
        iteration: int,
        orchestration_service,
        lanes: int = 1,
    ) -> tuple[Path | None, str]:
        """Buy the round's plans and charge the round for the wall-clock."""
        started_at = time.time()
        try:
            return await self._run_orchestration(
                iteration=iteration,
                orchestration_service=orchestration_service,
                lanes=lanes,
            )
        finally:
            self._round_planning_sec += max(0.0, time.time() - started_at)

    async def _run_orchestration(
        self,
        *,
        iteration: int,
        orchestration_service,
        lanes: int = 1,
    ) -> tuple[Path | None, str]:
        """Run planning and durably publish every lane's plan for the round."""
        context = self._with_case_config_coverage(
            self._active_analysis_context
            if self._active_analysis_context is not None
            else self._build_orchestration_context()
        )
        try:
            result = await orchestration_service.run(
                context,
                usage=self._usage,
                lanes=lanes,
            )
        except OrchestrationInfrastructureError as error:
            detail = f"{type(error).__name__}: {error}"
            print(f"  [orchestration] failed ({detail})")
            return None, detail
        finally:
            self._checkpoint_llm_usage()

        self._record_probe_hazard(iteration, result)
        self._last_lane_plans = [plan for plan in result.optimization_plans if str(plan).strip()]
        self._persist_orchestration_result(iteration, context, result)
        plan_path = self._persist_lane_plans(
            iteration,
            self._last_lane_plans,
            analysis_commit=context.analysis_commit,
        )
        self._record_orchestration_final_plan(iteration, plan_path)
        self._last_orchestration_plan_executable = bool(getattr(result, "optimization_plan_executable", True))
        critic = result.plan_critic
        self._last_critic_verdict = critic.verdict if critic is not None else ""
        self._last_critic_review = critic.review if critic is not None else ""
        self._record_critic_ruling(iteration, critic)
        self._latest_optimization_plan_path = str(plan_path)
        print(f"  [orchestration] optimization plan: {plan_path}")
        if len(self._last_lane_plans) > 1:
            print(f"  [orchestration] {len(self._last_lane_plans)} lane plans published under {plan_path.parent}")
        return plan_path, ""

    def _record_probe_hazard(self, iteration: int, result) -> None:
        """Turn a probe the analysis phase could not clear into a live hazard."""
        diagnostics = getattr(result, "structured_output_diagnostics", None)
        finding = (diagnostics or {}).get("probe_device_hazard")
        if not isinstance(finding, dict):
            return
        hazard = self.device_hazard.record(
            iteration=iteration,
            detail=f"probe round: {finding.get('describe', '')}",
            pids=finding.get("pids") or (),
        )
        print(
            "  [probe] the round's probe scratch tree left the device "
            f"contended; this round measures nothing. {hazard.describe()}"
        )

    def _record_critic_ruling(self, iteration: int, critic) -> None:
        """Put this round's verdict where the next process can still find it."""
        ruling = CriticRuling()
        if critic is not None and not critic.error:
            ruling = CriticRuling(
                verdict=critic.verdict,
                review_path=str((self._orchestration_root(iteration) / "critic_review.md").resolve()),
            )
        self.run_state.last_critic = ruling

    def _restore_critic_ruling(self) -> None:
        """Resume the ruling whose round ended with the process that bought it."""
        ruling = self.run_state.last_critic
        if not ruling.verdict or not ruling.review_path:
            return
        try:
            review = Path(ruling.review_path).read_text(encoding="utf-8").strip()
        except OSError as error:
            log.warning(
                "critic review at %s is unreadable (%s); resuming without the %s verdict it carried",
                ruling.review_path,
                error,
                ruling.verdict,
            )
            self.run_state.last_critic = CriticRuling()
            return
        if not review:
            self.run_state.last_critic = CriticRuling()
            return
        self._last_critic_verdict = ruling.verdict
        self._last_critic_review = review
        print(f"  [critic] resuming the {ruling.verdict} verdict an earlier process recorded")

    def _orchestration_root(self, iteration: int) -> Path:
        return Path(self.ic.workspace_dir).resolve() / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}"

    def _lane_plan_path(self, iteration: int, lane: int) -> Path:
        """Where one lane's plan lives, lane 1 keeping the historical name."""
        if lane <= 1:
            return self._orchestration_root(iteration) / "optimization_plan.md"
        return self._orchestration_root(iteration) / f"lane_{lane:03d}.md"

    def _lane_queue_path(self) -> Path:
        """Where a round's unspent candidates wait for the iteration that measures them."""
        return Path(self.ic.workspace_dir).resolve() / "forge_experiments" / "orchestration" / "lane_queue.json"

    def _persist_lane_queue(self) -> None:
        """Publish what this round has bought and not yet measured."""
        from kernelforge.loop.recovery import atomic_write_json

        path = self._lane_queue_path()
        try:
            if not self._lane_queue:
                path.unlink(missing_ok=True)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                path,
                {
                    "candidates": [
                        {
                            "lane_id": lane.lane_id,
                            "plan": lane.plan,
                            "diff": lane.diff,
                        }
                        for lane in self._lane_queue
                    ]
                },
            )
        except OSError as error:
            print(
                f"  [lanes] queue not durable ({error}); the "
                f"{len(self._lane_queue)} candidate(s) still queued are lost "
                "if this process does not measure them"
            )

    def _restore_lane_queue(self) -> None:
        """Pick up candidates a previous process bought and never measured."""
        path = self._lane_queue_path()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            queued = [
                LaneResult(
                    lane_id=str(entry["lane_id"]),
                    plan=str(entry["plan"]),
                    diff=str(entry["diff"]),
                )
                for entry in record["candidates"]
            ]
        except (OSError, ValueError, KeyError, TypeError):
            log.debug("no readable lane queue at %s", path)
            return
        queued = [lane for lane in queued if lane.produced_candidate]
        if not queued:
            return
        self._lane_queue = queued
        print(f"  [lanes] resuming {len(queued)} candidate(s) an earlier round bought and never measured")

    def _lane_plan_manifest_path(self, iteration: int) -> Path:
        """The record that says a round's plans are complete and what they are."""
        return self._orchestration_root(iteration) / "lane_plans.json"

    def _persist_lane_plans(
        self,
        iteration: int,
        plans: Sequence[str],
        *,
        analysis_commit: str,
    ) -> Path:
        """Atomically publish every lane's plan and return lane 1's path."""
        from kernelforge.loop.recovery import atomic_write_json

        published = [str(plan).strip() for plan in plans]
        if not published or not published[0]:
            raise ValueError("optimization plan must not be empty")
        if not all(published):
            raise ValueError("every lane plan must be non-empty")
        if not analysis_commit:
            raise ValueError("lane plans must record the commit they describe")
        manifest = self._lane_plan_manifest_path(iteration)
        manifest.unlink(missing_ok=True)
        keep = {self._lane_plan_path(iteration, lane) for lane in range(2, len(published) + 1)}
        for stale in self._orchestration_root(iteration).glob("lane_*.md"):
            if stale not in keep:
                stale.unlink()
        for lane, plan in enumerate(published, start=1):
            atomic_write_text(self._lane_plan_path(iteration, lane), plan + "\n")
        atomic_write_json(
            manifest,
            {"analysis_commit": analysis_commit, "lanes": len(published)},
        )
        return self._lane_plan_path(iteration, 1)

    def _load_lane_plans(self, iteration: int) -> tuple[str, list[str]] | None:
        """One round's commit and plans in lane order, or None if it has none."""
        try:
            manifest = json.loads(self._lane_plan_manifest_path(iteration).read_text(encoding="utf-8"))
            lanes = int(manifest["lanes"])
            analysis_commit = str(manifest["analysis_commit"])
        except (OSError, ValueError, KeyError, TypeError):
            log.debug("no readable lane plan manifest for iteration %s", iteration)
            return None
        plans: list[str] = []
        for lane in range(1, lanes + 1):
            try:
                plan = self._lane_plan_path(iteration, lane).read_text(encoding="utf-8").strip()
            except OSError:
                log.debug("lane %s of iteration %s is unreadable", lane, iteration)
                return None
            if not plan:
                return None
            plans.append(plan)
        if not plans or not analysis_commit:
            return None
        return analysis_commit, plans

    def _unfinished_iteration(self, before: int) -> int | None:
        """The iteration before ``before`` that started and reported no result."""
        started = 0
        finished: set[int] = set()
        for event in self.state_store.read_events():
            iteration = event.get("iter")
            if not isinstance(iteration, (int, float)):
                continue
            if int(iteration) >= before:
                continue
            if event.get("type") == "iteration_started":
                started = max(started, int(iteration))
            elif event.get("type") == "iteration_result":
                finished.add(int(iteration))
        if started and started not in finished:
            return started
        return None

    def _recoverable_lane_plans(
        self,
        iteration: int,
    ) -> tuple[int, list[str]] | None:
        """A previous round's plans that were paid for and never dispatched."""
        planned_iteration = self._unfinished_iteration(before=iteration)
        if planned_iteration is None:
            return None
        published = self._load_lane_plans(planned_iteration)
        if published is None:
            return None
        planned_commit, plans = published
        if len(plans) < 2:
            return None
        if planned_commit != self._canonical_commit():
            print(
                f"  [lanes] iteration {planned_iteration} planned against "
                f"{planned_commit}, which the tree has moved off; "
                "planning this round afresh"
            )
            return None
        return planned_iteration, plans

    def _persist_orchestration_result(self, iteration, context, result) -> None:
        """Persist planning diagnostics before publishing the executable plan."""
        from kernelforge.loop.recovery import atomic_write_json

        root = self._orchestration_root(iteration)
        atomic_write_json(
            root / "context.json",
            context.to_prompt_dict(),
        )
        if result.dispatch_plan is not None:
            atomic_write_json(
                root / "dispatch.json",
                result.dispatch_plan.to_dict(),
            )
        atomic_write_json(
            root / "specialists.json",
            {
                "analysis_commit": context.analysis_commit,
                "outcomes": [outcome.to_dict() for outcome in result.specialist_outcomes],
            },
        )
        diagnostics = dict(result.structured_output_diagnostics or {})
        artifact_paths = {}
        draft = str(result.optimization_plan_draft or "").strip()
        if draft:
            draft_path = root / "draft_plan.md"
            atomic_write_text(draft_path, draft + "\n")
            artifact_paths["draft_plan"] = str(draft_path.resolve())
        critic = result.plan_critic
        if critic is not None:
            critic_path = root / "critic_review.md"
            atomic_write_text(
                critic_path,
                critic.render_artifact().rstrip() + "\n",
            )
            artifact_paths["critic_review"] = str(critic_path.resolve())
        if artifact_paths:
            diagnostics["artifact_paths"] = artifact_paths
            diagnostics["plan_revised"] = bool(result.plan_revised)
        if diagnostics:
            atomic_write_json(
                root / "structured_output.json",
                diagnostics,
            )

    def _record_orchestration_final_plan(
        self,
        iteration: int,
        plan_path: Path,
    ) -> None:
        """Publish the final-plan pointer only after the plan exists."""
        from kernelforge.loop.recovery import atomic_write_json

        diagnostics_path = self._orchestration_root(iteration) / "structured_output.json"
        if not diagnostics_path.is_file():
            return
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if not isinstance(diagnostics, dict):
            raise ValueError(f"invalid orchestration diagnostics: {diagnostics_path}")
        artifact_paths = diagnostics.get("artifact_paths")
        if not isinstance(artifact_paths, dict):
            return
        artifact_paths["final_plan"] = str(plan_path.resolve())
        atomic_write_json(diagnostics_path, diagnostics)

    async def run(
        self,
        agent_fn=None,
        analysis_service=None,
        orchestration_service=None,
        on_iteration=None,
        on_best_committed=None,
        on_best_ready=None,
        usage=None,
        supervisor_fn=None,
        *,
        agent_factory=None,
        workspace_lock_held: bool = False,
    ) -> list[IterationResult]:
        """Run the loop and clean attempt-owned processes on every exit path."""
        try:
            return await self._run_impl(
                agent_fn=agent_fn,
                agent_factory=agent_factory,
                analysis_service=analysis_service,
                orchestration_service=orchestration_service,
                on_iteration=on_iteration,
                on_best_committed=on_best_committed,
                on_best_ready=on_best_ready,
                usage=usage,
                supervisor_fn=supervisor_fn,
                workspace_lock_held=workspace_lock_held,
            )
        finally:
            try:
                from kernelforge.loop.aiter_cache import (
                    cleanup_current_owned_aiter_locks,
                )

                cleanup_current_owned_aiter_locks()
            except Exception:
                log.debug("failed to clean AITER locks on loop exit", exc_info=True)

    async def _run_impl(
        self,
        agent_fn=None,
        analysis_service=None,
        orchestration_service=None,
        on_iteration=None,
        on_best_committed=None,
        on_best_ready=None,
        usage=None,
        supervisor_fn=None,
        *,
        agent_factory=None,
        workspace_lock_held: bool = False,
    ) -> list[IterationResult]:
        """Run while exclusively owning this campaign workspace."""
        if workspace_lock_held:
            return await self._run_locked(
                agent_fn=agent_fn,
                agent_factory=agent_factory,
                analysis_service=analysis_service,
                orchestration_service=orchestration_service,
                on_iteration=on_iteration,
                on_best_committed=on_best_committed,
                on_best_ready=on_best_ready,
                usage=usage,
                supervisor_fn=supervisor_fn,
            )
        store = LoopStateStore(self.ic.workspace_dir)
        with store.workspace_lock():
            return await self._run_locked(
                agent_fn=agent_fn,
                agent_factory=agent_factory,
                analysis_service=analysis_service,
                orchestration_service=orchestration_service,
                on_iteration=on_iteration,
                on_best_committed=on_best_committed,
                on_best_ready=on_best_ready,
                usage=usage,
                supervisor_fn=supervisor_fn,
            )

    async def _run_locked(
        self,
        agent_fn=None,
        analysis_service=None,
        orchestration_service=None,
        on_iteration=None,
        on_best_committed=None,
        on_best_ready=None,
        usage=None,
        supervisor_fn=None,
        *,
        agent_factory=None,
    ) -> list[IterationResult]:
        """Run the autonomous iteration loop."""
        import functools

        global print
        print = functools.partial(print, flush=True)

        self.start_time = time.time()
        self.results = []
        self._usage = usage

        # Safety net: if the kernel is an aiter HIP kernel, force it to recompile from the current source
        # (AITER_REBUILD) so the agent's edits are never silently ignored via aiter's prebuilt in-tree .so.
        force_jit_rebuild(self._jit_source_files())

        # Cross-iteration objective ledger: records each iteration's net diff, measured outcome, and real error
        # signatures, then feeds concise toolchain observations and recent entries into the next prompt.
        self.ledger = ExperienceLedger(self.ic.workspace_dir)

        # Per-iteration lesson documents.
        self.lessons = LessonStore(self.ic.workspace_dir)
        try:
            self.handoff_store = HandoffStore(self.ic.workspace_dir)
        except Exception:
            self.handoff_store = None
            log.debug("handoff store initialization failed", exc_info=True)

        # Full-fidelity candidate archive: persists each iteration's WHOLE solution (kernel snapshot + diff + full
        # profile + measurements + decision) so a later iteration can read back any prior attempt's real code.
        self.archive = CandidateArchive(self.ic.workspace_dir, self.ic.kernel_file)
        self.best_publisher = BestResultPublisher(self.ic.workspace_dir)
        # Candidates from one concurrent fan-out, spent one per iteration so each is measured and judged on its own by
        # the ordinary decision path.
        self._lane_queue: list[LaneResult] = []
        self._last_lane_plans: list[str] = []
        # Stacked iterations run back to back so far.
        self._merge_precedence_streak = 0
        # A device the campaign may not measure on, recorded by whichever iteration found it and re-checked by every
        # iteration after it.
        self.device_hazard = DeviceHazardLog(self.ic.workspace_dir)

        # Durable, file-backed run state + append-only event log.
        self.state_store = LoopStateStore(self.ic.workspace_dir)
        state_exists = self.state_store.state_path.exists()
        self.run_state = self.state_store.load()
        current_ruling_path = latest_supervisor_ruling_path(self.ic.workspace_dir)
        self._supervisor_ruling = load_latest_supervisor_ruling(self.ic.workspace_dir) if self.resume else ""

        # Ownership boundary for anything a candidate creates, taken before this loop touches the workspace.
        self._pre_untracked = self._untracked_snapshot()

        if self.resume:
            self._validate_resume_scoring_state(self.run_state)
            # Pending KEEP reconciliation promotes the committed candidate and checkpoints scoring state.
            self._restore_scoring_state()
            # Publication reconciliation consumes both baseline anchors.
            if self.run_state.baseline_wall_ms is not None:
                self.ic.baseline_wall_ms = self.run_state.baseline_wall_ms
            if self.run_state.pristine_baseline_wall_ms is not None:
                self.ic.pristine_baseline_wall_ms = self.run_state.pristine_baseline_wall_ms
            self._restore_published_analysis_commit()
            pending = self._load_pending_keep()
            planned, pending_status, _, _ = self._plan_resume_recovery(
                self.run_state,
                pending,
            )
            self._validate_resume_state(
                planned,
                allow_dirty=pending_status == "uncommitted",
            )
            self._coordinate_resume_recovery(on_best_committed)
            self._restore_resume_baseline_case_times(self.run_state)
            if self._recovered_pending_keep is None:
                self._reconcile_best_publication()
            archive_next = self.archive.reconcile_next_iteration(
                self.run_state.next_iteration,
            )
            event_next = max(
                (
                    int(event.get("iter", 0) or 0) + 1
                    for event in self.state_store.read_events()
                    if isinstance(event.get("iter"), (int, float))
                ),
                default=1,
            )
            self.run_state.next_iteration = max(archive_next, event_next)
            if self.run_state.termination_reason == "orchestration_failed":
                if self.run_state.orchestration_circuit_state != ORCHESTRATION_CIRCUIT_OPEN:
                    raise ValueError("orchestration_failed resume requires an open circuit")
                if agent_fn is None or orchestration_service is None:
                    raise ValueError("orchestration_failed resume requires one orchestration probe")
                begin_orchestration_probe(self.run_state)
                self.state_store.append_event(
                    make_event(
                        "orchestration_circuit_half_open",
                        self.run_state.iteration,
                    )
                )
                self.state_store.save(self.run_state)
        else:
            self._validate_driver_integrity(self.run_state)
            if (
                state_exists
                or self.state_store.events_path.exists()
                or self._pending_keep_path.exists()
                or self.archive.max_iteration() > 0
                or current_ruling_path.exists()
                or (self.handoff_store is not None and self.handoff_store.latest() is not None)
            ):
                raise ValueError("workspace already contains a campaign; pass --resume to continue it")
            result = self._git("checkout", "-b", self.ic.git_branch)
            if "already exists" in result:
                self._git("checkout", self.ic.git_branch)
            current_branch = self._git("branch", "--show-current").splitlines()[0]
            if current_branch != self.ic.git_branch:
                raise ValueError(f"failed to switch workspace to branch {self.ic.git_branch}")
            self.run_state = RunState()

        # Anchor the campaign clock now that the state carrying what earlier sessions spent is loaded.
        self._campaign_started_at = self.start_time - max(0.0, float(self.run_state.round_costs.campaign_sec))

        if state_exists and reconcile_stale_running_session(self.run_state):
            self.state_store.append_event(
                make_event(
                    "session_interrupted",
                    self.run_state.iteration,
                    reason="stale_running_session_reconciled",
                )
            )
            self.state_store.save(self.run_state)

        parent_experiment_id = self.run_state.last_experiment_id
        next_segment_index = self.run_state.session_index + 1
        self._set_state_identity(self.run_state)
        self._stage_validated_warm_start_state()
        start_session(self.run_state)
        self.experiment = self.tracker.create_segment(
            campaign_id=self.run_state.campaign_id,
            segment_index=next_segment_index,
            parent_experiment_id=parent_experiment_id,
            task_id=Path(self.ic.kernel_file).stem,
            backend=self.ic.backend,
            kernel_backend=self.ic.kernel_backend,
            description=f"Autonomous optimization of {self.ic.kernel_file}",
            target_wall_ms=self.ic.target_wall_ms,
            baseline_wall_ms=self.ic.baseline_wall_ms,
        )
        self.run_state.last_experiment_id = self.experiment.experiment_id
        self.state_store.append_event(
            make_event(
                "session_started",
                self.run_state.iteration,
            )
        )
        self.state_store.save(self.run_state)

        # Persist only after the fresh-campaign guard has completed.
        if self.ic.pr_kb_snapshot:
            from kernelforge.knowledge.pr_monitor_refs import commit_snapshot
            from kernelforge.knowledge.pr_query_context import REASON_LOCAL_FAILURE

            try:
                commit_snapshot(self.ic.workspace_dir, self.ic.pr_kb_snapshot)
            except (OSError, ValueError) as error:
                print(f"  [pr-kb] warning: snapshot not persisted ({error})")
                if self.ic.pr_kb_event:
                    self.ic.pr_kb_event = dict(self.ic.pr_kb_event)
                    self.ic.pr_kb_event["degraded_reason"] = REASON_LOCAL_FAILURE
                else:
                    self.ic.pr_kb_event = {
                        "position": "A",
                        "reason": REASON_LOCAL_FAILURE,
                        "degraded_reason": REASON_LOCAL_FAILURE,
                    }
            self.ic.pr_kb_snapshot = {}
        if self.ic.pr_kb_event:
            try:
                self.state_store.append_event(make_event("pr_refs_refreshed", 0, **self.ic.pr_kb_event))
            except (OSError, ValueError) as error:
                print(f"  [pr-kb] warning: event not recorded ({error})")
            self.ic.pr_kb_event = {}

        # Self-supervision monitor (AVO): tracks stall / unproductive-cycle signals so the loop can call the
        # supervisor to redirect the search instead of stopping at the first plateau.
        self.monitor = None
        if supervisor_fn is not None:
            from kernelforge.loop.supervisor import SupervisionMonitor

            self.monitor = SupervisionMonitor(
                supervise_after=self.ic.supervise_after,
                cooldown=self.ic.supervise_cooldown,
            )
            print(
                f"  Supervisor: enabled (after {self.ic.supervise_after} stalls, "
                f"cooldown {self.ic.supervise_cooldown}, no intervention cap)"
            )

        print("Starting autonomous iteration loop")
        print(f"  Kernel: {self.ic.kernel_file}")
        print(f"  Target: {self.ic.target_wall_ms} ms")
        print(
            f"  Budget: {self.ic.max_time_hours}h "
            f"(finalize reserve: {self.ic.budget_reserve_sec / 60:.0f} min; "
            "a round is admitted only when what remains also covers its "
            "estimated cost)"
        )
        # The finalize reserve is an absolute admission guard; on a SHORT budget it can swallow most of the window
        # (e.g. a 30-min reserve on a 1h run leaves only 30 min for iterations).
        _budget_sec = self.ic.max_time_hours * 3600.0
        if _budget_sec > 0 and self.ic.budget_reserve_sec >= 0.5 * _budget_sec:
            _pct = 100.0 * self.ic.budget_reserve_sec / _budget_sec
            print(
                f"  WARNING: finalize reserve ({self.ic.budget_reserve_sec / 60:.0f} min) "
                f"consumes {_pct:.0f}% of the {self.ic.max_time_hours}h budget; "
                f"the effective iteration window is only "
                f"{max(0.0, _budget_sec - self.ic.budget_reserve_sec) / 60:.0f} min. "
                f"Raise --max-hours for a longer run."
            )
        print(f"  Experiment: {self.experiment.experiment_id}")
        print()

        # The CLI constructs IterationLoop before applying a KB warm-start, then records the freshly measured pristine
        # case timings on IterationConfig.
        self._set_baseline_case_times(self.ic.baseline_case_times)
        if not self.resume and not self.ic.warm_start_commit and self._baseline_case_times:
            self._best_case_times = dict(self._baseline_case_times)
            self._unscored_cases = {str(case_id) for case_id in self.ic.preloop_baseline_unscored_cases}
        if self._best_case_times:
            self._persist_scoring_state()

        # Anchor speedup reporting.
        if not self.resume and self.ic.baseline_wall_ms is None:
            print("Measuring baseline on unmodified kernel...")
            baseline_ms = await self._measure_baseline()
            if baseline_ms is not None:
                self.ic.baseline_wall_ms = baseline_ms
                self.experiment.baseline_wall_ms = baseline_ms
                self.tracker.set_baseline(self.experiment.experiment_id, baseline_ms)
                print(f"  Baseline: {baseline_ms:.3f} ms\n")
            else:
                print(
                    "  Baseline measurement unavailable — see the "
                    "'Baseline build FAILED'/'Baseline bench FAILED' line above "
                    "for what the driver actually did\n"
                )

        if self.ic.pristine_baseline_wall_ms is None:
            self.ic.pristine_baseline_wall_ms = self.ic.baseline_wall_ms

        # The scoring model defines the pristine kernel as 1.0x.
        if self.ic.baseline_wall_ms is not None:
            self.best_wall_ms = self.ic.baseline_wall_ms
        if self._baseline_case_times:
            self.best_mean_case_speedup = 1.0

        # Seed the run state's baseline and, guardedly, resume a prior best from a reused workspace (only when the
        # recorded best commit is still HEAD).
        self._seed_and_hydrate_run_state()
        self._adopt_validated_warm_start()

        # After the resume restore and the warm-start adoption above, never before them.
        if not self._baseline_case_times:
            raise RuntimeError(
                "mean case scoring requires pristine per-case timings before starting an optimization iteration"
            )

        # Every speedup this run reports is a ratio against those timings, so a baseline that drifted from the task's
        # own reference poisons the whole campaign.
        baseline_check = check_baseline_against_reference(
            self.ic.workspace_dir,
            self._baseline_case_times,
        )
        if baseline_check.unverified_reason:
            print(
                "  [baseline] the pristine anchor every speedup divides by is "
                f"unverified: {baseline_check.unverified_reason}"
            )
        else:
            print(
                "  [baseline] pristine anchor agrees with the task reference on "
                f"{baseline_check.compared_case_count} of "
                f"{baseline_check.measured_case_count} measured case(s); the "
                f"reference declares {baseline_check.reference_case_count}"
            )
        if baseline_check.unusable_entries:
            unusable = baseline_check.unusable_entries
            declared = baseline_check.reference_case_count + len(unusable)
            print(
                f"  [baseline] could not read {len(unusable)} of the "
                f"{declared} entries the task reference declares, so this "
                "check covers less of the anchor than the file does: " + "; ".join(unusable)
            )
        if baseline_check.tolerance_overridden and not baseline_check.unverified_reason:
            print(
                "  [baseline] drift tolerance widened to "
                f"{baseline_check.drift_tolerance * 100:.0f}% by "
                f"{BASELINE_DRIFT_TOLERANCE_ENV}, from the "
                f"{BASELINE_DRIFT_TOLERANCE * 100:.0f}% default; the "
                "anchor was accepted under the widened bound"
            )

        # A crash immediately after a verified commit can leave the KEEP's archive unfinished.
        await self._finish_recovered_pending_keep()

        # Candidates an earlier round bought and never measured.
        self._restore_lane_queue()

        # The previous round's verdict, for the same reason: a REPLACE is spent on the round after the one it judged,
        # and the budget often ends between the two.
        self._restore_critic_ruling()

        # Analyze the baseline canonical commit once before any specialist or Implementer session.
        if analysis_service is not None:
            await self._resolve_analysis_context(analysis_service)

        iteration = self.run_state.next_iteration - 1
        while True:
            iteration += 1

            # Build the lineage digest once per iteration — reused by BOTH the supervisor (trajectory to review) and
            # the implementer (prompt history).
            digest = ""
            if getattr(self, "archive", None) is not None:
                try:
                    digest = self.archive.render_digest()
                except Exception as e:
                    log.debug("could not render lineage digest: %s", e)
                    digest = ""

            # Check terminal conditions
            if self._is_gate_met():
                self.termination_reason = "gate_met"
                print(f"\nGATE MET at iteration {iteration}: raw wall target reached at {self.best_wall_ms:.6f} ms")
                break
            if self.run_state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN:
                self.termination_reason = "orchestration_failed"
                print(
                    "\nORCHESTRATION FAILED repeatedly; stopping after "
                    f"{self.run_state.orchestration_error_streak} "
                    "consecutive infrastructure errors"
                )
                break
            if self._is_budget_exhausted():
                self.termination_reason = "budget_exhausted"
                print(f"\nBUDGET EXHAUSTED after {len(self.results)} iterations in this session")
                break
            if self._is_force_stopped():
                self.termination_reason = "force_stop"
                print("\nFORCE STOP: .stop file detected — remove it and --resume to continue")
                break
            # Ruled on once per iteration, here with the other conditions that decide whether this iteration may run
            # at all.
            hazard: DeviceHazard | None = self.device_hazard.recheck(iteration)
            if hazard is not None and hazard.exhausted:
                self.termination_reason = "device_contended"
                print(
                    "\nDEVICE CONTENDED: nothing this campaign may clear has "
                    f"released the device in {hazard.blocked_iterations} "
                    "iterations, so no measurement can be trusted; stopping "
                    f"rather than spending the budget on unmeasurable "
                    f"iterations. {hazard.describe()}"
                )
                break

            # Price the round before anything is spent on it.
            self._close_round()
            round_lanes = self.ic.lanes
            if not self._lane_queue:
                admitted_lanes = self._admit_next_round(iteration)
                if admitted_lanes is None:
                    break
                round_lanes = admitted_lanes
                self._open_round(iteration, lanes=round_lanes)

            supervisor_due = False
            supervisor_reason = ""
            if self.monitor is not None and supervisor_fn is not None:
                supervisor_due, supervisor_reason = self.monitor.should_intervene(iteration)

            # Resolve Analysis before any Supervisor or planning call.
            if analysis_service is not None:
                await self._resolve_analysis_context(
                    analysis_service,
                    supervisor_due=supervisor_due,
                    iteration=iteration,
                )

            # Self-supervision (AVO): when supervised, a stall triggers a reviewer that injects fresh directions and
            # the loop ALWAYS CONTINUES — it never self-terminates on stall.
            if self.monitor is not None and supervisor_fn is not None:
                if supervisor_due:
                    print(f"\n[supervisor] intervening at iteration {iteration}: {supervisor_reason}")
                    memo = ""
                    try:
                        try:
                            evidence_context = self._build_supervisor_evidence_context(iteration)
                        except Exception:
                            evidence_context = ""
                            log.debug(
                                "could not build supervisor evidence",
                                exc_info=True,
                            )
                        # A new review attempt supersedes the prior stall episode's ruling even when the backend
                        # returns empty.
                        self._expire_supervisor_ruling()
                        self.monitor.mark_attempted(iteration)
                        apply_supervisor_attempt(
                            self.run_state,
                            iteration=iteration,
                        )
                        self.state_store.append_event(
                            make_event(
                                "supervisor_attempt",
                                iteration,
                                reason=supervisor_reason,
                            )
                        )
                        self.state_store.save(self.run_state)
                        memo = await supervisor_fn(
                            digest=digest,
                            reason=supervisor_reason,
                            workspace=self.ic.workspace_dir,
                            iteration=iteration,
                            evidence_context=evidence_context,
                        )
                    except Exception as e:
                        print(f"  [supervisor] failed ({e}); continuing without a memo")
                    finally:
                        self._checkpoint_llm_usage()
                    memo = memo or ""
                    if memo.strip():
                        interaction_path, ruling_path = persist_supervisor_ruling(
                            self.ic.workspace_dir,
                            iteration,
                            supervisor_reason,
                            memo,
                        )
                        self._supervisor_ruling = memo
                        print(f"  [supervisor] injected free-form ruling: {len(self._supervisor_ruling)} chars")
                        try:
                            self.state_store.append_event(
                                make_event(
                                    "supervisor_ruling",
                                    iteration,
                                    reason=supervisor_reason,
                                    ruling_len=len(self._supervisor_ruling),
                                    interaction_path=(
                                        str(interaction_path.relative_to(Path(self.ic.workspace_dir)))
                                        if interaction_path is not None
                                        else None
                                    ),
                                    ruling_path=(
                                        str(ruling_path.relative_to(Path(self.ic.workspace_dir)))
                                        if ruling_path is not None
                                        else None
                                    ),
                                )
                            )
                        except Exception:  # noqa: BLE001 - best-effort
                            log.debug("run_state: supervisor event append failed", exc_info=True)
                    else:
                        print("  [supervisor] no new ruling returned; continuing without an active ruling")
                    if memo.strip():
                        self.monitor.mark_intervened(iteration)
                        try:
                            apply_supervisor_intervention(
                                self.run_state,
                                iteration=iteration,
                                stall_threshold=self.ic.supervise_after,
                            )
                            self.state_store.save(self.run_state)
                        except Exception:  # noqa: BLE001 - best-effort
                            log.debug(
                                "run_state: supervisor reset/save failed",
                                exc_info=True,
                            )

            self._update_search_policy(iteration)

            print(
                f"--- Iteration {iteration} "
                f"(best mean case speedup: {self.best_mean_case_speedup:.6f}x, "
                f"remaining: {self._time_remaining() / 60:.0f} min) ---"
                if self.best_mean_case_speedup is not None
                else f"--- Iteration {iteration} ---"
            )

            # Re-scope the ownership boundary to this iteration: untracked files already here are the operator's or an
            # earlier round's, and this iteration's REVERT must not delete them.
            snapshot = self._untracked_snapshot()
            if snapshot is not None:
                self._pre_untracked = snapshot

            # Durable per-iteration marker (facts only; detail lives in files).
            self.run_state.iteration = iteration
            try:
                self.state_store.append_event(
                    make_event(
                        "iteration_started",
                        iteration,
                        best_before_ms=self.best_wall_ms,
                        best_before_mean_case_speedup=self.best_mean_case_speedup,
                        phase=self.run_state.phase,
                    )
                )
            except Exception:  # noqa: BLE001 - best-effort
                log.debug("run_state: iteration_started append failed", exc_info=True)

            # Agent proposes modification
            session_sink: dict = {}
            optimization_plan_path = ""
            optimization_plan_executable = False
            # What a fan-out round leaves this iteration holding, so the single-session path below spends it instead
            # of buying it again.
            fan_out_plan: HeldRound | None = None
            if hazard is None and (
                not self._lane_queue
                and round_lanes > 1
                and agent_factory is not None
                and agent_fn is not None
                and orchestration_service is not None
            ):
                fan_out_plan = await self._fan_out_round(
                    iteration=iteration,
                    orchestration_service=orchestration_service,
                    agent_factory=agent_factory,
                    lanes=round_lanes,
                )
                # The round's own lanes share one device with the canonical measurement, so a lane whose teardown
                # could not clear it has just refused this iteration -- whatever its siblings produced.
                hazard = self.device_hazard.live
                if self._refused_round:
                    # Planning cost more than the round had left.
                    break
            # Stacking two rejected gains costs a measurement but no session, so it is tried before spending another
            # Implementer round on a search that has stopped producing a new best -- and, once that search has
            # stalled, before draining a candidate the same search bought.
            merge_pair = None if hazard or fan_out_plan is not None else self._select_merge_attempt()
            merge_refusal = "" if merge_pair is None else self._merge_attempt_refusal()
            lane_queue_depth = 0 if hazard else len(self._lane_queue)
            if merge_refusal:
                merge_diff, merge_obstacle = "", merge_refusal
            else:
                merge_diff, merge_obstacle = self._stage_merge_attempt(merge_pair)
            self._merge_precedence_streak = self._merge_precedence_streak + 1 if merge_diff else 0
            if merge_diff and lane_queue_depth:
                # Distinct from ``merge_attempt_staged``, and not foldable into it: that counts every stack measured,
                # this counts the ones that went ahead of a queue a round already paid for, which is the only thing
                # precedence can cost.
                print(f"  [merge] precedence over a lane queue {lane_queue_depth} deep")
                self.state_store.append_event(
                    make_event(
                        "merge_took_precedence",
                        iteration,
                        lane_queue_depth=lane_queue_depth,
                        first_iteration=merge_pair[0].iteration,
                        second_iteration=merge_pair[1].iteration,
                        unresolved_stall_iters=(self.run_state.stall.unresolved_stall_iters),
                    )
                )
            if merge_pair is not None and merge_obstacle:
                self._decline_merge_attempt(
                    iteration,
                    merge_pair,
                    merge_obstacle,
                    # A refusal did not reach the pair's diffs, so it is not evidence about them and must not be
                    # remembered against them.
                    about_the_iteration=bool(merge_refusal),
                )
            # A fan-out round already paid for these candidates, so they are measured before anything new is planned.
            queued_lane = None if hazard or merge_diff else self._take_lane_candidate()
            if hazard is not None:
                unmeasured_result = self._unmeasurable_on_a_held_device(
                    iteration=iteration,
                    detail=hazard.describe(),
                    session_sink=session_sink,
                )
                commit_hash = ""
                rationale = "device held; nothing was planned, run or measured"
                attempt_source = ""
                attempt_diff = ""
                reusable_benchmark = None
            elif queued_lane is not None:
                unmeasured_result = None
                commit_hash = ""
                rationale = f"lane {queued_lane.lane_id} of a fan-out round"
                attempt_source = self._read_kernel_source()
                attempt_diff = self._working_tree_diff()
                reusable_benchmark = None
                session_sink["plan"] = queued_lane.plan
                print(f"  [lane {queued_lane.lane_id}] measuring queued candidate")
            elif merge_diff and merge_pair is not None:
                unmeasured_result = None
                commit_hash = ""
                rationale = (
                    f"stacked iterations {merge_pair[0].iteration} and "
                    f"{merge_pair[1].iteration}; no Implementer session"
                )
                attempt_source = self._read_kernel_source()
                attempt_diff = merge_diff
                reusable_benchmark = None
                session_sink["plan"] = merge_plan(*merge_pair)
                print(f"  [merge] {session_sink['plan']}")
                # How often the mechanism engaged.
                self.state_store.append_event(
                    make_event(
                        "merge_attempt_staged",
                        iteration,
                        first_iteration=merge_pair[0].iteration,
                        second_iteration=merge_pair[1].iteration,
                        cases=sorted(merge_pair[0].winning_cases | merge_pair[1].winning_cases),
                        unresolved_stall_iters=(self.run_state.stall.unresolved_stall_iters),
                    )
                )
            elif agent_fn is not None:
                print("  [agent] Querying agent for kernel modification...")
                if orchestration_service is not None:
                    if fan_out_plan is not None:
                        plan_path, orchestration_error = fan_out_plan
                    else:
                        print("  [orchestration] analyzing and dispatching specialists...")
                        self._last_orchestration_plan_executable = None
                        plan_path, orchestration_error = await self._plan_round(
                            iteration=iteration,
                            orchestration_service=orchestration_service,
                        )
                    if plan_path is None:
                        result = IterationResult(
                            iteration=iteration,
                            duration_sec=0.0,
                            validation_passed=False,
                            validation_summary=(f"ORCHESTRATION ERROR: {orchestration_error}"),
                            session_end_reason="orchestration_error",
                        )
                        self.results.append(result)
                        self._apply_iteration_planning_state(
                            optimization_plan_created=False,
                        )
                        self._record_iteration_outcome(
                            result,
                            decision_label="ORCHESTRATION_ERROR",
                        )
                        await self._record_lesson(
                            iteration=iteration,
                            result=result,
                            decision="ORCHESTRATION_ERROR",
                            session_sink=session_sink,
                        )
                        self._record_iteration_handoff(
                            iteration=iteration,
                            decision="ORCHESTRATION_ERROR",
                            optimization_plan_path="",
                            session_sink=session_sink,
                        )
                        self._publish_optimization_history()
                        if on_iteration:
                            on_iteration(result)
                        continue
                    optimization_plan_path = str(plan_path)
                    optimization_plan_executable = (
                        self._last_orchestration_plan_executable
                        if self._last_orchestration_plan_executable is not None
                        else True
                    )
                    complete_orchestration_probe(self.run_state)
                    self.state_store.save(self.run_state)
                # The last point before the round buys its session, and the first at which what planning cost is a
                # measurement rather than an estimate.
                if not self._admit_dispatch(iteration):
                    break
                session_sink["session_started"] = True
                # Cross-iteration experience assembled from complementary sources (AVO-style lineage view), each
                # carrying what the others cannot: * the candidate ARCHIVE digest — the trajectory table + full diffs
                # of the best/near-miss/recent attempts + a pointer to the on-disk archive so the agent can Read any
                # prior kernel. * the LESSON documents — what each recent session actually explored, in its own words,
                # including the directions it abandoned (which leave no diff behind). * the experience LEDGER —
                # objective toolchain observations distilled from machine-verified failure signatures.
                lh_header = _long_horizon_header(
                    self.run_state,
                    self.state_store,
                    self.handoff_store,
                )

                # Lesson documents from the most recent iterations, verbatim, plus the absolute path of the directory
                # holding every past one.
                lessons_txt = ""
                if getattr(self, "lessons", None) is not None:
                    try:
                        lessons_txt = self.lessons.render_for_prompt(
                            current_cases=self._scored_case_ids(),
                            kernel_source=self._kernel_source_for_scope(),
                        )
                    except Exception:  # noqa: BLE001 - best-effort
                        log.debug("lessons: prompt render failed", exc_info=True)

                ledger_txt = ""
                if self.ledger:
                    # The session narrative lives in the lesson documents, so the ledger contributes only objective
                    # toolchain observations once any lesson is available.
                    ledger_txt = self.ledger.render_for_prompt(
                        include_recent=(not lessons_txt and (bool(lh_header) or not bool(digest)))
                    )
                if lh_header:
                    history = "\n\n".join(p for p in (lessons_txt, ledger_txt) if p)
                    print(
                        f"  [agent] injected long-horizon header: {len(lh_header)} chars "
                        f"(digest reserved for supervisor)"
                    )
                else:
                    history = "\n\n".join(p for p in (digest, lessons_txt, ledger_txt) if p)
                    if digest:
                        n_cand = len(self.archive.load_index())
                        print(
                            f"  [agent] injected lineage digest: {len(digest)} chars, "
                            f"{n_cand} prior candidates archived"
                        )
                if lessons_txt:
                    print(f"  [agent] injected lesson documents: {len(lessons_txt)} chars")
                if not history:
                    history = "\n".join(_compact_history_entry(r) for r in self.results[-5:])
                analysis_evidence = self._render_analysis_evidence_for_implementer()
                if analysis_evidence:
                    history = f"{analysis_evidence}\n\n{history}"
                    print(f"  [agent] injected Analysis evidence: {len(analysis_evidence)} chars")
                coverage_block = self._render_case_config_coverage()
                if coverage_block:
                    history = f"{coverage_block}\n\n{history}"
                    print(f"  [agent] injected per-case configuration coverage: {len(coverage_block)} chars")
                new_file_block = self._render_uncommittable_new_paths()
                if new_file_block:
                    history = f"{new_file_block}\n\n{history}"
                if self._search_policy_decision is not None:
                    policy = self._search_policy_decision
                    policy_lines = [
                        "## Search Policy (deterministic outer-loop decision)",
                        f"Mode: {policy.mode}",
                        f"Objective: {policy.objective_kind}",
                        "Reasons: " + ", ".join(policy.reason_codes),
                    ]
                    policy_lines.append(
                        f"Mode residence remaining after this iteration: {policy.residence_iterations_remaining}"
                    )
                    history = "\n".join(policy_lines) + "\n\n" + history
                # The latest free-form Supervisor Ruling is durable across KEEP and resume.
                if self._supervisor_ruling:
                    history = (
                        "## Latest Supervisor Ruling\n"
                        "This review is the current planning authority. It "
                        "overrides subjective recommendations or conclusions in "
                        "historical lesson records, but never overrides objective "
                        "validation or measurement facts.\n\n"
                        f"{self._supervisor_ruling}\n\n{history}"
                    )
                # The long-horizon header (rendered above) goes at the very TOP, above the supervisor/analyst/pmc
                # prepends, as the compact memory frame the implementer reads first.
                if lh_header:
                    history = f"{lh_header}\n\n{history}"
                if optimization_plan_path:
                    ruling_instruction = (
                        "The plan was synthesized from current evidence and the "
                        "latest Supervisor Ruling. If any plan statement conflicts "
                        "with that ruling, follow the ruling. "
                        if self._supervisor_ruling
                        else "The plan was synthesized from current evidence. "
                    )
                    history = (
                        "## Required optimization plan\n"
                        f"Read {optimization_plan_path} and execute the integrated "
                        "plan it contains. "
                        f"{ruling_instruction}"
                        "Historical lesson records are evidence, not instructions.\n\n"
                        f"{history}"
                    )
                # Pass extras only to agent_fns that declare them: the in-session gate uses the current best mean case
                # speedup and immutable pristine per-case timings; session_sink hands back findings and the
                # resumed-session factual-record callback.
                extra_kwargs = {}
                try:
                    params = inspect.signature(agent_fn).parameters
                    if "baseline_case_times" in params:
                        extra_kwargs["baseline_case_times"] = dict(self._baseline_case_times)
                    if "best_mean_case_speedup" in params:
                        extra_kwargs["best_mean_case_speedup"] = self.best_mean_case_speedup
                    if "session_sink" in params:
                        extra_kwargs["session_sink"] = session_sink
                except (ValueError, TypeError):
                    log.debug("could not introspect agent_fn signature", exc_info=True)
                agent_error = None
                try:
                    rationale = await agent_fn(
                        self.ic.kernel_file,
                        history,
                        **extra_kwargs,
                    )
                    print(f"  [agent] Rationale: {rationale[:200]}")
                except Exception as e:
                    agent_error = e
                    print(f"  [agent] ERROR: {e}")
                    rationale = f"agent session ended with error after edits: {e}"
                    session_sink.setdefault("end_reason", "sdk_error")
                    session_sink.setdefault(
                        "findings",
                        f"Agent session error before outer validation: {e}",
                    )
                finally:
                    # The SDK result stream has completed (or unwound).
                    self._checkpoint_llm_usage()

                commit_hash = ""
                # Either kind of "this candidate exists but must not be measured": protected state was tainted, or the
                # workspace is still busy.
                unmeasured_result: IterationResult | None = None
                if session_sink.get("integrity_violation") is True:
                    # Capture evidence before restoration.
                    attempt_diff = self._working_tree_diff()
                    attempt_source = self._read_kernel_source()
                    integrity_reason = str(session_sink.get("integrity_reason") or "protected workspace state changed")
                    restore_errors: list[str] = []
                    restore = session_sink.get("integrity_restore")
                    if callable(restore):
                        try:
                            restore()
                        except Exception as error:  # noqa: BLE001
                            restore_errors.append(f"protected snapshot restore failed: {type(error).__name__}: {error}")
                    else:
                        restore_errors.append("protected snapshot restore callback unavailable")
                    try:
                        self._git_discard_worktree()
                    except Exception as error:  # noqa: BLE001
                        restore_errors.append(f"tracked candidate restore failed: {type(error).__name__}: {error}")
                    if not restore_errors:
                        try:
                            self._validate_driver_integrity(self.run_state)
                        except Exception as error:  # noqa: BLE001
                            restore_errors.append(str(error))
                    summary = (
                        "REVERT (protected integrity violation): canonical "
                        "correctness and benchmark were skipped before executing "
                        f"the measurement surface. {integrity_reason}"
                    )
                    if restore_errors:
                        summary += " Restoration errors: " + "; ".join(restore_errors)
                    session_sink["findings"] = "\n---\n".join(
                        part
                        for part in (
                            str(session_sink.get("findings") or ""),
                            summary,
                        )
                        if part
                    )
                    unmeasured_result = IterationResult(
                        iteration=iteration,
                        duration_sec=0.0,
                        validation_passed=False,
                        validation_summary=summary,
                        kept=False,
                        integrity_violation=True,
                    )
                    reusable_benchmark = None
                    print("  [REVERT] Protected integrity violation; canonical validation skipped")
                elif str(session_sink.get("workspace_contention") or ""):
                    # The session's own processes are still running in the workspace, or someone else's are and they
                    # are not ours to kill.
                    contention = str(session_sink["workspace_contention"])
                    # Nothing about the end of this iteration makes those processes leave, so the refusal is recorded
                    # and re-checked rather than forgotten here.
                    self.device_hazard.record(
                        iteration=iteration,
                        detail=contention,
                        pids=processes_under(self.ic.workspace_dir),
                    )
                    attempt_diff = self._working_tree_diff()
                    attempt_source = self._read_kernel_source()
                    summary = (
                        "REVERT (workspace contention): canonical correctness "
                        "and benchmark were skipped because the session's "
                        f"workspace could not be cleared. {contention}"
                    )
                    try:
                        # The candidate itself may be sound, but nothing here can establish that, and HEAD has to stay
                        # at the last measured best rather than carry an unmeasured diff into the next iteration.
                        self._git_discard_worktree()
                    except Exception as error:  # noqa: BLE001
                        summary += f" Candidate restore failed: {type(error).__name__}: {error}"
                    session_sink["findings"] = "\n---\n".join(
                        part
                        for part in (
                            str(session_sink.get("findings") or ""),
                            summary,
                        )
                        if part
                    )
                    unmeasured_result = IterationResult(
                        iteration=iteration,
                        duration_sec=0.0,
                        validation_passed=False,
                        validation_summary=summary,
                        kept=False,
                        workspace_contention=contention,
                    )
                    reusable_benchmark = None
                    print("  [REVERT] Workspace still contended; canonical measurement skipped")
                else:
                    # The driver is the measurement boundary and must remain byte-for-byte canonical.
                    self._validate_driver_integrity(self.run_state)

                    # Keep HEAD at the last validated best state while this candidate remains unverified.
                    attempt_diff = self._working_tree_diff()
                    if not attempt_diff.strip():
                        # An outage leaves the same empty diff as a deliberate no-op.
                        api_failed = session_sink.get("end_reason") == EXHAUSTED_END_REASON
                        # A file the agent created is not in the tracked diff.
                        new_files_only = self._new_paths_need_discard()
                        if new_files_only:
                            self._git_discard_worktree()
                        if agent_error:
                            decision_label = "AGENT_ERROR"
                            summary = f"agent_fn error: {agent_error}"
                        elif api_failed:
                            decision_label = "API_ERROR"
                            summary = (
                                "LLM API never answered this Session; no candidate "
                                "was attempted (not an optimization result)"
                            )
                        elif new_files_only:
                            decision_label = "NO_CHANGES"
                            summary = (
                                "NO TRACKED CHANGES: the whole candidate was in "
                                "new file(s) matching "
                                f"{', '.join(self.ic.commit_new_paths)}. A KEEP "
                                "commit is built from the tracked diff, so an "
                                "allowlisted new file can only ship alongside a "
                                "tracked edit. The file was taken off the tree "
                                "rather than measured with the next candidate."
                            )
                        else:
                            decision_label = "NO_CHANGES"
                            summary = "NO TRACKED CHANGES: agent produced no candidate diff"
                        print("  [agent] No tracked source changes; skipping candidate")
                        result = IterationResult(
                            iteration=iteration,
                            duration_sec=0.0,
                            validation_passed=False,
                            validation_summary=summary,
                            kept=False,
                        )
                        result.agent_rationale = rationale
                        result.session_end_reason = session_sink.get("end_reason", "")
                        result.turns = session_sink.get("turns")
                        self.results.append(result)
                        self._apply_iteration_planning_state(
                            optimization_plan_created=(optimization_plan_executable),
                        )
                        self._record_iteration_outcome(
                            result,
                            plan=session_sink.get("plan", ""),
                            decision_label=decision_label,
                        )
                        await self._record_lesson(
                            iteration=iteration,
                            result=result,
                            decision=decision_label,
                            session_sink=session_sink,
                        )
                        self._record_iteration_handoff(
                            iteration=iteration,
                            decision=decision_label,
                            optimization_plan_path=optimization_plan_path,
                            session_sink=session_sink,
                        )
                        self._publish_optimization_history()
                        if on_iteration:
                            on_iteration(result)
                        continue
                    reusable_benchmark = None
                    gate_measurement = session_sink.get("benchmark_measurement")
                    if session_sink.get("gate_passed") is True and self._can_reuse_insession_benchmark(
                        gate_measurement,
                        attempt_diff=attempt_diff,
                    ):
                        reusable_benchmark = gate_measurement
                    # Capture the attempt before any discard or keep commit.
                    attempt_source = self._read_kernel_source()
            else:
                unmeasured_result = None
                commit_hash = ""
                rationale = "no-agent (baseline measurement)"
                attempt_source = ""
                attempt_diff = ""
                reusable_benchmark = None

            # Snapshot the best-so-far BEFORE this iteration is measured, so the archive can record the true delta vs
            # the standard it had to beat (run_one_iteration mutates self.best_wall_ms on an improvement).
            best_before = self.best_wall_ms
            best_mean_case_speedup_before = self.best_mean_case_speedup

            # Run validation + bench.
            if unmeasured_result is not None:
                result = unmeasured_result
            else:
                try:
                    run_kwargs = {}
                    if reusable_benchmark is not None:
                        run_kwargs["benchmark_measurement"] = reusable_benchmark
                    result = await self.run_one_iteration(
                        iteration,
                        plan=session_sink.get("plan", ""),
                        **run_kwargs,
                    )
                except Exception as e:
                    # Turn the crash into a FAILED result (crashed=True) and let it flow through the same
                    # verdict/ledger/archive path.
                    print(f"  [CRASH] iteration {iteration} crashed during run: {e}")
                    result = IterationResult(
                        iteration=iteration,
                        duration_sec=0.0,
                        validation_passed=False,
                        validation_summary=f"iteration crashed: {e}",
                        error_output=traceback.format_exc()[-4000:],
                        kept=False,
                        crashed=True,
                    )
            result.commit_hash = commit_hash
            result.agent_rationale = rationale
            # Session end reason + turns spent (from the in-session gate / SDK via session_sink) — persisted so a
            # run's end-reason distribution (edit cap / turn cap / converged / …) is analyzable.
            result.session_end_reason = session_sink.get("end_reason", "")
            result.turns = session_sink.get("turns")

            if agent_fn is not None:
                self._apply_iteration_planning_state(
                    optimization_plan_created=optimization_plan_executable,
                )

            # Keep or revert — detailed verdict
            pending_keep: dict | None = None
            keep_checkpoint_finalized = False
            elapsed = result.duration_sec
            raw_wall_txt = f"{result.wall_ms:.3f} ms" if result.wall_ms is not None else "unavailable"
            if not result.validation_passed:
                if commit_hash:
                    self._git_revert_last()
                elif attempt_diff or self._new_paths_need_discard():
                    self._git_discard_worktree()
                label = "Iteration crashed" if result.crashed else "Validation failed"
                print(f"  [REVERT] {label} ({elapsed:.0f}s)")
                print(f"           {result.validation_summary.splitlines()[-1] if result.validation_summary else ''}")
            elif not result.kept:
                if commit_hash:
                    self._git_revert_last()
                elif attempt_diff or self._new_paths_need_discard():
                    self._git_discard_worktree()
                speedup_txt = f"{result.mean_case_speedup:.6f}x" if result.mean_case_speedup is not None else "None"
                best_txt = f"{self.best_mean_case_speedup:.6f}x" if self.best_mean_case_speedup is not None else "?"
                print(
                    f"  [REVERT] mean case speedup={speedup_txt} not better than "
                    f"best={best_txt}; raw mean={raw_wall_txt} ({elapsed:.0f}s)"
                )
            elif result.kept:
                # Defer SIGTERM/SIGINT across the durable best-commit publication (main #hardening) so a kill
                # mid-checkpoint cannot leave the pending-keep/run-state half-written.
                with _defer_termination_signals(bool(attempt_diff)):
                    if attempt_diff:
                        try:
                            pending_keep = self._build_pending_keep(
                                result,
                                plan=session_sink.get("plan", ""),
                                best_before=best_before,
                                rationale=rationale,
                                kernel_source=attempt_source,
                            )
                            self._persist_pending_keep(pending_keep)
                            commit_hash = self._git_commit(str(pending_keep["commit_message"]))
                        except Exception as e:
                            result.kept = False
                            result.validation_passed = False
                            result.crashed = True
                            result.validation_summary = f"COMMIT FAILED: {e}"
                            result.error_output = str(e)
                            self._git_discard_all_tracked_changes()
                            self._clear_pending_keep()
                            print(f"  [REVERT] Commit failed after validation ({elapsed:.0f}s)")
                            print(f"           {str(e)[-300:]}")
                        else:
                            result.commit_hash = commit_hash
                            self._promote_best(result)
                            self.best_mean_case_speedup = result.mean_case_speedup
                            self._finalize_keep_checkpoint(
                                result,
                                plan=session_sink.get("plan", ""),
                                best_before=best_before,
                                pending=pending_keep,
                            )
                            keep_checkpoint_finalized = True
                            print(f"  [agent] Committed verified best: {commit_hash[:8]}")
                    else:
                        # No-agent measurement path: there is no candidate diff to commit, but the measurement can
                        # still establish a best.
                        self._promote_best(result)
                        self.best_mean_case_speedup = result.mean_case_speedup
                    # Bridge to the caller's checkpoint sink (main): lets the CLI persist a Hyperloom-recovery
                    # checkpoint JSON alongside our run-state durability.
                    if result.kept and on_best_committed:
                        on_best_committed(result)
                    if keep_checkpoint_finalized:
                        self._clear_pending_keep()
                if result.kept:
                    improvement = ""
                    if best_mean_case_speedup_before and result.mean_case_speedup:
                        pct = (result.mean_case_speedup / best_mean_case_speedup_before - 1.0) * 100
                        improvement = f" ({pct:+.1f}% vs previous best)"
                    snr_str = f" SNR={result.snr_db:.1f}dB" if result.snr_db is not None else ""
                    print(
                        f"  [KEEP] mean case speedup={result.mean_case_speedup:.6f}x "
                        f"— NEW BEST{improvement}; raw mean={raw_wall_txt}"
                        f"{snr_str} ({elapsed:.0f}s)"
                    )
            else:
                print(f"  [SKIP]   wall_ms={result.wall_ms} ({elapsed:.0f}s)")

            # Remote/external work belongs outside the SIGTERM deferral window but still precedes potentially long
            # post-KEEP profiling.
            if keep_checkpoint_finalized and pending_keep is not None:
                self._publish_best_result(
                    result,
                    plan=session_sink.get("plan", ""),
                    best_before=best_before,
                    pending=pending_keep,
                )
            if result.kept and on_best_ready:
                on_best_ready(result)

            if self.experiment:
                try:
                    self.tracker.log_iteration(
                        self.experiment.experiment_id,
                        config={"iteration": iteration, "kept": result.kept},
                        snr_db=result.snr_db,
                        wall_ms=result.wall_ms,
                        mean_case_speedup=result.mean_case_speedup,
                        pmc_diagnosis=result.pmc_diagnosis,
                        vgpr=result.vgpr,
                        decision="KEEP" if result.kept else "REVERT",
                        notes=session_sink.get("plan", ""),
                    )
                except Exception:
                    log.debug("failed to log iteration to experiment tracker", exc_info=True)

            self.results.append(result)

            # Reduce this finished iteration into the durable run state + append a factual event, then checkpoint.
            if not keep_checkpoint_finalized:
                self._record_iteration_outcome(
                    result,
                    plan=session_sink.get("plan", ""),
                )

            # The verdict is now known, so ask the just-finished implementer session to record what it explored, then
            # stamp the measured outcome onto the same document.
            decision_label = _decision_label(result)
            if merge_diff and merge_pair is not None:
                print(
                    f"  [merge] iterations {merge_pair[0].iteration}+"
                    f"{merge_pair[1].iteration} measured stacked: {decision_label}"
                )
                if decision_label == "KEEP":
                    # How often the mechanism changed an outcome, which is the other of the two numbers
                    # ``merge_attempt_staged`` carries.
                    self.state_store.append_event(
                        make_event(
                            "merge_attempt_kept",
                            iteration,
                            first_iteration=merge_pair[0].iteration,
                            second_iteration=merge_pair[1].iteration,
                            mean_case_speedup=result.mean_case_speedup,
                        )
                    )
            iteration_diff_summary = (
                self._diff_summary(commit_hash) if commit_hash else self._diff_summary_from_diff(attempt_diff)
            )
            await self._record_lesson(
                iteration=iteration,
                result=result,
                decision=decision_label,
                session_sink=session_sink,
                diff_summary=iteration_diff_summary,
            )

            # Record this iteration into the cross-iteration experience ledger.
            if getattr(self, "ledger", None) is not None and (commit_hash or attempt_diff):
                try:
                    if not result.validation_passed:
                        last = ""
                        if result.validation_summary:
                            lines = [l for l in result.validation_summary.splitlines() if l.strip()]
                            last = lines[-1][:120] if lines else ""
                        outcome = f"CRASH: {last}" if result.crashed else f"REVERT (validation failed): {last}"
                    elif result.kept:
                        outcome = f"KEPT — new best mean case speedup={result.mean_case_speedup:.6f}x"
                    else:
                        best_txt = (
                            f"{self.best_mean_case_speedup:.6f}x" if self.best_mean_case_speedup is not None else "?"
                        )
                        speedup_txt = (
                            f"{result.mean_case_speedup:.6f}x" if result.mean_case_speedup is not None else "?"
                        )
                        outcome = f"REVERT (correct but not faster): mean case speedup={speedup_txt} vs best={best_txt}"
                    error_text = (
                        session_sink.get("findings", "")
                        or getattr(result, "error_output", "")
                        or (result.validation_summary if not result.validation_passed else "")
                    )
                    self.ledger.record_iteration(
                        iteration=iteration,
                        outcome=outcome,
                        diff_summary=iteration_diff_summary,
                        error_text=error_text,
                    )
                except Exception:
                    log.debug("postmortem logging failed", exc_info=True)

            # Archive the full solution and measurements as a derived view.
            archived_path = None
            if getattr(self, "archive", None) is not None and (commit_hash or attempt_diff):
                try:
                    decision = decision_label
                    validation_text = result.validation_summary or ""
                    if getattr(result, "error_output", ""):
                        validation_text = f"{validation_text}\n\n{result.error_output}".strip()
                    archived_path = self.archive.record(
                        CandidateRecord(
                            iteration=iteration,
                            commit_hash=commit_hash,
                            decision=decision,
                            kept=result.kept,
                            validation_passed=result.validation_passed,
                            wall_ms=result.wall_ms,
                            mean_case_speedup=result.mean_case_speedup,
                            bench_detail=result.bench_detail,
                            snr_db=result.snr_db,
                            vgpr=result.vgpr,
                            pmc_diagnosis=result.pmc_diagnosis,
                            baseline_wall_ms=self.ic.baseline_wall_ms,
                            best_wall_ms_before=best_before,
                            best_mean_case_speedup_before=best_mean_case_speedup_before,
                            plan=session_sink.get("plan", ""),
                            rationale=rationale,
                            kernel_file=self.ic.kernel_file,
                            shape={},
                            kernel_source=attempt_source,
                            change_diff=self._full_diff(commit_hash) if commit_hash else attempt_diff,
                            pmc_full=result.pmc_full,
                            profile_meta=result.profile_meta,
                            validation_text=validation_text,
                            session_end_reason=result.session_end_reason,
                            turns=result.turns,
                        )
                    )
                    if keep_checkpoint_finalized and archived_path is None:
                        raise RuntimeError("candidate archive returned no published path")
                except Exception as e:
                    if keep_checkpoint_finalized:
                        self.persistence_degraded = True
                        self.persistence_errors.append(f"archive derived KEEP view iteration {iteration}: {e}")
                        self.persistence_errors = self.persistence_errors[-10:]
                    log.debug("could not archive iteration %s: %s", iteration, e)

            if not keep_checkpoint_finalized:
                self._publish_best_result(
                    result,
                    plan=session_sink.get("plan", ""),
                    best_before=best_before,
                )
            self._record_iteration_handoff(
                iteration=iteration,
                decision=decision_label,
                optimization_plan_path=optimization_plan_path,
                session_sink=session_sink,
                archived_path=archived_path,
            )
            self._publish_optimization_history()

            if on_iteration:
                on_iteration(result)

            # A KEEP makes the prior evidence stale but does not discard its paths.
            if result.kept:
                self._analysis_bundle = None

        # Whatever the last iteration cost belongs to this campaign's history even though no further round will read
        # it: a resumed session will.
        self._close_round()
        self.best_publisher.refresh_round_budget(self._round_budget_summary())

        # Persist the terminal control state so a resume/inspection sees why the run ended and what the final best
        # was.
        try:
            terminal_reason = self.termination_reason or self.run_state.termination_reason or "unknown"
            finish_session(
                self.run_state,
                status=(SESSION_COMPLETED if terminal_reason == "gate_met" else SESSION_PAUSED),
                reason=terminal_reason,
            )
            head_out = self._git("rev-parse", "HEAD").strip()
            if head_out:
                self.run_state.head_commit = head_out.splitlines()[0]
            self.state_store.append_event(
                make_event(
                    "run_terminated",
                    self.run_state.iteration,
                    reason=terminal_reason,
                    best_wall_ms=self.best_wall_ms,
                    best_mean_case_speedup=self.best_mean_case_speedup,
                )
            )
            self.state_store.save(self.run_state)
        except Exception:  # noqa: BLE001 - best-effort
            log.debug("run_state: terminal save failed", exc_info=True)
        self.persistence_degraded = self.persistence_degraded or self.state_store.degraded
        self.persistence_errors = (self.persistence_errors + self.state_store.persistence_errors)[-10:]

        # Persist the run's total LLM token spend onto the experiment record so external callers can read the token
        # cost.
        self._checkpoint_llm_usage()

        if self.experiment:
            try:
                self.tracker.mark_complete(self.experiment.experiment_id)
            except Exception:
                log.debug("failed to mark experiment complete", exc_info=True)

        # Final report
        total_time = time.time() - self.start_time
        kept_count = sum(1 for r in self.results if r.kept)
        print(f"\n{'=' * 60}")
        print("Autonomous loop complete")
        print(f"  Iterations: {len(self.results)}")
        print(f"  Kept: {kept_count}, Reverted: {len(self.results) - kept_count}")
        if self.monitor is not None:
            print(f"  Supervisor interventions: {self.monitor.intervention_count}")
        print(f"  Best mean case speedup: {self.best_mean_case_speedup}x")
        print(
            "  Selected candidate raw mean_ms (diagnostic; not monotonic, but "
            "the published manifest withdraws its improvement badge when it "
            f"contradicts the score): {self.best_wall_ms}"
        )
        print(f"  Total time: {total_time / 60:.1f} minutes")
        costs = self.run_state.round_costs
        if costs.rounds:
            # Campaign totals, not this session's, so they are labelled as such and the share is taken against the
            # campaign clock rather than the ``total_time`` printed just above.
            self._advance_campaign_clock()
            share = costs.planning_share_pct()
            share_text = f" ({share:.0f}% of campaign wall-clock)" if share is not None else ""
            print(
                f"  Rounds planned across the campaign: {costs.rounds}, "
                f"planning {costs.planning_total_sec / 60:.1f} min"
                f"{share_text}, "
                f"round wall-clock {costs.total_sec / 60:.1f} min"
            )
        if self._refused_round:
            print(
                "  ROUND REFUSED FOR BUDGET: the campaign stopped because no "
                "round fit the time left, not because it found nothing — "
                f"{self._refused_round}"
            )
        if self.llm_usage.get("calls"):
            for line in llm_spend_lines(self.llm_usage):
                print(line)
            # Printed next to the bill because that is where a reader asking
            # "why was this expensive" is looking. rtk degrades silently by
            # design; the degradation should not also be invisible.
            rtk_warning = rtk_unavailable_warning()
            if rtk_warning:
                print(f"  NOTE: {rtk_warning}")
        print(f"  Experiment: {self.experiment.experiment_id}")

        return self.results


# Decision labels that can possibly mean nothing came out worse this iteration: a kept candidate, and the ones that
# mean no candidate was ever measured.
_LABELS_WITHOUT_A_MEASURED_NEGATIVE = frozenset(
    {
        "KEEP",
        "NO_CHANGES",
        "API_ERROR",
        "AGENT_ERROR",
        "ORCHESTRATION_ERROR",
    }
)


def _decision_label(result: IterationResult) -> str:
    """The canonical keep/revert label for one finished attempt."""
    if result.integrity_violation:
        return "REVERT_INTEGRITY"
    if result.workspace_contention:
        return "REVERT_CONTENDED"
    if result.crashed:
        return "CRASH"
    if not result.validation_passed:
        if (result.validation_summary or "").startswith("BUILD FAILED"):
            return "BUILD_FAILED"
        if result.validation_outcome == "timeout":
            return "REVERT_VALIDATION_TIMEOUT"
        if result.validation_outcome in {"driver_error", "invalid_result"}:
            return "REVERT_VALIDATION_ERROR"
        return "REVERT_VALIDATION"
    return "KEEP" if result.kept else "REVERT_PERF"


def _long_horizon_header(
    state: RunState,
    store: LoopStateStore,
    handoff_store: HandoffStore | None = None,
) -> str:
    """The compact long-horizon header for the Implementer prompt, or \"\"."""
    outcomes = store.recent_results(LONG_HORIZON_OUTCOME_WINDOW)
    try:
        return render_long_horizon_header(
            state,
            outcomes,
            include_handoffs=bool(handoff_store and handoff_store.latest()),
        )
    except Exception:  # noqa: BLE001 - best-effort
        log.debug("run_state: prompt view render failed", exc_info=True)
        return ""


def _compact_history_entry(r: IterationResult) -> str:
    """One-line history entry for the agent prompt — keeps tokens bounded."""
    rat = (r.agent_rationale or "").replace("\n", " ").strip()[:80]
    if not r.validation_passed:
        last = ""
        if r.validation_summary:
            lines = [ln for ln in r.validation_summary.splitlines() if ln.strip()]
            if lines:
                last = lines[-1][:60]
        return f"iter {r.iteration} REVERT(validation) last='{last}' rat='{rat}'"
    parts = [f"iter {r.iteration}", "KEEP" if r.kept else "REVERT(perf)"]
    if r.mean_case_speedup is not None:
        parts.append(f"mean_case_speedup={r.mean_case_speedup:.4f}x")
    if r.wall_ms is not None:
        parts.append(f"wall={r.wall_ms:.3f}ms")
    if r.snr_db is not None:
        parts.append(f"snr={r.snr_db:.1f}dB")
    if r.vgpr:
        parts.append(f"vgpr={r.vgpr}")
    if r.pmc_diagnosis:
        parts.append(f"pmc='{r.pmc_diagnosis[:40]}'")
    if rat:
        parts.append(f"rat='{rat}'")
    return " ".join(parts)
