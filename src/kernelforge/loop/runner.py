# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable, evidence-driven autonomous kernel optimization loop.

Core pattern:
  1. Analyze the current canonical commit and build a durable evidence bundle.
  2. Select EXPLOIT or DIVERSIFY and synthesize one optimization plan.
  3. The Implementer edits tracked implementation files in the working tree.
  4. Run the driver-owned correctness suite and three independent benchmarks.
  5. Commit only a verified KEEP; otherwise restore the canonical working tree.
  6. Archive the attempt, lesson, handoff, and search state, then continue while
     the campaign budget can admit another session.

Key properties:
  - Git HEAD always identifies the latest validated canonical implementation.
  - Every changed attempt and its measurements are archived.
  - Commit-bound Analysis and Orchestration are resumable and evidence-backed.
  - Stalls trigger Supervisor-guided direction changes instead of plateau stops.
"""

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
from kernelforge.learning.auto_evolve import AutoEvolver
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
from kernelforge.rtk import smart_wrap
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

# How many recent iteration outcomes the long-horizon prompt header is built
# from. Counted in outcomes rather than in raw log events: one iteration writes
# several events (search policy decision, iteration_started, analysis result,
# outcome), so an event-counted window of the same length reaches two to four
# outcomes and starves both readers of it -- the header's recent-attempt lines
# and the measured mean case speedup it labels each pinned iteration with. Sized
# from the two budgets it has to serve rather than restating a number, so the
# window spans every pin the state can hold and can still fill the header's
# recent-attempt list.
LONG_HORIZON_OUTCOME_WINDOW = max(
    MAX_PINNED_ITERATIONS,
    MAX_RECENT_ATTEMPT_LINES,
)

# Where a campaign writes its own output inside the workspace. Untracked files
# under it are the loop's, not the agent's, so they are neither committed nor
# reported as something the agent left behind.
LOOP_ARTIFACT_ROOT = "forge_experiments"

# How far a KEEP has to improve a case's measured time before that case counts
# as one the KEEP's configuration was chosen for. Deliberately NOT derived from
# the KEEP gate: that gate is an admission rule on the suite MEAN, required to
# hold across every independent measurement, while this is signal detection on
# ONE case's median. Both now charge a margin against measured dispersion, but
# against different dispersions -- reusing the gate's called a case covered on a
# move any suite with per-case run-to-run noise produces by itself, and the
# planner was handed that as measured fact.
#
# The primary test is therefore the case's own dispersion: the improvement has
# to hold in every independent measurement of the KEEP and to exceed the spread
# those measurements show (times ``CONFIG_COVERAGE_DISPERSION_MULTIPLE``). The
# ratio below is the floor underneath it, and the whole test when a KEEP was
# recorded with aggregate case medians only.
CONFIG_COVERAGE_MIN_MOVE_RATIO = 0.01
CONFIG_COVERAGE_DISPERSION_MULTIPLE = 1.0


def _measurement_case_times(
    bench_detail: dict | None,
) -> dict[str, tuple[float, ...]]:
    """Per-case times from each independent measurement of one bench.

    ``bench_detail["case_times"]`` is the aggregate the loop scores on; the
    same run also records every independent measurement it was aggregated
    from, and that is where a case's run-to-run spread is readable. Returns an
    empty mapping for a record that has no per-measurement detail (a KEEP
    replayed from a pending journal, for instance) -- the caller falls back to
    the floor ratio rather than inventing a dispersion.
    """
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
    """The sigma the KEEP bar was charged to, and how it was arrived at.

    ``sigma`` is what :func:`~kernelforge.loop.scoring.required_keep_speedup`
    was given. When no case dominated it is the plain spread of the three
    aggregate scores, byte for byte the number the gate used before per-case
    attribution existed. ``unstable`` marks the honest failure: the dominant
    case was re-measured to the bound, still sets the bar, and the larger sample
    did not bring its spread down. The bar then stands inflated by one case and
    the operator is told so, rather than the case being quietly dropped from a
    score the arena owns and forge does not get to redefine.

    ``unstable`` is a weak signal and is reported as one. Simulating 2000
    candidates on the GQA campaign's own per-case noise -- stationary Gaussian,
    so no case is unstable by construction -- the flag still fires on 56% of
    them, because a nine-sample spread exceeds a three-sample spread roughly as
    often as not. It says the bar was set by one case, which is certain; it
    does not establish that the case is unstable, and nothing downstream treats
    it as though it did.

    ``detail`` carries why a resolution stopped where it did, including on the
    paths that name no dominant case, so a fallback to the aggregate estimate
    is never silent.
    """

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
    """The clause the bench line carries when one case set the bar.

    Empty when the split came out even, so an ordinary REVERT reads exactly as
    it read before. A REVERT is the thing a human debugs from, and
    "sigma=0.0132" does not distinguish a candidate that failed from a 10 us
    case that drew a wide sample this round. A candidate whose per-case times
    would not resolve a split at all is not silent: it fell back to the
    aggregate estimate, which is a weaker reading of the same number, and the
    line says which.

    Benches bought and samples used are printed separately because they can
    disagree -- a round whose bench failed or came back unusable still cost the
    campaign a whole-suite run, and its samples never reached the estimate.
    One figure standing for both would report an unmeasured thing as measured.
    """
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
    """The driver's own evidence for why a bench run produced nothing usable.

    ``bench_wallclock`` already reports the verdict (``BENCH CRASHED (exit 1)``,
    ``TIMEOUT after Ns``, ``NO TIMING DATA in output``) and, when the driver
    printed anything, the tail of its stdout+stderr. Both were being dropped in
    favour of a fixed line blaming the driver's output format, so a driver that
    crashed on a missing runtime input was reported as one that mis-formatted its
    timings — three rounds of forensics to find a traceback we already had.
    """
    message = str(bench_result.get("message") or "no failure message reported")
    output = str(bench_result.get("output") or "").strip()
    if not output:
        return message
    return f"{message}\n{textwrap.indent(output[-2000:], '    ')}"


def _patch_paths(patch: str, *, cwd: str) -> list[str]:
    """Every workspace path a patch writes, as git itself reads them.

    git parses the patch rather than this function reading its headers, so a
    candidate is judged on the paths git would actually touch, quoting and all.
    ``--numstat`` names only the post-image of a rename, which is exactly how a
    diff moves a file out of the way, so the pre-image lines are read as well.

    A patch git cannot parse raises: which paths it writes is not knowable, so
    it cannot be judged, and it must not be applied.
    """
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
    """One lane's plan, with the one thing about the wrapper the session cannot see.

    The command itself is installed in the lane session's system prompt, which is
    where a requirement that holds for the whole session belongs. What the system
    prompt cannot say is how the wrapper behaves once several lanes are actually
    running: it blocks until the lane ahead has finished benchmarking, and a
    session that reads that pause as a hang will go around it.
    """
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
    """Configuration for the autonomous iteration loop.

    Two kinds of time budget live near each other here and must not be
    confused. The per-step timeouts below (build / validate / bench) are FIXED
    ceilings on a single mechanical operation and are deliberately independent
    of the campaign budget -- how long a compile may take is a property of the
    kernel and backend, not of how long you plan to run. The implementer
    SESSION budget is the opposite: it IS a function of the campaign (sized in
    ``cli._forge_session_timeout_sec`` from ``max_time_hours`` and enforced as
    the agent's ``AgentRunSpec.timeout_sec``), because a session's fair share of
    wall clock only means anything relative to the whole run. It is not stored
    on this config -- it is computed at the CLI boundary and handed to the
    implementer agent directly.
    """

    # Target kernel file (single-file modification)
    kernel_file: str

    # Test driver for validation
    driver_script: str
    # Immutable digest captured by forge-loop's campaign configuration. Empty for
    # the generic loop command, which does not own or adapt its user-supplied driver.
    canonical_driver_sha256: str = ""
    # Original campaign HEAD used to publish a self-contained cumulative best.
    # Empty for generic loop callers, which retain per-commit publication.
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
    # Warm-start measurements are kept separate from the immutable pristine
    # baseline. KEEP/REVERT uses mean case speedup; wall time is diagnostics only.
    warm_start_wall_ms: float | None = None
    warm_start_mean_case_speedup: float | None = None
    warm_start_bench: dict = field(default_factory=dict)
    preloop_baseline_unscored_cases: list[str] = field(default_factory=list)
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB

    # Budget
    max_time_hours: float = 8.0  # overnight budget
    deadline_unix: float | None = None
    # Held back for finalization: the loop will not START another Agent session
    # once what remains falls below it (``_is_budget_exhausted``). An
    # already-started session may finish naturally; this is an admission guard,
    # not a per-session timeout.
    #
    # It is a bound of its OWN, not a term inside the round-admission
    # arithmetic. ``round_budget`` prices a round against the same UNRESERVED
    # remaining time this is compared with -- ``_time_remaining()`` is passed
    # to both admission checks with nothing subtracted -- so a round runs when
    # what remains clears both bounds independently. The larger of the two
    # binds; neither is stacked on the other, and no round has to cover
    # ``reserve + its own cost``.
    #
    # That is deliberate. ``_is_budget_exhausted()`` already holds these
    # seconds back, at every iteration, before any round is priced; adding them
    # again inside a round's own cost charges the campaign for the same reserve
    # twice. The first version of the round guard did exactly that, and it
    # refused rounds that went on to produce a KEEP. ``round_budget``'s module
    # docstring states the same relationship from the other side.
    budget_reserve_sec: int = 1800
    # Per-step timeouts (seconds): FIXED, sensible ceilings — like bench's 300s —
    # and independent of the campaign budget. How long a compile / one validation
    # stage may take is a property of the kernel+backend, NOT of how many
    # iterations you plan to run. Edit these defaults if a backend compiles slowly.
    build_timeout_sec: int = 900  # 15 min compile ceiling
    # Ceiling for the driver-owned complete correctness suite. Cold JIT and
    # multi-case repository tasks can legitimately take many minutes.
    validate_stage_timeout_sec: int = 1800
    bench_timeout_sec: int = 300

    # Git settings
    git_branch: str = "kernel-agent-optimize"
    workspace_dir: str = "."
    # Optional caller-owned ID. A stable ID lets an external timeout owner know
    # the exact experiment JSON path before forge-loop exits.
    experiment_id: str = ""

    # Experiment identity persisted by ExperimentTracker.
    backend: str = ""
    kernel_backend: str = ""

    # Task shape awareness (repository / image_kernel vs single-file snippet).
    # Empty task_type + empty source_files selects the single-file behavior;
    # these are only populated by the forge-loop for multi-file repository tasks
    # so the single-file (e.g. flydsl) path is byte-for-byte unchanged.
    task_type: str = ""
    # Declared implementation entry points used for orientation, profiling, JIT
    # hints, and KB identity. This is not an edit allowlist: any tracked
    # non-protected implementation file may be changed.
    source_files: list[str] = field(default_factory=list)
    # Target kernel/function names the task flagged (host entry + GPU kernels).
    # Used as extra PMC name hints and surfaced to the agent for repo tasks.
    target_functions: list[str] = field(default_factory=list)

    # Supervisor self-supervision (AVO): when the search stalls, a supervisor LLM
    # (a different model family than the implementer — see orchestrator/supervisor.py)
    # reviews the trajectory and injects fresh directions INSTEAD of the loop
    # stopping at the first plateau. Always active whenever a supervisor_fn is
    # passed to run() (the forge-loop always supplies one); these are its tunables.
    supervise_after: int = 3  # consecutive no-improvement iters that trigger
    supervise_cooldown: int = 3  # min iterations between interventions
    max_consecutive_orchestration_errors: int = 3
    # Task context handed to the Analysis Agent and planning chain.
    program_md: str = ""
    # References injected into this run's prompt.
    pr_reference_labels: tuple[str, ...] = ()
    pr_reference_context: str = ""
    # PR refresh event and snapshot deferred until campaign initialization, so a
    # rejected invocation leaves the workspace exactly as it found it.
    pr_kb_event: dict = field(default_factory=dict)
    pr_kb_snapshot: dict = field(default_factory=dict)

    # Human-readable caller identity for the profiled operator. Kept separately
    # from the profiler's low-level target kernel symbol.
    operator_name: str = ""
    implementation_signature: str = ""
    implementation_identity: dict = field(default_factory=dict)
    warm_start_commit: str = ""
    warm_start_solution_slug: str = ""
    # Result returned by the CLI's pre-loop recovery publication. When present,
    # the runner adopts the same warm-start into run_state without republishing
    # iteration 0 under a second campaign/session identity.
    warm_start_publication: dict = field(default_factory=dict)
    # How many times each bench repeats its measurement in-process, reporting the
    # per-case median. 1 selects single-shot behavior (and omits the
    # --repeat flag entirely, so drivers that don't accept it are unaffected).
    bench_repeat: int = 1
    # Implementer lanes run concurrently from one round's analysis, each in its
    # own workspace copy, and each candidate is measured on its own. 1 keeps the
    # single fused plan and single session this loop has always run; fan-out also
    # needs an ``agent_factory``, because a session is bound to its workspace.
    lanes: int = 1
    # Whether a stalled search may spend an iteration measuring two archived
    # rejected gains applied together. On by default: it costs one measurement
    # and no Implementer session, reads per-case evidence the run already paid
    # for, and only fires once consecutive iterations have stopped producing a
    # new best. Unlike ``lanes`` this changes what the ordinary single-session
    # path does, so an operator comparing against an older run needs a way off.
    merge_stacking: bool = True
    # Ranks the driver self-launches (via torchrun) for a collective task. >1
    # switches profiling to the per-rank backend, because wrapping the outer
    # process would only profile the launcher, which runs no kernel. Default 1
    # keeps every single-GPU task on the byte-identical existing path.
    nproc_per_node: int = 1
    # Paths the Implementer may CREATE and still have committed with a KEEP.
    # Untracked files are otherwise never staged (see ``_git_commit``), which is
    # what keeps build artifacts and caches out of a commit; a task whose tuned
    # configuration lives in a generated file it does not yet ship needs a way
    # past that without turning the stage step into ``git add -A``. Entries are
    # workspace-relative POSIX paths or anchored globs (``configs/*.json``); a
    # ``*`` never crosses a directory separator and ``**`` is rejected outright
    # (see ``new_path_allowlist``). Set from ``--commit-new-path`` and carried
    # by the campaign configuration. Nothing else an agent creates is
    # committed, removed by a REVERT, or passed over in silence: it is reported
    # at both.
    commit_new_paths: list[str] = field(default_factory=list)
    # How large a per-case improvement has to be, relative to the case's own
    # time, before a KEEP counts as having been configured for that case. Floor
    # only: the dispersion test in ``_case_move_rule`` is the primary one
    # wherever the KEEP recorded its independent measurements. See
    # ``CONFIG_COVERAGE_MIN_MOVE_RATIO``.
    config_coverage_min_move_ratio: float = CONFIG_COVERAGE_MIN_MOVE_RATIO

    def __post_init__(self) -> None:
        # Validated here rather than at the CLI boundary alone, so a pattern
        # can never reach the commit/delete sites unvalidated -- including via
        # ``dataclasses.replace``.
        self.commit_new_paths = normalize_commit_new_paths(self.commit_new_paths)


class CaseConfigCoverage(NamedTuple):
    """Which scored cases a shipped configuration has ever been chosen for.

    A campaign can raise its mean while a case nobody targeted rides on
    whatever generic path the canonical happens to ship for it. This is the
    ledger that says so: measured, per case, from the KEEPs on record rather
    than from what a plan claimed it would cover.
    """

    # case id -> the last KEEP iteration that moved its measured time.
    covered: dict[str, int]
    # Scored cases no KEEP has moved. Nothing has been tuned for them.
    fallback: tuple[str, ...]
    # Groups of covered cases that every KEEP has moved together. No shipped
    # configuration has yet distinguished the members of one group.
    undifferentiated: tuple[tuple[str, ...], ...]
    # The KEEP iterations this ledger was read off, in order.
    keeps: tuple[int, ...]
    # Scored cases no KEEP on record emitted a timing for, so their coverage is
    # unknown rather than absent.
    unmeasured: tuple[str, ...]
    # KEEP iterations that carried no per-case timings at all. Nothing could be
    # read off them, so every other field is a statement about the rest of the
    # record and not about the session.
    unreadable: tuple[int, ...]
    # Covered cases no KEEP ever tested the dispersion of, because no KEEP that
    # moved them carried its independent measurements. They cleared the floor
    # ratio and nothing more, which is a weaker statement than the rest of
    # ``covered`` and is rendered as one.
    floor_only: tuple[str, ...] = ()


class HeldRound(NamedTuple):
    """What a fan-out round hands the iteration that has to finish without it.

    Every way a round ends without a candidate gives the iteration back to the
    ordinary single-session path, and the only question that path must get
    right is what it may not buy a second time. Three answers, and ``None``
    carries the third:

    - a ``plan_path``, which the round already paid dispatch, every specialist
      and synthesis for, and which that path spends instead of planning again;
    - an ``error``, which is the outage that stopped the round -- handed over
      rather than retried, because the backend that just refused is the one the
      retry would ask;
    - ``None`` instead of this tuple, when the round holds neither and spent
      nothing, so the iteration plans for itself as usual.
    """

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
    # Real error tail from the first failing validation stage (for the ledger);
    # populated on validation failure so gate-off runs still record true errors.
    error_output: str = ""
    # True when the iteration raised an unexpected exception (build/validate/bench
    # crash) rather than merely failing validation. Drives the CRASH archive label
    # so the next agent can see the crashing diff + traceback and avoid repeating it.
    crashed: bool = False
    # Full measurement detail for the candidate archive (not just the scalars
    # above). ``bench_detail`` is the raw bench_wallclock dict (median/min/max/
    # n_samples); ``pmc_full`` is the complete rocprofv3 summary text. Kept
    # separate from the compact fields so the ledger/logs stay small while the
    # archive can persist the full picture for later iterations to inspect.
    bench_detail: dict = field(default_factory=dict)
    pmc_full: str = ""
    # Structured profile metadata (backend, bottleneck, target kernels, roofline
    # dtype/AI/HBM+compute pct, SoL metrics) for the candidate archive's meta.json,
    # so the supervisor / next agent can consume it without parsing prose.
    profile_meta: dict = field(default_factory=dict)
    # Why the agent session ended this iteration (from the in-session gate / SDK):
    # converged / block_budget_exhausted / block_cap / turn_cap / gate_error /
    # agent_stopped / sdk_*. "" when no agent ran. ``turns`` is the SDK turn count
    # actually spent. Persisted for per-iteration end-reason analysis.
    session_end_reason: str = ""
    turns: int | None = None
    # Independent of session termination: a candidate that changed protected
    # measurement state is rejected before any canonical driver is executed.
    integrity_violation: bool = False
    # Why the workspace could not be cleared of leftover processes. Non-empty
    # means the canonical driver was never executed: a measurement taken while
    # something else holds the device is not this candidate's measurement.
    workspace_contention: str = ""


@dataclass(frozen=True)
class WindowGain:
    """The exploit-window trend, or the named reason there is not one.

    A campaign that cannot produce this trend must not look like one whose
    ladder is healthy, so the absence travels as a reason string the decision
    event carries rather than as a field that is simply not written.
    """

    ratio: float | None
    unavailable: str | None

    def __post_init__(self) -> None:
        if (self.ratio is None) == (self.unavailable is None):
            raise ValueError(
                "a window gain is either a ratio or a reason, never both or "
                f"neither: {self.ratio!r} / {self.unavailable!r}"
            )


class IterationLoop(AnalysisRuntimeMixin):
    """Autonomous kernel optimization loop.

    Usage:
        loop = IterationLoop(config, experiment_tracker)
        results = await loop.run(agent_fn)

    Where agent_fn is an async function that:
      1. Reads the current kernel file + experiment history
      2. Proposes a single modification
      3. Returns the rationale for the change
    """

    def __init__(
        self,
        iter_config: IterationConfig,
        tracker: ExperimentTracker,
        config: Config | None = None,
        evolver: AutoEvolver | None = None,
        resume: bool = False,
    ):
        self.ic = iter_config
        # Declared here so persistence works before the methods that populate
        # them have run. The incumbent case medians are required for both normal
        # scoring and resume.
        self._best_case_times: dict[str, float] = {}
        # Pairs this process selected and could not stage. Whether two archived
        # diffs clash is a property of two immutable files, so nothing about a
        # later stall changes the answer -- and because the selector returns the
        # pair covering the most cases, an unremembered failure re-wins every
        # selection and permanently blocks the runner-up that would have staged.
        # In memory rather than on disk: a resumed process re-derives the same
        # verdict for the cost of one attempt and no Implementer session, which
        # is cheaper than either a control-state field to migrate or an archived
        # result that was never measured.
        self._declined_merge_pairs: set[frozenset[int]] = set()
        # Last reported per-case bandwidth. Diagnostic only: never scored,
        # never an incumbent, so it needs no resume semantics.
        self.last_case_bandwidth: dict[str, dict[str, float | int]] = {}
        self._scoring_state_restored = False
        self._unscored_cases: set[str] = set()
        # New files the last commit or discard could act on neither way,
        # because no ``commit_new_paths`` entry admits them. A KEEP cannot
        # carry them and a REVERT does not delete them, so the next
        # Implementer is told they are there.
        self._refused_new_paths: list[str] = []
        # Allowlisted new files a discard left on the tree because they were
        # already there when this loop took the workspace over. They are not
        # the candidate's to delete, but they are on the measured tree, so the
        # next Implementer is told about them too.
        self._retained_new_paths: list[str] = []
        # Why the last new-file enumeration could not be read, "" when it
        # could. An empty refusal list means "nothing to report" only when
        # this is empty as well.
        self._new_paths_unreadable: str = ""
        # Untracked paths present when the current iteration began -- captured
        # again before resume recovery, which also discards and runs before any
        # iteration. Whatever is in it is not this candidate's, and no REVERT
        # of this loop's deletes it. None only if the snapshot itself failed.
        self._pre_untracked: set[str] | None = None
        # Validation and benchmarking invoke the driver with no arguments, so a
        # multi-rank task has no other way to tell it how many ranks to launch.
        # Without this the driver falls back to its own default and measures a
        # different configuration than the profiler, which does get
        # --nproc-per-node and then trips the driver's WORLD_SIZE check.
        #
        # A single-rank task clears it rather than leaving it alone: the
        # variable is process-global, so a second campaign in the same process
        # would otherwise inherit the previous one's rank count and launch
        # torchrun for a task that never asked for it.
        if self.ic.nproc_per_node > 1:
            os.environ["FORGE_NPROC_PER_NODE"] = str(self.ic.nproc_per_node)
        else:
            os.environ.pop("FORGE_NPROC_PER_NODE", None)
        self.tracker = tracker
        self.config = config or Config.from_env()
        self.evolver = evolver or AutoEvolver.from_config(self.config)
        self.resume = resume
        self.experiment: Experiment | None = None
        self.results: list[IterationResult] = []
        # FIXED per-case baseline wall times (case_id -> ms), captured once on the
        # pristine kernel and never overwritten. They are
        # the denominators for the equal-weight mean of per-case speedups that
        # drives keep/revert; persisted in run_state so a resumed session keeps it.
        self._baseline_case_times: dict = dict(self.ic.baseline_case_times)
        # UsageAccumulator for the run (set in run()); lets the analyst fold its
        # token spend into the run total. None when no accumulator is supplied.
        self._usage = None
        self.best_wall_ms: float | None = None
        self.best_mean_case_speedup: float | None = None
        self.start_time: float = 0
        # Total LLM token spend for the run, populated from the UsageAccumulator
        # passed to run() (empty when no agent / no accumulator). Exposed so an
        # in-process caller can read the run's token cost without reloading the
        # experiment JSON.
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
        # A committed KEEP recovered during synchronous resume preflight waits
        # here until the async run can restore its post-KEEP profile/archive.
        self._recovered_pending_keep: tuple[dict, IterationResult] | None = None
        # Why the loop stopped: "gate_met" (target reached), "budget_exhausted"
        # (not enough time to admit another session), or "round_budget_exhausted"
        # (time is left, but not enough to finish even the narrowest round). The
        # loop NEVER self-stops on stall or plateau — a
        # stalled stretch just gets more supervisor directions. Exposed so an
        # in-process caller can record the reason.
        self.termination_reason: str = ""
        # The round currently being timed, opened when an iteration is admitted
        # and closed once the next one starts. Planning accumulates separately
        # because it is the half that decides admission.
        self._round_started_at: float | None = None
        self._round_iteration = 0
        self._round_lanes = 1
        self._round_planning_sec = 0.0
        # What the round spent in the canonical validation and benchmark, which
        # is what prices the next round's dispatch.
        self._round_measurement_sec = 0.0
        # The instant the CAMPAIGN began, which on a resumed session is before
        # this process did: ``start_time`` less the wall-clock earlier sessions
        # already banked. It is the origin of every campaign-cumulative span,
        # so the totals in ``run_state.round_costs`` and the wall-clock they
        # are reported against are read off one clock rather than two. Set in
        # ``run()``, once the state that carries the banked span is loaded.
        self._campaign_started_at: float = 0.0
        # The refusal that ended the campaign, as the line the operator sees,
        # kept so the run summary and the published report can say a round was
        # priced out rather than let it read as a round that found nothing.
        self._refused_round: str = ""
        # Supervisor self-supervision (AVO): the latest free-form ruling to pass
        # verbatim through planning, and the factual progress monitor that decides
        # when to request a new review.
        self._supervisor_ruling: str = ""
        self._latest_optimization_plan_path = ""
        self._last_orchestration_plan_executable: bool | None = None
        # The previous round's Plan Critic ruling, kept for the next round's
        # partition. A critic rules on a plan that has already been synthesized,
        # so a verdict that the route itself is dominated arrives too late to
        # change the round it judged.
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

    def _restore_search_control_snapshot(self, payload: dict) -> None:
        """Restore planning state from a verified pending KEEP journal."""
        control = payload.get("search_control")
        if not isinstance(control, dict):
            return
        self.run_state.diversification_cycle_completed = control.get("diversification_cycle_completed") is True

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
        """Every untracked, non-ignored path in the workspace, as git reports it.

        ``-z`` rather than line splitting: a filename may legally contain a
        newline, and a path parsed into two would drive both a wrong allowlist
        decision and a wrong deletion.
        """
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
        """Snapshot the untracked set an iteration or lane starts from.

        Everything in it predates the candidate, so a REVERT must leave it
        alone however the allowlist is written -- an operator's checked-out
        tuning file is not this iteration's to delete. ``None`` means the
        snapshot could not be taken at all; the discard side says what it does
        then.
        """
        try:
            return set(self._list_untracked())
        except RuntimeError as error:
            log.warning("could not snapshot untracked files: %s", error)
            return None

    def _new_paths(self) -> tuple[list[str], list[str]]:
        """Split the workspace's new files into the shippable ones and the rest.

        ``--exclude-standard`` drops everything the repository already ignores,
        and the loop's own output roots are dropped after it, because a
        campaign writes its archive and state into the workspace and neither
        is something the agent created. What is left is a file the agent
        created and the repository has no opinion about.

        ``commit_new_paths`` decides which of those a KEEP may carry; a
        protected path is never admitted however it is spelled there, because
        an allowlist that could name the driver would hand the agent the
        measurement surface.

        Raises ``RuntimeError`` when the enumeration itself fails. That is the
        right answer for the commit side, which must never build a KEEP out of
        a file set it could not read; the discard side goes through
        ``_new_paths_best_effort`` instead.
        """
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
        """``_new_paths`` for callers that are already recovering from a failure.

        The asymmetry with ``_new_paths`` is deliberate and is not to be
        collapsed. Refusing to build a KEEP from a file set that could not be
        enumerated is correct: the commit would otherwise ship an unknown tree.
        Refusing to run a discard is not: every caller of the discard path is
        cleaning up after something that already went wrong -- a patch that
        would not apply, a failed KEEP commit, resume recovery -- and a
        ``git ls-files`` failure raised from there replaces the failure being
        handled with a new one, in the resume case aborting the recovery
        outright. So it is logged, the new-file clean is skipped, and the
        ``git restore`` that is the handler's actual job still runs.
        ``knowledge/experience_integration`` treats worktree discard the same
        way.
        """
        try:
            listed = self._new_paths()
        except RuntimeError as error:
            log.warning("skipping the new-file clean: %s", error)
            print(f"  [git] could not enumerate new files, skipping the new-file clean: {error}")
            # Both callers return early from here without reporting, so a
            # refusal list left standing would be read as this iteration's.
            # Cleared and replaced by the reason it is empty, because the
            # Implementer reading silence as "nothing to report" is the same
            # leak the report exists to close.
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
        """Record and print the new files this ``action`` could not act on.

        ``refused`` are the ones no ``commit_new_paths`` entry admits;
        ``retained`` are allowlisted ones a discard deliberately left alone
        because they predate this loop. Reaching here at all means the
        enumeration succeeded, so it also clears
        ``_new_paths_unreadable``.
        """
        self._refused_new_paths = list(refused)
        self._retained_new_paths = list(retained)
        self._new_paths_unreadable = ""
        if refused:
            print(f"  [git] {len(refused)} new file(s) outside commit_new_paths, not {action}: " + ", ".join(refused))

    def _new_paths_need_discard(self) -> bool:
        """Whether new files alone make a discard necessary, refusals reported.

        A candidate that only created a file has no tracked diff, so the
        loop's ``attempt_diff`` test cannot see it. Refreshes
        ``_refused_new_paths`` on the way past, because that is otherwise only
        refreshed by a commit or a discard, and a stale list read as this
        iteration's would be the leak this whole path exists to close.

        Answers False when the enumeration fails: a discard that cannot be
        justified is not forced, and the tracked-diff test still decides.
        ``_new_paths_best_effort`` refreshes the report on that path too, so
        the promise above still holds.
        """
        listed = self._new_paths_best_effort()
        if listed is None:
            return False
        admitted, refused = listed
        self._report_refused_new_paths(refused, "removed")
        return bool(admitted)

    def _render_uncommittable_new_paths(self) -> str:
        """Tell the Implementer what the new-file report says this iteration.

        Three things can be worth saying, and an empty section is not one of
        them: files no allowlist entry admits, allowlisted files a discard
        left behind because they predate this loop, and the enumeration
        having failed outright. The last one matters most -- read as silence
        it says "nothing to report", which is the opposite of what it means.
        """
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
        """Stage ALL tracked modifications and commit, raising on failure.

        Uses ``git add -u`` (update tracked files) rather than adding only the
        kernel file: an agent commonly lands the winning change in a related
        tracked file the kernel imports (e.g. a ``*_config.py`` defaults module),
        not in ``kernel_file`` itself. If only the kernel file were staged, the
        keep/revert pattern would never revert those sibling edits — they would
        leak across iterations and the kept state would not match what was
        benchmarked. ``-u`` deliberately ignores untracked files (build
        artifacts) so they are never swept into the commit; the paths an
        operator allowlisted through ``commit_new_paths`` are staged by name
        instead, which is the one way a file the agent created can reach a
        commit. ``_git_discard_all_tracked_changes`` removes that same set
        minus anything already untracked when this loop took the workspace
        over, so a candidate that shipped a new file and was then reverted
        leaves none of ITS OWN behind -- and an allowlisted file that was
        there first, which a KEEP would also have staged, survives the REVERT
        and is reported instead of deleted.
        """
        before = self._git("rev-parse", "HEAD").strip()
        git("add", "-u", cwd=self.ic.workspace_dir)

        # Fail-fast here on purpose: a KEEP built from a file set that could
        # not be enumerated would ship an unknown tree. The discard side is
        # deliberately the other way round -- see ``_new_paths_best_effort``.
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
        """Discard staged and unstaged tracked edits in the workspace.

        The loop keeps HEAD at the last validated best state. A candidate stays
        in the working tree until it has passed validation and benchmarking; if
        the candidate fails or regresses, discarding the working tree returns the
        workspace to that last-known-good HEAD.
        """
        self._git_discard_all_tracked_changes()

    def _git_discard_all_tracked_changes(self) -> None:
        """Discard an exact pending candidate from both index and worktree.

        Allowlisted new files THIS iteration created go with it. ``_git_commit``
        can stage one, so leaving it on disk here would carry a rejected
        candidate's file into the next iteration's measurement -- the same leak
        that makes an uncommitted new file worse than an uncommittable one. The
        two sides read one allowlist, so whatever a KEEP can ship a REVERT can
        remove.

        Ownership is decided by ``_pre_untracked``, the untracked set
        captured before this loop did anything to the workspace and refreshed
        at the top of every iteration: an allowlisted file that was already on
        the tree then belongs to the operator or to an earlier round and is
        left alone. Resume recovery discards too, and runs before the first
        iteration, which is why the snapshot is taken in ``_run_impl`` and not
        only in the loop.

        With no snapshot at all the whole admitted set is cleaned and that
        is printed. That window used to include resume recovery and every
        caller before the first iteration, which is how an operator's file
        got deleted; taking the snapshot in ``_run_impl`` narrows it to a
        ``git ls-files`` that would not run even once, at which point nothing
        distinguishes the two owners.

        The restore is the job; the new-file clean is best-effort around it,
        since every caller here is already handling some other failure. See
        ``_new_paths_best_effort``.
        """
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
        """Read a source file's current content (best-effort).

        Resolves ``path`` relative to the workspace when it is not absolute.
        Returns "" if the file can't be read.
        """
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
        """Every declared source file's text, None per file that would not read.

        ``_read_source_file`` collapses an unreadable file to "", which a scope
        check reads as a source that assigns nothing: every pinned constant
        reported gone, every stored negative re-opened, on an I/O error. The
        distinction has to survive to ``scope_conflicts``, which says "not
        checked" rather than inventing a fact about the source.

        One unreadable file among several is the same fact about that file, so
        it travels as a ``None`` ENTRY rather than being dropped: dropping it
        would leave the survivors looking like the whole declared set, and a
        constant living in the missing file would be reported as one the task
        deleted. ``None`` for the whole list means nothing could be read.

        All of ``_target_source_files`` is read, not just the anchor: the
        implementer prompt tells the agent that tile, dispatch and JIT constants
        often live in a sibling file, so a constant that moved there is not a
        constant the task dropped.
        """
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
                # Printed, not only logged: a premise checked against part of
                # the declared source is a weaker check than the note reads as.
                print(f"  [lesson] source unreadable ({listed}): held-fixed premises checked against the rest only")
            return texts
        print(f"  [lesson] kernel source unreadable ({listed}): held-fixed premises not checked this round")
        return None

    def _target_source_files(self) -> list[str]:
        """Declared implementation hints, anchor first, de-duplicated.

        Repository tasks pass several files via ``source_files``; single-file
        tasks leave it empty and this collapses to ``[kernel_file]``. The list
        seeds profiling, JIT, and identity, and it is what the planning context
        publishes as ``editable_sources`` -- a FLOOR on the edit surface, never
        a ceiling. Everything named here may be edited (data and config files
        included); an edit inside one of them may still reach outside it, and
        new files may be added on top.
        """
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
        """Full unified diff of one iteration's commit (all files it touched).

        Unlike ``_diff_summary`` (a filtered ~8-line signal digest for the
        ledger), this is the complete patch, archived so a later iteration can
        reconstruct exactly what an attempt changed.
        """
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
        """Mechanical, loop-authored summary of one iteration's NET change.

        Ground-truth anchor for the experience ledger (cross-checks the agent's
        self-reported rationale). Returns a compact `--stat` header plus a few
        "signal" changed lines (tuning knobs / calls / signatures), not the raw
        multi-hunk diff.
        """
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
            # The first drop into degraded persistence is the one an operator can
            # still act on; the 12-hour run buried it at debug and the run looked
            # healthy until the final report. Repeats stay at debug to avoid a
            # flood once degraded.
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
        """Count the trailing run of empty diffs under the latest search mode.

        Derived from the append-only event log rather than a counter, so the
        streak cannot drift from the audit record and needs no schema field to
        survive a resume. An outcome that measured nothing -- an infrastructure
        failure or a crashed session -- is transparent here for the same reason
        an infrastructure decision is excluded from the stall streak: it says
        nothing about the direction under test.

        The run also ends where the search mode changes. The mode is the only
        durable direction identity the loop keeps: the recorded ``plan`` is the
        Implementer's own closing headline, which the prompt asks for in fresh
        prose every session, so two sessions handed one direction never word it
        alike and a streak keyed on it never reaches two. The mode is coarser --
        two different ideas pursued under EXPLOIT share a count -- but EXPLOIT is
        one objective, and repeatedly producing no edit under it is the fact this
        streak exists to report. Ending the run at a mode change is what keeps
        the escalation from consuming its own result: it moves the loop to
        DIVERSIFY, and that diversification then gets its own count instead of
        being ruled out on its first empty diff.

        An event written before the mode was recorded carries none, so it ends
        the run rather than being counted under a mode it cannot vouch for.
        """
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
        """Relative incumbent gain over the last full window of EXPLOIT outcomes.

        Read off the same append-only event log as the empty-diff streak, and
        for the same reasons: the incumbent score after each iteration is
        already recorded there, so the trend needs no counter of its own and
        survives a restart exactly as the audit record does.

        The scan ends where the search mode changes, so a window can only fill
        with outcomes from one uninterrupted run of exploitation -- a
        diversification round is itself the boundary, which is what stops the
        trigger from firing again on the evidence that already fired it. The
        mode of an outcome is read before anything else about it, including
        whether it concluded a verdict: a diversification round that failed
        outright is still a mode change, and skipping it would walk the scan
        back into the exploit outcomes that fired the trigger in the first
        place. The scan also ends at the last Supervisor intervention: the
        Supervisor injected a direction the loop has not measured yet, and gains
        recorded before it say nothing about that direction.

        No ratio is a named reason rather than a gap: a window that did not
        fill, a non-numeric score, a non-finite one and a non-positive one are
        four different facts about the campaign, and none of them is a gain of
        zero.
        """
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
        """Plan one round as lanes and run their sessions concurrently.

        ``lanes`` is the width the remaining budget admitted, which is at most
        the configured one and may be narrower.

        A planning outage or a lane-infrastructure failure -- the workspace
        copies, the room they need, the repositories they are given -- leaves the
        queue empty and the ordinary single-session path handles the iteration,
        including its own accounting. That is the loop's standing invariant: one
        iteration's failure must never kill a multi-hour run.

        What that path must not do is buy the round a second time, so this hands
        back what the round already holds: the published plan, the outage that
        stopped it, or None when it holds neither. Planning is dispatch plus
        every specialist plus synthesis -- the most expensive thing an iteration
        buys -- and a round that comes back empty has usually bought its lane
        sessions on top. An outage is handed back for the same reason: the
        backend that has just refused is the one the fallback would ask, so the
        retry buys a second timeout and records the same failure anyway.

        A round whose plans a previous process bought and never dispatched is
        recovered from disk instead of being planned again, which is the whole
        reason every lane's plan is published rather than only lane 1's.

        A programming error is not caught. Falling back on one would hide the bug
        behind a slower run that still looks like it is working.
        """
        self._last_lane_plans = []
        # Reset here rather than at the ordinary path's planning call, which the
        # round may now stand in for: a reused plan must report its own
        # executability and not the previous iteration's. A recovered round
        # leaves it unset and so reads as executable, which is not an assumption
        # but an invariant: a round whose synthesis failed publishes the single
        # framework plan, and a single plan is never recovered.
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
                # Republished under the iteration that actually runs them, so
                # this round is as recoverable as the one it inherited from and
                # an iteration's artifacts still describe what it did. A
                # workspace that cannot take a few KB of Markdown is in no state
                # to take a lane copy either, so this shares the fallback below
                # rather than ending the campaign on its own.
                plan_path = self._persist_lane_plans(
                    iteration,
                    self._last_lane_plans,
                    analysis_commit=self._canonical_commit(),
                )
                self._latest_optimization_plan_path = str(plan_path)
            # The round's plans exist and their cost is spent; what is priced
            # here is only the sessions and the measurement still to come.
            # Taken after the republish, not before it: a recovered round
            # refused now must leave its plans under THIS iteration, which is
            # the one the next process will find unfinished.
            if not self._admit_dispatch(iteration):
                # The plans stay on disk and this iteration records no result,
                # so the next session runs them instead of buying them again.
                return HeldRound(plan_path, "")
            await self._fill_lane_queue(
                iteration=iteration,
                agent_factory=agent_factory,
                lane_plans=self._last_lane_plans,
            )
        except (OSError, RuntimeError) as error:
            self._lane_queue = []
            print(f"  [lanes] fan-out unavailable ({error}); falling back")
        # A recovered round that could not even republish holds nothing, and
        # nothing was spent on it: that iteration plans for itself as usual.
        return None if plan_path is None else HeldRound(plan_path, "")

    async def _fill_lane_queue(self, *, iteration: int = 0, agent_factory, lane_plans) -> None:
        """Run this round's lane sessions concurrently and queue what they wrote.

        Sessions overlap because they are the expensive part; the driver runs
        inside them queue behind one cross-process lock because the device is
        single. Each lane is handed the serialized invocation of its own driver
        and told to use it twice over: the factory installs it as the command
        the session's own instructions name, and the plan repeats it.

        ``iteration`` stamps whatever contention the round's teardowns find, so
        the hazard it records names the round that found it.
        """
        kernel_relative = self._workspace_path(self.ic.kernel_file)

        async def _session(
            lane: LanePlan,
            lane_dir: Path,
            serialized_driver: Path,
        ) -> None:
            # The session edits the lane's own copy, so it is handed that copy's
            # kernel rather than the canonical one.
            await agent_factory(str(lane_dir), str(serialized_driver))(
                str(lane_dir / kernel_relative),
                _lane_prompt(lane.plan, serialized_driver=serialized_driver),
            )

        results = await run_lanes(
            workspace_dir=self.ic.workspace_dir,
            lanes=[LanePlan(lane_id=str(index + 1), plan=plan) for index, plan in enumerate(lane_plans)],
            session=_session,
            # Beside the workspace, not in /tmp: a lane copy carries the build
            # outputs and the whole experiment archive, and /tmp is typically a
            # smaller local filesystem than the one sized for the campaign.
            # Beside rather than inside, so a lane copy is never itself copied
            # by the next lane and never appears in the canonical git status.
            parent_dir=str(Path(self.ic.workspace_dir).resolve().parent),
            driver=self._workspace_path(self.ic.driver_script),
        )
        for result in results:
            if result.error:
                print(f"  [lane {result.lane_id}] session lost: {result.error}")
        self._lane_queue = [item for item in results if item.produced_candidate]
        print(f"  [lanes] {len(self._lane_queue)} of {len(results)} lanes produced a candidate")
        self._persist_lane_queue()
        # The device is not per-lane. A benchmark still running in one lane's
        # copy holds the same GPU a sibling's canonical measurement is about to
        # use, so one contended lane costs the ROUND its measurement rather than
        # costing itself its candidate -- and it says so even when that lane
        # also failed for a reason of its own, because the two are unrelated.
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
        """File an iteration that refused to measure on a device it does not own.

        Reported under the verdict a reader already knows for this --
        ``REVERT_CONTENDED`` -- because it is the same refusal the iteration
        that found the hazard made, and only the moment it is made differs.
        """
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
        """Why a lane candidate must not reach the canonical tree, or "".

        A lane session runs with the in-session gate off -- it benchmarks, and
        lanes are concurrent -- so no protected-path hook is installed and the
        lane's diff carries every tracked modification it made. The measurement
        surface is judged here instead, by the same rule the gate applies, and
        before the patch is written anywhere: a diff that must not land must
        never reach the canonical tree at all.
        """
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
        """Apply the next queued lane candidate to the canonical tree.

        A queued diff that no longer applies is dropped rather than retried: the
        tree it was written against has moved, and re-deriving it is the
        Implementer's job, not this one's. A diff that changes the measurement
        surface is dropped the same way, but before it is applied.

        What is left is republished on the way out, so a candidate this
        iteration has ruled on is never offered to a later process, and the ones
        it has not reached still are.
        """
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
                # The driver is the measurement boundary and must remain
                # byte-for-byte canonical. Recheck with the candidate applied so
                # a bypass this diff was not judged on cannot influence
                # correctness or KEEP.
                self._validate_driver_integrity(self.run_state)
            except ValueError as error:
                # Take the candidate back off the tree before anything else, so
                # the next candidate does not inherit it. If the driver is still
                # not canonical once the candidate is gone, the workspace itself
                # is tainted and no measurement may run at all.
                self._git_discard_worktree()
                self._validate_driver_integrity(self.run_state)
                print(f"  [lane {lane.lane_id}] candidate rejected: {error}")
                continue
            return lane
        return None

    def _git_apply_patch(self, patch: str) -> bool:
        """Apply one archived diff to the working tree, reporting whether it took.

        Checked before it is applied, so a diff that cannot land leaves nothing
        half-written behind for the next candidate to inherit.

        A diff the text no longer fits is retried as a three-way merge against
        the blobs it records. A hunk is located by the lines around it, so a
        KEEP on one lane's ground moves the context out from under a sibling
        that changed nothing it touched, and dropping that candidate throws
        away a finished Implementer session over a mismatch that is not a
        disagreement. The merge still refuses two edits to the same lines,
        which is the case the drop exists for.

        It is the fallback rather than the rule because ``--3way`` implies
        ``--index``: a worktree already carrying an applied patch does not
        match its index, and the stacking path applies two diffs in a row.
        ``--check`` cannot gate it either -- it reports success for a merge it
        would leave conflicted -- so the merge is its own test, and a failed
        one leaves markers behind for this to take off the tree.
        """
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
        """Two rejected gains worth measuring stacked, once single patches stall.

        A stacked attempt spends no Implementer session, only one correctness
        run and the usual benchmarks, so it is the cheapest thing to try when
        consecutive iterations stop producing a new best.

        The stall it answers to is ``unresolved_stall_iters`` rather than the
        supervisor's cooldown counter. ``no_improvement_iters`` is reset by an
        intervention as well as by a KEEP, and a supervisor memo changes what the
        next Implementer session is told -- not whether the archive holds two
        complementary gains that were never measured together. Reading the
        cooldown meant every intervention retired a stall this mechanism exists
        to answer: 37 of them across the thirty archived runs of 2026-08-22 and
        08-23, which is what reduced 121 qualifying iterations to 66.

        Pairs already measured are read back from the archive, which is where a
        measured stack leaves its record. Pairs that never reached a measurement
        leave no such record, so this process's own declines are carried
        alongside it -- see ``_declined_merge_pairs`` for why that set is not
        durable.
        """
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

    # The one obstacle a later iteration clears on its own, and so the one the
    # caller must not hold against the pair: a tree carrying work is this
    # iteration's accident, not a fact about two archived diffs.
    TREE_ALREADY_DIRTY_OBSTACLE = "the working tree already carried uncommitted work"

    def _merge_attempt_refusal(self) -> str:
        """Why a selected pair may not be measured this iteration, or "".

        Ruled on before the pair is staged, so a refusal leaves the tree
        untouched and leaves the pair selectable. It is not a verdict on the
        pair: what it rules on is the iteration.

        A stacked iteration is the one iteration that neither drains the queue
        nor buys a round, so running one brings the loop no closer to the
        queue-empty branch where the next round is priced. It does not end the
        stall that admitted it either -- the attempt reverts, which raises
        ``unresolved_stall_iters`` -- so nothing about having run one makes the
        next one less likely, and a streak runs until the pairs give out. The
        pool cannot grow while it does: the only candidate a stacked iteration
        archives is the stack, which
        :func:`~kernelforge.loop.merge_candidates.eligible_candidates` skips.
        So a streak is finite on its own, at the size of a frozen pool's pair
        set -- which goes as the square of the pool. What the archives measure
        of that is in :data:`MERGE_PRECEDENCE_STREAK_LIMIT`.

        The reachability this restores does not rest on the constant's value.
        The queue is refilled only by a round, a round is opened only on an
        iteration that has already priced one, and an iteration that stacks
        nothing takes a candidate off the queue unless the device is held --
        which is separately terminal once nothing clears it. So for any finite
        limit the queue empties, and ``_admit_next_round`` is reached, within
        (limit + 1) x its depth iterations.
        """
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
        """Put both candidates' diffs in the tree; the diff, or why there is none.

        Two patches that clash textually say nothing about whether their gains
        compose, so the tree is returned to canonical and the iteration falls
        back to an ordinary Implementer session. The obstacle is returned rather
        than swallowed: a selected pair that never reaches a measurement is the
        failure mode that hid this whole mechanism for two months, and it is only
        distinguishable from "no pair was selected" if the caller can say so.

        Returning to canonical discards every tracked edit, not only the ones
        applied here, so a tree that already carries work is declined outright
        rather than staged into. The loop reaches this point on a clean tree in
        its own steady state, but that is an invariant of the paths that run
        before it -- and a stacking attempt is not the thing that should be
        enforcing it by deleting the counter-example.
        """
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
                # Reported apart from a conflict because the two ask for
                # opposite responses. A conflict is a fact about these two
                # candidates and says the archive is working; an entry the
                # archive cannot produce says the archive lost a candidate it
                # claims to hold, which every other reader of it -- the
                # retrieval map, a resumed run -- is also relying on.
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
        """Report a pair that was selected and not measured, and drop it or not.

        A pair the archive offered and the loop could not measure is not the
        same event as no pair at all, and counting the two together is how a
        mechanism runs zero times without anyone noticing. So every decline is
        reported, whatever it was that stopped the pair.

        Whether the pair is also *dropped* is a different question, and it is
        the one this argument settles. A staging obstacle is a verdict on two
        archived diffs, and archived diffs do not change: the selector returns
        the pair covering the most cases, so a failure it does not remember
        wins the selection again at the next stall, fails the same way, and
        goes on blocking the runner-up that would have staged. Two obstacles
        are not verdicts on the pair at all, and dropping either one costs a
        measurement that nothing was ever wrong with:

        * a tree that already carried work, which is this iteration's accident
          and is gone by the next one -- named in
          :data:`TREE_ALREADY_DIRTY_OBSTACLE` because it is the one such case
          :meth:`_stage_merge_attempt` can return; and
        * a refusal from :meth:`_merge_attempt_refusal`, which is passed here
          as ``about_the_iteration``. It rules the iteration out before the
          pair is staged, so nothing about the pair has been tested -- the
          diffs were never read, let alone applied to each other. Remembering
          it would turn a deferral into a drop, and turn the streak limit into
          the cap on firings :data:`MERGE_PRECEDENCE_STREAK_LIMIT` says it is
          not.
        """
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
        """Pin a rejected candidate that still measured faster than the incumbent.

        A REVERT_PERF covers two different outcomes: a regression, and a real
        gain that landed under the KEEP threshold. The second is the most
        promising work the run has, and the long-horizon prompt ships a
        retrieval map rather than the candidate diffs, so an iteration nothing
        pins is one the Implementer has no reason to open.

        A regression is simply not pinned. It is deliberately not recorded as a
        spent direction either: a failed candidate does not make its direction a
        permanent search constraint, and the trajectory already carries what
        happened as fact.

        "Faster" is asked twice, because the equal-weight mean answers it only
        for the suite as a whole. A candidate that won 2% on one of the two
        cases carrying a campaign's entire deficit, and lost the mean to a third
        case, left no trace at all under the aggregate test -- yet it is the only
        measured step in the direction the run needs. So a candidate that beat
        the *incumbent's* per-case time on any scored case by more than that
        case's own spread across its measurements is pinned too. This changes
        nothing about the KEEP gate: a pin is a record and a merge input, and
        the incumbent is still whatever cleared the bar on the mean.

        Both per-case arguments are optional; without them this is the aggregate
        test alone, which is what a candidate replayed from a journal that
        carries no per-measurement detail gets.
        """
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
            # A non-KEEP leaves the incumbent untouched, so the recorded
            # post-decision score is the bar this candidate had to clear.
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
        # A resumed session recomputes session_index and experiment_id, which
        # legitimately differ from what the stored manifest was written with, so
        # republishing an already-current best tripped the same-iteration
        # conflict guard and reported persistence_degraded over a bundle that was
        # already correct. Skip the republish exactly when the fresh path would.
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
        """Bring the campaign's cumulative wall-clock up to now, and return it.

        The CAMPAIGN's span, not this process's: a resumed session starts with
        what earlier sessions spent already behind it. It is recorded into the
        run state instead of being computed wherever it is needed, because the
        totals it is reported against are themselves campaign-cumulative and
        outlive this process. Dividing them by this process's clock is what
        printed ``450% of the run`` for a session that ran 10 minutes against
        45 cumulative minutes of planning.

        An assignment from one monotonic origin rather than an increment, so
        calling it once or twenty times in an iteration says the same thing,
        and a clock already ahead of that origin -- one restored from a state
        file, or raised to cover the planning charged to it -- is never pulled
        back.

        Before ``run()`` anchors the origin there is no campaign span to read,
        and subtracting from an unset origin would return the age of the Unix
        epoch. What is already recorded is returned unchanged instead.
        """
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
        """Record what the open round cost, if it bought any planning.

        Closed lazily at the start of the next round rather than in a ``finally``
        around the iteration body: that body leaves at several points, and every
        one of them is followed either by the next iteration or by the end of
        the run, where this is called once more.

        A round that spent nothing on planning -- one draining a lane candidate
        a previous round already paid for, one stacking two rejected gains, one
        replaying plans recovered from disk -- records nothing. Its zero is not
        a measurement of what planning costs, and averaging it in would tell the
        next round that planning is free.

        What the round spent inside the canonical validation and benchmark is
        recorded with it, and is what the NEXT round's dispatch is priced from.
        A round that never reached the measurement records a zero there, which
        is read as no observation rather than as a free cycle.
        """
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
        """What the campaign's rounds have cost, for the published report.

        Empty until there is something to report, so a campaign whose rounds
        never planned -- every acceptance path that runs no orchestration --
        publishes exactly the manifest it always did.

        Every duration here is campaign-cumulative: it spans every session the
        campaign has run, not the one writing the report. The planning share is
        computed here too, rather than left for the report to divide, because
        here is where both of its halves are in hand and known to describe the
        same span. A reader downstream holding only this dict then has no clock
        of its own to reach for.
        """
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
        """Charge the open round for one canonical validate-and-benchmark cycle.

        Its cost is what the next round's dispatch is priced from, so it is
        taken from the clock rather than from the per-step timeout ceilings the
        first version of this guard used: across 171 cycles of ten production
        campaigns (2026-08-17) the cycle cost 36 seconds at the median and 150
        at its worst, against ceilings summing to 35 minutes.

        A cycle run while no round is open -- an iteration draining a lane
        candidate an earlier round already bought -- belongs to no round and is
        not recorded. Rounds are what the guard prices, and every round runs
        one of these itself.
        """
        if self._round_started_at is None:
            return
        self._round_measurement_sec += max(0.0, time.time() - started_at)

    def _measurement_estimate_sec(self) -> float:
        """Wall-clock the canonical validation and benchmark may still take.

        A round is not finished when its session returns: the candidate has yet
        to face the same correctness suite and benchmark every other candidate
        faced, and a round killed before that has produced nothing measurable.
        """
        return estimate_measurement_sec(list(self.run_state.round_costs.recent))

    def _admit_next_round(self, iteration: int) -> int | None:
        """How many lanes the next round may PLAN, or ``None`` if none fit.

        The cheap half of the decision, and a lower bound: it refuses only a
        round that could not run even if planning were as fast as any campaign
        has ever seen it, so that such a round does not buy planning first.
        Whether the round may then be dispatched is decided by
        :meth:`_admit_dispatch` once planning has returned.

        A narrowed round is announced, because the campaign is no longer
        searching as widely as it was asked to; a refusal is announced loudly
        and remembered, because it ends the campaign and must not be mistaken
        for a round that simply found nothing.
        """
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
        """Whether the round now holding plans may start its session.

        The decisive check, and the reason it is taken here: planning is the
        dominant term of a round and the most variable one, so before it runs
        the remaining budget says almost nothing about whether the round will
        finish. Replayed over 82 production rounds (2026-08-17), the two rounds
        killed by the external timeout entered planning with 30 and 32 minutes
        left -- among rounds that finished with 33, 42, 49 and 50 -- and came
        out of it with 7.3 and 8.3, against a worst survivor at 24.8.

        Unlike the check before planning, this one has a floor no observation
        lowers. The loop cannot interrupt the session it is about to start and
        does not size that session from what remains, so what this check really
        guards is the external timeout -- which does not move because this
        campaign's own validation got faster.

        What planning cost is spent whether or not the round runs, so a refusal
        here keeps the plans rather than discarding them: this iteration
        records no result, which is what marks it unfinished, and the next
        session republishes its lane plans instead of buying them again. That
        holds for a fan-out round, whose plans are all published before
        dispatch. A round narrowed to a single lane publishes one plan, which
        is deliberately never recovered -- a single plan is also what a round
        whose synthesis failed leaves behind, and the two are indistinguishable
        on disk -- so that round's planning is lost. It is the cheapest round
        there is, and the alternative is resuming a plan that may describe a
        partition that never happened.
        """
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
                    # Recorded because it is the one case where the parts do not
                    # add up to the requirement: this campaign estimated less than
                    # the external-timeout floor and was held at it.
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
        """End the campaign on a round the remaining budget cannot pay for.

        Remembered as well as printed: the run summary and the published report
        both say a round was priced out, because from the outside that is
        indistinguishable from a campaign that ran out of ideas and the two
        call for opposite responses.
        """
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
        """Bench the pristine kernel before any agent edit — the speedup anchor.

        Runs the build (if configured) and the driver's full benchmark suite.
        Three independent measurements establish per-case medians for the same
        scoring protocol used by every candidate.
        """
        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *smart_wrap(list(self.ic.build_command)),
                cwd=self.ic.build_dir or self.ic.workspace_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            _, stderr = await communicate_process_group(
                proc,
                timeout=self.ic.build_timeout_sec,
            )
            if proc.returncode != 0:
                print(f"  Baseline build FAILED: {stderr.decode()[-300:]}")
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
        """Record (once) the pristine per-case wall times and persist them.

        Idempotent: only the FIRST non-empty capture sticks, so a later
        iteration's ``case_times`` can never redefine the speedup denominators.
        """
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
        """Checkpoint the state that decides keep/revert.

        Without this a resumed session restarts from a weaker standard than the
        one that produced the incumbent, and compares candidates against it.
        Best-effort: losing the checkpoint must not abort a running campaign.
        """
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
        """The incumbent's per-case times, restricted to the scored cases.

        The per-case near-miss test is against what a candidate would have to
        replace, not against pristine, so this is ``_best_case_times`` and not
        ``_baseline_case_times``. Unscored cases are dropped: a case the driver
        excluded from the mean cannot earn a pin either.
        """
        scored = set(self._scored_case_ids())
        return {
            case_id: float(time_ms)
            for case_id, time_ms in (self._best_case_times or {}).items()
            if case_id in scored and isinstance(time_ms, (int, float)) and float(time_ms) > 0.0
        }

    def _scored_baseline_case_times(self, bench_result: dict) -> dict[str, float]:
        """The pristine per-case times the objective's mean is actually taken over.

        Cases the driver marked unscored in any measurement are dropped, the
        same exclusion ``calculate_mean_case_speedup`` applies when it builds
        the scores -- attributing sigma over a different set of cases than the
        objective is defined on would blame the bar on a case that is not in it.
        """
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
        """Estimate the objective's sigma from the case that supplies it.

        The KEEP rule is not touched here and neither is the objective. What is
        re-estimated is its *input*: three aggregate scores are a poor estimator
        of the objective's spread when one cheap case supplies the majority of
        it, because the sample standard deviation of three samples scatters by
        50% of itself, and that scatter is then multiplied by the KEEP rule's
        t critical value and charged to every candidate. On the 2026-08
        GQA campaign that produced a bar ranging from 0.32% to 8.42% of the
        incumbent, and two candidates with the same 0.92% gain were decided
        opposite ways by which side of that range they drew.

        When :func:`~kernelforge.loop.scoring.attribute_sigma` names a
        dominant case, more measurements are bought and every scored case's
        spread is re-estimated from the larger sample -- the dominant case is
        why the measurements are worth buying, but once bought they are data
        about all of them. The extra measurements never become scores: the
        candidate still stands or falls on the ``KEEP_MEASUREMENT_COUNT`` scores
        the protocol took.

        The loop stops when the dominant case's variance share falls back under
        the threshold, or at the bound. It is written not to depend on the first
        happening: a share is structural -- a case whose speedup term is eight
        times the others' holds most of the objective's variance at any sample
        size -- so the usual exit is the bound, and the win is the estimate, not
        the share. When the share does fall the three-measurement draw was
        simply an unlucky one, which is the case worth stopping early for.

        ``measure_wallclock`` runs the driver, and the driver runs its whole
        suite; neither it nor ``bench_wallclock`` takes a case selector, and no
        per-task convention for one exists in ``driver_args``. So a re-measure
        is a whole-suite bench, and its cost -- not the dominant case's cost --
        is what
        :data:`~kernelforge.loop.scoring.SIGMA_REMEASURE_MAX_ROUNDS` bounds.
        For the same reason it is bought only for a candidate whose verdict
        sigma can still decide, which is a band and not a floor. Below it, a
        weakest score that does not beat the incumbent is a REVERT at every
        sigma. Above it, a candidate already clearing the bar is a KEEP at the
        sigma the protocol measured, and re-estimating can only take that away
        -- which is not sigma deciding the verdict, it is sigma being drawn a
        second time, and only for candidates whose noise happens to come from a
        cheap case. Replaying 1240 archived candidates across 19 kernels, the
        floor-only form charged 28% of them for the estimate while just 6% could
        gain from it; the band charges those 6% and keeps the whole benefit.
        """
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
        # A case's *share* of the variance is structural -- q61's speedup term is
        # 8x the others', so it holds most of the variance at any sample size --
        # and re-measuring cannot be expected to move it. What re-measuring moves
        # is the estimate. So dominance alone cannot be the report, or every
        # candidate on the motivating campaign carries it; the second clause is
        # that the larger sample did not lower the case's spread either. That
        # clause discriminates poorly -- it fires on 56% of a stationary noise
        # model, see :class:`SigmaResolution` -- so what it produces is a
        # diagnostic hint on the bench line and nothing else. No verdict, bar or
        # bought measurement depends on it.
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

    def _scored_case_ids(self) -> list[str]:
        """The cases the mean this campaign is scored on is taken over."""
        return [case_id for case_id in sorted(self._baseline_case_times) if case_id not in self._unscored_cases]

    def _case_move_rule(
        self,
        before: float,
        measured: float,
        per_run: tuple[float, ...] | list[float],
    ) -> str | None:
        """Which rule, if any, admits one KEEP's move on one case as real.

        Three conditions, in the order they are checked:

        * the aggregate move clears ``config_coverage_min_move_ratio``, the
          floor under everything below;
        * every independent measurement of the KEEP timed the case faster than
          it was before, so the improvement is not one run carrying two;
        * the move is at least ``CONFIG_COVERAGE_DISPERSION_MULTIPLE`` times
          the spread those measurements show, so a case whose runs disagree by
          more than the move is not called covered.

        Returns ``"dispersion"`` when all three held, ``"floor"`` when there
        were fewer than two per-measurement times so only the first could be
        tested, and ``None`` when the case is not covered. The two verdicts
        are kept apart rather than collapsed to a bool because they establish
        different things, and the ledger has to say which one it is: a
        warm-started result or one rebuilt from published metadata carries no
        ``measurements``, and those are exactly the records a planner should
        read with the most caution.
        """
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
        """Read per-case configuration coverage off this session's KEEPs.

        A KEEP ships one configuration of the canonical. The cases whose
        measured time it improved are the cases that configuration was chosen
        for; a scored case that no KEEP has ever improved is being served by
        whatever generic path the source falls through to, and the suite mean
        cannot say so -- it averages that case in at 1.00x alongside the ones
        the campaign actually tuned. A case a KEEP made slower is not one that
        KEEP was tuned for, so a regression leaves it where it was.

        "Improved" is decided against the case's own measured noise, not
        against the suite-mean KEEP gate: see ``_case_move_rule``, whose
        verdict is carried through to ``floor_only`` so a case admitted
        without any dispersion to test is not reported as though there had
        been one.

        Read from ``self.results``, which is this session's record: a resumed
        campaign starts the ledger again, and every reader of it is told so
        rather than shown an empty ledger that reads like an untuned suite.
        """
        scored = self._scored_case_ids()
        previous = dict(self._baseline_case_times)
        moved_by: dict[str, list[int]] = {case_id: [] for case_id in scored}
        # Covered cases whose every admitting KEEP was admitted by the floor
        # alone. Tracked per case rather than per KEEP: one KEEP with the
        # per-measurement detail is enough to establish the strong statement.
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
                # A KEEP replayed from a pending record can arrive without its
                # per-case timings. Nothing can be read off it, and dropping it
                # here would make it indistinguishable from a KEEP that never
                # happened, so it is carried out and reported instead.
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
            # Before the first readable KEEP there is nothing to read coverage
            # off. Calling every case a fallback here would put the campaign's
            # starting state and a case a whole session failed to reach on
            # the same line.
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
            # Covered, but on the floor ratio alone. Flagged separately so the
            # planner is not told a run-to-run spread was tested when the
            # record it was read off carried none.
            flags.setdefault(case_id, []).append("config_coverage_floor_only")
        for case_id in coverage.fallback:
            flags.setdefault(case_id, []).append("config_coverage_fallback")
        for case_id in coverage.unmeasured:
            flags.setdefault(case_id, []).append("config_coverage_unmeasured")
        for group in coverage.undifferentiated:
            for case_id in group:
                flags.setdefault(case_id, []).append("config_coverage_undifferentiated")
        if coverage.unreadable:
            # Every other flag on this ledger was read off a partial record, so
            # the planner is told which cases were classified without it rather
            # than being handed the classification alone.
            for case_id in self._scored_case_ids():
                flags.setdefault(case_id, []).append("config_coverage_partial_record")
        return {case_id: tuple(dict.fromkeys(values)) for case_id, values in flags.items()}

    def _with_case_config_coverage(self, context):
        """Attach measured configuration coverage to a planning context.

        The planner already reads per-case flags; coverage travels the same
        way rather than as a second per-case channel it would have to be
        taught to read.
        """
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
                # Without this the resumed session re-derives its own noise
                # floor, incumbents and SNR reference, so it judges candidates
                # by different rules than the session it continues.
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
        """Validate the CLI's kill-recoverable warm-start publication.

        The CLI publishes iteration 0 before the loop starts so an external kill
        cannot lose a validated KB seed. The runner must adopt that publication,
        not write the same immutable ``best/iter_000`` bundle again under its
        newly-created campaign/session identity.
        """
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

            # The monitor remains operational even if durable state I/O fails,
            # but an idempotent recovery must not count the outcome twice. A
            # decision that reached no verdict is withheld from it for the same
            # reason it is withheld from the stall streak: the monitor escalates
            # to the supervisor on a run of no-improvements, and a gateway
            # outage is not something the supervisor can redirect.
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
            # The mode this iteration actually ran under, which is the direction
            # identity the empty-diff streak is counted against. Recorded here
            # because ``plan`` cannot serve: it is the model's own closing
            # headline, reworded every session.
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
                    # The planner reads handoffs, not lesson documents, and a
                    # refutation quoted out of a handoff is how one sweep at one
                    # M became a ban on every case. The scope rides along with
                    # the plan it qualifies.
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
        """The cases the suite actually scores, sorted.

        ``_baseline_case_times`` holds every baseline case, including the ones
        scoring excluded as too noisy to move a decision. The driver still times
        an excluded case — it is measured — but no decision is based on it, so
        it belongs neither in a recorded scope nor in the set a stored scope is
        re-validated against.
        """
        return sorted(case_id for case_id in self._baseline_case_times if case_id not in self._unscored_cases)

    def _loop_measured_a_negative(self, decision: str, result: IterationResult, session_sink: dict) -> bool:
        """Whether the LOOP itself saw something come out worse this iteration.

        Decided from a whitelist rather than by pattern-matching the label:
        only KEEP and the labels that mean no candidate was ever measured can
        possibly mean nothing came out worse. A crash and a build failure are
        neither a REVERT prefix nor a recorded speedup, so a rule built on
        those two reads them as clean iterations.

        ``findings`` is the in-session gate's rejection log. Most entries are
        measurement rejections the agent hit and retried ("correct but not
        faster"), but the same log carries policy denials — a denied edit to a
        protected measurement file, a denied Bash call — which are not results.
        A non-empty log is therefore read as "possibly a negative", which is
        the conservative direction: it re-opens a document rather than closing
        one.

        This sees only the loop's own view. A direction the session tried and
        reverted before submitting its candidate is invisible here, which is
        why the summarizer's ``NEGATIVES:`` marker exists.
        """
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
        """Whether this DOCUMENT records anything that measured worse.

        The loop's verdict wins where it says yes: machine truth beats prose.
        Where it says no, the document is mostly not about the loop's one
        candidate — the record is explicitly asked for every direction tried,
        including the four reverted inside the session — so the answer comes
        from the summarizer's ``NEGATIVES:`` marker, the only party that saw
        them. No marker means nobody answered: ``None``, treated as
        conservatively as a recorded negative, never rendered as "no negative".

        A machine-written fallback document is the exception: the loop authored
        it, so it contains exactly what the loop saw and nothing else, and the
        loop's own verdict is the whole truth about it. ``agent_narrative`` is
        what tells the two apart — it is true only when the resumed session
        itself wrote the record, which is also the only case where a marker
        could have been written.
        """
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
        """The conditions this iteration's observations were taken under.

        The cases are the scored suite — the baseline cases minus the ones
        scoring excluded as too noisy. The driver still times an excluded case,
        so it was measured; it just was not scored, and no decision was based
        on it. The suite is narrowed further to the subset the round named when
        it named one: a lane restricted to some cases measured only those, and
        that restriction is exactly what must not be dropped when the lane's
        negative results are read later.

        The held-fixed constants and whether anything measured worse both come
        from the session's own record, because the loop can see neither: it
        knows what a sweep pinned as little as it knows which of five tried
        directions were reverted before the one it measured. It overrides the
        record only where its own verdict is a negative. Recording the answer
        lets a later iteration tell an unrecorded premise that matters from one
        that does not: a document with nothing negative in it closed nothing,
        and re-opening it means nothing.

        The disproof answer comes from the record alone, for the same reason:
        only the session knows whether it concluded that something could not be
        reached, what — if anything — it ran against that conclusion, and which
        way that came out. No marker means the question went unanswered, which
        is not the answer "it claimed nothing"; a marker reporting the claim
        DISPROVED is not an obligation discharged but the axis shown open, and
        it is recorded as such so the next iteration re-enters it.

        Note that a KEEP is not the same as "nothing measured worse" — a
        session can reach a kept candidate through four measured regressions —
        which is exactly why the question is asked of the record and not
        inferred from the verdict.
        """
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
        """Write this iteration's free-form factual session record.

        The resumed Implementer records the actions and observations available
        only in its conversation, then the loop appends the measured verdict.
        Neither the ledger nor the candidate archive distills this free-form text
        into a behavioral instruction.

        When no summarizer can run, the narrative half is machine-written from
        the gate's block reasons, net diff, or provider progress. Every started
        agent session receives at least an objective outcome document, even if
        it produced no candidate and no fallback signal. Best-effort throughout.
        """
        store = getattr(self, "lessons", None)
        if store is None or session_sink.get("session_started") is not True:
            # No agent ran this iteration (e.g. the baseline measurement path):
            # there is no exploration to record.
            return

        has_narrative = False
        # Narrated by the resumed session itself, as opposed to machine-written
        # by the loop below. Only the first kind can carry a NEGATIVES: marker,
        # and only the second is fully visible to the loop's own verdict.
        agent_narrative = False
        summary_failure = ""
        if self._time_remaining() < SUMMARY_MIN_SECONDS:
            # Gate on whether there is time to PRODUCE the summary, not on the
            # loop's session-admission reserve: a campaign that runs out of room
            # for another implementer session is resumed later, and that session
            # reads this document. Skipping here would silently drop the record
            # of the last iteration of every session — the handoff point where
            # it matters most.
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
            # No session could describe what was explored, but the gate's block
            # reasons are a real record of what the agent ran into. Without this
            # the next iteration would inherit only a verdict.
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
        """Execute a single build→validate→bench→canonical→decide iteration.

        ``plan`` is the agent's one-sentence description of the modification it
        made this iteration; persisted onto the iteration record so downstream
        summaries (e.g. the forge run canvas) can show what was tried each round.
        """
        iter_start = time.time()
        force_jit_rebuild(self._jit_source_files())

        # Step 1: Build (if configured) — RTK-wrap so a build failure's tail
        # chars are signal, not boilerplate (ninja/cmake collapse 80%+).
        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *smart_wrap(list(self.ic.build_command)),
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
                    validation_summary=f"BUILD FAILED: {stderr.decode()[-500:]}",
                    kept=False,
                )

        # Step 2: driver-owned full correctness suite. The round is charged for
        # this and the benchmark below as one canonical measurement, because
        # that is the unit the next round's dispatch is priced against.
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

        # Step 3: Benchmark (only if validation passed). A converged in-session
        # gate can hand off the exact framework-owned measurement for this diff.
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
        # A candidate whose bench crashed is reverted for "no speedup" unless the
        # crash itself reaches the agent; the tool's output tail is the only place
        # the traceback exists.
        bench_error_output = (
            "" if bench_result.get("success") else "BENCH FAILED: " + _bench_failure_detail(bench_result)
        )

        # Collapse per-case times into an equal-weight mean of per-case speedups,
        # rather than allowing expensive cases to dominate an aggregate ratio.
        # Once baseline case data exists, incomplete candidate coverage fails closed.
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
        # The bar is a t multiple of the standard error of the mean, so a
        # REVERT is only readable next to the spread that set it and, when one
        # case supplied that spread, next to the case: a weak candidate and a
        # noisy 10 us dispatch print the same mean score.
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
            # The scoring verdict above has already overwritten ``message`` with
            # "candidate emitted no per-case timings", which describes the symptom
            # of a crash as if it were a formatting choice.
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

        # Step 6: the mean of the independent pristine-relative scores must
        # clear the current best by the candidate's own measurement noise.
        improved = bool(bench_result.get("success")) and passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=(self.best_mean_case_speedup or 1.0),
            sigma=sigma,
            sigma_sample_size=sigma_resolution.sample_size,
        )

        # Step 7: the arena's own verdict. SNR got this candidate here; only the
        # task's declared suite can accept it. Run it only for a candidate that
        # would otherwise be kept -- it is the expensive check, and a candidate
        # that is not faster is reverted whatever it says.
        if improved:
            canonical_started = time.time()
            canonical = await accept_candidate(
                self.ic.workspace_dir,
                timeout_cap_sec=self.ic.validate_stage_timeout_sec,
                candidate_label=f"iteration {iteration}",
            )
            # The suite only runs for a candidate the round produced, so it is
            # part of that round's measurement and has to be priced into the
            # next round's admission alongside the validate-and-bench cycle.
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

        # Auto-evolve: log benchmark to tuning DB
        if selected_raw_mean_ms is not None:
            try:
                self.evolver.on_benchmark(
                    operation=Path(self.ic.kernel_file).stem,
                    backend=self.experiment.backend if self.experiment else "unknown",
                    shape={},
                    config={"iteration": iteration, "kept": improved},
                    wall_ms=selected_raw_mean_ms,
                    snr_db=snr_db,
                    passed_correctness=snr_db is not None and snr_db >= self.ic.snr_threshold,
                    pmc_diagnosis=pmc_diagnosis,
                    vgpr=vgpr,
                    experiment_id=self.experiment.experiment_id if self.experiment else "",
                    gpu_target=self.config.gpu_target,
                )
            except Exception:
                log.debug("auto-evolve on_benchmark logging failed", exc_info=True)

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
                    # ``make_event`` drops empty fields, so a ratio of None on
                    # its own would leave the event silent about a trigger that
                    # could not be evaluated at all -- indistinguishable from a
                    # young campaign. Exactly one of these two is always written.
                    window_gain_ratio=window_gain.ratio,
                    window_gain_unavailable=window_gain.unavailable,
                    mode_changed=(decision.mode != previous_mode),
                )
            )
            self.state_store.save(self.run_state)
        except Exception:  # noqa: BLE001 - policy remains available in memory
            log.debug("search policy persistence failed", exc_info=True)
        # A window that has not filled yet is the ordinary state of a young
        # campaign. The other reasons mean the score series itself is unusable,
        # which disables the trigger for as long as it lasts, so the operator is
        # told once per distinct fault instead of only the event log knowing.
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
        """Buy the round's plans and charge the round for the wall-clock.

        Timed here, where planning is bought, rather than inside the call it
        wraps -- and around the whole purchase, outage included: an outage still
        spends the specialists' wall-clock, and the next round has to be priced
        against what planning actually costs, not what it costs when it works.
        """
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
        """Run planning and durably publish every lane's plan for the round.

        Returns the path of lane 1's plan, which is the one an ordinary session
        is handed; the rest are published beside it for audit and recovery.
        """
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
        """Turn a probe the analysis phase could not clear into a live hazard.

        The device is not the analysis phase's any more than it is one lane's. A
        specialist killed by its session timeout mid-probe leaves a benchmark on
        the same GPU this round's canonical measurement is about to use, so it
        costs the ROUND its measurement exactly as a contended lane does -- and
        it is recorded here, through the loop's own hazard log, so everything
        downstream reads it from the one place that already refuses on it.

        Carried in the planning diagnostics rather than through
        ``device_hazard.json``: the log is loaded once per process, so a hazard
        another layer wrote to disk would not be seen by this instance, and the
        measurement it has to stop is in this very iteration.
        """
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
        """Put this round's verdict where the next process can still find it.

        Only the verdict and the path travel: the review is already published
        beside the round's plans, and this file is a control checkpoint rather
        than somewhere to inline an artifact of unbounded length.

        A fail-open review records nothing. Its artifact holds the error that
        stopped it rather than a review, so a pointer to it would restore an
        outage as though it were a judgement.
        """
        ruling = CriticRuling()
        if critic is not None and not critic.error:
            ruling = CriticRuling(
                verdict=critic.verdict,
                review_path=str((self._orchestration_root(iteration) / "critic_review.md").resolve()),
            )
        self.run_state.last_critic = ruling

    def _restore_critic_ruling(self) -> None:
        """Resume the ruling whose round ended with the process that bought it.

        Read before the first iteration, because the verdict decides how that
        iteration is divided: a REPLACE spends one of its lanes challenging the
        route, and a round already dealt out cannot be asked again.

        A review that is no longer readable leaves the verdict unused. The
        challenge is stated in the review, so a verdict without one would ask
        the round for an alternative it was never told.
        """
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
        """Where one lane's plan lives, lane 1 keeping the historical name.

        Lane 1 is the plan the Implementer is pointed at on the single-session
        path, which predates lanes and is read by name from the archive, the
        handoff documents and the supervisor's evidence. It keeps that name at
        any width so none of them has to know how wide the round was.
        """
        if lane <= 1:
            return self._orchestration_root(iteration) / "optimization_plan.md"
        return self._orchestration_root(iteration) / f"lane_{lane:03d}.md"

    def _lane_queue_path(self) -> Path:
        """Where a round's unspent candidates wait for the iteration that measures them.

        One file for the campaign rather than one per round: the loop refuses to
        fan out while anything is queued, so there is only ever one live queue,
        and a per-round name would leave a reader guessing which round the
        candidates in hand belong to.
        """
        return Path(self.ic.workspace_dir).resolve() / "forge_experiments" / "orchestration" / "lane_queue.json"

    def _persist_lane_queue(self) -> None:
        """Publish what this round has bought and not yet measured.

        A fan-out round pays for one Implementer session per lane and then
        spends the candidates one per iteration, so a process that ends with any
        of them unspent throws finished sessions away -- and a budget that runs
        out mid-round is the ordinary way for a campaign to end, not only a
        crash. By then the lane workspaces are deleted and the diffs live
        nowhere but this process's memory, which is what this file answers.

        The plans are published for the opposite reason: they are cheap enough
        to buy again and are kept so a round need not be re-planned. A finished
        session cannot be bought again at all.

        A queue that cannot be written costs this round its durability and
        nothing else. The candidates are still in memory and this process still
        measures them, so the run continues -- and says what it stands to lose.
        """
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
        """Pick up candidates a previous process bought and never measured.

        Restored rather than re-derived: the sessions that wrote them have
        already run and their lane workspaces are already gone. Restored
        regardless of this run's ``--lanes`` too, because what is queued was
        paid for at the width the round was planned at, and narrowing the next
        round is not a decision to discard the last one's work.

        Whether each diff still applies to the tree is not asked here. That is
        exactly what taking a candidate already decides, one candidate at a
        time; asking it here would refuse a whole round over one stale diff.
        """
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
        """The record that says a round's plans are complete and what they are.

        Written last and removed first, so its presence is the commit point of
        a round's publication: a process that died partway through leaves plan
        files without it, and those are read as nothing rather than as a round
        that can be picked back up.
        """
        return self._orchestration_root(iteration) / "lane_plans.json"

    def _persist_lane_plans(
        self,
        iteration: int,
        plans: Sequence[str],
        *,
        analysis_commit: str,
    ) -> Path:
        """Atomically publish every lane's plan and return lane 1's path.

        All of them, not just the one the Implementer is handed: each is a
        separately paid-for LLM answer, and leaving lanes 2..N in memory alone
        means a round that crashed cannot say what it asked those lanes to do
        and cannot be resumed without buying the same plans again.

        A round can be planned twice at one iteration -- a fan-out that loses
        its workspace copies falls back to the single-session path, which plans
        the same iteration again at width one -- so a narrower round prunes the
        wider one's leftovers first. Reading a stale ``lane_003.md`` back would
        hand a recovered round a plan this iteration never issued.

        ``analysis_commit`` is the tree the plans describe, recorded with them
        because that is what a later process has to check before reusing them.
        """
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
        """One round's commit and plans in lane order, or None if it has none.

        None covers every way a round can fail to offer a usable set: it never
        published (which is most iterations, since only a planning round does),
        it died before its manifest, or a plan the manifest counts is missing.
        A partial set is not a narrower round -- it is damage, and the lanes it
        would silently drop were paid for like the rest.
        """
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
        """The iteration before ``before`` that started and reported no result.

        Every terminal path of an iteration -- including a planning outage --
        appends one ``iteration_result``, so a started iteration missing one is
        an iteration this process or a previous one died inside. There is at
        most one: the loop runs them in sequence, and the next iteration cannot
        start until the current one has been recorded.

        ``before`` is the iteration asking, and it is excluded: the loop marks
        an iteration started before it plans anything, so the asking iteration
        is itself always started and always unfinished, and answering with it
        would mean nothing is ever recovered.

        Only the latest below that is answered for. An older gap means the loop
        already moved past that iteration, which is a decision this must not
        revisit.
        """
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
        """A previous round's plans that were paid for and never dispatched.

        A fan-out round buys N plans before it runs a single session, and the
        lane workspaces those sessions edit are temporary: a process that dies
        anywhere in the round loses the candidates outright. Re-running the
        sessions is what the loop would do next in any case, so the only thing
        worth carrying across the crash is the planning, which is also the part
        that was paid for in tokens.

        The key is the iteration that started and never finished, because that
        is exactly the round whose plans were never spent -- a round that
        reported a result consumed them, and reusing those would re-issue
        directions the loop has already ruled on.

        The plans are still refused when the tree has moved under them: a KEEP
        committed just before the crash is recovered before this runs, and a
        plan written against the previous commit describes code that is no
        longer there.
        """
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
        """Run the autonomous iteration loop.

        Args:
            agent_fn: Async function that modifies the kernel file.
                      Signature: async fn(kernel_path, experiment_history) -> rationale
                      If None, runs validation/bench only (for testing the pipeline).
            orchestration_service: Optional read-only planning chain that dispatches
                      specialists and synthesizes one optimization plan before
                      each Implementer session.
            analysis_service: Optional commit-bound Analysis Agent that produces
                      source, profiling, bottleneck, potential, and direction artifacts.
            on_iteration: Callback after each iteration for logging/display.
            on_best_committed: Callback immediately after a validated KEEP is
                               committed and becomes the durable best, before
                               any post-KEEP profiling.
            usage: Optional ``UsageAccumulator`` the agent_fn folds each query's
                   token spend into. When given, the run's total is persisted
                   onto the experiment record and exposed as ``self.llm_usage``.
            supervisor_fn: Optional async ``fn(digest, reason, workspace) -> str``
                   that reviews the stalled trajectory and returns fresh
                   optimization directions. When supplied (the forge-loop always
                   supplies one), a detected stall makes the loop inject the
                   returned directions and CONTINUE. When None, no interventions
                   happen and the loop simply runs to the time / iteration budget
                   (there is no plateau early-stop).

        Returns:
            List of all iteration results.
        """
        import functools

        global print
        print = functools.partial(print, flush=True)

        self.start_time = time.time()
        self.results = []
        self._usage = usage

        # Safety net: if the kernel is an aiter HIP kernel, force it to recompile
        # from the current source (AITER_REBUILD) so the agent's edits are never
        # silently ignored via aiter's prebuilt in-tree .so. Env is set once and
        # inherited by every build spawned afterwards. Some upper-layer frameworks
        # also do this centrally; this covers direct forge-loop usage.
        force_jit_rebuild(self._jit_source_files())

        # Cross-iteration objective ledger: records each iteration's net diff,
        # measured outcome, and real error signatures, then feeds concise
        # toolchain observations and recent entries into the next prompt.
        self.ledger = ExperienceLedger(self.ic.workspace_dir)

        # Per-iteration lesson documents. A dedicated summarizer session resumes
        # each finished implementer session and records EVERY direction it explored
        # — including the ones it abandoned, which survive nowhere else. Written
        # after the keep/revert verdict so the loop can stamp the measured
        # outcome onto the same document. Best-effort throughout.
        self.lessons = LessonStore(self.ic.workspace_dir)
        try:
            self.handoff_store = HandoffStore(self.ic.workspace_dir)
        except Exception:
            self.handoff_store = None
            log.debug("handoff store initialization failed", exc_info=True)

        # Full-fidelity candidate archive: persists each iteration's WHOLE
        # solution (kernel snapshot + diff + full profile + measurements +
        # decision) so a later iteration can read back any prior attempt's real
        # code. Best-effort — never breaks the loop.
        self.archive = CandidateArchive(self.ic.workspace_dir, self.ic.kernel_file)
        self.best_publisher = BestResultPublisher(self.ic.workspace_dir)
        # Candidates from one concurrent fan-out, spent one per iteration so each
        # is measured and judged on its own by the ordinary decision path.
        self._lane_queue: list[LaneResult] = []
        self._last_lane_plans: list[str] = []
        # Stacked iterations run back to back so far. Held per session rather
        # than in the run state: what it bounds is how long this loop can go
        # without reaching the queue-empty branch, and a resumed session enters
        # that branch on its own terms.
        self._merge_precedence_streak = 0
        # A device the campaign may not measure on, recorded by whichever
        # iteration found it and re-checked by every iteration after it. What
        # holds the device is often nothing this campaign may kill, so nothing
        # about the end of an iteration makes it leave.
        self.device_hazard = DeviceHazardLog(self.ic.workspace_dir)

        # Durable, file-backed run state + append-only event log. These make the
        # loop's control signals (best / stall / phase / termination) resumable
        # and inspectable instead of living only in memory,
        # so a long-horizon run is driven from files rather than an ever-growing
        # prompt. Best-effort — never breaks the loop.
        self.state_store = LoopStateStore(self.ic.workspace_dir)
        state_exists = self.state_store.state_path.exists()
        self.run_state = self.state_store.load()
        current_ruling_path = latest_supervisor_ruling_path(self.ic.workspace_dir)
        self._supervisor_ruling = load_latest_supervisor_ruling(self.ic.workspace_dir) if self.resume else ""

        # Ownership boundary for anything a candidate creates, taken before
        # this loop touches the workspace. Resume recovery discards, and it
        # runs before the first iteration, so a snapshot taken only at the top
        # of the loop leaves it with none -- and it would then clean the whole
        # allowlisted set, deleting an operator's file irrecoverably. Refreshed
        # per iteration below; this is the floor under it.
        self._pre_untracked = self._untracked_snapshot()

        if self.resume:
            self._validate_resume_scoring_state(self.run_state)
            # Pending KEEP reconciliation promotes the committed candidate and
            # checkpoints scoring state. Restore calibrated floors and the
            # pristine SNR first so that checkpoint cannot replace them with
            # this new process's empty/default constructor values.
            self._restore_scoring_state()
            # Publication reconciliation consumes both baseline anchors. Restore
            # them before pending/best repair so manifests retain total and
            # incremental semantics from the original session.
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

        # Anchor the campaign clock now that the state carrying what earlier
        # sessions spent is loaded. A fresh campaign banks nothing and its
        # origin is this process's own start; a resumed one starts that much
        # further back, so the campaign-cumulative totals in ``round_costs``
        # and the span they are divided by keep measuring the same thing.
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

        # Persist only after the fresh-campaign guard has completed. Both writes
        # are best-effort: a rejected invocation must leave the PR sidecar
        # untouched, and a failed one must not abort the campaign.
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

        # Self-supervision monitor (AVO): tracks stall / unproductive-cycle signals
        # so the loop can call the supervisor to redirect the search instead of
        # stopping at the first plateau. Active whenever a supervisor_fn is
        # supplied (the forge-loop always supplies one — supervision is a
        # first-class part of the loop, not an optional toggle).
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
        # The finalize reserve is an absolute admission guard; on a SHORT budget
        # it can swallow most of the window (e.g. a 30-min reserve on a 1h run
        # leaves only 30 min for iterations). Warn when it claims >= half the
        # budget so the operator can raise --max-hours or shrink the reserve.
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

        # The CLI constructs IterationLoop before applying a KB warm-start, then
        # records the freshly measured pristine case timings on IterationConfig.
        # Refresh the immutable runner snapshot here so warm-start campaigns can
        # calculate mean case speedup instead of failing closed with no baseline cases.
        self._set_baseline_case_times(self.ic.baseline_case_times)
        if not self.resume and not self.ic.warm_start_commit and self._baseline_case_times:
            self._best_case_times = dict(self._baseline_case_times)
            self._unscored_cases = {str(case_id) for case_id in self.ic.preloop_baseline_unscored_cases}
        if self._best_case_times:
            self._persist_scoring_state()

        # Anchor speedup reporting. If the task didn't supply a baseline, bench
        # the pristine kernel before the agent touches it — otherwise the
        # experiment reports speedup=None ("no perf uplift" in `list`/`report`).
        # KB warm-start carries the same three-measurement aggregate in.
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

        # The scoring model defines the pristine kernel as 1.0x. Raw wall time is retained
        # only for diagnostics; KEEP/REVERT compares best_mean_case_speedup.
        if self.ic.baseline_wall_ms is not None:
            self.best_wall_ms = self.ic.baseline_wall_ms
        if self._baseline_case_times:
            self.best_mean_case_speedup = 1.0

        # Seed the run state's baseline and, guardedly, resume a prior best from
        # a reused workspace (only when the recorded best commit is still HEAD).
        self._seed_and_hydrate_run_state()
        self._adopt_validated_warm_start()

        # After the resume restore and the warm-start adoption above, never
        # before them. Seeding first would measure whatever is checked out --
        # on a resume that is the current incumbent, and the seed persists it,
        # overwriting the stored pristine reference that the restore is about
        # to read. The reference would then track the incumbent and ratchet
        # downwards one resume at a time, which is the drift this gate exists
        # to catch. A restored reference makes this a no-op.
        if not self._baseline_case_times:
            raise RuntimeError(
                "mean case scoring requires pristine per-case timings before starting an optimization iteration"
            )

        # Every speedup this run reports is a ratio against those timings, so a
        # baseline that drifted from the task's own reference poisons the whole
        # campaign. Check it before the agent spends any budget against it.
        # Only a minority of tasks ship the reference, so say which runs the
        # anchor was actually verified for, and over how many of this run's own
        # cases. Staying quiet when it is missing reads the same as passing,
        # and so does a bare count that covers one case out of twelve.
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

        # A crash immediately after a verified commit can leave the KEEP's
        # archive unfinished. Complete it before starting new work.
        await self._finish_recovered_pending_keep()

        # Candidates an earlier round bought and never measured. Read before the
        # first iteration because the loop only fans out on an empty queue, so
        # this is what decides whether the next round is measured or planned.
        self._restore_lane_queue()

        # The previous round's verdict, for the same reason: a REPLACE is spent
        # on the round after the one it judged, and the budget often ends
        # between the two.
        self._restore_critic_ruling()

        # Analyze the baseline canonical commit once before any specialist or
        # Implementer session. The result remains active across every REVERT.
        if analysis_service is not None:
            await self._resolve_analysis_context(analysis_service)

        iteration = self.run_state.next_iteration - 1
        while True:
            iteration += 1

            # Build the lineage digest once per iteration — reused by BOTH the
            # supervisor (trajectory to review) and the implementer (prompt history).
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
            # Ruled on once per iteration, here with the other conditions that
            # decide whether this iteration may run at all. A hazard nothing
            # clears is terminal rather than a quiet spin: a foreign process may
            # hold the device for the rest of the campaign, and retrying until
            # the budget ends produces nothing while reporting nothing wrong.
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

            # Price the round before anything is spent on it. An iteration that
            # drains a lane candidate a previous round already bought plans
            # nothing and is not a round; refusing it would throw away work the
            # budget has already paid for. A stacked iteration plans nothing
            # either and is passed over here the same way -- but it is the only
            # iteration that also drains nothing, so it is the only one that can
            # hold this branch off without limit. What bounds that is
            # :data:`MERGE_PRECEDENCE_STREAK_LIMIT`, argued at
            # :meth:`_merge_attempt_refusal`.
            #
            # The gate is deliberately not lifted out of this branch to catch
            # the stacked iteration instead. It prices a whole round -- planning,
            # session and measurement -- and its refusal ends the campaign, so
            # asking it of an iteration that buys only the last of the three
            # would end campaigns over a round they were not about to buy, while
            # they still held candidates a round had already paid for. What the
            # stacked iteration does buy is held back above it by the reserve:
            # :meth:`_is_budget_exhausted` keeps ``budget_reserve_sec`` (1800s by
            # default) back from every iteration of every kind, against a
            # canonical measurement priced at 600s before a campaign has observed
            # one of its own and costing 150s at the worst of the 171 production
            # cycles :mod:`kernelforge.loop.round_budget` is calibrated on.
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

            # Resolve Analysis before any Supervisor or planning call. Small
            # KEEP gains reuse the published evidence; cumulative gain or stale
            # evidence at a Supervisor boundary refreshes it exactly once.
            if analysis_service is not None:
                await self._resolve_analysis_context(
                    analysis_service,
                    supervisor_due=supervisor_due,
                    iteration=iteration,
                )

            # Self-supervision (AVO): when supervised, a stall triggers a reviewer
            # that injects fresh directions and the loop ALWAYS CONTINUES — it
            # never self-terminates on stall. The run stops ONLY when the remaining
            # time cannot admit another session or the gate is met; a stalled stretch just gets
            # more supervisor directions, not an early exit.
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
                        # A new review attempt supersedes the prior stall
                        # episode's ruling even when the backend returns empty.
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

            # Re-scope the ownership boundary to this iteration: untracked
            # files already here are the operator's or an earlier round's, and
            # this iteration's REVERT must not delete them. A snapshot that
            # cannot be taken leaves the previous one standing rather than
            # widening what a REVERT may delete.
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
            # What a fan-out round leaves this iteration holding, so the
            # single-session path below spends it instead of buying it again.
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
                # The round's own lanes share one device with the canonical
                # measurement, so a lane whose teardown could not clear it has
                # just refused this iteration -- whatever its siblings produced.
                # Read before the budget verdict below: the hazard is a fact the
                # next session has to see, and the verdict is a decision to stop.
                hazard = self.device_hazard.live
                if self._refused_round:
                    # Planning cost more than the round had left. The plans it
                    # bought are published, and this iteration deliberately
                    # records no result, which is what lets the next session
                    # recover them.
                    break
            # Stacking two rejected gains costs a measurement but no session, so
            # it is tried before spending another Implementer round on a search
            # that has stopped producing a new best -- and, once that search has
            # stalled, before draining a candidate the same search bought.
            #
            # A queued candidate held the iteration unconditionally, and across
            # the thirty archived forge runs of 2026-08-22 and 08-23 one was
            # waiting on 409 of 549 iterations (74.5%), so a pair was not even
            # selected on three iterations in four. What the deference buys is
            # measurable, and it stops buying anything at exactly the depth
            # stacking already waits for: a queued candidate is kept 55.1% of the
            # time (119/216) while the search is still producing, 33.7% (33/98)
            # from ``MERGE_ATTEMPT_STALL_THRESHOLD`` on (z=3.5), and no worse
            # deeper -- 33.3% (19/57) at a stall of three (z=0.04 against the
            # threshold). Waiting past the threshold therefore costs firings --
            # replayed with the held-plan guard in place, 9 at the threshold
            # against 6 at three and 4 at four -- and adds no evidence, so the
            # queue yields on the same stall depth that admits a pair at all,
            # and this needs no second constant.
            #
            # A stack is staged into the working tree and a taken candidate has
            # already applied its own diff there, so which one the iteration
            # runs is settled before the queue is touched. A pair that does not
            # stage leaves the queue to drain below in this same iteration.
            #
            # The queue yields, but a held fan-out plan does not. Its only
            # consumer is the single-session path below, and a merge iteration
            # records a result, which is what stops the next process from
            # recovering the round -- so a whole planning round, dispatch plus
            # every specialist plus synthesis, would disappear with nothing said.
            # When the round holds an outage rather than a plan, dropping it
            # means the orchestration error is never recorded and the circuit
            # breaker never counts it, and a stall -- the condition that selects
            # a pair -- is when repeated orchestration failure is most likely.
            # Deferring costs nothing: spending the plan does not clear the
            # stall, so the same pair is selectable next iteration.
            merge_pair = None if hazard or fan_out_plan is not None else self._select_merge_attempt()
            merge_refusal = "" if merge_pair is None else self._merge_attempt_refusal()
            lane_queue_depth = 0 if hazard else len(self._lane_queue)
            if merge_refusal:
                merge_diff, merge_obstacle = "", merge_refusal
            else:
                merge_diff, merge_obstacle = self._stage_merge_attempt(merge_pair)
            self._merge_precedence_streak = self._merge_precedence_streak + 1 if merge_diff else 0
            if merge_diff and lane_queue_depth:
                # Distinct from ``merge_attempt_staged``, and not foldable into
                # it: that counts every stack measured, this counts the ones
                # that went ahead of a queue a round already paid for, which is
                # the only thing precedence can cost.
                #
                # What is recorded is the depth of that queue, which is not the
                # number of measurements this stack displaced and is named so it
                # cannot be read as one. At most ONE of those entries would have
                # been measured this iteration -- ``_take_lane_candidate``
                # returns a single candidate -- and possibly none, since
                # ``_next_lane_candidate`` drops an entry that changes the
                # measurement surface or whose diff no longer applies, and both
                # of those are only knowable by popping the queue and writing
                # the tree. So the depth bounds from above what a stack delays,
                # and measures exactly what a run that ends early would leave
                # unmeasured, which is where the cost of precedence actually
                # lands: nothing is discarded here, and the queue drains an
                # iteration later. Replayed over the same thirty runs with the
                # held-plan guard in place, yielding strands nothing extra at
                # all -- 13 candidates across 13 runs, the same as never
                # yielding -- because a stack only ever takes an iteration the
                # queue was going to be drained on anyway, and the run has the
                # slack to absorb the delay: a stacked measurement is 0.8 min
                # against the 34.0 min of an iteration that opens a round, and
                # every run that stranded anything ended with at least 8.6 min
                # of budget it could not spend (median 23.5).
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
                    # A refusal did not reach the pair's diffs, so it is not
                    # evidence about them and must not be remembered against
                    # them. The pair stays selectable and is measured once a
                    # non-stacked iteration has reset the streak.
                    about_the_iteration=bool(merge_refusal),
                )
            # A fan-out round already paid for these candidates, so they are
            # measured before anything new is planned. Under a live hazard
            # nothing is taken and nothing is staged: the candidates were bought
            # and stay queued for an iteration that can measure them.
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
                # How often the mechanism engaged. Paired with
                # ``merge_attempt_kept`` below, which is how often it changed an
                # outcome; the two are different numbers and a reader counting
                # either one alone learns the wrong thing about it.
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
                # The last point before the round buys its session, and the
                # first at which what planning cost is a measurement rather
                # than an estimate. A round already refused inside the fan-out
                # never reaches this; one that fell back to a single session
                # after planning is priced here against what is left of the
                # budget now, not what was left before it planned.
                if not self._admit_dispatch(iteration):
                    break
                session_sink["session_started"] = True
                # Cross-iteration experience assembled from complementary
                # sources (AVO-style lineage view), each carrying what the
                # others cannot:
                #   * the candidate ARCHIVE digest — the trajectory table + full
                #     diffs of the best/near-miss/recent attempts + a pointer to
                #     the on-disk archive so the agent can Read any prior kernel.
                #   * the LESSON documents — what each recent session actually
                #     explored, in its own words, including the directions it
                #     abandoned (which leave no diff behind).
                #   * the experience LEDGER — objective toolchain observations
                #     distilled from machine-verified failure signatures.
                # Fall back to the compact in-memory history when all are empty
                # (e.g. iteration 1). ``digest`` was built once at the top of the
                # loop and is shared with the supervisor.
                # State-driven compact header (overview + retrieval map). When it
                # renders it REPLACES the heavy inline archive digest in the
                # IMPLEMENTER prompt: the agent reads full diffs from files on demand
                # (via the retrieval map) instead of carrying them in context, so
                # the prompt stays flat over a long run. The full digest is still
                # handed to the SUPERVISOR (an occasional call that reviews the
                # whole trajectory). Best-effort; empty on a cold start.
                lh_header = _long_horizon_header(
                    self.run_state,
                    self.state_store,
                    self.handoff_store,
                )

                # Lesson documents from the most recent iterations, verbatim,
                # plus the absolute path of the directory holding every past
                # one. These are factual session records, including abandoned
                # attempts that survive nowhere else. They are evidence, not
                # instructions for the current iteration.
                #
                # Each is rendered against the scored suite and the constants
                # the declared source files assign RIGHT NOW, so a negative
                # measured under a premise that has since moved arrives marked
                # re-openable instead of arriving as a standing ban.
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
                    # The session narrative lives in the lesson documents, so the
                    # ledger contributes only objective toolchain observations
                    # once any lesson is available. Without lessons, keep the prior
                    # behavior: full block when the digest is not inlined,
                    # constraints-only when it is.
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
                # The latest free-form Supervisor Ruling is durable across KEEP
                # and resume. It may reject subjective conclusions in historical
                # lesson records, while objective validation and measurements
                # remain authoritative.
                if self._supervisor_ruling:
                    history = (
                        "## Latest Supervisor Ruling\n"
                        "This review is the current planning authority. It "
                        "overrides subjective recommendations or conclusions in "
                        "historical lesson records, but never overrides objective "
                        "validation or measurement facts.\n\n"
                        f"{self._supervisor_ruling}\n\n{history}"
                    )
                # The long-horizon header (rendered above) goes at the very TOP,
                # above the supervisor/analyst/pmc prepends, as the compact memory
                # frame the implementer reads first.
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
                # Pass extras only to agent_fns that declare them: the in-session
                # gate uses the current best mean case speedup and immutable pristine
                # per-case timings; session_sink hands back findings and the
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
                    # The SDK result stream has completed (or unwound). Persist
                    # its cumulative usage before canonical validation can run
                    # long enough for an external hard timeout to kill the run.
                    self._checkpoint_llm_usage()

                commit_hash = ""
                # Either kind of "this candidate exists but must not be
                # measured": protected state was tainted, or the workspace is
                # still busy. Both skip the canonical surface entirely rather
                # than run it and believe the number it returns.
                unmeasured_result: IterationResult | None = None
                if session_sink.get("integrity_violation") is True:
                    # Capture evidence before restoration. No driver, harness, or
                    # source oracle may execute while protected state is tainted.
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
                    # The session's own processes are still running in the
                    # workspace, or someone else's are and they are not ours to
                    # kill. Either way the device is busy, so a benchmark taken
                    # now measures this candidate plus whatever is sharing the
                    # GPU with it -- which is worse than no measurement, because
                    # it is a number the loop would act on.
                    contention = str(session_sink["workspace_contention"])
                    # Nothing about the end of this iteration makes those
                    # processes leave, so the refusal is recorded and re-checked
                    # rather than forgotten here. Only the reaper's description
                    # crosses the backend boundary, so the pids are gathered
                    # again from the directory it was reporting on -- they are
                    # still there, which is the whole complaint.
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
                        # The candidate itself may be sound, but nothing here
                        # can establish that, and HEAD has to stay at the last
                        # measured best rather than carry an unmeasured diff
                        # into the next iteration.
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
                    # The driver is the measurement boundary and must remain
                    # byte-for-byte canonical. Recheck after the agent session so
                    # hook bypasses cannot influence correctness or KEEP.
                    self._validate_driver_integrity(self.run_state)

                    # Keep HEAD at the last validated best state while this
                    # candidate remains unverified.
                    attempt_diff = self._working_tree_diff()
                    if not attempt_diff.strip():
                        # An outage leaves the same empty diff as a deliberate
                        # no-op. Label it as what it was, so the ledger never tells
                        # the next Session "the agent chose to change nothing".
                        api_failed = session_sink.get("end_reason") == EXHAUSTED_END_REASON
                        # A file the agent created is not in the tracked diff.
                        # Skipping the candidate here would leave it on the tree
                        # for the next iteration to be measured with, which is
                        # the leak an uncommittable new file causes.
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

            # Snapshot the best-so-far BEFORE this iteration is measured, so the
            # archive can record the true delta vs the standard it had to beat
            # (run_one_iteration mutates self.best_wall_ms on an improvement).
            best_before = self.best_wall_ms
            best_mean_case_speedup_before = self.best_mean_case_speedup

            # Run validation + bench. Pass the agent's one-sentence change plan
            # (from the in-session gate) so it's persisted onto the iteration.
            # Defense-in-depth: a single iteration's build/validate/bench crash must
            # never kill a multi-hour run. No git revert has happened yet at this point,
            # so on an unexpected exception we revert this candidate once, record the
            # failed attempt, and continue while the time budget admits another session.
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
                    # Turn the crash into a FAILED result (crashed=True) and let it
                    # flow through the same verdict/ledger/archive path.
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
            # Session end reason + turns spent (from the in-session gate / SDK via
            # session_sink) — persisted so a run's end-reason distribution
            # (edit cap / turn cap / converged / …) is analyzable.
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
                # Defer SIGTERM/SIGINT across the durable best-commit publication
                # (main #hardening) so a kill mid-checkpoint cannot leave the
                # pending-keep/run-state half-written.
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
                        # No-agent measurement path: there is no candidate diff to
                        # commit, but the measurement can still establish a best.
                        self._promote_best(result)
                        self.best_mean_case_speedup = result.mean_case_speedup
                    # Bridge to the caller's checkpoint sink (main): lets the CLI
                    # persist a Hyperloom-recovery checkpoint JSON alongside our
                    # run-state durability.
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

            # Remote/external work belongs outside the SIGTERM deferral window
            # but still precedes potentially long post-KEEP profiling.
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

            # Reduce this finished iteration into the durable run state + append a
            # factual event, then checkpoint. This keeps run_state.json in lockstep
            # with the loop's live best/stall so a restart can resume from files.
            if not keep_checkpoint_finalized:
                self._record_iteration_outcome(
                    result,
                    plan=session_sink.get("plan", ""),
                )

            # The verdict is now known, so ask the just-finished implementer session
            # to record what it explored, then stamp the measured outcome onto
            # the same document. This runs AFTER keep/revert (the session cannot
            # know its own verdict). The free-form record is not distilled into
            # the ledger or archive.
            decision_label = _decision_label(result)
            if merge_diff and merge_pair is not None:
                print(
                    f"  [merge] iterations {merge_pair[0].iteration}+"
                    f"{merge_pair[1].iteration} measured stacked: {decision_label}"
                )
                if decision_label == "KEEP":
                    # How often the mechanism changed an outcome, which is the
                    # other of the two numbers ``merge_attempt_staged`` carries.
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
            # Objective fields (diff summary, outcome, error signatures) come
            # from the loop/gate. Best-effort — a ledger failure must never break
            # the loop.
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

            # Archive the full solution and measurements as a derived view. The
            # compact run state, KEEP event, Git commit, and external callback are
            # already authoritative and can rebuild this view after an I/O fault.
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

            # A KEEP makes the prior evidence stale but does not discard its
            # paths. The next iteration retargets the active context to the new
            # canonical and refreshes only at the cumulative-gain or Supervisor
            # boundary.
            if result.kept:
                self._analysis_bundle = None

        # Whatever the last iteration cost belongs to this campaign's history
        # even though no further round will read it: a resumed session will.
        self._close_round()
        self.best_publisher.refresh_round_budget(self._round_budget_summary())

        # Persist the terminal control state so a resume/inspection sees why the
        # run ended and what the final best was. Best-effort.
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

        # Persist the run's total LLM token spend onto the experiment record so
        # external callers can read the token cost.
        self._checkpoint_llm_usage()

        # Auto-evolve: run post-experiment learning
        if self.experiment:
            try:
                self.tracker.mark_complete(self.experiment.experiment_id)
            except Exception:
                log.debug("failed to mark experiment complete", exc_info=True)
            try:
                learned = self.evolver.on_experiment_complete(self.experiment)
                if learned.get("lessons"):
                    print(f"  Lessons learned: {len(learned['lessons'])}")
                if learned.get("transfer_rules"):
                    print(f"  Transfer rules discovered: {len(learned['transfer_rules'])}")
            except Exception:
                log.debug("auto-evolve post-experiment learning failed", exc_info=True)

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
            # Campaign totals, not this session's, so they are labelled as such
            # and the share is taken against the campaign clock rather than the
            # ``total_time`` printed just above. That line covers this process;
            # these counters cover every session the campaign has run, and
            # dividing one by the other is how a resumed campaign came to
            # report a planning share above 100%.
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
            cost_available = self.llm_usage.get(
                "cost_available",
                "total_cost_usd" in self.llm_usage,
            )
            cost_text = f"${self.llm_usage['total_cost_usd']:.2f}" if cost_available else "cost unavailable"
            print(
                f"  LLM spend: {self.llm_usage['input_tokens']:,} in / "
                f"{self.llm_usage['output_tokens']:,} out tokens, "
                f"{cost_text} "
                f"({self.llm_usage['calls']} calls)"
            )
        print(f"  Experiment: {self.experiment.experiment_id}")

        return self.results


# Decision labels that can possibly mean nothing came out worse this iteration:
# a kept candidate, and the ones that mean no candidate was ever measured. Every
# other label — CRASH and BUILD_FAILED as much as any REVERT_* — is the loop
# observing something fail or regress. A whitelist rather than a pattern because
# CRASH and BUILD_FAILED share no prefix with REVERT and leave no speedup behind.
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
    """The canonical keep/revert label for one finished attempt.

    Shared by the run-state event, the lesson document's outcome line, and the
    candidate archive so all three always agree on what happened.
    """
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
    """The compact long-horizon header for the Implementer prompt, or "".

    The header renders recent attempts up to its own budget and labels each
    pinned iteration with the measured mean case speedup it finds among the
    outcomes it is handed, so it is given a window counted in iteration outcomes
    (``LONG_HORIZON_OUTCOME_WINDOW``) rather than a tail of raw log events.

    The window is read outside the render guard below because
    ``recent_results`` refuses a request wider than its cache instead of
    answering short: swallowing that refusal would drop the header from every
    prompt of the run rather than report a window the store cannot serve.
    """
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
