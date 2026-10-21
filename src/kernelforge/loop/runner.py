# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Durable, evidence-driven autonomous kernel optimization loop."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


from kernelforge.agent_backends.session_resume import EXHAUSTED_END_REASON
from kernelforge.llm.process_reaping import processes_under
from kernelforge.config import Config
from kernelforge.evaluation.correctness.canonical import accept_candidate
from kernelforge.evaluation.correctness.pipeline import run_validation_pipeline
from kernelforge.loop.experience import ExperienceLedger
from kernelforge.loop.lessons import (
    SUMMARY_MIN_SECONDS,
    UNDISPROVEN_CLAIM,
    LessonStore,
    OutcomeMeasurement,
    PrReferenceNote,
    SessionEnd,
    SessionObservations,
    build_fallback_document,
    format_outcome_line,
    format_scope_line,
    is_claim_disproved,
    summarize_iteration,
)
from kernelforge.loop.archive import CandidateArchive
from kernelforge.loop.handoffs import HandoffStore
from kernelforge.loop.jit_rebuild import (
    force_jit_rebuild,
)
from kernelforge.loop.analysis_runtime import AnalysisRuntimeMixin
from kernelforge.loop.relay_runtime import RelayRuntimeMixin
from kernelforge.loop.search_policy import (
    SearchPolicyDecision,
    SearchPolicyEngine,
)
from kernelforge.loop.run_state import (
    MAX_PINNED_ITERATIONS,
    ORCHESTRATION_CIRCUIT_OPEN,
    SESSION_COMPLETED,
    BestRecord,
    LoopStateStore,
    RunState,
    begin_orchestration_probe,
    complete_orchestration_probe,
    is_infrastructure_decision,
    make_event,
    reconcile_stale_running_session,
    start_session,
)
from kernelforge.orchestrator.supervisor import (
    latest_supervisor_ruling_path,
    load_latest_supervisor_ruling,
)
from kernelforge.loop.prompt_view import (
    MAX_RECENT_ATTEMPT_LINES,
    render_long_horizon_header,
)
from kernelforge.loop.reporting import BestResultPublisher
from kernelforge.evaluation.benchmark.runner import (
    CaseCoverageError,
    calculate_mean_case_speedup,
    BenchRun,
    measure_wallclock,
)
from kernelforge.tools.process_group import communicate_process_group
from kernelforge.evaluation.scoring import (
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
    runs_task_suite_acceptance,
)
from kernelforge.loop.device_hazard import DeviceHazardLog
from kernelforge.loop.fanout import LaneResult
from kernelforge.loop.merge_candidates import (
    merge_plan,
)
from kernelforge.evaluation.profiling.registers import check_registers
from kernelforge.tracker import CampaignSegment, ExperimentTracker, Experiment, ExperimentSubject
from kernelforge.loop.iteration_types import (
    CONFIG_COVERAGE_MIN_MOVE_RATIO,
    CaseConfigCoverage,
    DirectionEvidence,
    HeldRound,
    IterationConfig,
    IterationResult,
    LoopAgents,
    LoopCallbacks,
    WindowGain,
    _Attempt,
    _IterationFrame,
    _IterationSupply,
    _RoundPlan,
    _Ruling,
)
from kernelforge.loop.budget_runtime import BudgetRuntimeMixin
from kernelforge.loop.ceiling_runtime import CeilingRuntimeMixin
from kernelforge.loop.lane_runtime import LaneRuntimeMixin
from kernelforge.loop.outcome_runtime import (
    OutcomeRuntimeMixin,
    _decision_label,
)
from kernelforge.loop.planning_runtime import PlanningRuntimeMixin
from kernelforge.loop.publication_runtime import (
    PublicationRuntimeMixin,
    llm_spend_lines,
)
from kernelforge.loop.resume_runtime import (
    PENDING_KEEP_FIELDS,
    ResumeRuntimeMixin,
)
from kernelforge.loop.scoring_runtime import (
    CONFIG_COVERAGE_DISPERSION_MULTIPLE,
    ScoringRuntimeMixin,
    _measurement_case_times,
)
from kernelforge.loop.warm_start_runtime import WarmStartRuntimeMixin
from kernelforge.loop.workspace_runtime import (
    LOOP_ARTIFACT_ROOT,
    WorkspaceRuntimeMixin,
)

__all__ = [
    "CONFIG_COVERAGE_DISPERSION_MULTIPLE",
    "CONFIG_COVERAGE_MIN_MOVE_RATIO",
    "CaseConfigCoverage",
    "DirectionEvidence",
    "HeldRound",
    "IterationConfig",
    "IterationLoop",
    "IterationResult",
    "LONG_HORIZON_OUTCOME_WINDOW",
    "LOOP_ARTIFACT_ROOT",
    "LoopAgents",
    "LoopCallbacks",
    "PENDING_KEEP_FIELDS",
    "SigmaResolution",
    "WindowGain",
    "llm_spend_lines",
]

log = logging.getLogger(__name__)

# How many recent iteration outcomes the long-horizon prompt header is built from.
LONG_HORIZON_OUTCOME_WINDOW = max(
    MAX_PINNED_ITERATIONS,
    MAX_RECENT_ATTEMPT_LINES,
)


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

    Both streams are read. ninja prints the compiler's own output on stdout, so
    reading stderr alone reports a ninja failure to the agent as
    ``BUILD FAILED:`` and nothing else.
    """
    combined = b"\n".join(part.strip() for part in (stdout or b"", stderr or b"") if part.strip())
    text = combined.decode("utf-8", errors="replace").strip()
    return text[-limit:] if text else "no build output"


def _line_buffer_stdout() -> None:
    """Flush stdout at every line, whichever loop module printed it.

    The caller reads the loop's progress from a pipe, which Python block-buffers by default.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(line_buffering=True)


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


class IterationLoop(
    WorkspaceRuntimeMixin,
    BudgetRuntimeMixin,
    CeilingRuntimeMixin,
    ScoringRuntimeMixin,
    LaneRuntimeMixin,
    PlanningRuntimeMixin,
    OutcomeRuntimeMixin,
    ResumeRuntimeMixin,
    PublicationRuntimeMixin,
    WarmStartRuntimeMixin,
    RelayRuntimeMixin,
    AnalysisRuntimeMixin,
):
    """Autonomous kernel optimization loop."""

    def __init__(
        self,
        iter_config: IterationConfig,
        tracker: ExperimentTracker,
        config: Config | None = None,
        resume: bool = False,
        ceiling_estimator: Callable[..., Awaitable[Any]] | None = None,
    ):
        self.ic = iter_config
        # Produces a ceiling report for this kernel, given the case set and per-case latencies the baseline
        # established; ``None`` unless the operator turned the roofline ceiling on. Injected rather than built here so
        # the loop needs no view of the agent-backend registry: the analyst is a different role, and resolving it is
        # the caller's job.
        self._ceiling_estimator = ceiling_estimator
        # Where this campaign's ceiling was published, once it has one: set by ``_establish_ceiling``, never by the
        # caller. It decides when to stop, never which candidate is better: see ``_is_roofline_target_met``.
        self._ceiling_report_path = ""
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
        # What the kernel the search starts from scores against the anchor. 1.0 whenever that kernel IS the anchor,
        # and the port's own speedup when a caller supplied the anchor it was ported from.
        self.search_start_mean_case_speedup: float | None = None
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
            candidate_diff, _ = self._candidate_changes(base_head)
            if candidate_diff and hashlib.sha256(candidate_diff.strip().encode()).hexdigest() != expected_hash:
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
        if subject != pending["commit_subject"]:
            raise ValueError("pending KEEP commit message mismatch")
        return "committed"

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
        # The gate's tracked-diff fingerprint does not bind untracked source bytes.
        if self.ic.commit_new_paths and self._new_paths()[0]:
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

    async def _measure_baseline(self) -> float | None:
        """Bench the pristine kernel before any agent edit — the speedup anchor."""
        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *self.ic.build_command,
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
            BenchRun(self.ic.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
            measurements=KEEP_MEASUREMENT_COUNT,
            timeout_sec=self.ic.bench_timeout_sec,
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
        self.search_start_mean_case_speedup = baseline_score
        self._best_case_times = dict(baseline_case_times)
        self._unscored_cases = set(unscored_cases)
        self._persist_scoring_state()
        return bench_result.get("median_ms")

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
                BenchRun(self.ic.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
                measurements=SIGMA_REMEASURE_BATCH,
                timeout_sec=self.ic.bench_timeout_sec,
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
            plan=f"Validated start: {self.ic.warm_start_solution_slug}".strip(),
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
            validation_summary=f"Validated start: {self.ic.warm_start_solution_slug}; correctness passed, timings recorded",
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
                self._apply_new_iteration_outcome(result, decision_label, plan)
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
                    evidence=DirectionEvidence(
                        mean_case_speedup=result.mean_case_speedup,
                        best_mean_case_speedup=self.best_mean_case_speedup,
                        bench_detail=result.bench_detail,
                        incumbent_case_times=self._scored_incumbent_case_times(),
                    ),
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
        except Exception as error:
            if require_durable:
                raise RuntimeError(f"failed to finalize iteration {result.iteration} checkpoint") from error
            log.debug("run_state: iteration reduce/save failed", exc_info=True)
            return False

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
                    references=PrReferenceNote(self.ic.pr_reference_labels, self.ic.pr_reference_context),
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
            except Exception as error:
                summary_failure = f"{type(error).__name__}: {str(error)[:200]}"
                log.debug("lessons: summarizer step failed", exc_info=True)
                print(f"  [lesson] summarizer step failed ({type(error).__name__}: {error}) — falling back")
            finally:
                self._checkpoint_llm_usage()

        if not has_narrative:
            # No session could describe what was explored, but the gate's block reasons are a real record of what the
            # agent ran into.
            fallback = build_fallback_document(
                SessionObservations(
                    diff_summary=diff_summary,
                    findings=session_sink.get("findings", ""),
                    plan=session_sink.get("plan", ""),
                    progress_log=session_sink.get("progress_log"),
                ),
                SessionEnd(result.session_end_reason, turns=result.turns, summary_failure=summary_failure),
            )
            if fallback and store.write(iteration, fallback) is not None:
                has_narrative = True
                print(f"  [lesson] machine-recorded iter {iteration} from gate findings: {len(fallback)} chars")

        try:
            scope = self._lesson_scope(
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
        except Exception:
            log.debug("lessons: scope append failed", exc_info=True)

        turns, failure = (None, "") if has_narrative else (result.turns, summary_failure)
        store.append_outcome(
            iteration,
            format_outcome_line(
                decision=decision,
                measurement=OutcomeMeasurement(
                    result.wall_ms,
                    self.best_wall_ms,
                    result.snr_db,
                    result.mean_case_speedup,
                    self.best_mean_case_speedup,
                ),
                session=SessionEnd(result.session_end_reason, turns=turns, summary_failure=failure),
            ),
        )

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

        # Step 1: Build (if configured).
        if self.ic.build_command:
            proc = await asyncio.create_subprocess_exec(
                *self.ic.build_command,
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
                BenchRun(self.ic.driver_script, driver_args=[], repeat=self.ic.bench_repeat),
                measurements=KEEP_MEASUREMENT_COUNT,
                timeout_sec=self.ic.bench_timeout_sec,
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
        reg_result = await check_registers(build_dir=self.ic.build_dir)
        vgpr = reg_result.get("vgpr") if reg_result.get("success") else None
        if vgpr:
            print(f"  [registers] VGPR={vgpr}")

        # Step 6: the mean of the independent pristine-relative scores must clear the current best by the candidate's
        # own measurement noise.
        improved = bool(bench_result.get("success")) and passes_keep_threshold(
            measurement_scores,
            best_mean_case_speedup=(self.best_mean_case_speedup or 1.0),
            sigma=sigma,
            sigma_sample_size=sigma_resolution.sample_size,
        )

        if improved and self.ic.kernel_backend == "assembly":
            # A second-stage ASM result must also beat the original caller's aggregate time.
            source_ms = self.ic.pristine_baseline_wall_ms
            improved = source_ms is not None and selected_raw_mean_ms is not None and selected_raw_mean_ms < source_ms

        # Step 7: the numerical contract, which only assembly declares and only it
        # needs. Every other backend was judged by the driver in Step 4 and measured
        # through it since; re-running that verdict here would answer the same
        # question with the same command, while reading a task configuration whose
        # shape the engine has no business knowing. The predicate is shared with the
        # gate description every agent is given, so the two cannot disagree.
        canonical_summary = ""
        if improved and runs_task_suite_acceptance(self.ic.kernel_backend):
            canonical_started = time.time()
            canonical = await accept_candidate(
                self.ic.workspace_dir,
                timeout_cap_sec=self.ic.validate_stage_timeout_sec,
                candidate_label=f"iteration {iteration}",
                kernel_backend=self.ic.kernel_backend,
            )
            # The suite only runs for a candidate the round produced, so it is part of that round's measurement and
            # has to be priced into the next round's admission alongside the validate-and-bench cycle.
            self._observe_measurement(canonical_started)
            canonical_summary = f"\n  Canonical correctness suite: {canonical.detail}"
            if canonical.numerical_evidence is not None:
                bench_result["numerical_validation"] = canonical.numerical_evidence
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

        return IterationResult(
            iteration=iteration,
            duration_sec=duration,
            validation_passed=True,
            validation_summary=report.summary() + canonical_summary,
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

    async def run(
        self,
        agents: LoopAgents = LoopAgents(),
        *,
        callbacks: LoopCallbacks = LoopCallbacks(),
        usage=None,
        workspace_lock_held: bool = False,
    ) -> list[IterationResult]:
        """Run the loop and clean attempt-owned processes on every exit path."""
        try:
            return await self._run_impl(
                agents,
                callbacks=callbacks,
                usage=usage,
                workspace_lock_held=workspace_lock_held,
            )
        finally:
            try:
                from kernelforge.tools.aiter_cache import (
                    cleanup_current_owned_aiter_locks,
                )

                cleanup_current_owned_aiter_locks()
            except (OSError, ValueError):
                log.debug("failed to clean AITER locks on loop exit", exc_info=True)

    async def _run_impl(
        self,
        agents: LoopAgents,
        *,
        callbacks: LoopCallbacks,
        usage=None,
        workspace_lock_held: bool = False,
    ) -> list[IterationResult]:
        """Run while exclusively owning this campaign workspace."""
        if workspace_lock_held:
            return await self._run_locked(agents, callbacks=callbacks, usage=usage)
        store = LoopStateStore(self.ic.workspace_dir)
        with store.workspace_lock():
            return await self._run_locked(agents, callbacks=callbacks, usage=usage)

    def _restore_or_begin_campaign(
        self, *, state_exists, current_ruling_path, agent_fn, orchestration_service, on_best_committed
    ) -> None:
        """Bring the run state up to where the campaign left off, or start a fresh one.

        A fresh campaign refuses a workspace that already holds one, because
        overwriting it would silently discard the earlier campaign's evidence.
        """
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

    def _rule_on_attempt(
        self,
        result,
        attempt: _Attempt,
        frame: _IterationFrame,
        *,
        on_best_committed,
    ) -> _Ruling:
        """Keep or revert this attempt, and say what the verdict left behind.

        Returns the pending KEEP record, whether its checkpoint was finalized,
        and the commit the verdict settled on -- a KEEP commits here, so the
        hash the caller came in with is not the one it leaves with.
        """
        session_sink = frame.session_sink
        best_before, best_mean_case_speedup_before = frame.best_before, frame.best_mean_case_speedup_before
        commit_hash, attempt_diff = attempt.commit_hash, attempt.attempt_diff
        attempt_source, rationale = attempt.attempt_source, attempt.rationale
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
        else:
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
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
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
        return _Ruling(pending_keep, keep_checkpoint_finalized, commit_hash)

    async def _close_iteration(
        self,
        result: IterationResult,
        attempt: _Attempt,
        frame: _IterationFrame,
        *,
        ruling: _Ruling,
        callbacks: LoopCallbacks,
    ) -> None:
        """Settle what this iteration owes the campaign now that it has a verdict.

        The publication, the lesson, the ledger entry, the archive and the
        handoff are all derived views of the same verdict, so they are
        written together and in that order: a resume reads them back as one
        iteration or not at all.
        """
        iteration, session_sink, best_before = frame.iteration, frame.session_sink, frame.best_before
        merge_pair, merge_diff = frame.merge
        pending_keep, keep_checkpoint_finalized, commit_hash = ruling
        # Remote/external work belongs outside the SIGTERM deferral window but still precedes potentially long
        # post-KEEP profiling.
        if keep_checkpoint_finalized and pending_keep is not None:
            self._publish_best_result(
                result,
                plan=session_sink.get("plan", ""),
                best_before=best_before,
                pending=pending_keep,
            )
        if result.kept and callbacks.on_best_ready:
            callbacks.on_best_ready(result)

        if self.experiment:
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
            self._diff_summary(commit_hash) if commit_hash else self._diff_summary_from_diff(attempt.attempt_diff)
        )
        await self._record_lesson(
            iteration=iteration,
            result=result,
            decision=decision_label,
            session_sink=session_sink,
            diff_summary=iteration_diff_summary,
        )

        # Record this iteration into the cross-iteration experience ledger.
        self._record_ledger_iteration(
            result,
            frame,
            iteration_diff_summary=iteration_diff_summary,
            commit_hash=commit_hash,
            attempt_diff=attempt.attempt_diff,
        )

        # Archive the full solution and measurements as a derived view.
        archived_path = self._archive_attempt(result, attempt, frame, ruling=ruling, decision_label=decision_label)

        if not keep_checkpoint_finalized:
            self._publish_best_result(
                result,
                plan=session_sink.get("plan", ""),
                best_before=best_before,
            )
        self._record_iteration_handoff(
            iteration=iteration,
            decision=decision_label,
            optimization_plan_path=attempt.optimization_plan_path,
            session_sink=session_sink,
            archived_path=archived_path,
        )
        self._publish_optimization_history()

        if callbacks.on_iteration:
            callbacks.on_iteration(result)

        # A KEEP makes the prior evidence stale but does not discard its paths.
        if result.kept:
            self._analysis_bundle = None

    async def _source_attempt(
        self,
        frame: _IterationFrame,
        supply: _IterationSupply,
        *,
        agents: LoopAgents,
        digest,
        callbacks: LoopCallbacks,
    ) -> _Attempt | None:
        """Decide what this iteration measures, and produce it.

        A held device, a queued lane candidate and a staged merge each cost
        nothing to source, so they are answered before an Implementer
        session is bought. Returns None when the iteration has nothing to
        measure: either it was closed out here, or the round was refused for
        budget and ``_refused_round`` now says the campaign is over.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        merge_pair, merge_diff = frame.merge
        hazard, queued_lane, fan_out_plan = supply.hazard, supply.queued_lane, supply.fan_out_plan
        agent_fn, orchestration_service, on_iteration = (
            agents.agent_fn,
            agents.orchestration_service,
            callbacks.on_iteration,
        )
        # Only an orchestrated round synthesizes a plan; every other way of
        # sourcing an attempt leaves the iteration without one.
        optimization_plan_path = ""
        optimization_plan_executable = False
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
            self._bind_session_producer(session_sink, queued_lane.agent_profile_id)
            print(f"  [lane {queued_lane.lane_id}] measuring queued candidate")
        elif merge_diff and merge_pair is not None:
            unmeasured_result = None
            commit_hash = ""
            rationale = (
                f"stacked iterations {merge_pair[0].iteration} and {merge_pair[1].iteration}; no Implementer session"
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
            self._bind_session_producer(session_sink, self.run_state.agent_relay.active_profile_id)
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
                    await self._abandon_iteration(
                        IterationResult(
                            iteration=iteration,
                            duration_sec=0.0,
                            validation_passed=False,
                            validation_summary=(f"ORCHESTRATION ERROR: {orchestration_error}"),
                            session_end_reason="orchestration_error",
                        ),
                        frame,
                        decision_label="ORCHESTRATION_ERROR",
                        on_iteration=on_iteration,
                    )
                    return None
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
                return None
            session_sink["session_started"] = True
            history = self._implementer_history(
                digest=digest,
                optimization_plan_path=optimization_plan_path,
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
            except Exception as e:  # noqa: BLE001 - agent backend failure is not enumerable
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

            return await self._collect_session_candidate(
                frame,
                agent_error=agent_error,
                rationale=rationale,
                plan=_RoundPlan(optimization_plan_path, optimization_plan_executable),
                on_iteration=on_iteration,
            )
        else:
            unmeasured_result = None
            commit_hash = ""
            rationale = "no-agent (baseline measurement)"
            attempt_source = ""
            attempt_diff = ""
            reusable_benchmark = None
        return _Attempt(
            unmeasured_result=unmeasured_result,
            commit_hash=commit_hash,
            rationale=rationale,
            attempt_source=attempt_source,
            attempt_diff=attempt_diff,
            reusable_benchmark=reusable_benchmark,
            optimization_plan_path=optimization_plan_path,
            optimization_plan_executable=optimization_plan_executable,
        )

    async def _collect_session_candidate(
        self,
        frame: _IterationFrame,
        *,
        agent_error,
        rationale,
        plan: _RoundPlan,
        on_iteration,
    ) -> _Attempt | None:
        """Collect what the Implementer session left behind, or None if nothing can be.

        A session that tainted protected state or left the workspace
        contended produced a candidate that exists but must not be measured,
        so it comes back already judged. A session that changed nothing
        tracked is closed out here and the iteration is abandoned.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        optimization_plan_path, optimization_plan_executable = plan.path, plan.executable
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
                except (RuntimeError, OSError) as error:
                    restore_errors.append(f"protected snapshot restore failed: {type(error).__name__}: {error}")
            else:
                restore_errors.append("protected snapshot restore callback unavailable")
            try:
                self._git_discard_worktree()
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                restore_errors.append(f"tracked candidate restore failed: {type(error).__name__}: {error}")
            if not restore_errors:
                try:
                    self._validate_driver_integrity(self.run_state)
                except (ValueError, OSError) as error:
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
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
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
                        "LLM API never answered this Session; no candidate was attempted (not an optimization result)"
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
                self._stamp_result_profile(result, session_sink)
                result.agent_rationale = rationale
                result.session_end_reason = session_sink.get("end_reason", "")
                result.turns = session_sink.get("turns")
                await self._abandon_iteration(
                    result,
                    frame,
                    decision_label=decision_label,
                    on_iteration=on_iteration,
                    round_plan=_RoundPlan(optimization_plan_path, optimization_plan_executable),
                )
                return None
            reusable_benchmark = None
            gate_measurement = session_sink.get("benchmark_measurement")
            if session_sink.get("gate_passed") is True and self._can_reuse_insession_benchmark(
                gate_measurement,
                attempt_diff=attempt_diff,
            ):
                reusable_benchmark = gate_measurement
            # Capture the attempt before any discard or keep commit.
            attempt_source = self._read_kernel_source()
        return _Attempt(
            unmeasured_result=unmeasured_result,
            commit_hash="",
            rationale=rationale,
            attempt_source=attempt_source,
            attempt_diff=attempt_diff,
            reusable_benchmark=reusable_benchmark,
            optimization_plan_path=optimization_plan_path,
            optimization_plan_executable=optimization_plan_executable,
        )

    async def _abandon_iteration(
        self,
        result: IterationResult,
        frame: _IterationFrame,
        *,
        decision_label: str,
        on_iteration,
        round_plan: _RoundPlan = _RoundPlan(),
    ) -> None:
        """Close out an iteration that ended before anything could be measured.

        Nothing reached the measurement surface, but the iteration still owes
        the campaign its outcome, its lesson and its handoff: without them a
        resume would re-plan the round this one already spent.
        """
        iteration, session_sink = frame.iteration, frame.session_sink
        optimization_plan_path, optimization_plan_created = round_plan.path, round_plan.executable
        self.results.append(result)
        self._apply_iteration_planning_state(
            optimization_plan_created=optimization_plan_created,
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

    def _implementer_history(self, *, digest, optimization_plan_path) -> str:
        """Assemble the cross-iteration experience the Implementer session reads.

        Each source carries what the others cannot: the candidate archive digest
        is the trajectory table with full diffs and a pointer to every archived
        kernel; the lesson documents are what each recent session explored in
        its own words, including the directions it abandoned, which leave no
        diff behind; the experience ledger is objective toolchain observation
        distilled from machine-verified failure signatures.

        Order is the message: the long-horizon header is the compact memory
        frame the implementer reads first, and the plan instruction sits above
        the evidence because lesson records are evidence, not instructions.
        """
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
            lessons_txt = self.lessons.render_for_prompt(
                current_cases=self._scored_case_ids(),
                kernel_source=self._kernel_source_for_scope(),
            )

        ledger_txt = ""
        if self.ledger:
            # The session narrative lives in the lesson documents, so the ledger contributes only objective
            # toolchain observations once any lesson is available.
            ledger_txt = self.ledger.render_for_prompt(
                include_recent=(not lessons_txt and (bool(lh_header) or not bool(digest)))
            )
        if lh_header:
            history = "\n\n".join(p for p in (lessons_txt, ledger_txt) if p)
            print(f"  [agent] injected long-horizon header: {len(lh_header)} chars (digest reserved for supervisor)")
        else:
            history = "\n\n".join(p for p in (digest, lessons_txt, ledger_txt) if p)
            if digest:
                n_cand = len(self.archive.load_index())
                print(f"  [agent] injected lineage digest: {len(digest)} chars, {n_cand} prior candidates archived")
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
        ceiling_block = self._render_ceiling_advisory()
        if ceiling_block:
            history = f"{ceiling_block}\n\n{history}"
            print(f"  [agent] injected roofline attainment standing: {len(ceiling_block)} chars")
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
        return history

    def _initialize_campaign_runtime(self, usage) -> None:
        """Initialize process-local stores and candidate queues for one session."""
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
        self.handoff_store = HandoffStore(self.ic.workspace_dir)

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

    async def _open_iteration(self, *, iteration, digest, supervisor_fn, analysis_service) -> None:
        """Resolve evidence, review the trajectory and re-scope what this iteration owns.

        Analysis resolves before any Supervisor or planning call, so both read
        the same evidence. A stall triggers the reviewer and the loop ALWAYS
        CONTINUES -- it never self-terminates on one. Untracked files already
        on the tree belong to the operator or to an earlier round, so the
        boundary is re-taken here and this iteration's REVERT cannot delete them.
        """
        supervisor_due = False
        supervisor_reason = ""
        if self.monitor is not None and supervisor_fn is not None:
            supervisor_due, supervisor_reason = self._supervisor_due(iteration)

        # Resolve Analysis before any Supervisor or planning call.
        if analysis_service is not None:
            await self._resolve_analysis_context(
                analysis_service,
                supervisor_due=supervisor_due,
                iteration=iteration,
            )

        # Self-supervision (AVO): when supervised, a stall triggers a reviewer that injects fresh directions and
        # the loop ALWAYS CONTINUES — it never self-terminates on stall.
        await self._run_supervisor_review(
            supervisor_fn,
            iteration=iteration,
            digest=digest,
            supervisor_due=supervisor_due,
            supervisor_reason=supervisor_reason,
        )

        self._update_search_policy(iteration)

        print(
            f"--- Iteration {iteration} "
            f"(best mean case speedup: {self.best_mean_case_speedup:.6f}x, "
            f"remaining: {self._time_remaining() / 60:.0f} min) ---"
            if self.best_mean_case_speedup is not None
            else f"--- Iteration {iteration} ---"
        )
        roofline_progress = self._render_roofline_progress()
        if roofline_progress:
            print(roofline_progress)

        # Re-scope the ownership boundary to this iteration: untracked files already here are the operator's or an
        # earlier round's, and this iteration's REVERT must not delete them.
        snapshot = self._untracked_snapshot()
        if snapshot is not None:
            self._pre_untracked = snapshot

        # Durable per-iteration marker (facts only; detail lives in files).
        self.run_state.iteration = iteration
        self.state_store.append_event(
            make_event(
                "iteration_started",
                iteration,
                best_before_ms=self.best_wall_ms,
                best_before_mean_case_speedup=self.best_mean_case_speedup,
                phase=self.run_state.phase,
            )
        )

    async def _run_locked(
        self,
        agents: LoopAgents,
        *,
        callbacks: LoopCallbacks,
        usage=None,
    ) -> list[IterationResult]:
        """Run the autonomous iteration loop."""
        self._search_actors = agents.search_actors

        _line_buffer_stdout()

        self._initialize_campaign_runtime(usage)

        # Durable, file-backed run state + append-only event log.
        self.state_store = LoopStateStore(self.ic.workspace_dir)
        state_exists = self.state_store.state_path.exists()
        self.run_state = self.state_store.load()
        actors = _with_search_actor(agents, self._session_start_bundle())
        agent_fn, agent_factory, supervisor_fn = actors.agent_fn, actors.agent_factory, actors.supervisor_fn
        analysis_service, orchestration_service = actors.analysis_service, actors.orchestration_service
        current_ruling_path = latest_supervisor_ruling_path(self.ic.workspace_dir)
        self._supervisor_ruling = load_latest_supervisor_ruling(self.ic.workspace_dir) if self.resume else ""

        # Ownership boundary for anything a candidate creates, taken before this loop touches the workspace.
        self._pre_untracked = self._untracked_snapshot()

        self._restore_or_begin_campaign(
            state_exists=state_exists,
            current_ruling_path=current_ruling_path,
            agent_fn=agent_fn,
            orchestration_service=orchestration_service,
            on_best_committed=callbacks.on_best_committed,
        )

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
        self._record_agent_availability()
        self.experiment = self.tracker.create_segment(
            CampaignSegment(self.run_state.campaign_id, next_segment_index, parent_experiment_id),
            ExperimentSubject(
                task_id=Path(self.ic.kernel_file).stem,
                backend=self.ic.backend,
                kernel_backend=self.ic.kernel_backend,
                description=f"Autonomous optimization of {self.ic.kernel_file}",
                target_wall_ms=self.ic.target_wall_ms,
                baseline_wall_ms=self.ic.baseline_wall_ms,
            ),
        )
        self.run_state.last_experiment_id = self.experiment.experiment_id
        self.state_store.append_event(
            make_event(
                "session_started",
                self.run_state.iteration,
            )
        )
        self.state_store.save(self.run_state)

        self._persist_pr_knowledge()
        self._announce_run(supervisor_fn)

        await self._anchor_speedup_reporting()

        # Seed the run state's baseline and, guardedly, resume a prior best from a reused workspace (only when the
        # recorded best commit is still HEAD).
        self._seed_and_hydrate_run_state()
        self._adopt_validated_warm_start()

        self._verify_baseline_anchor()

        # Once the anchor is settled and verified, so the estimate is handed the case set the objective scores and
        # latencies from the campaign's own clock.
        await self._establish_ceiling()

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
                digest = self.archive.render_digest()

            # Ruled on once per iteration, with the other conditions that decide whether this iteration may run at
            # all -- and after them, so a run that was stopping anyway never touches the device.
            stop, hazard = self._stop_condition_met(iteration)
            if stop:
                break

            actors = _with_search_actor(actors, self._iteration_bundle(iteration))
            agent_fn, agent_factory, supervisor_fn = actors.agent_fn, actors.agent_factory, actors.supervisor_fn
            orchestration_service = actors.orchestration_service

            # Price the round before anything is spent on it.
            self._close_round()
            round_lanes = self.ic.lanes
            if not self._lane_queue:
                admitted_lanes = self._admit_next_round(iteration)
                if admitted_lanes is None:
                    break
                round_lanes = admitted_lanes
                self._open_round(iteration, lanes=round_lanes)

            await self._open_iteration(
                iteration=iteration,
                digest=digest,
                supervisor_fn=supervisor_fn,
                analysis_service=analysis_service,
            )

            # Agent proposes modification
            session_sink: dict = {}
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
            merge = self._stage_merge(
                iteration=iteration,
                hazard=hazard,
                fan_out_plan=fan_out_plan,
            )
            # A fan-out round already paid for these candidates, so they are measured before anything new is planned.
            queued_lane = None if hazard or merge.diff else self._take_lane_candidate()
            frame = _IterationFrame(iteration, session_sink, merge)
            attempt = await self._source_attempt(
                frame,
                _IterationSupply(hazard, queued_lane, fan_out_plan),
                agents=agents,
                digest=digest,
                callbacks=callbacks,
            )
            if attempt is None:
                if self._refused_round:
                    break
                continue
            (
                unmeasured_result,
                commit_hash,
                rationale,
                attempt_source,
                attempt_diff,
                reusable_benchmark,
                optimization_plan_path,
                optimization_plan_executable,
            ) = attempt

            # Snapshot the best-so-far BEFORE this iteration is measured, so the archive can record the true delta vs
            # the standard it had to beat (run_one_iteration mutates self.best_wall_ms on an improvement).
            best_before = self.best_wall_ms
            best_mean_case_speedup_before = self.best_mean_case_speedup
            frame = replace(
                frame,
                best_before=best_before,
                best_mean_case_speedup_before=best_mean_case_speedup_before,
            )

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
                except Exception as e:  # noqa: BLE001
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
            self._stamp_result_profile(result, session_sink)
            # Session end reason + turns spent (from the in-session gate / SDK via session_sink) — persisted so a
            # run's end-reason distribution (edit cap / turn cap / converged / …) is analyzable.
            result.session_end_reason = session_sink.get("end_reason", "")
            result.turns = session_sink.get("turns")

            if agent_fn is not None:
                self._apply_iteration_planning_state(
                    optimization_plan_created=optimization_plan_executable,
                )

            ruling = self._rule_on_attempt(result, attempt, frame, on_best_committed=callbacks.on_best_committed)

            await self._close_iteration(result, attempt, frame, ruling=ruling, callbacks=callbacks)

        self._persist_terminal_state()

        self._print_final_report()

        return self.results


def _long_horizon_header(
    state: RunState,
    store: LoopStateStore,
    handoff_store: HandoffStore | None = None,
) -> str:
    """The compact long-horizon header for the Implementer prompt, or \"\"."""
    outcomes = store.recent_results(LONG_HORIZON_OUTCOME_WINDOW)
    return render_long_horizon_header(
        state,
        outcomes,
        include_handoffs=bool(handoff_store and handoff_store.latest()),
    )


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
