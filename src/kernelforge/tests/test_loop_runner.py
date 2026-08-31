# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the autonomous iteration loop and recovery contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kernelforge.llm import process_reaping
from kernelforge.llm.process_reaping import ReapReport
from kernelforge.agent_backends import AgentRunResult
from kernelforge.llm.git import GitError
from kernelforge.loop import analysis_evidence
from kernelforge.loop import fanout
from kernelforge.loop import runner as runner_module
from kernelforge.loop.analysis_refresh_policy import AnalysisRefreshDecision
from kernelforge.loop.archive import CandidateRecord
from kernelforge.loop.device_hazard import MAX_BLOCKED_ITERATIONS
from kernelforge.loop.reporting import _round_budget_lines
from kernelforge.loop.round_budget import (
    ADMISSION_SESSION_SEC,
    FIRST_ROUND_MEASUREMENT_SEC,
    admit_dispatch,
    admit_round,
)
from kernelforge.loop.run_state import (
    ORCHESTRATION_CIRCUIT_CLOSED,
    ORCHESTRATION_CIRCUIT_OPEN,
    PHASE_STALLED,
    SESSION_PAUSED,
    _RECENT_RESULT_CACHE,
    BestRecord,
    LoopStateStore,
    RoundCostState,
    RunState,
    apply_iteration,
    apply_round_cost,
    apply_supervisor_intervention,
    make_event,
)
from kernelforge.loop.merge_candidates import (
    MERGE_ATTEMPT_STALL_THRESHOLD,
    MERGE_PRECEDENCE_STREAK_LIMIT,
)
from kernelforge.loop.runner import (
    IterationConfig,
    IterationLoop,
    IterationResult,
    WindowGain,
)
from kernelforge.loop.scoring import passes_keep_threshold
from kernelforge.loop.search_policy import (
    MARGINAL_GAIN_WINDOW,
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    NO_CHANGES_ESCALATION_THRESHOLD,
    NO_CHANGES_STREAK_WINDOW,
    SEARCH_MODE_DIVERSIFY,
    SEARCH_MODE_EXPLOIT,
)
from kernelforge.loop.supervisor import SupervisionMonitor
from kernelforge.orchestrator.contracts import (
    EvidenceRef,
    OrchestrationRunResult,
    PlanCriticOutcome,
    SpecialistDefinition,
)
from kernelforge.orchestrator.orchestration import (
    OrchestrationAgent,
    OrchestrationService,
)
from kernelforge.orchestrator.analysis_session import (
    AnalysisAttemptLimitError,
)
from kernelforge.orchestrator.plan_critic import PLAN_CRITIC_TIMEOUT_SEC
from kernelforge.orchestrator.specialists import (
    SpecialistAgent,
    SpecialistPool,
)
from kernelforge.tracker import ExperimentTracker


class _NoopEvolver:
    def on_experiment_complete(self, experiment):
        return {}


# The stand-in for "budget is not what this test is about". It has to clear the
# round admission guard, which prices a whole round -- planning, a session worth
# starting, the canonical measurement and the finalize reserve -- so a value near
# the reserve itself would silently turn these tests into admission tests.
_AMPLE_BUDGET_SEC = 12 * 3600.0


def _make_loop(
    tmp_path,
    monkeypatch,
    *,
    supervise_after=5,
    session_count=1,
    resume=False,
    baseline_case_times=None,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 1\n")
    driver = workspace / "driver.py"
    driver.write_text("pass\n")

    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "KernelForge Tests"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    monkeypatch.setattr(runner_module, "force_jit_rebuild", lambda _files: None)

    config = IterationConfig(
        kernel_file=str(kernel),
        driver_script=str(driver),
        baseline_wall_ms=1.0,
        baseline_case_times=({"case": 1.0} if baseline_case_times is None else dict(baseline_case_times)),
        max_time_hours=1.0,
        git_branch="test-loop",
        workspace_dir=str(workspace),
        supervise_after=supervise_after,
        supervise_cooldown=0,
    )
    tracker = ExperimentTracker(tmp_path / "experiments")
    loop = IterationLoop(
        config,
        tracker,
        config=object(),
        evolver=_NoopEvolver(),
        resume=resume,
    )
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(loop.results) < session_count else 0.0,
    )
    return loop, workspace


async def _unused_supervisor(**_kwargs):
    return ""


async def _no_change_agent(_kernel_path, _history, session_sink):
    session_sink["plan"] = "inspect only"
    return "No source change was needed."


def test_staged_candidate_is_detected_and_discarded(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)

    assert "return 2" in loop._working_tree_diff()

    loop._git_discard_worktree()

    assert loop._git("status", "--porcelain", "--untracked-files=no") == ""
    assert "return 1" in kernel.read_text()


def test_working_tree_diff_failure_is_not_treated_as_empty(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)

    def _unreadable_index(*_args, **_kwargs):
        raise GitError(128, ["git", "diff"], "", "fatal: unable to read index")

    monkeypatch.setattr(runner_module, "git", _unreadable_index)

    with pytest.raises(GitError, match="unable to read index"):
        loop._working_tree_diff()


def test_reuses_only_measurement_for_exact_candidate(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop._best_case_times = {"case": 1.0}
    loop.best_mean_case_speedup = 1.0
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    attempt_diff = loop._working_tree_diff()
    measurement = {
        "success": True,
        "measurement_count": 3,
        "measurements": [{}, {}, {}],
        "bench_repeat": loop.ic.bench_repeat,
        "candidate_diff_sha256": hashlib.sha256(attempt_diff.encode()).hexdigest(),
        "driver_sha256": loop._driver_sha256(),
        "baseline_case_times": {"case": 1.0},
        "best_mean_case_speedup": 1.0,
    }

    assert loop._can_reuse_insession_benchmark(
        measurement,
        attempt_diff=attempt_diff,
    )
    measurement["candidate_diff_sha256"] = hashlib.sha256(b"").hexdigest()
    assert not loop._can_reuse_insession_benchmark(
        measurement,
        attempt_diff="",
    )
    measurement["candidate_diff_sha256"] = "0" * 64
    assert not loop._can_reuse_insession_benchmark(
        measurement,
        attempt_diff=attempt_diff,
    )


def test_pending_keep_publication_patch_is_cumulative(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    base_commit = loop._git("rev-parse", "HEAD").strip()
    loop.ic.baseline_wall_ms = 0.9
    loop.ic.publication_baseline_wall_ms = 1.0
    driver = workspace / "driver.py"
    driver.write_text("pass\n# prior kept optimization\n")
    subprocess.run(["git", "add", "driver.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "prior keep"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    loop.ic.campaign_base_commit = base_commit
    loop.run_state = RunState(campaign_id="campaign", session_index=1)
    result = IterationResult(
        iteration=2,
        duration_sec=0.1,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=0.8,
        mean_case_speedup=1.25,
        snr_db=40.0,
        kept=True,
        bench_detail={"median_ms": 0.8},
    )

    loop.run_state.diversification_cycle_completed = True
    pending = loop._build_pending_keep(
        result,
        plan="change kernel",
        best_before=0.9,
        rationale="change kernel",
        kernel_source=kernel.read_text(),
    )

    assert pending["schema_version"] == 2
    assert pending["changed_files"] == ["kernel.py"]
    assert set(pending["publication_changed_files"]) == {"driver.py", "kernel.py"}
    assert "prior kept optimization" in pending["publication_patch"]
    assert "return 2" in pending["publication_patch"]
    assert pending["baseline_wall_ms"] == 1.0
    assert pending["search_control"] == {
        "diversification_cycle_completed": True,
    }


def test_resume_replays_nonkeep_event_ahead_of_state(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    store = LoopStateStore(str(workspace))
    state = RunState(
        campaign_id="campaign",
        iteration=1,
        next_iteration=2,
        baseline_wall_ms=1.0,
    )
    state.cumulative.iterations = 1
    store.save(state)
    store.append_event(
        make_event(
            "iteration_result",
            2,
            decision="REVERT_PERF",
            plan="larger tile",
            wall_ms=1.1,
            best_after_ms=1.0,
            diversification_cycle_completed=True,
        )
    )
    loop.state_store = store
    loop.run_state = state

    planned, _, _, _ = loop._plan_resume_recovery(state, None)
    loop.run_state = planned
    store.save(planned)

    replayed = store.load()
    assert replayed.iteration == 2
    assert replayed.next_iteration == 3
    assert replayed.cumulative.iterations == 2
    assert replayed.cumulative.reverted == 1
    assert replayed.stall.no_improvement_iters == 1
    assert replayed.diversification_cycle_completed is True


def _reverted_candidate(iteration, mean_case_speedup):
    """One correct candidate that was rejected by the KEEP threshold."""
    return IterationResult(
        iteration=iteration,
        duration_sec=0.01,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=1.0,
        mean_case_speedup=mean_case_speedup,
        snr_db=40.0,
        kept=False,
        bench_detail={"case_times": {"case": 1.0}},
    )


def _reduction_loop(tmp_path, monkeypatch, workspace_state=None):
    """A loop wired only for reducing outcomes into durable control state."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = workspace_state if workspace_state is not None else RunState()
    loop.best_wall_ms = 1.0
    loop.best_mean_case_speedup = 1.0
    return loop, workspace


def test_a_near_miss_is_pinned_so_the_retrieval_map_points_at_it(
    tmp_path,
    monkeypatch,
):
    """A gain the KEEP gate rejected is still the best lead the run has.

    ``REVERT_PERF`` covers both a regression and a real gain that landed in the
    band between the incumbent and the accept threshold. The long-horizon prompt
    carries a retrieval map instead of the candidate diffs, and only KEEPs were
    ever pinned, so the most promising rejected work sat in the archive with
    nothing pointing at it and was re-derived from scratch.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    # A real 0.5% gain on every measurement, but spread widely enough that its
    # mean does not clear the t bound on its own scatter.
    scores = [1.00100, 1.00520, 1.00950]

    assert not passes_keep_threshold(scores, best_mean_case_speedup=1.0)

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, min(scores)),
        plan="stage the scales through LDS",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == [1]


def test_a_regression_is_not_pinned(tmp_path, monkeypatch):
    """The other half of the split: a candidate that lost is not a lead.

    The absence of a pin is the whole verdict. A failed candidate is
    deliberately not recorded as a spent direction anywhere else either: it does
    not become a permanent search constraint, and the trajectory already carries
    what happened as fact.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    loop._record_iteration_outcome(
        _reverted_candidate(1, 0.972),
        plan="stage the scales through LDS",
        decision_label="REVERT_PERF",
    )

    assert loop.run_state.pinned_iterations == []


def test_a_run_of_near_misses_cannot_evict_the_best_lineage_pin(
    tmp_path,
    monkeypatch,
):
    """The pin the retrieval map is built around outlives later near-misses.

    Near-misses are pinned into the same list as the KEEP behind the current
    best, and a run produces far more of them than KEEPs, so eviction purely by
    age drops the best lineage after eight later pins. It is only released once
    another KEEP takes its place.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.run_state.best = BestRecord(
        iteration=3,
        mean_case_speedup=1.2,
        commit_hash="kept",
        source="iteration",
    )
    runner_module.pin_iteration(loop.run_state, 3)

    for iteration in range(4, 14):
        loop._record_iteration_outcome(
            _reverted_candidate(iteration, 1.003),
            plan=f"near miss {iteration}",
            decision_label="REVERT_PERF",
        )
    while_best = list(loop.run_state.pinned_iterations)

    loop.run_state.best = BestRecord(
        iteration=14,
        mean_case_speedup=1.4,
        commit_hash="newer",
        source="iteration",
    )
    runner_module.pin_iteration(loop.run_state, 14)

    assert while_best == [3, 7, 8, 9, 10, 11, 12, 13]
    assert loop.run_state.pinned_iterations == [7, 8, 9, 10, 11, 12, 13, 14]


def _empty_diff(iteration):
    """One session that ended without a candidate diff at all."""
    return IterationResult(
        iteration=iteration,
        duration_sec=0.01,
        validation_passed=False,
        validation_summary="NO TRACKED CHANGES: agent produced no candidate diff",
        kept=False,
    )


def test_a_new_search_mode_starts_its_own_empty_diff_streak(tmp_path, monkeypatch):
    """Two empty diffs escalate the search, so the count is per search mode.

    Counting over iterations instead means that once the streak is at the
    threshold every later attempt is escalated away on its first empty diff, on
    one datum -- and the attempt hit hardest is the diversification the previous
    escalation just forced.

    Every outcome here records the same ``plan``, so a reset can only come from
    the mode. That is the pairing this test exists for: the end-to-end case
    proves one mode keeps counting across reworded headlines, and this one proves
    a mode change resets even when the headline does not change.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    headline = "rewrite the reduction with warp shuffles"

    def streak():
        return loop._consecutive_no_changes(loop.state_store.read_events())

    for iteration in (1, 2):
        loop._record_iteration_outcome(
            _empty_diff(iteration),
            plan=headline,
            decision_label="NO_CHANGES",
        )
    exhausted_streak = streak()
    loop.run_state.search_mode = SEARCH_MODE_DIVERSIFY
    loop._record_iteration_outcome(
        _empty_diff(3),
        plan=headline,
        decision_label="NO_CHANGES",
    )
    after_first_attempt = streak()
    loop._record_iteration_outcome(
        _empty_diff(4),
        plan=headline,
        decision_label="NO_CHANGES",
    )

    assert exhausted_streak == NO_CHANGES_ESCALATION_THRESHOLD
    assert after_first_attempt == 1
    assert streak() == NO_CHANGES_ESCALATION_THRESHOLD


def _stackable_workspace(loop, workspace, *, candidates=2):
    """Archive rejected gains whose diffs touch well-separated parts of the kernel.

    Each candidate wins one case and loses the other, alternating, so any two of
    opposite parity are mutually complementary and the field of selectable pairs
    grows as the square of the count -- which is what a streak draws on. The
    lines are further apart than a diff hunk carries context, so any two of the
    patches apply over each other.
    """
    edits = (
        (0, "line_0 = 0", "line_0 = 100", "prefill"),
        (11, "line_11 = 11", "line_11 = 111", "decode"),
        (22, "line_22 = 22", "line_22 = 122", "prefill"),
        (33, "line_33 = 33", "line_33 = 133", "decode"),
    )[:candidates]
    loop.archive = runner_module.CandidateArchive(str(workspace), loop.ic.kernel_file)
    kernel = workspace / "kernel.py"
    width = max(line for line, *_rest in edits) + 1
    kernel.write_text("\n".join(f"line_{n} = {n}" for n in range(width)) + "\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "wide kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical = kernel.read_text()

    def _diff(old: str, new: str) -> str:
        kernel.write_text(canonical.replace(old, new))
        diff = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True).stdout
        kernel.write_text(canonical)
        return diff

    for iteration, (_line, old, new, case) in enumerate(edits, start=1):
        runs = [
            {
                "prefill": (0.9 if case == "prefill" else 1.0) * jitter,
                "decode": (0.9 if case == "decode" else 1.0) * jitter,
            }
            for jitter in (1.0, 1.0005, 0.9995)
        ]
        loop.archive.record(
            CandidateRecord(
                iteration=iteration,
                decision="REVERT_PERF",
                validation_passed=True,
                mean_case_speedup=1.002 + iteration / 1000.0,
                bench_detail={
                    "case_times": {
                        "prefill": 0.9 if case == "prefill" else 1.0,
                        "decode": 0.9 if case == "decode" else 1.0,
                    },
                    "measurements": [{"success": True, "case_times": run, "unscored_cases": []} for run in runs],
                },
                change_diff=_diff(old, new),
                plan=f"tune {case}",
            )
        )
    return canonical


def _lane_agent_factory(edits):
    """An agent that writes the edit its lane was assigned, in that lane's copy."""

    def factory(lane_dir, _serialized_driver):
        async def agent(_kernel_file, prompt):
            path = Path(lane_dir) / "kernel.py"
            for plan, (old, new) in edits.items():
                if plan in prompt:
                    path.write_text(path.read_text().replace(old, new))
                    return

        return agent

    return factory


def _recording_lane_factory(seen):
    """A lane agent that records everything the round handed it."""

    def factory(lane_dir, serialized_driver):
        async def agent(kernel_file, prompt):
            seen.append(
                {
                    "lane_dir": lane_dir,
                    "serialized_driver": serialized_driver,
                    "kernel_file": kernel_file,
                    "prompt": prompt,
                }
            )
            path = Path(lane_dir) / "kernel.py"
            path.write_text(path.read_text().replace("return 1", "return 2"))

        return agent

    return factory


async def test_a_lane_is_told_to_run_the_serialized_driver(tmp_path, monkeypatch):
    """A lane that ran the driver itself would time against its own siblings.

    The lock lives in the wrapper, so the round has to hand it to the factory --
    which installs it as the command the lane's own instructions name -- and the
    plan has to explain the wait it causes, which those instructions cannot know
    about. The plan the archive records stays the direction the lane was
    assigned, without either wrapped around it.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    seen: list[dict] = []

    await loop._fill_lane_queue(
        agent_factory=_recording_lane_factory(seen),
        lane_plans=["tune the prefill epilogue"],
    )

    assert len(seen) == 1
    wrapper = str(Path(seen[0]["lane_dir"]) / fanout.SERIALIZED_DRIVER_NAME)
    assert seen[0]["serialized_driver"] == wrapper
    assert wrapper in seen[0]["prompt"]
    assert "not a hang" in seen[0]["prompt"]
    assert "tune the prefill epilogue" in seen[0]["prompt"]
    assert [item.plan for item in loop._lane_queue] == ["tune the prefill epilogue"]


async def test_a_fan_out_round_queues_one_candidate_per_lane(tmp_path, monkeypatch):
    """Lanes are spent one per iteration, so each is measured on its own."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("\n".join(f"line_{n} = {n}" for n in range(12)) + "\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "wide"], cwd=workspace, check=True, capture_output=True)

    await loop._fill_lane_queue(
        agent_factory=_lane_agent_factory(
            {
                "tune prefill": ("line_0 = 0", "line_0 = 100"),
                "tune decode": ("line_11 = 11", "line_11 = 111"),
            }
        ),
        lane_plans=["tune prefill", "tune decode"],
    )

    assert [item.plan for item in loop._lane_queue] == ["tune prefill", "tune decode"]

    first = loop._take_lane_candidate()

    assert first is not None and first.plan == "tune prefill"
    assert "line_0 = 100" in kernel.read_text()
    assert len(loop._lane_queue) == 1


def _published_plan(workspace):
    """Stand in for the planning chain with one durable plan on disk."""

    async def _plan(**_kwargs):
        plan_path = workspace / "forge_experiments" / "orchestration" / "iter_001" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text("# Optimization plan\nVectorize loads.\n")
        return plan_path, ""

    return _plan


def test_a_single_lane_runs_the_ordinary_session(tmp_path, monkeypatch):
    """--lanes 1 must behave exactly as it did before lanes existed."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    fanned_out: list[int] = []

    async def _fan_out(*, iteration, **_kwargs):
        fanned_out.append(iteration)

    monkeypatch.setattr(loop, "_fan_out_round", _fan_out)
    monkeypatch.setattr(loop, "_run_orchestration", _published_plan(workspace))

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=object(),
            agent_factory=_lane_agent_factory({}),
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert loop.ic.lanes == 1
    assert fanned_out == []
    assert decisions == ["NO_CHANGES"]


def test_more_than_one_lane_fans_the_round_out(tmp_path, monkeypatch):
    """Guards the single-lane assertion above from passing for the wrong reason."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    fanned_out: list[int] = []

    async def _fan_out(*, iteration, **_kwargs):
        fanned_out.append(iteration)

    monkeypatch.setattr(loop, "_fan_out_round", _fan_out)
    monkeypatch.setattr(loop, "_run_orchestration", _published_plan(workspace))

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=object(),
            agent_factory=_lane_agent_factory({}),
        )
    )

    assert fanned_out == [1]


def _counting_plan(loop, workspace, rounds, *, plans=("a", "b"), unavailable=False):
    """Stand in for planning, recording every round an iteration is charged for."""

    async def _plan(*, iteration, lanes=1, **_kwargs):
        rounds.append(lanes)
        if unavailable:
            return None, "OrchestrationInfrastructureError: backend unreachable"
        loop._last_lane_plans = list(plans)
        plan_path = workspace / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(f"# Optimization plan\n{plans[0]}\n")
        return plan_path, ""

    return _plan


def _fan_out_iteration(tmp_path, monkeypatch, rounds, **plan_kwargs):
    """One real fan-out iteration, run for what its fallback path costs.

    Every way a fan-out round ends with an empty queue hands the iteration to
    the ordinary single-session path, which plans for itself. Planning is
    dispatch plus every specialist plus synthesis -- the most expensive thing an
    iteration buys -- so what these tests read off ``rounds`` is how many times
    one iteration bought it.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    monkeypatch.setattr(loop, "_run_orchestration", _counting_plan(loop, workspace, rounds, **plan_kwargs))

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=object(),
            agent_factory=_lane_agent_factory({}),
        )
    )
    return [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]


def test_a_planning_outage_is_not_paid_for_twice(tmp_path, monkeypatch):
    """Retrying an outage inside the iteration it stopped cannot clear it.

    The backend that just refused the fan-out round is the backend the fallback
    path would ask, moments later, for the same round -- so the iteration pays
    twice and trips the orchestration circuit breaker twice for one outage.
    """
    rounds: list[int] = []

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds, unavailable=True)

    assert rounds == [2]
    assert decisions == ["ORCHESTRATION_ERROR"]


def test_a_round_that_could_only_plan_one_lane_spends_that_plan(tmp_path, monkeypatch):
    """One plan is not too few to run; it is exactly what one session needs."""
    rounds: list[int] = []

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds, plans=("a",))

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]


def test_lanes_that_produced_nothing_do_not_buy_the_round_again(tmp_path, monkeypatch):
    """This round has already paid for planning and for every lane session."""
    rounds: list[int] = []

    async def _no_candidates(**_kwargs):
        return []

    monkeypatch.setattr(runner_module, "run_lanes", _no_candidates)

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds)

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]


def test_candidates_refused_at_intake_do_not_buy_the_round_again(tmp_path, monkeypatch, capsys):
    """The one path where every lane produced something and none of it counts.

    A candidate refused at the boundary has already cost its own session, and
    the round it came from has already cost the planning. Both are spent before
    the refusal is known, so charging the same iteration for a second round
    would answer a candidate that must not be measured by buying another one.
    """
    rounds: list[int] = []
    tampered = "--- a/driver.py\n+++ b/driver.py\n@@ -1 +1 @@\n-pass\n+print('tampered')\n"

    async def _rejected_candidates(**_kwargs):
        return [
            runner_module.LaneResult(lane_id="1", plan="a", diff=tampered),
            runner_module.LaneResult(lane_id="2", plan="b", diff=tampered),
        ]

    monkeypatch.setattr(runner_module, "run_lanes", _rejected_candidates)

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds)

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]
    # Pin the path: both candidates were refused, not merely unmeasurable for
    # some other reason that would reach the same iteration count.
    output = capsys.readouterr().out
    assert output.count("candidate rejected") == 2
    assert "driver.py" in output


def test_a_lane_infrastructure_failure_does_not_buy_the_round_again(tmp_path, monkeypatch):
    """The plans survive a workspace that could not be copied; only lanes fail."""
    rounds: list[int] = []

    async def _no_room(**_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner_module, "run_lanes", _no_room)

    decisions = _fan_out_iteration(tmp_path, monkeypatch, rounds)

    assert rounds == [2]
    assert decisions == ["NO_CHANGES"]


def _seed_round_costs(workspace, *, planning_sec, lanes=3, rounds=2):
    """A campaign resumed with a durable record of what its rounds have cost.

    Observed cost only exists on a campaign that has run, so these tests resume
    one -- which is also the shape the guard matters most in, since the rounds
    that were killed in production were the last ones of a long run.
    """
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = RunState(
        campaign_id="campaign",
        baseline_case_times={"case": 1.0},
        head_commit=head,
    )
    for iteration in range(1, rounds + 1):
        apply_round_cost(
            state,
            iteration=iteration,
            lanes=lanes,
            planning_sec=planning_sec,
            total_sec=planning_sec + 2400.0,
            campaign_sec=iteration * (planning_sec + 2400.0),
        )
    LoopStateStore(str(workspace)).save(state)


def _round_admission_loop(
    tmp_path,
    monkeypatch,
    rounds,
    *,
    remaining_sec,
    planning_sec=2400.0,
    lanes=3,
):
    """One campaign whose budget decides how wide -- or whether -- it plans."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    loop.ic = replace(loop.ic, lanes=lanes)
    _seed_round_costs(workspace, planning_sec=planning_sec)
    monkeypatch.setattr(
        loop,
        "_run_orchestration",
        _counting_plan(loop, workspace, rounds),
    )
    # One round is all these tests need to see admitted, narrowed or refused;
    # the second is starved so the campaign ends on the older reserve guard
    # rather than on the one under test.
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: remaining_sec if not loop.results else 0.0,
    )

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=object(),
            agent_factory=_lane_agent_factory({}),
        )
    )
    return loop, workspace


def test_a_round_the_budget_can_finish_is_admitted_unchanged(tmp_path, monkeypatch):
    rounds: list[int] = []

    loop, _ = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=12 * 3600.0,
    )

    assert rounds[0] == 3
    assert loop.termination_reason != "round_budget_exhausted"


def test_a_round_the_budget_cannot_finish_narrows_instead_of_starting(tmp_path, monkeypatch):
    """A narrower round is worth more than a wide one that is killed halfway."""
    rounds: list[int] = []
    # A minute more than the single-lane round -- whose planning bound is the
    # seeded three-lane round less the two plan reads the Critic is spared --
    # and well short of what two lanes would cost.
    remaining = 2400.0 - 2 * PLAN_CRITIC_TIMEOUT_SEC + ADMISSION_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC + 60.0

    loop, _ = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=remaining,
    )

    assert rounds[0] == 1
    assert loop.termination_reason != "round_budget_exhausted"


# One minute short of the cheapest round this campaign could plan.
_UNAFFORDABLE_SEC = 2400.0 - 2 * PLAN_CRITIC_TIMEOUT_SEC + ADMISSION_SESSION_SEC + FIRST_ROUND_MEASUREMENT_SEC - 60.0


def test_a_round_no_width_can_pay_for_ends_the_campaign(tmp_path, monkeypatch):
    rounds: list[int] = []

    loop, workspace = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=_UNAFFORDABLE_SEC,
    )

    assert rounds == []
    assert loop.results == []
    assert loop.termination_reason == "round_budget_exhausted"
    assert LoopStateStore(str(workspace)).load().termination_reason == ("round_budget_exhausted")


def test_a_refused_round_is_reported_as_a_refusal_not_as_an_empty_round(tmp_path, monkeypatch, capsys):
    rounds: list[int] = []

    loop, _ = _round_admission_loop(
        tmp_path,
        monkeypatch,
        rounds,
        remaining_sec=_UNAFFORDABLE_SEC,
    )

    assert "ROUND REFUSED FOR BUDGET" in capsys.readouterr().out
    assert loop._round_budget_summary()["refused"]


# What an earlier session of the campaign already banked: 45 minutes of
# planning inside 50 minutes of wall-clock. The session under test then runs
# for seconds, which is the whole point -- the numerator outlives the process,
# the process clock does not.
_BANKED_PLANNING_SEC = 45.0 * 60.0
_BANKED_CAMPAIGN_SEC = 50.0 * 60.0


def test_a_resumed_campaign_reports_a_planning_share_within_its_definition(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The reviewer's reproduction, in the case the feature was built for.

    ``round_costs.planning_total_sec`` is campaign-cumulative and survives
    across sessions; this process's wall-clock does not. Divided one by the
    other, a resumed session running minutes against 45 cumulative minutes of
    planning published a share of several hundred percent -- in the operator
    summary and in ``optimization_report.md``'s ``Round Budget`` section.

    The share must be a share: bounded by 100, and equal to the division of the
    two numbers published beside it, both of which measure the campaign.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    LoopStateStore(str(workspace)).save(
        RunState(
            campaign_id="campaign",
            baseline_case_times={"case": 1.0},
            head_commit=head,
            round_costs=RoundCostState(
                rounds=3,
                planning_total_sec=_BANKED_PLANNING_SEC,
                total_sec=_BANKED_PLANNING_SEC + 600.0,
                campaign_sec=_BANKED_CAMPAIGN_SEC,
            ),
        )
    )
    # One iteration, then out -- so this session's own clock stays far below
    # the planning it inherited.
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: 0.0 if loop.results else 12 * 3600.0,
    )

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    summary = loop._round_budget_summary()
    share = summary["planning_share_pct"]
    # The banked planning is still there, and the campaign clock now covers it
    # rather than being replaced by this session's few seconds.
    assert summary["planning_total_sec"] == pytest.approx(_BANKED_PLANNING_SEC)
    assert summary["campaign_sec"] >= _BANKED_CAMPAIGN_SEC
    assert 0 < share <= 100.0
    assert share == pytest.approx(
        100.0 * summary["planning_total_sec"] / summary["campaign_sec"],
        abs=0.05,
    )

    # The operator summary -- the line that printed "450% of the run" -- and
    # the published report both say it about the campaign now, and both stay
    # inside 100.
    printed = next(
        line for line in capsys.readouterr().out.splitlines() if "Rounds planned across the campaign" in line
    )
    matched = re.search(r"\((\d+(?:\.\d+)?)% of campaign wall-clock\)", printed)
    assert matched, printed
    assert 0 < float(matched.group(1)) <= 100.0
    report = "\n".join(_round_budget_lines(summary))
    assert f"- Planning share of campaign wall-clock: {share:.0f}%" in report
    assert f"- Campaign wall-clock: {summary['campaign_sec'] / 60:.1f} min" in report

    # And the campaign clock is durable, so the NEXT session inherits a span
    # that still covers the planning inside it rather than starting over.
    reloaded = LoopStateStore(str(workspace)).load().round_costs
    assert reloaded.campaign_sec >= reloaded.planning_total_sec
    assert 0 < reloaded.planning_share_pct() <= 100.0


def test_a_finished_round_records_what_its_planning_cost(tmp_path, monkeypatch):
    """The first round has no history to price itself from; it makes some."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    rounds: list[int] = []
    monkeypatch.setattr(
        loop,
        "_run_orchestration",
        _counting_plan(loop, workspace, rounds),
    )

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=object(),
            agent_factory=_lane_agent_factory({}),
        )
    )

    costs = LoopStateStore(str(workspace)).load().round_costs
    assert rounds == [2]
    assert costs.rounds == 1
    assert costs.recent[0].lanes == 2
    assert costs.recent[0].planning_sec > 0
    assert costs.recent[0].total_sec >= costs.recent[0].planning_sec


def test_an_iteration_that_did_not_plan_records_no_round_cost(tmp_path, monkeypatch):
    """A campaign with no orchestration never learns that planning is free."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    assert loop.results
    assert LoopStateStore(str(workspace)).load().round_costs.rounds == 0


def _lane_plans_available(loop, plans):
    """Stand in for planning, which is not what these tests are about."""

    async def _plan(**_kwargs):
        loop._last_lane_plans = list(plans)
        return "forge_experiments/plan.md", ""

    return _plan


async def test_a_lane_infrastructure_failure_falls_back_to_one_session(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A single iteration's failure must never kill a multi-hour run."""
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    async def _no_room(**_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(loop, "_run_orchestration", _lane_plans_available(loop, ["a", "b"]))
    monkeypatch.setattr(runner_module, "run_lanes", _no_room)

    await loop._fan_out_round(
        iteration=3,
        orchestration_service=None,
        agent_factory=None,
    )

    assert loop._lane_queue == []
    assert "No space left on device" in capsys.readouterr().out


async def test_a_programming_error_in_a_fan_out_round_is_not_swallowed(
    tmp_path,
    monkeypatch,
):
    """Falling back on a lane bug would hide it behind a slower iteration."""
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    async def _wrong_call(**_kwargs):
        raise TypeError("session() got an unexpected keyword argument")

    monkeypatch.setattr(loop, "_run_orchestration", _lane_plans_available(loop, ["a", "b"]))
    monkeypatch.setattr(runner_module, "run_lanes", _wrong_call)

    with pytest.raises(TypeError):
        await loop._fan_out_round(
            iteration=3,
            orchestration_service=None,
            agent_factory=None,
        )


async def test_lane_copies_are_made_beside_the_workspace(tmp_path, monkeypatch):
    """/tmp is usually a smaller filesystem than the one the workspace lives on."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    seen: dict = {}

    async def _capture(*, workspace_dir, lanes, session, parent_dir, driver):
        seen["workspace_dir"] = workspace_dir
        seen["parent_dir"] = parent_dir
        seen["driver"] = driver
        return []

    monkeypatch.setattr(runner_module, "run_lanes", _capture)

    await loop._fill_lane_queue(agent_factory=None, lane_plans=["a", "b"])

    assert seen["workspace_dir"] == str(workspace)
    assert seen["parent_dir"] == str(Path(workspace).resolve().parent)
    assert seen["driver"] == "driver.py"


async def test_a_lane_that_wrote_nothing_is_never_queued(tmp_path, monkeypatch):
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)

    def factory(_lane_dir, _serialized_driver):
        async def agent(_kernel_file, _prompt):
            return None

        return agent

    await loop._fill_lane_queue(agent_factory=factory, lane_plans=["a", "b"])

    assert loop._lane_queue == []
    assert loop._take_lane_candidate() is None


def _diff_of(workspace: Path, edit) -> str:
    """A patch for one edit, taken back off the tree once it is captured."""
    restore = {path: path.read_text() for path in workspace.rglob("*.py") if ".git" not in path.parts}
    edit()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "-M", "--", "."],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    subprocess.run(
        ["git", "restore", "--source=HEAD", "--staged", "--worktree", "--", "."],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    for path, text in restore.items():
        path.write_text(text)
    return diff


def test_a_lane_candidate_that_edits_the_driver_never_reaches_the_tree(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The driver is the measurement boundary; a lane session has no gate at all.

    Lanes run with the in-session gate off, so no protected-path hook is
    installed, and a lane's diff carries every tracked modification it made.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    driver = workspace / "driver.py"
    canonical = driver.read_text()
    diff = _diff_of(
        workspace,
        lambda: driver.write_text("import time\nprint('bypass')\n"),
    )
    loop._lane_queue = [runner_module.LaneResult(lane_id="1", plan="tamper", diff=diff)]

    assert loop._take_lane_candidate() is None
    assert loop._lane_queue == []
    assert driver.read_text() == canonical
    output = capsys.readouterr().out
    assert "rejected" in output and "driver.py" in output


def test_a_lane_candidate_that_renames_the_driver_away_is_rejected(
    tmp_path,
    monkeypatch,
):
    """Moving the driver aside leaves only the new path in git's numstat."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    driver = workspace / "driver.py"

    def _rename() -> None:
        subprocess.run(
            ["git", "mv", "driver.py", "harmless.py"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )

    diff = _diff_of(workspace, _rename)
    loop._lane_queue = [runner_module.LaneResult(lane_id="1", plan="move it aside", diff=diff)]

    assert loop._take_lane_candidate() is None
    assert driver.is_file()
    assert not (workspace / "harmless.py").exists()


def test_a_lane_candidate_is_rejected_when_the_driver_stops_being_canonical(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Defence in depth: a bypass the protected-path rule missed still cannot
    reach a measurement, and the tree is returned to canonical before the next
    candidate inherits it."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    driver = workspace / "driver.py"
    loop.ic = replace(
        loop.ic,
        canonical_driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
    )
    diff = _diff_of(workspace, lambda: driver.write_text("print('bypass')\n"))
    loop._lane_queue = [runner_module.LaneResult(lane_id="1", plan="tamper", diff=diff)]
    monkeypatch.setattr(runner_module, "is_protected_path", lambda *_args, **_kwargs: False)

    assert loop._take_lane_candidate() is None
    assert loop._validate_driver_integrity(loop.run_state)
    assert "driver integrity check failed" in capsys.readouterr().out


def test_a_tainted_workspace_driver_stops_the_lane_path(tmp_path, monkeypatch):
    """A driver that is not canonical once the candidate is gone measures nothing."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, canonical_driver_sha256=hashlib.sha256(b"other").hexdigest())
    kernel = workspace / "kernel.py"
    diff = _diff_of(workspace, lambda: kernel.write_text("def kernel():\n    return 2\n"))
    loop._lane_queue = [runner_module.LaneResult(lane_id="1", plan="tune it", diff=diff)]

    with pytest.raises(ValueError, match="driver integrity"):
        loop._take_lane_candidate()


async def test_a_queued_candidate_that_no_longer_applies_is_dropped(
    tmp_path,
    monkeypatch,
):
    """Its tree moved underneath it; re-deriving it is the Implementer's job."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._lane_queue = [
        runner_module.LaneResult(
            lane_id="1",
            plan="stale",
            diff="--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-absent\n+edited\n",
        )
    ]

    assert loop._take_lane_candidate() is None
    assert loop._lane_queue == []


def test_a_queued_candidate_survives_a_keep_that_only_moved_its_context(
    tmp_path,
    monkeypatch,
):
    """A sibling's KEEP must not discard a session over a textual near-miss.

    A round's lanes are partitioned so that no two edit the same code, but a
    hunk is located by the lines around it, so a KEEP three lines away moves
    the context out from under a candidate that changed nothing it touched.
    The diff names the blobs it was written against and the lane copies share
    the canonical object store, so it is merged against them rather than being
    dropped for a mismatch that is not a disagreement.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    a = 1\n    b = 2\n    c = 3\n    y = 4\n    d = 5\n    e = 6\n    return y\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "wide kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    candidate = _diff_of(
        workspace,
        lambda: kernel.write_text(kernel.read_text().replace("y = 4", "y = 44")),
    )
    kernel.write_text(kernel.read_text().replace("a = 1", "a = 11"))
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "the sibling lane's KEEP"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    loop._lane_queue = [runner_module.LaneResult(lane_id="2", plan="tune y", diff=candidate)]

    taken = loop._take_lane_candidate()

    assert taken is not None and taken.lane_id == "2"
    assert "y = 44" in kernel.read_text()
    assert "a = 11" in kernel.read_text()


def test_a_queued_candidate_that_edits_the_same_lines_is_still_dropped(
    tmp_path,
    monkeypatch,
):
    """Merging against the recorded blobs must not become a way to guess.

    Two edits to the same line are a disagreement, not a moved context, and
    resolving one would measure a tree no plan describes.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    y = 4\n    return y\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "one line to fight over"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    candidate = _diff_of(
        workspace,
        lambda: kernel.write_text(kernel.read_text().replace("y = 4", "y = 44")),
    )
    kernel.write_text(kernel.read_text().replace("y = 4", "y = 55"))
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "a KEEP on the same line"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    loop._lane_queue = [runner_module.LaneResult(lane_id="2", plan="tune y", diff=candidate)]

    assert loop._take_lane_candidate() is None
    assert kernel.read_text() == "def kernel():\n    y = 55\n    return y\n"


def test_a_stalled_run_stacks_two_rejected_gains(tmp_path, monkeypatch):
    """The cheapest thing to try once single patches stop clearing the gate.

    Neither candidate passed alone, they win on different cases, and stacking
    them spends a measurement but no Implementer session.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    canonical = _stackable_workspace(loop, workspace)

    pair = loop._select_merge_attempt()

    assert pair is not None
    assert {item.iteration for item in pair} == {1, 2}

    staged, obstacle = loop._stage_merge_attempt(pair)

    assert staged
    assert obstacle == ""
    assert (workspace / "kernel.py").read_text() != canonical
    assert "line_0 = 100" in (workspace / "kernel.py").read_text()
    assert "line_11 = 111" in (workspace / "kernel.py").read_text()


def test_a_stack_does_not_take_an_iteration_that_is_holding_a_plan(tmp_path, monkeypatch):
    """The round's plan has only one consumer, and a stack is not it.

    A fan-out round can come back with an empty queue while still holding the
    plan it bought -- one lane plan, a dispatch the budget refused, a lane
    failure -- and the queue being empty is exactly the condition that lets a
    stack take the iteration. A stacked iteration records a result, which is
    what stops the next process from recovering the round, so planning
    (dispatch, every specialist, synthesis) would be paid for and thrown away.
    The stall that selected the pair is untouched by spending the plan, so the
    attempt is deferred, not lost.
    """
    cases = {"prefill": 1.0, "decode": 1.0}
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True, baseline_case_times=cases)
    loop.ic = replace(loop.ic, lanes=2)
    _stackable_workspace(loop, workspace)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    state = RunState(
        campaign_id="campaign",
        baseline_case_times=dict(cases),
        best_case_times=dict(cases),
        head_commit=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    )
    state.stall.unresolved_stall_iters = 2
    LoopStateStore(str(workspace)).save(state)
    rounds: list[int] = []
    monkeypatch.setattr(loop, "_run_orchestration", _counting_plan(loop, workspace, rounds))

    async def _no_candidates(**_kwargs):
        return []

    monkeypatch.setattr(runner_module, "run_lanes", _no_candidates)

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=object(),
            agent_factory=_lane_agent_factory({}),
        )
    )

    events = LoopStateStore(str(workspace)).read_events()

    assert [e["type"] for e in events if e["type"].startswith("merge_attempt")] == []
    # One round planned, and the iteration spent it rather than buying another.
    assert rounds == [2]
    assert [e.get("decision") for e in events if e["type"] == "iteration_result"] == ["NO_CHANGES"]
    # Two things at once: the pair and the stall the branch needs were both in
    # place -- so the assertions above are about precedence, not about a stack
    # that could never have formed -- and the attempt is deferred, not retired.
    assert loop._select_merge_attempt() is not None


def _three_candidate_workspace(loop, workspace):
    """Archive three rejected gains: the best pair clashes, the runner-up does not.

    Iterations 1 and 2 rewrite the same line to different values, so they cover
    the most cases between them and cannot both be applied. Iteration 3 wins one
    of the same cases as 2 from the other end of the file, so (1, 3) is the pair
    left once (1, 2) is out and it stages cleanly.
    """
    loop.archive = runner_module.CandidateArchive(str(workspace), loop.ic.kernel_file)
    kernel = workspace / "kernel.py"
    kernel.write_text("\n".join(f"line_{n} = {n}" for n in range(12)) + "\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "wide kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical = kernel.read_text()

    def _diff(old: str, new: str) -> str:
        kernel.write_text(canonical.replace(old, new))
        diff = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True).stdout
        kernel.write_text(canonical)
        return diff

    plan = (
        (1, "line_0 = 0", "line_0 = 100", {"prefill"}),
        (2, "line_0 = 0", "line_0 = 200", {"decode", "mixed"}),
        (3, "line_11 = 11", "line_11 = 111", {"decode"}),
    )
    for iteration, old, new, wins in plan:
        times = {case: 0.9 if case in wins else 1.0 for case in ("prefill", "decode", "mixed")}
        loop.archive.record(
            CandidateRecord(
                iteration=iteration,
                decision="REVERT_PERF",
                validation_passed=True,
                mean_case_speedup=1.002 + iteration / 1000.0,
                bench_detail={
                    "case_times": times,
                    "measurements": [
                        {
                            "success": True,
                            "case_times": {case: value * jitter for case, value in times.items()},
                            "unscored_cases": [],
                        }
                        for jitter in (1.0, 1.0005, 0.9995)
                    ],
                },
                change_diff=_diff(old, new),
                plan=f"tune {iteration}",
            )
        )
    return canonical


def test_a_pair_that_would_not_stage_is_not_selected_again(tmp_path, monkeypatch):
    """A textual clash between two archived diffs is the same clash next stall.

    The selector returns the pair covering the most cases, so a decline the run
    does not remember wins every later selection, fails identically, and blocks
    the runner-up that would have staged for the rest of the campaign. Nothing
    in the archive records it: a pair that never reached a measurement is never
    archived.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _three_candidate_workspace(loop, workspace)

    clashing = loop._select_merge_attempt()

    assert clashing is not None
    assert {item.iteration for item in clashing} == {1, 2}

    staged, obstacle = loop._stage_merge_attempt(clashing)

    assert staged == ""
    assert obstacle

    loop._decline_merge_attempt(4, clashing, obstacle)
    runner_up = loop._select_merge_attempt()

    assert runner_up is not None
    assert {item.iteration for item in runner_up} == {1, 3}
    assert loop._stage_merge_attempt(runner_up)[0]


def test_a_tree_that_carried_work_does_not_retire_the_pair(tmp_path, monkeypatch):
    """The one obstacle the pair is not responsible for and next iteration clears."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _three_candidate_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None

    loop._decline_merge_attempt(4, pair, loop.TREE_ALREADY_DIRTY_OBSTACLE)

    again = loop._select_merge_attempt()

    assert again is not None
    assert {item.iteration for item in again} == {item.iteration for item in pair}


def test_a_streak_refusal_does_not_retire_the_pair(tmp_path, monkeypatch):
    """The other obstacle that is a fact about the iteration, not about the pair.

    A refusal is ruled on before ``_stage_merge_attempt`` runs, so the pair's
    diffs are never read, let alone applied to each other -- there is no verdict
    on them to remember. Remembering one anyway turns the streak limit from a
    deferral into a drop, and costs the campaign a measurement that nothing was
    ever wrong with.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0, "mixed": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _three_candidate_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None
    loop._merge_precedence_streak = MERGE_PRECEDENCE_STREAK_LIMIT
    refusal = loop._merge_attempt_refusal()
    assert refusal

    loop._decline_merge_attempt(4, pair, refusal, about_the_iteration=True)

    # Reported, because a selected pair that reached no measurement is not the
    # same event as no pair at all -- and still selectable, because the report
    # was about the iteration.
    assert [
        (item["obstacle"], item["first_iteration"], item["second_iteration"])
        for item in LoopStateStore(str(workspace)).read_events()
        if item.get("type") == "merge_attempt_declined"
    ] == [(refusal, pair[0].iteration, pair[1].iteration)]
    again = loop._select_merge_attempt()

    assert again is not None
    assert {item.iteration for item in again} == {item.iteration for item in pair}


def test_an_archive_that_lost_a_diff_says_so_and_does_not_claim_a_conflict(tmp_path, monkeypatch):
    """A missing entry and a clashing patch ask for opposite responses.

    A clash is a fact about two candidates and says the archive is working. An
    entry the archive cannot produce says it lost a candidate it claims to hold,
    which the retrieval map and every resumed run are also reading.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _stackable_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None
    Path(loop.archive._iter_dir(2) / "change.diff").unlink()

    staged, obstacle = loop._stage_merge_attempt(pair)

    assert staged == ""
    assert obstacle == "iteration 2's archived diff is missing or unreadable"


def test_stacking_can_be_turned_off(tmp_path, monkeypatch):
    """It changes what the ordinary single-session path does at every --lanes.

    An operator comparing against a run that predates it needs the older
    behaviour back, and no amount of stall state should reach it.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, merge_stacking=False)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 9
    _stackable_workspace(loop, workspace)

    assert loop._select_merge_attempt() is None


def test_stacking_never_discards_work_it_did_not_stage(tmp_path, monkeypatch):
    """Returning to canonical takes every tracked edit, not just the staged ones.

    Two patches that clash send the tree back to HEAD, which would delete an
    edit that was already there. The loop should reach this on a clean tree, but
    that is an invariant of earlier paths -- a stacking attempt must not enforce
    it by destroying the evidence that it was broken.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    _stackable_workspace(loop, workspace)
    pair = loop._select_merge_attempt()
    assert pair is not None

    kernel = workspace / "kernel.py"
    uncommitted = kernel.read_text() + "# work this iteration did not create\n"
    kernel.write_text(uncommitted)

    staged, obstacle = loop._stage_merge_attempt(pair)

    assert staged == ""
    assert obstacle == "the working tree already carried uncommitted work"
    assert kernel.read_text() == uncommitted


def test_a_supervisor_intervention_does_not_retire_a_stack_worth_measuring(tmp_path, monkeypatch):
    """The stall a stack answers to is the one only a KEEP clears.

    A memo redirects the next Implementer session; it does not measure the two
    complementary gains already sitting in the archive. Gating on the cooldown
    counter meant each of the 37 interventions across the 2026-08 archives
    silently retired a stall this mechanism was written for.
    """
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 2
    loop.run_state.stall.no_improvement_iters = 0
    _stackable_workspace(loop, workspace)

    pair = loop._select_merge_attempt()

    assert pair is not None
    assert {item.iteration for item in pair} == {1, 2}


def test_stacking_waits_until_single_patches_have_stalled(tmp_path, monkeypatch):
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = 1
    _stackable_workspace(loop, workspace)

    assert loop._select_merge_attempt() is None


def _stalled_loop_behind_a_full_queue(
    tmp_path,
    monkeypatch,
    *,
    stall=MERGE_ATTEMPT_STALL_THRESHOLD,
    candidates=2,
    iterations=1,
):
    """A stalled run holding two bought candidates and a field of stackable gains."""
    loop, workspace = _reduction_loop(tmp_path, monkeypatch)
    loop._baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop._best_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.stall.unresolved_stall_iters = stall
    _stackable_workspace(loop, workspace, candidates=candidates)
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(loop.results) < iterations else 0.0,
    )
    # The archive is already two iterations deep, which is what a stall means.
    loop.resume = True
    subprocess.run(
        ["git", "checkout", "-B", loop.ic.git_branch],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    kernel = workspace / "kernel.py"
    loop._lane_queue = [
        runner_module.LaneResult(
            lane_id=str(lane),
            plan=f"tune line {line}",
            diff=_diff_of(
                workspace,
                lambda line=line: kernel.write_text(
                    kernel.read_text().replace(f"line_{line} = {line}", f"line_{line} = {line}00")
                ),
            ),
        )
        for lane, line in ((1, 5), (2, 6))
    ]
    loop._persist_lane_queue()
    loop.run_state.baseline_case_times = {"prefill": 1.0, "decode": 1.0}
    loop.run_state.git_branch = loop.ic.git_branch
    loop.run_state.head_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.state_store.save(loop.run_state)
    return loop, workspace


async def _reverting_iteration(_iteration, **_kwargs):
    """A candidate that measured cleanly and the KEEP gate turned down.

    Which is the outcome a stacked attempt has to have for a streak to be
    possible at all: it leaves ``unresolved_stall_iters`` higher than it found
    it. An ``IterationResult`` the constructor rejects would be caught by the
    loop's own crash guard and archived as a CRASH instead, which is a
    different decision on a different path.
    """
    return IterationResult(
        iteration=_iteration,
        duration_sec=0.0,
        validation_passed=True,
        validation_summary="",
        mean_case_speedup=1.0,
        kept=False,
    )


def test_a_queue_that_never_empties_does_not_starve_stacking(tmp_path, monkeypatch):
    """The queue holds the iteration only while it is still the thing working.

    A fan-out round refills whenever the queue drains, so on the thirty archived
    runs of 2026-08-22 and 08-23 a candidate was waiting on 409 of 549
    iterations. Deferring to that unconditionally is a gate stacking can never
    pass, and the mechanism ran 5 times on archives holding 20 pairs. Once the
    run is as stalled as a stack requires, the queue yields -- and it yields the
    iteration, not the candidates, which are still queued afterwards.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    events = LoopStateStore(str(workspace)).read_events()
    staged = [item for item in events if item.get("type") == "merge_attempt_staged"]
    precedence = [item for item in events if item.get("type") == "merge_took_precedence"]

    assert len(staged) == 1
    # The two counts answer different questions and the second cannot be read
    # off the first: how often a stack was measured, and how often one went
    # ahead of a candidate already paid for.
    assert len(precedence) == 1
    assert precedence[0]["lane_queue_depth"] == 2
    assert precedence[0]["unresolved_stall_iters"] == MERGE_ATTEMPT_STALL_THRESHOLD
    assert [item.plan for item in loop._lane_queue] == ["tune line 5", "tune line 6"]


def test_a_fresh_run_still_measures_the_candidates_it_bought(tmp_path, monkeypatch):
    """Precedence is the stall's, not stacking's.

    A queued candidate is kept 55.1% of the time while the search is still
    producing and 33.7% from the stall threshold on; the first of those numbers
    is why the queue keeps the iteration until the run has stopped resolving.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch, stall=MERGE_ATTEMPT_STALL_THRESHOLD - 1)
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    events = LoopStateStore(str(workspace)).read_events()

    assert not [item for item in events if item.get("type") == "merge_took_precedence"]
    assert [item.plan for item in loop._lane_queue] == ["tune line 6"]


def test_taking_precedence_and_failing_to_stage_costs_the_queue_nothing(tmp_path, monkeypatch):
    """A stack that cannot be built is not a turn anyone spent.

    The obstacle is still reported -- a selected pair that reaches no
    measurement is the failure this mechanism's counters exist to expose -- and
    the candidate that would have been displaced is measured by the same
    iteration, so nothing counts as displaced that was not.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(
        loop,
        "_stage_merge_attempt",
        lambda pair: ("", "iteration 1's diff would not apply over the other's"),
    )
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    events = LoopStateStore(str(workspace)).read_events()

    assert [item["obstacle"] for item in events if item.get("type") == "merge_attempt_declined"] == [
        "iteration 1's diff would not apply over the other's"
    ]
    assert not [item for item in events if item.get("type") == "merge_took_precedence"]
    assert not [item for item in events if item.get("type") == "merge_attempt_staged"]
    assert [item.plan for item in loop._lane_queue] == ["tune line 6"]


def _longest_consecutive_run(iterations):
    """The longest stretch of back-to-back iterations in a sorted list."""
    longest = streak = 0
    previous = None
    for value in iterations:
        streak = streak + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, streak)
        previous = value
    return longest


def test_a_merge_streak_is_bounded_so_the_queue_is_reached(tmp_path, monkeypatch):
    """A stall the archive keeps answering is not a licence to hold the loop.

    A stacked attempt reverts, so it leaves ``unresolved_stall_iters`` higher
    than it found it, and it drains nothing, so nothing about having run one
    makes the next one less likely. The pairs give out eventually -- the stack
    a streak archives carries the stacking prefix and ``eligible_candidates``
    skips it, so the pool is frozen while the streak spends it -- but only
    after as many iterations as the pool has pairs, which goes as the square of
    the pool. The queue-empty branch is where the next round is priced, so a
    streak that runs that long is a campaign that never asks whether it can
    still afford one.

    Four archived gains offer four complementary pairs here, which is enough
    to hold two lane candidates for four iterations. What the limit must do to
    them is defer, not drop: all four are still measured, in streaks of at most
    two, and the iteration the limit takes back goes to the queue.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch, candidates=4, iterations=5)
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    # Each of those iterations has to have reached the KEEP gate and been
    # turned down there, because that is the outcome that leaves the stall
    # standing and lets the next stack be selected. An iteration that raises
    # instead is caught by the loop's own crash guard and archived as a CRASH,
    # which leaves the merge events below asserting over a streak driven by a
    # different decision on a different path.
    assert [result.crashed for result in loop.results] == [False] * 5

    events = LoopStateStore(str(workspace)).read_events()
    staged = [item["iter"] for item in events if item.get("type") == "merge_attempt_staged"]
    precedence = [item for item in events if item.get("type") == "merge_took_precedence"]

    assert _longest_consecutive_run(staged) == MERGE_PRECEDENCE_STREAK_LIMIT
    # The limit defers a pair rather than refusing it: all four are still
    # measured inside these five iterations.
    assert len(staged) == 4
    # And the iteration the limit took back went to the queue, which is one
    # shallower for the last two stacks than it was for the first two.
    assert [item["lane_queue_depth"] for item in precedence] == [2, 2, 1, 1]
    assert [(item["iter"], item["obstacle"]) for item in events if item.get("type") == "merge_attempt_declined"] == [
        (
            staged[1] + 1,
            "2 stacked iterations have run back to back without the queue being reached",
        )
    ]


def test_what_precedence_records_is_a_queue_depth(tmp_path, monkeypatch):
    """The queue's length is not the number of measurements a stack displaces.

    ``_take_lane_candidate`` returns a single candidate, and only after
    ``_next_lane_candidate`` has dropped every entry that would move the
    measurement surface or whose diff no longer applies. So a stack goes ahead
    of at most one measurement and possibly none, and which it is cannot be
    known without popping the queue and writing the tree. Here one of the three
    entries could never have been measured at all and one is still queued when
    the run ends, against a recorded depth of three.
    """
    loop, workspace = _stalled_loop_behind_a_full_queue(tmp_path, monkeypatch, iterations=2)
    loop._lane_queue.insert(
        0,
        runner_module.LaneResult(
            lane_id="surface",
            plan="tune the driver",
            diff=_diff_of(
                workspace,
                lambda: (workspace / "driver.py").write_text("print('bypass')\n"),
            ),
        ),
    )
    loop._persist_lane_queue()
    monkeypatch.setattr(loop, "run_one_iteration", _reverting_iteration)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    events = LoopStateStore(str(workspace)).read_events()

    assert [item["lane_queue_depth"] for item in events if item.get("type") == "merge_took_precedence"] == [3]
    assert [item.plan for item in loop._lane_queue] == ["tune line 6"]


def test_an_unscored_candidate_is_not_pinned(tmp_path, monkeypatch):
    """A candidate that measured nothing cannot claim to have beaten anything.

    The pin gate reads the KEEP score, so a missing one has to fail
    closed. Reading it as a gain would point the retrieval map at work that was
    never shown to be worth re-reading.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, None),
        plan="fuse the two passes",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == []


def test_gain_over_pristine_is_pinned_before_the_first_keep(
    tmp_path,
    monkeypatch,
):
    """A missing incumbent is the pristine 1.0, matching the KEEP gate.

    Reading it as "no gain is possible" would lose every near miss of the
    cold-start iterations, where a real gain over pristine is necessarily still
    below the threshold.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.best_mean_case_speedup = None

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, 1.003),
        plan="hoist the mask computation",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == [1]


def test_a_regression_before_the_first_keep_is_not_pinned(
    tmp_path,
    monkeypatch,
):
    """The pristine fallback is a real bar, not a waiver.

    Paired with the test above: together they show a missing incumbent is read
    as 1.0 rather than as "anything qualifies", which would pin every cold-start
    candidate including the ones slower than the kernel they started from.
    """
    loop, _workspace = _reduction_loop(tmp_path, monkeypatch)
    loop.best_mean_case_speedup = None

    recorded = loop._record_iteration_outcome(
        _reverted_candidate(1, 0.971),
        plan="widen the accumulator",
        decision_label="REVERT_PERF",
    )

    assert recorded is True
    assert loop.run_state.pinned_iterations == []


def test_resume_replay_pins_sub_threshold_gains_but_not_regressions(
    tmp_path,
    monkeypatch,
):
    """The replay path must classify rejected candidates exactly as the loop does."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    store = LoopStateStore(str(workspace))
    state = RunState(campaign_id="campaign", baseline_wall_ms=1.0)
    store.save(state)
    store.append_event(
        make_event(
            "iteration_result",
            1,
            decision="REVERT_PERF",
            plan="vectorize the epilogue stores",
            wall_ms=0.998,
            mean_case_speedup=1.001918,
            best_after_ms=1.0,
            best_after_mean_case_speedup=1.0,
        )
    )
    store.append_event(
        make_event(
            "iteration_result",
            2,
            decision="REVERT_PERF",
            plan="raise the tile size",
            wall_ms=1.02,
            mean_case_speedup=0.984,
            best_after_ms=1.0,
            best_after_mean_case_speedup=1.0,
        )
    )
    loop.state_store = store
    loop.run_state = state

    planned, _, _, _ = loop._plan_resume_recovery(state, None)

    assert planned.next_iteration == 3
    assert planned.pinned_iterations == [1]


def test_keep_expires_active_supervisor_ruling(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.best_wall_ms = 0.8
    loop.best_mean_case_speedup = 1.25
    loop._supervisor_ruling = "Continue the previous stall direction."
    ruling_path = runner_module.latest_supervisor_ruling_path(str(workspace))
    ruling_path.parent.mkdir(parents=True, exist_ok=True)
    ruling_path.write_text(loop._supervisor_ruling)
    result = IterationResult(
        iteration=1,
        duration_sec=0.1,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=0.8,
        mean_case_speedup=1.25,
        kept=True,
        commit_hash="deadbeef",
    )

    loop._record_iteration_outcome(result, plan="new canonical best")

    assert loop._supervisor_ruling == ""
    assert not ruling_path.exists()


def test_session_admission_is_time_only_with_thirty_minute_reserve(
    tmp_path,
    monkeypatch,
):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop.results = [
        IterationResult(
            iteration=index,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary="test",
        )
        for index in range(1, 1001)
    ]

    assert not hasattr(loop.ic, "max_iterations")
    assert loop.ic.budget_reserve_sec == 30 * 60

    monkeypatch.setattr(loop, "_time_remaining", lambda: 30 * 60)
    assert loop._is_budget_exhausted() is False

    monkeypatch.setattr(loop, "_time_remaining", lambda: 30 * 60 - 1)
    assert loop._is_budget_exhausted() is True


def test_the_finalize_reserve_is_its_own_bound_not_a_term_inside_admission(
    tmp_path,
    monkeypatch,
    capsys,
):
    """The relationship ``budget_reserve_sec``'s defining comment states.

    That comment once said a round is admitted only when what remains covers
    its estimated cost ON TOP of the reserve. The code has never done that:
    both admission checks are handed ``_time_remaining()`` with nothing
    subtracted, so the reserve and a round's own requirement are two
    independent lower bounds and the larger of them binds -- ``max``, not a
    sum. That is deliberate, because the reserve is already withheld once by
    ``_is_budget_exhausted()`` and charging it again inside a round's cost
    refused rounds that went on to produce a KEEP.

    Pinned here so the comment cannot drift away from the code again in either
    direction: subtracting the reserve at either call site, or stacking it into
    a requirement, fails this test.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    reserve = float(loop.ic.budget_reserve_sec)

    # A campaign with no round of its own, so both requirements are the
    # constants and this test reads them rather than restating them.
    no_history: list = []
    admission_required = admit_round(
        remaining_sec=0.0,
        requested_lanes=3,
        history=no_history,
        measurement_sec=FIRST_ROUND_MEASUREMENT_SEC,
    ).required_sec
    dispatch_required = admit_dispatch(
        remaining_sec=0.0,
        measurement_sec=FIRST_ROUND_MEASUREMENT_SEC,
    ).required_sec
    round_required = max(admission_required, dispatch_required)
    clears_both = max(reserve, round_required)
    stacked = reserve + round_required
    assert 0 < clears_both < stacked

    seen: list[float] = []

    def _spy(real):
        def _wrapped(*, remaining_sec, **kwargs):
            seen.append(remaining_sec)
            return real(remaining_sec=remaining_sec, **kwargs)

        return _wrapped

    monkeypatch.setattr(runner_module, "admit_round", _spy(admit_round))
    monkeypatch.setattr(runner_module, "admit_dispatch", _spy(admit_dispatch))

    # Halfway between "clears each bound on its own" and "covers their sum".
    between = 0.5 * (clears_both + stacked)
    monkeypatch.setattr(loop, "_time_remaining", lambda: between)

    assert loop._admit_next_round(1) == 3
    assert loop._admit_dispatch(1) is True
    assert loop._is_budget_exhausted() is False
    # Both checks priced the round against the UNRESERVED remaining time. A
    # reserve subtracted at either call site shows up here as a smaller number.
    assert seen == [between, between]

    # Each bound refuses on its own, and neither needs the other's help. Above
    # the round requirement but below the reserve: the round would be
    # affordable, and the loop still will not start a session.
    assert dispatch_required < reserve
    monkeypatch.setattr(loop, "_time_remaining", lambda: dispatch_required + 1.0)
    assert loop._admit_dispatch(2) is True
    assert loop._is_budget_exhausted() is True

    # And the other way round: above the reserve but below what the cheapest
    # round costs, the reserve is satisfied and the round is still refused.
    assert round_required > reserve
    monkeypatch.setattr(loop, "_time_remaining", lambda: reserve + 1.0)
    assert loop._is_budget_exhausted() is False
    assert loop._admit_next_round(3) is None
    assert "ROUND REFUSED FOR BUDGET" in capsys.readouterr().out


def test_is_force_stopped_detects_stop_file(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    assert loop._is_force_stopped() is False
    (workspace / ".stop").touch()
    assert loop._is_force_stopped() is True


def _measurement_loop(monkeypatch, benchmark_result, workspace_dir="."):
    report = SimpleNamespace(
        all_passed=True,
        results=[
            SimpleNamespace(stage=5, stage_name="full", passed=True, snr_db=40.0),
        ],
        summary=lambda: "PASS",
    )
    benchmark_calls = []

    async def fake_validation(**_kwargs):
        return report

    async def fake_benchmark(**kwargs):
        benchmark_calls.append(kwargs)
        return dict(benchmark_result)

    async def fake_registers(**_kwargs):
        return {"success": False}

    monkeypatch.setattr(runner_module, "run_validation_pipeline", fake_validation)
    monkeypatch.setattr(runner_module, "measure_wallclock", fake_benchmark)
    monkeypatch.setattr(runner_module, "check_registers", fake_registers)
    monkeypatch.setattr(runner_module, "force_jit_rebuild", lambda _files: None)

    loop = object.__new__(IterationLoop)
    loop.ic = SimpleNamespace(
        build_command=None,
        driver_script="driver.py",
        snr_threshold=30.0,
        validate_stage_timeout_sec=300,
        bench_timeout_sec=300,
        bench_repeat=1,
        nproc_per_node=1,
        build_dir=None,
        baseline_wall_ms=5.0,
        kernel_file="kernel.py",
        source_files=[],
        target_functions=[],
        workspace_dir=str(workspace_dir),
    )
    loop.best_wall_ms = 5.0
    loop.best_mean_case_speedup = 1.0
    loop.experiment = None
    loop._baseline_case_times = {"small": 1.0, "large": 1.0}
    loop._best_case_times = {"small": 1.0, "large": 1.0}
    loop._case_times = {}
    loop._last_pmc_diagnosis = ""
    loop._last_pmc_full = ""
    loop.evolver = SimpleNamespace(on_benchmark=lambda **_kwargs: None)
    loop.config = SimpleNamespace(gpu_target="gfx942")
    # No round is open around these iterations, so the measurement they run is
    # charged to nothing -- which is also what a drain iteration does.
    loop._round_started_at = None
    loop._round_measurement_sec = 0.0
    return loop, benchmark_calls


@pytest.mark.asyncio
async def test_iteration_keeps_a_gain_that_repeats_across_all_three_runs(
    monkeypatch,
):
    candidate_ms = 1.0 / 1.005
    loop, benchmark_calls = _measurement_loop(
        monkeypatch,
        {
            "success": True,
            "median_ms": candidate_ms,
            "case_times": {"small": candidate_ms, "large": candidate_ms},
            "unscored_cases": [],
            "measurement_count": 3,
            "measurements": [
                {
                    "success": True,
                    "case_times": {
                        "small": candidate_ms,
                        "large": candidate_ms,
                    },
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
            "message": "three measurements",
        },
    )

    result = await loop.run_one_iteration(1)

    assert result.kept is True
    assert result.bench_detail["mean_case_speedup"] == pytest.approx(1.005)
    assert result.bench_detail["measurement_mean_case_speedups"] == pytest.approx([1.005, 1.005, 1.005])
    assert len(benchmark_calls) == 1
    assert benchmark_calls[0]["measurements"] == 3


def _faster_bench(candidate_ms=1.0 / 1.05):
    return {
        "success": True,
        "median_ms": candidate_ms,
        "case_times": {"small": candidate_ms, "large": candidate_ms},
        "unscored_cases": [],
        "measurement_count": 3,
        "measurements": [
            {
                "success": True,
                "case_times": {"small": candidate_ms, "large": candidate_ms},
                "unscored_cases": [],
            }
            for _ in range(3)
        ],
        "message": "three measurements",
    }


def _canonical_workspace(tmp_path, command: str) -> Path:
    # The arena fails a task that declares no compile_command, so the gate needs
    # a Step 1 that passes before it reaches the correctness command under test.
    tmp_path.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "compile_command": [f"{sys.executable} -c 'pass'"],
                "correctness_command": [f"{sys.executable} -c {command!r}"],
            }
        )
    )
    return tmp_path


@pytest.mark.asyncio
async def test_snr_pass_with_failing_canonical_suite_is_reverted(tmp_path, monkeypatch, capsys):
    """The mla-decode run: 33.4 dB cleared forge's gate, 0.02468 broke the task's.

    The SNR probe passes, the candidate is 5% faster, and the task's own suite
    rejects it. That candidate must not be kept, and the tolerance it broke --
    not the dB figure -- has to reach the agent.
    """
    workspace = _canonical_workspace(
        tmp_path,
        "raise AssertionError('normalized max err 0.02468 too high')",
    )
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=workspace)

    result = await loop.run_one_iteration(1)

    assert result.kept is False
    assert result.validation_passed is False
    assert result.validation_outcome == "canonical_correctness_failure"
    assert "0.02468" in result.error_output
    assert "[canonical] FAIL" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_canonical_suite_passing_keeps_the_faster_candidate(tmp_path, monkeypatch):
    workspace = _canonical_workspace(tmp_path, "print('all cases PASS')")
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=workspace)

    result = await loop.run_one_iteration(1)

    assert result.kept is True
    assert result.validation_passed is True


@pytest.mark.asyncio
async def test_canonical_suite_output_reporting_failure_reverts(tmp_path, monkeypatch):
    workspace = _canonical_workspace(tmp_path, "print('mla-decode-bs64-kv8192: FAILED')")
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=workspace)

    result = await loop.run_one_iteration(1)

    assert result.kept is False
    assert "mla-decode-bs64-kv8192" in result.error_output


@pytest.mark.asyncio
async def test_workspace_declaring_no_correctness_command_cannot_keep(tmp_path, monkeypatch):
    tmp_path.joinpath("config.yaml").write_text('compile_command:\n  - "true"\n')
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=tmp_path)

    result = await loop.run_one_iteration(1)

    assert result.kept is False
    assert "declares no 'correctness_command'" in result.validation_summary


@pytest.mark.asyncio
async def test_workspace_without_a_config_keeps_on_the_snr_verdict_alone(tmp_path, monkeypatch, capsys):
    """Non-arena runs (flydsl, fusion, the examples) must keep working.

    There is no canonical suite to consult, so the SNR verdict still decides --
    but the operator is told the KEEP carries nothing else behind it.
    """
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(), workspace_dir=tmp_path)

    result = await loop.run_one_iteration(1)

    assert result.kept is True
    assert "[canonical] UNVERIFIED" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_canonical_suite_is_skipped_for_a_candidate_that_is_not_faster(tmp_path, monkeypatch):
    """The suite is the expensive check; a slower candidate is reverted anyway."""
    workspace = _canonical_workspace(tmp_path, "raise AssertionError('this must never run')")
    loop, _benchmark_calls = _measurement_loop(monkeypatch, _faster_bench(candidate_ms=2.0), workspace_dir=workspace)

    result = await loop.run_one_iteration(1)

    assert result.kept is False
    assert result.validation_passed is True
    assert result.error_output == ""


@pytest.mark.asyncio
async def test_iteration_keeps_winning_mean_despite_one_regressed_case(monkeypatch):
    loop, _benchmark_calls = _measurement_loop(
        monkeypatch,
        {
            "success": True,
            "median_ms": 1.0,
            "case_times": {"small": 0.5, "large": 1.5},
            "unscored_cases": [],
            "measurement_count": 3,
            "measurements": [
                {
                    "success": True,
                    "case_times": {"small": 0.5, "large": 1.5},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
            "message": "three measurements",
        },
    )

    result = await loop.run_one_iteration(1)

    assert result.kept is True
    assert result.bench_detail["mean_case_speedup"] == pytest.approx((2.0 + 2.0 / 3.0) / 2.0)


def test_agent_error_is_reduced_and_persisted(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def failing_agent(*_args, **_kwargs):
        raise RuntimeError("agent exploded")

    asyncio.run(loop.run(agent_fn=failing_agent, supervisor_fn=_unused_supervisor))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]
    assert state.iteration == 1
    assert state.stall.no_improvement_iters == 1
    assert loop.monitor.no_improve_streak == 1
    assert decisions == ["AGENT_ERROR"]
    history = (workspace / "forge_experiments" / "optimization_history.md").read_text()
    assert "Iteration 1 — AGENT_ERROR" in history
    assert loop.archive.load_index() == []
    assert loop.tracker.get(loop.experiment.experiment_id).iterations == []


def test_agent_error_with_diff_is_handed_to_outer_validation(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    validated = []

    async def editing_then_failing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "change before SDK failure"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        raise RuntimeError("SDK stream failed after edit")

    async def reject_candidate(iteration, plan=""):
        validated.append((iteration, plan))
        return IterationResult(
            iteration=iteration,
            duration_sec=0.0,
            validation_passed=False,
            validation_summary="canonical correctness failed",
        )

    monkeypatch.setattr(loop, "run_one_iteration", reject_candidate)
    asyncio.run(
        loop.run(
            agent_fn=editing_then_failing_agent,
            supervisor_fn=_unused_supervisor,
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]
    assert validated == [(1, "change before SDK failure")]
    assert decisions == ["REVERT_VALIDATION"]


def test_integrity_violation_restores_and_skips_canonical_validation(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    protected = workspace / "nested" / "tests" / "oracle.bin"
    protected.parent.mkdir(parents=True)
    protected.write_text("original\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add protected oracle"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    validation_calls: list[int] = []

    async def unsafe_agent(kernel_path, _history, session_sink):
        Path(kernel_path).write_text("def kernel():\n    return 99\n")
        protected.write_text("gamed\n")
        session_sink["plan"] = "unsafe candidate"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["integrity_violation"] = True
        session_sink["integrity_verdict"] = "violation"
        session_sink["integrity_reason"] = "modified nested/tests/oracle.bin"
        session_sink["integrity_restore"] = lambda: protected.write_text("original\n")
        return "unsafe candidate"

    async def unexpected_validation(*_args, **_kwargs):
        validation_calls.append(1)
        raise AssertionError("canonical validation must be skipped")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(
        loop.run(
            agent_fn=unsafe_agent,
            supervisor_fn=_unused_supervisor,
        )
    )

    assert validation_calls == []
    assert kernel.read_text() == "def kernel():\n    return 1\n"
    assert protected.read_text() == "original\n"
    assert loop.results[0].integrity_violation is True
    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]
    assert decisions == ["REVERT_INTEGRITY"]


def test_a_contended_workspace_skips_the_canonical_measurement(
    tmp_path,
    monkeypatch,
):
    """A leftover process the reaper could not clear is still holding the GPU.

    The session ended, the candidate is on disk, and the benchmark that decides
    KEEP is about to run against a device something else is using. That number
    would be this candidate's plus whatever it is sharing the device with, and
    the loop would act on it -- so the measurement is skipped and the candidate
    reverted, the same way a protected-integrity violation is handled.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    kernel = workspace / "kernel.py"
    validation_calls: list[int] = []

    async def contended_agent(kernel_path, _history, session_sink):
        Path(kernel_path).write_text("def kernel():\n    return 99\n")
        session_sink["plan"] = "a candidate nothing can measure"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["workspace_contention"] = "pid(s) [4321] survived SIGKILL; pid(s) [4321] hold a device node"
        return "contended candidate"

    async def unexpected_validation(*_args, **_kwargs):
        validation_calls.append(1)
        raise AssertionError("canonical validation must be skipped")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(
        loop.run(
            agent_fn=contended_agent,
            supervisor_fn=_unused_supervisor,
        )
    )

    assert validation_calls == []
    # Unmeasured means unkept: HEAD stays at the last state a benchmark ever
    # backed, rather than carrying a candidate nobody verified.
    assert kernel.read_text() == "def kernel():\n    return 1\n"
    assert "4321" in loop.results[0].workspace_contention
    assert loop.results[0].kept is False
    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]
    assert decisions == ["REVERT_CONTENDED"]


def _fake_device(monkeypatch, holders: dict[int, int]) -> dict[int, int]:
    """The device state the hazard re-check reads, without a process on it.

    ``holders`` maps pid to start time for whatever currently has a device node
    open; mutating it afterwards is how a test frees the device. Both of the
    reaper's readers are replaced, so nothing here depends on what is really
    running on the machine the suite is on.
    """

    def _read_proc(pid: int):
        if pid not in holders:
            return None
        return process_reaping._Proc(pid=pid, state="R", ppid=1, pgid=pid, starttime=holders[pid])

    monkeypatch.setattr(process_reaping, "_read_proc", _read_proc)
    monkeypatch.setattr(process_reaping, "_holds_device", lambda pid: pid in holders)
    return holders


async def _contended_agent(kernel_path, _history, session_sink):
    """A session that finished and whose workspace could not be cleared."""
    Path(kernel_path).write_text("def kernel():\n    return 99\n")
    session_sink["plan"] = "a candidate nothing can measure"
    session_sink["end_reason"] = "candidate_submitted"
    session_sink["workspace_contention"] = "pid(s) [4321] hold a device node"
    return "contended candidate"


def test_one_contended_lane_costs_the_whole_round_its_measurement(
    tmp_path,
    monkeypatch,
):
    """The device is not per-lane, so neither is a lane that could not clear it.

    Dropping the contended lane and measuring its healthy sibling is the wrong
    half of the response: what the lane left running is on the same GPU the
    canonical benchmark is about to use, so the number the round would take is
    the sibling's plus whatever is still benching. The round takes no
    measurement, and the sibling's candidate -- already paid for -- stays queued
    for an iteration that can measure it.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    rounds: list[int] = []
    monkeypatch.setattr(
        loop,
        "_run_orchestration",
        _counting_plan(loop, workspace, rounds, plans=("tune prefill", "tune decode")),
    )

    async def one_lane_left_something_running(lane_dir: Path) -> ReapReport:
        if lane_dir.name == "2":
            return ReapReport(
                directory=str(lane_dir),
                unkillable=(4321,),
                holding_device=(4321,),
            )
        return ReapReport(directory=str(lane_dir))

    monkeypatch.setattr(fanout, "_reap_lane_processes", one_lane_left_something_running)

    async def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("a contended round must measure nothing")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=object(),
            agent_factory=_lane_agent_factory({"tune prefill": ("return 1", "return 2")}),
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert decisions == ["REVERT_CONTENDED"]
    assert loop.results[0].kept is False
    assert loop.results[0].validation_passed is False
    assert "4321" in loop.results[0].workspace_contention
    # Refused, not discarded: the healthy lane's session was the expensive part
    # and its candidate is still worth measuring once the device is free.
    assert [item.plan for item in loop._lane_queue] == ["tune prefill"]


def test_a_leaked_probe_costs_the_round_its_measurement(tmp_path, monkeypatch):
    """A probe is a benchmark, and the device it holds is not the round's alone.

    A specialist killed by its session timeout mid-probe leaves a process on the
    same GPU the canonical measurement is about to use. The lanes queue behind
    it on the campaign sentinel; the canonical measurement takes no lock and
    would have run straight into it. So the analysis phase's teardown finding
    becomes a hazard exactly as a contended lane's does, and the round it
    belongs to measures nothing.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, lanes=2)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )

    class _ProbeLeakingService:
        """Planning that succeeded and left a probe running behind it."""

        async def run(self, _context, *, usage=None, lanes=1):
            return OrchestrationRunResult(
                dispatch_plan=None,
                optimization_plans=("tune prefill", "tune decode"),
                structured_output_diagnostics={
                    "probe_device_hazard": {
                        "describe": ("pid(s) [4321] survived SIGKILL; pid(s) [4321] hold a device node"),
                        "pids": [4321],
                    }
                },
            )

    async def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("a round with a leaked probe must measure nothing")

    monkeypatch.setattr(loop, "run_one_iteration", unexpected_validation)

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
            orchestration_service=_ProbeLeakingService(),
            agent_factory=_lane_agent_factory({"tune prefill": ("return 1", "return 2")}),
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert decisions == ["REVERT_CONTENDED"]
    assert loop.results[0].kept is False
    assert loop.results[0].validation_passed is False
    assert "4321" in loop.results[0].workspace_contention
    assert "probe round" in loop.results[0].workspace_contention
    # Bought and kept: the lane sessions were the expensive part, and their
    # candidates are still worth measuring once the device is free.
    assert [item.plan for item in loop._lane_queue] == ["tune prefill"]


def test_a_hazard_outlives_the_iteration_that_found_it(tmp_path, monkeypatch):
    """Nothing about an iteration ending makes a foreign process let go.

    The first iteration refuses because the reaper said so. The second has no
    reaper finding of its own -- it never ran a session -- and must still refuse
    while the device is held, then run as usual once it is free. Both
    directions, decided by the device rather than by how many iterations have
    passed.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=3)
    holders = _fake_device(monkeypatch, {4321: 99})
    monkeypatch.setattr(runner_module, "processes_under", lambda _dir: {4321})
    sessions: list[int] = []

    async def agent(kernel_path, history, session_sink):
        sessions.append(len(loop.results) + 1)
        if len(loop.results) == 0:
            return await _contended_agent(kernel_path, history, session_sink)
        session_sink["plan"] = "inspect only"
        return "No source change was needed."

    def free_the_device_after_the_second_refusal(_result):
        if len(loop.results) == 2:
            holders.clear()

    asyncio.run(
        loop.run(
            agent_fn=agent,
            supervisor_fn=_unused_supervisor,
            on_iteration=free_the_device_after_the_second_refusal,
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert decisions == ["REVERT_CONTENDED", "REVERT_CONTENDED", "NO_CHANGES"]
    # The refused iteration bought nothing: no session, so no plan and no diff.
    assert sessions == [1, 3]
    assert loop.results[1].workspace_contention
    assert loop.results[2].workspace_contention == ""
    assert loop.termination_reason != "device_contended"


def test_a_hazard_nothing_clears_stops_the_run_rather_than_spinning(
    tmp_path,
    monkeypatch,
):
    """A foreign process may hold the device for the rest of the campaign.

    Retrying until the budget runs out spends a whole run producing nothing
    while reporting nothing wrong, which is no better than the bad measurement
    the refusal exists to prevent. The run ends under a reason of its own.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=20)
    _fake_device(monkeypatch, {4321: 99})
    monkeypatch.setattr(runner_module, "processes_under", lambda _dir: {4321})

    asyncio.run(
        loop.run(
            agent_fn=_contended_agent,
            supervisor_fn=_unused_supervisor,
        )
    )

    decisions = [
        event.get("decision")
        for event in LoopStateStore(str(workspace)).read_events()
        if event.get("type") == "iteration_result"
    ]

    assert loop.termination_reason == "device_contended"
    assert set(decisions) == {"REVERT_CONTENDED"}
    # The iteration that found it plus the ones it refused after: the re-check
    # that exhausts the hazard runs before that iteration spends anything, so
    # the run stops there rather than filing one more unmeasured result.
    assert len(decisions) == MAX_BLOCKED_ITERATIONS - 1


def test_an_api_outage_is_not_recorded_as_an_agent_decision(tmp_path, monkeypatch):
    """An outage and a deliberate no-op leave the same empty diff.

    Only the end reason separates them, and recording the outage as NO_CHANGES
    tells the next Session that the agent looked and chose to change nothing.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def api_failed_agent(_kernel_path, _history, session_sink):
        session_sink["end_reason"] = "api_error"
        return "agent session ended with an API error"

    asyncio.run(loop.run(agent_fn=api_failed_agent, supervisor_fn=_unused_supervisor))

    store = LoopStateStore(str(workspace))
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]
    assert decisions == ["API_ERROR"]
    history = (workspace / "forge_experiments" / "optimization_history.md").read_text()
    assert "Iteration 1 — API_ERROR" in history


def test_no_change_attempt_is_reduced_and_persisted(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]
    assert state.iteration == 1
    assert state.stall.no_improvement_iters == 1
    assert loop.monitor.no_improve_streak == 1
    assert decisions == ["NO_CHANGES"]
    history = (workspace / "forge_experiments" / "optimization_history.md").read_text()
    assert "Iteration 1 — NO_CHANGES" in history
    lesson = (workspace / "forge_experiments" / "lessons" / "iter_001.md").read_text()
    assert lesson.strip().startswith("SCOPE: measured on ")
    assert "OUTCOME: NO_CHANGES" in lesson
    assert loop.archive.load_index() == []
    assert loop.tracker.get(loop.experiment.experiment_id).iterations == []


def test_optimization_plan_path_is_injected_before_implementer(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    orchestration_service = object()
    captured = {}

    async def run_orchestration(
        *,
        iteration,
        orchestration_service,
        lanes=1,
    ):
        captured["iteration"] = iteration
        captured["service"] = orchestration_service
        captured["lanes"] = lanes
        plan_path = workspace / "forge_experiments" / "orchestration" / "iter_001" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.write_text("# Optimization plan\nVectorize loads.\n")
        return plan_path, ""

    async def agent_fn(_kernel_path, history, session_sink):
        captured["history"] = history
        session_sink["plan"] = "vectorize loads"
        return "Followed the selected direction."

    monkeypatch.setattr(loop, "_run_orchestration", run_orchestration)
    asyncio.run(
        loop.run(
            agent_fn=agent_fn,
            orchestration_service=orchestration_service,
            supervisor_fn=_unused_supervisor,
        )
    )

    assert captured["iteration"] == 1
    assert captured["service"] is orchestration_service
    assert captured["history"].startswith("## Required optimization plan")
    assert "optimization_plan.md" in captured["history"]
    assert "## Search Policy" in captured["history"]
    assert "Mode: EXPLOIT" in captured["history"]


def test_orchestration_context_uses_current_scored_cases(tmp_path, monkeypatch):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        baseline_case_times={"case-b": 2.0, "case-a": 1.0},
    )
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    loop._best_case_times = {"case-a": 0.8, "case-b": 1.5}

    context = loop._build_orchestration_context()

    assert context.analysis_commit == head
    assert [case.case_id for case in context.cases] == ["case-a", "case-b"]
    assert [case.latency_ms for case in context.cases] == [0.8, 1.5]
    assert "maximizing equal-weight mean incumbent-to-candidate" in context.objective
    assert context.source_map_path == str((workspace / "kernel.py").resolve())


def test_orchestration_context_publishes_the_campaign_editable_sources(
    tmp_path,
    monkeypatch,
):
    """The campaign's declared source set reaches the planner verbatim.

    ``campaign.source_files`` is ``[kernel, *sources]`` de-duplicated, so entry 0
    is the primary kernel path and the rest keep campaign order. Data and config
    files ride the same list as sources do -- a tuned CSV on that list is an
    editable file, and the planner has to be told so.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.run_state = RunState(head_commit=loop._git("rev-parse", "HEAD").splitlines()[0])
    kernel = str((workspace / "kernel.py").resolve())
    tuned_csv = str((workspace / "configs" / "tuned_shapes.csv").resolve())
    sibling = str((workspace / "pkg" / "dispatch_limits.py").resolve())
    loop.ic.source_files = [kernel, tuned_csv, sibling]
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )

    context = loop._build_orchestration_context()

    assert list(context.editable_sources) == [kernel, tuned_csv, sibling]
    assert context.to_prompt_dict()["editable_sources"] == [
        kernel,
        tuned_csv,
        sibling,
    ]


def test_orchestration_context_editable_sources_cover_a_single_file_task(
    tmp_path,
    monkeypatch,
):
    """A single-file task leaves ``source_files`` empty; the anchor is still it."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.run_state = RunState(head_commit=loop._git("rev-parse", "HEAD").splitlines()[0])
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )

    context = loop._build_orchestration_context()

    assert list(context.editable_sources) == [str((workspace / "kernel.py"))]


def test_lessons_are_orchestration_evidence_and_handoff_is_audit_only(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
        )
    )
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )

    context = loop._build_orchestration_context()
    evidence = {item.kind: item for item in context.evidence_refs}

    assert evidence["lesson_directory"].path.endswith("forge_experiments/lessons")
    assert evidence["latest_lesson"].path.endswith("lessons/iter_001.md")
    assert "run_state" in evidence
    assert "iteration_handoff" not in evidence
    handoff = json.loads((workspace / "forge_experiments" / "handoffs" / "iter_001.json").read_text())
    assert handoff["canonical_verdict"] == "NO_CHANGES"
    assert handoff["lesson_path"].endswith("lessons/iter_001.md")
    assert handoff["search_policy"]["mode"] == "EXPLOIT"


def test_implementer_receives_partial_analysis_artifact_catalog(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    catalog = workspace / "forge_experiments" / "analysis" / "catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}")
    base = loop._build_orchestration_context()
    loop._active_analysis_context = replace(
        base,
        source_map_path=str(workspace / "source_map.md"),
        cases=(
            replace(
                base.cases[0],
                bottleneck="memory-latency",
                profile_summary_path=str(workspace / "normalized_metrics.json"),
                flags=(
                    "analysis_checkpoint_normalized_only",
                    "analysis_static_only",
                ),
            ),
        ),
        evidence_refs=(
            *base.evidence_refs,
            EvidenceRef(
                kind="analysis_artifact_catalog",
                path=str(catalog),
                summary="Partial Analysis artifact map.",
            ),
        ),
    )

    rendered = loop._render_analysis_evidence_for_implementer()

    assert "Artifact catalog:" in rendered
    assert str(catalog) in rendered
    assert "bottleneck=memory-latency" in rendered
    assert "normalized_metrics.json" in rendered
    assert "analysis_checkpoint_normalized_only" in rendered
    assert "STATIC_ONLY" in rendered
    supervisor_context = json.loads(loop._build_supervisor_evidence_context(2))
    supervisor_paths = {item["path"] for item in supervisor_context["orchestration_context"]["evidence_refs"]}
    assert str(catalog) in supervisor_paths


def test_runner_uses_analysis_checkpoint_after_session_failure(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.state_store = LoopStateStore(str(workspace))
    catalog = workspace / "forge_experiments" / "analysis" / "catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}")

    class FailingAnalysisService:
        async def ensure_bundle(self, context, **_kwargs):
            raise RuntimeError("analysis interrupted")

        def apply_checkpoint(self, context):
            return replace(
                context,
                cases=(
                    replace(
                        context.cases[0],
                        flags=("analysis_checkpoint_raw_profile_only",),
                    ),
                ),
                evidence_refs=(
                    *context.evidence_refs,
                    EvidenceRef(
                        kind="analysis_artifact_catalog",
                        path=str(catalog),
                        summary="Partial Analysis checkpoint.",
                    ),
                ),
            )

    context = asyncio.run(loop._resolve_analysis_context(FailingAnalysisService()))

    assert context is loop._active_analysis_context
    assert any(reference.path == str(catalog) for reference in context.evidence_refs)
    assert "analysis_checkpoint_raw_profile_only" in (context.cases[0].flags)


def test_failed_initial_analysis_retries_next_planning_iteration(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.state_store = LoopStateStore(str(workspace))
    calls = 0

    class Bundle:
        analysis_commit = head
        root = workspace / "analysis"
        manifest = {"status": "READY"}
        outcome = SimpleNamespace(
            checkpoint_level="published",
            available_tier="profiled",
            upgrade_exhausted=False,
            to_dict=lambda: {},
        )

        def apply(self, context):
            return context

    class FlakyAnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, _context, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient gateway failure")
            return Bundle()

        def apply_checkpoint(self, context):
            return context

    async def exercise():
        service = FlakyAnalysisService()
        first = await loop._resolve_analysis_context(
            service,
            iteration=1,
        )
        duplicate = await loop._resolve_analysis_context(
            service,
            iteration=1,
        )
        recovered = await loop._resolve_analysis_context(
            service,
            iteration=2,
        )
        return first, duplicate, recovered

    first, duplicate, recovered = asyncio.run(exercise())

    assert calls == 2
    assert first.evidence_commit == ""
    assert duplicate.evidence_commit == ""
    assert recovered.evidence_commit == head
    assert loop.run_state.analysis.last_attempt_status == "success"
    events = loop.state_store.read_events()
    decisions = [(event["iter"], event["reasons"]) for event in events if event["type"] == "analysis_refresh_decision"]
    assert decisions == [
        (1, ["INITIAL_ANALYSIS"]),
        (1, ["ALREADY_ATTEMPTED_THIS_ITERATION"]),
        (2, ["RETRY_FAILED_ANALYSIS"]),
    ]


def test_exhausted_analysis_session_budget_is_not_retried(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.state_store = LoopStateStore(str(workspace))
    calls = 0

    class ExhaustedAnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, _context, **_kwargs):
            nonlocal calls
            calls += 1
            raise AnalysisAttemptLimitError("2/2 attempts used")

        def apply_checkpoint(self, context):
            return context

    async def exercise():
        service = ExhaustedAnalysisService()
        await loop._resolve_analysis_context(service, iteration=1)
        await loop._resolve_analysis_context(service, iteration=2)

    asyncio.run(exercise())

    assert calls == 1
    assert loop.run_state.analysis.last_attempt_status == "exhausted"
    decisions = [
        event["reasons"] for event in loop.state_store.read_events() if event["type"] == "analysis_refresh_decision"
    ]
    assert decisions == [
        ["INITIAL_ANALYSIS"],
        ["ANALYSIS_ATTEMPTS_EXHAUSTED"],
    ]


def test_stale_published_analysis_paths_survive_resume_style_reuse(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    evidence_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    commit_root = workspace / "forge_experiments" / "analysis" / evidence_commit
    generation = commit_root / "generation-001"
    generation.mkdir(parents=True)
    report = generation / "report.md"
    source_map = generation / "source_map.md"
    workflow = generation / "workflow.json"
    catalog = generation / "artifact_catalog.json"
    case_profile = generation / "cases" / "case" / "analysis.md"
    case_profile.parent.mkdir(parents=True)
    case_profile.write_text("# Memory bottleneck\n")
    report.write_text("# Last valid Analysis\n")
    source_map.write_text("# Source map\n")
    workflow.write_text("{}")
    catalog.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": str(report.resolve()),
                        "kind": "analysis_summary",
                        "description": "Published Analysis report.",
                    }
                ]
            }
        )
    )
    (commit_root / "published.json").write_text(json.dumps({"generation_root": generation.name}))

    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "small keep"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state.best = BestRecord(
        iteration=1,
        mean_case_speedup=1.01,
        commit_hash=canonical_commit,
    )
    loop.run_state.analysis.evidence_commit = evidence_commit
    loop.run_state.analysis.evidence_mean_case_speedup = 1.0
    loop.run_state.analysis.evidence_status = "profiled"
    loop.best_mean_case_speedup = 1.01

    class AnalysisService:
        def apply_published_evidence(
            self,
            context,
            *,
            evidence_commit,
        ):
            return replace(
                context,
                cases=(
                    replace(
                        context.cases[0],
                        bottleneck="memory",
                        profile_summary_path=str(case_profile.resolve()),
                        flags=(
                            "analysis_profiled",
                            "analysis_evidence_stale",
                        ),
                    ),
                ),
                evidence_refs=(
                    *context.evidence_refs,
                    EvidenceRef(
                        kind="analysis_bundle",
                        path=str(generation.resolve()),
                        summary="Published Analysis bundle.",
                    ),
                    EvidenceRef(
                        kind="analysis_artifact_catalog",
                        path=str(catalog.resolve()),
                        summary="Analysis artifact catalog.",
                    ),
                    EvidenceRef(
                        kind="analysis_summary",
                        path=str(report.resolve()),
                        summary="Analysis report.",
                    ),
                    EvidenceRef(
                        kind="analysis_workflow",
                        path=str(workflow.resolve()),
                        summary="Analysis workflow.",
                    ),
                    EvidenceRef(
                        kind="profile",
                        path=str(case_profile.resolve()),
                        summary="Per-case profiling analysis.",
                    ),
                ),
                evidence_commit=evidence_commit,
                evidence_stale=True,
            )

        def apply_checkpoint(self, context):
            return context

        async def ensure_bundle(self, *_args, **_kwargs):
            raise AssertionError("sub-threshold reuse must not refresh")

    context = asyncio.run(loop._resolve_analysis_context(AnalysisService()))
    evidence_paths = {reference.path for reference in context.evidence_refs}
    analysis_paths = {
        str(generation.resolve()),
        str(catalog.resolve()),
        str(report.resolve()),
        str(workflow.resolve()),
        str(case_profile.resolve()),
        context.cumulative_diff_path,
    }

    assert context.evidence_stale is True
    assert analysis_paths <= evidence_paths
    assert all(Path(path).is_absolute() for path in analysis_paths)
    assert all(Path(path).is_relative_to(workspace) for path in analysis_paths)
    rendered = loop._render_analysis_evidence_for_implementer()
    assert str(catalog.resolve()) in rendered
    assert context.cumulative_diff_path in rendered
    assert "bottleneck=memory" in rendered
    assert f"evidence={case_profile.resolve()}" in rendered
    supervisor = json.loads(loop._build_supervisor_evidence_context(2))
    assert supervisor["orchestration_context"]["analysis_evidence"]["commit"] == evidence_commit
    assert supervisor["artifact_paths"]["analysis_bundle"] == str(generation.resolve())


def test_warm_start_search_policy_is_exploit_and_persisted(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(
        best=BestRecord(
            iteration=0,
            mean_case_speedup=1.2,
            commit_hash="warm-commit",
            source="warm_start",
        )
    )
    loop.handoff_store = runner_module.HandoffStore(str(workspace))

    decision = loop._update_search_policy(1)
    persisted = loop.state_store.load()

    assert decision.mode == "EXPLOIT"
    assert decision.reason_codes == ("KB_WARM_START_EXPLOIT",)
    assert persisted.search_mode == "EXPLOIT"
    assert persisted.search_reason_codes == ["KB_WARM_START_EXPLOIT"]


def test_completed_diversify_cycle_enters_bounded_exploit_residence(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=3,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(
        search_mode="DIVERSIFY",
        diversification_cycle_completed=True,
        best=BestRecord(
            iteration=0,
            mean_case_speedup=1.2,
            commit_hash="warm-commit",
            source="warm_start",
        ),
    )
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    loop.run_state.stall.unresolved_stall_iters = 1

    exploit = loop._update_search_policy(2)
    residence = loop._update_search_policy(3)

    assert exploit.mode == "EXPLOIT"
    assert exploit.reason_codes == ("DIVERSIFY_PLAN_CREATED",)
    assert exploit.residence_iterations_remaining == 2
    assert residence.mode == "EXPLOIT"
    assert residence.reason_codes == ("MODE_RESIDENCE",)
    assert residence.residence_iterations_remaining == 1


def test_search_policy_uses_run_state_when_handoff_is_unavailable(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=3,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.handoff_store = None
    loop.run_state = RunState(search_mode="DIVERSIFY")

    loop._apply_iteration_planning_state(
        optimization_plan_created=True,
    )
    decision = loop._update_search_policy(2)

    assert decision.reason_codes == ("DIVERSIFY_PLAN_CREATED",)
    assert decision.mode == "EXPLOIT"


def test_unsuccessful_diversify_cycle_stays_in_diversify(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=3,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(search_mode="DIVERSIFY")
    loop.run_state.stall.unresolved_stall_iters = 5

    loop._apply_iteration_planning_state(
        optimization_plan_created=False,
    )
    decision = loop._update_search_policy(4)

    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)
    assert decision.mode == "DIVERSIFY"


def test_a_supervisor_intervention_no_longer_erases_the_stall_it_answers(
    tmp_path,
    monkeypatch,
):
    """The mla_decode sequence: three REVERTs, an intervention, then DIVERSIFY.

    While both mechanisms read one counter, the intervention zeroed it and
    ``_update_search_policy`` read the zero fourteen lines later, so the
    no-improvement route into DIVERSIFY could never fire: four and seven
    interventions in the 2026-08-18 batch produced no mode switch at all.
    Asking for advice and changing search direction are now simultaneous.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.monitor = SupervisionMonitor(supervise_after=3, cooldown=3)

    for iteration in (1, 2, 3):
        apply_iteration(
            loop.run_state,
            iteration=iteration,
            decision="REVERT_PERF",
            kept=False,
            wall_ms=1.1,
            commit_hash="",
            plan="another pass over the same loop body",
            baseline_wall_ms=1.0,
            best_wall_ms=1.0,
            stall_threshold=3,
        )
        loop.monitor.record(kept=False)

    should, reason = loop.monitor.should_intervene(4)
    assert should is True
    assert "3 consecutive" in reason

    loop.monitor.mark_intervened(4)
    apply_supervisor_intervention(
        loop.run_state,
        iteration=4,
        stall_threshold=3,
    )

    # The supervisor's own trigger is throttled exactly as before.
    assert loop.monitor.no_improve_streak == 0
    assert loop.run_state.stall.no_improvement_iters == 0
    assert loop.monitor.should_intervene(5) == (False, "")

    decision = loop._update_search_policy(4)

    assert loop.run_state.stall.unresolved_stall_iters == 3
    assert decision.mode == "DIVERSIFY"
    assert decision.reason_codes == ("NO_IMPROVEMENT_STALL",)
    assert decision.objective_kind == OBJECTIVE_DISCOVER_NEW_MECHANISM
    assert loop.run_state.phase == PHASE_STALLED


def _kept_outcome(iteration, best_after, *, mode="EXPLOIT"):
    """One measured iteration that left the incumbent at ``best_after``."""
    return make_event(
        "iteration_result",
        iteration,
        decision="KEEP",
        search_mode=mode,
        best_after_mean_case_speedup=best_after,
    )


def test_a_flat_window_of_keeps_diversifies_a_campaign_that_never_stalled(
    tmp_path,
    monkeypatch,
):
    """Every iteration improved, so nothing else in the policy would fire.

    The stall streak is 0 and stays 0 for as long as the ladder produces any
    gain at all, which is exactly the campaign that refines one direction to the
    end of its budget.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState(
        best=BestRecord(
            iteration=MARGINAL_GAIN_WINDOW + 1,
            mean_case_speedup=1.53,
            commit_hash="kept",
            source="iteration",
        )
    )
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    for offset in range(MARGINAL_GAIN_WINDOW + 1):
        loop.state_store.append_event(_kept_outcome(offset + 1, 1.50 + 0.005 * offset))

    decision = loop._update_search_policy(MARGINAL_GAIN_WINDOW + 2)
    persisted = loop.state_store.load()
    recorded = [event for event in loop.state_store.read_events() if event.get("type") == "search_policy_decision"]

    assert loop.run_state.stall.no_improvement_iters == 0
    assert decision.mode == SEARCH_MODE_DIVERSIFY
    assert decision.reason_codes == ("DIMINISHING_RETURNS",)
    assert persisted.search_reason_codes == ["DIMINISHING_RETURNS"]
    # The ratio the decision was taken on is part of its audit trail: without it
    # the log says a window was flat but not how flat.
    assert recorded[-1]["window_gain_ratio"] == pytest.approx(0.02, abs=1e-9)


def test_a_diversification_starts_the_marginal_gain_window_again():
    """The round that acted on a flat window cannot be inside the next one.

    Otherwise the same flat outcomes are still in reach once mode residence
    expires, and the campaign diversifies again on evidence it already spent.
    """
    older = [_kept_outcome(offset + 1, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW + 1)]
    diversified = _kept_outcome(
        MARGINAL_GAIN_WINDOW + 2,
        1.54,
        mode=SEARCH_MODE_DIVERSIFY,
    )
    after = [_kept_outcome(MARGINAL_GAIN_WINDOW + 3 + offset, 1.54) for offset in range(MARGINAL_GAIN_WINDOW - 1)]

    assert IterationLoop._exploit_window_gain(
        older,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ).ratio == pytest.approx(0.02, abs=1e-9)
    assert IterationLoop._exploit_window_gain(
        [*older, diversified, *after],
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ) == WindowGain(ratio=None, unavailable="short_window")


@pytest.mark.parametrize(
    "failed_decision",
    ["AGENT_ERROR", "API_ERROR", "ORCHESTRATION_ERROR"],
)
def test_a_diversification_that_concluded_nothing_is_still_a_boundary(
    failed_decision,
):
    """A round that failed still separates two directions.

    The outcomes that fired the trigger are on the far side of it. If the failed
    round is skipped before its mode is read, the scan walks back into them and
    the campaign diversifies again on evidence it already spent -- with only
    mode residence left to brake it.
    """
    older = [_kept_outcome(offset + 1, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW + 1)]
    failed = make_event(
        "iteration_result",
        MARGINAL_GAIN_WINDOW + 2,
        decision=failed_decision,
        search_mode=SEARCH_MODE_DIVERSIFY,
    )
    after = [_kept_outcome(MARGINAL_GAIN_WINDOW + 3 + offset, 1.54) for offset in range(MARGINAL_GAIN_WINDOW - 1)]

    assert IterationLoop._exploit_window_gain(
        [*older, failed, *after],
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ) == WindowGain(ratio=None, unavailable="short_window")


def test_an_exploit_outcome_that_concluded_nothing_is_transparent():
    """An infrastructure failure inside one direction is not a boundary.

    Only a mode change is. Otherwise a single gateway outage would keep the
    window from ever filling on a campaign that never left EXPLOIT.
    """
    events = [
        _kept_outcome(1, 1.50),
        make_event(
            "iteration_result",
            2,
            decision="AGENT_ERROR",
            search_mode=SEARCH_MODE_EXPLOIT,
        ),
    ] + [_kept_outcome(offset + 3, 1.505 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW)]

    assert IterationLoop._exploit_window_gain(
        events,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ).ratio == pytest.approx(0.02, abs=1e-9)


def test_a_supervisor_direction_gets_a_window_of_its_own():
    """Outcomes recorded before an intervention cannot judge what it injected."""
    events = [_kept_outcome(offset + 1, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW + 1)]

    assert IterationLoop._exploit_window_gain(
        events,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=2,
    ) == WindowGain(ratio=None, unavailable="short_window")


@pytest.mark.parametrize(
    ("anchor", "reason"),
    [
        (None, "non_numeric_score"),
        (0.0, "non_positive_score"),
        ("1.5", "non_numeric_score"),
        (True, "non_numeric_score"),
        (float("nan"), "non_finite_score"),
    ],
)
def test_a_window_without_a_usable_anchor_names_why(anchor, reason):
    """A ratio needs a score to divide by, and a missing one is not a flat one.

    Each of these would otherwise become a gain: a missing or non-numeric score
    read as zero, a bool read as a speedup of 1.0, and a zero or NaN anchor
    turning the division into an exception or an infinity that compares false
    against every floor. They are different facts and are reported as such.
    """
    events = [_kept_outcome(1, anchor)] + [
        _kept_outcome(offset + 2, 1.50 + 0.005 * offset) for offset in range(MARGINAL_GAIN_WINDOW)
    ]

    assert IterationLoop._exploit_window_gain(
        events,
        window=MARGINAL_GAIN_WINDOW,
        since_iteration=0,
    ) == WindowGain(ratio=None, unavailable=reason)


def test_a_window_gain_is_never_both_a_ratio_and_a_reason():
    """The two fields are exclusive, so neither can be read as the other."""
    with pytest.raises(ValueError, match="either a ratio or a reason"):
        WindowGain(ratio=None, unavailable=None)
    with pytest.raises(ValueError, match="either a ratio or a reason"):
        WindowGain(ratio=0.02, unavailable="short_window")


def test_an_unevaluable_window_says_so_in_the_decision_event(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A trigger that cannot run must not log like one that ran and found gain.

    ``make_event`` drops None fields, so a bare ratio of None leaves the event
    identical whether the window was short, the score series unusable, or the
    ladder healthy. A campaign whose incumbent score is never recorded has this
    trigger disabled for its whole life; that fact has to be legible.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    for offset in range(MARGINAL_GAIN_WINDOW + 1):
        loop.state_store.append_event(_kept_outcome(offset + 1, None))

    decision = loop._update_search_policy(MARGINAL_GAIN_WINDOW + 2)
    recorded = [event for event in loop.state_store.read_events() if event.get("type") == "search_policy_decision"]

    assert decision.mode == SEARCH_MODE_EXPLOIT
    assert "window_gain_ratio" not in recorded[-1]
    assert recorded[-1]["window_gain_unavailable"] == "non_numeric_score"
    assert "non_numeric_score" in capsys.readouterr().out


def test_a_short_window_says_so_rather_than_saying_nothing(
    tmp_path,
    monkeypatch,
):
    """The ordinary young-campaign case is still recorded as a named absence."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, supervise_after=3)
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    loop.handoff_store = runner_module.HandoffStore(str(workspace))
    loop.state_store.append_event(_kept_outcome(1, 1.50))

    loop._update_search_policy(2)
    recorded = [event for event in loop.state_store.read_events() if event.get("type") == "search_policy_decision"]

    assert recorded[-1]["window_gain_unavailable"] == "short_window"


def test_repeated_empty_diffs_escalate_before_the_generic_stall_threshold(
    tmp_path,
    monkeypatch,
):
    """Consecutive empty diffs must force a new direction on their own.

    A direction the Implementer cannot express as an edit costs a whole session
    per attempt, so waiting for ``supervise_after`` no-improvement iterations
    lets the same fruitless direction be retried at full price.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=3)
    observed_modes = []

    async def stuck_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = "rewrite the reduction with warp shuffles"
        observed_modes.append(loop.run_state.search_mode)
        return "No source change was needed."

    asyncio.run(loop.run(agent_fn=stuck_agent, supervisor_fn=_unused_supervisor))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "NO_CHANGES", "NO_CHANGES"]
    assert state.stall.unresolved_stall_iters < loop.ic.supervise_after
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]


def test_single_empty_diff_does_not_escalate(tmp_path, monkeypatch):
    """One session that found nothing to change is not proof of a dead end."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    state = LoopStateStore(str(workspace)).load()

    assert state.search_mode == "EXPLOIT"
    assert state.search_reason_codes == ["CANONICAL_GAIN_AVAILABLE"]


def test_the_empty_diff_streak_survives_a_reworded_plan_headline(
    tmp_path,
    monkeypatch,
):
    """Rewording the same direction must not reset the streak.

    Every session is asked to close with a fresh one-line ``PLAN:`` headline in
    plain prose, so two sessions handed the same direction never word it the
    same way. Counting the streak against that sentence therefore reset it on
    every iteration and the escalation could only ever fire in a test that
    pinned one literal across sessions.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=3)
    headlines = [
        "fuse the two reduction passes",
        "merge both reduction passes into one",
        "collapse the reduction into a single pass",
    ]
    observed_modes = []

    async def reworded_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = headlines[len(loop.results)]
        observed_modes.append(loop.run_state.search_mode)
        return "No source change was needed."

    asyncio.run(loop.run(agent_fn=reworded_agent, supervisor_fn=_unused_supervisor))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "NO_CHANGES", "NO_CHANGES"]
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]


def test_api_outage_does_not_break_the_empty_diff_streak(tmp_path, monkeypatch):
    """An outage measured nothing, so it neither extends nor resets the streak.

    Both halves are visible in the mode observed per iteration: the third
    session still explores because the outage did not count as an empty diff,
    and the fourth diversifies because the outage did not discard the first one
    either. Skipping the outage outright is what makes the second half work: it
    ran under EXPLOIT like its neighbours, but reading its mode at all would let
    an outcome that measured nothing speak for the direction.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=4)
    plans = ["fuse the two passes", "", "fuse the two passes", "fuse the two passes"]
    observed_modes = []

    async def flaky_agent(_kernel_path, _history, session_sink):
        attempt = len(loop.results)
        session_sink["plan"] = plans[attempt]
        observed_modes.append(loop.run_state.search_mode)
        if attempt == 1:
            session_sink["end_reason"] = runner_module.EXHAUSTED_END_REASON
            return "agent session ended with an API error"
        return "No source change was needed."

    asyncio.run(loop.run(agent_fn=flaky_agent, supervisor_fn=_unused_supervisor))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "API_ERROR", "NO_CHANGES", "NO_CHANGES"]
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]


def test_a_crashed_session_does_not_break_the_empty_diff_streak(
    tmp_path,
    monkeypatch,
):
    """A session that died measured nothing either, so it is transparent too.

    AGENT_ERROR is produced on the same empty-diff branch as API_ERROR. It stays
    out of INFRASTRUCTURE_DECISIONS because that set also picks the cumulative
    counter bucket and there is none for an agent error, so it is the streak
    that has to read it as an attempt which never reached the kernel.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=4)
    plan = "fuse the two passes"
    observed_modes = []

    async def crashing_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = plan
        observed_modes.append(loop.run_state.search_mode)
        if len(loop.results) == 1:
            raise RuntimeError("SDK stream died mid-session")
        return "No source change was needed."

    asyncio.run(loop.run(agent_fn=crashing_agent, supervisor_fn=_unused_supervisor))

    store = LoopStateStore(str(workspace))
    state = store.load()
    decisions = [event.get("decision") for event in store.read_events() if event.get("type") == "iteration_result"]

    assert decisions == ["NO_CHANGES", "AGENT_ERROR", "NO_CHANGES", "NO_CHANGES"]
    assert observed_modes == ["EXPLOIT", "EXPLOIT", "EXPLOIT", "DIVERSIFY"]
    assert state.search_reason_codes == ["REPEATED_NO_CHANGES"]
    assert state.cumulative.reverted == 4
    assert state.cumulative.orchestration_errors == 0
    assert state.stall.unresolved_stall_iters == 4


def test_an_outage_run_cannot_push_the_first_empty_diff_out_of_the_window(
    tmp_path,
    monkeypatch,
):
    """The streak window counts outcomes, so outages cannot exhaust it.

    An iteration writes several events, so a window measured in raw log events
    covers only a handful of iterations: five interleaved outages would push the
    first empty diff out of it and silently disable the escalation. The streak
    is rebuilt from that same log rather than carried in a schema field, so what
    each session saw is asserted directly.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=8)
    plan = "fuse the two passes"
    observed_modes = []
    observed_streaks = []

    async def outage_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = plan
        observed_modes.append(loop.run_state.search_mode)
        observed_streaks.append(loop._consecutive_no_changes(loop.state_store.recent_results(NO_CHANGES_STREAK_WINDOW)))
        if 1 <= len(loop.results) <= 5:
            session_sink["end_reason"] = runner_module.EXHAUSTED_END_REASON
            return "agent session ended with an API error"
        return "No source change was needed."

    asyncio.run(loop.run(agent_fn=outage_agent, supervisor_fn=_unused_supervisor))

    reopened = LoopStateStore(str(workspace))
    live = reopened.load()
    events = reopened.read_events()
    decisions = [event.get("decision") for event in events if event.get("type") == "iteration_result"]
    resumed_streak = loop._consecutive_no_changes(reopened.recent_results(NO_CHANGES_STREAK_WINDOW))

    assert decisions == ["NO_CHANGES"] + ["API_ERROR"] * 5 + ["NO_CHANGES"] * 2
    # Without this the run would not exercise the distinction: a window counted
    # in raw events would still have held every outcome.
    assert len(events) > NO_CHANGES_STREAK_WINDOW
    # The last session is the whole point: it reached two only because the first
    # empty diff was still inside the window after five outages, and because the
    # outages neither reset it nor counted toward it.
    assert observed_streaks == [0, 1, 1, 1, 1, 1, 1, 2]
    assert observed_modes[-1] == "DIVERSIFY"
    assert live.search_reason_codes == ["REPEATED_NO_CHANGES"]
    # Reading the same log back reaches the same verdict for the mode the run
    # now sits in: the escalation moved it to DIVERSIFY, under which only the
    # last iteration's empty diff has been observed.
    assert resumed_streak == 1


def test_the_outcome_window_is_servable_from_the_cache_or_refused(tmp_path):
    """The streak window must be answerable in full, from memory, or fail.

    A short answer is indistinguishable from a short streak, so the cache is
    sized for the window and a wider request is refused instead of truncated.
    """
    store = LoopStateStore(str(tmp_path))
    for iteration in range(1, NO_CHANGES_STREAK_WINDOW + 2):
        store.append_event(make_event("iteration_started", iteration))
        store.append_event(make_event("iteration_result", iteration, decision="NO_CHANGES"))

    window = store.recent_results(NO_CHANGES_STREAK_WINDOW)

    assert NO_CHANGES_STREAK_WINDOW <= _RECENT_RESULT_CACHE
    assert [event["iter"] for event in window] == list(range(2, NO_CHANGES_STREAK_WINDOW + 2))
    assert LoopStateStore(str(tmp_path)).recent_results(NO_CHANGES_STREAK_WINDOW) == window
    with pytest.raises(ValueError, match="exceeds the cached outcome bound"):
        store.recent_results(_RECENT_RESULT_CACHE + 1)


def test_orchestration_persists_optimization_plan_without_decision_json(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    result = SimpleNamespace(
        succeeded=True,
        failure=None,
        optimization_plans=("# Optimization plan\nVectorize global loads.",),
        optimization_plan_draft="",
        optimization_plan_executable=True,
        dispatch_plan=None,
        specialist_outcomes=(),
        structured_output_diagnostics={},
        plan_critic=None,
        plan_revised=False,
    )

    class OrchestrationService:
        async def run(self, _context, **_kwargs):
            return result

    plan_path, error = asyncio.run(
        loop._run_orchestration(
            iteration=1,
            orchestration_service=OrchestrationService(),
        )
    )

    assert error == ""
    assert plan_path is not None
    assert plan_path.read_text() == ("# Optimization plan\nVectorize global loads.\n")
    assert not (plan_path.parent / "draft_plan.md").exists()
    assert not (plan_path.parent / "critic_review.md").exists()


def test_orchestration_persists_critic_draft_review_and_final_paths(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    critic = PlanCriticOutcome(
        verdict="REVISE",
        review=("VERDICT: REVISE\n\nCompare the existing GEMM path."),
    )
    result = SimpleNamespace(
        optimization_plans=("# Final plan\nBenchmark GEMM.",),
        optimization_plan_draft="# Draft plan\nContinue VALU.",
        optimization_plan_executable=True,
        dispatch_plan=None,
        specialist_outcomes=(),
        structured_output_diagnostics={
            "plan_critic": critic.to_dict(),
        },
        plan_critic=critic,
        plan_revised=True,
    )

    class OrchestrationService:
        async def run(self, _context, **_kwargs):
            return result

    plan_path, error = asyncio.run(
        loop._run_orchestration(
            iteration=1,
            orchestration_service=OrchestrationService(),
        )
    )
    root = workspace / "forge_experiments" / "orchestration" / "iter_001"
    diagnostics = json.loads((root / "structured_output.json").read_text())

    assert error == ""
    assert plan_path is not None
    assert (root / "draft_plan.md").read_text().startswith("# Draft plan")
    assert (root / "critic_review.md").read_text().startswith("VERDICT: REVISE")
    assert plan_path.read_text().startswith("# Final plan")
    assert diagnostics["plan_revised"] is True
    assert diagnostics["artifact_paths"] == {
        "critic_review": str((root / "critic_review.md").resolve()),
        "draft_plan": str((root / "draft_plan.md").resolve()),
        "final_plan": str(plan_path.resolve()),
    }


def test_framework_fallback_plan_does_not_complete_diversify_cycle(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(
        head_commit=head,
        search_mode="DIVERSIFY",
    )
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    result = SimpleNamespace(
        optimization_plans=("# Optimization plan\nInspect the evidence and formulate an optimization.",),
        optimization_plan_executable=False,
        optimization_plan_draft="",
        dispatch_plan=None,
        specialist_outcomes=(),
        structured_output_diagnostics={},
        plan_critic=None,
        plan_revised=False,
    )

    class OrchestrationService:
        async def run(self, _context, **_kwargs):
            return result

    plan_path, error = asyncio.run(
        loop._run_orchestration(
            iteration=1,
            orchestration_service=OrchestrationService(),
        )
    )
    loop._apply_iteration_planning_state(
        optimization_plan_created=bool(plan_path and loop._last_orchestration_plan_executable)
    )

    assert error == ""
    assert plan_path is not None
    assert loop.run_state.diversification_cycle_completed is False


def test_orchestration_plan_persistence_error_propagates(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    critic = PlanCriticOutcome(
        verdict="REVISE",
        review="VERDICT: REVISE\n\nMeasure the canonical path.",
    )

    class OrchestrationService:
        async def run(self, _context, **_kwargs):
            return SimpleNamespace(
                optimization_plans=("# Optimization plan\nVectorize loads.",),
                optimization_plan_executable=True,
                optimization_plan_draft="# Draft plan\nVectorize loads.",
                dispatch_plan=None,
                specialist_outcomes=(),
                structured_output_diagnostics={
                    "plan_critic": critic.to_dict(),
                },
                plan_critic=critic,
                plan_revised=True,
            )

    def fail_persistence(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(
        loop,
        "_persist_lane_plans",
        fail_persistence,
    )

    with pytest.raises(OSError, match="disk unavailable"):
        asyncio.run(
            loop._run_orchestration(
                iteration=1,
                orchestration_service=OrchestrationService(),
            )
        )
    root = workspace / "forge_experiments" / "orchestration" / "iter_001"
    diagnostics = json.loads((root / "structured_output.json").read_text())

    assert "final_plan" not in diagnostics["artifact_paths"]
    assert not (root / "optimization_plan.md").exists()


def test_analysis_service_rechecks_same_commit_for_partial_upgrade(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    baseline = loop._build_orchestration_context()
    monkeypatch.setattr(
        loop,
        "_build_orchestration_context",
        lambda: baseline,
    )

    analysis_calls = []

    class Bundle:
        root = workspace / "analysis"

        def __init__(self, analysis_commit, status):
            self.analysis_commit = analysis_commit
            self.manifest = {"status": status}
            self.outcome = SimpleNamespace(
                checkpoint_level="published",
                available_tier="profiled",
                upgrade_exhausted=False,
                to_dict=lambda: {},
            )

        def apply(self, context):
            return context

    class AnalysisService:
        async def ensure_bundle(self, context, **_kwargs):
            analysis_calls.append(context.analysis_commit)
            return Bundle(
                context.analysis_commit,
                "PARTIAL" if len(analysis_calls) == 1 else "READY",
            )

        def apply_checkpoint(self, context):
            return context

    async def exercise():
        analysis = AnalysisService()
        await loop._resolve_analysis_context(analysis)
        await loop._resolve_analysis_context(analysis)
        await loop._resolve_analysis_context(analysis, iteration=1)

    asyncio.run(exercise())

    assert analysis_calls == [
        baseline.analysis_commit,
        baseline.analysis_commit,
    ]
    assert loop.run_state.iteration == 0
    assert loop.run_state.analysis.last_attempt_iteration == 1


def test_keep_defers_incremental_analysis_until_next_request(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    helper = workspace / "helper.py"
    helper.write_text("HELPER_VALUE = 1\n")
    subprocess.run(["git", "add", "helper.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add helper"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    analysis_calls = []
    incrementals = []

    class Bundle:
        def __init__(self, analysis_commit):
            self.analysis_commit = analysis_commit
            self.root = workspace / "forge_experiments" / "analysis" / analysis_commit
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "manifest.json").write_text("{}")
            self.outcome = SimpleNamespace(
                checkpoint_level="published",
                to_dict=lambda: {},
            )

        def apply(self, context):
            return context

    class AnalysisService:
        async def ensure_bundle(
            self,
            context,
            *,
            incremental=None,
            **_kwargs,
        ):
            analysis_calls.append(context.analysis_commit)
            incrementals.append(incremental)
            return Bundle(context.analysis_commit)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "keep one candidate"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        helper.write_text("HELPER_VALUE = 2\n")
        return "candidate ready"

    async def successful_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="canonical validation passed",
            wall_ms=0.9,
            mean_case_speedup=1.1,
            snr_db=40.0,
            kept=True,
            bench_detail={"case_times": {"case": 0.9}},
        )

    monkeypatch.setattr(
        loop,
        "run_one_iteration",
        successful_iteration,
    )

    async def exercise():
        service = AnalysisService()
        await loop.run(
            agent_fn=editing_agent,
            analysis_service=service,
            supervisor_fn=_unused_supervisor,
        )
        assert len(analysis_calls) == 1
        assert loop._analysis_bundle is None
        await loop._resolve_analysis_context(service)

    asyncio.run(exercise())

    assert len(analysis_calls) == 2
    assert incrementals[0] is None
    assert incrementals[1] is not None
    assert incrementals[1].parent_commit == analysis_calls[0]
    assert "return 2" in incrementals[1].commit_diff
    assert "HELPER_VALUE = 2" in incrementals[1].commit_diff
    assert incrementals[1].changed_source_files == ("helper.py", "kernel.py")


def test_small_keeps_reuse_analysis_until_cumulative_gain_reaches_threshold(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    initial_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.best_mean_case_speedup = 1.0
    loop.run_state.analysis.evidence_commit = initial_commit
    loop.run_state.analysis.evidence_mean_case_speedup = 1.0
    loop.run_state.analysis.evidence_status = "profiled"
    loop._last_published_analysis_commit = initial_commit
    initial_root = workspace / "forge_experiments" / "analysis" / initial_commit
    initial_generation = initial_root / "generation-001"
    initial_generation.mkdir(parents=True)
    (initial_root / "published.json").write_text(json.dumps({"generation_root": initial_generation.name}))
    loop._active_analysis_context = replace(
        loop._build_orchestration_context(),
        evidence_commit=initial_commit,
        evidence_status="profiled",
        evidence_mean_case_speedup=1.0,
    )

    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "small keep"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    small_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state.best = BestRecord(
        iteration=1,
        mean_case_speedup=1.029,
        commit_hash=small_commit,
    )
    loop.best_mean_case_speedup = 1.029

    calls = []

    class Bundle:
        def __init__(self, analysis_commit):
            self.analysis_commit = analysis_commit
            self.root = workspace / "forge_experiments" / "analysis" / analysis_commit
            self.root.mkdir(parents=True, exist_ok=True)
            self.manifest = {"status": "READY"}
            self.outcome = SimpleNamespace(
                checkpoint_level="published",
                available_tier="profiled",
                upgrade_exhausted=False,
                to_dict=lambda: {},
            )

        def apply(self, context):
            return context

    class AnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, context, **kwargs):
            calls.append((context.analysis_commit, kwargs["incremental"]))
            return Bundle(context.analysis_commit)

        def apply_checkpoint(self, context):
            return context

    async def exercise():
        service = AnalysisService()
        stale = await loop._resolve_analysis_context(service)
        assert stale.evidence_stale is True
        assert Path(stale.cumulative_diff_path).is_absolute()
        assert Path(stale.cumulative_diff_path).is_file()
        assert calls == []

        kernel.write_text("def kernel():\n    return 3\n")
        subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
        subprocess.run(
            ["git", "commit", "-m", "cumulative keep"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )
        threshold_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
        loop.run_state.best = BestRecord(
            iteration=2,
            mean_case_speedup=1.05,
            commit_hash=threshold_commit,
        )
        loop.best_mean_case_speedup = 1.05
        current = await loop._resolve_analysis_context(service)
        return threshold_commit, current

    threshold_commit, current = asyncio.run(exercise())

    assert len(calls) == 1
    assert calls[0][0] == threshold_commit
    assert calls[0][1].parent_commit == initial_commit
    assert "return 3" in calls[0][1].commit_diff
    assert loop.run_state.analysis.evidence_commit == threshold_commit
    assert loop.run_state.analysis.evidence_mean_case_speedup == 1.05
    assert current.evidence_stale is False


def test_cumulative_analysis_diff_is_reused_for_immutable_commits(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    evidence_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "new canonical"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    original_git = analysis_evidence.git
    diff_calls = 0

    def counted_git(*args, **kwargs):
        nonlocal diff_calls
        if args[:2] == ("diff", "--no-ext-diff"):
            diff_calls += 1
        return original_git(*args, **kwargs)

    monkeypatch.setattr(analysis_evidence, "git", counted_git)

    first = loop._analysis_cumulative_diff(
        evidence_commit=evidence_commit,
        canonical_commit=canonical_commit,
    )
    second = loop._analysis_cumulative_diff(
        evidence_commit=evidence_commit,
        canonical_commit=canonical_commit,
    )

    assert first.path == second.path
    assert first.error == second.error == ""
    assert Path(first.path).is_file()
    assert diff_calls == 1


def test_missing_cumulative_diff_degrades_without_forcing_refresh(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    evidence_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state.analysis.evidence_commit = evidence_commit
    loop.run_state.analysis.evidence_mean_case_speedup = 1.0
    loop.run_state.analysis.evidence_status = "profiled"
    loop._active_analysis_context = replace(
        loop._build_orchestration_context(),
        evidence_commit=evidence_commit,
        evidence_status="profiled",
        evidence_mean_case_speedup=1.0,
    )

    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "small keep"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state.best = BestRecord(
        iteration=1,
        mean_case_speedup=1.01,
        commit_hash=canonical_commit,
    )
    loop.best_mean_case_speedup = 1.01
    original_git = analysis_evidence.git

    def fail_analysis_diff(*args, **kwargs):
        if args[:2] == ("diff", "--no-ext-diff"):
            return subprocess.CompletedProcess(
                ["git", *args],
                returncode=1,
                stdout="",
                stderr="simulated diff failure",
            )
        return original_git(*args, **kwargs)

    monkeypatch.setattr(analysis_evidence, "git", fail_analysis_diff)

    class AnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, _context, **_kwargs):
            raise AssertionError("diff failure must not force Analysis refresh")

        def apply_checkpoint(self, context):
            return context

    async def exercise():
        service = AnalysisService()
        below_threshold = await loop._resolve_analysis_context(
            service,
            iteration=2,
        )
        loop.run_state.analysis.last_attempt_commit = canonical_commit
        loop.run_state.analysis.last_attempt_status = "exhausted"
        loop.run_state.analysis.last_attempt_iteration = 2
        exhausted = await loop._resolve_analysis_context(
            service,
            iteration=3,
        )
        supervisor = await loop._resolve_analysis_context(
            service,
            iteration=4,
            supervisor_due=True,
        )
        return below_threshold, exhausted, supervisor

    below_threshold, exhausted, supervisor = asyncio.run(exercise())

    for context in (below_threshold, exhausted, supervisor):
        assert context.evidence_commit == evidence_commit
        assert context.evidence_stale is True
        assert context.cumulative_diff_path == ""
        assert "simulated diff failure" in context.cumulative_diff_error
    assert loop.persistence_degraded is True
    decisions = [event for event in loop.state_store.read_events() if event["type"] == "analysis_refresh_decision"]
    assert [event["reasons"] for event in decisions[-3:]] == [
        ["CUMULATIVE_GAIN_BELOW_THRESHOLD"],
        ["ANALYSIS_ATTEMPTS_EXHAUSTED"],
        ["ANALYSIS_ATTEMPTS_EXHAUSTED"],
    ]
    assert all("simulated diff failure" in event["cumulative_diff_error"] for event in decisions[-3:])


def test_analysis_refresh_event_failure_marks_persistence_degraded(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    loop.run_state = RunState()
    loop.state_store = SimpleNamespace(
        append_event=lambda _event: (_ for _ in ()).throw(OSError("event log unavailable"))
    )
    context = loop._build_orchestration_context()

    loop._record_analysis_refresh_decision(
        context,
        AnalysisRefreshDecision(
            refresh=False,
            reasons=("CURRENT_EVIDENCE",),
            evidence_stale=False,
            gain_since_evidence=0.0,
        ),
        iteration=1,
    )

    assert loop.persistence_degraded is True
    assert any("event log unavailable" in error for error in loop.persistence_errors)


def test_partial_bundle_does_not_inherit_prior_commit_evidence_refs(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    evidence_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    old_bundle = workspace / "forge_experiments" / "analysis" / evidence_commit / "generation-001"
    old_bundle.mkdir(parents=True)
    loop.run_state.analysis.evidence_commit = evidence_commit
    loop.run_state.analysis.evidence_mean_case_speedup = 1.0
    loop.run_state.analysis.evidence_status = "profiled"
    loop._active_analysis_context = replace(
        loop._build_orchestration_context(),
        evidence_commit=evidence_commit,
        evidence_status="profiled",
        evidence_mean_case_speedup=1.0,
        evidence_refs=(
            EvidenceRef(
                kind="analysis_bundle",
                path=str(old_bundle),
                summary="Prior evidence.",
            ),
        ),
    )

    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "threshold keep"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    canonical_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state.best = BestRecord(
        iteration=1,
        mean_case_speedup=1.05,
        commit_hash=canonical_commit,
    )
    loop.best_mean_case_speedup = 1.05
    apply_input_paths = set()
    current_bundle = workspace / "forge_experiments" / "analysis" / canonical_commit
    current_bundle.mkdir(parents=True)

    class Bundle:
        analysis_commit = canonical_commit
        root = current_bundle
        manifest = {"status": "PARTIAL"}
        outcome = SimpleNamespace(
            checkpoint_level="published",
            available_tier="profiled",
            upgrade_exhausted=False,
            to_dict=lambda: {},
        )

        def apply(self, context):
            apply_input_paths.update(reference.path for reference in context.evidence_refs)
            return replace(
                context,
                evidence_refs=(
                    *context.evidence_refs,
                    EvidenceRef(
                        kind="analysis_bundle",
                        path=str(current_bundle),
                        summary="Current partial evidence.",
                    ),
                ),
            )

    class AnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, _context, **_kwargs):
            return Bundle()

        def apply_checkpoint(self, context):
            return context

    context = asyncio.run(loop._resolve_analysis_context(AnalysisService()))
    result_paths = {reference.path for reference in context.evidence_refs}

    assert str(old_bundle) not in apply_input_paths
    assert str(old_bundle) not in result_paths
    assert str(current_bundle) in result_paths
    assert context.evidence_commit == canonical_commit
    assert context.evidence_stale is False
    assert context.evidence_status == "partial"


def test_supervisor_refreshes_stale_analysis_once(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    initial_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state.analysis.evidence_commit = initial_commit
    loop.run_state.analysis.evidence_mean_case_speedup = 1.0
    loop.run_state.analysis.evidence_status = "profiled"
    loop._last_published_analysis_commit = initial_commit

    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "sub-threshold keep"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    current_commit = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state.best = BestRecord(
        iteration=1,
        mean_case_speedup=1.01,
        commit_hash=current_commit,
    )
    loop.best_mean_case_speedup = 1.01
    calls = []

    class Bundle:
        analysis_commit = current_commit
        root = workspace / "forge_experiments" / "analysis" / current_commit
        manifest = {"status": "READY"}
        outcome = SimpleNamespace(
            checkpoint_level="published",
            available_tier="profiled",
            upgrade_exhausted=False,
            to_dict=lambda: {},
        )

        def apply(self, context):
            return context

    class AnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, context, **_kwargs):
            calls.append(context.analysis_commit)
            return Bundle()

        def apply_checkpoint(self, context):
            return context

    async def exercise():
        service = AnalysisService()
        await loop._resolve_analysis_context(
            service,
            supervisor_due=True,
        )
        await loop._resolve_analysis_context(
            service,
            supervisor_due=True,
        )

    asyncio.run(exercise())

    assert calls == [current_commit]
    assert loop.run_state.analysis.evidence_commit == current_commit


def test_loop_refreshes_stale_analysis_before_supervisor(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=1,
        session_count=3,
    )
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    analysis_calls = []
    agent_calls = 0

    class Bundle:
        def __init__(self, commit):
            self.analysis_commit = commit
            self.root = workspace / "forge_experiments" / "analysis" / commit
            self.root.mkdir(parents=True, exist_ok=True)
            self.manifest = {"status": "READY"}
            self.outcome = SimpleNamespace(
                checkpoint_level="published",
                available_tier="profiled",
                upgrade_exhausted=False,
                to_dict=lambda: {},
            )

        def apply(self, context):
            return context

    class AnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, context, **_kwargs):
            analysis_calls.append(context.analysis_commit)
            return Bundle(context.analysis_commit)

        def apply_checkpoint(self, context):
            return context

    async def editing_agent(kernel_path, _history, session_sink):
        nonlocal agent_calls
        agent_calls += 1
        session_sink["plan"] = f"candidate {agent_calls}"
        Path(kernel_path).write_text(f"def kernel():\n    return {agent_calls + 1}\n")
        return "candidate ready"

    async def canonical_result(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="canonical validation passed",
            wall_ms=0.99,
            mean_case_speedup=1.01,
            snr_db=40.0,
            kept=iteration == 1,
            bench_detail={"case_times": {"case": 0.99}},
        )

    supervisor_contexts = []

    async def supervisor(**kwargs):
        supervisor_contexts.append(json.loads(kwargs["evidence_context"]))
        assert len(analysis_calls) == 2
        assert loop.run_state.analysis.evidence_commit == loop.run_state.best.commit_hash
        return ""

    monkeypatch.setattr(loop, "run_one_iteration", canonical_result)

    asyncio.run(
        loop.run(
            agent_fn=editing_agent,
            analysis_service=AnalysisService(),
            supervisor_fn=supervisor,
        )
    )

    assert len(analysis_calls) == 2
    assert len(supervisor_contexts) == 1
    evidence = supervisor_contexts[0]["orchestration_context"]["analysis_evidence"]
    assert evidence["commit"] == loop.run_state.best.commit_hash
    assert evidence["stale"] is False
    events = LoopStateStore(str(workspace)).read_events()
    refresh_events = [
        event for event in events if event["type"] == "analysis_refresh_decision" and event["action"] == "refresh"
    ]
    analysis_results = [event for event in events if event["type"] == "analysis_result"]
    assert [(event["iter"], event["reasons"]) for event in refresh_events] == [
        (0, ["INITIAL_ANALYSIS"]),
        (3, ["SUPERVISOR_STALE_EVIDENCE"]),
    ]
    assert [event["iter"] for event in analysis_results] == [0, 3]


def test_handoff_records_optimization_plan_path(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    plan_path = workspace / "forge_experiments" / "orchestration" / "iter_001" / "optimization_plan.md"

    async def create_plan(**_kwargs):
        plan_path.parent.mkdir(parents=True)
        plan_path.write_text("# Optimization plan\nVectorize loads.\n")
        return plan_path, ""

    async def agent_fn(_kernel_path, history, session_sink):
        assert str(plan_path) in history
        session_sink["plan"] = "read the optimization plan"
        return "No code change needed."

    monkeypatch.setattr(loop, "_run_orchestration", create_plan)
    asyncio.run(
        loop.run(
            agent_fn=agent_fn,
            orchestration_service=object(),
            supervisor_fn=_unused_supervisor,
        )
    )

    handoff = json.loads((workspace / "forge_experiments" / "handoffs" / "iter_001.json").read_text())
    assert handoff["canonical_verdict"] == "NO_CHANGES"
    assert handoff["optimization_plan_path"].endswith("orchestration/iter_001/optimization_plan.md")


def test_resume_hydrates_supervision_monitor(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    store = LoopStateStore(str(workspace))
    state = RunState(
        baseline_case_times={"case": 1.0},
        best=BestRecord(
            iteration=2,
            wall_ms=0.8,
            mean_case_speedup=1.25,
            commit_hash=head,
        ),
    )
    state.stall.no_improvement_iters = 4
    state.stall.last_supervisor_iter = 2
    store.save(state)

    loop.state_store = store
    loop.run_state = store.load()
    loop.monitor = SupervisionMonitor()
    loop.best_wall_ms = 1.0
    loop._seed_and_hydrate_run_state()

    assert loop.monitor.no_improve_streak == 4
    assert loop.monitor.last_intervention_iter == 2


def test_resume_propagates_complete_ruling_to_planner_and_implementer(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    store = LoopStateStore(str(workspace))
    store.save(
        RunState(
            baseline_case_times={"case": 1.0},
            best_case_times={"case": 0.8},
            best=BestRecord(
                iteration=2,
                wall_ms=0.8,
                mean_case_speedup=1.25,
                commit_hash=head,
            ),
        )
    )
    ruling = "# Supervisor Ruling\n\nIgnore the earlier hard-floor conclusion; fused merge was not tested."
    ruling_path = runner_module.latest_supervisor_ruling_path(str(workspace))
    ruling_path.parent.mkdir(parents=True, exist_ok=True)
    ruling_path.write_text(ruling)
    loop.config = SimpleNamespace(
        gpu_target="gfx942",
        local_knowledge_dir=None,
        experiments_dir=workspace / "forge_experiments",
    )
    captured = {}

    async def orchestration(*, iteration, **_kwargs):
        captured["guidance"] = loop._build_orchestration_context().supervisor_guidance
        plan_path = workspace / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text("# Optimization plan\nInspect fused merge.\n")
        return plan_path, ""

    async def agent(_kernel_path, history, session_sink):
        captured["history"] = history
        session_sink["plan"] = "inspect fused merge"
        return "No source change."

    monkeypatch.setattr(loop, "_run_orchestration", orchestration)
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if not loop.results else 0.0,
    )

    asyncio.run(
        loop.run(
            agent_fn=agent,
            orchestration_service=object(),
            supervisor_fn=_unused_supervisor,
        )
    )

    assert loop._supervisor_ruling == ruling
    assert captured["guidance"] == ruling
    assert ruling in captured["history"]


def test_run_state_rejects_wrong_schema(tmp_path, monkeypatch):
    _make_loop(tmp_path, monkeypatch, resume=True)
    prior_schema = RunState().to_dict()
    prior_schema["schema_version"] = 12

    with pytest.raises(ValueError, match="unsupported run state schema"):
        RunState.from_dict(prior_schema)


def test_resume_requires_authoritative_task_fingerprint(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = loop._git("rev-parse", "HEAD").strip()
    current = loop._task_fingerprint()
    state = RunState(
        kernel_path=loop._workspace_path(loop.ic.kernel_file),
        task_fingerprint=current,
        git_branch=loop.ic.git_branch,
        head_commit=head,
        baseline_case_times={"case": 1.0},
    )

    loop._validate_resume_state(state)

    state.task_fingerprint = f"{current[:-1]}{'0' if current[-1] != '0' else '1'}"
    with pytest.raises(ValueError, match="task fingerprint mismatch"):
        loop._validate_resume_state(state)


def test_resume_restores_baseline_cases_without_resumable_best(
    tmp_path,
    monkeypatch,
):
    loop, _workspace = _make_loop(
        tmp_path,
        monkeypatch,
        resume=True,
        baseline_case_times={},
    )
    state = RunState(
        baseline_case_times={"case": 1.0},
    )

    loop._restore_resume_baseline_case_times(state)

    assert loop._baseline_case_times == {"case": 1.0}
    assert loop.ic.baseline_case_times == {"case": 1.0}


def test_resume_rejects_missing_baseline_cases_before_iterations(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = loop._git("rev-parse", "HEAD").strip()
    state = RunState(
        kernel_path=loop._workspace_path(loop.ic.kernel_file),
        task_fingerprint=loop._task_fingerprint(),
        git_branch=loop.ic.git_branch,
        head_commit=head,
    )

    with pytest.raises(
        ValueError,
        match="no pristine per-case timings",
    ):
        loop._validate_resume_state(state)


def test_run_fails_before_agent_without_baseline_cases(
    tmp_path,
    monkeypatch,
):
    loop, _workspace = _make_loop(
        tmp_path,
        monkeypatch,
        baseline_case_times={},
    )
    called = False

    async def agent(*_args, **_kwargs):
        nonlocal called
        called = True
        return "should not run"

    with pytest.raises(
        RuntimeError,
        match="requires pristine per-case timings",
    ):
        asyncio.run(loop.run(agent_fn=agent))

    assert called is False


def _static_bench(result: dict):
    """A measurement stand-in that always returns ``result``."""

    async def bench(**_kwargs):
        return dict(result)

    return bench


def test_baseline_crash_reports_the_driver_failure_not_the_output_format(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A crashed driver must be diagnosed by its own exit code and output."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "measure_wallclock",
        _static_bench(
            {
                "success": False,
                "message": "BENCH CRASHED (exit 1)",
                "output": (
                    "Traceback (most recent call last):\n"
                    "FileNotFoundError: could not locate the invocation "
                    "specification JSON\n"
                ),
            }
        ),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "BENCH CRASHED (exit 1)" in printed
    assert "could not locate the invocation specification JSON" in printed
    assert "case_ms" not in printed


def test_baseline_timeout_reports_the_timeout(tmp_path, monkeypatch, capsys):
    """A timed-out driver reports no output tail, only its verdict."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "measure_wallclock",
        _static_bench({"success": False, "message": "TIMEOUT after 300s"}),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "TIMEOUT after 300s" in printed
    assert "case_ms" not in printed


def test_baseline_without_case_lines_names_the_missing_contract(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A driver that ran but reported no per-case timings is a distinct fault."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "measure_wallclock",
        _static_bench(
            {
                "success": True,
                "median_ms": 5.5635,
                "message": "BENCH: mean=5.5635 ms",
                "case_times": {},
            }
        ),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "case_ms" in printed
    assert "BENCH: mean=5.5635 ms" in printed
    assert "CRASHED" not in printed


def test_baseline_without_aggregate_line_is_not_silent(
    tmp_path,
    monkeypatch,
    capsys,
):
    """A missing aggregate must be reported, not returned as a bare ``None``."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "measure_wallclock",
        _static_bench(
            {
                "success": True,
                "median_ms": None,
                "message": "BENCH: cases only",
                "case_times": {"case": 1.0},
            }
        ),
    )

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "Baseline bench FAILED" in printed
    assert "median_ms" in printed


def test_baseline_case_coverage_drift_names_the_cases(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Coverage that moves between measurements must name what moved."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch)

    async def bench(**_kwargs):
        return {
            "success": False,
            "message": ("MEASUREMENT CASE COVERAGE MISMATCH: expected=['case_001', 'case_002'], got=['case_001']"),
        }

    monkeypatch.setattr(runner_module, "measure_wallclock", bench)

    assert asyncio.run(loop._measure_baseline()) is None

    printed = capsys.readouterr().out
    assert "CASE COVERAGE MISMATCH" in printed
    assert "case_002" in printed


def test_resume_restores_baselines_before_best_publication_reconcile(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch, resume=True)
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = loop._git("rev-parse", "HEAD").strip()
    store = LoopStateStore(str(workspace))
    state = RunState(
        campaign_id="campaign",
        session_status=SESSION_PAUSED,
        kernel_path=loop._workspace_path(loop.ic.kernel_file),
        task_fingerprint=loop._task_fingerprint(),
        git_branch=loop.ic.git_branch,
        head_commit=head,
        baseline_wall_ms=0.8,
        pristine_baseline_wall_ms=1.0,
        baseline_case_times={"case": 1.0},
        best_case_times={"case": 0.7},
        best=BestRecord(
            iteration=1,
            wall_ms=0.7,
            mean_case_speedup=1.4,
            commit_hash=head,
            source="iteration",
        ),
    )
    store.save(state)
    observed = []

    def inspect_reconcile():
        observed.append(
            (
                loop.ic.baseline_wall_ms,
                loop.ic.pristine_baseline_wall_ms,
            )
        )

    monkeypatch.setattr(loop, "_reconcile_best_publication", inspect_reconcile)
    monkeypatch.setattr(loop, "_time_remaining", lambda: 0.0)

    asyncio.run(loop.run(agent_fn=_no_change_agent))

    assert observed == [(0.8, 1.0)]


def test_reconcile_skips_republishing_a_best_the_manifest_already_names(
    tmp_path,
    monkeypatch,
):
    """A resume that changed nothing must not report persistence as degraded.

    Reconciliation republishes run_state.best to repair a crashed manifest, but
    a resumed session recomputes session_index and experiment_id that differ
    from what the stored manifest was written with. Republishing an
    already-current best then tripped the same-iteration conflict guard and set
    persistence_degraded, while the KEEP, the git state and run_state.best were
    all intact -- two resumed sessions in the 12-hour run ended degraded for
    exactly that harmless divergence.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.best_publisher = runner_module.BestResultPublisher(str(workspace))
    head = loop._git("rev-parse", "HEAD").strip()
    loop.best_publisher.publish(
        campaign_id="campaign",
        session_index=1,
        experiment_id="experiment-one",
        iteration=1,
        commit_hash=head,
        plan="prior keep",
        baseline_wall_ms=1.0,
        best_wall_ms=0.8,
        mean_case_speedup=1.4,
        search_start_mean_case_speedup=1.0,
        snr_db=None,
        validation_text="canonical correctness passed",
        benchmark={"median_ms": 0.8},
        changed_files=["kernel.py"],
        patch="prior patch\n",
    )
    loop.run_state = RunState(campaign_id="campaign", session_index=2)
    # The recomputed identity a resumed session would carry into the republish.
    loop.run_state.last_experiment_id = "experiment-two"
    loop.run_state.baseline_wall_ms = 1.0
    loop.run_state.best = BestRecord(
        iteration=1,
        wall_ms=0.8,
        mean_case_speedup=1.4,
        commit_hash=head,
        source="iteration",
    )
    republished: list[dict] = []
    monkeypatch.setattr(
        loop.best_publisher,
        "publish",
        lambda **kwargs: republished.append(kwargs),
    )

    loop._reconcile_best_publication()

    assert republished == []
    assert loop.persistence_degraded is False
    assert loop.persistence_errors == []


def test_reconcile_republishes_a_best_a_stale_manifest_does_not_name(
    tmp_path,
    monkeypatch,
):
    """A manifest that names a different commit is exactly what reconcile repairs."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.best_publisher = runner_module.BestResultPublisher(str(workspace))
    loop.archive = runner_module.CandidateArchive(str(workspace), loop.ic.kernel_file)
    stale_commit = loop._git("rev-parse", "HEAD").strip()
    loop.best_publisher.publish(
        campaign_id="campaign",
        session_index=1,
        experiment_id="experiment-one",
        iteration=1,
        commit_hash=stale_commit,
        plan="earlier keep",
        baseline_wall_ms=1.0,
        best_wall_ms=0.9,
        mean_case_speedup=1.1,
        search_start_mean_case_speedup=1.0,
        snr_db=None,
        validation_text="canonical correctness passed",
        benchmark={"median_ms": 0.9},
        changed_files=["kernel.py"],
        patch="earlier patch\n",
    )
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 3\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "newer best"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = loop._git("rev-parse", "HEAD").strip()
    loop.run_state = RunState(campaign_id="campaign", session_index=2)
    loop.run_state.baseline_wall_ms = 1.0
    loop.run_state.best = BestRecord(
        iteration=1,
        wall_ms=0.7,
        mean_case_speedup=1.5,
        commit_hash=head,
        source="iteration",
    )
    republished: list[dict] = []

    def _record(**kwargs):
        republished.append(kwargs)
        return {"iteration": kwargs["iteration"]}

    monkeypatch.setattr(loop.best_publisher, "publish", _record)

    loop._reconcile_best_publication()

    assert len(republished) == 1
    assert republished[0]["commit_hash"] == head
    assert loop.persistence_degraded is False


def test_intervention_reset_is_persisted_before_agent_runs(tmp_path, monkeypatch):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=1,
        resume=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "test-loop"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    store = LoopStateStore(str(workspace))
    state = RunState(
        baseline_case_times={"case": 1.0},
        best_case_times={"case": 0.8},
        best=BestRecord(
            iteration=2,
            wall_ms=0.8,
            mean_case_speedup=1.25,
            commit_hash=head,
        ),
        phase=PHASE_STALLED,
    )
    state.stall.no_improvement_iters = 1
    state.stall.last_supervisor_iter = -10_000
    store.save(state)

    async def supervisor(**_kwargs):
        return (
            "# Supervisor Ruling\n\n"
            "The prior hard-floor conclusion was unsupported. "
            "Try a different kernel decomposition."
        )

    async def cancelled_agent(*_args, **_kwargs):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.run(agent_fn=cancelled_agent, supervisor_fn=supervisor))

    persisted = LoopStateStore(str(workspace)).load()
    assert loop.run_state.stall.no_improvement_iters == 0
    assert loop.run_state.stall.last_supervisor_iter == 1
    assert loop.run_state.phase != PHASE_STALLED
    assert persisted.stall.no_improvement_iters == 0
    assert persisted.stall.last_supervisor_iter == 1


def test_empty_supervisor_expires_prior_ruling_without_resetting_stall(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=1,
        session_count=3,
    )
    loop.ic.supervise_cooldown = 3
    histories = []
    supervisor_calls = 0
    agent_calls = 0
    stale_ruling = "Continue the stale memory-only direction."

    async def empty_supervisor(**_kwargs):
        nonlocal supervisor_calls
        supervisor_calls += 1
        return ""

    async def editing_agent(kernel_path, history, session_sink):
        nonlocal agent_calls
        agent_calls += 1
        histories.append(history)
        if agent_calls == 1:
            loop._supervisor_ruling = stale_ruling
            ruling_path = runner_module.latest_supervisor_ruling_path(str(workspace))
            ruling_path.parent.mkdir(parents=True, exist_ok=True)
            ruling_path.write_text(stale_ruling)
        session_sink["plan"] = "repeat candidate"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "candidate"

    async def reverted_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="passed",
            wall_ms=1.1,
            mean_case_speedup=0.9,
            kept=False,
            bench_detail={"case_times": {"case": 1.1}},
        )

    monkeypatch.setattr(loop, "run_one_iteration", reverted_iteration)
    asyncio.run(
        loop.run(
            agent_fn=editing_agent,
            supervisor_fn=empty_supervisor,
        )
    )

    assert loop.monitor.intervention_count == 0
    assert loop.run_state.stall.no_improvement_iters == 3
    assert supervisor_calls == 1
    assert loop.run_state.stall.last_supervisor_attempt_iter == 2
    assert loop.run_state.stall.last_supervisor_iter == 0
    assert "Mode: DIVERSIFY" in histories[1]
    assert stale_ruling not in histories[1]
    assert loop._supervisor_ruling == ""
    assert not runner_module.latest_supervisor_ruling_path(str(workspace)).exists()


def test_free_form_supervisor_ruling_still_creates_fresh_plan_each_iteration(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        supervise_after=1,
        session_count=3,
    )
    loop.ic.supervise_cooldown = 2
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    orchestration_calls = []
    histories = []
    supervisor_evidence = []
    ruling = (
        "\n# Supervisor Ruling\n\n"
        "The memory path still has measured headroom. The earlier lesson's "
        "hard-floor conclusion is not supported by the recorded attempts.\n"
    )

    async def orchestration(*, iteration, **_kwargs):
        orchestration_calls.append(iteration)
        plan_path = workspace / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}" / "optimization_plan.md"
        plan_path.parent.mkdir(parents=True)
        plan_path.write_text(f"# Optimization plan {iteration}\n")
        loop._latest_optimization_plan_path = str(plan_path)
        return plan_path, ""

    async def supervisor(**kwargs):
        supervisor_evidence.append(kwargs["evidence_context"])
        return ruling

    async def editing_agent(kernel_path, history, session_sink):
        histories.append(history)
        session_sink["plan"] = "continue memory path"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "implemented another memory milestone"

    async def reverted_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="passed",
            wall_ms=1.1,
            mean_case_speedup=0.9,
            kept=False,
            bench_detail={"case_times": {"case": 1.1}},
        )

    monkeypatch.setattr(loop, "_run_orchestration", orchestration)
    monkeypatch.setattr(loop, "run_one_iteration", reverted_iteration)
    asyncio.run(
        loop.run(
            agent_fn=editing_agent,
            orchestration_service=object(),
            supervisor_fn=supervisor,
        )
    )

    assert orchestration_calls == [1, 2, 3]
    assert len(histories) == 3
    assert "iter_001/optimization_plan.md" in histories[0]
    assert "iter_002/optimization_plan.md" in histories[1]
    assert "iter_003/optimization_plan.md" in histories[2]
    assert "Mode: DIVERSIFY" in histories[1]
    assert "Latest Supervisor Ruling" in histories[1]
    assert "hard-floor conclusion is not supported" in histories[1]
    assert "hard-floor conclusion is not supported" in histories[2]
    assert len(supervisor_evidence) == 1
    evidence = json.loads(supervisor_evidence[0])
    assert evidence["latest_optimization_plan"].endswith("iter_001/optimization_plan.md")
    assert "orchestration_context" in evidence
    assert evidence["orchestration_context"]["search_policy"]["mode"] == "EXPLOIT"
    assert evidence["artifact_paths"]["latest_lesson"].endswith("lessons/iter_001.md")
    assert "latest_handoff" not in evidence["artifact_paths"]

    supervisor_root = workspace / "forge_experiments" / "supervisor"
    assert (supervisor_root / "latest.md").read_text() == ruling
    interaction = (supervisor_root / "intervention_iter_002.md").read_text()
    assert "source: injected callback" in interaction
    assert ruling in interaction


class _OrchestrationTestBackend:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def run(self, _spec, usage=None):
        self.calls += 1
        return self.results.pop(0)


def _runner_orchestration_service(
    orchestration_backend,
    *,
    specialist_result: AgentRunResult | None = None,
):
    definition = SpecialistDefinition(
        role_id="memory",
        description="Memory specialist",
        instructions="Analyze memory access behavior.",
    )
    specialist_backend = _OrchestrationTestBackend([specialist_result or AgentRunResult(text="Use vector loads.")])
    return OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=definition,
                    backend=specialist_backend,
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": definition},
    )


def _orchestration_api_failure() -> AgentRunResult:
    return AgentRunResult(
        text="SDK error text must not become a dispatch plan",
        end_reason="api_error",
        stderr_tail="gateway unavailable",
    )


def test_orchestration_failure_skips_implementer_without_fallback(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    implementer_called = False

    async def orchestration(**_kwargs):
        return None, "synthesis failed"

    async def implementer(*_args, **_kwargs):
        nonlocal implementer_called
        implementer_called = True
        return "must not run"

    monkeypatch.setattr(loop, "_run_orchestration", orchestration)
    results = asyncio.run(
        loop.run(
            agent_fn=implementer,
            orchestration_service=object(),
            supervisor_fn=_unused_supervisor,
        )
    )

    assert implementer_called is False
    assert results[0].validation_summary.startswith("ORCHESTRATION ERROR")
    state = LoopStateStore(str(workspace)).load()
    assert state.cumulative.orchestration_errors == 1
    assert state.stall.no_improvement_iters == 0


def test_consecutive_orchestration_errors_open_circuit(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        session_count=5,
    )
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    backend = _OrchestrationTestBackend([_orchestration_api_failure() for _ in range(3)])
    service = _runner_orchestration_service(backend)

    async def implementer(*_args, **_kwargs):
        raise AssertionError("Implementer must not run without a plan")

    results = asyncio.run(
        loop.run(
            agent_fn=implementer,
            orchestration_service=service,
            supervisor_fn=_unused_supervisor,
        )
    )

    state = LoopStateStore(str(workspace)).load()
    assert len(results) == 3
    assert backend.calls == 3
    assert state.orchestration_error_streak == 3
    assert state.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN
    assert state.termination_reason == "orchestration_failed"
    assert state.cumulative.orchestration_errors == 3


def test_orchestration_failed_resume_allows_one_half_open_probe(
    tmp_path,
    monkeypatch,
):
    first, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        session_count=5,
    )
    runtime_config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    first.config = runtime_config
    first_backend = _OrchestrationTestBackend([_orchestration_api_failure() for _ in range(3)])
    asyncio.run(
        first.run(
            agent_fn=_no_change_agent,
            orchestration_service=_runner_orchestration_service(first_backend),
            supervisor_fn=_unused_supervisor,
        )
    )

    failed_probe = IterationLoop(
        first.ic,
        first.tracker,
        config=runtime_config,
        evolver=_NoopEvolver(),
        resume=True,
    )
    monkeypatch.setattr(
        failed_probe,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(failed_probe.results) < 5 else 0.0,
    )
    failed_backend = _OrchestrationTestBackend([_orchestration_api_failure()])
    failed_results = asyncio.run(
        failed_probe.run(
            agent_fn=_no_change_agent,
            orchestration_service=_runner_orchestration_service(failed_backend),
            supervisor_fn=_unused_supervisor,
        )
    )

    reopened = LoopStateStore(str(workspace)).load()
    assert len(failed_results) == 1
    assert failed_backend.calls == 1
    assert reopened.orchestration_error_streak == 4
    assert reopened.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_OPEN
    assert reopened.termination_reason == "orchestration_failed"

    successful_probe = IterationLoop(
        first.ic,
        first.tracker,
        config=runtime_config,
        evolver=_NoopEvolver(),
        resume=True,
    )
    monkeypatch.setattr(
        successful_probe,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(successful_probe.results) < 1 else 0.0,
    )
    successful_backend = _OrchestrationTestBackend(
        [
            AgentRunResult(
                text=json.dumps(
                    {
                        "assignments": [
                            {
                                "role_id": "memory",
                                "target_case_ids": ["case"],
                                "reason": "Inspect memory access.",
                            }
                        ]
                    }
                )
            ),
            AgentRunResult(text="# Optimization plan\nVectorize global loads."),
        ]
    )
    successful_results = asyncio.run(
        successful_probe.run(
            agent_fn=_no_change_agent,
            orchestration_service=_runner_orchestration_service(successful_backend),
            supervisor_fn=_unused_supervisor,
        )
    )

    closed = LoopStateStore(str(workspace)).load()
    assert len(successful_results) == 1
    assert successful_backend.calls == 2
    assert closed.orchestration_error_streak == 0
    assert closed.orchestration_circuit_state == ORCHESTRATION_CIRCUIT_CLOSED
    assert closed.termination_reason == "budget_exhausted"


def test_two_loop_instances_resume_global_iteration_and_fresh_budget(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    first_state = LoopStateStore(str(workspace)).load()
    first_experiment = first.experiment
    assert first_state.session_index == 1
    assert first_state.session_status == SESSION_PAUSED
    assert first_state.next_iteration == 2
    assert first_experiment is not None

    second = IterationLoop(
        first.ic,
        first.tracker,
        config=object(),
        evolver=_NoopEvolver(),
        resume=True,
    )
    monkeypatch.setattr(
        second,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(second.results) < 1 else 0.0,
    )
    second_results = asyncio.run(second.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    resumed = LoopStateStore(str(workspace)).load()
    assert [result.iteration for result in second_results] == [2]
    assert resumed.campaign_id == first_state.campaign_id
    assert resumed.session_index == 2
    assert resumed.session_status == SESSION_PAUSED
    assert resumed.next_iteration == 3
    assert resumed.cumulative.iterations == 2
    assert second.experiment is not None
    assert second.experiment.parent_experiment_id == first_experiment.experiment_id
    assert second.experiment.segment_index == 2
    baseline_events = [
        event for event in LoopStateStore(str(workspace)).read_events() if event["type"] == "baseline_measured"
    ]
    assert len(baseline_events) == 1


def test_resume_advances_past_abruptly_started_event(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(agent_fn=_no_change_agent))
    store = LoopStateStore(str(workspace))
    state = store.load()
    interrupted_iteration = state.next_iteration
    store.append_event(
        make_event(
            "iteration_started",
            interrupted_iteration,
            phase=state.phase,
        )
    )

    resumed = IterationLoop(
        first.ic,
        first.tracker,
        config=object(),
        evolver=_NoopEvolver(),
        resume=True,
    )
    monkeypatch.setattr(
        resumed,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(resumed.results) < 1 else 0.0,
    )
    results = asyncio.run(resumed.run(agent_fn=_no_change_agent))

    assert [result.iteration for result in results] == [interrupted_iteration + 1]
    assert LoopStateStore(str(workspace)).load().next_iteration == (interrupted_iteration + 2)


def test_validated_warm_start_is_immediately_recoverable_without_keep(
    tmp_path,
    monkeypatch,
):
    """Adopt complete warm-start scoring evidence as iteration-zero state."""
    loop, workspace = _make_loop(
        tmp_path,
        monkeypatch,
        baseline_case_times={},
    )
    assert loop._baseline_case_times == {}
    loop.ic.baseline_case_times = {"case": 1.0}
    base_commit = loop._git("rev-parse", "HEAD").strip()
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "kb warm-start: apply prior"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    warm_commit = loop._git("rev-parse", "HEAD").strip()
    loop.ic.campaign_base_commit = base_commit
    loop.ic.baseline_wall_ms = 1.0
    loop.ic.pristine_baseline_wall_ms = 1.0
    loop.ic.warm_start_wall_ms = 0.8
    loop.ic.warm_start_mean_case_speedup = 1.25
    loop.ic.warm_start_bench = {
        "case_times": {"case": 0.8},
        "unscored_cases": ["noisy"],
    }
    loop.ic.warm_start_commit = warm_commit
    loop.ic.warm_start_solution_slug = "kernelforge-exp/op/run"

    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
        )
    )

    state = LoopStateStore(str(workspace)).load()
    manifest = json.loads((workspace / "forge_experiments" / "best" / "manifest.json").read_text())
    assert state.best.source == "warm_start"
    assert state.best.commit_hash == warm_commit
    assert state.best.wall_ms == 0.8
    assert manifest["iteration"] == 0
    assert manifest["commit_hash"] == warm_commit
    assert manifest["pristine_baseline_ms"] == 1.0
    assert manifest["search_start_ms"] == 0.8
    assert manifest["total_improved"] is True
    assert manifest["incremental_improved"] is False
    assert loop._baseline_case_times == {"case": 1.0}
    assert loop._best_case_times == {"case": 0.8}
    assert loop._unscored_cases == {"noisy"}
    assert state.baseline_case_times == {"case": 1.0}
    assert state.best_case_times == {"case": 0.8}


def test_validated_warm_start_reuses_cli_prepublication(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    base_commit = loop._git("rev-parse", "HEAD").strip()
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "kb warm-start: apply prior"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    warm_commit = loop._git("rev-parse", "HEAD").strip()
    patch = subprocess.run(
        ["git", "diff", base_commit, warm_commit, "--", "."],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    publisher = runner_module.BestResultPublisher(str(workspace))
    manifest = publisher.publish(
        campaign_id="warm-start:kernelforge-exp/op/run",
        session_index=0,
        experiment_id="caller-run",
        iteration=0,
        commit_hash=warm_commit,
        plan="apply prior solution kernelforge-exp/op/run",
        baseline_wall_ms=1.0,
        search_start_ms=0.8,
        best_wall_ms=0.8,
        mean_case_speedup=1.25,
        search_start_mean_case_speedup=1.25,
        snr_db=None,
        validation_text="validated KB warm-start passed canonical correctness",
        benchmark={"median_ms": 0.8, "warm_start": True},
        changed_files=["kernel.py"],
        patch=patch,
    )

    loop.ic.campaign_base_commit = base_commit
    loop.ic.baseline_wall_ms = 1.0
    loop.ic.pristine_baseline_wall_ms = 1.0
    loop.ic.warm_start_wall_ms = 0.8
    loop.ic.warm_start_mean_case_speedup = 1.25
    loop.ic.warm_start_commit = warm_commit
    loop.ic.warm_start_solution_slug = "kernelforge-exp/op/run"
    loop.ic.warm_start_publication = {
        "baseline_ms": 1.0,
        "best_ms": 0.8,
        "mean_case_speedup": 1.25,
        "best_iteration": 0,
        "best_commit": warm_commit,
        "best_manifest": str(workspace / "forge_experiments" / "best" / "manifest.json"),
    }

    def duplicate_publication(*_args, **_kwargs):
        pytest.fail("runner must not republish a CLI-published warm-start")

    monkeypatch.setattr(loop, "_publish_best_result", duplicate_publication)
    asyncio.run(
        loop.run(
            agent_fn=_no_change_agent,
            supervisor_fn=_unused_supervisor,
        )
    )

    persisted_manifest = json.loads((workspace / "forge_experiments" / "best" / "manifest.json").read_text())
    assert persisted_manifest == manifest
    assert not list((workspace / "forge_experiments" / "best").glob(".iter_000.corrupt-*"))


def test_keep_with_missing_diagnostic_wall_time_completes(
    tmp_path,
    monkeypatch,
    capsys,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "case-score-only candidate"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "verified improvement"

    async def successful_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="canonical validation passed",
            wall_ms=None,
            mean_case_speedup=1.1,
            snr_db=40.0,
            kept=True,
            bench_detail={
                "success": True,
                "median_ms": None,
                "case_times": {"case": 0.9},
            },
        )

    monkeypatch.setattr(loop, "run_one_iteration", successful_iteration)

    results = asyncio.run(loop.run(agent_fn=editing_agent))

    assert len(results) == 1
    assert results[0].kept is True
    assert results[0].wall_ms is None
    assert "raw mean=unavailable" in capsys.readouterr().out
    assert not (workspace / "forge_experiments" / "pending_keep.json").exists()
    state = LoopStateStore(str(workspace)).load()
    assert state.best.iteration == 1
    assert state.best.wall_ms is None
    assert state.best.mean_case_speedup == pytest.approx(1.1)


@pytest.mark.parametrize("failure_mode", ["none", "raise"])
def test_keep_archive_failure_retains_pending_journal(
    tmp_path,
    monkeypatch,
    failure_mode,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "archive failure candidate"
        Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "verified improvement"

    async def successful_iteration(iteration, plan=""):
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="canonical validation passed",
            wall_ms=0.9,
            mean_case_speedup=1.1,
            snr_db=40.0,
            kept=True,
        )

    def fail_record(_archive, _record):
        if failure_mode == "raise":
            raise OSError("simulated candidate archive failure")
        return None

    monkeypatch.setattr(loop, "run_one_iteration", successful_iteration)
    monkeypatch.setattr(runner_module.CandidateArchive, "record", fail_record)

    asyncio.run(loop.run(agent_fn=editing_agent))

    pending_path = workspace / "forge_experiments" / "pending_keep.json"
    assert not pending_path.is_file()
    assert loop.persistence_degraded is True
    assert any("archive derived KEEP view" in item for item in loop.persistence_errors)
    state = LoopStateStore(str(workspace)).load()
    assert state.best.iteration == 1
    assert state.cumulative.kept == 1


def test_recovered_keep_archive_none_retains_pending_journal(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.archive = runner_module.CandidateArchive(
        str(workspace),
        loop.ic.kernel_file,
    )
    pending = {
        "iteration": 1,
        "wall_ms": 0.9,
        "kernel_source": "def kernel():\n    return 2\n",
        "patch": "candidate patch\n",
        "validation_text": "canonical validation passed",
    }
    loop._pending_keep_path.write_text(json.dumps(pending))
    monkeypatch.setattr(loop.archive, "record", lambda _record: None)

    with pytest.raises(RuntimeError, match="recover candidate archive"):
        loop._archive_pending_keep(pending, "committed-hash")

    assert loop._pending_keep_path.is_file()


def test_fresh_run_rejects_existing_campaign_without_modifying_state(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
    state_path = workspace / "forge_experiments" / "run_state.json"
    before = state_path.read_bytes()

    duplicate_fresh = IterationLoop(
        first.ic,
        first.tracker,
        config=object(),
        evolver=_NoopEvolver(),
    )
    with pytest.raises(ValueError, match="--resume"):
        asyncio.run(
            duplicate_fresh.run(
                agent_fn=_no_change_agent,
                supervisor_fn=_unused_supervisor,
            )
        )

    assert state_path.read_bytes() == before


def test_a_rejected_fresh_run_persists_no_pr_references(tmp_path, monkeypatch):
    """Deferred PR writes must not survive an invocation the guard rejects."""
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    def tree() -> dict:
        """Every workspace file, git included, so nothing can slip through."""
        return {
            str(path.relative_to(workspace)): path.read_bytes()
            for path in sorted(workspace.rglob("*"))
            if path.is_file()
        }

    before = tree()
    duplicate_fresh = IterationLoop(
        replace(
            first.ic,
            pr_kb_snapshot={"entries": {"ROCm/aiter#1@sha:1": {}}},
            pr_kb_event={"position": "A", "reason": "ok"},
        ),
        first.tracker,
        config=object(),
        evolver=_NoopEvolver(),
    )
    with pytest.raises(ValueError, match="--resume"):
        asyncio.run(
            duplicate_fresh.run(
                agent_fn=_no_change_agent,
                supervisor_fn=_unused_supervisor,
            )
        )

    assert tree() == before
    assert not (workspace / "forge_experiments" / "pr_refs").exists()


def test_a_snapshot_write_failure_records_degradation_and_continues(
    tmp_path,
    monkeypatch,
    capsys,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic.pr_kb_snapshot = {"entries": {"ROCm/aiter#1@sha:1": {}}}
    loop.ic.pr_kb_event = {"position": "A", "reason": "ok"}

    def fail_snapshot(_workspace_dir, _snapshot):
        """Simulate a local sidecar write failure after the guard."""
        raise OSError("disk full")

    monkeypatch.setattr(
        "kernelforge.knowledge.pr_monitor_refs.commit_snapshot",
        fail_snapshot,
    )

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    events = [event for event in LoopStateStore(str(workspace)).read_events() if event["type"] == "pr_refs_refreshed"]
    assert len(events) == 1
    assert events[0]["reason"] == "ok"
    assert events[0]["degraded_reason"] == "local_failure"
    assert "warning: snapshot not persisted" in capsys.readouterr().out


def test_a_pr_event_write_failure_cannot_abort_the_campaign(
    tmp_path,
    monkeypatch,
    capsys,
):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic.pr_kb_event = {"position": "A", "reason": "ok"}
    append_event = LoopStateStore.append_event

    def fail_pr_event(self, event):
        """Fail only the optional PR event and retain normal loop persistence."""
        if event.get("type") == "pr_refs_refreshed":
            raise OSError("events unavailable")
        append_event(self, event)

    monkeypatch.setattr(LoopStateStore, "append_event", fail_pr_event)

    asyncio.run(loop.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))

    assert "warning: event not recorded" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field_name", "field_value", "error"),
    [
        ("git_branch", "different-branch", "branch mismatch"),
        ("task_type", "different-task", "task fingerprint mismatch"),
    ],
)
def test_resume_rejects_identity_mismatch_without_modifying_state(
    tmp_path,
    monkeypatch,
    field_name,
    field_value,
    error,
):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
    state_path = workspace / "forge_experiments" / "run_state.json"
    before = state_path.read_bytes()
    experiment_count = len(first.tracker.list_experiments())
    bad_config = replace(first.ic, **{field_name: field_value})

    mismatched = IterationLoop(
        bad_config,
        first.tracker,
        config=object(),
        evolver=_NoopEvolver(),
        resume=True,
    )
    with pytest.raises(ValueError, match=error):
        asyncio.run(
            mismatched.run(
                agent_fn=_no_change_agent,
                supervisor_fn=_unused_supervisor,
            )
        )

    assert state_path.read_bytes() == before
    assert len(first.tracker.list_experiments()) == experiment_count


def test_resume_rejects_head_mismatch_without_modifying_state(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
    state_path = workspace / "forge_experiments" / "run_state.json"
    before = state_path.read_bytes()

    (workspace / "kernel.py").write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "unexpected external change"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    mismatched = IterationLoop(
        first.ic,
        first.tracker,
        config=object(),
        evolver=_NoopEvolver(),
        resume=True,
    )
    with pytest.raises(ValueError, match="HEAD mismatch"):
        asyncio.run(
            mismatched.run(
                agent_fn=_no_change_agent,
                supervisor_fn=_unused_supervisor,
            )
        )

    assert state_path.read_bytes() == before


def test_checkpoint_llm_usage_is_idempotent_with_fake_usage_and_tracker(
    tmp_path,
    monkeypatch,
):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    totals = {
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_creation_input_tokens": 4,
        "cache_read_input_tokens": 8,
        "total_cost_usd": 0.25,
        "calls": 1,
    }

    class FakeUsage:
        def totals(self):
            return dict(totals)

    class FakeTracker:
        def __init__(self):
            self.checkpoints = []

        def set_llm_usage(self, experiment_id, usage):
            self.checkpoints.append((experiment_id, dict(usage)))

    tracker = FakeTracker()
    loop._usage = FakeUsage()
    loop.tracker = tracker
    loop.experiment = type("FakeExperiment", (), {"experiment_id": "exp-1"})()

    loop._checkpoint_llm_usage()
    loop._checkpoint_llm_usage()

    assert loop.llm_usage == totals
    assert tracker.checkpoints == [("exp-1", totals), ("exp-1", totals)]


def test_checkpoint_llm_usage_is_best_effort_and_accepts_token_only_totals(
    tmp_path,
    monkeypatch,
):
    loop, _workspace = _make_loop(tmp_path, monkeypatch)
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_cost_usd": 0.0,
        "calls": 0,
    }

    class FakeUsage:
        def totals(self):
            return dict(totals)

    class FailingTracker:
        def __init__(self):
            self.calls = 0

        def set_llm_usage(self, _experiment_id, _usage):
            self.calls += 1
            raise OSError("simulated usage checkpoint failure")

    tracker = FailingTracker()
    loop._usage = FakeUsage()
    loop.tracker = tracker
    loop.experiment = type("FakeExperiment", (), {"experiment_id": "exp-1"})()

    loop._checkpoint_llm_usage()
    assert tracker.calls == 0

    totals["input_tokens"] = 11
    loop._checkpoint_llm_usage()

    assert loop.llm_usage == totals
    assert tracker.calls == 1


def test_case_metric_fails_closed_on_incomplete_candidate_coverage():
    loop = IterationLoop(
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            baseline_wall_ms=10.0,
        ),
        tracker=object(),
        config=object(),
        evolver=object(),
    )
    loop._baseline_case_times = {"small": 2.0, "large": 8.0}
    bench = {
        "success": True,
        "median_ms": 7.0,
        "case_times": {"small": 1.0},
        "measurements": [
            {
                "success": True,
                "case_times": {"small": 1.0},
                "unscored_cases": [],
            }
            for _ in range(3)
        ],
    }

    loop._apply_mean_case_speedup_metric(bench)

    assert bench["success"] is False
    assert bench["mean_case_speedup"] is None
    assert bench["case_coverage_complete"] is False
    assert "large" in bench["message"]


def test_mean_case_speedup_metric_preserves_raw_mean_for_diagnostics():
    loop = IterationLoop(
        IterationConfig(
            kernel_file="kernel.py",
            driver_script="driver.py",
            baseline_wall_ms=5.0,
        ),
        tracker=object(),
        config=object(),
        evolver=object(),
    )
    loop._baseline_case_times = {"small": 1.0, "large": 9.0}
    bench = {
        "success": True,
        "median_ms": 5.25,
        "case_times": {"small": 0.5, "large": 10.0},
        "measurements": [
            {
                "success": True,
                "case_times": {"small": 0.5, "large": 10.0},
                "unscored_cases": [],
            }
            for _ in range(3)
        ],
    }

    loop._apply_mean_case_speedup_metric(bench)

    assert bench["median_ms"] == 5.25
    assert bench["mean_case_speedup"] == pytest.approx(1.45)


# ── lesson recording ──────────────────────────────────────────────────────────


def _lesson_result(**overrides) -> IterationResult:
    base = dict(
        iteration=3,
        duration_sec=1.0,
        validation_passed=True,
        validation_summary="all stages passed",
        wall_ms=1.5,
        snr_db=41.0,
        kept=False,
        session_end_reason="turn_cap",
    )
    base.update(overrides)
    return IterationResult(**base)


def _attach_lessons(loop, workspace):
    from kernelforge.loop.lessons import LessonStore

    loop.lessons = LessonStore(str(workspace))
    return loop.lessons


async def _fake_summarizer(_prompt: str) -> str:
    return "tried three tile shapes\n- [worse] BLOCK_N 128 | 0.94x"


def test_record_lesson_writes_narrative_and_outcome(tmp_path, monkeypatch):
    """Both authors land in one document: the session's, then the loop's."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    loop.best_wall_ms = 1.2

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
            },
        )
    )

    text = store.read(3)
    assert "BLOCK_N 128" in text
    assert "OUTCOME: REVERT_PERF" in text
    assert "wall 1.5000 ms vs best 1.2000 ms" in text
    assert "session ended: turn_cap" in text


def test_record_lesson_falls_back_to_gate_findings(tmp_path, monkeypatch):
    """A provider that cannot resume still leaves the gate's rejections behind."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=4,
            result=_lesson_result(iteration=4),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "x",
                "findings": "compile error: invalid cast\n---\ncorrect but not faster",
            },
            diff_summary="kernel.py | 3 +-",
        )
    )

    text = store.read(4)
    assert "invalid cast" in text
    assert "correct but not faster" in text
    assert "kernel.py | 3 +-" in text
    assert "OUTCOME: REVERT_PERF" in text


def test_record_lesson_outcome_only_records_machine_verdict(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=4,
            result=_lesson_result(iteration=4),
            decision="REVERT_PERF",
            # Nothing to summarize and nothing observed: no narrative is possible.
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "x",
            },
            diff_summary="",
        )
    )

    # An outcome-only document still says what the outcome was measured under.
    assert store.read(4).strip().startswith("SCOPE: measured on ")
    assert "OUTCOME:" in store.read(4)


_THREE_CASES = {"decode-t1": 1.0, "decode-t64": 2.0, "prefill-t16384": 3.0}


async def _scoped_summarizer(_prompt: str) -> str:
    return "swept split-K on decode-t1; every point slower\nHELD-FIXED: BLOCK_N=16, num_warps=8"


def test_record_lesson_scopes_the_negative_to_the_measured_suite(tmp_path, monkeypatch):
    """The whole suite was measured, and the session said what it pinned."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.cases == ("decode-t1", "decode-t64", "prefill-t16384")
    assert scope.held_fixed == (("BLOCK_N", "16"), ("num_warps", "8"))
    assert scope.lane_restricted is False


async def _unreachable_summarizer(_prompt: str) -> str:
    return (
        "the transposed read is the only real fix and this build CANNOT emit "
        "it\n"
        "HELD-FIXED: BLOCK_N=16\n"
        "DISPROOF: untested — a build-only screen of the one instruction "
        "would settle it"
    )


def test_record_lesson_carries_an_undisproven_cannot_into_the_scope(tmp_path, monkeypatch, capsys):
    """The claim closes nothing until the experiment it names is actually run."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _unreachable_summarizer,
                "plan": "transpose the LDS read",
            },
        )
    )

    from kernelforge.loop.lessons import UNDISPROVEN_CLAIM

    assert store.scope_of(3).disproof == UNDISPROVEN_CLAIM
    assert "feasibility claim not disproved" in store.read(3)
    assert "without running the experiment" in capsys.readouterr().out
    assert "VALIDITY: RE-OPENABLE (undisproven feasibility claim)" in (
        store.render_for_prompt(current_cases=tuple(_THREE_CASES), kernel_source="BLOCK_N = 16\n")
    )


async def _refuted_summarizer(_prompt: str) -> str:
    return (
        "the transposed read is the only real fix and this build CANNOT emit "
        "it\n"
        "HELD-FIXED: BLOCK_N=16\n"
        "DISPROOF: falsified — a build-only screen shows gfx950 assembles "
        "ds_read_b64_tr_b16"
    )


def test_record_lesson_carries_a_refuted_cannot_as_an_open_direction(tmp_path, monkeypatch, capsys):
    """The session's own experiment killed its premise; the axis is reachable.

    Scoring this as an obligation discharged would leave the record IN SCOPE
    and still suppressing the direction the same line proved open.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _refuted_summarizer,
                "plan": "transpose the LDS read",
            },
        )
    )

    from kernelforge.loop.lessons import is_claim_disproved

    assert is_claim_disproved(store.scope_of(3).disproof)
    assert "feasibility claim disproved by a build-only screen" in store.read(3)
    assert "shown reachable" in capsys.readouterr().out
    rendered = store.render_for_prompt(current_cases=tuple(_THREE_CASES), kernel_source="BLOCK_N = 16\n")
    assert "VALIDITY: RE-OPEN (feasibility claim disproved)" in rendered
    assert "VALIDITY: IN SCOPE" not in rendered


def test_record_lesson_keeps_a_restricted_lanes_scope(tmp_path, monkeypatch):
    """A lane told to touch one case measured one case; the ban stays there."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "waves_per_eu on prefill-t16384 only",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.cases == ("prefill-t16384",)
    assert scope.lane_restricted is True


def test_record_lesson_says_when_no_premise_was_recorded(tmp_path, monkeypatch):
    """A session that pinned nothing it could name leaves a re-openable record."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    assert store.scope_of(3).held_fixed == ()
    assert "held fixed (not recorded)" in store.read(3)
    assert "VALIDITY: RE-OPENABLE" in store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )


def test_an_unreadable_kernel_is_not_reported_as_a_kernel_that_dropped_it(tmp_path, monkeypatch, capsys):
    """An I/O failure must not become a factual claim about the source.

    ``_read_source_file`` collapses an unreadable file to "", which scans as a
    kernel that assigns nothing: every pinned constant reported missing,
    every stored negative re-opened, and no way to tell that from a kernel that
    really did move on.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    Path(loop.ic.kernel_file).unlink()

    assert loop._read_kernel_source() == ""
    assert loop._kernel_source_for_scope() is None
    assert "kernel source unreadable" in capsys.readouterr().out

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=loop._kernel_source_for_scope(),
    )
    assert "were not checked against the current kernel" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_an_unreadable_kernel_reaches_the_prompt_as_unchecked(tmp_path, monkeypatch):
    """The loop's own prompt build must carry the distinction, not just the store."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2, baseline_case_times=_THREE_CASES)
    _stub_measurement(monkeypatch, walls=[0.5, 0.9])

    kernel = Path(loop.ic.kernel_file)
    prompts: list[str] = []
    edits = {"n": 0}

    async def summarizer(_prompt: str) -> str:
        # The next prompt is built before the next session touches the file.
        kernel.unlink()
        return "swept split-K on decode-t1; every point slower\nHELD-FIXED: BLOCK_N=16"

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        edits["n"] += 1
        kernel.write_text(f"def kernel():\n    return {edits['n'] + 1}\n")
        session_sink["plan"] = "sweep split-K"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["summarize"] = summarizer
        return "rationale"

    asyncio.run(loop.run(agent_fn=agent_fn))

    assert len(prompts) == 2
    assert "were not checked against the current kernel" in prompts[1]
    assert "not assigned in the kernel source checked" not in prompts[1]


async def _positive_summarizer(_prompt: str) -> str:
    # Complies with the summary prompt's contract: the record itself states
    # that no direction it covers measured worse.
    return "widened the tile on every case; 1.2x, nothing measured worse\nNEGATIVES: none"


def test_a_pin_that_moved_to_a_sibling_file_is_not_reported_as_dropped(tmp_path, monkeypatch):
    """Tile and dispatch constants live outside the anchor kernel.

    The implementer prompt says so itself. Checking only the anchor turns a
    constant that moved into a constant the task no longer assigns, which
    re-opens every negative measured under it.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    sibling = workspace / "tiles.py"
    sibling.write_text('CONFIG = {"BLOCK_N": 16}\nnum_warps = 8\n')
    loop.ic.source_files.append(str(sibling))

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    sources = loop._kernel_source_for_scope()
    assert sources is not None and len(sources) == 2
    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=sources,
    )
    assert "VALIDITY: IN SCOPE" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_a_source_that_cannot_be_parsed_is_not_a_source_that_dropped_the_pin(tmp_path, monkeypatch):
    """A syntax error is "not checked", never "the constant is gone"."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    Path(loop.ic.kernel_file).write_text("def kernel(:\n    return 1\n")

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=loop._kernel_source_for_scope(),
    )
    assert "could not be parsed" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_an_iteration_that_measured_no_negative_is_not_reopenable(tmp_path, monkeypatch):
    """Nothing in the document was closed, so there is nothing to re-open.

    Only the summarizer writes HELD-FIXED:, and only for a direction that
    measured WORSE. Treating its absence as a re-opened premise labels every
    positive iteration RE-OPENABLE and degrades the store to "nothing is
    settled".
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.carries_negative is False
    assert scope.held_fixed == ()
    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )
    assert "VALIDITY: IN SCOPE" in rendered
    assert "VALIDITY: RE-OPENABLE" not in rendered


def test_an_in_session_gate_rejection_is_a_measured_negative(tmp_path, monkeypatch):
    """A rejection the agent hit is a negative even when the loop kept nothing."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="NO_CHANGES",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,
                "plan": "widen the tile",
                "findings": "correct but not faster: 0.97x\n---\ncompile error",
            },
        )
    )

    assert store.scope_of(3).carries_negative is True
    assert "VALIDITY: RE-OPENABLE" in store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )


async def _in_session_negative_summarizer(_prompt: str) -> str:
    """A KEEP session that measured four regressions on the way there."""
    return (
        "tried split-K=4 (0.91x), BLOCK_N=128 (0.94x) and two waves_per_eu "
        "settings, reverted all four inside the session, then submitted the "
        "tile widening that measured 1.2x\n"
        "NEGATIVES: split-K=4, BLOCK_N=128, waves_per_eu=2/4"
    )


async def _unmarked_summarizer(_prompt: str) -> str:
    """An older document, or a model that ignored the marker contract."""
    return "tried a few things and submitted the tile widening"


def test_a_kept_iteration_that_measured_negatives_in_session_is_reopenable(tmp_path, monkeypatch):
    """The loop kept a candidate; the record says four directions measured worse.

    Those four are invisible to the loop — they were reverted before the
    candidate it measured — so deciding from its own verdict would stamp this
    document "no measured negative" and, with no HELD-FIXED line, render it IN
    SCOPE. That promotes four negatives measured under unknown constants into a
    standing ban. The document's own marker is what the loop has to read.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _in_session_negative_summarizer,
                "plan": "widen the tile",
                # The in-session gate allowed on the first Stop, so it logged
                # nothing: findings cannot catch these either.
                "findings": "",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.carries_negative is True
    assert scope.held_fixed == ()

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )
    assert "VALIDITY: RE-OPENABLE" in rendered
    assert "no measured negative" not in store.read(3)
    assert "the constants it was measured under were not recorded" in rendered


def test_a_document_without_the_marker_is_unknown_not_negative_free(tmp_path, monkeypatch):
    """No marker means nobody answered, and the note must not answer for it."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _unmarked_summarizer,
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).carries_negative is None
    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )
    assert "no measured negative" not in store.read(3)
    assert "whether anything measured worse was not recorded" in rendered
    assert "VALIDITY: RE-OPENABLE" in rendered


def test_a_record_that_never_answered_the_disproof_question_says_so(tmp_path, monkeypatch, capsys):
    """An unanswered question is the state no verdict fires on, so it is printed.

    Nothing reads the prose for "cannot", so a session that ignored the marker
    leaves any feasibility claim in its record resting on the citation rule
    alone. That is the one degradation an operator cannot see in the verdict.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": _unmarked_summarizer,
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).disproof is None
    assert "recorded as unchecked" in capsys.readouterr().out


def test_the_loops_own_negative_overrides_a_record_that_claims_none(tmp_path, monkeypatch):
    """Machine truth beats prose: a REVERT is a negative whatever the text says."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,  # writes "NEGATIVES: none"
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).carries_negative is True


def test_a_crash_is_a_measured_negative(tmp_path, monkeypatch):
    """CRASH starts with no REVERT and leaves no speedup, so only a whitelist sees it.

    A session cut off before the Stop hook also leaves no findings, so nothing
    else in the loop's view would report this iteration as having gone wrong.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(crashed=True, validation_passed=False, mean_case_speedup=None),
            decision="CRASH",
            session_sink={
                "session_started": True,
                "summarize": _positive_summarizer,  # writes "NEGATIVES: none"
                "plan": "widen the tile",
                "findings": "",
            },
        )
    )

    assert store.scope_of(3).carries_negative is True
    assert "VALIDITY: RE-OPENABLE" in store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source="",
    )


def test_a_build_failure_is_a_measured_negative(tmp_path, monkeypatch):
    loop, _workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    assert loop._loop_measured_a_negative("BUILD_FAILED", _lesson_result(validation_passed=False), {}) is True
    assert loop._loop_measured_a_negative("KEEP", _lesson_result(), {}) is False
    assert loop._loop_measured_a_negative("NO_CHANGES", _lesson_result(mean_case_speedup=None), {}) is False


def test_a_machine_written_document_is_decided_by_the_loops_verdict_alone(tmp_path, monkeypatch):
    """The loop authored it, so the loop's view of it is complete.

    A fallback document contains what the loop observed and nothing else, so
    there is no unseen reverted direction for a marker to report — and no
    summarizer to write one.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(kept=True, mean_case_speedup=1.2),
            decision="KEEP",
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "widen the tile",
                "findings": "",
            },
            diff_summary="kernel.py | 3 +-",
        )
    )

    assert "(no agent summary)" in store.read(3)
    assert store.scope_of(3).carries_negative is False


def test_one_unreadable_sibling_does_not_indict_the_constant_it_holds(tmp_path, monkeypatch, capsys):
    """Part of the declared source went unchecked; say that, do not guess.

    Dropping the unreadable file leaves the survivors looking like the whole
    declared set, so a constant living in the missing one reads as deleted.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    sibling = workspace / "tiles.py"
    sibling.write_text("BLOCK_N = 16\nnum_warps = 8\n")
    loop.ic.source_files.append(str(sibling))

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    sibling.unlink()

    sources = loop._kernel_source_for_scope()
    assert sources is not None and sources[1] is None
    assert "source unreadable" in capsys.readouterr().out

    rendered = store.render_for_prompt(
        current_cases=tuple(_THREE_CASES),
        kernel_source=sources,
    )
    assert "BLOCK_N was not checked" in rendered
    assert "not assigned in the kernel source checked" not in rendered


def test_an_unscored_case_is_not_recorded_as_a_case_that_was_measured(tmp_path, monkeypatch):
    """Scoring excluded the noisy case, so nothing was measured on it."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    loop._unscored_cases = {"decode-t64"}

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    scope = store.scope_of(3)
    assert scope.cases == ("decode-t1", "prefill-t16384")
    assert scope.lane_restricted is False


def test_an_unscored_case_does_not_reopen_a_stored_negative_in_the_prompt(tmp_path, monkeypatch):
    """The render path validates against the scored suite, not every case."""
    from kernelforge.loop.lessons import LessonScope, LessonStore

    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    loop.ic.preloop_baseline_unscored_cases = ["decode-t64"]
    _stub_measurement(monkeypatch, walls=[0.5])

    store = LessonStore(str(workspace))
    store.write(1, "swept split-K; every point slower")
    store.append_scope(
        1,
        LessonScope(
            cases=("decode-t1", "prefill-t16384"),
            held_fixed=(("BLOCK_N", "16"),),
            carries_negative=True,
        ),
    )

    prompts: list[str] = []

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        session_sink["plan"] = "inspect only"
        return "No source change was needed."

    asyncio.run(loop.run(agent_fn=agent_fn))

    assert loop._scored_case_ids() == ["decode-t1", "prefill-t16384"]
    assert prompts and "swept split-K" in prompts[0]
    assert "not measured on decode-t64" not in prompts[0]


def test_a_scope_that_cannot_be_written_says_so(tmp_path, monkeypatch, capsys):
    """A failed append leaves an unscoped document; the operator hears about it."""
    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    monkeypatch.setattr(store, "append_scope", lambda *_a, **_k: False)

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
                "plan": "sweep split-K",
            },
        )
    )

    out = capsys.readouterr().out
    assert "scope not recorded for iter 3" in out
    assert "no held-fixed constants recorded" not in out


def test_handoff_carries_the_scope_of_the_iterations_lesson(tmp_path, monkeypatch):
    """The planner reads handoffs; a refutation quoted from one needs its scope."""
    from kernelforge.loop.handoffs import HandoffStore

    loop, workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times=_THREE_CASES)
    store = _attach_lessons(loop, workspace)
    loop.run_state = RunState()
    loop.handoff_store = HandoffStore(str(workspace))

    asyncio.run(
        loop._record_lesson(
            iteration=3,
            result=_lesson_result(),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _scoped_summarizer,
                "plan": "sweep split-K",
            },
        )
    )
    loop._record_iteration_handoff(
        iteration=3,
        decision="REVERT_PERF",
        optimization_plan_path="",
        session_sink={"plan": "sweep split-K"},
    )

    payload = json.loads(loop.handoff_store.path(3).read_text())
    assert payload["plan"].startswith("sweep split-K\nSCOPE: measured on ")
    assert "BLOCK_N=16" in payload["plan"]
    assert store.scope_of(3) is not None


def test_handoff_plan_is_unchanged_when_no_lesson_was_written(tmp_path, monkeypatch):
    from kernelforge.loop.handoffs import HandoffStore

    loop, workspace = _make_loop(tmp_path, monkeypatch)
    _attach_lessons(loop, workspace)
    loop.run_state = RunState()
    loop.handoff_store = HandoffStore(str(workspace))

    loop._record_iteration_handoff(
        iteration=3,
        decision="NO_CHANGES",
        optimization_plan_path="",
        session_sink={"plan": "inspect only"},
    )

    payload = json.loads(loop.handoff_store.path(3).read_text())
    assert payload["plan"] == "inspect only"
    assert payload["lesson_path"] == ""


def test_record_lesson_skips_an_empty_iteration(tmp_path, monkeypatch):
    """No exploration recorded and no candidate measured -> no document."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=5,
            result=_lesson_result(iteration=5),
            decision="NO_CHANGES",
            session_sink={"session_started": False, "summarize": None},
        )
    )

    assert store.existing_iterations() == []


def test_record_lesson_keeps_outcome_when_no_diff_summary_fails(tmp_path, monkeypatch):
    """A started, cut-off session must not disappear just because it has no diff."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    async def broken(_prompt: str) -> str:
        raise RuntimeError("resume handle expired")

    asyncio.run(
        loop._record_lesson(
            iteration=10,
            result=_lesson_result(
                iteration=10,
                session_end_reason="turn_cap",
                turns=40,
                wall_ms=None,
                snr_db=None,
            ),
            decision="NO_CHANGES",
            session_sink={
                "session_started": True,
                "summarize": broken,
                "findings": "",
                "progress_log": [],
            },
        )
    )

    text = store.read(10)
    assert text.strip() == (
        "SCOPE: measured on case | held fixed (not recorded) | "
        "no measured negative | whether a feasibility claim was disproved "
        "was not recorded\n\n"
        "OUTCOME: NO_CHANGES | session ended: turn_cap | turns 40 | "
        "summary unavailable: RuntimeError: resume handle expired"
    )


def test_record_lesson_uses_progress_when_no_diff_summary_is_unavailable(tmp_path, monkeypatch):
    """Machine-captured provider progress is the no-diff narrative fallback."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=11,
            result=_lesson_result(
                iteration=11,
                session_end_reason="turn_cap",
                turns=40,
            ),
            decision="NO_CHANGES",
            session_sink={
                "session_started": True,
                "summarize": None,
                "plan": "evaluate wider vector loads",
                "findings": "",
                "progress_log": [
                    "tool: Read kernel.py",
                    "progress: not supported by codex backend",
                    "tool: Bash python driver.py --bench BLOCK_N=128",
                ],
            },
        )
    )

    text = store.read(11)
    assert text.startswith("(no agent summary) last observed:")
    assert "Summary unavailable: provider cannot resume the session" in text
    assert "Implementer turns: 40" in text
    assert "Final plan: evaluate wider vector loads" in text
    assert "tool: Read kernel.py" in text
    assert "tool: Bash python driver.py --bench BLOCK_N=128" in text
    assert "not supported by codex" not in text
    assert "OUTCOME: NO_CHANGES" in text


def test_record_lesson_without_an_agent_session(tmp_path, monkeypatch):
    """The baseline measurement path has nothing to summarize."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    asyncio.run(
        loop._record_lesson(
            iteration=6,
            result=_lesson_result(iteration=6),
            decision="KEEP",
            session_sink={},
        )
    )

    assert store.existing_iterations() == []


def test_record_lesson_skips_the_summarizer_with_no_time_left(tmp_path, monkeypatch):
    """With no time to produce it, record the objective outcome alone."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    monkeypatch.setattr(loop, "_time_remaining", lambda: 0.0)

    started = {"n": 0}

    async def counting_summarizer(_prompt: str) -> str:
        started["n"] += 1
        return "should not run"

    asyncio.run(
        loop._record_lesson(
            iteration=7,
            result=_lesson_result(iteration=7),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": counting_summarizer,
            },
        )
    )

    assert started["n"] == 0
    assert "OUTCOME: REVERT_PERF" in store.read(7)


def test_record_lesson_still_summarizes_when_no_session_can_be_admitted(tmp_path, monkeypatch):
    """The last iteration of a session is the one whose record matters most.

    The loop stops admitting implementer sessions well before the clock runs out
    (``budget_reserve_sec``), but the campaign is resumable and its next
    session reads this document. Gating the summary on the session-admission
    reserve silently dropped that handoff record.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    # Below the session-admission reserve, far above what a summary needs.
    monkeypatch.setattr(loop, "_time_remaining", lambda: 600.0)
    assert loop._is_budget_exhausted() is True

    asyncio.run(
        loop._record_lesson(
            iteration=9,
            result=_lesson_result(iteration=9),
            decision="REVERT_PERF",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
            },
        )
    )

    assert "BLOCK_N 128" in store.read(9)


def test_record_lesson_survives_a_failing_summarizer(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)

    async def broken(_prompt: str) -> str:
        raise RuntimeError("session gone")

    asyncio.run(
        loop._record_lesson(
            iteration=8,
            result=_lesson_result(iteration=8),
            decision="CRASH",
            session_sink={
                "session_started": True,
                "summarize": broken,
                "findings": "boom",
            },
        )
    )

    assert "OUTCOME: CRASH" in store.read(8)
    # The reason must reach the operator, not just log.debug: an outcome-only
    # run is otherwise indistinguishable from a provider that cannot resume.
    printed = capsys.readouterr().out
    assert "RuntimeError" in printed
    assert "session gone" in printed


def test_record_lesson_falls_back_when_summary_cannot_be_persisted(tmp_path, monkeypatch, capsys):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    store = _attach_lessons(loop, workspace)
    real_write = store.write
    writes = {"count": 0}

    def fail_first_write(iteration, text):
        writes["count"] += 1
        if writes["count"] == 1:
            return None
        return real_write(iteration, text)

    monkeypatch.setattr(store, "write", fail_first_write)
    asyncio.run(
        loop._record_lesson(
            iteration=12,
            result=_lesson_result(iteration=12),
            decision="REVERT_VALIDATION",
            session_sink={
                "session_started": True,
                "summarize": _fake_summarizer,
                "findings": "compile error: invalid cast",
            },
            diff_summary="kernel.py | 2 +-",
        )
    )

    text = store.read(12)
    assert writes["count"] >= 2
    assert text.startswith("(no agent summary)")
    assert "compile error: invalid cast" in text
    assert "OUTCOME: REVERT_VALIDATION" in text
    assert "tried three tile shapes" not in text
    printed = capsys.readouterr().out
    assert "failed to persist lesson document" in printed
    assert "[lesson] recorded iter 12:" not in printed


def test_decision_label_matches_every_outcome():
    from kernelforge.loop.runner import _decision_label

    assert _decision_label(_lesson_result(crashed=True)) == "CRASH"
    assert (
        _decision_label(_lesson_result(validation_passed=False, validation_summary="BUILD FAILED: boom"))
        == "BUILD_FAILED"
    )
    assert (
        _decision_label(_lesson_result(validation_passed=False, validation_summary="stage 3 failed"))
        == "REVERT_VALIDATION"
    )
    assert (
        _decision_label(
            _lesson_result(
                validation_passed=False,
                validation_summary="full suite timed out",
                validation_outcome="timeout",
            )
        )
        == "REVERT_VALIDATION_TIMEOUT"
    )
    assert (
        _decision_label(
            _lesson_result(
                validation_passed=False,
                validation_summary="driver crashed",
                validation_outcome="driver_error",
            )
        )
        == "REVERT_VALIDATION_ERROR"
    )
    assert _decision_label(_lesson_result(kept=True)) == "KEEP"
    assert _decision_label(_lesson_result(kept=False)) == "REVERT_PERF"


# ── end-to-end: agent session -> lesson document -> next prompt ───────────────


class _StubStage:
    def __init__(self, stage, name, snr_db):
        self.stage, self.stage_name, self.snr_db = stage, name, snr_db
        self.passed = True


class _StubReport:
    all_passed = True
    failed_stage = 0
    failed_output = ""
    results = [_StubStage(5, "snr", 55.0)]

    def summary(self):
        return "all stages passed"


def _stub_measurement(monkeypatch, walls):
    """Stub build/validate/bench so the loop runs without a GPU."""

    async def fake_validation(**_kwargs):
        return _StubReport()

    async def fake_bench(**_kwargs):
        wall_ms = walls.pop(0) if walls else 1.0
        return {
            "success": True,
            "median_ms": wall_ms,
            "case_times": {"case": wall_ms},
            "unscored_cases": [],
            "measurements": [
                {
                    "success": True,
                    "case_times": {"case": wall_ms},
                    "unscored_cases": [],
                }
                for _ in range(3)
            ],
            "message": "stub",
        }

    async def fake_registers(**_kwargs):
        return {"success": False}

    monkeypatch.setattr(runner_module, "run_validation_pipeline", fake_validation)
    monkeypatch.setattr(runner_module, "measure_wallclock", fake_bench)
    monkeypatch.setattr(runner_module, "check_registers", fake_registers)


def test_lesson_flows_from_one_iteration_into_the_next_prompt(tmp_path, monkeypatch):
    """Lock the whole chain the feature exists for.

    agent edits -> session_sink["summarize"] -> lessons/iter_NNN.md
    -> next iteration's prompt.

    Every hop has a unit test; this asserts they are actually connected, which
    is where a wiring regression would otherwise hide.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2)
    # Two iterations: the first improves and is kept, the second does not.
    _stub_measurement(monkeypatch, walls=[0.5, 0.9])

    kernel = Path(loop.ic.kernel_file)
    prompts: list[str] = []
    edits = {"n": 0}

    async def summarizer(_prompt: str) -> str:
        return "vectorized the global load\n- [better] 128-bit loads | 1.8x"

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        edits["n"] += 1
        kernel.write_text(f"def kernel():\n    return {edits['n'] + 1}\n")
        session_sink["plan"] = f"edit {edits['n']}"
        session_sink["end_reason"] = "candidate_submitted"
        session_sink["summarize"] = summarizer
        return "rationale"

    asyncio.run(loop.run(agent_fn=agent_fn))

    # The first iteration produced a document with both authors' halves.
    first = (workspace / "forge_experiments" / "lessons" / "iter_001.md").read_text()
    assert "vectorized the global load" in first
    assert "128-bit loads" in first
    assert "OUTCOME: KEEP" in first

    # ...and the second iteration's prompt carried it, plus the absolute path
    # of the directory holding the full history.
    assert len(prompts) == 2
    assert "vectorized the global load" not in prompts[0]  # nothing to inject yet
    assert "128-bit loads" in prompts[1]
    assert "Implementer session records from recent iterations" in prompts[1]
    lessons_dir = str((workspace / "forge_experiments" / "lessons").resolve())
    assert lessons_dir in prompts[1]


def test_prompt_omits_the_digest_once_lessons_exist(tmp_path, monkeypatch):
    """The digest is reserved for the supervisor once the header renders.

    Both carry the recent iterations, so inlining them together would spend the
    prompt budget saying the same thing twice.
    """
    loop, workspace = _make_loop(tmp_path, monkeypatch, session_count=2)
    _stub_measurement(monkeypatch, walls=[0.5, 0.9])

    kernel = Path(loop.ic.kernel_file)
    prompts: list[str] = []
    edits = {"n": 0}

    async def summarizer(_prompt: str) -> str:
        return "Attempted one tile variant; measured 0.9x."

    async def agent_fn(_kernel_path, experiment_history, session_sink):
        prompts.append(experiment_history)
        edits["n"] += 1
        kernel.write_text(f"def kernel():\n    return {edits['n'] + 1}\n")
        session_sink["summarize"] = summarizer
        return "rationale"

    asyncio.run(loop.run(agent_fn=agent_fn))

    # The archive digest's own header would appear verbatim if it were inlined.
    assert "Solution archive — your lineage so far" not in prompts[1]
    assert "Long-Horizon Memory" in prompts[1]
    assert "Implementer session records from recent iterations" in prompts[1]


def test_the_anti_gaming_boundary_is_stated_wherever_it_is_enforced():
    """The rule that blocks harness edits also says what it does not forbid.

    `mhc-fused` banned a host-side weight cache to keep "perturbed inputs
    refresh", costing a mechanism worth 11.3%. The harness does not perturb that
    tensor, and the agent could have read it. The enforced boundary -- do not edit
    what measures you -- was never stated, so a stricter one was inferred from it.
    Both rule blocks must carry the clarification, or one template keeps the
    ambiguity the other lost.
    """
    from kernelforge.orchestrator import agent as agent_module

    source = Path(agent_module.__file__).read_text()
    enforcement = [
        "Never edit the measurement / driver / harness files",
        "Do NOT edit the test harness / driver",
    ]
    for marker in enforcement:
        assert marker in source, f"rule block missing: {marker}"
        # The prompts are wrapped source literals, so a clause can be split across
        # lines; compare on collapsed whitespace rather than the wrapped text.
        block = " ".join(source[source.index(marker) : source.index(marker) + 900].split())
        assert "are NOT gaming" in block, f"boundary not stated near: {marker}"
        assert "read the harness" in block, f"no read-before-refusing near: {marker}"
        assert "not a reason" in block, f"assumption not rejected near: {marker}"
