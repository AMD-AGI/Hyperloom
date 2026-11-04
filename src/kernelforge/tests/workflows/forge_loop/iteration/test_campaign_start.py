# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Starting or resuming a campaign: what a resume restores or refuses, the warm start it adopts, and the best it reconciles."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernelforge.utils.git import GitError
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.tests.workflows.forge_loop._experiment_records import persisted_experiments
from kernelforge.workflows.forge_loop.campaign.reporting import KeepBundle, KeepIdentity, KeepScore
from kernelforge.workflows.forge_loop.campaign.run_state import SESSION_PAUSED, BestRecord, LoopStateStore, RunState
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop
from kernelforge.workflows.forge_loop.search.supervisor import SupervisionMonitor
from kernelforge.workflows.forge_loop.iteration import campaign_runtime
from kernelforge.workflows.forge_loop.iteration import best_publication
from kernelforge.workflows.forge_loop.roles.supervisor import latest_supervisor_ruling_path
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _AMPLE_BUDGET_SEC,
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


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
    ruling_path = latest_supervisor_ruling_path(str(workspace))
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
            runner_module.LoopAgents(agent_fn=agent, orchestration_service=object(), supervisor_fn=_unused_supervisor)
        )
    )

    assert loop._supervisor_ruling == ruling
    assert captured["guidance"] == ruling
    assert ruling in captured["history"]


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
        asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent)))

    assert called is False


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

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent)))

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
    loop.best_publisher = campaign_runtime.BestResultPublisher(str(workspace))
    head = loop._git("rev-parse", "HEAD").strip()
    loop.best_publisher.publish(
        KeepIdentity(
            campaign_id="campaign",
            session_index=1,
            experiment_id="experiment-one",
            iteration=1,
            commit_hash=head,
            plan="prior keep",
        ),
        KeepScore(
            baseline_wall_ms=1.0,
            best_wall_ms=0.8,
            mean_case_speedup=1.4,
            search_start_mean_case_speedup=1.0,
            snr_db=None,
        ),
        KeepBundle(
            validation_text="canonical correctness passed",
            benchmark={"median_ms": 0.8},
            changed_files=["kernel.py"],
            patch="prior patch\n",
        ),
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
        lambda *args, **kwargs: republished.append(args),
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
    loop.best_publisher = campaign_runtime.BestResultPublisher(str(workspace))
    loop.archive = campaign_runtime.CandidateArchive(str(workspace), loop.ic.kernel_file)
    stale_commit = loop._git("rev-parse", "HEAD").strip()
    loop.best_publisher.publish(
        KeepIdentity(
            campaign_id="campaign",
            session_index=1,
            experiment_id="experiment-one",
            iteration=1,
            commit_hash=stale_commit,
            plan="earlier keep",
        ),
        KeepScore(
            baseline_wall_ms=1.0,
            best_wall_ms=0.9,
            mean_case_speedup=1.1,
            search_start_mean_case_speedup=1.0,
            snr_db=None,
        ),
        KeepBundle(
            validation_text="canonical correctness passed",
            benchmark={"median_ms": 0.9},
            changed_files=["kernel.py"],
            patch="earlier patch\n",
        ),
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

    def _record(keep, _score, _bundle, **_kwargs):
        republished.append(keep)
        return {"iteration": keep.iteration}

    monkeypatch.setattr(loop.best_publisher, "publish", _record)

    loop._reconcile_best_publication()

    assert len(republished) == 1
    assert republished[0].commit_hash == head
    assert loop.persistence_degraded is False


def test_resume_publication_diff_failure_degrades_persistence_without_propagating(
    tmp_path,
    monkeypatch,
    caplog,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.best_publisher = campaign_runtime.BestResultPublisher(str(workspace))
    loop.archive = campaign_runtime.CandidateArchive(str(workspace), loop.ic.kernel_file)
    base_commit = loop._git("rev-parse", "HEAD").strip()
    (workspace / "kernel.py").write_text("def kernel():\n    return 2\n")
    best_commit = loop._git_commit("best candidate")
    loop.ic.campaign_base_commit = base_commit
    loop.run_state = RunState(campaign_id="campaign", session_index=2, baseline_wall_ms=1.0)
    loop.run_state.best = BestRecord(
        iteration=1,
        wall_ms=0.8,
        mean_case_speedup=1.25,
        commit_hash=best_commit,
        source="iteration",
    )
    failure = GitError(128, ["git", "diff"], "", "fatal: bad object")
    real_git = best_publication.git

    def fail_publication_diff(*args, **kwargs):
        if args == ("diff", base_commit, best_commit, "--", "."):
            raise failure
        return real_git(*args, **kwargs)

    monkeypatch.setattr(best_publication, "git", fail_publication_diff)

    with caplog.at_level("WARNING", logger=best_publication.log.name):
        loop._reconcile_best_publication()

    assert loop.persistence_degraded is True
    assert loop.persistence_errors == [f"publish best iteration 1: {failure}"]
    assert caplog.messages == ["failed to publish best result"]
    assert caplog.records[0].levelname == "WARNING"


def test_two_loop_instances_resume_global_iteration_and_fresh_budget(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

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
        resume=True,
    )
    monkeypatch.setattr(
        second,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if len(second.results) < 1 else 0.0,
    )
    second_results = asyncio.run(
        second.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
    )

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

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

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
    publisher = campaign_runtime.BestResultPublisher(str(workspace))
    manifest = publisher.publish(
        KeepIdentity(
            campaign_id="warm-start:kernelforge-exp/op/run",
            session_index=0,
            experiment_id="caller-run",
            iteration=0,
            commit_hash=warm_commit,
            plan="apply prior solution kernelforge-exp/op/run",
        ),
        KeepScore(
            baseline_wall_ms=1.0,
            best_wall_ms=0.8,
            mean_case_speedup=1.25,
            search_start_mean_case_speedup=1.25,
            snr_db=None,
            search_start_ms=0.8,
        ),
        KeepBundle(
            validation_text="validated KB warm-start passed canonical correctness",
            benchmark={"median_ms": 0.8, "warm_start": True},
            changed_files=["kernel.py"],
            patch=patch,
        ),
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
    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

    persisted_manifest = json.loads((workspace / "forge_experiments" / "best" / "manifest.json").read_text())
    assert persisted_manifest == manifest
    assert not list((workspace / "forge_experiments" / "best").glob(".iter_000.corrupt-*"))


def test_fresh_run_rejects_existing_campaign_without_modifying_state(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))
    state_path = workspace / "forge_experiments" / "run_state.json"
    before = state_path.read_bytes()

    duplicate_fresh = IterationLoop(
        first.ic,
        first.tracker,
        config=object(),
    )
    with pytest.raises(ValueError, match="--resume"):
        asyncio.run(
            duplicate_fresh.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
        )

    assert state_path.read_bytes() == before


def test_a_rejected_fresh_run_persists_no_pr_references(tmp_path, monkeypatch):
    """Deferred PR writes must not survive an invocation the guard rejects."""
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))

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
    )
    with pytest.raises(ValueError, match="--resume"):
        asyncio.run(
            duplicate_fresh.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
        )

    assert tree() == before
    assert not (workspace / "forge_experiments" / "pr_refs").exists()


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
    asyncio.run(first.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))
    state_path = workspace / "forge_experiments" / "run_state.json"
    before = state_path.read_bytes()
    experiment_count = len(persisted_experiments(first.tracker.dir))
    bad_config = replace(first.ic, **{field_name: field_value})

    mismatched = IterationLoop(
        bad_config,
        first.tracker,
        config=object(),
        resume=True,
    )
    with pytest.raises(ValueError, match=error):
        asyncio.run(
            mismatched.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
        )

    assert state_path.read_bytes() == before
    assert len(persisted_experiments(first.tracker.dir)) == experiment_count


def test_resume_rejects_head_mismatch_without_modifying_state(tmp_path, monkeypatch):
    first, workspace = _make_loop(tmp_path, monkeypatch)
    asyncio.run(first.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))
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
        resume=True,
    )
    with pytest.raises(ValueError, match="HEAD mismatch"):
        asyncio.run(
            mismatched.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor))
        )

    assert state_path.read_bytes() == before
