"""Restart-style acceptance tests for a multi-session forge campaign."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time

import pytest

from kernelforge.conftest import SRC_ROOT

import kernelforge.loop.runner as runner_module
from kernelforge.loop.archive import CandidateArchive
from kernelforge.loop.experience import ExperienceLedger
from kernelforge.loop.run_state import LoopStateStore
from kernelforge.loop.runner import IterationConfig, IterationLoop, IterationResult
from kernelforge.tracker import ExperimentTracker


class _NoopEvolver:
    def on_experiment_complete(self, experiment):
        return {}


# The stand-in for "budget is not what this test is about"; it has to clear the
# round admission guard, which prices a whole round rather than only the reserve.
_AMPLE_BUDGET_SEC = 12 * 3600.0


def _initialize_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("def kernel():\n    return 1\n")
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
    return workspace, kernel, driver


def _make_loop(workspace, kernel, driver, tracker, *, session_count, resume=False):
    config = IterationConfig(
        kernel_file=str(kernel),
        driver_script=str(driver),
        baseline_wall_ms=1.0,
        baseline_case_times={"case": 1.0},
        max_time_hours=1.0,
        git_branch="campaign-test",
        workspace_dir=str(workspace),
    )
    loop = IterationLoop(
        config,
        tracker,
        config=object(),
        evolver=_NoopEvolver(),
        resume=resume,
    )
    loop._time_remaining = lambda: _AMPLE_BUDGET_SEC if len(loop.results) < session_count else 0.0
    return loop


async def _successful_iteration(self, iteration, plan=""):
    return IterationResult(
        iteration=iteration,
        duration_sec=0.01,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=1.0 - iteration * 0.05,
        mean_case_speedup=1.0 / (1.0 - iteration * 0.05),
        snr_db=40.0,
        kept=True,
    )


def test_two_sessions_preserve_lineage_history_and_global_ids(tmp_path, monkeypatch):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)
    attempt = 0

    async def editing_agent(kernel_path, _history, session_sink):
        nonlocal attempt
        attempt += 1
        session_sink["plan"] = f"attempt {attempt}"
        path = runner_module.Path(kernel_path)
        path.write_text(path.read_text() + f"\n# attempt {attempt}\n")
        return f"attempt {attempt}"

    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=2,
    )
    first_results = asyncio.run(first.run(agent_fn=editing_agent))
    second = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=2,
        resume=True,
    )
    second_results = asyncio.run(second.run(agent_fn=editing_agent))

    assert [item.iteration for item in first_results] == [1, 2]
    assert [item.iteration for item in second_results] == [3, 4]
    state_store = LoopStateStore(str(workspace))
    state = state_store.load()
    assert state.session_index == 2
    assert state.next_iteration == 5
    assert state.cumulative.iterations == 4

    experiments = sorted(
        tracker.list_experiments(),
        key=lambda experiment: experiment.segment_index,
    )
    assert [experiment.segment_index for experiment in experiments] == [1, 2]
    assert experiments[1].parent_experiment_id == experiments[0].experiment_id
    assert experiments[0].campaign_id == experiments[1].campaign_id

    events = state_store.read_events()
    assert [event["iter"] for event in events if event["type"] == "iteration_started"] == [
        1,
        2,
        3,
        4,
    ]
    assert [event["iter"] for event in events if event["type"] == "iteration_result"] == [
        1,
        2,
        3,
        4,
    ]
    assert len([event for event in events if event["type"] == "baseline_measured"]) == 1

    archive = CandidateArchive(str(workspace), str(kernel))
    assert [entry["iter"] for entry in archive.load_index()] == [1, 2, 3, 4]
    assert sorted(path.name for path in archive.root.glob("iter_*")) == [
        "iter_001",
        "iter_002",
        "iter_003",
        "iter_004",
    ]
    assert [entry.iteration for entry in ExperienceLedger(str(workspace)).entries] == [
        1,
        2,
        3,
        4,
    ]
    best_manifest = json.loads((workspace / "forge_experiments" / "best" / "manifest.json").read_text())
    best_report = (workspace / "forge_experiments" / "optimization_report.md").read_text()
    assert best_manifest["iteration"] == 4
    assert best_manifest["best_wall_ms"] == 0.8
    assert "attempt 4" in best_report
    assert "attempt 3" not in best_report


def test_runner_exposes_state_persistence_degradation(tmp_path, monkeypatch):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    loop = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
    )

    def fail_save(store, _state):
        store._mark_degraded("save", OSError("simulated state write failure"))

    async def no_change_agent(_kernel_path, _history, session_sink):
        session_sink["plan"] = "inspect only"
        return "No source change."

    monkeypatch.setattr(LoopStateStore, "save", fail_save)
    asyncio.run(loop.run(agent_fn=no_change_agent))

    assert loop.persistence_degraded is True
    assert any("simulated state write failure" in item for item in loop.persistence_errors)


def test_published_keep_survives_interruption_resume_and_later_failure(
    tmp_path,
    monkeypatch,
):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    attempt = 0

    async def editing_agent(kernel_path, _history, session_sink):
        nonlocal attempt
        attempt += 1
        session_sink["plan"] = f"candidate {attempt}"
        session_sink["end_reason"] = "candidate_submitted" if attempt == 1 else "turn_cap"
        session_sink["turns"] = 10 if attempt == 1 else 100
        path = runner_module.Path(kernel_path)
        path.write_text(path.read_text() + f"\n# candidate {attempt}\n")
        return f"candidate {attempt}"

    async def canonical_result(self, iteration, plan=""):
        if iteration == 1:
            return IterationResult(
                iteration=iteration,
                duration_sec=0.01,
                validation_passed=True,
                validation_summary="passed",
                wall_ms=0.9,
                mean_case_speedup=1.0 / 0.9,
                snr_db=40.0,
                kept=True,
            )
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=False,
            validation_summary="canonical correctness failed",
            kept=False,
        )

    monkeypatch.setattr(IterationLoop, "run_one_iteration", canonical_result)
    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=3,
    )

    def interrupt_after_keep(_result):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            first.run(
                agent_fn=editing_agent,
                on_iteration=interrupt_after_keep,
            )
        )

    root = workspace / "forge_experiments"
    manifest_after_interrupt = json.loads((root / "best" / "manifest.json").read_text())
    assert manifest_after_interrupt["iteration"] == 1
    assert manifest_after_interrupt["best_wall_ms"] == 0.9

    resumed = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
        resume=True,
    )
    asyncio.run(resumed.run(agent_fn=editing_agent))

    final_manifest = json.loads((root / "best" / "manifest.json").read_text())
    final_report = (root / "optimization_report.md").read_text()
    history = (root / "optimization_history.md").read_text()
    experiments = sorted(
        tracker.list_experiments(),
        key=lambda experiment: experiment.segment_index,
    )

    assert final_manifest == manifest_after_interrupt
    assert "candidate 1" in final_report
    assert "candidate 2" not in final_report
    assert "Iteration 1 — KEEP" in history
    assert "Iteration 2 — REVERT_VALIDATION" in history
    assert "candidate_submitted" in history
    assert "turn_cap" in history
    assert experiments[0].status == "interrupted"
    assert experiments[1].parent_experiment_id == experiments[0].experiment_id


def test_resume_recovers_keep_committed_before_state_checkpoint(tmp_path, monkeypatch):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "recover committed candidate"
        path = runner_module.Path(kernel_path)
        path.write_text(path.read_text() + "\n# verified candidate\n")
        return "verified candidate"

    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
    )

    def interrupt_before_checkpoint(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        first,
        "_finalize_keep_checkpoint",
        interrupt_before_checkpoint,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(first.run(agent_fn=editing_agent))

    root = workspace / "forge_experiments"
    committed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert (root / "pending_keep.json").is_file()
    assert LoopStateStore(str(workspace)).load().best.iteration == 0
    assert not (root / "best" / "manifest.json").exists()

    agent_started = False

    async def forbidden_agent(*_args, **_kwargs):
        nonlocal agent_started
        agent_started = True
        raise AssertionError("resume started an agent before reconciliation")

    resumed = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=0,
        resume=True,
    )
    asyncio.run(resumed.run(agent_fn=forbidden_agent))

    state = LoopStateStore(str(workspace)).load()
    events = LoopStateStore(str(workspace)).read_events()
    manifest = json.loads((root / "best" / "manifest.json").read_text())
    assert agent_started is False
    assert state.best.iteration == 1
    assert state.best.commit_hash == committed_head
    assert state.cumulative.iterations == 1
    assert state.cumulative.kept == 1
    assert manifest["iteration"] == 1
    assert manifest["commit_hash"] == committed_head
    assert len([event for event in events if event["type"] == "iteration_result" and event["iter"] == 1]) == 1
    assert not (root / "pending_keep.json").exists()


def test_resume_clears_reconciled_pending_keep(
    tmp_path,
    monkeypatch,
):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "recover committed candidate"
        path = runner_module.Path(kernel_path)
        path.write_text(path.read_text() + "\n# candidate\n")
        return "verified candidate"

    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
    )

    def interrupt_before_checkpoint(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        first,
        "_finalize_keep_checkpoint",
        interrupt_before_checkpoint,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(first.run(agent_fn=editing_agent))

    root = workspace / "forge_experiments"
    assert (root / "pending_keep.json").is_file()

    resumed = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=0,
        resume=True,
    )
    asyncio.run(resumed.run(agent_fn=None))

    assert not (root / "pending_keep.json").exists()
    archived = CandidateArchive(str(workspace), str(kernel)).load_meta(1)
    assert archived["decision"] == "KEEP"


def test_resume_repairs_publication_after_state_advanced_once(tmp_path, monkeypatch):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)
    original_publish = runner_module.BestResultPublisher.publish
    publish_calls = 0

    def interrupt_publication(self, **kwargs):
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise OSError("simulated publication interruption")
        return original_publish(self, **kwargs)

    monkeypatch.setattr(
        runner_module.BestResultPublisher,
        "publish",
        interrupt_publication,
    )

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "state advanced candidate"
        path = runner_module.Path(kernel_path)
        path.write_text(path.read_text() + "\n# state advanced candidate\n")
        return "state advanced candidate"

    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
    )
    asyncio.run(first.run(agent_fn=editing_agent))

    root = workspace / "forge_experiments"
    interrupted_state = LoopStateStore(str(workspace)).load()
    assert interrupted_state.best.iteration == 1
    assert interrupted_state.cumulative.iterations == 1
    assert interrupted_state.cumulative.kept == 1
    assert not (root / "pending_keep.json").is_file()

    resumed = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=0,
        resume=True,
    )
    asyncio.run(resumed.run(agent_fn=None))

    recovered = LoopStateStore(str(workspace)).load()
    events = LoopStateStore(str(workspace)).read_events()
    manifest = json.loads((root / "best" / "manifest.json").read_text())
    assert recovered.cumulative.iterations == 1
    assert recovered.cumulative.kept == 1
    assert manifest["iteration"] == 1
    assert len([event for event in events if event["type"] == "iteration_result" and event["iter"] == 1]) == 1
    assert not (root / "pending_keep.json").exists()


def test_resume_discards_pending_keep_when_commit_never_happened(tmp_path, monkeypatch):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "uncommitted verified candidate"
        path = runner_module.Path(kernel_path)
        path.write_text(path.read_text() + "\n# uncommitted candidate\n")
        return "uncommitted candidate"

    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
    )
    base_head = first._git("rev-parse", "HEAD").splitlines()[0]

    def interrupt_commit(_message):
        raise KeyboardInterrupt

    monkeypatch.setattr(first, "_git_commit", interrupt_commit)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(first.run(agent_fn=editing_agent))

    root = workspace / "forge_experiments"
    assert (root / "pending_keep.json").is_file()
    assert subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    resumed = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=0,
        resume=True,
    )
    asyncio.run(resumed.run(agent_fn=None))

    state = LoopStateStore(str(workspace)).load()
    assert resumed._git("rev-parse", "HEAD").splitlines()[0] == base_head
    assert resumed._git("status", "--porcelain", "--untracked-files=no") == ""
    assert state.best.iteration == 0
    assert state.cumulative.iterations == 0
    assert not (root / "pending_keep.json").exists()


def test_resume_rejects_pending_keep_with_unexpected_child_commit(
    tmp_path,
    monkeypatch,
):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "expected candidate"
        runner_module.Path(kernel_path).write_text("def kernel():\n    return 2\n")
        return "expected candidate"

    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
    )

    def interrupt_commit(_message):
        raise KeyboardInterrupt

    monkeypatch.setattr(first, "_git_commit", interrupt_commit)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(first.run(agent_fn=editing_agent))

    root = workspace / "forge_experiments"
    state_before = (root / "run_state.json").read_bytes()
    pending_before = (root / "pending_keep.json").read_bytes()
    kernel.write_text("def kernel():\n    return 99\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "unexpected child"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    resumed = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=0,
        resume=True,
    )
    with pytest.raises(ValueError, match="committed patch mismatch"):
        asyncio.run(resumed.run(agent_fn=None))

    assert (root / "run_state.json").read_bytes() == state_before
    assert (root / "pending_keep.json").read_bytes() == pending_before


def test_resume_repairs_best_views_from_run_state_before_agent(tmp_path, monkeypatch):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    tracker = ExperimentTracker(workspace / "forge_experiments")
    monkeypatch.setattr(IterationLoop, "run_one_iteration", _successful_iteration)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "durable state candidate"
        path = runner_module.Path(kernel_path)
        path.write_text(path.read_text() + "\n# durable state candidate\n")
        return "durable state candidate"

    first = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=1,
    )
    asyncio.run(first.run(agent_fn=editing_agent))

    root = workspace / "forge_experiments"
    (root / "best" / "manifest.json").unlink()
    (root / "best_result.json").unlink()
    (root / "optimization_report.md").unlink()
    agent_started = False

    async def forbidden_agent(*_args, **_kwargs):
        nonlocal agent_started
        agent_started = True
        raise AssertionError("resume started an agent before repairing best views")

    resumed = _make_loop(
        workspace,
        kernel,
        driver,
        tracker,
        session_count=0,
        resume=True,
    )
    asyncio.run(resumed.run(agent_fn=forbidden_agent))

    state = LoopStateStore(str(workspace)).load()
    manifest = json.loads((root / "best" / "manifest.json").read_text())
    assert agent_started is False
    assert manifest["iteration"] == state.best.iteration
    assert manifest["commit_hash"] == state.best.commit_hash
    assert json.loads((root / "best_result.json").read_text()) == manifest
    assert "durable state candidate" in (root / "optimization_report.md").read_text()


def test_hard_timeout_preserves_completed_agent_usage_checkpoint(
    tmp_path,
    monkeypatch,
):
    workspace, kernel, driver = _initialize_workspace(tmp_path, monkeypatch)
    validation_started = workspace / "validation-started"
    script = """
import asyncio
import sys
from pathlib import Path

from kernelforge.loop.runner import IterationConfig, IterationLoop
from kernelforge.tracker import ExperimentTracker


class NoopEvolver:
    def on_experiment_complete(self, experiment):
        return {}


class FakeUsage:
    def __init__(self):
        self.values = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_cost_usd": 0.0,
            "calls": 0,
        }

    def totals(self):
        return dict(self.values)


async def main():
    workspace = Path(sys.argv[1])
    kernel = Path(sys.argv[2])
    driver = Path(sys.argv[3])
    validation_started = Path(sys.argv[4])
    tracker = ExperimentTracker(workspace / "forge_experiments")
    loop = IterationLoop(
        IterationConfig(
            kernel_file=str(kernel),
            driver_script=str(driver),
            baseline_wall_ms=1.0,
            baseline_case_times={"case": 1.0},
            max_time_hours=1.0,
            git_branch="timeout-checkpoint",
            workspace_dir=str(workspace),
        ),
        tracker,
        config=object(),
        evolver=NoopEvolver(),
    )
    loop._time_remaining = lambda: 12 * 3600.0
    usage = FakeUsage()

    async def agent(kernel_path, _history, session_sink):
        usage.values.update({
            "input_tokens": 101,
            "output_tokens": 19,
            "total_cost_usd": 0.75,
            "calls": 1,
        })
        session_sink["plan"] = "completed candidate before timeout"
        Path(kernel_path).write_text("def kernel():\\n    return 2\\n")
        return "completed candidate"

    async def blocked_validation(iteration, plan=""):
        validation_started.touch()
        await asyncio.Event().wait()

    loop.run_one_iteration = blocked_validation
    await loop.run(agent_fn=agent, usage=usage)


asyncio.run(main())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(workspace),
            str(kernel),
            str(driver),
            str(validation_started),
        ],
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not validation_started.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"subprocess exited before canonical validation:\n{stdout}\n{stderr}")
            if time.monotonic() >= deadline:
                pytest.fail("subprocess did not reach canonical validation")
            time.sleep(0.01)
        process.kill()
        process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    experiment_payloads = []
    for path in (workspace / "forge_experiments").glob("*.json"):
        payload = json.loads(path.read_text())
        if payload.get("experiment_id"):
            experiment_payloads.append(payload)

    assert len(experiment_payloads) == 1
    assert experiment_payloads[0]["status"] == "running"
    assert experiment_payloads[0]["llm_usage"] == {
        "input_tokens": 101,
        "output_tokens": 19,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_cost_usd": 0.75,
        "calls": 1,
    }
