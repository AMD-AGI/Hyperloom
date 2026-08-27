"""Recovery-analysis contracts for loop resume and analysis publication."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from kernelforge.loop import runner as runner_module
from kernelforge.loop.run_state import LoopStateStore, RunState, make_event
from kernelforge.loop.runner import IterationLoop, IterationResult
from kernelforge.orchestrator.analysis import (
    ANALYSIS_SCHEMA_VERSION,
    AnalysisAgentService,
    AnalysisBundleError,
    _case_directory,
)
from kernelforge.tests.test_analysis_agent import _BundleBackend, _context, _service, _workspace
from kernelforge.tests.test_loop_runner import _make_loop, _no_change_agent, _unused_supervisor


def test_resume_recovery_rejects_gap_before_pending_keep(tmp_path, monkeypatch):
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
            3,
            decision="REVERT_PERF",
            plan="skipped iteration",
            wall_ms=1.1,
            best_after_ms=1.0,
        )
    )
    pending = {
        "iteration": 2,
        "wall_ms": 0.9,
        "validation_text": "passed",
        "plan": "keep candidate",
    }
    (workspace / "forge_experiments" / "pending_keep.json").write_text(json.dumps(pending))
    loop.state_store = store
    loop.run_state = state

    with pytest.raises(ValueError, match="after pending KEEP iteration"):
        loop._plan_resume_recovery(state, pending)


def test_keep_archive_failure_is_non_fatal_and_clears_pending_journal(
    tmp_path,
    monkeypatch,
):
    loop, workspace = _make_loop(tmp_path, monkeypatch)

    async def editing_agent(kernel_path, _history, session_sink):
        session_sink["plan"] = "derived view failure"
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

    monkeypatch.setattr(loop, "run_one_iteration", successful_iteration)
    monkeypatch.setattr(
        runner_module.CandidateArchive,
        "record",
        lambda _archive, _record: (_ for _ in ()).throw(OSError("simulated archive failure")),
    )

    asyncio.run(loop.run(agent_fn=editing_agent))

    pending_path = workspace / "forge_experiments" / "pending_keep.json"
    assert not pending_path.is_file()
    assert loop.persistence_degraded is True
    state = LoopStateStore(str(workspace)).load()
    assert state.best.iteration == 1


def test_failed_analysis_attempt_does_not_advance_published_commit(tmp_path):
    workspace, kernel, driver = _workspace(tmp_path)
    context = _context(workspace)
    service = _service(tmp_path, _BundleBackend())

    class FailingPublishService(AnalysisAgentService):
        def _publish_generation(self, staging_root, commit_root):  # noqa: ANN001
            raise OSError("simulated publish failure")

    failing = FailingPublishService(
        backend=service.backend,
        config=service.config,
        timeout_sec=service.timeout_sec,
        max_turns=service.max_turns,
        profiling_enabled=True,
    )

    with pytest.raises((AnalysisBundleError, OSError)):
        asyncio.run(
            failing.ensure_bundle(
                context,
                kernel_file=str(kernel),
                driver_script=str(driver),
                source_files=[str(kernel)],
            )
        )

    commit_root = workspace / "forge_experiments" / "analysis" / context.analysis_commit
    assert AnalysisAgentService._published_generation_root(commit_root) is None


@pytest.mark.asyncio
async def test_profiled_upgrade_preserves_prior_generation(tmp_path) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    context = _context(workspace)
    await _service(
        tmp_path,
        _BundleBackend(),
        profiling_enabled=False,
    ).ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )
    commit_root = workspace / "forge_experiments" / "analysis" / context.analysis_commit
    first_generation = AnalysisAgentService._published_generation_root(commit_root)
    assert first_generation is not None
    assert first_generation.name.startswith("generation-")

    profiled_bundle = await _service(
        tmp_path,
        _BundleBackend(),
        profiling_enabled=True,
    ).ensure_bundle(
        context,
        kernel_file=str(kernel),
        driver_script=str(driver),
        source_files=[str(kernel)],
    )

    second_generation = AnalysisAgentService._published_generation_root(commit_root)
    assert second_generation is not None
    assert second_generation != first_generation
    assert first_generation.is_dir()
    assert profiled_bundle.outcome is not None
    assert profiled_bundle.outcome.upgrade_exhausted is False


@pytest.mark.asyncio
async def test_malformed_analysis_checkpoint_is_rejected(tmp_path) -> None:
    workspace, kernel, driver = _workspace(tmp_path)
    context = _context(workspace)
    commit_root = workspace / "forge_experiments" / "analysis" / context.analysis_commit
    generation_root = commit_root / "generation-001"
    generation_root.mkdir(parents=True)
    (generation_root / "request.json").write_text(
        json.dumps(
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "analysis_commit": context.analysis_commit,
                "analysis_profiling_enabled": True,
                "cases": [{"case_id": "case-a", "directory": "case-a", "latency_ms": 1.0}],
            }
        )
    )
    (generation_root / "workflow.json").write_text('{"schema_version": 1, "session": {}}')
    (generation_root / "published.json").write_text(json.dumps({"generation_root": "generation-001"}))
    (commit_root / "published.json").write_text(json.dumps({"generation_root": "generation-001"}))

    service = _service(tmp_path, _BundleBackend(), profiling_enabled=True)
    with pytest.raises(AnalysisBundleError, match="workflow schema_version is invalid"):
        await service.ensure_bundle(
            context,
            kernel_file=str(kernel),
            driver_script=str(driver),
            source_files=[str(kernel)],
        )


@pytest.mark.asyncio
async def test_fake_agent_command_rows_do_not_mark_profiled(tmp_path) -> None:
    workspace, _kernel, _driver = _workspace(tmp_path)
    context = _context(workspace)
    work_root = workspace / "forge_experiments" / "analysis" / "work" / context.analysis_commit
    case = type(
        "Case",
        (),
        {
            "case_id": "case-a",
            "directory": _case_directory("case-a"),
        },
    )()
    case_root = work_root / "cases" / case.directory
    profile_root = case_root / "profile"
    profile_root.mkdir(parents=True, exist_ok=True)
    (work_root / "commands.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-a",
                "command": "rocprofv3 --kernel-trace",
                "success": True,
                "exit_code": 0,
            }
        )
        + "\n"
    )

    assert not AnalysisAgentService._has_valid_profile_evidence(work_root, case)

    (profile_root / "raw.txt").write_text("raw profile\n")
    (case_root / "normalized_metrics.json").write_text(json.dumps({"metrics": {"x": 1}}))
    assert AnalysisAgentService._has_valid_profile_evidence(work_root, case)
    framework_rows = [
        json.loads(line) for line in (work_root / "framework_commands.jsonl").read_text().splitlines() if line.strip()
    ]
    assert framework_rows
    assert all(row.get("framework_owned") is True for row in framework_rows)
    assert not any(row.get("command", "").startswith("rocprof") for row in framework_rows)


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
        evolver=type("Evolver", (), {"on_experiment_complete": lambda *_: {}})(),
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
