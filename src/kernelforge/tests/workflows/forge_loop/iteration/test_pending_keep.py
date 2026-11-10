# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pending-keep checkpoint: the patch it publishes, the journal a failed archive keeps, and the usage it records."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import LoopStateStore, RunState
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.workflows.forge_loop.iteration import campaign_runtime
from kernelforge.workflows.forge_loop.iteration import keep_checkpoint
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import _make_loop


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

    assert set(pending) == keep_checkpoint.PENDING_KEEP_FIELDS
    assert pending["changed_files"] == ["kernel.py"]
    assert set(pending["publication_changed_files"]) == {"driver.py", "kernel.py"}
    assert "prior kept optimization" in pending["publication_patch"]
    assert "return 2" in pending["publication_patch"]
    assert pending["baseline_wall_ms"] == 1.0
    assert pending["search_control"] == {
        "diversification_cycle_completed": True,
    }


@pytest.mark.parametrize("assembly_source", ["s_endpgm\n", "s_endpgm\n\n", "s_endpgm"])
def test_pending_keep_includes_new_assembly_in_recovery_and_publication(tmp_path, monkeypatch, assembly_source):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.ic = replace(loop.ic, commit_new_paths=["*.s"])
    loop.run_state = RunState(campaign_id="new-source", session_index=1)
    kernel = workspace / "kernel.py"
    kernel.write_text("def kernel():\n    return 2\n")
    (workspace / "kernel.s").write_text(assembly_source)
    (workspace / "notes.txt").write_text("not a candidate\n")
    result = IterationResult(
        iteration=1,
        duration_sec=0.1,
        validation_passed=True,
        validation_summary="passed",
        wall_ms=0.8,
        mean_case_speedup=1.25,
        kept=True,
    )
    pending = loop._build_pending_keep(
        result, plan="assembly route", best_before=1.0, rationale="assembly route", kernel_source=kernel.read_text()
    )
    assert loop._git("diff", "--cached", "--name-only") == ""
    commit = loop._git_commit(pending["commit_message"])
    committed_patch = loop._git("diff", pending["base_head"], commit, "--", ".")

    assert set(pending["changed_files"]) == {"kernel.py", "kernel.s"}
    assert set(pending["publication_changed_files"]) == {"kernel.py", "kernel.s"}
    assert pending["patch"] == committed_patch
    assert pending["publication_patch"].strip() == committed_patch
    assert pending["patch_sha256"] == hashlib.sha256(committed_patch.encode()).hexdigest()
    assert "notes.txt" not in committed_patch

    restored = tmp_path / "restored"
    subprocess.run(["git", "clone", str(workspace), str(restored)], check=True, capture_output=True)
    subprocess.run(["git", "checkout", pending["base_head"]], cwd=restored, check=True, capture_output=True)
    for patch in [pending["publication_patch"], loop._publication_patch(commit)]:
        subprocess.run(["git", "apply", "--check", "-"], cwd=restored, input=patch, text=True, check=True)
    subprocess.run(["git", "apply", "-"], cwd=restored, input=pending["publication_patch"], text=True, check=True)
    assert (restored / "kernel.py").read_bytes() == kernel.read_bytes()
    assert (restored / "kernel.s").read_bytes() == (workspace / "kernel.s").read_bytes()


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

    results = asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=editing_agent)))

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
        return

    monkeypatch.setattr(loop, "run_one_iteration", successful_iteration)
    monkeypatch.setattr(campaign_runtime.CandidateArchive, "record", fail_record)

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=editing_agent)))

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
    loop.archive = campaign_runtime.CandidateArchive(
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
        loop._archive_pending_keep(pending, loop._pending_keep_result(pending, "committed-hash"))

    assert loop._pending_keep_path.is_file()


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
