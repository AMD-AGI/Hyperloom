# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When the loop refreshes Analysis, what a refresh may reuse, and what a failed one leaves behind."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.roles import analysis_evidence
from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.policy.analysis_refresh_policy import AnalysisRefreshDecision
from kernelforge.workflows.forge_loop.campaign.run_state import BestRecord, LoopStateStore, RunState
from kernelforge.workflows.forge_loop.iteration.runner import IterationResult
from kernelforge.workflows.forge_loop.roles.contracts import EvidenceRef
from kernelforge.workflows.forge_loop.roles.analysis_session import AnalysisAttemptLimitError
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import _make_loop, _unused_supervisor


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
        profiling_enabled = True

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
            to_dict=dict,
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
                to_dict=dict,
            )

        def apply(self, context):
            return context

    class AnalysisService:
        profiling_enabled = True

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
            commit_root = workspace / "forge_experiments" / "analysis" / analysis_commit
            self.root = commit_root / "generation-001"
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "manifest.json").write_text("{}")
            (commit_root / "published.json").write_text(json.dumps({"generation_root": self.root.name}))
            self.manifest = {"status": "READY"}
            self.outcome = SimpleNamespace(
                checkpoint_level="published",
                available_tier="profiled",
                upgrade_exhausted=False,
                to_dict=dict,
            )

        def apply(self, context):
            return context

    class AnalysisService:
        profiling_enabled = True

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

        def apply_checkpoint(self, context):
            return context

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
            runner_module.LoopAgents(agent_fn=editing_agent, analysis_service=service, supervisor_fn=_unused_supervisor)
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
                to_dict=dict,
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


def test_analysis_refresh_event_failure_propagates(
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

    with pytest.raises(OSError, match="^event log unavailable$"):
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

    assert loop.persistence_degraded is False
    assert loop.persistence_errors == []


def test_analysis_refresh_state_save_failure_propagates(tmp_path, monkeypatch):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=None,
    )
    loop.state_store = LoopStateStore(str(workspace))
    loop.run_state = RunState()
    monkeypatch.setattr(loop.state_store, "save", lambda _state: (_ for _ in ()).throw(OSError("save unavailable")))

    class AnalysisService:
        profiling_enabled = True

        async def ensure_bundle(self, _context, **_kwargs):
            raise RuntimeError("bundle unavailable")

        def apply_checkpoint(self, context):
            return context

    with pytest.raises(OSError, match="^save unavailable$"):
        asyncio.run(loop._resolve_analysis_context(AnalysisService()))

    assert loop.persistence_degraded is False
    assert loop.persistence_errors == []


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
            to_dict=dict,
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
            to_dict=dict,
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
                to_dict=dict,
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
            runner_module.LoopAgents(
                agent_fn=editing_agent, analysis_service=AnalysisService(), supervisor_fn=supervisor
            )
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
