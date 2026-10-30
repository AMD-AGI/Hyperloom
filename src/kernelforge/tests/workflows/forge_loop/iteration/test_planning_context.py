# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What planning and the Implementer are handed each round, and the plan records the round persists."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.workflows.forge_loop.iteration import runner as runner_module
from kernelforge.workflows.forge_loop.campaign.run_state import RunState
from kernelforge.evaluation.scoring import GateScoring
from kernelforge.workflows.forge_loop.roles.contracts import EvidenceRef, PlanCriticOutcome
from kernelforge.tests.workflows.forge_loop.iteration._iteration_loop import (
    _make_loop,
    _no_change_agent,
    _unused_supervisor,
)


def test_a_session_is_handed_what_its_gate_scores_against(tmp_path, monkeypatch):
    """The pristine per-case times and the current best reach the session's gate as one value."""
    loop, _workspace = _make_loop(tmp_path, monkeypatch, baseline_case_times={"case": 2.0})
    seen: list[GateScoring] = []

    async def agent(_kernel_path, _history, session_sink, gate_scoring):
        seen.append(gate_scoring)
        session_sink["plan"] = "inspect only"
        return "No source change was needed."

    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=agent, supervisor_fn=_unused_supervisor)))

    assert seen == [GateScoring({"case": 2.0}, 1.0)]


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
            runner_module.LoopAgents(
                agent_fn=agent_fn, orchestration_service=orchestration_service, supervisor_fn=_unused_supervisor
            )
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
    asyncio.run(loop.run(runner_module.LoopAgents(agent_fn=_no_change_agent, supervisor_fn=_unused_supervisor)))
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


def test_a_round_the_critic_could_not_review_is_carried_as_unreviewed(
    tmp_path,
    monkeypatch,
):
    """The next round is told the review never happened, not that it passed."""
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    head = loop._git("rev-parse", "HEAD").splitlines()[0]
    loop.run_state = RunState(head_commit=head)
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
    )
    critic = PlanCriticOutcome(
        verdict="NOT_REVIEWED",
        error="TimeoutError: plan critic exceeded 600s",
        verdict_source="error",
    )
    result = SimpleNamespace(
        optimization_plans=("# Draft plan\nVectorize global loads.",),
        optimization_plan_draft="# Draft plan\nVectorize global loads.",
        optimization_plan_executable=True,
        dispatch_plan=None,
        specialist_outcomes=(),
        structured_output_diagnostics={"plan_critic": critic.to_dict()},
        plan_critic=critic,
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
    root = workspace / "forge_experiments" / "orchestration" / "iter_001"

    assert error == ""
    assert plan_path is not None
    # The plan still publishes -- an outage costs this round its review, not its round.
    assert plan_path.read_text().startswith("# Draft plan")
    assert loop._last_critic_verdict == "NOT_REVIEWED"
    assert (root / "critic_review.md").read_text().startswith("STATUS: CRITIC_ERROR")
    # No review exists to resume, so a later process inherits no ruling at all.
    assert loop.run_state.last_critic.verdict == ""


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
            runner_module.LoopAgents(
                agent_fn=agent_fn, orchestration_service=object(), supervisor_fn=_unused_supervisor
            )
        )
    )

    handoff = json.loads((workspace / "forge_experiments" / "handoffs" / "iter_001.json").read_text())
    assert handoff["canonical_verdict"] == "NO_CHANGES"
    assert handoff["optimization_plan_path"].endswith("orchestration/iter_001/optimization_plan.md")


def test_the_anti_gaming_boundary_is_stated_wherever_it_is_enforced():
    """The rule that blocks harness edits also says what it does not forbid.

    `mhc-fused` banned a host-side weight cache to keep "perturbed inputs
    refresh", costing a mechanism worth 11.3%. The harness does not perturb that
    tensor, and the agent could have read it. The enforced boundary -- do not edit
    what measures you -- was never stated, so a stricter one was inferred from it.
    Both rule blocks must carry the clarification, or one template keeps the
    ambiguity the other lost.
    """
    from kernelforge.workflows.forge_loop.roles import agent as agent_module

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
