# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base-guided planning: the playbook, its ledger, and the GUIDED -> FREE phase."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from kernelforge.agent_backends import AgentCapabilities, AgentRunResult
from kernelforge.cli import forge_loop_command
from kernelforge.cli.options import _validate_wiki_guided
from kernelforge.knowledge.wiki_tool import KnowledgeWikiTool
from kernelforge.loop import wiki_guidance as guidance
from kernelforge.loop.run_state import LoopStateStore
from kernelforge.loop.runner import IterationConfig, IterationLoop, IterationResult
from kernelforge.loop.wiki_guidance import (
    GuidanceState,
    WikiGuidanceStore,
    cited_items,
    decide_phase,
    parse_playbook,
    record_outcome,
)
from kernelforge.orchestrator import orchestration as orchestration_module
from kernelforge.orchestrator.contracts import OrchestrationRunResult, SpecialistOutcome, SynthesizedPlan
from kernelforge.orchestrator.orchestration import (
    OrchestrationAgent,
    OrchestrationInfrastructureError,
    OrchestrationService,
    default_specialist_definitions,
)
from kernelforge.orchestrator.specialists import SpecialistRunResult
from kernelforge.tests.test_forge_loop_resume import _install_cli_fakes, _invoke_forge_loop
from kernelforge.tests.test_forge_orchestration import _context
from kernelforge.tests.test_loop_runner import _AMPLE_BUDGET_SEC, _lane_agent_factory, _make_loop, _unused_supervisor
from kernelforge.tracker import ExperimentTracker

PLAYBOOK = {
    "summary": "two independent steps",
    "groups": [
        {"id": "G1", "title": "loop body", "why_together": "both edit the inner loop"},
        {"id": "G2", "title": "grid order", "why_together": "the launch"},
    ],
    "items": [
        {"id": "P1", "group": "G1", "title": "deferred normalization", "wiki_pages": ["technique-fa-forward"]},
        {"id": "P2", "group": "G2", "title": "XCD remap", "wiki_pages": ["technique-xcd-remap"]},
    ],
}


def _state_with_playbook() -> GuidanceState:
    playbook, notes = parse_playbook(json.dumps(PLAYBOOK))
    assert notes == []
    state = GuidanceState()
    state.adopt(playbook)
    return state


# --- the playbook and its ledger ------------------------------------------------------------------------------------


def test_a_playbook_keeps_only_items_a_knowledge_base_page_supports():
    payload = json.loads(json.dumps(PLAYBOOK))
    payload["items"] += [
        {"id": "P3", "group": "G1", "title": "the builder's own idea", "wiki_pages": []},
        {"id": "P4", "group": "G9", "title": "an unknown group", "wiki_pages": ["page"]},
        {"id": "X5", "group": "G1", "title": "a malformed id", "wiki_pages": ["page"]},
    ]

    playbook, notes = parse_playbook("Here it is:\n" + json.dumps(payload))

    assert playbook.item_ids() == ("P1", "P2")
    assert any("P3" in note and "no knowledge base page" in note for note in notes)
    assert any("P4" in note and "G9" in note for note in notes)
    assert len(notes) == 3


def test_an_answer_without_a_usable_item_is_no_playbook():
    assert parse_playbook("I could not find anything.")[0] is None
    empty = {"groups": [{"id": "G1", "title": "x"}], "items": [{"id": "P1", "group": "G1", "title": "x"}]}
    playbook, notes = parse_playbook(json.dumps(empty))
    assert playbook is None
    assert "the playbook has no usable item" in notes


def test_a_plan_claims_and_declines_items_only_by_citation():
    plan = (
        "Implement the loop body change; P2 is mentioned in passing.\n"
        "[playbook:P1, p2]\n[playbook-skip:P2] the grid is already XCD-ordered\n[playbook:P9]"
    )
    assert cited_items(plan, ["P1", "P2"]) == (["P1"], ["P2"])
    assert cited_items("mentions P1 without a citation", ["P1"]) == ([], [])


def test_the_ledger_settles_items_from_measured_decisions():
    state = _state_with_playbook()

    assert record_outcome(state, iteration=1, plan_text="[playbook:P1]", decision="KEEP") == ["P1"]
    assert record_outcome(state, iteration=2, plan_text="[playbook:P2]", decision="BUILD_FAILED") == ["P2"]
    assert state.ledger["P2"].status == guidance.STATUS_FAILED
    assert record_outcome(state, iteration=3, plan_text="[playbook:P2]", decision="REVERT_VALIDATION") == ["P2"]

    assert state.ledger["P1"].status == guidance.STATUS_APPLIED
    assert state.ledger["P2"].status == guidance.STATUS_REFUTED
    assert state.ledger["P2"].failures == 2
    assert guidance.exhausted(state)


def test_the_ledger_ignores_infrastructure_and_counts_an_iteration_once():
    state = _state_with_playbook()

    assert record_outcome(state, iteration=1, plan_text="[playbook:P1]", decision="ORCHESTRATION_ERROR") == []
    assert record_outcome(state, iteration=1, plan_text="[playbook:P1]", decision="REVERT_CONTENDED") == []
    assert state.ledger["P1"].status == guidance.STATUS_PENDING
    assert record_outcome(state, iteration=2, plan_text="[playbook:P1]", decision="KEEP") == ["P1"]
    assert record_outcome(state, iteration=2, plan_text="[playbook:P1]", decision="KEEP") == []
    # A later plan that re-cites an item already in the incumbent and loses does not take it back out.
    record_outcome(state, iteration=3, plan_text="[playbook:P1]", decision="REVERT_PERF")
    assert state.ledger["P1"].status == guidance.STATUS_APPLIED
    assert state.ledger["P1"].iterations == [2, 3]


@pytest.mark.parametrize(
    ("setup", "reasons", "used", "expected"),
    [
        ("settled_none", (), 0.0, ("FREE", ("PLAYBOOK_UNAVAILABLE",))),
        ("exhausted", (), 0.0, ("FREE", ("PLAYBOOK_EXHAUSTED",))),
        ("pending", (), 0.35, ("FREE", ("GUIDED_BUDGET_SPENT",))),
        # One GUIDED round cannot spend the share, however long it took.
        ("one_round", (), 0.9, None),
        ("pending", ("NO_IMPROVEMENT_STALL",), 0.1, ("FREE", ("GUIDED_STALL", "NO_IMPROVEMENT_STALL"))),
        ("pending", ("CANONICAL_GAIN_AVAILABLE",), 0.34, None),
        ("unbuilt", ("NO_IMPROVEMENT_STALL",), 0.1, None),
    ],
)
def test_the_phase_leaves_guided_on_exhaustion_budget_or_stall(setup, reasons, used, expected):
    if setup == "unbuilt":
        state = GuidanceState()
    elif setup == "settled_none":
        state = GuidanceState(playbook_settled=True)
    else:
        state = _state_with_playbook()
        state.guided_rounds = 1 if setup == "one_round" else guidance.MIN_GUIDED_ROUNDS
        if setup == "exhausted":
            for record in state.ledger.values():
                record.status = guidance.STATUS_SKIPPED

    assert decide_phase(state, search_reasons=reasons, budget_used_fraction=used, budget_fraction=0.35) == expected


def test_a_ranking_keeps_only_open_groups_and_leaves_the_rest_in_the_builders_order():
    state = _state_with_playbook()
    state.ledger["P1"].status = guidance.STATUS_APPLIED
    answer = {
        "ranking": [
            {"group": "g2", "cases": "decode", "expected_gain": "about half of decode", "evidence": "grid-starved"},
            {"group": "G1", "evidence": "already applied"},
            {"group": "G2"},
            {"group": "G7"},
        ]
    }

    order, evidence, notes = guidance.parse_ranking("ranked:\n" + json.dumps(answer), state.open_group_ids())

    assert state.open_group_ids() == ["G2"]
    assert order == ["G2"]
    assert evidence == {"G2": "cases: decode; expected gain: about half of decode; evidence: grid-starved"}
    assert len(notes) == 3
    state.group_order, state.rank_evidence = order, evidence
    rendered = guidance.render_for_planning(state)["playbook"]["groups"]
    assert [group["id"] for group in rendered] == ["G2", "G1"]
    assert rendered[0]["rank_evidence"].startswith("cases: decode")
    assert "rank_evidence" not in rendered[1]
    with pytest.raises(ValueError, match="names no open group"):
        guidance.parse_ranking(json.dumps({"ranking": [{"group": "G1"}]}), ["G2"])


def test_free_is_final():
    state = _state_with_playbook()
    state.phase = guidance.PHASE_FREE
    assert (
        decide_phase(state, search_reasons=("NO_IMPROVEMENT_STALL",), budget_used_fraction=1.0, budget_fraction=0.1)
        is None
    )


def test_the_state_survives_a_round_trip_and_an_unreadable_file_starts_afresh(tmp_path):
    store = WikiGuidanceStore(tmp_path)
    state = _state_with_playbook()
    record_outcome(state, iteration=4, plan_text="[playbook:P1]", decision="KEEP")
    state.inflight_iteration, state.inflight_plan = 5, "[playbook:P2]"
    store.save(state)

    loaded = store.load()
    assert loaded.to_dict() == state.to_dict()
    assert json.loads(store.playbook_path.read_text())["items"][0]["id"] == "P1"

    store.state_path.write_text("{not json")
    assert store.load().to_dict() == GuidanceState().to_dict()


# --- the loop -------------------------------------------------------------------------------------------------------


class _GuidedService:
    """An orchestration service whose plans and playbook answers are scripted."""

    def __init__(self, plans, *, playbook=json.dumps(PLAYBOOK), outages=0):
        self.plans = list(plans)
        self.playbook = playbook
        self.outages = outages
        self.builds = 0
        self.contexts = []

    async def build_wiki_playbook(self, context, *, usage=None):
        self.builds += 1
        if self.outages:
            self.outages -= 1
            raise OrchestrationInfrastructureError("playbook backend exceeded 600s timeout")
        return self.playbook

    async def run(self, context, *, usage=None, lanes=1):
        self.contexts.append(context)
        plans = self.plans.pop(0)
        return OrchestrationRunResult(dispatch_plan=None, optimization_plans=tuple(plans))


class _RankingService(_GuidedService):
    """A service that also ranks; each answer is the ranker's text, or the error it raises."""

    def __init__(self, plans, rankings, **kwargs):
        super().__init__(plans, **kwargs)
        self.rankings = list(rankings)
        self.ranked_contexts = []

    async def rank_wiki_playbook(self, context, *, usage=None):
        self.ranked_contexts.append(context)
        answer = self.rankings.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _ranking(*groups):
    return json.dumps({"ranking": [{"group": group, "evidence": f"profile says {group}"} for group in groups]})


def _guided_loop(
    tmp_path, monkeypatch, *, wiki_guided=True, rounds=3, resume=False, max_time_hours=1.0, kernel_text="", **ic
):
    """A loop built by the production constructor with guided planning on (or off)."""
    if resume:
        workspace = tmp_path / "workspace"
        base = IterationConfig(
            kernel_file=str(workspace / "kernel.py"),
            driver_script=str(workspace / "driver.py"),
            baseline_wall_ms=1.0,
            baseline_case_times={"case": 1.0},
            max_time_hours=max_time_hours,
            git_branch="test-loop",
            workspace_dir=str(workspace),
            supervise_after=5,
            supervise_cooldown=0,
        )
    else:
        scaffold, workspace = _make_loop(tmp_path, monkeypatch)
        base = scaffold.ic
        if kernel_text:
            (workspace / "kernel.py").write_text(kernel_text)
            subprocess.run(["git", "commit", "-qam", "kernel"], cwd=workspace, check=True)
    config = replace(base, wiki_guided=wiki_guided, max_time_hours=max_time_hours, **ic)
    loop = IterationLoop(
        config,
        ExperimentTracker(tmp_path / "experiments"),
        config=SimpleNamespace(experiments_dir=workspace / "forge_experiments", gpu_target="gfx942"),
        resume=resume,
    )
    monkeypatch.setattr(loop, "_time_remaining", lambda: _AMPLE_BUDGET_SEC if len(loop.results) < rounds else 0.0)
    return loop, workspace


def _editing_agent(histories):
    async def agent(kernel_path, history, session_sink):
        histories.append(history)
        session_sink["plan"] = history
        Path(kernel_path).write_text(f"def kernel():\n    return {len(histories) + 1}\n")
        return "candidate ready"

    return agent


def _measured(decisions):
    queue = list(decisions)

    async def measured(iteration, plan=""):
        kept = queue.pop(0) == "KEEP"
        speedup = 1.0 + 0.5 * iteration if kept else 1.0
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=True,
            validation_summary="canonical validation passed",
            wall_ms=1.0 / speedup,
            mean_case_speedup=speedup,
            kept=kept,
            bench_detail={"case_times": {"case": 1.0 / speedup}},
        )

    return measured


def _events(workspace, kind):
    return [event for event in LoopStateStore(str(workspace)).read_events() if event.get("type") == kind]


def _run(loop, service, agent, **kwargs):
    asyncio.run(loop.run(agent_fn=agent, supervisor_fn=_unused_supervisor, orchestration_service=service, **kwargs))


def test_without_the_flag_planning_is_unchanged(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, wiki_guided=False, rounds=1)
    service = _GuidedService([["tune the loop [playbook:P1]"]])
    histories: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))

    _run(loop, service, _editing_agent(histories))

    assert service.builds == 0
    assert service.contexts[0].wiki_guidance is None
    assert "wiki_guidance" not in service.contexts[0].to_prompt_dict()
    assert "## Wiki guidance" not in histories[0]
    assert "guidance_phase" not in _events(workspace, "iteration_result")[0]
    assert not (workspace / "forge_experiments" / "wiki_guidance").exists()


def test_a_guided_campaign_plans_from_the_playbook_until_it_is_spent(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    service = _GuidedService(
        [
            ["Implement the loop body group.\n[playbook:P1]"],
            ["Reorder the grid.\n[playbook:P2]"],
            ["Something the playbook never named."],
        ]
    )
    histories: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP", "REVERT", "REVERT"]))

    _run(loop, service, _editing_agent(histories))

    assert service.builds == 1
    first, second, third = (context.wiki_guidance for context in service.contexts)
    assert first["phase"] == "GUIDED"
    assert [item["status"] for item in first["playbook"]["items"]] == ["PENDING", "PENDING"]
    assert first["playbook"]["groups"][0]["items"] == ["P1"]
    assert "[playbook:P1,P3]" in first["instructions"]
    assert second["phase"] == "GUIDED"
    assert [item["status"] for item in second["playbook"]["items"]] == ["APPLIED", "PENDING"]
    # P2 was measured and lost, so the playbook is spent before the third round plans.
    assert third["phase"] == "FREE"
    assert third["reason_codes"] == ["PLAYBOOK_EXHAUSTED"]
    assert [item["status"] for item in third["playbook"]["items"]] == ["APPLIED", "REFUTED"]
    assert "Optimize beyond the playbook" in third["instructions"]
    assert "wiki_guidance" in service.contexts[0].to_prompt_dict()

    assert "Phase: GUIDED" in histories[0]
    assert "Phase: FREE (PLAYBOOK_EXHAUSTED)" in histories[2]
    assert [event["guidance_phase"] for event in _events(workspace, "iteration_result")] == ["GUIDED", "GUIDED", "FREE"]
    [built] = _events(workspace, "wiki_playbook_built")
    assert built["items"] == ["P1", "P2"]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["phase"], changed["reason_codes"]) == (3, "FREE", ["PLAYBOOK_EXHAUSTED"])
    ledger = [event["items"] for event in _events(workspace, "wiki_playbook_ledger")]
    assert ledger == [{"P1": "APPLIED"}, {"P2": "REFUTED"}]
    persisted = WikiGuidanceStore(workspace).load()
    assert persisted.phase == "FREE"
    assert persisted.ledger["P1"].iterations == [1]


def test_an_unusable_playbook_plans_as_without_the_flag(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    service = _GuidedService([["plan one"], ["plan two"]], playbook="no JSON here")
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 1
    assert [context.wiki_guidance for context in service.contexts] == [None, None]
    [unavailable] = _events(workspace, "wiki_playbook_unavailable")
    assert unavailable["settled"] is True
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["phase"], changed["reason_codes"]) == ("FREE", ["PLAYBOOK_UNAVAILABLE"])
    assert (workspace / "forge_experiments" / "wiki_guidance" / "playbook_response.md").read_text() == "no JSON here"


def test_a_builder_outage_is_retried_on_the_next_round(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    service = _GuidedService([["plan one"], ["plan two [playbook:P1]"]], outages=1)
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "KEEP"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 2
    assert service.contexts[0].wiki_guidance is None
    assert service.contexts[1].wiki_guidance["phase"] == "GUIDED"
    assert [event["settled"] for event in _events(workspace, "wiki_playbook_unavailable")] == [False]
    assert _events(workspace, "wiki_guidance_phase") == []
    assert WikiGuidanceStore(workspace).load().ledger["P1"].status == "APPLIED"


def test_a_second_outage_gives_up_on_the_playbook(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    service = _GuidedService([["one"], ["two"], ["three"]], outages=2)
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 2
    assert [event["settled"] for event in _events(workspace, "wiki_playbook_unavailable")] == [False, True]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (2, ["PLAYBOOK_UNAVAILABLE"])


def _budget_clock(loop, remaining_after_results):
    """``_time_remaining`` read off how many iterations have finished; 0 once the schedule runs out."""
    return lambda: (
        remaining_after_results[len(loop.results)] if len(loop.results) < len(remaining_after_results) else 0.0
    )


def test_the_guided_budget_share_ends_guided_only_after_two_guided_rounds(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, max_time_hours=4.0)
    # Three quarters of the budget are gone by the second round and seven eighths by the third.
    monkeypatch.setattr(loop, "_time_remaining", _budget_clock(loop, [14400.0, 3600.0, 1800.0]))
    service = _GuidedService([["one [playbook:P1]"], ["two"], ["three"]])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert [context.wiki_guidance["phase"] for context in service.contexts] == ["GUIDED", "GUIDED", "FREE"]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (3, ["GUIDED_BUDGET_SPENT"])
    assert (changed["guided_rounds"], changed["guided_budget_used_fraction"]) == (2, 0.875)


def test_setup_before_the_first_guided_round_is_not_charged_to_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, max_time_hours=4.0)
    # An hour of setup, then half an hour per round: by round three half the budget is gone, but only a third of what
    # was left when GUIDED began.
    monkeypatch.setattr(loop, "_time_remaining", _budget_clock(loop, [10800.0, 9000.0, 7200.0]))
    service = _GuidedService([["one [playbook:P1]"], ["two"], ["three"]])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert [context.wiki_guidance["phase"] for context in service.contexts] == ["GUIDED"] * 3
    assert _events(workspace, "wiki_guidance_phase") == []
    persisted = WikiGuidanceStore(workspace).load()
    assert (persisted.guided_started_used_sec, persisted.guided_rounds) == (3600.0, 3)


def test_a_stall_the_search_policy_reports_ends_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2, supervise_after=1)
    service = _GuidedService([["one [playbook:P1]"], ["two"]])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert service.contexts[1].wiki_guidance["phase"] == "FREE"
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert changed["iter"] == 2
    assert changed["reason_codes"][0] == "GUIDED_STALL"
    assert set(changed["reason_codes"][1:]) <= guidance.STALL_REASONS
    # P2 was never tried, so it is the stall and not exhaustion that ended GUIDED.
    assert WikiGuidanceStore(workspace).load().ledger["P2"].status == "PENDING"


def test_each_lane_credits_the_items_its_own_plan_cites(tmp_path, monkeypatch):
    # Two regions far enough apart that lane two's diff still applies once lane one's KEEP is committed.
    spacer = "".join(f"# line {index}\n" for index in range(12))
    loop, workspace = _guided_loop(
        tmp_path,
        monkeypatch,
        rounds=2,
        lanes=2,
        kernel_text=f"def kernel():\n    return 1\n{spacer}GRID = 1\n",
    )
    service = _GuidedService([["lane one: body\n[playbook:P1]", "lane two: grid\n[playbook:P2]"]])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP", "REVERT"]))

    _run(
        loop,
        service,
        _editing_agent([]),
        agent_factory=_lane_agent_factory(
            {"lane one": ("return 1", "return 11"), "lane two": ("GRID = 1", "GRID = 2")}
        ),
    )

    ledger = [(event["iter"], event["items"]) for event in _events(workspace, "wiki_playbook_ledger")]
    assert ledger == [(1, {"P1": "APPLIED"}), (2, {"P2": "REFUTED"})]


def test_every_guided_round_plans_in_the_order_its_own_evidence_ranks(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    service = _RankingService(
        [["grid [playbook:P2]"], ["body [playbook:P1]"], ["beyond the playbook"]],
        # The second ranking still names G2, which the first round applied.
        [_ranking("G2", "G1"), _ranking("G1", "G2")],
    )
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP", "REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    # The ranker reads the round's guidance as it stood, and the planners get it back in the ranked order.
    assert [group["id"] for group in service.ranked_contexts[0].wiki_guidance["playbook"]["groups"]] == ["G1", "G2"]
    first, second, third = (context.wiki_guidance for context in service.contexts)
    assert [group["id"] for group in first["playbook"]["groups"]] == ["G2", "G1"]
    assert first["playbook"]["groups"][0]["rank_evidence"] == "evidence: profile says G2"
    assert first["ranked_at_iteration"] == 1
    assert [group["id"] for group in second["playbook"]["groups"]] == ["G1", "G2"]
    assert second["ranked_at_iteration"] == 2
    # Both items are settled by the third round, which plans FREE and is not ranked.
    assert third["phase"] == "FREE"
    assert len(service.ranked_contexts) == 2
    assert [event["order"] for event in _events(workspace, "wiki_playbook_ranked")] == [["G2", "G1"], ["G1"]]
    persisted = WikiGuidanceStore(workspace).load()
    assert (persisted.group_order, persisted.ranked_at_iteration) == (["G1"], 2)


def test_a_ranker_that_fails_leaves_the_previous_order_and_the_round_plans_on(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    service = _RankingService(
        [["one"], ["two"], ["three"]],
        [_ranking("G2"), OrchestrationInfrastructureError("rank backend exceeded 600s timeout"), "no JSON here"],
    )
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert [[group["id"] for group in context.wiki_guidance["playbook"]["groups"]] for context in service.contexts] == [
        ["G2", "G1"]
    ] * 3
    errors = [event["error"] for event in _events(workspace, "wiki_playbook_rank_unavailable")]
    assert [error.split(":")[0] for error in errors] == ["OrchestrationInfrastructureError", "ValueError"]


def _conflicting_lanes(tmp_path, monkeypatch, *, wiki_guided, plans, decisions):
    """A two-lane round whose lanes both rewrite the kernel's one line, so lane two stops applying once lane one is
    kept."""
    loop, workspace = _guided_loop(tmp_path, monkeypatch, wiki_guided=wiki_guided, rounds=2, lanes=2)
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 1\n"
    service = _GuidedService(plans)
    histories: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(decisions))
    _run(
        loop,
        service,
        _editing_agent(histories),
        agent_factory=_lane_agent_factory(
            {"lane one": ("return 1", "return 11"), "lane two": ("return 1", "return 12")}
        ),
    )
    return workspace, service, histories


def test_a_guided_campaign_ports_a_lane_its_siblings_keep_stopped_applying(tmp_path, monkeypatch):
    workspace, service, histories = _conflicting_lanes(
        tmp_path,
        monkeypatch,
        wiki_guided=True,
        plans=[["lane one: body\n[playbook:P1]", "lane two: grid\n[playbook:P2]"]],
        decisions=["KEEP", "KEEP"],
    )

    # The port is the round's own second lane, so nothing is planned for it.
    assert len(service.contexts) == 1
    [port] = histories
    [planned] = _events(workspace, "lane_port_planned")
    assert (planned["iter"], planned["lane_id"]) == (2, "2")
    assert planned["base_commit"]
    plan_path = Path(planned["plan_path"])
    assert plan_path == workspace / "forge_experiments" / "orchestration" / "iter_002" / "optimization_plan.md"
    assert f"Read {plan_path}" in port
    plan = plan_path.read_text()
    assert plan.startswith("# Port lane 2 onto the current kernel")
    assert "lane two: grid\n[playbook:P2]" in plan
    lane_diff, since = plan.split("## The lane's diff")[1].split("## What the kept changes did")
    assert "+    return 12" in lane_diff
    assert "+    return 11" in since
    ledger = [(event["iter"], event["items"]) for event in _events(workspace, "wiki_playbook_ledger")]
    assert ledger == [(1, {"P1": "APPLIED"}), (2, {"P2": "APPLIED"})]
    assert WikiGuidanceStore(workspace).load().ledger["P2"].iterations == [2]


def test_an_unguided_campaign_still_drops_a_lane_that_stopped_applying(tmp_path, monkeypatch):
    workspace, service, histories = _conflicting_lanes(
        tmp_path,
        monkeypatch,
        wiki_guided=False,
        plans=[["lane one: body", "lane two: grid"], ["round two"]],
        decisions=["KEEP", "REVERT"],
    )

    assert len(service.contexts) == 2
    assert _events(workspace, "lane_port_planned") == []
    assert "Port lane" not in histories[0]


def test_a_queued_lane_keeps_the_commit_its_copy_was_made_from_across_a_restart(tmp_path, monkeypatch):
    from kernelforge.loop.fanout import LaneResult

    loop, _workspace = _guided_loop(tmp_path, monkeypatch)
    loop._lane_queue = [LaneResult(lane_id="2", plan="grid", diff="--- a/k\n+++ b/k\n", base_commit="abc123")]
    loop._persist_lane_queue()
    loop._lane_queue = []

    loop._restore_lane_queue()

    assert [(lane.lane_id, lane.base_commit) for lane in loop._lane_queue] == [("2", "abc123")]


def test_a_resumed_campaign_continues_its_phase_and_a_fresh_one_starts_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    service = _GuidedService([["body [playbook:P1]"]])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))
    _run(loop, service, _editing_agent([]))

    resumed, _ = _guided_loop(tmp_path, monkeypatch, rounds=1, resume=True)
    assert resumed.wiki_guidance.phase == "GUIDED"
    assert resumed.wiki_guidance.ledger["P1"].status == "APPLIED"
    later = _GuidedService([["grid [playbook:P2]"]])
    monkeypatch.setattr(resumed, "run_one_iteration", _measured(["REVERT"]))
    # A second candidate, distinct from the KEEP the first process committed.
    _run(resumed, later, _editing_agent(["first process"]))
    assert later.builds == 0
    assert [item["status"] for item in later.contexts[0].wiki_guidance["playbook"]["items"]] == ["APPLIED", "PENDING"]
    assert WikiGuidanceStore(workspace).load().ledger["P2"].status == "REFUTED"

    fresh = IterationLoop(
        replace(resumed.ic),
        ExperimentTracker(tmp_path / "experiments"),
        config=SimpleNamespace(experiments_dir=workspace / "forge_experiments", gpu_target="gfx942"),
        resume=False,
    )
    assert fresh.wiki_guidance.to_dict() == GuidanceState().to_dict()
    assert WikiGuidanceStore(workspace).load().playbook is None


# --- the builder role and the planning tasks ------------------------------------------------------------------------


class _McpBackend:
    capabilities = AgentCapabilities(mcp=True)

    def __init__(self, text):
        self.text = text
        self.specs = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        return AgentRunResult(text=self.text)


def test_the_builder_reads_the_knowledge_base_through_its_tools(tmp_path):
    backend = _McpBackend(json.dumps(PLAYBOOK))
    tool = KnowledgeWikiTool(root=tmp_path, log_path=tmp_path / "calls.jsonl")
    agent = OrchestrationAgent(backend=backend, timeout_sec=60, max_turns=8, knowledge_wiki=tool)

    text = asyncio.run(agent.build_wiki_playbook(_context()))

    assert parse_playbook(text)[0].item_ids() == ("P1", "P2")
    [spec] = backend.specs
    assert spec.role == "playbook"
    assert spec.writable is False
    assert set(tool.tool_names) <= set(spec.tool_policy.extra_tools)
    assert "knowledge_wiki" in spec.mcp_servers
    assert spec.mcp_servers["knowledge_wiki"].env["FORGE_KNOWLEDGE_WIKI_ROLE"] == "playbook"
    assert json.loads(spec.user_prompt)["context"]["gpu_target"] == _context().gpu_target


def test_the_ranker_reads_the_evidence_and_the_playbook_read_only(tmp_path):
    backend = _McpBackend(_ranking("G1"))
    tool = KnowledgeWikiTool(root=tmp_path, log_path=tmp_path / "calls.jsonl")
    agent = OrchestrationAgent(backend=backend, timeout_sec=60, max_turns=8, knowledge_wiki=tool)
    context = _guided_context()

    text = asyncio.run(
        OrchestrationService(
            agent=agent, specialist_pool=_RecordingPool(()), definitions=default_specialist_definitions()
        ).rank_wiki_playbook(context)
    )

    assert guidance.parse_ranking(text, ["G1", "G3"])[0] == ["G1"]
    [spec] = backend.specs
    assert spec.role == "playbook_rank"
    assert (spec.writable, spec.tool_policy.read, spec.tool_policy.shell) == (False, True, False)
    payload = json.loads(spec.user_prompt)
    assert payload["context"]["wiki_guidance"]["phase"] == "GUIDED"
    assert payload["context"]["cases"]
    with pytest.raises(ValueError, match="needs a playbook"):
        asyncio.run(agent.rank_wiki_playbook(_context()))


def test_a_guided_round_analyzes_the_groups_its_ranking_puts_first():
    service, _agent, pool = _service({"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE})
    state = GuidanceState()
    payload = json.loads(json.dumps(PLAYBOOK))
    payload["groups"].append({"id": "G3", "title": "epilogue"})
    payload["items"].append({"id": "P3", "group": "G3", "title": "nontemporal stores", "wiki_pages": ["page"]})
    state.adopt(parse_playbook(json.dumps(payload))[0])
    state.group_order = ["G3", "G1"]

    asyncio.run(service.run(replace(_context(), wiki_guidance=guidance.render_for_planning(state)), lanes=2))

    assert [assignment.reason.split(" (")[0] for assignment in pool.assignments] == [
        "Playbook group G3",
        "Playbook group G1",
    ]


def test_the_builder_refuses_without_a_knowledge_base():
    agent = OrchestrationAgent(backend=_McpBackend("{}"), timeout_sec=60, max_turns=8)
    with pytest.raises(ValueError, match="needs a knowledge base"):
        asyncio.run(agent.build_wiki_playbook(_context()))


def test_planning_tasks_point_at_the_guidance_only_when_there_is_some():
    from kernelforge.orchestrator.contracts import wiki_guidance_task

    plain = _context()
    guided = replace(plain, wiki_guidance={"phase": "GUIDED", "instructions": "x", "playbook": {}})

    assert wiki_guidance_task(plain) == ""
    assert "context.wiki_guidance" in wiki_guidance_task(guided)
    assert "wiki_guidance" not in plain.to_prompt_dict()
    assert guided.to_prompt_dict()["wiki_guidance"]["phase"] == "GUIDED"
    with pytest.raises(ValueError, match="wiki_guidance"):
        replace(plain, wiki_guidance=["not", "a", "mapping"])


# --- GUIDED rounds analyze each playbook group whole ----------------------------------------------------------------


class _PlanningAgent:
    """The planning agent's turns, recorded; a dispatch planner that should not be asked in GUIDED."""

    def __init__(self):
        self.dispatched_from: list[list[str]] = []
        self.synthesized_from: list[list[str]] = []
        self.structured_output_diagnostics: dict = {}
        self.phase_durations_sec: dict = {}

    async def plan_dispatch(self, context, definitions, *, usage=None):
        self.dispatched_from.append(sorted(definitions))
        return orchestration_module._bound_dispatch_plan(
            context, [("compute", [case.case_id for case in context.cases], "angle analysis")], ()
        )

    async def synthesize_lane_plan_results(self, context, outcomes, dispatch_plan, coverage, *, lanes, usage=None):
        self.synthesized_from.append([outcome.content for outcome in outcomes])
        return [SynthesizedPlan(text="the plan")]


class _RecordingPool:
    def __init__(self, roles):
        self.roles = set(roles)
        self.assignments = []

    def has_role(self, role_id):
        return role_id in self.roles

    async def run(self, assignments, context, *, usage=None):
        self.assignments.extend(assignments)
        return SpecialistRunResult(
            outcomes=tuple(
                SpecialistOutcome(
                    assignment_id=assignment.assignment_id,
                    role_id=assignment.role_id,
                    duration_sec=0.1,
                    content=f"analysis: {assignment.reason}",
                )
                for assignment in assignments
            )
        )


def _guided_context(phase="GUIDED"):
    payload = json.loads(json.dumps(PLAYBOOK))
    payload["groups"].append({"id": "G3", "title": "epilogue"})
    payload["items"].append({"id": "P3", "group": "G3", "title": "nontemporal stores", "wiki_pages": ["page"]})
    state = GuidanceState(phase=phase, reason_codes=[] if phase == "GUIDED" else ["GUIDED_STALL"])
    state.adopt(parse_playbook(json.dumps(payload))[0])
    state.ledger["P2"].status = "APPLIED"
    state.ledger["P3"].status = "FAILED"
    return replace(_context(), wiki_guidance=guidance.render_for_planning(state))


def _service(roles):
    agent = _PlanningAgent()
    pool = _RecordingPool(roles)
    service = OrchestrationService(
        agent=agent,
        specialist_pool=pool,
        definitions=default_specialist_definitions(),
    )
    return service, agent, pool


def test_a_guided_round_analyzes_each_open_group_whole_without_the_dispatch_planner():
    service, agent, pool = _service({"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE})

    result = asyncio.run(service.run(_guided_context(), lanes=3))

    assert agent.dispatched_from == []
    # G2's only item is applied, so the open groups are G1 and G3 (its item failed once and may be retried).
    assert [assignment.role_id for assignment in pool.assignments] == ["playbook_group", "playbook_group"]
    assert [assignment.reason.split(" (")[0] for assignment in pool.assignments] == [
        "Playbook group G1",
        "Playbook group G3",
    ]
    assert "P1 deferred normalization" in pool.assignments[0].reason
    assert all(
        assignment.target_case_ids == tuple(c.case_id for c in _context().cases) for assignment in pool.assignments
    )
    assert len(agent.synthesized_from[0]) == 2
    assert result.structured_output_diagnostics["dispatch"]["mode"] == "playbook_groups"


def test_a_one_lane_guided_round_takes_the_highest_priority_open_group():
    service, _agent, pool = _service({"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE})

    asyncio.run(service.run(_guided_context(), lanes=1))

    assert [assignment.reason.split(" (")[0] for assignment in pool.assignments] == ["Playbook group G1"]


def test_free_rounds_and_unguided_campaigns_keep_the_angle_specialists():
    service, agent, pool = _service({"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE})
    asyncio.run(service.run(_guided_context(phase="FREE"), lanes=3))
    asyncio.run(service.run(_context(), lanes=3))

    assert agent.dispatched_from == [["algorithm", "compute", "memory"]] * 2
    assert {assignment.role_id for assignment in pool.assignments} == {"compute"}


def test_a_pool_without_the_group_analyst_plans_a_guided_round_as_before():
    service, agent, pool = _service({"compute", "memory", "algorithm"})

    asyncio.run(service.run(_guided_context(), lanes=3))

    assert agent.dispatched_from == [["algorithm", "compute", "memory"]]
    assert {assignment.role_id for assignment in pool.assignments} == {"compute"}


def test_the_factory_adds_the_group_analyst_only_with_a_knowledge_base(monkeypatch):
    runtime = SimpleNamespace(timeout_sec=1800)
    monkeypatch.setattr(
        orchestration_module,
        "create_registered_backend",
        lambda *_args, **_kwargs: SimpleNamespace(
            runtime=runtime, name="test", capabilities=AgentCapabilities(mcp=True)
        ),
    )
    config = SimpleNamespace(
        agent_runtime=lambda: runtime,
        workspace="/workspace",
        max_turns=50,
        specialist_probe=False,
        specialist_probe_max=6,
        specialist_probe_budget_sec=600.0,
        specialist_probe_scratch_root="",
        knowledge_wiki_root="",
        knowledge_wiki_log="",
    )

    service = orchestration_module.make_orchestration_service(config=config)

    assert set(service._specialist_pool._agents) == {"compute", "memory", "algorithm"}
    assert not service._specialist_pool.has_role(orchestration_module.PLAYBOOK_GROUP_ROLE)


# --- the CLI --------------------------------------------------------------------------------------------------------


def test_the_flag_needs_a_knowledge_base_and_a_budget_share():
    assert _validate_wiki_guided(False, 0.35, "") == (False, 0.35)
    assert _validate_wiki_guided(True, 0.5, "/wiki") == (True, 0.5)
    with pytest.raises(click.BadParameter, match="needs a knowledge base"):
        _validate_wiki_guided(True, 0.35, "")
    with pytest.raises(click.BadParameter, match="share of the budget"):
        _validate_wiki_guided(False, 0.0, "")
    with pytest.raises(click.BadParameter, match="share of the budget"):
        _validate_wiki_guided(False, 1.5, "")


def _with_knowledge_base(monkeypatch, tmp_path):
    root = tmp_path / "wiki"
    (root / "scripts").mkdir(parents=True)
    for name in ("query.py", "get_page.py", "grep_wiki.py"):
        (root / "scripts" / name).write_text("")
    installed = forge_loop_command.Config.from_env

    def from_env(**overrides):
        config = installed(**overrides)
        config.knowledge_wiki_root = str(root)
        return config

    monkeypatch.setattr(forge_loop_command.Config, "from_env", staticmethod(from_env))


def test_the_flag_and_its_budget_share_reach_the_loop_config(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    _with_knowledge_base(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, ["--wiki-guided", "--guided-budget-fraction", "0.5"])

    assert result.exit_code == 0, result.output
    assert captured["loops"][-1].ic.wiki_guided is True
    assert captured["loops"][-1].ic.guided_budget_fraction == 0.5
    assert "Wiki-guided planning: on" in result.output


def test_guided_planning_is_off_unless_an_operator_asks(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, [])

    assert result.exit_code == 0, result.output
    assert captured["loops"][-1].ic.wiki_guided is False
    assert captured["loops"][-1].ic.guided_budget_fraction == guidance.DEFAULT_GUIDED_BUDGET_FRACTION


def test_the_flag_without_a_knowledge_base_is_refused(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, ["--wiki-guided"])

    assert result.exit_code != 0
    assert "needs a knowledge base" in result.output
    assert captured["loops"] == []
