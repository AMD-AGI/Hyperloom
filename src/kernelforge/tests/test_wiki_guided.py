# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base-guided planning: each round's batch, its one-session iteration, the ledger, and GUIDED -> FREE."""

from __future__ import annotations

import asyncio
import json
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
    parse_playbook,
    record_outcome,
)
from kernelforge.orchestrator import orchestration as orchestration_module
from kernelforge.orchestrator.contracts import OrchestrationRunResult, SpecialistOutcome, SynthesizedPlan
from kernelforge.orchestrator.orchestration import (
    OrchestrationAgent,
    OrchestrationInfrastructureError,
    OrchestrationOutputError,
    OrchestrationService,
    default_specialist_definitions,
)
from kernelforge.orchestrator.specialists import SpecialistRunResult
from kernelforge.tests.test_forge_loop_resume import _install_cli_fakes, _invoke_forge_loop
from kernelforge.tests.test_forge_orchestration import _context
from kernelforge.tests.test_loop_runner import _AMPLE_BUDGET_SEC, _lane_agent_factory, _make_loop, _unused_supervisor
from kernelforge.tracker import ExperimentTracker

PLAYBOOK = {
    "summary": "two independent parts",
    "groups": [
        {"id": "G1", "title": "loop body", "why_together": "both edit the inner loop"},
        {"id": "G2", "title": "grid order", "why_together": "the launch"},
    ],
    "items": [
        {"id": "P1", "group": "G1", "title": "deferred normalization", "wiki_pages": ["technique-fa-forward"]},
        {"id": "P2", "group": "G2", "title": "XCD remap", "wiki_pages": ["technique-xcd-remap"]},
    ],
}
# A later round's answer, numbered from 1 by the builder as every answer is.
SECOND_BUILD = {
    "summary": "the epilogue is what limits the cases now",
    "groups": [{"id": "G1", "title": "epilogue"}],
    "items": [{"id": "P1", "group": "G1", "title": "nontemporal stores", "wiki_pages": ["technique-cache-policy"]}],
}
DECLINED = {"summary": "the profile shows nothing more the knowledge base addresses", "groups": [], "items": []}
THREE_GROUPS = {
    "summary": "three parts",
    "groups": [*PLAYBOOK["groups"], {"id": "G3", "title": "epilogue"}],
    "items": [*PLAYBOOK["items"], {"id": "P3", "group": "G3", "title": "nontemporal stores", "wiki_pages": ["page"]}],
}


def _state_with(payload=PLAYBOOK, *, iteration=1) -> GuidanceState:
    build, notes = parse_playbook(json.dumps(payload))
    assert notes == []
    state = GuidanceState()
    state.add_build(build, iteration=iteration)
    return state


# --- builds, the playbook and its ledger ----------------------------------------------------------------------------


def test_a_build_keeps_only_items_a_knowledge_base_page_supports():
    payload = json.loads(json.dumps(PLAYBOOK))
    payload["items"] += [
        {"id": "P3", "group": "G1", "title": "the builder's own idea", "wiki_pages": []},
        {"id": "P4", "group": "G9", "title": "an unknown group", "wiki_pages": ["page"]},
        {"id": "X5", "group": "G1", "title": "a malformed id", "wiki_pages": ["page"]},
    ]

    build, notes = parse_playbook("Here it is:\n" + json.dumps(payload))

    assert build.item_ids() == ("P1", "P2")
    assert any("P3" in note and "no knowledge base page" in note for note in notes)
    assert any("P4" in note and "G9" in note for note in notes)
    assert len(notes) == 3


def test_an_answer_with_nothing_more_is_told_from_an_unusable_one():
    assert parse_playbook(json.dumps(DECLINED))[0] is None
    assert guidance.declined(json.dumps(DECLINED))
    assert not guidance.declined("I could not find anything.")
    assert not guidance.declined(json.dumps(PLAYBOOK))
    unusable = {"groups": [{"id": "G1", "title": "x"}], "items": [{"id": "P1", "group": "G1", "title": "x"}]}
    build, notes = parse_playbook(json.dumps(unusable))
    assert build is None
    assert "the playbook has no usable item" in notes
    assert not guidance.declined(json.dumps(unusable))


def test_a_build_implements_at_most_six_groups():
    payload = {
        "groups": [{"id": f"G{index}", "title": f"group {index}"} for index in range(1, 9)],
        "items": [
            {"id": f"P{index}", "group": f"G{index}", "title": "x", "wiki_pages": ["page"]} for index in range(1, 9)
        ],
    }

    build, notes = parse_playbook(json.dumps(payload))

    assert len(build.groups) == guidance.MAX_BATCH_GROUPS
    assert any("at most 6 groups" in note for note in notes)


def test_each_build_joins_the_playbook_under_fresh_ids_and_becomes_the_batch():
    state = _state_with()
    second = json.loads(json.dumps(PLAYBOOK))
    second["items"][1]["conflicts_with"] = ["P1"]

    batch = state.add_build(parse_playbook(json.dumps(second))[0], iteration=5)

    assert batch == ["G3", "G4"] == state.batch
    assert state.playbook.item_ids() == ("P1", "P2", "P3", "P4")
    new = {item.item_id: item for item in state.playbook.items}
    assert (new["P3"].group, new["P4"].group, new["P4"].conflicts_with) == ("G3", "G4", ("P3",))
    assert [group.built_at_iteration for group in state.playbook.groups] == [1, 1, 5, 5]
    assert state.ledger["P4"].status == guidance.STATUS_PENDING
    assert state.builds == 2
    rendered = guidance.render_for_planning(state)
    assert [group["id"] for group in rendered["playbook"]["groups"]] == ["G3", "G4", "G1", "G2"]
    assert rendered["batch"] == ["G3", "G4"]
    assert rendered["playbook"]["summary"] == "two independent parts"


def test_a_plan_claims_and_declines_items_only_by_citation():
    plan = (
        "Implement the loop body change; P2 is mentioned in passing.\n"
        "[playbook:P1, p2]\n[playbook-skip:P2] the grid is already XCD-ordered\n[playbook:P9]"
    )
    assert cited_items(plan, ["P1", "P2"]) == (["P1"], ["P2"])
    assert cited_items("mentions P1 without a citation", ["P1"]) == ([], [])
    assert cited_items("[playbook:P101]", ["P101", "P10"]) == (["P101"], [])


def test_the_ledger_settles_items_from_measured_decisions():
    state = _state_with()

    assert record_outcome(state, iteration=1, plan_text="[playbook:P1]", decision="KEEP") == ["P1"]
    assert record_outcome(state, iteration=2, plan_text="[playbook:P2]", decision="BUILD_FAILED") == ["P2"]
    assert state.ledger["P2"].status == guidance.STATUS_FAILED
    assert record_outcome(state, iteration=3, plan_text="[playbook:P2]", decision="REVERT_VALIDATION") == ["P2"]

    assert state.ledger["P1"].status == guidance.STATUS_APPLIED
    assert state.ledger["P2"].status == guidance.STATUS_REFUTED
    assert state.ledger["P2"].failures == 2


def test_the_ledger_ignores_infrastructure_and_counts_an_iteration_once():
    state = _state_with()

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
    ("decision", "expected"),
    [
        ("KEEP", None),
        ("REVERT_PERF", guidance.REASON_BATCH_NO_GAIN),
        ("BUILD_FAILED", guidance.REASON_BATCH_NO_GAIN),
        ("REVERT_VALIDATION", guidance.REASON_BATCH_NO_GAIN),
        ("ORCHESTRATION_ERROR", None),
        ("REVERT_CONTENDED", None),
    ],
)
def test_only_a_measured_batch_without_a_gain_ends_guided(decision, expected):
    assert guidance.batch_exit(decision) == expected


def test_a_batch_is_one_plan_whose_parts_keep_their_order():
    plan = guidance.batch_plan([("playbook group G4 (GEMV path)", "gemv part"), ("", "tile part")])

    assert plan.startswith("# GUIDED batch: 2 part(s) in one session")
    assert "Submit only after the last part" in plan
    assert "[playbook-skip:...]" in plan
    assert plan.index("## Part 1 of 2: playbook group G4 (GEMV path)\n\ngemv part") < plan.index(
        "## Part 2 of 2\n\ntile part"
    )


def test_the_state_survives_a_round_trip_and_an_unreadable_file_starts_afresh(tmp_path):
    store = WikiGuidanceStore(tmp_path)
    state = _state_with()
    state.add_build(parse_playbook(json.dumps(SECOND_BUILD))[0], iteration=4)
    record_outcome(state, iteration=4, plan_text="[playbook:P1]", decision="KEEP")
    state.inflight_iteration, state.inflight_plan = 5, "[playbook:P3]"
    store.save(state)

    loaded = store.load()
    assert loaded.to_dict() == state.to_dict()
    assert json.loads(store.playbook_path.read_text())["items"][2]["id"] == "P3"

    store.state_path.write_text("{not json")
    assert store.load().to_dict() == GuidanceState().to_dict()


def test_free_shows_what_the_playbook_settled_and_no_batch():
    state = _state_with()
    state.phase, state.reason_codes = guidance.PHASE_FREE, [guidance.REASON_NO_USEFUL_DIRECTION]

    rendered = guidance.render_for_planning(state)

    assert "batch" not in rendered
    assert "FREE phase" in rendered["instructions"] and "NO_USEFUL_DIRECTION" in rendered["instructions"]
    assert "Phase: FREE (NO_USEFUL_DIRECTION)" in guidance.render_for_implementer(state)
    assert guidance.render_for_implementer(GuidanceState()) == ""


# --- the loop -------------------------------------------------------------------------------------------------------


class _GuidedService:
    """An orchestration service whose plans and builder answers are scripted, one per round."""

    def __init__(self, plans, *, builds=()):
        self.plans = list(plans)
        # Each answer is the builder's text, or the error it raises; once they run out it finds nothing more.
        self.answers = list(builds)
        self.build_contexts = []
        self.contexts = []
        self.lanes = []

    @property
    def builds(self):
        return len(self.build_contexts)

    async def build_wiki_playbook(self, context, *, usage=None):
        self.build_contexts.append(context)
        answer = self.answers.pop(0) if self.answers else json.dumps(DECLINED)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)

    async def run(self, context, *, usage=None, lanes=1):
        self.contexts.append(context)
        self.lanes.append(lanes)
        plans = self.plans.pop(0)
        return OrchestrationRunResult(dispatch_plan=None, optimization_plans=tuple(plans))


def _guided_loop(tmp_path, monkeypatch, *, wiki_guided=True, rounds=3, resume=False, max_time_hours=1.0, **ic):
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
    config = replace(base, wiki_guided=wiki_guided, max_time_hours=max_time_hours, **ic)
    loop = IterationLoop(
        config,
        ExperimentTracker(tmp_path / "experiments"),
        config=SimpleNamespace(experiments_dir=workspace / "forge_experiments", gpu_target="gfx942"),
        resume=resume,
    )
    monkeypatch.setattr(loop, "_time_remaining", lambda: _AMPLE_BUDGET_SEC if len(loop.results) < rounds else 0.0)
    return loop, workspace


def _editing_agent(histories, reports=()):
    """An Implementer that writes a new kernel and ends with the next scripted closing message."""
    closing = list(reports)

    async def agent(kernel_path, history, session_sink):
        histories.append(history)
        session_sink["plan"] = history
        session_sink["final_text"] = closing.pop(0) if closing else ""
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


def _plan_file(workspace, iteration):
    return (
        workspace / "forge_experiments" / "orchestration" / f"iter_{iteration:03d}" / "optimization_plan.md"
    ).read_text()


def _lane_recorder(seen):
    """A lane-session factory that only records that a lane session was asked for."""

    def factory(_lane_dir, _serialized_driver):
        async def agent(_kernel_file, prompt):
            seen.append(prompt)

        return agent

    return factory


def _run(loop, service, agent, **kwargs):
    asyncio.run(loop.run(agent_fn=agent, supervisor_fn=_unused_supervisor, orchestration_service=service, **kwargs))


def test_without_the_flag_planning_is_unchanged(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, wiki_guided=False, rounds=1)
    service = _GuidedService([["tune the loop [playbook:P1]"]], builds=[PLAYBOOK])
    histories: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))

    _run(loop, service, _editing_agent(histories))

    assert service.builds == 0
    assert service.contexts[0].wiki_guidance is None
    assert "wiki_guidance" not in service.contexts[0].to_prompt_dict()
    assert "## Wiki guidance" not in histories[0]
    result = _events(workspace, "iteration_result")[0]
    assert "guidance_phase" not in result and "wiki_batch" not in result
    assert not (workspace / "forge_experiments" / "wiki_guidance").exists()


def test_each_guided_round_builds_its_batch_and_implements_it_in_one_session(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3, lanes=2)
    service = _GuidedService(
        [["body part\n[playbook:P1]", "grid part\n[playbook:P2]"], ["epilogue part\n[playbook:P3]"], ["free plan"]],
        builds=[PLAYBOOK, SECOND_BUILD],
    )
    histories: list[str] = []
    lane_sessions: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP", "KEEP", "REVERT"]))
    # The first session reports what it landed; the second writes no item lines, so its plan's citations count.
    agent = _editing_agent(histories, reports=["done\n[playbook:P1]\n[playbook-skip:P2] it slowed the kernel", ""])

    _run(loop, service, agent, agent_factory=_lane_recorder(lane_sessions))

    # One session per round, whatever the lane width, and every round asks the builder until it finds nothing more.
    assert (len(histories), lane_sessions, service.builds) == (3, [], 3)
    assert service.lanes[:2] == [2, 1]
    first = _plan_file(workspace, 1)
    assert first.startswith("# GUIDED batch: 2 part(s) in one session")
    assert first.index("## Part 1 of 2: playbook group G1 (loop body)\n\nbody part") < first.index(
        "## Part 2 of 2: playbook group G2 (grid order)\n\ngrid part"
    )
    assert "## Part 1 of 1: playbook group G3 (epilogue)\n\nepilogue part" in _plan_file(workspace, 2)
    assert "Read " in histories[0] and "Phase: GUIDED" in histories[0]
    # The second build sees what the first batch settled, and its items join the playbook under fresh ids.
    assert service.build_contexts[0].wiki_guidance is None
    history = service.build_contexts[1].wiki_guidance["playbook"]["items"]
    assert [(item["id"], item["status"]) for item in history] == [("P1", "APPLIED"), ("P2", "SKIPPED")]
    assert service.contexts[1].wiki_guidance["batch"] == ["G3"]
    built = [
        (event["iter"], event["build"], event["items"], event["groups"])
        for event in _events(workspace, "wiki_playbook_built")
    ]
    assert built == [(1, 1, ["P1", "P2"], ["G1", "G2"]), (2, 2, ["P3"], ["G3"])]
    assert [event["parts"] for event in _events(workspace, "wiki_batch_planned")] == [2, 1]
    reports = [
        (event["iter"], event["source"], event.get("landed"), event.get("dropped"))
        for event in _events(workspace, "wiki_batch_report")
    ]
    assert reports == [(1, "session", ["P1"], ["P2"]), (2, "plan", None, None)]
    ledger = WikiGuidanceStore(workspace).load().ledger
    assert [ledger[item].status for item in ("P1", "P2", "P3")] == ["APPLIED", "SKIPPED", "APPLIED"]
    results = _events(workspace, "iteration_result")
    assert [event.get("wiki_batch") for event in results] == [["G1", "G2"], ["G3"], None]
    assert [event["guidance_phase"] for event in results] == ["GUIDED", "GUIDED", "FREE"]
    # The third build finds nothing more, so that round already plans FREE.
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["phase"], changed["reason_codes"]) == (3, "FREE", ["NO_USEFUL_DIRECTION"])
    assert [event["iter"] for event in _events(workspace, "wiki_playbook_declined")] == [3]
    assert service.contexts[2].wiki_guidance["phase"] == "FREE"
    assert "Phase: FREE (NO_USEFUL_DIRECTION)" in histories[2]
    assert (workspace / "forge_experiments" / "wiki_guidance" / "playbook_response_002.md").is_file()


def test_a_batch_without_a_gain_ends_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"], ["free plan"]], builds=[PLAYBOOK, PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (1, ["BATCH_NO_GAIN"])
    assert service.builds == 1
    assert service.contexts[1].wiki_guidance["phase"] == "FREE"
    ledger = WikiGuidanceStore(workspace).load().ledger
    assert [ledger[item].status for item in ("P1", "P2")] == ["REFUTED", "REFUTED"]


def test_nothing_worth_a_batch_on_the_first_round_plans_freely_at_once(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    service = _GuidedService([["one"], ["two"]], builds=[DECLINED])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 1
    assert [context.wiki_guidance for context in service.contexts] == [None, None]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (1, ["NO_USEFUL_DIRECTION"])


def test_an_unusable_answer_plans_as_without_the_flag(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    service = _GuidedService([["plan one"], ["plan two"]], builds=["no JSON here"])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 1
    assert [context.wiki_guidance for context in service.contexts] == [None, None]
    [unavailable] = _events(workspace, "wiki_playbook_unavailable")
    assert unavailable["settled"] is True
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["phase"], changed["reason_codes"]) == ("FREE", ["PLAYBOOK_UNAVAILABLE"])
    response = workspace / "forge_experiments" / "wiki_guidance" / "playbook_response_001.md"
    assert response.read_text() == "no JSON here"


def test_an_unusable_answer_from_the_backend_ends_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    service = _GuidedService([["plan"]], builds=[OrchestrationOutputError("playbook returned an incomplete response")])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT"]))

    _run(loop, service, _editing_agent([]))

    [changed] = _events(workspace, "wiki_guidance_phase")
    assert changed["reason_codes"] == ["PLAYBOOK_UNAVAILABLE"]


def test_a_builder_outage_is_retried_on_the_next_round(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    outage = OrchestrationInfrastructureError("playbook backend exceeded 600s timeout")
    service = _GuidedService([["plan one"], ["plan two [playbook:P1]"]], builds=[outage, PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "KEEP"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 2
    assert service.contexts[0].wiki_guidance is None
    assert service.contexts[1].wiki_guidance["batch"] == ["G1", "G2"]
    assert [event["settled"] for event in _events(workspace, "wiki_playbook_unavailable")] == [False]
    assert _events(workspace, "wiki_guidance_phase") == []
    assert WikiGuidanceStore(workspace).load().ledger["P1"].status == "APPLIED"


def test_a_second_consecutive_outage_gives_up_on_the_knowledge_base(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    outage = OrchestrationInfrastructureError("playbook backend exceeded 600s timeout")
    service = _GuidedService([["one"], ["two"], ["three"]], builds=[outage, outage])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 2
    assert [event["settled"] for event in _events(workspace, "wiki_playbook_unavailable")] == [False, True]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (2, ["PLAYBOOK_UNAVAILABLE"])


def _conflicting_lanes(tmp_path, monkeypatch, *, wiki_guided, plans, decisions):
    """A two-lane FREE round whose lanes both rewrite the kernel's one line, so lane two stops applying once lane one
    is kept. A --wiki-guided campaign plans FREE from the start when the builder finds nothing to build."""
    loop, workspace = _guided_loop(tmp_path, monkeypatch, wiki_guided=wiki_guided, rounds=2, lanes=2)
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 1\n"
    service = _GuidedService(plans, builds=[DECLINED])
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
        plans=[["lane one: body", "lane two: grid"]],
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
    assert "lane two: grid" in plan
    lane_diff, since = plan.split("## The lane's diff")[1].split("## What the kept changes did")
    assert "+    return 12" in lane_diff
    assert "+    return 11" in since


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
    service = _GuidedService([["body [playbook:P1]"]], builds=[PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))
    _run(loop, service, _editing_agent([]))

    resumed, _ = _guided_loop(tmp_path, monkeypatch, rounds=1, resume=True)
    assert resumed.wiki_guidance.phase == "GUIDED"
    assert resumed.wiki_guidance.ledger["P1"].status == "APPLIED"
    later = _GuidedService([["epilogue [playbook:P3]"]], builds=[SECOND_BUILD])
    monkeypatch.setattr(resumed, "run_one_iteration", _measured(["KEEP"]))
    # A second candidate, distinct from the KEEP the first process committed.
    _run(resumed, later, _editing_agent(["first process"]))
    # The resumed build sees the first process's ledger and numbers its item after it.
    assert [item["status"] for item in later.build_contexts[0].wiki_guidance["playbook"]["items"]] == [
        "APPLIED",
        "PENDING",
    ]
    assert WikiGuidanceStore(workspace).load().ledger["P3"].status == "APPLIED"

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


def test_the_builder_reads_the_knowledge_base_and_the_earlier_batches(tmp_path):
    backend = _McpBackend(json.dumps(PLAYBOOK))
    tool = KnowledgeWikiTool(root=tmp_path, log_path=tmp_path / "calls.jsonl")
    agent = OrchestrationAgent(backend=backend, timeout_sec=60, max_turns=8, knowledge_wiki=tool)

    text = asyncio.run(agent.build_wiki_playbook(_context()))
    asyncio.run(agent.build_wiki_playbook(_guided_context()))

    assert parse_playbook(text)[0].item_ids() == ("P1", "P2")
    first, second = backend.specs
    assert first.role == "playbook"
    assert first.writable is False
    assert set(tool.tool_names) <= set(first.tool_policy.extra_tools)
    assert "knowledge_wiki" in first.mcp_servers
    assert first.mcp_servers["knowledge_wiki"].env["FORGE_KNOWLEDGE_WIKI_ROLE"] == "playbook"
    assert json.loads(first.user_prompt)["context"]["gpu_target"] == _context().gpu_target
    assert "wiki_guidance" not in json.loads(first.user_prompt)["context"]
    history = json.loads(second.user_prompt)["context"]["wiki_guidance"]["playbook"]["items"]
    assert [item["status"] for item in history] == ["PENDING", "APPLIED", "FAILED"]


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


def _batch_context(batch):
    state = _state_with(THREE_GROUPS)
    state.batch = list(batch)
    return replace(_context(), wiki_guidance=guidance.render_for_planning(state))


def test_a_batch_round_plans_one_part_per_group_without_dividing_the_round_by_code():
    backend = _McpBackend("the part's plan")
    agent = OrchestrationAgent(backend=backend, timeout_sec=60, max_turns=8)
    context = _batch_context(["G2", "G1"])
    dispatch = orchestration_module.playbook_group_dispatch(context, lanes=2)
    # Analyses come back in completion order, not batch order.
    outcomes = [
        SpecialistOutcome(
            assignment_id=assignment.assignment_id,
            role_id=assignment.role_id,
            duration_sec=0.1,
            content=f"analysis: {assignment.reason}",
        )
        for assignment in reversed(dispatch.assignments)
    ]

    plans = asyncio.run(agent.synthesize_lane_plan_results(context, outcomes, dispatch, {}, lanes=2))

    assert [plan.ground for plan in plans] == [
        "batch part 1 of 2: playbook group G2",
        "batch part 2 of 2: playbook group G1",
    ]
    # No partition turn: every call plans one part.
    assert {spec.system_prompt for spec in backend.specs} == {orchestration_module._BATCH_PART_SYSTEM_PROMPT}
    parts = sorted((json.loads(spec.user_prompt)["batch_part"] for spec in backend.specs), key=lambda p: p["position"])
    assert [(part["position"], part["of"], part["group"]) for part in parts] == [(1, 2, "G2"), (2, 2, "G1")]
    assert parts[1]["earlier_parts"] == ["G2 grid order"] and parts[0]["later_parts"] == ["G1 loop body"]
    analyses = json.loads(backend.specs[0].user_prompt)["specialist_analyses"]
    assert [entry["analysis"].split(" (")[0] for entry in analyses] == [
        "analysis: Playbook group G2",
        "analysis: Playbook group G1",
    ]
    assert agent.structured_output_diagnostics["batch"] == {"parts": ["G2", "G1"], "planned": 2}


def test_a_guided_round_analyzes_its_batch_groups_and_no_others():
    service, _agent, pool = _service({"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE})

    asyncio.run(service.run(_batch_context(["G3"]), lanes=3))

    assert [assignment.reason.split(" (")[0] for assignment in pool.assignments] == ["Playbook group G3"]


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
    state = _state_with(THREE_GROUPS)
    state.phase = phase
    state.reason_codes = [] if phase == "GUIDED" else ["NO_USEFUL_DIRECTION"]
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


def test_a_one_lane_guided_round_takes_the_first_open_batch_group():
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


def test_the_flag_needs_a_knowledge_base():
    assert _validate_wiki_guided(False, "") is False
    assert _validate_wiki_guided(True, "/wiki") is True
    with pytest.raises(click.BadParameter, match="needs a knowledge base"):
        _validate_wiki_guided(True, "")


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


def test_the_flag_reaches_the_loop_config(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    _with_knowledge_base(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, ["--wiki-guided"])

    assert result.exit_code == 0, result.output
    assert captured["loops"][-1].ic.wiki_guided is True
    assert "Wiki-guided planning: on" in result.output


def test_guided_planning_is_off_unless_an_operator_asks(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, [])

    assert result.exit_code == 0, result.output
    assert captured["loops"][-1].ic.wiki_guided is False


def test_the_guided_budget_share_is_no_longer_an_option(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    _with_knowledge_base(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, ["--wiki-guided", "--guided-budget-fraction", "0.5"])

    assert result.exit_code != 0
    assert "--guided-budget-fraction" in result.output
    assert captured["loops"] == []


def test_the_flag_without_a_knowledge_base_is_refused(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, ["--wiki-guided"])

    assert result.exit_code != 0
    assert "needs a knowledge base" in result.output
    assert captured["loops"] == []
