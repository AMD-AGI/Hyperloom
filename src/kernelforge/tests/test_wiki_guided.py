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
from kernelforge.orchestrator.contracts import OrchestrationRunResult
from kernelforge.orchestrator.orchestration import OrchestrationAgent, OrchestrationInfrastructureError
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
        if setup == "exhausted":
            for record in state.ledger.values():
                record.status = guidance.STATUS_SKIPPED

    assert decide_phase(state, search_reasons=reasons, budget_used_fraction=used, budget_fraction=0.35) == expected


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


def test_the_guided_budget_share_ends_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2, max_time_hours=4.0)
    # Half of a four-hour budget is spent by the second round, past the default 35% share.
    monkeypatch.setattr(
        loop,
        "_time_remaining",
        lambda: _AMPLE_BUDGET_SEC if not loop.results else (7200.0 if len(loop.results) < 2 else 0.0),
    )
    service = _GuidedService([["one [playbook:P1]"], ["two"]])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert [context.wiki_guidance["phase"] for context in service.contexts] == ["GUIDED", "FREE"]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (2, ["GUIDED_BUDGET_SPENT"])
    assert changed["budget_used_fraction"] == 0.5


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
