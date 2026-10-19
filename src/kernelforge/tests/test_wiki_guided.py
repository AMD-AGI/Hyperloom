# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knowledge-base-guided planning: the first round's batch, its one-session iteration, the ledger, and the KernelForge
rounds after it."""

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
from kernelforge.knowledge.wiki_tool import KnowledgeWikiTool, tool as wiki_tool_module
from kernelforge.loop import wiki_guidance as guidance
from kernelforge.loop import wiki_playbook as playbook_module
from kernelforge.loop.run_state import LoopStateStore
from kernelforge.loop.runner import IterationConfig, IterationLoop, IterationResult, LoopAgents
from kernelforge.loop.wiki_guidance import (
    GuidanceState,
    WikiGuidanceStore,
    cited_items,
    record_outcome,
)
from kernelforge.loop.wiki_playbook import parse_playbook
from kernelforge.orchestrator import orchestration as orchestration_module
from kernelforge.orchestrator import wiki_batch
from kernelforge.orchestrator.contracts import (
    DispatchIntent,
    LaneDrop,
    OrchestrationRunResult,
    PlanCriticOutcome,
    SpecialistOutcome,
    SpecialistRound,
    SynthesizedPlan,
)
from kernelforge.orchestrator.dispatch_binding import bind_dispatch
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
from kernelforge.tests.test_loop_runner import _AMPLE_BUDGET_SEC, _make_loop, _unused_supervisor
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
DECLINED = {"summary": "the profile shows nothing more the knowledge base addresses", "groups": [], "items": []}
THREE_GROUPS = {
    "summary": "three parts",
    "groups": [*PLAYBOOK["groups"], {"id": "G3", "title": "epilogue"}],
    "items": [*PLAYBOOK["items"], {"id": "P3", "group": "G3", "title": "nontemporal stores", "wiki_pages": ["page"]}],
}


def _state_with(payload=PLAYBOOK) -> GuidanceState:
    build, notes = parse_playbook(json.dumps(payload))
    assert notes == []
    state = GuidanceState()
    state.add_build(build)
    return state


# --- the build, the playbook and its ledger -------------------------------------------------------------------------


def test_a_build_keeps_only_items_a_knowledge_base_page_supports():
    payload = json.loads(json.dumps(PLAYBOOK))
    payload["items"] += [
        {"id": "P3", "group": "G1", "title": "the builder's own idea", "wiki_pages": []},
        {"id": "P4", "group": "G9", "title": "an unknown group", "wiki_pages": ["page"]},
        {"id": "X5", "group": "G1", "title": "a malformed id", "wiki_pages": ["page"]},
    ]

    build, notes = parse_playbook("Here it is:\n" + json.dumps(payload))

    # A malformed id is the answer's numbering, not its content: the item is kept under the next free id.
    assert build.item_ids() == ("P1", "P2", "P5")
    assert "renumbered item X5 as P5" in notes
    assert any("P3" in note and "no knowledge base page" in note for note in notes)
    assert any("P4" in note and "G9" in note for note in notes)
    assert len(notes) == 3


def test_an_answer_numbered_its_own_way_is_renumbered_not_refused():
    # The shape of a real answer that numbered its items N1..N6 and lost the whole round to PLAYBOOK_UNAVAILABLE.
    answer = {
        "summary": "split-K and the merge",
        "groups": [
            {"id": "G1", "title": "split-K loads"},
            {"id": "merge", "title": "in-kernel merge"},
        ],
        "items": [
            {"id": "N1", "group": "G1", "title": "alignment hint", "wiki_pages": ["a"], "conflicts_with": ["N2"]},
            {"id": "N2", "group": "G1", "title": "retune the buckets", "wiki_pages": ["b"]},
            {"id": "N3", "group": "MERGE", "title": "last-workgroup merge", "wiki_pages": ["c"]},
        ],
    }

    build, notes = parse_playbook(json.dumps(answer))

    assert [(item.item_id, item.group, item.conflicts_with) for item in build.items] == [
        ("P1", "G1", ("P2",)),
        ("P2", "G1", ()),
        ("P3", "G2", ()),
    ]
    assert [group.group_id for group in build.groups] == ["G1", "G2"]
    assert notes == [
        "renumbered group MERGE as G2",
        "renumbered item N1 as P1",
        "renumbered item N2 as P2",
        "renumbered item N3 as P3",
    ]
    # The campaign playbook, whose ids key the ledger, is read as stored and never renumbered.
    with pytest.raises(ValueError, match="N1"):
        playbook_module.Playbook.from_dict(answer)


def test_an_answer_with_nothing_more_is_told_from_an_unusable_one():
    assert parse_playbook(json.dumps(DECLINED))[0] is None
    assert playbook_module.declined(json.dumps(DECLINED))
    assert not playbook_module.declined("I could not find anything.")
    assert not playbook_module.declined(json.dumps(PLAYBOOK))
    unusable = {"groups": [{"id": "G1", "title": "x"}], "items": [{"id": "P1", "group": "G1", "title": "x"}]}
    build, notes = parse_playbook(json.dumps(unusable))
    assert build is None
    assert "the playbook has no usable item" in notes
    assert not playbook_module.declined(json.dumps(unusable))


def test_a_build_implements_at_most_six_groups():
    payload = {
        "groups": [{"id": f"G{index}", "title": f"group {index}"} for index in range(1, 9)],
        "items": [
            {"id": f"P{index}", "group": f"G{index}", "title": "x", "wiki_pages": ["page"]} for index in range(1, 9)
        ],
    }

    build, notes = parse_playbook(json.dumps(payload))

    assert len(build.groups) == playbook_module.MAX_BATCH_GROUPS
    assert any("at most 6 groups" in note for note in notes)


def test_a_group_stays_open_while_it_holds_an_item_no_session_got_to():
    state = _state_with(THREE_GROUPS)
    assert state.open_groups() == ["G1", "G2", "G3"]
    assert {record.status for record in state.ledger.values()} == {guidance.STATUS_PENDING}

    report = "[playbook:P1]\n[playbook-skip:P2 slower]\n[playbook-untried:P3 out of time]"
    record_outcome(state, iteration=1, plan_text=report, decision="KEEP")

    assert state.open_groups() == ["G3"]
    state.batch = state.open_groups()
    rendered = guidance.render_for_planning(state)
    assert rendered["batch"] == ["G3"]
    assert rendered["playbook"]["summary"] == "three parts"


def test_a_plan_claims_and_declines_items_only_by_citation():
    plan = (
        "Implement the loop body change; P2 is mentioned in passing.\n"
        "[playbook:p1]\n[playbook-skip:P2] the grid is already XCD-ordered\n[playbook:P9]"
    )
    assert cited_items(plan, ["P1", "P2"]) == (["P1"], ["P2"])
    assert cited_items("mentions P1 without a citation", ["P1"]) == ([], [])
    assert cited_items("[playbook:P101]", ["P101", "P10"]) == (["P101"], [])


def test_a_session_report_cites_the_ids_that_open_each_entry_and_a_claim_wins():
    known = [f"P{index}" for index in range(1, 30)]
    # Closing lines as sessions write them: an id inside another entry's reason is not cited, "none" cites
    # nothing, and an item the report claims is in the kernel stays claimed whatever else mentions it.
    report = (
        "[playbook:P1, P2 (B=1 only), P14]\n"
        "[playbook-skip:P6 reverted, slower on perf5 (0.160 vs 0.146 ms); P15 needs 96 KB of LDS after P14]\n"
        "[playbook-skip: none. P21 is the only item in G11, and it is implemented in full]\n"
        "[playbook-skip:P2 kept only for B=1 shapes]"
    )

    cites = guidance.read_citations(report, known)

    assert (cites.claimed, cites.skipped, cites.untried) == (("P1", "P2", "P14"), ("P6", "P15"), ())
    assert cites.reasons == {
        "P6": "reverted, slower on perf5 (0.160 vs 0.146 ms)",
        "P15": "needs 96 KB of LDS after P14",
    }


def test_an_id_after_a_comma_opens_an_entry_only_when_its_reason_follows_it():
    known = [f"P{index}" for index in range(1, 30)]

    mention = guidance.read_citations("[playbook-skip:P6 slower than the split path, P21 would be needed first]", known)
    assert (mention.skipped, mention.reasons) == (
        ("P6",),
        {"P6": "slower than the split path, P21 would be needed first"},
    )

    entries = guidance.read_citations("[playbook-skip:P6 slower, P7: spilled 14 VGPRs, P8 (no gain), P9]", known)
    assert entries.skipped == ("P6", "P7", "P8", "P9")
    assert entries.reasons == {"P6": "slower", "P7": "spilled 14 VGPRs", "P8": "(no gain)"}

    # A claim names each item with what it is, so any id after a comma is claimed (a real SimpleMatmulModule line).
    claim = guidance.read_citations(
        "[playbook:P18 bf16x3 128x128 instantiation (truncated hi), P19 per-shape dispatch (K-based split rule), "
        "P20a split done at LDS-staging time]",
        known,
    )
    assert claim.claimed == ("P18", "P19")


def test_a_skip_of_an_item_whose_precondition_failed_settles_it():
    state = _state_with(THREE_GROUPS)
    report = "[playbook-skip:P2 not implemented: the grid is already XCD-ordered; P3 not attempted, out of time]"

    record_outcome(state, iteration=1, plan_text=report, decision="KEEP")

    assert [state.ledger[item].status for item in ("P2", "P3")] == ["SKIPPED", "PENDING"]


def test_only_a_sessions_latest_report_counts():
    first = "[playbook:P1] kept\n[playbook-skip:P2 spilled]\nSUBMIT_CANDIDATE"
    second = "Reverted P1.\n[playbook-skip:P1 reverted after the stop hook: slower on case 3]\n[playbook:P2]\nSUBMIT_CANDIDATE"
    text = f"{first}\nThe kernel is not faster yet.\n{second}"

    assert guidance.latest_report(text) == "\nThe kernel is not faster yet.\n" + second
    # A session cut off after it reported again, without submitting, is described by what it wrote last.
    assert guidance.latest_report(f"{first}\n[playbook-skip:P1 slower]").strip() == "[playbook-skip:P1 slower]"
    assert guidance.latest_report(f"{first}\nno new report\nSUBMIT_CANDIDATE").startswith("[playbook:P1]")
    assert guidance.latest_report("no item lines\nSUBMIT_CANDIDATE") == ""


def test_an_item_the_session_did_not_get_to_stays_open_with_its_reason():
    state = _state_with(THREE_GROUPS)
    report = "[playbook:P1]\n[playbook-skip:P2 slower on every case; P3 not attempted for lack of time]"

    assert record_outcome(state, iteration=1, plan_text=report, decision="KEEP") == ["P2", "P3", "P1"]

    assert [state.ledger[item].status for item in ("P1", "P2", "P3")] == ["APPLIED", "SKIPPED", "PENDING"]
    assert state.ledger["P2"].note == "skipped in iteration 1: slower on every case"
    assert state.ledger["P3"].note == "not tried in iteration 1: not attempted for lack of time"
    # The explicit line says the same whatever the reason's wording, and the builder sees each reason.
    record_outcome(state, iteration=2, plan_text="[playbook-untried:P3 the session ran short]", decision="KEEP")
    assert (state.ledger["P3"].status, state.ledger["P3"].iterations) == ("PENDING", [1, 2])
    items = {item["id"]: item for item in guidance.render_for_planning(state)["playbook"]["items"]}
    assert items["P3"]["note"] == "not tried in iteration 2: the session ran short"
    assert "note" not in items["P1"]


def test_a_direction_only_the_evidence_supports_joins_the_batch_marked_as_such():
    payload = json.loads(json.dumps(PLAYBOOK))
    payload["items"] += [
        {
            "id": "P3",
            "group": "G1",
            "title": "BF16x3 emulation",
            "support": "profiling",
            "wiki_pages": [],
            "applies_when": "the numerics probe passes at 0.02x the tolerance and cases 1-6 are MFMA-bound",
        },
        {"id": "P4", "group": "G1", "title": "no evidence named", "support": "profiling", "wiki_pages": []},
        {"id": "P5", "group": "G2", "title": "a page after all", "support": "profiling", "wiki_pages": ["page"]},
        {"id": "P6", "group": "G2", "title": "a hunch", "support": "hunch", "wiki_pages": []},
    ]

    build, notes = parse_playbook(json.dumps(payload))

    assert {item.item_id: item.support for item in build.items} == {
        "P1": "wiki",
        "P2": "wiki",
        "P3": "profiling",
        "P5": "wiki",
    }
    assert any("P4" in note and "applies_when" in note for note in notes)
    assert any("P6" in note and "neither wiki nor profiling" in note for note in notes)
    state = GuidanceState()
    state.add_build(build)
    rendered = guidance.render_for_planning(state)
    items = {item["id"]: item for item in rendered["playbook"]["items"]}
    assert (items["P3"]["support"], items["P3"]["wiki_pages"]) == ("profiling", [])
    assert "`support` is `profiling`" in rendered["instructions"]
    assert GuidanceState.from_dict(state.to_dict()).playbook.items[2].support == "profiling"


def test_the_ledger_settles_items_from_measured_decisions():
    state = _state_with()

    assert record_outcome(state, iteration=1, plan_text="[playbook:P1]", decision="KEEP") == ["P1"]
    assert record_outcome(state, iteration=2, plan_text="[playbook:P2]", decision="BUILD_FAILED") == ["P2"]
    assert state.ledger["P2"].status == guidance.STATUS_FAILED
    assert record_outcome(state, iteration=3, plan_text="[playbook:P2]", decision="REVERT_PERF") == ["P2"]

    assert state.ledger["P1"].status == guidance.STATUS_APPLIED
    assert (state.ledger["P2"].status, state.ledger["P2"].iterations) == (guidance.STATUS_REFUTED, [2, 3])


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


def test_a_batch_is_one_plan_whose_parts_keep_their_order():
    plan = guidance.batch_plan([("playbook group G4 (GEMV path)", "gemv part"), ("", "tile part")])

    assert plan.startswith("# Knowledge base batch: 2 part(s) in one session")
    assert "Submit only after the last part" in plan
    assert "[playbook-skip:...]" in plan and "[playbook-untried:...]" in plan
    assert plan.index("## Part 1 of 2: playbook group G4 (GEMV path)\n\ngemv part") < plan.index(
        "## Part 2 of 2\n\ntile part"
    )


def test_the_state_survives_a_round_trip_and_an_unreadable_file_starts_afresh(tmp_path):
    store = WikiGuidanceStore(tmp_path)
    state = _state_with()
    state.batch = ["G2"]
    record_outcome(state, iteration=4, plan_text="[playbook:P1]\n[playbook-untried:P2 out of time]", decision="KEEP")
    state.outcomes.append({"iteration": 4, "decision": "KEEP", "detail": ""})
    state.inflight_iteration, state.inflight_plan = 5, "[playbook:P2]"
    store.save(state)

    loaded = store.load()
    assert loaded.to_dict() == state.to_dict()
    assert json.loads(store.playbook_path.read_text())["items"][1]["id"] == "P2"

    store.state_path.write_text("{not json")
    assert store.load().to_dict() == GuidanceState().to_dict()


def test_after_the_round_the_planners_see_only_what_it_settled():
    state = _state_with(THREE_GROUPS)
    report = "[playbook:P1]\n[playbook-skip:P2 slower on every case]\n[playbook-untried:P3 out of time]"
    record_outcome(state, iteration=1, plan_text=report, decision="KEEP")
    state.outcomes.append({"iteration": 1, "decision": "KEEP", "detail": ""})
    state.phase, state.reason_codes = guidance.PHASE_FREE, [guidance.REASON_ROUND_DONE]

    assert guidance.render_for_planning(state) == {
        "phase": "FREE",
        "knowledge_base_round": (
            "The campaign's knowledge base round measured iteration 1 KEEP.\n"
            "In the incumbent: P1 deferred normalization.\n"
            "Tried and dropped, or found not to apply: P2 XCD remap (skipped in iteration 1: slower on every case).\n"
            "Not reached: P3 nontemporal stores (not tried in iteration 1: out of time)."
        ),
    }
    assert guidance.render_for_implementer(state) == ""
    # A round that measured nothing leaves the planners nothing to read.
    state.outcomes = []
    assert guidance.render_for_planning(state) is None


def test_a_failed_batch_is_named_with_its_failure():
    state = _state_with()
    record_outcome(state, iteration=1, plan_text="[playbook:P1,P2]", decision="REVERT_VALIDATION")
    state.outcomes.append({"iteration": 1, "decision": "REVERT_VALIDATION", "detail": "case 3: max abs error 0.4"})
    state.phase = guidance.PHASE_FREE

    summary = guidance.render_for_planning(state)["knowledge_base_round"]

    assert summary.splitlines() == [
        "The campaign's knowledge base round measured iteration 1 REVERT_VALIDATION (case 3: max abs error 0.4).",
        "In a batch that failed: P1 deferred normalization; P2 XCD remap.",
    ]


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
        if isinstance(plans, OrchestrationRunResult):
            return plans
        return OrchestrationRunResult(dispatch_plan=None, optimization_plans=tuple(plans))


def _guided_loop(tmp_path, monkeypatch, *, wiki_guided=True, rounds=3, resume=False, max_time_hours=1.0, **ic):
    """A loop built by the production constructor with guided planning on (or off), over a knowledge base."""
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
    loop = IterationLoop(config, ExperimentTracker(tmp_path / "experiments"), config=_config(tmp_path), resume=resume)
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


def _config(tmp_path):
    root = tmp_path / "wiki"
    return SimpleNamespace(
        experiments_dir=tmp_path / "workspace" / "forge_experiments",
        gpu_target="gfx942",
        knowledge_wiki_root=str(root),
        knowledge_wiki_log=str(tmp_path / "wiki_calls.jsonl"),
    )


def _offered(loop) -> bool:
    """Whether a session built now would be offered the campaign's knowledge base tools."""
    return bool(wiki_tool_module.wiki_tool_names(KnowledgeWikiTool.from_config(loop.config)))


def _measured(decisions):
    """Measurements scripted by decision: KEEP, REVERT (slower) or INVALID (the canonical check failed)."""
    queue = list(decisions)

    async def measured(iteration, plan=""):
        decision = queue.pop(0)
        kept = decision == "KEEP"
        speedup = 1.0 + 0.5 * iteration if kept else 1.0
        return IterationResult(
            iteration=iteration,
            duration_sec=0.01,
            validation_passed=decision != "INVALID",
            validation_summary="case 3: max abs error 0.4" if decision == "INVALID" else "canonical validation passed",
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


def _run(loop, service, agent, *, agent_factory=None):
    agents = LoopAgents(
        agent_fn=agent, agent_factory=agent_factory, orchestration_service=service, supervisor_fn=_unused_supervisor
    )
    asyncio.run(loop.run(agents))


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


def test_the_first_round_implements_its_batch_in_one_session_and_then_kernelforge_plans(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2, lanes=3)
    service = _GuidedService(
        [["body part\n[playbook:P1]", "grid part\n[playbook:P2]"], ["free plan, still citing [playbook:P2]"]],
        builds=[PLAYBOOK],
    )
    histories: list[str] = []
    lane_sessions: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP", "REVERT"]))
    agent = _editing_agent(histories, reports=["done\n[playbook:P1]\n[playbook-skip:P2 it slowed the kernel]"])
    assert _offered(loop)

    _run(loop, service, agent, agent_factory=_lane_recorder(lane_sessions))

    # One session for the batch, one part per group; the next round plans at the configured width.
    assert (len(histories), lane_sessions, service.builds, service.lanes) == (2, [], 1, [2, 3])
    first = _plan_file(workspace, 1)
    assert first.startswith("# Knowledge base batch: 2 part(s) in one session")
    assert first.index("## Part 1 of 2: playbook group G1 (loop body)\n\nbody part") < first.index(
        "## Part 2 of 2: playbook group G2 (grid order)\n\ngrid part"
    )
    assert service.build_contexts[0].wiki_guidance is None
    assert service.contexts[0].wiki_guidance["batch"] == ["G1", "G2"]
    assert guidance.IMPLEMENTER_NOTE in histories[0]
    [built] = _events(workspace, "wiki_playbook_built")
    assert (built["iter"], built["items"], built["groups"]) == (1, ["P1", "P2"], ["G1", "G2"])
    [report] = _events(workspace, "wiki_batch_report")
    assert (report["source"], report["landed"], report["dropped"]) == ("session", ["P1"], ["P2"])
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["phase"], changed["reason_codes"]) == (1, "FREE", ["ROUND_DONE"])
    # After the round, KernelForge as without the flag: no batch, no Implementer note, no knowledge base tools, and
    # the planners told only what the round settled. Its plans no longer move the ledger.
    assert service.contexts[1].wiki_guidance == {
        "phase": "FREE",
        "knowledge_base_round": (
            "The campaign's knowledge base round measured iteration 1 KEEP.\n"
            "In the incumbent: P1 deferred normalization.\n"
            "Tried and dropped, or found not to apply: P2 XCD remap (skipped in iteration 1: it slowed the kernel)."
        ),
    }
    assert "## Wiki guidance" not in histories[1]
    assert not _offered(loop)
    state = WikiGuidanceStore(workspace).load()
    assert [(state.ledger[item].status, state.ledger[item].iterations) for item in ("P1", "P2")] == [
        ("APPLIED", [1]),
        ("SKIPPED", [1]),
    ]
    assert [event["iter"] for event in _events(workspace, "wiki_playbook_ledger")] == [1]
    results = _events(workspace, "iteration_result")
    assert [event.get("wiki_batch") for event in results] == [["G1", "G2"], None]
    assert [event["guidance_phase"] for event in results] == ["GUIDED", "FREE"]
    assert (workspace / "forge_experiments" / "wiki_guidance" / "playbook_response.md").is_file()


def test_a_kept_batch_that_left_items_untried_gets_one_round_for_them(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    service = _GuidedService(
        [["body [playbook:P1]", "grid [playbook:P2]", "epilogue [playbook:P3]"], ["epilogue again"], ["free plan"]],
        builds=[THREE_GROUPS],
    )
    histories: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP", "KEEP", "KEEP"]))
    reports = ["[playbook:P1, P2]\n[playbook-untried:P3 out of session time]", "[playbook-untried:P3 ran short again]"]

    _run(loop, service, _editing_agent(histories, reports=reports))

    # The completion round plans only the group still open, from the same playbook: the builder is not asked again.
    assert service.builds == 1
    assert service.lanes[:2] == [3, 1]
    assert service.contexts[1].wiki_guidance["batch"] == ["G3"]
    assert "## Part 1 of 1: playbook group G3 (epilogue)\n\nepilogue again" in _plan_file(workspace, 2)
    assert guidance.IMPLEMENTER_NOTE in histories[1]
    # It is the last knowledge base round even though P3 is still open.
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (2, ["ROUND_DONE"])
    assert service.contexts[2].wiki_guidance["knowledge_base_round"].splitlines() == [
        "The campaign's knowledge base round measured iteration 1 KEEP, iteration 2 KEEP.",
        "In the incumbent: P1 deferred normalization; P2 XCD remap.",
        "Not reached: P3 nontemporal stores (not tried in iteration 2: ran short again).",
    ]
    results = _events(workspace, "iteration_result")
    assert [event.get("wiki_batch") for event in results] == [["G1", "G2", "G3"], ["G3"], None]
    assert [event["guidance_phase"] for event in results] == ["GUIDED", "GUIDED", "FREE"]


def test_a_batch_that_is_not_kept_ends_the_round_with_its_failure_for_the_planners(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"], ["free plan"]], builds=[PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["INVALID", "REVERT"]))
    # Items the session did not get to get no completion round after a batch that was not kept.
    agent = _editing_agent([], reports=["[playbook:P1]\n[playbook-untried:P2 out of time]"])

    _run(loop, service, agent)

    assert service.builds == 1
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (1, ["ROUND_DONE"])
    assert service.contexts[1].wiki_guidance["knowledge_base_round"].splitlines() == [
        "The campaign's knowledge base round measured iteration 1 REVERT_VALIDATION (case 3: max abs error 0.4).",
        "In a batch that failed: P1 deferred normalization.",
        "Not reached: P2 XCD remap (not tried in iteration 1: out of time).",
    ]
    assert not _offered(loop)


def test_a_session_the_stop_hook_sent_back_is_credited_from_its_latest_report(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"]], builds=[PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))
    # The first submission kept P1; the kernel was not faster, so the session reverted P1 and landed P2.
    closing = (
        "[playbook:P1] kept\n[playbook-skip:P2 spilled 14 VGPRs]\nSUBMIT_CANDIDATE\n"
        "Reverted P1 and fixed P2's spill.\n[playbook-skip:P1 slower on case 3]\n[playbook:P2] kept\nSUBMIT_CANDIDATE"
    )

    _run(loop, service, _editing_agent([], reports=[closing]))

    [report] = _events(workspace, "wiki_batch_report")
    assert (report["source"], report["landed"], report["dropped"]) == ("session", ["P2"], ["P1"])
    ledger = WikiGuidanceStore(workspace).load().ledger
    assert [ledger[item].status for item in ("P1", "P2")] == ["SKIPPED", "APPLIED"]
    assert ledger["P1"].note == "skipped in iteration 1: slower on case 3"


def test_an_answer_numbered_its_own_way_still_gives_the_round_its_batch(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    answer = json.loads(json.dumps(PLAYBOOK))
    for index, item in enumerate(answer["items"], start=1):
        item["id"] = f"N{index}"
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"]], builds=[answer])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))

    _run(loop, service, _editing_agent([]))

    assert _events(workspace, "wiki_playbook_unavailable") == []
    [built] = _events(workspace, "wiki_playbook_built")
    assert (built["items"], built["groups"]) == (["P1", "P2"], ["G1", "G2"])
    assert service.contexts[0].wiki_guidance["batch"] == ["G1", "G2"]
    assert [event["guidance_phase"] for event in _events(workspace, "iteration_result")] == ["GUIDED"]


def _failing_first(failures, agent):
    """An Implementer whose first sessions fail on its backend, then ``agent``."""
    queue = list(failures)

    async def session(kernel_path, history, session_sink):
        if queue:
            raise queue.pop(0)
        return await agent(kernel_path, history, session_sink)

    return session


def test_a_batch_session_that_measured_nothing_is_run_once_more(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    parts = ["body [playbook:P1]", "grid [playbook:P2]"]
    service = _GuidedService([parts, parts, ["free plan"]], builds=[PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP", "REVERT"]))
    agent = _failing_first([RuntimeError("agent backend unavailable")], _editing_agent([]))

    _run(loop, service, agent)

    assert service.builds == 1
    assert [context.wiki_guidance.get("batch") for context in service.contexts[:2]] == [["G1", "G2"]] * 2
    results = _events(workspace, "iteration_result")
    assert [event["decision"] for event in results] == ["AGENT_ERROR", "KEEP", "REVERT_PERF"]
    assert [event.get("wiki_batch") for event in results] == [["G1", "G2"], ["G1", "G2"], None]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (2, ["ROUND_DONE"])
    summary = service.contexts[2].wiki_guidance["knowledge_base_round"]
    assert summary.startswith("The campaign's knowledge base round measured iteration 2 KEEP.\n")


def test_a_second_batch_session_that_measured_nothing_gives_up_on_the_batch(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=3)
    parts = ["body [playbook:P1]", "grid [playbook:P2]"]
    service = _GuidedService([parts, parts, ["free plan"]], builds=[PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT"]))
    outage = RuntimeError("agent backend unavailable")

    _run(loop, service, _failing_first([outage, outage], _editing_agent([])))

    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (2, ["ROUND_DONE"])
    assert service.contexts[2].wiki_guidance is None
    assert not _offered(loop)


def test_a_one_group_batch_whose_plan_cites_nothing_is_still_the_batch_its_measurement_decides(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    one_group = {"summary": "the epilogue", "groups": [PLAYBOOK["groups"][0]], "items": [PLAYBOOK["items"][0]]}
    service = _GuidedService([["the loop body part, written without citations"], ["free plan"]], builds=[one_group])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))
    agent = _editing_agent([], reports=["[playbook-skip:P1 slower on every case]"])

    _run(loop, service, agent)

    assert "## Part 1 of 1: playbook group G1 (loop body)" in _plan_file(workspace, 1)
    assert [event.get("wiki_batch") for event in _events(workspace, "iteration_result")] == [["G1"], None]
    [report] = _events(workspace, "wiki_batch_report")
    assert (report["source"], report["dropped"]) == ("session", ["P1"])
    assert WikiGuidanceStore(workspace).load().ledger["P1"].status == "SKIPPED"


def test_a_keep_the_process_died_before_crediting_is_credited_on_resume(tmp_path, monkeypatch):
    first, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"]], builds=[PLAYBOOK])
    monkeypatch.setattr(first, "run_one_iteration", _measured(["KEEP"]))

    def interrupt_before_checkpoint(*_args, **_kwargs):
        raise asyncio.CancelledError

    # The process dies after committing the KEEP and before crediting it.
    monkeypatch.setattr(first, "_finalize_keep_checkpoint", interrupt_before_checkpoint)
    with pytest.raises(asyncio.CancelledError):
        _run(first, service, _editing_agent([]))
    assert WikiGuidanceStore(workspace).load().ledger["P1"].status == "PENDING"

    resumed, _ = _guided_loop(tmp_path, monkeypatch, rounds=0, resume=True)
    _run(resumed, _GuidedService([]), _editing_agent([]))

    state = WikiGuidanceStore(workspace).load()
    assert [state.ledger[item].status for item in ("P1", "P2")] == ["APPLIED", "APPLIED"]
    assert (state.phase, state.reason_codes, state.outcomes) == (
        "FREE",
        ["ROUND_DONE"],
        [{"iteration": 1, "decision": "KEEP", "detail": ""}],
    )


def test_a_round_interrupted_before_its_measurement_is_planned_again_without_the_builder(tmp_path, monkeypatch):
    first, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"]], builds=[PLAYBOOK])

    class ProcessDied(BaseException):
        pass

    async def interrupted_session(kernel_path, history, session_sink):
        raise ProcessDied

    with pytest.raises(ProcessDied):
        _run(first, service, interrupted_session)
    assert service.builds == 1

    resumed, _ = _guided_loop(tmp_path, monkeypatch, rounds=1, resume=True)
    again = _GuidedService([["body again [playbook:P1]", "grid again [playbook:P2]"]])
    monkeypatch.setattr(resumed, "run_one_iteration", _measured(["KEEP"]))
    _run(resumed, again, _editing_agent(["first process"]))

    assert again.builds == 0
    assert again.contexts[0].wiki_guidance["batch"] == ["G1", "G2"]
    state = WikiGuidanceStore(workspace).load()
    assert [state.ledger[item].status for item in ("P1", "P2")] == ["APPLIED", "APPLIED"]
    assert (state.phase, state.reason_codes) == ("FREE", ["ROUND_DONE"])


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
    response = workspace / "forge_experiments" / "wiki_guidance" / "playbook_response.md"
    assert response.read_text() == "no JSON here"
    assert not _offered(loop)


def test_an_unusable_answer_from_the_backend_ends_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    service = _GuidedService([["plan"]], builds=[OrchestrationOutputError("playbook returned an incomplete response")])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT"]))

    _run(loop, service, _editing_agent([]))

    [changed] = _events(workspace, "wiki_guidance_phase")
    assert changed["reason_codes"] == ["PLAYBOOK_UNAVAILABLE"]


def test_a_builder_outage_is_retried_within_the_round(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    outage = OrchestrationInfrastructureError("playbook backend exceeded 600s timeout")
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"]], builds=[outage, PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 2
    assert service.contexts[0].wiki_guidance["batch"] == ["G1", "G2"]
    unavailable = [(event["attempt"], event["settled"]) for event in _events(workspace, "wiki_playbook_unavailable")]
    assert unavailable == [(1, False)]
    ledger = WikiGuidanceStore(workspace).load().ledger
    assert [ledger[item].status for item in ("P1", "P2")] == ["APPLIED", "APPLIED"]


def test_a_second_outage_gives_the_first_round_to_kernelforge(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=2)
    outage = OrchestrationInfrastructureError("playbook backend exceeded 600s timeout")
    service = _GuidedService([["one"], ["two"]], builds=[outage, outage, PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["REVERT", "REVERT"]))

    _run(loop, service, _editing_agent([]))

    assert service.builds == 2
    assert [context.wiki_guidance for context in service.contexts] == [None, None]
    unavailable = [(event["attempt"], event["settled"]) for event in _events(workspace, "wiki_playbook_unavailable")]
    assert unavailable == [(1, False), (2, True)]
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (1, ["PLAYBOOK_UNAVAILABLE"])
    assert not _offered(loop)


def test_a_batch_none_of_whose_parts_could_be_planned_gives_the_round_to_kernelforge(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1, lanes=3)
    # Every part failed, so planning handed back only the framework's generic fallback.
    fallback = OrchestrationRunResult(
        dispatch_plan=None, optimization_plans=("framework fallback",), optimization_plan_executable=False
    )
    service = _GuidedService([fallback, ["the round's plan"]], builds=[PLAYBOOK])
    histories: list[str] = []
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))

    _run(loop, service, _editing_agent(histories), agent_factory=_lane_recorder([]))

    # The same round is planned again as without the flag, at the configured width.
    assert service.lanes == [2, 3]
    assert service.contexts[0].wiki_guidance["batch"] == ["G1", "G2"]
    assert service.contexts[1].wiki_guidance is None
    [changed] = _events(workspace, "wiki_guidance_phase")
    assert (changed["iter"], changed["reason_codes"]) == (1, ["NO_BATCH_PLAN"])
    assert "## Wiki guidance" not in histories[0]
    [result] = _events(workspace, "iteration_result")
    assert (result["guidance_phase"], result.get("wiki_batch")) == ("FREE", None)
    assert not _offered(loop)


def test_a_resumed_campaign_continues_its_phase_and_a_fresh_one_starts_guided(tmp_path, monkeypatch):
    loop, workspace = _guided_loop(tmp_path, monkeypatch, rounds=1)
    service = _GuidedService([["body [playbook:P1]", "grid [playbook:P2]"]], builds=[PLAYBOOK])
    monkeypatch.setattr(loop, "run_one_iteration", _measured(["KEEP"]))
    _run(loop, service, _editing_agent([]))
    # A new process starts with every knowledge base offered.
    monkeypatch.setattr(wiki_tool_module, "_WITHDRAWN", set())

    resumed, _ = _guided_loop(tmp_path, monkeypatch, rounds=1, resume=True)
    assert resumed.wiki_rounds.state.phase == "FREE"
    assert not _offered(resumed)
    later = _GuidedService([["free plan"]])
    monkeypatch.setattr(resumed, "run_one_iteration", _measured(["KEEP"]))
    # A second candidate, distinct from the KEEP the first process committed.
    _run(resumed, later, _editing_agent(["first process"]))
    assert later.builds == 0
    assert "knowledge_base_round" in later.contexts[0].wiki_guidance

    monkeypatch.setattr(wiki_tool_module, "_WITHDRAWN", set())
    fresh = IterationLoop(
        replace(resumed.ic), ExperimentTracker(tmp_path / "experiments"), config=_config(tmp_path), resume=False
    )
    assert fresh.wiki_rounds.state.to_dict() == GuidanceState().to_dict()
    assert WikiGuidanceStore(workspace).load().playbook is None
    assert _offered(fresh)


# --- the builder role and the planning tasks ------------------------------------------------------------------------


class _McpBackend:
    capabilities = AgentCapabilities(mcp=True)

    def __init__(self, text):
        self.text = text
        self.specs = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        return AgentRunResult(text=self.text)


def test_the_builder_reads_the_knowledge_base_and_the_evidence(tmp_path):
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
    assert "wiki_guidance" not in json.loads(spec.user_prompt)["context"]
    # A direction only the evidence ranks may join the batch, marked as such and with the measurement behind it.
    assert '`"support": "profiling"`' in spec.system_prompt and "applies_when" in spec.system_prompt


def test_the_builder_refuses_without_a_knowledge_base():
    agent = OrchestrationAgent(backend=_McpBackend("{}"), timeout_sec=60, max_turns=8)
    with pytest.raises(ValueError, match="needs a knowledge base"):
        asyncio.run(agent.build_wiki_playbook(_context()))


def test_the_guidance_reaches_the_planners_only_when_there_is_some():
    plain = _context()
    guided = replace(plain, wiki_guidance={"phase": "GUIDED", "instructions": "x", "playbook": {}})

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

    plans = asyncio.run(agent.synthesize_lane_plan_results(context, SpecialistRound(dispatch, outcomes, {}), lanes=2))

    assert [plan.ground for plan in plans] == [
        "batch part 1 of 2: playbook group G2",
        "batch part 2 of 2: playbook group G1",
    ]
    # No partition turn: every call plans one part.
    assert {spec.system_prompt for spec in backend.specs} == {wiki_batch.BATCH_PART_SYSTEM_PROMPT}
    parts = sorted((json.loads(spec.user_prompt)["batch_part"] for spec in backend.specs), key=lambda p: p["position"])
    assert [(part["position"], part["of"], part["group"]) for part in parts] == [(1, 2, "G2"), (2, 2, "G1")]
    assert parts[1]["earlier_parts"] == ["G2 grid order"] and parts[0]["later_parts"] == ["G1 loop body"]
    analyses = json.loads(backend.specs[0].user_prompt)["specialist_analyses"]
    assert [entry["analysis"].split(" (")[0] for entry in analyses] == [
        "analysis: Playbook group G2",
        "analysis: Playbook group G1",
    ]
    assert agent.structured_output_diagnostics["batch"] == {"parts": ["G2", "G1"], "planned": 2}


def test_a_one_group_batch_is_planned_as_its_one_part():
    backend = _McpBackend("the part's plan")
    agent = OrchestrationAgent(backend=backend, timeout_sec=60, max_turns=8)
    context = _batch_context(["G3"])
    dispatch = orchestration_module.playbook_group_dispatch(context, lanes=1)
    [assignment] = dispatch.assignments
    outcome = SpecialistOutcome(
        assignment_id=assignment.assignment_id,
        role_id=assignment.role_id,
        duration_sec=0.1,
        content="analysis of G3",
    )

    plans = asyncio.run(agent.synthesize_lane_plan_results(context, SpecialistRound(dispatch, [outcome], {}), lanes=1))

    assert [plan.ground for plan in plans] == ["batch part 1 of 1: playbook group G3"]
    assert [spec.system_prompt for spec in backend.specs] == [wiki_batch.BATCH_PART_SYSTEM_PROMPT]


def test_a_guided_round_analyzes_its_batch_groups_and_no_others():
    service, _agent, pool = _service({"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE})

    asyncio.run(service.run(_batch_context(["G3"]), lanes=3))

    assert [assignment.reason.split(" (")[0] for assignment in pool.assignments] == ["Playbook group G3"]


# --- GUIDED rounds analyze each playbook group whole ----------------------------------------------------------------


class _PlanningAgent:
    """The planning agent's turns, recorded; a dispatch planner that should not be asked in GUIDED."""

    def __init__(self, *, plan_per_analysis=False):
        self.dispatched_from: list[list[str]] = []
        self.synthesized_from: list[list[str]] = []
        self.structured_output_diagnostics: dict = {}
        self.phase_durations_sec: dict = {}
        self.plan_per_analysis = plan_per_analysis

    async def plan_dispatch(self, context, definitions, *, usage=None):
        self.dispatched_from.append(sorted(definitions))
        cases = tuple(case.case_id for case in context.cases)
        return bind_dispatch(context, [DispatchIntent("compute", cases, "angle analysis")], ())

    async def synthesize_lane_plan_results(self, context, specialists, *, lanes, usage=None):
        outcomes = specialists.outcomes
        self.synthesized_from.append([outcome.content for outcome in outcomes])
        if self.plan_per_analysis:
            return [SynthesizedPlan(text=f"part for {outcome.content}") for outcome in outcomes]
        return [SynthesizedPlan(text="the plan")]


class _LaneCritic:
    """A plan critic that would drop every lane after the first, as it did to GUIDED batches."""

    def __init__(self):
        self.reviewed: list[int] = []

    async def review(self, *, context, drafts, specialists, usage=None):
        self.reviewed.append(len(drafts))
        drops = tuple(
            LaneDrop(lane_id=lane, reason="no-op when run concurrently from canonical")
            for lane in range(2, len(drafts) + 1)
        )
        return PlanCriticOutcome(verdict="ACCEPT", review="fine", lane_drops=drops, narrowing_status="answered")


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
    state.ledger["P2"].status = "APPLIED"
    state.batch = state.open_groups()
    state.outcomes = [{"iteration": 1, "decision": "KEEP", "detail": ""}]
    return replace(_context(), wiki_guidance=guidance.render_for_planning(state))


def _service(roles, *, critic=None, plan_per_analysis=False):
    agent = _PlanningAgent(plan_per_analysis=plan_per_analysis)
    pool = _RecordingPool(roles)
    service = OrchestrationService(
        agent=agent,
        specialist_pool=pool,
        definitions=default_specialist_definitions(),
        plan_critic=critic,
    )
    return service, agent, pool


def test_a_guided_batch_round_runs_no_plan_critic_and_keeps_every_part():
    critic = _LaneCritic()
    roles = {"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE}
    service, _agent, _pool = _service(roles, critic=critic, plan_per_analysis=True)

    guided = asyncio.run(service.run(_batch_context(["G1", "G2", "G3"]), lanes=3))

    assert critic.reviewed == []
    assert len(guided.optimization_plans) == 3
    assert guided.plan_critic is None and guided.optimization_plan_draft == ""
    assert guided.structured_output_diagnostics["plan_critic"] == {"status": "skipped_guided_batch"}
    assert "lane_narrowing" not in guided.structured_output_diagnostics
    assert guided.structured_output_diagnostics["lanes"]["published"] == 3
    # FREE rounds keep the critic, and its width ruling.
    free = asyncio.run(service.run(_guided_context(phase="FREE"), lanes=3))
    assert critic.reviewed == [1]
    assert free.plan_critic.verdict == "ACCEPT"


def test_a_guided_round_analyzes_each_open_group_whole_without_the_dispatch_planner():
    service, agent, pool = _service({"compute", "memory", "algorithm", orchestration_module.PLAYBOOK_GROUP_ROLE})

    result = asyncio.run(service.run(_guided_context(), lanes=3))

    assert agent.dispatched_from == []
    # G2's only item is applied, so the open groups are G1 and G3.
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


def test_the_flag_without_a_knowledge_base_is_refused(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, ["--wiki-guided"])

    assert result.exit_code != 0
    assert "needs a knowledge base" in result.output
    assert captured["loops"] == []
