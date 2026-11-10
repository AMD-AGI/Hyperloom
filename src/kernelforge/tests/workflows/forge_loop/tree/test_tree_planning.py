# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading what a planner wrote into slot plans: a BRANCH plan's direction record, a plan's title, its lane's focus."""

from __future__ import annotations

from types import SimpleNamespace

from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.roles.contracts import PublishedLane
from kernelforge.workflows.forge_loop.tree.admission import slot_prompt
from kernelforge.workflows.forge_loop.tree.model import NodeSource, Slot, SlotType, TreeNode
from kernelforge.workflows.forge_loop.tree.planning import (
    PlanningUnit,
    SlotPlan,
    TreePlanningMixin,
    planning_units,
    read_direction_record,
)

# The shape a real planner wrote: an opening sentence, then a record whose keys are bold.
PLAN = """I'll read the kernel source and the analysis report before planning.
## Direction

- **name:** Close the split-K tail at M = 64/128
- **hypothesis:** m_64 and m_128 are the only buckets still emitting a second graph node.
- **expected ceiling:** +0.8 to +1.9 points of mean speedup.
- **fails if:** membership and every re-tile measure flat at both buckets.
- **hint:** **Adopted in part, refuted in part.**
  - *Adopted:* the membership A/B.

## Implementation sequence
### Step 0 — in-session reference
"""


def test_a_bold_direction_record_is_read_and_its_hint_answer_kept():
    proposal, hint_answer = read_direction_record(PLAN)

    assert proposal.name == "Close the split-K tail at M = 64/128"
    assert proposal.hypothesis.startswith("m_64 and m_128")
    assert proposal.expected_ceiling == "+0.8 to +1.9 points of mean speedup."
    assert proposal.failure_criterion.startswith("membership and every re-tile")
    assert hint_answer == "Adopted in part, refuted in part."


def test_a_plan_without_a_record_opens_no_named_direction():
    proposal, hint_answer = read_direction_record("## Plan\n- name: not a record\n")

    assert proposal is None and hint_answer == ""


def test_a_plan_is_titled_by_its_direction_or_its_first_heading_never_its_opening_sentence():
    slot = Slot(1, SlotType.BRANCH, "a" * 40, "r")
    proposal, _ = read_direction_record(PLAN)

    assert SlotPlan(slot, PLAN, "p", proposal).title == "Close the split-K tail at M = 64/128"
    assert SlotPlan(slot, PLAN, "p").title == "Implementation sequence"
    assert SlotPlan(slot, "just one line", "p").title == "just one line"


class _Host(TreePlanningMixin):
    def __init__(self, workspace) -> None:
        self.ic = SimpleNamespace(workspace_dir=str(workspace))
        self._experiments = ExperimentsLayout.of(workspace)


def _slots(count: int) -> list[Slot]:
    return [Slot(index, SlotType.BRANCH, "a" * 40, "r") for index in range(1, count + 1)]


def _unit(tmp_path, slots: list[Slot]) -> PlanningUnit:
    (unit,) = planning_units(slots, tmp_path / "iter_001")
    return unit


def _planning_result(plans: tuple[str, ...], lanes: tuple[PublishedLane, ...], **diagnostics) -> SimpleNamespace:
    return SimpleNamespace(optimization_plans=plans, published_lanes=lanes, structured_output_diagnostics=diagnostics)


def test_a_lane_lost_mid_round_costs_only_its_own_slot(tmp_path):
    slots = _slots(3)
    result = _planning_result(
        ("plan for lane 1", "plan for lane 3"),
        (PublishedLane(1), PublishedLane(3)),
        lane_losses={"2": "no analysis read the epilogue"},
    )

    mapped = _Host(tmp_path)._map_plans(_unit(tmp_path, slots), slots, result)

    assert [(plan.slot.index, plan.text) for plan in mapped.plans] == [(1, "plan for lane 1"), (3, "plan for lane 3")]
    assert [(item.slot.index, item.reason) for item in mapped.unplanned] == [(2, "no analysis read the epilogue")]


def test_a_slot_whose_lane_the_critic_dropped_says_so(tmp_path):
    result = _planning_result(
        ("plan for lane 2",),
        (PublishedLane(2),),
        lane_narrowing={"dropped_lanes": {"1": "lane 2 already covers it"}},
    )

    mapped = _Host(tmp_path)._map_plans(_unit(tmp_path, _slots(2)), _slots(2), result)

    assert [(plan.slot.index, plan.text) for plan in mapped.plans] == [(2, "plan for lane 2")]
    assert [item.reason for item in mapped.unplanned] == ["the plan critic dropped its lane: lane 2 already covers it"]


def test_a_round_whose_synthesis_failed_runs_no_slot_on_the_framework_plan(tmp_path):
    result = _planning_result(
        ("framework plan",),
        (),
        synthesis={"status": "unavailable", "message": "OrchestrationOutputError: no lane plan"},
    )

    mapped = _Host(tmp_path)._map_plans(_unit(tmp_path, _slots(2)), _slots(2), result)

    assert mapped.plans == ()
    assert {item.reason for item in mapped.unplanned} == {"OrchestrationOutputError: no lane plan"}


def test_a_slot_is_planned_with_the_approach_and_cases_its_lane_was_given(tmp_path):
    result = _planning_result(
        ("plan for lane 1", "plan for lane 2"),
        (PublishedLane(1, approach="retime the MFMA issue", focus_cases=("m_64",)), PublishedLane(2)),
    )

    mapped = _Host(tmp_path)._map_plans(_unit(tmp_path, _slots(2)), _slots(2), result)

    assert [(plan.approach, plan.focus_cases) for plan in mapped.plans] == [
        ("retime the MFMA issue", ("m_64",)),
        ("", ()),
    ]


def test_a_start_whose_slots_run_on_several_profiles_is_planned_once_per_profile(tmp_path):
    slots = [
        Slot(1, SlotType.BRANCH, "a" * 40, "r", agent_profile_id="claude"),
        Slot(2, SlotType.BRANCH, "a" * 40, "r", agent_profile_id="codex"),
        Slot(3, SlotType.BRANCH, "b" * 40, "r", agent_profile_id="claude"),
        Slot(4, SlotType.BRANCH, "a" * 40, "r", agent_profile_id="claude"),
    ]

    units = planning_units(slots, tmp_path)

    assert [
        (unit.start[0], unit.profile_id, [slot.index for slot in unit.slots], unit.records.name) for unit in units
    ] == [
        ("a", "claude", [1, 4], "start_aaaaaaaaaaaa_claude"),
        ("a", "codex", [2], "start_aaaaaaaaaaaa_codex"),
        # A start planned on one profile keeps the directory it has always had.
        ("b", "claude", [3], "start_bbbbbbbbbbbb"),
    ]


def _start() -> TreeNode:
    return TreeNode(
        commit="a" * 40,
        parent="",
        direction="",
        slot_type=None,
        iteration=0,
        slot=0,
        samples=(1.0,),
        wall_ms=1.0,
        case_times={"m_64": 1.0, "m_128": 2.0},
        case_spreads={"m_64": 0.01, "m_128": 0.01},
        snr_db=None,
        plan="",
        source=NodeSource.BASELINE,
    )


def test_a_slot_is_told_its_lane_s_approach_and_how_to_time_the_cases_it_aims_at():
    slot = _slots(1)[0]
    plan = SlotPlan(slot, "## Plan\nretime", "p", approach="retime the MFMA issue", focus_cases=("m_64", "m_128"))

    prompt = slot_prompt(plan, _start())

    assert "retime the MFMA issue" in prompt
    assert "aimed at these cases: m_64, m_128" in prompt and "--bench-case <case_id>" in prompt
    assert prompt.endswith("## Plan\nretime")
    # A lane aimed at the plan alone is told nothing more than its plan.
    plain = slot_prompt(SlotPlan(slot, "## Plan\nretime", "p"), _start())
    assert "approach" not in plain and "--bench-case" not in plain
