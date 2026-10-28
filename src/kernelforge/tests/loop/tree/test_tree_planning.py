# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading what a planner wrote: the direction record of a BRANCH plan, and a plan's title."""

from __future__ import annotations

from types import SimpleNamespace

from kernelforge.loop.tree.model import Slot, SlotType
from kernelforge.loop.tree.planning import SlotPlan, TreePlanningMixin, read_direction_record

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


def _slots(count: int) -> list[Slot]:
    return [Slot(index, SlotType.BRANCH, "a" * 40, "r") for index in range(1, count + 1)]


def _planning_result(plans: tuple[str, ...], **diagnostics) -> SimpleNamespace:
    return SimpleNamespace(optimization_plans=plans, structured_output_diagnostics=diagnostics)


def test_a_lane_lost_mid_round_costs_only_its_own_slot(tmp_path):
    slots = _slots(3)
    result = _planning_result(
        ("plan for lane 1", "plan for lane 3"),
        lanes={"planned": 2, "published_lane_ids": [1, 3]},
        lane_losses={"2": "no analysis read the epilogue"},
    )

    mapped = _Host(tmp_path)._map_plans(1, slots, result)

    assert [(plan.slot.index, plan.text) for plan in mapped.plans] == [(1, "plan for lane 1"), (3, "plan for lane 3")]
    assert [(item.slot.index, item.reason) for item in mapped.unplanned] == [(2, "no analysis read the epilogue")]


def test_a_slot_whose_lane_the_critic_dropped_says_so(tmp_path):
    result = _planning_result(
        ("plan for lane 2",),
        lanes={"planned": 2, "published_lane_ids": [2]},
        lane_narrowing={"dropped_lanes": {"1": "lane 2 already covers it"}},
    )

    mapped = _Host(tmp_path)._map_plans(1, _slots(2), result)

    assert [(plan.slot.index, plan.text) for plan in mapped.plans] == [(2, "plan for lane 2")]
    assert [item.reason for item in mapped.unplanned] == ["the plan critic dropped its lane: lane 2 already covers it"]


def test_a_round_whose_synthesis_failed_runs_no_slot_on_the_framework_plan(tmp_path):
    result = _planning_result(
        ("framework plan",),
        lanes={"planned": 1, "published_lane_ids": []},
        synthesis={"status": "unavailable", "message": "OrchestrationOutputError: no lane plan"},
    )

    mapped = _Host(tmp_path)._map_plans(1, _slots(2), result)

    assert mapped.plans == ()
    assert {item.reason for item in mapped.unplanned} == {"OrchestrationOutputError: no lane plan"}
