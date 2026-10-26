# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading what a planner wrote: the direction record of a BRANCH plan, and a plan's title."""

from __future__ import annotations

from kernelforge.loop.tree.model import Slot, SlotType
from kernelforge.loop.tree.planning import SlotPlan, read_direction_record

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
