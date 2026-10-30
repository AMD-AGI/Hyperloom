# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dividing a round into lanes: the ground each lane owns, joint lanes, the round's widest move, and the lanes a partition loses."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kernelforge.workflows.forge_loop.roles import orchestration as orchestration_module
from kernelforge.agent_backends import AgentProviderUnavailableError, AgentRunResult
from kernelforge.workflows.forge_loop.roles.contracts import DispatchPlan, SpecialistRound
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationAgent
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import (
    GROUND_A,
    GROUND_B,
    JOINT_GROUND,
    _PerCallBackend,
    _assignment,
    _compute_and_memory_round,
    _context,
    _coverage,
    _lane,
    _outcome,
    _partition,
    _partition_lanes,
)


@pytest.mark.asyncio
async def test_each_lane_is_given_its_ground_and_every_other_lane_s() -> None:
    """Overlapping lanes spend two Implementer sessions on one change."""
    backend = _PerCallBackend(
        [
            _partition(GROUND_A, GROUND_B),
            "# Lane A\nRetime the MFMA.",
            "# Lane B\nStage via LDS.",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane A\nRetime the MFMA.", "# Lane B\nStage via LDS."]
    owned = [json.loads(spec.user_prompt)["lane"] for spec in backend.specs[1:]]
    assert owned[0]["ground"] == GROUND_A
    assert owned[0]["ground_owned_by_other_lanes"] == [GROUND_B]
    assert owned[1]["ground"] == GROUND_B
    assert owned[1]["ground_owned_by_other_lanes"] == [GROUND_A]


@pytest.mark.asyncio
async def test_every_lane_receives_the_whole_round_s_evidence() -> None:
    """A lane's ground is a slice of the code, not a slice of the reading."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    for spec in backend.specs[1:]:
        analyses = json.loads(spec.user_prompt)["specialist_analyses"]
        assert [item["role_id"] for item in analyses] == ["compute", "memory"]
        assert "scale stream" in json.dumps(analyses)


@pytest.mark.asyncio
async def test_the_partition_reads_every_analysis_before_it_divides() -> None:
    """Where the boundaries fall cannot be answered from a slice."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    partition = json.loads(backend.specs[0].user_prompt)
    assert [item["role_id"] for item in partition["specialist_analyses"]] == ["compute", "memory"]
    assert agent.structured_output_diagnostics["partition"]["status"] == "planned"


@pytest.mark.asyncio
async def test_a_replace_verdict_spends_one_lane_on_the_challenge() -> None:
    """A critic rules on a plan that exists, so REPLACE lands on the next round."""
    backend = _PerCallBackend([_partition("validate the CK GEMM path", GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()
    context = replace(
        _context(),
        last_critic_verdict="REPLACE",
        last_critic_review="A CK GEMM already exists for this shape.",
    )

    await agent.synthesize_lane_plan_results(context, specialists, lanes=2)

    partition = backend.specs[0]
    assert "Give exactly one lane to that challenge" in partition.system_prompt
    assert "A CK GEMM already exists" in partition.user_prompt
    assert agent.structured_output_diagnostics["partition"]["challenger_requested"]


@pytest.mark.asyncio
async def test_a_round_nobody_challenged_is_divided_as_usual() -> None:
    """Guards the challenger block from reaching every ordinary round."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert "Give exactly one lane" not in backend.specs[0].system_prompt
    assert agent.structured_output_diagnostics["partition"]["challenger_requested"] is False


@pytest.mark.asyncio
async def test_an_accepted_round_raises_no_challenger() -> None:
    """Only REPLACE says the route is wrong; REVISE corrected it in place."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()
    context = replace(
        _context(),
        last_critic_verdict="REVISE",
        last_critic_review="Add a stop condition.",
    )

    await agent.synthesize_lane_plan_results(context, specialists, lanes=2)

    assert "Give exactly one lane" not in backend.specs[0].system_prompt


@pytest.mark.asyncio
async def test_the_partition_is_told_a_cross_cutting_move_is_one_lane() -> None:
    """The width follows the directions, not the other way around."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1800, max_turns=500)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=3)

    partition = backend.specs[0].system_prompt
    assert "A change is one ground however many places it lands" in partition
    assert "not one lane per\nsite" in partition
    assert "never a\nnumber of pieces to cut one direction into" in partition


_CROSS_CUTTING = "fuse the post kernel into the GEMM, deleting one dispatch"


@pytest.mark.asyncio
async def test_the_partition_keeps_a_launch_config_with_the_body_it_serves() -> None:
    """A body lane measured at the old body's config closes an untested axis."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    partition = backend.specs[0].system_prompt
    assert "The launch configuration is not ground of its own" in partition
    assert "owns the configuration that serves it" in partition
    assert "2.80 ms" in partition and "1.28 ms" in partition


@pytest.mark.asyncio
async def test_a_joint_lane_reaches_its_implementer_time_boxed() -> None:
    """Wider ground is bought with a fallback, not granted for free."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [
                    _lane(
                        JOINT_GROUND,
                        joint=True,
                        fallback="re-tune num_warps at the current tile alone",
                    ),
                    _lane(GROUND_B),
                ]
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    joint_lane = json.loads(backend.specs[1].user_prompt)["lane"]
    assert joint_lane["joint"] is True
    assert joint_lane["fallback"] == "re-tune num_warps at the current tile alone"
    assert json.loads(backend.specs[2].user_prompt)["lane"]["joint"] is False
    assert "abandoned" in backend.specs[1].system_prompt
    assert "sequence the lane's `fallback`" in backend.specs[1].system_prompt
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == [1]
    assert diagnostics["grounds"][0]["fallback"]


@pytest.mark.asyncio
async def test_a_joint_lane_with_no_fallback_is_said_out_loud(caplog) -> None:
    """The lane still runs; what it is risking stops being invisible."""
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(JOINT_GROUND, joint=True), _lane(GROUND_B)]),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane A", "# Lane B"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == [1]
    assert any("named no fallback" in note for note in diagnostics["notes"])
    assert "joint lane(s) 1 carry no fallback" in caplog.text


@pytest.mark.asyncio
async def test_the_round_records_which_lane_owns_its_widest_move() -> None:
    """The move that fits no one region is what four kernels lost."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(GROUND_A), _lane(GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": 2},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert "Name the largest move" in backend.specs[0].system_prompt
    assert "largest cross-cutting move" in json.loads(backend.specs[0].user_prompt)["task"]
    move = agent.structured_output_diagnostics["partition"]["cross_cutting_move"]
    assert move == {
        "status": "assigned",
        "move": _CROSS_CUTTING,
        "lane_id": 2,
        "lane_ground": GROUND_B,
    }


@pytest.mark.asyncio
async def test_an_unowned_widest_move_is_in_the_round_s_output(caplog) -> None:
    """A move no lane took is a finding, not an absence."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(GROUND_A), _lane(GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": 0},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane A", "# Lane B"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"] == {
        "status": "unassigned",
        "move": _CROSS_CUTTING,
        "lane_id": 0,
        "unassigned_reason": "",
    }
    assert any("no reason was given" in note for note in diagnostics["notes"])
    assert _CROSS_CUTTING in caplog.text


@pytest.mark.asyncio
async def test_a_widest_move_naming_a_lane_the_round_lacks_is_unowned(
    caplog,
) -> None:
    """Owned by a lane that does not exist is unowned, and is reported as such."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(GROUND_A), _lane(GROUND_B), _lane("scale.py: the epilogue")],
                {"move": _CROSS_CUTTING, "lane_id": 3, "unassigned_reason": ""},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["planned"] == 2
    assert diagnostics["cross_cutting_move"]["status"] == "unassigned"
    assert any("which this partition of 2 lane(s) does not have" in note for note in diagnostics["notes"])
    assert "gave no lane its largest cross-cutting move" in caplog.text


@pytest.mark.asyncio
async def test_a_partition_that_named_no_widest_move_reports_that(caplog) -> None:
    """Named none and left one unowned are different answers."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"] == {"status": "missing"}
    assert any("named no largest cross-cutting move" in note for note in diagnostics["notes"])
    assert "the partition named none" in caplog.text


@pytest.mark.asyncio
async def test_a_move_named_as_a_plain_string_is_still_a_named_move(
    caplog,
) -> None:
    """The schema shows an object; a model answering it with prose is ordinary."""
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(GROUND_A), _lane(GROUND_B)], _CROSS_CUTTING),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"] == {
        "status": "unassigned",
        "move": _CROSS_CUTTING,
        "lane_id": 0,
        "unassigned_reason": "",
    }
    assert any("came back as a string" in note for note in diagnostics["notes"])
    assert _CROSS_CUTTING in caplog.text


@pytest.mark.asyncio
async def test_a_move_field_of_the_wrong_shape_is_not_reported_as_no_move(
    caplog,
) -> None:
    """A shape the parser cannot read is not evidence the partition looked."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(GROUND_A), _lane(GROUND_B)],
                [{"move": _CROSS_CUTTING}],
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with caplog.at_level("WARNING", logger=orchestration_module.log.name):
        await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    move = agent.structured_output_diagnostics["partition"]["cross_cutting_move"]
    assert move["status"] == "unreadable"
    assert "came back as a list" in move["field"]
    assert "came back as a list" in caplog.text
    assert "the partition named none" not in caplog.text


@pytest.mark.asyncio
async def test_an_owned_move_records_the_ground_that_owns_it() -> None:
    """The lane_id is the partitioner grading its own answer."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(GROUND_A), _lane(GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": 1},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    move = agent.structured_output_diagnostics["partition"]["cross_cutting_move"]
    assert move["lane_id"] == 1
    assert move["lane_ground"] == GROUND_A


@pytest.mark.asyncio
async def test_a_move_naming_a_lane_that_is_not_a_number_is_unowned() -> None:
    """A lane_id that is not a lane number is not a lane, and says so."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [_lane(GROUND_A), _lane(GROUND_B)],
                {"move": _CROSS_CUTTING, "lane_id": "lane 2"},
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["cross_cutting_move"]["status"] == "unassigned"
    assert any("not a lane number" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_an_unreadable_joint_answer_is_recorded_not_just_narrowed() -> None:
    """Falling back to narrow ground is right; doing it without a word is not."""
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(JOINT_GROUND, joint="partial"), _lane(GROUND_B)]),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == []
    assert any("'partial', which is not a boolean" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_a_lane_that_declined_joint_ground_does_not_get_it() -> None:
    """A quoted \"false\" is a no, and the flag it sets is the one that widens."""
    backend = _PerCallBackend(
        [
            _partition_lanes(
                [
                    _lane(GROUND_A, joint="false"),
                    _lane(JOINT_GROUND, joint="true", fallback="re-tune only"),
                ]
            ),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert agent.structured_output_diagnostics["partition"]["joint"] == [2]


def test_an_answer_that_is_not_a_boolean_is_read_as_the_note_promises() -> None:
    """The flag stored and the note written have to describe one reading."""
    assert orchestration_module._as_bool(True) == (True, True)
    assert orchestration_module._as_bool("true") == (True, True)
    assert orchestration_module._as_bool(None) == (False, True)
    assert orchestration_module._as_bool("false") == (False, True)
    # Not booleans, whichever way they would have gone through ``bool``.
    assert orchestration_module._as_bool(1) == (False, False)
    assert orchestration_module._as_bool(0) == (False, False)
    assert orchestration_module._as_bool(2.5) == (False, False)
    assert orchestration_module._as_bool({"a": 1}) == (False, False)
    assert orchestration_module._as_bool([]) == (False, False)


@pytest.mark.asyncio
async def test_a_lane_that_answered_joint_with_a_number_keeps_narrow_ground() -> None:
    """The lane list and the note are read by the same operator."""
    backend = _PerCallBackend(
        [
            _partition_lanes([_lane(JOINT_GROUND, joint=1), _lane(GROUND_B)]),
            "# Lane A",
            "# Lane B",
        ]
    )
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["joint"] == []
    assert diagnostics["grounds"][0]["joint"] is False
    assert json.loads(backend.specs[1].user_prompt)["lane"]["joint"] is False
    assert any("1, which is not a boolean" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_a_launch_site_two_bodies_share_is_owned_by_one_lane() -> None:
    """The exception that widens a lane cannot be handed to every lane."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    partition = backend.specs[0].system_prompt
    assert "One launch site belongs to exactly one lane" in partition
    assert "name that launch in exactly one\nlane's ground" in partition
    # The boundary is derived, never written twice: a lane's ground says only what it owns, so the prompt must not ask
    # for a "stay off" clause that would reach the owning lane as ground it does not own.
    assert "every other lane sees it as ground it does not own" in partition
    assert "keep the other lane off it" not in partition


@pytest.mark.asyncio
async def test_a_partition_that_could_not_be_bought_claims_no_move() -> None:
    """A collapsed round found nothing, so it must not report having looked."""
    backend = _PerCallBackend(["not json at all", "# One plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["collapsed"] is True
    assert diagnostics["cross_cutting_move"] == {"status": "unavailable"}
    assert diagnostics["joint"] == []


@pytest.mark.asyncio
async def test_the_partition_divides_what_it_was_given_without_reading_more() -> None:
    """The analyses already name the files and functions they are about."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "# Lane A", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1800, max_turns=500)
    specialists = await _compute_and_memory_round()

    await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    partition, lane = backend.specs[0], backend.specs[1]
    assert partition.tool_policy.read is False
    assert partition.tool_policy.search is False
    assert partition.tool_policy.max_turns == orchestration_module.ROUND_PARTITION_MAX_TURNS
    assert partition.timeout_sec == orchestration_module.ROUND_PARTITION_TIMEOUT_SEC
    # The partition no longer names an effort of its own; the runtime settles it.
    assert partition.reasoning_effort == ""
    assert lane.reasoning_effort == ""
    # A lane plan does need the source; only the division does not.
    assert lane.tool_policy.read is True
    assert lane.timeout_sec == 1800


@pytest.mark.asyncio
async def test_a_partition_that_cannot_be_bought_still_leaves_a_round() -> None:
    """A round that cannot be divided by code runs as one lane, not a wide one."""
    backend = _PerCallBackend(["not json at all", "# One plan", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# One plan"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["status"] == "fallback"
    assert diagnostics["collapsed"] is True
    assert diagnostics["planned"] == 1
    assert len(diagnostics["grounds"]) == 1
    assert "compute" in diagnostics["grounds"][0]["ground"]
    assert "memory" in diagnostics["grounds"][0]["ground"]


@pytest.mark.asyncio
async def test_a_challenge_survives_a_partition_that_could_not_be_bought() -> None:
    """A REPLACE is not asked for again, and the round it judged is this one."""
    backend = _PerCallBackend(["not json at all", "# Challenger plan", "# Lane B"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()
    context = replace(
        _context(),
        last_critic_verdict="REPLACE",
        last_critic_review="A CK GEMM already exists for this shape.",
    )

    results = await agent.synthesize_lane_plan_results(context, specialists, lanes=2)

    assert [result.text for result in results] == ["# Challenger plan"]
    diagnostics = agent.structured_output_diagnostics["partition"]
    assert diagnostics["status"] == "fallback"
    assert diagnostics["collapsed"] is True
    assert diagnostics["planned"] == 1
    assert diagnostics["challenger_requested"] is True
    grounds = diagnostics["grounds"]
    assert [ground["lane_id"] for ground in grounds] == [1]
    assert "last_plan_critic" in grounds[0]["ground"]
    # The single lane's own payload carries the challenger ground, not the ordinary synthesis prompt.
    lane_spec = backend.specs[1]
    assert "last_plan_critic" in json.loads(lane_spec.user_prompt)["lane"]["ground"]


@pytest.mark.asyncio
async def test_one_real_direction_is_planned_as_a_single_lane_round() -> None:
    """A slot filled to reach the requested width still costs a whole session."""
    backend = _PerCallBackend([_partition(GROUND_A), "# One plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# One plan"]
    assert "lane" not in json.loads(backend.specs[1].user_prompt)


class _OneLaneFailsBackend:
    """A backend that partitions, then loses the lane owning one named ground."""

    def __init__(self, *, failing_ground: str, answer: str) -> None:
        self.failing_ground = failing_ground
        self.answer = answer
        self.specs: list = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        lane = json.loads(spec.user_prompt).get("lane")
        if lane is None:
            return AgentRunResult(
                text=_partition(GROUND_A, GROUND_B),
                end_reason="agent_stopped",
                stderr_tail="",
            )
        if self.failing_ground in lane["ground"]:
            raise AgentProviderUnavailableError("the lane's provider went away")
        return AgentRunResult(
            text=self.answer,
            end_reason="agent_stopped",
            stderr_tail="",
        )


@pytest.mark.asyncio
async def test_one_lost_lane_does_not_cost_the_round_its_healthy_plans() -> None:
    """Each lane is its own call, so one failure is one lane, not the round."""
    backend = _OneLaneFailsBackend(failing_ground=GROUND_A, answer="# Lane B\nStage via LDS.")
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane B\nStage via LDS."]


@pytest.mark.asyncio
async def test_a_round_that_loses_every_lane_is_still_a_planning_outage() -> None:
    """Tolerating one loss must not turn a total outage into a silent success."""
    backend = _OneLaneFailsBackend(failing_ground=GROUND_B, answer="")
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with pytest.raises(Exception) as outage:
        await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert "lane plan" in str(outage.value)


@pytest.mark.asyncio
async def test_a_lane_that_answered_with_nothing_is_reported(caplog) -> None:
    """A round narrowed by an empty answer must not look like a narrower round."""
    backend = _PerCallBackend([_partition(GROUND_A, GROUND_B), "   ", "# Lane B\nStage via LDS."])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    specialists = await _compute_and_memory_round()

    with caplog.at_level("WARNING"):
        results = await agent.synthesize_lane_plan_results(_context(), specialists, lanes=2)

    assert [result.text for result in results] == ["# Lane B\nStage via LDS."]
    assert "empty plan" in caplog.text


@pytest.mark.asyncio
async def test_one_usable_analysis_is_one_lane() -> None:
    """A single lane must behave exactly like the fused single-plan path."""
    backend = _PerCallBackend(["# The only plan"])
    agent = OrchestrationAgent(backend=backend, timeout_sec=1, max_turns=2)
    outcome = await _outcome("memory", "The scale stream is re-read.")
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))

    results = await agent.synthesize_lane_plan_results(
        _context(), SpecialistRound(dispatch, (outcome,), _coverage(["memory"])), lanes=2
    )

    assert [result.text for result in results] == ["# The only plan"]
    assert "not a catalog" in backend.specs[0].system_prompt
