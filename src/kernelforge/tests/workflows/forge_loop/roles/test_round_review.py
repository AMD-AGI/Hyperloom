# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The critic's review of a round: how it narrows the round's lanes, what it may not drop, and what the round records."""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import pytest

from kernelforge.agent_backends import AgentRunResult
from kernelforge.workflows.forge_loop.roles.contracts import LaneDrop, PlanCriticOutcome, SynthesizedPlan
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationAgent, OrchestrationService
from kernelforge.workflows.forge_loop.roles.specialists import SpecialistAgent, SpecialistPool
from kernelforge.workflows.forge_loop.roles.plan_critic import PlanCriticAgent
from kernelforge.tests.workflows.forge_loop.roles._orchestration_fakes import (
    GROUND_A,
    GROUND_B,
    JOINT_GROUND,
    _QueuedBackend,
    _StaticBackend,
    _context,
    _definitions,
    _dispatch_payload,
    _lane,
    _partition,
    _partition_lanes,
    _single_lane_service,
)


@pytest.mark.asyncio
async def test_plan_critic_accepts_draft_without_revision() -> None:
    plan_markdown = "# Optimization plan\nImplement vector loads."
    orchestration_backend = _QueuedBackend([_dispatch_payload(), plan_markdown])
    specialist_backend = _StaticBackend("# Analysis\nVector loads are feasible.")
    critic_backend = _StaticBackend("VERDICT: ACCEPT\n\nThe plan is worth executing.")
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=specialist_backend,
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
        plan_critic=PlanCriticAgent(
            backend=critic_backend,
            timeout_sec=1,
        ),
    )

    result = await service.run(_context())

    assert result.optimization_plans[0] == plan_markdown
    assert result.optimization_plan_draft == plan_markdown
    assert result.plan_revised is False
    assert result.plan_critic is not None
    assert result.plan_critic.verdict == "ACCEPT"
    assert len(orchestration_backend.specs) == 2
    assert specialist_backend.calls == 1
    assert critic_backend.calls == 1


def _width_block(*drops: dict) -> str:
    """The trailing JSON block the round contract asks every review to end with."""
    return json.dumps({"lane_narrowing": list(drops)})


@pytest.mark.asyncio
async def test_plan_critic_reviews_a_multi_lane_round_once() -> None:
    definitions = _definitions()
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(GROUND_A, GROUND_B),
            "# Lane plan\nMemory route.",
            "# Lane plan\nCompute route.",
            "# Lane plan\nMemory route, revised.",
            "# Lane plan\nCompute route, revised.",
        ]
    )
    critic_backend = _StaticBackend("VERDICT: REVISE\n\nLane 2 repeats lane 1's change.\n" + _width_block())
    service = OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=2,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=definitions["memory"],
                    backend=_StaticBackend("Memory analysis"),
                    timeout_sec=1,
                    max_turns=2,
                ),
                "compute": SpecialistAgent(
                    definition=definitions["compute"],
                    backend=_StaticBackend("Compute analysis"),
                    timeout_sec=1,
                    max_turns=2,
                ),
            },
            max_parallel=2,
        ),
        definitions=definitions,
        plan_critic=PlanCriticAgent(
            backend=critic_backend,
            timeout_sec=1,
        ),
    )

    result = await service.run(_context(), lanes=2)

    assert len(result.optimization_plans) == 2
    # One review for the round, not one per lane: the width is what the round most needs reviewed, and it is a
    # question no single lane can be asked.
    assert critic_backend.calls == 1
    assert result.plan_critic is not None
    assert result.plan_critic.verdict == "REVISE"
    payload = json.loads(critic_backend.specs[0].user_prompt)
    assert [lane["lane_id"] for lane in payload["draft_lane_plans"]] == [1, 2]
    assert payload["draft_lane_plans"][0]["ground"] == GROUND_A
    assert payload["draft_lane_plans"][1]["ground"] == GROUND_B
    assert "draft_plan" not in payload
    assert "not worth an Implementer session of its own" in (critic_backend.specs[0].system_prompt)
    # The verdict was about the round, so it reached every lane in it.
    assert result.optimization_plans == (
        "# Lane plan\nMemory route, revised.",
        "# Lane plan\nCompute route, revised.",
    )
    assert result.plan_revised is True
    assert result.structured_output_diagnostics["plan_revision"]["status"] == ("revised")


def _two_lane_critic_service(orchestration_backend, critic_backend):
    """A two-role service whose round is reviewed by one critic."""
    definitions = _definitions()
    return OrchestrationService(
        agent=OrchestrationAgent(
            backend=orchestration_backend,
            timeout_sec=1,
            max_turns=2,
            min_assignments=2,
        ),
        specialist_pool=SpecialistPool(
            {
                role: SpecialistAgent(
                    definition=definitions[role],
                    backend=_StaticBackend(f"{role} analysis"),
                    timeout_sec=1,
                    max_turns=2,
                )
                for role in ("memory", "compute")
            },
            max_parallel=2,
        ),
        definitions=definitions,
        plan_critic=PlanCriticAgent(backend=critic_backend, timeout_sec=1),
    )


@pytest.mark.asyncio
async def test_the_round_reports_how_it_was_divided() -> None:
    """A round is audited after the fact or not at all."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(GROUND_A, GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
        ]
    )
    service = _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend("VERDICT: ACCEPT\n\nBoth lanes are worth a session.\n" + _width_block()),
    )

    result = await service.run(_context(), lanes=2)

    partition = result.structured_output_diagnostics["partition"]
    assert partition["status"] == "planned"
    assert partition["challenger_requested"] is False
    assert [ground["ground"] for ground in partition["grounds"]] == [
        GROUND_A,
        GROUND_B,
    ]
    # The earlier snapshot's own entries survive the merge.
    assert "coverage" in result.structured_output_diagnostics
    assert result.structured_output_diagnostics["lanes"]["planned"] == 2


def _two_lane_round(critic_review: str, *, revisions: list[str] | None = None):
    """A two-lane round whose review says ``critic_review`` about its width."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(GROUND_A, GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
            *(revisions or []),
        ]
    )
    return orchestration_backend, _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend(critic_review),
    )


@pytest.mark.asyncio
async def test_a_lane_the_review_will_not_pay_for_is_not_published() -> None:
    """The round's width is the one thing the verdict could not say."""
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {
                "lane_id": 2,
                "reason": "it is lane 1's epilogue rewrite in different words",
            }
        )
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft",)
    assert result.optimization_plan_draft == "# Lane 1 draft"
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["block"] == "answered"
    assert narrowing["planned"] == 2
    assert narrowing["kept"] == 1
    assert narrowing["dropped"] == [
        {
            "lane_id": 2,
            "reason": "it is lane 1's epilogue rewrite in different words",
        }
    ]
    assert result.structured_output_diagnostics["lanes"] == {
        "requested": 2,
        "planned": 2,
        "published": 1,
    }


@pytest.mark.asyncio
async def test_a_dropped_lane_is_not_revised_first() -> None:
    """A revision turn spent on a lane that will not run is the whole cost."""
    backend, service = _two_lane_round(
        "VERDICT: REVISE\n\n" + _width_block({"lane_id": 1, "reason": "no profile supports its ground"}),
        revisions=["# Lane 2 revised"],
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 2 revised",)
    # Dispatch, partition, two lane plans, and one revision -- not two.
    assert len(backend.specs) == 5
    revision = result.structured_output_diagnostics["plan_revision"]
    assert revision["status"] == "revised"
    assert revision["lanes"] == 1


@pytest.mark.asyncio
async def test_a_round_narrowed_to_nothing_keeps_every_lane() -> None:
    """A round that publishes nothing spent its planning window for no score."""
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {"lane_id": 1, "reason": "unsupported"},
            {"lane_id": 2, "reason": "also unsupported"},
        )
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "refused_empty_round"
    assert narrowing["kept"] == 2
    assert narrowing["dropped"] == []
    assert any("nothing to measure" in note for note in narrowing["notes"])
    assert result.structured_output_diagnostics["lanes"]["published"] == 2


@pytest.mark.asyncio
async def test_a_round_collapsed_to_one_lane_cannot_be_narrowed() -> None:
    """Two decisions about width meet here, and the floor wins."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            "not json at all",
            "# One plan",
        ]
    )
    service = _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend("VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 1, "reason": "not worth a session"})),
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# One plan",)
    assert result.structured_output_diagnostics["partition"]["collapsed"] is True
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "refused_single_lane"
    assert narrowing["kept"] == 1
    assert any("at least one lane" in note for note in narrowing["notes"])


@pytest.mark.asyncio
async def test_a_challenged_round_refuses_narrowing_it_cannot_aim() -> None:
    """One of these lanes is the challenger, and nothing records which."""
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 2, "reason": "it duplicates lane 1"})
    )
    context = replace(
        _context(),
        last_critic_verdict="REPLACE",
        last_critic_review="A CK GEMM already exists for this shape.",
    )

    result = await service.run(context, lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    assert result.structured_output_diagnostics["partition"]["challenger_requested"] is True
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "refused_challenger"
    assert narrowing["kept"] == 2
    assert any("challenger lane" in note for note in narrowing["notes"])


def _joint_round(joint_lanes: list[int], count: int, critic_review: str):
    """A round of ``count`` lanes, the ones named in ``joint_lanes`` widened."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition_lanes(
                [
                    _lane(
                        JOINT_GROUND,
                        joint=True,
                        fallback="re-tune num_warps at the current tile alone",
                    )
                    if lane_id in joint_lanes
                    else _lane(f"{GROUND_B} ({lane_id})")
                    for lane_id in range(1, count + 1)
                ]
            ),
            *[f"# Lane {lane_id} draft" for lane_id in range(1, count + 1)],
        ]
    )
    return orchestration_backend, _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend(critic_review),
    )


def _joint_lane_round(critic_review: str):
    """A two-lane round whose first lane the partition widened to joint ground."""
    return _joint_round([1], 2, critic_review)


@pytest.mark.asyncio
async def test_the_review_is_told_which_lane_the_partition_widened() -> None:
    """A ruling on width made blind to the width is not the ruling asked for."""
    _backend, service = _joint_lane_round("VERDICT: ACCEPT\n\n" + _width_block())

    await service.run(_context(), lanes=2)

    critic_spec = service._plan_critic.backend.specs[0]
    lanes = json.loads(critic_spec.user_prompt)["draft_lane_plans"]
    assert lanes[0]["joint"] is True
    assert lanes[0]["fallback"] == ("re-tune num_warps at the current tile alone")
    assert lanes[1]["joint"] is False
    assert lanes[1]["fallback"] == ""
    assert "`joint` is true was given wider ground" in critic_spec.system_prompt
    assert "exactly as for any other lane" in critic_spec.system_prompt


@pytest.mark.asyncio
async def test_a_drop_aimed_at_the_joint_lane_is_carried_out_and_recorded() -> None:
    """The width is sunk; refusing the drop spends a session instead of saving one."""
    _backend, service = _joint_lane_round(
        "VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 1, "reason": "the tile rewrite is too large a bet"})
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 2 draft",)
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["kept"] == 1
    assert narrowing["dropped"] == [{"lane_id": 1, "reason": "the tile rewrite is too large a bet"}]
    assert narrowing["joint"] == [1]
    assert narrowing["dropped_joint"] == [1]
    assert any("wider ground the partition bought" in note for note in narrowing["notes"])
    assert result.structured_output_diagnostics["lanes"]["published"] == 1


@pytest.mark.asyncio
async def test_a_drop_aimed_past_the_joint_lane_leaves_the_widened_lane() -> None:
    """A round that carries a joint lane narrows around it like any other."""
    _backend, service = _joint_lane_round(
        "VERDICT: ACCEPT\n\n" + _width_block({"lane_id": 2, "reason": "it is lane 1's staging in other words"})
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft",)
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["kept"] == 1
    assert narrowing["joint"] == [1]
    assert narrowing["dropped_joint"] == []
    assert result.structured_output_diagnostics["lanes"]["published"] == 1


def _drafts(*joint: bool) -> list:
    """One drafted lane per flag, the true ones widened to joint ground."""
    return [
        SynthesizedPlan(
            text=f"# Lane {index + 1} draft",
            ground=JOINT_GROUND if is_joint else f"{GROUND_B} ({index + 1})",
            joint=is_joint,
            fallback=("re-tune num_warps at the current tile alone" if is_joint else ""),
        )
        for index, is_joint in enumerate(joint)
    ]


def _ruling(*drops: tuple[int, str]) -> PlanCriticOutcome:
    """A review that answered its width block and named these lanes."""
    return PlanCriticOutcome(
        verdict="ACCEPT",
        review="VERDICT: ACCEPT",
        lane_drops=tuple(LaneDrop(lane_id=lane_id, reason=reason) for lane_id, reason in drops),
        narrowing_status="answered",
    )


def test_a_joint_lane_does_not_make_an_emptying_ruling_survivable() -> None:
    """The floor is one lane, and it must not be raised into a ranking."""
    drafts = _drafts(True, False, False)

    kept, diagnostics = OrchestrationService._narrow_round(
        drafts,
        critic_outcome=_ruling(
            (1, "the tile rewrite is too large a bet"),
            (2, "the evidence does not support it"),
            (3, "it is lane 2 in other words"),
        ),
        challenged=False,
    )

    assert diagnostics["status"] == "refused_empty_round"
    assert diagnostics["kept"] == 3
    assert diagnostics["joint"] == [1]
    assert diagnostics["dropped_joint"] == []
    assert kept == drafts


def test_a_round_of_only_joint_lanes_still_narrows() -> None:
    """Marking every lane joint must not switch narrowing off."""
    drafts = _drafts(True, True, True)

    kept, diagnostics = OrchestrationService._narrow_round(
        drafts,
        critic_outcome=_ruling(
            (2, "the evidence does not support it"),
            (3, "it is lane 1 in other words"),
        ),
        challenged=False,
    )

    assert [draft.text for draft in kept] == ["# Lane 1 draft"]
    assert diagnostics["status"] == "narrowed"
    assert diagnostics["kept"] == 1
    assert diagnostics["joint"] == [1, 2, 3]
    assert diagnostics["dropped_joint"] == [2, 3]
    assert any("wider ground the partition bought" in note for note in diagnostics["notes"])


@pytest.mark.asyncio
async def test_narrowing_the_round_cannot_read_names_what_it_kept() -> None:
    """ "The review wanted every lane" and "nobody could read it" differ."""
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {"lane_id": "two", "reason": "it duplicates lane 1"},
            {"lane_id": 5, "reason": "there is no lane 5"},
        )
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "not_applied"
    assert narrowing["block"] == "answered"
    assert narrowing["kept"] == 2
    assert any("lane drop names no lane" in note for note in narrowing["notes"])
    assert any("does not have" in note for note in narrowing["notes"])


@pytest.mark.asyncio
async def test_a_review_that_asks_for_nothing_leaves_the_round_alone() -> None:
    """The empty block is an answer; only it means "run every lane"."""
    _backend, service = _two_lane_round("VERDICT: ACCEPT\n\nBoth lanes earn it.\n" + _width_block())

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    assert result.structured_output_diagnostics["lane_narrowing"] == {
        "status": "not_requested",
        "block": "answered",
        "planned": 2,
        "kept": 2,
        "dropped": [],
        "joint": [],
        "dropped_joint": [],
        "notes": [],
    }


@pytest.mark.asyncio
async def test_a_round_whose_review_never_answered_on_width_says_so() -> None:
    """The case the DROP LANE regex passed over in silence."""
    _backend, service = _two_lane_round(
        "VERDICT: REVISE\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n",
        revisions=["# Lane 1 revised", "# Lane 2 revised"],
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 revised", "# Lane 2 revised")
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "not_applied"
    assert narrowing["block"] == "absent"
    assert narrowing["kept"] == 2
    assert narrowing["dropped"] == []
    assert narrowing["notes"] != []
    assert any("no lane_narrowing block" in note for note in narrowing["notes"])
    assert any("repair pass" in note for note in narrowing["notes"])
    assert result.structured_output_diagnostics["plan_critic"]["narrowing_status"] == "absent"


@pytest.mark.asyncio
async def test_a_round_narrows_on_a_ruling_its_review_was_asked_twice_for() -> None:
    """One repair call buys back an Implementer session the round would run."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(GROUND_A, GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
        ]
    )
    critic_backend = _QueuedBackend(
        [
            "VERDICT: ACCEPT\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n",
            _width_block(
                {
                    "lane_id": 2,
                    "reason": "it re-derives lane 1's autotune lever",
                }
            ),
        ]
    )
    service = _two_lane_critic_service(orchestration_backend, critic_backend)

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft",)
    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["block"] == "repaired"
    assert narrowing["dropped"] == [{"lane_id": 2, "reason": "it re-derives lane 1's autotune lever"}]
    assert result.structured_output_diagnostics["lanes"]["published"] == 1
    assert len(critic_backend.specs) == 2


@pytest.mark.asyncio
async def test_a_narrowed_round_never_records_that_it_kept_every_lane(
    caplog,
) -> None:
    """The record of a narrowed round has to agree with itself."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(GROUND_A, GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
        ]
    )
    critic_backend = _QueuedBackend(
        [
            "VERDICT: ACCEPT\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n",
            _width_block(
                {
                    "lane_id": 2,
                    "reason": "it re-derives lane 1's autotune lever",
                }
            ),
        ]
    )
    service = _two_lane_critic_service(orchestration_backend, critic_backend)

    with caplog.at_level(logging.INFO):
        result = await service.run(_context(), lanes=2)

    narrowing = result.structured_output_diagnostics["lane_narrowing"]
    assert narrowing["status"] == "narrowed"
    assert narrowing["kept"] == 1
    assert narrowing["dropped"] == [{"lane_id": 2, "reason": "it re-derives lane 1's autotune lever"}]
    # Each note says what reading the block found and leaves the outcome to `status` and `dropped`, which is the only
    # way the three can agree.
    assert narrowing["notes"] == [
        "the review ended with no lane_narrowing block",
        "the review did not end with a readable width block; one repair pass restated it",
    ]
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert not [line for line in warnings if "not applied" in line]
    assert not [line for line in warnings if "keeps every lane" in line]
    assert "round narrowed from 2 lanes to 1" in caplog.text


@pytest.mark.asyncio
async def test_a_narrowing_the_round_refused_is_still_reported_as_one(
    caplog,
) -> None:
    """The warning has to survive where it is true."""
    _backend, service = _two_lane_round(
        "VERDICT: ACCEPT\n\n"
        + _width_block(
            {"lane_id": 1, "reason": "unsupported"},
            {"lane_id": 2, "reason": "also unsupported"},
        )
    )

    with caplog.at_level(logging.INFO):
        result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert [line for line in warnings if "narrowing was not applied (refused_empty_round)" in line]


@pytest.mark.asyncio
async def test_a_review_that_asked_for_nothing_is_not_reported_as_a_failure(
    caplog,
) -> None:
    """The empty block is an answer, and answering costs the round nothing."""
    _backend, service = _two_lane_round("VERDICT: ACCEPT\n\nBoth lanes earn it.\n" + _width_block())

    with caplog.at_level(logging.INFO):
        result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 draft")
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING and "narrowing" in record.getMessage()
    ]
    assert "plan critic asked for no narrowing" in caplog.text


@pytest.mark.asyncio
async def test_the_round_records_what_each_planning_phase_cost() -> None:
    """A quarter of an eleven-hour budget went on planning, a third unseen."""
    _backend, service = _two_lane_round(
        "VERDICT: REVISE\n\nBoth lanes need a stop condition.",
        revisions=["# Lane 1 revised", "# Lane 2 revised"],
    )

    result = await service.run(_context(), lanes=2)

    durations = result.structured_output_diagnostics["phase_durations_sec"]
    assert list(durations) == [
        "dispatch",
        "specialists",
        "partition",
        "synthesis",
        "plan_critic",
        "plan_revision",
        "total",
    ]
    assert all(value >= 0 for value in durations.values())
    phases = [name for name in durations if name != "total"]
    # The named phases account for the round without exceeding it; what they do not cover stays visible as the
    # difference rather than being distributed.
    assert sum(durations[name] for name in phases) <= durations["total"] + 0.01
    assert durations["plan_critic"] == pytest.approx(result.plan_critic.duration_sec, abs=0.001)
    assert durations["plan_revision"] == pytest.approx(
        result.structured_output_diagnostics["plan_revision"]["duration_sec"],
        abs=0.001,
    )


@pytest.mark.asyncio
async def test_a_round_that_never_partitioned_reports_no_partition_cost() -> None:
    """An absent phase is absent, not zero: zero would read as instant."""
    result = await _single_lane_service().run(_context())

    durations = result.structured_output_diagnostics["phase_durations_sec"]
    assert list(durations) == ["dispatch", "specialists", "synthesis", "total"]


@pytest.mark.asyncio
async def test_a_lane_that_cannot_be_revised_keeps_its_draft() -> None:
    """The verdict was about the round, not about that lane being dangerous."""
    orchestration_backend = _QueuedBackend(
        [
            _dispatch_payload(both_roles=True),
            _partition(GROUND_A, GROUND_B),
            "# Lane 1 draft",
            "# Lane 2 draft",
            AgentRunResult(text="", end_reason="turn_cap"),
            "# Lane 2 revised",
        ]
    )
    service = _two_lane_critic_service(
        orchestration_backend,
        _StaticBackend("VERDICT: REVISE\n\nBoth lanes need a stop condition."),
    )

    result = await service.run(_context(), lanes=2)

    assert result.optimization_plans == ("# Lane 1 draft", "# Lane 2 revised")
    assert result.optimization_plan_executable is True
    assert result.plan_revised is True
    diagnostics = result.structured_output_diagnostics["plan_revision"]
    assert diagnostics["status"] == "partially_revised"
    assert diagnostics["unrevised_lanes"] == [1]
    assert "turn_cap" in diagnostics["message"]
