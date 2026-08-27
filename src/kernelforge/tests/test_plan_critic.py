"""Tests for the free-form, fail-open orchestration plan critic."""

from __future__ import annotations

import json

import pytest

from kernelforge.agent_backends import AgentRunResult
from kernelforge.orchestrator.contracts import (
    CaseEvidence,
    DispatchPlan,
    LaneDrop,
    OrchestrationContext,
    PlanCriticOutcome,
    SynthesizedPlan,
)
from kernelforge.orchestrator.plan_critic import (
    PlanCriticAgent,
    build_plan_critic_prompts,
    parse_plan_critic_verdict,
    parse_plan_critic_width_block,
)


class _Backend:
    def __init__(self, result: AgentRunResult | Exception):
        self.result = result
        self.specs = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _QueuedBackend:
    """Answer each call with the next queued result, reusing the last."""

    def __init__(self, *results: AgentRunResult | Exception):
        self.results = list(results)
        self.specs = []

    async def run(self, spec, usage=None):
        self.specs.append(spec)
        result = self.results[min(len(self.specs) - 1, len(self.results) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


def _width_block(*drops: dict) -> str:
    """Render the trailing block the round contract asks a review to end with."""
    return json.dumps({"lane_narrowing": list(drops)})


def _two_lane_drafts() -> list[SynthesizedPlan]:
    return [
        SynthesizedPlan(text="# Lane 1", ground="a.py"),
        SynthesizedPlan(text="# Lane 2", ground="b.py"),
    ]


def _context(tmp_path) -> OrchestrationContext:
    workspace = tmp_path.resolve()
    source = workspace / "kernel.py"
    source.write_text("def kernel():\n    return 1\n")
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace=str(workspace),
        gpu_target="gfx942",
        objective="mean case speedup",
        program_context="Optimize the kernel.",
        source_map_path=str(source),
        cases=(CaseEvidence(case_id="case-a", latency_ms=1.0),),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("VERDICT: ACCEPT\n\nLooks sound.", "ACCEPT"),
        ("# Review\n\n### VERDICT: REPLACE\nChange route.", "REPLACE"),
        ("**VERDICT:** REVISE\nAdd evidence.", "REVISE"),
        (
            "VERDICT: REVISE\nLater text says VERDICT: ACCEPT",
            "REVISE",
        ),
        ("The plan lacks a canonical comparison.", "REVISE"),
    ],
)
def test_parse_plan_critic_verdict(text, expected):
    assert parse_plan_critic_verdict(text) == expected


def test_empty_critic_review_is_invalid():
    with pytest.raises(ValueError, match="no review"):
        parse_plan_critic_verdict("   ")


def test_critic_prompt_uses_checklist_and_workspace_paths(tmp_path):
    context = _context(tmp_path)
    system, user = build_plan_critic_prompts(
        context=context,
        drafts=[SynthesizedPlan(text="# Plan\nUse vector loads.")],
        dispatch_plan=DispatchPlan(
            analysis_commit="abc123",
            assignments=(),
        ),
        specialist_outcomes=(),
        coverage={
            "successful_roles": [],
            "covered_cases": [],
            "missing_cases": ["case-a"],
            "failed_roles": [],
        },
    )
    payload = json.loads(user)

    assert payload["context"]["workspace"] == str(tmp_path.resolve())
    assert payload["context"]["source_map_path"] == str((tmp_path / "kernel.py").resolve())
    assert payload["draft_plan"].startswith("# Plan")
    assert "should continue to exist" in system
    assert "existing GEMM" in system
    assert "opportunity cost" in system
    assert "do not mechanically repeat every item" in system


@pytest.mark.asyncio
async def test_critic_runs_one_independent_read_only_session(tmp_path):
    backend = _Backend(AgentRunResult(text="VERDICT: ACCEPT\n\nThe plan is evidence-grounded."))
    critic = PlanCriticAgent(
        backend=backend,
        timeout_sec=2,
        max_turns=4,
    )

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=[SynthesizedPlan(text="# Plan\nUse vector loads.")],
        dispatch_plan=DispatchPlan(
            analysis_commit="abc123",
            assignments=(),
        ),
        specialist_outcomes=(),
        coverage={},
    )

    assert outcome.verdict == "ACCEPT"
    assert outcome.fail_open is False
    assert outcome.verdict_source == "explicit"
    assert outcome.duration_sec >= 0
    assert "review" not in outcome.to_dict()
    assert len(backend.specs) == 1
    spec = backend.specs[0]
    assert spec.writable is False
    assert spec.tool_policy.read is True
    assert spec.tool_policy.search is True
    assert spec.tool_policy.write is False
    assert spec.tool_policy.shell is False
    assert spec.tool_policy.max_turns == 4


@pytest.mark.asyncio
async def test_critic_infers_revision_and_records_missing_verdict(
    tmp_path,
    caplog,
):
    critic = PlanCriticAgent(
        backend=_Backend(AgentRunResult(text="The plan needs a canonical comparison.")),
        timeout_sec=2,
    )

    with caplog.at_level(
        "WARNING",
        logger="kernelforge.orchestrator.plan_critic",
    ):
        outcome = await critic.review(
            context=_context(tmp_path),
            drafts=[SynthesizedPlan(text="# Draft")],
            dispatch_plan=DispatchPlan(
                analysis_commit="abc123",
                assignments=(),
            ),
            specialist_outcomes=(),
            coverage={},
        )

    assert outcome.verdict == "REVISE"
    assert outcome.verdict_source == "inferred"
    assert outcome.duration_sec >= 0
    assert "omitted an explicit VERDICT" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        AgentRunResult(text=""),
        AgentRunResult(
            text="provider failure",
            end_reason="api_error",
            stderr_tail="gateway unavailable",
        ),
        AgentRunResult(
            text=("I will inspect the evidence.\n[session ended with SDK error: Reached maximum number of turns]"),
            end_reason="turn_cap",
        ),
        RuntimeError("provider crashed"),
    ],
)
async def test_critic_failure_accepts_draft_fail_open(
    tmp_path,
    result,
    caplog,
):
    critic = PlanCriticAgent(
        backend=_Backend(result),
        timeout_sec=2,
    )

    with caplog.at_level(
        "WARNING",
        logger="kernelforge.orchestrator.plan_critic",
    ):
        outcome = await critic.review(
            context=_context(tmp_path),
            drafts=[SynthesizedPlan(text="# Draft")],
            dispatch_plan=DispatchPlan(
                analysis_commit="abc123",
                assignments=(),
            ),
            specialist_outcomes=(),
            coverage={},
        )

    assert outcome.verdict == "ACCEPT"
    assert outcome.fail_open is True
    assert outcome.error
    assert outcome.verdict_source == "error"
    assert outcome.duration_sec >= 0
    assert outcome.to_dict()["status"] == "CRITIC_ERROR"
    artifact = outcome.render_artifact()
    assert artifact.startswith("STATUS: CRITIC_ERROR")
    assert "VERDICT: ACCEPT" not in artifact
    assert "plan critic failed open to the draft" in caplog.text


def _round_prompts(tmp_path, drafts):
    return build_plan_critic_prompts(
        context=_context(tmp_path),
        drafts=drafts,
        dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
        specialist_outcomes=(),
        coverage={
            "successful_roles": [],
            "covered_cases": [],
            "missing_cases": [],
            "failed_roles": [],
        },
    )


def test_a_single_plan_is_reviewed_as_one_plan(tmp_path):
    """There is no division to review and no sibling to compare against."""
    system, user = _round_prompts(tmp_path, [SynthesizedPlan(text="# Plan\nUse vector loads.")])
    payload = json.loads(user)

    assert payload["draft_plan"] == "# Plan\nUse vector loads."
    assert "draft_lane_plans" not in payload
    assert "This round was divided into several lanes" not in system


def test_a_round_is_reviewed_with_its_division_in_view(tmp_path):
    """What a round raises cannot be asked of any lane on its own."""
    system, user = _round_prompts(
        tmp_path,
        [
            SynthesizedPlan(text="# Lane 1", ground="chunk_intra.py: the epilogue"),
            SynthesizedPlan(text="# Lane 2", ground="chunk.py: the dispatch gate"),
        ],
    )
    payload = json.loads(user)

    assert "draft_plan" not in payload
    assert payload["draft_lane_plans"] == [
        {
            "lane_id": 1,
            "ground": "chunk_intra.py: the epilogue",
            "joint": False,
            "fallback": "",
            "draft_plan": "# Lane 1",
        },
        {
            "lane_id": 2,
            "ground": "chunk.py: the dispatch gate",
            "joint": False,
            "fallback": "",
            "draft_plan": "# Lane 2",
        },
    ]
    assert "One verdict covers the round" in system


def test_a_review_needs_something_to_review(tmp_path):
    """An empty round is a programming error, not a plan worth a critic call."""
    with pytest.raises(ValueError):
        _round_prompts(tmp_path, [])


@pytest.mark.asyncio
async def test_critic_error_detail_is_single_line_and_bounded(tmp_path):
    critic = PlanCriticAgent(
        backend=_Backend(RuntimeError("first line\n" + ("x" * 3000))),
        timeout_sec=2,
    )

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=[SynthesizedPlan(text="# Draft")],
        dispatch_plan=DispatchPlan(
            analysis_commit="abc123",
            assignments=(),
        ),
        specialist_outcomes=(),
        coverage={},
    )

    assert "\n" not in outcome.error
    assert len(outcome.error) <= 2000
    assert outcome.error.endswith("...")


def test_a_lane_not_worth_its_session_can_be_named_with_its_reason():
    """The finding existed before the vocabulary did.

    Six production reviews said outright that a specific lane was not worth its
    Implementer session, and every one of those rounds ran every lane: a round
    verdict of ACCEPT/REVISE/REPLACE has no way to say "two of these three".
    """
    ruling = parse_plan_critic_width_block(
        "VERDICT: REVISE\n\nThe division buys lane 1 twice.\n\n```json\n"
        + json.dumps(
            {
                "lane_narrowing": [
                    {
                        "lane_id": 2,
                        "reason": "lane 1 already rewrites that epilogue",
                    },
                    {
                        "lane_id": "3",
                        "reason": "no profile supports the occupancy claim",
                    },
                ]
            },
            indent=2,
        )
        + "\n```\n"
    )

    assert [(drop.lane_id, drop.reason) for drop in ruling.drops] == [
        (2, "lane 1 already rewrites that epilogue"),
        (3, "no profile supports the occupancy claim"),
    ]
    assert ruling.notes == ()
    assert ruling.status == "answered"


def test_the_width_block_is_read_from_the_end_past_json_the_prose_quotes():
    """A kernel review quotes JSON; the first object in it is not the ruling.

    Taking the first complete object would hand the round an autotune config
    and report the ruling the review actually gave as missing.
    """
    ruling = parse_plan_critic_width_block(
        "VERDICT: REVISE\n\n"
        'Lane 2 pins {"BLOCK_M": 128, "num_warps": 8}, which lane 1 autotunes.\n'
        "\n" + _width_block({"lane_id": 2, "reason": "lane 1 autotunes it"})
    )

    assert [(drop.lane_id, drop.reason) for drop in ruling.drops] == [(2, "lane 1 autotunes it")]
    assert ruling.status == "answered"


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        ({"lane_id": "two", "reason": "it duplicates lane 1"}, "names no lane"),
        ({"lane_id": 2, "reason": "   "}, "lane drop states no reason"),
        ({"lane_id": 2}, "lane drop states no reason"),
        ({"lane_id": 0, "reason": "there is no lane 0"}, "lane drop names no lane"),
        ({"reason": "some lane, I forget which"}, "lane drop names no lane"),
        ({"lane_id": True, "reason": "a bool is not a lane"}, "lane drop names no lane"),
        ("DROP LANE 2", "unreadable lane drop"),
    ],
)
def test_narrowing_that_cannot_be_read_is_named_not_discarded(entry, problem):
    """Silence would make "keep every lane" the answer to two questions."""
    ruling = parse_plan_critic_width_block(f"VERDICT: REVISE\n{_width_block(entry)}\n")

    assert ruling.drops == ()
    # The block itself was read; only what it asked for could not be used.
    assert ruling.status == "answered"
    assert len(ruling.notes) == 1
    assert problem in ruling.notes[0]
    # The note names the entry and stops there. What the round does about it is
    # the round's answer to give, in `status` and in what it dropped.
    assert "kept" not in ruling.notes[0]
    assert ruling.unread is True


def test_a_lane_is_dropped_once_or_the_second_entry_is_reported():
    """Two reasons for one lane is a review that changed its mind mid-answer."""
    ruling = parse_plan_critic_width_block(
        _width_block(
            {"lane_id": 2, "reason": "it duplicates lane 1"},
            {"lane_id": 2, "reason": "on reflection, the ground is unsupported"},
        )
    )

    assert [drop.reason for drop in ruling.drops] == ["it duplicates lane 1"]
    assert len(ruling.notes) == 1
    assert ruling.notes[0].startswith("lane drop repeats a lane")
    # Nothing was lost with it: the entry before it dropped that lane. A note
    # that had concluded "so its lane is kept" would have said the opposite of
    # the drop standing beside it.
    assert ruling.unread is False


def test_a_narrowing_note_is_one_bounded_line():
    """The note is persisted with the round, so a runaway entry cannot be."""
    ruling = parse_plan_critic_width_block(_width_block({"lane_id": "two " + ("x " * 500), "reason": "duplicate"}))

    assert ruling.drops == ()
    assert len(ruling.notes[0]) <= 240
    assert "\n" not in ruling.notes[0]
    assert ruling.notes[0].endswith("...")


@pytest.mark.parametrize(
    ("review", "status", "problem"),
    [
        (
            "VERDICT: REVISE\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n",
            "absent",
            "no lane_narrowing block",
        ),
        (
            'VERDICT: REVISE\n\n{"lane_narrowing": [{"lane_id": 2,\n',
            "malformed",
            "must contain one complete JSON object",
        ),
        (
            'VERDICT: REVISE\n\n{"lane_narrowing": "drop lane 2"}\n',
            "malformed",
            "was not a list",
        ),
        (
            'VERDICT: REVISE\n\nI left "lane_narrowing" out.\n',
            "malformed",
            "outside any JSON object",
        ),
    ],
)
def test_a_block_that_was_never_readable_is_told_apart_from_one_that_was(
    review,
    status,
    problem,
):
    """Absent, malformed and empty are three answers, not one empty list."""
    ruling = parse_plan_critic_width_block(review)

    assert ruling.drops == ()
    assert ruling.status == status
    assert problem in ruling.notes[0]
    # The note says what was seen in the review. It does not say what the round
    # will do, because at this point one repair pass has yet to run and the
    # round has yet to rule -- the note is composed before either has answered.
    assert "keeps every lane" not in ruling.notes[0]
    assert ruling.unread is True


def test_a_lane_cannot_be_dropped_for_nothing():
    """The reason is what makes a narrowed round auditable afterwards."""
    with pytest.raises(ValueError, match="lane drop.reason"):
        LaneDrop(lane_id=2, reason="   ")
    with pytest.raises(ValueError, match="must be positive"):
        LaneDrop(lane_id=0, reason="it duplicates lane 1")


def test_one_lane_is_dropped_for_one_reason():
    with pytest.raises(ValueError, match="name each lane once"):
        PlanCriticOutcome(
            verdict="REVISE",
            lane_drops=(
                LaneDrop(lane_id=2, reason="it duplicates lane 1"),
                LaneDrop(lane_id=2, reason="its ground is unsupported"),
            ),
        )


def test_a_review_with_an_empty_block_asks_for_no_narrowing():
    """The empty list is an answer, and it is the one that means "run them all"."""
    ruling = parse_plan_critic_width_block("VERDICT: ACCEPT\nBoth lanes earn it.\n" + _width_block())

    assert ruling.drops == ()
    assert ruling.notes == ()
    assert ruling.status == "answered"


def test_the_round_contract_states_and_shows_the_width_schema(tmp_path):
    """A round-wide verdict cannot say which lane is not worth its session."""
    system, _user = _round_prompts(
        tmp_path,
        [
            SynthesizedPlan(text="# Lane 1", ground="chunk_intra.py: the epilogue"),
            SynthesizedPlan(text="# Lane 2", ground="chunk.py: the dispatch gate"),
        ],
    )

    assert '{"lane_narrowing": []}' in system
    assert '"lane_id": 2' in system
    assert "`lane_narrowing` is a list" in system
    assert "One verdict covers the round" in system
    assert "At least one lane always runs" in system


@pytest.mark.asyncio
async def test_the_reviews_narrowing_reaches_the_round(tmp_path):
    backend = _Backend(
        AgentRunResult(
            text=(
                "VERDICT: REVISE\n\nThe division buys one answer twice.\n\n"
                + _width_block(
                    {
                        "lane_id": 2,
                        "reason": "it is lane 1's change in different words",
                    },
                    {"lane_id": 9, "reason": ""},
                )
            )
        )
    )
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=_two_lane_drafts(),
        dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
        specialist_outcomes=(),
        coverage={},
    )

    assert outcome.verdict == "REVISE"
    assert [drop.lane_id for drop in outcome.lane_drops] == [2]
    assert len(outcome.narrowing_notes) == 1
    assert outcome.narrowing_notes[0].startswith("lane drop states no reason")
    # A block that was read is never repaired: the review answered, and the one
    # entry it got wrong is its decision to have gotten wrong.
    assert len(backend.specs) == 1
    persisted = outcome.to_dict()
    assert persisted["lane_drops"] == [{"lane_id": 2, "reason": "it is lane 1's change in different words"}]
    assert persisted["narrowing_notes"] == list(outcome.narrowing_notes)
    assert persisted["narrowing_status"] == "answered"


@pytest.mark.asyncio
async def test_a_drop_stated_only_in_prose_is_recovered_by_one_repair(tmp_path):
    """The case the DROP LANE regex lost outright.

    A review that writes its ruling as a sentence matched neither the directive
    pattern nor the pattern that reported unreadable directives, so it produced
    no drop and no note. With the block required, the same review is one
    repair pass away from the decision it made in its prose.
    """
    backend = _QueuedBackend(
        AgentRunResult(text=("VERDICT: REVISE\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n")),
        AgentRunResult(
            text=_width_block(
                {
                    "lane_id": 2,
                    "reason": "it re-derives lane 1's autotune lever",
                }
            )
        ),
    )
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=_two_lane_drafts(),
        dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
        specialist_outcomes=(),
        coverage={},
    )

    assert [(drop.lane_id, drop.reason) for drop in outcome.lane_drops] == [
        (2, "it re-derives lane 1's autotune lever")
    ]
    assert outcome.narrowing_status == "repaired"
    assert any("no lane_narrowing block" in n for n in outcome.narrowing_notes)
    assert any("one repair pass restated it" in n for n in outcome.narrowing_notes)
    # The repair is one extra call, given no tools and the review it repairs.
    assert len(backend.specs) == 2
    repair = backend.specs[1]
    assert repair.writable is False
    assert repair.tool_policy.read is False
    assert repair.tool_policy.search is False
    assert repair.tool_policy.max_turns == 2
    assert repair.timeout_sec == 2
    assert "Lane 2 should be dropped" in repair.user_prompt
    assert "lane_narrowing" in repair.user_prompt


@pytest.mark.asyncio
async def test_a_recovered_width_ruling_is_not_logged_as_a_failure(
    tmp_path,
    caplog,
):
    """Every note was logged as a failure, including on the path that worked.

    Block absent, repair pass restated it, one lane to drop -- and the review
    warned twice that the narrowing had not been applied, which was decided
    nowhere and was about to be contradicted by the round. A warning an
    operator learns is wrong costs more than the line it occupies.
    """
    backend = _QueuedBackend(
        AgentRunResult(text=("VERDICT: REVISE\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n")),
        AgentRunResult(
            text=_width_block(
                {
                    "lane_id": 2,
                    "reason": "it re-derives lane 1's autotune lever",
                }
            )
        ),
    )
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    with caplog.at_level(
        "INFO",
        logger="kernelforge.orchestrator.plan_critic",
    ):
        outcome = await critic.review(
            context=_context(tmp_path),
            drafts=_two_lane_drafts(),
            dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
            specialist_outcomes=(),
            coverage={},
        )

    assert [drop.lane_id for drop in outcome.lane_drops] == [2]
    assert outcome.narrowing_notes == (
        "the review ended with no lane_narrowing block",
        "the review did not end with a readable width block; one repair pass restated it",
    )
    assert not [record for record in caplog.records if record.levelname == "WARNING"]
    assert "plan critic width block was read (repaired)" in caplog.text


@pytest.mark.asyncio
async def test_a_width_ruling_nothing_recovered_is_still_a_warning(
    tmp_path,
    caplog,
):
    """The reading that genuinely lost a decision has to stay readable.

    The review stated a drop in prose only and the repair pass came back with
    nothing either, so the round is about to run a lane the review said was not
    worth its session and nobody can say which. That is the case the warning
    exists for.
    """
    backend = _QueuedBackend(
        AgentRunResult(text=("VERDICT: REVISE\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n")),
        AgentRunResult(text="I could not tell what the review wanted."),
    )
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    with caplog.at_level(
        "WARNING",
        logger="kernelforge.orchestrator.plan_critic",
    ):
        outcome = await critic.review(
            context=_context(tmp_path),
            drafts=_two_lane_drafts(),
            dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
            specialist_outcomes=(),
            coverage={},
        )

    assert outcome.lane_drops == ()
    assert outcome.narrowing_status == "absent"
    assert "width ruling was never read (absent)" in caplog.text
    assert "no readable width block either" in caplog.text


@pytest.mark.asyncio
async def test_an_entry_the_block_wasted_is_warned_about_as_that(
    tmp_path,
    caplog,
):
    """A block that was read can still lose a decision, and says which.

    The review named two lanes and gave the second no reason, so that drop is
    gone while the first is applied. The warning names the entry rather than
    reporting the round as unnarrowed, which the drop beside it disproves.
    """
    backend = _Backend(
        AgentRunResult(
            text=(
                "VERDICT: REVISE\n\n"
                + _width_block(
                    {"lane_id": 2, "reason": "it duplicates lane 1"},
                    {"lane_id": 3, "reason": "   "},
                )
            )
        )
    )
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    with caplog.at_level(
        "WARNING",
        logger="kernelforge.orchestrator.plan_critic",
    ):
        outcome = await critic.review(
            context=_context(tmp_path),
            drafts=_two_lane_drafts(),
            dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
            specialist_outcomes=(),
            coverage={},
        )

    assert [drop.lane_id for drop in outcome.lane_drops] == [2]
    assert "part of what it asked for was not" in caplog.text
    assert "lane drop states no reason" in caplog.text
    assert "not applied" not in caplog.text


@pytest.mark.asyncio
async def test_a_drop_stated_only_in_prose_that_repair_misses_is_still_named(
    tmp_path,
):
    """The one outcome that must never be silence.

    Repair is the only thing standing between a prose-only ruling and a round
    that runs the lane anyway. When it fails, the round runs the lane -- and
    says, in the diagnostics it persists, that it was asked something it could
    not read.
    """
    backend = _QueuedBackend(
        AgentRunResult(text=("VERDICT: REVISE\n\nLane 2 should be dropped: it re-derives lane 1's autotune lever.\n")),
        RuntimeError("provider crashed"),
    )
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=_two_lane_drafts(),
        dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
        specialist_outcomes=(),
        coverage={},
    )

    assert outcome.verdict == "REVISE"
    assert outcome.fail_open is False
    assert outcome.lane_drops == ()
    assert outcome.narrowing_status == "absent"
    assert any("no lane_narrowing block" in n for n in outcome.narrowing_notes)
    assert any("one repair pass for the width block failed" in note for note in outcome.narrowing_notes)


@pytest.mark.asyncio
async def test_a_repair_that_answers_nothing_leaves_the_ruling_unread(tmp_path):
    """A repair pass that comes back without a block changes no width."""
    backend = _QueuedBackend(
        AgentRunResult(text="VERDICT: ACCEPT\n\nBoth lanes earn a session.\n"),
        AgentRunResult(text="I could not tell what the review wanted."),
    )
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=_two_lane_drafts(),
        dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
        specialist_outcomes=(),
        coverage={},
    )

    assert outcome.lane_drops == ()
    assert outcome.narrowing_status == "absent"
    assert any("no readable width block either" in note for note in outcome.narrowing_notes)


@pytest.mark.asyncio
async def test_a_one_plan_review_is_never_asked_for_a_width_block(tmp_path):
    """There is no division to rule on, so no block is owed and none is bought."""
    backend = _Backend(AgentRunResult(text="VERDICT: ACCEPT\n\nThe plan is grounded."))
    critic = PlanCriticAgent(backend=backend, timeout_sec=2)

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=[SynthesizedPlan(text="# Plan")],
        dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
        specialist_outcomes=(),
        coverage={},
    )

    assert outcome.narrowing_status == "not_asked"
    assert outcome.narrowing_notes == ()
    assert len(backend.specs) == 1


@pytest.mark.asyncio
async def test_a_review_that_failed_narrows_nothing(tmp_path):
    """A round the critic never reviewed keeps every lane it planned."""
    critic = PlanCriticAgent(
        backend=_Backend(RuntimeError("provider crashed")),
        timeout_sec=2,
    )

    outcome = await critic.review(
        context=_context(tmp_path),
        drafts=[SynthesizedPlan(text="# Draft")],
        dispatch_plan=DispatchPlan(analysis_commit="abc123", assignments=()),
        specialist_outcomes=(),
        coverage={},
    )

    assert outcome.fail_open is True
    assert outcome.lane_drops == ()
    assert outcome.narrowing_notes == ()


def test_a_one_plan_review_keeps_the_budget_it_always_had():
    """Guards the scaling from moving the single-lane path it was not for."""
    critic = PlanCriticAgent(backend=_Backend(AgentRunResult(text="")), timeout_sec=600)

    assert critic._budget_for(1) == 600


def test_a_round_of_several_plans_is_several_times_the_reading():
    """A budget sized for one plan fails a round open, losing its verdict.

    Measured on a real two-lane round: eleven minutes of review against a
    ten-minute budget, so the verdict -- which had found one lane not worth its
    session -- never reached the round.
    """
    critic = PlanCriticAgent(
        backend=_Backend(AgentRunResult(text="")),
        timeout_sec=600,
        ceiling_sec=1800,
    )

    assert critic._budget_for(2) == 1200
    assert critic._budget_for(3) == 1800
    # Never past what the provider allows one call.
    assert critic._budget_for(9) == 1800
