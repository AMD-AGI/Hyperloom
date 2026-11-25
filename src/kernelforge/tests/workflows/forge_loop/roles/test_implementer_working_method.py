# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A gated session is told how far to take its own loop: past the first gain when judged on speed, to the first pass when not."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.config import Config
from kernelforge.evaluation.scoring import CaseAggregate
from kernelforge.workflows.forge_loop.roles import agent, implementer_method


@pytest.fixture()
def gated_prompts(tmp_path, monkeypatch):
    """Run one gated implementer session and return its system and user prompts, whitespace-normalized."""

    def run(
        *,
        correctness_only: bool,
        max_blocks: int = 7,
        case_aggregate: CaseAggregate = CaseAggregate.ARITHMETIC,
        session_timeout_sec: int | None = None,
        capabilities: AgentCapabilities = AgentCapabilities(stop_hooks=True, session_process_mark=True),
    ) -> tuple[str, str]:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("# kernel\n")
        driver = tmp_path / "driver.py"
        driver.write_text("raise AssertionError('prompt tests must not execute the driver')\n")
        specs = []

        session_capabilities = capabilities

        class RecordingBackend:
            name = "claude"
            capabilities = session_capabilities

            def __init__(self, runtime):
                self.runtime = runtime

            async def run(self, spec, usage=None):
                specs.append(spec)
                return AgentRunResult(text="PLAN: nothing")

        monkeypatch.setattr(agent, "create_registered_backend", lambda runtime, **kwargs: RecordingBackend(runtime))
        agent_fn = agent.make_agent_fn(
            Config(
                gpu_target="gfx950",
                workspace=str(tmp_path),
                agent_backend="claude",
                agent_model="claude-test",
                agent_precheck=False,
            ),
            agent.ImplementerSetup(
                agent.ImplementerTask(program_md="Optimize the kernel.", kernel_backend_name="triton"),
                measurement=agent.GateMeasurement(
                    driver_script=str(driver),
                    correctness_only=correctness_only,
                    max_blocks=max_blocks,
                    case_aggregate=case_aggregate,
                ),
                session=agent.SessionOptions(insession_gate=True, session_timeout_sec=session_timeout_sec),
            ),
        )
        asyncio.run(agent_fn(str(kernel), ""))
        return " ".join(specs[0].system_prompt.split()), " ".join(specs[0].user_prompt.split())

    return run


@pytest.fixture()
def gated_system_prompt(gated_prompts):
    """The system prompt of one gated implementer session, whitespace-normalized."""
    return lambda **options: gated_prompts(**options)[0]


def test_a_session_judged_on_speed_works_one_direction_past_its_first_gain(gated_system_prompt) -> None:
    prompt = gated_system_prompt(correctness_only=False)

    # It commits to a direction and a number before it edits, and measures the kernel it starts from.
    assert "Fix this session's direction before the first edit" in prompt
    assert "That number is this session's goal" in prompt
    assert "Measure before you change anything" in prompt
    # The first gain opens the direction; it does not end the session.
    assert "A first improvement is where the session starts, not where it ends" in prompt
    assert "STOP exploring" not in prompt
    # What is handed off is the best version the session measured, kept outside the repository state.
    assert "Submit the best version, not the last one" in prompt
    assert "scratch directory outside the workspace" in prompt
    assert "Do not save or restore work with git" in prompt
    # It knows what the gate that judges its stop will run.
    assert "takes three benchmark measurements" in prompt


def test_a_session_judged_on_speed_takes_the_largest_direction_it_sees(gated_system_prompt) -> None:
    prompt = gated_system_prompt(correctness_only=False)

    assert "Take the larger expected gain even when it means restructuring the kernel" in prompt
    assert "tuning constants follows a structural change, it does not replace one" in prompt
    # While time is ample, neither the clock nor the risk is a reason to pass a direction over.
    assert (
        f"While more than {implementer_method.AMPLE_TIME_MINUTES} minutes remain before the wrap-up minute, "
        '"not enough time" and "too risky" are not reasons to pass over a direction'
    ) in prompt
    # Nothing in it asks for an early hand-off.
    assert "no candidate costs the campaign less" not in prompt


def test_a_session_judged_on_speed_knows_when_to_change_direction_and_when_to_finish(gated_system_prompt) -> None:
    prompt = gated_system_prompt(correctness_only=False)

    # The clock is read with every measurement rather than estimated.
    assert "--bench-mode; date" in prompt
    assert "End every later driver command with `; date` as well" in prompt
    assert "Read the clock, never estimate it" in prompt
    # A direction is left on evidence, and the session finishes on the clock, on a stall or on an exhausted list.
    assert f"when {implementer_method.DIRECTION_STALL_CHANGES} measured changes in a row have not improved" in prompt
    assert f"when {implementer_method.SESSION_STALL_CHANGES} measured changes in a row — across directions —" in prompt
    assert "when every direction worth measuring has been measured" in prompt
    assert "A first improvement is none of these three" in prompt
    # What a cut-off hands to the loop is the workspace, so the best version lives there between measurements.
    assert "the workspace holds the best version this session has measured" in prompt
    # A measurement neither runs in the background nor outlasts the prompt cache.
    assert f"keep each command under {implementer_method.MAX_COMMAND_MINUTES} minutes" in prompt


def test_a_session_judged_on_speed_is_given_its_wrap_up_minute(gated_prompts) -> None:
    _system, user = gated_prompts(correctness_only=False, session_timeout_sec=3600)

    assert "## Session budget" in user
    assert "Run `date` before anything else: that reading is minute 0 of this session." in user
    assert "You have about 60 minutes of wall-clock from it, and the session is meant to use them" in user
    assert "The wrap-up minute is 48" in user
    assert "the loop validates whatever tracked changes the workspace holds at that moment" in user
    # The paced session is not told to stop early.
    assert "do not reach for a larger gain" not in user


def test_a_session_judged_on_correctness_alone_is_given_only_its_deadline(gated_prompts) -> None:
    _system, user = gated_prompts(correctness_only=True, session_timeout_sec=3600)

    assert "## Session deadline" in user
    assert "wrap-up minute" not in user


def test_a_session_judged_on_speed_ends_without_a_candidate_rather_than_a_near_tie(gated_system_prompt) -> None:
    prompt = gated_system_prompt(correctness_only=False, max_blocks=7)

    assert "restore every file to how the session found it" in prompt
    assert "after 7 such rejected stops it ends the session" in prompt


def test_a_session_judged_on_correctness_alone_submits_its_first_passing_kernel(gated_system_prompt) -> None:
    prompt = gated_system_prompt(correctness_only=True, max_blocks=7)

    assert "This phase is judged on correctness alone" in prompt
    assert "As soon as the kernel passes the driver's correctness verdict, stop and submit it" in prompt
    assert "after 7 such rejected stops it ends the session" in prompt
    # Nothing in it asks for a performance search.
    assert "Fix this session's direction" not in prompt
    assert "A first improvement is where the session starts" not in prompt


@pytest.mark.parametrize(
    "case_aggregate,named,formula",
    [
        (CaseAggregate.ARITHMETIC, "equal-weight arithmetic mean", "`mean(pristine_case_ms / candidate_case_ms)`"),
        (CaseAggregate.GEOMETRIC, "equal-weight geometric mean", "`geomean(pristine_case_ms / candidate_case_ms)`"),
    ],
)
def test_a_session_is_told_the_score_its_campaign_is_judged_on(
    gated_system_prompt, case_aggregate, named, formula
) -> None:
    prompt = gated_system_prompt(correctness_only=False, case_aggregate=case_aggregate)

    assert f"The score is the {named} of per-case speedups: {formula}" in prompt
    assert f"Each of three independent measurements is scored as {formula} with equal case weight" in prompt
    assert "Judge a change by the score defined above" in prompt


_SCOPED_KILL = "kernelforge.utils.session_kill '<pattern>'"


def test_a_provider_that_marks_its_sessions_and_runs_the_hooks_is_told_the_process_rule(gated_system_prompt) -> None:
    prompt = gated_system_prompt(correctness_only=False)

    assert "- Stop only what this session started." in prompt
    assert _SCOPED_KILL in prompt
    assert "and are denied. - As your VERY LAST output" in prompt


@pytest.mark.parametrize(
    "capabilities",
    [
        AgentCapabilities(),
        AgentCapabilities(stop_hooks=True),
        AgentCapabilities(session_process_mark=True),
    ],
    ids=["neither", "hooks without a mark", "a mark without hooks"],
)
def test_a_provider_that_cannot_keep_the_process_rule_is_not_told_it(gated_system_prompt, capabilities) -> None:
    """Without the mark the scoped kill matches nothing, and without the hooks a kill by name is not denied."""
    prompt = gated_system_prompt(correctness_only=False, capabilities=capabilities)

    assert "Stop only what this session started" not in prompt
    assert _SCOPED_KILL not in prompt
    assert "are denied" not in prompt
    # The rule list closes up around the missing line.
    assert "- - " not in prompt
    assert "- As your VERY LAST output" in prompt


def test_only_the_provider_that_marks_its_sessions_declares_it() -> None:
    """The registry is where a provider says its sessions carry the mark the scoped kill selects by."""
    from kernelforge.agent_backends.registry import get_agent_provider, list_agent_providers

    marking = {name for name in list_agent_providers() if get_agent_provider(name).capabilities.session_process_mark}

    assert marking == {"claude"}


def test_the_method_leaves_the_direction_to_the_session_inside_its_lanes_focus(gated_system_prompt) -> None:
    """One method serves a round's only session and a lane given a focus: both choose their own direction."""
    prompt = gated_system_prompt(correctness_only=False)

    assert "Choose it from the evidence in this prompt" in prompt
    assert "When this invocation names a focus for this lane, choose inside it." in prompt
    assert "required optimization plan" not in prompt
    assert "Choose this session's direction" not in prompt
