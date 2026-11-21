# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A gated session is told how far to take its own loop: past the first gain when judged on speed, to the first pass when not."""

from __future__ import annotations

import asyncio

import pytest

from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.config import Config
from kernelforge.evaluation.scoring import CaseAggregate
from kernelforge.workflows.forge_loop.roles import agent


@pytest.fixture()
def gated_system_prompt(tmp_path, monkeypatch):
    """Run one gated implementer session and return the system prompt it was given, whitespace-normalized."""

    def run(
        *, correctness_only: bool, max_blocks: int = 7, case_aggregate: CaseAggregate = CaseAggregate.ARITHMETIC
    ) -> str:
        kernel = tmp_path / "kernel.py"
        kernel.write_text("# kernel\n")
        driver = tmp_path / "driver.py"
        driver.write_text("raise AssertionError('prompt tests must not execute the driver')\n")
        specs = []

        class RecordingBackend:
            name = "claude"
            capabilities = AgentCapabilities(stop_hooks=True)

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
                session=agent.SessionOptions(insession_gate=True),
            ),
        )
        asyncio.run(agent_fn(str(kernel), ""))
        return " ".join(specs[0].system_prompt.split())

    return run


def test_a_session_judged_on_speed_works_one_direction_past_its_first_gain(gated_system_prompt) -> None:
    prompt = gated_system_prompt(correctness_only=False)

    # It commits to a direction and a number before it edits, and measures the kernel it starts from.
    assert "Choose this session's direction before the first edit" in prompt
    assert "That number is this session's goal" in prompt
    assert "Measure before you change anything" in prompt
    # The first gain opens the direction; it does not end the session.
    assert "A first improvement is where the session starts, not where it ends" in prompt
    assert "STOP exploring" not in prompt
    # What is handed off is the best version the session measured, kept outside the repository state.
    assert "Submit the best version, not the last one" in prompt
    assert "scratch directory outside the workspace" in prompt
    assert "Do not save or restore work with git" in prompt
    # It leaves room for the gate that judges its stop.
    assert "takes three benchmark measurements" in prompt


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
    assert "Choose this session's direction" not in prompt
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
