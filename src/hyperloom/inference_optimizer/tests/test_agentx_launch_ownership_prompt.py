# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX prompts name no retired runner: inference_suite launches every round."""

from __future__ import annotations

import pytest

from inference_suite import golden

from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir
from hyperloom.orchestrator.prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
    SessionContext,
)
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs
from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.inference_optimizer.tests.test_prompt_surfaces_runner import _RETIRED


def test_the_agentx_orchestration_prompt_names_no_retired_runner():
    prompt = build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(benchmark_mode="agentx", agentx_backend="inferencex"),
    )
    assert _RETIRED.findall(prompt) == []


@pytest.mark.parametrize("domain", ["research_scout_specialist", "serving_specialist"])
def test_no_agentx_specialist_prompt_names_a_retired_runner(domain):
    inp = SpecialistPromptInputs(domain=get_domain(domain), benchmark_mode="agentx", agentx_backend="inferencex")
    assert _RETIRED.findall("\n".join(build_specialist_prompts(inp))) == []


def test_the_speculative_decoding_reference_points_at_the_golden_layer_inference_suite_enforces():
    doc = (asset_prompt_references_dir() / "speculative_decoding.md").read_text(encoding="utf-8")
    assert "## A base with a golden layer owns acceptance" in doc
    assert "inference_suite.golden" in doc
    for name in golden.__all__:
        if name.startswith("GOLDEN_OWNED_"):
            assert f"`{name}`" in doc
    assert _RETIRED.findall(doc) == []
