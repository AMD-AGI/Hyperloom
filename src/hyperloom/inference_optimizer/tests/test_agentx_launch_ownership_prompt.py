# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX prompts no longer send agents to Magpie's launch protocol: inference_suite launches every round."""

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

_MAGPIE_MODULE = "Magpie.modes.benchmark.agentx_launch"


def test_the_orchestration_prompt_names_no_magpie_launch_module():
    prompt = build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(benchmark_mode="agentx", agentx_backend="inferencex"),
    )
    assert _MAGPIE_MODULE not in prompt


@pytest.mark.parametrize("domain", ["research_scout_specialist", "serving_specialist"])
def test_no_specialist_prompt_names_a_magpie_launch_module(domain):
    inp = SpecialistPromptInputs(domain=get_domain(domain), benchmark_mode="agentx", agentx_backend="inferencex")
    assert _MAGPIE_MODULE not in "\n".join(build_specialist_prompts(inp))


def test_the_speculative_decoding_reference_points_at_the_golden_layer_inference_suite_enforces():
    doc = (asset_prompt_references_dir() / "speculative_decoding.md").read_text(encoding="utf-8")
    assert "## A base with a golden layer owns acceptance" in doc
    assert "inference_suite.golden" in doc
    for name in golden.__all__:
        if name.startswith("GOLDEN_OWNED_"):
            assert f"`{name}`" in doc
    assert "Magpie" not in doc
