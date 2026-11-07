# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent-facing prompts and references describe inference_suite rounds, never Magpie, bypass or runner_type."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hyperloom.common.failure_signature import classify_failure
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.paths import asset_root, asset_system_prompts_dir
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.prompts.prompt_builder import (
    PromptAssets,
    PromptCycle,
    SessionContext,
    build_orchestration_prompt,
    default_enabled_actions,
)
from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts
from hyperloom.orchestrator.specialists.domains import SPECIALIST_DOMAIN_KEYS, get_domain
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs

_RETIRED = re.compile(
    r"magpie|runner_type|HYPERLOOM_BENCHMARK_BACKEND|INFERENCE_OPTIMIZER_RESCUE_PATHS|rescued_from_leaked_path",
    re.IGNORECASE,
)


def _documents() -> list[Path]:
    package = asset_root()
    return sorted(
        [
            package / "SKILL.md",
            package / "multi_node" / "SKILL.md",
            *(package / "references").glob("*.md"),
            *(package / "actions").glob("*.md"),
            *asset_system_prompts_dir().rglob("*.md"),
        ]
    )


@pytest.mark.parametrize("path", _documents(), ids=lambda path: f"{path.parent.name}/{path.name}")
def test_no_agent_document_names_the_retired_runner(path: Path):
    assert _RETIRED.findall(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("phase", sorted(machine_state.PHASE_NAMES))
def test_no_orchestration_prompt_names_the_retired_runner(phase: str):
    prompt = build_orchestration_prompt(
        action_registry=ACTION_CATALOGUE,
        enabled_actions=default_enabled_actions(no_kernel=False),
        session=SessionContext(framework="sglang", framework_agent_phase_enabled=True, max_minutes=480),
        cycle=PromptCycle(phase=phase),
        assets=PromptAssets(rules_fragment_path=asset_system_prompts_dir() / "orchestration.md"),
    )
    assert _RETIRED.findall(prompt) == []


@pytest.mark.parametrize("domain", sorted(SPECIALIST_DOMAIN_KEYS))
def test_no_specialist_prompt_names_the_retired_runner(domain: str):
    # An enablement specialist is only dispatched on a classified failure.
    signature = classify_failure("ValueError: Model architectures ['X'] are not supported for now.").to_dict()
    inp = SpecialistPromptInputs(
        domain=get_domain(domain),
        framework="sglang",
        allocated_gpu_ids=(0, 1),
        enablement_failure_signature=signature,
    )
    assert _RETIRED.findall("\n".join(build_specialist_prompts(inp))) == []


def test_the_on_gpu_rebench_helper_is_the_gate_round_on_a_point_yaml():
    inp = SpecialistPromptInputs(domain=get_domain("serving_specialist"), allocated_gpu_ids=(0,))
    prompt = "\n".join(build_specialist_prompts(inp))
    assert "inference_suite round" in prompt
    assert "[--config <point.yaml>]" in prompt
