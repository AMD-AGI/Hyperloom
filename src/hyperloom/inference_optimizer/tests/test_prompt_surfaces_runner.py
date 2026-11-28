# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The serving specialist reruns a point on the GPU as the gate's inference_suite rounds."""

from __future__ import annotations


from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs


def test_the_on_gpu_rebench_helper_is_the_gates_rounds_on_a_point_yaml_and_says_it_discards_a_first_bench():
    inp = SpecialistPromptInputs(domain=get_domain("serving_specialist"), allocated_gpu_ids=(0,))
    prompt = "\n".join(build_specialist_prompts(inp))
    assert "the gate's own inference_suite rounds" in prompt
    assert "it discards a first bench and reports the second, measured on the same server" in prompt
    assert "takes about two benchmark passes" in prompt
    assert "[--config <point.yaml>]" in prompt
