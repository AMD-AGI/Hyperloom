# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The serving specialist reruns a point on the GPU as an inference_suite round."""

from __future__ import annotations


from hyperloom.orchestrator.prompts.specialist_prompt_builder import build_specialist_prompts
from hyperloom.orchestrator.specialists.domains import get_domain
from hyperloom.orchestrator.specialists.prompt_inputs import SpecialistPromptInputs


def test_the_on_gpu_rebench_helper_is_the_gate_round_on_a_point_yaml():
    inp = SpecialistPromptInputs(domain=get_domain("serving_specialist"), allocated_gpu_ids=(0,))
    prompt = "\n".join(build_specialist_prompts(inp))
    assert "inference_suite round" in prompt
    assert "[--config <point.yaml>]" in prompt
