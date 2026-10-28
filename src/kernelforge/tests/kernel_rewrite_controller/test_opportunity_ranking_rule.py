# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the opportunity agent ranks candidates on."""

from __future__ import annotations

from kernelforge.kernel_rewrite_controller import opportunity_agent as oa


def test_candidates_are_ranked_on_the_decode_step_and_a_sliver_of_it_is_not_published():
    """S1: a bf16 add at 0.05% of decode time, ranked on whole-trace share, held the fusion lane for 93 minutes."""
    prompt = oa._system_prompt()

    assert "Prefer the largest measured end-to-end GPU-time share." not in prompt
    assert oa.DECODE_SHARE_RULE in prompt
    assert "below 1% of the kernel time in every decode class" in oa.DECODE_SHARE_RULE
