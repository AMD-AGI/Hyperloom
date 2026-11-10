# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Token rollup arithmetic for the ``decision_trace.jsonl`` per-decision totals.

Pins the counter families apart: visible prompt/completion, cache, and hidden
reasoning output, so ``total_out`` keeps counting only what the model actually said.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.trace import decision_trace as dc


def test_bucket_rolls_up_reasoning_tokens_separately():
    bucket = dc._empty_token_bucket()
    dc._fold_call_into_bucket(
        bucket,
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 7,
            "reasoning_output_tokens": 4096,
        },
    )
    assert bucket["total_out"] == 20  # the visible reply only
    assert bucket["total_reasoning_out"] == 4096
    assert bucket["calls"] == 1
