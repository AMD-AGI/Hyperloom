# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Token rollup arithmetic for the ``decision_trace`` / ``token_usage`` sections.

Pins the counter families apart: visible prompt/completion, cache, and hidden
reasoning output. ``grand_total`` is documented as the all-in spend, so a
reasoning model's hidden output has to be in it — while ``total_out`` keeps
counting only what the model actually said.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.breakdown.collectors import decision as dc


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


def test_grand_total_is_all_in_including_reasoning():
    view = dc._token_convenience(
        {
            "total_in": 100,
            "total_out": 20,
            "total_cache_creation": 5,
            "total_cache_read": 7,
            "total_reasoning_out": 4096,
        }
    )
    assert view["total_in_out"] == 120
    assert view["grand_total"] == 100 + 20 + 5 + 7 + 4096


def test_grand_total_unchanged_without_reasoning_tokens():
    """A non-reasoning model reports 0 there, so its all-in figure is unmoved."""
    view = dc._token_convenience(
        {
            "total_in": 10,
            "total_out": 4,
            "total_cache": 6,
        }
    )
    assert view["grand_total"] == 20
    assert view["total_in_out"] == 14
