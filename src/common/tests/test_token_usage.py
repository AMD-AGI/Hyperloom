# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

from common.token_usage import anthropic_usage_counts, openai_usage_counts


def test_openai_input_excludes_both_cache_shares():
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        cache_creation_input_tokens=300,
        prompt_tokens_details=SimpleNamespace(cached_tokens=200),
    )
    assert openai_usage_counts(usage) == {
        "input_tokens": 500,
        "output_tokens": 50,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 300,
    }


def test_anthropic_counts_zero_fill_and_tolerate_garbage():
    usage = {"input_tokens": "x", "output_tokens": 4, "cache_read_input_tokens": -7}
    assert anthropic_usage_counts(usage) == {
        "input_tokens": 0,
        "output_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def test_a_missing_usage_has_no_counts():
    assert openai_usage_counts(None) == {}
    assert anthropic_usage_counts(None) == {}
