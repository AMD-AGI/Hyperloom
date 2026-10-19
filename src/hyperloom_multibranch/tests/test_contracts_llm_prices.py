# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL
from hyperloom_multibranch.contracts.llm_prices import ModelPrice, TokenCounts, model_key, price_of


@pytest.mark.parametrize(
    "name", ["Claude-Opus-5.5", "claude-opus-5-5", "claude-opus-5-5-20261001", "claude-opus-5-5[1m]"]
)
def test_a_model_is_found_whatever_way_its_name_is_written(name):
    assert model_key(name) == "claude-opus-5-5"
    assert price_of(name) == ModelPrice(4, 20, 0.2, 5, 8)


def test_every_default_model_a_session_runs_has_a_price():
    for model in (DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL, "deepseek-v4-pro"):
        assert price_of(model).input > 0


def test_a_model_the_sheet_does_not_list_has_no_price():
    assert price_of("glm-5-3") is None


def test_each_kind_of_token_is_billed_at_its_own_rate():
    tokens = TokenCounts(input=1_000, output=2_000, cache_read=100_000, cache_write=10_000, cache_write_1h=1_000)

    assert price_of("Claude-Opus-5.5").cost_usd(tokens) == pytest.approx(
        (1_000 * 4 + 2_000 * 20 + 100_000 * 0.2 + 10_000 * 5 + 1_000 * 8) / 1e6
    )
    assert tokens.prompt == 112_000


def test_a_cache_write_without_its_own_rate_takes_the_next_rate_listed():
    tokens = TokenCounts(cache_write=1_000_000, cache_write_1h=1_000_000)

    assert price_of("gpt-5.6-sol").cost_usd(tokens) == pytest.approx(3.4 * 2)
    assert price_of("DeepSeek-V4-Pro").cost_usd(tokens) == pytest.approx(1.183 * 2)


@pytest.mark.parametrize(
    ("usage", "counts"),
    [
        (
            {
                "input_tokens": 12,
                "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 400,
                "output_tokens": 9,
            },
            TokenCounts(12, 9, 3000, 400),
        ),
        (
            {
                "cache_creation_input_tokens": 400,
                "cache_creation": {"ephemeral_5m_input_tokens": 300, "ephemeral_1h_input_tokens": 100},
            },
            TokenCounts(cache_write=300, cache_write_1h=100),
        ),
        ({"input_tokens": None, "output_tokens": -1, "cache_read_input_tokens": True}, TokenCounts()),
    ],
    ids=["flat", "split-cache-write", "unusable"],
)
def test_an_anthropic_usage_block_maps_onto_the_counts_it_is_billed_by(usage, counts):
    assert TokenCounts.from_anthropic_usage(usage) == counts


def test_token_counts_add_up_and_are_never_negative():
    assert TokenCounts(1, 2, 3, 4, 5) + TokenCounts(1, 1, 1, 1, 1) == TokenCounts(2, 3, 4, 5, 6)
    with pytest.raises(ValueError):
        TokenCounts(output=-1)
