# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.trace import pricing


@pytest.fixture(autouse=True)
def _fresh_rates(monkeypatch, tmp_path: Path):
    """Point pricing at a small, known rate card for every test here."""
    card = tmp_path / "rates.yaml"
    card.write_text(
        "version: 1\n"
        "defaults:\n"
        "  cache_read: null\n"
        "  cache_write: null\n"
        "models:\n"
        "  test-model:\n"
        "    input: 1.0\n"
        "    output: 10.0\n"
        "    cache_read: 0.1\n"
        "    cache_write: 2.0\n"
        "  plain-model:\n"
        "    input: 3.0\n"
        "    output: 6.0\n"
    )
    monkeypatch.setenv("HYPERLOOM_LLM_PRICING_FILE", str(card))
    pricing.load_rates(refresh=True)
    yield
    monkeypatch.delenv("HYPERLOOM_LLM_PRICING_FILE", raising=False)
    pricing.load_rates(refresh=True)


def test_normalize_model_strips_provider_prefix_and_tag():
    assert pricing.normalize_model("Anthropic/Claude-Opus-5:thinking") == "claude-opus-5"
    assert pricing.normalize_model(None) == ""


def test_longest_prefix_match_wins():
    assert pricing.rates_for("test-model-20260101").input == 1.0
    assert pricing.rates_for("nothing-like-this") is None


def test_derived_total_is_the_sum_of_its_parts():
    cost = pricing.resolve_cost(
        model="test-model",
        tokens={
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "reasoning_output_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
        },
    )
    assert cost.source == pricing.SOURCE_DERIVED
    assert cost.input_usd == pytest.approx(1.0)
    assert cost.output_usd == pytest.approx(10.0)
    # Reasoning bills at the output rate, and is its own bucket.
    assert cost.thinking_usd == pytest.approx(10.0)
    assert cost.cache_usd == pytest.approx(2.1)
    assert cost.total_usd == pytest.approx(
        cost.input_usd + cost.output_usd + cost.thinking_usd + cost.cache_usd
    )


def test_unset_cache_rates_fall_back_to_the_input_rate():
    cost = pricing.resolve_cost(
        model="plain-model",
        tokens={"cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000},
    )
    assert cost.cache_usd == pytest.approx(6.0)


def test_provider_figure_wins_and_the_split_only_attributes_it():
    cost = pricing.resolve_cost(
        model="test-model",
        tokens={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        provider_usd=110.0,
    )
    assert cost.source == pricing.SOURCE_PROVIDER
    assert cost.total_usd == pytest.approx(110.0)
    # Card says 1.0 in / 10.0 out; the provider's total is split in that ratio.
    assert cost.input_usd == pytest.approx(10.0)
    assert cost.output_usd == pytest.approx(100.0)


def test_provider_figure_survives_an_unknown_model_without_a_split():
    cost = pricing.resolve_cost(model="who-knows", tokens={"input_tokens": 5}, provider_usd=0.25)
    assert cost.source == pricing.SOURCE_PROVIDER
    assert cost.total_usd == pytest.approx(0.25)
    assert cost.input_usd is None


def test_unknown_model_and_no_provider_figure_is_unavailable():
    cost = pricing.resolve_cost(model="who-knows", tokens={"input_tokens": 5})
    assert cost.source == pricing.SOURCE_UNAVAILABLE
    assert cost.total_usd is None
    assert not cost.available
