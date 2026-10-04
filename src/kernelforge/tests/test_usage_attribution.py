# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""A campaign total that cannot be attributed cannot be argued with.

These pin the property that makes the breakdown usable: it partitions the
same numbers the campaign total is built from, so the split can never tell a
different story than the headline, and a caller that names nothing is still
counted rather than dropped.
"""

from __future__ import annotations

from kernelforge.agent_backends.base import AgentRunSpec, AgentRuntimeConfig
from kernelforge.loop.runner import llm_spend_lines
from kernelforge.tracker.usage import UsageAccumulator


def _usage(inp: int, out: int) -> dict[str, int]:
    return {"input_tokens": inp, "output_tokens": out}


def test_role_buckets_sum_to_the_campaign_total() -> None:
    """The split is a partition of the headline, not a second measurement."""
    usage = UsageAccumulator()
    usage.add_usage(_usage(100, 10), total_cost_usd=1.5, role="implementer")
    usage.add_usage(_usage(900, 5), total_cost_usd=9.0, role="analysis")
    usage.add_usage(_usage(50, 1), total_cost_usd=0.5, role="implementer")

    totals = usage.totals()
    by_role = totals["by_role"]
    assert sum(item["input_tokens"] for item in by_role.values()) == totals["input_tokens"]
    assert sum(item["output_tokens"] for item in by_role.values()) == totals["output_tokens"]
    assert sum(item["calls"] for item in by_role.values()) == totals["calls"]
    assert round(sum(item["total_cost_usd"] for item in by_role.values()), 6) == round(totals["total_cost_usd"], 6)
    assert by_role["implementer"]["calls"] == 2


def test_unnamed_calls_are_counted_not_dropped() -> None:
    """Silence about a role must not become silence about its spend."""
    usage = UsageAccumulator()
    usage.add_usage(_usage(7, 3), total_cost_usd=0.25)
    usage.add_usage(_usage(1, 1), total_cost_usd=0.25, role="   ")

    by_role = usage.totals()["by_role"]
    assert list(by_role) == [UsageAccumulator.UNATTRIBUTED]
    assert by_role[UsageAccumulator.UNATTRIBUTED]["input_tokens"] == 8
    assert by_role[UsageAccumulator.UNATTRIBUTED]["calls"] == 2


def test_breakdown_is_ordered_most_expensive_first() -> None:
    """A reader looking for the next cut should not have to sort by hand."""
    usage = UsageAccumulator()
    usage.add_usage(_usage(10, 1), total_cost_usd=0.10, role="cheap")
    usage.add_usage(_usage(10, 1), total_cost_usd=5.00, role="dear")
    usage.add_usage(_usage(10, 1), total_cost_usd=1.00, role="middling")

    assert list(usage.totals()["by_role"]) == ["dear", "middling", "cheap"]


def test_breakdown_falls_back_to_input_tokens_when_cost_is_missing() -> None:
    """Codex reports no price on many turns; the split must still rank."""
    usage = UsageAccumulator()
    usage.add_usage(_usage(10, 1), role="small")
    usage.add_usage(_usage(9000, 1), role="large")

    assert list(usage.totals()["by_role"]) == ["large", "small"]


def test_run_spec_carries_a_role_and_survives_resolution() -> None:
    """Attribution set at the call site has to reach the folding site."""
    spec = AgentRunSpec(system_prompt="", user_prompt="", cwd=".", role="implementer")
    assert spec.role == "implementer"
    assert AgentRunSpec(system_prompt="", user_prompt="", cwd=".").role == ""

    runtime = AgentRuntimeConfig(provider="claude", model="claude-opus-5")
    assert spec.resolved(runtime).role == "implementer"


def test_the_summary_reports_the_columns_the_money_is_in() -> None:
    """`in`/`out` alone hides ~72% of the bill, and every prompt-size change.

    cache_read and cache_creation are what a change to prompt size moves. A
    summary that omits them reports the same line whether the prefix fell 83%
    or not at all, which is the one question this line exists to answer.
    """
    usage = UsageAccumulator()
    usage.add_usage(
        {
            "input_tokens": 120,
            "output_tokens": 4_000,
            "cache_creation_input_tokens": 8_415,
            "cache_read_input_tokens": 640_000,
        },
        total_cost_usd=9.29,
        role="analysis",
    )

    lines = llm_spend_lines(usage.totals())

    assert "8,415 cache-write" in lines[0]
    assert "640,000 cache-read" in lines[0]
    assert lines[1].strip().startswith("analysis:")
    assert "8,415 cache-write" in lines[1]
    assert "640,000 cache-read" in lines[1]


def test_the_summary_totals_agree_with_the_split_it_prints() -> None:
    """The headline is the sum of the rows under it, cache columns included."""
    usage = UsageAccumulator()
    usage.add_usage(
        {"input_tokens": 1, "output_tokens": 2, "cache_creation_input_tokens": 300, "cache_read_input_tokens": 4_000},
        total_cost_usd=1.0,
        role="implementer",
    )
    usage.add_usage(
        {"input_tokens": 3, "output_tokens": 4, "cache_creation_input_tokens": 700, "cache_read_input_tokens": 6_000},
        total_cost_usd=2.0,
        role="analysis",
    )

    lines = llm_spend_lines(usage.totals())

    assert "1,000 cache-write" in lines[0]
    assert "10,000 cache-read" in lines[0]
    assert len(lines) == 3


def test_a_usage_record_without_cache_counters_still_renders() -> None:
    """A checkpoint from an older run must not raise while reporting a result."""
    lines = llm_spend_lines({"input_tokens": 5, "output_tokens": 6, "calls": 1})

    assert lines == ["  LLM spend: 5 in / 6 out / 0 cache-write / 0 cache-read tokens, cost unavailable (1 calls)"]
