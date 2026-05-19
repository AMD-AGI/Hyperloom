"""Tests for framework_agent.keywords.extract_keywords.

Hermetic - pure-Python, no network/GPU/disk.
"""

from __future__ import annotations

from framework_agent.keywords import extract_keywords, score_title_against_keywords


def test_extract_whitelist_hits_are_lowercase_and_sorted() -> None:
    """Whitelist hits should be returned sorted in lowercase."""
    out = extract_keywords("improve vLLM fp8 MoE attention on ROCm AMD MI300X")
    assert out == sorted(out)
    assert "vllm" in out
    assert "fp8" in out
    assert "moe" in out
    assert "attention" in out
    assert "rocm" in out


def test_extract_keeps_camelcase_identifiers() -> None:
    """Strict PascalCase identifiers (alternating cap+lower) survive and are lowercased.

    Note: the regex [A-Z][a-z]+(?:[A-Z][a-z]+)+ does NOT match identifiers
    that contain runs of multiple capital letters (e.g. ``AsyncLLMEngine``);
    those are deliberately filtered to avoid noise from acronyms.
    """
    out = extract_keywords("RadixCache and KvCache interact at AsyncEngine boundary")
    assert "radixcache" in out
    assert "kvcache" in out
    assert "asyncengine" in out


def test_extract_fallback_when_no_whitelist_match() -> None:
    """When no whitelist term hits, return first few 3+ letter words."""
    out = extract_keywords("plain english words only no special terms here")
    assert out, "fallback path must return something"
    assert all(len(w) >= 3 for w in out), "fallback should keep 3+ letter words"


def test_extract_empty_returns_empty() -> None:
    """Empty description should not crash; returns an empty list."""
    assert extract_keywords("") == []


def test_extract_dedupes_and_orders_stably() -> None:
    """Repeated terms collapse; output must be a sorted list (idempotent)."""
    out_a = extract_keywords("rocm rocm fp8 fp8 attention")
    out_b = extract_keywords("attention fp8 rocm")
    assert sorted(set(out_a)) == out_a
    assert out_a == out_b


# ---------------------------------------------------------------------------
# score_title_against_keywords (B2 rerank helper)
# ---------------------------------------------------------------------------


def test_score_title_counts_overlap() -> None:
    """Returns the number of distinct keyword tokens present in the title."""
    assert score_title_against_keywords("fp8 MoE perf", ["fp8", "moe"]) == 2


def test_score_title_case_insensitive() -> None:
    """Matching is lowercase-insensitive on both the title and the keywords."""
    assert score_title_against_keywords("FP8 MoE perf", ["FP8", "MOE"]) == 2


def test_score_title_zero_when_no_overlap() -> None:
    """No overlapping tokens -> 0 (used by the rank function to tail-sort)."""
    assert score_title_against_keywords("doc update", ["fp8", "moe"]) == 0


def test_score_title_handles_empty_inputs() -> None:
    """Empty title or empty keyword list never crashes; returns 0."""
    assert score_title_against_keywords("", ["fp8"]) == 0
    assert score_title_against_keywords("fp8 stuff", []) == 0


def test_score_title_snake_case_token() -> None:
    """snake_case tokens count as a single token by the regex token split."""
    assert score_title_against_keywords(
        "tensor_parallel optimisation", ["tensor_parallel"]
    ) == 1
