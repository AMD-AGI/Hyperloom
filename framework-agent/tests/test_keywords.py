"""Tests for framework_agent.keywords.extract_keywords.

Hermetic - pure-Python, no network/GPU/disk.
"""

from __future__ import annotations

from framework_agent.keywords import extract_keywords


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
