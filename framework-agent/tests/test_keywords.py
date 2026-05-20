"""Tests for framework_agent.keywords.extract_keywords.

Hermetic - pure-Python, no network/GPU/disk.
"""

from __future__ import annotations

from framework_agent.keywords import extract_keywords, score_title_against_keywords


def test_extract_whitelist_hits_are_lowercase_and_sorted() -> None:
    """Whitelist hits should be returned sorted in lowercase.

    ``mi300x`` is asserted explicitly to guard the GPU-hardware codename
    extension (without it the primus_cortex search drops the hardware
    constraint and picks unrelated NVIDIA PRs; see fa-keywords-hardware
    fix rationale in _TECHNICAL_TERMS).
    """
    out = extract_keywords("improve vLLM fp8 MoE attention on ROCm AMD MI300X")
    assert out == sorted(out)
    assert "vllm" in out
    assert "fp8" in out
    assert "moe" in out
    assert "attention" in out
    assert "rocm" in out
    assert "mi300x" in out


# ---------------------------------------------------------------------------
# GPU hardware codename coverage (regression guard for the relevance bug
# where ``mi300x`` / ``gfx942`` / ``sm90`` etc. fell through the whitelist
# and Primus search lost the hardware dimension).
# ---------------------------------------------------------------------------


def test_extract_amd_cdna_codenames() -> None:
    """AMD CDNA accelerator codenames must survive extract_keywords()."""
    out = extract_keywords(
        "improve sglang bf16 throughput on mi300x; also gfx942 cdna3"
    )
    assert "mi300x" in out
    assert "gfx942" in out
    assert "cdna3" in out
    # baseline sanity: existing whitelist terms still recognised
    assert "sglang" in out
    assert "bf16" in out


def test_extract_nvidia_codenames() -> None:
    """NVIDIA Ampere/Hopper/Blackwell codenames must survive extract_keywords()."""
    out = extract_keywords(
        "port mega moe to sm90 hopper h100 from sm80 ampere a100"
    )
    assert "sm90" in out
    assert "sm80" in out
    assert "hopper" in out
    assert "ampere" in out
    assert "h100" in out
    assert "a100" in out
    assert "moe" in out  # existing whitelist term preserved


def test_extract_realistic_io_framework_gap() -> None:
    """End-to-end check on the actual IO ``--framework-gap`` template.

    Mirrors the gap inference_optimizer's SKILL.md Launch template renders
    by default (``improve {fw} {prec} {model_class} throughput on {gpu}``).
    Before the hardware-codename extension, ``mi300x`` was silently dropped,
    which caused the primus_cortex search query to collapse to
    ``"bf16 sglang"`` and surface NVIDIA SM90 PRs as the winner.
    """
    out = extract_keywords("improve sglang bf16 dense throughput on mi300x")
    # All four salient dimensions must be present.
    assert "sglang" in out, "framework token must be kept"
    assert "bf16" in out, "precision token must be kept"
    assert "mi300x" in out, "hardware codename must be kept"


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
