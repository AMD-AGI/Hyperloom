"""Tests for framework_agent.keywords.extract_keywords.

Hermetic - pure-Python, no network/GPU/disk.
"""

from __future__ import annotations

from framework_agent.keywords import (
    extract_keywords,
    score_title_against_keywords,
    score_title_with_anti_signal,
)


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


# ---------------------------------------------------------------------------
# atom framework keyword coverage (atom_plan/phase3_open_framework_agent 3.3)
# ---------------------------------------------------------------------------


def test_extract_atom_framework_token() -> None:
    """``atom`` must survive extract_keywords so PR scouting can rank
    ROCm/ATOM titles correctly. Listed alongside ``sglang``/``vllm``
    in the technical-term whitelist."""
    out = extract_keywords("improve atom fp8 moe throughput on mi300x")
    assert "atom" in out
    assert "fp8" in out
    assert "moe" in out
    assert "mi300x" in out


def test_extract_atom_specific_terms() -> None:
    """atom-flavoured PR titles often mention MTP / DP attention /
    kv_cache_dtype / torch_profiler_dir. Pinning these here keeps the
    primus_cortex search relevance on the atom-shaped axis instead of
    collapsing to generic moe / attention matches."""
    out = extract_keywords(
        "atom mtp dp_attention kv_cache_dtype fp8 torch_profiler_dir on mi355x"
    )
    assert "atom" in out
    assert "mtp" in out
    assert "dp_attention" in out
    assert "kv_cache_dtype" in out
    assert "torch_profiler_dir" in out
    assert "mi355x" in out


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


# ---------------------------------------------------------------------------
# score_title_with_anti_signal (B3 anti-correlation reranker).
#
# Anti pairs are gated on the gap keyword being present so the behaviour is
# strictly additive: a gap with no anti-trigger reduces to the same ordering
# as score_title_against_keywords. The tests below cover the four orthogonal
# axes that surfaced in session f219629b (dense vs MoE, AMD vs NVIDIA,
# bf16 vs low-bit), plus the trigger-gating, the zero-clamp floor, and the
# bug-driven PR:25769 regression.
# ---------------------------------------------------------------------------


def test_score_anti_dense_vs_moe_pr_demoted() -> None:
    """gap=['dense',...] should penalise a PR whose title screams MoE."""
    gap = ["sglang", "bf16", "dense", "throughput"]
    moe_title = "Enable MegaMoE for NextN with TP attn A2A scatter padding"
    dense_title = "optimize sglang dense attention prefill throughput"
    moe_score = score_title_with_anti_signal(moe_title, gap)
    dense_score = score_title_with_anti_signal(dense_title, gap)
    assert dense_score > moe_score, (
        f"dense PR must rank above MoE PR under dense gap "
        f"(got dense={dense_score} vs moe={moe_score})"
    )


def test_score_anti_mi300x_vs_nvidia_pr_demoted() -> None:
    """gap=['mi300x',...] should penalise NVIDIA-uarch PR titles."""
    gap = ["mi300x", "throughput", "sglang"]
    nv_title = "H100 fast moe kernel sm90 hopper"
    amd_title = "sglang mi300x throughput attention path"
    nv_score = score_title_with_anti_signal(nv_title, gap)
    amd_score = score_title_with_anti_signal(amd_title, gap)
    assert amd_score > nv_score


def test_score_anti_bf16_vs_low_bit_pr_demoted() -> None:
    """gap=['bf16',...] should penalise PRs targeting fp8/awq/gptq quant schemes."""
    gap = ["bf16", "throughput"]
    quant_title = "fp8 awq moe gptq throughput"
    bf16_title = "bf16 throughput improvement attention"
    quant_score = score_title_with_anti_signal(quant_title, gap)
    bf16_score = score_title_with_anti_signal(bf16_title, gap)
    assert bf16_score > quant_score


def test_score_anti_inactive_when_trigger_absent() -> None:
    """Anti is gated on the gap keyword being present.

    A gap of just ['throughput'] does NOT carry the ``dense`` trigger, so a
    MoE-heavy PR title must score the same as it would under the legacy
    positive-only scorer (no demotion fires).
    """
    gap = ["throughput"]
    moe_title = "MegaMoE throughput optimization"
    score = score_title_with_anti_signal(moe_title, gap)
    legacy = score_title_against_keywords(moe_title, gap)
    assert score == float(legacy), (
        f"anti must not activate without a trigger keyword in the gap "
        f"(got new={score} vs legacy={legacy})"
    )


def test_score_anti_clamps_at_zero_never_negative() -> None:
    """When anti penalty exceeds positive overlap, score is clamped to 0.0."""
    gap = ["dense", "mi300x", "bf16"]
    # Title overlaps once on dense-anti (moe), once on mi300x-anti (h100),
    # once on bf16-anti (fp8) but matches no gap keyword positively.
    title = "fp8 moe on h100"
    score = score_title_with_anti_signal(title, gap)
    assert score == 0.0, f"score must clamp to 0.0, not go negative (got {score})"


def test_score_anti_bidirectional_nvidia_gap_demotes_amd_pr() -> None:
    """Reverse direction: gap=['h100',...] should penalise AMD-only PR titles."""
    gap = ["h100", "throughput"]
    amd_title = "rocm mi300x cdna3 throughput"
    nv_title = "h100 throughput improvement"
    amd_score = score_title_with_anti_signal(amd_title, gap)
    nv_score = score_title_with_anti_signal(nv_title, gap)
    assert nv_score > amd_score


def test_score_anti_returns_float_and_handles_empty_inputs() -> None:
    """Type contract: float return; empty title or empty keywords yields 0.0."""
    assert score_title_with_anti_signal("", ["fp8"]) == 0.0
    assert score_title_with_anti_signal("fp8 stuff", []) == 0.0
    score = score_title_with_anti_signal("fp8 moe perf", ["fp8", "moe"])
    assert isinstance(score, float)
    assert score == 2.0


def test_score_anti_penalty_coefficient_tunable() -> None:
    """anti_penalty kwarg lets callers tune the demotion strength."""
    gap = ["dense", "throughput"]
    title = "MoE throughput improvement"  # +1 positive (throughput), +1 anti (moe)
    default_score = score_title_with_anti_signal(title, gap)  # 1 - 2*1 -> 0.0
    soft_score = score_title_with_anti_signal(title, gap, anti_penalty=0.5)  # 1-0.5 -> 0.5
    hard_score = score_title_with_anti_signal(title, gap, anti_penalty=5.0)  # 1-5 -> 0.0
    assert default_score == 0.0
    assert soft_score == 0.5
    assert hard_score == 0.0


def test_score_anti_pr25769_regression_session_f219629b() -> None:
    """Bug-driven: session f219629b on Qwen-Qwen3-32B (dense, bf16, mi300x).

    fa picked PR:25769 ("Enable MegaMoE for NextN with TP attn A2A scatter
    padding") because positive-only overlap matched ``throughput``. With the
    anti-signal fix, the MegaMoE PR must rank below a hypothetical dense /
    mi300x PR that targets the correct axis.
    """
    gap = ["sglang", "bf16", "dense", "mi300x", "throughput"]
    pr25769 = "Enable MegaMoE for NextN with TP attn A2A scatter padding"
    relevant = "optimize sglang bf16 attention prefill on mi300x"
    pr25769_score = score_title_with_anti_signal(pr25769, gap)
    relevant_score = score_title_with_anti_signal(relevant, gap)
    assert relevant_score > pr25769_score, (
        f"PR:25769-class MegaMoE PR must rank below dense+mi300x PR "
        f"(got pr25769={pr25769_score} vs relevant={relevant_score})"
    )
