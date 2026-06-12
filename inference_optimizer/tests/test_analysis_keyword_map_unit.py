# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the analysis.md keyword -> variant mapping (N22)."""

from __future__ import annotations

from inference_optimizer.orchestrator import _analysis_keyword_map as akm


def test_normalize_lowercases_and_collapses_ws():
    assert akm._normalize("Foo   BAR\n\tBaz") == "foo bar baz"


def test_extract_empty_text():
    assert akm.extract_required_variants_from_analysis("", ["x"]) == ([], [])
    assert akm.extract_required_variants_from_analysis(None, ["x"]) == ([], [])


def test_extract_matches_narrowed_to_available():
    text = "The kernel uses torch.compile heavily and CUDA graphs."
    available = ["torch_compile_on", "cuda_graph_max_bs_8"]
    required, matches = akm.extract_required_variants_from_analysis(text, available)
    assert "torch_compile_on" in required
    assert "cuda_graph_max_bs_8" in required
    # Variants not in available are dropped.
    assert all(v in available for v in required)
    assert matches


def test_extract_no_available_drops_all():
    text = "torch.compile"
    required, matches = akm.extract_required_variants_from_analysis(text, [])
    assert required == []
    assert matches == []


def test_extract_no_keyword_match():
    required, matches = akm.extract_required_variants_from_analysis(
        "nothing relevant here", ["torch_compile_on"],
    )
    assert required == []
    assert matches == []


def test_format_advice_none_when_no_required():
    assert akm.format_missing_variants_advice([], [], [], action_name="propose") is None


def test_format_advice_none_when_all_included():
    out = akm.format_missing_variants_advice(
        ["torch_compile_on"], ["torch_compile_on"], [], action_name="propose",
    )
    assert out is None


def test_format_advice_with_missing_and_trigger_lines():
    matches = [("torch.compile", ("torch_compile_on",))]
    out = akm.format_missing_variants_advice(
        proposed_variants=[],
        required_variants=["torch_compile_on"],
        matches=matches,
        action_name="propose",
    )
    assert out is not None
    assert "N22 advisory" in out
    assert "torch_compile_on" in out
    assert "torch.compile" in out


def test_format_advice_body_fallback_without_matches():
    out = akm.format_missing_variants_advice(
        proposed_variants=[],
        required_variants=["torch_compile_on"],
        matches=[],
        action_name="propose",
    )
    assert "missing variants" in out
