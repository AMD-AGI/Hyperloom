"""Unit tests for ``orchestrator.action_executors._framework_gap_composer``.

Covers the pure-Python ``compose_gap`` helper plus the kernel-breakdown
parser used to derive the bottleneck keyword.
"""

from __future__ import annotations

import json

import pytest

from inference_optimizer.orchestrator.action_executors._framework_gap_composer import (
    _extract_bottleneck_from_breakdown,
    _model_class_to_search_token,
    _normalize_model_class,
    compose_gap,
)


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------

class TestNormalizeModelClass:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("moe-mla", "moe_mla"),
            ("MoE+MLA", "moe_mla"),
            ("Dense ", "dense"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalisation(self, raw, expected):
        assert _normalize_model_class(raw) == expected


class TestModelClassSearchToken:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("dense", "dense"),
            ("moe_mla", "moe"),
            ("moe-swa", "moe"),
            ("hybrid", "hybrid"),
            ("", ""),
        ],
    )
    def test_token_map(self, raw, expected):
        assert _model_class_to_search_token(raw) == expected


# ---------------------------------------------------------------------------
# _extract_bottleneck_from_breakdown
# ---------------------------------------------------------------------------

class TestExtractBottleneck:
    def test_returns_empty_when_path_blank(self):
        assert _extract_bottleneck_from_breakdown("") == ""
        assert _extract_bottleneck_from_breakdown(None) == ""

    def test_returns_empty_when_file_missing(self, tmp_path):
        assert _extract_bottleneck_from_breakdown(tmp_path / "no.json") == ""

    def test_handles_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        assert _extract_bottleneck_from_breakdown(path) == ""

    def test_dict_with_top_kernels(self, tmp_path):
        path = tmp_path / "br.json"
        path.write_text(json.dumps({"top_kernels": [
            {"name": "attn_fwd_paged"},
            {"name": "gemm_a16w16"},
        ]}))
        assert _extract_bottleneck_from_breakdown(path) == "attention"

    def test_dict_with_rows_alias(self, tmp_path):
        path = tmp_path / "br.json"
        path.write_text(json.dumps({"rows": [
            {"kernel": "moe_ck_tile_fused"},
        ]}))
        assert _extract_bottleneck_from_breakdown(path) == "moe"

    def test_list_payload(self, tmp_path):
        path = tmp_path / "br.json"
        path.write_text(json.dumps([
            "rmsnorm_kernel", "softmax_kernel",
        ]))
        assert _extract_bottleneck_from_breakdown(path) == "norm"

    def test_no_match_returns_empty(self, tmp_path):
        path = tmp_path / "br.json"
        path.write_text(json.dumps([{"name": "totally_unknown_op"}]))
        assert _extract_bottleneck_from_breakdown(path) == ""


# ---------------------------------------------------------------------------
# compose_gap
# ---------------------------------------------------------------------------

class TestComposeGap:
    def test_minimal_inputs(self):
        gap, kw = compose_gap()
        # No tokens → gap is just "improve throughput".
        assert gap == "improve throughput"
        assert kw == []

    def test_full_inputs_with_bottleneck(self, tmp_path):
        breakdown = tmp_path / "br.json"
        breakdown.write_text(json.dumps({"top_kernels": [{"name": "flash_attn"}]}))
        gap, kw = compose_gap(
            framework="sglang",
            gpu_type="MI300X",
            model_class="moe_mla",
            precision="BF16",
            profile_kernel_breakdown_path=breakdown,
        )
        # Phrase ordering: improve <fw> <prec> <arch> <bottleneck> throughput on <gpu>.
        assert gap == "improve sglang bf16 moe attention throughput on mi300x"
        assert kw == sorted({"sglang", "mi300x", "moe", "bf16", "attention"})

    def test_missing_pieces_dropped(self):
        gap, kw = compose_gap(
            framework="vllm",
            gpu_type="",
            model_class="dense",
        )
        assert "throughput" in gap and "vllm" in gap and "dense" in gap
        assert "on " not in gap  # no gpu_type → no trailing 'on ...'
        assert sorted(kw) == sorted({"vllm", "dense"})

    def test_bottleneck_dedup_with_existing_token(self, tmp_path):
        # When bottleneck duplicates an arch/fw/prec token it must NOT be
        # appended again to the gap phrase.
        breakdown = tmp_path / "br.json"
        breakdown.write_text(json.dumps([{"name": "moe_ck_tile_fused"}]))
        gap, kw = compose_gap(
            framework="vllm",
            gpu_type="",
            model_class="moe_mla",
            precision="",
            profile_kernel_breakdown_path=breakdown,
        )
        assert gap == "improve vllm moe throughput"
        # `moe` appears once in keywords.
        assert kw.count("moe") == 1
