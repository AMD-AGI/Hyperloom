# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``orchestrator.actions.executors._framework_gap_composer``."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.actions.executors._framework_gap_composer import (
    _model_class_to_search_token,
    _normalize_model_class,
    compose_gap,
)


# normalisation helpers


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


# compose_gap


class TestComposeGap:
    def test_minimal_inputs(self):
        gap, kw = compose_gap()
        assert gap == "improve throughput"
        assert kw == []

    def test_missing_pieces_dropped(self):
        gap, kw = compose_gap(
            framework="vllm",
            gpu_type="",
            model_class="dense",
        )
        assert "throughput" in gap and "vllm" in gap and "dense" in gap
        assert "on " not in gap
        assert sorted(kw) == sorted({"vllm", "dense"})
