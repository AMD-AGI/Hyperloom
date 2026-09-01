# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for shapes module."""

from kernelforge.gemm_tune.shapes import (
    compute_token_coverage,
    compute_dense_gemm_shapes,
    compute_vllm_moe_batch_sizes,
)


class TestComputeTokenCoverage:
    def test_explicit_override(self):
        result = compute_token_coverage(conc=256, explicit_tokens=[32, 64, 128])
        assert result == [32, 64, 128]

    def test_dedup_and_sort(self):
        result = compute_token_coverage(explicit_tokens=[128, 32, 32, 64])
        assert result == [32, 64, 128]

    def test_conc_64_default(self):
        result = compute_token_coverage(conc=64)
        assert 4 in result
        assert 64 in result
        assert 512 in result
        assert 768 not in result

    def test_conc_128_adds_high(self):
        result = compute_token_coverage(conc=128)
        assert 768 in result
        assert 1024 in result

    def test_conc_256_same_as_128(self):
        r128 = compute_token_coverage(conc=128)
        r256 = compute_token_coverage(conc=256)
        assert r128 == r256

    def test_conc_512_adds_very_high(self):
        result = compute_token_coverage(conc=512)
        assert 2048 in result
        assert 4096 in result


class TestComputeDenseGemmShapes:
    def test_basic_shapes(self):
        shapes = compute_dense_gemm_shapes(hidden_size=4096, intermediate_size=11008, tokens=[1, 64], tp=1)
        assert (1, 11008, 4096) in shapes  # gate/up
        assert (1, 4096, 11008) in shapes  # down
        assert (64, 11008, 4096) in shapes
        assert (64, 4096, 11008) in shapes

    def test_tp_splits_intermediate(self):
        shapes = compute_dense_gemm_shapes(hidden_size=4096, intermediate_size=11008, tokens=[1], tp=2)
        assert (1, 5504, 4096) in shapes  # inter/tp
        assert (1, 4096, 5504) in shapes


class TestComputeVllmMoeBatchSizes:
    def test_explicit_override(self):
        result = compute_vllm_moe_batch_sizes(explicit_tokens=[100, 200])
        assert result == [100, 200]

    def test_low_conc_caps(self):
        result = compute_vllm_moe_batch_sizes(conc=32)
        assert 8192 not in result
        assert max(result) <= 2048

    def test_high_conc_full(self):
        result = compute_vllm_moe_batch_sizes(conc=256)
        assert 8192 in result
