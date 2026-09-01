# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for dense GEMM (N,K) shape derivation across attention architectures.

The MLA and separate-head-dim shape sets are anchored to the actual GEMM shapes
recorded by GEAK tuning runs. The generic path must stay byte-identical to the
historical Llama formula.
"""

from __future__ import annotations

import pytest

from kernelforge.gemm_tune.dense_shapes import compute_dense_nk_shapes


def test_mla_deepseek_v3_matches_recorded_shapes():
    """DeepSeek-R1 (MLA) derives exactly the 6 recorded (N,K) at tp=8."""
    nk = set(
        compute_dense_nk_shapes(
            hidden_size=7168,
            intermediate_size=18432,
            num_heads=128,
            num_kv_heads=128,
            tp=8,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )
    )
    assert nk == {
        (2112, 7168),  # fused q_a + kv_a down-proj (replicated)
        (3072, 1536),  # q_b
        (4096, 512),  # kv_b
        (7168, 2048),  # o_proj
        (4608, 7168),  # dense FFN gate+up
        (7168, 2304),  # dense FFN down
    }


def test_deepseek_v4_sparse_mla_matches_runtime_shapes():
    """DeepSeek-V4-Flash attention GEMMs at tp=4 (no kv_lora_rank, no dense FFN)."""
    nk = set(
        compute_dense_nk_shapes(
            hidden_size=4096,
            intermediate_size=0,
            num_heads=64,
            num_kv_heads=1,
            tp=4,
            head_dim=512,
            q_lora_rank=1024,
            kv_lora_rank=0,
            o_lora_rank=1024,
            o_groups=8,
        )
    )
    assert nk == {
        (1536, 4096),  # fused_wqa_wkv (replicated)
        (8192, 1024),  # wq_b
    }


def test_separate_qk_v_head_dims_matches_recorded_shapes():
    """MiMo (GQA, qk head=192, v head=128) derives the 3 recorded (N,K) at tp=8."""
    nk = set(
        compute_dense_nk_shapes(
            hidden_size=6144,
            intermediate_size=16384,
            num_heads=128,
            num_kv_heads=8,
            tp=8,
            head_dim=192,
            v_head_dim=128,
        )
    )
    assert nk == {
        (3392, 6144),  # fused QKV (q:128*192, k:8*192, v:8*128) // 8
        (4096, 6144),  # dense FFN gate+up
        (6144, 2048),  # o_proj == dense FFN down (coincide)
    }


def test_generic_llama_is_unchanged():
    """With no extra dims the formula reduces to the historical QKV/O/gate/down."""

    def _old(h, i, nh, nkv, tp):
        raw = [
            ((nh + 2 * nkv) * (h // nh) // tp, h),
            (h, h // tp),
            (i * 2 // tp, h),
            (h, i // tp),
        ]
        seen, out = set(), []
        for x in raw:
            if x[0] > 0 and x[1] > 0 and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    for h, i, nh, nkv, tp in [
        (4096, 11008, 32, 32, 1),
        (4096, 14336, 32, 8, 2),
        (8192, 28672, 64, 8, 8),
    ]:
        assert compute_dense_nk_shapes(h, i, nh, nkv, tp) == _old(h, i, nh, nkv, tp)


# ── ISL-derived M-value capping (review: long-context ISL -> giant GEMM) ──────
from kernelforge.gemm_tune.dense_shapes import compute_dense_m_values


class TestDenseMValueIslCap:
    """A long-context ISL (e.g. ~32k) must not inject M=32k/65k giant GEMMs;
    ISL-derived M is capped at the same high-watermark as the other terms
    (8192 fast / 16384 thorough)."""

    def test_fast_mode_caps_isl_at_8192(self):
        m = compute_dense_m_values(conc=64, thorough=False, isl=32768)
        assert max(m) <= 8192
        assert 32768 not in m
        assert 8192 in m  # capped ISL still contributes the high-watermark

    def test_thorough_mode_caps_isl_and_double_at_16384(self):
        m = compute_dense_m_values(conc=64, thorough=True, isl=32768)
        assert max(m) <= 16384
        assert 32768 not in m
        assert 65536 not in m
        assert 16384 in m

    def test_moderate_isl_passes_through_uncapped(self):
        # ISL below the cap is used as-is (no spurious clamping).
        m = compute_dense_m_values(conc=64, thorough=False, isl=1024)
        assert 1024 in m

    def test_thorough_moderate_isl_and_double_present(self):
        m = compute_dense_m_values(conc=64, thorough=True, isl=1024)
        assert 1024 in m
        assert 2048 in m


# ── decode M band (review: must respect the concurrency cap) ─────────────────
from kernelforge.gemm_tune.dense_shapes import compute_decode_m_values


class TestDecodeMValues:
    """Decode M is the number of running requests, so it cannot exceed ``conc``.
    The cap itself is always present (steady-state decode sits there) and so is
    1 (ramp-up / tail)."""

    @pytest.mark.parametrize(
        ("conc", "expected"),
        [
            (1, [1]),
            (8, [1, 4, 8]),
            (64, [1, 4, 16, 32, 64]),
            (100, [1, 4, 16, 32, 64, 100]),
            (256, [1, 4, 16, 32, 64, 128, 256]),
        ],
    )
    def test_grid_is_clamped_to_concurrency(self, conc, expected):
        assert compute_decode_m_values(conc) == expected

    def test_never_exceeds_conc(self):
        for conc in (1, 2, 7, 8, 63, 64, 129, 256, 512):
            assert max(compute_decode_m_values(conc)) <= conc

    def test_degenerate_conc_falls_back_to_a_usable_grid(self):
        assert compute_decode_m_values(0) == compute_decode_m_values(64)
        assert compute_decode_m_values(-5) == [1]
