# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom_common.kernel_names import without_model_names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("_gemma_rmsnorm_kernel", "__rmsnorm_kernel"),
        ("_gemma_rotary_embedding_kernel", "__rotary_embedding_kernel"),
        ("_gemma3_rotary_kernel", "__rotary_kernel"),
        ("flydsl_gemma_rmsnorm", "flydsl__rmsnorm"),
        # CK and aiter spell GEMM types with a letter or digit after ``gemma``; they keep the spelling.
        ("ck_tile::blockgemmasmembsmemcregv1defaultpolicy", "ck_tile::blockgemmasmembsmemcregv1defaultpolicy"),
        ("gemma8w8", "gemma8w8"),
        ("batchedgemma8w8", "batchedgemma8w8"),
        ("_batched_gemm_a8w8", "_batched_gemm_a8w8"),
        ("bf16gemm_bf16_tn", "bf16gemm_bf16_tn"),
    ],
)
def test_only_the_model_family_token_is_removed(name, expected):
    assert without_model_names(name) == expected
