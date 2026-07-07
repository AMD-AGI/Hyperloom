###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the shared canonical kernel-category display vocabulary."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _kernel_category as kc  # noqa: E402


def test_casing_normalized_across_routes():
    # The exact inconsistency observed on Qwen3-4B: bypass "Elementwise" vs
    # deterministic raw "gemm"/"elementwise" -> one canonical TitleCase label.
    assert kc.canonical_category("gemm") == "GEMM"
    assert kc.canonical_category("GEMM") == "GEMM"
    assert kc.canonical_category("elementwise") == "Elementwise"
    assert kc.canonical_category("Elementwise") == "Elementwise"


def test_marginal_vocabularies_unified():
    # bypass "Others" and GEAK "Other" collapse to one label.
    assert kc.canonical_category("Others") == "Other"
    assert kc.canonical_category("other") == "Other"
    # bypass "Normalization" and GEAK "LayerNorm"/rmsnorm collapse to one label.
    assert kc.canonical_category("Normalization") == "Normalization"
    assert kc.canonical_category("LayerNorm") == "Normalization"
    assert kc.canonical_category("rmsnorm") == "Normalization"
    # deterministic-only bucket keeps a stable canonical label.
    assert kc.canonical_category("reduce") == "Reduction"


def test_separators_and_attention_aliases():
    assert kc.canonical_category("groupedgemm_fwd") == "GEMM"
    assert kc.canonical_category("inferenceattention") == "SDPA"
    assert kc.canonical_category("sdpa-fwd") == "SDPA"
    assert kc.canonical_category("moe_fused") == "MoE"


def test_empty_and_unknown():
    assert kc.canonical_category(None) is None
    assert kc.canonical_category("") is None
    assert kc.canonical_category("   ") is None
    # unknown category surfaced verbatim (stripped), never silently dropped
    assert kc.canonical_category("SomethingNew") == "SomethingNew"
