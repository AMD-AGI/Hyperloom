"""Tests for KB framework resolution + its fault tolerance (soft slug input)."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


def test_resolve_framework_from_explicit_candidate_field():
    assert forge_submit._resolve_framework({"framework": "vLLM"}) == "vllm"
    assert forge_submit._resolve_framework({"backend": "sglang"}) == "sglang"


def test_resolve_framework_ignores_language_backends():
    # 'triton' is a backend LANGUAGE, not a framework — must not become framework.
    assert forge_submit._resolve_framework({"backend": "triton"}) == ""
    assert forge_submit._resolve_framework({"framework": "triton"}) == ""


def test_resolve_framework_from_kernel_path_when_candidate_silent():
    fw = forge_submit._resolve_framework(
        {}, "/ws/worktree/vllm/model_executor/layers/fused_moe/x.py")
    assert fw == "vllm"


def test_resolve_framework_owning_package_not_deep_subdir():
    # Kernel lives directly in vllm; the owning package (shallowest) wins, not a
    # deep dir. Mirrors the arena side for the same operator.
    fw = forge_submit._resolve_framework(
        {}, "/ws/worktree/vllm/v1/attention/ops/paged.py")
    assert fw == "vllm"


def test_resolve_framework_aiter_meta_maps_to_aiter():
    assert forge_submit._resolve_framework({"framework": "aiter_meta"}) == "aiter"
    assert forge_submit._resolve_framework(
        {}, "/ws/worktree/aiter_meta/csrc/gemm.cu") == "aiter"


def test_op_name_for_target_functions_strips_backend_prefix():
    # The specific operation identity (fed as --target-functions) becomes the KB
    # slug op when the anchor is a wrapper; strip the ``backend::`` prefix.
    assert forge_submit._op_name_for_target_functions(
        {"name": "vllm::unified_attention_with_output"}) == "unified_attention_with_output"
    assert forge_submit._op_name_for_target_functions(
        {"operation": "fused_moe"}) == "fused_moe"
    # operation preferred over name
    assert forge_submit._op_name_for_target_functions(
        {"operation": "gemm_a8w8", "name": "vllm::rocm_unquantized_gemm"}) == "gemm_a8w8"
    assert forge_submit._op_name_for_target_functions({}) == ""
    assert forge_submit._op_name_for_target_functions(None) == ""


def test_resolve_framework_follows_kernel_sources_across_packages():
    # Cross-package indirection: the traced entry/anchor is a vLLM dispatch, but
    # the real kernel is defined in aiter (kernel_sources). Must resolve 'aiter'
    # to match the arena producer, not 'vllm' (the caller).
    candidate = {
        "kernel_sources": [
            "/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/unified.py"
        ],
    }
    anchor = "/ws/worktree/vllm/attention/ops/entry.py"
    assert forge_submit._resolve_framework(candidate, anchor) == "aiter"


def test_resolve_framework_returns_empty_when_unknown():
    # Unresolvable -> "" so the caller OMITS --framework and forge-loop infers.
    # Fault tolerance: never raises, never guesses a wrong framework.
    assert forge_submit._resolve_framework({}, "/tmp/scratch/kernel.py") == ""
    assert forge_submit._resolve_framework(None, "") == ""
    assert forge_submit._resolve_framework({"framework": None, "backend": None}) == ""


def test_resolve_framework_explicit_beats_path():
    # An explicit recognized framework wins over a (possibly flattened) path.
    fw = forge_submit._resolve_framework({"framework": "sglang"}, "/x/vllm/y/k.py")
    assert fw == "sglang"
