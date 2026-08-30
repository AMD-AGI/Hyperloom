"""Tests for KB framework resolution + its fault tolerance (soft slug input)."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import _flydsl_rewrite  # noqa: E402
import forge_submit  # noqa: E402


def test_resolve_framework_from_explicit_source_owner_field():
    assert forge_submit._resolve_framework({"source_framework": "vLLM"}) == "vllm"
    assert forge_submit._resolve_framework({"source_framework": "aiter_meta"}) == "aiter"


def test_resolve_framework_ignores_serving_and_language_backends():
    assert forge_submit._resolve_framework({"backend": "triton"}) == ""
    assert forge_submit._resolve_framework({"framework": "vllm"}) == ""


def test_resolve_framework_from_kernel_path_when_candidate_silent():
    fw = forge_submit._resolve_framework({}, "/ws/worktree/vllm/model_executor/layers/fused_moe/x.py")
    assert fw == "vllm"


def test_resolve_framework_owning_package_not_deep_subdir():
    # Kernel lives directly in vllm; the owning package (shallowest) wins, not a
    # deep dir. Mirrors the arena side for the same operator.
    fw = forge_submit._resolve_framework({}, "/ws/worktree/vllm/v1/attention/ops/paged.py")
    assert fw == "vllm"


def test_resolve_framework_aiter_meta_maps_to_aiter():
    assert forge_submit._resolve_framework({}, "/ws/worktree/aiter_meta/csrc/gemm.cu") == "aiter"


def test_logical_operator_priority_and_namespace_normalization():
    assert (
        forge_submit._logical_operator(
            {
                "task_group": {"operator_identity": {"operation": "vllm :: unified_attention"}},
                "operation": "fallback_operation",
                "name": "fallback_name",
            }
        )
        == "vllm::unified_attention"
    )
    assert forge_submit._logical_operator({"operation": "aiter::fused_moe", "name": "fallback"}) == "aiter::fused_moe"
    assert (
        forge_submit._logical_operator({"operation": "aiter::Attention<ck::Tile<64, 128>>::forward"})
        == "aiter::Attention::forward"
    )
    assert forge_submit._logical_operator({"name": "direct_triton"}) == "direct_triton"


def test_logical_operator_is_stable_across_launch_attribution():
    """Both shapes of a trace name reduce to one identity.

    A candidate is named after the two rows it occupies, so the same kernel
    reads ``hipModuleLaunchKernel->_gqa_sparse_fwd_kernel`` in an analysis whose
    trace paired the launch call with the device row and ``_gqa_sparse_fwd_kernel``
    in one whose trace did not. One session here produced both, from two profiles
    of the same configuration. Forge keys its experience store on this name, so
    letting the launch call through writes two identities for one kernel and the
    warm-start read of either finds no prior record.
    """
    composite = {"name": "hipModuleLaunchKernel->_gqa_sparse_fwd_kernel"}
    bare = {"name": "_gqa_sparse_fwd_kernel"}
    assert forge_submit._logical_operator(composite) == forge_submit._logical_operator(bare) == "_gqa_sparse_fwd_kernel"
    # Graph-launched rows carry a different call and must not fork the identity.
    assert (
        forge_submit._logical_operator({"name": "hipGraphLaunch->_gqa_sparse_decode_kernel"})
        == "_gqa_sparse_decode_kernel"
    )
    # A namespaced operation has no launch call to strip and is left alone.
    assert (
        forge_submit._logical_operator({"operation": "vllm::unified_attention_with_output"})
        == "vllm::unified_attention_with_output"
    )


def test_resolve_framework_follows_kernel_sources_across_packages():
    # Cross-package indirection: the traced entry/anchor is a vLLM dispatch, but
    # the real kernel is defined in aiter (kernel_sources). Must resolve 'aiter'
    # to match the arena producer, not 'vllm' (the caller).
    candidate = {
        "kernel_sources": ["/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/unified.py"],
    }
    anchor = "/ws/worktree/vllm/attention/ops/entry.py"
    assert forge_submit._resolve_framework(candidate, anchor) == "aiter"


def test_resolve_framework_returns_empty_when_unknown():
    # Unresolvable -> "" so the caller OMITS --framework and forge-loop infers.
    # Fault tolerance: never raises, never guesses a wrong framework.
    assert forge_submit._resolve_framework({}, "/tmp/scratch/kernel.py") == ""
    assert forge_submit._resolve_framework(None, "") == ""
    assert forge_submit._resolve_framework({"framework": None, "backend": None}) == ""


def test_resolve_source_framework_explicit_beats_path():
    fw = forge_submit._resolve_framework(
        {"source_framework": "sglang"},
        "/x/vllm/y/k.py",
    )
    assert fw == "sglang"


def test_resolve_framework_uses_aiter_source_not_vllm_serving_wrapper():
    candidate = {
        "framework": "vllm",
        "backend": "vllm",
        "source_file": "/repo/vllm/attention.py",
        "kernel_sources": ["/repo/aiter/ops/triton/attention.py"],
    }
    assert forge_submit._resolve_framework(candidate, "/repo/vllm/attention.py") == "aiter"


def test_kernel_backend_resolution_uses_kernel_kind_for_ck_and_flydsl():
    assert forge_submit._resolve_kernel_backend("hip_cpp", "aiter_ck") == "ck"
    assert forge_submit._resolve_kernel_backend("python", "flydsl") == "flydsl"
    assert forge_submit._resolve_kernel_backend("flydsl", "") == "flydsl"


def test_direct_triton_uses_concrete_symbols_not_logical_operator(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "@triton.jit\ndef attention_kernel(x):\n    return x\n",
        encoding="utf-8",
    )
    candidate = {
        "operation": "vllm::logical_attention",
        "source_file": str(source),
        "device_kernel_names": [
            "attention_kernel_BLOCK_M_64...",
            "_ZN4impl24attention_kernel_specialILi64EEEvT_",
        ],
    }
    symbols = forge_submit._stable_implementation_symbols(
        candidate,
        source_files=[str(source)],
    )
    assert symbols == ["attention_kernel"]
    assert forge_submit._logical_operator(candidate) not in (symbols)


def test_gpu_target_normalization_extracts_canonical_gfx_arch():
    assert forge_submit._normalize_gpu_target("GFX942:sramecc+:xnack-") == "gfx942"
    assert forge_submit._normalize_gpu_target("MI355X") == "gfx950"


def test_rewrite_candidate_identity_reuses_the_shared_resolvers(tmp_path, monkeypatch):
    monkeypatch.setenv(_flydsl_rewrite.REWRITE_ENV, "1")
    _flydsl_rewrite.reset_capability_cache()
    workspace = tmp_path / "worktree"
    source = workspace / "aiter" / "ops" / "triton" / "attention.py"
    source.parent.mkdir(parents=True)
    source.write_text("@triton.jit\ndef attention_kernel(x):\n    return x\n", encoding="utf-8")
    candidate = {
        "framework": "vllm",
        "operation": "vllm :: unified_attention",
        "kernel_sources": [str(source)],
    }
    invocation_spec = tmp_path / "invocation_spec.json"
    invocation_spec.write_text("{}", encoding="utf-8")

    decision = _flydsl_rewrite.evaluate_rewrite_route(
        candidate=candidate,
        source_type="triton",
        kernel_kind="triton",
        logical_operator=forge_submit._logical_operator(candidate),
        source_kernel=str(source),
        workspace=str(workspace),
        implementation_sources=[str(source)],
        implementation_symbols=forge_submit._stable_implementation_symbols(
            candidate,
            source_files=[str(source)],
        ),
        framework=forge_submit._resolve_framework(candidate, str(source)),
        gpu_target=forge_submit._normalize_gpu_target("MI355X"),
        shape_cases=[{"M": 128}],
        shapes={"M": 128},
        branch="forge/session/attention-0011223344",
        attempt_id="attempt-7",
        timeout_s=7200,
        invocation_spec_file=str(invocation_spec),
        capability_probe=lambda **_kwargs: _flydsl_rewrite.RewriteCapabilities(
            True,
            "capability_ok",
            "",
            ("aiter", "sglang", "vllm"),
            source_languages=("triton", "hip", "cuda", "cpp"),
            source_kinds=("triton", "hip_cpp"),
            driver_preparation=True,
        ),
    )

    assert decision.eligible is True
    assert decision.spec.framework == "aiter"
    assert decision.spec.logical_operator == "vllm::unified_attention"
    assert decision.spec.implementation_symbols == ("attention_kernel",)
    assert decision.spec.gpu_target == "gfx950"
    # The producer owns the FlyDSL builder symbol; nothing here may re-derive it.
    assert "builder_symbol" not in decision.spec.as_dict()
