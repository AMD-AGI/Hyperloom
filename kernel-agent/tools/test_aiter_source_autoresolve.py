# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Table-free aiter source resolution via aiter's OWN build registry.

The legacy ``_AITER_COMPILE_OPS_PROMOTIONS`` table only covered a fixed
allowlist of kernel families and silently failed for newer ones (e.g. the
``moe_cktile2stages`` MXFP4 MoE expert GEMM that dominates gpt-oss-120b).
``resolve_aiter_source_from_name`` instead derives the mapping from aiter
itself: ``op name -> @compile_ops module -> optCompilerConfig srcs -> .cu``.

These tests require the real ``aiter`` package (they assert against the
installed wheel's ``optCompilerConfig.json`` + ``aiter_meta/csrc`` layout)
and skip when it is unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Skip the whole module when aiter is not importable (CPU-only CI image).
pytest.importorskip("aiter")

_TLA_PATH = Path(__file__).resolve().parent / "tracelens_analysis.py"


@pytest.fixture(scope="module")
def tla() -> types.ModuleType:
    """Load tracelens_analysis.py without running its CLI bootstrap."""
    sys.path.insert(0, str(_TLA_PATH.parent))  # make tracelens_skill_runner importable
    spec = importlib.util.spec_from_file_location(
        "_tracelens_analysis_autoresolve", _TLA_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cktile2stages_resolves_to_device_cu(tla) -> None:
    """The kernel the static table predated must auto-resolve to its .cu."""
    for op in ("aiter::moe_cktile2stages_gemm1_ck", "moe_cktile2stages_gemm2_ck"):
        src = tla.resolve_aiter_source_from_name(op)
        assert src, f"{op} failed to resolve to a source file"
        assert src.endswith("moe_cktile2stages.cu"), src
        assert Path(src).is_file(), src
        assert "/pybind/" not in src.replace("\\", "/")  # device TU, not glue


def test_legacy_table_kernel_also_resolves_via_registry(tla) -> None:
    """A kernel that WAS in the static table (ck_moe_stage1) resolves via the
    registry too, proving the registry path is self-sufficient (table is now
    redundant for resolution)."""
    src = tla.resolve_aiter_source_from_name("aiter::ck_moe_stage1")
    assert src.endswith("gemm_moe_ck2stages.cu"), src
    assert Path(src).is_file(), src


def test_non_compile_ops_op_returns_empty(tla) -> None:
    """rmsnorm2d_fwd is a Triton wrapper, not a @compile_ops codegen kernel."""
    assert tla.resolve_aiter_source_from_name("aiter::rmsnorm2d_fwd") == ""


def test_non_aiter_op_returns_empty(tla) -> None:
    """Framework-native (non-aiter) ops are out of scope for this resolver."""
    assert tla.resolve_aiter_source_from_name("vllm::unified_attention_with_output") == ""
    assert tla.resolve_aiter_source_from_name("aten::mm") == ""
    assert tla.resolve_aiter_source_from_name("") == ""


def test_finalize_candidates_autoresolves_cktile2stages_from_empty_source(tla) -> None:
    """End-to-end: a cktile2stages candidate with NO source_file (as produced
    by a HIP-graph-captured trace) exits finalize with the device .cu resolved
    and marked as a reusable native kernel."""
    candidates = [{
        "name": "aiter::moe_cktile2stages_gemm1_ck",
        "duration_us": 5000.0,
        "call_count": 72,
        "source_file": "",  # HIP-graph capture carried no source_file
        "shapes": [[7211, 3072]],
    }]
    out = tla._finalize_candidates(candidates, total_dur=10000.0)[0]
    assert out["source_file"].endswith("moe_cktile2stages.cu"), out["source_file"]
    assert out["source_type"] == "hip_cpp"
    assert out["reusable_native_kernel"] is True
    # No pre-existing source => no bogus launcher_source_file recorded.
    assert "launcher_source_file" not in out


def test_finalize_candidates_promotes_cktile2stages_wrapper(tla) -> None:
    """When the trace named the aiter @compile_ops .py wrapper as the source,
    finalize promotes it to the device .cu and records the wrapper as the
    launcher_source_file (call-site context for the prompt)."""
    import aiter.ops.moe_op as moe_op  # the wrapper module
    wrapper = moe_op.__file__
    candidates = [{
        "name": "aiter::moe_cktile2stages_gemm2_ck",
        "duration_us": 4000.0,
        "call_count": 72,
        "source_file": wrapper,
        "shapes": [[7211, 3072]],
    }]
    out = tla._finalize_candidates(candidates, total_dur=10000.0)[0]
    assert out["source_file"].endswith("moe_cktile2stages.cu"), out["source_file"]
    assert out["launcher_source_file"] == wrapper
    assert out["source_promoted_from_launcher"] is True
