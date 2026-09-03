###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the version-robust v2 source resolver + its latency timer.

These build a tiny fake ``csrc`` tree on disk so resolution is exercised against
a real index without needing an installed vLLM/SGLang/aiter.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.agents.kernel.tools import (
    kernel_source_index,
    source_env,
    source_resolver,
)


def _fake_framework(tmp_path: Path) -> dict[str, source_env.FrameworkRoot]:
    """Create a fake vllm csrc tree with one native kernel definition."""
    csrc = tmp_path / "csrc"
    csrc.mkdir(parents=True)
    (csrc / "activation_kernels.cu").write_text(
        "#include <hip/hip_runtime.h>\n"
        "\n"
        "namespace vllm {\n"
        "__global__ void my_test_kernel(const float* a, float* b, int n) {\n"
        "  int i = threadIdx.x;\n"
        "  b[i] = a[i];\n"
        "}\n"
        "}  // namespace vllm\n",
        encoding="utf-8",
    )
    fr = source_env.FrameworkRoot(name="vllm", root=tmp_path, version="9.9.9", csrc_roots=(csrc,))
    return {"vllm": fr}


# --- is_editable_source (migrated from the deleted bypass source resolver) --
def test_native_sources_are_editable():
    for p in ("/x/a.cu", "/x/b.cuh", "/x/c.hip", "/x/d.h"):
        assert kernel_source_index.is_editable_source(p) is True


def test_repo_triton_py_is_editable_but_generated_is_not():
    assert kernel_source_index.is_editable_source("/repo/aiter/ops/triton/fused.py") is True
    assert kernel_source_index.is_editable_source("/tmp/torchinductor_root/xx/cabc.py") is False
    assert kernel_source_index.is_editable_source("/repo/x.py", "triton_inductor_generated") is False
    assert kernel_source_index.is_editable_source("/x/notes.txt") is False
    assert kernel_source_index.is_editable_source("") is False


# --- base_symbol -----------------------------------------------------------
def test_base_symbol_from_demangled():
    name = "void vllm::act_and_mul_kernel<c10::BFloat16, true>(c10::BFloat16*, c10::BFloat16 const*, int)"
    assert source_resolver.base_symbol(name) == "act_and_mul_kernel"


def test_base_symbol_from_plain_name():
    assert source_resolver.base_symbol("reshape_and_cache_kernel") == "reshape_and_cache_kernel"


def test_base_symbol_from_mangled_fallback(monkeypatch):
    # Force the pure-Python fallback (no c++filt) to prove it extracts the base.
    monkeypatch.setattr(source_resolver, "_cxxfilt_base", lambda _m: "")
    mangled = "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16ELb1EEEvPT_PKS3_i"
    assert source_resolver.base_symbol(mangled) == "act_and_mul_kernel"


# --- index + resolve -------------------------------------------------------
def test_build_index_finds_kernel(tmp_path):
    fw = _fake_framework(tmp_path)
    index = kernel_source_index.build_index(fw)
    recs = index.lookup("my_test_kernel")
    assert len(recs) == 1
    assert recs[0]["line"] == 4  # the __global__ def line
    assert index.build_ms >= 0.0


def test_resolve_symbol_first(tmp_path):
    fw = _fake_framework(tmp_path)
    index = kernel_source_index.build_index(fw)
    source_resolver.reset_latency()
    res = source_resolver.resolve(
        "some::op_not_in_hints",
        framework="vllm",
        device_kernel_name="void vllm::my_test_kernel<float>(float const*, float*, int)",
        index=index,
    )
    assert res.source_file.endswith("activation_kernels.cu")
    assert res.line == 4
    assert res.symbol == "my_test_kernel"
    assert res.patchable is True
    assert res.method == "symbol_index"


def test_resolve_unresolved_returns_reason(tmp_path):
    fw = _fake_framework(tmp_path)
    index = kernel_source_index.build_index(fw)
    res = source_resolver.resolve(
        "some::missing_op",
        framework="vllm",
        device_kernel_name="void vllm::does_not_exist_kernel<float>()",
        index=index,
    )
    assert res.source_file == ""
    assert res.method == "unresolved"
    assert res.reason


def test_non_patchable_ck_symbol_is_flagged(tmp_path):
    # CK template instantiations are detected from the symbol alone (no JSON):
    # they have no single editable __global__ source, so the gate bails early.
    fw = _fake_framework(tmp_path)
    index = kernel_source_index.build_index(fw)
    res = source_resolver.resolve(
        "ck::op",
        framework="vllm",
        device_kernel_name="ck_tile::GemmKernel<ck_tile::fp16_t>(void*)",
        index=index,
    )
    assert res.patchable is False
    assert res.method == "non_patchable"
    assert res.reason == "aiter_ck"


def test_ck_marker_is_namespace_boundary_not_substring():
    # Names that merely end in "ck" must NOT be gated as CK (regression: the
    # unbounded "ck::" substring dropped unrelated kernels silently).
    for sym in (
        "void vllm::block::my_kernel(float*)",
        "void unpack::helper_kernel(int*)",
        "void flashck::fmha_kernel(void*)",
    ):
        assert source_resolver._non_patchable_kind(sym) == "", sym
    # Real CK namespaces are still detected.
    assert source_resolver._non_patchable_kind("void ck::gemm(float*)") == "aiter_ck"
    assert source_resolver._non_patchable_kind("ck_tile::Gemm<half>(void*)") == "aiter_ck"


def test_ck_detection_falls_back_to_mangled_without_cxxfilt(monkeypatch):
    # With c++filt unavailable, a mangled CK symbol must still be classified from
    # its length-prefixed namespace, so the verdict does not depend on binutils.
    monkeypatch.setattr(source_resolver, "_cxxfilt_base", lambda _m: "")
    assert source_resolver._non_patchable_kind("_ZN2ck15kernel_moe_gemmIiEEvPf") == "aiter_ck"
    # A non-CK mangled symbol is not misclassified by the fallback.
    assert source_resolver._non_patchable_kind("_ZN4vllm11some_kernelIiEEvPf") == ""


def test_legacy_tuple_preserves_non_patchable(tmp_path):
    fw = _fake_framework(tmp_path)
    index = kernel_source_index.build_index(fw)
    res = source_resolver.resolve(
        "ck::op", framework="vllm", device_kernel_name="ck_tile::Gemm<half>(void*)", index=index
    )
    # non_patchable carries an empty source_file but must NOT collapse to
    # "unresolved" -- callers distinguish "known not rewritable" from "not found".
    assert res.as_legacy_tuple() == ("", "non_patchable")


def test_header_declaration_not_selected_over_definition(tmp_path):
    # A header holding only a forward declaration must not outrank the .cu that
    # actually defines the kernel (regression: shorter header path won on ties).
    csrc = tmp_path / "csrc"
    (csrc / "include").mkdir(parents=True)
    (csrc / "kernels").mkdir(parents=True)
    (csrc / "include" / "ops.h").write_text(
        "namespace vllm {\n__global__ void paged_attn_kernel(const float*, float*);\n}\n",
        encoding="utf-8",
    )
    (csrc / "kernels" / "paged_attention.cu").write_text(
        "namespace vllm {\n__global__ void paged_attn_kernel(const float* q, float* o) { o[0]=q[0]; }\n}\n",
        encoding="utf-8",
    )
    fw = {"vllm": source_env.FrameworkRoot("vllm", tmp_path, "1.0", (csrc,))}
    index = kernel_source_index.build_index(fw)
    # Only the definition is indexed, so the declaration header can never be picked.
    assert index.lookup("paged_attn_kernel") and all(
        rec["file"].endswith("paged_attention.cu") for rec in index.lookup("paged_attn_kernel")
    )
    res = source_resolver.resolve(
        "x::y",
        framework="vllm",
        device_kernel_name="void vllm::paged_attn_kernel(float const*, float*)",
        index=index,
    )
    assert res.source_file.endswith("paged_attention.cu")


# --- latency timer ---------------------------------------------------------
def test_latency_report_records_samples(tmp_path):
    fw = _fake_framework(tmp_path)
    index = kernel_source_index.build_index(fw)
    source_resolver.reset_latency()
    for _ in range(3):
        source_resolver.resolve(
            "op",
            framework="vllm",
            device_kernel_name="void vllm::my_test_kernel<float>()",
            index=index,
        )
    report = source_resolver.latency_report()
    assert index.version_tag in report
    stats = report[index.version_tag]
    assert stats["count"] == 3
    assert "avg_ms" in stats and stats["avg_ms"] >= 0.0


def test_legacy_tuple_shape(tmp_path):
    fw = _fake_framework(tmp_path)
    index = kernel_source_index.build_index(fw)
    res = source_resolver.resolve(
        "op", framework="vllm", device_kernel_name="void vllm::my_test_kernel<float>()", index=index
    )
    src, method = res.as_legacy_tuple()
    assert src.endswith("activation_kernels.cu")
    assert method == "symbol_index"


# --- env fingerprint -------------------------------------------------------
def test_fingerprint_is_stable(tmp_path):
    fw = _fake_framework(tmp_path)
    assert source_env.fingerprint(fw) == source_env.fingerprint(fw)


def test_fingerprint_detects_nested_edit(tmp_path):
    # A modification to a file in a NESTED subdir must change the fingerprint
    # (regression: the old signature used only the root's mtime + child count and
    # missed nested edits, so GEAK's own .cu rewrites reused a stale index).
    import time

    csrc = tmp_path / "csrc"
    sub = csrc / "sub"
    sub.mkdir(parents=True)
    k = sub / "k.cu"
    k.write_text("__global__ void a() {}\n", encoding="utf-8")
    fw = {"x": source_env.FrameworkRoot("x", tmp_path, "1.0", (csrc,))}
    before = source_env.fingerprint(fw)
    time.sleep(0.01)
    k.write_text("__global__ void a() {}\n__global__ void b() {}\n", encoding="utf-8")
    assert source_env.fingerprint(fw) != before
