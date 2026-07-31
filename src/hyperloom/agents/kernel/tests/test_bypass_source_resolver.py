###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the independent bypass op->source resolver.

Covers the editability filter, container selection, dispatch-kind matching, and
the trace ``kernel_file`` fast-path — all against a synthetic in-memory mapping
so no real ``op_to_source.json`` / on-disk sources are required.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _bypass_source_resolver as resolver  # noqa: E402


def _patch_mapping(monkeypatch, mapping):
    """Force the resolver to use ``mapping`` instead of the on-disk JSON."""
    monkeypatch.setattr(resolver, "_load_mapping", lambda: mapping)


def test_native_sources_are_editable():
    for p in ("/x/act.cu", "/x/attn.cuh", "/x/k.hip", "/x/decl.h"):
        assert resolver.is_editable_source(p) is True


def test_repo_triton_py_is_editable_but_generated_is_not():
    assert resolver.is_editable_source("/repo/aiter/ops/triton/fused.py") is True
    assert resolver.is_editable_source("/tmp/torchinductor_root/xx/cabc.py") is False
    assert resolver.is_editable_source("/repo/x.py", "triton_inductor_generated") is False
    assert resolver.is_editable_source("/x/notes.txt") is False
    assert resolver.is_editable_source("") is False


def test_resolve_single_native_source(monkeypatch):
    mapping = {
        "_C::silu_and_mul": {
            "kind": "single",
            "vllm": {
                "act_kernel": {
                    "kernel_source_path": "/opt/aiter/csrc/activation_kernels.cu",
                    "kernel_kind": "aiter_hip",
                    "patchable": True,
                }
            },
            "sglang": {},
        }
    }
    _patch_mapping(monkeypatch, mapping)
    src, method = resolver.resolve_source("_C::silu_and_mul", framework="vllm")
    assert src == "/opt/aiter/csrc/activation_kernels.cu"
    assert method == "op_to_source"


def test_resolve_strips_phase_suffix(monkeypatch):
    mapping = {
        "aten::mm": {
            "kind": "single",
            "sglang": {"g": {"kernel_source_path": "/s/gemm.cu", "patchable": True}},
        }
    }
    _patch_mapping(monkeypatch, mapping)
    src, method = resolver.resolve_source("aten::mm (decode)", framework="sglang")
    assert src == "/s/gemm.cu" and method == "op_to_source"


def test_resolve_skips_non_patchable_and_non_editable(monkeypatch):
    mapping = {
        "op::x": {
            "kind": "single",
            "sglang": {
                "a": {"kernel_source_path": "/s/a.cu", "patchable": False},
                "b": {"kernel_source_path": "/tmp/torchinductor_x/b.py", "patchable": True},
            },
        }
    }
    _patch_mapping(monkeypatch, mapping)
    src, method = resolver.resolve_source("op::x", framework="sglang")
    assert src == "" and method == "unresolved"


def test_resolve_miss_returns_unresolved(monkeypatch):
    _patch_mapping(monkeypatch, {"op::y": {"kind": "single", "sglang": {}}})
    assert resolver.resolve_source("op::not_present") == ("", "unresolved")
    assert resolver.resolve_source("") == ("", "unresolved")


def test_resolve_framework_hint_selects_container(monkeypatch):
    mapping = {
        "op::z": {
            "kind": "single",
            "vllm": {"v": {"kernel_source_path": "/v/z.cu", "patchable": True}},
            "sglang": {"s": {"kernel_source_path": "/s/z.cu", "patchable": True}},
        }
    }
    _patch_mapping(monkeypatch, mapping)
    assert resolver.resolve_source("op::z", framework="vllm")[0] == "/v/z.cu"
    assert resolver.resolve_source("op::z", framework="sglang")[0] == "/s/z.cu"


def test_resolve_dispatch_matches_device_kernel(monkeypatch):
    mapping = {
        "op::disp": {
            "kind": "dispatch",
            "vllm": {
                "kernel_A": {"kernel_source_path": "/v/a.cu", "patchable": True},
                "kernel_B": {"kernel_source_path": "/v/b.cu", "patchable": True},
            },
        }
    }
    _patch_mapping(monkeypatch, mapping)
    src, method = resolver.resolve_source("op::disp", framework="vllm", device_kernel_name="kernel_B")
    assert src == "/v/b.cu" and method == "op_to_source"


def test_resolve_dispatch_unknown_kernel_falls_back(monkeypatch):
    mapping = {
        "op::disp": {
            "kind": "dispatch",
            "vllm": {"kernel_A": {"kernel_source_path": "/v/a.cu", "patchable": True}},
        }
    }
    _patch_mapping(monkeypatch, mapping)
    src, _ = resolver.resolve_source("op::disp", framework="vllm", device_kernel_name="kernel_ZZZ")
    assert src == "/v/a.cu"


def test_editable_trace_source_repo_py():
    assert resolver.editable_trace_source("/repo/aiter/triton/fused.py") == "/repo/aiter/triton/fused.py"


def test_editable_trace_source_rejects_generated_and_empty():
    assert resolver.editable_trace_source("/tmp/torchinductor_x/c.py") == ""
    assert resolver.editable_trace_source("") == ""


def test_missing_json_yields_unresolved(monkeypatch):
    _patch_mapping(monkeypatch, {})
    assert resolver.resolve_source("anything", framework="vllm") == ("", "unresolved")


def test_demangle_itanium_nested():
    n = "_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataE"
    assert resolver._demangle_kernel_name(n) == "cross_device_reduce_2stage"


def test_demangle_plain_triton_name_passthrough():
    assert resolver._demangle_kernel_name("_fwd_grouped_kernel_stage1") == "_fwd_grouped_kernel_stage1"


def test_demangle_anonymous_namespace_template():
    n = "void (anonymous namespace)::kda_packed_decode_kernel<8, false>(x)"
    assert resolver._demangle_kernel_name(n) == "kda_packed_decode_kernel"


@pytest.fixture
def repo_dir():
    """A repo-like dir avoiding /tmp and the 'test' skip marker in its path."""
    base = Path(__file__).resolve().parents[2] / "_bypass_repo_scan_fixture" / "src"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base.parent, ignore_errors=True)


def test_resolve_by_kernel_name_triton_and_native(monkeypatch, repo_dir):
    py = repo_dir / "fused.py"
    py.write_text("@triton.jit\ndef foo(x):\n    return x\n", encoding="utf-8")
    cu = repo_dir / "kern.cu"
    cu.write_text("__global__ void bar(float* p) {}\n", encoding="utf-8")
    monkeypatch.setattr(
        resolver,
        "_build_repo_kernel_index",
        lambda: {"foo": str(py), "bar": str(cu), "gen": "/tmp/torchinductor_x/gen.py"},
    )
    assert resolver.resolve_by_kernel_name("foo") == (str(py), "repo_scan")
    assert resolver.resolve_by_kernel_name("bar") == (str(cu), "repo_scan")
    assert resolver.resolve_by_kernel_name("gen") == ("", "unresolved")
    assert resolver.resolve_by_kernel_name("missing") == ("", "unresolved")


def test_build_repo_kernel_index_scans_roots(monkeypatch, repo_dir):
    (repo_dir / "a.py").write_text("@triton.jit\ndef tri_k(x):\n    pass\n", encoding="utf-8")
    (repo_dir / "b.cu").write_text("__global__ void nat_k(int* p) {}\n", encoding="utf-8")
    monkeypatch.setattr(resolver, "_repo_scan_roots", lambda: (str(repo_dir),))
    resolver._build_repo_kernel_index.cache_clear()
    index = resolver._build_repo_kernel_index()
    resolver._build_repo_kernel_index.cache_clear()
    assert index["tri_k"] == str(repo_dir / "a.py")
    assert index["nat_k"] == str(repo_dir / "b.cu")


def test_repo_index_marks_duplicate_name_ambiguous(monkeypatch, repo_dir):
    # Same kernel name defined in two files: the index maps it to "" and
    # resolve_by_kernel_name refuses to guess (no arbitrary first-seen file).
    (repo_dir / "a.py").write_text("@triton.jit\ndef dup_k(x):\n    pass\n", encoding="utf-8")
    (repo_dir / "b.py").write_text("@triton.jit\ndef dup_k(x):\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(resolver, "_repo_scan_roots", lambda: (str(repo_dir),))
    resolver._build_repo_kernel_index.cache_clear()
    try:
        index = resolver._build_repo_kernel_index()
        assert index["dup_k"] == ""
        # resolve_by_kernel_name reads the same cached index and refuses to guess.
        assert resolver.resolve_by_kernel_name("dup_k") == ("", "unresolved")
    finally:
        resolver._build_repo_kernel_index.cache_clear()
