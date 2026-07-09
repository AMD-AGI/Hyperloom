###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the independent bypass op->source resolver.

Covers the editability filter, container selection, dispatch-kind matching, and
the trace ``kernel_file`` fast-path — all against a synthetic in-memory mapping
so no real ``op_to_source.json`` / on-disk sources are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _bypass_source_resolver as resolver  # noqa: E402


def _patch_mapping(monkeypatch, mapping):
    """Force the resolver to use ``mapping`` instead of the on-disk JSON."""
    monkeypatch.setattr(resolver, "_load_mapping", lambda: mapping)


# ── is_editable_source ───────────────────────────────────────────────────────


def test_native_sources_are_editable():
    for p in ("/x/act.cu", "/x/attn.cuh", "/x/k.hip", "/x/decl.h"):
        assert resolver.is_editable_source(p) is True


def test_repo_triton_py_is_editable_but_generated_is_not():
    assert resolver.is_editable_source("/repo/aiter/ops/triton/fused.py") is True
    assert resolver.is_editable_source("/tmp/torchinductor_root/xx/cabc.py") is False
    assert resolver.is_editable_source("/repo/x.py", "triton_inductor_generated") is False
    assert resolver.is_editable_source("/x/notes.txt") is False
    assert resolver.is_editable_source("") is False


# ── resolve_source: single / container selection ─────────────────────────────


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
                "a": {"kernel_source_path": "/s/a.cu", "patchable": False},  # not patchable
                "b": {"kernel_source_path": "/tmp/torchinductor_x/b.py", "patchable": True},  # generated
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
    # Neither exists on disk -> framework hint breaks the tie.
    assert resolver.resolve_source("op::z", framework="vllm")[0] == "/v/z.cu"
    assert resolver.resolve_source("op::z", framework="sglang")[0] == "/s/z.cu"


# ── resolve_source: dispatch kind ────────────────────────────────────────────


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
    # Unknown device kernel -> falls back to container selection (only editable src).
    src, _ = resolver.resolve_source("op::disp", framework="vllm", device_kernel_name="kernel_ZZZ")
    assert src == "/v/a.cu"


# ── editable_trace_source (Triton kernel_file fast-path) ─────────────────────


def test_editable_trace_source_repo_py():
    assert resolver.editable_trace_source("/repo/aiter/triton/fused.py") == "/repo/aiter/triton/fused.py"


def test_editable_trace_source_rejects_generated_and_empty():
    assert resolver.editable_trace_source("/tmp/torchinductor_x/c.py") == ""
    assert resolver.editable_trace_source("") == ""


def test_missing_json_yields_unresolved(monkeypatch):
    # A resolver whose data file is absent must degrade to unresolved, not crash.
    _patch_mapping(monkeypatch, {})
    assert resolver.resolve_source("anything", framework="vllm") == ("", "unresolved")
