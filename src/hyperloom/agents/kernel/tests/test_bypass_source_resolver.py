###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass op->source resolver.

Covers the editability filter, kernel-name demangling, the active-finder
delegation (native ``.cu``/``.hip`` via the symbol index), Triton ``.py`` def-line
pinning via AST, the trace ``kernel_file`` fast-path, and the repo-scan fallback.
There is no static ``op_to_source.json`` and no mapping-driven path.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _bypass_source_resolver as resolver  # noqa: E402
import source_resolver  # noqa: E402


def test_native_sources_are_editable():
    for p in ("/x/act.cu", "/x/attn.cuh", "/x/k.hip", "/x/decl.h"):
        assert resolver.is_editable_source(p) is True


def test_repo_triton_py_is_editable_but_generated_is_not():
    assert resolver.is_editable_source("/repo/aiter/ops/triton/fused.py") is True
    assert resolver.is_editable_source("/tmp/torchinductor_root/xx/cabc.py") is False
    assert resolver.is_editable_source("/repo/x.py", "triton_inductor_generated") is False
    assert resolver.is_editable_source("/x/notes.txt") is False
    assert resolver.is_editable_source("") is False


# --- active-finder delegation (native .cu/.hip via the live symbol index) -----


def test_resolve_source_without_symbol_is_unresolved():
    # The finder is symbol-driven: no device kernel name -> nothing to look up.
    assert resolver.resolve_source("_C::silu_and_mul", framework="vllm") == ("", "unresolved")
    assert resolver.resolve_source("", device_kernel_name="") == ("", "unresolved")


def test_resolve_source_delegates_to_active_finder(monkeypatch):
    calls = {}

    def fake_resolve_source(op_name, *, framework="", device_kernel_name=""):
        calls["args"] = (op_name, framework, device_kernel_name)
        return "/opt/vllm/csrc/act.cu", "symbol_index"

    monkeypatch.setattr(source_resolver, "resolve_source", fake_resolve_source)
    src, method = resolver.resolve_source("_C::silu_and_mul", framework="vllm", device_kernel_name="act_kernel")
    assert src == "/opt/vllm/csrc/act.cu"
    assert method == "symbol_index"
    assert calls["args"] == ("_C::silu_and_mul", "vllm", "act_kernel")


def test_resolve_source_finder_miss_is_unresolved(monkeypatch):
    monkeypatch.setattr(source_resolver, "resolve_source", lambda *a, **k: ("", "unresolved"))
    assert resolver.resolve_source("op::x", device_kernel_name="zzz") == ("", "unresolved")


def test_resolve_source_swallows_finder_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("index build failed")

    monkeypatch.setattr(source_resolver, "resolve_source", boom)
    assert resolver.resolve_source("op::x", device_kernel_name="k") == ("", "unresolved")


# --- Triton .py resolution (trace kernel_file + AST def-line pinning) ----------


def test_editable_trace_source_repo_py():
    assert resolver.editable_trace_source("/repo/aiter/triton/fused.py") == "/repo/aiter/triton/fused.py"


def test_editable_trace_source_rejects_generated_and_empty():
    assert resolver.editable_trace_source("/tmp/torchinductor_x/c.py") == ""
    assert resolver.editable_trace_source("") == ""


@pytest.fixture
def repo_dir():
    """A repo-like dir avoiding /tmp and the 'test' skip marker in its path."""
    base = Path(__file__).resolve().parents[2] / "_bypass_repo_scan_fixture" / "src"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base.parent, ignore_errors=True)


def test_resolve_triton_py_pins_def_line(repo_dir):
    py = repo_dir / "fused.py"
    py.write_text(
        "import triton\n\n@triton.jit\ndef my_fused_kernel(x):\n    return x\n",
        encoding="utf-8",
    )
    source, line, method = resolver.resolve_triton_py(str(py), symbol="my_fused_kernel")
    assert source == str(py)
    assert method == "trace_kernel_file_ast"
    assert py.read_text().splitlines()[line - 1].strip() == "def my_fused_kernel(x):"


def test_resolve_triton_py_launcher_form_parsed(repo_dir):
    py = repo_dir / "attn.py"
    py.write_text("import triton\n@triton.jit\ndef attn_kernel(x):\n    return x\n", encoding="utf-8")
    # Launcher form "<path>:<line>:<func>" must be split down to the bare .py.
    source, line, method = resolver.resolve_triton_py(f"{py}:2:attn_kernel")
    assert source == str(py)
    assert line == 3  # the def line, pinned via AST (not the launcher's :2)
    assert method == "trace_kernel_file_ast"


def test_resolve_triton_py_rejects_generated():
    src, line, method = resolver.resolve_triton_py("/tmp/torchinductor_x/c.py")
    assert (src, line, method) == ("", None, "unresolved")


def test_triton_def_line_single_unambiguous(repo_dir):
    py = repo_dir / "solo.py"
    py.write_text("import triton\n@triton.jit\ndef only_kernel(x):\n    return x\n", encoding="utf-8")
    assert resolver.triton_def_line(str(py)) == 3


def test_triton_def_line_require_name_match_skips_single_def_fallback(repo_dir):
    """require_name_match=True must not claim a file for an unrelated symbol."""
    py = repo_dir / "helper.py"
    py.write_text(
        "import triton\n@triton.jit\ndef _helper_kernel(x):\n    return x\n",
        encoding="utf-8",
    )
    # Without require_name_match the single-def fallback fires.
    assert resolver.triton_def_line(str(py), symbol="_absent_kernel") == 3
    # With require_name_match it must return None for an unrelated symbol.
    assert resolver.triton_def_line(str(py), symbol="_absent_kernel", require_name_match=True) is None


# --- kernel-name demangling ---------------------------------------------------


def test_demangle_itanium_nested():
    n = "_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataE"
    assert resolver._demangle_kernel_name(n) == "cross_device_reduce_2stage"


def test_demangle_plain_triton_name_passthrough():
    assert resolver._demangle_kernel_name("_fwd_grouped_kernel_stage1") == "_fwd_grouped_kernel_stage1"


def test_demangle_anonymous_namespace_template():
    n = "void (anonymous namespace)::kda_packed_decode_kernel<8, false>(x)"
    assert resolver._demangle_kernel_name(n) == "kda_packed_decode_kernel"


# --- repo-scan fallback (by demangled kernel name) ----------------------------


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


def test_repo_scan_disabled_by_env(monkeypatch, repo_dir):
    py = repo_dir / "fused.py"
    py.write_text("@triton.jit\ndef foo(x):\n    return x\n", encoding="utf-8")
    monkeypatch.setattr(resolver, "_build_repo_kernel_index", lambda: {"foo": str(py)})
    monkeypatch.setenv("HYPERLOOM_BYPASS_DISABLE_REPO_SCAN", "1")
    assert resolver.resolve_by_kernel_name("foo") == ("", "unresolved")


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
