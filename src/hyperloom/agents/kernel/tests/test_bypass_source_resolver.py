###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the bypass op->source entry point.

Covers the editability filter, the trace ``kernel_file`` fast-path, the AST
Triton ``.py`` resolution (:func:`resolve_triton_py` / :func:`triton_def_line`),
and that ``resolve_source`` forwards native ``.cu`` lookups to the active finder
(and degrades to ``unresolved`` when the finder is unavailable). There is no
static ``op_to_source.json`` and no repo-scan fallback.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from hyperloom.agents.kernel.tools import _bypass_source_resolver as resolver
from hyperloom.agents.kernel.tools import source_resolver


@pytest.fixture
def repo_tmp():
    """A temp dir *outside* ``/tmp`` so ``is_editable_source`` accepts the path.

    ``is_editable_source`` treats any ``/tmp/`` path as generated Triton, which
    is where pytest's ``tmp_path`` lives -- so AST-fallback tests that need a
    path to survive the editability filter root their files under the tests dir.
    """
    d = tempfile.mkdtemp(prefix="hl_ast_", dir=os.path.dirname(__file__))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- is_editable_source ----------------------------------------------------
def test_native_sources_are_editable():
    for p in ("/x/act.cu", "/x/attn.cuh", "/x/k.hip", "/x/decl.h"):
        assert resolver.is_editable_source(p) is True


def test_repo_triton_py_is_editable_but_generated_is_not():
    assert resolver.is_editable_source("/repo/aiter/ops/triton/fused.py") is True
    assert resolver.is_editable_source("/tmp/torchinductor_root/xx/cabc.py") is False
    assert resolver.is_editable_source("/repo/x.py", "triton_inductor_generated") is False
    assert resolver.is_editable_source("/x/notes.txt") is False
    assert resolver.is_editable_source("") is False


# --- resolve_source delegates to the active finder -------------------------
def test_resolve_source_delegates_to_finder(monkeypatch):
    seen: dict[str, object] = {}

    def _fake(op_name, *, framework="", device_kernel_name=""):
        seen.update(op=op_name, framework=framework, dkn=device_kernel_name)
        return ("/opt/vllm/csrc/activation_kernels.cu", "symbol_index")

    monkeypatch.setattr(source_resolver, "resolve_source", _fake)
    src, method = resolver.resolve_source(
        "_C::silu_and_mul", framework="vllm", device_kernel_name="void vllm::act_and_mul_kernel<x>()"
    )
    assert src == "/opt/vllm/csrc/activation_kernels.cu"
    assert method == "symbol_index"
    assert seen == {
        "op": "_C::silu_and_mul",
        "framework": "vllm",
        "dkn": "void vllm::act_and_mul_kernel<x>()",
    }


def test_resolve_source_passes_through_unresolved(monkeypatch):
    monkeypatch.setattr(source_resolver, "resolve_source", lambda *a, **k: ("", "unresolved"))
    assert resolver.resolve_source("op::missing", framework="vllm") == ("", "unresolved")


def test_resolve_source_degrades_gracefully_on_finder_error(monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("index build failed")

    monkeypatch.setattr(source_resolver, "resolve_source", _boom)
    assert resolver.resolve_source("op::x", framework="vllm") == ("", "unresolved")


# --- editable_trace_source (trace kernel_file fast-path) --------------------
def test_editable_trace_source_repo_py():
    assert resolver.editable_trace_source("/repo/aiter/triton/fused.py") == "/repo/aiter/triton/fused.py"


def test_editable_trace_source_rejects_generated_and_empty():
    assert resolver.editable_trace_source("/tmp/torchinductor_x/c.py") == ""
    assert resolver.editable_trace_source("") == ""


# --- AST Triton .py resolution: triton_def_line ----------------------------
_TRITON_PY = '''\
import triton


@triton.jit
def _fwd_kernel(x_ptr, y_ptr, n):
    pass


@triton.autotune(configs=[], key=["n"])
@triton.jit
def _bwd_kernel(x_ptr, n):
    pass


def _host_helper(x):
    return x
'''


def _write(directory, name, text):
    p = os.path.join(str(directory), name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def test_triton_def_line_exact_func_name(tmp_path):
    path = _write(tmp_path, "k.py", _TRITON_PY)
    assert resolver.triton_def_line(path, func="_bwd_kernel") == 11


def test_triton_def_line_matches_normalized_symbol(tmp_path):
    path = _write(tmp_path, "k.py", _TRITON_PY)
    # A mangled device symbol (autotune/hash suffix) still maps to the jit def.
    assert resolver.triton_def_line(path, symbol="_fwd_kernel_0d1d2d3") == 5


def test_triton_def_line_prefers_jit_over_plain_def(tmp_path):
    single = '''\
import triton


def _fwd_kernel(x):
    return x


@triton.jit
def real_kernel(x_ptr, n):
    pass
'''
    path = _write(tmp_path, "k.py", single)
    # Sole @triton.jit def wins even though a plain def name collides with symbol.
    # (AST lineno points at the ``def`` line, not the decorator.)
    assert resolver.triton_def_line(path, symbol="_fwd_kernel") == 9


def test_triton_def_line_none_when_ambiguous_and_no_hint(tmp_path):
    path = _write(tmp_path, "k.py", _TRITON_PY)
    assert resolver.triton_def_line(path, symbol="totally_unrelated") is None


def test_triton_def_line_handles_unreadable_file(tmp_path):
    assert resolver.triton_def_line(str(tmp_path / "missing.py"), symbol="x") is None


# --- AST fallback: resolve_triton_py ---------------------------------------
def test_resolve_triton_py_pins_line_from_symbol(repo_tmp):
    path = _write(repo_tmp, "fused.py", _TRITON_PY)
    src, line, method = resolver.resolve_triton_py(path, symbol="_fwd_kernel_0d1d")
    assert src == path
    assert line == 5
    assert method == "trace_kernel_file_ast"


def test_resolve_triton_py_parses_launcher_form(repo_tmp):
    path = _write(repo_tmp, "fused.py", _TRITON_PY)
    # Launcher form "<path>:<line>:<func>" is reduced to the bare .py + AST line.
    src, line, method = resolver.resolve_triton_py(f"{path}:99:_bwd_kernel", symbol="")
    assert src == path
    assert line == 11
    assert method == "trace_kernel_file_ast"


def test_resolve_triton_py_path_only_when_line_unpinnable(repo_tmp):
    path = _write(repo_tmp, "fused.py", _TRITON_PY)
    src, line, method = resolver.resolve_triton_py(path, symbol="no_such_kernel")
    assert src == path
    assert line is None
    assert method == "trace_kernel_file"


def test_resolve_triton_py_rejects_generated_and_empty(tmp_path):
    assert resolver.resolve_triton_py("/tmp/torchinductor_x/c.py") == ("", None, "unresolved")
    assert resolver.resolve_triton_py("") == ("", None, "unresolved")


def test_parse_launcher_form_variants():
    assert resolver._parse_launcher_form("a/b.py") == ("a/b.py", None, "")
    assert resolver._parse_launcher_form("a/b.py:12:foo") == ("a/b.py", 12, "foo")
    assert resolver._parse_launcher_form("a/b.py(12): foo") == ("a/b.py", 12, "foo")
    assert resolver._parse_launcher_form("a/b.py#L7") == ("a/b.py", 7, "")
    assert resolver._parse_launcher_form("") == ("", None, "")
