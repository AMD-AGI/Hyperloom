###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the shared source-resolution helper.

``_kernel_source`` is the one seam both the compute and bypass routes call.
:func:`resolve_source_verdict` owns the HL-side Triton-vs-native routing decision
(the only HL logic; path-finding itself is TraceLens'), and :func:`triton_def_line`
is the AST reading of a Triton kernel def that ``source_type_for`` relies on. The
field mapping of TraceLens' ``ResolveResult`` onto candidate keys, and the
non-patchable-with-source / fail-closed behaviors, are pinned end to end through
the reader in ``test_analysis_json_reader.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _kernel_source as ks
from TraceLens.TraceUtils.kernel_source import ResolveResult


def _spy(monkeypatch):
    """Capture the args ``resolve_source_verdict`` forwards to TraceLens."""
    seen: dict = {}

    def fake(kernel_name="", *, kernel_file="", is_triton=False, op_name="", search_paths=None):
        seen.update(kernel_name=kernel_name, kernel_file=kernel_file, is_triton=is_triton, op_name=op_name)
        return ResolveResult(location=None, patchable=False, method="unresolved")

    monkeypatch.setattr(ks, "resolve_kernel_source", fake)
    return seen


def test_triton_symbol_routes_with_launcher(monkeypatch):
    """A Triton SYMBOL takes the Triton route with its launcher; op_name rides along."""
    seen = _spy(monkeypatch)
    ks.resolve_source_verdict("triton_poi_fused_add", kernel_file="moe.py(10): fwd", op_name="aiter::gemm")
    assert seen["is_triton"] is True
    assert seen["kernel_file"] == "moe.py(10): fwd"
    assert seen["op_name"] == "aiter::gemm"


def test_triton_library_routes(monkeypatch):
    """A clean symbol with a triton library still routes Triton."""
    seen = _spy(monkeypatch)
    ks.resolve_source_verdict("some_kernel", kernel_file="k.py(3): k", library="TRITON")
    assert seen["is_triton"] is True
    assert seen["kernel_file"] == "k.py(3): k"


def test_triton_launcher_path_routes_even_when_symbol_is_clean(monkeypatch):
    """An aiter Gluon kernel names triton only in its ``.../triton/...`` launcher path."""
    seen = _spy(monkeypatch)
    ks.resolve_source_verdict(
        "paged_attention_decode_sliding_window",
        kernel_file="aiter/ops/triton/gluon/pa_decode_gluon.py(5194): pa_decode_gluon",
        library="AITER",
    )
    assert seen["is_triton"] is True
    assert seen["kernel_file"] == "aiter/ops/triton/gluon/pa_decode_gluon.py(5194): pa_decode_gluon"


def test_native_symbol_with_py_launcher_stays_native(monkeypatch):
    """A native kernel dispatched through a .py must NOT take the Triton route.

    Forcing Triton on a Tensile/CK symbol would bypass TraceLens' native gate and
    mislabel a precompiled kernel as patchable, so ``kernel_file`` is dropped.
    """
    seen = _spy(monkeypatch)
    ks.resolve_source_verdict("Cijk_Ailk_Bljk", kernel_file="tuned_gemm.py(9): g", library="aiter")
    assert seen["is_triton"] is False
    assert seen["kernel_file"] == ""


def test_triton_def_line_single_unambiguous(tmp_path):
    py = tmp_path / "solo.py"
    py.write_text("import triton\n@triton.jit\ndef only_kernel(x):\n    return x\n", encoding="utf-8")
    assert ks.triton_def_line(str(py)) == 3


def test_triton_def_line_matches_named_symbol(tmp_path):
    py = tmp_path / "fused.py"
    py.write_text(
        "import triton\n\n@triton.jit\ndef my_fused_kernel(x):\n    return x\n\n@triton.jit\ndef other(x):\n    return x\n",
        encoding="utf-8",
    )
    line = ks.triton_def_line(str(py), symbol="my_fused_kernel")
    assert py.read_text().splitlines()[line - 1].strip() == "def my_fused_kernel(x):"


def test_triton_def_line_require_name_match_skips_single_def_fallback(tmp_path):
    """require_name_match=True must not claim a file for an unrelated symbol."""
    py = tmp_path / "helper.py"
    py.write_text("import triton\n@triton.jit\ndef _helper_kernel(x):\n    return x\n", encoding="utf-8")
    # Without require_name_match the single-def fallback fires.
    assert ks.triton_def_line(str(py), symbol="_absent_kernel") == 3
    # With require_name_match it must return None for an unrelated symbol.
    assert ks.triton_def_line(str(py), symbol="_absent_kernel", require_name_match=True) is None


def test_triton_def_line_ignores_non_jit_defs(tmp_path):
    py = tmp_path / "mixed.py"
    py.write_text("def plain(x):\n    return x\n\n@triton.jit\ndef jitted(x):\n    return x\n", encoding="utf-8")
    assert ks.triton_def_line(str(py)) == 5


def test_triton_def_line_unparseable_returns_none(tmp_path):
    py = tmp_path / "broken.py"
    py.write_text("def (:::\n", encoding="utf-8")
    assert ks.triton_def_line(str(py)) is None
