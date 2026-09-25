###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the shared source-resolution helper.

``_kernel_source`` is the one seam both the compute and bypass routes call.
:func:`resolve_source_verdict` owns the HL-side Triton-vs-native routing decision
(the only HL logic; path-finding itself is TraceLens'). The field mapping of
TraceLens' ``ResolveResult`` onto candidate keys, and the non-patchable-with-source
/ fail-closed behaviors, are pinned end to end through the reader in
``test_analysis_json_reader.py``.
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
