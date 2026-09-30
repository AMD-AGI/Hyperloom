###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Contract tests for the shared source-resolution helper.

``_kernel_source`` is the one seam both the compute and bypass routes call.
:func:`resolve_source_verdict` forwards the symbol and its launcher to TraceLens
and returns the ``ResolveResult`` straight through; TraceLens owns the
native-vs-Triton routing decision. These tests pin that forwarding contract. The
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
    """Capture the exact kwargs ``resolve_source_verdict`` forwards to TraceLens."""
    seen: dict = {}

    def fake(kernel_name="", **kwargs):
        seen.update(symbol=kernel_name, **kwargs)
        return ResolveResult(location=None, patchable=False, method="unresolved")

    monkeypatch.setattr(ks, "resolve_kernel_source", fake)
    return seen


def test_forwards_symbol_kernel_file_and_op_name(monkeypatch):
    """The symbol, launcher, and op_name pass through verbatim, and nothing else."""
    seen = _spy(monkeypatch)
    ks.resolve_source_verdict("triton_poi_fused_add", kernel_file="moe.py(10): fwd", op_name="aiter::gemm")
    assert seen == {"symbol": "triton_poi_fused_add", "kernel_file": "moe.py(10): fwd", "op_name": "aiter::gemm"}
    assert "is_triton" not in seen
    assert "library" not in seen


def test_forwards_empty_launcher_without_guessing(monkeypatch):
    """A launcher-absent call forwards an empty ``kernel_file`` unchanged; HL adds no guess."""
    seen = _spy(monkeypatch)
    ks.resolve_source_verdict("Cijk_Ailk_Bljk", op_name="aiter::gemm")
    assert seen == {"symbol": "Cijk_Ailk_Bljk", "kernel_file": "", "op_name": "aiter::gemm"}
    assert "is_triton" not in seen
    assert "library" not in seen
