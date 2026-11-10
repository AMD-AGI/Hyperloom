# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake torch and aiter modules whose split-K GEMM entry points accept the splits production aiter accepts, shared by the split-K tests."""

from __future__ import annotations

import sys
import types
from unittest import mock


def _asm_accepts(k: int, split_k: int) -> bool:
    """The explicit-splitK alignment check of aiter's asm_a8w8_blockscale_bpreshuffle.cu."""
    sk = split_k or 1
    per_split = -(-(-(-k // sk)) // 128) * 128
    return -(-k // per_split) == sk


def fake_aiter(monkeypatch, k: int) -> list[tuple[str, int]]:
    """Install fake ``torch``/``aiter`` modules whose GEMM entry points mimic production dispatch for K=``k``."""
    calls: list[tuple[str, int]] = []

    def asm(x, w, out, xs, ws, bias=None, splitK=None, kernelName=None):
        calls.append(("asm", splitK))
        if not _asm_accepts(k, splitK):
            raise RuntimeError("gemm_a8w8_blockscale_bpreshuffle_asm failed: Kdim alignment check failed for splitK!")
        return out

    def ck(x, w, xs, ws, out, splitK=0):
        calls.append(("ck", splitK))
        return out

    aiter = types.ModuleType("aiter")
    aiter.dtypes = mock.MagicMock()
    aiter.gemm_a8w8_blockscale_bpreshuffle_asm = asm
    aiter.gemm_a8w8_blockscale_ck = ck
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "torch", mock.MagicMock())
    return calls
