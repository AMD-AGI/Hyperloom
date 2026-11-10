# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the GPU-preflight failure seen on Kimi-K3."""

from __future__ import annotations

from hyperloom.orchestrator.actions.executors._aiter_jit import is_aiter_jit_registry_mismatch


def test_compiled_registry_mismatch_is_not_a_cuda_graph_class():
    blob = "Exception: Capture cuda graph failed: kernel 'k' is not present in the compiled registry."
    assert is_aiter_jit_registry_mismatch(blob)
