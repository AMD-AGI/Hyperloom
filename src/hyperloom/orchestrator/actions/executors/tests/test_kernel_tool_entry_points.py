# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every kernel tool the orchestrator starts with ``python -m`` is an importable module with a ``main``."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from hyperloom.orchestrator.actions.executors import trace_analyze
from hyperloom.orchestrator.kernel import request_handlers
from hyperloom.orchestrator.phases import kernel as kernel_phase

_TOOL_MODULES = (
    trace_analyze._TRACELENS_ANALYSIS_MODULE,
    trace_analyze._BYPASS_TRACE_ANALYSIS_MODULE,
    request_handlers._FORGE_GEMM_TUNING_MODULE,
    request_handlers._GEMM_TUNING_MODULE,
    request_handlers._FORGE_FUSION_MODULE,
    kernel_phase._GEAK_RUNNER_MODULE,
)


@pytest.mark.parametrize("module_name", _TOOL_MODULES)
def test_tool_module_resolves_to_a_callable_main(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None
    assert callable(importlib.import_module(module_name).main)
