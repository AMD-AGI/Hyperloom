# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test resolves aiter to a sandbox, never to the host install.

The baseline executor's serving-.so preflight and the kernel lane's cache invalidation move ``module_*.so`` files and
the whole ``jit/build`` aside. Run inside an image that ships aiter, a test that reaches them would rebuild the host's
compiled kernels for whatever runs there next.
"""

from __future__ import annotations

import importlib.util
import os

from hyperloom.common.aiter_jit_cache import resolve_serving_context
from hyperloom.orchestrator.actions.executors import _aiter_jit


def test_aiter_discovery_resolves_inside_the_test_sandbox(tmp_path_factory):
    sandbox = tmp_path_factory.getbasetemp()
    context = resolve_serving_context(
        os.environ.get("INFERENCE_OPTIMIZER_AITER_JIT_DIR"), jit_probe_paths=_aiter_jit.AITER_JIT_PROBE_PATHS
    )

    assert context is not None
    package, build = context
    assert package.is_relative_to(sandbox)
    assert build.is_relative_to(sandbox)


def test_no_host_aiter_path_is_reachable():
    spec = importlib.util.find_spec("aiter")

    assert spec is not None
    assert _aiter_jit.AITER_JIT_PROBE_PATHS == ()
    for name in ("AITER_JIT_DIR", "INFERENCE_OPTIMIZER_AITER_JIT_DIR", "VLLM_VENV_ROOT"):
        assert name not in os.environ
