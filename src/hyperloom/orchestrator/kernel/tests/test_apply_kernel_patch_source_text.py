# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The complete-source heuristic that gates a patch before it replaces its target."""

from __future__ import annotations

from hyperloom.orchestrator.kernel import apply_kernel_patch as apk


def test_source_text_looks_complete_python():
    assert apk._source_text_looks_complete("import torch\n", ".py") is True
    # No top-level marker -> rejected.
    assert apk._source_text_looks_complete("x = 1\n", ".py") is False
    # Syntax error -> rejected.
    assert apk._source_text_looks_complete("def (:\n", ".py") is False


def test_source_text_looks_complete_compiled_and_rejections():
    assert apk._source_text_looks_complete("#include <cuda.h>\n", ".cu") is True
    # Fenced text rejected regardless of suffix.
    assert apk._source_text_looks_complete("```\n#include <x>\n```", ".cu") is False
    # Empty rejected.
    assert apk._source_text_looks_complete("   ", ".cu") is False
    # Unknown suffix rejected.
    assert apk._source_text_looks_complete("void f(){}", ".txt") is False
    # Compiled suffix without any marker rejected.
    assert apk._source_text_looks_complete("just some prose", ".cpp") is False
