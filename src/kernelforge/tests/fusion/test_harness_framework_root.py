# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The harness is authored in the framework tree and run from the output dir.

Anything the author derived from ``__file__`` therefore points at the wrong
place by the time the loop scores it, so the runner names the tree outright.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.fusion.models import Recipe
from kernelforge.fusion.validate import HarnessKernelRunner

# Reports the framework root it was given, so the test can assert what reached it.
HARNESS = """
import json, os
print(json.dumps({
    "compiled": True,
    "is_triton": False,
    "error": "",
    "parity": [],
    "seen_root": os.environ.get("FORGE_FUSION_FRAMEWORK_ROOT", ""),
}))
"""


def _recipe() -> Recipe:
    return Recipe(
        pattern_id="llm:x",
        description="d",
        env_flag="X_FUSED",
        source_file="",
        source_hints=[],
        fusion_math="",
        eager_reference_hint="",
        shapes={},
        matched_categories=[],
        trigger_share=1.0,
    )


def test_framework_root_reaches_the_harness(tmp_path: Path) -> None:
    tree = tmp_path / "fwroot"
    tree.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    harness = out / "kernel_harness.py"
    harness.write_text(HARNESS, encoding="utf-8")

    runner = HarnessKernelRunner(harness_path=str(harness), workdir=str(tree), framework_root=str(tree))
    runner.compile_check(_recipe())

    assert json.loads(json.dumps(runner._cache))["seen_root"] == str(tree)


def test_absent_framework_root_leaves_the_variable_unset(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    harness = out / "kernel_harness.py"
    harness.write_text(HARNESS, encoding="utf-8")

    runner = HarnessKernelRunner(harness_path=str(harness), workdir=str(tmp_path))
    runner.compile_check(_recipe())

    assert runner._cache["seen_root"] == ""


def test_missing_harness_is_reported_not_raised(tmp_path: Path) -> None:
    runner = HarnessKernelRunner(harness_path=str(tmp_path / "nope.py"), workdir=str(tmp_path))

    outcome = runner.compile_check(_recipe())

    assert outcome.ok is False
    assert "not found" in outcome.error
