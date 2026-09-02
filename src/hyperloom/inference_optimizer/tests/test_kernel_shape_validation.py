# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-opt dispatch shape-provenance / path validation.

A shapeless candidate dispatches: the backend's driver preparation recovers the
operand dims the trace never recorded. Provenance is validated only for a shape
that is present, and source/workspace paths must exist; ``dry_run`` bypasses all
of it.
"""

from __future__ import annotations

from pathlib import Path


def _candidate(**over):
    base = {
        "kernel_id": "k001",
        "name": "rmsnorm",
        "shapes": [{"input_shape": "(8,4096) bf16"}],
        "shape_provenance": "torch_trace",
    }
    base.update(over)
    return base


def test_finalize_candidates_stamps_trace_provenance():
    import importlib.util
    import sys

    tools_dir = Path("src/hyperloom/agents/kernel/tools").resolve()
    sys.path.insert(0, str(tools_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "tracelens_analysis_for_test",
            str(tools_dir / "tracelens_analysis.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so self-referential dataclass annotations resolve under py3.10.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(tools_dir))

    rows = [
        {"name": "with_shape", "duration_us": 10.0, "shapes": [{"a": 1}]},
        {"name": "no_shape", "duration_us": 5.0, "shapes": []},
    ]
    out = mod._finalize_candidates(rows)
    by_name = {r["name"]: r for r in out}
    assert by_name["with_shape"]["shape_provenance"] == "torch_trace"
    assert "shape_provenance" not in by_name["no_shape"]
