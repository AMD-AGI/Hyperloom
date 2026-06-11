# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR-K — Prompt rendering carries the source-promotion notice.

Promoted candidates get a ``SOURCE ATTRIBUTION NOTE`` block + launcher
fields in metadata; un-promoted candidates carry neither.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_optimization as ko  # noqa: E402


def _build_args(**overrides) -> Namespace:
    base = {
        "source_file": "",
        "kernel_id": "k001",
        "num_gpus": 1,
        "budget_minutes": 60,
        "target_platform": "MI355X",
        "geak_cost_limit": 5.0,
        "oob_max_turns": 8,
        "dry_run": False,
        "extra_server_args": "",
    }
    base.update(overrides)
    return Namespace(**base)


def _candidate(**overrides) -> dict:
    base = {
        "name": "aiter::ck_moe_stage1",
        "kernel_id": "k001",
        "source_file": "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu",
        "source_type": "hip_cpp",
        "kernel_repo": "/sgl-workspace/aiter",
        "gpu_pct": 25.0,
        "shapes": [[128, 4096]],
        "duration_us": 1234.5,
        "call_count": 100,
        "is_multigpu": False,
        "num_gpus_recommended": 1,
        "benchmark_files": [],
    }
    base.update(overrides)
    return base


def test_build_prompt_renders_promotion_block_for_promoted_candidate() -> None:
    """Promoted candidate prompt contains the SOURCE ATTRIBUTION NOTE block with both paths and three hard rules."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        source_promoted_from_launcher=True,
    )
    args = _build_args()
    prompt = ko.build_prompt(cand, args)

    assert "SOURCE ATTRIBUTION NOTE" in prompt
    assert "/sgl-workspace/aiter/aiter/ops/moe_op.py" in prompt
    assert "/sgl-workspace/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu" in prompt
    # All three hard rules must be present.
    assert "1. DO NOT modify the Python wrapper" in prompt
    assert "2. The device source may be a CODEGEN ENTRY" in prompt
    assert "3. Preserve function names, signatures" in prompt
    # Notice calls out the bypass mechanism (why wrapper patches don't work).
    assert "@compile_ops" in prompt
    assert "ZERO runtime effect" in prompt


def test_build_prompt_no_promotion_block_for_un_promoted_candidate() -> None:
    """Kernels traced to the right file on the first pass must NOT carry the notice."""
    cand = _candidate(
        # No launcher_source_file, no source_promoted_from_launcher flag.
    )
    args = _build_args()
    prompt = ko.build_prompt(cand, args)
    assert "SOURCE ATTRIBUTION NOTE" not in prompt
    assert "@compile_ops" not in prompt


def test_build_prompt_no_promotion_block_when_flag_false_with_launcher() -> None:
    """Defensive: stray ``launcher_source_file`` without the flag → no notice (the flag is the gate)."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        # Missing source_promoted_from_launcher → no notice.
    )
    args = _build_args()
    prompt = ko.build_prompt(cand, args)
    assert "SOURCE ATTRIBUTION NOTE" not in prompt


def test_build_kernel_metadata_carries_launcher_fields_when_promoted() -> None:
    """Structured JSON metadata surfaces the launcher / promoted-from fields for non-prose consumers."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        source_promoted_from_launcher=True,
    )
    args = _build_args()
    md = ko.build_kernel_metadata(cand, args)
    assert md["kernel_path"].endswith("gemm_moe_ck2stages.cu")
    assert md["launcher_source_file"] == "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    assert md["source_promoted_from_launcher"] is True


def test_build_kernel_metadata_default_launcher_fields_when_not_promoted() -> None:
    """Un-promoted candidates carry empty / False values, never None (JSON parsers expect string + bool)."""
    cand = _candidate()
    args = _build_args()
    md = ko.build_kernel_metadata(cand, args)
    assert md["launcher_source_file"] == ""
    assert md["source_promoted_from_launcher"] is False


def test_build_kernel_metadata_metadata_is_json_serializable_when_promoted() -> None:
    """Round-trip the metadata through json to pin serializability (it ships in the prompt via json.dumps)."""
    cand = _candidate(
        launcher_source_file="/sgl-workspace/aiter/aiter/ops/moe_op.py",
        source_promoted_from_launcher=True,
    )
    args = _build_args()
    md = ko.build_kernel_metadata(cand, args)
    encoded = json.dumps(md, sort_keys=True)
    parsed = json.loads(encoded)
    assert parsed["launcher_source_file"] == "/sgl-workspace/aiter/aiter/ops/moe_op.py"
    assert parsed["source_promoted_from_launcher"] is True
