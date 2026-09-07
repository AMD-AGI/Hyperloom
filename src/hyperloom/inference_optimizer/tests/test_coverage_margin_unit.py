# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from hyperloom.common.env import env_float
from hyperloom.inference_optimizer.breakdown.recorder import section_shape
from hyperloom.inference_optimizer.breakdown.reporters._renderers import (
    roofline,
    source_files,
    workload,
)


def test_env_float_invalid_returns_default(monkeypatch):
    monkeypatch.setenv("HL_BAD_FLOAT", "not-a-float")
    assert env_float("HL_BAD_FLOAT", 3.5) == 3.5


def test_section_shape_unknown_is_none():
    assert section_shape("not_registered") is None


def test_source_files_renderer_skips_empty_entries():
    sec = source_files.render(
        {
            "source_files": {
                "empty": [],
                "none": None,
                "single": "state.json",
                "many": ["a", "b", "c", "d"],
            }
        }
    )
    assert not sec.skipped
    assert "single" in sec.markdown_block
    assert "none" not in sec.markdown_block
    assert "a, b, c" in sec.markdown_block


def test_roofline_and_workload_render_minimal_inputs():
    roof = roofline.render(
        {
            "roofline": [
                {
                    "source_path": "final.json",
                    "mode": "compare",
                    "baseline": {"top_kernel": {"name": "k1", "gpu_pct": 12.3}},
                    "delta": {"compute_pct": "+1.0"},
                }
            ]
        }
    )
    assert not roof.skipped
    assert "k1" in roof.markdown_block

    wk = workload.render({"metadata": {"task_config": {"model_name": "m", "framework_name": "sglang"}}})
    assert not wk.skipped
    assert "sglang" in wk.markdown_block
