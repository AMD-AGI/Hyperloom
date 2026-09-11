# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from hyperloom.common.env import env_float
from hyperloom.inference_optimizer.breakdown.recorder import section_shape
from hyperloom.inference_optimizer.breakdown.reporters._renderers import (
    roofline,
    workload,
)


def test_env_float_invalid_returns_default(monkeypatch):
    monkeypatch.setenv("HL_BAD_FLOAT", "not-a-float")
    assert env_float("HL_BAD_FLOAT", 3.5) == 3.5


def test_section_shape_unknown_is_none():
    assert section_shape("not_registered") is None


def test_roofline_and_workload_render_minimal_inputs():
    roof = roofline.render(
        {
            "timeline": [
                {
                    "type": "roofline",
                    "ext": {
                        "actions": [
                            {"outcome": {"snapshot": {"snapshot_id": 1, "top_kernel": {"name": "k1", "gpu_pct": 12.3}}}}
                        ]
                    },
                }
            ]
        }
    )
    assert not roof.skipped
    assert "k1" in roof.markdown_block

    wk = workload.render({"metadata": {"task_config": {"model_name": "m", "framework_name": "sglang"}}})
    assert not wk.skipped
    assert "sglang" in wk.markdown_block
