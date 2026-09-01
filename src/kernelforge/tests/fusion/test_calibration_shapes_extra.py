# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover calibration load/interp error paths and shapes nested-config branches."""

from __future__ import annotations

import json

from kernelforge.fusion import calibration as cal
from kernelforge.fusion.calibration import _interp, load_calibration_points
from kernelforge.fusion.shapes import resolve_decode_shapes


# ── calibration ──────────────────────────────────────────────────────────────
def test_load_calibration_no_path(monkeypatch):
    monkeypatch.delenv("FORGE_FUSION_CALIBRATION", raising=False)
    assert load_calibration_points() == []


def test_load_calibration_bad_json(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text("not json {")
    assert load_calibration_points(str(p)) == []


def test_load_calibration_missing_file(tmp_path):
    assert load_calibration_points(str(tmp_path / "nope.json")) == []


def test_load_calibration_skips_bad_rows(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(
        json.dumps(
            [
                {"share": 0.2, "gain": 0.02},  # ok
                {"share": "x", "gain": 0.01},  # bad float -> skipped
                {"gain": 0.01},  # missing key -> skipped
                [0.4, 0.06],  # list form ok
                [0.5],  # index error -> skipped
                {"share": -0.1, "gain": 0.5},  # negative -> filtered
            ]
        )
    )
    pts = load_calibration_points(str(p))
    assert pts == [(0.2, 0.02), (0.4, 0.06)]


def test_load_calibration_non_list_json(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"share": 0.2}))  # dict, not list -> empty
    assert load_calibration_points(str(p)) == []


def test_interp_empty_returns_zero():
    assert _interp([], 0.3) == 0.0


def test_interp_clamps_below_and_above():
    pts = [(0.2, 0.02), (0.4, 0.06)]
    assert _interp(pts, 0.1) == 0.02  # below first
    assert _interp(pts, 0.9) == 0.06  # above last


def test_interp_exact_and_midpoint():
    pts = [(0.2, 0.02), (0.4, 0.06)]
    assert abs(_interp(pts, 0.3) - 0.04) < 1e-9


def test_predict_uses_env_calibration(tmp_path, monkeypatch):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps([[0.2, 0.02], [0.4, 0.06]]))
    monkeypatch.setenv("FORGE_FUSION_CALIBRATION", str(p))
    g = cal.predict_cuda_graph_on_gain(0.3)
    assert abs(g - 0.04) < 1e-6  # from env-loaded points


# ── shapes ───────────────────────────────────────────────────────────────────
def test_shapes_reads_nested_text_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "multimodal",
                "text_config": {"hidden_size": 4096, "num_attention_heads": 32},
            }
        )
    )
    s = resolve_decode_shapes(str(tmp_path))
    assert s["hidden_size"] == 4096
    assert s["num_attention_heads"] == 32
    assert s["head_dim"] == 128  # derived 4096//32


def test_shapes_head_dim_zero_heads_no_crash(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "weird",
                "hidden_size": 2048,
                "num_attention_heads": 0,
            }
        )
    )
    s = resolve_decode_shapes(str(tmp_path))
    # division by zero -> head_dim omitted, no crash
    assert "head_dim" not in s
    assert s["model_type"] == "weird"


def test_shapes_gqa_groups_computed(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "gqa",
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
            }
        )
    )
    s = resolve_decode_shapes(str(tmp_path))
    assert s["gqa_groups"] == 4


def test_shapes_missing_config_returns_minimal(tmp_path):
    s = resolve_decode_shapes(str(tmp_path))  # no config.json
    assert s["model_type"] == ""
    assert s["T"] == 16
