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


def test_shapes_non_dsv4_model_omits_o_groups_fields(tmp_path):
    """A model without ``o_groups`` (real Llama-2-70B GQA config) must NOT gain the
    DSv4 wo_a group axis, and the group-axis guard must be a no-op for it."""
    from kernelforge.fusion.shapes import harness_group_dim_mismatch

    # Real meta-llama/Llama-2-70b-hf config values.
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 28672,
                "num_hidden_layers": 80,
            }
        )
    )
    # Even under attention TP the group axis stays absent for a non-DSv4 model.
    s = resolve_decode_shapes(str(tmp_path), attn_tp_size=8)
    assert "o_groups" not in s
    assert "n_local_groups" not in s
    # The whole shapes dict is rendered into the authoring prompt, so the DSv4
    # instruction must not reach a model that has no group axis.
    assert "group_axis_note" not in s
    # n_local_heads is generic attention TP math and still stamped.
    assert s["n_local_heads"] == 64 // 8
    assert s["gqa_groups"] == 64 // 8
    # No o_groups -> the guard cannot fire, whatever the harness sets G to.
    assert harness_group_dim_mismatch("G = 8  # gqa_groups\n", s) == ""
    assert harness_group_dim_mismatch("G = shapes['gqa_groups']\n", s) == ""


def test_shapes_dsv4_o_groups_not_confused_with_gqa(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_attention_heads": 128,
                "num_key_value_heads": 1,
                "o_groups": 16,
                "head_dim": 512,
                "qk_rope_head_dim": 64,
                "o_lora_rank": 1024,
                "hidden_size": 7168,
            }
        )
    )
    s = resolve_decode_shapes(str(tmp_path), attn_tp_size=1)
    assert s["gqa_groups"] == 128
    assert s["o_groups"] == 16
    assert s["n_local_groups"] == 16
    assert s["n_local_heads"] == 128
    assert s["qk_rope_head_dim"] == 64
    assert "n_local_groups" in s["group_axis_note"]


def test_shapes_attn_tp_scales_local_dims(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_attention_heads": 128,
                "o_groups": 16,
            }
        )
    )
    s = resolve_decode_shapes(str(tmp_path), attn_tp_size=8)
    assert s["n_local_heads"] == 16
    assert s["n_local_groups"] == 2


def test_shapes_dsv4_tp_sharded_attention_scales_group_axis(tmp_path):
    """DSv4-Flash at tp=4 with attention sharded by TP (no DP-attention).

    Same config as ``test_deepseek_v4_sparse_mla_matches_runtime_shapes`` in
    gemm_tune, which is the sharding this repo already serves DSv4-Flash with.
    """
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "num_attention_heads": 64,
                "num_key_value_heads": 1,
                "o_groups": 8,
                "head_dim": 512,
                "o_lora_rank": 1024,
            }
        )
    )
    s = resolve_decode_shapes(str(tmp_path), attn_tp_size=4)
    assert s["n_local_groups"] == 2
    assert s["n_local_heads"] == 16


def test_harness_group_dim_mismatch_catches_gqa_as_g(tmp_path):
    from kernelforge.fusion.shapes import harness_group_dim_mismatch

    shapes = {
        "o_groups": 16,
        "n_local_groups": 16,
        "gqa_groups": 128,
        "num_attention_heads": 128,
        "n_local_heads": 128,
    }
    bad = "G = 128  # gqa_groups / n_local_groups\n"
    why = harness_group_dim_mismatch(bad, shapes)
    assert why and "n_local_groups=16" in why
    assert harness_group_dim_mismatch("G = 16  # n_local_groups\n", shapes) == ""
    assert "gqa_groups" in harness_group_dim_mismatch("G = shapes['gqa_groups']\n", shapes)


def test_shapes_missing_config_returns_minimal(tmp_path):
    s = resolve_decode_shapes(str(tmp_path))  # no config.json
    assert s["model_type"] == ""
    assert s["T"] == 16
