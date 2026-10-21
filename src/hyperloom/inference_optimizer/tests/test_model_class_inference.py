# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Model-class inference from local model metadata."""

from __future__ import annotations

import json

from hyperloom.inference_optimizer.model_config_utils import infer_model_class


def test_infer_model_class_dense_empty():
    assert infer_model_class("") == "dense"


def test_infer_model_class_moe_from_text():
    assert infer_model_class("/models/Mixtral-8x7B") == "moe_swa"


def test_infer_model_class_moe_mla_nsa_from_text():
    assert infer_model_class("/models/GLM-5-air") == "moe_mla_nsa"


def test_infer_model_class_moe_mla_from_text():
    assert infer_model_class("/models/DeepSeek-V3") == "moe_mla"


def test_infer_model_class_reads_config_json(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "num_experts": 8}),
        encoding="utf-8",
    )
    # num_experts > 0 -> MoE; no MLA/NSA text -> moe_swa.
    assert infer_model_class(str(tmp_path)) == "moe_swa"


def test_infer_model_class_ignores_bool_experts(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_experts": True, "model_type": "llama"}),
        encoding="utf-8",
    )
    assert infer_model_class(str(tmp_path)) == "dense"
