# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local HF checkpoint the main flow's decode ceiling can be modelled from: its config and the size of its weights."""

from __future__ import annotations

import json
from pathlib import Path

#: An 8B dense Llama in bf16.
CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "torch_dtype": "bfloat16",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
}
WEIGHT_BYTES = 16_060_522_496


def model_checkout(tmp_path: Path) -> Path:
    """The checkpoint's directory under ``tmp_path``: ``config.json`` and the safetensors index, without weights."""

    model = tmp_path / "Llama-3-8B"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(CONFIG))
    (model / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": WEIGHT_BYTES}}))
    return model
