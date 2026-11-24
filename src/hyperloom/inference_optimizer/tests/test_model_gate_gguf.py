# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A checkpoint that ships only GGUF weights is what a framework that loads a GGUF file serves."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.model_gate import _detect_incompatible_model_config


@pytest.fixture()
def gguf_only(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}))
    (model / "model-Q8_0.gguf").write_bytes(b"GGUF")
    (model / "tokenizer.json").write_text("{}")
    (model / "tokenizer_config.json").write_text("{}")
    return model


def test_a_gguf_only_checkpoint_is_served_by_a_framework_that_loads_the_gguf(gguf_only: Path) -> None:
    assert _detect_incompatible_model_config(str(gguf_only), framework="llamacpp") is None
