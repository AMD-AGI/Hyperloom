# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A fake vLLM source tree with one model file, plus the model config that names it."""

from __future__ import annotations

import json
from pathlib import Path


def fake_vllm_tree(tmp_path: Path, model_type: str, body: str) -> tuple[str, str]:
    """Write ``body`` as vLLM's ``model_type`` model; return the model dir and the framework root."""
    mdir = tmp_path / "fw" / "vllm" / "model_executor" / "models"
    mdir.mkdir(parents=True)
    (mdir / f"{model_type}.py").write_text(body, encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 2048, "num_attention_heads": 16}),
        encoding="utf-8",
    )
    return str(model), str(tmp_path / "fw")
