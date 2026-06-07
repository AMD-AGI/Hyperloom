# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the config.json-derived KB architecture tags.

Covers two layers:

1. ``cli._load_model_config_tags`` — the soft-degrade loader that lifts
   ``architectures`` + ``model_type`` out of a model's ``config.json``.
2. The T0 anchor write path — ``run_t0_anchor`` stamps those tags into the
   recipe-snapshot ``extras`` so a fine-tuned model's row records the same
   architecture identity as the base model it derives from.

The amend (KEEP / REVERT / CLOSE) write path is covered in
``test_coordinator_kb_writes.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.cli import _load_model_config_tags
from inference_optimizer.orchestrator.cortex_t0 import run_t0_anchor
from inference_optimizer.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    recipe_canonical_id,
)


def _write_config(model_dir: Path, payload: Any) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_dir / "config.json"
    if isinstance(payload, str):
        cfg.write_text(payload, encoding="utf-8")
    else:
        cfg.write_text(json.dumps(payload), encoding="utf-8")
    return cfg


# ===========================================================================
# 1. _load_model_config_tags — happy path + soft-degrade matrix
# ===========================================================================
def test_load_config_tags_valid(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
    })
    out = _load_model_config_tags(str(tmp_path / "m"))
    assert out == {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}


def test_load_config_tags_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _load_model_config_tags(str(tmp_path / "nonexistent")) == {}


def test_load_config_tags_empty_path_returns_empty() -> None:
    assert _load_model_config_tags("") == {}


def test_load_config_tags_invalid_json_returns_empty(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", "{not valid json")
    assert _load_model_config_tags(str(tmp_path / "m")) == {}


def test_load_config_tags_non_dict_returns_empty(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", ["LlamaForCausalLM"])
    assert _load_model_config_tags(str(tmp_path / "m")) == {}


def test_load_config_tags_scalar_architectures_wrapped(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", {
        "architectures": "LlamaForCausalLM",
        "model_type": "llama",
    })
    out = _load_model_config_tags(str(tmp_path / "m"))
    assert out["architectures"] == ["LlamaForCausalLM"]


def test_load_config_tags_omits_empty_fields(tmp_path: Path) -> None:
    # Only model_type present -> architectures key omitted entirely.
    _write_config(tmp_path / "m", {"model_type": "qwen3_moe"})
    assert _load_model_config_tags(str(tmp_path / "m")) == {"model_type": "qwen3_moe"}
    # Only architectures present -> model_type key omitted.
    _write_config(tmp_path / "n", {"architectures": ["Qwen3MoeForCausalLM"]})
    assert _load_model_config_tags(str(tmp_path / "n")) == {
        "architectures": ["Qwen3MoeForCausalLM"],
    }


def test_load_config_tags_drops_blank_entries(tmp_path: Path) -> None:
    _write_config(tmp_path / "m", {
        "architectures": ["LlamaForCausalLM", "", "  "],
        "model_type": "  llama  ",
    })
    out = _load_model_config_tags(str(tmp_path / "m"))
    assert out["architectures"] == ["LlamaForCausalLM"]
    assert out["model_type"] == "llama"


# ===========================================================================
# 2. T0 anchor stamps the tags into the recipe extras
# ===========================================================================
@dataclass
class _FakeSharedState:
    cortex_session_id: str = ""
    warm_start_ts: str = ""
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[Any] = field(default_factory=list)
    warm_start_lessons: list[Any] = field(default_factory=list)
    framework: str = "sglang"
    framework_version: str = "0.4.5"
    precision: str = "fp8"
    tp: int = 8
    ep: int = 0
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    model_class: str = ""
    model_architectures: list[str] = field(default_factory=list)
    model_type: str = ""

    def save(self, _path: Path) -> None:  # noqa: D401
        """No-op save — these tests don't assert on disk persistence."""


@pytest.fixture
def kb(tmp_path: Path) -> RecipeKB:
    return RecipeKB(local=LocalRecipeStore(root=tmp_path / "kb"), remote=None)


def _cid(state: _FakeSharedState, workload: str, hw: str) -> str:
    return recipe_canonical_id(
        model=workload, hardware=hw,
        framework=state.framework,
        framework_version=state.framework_version,
        precision=state.precision,
    )


def test_t0_anchor_stamps_architecture_tags(kb: RecipeKB, tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    state = _FakeSharedState(
        model_architectures=["LlamaForCausalLM"],
        model_type="llama",
    )
    run_t0_anchor(
        kb, state,
        workload="Llama-3.1-8B-MyFinetune", hw="MI300X",
        extra_attrs={"framework": "sglang"},
        session_dir=sd,
    )
    row = kb.get_recipe(
        canonical_id=_cid(state, "Llama-3.1-8B-MyFinetune", "MI300X"),
    )
    assert row is not None
    assert row.get("architectures") == ["LlamaForCausalLM"]
    assert row.get("model_type") == "llama"


def test_t0_anchor_skips_empty_architecture_tags(
    kb: RecipeKB, tmp_path: Path,
) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    state = _FakeSharedState()  # no config.json tags
    run_t0_anchor(
        kb, state,
        workload="m", hw="mi300x",
        extra_attrs={"framework": "sglang"},
        session_dir=sd,
    )
    row = kb.get_recipe(canonical_id=_cid(state, "m", "mi300x"))
    assert row is not None
    assert "architectures" not in row
    assert "model_type" not in row
