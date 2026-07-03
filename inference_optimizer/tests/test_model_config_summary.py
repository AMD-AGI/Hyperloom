# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``summarize_model_config`` (config.json -> structured model info)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.model_config_utils import summarize_model_config


def _write_config(model_dir: Path, payload: dict) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = model_dir / "config.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def test_summary_missing_config_returns_empty(tmp_path: Path) -> None:
    assert summarize_model_config(str(tmp_path / "nope")) == {}
    assert summarize_model_config("") == {}


def test_summary_mha(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 128000,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        },
    )
    out = summarize_model_config(str(m))
    assert out["model_type"] == "llama"
    assert out["architectures"] == ["LlamaForCausalLM"]
    assert out["attention_type"] == "MHA"
    assert out["num_attention_heads"] == 32
    assert out["num_key_value_heads"] == 32
    assert out["head_dim"] == 128
    assert out["is_moe"] is False


def test_summary_gqa(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "GQA"


def test_summary_mqa(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "MQA"


def test_summary_mla(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "deepseek_v3",
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "MLA"


def test_summary_moe(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen3_moe",
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["num_experts"] == 128
    assert out["num_experts_per_tok"] == 8


def test_summary_moe_deepseek_alias(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "num_attention_heads": 128,
            "kv_lora_rank": 512,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["num_experts"] == 256
    assert out["attention_type"] == "MLA"


def test_summary_quantization(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        },
    )
    out = summarize_model_config(str(m))
    assert out["quantization"] == "fp8"


def test_summary_nested_text_config(tmp_path: Path) -> None:
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "text_config": {
                "hidden_size": 8192,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "num_hidden_layers": 80,
            },
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "GQA"
    assert out["hidden_size"] == 8192
    assert out["num_hidden_layers"] == 80


def test_summary_malformed_numeric_is_fail_soft(tmp_path: Path) -> None:
    # Non-integer/garbage numeric fields must not crash; bad fields are skipped.
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "llama",
            "num_attention_heads": "thirty-two",
            "num_key_value_heads": 8,
            "hidden_size": "4096",
            "num_hidden_layers": None,
            "vocab_size": 1.5,
            "max_position_embeddings": "8k",
        },
    )
    out = summarize_model_config(str(m))
    assert out["model_type"] == "llama"
    assert out["num_key_value_heads"] == 8
    # bad scalars skipped, never raise:
    assert "num_attention_heads" not in out
    assert "max_position_embeddings" not in out


def test_summary_llm_config_nesting(tmp_path: Path) -> None:
    # InternVL/Ovis put the text tower under llm_config, not text_config.
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "internvl_chat",
            "architectures": ["InternVLChatModel"],
            "llm_config": {
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "num_hidden_layers": 48,
                "num_experts": 128,
                "num_experts_per_tok": 8,
            },
        },
    )
    out = summarize_model_config(str(m))
    assert out["attention_type"] == "GQA"
    assert out["num_hidden_layers"] == 48
    assert out["is_moe"] is True
    assert out["num_experts"] == 128


def test_summary_stub_high_priority_scope_backfilled_by_lower(tmp_path: Path) -> None:
    # text_config is a stub (only model_type); the real decoder lives in
    # llm_config. Fields absent from the stub must be backfilled from llm_config.
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "wrapper",
            "text_config": {"model_type": "qwen2"},
            "llm_config": {
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "num_hidden_layers": 48,
                "hidden_size": 8192,
            },
        },
    )
    out = summarize_model_config(str(m))
    # Nested text_config wins for model_type/family; llm_config backfills the
    # structural fields the stub text_config omits.
    assert out["model_type"] == "qwen2"
    assert out["model_family"] == "qwen2"
    assert out["num_attention_heads"] == 64
    assert out["num_hidden_layers"] == 48
    assert out["hidden_size"] == 8192
    assert out["attention_type"] == "GQA"


def test_wrapper_model_type_reports_nested_decoder(tmp_path: Path) -> None:
    # An unrecognized top-level wrapper must report the nested decoder for both
    # model_type and model_family so collectors group by the real LLM.
    m = _write_config(
        tmp_path / "wrapper-model",
        {
            "model_type": "wrapper",
            "text_config": {
                "model_type": "qwen2",
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
            },
        },
    )
    out = summarize_model_config(str(m))
    assert out["model_type"] == "qwen2"
    assert out["model_family"] == "qwen2"


def test_qwen3_vl_resolves_to_qwen3_family(tmp_path: Path) -> None:
    # Real Qwen3-VL nests model_type=qwen3_vl_text; merged model_type carries
    # that internal name but the family still collapses to qwen3.
    m = _write_config(
        tmp_path / "qwen3-vl",
        {"model_type": "qwen3_vl", "text_config": {"model_type": "qwen3_vl_text"}},
    )
    out = summarize_model_config(str(m))
    assert out["model_type"] == "qwen3_vl_text"
    assert out["model_family"] == "qwen3"


def test_summary_round_trips_through_state_json(tmp_path: Path) -> None:
    # E2E: model_info computed at launch must survive save -> load on state.json.
    from inference_optimizer.orchestrator.shared_state import SharedState

    m = _write_config(
        tmp_path / "Qwen3-8B",
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 36,
        },
    )
    info = summarize_model_config(str(m))
    assert info  # non-empty for a real config

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    SharedState(session_id="s1", model_path=str(m), model_info=info).save(session_dir)

    reloaded = SharedState.load_or_init(session_dir)
    assert reloaded.model_info == info
    assert reloaded.model_info["model_family"] == "qwen3"
    assert reloaded.model_info["attention_type"] == "GQA"


@pytest.mark.parametrize(
    "model_type,name,expected_family",
    [
        ("rwkv6qwen2", "RWKV6-Qwen2", "qwen2"),
        ("llava_qwen2", "llava-qwen2", "qwen2"),
        ("hybrid_qwen3", "Hybrid-Qwen3", "qwen3"),
    ],
)
def test_model_family_derived_base(tmp_path: Path, model_type, name, expected_family) -> None:
    m = _write_config(
        tmp_path / name.replace("/", "_"),
        {"model_type": model_type, "num_attention_heads": 8, "num_key_value_heads": 8},
    )
    out = summarize_model_config(str(m))
    assert out.get("model_family", "") == expected_family


@pytest.mark.parametrize(
    "model_type,arches,name,expected_family",
    [
        # Qwen: generation collapses moe/next/vl/5 variants.
        ("qwen3", [], "Qwen-Qwen3-8B", "qwen3"),
        ("qwen3_moe", [], "Qwen-Qwen3-235B-A22B", "qwen3"),
        ("qwen3_next", [], "Qwen3-Next-80B", "qwen3"),
        ("qwen3_vl_moe", [], "Qwen3-VL-235B", "qwen3"),
        ("qwen3_5_moe", [], "Qwen3.5", "qwen3"),
        ("qwen2", [], "Qwen2.5-7B", "qwen2"),
        ("qwen2_5_vl", [], "Qwen2.5-VL", "qwen2"),
        # DeepSeek: keep major version.
        ("deepseek_v3", [], "DeepSeek-V3", "deepseek_v3"),
        ("deepseek_v2", [], "DeepSeek-V2", "deepseek_v2"),
        ("deepseek_v32", [], "DeepSeek-V3.2", "deepseek_v3"),
        # Gemma generations.
        ("gemma4", [], "Gemma-4", "gemma4"),
        ("gemma3_text", [], "Gemma-3", "gemma3"),
        ("gemma2", [], "Gemma-2", "gemma2"),
        # Mistral vs Mixtral kept distinct.
        ("mixtral", [], "Mixtral-8x7B", "mixtral"),
        ("mistral3", [], "Mistral-3", "mistral"),
        # Llama generation from name when model_type is bare 'llama'.
        ("llama", ["LlamaForCausalLM"], "Llama-3.1-8B-Instruct", "llama3"),
        ("llama", ["LlamaForCausalLM"], "Llama-2-7b", "llama2"),
        ("llama4", [], "Llama-4", "llama4"),
        # Others use family prefix.
        ("glm4_moe", [], "GLM-4.6", "glm4"),
        ("phi3", [], "Phi-3-mini", "phi3"),
        ("minimax_m2", [], "MiniMax-M2", "minimax"),
        ("nemotron_h", [], "Nemotron-3", "nemotron"),
        ("internvl_chat", [], "InternVL3_5", "internvl"),
        ("", [], "random-unknown-model", ""),
    ],
)
def test_model_family(tmp_path: Path, model_type, arches, name, expected_family) -> None:
    payload: dict = {"num_attention_heads": 8, "num_key_value_heads": 8}
    if model_type:
        payload["model_type"] = model_type
    if arches:
        payload["architectures"] = arches
    # The model dir name carries the name hint (path basename drives llama gen).
    out = summarize_model_config(str(_write_config(tmp_path / name.replace("/", "_"), payload)))
    assert out.get("model_family", "") == expected_family


# ---------------------------------------------------------------------------
# Shared-expert detection
# ---------------------------------------------------------------------------

def test_shared_expert_detected_via_n_shared_experts(tmp_path: Path) -> None:
    """MiniMax-M3 / DeepSeek-style: n_shared_experts key emits has_shared_expert."""
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "deepseek_v3",
            "num_experts": 256,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["has_shared_expert"] is True
    assert out["num_shared_experts"] == 1


def test_shared_expert_detected_via_num_shared_experts(tmp_path: Path) -> None:
    """Alternate key num_shared_experts also triggers detection."""
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen2_moe",
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "num_shared_experts": 2,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["has_shared_expert"] is True
    assert out["num_shared_experts"] == 2


def test_shared_expert_detected_via_shared_expert_intermediate_size(tmp_path: Path) -> None:
    """Qwen-MoE style: shared_expert_intermediate_size without an explicit count."""
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "qwen2_moe",
            "num_experts": 64,
            "num_experts_per_tok": 4,
            "shared_expert_intermediate_size": 4096,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["has_shared_expert"] is True
    # num_shared_experts not known from this key alone — should not be emitted
    assert "num_shared_experts" not in out


def test_plain_moe_without_shared_expert_does_not_emit(tmp_path: Path) -> None:
    """Pure routed MoE (Mixtral-style) must not emit shared-expert fields."""
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "mixtral",
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert "has_shared_expert" not in out
    assert "num_shared_experts" not in out


def test_non_moe_with_shared_marker_does_not_emit(tmp_path: Path) -> None:
    """A non-MoE model with a shared-looking key must not trigger fusion hint."""
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "llama",
            "n_shared_experts": 1,  # anomalous field on a non-MoE model
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is False
    assert "has_shared_expert" not in out


def test_shared_expert_via_nested_text_config(tmp_path: Path) -> None:
    """Multimodal wrapper: shared-expert fields inside text_config are detected."""
    m = _write_config(
        tmp_path / "m",
        {
            "model_type": "multimodal_wrapper",
            "text_config": {
                "model_type": "deepseek_v3",
                "num_experts": 256,
                "n_shared_experts": 1,
            },
        },
    )
    out = summarize_model_config(str(m))
    assert out["is_moe"] is True
    assert out["has_shared_expert"] is True
    assert out["num_shared_experts"] == 1
