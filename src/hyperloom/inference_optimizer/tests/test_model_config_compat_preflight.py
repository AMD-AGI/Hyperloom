# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the model-config compatibility preflight.

Policy: fail fast (with a persisted stop reason) when config.json is present
but statically known to crash vLLM/transformers at load — a corrupt config, or
a RoPE block without any max-position field (the DeepSeek-V3.2-Exp shape that
dies with "'PreTrainedConfig' object has no attribute 'max_position_embeddings'"
deep in engine init). A fully absent config is NOT blocked (soft-degrade).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import model_gate as cli
from hyperloom.inference_optimizer.cli import model_gate as cli_model_gate


def _write_config(model_dir: Path, *, with_tokenizer: bool = True, **fields) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(fields), encoding="utf-8")
    # Most config-compat tests are unrelated to the tokenizer-artifact check;
    # ship a tokenizer by default so they exercise only the field they target.
    if with_tokenizer:
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def _args(model: str, *, gpu_type: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(model=model, isl=1024, osl=1024, gpu_type=gpu_type)


def _seed_state(session_dir: Path, monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR",
        str(session_dir),
    )
    from hyperloom.orchestrator.state.shared_state import SharedState

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(session_id="t", model_name="m", model_path="m").save(session_dir)


@pytest.fixture(autouse=True)
def _default_non_amd_gpu(monkeypatch):
    """Keep config checks hermetic unless a test passes gpu_type explicitly."""
    monkeypatch.delenv("GPU_TYPE", raising=False)
    # _detect_incompatible_model_config -> _resolve_amd_gpu_type ->
    # _autodetect_gpu_type all live in cli_model_gate after the phase-6D fold;
    # patch the real call site (cli re-exports the same object).
    monkeypatch.setattr(cli_model_gate, "_autodetect_gpu_type", lambda: None)


# ---------------------------------------------------------------------------
# _detect_incompatible_model_config
# ---------------------------------------------------------------------------
def test_compat_detector_registry_order_is_pinned():
    """The waterfall order is a behavioral contract (first match wins).

    Steps 1 (diffusers) and 2 (config absent/corrupt) are the inline prologue in
    ``_detect_incompatible_model_config``; this registry is steps 3-15. Pin the
    exact order + the amd_only / skip_when_scriptable flags so an accidental
    reorder (which would silently change which reason wins) is caught.
    """
    specs = cli_model_gate._COMPAT_DETECTORS
    assert [s.name for s in specs] == [
        "amd_unsupported_quant",  # 3
        "amd_unsupported_architecture",  # 4
        "null_strict_bool",  # 5
        "rope_without_max_position",  # 6
        "phi3_rope_scaling",  # 7
        "gemma2_hidden_act",  # 8
        "unrecognized_architecture",  # 9
        "private_quant",  # 10
        "peft_adapter_only",  # 11
        "vocab_weight_shape",  # 12
        "tokenizer_artifacts",  # 13
        "unregistered_custom_autoconfig",  # 14
        "amd_dual_chunk_attention",  # 15
    ]
    assert {s.name for s in specs if s.amd_only} == {
        "amd_unsupported_quant",
        "amd_unsupported_architecture",
        "amd_dual_chunk_attention",
    }
    assert {s.name for s in specs if s.skip_when_scriptable} == {
        "tokenizer_artifacts",
    }
    # Step 13 is a 3-detector short-circuiting sub-chain.
    tok = next(s for s in specs if s.name == "tokenizer_artifacts")
    assert isinstance(tok.fn, tuple)
    assert len(tok.fn) == 3


def test_detect_healthy_config_returns_none(tmp_path):
    m = tmp_path / "ok"
    _write_config(m, model_type="llama", max_position_embeddings=4096)
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_diffusers_pipeline_blocks_without_config_json(tmp_path):
    m = tmp_path / "flux"
    m.mkdir()
    (m / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}),
        encoding="utf-8",
    )

    reason = cli._detect_incompatible_model_config(str(m))

    assert reason is not None
    assert "Diffusers pipeline" in reason
    assert "causal LM" in reason


def test_detect_null_use_cache_blocks_strict_config_validation(tmp_path):
    m = tmp_path / "null_use_cache"
    _write_config(
        m,
        model_type="qwen2",
        max_position_embeddings=4096,
        use_cache=None,
    )

    reason = cli._detect_incompatible_model_config(str(m))

    assert reason is not None
    assert "use_cache" in reason
    assert "StrictDataclassFieldValidationError" in reason


def test_detect_null_use_cache_in_text_config_blocks(tmp_path):
    m = tmp_path / "nested_null_use_cache"
    _write_config(
        m,
        model_type="wrapper",
        max_position_embeddings=4096,
        text_config={"model_type": "qwen2", "use_cache": None},
    )

    reason = cli._detect_incompatible_model_config(str(m))

    assert reason is not None
    assert "text_config.use_cache" in reason


def test_detect_rope_with_maxpos_ok(tmp_path):
    m = tmp_path / "rope_ok"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        rope_scaling={"type": "yarn", "factor": 4.0},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_missing_tokenizer_blocks(tmp_path):
    # Gensyn-Swarm fine-tune class: weights + config, no tokenizer artifacts.
    m = tmp_path / "no_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "tokenizer" in reason.lower()


def test_detect_missing_tokenizer_skipped_for_scriptable_xdit(tmp_path):
    """xDiT (scriptable) is a server-less image workload that never loads a HF
    tokenizer, so a missing-tokenizer config.json must NOT block it."""
    m = tmp_path / "xdit_no_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    assert (
        cli._detect_incompatible_model_config(str(m), framework="xdit") is None
    )


def test_detect_missing_tokenizer_still_blocks_serving_framework(tmp_path):
    """Regression guard: the skip is scoped to scriptable — an explicit serving
    framework (sglang) must still block a missing-tokenizer checkpoint."""
    m = tmp_path / "sglang_no_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    reason = cli._detect_incompatible_model_config(str(m), framework="sglang")
    assert reason is not None
    assert "tokenizer" in reason.lower()


def test_detect_with_tokenizer_ok(tmp_path):
    m = tmp_path / "with_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    (m / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_mistral_tokenizer_json_without_common_files_blocks(tmp_path):
    m = tmp_path / "mistral_json_only"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="mistral",
        architectures=["MistralForCausalLM"],
        max_position_embeddings=32768,
    )
    (m / "tokenizer.json").write_text("{}", encoding="utf-8")

    reason = cli._detect_incompatible_model_config(str(m))

    assert reason is not None
    assert "MistralCommonBackend" in reason
    assert "No tokenizer file found" in reason


def test_detect_mistral_tokenizer_model_file_ok(tmp_path):
    m = tmp_path / "mistral_tokenizer_model"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="mistral",
        architectures=["MistralForCausalLM"],
        max_position_embeddings=32768,
    )
    (m / "tokenizer.json").write_text("{}", encoding="utf-8")
    (m / "tokenizer.model").write_text("dummy", encoding="utf-8")

    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_missing_tokenizer_but_auto_map_ok(tmp_path):
    # A custom AutoTokenizer in auto_map can supply the tokenizer at load time.
    m = tmp_path / "auto_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="custom",
        max_position_embeddings=4096,
        auto_map={"AutoTokenizer": ["x.TokClass", None]},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_minimax_m1_blocked_on_amd(tmp_path):
    # minimax_m1 lightning-attention kernel needs 128KB LDS > MI300X 64KB.
    m = tmp_path / "minimax"
    _write_config(
        m,
        model_type="minimax_m1",
        architectures=["MiniMaxM1ForCausalLM"],
        max_position_embeddings=80000,
    )
    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "AMD/ROCm" in reason


def test_detect_minimax_m1_not_blocked_off_amd(tmp_path):
    # AMD-specific LDS limit; do not block on non-AMD hardware.
    m = tmp_path / "minimax_non_amd"
    _write_config(
        m,
        model_type="minimax_m1",
        architectures=["MiniMaxM1ForCausalLM"],
        max_position_embeddings=80000,
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_unrecognized_arch_blocked_hardware_agnostic(tmp_path):
    # glm4_moe_lite: Transformers does not recognize → ValidationError on any GPU.
    m = tmp_path / "glm47flash"
    _write_config(
        m,
        model_type="glm4_moe_lite",
        architectures=["Glm4MoeLiteForCausalLM"],
        max_position_embeddings=131072,
    )
    reason_amd = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    reason_off = cli._detect_incompatible_model_config(str(m))
    assert reason_amd is not None and "not recognized" in reason_amd
    assert reason_off is not None and "not recognized" in reason_off


def test_detect_mimo_v2_flash_unrecognized_blocked(tmp_path):
    # mimo_v2_flash: unrecognized arch + Unknown attention backend TRITON.
    m = tmp_path / "mimo"
    _write_config(
        m,
        model_type="mimo_v2_flash",
        architectures=["MiMoV2FlashForCausalLM"],
        max_position_embeddings=131072,
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "not recognized" in reason


def test_detect_deepseek_v4_now_recognized(tmp_path):
    # DeepSeek-V4: the current runtime (transformers 5.8.1 + vLLM 0.21.0 rocm722)
    # resolves DeepseekV4Config via AutoConfig and vLLM get_config, so it was
    # removed from the unrecognized blocklist and is no longer fail-fasted by the
    # model-config gate.
    m = tmp_path / "deepseek_v4"
    _write_config(
        m,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        max_position_embeddings=1048576,
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is None


def test_detect_glm_moe_dsa_now_recognized(tmp_path):
    # zai-org GLM-5.x (glm_moe_dsa): the current stack (transformers 5.12.1 +
    # vLLM 0.24.0) registers it (AutoConfig resolves GlmMoeDsaConfig, vLLM
    # ModelRegistry registers GlmMoeDsaForCausalLM; gfx942 sparse-MLA falls back
 # to Triton via vllm-project/vllm#45782). It was removed from the
    # unrecognized blocklist, so it is no longer fail-fasted by the gate.
    m = tmp_path / "glm_moe_dsa"
    _write_config(
        m,
        model_type="glm_moe_dsa",
        architectures=["GlmMoeDsaForCausalLM"],
        max_position_embeddings=131072,
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is None


def test_detect_gemma4_now_recognized(tmp_path):
    # google-gemma-4-26B-A4B-it: the current pinned stack (transformers 5.5.0 +
    # vLLM 0.18.2rc1) registers gemma4 (CONFIG_MAPPING has "gemma4", AutoConfig
    # resolves Gemma4Config, vLLM ModelConfig resolves Gemma4ForCausalLM). It was
    # removed from the unrecognized blocklist, so a plain text Gemma4 config is no
    # longer fail-fasted by the model-config gate.
    m = tmp_path / "gemma4"
    _write_config(
        m,
        model_type="gemma4",
        architectures=["Gemma4ForCausalLM"],
        max_position_embeddings=32768,
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is None


def test_detect_bailing_ling_left_to_runtime_scope(tmp_path):
 # #649 does not add Bailing filters; keep these out of the fail-fast gate.
    for model_type, arch in (
        ("bailing_moe", "BailingMoeV2ForCausalLM"),
        ("bailing_hybrid", "BailingMoeV2_5ForCausalLM"),
    ):
        m = tmp_path / model_type
        _write_config(
            m,
            model_type=model_type,
            architectures=[arch],
            max_position_embeddings=32768,
        )
        reason = cli._detect_incompatible_model_config(str(m))
        assert reason is None


def test_detect_ovis_next_left_to_runtime_scope(tmp_path):
 # #649 does not add Ovis filters; keep this out of the fail-fast gate.
    m = tmp_path / "ovis"
    _write_config(
        m,
        model_type="ovis2_6_next",
        architectures=["Ovis2_6_NextForCausalLM"],
        max_position_embeddings=32768,
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is None


def test_detect_nested_ministral3_unrecognized_blocked(tmp_path):
    # Mistral3 wrapper exposes text_config.model_type=ministral3; vLLM raises KeyError.
    m = tmp_path / "mistral3"
    _write_config(
        m,
        model_type="mistral3",
        architectures=["Mistral3ForConditionalGeneration"],
        image_token_index=10,
        text_config={
            "model_type": "ministral3",
            "max_position_embeddings": 393216,
            "vocab_size": 131072,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
        },
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "ministral3" in reason


def test_detect_pure_nested_ministral3_blocked(tmp_path):
    # Parent model_type is NOT ministral3 (a generic wrapper); only the nested
    # text_config carries ministral3. Verifies the nested-only gate in isolation
    # from the parent scope.
    m = tmp_path / "wrapper_nested_ministral3"
    _write_config(
        m,
        model_type="some_wrapper",
        architectures=["SomeWrapperForConditionalGeneration"],
        text_config={
            "model_type": "ministral3",
            "max_position_embeddings": 32768,
            "vocab_size": 131072,
        },
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "ministral3" in reason


def test_detect_nested_qwen3_5_moe_text_not_blocked(tmp_path):
    # The Qwen3.5-MoE wrapper exposes text_config.model_type=qwen3_5_moe_text.
    # The current runtime registers Qwen3_5MoeConfig/Qwen3_5MoeTextConfig and
    # vLLM ModelRegistry knows Qwen3_5MoeForConditionalGeneration, so the nested
    # type is no longer gated and falls through to the text_coercible degraded
    # path.
    m = tmp_path / "wrapper_nested_qwen3_5_moe_text"
    _write_config(
        m,
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config={
            "model_type": "qwen3_5_moe_text",
            "max_position_embeddings": 262144,
            "vocab_size": 151936,
        },
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_top_level_qwen3_5_moe_not_blocked(tmp_path):
    # Bare top-level qwen3_5_moe remains in the text-coercible runtime path; only
    # the nested text_config subtype has a confirmed registry failure.
    m = tmp_path / "bare_qwen3_5_moe"
    _write_config(
        m,
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        max_position_embeddings=262144,
        vocab_size=151936,
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_top_level_ministral3_not_blocked(tmp_path):
    # A bare top-level model_type=ministral3 (no Mistral3 wrapper) is left to the
    # framework; only the nested text_config form is a confirmed failure.
    m = tmp_path / "bare_ministral3"
    _write_config(
        m,
        model_type="ministral3",
        architectures=["Ministral3ForCausalLM"],
        max_position_embeddings=32768,
        vocab_size=131072,
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_glm4_moe_not_blocked(tmp_path):
    # glm4_moe (GLM-4.5/4.6 mainline) is a supported arch; must NOT be blocked
    # by the unrecognized-arch rule (only glm4_moe_lite is unrecognized).
    m = tmp_path / "glm4moe"
    _write_config(
        m,
        model_type="glm4_moe",
        architectures=["Glm4MoeForCausalLM"],
        max_position_embeddings=131072,
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_amd_unsupported_arch_with_maxpos_blocks(tmp_path):
    m = tmp_path / "deepseek_v32_amd"
    _write_config(
        m,
        model_type="deepseek_v32",
        architectures=["DeepseekV32ForCausalLM"],
        max_position_embeddings=163840,
        rope_scaling={"type": "yarn", "factor": 40.0},
    )

    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "AMD/ROCm" in reason


def test_detect_amd_unsupported_arch_not_blocked_off_amd(tmp_path):
    m = tmp_path / "deepseek_v32_non_amd"
    _write_config(
        m,
        model_type="deepseek_v32",
        architectures=["DeepseekV32ForCausalLM"],
        max_position_embeddings=163840,
        rope_scaling={"type": "yarn", "factor": 40.0},
    )

    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_rope_without_maxpos_blocks(tmp_path):
    m = tmp_path / "rope_no_maxpos"
    _write_config(
        m,
        model_type="deepseek_v32",
        rope_scaling={"factor": 4.0},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "RoPE" in reason


# Phi-3 su/longrope rope_scaling: even a canonical 3-key rope_scaling crashes,
# because transformers folds the top-level rope_theta into it at load so the
# Phi3Config validator sees 4 keys and raises before any override can apply.
# This mirrors the real failing checkpoints (Seacom/anakin87/ReDiX/SciPhi).
def test_detect_phi3_su_canonical_three_fields_blocks(tmp_path):
    m = tmp_path / "phi3_su"
    _write_config(
        m,
        model_type="phi3",
        architectures=["Phi3ForCausalLM"],
        max_position_embeddings=131072,
        rope_theta=10000.0,
        rope_scaling={
            "type": "su",
            "short_factor": [1.05, 1.1],
            "long_factor": [1.03, 1.05],
        },
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "Phi-3" in reason
    assert "rope_scaling" in reason


def test_detect_phi3_longrope_blocks(tmp_path):
    m = tmp_path / "phi3_longrope"
    _write_config(
        m,
        model_type="phi3",
        max_position_embeddings=131072,
        rope_theta=10000.0,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.05, 1.1],
            "long_factor": [1.03, 1.05],
        },
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "Phi-3" in reason


def test_detect_phi3_yarn_rope_not_blocked(tmp_path):
    # yarn is the non-longrope path and validates differently; do not block it.
    m = tmp_path / "phi3_yarn"
    _write_config(
        m,
        model_type="phi3",
        max_position_embeddings=131072,
        rope_scaling={"type": "yarn", "factor": 4.0},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_phi3_longrope_without_rope_theta_not_blocked(tmp_path):
    # Without a top-level rope_theta, transformers does not fold an extra key
    # into rope_scaling, so the 3-key dict passes Phi3Config validation fine.
    m = tmp_path / "phi3_longrope_no_theta"
    _write_config(
        m,
        model_type="phi3",
        max_position_embeddings=131072,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.05, 1.1],
            "long_factor": [1.03, 1.05],
        },
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_non_phi3_su_rope_not_blocked(tmp_path):
    # The Phi-3 validator is Phi3Config-specific; a non-phi3 model with a
    # su/longrope-typed rope_scaling must not be caught by this gate.
    m = tmp_path / "llama_su"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        rope_scaling={"type": "su", "short_factor": [1.05], "extra": 1},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


# Gemma2 missing hidden_act: sglang's gemma2 runtime reads config.hidden_act
# unconditionally; some checkpoints ship only hidden_activation (the HF name)
# and crash with AttributeError in engine init. Hardware-agnostic.
def test_detect_gemma2_missing_hidden_act_blocks(tmp_path):
    m = tmp_path / "gemma2_bad"
    _write_config(
        m,
        model_type="gemma2",
        architectures=["Gemma2ForCausalLM"],
        max_position_embeddings=8192,
        hidden_activation="gelu_pytorch_tanh",
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "Gemma2" in reason
    assert "hidden_act" in reason


def test_detect_gemma2_arch_only_missing_hidden_act_blocks(tmp_path):
    m = tmp_path / "gemma2_arch_only"
    _write_config(
        m,
        architectures=["Gemma2ForCausalLM"],
        max_position_embeddings=8192,
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "Gemma2" in reason


def test_detect_gemma2_with_hidden_act_ok(tmp_path):
    m = tmp_path / "gemma2_ok"
    _write_config(
        m,
        model_type="gemma2",
        architectures=["Gemma2ForCausalLM"],
        max_position_embeddings=8192,
        hidden_act="gelu_pytorch_tanh",
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_gemma2_hidden_act_in_text_config_ok(tmp_path):
    m = tmp_path / "gemma2_text_cfg"
    _write_config(
        m,
        model_type="gemma2",
        architectures=["Gemma2ForCausalLM"],
        max_position_embeddings=8192,
        text_config={"hidden_act": "gelu_pytorch_tanh"},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_non_gemma2_missing_hidden_act_not_blocked(tmp_path):
    # Only gemma2 reads config.hidden_act unconditionally in sglang; other
    # model types that omit hidden_act must not be caught by this gate.
    m = tmp_path / "llama_no_act"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_corrupt_config_blocks(tmp_path):
    m = tmp_path / "corrupt"
    m.mkdir(parents=True, exist_ok=True)
    (m / "config.json").write_text("{not valid json", encoding="utf-8")
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "unparseable" in reason


def test_detect_absent_config_not_blocked(tmp_path):
    m = tmp_path / "no_config"
    m.mkdir(parents=True, exist_ok=True)
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_rope_in_nested_text_config(tmp_path):
    m = tmp_path / "nested"
    m.mkdir(parents=True, exist_ok=True)
    (m / "config.json").write_text(
        json.dumps({"text_config": {"rope_theta": 10000.0}}),
        encoding="utf-8",
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None


def test_detect_dual_chunk_blocks_on_amd(tmp_path):
    m = tmp_path / "dual_chunk_amd"
    _write_config(
        m,
        model_type="qwen2",
        max_position_embeddings=1010000,
        dual_chunk_attention_config={"chunk_size": 262144},
    )
    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "dual_chunk" in reason


def test_detect_dual_chunk_not_blocked_off_amd(tmp_path):
    m = tmp_path / "dual_chunk_non_amd"
    _write_config(
        m,
        model_type="qwen2",
        max_position_embeddings=1010000,
        dual_chunk_attention_config={"chunk_size": 262144},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def _write_quant_config(model_dir: Path, payload: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "hf_quant_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_detect_modelopt_fp8_blocks_on_amd(tmp_path):
    """ModelOpt FP8 (declared in hf_quant_config.json) has no ROCm loader."""
    m = tmp_path / "modelopt_fp8"
    _write_config(m, model_type="llama", max_position_embeddings=8192)
    _write_quant_config(
        m,
        {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"},
        },
    )
    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "modelopt" in reason.lower() or "FP8" in reason


def test_detect_nvfp4_blocks_on_amd(tmp_path):
    m = tmp_path / "nvfp4"
    _write_config(m, model_type="llama", max_position_embeddings=8192)
    _write_quant_config(
        m,
        {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "NVFP4"},
        },
    )
    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "NVFP4" in reason or "nvfp4" in reason.lower()


def test_detect_bitsandbytes_blocks_on_amd(tmp_path):
    """bitsandbytes declared in config.json.quantization_config; CUDA-only kernels."""
    m = tmp_path / "bnb"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        quantization_config={"quant_method": "bitsandbytes"},
    )
    reason = cli._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "bitsandbytes" in reason.lower()


def test_detect_modelopt_fp8_not_blocked_off_amd(tmp_path):
    """The same checkpoint can still run on a vendor (NVIDIA) engine."""
    m = tmp_path / "modelopt_fp8_nv"
    _write_config(m, model_type="llama", max_position_embeddings=8192)
    _write_quant_config(
        m,
        {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8"},
        },
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_amd_native_fp8_not_blocked(tmp_path):
    """AMD Quark / compressed-tensors FP8 is ROCm-native; must NOT be blocked."""
    m = tmp_path / "quark_fp8"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        quantization_config={"quant_method": "fp8"},
    )
    assert cli._detect_incompatible_model_config(str(m), gpu_type="mi300x") is None


def _write_safetensors_header(model_dir: Path, tensors: dict) -> None:
    header = json.dumps(tensors).encode("utf-8")
    with (model_dir / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def test_detect_vocab_shape_mismatch_blocks(tmp_path):
    m = tmp_path / "qwen_vocab_mismatch"
    _write_config(
        m,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=4096,
        vocab_size=152064,
    )
    _write_safetensors_header(
        m,
        {
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [151936, 1536],
                "data_offsets": [0, 0],
            },
            "lm_head.weight": {
                "dtype": "BF16",
                "shape": [151936, 1536],
                "data_offsets": [0, 0],
            },
        },
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "vocab_size=152064" in reason
    assert "151936" in reason


def test_detect_vocab_shape_match_not_blocked(tmp_path):
    # config vocab_size matches the weight output dim -> must NOT block.
    m = tmp_path / "qwen_vocab_ok"
    _write_config(
        m,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=4096,
        vocab_size=151936,
    )
    _write_safetensors_header(
        m,
        {
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [151936, 1536],
                "data_offsets": [0, 0],
            },
        },
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_detect_vocab_shape_padded_not_blocked(tmp_path):
    # actual > config vocab_size -> commonly a padded embedding (rounded up to
    # an alignment / TP boundary while config keeps the unpadded value). The
    # framework handles padding, so preflight must NOT skip such a checkpoint.
    m = tmp_path / "qwen_vocab_padded"
    _write_config(
        m,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=4096,
        vocab_size=151936,
    )
    _write_safetensors_header(
        m,
        {
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [152064, 1536],  # padded up from 151936
                "data_offsets": [0, 0],
            },
            "lm_head.weight": {
                "dtype": "BF16",
                "shape": [152064, 1536],
                "data_offsets": [0, 0],
            },
        },
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_read_safetensors_header_parses_and_rejects(tmp_path):
    # Valid header round-trips; truncated / oversized headers return None.
    good = tmp_path / "model.safetensors"
    _write_safetensors_header(tmp_path, {"x.weight": {"shape": [4, 4]}})
    assert cli._read_safetensors_header(good) == {"x.weight": {"shape": [4, 4]}}

    truncated = tmp_path / "trunc.safetensors"
    truncated.write_bytes(struct.pack("<Q", 999) + b"{")  # claims 999, has 1
    assert cli._read_safetensors_header(truncated) is None

    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01\x02")  # < 8-byte length prefix
    assert cli._read_safetensors_header(short) is None


# ---------------------------------------------------------------------------
# Gemma2 detection helpers (model_config_utils)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("gemma2", True),
        ("gemma2-9b", True),
        ("google-gemma-2-9b-it", True),
        ("gemma_2_2b", True),
        ("llama-gemma2-test", True),
        ("gemma-3-12b", False),
        ("gemma3-12b", False),
        ("gemma25", False),
        ("notgemma2", False),
        ("mygemma2", False),
        ("gemma12-model", False),
        ("", False),
    ],
)
def test_path_looks_like_gemma2(name, expected):
    from hyperloom.inference_optimizer import model_config_utils as mcu

    assert mcu._path_looks_like_gemma2(name) is expected


def test_model_is_gemma2_falls_back_to_path_on_residual_config(tmp_path):
    # config.json present but empty (no model_type/architectures) -> the path
    # heuristic decides; a gemma-2 path is still detected.
    from hyperloom.inference_optimizer import model_config_utils as mcu

    m = tmp_path / "google-gemma-2-9b-it"
    m.mkdir()
    (m / "config.json").write_text("{}", encoding="utf-8")
    assert mcu._model_is_gemma2(str(m)) is True


def test_model_is_gemma2_trusts_identified_non_gemma_config(tmp_path):
    # config clearly identifies llama; even a gemma-2 path must NOT override it.
    from hyperloom.inference_optimizer import model_config_utils as mcu

    m = tmp_path / "gemma-2-distill-llama"
    m.mkdir()
    (m / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    assert mcu._model_is_gemma2(str(m)) is False


def test_model_is_gemma2_detects_nested_text_config(tmp_path):
    from hyperloom.inference_optimizer import model_config_utils as mcu

    m = tmp_path / "wrapper"
    m.mkdir()
    (m / "config.json").write_text(
        json.dumps(
            {
                "model_type": "wrapper",
                "text_config": {"model_type": "gemma2"},
            }
        ),
        encoding="utf-8",
    )
    assert mcu._model_is_gemma2(str(m)) is True


def test_detect_unregistered_custom_config_blocks(tmp_path):
    m = tmp_path / "kimi_k2"
    _write_config(
        m,
        model_type="kimi_k2",
        max_position_embeddings=131072,
        auto_map={"AutoConfig": "configuration_deepseek.DeepseekV3Config"},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "kimi_k2" in reason


def test_detect_custom_automap_known_type_not_blocked(tmp_path):
    m = tmp_path / "known_automap"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        auto_map={"AutoConfig": "configuration_custom.CustomConfig"},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


# ---------------------------------------------------------------------------
# _preflight_model_config_compat — persistence + return contract
# ---------------------------------------------------------------------------
def test_preflight_blocks_and_persists(tmp_path, monkeypatch):
    model = tmp_path / "bad"
    _write_config(model, model_type="x", rope_scaling={"factor": 2.0})
    sd = tmp_path / "session"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_config_incompatible"
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"
    breakdown = json.loads((sd / "session_breakdown.json").read_text())
    assert breakdown["session"]["stop_reason"] == "model_config_incompatible"


def test_preflight_passes_for_healthy_model(tmp_path, monkeypatch):
    model = tmp_path / "good"
    _write_config(model, model_type="llama", max_position_embeddings=8192)
    sd = tmp_path / "session_ok"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


def test_preflight_blocks_amd_unsupported_arch_from_args_gpu_type(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "deepseek_v32"
    _write_config(
        model,
        model_type="deepseek_v32",
        architectures=["DeepseekV32ForCausalLM"],
        max_position_embeddings=163840,
        rope_scaling={"type": "yarn", "factor": 40.0},
    )
    sd = tmp_path / "session_amd_arch"
    _seed_state(sd, monkeypatch)

    assert (
        cli._preflight_model_config_compat(
            _args(str(model), gpu_type="mi300x"),
            sd,
        )
        is True
    )
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_config_incompatible"


def test_stop_reason_is_canonical_vocab():
    from hyperloom.orchestrator.phases.machine_state import (
        STOP_REASON_VOCAB,
        is_valid_stop_reason,
    )

    assert "model_config_incompatible" in STOP_REASON_VOCAB
    assert is_valid_stop_reason("model_config_incompatible")


def test_preflight_persists_under_strict_env(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "1")
    model = tmp_path / "bad_strict"
    _write_config(model, model_type="x", rope_parameters={"factor": 2.0})
    sd = tmp_path / "session_strict"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"


# ---------------------------------------------------------------------------
# Private / third-party quantization formats (paroquant, MLX, mxtq, GGUF)
# ---------------------------------------------------------------------------
def test_private_quant_paroquant_blocks(tmp_path):
    m = tmp_path / "paro"
    _write_config(
        m, model_type="qwen3_5", max_position_embeddings=32768,
        quantization_config={"quant_method": "paroquant", "bits": 4,
                             "group_size": 128, "krot": 8},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "paroquant" in reason


def test_private_quant_mlx_affine_blocks(tmp_path):
    # MTPLX / MLX 8-bit affine: mode set, no quant_method.
    m = tmp_path / "mlx_affine"
    _write_config(
        m, model_type="qwen3_5", max_position_embeddings=32768,
        quantization_config={"bits": 8, "group_size": 64, "mode": "affine"},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "MLX" in reason


def test_private_quant_no_method_blocks(tmp_path):
    # quantization_config carries bits/group_size but no quant_method/mode;
    # sglang raises "Unknown quantization method: ''" in engine init.
    m = tmp_path / "no_method"
    _write_config(
        m, model_type="llama", max_position_embeddings=32768,
        quantization_config={"group_size": 64, "bits": 8},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "quant_method" in reason


def test_private_quant_mxtq_blocks(tmp_path):
    m = tmp_path / "mxtq"
    _write_config(
        m, model_type="qwen3_5_moe", max_position_embeddings=32768,
        quantization_config={"weight_format": "mxtq", "method": "affine",
                             "group_size": 64},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "mxtq" in reason


def test_private_quant_empty_method_blocks(tmp_path):
    m = tmp_path / "empty_quant_method"
    _write_config(
        m,
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        max_position_embeddings=4096,
        quantization_config={"quant_method": ""},
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "quant_method is empty" in reason


def test_private_quant_mlx_weights_index_blocks(tmp_path):
    # JANG/MLX often declares no quant config; the tell is '.biases'/'.scales'.
    m = tmp_path / "mlx_weights"
    _write_config(m, model_type="qwen3_5", max_position_embeddings=32768)
    (m / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            "lm_head.biases": "model-00001.safetensors",
            "lm_head.scales": "model-00001.safetensors",
            "model.embed_tokens.weight": "model-00001.safetensors",
        }}), encoding="utf-8",
    )
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "MLX" in reason


def test_private_quant_gguf_only_blocks(tmp_path):
    m = tmp_path / "gguf_only"
    _write_config(m, model_type="qwen3_5_moe", max_position_embeddings=32768)
    (m / "model-TQ3_4S.gguf").write_text("dummy", encoding="utf-8")
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "GGUF" in reason


def test_peft_adapter_only_checkpoint_blocks(tmp_path):
    m = tmp_path / "adapter_only"
    _write_config(m, model_type="qwen2", max_position_embeddings=32768)
    (m / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen2-7B"}),
        encoding="utf-8",
    )
    (m / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            "base_model.model.lm_head.lora_A.default.weight": "adapter_model.safetensors",
            "base_model.model.lm_head.lora_B.default.weight": "adapter_model.safetensors",
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": (
                "adapter_model.safetensors"
            ),
        }}),
        encoding="utf-8",
    )

    reason = cli._detect_incompatible_model_config(str(m))

    assert reason is not None
    assert "PEFT/LoRA adapter" in reason
    assert "base_model.model.lm_head.base_layer.weight" in reason


def test_merged_peft_like_checkpoint_with_base_weights_not_blocked(tmp_path):
    m = tmp_path / "merged_adapter"
    _write_config(m, model_type="qwen2", max_position_embeddings=32768)
    (m / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            "model.embed_tokens.weight": "model-00001.safetensors",
            "lm_head.weight": "model-00001.safetensors",
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": (
                "model-00001.safetensors"
            ),
        }}),
        encoding="utf-8",
    )

    assert cli._detect_incompatible_model_config(str(m)) is None


def test_standard_quant_fp8_not_blocked(tmp_path):
    m = tmp_path / "fp8"
    _write_config(
        m, model_type="qwen3", max_position_embeddings=32768,
        quantization_config={"quant_method": "fp8"},
    )
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_standard_quant_awq_gptq_not_blocked(tmp_path):
    for method in ("awq", "gptq", "compressed-tensors"):
        m = tmp_path / f"std_{method.replace('-', '_')}"
        _write_config(
            m, model_type="qwen3", max_position_embeddings=32768,
            quantization_config={"quant_method": method},
        )
        assert cli._detect_incompatible_model_config(str(m)) is None, method


def test_gguf_with_safetensors_not_blocked(tmp_path):
    # A normal safetensors model that merely also ships a .gguf must still pass.
    m = tmp_path / "gguf_plus_st"
    _write_config(m, model_type="qwen3", max_position_embeddings=32768)
    (m / "model.gguf").write_text("dummy", encoding="utf-8")
    (m / "model.safetensors").write_text("dummy", encoding="utf-8")
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_gguf_with_auxiliary_bin_still_blocks(tmp_path):
    # An arbitrary .bin next to GGUF is not an HF-loadable weight file.
    m = tmp_path / "gguf_plus_aux_bin"
    _write_config(m, model_type="qwen3", max_position_embeddings=32768)
    (m / "model.gguf").write_text("dummy", encoding="utf-8")
    (m / "tokenizer_cache.bin").write_text("dummy", encoding="utf-8")
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None and "GGUF" in reason


def test_gguf_with_pytorch_model_bin_not_blocked(tmp_path):
    # HF PyTorch weights are loadable by the default framework loaders.
    m = tmp_path / "gguf_plus_hf_bin"
    _write_config(m, model_type="qwen3", max_position_embeddings=32768)
    (m / "model.gguf").write_text("dummy", encoding="utf-8")
    (m / "pytorch_model.bin").write_text("dummy", encoding="utf-8")
    assert cli._detect_incompatible_model_config(str(m)) is None


def test_llama_sentencepiece_without_metadata_blocks(tmp_path):
    m = tmp_path / "llama_sentencepiece_gap"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        max_position_embeddings=4096,
    )
    (m / "tokenizer.model").write_text("dummy", encoding="utf-8")
    (m / "pytorch_model.bin").write_text("dummy", encoding="utf-8")
    reason = cli._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "tokenizer_config.json" in reason
    assert "HFValidationError" in reason


def test_llama_sentencepiece_with_tokenizer_config_ok(tmp_path):
    m = tmp_path / "llama_sentencepiece_ok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        max_position_embeddings=4096,
    )
    (m / "tokenizer.model").write_text("dummy", encoding="utf-8")
    (m / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (m / "pytorch_model.bin").write_text("dummy", encoding="utf-8")
    assert cli._detect_incompatible_model_config(str(m)) is None


# ---------------------------------------------------------------------------
# Langfuse parity on fail-fast — the pre-flight gates exit before
# coordinator.run()'s finally (the normal Langfuse flush point), so each must
# push the breakdown to Langfuse itself or early-aborted sessions land on disk
# (collector) but never in Langfuse.
# ---------------------------------------------------------------------------
def _spy_langfuse_emit(monkeypatch) -> dict[str, list]:
    calls: dict[str, list] = {"flush": [], "patch": [], "record": []}
    from hyperloom.inference_optimizer import breakdown as bd
    from hyperloom.orchestrator.trace import langfuse_emitter as lfe

    monkeypatch.setattr(
        lfe, "flush_session", lambda sd: calls["flush"].append(Path(sd))
    )
    monkeypatch.setattr(
        bd, "patch_breakdown_langfuse", lambda sd: calls["patch"].append(Path(sd))
    )
    monkeypatch.setattr(
        lfe,
        "record_session_breakdown",
        lambda sd, *a, **k: calls["record"].append(Path(sd)),
    )
    return calls


def test_model_config_fail_fast_emits_to_langfuse(tmp_path, monkeypatch):
    model = tmp_path / "bad_lf"
    _write_config(model, model_type="x", rope_scaling={"factor": 2.0})
    sd = tmp_path / "session_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    # written to disk AND pushed to Langfuse in flush -> patch -> record order.
    assert (sd / "session_breakdown.json").exists()
    assert calls["flush"] == [sd]
    assert calls["patch"] == [sd]
    assert calls["record"] == [sd]


def test_unsupported_model_fail_fast_emits_to_langfuse(tmp_path, monkeypatch):
    model = tmp_path / "vlm_lf"
    _write_config(
        model,
        model_type="llava",
        architectures=["LlavaForConditionalGeneration"],
        max_position_embeddings=4096,
    )
    sd = tmp_path / "session_vlm_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True
    assert calls["record"] == [sd]


def test_context_window_fail_fast_emits_to_langfuse(tmp_path, monkeypatch):
    model = tmp_path / "ctx_lf"
    _write_config(model, model_type="llama", max_position_embeddings=128)
    sd = tmp_path / "session_ctx_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    # ISL+OSL (1024+1024) + headroom >> 128 -> fail fast.
    assert cli._preflight_context_window(_args(str(model)), sd) is True
    assert calls["record"] == [sd]


def test_emit_to_langfuse_is_best_effort(tmp_path, monkeypatch):
    # A Langfuse outage must never turn a clean fail-fast into a crash, nor
    # mask the persisted stop reason.
    model = tmp_path / "bad_raise"
    _write_config(model, model_type="x", rope_scaling={"factor": 2.0})
    sd = tmp_path / "session_raise"
    _seed_state(sd, monkeypatch)

    from hyperloom.orchestrator.trace import langfuse_emitter as lfe

    def _boom(*a, **k):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr(lfe, "flush_session", _boom)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"


def test_healthy_model_does_not_emit_to_langfuse(tmp_path, monkeypatch):
    # A passing pre-flight must not touch Langfuse — the normal end-of-session
    # path owns that for runs that actually start.
    model = tmp_path / "good_lf"
    _write_config(model, model_type="llama", max_position_embeddings=8192)
    sd = tmp_path / "session_good_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is False
    assert calls["record"] == []


def test_declared_standard_quant_with_scales_index_not_blocked(tmp_path):
    # AWQ/GPTQ/compressed-tensors legitimately ship '.scales'/'.biases' tensors;
    # a declared supported quant_method must NOT be misread as MLX (the weight-
    # index tell only applies to checkpoints with NO quant_method declared).
    for method in ("awq", "gptq", "compressed-tensors"):
        m = tmp_path / f"std_scales_{method.replace('-', '_')}"
        _write_config(
            m, model_type="qwen3", max_position_embeddings=32768,
            quantization_config={"quant_method": method},
        )
        (m / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "model.layers.0.self_attn.q_proj.scales":
                    "model-00001.safetensors",
                "model.layers.0.self_attn.q_proj.biases":
                    "model-00001.safetensors",
                "model.embed_tokens.weight": "model-00001.safetensors",
            }}), encoding="utf-8",
        )
        assert cli._detect_incompatible_model_config(str(m)) is None, method
