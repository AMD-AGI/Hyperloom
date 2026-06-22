# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/model_compat.py.

Covers the shared structural-compatibility predicate ``unrunnable_reason`` (the
config rules enforced both offline by ``filter_candidates.py`` and online by
``optimize_submit.py``) and the HF ``hf_gated`` probe.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import model_compat  # noqa: E402


# ── unrunnable_reason: per-rule hits ────────────────────────────────────────


def _reason(cfg, **kw):
    r = model_compat.unrunnable_reason(cfg, **kw)
    return r[0] if r else None


def test_multimodal_by_architecture():
    assert _reason({"architectures": ["Qwen2_5_VLForConditionalGeneration"],
                    "max_position_embeddings": 128000}) == "multimodal"


def test_multimodal_by_model_type():
    assert _reason({"architectures": ["LlavaForCausalLM"], "model_type": "llava",
                    "max_position_embeddings": 4096}) == "multimodal"


def test_multimodal_by_vision_config():
    assert _reason({"architectures": ["FooForCausalLM"], "vision_config": {"x": 1},
                    "max_position_embeddings": 4096}) == "multimodal"


def test_bare_for_conditional_generation_without_vision_is_kept():
    # Text-only MoE that merely use the *ForConditionalGeneration suffix
    # (no vision_config) must NOT be filtered as multimodal.
    assert _reason({"architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "model_type": "qwen3_5_moe",
                    "max_position_embeddings": 262144}) is None
    assert _reason({"architectures": ["KimiK25ForConditionalGeneration"],
                    "model_type": "kimi_k25",
                    "max_position_embeddings": 131072}) is None


def test_for_conditional_generation_with_vision_config_is_multimodal():
    assert _reason({"architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "model_type": "qwen3_5_moe", "vision_config": {"x": 1},
                    "max_position_embeddings": 262144}) == "multimodal"


def test_short_ctx_at_threshold():
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "max_position_embeddings": 2048}) == "short_ctx"


def test_short_ctx_nested_text_config():
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "text_config": {"max_position_embeddings": 1024}}) == "short_ctx"


def test_short_ctx_above_threshold_is_kept():
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "max_position_embeddings": 2049}) is None


def test_phi3_longrope():
    assert _reason({"architectures": ["Phi3ForCausalLM"], "model_type": "phi3",
                    "max_position_embeddings": 131072,
                    "rope_scaling": {"type": "longrope"}}) == "phi3_longrope"


def test_dual_chunk_attention():
    assert _reason({"architectures": ["Qwen2ForCausalLM"],
                    "max_position_embeddings": 1010000,
                    "dual_chunk_attention_config": {"chunk": 1}}) == "dual_chunk_attention"


def test_gemma2():
    assert _reason({"architectures": ["Gemma2ForCausalLM"], "model_type": "gemma2",
                    "max_position_embeddings": 8192}) == "gemma2"


def test_modelopt_fp8():
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "max_position_embeddings": 8192,
                    "quantization_config": {"quant_method": "modelopt"}}) == "modelopt_fp8"


def test_flashinfer_backend():
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "max_position_embeddings": 8192,
                    "attn_implementation": "flashinfer"}) == "attn_backend"


# ── unsupported serving registry (config-based, GPU-independent) ────────────


def test_unsupported_arch_glm_moe_dsa_by_model_type():
    assert _reason({"architectures": ["GlmMoeDsaForCausalLM"],
                    "model_type": "glm_moe_dsa",
                    "max_position_embeddings": 131072}) == "unsupported_arch"


def test_unsupported_arch_deepseek_v32_by_model_type():
    assert _reason({"architectures": ["DeepseekV32ForCausalLM"],
                    "model_type": "deepseek_v32", "max_position_embeddings": 163840,
                    "quantization_config": {"quant_method": "fp8"}}) == "unsupported_arch"


def test_unsupported_arch_matched_by_architecture_fallback():
    # No/blank model_type -> architecture fallback still catches it.
    assert _reason({"architectures": ["GlmMoeDsaForCausalLM"],
                    "max_position_embeddings": 131072}) == "unsupported_arch"


def test_unsupported_arch_is_gpu_independent():
    # Registry rules are config-based: hit on any gpu_type and with none.
    cfg = {"architectures": ["DeepseekV32ForCausalLM"], "model_type": "deepseek_v32",
           "max_position_embeddings": 163840}
    assert _reason(cfg, gpu_type="MI300X") == "unsupported_arch"
    assert _reason(cfg, gpu_type="mi355x") == "unsupported_arch"
    assert _reason(cfg) == "unsupported_arch"


def test_supported_glm4_moe_and_deepseek_v3_are_kept():
    # GLM-4.7 (glm4_moe) and DeepSeek-V3 (deepseek_v3) must NOT match the
    # unsupported-registry rule.
    assert _reason({"architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe",
                    "max_position_embeddings": 131072}) is None
    assert _reason({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3",
                    "max_position_embeddings": 163840}) is None


def test_deepseek_v4_flash_registry_is_kept():
    # V4-Flash shares DeepseekV4ForCausalLM with the full V4 but is supported;
    # the registry rule must not list deepseek_v4 (full V4 is name-filtered on
    # MI300X instead).
    assert _reason({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
                    "max_position_embeddings": 163840,
                    "quantization_config": {"quant_method": "fp8"}}) is None


# ── MI300X gpu-specific rules (gpu_type=MI300X only) ────────────────────────


_BASE_CFG = {"architectures": ["LlamaForCausalLM"], "max_position_embeddings": 8192}


def _fp4_cfg(tag, *, field="quant_method"):
    cfg = dict(_BASE_CFG)
    cfg["quantization_config"] = {field: tag}
    return cfg


@pytest.mark.parametrize("tag", ["mxfp4", "MXFP4", "nvfp4"])
def test_fp4_unsupported_on_mi300x(tag):
    assert _reason(_fp4_cfg(tag), gpu_type="MI300X") == "fp4_unsupported"


@pytest.mark.parametrize("gpu", ["mi355x", "MI355X", "mi325x"])
def test_fp4_kept_on_non_mi300x(gpu):
    assert _reason(_fp4_cfg("mxfp4"), gpu_type=gpu) is None


def test_fp4_kept_without_gpu_type():
    # Backward-compatible default: no gpu_type -> gpu rules disabled.
    assert _reason(_fp4_cfg("mxfp4")) is None


def test_fp8_kept_on_mi300x():
    assert _reason(_fp4_cfg("fp8"), gpu_type="MI300X") is None


@pytest.mark.parametrize("repo", [
    "deepseek-ai/DeepSeek-V4",
    "deepseek-ai/DeepSeek-V4-0501",
    "deepseek-ai/DeepSeek-V4.1",
    "zai-org/GLM-5",
    "zai-org/GLM-5.1",
    "zai-org/GLM5-Air",
])
def test_unsupported_model_on_mi300x(repo):
    assert _reason(_BASE_CFG, repo=repo, gpu_type="MI300X") == "mi300x_unsupported_model"


@pytest.mark.parametrize("repo", [
    "deepseek-ai/DeepSeek-V4-Flash",   # explicitly exempt
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-Prover-V2-671B",
    "zai-org/GLM-4.7-FP8",             # GLM-4.7 must not match GLM-5
    "zai-org/GLM-512B",                # digit-guard: GLM-51x is not GLM-5
    "meta-llama/Llama-3.1-8B-Instruct",
])
def test_model_kept_on_mi300x(repo):
    assert _reason(_BASE_CFG, repo=repo, gpu_type="MI300X") is None


def test_unsupported_model_kept_on_non_mi300x():
    # DeepSeek-V4 / GLM-5 run fine on MI355X; rule is MI300X-only.
    assert _reason(_BASE_CFG, repo="deepseek-ai/DeepSeek-V4", gpu_type="mi355x") is None
    assert _reason(_BASE_CFG, repo="zai-org/GLM-5", gpu_type="mi355x") is None


def test_unsupported_model_kept_without_gpu_type():
    assert _reason(_BASE_CFG, repo="deepseek-ai/DeepSeek-V4") is None


def test_v4_flash_exempt_even_without_whitelist():
    # The allow-regex protects V4-Flash on any path, not just the whitelist.
    assert model_compat.mi300x_blocked_model("deepseek-ai/DeepSeek-V4-Flash") == ""
    assert model_compat.mi300x_blocked_model("deepseek-ai/DeepSeek-V4") == "DeepSeek-V4"


def test_normal_model_is_kept():
    assert _reason({"architectures": ["MistralForCausalLM"], "model_type": "mistral",
                    "max_position_embeddings": 32768}) is None


def test_non_dict_config_is_kept():
    assert model_compat.unrunnable_reason(None) is None
    assert model_compat.unrunnable_reason("nope") is None


def test_whitelist_exempts_otherwise_filtered_model():
    cfg = {"architectures": ["Qwen2_5_VLForConditionalGeneration"],
           "vision_config": {"x": 1}, "max_position_embeddings": 128000}
    wl = {"org/keep-me"}
    # Without whitelist -> filtered; whitelisted repo -> exempt (None).
    assert model_compat.unrunnable_reason(cfg, repo="org/keep-me") == ("multimodal", "arch=Qwen2_5_VLForConditionalGeneration")
    assert model_compat.unrunnable_reason(cfg, repo="org/keep-me", whitelist=wl) is None
    assert model_compat.unrunnable_reason(cfg, repo="org/other", whitelist=wl) is not None


# ── missing_tokenizer (needs local model dir) ───────────────────────────────


def _mk_dir(tmp_path, files):
    for name in files:
        (tmp_path / name).write_bytes(b"x")
    return str(tmp_path)


def test_missing_tokenizer_weights_without_tokenizer(tmp_path):
    mdir = _mk_dir(tmp_path, ["config.json", "model.safetensors"])
    assert _reason({"architectures": ["Qwen2ForCausalLM"],
                    "max_position_embeddings": 32768}, model_dir=mdir) == "missing_tokenizer"


def test_tokenizer_present_is_kept(tmp_path):
    mdir = _mk_dir(tmp_path, ["config.json", "model.safetensors", "tokenizer.json"])
    assert _reason({"architectures": ["Qwen2ForCausalLM"],
                    "max_position_embeddings": 32768}, model_dir=mdir) is None


def test_no_weights_does_not_flag_missing_tokenizer(tmp_path):
    # Partial cache (no weights yet) must not be flagged as missing_tokenizer.
    mdir = _mk_dir(tmp_path, ["config.json"])
    assert _reason({"architectures": ["Qwen2ForCausalLM"],
                    "max_position_embeddings": 32768}, model_dir=mdir) is None


def test_no_model_dir_skips_tokenizer_check():
    assert _reason({"architectures": ["Qwen2ForCausalLM"],
                    "max_position_embeddings": 32768}) is None


# ── hf_gated (network probe, mocked) ─────────────────────────────────────────


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _patch_urlopen(monkeypatch, payload=None, error=None):
    def fake(req, timeout=None):
        if error is not None:
            raise error
        return _Resp(json.dumps(payload).encode())
    monkeypatch.setattr(model_compat.urllib.request, "urlopen", fake)


def test_hf_gated_no_tokens_returns_none():
    assert model_compat.hf_gated("org/model", []) is None


def test_hf_gated_auto(monkeypatch):
    _patch_urlopen(monkeypatch, payload={"gated": "auto"})
    assert model_compat.hf_gated("org/model", ["hf_x"]) == "gated"


def test_hf_gated_false(monkeypatch):
    _patch_urlopen(monkeypatch, payload={"gated": False})
    assert model_compat.hf_gated("org/model", ["hf_x"]) is None


def test_hf_gated_not_found(monkeypatch):
    err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
    _patch_urlopen(monkeypatch, error=err)
    assert model_compat.hf_gated("org/model", ["hf_x"]) == "not_found"
