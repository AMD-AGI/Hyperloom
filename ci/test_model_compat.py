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
import filter_candidates  # noqa: E402


def test_load_whitelist_missing_or_malformed_returns_empty(tmp_path):
    model_compat.load_whitelist.cache_clear()
    assert model_compat.load_whitelist(tmp_path / "missing.json") == frozenset()
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    model_compat.load_whitelist.cache_clear()
    assert model_compat.load_whitelist(bad) == frozenset()


def test_load_whitelist_collects_repo_ids(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps({"candidates": [{"repo_id": "org/a"}, {"repo_id": ""}, {"other": "x"}]}),
        encoding="utf-8",
    )
    model_compat.load_whitelist.cache_clear()
    assert model_compat.load_whitelist(path) == frozenset({"org/a"})


def test_http_urlopen_rejects_non_http_scheme():
    try:
        model_compat._http_urlopen("file:///tmp/not-allowed", timeout=1)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported URL scheme" in str(exc)


def test_http_urlopen_allows_https_url(monkeypatch):
    def fake_urlopen(url_or_req, timeout=0):
        url = getattr(url_or_req, "full_url", url_or_req)
        assert url == "https://huggingface.co/api/test"
        return io.BytesIO(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", fake_urlopen)
    with model_compat._http_urlopen("https://huggingface.co/api/test", timeout=1) as resp:
        assert json.loads(resp.read().decode("utf-8")) == {"ok": True}


def test_has_weights_and_tokenizer_unknown_dir_is_lenient(tmp_path):
    missing = tmp_path / "missing"
    assert model_compat.has_weights(missing) is False
    assert model_compat.has_tokenizer(missing) is True


# ── unrunnable_reason: per-rule hits ────────────────────────────────────────


def _reason(cfg, **kw):
    r = model_compat.unrunnable_reason(cfg, **kw)
    return r[0] if r else None


def test_model_compat_http_urlopen_rejects_non_http_scheme():
    try:
        model_compat._http_urlopen("file:///tmp/not-allowed", timeout=1)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unsupported URL scheme" in str(exc)


def test_model_compat_http_urlopen_allows_https_url(monkeypatch):
    def fake_urlopen(url_or_req, timeout=0):
        url = getattr(url_or_req, "full_url", url_or_req)
        assert url == "https://huggingface.co/api/test"
        return io.BytesIO(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", fake_urlopen)
    with model_compat._http_urlopen("https://huggingface.co/api/test", timeout=1) as resp:
        assert json.loads(resp.read().decode("utf-8")) == {"ok": True}


def test_multimodal_by_architecture():
    assert _reason({"architectures": ["Qwen2_5_VLForConditionalGeneration"],
                    "max_position_embeddings": 128000}) == "multimodal"


def test_multimodal_by_model_type():
    assert _reason({"architectures": ["LlavaForCausalLM"], "model_type": "llava",
                    "max_position_embeddings": 4096}) == "multimodal"


def test_multimodal_by_vision_config():
    assert _reason({"architectures": ["FooForCausalLM"], "vision_config": {"x": 1},
                    "max_position_embeddings": 4096}) == "multimodal"


def test_non_llm_diffusers_repo_filtered_without_config():
    assert _reason(None, repo="black-forest-labs/FLUX.1-dev") == "non_text_generation"


def test_filter_candidates_repo_gate_without_config(tmp_path, monkeypatch):
    monkeypatch.setattr(filter_candidates, "MODELS_DIR", str(tmp_path))

    r = filter_candidates.classify_local("black-forest-labs/FLUX.1-dev")

    assert r is not None
    assert r[0] == "non_text_generation"


def test_filter_candidates_classify_local_config_and_bad_json(tmp_path, monkeypatch):
    monkeypatch.setattr(filter_candidates, "MODELS_DIR", str(tmp_path))
    short_dir = tmp_path / "org-short"
    short_dir.mkdir()
    (short_dir / "config.json").write_text(
        json.dumps({
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 2048,
        }),
        encoding="utf-8",
    )
    bad_dir = tmp_path / "org-bad-json"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text("{", encoding="utf-8")

    assert filter_candidates.classify_local("org/short")[0] == "short_ctx"
    assert filter_candidates.classify_local("org/bad-json") is None


def test_filter_candidates_tokens_slug_and_gated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKENS", "hf_a, hf_b")
    monkeypatch.setenv("HF_TOKEN", "hf_b")
    monkeypatch.setenv("HF_TOKEN_2", "hf_c")
    monkeypatch.setattr(
        filter_candidates,
        "GATED_CACHE",
        str(tmp_path / "gated_cache.tsv"),
    )
    (tmp_path / "gated_cache.tsv").write_text(
        "cached/repo\tgated\nmalformed\n",
        encoding="utf-8",
    )

    assert filter_candidates.hf_tokens() == ["hf_a", "hf_b", "hf_c"]
    assert filter_candidates.slug("org/model") == "org-model"
    assert filter_candidates.load_gated_cache() == {"cached/repo": "gated"}


def test_filter_candidates_gated_check_all_appends_uncached(tmp_path, monkeypatch):
    cache_path = tmp_path / "gated_cache.tsv"
    cache_path.write_text("cached/repo\tok\n", encoding="utf-8")
    monkeypatch.setattr(filter_candidates, "GATED_CACHE", str(cache_path))
    monkeypatch.setattr(
        filter_candidates,
        "hf_gated",
        lambda repo: "gated" if repo == "new/gated" else None,
    )
    monkeypatch.setattr(filter_candidates.time, "sleep", lambda _seconds: None)

    cache = filter_candidates.gated_check_all(["cached/repo", "new/gated", "new/ok"])

    assert cache == {
        "cached/repo": "ok",
        "new/gated": "gated",
        "new/ok": "ok",
    }
    assert "new/gated\tgated" in cache_path.read_text(encoding="utf-8")


def test_filter_candidates_main_filters_local_and_gated(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    out_dir = tmp_path / "out"
    pool_path = tmp_path / "daily.json"
    pool_path.write_text(
        json.dumps({
            "candidates": [
                {"repo_id": "black-forest-labs/FLUX.1-dev"},
                {"repo_id": "org/ok"},
                {"repo_id": "org/gated"},
                {"repo_id": "org/whitelist"},
            ]
        }),
        encoding="utf-8",
    )
    ok_dir = models_dir / "org-ok"
    ok_dir.mkdir(parents=True)
    (ok_dir / "config.json").write_text(
        json.dumps({
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 8192,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(filter_candidates, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(filter_candidates, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(
        filter_candidates,
        "GATED_CACHE",
        str(out_dir / "gated_cache.tsv"),
    )
    monkeypatch.setattr(
        filter_candidates.model_compat,
        "load_whitelist",
        lambda: {"org/whitelist"},
    )
    monkeypatch.setattr(
        filter_candidates,
        "gated_check_all",
        lambda repos: {"org/ok": "ok", "org/gated": "gated"},
    )

    filter_candidates.main([str(pool_path)])

    filtered = json.loads((out_dir / "daily_filtered.json").read_text(encoding="utf-8"))
    assert [c["repo_id"] for c in filtered["candidates"]] == [
        "org/ok",
        "org/whitelist",
    ]
    report = (out_dir / "pool_filter_report.tsv").read_text(encoding="utf-8")
    assert "black-forest-labs/FLUX.1-dev\tnon_text_generation" in report
    assert "org/gated\tgated\tHF API" in report


def test_diffusion_substring_text_repo_is_kept():
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "max_position_embeddings": 8192},
                   repo="org/diffusion-language-model") is None


def test_bare_for_conditional_generation_without_vision_is_kept():
    # Text-only MoE with the *ForConditionalGeneration suffix (no
    # vision_config) must NOT be filtered as multimodal.
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


def test_invalid_context_value_is_ignored():
    assert _reason({"architectures": ["LlamaForCausalLM"], "max_position_embeddings": "not-int"}) is None


def test_hf_gated_retries_401_then_reports_gated(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", fake_urlopen)
    assert model_compat.hf_gated("org/model", ["hf_a", "hf_b"]) == "gated"
    assert calls["n"] == 3


def test_hf_gated_retries_429_then_succeeds(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(model_compat.time, "sleep", lambda *_a, **_k: None)

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "Too Many", {}, None)
        return _Resp(json.dumps({"gated": False}).encode())

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", fake_urlopen)
    assert model_compat.hf_gated("org/model", ["hf_a", "hf_b"]) is None
    assert calls["n"] == 2


def test_hf_gated_fail_open_on_url_error(monkeypatch):
    monkeypatch.setattr(model_compat.time, "sleep", lambda *_a, **_k: None)
    _patch_urlopen(monkeypatch, error=urllib.error.URLError("boom"))
    assert model_compat.hf_gated("org/model", ["hf_x"]) is None


def test_hf_missing_tokenizer_retries_429_then_succeeds(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(model_compat.time, "sleep", lambda *_a, **_k: None)

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "Too Many", {}, None)
        return _Resp(json.dumps(_siblings("model.safetensors", "tokenizer.json")).encode())

    monkeypatch.setattr(model_compat.urllib.request, "urlopen", fake_urlopen)
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_a", "hf_b"]) is None
    assert calls["n"] == 2


# ── unsupported serving registry (config-based, GPU-independent) ────────────


def test_unsupported_arch_glm_moe_dsa_by_model_type():
    assert _reason({"architectures": ["GlmMoeDsaForCausalLM"],
                    "model_type": "glm_moe_dsa",
                    "max_position_embeddings": 131072}) == "unsupported_arch"


def test_unsupported_arch_deepseek_v32_by_model_type():
    assert _reason({"architectures": ["DeepseekV32ForCausalLM"],
                    "model_type": "deepseek_v32", "max_position_embeddings": 163840,
                    "quantization_config": {"quant_method": "fp8"}}) == "unsupported_arch"


def test_unsupported_arch_gemma4_by_model_type():
    assert _reason({"architectures": ["Gemma4ForCausalLM"],
                    "model_type": "gemma4",
                    "max_position_embeddings": 32768}) == "unsupported_arch"


def test_empty_quant_method_filtered():
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "max_position_embeddings": 4096,
                    "quantization_config": {"quant_method": ""}}) == "quant_empty_method"


def test_unsupported_arch_qwen3_5_moe_text_via_text_config():
    # The rule must catch qwen3_5_moe_text via text_config.model_type.
    assert _reason({"architectures": ["Qwen3_5MoeForCausalLM"],
                    "model_type": "qwen3_5_moe",
                    "text_config": {"model_type": "qwen3_5_moe_text"},
                    "max_position_embeddings": 262144}) == "unsupported_arch"


def test_qwen3_5_moe_without_text_subtype_is_kept():
    # Bare qwen3_5_moe must NOT be filtered by the registry rule.
    assert _reason({"architectures": ["Qwen3_5MoeForConditionalGeneration"],
                    "model_type": "qwen3_5_moe",
                    "max_position_embeddings": 262144}) is None


def test_unsupported_arch_matched_by_architecture_fallback():
    # No/blank model_type -> architecture fallback catches it.
    assert _reason({"architectures": ["GlmMoeDsaForCausalLM"],
                    "max_position_embeddings": 131072}) == "unsupported_arch"


def test_unsupported_arch_qrwkv6_hybrid_is_filtered():
    assert _reason({"architectures": ["RWKV6Qwen2ForCausalLM"],
                    "model_type": "rwkv6qwen2",
                    "max_position_embeddings": 131072}) == "unsupported_arch"


def test_unsupported_arch_qrwkv6_hybrid_matched_by_model_type():
    assert _reason({"model_type": "rwkv6qwen2",
                    "max_position_embeddings": 131072}) == "unsupported_arch"


def test_unsupported_arch_qrwkv6_hybrid_matched_by_text_config_model_type():
    assert _reason({"model_type": "wrapper",
                    "text_config": {"model_type": "rwkv6qwen2"},
                    "max_position_embeddings": 131072}) == "unsupported_arch"


def test_unsupported_arch_is_gpu_independent():
    # Registry rules are config-based: hit on any gpu_type or none.
    cfg = {"architectures": ["DeepseekV32ForCausalLM"], "model_type": "deepseek_v32",
           "max_position_embeddings": 163840}
    assert _reason(cfg, gpu_type="MI300X") == "unsupported_arch"
    assert _reason(cfg, gpu_type="mi355x") == "unsupported_arch"
    assert _reason(cfg) == "unsupported_arch"


def test_supported_glm4_moe_and_deepseek_v3_are_kept():
    # GLM-4.7 and DeepSeek-V3 must NOT match the unsupported-registry rule.
    assert _reason({"architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe",
                    "max_position_embeddings": 131072}) is None
    assert _reason({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3",
                    "max_position_embeddings": 163840}) is None


def test_deepseek_v4_flash_registry_is_kept():
    # V4-Flash is supported and must not be listed by the registry rule.
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
    # No gpu_type -> gpu rules disabled.
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
    "deepseek-ai/DeepSeek-V4-Flash",   # exempt
    "deepseek-ai/DeepSeek-V3.2",
    "deepseek-ai/DeepSeek-Prover-V2-671B",
    "zai-org/GLM-4.7-FP8",             # must not match GLM-5
    "zai-org/GLM-512B",                # GLM-51x is not GLM-5
    "meta-llama/Llama-3.1-8B-Instruct",
])
def test_model_kept_on_mi300x(repo):
    assert _reason(_BASE_CFG, repo=repo, gpu_type="MI300X") is None


def test_unsupported_model_kept_on_non_mi300x():
    # Rule is MI300X-only.
    assert _reason(_BASE_CFG, repo="deepseek-ai/DeepSeek-V4", gpu_type="mi355x") is None
    assert _reason(_BASE_CFG, repo="zai-org/GLM-5", gpu_type="mi355x") is None


def test_unsupported_model_kept_without_gpu_type():
    assert _reason(_BASE_CFG, repo="deepseek-ai/DeepSeek-V4") is None


def test_v4_flash_exempt_even_without_whitelist():
    # The allow-regex protects V4-Flash on any path.
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
    # Whitelisted repo -> exempt.
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


def test_llama_sentencepiece_without_metadata_filtered(tmp_path):
    mdir = _mk_dir(tmp_path, ["config.json", "pytorch_model.bin", "tokenizer.model"])
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "max_position_embeddings": 4096}, model_dir=mdir) == "tokenizer_metadata_gap"


def test_llama_sentencepiece_with_tokenizer_config_kept(tmp_path):
    mdir = _mk_dir(
        tmp_path,
        ["config.json", "pytorch_model.bin", "tokenizer.model", "tokenizer_config.json"],
    )
    assert _reason({"architectures": ["LlamaForCausalLM"],
                    "model_type": "llama",
                    "max_position_embeddings": 4096}, model_dir=mdir) is None


def test_no_weights_does_not_flag_missing_tokenizer(tmp_path):
    # Partial cache (no weights) must not be flagged missing_tokenizer.
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


# ── hf_missing_tokenizer (network probe, mocked) ─────────────────────────────


def _siblings(*names):
    return {"siblings": [{"rfilename": n} for n in names]}


def test_hf_missing_tokenizer_no_tokens_returns_none():
    assert model_compat.hf_missing_tokenizer("org/model", []) is None


def test_hf_missing_tokenizer_weights_without_tokenizer(monkeypatch):
    # Weights present, no tokenizer -> missing_tokenizer.
    _patch_urlopen(monkeypatch, payload=_siblings(
        "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"))
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) == "missing_tokenizer"


def test_hf_missing_tokenizer_weights_with_tokenizer_kept(monkeypatch):
    _patch_urlopen(monkeypatch, payload=_siblings(
        "config.json", "model.safetensors", "tokenizer.json"))
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) is None


def test_hf_missing_tokenizer_bin_weights_detected(monkeypatch):
    # .bin shards count as weights.
    _patch_urlopen(monkeypatch, payload=_siblings("config.json", "pytorch_model.bin"))
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) == "missing_tokenizer"


def test_hf_missing_tokenizer_no_weights_kept(monkeypatch):
    # No weight shards -> cannot judge -> keep.
    _patch_urlopen(monkeypatch, payload=_siblings("config.json", "README.md"))
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) is None


def test_hf_missing_tokenizer_empty_siblings_kept(monkeypatch):
    _patch_urlopen(monkeypatch, payload={"siblings": []})
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) is None


def test_hf_missing_tokenizer_alt_tokenizer_file_kept(monkeypatch):
    # tokenizer.model counts as a tokenizer.
    _patch_urlopen(monkeypatch, payload=_siblings("model.safetensors", "tokenizer.model"))
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) is None


@pytest.mark.parametrize("code", [401, 403, 404])
def test_hf_missing_tokenizer_gated_or_notfound_defers(monkeypatch, code):
    # 401/403 (gated) and 404 -> fail-open None.
    err = urllib.error.HTTPError("u", code, "x", {}, None)
    _patch_urlopen(monkeypatch, error=err)
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) is None


def test_hf_missing_tokenizer_fail_open_on_error(monkeypatch):
    # Any non-HTTP fetch error -> fail-open None. Patch sleep to skip the retry wait.
    monkeypatch.setattr(model_compat.time, "sleep", lambda *_a, **_k: None)
    _patch_urlopen(monkeypatch, error=urllib.error.URLError("boom"))
    assert model_compat.hf_missing_tokenizer("org/model", ["hf_x"]) is None
