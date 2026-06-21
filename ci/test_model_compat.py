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


def test_normal_model_is_kept():
    assert _reason({"architectures": ["MistralForCausalLM"], "model_type": "mistral",
                    "max_position_embeddings": 32768}) is None


def test_non_dict_config_is_kept():
    assert model_compat.unrunnable_reason(None) is None
    assert model_compat.unrunnable_reason("nope") is None


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
