#!/usr/bin/env python3
"""Unit tests for the per-architecture diffusion FLOPs / ceiling estimator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import diffusion_flops as df


def _write_denoiser(tmp: Path, cfg: dict, sub: str = "transformer") -> Path:
    (tmp / sub).mkdir(parents=True, exist_ok=True)
    (tmp / sub / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp


# ---- peak table ----------------------------------------------------------
def test_peak_tflops_mi355_precisions():
    assert df.peak_tflops("mi355x", "bf16") == 2516.6
    assert df.peak_tflops("mi355x", "fp8") == 5033.2
    # fp8 is exactly 2x bf16 on the matrix core.
    assert df.peak_tflops("mi355x", "fp8") == pytest.approx(2 * df.peak_tflops("mi355x", "bf16"))
    assert df.peak_tflops("mi300x", "bf16") == 1307.4
    assert df.peak_tflops("mi355x", "nonsense") == 0.0
    assert df.peak_tflops("unknown", "bf16") == 0.0


# ---- family routing ------------------------------------------------------
def test_family_routing_for_each_class(tmp_path):
    cases = {
        "SD3Transformer2DModel": "mmdit",
        "FluxTransformer2DModel": "flux",
        "QwenImageTransformer2DModel": "mmdit",
        "AuraFlowTransformer2DModel": "flux",
        "HunyuanImageTransformer2DModel": "flux",
        "HiDreamImageTransformer2DModel": "moe_flux",
        "NucleusMoEImageTransformer2DModel": "moe_single",
        "Ideogram4Transformer2DModel": "single",
        "ErnieImageTransformer2DModel": "single",
        "ZImageTransformer2DModel": "single",
        "SanaTransformer2DModel": "sana",
        "UNet2DConditionModel": "unet",
    }
    for cls, fam in cases.items():
        d = _write_denoiser(
            tmp_path / cls,
            {
                "_class_name": cls,
                "num_layers": 8,
                "num_attention_heads": 16,
                "attention_head_dim": 64,
                "hidden_size": 1024,
                "dim": 1024,
                "block_out_channels": [320, 640, 1280],
                "transformer_layers_per_block": [1, 2, 10],
            },
            sub="unet" if fam == "unet" else "transformer",
        )
        g = df.resolve_geometry(d)
        assert g is not None, cls
        assert g.family == fam, (cls, g.family)


def test_unknown_class_returns_none(tmp_path):
    d = _write_denoiser(tmp_path / "x", {"_class_name": "TotallyUnknownModel"})
    assert df.resolve_geometry(d) is None


# ---- FLOPs math ----------------------------------------------------------
def test_mmdit_forward_flops_matches_formula(tmp_path):
    # Minimal SD3-like MMDiT: 2 layers, h=8 heads x 4 = 32, non-gated FFN 4x.
    cfg = {
        "_class_name": "SD3Transformer2DModel",
        "num_layers": 2,
        "num_attention_heads": 8,
        "attention_head_dim": 4,
        "patch_size": 2,
    }
    d = _write_denoiser(tmp_path / "m", cfg)
    g = df.resolve_geometry(d)
    assert g.hidden == 32 and g.intermediate == 128  # 4x default
    fwd = df.forward_flops(g, 1024, 1024)
    ti = fwd["image_tokens"]
    tt = fwd["text_tokens"]
    h, inter, L = 32, 128, 2
    qkvo = 4 * 2 * h * h
    ffn = 2 * 2 * h * inter  # non-gated
    per_layer = (ti + tt) * (qkvo + ffn) + 4.0 * (ti + tt) ** 2 * h
    assert fwd["forward_flops"] == pytest.approx(per_layer * L)


def test_ceiling_scales_with_steps_and_precision(tmp_path):
    cfg = {"_class_name": "SD3Transformer2DModel", "num_layers": 4,
           "num_attention_heads": 8, "attention_head_dim": 8, "patch_size": 2}
    d = _write_denoiser(tmp_path / "m", cfg)
    e10 = df.analytic_ceiling(d, gpu_type="mi355x", precision="bf16", num_steps=10, cfg_batch=1)
    e20 = df.analytic_ceiling(d, gpu_type="mi355x", precision="bf16", num_steps=20, cfg_batch=1)
    # doubling steps doubles total flops + ideal_ms
    assert e20["total_flops"] == pytest.approx(2 * e10["total_flops"])
    assert e20["ideal_ms"] == pytest.approx(2 * e10["ideal_ms"])
    # fp8 peak is 2x -> ideal_ms halves
    e_fp8 = df.analytic_ceiling(d, gpu_type="mi355x", precision="fp8", num_steps=10, cfg_batch=1)
    assert e_fp8["ideal_ms"] == pytest.approx(e10["ideal_ms"] / 2)
    # cfg batch 2 doubles vs 1
    e_cfg2 = df.analytic_ceiling(d, gpu_type="mi355x", precision="bf16", num_steps=10, cfg_batch=2)
    assert e_cfg2["total_flops"] == pytest.approx(2 * e10["total_flops"])


def test_moe_uses_active_experts(tmp_path):
    # MoE single with 64 experts, top-2 active -> FFN counts only 2 experts.
    cfg = {"_class_name": "NucleusMoEImageTransformer2DModel", "num_layers": 4,
           "num_attention_heads": 8, "attention_head_dim": 16, "num_experts": 64,
           "num_experts_per_tok": 2, "moe_intermediate_dim": 512, "patch_size": 2}
    d = _write_denoiser(tmp_path / "m", cfg)
    g = df.resolve_geometry(d)
    assert g.num_experts == 64 and g.active_experts == 2
    # active-expert FFN per token = 2 experts x gated(3 matmul) x 2*h*moe_inter
    assert df._moe_ffn_per_token(g) == pytest.approx(2 * 3 * 2 * g.hidden * 512)


def test_moe_defaults_active_when_topk_absent(tmp_path):
    cfg = {"_class_name": "NucleusMoEImageTransformer2DModel", "num_layers": 2,
           "num_attention_heads": 8, "attention_head_dim": 16, "num_experts": 64,
           "moe_intermediate_dim": 512}
    g = df.resolve_geometry(_write_denoiser(tmp_path / "m", cfg))
    assert g.active_experts == 2  # documented fallback


def test_unet_flops_positive_and_resolution_scales(tmp_path):
    cfg = {"_class_name": "UNet2DConditionModel", "block_out_channels": [320, 640, 1280],
           "layers_per_block": 2, "transformer_layers_per_block": [1, 2, 10],
           "cross_attention_dim": 2048, "in_channels": 4}
    d = _write_denoiser(tmp_path / "u", cfg, sub="unet")
    g = df.resolve_geometry(d)
    assert g.family == "unet"
    f512 = df.forward_flops(g, 512, 512)["forward_flops"]
    f1024 = df.forward_flops(g, 1024, 1024)["forward_flops"]
    assert f1024 > f512 > 0


def test_sana_linear_attention_cheaper_than_full(tmp_path):
    # Sana's linear attention must be far cheaper than full softmax attention
    # at the same token count.
    seq, h, hd = 4096, 2240, 32
    assert df._linear_attention_flops(seq, h, hd) < df._full_attention_flops(seq, h) / 10


# ---- real model configs (skipped when /primus/models is not mounted) -----
_MODELS_ROOT = Path("/primus/models")
_REAL = [
    "stabilityai-stable-diffusion-xl-base-1.0", "black-forest-labs-FLUX.1-dev",
    "black-forest-labs-FLUX.1-schnell", "Tongyi-MAI-Z-Image", "Tongyi-MAI-Z-Image-Turbo",
    "stabilityai-stable-diffusion-3.5-medium", "stabilityai-stable-diffusion-3.5-large",
    "stabilityai-stable-diffusion-3.5-large-turbo", "stabilityai-stable-diffusion-3-medium-diffusers",
    "Qwen-Qwen-Image", "Qwen-Qwen-Image-2512", "baidu-ERNIE-Image", "baidu-ERNIE-Image-Turbo",
    "HiDream-ai-HiDream-I1-Fast", "ideogram-ai-ideogram-4-fp8", "fal-AuraFlow",
    "NucleusAI-Nucleus-Image", "hunyuanvideo-community-HunyuanImage-2.1-Diffusers",
    "Warlord-K-Sana-1024", "tencent-SRPO", "stabilityai-stable-diffusion-3-medium",
]


@pytest.mark.parametrize("slug", _REAL)
def test_real_models_resolve_or_reference(slug):
    d = _MODELS_ROOT / slug
    if not d.is_dir():
        pytest.skip(f"{d} not mounted")
    g = df.resolve_geometry(d)
    if g is None:
        # SRPO / raw sd3-medium ship only single-file weights (no diffusers
        # config); they map to FLUX / SD3 reference archs at dispatch time.
        assert slug in ("tencent-SRPO", "stabilityai-stable-diffusion-3-medium")
        return
    est = df.analytic_ceiling(d, gpu_type="mi355x", precision="bf16")
    assert est["total_flops"] > 0
    assert est["ideal_ms"] > 0
