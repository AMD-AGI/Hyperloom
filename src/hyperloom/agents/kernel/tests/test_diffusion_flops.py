#!/usr/bin/env python3
"""Unit tests for the per-architecture diffusion FLOPs / ceiling estimator."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import diffusion_flops as df  # noqa: E402


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
        "Flux2Transformer2DModel": "flux",
        "WanTransformer3DModel": "wan",
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
    # Minimal SD3-like MMDiT: 2 layers, h=32, non-gated FFN 4x.
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
    cfg = {
        "_class_name": "SD3Transformer2DModel",
        "num_layers": 4,
        "num_attention_heads": 8,
        "attention_head_dim": 8,
        "patch_size": 2,
    }
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
    # 64 experts, top-2 active -> FFN counts only 2 experts.
    cfg = {
        "_class_name": "NucleusMoEImageTransformer2DModel",
        "num_layers": 4,
        "num_attention_heads": 8,
        "attention_head_dim": 16,
        "num_experts": 64,
        "num_experts_per_tok": 2,
        "moe_intermediate_dim": 512,
        "patch_size": 2,
    }
    d = _write_denoiser(tmp_path / "m", cfg)
    g = df.resolve_geometry(d)
    assert g.num_experts == 64 and g.active_experts == 2
    # active-expert FFN per token = 2 experts x gated(3 matmul) x 2*h*moe_inter
    assert df._moe_ffn_per_token(g) == pytest.approx(2 * 3 * 2 * g.hidden * 512)


def test_moe_defaults_active_when_topk_absent(tmp_path):
    cfg = {
        "_class_name": "NucleusMoEImageTransformer2DModel",
        "num_layers": 2,
        "num_attention_heads": 8,
        "attention_head_dim": 16,
        "num_experts": 64,
        "moe_intermediate_dim": 512,
    }
    g = df.resolve_geometry(_write_denoiser(tmp_path / "m", cfg))
    assert g.active_experts == 2  # documented fallback


def test_unet_flops_positive_and_resolution_scales(tmp_path):
    cfg = {
        "_class_name": "UNet2DConditionModel",
        "block_out_channels": [320, 640, 1280],
        "layers_per_block": 2,
        "transformer_layers_per_block": [1, 2, 10],
        "cross_attention_dim": 2048,
        "in_channels": 4,
    }
    d = _write_denoiser(tmp_path / "u", cfg, sub="unet")
    g = df.resolve_geometry(d)
    assert g.family == "unet"
    f512 = df.forward_flops(g, 512, 512)["forward_flops"]
    f1024 = df.forward_flops(g, 1024, 1024)["forward_flops"]
    assert f1024 > f512 > 0


def test_sana_linear_attention_cheaper_than_full(tmp_path):
    # Linear attention must be far cheaper than full softmax at the same token count.
    seq, h, hd = 4096, 2240, 32
    assert df._linear_attention_flops(seq, h, hd) < df._full_attention_flops(seq, h) / 10


# ---- real model configs (skipped when /primus/models is not mounted) -----
_MODELS_ROOT = Path("/primus/models")
_REAL = [
    "stabilityai-stable-diffusion-xl-base-1.0",
    "black-forest-labs-FLUX.1-dev",
    "black-forest-labs-FLUX.1-schnell",
    "Tongyi-MAI-Z-Image",
    "Tongyi-MAI-Z-Image-Turbo",
    "stabilityai-stable-diffusion-3.5-medium",
    "stabilityai-stable-diffusion-3.5-large",
    "stabilityai-stable-diffusion-3.5-large-turbo",
    "stabilityai-stable-diffusion-3-medium-diffusers",
    "Qwen-Qwen-Image",
    "Qwen-Qwen-Image-2512",
    "baidu-ERNIE-Image",
    "baidu-ERNIE-Image-Turbo",
    "HiDream-ai-HiDream-I1-Fast",
    "ideogram-ai-ideogram-4-fp8",
    "fal-AuraFlow",
    "NucleusAI-Nucleus-Image",
    "hunyuanvideo-community-HunyuanImage-2.1-Diffusers",
    "Warlord-K-Sana-1024",
    "tencent-SRPO",
    "stabilityai-stable-diffusion-3-medium",
]


@pytest.mark.parametrize("slug", _REAL)
def test_real_models_resolve_or_reference(slug):
    d = _MODELS_ROOT / slug
    if not d.is_dir():
        pytest.skip(f"{d} not mounted")
    g = df.resolve_geometry(d)
    if g is None:
        # SRPO / raw sd3-medium ship only single-file weights (no diffusers config).
        assert slug in ("tencent-SRPO", "stabilityai-stable-diffusion-3-medium")
        return
    est = df.analytic_ceiling(d, gpu_type="mi355x", precision="bf16")
    assert est["total_flops"] > 0
    assert est["ideal_ms"] > 0


# ---- safetensors-header inference fallback -------------------------------
def _write_safetensors(model_dir: Path, header: dict, name: str = "diffusion_pytorch_model.safetensors") -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    jb = json.dumps(header).encode("utf-8")
    (model_dir / name).write_bytes(struct.pack("<Q", len(jb)) + jb)
    return model_dir


def test_safetensors_header_bad_file_returns_none(tmp_path):
    p = tmp_path / "bad.safetensors"
    # Valid length prefix but non-JSON body -> None.
    p.write_bytes(struct.pack("<Q", 12) + b"not-json----")
    assert df._safetensors_header(p) is None


def test_infer_geometry_diffusers_naming_flux(tmp_path):
    header = {
        "__metadata__": {"format": "pt"},
        "transformer_blocks.0.attn.to_q.weight": {"shape": [3072, 3072]},
        "transformer_blocks.1.attn.to_q.weight": {"shape": [3072, 3072]},
        "single_transformer_blocks.0.attn.to_q.weight": {"shape": [3072, 3072]},
    }
    md = _write_safetensors(tmp_path / "flux.1-schnell", header)
    g = df._infer_geometry_from_safetensors(md)
    assert g is not None
    assert g.model_class == "FluxTransformer2DModel"
    assert g.hidden == 3072
    assert g.num_double_layers == 2 and g.num_single_layers == 1


def test_infer_geometry_hunyuan_naming_img_attn_qkv(tmp_path):
    header = {
        "double_blocks.0.img_attn_qkv.weight": {"shape": [9216, 3072]},
        "single_blocks.0.linear1.weight": {"shape": [12288, 3072]},
    }
    md = _write_safetensors(tmp_path / "hunyuan-dit", header)
    g = df._infer_geometry_from_safetensors(md)
    assert g is not None
    assert g.model_class == "HunyuanImageTransformer2DModel"
    assert g.hidden == 3072


def test_infer_geometry_no_blocks_returns_none(tmp_path):
    md = _write_safetensors(tmp_path / "x", {"some.random.weight": {"shape": [10, 10]}})
    assert df._infer_geometry_from_safetensors(md) is None


def test_infer_geometry_blocks_without_hidden_returns_none(tmp_path):
    md = _write_safetensors(tmp_path / "y", {"double_blocks.0.mlp.weight": {"shape": [10, 10]}})
    assert df._infer_geometry_from_safetensors(md) is None


def test_infer_geometry_unreadable_header_returns_none(tmp_path):
    md = tmp_path / "z"
    md.mkdir()
    # Header length points far past EOF -> header None.
    (md / "diffusion_pytorch_model.safetensors").write_bytes(struct.pack("<Q", 10**9))
    assert df._infer_geometry_from_safetensors(md) is None


def test_find_denoiser_safetensors_missing_returns_none(tmp_path):
    assert df._find_denoiser_safetensors(tmp_path) is None


# ---- unresolved geometry -------------------------------------------------
def test_estimate_and_ceiling_none_for_unknown_class(tmp_path):
    d = _write_denoiser(tmp_path / "u", {"_class_name": "TotallyUnknownModel"})
    assert df.estimate_image_flops(d) is None
    assert df.analytic_ceiling(d, gpu_type="mi355x", precision="bf16") is None


def test_unet_int_transformer_layers_per_block():
    """Exercise the ``isinstance(tlpb, int)`` broadcast branch in ``_unet_forward_flops``."""
    g = df.DenoiserGeometry(
        model_class="UNet2DConditionModel",
        family="unet",
        hidden=0,
        num_double_layers=0,
        num_single_layers=0,
        head_dim=0,
        intermediate=0,
        gated_ffn=False,
        unet={
            "block_out_channels": [320, 640],
            "layers_per_block": 2,
            "transformer_layers_per_block": 2,  # int -> broadcast branch
            "cross_attention_dim": 2048,
            "in_channels": 4,
        },
    )
    assert df.forward_flops(g, 512, 512)["forward_flops"] > 0


# ---- _fmt + CLI ----------------------------------------------------------
def test_fmt_with_and_without_ideal_ms(tmp_path):
    cfg = {
        "_class_name": "SD3Transformer2DModel",
        "num_layers": 4,
        "num_attention_heads": 8,
        "attention_head_dim": 8,
        "patch_size": 2,
    }
    d = _write_denoiser(tmp_path / "m", cfg)
    est = df.analytic_ceiling(d, gpu_type="mi355x", precision="bf16")
    line = df._fmt(est)
    assert "TFLOP/img" in line and "ideal=" in line
    # unknown gpu -> peak 0 -> no "ideal=" suffix.
    est_no_peak = df.analytic_ceiling(d, gpu_type="unknown", precision="bf16")
    assert "ideal=" not in df._fmt(est_no_peak)


def test_main_text_and_json_and_failure(tmp_path, monkeypatch, capsys):
    cfg = {
        "_class_name": "SD3Transformer2DModel",
        "num_layers": 4,
        "num_attention_heads": 8,
        "attention_head_dim": 8,
        "patch_size": 2,
    }
    d = _write_denoiser(tmp_path / "m", cfg)

    monkeypatch.setattr(sys, "argv", ["diffusion_flops", "--model-dir", str(d)])
    assert df.main() == 0
    assert "TFLOP/img" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["diffusion_flops", "--model-dir", str(d), "--json"])
    assert df.main() == 0
    assert '"total_flops"' in capsys.readouterr().out

    bad = tmp_path / "empty"
    bad.mkdir()
    monkeypatch.setattr(sys, "argv", ["diffusion_flops", "--model-dir", str(bad)])
    assert df.main() == 1
    assert "could not resolve" in capsys.readouterr().out


# ---- FLUX.2 --------------------------------------------------------------
_FLUX2_CFG = {
    "_class_name": "Flux2Transformer2DModel",
    "num_layers": 8,
    "num_single_layers": 48,
    "num_attention_heads": 48,
    "attention_head_dim": 128,
    "in_channels": 128,
    "patch_size": 1,
    "joint_attention_dim": 15360,
    "axes_dims_rope": [32, 32, 32, 32],
}


def test_flux2_geometry_and_defaults(tmp_path):
    """Real geometry from the shipped black-forest-labs/FLUX.2-dev config."""
    g = df.resolve_geometry(_write_denoiser(tmp_path / "flux2", _FLUX2_CFG))
    assert g is not None
    assert (g.family, g.model_class) == ("flux", "Flux2Transformer2DModel")
    assert g.hidden == 6144  # 48 heads x 128
    assert (g.num_double_layers, g.num_single_layers) == (8, 48)
    # patch_size=1 in the config, but FLUX packs 2x2 for the token count.
    assert g.patch == 2
    # FLUX.2-dev is guidance-distilled: one forward per step.
    assert (g.default_cfg_batch, g.default_steps) == (1, 50)

    est = df.estimate_image_flops(_write_denoiser(tmp_path / "flux2b", _FLUX2_CFG))
    assert est["image_tokens"] == 4096  # (1024/8/2)^2
    assert est["layers"] == 56
    assert est["total_flops"] > 0


def test_flux1_is_unaffected_by_the_flux2_entry(tmp_path):
    cfg = {
        "_class_name": "FluxTransformer2DModel",
        "num_layers": 19,
        "num_single_layers": 38,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
    }
    g = df.resolve_geometry(_write_denoiser(tmp_path / "flux1", cfg))
    assert g.hidden == 3072
    assert (g.num_double_layers, g.num_single_layers) == (19, 38)
    assert (g.default_cfg_batch, g.default_steps) == (1, 28)


# ---- cross-attention completeness ----------------------------------------
def _cross_geom(**kw):
    base = dict(
        model_class="x",
        family="sana",
        hidden=5120,
        num_double_layers=1,
        num_single_layers=0,
        head_dim=128,
        intermediate=13824,
        gated_ffn=False,
    )
    base.update(kw)
    return df.DenoiserGeometry(**base)


def test_cross_attention_counts_q_and_o_projections():
    """All four projections, not just K and V.

    The query stream needs its own Q and O matmuls, charged to the query token
    count. They were previously omitted, understating a cross-attending block.
    """
    g = _cross_geom()
    q_tokens, tt = 75_600, 512
    total = df._cross_attention_block_flops(g, q_tokens, tt)

    attn = df._cross_attention_flops(q_tokens, tt, g.hidden)
    kv = tt * df._linear_flops(1.0, g.hidden, g.hidden) * 2
    qo = q_tokens * df._linear_flops(1.0, g.hidden, g.hidden) * 2
    assert total == pytest.approx(attn + kv + qo)
    # Q/O scale with the query tokens, so they dominate the K/V term here.
    assert qo > kv * 100


def test_cross_attention_qo_scales_with_query_tokens():
    """Q/O scale with the queries; K/V does not, so growth is sub-linear."""
    g, tt = _cross_geom(), 512
    small = df._cross_attention_block_flops(g, 1_000, tt)
    large = df._cross_attention_block_flops(g, 2_000, tt)

    # The K/V projection is charged to the text length and is invariant in q,
    # so the delta is exactly the q-proportional attention + Q/O terms.
    attn_delta = df._cross_attention_flops(2_000, tt, g.hidden) - df._cross_attention_flops(1_000, tt, g.hidden)
    qo_delta = 1_000 * df._linear_flops(1.0, g.hidden, g.hidden) * 2
    assert large - small == pytest.approx(attn_delta + qo_delta)
    # Sub-linear overall precisely because K/V stays put.
    assert 1.0 < large / small < 2.0


def test_sana_layer_includes_cross_attention_projections(tmp_path):
    cfg = {
        "_class_name": "SanaTransformer2DModel",
        "num_layers": 2,
        "num_attention_heads": 8,
        "attention_head_dim": 32,
    }
    g = df.resolve_geometry(_write_denoiser(tmp_path / "sana", cfg))
    ffn_pt = df._ffn_per_token(g)
    ti, tt = 1024, g.text_tokens
    layer = df._sana_layer(g, ti, tt, ffn_pt)
    # The layer must contain the full cross-attention block, Q/O included.
    assert layer == pytest.approx(
        ti * df._qkvo_flops(1.0, g.hidden)
        + ti * ffn_pt
        + df._linear_attention_flops(ti, g.hidden, g.head_dim or 32)
        + df._cross_attention_block_flops(g, ti, tt)
    )


# ---- Wan (3D video) ------------------------------------------------------
_WAN_CFG = {
    "_class_name": "WanTransformer3DModel",
    "num_layers": 40,
    "num_attention_heads": 40,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "text_dim": 4096,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
}


def test_wan_geometry_from_real_config(tmp_path):
    g = df.resolve_geometry(_write_denoiser(tmp_path / "wan", _WAN_CFG))
    assert (g.family, g.model_class) == ("wan", "WanTransformer3DModel")
    assert g.hidden == 5120 and g.num_double_layers == 40
    # patch_size is (t, h, w); the spatial patch is 2, NOT the leading 1.
    assert (g.patch, g.patch_t) == (2, 1)
    # ffn_dim must win over the 4*hidden fallback (which would be 20480).
    assert g.intermediate == 13824
    assert (g.vae_spatial, g.vae_temporal, g.default_frames) == (8, 4, 81)


def test_wan_latent_frames_causal_vae(tmp_path):
    g = df.resolve_geometry(_write_denoiser(tmp_path / "wan", _WAN_CFG))
    assert df._latent_frames(g, 81) == 21  # causal: (81-1)//4 + 1
    assert df._latent_frames(g, 1) == 1
    assert df._latent_frames(g, 4) == 1
    assert df._latent_frames(g, 5) == 2
    assert df._latent_frames(g, 0) == 1


def test_wan_video_tokens_match_observed_trace_shape(tmp_path):
    """720p x 81 frames must give 75600 tokens.

    That is the operand shape a real Wan2.2 trace carries under Ulysses-8
    ((1, 75600, 5, 128)), so it pins the whole token model at once.
    """
    g = df.resolve_geometry(_write_denoiser(tmp_path / "wan", _WAN_CFG))
    assert df._video_tokens(g, 720, 1280, 81) == 75600  # 21 * 45 * 80
    fwd = df.forward_flops(g, 720, 1280, 81)
    assert fwd["image_tokens"] == 75600 and fwd["latent_frames"] == 21


def test_forward_flops_frames_defaults_to_the_family_not_one(tmp_path):
    """A bare ``frames=1`` default silently undercut Wan by the frame count.

    ``forward_flops`` is public, so omitting ``frames`` must mean "this model's
    own default", exactly as ``estimate_image_flops`` already behaved.
    """
    g = df.resolve_geometry(_write_denoiser(tmp_path / "wan", _WAN_CFG))
    implicit = df.forward_flops(g, 720, 1280)["forward_flops"]
    explicit = df.forward_flops(g, 720, 1280, 81)["forward_flops"]
    single = df.forward_flops(g, 720, 1280, 1)["forward_flops"]
    assert implicit == explicit
    assert explicit / single > 50  # the magnitude of the old silent undercount


def test_image_families_ignore_frames(tmp_path):
    cfg = {"_class_name": "FluxTransformer2DModel", "num_layers": 2, "num_attention_heads": 8, "attention_head_dim": 64}
    d = _write_denoiser(tmp_path / "flux", cfg)
    assert df.estimate_image_flops(d, num_frames=1)["total_flops"] == (
        df.estimate_image_flops(d, num_frames=97)["total_flops"]
    )
    assert "num_frames" not in df.estimate_image_flops(d)
    g = df.resolve_geometry(d)
    assert df._latent_frames(g, 81) == 1


def test_wan_estimate_and_ceiling_thread_num_frames(tmp_path):
    d = _write_denoiser(tmp_path / "wan", _WAN_CFG)
    est = df.estimate_image_flops(d, height=720, width=1280, num_frames=81, num_steps=8)
    assert est["image_tokens"] == 75600
    assert (est["num_frames"], est["latent_frames"]) == (81, 21)
    assert est["cfg_batch"] == 2  # WanPipeline uses CFG
    assert df.estimate_image_flops(d, height=720, width=1280)["num_frames"] == 81
    fewer = df.estimate_image_flops(d, height=720, width=1280, num_frames=41, num_steps=8)
    assert fewer["total_flops"] < est["total_flops"]

    ceil = df.analytic_ceiling(d, height=720, width=1280, num_frames=81, num_steps=8)
    assert ceil["ideal_ms"] > 0
    line = df._fmt(ceil)
    assert "TFLOP/clip" in line and "f=81(21lat)" in line


def test_wan_keeps_text_out_of_self_attention(tmp_path):
    """Wan cross-attends to text rather than concatenating it."""
    g = df.resolve_geometry(_write_denoiser(tmp_path / "wan", _WAN_CFG))
    ffn_pt = df._ffn_per_token(g)
    base = df._wan_layer(g, 10_000, 512, ffn_pt)
    more_text = df._wan_layer(g, 10_000, 1024, ffn_pt)
    assert base < more_text < 1.10 * base


def test_main_frames_flag(tmp_path, monkeypatch, capsys):
    d = _write_denoiser(tmp_path / "wan", _WAN_CFG)
    monkeypatch.setattr(
        sys,
        "argv",
        ["diffusion_flops", "--model-dir", str(d), "--height", "720", "--width", "1280", "--frames", "81", "--json"],
    )
    assert df.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["image_tokens"] == 75600 and out["latent_frames"] == 21


def test_cross_attention_kv_reads_the_context_width(tmp_path):
    """K/V project from the text-encoder width, not from ``hidden``."""
    g = df.resolve_geometry(_write_denoiser(tmp_path / "wan", _WAN_CFG))
    assert g.context_dim == 4096  # Wan text_dim, vs hidden 5120
    tt = 512
    kv = tt * df._linear_flops(1.0, g.context_dim, g.hidden) * 2
    qo = 1_000 * df._linear_flops(1.0, g.hidden, g.hidden) * 2
    attn = df._cross_attention_flops(1_000, tt, g.hidden)
    assert df._cross_attention_block_flops(g, 1_000, tt) == pytest.approx(attn + qo + kv)
