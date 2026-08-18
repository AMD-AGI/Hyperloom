# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The diffusion (xDiT) arm of the roofline ceiling.

Diffusion is measured in images/sec, and its sequence length comes from the
latent grid rather than a token count, so the DiT geometry readers are what
decide whether the ceiling is meaningful at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import roofline_ceiling as rc


def _model(tmp_path: Path, *, transformer: dict | None = None, vae: dict | None = None) -> str:
    """Write a diffusers-layout model dir with the given sub-configs."""
    root = tmp_path / "diffusion-model"
    for name, cfg in (("transformer", transformer), ("vae", vae)):
        if cfg is None:
            continue
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


# ---- VAE geometry ----


def test_vae_scale_is_one_stride_two_stage_per_extra_block(tmp_path):
    path = _model(tmp_path, vae={"block_out_channels": [128, 256, 512, 512], "latent_channels": 16})
    assert rc._read_vae_geometry(path) == (2 ** 3, 16)


@pytest.mark.parametrize(
    "vae",
    [None, {"block_out_channels": [], "latent_channels": 4}, {"latent_channels": 4}],
)
def test_vae_geometry_falls_back_to_the_standard_downscale(tmp_path, vae):
    """8x is the SD/FLUX VAE; an unreadable config must not change the scale."""
    path = _model(tmp_path, vae=vae)
    assert rc._read_vae_geometry(path)[0] == 8


def test_vae_geometry_survives_a_malformed_config(tmp_path):
    root = tmp_path / "m"
    (root / "vae").mkdir(parents=True)
    (root / "vae" / "config.json").write_text("[]", encoding="utf-8")
    assert rc._read_vae_geometry(str(root)) == (8, 0)


# ---- latent token grid ----


def test_latent_tokens_divide_the_resolution_by_vae_and_patch_pack(tmp_path):
    """FLUX packs a 2x2 latent patch into channels, so in_channels / latent = 4."""
    path = _model(tmp_path, vae={"block_out_channels": [1, 2, 3, 4], "latent_channels": 16})
    tokens = rc._diffusion_latent_tokens_from_resolution(path, {"in_channels": 64}, 1024, 1024)
    # downscale = vae 8 * pack 2
    assert tokens == (1024 // 16) * (1024 // 16)


def test_latent_tokens_default_the_pack_when_it_is_not_a_square_ratio(tmp_path):
    path = _model(tmp_path, vae={"block_out_channels": [1, 2, 3, 4], "latent_channels": 16})
    # 48/16 = 3, not a perfect square -> keep the FLUX/SD3 default pack of 2.
    tokens = rc._diffusion_latent_tokens_from_resolution(path, {"in_channels": 48}, 1024, 1024)
    assert tokens == (1024 // 16) * (1024 // 16)


@pytest.mark.parametrize("height, width", [(0, 1024), (1024, 0), (0, 0)])
def test_latent_tokens_need_a_resolution(tmp_path, height, width):
    path = _model(tmp_path, vae={"block_out_channels": [1, 2], "latent_channels": 4})
    assert rc._diffusion_latent_tokens_from_resolution(path, {"in_channels": 16}, height, width) == 0


# ---- DiT metadata ----


def test_dit_meta_counts_dual_stream_blocks_twice(tmp_path):
    """A FLUX dual-stream block runs separate image and text projections."""
    hidden = 256
    path = _model(
        tmp_path,
        transformer={
            "num_layers": 2,
            "num_single_layers": 3,
            "num_attention_heads": 4,
            "attention_head_dim": 64,
            "sample_size": 64,
            "patch_size": 2,
            "in_channels": 16,
        },
    )
    meta = rc._read_diffusion_dit_meta(path)
    assert meta is not None
    params, tokens, layers, h = meta
    assert h == hidden
    assert layers == 2 + 3
    assert params == 12 * hidden**2 * (2 * 2 + 3)
    assert tokens == (64 // 2) * (64 // 2)


def test_dit_meta_reads_a_non_square_patch(tmp_path):
    path = _model(
        tmp_path,
        transformer={
            "num_layers": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "sample_size": 64,
            "patch_size": [2, 4],
        },
    )
    assert rc._read_diffusion_dit_meta(path)[1] == (64 // 2) * (64 // 4)


def test_dit_meta_falls_back_to_the_runtime_resolution(tmp_path):
    """FLUX/SD3 carry no sample_size; the runtime resolution sets the grid."""
    path = _model(
        tmp_path,
        transformer={"num_layers": 2, "num_attention_heads": 4, "attention_head_dim": 64, "in_channels": 64},
        vae={"block_out_channels": [1, 2, 3, 4], "latent_channels": 16},
    )
    params, tokens, _layers, _h = rc._read_diffusion_dit_meta(path, height=1024, width=512)
    assert tokens == (1024 // 16) * (512 // 16)
    assert params > 0


def test_dit_meta_declines_an_unusable_transformer(tmp_path):
    assert rc._read_diffusion_dit_meta(str(tmp_path / "absent")) is None
    # Present but missing the block count / model dim.
    assert rc._read_diffusion_dit_meta(_model(tmp_path, transformer={"num_layers": 0})) is None
    no_hidden = _model(tmp_path / "b", transformer={"num_layers": 2})
    assert rc._read_diffusion_dit_meta(no_hidden) is None


# ---- ceilings ----


def test_diffusion_memory_ceiling_reads_the_weights_once_per_step():
    gb = 1024**3
    img_s = rc.compute_diffusion_mem_img_per_sec(
        gpu_type="mi300x", num_gpus=1, weight_bytes=10 * gb, num_steps=25
    )
    bw = rc.HW_SPECS["mi300x"]["hbm_bw_gbps"] * 1e9
    assert img_s == pytest.approx(1.0 / (25 * (10 * gb / bw)))


def test_diffusion_memory_ceiling_scales_with_the_gpu_count():
    kw = dict(gpu_type="mi300x", weight_bytes=1024**3, num_steps=10)
    assert rc.compute_diffusion_mem_img_per_sec(num_gpus=4, **kw) == pytest.approx(
        4 * rc.compute_diffusion_mem_img_per_sec(num_gpus=1, **kw)
    )


@pytest.mark.parametrize(
    "over",
    [
        {"gpu_type": "not-a-gpu"},
        {"weight_bytes": 0},
        {"num_steps": 0},
    ],
)
def test_diffusion_memory_ceiling_declines_degenerate_input(over):
    kw = dict(gpu_type="mi300x", num_gpus=1, weight_bytes=1024**3, num_steps=10)
    kw.update(over)
    assert rc.compute_diffusion_mem_img_per_sec(**kw) == 0.0


def test_diffusion_compute_ceiling_sums_linear_and_attention_flops():
    kw = dict(
        gpu_type="mi300x",
        num_gpus=1,
        precision_tag="bf16",
        dit_params=1_000_000,
        latent_tokens=1024,
        num_layers=8,
        hidden_size=256,
        num_steps=20,
    )
    linear = 2.0 * kw["dit_params"] * kw["latent_tokens"]
    attn = 4.0 * kw["num_layers"] * kw["latent_tokens"] ** 2 * kw["hidden_size"]
    peak = rc._resolve_achievable_tflops("mi300x", "bf16") * 1e12
    assert rc.compute_diffusion_compute_img_per_sec(**kw) == pytest.approx(
        peak / (kw["num_steps"] * (linear + attn))
    )


def test_diffusion_compute_ceiling_scales_with_the_gpu_count():
    kw = dict(
        gpu_type="mi300x",
        precision_tag="bf16",
        dit_params=1_000_000,
        latent_tokens=512,
        num_layers=4,
        hidden_size=128,
        num_steps=10,
    )
    assert rc.compute_diffusion_compute_img_per_sec(num_gpus=8, **kw) == pytest.approx(
        8 * rc.compute_diffusion_compute_img_per_sec(num_gpus=1, **kw)
    )


@pytest.mark.parametrize("over", [{"dit_params": 0}, {"latent_tokens": 0}, {"num_steps": 0}, {"gpu_type": "xpu"}])
def test_diffusion_compute_ceiling_declines_degenerate_input(over):
    kw = dict(
        gpu_type="mi300x",
        num_gpus=1,
        precision_tag="bf16",
        dit_params=1_000,
        latent_tokens=64,
        num_layers=2,
        hidden_size=64,
        num_steps=4,
    )
    kw.update(over)
    assert rc.compute_diffusion_compute_img_per_sec(**kw) == 0.0
