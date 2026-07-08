#!/usr/bin/env python3
"""Per-architecture analytic FLOPs / compute-ceiling estimator for the xDiT
text-to-image models Hyperloom optimizes.

This is the "approach a" absolute compute ceiling for diffusion, the dual of
``roofline_ceiling.py`` (LLM decode memory roofline). Instead of deriving the
ideal time from a profiling trace, it computes the *architecture-analytic*
forward FLOPs of one denoise step from the model's own diffusers config, scales
by (denoise_steps x cfg_batch), and divides by the hardware matrix-core peak at
the runtime precision:

    ideal_ms = total_flops / (peak_tflops * 1e12) * 1e3

Every model is routed by its denoiser ``_class_name`` to the correct family
formula (standard/dual-stream MMDiT, FLUX double+single stream, MoE, UNet, or
Sana linear-attention), reading each config's real dimensions. Constants that
the config does not expose (text sequence length, VAE spatial compression,
FFN gating) use documented per-family defaults and are overridable, since the
compute ceiling is dominated by the linear-projection term (~L*h^2*tokens)
which is always read directly from the config.

FLOP convention: 2 FLOPs per multiply-accumulate. Attention softmax and
element-wise ops are omitted (dominated by the matmuls), so the number is a
tight lower bound on true compute -> an optimistic (never violated) ceiling.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hardware matrix-core peak TFLOPS (self-contained; mirrors
# roofline_ceiling.HW_SPECS so this tool runs standalone in kernel-agent).
# ---------------------------------------------------------------------------
_PEAK_TFLOPS: dict[str, dict[str, float]] = {
    "mi300x": {"bf16": 1307.4, "fp16": 1307.4, "fp8": 2614.9, "fp32": 163.4},
    "mi308x": {"bf16": 1307.4, "fp16": 1307.4, "fp8": 2614.9, "fp32": 163.4},
    "mi325x": {"bf16": 1307.4, "fp16": 1307.4, "fp8": 2614.9, "fp32": 163.4},
    "mi355x": {"bf16": 2516.6, "fp16": 2516.6, "fp8": 5033.2, "mxfp4": 10066.4, "fp32": 157.3},
}

#: Precision tag -> canonical peak-table key.
_PRECISION_ALIASES: dict[str, str] = {
    "bf16": "bf16", "bfloat16": "bf16", "fp16": "fp16", "float16": "fp16",
    "fp8": "fp8", "float8_e4m3fn": "fp8", "float8_e5m2": "fp8", "fp8_e4m3": "fp8",
    "mxfp4": "mxfp4", "fp4": "mxfp4", "float4": "mxfp4",
    "fp32": "fp32", "float32": "fp32",
}


def peak_tflops(gpu_type: str, precision: str) -> float:
    """Matrix-core peak TFLOPS for ``(gpu, precision)``; 0.0 on any miss."""
    table = _PEAK_TFLOPS.get((gpu_type or "").strip().lower())
    if not table:
        return 0.0
    key = _PRECISION_ALIASES.get((precision or "").strip().lower(), (precision or "").strip().lower())
    return float(table.get(key, 0.0))


# ---------------------------------------------------------------------------
# Low-level FLOPs primitives (2 FLOPs / MAC).
# ---------------------------------------------------------------------------
def _linear_flops(tokens: float, k_in: float, n_out: float) -> float:
    """Dense matmul ``[tokens, k_in] x [k_in, n_out]``."""
    return 2.0 * tokens * k_in * n_out


def _qkvo_flops(tokens: float, h: float) -> float:
    """Q, K, V, O projections (each h->h) over ``tokens``."""
    return 4.0 * _linear_flops(tokens, h, h)


def _ffn_flops(tokens: float, h: float, inter: float, gated: bool = False) -> float:
    """FFN matmuls: up + down (+ gate when gated). Ratio-agnostic (uses the
    real intermediate size)."""
    mats = 3 if gated else 2
    # up/gate: h->inter ; down: inter->h. Each is 2*tokens*h*inter.
    return mats * _linear_flops(tokens, h, inter)


def _full_attention_flops(seq: float, h: float) -> float:
    """Softmax attention over ``seq`` tokens with model hidden ``h`` (summed
    across heads): QK^T + softmax@V = 2 * (2*seq^2*h)."""
    return 4.0 * seq * seq * h


def _cross_attention_flops(q_tokens: float, kv_tokens: float, h: float) -> float:
    """Cross-attention: query x key/value. QK^T + attn@V = 2 * (2*q*kv*h)."""
    return 4.0 * q_tokens * kv_tokens * h


def _linear_attention_flops(seq: float, h: float, head_dim: float) -> float:
    """ReLU/linear attention (Sana): O(seq) instead of O(seq^2). K^T V builds a
    (head_dim x head_dim) state then Q applies it: ~ 2 * 2 * seq * h * head_dim."""
    return 4.0 * seq * h * head_dim


# ---------------------------------------------------------------------------
# Model family resolution.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DenoiserGeometry:
    """Resolved denoiser architecture for the analytic FLOPs estimator."""

    model_class: str
    family: str  # mmdit | flux | single | moe_single | unet | sana
    hidden: int
    num_double_layers: int
    num_single_layers: int
    head_dim: int
    intermediate: int
    gated_ffn: bool
    # MoE
    num_experts: int = 0
    active_experts: int = 0
    moe_intermediate: int = 0
    # token geometry
    vae_spatial: int = 8  # latent downsample from pixels
    patch: int = 2  # transformer patch size on the latent
    text_tokens: int = 256
    default_steps: int = 28
    default_cfg_batch: int = 2  # classifier-free guidance -> 2 forwards/step
    # UNet-only (SDXL)
    unet: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# _class_name -> family key.
_CLASS_FAMILY: dict[str, str] = {
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

# Per-class overrides for constants the config does not expose.
# steps/cfg are diffusers pipeline defaults; turbo/schnell are few-step & CFG-free.
_CLASS_DEFAULTS: dict[str, dict[str, Any]] = {
    "SD3Transformer2DModel": {"text_tokens": 333, "default_steps": 28, "default_cfg_batch": 2},
    "FluxTransformer2DModel": {"text_tokens": 512, "default_steps": 28, "default_cfg_batch": 1},
    "QwenImageTransformer2DModel": {"text_tokens": 256, "default_steps": 30, "default_cfg_batch": 2},
    "AuraFlowTransformer2DModel": {"text_tokens": 256, "default_steps": 50, "default_cfg_batch": 2},
    "HunyuanImageTransformer2DModel": {"text_tokens": 256, "default_steps": 50, "default_cfg_batch": 2},
    "HiDreamImageTransformer2DModel": {"text_tokens": 128, "default_steps": 16, "default_cfg_batch": 1},
    "NucleusMoEImageTransformer2DModel": {"text_tokens": 256, "default_steps": 28, "default_cfg_batch": 2},
    "Ideogram4Transformer2DModel": {"text_tokens": 256, "default_steps": 28, "default_cfg_batch": 2},
    "ErnieImageTransformer2DModel": {"text_tokens": 512, "default_steps": 28, "default_cfg_batch": 2},
    "ZImageTransformer2DModel": {"text_tokens": 512, "default_steps": 30, "default_cfg_batch": 1},
    "SanaTransformer2DModel": {"text_tokens": 300, "default_steps": 20, "default_cfg_batch": 2, "vae_spatial": 32},
    "UNet2DConditionModel": {"text_tokens": 77, "default_steps": 40, "default_cfg_batch": 2},
}

# Per-model-basename step/cfg refinements (turbo / schnell / fast distilled).
_BASENAME_HINTS: dict[str, dict[str, Any]] = {
    "flux.1-schnell": {"text_tokens": 256, "default_steps": 4, "default_cfg_batch": 1},
    "stable-diffusion-3.5-large-turbo": {"default_steps": 4, "default_cfg_batch": 1},
    "z-image-turbo": {"default_steps": 8, "default_cfg_batch": 1},
    "hidream-i1-fast": {"default_steps": 16, "default_cfg_batch": 1},
    "ernie-image-turbo": {"default_steps": 4, "default_cfg_batch": 1},
    "sana-1024": {"default_steps": 20, "default_cfg_batch": 2},
}


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _denoiser_config(model_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """Return ``(config, subdir)`` for the transformer or unet denoiser."""
    for sub in ("transformer", "unet"):
        c = _read_json(model_dir / sub / "config.json")
        if c:
            return c, sub
    # single-file / top-level config
    c = _read_json(model_dir / "config.json")
    return c, ""


def _resolve_hidden(cfg: dict[str, Any]) -> int:
    for k in ("hidden_size", "dim"):
        if cfg.get(k):
            return int(cfg[k])
    heads = cfg.get("num_attention_heads") or cfg.get("n_heads")
    hd = cfg.get("attention_head_dim")
    if heads and hd:
        return int(heads) * int(hd)
    return 0


def _resolve_head_dim(cfg: dict[str, Any], hidden: int) -> int:
    hd = cfg.get("attention_head_dim")
    if isinstance(hd, (int, float)) and hd:
        return int(hd)
    heads = cfg.get("num_attention_heads") or cfg.get("n_heads")
    if heads and hidden:
        return hidden // int(heads)
    return 0


def _resolve_intermediate(cfg: dict[str, Any], hidden: int) -> int:
    for k in ("intermediate_size", "ffn_hidden_size"):
        if cfg.get(k):
            return int(cfg[k])
    if cfg.get("mlp_ratio"):
        return int(float(cfg["mlp_ratio"]) * hidden)
    if cfg.get("expand_ratio"):
        return int(float(cfg["expand_ratio"]) * hidden)
    return int(4 * hidden)


def _safetensors_header(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _find_denoiser_safetensors(model_dir: Path) -> Path | None:
    """Locate the main denoiser weight file in a non-diffusers checkpoint.

    Handles bare single-file DiTs (e.g. ``diffusion_pytorch_model.safetensors``)
    and original ``dit/`` layouts, skipping fp8 / scale / refiner sidecars.
    """
    cands: list[Path] = list(model_dir.glob("diffusion_pytorch_model*.safetensors"))
    for sub in ("dit", "transformer"):
        cands += list((model_dir / sub).glob("*.safetensors"))

    def _is_main(p: Path) -> bool:
        n = p.name.lower()
        return not any(t in n for t in ("fp8", "scale", "refiner", "distilled", "index"))

    mains = [p for p in cands if _is_main(p)] or cands
    if not mains:
        return None
    return max(mains, key=lambda p: p.stat().st_size if p.exists() else 0)


def _infer_geometry_from_safetensors(model_dir: Path) -> DenoiserGeometry | None:
    """Fallback for checkpoints without a resolvable transformer config.

    Infers a FLUX-style dual + single-stream geometry from the safetensors
    tensor-name index (block counts + hidden size), then maps the naming
    convention onto a known ``_class_name`` so the per-class defaults apply.
    """
    st = _find_denoiser_safetensors(model_dir)
    if st is None:
        return None
    hdr = _safetensors_header(st)
    if not hdr:
        return None
    keys = [k for k in hdr if k != "__metadata__"]

    double: set[int] = set()
    single: set[int] = set()
    for k in keys:
        m = re.search(
            r"(single_transformer_blocks|transformer_blocks|single_blocks|double_blocks)\.(\d+)\.",
            k,
        )
        if not m:
            continue
        (single if m.group(1).startswith("single") else double).add(int(m.group(2)))
    if not double and not single:
        return None

    hidden = 0
    for k in keys:
        shp = hdr[k].get("shape") or []
        if k.endswith("attn.to_q.weight") and len(shp) == 2:
            hidden = int(shp[0])
            break
        if k.endswith("img_attn_qkv.weight") and len(shp) == 2:
            hidden = int(shp[1])
            break
    if not hidden:
        return None

    diffusers_naming = any(
        k.startswith(("transformer_blocks.", "single_transformer_blocks."))
        for k in keys
    )
    model_class = (
        "FluxTransformer2DModel" if diffusers_naming else "HunyuanImageTransformer2DModel"
    )
    defaults = dict(_CLASS_DEFAULTS.get(model_class, {}))
    basename = model_dir.name.lower()
    for hint_key, hint in _BASENAME_HINTS.items():
        if hint_key in basename:
            defaults.update(hint)

    return DenoiserGeometry(
        model_class=model_class,
        family="flux",
        hidden=hidden,
        num_double_layers=len(double),
        num_single_layers=len(single),
        head_dim=128,
        intermediate=4 * hidden,
        gated_ffn=False,
        vae_spatial=int(defaults.get("vae_spatial", 8)),
        patch=2,
        text_tokens=int(defaults.get("text_tokens", 256)),
        default_steps=int(defaults.get("default_steps", 28)),
        default_cfg_batch=int(defaults.get("default_cfg_batch", 2)),
        notes=f"{basename} (inferred from {st.name})",
    )


def resolve_geometry(model_dir: str | Path) -> DenoiserGeometry | None:
    """Read a local diffusers model dir and resolve its denoiser geometry.

    Args:
        model_dir: Local model directory (with ``transformer/`` or ``unet/``).

    Returns:
        The resolved :class:`DenoiserGeometry`, or ``None`` when no denoiser
        config is found / the family is unknown.
    """
    d = Path(model_dir).expanduser()
    cfg, _sub = _denoiser_config(d)
    if not cfg:
        return _infer_geometry_from_safetensors(d)
    model_class = str(cfg.get("_class_name") or "")
    family = _CLASS_FAMILY.get(model_class)
    if family is None:
        return _infer_geometry_from_safetensors(d)

    hidden = _resolve_hidden(cfg)
    head_dim = _resolve_head_dim(cfg, hidden)
    intermediate = _resolve_intermediate(cfg, hidden)

    # layer split
    n_double = int(
        cfg.get("num_layers")
        or cfg.get("num_mmdit_layers")
        or cfg.get("n_layers")
        or 0
    )
    n_single = int(cfg.get("num_single_layers") or cfg.get("num_single_dit_layers") or 0)
    # Z-Image adds refiner layers on top of n_layers.
    n_double += int(cfg.get("n_refiner_layers") or 0)

    # MoE
    num_experts = int(cfg.get("num_experts") or cfg.get("num_routed_experts") or 0)
    active = int(
        cfg.get("num_activated_experts")
        or cfg.get("num_experts_per_tok")
        or cfg.get("top_k")
        or 0
    )
    moe_inter = int(cfg.get("moe_intermediate_dim") or cfg.get("moe_intermediate_size") or 0)
    if num_experts and not active:
        active = 2  # documented fallback when the config omits top-k
    if num_experts and not moe_inter:
        moe_inter = intermediate

    # gated FFN heuristic: Sana / Z-Image style use gated MLP; MMDiT/Flux GELU are not.
    gated = family in ("sana",) or bool(cfg.get("use_gated_mlp"))

    defaults = dict(_CLASS_DEFAULTS.get(model_class, {}))
    basename = d.name.lower()
    for hint_key, hint in _BASENAME_HINTS.items():
        if hint_key in basename:
            defaults.update(hint)

    patch = 2
    ps = cfg.get("patch_size") or cfg.get("all_patch_size")
    if isinstance(ps, list) and ps:
        patch = int(ps[0])
    elif isinstance(ps, int):
        patch = int(ps)
    # FLUX / ERNIE / Sana declare patch_size=1 but pack 2x2 (or use a high-
    # compression VAE); their effective latent token stride is captured by the
    # per-family vae_spatial + a forced patch=2 for the pixel->token count.
    if family in ("flux", "moe_flux") or model_class in ("ErnieImageTransformer2DModel",):
        patch = 2

    unet_geom: dict[str, Any] = {}
    if family == "unet":
        unet_geom = {
            "block_out_channels": list(cfg.get("block_out_channels") or []),
            "layers_per_block": int(cfg.get("layers_per_block") or 2),
            "transformer_layers_per_block": list(cfg.get("transformer_layers_per_block") or []),
            "down_block_types": list(cfg.get("down_block_types") or []),
            "cross_attention_dim": int(cfg.get("cross_attention_dim") or 0),
            "in_channels": int(cfg.get("in_channels") or 4),
        }

    return DenoiserGeometry(
        model_class=model_class,
        family=family,
        hidden=hidden,
        num_double_layers=n_double,
        num_single_layers=n_single,
        head_dim=head_dim,
        intermediate=intermediate,
        gated_ffn=gated,
        num_experts=num_experts,
        active_experts=active,
        moe_intermediate=moe_inter,
        vae_spatial=int(defaults.get("vae_spatial", 8)),
        patch=patch,
        text_tokens=int(defaults.get("text_tokens", 256)),
        default_steps=int(defaults.get("default_steps", 28)),
        default_cfg_batch=int(defaults.get("default_cfg_batch", 2)),
        unet=unet_geom,
        notes=basename,
    )


# ---------------------------------------------------------------------------
# Per-family single-forward FLOPs.
# ---------------------------------------------------------------------------
def _image_tokens(g: DenoiserGeometry, height: int, width: int) -> int:
    lat_h = max(height // g.vae_spatial // g.patch, 1)
    lat_w = max(width // g.vae_spatial // g.patch, 1)
    return int(lat_h * lat_w)


def _ffn_per_token(g: DenoiserGeometry) -> float:
    return _ffn_flops(1.0, g.hidden, g.intermediate, gated=g.gated_ffn)


def _moe_ffn_per_token(g: DenoiserGeometry) -> float:
    return g.active_experts * _ffn_flops(1.0, g.hidden, g.moe_intermediate, gated=True)


def _dual_stream_layer(g: DenoiserGeometry, ti: int, tt: int, ffn_pt: float) -> float:
    """One MMDiT double-stream block: separate img/txt QKVO+FFN, joint attention."""
    lin = (ti + tt) * _qkvo_flops(1.0, g.hidden) + (ti + tt) * ffn_pt
    attn = _full_attention_flops(ti + tt, g.hidden)
    return lin + attn


def _single_stream_layer(g: DenoiserGeometry, s: int, ffn_pt: float) -> float:
    """One single-stream block over ``s`` concatenated tokens."""
    lin = s * _qkvo_flops(1.0, g.hidden) + s * ffn_pt
    attn = _full_attention_flops(s, g.hidden)
    return lin + attn


def _sana_layer(g: DenoiserGeometry, ti: int, tt: int, ffn_pt: float) -> float:
    """Sana block: linear self-attention over image tokens + cross-attn to text."""
    lin = ti * _qkvo_flops(1.0, g.hidden) + ti * ffn_pt
    self_attn = _linear_attention_flops(ti, g.hidden, g.head_dim or 32)
    cross = _cross_attention_flops(ti, tt, g.hidden) + tt * _linear_flops(1.0, g.hidden, g.hidden) * 2
    return lin + self_attn + cross


def _unet_forward_flops(g: DenoiserGeometry, height: int, width: int) -> float:
    """SDXL-style UNet: resnet convs + self/cross attention across resolutions.

    Down path (+ symmetric up path with one extra resnet per block) and mid.
    Conv FLOPs for a 3x3 conv over an HxW feature map with Cin->Cout channels:
    2 * H * W * Cin * Cout * 9. Attention is over the flattened spatial tokens.
    """
    u = g.unet
    chans = u.get("block_out_channels") or [320, 640, 1280]
    lpb = int(u.get("layers_per_block") or 2)
    tlpb = u.get("transformer_layers_per_block") or [0] * len(chans)
    if isinstance(tlpb, int):
        tlpb = [tlpb] * len(chans)
    ctx = int(u.get("cross_attention_dim") or 2048)
    txt = g.text_tokens
    lat = max(height // g.vae_spatial, 1)  # SDXL latent spatial (128 for 1024)

    total = 0.0
    for lvl, cout in enumerate(chans):
        # spatial halves at each deeper level
        sp = max(lat // (2 ** lvl), 1)
        s_tokens = sp * sp
        cin = chans[lvl - 1] if lvl > 0 else cout
        # resnets: two 3x3 convs each; first resnet maps cin->cout, rest cout->cout
        for i in range(lpb):
            c0 = cin if i == 0 else cout
            total += 2.0 * s_tokens * c0 * cout * 9  # conv1
            total += 2.0 * s_tokens * cout * cout * 9  # conv2
        # transformer blocks (self-attn over spatial + cross-attn to text)
        depth = int(tlpb[lvl]) if lvl < len(tlpb) else 0
        for _ in range(depth):
            total += s_tokens * _qkvo_flops(1.0, cout)
            total += _full_attention_flops(s_tokens, cout)
            total += _cross_attention_flops(s_tokens, txt, cout)
            total += txt * _linear_flops(1.0, ctx, cout) * 2  # cross kv proj
            total += s_tokens * _ffn_flops(1.0, cout, 4 * cout)  # ff
    # up path ~ down path with an extra resnet per block; approximate as 1.5x down.
    down_and_mid = total
    return down_and_mid * 2.5  # down + mid(~small) + up(~1.5x down)


def forward_flops(g: DenoiserGeometry, height: int, width: int) -> dict[str, float]:
    """FLOPs for ONE denoiser forward pass (no CFG, no step scaling)."""
    if g.family == "unet":
        total = _unet_forward_flops(g, height, width)
        return {"forward_flops": total, "image_tokens": 0, "text_tokens": g.text_tokens}

    ti = _image_tokens(g, height, width)
    tt = g.text_tokens
    if g.family == "moe_single" or g.family == "moe_flux":
        ffn_pt = _moe_ffn_per_token(g)
    else:
        ffn_pt = _ffn_per_token(g)

    total = 0.0
    if g.family in ("mmdit",):
        for _ in range(g.num_double_layers):
            total += _dual_stream_layer(g, ti, tt, ffn_pt)
    elif g.family in ("flux", "moe_flux"):
        for _ in range(g.num_double_layers):
            total += _dual_stream_layer(g, ti, tt, ffn_pt)
        for _ in range(g.num_single_layers):
            total += _single_stream_layer(g, ti + tt, ffn_pt)
    elif g.family in ("single", "moe_single"):
        for _ in range(g.num_double_layers + g.num_single_layers):
            total += _single_stream_layer(g, ti + tt, ffn_pt)
    elif g.family == "sana":
        for _ in range(g.num_double_layers + g.num_single_layers):
            total += _sana_layer(g, ti, tt, ffn_pt)
    return {"forward_flops": total, "image_tokens": ti, "text_tokens": tt}


def estimate_image_flops(
    model_dir: str | Path,
    *,
    height: int = 1024,
    width: int = 1024,
    num_steps: int | None = None,
    cfg_batch: int | None = None,
) -> dict[str, Any] | None:
    """Full per-image FLOPs = forward x steps x cfg_batch.

    Args:
        model_dir: Local diffusers model directory.
        height/width: Output image resolution in pixels.
        num_steps: Denoise steps (default: family default).
        cfg_batch: Forwards per step (2 with classifier-free guidance, else 1).

    Returns:
        Dict with the geometry, per-forward / per-step / total FLOPs, or
        ``None`` when the model geometry cannot be resolved.
    """
    g = resolve_geometry(model_dir)
    if g is None:
        return None
    steps = int(num_steps if num_steps is not None else g.default_steps)
    cfgb = int(cfg_batch if cfg_batch is not None else g.default_cfg_batch)
    fwd = forward_flops(g, height, width)
    per_step = fwd["forward_flops"] * cfgb
    total = per_step * steps
    return {
        "model_class": g.model_class,
        "family": g.family,
        "hidden": g.hidden,
        "layers": g.num_double_layers + g.num_single_layers,
        "num_experts": g.num_experts,
        "active_experts": g.active_experts,
        "image_tokens": fwd["image_tokens"],
        "text_tokens": fwd["text_tokens"],
        "height": height,
        "width": width,
        "num_steps": steps,
        "cfg_batch": cfgb,
        "forward_flops": fwd["forward_flops"],
        "per_step_flops": per_step,
        "total_flops": total,
    }


def analytic_ceiling(
    model_dir: str | Path,
    *,
    gpu_type: str = "mi355x",
    precision: str = "bf16",
    height: int = 1024,
    width: int = 1024,
    num_steps: int | None = None,
    cfg_batch: int | None = None,
) -> dict[str, Any] | None:
    """Analytic compute-bound ceiling (ideal ms) for one image.

    Returns the FLOPs estimate plus ``peak_tflops`` and ``ideal_ms`` =
    total_flops / (peak * 1e12) * 1e3. ``ideal_ms`` is absent when the
    (gpu, precision) peak is unknown.
    """
    est = estimate_image_flops(
        model_dir, height=height, width=width, num_steps=num_steps, cfg_batch=cfg_batch
    )
    if est is None:
        return None
    pk = peak_tflops(gpu_type, precision)
    est["gpu_type"] = gpu_type
    est["precision"] = precision
    est["peak_tflops"] = pk
    if pk > 0:
        est["ideal_ms"] = est["total_flops"] / (pk * 1e12) * 1e3
        est["ideal_ms_per_step"] = est["per_step_flops"] / (pk * 1e12) * 1e3
    return est


def _fmt(est: dict[str, Any]) -> str:
    tf = est["total_flops"] / 1e12
    line = (
        f"{est['model_class']:<34} {est['family']:<10} "
        f"h={est['hidden']:<5} L={est['layers']:<3} "
        f"tok={est['image_tokens']:<5}+{est['text_tokens']:<4} "
        f"steps={est['num_steps']:<3}x{est['cfg_batch']} "
        f"{tf:8.1f} TFLOP/img"
    )
    if "ideal_ms" in est:
        line += f"  ideal={est['ideal_ms']:8.1f} ms ({est['peak_tflops']:.0f} TFLOPS)"
    return line


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-architecture diffusion FLOPs / compute ceiling")
    ap.add_argument("--model-dir", required=True, help="Local diffusers model directory.")
    ap.add_argument("--gpu-type", default="mi355x")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=0, help="Override denoise steps (0 = family default).")
    ap.add_argument("--cfg-batch", type=int, default=0, help="Override forwards/step (0 = family default).")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    est = analytic_ceiling(
        args.model_dir,
        gpu_type=args.gpu_type,
        precision=args.precision,
        height=args.height,
        width=args.width,
        num_steps=args.steps or None,
        cfg_batch=args.cfg_batch or None,
    )
    if est is None:
        print(f"could not resolve denoiser geometry for {args.model_dir}")
        return 1
    print(json.dumps(est, indent=2) if args.json else _fmt(est))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
