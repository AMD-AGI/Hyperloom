# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Model-agnostic library of fusible op-chain patterns (the "hybrid" discovery).

Each :class:`FusionPattern` maps a set of launch-bound trace categories to a
fusion HYPOTHESIS: a description, what to grep for in the model source, the fused
math sketch handed to the author, and -- critically -- how to build the
correctness reference by IMPORTING the real eager ops (never re-implemented).

v1 covers the fusions already validated in kernel/docs (ZAYA, LFM2). New patterns
are added here, not per-model; the locate stage confirms/localizes them against
the actual framework source.
"""

from __future__ import annotations

from .models import Diagnosis, FusionPattern

_SGLANG_VLLM = frozenset({"sglang", "vllm", "vllm-aiter"})

# ROCm authoring guard appended to every pattern's fusion_math: sglang/vllm ship
# some CUDA-only fused ops (notably `fused_qk_norm_rope`: `cuda_bf16.h` + nvcc-only
# `--use_fast_math`) that fail to build on ROCm. Prefer a ROCm-native Triton/aiter
# kernel and verify it compiles + runs on the target GPU (hipcc), not just parity.
_ROCM_GUARD = (
    " [ROCm] Author a ROCm-native Triton (or aiter) kernel; do NOT reuse a "
    "framework CUDA-only fused op (e.g. `fused_qk_norm_rope`). Verify it BUILDS "
    "and RUNS on the target GPU, not only numerical parity."
)

PATTERNS: tuple[FusionPattern, ...] = (
    FusionPattern(
        id="residual_add_rmsnorm",
        trigger_categories=frozenset({"add", "rmsnorm"}),
        min_trigger_share=0.10,
        description="Fold the residual-add into the following RMSNorm (fused add+rmsnorm, llama-style residual threading).",
        source_hints=(
            "hidden_states = hidden_states + residual",
            "+ residual",
            "RMSNorm(",
            "input_layernorm",
            "post_attention_layernorm",
            "ffn_norm",
        ),
        fusion_math=(
            "For each decoder layer, replace the standalone `x = x + residual; y = norm(x)` "
            "with a fused add+rmsnorm `y, residual = norm(x, residual)`. Thread `residual` "
            "across layers; close the final add into the last norm. Prefer the framework's "
            "fused add+rmsnorm ONLY if it has a ROCm (aiter/HIP) implementation; otherwise "
            "author a Triton kernel computing rmsnorm(x + residual) in one pass." + _ROCM_GUARD
        ),
        eager_reference_hint=(
            "Reference = the framework's own RMSNorm eager forward applied to (x + residual). "
            "Import the real RMSNorm class from the framework and call its forward; do NOT "
            "re-implement rmsnorm."
        ),
        env_flag="FUSED_RESIDUAL",
        frameworks=_SGLANG_VLLM,
        fused_markers=(r"fused_add_rmsnorm", r"add_rmsnorm", r"norm\([^)\n]*,\s*residual"),
    ),
    FusionPattern(
        id="swiglu_silu_mul",
        trigger_categories=frozenset({"activation", "mul"}),
        min_trigger_share=0.03,
        description="Merge the gate/up SwiGLU projections into one GEMM and use the fused SiluAndMul kernel.",
        source_hints=(
            "F.silu(",
            "silu(gate) * up",
            "self.w1(",
            "self.w3(",
            "gate_up_proj",
            "SiluAndMul",
        ),
        fusion_math=(
            "Replace two separate gate/up projections + eager `F.silu(gate) * up` with a single "
            "MergedColumnParallelLinear([intermediate]*2) GEMM followed by the framework's fused "
            "`SiluAndMul` activation. Update weight loading to map gate->shard0, up->shard1." + _ROCM_GUARD
        ),
        eager_reference_hint=(
            "Reference = eager `F.silu(gate) * up` on the same inputs. For the merged-GEMM part, "
            "compare against the two original Linear ops; import the framework SiluAndMul for the "
            "fused activation. Do NOT re-implement silu."
        ),
        env_flag="FUSED_SILU",
        frameworks=_SGLANG_VLLM,
        fused_markers=(r"SiluAndMul", r"gate_up_proj"),
    ),
    FusionPattern(
        id="scaled_residual_add_rmsnorm",
        trigger_categories=frozenset({"add", "mul", "rmsnorm"}),
        min_trigger_share=0.08,
        description="Fuse per-branch `residual + branch*scalar` then RMSNorm (Granite muP residual_multiplier).",
        source_hints=(
            "residual_multiplier",
            "attention_multiplier",
            "* self.residual_multiplier",
            "residual + ",
            "input_layernorm",
            "post_attention_layernorm",
        ),
        fusion_math=(
            "Fuse `new_residual = branch*scale + residual; out = rmsnorm(new_residual, w)` into one "
            "Triton kernel (`scaled_add_rmsnorm`), plus a `scaled_add` for the final branch that has "
            "no immediately-following norm. For residual-threaded models (Granite dense) fold the "
            "scalar into the NEXT layer's `input_layernorm` and the final `model.norm` by returning "
            "the RAW branch output." + _ROCM_GUARD
        ),
        eager_reference_hint=(
            "Reference = import the framework RMSNorm and compare `rmsnorm(x*scale + r)` on "
            "representative tensors. Author template: kernel/docs/fusion_templates/granite_fused.py."
        ),
        env_flag="GRANITE_FUSED_RESIDUAL",
        frameworks=_SGLANG_VLLM,
        fused_markers=(r"scaled_add_rmsnorm", r"GRANITE_FUSED"),
    ),
    FusionPattern(
        id="hybrid_scale_combine",
        trigger_categories=frozenset({"mul", "add"}),
        min_trigger_share=0.06,
        description="Fuse hybrid attn+mamba input-prescale and output-combine scalar muls (Falcon-H1).",
        source_hints=(
            "attn_in_mult",
            "ssm_in_mult",
            "attn_out_mult",
            "ssm_out_mult",
            "key_multiplier",
            "* self.attention_in_multiplier",
        ),
        fusion_math=(
            "(a) prescale: read `hidden` once, emit `hidden*attn_in_mult` and `hidden*ssm_in_mult` "
            "(2 muls + 2 reads -> 1 kernel); (b) combine: `attn_out*attn_out_mult + "
            "mamba_out*ssm_out_mult` in one kernel." + _ROCM_GUARD
        ),
        eager_reference_hint=(
            "Reference = the eager scalar muls / combine on representative tensors. Author "
            "template: kernel/docs/fusion_templates/falcon_h1_fused.py."
        ),
        env_flag="FALCON_H1_FUSED_SCALES",
        frameworks=_SGLANG_VLLM,
        fused_markers=(r"FALCON_H1_FUSED", r"fused_scales"),
    ),
    FusionPattern(
        id="qk_norm_rope",
        # Raised from 0.04: on dense Qwen3 the QK-norm+RoPE tail measured only
        # ~+0.3% (and sglang's fused_qk_norm_rope is CUDA-only). The predicted-gain
        # gate is the primary filter; this keeps the pattern from over-triggering.
        trigger_categories=frozenset({"rmsnorm", "rope"}),
        min_trigger_share=0.12,
        description="Fuse per-head Q/K RMSNorm (+ any grouped blend / temperature) with RoPE into one kernel.",
        source_hints=(
            "q_norm",
            "k_norm",
            "_normalize_qk",
            "_add_grouped_qk_means",
            "rotary_emb(",
            "apply_qk_norm",
            "clamp_temp",
        ),
        fusion_math=(
            "Collapse the per-(token,k-head) QK post-processing chain -- grouped-mean blend (if "
            "present) -> RMSNorm(rsqrt) -> optional temperature -> (optionally RoPE) -- into one "
            "Triton kernel. A natural grid is one program per (token, k-head) looping the GQA "
            "q-heads inside; outputs match the eager fp32 dtype." + _ROCM_GUARD
        ),
        eager_reference_hint=(
            "Reference = the model's real eager QK methods (e.g. `_add_grouped_qk_means` + "
            "`_normalize_qk`, or `q_norm`/`k_norm` + `rotary_emb`). Import and call them directly "
            "on representative q/k tensors; do NOT re-derive the math."
        ),
        env_flag="FUSED_QK",
        frameworks=_SGLANG_VLLM,
        fused_markers=(r"fused_qk_norm", r"fused_qk_norm_rope", r"fused_qk_norm_mrope"),
    ),
    FusionPattern(
        id="dual_affine_scaling",
        trigger_categories=frozenset({"add", "mul", "elementwise"}),
        min_trigger_share=0.06,
        description="Fuse a dual (x + bias) * scale affine on the hidden (and residual) streams into one kernel.",
        source_hints=(
            "ResidualScaling",
            "residual_scale",
            "residual_bias",
            "* scale",
            "(x + bias)",
            "has_residual",
        ),
        fusion_math=(
            "Fuse `(x + bias) * scale` applied per-row over the hidden dim on both the hidden and "
            "(when present) residual streams into a single Triton kernel; fp32 output." + _ROCM_GUARD
        ),
        eager_reference_hint=(
            "Reference = the model's eager affine (e.g. `ResidualScaling.forward`). Import and call "
            "it on representative tensors; do NOT re-implement the affine."
        ),
        env_flag="FUSED_RESIDUAL_SCALE",
        frameworks=_SGLANG_VLLM,
        fused_markers=(r"fused_residual_scaling",),
    ),
)


def match_patterns(diagnosis: Diagnosis, framework: str) -> list[tuple[FusionPattern, float]]:
    """Return fusion patterns triggered by a diagnosis, ranked by trigger share.

    A pattern triggers when (a) the framework matches, and (b) the combined
    GPU-busy-time share of its ``trigger_categories`` present in the trace meets
    ``min_trigger_share``. The returned share is that combined trigger share,
    used to rank competing hypotheses.

    Args:
        diagnosis: Stage-1 diagnosis (carries ``category_shares``).
        framework: Target framework (``sglang`` / ``vllm`` / ``vllm-aiter``).

    Returns:
        ``[(pattern, trigger_share), ...]`` sorted by descending trigger share.
        Empty when the diagnosis is not a candidate or nothing triggers.
    """
    if not diagnosis.is_candidate:
        return []
    fw = (framework or "").strip().lower()
    shares = diagnosis.category_shares or {}
    out: list[tuple[FusionPattern, float]] = []
    for pat in PATTERNS:
        if fw and fw not in pat.frameworks:
            continue
        trigger_share = sum(shares.get(c, 0.0) for c in pat.trigger_categories)
        if trigger_share >= pat.min_trigger_share:
            out.append((pat, trigger_share))
    out.sort(key=lambda ps: ps[1], reverse=True)
    return out
