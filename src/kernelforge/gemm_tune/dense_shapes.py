# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared dense-GEMM shape derivation from a model config.

Single source of truth for the (N, K) projection shapes and the M (batch)
coverage used by every dense tuner (bf16 and fp8 a8w8*/a4w4*). Deriving shapes
from config.json means a dense tuner never needs an externally-recorded CSV to
run -- it can always synthesize one. The fp8 dense path previously skipped when
no CSV was supplied; reusing this logic lets it run on every model.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# sglang CUDAGraph capture batch sizes - key decode sizes (thorough mode).
_SGLANG_CUDAGRAPH_BS_THOROUGH = [
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
]


def compute_dense_nk_shapes(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    tp: int,
    *,
    head_dim: int = 0,
    v_head_dim: int = 0,
    q_lora_rank: int = 0,
    kv_lora_rank: int = 0,
    qk_nope_head_dim: int = 0,
    qk_rope_head_dim: int = 0,
    o_lora_rank: int = 0,
    o_groups: int = 0,
) -> list[tuple[int, int]]:
    """Compute the dense GEMM (N, K) = (output_dim, input_dim) projection shapes.

    Three regimes (selected by the dims supplied):

    * **MLA** (DeepSeek-V3 family, ``q_lora_rank`` + ``kv_lora_rank`` set): the
      low-rank attention path -- fused q_a+kv_a down-proj (replicated), q_b, kv_b,
      o_proj -- plus the dense FFN.
    * **DeepSeek-V4 sparse MLA** (``q_lora_rank`` without ``kv_lora_rank``): the
      fused wqa+wkv down-proj (replicated) and q_b up-proj. ``wo_a``/``wo_b`` are
      omitted here because ``wo_a`` is a batched GEMM in vLLM, not a plain dense
      linear.
    * **Separate qk/v head dims** (e.g. MiMo: qk ``head_dim`` != ``v_head_dim``):
      a GQA-style fused QKV that sizes K and V with their own head dims, o_proj
      over ``num_heads * v_head_dim``, plus the dense FFN.
    * **Generic Llama** (no extra dims): equivalent to the historical formula
      (QKV/O/gate+up/down) -- the branches below reduce to it when
      ``v_head_dim == head_dim == hidden_size // num_heads``.

    Only the attention + dense-FFN GEMMs are returned; per-expert MoE GEMMs are
    tuned by the MoE tuners, not the dense path. Degenerate and duplicate shapes
    are dropped while preserving order.
    """
    tp = max(1, int(tp or 1))
    nh = max(1, int(num_heads or 1))
    nkv = max(1, int(num_kv_heads or nh))
    qk_head = int(head_dim or 0) or (hidden_size // nh if hidden_size else 0)
    vh = int(v_head_dim or 0) or qk_head

    shapes: list[tuple[int, int]] = []
    if q_lora_rank and kv_lora_rank:
        # MLA: q_a + kv_a_with_mqa are low-rank down-projections, kept full
        # (not tensor-parallel sharded); q_b/kv_b/o_proj are sharded by tp.
        qk_h = (qk_nope_head_dim + qk_rope_head_dim) or qk_head
        shapes.append((q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden_size))
        shapes.append((nh * qk_h // tp, q_lora_rank))
        shapes.append((nh * (qk_nope_head_dim + vh) // tp, kv_lora_rank))
        shapes.append((hidden_size, nh * vh // tp))
    elif q_lora_rank and qk_head:
        # DeepSeek-V4 sparse MLA: fused_wqa_wkv is replicated; wq_b is column-sharded.
        shapes.append((q_lora_rank + qk_head, hidden_size))
        shapes.append((nh * qk_head // tp, q_lora_rank))
    else:
        # Standard / GQA attention with possibly distinct qk vs v head dims.
        qkv_out = (nh * qk_head + nkv * qk_head + nkv * vh) // tp
        shapes.append((qkv_out, hidden_size))  # fused QKV
        shapes.append((hidden_size, nh * vh // tp))  # O

    # Dense FFN (SwiGLU gate+up fused, then down). MoE-only models may omit it.
    if intermediate_size > 0:
        shapes.append((intermediate_size * 2 // tp, hidden_size))  # gate+up
        shapes.append((hidden_size, intermediate_size // tp))  # down

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for n, k in shapes:
        if n > 0 and k > 0 and (n, k) not in seen:
            seen.add((n, k))
            out.append((n, k))
    return out


_DECODE_M_GRID = (1, 4, 16, 32, 64, 128, 256)


def compute_decode_m_values(conc: int) -> list[int]:
    """Decode-step M (batch) sizes: ``M ≈ num_running_requests``, capped by ``conc``.

    Decode steps emit one token per running request, so they run far more often
    than prefill and dominate serving token throughput. Tuning must cover this
    band, otherwise a config tuned only for a large prefill ``M`` gets applied to
    the (throughput-dominant) small-``M`` decode GEMMs and regresses E2E even
    when the micro benchmark wins.

    ``M`` above ``conc`` cannot occur -- the scheduler never runs more requests
    than the concurrency cap -- so the grid is clamped rather than fixed. ``conc``
    itself is always included: steady-state decode sits at the cap, making it the
    single most-executed decode shape. ``1`` is always included as the ramp-up /
    tail boundary.
    """
    conc = max(1, int(conc or 64))
    m_set = {m for m in _DECODE_M_GRID if m <= conc}
    m_set.update((1, conc))
    return sorted(m_set)


def compute_dense_m_values(
    conc: int,
    thorough: bool = False,
    isl: int = 0,
    osl: int = 0,
    max_model_len: int = 0,
) -> list[int]:
    """Compute M (batch) values to tune.

    Fast mode: decode sizes plus prefill-representative sizes derived from ISL
    and concurrency. Thorough mode: full CUDAGraph capture list plus a dense
    prefill grid.

    The M dimension of a serving GEMM equals the total tokens in a batch step.
    Decode steps: M ≈ num_running_requests (bounded by ``conc``). Prefill
    steps: M ≈ chunked_prefill_size or ISL × batch (can reach thousands).
    Tuning only small M values misses the prefill-heavy hot path entirely.
    """
    conc = int(conc or 64)
    isl = int(isl or 0)

    if thorough:
        m_set = set(_SGLANG_CUDAGRAPH_BS_THOROUGH)
        m_set.update([256, 512, 1024])
        if conc >= 128:
            m_set.update([2048, 4096])
        if isl >= 512:
            # Cap ISL-derived M at the same 16384 high-watermark as the
            # concurrency term: a long-context ISL (e.g. ~32k) would otherwise
            # tune M=32k/65k giant GEMMs -> huge tune time / OOM.
            m_set.update([min(isl, 16384), min(isl * 2, 16384), min(isl * conc // 8, 16384)])
        return sorted(m_set)

    # Decode-representative sizes (throughput-dominant small M).
    m_set = set(compute_decode_m_values(conc))

    # Prefill-representative sizes: chunked prefill typically processes ISL
    # tokens per step; high concurrency multiplies that. Add ISL, a mid-range
    # prefill batch, and the practical high-watermark.
    if isl >= 256:
        # Cap at the same 8192 high-watermark as the terms below; a long-context
        # ISL would otherwise inject an M=32k-class giant GEMM (tune time / OOM).
        m_set.add(min(isl, 8192))
    if isl >= 512:
        m_set.add(min(isl * conc // 16, 8192))
    if conc >= 32:
        m_set.add(min(conc * 128, 8192))

    return sorted(m_set)


def write_mnk_untuned_csv(
    nk_shapes: list[tuple[int, int]],
    m_values: list[int],
    output_path: Path,
    *,
    needs_q_dtype_w: bool = False,
    q_dtype: str = "",
    filename: str = "untuned_dense.csv",
) -> Path:
    """Write an aiter-style untuned CSV for the fp8/fp4 dense tuners.

    blockscale / a4w4 expect ``M,N,K``; a8w8 / bpreshuffle additionally expect a
    ``q_dtype_w`` column. Mirrors :func:`_shapes_json_to_csv` so downstream aiter
    scripts accept the file unchanged.
    """
    csv_path = output_path / filename
    output_path.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        if needs_q_dtype_w:
            from .tuners._aiter_dense_common import _aiter_fp8_dtype_str

            effective_q = q_dtype or _aiter_fp8_dtype_str()
            f.write("M,N,K,q_dtype_w\n")
            for m in m_values:
                for n, k in nk_shapes:
                    f.write(f"{m},{n},{k},{effective_q}\n")
        else:
            f.write("M,N,K\n")
            for m in m_values:
                for n, k in nk_shapes:
                    f.write(f"{m},{n},{k}\n")
    log.info(
        "Derived %d dense shapes from config -> %s",
        len(m_values) * len(nk_shapes),
        csv_path,
    )
    return csv_path
