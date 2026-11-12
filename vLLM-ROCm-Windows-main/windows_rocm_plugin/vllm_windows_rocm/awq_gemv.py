# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Fast M=1 W4A16 dequant-GEMV for single-stream decode on RDNA3, wired as an MPLinearKernel
registered ahead of conch in vLLM's ROCm kernel priority.

Why: on ROCm/Windows, AWQ uint4 has no fast kernel -- vLLM routes AWQ -> awq_marlin ->
choose_mp_linear_kernel([Conch, Exllama]); Exllama rejects uint4 (only uint4b8), Marlin is
CUDA-only, so ONLY conch remains, which runs a throughput tile (block_m=128, tl.dot needs >=16
rows) for a single decode row -> ~21x off bandwidth at M=1. This is a true GEMV (reduction, no
tl.dot / split-K / atomicAdd / zero-init).

Layout: vLLM's awq_marlin path delivers weights in a packed-along-K layout. We REUSE conch's
process_weights_after_loading to normalize (and conch itself for the M>1 prefill path), so this
kernel consumes conch's exact post-process layout:
  w_q : [K//8, N]   int32, packed along K (8 K-values per int32, straight order shift=(k%8)*4)
  w_s : [K//G, N]   fp16  group scales
  w_zp: [K//G, N]   uint8  UNPACKED zero points
Dequant (conch SYMMETRIC_WITH_SHIFT, weight_bias=0 for uint4): w[k,n] = (q(k,n) - z(g,n))*s(g,n).
Validated by token-agreement vs conch on the real model (run/precision_check.py).
"""
import logging
import os
import sys

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

from .cops import _default_dir  # noqa: E402  (same package; cops has no plugin imports)

# Optional hand-written HIP GEMV (buffer_load dwordx4 + U=8 unroll + split-K) -- beats the Triton
# kernel on every shape cache-cold (o 24->36%, qkv 33->52%, down 35->61%, gate 67->71% of DRAM).
# Loaded if built (experiments/w4_gemv/hip/build_run_hip.bat); set VLLM_WIN_HIPGEMV=0 to force Triton.
_HIP_OK = False        # set True once the HIP .pyd is imported (registers torch.ops.vllm_win_hip)
_HIP_TRIED = False


def _load_hip_gemv():
    """Import the HIP GEMV .pyd EAGERLY (at registration), so its TORCH_LIBRARY op is available
    and no import runs inside the dynamo-traced forward. Sets _HIP_OK."""
    global _HIP_OK, _HIP_TRIED
    if _HIP_TRIED:
        return
    _HIP_TRIED = True
    # Opt-in (default OFF): the HIP GEMV is faster in the cold microbench but currently LOSES
    # end-to-end (44 vs Triton-autotune 50.9) -- its split-K atomic+zeros+cast per call costs more
    # than its bandwidth edge. Needs an atomic-free direct-write path to win e2e. Triton is default.
    if os.environ.get("VLLM_WIN_HIPGEMV", "0") != "1":
        return
    d = os.environ.get("VLLM_WIN_HIPGEMV_DIR") or os.path.join(_default_dir("vw_hipgemv_build"), "gemv_w4_hip")
    try:
        if d not in sys.path:
            sys.path.insert(0, d)
        import gemv_w4_hip  # noqa: F401  -- import triggers the TORCH_LIBRARY(vllm_win_hip) static init
        _ = torch.ops.vllm_win_hip.gemv_w4  # ensure the op resolved
        _HIP_OK = True
        logger.info("loaded native HIP W4 GEMV (torch.ops.vllm_win_hip.gemv_w4) from %s", d)
    except Exception as e:  # noqa: BLE001
        logger.warning("HIP GEMV not available, using Triton: %r", e)


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn}, num_warps=nw)
             for bn in (8, 16, 32) for nw in (1, 2)],
    key=["K", "N"],
)
@triton.jit
def _gemv_k_kernel(
    a_ptr, qw_ptr, s_ptr, z_ptr, c_ptr,
    K, N,
    BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr,   # quant group size (== K rows processed per iteration)
):
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)         # output columns
    nmask = n < N
    ROWS: tl.constexpr = GROUP // 8                   # int32 rows per group
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)       # straight order
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    num_groups = K // GROUP
    for g in range(0, num_groups):
        k0 = g * GROUP
        row = (k0 // 8) + tl.arange(0, ROWS)          # int32 rows for this group
        qw = tl.load(qw_ptr + row[:, None] * N + n[None, :],
                     mask=nmask[None, :], other=0)     # [ROWS, BLOCK_N] int32
        # unpack along K: [ROWS, BLOCK_N] -> [ROWS, 8, BLOCK_N] -> [GROUP, BLOCK_N]
        q = (qw[:, None, :] >> shifts[None, :, None]) & 0xF
        q = tl.reshape(q, (GROUP, BLOCK_N)).to(tl.float32)   # k-index = row*8 + nibble

        a = tl.load(a_ptr + k0 + tl.arange(0, GROUP)).to(tl.float32)  # [GROUP]
        contrib = tl.sum(a[:, None] * q, axis=0)      # sum_k a*q  -> [BLOCK_N]
        asum = tl.sum(a)                               # sum_k a (scalar)

        s = tl.load(s_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        # sum_k a*(q - z)*s = s * (sum_k a*q - z*sum_k a)
        acc += (contrib - z * asum) * s

    tl.store(c_ptr + n, acc.to(c_ptr.type.element_ty), mask=nmask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn}, num_warps=nw)
             for bn in (8, 16, 32) for nw in (1, 2)],
    key=["K", "N"],
)
@triton.jit
def _gemv_k_sym_kernel(
    a_ptr, qw_ptr, s_ptr, c_ptr,
    K, N,
    BLOCK_N: tl.constexpr,
    GROUP: tl.constexpr,
    BIAS: tl.constexpr,   # symmetric zero-point (8 for uint4b8); dequant = (q - BIAS) * s
):
    pid = tl.program_id(0)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N
    ROWS: tl.constexpr = GROUP // 8
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    num_groups = K // GROUP
    for g in range(0, num_groups):
        k0 = g * GROUP
        row = (k0 // 8) + tl.arange(0, ROWS)
        qw = tl.load(qw_ptr + row[:, None] * N + n[None, :],
                     mask=nmask[None, :], other=0)
        q = (qw[:, None, :] >> shifts[None, :, None]) & 0xF
        q = tl.reshape(q, (GROUP, BLOCK_N)).to(tl.float32)
        a = tl.load(a_ptr + k0 + tl.arange(0, GROUP)).to(tl.float32)
        contrib = tl.sum(a[:, None] * q, axis=0)
        asum = tl.sum(a)
        s = tl.load(s_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        # sum_k a*(q - BIAS)*s = s * (sum_k a*q - BIAS*sum_k a)
        acc += (contrib - BIAS * asum) * s
    tl.store(c_ptr + n, acc.to(c_ptr.type.element_ty), mask=nmask)


def gemv_k_sym(a, w_q, w_s, group_size, bias):
    # a: [1, K] ; w_q: [K//8, N] int32 packed-K ; w_s: [K//G, N] ; symmetric (no zero points)
    K = a.shape[-1]
    N = w_s.shape[1]
    c = torch.empty((a.shape[0], N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_k_sym_kernel[grid](a, w_q, w_s, c, K, N, GROUP=group_size, BIAS=bias)
    return c


def gemv_k(a, w_q, w_s, w_zp, group_size):
    # a: [1, K] fp16 ; w_q: [K//8, N] int32 ; w_s: [K//G, N] fp16 ; w_zp: [K//G, N] uint8
    # @triton.autotune picks BLOCK_N/num_warps per (K,N): cache-cold, small-N favors BLOCK_N=8,
    # large-N (gate_up) BLOCK_N=16. Autotune runs during vLLM's eager warmup, before cudagraph capture.
    K = a.shape[-1]
    N = w_s.shape[1]
    c = torch.empty((a.shape[0], N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _gemv_k_kernel[grid](a, w_q, w_s, w_zp, c, K, N, GROUP=group_size)
    return c


def _register_kernel_class():
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearKernel,
    )
    from vllm.scalar_type import scalar_types

    class WinRocmAwqGemvKernel(MPLinearKernel):
        """M=1 W4A16 GEMV on conch's normalized layout; conch handles weight prep + M>1."""

        @classmethod
        def get_min_capability(cls) -> int:
            return 0

        @classmethod
        def can_implement(cls, c):
            if c.weight_type != scalar_types.uint4:
                return False, "WinRocmAwqGemv supports only uint4 (AWQ)"
            if not c.zero_points:
                return False, "WinRocmAwqGemv needs zero points (AWQ)"
            if c.has_g_idx:
                return False, "WinRocmAwqGemv does not support act-order g_idx"
            if c.group_size != 128:
                return False, "WinRocmAwqGemv supports only group_size 128"
            return True, None

        def process_weights_after_loading(self, layer) -> None:
            # Reuse conch's normalization so we consume its exact post-process layout, and keep
            # the conch instance for the M>1 (prefill/batched) path.
            from vllm.model_executor.kernels.linear.mixed_precision.conch import (
                ConchLinearKernel,
            )
            self._conch = ConchLinearKernel(
                self.config, self.w_q_name, self.w_s_name, self.w_zp_name, self.w_gidx_name
            )
            self._conch.process_weights_after_loading(layer)

        def apply_weights(self, layer, x, bias=None):
            x2d = x.reshape(-1, x.shape[-1])
            if x2d.shape[0] == 1:
                w_q, w_s, w_zp, _ = self._get_weight_params(layer)
                N = w_s.shape[1]
                if _HIP_OK:
                    # TORCH_LIBRARY op (registered when the .pyd is imported at registration):
                    # dynamo-traceable + cudagraph-safe, like the _C ops. split_groups=8 (best cold).
                    out = torch.ops.vllm_win_hip.gemv_w4(
                        x2d.contiguous(), w_q, w_s, w_zp, self.config.group_size, 8)
                else:
                    # NOTE: the torch custom-op wrapper was tested to remove inductor graph breaks
                    # but regressed (opaque op disrupts FULL_DECODE_ONLY cudagraph/autotune).
                    out = gemv_k(x2d.contiguous(), w_q, w_s, w_zp, self.config.group_size)
                if bias is not None:
                    out = out + bias
                return out.reshape(x.shape[:-1] + (N,))
            # prefill / batched: delegate to conch (correct, same normalized weights)
            return self._conch.apply_weights(layer, x, bias)

    return WinRocmAwqGemvKernel


def _register_dequant_kernel_class():
    """Correctness fallback MPLinearKernel: dequantize W4A16 to the activation dtype and do a
    plain matmul. Handles ANY group_size -- notably group_size 32, which conch computes
    incorrectly on ROCm (its Triton kernel applies one scale per block_k=64 tile) and which
    exllama (fp16-only) also can't take in bf16. Registered LAST in the ROCm priority so the
    faster kernels win whenever they legitimately apply; this only catches what they reject."""
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearKernel,
    )
    from vllm.scalar_type import scalar_types

    class WinRocmW4A16DequantKernel(MPLinearKernel):
        @classmethod
        def get_min_capability(cls) -> int:
            return 0

        @classmethod
        def can_implement(cls, c):
            if c.weight_type not in (scalar_types.uint4, scalar_types.uint4b8):
                return False, "WinRocmW4A16Dequant supports only uint4 / uint4b8"
            if c.has_g_idx:
                return False, "WinRocmW4A16Dequant does not support act-order g_idx"
            return True, None

        def process_weights_after_loading(self, layer) -> None:
            # Reuse conch's normalization (layout permute + zero-point unpack); it does not
            # depend on group_size, only can_implement does.
            from vllm.model_executor.kernels.linear.mixed_precision.conch import (
                ConchLinearKernel,
            )
            self._conch = ConchLinearKernel(
                self.config, self.w_q_name, self.w_s_name, self.w_zp_name, self.w_gidx_name
            )
            self._conch.process_weights_after_loading(layer)

        def apply_weights(self, layer, x, bias=None):
            # conch-normalized layout: w_q [K//8, N] int32 packed-along-K (straight order),
            # w_s [K//G, N], w_zp [K//G, N] uint8 unpacked (or None for symmetric uint4b8).
            w_q, w_s, w_zp, _ = self._get_weight_params(layer)
            K = x.shape[-1]
            N = w_s.shape[1]
            G = self.config.group_size
            if G is None or G < 0:
                G = K
            wb = self.config.weight_type.bias or 0  # uint4b8 (symmetric) -> 8
            x2d = x.reshape(-1, K)
            M = x2d.shape[0]

            if M == 1:
                # Decode (latency path): fused streaming dequant-GEMV, no weight materialization.
                xc = x2d.contiguous()
                if w_zp is None:
                    out = gemv_k_sym(xc, w_q, w_s, G, wb)
                else:
                    out = gemv_k(xc, w_q, w_s, w_zp, G)
            else:
                # Prefill (one-shot): dequant to the activation dtype (bf16/fp16, NOT fp32) and
                # matmul. Transient [K, N] weight is freed immediately after the GEMM.
                dev = w_q.device
                shifts = (torch.arange(8, device=dev, dtype=torch.int32) * 4)
                codes = ((w_q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N)
                gidx = torch.arange(K, device=dev) // G
                s = w_s.index_select(0, gidx)
                if w_zp is not None:
                    z = w_zp.index_select(0, gidx).to(x.dtype)
                    w = (codes.to(x.dtype) - z) * s.to(x.dtype)
                else:
                    w = (codes.to(x.dtype) - wb) * s.to(x.dtype)
                out = x2d @ w
            if bias is not None:
                out = out + bias
            return out.reshape(x.shape[:-1] + (N,))

    return WinRocmW4A16DequantKernel


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _load_hip_gemv()  # eager import of the HIP .pyd (registers torch.ops.vllm_win_hip), before any compile
    try:
        from . import moe_decode
        moe_decode.patch_moe()  # opt-in M=1 MoE-decode GEMV (VLLM_WIN_MOE_DECODE=1)
    except Exception as e:  # noqa: BLE001
        logger.warning("moe_decode wire warning: %r", e)
    try:
        from . import bf16_gemv
        bf16_gemv.patch_unquantized_linear()  # opt-in M=1 dense bf16 GEMV (VLLM_WIN_BF16_GEMV=1)
    except Exception as e:  # noqa: BLE001
        logger.warning("bf16_gemv wire warning: %r", e)
    try:
        # Native _rocm_C skinny GEMMs (LLMM1 + wvSplitK), built 1:1 from csrc/rocm/skinny_gemms.cu.
        # Loading registers torch.ops._rocm_C.*; with VLLM_ROCM_USE_SKINNY_GEMM=1, vLLM's
        # rocm_unquantized_gemm routes the M=1 dense (MLP/attn-proj) GEMVs through wvSplitK.
        if os.environ.get("VLLM_WIN_ROCM_C", "0") == "1":
            import glob
            import torch
            d = os.environ.get("VLLM_WIN_ROCM_C_DIR") or _default_dir("vw_rocmc_build")
            for p in sorted(glob.glob(os.path.join(d, "*.pyd"))):
                torch.ops.load_library(p)
                if hasattr(torch.ops, "_rocm_C") and hasattr(torch.ops._rocm_C, "wvSplitK"):
                    logger.info("loaded native _rocm_C skinny GEMM from %s", p)
                    break
    except Exception as e:  # noqa: BLE001
        logger.warning("_rocm_C load warning: %r", e)
    try:
        from vllm.model_executor.kernels.linear import _POSSIBLE_KERNELS
        from vllm.platforms import PlatformEnum

        kcls = _register_kernel_class()
        lst = _POSSIBLE_KERNELS.get(PlatformEnum.ROCM, [])
        if kcls not in lst:
            lst.insert(0, kcls)
        # Correctness fallback for group sizes conch can't do (e.g. 32): append LAST.
        dq = _register_dequant_kernel_class()
        if dq not in lst:
            lst.append(dq)
        _REGISTERED = True
        logger.info("registered WinRocmAwqGemvKernel (front) + W4A16 dequant fallback (last)")
    except Exception as e:  # noqa: BLE001
        logger.warning("awq_gemv register warning: %r", e)
