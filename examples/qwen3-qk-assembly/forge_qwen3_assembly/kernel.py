# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Explicit ABI wrapper for the experimental Qwen3 Q/K assembly kernel."""

from __future__ import annotations

from pathlib import Path

import torch

from kernelforge.assembly import assemble
from kernelforge.assembly.hip import HipKernel


class QkNormRope:
    """Fuse BF16 Q/K RMSNorm and full-width NeoX RoPE for 128-element heads."""

    def __init__(self, build_dir: Path, source: Path | None = None) -> None:
        if not torch.version.hip or not torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.startswith("gfx950"):
            raise RuntimeError("This assembly example requires a gfx950 ROCm device")
        source = source or Path(__file__).with_name("qk_norm_rope.s")
        code_object = assemble(
            source,
            build_dir / "qk_norm_rope.co",
            gpu_target="gfx950",
            toolchain_dir=Path("/opt/rocm/llvm/bin"),
        )
        self.kernel = HipKernel(
            code_object,
            "forge_qk_norm_rope_h128",
            ["ptr", "ptr", "ptr", "ptr", "ptr", "ptr", "ptr", "f32", "i32", "i32", "i32"],
        )

    def __call__(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cache: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_heads: int,
        k_heads: int,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions must index valid rows of the supplied BF16 cos/sin cache."""
        tensors = (qkv, cache, q_weight, k_weight)
        if not qkv.is_cuda or any(
            t.dtype != torch.bfloat16 or t.device != qkv.device or not t.is_contiguous() for t in tensors
        ):
            raise ValueError("QKV, weights, and cache must be contiguous BF16 tensors on one device")
        if qkv.ndim != 2 or q_heads <= 0 or k_heads <= 0 or qkv.shape[1] != (q_heads + 2 * k_heads) * 128:
            raise ValueError("Expected packed [tokens, (q_heads + 2*k_heads)*128] QKV")
        if (
            positions.dtype != torch.int64
            or positions.device != qkv.device
            or not positions.is_contiguous()
            or positions.numel() != qkv.shape[0]
        ):
            raise ValueError("Expected one contiguous int64 position per token on the QKV device")
        if cache.ndim != 2 or cache.shape[1] != 128 or q_weight.numel() != 128 or k_weight.numel() != 128:
            raise ValueError("Expected 128-element norm weights and a [positions, 128] cos/sin cache")
        if any(t.numel() * t.element_size() >= 2**32 for t in (qkv, positions, cache)):
            raise ValueError("This example requires tensor spans below 4 GiB for its uint32 relative offsets")
        q = qkv.new_empty((qkv.shape[0], q_heads * 128))
        k = qkv.new_empty((qkv.shape[0], k_heads * 128))
        if qkv.shape[0]:
            self.kernel.launch(
                [
                    *(t.data_ptr() for t in (qkv, positions, cache, q_weight, k_weight, q, k)),
                    epsilon,
                    q_heads,
                    k_heads,
                    qkv.stride(0) * qkv.element_size(),
                ],
                grid=(qkv.shape[0], q_heads + k_heads, 1),
                block=(64, 1, 1),
                stream=torch.cuda.current_stream(qkv.device).cuda_stream,
            )
        return q, k
