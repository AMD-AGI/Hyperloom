# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Standalone launch glue to copy to kernel.py during the correctness-only PORT."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from kernelforge.assembly.compiler import assemble
from kernelforge.assembly.hip import HipKernel


class Score:
    """Fixed T=64, NVB=8, H=7168 specialization; build before capture or timing."""

    def __init__(self):
        torch.cuda.init()
        target = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        if target.split(":")[0] != "gfx950":
            raise ValueError("AttnRes score requires gfx950")
        with tempfile.TemporaryDirectory(prefix="forge-attnres-") as build:
            binary = assemble(
                Path(__file__).with_name("kernel.s"),
                Path(build) / "score.hsaco",
                gpu_target="gfx950",
                toolchain_dir=Path("/opt/rocm/llvm/bin"),
            )
            self.kernel = HipKernel(
                binary,
                "kimik3_attnres_score",
                ["ptr", "ptr", "ptr", "ptr", "i32", "f32", "i32", "i32", "i32", "i32"],
            )
        self.device = torch.device("cuda", torch.cuda.current_device())

    def __call__(self, prefix, bank, weight, output):
        for tensor, shape, dtype in (
            (prefix, (64, 7168), torch.bfloat16),
            (bank, (64, 8, 7168), torch.bfloat16),
            (weight, (7168,), torch.float32),
            (output, (64, 16), torch.float32),
        ):
            if tensor.device != self.device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError("unsupported AttnRes tensor shape, dtype, or device")
            if not tensor.is_contiguous():
                raise ValueError("AttnRes example supports contiguous tensors only")
        self.kernel.launch(
            [
                prefix.data_ptr(),
                bank.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                8,
                1e-6,
                prefix.stride(0),
                bank.stride(0),
                bank.stride(1),
                output.stride(0),
            ],
            grid=(64, 9, 1),
            block=(1024, 1, 1),
            stream=torch.cuda.current_stream(self.device).cuda_stream,
        )

    def close(self):
        """Call only after synchronization and retirement of all captured graphs."""
        self.kernel.close()
