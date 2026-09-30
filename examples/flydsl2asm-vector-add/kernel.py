# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One FlyDSL vector-add specialization with its original host launcher."""

from __future__ import annotations

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

N = 4103


@flyc.kernel
def _vector_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr):
    index = fx.block_idx.x * 256 + fx.thread_idx.x
    if index < N:
        a = fx.logical_divide(A, fx.make_layout(1, 1))
        b = fx.logical_divide(B, fx.make_layout(1, 1))
        c = fx.logical_divide(C, fx.make_layout(1, 1))
        copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        ra = fx.make_rmem_tensor(1, fx.Float32)
        rb = fx.make_rmem_tensor(1, fx.Float32)
        rc = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy_atom_call(copy, fx.slice(a, (None, index)), ra)
        fx.copy_atom_call(copy, fx.slice(b, (None, index)), rb)
        rc.store(ra.load() + rb.load())
        fx.copy_atom_call(copy, rc, fx.slice(c, (None, index)))


@flyc.jit
def _vector_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr, stream: fx.Stream):
    _vector_add_kernel(A, B, C, N).launch(grid=((N + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


class VectorAdd:
    def __init__(self):
        a, b = torch.randn(N, device="cuda"), torch.randn(N, device="cuda")
        out = torch.empty_like(a)
        self.function = flyc.compile(_vector_add, a, b, out, N, torch.cuda.current_stream())

    def __call__(self, a, b, out):
        self.function(a, b, out, N, torch.cuda.current_stream())

    def close(self):
        self.function = None
