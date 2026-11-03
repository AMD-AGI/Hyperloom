"""Triton fused softmax kernel — the SOURCE this task ports to FlyDSL.

`forge-rewrite-by-flydsl` reads this file as the reference to translate FROM and
uses its public host entry as the live correctness ORACLE + performance baseline.
It is protected during the rewrite (never edited): the pipeline only writes the
new FlyDSL `kernel.py`.

Contract the driver relies on (do NOT rename / change the signature):
  * Public entry `softmax(x)` — row-wise softmax over the last dim of a 2D tensor,
    returning a tensor of the same shape/dtype as `x`.
  * `@triton.jit` kernel `_softmax_kernel` — named in `config.yaml`
    `target_kernel_functions` (a hint shown to the port agent + PMC label).

softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))  (per row)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    out_row_stride,
    in_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # One program instance handles one row of the input.
    row = tl.program_id(0)
    in_row_ptr = in_ptr + row * in_row_stride
    out_row_ptr = out_ptr + row * out_row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load in fp32 for a numerically stable reduction; masked lanes are -inf so
    # they contribute exp(-inf) = 0 to the sum.
    x = tl.load(in_row_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out_row_ptr + offsets, numerator / denominator, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D tensor. Public entry point."""
    assert x.dim() == 2, "expected a 2D (rows, cols) tensor"
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    # BLOCK_SIZE must cover a full row so the reduction sees every element.
    block_size = triton.next_power_of_2(n_cols)

    _softmax_kernel[(n_rows,)](
        out,
        x,
        out.stride(0),
        x.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return out
