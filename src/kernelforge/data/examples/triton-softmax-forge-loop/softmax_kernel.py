"""Triton fused softmax kernel — the target forge-loop optimizes.

This is the file forge-loop edits for this (single-file) example. To play nicely
with the loop it must stay:

  * numerically correct (the driver gates it against ``torch.softmax`` via SNR),
  * with a STABLE public entry point ``softmax(x)`` — the driver imports this
    exact name and signature; do NOT rename or change its arguments,
  * in Triton (do not rewrite it in another framework).

The initial launch configuration below is deliberately conservative
(``num_warps=1``), which is correct but leaves obvious optimization headroom
(warp count, pipelining, block size, memory-access pattern) for the loop to
discover. That is the point of the example: watch the loop turn a slow-but-
correct baseline into a faster one, keeping only the changes that measurably win.
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

    # Baseline launch config — intentionally conservative; the loop may tune it.
    num_warps = 1

    _softmax_kernel[(n_rows,)](
        out,
        x,
        out.stride(0),
        x.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out
