"""Gluon fused softmax kernel — the target forge-loop optimizes."""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# Wavefront is 64 lanes on CDNA.
_WAVEFRONT = 64

# Baseline launch config — intentionally conservative; the loop may tune it.
_NUM_WARPS = 4

# Elements each thread owns per load. 1 == scalar loads, no vectorization.
_SIZE_PER_THREAD = 1


@gluon.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    out_row_stride,
    in_row_stride,
    n_cols,
    BLOCK_SIZE: gl.constexpr,
    SIZE_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # The layout is the Gluon-specific part: it states how BLOCK_SIZE elements are distributed over (registers, lanes,
    # warps).
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[SIZE_PER_THREAD],
        threads_per_warp=[64],
        warps_per_cta=[NUM_WARPS],
        order=[0],
    )

    # One program instance handles one row of the input.
    row = gl.program_id(0)
    in_row_ptr = in_ptr + row * in_row_stride
    out_row_ptr = out_ptr + row * out_row_stride

    # Seed the layout on the index tensor; it propagates forward from here through type inference, so nothing below
    # needs annotating.
    offsets = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_cols

    # Load in fp32 for a numerically stable reduction; masked lanes are -inf so they contribute exp(-inf) = 0 to the
    # sum.
    x = gl.load(in_row_ptr + offsets, mask=mask, other=-float("inf")).to(gl.float32)
    x = x - gl.max(x, 0)
    numerator = gl.exp(x)
    denominator = gl.sum(numerator, 0)
    gl.store(out_row_ptr + offsets, numerator / denominator, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim of a 2D tensor. Public entry point."""
    assert x.dim() == 2, "expected a 2D (rows, cols) tensor"
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    # BLOCK_SIZE must cover a full row so the reduction sees every element.
    block_size = triton.next_power_of_2(n_cols)

    # The layout must tile the whole block: size_per_thread * 64 * num_warps has to reach BLOCK_SIZE.
    num_warps = _NUM_WARPS
    size_per_thread = _SIZE_PER_THREAD
    while size_per_thread * _WAVEFRONT * num_warps < block_size:
        num_warps *= 2
    # A block narrower than one full wave still needs a layout that covers it.
    block_size = max(block_size, size_per_thread * _WAVEFRONT * num_warps)

    _softmax_kernel[(n_rows,)](
        out,
        x,
        out.stride(0),
        x.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        SIZE_PER_THREAD=size_per_thread,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out
