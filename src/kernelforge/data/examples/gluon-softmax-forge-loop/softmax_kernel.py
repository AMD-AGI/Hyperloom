"""Gluon fused softmax kernel — the target forge-loop optimizes.

This is the file forge-loop edits for this (single-file) example. To play nicely
with the loop it must stay:

  * numerically correct (the driver gates it against ``torch.softmax`` via SNR),
  * with a STABLE public entry point ``softmax(x)`` — the driver imports this
    exact name and signature; do NOT rename or change its arguments,
  * in the Triton/Gluon toolchain.

Gluon is Triton's low-level dialect: same ``@…jit``, same launch surface, same
JIT cache, same ``Triton -> TritonGPU -> TritonAMDGPU -> AMDGCN`` lowering. What
differs is that the tile LAYOUT is an explicit object you write down, and the
compiler no longer chooses it for you. Everything in ``_LAYOUT`` below is
therefore a real, measurable degree of freedom — which is the whole reason this
example exists.

The baseline is deliberately a **v0**: correct, with the layout stated, and
nothing else. ``size_per_thread=[1]`` means one element per thread per load with
no vectorization at all, and the loads are ordinary masked ``gl.load`` rather
than AMD buffer ops. Both are obvious headroom for the loop to discover and
measure — see ``program.md`` for the ladder.
"""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# Wavefront is 64 lanes on CDNA. Every blocked-layout literal copied from an
# upstream (NVIDIA) Gluon tutorial says 32 and is wrong here.
_WAVEFRONT = 64

# Baseline launch config — intentionally conservative; the loop may tune it.
_NUM_WARPS = 4

# Elements each thread owns per load. 1 == scalar loads, no vectorization. This
# is the single clearest axis in the file: on a 1D tile the entire space of
# blocked layouts is this one number.
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
    # The layout is the Gluon-specific part: it states how BLOCK_SIZE elements
    # are distributed over (registers, lanes, warps). The three vectors multiply
    # out to the block shape, so SIZE_PER_THREAD * 64 * NUM_WARPS must cover
    # BLOCK_SIZE.
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

    # Seed the layout on the index tensor; it propagates forward from here
    # through type inference, so nothing below needs annotating.
    offsets = gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = offsets < n_cols

    # Load in fp32 for a numerically stable reduction; masked lanes are -inf so
    # they contribute exp(-inf) = 0 to the sum.
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

    # The layout must tile the whole block: size_per_thread * 64 * num_warps
    # has to reach BLOCK_SIZE. Grow the warp count when the row is too wide for
    # the baseline config, rather than silently computing a partial row.
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
