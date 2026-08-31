"""Tunable MoRI-EP dispatch/combine launch configuration.

forge-loop optimizes THIS file. The workload is fixed by ``driver.py``
(EP8, 4096 tokens/rank, hidden_dim=7168, top-8 routing, fp8 dispatch + bf16
combine on the benchmark path) -- nothing here changes the math, only how the
dispatch and combine GPU kernels are launched.

The values below are mori's own out-of-the-box **class defaults**
(``EpDispatchCombineConfig``'s constructor defaults for ``block_num``/
``warp_num_per_block``), i.e. what you get by using mori without tuning
anything -- deliberately **not** the best config found so far, so this
example (like every other task under ``examples/``, see ``examples/README.md``
§1) has real, measurable headroom for forge-loop to find. A prior
investigation already searched this exact workload and found a
meaningfully faster config -- see ``program.md`` for whether and how that
prior work is surfaced to you *this run* (it is an ablation-gated knob, off
in some runs on purpose). Do NOT go looking for a KB card by path yourself
if program.md and your own knowledge-section listing do not mention one --
that is a deliberate no-KB condition, not an oversight, and hand-navigating
to it defeats the point of that run. If a ``framework/mori/`` entry IS
listed in your knowledge section this run, that is where it lives -- but
even then, do not hand-copy its answer into this file, that defeats the
point of running the loop.

Shape-specific caveat: any known-good numbers cited in ``program.md`` /
``tuning.md`` were measured at the **default** shape (4096 tokens/rank,
hidden_dim=7168, top-8 -- see driver.py's ``_HIDDEN_DIM`` /
``_NUM_EXPERTS_PER_TOKEN`` / ``_BENCH_TOKENS_PER_RANK``). If you run this
task with ``MORI_TOKENS_PER_RANK`` / ``MORI_HIDDEN_DIM`` / ``MORI_TOPK``
overridden to a different shape, those specific block/warp numbers do not
necessarily transfer (round 1's own results show the decode shape, 256
tokens/rank, wants a very different ``dispatch_block_num`` than the
4096/8192-token shapes) -- the class defaults below remain a valid untuned
starting point for any shape, but treat any *tuned* number as shape-scoped
unless you re-measure.

Public entry point (stable -- do not rename or change the signature):
    get_ep_launch_config() -> dict
"""

from __future__ import annotations


def get_ep_launch_config() -> dict:
    """Return MoRI-EP dispatch/combine launch-config overrides.

    Keys (all required; the driver passes each straight through as
    ``block_num=`` / ``warp_per_block=`` on the dispatch/combine calls,
    except ``kernel_type`` which selects the compiled kernel entry point at
    construction time, and ``combine_zero_copy`` which selects the combine
                                  buffer mode -- see program.md for whatever prior-experience context is
                                  authorized for this run):
      - dispatch_block_num:      int, GPU blocks used by dispatch's kernel.
      - dispatch_warp_per_block: int, warps per block for dispatch.
      - combine_block_num:       int, GPU blocks used by combine's kernel.
      - combine_warp_per_block:  int, warps per block for combine.
      - kernel_type:             str, one of "IntraNode" | "IntraNodeLL".
                                  Any other value is unsupported on this
                                  single-node box and fails the correctness
                                  gate (see driver.py's ``_make_config``).
      - combine_zero_copy:       bool, False = externally-managed combine
                                  buffer (mori's class-level default), True =
                                  mori's registered zero-copy buffer (a prior
                                  investigation's finding on this at the
                                  default shape on MI300X, if any, may be
                                  surfaced via program.md -- see the note
                                  there, do not go hunting a KB path yourself).
    """
    return {
        "dispatch_block_num": 80,
        "dispatch_warp_per_block": 8,
        "combine_block_num": 80,
        "combine_warp_per_block": 8,
        "kernel_type": "IntraNode",
        "combine_zero_copy": False,
    }
