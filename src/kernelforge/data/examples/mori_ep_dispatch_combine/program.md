# Task: tune MoRI-EP dispatch/combine launch config for EP8

## Objective

Minimize the combined dispatch+combine wall time (`case_ms`, reported by
`driver.py --bench-mode`) for a fixed EP8 MoE all-to-all workload:
8 GPUs, 4096 tokens/rank, hidden_dim=7168, top-8 routing, fp8 (e4m3fnuz)
dispatch + bf16 combine, MoRI-EP `IntraNode` kernel (single node, xGMI only).

You edit **only** `mori_ep_config.py`'s `get_ep_launch_config()` return dict.
The workload itself (world size, token count, hidden dim, top-k, dtypes) is
fixed in the protected `driver.py` — do not try to change it, and do not
edit `driver.py`.

## Prior work (read the KB card, don't re-derive this blind)

`mori_ep_config.py` starts from mori's own **out-of-the-box class defaults**
(`80/8/80/8`, `IntraNode`, external buffer) — untuned, with real headroom.
A prior investigation already searched this *exact* workload (same shape:
4096 tokens/rank, hidden_dim=7168, top-8) over two forge-loop campaigns. Full
history, tables, and sources are in the `framework/mori/` knowledge section
of your system prompt (`run_example.sh` enables it by default via
`KERNELFORGE_INCLUDE_MORI_KB=1`) — under that section's listed absolute
`base:` path, read `operators/ep_dispatch_combine/tuning.md`.
**A bare relative path will NOT resolve from your working directory —
use the absolute base path the knowledge section gives you.** Read it
before guessing, but in short:

> **Re-measurement notice (updated)**: the numbers below were originally
> produced by an earlier version of `driver.py` that had real timing/
> lifecycle bugs (forced mid-timing sync, wrong per-round aggregation, and
> — critically for the zero-copy line specifically — a manual buffer copy
> inside the timed region that defeats the entire point of zero-copy).
> Those bugs are now fixed, and the two items below marked
> **[re-measured]** were directly re-confirmed against the fixed driver
> (interleaved A/B samples on the same MI300X box). The block/warp search
> result itself (`152/16/304/16`) was NOT re-run from scratch through a
> fresh forge-loop search — only re-measured at that one known config — so
> treat "converged twice independently" as a pre-fix claim about the
> *search process*, while the *ms numbers* for that exact config are
> current.

- A search over `dispatch_block_num` / `dispatch_warp_per_block` /
  `combine_block_num` / `combine_warp_per_block` / `kernel_type`, starting
  from the same class-default baseline this file ships, converged twice
  independently (pre-fix runs) on `dispatch_block_num=152,
  dispatch_warp_per_block=16, combine_block_num=304,
  combine_warp_per_block=16, kernel_type=IntraNode`. **[re-measured]** with
  the fixed driver: this config now measures ~1.578 ms vs. the class-default
  baseline's ~1.901 ms — a real **1.20x** speedup (not the pre-fix-driver
  1.34x figure; both the "before" and "after" side of that ratio were
  inflated by the same timing bugs, so the ratio itself shifted along with
  the absolute numbers). Still a solid, reproducible win, just a smaller one
  than originally reported.
  `kernel_type="IntraNodeLL"` was also tried and was consistently 2-4%
  *slower* at this shape in that (pre-fix) search — but see the KB card's
  "Round 1" section for a caveat: a separate local investigation (not
  tracked in this repo) measured the opposite result on the same hardware
  before retracting it as non-reproducible, so treat the kernel-type
  comparison as a reasonable prior, not settled fact.
- The `combine_zero_copy` knob was tested separately (mori's own official
  tuner data shows a +29% bandwidth win from it on a different chip,
  MI308X, at this same shape). The prior campaign measured **no win on
  MI300X** at this shape, but that measurement was contaminated by a
  `copy_()` inside the timed region. **[re-measured]**, corrected: at the
  *same* class-default block/warp config (`80/8/80/8`), 5 interleaved A/B
  samples gave external-buffer ~1.901 ms vs. zero-copy ~1.775 ms — a
  consistent, reproducible **~6.6% win for zero-copy**, reversing the prior
  conclusion. `combine_zero_copy` still defaults to `False` in this file
  (forge-loop should verify this itself rather than take it as settled, and
  should search zero-copy's own block/warp optimum rather than assume
  `152/16/304/16` transfers unchanged), but "no win" is no longer an
  accurate prior — expect zero-copy to be a live contender.

**These numbers are scoped to this exact shape.** If you're running this
task with `MORI_TOKENS_PER_RANK` / `MORI_HIDDEN_DIM` / `MORI_TOPK`
overridden, the config above does not necessarily transfer — round 1's own
data shows the 256-token decode shape wants a very different
`dispatch_block_num` (40, not 152) than this shape.

**Nothing here hands you that answer as a starting point** — you're free to
consult the card and use its config as a hypothesis to verify, but you start
from the untuned baseline above and have to re-measure to claim it. Both KB
findings above are strong, independently confirmed priors for *this* shape.
See "what's still open" in the KB card for concrete, not-yet-tested
directions (repeat-validation rigor, the decode/prefill transition point for
`dispatch_block_num`, reconciling the kernel-type contradiction above, etc.)
— or bring your own hypothesis if you have one, as long as it's backed by an
honest measurement, not a guess.

## Hard rules

1. **Only edit `mori_ep_config.py`.** `driver.py` is protected (the loop
   blocks edits to it anyway).
2. **Keep the `get_ep_launch_config() -> dict` signature** — no args, returns
   a dict with **all six** keys already there (`dispatch_block_num`,
   `dispatch_warp_per_block`, `combine_block_num`, `combine_warp_per_block`,
   `kernel_type`, `combine_zero_copy`). All six are mandatory — the driver
   validates this and raises a clear error naming any missing key rather
   than silently defaulting one, so don't drop a key you don't intend to
   change; keep it at its current value instead.
3. **Correctness is a real distributed round trip, not a proxy.** The
   correctness gate spawns all 8 GPUs and does an actual
   `dispatch -> identity expert -> combine` round trip through MoRI-EP with
   your launch config, including your `combine_zero_copy` choice — the gate
   exercises the exact same code path the benchmark times. A config that
   produces wrong results or hangs/asserts fails validation and gets
   reverted.
4. **Stay single-node.** `kernel_type` may be `"IntraNode"` or `"IntraNodeLL"`
   only. This box has no RDMA fabric configured for MoRI.
5. **Don't reduce `max_num_inp_token_per_rank` or the token/hidden/top-k
   workload** — that's fixed in `driver.py`, not a knob you own.
6. **Measurement rigor**: single-shot benchmark numbers on this box can be
   noisy. Before keeping a change, prefer re-running the benchmark at least
   once more to confirm the delta isn't noise, per the same logic as
   `local_knowledge/common_methodology/profiling/benchmarking_methodology.md`
   (treat a <1% delta with suspicion).

## Off-limits

- Do not add a new file or change `driver.py`.
- Do not try to install/upgrade the `mori` package, rebuild it from source,
  or wire up `dispatch_combine_v2`.
- Do not set `MORI_EP_LAUNCH_CONFIG_MODE` or other env vars to route around
  the tunable surface in `mori_ep_config.py`.
- Do not disable or weaken the correctness round-trip check.
