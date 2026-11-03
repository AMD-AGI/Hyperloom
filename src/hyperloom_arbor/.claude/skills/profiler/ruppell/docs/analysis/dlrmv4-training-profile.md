# What ruppell shows on DLRMv4 training (MI350X)

The findings of the runner's DLRMv4 Plans 02 and 03.

**What was measured:**
- one-node MLPerf DLRMv4 training on Cluster A (MI350X): training `d88f609`, 8
  ranks, global batch 8,192, 200 steps;
- traced with full detail and the CPU op tree on steps 41–80, exported at
  steps 62–66;
- untraced step times: three alternating runs per arm, steps 100–200.

The traces' validity (attribution, witness, overhead) is in
[`docs/milestones/dlrmv4.md`](../milestones/dlrmv4.md). The build is
`libruppell_tool.so` sha256 `b8386fcb…5e112507`.

**The arms**, from the training branch's FP8 knobs:

| Arm | What it is |
| --- | --- |
| R | bf16 throughout |
| B0 | FP8 forward, bf16 backward |
| B1 | B0 + the output-projection dgrad in FP8 |
| B2 | B1 + the attention backward's qk GEMM on k128 FP8 MFMA |
| B3 | B2 + all five attention-backward GEMMs in FP8, with block-scaled dO (the branch default) |

## The step

Job 59174, `c12-18`, ruppell `088f90e`. Median rank, 37 detailed steps
(41–80 without the window-boundary steps 46, 56 and 68), ms per step:

| | R (bf16) | B0 (FP8 forward) |
| --- | --- | --- |
| untraced step time, steps 100–200 (3 runs) | 284.52 (spread 0.29) | 244.61 (spread 1.23): **−14.0%** |
| GPU step period, begin marker to end marker | 283.2 | 241.9 |
| GPU busy share of the period | 98.4% | 97.7% |
| gap to the next step | 0.0 | 0.0 |
| **Kernel occupancy by phase** (all queues; collectives include waits) | | |
| backward (`bwd:` scopes) | 186.6 | 166.6 |
| of which DDP gradient sync (`bwd:op:…AccumulateGrad`) | 13.8 | 10.1 |
| forward (`model.*`) | 90.4 | 69.8 |
| input distribution (the sparse all-to-all) | 39.5 | 31.1 |
| metrics | 3.0 | 7.5 |
| optimizer | 0.5 | 0.5 |
| **Side streams** (share overlapped with compute) | 63.1 (74%) | 50.6 (77%) |
| **Host lead** (GPU begin marker − host `step_begin`) | 175 | 147 |
| **Host blocked in synchronizing HIP calls** | 13.6 | 11.0 |

- **The GPU never idles between steps.** The host runs 0.6 of a step ahead,
  and the GPU is busy for 98% of each step.
- **The STU layers' backward is the largest block,** about half the step:
  146 ms in R and 127 ms in B0, on rank 0.
- **Its largest kernel is FlyDSL's `bwd_dkdv_kernel_0`,** 52–56 ms.
- **The input all-to-all** (`mscclKernel_Sum_int64_t`) takes 25–28 ms, on a
  side stream.

**Kernel time is not step time.** A collective's kernel runs while it waits
for the slowest rank, and on a side stream it overlaps compute. At step 64,
rank 0 spent 60 ms more in collective kernels than at its neighbours, yet
every rank's step period was 233 ms, the shortest in the window. Read the
period and the busy share for wall time, and phase times for where the GPU's
occupancy goes.

**A late rank, not a heavy one.** Step 66 lasted 262 ms on every rank,
against 236–240 for its neighbours.
- Rank 7's GPU was idle 31 ms inside it.
- The other seven ranks spent that time in collectives.
- STU compute was 161–189 ms on every rank.

So rank 7 waited on something other than a collective, plausibly its host or
a stream dependency. The trace bounds where to look; it does not name the
cause.

## Why B0's backward is faster than R's

B0's backward is 18 ms per step faster than R's, though its backward math is
bf16 in both arms. Paired by kernel (rank 0, steps 62–66,
`tests/d4/d4_pairdiff.py`):

| In the STU backward | R | B0 |
| --- | --- | --- |
| forward projection recomputed: bf16 `aten::addmm`, about 2.3M×512 × 512×2048 (layers 0, 1) or ×1024 (layer 2) | 18.1 ms, all 3 layers | — |
| `aten::add_`, `aten::silu_backward` | 3.7 + 3.2 ms | — |
| forward projection recomputed: FP8 `aten::_scaled_mm`, layer 2 only | — | 2.4 ms |
| `bwd_dkdv_kernel_0` | 52.9 ms | 56.1 ms |

R recomputes every layer's forward projection inside the backward, in bf16.
B0 recomputes only one, in FP8. ruppell finds this because it attributes the
recompute to the layer whose backward runs it
(`bwd:model._hstu_transducer._stu_module._stu_layers.2`). In the profiler's
trace it is an `aten::addmm` under an autograd node, with no layer named.

## The FP8 backward, one knob at a time

Job 59310, `c14-18`, ruppell `a495b57`. A first set, 59201 on the same node,
agrees.

| Arm | Untraced step time (median of 3) | vs the previous arm | Floor (larger spread of the pair) |
| --- | --- | --- | --- |
| B0 | 234.87 ms | — | — |
| B1 | 233.00 ms | −1.87 ms (−0.80%) | 1.11 |
| B2 | 229.11 ms | −3.89 ms (−1.67%) | 1.11 |
| B3 | 223.59 ms | −5.52 ms (−2.41%) | 0.75 |

**What each knob changes, per rank per step** (steps 62–66):

| Pair | The targeted kernels | Elsewhere |
| --- | --- | --- |
| B0 → B1 | Output-projection dgrad: bf16 `aten::mm` (2 launches, 7.83 ms) → FP8 `_scaled_mm` `F8B8BS` (5.77 ms) plus `_quantize_amax_kernel` (4 launches, 1.23 ms). **Net −0.83 ms.** | Unchanged except collective waits |
| B1 → B2 | `bwd_dkdv_kernel_0` 56.14 → 52.04 ms at the same 2 launches: **−7.3%** | Unchanged |
| B2 → B3 | `bwd_dkdv_kernel_0` 52.04 → 46.04 ms, plus the dO quantizer `_blk_kernel` (2 launches, 1.58 ms): **net −8.5%** | Unchanged |

**Per-step variation hides the attention gains in a single profile.**
- `bwd_dkdv` varies 49–58 ms from step to step with the batch, more than
  B2's or B3's effect.
- Every arm runs the same batches, though, and paired by step all five are
  lower: by 3.0–5.6 ms in B2 and 3.2–5.3 ms in B3.

**B2's knob cannot be observed directly.** Its FP8 path runs inside the same
kernel symbol, and the training code logs nothing for it. Only its effect
shows.

**The backward, B0 → B3** (ms per rank per step). GEMMs are split by operand
shape: a wgrad contracts over the token dimension.

| | B0 | B1 | B2 | B3 |
| --- | --- | --- | --- | --- |
| attention backward (`bwd_dkdv`) | 56.4 | 56.1 | 52.0 | 46.0 |
| FP8 dO quantizer | — | — | — | 1.6 |
| norms, reductions, elementwise | 44.4 | 45.2 | 45.0 | 44.9 |
| bf16 dgrad and recompute GEMMs | 24.4 | 16.3 | 17.3 | 16.3 |
| FP8 GEMMs | 2.3 | 8.1 | 8.4 | 8.1 |
| FP8 quantize (dgrad) | — | 1.2 | 1.2 | 1.2 |
| bf16 wgrads | 19.6 | 19.2 | 20.1 | 19.0 |
| embeddings and their MLPs | 8.2 | 8.2 | 8.3 | 8.2 |

- **FP8's share of backward GEMM time** is 4.9% in B0 (the recompute) and
  18.2–18.5% in B1–B3. The attention backward's GEMMs run inside the FlyDSL
  kernel and are not counted.
- **After B3, the largest bf16 block is not a GEMM.** The norms, reductions
  and elementwise kernels hold 44 ms:

  | Kernel | ms |
  | --- | --- |
  | LayerNorm-dropout backward | 9.8 |
  | weighted LayerNorm backward | 7.1 |
  | 23 `aten::sum` reductions | 5.7 |
  | LayerNorm forward recompute | 3.0 |
  | FP32 cast-expand reductions | 2.5 |
  | attention delta | 1.9 |

## The metrics step

A 30-step run (58091; `METRIC_LOG_FREQ=10`) shows how logging metrics costs
time:
- **The first logged step lasted 739 ms on rank 0,** against about 285 ms for
  the others.
  - Rank 0's AUC sort blocked 326 ms in a single `hipLaunchKernel`, which
    looks like the first launch of rocPRIM's sort kernels loading their code.
  - Meanwhile ranks 1–7 waited 410 ms in `c10d::barrier`.
- **The second logged step lasted 401 ms:** about 115 ms per logged step
  after the first, about 2% of step time at `METRIC_LOG_FREQ=20`.

In the window that held the first logged step, the barrier kernels made
`loop.metrics.compute_and_log` 18% of GPU time, all of it waiting.

## Against the repository's `torch.profiler` trace

Arm B0, same node, another run (59174). The comparison script is
`tests/d4/d4_torchprof_compare.py`.
- **Its steps do not bound GPU work.** A `ProfilerStep` is a CPU span of the
  training loop, and with the host 150 ms ahead those spans don't line up
  with the GPU's steps.
  - Its exported steps hold between 117 kernels (1.5–3.5 ms) and 448 ms of
    kernel time.
  - ruppell's steps, bounded by markers on the GPU queue, hold 255–363 ms.
- **Over each five-step window,** the profiler's file holds 66–83% of the
  kernel time ruppell records for the same steps. The runs differ, and
  collective waits vary between runs; this comparison does not separate the
  two.
- **It names the backward by aten op only.** 54–83% of a full step's kernel
  time is launched inside `autograd::engine::evaluate_function`, and the
  trace holds no `nn.Module` records. ruppell names the module for 99.99% of
  backward GPU time.

## Why the FP8 wgrads stay bf16

The B3 trace gives the five STU weight gradients' shapes and in-step times:
- the uvqk projection: dW 512×2048, 3.35 ms;
- the output projection: dW 1536×512, 2.86 ms;
- layer 2's projection: dW 512×1024, 1.79 ms.

All contract over about 2.3M token rows, and all run as pinned hipBLASLt bf16
kernels. A kernel study at those shapes (the runner's DLRMv4 Plan 04;
`tests/d4/d4_wgrad_probe.py`, `d4_wgrad_study.sh`) found:

- **The pinned bf16 GEMM reads each operand about once,** at 1.1–1.5 PFLOPS.
- **An FP8 GEMM with ideal operand layout is only 1.3–1.4× faster.** At most
  that saves about 4.6 ms per step.
- **That layout needs transposes** that cost more than the saving.
- **The transpose-free Triton split-K kernel is 1.44–1.55× slower than
  bf16.** It re-reads its operands 2.4–3.6× (from `rocprofv3` counters) and
  runs at half of hipBLASLt's FP8 throughput.
