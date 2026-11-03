# DLRMv4 training on MI350X (record): a second workload

Closed 2026-10-06. ruppell traced a second workload, with no change to the
runtime's design: MLPerf DLRMv4 training, an HSTU recommender on TorchRec. It
differs from M2's DeepSeek V4.1 Flash serving almost everywhere:

| | M2: vLLM serving | This record: DLRMv4 training |
| --- | --- | --- |
| Program | inference, a server answering requests | training: forward, autograd backward, optimizer |
| GPU work | mostly captured graphs, replayed | eager; no graphs, no `torch.compile` |
| Step | one `execute_model` | one TorchRec `TrainPipelineSparseDist.progress()`, overlapping three batches |
| Host | waits on the GPU each step | runs about 150 ms (0.6 step) ahead (`StepFence`, lag 1) |
| Backward | none | autograd, on the calling thread, where no module hook runs |
| Model | MoE transformer, TP4 | 3 STU layers with FlyDSL jagged attention, TorchRec embeddings, DDP over 8 ranks |
| Collectives | TP all-reduce | sparse all-to-all on side streams, DDP gradient all-reduce, MLPerf barriers |
| Processes | API server, engine core, workers | 8 ranks spawned, each with forked DataLoader workers |
| Kernels | prebuilt and pinned | FlyDSL and Triton compiled in the process; Triton autotunes by timing |
| ROCm stack | the image's | torch's wheel bundles its own HIP runtime and profiler SDK |

Every gate the runner set for it is met. One, "the same program", was met
under a rule the operator set (below).

**Environment:**
- Cluster A (AMD Instinct MI350X, gfx950), one node, 8 ranks;
- image `rocm/primus:v26.3` (ROCm 7.2.1), with the torch overlay
  `2.12.0+rocm7.2` (its own HIP runtime and profiler SDK 1.1.0);
- training `chriscai-amd/training_dlrmv4` `<colleague>/dlrmv4_opt` @ `d88f609`,
  Yambda-5b, global batch 8,192, 200 steps (`START_TS=0 NUM_TRAIN_TS=31`),
  `METRIC_LOG_FREQ=20`.

**The build:** `libruppell_tool.so` sha256 `b8386fcbf380…5e112507` in every
200-step job below. The marker library's digest differs per build because it
embeds its build path.

**The plans:** the runner's `plans/dlrmv4/` 01 (adapter and validation), 02
(baseline profile) and 03 (FP8 backward arms). What the traces show:
[`docs/analysis/dlrmv4-training-profile.md`](../analysis/dlrmv4-training-profile.md).

| Run | Job, node, commit | What |
| --- | --- | --- |
| Overhead | 58152, `c13-08`, `2b6318e` | arm B3, 200 steps: untraced and three detail tiers, alternating, three rounds; full detail on every step in the traced runs |
| Structure, witness, program | 58206 (B0, `c09-08`), 58207 (B3, `e17-08`), `bc2faf2` | per arm: two untraced, one traced, one traced inside `rocprofv3`, two untraced inside `rocprofv3` (markers only) |
| Program, autotune fixed | 58259, `c19-18`, `bc2faf2` | as above with Triton's autotune cache, so every run uses the first run's kernel choices |
| Profiles | 59174 (`c12-18`, `088f90e`), 59310 (`c14-18`, `a495b57`) | the runner's Plans 02 and 03: arms R, B0–B3, three untraced and one traced run each, detail on steps 41–80 |

Reproduce with `tests/d4/` (`d4_trace.sh`, `d4_reexport.sh`), one Slurm job
each, from a worktree per commit.

## Gates (runner Plan 01 §5, and Plans 02 and 03's validity)

| Gate | Result | Evidence |
| --- | --- | --- |
| **Steps:** every kernel after the first train step in exactly one step; step ids and fingerprints equal on all ranks; the schema valid | **Met** | 200 engine steps per rank (58152, 58206); 0 kernels in two steps; 0 straddling a step end; 0 fingerprint mismatches; `schemas/training.json` valid on all 8 ranks. Start-up, window boundaries, eval and cleanup are named steps. |
| **Attribution** ≥ 99% of GPU time | **Met** | 99.64–99.90% per rank (58152, full detail on every step); 99.72–99.94% (58206); 99.82–99.92% (59310, detail on steps 41–80). Inside train steps: 81 ms of 700.7 s (58152, all ranks), 0 ms in 59310. What remains is start-up copies and fills (`__amd_rocclr_copyBuffer`, `fillBufferAligned`), 3–4 ms over all ranks. |
| **Backward:** kernels with a `bwd:` scope ≥ 95% of backward GPU time | **Met: 99.99%** on every rank | 58152, 58206, 59174, 59310. A 30-step run (58091) linked 81,167 (rank 0) and 79,302 (rank 1) kernels to their forward module, and 2,058 per rank to `AccumulateGrad` (DDP's gradient sync). |
| **Witness:** `rocprofv3` in the same process; dispatch identity; per-kernel time within 3% | **Met, on the system runtime** | 58206 and 58207: 479,639–493,963 dispatches per rank, every one matched by id, same kernel, identical start and end (largest difference 0 ns). |
| **The same training:** losses traced and untraced | **Met** | Window and lifetime NE identical at the logged four decimals at all 10 logged steps, in both arms, across two untraced runs, a traced run, the witness and the markers-only run. AUC within 1e-4 and accuracy within 3e-4, which is what two untraced runs differ by. |
| **The same program:** identical kernel sequence per step | **Met under the operator's rule** (2026-10-06): "differs no more than two untraced runs differ", autotune cache on | See "The same program in training" |
| **Overhead** | **Measured** | Table below |
| **Viewers** | Perfetto: **met**. `chrome://tracing`: not checked. | 59310's B3 window (steps 62–66, 8 ranks) loads in Perfetto's trace processor with 368,742 slices and 48,586 flows, and no error or data-loss stats. |

**Overhead** (58152, arm B3, mean `step_ms` over steps 100–200, three
alternating rounds, one node):

| Tier | `step_ms` | vs untraced |
| --- | --- | --- |
| untraced | 237.30 (237.11, 236.98, 237.82) | — |
| markers only (step records and API records, no kernel detail) | 236.68 | −0.26%, inside the untraced spread (0.84 ms) |
| full detail | 239.38 | +0.88% |
| full detail with the CPU op tree | 239.96 | +1.12% |

**Volume:**
- Full detail with the CPU op tree writes 497 MB for 8 ranks × 30 steps
  (57972).
- A rank records about 480–494K dispatches over 200 fully detailed steps.
- The CPU op tree holds 379,332 aten ops on rank 0 in 30 steps, on 4
  threads, with 0 malformed records. 99.97% of GPU time has an aten op.

## What it took

The runtime's design held. What changed was training-specific, and each
change was found by a check failing on a real run:

| Found by | Change | Commit |
| --- | --- | --- |
| 0 backward kernels linked (57923): autograd runs on the calling thread, so backward kernels already carried `pipeline._backward` | A backward function's scope replaces any scope its kernels carry | `1cdc0a8` |
| 976,381 "threads" in the CPU op tree; ops records read out of alignment (57923) | `pthread_atfork` child handler: a forked DataLoader worker stops tracing and drops the inherited streams unflushed (`__fpurge`) | `1cdc0a8` |
| 2.9% of backward GPU time still on `pipeline._backward` (57972): DDP's all-reduce in `AccumulateGrad`'s post-hook, and input-buffer adds | A kernel anywhere in autograd's `evaluate_function` wrapper belongs to its node; a node without a forward op gives `bwd:op:<node>` | `8699e06` |
| Scope slices of −33,811 s in exports (57972) | Exporters draw aten and scope records only, never the autograd sequence-number records | `03e5917` |
| Run-level attribution 97.8–99.4% while train steps were 99.98% (57972, 58091, 58992, 59201): start-up and barrier waits with no scope | Scopes for the trainer's start-up functions and loops, the window barrier and the MLPerf logger's barrier; a coarse scope pops only what it pushed, since a loop's scope spans detail toggles | `d78e22c`, `f2b4f0d`, `a495b57` |
| `RUPPELL_MODE=witness` crashed the adapter | Scopes and hooks only while tracing; witness mode launches the step markers and nothing else | `c53050e` |
| `rocprofv3` crashed at start in every combination (57971, 58003) | The witness runs torch on the system ROCm runtime; `rocprofv3`'s temporaries go to node-local disk | `8699e06`, `4973f5e` |
| A traced run that had to JIT-compile aborted, LLVM "Cannot select" `mfma.f32.16x16x32.bf16` (59057, 59086) | Traced runs preload nothing; the tool binds to torch's SDK by itself | `6d74770` (job scripts) |

## The witness on this stack

`rocprofv3` 7.2.1 cannot run against the profiler SDK bundled in torch's
wheel, with or without ruppell (`tests/d4/d4_rpprobe.sh`, jobs 57971, 58003,
58051, 58090):
- **Without a preload, the tool aborts:** "configuration request outside of
  valid rocprofiler configuration period". torch's SDK copy configured it,
  and it registers with the system's copy, which was never initialised.
- **With torch's SDK preloaded, it segfaults in `rocprofiler_configure`.** It
  still does with torch's HSA runtime, register library, aqlprofile and
  comgr preloaded as well.

So the witness runs torch on the system's ROCm 7.2.1 runtime:
`d4_sysrt_shim` links the system libraries under the unversioned names
torch's libraries ask for (`libamdhip64.so`, …), and preloads them by those
names. That is the only witness this stack allows; the traced runs use
torch's own runtime.

Two more things go wrong under `rocprofv3` here, never under ruppell alone:
- **Temporaries:** it writes them to `<cwd>/.rocprofv3`. The training tree is
  on NFS with root_squash, so it failed at exit and hung (58051); fixed with
  `ROCPROF_TMPDIR`.
- **Shutdown:** DataLoader workers segfault there (16 in one run, 58052). In
  58259 that cut one rank's `rocprofv3` output short; ruppell's record of the
  rank was complete.

## The same program in training

On serving (M2.8), every checked step's program was identical traced and
untraced. Training cannot meet that test as worded, because **two untraced
runs differ**:
- **201 of 467 step segments per rank differ,** in both comparisons and at
  the same segments (`tests/d4/d4_program_check.py`, 58206, 58207).
- **190–198 of them run the same kernels in the same order,** with other
  grid sizes. The kernels are the training code's Triton kernels:
  `_weighted_layer_norm_*`, `_ln_mul_dropout_*`, the jagged split and
  concat. Triton autotune picks a block size by timing, separately in each
  process.
- **The first compute step also differs in kernel count.** Autotune's
  benchmark loop runs for a fixed time, so a traced run fits about 7K fewer
  launches there.

With Triton's autotune cache on (`TRITON_CACHE_AUTOTUNING=1`), a first run
fixes the choices and every later run reuses them (58259).
- **Traced against untraced:** 36 segments differ over 8 ranks.
- **Untraced against untraced:** 35.
- **The difference is one or two small kernels each:** `bfloat16_copy`,
  `FillFunctor`, `_add_embeddings_bwd_kernel`.

The operator settled the gate as "no more than two untraced runs differ",
with the cache on.

## Limits found here

- **The witness is on the system runtime, not torch's own** (above).
- **`export --steps` selects engine steps only.** A named step such as
  `startup` cannot be exported on its own yet.
- **A kernel's duration includes waiting.** A collective's kernel runs while
  it waits for the slowest rank, and on a side stream it overlaps compute.
  So kernel time summed per phase is GPU occupancy, not step time: one step
  showed 60 ms more collective time on a rank and was still the shortest.
  Read the step's GPU period and busy share for wall time.
- **A knob whose kernel keeps its symbol cannot be told apart in a census.**
  The FP8 qk MFMA inside `bwd_dkdv_kernel_0` is an example: only its effect
  shows.
- **Peak memory is not recorded.** The trainer logs none, and ruppell does
  not sample device memory.
