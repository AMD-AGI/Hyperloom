# DeepSeek V4.1 Flash, C8 decode, read in TraceLens

What the M2 replay's traces show, analysed with TraceLens (`e34b294`) through
`python -m ruppell tracelens`. A worked example of the workflow, and a first
look at the workload; not a benchmark result (the replay was traced, and
tracing's overhead is M3's measurement).

**Run.** Job 48834 (Cluster A `e20-08`, MI350, TP4 on GPUs 0–3), pinned vLLM
`8f400a0ac` (v2+v6) with the knob-ab screen's base arm, C8 AgentX replay at
seed 42, 317 s measured, 35 requests; records analysed at `014805d` (48966).
**Window.** Engine steps 12,452–12,472 from the middle of aiperf's profiling
phase: 21 decode steps, 1 running request, 6 rows (1 + 5 speculative), about
137K tokens of context; 315 ms. The C8 regime is 1–1.5 requests in flight,
so this is its typical step.

The files analysed here came from the first converter (`ruppell tracelens`
on an exported Chrome trace, since replaced by the one in
`export/torch_profiler.py`), which stood the host events at kernel time.
The tables below are GPU-side (kernel times, attribution, collectives) and
read the same either way.

```bash
python -m ruppell tracelens RUN --steps 12452:12472 -o tl/
TraceLens_generate_perf_report_pytorch --profile_json_path tl/rank0.json --output_csvs_dir rank0_csv
TraceLens_generate_multi_rank_collective_report_pytorch --trace_pattern "tl/rank*.json" --world_size 4 --output_csvs_dir coll_csv
```

## GPU timeline (rank 0)

| | ms | % of window |
| --- | --- | --- |
| computation | 201.6 | 64.1 |
| exposed communication | 86.4 | 27.5 |
| exposed memcpy | 1.4 | 0.5 |
| idle | 25.1 | 8.0 |

A decode step takes 14.9 ms (median, every rank).

## Where computation goes (rank 0, 228.6 ms of kernel time)

| Component (scopes) | ms | % |
| --- | --- | --- |
| attention: MLA, sparse indexer, projections | 81.0 | 35.4 |
| MoE: routed experts, gate, shared experts | 75.0 | 32.8 |
| mHC pre/post | 34.6 | 15.1 |
| drafter (dspark, all) | 18.8 | 8.2 |
| runner: input preparation, sampling | 13.1 | 5.7 |
| other | 6.1 | 2.7 |

The largest single scopes, summed over the 43 layers: routed experts
45.7 ms, attention 36.2 ms, `mhc_pre_delayed` 25.9 ms, the sparse indexer op
13.5 ms, `fused_wqa_wkv` 11.5 ms, shared experts' `gate_up_proj` 11.0 ms,
`wo_b` 9.7 ms. The largest kernels: `_mxfp8_linear_kernel` 37.5 ms (4,809
launches), `_kernel` 17.1 ms (3,738 launches of an unnamed Triton kernel:
give it a name), MoE stage 1 15.8 ms, `mhc_pre_big_fuse` 9.9 ms, `mhc_post`
9.3 ms, MoE stage 2 9.1 ms.

## Communication: the cost is one rank arriving late

TraceLens's collective report over the four ranks:

- All-reduces (`cross_device_reduce_1stage`, 1,848 per rank in the window)
  take **6.3 µs** on average; their pure communication costs 11.7 ms in all.
- Their start times are skewed by up to **3.3 ms**. Ranks 0–2 waited 73–78 ms
  in total, rank 3 7.3 ms (straggler summary).
- The waiting is in the **first all-reduce of each forward** (after the
  vocab-parallel embedding; twice per step, target and drafter): 1.9 ms
  median on ranks 0–2, 6 µs on rank 3. That is about 3.8 ms of a 14.9 ms
  step, a quarter of it, spent by three GPUs waiting for the fourth.

Why rank 3 is late (from ruppell's trace, not TraceLens): all four ranks
start the step together (within 6 µs) and run the same 54 kernels before
that all-reduce, but on rank 3 each eager input-preparation kernel
(`runner.execute_model`: copies, index mapping, KV-block zeroing) is preceded
by 120–160 µs of idle GPU, against about 10 µs on rank 0. The HIP API
timeline says why: on ranks 0–2 those kernels are launched a whole step
ahead (launch call to kernel start: median 14.8–15.2 ms), so their hosts run
one step ahead of their GPUs as vLLM's asynchronous scheduling intends; on
rank 3 they start 12 µs after their launch call. Rank 3's host has no slack:
its per-step CPU work takes as long as a GPU step, its GPU waits on it
kernel by kernel, and the whole TP group waits for rank 3 at the first
all-reduce. Its `hipGraphLaunch` calls also take longer (144 ms over the
window's 42, against 93–125 ms on the other ranks).

**Is this the tracer? Partly.** Job 49194 measured the same wait on job
49062's two servers (same image, same prompts, both under `rocprofv3`; the
greedy phase, 178–191 steps each):

| | waiting ranks, first all-reduce (median) | the straggler rank |
| --- | --- | --- |
| untraced (markers only) | 1.35–1.65 ms | 6.5 µs |
| traced | 2.66–2.68 ms | 6.4 µs |

The straggler is there untraced: one rank's host lags and the others wait
at the first collective, which is the served stack's own behaviour, and the
lever is that host loop (worth over 10% of a step even untraced). Which rank
lags differs from run to run, so it is not something rank-specific in the
code. But tracing lengthens the wait by about 1.1 ms: ruppell's host-side
cost falls on the step's critical path through the lagging rank. M3's paired
A/B (plan-v0 4.7) measures what that does to per-request ITL.

## What TraceLens could not do here

- **No roofline or TFLOP/s**: its perf models key on aten op names and
  argument shapes; ruppell's ops are module scopes with the first input's
  shape (`Input Dims` of `attn` is `(6,)`), so every op is "other".
- **No CPU-side view**: the host events in the TraceLens file stand at kernel
  time. Host gaps like rank 3's come from ruppell's own trace.
