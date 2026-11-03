# Route: vLLM serving DeepSeek V4.1 Flash on ROCm (AgentX)

The one target where the profiling mode has been through the gates. Follow
[../SKILL.md](../SKILL.md) for the workflow; this file gives this target's
facts. Update it when a campaign settles.

## The target

| | |
| --- | --- |
| framework | vLLM, v2 model runner, pinned `8f400a0ac` (v2+v6), the knob-ab screen's base arm (its env minus knob flags); `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `cudagraph_mode` FULL_AND_PIECEWISE, compilation mode NONE |
| image | `sha256:960228cf…` (torch 2.12.0+git6bbd260, triton 3.7.1) |
| model | DeepSeek V4.1 Flash, TP4, speculative decoding with the dspark drafter (5 tokens, synthetic acceptance) |
| platform | Cluster A, MI350 (gfx950), GPUs 0–3 |
| workload | AgentX replay, seed 42, `NUM_PROMPTS` 320, `DURATION` 600, after a capped warmup replay; C8 and C32 |
| regimes | C8: one. C32: the first 300 s of the measured phase (about 5 running, no queue) and the rest (about 18 running, queued) |

## Modes

| mode | settings | operating point C8 | operating point C32 | use |
| --- | --- | --- | --- | --- |
| full detail | `RUPPELL_DETAIL=all` (default) | **fails** (ITL 1.26–1.47×, in flight +23–33%; round a) | **fails** (ITL 1.08–1.17×) | structure and correctness checks only |
| sampled | `RUPPELL_DETAIL=5/1000 RUPPELL_CPU_OPS=1 RUPPELL_TUNING=FILE` | passes (rounds b, d) | passes (rounds c, d) | valid, but mixed and prefill steps under-sampled (50402/50403) |
| every mixed and prefill step | the above plus `RUPPELL_DETAIL_KINDS=mixed,prefill` | outside the ±1% band: in flight 1.174–1.180 against ≤ 1.172 (50632, round e) | outside: throughput 315.0 against ≥ 320.0 (−2.5%; 50633, round e) | **the profiling mode**: a slightly slower server, every mixed and prefill step traced (operator, 2026-09-30: tracing finds optimization sites, it is not a benchmark). Profiles: prof_c8_b, prof_c32_b |
| mixed and prefill at 1/5 | the sampled mode plus `RUPPELL_DETAIL_KINDS=mixed:1/5,prefill:1/5` | not run (round f cancelled) | not run | cheaper, if a closer operating point is ever needed |

Correctness, full detail with the CPU op tree and the tuning pin (50303):
- **GPU program:** passes.
- **Compute parity:** fails. Tracing turned rank 2 into the straggler; its
  compute total read 0.958, with four families 3–4% faster.
- **Collective transfer:** fails, one family at 1.032.
- **Two untraced servers (50304):** pass everything.
- **Parity in the sampled modes:** not measured.

Records: `docs/milestones/m3.md`; gates: `docs/plan-m3.md`.

## Run

On Cluster A, one job serves, replays and analyses (`tests/m3/profile_agentx.sh`):

```bash
ssh <login node> 'PROFILE_CONC=8  PROFILE_NAME=prof_c8_<tag>  PROFILE_KINDS=mixed,prefill bash -s -- profile_agentx.sh 04:00:00 any' < cluster_m2_submit.sh
ssh <login node> 'PROFILE_CONC=32 PROFILE_NAME=prof_c32_<tag> PROFILE_KINDS=mixed,prefill bash -s -- profile_agentx.sh 04:00:00 any' < cluster_m2_submit.sh
```

- **Where results go.** `~/ruppell-work/profile/<name>/`; the records stay on
  the node.
- **Redoing the analysis.** Resubmitting the same name to the same node
  skips the replay and redoes the analysis.
- **Operating-point A/B for a mode:** `tests/m3/m3_ab.sh` (`M3_CONC`,
  `M3_DETAIL`, `M3_ON_ENV`, `M3_CAMPAIGN`), then `tests/m3/op_point.py`.
- **Correctness:** `tests/m2/m2_dsv41_correct.sh` (`M3_TRACED_ENV`;
  `M2_SECOND=control` for control against control).

## This target's measured limits

| limitation | measured here | what to do |
| --- | --- | --- |
| collective wait | about 95% of all-reduce time in first-token steps is waiting; the rank that waits least moved from 3 to 1 (49794), 0.05× to 23× per rank | use transfer (`collectives.<regime>.transfer_ms`) |
| straggler's compute shift | the traced straggler's compute total 0.958; untraced pairs agree within 1% (50303, 50304) | differences under about 5% between sites or ranks are unresolved |
| host time of always-traced steps | mixed steps' share of engine time 55% → 64% (C32 after 300 s), 29% → 37% (first 300 s), 5.5% → 7.6% (C8) | step-time weights from a 5/1000 run; kernel-time estimates are fine |
| sampling by count | at 5/1000: mixed steps 1–3 samples where they were 5.5% (C8), 29% and 55% (C32) of engine time; sampled rows 38.6 against 151.7 | trace mixed and prefill (`RUPPELL_DETAIL_KINDS`); use `stratified` |
| runner work | coverage 95.01% and 94.81% on ranks 0 and 1 at C8 (gate 95%); C32 96.2–96.3% | `runner.*` is framework overhead; the C8 coverage gate fails as it stands |
| no aten op | 19–24% of kernel time (`_mxfp8_linear_kernel`, Triton from Python) | rank by module for these |
| TraceLens categories | 78–79% of kernel time in `other` | use kernel and op summaries |
| tuning pin | 16 choices recorded and replayed per server | block sizes are rank 0's-first choice everywhere |
| empty steps | about half the first-token marker pairs hold no kernels; one appears in one untraced server and not another | counted, not compared |

## What the model spends time on (for orientation, not a result)

Decode steps at 5/1000 (docs/analysis/dsv41-agentx-profile.md):
- **Modules:** MoE experts 24–32% of kernel time; MLA attention 14%; the
  sparse attention indexer 6–17% (it grows with the queue at C32);
  `mhc_pre_delayed` 6–9%; `attn.wo_b` 6–9%; collectives 6–14%.
- **Kernels:** `_mxfp8_linear_kernel` 11–14%; the MoE `mfma` pair 11–23%.

Stratified by kind (sampled-by-kind mode, pending its gate): MoE experts
25% (C8), 31% (C32); attention about 14%.

## Reading the Chrome trace (Perfetto)

`ruppell export RUN --steps A:B -o trace.json` writes a window that opens in
https://ui.perfetto.dev (a `.json.gz` opens as it is). Per rank:
- **Host process:**
  - HIP API calls on their real threads;
  - the aten op tree (`RUPPELL_CPU_OPS`), with input shapes;
  - ruppell's module scopes, all nested.
- **GPU process:**
  - one lane per queue;
  - kernels named `<module scope> · <kernel>`, with scope, attribution,
    graph key, replay, node and step in their args;
  - step tracks whose args are the step's fields (rows, running requests,
    context lengths, kind).
- **Flow arrows** from each launch call to the kernels it launched.

Example: prof_c32_b steps 20003–20012, 216 MB of JSON
(`~/ruppell-work/profile/prof_c32_b/viz_c32_steps20003-20012.json.gz` on
Cluster A).

**A `hipGraphLaunch` arrow that lands much later, on another lane, is
normal.** The call only enqueues the graph. Its kernels run on the graph's
own queues when the GPU reaches them, and vLLM's host runs ahead of the GPU:
it launches a step's graphs, the drafter's included, while the GPU is still
on the previous step or two.

Measured on that window: every one of 36 graph launches reaches the kernels
of exactly one replay of one graph (for example `drafter|pass2|g3524`
replay 115, 163 kernels; a target replay is about 1,534). The first kernel
starts after the call by:
- 10–12 ms in decode steps;
- up to 0.85 s behind full-chunk mixed steps (0.46–0.76 s each).

For 99.9–100% of that delay, the same GPU was running earlier kernels.
Tracing slows the host, so an untraced host runs further ahead, not less.

What would be wrong instead, and how to tell:
- **Arrows reaching kernels of two replays, or of another graph.** Check
  with `python3 tests/m3/check_flows.py TRACE.json[.gz]`. It prints, per
  graph launch, the replay its arrows reach, the delay, and the share of the
  delay the GPU was busy. It exits 1 if any launch reaches more than one
  replay.
- **A delay during which the GPU sat idle.** That would be a real gap, and
  belongs to the host, not the queue.
- **Eager kernels late relative to their launch call** (not a graph). They
  wait in the queue the same way. Read their arrows the same way.

Other things that look odd and are not:
- **Steps with host calls but no kernels:** untraced steps under detail
  sampling.
- **Per-rank all-reduce durations differing by large factors:** waiting for
  the slowest rank. Read the shortest duration across ranks as the transfer
  (`collective-wait`).

## Known quirks

- **Module shares include the module's all-reduce.** A row-parallel module's
  output all-reduce runs inside its scope. Of 480 ms under `attn.wo_b` in the
  C32 window's mixed steps, 383 ms were all-reduce kernels. Before ranking a
  module, split its kernels into collectives and the rest (`by_kernel`, or
  the trace).
- **Mixed steps above 1,024 tokens run eagerly** (`--max-cudagraph-capture-size
  1024`): 2,500–2,900 eager kernels per step, with only the drafter's graph
  replayed. Untraced, they cost the host about 118 ms whatever their size;
  mixed steps up to 1,024 tokens cost about 30 ms. Steps of 1,025–about 3,000
  tokens are host-bound (GPU work about 50–90 ms). A rank whose host is a
  little slower then holds up the others in the all-reduces. That is the
  rank imbalance seen in mid-size mixed steps, and it is not the tracer's
  doing (`tests/m3/step_times.py`, untraced against traced).

- **Startup.** The CPU op tree needs `fb7d838` or later: earlier builds
  crash at startup on a Python-dispatch tensor.
- **TraceLens output.** Its Excel writer fails on these files; ask for CSVs
  only.
- **Collective kernels.** vLLM's and AITER's custom all-reduce
  (`cross_device_reduce_1stage` and `cross_device_reduce_2stage`) plus RCCL.
  `ruppell.limits.is_collective` knows all three.
- **Outputs.** DeepSeek V4.1 outputs vary run to run even untraced (up to 20
  nats), so correctness rests on the GPU-program check, not on outputs.
