# DeepSeek V4.1 Flash on AgentX: first profile (decode steps; mixed steps under-sampled)

**Status: a profile of the decode steps, not of the engine.** The replays
were traced with detail sampling (5 of every 1000 steps), which picks steps
by count. Mixed steps (prefill chunks batched with decode) are few steps but
much of the engine's time, and got 1–3 samples each (table below). The shares
here are therefore decode-step shares. A profile of the whole engine needs
mixed and prefill steps sampled on purpose ([next](#next)).

**Runs.**
- Cluster A (MI350, TP4 on GPUs 0–3), pinned vLLM `8f400a0ac` (v2+v6) with the
  knob-ab screen's base arm.
- AgentX replay at seed 42, `NUM_PROMPTS` 320, `DURATION` 600, after the
  capped warmup replay.
- The mode M3's gates pass: `RUPPELL_DETAIL=5/1000`, the CPU op tree
  (`RUPPELL_CPU_OPS=1`), run-time tuning pinned (`RUPPELL_TUNING`).
- Job 50402 (C8, node e17-18) and job 50403 (C32, e16-18), both at `4bc6c4d`.
- Analysed by 50495 and 50531 at `02da83d`: census, checks,
  `tests/m3/profile_report.py`, TraceLens CSVs.
- Results in `~/ruppell-work/profile/prof_c{8,32}_a/` on Cluster A.

**Checks.**

| Check | C8 | C32 |
| --- | --- | --- |
| Registry, steps, schema | pass | pass |
| Running requests, census vs aiperf in flight | pass (1.134 against 1.152) | pass (13.45 against 13.67) |
| Coverage: decode GPU time on a module scope (gate: ≥ 95%) | **fail on ranks 0 and 1** (95.01%, 94.81%); the rest is the runner's own work | pass (96.2–96.3%) |

## Phase coverage: what the sample holds

Share of engine time is over every engine step (host step windows, begin to
end, recorded whether the step was sampled or not):

| | step kind | share of engine time | steps | sampled |
| --- | --- | --- | --- | --- |
| C8 | decode | 91.8% | 37,037 | 183 |
| | mixed | 5.5% | 168 | 1 |
| | prefill | 2.6% | 77 | 1 |
| C32, first 300 s | decode | 70.7% | 15,267 | 79 |
| | mixed | 29.0% | 449 | 1 |
| | prefill | 0.4% | 6 | 0 |
| C32, after 300 s | mixed | **54.7%** | 625 | 3 |
| | decode | 45.3% | 7,806 | 37 |

The sampled steps match the run on running requests (C32 first 300 s: 5.3
sampled against 4.8 over all steps), but not on rows (38.6 against 151.7).
Rows, the tokens processed per step, are what mixed steps carry. After 300 s
at C32, more than half the engine's time goes to steps the sample barely
contains. This is the same failure as Hyperloom's missed prefill
([related-tools.md](../related-tools.md)), measured here rather than inferred.

## Decode steps

Shares of rank 0's kernel time over the sampled steps (collective time
included, as its transfer plus waiting; see below).

| | C8 | C32 first 300 s | C32 after 300 s |
| --- | --- | --- | --- |
| kernel ms per sampled step | 13.2 | 16.6 | 39.2 |
| busy over period | 0.90 | 0.93 | 0.95 |
| MoE experts (`ffn.experts`) | 24.5% | 32.0% | 32.0% |
| attention (`attn`, the MLA op) | 14.0% | 14.0% | 14.0% |
| sparse attention indexer (`attn.indexer`) | 6.1% | 7.1% | **16.6%** |
| `mhc_pre_delayed` | 9.5% | 7.6% | 5.7% |
| `attn.wo_b` | 7.2% | 5.6% | 8.0% |
| collective kernels | 8–11% | 6–7% | 12–14% |

- **Kernels.** The largest are `_mxfp8_linear_kernel` (14.3%, 12.9%, 11.4%)
  and the MoE `mfma_moe1`/`mfma_moe2` pair (11%, 22%, 23% together).
- **Past 300 s at C32,** the indexer's `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`
  (8.8%) and prefill attention (`pa_prefill`, 2.3%) appear, from the 3 mixed
  steps sampled.
- **Aten ops.**
  - `aiter::fused_moe_` 13–23%, `vllm::moe_forward_shared` 4–8%.
  - `vllm::mxfp8_quantize` 3–6%.
  - `vllm::rocm_aiter_sparse_attn_indexer` 3–13%.
  - **No aten op:** 19–24% of kernel time, mostly Triton kernels launched
    straight from Python (`_mxfp8_linear_kernel`). These are still attributed
    by module.

**Collectives.** Transfer (each launch's shortest duration across ranks) is:
- 158 ms over the 185 sampled steps at C8;
- 75 ms over 80 steps at C32 before 300 s;
- 180 ms over 40 steps after.

Per-rank waiting is recorded but untrusted (`collective-wait`): tracing
moves which rank waits.

**TraceLens** (perf report on rank 0's sampled steps) puts 78–79% of kernel
time in category `other`: its op categories do not know AITER, Triton and
the custom HIP ops, which is most of this model. Its GPU timeline counts the
gaps between sampled steps as idle; use the census's busy-over-period
instead.

## Next

Sample mixed and prefill steps on purpose:
- **Trace them.** The adapter knows a step's kind from the scheduler output
  before the step runs, so it can trace every mixed and prefill step (about
  1–5% of steps) and keep sampling decode at 5/1000.
- **Stratify the report.** Report per kind, and combine the kinds weighted by
  each kind's share of engine time.
- **Gate the mode.** That is a new tracing mode, so it has to pass the
  operating-point gate before its profile stands.

## Second pass: every mixed and prefill step traced (prof_c8_b, prof_c32_b)

Mode: decode at 5/1000, every mixed and prefill step traced
(`RUPPELL_DETAIL_KINDS=mixed,prefill`), the CPU op tree and the tuning pin.
Jobs 50634 (C8, c18-08) and 50635 (C32, e12-08); analysed by 50882, 50883,
50915, 50916; host step times by 50929, 50930, 51356. The server runs
slightly slower (round e: C32 throughput −2.5%). Tracing is for finding sites,
not benchmarking (operator, 2026-09-30).

Stratified kernel time by step kind, rank 0:

| | C8 | C32 first 300 s | C32 after 300 s |
| --- | --- | --- | --- |
| decode: share, ms per step | 93%, 12.6 | 75%, 15.3 | 59%, 26.4 |
| mixed: share, ms per step | 4.4%, 128.7 | 25%, 182.3 | 41%, 222.8 |
| prefill: share, ms per step | 2.5%, 166.4 | 0.1% | none |
| collectives in mixed steps: share; transfer per step | 34%; 19.0 ms | 33%; 32.6 ms | 28%; 40.9 ms |

What it shows:

- **Mixed steps above 1,024 tokens run eagerly and cost the host about 118 ms
  whatever their size** (untraced; about 30 ms up to 1,024). Mid-size ones
  (1,025–about 3,000 tokens) are host-bound.
- **Mixed-step all-reduces are bandwidth-bound.** They are 10–160 MB each,
  about 100–107 GB/s per call, on RCCL above about 6,500 rows and on AITER's
  two-stage kernel below.
- **The indexer's prefill path** is 11–18% of mixed-step op time at C32. It
  includes two top-k prefill implementations in the same steps.
- **Decode** confirms the ranking that Plan 22's D06b, D07, D08, D09 and D11
  own.

The follow-ups are the runner's Plan 24
(`plans/24_DSV41_FLASH_REPLAY_PROFILE_FOLLOWUPS.md` in jiaweic-runner). The
C32 Chrome trace window is `viz_c32_steps20003-20012.json.gz` in
`~/ruppell-work/profile/prof_c32_b/` on Cluster A.
