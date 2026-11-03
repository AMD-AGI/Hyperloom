# DeepSeek-V4.1-Flash AgentX on B200: what ruppell's traces show (C8 and C32)

A profile, not a benchmark: one 300 s seeded AgentX replay per point, where
InferenceX's campaign runs an hour. Numbers are for this node and these
runs.

| | |
| --- | --- |
| Node, date | `<host>`, 8× B200, driver 580.126.09; 2026-09-30 |
| Server | InferenceX's B200 recipe (`benchmarks/single_node/agentic/dsv41flash_fp4_vllm_mtp.sh`): `vllm/vllm-openai:deepseekv41-flash-0909` (vLLM 0.1.dev20904+g179dd0fa9, torch 2.13.0+cu130), `deepseek-ai/DeepSeek-V4.1-Flash` at revision `dba1be0a`, TP4, dspark MTP (5 draft tokens, synthetic acceptance 3.51), Engram weights in CPU memory, CUDA-graph mode FULL_AND_PIECEWISE, capture size 64 (C8) and 256 (C32) |
| Load | Arbor's `bench_agentx.sh` (Arbor `6516ee9`, InferenceX `32c2d45ca`, aiperf `754356e`): WEKA agentic-coding traces `semianalysisai/cc-traces-weka-062126` (snapshot `23f152f6`), seed 42, 2 warmup requests per lane (the harness default is 10), 300 s profiling |
| Arms | an untraced server on GPUs 0-3 and a traced one on GPUs 4-7 running the same replay at the same time (`tests/cuda/run_agentx.sh`) |
| ruppell | C8 served at `1b588a7`, replayed at `fe4df7b`, aggregated and analysed at `c9f9502`; C32 served, replayed and analysed at `c9f9502`; the same tool digest `4685a2f8…` for both |

## The client's view

| | C8 untraced | C8 traced | C32 untraced | C32 traced |
| --- | --- | --- | --- | --- |
| Output tok/s per GPU | 69.9 | 68.4 (−2.2%) | 149.6 | 148.8 (−0.5%) |
| Requests completed in profiling | 46 | 46 | 259 | 258 |
| TTFT p50 / p90 (s) | 0.195 / 0.646 | 0.240 / 0.877 | 0.137 / 0.589 | 0.190 / 1.475 |
| TPOT p50 / p90 (ms) | 2.55 / 3.36 | 2.64 / 3.61 | 3.27 / 4.50 | 3.61 / 6.37 |
| KV cache hit rate | 0.764 | 0.748 | 0.690 | 0.686 |

Both arms ran side by side on one host, so they share its CPU; tracing's
cost here is an upper bound on the latency side (TTFT +23% and +39% at the
median, TPOT +3.5% and +10%) and about nothing on throughput.

## The server's view (the traced server's census, aiperf's profiling phase, rank 0)

| | C8 | C32 |
| --- | --- | --- |
| Requests in flight, time-weighted (aiperf / census) | 0.775 / 0.759 (−2%) | 2.675 / 2.853 (+6.7%) |
| Engine steps; share of the phase inside a step | 13,554; **40%** | 12,646; **62%** |
| Decode: share of step time, steps, period p50, GPU busy | 97%, 13,529, 8.7 ms, 92% | 68%, 12,401, 9.7 ms, 93% |
| Decode batch | one request in 98% of steps, 6 rows (1 + 5 draft), context ~225 k tokens | 1-6 requests, 12 rows (p50), 30 (p90), context ~100 k (p50) |
| Mixed prefill + decode steps | 2 | **203, 29% of step time, 264 ms mean, GPU busy 53%** |
| Prefill steps | 11, 336 ms mean, GPU busy 82% | 21, 238 ms mean, GPU busy 51% |

- **AgentX at C8 barely loads the server.** Eight lanes replaying agentic
  sessions (with their tool and think gaps) keep 0.77 requests in flight:
  the server steps 40% of the time and decodes one request at a time. The
  census and aiperf agree on the load within 2% (6.7% at C32).
- **Decode GPU time** (C8; C32 alike): MoE 23%, FlashMLA attention 22%, the
  decoder layer's own ops (the hyper-connection pre-norm) 16%, row- and
  column-parallel linears 17%, the model runner's own work (sampling,
  rejection, drafting glue) 5.6%, the sparse-attention indexer 4.6%. At C32
  MoE rises to 28% (more tokens per step).
- **Prefill** at these context lengths (~230 k tokens) is dominated by the
  sparse-attention indexer (34% of prefill GPU time at C8).

## Leads

1. **C32's mixed steps stall on kernels compiled or loaded during serving.**
   The worst mixed steps last 1.7-3.5 s with the GPUs 3-33% busy. On every
   rank, the longest host calls in the profiling phase are
   `cuKernelSetAttribute` (up to 0.59 s, 13-19 per rank), and each falls
   inside one of those steps (step 1202: 2.1 s, calls of 556, 587 and 506 ms
   on ranks 0, 1 and 3; step 1516: 3.5 s). vLLM's own JIT monitor warns of
   Triton kernels compiled during inference (the Engram lookup, dspark's
   block scores and candidate masks, the rejection and resample kernels, the
   KV ring slot mapping), in the untraced server as well: the recipe's
   warmup does not cover the mixed-batch shapes. Extending warmup to them
   should remove most of the 21 s of GPU idle inside C32's mixed and prefill
   steps (not yet tried).
2. **That is also C32's rank skew.** Ranks stall in different steps, and the
   others wait inside the one-shot all-reduce: in mixed and prefill steps,
   the ops that communicate (row-parallel linears, MoE, the vocabulary
   embedding, the logits) take 5.8-9.6 s on one rank against 9.0-12.4 s on
   another, while attention and the indexer are equal on every rank.
   Across all steps the per-step spread of GPU time between ranks is
   0.7% (C8) and 1.0% (C32) at the median, and 66% at C32's p99.
3. **FlashMLA's decode metadata kernel costs more than the attention it
   schedules** at these contexts: `get_mla_metadata_kernel` averages 29 µs
   per call against 12 µs for the FP8 sparse decode kernel (C8 window,
   TraceLens), about 15% of attention's time.
4. **The decode GEMMs are small**: M = 6 (C8) or ~12 (C32) rows. The FP4 MoE
   expert GEMMs take 20 µs (C8) to 31 µs (C32) per call; the vocabulary
   projection is a 54 µs split-K GEMM per step.

## What the traces covered (checks, `tests/m2/check_m2.py`)

| | C8 | C32 |
| --- | --- | --- |
| Kernels per rank; dropped | 26.75 M; 0 | 26.91 M; 0 |
| Captures; replays joined (by node id) | 52,832; 79,259 of 79,259 | 54,872; 77,849 of 77,849 |
| Attributed share of GPU time | 0.99992-0.99994 | 0.99996 |
| Step ids and fingerprints across ranks | equal | equal |
| Census running requests vs aiperf in flight (bar ±10%) | 0.759 vs 0.775 (−2.1%) | 2.853 vs 2.675 (+6.7%) |
| Windowed Chrome trace in Perfetto | 479,541 slices, no error stats | 480,839 slices, none |
| Kernels outside every step after the first engine step | 371 per rank, **all after the last engine step** | 468, all after the last; and 955 unowned side-stream kernels, all after the last |
| Decode GPU time in a model module (bar 95%) | 94.4-95.2%: the rest is the model runner's own scope (attributed, not a module) | 94.9-95.3%, the same |

The two checks that fail count work after the replay ended (a trailing graph
replay when the servers went idle) and the runner's own share of decode;
neither is missing attribution.

## How it was run, and what went wrong on the way

- The adapter at first refused this vLLM build (`capture_model` takes no
  `profile_only`; `warmup_kernels` is public): fixed in `457caeb`, checked
  against the build's `GPUModelRunner`.
- The harness copied from Windows working trees had CRLF endings (even
  through `git archive`, which applies `core.autocrlf`); it is now exported
  with `-c core.autocrlf=false`.
- `benchmark_lib.sh` rebuilds aiperf's environment on every run, so two
  replays sharing one removed each other's: one per arm now (`fe4df7b`).
  InferenceX's aggregation needs its `infx` package (`c9f9502`).
- C8's first untraced replay attempt was still failing in the background
  for about a minute of the second attempt's warmup; the profiling phase
  was not affected.

Artifacts on the node (not in git): `/mnt/local_disk$HOME/agentx_c8`,
`agentx_c32` (records, census, Chrome windows, TraceLens reports, summaries).
