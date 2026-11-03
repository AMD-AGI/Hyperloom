# Adapting ruppell to another framework

How to write an adapter for a serving or training framework other than vLLM
(SGLang, ATOM, a training loop), from what the vLLM adapter
(`python/ruppell/adapters/vllm/`) took to get right. The runtime and the
analysis are framework-independent; an adapter's job is to tell the runtime
five things from inside the framework, and to prove it did.

## What an adapter provides

| What | Why | vLLM (what M2 used) |
| --- | --- | --- |
| **Run fields** (`ruppell.context`) | name each process: model, TP size, rank, role, graph mode | after `GPUModelRunner.load_model` |
| **Scopes** for every kernel | attribute GPU time to code | module hooks on the target and drafter models (`instrument`), plus `runner.<method>` scopes around the framework's own step work |
| **Graph keys** for every capture | name what a replay replays | `torch.cuda.CUDAGraph.capture_begin`, which every capture path goes through, keyed `role|passN|gM`; role from a thread-local set around the target's and the drafter's capture calls |
| **Steps** with the workload's fields | per-step load (running requests, rows, context lengths, kind) and a cross-rank fingerprint | one engine step per non-dummy `execute_model`, closed after `sample_tokens`; fields from the scheduler output every rank receives |
| **Named steps** for everything else | startup work lands in a step, not between steps | `load_model`, `profile_run`, `capture_model`, and all of `Worker.compile_or_warm_up_model` as `warmup` |
| **Reproducible run-time choices** | a traced and an untraced server must run the same GPU program, or no comparison of the two means anything | `RUPPELL_TUNING=FILE` installs `ruppell/adapters/tuning.py` (Inductor's and Triton's autotuners replay the first process's choice); in witness mode too |

## Process

1. **Facts first, on the pinned framework, before code** (M2.0). In its
   container, record: which processes load HIP and which run the model; the
   execution mode (eager, graphs, breakable graphs, `torch.compile`); whether
   module forward hooks fire inside captures; every place that captures a
   graph; the step function and what the scheduler hands it; what runs at
   startup (profiling, capture, warmup); how the server stops. A small job
   that serves a small model with the same flags answers most of it in
   minutes.
2. **Find the points, and wrap them from outside.** Prefer the framework's
   plugin mechanism (vLLM's `vllm.general_plugins` entry point) and patch
   classes right after their module is imported (a `sys.meta_path` finder),
   so processes that never import them stay untouched. Changing the
   framework's source changes the measured tree; avoid it.
3. **Check every signature you wrap, and refuse rather than half-trace.**
   A framework version that moved a method should produce no trace and a
   message naming the hook, never a trace with steps but no scopes.
4. **Start on a small model** (Qwen2.5-0.5B at TP1 and TP2 in M2): minutes
   per server start instead of tens of minutes, and exactly reproducible
   outputs for the correctness check. Iterate there until the section 5
   checks pass (`tests/m2/check_m2.py`), then move to the target model.
5. **Validate with the same checks, every run**: every replay joined or
   refused with a reason; attributed share; every kernel after the first
   engine step in exactly one step; side streams owned; fingerprints equal
   across ranks; schema on every step; the in-process witness (`rocprofv3`)
   identical per dispatch; the trace opening in both viewers and TraceLens;
   greedy outputs identical traced and untraced, or, where the target's
   outputs are not reproducible, the GPU program identical
   (`tests/m2/program_check.py`).

## What bit us in vLLM, and will bite again elsewhere

- **Startup drives the step function with made-up batches.** vLLM's warmup
  calls `execute_model` with an empty batch and one of `max_num_seqs`
  requests, and launches kernels directly between them. Counted as engine
  steps they looked like served load and left kernels between steps. Find
  every startup path (profiling, warmup, capture) and put it in named steps.
- **The framework's own work has no module.** Input copies, fills, index and
  sampling kernels ran outside any module forward: 11% of GPU time on the
  small model was unattributed until the runner's methods got scopes.
- **Graph handles are reused.** PyTorch destroys each `hipGraph_t` after
  instantiating it, so captures share addresses, and exec handles return
  between capture passes. Resolve handles in record order, never by value.
- **Breakable graphs break inside module forwards.** A scope opened at one
  segment boundary and closed at the next would corrupt the scope stack; the
  eager function between segments runs again at every replay, outside any
  hook, so it must carry the scope it was added in.
- **Steps across ranks.** A per-worker counter works only if every rank sees
  the same calls in the same order; prove it with a fingerprint of the
  scheduled work (request ids and token counts) and refuse a merge that
  disagrees.
- **Side streams.** Kernels on queues other than the step's stream are placed
  by the step whose CPU window launched them; record CPU step windows.
- **Servers are killed, not finalized.** Workers end before the profiler's
  finalizer; under `rocprofv3` SIGINT is held by its handler. The tool
  flushes every 5 s, the reader drops a partial record, and stop scripts wait
  two flush intervals first. Give the adapter a flush at the framework's
  shutdown that marks the records complete once it returns
  (`ruppell.flush(complete=True)`), so the analysis can tell a worker that
  reached it from one killed earlier; and find out whether the framework
  lets workers reach it: vLLM with `--shutdown-timeout 0` (its default)
  SIGKILLs its engine core's process tree, workers included, the moment it
  is asked to stop (`kill_process_tree`; B200, 2026-10-07). Also stop every helper process (vLLM's Rust
  frontend outlived its server and held the port), run containers with
  `--init` so killed processes do not linger as zombies, and bound the log of
  anything under `rocprofv3` (its signal handler once looped for 20 million
  lines).
- **Graphs on two queues.** The profiler SDK once gave a replay's kernels on
  a graph's second queue another call's correlation id (4 kernels in 17,503
  replays). The engine repairs a replay from the dispatch ids it lacks, only
  when its kernel sequence then matches exactly, and reports it.
- **Outputs may not be reproducible.** DeepSeek V4.1 Flash on vLLM differs
  between two untraced passes by up to 20 nats in prompt logprobs. Check the
  target's own reproducibility before choosing the correctness check.
- **Kernels chosen by timing at run time.** vLLM `torch.compile`s a helper
  even with its compilation off (`get_masked_input_and_mask`), and Inductor
  picks the kernel's block size by timing candidates in each process: the
  traced server's ranks ran grids 128 and 512 where the untraced ran 256
  (job 49826). Anything tuned by timing (Inductor, `@triton.autotune`, a
  library's own tuner) must be pinned by the adapter, and the pin must
  cover every rank: ranks of one server also chose differently.
- **Which rank waits is not the tracer's to report.** A collective kernel's
  time is transfer plus waiting for the slowest rank, and the tracer's host
  cost moves which rank is slowest. Compare collectives by their transfer
  estimate (shortest duration across ranks per launch); outputs label
  per-rank collective time (`ruppell.limits`, `collective-wait`).

## A training loop: the DLRMv4 adapter

`python/ruppell/adapters/dlrm/` traces MLPerf DLRMv4 training (TorchRec,
eager, no graphs), installed in every rank by a `sitecustomize` when
`RUPPELL_DLRM=1`. Its record: [`docs/milestones/dlrmv4.md`](milestones/dlrmv4.md).

| What | DLRMv4 |
| --- | --- |
| **Run fields** | rank, world size, role `trainer`, the FP8 and run knobs; `schemas/training.json` |
| **Scopes** | module hooks on the sharded model; `pipeline.<method>` scopes on TorchRec's train pipeline; `loop.<function>` scopes on the trainer's start-up functions, loops, window barrier, metrics and the MLPerf logger's barrier. Backward kernels get `bwd:<forward scope>` in the analysis (below). |
| **Graph keys** | none: the adapter warns if a graph is ever captured |
| **Steps** | one train step per `progress()` with grad enabled, fields `step`, `batch_index`, and a fingerprint of both |
| **Named steps** | `startup`, `window_boundary`, `eval`, `cleanup` |
| **Reproducible choices** | Triton autotunes by timing in each process, so a traced and an untraced run differ in grid sizes. `TRITON_CACHE_AUTOTUNING=1` with a first run that chooses makes them comparable. |

What a training loop needs that serving did not:

- **Backward attribution.** No module hook runs in autograd's backward.
  - The CPU op library records each op's autograd sequence number
    (`RUPPELL_OP_SEQ`).
  - The analysis gives a kernel launched anywhere inside autograd's
    `evaluate_function: <node>` wrapper the scope of the forward op that
    node differentiates: `bwd:<scope>`.
  - A node with no forward op (`AccumulateGrad`, where DDP's gradient
    all-reduce runs) gives `bwd:op:<node>`.
  - The backward function's scope replaces whatever coarse scope the
    caller's code had pushed.
- **Forked workers.** DataLoader workers fork from a traced rank and inherit
  its streams. The runtime's fork handler stops tracing in the child and
  drops those streams unflushed.
- **Scopes that span detail toggles.** A loop function's scope opens with
  detail on and may close with it off. The adapter's coarse scopes pop only
  what they pushed.
- **Collectives outside the step code wait for the slowest rank,** for up to
  hundreds of milliseconds per rank. Every one needs a scope: seed
  broadcasts, start-up barriers, logging barriers. Look for them by rank
  dependence: unattributed time that differs by rank is a wait.

## Starting points for SGLang (to verify in its facts job)

Where the same five things probably live, by analogy; each needs checking
against the pinned SGLang, as vLLM's did:

| What | Likely point |
| --- | --- |
| Run fields, scopes | the model runner's model load (`ModelRunner.load_model`); the draft worker's model for speculative decoding |
| Graph keys | `torch.cuda.CUDAGraph.capture_begin` as in vLLM (framework-independent); roles around the CUDA-graph runner's capture and the draft runner's |
| Steps and fields | the scheduler's batch step (`run_batch` / the model runner's `forward`), fields from the `ScheduleBatch` / `ForwardBatch` (requests, extend and decode lengths) |
| Named steps | server warmup requests and graph capture at startup |

The serving schema (`schemas/serving.json`) stays the same: an SGLang adapter
fills the same fields, so runs of the two frameworks compare in the census and
in TraceLens.
