---
name: ruppell-profile
description: Profile LLM serving with ruppell (graph-aware GPU tracing with module and aten attribution), check that the trace can be trusted, reconcile its limitations, read it with TraceLens's reports, and rank optimization sites. Routes to target-specific instructions (vLLM on DeepSeek V4.1 Flash, ROCm, is the validated serving target; MLPerf DLRMv4 training on MI350X has a training adapter) and says what to build and validate for any other target. Use when asked to profile, trace or find bottlenecks in model serving or DLRMv4 training, or to read a ruppell run.
---

# Profiling model serving with ruppell

Audience: an agent (or a person) asked "where does the time go, and what
should we optimize?" for a model server. Read this file, then the route for
your target; look things up in [reference.md](reference.md).

ruppell records every GPU kernel a program runs, including kernels replayed
from CUDA/HIP graphs, and attributes each to the module (and, with the CPU op
tree, the aten op) that launched it, per engine step. Tracing changes what it
measures. How much, and in which numbers, has to be measured for each
target. Most of this skill is about that: which numbers from a traced run
may be used, and for what.

## Route by target

A target is a framework and version, a model, a workload, and a platform.
Find yours; if no row matches exactly, what differs decides what you must do
first ([new-target.md](new-target.md)).

| target | status | follow |
| --- | --- | --- |
| vLLM (v2 model runner, pinned `8f400a0ac`) serving DeepSeek V4.1 Flash at TP4, ROCm (MI350), AgentX at C8 and C32, seed 42 | a profiling mode that passes the operating-point gate; correctness checked; limits measured | [targets/vllm-dsv41-rocm.md](targets/vllm-dsv41-rocm.md) |
| the same stack, another workload, concurrency, seed or MI355X | the adapter works; gates not run for that load | [new-target.md](new-target.md), "Same stack, different load" |
| vLLM on CUDA (B200: Qwen2.5-0.5B, a 35B MoE) | traced and checked against Nsight Systems (C3, C4); every step traced (no sampling on CUDA); overhead measured (throughput −5.8% to −8.3% at C8); no operating-point gate | [new-target.md](new-target.md), "Another platform" |
| another model on vLLM, another vLLM version | not validated | [new-target.md](new-target.md) |
| MLPerf DLRMv4 training (TorchRec, HSTU on Yambda-5b, eager), ROCm (MI350X), one node, 8 ranks, `<colleague>/dlrmv4_opt` `d88f609` | the `dlrm` adapter (`RUPPELL_DLRM=1`), with backward attribution; the runner's gates met (one under an operator rule); findings recorded | [docs/adapting-frameworks.md](../../../docs/adapting-frameworks.md), "A training loop: the DLRMv4 adapter"; record [docs/milestones/dlrmv4.md](../../../docs/milestones/dlrmv4.md); findings [docs/analysis/dlrmv4-training-profile.md](../../../docs/analysis/dlrmv4-training-profile.md) |
| the same DLRMv4 training on MI355X, on more than one node, or on a later training commit | the adapter is unchanged; not traced there | [new-target.md](new-target.md), "Same stack, different load", after the DLRMv4 row above |
| SGLang, ATOM, TensorRT-LLM, another training loop | no adapter | [new-target.md](new-target.md), "Another framework" |

## General recommendations (any target)

1. **Know what the mode does to the server.** Tracing is for finding where
   to optimize, not for benchmarking (operator, 2026-09-30): a traced server
   that runs slower is acceptable. What a profile needs is:
   - the same GPU program (the GPU-program check);
   - kernel times that carry over (kernel-time parity);
   - a load close to the untraced one.

   The operating-point measurement says how close the load is; report it
   with the profile, and say what moved (C32 throughput −2.5% is a slower
   server; in-flight far off is another batching regime). A mode never
   tested on your target is unvalidated: say so beside every number.
2. **Check before you read.** Run the section 5 checks (joins, coverage,
   steps, schema, dropped records, load) and the phase coverage before
   quoting any share. A failed or not-run check travels with every number
   it affects.
3. **Know what the sample holds.** Sampling by step count starves kinds of
   step that are few but long (prefill, mixed batches). Compare each kind's
   share of engine time over every step with its sampled count. Trace the
   long kinds on purpose (`RUPPELL_DETAIL_KINDS`) and weight each kind by
   its steps. A profile that silently drops a phase steers optimization to
   the phase it happened to capture.
4. **Rank by GPU kernel time, weighted by step kind, per regime.** Not by:
   - collective waiting;
   - host step time of traced steps;
   - a timeline that counts unsampled gaps as idle;
   - a categorizer that does not know the model's kernels.
5. **Collectives: transfer, not wait.** A collective kernel's time is
   transfer plus waiting for the slowest rank, and any tracer moves which
   rank is slowest. Use each launch's shortest duration across ranks.
6. **Latency, throughput and host-side behaviour come from untraced runs.**
   Tracing slows the host. The operating-point gate says the load is kept,
   not the latency.
7. **Make run-time choices reproducible before comparing.** Kernels chosen
   by timing (Inductor, `@triton.autotune`, a library's own tuner) must be
   pinned across ranks and servers (`RUPPELL_TUNING`), or a traced and an
   untraced server run different programs.
8. **Know the workload before reading its profile.** The load regimes
   (C32 at seed 42 is two regimes), what each concurrency actually puts in
   flight, and what the benchmark sends. Report every regime and
   concurrency separately; never pool them.
9. **A kernel share is not a serving gain.** Give the Amdahl ceiling, and
   say an A/B decides.
10. **Only the operator sets gates.** Report a gate as it stands; if it
    cannot be evaluated or looks wrong, say so and still report it.

## Workflow (any target)

1. **Route.** Find the target above. If it is not validated, do the work in
   [new-target.md](new-target.md) first, or label the profile unvalidated.
2. **Choose the mode and the load.** From the route file: the mode that
   passed; the workload, concurrency, seed and duration; the regimes.
3. **Run.** The route file's job, or by hand ([reference.md](reference.md#running-by-hand)).
4. **Check.** Section 5 checks, phase coverage, and the sampled load against
   the whole run's ([reference.md](reference.md#checks)).
5. **Read.** The stratified profile per regime and rank; collectives by
   transfer ([reference.md](reference.md#outputs)).
6. **Reconcile.** Every number against the limitations
   ([reference.md](reference.md#reconciling-the-limitations)), with the
   target's measured values from its route file.
7. **Read it with TraceLens**, using only the reports that hold on these
   files ([reference.md](reference.md#tracelens)).
8. **Rank and report** with the template
   ([reference.md](reference.md#report-template)). Say which numbers are
   measured and which inferred.

## Improving it

If a limitation blocks the question you were asked, [new-target.md](new-target.md)
lists what would lift it and how to contribute it to this repository.
