# Related tools: what they do on vLLM serving, and why ruppell does it differently

Written 2026-09-30 from the sources named in each section. Recheck them when
the tools move.

## TraceLens's Analysis agent (not used)

Source: [`TraceLens/Agent/Analysis/README.md`](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/Agent/Analysis/README.md),
`TraceLens/Agent/Profiling/README.md`, and the code at `e34b294`
(`Trace2Tree/trace_capture_merge_experimental.py`, `TraceUtils/split_trace/`).

**What it is.** An LLM front end to TraceLens's PyTorch-trace reports. It
takes one rank's `torch.profiler` trace and runs a skill-file workflow:
1. It generates the perf report (`TraceLens_generate_perf_report_pytorch_inference`
   for vLLM, SGLang and ATOM).
2. It prepares data by category.
3. Sub-agents run in parallel: three system analyzers (CPU idle; memcpy and
   blocking communication; fusion) and nine compute categories (GEMM, SDPA,
   elementwise, reduce, Triton, MoE, norm, convolution, generic).
4. It writes a ranked `analysis.md` of bottlenecks: roofline efficiency per
   kernel, with an impact estimate.

It needs:
- a `<platform>.json` arch file;
- aten ops with shapes, which the roofline models key on;
- a frontier model with 1M context running 13 sub-agents.

**How it handles CUDA graphs.** "Graph-only traces (no capture) are not
supported." A graph-mode run needs a replay trace (the torch profiler during
serving) and a capture folder (the torch profiler around graph capture at
startup, `--profiler-config.capture_torch_profiler`). The capture folder
comes from a patched image for vLLM 0.14–0.25, or upstream flags from 0.26.
The merge works like this:
- **Iteration → capture.** Each engine iteration is found from vLLM's
  annotations and its batch size read from them. The batch size is rounded up
  to the nearest captured size and looked up as `{bs}_FULL` or `{bs}_PIECEWISE`.
  The choice between the two is a heuristic: PIECEWISE when the iteration
  launched more than one graph.
- **Capture graph → replay graph.** Capture graphs are paired with the
  iteration's graph launches either one to all, or one to one by position
  when the counts are equal. Otherwise the iteration is skipped with "merging
  may be incorrect".
- **Kernel → captured launch.** Replayed kernels are matched to captured
  launches by kernel name in sequence: a greedy forward scan, or grouping by
  name and pairing by position, with tie-breaks across streams.
- **Window.** A steady-state window is picked from rank 0 only, by the
  concurrency in the annotations.

**Why we do not use it.** It works around what the trace lacks rather than
recording it: a second trace, a vLLM patch, and matching by name and batch
size.

**Limits on the DSV4.1 configuration:**
1. **Patches and annotations.** It needs a vLLM patch or new upstream flags
   and the annotations they add. Our pinned tree (`8f400a0ac`, v2 runner,
   breakable graphs, the dspark drafter) matches none of the patched versions.
2. **Pairing by position and name.** Breakable graphs launch many segments
   per forward, the drafter captures its own graphs (125 graphs on one rank,
   job 50303), and batches are padded to a captured size. "Counts equal,
   pair by position" will often skip iterations or pair them wrongly. Kernel
   names repeat in every layer, so matching by name leans entirely on order.
   ruppell joins each replayed kernel to its graph node by graph and node
   identity, refuses a replay that does not fit, and joined 100% on DSV4.1.
3. **Rank 0 only.** Capture comes from local rank 0 and the window is picked
   on rank 0. Rank skew and collective waiting (about 95% of all-reduce time
   in DSV4.1 first-token steps) are outside the agent's view.
4. **Unmeasured profiler cost.** The replay trace is the torch profiler
   during serving (shapes and Python stacks). Its effect on the server's
   operating point is not measured in their workflow. M3 found that tracing
   every kernel changes it (ITL 1.26–1.47× at C8), and that any tracer moves
   which rank waits.
5. **One window.** C32 at seed 42 has two regimes (first 300 s, then a
   queue), and one window describes one of them. Nothing checks the window
   against the rest of the run.
6. **No arch file for our GPUs.** Only MI300X, MI325X and Radeon 8060S are
   bundled. Without MI350 or MI355X there is no roofline.
7. **Hot kernels outside its models.** DSV4.1's hot kernels are AITER,
   Triton and HIP custom kernels (`_mxfp8_linear_kernel`, `mfma_moe*`,
   sparse MLA decode). Roofline and fusion models key on aten ops; these
   kernels land in the triton or generic categories.
8. **LLM-written ranking.** Two runs can rank findings differently. The
   perf-report CSVs underneath are deterministic, and those are what we use.

**What we use from TraceLens.** Its deterministic reports (the perf report
and the multi-rank collective report), on ruppell's files
([tracelens.md](tracelens.md)). In those files every replayed kernel already
sits under its op, with aten names and shapes when the CPU op tree is on. So
they need no capture folder and no vLLM patch.

## Hyperloom: the missed prefill profile (PRs #1575 and #1684)

**The incident.** Two paired AgentX sessions, same start. The profile had
captured no prefill steps. One arm wrote its own scripts to measure prefill,
found prefill sparse-indexer wins, and reached +39.7%. The other optimized
what its decode-only trace showed and reached +14.7%. The missing phase's
share had silently dropped out of the denominator the optimizer ranks by.

**#1575 (devalshahamd): move the profile window.** AgentX captures had
started as soon as aiperf entered its measured phase, so they sampled
ramp-up. The PR starts profiling 300 s later (`AGENTX_PROFILE_START_DELAY_S`),
stops after 256 engine steps or 100 s, and analyses a 64-step window.

**#1684 (Rita Brugarolas): make a missing phase a signal.**
- **The check.** The trace certificate gets check 8, `phase_coverage`: which
  serving phases (prefill, decode) the TraceLens split contains. A split with
  one phase is marked failed (advisory: the trace is not rejected), with an
  issue naming the missing phase.
- **The gap.** On an AgentX workload a missing prefill also becomes a
  high-severity `#phase_unmeasured:prefill` gap in the optimizer's ranked gap
  ledger. It resolves itself once a profile captures prefill.
- **Where it applies.** It is scoped to prefill misses on agentic runs, so a
  synthetic decode-only capture stays quiet. It comes with four unit tests.

**The good part.** The diagnosis is right, and it is the general lesson: a
phase the profile did not capture is unmeasured, not free. Ranking by
shares of what was captured optimizes the phase that happened to be measured.
The PR turns a silent failure into data that travels, both in the
certificate and in the gap ledger, instead of text an agent has to notice.
It is scoped so a legitimate decode-only capture does not raise it.

**The limitations:**
1. **It detects; it does not measure.** Presence is binary: a split holding
   one prefill iteration passes, even when prefill is far under-sampled for
   its share of time. The time the missing phase takes is asserted in the
   advisory text ("prefill dominates the engine"), not measured. The size of
   the hole, which is what ranking needs, is not computed.
2. **The file names it keys on do not look like the names the splitter
   writes.** It looks for `_extend_`/`extend_only_` (prefill) and
   `_decode_`/`decode_only_` (decode). That vocabulary comes from check 6,
   where such names mark the failure case (SGLang's per-stage profiling
   leaking through). A healthy split, as Hyperloom's own
   `agents/kernel/tools/trace_selfcert.py` parses it, is named
   `{mode}_steady_state_prefill_N_prefilldecode_N_decode_N_bs…_conc…`. That
   contains `_decode_` and never `_extend_`. As written, a healthy vLLM split
   reads as "prefill missing" whether or not the window holds prefill
   iterations. The per-kind iteration counts are already in the name and would
   be the signal to read. The PR's tests use made-up names
   (`steady_state_extend_0`); we did not run it on a real split.
3. **It checks the window, not the run.** Coverage is judged on the split of
   one profiled window. #1575's window (256 steps or 100 s, starting 300 s in)
   is short. At C8, 1.1–1.5 requests are in flight, and prefill steps come
   once per turn, so a window can hold few or none by chance. The check would
   then flag a miss the next capture might not have. Neither PR asks whether
   the window's mix of steps is the run's.
4. **One direction only.** A decode miss, or a regime miss (C32's first
   300 s), raises no gap.
5. **The fix is left to an agent** ("build a targeted per-component eval").
   The profile is not made complete.

**What ruppell does about the same failure.** Every engine step's fields
(kind, rows, running requests, context lengths) and its host begin and end
times are recorded, sampled or not. So coverage can be measured, not
guessed. `tests/m3/profile_report.py` reports, per regime and step kind, the
kind's share of engine time over every step and how many of its steps were
sampled. It lists a kind as undersampled when it has at least 5% of the
time and fewer than 20 sampled steps.

Our own exposure to this failure: detail sampling picks steps by count (5 in
every 1000). So a kind that is few steps but much time, like prefill on
agentic replays, can be under-sampled. The report now says so when it
happens. Sampling prefill and mixed steps at a higher rate is the remedy if
it does; it would need its own operating-point check.
