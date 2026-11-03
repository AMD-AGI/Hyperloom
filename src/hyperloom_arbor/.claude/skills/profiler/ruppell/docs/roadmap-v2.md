# Roadmap v2: closing the gaps to Nsight Systems

The milestones after [`roadmap.md`](roadmap.md)'s M0–M5, from the gaps
[`nsys-comparison.md`](nsys-comparison.md) found (2026-10-20). Each has exit criteria that can be
checked. Milestones are ordered by what they are worth to ruppell's use (LLM serving and training,
on AMD and NVIDIA); where one depends on another it says so.

**Every milestone holds to three rules:**
- **Both platforms.** It is built and verified on ROCm (MI350X, MI355X) and on CUDA (B200), or
  says why one is out of reach.
- **Checked against the vendor's collector.** On ROCm that is rocprofv3 in the same process; on
  CUDA it is Nsight Systems on separate runs. Anything new that ruppell records is compared with
  the same thing there, with the bar and the collector's own spread stated.
- **Cost measured.** A paired A/B against plain serving (throughput, TPOT, TTFT at C8 and C32),
  at full detail and sampled, before the feature is on by default or used for a claim.

The numbers in the exit criteria (5% here, 1% there) are proposals. The operator sets the gates
([`plan-v0.md`](plan-v0.md) and the runner's AGENTS.md).

| Milestone | What exists at the end | Gap it closes | Size |
| --- | --- | --- | --- |
| **M6** copies and fills | every memory copy and fill on the timeline, in graphs too | copy time counted as idle | small; **done** 2026-10-20 |
| **M7** NVTX and ROCTX | the program's own annotations as scopes, on replayed kernels too | annotations ignored | small; **done** 2026-10-20 |
| **M8** collectives | each collective kernel with its operation, size and communicator; bandwidth and stragglers per operation | collectives known only by name | medium; **done** 2026-10-21 |
| **M9** counters | hardware counters per kernel on sampled steps; device activity over the run | no hardware metrics | large; scoped 2026-10-21 as its own module, **gray-parrot** ([`gray-parrot/`](gray-parrot/README.md)) |
| **M10** CUDA breadth | more CUPTI versions and GPUs; hardware timestamps; the sampled TTFT cost understood | CUPTI 130001 only; +0.2–0.5 µs per kernel; C32 TTFT +24–31% | medium, partly waiting on NVIDIA |
| **M11** viewer and export | a viewer over whole runs; a columnar export | windows only; no tables outside Python | medium |
| **M12** collection control and many nodes | start and stop from outside the process; runs across nodes, aligned | no external control; one node only | medium |
| **M13** ruppell's own cost | a cost table per feature on both platforms; less of the tracer on the program's thread; faster analysis of full runs | full detail moves serving (−4% to −6% throughput); a ROCTX call costs about 4 µs; no cost table | medium |
| **M14** components | the adapter picks components (a layer of each kind, the router, the sampler) and ruppell profiles them in depth by default, the rest of the model at step level | the whole model or nothing; deep collection only in a separate rocprofv3 pass | medium |

**Not planned:** CPU sampling, OS runtime and thread states, network and storage metrics, and
library API traces. The last section says what to run instead.

## M6: copies and fills

> **Done** on 2026-10-20: `copies.bin` on both platforms, read by the engine, birdeye, the census,
> the Chrome export and `ruppell witness`.
> - **MI350, rocprofv3's memory-copy trace in the same process:** every copy matched with
>   identical times (the demo 4; Qwen3-8B serving 15,627; Qwen3-30B-A3B TP2 38,381 per rank).
>   100% and 99.998% of copy time scoped, all but 2 per rank in a step.
> - **B200, nsys on separate runs:** copy counts equal by kind (Qwen3-8B serving: 4,349 HtoD,
>   6,270 DtoH, 1,729 DtoD, 12 memsets). Bytes are equal except device-to-device, which follows
>   each run's batch sizes. 99.9998% of copy time scoped, 1,563 copies replayed from graphs.
> - **Cost** (Qwen3-8B TP1 served plain, then copies off and on at full detail; three rotated
>   rounds):
>   - **B200:** none measurable. Throughput with copies on against off was −0.00% at C8 and
>     +0.03% at C32, every round within 0.2%.
>   - **MI350:** not resolved. At C32, full detail alone halves throughput on this model, and
>     both ruppell arms vary by up to 62%. C8's TPOT with copies on against off is −0.64%, within
>     its spread. A sampled A/B is to settle it (runner `ruppell-accuracy/results/m6/`).
> - **Left open:** graph memory nodes on ROCm (none in these runs); ROCm copies issued by a graph
>   launch stay unscoped.

**Why:** a timeline that counts copy time as idle overstates idle and understates exposed copies,
and birdeye's memcpy columns are empty.
- **On CUDA** ruppell records no `cudaMemcpyAsync`, `cudaMemsetAsync`, or memory node in a graph
  (`memcpy-unrecorded`).
- **On ROCm** it records copy kernels (`__amd_rocclr_copyBuffer`) but not copies the DMA engines
  run.

**Build:**
- **CUDA:** CUPTI's memcpy and memset activity kinds. A copy or fill replayed from a graph carries
  its graph node, like a kernel, and joins the same way.
- **ROCm:** the profiler SDK's memory-copy buffer tracing (what `rocprofv3 --memory-copy-trace`
  uses). A copy made while capturing is already recorded in `capture.bin`; replayed ones join by
  position, as kernels do.
- **The records:** a copy record per operation, with direction, bytes, stream and device,
  correlation and graph node. Steps and scopes follow from the launching call, as for kernels.
- **The analysis:** birdeye's `gpu_timeline` (exposed and total memcpy), `kernel_summary` (as
  `memcpy` rows) and the census.

**Exit:**
- **ROCm:** on the W0 demo and a served model, every copy rocprofv3 records is in ruppell's
  records with identical times.
- **CUDA:** copy counts and bytes per step match nsys, and time by direction is within nsys's
  spread.
- **Graphs:** a captured graph with memory nodes joins whole on both platforms. Nodes come from
  `TODOs.md`'s "Memory nodes in a graph", never exercised so far.
- **The limitation goes:** `memcpy-unrecorded` is removed, and birdeye's timeline on W1/W2 shows
  the copies.

**Risks:** on ROCm, the SDK may not correlate DMA copies made by a graph replay to the replay. Then
replayed copies get a step by time, and the report says so.

## M7: NVTX and ROCTX as scopes

> **Done** on 2026-10-20: `ranges.bin` on both platforms, with `RUPPELL_NVTX=1` (off by default).
> The engine (report `nvtx`), birdeye and the Chrome export read it; the checks are
> `tests/m7/check_m7.py`. Records are in the runner's `ruppell-accuracy/results/m7/`.
> - **The demo:** every `emit_nvtx` op range matched the CPU op tree's op for the same kernel:
>   on B200, 39 of 39 eager and 2,400 of 2,400 replayed; on MI350X, 35 and 2,000.
> - **Against the vendor:**
>   - **rocprofv3's marker trace, in the same process:** 768 of 768 ranges paired. The median
>     difference is 0.33 µs; five events are 0.76–1.0 ms apart, a TODO in M13.
>   - **nsys, on a separate run:** the same 768 ranges. Its NVTX projection agrees in instances
>     and kernels for every eagerly launched range; ruppell's time reads longer by the
>     software-timestamp offset.
> - **vLLM:**
>   - its layerwise ranges, on the V1 model runner (the V2 runner has none): 558,805 on B200 and
>     556,770 on MI350X, covering 91% and 88% of kernel time;
>   - NCCL's ranges on its 3,057 kernels at TP2. Our image's RCCL emits none.
> - **ROCm needs a preload:** PyTorch calls ROCTX through roctracer's `libroctx64`, which never
>   reaches the profiler SDK, so `ruppell run` preloads the SDK's ROCTX library.
> - **Cost:**
>   - about 0.5 µs per NVTX call on B200, and about 4 µs per ROCTX call on MI350X;
>   - with vLLM's layerwise ranges, −9% and −31% of throughput (one run each).
>   - It stays off by default; the cost is M13's.
> - **Found:**
>   - **nsys does it the other way round:** its projection counts a replay under the range around
>     its launch, not under the ranges open at capture as ruppell does (`nvtx-capture-time`).
>   - **Start/end ranges** are recorded but not projected.
>   - **Most collectives** are vLLM's own and carry no range, which is why M8 was rewritten.

**Why:** vLLM, PyTorch (`emit_nvtx`) and NCCL already annotate their work with NVTX (on ROCm,
ROCTX). ruppell ignores those annotations, and on a framework it has no adapter for, it has
nothing else.

**Build:**
- **CUDA:** CUPTI's NVTX support (marker activity and the NVTX callback domain). It needs NVTX's
  injection to point at the tool, which `ruppell run` sets.
- **ROCm:** the profiler SDK's marker (ROCTX) tracing.
- **The mapping:** a push/pop range becomes a scope on its thread, as `ruppell.scope` does. Its
  domain and name are kept, and payloads are kept as attributes.
- **Where they sit:** ruppell's own scopes stay innermost; NVTX ranges sit outside them, so a
  module path never loses to a coarser range.
- **Replayed kernels:** ranges open while a graph is captured go onto the captured launches and
  reach replayed kernels through the existing join. nsys's documentation describes the same for
  `nvtx-live`; its projection report counted replays under the launch's range instead (above).
- **Filters:** by domain, to keep cost and noise down (`RUPPELL_NVTX_DOMAINS`).

**Exit:**
- **The demo:** a PyTorch demo with `emit_nvtx` gets aten op ranges on eager and replayed kernels
  alike, identical to the CPU op tree's ops for the same kernels.
- **vLLM:** a served model shows vLLM's NVTX ranges, if the build emits them, and NCCL's on its
  collectives.
- **Against nsys:** the same ranges as nsys's NVTX GPU projection (`nvtx_gpu_proj_sum`) on a
  separate run.
- **Cost** is measured with NVTX on and off.

**Risks:** on CUDA, NVTX allows one injection library. A program that already injects (nsys,
another tool) leaves ruppell without NVTX, and it says so, as it does for CUPTI.

## M8: collectives

> **Done** on 2026-10-21: the vLLM adapter's calls (on by default; `RUPPELL_COLLECTIVES=0` turns
> them off) and ruppell's NCCL profiler plugin (on by default since `8607248`; `RUPPELL_NCCL=0`),
> read by the engine (reports `collectives` and `nccl`), birdeye (`coll_by_op`) and the Chrome
> export. Records are in the runner's `ruppell-accuracy/results/m8/`, whose "Where M8 ends" takes
> the exit item by item.
> - **Coverage, served** (W2 at TP2, W2 at TP4 with expert parallelism, Qwen3-8B at TP8): every
>   collective kernel but 1–2 startup all-reduces per rank has its call, graph replays included.
>   The library's record names those.
> - **Sizes:** every call's bytes equal NCCL's or RCCL's record of it, on both platforms.
> - **The library's GPU timer:** paired as recorded, the residuals at TP2 are 1.3–9 µs (p1–p99),
>   against 0.25–1.5 s had the pairing been one call off; at most 172 µs at TP4 and TP8. Not at
>   8 ranks on RCCL, whose spans there are not the kernels'.
> - **busbw,** reported against nccl-tests and rccl-tests (runner steps 4 and 4b).
> - **Against the vendor:**
>   - counts equal Nsight Systems' per step (B200, separate run), and `rocprofv3 --rccl-trace`'s
>     outermost calls (MI350X, same process);
>   - on DLRMv4, rocprofv3's kernel trace matched ruppell's dispatches one for one (7 of 8 ranks;
>     rocprofv3 lost rank 0's file at teardown).
> - **Training:**
>   - **DDP at 8 ranks:** 672 of 672 collective kernels per rank with the library's record on
>     B200, and 650 of 672 on MI350X.
>   - **DLRMv4 on MI350X:** 2,923 of 3,136. Each paired record's function and datatype agree with
>     the PyTorch op that launched its kernel, and each all-reduce's count with its tensor.
>   - **The rest are RCCL's MSCCL kernels,** which RCCL reports nothing about.
> - **Cost:**
>   - **The plugin:** about what the library's own events cost, after `6534c67` and `d73acb6`.
>   - **The adapter:** 10–20 µs of host time per eager call, nothing measurable in the serving
>     loads, after `d096517`.
>   - **On training:** +0.15% on MI350X's DDP; not resolved on B200's DDP or on DLRMv4.
> - **Left open:** the runner's "Where M8 ends", and `TODOs.md`.

**Why:** a collective is known by its kernel name alone, and the only cost estimate is the
shortest duration across ranks per launch. Without its operation, bytes and communicator there is
no bandwidth, no busbw, and no per-operation straggler.

**What runs the collectives (M7's runs, 2026-10-20).** On Qwen3-30B-A3B served at TP2, rank 0,
most collective kernels are not the communication library's:

| Collective kernels | B200 | MI350X |
| --- | ---: | ---: |
| vLLM's custom all-reduce (`cross_device_reduce_1stage`), replayed from graphs | 153,066 (1,079 ms) | 102,626 |
| vLLM's custom all-reduce, eager | 97 | 679 |
| PyTorch's symmetric-memory all-reduce (`two_shot_all_reduce_kernel_inplace`) | 6,790 eager (767 ms), 194 replayed | none |
| NCCL or RCCL | 3,057 eager (399 ms) | 1,657 eager |
| With an NVTX or ROCTX range | the 3,057 NCCL ones | none |

- **The library sees little of it:** NCCL's calls are 2% of B200's collective kernels and 18% of
  their time, and RCCL's about 1.6% of MI350X's.
- **No ranges on AMD:** the RCCL in our image is built without ROCTX.
- **So the main source is the framework.** vLLM picks a backend per call (its custom
  all-reduce, PyTorch's symmetric memory, quick all-reduce on ROCm, or its own NCCL wrapper), and
  captures most of them into graphs. Only the framework sees each call with its tensor and group,
  including at capture.

**Build:**
- **The framework's calls, the main source:**
  - **What is recorded:** the vLLM adapter wraps the device communicator's collectives. Each
    call records the operation, dtype, element count, the group (ranks and size) and the backend
    vLLM chose, through the same mechanism as a scope.
  - **Replays:** a call made during capture marks the launches captured inside it. Replayed
    kernels carry it through the existing graph join, as they carry scopes.
  - **Training needs no adapter in M8** (operator, 2026-10-21). DLRMv4 captures no graphs, and
    every collective it runs goes through PyTorch's NCCL backend (RCCL on AMD), eagerly: DDP's
    gradient all-reduces, TorchRec's all-to-alls, the metrics all-gather, the loss all-reduce
    and the barriers. So the library's record covers them.
    - **What M8 does:** checks that, on traced training runs (Exit).
    - **Left to the DLRM plans:**
      - which DDP bucket each all-reduce carries, read from DDP's bucket layout after its rebuild
        and matched by order and size (not `register_comm_hook`, which replaces DDP's C++
        all-reduce with a Python one);
      - TorchRec's all-to-alls by sharding group, tables and split sizes, wrapped as the vLLM
        adapter wraps vLLM's communicators.
- **The libraries' calls,** for programs without an adapter and to check the adapter:
  - **NCCL:** its profiler-plugin interface (`NCCL_PROFILER_PLUGIN`, NCCL 2.23 and later),
    loaded by `ruppell run`. It gives the operation, count and datatype, the communicator and
    the rank.
  - **NCCL's NVTX ranges** (M7) are the fallback. ruppell keeps their payloads as raw bits, so
    they need decoding if they carry size and communicator.
  - **RCCL:** the profiler SDK's RCCL API tracing (`ROCPROFILER_CALLBACK_TRACING_RCCL_API`, in
    our image's headers), which gives each call with its arguments; or RCCL's own profiler
    plugin, if it has one.
- **Joining:**
  - an eager call joins to the kernels it launched by correlation id;
  - a captured call joins through the graph join;
  - matching by stream and order is the fallback only.
- **Analysis:** birdeye's `coll_analysis` gets operation, bytes, algorithmic and bus bandwidth
  from the transfer estimate, and start skew and wait against transfer per operation.

**Probe first, on both platforms:** what each source gives, before building.
1. **The adapter:** wrap vLLM's communicators on W2 and count the calls per backend, eager and
   captured, with their sizes. Does every collective kernel have a call?
2. **NCCL's profiler plugin** (NCCL's example plugin) on B200's W2: which events, which fields,
   and the cost.
3. **NCCL's NVTX payloads** in M7's records: their types, and whether size and communicator can
   be read.
4. **RCCL:** `rocprofv3 --rccl-trace` on MI350X's W2: the calls and their arguments; and whether
   this RCCL loads a profiler plugin.

> **Probe done** on 2026-10-20 (runner `ruppell-accuracy/results/m8_probe/`).
> 1. **The adapter: every collective kernel but 2 per rank has its call, replays included.**
>    - **Coverage:** vLLM's communicators and backends, wrapped and pushed as ruppell scopes,
>      named the call behind 163,300 of 163,302 collective kernels per rank on B200 and 155,724
>      of 155,726 on MI350X. That includes all 153,357 and 152,484 graph-replayed ones.
>    - **What each call gives:** dtype, element count and backend.
>    - **The misses:** two startup all-reduces per rank.
>    - **A design point:** a call also encloses copies and fills, so the metadata goes beside the
>      module scope.
> 2. **NCCL's plugin works** (NCCL 2.29.7, interface v5).
>    - **What it gives:** count, datatype, communicator, rank, algorithm, protocol, channels, and
>      per-channel GPU clocks; on its API events, the stream and whether the call was captured.
>    - **Its limits:** NCCL's calls only. The example writes at `ncclCommDestroy`, so ruppell
>      needs its own plugin that writes as it goes.
> 3. **NCCL's NVTX payloads are extension (schema) payloads,** pointers valid only during the
>    call. They are dropped as a source. ruppell now keeps their type (`ec2a343`).
> 4. **RCCL:**
>    - rocprofv3's `--rccl-trace` records every RCCL call but writes no arguments; whether the
>      SDK's callback carries them is to check.
>    - This RCCL also has a profiler-plugin interface (v1–v4).

**Exit:**
- **Coverage:** on W2 (TP2) on both platforms and on a TP4 or TP8 model, every collective kernel
  has its operation, dtype, bytes and group. That includes the custom all-reduces and the graph
  replays (today: none on MI350X, 2% on B200).
- **Correctness:**
  - sizes equal the tensors the calls were given;
  - each launch's transfer agrees with the library's own GPU timer for the same calls in the same
    run (NCCL's and RCCL's channel spans), where those spans are the kernel's (operator,
    2026-10-21). They are not at 8 ranks on RCCL, whose tasks carry other kernels' channel
    events: vLLM's all-gathers at TP8, and DDP's all-reduces, whose spans run about 8 times
    their kernels;
  - busbw per message size is reported against nccl-tests and rccl-tests on the same node, with
    the run's gap to the tests per call kind, and not gated.
    - **Why not gated:** the run's collectives are slower than the tests' by a roughly constant
      time per call, which the libraries' own timers show too, and below a few MB nccl-tests
      measures the host's launches (runner `ruppell-accuracy/results/m8`, steps 4 and 4b).
- **Against the vendor,** for the library calls:
  - counts by operation agree with `nsys -t nccl` (CUDA, a separate run);
  - and with `rocprofv3 --rccl-trace` (ROCm, the same process).
- **Training** (operator, 2026-10-21): on DLRMv4 at 8 ranks on MI350X, and on a DDP model on
  B200, every collective kernel has the library's record; its bytes agree with the call's;
  counts agree with `rocprofv3 --rccl-trace`; and the plugin's cost per training step is
  measured.
  - **What the libraries never report as a task** (DDP at 8 ranks, runner
    `ruppell-accuracy/results/m8`, step 12):
    - NCCL 2.29's broadcasts: it schedules them from its own queue, which its profiler's task
      events do not walk. It reports each call, and the engine takes the call as the record
      (`ec4abb7`), with no algorithm, channels or GPU span;
    - RCCL's MSCCL kernels, of which RCCL reports nothing over interface v4: DDP's 6.3 MB
      bucket, and every `ncclAllToAll` in DLRMv4 (213 of its 3,136 collective kernels per rank).
  - **DLRMv4** (Cluster A job 89583, runner step 12): RCCL's channel spans are not its kernels' there
    either (an all-reduce's runs 1,300–1,700 times its kernel at p50), so the timer cannot check
    the pairing. The PyTorch op behind each kernel checks it instead.
- **Cost:** an A/B with collective records on and off.

**Depends on:** M7 (ranges, for the NVTX fallback).

## M9: counters

> **Scoped** on 2026-10-21 as its own module, **gray-parrot**, off by default.
> - **Where:** [`gray-parrot/README.md`](gray-parrot/README.md) has its design and milestones
>   GP0–GP5. [`gray-parrot/survey.md`](gray-parrot/survey.md) has what Nsight Systems, Nsight
>   Compute, CUPTI, rocprofv3 and the ROCm profiler SDK offer, and what our MI350X image and
>   B200 have.
> - **What the survey changed below:**
>   - **ROCm serializes every kernel on the device** while a counting context runs, not only the
>     counted ones. Configuring counting may turn the Peak TOPS Limiter off for the process. So
>     nothing is configured unless counters are asked for.
>   - **The ROCm exit cannot be in the same process.** The SDK allows one counting context per
>     process, so "equal rocprofv3 `--pmc` in the same process" becomes separate runs of
>     deterministic work (GP0's probe P2 confirms).
>   - **The CUDA risk is real.** The Range Profiler beside activity tracing is undocumented, an
>     NVIDIA engineer reports lost kernel records, and Kineto removed its integration. GP0's
>     probe P3 decides it before GP2 is built.
>   - **Per-step counters, without serializing,** are added: the SDK's device counting service
>     and CUPTI's user ranges.
>   - **SPM is not the AMD counterpart of nsys's GPU metrics yet.** In our SDK it samples within
>     one dispatch; device-wide SPM arrives in ROCm 10.2.0.
> - The text below is the original sketch, kept as it was.

**Why:** ruppell says where GPU time goes but not why a kernel is slow ("No hardware counters").
On ROCm that takes a separate rocprofv3 counter pass today (`counters.md` in Arbor's profiler
skill). On CUDA there is no route at all.

**Build:**
- **Per kernel, on sampled steps:**
  - **ROCm:** the profiler SDK's dispatch counting service.
  - **CUDA:** CUPTI's range profiler.
  - **Which steps:** counter collection serializes and slows kernels, so counters run only on
    steps `RUPPELL_DETAIL` samples for counters (`RUPPELL_COUNTER_STEPS`). Those steps are marked
    and kept apart from timing steps; a step never has both.
  - **Counter sets:** per architecture, named in the records (bytes fetched and written, cache
    hit rates, occupancy, VALU and MFMA or tensor-core activity).
- **Device activity over the run:**
  - **ROCm:** the SDK's device counting service.
  - **CUDA:** CUPTI's PM sampling.
  - **Rate and use:** sampled at a fixed rate, the counterpart of nsys's GPU metrics, and placed
    on the step timeline by clock pairs.
- **birdeye:** a counters table per kernel and shape, and a roofline position where the GEMM
  table has FLOPs and bytes.

**Exit:**
- **ROCm:** counters on sampled steps equal rocprofv3 `--pmc` on the same kernels in the same
  process.
- **CUDA:** they are consistent with Nsight Compute on a kernel bench at the served shapes.
- **Timing steps are untouched:** their kernel durations match a counter-free run.
- **Cost:** at the default rate, device sampling costs under 1% of throughput.

**Risks:**
- Counters and kernel tracing may not run in one session on CUPTI 130001, or may need the
  profiler's exclusive device access.
- On ROCm some counter groups need several passes, so a sampled step can carry one group only.

## M10: CUDA breadth

**Why:**
- ruppell refuses any CUPTI other than API 130001, so it runs on CUDA 13.0 alone, tested on B200
  only.
- Its software timestamps read 0.2–0.5 µs longer per kernel than Blackwell's hardware ones.
- When sampled, C32's TTFT stays +24–31% against plain serving, from the tool, not the adapter
  ([`feasibility-cuda.md`](feasibility-cuda.md)).

**Build:**
- **CUPTI versions and GPUs:** support the CUPTI API versions the record layout can take, with
  the version in `meta.json`. Test H100 (CUDA 12.x) and GB200.
- **Hardware timestamps:** `cuptiActivityEnableHWTrace` behind a flag, retried on each newer
  CUPTI. It is the default once graph-replayed records arrive complete and periodic flushes do
  not hang (both failed on 130001).
- **The sampled TTFT cost:** find what the tool still does in an unsampled step. The candidates:
  - external-correlation records;
  - graph-launch callbacks;
  - re-enabling records at the next sampled step.

  Remove what is not needed there.

**Exit:**
- **Versions:** each supported version or GPU passes the C3 and C4 checks
  ([`milestones/cuda-c3.md`](milestones/cuda-c3.md), [`milestones/cuda-c4.md`](milestones/cuda-c4.md)).
- **Hardware timestamps:** with them, kernel time matches `nsys -t cuda` within its spread.
- **TTFT:** at 5/1000 sampling, C32's TTFT is within 5% of plain serving.

**Depends on:** NVIDIA's CUPTI releases, for hardware timestamps.

## M11: viewer and export

**Why:**
- A full run (37 M kernels per rank) is too large for a Chrome or Perfetto trace, so exploration
  happens in windows.
- Analysis outside ruppell's Python means reading its binary records.

**Build:**
- **The viewer:** merge the `viewer` branch (`593041c`), with its pieces:
  - the index: every kernel as columns, with its step, scope, op and graph;
  - tiles per pixel;
  - a timeline with per-rank tracks, a steps ruler, and a step explorer.

  Bring it up to `main`'s records: sampled runs, copies (M6) and collectives (M8).
- **The export:** `ruppell export --format parquet` of the kernel, copy, step and API tables, one
  file per table, joined by ids. pyarrow is optional and needed only for this, so birdeye stays
  numpy-only.

**Exit:**
- **The viewer** opens a full C32 run of DSV4.1 (four ranks) and stays interactive. Its numbers
  agree with the census for the same steps.
- **The export:** the Parquet tables reproduce birdeye's `kernel_summary` in DuckDB or pandas.

## M12: collection control and many nodes

**Why:**
- ruppell traces from launch to exit. Detail can be switched only from inside the process
  (`ruppell.detail()`, `RUPPELL_DETAIL`), so there is no start and stop from outside.
- Clock alignment across nodes is untested, so multi-node serving and training are out of reach.

**Build:**
- **External control:**
  - **The command:** `ruppell ctl RUN start|stop|detail K/N`. A control file in the run
    directory, read by each process's clock thread, switches detail at the next step boundary.
  - **Capture ranges** like nsys's, for programs without steps: start and stop on an NVTX or
    ROCTX range name (needs M7).
- **Many nodes:** clock pairs per node aligned through a cross-node reference. The candidates are
  PTP-disciplined realtime or a timestamped collective at startup. The error bound is stated in
  the export, as one node's 256 ns is now.

**Exit:**
- **Control:** start and stop from outside a running server change what is recorded within one
  step, with nothing lost or split.
- **Many nodes:** on a two-node run, step fingerprints agree across all ranks, and collective
  start skew across nodes is reported with its bound.

## M13: ruppell's own cost

**Why:** what ruppell costs decides whether it can stay on, and for a long run whether it is used
at all ([`validation.md`](validation.md) §5).
- **Full detail moves serving.**
  - DSV4.1 Flash on MI350X: −4.5% throughput at C8 and −6.2% at C32, ITL 1.08–1.47×.
  - Qwen3-8B on B200: −3.6% and −4.1%, with C32's TTFT +35%.
- **Most of the per-kernel cost is the vendor's.**
  - On ROCm it is the SDK's dispatch tracing: about 2 µs of host time per dispatch, and ITL is
    1.00× without dispatch records.
  - On CUDA it is CUPTI's kernel tracing: 0.7–1.5 µs per kernel, the same as nsys pays.
  - ruppell cannot remove that cost, only avoid it (detail sampling, already under 1% on B200).
- **ruppell's own part runs on the program's thread.**
  - The CPU op tree and the NVTX and ROCTX ranges write their records there, under a lock.
  - A ROCTX call (a push, a pop or a mark) costs about 4 µs (MI350X, M7), against about 0.5 µs for
    an NVTX call on B200. With vLLM's layerwise ranges that is −31% of throughput on MI350X and
    −9% on B200.
- **Nothing tracks the cost as features land.** M6's and M7's cost runs were one-offs.
- **Analysis of a full run** (37 M dispatches per rank) runs per-event Python loops: the step and
  scope sweeps, and the op tree's and the ranges' matching. Its time and memory are not measured.

**Build:**
- **A cost table, rerun per release.**
  - **What it runs:** the probe and demo loops per launch, and a serving A/B: plain, full detail
    and sampled, at C8 and C32.
  - **Its rows, per feature:** kernels, API records, copies, the CPU op tree, ranges, the adapter.
  - **Where:** both platforms, with each milestone's cost stage feeding it (`copies_ab.sh`, M7's
    `cost`).
  - It is kept in the docs beside `validation.md`.
- **Off the program's thread:**
  - per-thread buffers for op-tree and range records, drained by the clock thread;
  - names interned without a global lock;
  - on ROCm, the split of a ROCTX call's 4 µs between the SDK (two callbacks per call, enter and
    exit) and ruppell, and ruppell's share cut.
- **Less recorded per dispatch where the analysis does not need it.** API records and callbacks
  are the first candidates, with each one's use stated.
- **Analysis:**
  - the engine's sweeps in numpy without per-event loops;
  - the engine, census and birdeye streaming records instead of loading every array;
  - time and peak memory measured on a full DSV4.1 C32 run before and after.
- **TODO, timing against rocprofv3 (from M7):** on MI350X, five of 768 ROCTX range events read
  0.76–1.0 ms later in rocprofv3's marker trace than in ruppell's records, in the same process.
  - The median difference is 0.33 µs and the 99th percentile 4.6 µs; ruppell's timestamp is
    always the earlier.
  - It was not ruppell's buffer flushes: it is unchanged with 1 MB buffers.
  - Find which callback holds the thread in between (the SDK's ROCTX library, rocprofv3's
    callback, or ruppell's), and whether kernel and copy timestamps see the same.

**Exit (proposals):**
- **The cost table** exists for both platforms, per feature, and is rerun for each release.
- **Sampled serving** (5/1000) is within 1% of plain throughput and TPOT at C8 and C32, on both
  platforms.
- **An NVTX or ROCTX call** costs at most 1 µs on both platforms.
- **The program's thread:** no record write holds it more than 10 µs at the 99th percentile.
- **Analysis:** the engine, census and birdeye run on a full DSV4.1 C32 run within a time and
  memory the operator sets from the measured baseline.
- **Timing:** the ROCTX outliers against rocprofv3 are explained and fixed, or shown to be
  rocprofv3's.

**Depends on:** M10, for the sampled C32 TTFT cost on CUDA, which M10 owns.

## M14: components, profiled in depth by default

**Why:** ruppell has two depths today.
- **The whole model:** every kernel of every step, or of the steps detail sampling keeps.
- **Nothing deeper:** counters, PC sampling and thread traces mean a separate rocprofv3 pass
  over the whole program (M9 brings counters in, on sampled steps).

Most of a model is repeats: Qwen3-30B-A3B is 48 identical layers, and DSV4.1 Flash has a few
kinds of layer. One instance of each kind says most of what the whole model says.
- **Cost:** full detail costs serving 2–6% of throughput on most runs, and half of it on Qwen3-8B
  at C32 on MI350X.
- **Volume:** a full DSV4.1 C32 run is 37 M dispatches per rank.
- **Deep collection:** counters, PC sampling and thread traces slow every kernel they touch, so
  they are affordable on a few kernels only.

Which components are worth it depends on the model, so the adapter chooses (a proposal from
outside the team, 2026-10-20). The adapter knows the module tree, which layers are alike, and
which parts (attention, the MoE, the router, the sampler) matter for the workload.

**Build:**
- **The component plan, from the adapter.**
  - **What it names:** components by module path. The default is one instance of each layer kind
    (say the first dense layer, the first MoE layer and one in the middle), plus the router and
    the sampler.
  - **Overrides:** `RUPPELL_COMPONENTS` takes paths and patterns.
  - **The record:** the plan goes into `defs.tsv`, so the analysis knows what was chosen and how
    many instances each kind has.
- **What profiling a component means:**
  - **Timing:** every kernel of the component, on the steps detail sampling keeps.
  - **Deep collection,** on the component's kernels only: counters (M9), and on ROCm PC sampling.
  - **The rest of the model:** step-level timing (markers, step fields), and detail sampling as
    today.
- **How a component's kernels are selected,** by platform and launch mode:
  - **Eager launches:** the adapter's module hooks switch collection on and off around the
    component.
  - **Graph replays:** both SDKs record a replay whole. So selection happens after collection:
    the tool keeps the component's records and drops the rest in its buffer callback. It knows
    them by graph node on CUDA and by captured position on ROCm, from capture. That cuts records
    and analysis, not the SDK's per-dispatch cost.
  - **Counters:** the ROCm SDK's dispatch counting service decides per dispatch, so it can
    collect for the component's kernels only, replays included (to verify). On CUDA, the range
    profiler works by range, which suits eager launches; replays are to check.
- **A component report:**
  - **Per instance and per step kind:** kernels, time, and the spread over steps.
  - **Layer to layer,** where several instances of a kind are profiled.
  - **The model from its parts:** instance time × instances against the measured step time, with
    the gap stated.
  - **Counters** per component kernel (with M9).
- **The default:** component mode becomes the serving adapters' default once its exit holds;
  whole-model detail stays a switch.

**Exit (proposals):**
- **The plan:** the vLLM adapter plans components for DSV4.1 Flash and the Qwen3 models (dense
  and MoE) with no input from the user.
- **Fidelity:** a component's kernels and times match the same component in a full-detail run,
  within the run-to-run spread.
- **The model from its parts:** within 5% of the measured step time, or the report names what is
  missing.
- **Cost:**
  - records and analysis at most a tenth of full detail;
  - serving cost no more than detail sampling's.
- **Counters** (with M9) on the component's kernels match rocprofv3 `--pmc` on the same kernels.

**Risks:**
- **Graph replays:** the SDKs record a replay whole. In graph mode, components save volume and
  analysis, not tracing cost.
- **One instance may not stand for the rest.** MoE routing differs by layer and by step, and
  DSV4.1's sparse attention depends on the context. The plan keeps an occasional step at full
  detail to compare instances, and the report says when they differ.
- **Kernels shared across components:** fused ops at layer boundaries, and collectives, need a
  rule for which component owns them.

**Depends on:** the adapters' module scopes (exist); M9 for counters; M13's cost table.

## Not planned, and what to use instead

| Need | Use |
| --- | --- |
| CPU sampling, thread states, OS runtime, syscalls | nsys (`--sample`, `--cpuctxsw`, `-t osrt`), or rocprof-sys on ROCm, on a separate run |
| Python stacks and the GIL | nsys (`--python-sampling`, `-t python-gil`) or py-spy |
| Network, switch and storage metrics | nsys (`--nic-metrics`, `--storage-metrics`) |
| cuBLAS, cuDNN, MPI, UCX, OpenMP API traces | nsys; ruppell names their kernels |
| Unified-memory faults, GPU context switches | nsys |

Revisit an item when a workload needs it in the same run as ruppell's attribution, which a
separate run cannot give.

## Order

1. **M6 and M7 first,** in parallel: both are small, and both make existing numbers more complete.
2. **Then M8,** which builds on M7.
3. **M9 and M11** are independent of each other; M9 is the largest.
4. **M10** whenever NVIDIA ships a CUPTI that fixes hardware timestamps; its TTFT part can start
   now.
5. **M12** when a multi-node workload needs it.
6. **M13** alongside the rest:
   - its cost table now, fed by each milestone's cost runs;
   - its changes to the program's thread once M7 and M8 have settled what is recorded there.
7. **M14** after M8:
   - its plan, timing and report first;
   - its deep collection with M9.
