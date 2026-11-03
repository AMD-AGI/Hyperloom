# Limitations

What the traces and the census cannot be trusted for yet, and why. Items
that are work to do are also in [`TODOs.md`](../TODOs.md).

The ones a reader of an output needs travel with it
([`python/ruppell/limits.py`](../python/ruppell/limits.py)): every Chrome trace,
TraceLens file and census summary has `limitations` (id, what it applies to,
text) for its run; collective kernels carry `ruppell_limits:
["collective-wait"]`; `LIMITATIONS.md` is written beside the TraceLens files,
since TraceLens's own reports drop the metadata.

- **Per-rank collective time and rank skew are not the untraced server's.**
  A collective kernel's time is transfer plus waiting for the slowest rank,
  and tracing moves which rank is slowest: on DSV4.1 first-token steps
  about 95% of each rank's all-reduce time was waiting, and the rank that
  waited least changed from rank 3 untraced to rank 1 traced (0.05× to 23×
  per rank). The transfer estimate (each launch's shortest duration across
  ranks) agreed within 1%. Use it for collective cost; do not rank
  optimization sites by collective wait or by which rank straggles.
- **Full tracing changes the operating point; sample for serving.** Tracing
  every kernel costs the host about 2 µs per dispatch (the profiler SDK's
  dispatch tracing, about 3 ms per DSV4.1 decode step). At C8 that made ITL
  1.26–1.47× and raised requests in flight by a quarter, so the traced server
  ran a different load. With detail sampling (`RUPPELL_DETAIL=5/1000`) the
  operating point holds at C8, and at C32 all but in-flight (0.2% over the
  ±1% bound) ([`milestones/m3.md`](milestones/m3.md)). Use traces
  for where GPU time goes within a step; do not compare a traced run's
  latency or throughput with an untraced baseline. On CUDA (B200, Qwen3-8B
  at TP1), full tracing cost 4% of throughput, and 5/1000 sampling under 1%;
  C32's TTFT stays +24-31% even when sampled
  ([`feasibility-cuda.md`](feasibility-cuda.md)).
- **Long runs strain the analysis.** M3 traced full 600 s AgentX replays at
  C8 and C32 (jobs 49214 and 49215; each traced point's whole measured phase
  analysed by 49457 and 49458), so run length is no longer untested. The
  analysis holds a process's records in memory: 6.4 GB per rank per 18
  minutes on disk, and the census took about 2 minutes for four ranks of
  37 M dispatches. Longer or busier runs may need streaming by step range.
- **The vLLM adapter covers the v2 model runner with breakable graphs.** It
  has traced the pinned tree `8f400a0ac` (DSV4.1, ROCm), vLLM 0.30.1rc1 and
  0.28.0 (B200), and the 2026-10-13 ROCm build in `hy-model-w-10132026`
  (MI350, the runner's accuracy study), each with breakable CUDA graphs, so
  module hooks run. vLLM's compiled path (`-O2`) is covered since 0.9.0
  (`adapters/vllm/compiled.py`): each compiled piece, each launch site of its
  Inductor program and each split op gets its module's scope, checked against
  breakable runs by launches per module on Qwen3-8B, Qwen3-30B-A3B TP2 (B200,
  MI350X) and Model W TP8 (MI350X; runner ruppell-accuracy plan 06).
  A kernel no site places keeps the scope `compiled.piece`. It disables vLLM's
  compile cache (a hit loads no FX graphs): a second start runs Dynamo again,
  15-20 s more to ready on Qwen3 (MI350X), and since 0.9.1 keeps Triton's
  cache where Inductor's is, so vLLM's own Triton kernels are not compiled
  again. Hooks do not fire inside other
  `torch.compile` regions, whose kernels take the scope around the region;
  other vLLM versions may move the wrapped methods (the adapter checks their
  signatures and refuses rather than half-tracing).
- **Scopes name modules, not aten ops.** An op is the module's class and the
  shape is the first input's. TraceLens's FLOP and roofline models key on
  aten op names and argument shapes, so they do not apply: it reports time
  per scope and kernel, not TFLOP/s or bound classification. The CPU op tree
  (`RUPPELL_CPU_OPS`, [`cpu-ops.md`](cpu-ops.md)) records the aten ops. On
  DSV4.1 (ROCm), with run-time tuning pinned, a server traced with it ran the
  same GPU programs as an untraced one (job 50303); its kernel-time parity
  check (50342) is not yet recorded ([`milestones/m3.md`](milestones/m3.md)).
- **Hardware counters are off by default, and on CUDA they are per step.**
  gray-parrot (`RUPPELL_COUNTERS`, [`gray-parrot/`](gray-parrot/README.md))
  counts on chosen counter steps, which are left out of every time.
  - **ROCm:** per kernel.
  - **CUDA:** one value per step: CUPTI 130001's per-kernel mode skipped
    CUDA graph launches (runner plan 04, probe A3). Per-kernel numbers on
    CUDA, and multi-pass metrics, still need Nsight Compute or rocprof.
  - **CUDA's cost:** a counter step costs the program 60–120 ms of host time,
    mostly the Range Profiler's start, which grows with the program. At
    1/100 a served Qwen3-8B lost 10% of its throughput; the CUDA default is
    1/1000, whose cost three rounds could not resolve (runner plan 04, Part
    B). Groups rotate every 10 counter steps, so a set with several groups
    needs many counter steps to read them all.
  - **CUDA's device activity** (`RUPPELL_DEVICE_COUNTERS`, PM sampling):
    - **What it reads:** the whole GPU, every process's work, at the sample interval.
    - **Which steps get samples:** a step gets those whose midpoint lies in its GPU span, so only
      steps with kernel records (the detail sample) have any.
    - **Beside per-step counters:** it cannot run in the same process.
    - **Cost at 10 kHz:** about 0.65 of a host core on its thread, and 0.5–1.1% of a served
      Qwen3-8B's throughput.
    - **Against Nsight Systems' GPU metrics:** within a few percent, a little lower (runner
      plan 04, Part D).
- **On CUDA, durations are software timestamps.** Against Blackwell's
  hardware timestamps (`nsys -t cuda`) each kernel reads 0.2–0.5 µs longer:
  1.1–1.3× on 1–3 µs kernels, +2–6% of kernel time on serving and training.
  Against nsys with software timestamps (`-t cuda-sw`) it agrees within 0.2%.
  CUPTI 130001's hardware timestamps lost graph-replayed records, so ruppell
  stays on software ones (the runner's
  `ruppell-accuracy/results/b200/FINDINGS.md`; `limits.py`:
  `software-timestamps`).
- **NVTX and ROCTX ranges (M7) are off by default.** `RUPPELL_NVTX=1` records them, and their
  limits differ from scopes':
  - **Replayed kernels:** a replayed kernel takes the range open when its graph was captured,
    not the one around the replay. Nsight Systems' projection counts a replay under the range
    around its launch, so the two agree on eagerly launched kernels only
    (`nvtx-capture-time`).
  - **Start/end ranges and marks** are recorded but attribute no kernel. Ranges attribute
    kernels, not copies.
  - **Payloads** are kept as raw bits; NVTX extension payloads (NCCL's) are not decoded.
  - **Cost:** about 0.5 µs per NVTX call on B200, and about 4 µs per ROCTX call on MI350X. With
    vLLM's layerwise ranges that cost −9% and −31% of throughput (roadmap-v2 M13).
  - **ROCm:** PyTorch calls ROCTX through roctracer's `libroctx64`, which reaches the profiler
    SDK only with the SDK's ROCTX library preloaded. `ruppell run` preloads it, and a program
    started otherwise records no ranges.
  - **CUDA:** NVTX takes one injection library. Under another tool's injection (nsys) no range
    is recorded (`nvtx-not-injected`).
  - **What emits them:**
    - our image's RCCL emits no ROCTX;
    - vLLM's layerwise ranges exist only in its V1 model runner, in eager mode;
    - `emit_nvtx` names each op call apart (`op_id`), so `ranges.tsv` grows with every call.
- **Collectives (M8) come from vLLM's adapter, or from the library's own record.**
  - **The adapter is vLLM's only.** Elsewhere ruppell has the NCCL plugin's record, which sees
    NCCL's and RCCL's calls but no custom all-reduce or symmetric-memory kernel.
  - **The 1–2 all-reduces per rank made before `ruppell.init`** have no adapter call; the
    library's record names them.
  - **What the libraries leave out:**
    - **RCCL's MSCCL kernels** get no record: RCCL reports nothing for them over interface v4.
      On MI350X that was DDP's 6.3 MB bucket and DLRMv4's every `ncclAllToAll`.
    - **NCCL 2.29's broadcasts** have no task. Their record is the API call: function, count,
      datatype and root, with no algorithm, channels or GPU span.
  - **RCCL's kernels name no function,** so its tasks pair with kernels by order, checked by the
    library's GPU timer.
    - **At 8 ranks** RCCL attaches other kernels' channel events to a task, so its spans are not
      the kernel's and the timer checks nothing.
    - **Other checks** then do the job: the adapter's sizes, or the PyTorch op behind each kernel.
  - **Bandwidth is the transfer estimate's,** each aligned launch's shortest duration across
    ranks. The run's collective kernels are longer than nccl-tests' and rccl-tests' by a few µs
    per call, which the libraries' own timers show too. So busbw is reported against the tests,
    not gated.
  - **Cost:** the adapter adds 10–20 µs of host time per eager collective call, which the
    wrapper's own 2–3 µs does not explain. Graph replays pay none.
- **Synthetic CPU timing for replayed kernels in the TraceLens file.** A
  replay's one launch call is shared out among its kernels. CPU-side analysis
  of graph launches (launch overhead, host gaps) uses ruppell's own trace,
  where the HIP API calls are at their real times.
- **Assignment rules, not observations, in a few places.** Kernels on side
  streams are assigned to the step whose CPU window launched them; a kernel
  that starts in a step and ends a few µs after its end marker belongs to the
  step it started in; a replay whose dispatches the profiler SDK
  mis-correlated is repaired only when its kernel sequence then matches the
  graph exactly. All are counted in the report.
- **Platform.** The gates (M1 to M3) ran on MI350 (Cluster A), SDK 1.0.1 and
  1.3.2. MI355X (Cluster B, gfx950) has traced DeepSeek V4.1 Flash serving in
  the pinned image (ROCm 7.2.3) for the runner's Plan 31: C8 and C32, every
  mixed step traced and decode at 5/1000 (jobs 194612 and 194611,
  2026-10-05). On its four points, 156,340 steps agreed in kind and rows on
  all four ranks, and every traced step's collectives aligned but one (1 of
  509 at C8). The section 5 checks and the operating-point gate have not run
  on MI355X. ROCm 7.2 with torch 2.13 (`hy-model-w-10132026`) on MI350: in
  the runner's accuracy study, `rocprofv3` in the same process saw every
  dispatch ruppell recorded, with identical times, on the probe, the PyTorch
  demo, two served models and a training step (2026-10-20). Other ROCm
  versions are untested. CUDA (B200): see [`status.md`](status.md).
- **Stopping a served model under `rocprofv3` takes care.** rocprofv3 writes
  each process's files when that process ends, from its own SIGINT handler.
  A SIGINT to the API server alone ends the server first, and the engine
  and worker processes, which ran every kernel, are killed without writing.
  What works (the runner's `ruppell-accuracy/tools/serve_bench.sh`, MI350,
  2026-10-20):
  - signal each GPU process in turn (workers, then the engine core, then the
    server), with the engine core and the server stopped meanwhile;
  - have rocprofv3 write to local disk (on an NFS mount with `sync` it wrote
    about 300 KB/s and outlasted every wait);
  - kill a process once rocprofv3 has logged its finalization, since vLLM's
    own shutdown can then hang.

  Served that way, rocprofv3 matched every dispatch (1,063,928 at TP1;
  2,008,140 per rank at TP2). ruppell's own records do not depend on it.
- **A killed worker loses what it recorded after its last flush**, and its
  `dropped_records` does not count it. vLLM with `--shutdown-timeout 0` (its
  default) SIGKILLs the engine core's process tree on SIGINT, workers
  included, while they are in their own shutdown (B200, 2026-10-07: 3 and 6
  of 8 ranks lost their last ~140 steps). Serve traced with
  `--shutdown-timeout 60`; the adapter warns at startup when it is 0. The
  adapter marks a worker's records complete once its shutdown flush has
  returned; a process with engine steps whose records end at neither that
  nor the final flush is listed by the census (`incomplete_processes`,
  limitation `incomplete-records`) and fails `check_m2.py`'s "records
  complete". Runs traced before the mark existed read as incomplete:
  unproven, not necessarily short.
