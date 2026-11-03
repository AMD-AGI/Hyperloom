# TODOs

What `ruppell` does not do yet, and what it does but nobody has shown works.
State as of M2 (2026-09-29, Cluster A MI350; M1 on the host's ROCm 7.0.2, M2 in
the pinned vLLM image, ROCm 7.2.3, SDK 1.3.2). Records:
[`docs/milestones/m1.md`](docs/milestones/m1.md),
[`docs/milestones/m2.md`](docs/milestones/m2.md). Close an item by pointing at
the job or test that settles it.

## Unverified: built, not yet shown to work

### Capture and replay

- [ ] **`torch.compile` and piecewise CUDA graphs.** Forward hooks may not run
      inside compiled regions, which would leave captured launches without a
      scope (they would show as `graph:no-scope`). The served configuration
      does not compile (breakable graphs set compilation mode NONE), so M2
      did not need it; vLLM without breakable graphs would.
- [x] **Breakable CUDA graphs** (segments, eager breaks): 6,987 captures, 418
      replayed and joined per rank; eager breaks scoped (48490).
- [ ] **PyTorch CUDA-graph trees** (`torch.compile(mode="reduce-overhead")`),
      and graphs replayed on a different stream than they were captured on.
- [ ] **Graphs with parallel branches.** A capture that forks onto several
      streams may replay on several queues; ordering a replay's dispatches by
      dispatch id may then not follow the capture order. The join should
      refuse (sequence differs) rather than mismatch; neither is tested.
- [ ] **Memory nodes in a graph** (none in the served graphs so far) (`hipMemcpyAsync`, `hipMemsetAsync` while
      capturing). The join counts them when the dispatch count calls for it;
      never exercised on hardware.
- [ ] **Capture modes other than thread-local**, and capture from more than
      one thread at a time. Captured launches are recorded per thread.
- [ ] **Graph update** (`hipGraphExecUpdate`, kernel node parameter changes)
      after instantiation: the registry would still hold the old sequence.
- [ ] **Per-thread default stream variants** (`hipGraphLaunch_spt` and the
      other `_spt` calls): handled in code, not tested.

### Scale and platforms

- [x] **Long runs**, to the size of a short C8 replay: 37 M dispatches and
      67 M API records per rank over 18 minutes, 6.4 GB per rank, 0 dropped,
      3.8 s in the buffer callback (48490). A full-length replay (600 s and
      C32) is M3's.
- [ ] **Overhead on real serving.** 5.1 µs per dispatch on the probe in the
      container; the effect on paired per-request latency of a served model
      is M3's measurement. The tool is on the step path: the buffer callback
      runs on the SDK's thread, and the flush every 5 s on the clock thread.
- [ ] **Analysis at C32 and full length.** The census took 104 s and the
      windowed export 68 s for 4 × 37 M dispatches, in memory; a full-length
      C32 replay may need streaming. A full, unwindowed export is too slow at
      this size (788 s for a greedy-only run).
- [ ] **MI355X (Cluster B)**: traced in the pinned image for the runner's Plan
      31 (jobs 194611 and 194612, 2026-10-05; steps agree across ranks), but
      the section 5 checks and the operating-point gate have not run there.
      Other ROCm versions than 7.0.2.2 and 7.2.3, other SDK versions than
      1.0.1 and 1.3.2.
- [ ] **Scope attributes** from unusual inputs: `shape` is the first
      positional argument's shape only; keyword-only inputs, tuples and
      non-tensor inputs give `null`.

### Outputs

- [ ] **Looked at by a person** in ui.perfetto.dev and `chrome://tracing`. Both
      importers load the M1 traces without errors or warnings, but nobody has
      read them in the UIs.
- [x] **TraceLens** reading ruppell's traces: `python -m ruppell tracelens`
      writes PyTorch-profiler-shaped files per rank; TraceLens `e34b294`'s
      perf report and multi-rank collective report run on the M2 window
      (docs/analysis/dsv41-c8-tracelens.md).
- [x] **TraceLens perf models** (FLOPs, bytes, TFLOPS/s): with the CPU op tree
      (`RUPPELL_CPU_OPS=1`, `export --format torch`), shown on B200
      ([`docs/cpu-ops.md`](docs/cpu-ops.md)). Not yet: the CPU op tree on
      ROCm (gfx950), its `=names` mode's cost, and TraceLens's roofline
      percentages, which need a GPU arch file (none for B200 or MI355X).
- [ ] **TraceLens's inference report**
      (`TraceLens_generate_perf_report_pytorch_inference`) on ruppell's
      files: not tried.
- [x] **TraceLens on CUDA runs**: `ruppell export --format torch` (built from
      the records: scopes as the ops that launched each kernel, real CPU
      intervals for eager launches) on the C4 MoE run
      ([`docs/tracelens.md`](docs/tracelens.md)). TraceLens counts FlashInfer's
      all-reduce kernels as computation.
- [x] **One TraceLens exporter**: `export/torch_profiler.py` (from the
      records, eager CPU intervals, the CPU op tree) now also writes every
      rank (`ruppell tracelens RUN`), layer-folded op names
      (`--op-names path`) and engine steps as annotations; the Chrome-trace
      converter is gone. Not yet checked: TraceLens on the new files from a
      DSV4.1 run (m3_reanalyze writes them).
- [x] **Clock pairs**: the census places steps in the client's phases
      through them; processes on one node agree within 256 ns (48490).
- [ ] **Clock alignment across nodes** (multi-node serving): untested.
- [x] **birdeye against TraceLens**: on the B200 AgentX C8 and C32 windows,
      with TraceLens's collective names, the GPU timeline, kernel summary and
      ops summary equal TraceLens's CSVs on the same input
      (`tests/test_birdeye_offline.py`, [`docs/birdeye.md`](docs/birdeye.md)).
- [ ] **birdeye on a ruppell run directory of a served model** (tested on
      synthetic runs and on the Chrome traces of real ones), and its readers
      on real nsys, rocprofv3 and PyTorch-profiler files (synthetic only).
      Its GEMM table on a real CPU-op-tree run is not yet compared with
      TraceLens's `GEMM.csv` (the input of `tracelens-cpu-ops` is not kept).
- [ ] **birdeye on full-length runs**: the run is analysed whole in memory
      before `--steps` cuts it; streaming by step range would bound it.

## Not implemented

### Left from M2

- [x] **Kernels outside every step on DSV4.1** after the first engine step:
      vLLM's KV-zeroer warmup, named by 48796; 0 on every rank since
      `50deb19` (48834).
- [x] **One replay with four dispatches missing** (48834 rank 0): not
      missing; the SDK tagged 4 MoE kernels on the graph's second queue with
      a `hipEventRecordWithFlags` call's correlation id (48942). Repaired
      under a strict guard and reported since `014805d` (48966: 40,126 of
      40,126 replays joined on every rank).
- [ ] **Report the mis-correlation upstream** (ROCm profiler SDK): graph
      dispatches on a second queue can carry a concurrent call's
      correlation id.
- [x] **A decidable correctness check on DSV4.1**: outputs are not
      reproducible even untraced (prompt logprobs differ by up to 20 nats
      between passes, 48946), so M2.8 is decided by the GPU program: every
      first-token step identical traced vs untraced on all ranks (49062,
      `tests/m2/program_check.py`); output differences reported as
      distributions (p = 0.20, 0.78). The operator decides whether this
      stands for the plan's exact-greedy check.
- [ ] **DSV4.1's run-to-run nondeterminism** is the served stack's, not
      ruppell's; its size (20 nats at the second prompt position, cold cache)
      suggests state that differs between runs. For the DSV4.1 owners.
- [ ] **Drafter coverage in the census**: drafter captures are keyed and
      joined (4 graphs, one replay per step), but the census does not yet
      report target and drafter time apart.
- [ ] **The adapter on other vLLM versions**: it checks the signatures it
      wraps; besides the pin (`8f400a0ac`) it has traced vLLM 0.30.1rc1 and
      0.28.0 (B200) and the 2026-10-13 ROCm build (MI350, the runner's
      accuracy study). Others may move the wrapped methods.
- [x] **Serving under `rocprofv3`**: a SIGINT to the server alone ended it
      before vLLM stopped its engine, and the GPU processes died unwritten.
      Stopped process by process, with rocprofv3 writing to local disk and
      each process killed once rocprofv3 logged its finalization, rocprofv3
      matched every dispatch of two served models on MI350 (runner
      `ruppell-accuracy/tools/serve_bench.sh`, jobs 84380 and 84424;
      [`docs/limitations.md`](docs/limitations.md)). Worth an upstream note:
      rocprofv3 writes its files from inside its SIGINT handler, and the
      process did not always end afterwards (W1's engine core ran on for
      900 s after writing).
- [ ] **Shutdown at the ROCm pin**: vLLM 0.30 with `--shutdown-timeout 0`
      SIGKILLs its workers mid-shutdown and they lose their last records
      ([`docs/milestones/cuda-shutdown.md`](docs/milestones/cuda-shutdown.md)).
      Check the pin (`8f400a0ac`) does the same, and add
      `--shutdown-timeout` to `tests/m2` and `tests/m3`'s serve scripts; with
      the adapter's mark, `check_m2.py`'s "records complete" will say.

### For M3 (the plan's gates)

- [ ] **Out-of-process drain.** Records stay in the SDK buffer and are written
      by a thread in the traced process; the design calls for a sidecar so a
      long run never stalls the server on I/O.
- [ ] **Detail sampling** (`detail_sample_rate` is accepted and ignored):
      full kernel detail on a sample of steps, aggregates for the rest.
- [ ] **Perfetto protobuf output** for traces too large for JSON.
- [ ] **The tool's own intervals in the trace** (recorded in `self.bin`, not
      exported), so its perturbation is visible.
- [ ] **Paired per-request overhead A/B** on the served model (plan 4.7).

### Hardening

- [ ] **Refuse unknown platform versions** at load (AGENTS.md says an unknown
      ROCm version fails loudly; the tool does not check).
- [ ] **CI**: the offline test runs anywhere; a GPU job per change, recorded
      with the commit and library digest, is manual today (`tests/run_m1.sh`).
- [ ] **Packaging**: `pip install` covers the Python side only; the native
      libraries are built by `runtime/build.sh`. CMake comes with the CUDA
      backend.
- [ ] **Refusal paths tested on hardware**: a graph whose replays differ, an
      eager kernel with no scope, a marker without a code (covered offline
      only).

### Beyond v0 (M4)

- [x] **CUDA backend (CUPTI)**: built and verified on B200 (branch `cuda`;
      records [`docs/milestones/cuda-c3.md`](docs/milestones/cuda-c3.md),
      [`docs/milestones/cuda-c4.md`](docs/milestones/cuda-c4.md)). What is open on
      CUDA is below.
- [ ] SGLang adapter, to test that the layers really separate.
- [ ] Training schema and adapter.

### NVTX and ROCTX ranges (roadmap-v2 M7, 2026-10-20)

- [x] **Ranges on both platforms** (`ranges.bin`, `RUPPELL_NVTX=1`): checked against the CPU op
      tree, rocprofv3's marker trace and nsys's NVTX table and projection (`docs/validation.md`
      §7; runner `ruppell-accuracy/results/m7/`).
- [ ] **Project start/end ranges** onto kernels, as nsys does: kernels launched between the start
      and the end, on any thread or on the starting one.
- [ ] **Decode NVTX extension payloads** (NCCL's schema-based ones): ruppell keeps the raw bits.
      M8's probe found them to be pointers valid only during the call, and M8 takes what they
      carry from NCCL's profiler plugin instead.
- [ ] **The cost of a ROCTX call** (about 4 µs on MI350X, against 0.5 µs for NVTX on B200): split
      the SDK's share from ruppell's (roadmap-v2 M13).
- [ ] **Five of 768 ROCTX events 0.76–1.0 ms apart from rocprofv3's record** in the same process
      (roadmap-v2 M13, TODO).
- [ ] **Per-call names** (`emit_nvtx`'s `op_id`): one `ranges.tsv` line per call; group names
      by their text before `, ` in the report.
- [ ] **Graph launches inside a range:** the report could also count each replay under the range
      around its launch, as nsys's projection does, beside the capture-time attribution.

### Collectives (roadmap-v2 M8, 2026-10-21)

- [x] **Collective calls and the library's record on both platforms:** the vLLM adapter and
      ruppell's NCCL profiler plugin, both on by default. Checked against nccl-tests, rccl-tests,
      Nsight Systems and `rocprofv3 --rccl-trace`, and on DDP and DLRMv4 training
      (`docs/validation.md` §8; runner `ruppell-accuracy/results/m8/`).
- [ ] **RCCL's MSCCL kernels** get no record over interface v4 (DDP's 6.3 MB bucket; DLRMv4's
      `ncclAllToAll`, 213 per rank). Look for another source: RCCL's MSCCL hooks, or the SDK's
      RCCL API tracing with arguments.
- [ ] **The adapter's 10–20 µs per eager call,** beyond the wrapper's 2–3 µs: time the wrapper
      inside a served model (roadmap-v2 M13).
- [ ] **The plugin's cost on training,** not resolved on B200's DDP (rounds spread 4–7%) or on
      DLRMv4 (+0.5% from one clean round): more rounds.
- [ ] **Why the run's collective kernels are longer than the tests'** by a few µs per call (fresh
      output buffers, or isolated calls), and the 67 MB all-reduce launches on MI350X faster than
      any rccl-tests call.
- [ ] **DLRMv4:** what ended the witness run's rank 5 at teardown (rocprofv3 then lost rank 0's
      file); the 11 SendRecv launches per rank whose datatype reads `Unknown`; the all-to-alls'
      bytes, whose 16 parts carry one count.
- [ ] **Training's calls by meaning** (left to the DLRM plans): which DDP bucket each all-reduce
      carries, and TorchRec's all-to-alls by sharding group.
- [ ] **birdeye's start skew per call kind:** `coll_by_op` has the wait per call kind, but the
      start skew only `coll_analysis` gives, per kernel name. Both are `collective-wait`'s,
      untrusted when traced.

## On CUDA (branch `cuda`, B200)

Shown to work: capture, clone and replay joined by graph node id, including
a graph forked onto two streams (C3); vLLM's breakable graphs through the
unchanged adapter (C4). Not yet:

- [ ] **A witness in the same process.** CUPTI takes one subscriber on CUDA
      13.0 and driver r580, so nsys runs separately: exact for deterministic
      programs, the program check for served ones. Several subscribers need
      CUDA 13.3, driver r610 and an nsys using `cuptiSubscribe_v2`.
- [x] **Overhead on a served model**, measured in two runs: 35B MoE at TP2,
      concurrency 8: throughput −5.8%, TPOT +4.3%, TTFT +24.9% (medians of
      3 paired rounds; another run: −8.3%, +6.5%, +33.6%), all of it CUPTI's
      tracing (the adapter alone is within noise). Reducing it (fewer API
      kinds on served runs; hardware timestamps) is open, and so is the
      cost at C32 and on long runs (M3).
- [x] **Detail sampling on CUDA** (`RUPPELL_DETAIL=K/N`): "off" waits for the
      GPU, then switches kernel and API records off before the next step
      launches (`b74585f..82b1a50`). Sampled steps read like full detail; at
      5/1000 throughput and TPOT cost under 1% (B200, Qwen3-8B TP1;
      [`docs/feasibility-cuda.md`](docs/feasibility-cuda.md)).
- [ ] **C32's TTFT at 5/1000 on CUDA: +24-31% against plain** (full detail
      +35%), from the tool, not the adapter (+2%). What the tool still does
      in an unsampled step (external-correlation records, graph-launch
      callbacks, switching back on) is unmeasured. Also untested: sampling on
      a MoE at TP2 and on long runs.
- [x] **Memory operations** (`cudaMemcpyAsync`, `cudaMemsetAsync`, memcpy and
      memset graph nodes): recorded since M6 (`copies.bin`; docs/roadmap-v2.md).
      On B200 their counts equal nsys's by kind, and 1,563 replayed from graphs
      took their nodes' scopes.
- [ ] **Graph updates** (`cudaGraphExecUpdate`, `cudaGraphExec*SetParams`):
      recorded and refused with the reason; never exercised.
- [ ] **Device-launched graphs, conditional and child-graph nodes**: kernels
      from nodes the capture did not record are refused; never exercised.
- [ ] **Blackwell hardware timestamps** (`cuptiActivityEnableHWTrace`): lost
      graph kernel records and hung under periodic flushes on CUPTI 130001;
      unused until a CUPTI where they work.
- [ ] **Other CUDA versions and GPUs** (H100, GB200/GB300): the CUPTI API
      version is pinned to 130001; another one refuses to trace.
- [x] **ROCm tool's meta.json**: written to a temporary file and renamed, as
      the CUDA tool does (`b6e8e09`); on MI350 every process's `meta.json` was
      final and no temporary file was left (runner job 85169).
