# gray-parrot survey: hardware counters on NVIDIA and AMD

What Nsight Systems, Nsight Compute and CUPTI offer for hardware counters, what rocprofv3 and
the ROCm profiler SDK offer, and what our hardware and images have (2026-10-21). It is GP0's
survey ([`README.md`](README.md)).

**Sources:**
- **The vendors' documentation and source,** cited per section (Sources, at the end). NVIDIA:
  Nsight Systems 2026.5, Nsight Compute 2026.3.1 (and the 2025.3 CLI), CUPTI 2026.3.1. AMD:
  `ROCm/rocm-systems` at `rocm-7.2.1` and `develop`.
- **Our installs:**
  - **MI350X:** runner job 89916 on Cluster A, in the image of ruppell's MI350 records
    (`ruppell-accuracy/results/gray-parrot/survey/`).
  - **B200:** dgx-025's disk and driver. Its counter lists wait for the node (GP0's probe P6).

Points the documentation leaves open are marked *unconfirmed*; GP0's probes take the ones that
matter.

## 1. The short answer

| | NVIDIA | AMD |
| --- | --- | --- |
| **Per-kernel counters, a tool** | Nsight Compute: replays each kernel once per pass, serializes launches, by default locks clocks and flushes caches | `rocprofv3 --pmc`: one application run per pass, serializes kernels |
| **Per-kernel counters, an API** | CUPTI's Range Profiler: auto range makes each launch a range, graph kernel nodes included, and synchronizes after each one | the SDK's dispatch counting service: a callback before each dispatch chooses its counters, or none |
| **Counters over a step, kernels concurrent** | Range Profiler, user range | the SDK's device counting service, read when the tool asks |
| **Device activity over time** | Nsight Systems' GPU metrics (10 Hz–200 kHz, about 15 metrics, device-wide); CUPTI's PM sampling underneath (*unconfirmed*); NVML GPM (100 ms and slower) | nothing packaged in our image: poll the device counting service, as AMD's RDC does, or `amd-smi`'s activity (100 ms cache). Device-wide SPM arrives in ROCm 10.2.0, beta |
| **How many** | Blackwell's count to be listed (P6) | gfx950: 428 hardware counters and 202 derived (our image) |
| **One pass** | depends on counter domains; CUPTI reports the pass count of a configuration | per-block slots: SQ 8, SPI 6, TCC 4, TCP 4, TA, TD, GRBM, CPC, CPF 2 each; a configuration that exceeds them is refused |
| **Who else may count** | one counter client per GPU: nsys GPU metrics, Nsight Compute, DCGM profiling, CUPTI profiling or PM sampling | one started dispatch-counting context per process; across processes, a device lock |
| **With kernel tracing in the same process** | not documented; an NVIDIA engineer reports incomplete records and serialized kernels | yes: rocprofv3's own `--kernel-trace --pmc`; while counting, every kernel on the device is serialized |
| **CUDA or HIP graphs** | auto range profiles every kernel node; a user range takes a whole graph as one workload | the callback runs for every kernel packet, graph replays included; a hang on graphs (ROCM-24852) was fixed in May 2026 |
| **Permissions** | `NVreg_RestrictProfilingToAdminUsers=0`, or `CAP_SYS_ADMIN` or `CAP_PERFMON`; dgx-025 allows everyone | dispatch counting worked in ruppell's container with no added capability; the device counting service takes a device lock that wants `CAP_SYS_PERFMON` |
| **Deeper** | SASS metrics, PC sampling | PC sampling (beta; host trap and stochastic on gfx950), thread trace, SPM per dispatch (beta) |

## 2. NVIDIA

### 2.1 Nsight Systems: GPU metrics

- **Options:** `--gpu-metrics-devices=<ids>|all|cuda-visible|none` (default `none`),
  `--gpu-metrics-set=help|<alias>|file:<file>`, `--gpu-metrics-frequency` (10 Hz to 200 kHz,
  default 10 kHz).
- **Sets:** a General and a Compute Triage set per chip (`gh100`, `gh100-ct` on H100).
  Blackwell's are probably `gb10x` and `gb10x-ct` (*unconfirmed*; P6 lists them).
- **Metrics, device-wide:**
  - clocks (GPC, SYS);
  - GR Active, Dispatch Started;
  - compute warps in flight, unused warp slots on active and idle SMs;
  - SMs Active, SM Issue, Tensor Active;
  - DRAM read and write bandwidth;
  - PCIe read and write throughput, NVLink bytes received and sent, PCIe requests to BAR1.

  Each is a percentage of peak sustained. The rows are the same as CUPTI's PM sampling table,
  with the raw metric names (`sm__cycles_active.avg.pct_of_peak_sustained_elapsed`,
  `dramc__read_throughput…`, …).
- **Not per kernel or per process.** "It does not know which process or context is involved."
  Since 2025.1 a recipe sums samples per NVTX range or kernel by time overlap.
- **The rate it sustains** depends on the GPU's size, its load and the host's decoding. On
  H100, `gh100` failed at 120 kHz, and an NVLink-only set held 200 kHz. A buffer overflow
  shows as missing data.
- **Exclusive:** "only one tool that subscribes to these counters can be used at a time":
  Nsight Compute, DCGM (unless `dcgmi profile --pause`), an application using CUPTI's sampling
  ("CUPTI trace is okay"). For multi-rank jobs nsys advises one rank per node.
- **A known GB200 bug:** NVLink percentages read low (about 66% against 80% measured); the
  bytes per second are right.

### 2.2 Nsight Compute: per-kernel counters

- **Names:** `unit__counter.rollup.submetric`, for example
  `sm__throughput.avg.pct_of_peak_sustained_elapsed` or `dram__bytes_read.sum`. Rollups
  (`.sum`, `.avg`, `.min`, `.max`) are taken across a unit's instances. `sass__` metrics are
  software-instrumented and always need a pass of their own.
- **Sets and sections:**
  - `basic` is the default;
  - `full` includes every roofline section;
  - sections include SpeedOfLight, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, Occupancy,
    LaunchStats, SchedulerStats, WarpStateStats, InstructionStats, SourceCounters and
    PmSampling.

  How many passes a set takes is not documented; each report records it.
- **Replay:**
  - **`kernel` (the default):** saves all the memory a kernel can reach, and restores what it
    wrote before each later pass. A single-pass collection saves nothing.
  - **`application`:** reruns the program once per pass.
  - **`range`:** supports no CUDA graphs.
- **It changes what it measures:**
  - **Serialization:** kernel launches are serialized.
  - **Caches:** `--cache-control all` flushes them before every pass, by default.
  - **Clocks:** `--clock-control` locked them to base clocks by default up to 2025.3 (dgx-025
    has 2025.3.1), and boosts them since 2026.1.
  - **To compare with an untouched run:** `--clock-control none --cache-control none`.
- **Graphs:** `--graph-profiling node` (the default) profiles each kernel node like a kernel.
- **Attaching:** only to a process it launched (`--mode launch`). It filters by kernel name,
  launch count and NVTX range.

### 2.3 CUPTI: the APIs a tool builds on

- **Range Profiler** (`cupti_range_profiler.h`, CUDA 12.6 Update 2) with the **Profiler Host
  API** (`cupti_profiler_host.h`).
  - **Scope:** per CUDA context, Turing and later.
  - **On the host:**
    - choose metrics;
    - build a configuration image (once per chip, reusable);
    - read its pass count (`cuptiProfilerHostGetNumOfPasses`);
    - after collection, evaluate values per range.
  - **Auto range:** each kernel launch is a range, indexed from 0 and unnamed. A tool maps
    ranges to launches by recording the launch order in its callbacks. It synchronizes the
    context after each launch, which can hang kernels that wait on each other.
  - **User range:** push and pop around many kernels, which stay concurrent.
  - **Replay modes:**
    - **Kernel replay:** CUPTI saves and restores memory; auto range only.
    - **User replay:** the tool saves and restores, for example with the Checkpoint API.
    - **Application replay:** reruns the program.
    - **Single-pass metrics need no replay.**
  - **Graphs:** auto range with kernel replay profiles each kernel node. User and application
    replay with auto range do not support graphs. In user range a graph is one workload.
  - **Switching per step:** Enable, SetConfig, Start, Stop, Decode, Disable are per context, and
    only kernels between Start and Stop are profiled. Nothing documented limits doing it many
    times in a run; nothing shows it either (*unconfirmed*; probe P3).
- **The legacy Profiling API** (`cupti_profiler_target.h`) is deprecated in CUDA 13.0, and fails
  on Rubin. The Event and Metric APIs never supported Turing and later, and were dropped in 13.0.
- **What CUDA 13.0 has** (ruppell's CUPTI 130001 is 13.0 with CUPTI update 1):
  - the Range Profiler, the Host API, PM sampling, SASS metrics and PC sampling.
  - **Later releases add:**
    - device-level metrics mixed with context-level ones (13.1);
    - queries for single-pass metric sets (13.2);
    - several subscribers, for the Activity API only (13.3, driver r610);
    - clock and cache control in the Range Profiler (13.4).
- **PM sampling** (`cupti_pmsampling.h`, CUDA 12.6), the device time series:
  - **Scope:** one session per device; single-pass metrics only.
  - **Interval:** a time interval on GA10x and later (so B200), or a SYS-clock count. CUPTI
    states no minimum; Nsight Compute documents 1 µs and later.
  - **Buffers:** decode them repeatedly from a thread; discard the first sample.
  - **Timestamps:** the documentation calls them both GPU-based and CPU-based. Whether they share
    a clock with activity records is *unconfirmed* (probe P5).
- **Coexistence with activity tracing, the key risk.** The manual says nothing either way.
  - An NVIDIA engineer on the forum: the two "can be used together, but with some caveats".
    - Kernel records may be incomplete, "typically with zero timestamps — particularly in
      Kernel Replay mode".
    - Auto range serializes every launch, which defeats concurrent kernel tracing.
    - "Personally, I couldn't see any records when combining Activity API with Range
      Profiling."
  - **Kineto removed its Range Profiler integration** in June 2026 (pytorch#187204), so there is
    no maintained open-source user to learn from.
  - **Measured on B200** (GP0's CUDA probes, 2026-10-22; runner
    `ruppell-accuracy/results/gray-parrot/cuda-probes/`):
    - **Contrary to the release notes,** auto range in kernel-replay mode skips CUDA graph
      launches silently, on both APIs.
    - **User ranges coexist with activity tracing:** records complete, streams concurrent.
    - **PM sampling** is on the activity clock at 100 µs, and cannot share a process with the
      Range Profiler.
- **Exclusive and permissions:**
  - **One client:** a second counter client gets `CUPTI_ERROR_HARDWARE_BUSY`.
  - **The restriction:** since driver 418.43, every counter tool is restricted to administrators
    unless `NVreg_RestrictProfilingToAdminUsers=0`, or the process has `CAP_SYS_ADMIN` (or
    `CAP_PERFMON` from R565). Without either, `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`.
  - **Driver r610** adds per-scope capability nodes.
  - **Not supported:** MPS.
- **Cost:** no numbers anywhere. "Overhead of tracing i.e. activity APIs is much lesser than the
  metrics profiling APIs." Kernel replay costs with the memory a kernel writes.

### 2.4 Coarse device metrics: NVML and DCGM

- **NVML GPM** (Hopper and later): two samples at least 100 ms apart.
  - **What it gives:** SM utilization and occupancy, tensor utilization by kind, DRAM
    bandwidth, FP64/32/16 utilization, PCIe, NVLink and C2C bandwidth.
  - **Exclusive?** One paper found it does not conflict with ncu, nsys or DCGM on H100
    (*secondary*).
- **DCGM's profiling fields** (`DCGM_FI_PROF_*`): 100 ms and slower. They conflict with nsys and
  ncu. A cluster running `dcgm-exporter` with them holds the counters.

### 2.5 "Why is this kernel slow" on Blackwell

| Question | Metric |
| --- | --- |
| SM and memory throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`, `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes | `dram__bytes_read.sum`, `dram__bytes_write.sum` (`dram__sectors.sum` is the authoritative traffic) |
| L2 and L1 | `lts__t_sector_hit_rate.pct`; L1 global hit rate from `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` over `…_op_ld.sum` |
| Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active`, `launch__occupancy_limit_*` |
| Issue | `smsp__issue_active.avg.pct_of_peak_sustained_active`, `smsp__inst_executed.sum` |
| Tensor cores | `sm__pipe_tensor_cycles_active`; on SM 10.x, `sm__pipe_tc_cycles_active` (tcgen05's UTC*MMA); per datatype `sm__ops_path_tensor_*` (exact names *unconfirmed*, P6) |
| FLOPs, non-tensor | `smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma}_pred_on.sum` and the `h` and `d` forms: SASS-instrumented, a pass of their own |
| Stalls | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` |

A tcgen05 MMA counts as one warp instruction that drives all four schedulers. Low issue with a
busy tensor pipe is expected on Blackwell, not starvation.

## 3. AMD

### 3.1 rocprofv3

In our image, rocprofv3 1.3.2 (below).
- **Collecting:** `--pmc C1 C2 …` takes one pass; repeated `--pmc "…"` takes several, each pass
  one application run.
  - **What passes take:** a set that does not fit in one pass fails.
  - **Choosing kernels:** `--kernel-include-regex`, `--kernel-exclude-regex` and
    `--kernel-iteration-range` filter which kernels are counted.
  - **Input files:** they also take `pmc_groups` with a `pmc_group_interval`, switching group
    every N dispatches (since 7.0).
- **Listing and checking:**
  - `-L` lists counters, PC sampling and SPM configurations per GPU.
  - `rocprofv3-avail info` lists counters with their dimensions.
  - `rocprofv3-avail pmc-check` says whether counters fit together.
- **Output:** `counter_collection.csv` has one row per dispatch and counter, with the
  dispatch's ids, kernel, grid, registers and start and end times.
- **No device-level counting mode.** `--collection-period` and ROCTX pause and resume switch
  tracing, not counting. With `--pmc`, every kernel is serialized for the whole run (read from
  the 7.2.1 tool's source; *unconfirmed* by documentation).
- **Also:**
  - `--spm` (beta, SPM per dispatch, sampled every N shader-clock cycles);
  - PC sampling (beta);
  - `--att` (thread trace);
  - `--attach PID`.

### 3.2 The SDK's counting services

- **Dispatch counting** (`dispatch_counting_service.h`, marked experimental):
  - **The callback:** before each dispatch is queued, on the launching thread, a callback
    returns a counter configuration for it, or none.
  - **Records:** each counter's value per dimension instance, with the dispatch id. That id is
    the same sequence every service sees, so records join to the kernel-dispatch trace on it.
  - **Configurations:** per agent, immutable, built at tool init because building them is slow.
- **Serialization is device-wide while a counting context is started.**
  - The SDK serializes every dispatch on the agent: barrier packets with system-scope fences,
    released by a host handler when the previous kernel ends. Dispatches the callback skips are
    serialized too.
  - Stopping the context ends it. The cost is not quantified anywhere.
  - Kernels that depend on each other deadlock.
- **Configuring it has a side effect:** in 7.2.1's source, configuring the service turns the Peak
  TOPS Limiter (PTL) off on every GPU for the rest of the process. Whether PTL is on by default
  on MI350X and MI355X, and whether SDK 1.3.2 does the same, are *unconfirmed* (probe P1).
- **Device counting** (`device_counting_service.h`, experimental): device-wide counters, read
  when the tool calls `rocprofiler_sample_device_counting_service`.
  - **How a read works:** a read packet on a separate queue. It does not serialize the
    program's kernels.
  - **Who may count:** one context per agent at a time. Starting it takes the KFD device lock,
    which wants `CAP_SYS_PERFMON`; without it the SDK warns that counters "may be inaccurate".
  - **Contexts:** it cannot share a context with dispatch counting (`AGENT_DISPATCH_CONFLICT`).
  - **Sample rate:** no maximum is documented. AMD's sample polls every 50 ms, and RDC samples
    10 ms windows.
- **Coexistence:**
  - **With tracing:** dispatch counting coexists with kernel, copy and API tracing in one
    process, as rocprofv3 (separate contexts) and rocprof-sys (one context) both do.
  - **Starting and stopping:** contexts can be started and stopped at any time after init.
  - **One counter at a time:** only one dispatch-counting context may be started per process, a
    global check across tool libraries (`CONTEXT_CONFLICT`). So `rocprofv3 --pmc` and another
    counting tool cannot count in one process (probe P2 confirms).
  - **PC sampling** cannot run at the same time as counter collection.
- **Graphs:** the queue interceptor calls the counting callback for every kernel packet in a
  write, graph replays included: the path that gives ruppell its replayed kernels.
  - **The hang:** AMD fixed a "PMC hang on HIP graph" (ROCM-24852) in May 2026, after 7.2.1.
  - **Our image's SDK** (`2b22ab0`, 2026-07-09) does not have the fix's commits as ancestors by
    their SHAs; they may have landed under others (*unconfirmed*). Probe P1 tests graphs first.
- **Names changed:**
  - In 6.x: `…_dispatch_profile_counting_service`, `create_profile_config`.
  - In 7.0: `…_buffer_dispatch_counting_service`, `rocprofiler_create_counter_config`,
    `rocprofiler_counter_record_t`. Old names survive under `ROCPROFILER_SDK_BETA_COMPAT`.

### 3.3 The gfx950 counter list (our image)

From `rocprofv3 -L` in job 89916 (`gfx950_counters.tsv` in the runner):
- **Base counters:** 428 by block. SQ 139, SPI 81, TCC 67, TCP 65, CPC 28, TA 20, GRBM 10, TD 9,
  CPF 7, TCA 2. gfx942 has 265 in 7.2.1. Every one except GRBM is marked as supported by SPM.
- **Derived counters:** 202, each with its formula. None is SPM-capable.
- **Constants:** 59 agent properties (CU count, SIMDs, XCCs and so on).
- **Dimensions:** XCC (8 on MI350X), shader engine (4 per XCC), instance (16 TCC channels, 9 TCP
  instances per SE). A TCC counter returns 8 × 16 values per dispatch, so derived counters
  reduce them (`reduce(…, sum|max)`).
- **Per-block slots per pass** (aqlprofile; MI350 inherits MI300's): SQ 8, SPI 6, TCC 4, TCP 4,
  TCA 4, TA 2, TD 2, GRBM 2, CPC 2, CPF 2.
  - A derived counter takes the slots of all its base counters.
  - **What it explains:** `FETCH_SIZE` uses three TCC counters and `WRITE_SIZE` two. Five TCC
    counters against four slots is why the pair could not be collected together on gfx950
    (Arbor's `counters.md`, job 59733).
- **MFMA:**
  - `SQ_INSTS_VALU_MFMA_{F16,BF16,F32,F64,I8,F8,XF32,F6F4}` count instructions.
  - `SQ_INSTS_VALU_MFMA_MOPS_*` count operations divided by 512. Their eight precisions fill
    SQ's eight slots: one pass.
  - Also `SQ_VALU_MFMA_BUSY_CYCLES` and `SQ_VALU_MFMA_COEXEC_CYCLES`.
  - New on gfx950: F6F4, XF32, the VALU FLOP counters `SQ_INSTS_VALU_FLOPS_FP{16,32,64}`,
    `TCC_EA0_RDREQ_DRAM_32B` (HBM reads specifically) and the TCP latency counters.
- **Missing from the SDK's derived set for gfx950:**
  - **Absent:** `L2CacheHit` (compute `TCC_HIT / (TCC_HIT + TCC_MISS)`), `GPUBusy` (use
    `GPU_UTIL`), `Wavefronts`, `MemUnitBusy`.
  - **Incomplete:** `MfmaFlops` leaves out F8, F6F4, I8 and XF32.
- **Formulas move between SDK versions.**
  - In our image (SDK 1.3.2), `FETCH_SIZE` is
    `(TCC_EA0_RDREQ_128B_sum*128 + TCC_EA0_RDREQ_64B_sum*64 + TCC_EA0_RDREQ_32B_sum*32)/1024`.
  - In 7.2.1 (SDK 1.1.0) it is built from `TCC_BUBBLE`.
  - A record must carry the formula it used.

### 3.4 Deeper services

- **PC sampling** (beta), one context per process. Settings on gfx950:
  - **Host trap:** by time, from 1 µs.
  - **Stochastic:** by cycles, from 256, powers of 2; zero skid, with stall reason and
    instruction type.

  It does not serialize kernels. The newest documentation warns of a hardware freeze that needs
  a cold restart.
- **Thread trace** (`--att`):
  - **Scope:** one CU per SE; only traced kernels are serialized.
  - **gfx950:** 7.2.1's documentation lists MI200 and MI300 only; `develop` lists gfx950 in full.
- **SPM:** in our image's SDK, per dispatch only (samples within one kernel, serialized).
  Device-wide SPM, which does not serialize, arrives in ROCm 10.2.0, beta, basic counters only.

### 3.5 Other AMD tools

- **rocprof-compute** (formerly Omniperf): gfx950 panels from speed of light to L2 per channel,
  an empirical roofline with FP4 and FP6. It collects by application replay, one run per counter
  group.
- **rocprof-sys** (formerly Omnitrace):
  - **Device metrics:** samples `amd-smi` (GFX, UMC and MM busy, power, temperature, memory) at
    100 Hz by default.
  - **Hardware counters:** per dispatch only; its device-counting code is commented out in 7.2.1.
- **`amd-smi`:**
  - **Activity:** `gfx_activity`, `umc_activity`, `mm_activity` in percent.
  - **Per XCC:** `gfx_busy_inst`.
  - **Accumulators:** energy and a 10 ns firmware timestamp.
  - **Rate:** metrics are cached for 100 ms by default. The firmware's own refresh rate is
    *unconfirmed*.
  - In job 89916 it showed the eight GFX clocks (500–2200 MHz) and memory at 1900 MHz.

### 3.6 "Why is this kernel slow" on gfx950

Passes are slot arithmetic for each item alone; `rocprofv3-avail pmc-check` confirms them.

| Question | Counters | Passes alone |
| --- | --- | --- |
| Bytes from beyond L2 | `FETCH_SIZE` (KB; 3 TCC) | 1 |
| Bytes written | `WRITE_SIZE` (KB; 2 TCC) | 1; 2 with `FETCH_SIZE` |
| HBM bytes | `TCC_EA0_RDREQ_DRAM_32B` × 32, `TCC_EA0_WRREQ_WRITE_DRAM_32B` × 32 | 1 |
| L2 hit rate | `TCC_HIT`, `TCC_MISS` | 1 |
| vL1D hit rate | `TCP_CACHE_MISS` over `TCP_TOTAL_CACHE_ACCESSES` | 1 |
| LDS bank conflicts | `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_IDX_ACTIVE` | 1 |
| Occupancy | `OccupancyPercent` (`SQ_WAVE_CYCLES`); limiters `SPI_RA_*_FULL_CSN` | 1 |
| VALU and MFMA busy | `VALUBusy`, `VALUUtilization`, `MfmaUtil` | 1 |
| FLOPs by precision | the eight `SQ_INSTS_VALU_MFMA_MOPS_*` × 512; `SQ_INSTS_VALU_FLOPS_FP*` | 1 for the eight MOPS |
| Instruction mix | `SQ_WAVES`, `SQ_INSTS_{VALU,SALU,SMEM,VMEM_RD,VMEM_WR,LDS,MFMA,BRANCH}` | 2 |
| Stalls and latency | `SQ_WAIT_INST_ANY`, `VmemLatency`, `TA_*_STALLED_BY_TC_CYCLES`, `TCC_TAG_STALL`, `TCC_EA0_RDREQ_DRAM_CREDIT_STALL` | 1 per small group |
| GPU busy | `GPU_UTIL` (`GRBM_GUI_ACTIVE`, `GRBM_COUNT`) | 1 |

## 4. What our hardware and images have

### 4.1 MI350X (Cluster A, runner job 89916, node `<node>`)

- **The image:** `rocm/pytorch-private:hy-model-w-10132026` (`sha256:eebc0825…`), the image of
  ruppell's MI350 records, run with the same devices and options.
- **Its ROCm:**
  - ROCm 7.2.3; torch 2.13.0 (HIP 7.2.53211).
  - **The SDK is 1.3.2:** the image installs SDK 1.3.2 over ROCm's own 1.1.0, and every
    `librocprofiler-sdk*.so.1` points to 1.3.2 (runner `results/m7/mi350_probe.txt`).
  - rocprofv3 reports 1.3.2, git `2b22ab0`; aqlprofile 1.0.70203. So ruppell and rocprofv3 run
    on SDK 1.3.2, not on 7.2.1's 1.1.0 that the documentation describes.
- **The GPU:** gfx950: 256 CUs, 32 shader engines, 8 XCCs, wave 64, 32 waves per CU.
- **The counter smoke,** one pass per counter group on a bf16 GEMM and three element-wise
  kernels, in ruppell's container options (no privileged mode, no added capability). Every
  pass ran, and every value is what the kernel's arithmetic says:

  | Kernel | Counter | Expected | Counted |
  | --- | --- | --- | --- |
  | `x * 2.0`, 64 Mi fp32 | `FETCH_SIZE` / `WRITE_SIZE` (KB) | 262,144 / 262,144 | 262,170 / 262,144 |
  | bf16 to fp32, 16 Mi elements | `FETCH_SIZE` / `WRITE_SIZE` | 32,768 / 65,536 | 32,792.5 / 65,536 |
  | `abs`, 16 Mi fp32 | `FETCH_SIZE` / `WRITE_SIZE` | 65,536 / 65,536 | 65,558 / 65,536 |
  | GEMM 4096³ bf16 | `SQ_INSTS_VALU_MFMA_MOPS_BF16` × 512 | 2·4096³ = 137,438,953,472 | 137,438,953,472 |
  | GEMM | `WRITE_SIZE` | C, 32,768 | 32,768 |
  | GEMM | `FETCH_SIZE` | A and B at least 65,536 | 202,970 (3.1×) |
  | GEMM, `x * 2.0` | `SQ_WAVES` | 256 tiles × 4 = 1,024; 16 Mi threads / 64 = 262,144 | 1,024; 262,144 |

  - **Durations:** under counting, kernel durations were within a few percent of a kernel trace
    (the GEMM 110–112 µs against 105–112 µs).
  - **Why that is no cost measurement:** this program runs one stream and synchronizes, so its
    kernels were serialized anyway.

### 4.2 B200 (dgx-025)

- **The host:** driver 580.173.02, CUDA 13.0, Nsight Compute 2025.3.1, Nsight Systems 2025.3.2.
- **Permissions:** `RmProfilingAdminOnly: 0`, so counters need no capability. The M8 container
  runs with none.
- **ruppell's image:** `<fork>/vllm-cupti:0.28.0` is vLLM 0.28.0 plus `cuda-cupti-dev-13-0`
  13.0.85 (CUPTI API 130001).
- **Not yet listed** (the node was serving another session's measurement): Blackwell's metric
  count and sets, nsys's set names, and the pass counts of candidate groups (probe P6).

## 5. What this means for gray-parrot

Taken into [`README.md`](README.md):
1. **Off means not configured.**
   - **On ROCm:** configuring dispatch counting turns PTL off for the whole process, and a started
     counting context serializes every kernel on the device.
   - **On CUDA:** the Range Profiler holds the GPU's counters.
   - So with `RUPPELL_COUNTERS` unset, nothing is configured, and when it is set, the counting
     context is started only for counter steps.
2. **A counter step's timing is lost on both platforms.** ROCm serializes; CUDA synchronizes
   after each kernel and may drop the step's kernel records. Counter steps are kept out of
   timing.
3. **Single-pass groups only, rotated over counter steps.** gfx950's slots make the useful sets
   small: the eight MFMA precisions are one group; `FETCH_SIZE` with `WRITE_SIZE` is two.
4. **Per step without serializing:** ROCm's device counting service in its own context, and
   CUDA's user range. Both are cheaper than per-kernel counting and lose attribution to kernels.
5. **Over the run:** poll the device counting service on ROCm (SPM's device mode later), and
   PM sampling on CUDA (GPM as a fallback). Both need probing for rate, cost and clocks.
6. **The witness is a separate run on both platforms** for anything nondeterministic. The SDK
   allows one counting context per process, and CUPTI one counter client per GPU.
7. **Derived metrics are gray-parrot's own,** computed from raw counters with the formula
   recorded. The vendors' sets miss what we need (L2 hit and FP8/FP4 FLOPs on gfx950), and their
   formulas change between versions.
8. **Exclusive:** check once at start, and refuse with who holds the counters.

## Sources

**NVIDIA**
- Nsight Systems User Guide 2026.5: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>;
  release notes <https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html>.
- Nsight Compute Profiling Guide, CLI and release notes 2026.3.1:
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>,
  <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>,
  <https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html>; the 2025.3 CLI (clock
  control default):
  <https://archive.docs.nvidia.com/nsight-compute/2025.3/NsightComputeCli/index.html>; Compute
  Triage: <https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html>.
- CUPTI usage, release notes and API reference: <https://docs.nvidia.com/cupti/main/main.html>,
  <https://docs.nvidia.com/cupti/release-notes/release-notes.html>,
  <https://docs.nvidia.com/cupti/api/group__CUPTI__RANGE__PROFILER__API.html>,
  <https://docs.nvidia.com/cupti/api/group__CUPTI__PM__SAMPLING__API.html>.
- Counter permissions:
  <https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters>.
- NVIDIA forums:
  - Range Profiler with activity tracing:
    <https://forums.developer.nvidia.com/t/associating-kernel-names-with-range-id/308365>
  - GPU metrics rates on H100:
    <https://forums.developer.nvidia.com/t/nsight-systems-gpu-metrics-become-unaligned-or-empty-near-a-short-kernel-at-higher-sampling-frequencies/363103>
  - GB200 NVLink metric:
    <https://forums.developer.nvidia.com/t/nsight-systems-utilization-nvlink-metric-appears-incorrect-on-gb200/344106>
- NVML GPM: <https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlGpmFunctions.html>;
  DCGM: <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>.
- Kineto's Range Profiler removed: <https://github.com/pytorch/pytorch/pull/187204>.

**AMD** (`github.com/ROCm/rocm-systems`, `projects/rocprofiler-sdk` unless noted):
- rocprofv3: `source/bin/rocprofv3.py`, `source/docs/how-to/using-rocprofv3.rst`,
  `using-rocprofv3-avail.rst`, at `rocm-7.2.1`; the tool's contexts in
  `source/lib/rocprofiler-sdk-tool/tool.cpp`.
- Counting services:
  - `source/include/rocprofiler-sdk/{dispatch,device}_counting_service.h`;
  - `source/docs/api-reference/counter_collection_services.rst`;
  - serialization in `source/lib/rocprofiler-sdk/counters/{core,dispatch_handlers}.cpp` and
    `hsa/profile_serializer.cpp`;
  - PTL and the device lock in `counters/{controller,device_counting,ioctl}.cpp`;
  - the context conflict in `context/context.cpp`;
  - the graph path in `hsa/queue.cpp`.
- Counter definitions: `source/share/rocprofiler-sdk/counter_defs.yaml` (7.2.1; `config.yaml` on
  `develop`); block slots in `projects/aqlprofile/gfxip/gfx9/gfx9_block_info.h`.
- The graph hang fix: commits `e478e2c`, `67e698f`, `e9905bf` (2026-05-19).
- PC sampling, thread trace, SPM: `source/docs/how-to/using-pc-sampling.rst`,
  `using-thread-trace.rst`, `source/docs/api-reference/spm.rst` (`develop`); `CHANGELOG.md`.
- rocprof-compute, rocprof-sys, amd-smi: `projects/rocprofiler-compute`,
  `projects/rocprofiler-systems`, `projects/amdsmi` at `rocm-7.2.1`.
- MI350 counters: <https://rocm.docs.amd.com/en/docs-7.2.1/conceptual/gpu-arch/mi350-performance-counters.html>.

**Ours:** runner `ruppell-accuracy/results/gray-parrot/survey/` (job 89916), dgx-025 as listed
in §4.2.
