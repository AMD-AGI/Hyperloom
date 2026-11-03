# gray-parrot: hardware counters in ruppell

gray-parrot is ruppell's hardware-counter module: roadmap-v2's M9
([`../roadmap-v2.md`](../roadmap-v2.md)), scoped on 2026-10-21 as a module of its own with its
own milestones, because it is large. (Grey parrots are smart, and they count.)

**What it adds:** why a kernel is slow, beside where the time went.
- **Per kernel,** on the steps it samples: bytes moved to and from HBM, cache hit rates,
  occupancy, VALU and MFMA or tensor-core activity, FLOPs by precision.
- **Per step:** the same counters over a whole step, without serializing its kernels.
- **Over the run:** device activity sampled at a fixed rate, the counterpart of Nsight
  Systems' GPU metrics.

**Off by default, because it is heavy.**
- **Per-kernel counting serializes the GPU's kernels.**
  - **On ROCm,** every kernel on the device is serialized while counting is on, counted or not.
  - **On CUDA,** the context is also synchronized after every kernel ([`survey.md`](survey.md)
    §2.3, §3.2).
- **Configuring counting changes the device for the rest of the process.** In ROCm 7.2.1's SDK it
  turns the Peak TOPS Limiter off on every GPU (§3.2 there; GP0's probe P1 checks our SDK).
- **Only one client can count on a GPU at a time.** gray-parrot competes with Nsight Systems'
  GPU metrics, Nsight Compute, DCGM's profiling fields and `rocprofv3 --pmc`
  ([`survey.md`](survey.md) §1).
- **With `RUPPELL_COUNTERS` unset, gray-parrot creates nothing:**
  - no profiler SDK context and no CUPTI profiler object;
  - no thread and no record file.
  - ruppell's records and cost are those of a build without it, and `meta.json` says
    `"counters": "off"`.

What the vendors offer, and what our hardware and images have: [`survey.md`](survey.md).

## 1. How it works

This is the design GP0's probes confirm or change (§2). Where a probe decides a point, the
point names it.

### 1.1 Switches

| Variable | Values | Default |
| --- | --- | --- |
| `RUPPELL_COUNTERS` | a set (`memory`, `compute`, `occupancy`, `roofline`), several sets, or `raw:NAME,NAME` in the vendor's names | unset: off |
| `RUPPELL_COUNTER_STEPS` | `K/N`: the steps that carry counters, chosen by step id, so every rank picks the same ones | ROCm `1/100` (a proposal); CUDA `1/1000`, since a counter step costs 60–120 ms of host time there (GP2) |
| `RUPPELL_COUNTER_GROUP_STEPS` | CUDA: consecutive counter steps a group is kept for; a switch costs 15–55 ms more | `10` |
| `RUPPELL_COUNTER_SCOPE` | `kernel` (per kernel, serialized) or `step` (one reading per step, kernels concurrent) | `kernel` |
| `RUPPELL_COUNTER_KERNELS` | a regex on kernel names: count those kernels only | every kernel |
| `RUPPELL_DEVICE_COUNTERS` | `SET@RATE`, for example `activity@10kHz` (CUDA's rate, GP4) | unset: off |
| `RUPPELL_DEVICE_EVAL` | CUDA: `tool` (samples evaluated as the run goes) or `after` (raw counter data, `ruppell_pm_eval`) | `tool` |

A program without an adapter switches counter steps itself with `ruppell.counters(on)`, as it
does detail with `ruppell.detail(on)`.

### 1.2 Counter steps are not timing steps

- **Counting changes timing.** A counter step is marked in `steps.tsv`, with the group it
  collected.
  - Its kernels keep their identity, step, scope and graph join.
  - Their durations are left out of every timing table: the census, birdeye's time columns,
    the Chrome export's step totals.
  - Time comes from the timing steps of the same kernel and shape.
- **A step never has both.**
  - With `RUPPELL_DETAIL=K/N`, counter steps come from the steps outside the detail sample.
  - With full detail, the counter steps are taken out of timing.
- **Every rank counts the same step,** as detail sampling does. A rank that serialized its step
  alone would make the others wait in its collectives.
- **The step after a counter step can be disturbed.** vLLM schedules it while the counter step
  runs. GP1 checks for that (its exit, "timing steps untouched"), and if it is disturbed, the
  next step is left out of timing too.
- **On ROCm, counting leaves a cost behind** (probe A3c, 2026-10-22). After a process's first
  counter step, every launch costs the launching thread more for the rest of the process,
  counted or not.
  - **How much:** on the W0 demo, a graph launch took 41-43 µs on the host against 21 µs before,
    and each step's CPU window 30 % more; kernel durations moved 0.5-4 %. The step right after a
    counter step was no worse than the others.
  - **So:** a run's timing steps after its first counter step are not those of a run without
    counters. Compare a counting run with a counting run, and say so in the report (runner plan
    03, decision 1).

### 1.3 Groups, passes, rotation

- **Sets and groups:** a set is defined per architecture as one or more groups.
- **One pass per group.** Each group must be collectable in one pass, checked when the run
  starts:
  - **ROCm:** `rocprofiler_create_counter_config` refuses a group that needs more than one;
  - **CUDA:** `cuptiProfilerHostGetNumOfPasses` must return 1.
  - A set that does not fit is refused at start, with the counters that collided.
- **gray-parrot never replays a kernel.**
  - **What replay costs:** CUPTI's kernel replay saves and restores all the memory a kernel can
    write. In a serving step that is the KV cache, too much to copy and a change to the step.
  - **What runs instead:** a single-pass configuration needs no replay
    ([`survey.md`](survey.md) §2.3).
- **One group per counter step,** rotating over counter steps.
  - A kernel and shape gets a full set after as many counter steps as the set has groups.
  - The report gives the number of samples behind each counter.

### 1.4 Per platform

**ROCm (the profiler SDK, already ruppell's tool interface):**
- **Per kernel: the dispatch counting service, in a context of its own.** Dispatch tracing
  already has one (`ctx_disp`), so that detail can be switched per step.
  - **Started only for counter steps.** While the context runs, the SDK serializes every
    dispatch on the GPU, the ones it does not count included.
  - **Stopped only once the device has finished the step** (probe A2, 2026-10-22). Stopping it
    with counted kernels still queued hung the GPU; `ruppell.counters(False)` waits for the
    device first.
  - **The callback** runs before each dispatch is queued and returns the step's group, or no
    counters. So the kernel filter and M14's components save counting work, not serialization.
  - **Records** carry the dispatch id, the sequence every SDK service shares. They join to
    `dispatch.bin`, and through it to steps, scopes and graph replays. The buffered records do
    not carry the callback's user data, so the tool keeps each dispatch's group and step itself.
  - **The SDK's interceptor** calls the callback for every kernel packet, graph replays
    included. AMD fixed a hang of counters on HIP graphs in May 2026 (ROCM-24852). Our image's
    SDK (1.3.2) counted every replayed kernel of 1,000 counter steps without a hang (probe A1).
  - **Each counted kernel sees a cold L2** (probe A6). The W0 demo's counted `FETCH_SIZE` was its
    whole working set every step. So bytes fetched and hit rates from counter steps are
    cold-cache values, an upper bound on what the served kernel moves.
- **Per step: the device counting service,** read at the step's markers.
  - **Its own context:** it cannot share one with dispatch counting.
  - **No serialization:** it does not serialize the program's kernels.
  - **The device lock** wants `CAP_SYS_PERFMON`. Without it the SDK warns that counts may be
    inaccurate, but they matched the counts taken with it (probe A6).
  - **What a read gives:** the count since the context started; the tool records the difference
    from the previous read. A synchronous read takes about 90 µs, and each read seems to refill
    L2: `FETCH_SIZE` per step grew with the read rate (probe A6).
- **The Peak TOPS Limiter:** its state cannot be read in ruppell's MI350X container (`amd-smi`
  reports N/A; probe A3). Configured counting, never started, moved timing within the runs'
  spread.
- **Over the run:**
  - the device counting service, read by gray-parrot's thread at the rate asked;
  - `amd-smi`'s activity metrics as a coarse fallback.
- **Not used for the run's time series:** SPM.
  - **In our image's SDK (1.3.2)** it streams samples within one dispatch, not across the
    device.
  - **Device-wide SPM** arrives in ROCm 10.2.0, beta ([`survey.md`](survey.md) §3.4).

**CUDA (CUPTI, already ruppell's tool interface):**

> **Changed by GP0's CUDA probes** (2026-10-22, B200 dgx-029, CUPTI 130001, driver 580; runner
> `ruppell-accuracy/results/gray-parrot/cuda-probes/`).
> - **Per kernel is out.** In auto range, a counted step's CUDA graph launches silently do not
>   run, and the session records nothing after them. It holds for both CUPTI APIs, and for a
>   graph instantiated by PyTorch or with flags 0. On eager work, auto range costs about 18 ms
>   per kernel and serializes the streams.
> - **CUDA counts per step:** one user range per counter step.
>   - **Safe:** replays run, outputs are unchanged, streams stay concurrent, and ruppell's
>     records stay complete.
>   - **Accurate:** totals match per-kernel sums.
>   - **Cost:** about 27 ms of host time per counted step in ruppell's demo.
>   - **One extra kernel:** the profiler adds a `WaitNs` kernel per counted step.
> - **Per-kernel numbers on CUDA** come from a separate Nsight Compute run of a kernel bench
>   (runner plan 04 §10, decision 5).
> - **PM sampling** works at 100 µs, on the activity clock, evaluated after the run. It cannot
>   share a process with the Range Profiler.
>
> The design below is as it was scoped; plan 04's Part B builds the per-step path.

- **Per kernel: the Range Profiler,** with these settings:
  - **Mode:** auto range and kernel-replay mode, with a single-pass configuration, so nothing
    is replayed.
  - **Ranges:** every kernel launch is a range, graph kernel nodes included. Ranges are
    indexed in launch order and carry no kernel name. ruppell maps them to its kernel records
    through its launch callbacks and its graph-node records (probe P3).
  - **Synchronization:** auto range synchronizes the context after each kernel.
  - **Kernel records on a counter step** may come incomplete or with zero timestamps (an
    NVIDIA engineer, [`survey.md`](survey.md) §2.3). §1.2's rule already does without them.
- **Per step:** a user range around the step, which keeps kernels concurrent.
- **Over the run:**
  - CUPTI's PM sampling, on a time interval (GA10x and later, so B200);
  - NVML's GPM as a coarse fallback (at least 100 ms per sample).

### 1.5 Records

Settled on 2026-10-21, before either platform writes one. The contract, field by field:
[`records.md`](records.md).

| File | What it holds |
| --- | --- |
| `counters.bin` | `ruppell_counter_rec`, one per counted kernel and counter: a key (ROCm: the dispatch id; CUDA: the correlation id, the graph node and the range index), the agent, the step, the group, the counter, the value (double) |
| `counters.tsv` | each (group, counter): the vendor's name, the set, base or derived (with the vendor's formula), how dimensions (XCC, SE, instance) were reduced, the unit |
| `device_counters.bin` | `ruppell_device_counter_rec`, device-wide readings: the interval, the agent, the step for a step reading, the group, the counter, the value; placed on the step timeline by clock pairs |
| `meta.json` | `counters`: `off`, or the version, platform, collector, scope, sets and groups, counter steps, records and drops, a refusal's reason, the device readings |

Values are summed over XCCs, shader engines and instances by default. Per-XCC values are a
switch: on MI350X, 8 XCDs share one dispatch.

### 1.6 Exclusive, and refusing

- **One client counts on a GPU at a time.** Before collecting, gray-parrot checks that it holds
  the GPU's counters. On CUDA that is `CUPTI_ERROR_HARDWARE_BUSY`; on ROCm, the SDK's refusal.
  - **Who else may hold them:** nsys with GPU metrics, Nsight Compute, DCGM profiling
    (`dcgm-exporter`'s DCP fields), `rocprofv3 --pmc`, AMD's RDC, another process's gray-parrot.
  - **Holding them:** the check is made once, not per step. A run that cannot have them
    refuses at start and says who holds them, rather than collecting part of a run.
- **Permissions:**
  - **CUDA:** `NVreg_RestrictProfilingToAdminUsers=0` on the host, or `CAP_SYS_ADMIN` or
    `CAP_PERFMON` in the container. dgx-025 has the first, so no capability is needed there
    ([`survey.md`](survey.md) §4.2).
  - **ROCm:**
    - `rocprofv3 --pmc` worked in ruppell's MI350X container options (no privileged mode, no
      added capability; runner job 89916).
    - The device counting service's lock wants `CAP_SYS_PERFMON` (probe P5).
- **One counting context per process on ROCm.** A second one gets `CONTEXT_CONFLICT`. So
  rocprofv3 cannot count in the process gray-parrot counts in. Probe A5 (P2) confirmed it:
  - `rocprofv3 --pmc` around a ruppell run took the context, and gray-parrot's first counter
    step got `CONTEXT_CONFLICT`;
  - that first conflict now refuses counters for the run, and tracing goes on;
  - ruppell tracing under `rocprofv3 --pmc` sees the same dispatch ids, so separate runs are
    GP1's witness.

### 1.7 Analysis

- **The engine:** `report counters`, by kernel and shape, step kind and scope.
- **birdeye:** a `counters` table per kernel and shape: each counter's mean, minimum, maximum
  and number of samples, and the derived metrics.
  - **Derived metrics:** bytes, hit rates, occupancy, utilization, FLOPs by precision.
  - **How they are computed:** from the raw values, in Python, by formulas kept per
    architecture in `python/ruppell/gray_parrot/`, with the vendor's formula cited for each.
- **A roofline position:**
  - **Intensity:** from the counters (FLOPs over bytes), beside the GEMM table's model (2·M·N·K
    and the operands' bytes), with the gap stated.
  - **Rates:** achieved FLOP/s and bandwidth use the same kernel and shape's time from timing
    steps, never from counter steps.
- **The Chrome export:** device counters as counter tracks.

### 1.8 Where the code goes

| Path | What |
| --- | --- |
| `runtime/src/gray_parrot.cpp`, `gray_parrot_cuda.cpp` | collection, compiled into the two tool libraries; inert until switched on |
| `runtime/src/records.h` | the counter record layouts |
| `python/ruppell/gray_parrot/` | sets per architecture, readers, derived metrics, the report |
| `python/ruppell/birdeye/` | the `counters` table and the roofline position, read through `gray_parrot` |
| `tests/gp/` | the probes and the checks |
| runner `ruppell-accuracy/` | `tools/gp_*`, `results/gray-parrot/` |

## 2. Milestones

Each milestone holds to roadmap-v2's three rules: both platforms, checked against the vendor's
collector, cost measured. The numbers in the exits are proposals; the operator sets the gates.

| Milestone | What exists at the end | Size |
| --- | --- | --- |
| **GP0** survey and probes | the survey; answers to the probes that decide the design | small; survey **done** 2026-10-21 |
| **GP1** ROCm counters per kernel | counters on counter steps, every kernel, graph replays included | medium |
| **GP2** CUDA counters per kernel | the same through the Range Profiler | large, the riskiest |
| **GP3** sets, derived metrics, birdeye | portable sets per architecture; the `counters` table and the roofline position | medium |
| **GP4** device activity over the run | a time series on the step timeline, both platforms | medium |
| **GP5** components | counters on M14's components only | small, after M14 |

### GP0: survey and probes

**Survey:** done, [`survey.md`](survey.md).

**Probes,** each a small program run on both platforms where it applies. Each probe answers a
question the design above depends on.
- **P1, ROCm per kernel:** the dispatch counting service in ruppell's tool, beside its dispatch,
  copy and API tracing.
  - Does the callback run for kernels replayed from a HIP graph, and do their records carry
    the dispatch id?
  - Can its context be started and stopped a thousand times in a run?
  - What does a counter step cost on a decode-like loop: host time per dispatch, and the
    kernels' serialization?
- **P1, also:** is the Peak TOPS Limiter on by default on MI350X and MI355X? Does configuring
  dispatch counting in our SDK turn it off, and does that move kernel times in timing steps?
- **P2, the ROCm witness:** can `rocprofv3 --pmc` and gray-parrot count in one process?
  - **Expected no:** the SDK allows one started counting context per process.
  - **Then** GP1's witness is a separate run of deterministic work, as on CUDA.
- **P3, CUDA per kernel:** the Range Profiler in ruppell's CUPTI subscriber, with activity
  tracing on, on CUPTI 130001. Settings: auto range, kernel-replay mode, a single-pass
  configuration.
  - Does every graph kernel node get a range?
  - Does the range index map to ruppell's kernel records?
  - Can it be enabled and disabled a thousand times?
  - What happens to the counter step's activity records?
  - Do counters work in our container without capabilities?
- **P4, CUDA per step:** does a user range around a whole step keep kernels concurrent, and
  does it read the same totals as the sum of P3's per-kernel values?
- **P5, device sampling:**
  - **ROCm:** the device counting service, its fastest rate and its cost.
  - **CUDA:** PM sampling in the same process as activity tracing, and the clock its
    timestamps use (the documentation contradicts itself); NVML GPM.
- **P6, the B200's lists:**
  - `ncu --query-metrics --chips gb100` and `ncu --list-sets`;
  - `nsys profile --gpu-metrics-set=help`;
  - the pass count of each candidate group under CUPTI 130001, which lacks the single-pass
    set queries that CUDA 13.2 added.

**Exit:** each probe answered with its record, in the runner's
`ruppell-accuracy/results/gray-parrot/`; this design confirmed or changed, with the date.

**ROCm's probes (P1, P2, P5), answered on 2026-10-22** on MI350X (runner plan 03, Part A; record:
`ruppell-accuracy/results/gray-parrot/probes/`).
- **P1:**
  - graph replays are counted with their dispatch ids, without a hang;
  - 1,000 starts and stops run clean once each stop waits for the device; without the wait, the
    GPU hangs;
  - a counter step costs 5-6 timing steps on the demo, and once counting has started, every later
    launch costs more (§1.2);
  - PTL's state cannot be read in the container.
- **P2:** no counting in one process with rocprofv3; the same dispatch ids; separate runs agree
  on every deterministic counter.
- **P5:**
  - the device service works beside tracing, with or without `CAP_SYS_PERFMON`;
  - a read takes about 90 µs and counts from the context's start;
  - deterministic totals agree with the dispatch counts to 0.1 %;
  - cache-dependent ones depend on the read rate.
- **The design changed** in §1.2 and §1.4: the stop drains, counting leaves a cost behind, and
  counted kernels see a cold L2.

### GP1: ROCm counters per kernel

**Build:**
- the counting context and its callback;
- the switches in §1.1;
- `counters.bin` and `counters.tsv`, read by the engine;
- counter steps marked and kept out of timing;
- groups rotated over counter steps.

**Exit:**
- **Against rocprofv3** (`--pmc`, in the same process if P2 allows, otherwise on separate runs),
  on the W0 demo and a kernel bench at served shapes:
  - deterministic counters (waves, instructions, MFMA operations, bytes written) are equal;
  - cache-dependent counters (bytes fetched, hit counts) lie within rocprofv3's own spread
    across runs.
- **Coverage:** on a served model with graph replays (W1, Qwen3-8B), every kernel of a counter
  step has its counters, replays included, or the report names the kernels without them.
- **Timing steps untouched:** kernel durations in timing steps match a counter-free run within
  the run-to-run spread, the step after each counter step included.
- **Off is off:** with `RUPPELL_COUNTERS` unset, the records equal a build without gray-parrot,
  and the cost table shows no difference.
- **Cost:**
  - one counter step's time against a timing step's;
  - throughput, TPOT and TTFT at C8 and C32 at the default rate, against counters off.

**Risks:**
- **Experimental:** the SDK marks dispatch counting experimental in our image (ROCm 7.2.3, SDK
  1.3.2).
- **Graphs:** counters on HIP graphs hung until AMD's May 2026 fix (ROCM-24852). If our SDK
  lacks it, GP1 waits for an image that has it, or counts eager steps only and says so.
- **The Peak TOPS Limiter:** if configuring counting turns it off and it was on, timing steps of
  a run with counters configured are not those of a run without. Then the comparison run
  configures counting too, and the report says PTL was off.
- **Several passes:** some groups need more than one.
  - **The example:** `FETCH_SIZE` with `WRITE_SIZE` is five TCC counters against four slots, and
    it hung rocprofv3 on gfx950 (Arbor's `counters.md`).
  - **The guard:** the start check refuses such a group before any kernel runs.

### GP2: CUDA counters per kernel

> **Changed** (2026-10-22, GP0's CUDA probes): per kernel is not possible on CUPTI 130001 with
> CUDA graphs (§1.4's note). GP2 becomes **CUDA counters per step**.
> - **The range:** one user range around each counter step, single-pass groups rotated as in
>   §1.3.
> - **Per-kernel numbers:** from Nsight Compute on a kernel bench.
> - **The exits:** the same, with "every kernel of a counter step" read as "every counter step".
>   Against Nsight Compute, a step's totals are compared with the sum of its kernels' values.
> - **Later:** retested on each newer CUPTI (M10), by the probe `graphx`.

> **Done on B200** (2026-10-22, dgx-029; the runner's `ruppell-accuracy/results/gray-parrot/gp2/`).
> - **Coverage:** every counter step had its reading, with graph replays inside: the demo 30 of
>   30, Qwen3-8B served 32 of 32, Qwen3-30B-A3B at TP2 32 of 32 on each rank (the same steps), a
>   training step 6 of 6. ruppell's records stayed complete.
> - **Against Nsight Compute,** on the demo: warps launched equal; instructions 3.6% more in
>   every step, not explained yet. Not run: a kernel bench at served shapes.
> - **Timing steps untouched, off is off, outputs** within the program's own run-to-run spread.
> - **Cost:** 60–120 ms of host time per counter step, mostly the Range Profiler's start, which
>   grows with the program. At 1/100 a served Qwen3-8B lost 10% of its throughput, so the CUDA
>   default is 1/1000.
> - **Found:**
>   - **Switching groups:** the Range Profiler is disabled and enabled again around a switch,
>     or the new group reads NaN.
>   - **Units:** tcgen05's tensor-op counters count the operations the tensor cores execute: 2·M·N·K
>     on plain GEMMs, more where a kernel pads (the demo's fc2 kernel pads M = 16 to 64 rows: 4×).
>   - **L2:** lookup misses are not DRAM traffic. On Qwen3-8B they came to twice the bytes
>     read from DRAM; B200's second die's L2 is the likely reason, not checked.

**Build:**
- **Configuration:** the Range Profiler (auto range, kernel-replay mode, single-pass groups). One
  config image per group, built when the run starts.
- **Switching:** profiling is enabled at a counter step's start and disabled at its end, after
  the drain detail sampling already does (`ruppell_detail_on_stream`).
- **Mapping:** range index to kernel, through launch callbacks and graph-node records.

**Exit:**
- **Against Nsight Compute,** on a kernel bench at served shapes and on the W0 demo.
  - **Settings:** `--clock-control none --cache-control none --graph-profiling node`, so that
    ncu neither locks clocks nor flushes caches. Its default clock control changed between
    2025.3 and 2026.1 ([`survey.md`](survey.md) §2.2).
  - **Deterministic metrics** (instructions, tensor operations, bytes a streaming kernel
    writes) are equal.
  - **The rest** lie within ncu's spread.
- **Coverage, timing steps untouched, off is off, cost:** as for GP1, on B200.

**Risks:**
- **No working reference:** Kineto removed its Range Profiler integration in June 2026, after it
  saw little use.
- **Activity tracing and the Range Profiler in one process are not documented.** An NVIDIA
  engineer reported incomplete kernel records, and none at all in one case.
- **Auto range synchronizes after each kernel:** kernels that wait on each other across streams
  can hang.
- **If P3 fails:** per-kernel counters on CUDA come from user ranges around chosen eager
  launches only, with replays covered per step (§1.4). The report says so.

### GP3: sets, derived metrics, birdeye

> **sm_100, 2026-10-22** (B200 dgx-025; the runner's `ruppell-accuracy/results/gray-parrot/gp3/`).
> - **The sets** are one pass per group (`sets/sm_100.yaml`).
>   - **Merged:** DRAM read and write with L2 hits.
>   - **Added:** L1's global-load requests, a group of their own.
>   - **Missing:** the occupancy limits (`launch__*`) are launch attributes, not Range Profiler
>     metrics.
> - **Tensor FLOPs:** tcgen05's bf16 and fp16 counters read exactly 2·M·N·K on 9 of 10 GEMM
>   shapes (one +3.1%).
>   - **Padding is counted:** the demo's fc2 kernel pads M = 16 to 64 rows and reads 4×
>     (Nsight Compute per kernel).
>   - **On served steps:** Qwen3-8B reads 1.09–1.13× a step's model FLOPs; Qwen3-30B-A3B 1.7–9.7×
>     (the MoE kernels' padding).
> - **Bytes:** a Qwen3-8B decode step's DRAM read equals its weights and KV cache within 0.5%.
> - **Not built yet:** the derived metrics (`gray_parrot/derived.py`), `ruppell counters list`,
>   birdeye's table. They are shared with ROCm (plan 03).

**Build:**
- **Portable sets** (`memory`, `compute`, `occupancy`, `roofline`) mapped to single-pass groups
  per architecture: gfx942, gfx950, sm_100.
  - **Each counter** has its unit (bytes, FLOPs, cycles, percent) and the formula used.
  - **The sets** are versioned in the repository.
  - **Grouping by slots:** on gfx950, SQ 8, TCC 4, TCP 4, TA 2, GRBM 2 per pass. The eight MFMA
    precisions are one group; `FETCH_SIZE` and `WRITE_SIZE` are two.
- **Metrics gray-parrot computes itself.** The vendors' sets lack what we need: the SDK's gfx950
  set has no L2 hit rate, and its `MfmaFlops` leaves out F8 and F6F4. Formulas also change
  between SDK versions (`FETCH_SIZE` did between 1.1.0 and 1.3.2), so each record names the
  formula it used.
- **`ruppell counters list`:** an architecture's sets, their groups and passes, checked against
  the device.
- **birdeye's `counters` table and the roofline position** (§1.7).

**Exit:**
- **FLOPs from counters equal 2·M·N·K for GEMMs** on both platforms. On MI350X the smoke
  already showed it: `SQ_INSTS_VALU_MFMA_MOPS_BF16` × 512 = 2·4096³ exactly (runner job 89916).
- **Measured bytes:** beside the GEMM table's model bytes for every GEMM on W1 and W2, with the
  gap.
- **The `counters` table** on W1 and W2 on both platforms, its numbers equal to the engine's.

### GP4: device activity over the run

> **CUDA, 2026-10-22** (B200 dgx-025; the runner's `ruppell-accuracy/results/gray-parrot/gp4/`).
> - **Built:** `RUPPELL_DEVICE_COUNTERS=activity@10kHz`: a PM sampling session per device, every 100 µs.
>   - **Its set:** GR, SMs, issue, tensor and warps active, DRAM and NVLink bytes; nsys's rows.
>   - **Values:** evaluated on its own thread into `device_counters.bin` (360 bytes a sample).
>   - **Raw:** `RUPPELL_DEVICE_EVAL=after` stores raw counter data (about 30 KB a sample) for
>     `ruppell_pm_eval`.
>   - **Report:** `ruppell report counters` gives activity over the run and per step kind.
> - **Against nsys's GPU metrics,** per load (not per step kind): within a few percent of their
>   value, ruppell a little lower. Not within nsys's two-run spread throughout.
> - **Cost at 10 kHz:** −1.1% (C8) and −0.5% (C32) of throughput, inside the run-to-run spread;
>   about 0.65 of a host core on the sampling thread. 10 kHz is CUDA's rate (plan 04's decision 6,
>   accepted by the operator on these numbers, 2026-10-22).
> - **Found:** stopped from the tool's own exit handler, the sessions ran into CUPTI's teardown and
>   corrupted the heap. They stop from a handler registered after the first session starts.
> - **Not built:** counter tracks in the Chrome export, birdeye's per-step-kind means, steps placed
>   by their CPU window (only steps with kernel records get samples).

**Build:**
- **Collection:**
  - **ROCm:** the device counting service; device-wide SPM once an image has ROCm 10.2.0.
  - **CUDA:** PM sampling, or NVML GPM where PM sampling is not possible.
- **Placement:** on the step timeline by clock pairs.
- **Output:** per-step-kind means in birdeye; counter tracks in the Chrome export.

**Exit:**
- **Against Nsight Systems' GPU metrics** (B200, a separate run of the same load): per step kind,
  SM active, tensor active and DRAM bandwidth lie within nsys's run-to-run spread.
- **On ROCm:** against rocprofv3's counters over the same window, or `amd-smi`'s activity,
  whichever P5 shows is comparable.
- **Cost:** at the default rate, under 1% of throughput.

### GP5: components

**Build:** counters on M14's components only. Settings:
- **ROCm:** the dispatch callback chooses per dispatch, replays included.
- **CUDA:** profiling switched around a component's eager launches; replays per step.

**Exit:** M14's counter exit: counters on the components' kernels match the vendor's on the
same kernels.

**Depends on:** M14, GP1, GP2.

## 3. What stays with the vendors' tools

| Need | Use |
| --- | --- |
| PC sampling, stall reasons per instruction | `rocprofv3 --pc-sampling-*` (beta, gfx950: host trap and stochastic); Nsight Compute's source counters |
| Thread trace | `rocprofv3 --att` |
| SASS-instrumented metrics, multi-pass sets (ncu `full`, rocprof-compute's panels) | Nsight Compute, rocprof-compute, on a kernel bench at the census shapes (Arbor's `counters.md`) |

Revisit PC sampling on ROCm for M14's components after GP1.
- **For it:** its stochastic mode exists on gfx950 and does not serialize kernels.
- **Against it:** it cannot run while counters are collected, so it would take steps of its own.
  The newest SDK documentation warns of a hardware freeze.

## 4. Order

1. **GP0's probes P1 to P6,** together: they are small, and the design depends on them.
2. **GP1, then GP3 on ROCm**: the SDK path is the more likely to work as designed.
3. **GP2** once P3 has answered; GP3 extends to CUDA with it.
4. **GP4** independent of GP1–GP3, after P5.
5. **GP5** after M14.

**The execution plans:**
- **ROCm:** the runner's `ruppell-accuracy/plans/03_GRAY_PARROT_ROCM.md`, 2026-10-21. It
  covers P1, P2, P5, GP1, GP3 on gfx950 and GP4 on ROCm, on MI350X.
- **CUDA:** the runner's `ruppell-accuracy/plans/04_GRAY_PARROT_CUDA.md`, 2026-10-21. It
  covers P3, P4, P5 on CUDA, P6, GP2, GP3 on sm_100 and GP4 on CUDA, on B200. It also proposes
  the counter record both platforms share.
