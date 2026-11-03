# Validation: the numbers

What every validation run of ruppell so far measured, numbers only. The evidence (jobs, files,
builds) is in [`status.md`](status.md), the [`milestones/`](milestones/) records and the runner
repository's plans and `ruppell-accuracy/results/`.

**Devices:**
- **MI350X:** Cluster A.
- **MI355X:** Cluster B.
- **B200:** NVIDIA, CUPTI API 130001.

Times are never compared across devices.

## 1. Against the vendor's collector

### rocprofv3 in the same process (ROCm): identical

Every dispatch matched by id, with the same kernel and the same start and end timestamps.

| Workload | Device | Dispatches, ruppell = rocprofv3, per rank | Largest timestamp difference |
| --- | --- | ---: | ---: |
| HIP graph probe (M1, 2026-09-29) | MI350X | 12,004 | 0 ns |
| PyTorch graph demo (M1) | MI350X | 2,656 | 0 ns |
| DeepSeek V4.1 Flash, TP4, AgentX C8 (M2) | MI350X | 9.6 M | 0 ns |
| The same at C32 (M3) | MI350X | 9,678,663 | 0 ns |
| Small model at TP1 / TP2, detail sampled 2/10: dispatches inside the sampled steps | MI350X | 29,733 / 55,098 and 52,602, 0 missing | 0 ns |
| DLRMv4 training, 8 ranks, 200 steps (2026-10-06) | MI350X | 479,639–493,963 | 0 ns |
| Accuracy study W0: probe, probe with a forked graph, PyTorch demo (2026-10-20) | MI350X | 12,004; 12,004; 13,056 | 0 ns |
| Accuracy study W1: Qwen3-8B served at TP1 | MI350X | 1,063,928 | 0 ns |
| Accuracy study W2: Qwen3-30B-A3B served at TP2 | MI350X | 2,008,140 | 0 ns |
| Accuracy study W3: Qwen3-0.6B training | MI350X | 168,445 | 0 ns |

- **DLRMv4** ran on the system ROCm runtime, because rocprofv3 cannot run against the one bundled
  in torch's wheel.
- **Once, on one rank of DLRMv4,** rocprofv3's own trace stopped short at shutdown: 268,001 of
  301,901 dispatches. ruppell's record was complete.

### Nsight Systems, separate runs (B200): equal counts, time within nsys's method

| Workload | Kernels, ruppell / nsys | Steps identical by kernel name | GPU time, ruppell / nsys |
| --- | --- | --- | --- |
| CUDA graph probe, one stream (C3, 2026-09-30) | equal | 1,001 of 1,001 | within 2.7% by name (nsys against itself: 1.5%) |
| Probe, forked graph (C3) | equal | 1,001 of 1,001 | within 1.2% (nsys against itself: up to 23%) |
| PyTorch demo (C3) | equal | 1,005 of 1,005 | within 0.6% (nsys against itself: 0.2%) |
| Accuracy study W0, PyTorch demo | 13,041 / 13,041 | 1,005 of 1,005 | 1.0001 |
| Accuracy study W3, training | 153,443 / 153,443 | | 1.0007 |
| Accuracy study W1, serving, per load | within 0.1% | | 0.996 (mean duration by name) |
| Accuracy study W2, serving, rank 0, per load | within 0.1% | | 1.0002 (mean duration by name) |
| vLLM, a small model at TP1/TP2 and a 35B-A3B MoE at TP2/TP4 (C4): the prefill steps of fixed prompts | 16 of 16 programs the same on every rank | | |

- **The time ratios are against nsys's software timestamps,** which is the method ruppell's
  CUPTI runtime uses. Against Blackwell's hardware timestamps, ruppell reads 0.2–0.5 µs longer
  per kernel: 1.114 (demo), 1.021 (training), 1.027 (W1) and 1.055 (W2).
- **Collectives** cannot be checked rank by rank across separate runs. The two ranks' totals
  agree: 973 ms against 958 ms (W2, C32).

## 2. Against torch.profiler (MI350X, separate runs)

| Workload, load | Kernels, ruppell / torch.profiler | Mean duration, ruppell / torch.profiler | ruppell against ruppell (the floor) |
| --- | --- | ---: | ---: |
| W1, C8 | 632,565 / 632,565 | 0.958 | 0.996–1.002 |
| W1, C32 | 300,809 / 299,821 | 0.949 | 1.008–1.020 |
| W2, C8, rank 0 / rank 1 | 1,082,984 / 1,082,984 | 0.987 / 1.002 | 1.001 / 1.005 |
| W2, C32, rank 0 / rank 1 | 543,809 / 545,215 | 0.994 / 1.001 | 0.997 / 0.996 |
| W3, 30 training steps | 152,400 / 152,400 | 1.042 | not run |

- **GPU time with an op:**

  | | W1 | W2 | W3 |
  | --- | ---: | ---: | ---: |
  | ruppell | 99.8% | 99.9% | 100% |
  | torch.profiler | 10.7% | 8.9% | 100% |

  In torch.profiler's trace, a kernel replayed from a graph has no op.
- **Cost of torch.profiler:**
  - Its trace export held the server for 6–39 minutes per profiled load.
  - One load's trace per rank was 88–121 MB gzipped on W1 and 258–540 MB on W2. ruppell's
    records of the whole run were 125 MB (W1) and 451 MB (W2).
- **DLRMv4 training,** over five-step windows: torch.profiler's file holds 66–83% of the kernel
  time ruppell records for the same steps (median 76%).
  - Its `ProfilerStep`s hold from 117 kernels (1.5–3.5 ms) to 448 ms. ruppell's steps run
    255–363 ms.
  - 54–83% of backward kernel time is named only by an aten op. ruppell names the module for
    99.99%.

## 3. Graph replays, attribution and steps

| Workload | Device | Graph replays joined | GPU time attributed | Steps |
| --- | --- | --- | --- | --- |
| HIP graph probe (M1) | MI350X | 1,000 of 1,000 | 100% | |
| DSV4.1 Flash, TP4, AgentX C8 (M2) | MI350X | 40,126 of 40,126 per rank (1 repaired) | 99.994–99.995% | 19,964 per rank; 0 kernels outside a step or in two; fingerprints equal on 4 ranks |
| DSV4.1 Flash, C32 (M3) | MI350X | 9,728 of 9,728 | | |
| DSV4.1 Flash, decode, by module scope (M3) | MI350X | | 95.6–96.3% (any scope: 100%) | |
| DLRMv4 training, 8 ranks (eager) | MI350X | | 99.64–99.94% per rank; backward 99.99% | 200 per rank; 0 kernels in two steps; 0 fingerprint mismatches |
| Model M campaign B, C32 | MI350X | | decode by (layer, op) 99.90% | checks pass |
| model-w, TP8, C32, 8k in / 1k out | MI350X | 51 of 51 | 99.95–99.96%; decode by (layer, op) 99.3% | 0 fingerprint mismatches |
| Accuracy study W1, W2, W3 | MI350X | | 99.8%, 99.9%, 100% | |
| Model M, TP and EP captures | MI355X | | | checks pass, except one kernel of about 680,000 (EP, rank 1) |
| CUDA graph probe and PyTorch demo (C3) | B200 | 1,000 of 1,000 each | 100% | 1,001 and 1,005; 12,002 and 15,051 kernels, 0 dropped |
| vLLM, a small model, TP1 (C4) | B200 | 1,607 of 1,607 | 97.83% | 505 |
| vLLM, a small model, TP2 (C4) | B200 | 1,680 of 1,680 | 98.99–99.26% | 506; ranks equal |
| vLLM, 35B-A3B MoE, TP2 (C4) | B200 | 2,415 of 2,415 | 99.83–99.87% | 505; ranks equal |
| vLLM, 35B-A3B MoE, TP4 (C4) | B200 | 2,296 of 2,296 | 99.74–99.91% | 506; ranks equal |
| Model M, TP8+EP8, C32, 8k in / 1k out | B200 | 289,729 per rank, 0 refused | 99.98%; decode by (layer, op) 97.8% | 4,671 step ids identical on 8 ranks; 0 fingerprint mismatches |

- **C4's unattributed 0.1–2.2%** is startup work before the first engine step. After it, 0
  kernels are unattributed.
- **The SDK mis-correlates** about one graph replay in 17,503. The join repairs it.
- **The trace's window against the client:**

  | | Requests running, ruppell / client | Window, ruppell / client |
  | --- | --- | --- |
  | DSV4.1 Flash | 0.864 / 0.866 | |
  | model-w | 30.93 / 31.97 | 357.59 s / 357.63 s |

## 4. Does tracing change the computation?

- **DSV4.1 Flash, TP4:** every checked step's GPU program (kernels, order, grids) was identical
  traced and untraced, on every rank, with the tuning choices pinned.
  - Without the pin, Inductor timed its block size per process and chose grids of 128 and 512
    traced, where untraced chose 256.
  - Two untraced servers: compute totals within 1% per rank.
- **vLLM on B200 (C4):**
  - greedy outputs identical (8 of 8);
  - first-token log-probabilities identical (0.0 nats at 98 positions);
  - prefill programs identical.
- **DLRMv4 training:**
  - Normalized entropy identical to four decimals at all 10 logged steps.
  - AUC within 1e-4 and accuracy within 3e-4, no more than two untraced runs differ.
  - Programs: with Triton's autotune cache, 36 of 467 segments differed traced against
    untraced, and 35 untraced against untraced. Without the cache, 201 of 467 differed both ways.

## 5. Cost

**Per launch, on programs that launch back to back:**

| | Cost |
| --- | --- |
| ROCm, HIP graph probe (M1) | 2.5 µs per dispatch, plus 259 ms per process |
| ROCm, DSV4.1 Flash serving (M3) | about 2 µs of host time per dispatch, inside `hipGraphLaunch`. It is the SDK's dispatch tracing: without dispatch records, ITL is 1.00× |
| CUDA, graph probe (C3) | 0.66–0.72 µs per kernel; nsys pays the same on the same loop |
| CUDA, PyTorch demo (C3) | 1.47 µs per kernel |

**Serving, every kernel traced (full detail):**

| Workload | Load | Throughput | Per token (ITL / TPOT) | TTFT |
| --- | --- | ---: | ---: | ---: |
| DSV4.1 Flash, TP4, AgentX, MI350X | C8 | −4.5% | 1.26–1.47× | 1.15–1.44× |
| DSV4.1 Flash, TP4, AgentX, MI350X | C32 | −6.2% | 1.08–1.17× | 1.03–1.19× |
| Qwen3-8B, TP1, B200 | C8 | −3.6% | +3.4% | +7.3% |
| Qwen3-8B, TP1, B200 | C32 | −4.1% | +3.3% | +35% |

**Serving, detail sampled:**

| Workload, mode | Cost |
| --- | --- |
| DSV4.1 Flash, MI350X: every mixed and prefill step, decode 5/1000, CPU op tree | C32 throughput −2.5%; C8 in flight +0.7%; a traced mixed step +35–50 ms of host time |
| Qwen3-8B, B200, 5/1000 | throughput −0.4 to −0.6% at C8 and −0.55 to −0.6% at C32; TPOT within ±0.6%; C32 TTFT +24–31% (open) |
| Qwen3-8B, B200, the adapter and markers alone | throughput −0.05% (C8) and −0.06% (C32) |
| Model M, TP8+EP8, C32, B200 | −1.2% tok/s/GPU (174.4 against 176.5; the untraced load was longer) |
| Model M campaign B, C32, MI350X | −1.4% |

**Sampled steps read like full detail** (Qwen3-8B, B200, 2/10, decode steps):

| | Full | A second full run | Sampled |
| --- | ---: | ---: | ---: |
| Kernels per step | 578.0 | 577.9 | 578.1 |
| Mean duration by name, of full's | 1 | 1.000 | 1.004 |
| Full's time in names within 3% | | 99.1% | 98.8% |
| Kernels outside every step, or ending late | 0 | 0 | 0 |

**Training (DLRMv4, MI350X, step time):**

| Mode | Step time |
| --- | ---: |
| markers only | −0.26% |
| full detail | +0.88% |
| full detail with the CPU op tree | +1.12% |

**Volume:**

| Run | Records |
| --- | --- |
| HIP graph probe | 0 dropped at 129,412 dispatches/s |
| DSV4.1 Flash, C8 | 37 M dispatches and 6.4 GB per rank, 0 dropped |
| Model M on B200 | 36 GB per run |
| Qwen3-8B, whole run | 125 MB |
| Qwen3-30B-A3B, whole run | 451 MB |

## 6. Copies and fills (M6, 2026-10-20)

| Workload | Device | Copies, ruppell = witness | Bytes | Recorded in a step |
| --- | --- | --- | --- | --- |
| PyTorch demo | MI350X (rocprofv3, same process) | 4, identical times | 33.5 MB | 100% |
| W1, Qwen3-8B TP1 | MI350X | 15,627, identical times | 16.4 GB | 100% |
| W2, Qwen3-30B-A3B TP2 | MI350X | 38,381 per rank, identical times | 30.6 GB | 99.998% (2 per rank outside) |
| PyTorch demo | B200 (nsys, separate run) | 12 host-to-device, equal | 33.6 MB, equal | 100% |
| W1, Qwen3-8B TP1 | B200 | equal by kind: host-to-device 4,349, device-to-host 6,270, device-to-device 1,729, fills 12 | equal, except device-to-device: 613 MB against 788 MB (batching differs between runs) | 99.9998%; 1,563 copies replayed from graphs |

On MI350X, the kernel records were unchanged with copies on: 1,064,657 (W1) and 2,006,992 (W2)
per rank, still identical to rocprofv3's.

**Cost.** Qwen3-8B TP1 was served plain, then under ruppell at full detail with copies off and
with copies on, in three rotated rounds.

| | Copies on against off: throughput | TPOT | Full detail against plain: throughput |
| --- | ---: | ---: | ---: |
| B200, C8 | −0.00% (every round within 0.2%) | +0.17% | −2.6% |
| B200, C32 | +0.03% (every round within 0.2%) | +0.88% | −2.0% |
| MI350X, C8 | not resolved (two copies-off rounds stalled, −17%) | −0.64% | −4.4% |
| MI350X, C32 | not resolved (the arms vary by up to 62%) | not resolved | about −50% |

## 7. NVTX and ROCTX ranges (M7, 2026-10-20)

**The demo:** the PyTorch graph demo with `emit_nvtx`, explicit ranges and the CPU op tree.

| | B200 | MI350X |
| --- | --- | --- |
| Ranges recorded; open; unmatched | 768; 0; 0 | 768; 0; 0 |
| `emit_nvtx` op range = the CPU op tree's op, same kernel | 39 of 39 eager; 2,400 of 2,400 replayed | 35 of 35 eager; 2,000 of 2,000 replayed |
| Against the vendor | nsys, separate run: the same 768 ranges, counts equal for all 18 names | rocprofv3, same process: 768 of 768 paired by name and thread |
| Kernel time inside a range, without ruppell's scopes | 99.9% | 99.7% |

- **MI350X timestamps against rocprofv3:**
  - start +0.33 µs and end −0.34 µs at the median, 4.6 µs at the 99th percentile;
  - five of 768 events 0.76–1.0 ms apart, ruppell earlier: open (roadmap-v2 M13).
- **B200 against nsys's NVTX projection:**
  - instances and kernels are equal for every eagerly launched range;
  - ruppell's projected time is longer by about 0.25 µs per kernel (software timestamps);
  - a replay is counted under the range around its launch by nsys, and under the ranges open at
    capture by ruppell.

**Served models:**

| | B200 | MI350X |
| --- | --- | --- |
| vLLM's layerwise ranges (Qwen3-8B, V1 model runner, eager) | 558,805; 90.8% of kernel time | 556,770; 88.4% |
| Collective kernels with a range (Qwen3-30B-A3B TP2) | NCCL's 3,057 of 163,204 | 0 of 104,962 |

**Cost:**

| | B200 | MI350X |
| --- | --- | --- |
| Per call (a push, a pop or a mark), the demo's replay loop | about 0.5 µs | about 4 µs |
| With vLLM's layerwise ranges, throughput (one run each) | −9% | −31% |

## 8. Collectives (M8, 2026-10-21)

**Served:** each collective kernel's call from the vLLM adapter, and the library's record from
ruppell's NCCL profiler plugin (NCCL 2.29–2.30 on B200, RCCL 2.27.7 on MI350X), per rank.

| | B200 | MI350X |
| --- | --- | --- |
| Collective kernels with their call: W2 at TP2 | 163,790 of 163,792 | 155,724 of 155,726 |
| W2 at TP4 (expert parallel); Qwen3-8B at TP8 | 163,300 of 163,302; 128,038 of 128,039 | 155,822 of 155,824; 118,032 of 118,033 |
| Of those, replayed from graphs | all, at every size | all, at every size |
| Library launches paired with kernels (W2, TP2) | 2,869 of 2,869, names agreeing | 2,466 of 2,466 (RCCL's kernels name no function) |
| Calls' bytes against the library's | every call agrees, at TP2, TP4 and TP8 | every call agrees |
| The library's GPU timer against ruppell's kernel start, residuals p1–p99 (TP2) | 5.9–6.7 µs; 0.25–0.46 s had the pairing been one call off | 1.3–9 µs; 0.6–1.5 s one call off |
| Counts against the vendor | Nsight Systems, separate run: NCCL's NVTX ranges, nsys's kernels, the plugin's calls and tasks equal (1,604 all-gathers, 1,651 all-reduces) | `rocprofv3 --rccl-trace`, same process: 557 all-gathers and 390 all-reduces, as ruppell's tasks and kernels |

- **The kernels without a call** are 1–2 float32 all-reduces per rank at startup, before
  `ruppell.init`. The library's record names them.
- **Counts across separate runs** differ by whole eager steps (97 all-reduces each on W2), so
  "equal to nsys" holds per step.

**Against nccl-tests and rccl-tests** (2 GPUs, the run's sizes, the same node), each launch's
transfer estimate against the tests' kernels:

| | B200 (nccl-tests) | MI350X (rccl-tests) |
| --- | --- | --- |
| All-reduce, 8.4–67 MB | the run's kernels 8–10 µs longer (29 and 67 MB) | within 1.3% |
| All-gather, 0.3–9.7 MB | 1.5–7.3 µs longer | 7.5–8.6 µs longer |
| All-gather, 311 MB | 28 µs (8%) longer | 1.4% shorter |
| The library's own channel spans | show the same differences | show the same differences |

Below a few MB, nccl-tests' bus bandwidth measures the host's launches (GPUs idle 15–22 µs
between 16 µs kernels at 2.4 MB).

**Training, 8 ranks, the library's record alone** (no adapter):

| Per rank | B200, DDP (W3) | MI350X, DDP (W3) | MI350X, DLRMv4 (200 steps) |
| --- | --- | --- | --- |
| Collective kernels with the library's record | 672 of 672 | 650 of 672 | 2,923 of 3,136 |
| Without | none (NCCL 2.29's 32 broadcasts per rank have no task; their API calls are the record) | 22 MSCCL all-reduces | 213 MSCCL all-to-alls |
| Against the vendor | Nsight Systems: 639, 32 and 1 | rocprofv3's outermost calls: 639, 32 and 1 | rocprofv3's outermost calls equal per function, and its kernel trace dispatch for dispatch (7 of 8 ranks) |
| Sizes | all-reduced elements per step exactly DDP's gradients | short by the MSCCL bucket | each all-reduce's count is its PyTorch op's elements (88 of 88, steps 10–14) |
| The plugin's cost per step | not resolved (−0.2%, rounds spread 4–7%) | +0.15% | not resolved (+0.5%, one clean round) |

- **RCCL's channel spans at 8 ranks are not the kernels'**, so its timer checks nothing there:
  13 times the all-reduce kernel on DDP, and 1,300–1,700 times on DLRMv4, at p50. On DLRMv4 the
  PyTorch op behind each kernel checks the pairing: function and datatype agree on 429 of 429.

**Cost** (extra host time per call, median of rounds):

| | B200 | MI350X |
| --- | --- | --- |
| The plugin, calls of 8.4 MB and up | +2.4–4.3 µs (the library's own events: +2.4–4.1) | +0.4–0.9 µs |
| The plugin, small all-gathers, one thread | +1.1–2.3 µs | +4.5–5.1 µs |
| The adapter, per eager collective call | 10–14 µs | about 20 µs |
| The adapter in the serving loads (C8, C32) | within the spread | within the spread |

## 9. What validation found wrong

| Found | Numbers | Now |
| --- | --- | --- |
| Workers killed at shutdown (`--shutdown-timeout 0`) | about 140 steps lost on 5 of 8 ranks, while `meta.json` said 0 dropped | `records_complete` marks a whole record; the census names the incomplete ranks |
| The first CUDA detail-sampling design | 177,282 kernels recorded outside every step; sampled kernels' end times 3.7–9.8 ms late | fixed: wait for the GPU, then switch |
| CUDA software timestamps | 0.2–0.5 µs per kernel longer than Blackwell's hardware timestamps | the `software-timestamps` limitation |
| DLRMv4 attribution before the backward and start-up scope fixes | 98.06–99.44% | 99.64–99.94% |
| Every kernel traced, against plan-v0's overhead gate (DSV4.1 Flash) | fails at C8 and C32 | detail sampling |
| W1 against torch.profiler | 4–5% longer under torch.profiler, against a 2% floor | open |
| C32 TTFT, CUDA, sampled at 5/1000 | +24–31% | open |
| MI355X | Model M only, one check failing on one kernel; the accuracy study not run | open |
| ROCTX without rocprofv3 (MI350X) | 0 ranges: PyTorch's `libroctx64` never reaches the profiler SDK | `ruppell run` preloads the SDK's ROCTX library: 318 of 318 |
| ROCTX timestamps against rocprofv3 | five of 768 events 0.76–1.0 ms apart | open (roadmap-v2 M13) |
| The cost of a ROCTX call | about 4 µs, eight times an NVTX call on B200 | ranges off by default; roadmap-v2 M13 |
| The vLLM adapter's all-gathers | the gathered output recorded as the input: bytes and bandwidth 2× | fixed (`b346731`); every call's bytes now equal the library's |
| The NCCL plugin writing on the library's threads | +116–217 µs per call on MI350X, writing to NFS | a writer thread, one line per task (`6534c67`, `d73acb6`): +0.4–0.9 µs |
| The adapter's definitions written on the calling thread | a call kind's first launch waited 1.3 ms at p50; C32 throughput −8% | written off the thread (`d096517`): 46 µs |
| RCCL at 8 ranks: one kernel, many p2p tasks | 18,673 tasks for about 2,400 kernels per rank | a kernel pairs with a launch, the tasks of one call or group (`549810d`) |
| NCCL 2.29 gives broadcasts no task; pairing by order | every later pair slipped: DDP's elements per step wrong, residuals 20 ms | paired within each function (`930b21e`); the API call is the record (`ec4abb7`) |
| The plugin unloaded under its writer thread | SIGSEGV at exit after `destroy_process_group`, 8 of 8 ranks on MI350X; rocprofv3's output lost | it pins itself (`706f4ac`) |
| The timer fit over every launch | one stale 10.8 s start: none within 100 µs | fitted near the median offset (`6952703`): 65–70% |
| RCCL's MSCCL kernels | no record: 22 all-reduces per rank on DDP, 213 all-to-alls on DLRMv4 | open: RCCL reports nothing for them over interface v4 |
| The adapter's host time per eager call | 10–20 µs, of which the wrapper is 2–3 | open |
