# Feasibility on CUDA: what CUPTI provides on B200 (C0)

Answers to the questions of [`plan-cuda.md`](plan-cuda.md) §6 C0, from
`tests/probes/cupti_probe.cu` run on the B200 node on 2026-09-29.

| Stack | Version |
| --- | --- |
| Node | `<host>`, 8× NVIDIA B200 |
| Driver | 580.126.09 (CUDA driver API 13000) |
| Toolkit | CUDA 13.0.88 (`/usr/local/cuda-13.0`), runtime 13000 |
| CUPTI | API version 130001 (`libcupti.so.2025.3.1`) |
| Nsight Systems | 2025.3.2.474 |

The probe captures a graph on stream s1 that forks onto s2 through events:
`kA` (external id 101), `kB` on s2 (102), `kC` (103), a memset (104), `kA`
again on s2 (105); instantiates it; replays it 100 times under external id
300; and launches one eager kernel before (200) and after (201).

## Answers

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does a replayed kernel's `graphNodeId` name the captured node or the instantiated one? | **The instantiated one.** Instantiation clones the captured graph (id 1) into a new graph (id 2), and replayed kernels carry graph 2's node ids. For every cloned node, `CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED` reports the clone and its `originalNode`, so the clone's id maps back to the captured node exactly. Node ids are unique in the process. |
| 2 | Does `GRAPHNODE_CREATED` fire during stream capture, inside the launch, on the capturing thread? | **Yes.** One callback per captured operation, in capture order, on the capturing thread, while the launching runtime call (`cudaLaunchKernel`, `cudaMemsetAsync`) is still in progress, so the external id pushed around the launch is the one on top. Each node's scope is therefore known at creation. |
| 3 | Which correlation and external ids do replayed kernels carry? | **The launch's.** All four kernels of a replay share the `cudaGraphLaunch` call's correlation id, and its external id (300). The fork is real at replay: `kC` ran on stream 16 and the others on 13. Eager kernels carry their own launch's ids (200, 201). |
| 4 | Can nsys and a second CUPTI client trace one process? | **No, on this stack.** Under `nsys profile -t cuda`, `cuptiSubscribe_v2` fails with `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` (reported old subscriber: "CUPTI for <not provided>"). Without a callback subscriber, the activity API alone fails the same way, including external-id push and pop. Several subscribers need CUDA 13.3 and driver r610 or later. |
| 5 | Record rate and drops | 402 kernel records, 133 API records, 118 external correlation records, 100 memset records; 0 dropped. Rates at serving size are C3's measurement. |

## Consequences for the port

- **Join by node id, not position.** A replayed kernel's scope is the scope
  on top when its original node was created: follow `graphNodeId` through
  the clone map to the captured node. This needs no ordering of a replay's
  kernels, so graphs that fork onto several streams join exactly, and
  there is nothing like the ROCm repair for mis-correlated dispatches.
- **No parameter parsing in the tool.** Graph lifecycle comes from CUPTI's
  ids (node, graph, clone), not from API handles, so the tool needs runtime
  and driver callbacks only on capture begin and end and on graph launch.
  Captured launches need no API callbacks at all; the resource callback
  sees them.
- **The witness is a separate run.** With one subscriber per process, the
  independent collector (nsys) runs the same program in ruppell's witness
  mode (markers only, no tool), and the comparison is by kernel name, with
  counts per step through the marker kernels both collectors see. This is
  weaker than ROCm's per-dispatch match in the same process, and every
  witness result says so.
- **Memory operations are not kernels on CUDA.** A memset or memcpy node
  is recorded as its own activity kind, not as a kernel dispatch (on ROCm a
  memset is a fill kernel). The graph join counts kernel nodes only.

## Raw output (probe, no nsys)

```
versions  cupti_api 130001  driver 13000  runtime 13000
subscribe CUPTI_SUCCESS
node   <tid> 1 211 1 4294967296 0 101      (capturing, in cudaLaunchKernel, graph 1, kernel node, ext 101)
node   <tid> 1 211 1 4294967297 0 102
node   <tid> 1 211 1 4294967298 0 103
node   <tid> 1  51 1 4294967299 2 104      (memset node, in cudaMemsetAsync)
node   <tid> 1 211 1 4294967300 0 105
captured graph 1
clone  2 8589934592 <- 1 4294967296        (one per node, during cudaGraphInstantiate)
...
exec   2 1
replayed kA node 8589934592 x100, kA node 8589934596 x100, kB node 8589934593 x100, kC node 8589934594 x100
replay0  kA,kB,kA on stream 13, kC on stream 16; corr 30, ext 300, graph 2
summary  kernels 402  graph launches 100  api 133  memset 100  extc 118
```

## Found in C3: the cost of kernel tracing

On a launch-bound loop of ~1.3 µs kernels (the probe, 5,000 steps: 82 ms
untraced), every configuration that records kernels costs the same, about
0.7 µs per kernel: ruppell with all its activity kinds, ruppell with kernel
records only (125.6 ms either way), and Nsight Systems (125.6 ms at node
level, 117–118 ms at graph level). The cost is CUPTI's kernel
instrumentation, not what ruppell records. Blackwell's hardware timestamps
(`cuptiActivityEnableHWTrace`, enabled at the first context) bring the loop
to ~90 ms but, on CUPTI 130001, drop every graph-replayed kernel record,
leave eager kernels without timestamps, and hang with periodic flushes; the
tool does not use them. Record: [`milestones/cuda-c3.md`](milestones/cuda-c3.md).

## Found at the merge with main: detail sampling

Main's detail sampling (`ruppell_detail`, `RUPPELL_DETAIL=K/N`) stops and
starts kernel records between engine steps; on ROCm a sampled step loses
nothing. On CUDA, `cuptiActivityDisable(CONCURRENT_KERNEL)` at the next
step's begin, while the GPU still runs the sampled step (vLLM's async
scheduling), kept the right kernels but gave them end times milliseconds
late: in a TP1 small-model run with 1/4 sampling, every sampled step's
kernels had median durations of 3.7–9.8 ms (the same steps fully traced:
microseconds), and 31,117 kernels ended after their step's end marker. The
CUDA tool therefore refused detail sampling until 2026-10-20.

## Detail sampling on CUDA (2026-10-20)

Record: the runner's `ruppell-accuracy/results/b200/SAMPLING.md` (B200,
Qwen3-8B at TP1, vLLM 0.28.0, CUPTI 130001).

**How "off" works now.** It must not run while the sampled step's kernels are
in flight, and it must come before the next step's kernels are launched:
CUPTI decides at launch whether a kernel is recorded.
- A host function on the step's stream (run once the GPU has finished the
  sampled step) kept the sampled steps clean, but recorded the next steps'
  kernels, already launched, outside every step: 177,282 at 2/10.
- So `ruppell_detail_on_stream(0, stream)` waits for the GPU
  (`cuCtxSynchronize`, the sampled step on every stream), then switches kernel
  records off. It costs one drained step per switch, one in N steps.
- It switches the runtime and driver API records with the kernel records
  (`RUPPELL_DETAIL_API=0` keeps the API records on in every step). With kernel
  records alone, the first step after records came back on read slow (decode
  attention +6.8%).
- "On" takes effect at once.
- The Python API passes torch's current stream: the default stream, handle 0,
  where ruppell's markers go too.

**Checked at 2/10, against full detail and a second full run:**
- 322 sampled steps of 578 kernels each, as at full detail;
- the same median kernel, 3.52 µs;
- no kernel ending after its step's end marker, none outside every step;
- decode kernels within 3% of full detail's durations for 98.8% of the time
  (two full runs: 99.1%).

**Cost at 5/1000**, against plain serving, three rounds:
- **Throughput:** −0.4 to −0.6% (full detail: −3.6 to −4.1%).
- **Per-token time:** within ±0.6% (full: +3.3%).
- **TTFT at C8:** +0.5 to +8.7% (full: +7.3%).
- **TTFT at C32:** +24 to +31% (full: +35%). That comes from the tool, not the adapter (the
  adapter alone: +2%), and what part of it is open ([`TODOs.md`](../TODOs.md)).
