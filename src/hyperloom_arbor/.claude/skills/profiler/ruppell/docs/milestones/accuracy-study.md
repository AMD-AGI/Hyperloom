# The accuracy study (record, 2026-10-20)

ruppell against the vendor's collector and against torch.profiler, on four open-source
workloads: B200 and MI350. Protocol, tools and every number: the runner's
`ruppell-accuracy/` (`plans/01_STUDY.md`, `results/SUMMARY.md`, `results/b200/FINDINGS.md`).
Captures at `928fa80`; analysis with birdeye at `29c105f` (B200) and `1f231f8` (MI350).

**The workloads:**
- W0: the graph probe (plain and forked) and the PyTorch demo;
- W1: Qwen3-8B served by vLLM at TP1;
- W2: Qwen3-30B-A3B served at TP2;
- W3: a Qwen3-0.6B training step.

Serving loads are C8 and C32 at seed 42.

## Results

| Comparison | Result |
| --- | --- |
| **MI350: rocprofv3 in the same process** (ROCm 7.2, torch 2.13) | every dispatch matched, identical start and end (0 ns): 12,004 / 12,004 / 13,056 (W0), 1,063,928 (W1), 2,008,140 per rank (W2), 168,445 (W3) |
| **B200: Nsight Systems, separate runs** (CUPTI 130001) | counts identical (W0, W3), within 0.1% per serving load; time 1.0001 (demo) and 1.0007 (training) of nsys's software timestamps, serving kernel time by name within 0.4%; against Blackwell's hardware timestamps +0.2–0.5 µs per kernel (`software-timestamps`) |
| **MI350: torch.profiler, separate runs** | counts identical where the work was (C8, training); mean duration by name within the ruppell-vs-ruppell control (≤ 2%) on W2, torch.profiler 4–5% longer on W1 (unexplained), ruppell 4% longer on W3 (no control); torch.profiler attributes 9–11% of serving GPU time to an op, ruppell 99.8–99.9% |

## What it changed here

- **`928fa80`:** `api._define` no longer calls the tool in witness mode, which had crashed a
  hooked model.
- **`29c105f`:** birdeye demangles nsys's kernel names with ruppell's demangler.
- **`1f231f8`:** birdeye drops the `.kd` that ROCm's torch.profiler keeps in kernel names, and
  reads rocprofv3 7.2's `Agent_Id` (`Agent 2`).
- **`b6e8e09`:**
  - [`limitations.md`](../limitations.md): `software-timestamps`, and the recipe for stopping a
    served model under rocprofv3 (each GPU process in turn, output on local disk, killed once
    written; it had been "ends by SIGKILL").
  - `meta.json` is now written whole and renamed on ROCm too.

## Open

- W1's 4–5% against torch.profiler needs both profilers in one process.
- MI355X was not run.
- birdeye builds no steps from nsys's serving traces: there the markers do not alternate begin
  and end.
