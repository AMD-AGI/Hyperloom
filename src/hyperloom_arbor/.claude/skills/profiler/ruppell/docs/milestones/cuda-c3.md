# C3 record: ruppell on CUDA, an M1 equivalent on B200

Plan: [`../plan-cuda.md`](../plan-cuda.md) §6 C3. What CUPTI provides and why
the tool is built as it is: [`../feasibility-cuda.md`](../feasibility-cuda.md).

| | |
| --- | --- |
| Job | `tests/cuda/run_c3.sh`, run directly on `<host>` on 2026-09-30 01:41–01:44 UTC |
| Build | commit `885d4ab`, clean tree; `libruppell_tool.so` `0ad4a9e5…c93be`, `libruppell_marker.so` `b5942459…eee3`, `graph_probe` `a78e464a…c445` |
| Stack | NVIDIA B200, driver 580.126.09, CUDA 13.0.88 (host), CUPTI API 130001; the PyTorch demo in `vllm/vllm-openai@sha256:ed3c505d…` (torch 2.13.0+cu130, `nvidia-cuda-cupti` 13.0.85, also API 130001) |
| Witness | Nsight Systems 2025.3.2, `-t cuda --cuda-graph-trace=node`, on separate runs in witness mode (markers only), twice |

## Exit criteria

| Criterion | Probe, one stream (1,000 steps) | Probe, forked graph (1,000 steps) | PyTorch demo (1,000 replay steps) |
| --- | --- | --- | --- |
| Every kernel carries its scope | 100% of GPU time attributed: graph 8,000, eager 2,000 | the same | 100%: graph 12,000, eager 1,041 |
| Replays joined to their capture (by node id) | 1,000 of 1,000 | 1,000 of 1,000; kernels 4–7 replay on a second stream, each with its own node's scope | 1,000 of 1,000 |
| Every kernel in exactly one GPU-defined step | yes, 1,001 steps | yes, 1,001 steps | yes, 1,005 steps (setup, 3 warmup, capture, 1,000 replay) |
| Records | 12,002 kernels, 0 dropped | 12,002, 0 dropped | 15,051, 0 dropped |
| Trace loads in Perfetto's trace processor | 28,739 slices, 10,000 flows, no error stats | 28,743, 10,000, none | 96,410, 13,041, none |
| Witness: kernel counts by name | equal | equal | equal |
| Witness: kernels by name, step by step | 1,001 of 1,001 steps identical | 1,001 of 1,001 | 1,005 of 1,005 |
| Witness: GPU time by name | within 2.7% (bar 3%; the witness's own two runs differ by up to 1.5%) | within 1.2% (the witness's two runs differ by up to 23%: timing on two concurrent streams varies run to run) | within 0.6% (bar 3%; witness runs within 0.2%) |

Each kernel's scope was also checked one by one on the probe: kernel
`probe_kernel<K>` carries scope K+1 and node position K in every replay,
whichever stream it ran on and in whatever order it started.

## Cost

Measured inside the programs' step loops (first launch to last completion),
so process start and exit are excluded; tracer off (witness mode: markers,
no tool) and on, alternating, same GPU.

| Program | Loop off | Loop on | Paired difference (median of 5) | Per kernel |
| --- | --- | --- | --- | --- |
| Probe, 5,000 steps (60,000 kernels) | 82.1 ms | 125.5 ms | +43.3 ms (+52.8%) | +0.72 µs |
| Probe, 20,000 steps (240,000 kernels) | 356.2 ms | 512.6 ms | +158.2 ms (+44.4%) | +0.66 µs |
| Demo, 2,000 steps (30,000 kernels) | 88.2 ms | 132.3 ms | +44.0 ms (+49.9%) | +1.47 µs |

These loops launch ~1.3 µs kernels as fast as the CPU can, so any cost per
launch shows at full size; the percentage says nothing about a served model,
whose kernels are longer (M3's measurement). What the cost is:

- **CUPTI's kernel tracing, not ruppell's records.** With only kernel
  records enabled (no API, driver or external correlation records), the
  probe loop costs the same (125.6 ms). Nsight Systems pays the same on the
  same loop: 125.6 ms with node-level graph tracing, 117–118 ms with
  graph-level (off: 82.0 ms).
- **ruppell's own time** in its buffer callbacks: 10–12 ms per 60,000
  kernels and 76,000 API records, 40–47 ms per 240,000 kernels; on CUPTI's
  thread, not the launching one.
- **Blackwell's hardware timestamps** (`cuptiActivityEnableHWTrace`) cut the
  probe loop to ~90 ms, but on CUPTI 130001 they lost every graph-replayed
  kernel record, delivered eager kernels without timestamps, and hung the
  process when records were flushed every second. Not used.

For comparison, M1 measured 5.1 µs per dispatch on ROCm's probe.

## Limits

- The witness cannot share the process on this stack (one CUPTI
  subscriber), so it is a separate run: counts and kernels per step are
  exact for these deterministic programs, times are within the witness's own
  run-to-run spread. A served workload's steps differ between runs, so C4
  compares the programs of fixed-prompt steps instead.
- Memory operations are not kernels on CUDA and are not in the dispatch
  records; the probe's setup memsets appear in no step.
