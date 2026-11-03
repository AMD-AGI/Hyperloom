# TraceLens Usage

TraceLens analyzes GPU traces captured with rocprofv3. Use it to identify bottlenecks and measure roofline distance.

## Capturing a trace

- Use `vllm serve --profiler-config` with a JSON config, then POST to `/start_profile`
- Parse the JSON output, not the txt — the txt is a summary, the JSON has per-kernel detail
- View in Perfetto UI, not TensorBoard
- rocprofv3 needs `ptrace_scope=0` on the host (or `--privilege` in the container)

## Reading the output

- **GPU % time** — which kernels dominate wall-clock. Optimize the top ones first.
- **WAIT/VMEM ratio** — high means memory-latency-bound (software pipelining, prefetching). Low means compute-bound (tiling, warp specialization).
- **Roofline gap %** — distance from the Max-Achievable Frontier. >40% gap means real headroom exists.
- Idle gate analysis shows scheduling overhead and bubble time between kernels.

## Common bottleneck patterns

- MoE dispatch: often dispatch-bound, not compute-bound. Fusing doesn't help if the bottleneck is in the routing/permutation.
- Attention: check if it's prefill-dominated or decode-dominated. Different optimizations apply.
- Allreduce: check if it's bandwidth-bound or latency-bound. Small messages need latency optimization, large messages need bandwidth.
