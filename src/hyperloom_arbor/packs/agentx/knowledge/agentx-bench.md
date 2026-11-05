# AgentX Benchmarking

AgentX is NOT a synthetic fixed-token benchmark. It replays real multi-turn agentic coding sessions ("WEKA traces") — variable-length, long-context (up to 1M tokens), with sub-agent spawning.

## What output_throughput measures

`output_throughput` is output tokens per second per GPU, extracted from `request_metrics.throughput.per_gpu.output_tput_tps` in the aiperf aggregate. This is the primary optimization metric.

`total_throughput_per_gpu` includes both input and output tokens — useful for understanding prefill vs decode balance but not the optimization target.

## Measurement characteristics

- Full run: ~3600s replay + ~30min warmup + ~5min aiperf install = ~70min total
- Fast mode (NUM_PROMPTS ≤ 50): 1200s replay, 1 warmup request per lane — for profiling only, not gating
- Error rate must stay ≤ 10% for a valid measurement
- KV cache hit rate from server metrics — relevant for HiCache and prefix caching optimizations
- Noise: expect a wide spread depending on host tuning and workload mix

## What affects throughput

- **Decode speed** — dominated by attention and MoE kernel performance
- **KV cache management** — HiCache offloading, eviction policy, cache hit rate
- **Speculative decoding** — EAGLE/MTP acceptance length directly impacts effective throughput
- **Scheduling** — batching, chunked prefill, max running requests, queue management
- **Collective communication** — allreduce for TP>1, topology-aware routing
