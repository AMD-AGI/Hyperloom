# AgentX Benchmarking

AgentX is NOT a synthetic fixed-token benchmark. It replays real multi-turn agentic coding sessions ("WEKA traces") — variable-length, long-context (up to 1M tokens), with sub-agent spawning.

## What total_throughput measures

`total_throughput` is input and output tokens per second per GPU, extracted from `request_metrics.throughput.per_gpu.total_tput_tps` in the aiperf aggregate: the InferenceX dashboard's tput/GPU, and the primary optimization metric. At AgentX's long contexts most of it is input, so prefill-side work moves it most.

`output_throughput` (output tokens per second per GPU, `per_gpu.output_tput_tps`) is recorded beside it: a decode-side gain that hardly moves the total shows there, and in the interactivity and ITL fields.

## Measurement characteristics

- Full run: a 3600 s replay, plus the client's warmup and the aiperf install
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
