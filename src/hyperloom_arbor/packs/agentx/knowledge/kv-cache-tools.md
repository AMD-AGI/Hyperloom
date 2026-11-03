# KV Cache Trace Tools

The harness automatically captures a KV cache trace when HiCache is enabled. After each bench run, the trace is saved to `iter_N/kv_trace.jsonl` with an analysis summary at `iter_N/kv_trace_analysis.json`.

## What the trace tells you

| Event | Meaning |
|---|---|
| `request_cache` | Per-request: how many tokens were GPU-cached, CPU-cached, or cold misses |
| `backup_done` | A GPU→CPU transfer completed — includes BW in GB/s and latency in ms |
| `load_back_done` | A CPU→GPU transfer completed — this is where decode stalls waiting for data |
| `evict` | GPU cache freed tokens — shows eviction pressure |
| `snapshot` | Periodic cache utilization (every 5s) |

## When to use the trace

- **Before changing HiCache config** (ratio, write_policy, io_backend): run the simulator with different configs on the captured trace to predict which one helps
- **When throughput drops at high concurrency**: the trace shows if BW saturation or cache misses are the cause
- **When comparing CONC=2 vs CONC=14 vs CONC=32**: the trace shows how eviction pressure scales

## Using the simulator

After a bench run that produced a trace:

```bash
# See BW stats and hit rates from the actual run
python3 packs/agentx/tools/kv_trace_analyze.py iter_N/kv_trace.jsonl

# Try different cache configs without re-running the bench
python3 packs/agentx/tools/kv_trace_simulator.py iter_N/kv_trace.jsonl --sweep

# Try a specific config
python3 packs/agentx/tools/kv_trace_simulator.py iter_N/kv_trace.jsonl \
    --hicache-ratio 2.0 --gpu-tokens 2900000
```

## Key numbers to look for

- **GPU→CPU BW**: PCIe on MI355X peaks at ~32 GB/s. If you see <10 GB/s, the transfers are too small (overhead-dominated) or contended
- **CPU→GPU BW**: This is the stall path. Every ms here is a ms of decode stall
- **Combined hit rate**: >95% means HiCache is working. <80% means the cache is too small for this concurrency
- **Eviction rate**: >100/s sustained means the GPU cache is under constant pressure — increase mem-fraction-static or reduce max-running-requests
