"""Offline KV cache simulator — replay a trace with different cache configs.

Given a trace from kv_trace_collector.py, simulates what would happen with
different GPU/CPU cache sizes, eviction policies, and write policies.

Usage:
    python3 kv_trace_simulator.py /tmp/kv_trace.jsonl
    python3 kv_trace_simulator.py /tmp/kv_trace.jsonl --gpu-tokens 2000000 --host-tokens 5000000
    python3 kv_trace_simulator.py /tmp/kv_trace.jsonl --sweep  # try multiple configs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CacheConfig:
    gpu_tokens: int = 2_900_000
    host_tokens: int = 3_700_000
    gpu_to_host_bw_gbps: float = 16.0
    host_to_gpu_bw_gbps: float = 12.0
    bytes_per_token: int = 73728  # GLM-5.2 MLA: (nope=512 + rope=64) * fp8 * 78 layers / TP4
    write_policy: str = "write_through"


@dataclass
class SimState:
    gpu_cache: OrderedDict = field(default_factory=OrderedDict)
    host_cache: OrderedDict = field(default_factory=OrderedDict)
    gpu_used: int = 0
    host_used: int = 0
    hits_gpu: int = 0
    hits_host: int = 0
    misses: int = 0
    total_backup_bytes: int = 0
    total_loadback_bytes: int = 0
    total_backup_stall_ms: float = 0.0
    total_loadback_stall_ms: float = 0.0
    eviction_count: int = 0
    transfer_count: int = 0

    @property
    def total_accesses(self):
        return self.hits_gpu + self.hits_host + self.misses

    @property
    def gpu_hit_rate(self):
        return self.hits_gpu / self.total_accesses if self.total_accesses > 0 else 0

    @property
    def combined_hit_rate(self):
        return (self.hits_gpu + self.hits_host) / self.total_accesses if self.total_accesses > 0 else 0


def _evict_lru(cache: OrderedDict, num_tokens: int) -> int:
    """Evict least-recently-used entries until num_tokens are freed."""
    freed = 0
    while freed < num_tokens and cache:
        key, size = cache.popitem(last=False)
        freed += size
    return freed


def simulate_trace(events: list[dict], config: CacheConfig) -> SimState:
    """Replay trace events through a simulated two-tier cache."""
    state = SimState()

    for event in events:
        etype = event.get("event")

        if etype == "evict":
            requested = event.get("requested", 0)
            actual = event.get("evicted", 0)
            policy = event.get("policy", config.write_policy)

            if state.gpu_used > config.gpu_tokens:
                need = state.gpu_used - config.gpu_tokens + requested
            else:
                need = requested

            if need > 0:
                freed = _evict_lru(state.gpu_cache, need)
                state.gpu_used = max(0, state.gpu_used - freed)
                state.eviction_count += 1

                if policy == "write_through" and state.host_used < config.host_tokens:
                    can_store = min(freed, config.host_tokens - state.host_used)
                    if can_store > 0:
                        state.host_cache[f"evicted_{state.eviction_count}"] = can_store
                        state.host_used += can_store

        elif etype == "backup_done":
            num_tokens = event.get("num_tokens", 0)
            num_bytes = event.get("num_bytes", 0)
            duration_ms = event.get("duration_ms", 0)
            bw = event.get("bw_gbps", config.gpu_to_host_bw_gbps)

            sim_bytes = num_tokens * config.bytes_per_token
            sim_duration_ms = sim_bytes / (config.gpu_to_host_bw_gbps * 1e6) if config.gpu_to_host_bw_gbps > 0 else 0

            state.total_backup_bytes += sim_bytes
            state.total_backup_stall_ms += sim_duration_ms
            state.transfer_count += 1

            if state.host_used + num_tokens <= config.host_tokens:
                state.host_cache[f"backup_{state.transfer_count}"] = num_tokens
                state.host_used += num_tokens
            else:
                overshoot = (state.host_used + num_tokens) - config.host_tokens
                _evict_lru(state.host_cache, overshoot)
                state.host_used = max(0, state.host_used - overshoot)
                state.host_cache[f"backup_{state.transfer_count}"] = num_tokens
                state.host_used += num_tokens

        elif etype == "load_back_done":
            num_tokens = event.get("num_tokens", 0)
            num_bytes = event.get("num_bytes", 0)

            sim_bytes = num_tokens * config.bytes_per_token
            sim_duration_ms = sim_bytes / (config.host_to_gpu_bw_gbps * 1e6) if config.host_to_gpu_bw_gbps > 0 else 0

            if state.host_used > 0:
                state.hits_host += num_tokens
                state.host_used = max(0, state.host_used - num_tokens)
            else:
                state.misses += num_tokens

            state.total_loadback_bytes += sim_bytes
            state.total_loadback_stall_ms += sim_duration_ms
            state.transfer_count += 1

            state.gpu_cache[f"loaded_{state.transfer_count}"] = num_tokens
            state.gpu_used += num_tokens

        elif etype == "snapshot":
            token_usage = event.get("token_usage", 0)
            if token_usage > 0:
                state.gpu_used = int(token_usage * config.gpu_tokens)
                state.hits_gpu += int(token_usage * config.gpu_tokens * 0.01)

    return state


def print_result(config: CacheConfig, state: SimState, label: str = ""):
    print(f"\n{'─'*50}")
    if label:
        print(f"  {label}")
    print(f"  GPU: {config.gpu_tokens/1e6:.1f}M tokens  Host: {config.host_tokens/1e6:.1f}M tokens")
    print(f"  BW: ↓{config.gpu_to_host_bw_gbps:.0f} GB/s  ↑{config.host_to_gpu_bw_gbps:.0f} GB/s")
    print(f"{'─'*50}")
    print(f"  GPU hit rate:       {state.gpu_hit_rate*100:.1f}%")
    print(f"  Combined hit rate:  {state.combined_hit_rate*100:.1f}%")
    print(f"  Cold misses:        {state.misses:,} tokens")
    print(f"  Evictions:          {state.eviction_count:,}")
    print(f"  Transfers:          {state.transfer_count:,}")
    print(f"  Backup stall:       {state.total_backup_stall_ms:.1f} ms total")
    print(f"  LoadBack stall:     {state.total_loadback_stall_ms:.1f} ms total")
    print(f"  Total BW used:      ↓{state.total_backup_bytes/1e9:.2f} GB  ↑{state.total_loadback_bytes/1e9:.2f} GB")


def sweep(events: list[dict], base_config: CacheConfig):
    """Try multiple configs and compare."""
    configs = [
        ("baseline (1.5x ratio)", CacheConfig(
            gpu_tokens=base_config.gpu_tokens,
            host_tokens=int(base_config.gpu_tokens * 1.5),
            gpu_to_host_bw_gbps=base_config.gpu_to_host_bw_gbps,
            host_to_gpu_bw_gbps=base_config.host_to_gpu_bw_gbps,
            bytes_per_token=base_config.bytes_per_token,
        )),
        ("2x host ratio", CacheConfig(
            gpu_tokens=base_config.gpu_tokens,
            host_tokens=int(base_config.gpu_tokens * 2.0),
            gpu_to_host_bw_gbps=base_config.gpu_to_host_bw_gbps,
            host_to_gpu_bw_gbps=base_config.host_to_gpu_bw_gbps,
            bytes_per_token=base_config.bytes_per_token,
        )),
        ("3x host ratio", CacheConfig(
            gpu_tokens=base_config.gpu_tokens,
            host_tokens=int(base_config.gpu_tokens * 3.0),
            gpu_to_host_bw_gbps=base_config.gpu_to_host_bw_gbps,
            host_to_gpu_bw_gbps=base_config.host_to_gpu_bw_gbps,
            bytes_per_token=base_config.bytes_per_token,
        )),
        ("0.5x host (less CPU)", CacheConfig(
            gpu_tokens=base_config.gpu_tokens,
            host_tokens=int(base_config.gpu_tokens * 0.5),
            gpu_to_host_bw_gbps=base_config.gpu_to_host_bw_gbps,
            host_to_gpu_bw_gbps=base_config.host_to_gpu_bw_gbps,
            bytes_per_token=base_config.bytes_per_token,
        )),
        ("no host cache", CacheConfig(
            gpu_tokens=base_config.gpu_tokens,
            host_tokens=0,
            gpu_to_host_bw_gbps=0,
            host_to_gpu_bw_gbps=0,
            bytes_per_token=base_config.bytes_per_token,
        )),
        ("2x GPU (bigger GPU cache)", CacheConfig(
            gpu_tokens=int(base_config.gpu_tokens * 2),
            host_tokens=base_config.host_tokens,
            gpu_to_host_bw_gbps=base_config.gpu_to_host_bw_gbps,
            host_to_gpu_bw_gbps=base_config.host_to_gpu_bw_gbps,
            bytes_per_token=base_config.bytes_per_token,
        )),
    ]

    print(f"\n{'='*60}")
    print(f"  KV Cache Config Sweep — {len(configs)} configs")
    print(f"{'='*60}")

    results = []
    for label, cfg in configs:
        state = simulate_trace(events, cfg)
        print_result(cfg, state, label)
        results.append((label, cfg, state))

    print(f"\n{'='*60}")
    print(f"  Summary Comparison")
    print(f"{'='*60}")
    print(f"  {'Config':<28s} {'GPU Hit%':>10s} {'Combined%':>10s} {'Misses':>10s} {'Stall ms':>10s}")
    for label, cfg, state in results:
        stall = state.total_backup_stall_ms + state.total_loadback_stall_ms
        print(f"  {label:<28s} {state.gpu_hit_rate*100:>9.1f}% {state.combined_hit_rate*100:>9.1f}% {state.misses:>10,} {stall:>10.1f}")


def load_trace(path: str) -> list[dict]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV cache offline simulator")
    parser.add_argument("trace", help="Path to trace JSONL file")
    parser.add_argument("--gpu-tokens", type=int, default=2_900_000, help="GPU cache capacity in tokens")
    parser.add_argument("--host-tokens", type=int, default=3_700_000, help="Host cache capacity in tokens")
    parser.add_argument("--hicache-ratio", type=float, default=None, help="Host/GPU ratio (overrides --host-tokens)")
    parser.add_argument("--gpu-to-host-bw", type=float, default=16.0, help="GPU→CPU BW in GB/s")
    parser.add_argument("--host-to-gpu-bw", type=float, default=12.0, help="CPU→GPU BW in GB/s")
    parser.add_argument("--bytes-per-token", type=int, default=73728, help="Bytes per KV token per rank")
    parser.add_argument("--sweep", action="store_true", help="Try multiple configs and compare")
    parser.add_argument("--json", default=None, help="Export results as JSON")
    args = parser.parse_args()

    events = load_trace(args.trace)
    if not events:
        print(f"No events in {args.trace}", file=sys.stderr)
        sys.exit(1)

    host_tokens = args.host_tokens
    if args.hicache_ratio is not None:
        host_tokens = int(args.gpu_tokens * args.hicache_ratio)

    config = CacheConfig(
        gpu_tokens=args.gpu_tokens,
        host_tokens=host_tokens,
        gpu_to_host_bw_gbps=args.gpu_to_host_bw,
        host_to_gpu_bw_gbps=args.host_to_gpu_bw,
        bytes_per_token=args.bytes_per_token,
    )

    if args.sweep:
        sweep(events, config)
    else:
        state = simulate_trace(events, config)
        print_result(config, state, "single config")

        if args.json:
            result = {
                "gpu_tokens": config.gpu_tokens,
                "host_tokens": config.host_tokens,
                "gpu_hit_rate": state.gpu_hit_rate,
                "combined_hit_rate": state.combined_hit_rate,
                "misses": state.misses,
                "evictions": state.eviction_count,
                "backup_stall_ms": state.total_backup_stall_ms,
                "loadback_stall_ms": state.total_loadback_stall_ms,
            }
            with open(args.json, "w") as f:
                json.dump(result, f, indent=2)
