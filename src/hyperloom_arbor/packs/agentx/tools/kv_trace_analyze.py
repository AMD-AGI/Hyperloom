"""Analyze a KV cache trace captured by kv_trace_collector.py.

Usage:
    python3 kv_trace_analyze.py /tmp/kv_trace.jsonl
    python3 kv_trace_analyze.py /tmp/kv_trace.jsonl --csv /tmp/bw_timeline.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    d = k - f
    return values[f] + d * (values[c] - values[f])


def analyze(events: list[dict]):
    backups = [e for e in events if e.get("event") == "backup_done"]
    loadbacks = [e for e in events if e.get("event") == "load_back_done"]
    evictions = [e for e in events if e.get("event") == "evict"]
    snapshots = [e for e in events if e.get("event") == "snapshot"]

    duration = events[-1].get("ts", 0) - events[0].get("ts", 0) if len(events) > 1 else 0

    print(f"{'='*60}")
    print(f"KV Cache Trace Analysis")
    print(f"{'='*60}")
    print(f"Duration:          {duration:.1f}s")
    print(f"Total events:      {len(events)}")
    print()

    # ── Evictions ──
    print(f"── Evictions ({len(evictions)} events) ──")
    if evictions:
        total_evicted = sum(e.get("evicted", 0) for e in evictions)
        total_requested = sum(e.get("requested", 0) for e in evictions)
        print(f"  Total requested:  {total_requested:,} tokens")
        print(f"  Total evicted:    {total_evicted:,} tokens")
        print(f"  Fill rate:        {total_evicted/total_requested*100:.1f}%" if total_requested > 0 else "")
        print(f"  Rate:             {len(evictions)/duration:.1f}/s" if duration > 0 else "")
    print()

    # ── GPU → CPU (backups) ──
    print(f"── GPU → CPU Transfers ({len(backups)} events) ──")
    if backups:
        bw_values = [e.get("bw_gbps", 0) for e in backups if e.get("bw_gbps", 0) > 0]
        dur_values = [e.get("duration_ms", 0) for e in backups]
        total_bytes = sum(e.get("num_bytes", 0) for e in backups)
        total_tokens = sum(e.get("num_tokens", 0) for e in backups)

        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Total bytes:      {total_bytes/1e9:.2f} GB")
        print(f"  Transfers/s:      {len(backups)/duration:.1f}" if duration > 0 else "")
        if bw_values:
            print(f"  BW p50:           {percentile(bw_values, 50):.2f} GB/s")
            print(f"  BW p90:           {percentile(bw_values, 90):.2f} GB/s")
            print(f"  BW max:           {max(bw_values):.2f} GB/s")
        if dur_values:
            print(f"  Latency p50:      {percentile(dur_values, 50):.3f} ms")
            print(f"  Latency p90:      {percentile(dur_values, 90):.3f} ms")
    print()

    # ── CPU → GPU (load-backs) ──
    print(f"── CPU → GPU Transfers ({len(loadbacks)} events) ──")
    if loadbacks:
        bw_values = [e.get("bw_gbps", 0) for e in loadbacks if e.get("bw_gbps", 0) > 0]
        dur_values = [e.get("duration_ms", 0) for e in loadbacks]
        total_bytes = sum(e.get("num_bytes", 0) for e in loadbacks)
        total_tokens = sum(e.get("num_tokens", 0) for e in loadbacks)

        print(f"  Total tokens:     {total_tokens:,}")
        print(f"  Total bytes:      {total_bytes/1e9:.2f} GB")
        print(f"  Transfers/s:      {len(loadbacks)/duration:.1f}" if duration > 0 else "")
        if bw_values:
            print(f"  BW p50:           {percentile(bw_values, 50):.2f} GB/s")
            print(f"  BW p90:           {percentile(bw_values, 90):.2f} GB/s")
            print(f"  BW max:           {max(bw_values):.2f} GB/s")
        if dur_values:
            print(f"  Latency p50:      {percentile(dur_values, 50):.3f} ms")
            print(f"  Latency p90:      {percentile(dur_values, 90):.3f} ms")
    print()

    # ── Token usage over time ──
    if snapshots:
        usages = [s.get("token_usage", 0) for s in snapshots if "token_usage" in s]
        if usages:
            print(f"── Cache Utilization ({len(snapshots)} snapshots) ──")
            print(f"  Token usage p50:  {percentile(usages, 50)*100:.1f}%")
            print(f"  Token usage p90:  {percentile(usages, 90)*100:.1f}%")
            print(f"  Token usage max:  {max(usages)*100:.1f}%")
    print()

    # ── Timeline summary (10 time windows) ──
    if duration > 10 and (backups or loadbacks):
        window = duration / 10
        print(f"── Timeline (10 windows of {window:.0f}s each) ──")
        print(f"  {'Window':>8s}  {'Evict':>8s}  {'Backup':>8s}  {'LoadBk':>8s}  {'BW↓ GB/s':>10s}  {'BW↑ GB/s':>10s}")
        for i in range(10):
            t0 = i * window
            t1 = (i + 1) * window
            w_evict = sum(1 for e in evictions if t0 <= e.get("ts", 0) < t1)
            w_backup = [e for e in backups if t0 <= e.get("ts", 0) < t1]
            w_loadback = [e for e in loadbacks if t0 <= e.get("ts", 0) < t1]
            bw_down = sum(e.get("num_bytes", 0) for e in w_backup) / window / 1e9 if w_backup else 0
            bw_up = sum(e.get("num_bytes", 0) for e in w_loadback) / window / 1e9 if w_loadback else 0
            print(f"  {t0:7.0f}s  {w_evict:8d}  {len(w_backup):8d}  {len(w_loadback):8d}  {bw_down:10.2f}  {bw_up:10.2f}")

    return {
        "duration_s": duration,
        "num_evictions": len(evictions),
        "num_backups": len(backups),
        "num_loadbacks": len(loadbacks),
        "total_backup_gb": sum(e.get("num_bytes", 0) for e in backups) / 1e9,
        "total_loadback_gb": sum(e.get("num_bytes", 0) for e in loadbacks) / 1e9,
        "backup_bw_p50_gbps": percentile([e.get("bw_gbps", 0) for e in backups if e.get("bw_gbps", 0) > 0], 50) if backups else 0,
        "loadback_bw_p50_gbps": percentile([e.get("bw_gbps", 0) for e in loadbacks if e.get("bw_gbps", 0) > 0], 50) if loadbacks else 0,
    }


def export_csv(events: list[dict], path: str):
    """Export transfer events as a CSV for plotting."""
    transfers = [e for e in events if e.get("event") in ("backup_done", "load_back_done")]
    with open(path, "w") as f:
        f.write("ts,direction,num_tokens,num_bytes,duration_ms,bw_gbps\n")
        for e in transfers:
            f.write(f"{e.get('ts',0)},{e.get('direction','')},{e.get('num_tokens',0)},"
                    f"{e.get('num_bytes',0)},{e.get('duration_ms',0)},{e.get('bw_gbps',0)}\n")
    print(f"\nCSV exported to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze KV cache trace")
    parser.add_argument("trace", help="Path to trace JSONL file")
    parser.add_argument("--csv", help="Export transfer events to CSV", default=None)
    parser.add_argument("--json", help="Export summary as JSON", default=None)
    args = parser.parse_args()

    events = load_trace(args.trace)
    if not events:
        print(f"No events found in {args.trace}", file=sys.stderr)
        sys.exit(1)

    summary = analyze(events)

    if args.csv:
        export_csv(events, args.csv)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary JSON exported to {args.json}")
