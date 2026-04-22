#!/usr/bin/env python3
"""
analyze_profile.py — print a top-N kernel breakdown from a profiler trace.

Supported schemas:
  * rocprofv3       : { "rocprofiler-sdk-tool": [{ "kernel_symbols": [...],
                                                   "buffer_records": {"kernel_dispatch": [...]}}] }
  * rocprof v2 wrap : { "kernels": [{"KernelName", "DurationNs"}, ...] }
  * torch.profiler  : { "traceEvents": [{"cat":"kernel","name","dur"}, ...] }

Usage:
  analyze_profile.py <trace.json> [--top N] [--json]

Prints a ranked list to stdout; if --json is given, also emits a structured
list to <trace>.kernels.json suitable for ingestion by kernel-opt.md.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path


def kernel_breakdown(trace_path):
    """Return (times_us, counts) dicts keyed by kernel name."""
    data = json.load(open(trace_path))
    times, counts = defaultdict(float), defaultdict(int)

    if isinstance(data, dict) and "rocprofiler-sdk-tool" in data:
        top = data["rocprofiler-sdk-tool"][0]
        sym = {s["kernel_id"]: s["kernel_name"]
               for s in top.get("kernel_symbols", [])
               if s.get("kernel_name")}
        for d in top.get("buffer_records", {}).get("kernel_dispatch", []):
            kid = d.get("kernel_id") or d.get("dispatch_info", {}).get("kernel_id")
            name = sym.get(kid, f"<id_{kid}>")
            dur_ns = d["end_timestamp"] - d["start_timestamp"]
            times[name] += dur_ns / 1000.0
            counts[name] += 1
        return times, counts

    if isinstance(data, dict) and "kernels" in data:
        for e in data["kernels"]:
            times[e["KernelName"]] += e.get("DurationNs", 0) / 1000.0
            counts[e["KernelName"]] += 1
        return times, counts

    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    for e in events:
        if isinstance(e, dict) and e.get("cat") == "kernel" and "dur" in e:
            times[e["name"]] += e["dur"]
            counts[e["name"]] += 1
    return times, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", action="store_true",
                    help="also write <trace>.kernels.json with structured ranking")
    args = ap.parse_args()

    times, counts = kernel_breakdown(args.trace)
    total = sum(times.values())
    if not total:
        print(f"WARN: no kernel time found in {args.trace}", file=sys.stderr)
        sys.exit(2)

    ranked = sorted(times.items(), key=lambda x: -x[1])
    print(f"\nTop-{args.top} kernels ({total/1e3:.2f} ms total GPU time):\n")
    for name, t in ranked[:args.top]:
        short = name if len(name) <= 80 else name[:77] + "..."
        print(f"  {short:80s}  {t:>9.1f}us  {t/total*100:>5.1f}%  {counts[name]:>5d}x")

    if args.json:
        out = Path(args.trace).with_suffix(".kernels.json")
        json.dump(
            [{"name": n, "time_us": t, "pct_total": t/total*100, "count": counts[n]}
             for n, t in ranked],
            open(out, "w"),
            indent=2,
        )
        print(f"\nWrote ranked breakdown -> {out}")


if __name__ == "__main__":
    main()
