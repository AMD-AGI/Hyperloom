#!/usr/bin/env python3
"""MLPerf optimization utilities.

CLI used by scripts/common.sh thin wrappers. Every subcommand here corresponds
to one former python3 -c heredoc in common.sh. The stdout contract (field order,
delimiters, numeric precision), exit codes, and stderr behavior are preserved
verbatim — callers parse outputs with cut/grep/eval.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys


# -------------------------------------------------------------------------
# Helpers: iterate MLLOG events
# -------------------------------------------------------------------------

def _iter_mllog(log_path: str):
    """Yield parsed JSON dicts for every ':::MLLOG' line in the log file."""
    with open(log_path) as f:
        for line in f:
            if not line.startswith(":::MLLOG"):
                continue
            try:
                yield json.loads(line.replace(":::MLLOG ", ""))
            except json.JSONDecodeError:
                continue


# -------------------------------------------------------------------------
# extract-ms-per-iter: average iteration time from consecutive train_loss events
# -------------------------------------------------------------------------

def cmd_extract_ms_per_iter(args):
    train_loss_events = [e for e in _iter_mllog(args.log_file) if e.get("key") == "train_loss"]
    iter_times_ms = []
    for i in range(1, len(train_loss_events)):
        delta = train_loss_events[i]["time_ms"] - train_loss_events[i - 1]["time_ms"]
        iter_times_ms.append(delta)

    if not iter_times_ms:
        print("ERROR: no iteration times found", file=sys.stderr)
        sys.exit(1)

    warmup = args.warmup
    measure = args.measure
    if len(iter_times_ms) >= warmup + measure:
        measured = iter_times_ms[warmup:warmup + measure]
    elif len(iter_times_ms) > warmup:
        measured = iter_times_ms[warmup:]
    else:
        measured = iter_times_ms

    avg = sum(measured) / len(measured)
    print(f"{avg:.1f}")


# -------------------------------------------------------------------------
# extract-mllog-field: first value for a given MLLOG key
# -------------------------------------------------------------------------

def cmd_extract_mllog_field(args):
    for data in _iter_mllog(args.log_file):
        if data.get("key") == args.field:
            print(data["value"])
            break


# -------------------------------------------------------------------------
# verify-gbs: assert logged global_batch_size matches expected
# -------------------------------------------------------------------------

def cmd_verify_gbs(args):
    gbs = None
    for data in _iter_mllog(args.log_file):
        if data.get("key") == "global_batch_size":
            gbs = data["value"]
            break

    if gbs is None:
        print("WARNING: could not find GBS in MLLOG", file=sys.stderr)
        sys.exit(0)

    if gbs != args.expected:
        print(f"ERROR: GBS mismatch: found {gbs}, expected {args.expected}", file=sys.stderr)
        sys.exit(1)

    print(f"GBS verified: {gbs}")


# -------------------------------------------------------------------------
# extract-losses: tab-separated train_loss samples/loss/lr list
# -------------------------------------------------------------------------

def cmd_extract_losses(args):
    for data in _iter_mllog(args.log_file):
        if data.get("key") == "train_loss":
            md = data.get("metadata", {})
            sc = md.get("samples_count", "?")
            lr = md.get("lr", "?")
            print(f"samples={sc}\tloss={data['value']:.6f}\tlr={lr}")


# -------------------------------------------------------------------------
# extract-ttt: wall seconds between run_start and run_stop, plus status
# -------------------------------------------------------------------------

def cmd_extract_ttt(args):
    events = list(_iter_mllog(args.log_file))
    run_start = next((e["time_ms"] for e in events if e.get("key") == "run_start"), None)
    run_stop = next((e for e in events if e.get("key") == "run_stop"), None)

    if run_start and run_stop:
        seconds = (run_stop["time_ms"] - run_start) / 1000
        status = run_stop.get("metadata", {}).get("status", "unknown")
        print(f"{seconds:.1f}\t{status}")
    else:
        print("ERROR: run_start/run_stop not found")


# -------------------------------------------------------------------------
# project-ttt: power-law extrapolation of TTT from a Tier 3 raw log
# Output: "<projected_ttt_seconds>\t<projected_samples>\t<samples_per_sec>\t<r_squared>"
# -------------------------------------------------------------------------

def cmd_project_ttt(args):
    TARGET = 3.34
    gbs = int(args.gbs)

    train_loss_events = []
    eval_events = []

    for data in _iter_mllog(args.log_file):
        if data.get("key") == "train_loss":
            train_loss_events.append(data)
        elif data.get("key") == "eval_accuracy":
            sc = data.get("metadata", {}).get("samples_count")
            val = data.get("value")
            if sc is not None and val is not None:
                eval_events.append((int(sc), float(val)))

    iter_deltas = []
    for i in range(1, len(train_loss_events)):
        delta = train_loss_events[i]["time_ms"] - train_loss_events[i - 1]["time_ms"]
        iter_deltas.append(delta)

    warmup = min(5, len(iter_deltas) // 2)
    if len(iter_deltas) > warmup:
        measured = iter_deltas[warmup:]
    else:
        measured = iter_deltas

    if not measured:
        print("ERROR: no iteration timing data", file=sys.stderr)
        sys.exit(1)

    ms_per_iter = sum(measured) / len(measured)
    samples_per_sec = gbs / (ms_per_iter / 1000.0)

    valid_evals = [(s, v) for s, v in eval_events if v > TARGET]

    if len(valid_evals) < 2:
        if eval_events:
            last_s, last_v = eval_events[-1]
            if last_v <= TARGET:
                print(f"{last_s / samples_per_sec:.1f}\t{last_s}\t{samples_per_sec:.1f}\t1.00")
            else:
                if len(eval_events) >= 2:
                    s1, v1 = eval_events[-2]
                    s2, v2 = eval_events[-1]
                    if v1 != v2:
                        slope = (v2 - v1) / (s2 - s1)
                        if slope < 0:
                            projected_samples = s2 + (TARGET - v2) / slope
                            projected_ttt = projected_samples / samples_per_sec
                            print(f"{projected_ttt:.1f}\t{projected_samples:.0f}\t{samples_per_sec:.1f}\t0.50")
                            sys.exit(0)
                print(f"-1\t-1\t{samples_per_sec:.1f}\t0.00")
        else:
            print(f"-1\t-1\t{samples_per_sec:.1f}\t0.00")
        sys.exit(0)

    xs = [math.log(s) for s, v in valid_evals]
    ys = [math.log(v - TARGET) for s, v in valid_evals]

    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_x2 = sum(x * x for x in xs)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        print(f"-1\t-1\t{samples_per_sec:.1f}\t0.00")
        sys.exit(0)

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    if slope >= 0:
        print(f"-1\t-1\t{samples_per_sec:.1f}\t0.00")
        sys.exit(0)

    epsilon = 0.01
    target_log = math.log(epsilon)
    projected_samples = math.exp((target_log - intercept) / slope)

    if projected_samples < 0:
        print(f"-1\t-1\t{samples_per_sec:.1f}\t0.00")
        sys.exit(0)

    projected_ttt = projected_samples / samples_per_sec

    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"{projected_ttt:.1f}\t{projected_samples:.0f}\t{samples_per_sec:.1f}\t{r_squared:.2f}")


# -------------------------------------------------------------------------
# parse-trial-result: turn a TRIAL_RESULT line into shell assignments
# Usage: eval "$(parse_trial_result "$line")"
# -------------------------------------------------------------------------

def cmd_parse_trial_result(args):
    line = args.line
    if not line.startswith("TRIAL_RESULT"):
        print("# not a TRIAL_RESULT line")
        return
    parts = line.split()
    for part in parts[1:]:
        key, _, val = part.partition("=")
        print(f'TRIAL_{key.upper()}="{val}"')


# -------------------------------------------------------------------------
# compute-gain-pct: (baseline - current) / baseline * 100 (positive = faster)
# -------------------------------------------------------------------------

def cmd_compute_gain_pct(args):
    b = float(args.baseline)
    c = float(args.current)
    if b > 0:
        print(f"{(b - c) / b * 100:.2f}")
    else:
        print("0.00")


# -------------------------------------------------------------------------
# compute-loss-efficiency: (first_eval - last_eval) / wall_seconds (Layer 2)
# -------------------------------------------------------------------------

def cmd_compute_loss_efficiency(args):
    eval_events = []
    first_ts = None

    for data in _iter_mllog(args.log_file):
        if first_ts is None:
            first_ts = data.get("time_ms", 0)
        if data.get("key") == "eval_accuracy":
            sc = data.get("metadata", {}).get("samples_count")
            val = data.get("value")
            ts = data.get("time_ms", 0)
            if sc is not None and val is not None:
                eval_events.append((int(sc), float(val), float(ts)))

    if len(eval_events) < 2 or first_ts is None:
        print("0.0")
        sys.exit(0)

    first_eval = eval_events[0]
    last_eval = eval_events[-1]
    delta_loss = first_eval[1] - last_eval[1]
    delta_time_s = (last_eval[2] - first_ts) / 1000.0

    if delta_time_s <= 0:
        print("0.0")
        sys.exit(0)

    efficiency = delta_loss / delta_time_s
    print(f"{efficiency:.6f}")


# -------------------------------------------------------------------------
# compute-ttt-gain-pct: (baseline_ttt - candidate_ttt) / baseline_ttt * 100
# -------------------------------------------------------------------------

def cmd_compute_ttt_gain_pct(args):
    b = float(args.baseline)
    c = float(args.candidate)
    if b > 0 and c > 0:
        print(f"{(b - c) / b * 100:.2f}")
    else:
        print("0.00")


# -------------------------------------------------------------------------
# detect-nan: scan train_loss values for NaN/Inf; print NaN_DETECTED or OK
# -------------------------------------------------------------------------

def cmd_detect_nan(args):
    for data in _iter_mllog(args.log_file):
        if data.get("key") == "train_loss":
            v = data.get("value")
            if v is not None and (math.isnan(float(v)) or math.isinf(float(v))):
                print("NaN_DETECTED")
                return
    print("OK")


# -------------------------------------------------------------------------
# extract-eval-trajectory: tab-separated samples_count/value/time_ms
# -------------------------------------------------------------------------

def cmd_extract_eval_trajectory(args):
    for data in _iter_mllog(args.log_file):
        if data.get("key") == "eval_accuracy":
            md = data.get("metadata", {})
            sc = md.get("samples_count", "?")
            val = data.get("value")
            time_ms = data.get("time_ms", 0)
            if val is not None:
                print(f"{sc}\t{val:.6f}\t{time_ms}")


# -------------------------------------------------------------------------
# compute-eval-overhead: total eval seconds, num_evals, avg_eval_seconds
# -------------------------------------------------------------------------

def cmd_compute_eval_overhead(args):
    events = list(_iter_mllog(args.log_file))
    eval_starts = [e["time_ms"] for e in events if e.get("key") == "eval_start"]
    eval_stops = [e["time_ms"] for e in events if e.get("key") == "eval_stop"]

    total_eval_ms = 0
    for start, stop in zip(eval_starts, eval_stops):
        total_eval_ms += stop - start

    num_evals = len(eval_starts)
    avg_eval_ms = total_eval_ms / num_evals if num_evals > 0 else 0

    print(f"{total_eval_ms / 1000:.1f}\t{num_evals}\t{avg_eval_ms / 1000:.1f}")


# -------------------------------------------------------------------------
# decompress-gz-trace: gunzip a *.pt.trace.json.gz into plain JSON
# (replaces inline heredocs in discover_trace())
# -------------------------------------------------------------------------

def cmd_decompress_gz_trace(args):
    with gzip.open(args.src, "rt") as f:
        trace = json.load(f)
    with open(args.dst, "w") as f:
        json.dump(trace, f)


# -------------------------------------------------------------------------
# rpd-to-chrome: convert a ROCm RPD SQLite database to Chrome trace JSON
# -------------------------------------------------------------------------

def cmd_rpd_to_chrome(args):
    import sqlite3

    rpd_path = args.rpd_path
    json_path = args.json_path

    # Method 1: rpd2tracing from rocmProfileData package
    try:
        from rpd2tracing import rpd2tracing  # type: ignore
        rpd2tracing(rpd_path, json_path)
        size_mb = os.path.getsize(json_path) / 1024 / 1024
        print(f"RPD\u2192JSON via rpd2tracing: {size_mb:.1f}MB", file=sys.stderr)
        return
    except ImportError:
        pass

    # Method 2: Direct SQLite query (fallback)
    conn = sqlite3.connect(rpd_path)
    cursor = conn.cursor()

    tables = {row[0] for row in cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    trace_events = []

    if "rocpd_api" in tables and "rocpd_op" in tables:
        for row in cursor.execute(
            """
            SELECT api.Name, op.KernelName, op.BeginNs, op.EndNs, op.gpuId,
                   op.queueId, op.pid, op.tid
            FROM rocpd_op op
            LEFT JOIN rocpd_api api ON op.api_id = api.id
            ORDER BY op.BeginNs
            LIMIT 200000
            """
        ):
            api_name, kernel_name, begin_ns, end_ns, gpu_id, queue_id, pid, tid = row
            name = kernel_name or api_name or "unknown"
            dur_us = (end_ns - begin_ns) / 1000 if begin_ns and end_ns else 0
            ts_us = begin_ns / 1000 if begin_ns else 0
            trace_events.append({
                "name": name, "cat": "kernel", "ph": "X",
                "ts": ts_us, "dur": dur_us,
                "pid": pid or gpu_id or 0, "tid": tid or queue_id or 0,
            })

    elif "api" in tables and "op" in tables:
        for row in cursor.execute(
            """
            SELECT Name, KernelName, BeginNs, EndNs, gpuId, pid, tid
            FROM op LEFT JOIN api ON op.api_id = api.id
            ORDER BY BeginNs LIMIT 200000
            """
        ):
            api_name, kernel_name, begin_ns, end_ns, gpu_id, pid, tid = row
            name = kernel_name or api_name or "unknown"
            dur_us = (end_ns - begin_ns) / 1000 if begin_ns and end_ns else 0
            ts_us = begin_ns / 1000 if begin_ns else 0
            trace_events.append({
                "name": name, "cat": "kernel", "ph": "X",
                "ts": ts_us, "dur": dur_us,
                "pid": pid or gpu_id or 0, "tid": tid or 0,
            })

    conn.close()

    if not trace_events:
        print(f"WARNING: RPD file has no ops (tables: {tables})", file=sys.stderr)
        sys.exit(1)

    with open(json_path, "w") as f:
        json.dump({"traceEvents": trace_events}, f)

    size_mb = os.path.getsize(json_path) / 1024 / 1024
    print(
        f"RPD\u2192JSON via SQLite fallback: {len(trace_events)} events, {size_mb:.1f}MB",
        file=sys.stderr,
    )


# -------------------------------------------------------------------------
# filter-trace: shrink Chrome trace to GPU-relevant categories
# -------------------------------------------------------------------------

def cmd_filter_trace(args):
    src = args.src
    dst = args.dst

    opener = gzip.open if src.endswith(".gz") else open
    with opener(src, "rt") as f:
        trace = json.load(f)

    keep = {
        "kernel", "gpu_memcpy", "gpu_memset", "cpu_op", "cuda_runtime",
        "ac2g", "user_annotation", "gpu_user_annotation",
    }
    orig = len(trace["traceEvents"])
    trace["traceEvents"] = [e for e in trace["traceEvents"] if e.get("cat", "") in keep]
    filt = len(trace["traceEvents"])

    writer = gzip.open if dst.endswith(".gz") else open
    with writer(dst, "wt") as f:
        json.dump(trace, f)

    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"Filtered: {orig} -> {filt} events ({size_mb:.1f}MB)")


# -------------------------------------------------------------------------
# argparse wiring
# -------------------------------------------------------------------------

def _add_log(p):
    p.add_argument("log_file", help="path to raw training log")


def _add_log_with_warmup_measure(p):
    p.add_argument("log_file")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measure", type=int, default=5)


def _add_log_and_field(p):
    p.add_argument("log_file")
    p.add_argument("field", help="MLLOG key to extract")


def _add_log_and_expected(p):
    p.add_argument("log_file")
    p.add_argument("expected", type=int, help="expected GBS")


def _add_log_and_gbs(p):
    p.add_argument("log_file")
    p.add_argument("gbs", type=int)


def _add_line(p):
    p.add_argument("line", help="raw TRIAL_RESULT line")


def _add_baseline_current(p):
    p.add_argument("baseline")
    p.add_argument("current")


def _add_baseline_candidate(p):
    p.add_argument("baseline")
    p.add_argument("candidate")


def _add_src_dst(p):
    p.add_argument("src")
    p.add_argument("dst")


def _add_rpd_json(p):
    p.add_argument("rpd_path")
    p.add_argument("json_path")


COMMANDS = {
    "extract-ms-per-iter": (cmd_extract_ms_per_iter, _add_log_with_warmup_measure),
    "extract-mllog-field": (cmd_extract_mllog_field, _add_log_and_field),
    "verify-gbs": (cmd_verify_gbs, _add_log_and_expected),
    "extract-losses": (cmd_extract_losses, _add_log),
    "extract-ttt": (cmd_extract_ttt, _add_log),
    "project-ttt": (cmd_project_ttt, _add_log_and_gbs),
    "parse-trial-result": (cmd_parse_trial_result, _add_line),
    "compute-gain-pct": (cmd_compute_gain_pct, _add_baseline_current),
    "compute-loss-efficiency": (cmd_compute_loss_efficiency, _add_log),
    "compute-ttt-gain-pct": (cmd_compute_ttt_gain_pct, _add_baseline_candidate),
    "detect-nan": (cmd_detect_nan, _add_log),
    "extract-eval-trajectory": (cmd_extract_eval_trajectory, _add_log),
    "compute-eval-overhead": (cmd_compute_eval_overhead, _add_log),
    "decompress-gz-trace": (cmd_decompress_gz_trace, _add_src_dst),
    "rpd-to-chrome": (cmd_rpd_to_chrome, _add_rpd_json),
    "filter-trace": (cmd_filter_trace, _add_src_dst),
}


def main():
    parser = argparse.ArgumentParser(
        description="MLPerf optimization utilities (called by common.sh wrappers).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, (fn, add_args) in COMMANDS.items():
        p = sub.add_parser(name)
        add_args(p)
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
