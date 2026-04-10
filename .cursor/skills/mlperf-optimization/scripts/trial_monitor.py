#!/usr/bin/env python3
"""
MLPerf Trial Monitor — stdin log filter with real-time progress and anomaly detection.

Reads training output from stdin, writes filtered output to stdout, and always
preserves the raw unfiltered log for debugging.

Usage:
    torchrun ... 2>&1 | python3 trial_monitor.py --raw-log /tmp/raw.log --max-iters 100 --label baseline

Pass-through rules (preserves MLPerf compliance):
    - :::MLLOG lines          → always pass (submission-critical)
    - RESULT, lines           → always pass (wall-clock timing)
    - STARTING/ENDING TIMING  → always pass
    - [MLPERF] messages       → always pass
    - ERROR/CRITICAL/Traceback → always pass (debugging)
    - OOM                     → always pass

Everything else (Megatron INFO/DEBUG, Python warnings, torch.distributed noise)
is suppressed from stdout but preserved in --raw-log.
"""

import argparse
import json
import math
import re
import sys
import time


PASS_PATTERNS = [
    re.compile(r"^:::MLLOG\s"),
    re.compile(r"^RESULT,"),
    re.compile(r"^STARTING TIMING"),
    re.compile(r"^ENDING TIMING"),
    re.compile(r"\[MLPERF\]"),
    re.compile(r"\bERROR\b.*(?:failed|exception|abort|cannot|fatal|invalid)", re.IGNORECASE),
    re.compile(r"^Traceback \(most recent"),
    re.compile(r"^\w+Error:"),
    re.compile(r"CRITICAL", re.IGNORECASE),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"torch\.cuda\.OutOfMemoryError"),
    re.compile(r"RuntimeError:"),
]

MLLOG_PREFIX = ":::MLLOG "


def parse_mllog(line: str):
    """Parse a :::MLLOG JSON line. Returns dict or None."""
    if not line.startswith(MLLOG_PREFIX):
        return None
    try:
        return json.loads(line[len(MLLOG_PREFIX):])
    except (json.JSONDecodeError, ValueError):
        return None


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(description="MLPerf trial log filter")
    parser.add_argument("--raw-log", required=True, help="Path to write raw unfiltered log")
    parser.add_argument("--max-iters", type=int, default=0, help="Expected iteration count (for progress display)")
    parser.add_argument("--label", default="trial", help="Trial label for display")
    args = parser.parse_args()

    raw_fp = open(args.raw_log, "w", buffering=1)

    start_time = time.time()
    iter_count = 0
    prev_time_ms = None
    iter_times = []
    last_loss = None
    last_lr = None
    gbs = None
    eval_losses = []
    run_start_ms = None
    run_stop_ms = None
    run_status = None
    nan_detected = False

    max_label = f"/{args.max_iters}" if args.max_iters > 0 else ""

    def should_pass(line: str) -> bool:
        return any(p.search(line) for p in PASS_PATTERNS)

    try:
        for raw_line in sys.stdin:
            raw_fp.write(raw_line)
            raw_fp.flush()

            line = raw_line.rstrip("\n")

            mllog = parse_mllog(line)
            if mllog:
                key = mllog.get("key", "")

                if key == "train_loss":
                    iter_count += 1
                    loss_val = mllog.get("value")
                    meta = mllog.get("metadata", {})
                    cur_time_ms = mllog.get("time_ms")
                    last_lr = meta.get("lr")
                    last_loss = loss_val

                    if prev_time_ms is not None and cur_time_ms is not None:
                        delta = cur_time_ms - prev_time_ms
                        if delta > 0:
                            iter_times.append(delta)
                    prev_time_ms = cur_time_ms

                    if loss_val is not None:
                        try:
                            fval = float(loss_val)
                            if math.isnan(fval) or math.isinf(fval):
                                nan_detected = True
                                alert = f"[ALERT] NaN/Inf loss detected at iter {iter_count}: {loss_val}"
                                print(alert, flush=True)
                        except (ValueError, TypeError):
                            pass

                    elapsed = format_elapsed(time.time() - start_time)
                    lr_str = f" lr={last_lr:.2e}" if last_lr is not None else ""
                    loss_str = f" loss={loss_val:.4f}" if loss_val is not None else ""
                    progress = f"[ITER {iter_count}{max_label}]{loss_str}{lr_str} | elapsed={elapsed}"
                    print(progress, flush=True)
                    continue

                elif key == "global_batch_size":
                    gbs = mllog.get("value")

                elif key == "eval_accuracy":
                    eval_val = mllog.get("value")
                    if eval_val is not None:
                        eval_losses.append(eval_val)
                        elapsed = format_elapsed(time.time() - start_time)
                        target_gap = eval_val - 3.34
                        marker = " *** TARGET REACHED ***" if target_gap <= 0 else ""
                        print(f"[EVAL] loss={eval_val:.4f} (gap={target_gap:+.4f} to 3.34) | elapsed={elapsed}{marker}",
                              flush=True)

                elif key == "run_start":
                    run_start_ms = mllog.get("time_ms")

                elif key == "run_stop":
                    run_stop_ms = mllog.get("time_ms")
                    run_status = mllog.get("metadata", {}).get("status", "unknown")
                    if run_status == "success":
                        print("\n*** CONVERGENCE: Target eval loss 3.34 REACHED ***", flush=True)
                    elif run_status == "aborted":
                        final_eval = eval_losses[-1] if eval_losses else "N/A"
                        print(f"\n*** RUN ENDED: status=aborted, final_eval_loss={final_eval} (target: 3.34) ***",
                              flush=True)

            if should_pass(line):
                print(line, flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        raw_fp.close()

    warmup = min(5, len(iter_times) // 2)
    if len(iter_times) > warmup:
        measured = iter_times[warmup:]
        avg_ms = sum(measured) / len(measured)
    elif iter_times:
        avg_ms = sum(iter_times) / len(iter_times)
    else:
        avg_ms = 0.0

    ttt_s = 0.0
    ttt_str = ""
    if run_start_ms and run_stop_ms:
        ttt_s = (run_stop_ms - run_start_ms) / 1000.0
        ttt_str = f" | ttt={ttt_s:.1f}s ({run_status})"

    eval_str = ""
    if eval_losses:
        eval_str = f" | eval_loss={eval_losses[-1]:.4f}"

    gbs_str = f" | GBS={gbs}" if gbs is not None else ""
    nan_str = " | NaN_DETECTED" if nan_detected else ""
    loss_str = f" | last_loss={last_loss:.4f}" if last_loss is not None else ""

    print(f"\n=== Trial Summary [{args.label}]: {avg_ms:.1f} ms/iter"
          f" | {iter_count} iters{gbs_str}{loss_str}{eval_str}{ttt_str}{nan_str} ===",
          flush=True)

    if run_status == "success":
        print(f"=== CONVERGED: time-to-train = {ttt_s:.1f}s"
              f" ({ttt_s/60:.1f} min) | target 3.34 reached ===", flush=True)
    elif run_status == "aborted" and eval_losses:
        print(f"=== DID NOT CONVERGE: final eval_loss={eval_losses[-1]:.4f}"
              f" (target: 3.34, gap: {eval_losses[-1]-3.34:+.4f}) ===", flush=True)

    status = "nan" if nan_detected else "ok"
    if avg_ms == 0.0 and iter_count == 0:
        status = "no_data"
    if run_status == "success":
        status = "converged"

    print(f"TRIAL_RESULT label={args.label} ms_per_iter={avg_ms:.1f} gbs={gbs or 0}"
          f" last_loss={last_loss if last_loss is not None else 0:.6f}"
          f" iters={iter_count} ttt={ttt_s:.1f} run_status={run_status or 'unknown'}"
          f" status={status}",
          flush=True)


if __name__ == "__main__":
    main()
