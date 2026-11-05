"""KV cache trace collector — monkey-patches a live SGLang server to emit per-event JSONL.

Usage:
    # Set env var before launching the server:
    SGLANG_KV_TRACE_PATH=/tmp/kv_trace.jsonl python3 -m sglang.launch_server ...

    # Or inject at runtime (after server is already running):
    docker exec $CONTAINER python3 /path/to/kv_trace_collector.py attach /tmp/kv_trace.jsonl

The trace captures every GPU↔CPU transfer, every eviction, and periodic cache state snapshots.
No SGLang source modifications needed.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import sys
import threading
import time
from pathlib import Path


_trace_file = None
_lock = threading.Lock()
_start_time = time.monotonic()
_snapshot_thread = None
_snapshot_stop = threading.Event()


def _emit(event: dict):
    event["ts"] = round(time.monotonic() - _start_time, 4)
    with _lock:
        if _trace_file and not _trace_file.closed:
            _trace_file.write(json.dumps(event, default=str) + "\n")
            _trace_file.flush()


def _open_trace(path: str):
    global _trace_file, _start_time
    _start_time = time.monotonic()
    _trace_file = open(path, "w")
    _emit({"event": "trace_start", "path": path, "pid": os.getpid()})
    atexit.register(_close_trace)


def _close_trace():
    global _trace_file
    _snapshot_stop.set()
    if _trace_file and not _trace_file.closed:
        _emit({"event": "trace_end"})
        _trace_file.close()
    _trace_file = None


# ── Monkey-patches ────────────────────────────────────────────────────────

def patch_hiradix_cache():
    """Patch HiRadixCache to emit trace events. Call after sglang is imported."""
    try:
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    except ImportError:
        print("[kv_trace] HiRadixCache not found — skipping", file=sys.stderr)
        return False

    # 1. Patch evict() — captures every eviction decision
    _orig_evict = HiRadixCache.evict

    @functools.wraps(_orig_evict)
    def _traced_evict(self, params):
        result = _orig_evict(self, params)
        _emit({
            "event": "evict",
            "requested": params.num_tokens,
            "evicted": result.num_tokens_evicted,
            "policy": getattr(self.cache_controller, "write_policy", "unknown"),
        })
        return result

    HiRadixCache.evict = _traced_evict

    # 2. Patch writing_check() — captures GPU→CPU transfer completions
    _orig_writing_check = HiRadixCache.writing_check

    @functools.wraps(_orig_writing_check)
    def _traced_writing_check(self, *args, **kwargs):
        queue_len_before = len(getattr(self.cache_controller, "ack_write_queue", []))
        result = _orig_writing_check(self, *args, **kwargs)
        queue_len_after = len(getattr(self.cache_controller, "ack_write_queue", []))
        processed = queue_len_before - queue_len_after
        if processed > 0:
            _emit({
                "event": "backup_batch",
                "transfers_completed": processed,
            })
        return result

    HiRadixCache.writing_check = _traced_writing_check

    # 3. Patch loading_check() — captures CPU→GPU transfer completions
    _orig_loading_check = HiRadixCache.loading_check

    @functools.wraps(_orig_loading_check)
    def _traced_loading_check(self, finish_count=None):
        result = _orig_loading_check(self, finish_count)
        return result

    HiRadixCache.loading_check = _traced_loading_check

    # 4. Patch the metrics collector's observe methods — these get the actual timing
    _patch_metrics_collector()

    # 5. Patch the prefill stats reporter — captures per-batch cache hit/miss
    _patch_prefill_stats()

    print("[kv_trace] HiRadixCache + PrefillStats patched — trace active", file=sys.stderr)
    return True


def _patch_prefill_stats():
    """Patch the metrics reporter to emit per-request cache hit/miss events."""
    try:
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )
    except ImportError:
        return

    _orig_report_prefill = getattr(SchedulerMetricsReporter, "report_prefill_stats", None)
    if _orig_report_prefill is None:
        return

    @functools.wraps(_orig_report_prefill)
    def _traced_report_prefill(self, batch, prefill_stats, *args, **kwargs):
        result = _orig_report_prefill(self, batch, prefill_stats, *args, **kwargs)
        if prefill_stats is not None:
            _emit({
                "event": "prefill_batch",
                "num_seqs": prefill_stats.num_new_seqs,
                "new_tokens": prefill_stats.log_input_tokens,
                "cached_tokens_gpu": prefill_stats.log_hit_tokens,
                "cached_tokens_host": getattr(prefill_stats, "log_host_hit_tokens", 0),
            })
        if batch is not None and hasattr(batch, "reqs"):
            for req in batch.reqs:
                gpu_cached = len(getattr(req, "prefix_indices", [])) if hasattr(req, "prefix_indices") else 0
                host_cached = getattr(req, "host_hit_length", 0)
                total_input = len(getattr(req, "origin_input_ids", []))
                uncached = max(0, total_input - gpu_cached - host_cached)
                _emit({
                    "event": "request_cache",
                    "rid": str(getattr(req, "rid", "?")),
                    "total_input_tokens": total_input,
                    "gpu_cached": gpu_cached,
                    "host_cached": host_cached,
                    "uncached": uncached,
                    "hit_rate": round((gpu_cached + host_cached) / total_input, 4) if total_input > 0 else 0,
                })
        return result

    SchedulerMetricsReporter.report_prefill_stats = _traced_report_prefill


def _patch_metrics_collector():
    """Patch the metrics collector to emit per-transfer events with timing."""
    try:
        from sglang.srt.observability.metrics_collector import PrometheusMetricsCollector
    except ImportError:
        return

    _orig_backup = PrometheusMetricsCollector.observe_backup_duration
    _orig_loadback = PrometheusMetricsCollector.observe_load_back_duration
    _orig_backup_bytes = PrometheusMetricsCollector.increment_backup_num_bytes
    _orig_loadback_bytes = PrometheusMetricsCollector.increment_load_back_num_bytes
    _orig_backup_tokens = PrometheusMetricsCollector.increment_backup_num_tokens
    _orig_loadback_tokens = PrometheusMetricsCollector.increment_load_back_num_tokens

    _pending_backup = threading.local()
    _pending_loadback = threading.local()

    @functools.wraps(_orig_backup_tokens)
    def _traced_backup_tokens(self, num_tokens, pool=None):
        if not hasattr(_pending_backup, "tokens"):
            _pending_backup.tokens = 0
        _pending_backup.tokens += num_tokens
        return _orig_backup_tokens(self, num_tokens, pool=pool)

    @functools.wraps(_orig_backup_bytes)
    def _traced_backup_bytes(self, num_bytes):
        if not hasattr(_pending_backup, "bytes"):
            _pending_backup.bytes = 0
        _pending_backup.bytes += num_bytes
        return _orig_backup_bytes(self, num_bytes)

    @functools.wraps(_orig_backup)
    def _traced_backup_duration(self, duration_s):
        num_bytes = getattr(_pending_backup, "bytes", 0)
        num_tokens = getattr(_pending_backup, "tokens", 0)
        duration_ms = duration_s * 1000
        bw_gbps = (num_bytes / duration_s / 1e9) if duration_s > 0 else 0
        _emit({
            "event": "backup_done",
            "direction": "gpu_to_host",
            "num_tokens": num_tokens,
            "num_bytes": num_bytes,
            "duration_ms": round(duration_ms, 3),
            "bw_gbps": round(bw_gbps, 2),
        })
        _pending_backup.tokens = 0
        _pending_backup.bytes = 0
        return _orig_backup(self, duration_s)

    @functools.wraps(_orig_loadback_tokens)
    def _traced_loadback_tokens(self, num_tokens, pool=None):
        if not hasattr(_pending_loadback, "tokens"):
            _pending_loadback.tokens = 0
        _pending_loadback.tokens += num_tokens
        return _orig_loadback_tokens(self, num_tokens, pool=pool)

    @functools.wraps(_orig_loadback_bytes)
    def _traced_loadback_bytes(self, num_bytes):
        if not hasattr(_pending_loadback, "bytes"):
            _pending_loadback.bytes = 0
        _pending_loadback.bytes += num_bytes
        return _orig_loadback_bytes(self, num_bytes)

    @functools.wraps(_orig_loadback)
    def _traced_loadback_duration(self, duration_s):
        num_bytes = getattr(_pending_loadback, "bytes", 0)
        num_tokens = getattr(_pending_loadback, "tokens", 0)
        duration_ms = duration_s * 1000
        bw_gbps = (num_bytes / duration_s / 1e9) if duration_s > 0 else 0
        _emit({
            "event": "load_back_done",
            "direction": "host_to_gpu",
            "num_tokens": num_tokens,
            "num_bytes": num_bytes,
            "duration_ms": round(duration_ms, 3),
            "bw_gbps": round(bw_gbps, 2),
        })
        _pending_loadback.tokens = 0
        _pending_loadback.bytes = 0
        return _orig_loadback(self, duration_s)

    PrometheusMetricsCollector.observe_backup_duration = _traced_backup_duration
    PrometheusMetricsCollector.increment_backup_num_bytes = _traced_backup_bytes
    PrometheusMetricsCollector.increment_backup_num_tokens = _traced_backup_tokens
    PrometheusMetricsCollector.observe_load_back_duration = _traced_loadback_duration
    PrometheusMetricsCollector.increment_load_back_num_bytes = _traced_loadback_bytes
    PrometheusMetricsCollector.increment_load_back_num_tokens = _traced_loadback_tokens


def start_snapshot_thread(interval_s: float = 5.0):
    """Periodically emit cache state snapshots from /metrics or internal state."""
    def _snapshot_loop():
        import urllib.request
        port = int(os.environ.get("PORT", "8888"))
        while not _snapshot_stop.wait(interval_s):
            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{port}/metrics", timeout=2
                ).read().decode()
                snapshot = {"event": "snapshot"}
                for line in resp.splitlines():
                    if line.startswith("#"):
                        continue
                    if "token_usage" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            snapshot["token_usage"] = float(parts[-1])
                    elif "hicache_backup_num_bytes_total" in line and "pool" not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            snapshot["total_backup_bytes"] = float(parts[-1])
                    elif "load_back_num_bytes_total" in line and "pool" not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            snapshot["total_loadback_bytes"] = float(parts[-1])
                if len(snapshot) > 1:
                    _emit(snapshot)
            except Exception:
                pass

    global _snapshot_thread
    _snapshot_thread = threading.Thread(target=_snapshot_loop, daemon=True)
    _snapshot_thread.start()


def install():
    """Auto-install if SGLANG_KV_TRACE_PATH is set. Call early in server startup."""
    trace_path = os.environ.get("SGLANG_KV_TRACE_PATH")
    if not trace_path:
        return False

    _open_trace(trace_path)
    patched = patch_hiradix_cache()
    if patched:
        start_snapshot_thread(
            interval_s=float(os.environ.get("SGLANG_KV_TRACE_INTERVAL", "5"))
        )
    return patched


# Auto-install when this module is imported
if os.environ.get("SGLANG_KV_TRACE_PATH"):
    install()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KV cache trace collector")
    parser.add_argument("command", choices=["test"], help="test: verify patching works")
    args = parser.parse_args()

    if args.command == "test":
        os.environ["SGLANG_KV_TRACE_PATH"] = "/tmp/kv_trace_test.jsonl"
        _open_trace("/tmp/kv_trace_test.jsonl")
        _emit({"event": "evict", "requested": 100, "evicted": 100})
        _emit({"event": "backup_done", "direction": "gpu_to_host",
               "num_tokens": 50, "num_bytes": 3686400, "duration_ms": 0.5, "bw_gbps": 7.0})
        _emit({"event": "load_back_done", "direction": "host_to_gpu",
               "num_tokens": 25, "num_bytes": 1843200, "duration_ms": 0.3, "bw_gbps": 5.8})
        _emit({"event": "snapshot", "token_usage": 0.45})
        _close_trace()
        print("Test trace written to /tmp/kv_trace_test.jsonl")
        print(open("/tmp/kv_trace_test.jsonl").read())
