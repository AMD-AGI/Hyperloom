#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""
Push session_breakdown.json files to the production perf-leaderboard API.

Pipeline per file
-----------------
    1. Read JSON                                    (./remote_sessions/<task>/session_breakdown.json)
    2. transform_to_session_summary_v2.transform(data)  -> {source, data, ...}
    3. import_session_breakdown.extract_row(wrapped.data) -> row dict (matches
       perf_runs DTO 1:1 -- model_name / framework / image / unique_key /
       claw_session_id / gain (validated) / roofline / raw_data / ...)
    4. POST row as JSON body to /perf-leaderboard/api/v1/perf-runs

Concurrency
-----------
    Default 5 workers. 429 / 5xx use exponential backoff with full jitter.
    Successful upserts and per-file errors are appended to a JSONL ledger so
    you can resume / inspect a partial run.

Usage
-----
    # Dry-run (no POST): print body sizes and unique_keys
    python scripts/post_perf_runs.py --dir ./remote_sessions --dry-run

    # Upload a single file (smoke test)
    python scripts/post_perf_runs.py ./remote_sessions/<task>/session_breakdown.json

    # Full batch
    python scripts/post_perf_runs.py --dir ./remote_sessions

    # Custom workers / endpoint / token
    python scripts/post_perf_runs.py --dir ./remote_sessions \\
        --workers 5 \\
        --endpoint https://core42.primus-safe.amd.com/perf-leaderboard/api/v1/perf-runs \\
        --token "Bearer ..."
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reuse the same row extraction + V2 transform logic that import_session_breakdown.py uses.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from import_session_breakdown import (  # noqa: E402
    looks_like_session_breakdown,
    looks_like_v1_flat_schema,
    looks_like_universal_schema,
    migrate_v1_to_v2,
    migrate_universal_to_v2,
    extract_row,
)
from transform_to_session_summary_v2 import transform  # noqa: E402


DEFAULT_ENDPOINT = "https://core42.primus-safe.amd.com/perf-leaderboard-dev/api/v1/perf-runs"
DEFAULT_TOKEN = "Bearer 01e00f5d54dcc9a237bee1a34fe6d741d8d1829545d2d8a1b819538ca6733889"
DEFAULT_WORKERS = 5
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 4

# stop_reason values for true crashes only. 'signal'/'time_exhausted'/'killed'/
# 'timeout' are NOT rejected — they're graceful budget-elapsed shutdowns with
# valid baseline + final metrics.
_CRASH_STOP_REASONS = {
    "exception", "error", "failed", "baseline_failed",
    "oom_killed", "oom", "segfault",
}


def is_complete_session(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate that a session is complete and did not crash.

    A "good" session needs baseline + final ``throughput_tok_s_per_gpu`` > 0
    and a ``stop_reason`` not in ``_CRASH_STOP_REASONS``
    (``signal`` / ``time_exhausted`` are kept).

    Args:
        data: The session breakdown data dict.

    Returns:
        An ``(ok, reason_if_rejected)`` tuple; ``reason`` is ``None`` when
        accepted.
    """
    baseline = data.get("baseline") or {}
    final = data.get("final") or {}
    session = data.get("session") or {}

    b_tp = baseline.get("throughput_tok_s_per_gpu")
    if not isinstance(b_tp, (int, float)) or b_tp <= 0:
        return False, f"incomplete:baseline_throughput={b_tp!r}"

    f_tp = final.get("throughput_tok_s_per_gpu")
    if not isinstance(f_tp, (int, float)) or f_tp <= 0:
        return False, f"incomplete:final_throughput={f_tp!r}"

    stop_reason = (session.get("stop_reason") or "").strip().lower()
    if stop_reason in _CRASH_STOP_REASONS:
        return False, f"crashed:stop_reason={stop_reason!r}"

    return True, None


# ---------------------------------------------------------------------------
# Body construction
# ---------------------------------------------------------------------------

def build_body(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build the perf_runs POST body from a breakdown file.

    Accepts four layouts (strictest first): bare V2 breakdown, V2 wrapper
    (``{source, data}``), legacy V1 flat schema, and the universal fallback.

    Args:
        path: Path to the session breakdown JSON file.

    Returns:
        A ``(body, error_msg)`` tuple; ``body`` is ``None`` on error and
        ``error_msg`` carries the reason. The body keys match the perf_runs
        DTO 1:1.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"parse_failed:{e}"

    # Accept four layouts, strictest first: bare V2 breakdown, V2 wrapper
    # ({source, data}), legacy V1 flat schema, universal fallback.
    if looks_like_session_breakdown(data):
        pass
    elif isinstance(data, dict) and isinstance(data.get("data"), dict) and \
            looks_like_session_breakdown(data["data"]):
        data = data["data"]
    elif looks_like_v1_flat_schema(data):
        data = migrate_v1_to_v2(data)
    elif looks_like_universal_schema(data):
        migrated = migrate_universal_to_v2(data)
        if migrated is None:
            return None, "universal_migrate_failed"
        data = migrated
    else:
        return None, "not_a_session_breakdown"

    ok, reason = is_complete_session(data)
    if not ok:
        return None, reason

    wrapped = transform(data)
    enriched = wrapped["data"]

    try:
        row = extract_row(enriched)
    except Exception as e:
        return None, f"extract_row_failed:{e}"

    # Strict client-side validation so no leaderboard column lands NULL/empty:
    # reject blank strings and null numerics (0 gain is a valid no-op result).
    required_nonblank = ("model_name", "framework", "image", "prec", "category",
                         "unique_key", "claw_session_id", "status")
    blank = [k for k in required_nonblank
             if not (isinstance(row.get(k), str) and row.get(k).strip())]
    if blank:
        return None, f"blank_field:{','.join(blank)}"

    required_numeric = ("gain", "baseline_tok_per_s_per_gpu", "opt_tok_per_s_per_gpu",
                        "tp", "isl", "osl", "conc", "duration_seconds",
                        "kernel_gain", "param_gain", "backend_gain")
    null_num = [k for k in required_numeric if row.get(k) is None]
    if null_num:
        return None, f"null_numeric:{','.join(null_num)}"

    # raw_data.session.image must be backfilled by enrich_raw_data; if still
    # missing it's a bug, not a data issue.
    raw_session_image = (row.get("raw_data") or {}).get("session", {}).get("image")
    if not raw_session_image:
        return None, "raw_data.session.image_missing"

    return row, None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

# 500 excluded on purpose: the server returns it for deterministic business
# errors (e.g. duplicate unique_key); transient outages surface as 502/503/504.
_RETRIABLE_STATUS = {408, 425, 429, 502, 503, 504}


def post_once(endpoint: str, token: str, body: Dict[str, Any], timeout: float) -> Tuple[int, str]:
    """Issue a single POST request to the perf-leaderboard API.

    Low-level connection errors are mapped to status ``599`` so the retry loop
    can decide whether to retry.

    Args:
        endpoint (str): Full URL to POST to.
        token (str): Value for the ``Authorization`` header.
        body (Dict[str, Any]): JSON-serializable request body.
        timeout (float): Request timeout in seconds.

    Returns:
        Tuple[int, str]: The HTTP status code and response text.
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Authorization": token,
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return e.code, text
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        # 599 = low-level connection error, for the retry loop to decide.
        return 599, f"{type(e).__name__}: {e}"


def post_with_retry(endpoint: str, token: str, body: Dict[str, Any],
                    timeout: float, max_retries: int) -> Tuple[int, str, int]:
    """POST with exponential backoff and full jitter on retriable failures.

    Only statuses in ``_RETRIABLE_STATUS`` (plus the synthetic ``599``) are
    retried; ``429`` uses a heavier base backoff.

    Args:
        endpoint (str): Full URL to POST to.
        token (str): Value for the ``Authorization`` header.
        body (Dict[str, Any]): JSON-serializable request body.
        timeout (float): Per-request timeout in seconds.
        max_retries (int): Maximum number of retries after the first attempt.

    Returns:
        Tuple[int, str, int]: The final status code, response text, and the
        number of attempts made.
    """
    attempt = 0
    while True:
        attempt += 1
        status, text = post_once(endpoint, token, body, timeout)
        if status < 400:
            return status, text, attempt
        retriable = status in _RETRIABLE_STATUS or status == 599
        if not retriable or attempt > max_retries:
            return status, text, attempt
        # Exponential backoff with full jitter; heavier wait for 429.
        base = 5.0 if status == 429 else 1.0
        sleep_s = base * (2 ** (attempt - 1))
        sleep_s = random.uniform(0.5 * sleep_s, sleep_s)
        time.sleep(min(sleep_s, 60.0))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iter_files(paths: List[str], scan_dir: Optional[str]) -> List[Path]:
    """Collect unique ``session_breakdown.json`` paths from inputs.

    Directories are scanned recursively; explicit file paths are included
    as-is. Duplicates (by resolved path) are removed while preserving order.

    Args:
        paths (List[str]): File or directory paths supplied on the CLI.
        scan_dir (Optional[str]): Additional directory to scan recursively.

    Returns:
        List[Path]: De-duplicated list of session breakdown file paths.
    """
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.rglob("session_breakdown.json")))
        elif pp.exists():
            files.append(pp)
    if scan_dir:
        files.extend(sorted(Path(scan_dir).rglob("session_breakdown.json")))
    seen = set()
    out = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            out.append(f)
    return out


def process_file(path: Path, endpoint: str, token: str, timeout: float,
                 max_retries: int, dry_run: bool) -> Dict[str, Any]:
    """Build, validate, and (optionally) POST a single breakdown file.

    Args:
        path (Path): Path to a ``session_breakdown.json`` file.
        endpoint (str): Full URL to POST to.
        token (str): Value for the ``Authorization`` header.
        timeout (float): Per-request timeout in seconds.
        max_retries (int): Maximum number of retries after the first attempt.
        dry_run (bool): When True, build and validate the body but skip the POST.

    Returns:
        Dict[str, Any]: A ledger record describing the outcome, including
        ``ok``, ``status``, and identifying fields or an ``error``.
    """
    body, err = build_body(path)
    base_record = {
        "path":      str(path),
        "ts":        datetime.now(timezone.utc).isoformat(),
    }
    if err is not None:
        return {**base_record, "ok": False, "error": err}

    base_record.update({
        "unique_key":      body["unique_key"],
        "model_name":      body["model_name"],
        "claw_session_id": body.get("claw_session_id"),
        "body_size_bytes": len(json.dumps(body, ensure_ascii=False).encode("utf-8")),
    })

    if dry_run:
        return {**base_record, "ok": True, "dry_run": True, "attempts": 0, "status": 0}

    status, text, attempts = post_with_retry(endpoint, token, body, timeout, max_retries)
    ok = 200 <= status < 300
    return {
        **base_record,
        "ok": ok,
        "status": status,
        "attempts": attempts,
        "response": text[:512],
    }


def main() -> int:
    """Parse CLI arguments and upload session breakdown files concurrently.

    Discovers input files, dispatches uploads across a thread pool, appends
    per-file outcomes to a JSONL ledger, and prints a summary with a status
    breakdown.

    Returns:
        int: ``0`` if every file succeeded, otherwise ``1``.
    """
    ap = argparse.ArgumentParser(
        description="Upload session_breakdown.json files to the perf-leaderboard API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("paths", nargs="*", help="JSON files or directories")
    ap.add_argument("--dir", help="Recursively scan directory for session_breakdown.json")
    ap.add_argument("--endpoint", default=os.environ.get("PERF_API_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--token", default=os.environ.get("PERF_API_TOKEN", DEFAULT_TOKEN),
                    help="Authorization header (include 'Bearer ' prefix)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build bodies but do not POST")
    ap.add_argument("--ledger",
                    default=f"post_perf_runs_ledger_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl",
                    help="JSONL file to append per-file outcomes to")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N files (smoke test)")
    args = ap.parse_args()

    files = iter_files(args.paths or [], args.dir)
    if args.limit:
        files = files[:args.limit]
    if not files:
        ap.error("no input files (pass paths or --dir)")

    print(f"== {'DRY-RUN ' if args.dry_run else ''}POST to {args.endpoint}", flush=True)
    print(f"==   files={len(files)} workers={args.workers} ledger={args.ledger}", flush=True)

    ledger_path = Path(args.ledger)
    ledger_lock = threading.Lock()

    ok_count = 0
    err_count = 0
    by_status: Dict[int, int] = {}
    err_samples: List[Dict[str, Any]] = []

    start = time.monotonic()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_file, f, args.endpoint, args.token,
                      args.timeout, args.max_retries, args.dry_run): f
            for f in files
        }
        for i, fut in enumerate(cf.as_completed(futures), 1):
            f = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"path": str(f), "ok": False, "error": f"exception:{e}"}

            with ledger_lock:
                with ledger_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if rec.get("ok"):
                ok_count += 1
            else:
                err_count += 1
                if len(err_samples) < 10:
                    err_samples.append(rec)

            s = rec.get("status", -1)
            by_status[s] = by_status.get(s, 0) + 1

            if i % 25 == 0 or i == len(files):
                elapsed = time.monotonic() - start
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i}/{len(files)}] ok={ok_count} err={err_count} "
                    f"rate={rate:.1f}/s elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.monotonic() - start
    print()
    print(f"== done in {elapsed:.1f}s: ok={ok_count} err={err_count}", flush=True)
    print(f"   status breakdown: {dict(sorted(by_status.items()))}", flush=True)
    if err_samples:
        print("   first errors:")
        for s in err_samples:
            print(f"     - {s.get('path')}  status={s.get('status')}  "
                  f"error={s.get('error') or s.get('response','')[:120]}", flush=True)
    print(f"   full ledger: {ledger_path}", flush=True)
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
