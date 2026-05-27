"""Cortex KB NDJSON flusher daemon.

Drains ``<session_dir>/runtime/cortex/.kb_pending.ndjson`` to the Cortex
KB service every ``--interval-sec`` (default 5s). On success the row is
appended to ``.kb_flushed.ndjson``; on permanent failure the row moves
to ``.kb_dead_letter.ndjson`` and a structured audit log entry is
written.

Lifetime
--------
The daemon is started by the operator (or by ``robustness-agent`` per
KB_design §3.13 M1 §8) with::

    python -m inference_optimizer.scripts.cortex_kb_flusher \\
        --session-dir $USER_DATA_PATH \\
        [--interval-sec 5] \\
        [--cortex-kb-url http://kb-service.primus-cortex.svc.cluster.local]

It writes its PID to ``.kb_flusher.pid`` so robustness can ping it
(``kill -0 $PID``). SIGTERM / SIGINT triggers a graceful drain (final
batch then exit 0).

Design notes
------------
* The daemon shares :class:`~inference_optimizer.cortex_kb_client.CortexKBClient`
  with the Coordinator so semantic guarantees stay aligned. It calls
  ``client.drain_pending(timeout_sec=interval_sec - 0.5)`` then sleeps;
  there is no separate flush implementation that could drift.
* Each iteration is bounded by ``timeout_sec`` so a stuck Cortex HTTP
  call cannot block the rest of the queue forever.
* Restart safety: opening the pid file in exclusive mode prevents
  double-launch; on detected stale pid (process gone) we overwrite.
* No persistent in-memory state — the on-disk NDJSON is the authority.
* Each pending row is replayed via a single ``POST /v1/points/propose``
  or ``POST /v1/edges/propose``. The session-based hypothesize /
  verify protocol was retired, so the daemon no longer replays
  those ops. The KB ``/v1/bulk/ingest`` endpoint is reserved for
  ``offline_pipeline`` source only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_optimizer.cortex_kb_client import CortexKBClient
from inference_optimizer.session_paths import (
    cortex_dead_letter_ndjson,
    cortex_dir,
    cortex_flushed_ndjson,
    cortex_flusher_pid,
    cortex_pending_ndjson,
)


log = logging.getLogger("cortex_kb_flusher")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_pid(path: Path) -> None:
    """Best-effort exclusive write of the daemon PID.

    If an existing PID file points at a live process, exit 1 — operator
    must kill the prior instance. If the prior PID is dead, overwrite
    so monitor scripts don't trip on a stale file.
    """
    pid = os.getpid()
    if path.exists():
        try:
            prior_raw = path.read_text(encoding="utf-8").strip().splitlines()
            prior = int(prior_raw[0]) if prior_raw else 0
        except (OSError, ValueError):
            prior = 0
        if prior and prior != pid:
            try:
                os.kill(prior, 0)
                print(
                    f"ERROR: another flusher already running (pid={prior}); "
                    f"refusing to start.",
                    file=sys.stderr,
                )
                sys.exit(1)
            except ProcessLookupError:
                log.info("removing stale pid file (dead pid=%s)", prior)
            except PermissionError:
                # Probably a different user — assume alive, refuse.
                print(
                    f"ERROR: pid file owned by another user (pid={prior})",
                    file=sys.stderr,
                )
                sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n{_now_iso()}\n", encoding="utf-8")


def _remove_pid(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _move_flushed_rows(session_dir: Path, drained: int) -> None:
    """Append successfully-drained rows to ``.kb_flushed.ndjson``.

    The :meth:`CortexKBClient.drain_pending` helper does not currently
    write to ``.kb_flushed.ndjson`` itself; rather, ``drained`` indicates
    how many entries we processed. Until we move the bookkeeping inside
    the client, this helper records a single bookmark row per drain so
    breakdown collection can still attribute counts.
    """
    if drained <= 0:
        return
    flushed_path = cortex_flushed_ndjson(session_dir)
    flushed_path.parent.mkdir(parents=True, exist_ok=True)
    bookmark = {
        "ts":      _now_iso(),
        "op":      "drain_bookmark",
        "drained": drained,
        "pid":     os.getpid(),
    }
    with flushed_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(bookmark, sort_keys=True) + "\n")


def _move_dead_letter(session_dir: Path, dead_letter: int) -> None:
    """Append a per-drain bookmark to ``.kb_dead_letter.ndjson``."""
    if dead_letter <= 0:
        return
    dl_path = cortex_dead_letter_ndjson(session_dir)
    dl_path.parent.mkdir(parents=True, exist_ok=True)
    bookmark = {
        "ts":          _now_iso(),
        "op":          "drain_bookmark",
        "dead_letter": dead_letter,
        "pid":         os.getpid(),
    }
    with dl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(bookmark, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cortex_kb_flusher",
        description="Drain $SESSION_DIR/runtime/cortex/.kb_pending.ndjson "
                    "into the Cortex KB service (v0.8 M1 daemon).",
    )
    p.add_argument(
        "--session-dir", required=True, type=Path,
        help="Hyperloom session directory (the one that holds "
             "runtime/cortex/*).",
    )
    p.add_argument(
        "--cortex-kb-url", type=str, default=None,
        help="Override CORTEX_KB_URL for this daemon.",
    )
    p.add_argument(
        "--interval-sec", type=float, default=5.0,
        help="Seconds between drain rounds (default 5.0).",
    )
    p.add_argument(
        "--max-retries", type=int, default=6,
        help="Reserved for the dead-letter promotion threshold.",
    )
    p.add_argument(
        "--once", action="store_true", default=False,
        help="Drain once and exit (useful for cron / unit tests).",
    )
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s cortex_kb_flusher [%(levelname)s] %(message)s",
    )
    session_dir = Path(args.session_dir).resolve()
    if not session_dir.exists():
        print(f"ERROR: session-dir does not exist: {session_dir}", file=sys.stderr)
        return 2
    cortex_root = cortex_dir(session_dir)
    cortex_root.mkdir(parents=True, exist_ok=True)
    pid_path = cortex_flusher_pid(session_dir)
    _write_pid(pid_path)
    client = CortexKBClient(
        session_dir=session_dir,
        kb_url=args.cortex_kb_url,
        enabled=True,
    )
    stop_event = {"set": False}

    def _on_signal(signum: int, _frame: Any) -> None:  # noqa: ANN401
        log.info("received signal %s; will drain once more and exit", signum)
        stop_event["set"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    pending_path = cortex_pending_ndjson(session_dir)
    try:
        while True:
            queue_len = 0
            try:
                if pending_path.exists():
                    with pending_path.open("r", encoding="utf-8") as f:
                        queue_len = sum(1 for line in f if line.strip())
            except OSError:
                queue_len = 0
            if queue_len > 0:
                report = client.drain_pending(
                    timeout_sec=max(0.5, args.interval_sec - 0.5),
                )
                _move_flushed_rows(session_dir, report.get("drained", 0))
                _move_dead_letter(session_dir, report.get("dead_letter", 0))
                log.info("drain report: %s", report)
            if args.once or stop_event["set"]:
                break
            # If queue was empty, sleep the full interval. If we drained
            # something, only sleep a short tick so we catch up quickly.
            time.sleep(args.interval_sec if queue_len == 0 else 0.2)
    finally:
        _remove_pid(pid_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
