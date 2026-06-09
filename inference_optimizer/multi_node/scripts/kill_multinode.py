#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node sglang / vllm server killer (counterpart to ``launch_multinode.py``).

Per alive node, a node-pinned actor SIGTERMs each ``rank_*.pid`` process
group under ``--pid-dir``, waits ``GRACE``, then SIGKILLs. Idempotent
(missing/dead PIDs = success). Only kills PIDs from ``--pid-dir``, never
``pkill -f sglang`` (IR-5). Returns 0 on success.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[kill_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _kill_remote(pid_dir: str, grace_sec: int) -> dict:
    """Kill the rank_*/prefill_*/decode_*/router* PID-file processes under ``pid_dir`` on this pod; returns a per-PID summary.

    One sweep covers both colocated and PD-disaggregated modes (unused
    patterns are no-ops).
    """
    summary: dict[str, list] = {"killed": [], "stale": [], "missing": []}
    p = Path(pid_dir)
    if not p.is_dir():
        return summary

    pid_files = sorted(
        list(p.glob("rank_*.pid"))
        + list(p.glob("prefill_*.pid"))
        + list(p.glob("decode_*.pid"))
        + list(p.glob("router*.pid"))
    )
    for pid_file in pid_files:
        try:
            text = pid_file.read_text(encoding="utf-8").strip()
        except OSError:
            summary["missing"].append(pid_file.name)
            continue
        if not text.isdigit():
            summary["stale"].append(pid_file.name)
            try:
                pid_file.unlink()
            except OSError:
                pass
            continue

        pid = int(text)
        # Sentinel 0 = "no real process"; treat as stale (os.kill(0,...) would hit our own pg).
        if pid <= 0:
            summary["stale"].append(f"{pid_file.name}:{pid}")
            try:
                pid_file.unlink()
            except OSError:
                pass
            continue
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            summary["stale"].append(f"{pid_file.name}:{pid}")
            try:
                pid_file.unlink()
            except OSError:
                pass
            continue

        # SIGTERM the whole process group (each launcher is its own pg leader via setsid).
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        time.sleep(grace_sec)

        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            pass
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        summary["killed"].append(f"{pid_file.name}:{pid}")
        try:
            pid_file.unlink()
        except OSError:
            pass

    # Clean up legacy rayjoin pid files (current launch no longer creates these).
    for pid_file in sorted(p.glob("rank_*_rayjoin.pid")):
        try:
            text = pid_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text.isdigit():
            pid = int(text)
            if pid > 0:  # skip sentinel 0
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        try:
            pid_file.unlink()
        except OSError:
            pass

    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        prog="kill_multinode.py",
        description="Kill every multi-node server process spawned by launch_multinode.py.",
    )
    p.add_argument("--pid-dir", required=True,
                   help="dir containing rank_*.pid files (same value passed to launch_multinode)")
    p.add_argument("--grace-sec", type=int, default=5,
                   help="seconds between SIGTERM and SIGKILL (default 5)")
    args = p.parse_args()

    _log(f"pid_dir={args.pid_dir} grace={args.grace_sec}s")

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    _log(f"alive nodes: {len(nodes)}")

    KillActor = ray.remote(num_cpus=0, num_gpus=0)(_kill_remote)
    refs = []
    for node in nodes:
        node_id = node["NodeID"]
        ref = KillActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False,
            ),
        ).remote(args.pid_dir, args.grace_sec)
        refs.append((node_id[:16], ref))

    out: dict[str, dict] = {}
    for short_id, ref in refs:
        try:
            out[short_id] = ray.get(ref, timeout=60)
        except Exception as exc:  # noqa: BLE001
            _log(f"node {short_id}: kill FAILED: {type(exc).__name__}: {exc}")
            out[short_id] = {"error": str(exc)}

    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    sys.stdout.flush()
    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
