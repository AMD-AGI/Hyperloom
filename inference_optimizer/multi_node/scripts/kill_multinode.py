#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node sglang / vllm server killer.

Counterpart to ``launch_multinode.py``. Submitted via Ray Dashboard REST
by ``inference_optimizer.multi_node restart-server`` (during the kill
phase) and by ``inference_optimizer.multi_node stop-rayjob`` (when the
caller wants a clean shutdown).

Algorithm:

  1. ``ray.init()`` (no address; in-pod).
  2. Enumerate all alive nodes.
  3. For each node, spawn a ``@ray.remote`` actor pinned via
     ``NodeAffinitySchedulingStrategy(node_id, soft=False)``.
  4. Inside each actor: read every ``rank_*.pid`` file under
     ``--pid-dir`` (the same dir launch_multinode wrote), SIGTERM the
     process group, sleep ``GRACE``, SIGKILL if still alive, remove
     stale PID files.

Idempotent: missing ``pid_dir`` / missing ``rank_*.pid`` / dead PID is
treated as "nothing to kill, success". Only kills processes whose PID
files are inside ``--pid-dir`` (NEVER ``pkill -f sglang`` per IR-5).

Returns 0 on success even if some nodes had nothing to kill.
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
    """Kill every process referenced by rank_*.pid / prefill_*.pid /
    decode_*.pid / router*.pid under ``pid_dir`` on THIS pod (the
    actor's host). Returns a per-PID summary.

    Glob patterns:
      * ``rank_*.pid``     — colocated mode (one server group, all ranks)
      * ``prefill_*.pid``  — PD disaggregated, prefill group ranks
      * ``decode_*.pid``   — PD disaggregated, decode group ranks
      * ``router*.pid``    — PD disaggregated router (only present on head)

    A single sweep covers both modes; pid files only exist for the
    mode that was actually launched, so unused patterns are no-ops.
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
        # Sentinel value 0 means "no real process" (e.g. vllm worker
        # ranks where KubeRay's own ``ray start`` is the actual worker
        # process and we have nothing of our own to kill). Treat as
        # stale, NOT as ``os.kill(0, 0)`` which would target our own
        # process group on Linux.
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

        # SIGTERM the entire process group (``setsid`` in the bash detach
        # path in launch_multinode.py makes each launcher its own pg leader).
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

    # Also clean up the rayjoin pid files written by vllm worker actors
    # in earlier versions. Current launch_multinode.py no longer creates
    # these (KubeRay manages worker ray start) but old state can leave
    # orphan files behind; remove them if found.
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
