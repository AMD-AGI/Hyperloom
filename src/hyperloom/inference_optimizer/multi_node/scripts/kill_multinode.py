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
import socket
import sys
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Default rendezvous / serving ports to drain before returning success, so the
# subsequent launch_multinode.py can bind rank-0's TCPStore + HTTP without
# colliding with a still-dying prior server (the cause of "Rank N scheduler
# died during initialization (exit -6)" + NCCL "TCPStore shut down too early"
# on restart). dist-init defaults to $RAYJOB_DIST_INIT_PORT else 29500; 8888 is
# the colocated inference port; 30000/30001 are the PD prefill/decode ports.
_DEFAULT_DIST_INIT_PORT = 29500


def _log(msg: str) -> None:
    """Write a timestamped progress line to stderr and flush it.

    Args:
        msg (str): The message text to emit.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[kill_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` still exists (signal-0 probe).

    Args:
        pid: Process id to probe.

    Returns:
        bool: True when the process is still present.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _wait_pids_gone(pids: list[int], timeout_s: float) -> list[int]:
    """Poll until every pid exits, escalating SIGKILL to the group on the way.

    A returned "SUCCEEDED" kill that leaves the sglang scheduler workers still
    dying keeps the rendezvous/serving ports bound, so the next launch aborts.
    Block here until the process group is truly gone (or timeout).

    Args:
        pids: Process ids that were signalled.
        timeout_s: Max seconds to wait for all pids to disappear.

    Returns:
        list[int]: Pids still alive after ``timeout_s`` (empty on success).
    """
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_alive(pid)]
        if not alive:
            return []
        for pid in alive:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(1.0)
    return [pid for pid in pids if _pid_alive(pid)]


def _port_free(port: int) -> bool:
    """Return True if a fresh listener can bind the wildcard ``port`` now.

    Bind without SO_REUSEADDR so a live process still holding the port reports
    EADDRINUSE (a dead process's listen socket is released immediately, so
    there are no TIME_WAIT false positives for a listen port).

    Args:
        port: TCP port number to probe.

    Returns:
        bool: True when the port is bindable (free).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _wait_ports_free(ports: list[int], timeout_s: float) -> list[int]:
    """Poll until every port is bindable, returning any still busy at timeout.

    Args:
        ports: TCP ports the next launch's rank-0 must bind.
        timeout_s: Max seconds to wait for all ports to drain.

    Returns:
        list[int]: Ports still busy after ``timeout_s`` (empty on success).
    """
    if not ports:
        return []
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        busy = [p for p in ports if not _port_free(p)]
        if not busy or time.time() >= deadline:
            return busy
        time.sleep(1.0)


def _kill_remote(
    pid_dir: str,
    grace_sec: int,
    drain_ports: list[int] | None = None,
    death_timeout_s: float = 30.0,
    port_timeout_s: float = 60.0,
) -> dict:
    """Kill the rank_*/prefill_*/decode_*/router* PID-file processes under ``pid_dir`` on this pod; returns a per-PID summary.

    One sweep covers both colocated and PD-disaggregated modes (unused
    patterns are no-ops). After signalling, block until every killed process
    truly exits and the rendezvous/serving ports drain, so the next launch's
    rank-0 can bind its TCPStore + HTTP without an EADDRINUSE abort.

    Args:
        pid_dir: Directory containing the PID files to sweep.
        grace_sec: Seconds to wait between SIGTERM and SIGKILL.
        drain_ports: Ports to wait free after processes exit (rank-0 only binds
            them; worker nodes drain instantly).
        death_timeout_s: Max seconds to wait for signalled pids to disappear.
        port_timeout_s: Max seconds to wait for ``drain_ports`` to free.

    Returns:
        dict: Summary with ``killed``, ``stale``, ``missing`` lists plus
        ``still_alive`` / ``ports_busy`` diagnostics (empty on a clean drain).
    """
    summary: dict[str, list] = {
        "killed": [],
        "stale": [],
        "missing": [],
        "still_alive": [],
        "ports_busy": [],
    }
    p = Path(pid_dir)
    if not p.is_dir():
        return summary
    killed_pids: list[int] = []

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
                # Stale PID file already removed; nothing to clean up.
                pass
            continue

        pid = int(text)
        # Sentinel 0 = "no real process"; treat as stale.
        if pid <= 0:
            summary["stale"].append(f"{pid_file.name}:{pid}")
            try:
                pid_file.unlink()
            except OSError:
                # Stale PID file already removed; nothing to clean up.
                pass
            continue
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            summary["stale"].append(f"{pid_file.name}:{pid}")
            try:
                pid_file.unlink()
            except OSError:
                # Stale PID file already removed; nothing to clean up.
                pass
            continue

        # SIGTERM the whole process group (each launcher is a pg leader via setsid).
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Process already exited; nothing to signal.
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
                    # Process already exited; nothing to signal.
                    pass

        summary["killed"].append(f"{pid_file.name}:{pid}")
        killed_pids.append(pid)
        try:
            pid_file.unlink()
        except OSError:
            # PID file already gone; nothing to clean up.
            pass

    # Clean up legacy rayjoin pid files.
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
                    # Process already exited; nothing to signal.
                    pass
        try:
            pid_file.unlink()
        except OSError:
            # PID file already gone; nothing to clean up.
            pass

    # Block until the signalled processes truly exit, then until the
    # rendezvous/serving ports drain. Returning before both leaves the next
    # launch's rank-0 racing a still-dying server for the TCPStore/HTTP port.
    if killed_pids:
        still = _wait_pids_gone(killed_pids, death_timeout_s)
        if still:
            summary["still_alive"] = [str(pid) for pid in still]
            _log(f"WARN pids still alive after {death_timeout_s:.0f}s: {still}")
    if drain_ports:
        busy = _wait_ports_free(list(drain_ports), port_timeout_s)
        if busy:
            summary["ports_busy"] = [str(port) for port in busy]
            _log(f"WARN ports still bound after {port_timeout_s:.0f}s: {busy}")

    return summary


def main() -> int:
    """Parse CLI arguments and fan out kill actors across all alive nodes.

    Connects to the in-pod Ray cluster, schedules one pinned kill actor per
    alive node, collects each node's kill summary, and prints the aggregate
    as JSON to stdout.

    Returns:
        int: Process exit code; ``0`` on success even when some nodes had
        nothing to kill.
    """
    p = argparse.ArgumentParser(
        prog="kill_multinode.py",
        description="Kill every multi-node server process spawned by launch_multinode.py.",
    )
    p.add_argument(
        "--pid-dir", required=True, help="dir containing rank_*.pid files (same value passed to launch_multinode)"
    )
    p.add_argument("--grace-sec", type=int, default=5, help="seconds between SIGTERM and SIGKILL (default 5)")
    p.add_argument(
        "--drain-ports",
        default="",
        help="comma-separated ports to wait free after kill (default: "
        "$RAYJOB_DIST_INIT_PORT|29500,8888,30000,30001)",
    )
    p.add_argument("--death-timeout", type=float, default=30.0, help="max seconds to wait for pids to exit (default 30)")
    p.add_argument("--port-timeout", type=float, default=60.0, help="max seconds to wait for ports to drain (default 60)")
    args = p.parse_args()

    if args.drain_ports.strip():
        drain_ports = [int(x) for x in args.drain_ports.split(",") if x.strip().isdigit()]
    else:
        dist_port = int(os.environ.get("RAYJOB_DIST_INIT_PORT", _DEFAULT_DIST_INIT_PORT) or _DEFAULT_DIST_INIT_PORT)
        # dist-init/TCPStore + colocated HTTP + PD prefill/decode HTTP.
        drain_ports = sorted({dist_port, 8888, 30000, 30001})

    _log(
        f"pid_dir={args.pid_dir} grace={args.grace_sec}s "
        f"drain_ports={drain_ports} death_timeout={args.death_timeout:.0f}s "
        f"port_timeout={args.port_timeout:.0f}s"
    )

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    _log(f"alive nodes: {len(nodes)}")

    KillActor = ray.remote(num_cpus=0, num_gpus=0)(_kill_remote)
    refs = []
    for node in nodes:
        node_id = node["NodeID"]
        ref = KillActor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
        ).remote(args.pid_dir, args.grace_sec, drain_ports, args.death_timeout, args.port_timeout)
        refs.append((node_id[:16], ref))

    # Actor upper bound: grace + death-wait + port-wait + margin.
    get_timeout = int(args.grace_sec + args.death_timeout + args.port_timeout + 30)
    out: dict[str, dict] = {}
    for short_id, ref in refs:
        try:
            out[short_id] = ray.get(ref, timeout=get_timeout)
        except Exception as exc:  # noqa: BLE001
            _log(f"node {short_id}: kill FAILED: {type(exc).__name__}: {exc}")
            out[short_id] = {"error": str(exc)}

    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    sys.stdout.flush()
    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
