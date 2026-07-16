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
import subprocess
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
    """Return True if ``pid`` is a live (non-zombie) process.

    A killed sglang scheduler whose launcher parent is already dead lingers as
    a zombie (``<defunct>``) until pid-1 reaps it; SIGKILL cannot clear it. A
    zombie has already released every resource (GPU/VRAM/ports included), so it
    must count as gone — otherwise the death-wait spins on it for no reason.

    Args:
        pid: Process id to probe.

    Returns:
        bool: True only when the process exists and is not a zombie.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
        # Fields: "pid (comm) state ...". comm may contain ')', so split after
        # the last ')': the state char is two bytes past it.
        rparen = data.rfind(b")")
        if rparen != -1 and data[rparen + 2 : rparen + 3] == b"Z":
            return False
    except OSError:
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


def _gpu_vram_used_mb() -> list[float] | None:
    """Return per-GPU used VRAM (MiB) via rocm-smi, or None if unavailable.

    Uses ``rocm-smi --showmeminfo vram --json`` so it works without HIP device
    visibility (the kill actor runs with num_gpus=0). Best-effort: any parse or
    exec failure returns None so the caller skips the GPU-free wait.

    Returns:
        list[float] | None: Used VRAM per GPU in MiB, or None when rocm-smi is
        missing / unparseable.
    """
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    used: list[float] = []
    for fields in data.values():
        if not isinstance(fields, dict):
            continue
        for key, val in fields.items():
            kl = key.lower()
            if "vram" in kl and "used" in kl:
                try:
                    used.append(float(val) / (1024.0 * 1024.0))
                except (TypeError, ValueError):
                    pass
    return used or None


def _wait_gpu_free(threshold_mb: float, timeout_s: float) -> list[float]:
    """Wait until every GPU's used VRAM drops below ``threshold_mb``.

    ROCm reclaims a dead process's VRAM asynchronously, so a launch fired the
    instant the pids exit can hit a still-occupied GPU and abort the scheduler
    during init (``EOFError`` / exit -6). Block until the driver has actually
    returned the memory. Best-effort: returns [] immediately when rocm-smi is
    unavailable.

    Args:
        threshold_mb: Per-GPU used-VRAM ceiling considered "free".
        timeout_s: Max seconds to wait for reclamation.

    Returns:
        list[float]: Per-GPU used VRAM (MiB) still above threshold at timeout
        (empty on a clean reclaim or when rocm-smi is unavailable).
    """
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        used = _gpu_vram_used_mb()
        if used is None:
            return []
        busy = [round(u, 1) for u in used if u > threshold_mb]
        if not busy or time.time() >= deadline:
            return busy
        time.sleep(2.0)


def _gpu_total_used_mb() -> float | None:
    """Return total used VRAM (MiB) summed across all GPUs, or None.

    Returns:
        float | None: Sum of per-GPU used VRAM in MiB, or None when rocm-smi is
        unavailable / unparseable.
    """
    used = _gpu_vram_used_mb()
    if used is None:
        return None
    return sum(used)


def _gpu_used_mb_for_pgids(pgids: set[int]) -> float | None:
    """Return VRAM (MiB) held by processes in ``pgids`` per rocm-smi --showpids.

    Attributes VRAM to THIS workload by matching each listed pid's process group
    (a killed launcher is a pg leader via setsid, so its GPU child shares the
    pgid). This is what scopes the post-kill reclaim wait to our own memory
    instead of a co-tenant's unrelated allocation on another GPU. Must be sampled
    BEFORE the kill (a dead pid leaves --showpids immediately while its VRAM
    reclaims asynchronously).

    Args:
        pgids: Process-group ids owned by this workload.

    Returns:
        float | None: MiB our process groups hold (0.0 when none match), or None
        when rocm-smi is missing / non-zero / unparseable (caller uses fallback).
    """
    if not pgids:
        return 0.0
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showpids", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    total_b = 0.0
    # --showpids --json shape: {"system": {"PID<pid>": "name, #gpus, vram_b, ..."}}.
    for fields in data.values():
        if not isinstance(fields, dict):
            continue
        for key, val in fields.items():
            if not key.startswith("PID"):
                continue
            try:
                pid = int(key[3:])
            except ValueError:
                continue
            try:
                if os.getpgid(pid) not in pgids:
                    continue
            except OSError:
                continue
            parts = [x.strip() for x in str(val).split(",")]
            if len(parts) >= 3:
                try:
                    total_b += float(parts[2])
                except (TypeError, ValueError):
                    pass
    return total_b / (1024.0 * 1024.0)


def _wait_gpu_reclaimed(target_used_mb: float, slack_mb: float, timeout_s: float) -> float | None:
    """Wait until total used VRAM falls to ``target_used_mb`` (our footprint freed).

    ``target_used_mb`` is the pre-kill total minus this workload's footprint, so
    unrelated static allocations on other GPUs are already baked in and never
    extend the wait. Best-effort: returns None immediately when rocm-smi is
    unavailable.

    Args:
        target_used_mb: Total used-VRAM (MiB) expected once our memory is freed.
        slack_mb: Tolerance above target (driver rounding / idle baseline noise).
        timeout_s: Max seconds to wait for reclamation.

    Returns:
        float | None: None on a clean reclaim (or rocm-smi unavailable);
        otherwise the residual total used VRAM (MiB) still above target+slack.
    """
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        total = _gpu_total_used_mb()
        if total is None:
            return None
        if total <= target_used_mb + slack_mb:
            return None
        if time.time() >= deadline:
            return round(total, 1)
        time.sleep(2.0)


def _kill_remote(
    pid_dir: str,
    grace_sec: int,
    drain_ports: list[int] | None = None,
    death_timeout_s: float = 30.0,
    port_timeout_s: float = 60.0,
    gpu_free_threshold_mb: float = 2048.0,
    gpu_free_timeout_s: float = 120.0,
    gpu_fallback_timeout_s: float = 45.0,
) -> dict:
    """Kill the rank_*/prefill_*/decode_*/router* PID-file processes under ``pid_dir`` on this pod; returns a per-PID summary.

    One sweep covers both colocated and PD-disaggregated modes (unused
    patterns are no-ops). After signalling, block until every killed process
    truly exits, the rendezvous/serving ports drain, and the GPUs reclaim their
    VRAM, so the next launch's rank-0 binds its TCPStore + HTTP and inits its
    scheduler on a clean GPU (avoiding EADDRINUSE and the async-VRAM-reclaim
    ``EOFError`` / exit -6 scheduler abort).

    Args:
        pid_dir: Directory containing the PID files to sweep.
        grace_sec: Seconds to wait between SIGTERM and SIGKILL.
        drain_ports: Ports to wait free after processes exit (rank-0 only binds
            them; worker nodes drain instantly).
        death_timeout_s: Max seconds to wait for signalled pids to disappear.
        port_timeout_s: Max seconds to wait for ``drain_ports`` to free.
        gpu_free_threshold_mb: Reclaim-target slack (MiB), and the per-GPU
            used-VRAM ceiling for the fallback path.
        gpu_free_timeout_s: Max seconds to wait for our footprint to be reclaimed
            (primary, workload-scoped path).
        gpu_fallback_timeout_s: Max seconds for the coarse per-card fallback wait
            used only when our footprint cannot be attributed (rocm-smi/showpids
            unavailable); time-boxed so a co-tenant's GPU cannot stall teardown.

    Returns:
        dict: Summary with ``killed``, ``stale``, ``missing`` lists plus
        ``still_alive`` / ``ports_busy`` / ``gpu_busy`` diagnostics (empty on a
        clean teardown).
    """
    summary: dict[str, list] = {
        "killed": [],
        "stale": [],
        "missing": [],
        "still_alive": [],
        "ports_busy": [],
        "gpu_busy": [],
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

    # Pre-kill GPU snapshot: record OUR process groups + system-wide used VRAM
    # so the post-kill wait targets only our reclaim. Sampled now because a dead
    # pid drops out of rocm-smi --showpids before its VRAM is actually freed.
    pre_pgids: set[int] = set()
    for pf in pid_files:
        try:
            t = pf.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if t.isdigit() and int(t) > 0:
            try:
                pre_pgids.add(os.getpgid(int(t)))
            except OSError:
                pass
    gpu_total_before_mb = _gpu_total_used_mb()
    gpu_footprint_mb = _gpu_used_mb_for_pgids(pre_pgids)

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
    if killed_pids:
        if gpu_total_before_mb is not None and gpu_footprint_mb:
            # Primary: wait only for OUR footprint to be reclaimed system-wide;
            # a co-tenant's static VRAM on other GPUs is already in the target.
            target_mb = gpu_total_before_mb - gpu_footprint_mb
            residual = _wait_gpu_reclaimed(target_mb, gpu_free_threshold_mb, gpu_free_timeout_s)
            if residual is not None:
                summary["gpu_busy"] = [str(residual)]
                _log(
                    f"WARN {residual:.0f} MiB used VRAM still above reclaim target "
                    f"{target_mb:.0f}+{gpu_free_threshold_mb:.0f} MiB after {gpu_free_timeout_s:.0f}s"
                )
        else:
            # Fallback (footprint unattributable): coarse per-card threshold wait,
            # time-boxed at gpu_fallback_timeout_s so an unrelated co-tenant GPU
            # cannot stall teardown.
            busy = _wait_gpu_free(gpu_free_threshold_mb, gpu_fallback_timeout_s)
            if busy:
                summary["gpu_busy"] = [str(mb) for mb in busy]
                _log(f"WARN GPUs still hold VRAM after {gpu_fallback_timeout_s:.0f}s (MiB): {busy}")

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
    p.add_argument(
        "--gpu-free-threshold-mb",
        type=float,
        default=2048.0,
        help="per-GPU used-VRAM ceiling (MiB) treated as free (default 2048)",
    )
    p.add_argument(
        "--gpu-free-timeout",
        type=float,
        default=120.0,
        help="max seconds to wait for GPU VRAM reclaim (default 120)",
    )
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
        f"port_timeout={args.port_timeout:.0f}s "
        f"gpu_free_threshold_mb={args.gpu_free_threshold_mb:.0f} "
        f"gpu_free_timeout={args.gpu_free_timeout:.0f}s"
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
        ).remote(
            args.pid_dir,
            args.grace_sec,
            drain_ports,
            args.death_timeout,
            args.port_timeout,
            args.gpu_free_threshold_mb,
            args.gpu_free_timeout,
        )
        refs.append((node_id[:16], ref))

    # Actor upper bound: grace + death-wait + port-wait + gpu-wait + margin.
    get_timeout = int(
        args.grace_sec + args.death_timeout + args.port_timeout + args.gpu_free_timeout + 30
    )
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
