#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node sglang / vllm server killer (counterpart to ``launch_multinode.py``)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Default rendezvous / serving ports to drain before returning success, so the subsequent launch_multinode.py can bind
# rank-0's TCPStore + HTTP without colliding with a still-dying prior server (the cause of "Rank N scheduler died
# during initialization (exit -6)" + NCCL "TCPStore shut down too early" on restart). dist-init defaults to
# $RAYJOB_DIST_INIT_PORT else 29500; 8888 is the aggregated inference port; 30000/30001 are the PD prefill/decode
# ports.
_DEFAULT_DIST_INIT_PORT = 29500

# _kill_remote runs SIGTERM -> sleep(grace) -> SIGKILL sequentially per pid file, so a node with K pid files spends up
# to K*grace in the grace phase. main() does not know K per node (each actor globs its own remote pid_dir), so budget
# the ray.get timeout for this many pid files/node (ranks + prefill/decode/router).
_GRACE_PID_BUDGET = 16

# Per-card VRAM wait when this pod's footprint cannot be attributed.
_GPU_FALLBACK_TIMEOUT_S = 45.0


def _log(msg: str) -> None:
    """Write a timestamped progress line to stderr and flush it."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sys.stderr.write(f"[kill_multinode {ts}] {msg}\n")
    sys.stderr.flush()


def _pid_alive(pid: int) -> bool:
    """Return True if ``pid`` is a live (non-zombie) process."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
        # Fields: "pid (comm) state ...". comm may contain ')', so split after the last ')': the state char is two
        # bytes past it.
        rparen = data.rfind(b")")
        if rparen != -1 and data[rparen + 2 : rparen + 3] == b"Z":
            return False
    except OSError:
        return False
    return True


def _wait_pids_gone(pids: list[int], timeout_s: float) -> list[int]:
    """Poll until every pid exits, escalating SIGKILL to the group on the way."""
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
    """Return True if a fresh listener can bind the wildcard ``port`` now."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))  # nosec B104 - bind probe checks whether the public service port is free.
        return True
    except OSError:
        return False
    finally:
        s.close()


def _wait_ports_free(ports: list[int], timeout_s: float) -> list[int]:
    """Poll until every port is bindable, returning any still busy at timeout."""
    if not ports:
        return []
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        busy = [p for p in ports if not _port_free(p)]
        if not busy or time.time() >= deadline:
            return busy
        time.sleep(1.0)


def _gpu_vram_used_mb() -> list[float] | None:
    """Return per-GPU used VRAM (MiB) via rocm-smi, or None if unavailable."""
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
    """Wait until every GPU's used VRAM drops below ``threshold_mb``."""
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
    """Return total used VRAM (MiB) summed across all GPUs, or None."""
    used = _gpu_vram_used_mb()
    if used is None:
        return None
    return sum(used)


def _gpu_used_mb_for_pgids(pgids: set[int]) -> float | None:
    """Return VRAM (MiB) held by processes in ``pgids`` per rocm-smi --showpids."""
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
    """Wait until total used VRAM falls to ``target_used_mb`` (our footprint freed)."""
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


def _unlink_quietly(path: Path) -> None:
    """Remove a PID file that may already be gone."""
    try:
        path.unlink()
    except OSError:
        # Already removed, or never written: either way there is nothing to clean up.
        pass


def _read_pid(pid_file: Path) -> int | None:
    """The PID a file names, or None when it names none.

    An unreadable file, a non-numeric body and the sentinel ``0`` are all "no
    real process"; the caller decides which of those is stale and which is
    missing.
    """
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.isdigit():
        return None
    pid = int(text)
    return pid if pid > 0 else None


def _process_alive(pid: int) -> bool:
    """Whether a signal can still reach this process."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _signal_group(pid: int, sig: int) -> None:
    """Signal the process group this PID leads, falling back to the process itself.

    Each launcher is a group leader via ``setsid``, so the group is what carries
    its children; a process that has already exited is nothing to signal.
    """
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            # Process already exited; nothing to signal.
            pass


def _pre_kill_pgids(pid_files: list[Path]) -> set[int]:
    """The process groups this pod is about to kill, for attributing the VRAM they hold."""
    pgids: set[int] = set()
    for pid_file in pid_files:
        pid = _read_pid(pid_file)
        if pid is None:
            continue
        try:
            pgids.add(os.getpgid(pid))
        except OSError:
            # The process is already gone, so it leads no group this kill has to spare.
            pass
    return pgids


def _kill_pid_file(pid_file: Path, grace_sec: int, summary: dict[str, list]) -> int | None:
    """Kill what one PID file names, booking it under the heading it belongs to.

    Returns the PID that was signalled, or None when the file named no live
    process.
    """
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except OSError:
        summary["missing"].append(pid_file.name)
        return None
    if not text.isdigit():
        summary["stale"].append(pid_file.name)
        _unlink_quietly(pid_file)
        return None
    pid = int(text)
    # Sentinel 0 = "no real process"; treat as stale, as for a process no signal reaches.
    if pid <= 0 or not _process_alive(pid):
        summary["stale"].append(f"{pid_file.name}:{pid}")
        _unlink_quietly(pid_file)
        return None
    _signal_group(pid, signal.SIGTERM)
    time.sleep(grace_sec)
    if _process_alive(pid):
        _signal_group(pid, signal.SIGKILL)
    summary["killed"].append(f"{pid_file.name}:{pid}")
    _unlink_quietly(pid_file)
    return pid


def _wait_gpu_reclaim(
    summary: dict[str, list],
    *,
    gpu_total_before_mb: float | None,
    gpu_footprint_mb: float,
    gpu_free_threshold_mb: float,
    gpu_free_timeout_s: float,
) -> None:
    """Wait for the VRAM this pod held to come back, and record what did not."""
    if gpu_total_before_mb is not None and gpu_footprint_mb:
        # Primary: wait only for OUR footprint to be reclaimed system-wide; a co-tenant's static VRAM on other
        # GPUs is already in the target.
        target_mb = gpu_total_before_mb - gpu_footprint_mb
        residual = _wait_gpu_reclaimed(target_mb, gpu_free_threshold_mb, gpu_free_timeout_s)
        if residual is not None:
            summary["gpu_busy"] = [str(residual)]
            _log(
                f"WARN {residual:.0f} MiB used VRAM still above reclaim target "
                f"{target_mb:.0f}+{gpu_free_threshold_mb:.0f} MiB after {gpu_free_timeout_s:.0f}s"
            )
        return
    # Fallback (footprint unattributable): coarse per-card threshold wait, time-boxed at
    # _GPU_FALLBACK_TIMEOUT_S so an unrelated co-tenant GPU cannot stall teardown.
    busy = _wait_gpu_free(gpu_free_threshold_mb, _GPU_FALLBACK_TIMEOUT_S)
    if busy:
        summary["gpu_busy"] = [str(mb) for mb in busy]
        _log(f"WARN GPUs still hold VRAM after {_GPU_FALLBACK_TIMEOUT_S:.0f}s (MiB): {busy}")


@dataclass(frozen=True)
class KillWaits:
    """How long a pod's kill waits for its pids to exit, its ports to drain and its VRAM to come back."""

    death_timeout_s: float = 30.0
    port_timeout_s: float = 60.0
    gpu_free_threshold_mb: float = 2048.0
    gpu_free_timeout_s: float = 120.0


def _kill_remote(
    pid_dir: str,
    grace_sec: int,
    drain_ports: list[int] | None = None,
    waits: KillWaits = KillWaits(),
) -> dict:
    """Kill the rank_*/prefill_*/decode_*/router* PID-file processes under ``pid_dir`` on this pod; returns a per-PID summary."""
    summary: dict[str, list] = {
        "killed": [],
        "stale": [],
        "missing": [],
        "still_alive": [],
        "ports_busy": [],
        "gpu_busy": [],
    }
    pid_dir_path = Path(pid_dir)
    if not pid_dir_path.is_dir():
        return summary

    pid_files = sorted(
        list(pid_dir_path.glob("rank_*.pid"))
        + list(pid_dir_path.glob("prefill_*.pid"))
        + list(pid_dir_path.glob("decode_*.pid"))
        + list(pid_dir_path.glob("router*.pid"))
    )

    # Pre-kill GPU snapshot: record OUR process groups + system-wide used VRAM so the post-kill wait targets only our
    # reclaim.
    pre_pgids = _pre_kill_pgids(pid_files)
    gpu_total_before_mb = _gpu_total_used_mb()
    gpu_footprint_mb = _gpu_used_mb_for_pgids(pre_pgids)

    killed_pids = [pid for pid in (_kill_pid_file(f, grace_sec, summary) for f in pid_files) if pid is not None]

    # Block until the signalled processes truly exit, then until the rendezvous/serving ports drain.
    if killed_pids:
        still = _wait_pids_gone(killed_pids, waits.death_timeout_s)
        if still:
            summary["still_alive"] = [str(pid) for pid in still]
            _log(f"WARN pids still alive after {waits.death_timeout_s:.0f}s: {still}")
    if drain_ports:
        busy = _wait_ports_free(list(drain_ports), waits.port_timeout_s)
        if busy:
            summary["ports_busy"] = [str(port) for port in busy]
            _log(f"WARN ports still bound after {waits.port_timeout_s:.0f}s: {busy}")
    if killed_pids:
        _wait_gpu_reclaim(
            summary,
            gpu_total_before_mb=gpu_total_before_mb,
            gpu_footprint_mb=gpu_footprint_mb,
            gpu_free_threshold_mb=waits.gpu_free_threshold_mb,
            gpu_free_timeout_s=waits.gpu_free_timeout_s,
        )

    return summary


def main() -> int:
    """Parse CLI arguments and fan out kill actors across all alive nodes."""
    p = argparse.ArgumentParser(
        prog="kill_multinode.py",
        description="Kill every multi-node server process spawned by launch_multinode.py.",
    )
    p.add_argument(
        "--pid-dir", required=True, help="dir containing rank_*.pid files (same value passed to launch_multinode)"
    )
    p.add_argument("--grace-sec", type=int, default=5, help="seconds between SIGTERM and SIGKILL (default 5)")
    args = p.parse_args()

    dist_port = int(os.environ.get("RAYJOB_DIST_INIT_PORT", _DEFAULT_DIST_INIT_PORT) or _DEFAULT_DIST_INIT_PORT)
    # dist-init/TCPStore + aggregated HTTP + PD prefill/decode HTTP.
    drain_ports = sorted({dist_port, 8888, 30000, 30001})
    waits = KillWaits()

    _log(f"pid_dir={args.pid_dir} grace={args.grace_sec}s drain_ports={drain_ports} {waits}")

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
            waits,
        )
        refs.append((node_id[:16], ref))

    # Actor upper bound: per-pid grace (budgeted for up to _GRACE_PID_BUDGET pid files/node) + death-wait + port-wait
    # + gpu-wait + margin.
    get_timeout = int(
        args.grace_sec * _GRACE_PID_BUDGET
        + waits.death_timeout_s
        + waits.port_timeout_s
        + waits.gpu_free_timeout_s
        + 30
    )
    out: dict[str, dict] = {}
    for short_id, ref in refs:
        try:
            out[short_id] = ray.get(ref, timeout=get_timeout)
        except (ray.exceptions.RayError, OSError) as exc:
            _log(f"node {short_id}: kill FAILED: {type(exc).__name__}: {exc}")
            out[short_id] = {"error": str(exc)}

    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    sys.stdout.flush()

    # The caller sequences LAUNCH directly after this, so an incomplete kill has to be reported as a failure: a
    # surviving pid or a bound rendezvous/serving port leaves the next rank 0 racing a dying server for the TCPStore /
    # :8888.
    blocking = ("error", "still_alive", "ports_busy")
    bad = {node: {k: summary[k] for k in blocking if summary.get(k)} for node, summary in out.items()}
    bad = {node: reasons for node, reasons in bad.items() if reasons}
    if bad:
        _log(f"FAILED kill incomplete on {len(bad)} node(s): {json.dumps(bad, sort_keys=True)}")
        return 1
    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
