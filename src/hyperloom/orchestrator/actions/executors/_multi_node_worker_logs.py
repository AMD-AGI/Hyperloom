# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node-only: reading per-worker sglang server logs, live and after a failed restart."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

# Tail sampled when deciding whether a worker is wedged or crashed; the markers below repeat, so the recent window
# carries the verdict.
_VERDICT_TAIL_BYTES = 65536

#: Log lines that mean the worker will never report ready, so /health can only time out.
_STARTUP_FAILURE_PATTERNS: tuple[str, ...] = (
    r"error: unrecognized arguments:.*",
    r"[A-Za-z0-9_./-]*: error: .*",  # argparse "<prog>: error: ..."
    r"error: the following arguments are required:.*",
    r"error: argument .*",
    # Worker subprocess died BEFORE ever reporting ready -- SIGKILL/OOM (exit code -9), non-zero exit, or an
    # explicit early-exit RuntimeError (e.g. ``sglang subprocess exited with code -9 before reporting ready``).
    # /health will never come up for a dead engine, so fast-fail instead of burning the (aiter-widened) health
    # gate.
    r".*exited with code -?\d+ before reporting ready.*",
    r".*(sglang|engine|worker) (sub)?process exited with code -?\d+.*",
    r"(?i).*(cuda|hip|rocm|torch)[^\n]{0,40}out of memory.*",
    r"(?i).*torch\.OutOfMemoryError.*",
)


class _CollectedLog(NamedTuple):
    """One pod's captured server log and where it was read from."""

    body: str
    source: str


@dataclass(frozen=True)
class _SshAccess:
    """What it takes to SSH into this run's pods and tail their server logs."""

    key_path: str
    known_hosts: str
    default_port: int
    remote_path: str


def shared_server_log_dir() -> str:
    """The shared-FS directory holding per-worker server logs, or ``""`` when it is not usable from here."""
    shared = os.path.expandvars(os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip())
    return shared if shared.startswith("/") and "$" not in shared else ""


def _tail_bytes(path: Path, max_bytes: int) -> str | None:
    """Decoded tail of ``path``, or None when it cannot be read."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return None


def _rank0_log_tail(shared_dir: str, ip: str) -> str:
    """Tail of worker ``ip``'s rank-0 server log under ``shared_dir``, or ``""`` when it cannot be read."""
    if not shared_dir:
        return ""
    return _tail_bytes(Path(shared_dir) / ("mn_infera_server_" + str(ip) + "_r0.log"), _VERDICT_TAIL_BYTES) or ""


def worker_detokenizer_wedged(shared_dir: str, ip: str) -> bool:
    """Best-effort: True when worker ``ip``'s server log shows a persistent detokenizer wedge -- weights are loaded but the detokenizer never heartbeats, so the engine's /health never flips and the HTTP port never binds."""
    tail = _rank0_log_tail(shared_dir, ip)
    # >=15 recent markers ~= 30s+ of continuous stall (marker ~every 2s).
    return len(re.findall(r"Health check failed.*detokenizer", tail, re.IGNORECASE)) >= 15


def worker_startup_crashed(shared_dir: str, ip: str) -> str | None:
    """Best-effort: return a short reason when worker ``ip``'s server log shows a NON-RECOVERABLE startup failure (so /health will never flip), else None."""
    tail = _rank0_log_tail(shared_dir, ip)
    for pattern in _STARTUP_FAILURE_PATTERNS:
        match = re.search(pattern, tail)
        if match:
            return match.group(0).strip()[:200]
    return None


def _shared_worker_log_tail(ip: str, max_bytes: int = 2_000_000) -> str | None:
    """Best-effort tail of a worker's shared-FS sglang server log."""
    import glob as _glob

    shared = shared_server_log_dir()
    if not shared:
        return None
    matches = _glob.glob(os.path.join(shared, "mn_infera_server_" + str(ip) + "_r*.log"))
    if not matches:
        return None
    # Newest by mtime: the current (failed) launch overwrites/appends this file.
    path = max(matches, key=lambda p: os.path.getmtime(p))
    tail = _tail_bytes(Path(path), max_bytes)
    return None if tail is None else "# source: " + path + "\n" + tail


def _ssh_access(state: dict) -> _SshAccess | None:
    """How to SSH into this run's pods, or None when state.json carries no usable key."""
    key_path = state.get("ssh_key_path")
    known_hosts = state.get("ssh_known_hosts")
    if not key_path or not known_hosts:
        return None
    return _SshAccess(
        key_path=key_path,
        known_hosts=known_hosts,
        default_port=int(state.get("ssh_port") or 2233),
        remote_path=os.environ.get(
            "HYPERLOOM_MN_SERVER_LOG_PATH", str(Path(tempfile.gettempdir()) / "mn_infera_server.log")
        ),
    )


def _pod_server_log(pod: dict, ip: str, access: _SshAccess | None) -> _CollectedLog | None:
    """One pod's server log: the shared-FS copy when there is one, else an SSH tail of the pod-local log."""
    shared_tail = _shared_worker_log_tail(ip)
    if shared_tail is not None:
        return _CollectedLog(shared_tail, "shared-fs")
    if access is None:
        return None
    from hyperloom.inference_optimizer.multi_node import SshTarget, ssh_run

    try:
        port = int(pod.get("sshPort") or access.default_port)
        cp = ssh_run(
            SshTarget(host=ip, key_path=access.key_path, known_hosts=access.known_hosts, port=port),
            "tail -c 2000000 " + access.remote_path + " 2>/dev/null",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    body = cp.stdout or ""
    if cp.stderr:
        body += "\n--- ssh stderr ---\n" + cp.stderr
    return _CollectedLog(body, "ssh:" + str(access.remote_path))


def _write_post_mortem(dest: str, collected: _CollectedLog, reason: str) -> None:
    """Write one pod's captured server log, headed by the failure that asked for it."""
    try:
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write("# collected on restart failure (" + collected.source + "): " + str(reason) + "\n")
            fh.write(collected.body)
    except OSError:
        return
    log.info("collected worker server log -> %s", dest)


def _post_mortem_dir() -> str:
    """The session directory this run writes post-mortem server logs into, or ``""`` when there is none."""
    sess = (
        os.environ.get("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR")
        or os.environ.get("INFERENCE_OPTIMIZER_SESSION_DIR")
        or ""
    )
    if not sess:
        return ""
    out_dir = os.path.join(sess, "server_logs")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        return ""
    return out_dir


def collect_worker_server_logs(state: dict, reason: str) -> None:
    """Best-effort: capture each prefill/decode pod sglang server log into ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/server_logs`` when a restart fails its health/serving probe, so a 503 / KV handoff / cold-JIT failure leaves a post-mortem after the pods are torn down."""
    import time as _t

    out_dir = _post_mortem_dir()
    if not out_dir:
        return
    ts = _t.strftime("%Y%m%dT%H%M%SZ", _t.gmtime())
    access = _ssh_access(state)
    for role in ("prefill", "decode", "worker"):
        for pod in state.get(role + "_pods") or []:
            ip = pod.get("podIP")
            if not ip:
                continue
            collected = _pod_server_log(pod, str(ip), access)
            if collected is None:
                continue
            tag = pod.get("podId") or ip
            _write_post_mortem(os.path.join(out_dir, ts + "_" + role + "_" + str(tag) + ".log"), collected, reason)
