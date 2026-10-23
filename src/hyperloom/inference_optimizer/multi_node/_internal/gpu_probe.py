# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Remote GPU-type probe for multi-node runs."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from hyperloom.common.gpu_identity import board_named_in

from ...gpu_types import reference_gpu_type_for_gfx
from . import ray_dashboard, ssh_client, ssh_known_hosts
from .external_state import build_external_state_from_env, external_service_url

log = logging.getLogger(__name__)

# One command that prints the product name, falling back to torch's gcnArchName (gfx942 / gfx950) when rocm-smi is
# unavailable on PATH.
_PROBE_CMD = (
    "rocm-smi --showproductname 2>/dev/null || "
    'python3 -c "import torch;'
    'print(torch.cuda.get_device_properties(0).gcnArchName)" 2>/dev/null'
)

# Ray Dashboard job terminal states (mirror multi_node.cli).
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}


def _parse_gpu_type(text: str) -> str | None:
    """Map rocm-smi / gcnArchName output to a known AMD board."""
    board = board_named_in(text)
    return board.product if board else reference_gpu_type_for_gfx(text)


def remote_autodetect_gpu_type(*, timeout_s: int = 60) -> str | None:
    """Detect the inference GPU type from the handed-over cluster."""
    if not external_service_url():
        return None
    state = build_external_state_from_env()
    backend = str(state.get("backend") or "").lower()
    try:
        if backend == "rayjob":
            return _probe_via_ray_head(state, timeout_s=timeout_s)
        if backend == "infera":
            return _probe_via_infera_ssh(state, timeout_s=timeout_s)
        log.warning("remote GPU probe: unsupported backend %r", backend)
    except (httpx.HTTPError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        log.warning("remote GPU probe (%s) failed: %s", backend, exc)
    return None


def _probe_via_ray_head(state: dict[str, Any], *, timeout_s: int) -> str | None:
    """Submit ``rocm-smi`` on the Ray head and parse its logs."""
    head = str(state.get("head_pod_ip") or "").strip()
    if not head:
        return None
    token = str(state.get("ray_dashboard_token") or "").strip() or None
    with ray_dashboard.RayDashboardClient(head, token=token) as ray:
        sub_id = ray.submit_job(_PROBE_CMD)
        deadline = time.monotonic() + max(1, timeout_s)
        status = ""
        while time.monotonic() < deadline:
            status = str(ray.get_job(sub_id).get("status", "")).upper()
            if status in _TERMINAL_STATUSES:
                break
            time.sleep(2)
        if status != "SUCCEEDED":
            # Logged, not fatal: _PROBE_CMD's `||` fallback can print a usable name and still exit non-zero, so the
            # logs are parsed either way and an unparsable result already degrades to None.
            log.warning("remote GPU probe: ray job %s ended %r", sub_id, status or "no terminal status")
        logs = ray.get_job_logs(sub_id)
    return _parse_gpu_type(logs)


def _first_gpu_pod(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first available GPU pod target (prefill, else worker, else decode)."""
    for key in ("prefill_pods", "worker_pods", "decode_pods"):
        pods = state.get(key) or []
        if pods:
            return pods[0]
    return None


def _probe_via_infera_ssh(state: dict[str, Any], *, timeout_s: int) -> str | None:
    """SSH into the first Infera GPU pod and parse ``rocm-smi`` output."""
    pod = _first_gpu_pod(state)
    key_path = str(state.get("ssh_key_path") or "").strip()
    if not pod or not key_path:
        return None
    host = str(pod.get("podIP") or "").strip()
    if not host:
        return None
    port = int(pod.get("sshPort") or state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)

    known_hosts_hint = str(state.get("ssh_known_hosts") or "").strip()
    scratch: Path | None = None
    if known_hosts_hint and Path(known_hosts_hint).is_file():
        known_hosts = Path(known_hosts_hint)
    else:
        # Keyscan the pod into a throwaway known_hosts so StrictHostKeyChecking still holds for this one-shot probe.
        scratch = Path(tempfile.mkdtemp(prefix="mn_gpu_probe_"))
        known_hosts = ssh_known_hosts.refresh_known_hosts([(host, port)], scratch / "known_hosts")

    try:
        cp = ssh_client.ssh_run(
            ssh_client.SshTarget(host=host, key_path=key_path, known_hosts=known_hosts, port=port),
            _PROBE_CMD,
            timeout=max(1, timeout_s),
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    if cp.returncode != 0:
        # Logged, not fatal: _PROBE_CMD's `||` fallback can print a usable name and still exit non-zero, so the output
        # is parsed either way and an unparsable result already degrades to None.
        log.warning("remote GPU probe: ssh to %s exited %s", host, cp.returncode)
    return _parse_gpu_type(f"{cp.stdout or ''}\n{cp.stderr or ''}")
