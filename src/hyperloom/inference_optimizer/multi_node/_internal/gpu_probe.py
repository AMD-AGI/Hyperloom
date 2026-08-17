# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Remote GPU-type probe for multi-node runs.

The optimizer CLI runs in a sandbox pod that has no GPU of its own, so the
local ``rocm-smi`` / torch probe in :mod:`gpu_types` returns nothing on a
``--nodes>=2`` run. This module instead detects the *real* inference GPU on the
handed-over cluster: for the ``rayjob`` backend it submits ``rocm-smi`` on the
Ray head via the Dashboard REST API, and for ``infera`` it SSHes into the first
GPU pod. Every probe is best-effort -- any failure returns ``None`` so the
caller can fall back to the ``--gpu-type`` / ``$GPU_TYPE`` hint.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ...gpu_types import _AMD_GPU_TYPES, _GFX_TO_RUNNER, _PRODUCT_TAGS
from . import ray_dashboard, ssh_client, ssh_known_hosts
from .external_state import build_external_state_from_env, external_service_url

log = logging.getLogger(__name__)

# One command that prints the product name, falling back to torch's
# gcnArchName (gfx942 / gfx950) when rocm-smi is unavailable on PATH.
_PROBE_CMD = (
    "rocm-smi --showproductname 2>/dev/null || "
    'python3 -c "import torch;'
    'print(torch.cuda.get_device_properties(0).gcnArchName)" 2>/dev/null'
)

# Ray Dashboard job terminal states (mirror multi_node.cli).
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}


def _parse_gpu_type(text: str) -> str | None:
    """Map rocm-smi / gcnArchName output to a known AMD runner type.

    Args:
        text: Combined stdout/stderr from the remote probe command.

    Returns:
        str | None: ``mi300x`` / ``mi308x`` / ``mi325x`` / ``mi355x`` when a
        product tag or gfx arch is recognized, else ``None``.
    """
    upper = (text or "").upper()
    for tag in _PRODUCT_TAGS:
        if tag in upper:
            candidate = tag.lower()
            if candidate in _AMD_GPU_TYPES:
                return candidate
    lower = (text or "").lower()
    for gfx, runner in _GFX_TO_RUNNER.items():
        if gfx in lower:
            return runner
    return None


def remote_autodetect_gpu_type(*, timeout_s: int = 60) -> str | None:
    """Detect the inference GPU type from the handed-over cluster.

    Routes by ``state.backend``: ``rayjob`` submits the probe on the Ray head
    via the Dashboard; ``infera`` SSHes into the first GPU pod. Best-effort: any
    error (no hand-off, unreachable head/pod, unparseable output) returns
    ``None`` so the caller falls back to the ``--gpu-type`` / ``$GPU_TYPE`` hint.

    Args:
        timeout_s: Per-probe budget (job poll ceiling / SSH timeout) in seconds.

    Returns:
        str | None: The resolved AMD runner type, or ``None`` when undetectable.
    """
    if not external_service_url():
        return None
    try:
        state = build_external_state_from_env()
    except Exception as exc:  # noqa: BLE001 - best-effort probe
        log.warning("remote GPU probe: could not build external state: %s", exc)
        return None
    backend = str(state.get("backend") or "").lower()
    try:
        if backend == "rayjob":
            return _probe_via_ray_head(state, timeout_s=timeout_s)
        if backend == "infera":
            return _probe_via_infera_ssh(state, timeout_s=timeout_s)
        log.warning("remote GPU probe: unsupported backend %r", backend)
    except Exception as exc:  # noqa: BLE001 - best-effort probe
        log.warning("remote GPU probe (%s) failed: %s", backend, exc)
    return None


def _probe_via_ray_head(state: dict[str, Any], *, timeout_s: int) -> str | None:
    """Submit ``rocm-smi`` on the Ray head and parse its logs.

    Args:
        state: Multi-node state carrying ``head_pod_ip`` (+ optional token).
        timeout_s: Poll ceiling in seconds for the probe job.

    Returns:
        str | None: The detected GPU type, or ``None`` on any failure.
    """
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
            # Logged, not fatal: _PROBE_CMD's `||` fallback can print a usable
            # name and still exit non-zero, so the logs are parsed either way
            # and an unparsable result already degrades to None.
            log.warning("remote GPU probe: ray job %s ended %r", sub_id, status or "no terminal status")
        logs = ray.get_job_logs(sub_id)
    return _parse_gpu_type(logs)


def _first_gpu_pod(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first available GPU pod target (prefill, else worker, else decode).

    Args:
        state: Multi-node state carrying the per-role pod target lists.

    Returns:
        dict | None: A pod target (``podIP`` / ``sshPort``), or ``None``.
    """
    for key in ("prefill_pods", "worker_pods", "decode_pods"):
        pods = state.get(key) or []
        if pods:
            return pods[0]
    return None


def _probe_via_infera_ssh(state: dict[str, Any], *, timeout_s: int) -> str | None:
    """SSH into the first Infera GPU pod and parse ``rocm-smi`` output.

    Args:
        state: Multi-node state carrying SSH key / port and pod targets.
        timeout_s: SSH subprocess timeout in seconds.

    Returns:
        str | None: The detected GPU type, or ``None`` on any failure.
    """
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
        # Keyscan the pod into a throwaway known_hosts so StrictHostKeyChecking
        # still holds for this one-shot probe.
        scratch = Path(tempfile.mkdtemp(prefix="mn_gpu_probe_"))
        known_hosts = ssh_known_hosts.refresh_known_hosts([(host, port)], scratch / "known_hosts")

    try:
        cp = ssh_client.ssh_run(
            host,
            _PROBE_CMD,
            key_path=key_path,
            known_hosts=known_hosts,
            port=port,
            timeout=max(1, timeout_s),
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    if cp.returncode != 0:
        # Logged, not fatal: _PROBE_CMD's `||` fallback can print a usable name
        # and still exit non-zero, so the output is parsed either way and an
        # unparsable result already degrades to None.
        log.warning("remote GPU probe: ssh to %s exited %s", host, cp.returncode)
    return _parse_gpu_type(f"{cp.stdout or ''}\n{cp.stderr or ''}")
