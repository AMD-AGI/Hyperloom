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
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ...gpu_types import _AMD_GPU_TYPES, _GFX_TO_RUNNER
from . import ray_dashboard, ssh_client, ssh_known_hosts
from .external_state import build_external_state_from_env, external_service_url

log = logging.getLogger(__name__)

# rocm-smi product-name tags, longest/most-specific first so MI325X is not
# shadowed by an MI300X substring match.
_PRODUCT_TAGS = ("MI355X", "MI325X", "MI308X", "MI300X")

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
    out = _remote_run(_PROBE_CMD, timeout_s=timeout_s, label="GPU probe")
    return _parse_gpu_type(out) if out is not None else None


# Sentinel wrapper so the value is unambiguously extractable from a remote run's
# combined output, which is interleaved with Ray runtime-env / INFO log lines.
_ENV_SENTINEL_RE = re.compile(r"___MNENV\[(.*?)\]MNENV___")


def _parse_env_value(text: str) -> str | None:
    """Extract a sentinel-wrapped env value from remote output.

    Args:
        text: Combined stdout/stderr from a remote sentinel ``echo`` run.

    Returns:
        str | None: The variable's value, or ``None`` when unset/empty/absent.
    """
    match = _ENV_SENTINEL_RE.search(text or "")
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def remote_read_env(var: str, *, timeout_s: int = 120) -> str | None:
    """Read an env var from the handed-over inference cluster's pods.

    The operator's server env (e.g. ``SGLANG_USE_AITER``) is baked into the
    inference pods, not the sandbox, so it must be read remotely: ``rayjob``
    runs the probe on the Ray head, ``infera`` SSHes into the first GPU pod. The
    value is wrapped in a sentinel so it survives interleaved Ray log lines.
    Best-effort: any failure or an unset variable returns ``None``.

    Args:
        var: The environment variable name to read on the cluster pods.
        timeout_s: Per-probe budget (job poll ceiling / SSH timeout) in seconds.

    Returns:
        str | None: The variable's value on the pods, or ``None``.
    """
    name = (var or "").strip()
    if not name:
        return None
    # Prints e.g. ``___MNENV[0]MNENV___`` (empty brackets when unset).
    cmd = f'printf "___MNENV[%s]MNENV___\\n" "$(printenv {name})"'
    out = _remote_run(cmd, timeout_s=timeout_s, label=f"env read {name}")
    return _parse_env_value(out) if out is not None else None


def _remote_run(command: str, *, timeout_s: int, label: str) -> str | None:
    """Run ``command`` on the handed-over cluster and return its raw output.

    Routes by ``state.backend``: ``rayjob`` submits the command on the Ray head
    via the Dashboard; ``infera`` SSHes into the first GPU pod. Best-effort: any
    failure (no hand-off, unreachable head/pod) returns ``None``.

    Args:
        command: Shell command to run on the cluster.
        timeout_s: Per-run budget (job poll ceiling / SSH timeout) in seconds.
        label: Short label used in warning logs.

    Returns:
        str | None: The combined stdout/stderr text, or ``None`` on failure.
    """
    if not external_service_url():
        return None
    try:
        state = build_external_state_from_env()
    except Exception as exc:  # noqa: BLE001 - best-effort probe
        log.warning("remote %s: could not build external state: %s", label, exc)
        return None
    backend = str(state.get("backend") or "").lower()
    try:
        if backend == "rayjob":
            return _run_on_ray_head(state, command, timeout_s=timeout_s, label=label)
        if backend == "infera":
            return _run_on_infera_ssh(state, command, timeout_s=timeout_s, label=label)
        log.warning("remote %s: unsupported backend %r", label, backend)
    except Exception as exc:  # noqa: BLE001 - best-effort probe
        log.warning("remote %s (%s) failed: %s", label, backend, exc)
    return None


def _run_on_ray_head(state: dict[str, Any], command: str, *, timeout_s: int, label: str) -> str | None:
    """Submit ``command`` on the Ray head and return its raw job logs.

    Args:
        state: Multi-node state carrying ``head_pod_ip`` (+ optional token).
        command: Shell command to submit as the Ray job entrypoint.
        timeout_s: Poll ceiling in seconds for the job.
        label: Short label used in warning logs.

    Returns:
        str | None: The job logs, or ``None`` when the head IP is missing.
    """
    head = str(state.get("head_pod_ip") or "").strip()
    if not head:
        return None
    token = str(state.get("ray_dashboard_token") or "").strip() or None
    with ray_dashboard.RayDashboardClient(head, token=token) as ray:
        sub_id = ray.submit_job(command)
        deadline = time.monotonic() + max(1, timeout_s)
        status = ""
        while time.monotonic() < deadline:
            status = str(ray.get_job(sub_id).get("status", "")).upper()
            if status in _TERMINAL_STATUSES:
                break
            time.sleep(2)
        if status != "SUCCEEDED":
            # Logged, not fatal: the logs are parsed either way and an
            # unparsable/empty result already degrades to None downstream.
            log.warning("remote %s: ray job %s ended %r", label, sub_id, status or "no terminal status")
        return ray.get_job_logs(sub_id)


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


def _run_on_infera_ssh(state: dict[str, Any], command: str, *, timeout_s: int, label: str) -> str | None:
    """SSH into the first Infera GPU pod, run ``command``, return raw output.

    Args:
        state: Multi-node state carrying SSH key / port and pod targets.
        command: Shell command to run on the pod.
        timeout_s: SSH subprocess timeout in seconds.
        label: Short label used in warning logs.

    Returns:
        str | None: Combined stdout/stderr, or ``None`` when SSH is unavailable.
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
        # still holds for this one-shot command.
        scratch = Path(tempfile.mkdtemp(prefix="mn_probe_"))
        known_hosts = ssh_known_hosts.refresh_known_hosts([(host, port)], scratch / "known_hosts")

    try:
        cp = ssh_client.ssh_run(
            host,
            command,
            key_path=key_path,
            known_hosts=known_hosts,
            port=port,
            timeout=max(1, timeout_s),
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    if cp.returncode != 0:
        # Logged, not fatal: output is parsed either way and an unparsable/empty
        # result already degrades to None downstream.
        log.warning("remote %s: ssh to %s exited %s", label, host, cp.returncode)
    return f"{cp.stdout or ''}\n{cp.stderr or ''}"
