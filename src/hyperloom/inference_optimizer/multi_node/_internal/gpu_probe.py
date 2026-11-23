# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cards of a multi-node run's handed-over cluster, read by ``amd-smi static --json`` on one of its GPU nodes."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from hyperloom_common import amd_smi
from hyperloom_common.amd_smi import AmdSmiError
from hyperloom_common.gpu_card import GpuCard, cards_of

from . import ray_dashboard, ssh_client, ssh_known_hosts
from .external_state import build_external_state_from_env, external_service_url

_PROBE_CMD = "amd-smi static --json"

# Ray Dashboard job terminal states (mirror multi_node.cluster_access).
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "STOPPED"}

_DOCUMENT_START = re.compile(r"[\[{]")


def remote_cards(*, timeout_s: int = 60) -> list[GpuCard]:
    """The cards one GPU node of the handed-over cluster reports, in amd-smi order.

    Raises:
        AmdSmiError: no cluster was handed over, its backend has no probe, the probe could not run, or what it
            printed describes no card.
    """
    if not external_service_url():
        raise AmdSmiError("no multi-node cluster was handed over to read the cards of")
    state = build_external_state_from_env()
    backend = str(state.get("backend") or "").lower()
    probes = {"rayjob": _probe_via_ray_head, "infera": _probe_via_infera_ssh}
    if backend not in probes:
        raise AmdSmiError(f"the multi-node backend {backend!r} has no remote amd-smi probe")
    try:
        output = probes[backend](state, timeout_s=timeout_s)
    except (httpx.HTTPError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise AmdSmiError(f"remote amd-smi static via {backend} failed: {exc}") from exc
    cards = cards_of(_amd_smi_document(output))
    return [cards[index] for index in sorted(cards)]


def _amd_smi_document(output: str) -> Any:
    """The first ``amd-smi --json`` document in ``output``, which a job log may wrap in lines of its own."""
    decoder = json.JSONDecoder()
    for start in _DOCUMENT_START.finditer(output):
        try:
            document, _ = decoder.raw_decode(output, start.start())
        except ValueError:
            continue
        if amd_smi.gpu_rows(document):
            return document
    raise AmdSmiError(f"the remote {_PROBE_CMD} printed no card: {output.strip()[-500:]!r}")


def _probe_via_ray_head(state: dict[str, Any], *, timeout_s: int) -> str:
    """Submit the probe on the Ray head and return its logs."""
    head = str(state.get("head_pod_ip") or "").strip()
    if not head:
        raise AmdSmiError("the Ray hand-off names no head pod IP")
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
            raise AmdSmiError(f"ray job {sub_id} ended {status or 'without a terminal status'}")
        return str(ray.get_job_logs(sub_id))


def _first_gpu_pod(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first available GPU pod target (prefill, else worker, else decode)."""
    for key in ("prefill_pods", "worker_pods", "decode_pods"):
        pods = state.get(key) or []
        if pods:
            return pods[0]
    return None


def _probe_via_infera_ssh(state: dict[str, Any], *, timeout_s: int) -> str:
    """Run the probe over SSH on the first Infera GPU pod and return what it printed."""
    pod = _first_gpu_pod(state) or {}
    key_path = str(state.get("ssh_key_path") or "").strip()
    host = str(pod.get("podIP") or "").strip()
    if not (host and key_path):
        raise AmdSmiError("the Infera hand-off names no GPU pod IP or no SSH key")
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
        raise AmdSmiError(f"ssh to {host}: {_PROBE_CMD} exited {cp.returncode}: {(cp.stderr or '').strip()[-500:]}")
    return cp.stdout or ""
