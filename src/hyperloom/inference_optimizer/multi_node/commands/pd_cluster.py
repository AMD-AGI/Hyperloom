# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SSH restart for atomesh and sglang-router clusters.

The platform launcher waits on the server pid. The remote helper execs the new
command into that pid so the pod is not recreated with the original entrypoint.
The platform runs the router.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from typing import Any

from .. import cluster_access as _mn_cli
from .._internal.log import err, info
from .._internal.pd_cluster import (
    ENGINE_NEEDLE,
    ENGINE_PORT,
    PD_ROUTER_BACKENDS,
    launch_script,
    launch_spec,
    role_engine_pods,
    set_pods,
)
from .._internal.pd_remote import bash_payload, launch_path, pidfile_path
from .._internal.server_args_safety import ServerArgsRejected, validate_server_args
from .._internal.topology_file import normalize_mn_backend
from ..pod_entrypoints import extract_pod_json
from .infera import _collect_forward_env, _record_last_restart


def _require_state() -> dict[str, Any]:
    """Load the state and require an atomesh or sglangrouter hand-off with SSH access to both roles."""
    state = _mn_cli._load_state()
    backend = str(state.get("backend") or "")
    if backend not in PD_ROUTER_BACKENDS:
        raise _mn_cli.ConfigurationError(f"state backend is {backend!r}, not atomesh or sglangrouter")
    if not state.get("ssh_key_path"):
        raise _mn_cli.ConfigurationError("no ssh_key_path in state; set HYPERLOOM_MN_EXT_SSH_KEY")
    if not state.get("ssh_port"):
        raise _mn_cli.ConfigurationError("no ssh_port in state; set HYPERLOOM_MN_EXT_SSH_PORT")
    if not state.get("prefill_pod_ips") or not state.get("decode_pod_ips"):
        raise _mn_cli.ConfigurationError(
            "atomesh/sglangrouter restart needs HYPERLOOM_MN_EXT_PREFILL_IPS and HYPERLOOM_MN_EXT_DECODE_IPS"
        )
    return state


def _ssh_pod(
    state: dict[str, Any],
    pod: dict[str, Any],
    *,
    extra_env: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    """Run the in-place restart payload on one pod and return its result record."""
    role = str(pod.get("role") or "")
    ip = str(pod.get("podIP") or "")
    port = pod["sshPort"]
    body = launch_script(pod["launch_command"], extra_env, log_ip=ip)
    payload = bash_payload(role, body, ENGINE_NEEDLE)
    info(f"{state.get('backend')} restart: ssh -> {ip}:{port} role={role}")
    try:
        completed = _mn_cli._infera_ssh_bash_with_env(state, ip, payload, timeout=timeout, port=port)
    except subprocess.TimeoutExpired:
        return {"podIP": ip, "sshPort": port, "role": role, "rc": 124, "error": "timeout"}
    record: dict[str, Any] = {
        "podIP": ip,
        "sshPort": port,
        "role": role,
        "rc": completed.returncode,
    }
    parsed = extract_pod_json(completed.stdout or "")
    if parsed:
        record["summary"] = parsed
    if completed.returncode != 0:
        record["stderr"] = (completed.stderr or "")[-1500:]
    return record


def _engine_pods(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefill then decode pod records that carry an IP."""
    return role_engine_pods(state, "prefill") + role_engine_pods(state, "decode")


def engines_not_running(state: dict[str, Any]) -> list[str]:
    """``"<role> <ip>"`` of each engine pod whose launched server is no longer running.

    A pod whose container restarted has lost the pidfile and runs its platform
    entrypoint instead of the launched command. A pid still running the launch
    script (waiting for the GPU before it execs the server) counts as running.
    Unreachable pods are not listed.
    """
    gone: list[str] = []
    for pod in _engine_pods(state):
        role = str(pod.get("role") or "")
        ip = str(pod["podIP"])
        port = int(pod.get("sshPort") or state.get("ssh_port") or 0)
        script = (
            f'cmd=""; pid=$(cat {shlex.quote(pidfile_path(role))} 2>/dev/null) && kill -0 "$pid" 2>/dev/null '
            f"&& cmd=$(tr '\\0' ' ' < /proc/\"$pid\"/cmdline 2>/dev/null); "
            f'if printf "%s" "$cmd" | grep -qF -e {ENGINE_NEEDLE}; then echo HL_ALIVE; '
            f'elif printf "%s" "$cmd" | grep -qF -e {shlex.quote(launch_path(role))}; then echo HL_STARTING; '
            "else echo HL_GONE; fi"
        )
        try:
            completed = _mn_cli._infera_ssh_bash_with_env(state, ip, script, timeout=30, port=port)
        except subprocess.TimeoutExpired:
            continue
        if "HL_GONE" in (completed.stdout or ""):
            gone.append(f"{role} {ip}")
    return gone


def engines_lost_reason(state: dict[str, Any]) -> str | None:
    """Failure reason naming the engine pods whose launched server is gone, or None; None for non-PD-router backends."""
    if normalize_mn_backend(state.get("backend")) not in PD_ROUTER_BACKENDS:
        return None
    gone = engines_not_running(state)
    if not gone:
        return None
    return (
        f"launched engine exited on {gone}; the container restarted with its platform entrypoint, so /health no "
        "longer reflects this round's launch args (the engine's error is in the pod's previous container log)"
    )


def restart_pd_cluster(args: argparse.Namespace) -> int:
    """SSH-restart every prefill and decode pod."""
    state = _require_state()
    try:
        for flag in ("extra_args", "pd_prefill_extra_args", "pd_decode_extra_args"):
            validate_server_args(getattr(args, flag, "") or "", context=f"restart-server --{flag.replace('_', '-')}")
    except ServerArgsRejected as exc:
        err(str(exc))
        return _mn_cli.EXIT_CONFIG_ERROR
    spec = launch_spec(state, args)
    if not spec.model:
        err("restart-server: model is empty")
        return _mn_cli.EXIT_CONFIG_ERROR
    timeout = int(getattr(args, "poll_timeout", 180) or 180)
    set_pods(state, spec)
    extra_env = _collect_forward_env()
    results = [_ssh_pod(state, pod, extra_env=extra_env, timeout=timeout) for pod in _engine_pods(state)]
    rc = 0 if results and all(int(item.get("rc") or 0) == 0 for item in results) else 1
    args.model, args.tp, args.ep = spec.model, int(getattr(args, "tp", 0) or spec.prefill_tp), spec.ep
    _record_last_restart(state, args, spec.framework, "disaggregated")
    _mn_cli._save_state(state)
    print(
        json.dumps(
            {"backend": state.get("backend"), "pd_mode": "disaggregated", "rc": rc, "results": results},
            indent=2,
        )
    )
    if rc != 0:
        info("pd-cluster restart: at least one pod failed")
        return 1
    info(f"pd-cluster servers launched on :{ENGINE_PORT}; benchmark via $service_url")
    return 0
