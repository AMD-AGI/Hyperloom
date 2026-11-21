# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SSH restart for atomesh and sglang-router pod hand-offs.

The platform launcher waits on the server pid. InferenceSuite's host driver execs the new
command into that pid so the pod is not recreated with the original entrypoint.
The platform runs the router; each prefill and decode pod receives its role's ``SGLANG_TORCH_PROFILER_DIR``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from inference_suite.multinode import engine_liveness_script, in_place_restart_payload

from .._internal.log import err, info, warn
from .._internal.pd_cluster import (
    ENGINE_NEEDLE,
    ENGINE_PORT,
    LaunchSpec,
    launch_script,
    launch_spec,
    role_engine_pods,
    set_pods,
    uses_split_role_flags,
)
from .._internal.pd_handoff import PD_BACKENDS, TRACE_SHARE_ERROR_KEY, TRACE_SHARED_KEY
from .._internal.pod_ssh import (
    CONTAINER_ENVIRON_PATH,
    EXIT_CONFIG_ERROR,
    POD_FORWARD_ENV_PREFIXES,
    ConfigurationError,
    _collect_forward_env,
    _extract_pod_json,
    _infera_ssh_bash_with_env,
    _load_state,
    _record_last_restart,
    _save_state,
    container_env_line,
)
from .._internal.server_args_safety import ServerArgsRejected, validate_server_args
from ..pd_session import PROFILER_DIR_ENV


def is_pod_handoff(state: dict[str, Any]) -> bool:
    """Whether a multi-node state is an atomesh or sglangrouter pod hand-off."""
    return str(state.get("backend") or "") in PD_BACKENDS


def kill_pd_cluster(state: dict[str, Any]) -> int:
    """Leave a pod hand-off's engines running: restart-server replaces them in place."""
    info(f"kill-inference ({state.get('backend')}): restart-server replaces the engines in place; nothing to stop")
    return 0


def _require_state() -> dict[str, Any]:
    """Load the state and require an atomesh or sglangrouter hand-off with SSH access to both roles."""
    state = _load_state()
    backend = str(state.get("backend") or "")
    if backend not in PD_BACKENDS:
        raise ConfigurationError(f"state backend is {backend!r}, not atomesh or sglangrouter")
    if not state.get("ssh_key_path"):
        raise ConfigurationError("no ssh_key_path in state; set HYPERLOOM_MN_EXT_SSH_KEY")
    if not state.get("ssh_port"):
        raise ConfigurationError("no ssh_port in state; set HYPERLOOM_MN_EXT_SSH_PORT")
    if not state.get("prefill_pod_ips") or not state.get("decode_pod_ips"):
        raise ConfigurationError(
            "atomesh/sglangrouter restart needs HYPERLOOM_MN_EXT_PREFILL_IPS and HYPERLOOM_MN_EXT_DECODE_IPS"
        )
    return state


def _pod_env(extra_env: dict[str, str], pod: dict[str, Any]) -> dict[str, str]:
    """``extra_env`` with an engine pod's ``SGLANG_TORCH_PROFILER_DIR`` narrowed to ``<dir>/<role>``."""
    role = str(pod.get("role") or "")
    profiler_dir = str(extra_env.get(PROFILER_DIR_ENV) or "").rstrip("/")
    if role not in ("prefill", "decode") or not profiler_dir:
        return extra_env
    return {**extra_env, PROFILER_DIR_ENV: f"{profiler_dir}/{role}"}


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
    body = launch_script(pod["launch_command"], _pod_env(extra_env, pod), log_ip=ip)
    payload = in_place_restart_payload(role, body, ENGINE_NEEDLE)
    info(f"{state.get('backend')} restart: ssh -> {ip}:{port} role={role}")
    try:
        completed = _infera_ssh_bash_with_env(state, ip, payload, timeout=timeout, port=port)
    except subprocess.TimeoutExpired:
        return {"podIP": ip, "sshPort": port, "role": role, "rc": 124, "error": "timeout"}
    record: dict[str, Any] = {
        "podIP": ip,
        "sshPort": port,
        "role": role,
        "rc": completed.returncode,
    }
    parsed = _extract_pod_json(completed.stdout or "")
    if parsed:
        record["summary"] = parsed
    if completed.returncode != 0:
        record["stderr"] = (completed.stderr or "")[-1500:]
    return record


def _engine_pods(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefill then decode pod records that carry an IP."""
    return role_engine_pods(state, "prefill") + role_engine_pods(state, "decode")


def _flag_probe_script(environ_path: str = CONTAINER_ENVIRON_PATH) -> str:
    """Bash that lists ``sglang.launch_server --help`` options under the container environment in ``environ_path``."""
    return (
        container_env_line(environ_path)
        + "python3 -m sglang.launch_server --help 2>/dev/null | grep -oE -- '--[a-z][a-z0-9-]*' | sort -u"
    )


def probe_engine_flags(state: dict[str, Any], spec: LaunchSpec, *, timeout: int) -> frozenset[str]:
    """Option strings of the pod's ``sglang.launch_server``, cached in ``state``.

    Probed only when the round carries a ``SPLIT_ROLE_FLAGS`` flag. Empty when
    the probe fails, which leaves the launch args unchanged.
    """
    cached = state.get("engine_cli_flags")
    if isinstance(cached, list):
        return frozenset(str(flag) for flag in cached)
    words = shlex.split(" ".join((spec.server_args, spec.prefill_extra, spec.decode_extra)))
    pods = _engine_pods(state)
    if not uses_split_role_flags(words) or not pods:
        return frozenset()
    pod = pods[0]
    ip = str(pod["podIP"])
    port = int(pod.get("sshPort") or state.get("ssh_port") or 0)
    script = _flag_probe_script()
    try:
        completed = _infera_ssh_bash_with_env(state, ip, script, timeout=timeout, port=port)
    except subprocess.TimeoutExpired:
        warn(f"engine flag probe timed out on {ip}:{port}")
        return frozenset()
    flags = frozenset(line.strip() for line in (completed.stdout or "").splitlines() if line.startswith("--"))
    if not flags:
        warn(f"engine flag probe on {ip}:{port} listed no options; launch args are used as given")
        return frozenset()
    state["engine_cli_flags"] = sorted(flags)
    return flags


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
        script = engine_liveness_script(role, ENGINE_NEEDLE)
        try:
            completed = _infera_ssh_bash_with_env(state, ip, script, timeout=30, port=port)
        except subprocess.TimeoutExpired:
            continue
        if "HL_GONE" in (completed.stdout or ""):
            gone.append(f"{role} {ip}")
    return gone


def engines_lost_reason(state: dict[str, Any]) -> str | None:
    """Failure reason naming the engine pods whose launched server is gone, or None; None unless a pod hand-off."""
    if not is_pod_handoff(state):
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
        return EXIT_CONFIG_ERROR
    spec = launch_spec(state, args)
    if not spec.model:
        err("restart-server: model is empty")
        return EXIT_CONFIG_ERROR
    timeout = int(getattr(args, "poll_timeout", 180) or 180)
    spec = dataclasses.replace(spec, engine_flags=probe_engine_flags(state, spec, timeout=timeout))
    set_pods(state, spec)
    extra_env = _collect_forward_env(POD_FORWARD_ENV_PREFIXES)
    results = [_ssh_pod(state, pod, extra_env=extra_env, timeout=timeout) for pod in _engine_pods(state)]
    rc = 0 if results and all(int(item.get("rc") or 0) == 0 for item in results) else 1
    args.model, args.tp, args.ep = spec.model, int(getattr(args, "tp", 0) or spec.prefill_tp), spec.ep
    _record_last_restart(state, args, spec.framework, "disaggregated")
    _save_state(state)
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


def probe_profile_dir_shared(state: dict[str, Any]) -> str:
    """Check that every prefill and decode pod reads the sandbox's ``profile_trace`` dir; '' when shared, else why not."""
    root = str(state.get("profile_trace") or "").strip()
    if not root:
        return "no profile_trace dir in state"
    if not state.get("ssh_key_path"):
        return "no ssh_key_path in state"
    pods = _engine_pods(state)
    if not pods:
        return "no prefill or decode pods in state"
    token = uuid.uuid4().hex
    sentinel = Path(root) / f".share_probe-{token}"
    try:
        sentinel.write_text(token, encoding="utf-8")
    except OSError as exc:
        return f"cannot write {sentinel}: {exc}"
    try:
        for pod in pods:
            reason = _pod_reads_sentinel(state, pod, sentinel, token)
            if reason:
                return reason
    finally:
        sentinel.unlink(missing_ok=True)
    return ""


def _pod_reads_sentinel(state: dict[str, Any], pod: dict[str, Any], sentinel: Path, token: str) -> str:
    """'' when ``pod`` reads ``token`` from ``sentinel`` over SSH, else why it does not."""
    ip = str(pod["podIP"])
    port = int(pod.get("sshPort") or state.get("ssh_port") or 0) or None
    where = f"{pod.get('role') or 'engine'} pod {ip}"
    try:
        completed = _infera_ssh_bash_with_env(state, ip, f"cat {shlex.quote(str(sentinel))}", timeout=30, port=port)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"{where}: ssh failed: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-200:]
        return f"{where}: cannot read {sentinel} (rc={completed.returncode}) {detail}".strip()
    if (completed.stdout or "").strip() != token:
        return f"{where}: {sentinel} is a different file there"
    return ""


def record_profile_dir_share(state: dict[str, Any]) -> str:
    """Store the :func:`probe_profile_dir_shared` verdict on ``state`` and return its reason ('' when shared)."""
    reason = probe_profile_dir_shared(state)
    state[TRACE_SHARED_KEY] = not reason
    if reason:
        state[TRACE_SHARE_ERROR_KEY] = reason
    else:
        state.pop(TRACE_SHARE_ERROR_KEY, None)
    return reason
