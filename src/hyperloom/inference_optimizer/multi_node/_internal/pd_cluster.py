# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomesh and SGLang-router multi-node cluster records.

The state file is the contract other tools read (path in ``$MULTI_NODE_STATE``).
Launch commands match the platform entrypoints: atomesh or sglang_router in
front, ``sglang.launch_server`` on every prefill and decode pod. Every pod
uses the one SSH port from ``HYPERLOOM_MN_EXT_SSH_PORT`` (no per-role stride).
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Callable

from ..state_paths import resolve_state_file, state_file_safe_to_read

PD_ROUTER_BACKENDS = ("atom", "sglangrouter")
ENGINE_PORT = 8000
PREFILL_BOOTSTRAP_PORT = 8998
DECODE_BOOTSTRAP_PORT = 9098

Lookup = Callable[[str], str]


@dataclass(frozen=True)
class LaunchSpec:
    """Inputs for one round of router and engine commands."""

    backend: str
    model: str
    framework: str
    prefill_tp: int
    decode_tp: int
    ep: int
    kv: str
    ib: str
    server_args: str
    prefill_extra: str
    decode_extra: str
    prefill_url: str
    decode_url: str


def _words(raw: str) -> list[str]:
    """Split a server-arg string the way the shell would."""
    text = (raw or "").strip()
    return shlex.split(text) if text else []


def _join(argv: list[str]) -> str:
    """Shell-quote an argv into one command line."""
    return " ".join(shlex.quote(part) for part in argv)


def engine_command(role: str, spec: LaunchSpec) -> str:
    """sglang prefill or decode command for one pod."""
    tp = spec.prefill_tp if role == "prefill" else spec.decode_tp
    bootstrap = PREFILL_BOOTSTRAP_PORT if role == "prefill" else DECODE_BOOTSTRAP_PORT
    extra = _words(spec.server_args) + _words(spec.prefill_extra if role == "prefill" else spec.decode_extra)
    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(ENGINE_PORT),
        "--tp-size",
        str(tp),
        "--disaggregation-mode",
        role,
        "--disaggregation-transfer-backend",
        spec.kv or "mooncake",
        "--disaggregation-bootstrap-port",
        str(bootstrap),
    ]
    if spec.ib.strip():
        argv.extend(["--disaggregation-ib-device", spec.ib.strip()])
    if spec.ep > 1:
        argv.extend(["--ep-size", str(spec.ep)])
    argv.extend(extra)
    return _join(argv)


def router_command(spec: LaunchSpec) -> str:
    """Router command. Atomesh for ``atom``, sglang_router for ``sglangrouter``."""
    if spec.backend == "sglangrouter":
        argv = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            spec.prefill_url,
            str(PREFILL_BOOTSTRAP_PORT),
            "--decode",
            spec.decode_url,
            "--host",
            "0.0.0.0",
            "--port",
            str(ENGINE_PORT),
        ]
    else:
        argv = [
            "atomesh",
            "launch",
            "--host",
            "0.0.0.0",
            "--port",
            str(ENGINE_PORT),
            "--pd-disaggregation",
            "--prefill",
            spec.prefill_url,
            str(PREFILL_BOOTSTRAP_PORT),
            "--decode",
            spec.decode_url,
            "--backend",
            spec.framework or "sglang",
        ]
    return _join(argv)


def command_for(role: str, spec: LaunchSpec) -> str:
    """Launch command recorded on one pod."""
    if role == "router":
        return router_command(spec)
    return engine_command(role, spec)


def cmdline_needle(role: str, backend: str) -> str:
    """Substring that identifies the server process on a pod of this role."""
    if role == "router":
        return "sglang_router.launch_router" if backend == "sglangrouter" else "atomesh"
    return "sglang.launch_server"


def _csv(get: Lookup, name: str) -> list[str]:
    return [part.strip() for part in get(name).split(",") if part.strip()]


def role_ips(get: Lookup) -> tuple[list[str], list[str], list[str]]:
    """Router, prefill, and decode IPs from the hand-off."""
    return (
        _csv(get, "HYPERLOOM_MN_EXT_ROUTER_IPS"),
        _csv(get, "HYPERLOOM_MN_EXT_PREFILL_IPS"),
        _csv(get, "HYPERLOOM_MN_EXT_DECODE_IPS"),
    )


def ssh_port_from_handoff(get: Lookup) -> int | None:
    """The shared SSH port, or None when the hand-off does not name one.

    Unlike Infera, atom and sglangrouter do not invent a port and do not add a
    decode stride. ``HYPERLOOM_MN_EXT_SSH_PORT`` is the port on every pod.
    """
    raw = get("HYPERLOOM_MN_EXT_SSH_PORT").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        return None
    if port < 1 or port > 65535:
        return None
    return port


def pd_cluster_has_control(get: Lookup) -> bool:
    """True when this hand-off can SSH-restart an atom or sglangrouter cluster."""
    if get("INFERENCE_OPTIMIZER_MN_BACKEND").strip().lower() not in PD_ROUTER_BACKENDS:
        return False
    if not get("HYPERLOOM_MN_EXT_SSH_KEY").strip():
        return False
    if ssh_port_from_handoff(get) is None:
        return False
    _router, prefill, decode = role_ips(get)
    return bool(prefill and decode)


def _positive(raw: str, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def spec_from_env(state: dict[str, Any], get: Lookup | None = None) -> LaunchSpec:
    """Build a launch spec from the process environment and cluster URLs."""
    lookup = get or (lambda name: os.environ.get(name, ""))
    gpus = _positive(str(state.get("gpus_per_node") or ""), 8)
    tp = _positive(os.environ.get("TP", ""), gpus)
    prefill_ips = list(state.get("prefill_pod_ips") or [])
    decode_ips = list(state.get("decode_pod_ips") or [])
    prefill_url = lookup("HYPERLOOM_MN_EXT_PREFILL_URL").strip()
    decode_url = lookup("HYPERLOOM_MN_EXT_DECODE_URL").strip()
    if not prefill_url and prefill_ips:
        prefill_url = f"http://{prefill_ips[0]}:{ENGINE_PORT}"
    if not decode_url and decode_ips:
        decode_url = f"http://{decode_ips[0]}:{ENGINE_PORT}"
    return LaunchSpec(
        backend=str(state.get("backend") or "atom"),
        model=os.environ.get("MODEL_PATH", "").strip(),
        framework=(os.environ.get("FRAMEWORK", "") or "sglang").strip().lower() or "sglang",
        prefill_tp=_positive(os.environ.get("PD_PREFILL_TP", ""), tp),
        decode_tp=_positive(os.environ.get("PD_DECODE_TP", ""), tp),
        ep=_positive(os.environ.get("EP", ""), 1),
        kv=os.environ.get("PD_TRANSFER_BACKEND", "").strip(),
        ib=os.environ.get("PD_IB_DEVICE", "").strip(),
        server_args=os.environ.get("INFERENCE_OPTIMIZER_SERVER_ARGS", "").strip(),
        prefill_extra=os.environ.get("PD_PREFILL_EXTRA_ARGS", "").strip(),
        decode_extra=os.environ.get("PD_DECODE_EXTRA_ARGS", "").strip(),
        prefill_url=prefill_url,
        decode_url=decode_url,
    )


def _endpoint(explicit: str, ips: list[str]) -> str:
    text = (explicit or "").strip()
    if text:
        return text
    return f"http://{ips[0]}:{ENGINE_PORT}" if ips else ""


def spec_from_restart(args: Any, state: dict[str, Any]) -> LaunchSpec:
    """Build a launch spec from one restart-server invocation."""
    gpus = _positive(str(state.get("gpus_per_node") or ""), 8)
    tp = _positive(str(getattr(args, "tp", 0) or 0), _positive(os.environ.get("TP", ""), gpus))
    prefill_ips = [str(ip) for ip in (state.get("prefill_pod_ips") or [])]
    decode_ips = [str(ip) for ip in (state.get("decode_pod_ips") or [])]
    prefill_url = _endpoint(os.environ.get("HYPERLOOM_MN_EXT_PREFILL_URL", ""), prefill_ips)
    decode_url = _endpoint(os.environ.get("HYPERLOOM_MN_EXT_DECODE_URL", ""), decode_ips)
    return LaunchSpec(
        backend=str(state.get("backend") or "atom"),
        model=str(getattr(args, "model", "") or os.environ.get("MODEL_PATH", "")).strip(),
        framework=str(getattr(args, "framework", "") or os.environ.get("FRAMEWORK", "") or "sglang").strip().lower(),
        prefill_tp=_positive(str(getattr(args, "pd_prefill_tp", 0) or 0), tp),
        decode_tp=_positive(str(getattr(args, "pd_decode_tp", 0) or 0), tp),
        ep=_positive(str(getattr(args, "ep", 0) or 0), _positive(os.environ.get("EP", ""), 1)),
        kv=str(getattr(args, "pd_transfer_backend", "") or os.environ.get("PD_TRANSFER_BACKEND", "")).strip(),
        ib=str(getattr(args, "pd_ib_device", "") or os.environ.get("PD_IB_DEVICE", "")).strip(),
        server_args=str(getattr(args, "extra_args", "") or "").strip(),
        prefill_extra=str(getattr(args, "pd_prefill_extra_args", "") or "").strip(),
        decode_extra=str(getattr(args, "pd_decode_extra_args", "") or "").strip(),
        prefill_url=prefill_url,
        decode_url=decode_url,
    )


def _pod(role: str, index: int, ip: str, port: int | None, spec: LaunchSpec) -> dict[str, Any]:
    return {
        "role": role,
        "podIP": ip,
        "podId": f"external-{role}-{index}",
        "sshPort": port,
        "launch_command": command_for(role, spec),
    }


def pods_for(
    router: list[str],
    prefill: list[str],
    decode: list[str],
    port: int | None,
    spec: LaunchSpec,
) -> list[dict[str, Any]]:
    """One pod record per host, router then prefill then decode."""
    pods: list[dict[str, Any]] = []
    for index, ip in enumerate(router):
        pods.append(_pod("router", index, ip, port, spec))
    for index, ip in enumerate(prefill):
        pods.append(_pod("prefill", index, ip, port, spec))
    for index, ip in enumerate(decode):
        pods.append(_pod("decode", index, ip, port, spec))
    return pods


def _pin_profile_dir(state: dict[str, Any]) -> None:
    """Point profile_trace at a stable directory under the session."""
    explicit = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if not explicit:
        session = os.environ.get("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "").strip()
        if session:
            path = os.path.join(session, "runs", "profile", "torch_trace")
            try:
                os.makedirs(path, mode=0o755, exist_ok=True)
            except OSError:
                path = ""
            if path:
                os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = path
                explicit = path
    state["profile_trace"] = explicit
    state.setdefault("steady_state_trace", "")


def _disk_state() -> dict[str, Any]:
    try:
        path = resolve_state_file()
    except RuntimeError:
        return {}
    if not path.is_file() or not state_file_safe_to_read(path):
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _keep_profile_bookkeeping(state: dict[str, Any]) -> None:
    """Keep an already-recorded steady trace when this is the same cluster."""
    disk = _disk_state()
    if not disk:
        return
    same = disk.get("service_url") == state.get("service_url") and disk.get("prefill_pod_ips") == state.get(
        "prefill_pod_ips"
    )
    if not same:
        return
    if not state.get("steady_state_trace") and disk.get("steady_state_trace"):
        state["steady_state_trace"] = disk["steady_state_trace"]
    if not state.get("profile_trace") and disk.get("profile_trace"):
        state["profile_trace"] = disk["profile_trace"]


def stamp_pd_cluster(state: dict[str, Any], get: Lookup) -> None:
    """Fill router/prefill/decode pods and profile paths on an atom or sglangrouter state."""
    if str(state.get("backend") or "").strip().lower() not in PD_ROUTER_BACKENDS:
        return
    router, prefill, decode = role_ips(get)
    port = ssh_port_from_handoff(get)
    key = get("HYPERLOOM_MN_EXT_SSH_KEY").strip()
    state["pd_mode"] = "disaggregated"
    state["external"] = True
    state["ssh_port"] = port
    if key:
        state["ssh_key_path"] = key
    state["router_pod_ips"] = router
    state["prefill_pod_ips"] = prefill
    state["decode_pod_ips"] = decode
    state["worker_pod_ips"] = []
    state["worker_pods"] = []
    spec = spec_from_env(state, get)
    pods = pods_for(router, prefill, decode, port, spec)
    state["pods"] = pods
    state["router_pods"] = [pod for pod in pods if pod["role"] == "router"]
    state["prefill_pods"] = [pod for pod in pods if pod["role"] == "prefill"]
    state["decode_pods"] = [pod for pod in pods if pod["role"] == "decode"]
    _pin_profile_dir(state)
    _keep_profile_bookkeeping(state)


def record_profile_paths(*, profile_trace: str = "", steady_state_trace: str = "") -> None:
    """Write TraceLens inputs back into the multi-node state file."""
    try:
        path = resolve_state_file()
    except RuntimeError:
        return
    if not path.is_file() or not state_file_safe_to_read(path):
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict) or str(data.get("backend") or "") not in PD_ROUTER_BACKENDS:
        return
    if profile_trace:
        data["profile_trace"] = profile_trace
    if steady_state_trace:
        data["steady_state_trace"] = steady_state_trace
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def launch_script(role: str, command: str, extra_env: dict[str, str]) -> str:
    """Bash script that execs ``command`` with the profile and tuning environment."""
    lines = ["#!/bin/bash", "set -euo pipefail", f"export ATOM_SERVER_PORT={ENGINE_PORT}"]
    if role != "router":
        lines.append("export SGLANG_HOST_IP=\"${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}\"")
    for key, value in extra_env.items():
        if role == "router" and key == "SGLANG_TORCH_PROFILER_DIR":
            continue
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(f"exec {command}")
    return "\n".join(lines) + "\n"
