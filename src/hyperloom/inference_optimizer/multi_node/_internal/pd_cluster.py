# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomesh and SGLang-router multi-node cluster records.

The state file is the contract other tools read.
The platform runs the router; Hyperloom launches ``sglang.launch_server`` on
every prefill and decode pod. Every pod uses the one SSH port from
``HYPERLOOM_MN_EXT_SSH_PORT`` (no per-role stride).
"""

from __future__ import annotations

import functools
import os
import shlex
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from hyperloom_common.coerce import optional_positive_int
from inference_suite.multinode import engine_prologue, replace_flags

from .pd_handoff import PD_BACKENDS
from .topology_file import Lookup, normalize_mn_backend

ENGINE_PORT = 8000
PREFILL = "prefill"
DECODE = "decode"
PREFILL_BOOTSTRAP_PORT = 8998
#: Engine env defaults a launch keeps unless the forwarded env or the pod already sets them.
_ENGINE_ENV_DEFAULTS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "100000",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "100000",
}
DECODE_BOOTSTRAP_PORT = 9098
# ``managed_by`` value of a pod whose process the platform owns; it is recorded, never restarted.
PLATFORM_MANAGED = "platform"
# Substring of the engine command line that identifies the server process on a pod.
ENGINE_NEEDLE = "sglang.launch_server"


@dataclass(frozen=True)
class LaunchSpec:
    """Inputs for one round of engine commands."""

    model: str
    framework: str
    prefill_tp: int
    decode_tp: int
    # Shared EP size; each role's own size is resolved into ``prefill_ep`` / ``decode_ep``.
    ep: int
    prefill_ep: int
    decode_ep: int
    kv: str
    ib: str
    server_args: str
    prefill_extra: str
    decode_extra: str
    prefill_url: str
    decode_url: str


def engine_port(role: str, spec: LaunchSpec) -> int:
    """Listen port of a prefill or decode engine: the port of its role URL, else ENGINE_PORT."""
    url = spec.prefill_url if role == PREFILL else spec.decode_url
    try:
        port = urlsplit((url or "").strip()).port
    except ValueError:
        port = None
    return port or ENGINE_PORT


def engine_command(role: str, spec: LaunchSpec) -> str:
    """sglang prefill or decode command for one pod."""
    tp = spec.prefill_tp if role == PREFILL else spec.decode_tp
    ep = spec.prefill_ep if role == PREFILL else spec.decode_ep
    bootstrap = PREFILL_BOOTSTRAP_PORT if role == PREFILL else DECODE_BOOTSTRAP_PORT
    # Every flag the shared args set replaces the role's value of it, so each flag appears once.
    role_args = shlex.split((spec.prefill_extra if role == PREFILL else spec.decode_extra) or "")
    extra = list(replace_flags(role_args, shlex.split(spec.server_args or "")))
    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(engine_port(role, spec)),
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
    if ep > 1:
        argv.extend(["--ep-size", str(ep)])
    argv.extend(extra)
    return shlex.join(argv)


def _ssh_port(get: Lookup) -> int | None:
    """``HYPERLOOM_MN_EXT_SSH_PORT``, the port on every pod; None when unset or invalid (no default, no stride)."""
    port = optional_positive_int(get("HYPERLOOM_MN_EXT_SSH_PORT"))
    return port if port and port <= 65535 else None


def _csv(get: Lookup, name: str) -> list[str]:
    """Comma-separated ``name`` as a list of trimmed, non-empty parts."""
    return [part.strip() for part in get(name).split(",") if part.strip()]


def pd_cluster_has_control(get: Lookup) -> bool:
    """True when this hand-off can SSH-restart an atomesh or sglangrouter cluster."""
    if normalize_mn_backend(get("INFERENCE_OPTIMIZER_MN_BACKEND")) not in PD_BACKENDS:
        return False
    if not get("HYPERLOOM_MN_EXT_SSH_KEY").strip():
        return False
    if _ssh_port(get) is None:
        return False
    return bool(_csv(get, "HYPERLOOM_MN_EXT_PREFILL_IPS") and _csv(get, "HYPERLOOM_MN_EXT_DECODE_IPS"))


def _endpoint(explicit: str, ips: list[Any]) -> str:
    """``explicit`` when set, else the engine URL of the first IP, else empty."""
    text = (explicit or "").strip()
    if text:
        return text
    return f"http://{ips[0]}:{ENGINE_PORT}" if ips else ""


def launch_spec(state: dict[str, Any], args: Any = None, get: Lookup | None = None) -> LaunchSpec:
    """Build a launch spec from a restart-server ``args``, falling back to the process environment.

    Without ``args`` the shared and per-role server args come from the environment; role URLs come from ``get``.
    """
    lookup = get or (lambda name: os.environ.get(name, ""))

    def arg(name: str) -> str:
        """Stripped ``args`` attribute ``name``, empty when absent."""
        return str(getattr(args, name, "") or "").strip()

    def env(name: str) -> str:
        """Stripped environment variable ``name``, empty when unset."""
        return os.environ.get(name, "").strip()

    gpus = optional_positive_int(str(state.get("gpus_per_node") or ""), 8)
    tp = optional_positive_int(arg("tp"), optional_positive_int(env("TP"), gpus))
    ep = optional_positive_int(arg("ep"), optional_positive_int(env("EP"), 1))
    if args is None:
        server, prefill, decode = (
            env("INFERENCE_OPTIMIZER_SERVER_ARGS"),
            env("PD_PREFILL_EXTRA_ARGS"),
            env("PD_DECODE_EXTRA_ARGS"),
        )
    else:
        server, prefill, decode = arg("extra_args"), arg("pd_prefill_extra_args"), arg("pd_decode_extra_args")
    return LaunchSpec(
        model=arg("model") or env("MODEL_PATH"),
        framework=(arg("framework") or env("FRAMEWORK") or "sglang").lower(),
        prefill_tp=optional_positive_int(arg("pd_prefill_tp"), optional_positive_int(env("PD_PREFILL_TP"), tp)),
        decode_tp=optional_positive_int(arg("pd_decode_tp"), optional_positive_int(env("PD_DECODE_TP"), tp)),
        ep=ep,
        prefill_ep=optional_positive_int(arg("pd_prefill_ep"), optional_positive_int(env("PD_PREFILL_EP"), ep)),
        decode_ep=optional_positive_int(arg("pd_decode_ep"), optional_positive_int(env("PD_DECODE_EP"), ep)),
        kv=arg("pd_transfer_backend") or env("PD_TRANSFER_BACKEND"),
        ib=arg("pd_ib_device") or env("PD_IB_DEVICE"),
        server_args=server,
        prefill_extra=prefill,
        decode_extra=decode,
        prefill_url=_endpoint(lookup("HYPERLOOM_MN_EXT_PREFILL_URL"), list(state.get("prefill_pod_ips") or [])),
        decode_url=_endpoint(lookup("HYPERLOOM_MN_EXT_DECODE_URL"), list(state.get("decode_pod_ips") or [])),
    )


def _pod(role: str, index: int, ip: str, port: int | None, spec: LaunchSpec) -> dict[str, Any]:
    """Pod record of the ``index``-th ``role`` engine with its launch command."""
    return {
        "role": role,
        "podIP": ip,
        "podId": f"external-{role}-{index}",
        "sshPort": port,
        "launch_command": engine_command(role, spec),
        "enginePort": engine_port(role, spec),
    }


def role_engine_pods(state: dict[str, Any], role: str) -> list[dict[str, Any]]:
    """``role``'s pod records that carry an IP, in state order; a pod's position is its trace-dir index."""
    return [pod for pod in (state.get(f"{role}_pods") or []) if isinstance(pod, dict) and pod.get("podIP")]


@functools.lru_cache(maxsize=None)
def _resolve_host(host: str) -> str:
    """IP ``host`` resolves to, memoized per host for the process.

    A failed lookup raises ``OSError`` and is not memoized, so a service name the platform has not published yet
    resolves on a later call.
    """
    return socket.gethostbyname(host)


def _service_ip(service_url: str) -> str:
    """IP the service host resolves to, or the host itself when it does not resolve."""
    host = urlsplit(service_url).hostname or ""
    if not host:
        return ""
    try:
        return _resolve_host(host)
    except OSError:
        return host


def set_pods(state: dict[str, Any], spec: LaunchSpec) -> None:
    """Record one pod per host on ``state``: the platform router from ``service_url``, then prefill, then decode.

    The router is marked ``managed_by: platform``; restarts leave it alone.
    """
    port = state.get("ssh_port")
    pods: list[dict[str, Any]] = []
    service_ip = _service_ip(str(state.get("service_url") or ""))
    if service_ip:
        pods.append(
            {"role": "router", "podIP": service_ip, "podId": "external-router-0", "managed_by": PLATFORM_MANAGED}
        )
    for role in (PREFILL, DECODE):
        pods.extend(
            _pod(role, index, str(ip), port, spec) for index, ip in enumerate(state.get(f"{role}_pod_ips") or [])
        )
    for role in ("router", PREFILL, DECODE):
        state[f"{role}_pods"] = [pod for pod in pods if pod["role"] == role]
    state["router_pod_ips"] = [str(pod["podIP"]) for pod in state["router_pods"]]


def stamp_pd_cluster(state: dict[str, Any], get: Lookup) -> None:
    """Fill router/prefill/decode pods on an atomesh or sglangrouter state from its ``prefill/decode_pod_ips``."""
    state["pd_mode"] = "disaggregated"
    state["ssh_port"] = _ssh_port(get)
    state["worker_pod_ips"] = []
    state["worker_pods"] = []
    set_pods(state, launch_spec(state, get=get))


def _engine_log(log_dir: str, log_ip: str) -> str:
    """The engine log ``<log_dir>/mn_infera_server_<ip>_r0.log``; empty unless ``log_dir`` is absolute and ``log_ip``
    names the pod."""
    return f"{log_dir}/mn_infera_server_{log_ip}_r0.log" if log_dir.startswith("/") and log_ip else ""


def launch_script(command: str, extra_env: dict[str, str], *, log_ip: str = "") -> str:
    """Bash script that execs an engine ``command`` with the profile and tuning environment.

    The disaggregation timeout defaults apply to any not in ``extra_env`` or the pod environment. Output is teed to
    the shared server-log dir when one is set and ``log_ip`` names the pod. The command starts only after the
    previous engine released the GPU.
    """
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"export ATOM_SERVER_PORT={ENGINE_PORT}",
        "export SGLANG_HOST_IP=\"${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}\"",
    ]
    lines.extend(f"export {key}={shlex.quote(str(value))}" for key, value in extra_env.items())
    lines.extend(
        f'export {key}="${{{key}:-{value}}}"' for key, value in _ENGINE_ENV_DEFAULTS.items() if key not in extra_env
    )
    log_dir = str(extra_env.get("HYPERLOOM_MN_SERVER_LOG_DIR") or "")
    lines.extend(engine_prologue(log_dir, _engine_log(log_dir, log_ip)))
    lines.append(f"exec {command}")
    return "\n".join(lines) + "\n"
