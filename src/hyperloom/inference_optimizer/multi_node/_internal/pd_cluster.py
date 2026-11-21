# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomesh and SGLang-router multi-node cluster records.

The state file is the contract other tools read.
The platform runs the router; Hyperloom launches ``sglang.launch_server`` on
every prefill and decode pod. Every pod uses the one SSH port from
``HYPERLOOM_MN_EXT_SSH_PORT`` (no per-role stride).
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shlex
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hyperloom_common.coerce import optional_positive_int

from ..state_paths import resolve_state_file, state_file_safe_to_read
from .topology_file import PD_ROUTER_BACKENDS, Lookup, normalize_mn_backend

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
# Engine flags that newer SGLang splits into ``<flag>-decode`` and ``<flag>-prefill``.
SPLIT_ROLE_FLAGS = ("--cuda-graph-bs", "--cuda-graph-max-bs")
# Device files whose open handles mean a process still holds GPU memory.
GPU_DEVICE_PATHS = ("/dev/kfd",)
# Seconds an engine launch waits for the previous engine to release the GPU.
GPU_RELEASE_WAIT_S = 120


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
    # Option strings the pod's ``sglang.launch_server --help`` lists; empty when not probed.
    engine_flags: frozenset[str] = field(default_factory=frozenset)


def engine_port(role: str, spec: LaunchSpec) -> int:
    """Listen port of a prefill or decode engine: the port of its role URL, else ENGINE_PORT."""
    url = spec.prefill_url if role == PREFILL else spec.decode_url
    try:
        port = urlsplit((url or "").strip()).port
    except ValueError:
        port = None
    return port or ENGINE_PORT


def _flag_end(words: list[str], i: int) -> int:
    """Index just past the flag at ``i`` and its space-separated values."""
    j = i + 1
    if "=" not in words[i]:
        while j < len(words) and not words[j].startswith("--"):
            j += 1
    return j


def drop_unsupported_baseline_flags(
    words: list[str], baseline_words: list[str], engine_flags: frozenset[str]
) -> tuple[list[str], list[str]]:
    """Remove flags that the baseline supplied and the engine does not list.

    Returns the kept words and the dropped flag names. ``SPLIT_ROLE_FLAGS`` and
    flags absent from the baseline are kept; nothing is dropped without
    ``engine_flags``.
    """
    if not engine_flags:
        return words, []
    baseline = {word.split("=", 1)[0] for word in baseline_words if word.startswith("--")}
    out: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(words):
        name = words[i].split("=", 1)[0]
        j = _flag_end(words, i) if name.startswith("--") else i + 1
        if name in baseline and name not in engine_flags and name not in SPLIT_ROLE_FLAGS:
            if name not in dropped:
                dropped.append(name)
        else:
            out.extend(words[i:j])
        i = j
    return out, dropped


def uses_split_role_flags(words: list[str]) -> bool:
    """True when ``words`` carries a flag from ``SPLIT_ROLE_FLAGS``."""
    return any(word.split("=", 1)[0] in SPLIT_ROLE_FLAGS for word in words)


def adapt_split_role_flags(role: str, words: list[str], engine_flags: frozenset[str]) -> list[str]:
    """Map ``SPLIT_ROLE_FLAGS`` onto an engine that only has the per-role variants.

    Decode takes ``<flag>-decode`` with the same values. Prefill drops the flag:
    its per-role variant counts tokens, not decode batch sizes.
    """
    split = {flag for flag in SPLIT_ROLE_FLAGS if flag not in engine_flags and f"{flag}-decode" in engine_flags}
    if not split:
        return words
    out: list[str] = []
    i = 0
    while i < len(words):
        name, sep, value = words[i].partition("=")
        if name not in split:
            out.append(words[i])
            i += 1
            continue
        j = _flag_end(words, i)
        if role == "decode":
            out.append(f"{name}-decode{sep}{value}")
            out.extend(words[i + 1 : j])
        i = j
    return out


def engine_command(role: str, spec: LaunchSpec) -> str:
    """sglang prefill or decode command for one pod."""
    tp = spec.prefill_tp if role == PREFILL else spec.decode_tp
    ep = spec.prefill_ep if role == PREFILL else spec.decode_ep
    bootstrap = PREFILL_BOOTSTRAP_PORT if role == PREFILL else DECODE_BOOTSTRAP_PORT
    # Shared args go last: the engine keeps the last value of a repeated flag, so a candidate overrides the baseline.
    extra = shlex.split((spec.prefill_extra if role == PREFILL else spec.decode_extra) or "")
    extra += shlex.split(spec.server_args or "")
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
    argv.extend(adapt_split_role_flags(role, extra, spec.engine_flags))
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
    if normalize_mn_backend(get("INFERENCE_OPTIMIZER_MN_BACKEND")) not in PD_ROUTER_BACKENDS:
        return False
    if not get("HYPERLOOM_MN_EXT_SSH_KEY").strip():
        return False
    if _ssh_port(get) is None:
        return False
    return bool(_csv(get, "HYPERLOOM_MN_EXT_PREFILL_IPS") and _csv(get, "HYPERLOOM_MN_EXT_DECODE_IPS"))


def _endpoint(explicit: str, ips: list[Any]) -> str:
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
        return str(getattr(args, name, "") or "").strip()

    def env(name: str) -> str:
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


def pd_pod_trace_dir(root: str, role: str, pod_index: int) -> str:
    """Torch trace dir of ``role``'s ``pod_index``-th engine under a profiling round's trace root."""
    return os.path.join(root, role, str(pod_index))


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


def _pin_profile_dir(state: dict[str, Any]) -> None:
    """Point profile_trace at a stable directory under the session."""
    explicit = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    session = os.environ.get("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "").strip()
    path = explicit or (os.path.join(session, "runs", "profile", "torch_trace") if session else "")
    if path:
        try:
            os.makedirs(path, mode=0o755, exist_ok=True)
        except OSError:
            path = explicit
    if path and not explicit:
        os.environ["HYPERLOOM_MN_PROFILE_TRACE_DIR"] = path
    state["profile_trace"] = path
    state.setdefault("steady_state_trace", "")


def _read_state_file(path: Path) -> dict[str, Any]:
    """Load the state file when it is present, safe to read and holds a JSON object."""
    if not path.is_file() or not state_file_safe_to_read(path):
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def previous_cluster_state(state: dict[str, Any]) -> dict[str, Any]:
    """The state file a prior launch wrote for the same cluster, or an empty dict."""
    try:
        disk = _read_state_file(resolve_state_file())
    except RuntimeError:
        return {}
    return disk if disk.get("service_url") == state.get("service_url") else {}


def stamp_pd_cluster(state: dict[str, Any], get: Lookup) -> None:
    """Fill router/prefill/decode pods and profile paths on an atomesh or sglangrouter state.

    TraceLens inputs a prior launch recorded for the same cluster are kept; ``profile_trace`` stays pinned.
    """
    state["pd_mode"] = "disaggregated"
    state["ssh_port"] = _ssh_port(get)
    state["worker_pod_ips"] = []
    state["worker_pods"] = []
    set_pods(state, launch_spec(state, get=get))
    _pin_profile_dir(state)
    previous = previous_cluster_state(state)
    for name in ("steady_state_trace", "decode_trace", "prefill_trace"):
        if not state.get(name) and previous.get(name):
            state[name] = previous[name]


def record_profile_paths(
    *,
    profile_trace: str = "",
    steady_state_trace: str = "",
    role_traces: dict[str, str] | None = None,
) -> None:
    """Write TraceLens inputs (shared, steady-state and per-role ``<role>_trace``) back into the multi-node state file."""
    try:
        path = resolve_state_file()
    except RuntimeError:
        return
    data = _read_state_file(path)
    if str(data.get("backend") or "") not in PD_ROUTER_BACKENDS:
        return
    if profile_trace:
        data["profile_trace"] = profile_trace
    if steady_state_trace:
        data["steady_state_trace"] = steady_state_trace
    for role, trace in (role_traces or {}).items():
        if trace:
            data[f"{role}_trace"] = trace
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    with contextlib.suppress(OSError):
        path.chmod(0o600)


def _engine_log_lines(log_dir: str, log_ip: str) -> list[str]:
    """Bash that appends the engine's output to ``<log_dir>/mn_infera_server_<ip>_r0.log`` beside the container log."""
    log_file = shlex.quote(f"{log_dir}/mn_infera_server_{log_ip}_r0.log")
    return [
        f"if mkdir -p {shlex.quote(log_dir)} 2>/dev/null; then",
        f"  exec > >(tee -ai {log_file}) 2>&1",
        "fi",
        'echo "[hyperloom] engine launch $(date -u +%Y-%m-%dT%H:%M:%SZ)"',
    ]


def _close_inherited_handle_lines() -> list[str]:
    """Bash that closes GPU device handles and sockets this process inherited from the image it replaced."""
    patterns = "|".join(shlex.quote(path) for path in GPU_DEVICE_PATHS)
    return [
        "for hl_fd in /proc/$$/fd/*; do",
        '  [ "${hl_fd##*/}" -gt 2 ] || continue',
        f'  case "$(readlink "$hl_fd" 2>/dev/null || true)" in {patterns}|socket:*)'
        ' eval "exec ${hl_fd##*/}>&-" ;; esac',
        "done",
    ]


def _gpu_release_wait_lines() -> list[str]:
    """Bash that waits until no other process holds a GPU device, up to ``GPU_RELEASE_WAIT_S``."""
    listing = "|".join("*" + shlex.quote(" -> " + path) + "*" for path in GPU_DEVICE_PATHS)
    return [
        "hl_gpu_holders() {",
        "  local p",
        "  for p in /proc/[0-9]*; do",
        '    [ "${p#/proc/}" = "$$" ] && continue',
        f'    case "$(ls -l "$p"/fd 2>/dev/null || true)" in {listing}) echo "${{p#/proc/}}" ;; esac',
        "  done",
        "}",
        f"hl_wait_end=$((SECONDS + {GPU_RELEASE_WAIT_S}))",
        'while hl_held="$(hl_gpu_holders | tr "\\n" " ")" && [ -n "${hl_held// /}" ]; do',
        '  if [ "$SECONDS" -ge "$hl_wait_end" ]; then',
        f'    echo "[hyperloom] WARN GPU still held by pid(s) ${{hl_held% }} after {GPU_RELEASE_WAIT_S}s;'
        ' launching anyway" >&2',
        "    break",
        "  fi",
        "  sleep 1",
        "done",
    ]


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
    lines.extend(_close_inherited_handle_lines())
    log_dir = str(extra_env.get("HYPERLOOM_MN_SERVER_LOG_DIR") or "")
    if log_dir.startswith("/") and log_ip:
        lines.extend(_engine_log_lines(log_dir, log_ip))
    lines.extend(_gpu_release_wait_lines())
    lines.append(f"exec {command}")
    return "\n".join(lines) + "\n"
