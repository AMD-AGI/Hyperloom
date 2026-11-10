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
import socket
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

from ..state_paths import resolve_state_file, state_file_safe_to_read
from .topology_file import normalize_mn_backend

PD_ROUTER_BACKENDS = ("atomesh", "sglangrouter")
ENGINE_PORT = 8000
PREFILL_BOOTSTRAP_PORT = 8998
DECODE_BOOTSTRAP_PORT = 9098
# ``managed_by`` value of a pod whose process the platform owns; it is recorded, never restarted.
PLATFORM_MANAGED = "platform"
# Engine flags that newer SGLang splits into ``<flag>-decode`` and ``<flag>-prefill``.
SPLIT_ROLE_FLAGS = ("--cuda-graph-bs", "--cuda-graph-max-bs")
# Device files whose open handles mean a process still holds GPU memory.
GPU_DEVICE_PATHS = ("/dev/kfd",)
# Seconds an engine launch waits for the previous engine to release the GPU.
GPU_RELEASE_WAIT_S = 120

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
    # Option strings the pod's ``sglang.launch_server --help`` lists; empty when not probed.
    engine_flags: frozenset[str] = field(default_factory=frozenset)


def _words(raw: str) -> list[str]:
    """Split a server-arg string the way the shell would."""
    text = (raw or "").strip()
    return shlex.split(text) if text else []


def _join(argv: list[str]) -> str:
    """Shell-quote an argv into one command line."""
    return " ".join(shlex.quote(part) for part in argv)


def engine_port(role: str, spec: LaunchSpec) -> int:
    """Listen port of a prefill or decode engine: the port of its role URL, else ENGINE_PORT."""
    url = spec.prefill_url if role == "prefill" else spec.decode_url
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
    split = {
        flag
        for flag in SPLIT_ROLE_FLAGS
        if engine_flags and flag not in engine_flags and f"{flag}-decode" in engine_flags
    }
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
    tp = spec.prefill_tp if role == "prefill" else spec.decode_tp
    bootstrap = PREFILL_BOOTSTRAP_PORT if role == "prefill" else DECODE_BOOTSTRAP_PORT
    # Shared args go last: the engine keeps the last value of a repeated flag, so a candidate overrides the baseline.
    extra = _words(spec.prefill_extra if role == "prefill" else spec.decode_extra) + _words(spec.server_args)
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
    if spec.ep > 1:
        argv.extend(["--ep-size", str(spec.ep)])
    argv.extend(adapt_split_role_flags(role, extra, spec.engine_flags))
    return _join(argv)


def router_command(spec: LaunchSpec) -> str:
    """Router command. Atomesh for ``atomesh``, sglang_router for ``sglangrouter``."""
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

    Unlike Infera, atomesh and sglangrouter do not invent a port and do not add a
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
    """True when this hand-off can SSH-restart an atomesh or sglangrouter cluster."""
    if normalize_mn_backend(get("INFERENCE_OPTIMIZER_MN_BACKEND")) not in PD_ROUTER_BACKENDS:
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
        backend=str(state.get("backend") or "atomesh"),
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
        backend=str(state.get("backend") or "atomesh"),
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
    pod: dict[str, Any] = {
        "role": role,
        "podIP": ip,
        "podId": f"external-{role}-{index}",
        "sshPort": port,
        "launch_command": command_for(role, spec),
    }
    if role != "router":
        pod["enginePort"] = engine_port(role, spec)
    return pod


def _service_ip(service_url: str) -> str:
    """IP the service host resolves to, or the host itself when it does not resolve."""
    host = urlsplit(service_url).hostname or ""
    if not host:
        return ""
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def pods_for(
    router: list[str],
    prefill: list[str],
    decode: list[str],
    port: int | None,
    spec: LaunchSpec,
    *,
    service_url: str = "",
) -> list[dict[str, Any]]:
    """One pod record per host, router then prefill then decode.

    Without hand-off router IPs, the router is recorded from ``service_url`` and
    marked ``managed_by: platform``; restart and stop leave it alone.
    """
    pods: list[dict[str, Any]] = []
    for index, ip in enumerate(router):
        pods.append(_pod("router", index, ip, port, spec))
    service_ip = "" if router else _service_ip(service_url)
    if service_ip:
        pods.append({**_pod("router", 0, service_ip, port, spec), "managed_by": PLATFORM_MANAGED})
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


def previous_cluster_state(state: dict[str, Any]) -> dict[str, Any]:
    """The state file a prior launch wrote for the same cluster, or an empty dict."""
    disk = _disk_state()
    if not disk or disk.get("service_url") != state.get("service_url"):
        return {}
    return disk


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
    """Fill router/prefill/decode pods and profile paths on an atomesh or sglangrouter state."""
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
    pods = pods_for(router, prefill, decode, port, spec, service_url=str(state.get("service_url") or ""))
    state["pods"] = pods
    state["router_pods"] = [pod for pod in pods if pod["role"] == "router"]
    state["prefill_pods"] = [pod for pod in pods if pod["role"] == "prefill"]
    state["decode_pods"] = [pod for pod in pods if pod["role"] == "decode"]
    _pin_profile_dir(state)
    _keep_profile_bookkeeping(state)


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
    for role, trace in (role_traces or {}).items():
        if trace:
            data[f"{role}_trace"] = trace
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _engine_log_lines(log_dir: str, log_ip: str) -> list[str]:
    """Bash that appends the engine's output to ``<log_dir>/mn_infera_server_<ip>_r0.log`` beside the container log."""
    log_file = shlex.quote(f"{log_dir}/mn_infera_server_{log_ip}_r0.log")
    return [
        f"if mkdir -p {shlex.quote(log_dir)} 2>/dev/null; then",
        f"  exec > >(tee -ai {log_file}) 2>&1",
        "fi",
        'echo "[hyperloom] engine launch $(date -u +%Y-%m-%dT%H:%M:%SZ)"',
    ]


def _gpu_patterns() -> str:
    """``case`` pattern matching any of ``GPU_DEVICE_PATHS``."""
    return "|".join(shlex.quote(path) for path in GPU_DEVICE_PATHS)


def _close_inherited_handle_lines() -> list[str]:
    """Bash that closes GPU device handles and sockets this process inherited from the image it replaced."""
    return [
        "for hl_fd in /proc/$$/fd/*; do",
        '  [ "${hl_fd##*/}" -gt 2 ] || continue',
        f'  case "$(readlink "$hl_fd" 2>/dev/null || true)" in {_gpu_patterns()}|socket:*)'
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
        f"hl_wait_end=$((SECONDS + {int(GPU_RELEASE_WAIT_S)}))",
        'while hl_held="$(hl_gpu_holders | tr "\\n" " ")" && [ -n "${hl_held// /}" ]; do',
        '  if [ "$SECONDS" -ge "$hl_wait_end" ]; then',
        f'    echo "[hyperloom] WARN GPU still held by pid(s) ${{hl_held% }} after {int(GPU_RELEASE_WAIT_S)}s;'
        ' launching anyway" >&2',
        "    break",
        "  fi",
        "  sleep 1",
        "done",
    ]


def launch_script(role: str, command: str, extra_env: dict[str, str], *, log_ip: str = "") -> str:
    """Bash script that execs ``command`` with the profile and tuning environment.

    Engine roles tee their output to the shared server-log dir when one is set and
    ``log_ip`` names the pod, and start only after the previous engine released the GPU.
    """
    lines = ["#!/bin/bash", "set -euo pipefail", f"export ATOM_SERVER_PORT={ENGINE_PORT}"]
    if role != "router":
        lines.append("export SGLANG_HOST_IP=\"${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}\"")
    for key, value in extra_env.items():
        if role == "router" and key == "SGLANG_TORCH_PROFILER_DIR":
            continue
        lines.append(f"export {key}={shlex.quote(str(value))}")
    if role != "router":
        lines.extend(_close_inherited_handle_lines())
        log_dir = str(extra_env.get("HYPERLOOM_MN_SERVER_LOG_DIR") or "")
        if log_dir.startswith("/") and log_ip:
            lines.extend(_engine_log_lines(log_dir, log_ip))
        lines.extend(_gpu_release_wait_lines())
    lines.append(f"exec {command}")
    return "\n".join(lines) + "\n"
