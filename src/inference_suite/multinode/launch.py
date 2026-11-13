# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start, restart and stop a frozen multi-node launch on the hosts of a node list over SSH.

Engines start first; the router starts once every worker answers ``/health`` from here, and the launch is up once
the router and the node list's service URL answer too. A launch behind the platform's Infera frontend starts no
router and is up once the service URL answers. An engine process that dies on the way fails the launch with
its log tail. Each engine waits for the GPUs to be released by the previous one and closes the device handles it
inherited. SGLang's ``--help`` is probed once per launch: flags from a non-operator source that the engine does not
list are dropped, and ``--cuda-graph-bs`` / ``--cuda-graph-max-bs`` move to their decode spelling on an engine that
only has the per-role ones. What was dropped is named in the role's ``stripped``.

Hosts of an ``in_place`` node list run the platform's container: the launch takes over the process the container
launcher waits on and runs in the container's environment. Hosts of a ``direct`` one run what the launch spawns, in
the SSH session's environment.
"""

from __future__ import annotations

import http.client
import json
import re
import shlex
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from inference_suite.frameworks import Framework
from inference_suite.multinode.commands import (
    ROUTER,
    ROUTER_PORT,
    HostCommand,
    cluster_commands,
    engine_urls,
    process_needle,
    router_command,
    runs_router,
)
from inference_suite.multinode.nodes import LaunchMode, NodeList
from inference_suite.multinode.record import PdLaunch, PdSourceKind, Role, RoleLaunch
from inference_suite.multinode.render import flag_groups, group_flag, render_role
from inference_suite.multinode.ssh import SshClient

#: Flags newer SGLang splits into ``<flag>-decode`` and ``<flag>-prefill``.
SPLIT_ROLE_FLAGS = ("cuda-graph-bs", "cuda-graph-max-bs")
#: Device files whose open handles mean a process still holds GPU memory.
GPU_DEVICE_PATHS = ("/dev/kfd",)
GPU_RELEASE_WAIT_S = 120
CONTAINER_ENVIRON = "/proc/1/environ"
_DRIVER = Path(__file__).with_name("remote_driver.py")
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEALTH_TIMEOUT_S = 5
_HTTP_OK = 200
_LOG_TAIL_LINES = 40
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LaunchError(RuntimeError):
    """A host refused the launch, or the cluster did not come up."""


def http_ok(url: str) -> bool:
    """Whether ``url`` answers 200, bypassing any proxy."""
    try:
        with _OPENER.open(url, timeout=_HEALTH_TIMEOUT_S) as response:  # nosec B310
            return response.status == _HTTP_OK
    except (OSError, http.client.HTTPException):
        return False


@dataclass(frozen=True)
class Cluster:
    """The hosts of a launch and how they are reached: SSH, HTTP probes from here, and the clock between polls."""

    nodes: NodeList
    client: SshClient
    probe: Callable[[str], bool] = http_ok
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class LaunchOptions:
    """What every host's server additionally gets (``role_env`` per engine role, over ``extra_env``), where engines
    tee their logs, and how long to wait."""

    extra_env: Mapping[str, str] = field(default_factory=dict)
    role_env: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    log_dir: str = ""
    ssh_timeout_s: int = 180
    ready_timeout_s: int = 1800
    poll_interval_s: float = 10.0


def _pidfile(role: str) -> str:
    return f"/tmp/hyperloom-mn-{role}.pid"


def _launch_path(role: str) -> str:
    return f"/tmp/hyperloom-mn-{role}.sh"


def _container_env_lines() -> list[str]:
    """Bash that exports the container's interpreter search paths."""
    return [
        "while IFS= read -r -d '' hl_kv; do case \"$hl_kv\" in PATH=*|PYTHONPATH=*|VIRTUAL_ENV=*|LD_LIBRARY_PATH=*)"
        f' export "$hl_kv";; esac; done < {CONTAINER_ENVIRON}'
    ]


def _ib_device_lines() -> list[str]:
    """Bash that sets ``HL_IB_DEVICE`` from ``IBDEVICES``, else from ``NCCL_IB_HCA`` without its port suffixes, read
    from the server's environment or the container's; empty when neither names a device."""
    return [
        f"hl_env1() {{ tr '\\0' '\\n' < {CONTAINER_ENVIRON} 2>/dev/null | sed -n \"s/^$1=//p\" | head -n 1; }}",
        'HL_IB_DEVICE="${IBDEVICES:-$(hl_env1 IBDEVICES)}"',
        'hl_hca="${NCCL_IB_HCA:-$(hl_env1 NCCL_IB_HCA)}"',
        'case "$hl_hca" in ^*) hl_hca="" ;; =*) hl_hca="${hl_hca#=}" ;; esac',
        '[ -n "$HL_IB_DEVICE" ] || HL_IB_DEVICE="$(printf \'%s\' "$hl_hca" | sed -E \'s/:[0-9]+//g\')"',
    ]


def _close_inherited_lines() -> list[str]:
    """Bash that closes the GPU device handles and sockets inherited from the image this one replaced."""
    devices = "|".join(shlex.quote(path) for path in GPU_DEVICE_PATHS)
    return [
        "for hl_fd in /proc/$$/fd/*; do",
        '  [ "${hl_fd##*/}" -gt 2 ] || continue',
        f'  case "$(readlink "$hl_fd" 2>/dev/null || true)" in {devices}|socket:*) eval "exec ${{hl_fd##*/}}>&-" ;; esac',
        "done",
    ]


def _gpu_release_lines() -> list[str]:
    """Bash that waits until no other process holds a GPU device, up to ``GPU_RELEASE_WAIT_S``."""
    listing = "|".join("*" + shlex.quote(" -> " + path) + "*" for path in GPU_DEVICE_PATHS)
    return [
        "hl_gpu_holders() { local p; for p in /proc/[0-9]*; do",
        '  [ "${p#/proc/}" = "$$" ] && continue',
        f'  case "$(ls -l "$p"/fd 2>/dev/null || true)" in {listing}) echo "${{p#/proc/}}" ;; esac',
        "done; }",
        f"hl_wait_end=$((SECONDS + {GPU_RELEASE_WAIT_S}))",
        'while hl_held="$(hl_gpu_holders | tr "\\n" " ")" && [ -n "${hl_held// /}" ]; do',
        f'  if [ "$SECONDS" -ge "$hl_wait_end" ]; then echo "[hyperloom] GPU still held by ${{hl_held% }} after'
        f' {GPU_RELEASE_WAIT_S}s; launching anyway" >&2; break; fi',
        "  sleep 1",
        "done",
    ]


def _log_lines(log_dir: str, command: HostCommand) -> list[str]:
    """Bash that also appends the engine's output to ``<log_dir>/pd_<role>_<host>.log``."""
    log = shlex.quote(f"{log_dir}/pd_{command.role}_{command.host}.log")
    return [
        f"if mkdir -p {shlex.quote(log_dir)} 2>/dev/null; then exec > >(tee -ai {log}) 2>&1; fi",
        'echo "[hyperloom] engine launch $(date -u +%Y-%m-%dT%H:%M:%SZ)"',
    ]


def launch_script(command: HostCommand, env: Mapping[str, str], framework: Framework, log_dir: str) -> str:
    """The script one host execs: ``env`` exported, and for an engine the IB device found, inherited handles
    closed, its log teed under ``log_dir`` and the GPUs released before ``command`` runs."""
    lines = ["#!/bin/bash", "set -uo pipefail"]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in env.items()]
    if command.role != ROUTER:
        if framework is Framework.SGLANG:
            lines += _ib_device_lines()
            lines.append("export SGLANG_HOST_IP=\"${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}\"")
        lines += _close_inherited_lines()
        lines += _log_lines(log_dir, command) if log_dir.startswith("/") else []
        lines += _gpu_release_lines()
    return "\n".join([*lines, f"exec {command.command}"]) + "\n"


def _driver_payload(role: str, script: str, needle: str, action: str, mode: LaunchMode) -> str:
    """Bash that writes the launch script of ``role`` and runs the driver with ``action`` on it."""
    marker = uuid.uuid4().hex
    environment = [] if mode is LaunchMode.DIRECT else _container_env_lines()
    return "\n".join(
        [
            f"cat > {_launch_path(role)} << 'HL_LAUNCH_{marker}'",
            script.rstrip("\n"),
            f"HL_LAUNCH_{marker}",
            f"chmod +x {_launch_path(role)}",
            *environment,
            f"export HL_LAUNCH_PATH={_launch_path(role)} HL_PIDFILE={_pidfile(role)} HL_ACTION={action}",
            f"export HL_NEEDLE={shlex.quote(needle)} HL_ENV_SOURCE={'session' if mode is LaunchMode.DIRECT else 'container'}",
            f"python3 - << 'HL_PY_{marker}'",
            _DRIVER.read_text(encoding="utf-8").rstrip("\n"),
            f"HL_PY_{marker}",
        ]
    )


def _drive(cluster: Cluster, command: HostCommand, script: str, launch: PdLaunch, timeout: int) -> None:
    """Run the driver for ``command``'s host: a restart with ``script``, or a stop when there is none.

    Raises:
        LaunchError: the host did not report success.
    """
    action = "restart" if script else "stop"
    needle = process_needle(command.role, launch.framework, launch.frontend.type)
    payload = _driver_payload(command.role, script, needle, action, cluster.nodes.launch_mode)
    result = cluster.client.run(command.host, payload, timeout)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    report = json.loads(lines[-1]) if lines else {}
    if result.returncode != 0 or report.get("status") != "ok":
        raise LaunchError(f"{action} of {command.role} on {command.host} failed: {result.stderr.strip()[-1500:]}")


def check_env(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` as a launch exports it.

    Raises:
        LaunchError: a key is no shell identifier, or names a driver input (``HL_*``).
    """
    for key in env:
        if not _ENV_KEY.fullmatch(key) or key.startswith("HL_"):
            raise LaunchError(f"{key!r} cannot be exported to a PD server")
    return {key: str(value) for key, value in env.items()}


def probe_engine_flags(launch: PdLaunch, cluster: Cluster, timeout: int) -> frozenset[str]:
    """The canonical flags the first engine host's ``sglang.launch_server --help`` lists; empty for ATOM or when
    the probe lists nothing."""
    if launch.framework is not Framework.SGLANG:
        return frozenset()
    environment = [] if cluster.nodes.launch_mode is LaunchMode.DIRECT else _container_env_lines()
    script = "\n".join(
        [
            *environment,
            "python3 -m sglang.launch_server --help 2>/dev/null | grep -oE -- '--[a-z][a-z0-9_-]*' | sort -u",
        ]
    )
    result = cluster.client.run(launch.role(launch.served_roles[0]).hosts[0], script, timeout)
    return frozenset(group_flag([line.strip()]) for line in result.stdout.splitlines() if line.startswith("--"))


def _fit_role(role: Role, launched: RoleLaunch, flags: frozenset[str], source: PdSourceKind) -> RoleLaunch:
    """``launched`` on an engine listing ``flags``: unlisted flags of a non-operator source dropped, split flags
    in their decode spelling (taken out of prefill)."""
    kept: list[str] = []
    dropped: list[str] = []
    for group in flag_groups(launched.args):
        name = group_flag(group)
        spelled, sign, value = group[0].partition("=")
        split = name in SPLIT_ROLE_FLAGS and name not in flags and f"{name}-decode" in flags
        if split and role is Role.DECODE:
            kept += [f"--{name}-decode{sign}{value}", *group[1:]]
        elif split or (name and name not in flags and source is not PdSourceKind.OPERATOR):
            dropped.append(f"unsupported {spelled}")
        else:
            kept += group
    return replace(launched, args=tuple(kept), stripped=(*launched.stripped, *dropped))


def fit_to_engine(launch: PdLaunch, flags: frozenset[str]) -> PdLaunch:
    """``launch`` with each role fitted to an engine listing ``flags``; unchanged when ``flags`` is empty."""
    if not flags:
        return launch
    kind = launch.source.kind
    return launch.with_roles({role: _fit_role(role, launch.role(role), flags, kind) for role in launch.served_roles})


def replace_flags(args: Sequence[str], overrides: Sequence[str]) -> tuple[str, ...]:
    """``args`` with every flag ``overrides`` sets replaced by its value there, the rest appended."""
    named = {group_flag(group) for group in flag_groups(overrides)} - {""}
    kept = [token for group in flag_groups(args) if group_flag(group) not in named for token in group]
    return (*kept, *overrides)


def with_overrides(launch: PdLaunch, overrides: Mapping[str, Sequence[str]]) -> PdLaunch:
    """``launch`` with flag overrides of each role it runs (``prefill`` and ``decode``, or ``worker``) and of the
    ``router``.

    Raises:
        LaunchError: an override sets a flag the launcher owns, or names no role of the launch.
    """
    unknown = set(overrides) - {*(role.value for role in launch.served_roles), ROUTER}
    if unknown:
        raise LaunchError(f"overrides name no role: {sorted(unknown)}")
    roles = {}
    for role in launch.served_roles:
        given = list(overrides.get(role.value, ()))
        owned = render_role(launch.framework, given, {}).stripped
        if owned:
            raise LaunchError(f"the launcher sets {list(owned)}; they cannot be overridden on {role.value}")
        roles[role] = replace(launch.role(role), args=replace_flags(launch.role(role).args, given))
    frontend = replace(launch.frontend, args=replace_flags(launch.frontend.args, list(overrides.get(ROUTER, ()))))
    return replace(launch.with_roles(roles), frontend=frontend)


def _health_urls(launch: PdLaunch, nodes: NodeList, role: Role) -> list[str]:
    return [f"{url}/health" for url in engine_urls(launch, nodes)[role.value]]


def _log_tail(cluster: Cluster, command: HostCommand, timeout: int) -> str:
    result = cluster.client.run(command.host, f"tail -n {_LOG_TAIL_LINES} {_launch_path(command.role)}.log", timeout)
    return result.stdout.strip()


def _engine_commands(launch: PdLaunch, nodes: NodeList) -> list[HostCommand]:
    """Every engine host's command, in launch order."""
    return [command for command in cluster_commands(launch, nodes) if command.role != ROUTER]


def engines_gone(launch: PdLaunch, cluster: Cluster, timeout: int = 30) -> list[HostCommand]:
    """The engine hosts whose launched server no longer runs (a restarted container lost it, or it exited)."""
    gone = []
    for command in _engine_commands(launch, cluster.nodes):
        needle = process_needle(command.role, launch.framework, launch.frontend.type)
        patterns = " ".join(f"-e {shlex.quote(part)}" for part in needle.split("|"))
        script = (
            f'pid=$(cat {_pidfile(command.role)} 2>/dev/null) && kill -0 "$pid" 2>/dev/null && tr "\\0" " " '
            f'< /proc/"$pid"/cmdline | grep -qF {patterns} && echo HL_ALIVE || echo HL_GONE'
        )
        if "HL_GONE" in cluster.client.run(command.host, script, timeout).stdout:
            gone.append(command)
    return gone


def _await(urls: Sequence[str], launch: PdLaunch, cluster: Cluster, options: LaunchOptions) -> None:
    """Poll ``urls`` until each answers, failing when an engine dies or ``ready_timeout_s`` passes.

    Raises:
        LaunchError: an engine is gone, or a URL does not answer in time.
    """
    deadline = cluster.clock() + options.ready_timeout_s
    waiting = list(urls)
    while True:
        waiting = [url for url in waiting if not cluster.probe(url)]
        if not waiting:
            return
        gone = engines_gone(launch, cluster)
        if gone:
            tail = _log_tail(cluster, gone[0], options.ssh_timeout_s)
            raise LaunchError(f"{gone[0].role} on {gone[0].host} exited during startup:\n{tail}")
        if cluster.clock() >= deadline:
            raise LaunchError(f"not ready after {options.ready_timeout_s}s: {waiting}")
        cluster.sleep(options.poll_interval_s)


def start(launch: PdLaunch, cluster: Cluster, options: LaunchOptions) -> PdLaunch:
    """Run ``launch`` on the cluster, replacing what each host serves; returns the launch as it was run.

    Raises:
        LaunchError: a host refused, an engine died, or the cluster did not come up in time.
        SshError: a host key cannot be learned.
    """
    env = check_env(options.extra_env)
    fitted = fit_to_engine(launch, probe_engine_flags(launch, cluster, options.ssh_timeout_s))
    for command in _engine_commands(fitted, cluster.nodes):
        role_env = {**fitted.role(Role(command.role)).env, **env, **check_env(options.role_env.get(command.role, {}))}
        script = launch_script(command, role_env, fitted.framework, options.log_dir)
        _drive(cluster, command, script, fitted, options.ssh_timeout_s)
    engines = [url for role in fitted.served_roles for url in _health_urls(fitted, cluster.nodes, role)]
    _await(engines, fitted, cluster, options)
    service = [f"{cluster.nodes.service_url.rstrip('/')}/health"]
    if runs_router(fitted):
        router = router_command(fitted, cluster.nodes)
        _drive(cluster, router, launch_script(router, env, fitted.framework, ""), fitted, options.ssh_timeout_s)
        service.insert(0, f"http://{router.host}:{ROUTER_PORT}/health")
    _await(service, fitted, cluster, options)
    return fitted


def restart(
    launch: PdLaunch, overrides: Mapping[str, Sequence[str]], cluster: Cluster, options: LaunchOptions
) -> PdLaunch:
    """Run ``launch`` with per-role flag overrides; returns the launch as it was run."""
    return start(with_overrides(launch, overrides), cluster, options)


def stop(launch: PdLaunch, cluster: Cluster, options: LaunchOptions) -> None:
    """Stop the router, then every engine; a server the container launcher waits on is parked, not killed."""
    routers = [router_command(launch, cluster.nodes)] if runs_router(launch) else []
    for command in [*routers, *_engine_commands(launch, cluster.nodes)]:
        _drive(cluster, command, "", launch, options.ssh_timeout_s)


__all__ = [
    "GPU_RELEASE_WAIT_S",
    "SPLIT_ROLE_FLAGS",
    "Cluster",
    "LaunchError",
    "LaunchOptions",
    "check_env",
    "engines_gone",
    "fit_to_engine",
    "http_ok",
    "launch_script",
    "probe_engine_flags",
    "replace_flags",
    "restart",
    "start",
    "stop",
    "with_overrides",
]
