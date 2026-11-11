# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SSH restart for atomesh and sglang-router clusters.

The platform launcher waits on the server pid. The remote helper execs the new
command into that pid so the pod is not recreated with the original entrypoint.
Prefill and decode receive ``SGLANG_TORCH_PROFILER_DIR``; the router does not.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .._internal import process_image
from .._internal.log import err, info, warn
from .._internal.pd_cluster import (
    ENGINE_PORT,
    PLATFORM_MANAGED,
    LaunchSpec,
    cmdline_needle,
    drop_unsupported_baseline_flags,
    launch_script,
    pods_for,
    spec_from_restart,
    uses_split_role_flags,
)
from .._internal.baseline_args import SOURCE_INFERENCEX
from .._internal.server_args_safety import ServerArgsRejected, validate_server_args

_REMOTE_DRIVER = r"""
import json

def _spawn_detached(launch):
    pid = os.fork()
    if pid == 0:
        os.setsid()
        log_fd = os.open(launch + ".log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.execv("/bin/bash", ["/bin/bash", launch])
    return pid


def _ppid(pid):
    stat = open("/proc/%d/stat" % pid, encoding="utf-8").read()
    return int(stat[stat.rfind(")") + 2 :].split()[1])


def _sid(pid):
    return os.getsid(pid)


def _is_parked(pid, parent, cmd):
    # The launcher's own child exec'd into `sleep infinity` (by this driver's stop or another tool).
    argv = cmd.split()
    if parent != 1 or len(argv) != 2 or os.path.basename(argv[0]) != "sleep" or argv[1] != "infinity":
        return False
    try:
        return _sid(pid) == _sid(1)
    except OSError:
        return False


def _find_server(pidfile, needle):
    if os.path.exists(pidfile):
        try:
            cand = int(open(pidfile, encoding="utf-8").read().strip())
            os.kill(cand, 0)
            return cand
        except (OSError, ValueError):
            pass
    self_pid = os.getpid()
    fallback = 0
    parked = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        cand = int(entry)
        if cand in (1, self_pid):
            continue
        try:
            cmd = open("/proc/%d/cmdline" % cand, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
            parent = _ppid(cand)
        except OSError:
            continue
        if not any(part in cmd for part in needle.split("|")):
            if parked == 0 and _is_parked(cand, parent, cmd):
                parked = cand
            continue
        if parent == 1:
            return cand
        if fallback == 0:
            fallback = cand
    return parked or fallback


def restart_remote_main():
    launch = os.environ.get("HL_LAUNCH_PATH", "")
    needle = os.environ.get("HL_NEEDLE", "")
    pidfile = os.environ["HL_PIDFILE"]
    action = os.environ.get("HL_ACTION", "restart")
    pid = _find_server(pidfile, needle)
    if action == "stop":
        if pid <= 0:
            print(json.dumps({"status": "ok", "action": "stop", "pid": 0}))
            return
        if _ppid(pid) == 1:
            replace_process_image(pid, ["/bin/sleep", "infinity"], env_snapshot=pidfile + ".environ")
            open(pidfile, "w", encoding="utf-8").write(str(pid))
            print(json.dumps({"status": "ok", "action": "stop-exec", "pid": pid}))
            return
        for victim in [pid] + descendant_pids(pid):
            try:
                os.kill(victim, 9)
            except OSError:
                pass
        try:
            os.remove(pidfile)
        except OSError:
            pass
        print(json.dumps({"status": "ok", "action": "stop-kill", "pid": pid}))
        return
    if pid <= 0:
        new_pid = _spawn_detached(launch)
        open(pidfile, "w", encoding="utf-8").write(str(new_pid))
        print(json.dumps({"status": "ok", "action": "spawn", "pid": new_pid}))
        return
    if _ppid(pid) == 1:
        replace_process_image(pid, ["/bin/bash", launch], env_snapshot=pidfile + ".environ")
        open(pidfile, "w", encoding="utf-8").write(str(pid))
        print(json.dumps({"status": "ok", "action": "exec", "pid": pid}))
        return
    for victim in [pid] + descendant_pids(pid):
        try:
            os.kill(victim, 9)
        except OSError:
            pass
    new_pid = _spawn_detached(launch)
    open(pidfile, "w", encoding="utf-8").write(str(new_pid))
    print(json.dumps({"status": "ok", "action": "respawn", "pid": new_pid}))


restart_remote_main()
"""


def _cli():
    """The multi-node CLI, imported lazily to avoid an import cycle."""
    from .. import cli as mn_cli

    return mn_cli


def _remote_program() -> str:
    """process_image helpers plus the pod-side restart driver."""
    return Path(process_image.__file__).read_text(encoding="utf-8") + "\n" + _REMOTE_DRIVER


def _pidfile(role: str) -> str:
    """Pod-local file holding the pid of the server this restart launched."""
    return f"/tmp/hyperloom-mn-{role}.pid"


def _bash_payload(role: str, script_body: str, needle: str, action: str) -> str:
    """Bash that writes the launch script and runs the in-place restart helper."""
    marker = uuid.uuid4().hex
    launch_end = f"HL_LAUNCH_{marker}"
    py_end = f"HL_PY_{marker}"
    return "\n".join(
        [
            f"launch=/tmp/hyperloom-mn-{role}.sh",
            f"cat > \"$launch\" << '{launch_end}'",
            script_body.rstrip("\n"),
            launch_end,
            "chmod +x \"$launch\"",
            f"export HL_LAUNCH_PATH=\"$launch\"",
            f"export HL_NEEDLE={shlex.quote(needle)}",
            f"export HL_PIDFILE={_pidfile(role)}",
            f"export HL_ACTION={shlex.quote(action)}",
            f"python3 - << '{py_end}'",
            _remote_program().rstrip("\n"),
            py_end,
        ]
    )


def _require_state() -> dict[str, Any]:
    state = _cli()._load_state()
    backend = str(state.get("backend") or "")
    if backend not in ("atomesh", "sglangrouter"):
        raise _cli().ConfigurationError(f"state backend is {backend!r}, not atomesh or sglangrouter")
    if not state.get("ssh_key_path"):
        raise _cli().ConfigurationError("no ssh_key_path in state; set HYPERLOOM_MN_EXT_SSH_KEY")
    if not state.get("ssh_port"):
        raise _cli().ConfigurationError("no ssh_port in state; set HYPERLOOM_MN_EXT_SSH_PORT")
    if not state.get("prefill_pod_ips") or not state.get("decode_pod_ips"):
        raise _cli().ConfigurationError(
            "atomesh/sglangrouter restart needs HYPERLOOM_MN_EXT_PREFILL_IPS and HYPERLOOM_MN_EXT_DECODE_IPS"
        )
    return state


def _forward_env() -> dict[str, str]:
    from .infera import _collect_forward_env

    return dict(_collect_forward_env())


def _refresh_pods(state: dict[str, Any], spec: LaunchSpec) -> None:
    """Rebuild pod records for ``spec``; a platform-managed router is re-derived from ``service_url``, never SSH'd."""
    port = state.get("ssh_port")
    port_i = int(port) if port else None
    platform_ips = {
        str(pod.get("podIP"))
        for pod in state.get("router_pods") or []
        if isinstance(pod, dict) and pod.get("managed_by") == PLATFORM_MANAGED
    }
    pods = pods_for(
        [str(ip) for ip in state.get("router_pod_ips") or [] if str(ip) not in platform_ips],
        [str(ip) for ip in state.get("prefill_pod_ips") or []],
        [str(ip) for ip in state.get("decode_pod_ips") or []],
        port_i,
        spec,
        service_url=str(state.get("service_url") or ""),
    )
    state["pods"] = pods
    state["router_pods"] = [pod for pod in pods if pod["role"] == "router"]
    state["router_pod_ips"] = [str(pod["podIP"]) for pod in state["router_pods"]]
    state["prefill_pods"] = [pod for pod in pods if pod["role"] == "prefill"]
    state["decode_pods"] = [pod for pod in pods if pod["role"] == "decode"]


def _role_env(extra_env: dict[str, str], role: str) -> dict[str, str]:
    """``extra_env`` with an engine role's ``SGLANG_TORCH_PROFILER_DIR`` narrowed to ``<dir>/<role>``."""
    profiler_dir = str(extra_env.get("SGLANG_TORCH_PROFILER_DIR") or "").rstrip("/")
    if role not in ("prefill", "decode") or not profiler_dir:
        return extra_env
    return {**extra_env, "SGLANG_TORCH_PROFILER_DIR": f"{profiler_dir}/{role}"}


def _ssh_pod(
    state: dict[str, Any],
    pod: dict[str, Any],
    *,
    action: str,
    extra_env: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    role = str(pod.get("role") or "")
    ip = str(pod.get("podIP") or "")
    port = int(pod.get("sshPort") or state.get("ssh_port") or 0)
    body = launch_script(role, str(pod.get("launch_command") or "true"), _role_env(extra_env, role), log_ip=ip)
    needle = cmdline_needle(role, str(state.get("backend") or "atomesh"))
    payload = _bash_payload(role, body, needle, action)
    info(f"{state.get('backend')} {action}: ssh -> {ip}:{port} role={role}")
    try:
        completed = _cli()._infera_ssh_bash_with_env(state, ip, payload, timeout=timeout, port=port)
    except subprocess.TimeoutExpired:
        return {"podIP": ip, "sshPort": port, "role": role, "rc": 124, "error": "timeout"}
    record: dict[str, Any] = {
        "podIP": ip,
        "sshPort": port,
        "role": role,
        "rc": completed.returncode,
    }
    parsed = _cli()._extract_pod_json(completed.stdout or "")
    if parsed:
        record["summary"] = parsed
    if completed.returncode != 0:
        record["stderr"] = (completed.stderr or "")[-1500:]
    return record


def _engine_pods(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefill then decode pod records that carry an IP."""
    pods = list(state.get("prefill_pods") or []) + list(state.get("decode_pods") or [])
    return [pod for pod in pods if isinstance(pod, dict) and pod.get("podIP")]


CONTAINER_ENVIRON_PATH = "/proc/1/environ"


def _container_env_line(environ_path: str) -> str:
    """Bash that exports the container's interpreter search paths read from ``environ_path``."""
    return (
        "while IFS= read -r -d '' kv; do case \"$kv\" in PATH=*|PYTHONPATH=*|VIRTUAL_ENV=*|LD_LIBRARY_PATH=*) "
        f'export "$kv";; esac; done < {shlex.quote(environ_path)}\n'
    )


def _flag_probe_script(environ_path: str = CONTAINER_ENVIRON_PATH) -> str:
    """Bash that lists ``sglang.launch_server --help`` options under the container environment in ``environ_path``."""
    return (
        _container_env_line(environ_path)
        + "python3 -m sglang.launch_server --help 2>/dev/null | grep -oE -- '--[a-z][a-z0-9-]*' | sort -u"
    )


def _pod_python_script(text: str, interpreter: str, args: str) -> str:
    """Bash that writes the Python ``text`` to a temp file and runs it with ``args`` under the container environment."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return (
        _container_env_line(CONTAINER_ENVIRON_PATH)
        + 'script="$(mktemp)"\n'
        + f'echo {encoded} | base64 -d > "$script"\n'
        + f'{interpreter} "$script" {args}\n'
        + 'rc=$?\nrm -f "$script"\nexit $rc'
    )


def probe_engine_flags(state: dict[str, Any], spec: LaunchSpec, *, timeout: int) -> frozenset[str]:
    """Option strings of the pod's ``sglang.launch_server``, cached in ``state``.

    Probed only when the round carries a ``SPLIT_ROLE_FLAGS`` flag or the
    baseline comes from InferenceX. Empty when the probe fails, which leaves
    the launch args unchanged.
    """
    cached = state.get("engine_cli_flags")
    if isinstance(cached, list):
        return frozenset(str(flag) for flag in cached)
    words = shlex.split(" ".join((spec.server_args, spec.prefill_extra, spec.decode_extra)))
    pods = _engine_pods(state)
    needed = uses_split_role_flags(words) or _inferencex_baseline(state)
    if not needed or not pods:
        return frozenset()
    pod = pods[0]
    ip = str(pod["podIP"])
    port = int(pod.get("sshPort") or state.get("ssh_port") or 0)
    script = _flag_probe_script()
    try:
        completed = _cli()._infera_ssh_bash_with_env(state, ip, script, timeout=timeout, port=port)
    except subprocess.TimeoutExpired:
        warn(f"engine flag probe timed out on {ip}:{port}")
        return frozenset()
    flags = frozenset(line.strip() for line in (completed.stdout or "").splitlines() if line.startswith("--"))
    if not flags:
        warn(f"engine flag probe on {ip}:{port} listed no options; launch args are used as given")
        return frozenset()
    state["engine_cli_flags"] = sorted(flags)
    return flags


def _inferencex_baseline(state: dict[str, Any]) -> bool:
    """True when the session baseline args came from InferenceX."""
    return str(state.get("baseline_args_source") or "") == SOURCE_INFERENCEX


def without_unsupported_baseline_flags(state: dict[str, Any], spec: LaunchSpec) -> LaunchSpec:
    """``spec`` minus InferenceX baseline flags that the probed engine does not list."""
    if not _inferencex_baseline(state) or not spec.engine_flags:
        return spec
    baseline = shlex.split(f"{state.get('baseline_prefill_args') or ''} {state.get('baseline_decode_args') or ''}")
    changes: dict[str, str] = {}
    for name in ("server_args", "prefill_extra", "decode_extra"):
        kept, dropped = drop_unsupported_baseline_flags(shlex.split(getattr(spec, name)), baseline, spec.engine_flags)
        if dropped:
            warn(f"engine does not list baseline flag(s) {dropped}; dropped from {name}")
            changes[name] = shlex.join(kept)
    return dataclasses.replace(spec, **changes) if changes else spec


def engines_not_running(state: dict[str, Any], *, timeout: int = 30) -> list[str]:
    """``"<role> <ip>"`` of each engine pod whose launched server is no longer running.

    A pod whose container restarted has lost the pidfile and runs its platform
    entrypoint instead of the launched command. Unreachable pods are not listed.
    """
    gone: list[str] = []
    for pod in _engine_pods(state):
        role = str(pod.get("role") or "")
        ip = str(pod["podIP"])
        port = int(pod.get("sshPort") or state.get("ssh_port") or 0)
        needle = cmdline_needle(role, str(state.get("backend") or "atomesh"))
        patterns = " ".join(f"-e {shlex.quote(part)}" for part in needle.split("|"))
        script = (
            f'pid=$(cat {_pidfile(role)} 2>/dev/null) && kill -0 "$pid" 2>/dev/null '
            f"&& tr '\\0' ' ' < /proc/\"$pid\"/cmdline | grep -qF {patterns} && echo HL_ALIVE || echo HL_GONE"
        )
        try:
            completed = _cli()._infera_ssh_bash_with_env(state, ip, script, timeout=timeout, port=port)
        except subprocess.TimeoutExpired:
            continue
        if "HL_GONE" in (completed.stdout or ""):
            gone.append(f"{role} {ip}")
    return gone


def _ordered_pods(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Engines first, then the router, so the router restarts against live workers; platform-managed pods excluded."""
    pods = [
        pod for pod in (state.get("pods") or []) if isinstance(pod, dict) and pod.get("managed_by") != PLATFORM_MANAGED
    ]
    return [pod for pod in pods if pod.get("role") != "router"] + [pod for pod in pods if pod.get("role") == "router"]


def restart_pd_cluster(args: argparse.Namespace) -> int:
    """SSH-restart every router, prefill, and decode pod."""
    state = _require_state()
    try:
        for flag in ("extra_args", "pd_prefill_extra_args", "pd_decode_extra_args"):
            validate_server_args(getattr(args, flag, "") or "", context=f"restart-server --{flag.replace('_', '-')}")
    except ServerArgsRejected as exc:
        err(str(exc))
        return _cli().EXIT_CONFIG_ERROR
    spec = spec_from_restart(args, state)
    if not spec.model:
        err("restart-server: model is empty")
        return _cli().EXIT_CONFIG_ERROR
    timeout = int(getattr(args, "poll_timeout", 180) or 180)
    spec = dataclasses.replace(spec, engine_flags=probe_engine_flags(state, spec, timeout=timeout))
    spec = without_unsupported_baseline_flags(state, spec)
    _refresh_pods(state, spec)
    extra_env = _forward_env()
    results = [
        _ssh_pod(state, pod, action="restart", extra_env=extra_env, timeout=timeout) for pod in _ordered_pods(state)
    ]
    rc = 0 if results and all(int(item.get("rc") or 0) == 0 for item in results) else 1
    state["last_restart_framework"] = spec.framework
    state["last_restart_model"] = spec.model
    state["last_restart_tp"] = int(getattr(args, "tp", 0) or spec.prefill_tp)
    state["last_restart_ep"] = spec.ep
    state["last_restart_pd_mode"] = "disaggregated"
    state["last_restart_extra_args"] = _cli()._normalize_extra_args(getattr(args, "extra_args", ""))
    _cli()._save_state(state)
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


def kill_pd_cluster(args: argparse.Namespace) -> int:
    """Stop engine and router processes without dropping the launcher's waited pid."""
    state = _require_state()
    timeout = int(getattr(args, "poll_timeout", 180) or 180)
    results = [_ssh_pod(state, pod, action="stop", extra_env={}, timeout=timeout) for pod in _ordered_pods(state)]
    rc = 0 if results and all(int(item.get("rc") or 0) == 0 for item in results) else 1
    print(json.dumps({"backend": state.get("backend"), "action": "kill", "rc": rc, "results": results}, indent=2))
    return 0 if rc == 0 else 1


def apply_tracelens_patch(args: argparse.Namespace) -> int:
    """SSH the TraceLens SGLang patch onto every prefill and decode pod."""
    cli = _cli()
    state = _require_state()
    tracelens_root = args.tracelens_root or os.environ.get("TRACELENS_ROOT", "").strip()
    if not tracelens_root:
        err("apply-tracelens-patch requires --tracelens-root or $TRACELENS_ROOT")
        return cli.EXIT_CONFIG_ERROR
    targets = list(state.get("prefill_pods") or []) + list(state.get("decode_pods") or [])
    script = cli._read_bundled_pod_python_script("apply_tracelens_patch_multinode.py", cli._SHAPE_GATE_DEPS)
    op_args = f"--local --tracelens-root {shlex.quote(str(tracelens_root))}"
    if getattr(args, "sglang_version_pin", None):
        op_args += f" --sglang-version-pin {shlex.quote(str(args.sglang_version_pin))}"
    patch_set = getattr(args, "patch_set", None)
    if patch_set:
        op_args += f" --patch-set {shlex.quote(str(patch_set))}"
    timeout = cli._poll_timeout_from_args(args)
    pod_python = os.environ.get("HYPERLOOM_MN_POD_PYTHON", "python3")
    pod_script = _pod_python_script(script, pod_python, op_args)

    per_pod: list[dict] = []
    failures: list[dict] = []
    for target in targets:
        ip = str(target.get("podIP") or "")
        port = int(target.get("sshPort") or state.get("ssh_port") or 0)
        info(f"apply-tracelens-patch ({state.get('backend')}): ssh -> {ip}:{port}")
        try:
            completed = cli._infera_ssh_bash_with_env(state, ip, pod_script, timeout=timeout, port=port)
        except subprocess.TimeoutExpired:
            failures.append({"host": ip, "sshPort": port, "error": f"timeout after {timeout}s"})
            continue
        parsed = cli._extract_pod_json(completed.stdout) or {}
        rows = parsed.get("per_pod")
        if str(parsed.get("status")) in ("applied", "skipped") and rows:
            for row in rows:
                row["host"] = ip
                per_pod.append(row)
        else:
            pod_errors = "; ".join(str(row["error"]) for row in rows or [] if isinstance(row, dict) and row.get("error"))
            failures.append(
                {
                    "host": ip,
                    "sshPort": port,
                    "error": parsed.get("error") or pod_errors or (completed.stderr or "")[-800:] or "unknown",
                    "rc": completed.returncode,
                }
            )
    overall = "applied" if not failures else "failed"
    if overall == "applied" and per_pod and all(row.get("status") == "skipped" for row in per_pod):
        overall = "skipped"
    print(
        json.dumps(
            {
                "command": "apply-tracelens-patch",
                "backend": state.get("backend"),
                "status": overall,
                "per_pod": per_pod,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


PROFILE_SHARE_PROBE_TIMEOUT_S = 30


def probe_profile_dir_shared(state: dict[str, Any]) -> str:
    """Check that every prefill and decode pod reads the sandbox's ``profile_trace`` dir; '' when shared, else why not."""
    root = str(state.get("profile_trace") or "").strip()
    if not root:
        return "no profile_trace dir in state"
    if not state.get("ssh_key_path"):
        return "no ssh_key_path in state"
    pods = [
        pod
        for pod in list(state.get("prefill_pods") or []) + list(state.get("decode_pods") or [])
        if isinstance(pod, dict) and pod.get("podIP")
    ]
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
        completed = _cli()._infera_ssh_bash_with_env(
            state, ip, f"cat {shlex.quote(str(sentinel))}", timeout=PROFILE_SHARE_PROBE_TIMEOUT_S, port=port
        )
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
    state["profile_trace_shared"] = not reason
    if reason:
        state["profile_trace_share_error"] = reason
    else:
        state.pop("profile_trace_share_error", None)
    return reason
