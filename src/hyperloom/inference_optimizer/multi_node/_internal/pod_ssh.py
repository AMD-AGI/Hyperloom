# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""State file, forward-env and pod SSH helpers shared by the multi-node CLI and its command modules."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..state_paths import _ensure_runtime_dir, resolve_state_file
from . import ssh_client, ssh_known_hosts
from .env_safety import filter_forward_env, server_log_dir
from .external_state import external_service_url, load_multi_node_state
from .log import warn

EXIT_CONFIG_ERROR = 3  # missing env / required arg — fix the call, don't retry blindly


class ConfigurationError(RuntimeError):
    """Raised when the environment is wrong in a way retrying cannot fix. Exit code -> EXIT_CONFIG_ERROR."""


def _normalize_extra_args(s: str | None) -> str:
    """Normalize ``--extra-args`` whitespace for equality (order-preserving; argv order matters)."""
    return " ".join((s or "").split())


# State file
def _state_file() -> Path:
    """Return the active multi-node state file path."""
    return resolve_state_file()


def _infera_ssh_dir() -> Path:
    """Session-scoped directory for the ephemeral multi-node SSH keypair."""
    return _state_file().parent / "mn_ssh"


def _load_state() -> dict[str, Any]:
    """Load the CLI state file as a dict."""
    state = load_multi_node_state()
    if not external_service_url():
        try:
            nodes = int(os.environ.get("INFERENCE_OPTIMIZER_NODES", "") or 1)
        except ValueError:
            nodes = 1
        if nodes >= 2 or state.get("external"):
            raise ConfigurationError(
                "multi-node state requested without a cluster hand-off: "
                "HYPERLOOM_MN_EXT_SERVICE_URL is unset"
                + (" while the state file describes a handed-over cluster" if state.get("external") else "")
                + ". The platform provisions the cluster and exports the HYPERLOOM_MN_EXT_* "
                "vars; see multi_node/SKILL.md 'Cluster hand-off'."
            )
    return state


def _save_state(state: dict[str, Any]) -> None:
    """Write the CLI state dict to the state file as pretty JSON."""
    path = _state_file()
    _ensure_runtime_dir(path.parent)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError as exc:
        warn(f"could not chmod state file {path} to 0600: {exc}")


def _infera_known_hosts_path(state: dict[str, Any] | None = None) -> Path:
    """Resolve the session known_hosts file for Infera SSH."""
    st = state if state is not None else _load_state()
    raw = str(st.get("ssh_known_hosts") or "").strip()
    if raw:
        return Path(raw)
    return ssh_known_hosts.default_known_hosts_path(_infera_ssh_dir())


def _infera_default_ssh_port(state: dict[str, Any]) -> int:
    """Return the session default SSH port from state."""
    return int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)


def _infera_ssh_with_host_key_retry(
    state: dict[str, Any],
    ip: str,
    port: int | None,
    run: Callable[[ssh_client.SshTarget], subprocess.CompletedProcess],
) -> subprocess.CompletedProcess:
    """Run ``run`` against an Infera pod; on a host-key mismatch, re-scan known_hosts and retry once."""
    target = ssh_client.SshTarget(
        host=ip,
        key_path=state["ssh_key_path"],
        port=int(port if port is not None else _infera_default_ssh_port(state)),
        known_hosts=_infera_known_hosts_path(state),
    )
    cp = run(target)
    if cp.returncode != 0 and ssh_known_hosts.is_host_key_error(cp.stderr):
        warn(f"infera ssh {ip}:{target.port}: host key mismatch; refreshing known_hosts and retrying once")
        try:
            known_hosts = ssh_known_hosts.refresh_known_hosts([(ip, target.port)], target.known_hosts)
        except RuntimeError as exc:
            warn(f"infera ssh {ip}:{target.port}: known_hosts refresh failed: {exc}")
            return cp
        cp = run(replace(target, known_hosts=known_hosts))
    return cp


def _infera_ssh_bash_with_env(
    state: dict[str, Any],
    ip: str,
    script: str,
    *,
    timeout: int,
    port: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a bash script on an Infera pod via SSH stdin with host-key retry."""
    return _infera_ssh_with_host_key_retry(
        state, ip, port, lambda target: ssh_client.ssh_run_bash_stdin(target, script, timeout=timeout)
    )


def _json_object_candidates(text: str) -> Iterator[str]:
    """Yield balanced ``{...}`` blocks from the end of ``text`` backward, ignoring braces inside JSON strings."""
    depth = 0
    in_str = False
    end = -1
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if depth and ch == '"':
            j = i
            while j and text[j - 1] == "\\":
                j -= 1
            if (i - j) % 2 == 0:
                in_str = not in_str
        elif in_str:
            continue
        elif ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{" and depth:
            depth -= 1
            if depth == 0:
                yield text[i : end + 1]


def _extract_pod_json(logs: str) -> dict | None:
    """Parse the last top-level JSON document from an interleaved Ray Dashboard job_logs blob (the in-pod scripts emit one)."""
    for candidate in _json_object_candidates(logs or ""):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def per_round_forward_overrides() -> dict[str, Any]:
    """Parse the per-round env control vars into ``{"set": {...}, "unset": [...]}``.

    Both multi-node backends launch from these, so they are parsed once here rather than
    once per backend. What each backend then does with them differs: the RayJob path
    filters the set through :func:`filter_forward_env`, while the Infera SSH path forwards
    it verbatim.
    """
    raw = os.environ.get("HYPERLOOM_MN_EXTRA_FWD_ENV", "").strip()
    overrides: dict[str, str] = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            warn("HYPERLOOM_MN_EXTRA_FWD_ENV is not valid JSON; skipping per-variant env forwarding")
            parsed = None
        if isinstance(parsed, dict):
            overrides = {str(k): str(v) for k, v in parsed.items()}
        elif parsed is not None:
            warn("HYPERLOOM_MN_EXTRA_FWD_ENV is not a JSON object; skipping per-variant env forwarding")

    raw_unset = os.environ.get("HYPERLOOM_MN_UNSET_FWD_ENV", "").strip()
    unset: list[str] = []
    if raw_unset:
        try:
            parsed_unset = json.loads(raw_unset)
        except (ValueError, TypeError):
            warn("HYPERLOOM_MN_UNSET_FWD_ENV is not valid JSON; skipping per-variant env unsets")
            parsed_unset = None
        if isinstance(parsed_unset, list):
            unset = sorted({str(k).strip() for k in parsed_unset if str(k).strip()})

    return {"set": overrides, "unset": unset}


# Env-var prefixes forwarded from the controller's os.environ to the SSH-launched framework child (sandbox-side tuning
# vars not present in the pod container env and not recovered from pid1).
_FORWARD_ENV_PREFIXES = ("MORI_", "SGLANG_MORI_", "SGLANG_DISAGGREGATION_")
# The prefixes a pod launch (infera, atomesh/sglangrouter pod hand-off) forwards: also the Mooncake transfer settings.
POD_FORWARD_ENV_PREFIXES = (*_FORWARD_ENV_PREFIXES, "MC_")


def _collect_forward_env(prefixes: tuple[str, ...] = _FORWARD_ENV_PREFIXES) -> dict[str, str]:
    """Read prompt-provided tuning vars whose names start with one of ``prefixes`` from os.environ for SSH
    forwarding."""
    fwd = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    # Multi-node torch profiler: the infera SSH path (unlike the RayJob path in launch_multinode.py) never pins
    # SGLANG_TORCH_PROFILER_DIR, so sglang writes traces to pod-local /tmp where the sandbox cannot read them ->
    # roofline's profile_no_trace_failed.
    trace_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if trace_dir and "SGLANG_TORCH_PROFILER_DIR" not in fwd:
        fwd["SGLANG_TORCH_PROFILER_DIR"] = trace_dir
    # Forward no-patch shape-discovery config; the pod-side launcher sets
    # PYTHONPATH itself (it is blocked from SSH forwarding).
    for _shape_key in (
        "TRACELENS_ROOT",
        "TRACELENS_SHAPE_DISCOVERY",
        "HYPERLOOM_SGLANG_VERSION_PIN",
    ):
        _shape_val = os.environ.get(_shape_key, "").strip()
        if _shape_val and _shape_key not in fwd:
            fwd[_shape_key] = _shape_val
    # Explicit per-variant overrides are forwarded verbatim regardless of prefix and take precedence over
    # prefix-matched values for the same key; the unsets are applied first so an override can reinstate a key.
    overrides = per_round_forward_overrides()
    for key in overrides["unset"]:
        fwd.pop(key, None)
    fwd.update(overrides["set"])
    # Expand any $VAR (e.g. $USER_DATA_PATH) left in the profiler dir so the SSH-launched sglang on the pod (where
    # those vars are undefined) writes traces to an absolute shared-FS path, not an unresolved literal.
    if fwd.get("SGLANG_TORCH_PROFILER_DIR"):
        fwd["SGLANG_TORCH_PROFILER_DIR"] = os.path.expandvars(fwd["SGLANG_TORCH_PROFILER_DIR"])
    # Forward a shared-FS (WekaFS) server-log dir so the SSH-launched sglang writes server.log to shared storage the
    # client can read, not pod-local /tmp.
    _slog = server_log_dir()
    if _slog:
        fwd["HYPERLOOM_MN_SERVER_LOG_DIR"] = _slog
    # aiter/cpp_itfs runtime-compiled kernels (GH #458): the integrate re-baseline sets AITER_REBUILD=1 (transiently)
    # so aiter wipes its build dir and recompiles the patched kernel on import.
    aiter_rebuild = os.environ.get("AITER_REBUILD", "").strip()
    if aiter_rebuild:
        fwd["AITER_REBUILD"] = aiter_rebuild
    return filter_forward_env(fwd)


def _record_last_restart(state: dict[str, Any], args: argparse.Namespace, framework: str, pd_mode: str) -> None:
    """Stamp the launched config onto ``state`` for the next restart's resume fast-path."""
    state["last_restart_framework"] = framework
    state["last_restart_model"] = args.model
    state["last_restart_tp"] = int(args.tp)
    state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
    state["last_restart_pd_mode"] = pd_mode
    state["last_restart_extra_args"] = _normalize_extra_args(getattr(args, "extra_args", ""))
    state["last_restart_forward_env"] = per_round_forward_overrides()


CONTAINER_ENVIRON_PATH = "/proc/1/environ"


def container_env_line(environ_path: str) -> str:
    """Bash that exports the container's interpreter search paths read from ``environ_path``."""
    return (
        "while IFS= read -r -d '' kv; do case \"$kv\" in PATH=*|PYTHONPATH=*|VIRTUAL_ENV=*|LD_LIBRARY_PATH=*) "
        f'export "$kv";; esac; done < {shlex.quote(environ_path)}\n'
    )
