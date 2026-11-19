# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cluster access shared by the multi-node commands: state file, Ray Dashboard and SSH, polling, exit codes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from hyperloom_common.env import env_int

from ._internal import ray_dashboard
from ._internal import ssh_client, ssh_known_hosts
from ._internal.external_state import external_service_url, load_multi_node_state
from ._internal.log import info, warn
from .state_paths import _ensure_runtime_dir, resolve_state_file


# Default poll budget sized under the sandbox 120s ceiling.
_DEFAULT_POLL_INTERVAL_S = 6
_DEFAULT_POLL_TIMEOUT_S = 110


def resolve_poll_timeout_s(default: int = _DEFAULT_POLL_TIMEOUT_S) -> int:
    """Poll budget (seconds): ``HYPERLOOM_MN_POLL_TIMEOUT_S`` else *default*, never below one second."""
    return max(1, env_int("HYPERLOOM_MN_POLL_TIMEOUT_S", default))


def _poll_timeout_from_args(args: argparse.Namespace) -> int:
    """Resolve the poll timeout for a subcommand."""
    pt = getattr(args, "poll_timeout", None)
    if pt is not None:
        return max(1, int(pt))
    return resolve_poll_timeout_s()


# Ray dashboard job status strings.
_TERMINAL_FAIL_STATUSES = {"FAILED", "STOPPED"}
_TERMINAL_OK_STATUSES = {"SUCCEEDED"}


def _normalize_extra_args(s: str | None) -> str:
    """Normalize ``--extra-args`` whitespace for equality (order-preserving; argv order matters)."""
    return " ".join((s or "").split())


# Exit codes — part of the CLI's contract with the agent; keep stable.
EXIT_OK = 0
EXIT_TRANSIENT = 1  # network / Ray Dashboard or SSH error / timeout — caller may retry
EXIT_CONFIG_ERROR = 3  # missing env / required arg — fix the call, don't retry blindly
EXIT_INTERRUPT = 130  # Ctrl-C / SIGINT


class TransientFailure(RuntimeError):
    """Raised on poll timeout or repeated fetch failure. Exit code -> 1; caller may retry."""


class ConfigurationError(RuntimeError):
    """Raised when the environment is wrong in a way retrying cannot fix. Exit code -> EXIT_CONFIG_ERROR."""


# State file
def _state_file() -> Path:
    """Return the active multi-node state file path."""
    return resolve_state_file()


def _infera_ssh_dir() -> Path:
    """Session-scoped directory for the ephemeral multi-node SSH keypair."""
    return _state_file().parent / "mn_ssh"


def _multinode_server_log_dir() -> str:
    """Resolve a cross-node-visible directory for the per-rank server logs."""
    explicit = os.path.expandvars(os.environ.get("HYPERLOOM_MN_SERVER_LOG_DIR", "").strip())
    if explicit.startswith("/") and "$" not in explicit:
        return explicit
    try:
        return str(_state_file().parent / "server_logs")
    except RuntimeError:  # state file unresolvable (dev/single box)
        return str(Path(tempfile.gettempdir()) / "multi_node_logs")


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


def _require_state(*keys: str) -> dict[str, Any]:
    """Load state and assert all required keys are present."""
    state = _load_state()
    missing = [k for k in keys if not state.get(k)]
    if missing:
        raise RuntimeError(
            f"State file {_state_file()} missing required keys: {missing}. The cluster is provisioned by the "
            "platform and handed over via HYPERLOOM_MN_EXT_* -- check those are set (see multi_node/SKILL.md)."
        )
    return state


def _ray_dashboard_client(state: dict[str, Any] | None = None) -> ray_dashboard.RayDashboardClient:
    """Open a Ray Dashboard client using ``head_pod_ip`` and optional token from state."""
    st = state if state is not None else _load_state()
    head = str(st.get("head_pod_ip") or "").strip()
    if not head:
        raise RuntimeError("head_pod_ip missing from multi_node state")
    token = str(st.get("ray_dashboard_token") or "").strip() or None
    return ray_dashboard.RayDashboardClient(head, token=token)


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


def _infera_ssh_run_script(
    state: dict[str, Any],
    ip: str,
    script: ssh_client.RemoteScript,
    *,
    timeout: int,
    port: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a script on an Infera pod over SSH with host-key retry."""
    return _infera_ssh_with_host_key_retry(
        state, ip, port, lambda target: ssh_client.ssh_run_script(target, script, timeout=timeout)
    )


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


def _short_poll(
    *,
    label: str,
    fetch: Callable[[], tuple[Any, str]],
    interval_s: int,
    timeout_s: int,
) -> Any:
    """Poll a Ray job with many short fetches within one CLI invocation budget; returns the job on a terminal OK."""
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            obj, summary = fetch()
        except (httpx.HTTPError, httpx.InvalidURL, ray_dashboard.RayDashboardError, ValueError, AttributeError) as exc:
            warn(f"{label} poll #{attempt} fetch error: {exc}")
            obj, summary = None, str(exc)

        elapsed = time.monotonic() - started
        info(f"{label} poll #{attempt} t={elapsed:.0f}s -> {summary}")

        if obj is not None:
            status = str(obj.get("status", "")).upper()
            if status in _TERMINAL_OK_STATUSES:
                return obj
            if status in _TERMINAL_FAIL_STATUSES:
                raise RuntimeError(f"{label} terminal failure: {summary}")

        if elapsed >= timeout_s:
            raise TransientFailure(
                f"{label} did not reach a terminal state within {timeout_s}s "
                f"(last: {summary}). Re-run the same subcommand to keep polling."
            )
        time.sleep(interval_s)


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
