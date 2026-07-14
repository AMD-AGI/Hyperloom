# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``hyperloom.inference_optimizer.multi_node`` — single-entry sandbox CLI managing one session-scoped RayJob.

Subcommands (``python3 -m hyperloom.inference_optimizer.multi_node <sub>``):
``create-rayjob`` (create via SaFE + wait for Running, checkpointing
``rayjob_id`` so retries don't leak a second workload), ``bootstrap``,
``verify``, ``restart-server`` (kill + relaunch nohup'd server,
idempotent), ``kill-inference``, ``stop-rayjob``.

State lives in ``$MULTI_NODE_STATE_FILE`` (default:
``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR/runtime/multi_node_state.json``
when the session is pinned, else legacy ``/tmp/multi_node_state.json``).
HTTP polls under the sandbox's 120s ceiling (ADDENDUM-09) and surface
progress on stderr. Credentials must already be in sandbox env
(ADDENDUM-13); this module never invents URLs or keys.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..session.paths import mn_profile_trace_root
from ._internal import safe_client, ray_dashboard
from ._internal import ssh_client, ssh_known_hosts
from ._internal.log import info, warn, err
from ._internal.server_args_safety import ServerArgsRejected, validate_server_args
from .state_paths import legacy_state_file, resolve_state_file, state_file_safe_to_read

# Backward-compat alias for tests that monkeypatch ``STATE_FILE``.
STATE_FILE = resolve_state_file()

# Default poll budget sized under the sandbox 120s ceiling (ADDENDUM-09).
_DEFAULT_POLL_INTERVAL_S = 6
_DEFAULT_POLL_TIMEOUT_S = 110
# MoE cold-start often needs 20-30 min; set HYPERLOOM_MN_POLL_TIMEOUT_S=1800.
_DEFAULT_JIT_POLL_TIMEOUT_S = 1800


def _resolve_poll_timeout_s() -> int:
    """Poll budget (seconds): ``HYPERLOOM_MN_POLL_TIMEOUT_S`` env else ``_DEFAULT_POLL_TIMEOUT_S``.

    Returns:
        int: The poll timeout in seconds (at least 1); falls back to the
        default when the env var is unset or invalid.
    """
    raw = (os.environ.get("HYPERLOOM_MN_POLL_TIMEOUT_S") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            warn(f"invalid HYPERLOOM_MN_POLL_TIMEOUT_S={raw!r}; using {_DEFAULT_POLL_TIMEOUT_S}")
    return _DEFAULT_POLL_TIMEOUT_S


def _poll_timeout_from_args(args: argparse.Namespace) -> int:
    """Resolve the poll timeout for a subcommand.

    The ``--poll-timeout`` CLI flag wins; otherwise falls back to the
    env/default from :func:`_resolve_poll_timeout_s`.

    Args:
        args (argparse.Namespace): Parsed CLI args (may carry
            ``poll_timeout``).

    Returns:
        int: The poll budget in seconds (at least 1).
    """
    pt = getattr(args, "poll_timeout", None)
    if pt is not None:
        return max(1, int(pt))
    return _resolve_poll_timeout_s()




# Ray dashboard job status strings.
_TERMINAL_FAIL_STATUSES = {"FAILED", "STOPPED"}
_TERMINAL_OK_STATUSES = {"SUCCEEDED"}


def _normalize_extra_args(s: str | None) -> str:
    """Normalize ``--extra-args`` whitespace for equality (order-preserving; argv order matters).

    Args:
        s (str | None): The raw extra-args string, or ``None``.

    Returns:
        str: The string with runs of whitespace collapsed to single spaces
        (token order preserved). Empty for ``None``.
    """
    return " ".join((s or "").split())


# Exit codes — part of the CLI's contract with the agent; keep stable.
EXIT_OK = 0
EXIT_TRANSIENT = 1  # network / SaFE 5xx / timeout — caller may retry
EXIT_TERMINAL_FAILURE = 2  # workload entered Failed/Stopped/Cancelled — DO NOT retry; fix and recreate
EXIT_CONFIG_ERROR = 3  # missing env / required arg — fix the call, don't retry blindly
EXIT_INTERRUPT = 130  # Ctrl-C / SIGINT


class WorkloadTerminalFailure(RuntimeError):
    """Raised when SaFE reports a terminal failure phase (Failed/Stopped/Cancelled); carries the diag snapshot. Exit code -> 2."""

    def __init__(self, label: str, phase: str, diag: str, snapshot: dict[str, Any]) -> None:
        """Initialize the terminal-failure error.

        Args:
            label (str): The poll label that detected the failure.
            phase (str): The terminal SaFE phase.
            diag (str): One-line human-readable diagnostic.
            snapshot (dict[str, Any]): Structured failure snapshot.
        """
        super().__init__(f"{label} terminal phase={phase}: {diag}")
        self.label = label
        self.phase = phase
        self.diag = diag
        self.snapshot = snapshot


class TransientFailure(RuntimeError):
    """Raised on poll timeout or repeated SaFE communication failure. Exit code -> 1; caller may retry."""


# State file
def _state_file() -> Path:
    """Return the active multi-node state file path.

    Returns:
        Path: The state file path to read or write.
    """
    return resolve_state_file()


def _dynamo_ssh_dir() -> Path:
    """Session-scoped directory for the ephemeral multi-node SSH keypair.

    Returns:
        Path: Directory adjacent to the state file (``.../runtime/mn_ssh``).
    """
    return _state_file().parent / "mn_ssh"


def _load_state() -> dict[str, Any]:
    """Load the CLI state file as a dict.

    Returns:
        dict[str, Any]: The parsed state, or an empty dict if the file is
        missing, unreadable, or fails the ownership/permission check.
    """
    path = _state_file()
    if not path.is_file():
        legacy = legacy_state_file()
        if path != legacy and legacy.is_file() and state_file_safe_to_read(legacy):
            warn(f"state file {path} missing; reading legacy {legacy}")
            path = legacy
        else:
            return {}
    if not state_file_safe_to_read(path):
        warn(f"state file {path} failed ownership/permission check; ignoring")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"state file {path} unreadable: {exc}")
        return {}


def _save_state(state: dict[str, Any]) -> None:
    """Write the CLI state dict to the state file as pretty JSON.

    Args:
        state (dict[str, Any]): The state to persist.
    """
    path = _state_file()
    runtime_dir = path.parent
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime_dir.chmod(0o700)
    except OSError as exc:
        warn(f"could not chmod runtime dir {runtime_dir} to 0700: {exc}")
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError as exc:
        warn(f"could not chmod state file {path} to 0600: {exc}")




def _require_state(*keys: str) -> dict[str, Any]:
    """Load state and assert all required keys are present.

    Args:
        *keys (str): State keys that must be present and truthy.

    Returns:
        dict[str, Any]: The loaded state.

    Raises:
        RuntimeError: If any required key is missing or falsy.
    """
    state = _load_state()
    missing = [k for k in keys if not state.get(k)]
    if missing:
        raise RuntimeError(
            f"State file {_state_file()} missing required keys: {missing}. Have you run 'create-rayjob' first?"
        )
    return state




def _b64(s: str) -> str:
    """Base64-encode a string as ASCII.

    Args:
        s (str): The text to encode (UTF-8).

    Returns:
        str: The base64-encoded value as an ASCII string.
    """
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _wrap_for_dash(body: str) -> str:
    """Base64-wrap a bash entrypoint so it survives Ray Dashboard exec under /bin/sh (dash rejects ``set -o pipefail``).

    Args:
        body (str): The bash entrypoint body to wrap.

    Returns:
        str: A ``echo <b64> | base64 -d | bash`` command that decodes and runs
        ``body`` under bash.
    """
    enc = _b64(body)
    return f"echo {enc} | base64 -d | bash"


def _parse_kv_list(values: list[str] | None) -> dict[str, str]:
    """Convert ['K=V', 'K2=V2', ...] into a dict; ignore malformed entries.

    Args:
        values (list[str] | None): The raw ``K=V`` tokens, or ``None``.

    Returns:
        dict[str, str]: The parsed key/value mapping; malformed or
        empty-key tokens are skipped.
    """
    out: dict[str, str] = {}
    if not values:
        return out
    for raw in values:
        if "=" not in raw:
            warn(f"ignoring malformed K=V token: {raw!r}")
            continue
        k, _, v = raw.partition("=")
        k = k.strip()
        if not k:
            continue
        out[k] = v
    return out


def _ray_dashboard_client(state: dict[str, Any] | None = None) -> ray_dashboard.RayDashboardClient:
    """Open a Ray Dashboard client using ``head_pod_ip`` and optional token from state.

    Args:
        state: Multi-node state dict; loads from disk when omitted.

    Returns:
        ray_dashboard.RayDashboardClient: Client bound to the head pod.

    Raises:
        RuntimeError: When ``head_pod_ip`` is missing from state.
    """
    st = state if state is not None else _load_state()
    head = str(st.get("head_pod_ip") or "").strip()
    if not head:
        raise RuntimeError("head_pod_ip missing from multi_node state")
    token = str(st.get("ray_dashboard_token") or "").strip() or None
    return ray_dashboard.RayDashboardClient(head, token=token)


def _dynamo_known_hosts_path(state: dict[str, Any] | None = None) -> Path:
    """Resolve the session known_hosts file for Dynamo SSH.

    Args:
        state: Optional loaded multi-node state; when omitted, reads state.

    Returns:
        Path: The known_hosts file used by :mod:`ssh_client`.
    """
    st = state if state is not None else _load_state()
    raw = str(st.get("ssh_known_hosts") or "").strip()
    if raw:
        return Path(raw)
    return ssh_known_hosts.default_known_hosts_path(_dynamo_ssh_dir())


def _refresh_dynamo_known_hosts(
    ips: list[str],
    port: int,
    *,
    state: dict[str, Any] | None = None,
) -> Path:
    """Run ssh-keyscan for Dynamo GPU pods and persist the path in state.

    Args:
        ips: Pod IP addresses to scan.
        port: SSH port on each pod.
        state: Optional state dict to update with ``ssh_known_hosts``.

    Returns:
        Path: The refreshed known_hosts file.
    """
    kh = ssh_known_hosts.refresh_known_hosts(
        [(ip, int(port)) for ip in ips if (ip or "").strip()],
        _dynamo_known_hosts_path(state),
    )
    if state is not None:
        state["ssh_known_hosts"] = str(kh)
    return kh


def _dynamo_ssh_run_script(
    state: dict[str, Any],
    ip: str,
    script: str,
    interpreter: str,
    script_args: str,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
    remote_path: str = "/tmp/mn_dynamo_launch",
):
    """Run a script on a Dynamo pod over SSH with host-key retry.

    Args:
        state: Dynamo multi-node state (ssh key / port / known_hosts).
        ip: Target pod IP.
        script: Script body to ship.
        interpreter: Remote interpreter (e.g. ``python3``).
        script_args: Arguments appended after the script path.
        timeout: SSH timeout in seconds.
        env: Optional env assignments prepended before the interpreter.
        remote_path: Remote path for the decoded script.

    Returns:
        subprocess.CompletedProcess: The SSH subprocess result.
    """
    key_path = state["ssh_key_path"]
    port = int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)
    known_hosts = _dynamo_known_hosts_path(state)

    def _run(kh: Path):
        return ssh_client.ssh_run_script(
            ip,
            script,
            interpreter,
            script_args,
            key_path=key_path,
            known_hosts=kh,
            port=port,
            timeout=timeout,
            env=env,
            remote_path=remote_path,
        )

    cp = _run(known_hosts)
    if cp.returncode != 0 and ssh_known_hosts.is_host_key_error(cp.stderr):
        warn(f"dynamo ssh {ip}: host key mismatch; refreshing known_hosts and retrying once")
        try:
            known_hosts = _refresh_dynamo_known_hosts([ip], port, state=state)
        except RuntimeError as exc:
            warn(f"dynamo ssh {ip}: known_hosts refresh failed: {exc}")
            return cp
        _save_state(state)
        cp = _run(known_hosts)
    return cp


def _dynamo_ssh_bash_with_env(
    state: dict[str, Any],
    ip: str,
    script: str,
    env: dict[str, str] | None,
    *,
    timeout: int,
):
    """Run a bash script on a Dynamo pod via SSH stdin with host-key retry.

    Args:
        state: Dynamo multi-node state.
        ip: Target pod IP.
        script: Script body for ``bash -s``.
        env: Environment exports prepended to the script.
        timeout: SSH timeout in seconds.

    Returns:
        subprocess.CompletedProcess: The SSH subprocess result.
    """
    key_path = state["ssh_key_path"]
    port = int(state.get("ssh_port") or ssh_client.DEFAULT_SSH_PORT)
    known_hosts = _dynamo_known_hosts_path(state)

    def _run(kh: Path):
        return ssh_client.ssh_run_bash_with_env(
            ip,
            script,
            env,
            key_path=key_path,
            known_hosts=kh,
            port=port,
            timeout=timeout,
        )

    cp = _run(known_hosts)
    if cp.returncode != 0 and ssh_known_hosts.is_host_key_error(cp.stderr):
        warn(f"dynamo ssh {ip}: host key mismatch; refreshing known_hosts and retrying once")
        try:
            known_hosts = _refresh_dynamo_known_hosts([ip], port, state=state)
        except RuntimeError as exc:
            warn(f"dynamo ssh {ip}: known_hosts refresh failed: {exc}")
            return cp
        _save_state(state)
        cp = _run(known_hosts)
    return cp


def _validate_extra_server_args(raw: str, *, context: str) -> None:
    """Validate server args before fan-out to RayJob or Dynamo pods.

    Args:
        raw: Extra server CLI flags string.
        context: Label for error messages.

    Raises:
        ServerArgsRejected: When a denied flag is present.
    """
    validate_server_args(raw, context=context)




def _short_poll(
    *,
    label: str,
    fetch: callable,
    is_ok: callable,
    is_fail: callable,
    interval_s: int,
    timeout_s: int,
    failure_diag: callable | None = None,
    quiet_fetch_error_grace_s: float = 0.0,
    is_quiet_fetch_error: Callable[[BaseException], bool] | None = None,
) -> Any:
    """Run many short polls within one CLI invocation budget; returns the state_obj on success.

    ``fetch()`` returns ``(state_obj, summary_str)``; each poll is logged.
    A ``quiet_fetch_error`` within the grace window logs at INFO (SaFE
    post-create 404 lag). Terminal failure raises
    :class:`WorkloadTerminalFailure` (exit 2); timeout raises
    :class:`TransientFailure` (exit 1, safe to rerun).

    Args:
        label (str): Human-readable label used in log lines.
        fetch (callable): Returns ``(state_obj, summary_str)`` each poll.
        is_ok (callable): Predicate marking a successful terminal state.
        is_fail (callable): Predicate marking a terminal failure state.
        interval_s (int): Seconds to sleep between polls.
        timeout_s (int): Total poll budget before giving up.
        failure_diag (callable | None): Optional ``state_obj -> (diag,
            snapshot)`` used to enrich a terminal failure. Defaults to ``None``.
        quiet_fetch_error_grace_s (float): Window during which a quiet fetch
            error is logged at INFO rather than WARN. Defaults to ``0.0``.
        is_quiet_fetch_error (Callable[[BaseException], bool] | None): Predicate
            classifying a fetch exception as quiet. Defaults to ``None``.

    Returns:
        Any: The ``state_obj`` once ``is_ok`` is satisfied.

    Raises:
        WorkloadTerminalFailure: When ``is_fail`` matches and a
            ``failure_diag`` is supplied.
        RuntimeError: When ``is_fail`` matches without a ``failure_diag``.
        TransientFailure: When the poll budget elapses before a terminal state.
    """
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        fetch_was_quiet = False
        try:
            obj, summary = fetch()
        except Exception as exc:  # noqa: BLE001
            elapsed_mid = time.monotonic() - started
            fetch_was_quiet = (
                quiet_fetch_error_grace_s > 0.0
                and elapsed_mid < quiet_fetch_error_grace_s
                and is_quiet_fetch_error is not None
                and is_quiet_fetch_error(exc)
            )
            if fetch_was_quiet:
                info(
                    f"{label} poll #{attempt} t={elapsed_mid:.0f}s -> "
                    f"GET workload not visible yet (404); retrying "
                    f"(expected within ~{quiet_fetch_error_grace_s:.0f}s after create)"
                )
            else:
                warn(f"{label} poll #{attempt} fetch error: {exc}")
            obj, summary = None, str(exc)

        elapsed = time.monotonic() - started
        if not fetch_was_quiet:
            info(f"{label} poll #{attempt} t={elapsed:.0f}s -> {summary}")

        if obj is not None:
            if is_ok(obj):
                return obj
            if is_fail(obj):
                if failure_diag is not None:
                    diag, snapshot = failure_diag(obj)
                    raise WorkloadTerminalFailure(
                        label=label,
                        phase=str(obj.get("phase") if isinstance(obj, dict) else "?"),
                        diag=diag,
                        snapshot=snapshot,
                    )
                raise RuntimeError(f"{label} terminal failure: {summary}")

        if elapsed >= timeout_s:
            raise TransientFailure(
                f"{label} did not reach a terminal state within {timeout_s}s "
                f"(last: {summary}). Re-run the same subcommand to keep polling."
            )
        time.sleep(interval_s)








# ---------------------------------------------------------------------------
# Subcommand: create-dynamo (Dynamo idle-pod backend)
#
# Mirrors create-rayjob but provisions a SaFE DynamoDeployment with idle
# worker pods (mn-idle.sh) and an SSH control plane instead of a RayJob with
# the Ray Dashboard. The benchmark entry point is the Dynamo frontend
# (:8000), NOT sglang rank-0 :8888.



































def install_geak_on_pods_best_effort() -> int:
    """Best-effort GEAK install on the Dynamo GPU pods (provisioner hook).

    No-op (returns 0) for non-dynamo state. Failures are logged but do not
    abort provisioning — the kernel phase will surface a clear pod-side
    ``geak CLI not found`` error if install genuinely failed.

    Returns:
        int: The install return code, or ``0`` for non-dynamo state / on a
        swallowed error.
    """
    if _load_state().get("backend") != "dynamo":
        return 0
    ns = argparse.Namespace(
        geak_src=None,
        print_logs=False,
        poll_interval=_DEFAULT_POLL_INTERVAL_S,
        poll_timeout=_resolve_poll_timeout_s(),
    )
    try:
        return cmd_install_geak(ns)
    except Exception as exc:  # noqa: BLE001
        warn(f"install-geak skipped: {type(exc).__name__}: {exc}")
        return 0


def install_kernel_tools_on_pods_best_effort() -> int:
    """Provisioner hook: install GEAK on the Dynamo GPU pods.

    No-op for non-dynamo. Returns non-zero only if the install reported a
    hard failure (best-effort; provisioning continues regardless).

    Returns:
        int: Non-zero if the install reported a hard failure, else ``0``.
    """
    return install_geak_on_pods_best_effort()


# ---------------------------------------------------------------------------
# Subcommand: bootstrap
def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Run the BYOI bootstrap script inside the RayJob via Ray Dashboard REST.

    Streams the sandbox-side ``scripts/bootstrap.sh`` into the head pod via a
    heredoc entrypoint (``--script PATH`` overrides with a pod-visible script).

    Args:
        args (argparse.Namespace): Parsed ``bootstrap`` arguments.

    Returns:
        int: ``0`` on success.
    """
    state = _require_state("rayjob_id", "head_pod_ip")

    if args.script:
        # Operator override: assume the path is pod-visible.
        entrypoint = f"bash {args.script}" + (" --force" if args.force else "")
    else:
        bootstrap_sh = _read_pod_script("bootstrap.sh")
        force_arg = " --force" if args.force else ""
        entrypoint = (
            "set -euo pipefail; "
            'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
            "cat > \"$WORK_DIR/bootstrap.sh\" <<'__MN_BOOT_EOF__'\n"
            f"{bootstrap_sh}__MN_BOOT_EOF__\n"
            f'chmod +x "$WORK_DIR/bootstrap.sh"; '
            f'"$WORK_DIR/bootstrap.sh"{force_arg}'
        )

    with _ray_dashboard_client(state) as ray:
        info(f"submitting bootstrap entrypoint: {entrypoint}")
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"submission_id={sub_id}")

        def _fetch():
            """Fetch the bootstrap job and summarize its status.

            Returns:
                tuple: ``(job_dict, summary_str)`` for the poll loop.
            """
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"bootstrap {sub_id}",
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        info(f"bootstrap done: {result.get('status')}")
        if args.print_logs:
            logs = ray.get_job_logs(sub_id)
            print(logs)

    # Persist the bootstrap submission id for later debugging.
    state["last_bootstrap_submission_id"] = sub_id
    _save_state(state)
    return 0


# Subcommand: verify
def cmd_verify(args: argparse.Namespace) -> int:
    """Sanity-check the toolchain bootstrap installed inside the RayJob.

    Args:
        args (argparse.Namespace): Parsed ``verify`` arguments.

    Returns:
        int: ``0`` on success.
    """
    state = _require_state("head_pod_ip")

    # Source the env file, then verify ``ray`` is on PATH.
    script = (
        "set -e; "
        "if [ -f /etc/profile.d/hyperloom-env.sh ]; "
        "then source /etc/profile.d/hyperloom-env.sh; fi; "
        "for bin in ray; do "
        '  echo "-- which $bin --"; '
        '  which "$bin" || { echo "MISSING: $bin" >&2; exit 1; }; '
        "done; "
        "echo OK"
    )
    entrypoint = script  # _wrap_for_dash will wrap as bash; -lc breaks PATH
    with _ray_dashboard_client(state) as ray:
        info("submitting verify entrypoint")
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"submission_id={sub_id}")

        def _fetch():
            """Fetch the verify job and summarize its status.

            Returns:
                tuple: ``(job_dict, summary_str)`` for the poll loop.
            """
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"verify {sub_id}",
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if args.print_logs or str(result.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES:
            logs = ray.get_job_logs(sub_id)
            print(logs)
        info(f"verify done: {result.get('status')}")

    state["last_verify_submission_id"] = sub_id
    _save_state(state)
    return 0


# Subcommand: restart-server
_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _read_pod_script(name: str) -> str:
    """Read a pod-side script from ``multi_node/scripts/`` (embedded into the dashboard entrypoint at submit time).

    Args:
        name (str): The script filename under ``multi_node/scripts/``.

    Returns:
        str: The script's text contents.

    Raises:
        RuntimeError: If the named script is missing.
    """
    p = _SCRIPTS_DIR / name
    if not p.is_file():
        raise RuntimeError(f"missing pod-side script: {p}. Did you trim the multi_node/scripts/ directory?")
    return p.read_text(encoding="utf-8")


def _strip_pod_script_header(body: str) -> str:
    """Drop shebang and ``from __future__ import annotations`` from a pod script."""
    lines = body.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    lines = [ln for ln in lines if ln.strip() != "from __future__ import annotations"]
    return "\n".join(lines).strip()


def _read_bundled_pod_python_script(
    main: str,
    *,
    deps: tuple[str, ...] = ("patch_path_safety.py",),
) -> str:
    """Read a pod Python script with stdlib-only dependencies inlined.

    Dynamo SSH ships a single decoded file per invocation, so dependency modules
    are prepended into one executable script body.

    Args:
        main: Primary script filename under ``multi_node/scripts/``.
        deps: Dependency scripts to prepend (shebangs stripped).

    Returns:
        str: Combined script text with one shebang header.
    """
    chunks = [_strip_pod_script_header(_read_pod_script(dep)) for dep in deps]
    main_body = _strip_pod_script_header(_read_pod_script(main))
    return "from __future__ import annotations\n\n" + "\n\n".join(chunks) + "\n\n" + main_body + "\n"


def _build_restart_entrypoint(
    args: argparse.Namespace,
    pid_file: str,
    log_file: str,
) -> str:
    """Compose the single-node restart entrypoint (heredoc kill_server.sh + launch_server.sh; IR-5 PID-file kill).

    Args:
        args (argparse.Namespace): Parsed ``restart-server`` arguments.
        pid_file (str): PID-file path the kill / launch scripts use.
        log_file (str): Server log path on the head pod.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.

    Raises:
        RuntimeError: For an unsupported framework.
    """
    framework = args.framework.lower()
    if framework not in ("sglang", "vllm"):
        raise RuntimeError(f"unsupported framework: {args.framework!r} (use sglang or vllm)")

    kill_sh = _read_pod_script("kill_server.sh")
    launch_sh = _read_pod_script("launch_server.sh")

    wait_flag = "--no-wait-health" if args.no_wait_health else "--wait-health"

    entrypoint = (
        "set -euo pipefail; "
        f'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
        f"cat > \"$WORK_DIR/kill_server.sh\" <<'__MN_KILL_EOF__'\n"
        f"{kill_sh}__MN_KILL_EOF__\n"
        f"cat > \"$WORK_DIR/launch_server.sh\" <<'__MN_LAUNCH_EOF__'\n"
        f"{launch_sh}__MN_LAUNCH_EOF__\n"
        'chmod +x "$WORK_DIR/kill_server.sh" "$WORK_DIR/launch_server.sh"; '
        f'"$WORK_DIR/kill_server.sh" {pid_file!s}; '
        f'"$WORK_DIR/launch_server.sh" {framework!s} {args.model!s} {args.tp!s} '
        f"{pid_file!s} {log_file!s} {wait_flag} -- {args.extra_args}"
    )
    return entrypoint


# Common entrypoint preamble: sources the bootstrap env file so PATH points
# at /opt/venv/bin (no-op when bootstrap was skipped). See ADDENDUM-13.
_MN_ENTRYPOINT_PREAMBLE = (
    "set -euo pipefail; "
    "if [ -f /etc/profile.d/hyperloom-env.sh ]; then "
    "source /etc/profile.d/hyperloom-env.sh; "
    "fi; "
    'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
)


def _build_kill_single_entrypoint(pid_file: str) -> str:
    """Compose a head-pod entrypoint that runs only kill_server.sh.

    Uses the IR-5 PID-file kill (no ``pkill -f``).

    Args:
        pid_file (str): Path to the PID file the kill script reads.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    kill_sh = _read_pod_script("kill_server.sh")
    return (
        "set -euo pipefail; "
        f'WORK_DIR=/tmp/multi_node_pod_scripts; mkdir -p "$WORK_DIR"; '
        f"cat > \"$WORK_DIR/kill_server.sh\" <<'__MN_KILL_EOF__'\n"
        f"{kill_sh}__MN_KILL_EOF__\n"
        'chmod +x "$WORK_DIR/kill_server.sh"; '
        f'"$WORK_DIR/kill_server.sh" {pid_file!s}'
    )


def _exec_kill_submission(
    state: dict[str, Any],
    entrypoint: str,
    *,
    label: str,
    args: argparse.Namespace,
) -> str:
    """Submit a kill entrypoint via Ray Dashboard and poll to SUCCEEDED.

    Args:
        state: Multi-node state (head IP + dashboard token).
        entrypoint (str): The kill entrypoint shell command to submit.
        label (str): Human-readable label used in log lines and polling.
        args (argparse.Namespace): Parsed CLI args (poll interval/timeout,
            print_logs).

    Returns:
        str: The Ray Dashboard submission id of the kill job.
    """
    with _ray_dashboard_client(state) as ray:
        kill_sub = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"{label} submission_id={kill_sub}")

        def _fetch_kill():
            """Fetch the kill job status for the poll loop.

            Returns:
                tuple[dict, str]: The job dict and a short status message.
            """
            j = ray.get_job(kill_sub)
            return j, f"kill status={j.get('status', '?')}"

        _short_poll(
            label=f"{label} {kill_sub}",
            fetch=_fetch_kill,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if getattr(args, "print_logs", False):
            logs = ray.get_job_logs(kill_sub)
            print(logs)
    return kill_sub


def _build_multinode_kill_entrypoint(pid_dir: str, grace_sec: int = 5) -> str:
    """Compose the head-pod entrypoint that kills every rank's server via heredoc-embedded kill_multinode.py (fans out via ray actors).

    Args:
        pid_dir (str): Directory of per-rank PID files.
        grace_sec (int): Grace period before a hard kill. Defaults to ``5``.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    py = _read_pod_script("kill_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/kill_multinode.py\" <<'__MN_KILL_PY_EOF__'\n"
        f"{py}__MN_KILL_PY_EOF__\n"
        f'python3 "$WORK_DIR/kill_multinode.py" '
        f"--pid-dir {pid_dir!s} --grace-sec {grace_sec}"
    )


def _extract_launcher_summary(launch_logs: str) -> dict:
    """Parse the JSON summary launch_multinode.py writes to stdout (the last balanced ``{...}`` in the interleaved logs); ``{}`` on failure.

    Args:
        launch_logs (str): The interleaved launcher stdout / logs.

    Returns:
        dict: The parsed summary object, or ``{}`` when none can be parsed.
    """
    if not launch_logs:
        return {}
    text = launch_logs.rstrip()
    depth = 0
    end_idx = -1
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            if depth == 0:
                end_idx = i
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                try:
                    candidate = text[i : end_idx + 1]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    # Keep walking; an inner brace may be a repr, not JSON.
                    pass
                end_idx = -1
    return {}


def _build_multinode_launch_entrypoint(
    args: argparse.Namespace,
    nnodes: int,
    pid_dir: str,
    log_dir: str,
) -> str:
    """Compose the head-pod entrypoint that spawns one rank per node via heredoc-embedded launch_multinode.py.

    Killed ranks MUST be cleared before this runs (sequenced by
    cmd_restart_server) or rank 0's old process still holds :8888.

    Args:
        args (argparse.Namespace): Parsed ``restart-server`` arguments.
        nnodes (int): Number of nodes (ranks) to launch.
        pid_dir (str): Directory for per-rank PID files.
        log_dir (str): Directory for per-rank logs.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    py = _read_pod_script("launch_multinode.py")
    wait_flag = "--no-wait-health" if args.no_wait_health else ""
    extra_args = args.extra_args or ""
    try:
        _validate_extra_server_args(extra_args, context="rayjob restart-server --extra-args")
    except ServerArgsRejected as exc:
        raise RuntimeError(str(exc)) from exc
    # Pin SGLANG_TORCH_PROFILER_DIR to a shared-FS path: $HYPERLOOM_MN_PROFILE_TRACE_DIR,
    # else derive from state.json's rayjob_id; empty => skip the flag.
    profiler_dir = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if not profiler_dir:
        _st = _load_state()
        _rid = str(_st.get("rayjob_id") or "").strip()
        if _rid:
            _profiler_path = mn_profile_trace_root() / _rid / "torch_trace"
            # Best-effort mkdir (asymmetric mounts may PermissionError yet work pod-side).
            try:
                _profiler_path.mkdir(parents=True, exist_ok=True)
            except OSError as _exc:
                warn(f"cannot mkdir profile-traces dir {_profiler_path}: {_exc}; pod-side launch will retry the mkdir")
            profiler_dir = str(_profiler_path)
            info(f"profile-traces dir derived from rayjob_id: {profiler_dir}")
    profiler_arg = f"--torch-profiler-dir {shlex.quote(str(profiler_dir))} " if profiler_dir else ""
    # Expert-parallel size; ep <= 1 emits no flag.
    try:
        ep_val = int(getattr(args, "ep", 1) or 1)
    except (TypeError, ValueError):
        ep_val = 1
    ep_arg = f"--ep {ep_val} " if ep_val > 1 else ""
    # Forward PD knobs only when disaggregated (else keep the colocated default).
    pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
    pd_args = ""
    if pd_mode == "disaggregated":
        pn = int(getattr(args, "pd_prefill_nodes", 0) or 0)
        dn = int(getattr(args, "pd_decode_nodes", 0) or 0)
        ptp = int(getattr(args, "pd_prefill_tp", 0) or 0)
        dtp = int(getattr(args, "pd_decode_tp", 0) or 0)
        tb = (getattr(args, "pd_transfer_backend", "") or "").strip()
        ib = (getattr(args, "pd_ib_device", "") or "").strip()
        bp = int(getattr(args, "pd_bootstrap_port", 0) or 0)
        chunks = [
            "--pd-mode disaggregated",
            f"--pd-prefill-nodes {pn}",
            f"--pd-decode-nodes {dn}",
        ]
        if ptp > 0:
            chunks.append(f"--pd-prefill-tp {ptp}")
        if dtp > 0:
            chunks.append(f"--pd-decode-tp {dtp}")
        if tb:
            chunks.append(f"--pd-transfer-backend {tb}")
        if ib:
            chunks.append(f"--pd-ib-device {ib}")
        if bp > 0:
            chunks.append(f"--pd-bootstrap-port {bp}")
        pd_args = " ".join(chunks) + " "
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/launch_multinode.py\" <<'__MN_LAUNCH_PY_EOF__'\n"
        f"{py}__MN_LAUNCH_PY_EOF__\n"
        f'python3 "$WORK_DIR/launch_multinode.py" '
        f"--framework {args.framework!s} --model {args.model!s} "
        f"--tp {args.tp!s} --nnodes {nnodes!s} "
        f"--pid-dir {pid_dir!s} --log-dir {log_dir!s} "
        f"{ep_arg}{profiler_arg}{pd_args}{wait_flag} --extra-args {shlex.quote(str(extra_args))}"
    )


def _build_multinode_router_entrypoint(
    args: argparse.Namespace,
    prefill_url: str,
    decode_url: str,
    pid_dir: str,
    log_dir: str,
) -> str:
    """Compose the head-pod entrypoint that detaches the PD router (rank 0 only, binds 8888) via heredoc-embedded launch_router.py.

    Args:
        args (argparse.Namespace): Parsed ``restart-server`` arguments.
        prefill_url (str): The prefill group's server URL.
        decode_url (str): The decode group's server URL.
        pid_dir (str): Directory for the router PID file.
        log_dir (str): Directory for the router log.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    py = _read_pod_script("launch_router.py")
    public_port = 8888
    pid_file = f"{pid_dir.rstrip('/')}/router.pid"
    log_file = f"{log_dir.rstrip('/')}/router.log"
    vllm_router_cmd = (getattr(args, "pd_vllm_router_cmd", "") or "").strip()
    vrc_arg = f"--vllm-router-cmd {shlex.quote(str(vllm_router_cmd))} " if vllm_router_cmd else ""
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f"cat > \"$WORK_DIR/launch_router.py\" <<'__MN_ROUTER_PY_EOF__'\n"
        f"{py}__MN_ROUTER_PY_EOF__\n"
        f'python3 "$WORK_DIR/launch_router.py" '
        f"--framework {args.framework!s} "
        f"--prefill-url {shlex.quote(str(prefill_url))} --decode-url {shlex.quote(str(decode_url))} "
        f"--public-port {public_port} "
        f"--pid-file {shlex.quote(str(pid_file))} --log-file {shlex.quote(str(log_file))} "
        f"{vrc_arg}"
    )


def _build_multinode_apply_patch_entrypoint(
    target_path: str,
    patch_b64: str,
    backup_dir: str,
    kernel_id: str,
    timeout_sec: int,
) -> str:
    """Compose the head-pod entrypoint that fans out a kernel patch to every pod via heredoc-embedded kernel_patch_multinode.py.

    Args:
        target_path (str): The pod-side file path to patch.
        patch_b64 (str): Base64-encoded unified diff to apply.
        backup_dir (str): Directory where pre-patch backups are written.
        kernel_id (str): Identifier tying the patch to a kernel.
        timeout_sec (int): Per-pod apply timeout in seconds.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    pps = _read_pod_script("patch_path_safety.py")
    py = _read_pod_script("kernel_patch_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f'cat > "$WORK_DIR/patch_path_safety.py" '
        f"<<'__MN_PPATH_EOF__'\n"
        f"{pps}__MN_PPATH_EOF__\n"
        f'cat > "$WORK_DIR/kernel_patch_multinode.py" '
        f"<<'__MN_KPATCH_PY_EOF__'\n"
        f"{py}__MN_KPATCH_PY_EOF__\n"
        f'python3 "$WORK_DIR/kernel_patch_multinode.py" apply '
        f"--target-path {shlex.quote(str(target_path))} "
        f"--patch-b64 {shlex.quote(str(patch_b64))} "
        f"--backup-dir {shlex.quote(str(backup_dir))} "
        f"--kernel-id {shlex.quote(str(kernel_id))} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _build_multinode_revert_patch_entrypoint(
    target_path: str,
    backup_map_json: str,
    timeout_sec: int,
) -> str:
    """Compose the head-pod entrypoint that fans out a revert via heredoc-embedded kernel_patch_multinode.py (``backup_map_json`` from the matching apply).

    Args:
        target_path (str): The pod-side file path to revert.
        backup_map_json (str): JSON map of per-pod backups from the matching
            apply.
        timeout_sec (int): Per-pod revert timeout in seconds.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    pps = _read_pod_script("patch_path_safety.py")
    py = _read_pod_script("kernel_patch_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f'cat > "$WORK_DIR/patch_path_safety.py" '
        f"<<'__MN_PPATH_EOF__'\n"
        f"{pps}__MN_PPATH_EOF__\n"
        f'cat > "$WORK_DIR/kernel_patch_multinode.py" '
        f"<<'__MN_KPATCH_PY_EOF__'\n"
        f"{py}__MN_KPATCH_PY_EOF__\n"
        f'python3 "$WORK_DIR/kernel_patch_multinode.py" revert '
        f"--target-path {shlex.quote(str(target_path))} "
        f"--backup-map-json {shlex.quote(str(backup_map_json))} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _build_multinode_apply_tracelens_patch_entrypoint(
    tracelens_root: str,
    sglang_version_pin: str,
) -> str:
    """Compose the head-pod entrypoint fanning out the TraceLens patch set via heredoc-embedded apply_tracelens_patch_multinode.py.

    Forwards only ``$TRACELENS_ROOT`` (patches are read on the pods'
    wekafs mount); the in-pod script is idempotent.

    Args:
        tracelens_root (str): Pod-visible TraceLens root forwarded as
            ``--tracelens-root``.
        sglang_version_pin (str): Optional SGLang version pin; omitted from the
            command when empty.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    py = _read_pod_script("apply_tracelens_patch_multinode.py")
    pin_arg = ""
    if sglang_version_pin:
        pin_arg = f" --sglang-version-pin {shlex.quote(str(sglang_version_pin))}"
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f'cat > "$WORK_DIR/apply_tracelens_patch_multinode.py" '
        f"<<'__MN_TLPATCH_PY_EOF__'\n"
        f"{py}__MN_TLPATCH_PY_EOF__\n"
        f'python3 "$WORK_DIR/apply_tracelens_patch_multinode.py" '
        f"--tracelens-root {shlex.quote(str(tracelens_root))}"
        f"{pin_arg}"
    )


def _build_multinode_kernel_bench_entrypoint(
    workspace: str,
    bench_command: str,
    files_b64_json: str,
    result_glob: str,
    timeout_sec: int,
) -> str:
    """Compose the head-pod entrypoint running a kernel micro-benchmark on a single GPU node via heredoc-embedded kernel_bench_multinode.py.

    Args:
        workspace (str): Pod-side workspace dir for the benchmark.
        bench_command (str): The benchmark command to run.
        files_b64_json (str): JSON map of base64-encoded files to materialize.
        result_glob (str): Glob matching the result files to collect.
        timeout_sec (int): Benchmark timeout in seconds.

    Returns:
        str: The composed Ray Dashboard entrypoint shell command.
    """
    py = _read_pod_script("kernel_bench_multinode.py")
    return (
        f"{_MN_ENTRYPOINT_PREAMBLE}"
        f'cat > "$WORK_DIR/kernel_bench_multinode.py" '
        f"<<'__MN_KBENCH_PY_EOF__'\n"
        f"{py}__MN_KBENCH_PY_EOF__\n"
        f'python3 "$WORK_DIR/kernel_bench_multinode.py" bench '
        f"--workspace {shlex.quote(str(workspace))} "
        f"--bench-command {shlex.quote(str(bench_command))} "
        f"--files-b64-json {shlex.quote(str(files_b64_json))} "
        f"--result-glob {shlex.quote(str(result_glob))} "
        f"--timeout-sec {int(timeout_sec)}"
    )


def _extract_pod_json(logs: str) -> dict | None:
    """Parse the last top-level JSON document from an interleaved Ray Dashboard job_logs blob (the in-pod scripts emit one).

    Args:
        logs (str): The interleaved Ray Dashboard ``job_logs`` text.

    Returns:
        dict | None: The parsed JSON object, or ``None`` when none can be
        parsed.
    """
    if not logs:
        return None
    text = logs.rstrip()
    end = text.rfind("}")
    if end == -1:
        return None
    # Scan backward for the matching open brace at depth 0.
    depth = 0
    in_str = False
    esc = False
    for i in range(end, -1, -1):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i : end + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _submit_and_collect_pod_json(
    state: dict[str, Any],
    entrypoint: str,
    *,
    label: str,
    poll_interval: int,
    poll_timeout: int,
    runtime_env: dict | None = None,
    success_statuses: frozenset[str] | None = None,
) -> tuple[int, dict | None, str]:
    """Submit ``entrypoint``, poll to terminal, parse the per-pod JSON, and return ``(returncode, parsed_or_None, logs)``.

    Args:
        state: Multi-node state (head IP + dashboard token).
        entrypoint (str): The entrypoint shell command to submit.
        label (str): Human-readable label used in log lines and polling.
        poll_interval (int): Seconds between status polls.
        poll_timeout (int): Overall poll budget in seconds.
        runtime_env: Optional Ray runtime environment payload.
        success_statuses: Parsed JSON ``status`` values treated as success
            (default: ``{"ok"}``).

    Returns:
        tuple[int, dict | None, str]: The job return code, the parsed per-pod
        JSON (or ``None``), and the raw logs.
    """
    with _ray_dashboard_client(state) as ray:
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint), runtime_env=runtime_env)
        info(f"{label} submission_id={sub_id}")

        def _fetch():
            """Fetch the submission status for the poll loop.

            Returns:
                tuple[dict, str]: The job dict and a short status message.
            """
            j = ray.get_job(sub_id)
            return j, f"{label} status={j.get('status', '?')}"

        result = _short_poll(
            label=label,
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=poll_interval,
            timeout_s=poll_timeout,
        )
        status = str(result.get("status", "")).upper()
        logs = ray.get_job_logs(sub_id)
        parsed = _extract_pod_json(logs)
        if status in _TERMINAL_FAIL_STATUSES:
            return EXIT_TRANSIENT, parsed, logs
        if status not in _TERMINAL_OK_STATUSES:
            return EXIT_TRANSIENT, parsed, logs
        # Dashboard SUCCEEDED; the sub-script's own ``status`` says if the fan-out worked.
        if parsed is None:
            return EXIT_TRANSIENT, None, logs
        ok_statuses = success_statuses or frozenset({"ok"})
        sub_status = str(parsed.get("status", "")).lower()
        return (EXIT_OK if sub_status in ok_statuses else EXIT_TRANSIENT), parsed, logs


# Subcommand: apply-patch / revert-patch / kernel-bench (multi-node only)
# Cohesive rayjob/dynamo clusters live in commands/{rayjob,dynamo}.py. Bind
# only the command hooks used below.
from .commands.rayjob import cmd_create_rayjob as cmd_create_rayjob
from .commands.dynamo import (
    cmd_create_dynamo as cmd_create_dynamo,
    _dynamo_restart_server as _dynamo_restart_server,
    _dynamo_kill_inference as _dynamo_kill_inference,
    _dynamo_apply_tracelens_patch as _dynamo_apply_tracelens_patch,
    _dynamo_apply_patch as _dynamo_apply_patch,
    _dynamo_revert_patch as _dynamo_revert_patch,
    _dynamo_kernel_bench as _dynamo_kernel_bench,
    cmd_install_geak as cmd_install_geak,
)


def cmd_apply_patch(args: argparse.Namespace) -> int:
    """Fan out a kernel patch to every pod (head + workers).

    Read the patch file from sandbox, base64-encode it, submit a Ray
    Dashboard entrypoint that runs kernel_patch_multinode.py apply on
    the head pod; that script spawns per-node actors to write the same
    patch to ``--target-path`` on each pod.

    Multi-node only. Single-node falls back to ``apply_kernel_patch.py``
    in the sandbox (no Ray dispatch needed).

    Stdout: the same JSON document kernel_patch_multinode.py emits,
    re-printed verbatim so sandbox-side callers (apply_kernel_patch.py
    multi-node dispatch) can parse it deterministically.

    Args:
        args (argparse.Namespace): Parsed ``apply-patch`` arguments
            (``patch_file``, ``target_path``, ``kernel_id``, ``backup_dir``,
            ``timeout_sec``, poll knobs).

    Returns:
        int: ``EXIT_OK`` on a successful fan-out, ``EXIT_CONFIG_ERROR`` for
        missing state / unreadable patch, ``EXIT_TRANSIENT`` when the JSON
        can't be parsed or the fan-out failed.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_apply_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("apply-patch requires head_pod_ip in state file; run create-rayjob first")
        return EXIT_CONFIG_ERROR

    patch_path = Path(args.patch_file)
    if not patch_path.is_file():
        err(f"patch_file does not exist: {patch_path}")
        return EXIT_CONFIG_ERROR
    try:
        patch_bytes = patch_path.read_bytes()
    except OSError as exc:
        err(f"failed to read patch_file {patch_path}: {exc}")
        return EXIT_CONFIG_ERROR
    patch_b64 = base64.b64encode(patch_bytes).decode("ascii")

    info(
        f"apply-patch: target={args.target_path} kernel_id={args.kernel_id!r} "
        f"backup_dir={args.backup_dir} bytes={len(patch_bytes)}"
    )

    entrypoint = _build_multinode_apply_patch_entrypoint(
        args.target_path,
        patch_b64,
        args.backup_dir,
        args.kernel_id,
        args.timeout_sec,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="apply-patch",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("apply-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_revert_patch(args: argparse.Namespace) -> int:
    """Fan out a kernel patch revert across the pods that originally
    received it. ``--backup-map-json`` is the per-host map returned by
    the matching ``apply-patch`` call; callers MUST pass it through
    unchanged so the right backups are read on each pod.

    Args:
        args (argparse.Namespace): Parsed ``revert-patch`` arguments
            (``target_path``, ``backup_map_json``, ``timeout_sec``, poll
            knobs).

    Returns:
        int: ``EXIT_OK`` on success, ``EXIT_CONFIG_ERROR`` for missing state /
        invalid backup map, ``EXIT_TRANSIENT`` when the JSON can't be parsed.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_revert_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("revert-patch requires head_pod_ip in state file; run create-rayjob first")
        return EXIT_CONFIG_ERROR

    try:
        decoded_map = json.loads(args.backup_map_json or "{}")
    except json.JSONDecodeError as exc:
        err(f"--backup-map-json is not valid JSON: {exc}")
        return EXIT_CONFIG_ERROR
    if not decoded_map:
        err("--backup-map-json must be a non-empty {host: backup_path} object")
        return EXIT_CONFIG_ERROR

    info(f"revert-patch: target={args.target_path} backup_hosts={list(decoded_map.keys())}")

    entrypoint = _build_multinode_revert_patch_entrypoint(
        args.target_path,
        args.backup_map_json,
        args.timeout_sec,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="revert-patch",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("revert-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_apply_tracelens_patch(args: argparse.Namespace) -> int:
    """Fan out the TraceLens SGLang patch set to every pod via apply_tracelens_patch_multinode.py; multi-node only.

    Needed because the sandbox can't ``import sglang`` to run the local
    patcher. Idempotent (already-patched pods return ``status=skipped``).
    Re-prints the script's JSON (``status`` + ``per_pod``) verbatim.

    Args:
        args (argparse.Namespace): Parsed ``apply-tracelens-patch`` arguments
            (``tracelens_root``, ``sglang_version_pin``, poll knobs).

    Returns:
        int: ``EXIT_OK`` when the patch was applied or skipped,
        ``EXIT_CONFIG_ERROR`` for missing state / TraceLens root,
        ``EXIT_TRANSIENT`` otherwise.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_apply_tracelens_patch(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("apply-tracelens-patch requires head_pod_ip in state file; run create-rayjob first")
        return EXIT_CONFIG_ERROR

    tracelens_root = args.tracelens_root or os.environ.get("TRACELENS_ROOT", "").strip()
    if not tracelens_root:
        err(
            "apply-tracelens-patch requires --tracelens-root or "
            "$TRACELENS_ROOT to point at the TraceLens checkout "
            "(must be visible from every pod, typically a wekafs path)"
        )
        return EXIT_CONFIG_ERROR

    info(f"apply-tracelens-patch: tracelens_root={tracelens_root!r} version_pin={args.sglang_version_pin!r}")

    entrypoint = _build_multinode_apply_tracelens_patch_entrypoint(
        tracelens_root,
        args.sglang_version_pin or "",
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="apply-tracelens-patch",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("apply-tracelens-patch: could not parse per-pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    # The helper keys success on ``status=="ok"``; this patcher uses
    # ``applied``/``skipped``, so derive the exit code ourselves.
    if parsed.get("status") in ("applied", "skipped"):
        return EXIT_OK
    return EXIT_TRANSIENT


def cmd_kernel_bench(args: argparse.Namespace) -> int:
    """Run a kernel micro-benchmark on a GPU-bearing pod.

    Stages an optional bundle of helper files into ``--workspace``,
    invokes ``--bench-command`` under that workspace with GPU
    acceleration, and reads back result artifacts matching
    ``--result-glob``.

    The sandbox calls this when ``is_multi_node()`` is True and the
    kernel-agent micro-benchmark step would otherwise try to compile +
    run on the sandbox (which lacks GPUs in multi-node mode).

    Args:
        args (argparse.Namespace): Parsed ``kernel-bench`` arguments
            (``workspace``, ``bench_command``, ``files_b64_json``,
            ``result_glob``, ``timeout_sec``, poll knobs).

    Returns:
        int: ``EXIT_OK`` on a successful benchmark, ``EXIT_CONFIG_ERROR`` for
        missing state / invalid files JSON, ``EXIT_TRANSIENT`` when the JSON
        can't be parsed.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_kernel_bench(args)
    state = _load_state()
    head_ip = (state.get("head_pod_ip") or "").strip()
    if not head_ip:
        err("kernel-bench requires head_pod_ip in state file; run create-rayjob first")
        return EXIT_CONFIG_ERROR

    # Validate the optional files-b64 JSON before hitting the dashboard.
    if args.files_b64_json:
        try:
            json.loads(args.files_b64_json)
        except json.JSONDecodeError as exc:
            err(f"--files-b64-json is not valid JSON: {exc}")
            return EXIT_CONFIG_ERROR

    info(f"kernel-bench: workspace={args.workspace} cmd={args.bench_command!r} result_glob={args.result_glob}")

    entrypoint = _build_multinode_kernel_bench_entrypoint(
        args.workspace,
        args.bench_command,
        args.files_b64_json or "{}",
        args.result_glob,
        args.timeout_sec,
    )
    rc, parsed, logs = _submit_and_collect_pod_json(
        state,
        entrypoint,
        label="kernel-bench",
        poll_interval=args.poll_interval,
        poll_timeout=_poll_timeout_from_args(args),
    )
    if parsed is None:
        err("kernel-bench: could not parse pod JSON from dashboard logs")
        if args.print_logs:
            print(logs)
        return EXIT_TRANSIENT
    print(json.dumps(parsed, indent=2, sort_keys=True))
    return rc


def cmd_restart_server(args: argparse.Namespace) -> int:
    """Kill any prior vllm/sglang server and launch a new one.

    Two paths, picked from state.json's ``nodes`` field (written by
    create-rayjob):

    * ``nodes <= 1`` (single-pod) — submit a bash entrypoint that runs
      kill_server.sh + launch_server.sh on the head pod. Same as the
      pre-multinode behaviour; nothing changes for single-pod sessions.
    * ``nodes >= 2`` (multi-pod) — submit a Python entrypoint
      that uses ray actors to fan out kill_multinode.py + launch_multinode.py
      across every node, wiring sglang/vllm with --nnodes / --node-rank /
      --dist-init-addr per upstream multi-node docs. The agent runs ONE
      restart-server invocation; the driver inside the pod handles the rest.

    Dynamo backend (state.backend == 'dynamo') routes to the SSH fan-out path
    instead of the Ray Dashboard; the RayJob path below is byte-for-byte
    unchanged for non-dynamo state.

    Args:
        args (argparse.Namespace): Parsed ``restart-server`` arguments
            (``framework``, ``model``, ``tp``, ``ep``, PD knobs, ``pid_file``,
            ``log_file``, poll knobs).

    Returns:
        int: ``EXIT_OK`` once the server is healthy, ``EXIT_CONFIG_ERROR`` for
        missing state, or a transient exit code on launch / poll failure.
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_restart_server(args)
    try:
        _validate_extra_server_args(
            getattr(args, "extra_args", "") or "",
            context="restart-server --extra-args",
        )
    except ServerArgsRejected as exc:
        err(str(exc))
        return EXIT_CONFIG_ERROR
    state = _require_state("head_pod_ip")
    nnodes = int(state.get("nodes") or 1)

    if nnodes >= 2:
        # Multi-node: dir-based PID/log layout (one file per rank).
        pid_dir = args.pid_file or state.get("last_server_pid_dir") or "/tmp/multi_node_pids"
        log_dir = args.log_file or state.get("last_server_log_dir") or "/tmp/multi_node_logs"
        info(f"restart-server (multi-node): framework={args.framework} model={args.model} tp={args.tp} nnodes={nnodes}")

        kill_ep = _build_multinode_kill_entrypoint(pid_dir)
        launch_ep = _build_multinode_launch_entrypoint(args, nnodes, pid_dir, log_dir)

        # Resume fast path: if the prior launch had identical
        # framework/model/tp/ep/pd_mode and is still RUNNING, skip KILL+LAUNCH
        # and resume polling (large MoE boots outlast a 110s retry window).
        # Disable with MULTI_NODE_RESTART_RESUME_RUNNING=0.
        launch_sub: str = ""
        resume_enabled = os.environ.get("MULTI_NODE_RESTART_RESUME_RUNNING", "1").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        prev_sub = str(state.get("last_restart_submission_id") or "").strip()
        # ``last_restart_extra_args`` is normalized at write time; do the
        # same to the live args here so whitespace differences don't make
        # an otherwise-identical restart miss the resume fast path.
        # CRITICAL: extra_args carries every backend / params variant flag
        # (--attention-backend, --cuda-graph-max-bs, --moe-runner-backend,
        # etc.); a resume that ignores it leaves sglang running with the
        # PREVIOUS variant's args, so every benchmark measurement after
        # the first variant reflects stale flags instead of the round's
        # intended config.
        prev_match = bool(prev_sub) and (
            str(state.get("last_restart_framework") or "") == str(args.framework)
            and str(state.get("last_restart_model") or "") == str(args.model)
            and int(state.get("last_restart_tp") or 0) == int(args.tp)
            and int(state.get("last_restart_ep") or 1) == int(getattr(args, "ep", 1) or 1)
            and str(state.get("last_restart_pd_mode") or "colocated")
            == (getattr(args, "pd_mode", "") or "colocated").lower()
            and _normalize_extra_args(state.get("last_restart_extra_args"))
            == _normalize_extra_args(getattr(args, "extra_args", ""))
        )
        if resume_enabled and prev_match:
            _prev_status = ""
            try:
                with _ray_dashboard_client(state) as _probe:
                    _job = _probe.get_job(prev_sub)
                    _prev_status = str(_job.get("status", "")).upper()
            except Exception as _exc:  # noqa: BLE001
                info(f"resume probe failed: {_exc!r}; falling back to KILL+LAUNCH")
            if _prev_status == "RUNNING":
                info(
                    f"resume: reusing launch_sub={prev_sub} "
                    f"(framework={args.framework} model={args.model} "
                    f"tp={args.tp} ep={getattr(args, 'ep', 1)}) "
                    f"— skipping KILL+LAUNCH, just polling"
                )
                launch_sub = prev_sub
            elif _prev_status in _TERMINAL_OK_STATUSES:
                info(
                    f"resume: prior launch_sub={prev_sub} already SUCCEEDED; skipping KILL+LAUNCH, treating as healthy"
                )
                launch_sub = prev_sub

        kill_sub = ""
        if not launch_sub:
            kill_sub = _exec_kill_submission(
                state,
                kill_ep,
                label="restart kill",
                args=args,
            )

        with _ray_dashboard_client(state) as ray:
            # Phase B: launch new (skipped above when resuming an
            # existing RUNNING launch). Driver returns once every rank
            # spawned its launcher; rank 0 /health probe is best-effort
            # (driver internal, see launch_multinode.py).
            if not launch_sub:
                launch_sub = ray.submit_job(_wrap_for_dash(launch_ep))
                info(f"launch submission_id={launch_sub} (driver waits for actors, then returns; servers detached)")

            # EARLY checkpoint: persist the launch identity + config NOW,
            # before the (potentially long) _short_poll. Without this,
            # the next 110s poll timeout raises TransientFailure and
            # cmd_restart_server returns early — leaving state.json
            # untouched. The retry loop's next invocation can't see
            # last_restart_submission_id, falls into KILL+LAUNCH again,
            # and the server bootstrap is reset from zero. Persisting
            # here lets the resume-running-launch fast path above
            # actually fire on retry.
            state["last_server_pid_dir"] = pid_dir
            state["last_server_log_dir"] = log_dir
            if kill_sub:
                state["last_kill_submission_id"] = kill_sub
            state["last_restart_submission_id"] = launch_sub
            state["last_restart_framework"] = args.framework
            state["last_restart_model"] = args.model
            state["last_restart_tp"] = int(args.tp)
            state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
            state["last_restart_pd_mode"] = (getattr(args, "pd_mode", "") or "colocated").lower()
            state["last_restart_extra_args"] = _normalize_extra_args(getattr(args, "extra_args", ""))
            _save_state(state)

            def _fetch_launch():
                """Fetch the launch job status for the poll loop.

                Returns:
                    tuple[dict, str]: The job dict and a short status message.
                """
                j = ray.get_job(launch_sub)
                return j, f"launch status={j.get('status', '?')}"

            result = _short_poll(
                label=f"launch {launch_sub}",
                fetch=_fetch_launch,
                is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
                is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
                interval_s=args.poll_interval,
                timeout_s=_poll_timeout_from_args(args),
            )
            launch_status = str(result.get("status", "")).upper()
            if args.print_logs or launch_status in _TERMINAL_FAIL_STATUSES:
                logs = ray.get_job_logs(launch_sub)
                print(logs)

            # Fail-fast on a terminal launch status so the caller raises
            # ServerRestartFailed in seconds instead of burning the 1800s health wait.
            if launch_status in _TERMINAL_FAIL_STATUSES:
                info(
                    f"ERROR launch driver terminal status={launch_status}; "
                    f"multi-node restart failed (sub_id={launch_sub}). "
                    f"Check stderr above for MULTI_NODE_FAILURE_SNAPSHOT."
                )
                return 1

            # PD disaggregated: read the launcher's prefill/decode URLs and
            # submit a separate router entrypoint binding 8888 (fatal if it fails).
            pd_mode = (getattr(args, "pd_mode", "") or "colocated").lower()
            router_sub = ""
            router_state: dict = {}
            if pd_mode == "disaggregated":
                launch_logs = ray.get_job_logs(launch_sub)
                router_state = _extract_launcher_summary(launch_logs)
                prefill_url = str(router_state.get("pd_prefill_url") or "").strip()
                decode_url = str(router_state.get("pd_decode_url") or "").strip()
                if not prefill_url or not decode_url:
                    info(
                        "ERROR PD launcher summary missing prefill/decode URL; "
                        "cannot start router. Inspect launch logs above."
                    )
                    return 1

                info(f"PD launcher summary: prefill={prefill_url} decode={decode_url}; submitting router entrypoint")
                router_ep = _build_multinode_router_entrypoint(
                    args,
                    prefill_url,
                    decode_url,
                    pid_dir,
                    log_dir,
                )
                router_sub = ray.submit_job(_wrap_for_dash(router_ep))
                info(
                    f"router submission_id={router_sub} "
                    f"(detaches launch_router.py; dashboard exits when router is alive)"
                )

                def _fetch_router():
                    """Fetch the router job status for the poll loop.

                    Returns:
                        tuple[dict, str]: The job dict and a short status message.
                    """
                    j = ray.get_job(router_sub)
                    return j, f"router status={j.get('status', '?')}"

                router_result = _short_poll(
                    label=f"router {router_sub}",
                    fetch=_fetch_router,
                    is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
                    is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
                    interval_s=args.poll_interval,
                    timeout_s=_poll_timeout_from_args(args),
                )
                router_status = str(router_result.get("status", "")).upper()
                if args.print_logs or router_status in _TERMINAL_FAIL_STATUSES:
                    print(ray.get_job_logs(router_sub))
                if router_status in _TERMINAL_FAIL_STATUSES:
                    info(f"ERROR router job ended {router_status}; PD restart failed")
                    return 1

        state["last_server_pid_dir"] = pid_dir
        state["last_server_log_dir"] = log_dir
        state["last_kill_submission_id"] = kill_sub
        state["last_restart_submission_id"] = launch_sub
        state["last_restart_framework"] = args.framework
        state["last_restart_model"] = args.model
        state["last_restart_tp"] = args.tp
        state["last_restart_ep"] = int(getattr(args, "ep", 1) or 1)
        # Persist PD state so later invocations can fall back when a PD flag is omitted.
        pd_mode_persist = (getattr(args, "pd_mode", "") or "colocated").lower()
        state["last_restart_pd_mode"] = pd_mode_persist
        if pd_mode_persist == "disaggregated":
            state["last_restart_pd_prefill_nodes"] = int(
                getattr(args, "pd_prefill_nodes", 0) or 0,
            )
            state["last_restart_pd_decode_nodes"] = int(
                getattr(args, "pd_decode_nodes", 0) or 0,
            )
            state["last_restart_pd_prefill_tp"] = int(
                getattr(args, "pd_prefill_tp", 0) or 0,
            )
            state["last_restart_pd_decode_tp"] = int(
                getattr(args, "pd_decode_tp", 0) or 0,
            )
            state["last_restart_pd_transfer_backend"] = getattr(args, "pd_transfer_backend", "") or ""
            state["last_restart_pd_ib_device"] = getattr(args, "pd_ib_device", "") or ""
            state["last_router_submission_id"] = router_sub
            state["pd_prefill_url"] = router_state.get("pd_prefill_url", "")
            state["pd_decode_url"] = router_state.get("pd_decode_url", "")
        _save_state(state)
        info("multi-node servers launched; benchmarks should target $service_url")
        return 0

    # Single-node path. PD is meaningless on one pod; fail loudly rather than drop the flags.
    if (getattr(args, "pd_mode", "") or "colocated").lower() == "disaggregated":
        info(
            "ERROR --pd-mode disaggregated is not supported in single-node "
            "mode (state.json says nodes=1). PD requires >=2 nodes so "
            "prefill and decode can run on disjoint pods. Re-create the "
            "RayJob with `multi_node create-rayjob --nodes >=2` before "
            "asking for PD."
        )
        return 2
    pid_file = args.pid_file or state.get("last_server_pid_file") or "/tmp/multi_node_server.pid"
    log_file = args.log_file or "/tmp/multi_node_server.log"
    entrypoint = _build_restart_entrypoint(args, pid_file, log_file)

    with _ray_dashboard_client(state) as ray:
        info(f"restart-server (single-node): framework={args.framework} model={args.model} tp={args.tp}")
        sub_id = ray.submit_job(_wrap_for_dash(entrypoint))
        info(f"submission_id={sub_id} (entrypoint will exit after launch; server keeps running via nohup)")

        def _fetch():
            """Fetch the restart job status for the poll loop.

            Returns:
                tuple[dict, str]: The job dict and a short status message.
            """
            j = ray.get_job(sub_id)
            return j, f"status={j.get('status', '?')}"

        result = _short_poll(
            label=f"restart {sub_id}",
            fetch=_fetch,
            is_ok=lambda j: str(j.get("status", "")).upper() in _TERMINAL_OK_STATUSES,
            is_fail=lambda j: str(j.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES,
            interval_s=args.poll_interval,
            timeout_s=_poll_timeout_from_args(args),
        )
        if args.print_logs or str(result.get("status", "")).upper() in _TERMINAL_FAIL_STATUSES:
            logs = ray.get_job_logs(sub_id)
            print(logs)

    state["last_server_pid_file"] = pid_file
    state["last_server_log_file"] = log_file
    state["last_restart_submission_id"] = sub_id
    state["last_restart_framework"] = args.framework
    state["last_restart_model"] = args.model
    state["last_restart_tp"] = args.tp
    _save_state(state)
    info("server launch acknowledged; subsequent benchmark jobs can target $service_url")
    return 0


def cmd_kill_inference(args: argparse.Namespace) -> int:
    """Kill vllm/sglang on the RayJob without starting replacements.

    Frees the GPUs (single- or multi-node aware) without launching a new
    server. Used before kernel-agent GPU tasks.

    Args:
        args (argparse.Namespace): Parsed CLI args (pid_file, polling,
            print_logs).

    Returns:
        int: A process exit code (``EXIT_OK`` on success, otherwise a
            config error code).
    """
    if _load_state().get("backend") == "dynamo":
        return _dynamo_kill_inference(args)
    state = _require_state("head_pod_ip")
    nnodes = int(state.get("nodes") or 1)

    if nnodes >= 2:
        pid_dir = args.pid_file or state.get("last_server_pid_dir") or "/tmp/multi_node_pids"
        info(f"kill-inference (multi-node): pid_dir={pid_dir}")
        kill_ep = _build_multinode_kill_entrypoint(pid_dir)
        kill_sub = _exec_kill_submission(
            state,
            kill_ep,
            label="kill-inference",
            args=args,
        )
        state["last_kill_submission_id"] = kill_sub
        _save_state(state)
        return 0

    pid_file = args.pid_file or state.get("last_server_pid_file") or "/tmp/multi_node_server.pid"
    info(f"kill-inference (single-node): pid_file={pid_file}")
    entrypoint = _build_kill_single_entrypoint(pid_file)
    kill_sub = _exec_kill_submission(
        state,
        entrypoint,
        label="kill-inference",
        args=args,
    )
    state["last_kill_submission_id"] = kill_sub
    _save_state(state)
    return 0


def kill_inference_for_kernel_agent_best_effort() -> None:
    """Best-effort inference teardown before kernel-agent Ray GPU tasks (swallows errors)."""
    ns = argparse.Namespace(
        pid_file=None,
        print_logs=False,
        poll_interval=_DEFAULT_POLL_INTERVAL_S,
        poll_timeout=_resolve_poll_timeout_s(),
    )
    try:
        cmd_kill_inference(ns)
    except Exception as exc:  # noqa: BLE001
        warn(f"kill-inference skipped: {type(exc).__name__}: {exc}")


# Subcommand: stop-rayjob
def cmd_stop_rayjob(args: argparse.Namespace) -> int:
    """Stop the RayJob via SaFE REST. Idempotent.

    Optionally deletes the workload and/or clears the local state file.
    Does nothing (returns success) when no workload id is known.

    Args:
        args (argparse.Namespace): Parsed CLI args (workload_id, delete,
            clear_state).

    Returns:
        int: A process exit code (always ``EXIT_OK`` here).
    """
    state = _load_state()
    wid = args.workload_id or state.get("rayjob_id")
    if not wid:
        warn("no rayjob_id in state file and --workload-id not provided; nothing to stop")
        return 0
    with safe_client.from_env() as safe:
        info(f"stopping workload {wid} (delete={args.delete})")
        if args.delete:
            safe.delete_workload(wid)
        else:
            safe.stop_workload(wid)
    if args.clear_state:
        path = _state_file()
        try:
            path.unlink(missing_ok=True)
            info(f"cleared {path}")
        except OSError as exc:
            warn(f"could not unlink {path}: {exc}")
    return 0


# argparse
def _add_common_poll_flags(p: argparse.ArgumentParser) -> None:
    """Register the shared ``--poll-interval`` / ``--poll-timeout`` flags.

    Args:
        p (argparse.ArgumentParser): The (sub)parser to add the flags to.
    """
    p.add_argument(
        "--poll-interval",
        type=int,
        default=_DEFAULT_POLL_INTERVAL_S,
        help=f"seconds between polls (default {_DEFAULT_POLL_INTERVAL_S})",
    )
    p.add_argument(
        "--poll-timeout",
        type=int,
        default=None,
        help=(
            "max seconds before this CLI invocation gives up "
            f"(default: HYPERLOOM_MN_POLL_TIMEOUT_S env or {_DEFAULT_POLL_TIMEOUT_S}; "
            f"use 1800 for MoE JIT cold-start). Re-run the same subcommand to "
            "keep polling."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with every subcommand.

    Registers the ``create-rayjob`` / ``restart-server`` / ``kill-
    inference`` / ``apply-patch`` / ``revert-patch`` / ``apply-tracelens-
    patch`` / ``kernel-bench`` / ``stop-rayjob`` subparsers and their
    flags.

    Returns:
        argparse.ArgumentParser: The fully-configured argument parser.
    """
    p = argparse.ArgumentParser(
        prog="python3 -m hyperloom.inference_optimizer.multi_node",
        description=(
            "Manage a session-scoped SaFE RayJob for multi-node inference "
            "optimization. State persists in /tmp/multi_node_state.json."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # create-rayjob
    sp = sub.add_parser("create-rayjob", help="create the RayJob via SaFE REST")
    sp.add_argument(
        "--workspace",
        default=None,
        help="SaFE workspace id. Resolution: --workspace > $SAFE_WORKSPACE env. Bails fast if neither is set.",
    )
    sp.add_argument("--image", required=True, help="container image for both head and worker pods")
    sp.add_argument("--nodes", type=int, required=True, help="total node count (>=1)")
    sp.add_argument(
        "--gpus-per-node",
        type=int,
        default=8,
        help="GPUs per pod (default 8 — full MI300X / MI355X node). "
        "Override only if the user prompt explicitly asks for a "
        "smaller per-pod GPU count.",
    )
    sp.add_argument(
        "--cpus-per-node",
        type=int,
        default=96,
        help="default 96 — matches a full MI300X / MI355X pod. Override only if the user prompt asks for less.",
    )
    sp.add_argument(
        "--mem-per-node",
        type=int,
        default=1024,
        help="GiB per pod. default 1024 — matches a full MI300X / "
        "MI355X pod. Override only if the user prompt asks for less.",
    )
    sp.add_argument("--ephemeral-per-node", type=int, default=400, help="GiB per pod. default 400.")
    sp.add_argument(
        "--display-name",
        default=None,
        help="Optional human-readable RayJob name (shows up in SaFE UI). "
        "Resolution: $DISPLAY_NAME env > --display-name > "
        "auto-generated multi_node_<unix-ts>.",
    )
    sp.add_argument("--description", default=None)
    sp.add_argument(
        "--owner-id",
        default=None,
        help="ownerId for SaFE cascading cleanup. Resolution: --owner-id > "
        "$WORKLOAD_ID (sandbox workload id). When set, SaFE GCs the "
        "RayJob when the owner workload stops (safety net for missed "
        "`stop-rayjob`).",
    )
    sp.add_argument("--extra-env", action="append", default=[], help="K=V (repeatable); merged AFTER credential fanout")
    sp.add_argument(
        "--extra-label",
        action="append",
        default=[],
        help="K=V (repeatable); reserved primus-safe.* prefixes are stripped",
    )
    sp.add_argument("--no-wait", action="store_true", help="don't poll for Running; just create and exit")
    sp.add_argument(
        "--recreate",
        action="store_true",
        help="force creating a fresh workload even if state.json "
        "already has a live rayjob_id. Default behaviour is "
        "to REUSE the prior workload (idempotent retries).",
    )
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_create_rayjob)

    # create-dynamo (Dynamo idle-pod backend)
    sp = sub.add_parser(
        "create-dynamo",
        help="create an idle multi-node DynamoDeployment (SSH control plane); "
        "benchmark entry point is the Dynamo frontend :8000",
    )
    sp.add_argument("--workspace", default=None, help="SaFE workspace id (--workspace > $SAFE_WORKSPACE)")
    sp.add_argument("--image", required=True, help="dynamo image WITH the sshd layer (mn-idle.sh present)")
    sp.add_argument("--nodes", type=int, required=True, help="worker LWS node count (>=1); worker.replica == nodes")
    sp.add_argument("--gpus-per-node", type=int, default=8)
    sp.add_argument("--cpus-per-node", type=int, default=96)
    sp.add_argument("--mem-per-node", type=int, default=1024, help="GiB per worker pod")
    sp.add_argument("--ephemeral-per-node", type=int, default=400, help="GiB per worker pod")
    sp.add_argument("--shared-mem-per-node", type=int, default=200, help="GiB /dev/shm per worker pod (sharedMemory)")
    sp.add_argument("--backend-framework", default="sglang", choices=("sglang", "vllm", "trtllm"))
    sp.add_argument("--kv-transfer-backend", default="nixl", choices=("nixl", "mori", "mooncake"))
    sp.add_argument(
        "--ssh-port",
        type=int,
        default=ssh_client.DEFAULT_SSH_PORT,
        help=f"pod sshd port (default {ssh_client.DEFAULT_SSH_PORT}; not 22 to avoid hostNetwork collision)",
    )
    sp.add_argument(
        "--pd-mode",
        choices=("aggregated", "disaggregated"),
        default="aggregated",
        help="aggregated [frontend,worker] (default) or disaggregated [frontend,prefill,decode]",
    )
    sp.add_argument("--pd-prefill-nodes", type=int, default=0, help="prefill role replica (disaggregated only)")
    sp.add_argument("--pd-decode-nodes", type=int, default=0, help="decode role replica (disaggregated only)")
    sp.add_argument(
        "--pd-prefill-tp", type=int, default=0, help="prefill TP; a role spans nodes (LWS) when tp > gpus-per-node"
    )
    sp.add_argument(
        "--pd-decode-tp", type=int, default=0, help="decode TP; a role spans nodes (LWS) when tp > gpus-per-node"
    )
    sp.add_argument("--display-name", default=None)
    sp.add_argument("--description", default=None)
    sp.add_argument("--owner-id", default=None)
    sp.add_argument("--extra-env", action="append", default=[], help="K=V (repeatable); merged AFTER credential fanout")
    sp.add_argument("--extra-label", action="append", default=[])
    sp.add_argument("--no-wait", action="store_true")
    sp.add_argument("--recreate", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_create_dynamo)

    # bootstrap
    sp = sub.add_parser(
        "bootstrap",
        help="install Hyperloom RayJob toolchain inside the RayJob via Ray Dashboard REST. "
        "Default: uses multi_node/scripts/bootstrap.sh from this checkout.",
    )
    sp.add_argument(
        "--script",
        default=None,
        help="optional override: absolute path to a bootstrap.sh "
        "ALREADY VISIBLE inside the RayJob pod (e.g. baked into "
        "the image). When omitted, the bundled script is streamed in.",
    )
    sp.add_argument("--force", action="store_true", help="re-run bootstrap even if the marker file says it's done")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_bootstrap)

    # verify
    sp = sub.add_parser("verify", help="confirm ray is on PATH inside the RayJob")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_verify)

    # restart-server
    sp = sub.add_parser("restart-server", help="kill prior vllm/sglang server and start a new one")
    sp.add_argument("--framework", required=True, choices=("sglang", "vllm"))
    sp.add_argument("--model", required=True, help="model path or HF id")
    sp.add_argument("--tp", type=int, required=True)
    sp.add_argument(
        "--ep",
        type=int,
        default=1,
        help="expert-parallel size for MoE inference. 1 (default) keeps "
        "experts sharded by TP. >=2 enables EP: sglang gets "
        "`--enable-ep-moe --ep-size N`, vllm gets "
        "`--enable-expert-parallel`. EP > TP is rejected by the "
        "orchestrator helper before this CLI is invoked.",
    )
    # Prefill-Decode disaggregation. colocated (default) keeps the
    # legacy single-server-group behaviour. disaggregated splits the
    # cluster into prefill + decode groups, launches sglang_router /
    # vllm proxy on the head pod, and binds the public 8888 port at
    # the router (so magpie's BENCHMARK_BASE_URL stays unchanged).
    sp.add_argument(
        "--pd-mode",
        choices=("colocated", "disaggregated"),
        default="colocated",
        help="PD disaggregation mode (default colocated).",
    )
    sp.add_argument(
        "--pd-prefill-nodes",
        type=int,
        default=0,
        help="number of prefill nodes (disaggregated only); pn+dn==nnodes",
    )
    sp.add_argument(
        "--pd-decode-nodes",
        type=int,
        default=0,
        help="number of decode nodes (disaggregated only)",
    )
    sp.add_argument(
        "--pd-prefill-tp",
        type=int,
        default=0,
        help="TP for prefill group (disaggregated only); default = --tp",
    )
    sp.add_argument(
        "--pd-decode-tp",
        type=int,
        default=0,
        help="TP for decode group (disaggregated only); default = --tp",
    )
    # Per-role EP / extra server args (disaggregated only). The InferenceX
    # disagg recipes give prefill and decode DIFFERENT MoE topologies
    # (e.g. prefill EP1 no-DP vs decode EP8 DP-attn), so a single shared
    # --ep / --extra-args cannot express them. These per-role knobs default
    # to 0 / "" (fall back to the shared --ep / --extra-args, preserving the
    # legacy behaviour) and read $PD_*_EP / $PD_*_EXTRA_ARGS env as defaults.
    sp.add_argument(
        "--pd-prefill-ep",
        type=int,
        default=int(os.environ.get("PD_PREFILL_EP", "0") or 0),
        help="EP for prefill group (disaggregated only); 0 = use --ep",
    )
    sp.add_argument(
        "--pd-decode-ep",
        type=int,
        default=int(os.environ.get("PD_DECODE_EP", "0") or 0),
        help="EP for decode group (disaggregated only); 0 = use --ep",
    )
    sp.add_argument(
        "--pd-prefill-extra-args",
        default=os.environ.get("PD_PREFILL_EXTRA_ARGS", ""),
        help="extra sglang args appended to the PREFILL group only "
        "(merged after the shared --extra-args); disaggregated only",
    )
    sp.add_argument(
        "--pd-decode-extra-args",
        default=os.environ.get("PD_DECODE_EXTRA_ARGS", ""),
        help="extra sglang args appended to the DECODE group only "
        "(merged after the shared --extra-args); disaggregated only",
    )
    sp.add_argument(
        "--pd-transfer-backend",
        default="",
        help="sglang: mooncake|nixl ; vllm: NixlConnector|P2pNcclConnector|"
        "MooncakeConnector|LMCacheConnectorV1; empty = framework default",
    )
    sp.add_argument(
        "--pd-ib-device",
        default="",
        help="comma-separated IB/RoCE device list (e.g. mlx5_0,mlx5_1). Empty = read $NCCL_IB_HCA from RayJob pod env.",
    )
    sp.add_argument(
        "--pd-bootstrap-port",
        type=int,
        default=8998,
        help="sglang PD bootstrap rendezvous port (default 8998)",
    )
    sp.add_argument(
        "--pd-vllm-router-cmd",
        default="",
        help="vllm-only override for router cmdline; supports {prefill}/{decode}/{port} placeholders",
    )
    sp.add_argument("--extra-args", default="", help="extra CLI args appended verbatim to the framework launch command")
    sp.add_argument(
        "--pid-file", default=None, help="PID file path inside the head pod; defaults to /tmp/multi_node_server.pid"
    )
    sp.add_argument(
        "--log-file", default=None, help="server log path inside the head pod; defaults to /tmp/multi_node_server.log"
    )
    sp.add_argument(
        "--no-wait-health",
        action="store_true",
        help="exit the dashboard job immediately after launch instead of waiting for /health",
    )
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_restart_server)

    sp = sub.add_parser(
        "kill-inference",
        help="kill vllm/sglang on the RayJob only (no new server; frees GPU for kernel-agent)",
    )
    sp.add_argument(
        "--pid-file",
        default=None,
        help="single-node: head-pod PID file path; multi-node: pid-dir path "
        "(same override semantics as restart-server --pid-file)",
    )
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_kill_inference)

    # stop-rayjob
    sp = sub.add_parser("stop-rayjob", help="stop the RayJob via SaFE REST")
    sp.add_argument("--workload-id", default=None, help="override the workload id from state file")
    sp.add_argument("--delete", action="store_true", help="hard delete instead of soft stop (default: stop)")
    sp.add_argument("--clear-state", action="store_true", help="remove /tmp/multi_node_state.json on success")
    sp.set_defaults(func=cmd_stop_rayjob)

    # apply-patch (multi-node only)
    sp = sub.add_parser(
        "apply-patch",
        help="fan-out a kernel patch to every pod (head + workers); multi-node only",
    )
    sp.add_argument(
        "--patch-file",
        required=True,
        help="path to the patch source on sandbox filesystem; contents will be base64-encoded into the dashboard entrypoint",
    )
    sp.add_argument(
        "--target-path",
        required=True,
        help="absolute file path on each pod to overwrite (e.g. /sgl-workspace/aiter/aiter/ops/gemm.py)",
    )
    sp.add_argument(
        "--backup-dir",
        required=True,
        help="directory on each pod where the pre-patch original is saved (e.g. /var/kernel_patch_backups)",
    )
    sp.add_argument("--kernel-id", default="", help="optional id used to construct backup filename")
    sp.add_argument("--timeout-sec", type=int, default=120, help="per-actor timeout (default 120s)")
    sp.add_argument("--print-logs", action="store_true", help="dump full dashboard job_logs on parse failure")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_apply_patch)

    # revert-patch (multi-node only)
    sp = sub.add_parser(
        "revert-patch",
        help="fan-out a kernel patch revert; multi-node only",
    )
    sp.add_argument("--target-path", required=True)
    sp.add_argument(
        "--backup-map-json",
        required=True,
        help="JSON object {hostname: backup_path} from the matching apply-patch result",
    )
    sp.add_argument("--timeout-sec", type=int, default=60)
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_revert_patch)

    # apply-tracelens-patch (multi-node only)
    sp = sub.add_parser(
        "apply-tracelens-patch",
        help=(
            "fan-out the TraceLens SGLang patch set to every pod; "
            "multi-node only. Idempotent: skips pods already patched."
        ),
    )
    sp.add_argument(
        "--tracelens-root",
        default=None,
        help=(
            "absolute path to public TraceLens checkout (must be visible "
            "from every pod, typically /wekafs/...). Defaults to "
            "$TRACELENS_ROOT."
        ),
    )
    sp.add_argument(
        "--sglang-version-pin",
        default=None,
        help=("advisory pin (e.g. '0.5.11'); logged on mismatch with the sglang installed in the pod. Optional."),
    )
    sp.add_argument("--print-logs", action="store_true", help="dump full dashboard job_logs on parse failure")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_apply_tracelens_patch)

    # kernel-bench (multi-node only)
    sp = sub.add_parser(
        "kernel-bench",
        help="run a kernel micro-benchmark on a GPU-bearing pod; multi-node only",
    )
    sp.add_argument("--workspace", required=True, help="absolute dir on pod that will be CWD for the bench")
    sp.add_argument("--bench-command", required=True, help="shell command to invoke (passed to 'bash -lc')")
    sp.add_argument(
        "--files-b64-json",
        default="{}",
        help="JSON {rel_path: base64_content} of helper files to stage into workspace before the bench",
    )
    sp.add_argument(
        "--result-glob",
        default="*.json",
        help="glob (relative to workspace) of result artifacts to read back after the bench",
    )
    sp.add_argument("--timeout-sec", type=int, default=600, help="hard timeout for the bench command (default 600s)")
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_kernel_bench)

    # install-geak (dynamo only)
    sp = sub.add_parser(
        "install-geak",
        help="install the GEAK CLI on every Dynamo GPU pod over SSH "
        "(idempotent; pip-installs the shared-FS GEAK checkout); dynamo only",
    )
    sp.add_argument(
        "--geak-src",
        default=None,
        help="GEAK source dir on the shared mount (default: "
        "$HYPERLOOM_GEAK_SRC > $HYPERLOOM_ROOT/geak > "
        "$USER_DATA_PATH/runtime/geak)",
    )
    sp.add_argument("--print-logs", action="store_true")
    _add_common_poll_flags(sp)
    sp.set_defaults(func=cmd_install_geak)

    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint with stable exit codes: 0 ok, 1 transient (retryable), 2 terminal workload failure (do not retry), 3 config error, 130 SIGINT.

    Args:
        argv (list[str] | None): Argument vector to parse; ``None`` uses
            ``sys.argv``.

    Returns:
        int: The process exit code mapped from the subcommand result or the
        caught exception type.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        err("interrupted")
        return EXIT_INTERRUPT
    except WorkloadTerminalFailure as wtf:
        err(str(wtf))
        # Single-line JSON marker the controller can grep on stderr.
        err("MULTI_NODE_FAILURE_SNAPSHOT=" + json.dumps(wtf.snapshot, default=str))
        return EXIT_TERMINAL_FAILURE
    except TransientFailure as tf:
        err(str(tf))
        return EXIT_TRANSIENT
    except safe_client.SafeApiError as sae:
        # 4xx (incl. 401/403) = caller-fixable -> config error (exit 3); 5xx = retryable (exit 1).
        err(str(sae))
        if sae.status is not None and 400 <= sae.status < 500:
            return EXIT_CONFIG_ERROR
        return EXIT_TRANSIENT
    except (RuntimeError, ValueError) as exc:
        # Caller-fixable input/state errors -> config error; else transient.
        msg = str(exc)
        if any(
            s in msg
            for s in (
                "is required",
                "Missing required environment variable",
                "missing required keys",
                "unsupported framework",
            )
        ):
            err(f"{type(exc).__name__}: {exc}")
            return EXIT_CONFIG_ERROR
        err(f"{type(exc).__name__}: {exc}")
        return EXIT_TRANSIENT
    except Exception as exc:  # noqa: BLE001
        err(f"{type(exc).__name__}: {exc}")
        return EXIT_TRANSIENT


if __name__ == "__main__":
    sys.exit(main())
