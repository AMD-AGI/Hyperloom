# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ray cluster lifecycle helpers for kernel-agent backends.

Conventions:
- Prefer connecting to an existing cluster (RAY_ADDRESS=auto by default).
- Only `ray start --head` when no cluster is reachable.
- Never set HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES on the driver and never
  forward them via runtime_env: Ray sets them on workers itself.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Minimum soft RLIMIT_NOFILE the Ray raylet needs to stay up. Override via
# RAY_MIN_NOFILE.
DEFAULT_MIN_NOFILE = 65536
DEFAULT_RAY_STATUS_TIMEOUT_SEC = 5.0
DEFAULT_RAY_STOP_TIMEOUT_SEC = 30.0

# Custom Ray resource declared on the single-node head so serving-family work
# (serving / benchmark / profile / gpu_research) can hold a whole-machine
# ``serving_slot`` as the authoritative physical mutex.
# Capacity 1 => at most one serving-family task holds the node
# at a time; GPU specialists request ``num_gpus`` only (serving-disjoint) and do
# not take the slot. Declared here (rather than only in the orchestrator) so
# whichever caller starts the local head first — kernel-agent or orchestrator —
# declares it; a tiny unused resource is harmless to GEAK. Only single-node
# local heads are affected: multi-node connects to an external cluster and this
# ``ray start`` path is skipped.
RAY_SERVING_SLOT = "serving_slot"
_HEAD_CUSTOM_RESOURCES = {RAY_SERVING_SLOT: 1}


def _resources_start_args() -> list[str]:
    """Return the ``ray start`` argv for the head node's custom resources.

    Returns:
        ``["--resources", "<json>"]`` declaring :data:`_HEAD_CUSTOM_RESOURCES`.
    """
    return ["--resources", json.dumps(_HEAD_CUSTOM_RESOURCES)]


# --- Local-head port isolation (spur host-network co-location) ---------------
# Many optimizer sessions can be co-scheduled on ONE compute node (SLURM packs
# sub-node ``--gpus`` requests), and spur runs the container on the host network
# stack (the bridge has no egress). Only the HOST NETWORK is shared between
# co-located containers: each keeps a PRIVATE filesystem (no ``/tmp`` bind mount)
# and a PRIVATE PID namespace (no ``--pid=host``). So the ONLY thing that
# collides is Ray's FIXED default host ports (GCS 6379, dashboard 8265, client
# 10001): the later head connects to the earlier head's GCS over 127.0.0.1:6379
# and aborts with a session-name mismatch (``node._write_cluster_info_to_kv``),
# hanging the kernel agent and failing every serving-lease "cluster ensure".
#
# Fix: bind each head to FREE, probed ports instead of the fixed defaults. No
# cross-process coordination is needed -- Ray records the chosen GCS address in
# the container-private ``/tmp/ray/ray_current_cluster``, so the serving lease's
# later ``ensure()`` discovers it via ``ray status`` / ``ray.init(address=
# "auto")`` without knowing the port. We deliberately do NOT pass ``--temp-dir``
# (that reintroduces Ray issue #55244, where ``address="auto"`` still looks in
# the default ``/tmp/ray`` rather than the custom dir); the default dir is
# already container-private here. ``HL_RAY_HEAD_PORT`` pins the GCS port for
# operators/debugging.
_HL_RAY_HEAD_PORT_ENV = "HL_RAY_HEAD_PORT"


def _free_tcp_port() -> int:
    """Reserve and return a currently-free loopback TCP port.

    Binds an ephemeral socket, reads the OS-assigned port, and releases it.
    There is an inherent (tiny) TOCTOU window before ``ray start`` rebinds it;
    on a host-network node with per-session ports this is far less likely than
    the guaranteed 6379 collision it replaces.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _isolated_head_port_args() -> Tuple[int, list[str]]:
    """Return ``(gcs_port, extra_start_args)`` bound to FREE probed ports.

    Isolates the GCS / dashboard / Ray-client ports so co-located sessions never
    share Ray's fixed defaults. ``HL_RAY_HEAD_PORT`` pins the GCS port (dashboard
    and client are still probed to avoid their own collisions).
    """
    port_override = os.environ.get(_HL_RAY_HEAD_PORT_ENV, "").strip()
    gcs_port = int(port_override) if port_override.isdigit() else _free_tcp_port()
    return gcs_port, [
        f"--dashboard-port={_free_tcp_port()}",
        f"--ray-client-server-port={_free_tcp_port()}",
    ]


def _fd_limit_warn(msg: str) -> None:
    """Emit an fd-limit warning to stderr with a stable prefix.

    Args:
        msg: The warning message body.
    """
    print(f"[kernel-agent WARN] {msg}", file=sys.stderr)


def _min_nofile_target() -> int:
    """Return the target soft RLIMIT_NOFILE value.

    Returns:
        The positive integer from ``RAY_MIN_NOFILE`` when set, otherwise
        ``DEFAULT_MIN_NOFILE``.
    """
    raw = os.environ.get("RAY_MIN_NOFILE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MIN_NOFILE


def _positive_float_env(name: str, default: float) -> float:
    """Return a positive float env override, else ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _ray_status_timeout_sec() -> float:
    """Return the ``ray status`` probe timeout in seconds."""
    return _positive_float_env("HYPERLOOM_RAY_STATUS_TIMEOUT_SEC", DEFAULT_RAY_STATUS_TIMEOUT_SEC)


def _ray_stop_timeout_sec() -> float:
    """Return the ``ray stop --force`` timeout in seconds."""
    return _positive_float_env("HYPERLOOM_RAY_STOP_TIMEOUT_SEC", DEFAULT_RAY_STOP_TIMEOUT_SEC)


def ensure_fd_limit(
    min_soft: Optional[int] = None,
    log_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Raise this process's RLIMIT_NOFILE soft limit before Ray starts.

    The child ``ray start`` process inherits this process's limits, so the
    raylet's open-files ceiling is whatever we set here. We raise the soft
    limit to ``min(min_soft, hard)``. Raising the soft limit up to the hard
    cap needs no privileges; lifting the hard cap does (CAP_SYS_RESOURCE),
    so when the hard cap is itself below ``min_soft`` we raise soft as high
    as allowed and warn — only ``docker run --ulimit nofile=...`` at
    container launch can lift the hard cap in an unprivileged container.

    Args:
        min_soft: Target soft limit; defaults to the configured target.
        log_path: Optional path to append a lifecycle log line.

    Returns:
        The ``(soft, hard)`` limit in effect after the call.
    """
    if min_soft is None:
        min_soft = _min_nofile_target()
    inf = resource.RLIM_INFINITY
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # RLIM_INFINITY (-1) means "unlimited"; don't treat it as a tiny number.
    if soft == inf or soft >= min_soft:
        return soft, hard
    target = min_soft if hard == inf else min(min_soft, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft = target
    except (ValueError, OSError) as exc:  # pragma: no cover - defensive
        _fd_limit_warn(
            f"could not raise RLIMIT_NOFILE soft limit to {target} "
            f"(soft={soft}, hard={hard}): {exc}; Ray raylet may be unstable "
            f"(issue #433). Launch the container with --ulimit nofile=1048576."
        )
        return soft, hard
    if hard != inf and hard < min_soft:
        _fd_limit_warn(
            f"RLIMIT_NOFILE hard cap {hard} is below the raylet target "
            f"{min_soft}; raised soft to {soft} but this may still be too low "
            f"(issue #433). Launch the container with --ulimit nofile=1048576 "
            f"(>= {min_soft})."
        )
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"[fd_limit] RLIMIT_NOFILE soft raised to {soft} (hard={hard}) for raylet stability (issue #433)\n"
                )
        except OSError:  # pragma: no cover - logging must never break startup
            pass
    return soft, hard


def ray_status_ok() -> bool:
    """Check whether a Ray cluster is currently reachable.

    Runs ``ray status`` with output suppressed and inspects the exit
    code.

    Returns:
        bool: True if ``ray status`` exits 0 (a cluster is reachable),
            False otherwise.
    """
    try:
        proc = subprocess.run(
            ["ray", "status"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=_ray_status_timeout_sec(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _stop_ray_force(log_path: Optional[Path] = None, *, reason: str = "") -> None:
    """Run ``ray stop --force`` with a bounded timeout; never raises."""
    cmd = ["ray", "stop", "--force"]
    timeout = _ray_stop_timeout_sec()
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                if reason:
                    log.write(f"{reason}\n")
                log.write(f"$ {' '.join(cmd)}\n")
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def ensure_ray_cluster(num_gpus: Optional[int] = None, log_path: Optional[Path] = None) -> None:
    """Ensure a Ray cluster is reachable, starting a head node if needed.

    Args:
        num_gpus: Optional GPU count to pass to ``ray start --head``.
        log_path: Optional path to append ``ray start`` output.

    Raises:
        RuntimeError: If starting the Ray head node fails.
    """
    if ray_status_ok():
        return
    _stop_ray_force(log_path=log_path, reason="Clearing stale Ray discovery state before starting a local head")
    ensure_fd_limit(log_path=log_path)
    gcs_port, iso_args = _isolated_head_port_args()
    # Dashboard bound to loopback: keeps the unauthenticated Ray Jobs endpoint off the pod network.
    cmd = ["ray", "start", "--head", f"--port={gcs_port}", "--dashboard-host=127.0.0.1"]
    if num_gpus is not None:
        cmd.append(f"--num-gpus={num_gpus}")
    cmd.extend(iso_args)
    # serving_slot: whole-machine mutex so serving-family tasks serialise GPU access.
    cmd.extend(_resources_start_args())
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            log.write(f"\n[ray_start_exit_code] {proc.returncode}\n")
    else:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to start Ray; see {log_path}")


def _is_ray_version_mismatch(text: str) -> bool:
    """Detect Ray's version-mismatch banner in captured output.

    Args:
        text: The captured error or output text.

    Returns:
        ``True`` if the text contains the stable version-mismatch banner.
    """
    return "version mismatch" in (text or "").lower()


def force_restart_local_cluster(
    num_gpus: Optional[int] = None,
    log_path: Optional[Path] = None,
) -> None:
    """Tear down any reachable Ray cluster and start a fresh local head.

    The fresh head runs under this interpreter, recovering from a
    stale/foreign cluster whose version mismatch otherwise mislabels as a
    "compile failed" REVERT; this also clears raylet zombies.

    Args:
        num_gpus: Optional GPU count for the fresh head node.
        log_path: Optional path to append restart output.

    Raises:
        RuntimeError: If the fresh head node fails to start.
    """
    ensure_fd_limit(log_path=log_path)
    _stop_ray_force(log_path=log_path, reason="Stopping foreign cluster before version-mismatch recovery")
    gcs_port, iso_args = _isolated_head_port_args()
    start_cmd = ["ray", "start", "--head", f"--port={gcs_port}", "--dashboard-host=127.0.0.1"]
    if num_gpus is not None:
        start_cmd.append(f"--num-gpus={num_gpus}")
    start_cmd.extend(iso_args)
    start_cmd.extend(_resources_start_args())
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(start_cmd)}\n")
            proc = subprocess.run(start_cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            log.write(f"\n[ray_restart_exit_code] {proc.returncode}\n")
    else:
        proc = subprocess.run(start_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to restart local Ray after version mismatch; see {log_path}")


# Env vars safe to forward to Ray workers; excludes *_VISIBLE_DEVICES (Ray-owned; forcing them triggers set_visible_accelerator_ids IndexError on ROCm).
SAFE_ENV_KEYS = (
    "PATH",
    "HOME",
    "LD_LIBRARY_PATH",
    "HYPERLOOM_KERNEL_AGENT_ROOT",
    "KERNEL_AGENT_ROOT",
    # Single artefact root others default under.
    "USER_DATA_PATH",
    "HYPERLOOM_RUNTIME_DIR",
    "KERNEL_AGENT_ENV",
    "MAGPIE_PATH",
    "INFERENCEX_PATH",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    # Gateway auth headers travel with their endpoint, so a worker that gets the
    # URL and key but not the header is rejected by a header-authenticated
    # gateway (an AMD APIM subscription key, for one).
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CUSTOM_HEADERS",
    "AMD_API_KEY",
    "AMD_LLM_API_KEY",
    "LLM_GATEWAY_KEY",
    "LLM_API_KEY",
    "LLM_API_BASE",
    "LLM_PROXY_API_KEY",
    "LLM_PROXY_BASE_URL",
    # GEAK LLM connection (e2e runner reads these).
    "GEAK_API_KEY",
    "GEAK_BASE_URL",
    # GEAK/Forge harness contract: patched candidate dir the generated harness
    # prepends to sys.path.
    "GEAK_WORK_DIR",
    # e2e optimizer runner path + repo root so a Ray worker can locate
    # interface/run_e2e.py and the e2e_workflow/ checkout.
    "GEAK_ROOT",
    "GEAK_E2E_RUNNER",
    "GEAK_CLAUDE_EFFORT",
    "GEAK_CLAUDE_MODEL",
    # Forge-fusion model overrides (mirrors GEAK_CLAUDE_MODEL for each backend).
    "FORGE_CLAUDE_MODEL",
    "FORGE_CODEX_MODEL",
    "GEAK_E2E_TIMEOUT_S",
    # Scoring/profiler/run knobs read by GEAK itself; stripped at the Ray
    # boundary without this allowlist entry.
    "GEAK_SCORE_TARGET",
    "GEAK_SKIP_PROFILE",
    "GEAK_MAX_BENCHMARK_SHAPES",
    "GEAK_RUN_MODE",
)


def safe_runtime_env() -> dict:
    """Build a Ray ``runtime_env`` from the allowlisted environment keys.

    Copies only the keys in :data:`SAFE_ENV_KEYS` from the current
    environment, then fills sensible fallbacks (e.g. deriving the
    per-provider API keys and base URLs from ``OPENAI_API_KEY`` /
    ``ANTHROPIC_API_KEY`` / ``OPENAI_BASE_URL``). GPU-visibility variables are
    deliberately excluded so Ray manages device assignment itself.

    Returns:
        dict: A ``{"env_vars": {...}}`` mapping suitable for passing as
            Ray's ``runtime_env``.
    """
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    # Each side's aliases come from that side's own credentials. GEAK_API_KEY /
    # GEAK_BASE_URL are never derived: GEAK runs on the Anthropic side via
    # GEAK_CLAUDE_MODEL + ANTHROPIC_*, so an OpenAI-side value could not start it.
    # They are forwarded verbatim when an operator sets them.
    openai_key = env.get("OPENAI_API_KEY")
    if openai_key:
        env.setdefault("LLM_API_KEY", openai_key)
        env.setdefault("AMD_LLM_API_KEY", openai_key)
        env.setdefault("LLM_GATEWAY_KEY", openai_key)
    # CLAUDE_CODE_OAUTH_TOKEN is forwarded verbatim, never mirrored into these:
    # either key var switches the Claude CLI out of subscription mode.
    anthropic_key = env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
    if anthropic_key:
        env.setdefault("ANTHROPIC_API_KEY", anthropic_key)
        env.setdefault("ANTHROPIC_AUTH_TOKEN", anthropic_key)
    openai_url = env.get("OPENAI_BASE_URL")
    if openai_url:
        env.setdefault("LLM_API_BASE", openai_url)
    if "AMD_LLM_API_KEY" not in env and "AMD_API_KEY" in env:
        env["AMD_LLM_API_KEY"] = env["AMD_API_KEY"]
    return {"env_vars": env}


def quiet_ray_init(num_gpus: Optional[int] = None, log_path: Optional[Path] = None):
    """Initialize ray while suppressing the connect banner on stdout.

    On a "Version mismatch" RuntimeError (foreign cluster under a different
    Python/Ray), tear the foreign cluster down, bring up a fresh local head
    under this interpreter, and retry ``ray.init`` once.

    Args:
        num_gpus: Optional GPU count forwarded to a restart, if needed.
        log_path: Optional path to audit Ray lifecycle actions.
    """
    import contextlib
    import io
    import ray

    runtime_env = safe_runtime_env()

    def _init() -> None:
        """Call ``ray.init`` with stdout suppressed and standard options."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ray.init(
                address=os.environ.get("RAY_ADDRESS", "auto"),
                ignore_reinit_error=True,
                log_to_driver=False,
                logging_level="error",
                runtime_env=runtime_env,
            )

    try:
        _init()
    except Exception as exc:  # noqa: BLE001
        if not _is_ray_version_mismatch(str(exc)):
            raise
        # Foreign cluster: replace with a local head, then retry exactly once.
        try:
            ray.shutdown()
        except Exception:  # noqa: BLE001
            pass
        force_restart_local_cluster(num_gpus=num_gpus, log_path=log_path)
        _init()
    return runtime_env
