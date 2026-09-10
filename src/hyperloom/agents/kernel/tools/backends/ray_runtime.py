# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ray cluster lifecycle helpers for kernel-agent backends."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Minimum soft RLIMIT_NOFILE the Ray raylet needs to stay up.
DEFAULT_MIN_NOFILE = 65536
DEFAULT_RAY_STATUS_TIMEOUT_SEC = 5.0
DEFAULT_RAY_STOP_TIMEOUT_SEC = 30.0

# Custom Ray resource declared on the single-node head so serving-family work (serving / benchmark / profile /
# gpu_research) can hold a whole-machine ``serving_slot`` as the authoritative physical mutex.
RAY_SERVING_SLOT = "serving_slot"
_HEAD_CUSTOM_RESOURCES = {RAY_SERVING_SLOT: 1}


def _resources_start_args() -> list[str]:
    """Return the ``ray start`` argv for the head node's custom resources."""
    return ["--resources", json.dumps(_HEAD_CUSTOM_RESOURCES)]


# --- Local-head port isolation (spur host-network co-location) --------------- Many optimizer sessions can be
# co-scheduled on ONE compute node (SLURM packs sub-node ``--gpus`` requests), and spur runs the container on the host
# network stack (the bridge has no egress).
_HL_RAY_HEAD_PORT_ENV = "HL_RAY_HEAD_PORT"


def _free_tcp_port() -> int:
    """Reserve and return a currently-free loopback TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _isolated_head_port_args() -> Tuple[int, list[str]]:
    """Return ``(gcs_port, extra_start_args)`` bound to FREE probed ports."""
    override = os.environ.get(_HL_RAY_HEAD_PORT_ENV, "").strip()
    port = int(override) if override.isdigit() else 0
    gcs_port = port if 1 <= port <= 65535 else _free_tcp_port()
    return gcs_port, [
        f"--dashboard-port={_free_tcp_port()}",
        f"--ray-client-server-port={_free_tcp_port()}",
    ]


def _fd_limit_warn(msg: str) -> None:
    """Emit an fd-limit warning to stderr with a stable prefix."""
    print(f"[kernel-agent WARN] {msg}", file=sys.stderr)


def _min_nofile_target() -> int:
    """Return the target soft RLIMIT_NOFILE value."""
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
    """Raise this process's RLIMIT_NOFILE soft limit before Ray starts."""
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
    """Check whether a Ray cluster is currently reachable."""
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
    """Ensure a Ray cluster is reachable, starting a head node if needed."""
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
    if not ray_status_ok():
        raise RuntimeError(f"ray start exited 0 but cluster is not reachable; see {log_path}")


def _is_ray_version_mismatch(text: str) -> bool:
    """Detect Ray's version-mismatch banner in captured output."""
    return "version mismatch" in (text or "").lower()


def force_restart_local_cluster(
    num_gpus: Optional[int] = None,
    log_path: Optional[Path] = None,
) -> None:
    """Tear down any reachable Ray cluster and start a fresh local head."""
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
    # Gateway auth headers travel with their endpoint, so a worker that gets the URL and key but not the header is
    # rejected by a header-authenticated gateway (an AMD APIM subscription key, for one).
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
    # GEAK/Forge harness contract: patched candidate dir the generated harness prepends to sys.path.
    "GEAK_WORK_DIR",
    # e2e optimizer runner path + repo root so a Ray worker can locate interface/run_e2e.py and the e2e_workflow/
    # checkout.
    "GEAK_ROOT",
    "GEAK_E2E_RUNNER",
    "GEAK_CLAUDE_EFFORT",
    "GEAK_CLAUDE_MODEL",
    "GEAK_E2E_TIMEOUT_S",
    # Scoring/profiler/run knobs read by GEAK itself; stripped at the Ray boundary without this allowlist entry.
    "GEAK_SCORE_TARGET",
    "GEAK_SKIP_PROFILE",
    "GEAK_MAX_BENCHMARK_SHAPES",
    "GEAK_RUN_MODE",
)


def safe_runtime_env() -> dict:
    """Build a Ray ``runtime_env`` from the allowlisted environment keys."""
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    # Each side's aliases come from that side's own credentials.
    openai_key = env.get("OPENAI_API_KEY")
    if openai_key:
        env.setdefault("LLM_API_KEY", openai_key)
        env.setdefault("AMD_LLM_API_KEY", openai_key)
        env.setdefault("LLM_GATEWAY_KEY", openai_key)
    # CLAUDE_CODE_OAUTH_TOKEN is forwarded verbatim, never mirrored into these: either key var switches the Claude CLI
    # out of subscription mode.
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
    """Initialize ray while suppressing the connect banner on stdout."""
    import contextlib
    import io
    import ray

    runtime_env = safe_runtime_env()

    def _init(address: str) -> None:
        """Call ``ray.init`` with stdout suppressed and standard options."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ray.init(
                address=address,
                ignore_reinit_error=True,
                log_to_driver=False,
                logging_level="error",
                runtime_env=runtime_env,
            )

    try:
        _init(os.environ.get("RAY_ADDRESS", "auto"))
    except Exception as exc:  # noqa: BLE001
        if not _is_ray_version_mismatch(str(exc)):
            raise
        # Foreign cluster: replace with a local head, then retry exactly once.
        try:
            ray.shutdown()
        except Exception:  # noqa: BLE001
            pass
        force_restart_local_cluster(num_gpus=num_gpus, log_path=log_path)
        _init("auto")
    return runtime_env
