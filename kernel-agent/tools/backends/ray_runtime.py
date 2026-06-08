# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Ray cluster lifecycle helpers for kernel-agent backends.

Conventions:
- Prefer connecting to an existing cluster (RAY_ADDRESS=auto by default).
- Only `ray start --head` when no cluster is reachable.
- Never set HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES on the driver and never
  forward them via runtime_env: Ray sets them on workers itself.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def ray_status_ok() -> bool:
    """Check whether a Ray cluster is currently reachable.

    Runs ``ray status`` with output suppressed and inspects the exit
    code.

    Returns:
        bool: True if ``ray status`` exits 0 (a cluster is reachable),
            False otherwise.
    """
    proc = subprocess.run(
        ["ray", "status"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def ensure_ray_cluster(num_gpus: Optional[int] = None, log_path: Optional[Path] = None) -> bool:
    """Ensure a Ray cluster is reachable, starting a head node if needed.

    If no cluster is reachable, runs ``ray start --head`` (optionally
    pinning the GPU count and tee-ing output to a log file).

    Args:
        num_gpus (Optional[int]): When set, passed as ``--num-gpus`` to
            the new head node. Ignored when a cluster already exists.
        log_path (Optional[Path]): When set, the start command and its
            output are appended here; the parent directory is created.

    Returns:
        bool: True if this call started Ray (so the caller may stop it
            later); False if Ray was already running.

    Raises:
        RuntimeError: If the ``ray start`` command exits non-zero.
    """
    if ray_status_ok():
        return False
    cmd = ["ray", "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0"]
    if num_gpus is not None:
        cmd.append(f"--num-gpus={num_gpus}")
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
    return True


def stop_ray_if_owned(started: bool, log_path: Optional[Path] = None) -> None:
    """Stop Ray only if this process started it.

    A no-op when ``started`` is False, so callers can pair this with
    :func:`ensure_ray_cluster` without tracking ownership themselves.

    Args:
        started (bool): The return value from :func:`ensure_ray_cluster`;
            True means this process owns the cluster and should stop it.
        log_path (Optional[Path]): When set, ``ray stop --force`` output
            is appended here.
    """
    if not started:
        return
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Stopping Ray started by kernel-agent\n")
            subprocess.run(["ray", "stop", "--force"], stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        subprocess.run(["ray", "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)


def _is_ray_version_mismatch(text: str) -> bool:
    """True when Ray refused the job because the reachable cluster was
    started under a different Python / Ray than this process (issue #432).

    Ray raises a RuntimeError whose message reads e.g.::

        Version mismatch: The cluster was started with:
            Ray: 2.44.1
            Python: 3.10.12
        This process on node ... was started with:
            Ray: 2.44.1
            Python: 3.12.13

    Matching on the stable ``Version mismatch`` banner keeps this robust to
    the surrounding Ray/Python version numbers.
    """
    return "version mismatch" in (text or "").lower()


def force_restart_local_cluster(
    num_gpus: Optional[int] = None, log_path: Optional[Path] = None,
) -> None:
    """Tear down any reachable Ray cluster and start a fresh local head
    under THIS interpreter.

    Recovers from a stale/foreign cluster started under a different Python
    (issue #432): reusing it makes ``ray.init`` fail in ~0.8s with a
    "Version mismatch" RuntimeError that otherwise surfaces as a mislabeled
    "compile failed" REVERT. ``ray stop --force`` also clears raylet
    zombies, so this doubles as wedged-cluster recovery.

    Raises RuntimeError if the fresh head fails to start.
    """
    stop_cmd = ["ray", "stop", "--force"]
    start_cmd = ["ray", "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0"]
    if num_gpus is not None:
        start_cmd.append(f"--num-gpus={num_gpus}")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(stop_cmd)}  # issue #432 version-mismatch recovery\n")
            subprocess.run(stop_cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            log.write(f"$ {' '.join(start_cmd)}\n")
            proc = subprocess.run(start_cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            log.write(f"\n[ray_restart_exit_code] {proc.returncode}\n")
    else:
        subprocess.run(stop_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
        proc = subprocess.run(start_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to restart local Ray after version mismatch; see {log_path}"
        )


# Env vars safe to forward to Ray workers. Notice we do NOT include
# HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES; those are
# Ray's responsibility and forcing them from the driver triggers
# `set_visible_accelerator_ids` IndexError on ROCm.
SAFE_ENV_KEYS = (
    "PATH", "HOME", "LD_LIBRARY_PATH",
    "HYPERLOOM_KERNEL_AGENT_ROOT", "KERNEL_AGENT_ROOT",
    # USER_DATA_PATH is the single artefact root; HYPERLOOM_RUNTIME_DIR /
    # KERNEL_AGENT_ENV / MAGPIE_DIR / INFERENCEX_PATH all default under it.
    # WORKSPACE_ROOT / WORKSPACE_PATH / AGENT_WORKSPACE_ROOT were retired
    # during the "all artefacts under USER_DATA_PATH" migration — drop them
    # from the propagate list so we don't accidentally leak stale values
    # into Ray workers.
    "USER_DATA_PATH", "HYPERLOOM_RUNTIME_DIR", "KERNEL_AGENT_ENV",
    "MAGPIE_DIR", "INFERENCEX_PATH",
    "SAFE_API_KEY",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "OOB_API_KEY", "OOB_BASE_URL", "OOB_LOCAL", "OOB_HOME",
    "CURSOR_API_KEY", "CURSOR_DEFAULT_MODEL",
    "AMD_API_KEY", "AMD_LLM_API_KEY", "LLM_GATEWAY_KEY",
    "LLM_API_KEY", "LLM_API_BASE", "LLM_PROXY_API_KEY", "LLM_PROXY_BASE_URL",
    "GEAK_CONFIG", "GEAK_MODEL_NAME", "GEAK_API_KEY", "GEAK_BASE_URL",
    "GEAK_WORK_DIR",
    "GEAK_MEMORY_STORE_PATH", "GEAK_SAVE_TO_KNOWLEDGE_BASE",
    "GEAK_MEMORY_MIN_SPEEDUP", "GEAK_CROSS_SESSION_MEMORY_URL",
    "GEAK_MEMORY_API_KEY", "GEAK_USE_KNOWLEDGE_BASE",
    "GEAK_MEMORY_DISABLE", "GEAK_MEMORY_NO_CROSS_SESSION",
    "MSWEA_MODEL_NAME",
)


def safe_runtime_env() -> dict:
    """Build a Ray ``runtime_env`` from the allowlisted environment keys.

    Copies only the keys in :data:`SAFE_ENV_KEYS` from the current
    environment, then fills sensible fallbacks (e.g. deriving the
    per-provider API keys and base URLs from ``SAFE_API_KEY`` /
    ``OPENAI_BASE_URL``). GPU-visibility variables are deliberately
    excluded so Ray manages device assignment itself.

    Returns:
        dict: A ``{"env_vars": {...}}`` mapping suitable for passing as
            Ray's ``runtime_env``.
    """
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    if "SAFE_API_KEY" in env:
        env.setdefault("OPENAI_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("ANTHROPIC_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("ANTHROPIC_AUTH_TOKEN", env["SAFE_API_KEY"])
        env.setdefault("OOB_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("GEAK_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("LLM_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("AMD_LLM_API_KEY", env["SAFE_API_KEY"])
        env.setdefault("LLM_GATEWAY_KEY", env["SAFE_API_KEY"])
    if "OPENAI_BASE_URL" in env:
        env.setdefault("ANTHROPIC_BASE_URL", env["OPENAI_BASE_URL"])
        env.setdefault("OOB_BASE_URL", env["OPENAI_BASE_URL"])
        env.setdefault("GEAK_BASE_URL", env["OPENAI_BASE_URL"])
        env.setdefault("LLM_API_BASE", env["OPENAI_BASE_URL"])
    if "AMD_LLM_API_KEY" not in env and "AMD_API_KEY" in env:
        env["AMD_LLM_API_KEY"] = env["AMD_API_KEY"]
    return {"env_vars": env}


def quiet_ray_init(num_gpus: Optional[int] = None, log_path: Optional[Path] = None):
    """Initialize ray while suppressing the connect banner on stdout.

    If the reachable cluster was started under a different Python/Ray than
    this process (issue #432 — e.g. cluster py3.10 vs submitter py3.12),
    ``ray.init`` raises a "Version mismatch" RuntimeError in ~0.8s. Rather
    than letting that bubble up as a mislabeled "compile failed" REVERT, we
    tear the foreign cluster down, bring up a fresh local head under THIS
    interpreter (``force_restart_local_cluster``), and retry ``ray.init``
    once. ``num_gpus`` / ``log_path`` are threaded through to the restart so
    the new head matches the requested GPU count and the action is audited
    in ``ray_lifecycle.log``.
    """
    import contextlib
    import io
    import ray
    runtime_env = safe_runtime_env()

    def _init() -> None:
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
        # Foreign cluster under a different interpreter — replace it with a
        # local head under this Python, then retry exactly once.
        try:
            ray.shutdown()
        except Exception:  # noqa: BLE001
            pass
        force_restart_local_cluster(num_gpus=num_gpus, log_path=log_path)
        _init()
    return runtime_env
