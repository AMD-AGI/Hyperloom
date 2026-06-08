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
    proc = subprocess.run(
        ["ray", "status"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def ensure_ray_cluster(num_gpus: Optional[int] = None, log_path: Optional[Path] = None) -> bool:
    """Ensure a Ray cluster is reachable.

    Returns True if this call started Ray, False if already running; raises on failure.
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
    if not started:
        return
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("Stopping Ray started by kernel-agent\n")
            subprocess.run(["ray", "stop", "--force"], stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        subprocess.run(["ray", "stop", "--force"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)


# Env vars safe to forward to Ray workers; excludes *_VISIBLE_DEVICES (forcing them triggers ROCm IndexError).
SAFE_ENV_KEYS = (
    "PATH", "HOME", "LD_LIBRARY_PATH",
    "HYPERLOOM_KERNEL_AGENT_ROOT", "KERNEL_AGENT_ROOT",
    # USER_DATA_PATH is the single artefact root others default under.
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


def quiet_ray_init():
    """Initialize ray while suppressing the connect banner on stdout."""
    import contextlib
    import io
    import ray
    runtime_env = safe_runtime_env()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "auto"),
            ignore_reinit_error=True,
            log_to_driver=False,
            logging_level="error",
            runtime_env=runtime_env,
        )
    return runtime_env
