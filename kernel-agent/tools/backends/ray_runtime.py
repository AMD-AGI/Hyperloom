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
import resource
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Minimum soft RLIMIT_NOFILE the Ray raylet needs to stay up (issue #433).
# The raylet opens a large number of fds (sockets, plasma store, per-worker
# pipes); at the container default soft limit (1024) it aborts on startup /
# is left as a zombie that only `ray stop --force` can clear. Operators can
# override via RAY_MIN_NOFILE.
DEFAULT_MIN_NOFILE = 65536


def _fd_limit_warn(msg: str) -> None:
    """Single indirection for fd-limit warnings.

    Kept as a module function so tests can capture it and so every warning
    carries the same prefix. Goes to stderr; callers that own a log_path
    also get a line in the Ray lifecycle log.
    """
    print(f"[kernel-agent WARN] {msg}", file=sys.stderr)


def _min_nofile_target() -> int:
    raw = os.environ.get("RAY_MIN_NOFILE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MIN_NOFILE


def ensure_fd_limit(
    min_soft: Optional[int] = None, log_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Raise this process's RLIMIT_NOFILE soft limit before Ray spawns the
    raylet (issue #433).

    The child ``ray start`` process inherits this process's limits, so the
    raylet's open-files ceiling is whatever we set here. We raise the soft
    limit to ``min(min_soft, hard)``. Raising the soft limit up to the hard
    cap needs no privileges; lifting the hard cap does (CAP_SYS_RESOURCE),
    so when the hard cap is itself below ``min_soft`` we raise soft as high
    as allowed and warn — only ``docker run --ulimit nofile=...`` at
    container launch can lift the hard cap in an unprivileged container.

    Returns the ``(soft, hard)`` limit in effect after the call.
    """
    if min_soft is None:
        min_soft = _min_nofile_target()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= min_soft:
        return soft, hard
    target = min(min_soft, hard)
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
    if hard < min_soft:
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
                    f"[fd_limit] RLIMIT_NOFILE soft raised to {soft} "
                    f"(hard={hard}) for raylet stability (issue #433)\n"
                )
        except OSError:  # pragma: no cover - logging must never break startup
            pass
    return soft, hard


def isolated_compile_cache_env(output_dir, base_env: Optional[dict] = None) -> dict:
    """Return an env dict with per-attempt JIT/compile cache dirs.

    The kernel-opt parallel path runs the GEAK and OOB ladders at the same
    time and both trigger aiter / triton / inductor compiles. aiter
    (FileBaton) and triton self-serialize per cache *key*, so steady-state
    concurrent compiles are safe -- but a sibling killed on timeout can leave
    a stale lock, and an ``AITER_REBUILD=1`` import wipes the shared build dir
    mid-compile. Pinning each attempt to caches under its unique ``output_dir``
    removes that cross-talk without serializing the backends.

    Isolated (all under ``<output_dir>/.cache``):
      - ``TRITON_CACHE_DIR``        triton @jit .hsaco/.json cache
      - ``AITER_ROOT_DIR``          aiter cpp_itfs runtime build (``$AITER_ROOT_DIR/build``)
      - ``TORCHINDUCTOR_CACHE_DIR`` torch.compile inductor cache

    Deliberately NOT isolated: ``AITER_JIT_DIR`` (the ``@compile_ops``
    ``jit/build`` dir ships ~100 prebuilt modules; redirecting it would force a
    full recompile per attempt and that path is already FileBaton-locked).
    ``AITER_ROOT_DIR`` only steers aiter's cpp_itfs ``BUILD_DIR`` -- sources
    and configs resolve off the package dir -- so redirecting it is
    compile-safe.
    """
    env = dict(os.environ if base_env is None else base_env)
    base = os.path.join(str(output_dir), ".cache")
    for var, sub in (
        ("TRITON_CACHE_DIR", "triton"),
        ("AITER_ROOT_DIR", "aiter"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ):
        path = os.path.join(base, sub)
        os.makedirs(path, exist_ok=True)
        env[var] = path
    return env


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

    Returns True if this call started Ray, False if already running; raises on failure.
    """
    if ray_status_ok():
        return False
    # issue #433: raise the open-files limit before the raylet starts so it
    # inherits a high enough ceiling and does not abort / zombie at the
    # container default (1024).
    ensure_fd_limit(log_path=log_path)
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
    """True when Ray refused the job due to a cluster started under a different Python/Ray (issue #432); matches the stable ``Version mismatch`` banner."""
    return "version mismatch" in (text or "").lower()


def force_restart_local_cluster(
    num_gpus: Optional[int] = None, log_path: Optional[Path] = None,
) -> None:
    """Tear down any reachable Ray cluster and start a fresh local head under THIS interpreter. Recovers from a stale/foreign cluster (issue #432) whose version mismatch otherwise mislabels as a "compile failed" REVERT; also clears raylet zombies. Raises RuntimeError if the fresh head fails to start."""
    # issue #433: raise the open-files limit before the fresh raylet starts
    # so it inherits a high enough ceiling (the container default 1024 makes
    # the raylet abort on startup / linger as a zombie).
    ensure_fd_limit(log_path=log_path)
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


# Env vars safe to forward to Ray workers; excludes *_VISIBLE_DEVICES (Ray-owned; forcing them triggers set_visible_accelerator_ids IndexError on ROCm).
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
