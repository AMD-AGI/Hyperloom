# Copyright Advanced Micro Devices, Inc. All rights reserved.

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

# Custom Ray resource declared on the single-node head so serving-family work
# (serving / benchmark / profile / gpu_research) can hold a whole-machine
# ``serving_slot`` as the authoritative physical mutex (ray_modify.plan.md §12
# T6, decision 1). Capacity 1 => at most one serving-family task holds the node
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


def isolated_compile_cache_env(output_dir, base_env: Optional[dict] = None) -> dict:
    """Return an env dict with per-attempt JIT/compile cache dirs.

    The kernel-opt parallel path can run multiple backend attempts at the same
    time and they trigger aiter / triton / inductor compiles. aiter
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

    Args:
        output_dir: The per-attempt output directory caches are nested under.
        base_env: Base environment to copy; defaults to ``os.environ``.

    Returns:
        An environment dict with per-attempt cache directories set.
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

    Args:
        num_gpus: Optional GPU count to pass to ``ray start --head``.
        log_path: Optional path to append ``ray start`` output.

    Returns:
        ``True`` if this call started Ray, ``False`` if it was already running.

    Raises:
        RuntimeError: If starting the Ray head node fails.
    """
    if ray_status_ok():
        return False
    # Raise the open-files limit before the raylet starts.
    ensure_fd_limit(log_path=log_path)
    # Bind the dashboard/jobs API to loopback (avoids exposing the
    # unauthenticated Ray Jobs RCE surface); GCS (:6379) is unaffected.
    cmd = ["ray", "start", "--head", "--port=6379", "--dashboard-host=127.0.0.1"]
    if num_gpus is not None:
        cmd.append(f"--num-gpus={num_gpus}")
    # Declare the ``serving_slot`` custom resource (§12 T6 authoritative mutex).
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
    # Raise the open-files limit before the fresh raylet starts.
    ensure_fd_limit(log_path=log_path)
    stop_cmd = ["ray", "stop", "--force"]
    # Bind the dashboard/jobs API to loopback (see ensure_ray_cluster).
    start_cmd = ["ray", "start", "--head", "--port=6379", "--dashboard-host=127.0.0.1"]
    if num_gpus is not None:
        start_cmd.append(f"--num-gpus={num_gpus}")
    # Re-declare the ``serving_slot`` custom resource after a fresh head (§12 T6).
    start_cmd.extend(_resources_start_args())
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
    "SAFE_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
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
    "GEAK_ROOT", "GEAK_E2E_RUNNER",
    "GEAK_CLAUDE_EFFORT", "GEAK_CLAUDE_MODEL", "GEAK_E2E_TIMEOUT_S",
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
    per-provider API keys and base URLs from ``SAFE_API_KEY`` /
    ``OPENAI_BASE_URL``). GPU-visibility variables are deliberately
    excluded so Ray manages device assignment itself.

    Returns:
        dict: A ``{"env_vars": {...}}`` mapping suitable for passing as
            Ray's ``runtime_env``.
    """
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    # SAFE_API_KEY is primary; a split deploy falls back to the per-provider
    # key. GEAK speaks the OpenAI protocol, so it takes the OpenAI-side key.
    openai_key = (
        env.get("SAFE_API_KEY")
        or env.get("OPENAI_API_KEY")
        or env.get("ANTHROPIC_AUTH_TOKEN")
        or env.get("ANTHROPIC_API_KEY")
    )
    if openai_key:
        env.setdefault("OPENAI_API_KEY", openai_key)
        env.setdefault("GEAK_API_KEY", openai_key)
        env.setdefault("LLM_API_KEY", openai_key)
        env.setdefault("AMD_LLM_API_KEY", openai_key)
        env.setdefault("LLM_GATEWAY_KEY", openai_key)
    anthropic_key = (
        env.get("SAFE_API_KEY")
        or env.get("ANTHROPIC_API_KEY")
        or env.get("ANTHROPIC_AUTH_TOKEN")
        or env.get("OPENAI_API_KEY")
    )
    if anthropic_key:
        env.setdefault("ANTHROPIC_API_KEY", anthropic_key)
        env.setdefault("ANTHROPIC_AUTH_TOKEN", anthropic_key)
    # OPENAI_BASE_URL primary; fall back to ANTHROPIC_BASE_URL.
    base_url = env.get("OPENAI_BASE_URL") or env.get("ANTHROPIC_BASE_URL")
    if base_url:
        env.setdefault("ANTHROPIC_BASE_URL", base_url)
        env.setdefault("OPENAI_BASE_URL", base_url)
        env.setdefault("GEAK_BASE_URL", base_url)
        env.setdefault("LLM_API_BASE", base_url)
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
