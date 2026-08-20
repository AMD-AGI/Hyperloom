# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Off-loop targeted-build runner.

Spawns a build command as a **detached** subprocess (its own process group) so a
multi-hour compile never blocks the coordinator tick loop, then polls it across
ticks against a monotonic wall-clock deadline. On timeout it tears the whole
process group down non-blocking (SIGTERM, then SIGKILL after a grace window so a
poll never sleeps).

The build command is argv-only (never a shell string). Each build gets a
per-attempt ``INFERENCE_OPTIMIZER_AITER_JIT_DIR`` so it never shares the
node-global aiter JIT cache.

Alongside the spawn/poll/kill supervisor, this module holds the three build
recipes -- :func:`run_aiter_build`, :func:`run_sgl_kernel_build`,
:func:`run_vllm_source_build` -- plus ``_driver_main``. When an action carries
no explicit ``build_command``, the detached subprocess re-enters this module as
``__main__`` and the driver dispatches to the matching recipe.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

from hyperloom.common.git_safety import safe_directory_args
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .build_actions import BuildResult, FrameworkRuntime, TargetedBuildAction

log = logging.getLogger(__name__)

# Seconds between SIGTERM and the escalation to SIGKILL on the process group.
_KILL_GRACE_SEC = 5.0

# Default per-component build budgets (upper bounds), in seconds.
_DEFAULT_BUDGET_SEC: dict[str, int] = {
    "aiter": 40 * 60,
    "sgl_kernel": 60 * 60,
    "vllm_source": 90 * 60,
    "framework_ext": 90 * 60,
}


def default_budget_sec(component: str) -> int:
    """Per-component wall-clock budget upper bound."""
    return _DEFAULT_BUDGET_SEC.get(component, 40 * 60)


@dataclass
class BuildHandle:
    """In-memory handle for one in-flight detached build.

    Not persisted directly; the durable copy is ``pending_targeted_build`` in
    shared state. ``sigterm_at`` tracks the two-phase kill across poll calls so
    the reaper never blocks the tick.
    """

    action: TargetedBuildAction
    attempt_root: str
    aiter_jit_dir: str
    build_log_path: str
    proc: Any
    pid: int
    pgid: int
    deadline: float
    sigterm_at: float = 0.0

    def to_sentinel(self, task_id: str) -> dict[str, Any]:
        """Project onto the ``pending_targeted_build`` sentinel dict."""
        return {
            "task_id": task_id,
            "pid": self.pid,
            "pgid": self.pgid,
            "attempt_root": self.attempt_root,
            "aiter_jit_dir": self.aiter_jit_dir,
            "deadline": self.deadline,
            "action": self.action.to_state(),
            "build_log_path": self.build_log_path,
            "ts": time.time(),
        }


def _resolve_budget_sec(action: TargetedBuildAction) -> int:
    budget = int(action.build_budget_sec or 0)
    return budget if budget > 0 else default_budget_sec(action.component)


def spawn_build(
    action: TargetedBuildAction,
    *,
    attempt_root: str,
    command: list[str] | None = None,
    run: Callable[..., Any] = subprocess.Popen,
    now: Callable[[], float] = time.monotonic,
) -> BuildHandle:
    """Spawn a targeted build as a detached process group.

    Creates ``attempt_root`` and a per-attempt ``aiter_jit`` dir, exports
    ``INFERENCE_OPTIMIZER_AITER_JIT_DIR`` for it, and starts the argv command in
    a new session (own process group) with output redirected to ``build.log``.

    Args:
        action: The build to run.
        attempt_root: Directory anchoring this attempt's logs and JIT cache.
        command: Explicit argv to spawn; overrides ``action.build_command`` when
            not None. The production caller
            (``build_lifecycle._driver_command``) always passes this, using the
            off-loop driver entrypoint when ``action.build_command`` is empty.
        run: Injectable process spawner (defaults to ``subprocess.Popen``).
        now: Injectable monotonic clock (defaults to ``time.monotonic``).

    Returns:
        BuildHandle: The in-flight handle (pid/pgid/deadline/log path).

    Raises:
        ValueError: If neither ``command`` nor ``action.build_command`` yields a
            non-empty argv.
    """
    argv = command if command is not None else list(action.build_command)
    if not argv:
        raise ValueError("targeted_build: build_command must be a non-empty argv (or pass command=)")

    root = Path(attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    jit_dir = root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    build_log = root / "build.log"

    env = dict(os.environ)
    env["INFERENCE_OPTIMIZER_AITER_JIT_DIR"] = str(jit_dir)
    for k, v in dict(action.envs).items():
        env[str(k)] = str(v)

    log_fh = build_log.open("w", encoding="utf-8")
    try:
        proc = run(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(root),
            start_new_session=True,
        )
    except Exception:
        log_fh.close()
        raise

    pid = int(getattr(proc, "pid", 0) or 0)
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = pid

    deadline = now() + float(_resolve_budget_sec(action))
    log.info(
        "targeted_build: spawned %s build pid=%d pgid=%d budget=%ds root=%s",
        action.component,
        pid,
        pgid,
        _resolve_budget_sec(action),
        attempt_root,
    )
    return BuildHandle(
        action=action,
        attempt_root=str(root),
        aiter_jit_dir=str(jit_dir),
        build_log_path=str(build_log),
        proc=proc,
        pid=pid,
        pgid=pgid,
        deadline=deadline,
    )


def kill_build_pgroup(pgid: int, *, sig: int = signal.SIGTERM) -> None:
    """Signal an entire build process group (best-effort, non-blocking)."""
    if pgid <= 0:
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError, OverflowError, ValueError):
        # Best-effort: the group may already be gone or unsignalable.
        pass


def _build_runtime(handle: BuildHandle) -> FrameworkRuntime:
    """The runtime a KEEP would promote."""
    return FrameworkRuntime(
        source_root=handle.attempt_root,
        attempt_root=handle.attempt_root,
        runtime_env={"INFERENCE_OPTIMIZER_AITER_JIT_DIR": handle.aiter_jit_dir},
    )


def _finalize(handle: BuildHandle, *, ok: bool, failure_class: str, summary: str) -> BuildResult:
    return BuildResult(
        ok=ok,
        attempt_root=handle.attempt_root,
        runtime=_build_runtime(handle) if ok else FrameworkRuntime(),
        build_log_path=handle.build_log_path,
        failure_class="ok" if ok else failure_class,
        failure_summary=summary,
        error="" if ok else summary,
    )


def poll_build(
    handle: BuildHandle,
    *,
    now: Callable[[], float] = time.monotonic,
) -> BuildResult | None:
    """Poll a build once; return ``None`` while still running.

    Non-blocking: on deadline it sends SIGTERM and records ``sigterm_at``, then
    on a later poll past the grace window escalates to SIGKILL, so the reaper
    never sleeps inside a tick.  When the process exits, attempts to load a
    rich ``result.json`` written by the driver; falls back to the exit-code
    classification when the file is absent.

    Args:
        handle: The in-flight build handle.
        now: Injectable monotonic clock.

    Returns:
        BuildResult when terminal (exited / timed out+dead), else ``None``.
    """
    rc = handle.proc.poll()
    if rc is not None:
        if handle.sigterm_at > 0.0:
            return _finalize(
                handle,
                ok=False,
                failure_class="timeout",
                summary=f"build exceeded wall-clock budget and was terminated (rc={rc})",
            )
        # Try to load a rich result written by the driver subprocess.
        rich = _load_result_json(handle.attempt_root)
        if rich is not None:
            return rich
        if int(rc) == 0:
            return _finalize(handle, ok=True, failure_class="ok", summary="")
        return _finalize(
            handle,
            ok=False,
            failure_class="compile_error",
            summary=f"build command exited {int(rc)}",
        )

    t = now()
    if handle.sigterm_at > 0.0:
        # Already asked to stop; escalate to SIGKILL once the grace elapses.
        if t - handle.sigterm_at >= _KILL_GRACE_SEC:
            kill_build_pgroup(handle.pgid, sig=signal.SIGKILL)
        return None
    if t >= handle.deadline:
        kill_build_pgroup(handle.pgid, sig=signal.SIGTERM)
        handle.sigterm_at = t
        return None
    return None


def _load_result_json(attempt_root: str) -> BuildResult | None:
    """Try to load a ``result.json`` written by the driver; return None on miss."""
    import json

    result_path = Path(attempt_root) / "result.json"
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return BuildResult.from_state(data)
    except Exception:  # noqa: BLE001 — missing/corrupt result falls back
        return None


def _read_build_system_requires(worktree_dir: Any) -> list[str]:
    """Return a checkout's PEP 518 ``[build-system].requires`` for pre-install.

    Used before a ``--no-build-isolation`` editable install so the attempt venv
    carries the backend deps (setuptools-scm / setuptools-rust / packaging /
    cmake / ninja / wheel / jinja2 / ...) the checkout pins. Any ``torch``
    requirement is dropped so the venv's ROCm torch is never clobbered by an
    upstream CUDA torch pin. Missing / unparseable pyproject → empty list
    (caller degrades to the plain editable install).
    """
    try:
        import tomllib  # py3.11+
    except Exception:  # noqa: BLE001
        return []
    pyproject = Path(worktree_dir) / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — no/invalid pyproject → skip pre-install
        return []
    reqs = (data.get("build-system") or {}).get("requires") or []
    out: list[str] = []
    for r in reqs:
        if not isinstance(r, str):
            continue
        # Drop torch pins — the ROCm torch already in the venv must win.
        name = r.strip().lower()
        if name.startswith("torch") and not name.startswith("torchvision") and not name.startswith("torchaudio"):
            continue
        out.append(r.strip())
    return out


# ---------------------------------------------------------------------------
# AITER real-build recipe
# ---------------------------------------------------------------------------

# Default AITER upstream; overridable via TargetedBuildAction.repo_url.
_AITER_DEFAULT_REPO = "https://github.com/ROCm/aiter"
# Headroom for a full AITER compile + build cache (GB).
_AITER_DISK_PER_CANDIDATE_GB = 6.0
# Default max parallel compile jobs; can be overridden by action.max_jobs.
_AITER_DEFAULT_MAX_JOBS = 8


def run_aiter_build(
    action: TargetedBuildAction,
    attempt_root: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[..., Any] | None = None,
    disk_preflight_fn: Callable[..., Any] | None = None,
) -> BuildResult:
    """Run a full isolated AITER targeted build.

    Executes entirely in-process (call it from the detached driver subprocess
    so the coordinator tick loop is never blocked).  All subprocess calls go
    through the injectable ``run`` shim for testability.
    """
    import time as _time

    from .build_utils import (
        AbiMismatchError,
        check_rocm_toolchain_alignment,
        probe_torch_abi,
        run_argv,
        sort_tags_desc,
        verify_fresh_artifacts,
        verify_symbols,
        write_rocm_torch_constraints,
    )

    root = Path(attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    jit_dir = root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    aiter_home = root / "home"
    aiter_home.mkdir(parents=True, exist_ok=True)
    build_log = root / "build.log"

    def _run(argv, **kw):
        """Route through injectable run shim; capture output for logs."""
        kw.setdefault("capture_output", True)
        kw.setdefault("text", True)
        kw.setdefault("timeout", 3600)
        return run(argv, **kw)

    def _fail(failure_class: str, summary: str) -> BuildResult:
        return BuildResult(
            ok=False,
            attempt_root=str(root),
            build_log_path=str(build_log),
            failure_class=failure_class,
            failure_summary=summary,
            error=summary,
        )

    # 1. Disk preflight -------------------------------------------------------
    try:
        if disk_preflight_fn is not None:
            disk_preflight_fn(root, 1, per_candidate_gb=_AITER_DISK_PER_CANDIDATE_GB)
        else:
            from hyperloom.agents.framework.isolation import (
                DiskPreflightError,
                disk_preflight,
            )
            try:
                disk_preflight(root, 1, per_candidate_gb=_AITER_DISK_PER_CANDIDATE_GB)
            except DiskPreflightError as exc:
                return _fail("preflight_disk", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _fail("preflight_disk", f"disk preflight raised: {exc!r}")

    # 2. Toolchain + ABI preflight --------------------------------------------
    import os as _os

    tc_ok, tc_msg = check_rocm_toolchain_alignment(env=dict(_os.environ), run=_run)
    if not tc_ok:
        return _fail("preflight_toolchain", tc_msg)

    # Choose the Python interpreter in the attempt venv (created below).
    # For preflight we use the host interpreter for the ABI probe.
    import sys as _sys

    host_py = _sys.executable
    abi = probe_torch_abi(host_py, run=_run)
    if not abi.get("is_rocm"):
        return _fail(
            "preflight_toolchain",
            f"host torch is not a ROCm build (torch={abi.get('torch_version')})",
        )

    # 3. Isolation worktree + venv -------------------------------------------
    repo_url = str(action.repo_url or _AITER_DEFAULT_REPO).strip() or _AITER_DEFAULT_REPO
    ref = str(action.ref or "").strip()
    worktree_dir: Path | None = None
    venv_dir: Path | None = None

    try:
        from hyperloom.agents.framework.isolation import (
            prepare_candidate_workspace,
            prepare_repo_cache,
        )
        from hyperloom.agents.framework.models import Baseline, Candidate, ExploreRequest

        req = ExploreRequest(
            framework="aiter",
            repo_url=repo_url,
            work_dir=root,
            baseline=Baseline(throughput=0.0),
            prepare_candidate_env=True,
        )
        prepare_repo_cache(req)
        candidate = Candidate(ref=ref or "HEAD", repo=repo_url)
        ws = prepare_candidate_workspace(req, candidate, index=0, execute=True)
        worktree_dir = ws.worktree_dir
        venv_dir = ws.venv_dir
        attempt_py = str(venv_dir / "bin" / "python")
    except Exception as exc:  # noqa: BLE001
        return _fail("compile_error", f"workspace preparation failed: {exc!r}")

    # 4. ROCm torch constraint file -------------------------------------------
    constraint_path = root / "torch_constraints.txt"
    try:
        write_rocm_torch_constraints(attempt_py, str(constraint_path), run=_run)
    except AbiMismatchError as exc:
        return _fail("abi_mismatch", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _fail("preflight_toolchain", f"torch constraint probe failed: {exc!r}")

    # 5. pip install (pinned ref or tag-desc autoselect) ----------------------
    gpu_arch = str(action.gpu_arch or "").strip()
    max_jobs = int(action.max_jobs or _AITER_DEFAULT_MAX_JOBS)
    install_env = {
        **_os.environ,
        "AITER_ROOT_DIR": str(aiter_home),
        "HOME": str(aiter_home),
        "INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(jit_dir),
        "AITER_REBUILD": "1",
    }
    if gpu_arch:
        install_env["PYTORCH_ROCM_ARCH"] = gpu_arch
        install_env["AITER_ROCM_ARCH"] = gpu_arch
    if max_jobs:
        install_env["MAX_JOBS"] = str(max_jobs)

    git_run = git if git is not None else _run
    since_unix = _time.time()
    selected_ref = ref
    installed_ok = False

    pip_base = [
        attempt_py, "-m", "pip", "install",
        "--constraint", str(constraint_path),
        "--config-settings", "editable_mode=compat",
        "-e", str(worktree_dir),
    ]

    if ref:
        res = run_argv(pip_base, cwd=str(worktree_dir), env=install_env, timeout_sec=3600, run=_run)
        if res.returncode != 0:
            log_msg = res.stderr_tail or res.stdout_tail
            build_log.write_text(log_msg, encoding="utf-8")
            return _fail("compile_error", f"pip install failed for ref={ref!r}: rc={res.returncode}")
    else:
        # Tag-descending autoselect
        tags_res = git_run(
            ["git", *safe_directory_args(["-C", str(worktree_dir), "tag", "-l", "v*"])],
            capture_output=True, text=True, timeout=60,
        )
        raw_tags = (getattr(tags_res, "stdout", "") or "").strip().splitlines()
        tags = sort_tags_desc([t.strip() for t in raw_tags if t.strip()])
        if not tags:
            return _fail("compile_error", "no AITER version tags found in the cloned repo")

        for tag in tags:
            checkout_res = git_run(
                ["git", *safe_directory_args(["-C", str(worktree_dir), "checkout", tag])],
                capture_output=True, text=True, timeout=120,
            )
            if getattr(checkout_res, "returncode", 1) != 0:
                continue
            res = run_argv(pip_base, cwd=str(worktree_dir), env=install_env, timeout_sec=3600, run=_run)
            if res.returncode == 0:
                probe = run_argv(
                    [attempt_py, "-c", "import aiter"],
                    cwd=str(root), env=install_env, timeout_sec=60, run=_run,
                )
                if probe.returncode == 0:
                    installed_ok = True
                    selected_ref = tag
                    break
        if not installed_ok:
            return _fail("compile_error", "no AITER tag installed and imported successfully")

    # 6. Artifact freshness + symbol verify ------------------------------------
    expected_artifacts = list(action.expected_artifacts) or ["**/*.so"]
    freshness = verify_fresh_artifacts(str(worktree_dir), since_unix, expected_artifacts)
    built_paths: tuple[str, ...] = tuple(freshness.get("fresh", []))

    sym_result: dict[str, Any] = {"verified": True}
    if action.expected_symbols:
        sym_result = verify_symbols(attempt_py, list(action.expected_symbols), run=_run)
        if not sym_result["verified"]:
            return _fail(
                "symbol_missing",
                f"expected symbols not importable after build: {sym_result['missing']}",
            )

    # 7. Collect installed_versions + hashes, return BuildResult ---------------
    sha_res = git_run(
        ["git", *safe_directory_args(["-C", str(worktree_dir), "rev-parse", "--short", "HEAD"])],
        capture_output=True, text=True, timeout=30,
    )
    commit_sha = (getattr(sha_res, "stdout", "") or "").strip()

    installed_versions: dict[str, str] = {
        "torch": abi.get("torch_version", ""),
        "aiter_ref": selected_ref,
        "aiter_sha": commit_sha,
        "arch": gpu_arch,
        "hip_version": abi.get("hip_version", ""),
    }
    if action.source_pr_url:
        installed_versions["source_pr_url"] = action.source_pr_url

    runtime = FrameworkRuntime(
        pythonpath_prefixes=(str(worktree_dir),),
        runtime_env={"INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(jit_dir)},
        source_root=str(worktree_dir),
        attempt_root=str(root),
    )

    return BuildResult(
        ok=True,
        attempt_root=str(root),
        runtime=runtime,
        built_artifacts=built_paths,
        installed_versions=installed_versions,
        build_log_path=str(build_log),
        failure_class="ok",
    )


# ---------------------------------------------------------------------------
# sgl-kernel recipe
# ---------------------------------------------------------------------------

_SGLANG_DEFAULT_REPO = "https://github.com/sgl-project/sglang"
_SGLANG_DISK_PER_CANDIDATE_GB = 8.0
_SGLANG_DEFAULT_MAX_JOBS = 8


def run_sgl_kernel_build(
    action: TargetedBuildAction,
    attempt_root: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[..., Any] | None = None,
    disk_preflight_fn: Callable[..., Any] | None = None,
) -> BuildResult:
    """Run an isolated sgl-kernel targeted build.

    Clones SGLang into an isolated worktree/venv, builds the ROCm sgl-kernel
    extension for the explicit gpu_arch (AMDGPU_TARGET), then installs the
    Python package and verifies a fresh compiled artifact.
    """
    import os as _os
    import sys as _sys
    import time as _time

    from .build_utils import (
        AbiMismatchError,
        check_rocm_toolchain_alignment,
        probe_torch_abi,
        run_argv,
        verify_fresh_artifacts,
        verify_symbols,
        write_rocm_torch_constraints,
    )

    root = Path(attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    build_log = root / "build.log"

    def _run(argv, **kw):
        kw.setdefault("capture_output", True)
        kw.setdefault("text", True)
        kw.setdefault("timeout", 3600)
        return run(argv, **kw)

    def _fail(failure_class: str, summary: str) -> BuildResult:
        return BuildResult(
            ok=False,
            attempt_root=str(root),
            build_log_path=str(build_log),
            failure_class=failure_class,
            failure_summary=summary,
            error=summary,
        )

    # Disk preflight
    try:
        if disk_preflight_fn is not None:
            disk_preflight_fn(root, 1, per_candidate_gb=_SGLANG_DISK_PER_CANDIDATE_GB)
        else:
            from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

            try:
                disk_preflight(root, 1, per_candidate_gb=_SGLANG_DISK_PER_CANDIDATE_GB)
            except DiskPreflightError as exc:
                return _fail("preflight_disk", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _fail("preflight_disk", f"disk preflight raised: {exc!r}")

    tc_ok, tc_msg = check_rocm_toolchain_alignment(env=dict(_os.environ), run=_run)
    if not tc_ok:
        return _fail("preflight_toolchain", tc_msg)

    host_py = _sys.executable
    abi = probe_torch_abi(host_py, run=_run)
    if not abi.get("is_rocm"):
        return _fail("preflight_toolchain", f"host torch is not a ROCm build (torch={abi.get('torch_version')})")

    gpu_arch = str(action.gpu_arch or "").strip()
    if not gpu_arch:
        return _fail("preflight_toolchain", "gpu_arch must be set explicitly for sgl-kernel (L6)")

    repo_url = str(action.repo_url or _SGLANG_DEFAULT_REPO).strip() or _SGLANG_DEFAULT_REPO
    ref = str(action.ref or "").strip()
    max_jobs = int(action.max_jobs or _SGLANG_DEFAULT_MAX_JOBS)

    # Isolation worktree + venv
    worktree_dir: Path | None = None
    venv_dir: Path | None = None

    try:
        from hyperloom.agents.framework.isolation import prepare_candidate_workspace, prepare_repo_cache
        from hyperloom.agents.framework.models import Baseline, Candidate, ExploreRequest

        req = ExploreRequest(
            framework="sglang",
            repo_url=repo_url,
            work_dir=root,
            baseline=Baseline(throughput=0.0),
            prepare_candidate_env=True,
        )
        prepare_repo_cache(req)
        candidate = Candidate(ref=ref or "HEAD", repo=repo_url)
        ws = prepare_candidate_workspace(req, candidate, index=0, execute=True)
        worktree_dir = ws.worktree_dir
        venv_dir = ws.venv_dir
        attempt_py = str(venv_dir / "bin" / "python")
    except Exception as exc:  # noqa: BLE001
        return _fail("compile_error", f"workspace preparation failed: {exc!r}")

    constraint_path = root / "torch_constraints.txt"
    try:
        write_rocm_torch_constraints(attempt_py, str(constraint_path), run=_run)
    except AbiMismatchError as exc:
        return _fail("abi_mismatch", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _fail("preflight_toolchain", f"torch constraint probe failed: {exc!r}")

    since_unix = _time.time()
    sgl_kernel_dir = worktree_dir / "sgl-kernel"

    # Build sgl-kernel with explicit AMDGPU_TARGET
    install_env = {
        **_os.environ,
        "AMDGPU_TARGET": gpu_arch,
        "MAX_JOBS": str(max_jobs),
    }
    kernel_build = run_argv(
        [attempt_py, "setup_rocm.py", "install"],
        cwd=str(sgl_kernel_dir),
        env=install_env,
        timeout_sec=3600,
        run=_run,
    )
    if kernel_build.returncode != 0:
        build_log.write_text(kernel_build.stderr_tail or kernel_build.stdout_tail, encoding="utf-8")
        return _fail("compile_error", f"sgl-kernel compile failed (rc={kernel_build.returncode})")

    # Copy pyproject_other.toml if present (mirrors the installer).
    py_other = worktree_dir / "python" / "pyproject_other.toml"
    if py_other.is_file():
        import shutil

        shutil.copy2(str(py_other), str(worktree_dir / "python" / "pyproject.toml"))

    # Install SGLang python package
    pip_cmd = [
        attempt_py, "-m", "pip", "install",
        "--constraint", str(constraint_path),
        "-e", str(worktree_dir / "python[srt_hip]"),
    ]
    pip_res = run_argv(pip_cmd, cwd=str(root), env=dict(_os.environ), timeout_sec=3600, run=_run)
    if pip_res.returncode != 0:
        build_log.write_text(pip_res.stderr_tail or pip_res.stdout_tail, encoding="utf-8")
        return _fail("compile_error", f"sgl-kernel pip install failed (rc={pip_res.returncode})")

    # Verify
    expected_artifacts = list(action.expected_artifacts) or ["**/*.so"]
    freshness = verify_fresh_artifacts(str(sgl_kernel_dir), since_unix, expected_artifacts)
    built_paths: tuple[str, ...] = tuple(freshness.get("fresh", []))

    if action.expected_symbols:
        sym_result = verify_symbols(attempt_py, list(action.expected_symbols), run=_run)
        if not sym_result["verified"]:
            return _fail("symbol_missing", f"symbols not importable: {sym_result['missing']}")

    git_run = git if git is not None else _run
    sha_res = git_run(["git", *safe_directory_args(["-C", str(worktree_dir), "rev-parse", "--short", "HEAD"])],
                      capture_output=True, text=True, timeout=30)
    commit_sha = (getattr(sha_res, "stdout", "") or "").strip()

    installed_versions = {
        "torch": abi.get("torch_version", ""),
        "sgl_kernel_ref": ref,
        "sgl_kernel_sha": commit_sha,
        "arch": gpu_arch,
        "hip_version": abi.get("hip_version", ""),
    }
    if action.source_pr_url:
        installed_versions["source_pr_url"] = action.source_pr_url

    runtime = FrameworkRuntime(
        pythonpath_prefixes=(str(worktree_dir / "python"),),
        entrypoint_bin_dir=str(venv_dir / "bin") if venv_dir else "",
        runtime_python_exe=attempt_py,
        runtime_env={"SGLANG_USE_AITER": "1"},
        source_root=str(worktree_dir),
        attempt_root=str(root),
    )
    return BuildResult(
        ok=True,
        attempt_root=str(root),
        runtime=runtime,
        built_artifacts=built_paths,
        installed_versions=installed_versions,
        build_log_path=str(build_log),
        failure_class="ok",
    )


# ---------------------------------------------------------------------------
# vLLM from source recipe
# ---------------------------------------------------------------------------

_VLLM_DEFAULT_REPO = "https://github.com/ROCm/vllm"
_VLLM_DISK_PER_CANDIDATE_GB = 20.0
_VLLM_DEFAULT_MAX_JOBS = 8

# Inline verify_vllm_rocm probe.
_VERIFY_VLLM_ROCM_SCRIPT = """\
import sys, torch
if not getattr(torch.version, "hip", None):
    print("torch is not a ROCm build", file=sys.stderr); raise SystemExit(1)
import vllm
try:
    from vllm.platforms import current_platform
except Exception as exc:
    print(f"cannot import vllm platform: {exc}", file=sys.stderr); raise SystemExit(1)
is_rocm = False
checker = getattr(current_platform, "is_rocm", None)
if callable(checker):
    try: is_rocm = bool(checker())
    except Exception: pass
if "rocm" in f"{current_platform!r} {current_platform.__class__.__name__}".lower():
    is_rocm = True
if not is_rocm:
    print("vLLM did not report ROCm platform", file=sys.stderr); raise SystemExit(1)
print("vllm_rocm_ok")
"""


def run_vllm_source_build(
    action: TargetedBuildAction,
    attempt_root: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[..., Any] | None = None,
    disk_preflight_fn: Callable[..., Any] | None = None,
) -> BuildResult:
    """Run an isolated vLLM-from-source targeted build.

    Clones ROCm/vllm into an isolated worktree, runs ``pip install -e``
    which triggers the CMake ``build_ext`` pass, then verifies ROCm platform
    and fresh compiled artefacts. A non-ROCm torch build is a hard failure
    (``abi_mismatch``, raised by ``write_rocm_torch_constraints``); a Python
    major.minor mismatch between the torch ABI and the launcher is logged as an
    advisory only, and ``runtime_python_exe`` is set to the attempt-venv
    interpreter so the server launches with the right Python.
    """
    import os as _os
    import sys as _sys
    import time as _time

    from .build_utils import (
        AbiMismatchError,
        check_rocm_toolchain_alignment,
        probe_torch_abi,
        run_argv,
        verify_fresh_artifacts,
        verify_symbols,
        write_rocm_torch_constraints,
    )

    root = Path(attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    build_log = root / "build.log"

    def _run(argv, **kw):
        kw.setdefault("capture_output", True)
        kw.setdefault("text", True)
        kw.setdefault("timeout", 5400)
        return run(argv, **kw)

    def _fail(failure_class: str, summary: str) -> BuildResult:
        return BuildResult(
            ok=False,
            attempt_root=str(root),
            build_log_path=str(build_log),
            failure_class=failure_class,
            failure_summary=summary,
            error=summary,
        )

    # Disk preflight (raised headroom: 20 GB for full vLLM build cache)
    try:
        if disk_preflight_fn is not None:
            disk_preflight_fn(root, 1, per_candidate_gb=_VLLM_DISK_PER_CANDIDATE_GB)
        else:
            from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

            try:
                disk_preflight(root, 1, per_candidate_gb=_VLLM_DISK_PER_CANDIDATE_GB)
            except DiskPreflightError as exc:
                return _fail("preflight_disk", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _fail("preflight_disk", f"disk preflight raised: {exc!r}")

    tc_ok, tc_msg = check_rocm_toolchain_alignment(env=dict(_os.environ), run=_run)
    if not tc_ok:
        return _fail("preflight_toolchain", tc_msg)

    host_py = _sys.executable
    abi = probe_torch_abi(host_py, run=_run)
    if not abi.get("is_rocm"):
        return _fail("preflight_toolchain", f"host torch is not a ROCm build (torch={abi.get('torch_version')})")

    gpu_arch = str(action.gpu_arch or "").strip()
    if not gpu_arch:
        return _fail("preflight_toolchain", "gpu_arch must be set explicitly for vLLM source (L6)")

    # ABI-match guard: log an advisory when the torch ABI reports a different Python
    # version; the build continues and runtime_python_exe points to the attempt venv
    # python so the server uses the correct interpreter.
    host_pyver = f"{_sys.version_info.major}.{_sys.version_info.minor}"
    abi_pyver = str(abi.get("python_version") or "").strip()
    if abi_pyver and not abi_pyver.startswith(host_pyver):
        import logging as _log
        _log.getLogger(__name__).info(
            "vLLM source build: torch ABI python %s != host %s; "
            "runtime_python_exe will be set to the attempt venv interpreter",
            abi_pyver,
            host_pyver,
        )

    repo_url = str(action.repo_url or _VLLM_DEFAULT_REPO).strip() or _VLLM_DEFAULT_REPO
    ref = str(action.ref or "").strip()
    max_jobs = int(action.max_jobs or _VLLM_DEFAULT_MAX_JOBS)

    # Isolation worktree + venv
    worktree_dir: Path | None = None
    venv_dir: Path | None = None

    try:
        from hyperloom.agents.framework.isolation import prepare_candidate_workspace, prepare_repo_cache
        from hyperloom.agents.framework.models import Baseline, Candidate, ExploreRequest

        req = ExploreRequest(
            framework="vllm",
            repo_url=repo_url,
            work_dir=root,
            baseline=Baseline(throughput=0.0),
            prepare_candidate_env=True,
        )
        prepare_repo_cache(req)
        candidate = Candidate(ref=ref or "HEAD", repo=repo_url)
        ws = prepare_candidate_workspace(req, candidate, index=0, execute=True)
        worktree_dir = ws.worktree_dir
        venv_dir = ws.venv_dir
        attempt_py = str(venv_dir / "bin" / "python")
    except Exception as exc:  # noqa: BLE001
        return _fail("compile_error", f"workspace preparation failed: {exc!r}")

    constraint_path = root / "torch_constraints.txt"
    try:
        write_rocm_torch_constraints(attempt_py, str(constraint_path), run=_run)
    except AbiMismatchError as exc:
        return _fail("abi_mismatch", str(exc))
    except Exception as exc:  # noqa: BLE001
        return _fail("preflight_toolchain", f"torch constraint probe failed: {exc!r}")

    since_unix = _time.time()
    # vLLM's ROCm setup.py asserts ``CUDA_HOME is not set`` and reuses it as the
    # toolchain root even on ROCm. The pip build-env overlay does not inherit an
    # unset CUDA_HOME/ROCM_HOME, so derive the ROCm root and export it explicitly.
    _rocm_root = (
        _os.environ.get("ROCM_HOME")
        or _os.environ.get("CUDA_HOME")
        or _os.environ.get("ROCM_PATH")
        or _os.environ.get("HIP_PATH")
        or "/opt/rocm"
    ).strip() or "/opt/rocm"
    install_env = {
        **_os.environ,
        "PYTORCH_ROCM_ARCH": gpu_arch,
        "MAX_JOBS": str(max_jobs),
        "CUDA_HOME": _rocm_root,
        "ROCM_HOME": _rocm_root,
        "ROCM_PATH": _rocm_root,
        "HIP_HOME": _rocm_root,
    }

    # ``--no-build-isolation`` (below) means pip will NOT install the checkout's
    # [build-system].requires — they must already be in the attempt venv. Pre-
    # install them (minus any ``torch`` pin, which would clobber the ROCm torch
    # the venv was built against). Without this, a checkout that pins e.g.
    # setuptools-scm / setuptools-rust / a different torch fails PEP517 metadata
    # prep with the cryptic ``OSError ... output.json: No such file or directory``.
    try:
        build_requires = _read_build_system_requires(worktree_dir)
        if build_requires:
            run_argv(
                [attempt_py, "-m", "pip", "install", *build_requires],
                cwd=str(worktree_dir), env=install_env, timeout_sec=1800, run=_run,
            )
    except Exception:  # noqa: BLE001 — best-effort; the editable install still runs
        import logging as _logmod
        _logmod.getLogger(__name__).debug(
            "vLLM source build: build-requires pre-install skipped", exc_info=True
        )

    # pip install -e triggers CMake build_ext. ``--no-build-isolation`` keeps the
    # build in the attempt venv (which has the pinned ROCm torch + numpy) so
    # setup.py sees torch/numpy and the exported CUDA_HOME/ROCM_HOME.
    pip_cmd = [
        attempt_py, "-m", "pip", "install",
        "--no-build-isolation",
        "--constraint", str(constraint_path),
        "-e", str(worktree_dir),
    ]
    pip_res = run_argv(pip_cmd, cwd=str(worktree_dir), env=install_env, timeout_sec=5400, run=_run)
    if pip_res.returncode != 0:
        build_log.write_text(pip_res.stderr_tail or pip_res.stdout_tail, encoding="utf-8")
        return _fail("compile_error", f"vLLM source pip install failed (rc={pip_res.returncode})")

    # ROCm platform verify (port of verify_vllm_rocm from installer)
    vllm_verify = run_argv(
        [attempt_py, "-c", _VERIFY_VLLM_ROCM_SCRIPT],
        cwd=str(root), env=dict(_os.environ), timeout_sec=120, run=_run,
    )
    if vllm_verify.returncode != 0:
        return _fail("boot_failed", f"vLLM ROCm platform check failed: {vllm_verify.stderr_tail[:500]}")

    # Load probe: confirm the attempt venv loads vllm from the worktree.
    load_probe_script = (
        f"import vllm, inspect, sys; f = inspect.getfile(vllm); print(f); "
        f"sys.exit(0 if f.startswith({str(worktree_dir)!r}) else 3)"
    )
    load_probe = run_argv(
        [attempt_py, "-c", load_probe_script],
        cwd=str(root), env=dict(_os.environ), timeout_sec=60, run=_run,
    )
    if load_probe.returncode != 0:
        return _fail(
            "boot_failed",
            f"vLLM load probe failed: attempt venv does not load vllm from worktree "
            f"(rc={load_probe.returncode}; out={load_probe.stdout_tail[:200]})",
        )

    # Artifact freshness (fresh _C*.so means the extension was compiled)
    expected_artifacts = list(action.expected_artifacts) or ["vllm/_C*.so", "**/_C*.so"]
    freshness = verify_fresh_artifacts(str(worktree_dir), since_unix, expected_artifacts)
    built_paths: tuple[str, ...] = tuple(freshness.get("fresh", []))

    if action.expected_symbols:
        sym_result = verify_symbols(attempt_py, list(action.expected_symbols), run=_run)
        if not sym_result["verified"]:
            return _fail("symbol_missing", f"symbols not importable: {sym_result['missing']}")

    git_run = git if git is not None else _run
    sha_res = git_run(["git", *safe_directory_args(["-C", str(worktree_dir), "rev-parse", "--short", "HEAD"])],
                      capture_output=True, text=True, timeout=30)
    commit_sha = (getattr(sha_res, "stdout", "") or "").strip()

    installed_versions = {
        "torch": abi.get("torch_version", ""),
        "vllm_ref": ref,
        "vllm_sha": commit_sha,
        "arch": gpu_arch,
        "hip_version": abi.get("hip_version", ""),
    }
    if action.source_pr_url:
        installed_versions["source_pr_url"] = action.source_pr_url

    # vLLM source overlay: prepend the worktree so the attempt venv's vllm wins.
    runtime = FrameworkRuntime(
        pythonpath_prefixes=(str(worktree_dir),),
        entrypoint_bin_dir=str(venv_dir / "bin") if venv_dir else "",
        runtime_python_exe=attempt_py,
        runtime_env={"PYTORCH_ROCM_ARCH": gpu_arch},
        source_root=str(worktree_dir),
        attempt_root=str(root),
    )
    return BuildResult(
        ok=True,
        attempt_root=str(root),
        runtime=runtime,
        built_artifacts=built_paths,
        installed_versions=installed_versions,
        build_log_path=str(build_log),
        failure_class="ok",
    )


# ---------------------------------------------------------------------------
# Off-loop driver entrypoint
# ---------------------------------------------------------------------------

def _driver_main(argv: list[str] | None = None) -> int:
    """Driver subprocess entry: load plan.json, call run_aiter_build, write result.json."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Off-loop targeted-build driver")
    parser.add_argument("--attempt-root", required=True, help="Attempt directory")
    args = parser.parse_args(argv)

    root = Path(args.attempt_root)
    plan_path = root / "plan.json"
    result_path = root / "result.json"

    try:
        action = TargetedBuildAction.from_state(
            json.loads(plan_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:  # noqa: BLE001
        result_path.write_text(
            json.dumps(
                BuildResult(
                    ok=False,
                    attempt_root=str(root),
                    failure_class="compile_error",
                    failure_summary=f"failed to load plan.json: {exc!r}",
                    error=repr(exc),
                ).to_state()
            ),
            encoding="utf-8",
        )
        return 1

    dispatcher = {
        "aiter": run_aiter_build,
        "framework_ext": run_aiter_build,
        "sgl_kernel": run_sgl_kernel_build,
        "vllm_source": run_vllm_source_build,
    }
    recipe = dispatcher.get(action.component)
    if recipe is None:
        result_path.write_text(
            json.dumps(
                BuildResult(
                    ok=False,
                    attempt_root=str(root),
                    failure_class="compile_error",
                    failure_summary=f"unknown component: {action.component!r}",
                    error=f"no recipe for component {action.component!r}",
                ).to_state()
            ),
            encoding="utf-8",
        )
        return 1

    try:
        result = recipe(action, str(root))
    except Exception as exc:  # noqa: BLE001
        result = BuildResult(
            ok=False,
            attempt_root=str(root),
            failure_class="compile_error",
            failure_summary=f"recipe raised: {exc!r}",
            error=repr(exc),
        )

    result_path.write_text(json.dumps(result.to_state()), encoding="utf-8")
    return 0 if result.ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_driver_main())


__all__ = [
    "BuildHandle",
    "_driver_main",
    "_load_result_json",
    "default_budget_sec",
    "kill_build_pgroup",
    "poll_build",
    "run_aiter_build",
    "run_sgl_kernel_build",
    "run_vllm_source_build",
    "spawn_build",
]
