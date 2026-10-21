# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Off-loop targeted-build runner."""

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
from .build_utils import ROCM_CONSTRAINTS_FILE, sglang_rocm_install_target

log = logging.getLogger(__name__)

# Upper bound on confirming a SIGKILLed process group is gone.
_REAP_TIMEOUT_SEC = 5.0

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
    """In-memory handle for one in-flight detached build."""

    action: TargetedBuildAction
    attempt_root: str
    aiter_jit_dir: str
    build_log_path: str
    proc: Any
    pid: int
    pgid: int

    def to_sentinel(self, task_id: str) -> dict[str, Any]:
        """Project onto the ``pending_targeted_build`` sentinel dict."""
        return {
            "task_id": task_id,
            "pid": self.pid,
            "pgid": self.pgid,
            "attempt_root": self.attempt_root,
            "aiter_jit_dir": self.aiter_jit_dir,
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
) -> BuildHandle:
    """Spawn a targeted build as a detached process group."""
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


def classify_build_exit(handle: BuildHandle, rc: int) -> BuildResult:
    """Turn an exited build into a :class:`BuildResult`."""
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


def ensure_build_dead(handle: BuildHandle) -> bool:
    """SIGKILL the build's process group unless it already exited."""
    if handle.proc.poll() is not None:
        return True
    kill_build_pgroup(handle.pgid, sig=signal.SIGKILL)
    try:
        handle.proc.wait(timeout=_REAP_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        log.warning(
            "targeted_build: pgid=%d unreaped after SIGKILL; keeping the sentinel",
            handle.pgid,
        )
        return False
    return True


def _load_result_json(attempt_root: str) -> BuildResult | None:
    """Try to load a ``result.json`` written by the driver; return None on miss."""
    import json

    result_path = Path(attempt_root) / "result.json"
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return BuildResult.from_state(data)
    except (OSError, ValueError):
        return None


def _read_build_system_requires(worktree_dir: Any) -> list[str]:
    """Return a checkout's PEP 518 ``[build-system].requires`` for pre-install."""
    try:
        import tomllib  # py3.11+
    except ImportError:
        return []
    pyproject = Path(worktree_dir) / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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


# AITER real-build recipe

# Default AITER upstream; overridable via TargetedBuildAction.repo_url.
_AITER_DEFAULT_REPO = "https://github.com/ROCm/aiter"
# Headroom for a full AITER compile + build cache (GB).
_AITER_DISK_PER_CANDIDATE_GB = 6.0
# Default max parallel compile jobs; can be overridden by action.max_jobs.
_AITER_DEFAULT_MAX_JOBS = 8

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


class _TargetedRecipe:
    """Shared setup for an isolated ROCm package build; the install step is the subclass's."""

    disk_gb: float
    default_repo: str
    default_max_jobs: int
    replacing: str
    run_timeout: int = 3600

    def __init__(
        self,
        action: TargetedBuildAction,
        attempt_root: str,
        *,
        run: Callable[..., Any],
        git: Callable[..., Any] | None,
        disk_preflight_fn: Callable[..., Any] | None,
    ) -> None:
        from .build_utils import (
            AbiMismatchError,
            check_rocm_toolchain_alignment,
            probe_torch_abi,
            run_argv,
            write_rocm_constraints,
        )

        self._AbiMismatchError = AbiMismatchError
        self._check_rocm = check_rocm_toolchain_alignment
        self._probe_abi = probe_torch_abi
        self._run_argv = run_argv
        self._write_constraints = write_rocm_constraints
        self.action = action
        self.root = Path(attempt_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.build_log = self.root / "build.log"
        self._run_raw = run
        self._git = git
        self._disk_preflight_fn = disk_preflight_fn
        self.result: BuildResult | None = None
        self.abi: dict[str, Any] = {}
        self.worktree_dir: Path
        self.venv_dir: Path
        self.attempt_py = ""
        self.repo_url = ""
        self.ref = ""

    def _run(self, argv: Any, **kw: Any) -> Any:
        kw.setdefault("capture_output", True)
        kw.setdefault("text", True)
        kw.setdefault("timeout", self.run_timeout)
        return self._run_raw(argv, **kw)

    def _fail(self, failure_class: str, summary: str) -> BuildResult:
        return BuildResult(
            ok=False,
            attempt_root=str(self.root),
            build_log_path=str(self.build_log),
            failure_class=failure_class,
            failure_summary=summary,
            error=summary,
        )

    def preflight(self) -> None:
        """Disk and toolchain checks; sets ``result`` on failure."""
        import os as _os
        import sys as _sys

        try:
            if self._disk_preflight_fn is not None:
                self._disk_preflight_fn(self.root, 1, per_candidate_gb=self.disk_gb)
            else:
                from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

                try:
                    disk_preflight(self.root, 1, per_candidate_gb=self.disk_gb)
                except DiskPreflightError as exc:
                    self.result = self._fail("preflight_disk", str(exc))
                    return
        except (OSError, ValueError) as exc:
            self.result = self._fail("preflight_disk", f"disk preflight raised: {exc!r}")
            return

        tc_ok, tc_msg = self._check_rocm(env=dict(_os.environ), run=self._run)
        if not tc_ok:
            self.result = self._fail("preflight_toolchain", tc_msg)
            return
        self.abi = self._probe_abi(_sys.executable, run=self._run)
        if not self.abi.get("is_rocm"):
            self.result = self._fail(
                "preflight_toolchain",
                f"host torch is not a ROCm build (torch={self.abi.get('torch_version')})",
            )

    def prepare(self) -> None:
        """Isolation workspace and ROCm constraints; sets ``result`` on failure."""
        self.repo_url = str(self.action.repo_url or self.default_repo).strip() or self.default_repo
        self.ref = str(self.action.ref or "").strip()
        try:
            from hyperloom.agents.framework.isolation import prepare_candidate_workspace, prepare_repo_cache
            from hyperloom.agents.framework.models import Candidate

            prepare_repo_cache(self.repo_url, self.root)
            candidate = Candidate(ref=self.ref or "HEAD", repo=self.repo_url)
            ws = prepare_candidate_workspace(candidate, repo_url=self.repo_url, work_dir=self.root, index=0)
            self.worktree_dir = ws.worktree_dir
            self.venv_dir = ws.venv_dir
            self.attempt_py = str(self.venv_dir / "bin" / "python")
        except (OSError, subprocess.SubprocessError) as exc:
            self.result = self._fail("compile_error", f"workspace preparation failed: {exc!r}")
            return

        try:
            self._write_constraints(
                self.attempt_py, str(self.root / ROCM_CONSTRAINTS_FILE), replacing=self.replacing, run=self._run
            )
        except self._AbiMismatchError as exc:
            self.result = self._fail("abi_mismatch", str(exc))
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            self.result = self._fail("preflight_toolchain", f"ROCm constraint probe failed: {exc!r}")

    def installed_versions(self, prefix: str, selected_ref: str, gpu_arch: str) -> dict[str, str]:
        git_run = self._git if self._git is not None else self._run
        sha_res = git_run(
            ["git", *safe_directory_args(["-C", str(self.worktree_dir), "rev-parse", "--short", "HEAD"])],
            capture_output=True,
            text=True,
            timeout=30,
        )
        versions = {
            "torch": self.abi.get("torch_version", ""),
            f"{prefix}_ref": selected_ref,
            f"{prefix}_sha": (getattr(sha_res, "stdout", "") or "").strip(),
            "arch": gpu_arch,
            "hip_version": self.abi.get("hip_version", ""),
        }
        if self.action.source_pr_url:
            versions["source_pr_url"] = self.action.source_pr_url
        return versions


class _AiterBuild(_TargetedRecipe):
    """Isolated AITER compile: shared setup, then pinned-ref or tag-descending install."""

    disk_gb = _AITER_DISK_PER_CANDIDATE_GB
    default_repo = _AITER_DEFAULT_REPO
    default_max_jobs = _AITER_DEFAULT_MAX_JOBS
    replacing = "amd-aiter"

    def run(self) -> BuildResult:
        self.jit_dir = self.root / "aiter_jit"
        self.jit_dir.mkdir(parents=True, exist_ok=True)
        self.aiter_home = self.root / "home"
        self.aiter_home.mkdir(parents=True, exist_ok=True)
        self.preflight()
        if self.result is not None:
            return self.result
        self.prepare()
        if self.result is not None:
            return self.result
        return self.install()

    def install(self) -> BuildResult:
        import os as _os

        from .build_utils import sort_tags_desc

        gpu_arch = str(self.action.gpu_arch or "").strip()
        max_jobs = int(self.action.max_jobs or self.default_max_jobs)
        env = {
            **_os.environ,
            "AITER_ROOT_DIR": str(self.aiter_home),
            "HOME": str(self.aiter_home),
            "INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(self.jit_dir),
            "AITER_REBUILD": "1",
        }
        if gpu_arch:
            env["PYTORCH_ROCM_ARCH"] = gpu_arch
            env["AITER_ROCM_ARCH"] = gpu_arch
        if max_jobs:
            env["MAX_JOBS"] = str(max_jobs)
        pip = [
            self.attempt_py,
            "-m",
            "pip",
            "install",
            "--constraint",
            str(self.root / ROCM_CONSTRAINTS_FILE),
            "--config-settings",
            "editable_mode=compat",
            "-e",
            str(self.worktree_dir),
        ]
        if self.ref:
            return self._pinned(pip, env, gpu_arch)
        return self._autoselect(pip, env, gpu_arch, sort_tags_desc)

    def _ok(self, selected_ref: str, gpu_arch: str) -> BuildResult:
        return BuildResult(
            ok=True,
            attempt_root=str(self.root),
            runtime=FrameworkRuntime(
                pythonpath_prefixes=(str(self.worktree_dir),),
                runtime_env={"INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(self.jit_dir)},
                source_root=str(self.worktree_dir),
                attempt_root=str(self.root),
            ),
            installed_versions=self.installed_versions("aiter", selected_ref, gpu_arch),
            build_log_path=str(self.build_log),
            failure_class="ok",
        )

    def _pinned(self, pip: list[str], env: dict[str, str], gpu_arch: str) -> BuildResult:
        res = self._run_argv(pip, cwd=str(self.worktree_dir), env=env, timeout_sec=3600, run=self._run)
        if res.returncode != 0:
            self.build_log.write_text(res.stderr_tail or res.stdout_tail, encoding="utf-8")
            return self._fail("compile_error", f"pip install failed for ref={self.ref!r}: rc={res.returncode}")
        probe = self._run_argv(
            [self.attempt_py, "-c", "import aiter"],
            cwd=str(self.root),
            env=env,
            timeout_sec=60,
            run=self._run,
        )
        if probe.returncode != 0:
            return self._fail(
                "boot_failed",
                f"aiter not importable after pip install (ref={self.ref!r}): {probe.stderr_tail or probe.stdout_tail}",
            )
        return self._ok(self.ref, gpu_arch)

    def _autoselect(
        self, pip: list[str], env: dict[str, str], gpu_arch: str, sort_tags_desc: Callable[..., list[str]]
    ) -> BuildResult:
        git_run = self._git if self._git is not None else self._run
        tags_res = git_run(
            ["git", *safe_directory_args(["-C", str(self.worktree_dir), "tag", "-l", "v*"])],
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw_tags = (getattr(tags_res, "stdout", "") or "").strip().splitlines()
        tags = sort_tags_desc([t.strip() for t in raw_tags if t.strip()])
        if not tags:
            return self._fail("compile_error", "no AITER version tags found in the cloned repo")
        for tag in tags:
            checkout_res = git_run(
                ["git", *safe_directory_args(["-C", str(self.worktree_dir), "checkout", tag])],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if getattr(checkout_res, "returncode", 1) != 0:
                continue
            res = self._run_argv(pip, cwd=str(self.worktree_dir), env=env, timeout_sec=3600, run=self._run)
            if res.returncode != 0:
                continue
            probe = self._run_argv(
                [self.attempt_py, "-c", "import aiter"],
                cwd=str(self.root),
                env=env,
                timeout_sec=60,
                run=self._run,
            )
            if probe.returncode == 0:
                return self._ok(tag, gpu_arch)
        return self._fail("compile_error", "no AITER tag installed and imported successfully")


def run_aiter_build(
    action: TargetedBuildAction,
    attempt_root: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[..., Any] | None = None,
    disk_preflight_fn: Callable[..., Any] | None = None,
) -> BuildResult:
    """Run a full isolated AITER targeted build."""
    build = _AiterBuild(action, attempt_root, run=run, git=git, disk_preflight_fn=disk_preflight_fn)
    return build.run()


# sgl-kernel recipe

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
    """Run an isolated sgl-kernel targeted build."""
    import os as _os
    import sys as _sys

    from .build_utils import (
        AbiMismatchError,
        check_rocm_toolchain_alignment,
        probe_torch_abi,
        run_argv,
        write_rocm_constraints,
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
    except (OSError, ValueError) as exc:
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
        from hyperloom.agents.framework.models import Candidate

        prepare_repo_cache(repo_url, root)
        candidate = Candidate(ref=ref or "HEAD", repo=repo_url)
        ws = prepare_candidate_workspace(candidate, repo_url=repo_url, work_dir=root, index=0)
        worktree_dir = ws.worktree_dir
        venv_dir = ws.venv_dir
        attempt_py = str(venv_dir / "bin" / "python")
    except (OSError, subprocess.SubprocessError) as exc:
        return _fail("compile_error", f"workspace preparation failed: {exc!r}")

    constraint_path = root / ROCM_CONSTRAINTS_FILE
    try:
        write_rocm_constraints(attempt_py, str(constraint_path), replacing="sglang-kernel", run=_run)
    except AbiMismatchError as exc:
        return _fail("abi_mismatch", str(exc))
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return _fail("preflight_toolchain", f"ROCm constraint probe failed: {exc!r}")

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

    try:
        with sglang_rocm_install_target(worktree_dir / "python", abi["torch_version"]) as target:
            pip_cmd = [attempt_py, "-m", "pip", "install", "--constraint", str(constraint_path), "-e", target]
            pip_res = run_argv(pip_cmd, cwd=str(root), env=dict(_os.environ), timeout_sec=3600, run=_run)
    except (OSError, ValueError) as exc:
        return _fail("compile_error", f"no ROCm install of the SGLang checkout: {exc}")
    if pip_res.returncode != 0:
        build_log.write_text(pip_res.stderr_tail or pip_res.stdout_tail, encoding="utf-8")
        return _fail("compile_error", f"sgl-kernel pip install failed (rc={pip_res.returncode})")

    probe = run_argv(
        [attempt_py, "-c", "import sgl_kernel"],
        cwd=str(root),
        env=dict(_os.environ),
        timeout_sec=60,
        run=_run,
    )
    if probe.returncode != 0:
        return _fail(
            "boot_failed",
            f"sgl_kernel not importable after pip install: {probe.stderr_tail or probe.stdout_tail}",
        )

    git_run = git if git is not None else _run
    sha_res = git_run(
        ["git", *safe_directory_args(["-C", str(worktree_dir), "rev-parse", "--short", "HEAD"])],
        capture_output=True,
        text=True,
        timeout=30,
    )
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
        installed_versions=installed_versions,
        build_log_path=str(build_log),
        failure_class="ok",
    )


# vLLM from source recipe


class _VllmBuild(_TargetedRecipe):
    """Isolated vLLM-from-source compile: shared setup, then pip + ROCm probes."""

    disk_gb = _VLLM_DISK_PER_CANDIDATE_GB
    default_repo = _VLLM_DEFAULT_REPO
    default_max_jobs = _VLLM_DEFAULT_MAX_JOBS
    replacing = "vllm"
    run_timeout = 5400

    def run(self) -> BuildResult:
        import sys as _sys

        self.preflight()
        if self.result is not None:
            return self.result
        gpu_arch = str(self.action.gpu_arch or "").strip()
        if not gpu_arch:
            return self._fail("preflight_toolchain", "gpu_arch must be set explicitly for vLLM source (L6)")
        host_pyver = f"{_sys.version_info.major}.{_sys.version_info.minor}"
        abi_pyver = str(self.abi.get("python_version") or "").strip()
        if abi_pyver and not abi_pyver.startswith(host_pyver):
            import logging as _log

            _log.getLogger(__name__).info(
                "vLLM source build: torch ABI python %s != host %s; "
                "runtime_python_exe will be set to the attempt venv interpreter",
                abi_pyver,
                host_pyver,
            )
        self.prepare()
        if self.result is not None:
            return self.result
        return self.install(gpu_arch)

    def install(self, gpu_arch: str) -> BuildResult:
        import os as _os

        max_jobs = int(self.action.max_jobs or self.default_max_jobs)
        rocm_root = (
            _os.environ.get("ROCM_HOME")
            or _os.environ.get("CUDA_HOME")
            or _os.environ.get("ROCM_PATH")
            or _os.environ.get("HIP_PATH")
            or "/opt/rocm"
        ).strip() or "/opt/rocm"
        env = {
            **_os.environ,
            "PYTORCH_ROCM_ARCH": gpu_arch,
            "MAX_JOBS": str(max_jobs),
            "CUDA_HOME": rocm_root,
            "ROCM_HOME": rocm_root,
            "ROCM_PATH": rocm_root,
            "HIP_HOME": rocm_root,
        }
        # ``--no-build-isolation`` (below) means pip will NOT install the checkout's [build-system].requires — they must
        # already be in the attempt venv.
        try:
            build_requires = _read_build_system_requires(self.worktree_dir)
            if build_requires:
                self._run_argv(
                    [self.attempt_py, "-m", "pip", "install", *build_requires],
                    cwd=str(self.worktree_dir),
                    env=env,
                    timeout_sec=1800,
                    run=self._run,
                )
        except OSError:
            import logging as _logmod

            _logmod.getLogger(__name__).debug("vLLM source build: build-requires pre-install skipped", exc_info=True)

        pip_cmd = [
            self.attempt_py,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--constraint",
            str(self.root / ROCM_CONSTRAINTS_FILE),
            "-e",
            str(self.worktree_dir),
        ]
        pip_res = self._run_argv(pip_cmd, cwd=str(self.worktree_dir), env=env, timeout_sec=5400, run=self._run)
        if pip_res.returncode != 0:
            self.build_log.write_text(pip_res.stderr_tail or pip_res.stdout_tail, encoding="utf-8")
            return self._fail("compile_error", f"vLLM source pip install failed (rc={pip_res.returncode})")
        return self._probed_success(gpu_arch)

    def _probed_success(self, gpu_arch: str) -> BuildResult:
        import os as _os

        # The probes import vllm the way the runtime will: the editable finder runs
        # after sys.path, where the base image's system-site vllm would otherwise win.
        pythonpath_prefixes = (str(self.worktree_dir),)
        pythonpath = _os.pathsep.join(p for p in (*pythonpath_prefixes, _os.environ.get("PYTHONPATH", "")) if p)
        probe_env = {**_os.environ, "PYTHONPATH": pythonpath}

        vllm_verify = self._run_argv(
            [self.attempt_py, "-c", _VERIFY_VLLM_ROCM_SCRIPT],
            cwd=str(self.root),
            env=probe_env,
            timeout_sec=120,
            run=self._run,
        )
        if vllm_verify.returncode != 0:
            return self._fail("boot_failed", f"vLLM ROCm platform check failed: {vllm_verify.stderr_tail[:500]}")

        load_probe_script = (
            f"import vllm, inspect, sys; f = inspect.getfile(vllm); print(f); "
            f"sys.exit(0 if f.startswith({str(self.worktree_dir)!r}) else 3)"
        )
        load_probe = self._run_argv(
            [self.attempt_py, "-c", load_probe_script],
            cwd=str(self.root),
            env=probe_env,
            timeout_sec=60,
            run=self._run,
        )
        if load_probe.returncode != 0:
            return self._fail(
                "boot_failed",
                f"vLLM load probe failed: attempt venv does not load vllm from worktree "
                f"(rc={load_probe.returncode}; out={load_probe.stdout_tail[:200]})",
            )

        return BuildResult(
            ok=True,
            attempt_root=str(self.root),
            runtime=FrameworkRuntime(
                pythonpath_prefixes=pythonpath_prefixes,
                entrypoint_bin_dir=str(self.venv_dir / "bin"),
                runtime_python_exe=self.attempt_py,
                runtime_env={"PYTORCH_ROCM_ARCH": gpu_arch},
                source_root=str(self.worktree_dir),
                attempt_root=str(self.root),
            ),
            installed_versions=self.installed_versions("vllm", self.ref, gpu_arch),
            build_log_path=str(self.build_log),
            failure_class="ok",
        )


def run_vllm_source_build(
    action: TargetedBuildAction,
    attempt_root: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[..., Any] | None = None,
    disk_preflight_fn: Callable[..., Any] | None = None,
) -> BuildResult:
    """Run an isolated vLLM-from-source targeted build."""
    build = _VllmBuild(action, attempt_root, run=run, git=git, disk_preflight_fn=disk_preflight_fn)
    return build.run()


# Off-loop driver entrypoint


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
        action = TargetedBuildAction.from_state(json.loads(plan_path.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        log.warning("targeted build driver: failed to load plan.json: %s", exc)
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
        log.warning("targeted build driver: recipe raised: %s", exc)
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
    "classify_build_exit",
    "default_budget_sec",
    "ensure_build_dead",
    "kill_build_pgroup",
    "run_aiter_build",
    "run_sgl_kernel_build",
    "run_vllm_source_build",
    "spawn_build",
]
