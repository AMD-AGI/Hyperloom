# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Off-loop targeted-build runner."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys

from hyperloom_common.git_safety import safe_directory_args
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .build_actions import BuildResult, FrameworkRuntime, TargetedBuildAction
from .build_utils import ROCM_CONSTRAINTS_FILE, VLLM_ROCM_PROBE, sglang_rocm_install_target

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
            "pgid": self.pgid,
            "attempt_root": self.attempt_root,
            "aiter_jit_dir": self.aiter_jit_dir,
        }


def spawn_build(
    action: TargetedBuildAction,
    *,
    attempt_root: str,
    command: list[str],
    run: Callable[..., Any] = subprocess.Popen,
) -> BuildHandle:
    """Spawn a targeted build as a detached process group."""
    root = Path(attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    jit_dir = root / "aiter_jit"
    jit_dir.mkdir(parents=True, exist_ok=True)
    build_log = root / "build.log"

    env = dict(os.environ)
    env["INFERENCE_OPTIMIZER_AITER_JIT_DIR"] = str(jit_dir)

    log_fh = build_log.open("w", encoding="utf-8")
    try:
        proc = run(
            command,
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
        default_budget_sec(action.component),
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


def classify_build_exit(handle: BuildHandle, rc: int) -> BuildResult:
    """Turn an exited build into a :class:`BuildResult`; the driver writes ``result.json`` before it exits 0."""
    rich = _load_result_json(handle.attempt_root)
    if rich is not None:
        return rich
    summary = f"build command exited {int(rc)}"
    return BuildResult(
        ok=False,
        attempt_root=handle.attempt_root,
        build_log_path=handle.build_log_path,
        failure_class="compile_error",
        failure_summary=summary,
        error=summary,
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
# Max parallel compile jobs.
_AITER_DEFAULT_MAX_JOBS = 8

_VLLM_DEFAULT_REPO = "https://github.com/ROCm/vllm"
_VLLM_DISK_PER_CANDIDATE_GB = 20.0  # raised headroom for the full vLLM build cache
_VLLM_DEFAULT_MAX_JOBS = 8


class _TargetedRecipe:
    """Shared setup for an isolated ROCm package build; the install step is the subclass's."""

    disk_gb: float
    default_repo: str
    default_max_jobs: int
    replacing: str
    gpu_arch_error: str | None

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
            run_argv as run_argv_impl,
            write_rocm_constraints,
        )

        self._AbiMismatchError = AbiMismatchError
        self._check_rocm = check_rocm_toolchain_alignment
        self._probe_abi = probe_torch_abi
        self._run_argv_impl = run_argv_impl
        self._write_constraints = write_rocm_constraints
        self.action = action
        self.root = Path(attempt_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.build_log = self.root / "build.log"
        self._run = run
        self._git = git
        self._disk_preflight_fn = disk_preflight_fn
        self.abi: dict[str, Any] = {}
        self.worktree_dir: Path
        self.venv_dir: Path
        self.attempt_py = ""
        self.repo_url = ""
        self.ref = ""

    def _run_argv(self, argv: Any, **kw: Any) -> Any:
        return self._run_argv_impl(argv, run=self._run, **kw)

    @property
    def gpu_arch(self) -> str:
        return str(self.action.gpu_arch or "").strip()

    def _fail(self, failure_class: str, summary: str) -> BuildResult:
        return BuildResult(
            ok=False,
            attempt_root=str(self.root),
            build_log_path=str(self.build_log),
            failure_class=failure_class,
            failure_summary=summary,
            error=summary,
        )

    def preflight(self) -> BuildResult | None:
        """Disk and toolchain checks; returns the failed result, or ``None`` when they pass."""
        try:
            if self._disk_preflight_fn is not None:
                self._disk_preflight_fn(self.root, 1, per_candidate_gb=self.disk_gb)
            else:
                from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

                try:
                    disk_preflight(self.root, 1, per_candidate_gb=self.disk_gb)
                except DiskPreflightError as exc:
                    return self._fail("preflight_disk", str(exc))
        except (OSError, ValueError) as exc:
            return self._fail("preflight_disk", f"disk preflight raised: {exc!r}")

        tc_ok, tc_msg = self._check_rocm(env=dict(os.environ), run=self._run)
        if not tc_ok:
            return self._fail("preflight_toolchain", tc_msg)
        self.abi = self._probe_abi(sys.executable, run=self._run)
        if not self.abi.get("is_rocm"):
            return self._fail(
                "preflight_toolchain",
                f"host torch is not a ROCm build (torch={self.abi.get('torch_version')})",
            )
        if self.gpu_arch_error is not None and not self.gpu_arch:
            return self._fail("preflight_toolchain", self.gpu_arch_error)
        return None

    def prepare(self) -> BuildResult | None:
        """Isolation workspace and ROCm constraints; returns the failed result, or ``None`` when ready."""
        self.repo_url = str(self.action.repo_url or self.default_repo).strip() or self.default_repo
        self.ref = str(self.action.ref or "").strip()
        try:
            from hyperloom.agents.framework.isolation import prepare_candidate_workspace
            from hyperloom.agents.framework.models import Candidate

            candidate = Candidate(ref=self.ref or "HEAD")
            ws = prepare_candidate_workspace(candidate, repo_url=self.repo_url, work_dir=self.root, index=0)
            self.worktree_dir = ws.worktree_dir
            self.venv_dir = ws.venv_dir
            self.attempt_py = str(self.venv_dir / "bin" / "python")
        except (OSError, subprocess.SubprocessError) as exc:
            return self._fail("compile_error", f"workspace preparation failed: {exc!r}")

        try:
            self._write_constraints(
                self.attempt_py, str(self.root / ROCM_CONSTRAINTS_FILE), replacing=self.replacing, run=self._run
            )
        except self._AbiMismatchError as exc:
            return self._fail("abi_mismatch", str(exc))
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return self._fail("preflight_toolchain", f"ROCm constraint probe failed: {exc!r}")
        return None

    def installed_versions(self, prefix: str, selected_ref: str) -> dict[str, str]:
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
            "arch": self.gpu_arch,
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
    gpu_arch_error = None

    def run(self) -> BuildResult:
        self.jit_dir = self.root / "aiter_jit"
        self.jit_dir.mkdir(parents=True, exist_ok=True)
        self.aiter_home = self.root / "home"
        self.aiter_home.mkdir(parents=True, exist_ok=True)
        return self.preflight() or self.prepare() or self.install()

    def install(self) -> BuildResult:
        from .build_utils import sort_tags_desc

        gpu_arch = self.gpu_arch
        env = {
            **os.environ,
            "AITER_ROOT_DIR": str(self.aiter_home),
            "HOME": str(self.aiter_home),
            "INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(self.jit_dir),
            "AITER_REBUILD": "1",
            "MAX_JOBS": str(self.default_max_jobs),
        }
        if gpu_arch:
            env["PYTORCH_ROCM_ARCH"] = gpu_arch
            env["AITER_ROCM_ARCH"] = gpu_arch
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
            return self._pinned(pip, env)
        return self._autoselect(pip, env, sort_tags_desc)

    def _ok(self, selected_ref: str) -> BuildResult:
        return BuildResult(
            ok=True,
            attempt_root=str(self.root),
            runtime=FrameworkRuntime(
                pythonpath_prefixes=(str(self.worktree_dir),),
                runtime_env={"INFERENCE_OPTIMIZER_AITER_JIT_DIR": str(self.jit_dir)},
                source_root=str(self.worktree_dir),
                attempt_root=str(self.root),
            ),
            installed_versions=self.installed_versions("aiter", selected_ref),
            build_log_path=str(self.build_log),
            failure_class="ok",
        )

    def _pinned(self, pip: list[str], env: dict[str, str]) -> BuildResult:
        res = self._run_argv(pip, cwd=str(self.worktree_dir), env=env, timeout_sec=3600)
        if res.returncode != 0:
            self.build_log.write_text(res.stderr_tail or res.stdout_tail, encoding="utf-8")
            return self._fail("compile_error", f"pip install failed for ref={self.ref!r}: rc={res.returncode}")
        probe = self._run_argv(
            [self.attempt_py, "-c", "import aiter"],
            cwd=str(self.root),
            env=env,
            timeout_sec=60,
        )
        if probe.returncode != 0:
            return self._fail(
                "boot_failed",
                f"aiter not importable after pip install (ref={self.ref!r}): {probe.stderr_tail or probe.stdout_tail}",
            )
        return self._ok(self.ref)

    def _autoselect(self, pip: list[str], env: dict[str, str], sort_tags_desc: Callable[..., list[str]]) -> BuildResult:
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
            res = self._run_argv(pip, cwd=str(self.worktree_dir), env=env, timeout_sec=3600)
            if res.returncode != 0:
                continue
            probe = self._run_argv(
                [self.attempt_py, "-c", "import aiter"],
                cwd=str(self.root),
                env=env,
                timeout_sec=60,
            )
            if probe.returncode == 0:
                return self._ok(tag)
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


class _SglKernelBuild(_TargetedRecipe):
    """Isolated sgl-kernel compile and SGLang ROCm install."""

    disk_gb = _SGLANG_DISK_PER_CANDIDATE_GB
    default_repo = _SGLANG_DEFAULT_REPO
    default_max_jobs = _SGLANG_DEFAULT_MAX_JOBS
    replacing = "sglang-kernel"
    gpu_arch_error = "gpu_arch must be set explicitly for sgl-kernel (L6)"

    def run(self) -> BuildResult:
        return self.preflight() or self.prepare() or self.install()

    def install(self) -> BuildResult:
        install_env = {
            **os.environ,
            "AMDGPU_TARGET": self.gpu_arch,
            "MAX_JOBS": str(self.default_max_jobs),
        }
        kernel_build = self._run_argv(
            [self.attempt_py, "setup_rocm.py", "install"],
            cwd=str(self.worktree_dir / "sgl-kernel"),
            env=install_env,
            timeout_sec=3600,
        )
        if kernel_build.returncode != 0:
            self.build_log.write_text(kernel_build.stderr_tail or kernel_build.stdout_tail, encoding="utf-8")
            return self._fail("compile_error", f"sgl-kernel compile failed (rc={kernel_build.returncode})")

        try:
            with sglang_rocm_install_target(self.worktree_dir / "python", self.abi["torch_version"]) as target:
                constraint_path = str(self.root / ROCM_CONSTRAINTS_FILE)
                pip_cmd = [self.attempt_py, "-m", "pip", "install", "--constraint", constraint_path, "-e", target]
                pip_res = self._run_argv(
                    pip_cmd,
                    cwd=str(self.root),
                    env=dict(os.environ),
                    timeout_sec=3600,
                )
        except (OSError, ValueError) as exc:
            return self._fail("compile_error", f"no ROCm install of the SGLang checkout: {exc}")
        if pip_res.returncode != 0:
            self.build_log.write_text(pip_res.stderr_tail or pip_res.stdout_tail, encoding="utf-8")
            return self._fail("compile_error", f"sgl-kernel pip install failed (rc={pip_res.returncode})")

        probe = self._run_argv(
            [self.attempt_py, "-c", "import sgl_kernel"],
            cwd=str(self.root),
            env=dict(os.environ),
            timeout_sec=60,
        )
        if probe.returncode != 0:
            return self._fail(
                "boot_failed",
                f"sgl_kernel not importable after pip install: {probe.stderr_tail or probe.stdout_tail}",
            )

        return BuildResult(
            ok=True,
            attempt_root=str(self.root),
            runtime=FrameworkRuntime(
                pythonpath_prefixes=(str(self.worktree_dir / "python"),),
                entrypoint_bin_dir=str(self.venv_dir / "bin"),
                runtime_python_exe=self.attempt_py,
                runtime_env={"SGLANG_USE_AITER": "1"},
                source_root=str(self.worktree_dir),
                attempt_root=str(self.root),
            ),
            installed_versions=self.installed_versions("sgl_kernel", self.ref),
            build_log_path=str(self.build_log),
            failure_class="ok",
        )


def run_sgl_kernel_build(
    action: TargetedBuildAction,
    attempt_root: str,
    *,
    run: Callable[..., Any] = subprocess.run,
    git: Callable[..., Any] | None = None,
    disk_preflight_fn: Callable[..., Any] | None = None,
) -> BuildResult:
    """Run an isolated sgl-kernel targeted build."""
    build = _SglKernelBuild(action, attempt_root, run=run, git=git, disk_preflight_fn=disk_preflight_fn)
    return build.run()


# vLLM from source recipe


class _VllmBuild(_TargetedRecipe):
    """Isolated vLLM-from-source compile: shared setup, then pip + ROCm probes."""

    disk_gb = _VLLM_DISK_PER_CANDIDATE_GB
    default_repo = _VLLM_DEFAULT_REPO
    default_max_jobs = _VLLM_DEFAULT_MAX_JOBS
    replacing = "vllm"
    gpu_arch_error = "gpu_arch must be set explicitly for vLLM source (L6)"

    def run(self) -> BuildResult:
        failed = self.preflight()
        if failed is not None:
            return failed
        # ABI-match guard: log an advisory when the torch ABI reports a different Python version; the build continues and
        # runtime_python_exe points to the attempt venv python so the server uses the correct interpreter.
        host_pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        abi_pyver = str(self.abi.get("python_version") or "").strip()
        if abi_pyver and not abi_pyver.startswith(host_pyver):
            log.info(
                "vLLM source build: torch ABI python %s != host %s; "
                "runtime_python_exe will be set to the attempt venv interpreter",
                abi_pyver,
                host_pyver,
            )
        return self.prepare() or self.install()

    def install(self) -> BuildResult:
        gpu_arch = self.gpu_arch
        # vLLM's ROCm setup.py asserts ``CUDA_HOME is not set`` and reuses it as the toolchain root even on ROCm.
        rocm_root = (
            os.environ.get("ROCM_HOME")
            or os.environ.get("CUDA_HOME")
            or os.environ.get("ROCM_PATH")
            or os.environ.get("HIP_PATH")
            or "/opt/rocm"
        ).strip() or "/opt/rocm"
        env = {
            **os.environ,
            "PYTORCH_ROCM_ARCH": gpu_arch,
            "MAX_JOBS": str(self.default_max_jobs),
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
                )
        except OSError:
            log.debug("vLLM source build: build-requires pre-install skipped", exc_info=True)

        # pip install -e triggers CMake build_ext.
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
        pip_res = self._run_argv(pip_cmd, cwd=str(self.worktree_dir), env=env, timeout_sec=5400)
        if pip_res.returncode != 0:
            self.build_log.write_text(pip_res.stderr_tail or pip_res.stdout_tail, encoding="utf-8")
            return self._fail("compile_error", f"vLLM source pip install failed (rc={pip_res.returncode})")
        return self._probed_success()

    def _probed_success(self) -> BuildResult:
        gpu_arch = self.gpu_arch
        # The probes import vllm the way the runtime will: the editable finder runs
        # after sys.path, where the base image's system-site vllm would otherwise win.
        pythonpath_prefixes = (str(self.worktree_dir),)
        pythonpath = os.pathsep.join(p for p in (*pythonpath_prefixes, os.environ.get("PYTHONPATH", "")) if p)
        probe_env = {**os.environ, "PYTHONPATH": pythonpath}

        vllm_verify = self._run_argv(
            [self.attempt_py, "-c", VLLM_ROCM_PROBE],
            cwd=str(self.root),
            env=probe_env,
            timeout_sec=120,
        )
        if vllm_verify.returncode != 0:
            return self._fail("boot_failed", f"vLLM ROCm platform check failed: {vllm_verify.stderr_tail[:500]}")

        # Load probe: confirm the attempt venv loads vllm from the worktree.
        load_probe_script = (
            f"import vllm, inspect, sys; f = inspect.getfile(vllm); print(f); "
            f"sys.exit(0 if f.startswith({str(self.worktree_dir)!r}) else 3)"
        )
        load_probe = self._run_argv(
            [self.attempt_py, "-c", load_probe_script],
            cwd=str(self.root),
            env=probe_env,
            timeout_sec=60,
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
            installed_versions=self.installed_versions("vllm", self.ref),
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
    """Driver subprocess entry: load plan.json, run the component's recipe, write result.json."""
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
    except (OSError, ValueError) as exc:
        log.warning("targeted build driver: failed to load plan.json: %s", exc, exc_info=True)
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
    recipe = dispatcher[action.component]
    try:
        result = recipe(action, str(root))
    except Exception as exc:  # outermost process entry: the failure is written to result.json
        log.warning("targeted build driver: recipe raised: %s", exc, exc_info=True)
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
