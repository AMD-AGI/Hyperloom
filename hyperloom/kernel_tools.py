"""Kernel optimization tools — GEAK, OOB, patch application.

Wraps the actual CLI invocations for kernel optimization backends.
These are called by the kernel specialist agent. The invocation
patterns are preserved exactly from kernel-agent/tools/backends/.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cluster import RayJobResult, submit_gpu_task

log = logging.getLogger(__name__)


@dataclass
class KernelOptResult:
    """Result from a kernel optimization run."""

    success: bool = False
    backend: str = ""
    patches: list[str] = field(default_factory=list)
    speedup: float = 0.0
    output_dir: str = ""
    stdout: str = ""
    stderr: str = ""


# ─── GEAK ────────────────────────────────────────────────────────────────────


def find_geak_binary() -> str | None:
    """Locate the GEAK binary on PATH."""
    for name in ("geak", "mini", "geak-gaagent"):
        path = shutil.which(name)
        if path:
            return path
    return None


def run_geak(
    prompt_file: str,
    output_dir: str,
    gpu_ids: str = "0",
    kernel_path: str | None = None,
    kernel_repo: str | None = None,
    test_command: str | None = None,
    cost_limit: float | None = None,
    timeout_s: int = 7800,
    num_gpus: int = 1,
    use_ray: bool = False,
) -> KernelOptResult:
    """Run GEAK kernel optimization.

    Invocation pattern preserved from kernel-agent/tools/backends/geak_submit.py.
    """
    geak_bin = find_geak_binary()
    if not geak_bin:
        log.error("GEAK binary not found on PATH")
        return KernelOptResult(backend="geak", stderr="geak not found")

    config_path = _resolve_geak_config()

    cmd = [
        geak_bin, "-t", prompt_file, "--yolo",
        "--output", output_dir, "--gpu-ids", gpu_ids,
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if kernel_path:
        cmd.extend(["--kernel-path", kernel_path])
    if kernel_repo:
        cmd.extend(["--repo", kernel_repo])
    if test_command:
        cmd.extend(["--test-command", test_command])
    if cost_limit is not None:
        cmd.extend(["--cost-limit", str(cost_limit)])

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if use_ray:
        result = submit_gpu_task(cmd, num_gpus=num_gpus, timeout_s=timeout_s, cwd=output_dir)
        return _parse_geak_result(output_dir, result.stdout, result.stderr, result.success)

    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = gpu_ids
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, env=env, cwd=output_dir,
        )
        return _parse_geak_result(output_dir, proc.stdout, proc.stderr, proc.returncode == 0)
    except subprocess.TimeoutExpired:
        return KernelOptResult(backend="geak", stderr="timeout", output_dir=output_dir)


def _resolve_geak_config() -> str:
    """Resolve GEAK config file path."""
    config = os.environ.get("GEAK_CONFIG", "")
    if config and Path(config).exists():
        return config
    return ""


def _parse_geak_result(output_dir: str, stdout: str, stderr: str, success: bool) -> KernelOptResult:
    """Parse GEAK output directory for patches."""
    patches = []
    out_path = Path(output_dir)
    for patch_file in out_path.rglob("*.patch"):
        patches.append(str(patch_file))
    for py_file in out_path.rglob("optimized_*.py"):
        patches.append(str(py_file))

    return KernelOptResult(
        success=success and len(patches) > 0,
        backend="geak",
        patches=patches,
        output_dir=output_dir,
        stdout=stdout,
        stderr=stderr,
    )


# ─── OOB (Claude/Codex/Cursor) ──────────────────────────────────────────────


def run_oob(
    agent: str,
    prompt_file: str,
    output_dir: str,
    source_file: str | None = None,
    extra_files: list[str] | None = None,
    max_turns: int = 100,
    timeout_s: int = 1800,
    num_gpus: int = 1,
    gpu_ids: str = "0",
    use_ray: bool = False,
) -> KernelOptResult:
    """Run OOB (out-of-band) agent for kernel optimization.

    Invocation pattern preserved from kernel-agent/tools/backends/oob_submit.py.
    """
    oob_bin = shutil.which("oob")
    if not oob_bin:
        log.error("OOB binary not found on PATH")
        return KernelOptResult(backend=f"oob-{agent}", stderr="oob not found")

    cmd = [
        oob_bin, "run", "-a", agent,
        "--prompt-file", prompt_file,
        "--max-turns", str(max_turns),
        "--timeout", str(timeout_s),
        "--json", "--no-live",
        "-o", output_dir,
    ]
    if source_file:
        cmd.extend(["-f", source_file])
    for ef in extra_files or []:
        cmd.extend(["-f", ef])

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if use_ray:
        result = submit_gpu_task(cmd, num_gpus=num_gpus, timeout_s=timeout_s, cwd=output_dir)
        return _parse_oob_result(output_dir, agent, result.stdout, result.stderr, result.success)

    env = os.environ.copy()
    env["ROCR_VISIBLE_DEVICES"] = gpu_ids
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_s, env=env, cwd=output_dir,
        )
        return _parse_oob_result(output_dir, agent, proc.stdout, proc.stderr, proc.returncode == 0)
    except subprocess.TimeoutExpired:
        return KernelOptResult(backend=f"oob-{agent}", stderr="timeout", output_dir=output_dir)


def _parse_oob_result(output_dir: str, agent: str, stdout: str, stderr: str, success: bool) -> KernelOptResult:
    """Parse OOB output directory for patches."""
    patches = []
    out_path = Path(output_dir)
    for patch_file in out_path.rglob("*.patch"):
        patches.append(str(patch_file))
    for diff_file in out_path.rglob("*.diff"):
        patches.append(str(diff_file))

    return KernelOptResult(
        success=success,
        backend=f"oob-{agent}",
        patches=patches,
        output_dir=output_dir,
        stdout=stdout,
        stderr=stderr,
    )


# ─── Patch Application ───────────────────────────────────────────────────────


def apply_patch(patch_path: str, target_dir: str, dry_run: bool = False) -> bool:
    """Apply a patch file to a target directory."""
    cmd = ["git", "apply"]
    if dry_run:
        cmd.append("--check")
    cmd.append(patch_path)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=target_dir)
    if result.returncode != 0:
        log.warning("Patch apply failed: %s", result.stderr[:200])
        return False
    return True


def revert_patch(patch_path: str, target_dir: str) -> bool:
    """Revert a previously applied patch."""
    cmd = ["git", "apply", "--reverse", patch_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=target_dir)
    return result.returncode == 0
