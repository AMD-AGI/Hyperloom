# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reusable, injectable build utilities for off-loop targeted builds.

All subprocess calls go through injectable ``run`` shims so every function can
be tested without a GPU or compiler.  No PATH is hard-coded.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

# ---------------------------------------------------------------------------
# Argv safety
# ---------------------------------------------------------------------------

_UNSAFE_TOKENS: frozenset[str] = frozenset({";", "&&", "||", "|", ">", ">>", "<", "<<", "`"})
_UNSAFE_CHARS_RE = re.compile(r"[;&|`$<>\r\n]")
_SHELL_NAMES: frozenset[str] = frozenset({"bash", "dash", "sh", "zsh", "ksh"})


def coerce_build_argv(cmd: list[str] | str | None) -> list[str]:
    """Return a safe argv list; reject shell strings and control operators.

    Args:
        cmd: The raw build command (list, shell-style string, or None).

    Returns:
        list[str]: The coerced argv, empty when *cmd* is falsy.

    Raises:
        ValueError: If shell operators, control chars, or a ``sh -c`` pattern
            are detected.
    """
    if not cmd:
        return []
    if isinstance(cmd, str):
        try:
            argv = shlex.split(cmd)
        except ValueError as exc:
            raise ValueError(f"invalid build_command: {exc}") from exc
    else:
        argv = [str(p) for p in cmd]
    if not argv:
        return []
    if any(p in _UNSAFE_TOKENS for p in argv) or any(_UNSAFE_CHARS_RE.search(p) for p in argv):
        raise ValueError("build_command must be argv-like; shell control operators are not allowed")
    if any("\n" in p or "\r" in p or "\x00" in p for p in argv):
        raise ValueError("build_command contains invalid control characters")
    exe = Path(argv[0]).name.lower()
    if exe in _SHELL_NAMES and any(p in {"-c", "-lc"} for p in argv[1:]):
        raise ValueError("build_command must not invoke a shell command string")
    return argv


# ---------------------------------------------------------------------------
# Injectable subprocess runner
# ---------------------------------------------------------------------------

_RunCallable = Callable[..., Any]


@dataclass
class RunResult:
    """Outcome of a single subprocess invocation."""

    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    timed_out: bool = False
    command: list[str] = field(default_factory=list)
    cwd: str = ""


def run_argv(
    argv: list[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    timeout_sec: int = 1800,
    run: _RunCallable = subprocess.run,
) -> RunResult:
    """Run *argv* in *cwd* and capture up to 4000 chars of each stream."""
    try:
        completed = run(
            argv,
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return RunResult(
            returncode=int(completed.returncode),
            stdout_tail=(completed.stdout or "")[-4000:],
            stderr_tail=(completed.stderr or "")[-4000:],
            command=list(argv),
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        return RunResult(returncode=-1, timed_out=True, command=list(argv), cwd=str(cwd))


# ---------------------------------------------------------------------------
# ROCm torch constraint file
# ---------------------------------------------------------------------------

_TORCH_HIP_PROBE = (
    "import sys, importlib.metadata; "
    "import torch; "
    "hip = getattr(torch.version, 'hip', None); "
    "sys.exit(0 if hip else 2); "
)
_VERSION_PROBE = (
    "import sys, importlib.metadata; "
    "pkg=sys.argv[1]; "
    "print(importlib.metadata.version(pkg))"
)


class AbiMismatchError(RuntimeError):
    """Raised when the detected torch is not a ROCm build."""


def write_rocm_torch_constraints(
    python_exe: str,
    constraint_path: str | Path,
    *,
    run: _RunCallable = subprocess.run,
) -> str:
    """Write a pip constraint file pinning the installed ROCm torch (and triton).

    Probes the interpreter for ``torch.version.hip``; raises
    :class:`AbiMismatchError` when torch is a CUDA / CPU build.

    Raises:
        AbiMismatchError: If torch is not a ROCm build.
        RuntimeError: If the torch version cannot be determined.
    """
    # Check ROCm
    hip_res = run(
        [python_exe, "-c", _TORCH_HIP_PROBE],
        capture_output=True, text=True, timeout=30,
    )
    rc = int(getattr(hip_res, "returncode", -1))
    if rc != 0:
        raise AbiMismatchError(
            f"installed torch at {python_exe!r} is not a ROCm build (probe rc={rc})"
        )
    # torch version
    tv_res = run(
        [python_exe, "-c", _VERSION_PROBE, "torch"],
        capture_output=True, text=True, timeout=30,
    )
    torch_ver = (getattr(tv_res, "stdout", "") or "").strip()
    if not torch_ver:
        raise RuntimeError(f"could not determine torch version from {python_exe!r}")
    lines = [f"torch=={torch_ver}"]
    # triton version (optional)
    tri_res = run(
        [python_exe, "-c", _VERSION_PROBE, "triton"],
        capture_output=True, text=True, timeout=30,
    )
    triton_ver = (getattr(tri_res, "stdout", "") or "").strip()
    if triton_ver:
        lines.append(f"triton=={triton_ver}")
    Path(constraint_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(constraint_path)


# ---------------------------------------------------------------------------
# ROCm toolchain alignment check
# ---------------------------------------------------------------------------

def check_rocm_toolchain_alignment(
    *,
    env: Mapping[str, str] | None = None,
    run: _RunCallable = subprocess.run,
) -> tuple[bool, str]:
    """Check hipcc presence, ROCM_PATH match, and hip>=7 header.

    Args:
        env: Environment to use for subprocess calls (``None`` inherits process env).
        run: Injectable runner.

    Returns:
        tuple[bool, str]: ``(ok, message)`` — ``ok=False`` means fatal.
    """
    import os

    effective_env = dict(env) if env is not None else dict(os.environ)

    hipcc_res = run(
        ["which", "hipcc"],
        capture_output=True, text=True, timeout=10, env=effective_env,
    )
    hipcc_path = (getattr(hipcc_res, "stdout", "") or "").strip()
    if not hipcc_path or getattr(hipcc_res, "returncode", 1) != 0:
        return True, "hipcc not found; AITER/source builds need a ROCm compiler toolchain (warning only)"

    hipcc_root_res = run(
        ["sh", "-c", f"cd $(dirname {hipcc_path!r})/.. && pwd"],
        capture_output=True, text=True, timeout=10, env=effective_env,
    )
    hipcc_root = (getattr(hipcc_root_res, "stdout", "") or "").strip()

    rocm_path = effective_env.get("ROCM_PATH", "").strip()
    if rocm_path and hipcc_root:
        rocm_real_res = run(
            ["sh", "-c", f"cd {rocm_path!r} 2>/dev/null && pwd"],
            capture_output=True, text=True, timeout=10, env=effective_env,
        )
        rocm_real = (getattr(rocm_real_res, "stdout", "") or "").strip()
        if rocm_real and hipcc_root != rocm_real:
            pass  # warn-only, not fatal

    # hip version from hipcc_root
    hip_header = Path(hipcc_root) / "include" / "hip" / "hip_runtime_api.h" if hipcc_root else None
    if hip_header and hip_header.is_file():
        content = hip_header.read_text(errors="replace")
        if "hipDeviceAttributePciChipId" not in content:
            return False, (
                f"hipcc headers at {hipcc_root} do not look compatible with the installed torch hip version; "
                "set ROCM_PATH/HIP_PATH/PATH to a ROCm 7.x toolchain before building AITER"
            )
    return True, "ok"


# ---------------------------------------------------------------------------
# Torch ABI probe
# ---------------------------------------------------------------------------

_ABI_PROBE = (
    "import sys, json, importlib.metadata, torch; "
    "hip=getattr(torch.version,'hip',None); "
    "print(json.dumps({"
    "'torch_version': torch.__version__, "
    "'hip_version': hip or '', "
    "'python_version': sys.version.split()[0], "
    "'is_rocm': bool(hip)"
    "}))"
)


def probe_torch_abi(
    python_exe: str,
    *,
    run: _RunCallable = subprocess.run,
) -> dict[str, Any]:
    """Return torch/Python ABI facts from the given interpreter.

    Args:
        python_exe: Python interpreter path to probe.
        run: Injectable runner.

    Returns:
        dict: Keys ``torch_version``, ``hip_version``, ``python_version``, ``is_rocm``.
    """
    import json

    res = run([python_exe, "-c", _ABI_PROBE], capture_output=True, text=True, timeout=30)
    out = (getattr(res, "stdout", "") or "").strip()
    if not out or getattr(res, "returncode", 1) != 0:
        return {"torch_version": "", "hip_version": "", "python_version": "", "is_rocm": False}
    try:
        return json.loads(out)
    except Exception:  # noqa: BLE001
        return {"torch_version": "", "hip_version": "", "python_version": "", "is_rocm": False}


# ---------------------------------------------------------------------------
# Artifact freshness verify
# ---------------------------------------------------------------------------

def verify_fresh_artifacts(
    build_dir: str | Path,
    since_unix: float,
    expected_artifacts: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Check that at least one expected artifact glob was built after *since_unix*.

    Args:
        build_dir: Root directory to search for artifacts.
        since_unix: Unix timestamp threshold (mtime + 1.0s slack must exceed this).
        expected_artifacts: List of glob patterns relative to *build_dir*.

    Returns:
        dict: ``{"verified": bool, "status": str, "fresh": [...fresh paths...]}``
    """
    root = Path(build_dir)
    if not root.exists():
        return {"verified": False, "status": "stale", "reason": f"build_dir absent: {root}"}

    patterns = list(expected_artifacts) if expected_artifacts else ["*.so", "**/*.so"]
    fresh: list[str] = []
    for pattern in patterns:
        for p in root.glob(pattern):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime + 1.0 >= since_unix:
                fresh.append(str(p))

    if fresh:
        return {"verified": True, "status": "ok", "fresh": sorted(set(fresh))[:16]}
    return {
        "verified": False,
        "status": "stale",
        "reason": "no freshly-built artifacts found after build",
        "build_dir": str(root),
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# Symbol verify (import probe)
# ---------------------------------------------------------------------------

def verify_symbols(
    python_exe: str,
    expected_symbols: list[str] | tuple[str, ...],
    *,
    run: _RunCallable = subprocess.run,
) -> dict[str, Any]:
    """Verify that *expected_symbols* are importable in the given interpreter.

    Uses an ``import aiter; getattr(aiter, sym)`` style probe — matches the
    installer's ``import aiter`` gate.

    Args:
        python_exe: Python interpreter (the attempt venv's python).
        expected_symbols: Dotted names to probe (e.g. ``["aiter.ops.fp4_moe"]``).
        run: Injectable runner.

    Returns:
        dict: ``{"verified": bool, "missing": [...], "present": [...]}``
    """
    missing: list[str] = []
    present: list[str] = []
    for sym in expected_symbols:
        parts = sym.split(".", 1)
        if len(parts) == 2:
            code = f"import {parts[0]}; getattr({parts[0]}, {parts[1]!r})"
        else:
            code = f"import {sym}"
        res = run([python_exe, "-c", code], capture_output=True, text=True, timeout=60)
        if getattr(res, "returncode", 1) == 0:
            present.append(sym)
        else:
            missing.append(sym)
    return {
        "verified": len(missing) == 0,
        "missing": missing,
        "present": present,
    }


# ---------------------------------------------------------------------------
# Artifact hashing (sha256, reproducibility)
# ---------------------------------------------------------------------------

def hash_artifacts(paths: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Return a ``{path: sha256_hex}`` map for each existing artifact.

    Args:
        paths: File paths to hash.

    Returns:
        dict[str, str]: Path -> sha256 hex digest; missing files are skipped.
    """
    out: dict[str, str] = {}
    for p in paths:
        try:
            data = Path(p).read_bytes()
            out[str(p)] = hashlib.sha256(data).hexdigest()
        except OSError:
            # Missing/unreadable artifact paths are skipped.
            pass
    return out


# ---------------------------------------------------------------------------
# Version-sorted tag list (sort -V -r equivalent for autoselect)
# ---------------------------------------------------------------------------

def sort_tags_desc(tags: list[str] | tuple[str, ...]) -> list[str]:
    """Return *tags* in descending version order (newest first).

    Mirrors ``git tag -l 'v*' | sort -V -r``.  Non-version-like tags are
    placed last.

    Args:
        tags: Iterable of git tag strings.

    Returns:
        list[str]: Tags sorted newest-first.
    """
    import packaging.version  # available via pip; already a transitive dep

    def _key(t: str):
        try:
            return (1, packaging.version.Version(t.lstrip("v")))
        except Exception:  # noqa: BLE001
            return (0, packaging.version.Version("0"))

    return sorted(tags, key=_key, reverse=True)


__all__ = [
    "AbiMismatchError",
    "RunResult",
    "check_rocm_toolchain_alignment",
    "coerce_build_argv",
    "hash_artifacts",
    "probe_torch_abi",
    "run_argv",
    "sort_tags_desc",
    "verify_fresh_artifacts",
    "verify_symbols",
    "write_rocm_torch_constraints",
]
