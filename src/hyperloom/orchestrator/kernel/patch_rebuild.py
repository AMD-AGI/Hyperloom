# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rebuild strategy detection, rebuild execution and post-rebuild import checks for kernel patches."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from hyperloom_common import aiter_jit_cache
from hyperloom.inference_optimizer.framework_paths import resolve_flydsl_source_roots

from .patch_jit_cache import (
    _ALLOWED_KERNEL_DEPLOY_PACKAGES,
    _EDITABLE_AITER_ROOT,
    _aiter_checkout_root,
    _aiter_jit_build_dir,
    _installed_aiter_runtime_root,
    _installed_kernel_package,
    _target_is_in_aiter_csrc,
    _target_is_sglang_jit_source,
    _trusted_aiter_jit_build_dir,
)


COMPILED_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"}
PYTHON_SOURCE_SUFFIXES = {".py"}
COMPILED_ARTIFACT_SUFFIXES = {".so", ".co", ".hsaco"}

_EDITABLE_KERNEL_DEPLOY_ROOTS = (
    _EDITABLE_AITER_ROOT,
    Path("/sgl-workspace/sglang"),
    Path("/sgl-workspace/vllm"),
)

_REBUILD_MODE_COMMAND = "command"
_REBUILD_MODE_NONE = "none"
_REBUILD_MODE_RUNTIME_JIT = "runtime_jit"
_REBUILD_MODE_CONTENT_ADDRESSED_JIT = "content_addressed_jit"


def _flydsl_root_for(target_file: Path) -> Path | None:
    """The FlyDSL checkout root containing ``target_file``, else ``None``.

    Matches case-insensitively, as the surrounding patchability and apply
    gates do, but returns the matching prefix of ``target_file`` rather than
    the matched root: the root list carries a lower-cased variant of every env
    root, and handing back a spelling that does not exist on disk would send
    snapshot mode into a fabricated tree. An env root keeps any repeated
    trailing slash, so each root is matched at its directory boundary.
    """
    raw = str(target_file).replace(os.sep, "/")
    norm = raw.lower()
    for root in resolve_flydsl_source_roots():
        lowered = root.lower().rstrip("/") + "/"
        if norm.startswith(lowered):
            return Path(raw[: len(lowered)].rstrip("/"))
    return None


def _installed_kernel_deploy_roots(coordinate_root: Path) -> list[Path]:
    """Return existing, explicitly allowed framework packages under a wheel root."""
    return [
        coordinate_root / package
        for package in sorted(_ALLOWED_KERNEL_DEPLOY_PACKAGES)
        if (coordinate_root / package).is_dir()
    ]


def _editable_kernel_deploy_roots() -> list[Path]:
    """Return existing editable framework roots supported by kernel integration."""
    return [root for root in _EDITABLE_KERNEL_DEPLOY_ROOTS if root.is_dir()]


def _detect_strategy(target_file: Path) -> dict[str, Any]:
    """Determine the rebuild strategy for a patch target.

    Matches known framework roots and the installed AITER runtime's checkout
    to select rebuild commands and artifact roots. Python codegen under AITER
    ``csrc`` invalidates build state and serving modules like native source in
    every AITER layout, and recompiles on the next runtime import instead of
    through a framework reinstall. Python patches elsewhere stay source-only.
    Native SGLang ``kernels/jit`` sources use a source-hashed cache, deferring
    compilation to the runtime instead of rebuilding.

    Args:
        target_file (Path): The file being patched.

    Returns:
        dict[str, Any]: A strategy dict with ``compiled`` (bool), ``root``
            (str), ``rebuild_mode`` (str), ``rebuild_command`` (list[str]) and
            ``artifact_roots`` (list[Path]). Runtime-JIT strategies also carry
            the authoritative ``jit_build_dir``; strategies whose rebuild
            reinstalls the framework carry the ``import_probes`` that must
            still resolve afterwards.
    """
    suffix = target_file.suffix.lower()
    strategy = _editable_workspace_strategy(target_file)
    if strategy is None:
        strategy = _installed_framework_strategy(target_file)
    strategy["compiled"] = suffix in COMPILED_SOURCE_SUFFIXES

    # Codegen under AITER csrc feeds the JIT compile, so every strategy owning an
    # AITER JIT build invalidates it and lets the runtime recompile; no reinstall
    # republishes a generated kernel, so there is nothing to run or import-probe.
    if (
        suffix in PYTHON_SOURCE_SUFFIXES
        and (strategy["rebuild_mode"] == _REBUILD_MODE_RUNTIME_JIT or strategy["jit_build_dir"])
        and _target_is_in_aiter_csrc(target_file)
    ):
        strategy.update(
            compiled=True,
            rebuild_mode=_REBUILD_MODE_RUNTIME_JIT,
            rebuild_command=[],
            artifact_roots=[],
            import_probes=[],
        )
    elif suffix in PYTHON_SOURCE_SUFFIXES:
        strategy.update(
            compiled=False,
            rebuild_mode=_REBUILD_MODE_NONE,
            rebuild_command=[],
            artifact_roots=[],
            jit_build_dir="",
            import_probes=[],
        )
    if not strategy["deploy_roots"] and strategy["root"]:
        strategy["deploy_roots"] = [Path(strategy["root"])]
    return strategy


def _source_only_strategy() -> dict[str, Any]:
    return {
        "root": "",
        "rebuild_mode": _REBUILD_MODE_NONE,
        "rebuild_command": [],
        "artifact_roots": [],
        "deploy_roots": [],
        "jit_build_dir": "",
        "import_probes": [],
    }


def _editable_workspace_strategy(target_file: Path) -> dict[str, Any] | None:
    lower = str(target_file).lower()
    if "/sgl-workspace/" not in lower:
        return None
    strategy = _source_only_strategy()
    strategy["deploy_roots"] = _editable_kernel_deploy_roots()
    if "/sgl-workspace/aiter/" in lower:
        strategy.update(
            root=str(_EDITABLE_AITER_ROOT),
            rebuild_mode=_REBUILD_MODE_COMMAND,
            jit_build_dir=str(_EDITABLE_AITER_ROOT / "aiter" / "jit" / "build"),
            rebuild_command=["/opt/venv/bin/python", "setup.py", "develop"],
            artifact_roots=[_EDITABLE_AITER_ROOT],
            import_probes=["aiter"],
        )
        return strategy
    if "/sgl-workspace/sglang/sgl-kernel/" in lower:
        root = Path("/sgl-workspace/sglang/sgl-kernel")
        strategy.update(
            root=str(root),
            rebuild_mode=_REBUILD_MODE_COMMAND,
            rebuild_command=["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."],
            artifact_roots=[root],
            import_probes=["sgl_kernel"],
        )
        return strategy
    if "/sgl-workspace/sglang/" in lower:
        root = Path("/sgl-workspace/sglang")
        strategy["root"] = str(root)
        if _target_is_sglang_jit_source(target_file):
            strategy["rebuild_mode"] = _REBUILD_MODE_CONTENT_ADDRESSED_JIT
            return strategy
        strategy.update(
            rebuild_mode=_REBUILD_MODE_COMMAND,
            rebuild_command=["/opt/venv/bin/python", "-m", "pip", "install", "-e", "python"],
            artifact_roots=[root],
            import_probes=["sglang", "sglang.srt.server_args"],
        )
        return strategy
    if "/sgl-workspace/vllm/" in lower:
        root = Path("/sgl-workspace/vllm")
        strategy.update(
            root=str(root),
            rebuild_mode=_REBUILD_MODE_COMMAND,
            rebuild_command=["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."],
            artifact_roots=[root],
            import_probes=["vllm"],
        )
        return strategy
    return None


def _installed_framework_strategy(target_file: Path) -> dict[str, Any]:
    strategy = _source_only_strategy()
    if flydsl_root := _flydsl_root_for(target_file):
        # FlyDSL compiles at import time, so there is nothing to rebuild; the
        # root only has to be right for cache invalidation and rebuild cwd.
        strategy["root"] = str(flydsl_root)
        return strategy
    installed_aiter_root = _installed_aiter_runtime_root(target_file)
    if installed_aiter_root:
        # The shared JIT transaction backs up build/ and sibling serving .so files.
        # Recursively copying package artifacts here would duplicate that work.
        strategy.update(
            root=str(installed_aiter_root),
            deploy_roots=_installed_kernel_deploy_roots(installed_aiter_root),
            rebuild_mode=_REBUILD_MODE_RUNTIME_JIT,
            jit_build_dir=str(aiter_jit_cache.resolve_jit_build_dir(installed_aiter_root / "aiter") or ""),
        )
        return strategy
    checkout_root = _aiter_checkout_root(target_file)
    if checkout_root is not None:
        strategy.update(
            root=str(checkout_root),
            rebuild_mode=_REBUILD_MODE_RUNTIME_JIT,
            jit_build_dir=str(_aiter_jit_build_dir() or ""),
        )
        return strategy
    installed_kernel = _installed_kernel_package(target_file)
    if installed_kernel:
        strategy["root"] = str(installed_kernel[0])
        strategy["deploy_roots"] = _installed_kernel_deploy_roots(installed_kernel[0])
        if _target_is_sglang_jit_source(target_file):
            strategy["rebuild_mode"] = _REBUILD_MODE_CONTENT_ADDRESSED_JIT
    return strategy


def _discover_artifacts(
    roots: Iterable[Path],
    explicit_artifact_paths: Iterable[str] | None,
    *,
    max_artifacts: int = 400,
) -> list[Path]:
    """Collect compiled-artifact files to back up before a rebuild.

    Includes any explicitly provided artifact paths, then walks each root for
    files with a compiled-artifact suffix, deduplicating and capping the total.

    Args:
        roots (Iterable[Path]): Directories to scan recursively for artifacts.
        explicit_artifact_paths (Iterable[str] | None): Caller-specified
            artifact files to always include.
        max_artifacts (int): Maximum number of artifacts to collect.

    Returns:
        list[Path]: The discovered artifact file paths.
    """
    seen: set[Path] = set()
    artifacts: list[Path] = []
    for raw in explicit_artifact_paths or []:
        p = Path(raw)
        if p.is_file() and p not in seen:
            artifacts.append(p)
            seen.add(p)
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if len(artifacts) >= max_artifacts:
                break
            if p.is_file() and p.suffix.lower() in COMPILED_ARTIFACT_SUFFIXES and p not in seen:
                artifacts.append(p)
                seen.add(p)
    return artifacts


def _run_rebuild(command: list[str], cwd: Path, timeout_sec: int) -> dict[str, Any]:
    """Run a rebuild command and capture its result.

    The subprocess runs with ``/opt/venv/bin`` prepended to ``PATH``.

    Args:
        command (list[str]): The rebuild command argv; empty means skip.
        cwd (Path): Working directory for the rebuild.
        timeout_sec (int): Subprocess timeout in seconds.

    Returns:
        dict[str, Any]: A result dict with ``status`` (``ok`` / ``failed`` /
            ``skipped``), ``returncode``, captured ``stdout_tail`` /
            ``stderr_tail``, and the ``command`` / ``cwd`` used.
    """
    if not command:
        return {"status": "skipped", "reason": "no rebuild command"}
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env={**os.environ, "PATH": f"/opt/venv/bin:{os.environ.get('PATH', '')}"},
    )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "command": command,
        "cwd": str(cwd),
    }


_IMPORT_PROBE_TIMEOUT_SEC = 600

_IMPORT_PROBE_SCRIPT = """\
import importlib.util
import json
import sys
import traceback

report = {}
for name in sys.argv[1:]:
    try:
        spec = importlib.util.find_spec(name)
    except BaseException:  # noqa: BLE001 - a half-installed parent raises anything
        report[name] = {"importable": False, "error": traceback.format_exc()}
        continue
    if spec is None:
        report[name] = {"importable": False, "error": "no module spec"}
    else:
        report[name] = {"importable": True, "origin": spec.origin or ""}
json.dump({"modules": report, "sys_path": sys.path}, sys.stdout)
"""


def _verify_rebuild_imports(
    probes: list[str],
    interpreter: str,
    *,
    timeout_sec: int = _IMPORT_PROBE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Prove the rebuilt framework is still importable before benchmarking it.

    An eager rebuild is a package (re)install: pip drops the old distribution
    and writes fresh metadata plus a new editable import hook, so a returncode
    of ``0`` says nothing about whether a new interpreter can still resolve the
    modules the server loads first. Resolving the probes in a fresh interpreter
    (the one that ran the rebuild, so the probe sees the install that was just
    written) turns that into a checked precondition, and records each resolved
    ``origin`` + ``sys.path`` so a later disagreement between baseline and
    candidate has the import metadata to explain it.

    Args:
        probes: Module names that must resolve after the rebuild.
        interpreter: The rebuild command's interpreter, whose install the
            probes must observe.
        timeout_sec: Probe subprocess timeout in seconds.

    Returns:
        A dict with ``status`` (``ok`` / ``failed`` / ``skipped``), the
        per-module ``modules`` report and the probed ``sys_path``.
    """
    if not probes:
        return {"status": "skipped", "reason": "strategy declares no import probes"}
    if not Path(interpreter).name.lower().startswith("python"):
        # An override can rebuild via make/cmake/sh; probing the wrong
        # interpreter would report on an unrelated environment.
        return {
            "status": "skipped",
            "reason": f"rebuild command is not interpreter-led: {interpreter}",
        }
    # Probe from a neutral cwd: `python -c` puts the working directory on
    # sys.path, and a framework checkout root can shadow its own package.
    try:
        proc = subprocess.run(
            [interpreter, "-c", _IMPORT_PROBE_SCRIPT, *probes],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PATH": f"/opt/venv/bin:{os.environ.get('PATH', '')}"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "failed",
            "error": f"import probe could not run: {type(exc).__name__}: {exc}",
            "probes": probes,
        }
    if proc.returncode != 0:
        return {
            "status": "failed",
            "error": f"import probe exited {proc.returncode}",
            "probes": probes,
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    try:
        report = json.loads(proc.stdout or "{}")
    except ValueError as exc:
        return {
            "status": "failed",
            "error": f"unparseable import probe output: {exc}",
            "probes": probes,
            "stdout_tail": (proc.stdout or "")[-4000:],
        }
    modules = dict(report.get("modules") or {})
    broken = {name: entry for name, entry in modules.items() if not entry.get("importable")}
    result: dict[str, Any] = {
        "status": "failed" if broken else "ok",
        "probes": probes,
        "modules": modules,
        "sys_path": list(report.get("sys_path") or []),
        "interpreter": interpreter,
    }
    if broken:
        result["error"] = "; ".join(f"{name}: {entry.get('error')}" for name, entry in sorted(broken.items()))
    return result


def _run_strategy_rebuild(
    strategy: dict[str, Any],
    *,
    command_override: list[str],
    fallback_cwd: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    """Run an eager rebuild or record a runtime-JIT handoff."""
    command = command_override or list(strategy.get("rebuild_command") or [])
    strategy_root = str(strategy.get("root") or "").strip()
    cwd = Path(strategy_root) if strategy_root else fallback_cwd
    if command:
        result = _run_rebuild(command, cwd, timeout_sec)
        if result.get("status") != "ok":
            return result
        import_check = _verify_rebuild_imports(
            [str(name) for name in strategy.get("import_probes") or []],
            command[0],
        )
        result["import_check"] = import_check
        if import_check.get("status") == "failed":
            result["status"] = "failed"
            result["error"] = f"rebuild returned 0 but the framework is not importable: {import_check.get('error')}"
        return result
    if strategy.get("rebuild_mode") == _REBUILD_MODE_RUNTIME_JIT:
        return {
            "status": "deferred",
            "mode": _REBUILD_MODE_RUNTIME_JIT,
            "reason": ("installed AITER sources compile on runtime import after JIT cache invalidation"),
            "cwd": str(cwd),
        }
    if strategy.get("rebuild_mode") == _REBUILD_MODE_CONTENT_ADDRESSED_JIT:
        return {
            "status": "deferred",
            "mode": _REBUILD_MODE_CONTENT_ADDRESSED_JIT,
            "reason": ("SGLang in-tree JIT sources compile on runtime import under a source-hashed cache key"),
            "cwd": str(cwd),
        }
    return _run_rebuild([], cwd, timeout_sec)


def _runtime_jit_invalidation_error(
    strategy: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> str:
    """Validate restore trust before mutation and cache invalidation afterwards."""
    if strategy.get("rebuild_mode") != _REBUILD_MODE_RUNTIME_JIT:
        return ""
    expected_raw = str(strategy.get("jit_build_dir") or "").strip()
    if not expected_raw:
        return "runtime-JIT strategy has no authoritative jit_build_dir; correct AITER_JIT_DIR and initialize the runtime cache before applying patches"
    if not _trusted_aiter_jit_build_dir(Path(expected_raw)):
        return f"untrusted strategy jit/build dir: {expected_raw}"
    if result is None:
        return ""
    actual_raw = str(result.get("src") or "").strip()
    if not actual_raw or Path(actual_raw).resolve() != Path(expected_raw).resolve():
        return f"JIT invalidation did not inspect the strategy-pinned build dir {expected_raw}"
    if result.get("status") not in {"ok", "clean"}:
        return (
            "strategy-pinned JIT build dir was not invalidated or proven clean: "
            f"{result.get('reason') or result.get('error') or result.get('status')}"
        )
    return ""


def _remote_jit_invalidation_error(records_by_host: dict[str, list[dict[str, Any]]]) -> str:
    """Name the pods whose apply records do not hold exactly one JIT backup, the one pod-side revert restores."""
    if not records_by_host:
        return "no pod reported an invalidation"
    missed = sorted(
        host
        for host, records in records_by_host.items()
        if sum((record.get("jit_backup") or {}).get("status") in {"ok", "clean"} for record in records) != 1
    )
    return f"pods {missed} did not invalidate it exactly once" if missed else ""


def _rebuild_ok_to_proceed(result: dict[str, Any]) -> bool:
    """Return whether apply may proceed after this rebuild result."""
    return result.get("status") in {"ok", "deferred"}
