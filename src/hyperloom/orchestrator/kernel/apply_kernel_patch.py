#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply an optimized kernel file with source/artifact backup and fast revert."""

from __future__ import annotations

import json
import logging
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from hyperloom.common import aiter_jit_cache
from hyperloom.common.timeutil import now_iso
from hyperloom.common.unified_diff import PatchOp, PatchTarget, repo_relative_manifest
from hyperloom.inference_optimizer.framework_paths import resolve_flydsl_source_roots

from ._patch_tree import (
    _claim_backup_dir,
    _contained_dest,
    _copy_to_backup,
    _delete_recorded_backups,
    _revert_backup_trusted,
    _within_root,
    apply_snapshot,
)

log = logging.getLogger(__name__)


COMPILED_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"}
PYTHON_SOURCE_SUFFIXES = {".py"}
COMPILED_ARTIFACT_SUFFIXES = {".so", ".co", ".hsaco"}
TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".patch", ".diff"}

_ALLOWED_KERNEL_DEPLOY_PACKAGES = frozenset({"aiter", "aiter_meta", "sglang", "vllm"})
_EDITABLE_AITER_ROOT = Path("/sgl-workspace/aiter")
_EDITABLE_KERNEL_DEPLOY_ROOTS = (
    _EDITABLE_AITER_ROOT,
    Path("/sgl-workspace/sglang"),
    Path("/sgl-workspace/vllm"),
)


_UNSAFE_COMMAND_TOKENS = {";", "&&", "||", "|", ">", ">>", "<", "<<", "`"}
_UNSAFE_COMMAND_CHARS_RE = re.compile(r"[;&|`$<>\r\n]")
_SHELL_COMMAND_NAMES = {"bash", "dash", "sh", "zsh", "ksh"}


def _coerce_rebuild_command(rebuild_command: "list[str] | str | None") -> list[str]:
    """Return a rebuild argv, never a shell command string.

    A string rebuild_command is tokenised with shlex and run argv-only
    (shell=False) instead of ``bash -lc <str>``, so it cannot inject shell
    control operators. A legitimate argv (already a list, or a plain-command
    string like ``ninja -C build``) is unaffected; only commands that rely on
    shell syntax are rejected.
    """
    if not rebuild_command:
        return []
    if isinstance(rebuild_command, str):
        try:
            argv = shlex.split(rebuild_command)
        except ValueError as exc:
            raise ValueError(f"invalid rebuild_command: {exc}") from exc
    else:
        argv = [str(part) for part in rebuild_command]
    if not argv:
        return []
    if any(part in _UNSAFE_COMMAND_TOKENS for part in argv) or any(
        _UNSAFE_COMMAND_CHARS_RE.search(part) for part in argv
    ):
        raise ValueError("rebuild_command must be argv-like and cannot contain shell control operators")
    if any(("\n" in part or "\r" in part or "\x00" in part) for part in argv):
        raise ValueError("rebuild_command contains invalid control characters")
    exe = Path(argv[0]).name.lower()
    if exe in _SHELL_COMMAND_NAMES and any(part in {"-c", "-lc"} for part in argv[1:]):
        raise ValueError("rebuild_command must not invoke a shell command string")
    return argv


# Pod-local multi-node backup dir; overridable via $HYPERLOOM_MN_KERNEL_BACKUP_DIR.
_MN_POD_BACKUP_DIR_DEFAULT = "/var/kernel_patch_backups"
_MANIFEST_MULTINODE = "multinode"


def _is_multi_node() -> bool:
    """Report whether a multi-node RayJob is active.

    Relies solely on the trusted in-process ``$INFERENCE_OPTIMIZER_NODES``
    signal that the optimizer CLI exports at launch (inherited by this
    process). No state file is read, so a co-tenant cannot force multi-node
    fan-out by planting a world-writable ``multi_node_state.json``.

    Returns:
        ``True`` when ``$INFERENCE_OPTIMIZER_NODES`` is ``>= 2``; ``False``
        otherwise (including when the env var is unset or non-numeric).
    """
    try:
        return int(os.environ.get("INFERENCE_OPTIMIZER_NODES", "0") or 0) >= 2
    except ValueError:
        return False


@dataclass(frozen=True)
class PodPatchApply:
    """One file to patch on every pod.

    Attributes:
        target_file: The file to patch on each pod.
        patch_path: Path to the patch file to apply.
        kernel_id: Identifier of the kernel being patched.
        backup_dir_on_pod: Directory on each pod to store backups.
        jit_build_dir: JIT build directory to back up on each pod, or ``""``.
    """

    target_file: Path
    patch_path: Path
    kernel_id: str
    backup_dir_on_pod: str
    jit_build_dir: str = ""


def _dispatch_multinode_apply(apply: PodPatchApply, timeout_sec: int = 180) -> dict[str, Any]:
    """Fan a patch out to every pod (head + workers).

    Args:
        apply: The file, its patch and the per-pod backup locations.
        timeout_sec: Subprocess timeout in seconds.

    Returns:
        The parsed JSON result from the multi-node dispatch.

    Raises:
        RuntimeError: On subprocess failure, non-JSON output, or a pod status
            other than ``ok`` (so the caller can roll back the local copy).
    """
    cmd = [
        sys.executable,
        "-m",
        "hyperloom",
        "multi-node",
        "apply-patch",
        "--patch-file",
        str(apply.patch_path),
        "--target-path",
        str(apply.target_file),
        "--backup-dir",
        apply.backup_dir_on_pod,
        "--kernel-id",
        apply.kernel_id or "",
        "--jit-build-dir",
        str(apply.jit_build_dir or ""),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"multi-node apply-patch returned rc={proc.returncode}: stderr={(proc.stderr or '')[-2000:]!r}"
        )
    try:
        parsed = (
            json.loads(proc.stdout.strip().splitlines()[-1])
            if proc.stdout.strip().startswith("{") is False
            else json.loads(proc.stdout)
        )
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"multi-node apply-patch stdout not JSON: {exc!r}; stdout_tail={(proc.stdout or '')[-2000:]!r}"
        ) from exc
    if str(parsed.get("status", "")).lower() != "ok":
        raise RuntimeError(
            f"multi-node apply-patch reported status={parsed.get('status')!r}: failures={parsed.get('failures')!r}"
        )
    return parsed


def _dispatch_multinode_revert(
    *,
    target_path: str = "",
    backup_map: dict[str, str] | None = None,
    records_by_host: dict[str, list[dict[str, Any]]] | None = None,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Restore the original file on every pod that received the apply.

    Best-effort: failures are reported in the result rather than raised.

    Args:
        target_path: The patched file path to restore on each pod.
        backup_map: Mapping of pod identifier to its backup path.
        timeout_sec: Subprocess timeout in seconds.

    Returns:
        The parsed JSON result, or an empty dict when output is not JSON.
    """
    cmd = [
        sys.executable,
        "-m",
        "hyperloom",
        "multi-node",
        "revert-patch",
        "--target-path",
        str(target_path),
        "--records-json",
        json.dumps(records_by_host or {}, sort_keys=True),
        "--backup-map-json",
        json.dumps(backup_map or {}, sort_keys=True),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    out = (proc.stdout or "").strip()
    try:
        parsed = json.loads(out) if out.startswith("{") else {}
    except json.JSONDecodeError:
        parsed = {}
    if proc.returncode != 0 or str(parsed.get("status", "")).lower() != "ok":
        # Warn-only: sandbox revert already won.
        sys.stderr.write(
            f"WARN multi-node revert-patch rc={proc.returncode} "
            f"status={parsed.get('status')!r} "
            f"stderr_tail={(proc.stderr or '')[-1000:]!r}\n"
        )
    return parsed or {
        "status": "partial",
        "returncode": proc.returncode,
        "stdout_tail": out[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def _dispatch_multinode_finalize(
    records_by_host: dict[str, list[dict[str, Any]]],
    timeout_sec: int = 120,
) -> dict[str, Any]:
    """Delete accepted transaction backups on every inference pod."""
    cmd = [
        sys.executable,
        "-m",
        "hyperloom",
        "multi-node",
        "finalize-patch",
        "--records-json",
        json.dumps(records_by_host, sort_keys=True),
        "--timeout-sec",
        str(timeout_sec),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    out = (proc.stdout or "").strip()
    try:
        parsed = json.loads(out) if out.startswith("{") else {}
    except json.JSONDecodeError:
        parsed = {}
    if proc.returncode != 0 or parsed.get("status") != "ok":
        return {
            "status": "partial",
            "returncode": proc.returncode,
            "result": parsed,
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    return parsed


def _source_text_looks_complete(text: str, suffix: str) -> bool:
    """Heuristically decide whether ``text`` is a complete source file."""
    stripped = text.strip()
    if not stripped or "```" in stripped:
        return False
    if suffix == ".py":
        try:
            compile(stripped + "\n", "<optimized_kernel>", "exec")
        except SyntaxError:
            return False
        return any(marker in stripped for marker in ("def ", "class ", "import ", "@triton.jit", "torch."))
    if suffix in COMPILED_SOURCE_SUFFIXES:
        return any(
            marker in stripped
            for marker in (
                "#include",
                "__global__",
                "__device__",
                "extern ",
                "namespace ",
                "template",
                "void ",
                "int ",
                "float ",
                "half",
                "torch::",
            )
        )
    return False


def _validate_patch_source(patch: Path, target: Path) -> None:
    """Validate that ``patch`` is a complete drop-in replacement for ``target``.

    Checks that the patch is not a text artifact, that its suffix matches the
    target's, that it reads as text and looks like a complete source file, that
    it is compatible with the target (see
    :func:`_validate_replacement_compatibility`), and that Python patches
    compile.

    Args:
        patch (Path): The candidate replacement source file.
        target (Path): The file that would be replaced.

    Raises:
        ValueError: If the patch is incompatible, not a complete source file,
            or not decodable as text.
    """
    patch_suffix = patch.suffix.lower()
    target_suffix = target.suffix.lower()
    if patch_suffix in TEXT_ARTIFACT_SUFFIXES:
        raise ValueError(f"patch_path is not a complete source file: {patch}")
    if patch_suffix != target_suffix:
        raise ValueError(
            f"patch suffix {patch_suffix or '<none>'} does not match target suffix {target_suffix or '<none>'}"
        )
    try:
        text = patch.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError as exc:
        raise ValueError(f"patch_path is not text source: {patch}") from exc
    if not _source_text_looks_complete(text, target_suffix):
        raise ValueError(f"patch_path does not look like a complete {target_suffix} source file: {patch}")
    target_text = target.read_text(encoding="utf-8", errors="replace")
    _validate_replacement_compatibility(text, target_text, target)
    if target_suffix == ".py":
        py_compile.compile(str(patch), doraise=True)


def _host_entry_functions(source_text: str) -> set[str]:
    """Extract public host (non-``__global__``) entry function names.

    Scans for C/C++ free-function definitions with common return types,
    excluding ``__global__`` kernels, leading-underscore helpers, and ``main``.

    Args:
        source_text (str): The C/C++/HIP source text to scan.

    Returns:
        set[str]: The set of discovered host entry-function names.
    """
    names: set[str] = set()
    pattern = re.compile(
        r"(?m)^\s*(?!__global__)(?:template\s*<[^>]+>\s*)?"
        r"(?:void|int|bool|float|double|auto|at::Tensor|torch::Tensor)\s+"
        r"([A-Za-z_]\w*)\s*\("
    )
    for match in pattern.finditer(source_text):
        name = match.group(1)
        if name.startswith("_") or name in {"main"}:
            continue
        names.add(name)
    return names


def _validate_replacement_compatibility(patch_text: str, target_text: str, target: Path) -> None:
    """Guard against patches that break the target's integration contract.

    Rejects patches that introduce standalone PYBIND11/TORCH_LIBRARY
    registrations absent from the target, drop a required ``namespace aiter``,
    or omit any host entry function the target exposes.

    Args:
        patch_text (str): The candidate replacement source text.
        target_text (str): The current target source text.
        target (Path): The target path (used in error messages).

    Raises:
        ValueError: If the patch is incompatible with the target's contract.
    """
    if "PYBIND11_MODULE" in patch_text and "PYBIND11_MODULE" not in target_text:
        raise ValueError("patch creates a standalone PYBIND11 module but target is a framework source file")
    if "TORCH_LIBRARY" in patch_text and "TORCH_LIBRARY" not in target_text:
        raise ValueError("patch creates standalone TORCH_LIBRARY registration absent from target")
    if "namespace aiter" in target_text and "namespace aiter" not in patch_text:
        raise ValueError("patch does not preserve namespace aiter")

    required = _host_entry_functions(target_text)
    if required:
        present = _host_entry_functions(patch_text)
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                f"patch does not preserve target host entry function(s) for {target}: " + ", ".join(missing[:12])
            )


def _clear_python_kernel_caches(target: Path) -> dict[str, Any]:
    """Clear Python / Triton / Inductor caches around a Python patch.

    Removes ``__pycache__`` dirs near the target plus the well-known Triton and
    torch-inductor cache directories so a re-import recompiles the patched
    kernel.

    Args:
        target (Path): The patched Python source file.

    Returns:
        dict[str, Any]: ``{"status": "ok", "removed": [...]}`` listing the
            paths that were removed.
    """
    removed: list[str] = []

    def remove_path(path: Path) -> None:
        """Remove a file or directory, recording success and ignoring errors.

        Args:
            path (Path): The file or directory to remove.
        """
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed.append(str(path))
        except OSError:
            return

    for cache_dir in target.parent.rglob("__pycache__"):
        remove_path(cache_dir)

    home = Path(os.environ.get("HOME", "~")).expanduser()
    for path in (
        home / ".triton" / "cache",
        home / ".cache" / "triton",
        home / ".cache" / "torch" / "inductor",
    ):
        if path.exists():
            remove_path(path)
    temp_root = Path(tempfile.gettempdir())
    for pattern in ("torchinductor_*", "triton_*"):  # nosec B108 - intentionally clears known runtime cache prefixes.
        for path in temp_root.glob(pattern):
            if path.exists():
                remove_path(path)
    return {"status": "ok", "removed": removed}


# aiter JIT cache invalidation around rebuilds (setup.py develop won't invalidate jit/build/ .so).
# aiter device sources ship either in-tree (``/aiter/csrc/``) or in the sibling
# split wheel (``/aiter_meta/csrc/``); both feed the same importable ``aiter``
# JIT build, so the rebuild gates must recognise both layouts.
_AITER_CSRC_MARKERS = ("/aiter/csrc/", "/aiter_meta/csrc/")
_REBUILD_MODE_COMMAND = "command"
_REBUILD_MODE_NONE = "none"
_REBUILD_MODE_RUNTIME_JIT = "runtime_jit"
_REBUILD_MODE_CONTENT_ADDRESSED_JIT = "content_addressed_jit"

# SGLang's in-tree JIT kernels are compiled by tvm-ffi on first import into a
# cache directory keyed on a hash of the source closure, so a patched source
# builds under a new key instead of colliding with the baseline module.
_SGLANG_JIT_SOURCE_MARKER = "/sglang/kernels/jit/"


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


def _installed_kernel_package(
    target_file: Path,
) -> tuple[Path, str] | None:
    """Return ``(site-packages root, package name)`` for an allowed framework."""
    # Preserve the lexical install path: venv/site-packages is commonly a
    # symlink, and resolving it here would make strategy classification disagree
    # with the path matching performed by _detect_strategy.
    target = target_file.absolute()
    for parent in target.parents:
        if parent.name in _ALLOWED_KERNEL_DEPLOY_PACKAGES and parent.parent.name in {"site-packages", "dist-packages"}:
            return parent.parent, parent.name
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


def _installed_aiter_runtime_root(target_file: Path) -> Path | None:
    """Return the package root shared by installed ``aiter`` and ``aiter_meta``.

    AITER wheels split runtime Python/JIT code into ``aiter`` and device sources
    into the sibling ``aiter_meta`` package. Both are runtime-JIT inputs and must
    share one deployment/rebuild strategy.
    """
    installed = _installed_kernel_package(target_file)
    if installed is None or installed[1] not in {"aiter", "aiter_meta"}:
        return None
    return installed[0]


def _aiter_checkout_root(target_file: Path) -> Path | None:
    """Identify a source checkout through the installed runtime, not its basename."""
    package = aiter_jit_cache.resolve_package_root()
    if package is None or package.parent.name in {"site-packages", "dist-packages"}:
        return None
    if not (package / "__init__.py").is_file() or not (package / "jit" / "__init__.py").is_file():
        return None
    meta = os.environ.get("AITER_META_DIR", "").strip()
    source_root = Path(meta).absolute() if meta else package.parent
    if not (source_root / "csrc").is_dir():
        source_root = package.parent
    if (source_root / "csrc").is_dir() and _within_root(target_file, source_root):
        return source_root
    if _within_root(target_file, package):
        return package.parent
    return None


def _target_is_in_aiter_csrc(target_file: Path) -> bool:
    """Report whether a file belongs to AITER device sources or codegen.

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` under ``aiter/csrc/``, ``aiter_meta/csrc/``, or the ``csrc/``
        tree of the checkout identified by the installed runtime package.
    """
    norm = str(target_file).replace(os.sep, "/")
    checkout = _aiter_checkout_root(target_file)
    return any(marker in norm for marker in _AITER_CSRC_MARKERS) or (
        checkout is not None and _within_root(target_file, checkout / "csrc")
    )


def _target_is_sglang_jit_source(target_file: Path) -> bool:
    """Report whether a file lives under SGLang's in-tree JIT source tree.

    Matches both the editable checkout
    (``/sgl-workspace/sglang/python/sglang/kernels/jit/...``) and an installed
    wheel (``.../site-packages/sglang/kernels/jit/...``).

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` if the path is under a ``sglang/kernels/jit/`` directory.
    """
    norm = str(target_file).replace(os.sep, "/").lower()
    return _SGLANG_JIT_SOURCE_MARKER in norm


def _target_uses_aiter_jit(target_file: Path) -> bool:
    """Return whether a source belongs to an installed/editable AITER runtime."""
    normalized = str(target_file).replace(os.sep, "/")
    return (
        _installed_aiter_runtime_root(target_file) is not None
        or "/sgl-workspace/aiter/" in normalized
        or _aiter_checkout_root(target_file) is not None
    )


def _aiter_jit_build_dir() -> Path | None:
    """Resolve AITER's build directory through the shared runtime resolver.

    Returns:
        The build path under ``AITER_JIT_DIR``, the package JIT directory, or
        the initialized user cache; ``None`` when the package/cache is unavailable.
    """
    return aiter_jit_cache.resolve_jit_build_dir(aiter_jit_cache.resolve_package_root())


def _invalidate_aiter_jit_build(
    target_file: Path,
    backup_dir: Path,
    *,
    jit_build_dir: Path | None = None,
) -> dict[str, Any]:
    """Back up AITER's build tree and all serving modules for a source patch.

    Args:
        target_file: The file being patched (gates the operation).
        backup_dir: Parent of the build-tree and serving-module backups.
        jit_build_dir: Authoritative JIT build path from the apply strategy.
            Falls back to runtime discovery for non-runtime-JIT strategies.

    Returns:
        The shared transaction record: ``ok`` when cache state was moved,
        ``clean`` when neither build nor modules existed, ``skipped``, or ``failed``.
    """
    if jit_build_dir is None and not _target_is_in_aiter_csrc(target_file):
        return {
            "status": "skipped",
            "reason": ("target is outside aiter csrc and no runtime-JIT build dir was supplied"),
        }
    jit_build = jit_build_dir or _aiter_jit_build_dir()
    if jit_build is None:
        return {
            "status": "skipped",
            "reason": "AITER runtime JIT directory unavailable; correct AITER_JIT_DIR and initialize the runtime cache before applying patches",
        }
    return aiter_jit_cache.invalidate_jit_cache(jit_build, backup_dir)


def _trusted_aiter_jit_build_dir(path: Path) -> bool:
    """Validate a strategy-persisted AITER JIT destination.

    Accepts both shapes apply can pin: a wheel install under site-packages and
    the editable checkout :data:`_EDITABLE_AITER_ROOT`.
    """
    root = _installed_aiter_runtime_root(path) or _EDITABLE_AITER_ROOT
    package = root / "aiter"
    if aiter_jit_cache.trusted_jit_build_dir(path, package / "jit" / "build"):
        return (package / "__init__.py").is_file() and (package / "jit" / "__init__.py").is_file()
    package = aiter_jit_cache.resolve_package_root()
    expected = aiter_jit_cache.resolve_jit_build_dir(package)
    return (
        package is not None
        and expected is not None
        and aiter_jit_cache.trusted_jit_build_dir(path, expected)
        and (package / "__init__.py").is_file()
        and (package / "jit" / "__init__.py").is_file()
    )


def _restore_aiter_jit_build(
    jit_build_backup: dict[str, Any],
    *,
    expected_jit_build_dir: str | Path | None = None,
    backup_root: str | Path | None = None,
) -> dict[str, Any]:
    """Restore AITER build and serving-module state within the recorded scope.

    The shared transaction validates backups before removing candidate artifacts.
    Legacy build-only records leave serving modules untouched.

    Args:
        jit_build_backup: The backup record returned by
            :func:`_invalidate_aiter_jit_build`.
        expected_jit_build_dir: Apply-time strategy path. Runtime-JIT restores
            use this instead of rediscovering aiter from the current process.
        backup_root: Apply manifest's backup directory. The recorded backup
            must remain contained under this root.

    Returns:
        A status dict with ``status`` of ``ok``, ``skipped``, or ``failed``.
    """
    if not isinstance(jit_build_backup, dict) or jit_build_backup.get("status") not in {"ok", "clean"}:
        return {"status": "skipped", "reason": "no backup recorded"}
    expected = Path(expected_jit_build_dir) if expected_jit_build_dir else _aiter_jit_build_dir()
    if expected is None or not _trusted_aiter_jit_build_dir(expected):
        return {"status": "failed", "error": f"untrusted strategy jit/build dir: {expected}"}
    return aiter_jit_cache.restore_jit_cache(jit_build_backup, expected, backup_root)


# aiter cpp_itfs kernels are runtime-compiled into parameter-keyed caches.
_AITER_CPP_ITFS_MARKERS = ("/aiter/csrc/cpp_itfs/", "/aiter_meta/csrc/cpp_itfs/")
_MD_NAME_RE = re.compile(r"""(?m)^\s*MD_NAME\s*=\s*["']([^"']+)["']""")


def _target_is_in_aiter_cpp_itfs(target_file: Path) -> bool:
    """Report whether a file lives under an ``aiter/csrc/cpp_itfs/`` tree.

    Strict subset of :func:`_target_is_in_aiter_csrc`: these are the
    runtime-compiled kernels whose served ``.so`` lives in
    ``$HOME/.aiter/build`` rather than in ``<aiter>/jit/build`` or the
    statically-linked wheel. Matches the editable checkout
    (``/sgl-workspace/aiter/csrc/cpp_itfs/...``), the dist-packages layout
    (``.../aiter/csrc/cpp_itfs/...``), and the split-wheel layout
    (``.../aiter_meta/csrc/cpp_itfs/...``).

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` if the path is under an ``aiter/csrc/cpp_itfs/`` directory.
    """
    norm = str(target_file).replace(os.sep, "/")
    checkout = _aiter_checkout_root(target_file)
    return any(marker in norm for marker in _AITER_CPP_ITFS_MARKERS) or (
        checkout is not None and _within_root(target_file, checkout / "csrc" / "cpp_itfs")
    )


def _aiter_cpp_itfs_build_dir() -> Path:
    """Resolve aiter's cpp_itfs runtime ``BUILD_DIR``.

    Mirrors aiter ``csrc/cpp_itfs/utils.py``: ``$AITER_ROOT_DIR/build`` with
    ``$AITER_ROOT_DIR`` defaulting to ``$HOME/.aiter``. Honouring both env
    vars keeps non-default deployments + unit tests correct without importing
    aiter into this standalone tool.

    Returns:
        The resolved cpp_itfs ``build`` directory path.
    """
    root = os.environ.get("AITER_ROOT_DIR", "").strip()
    if not root:
        home = Path(os.environ.get("HOME", "~")).expanduser()
        root = str(home / ".aiter")
    return Path(root) / "build"


def _cpp_itfs_module_names(target_file: Path) -> list[str]:
    """Collect ``MD_NAME`` prefixes for the cpp_itfs module(s) a source feeds.

    The cpp_itfs ``.py`` driver next to the patched source declares
    ``MD_NAME = "pa_ragged"`` (etc.), which becomes the
    ``<md_name>_<hash>`` runtime-cache folder prefix. A single shared
    ``.cuh`` (e.g. ``pa_kernels.cuh``) is pulled into several drivers in the
    same directory, so we collect EVERY ``MD_NAME`` declared in the target's
    directory.

    Args:
        target_file: The patched source file whose directory is scanned.

    Returns:
        The discovered ``MD_NAME`` prefixes. An empty list tells the caller to
        fall back to clearing the whole cpp_itfs build root.
    """
    names: set[str] = set()
    search_dir = target_file.parent
    try:
        py_files = sorted(search_dir.glob("*.py"))
    except OSError:
        py_files = []
    if target_file.suffix.lower() == ".py":
        py_files = list(dict.fromkeys([target_file, *py_files]))
    for py in py_files:
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _MD_NAME_RE.finditer(text):
            names.add(match.group(1))
    return sorted(names)


def _invalidate_aiter_cpp_itfs_cache(
    target_file: Path,
    backup_dir: Path,
    *,
    build_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Move aiter cpp_itfs runtime-cache dirs aside so the re-baseline server
    runtime-recompiles the patched kernel from clean state.

    No-op (``skipped``, ``is_cpp_itfs=False``) for non-cpp_itfs targets. For
    cpp_itfs targets the affected ``<build_dir>/<md_name>_*`` dirs (scoped by
    ``MD_NAME`` when determinable, else every child of the build root) are
    MOVED into ``backup_dir/cpp_itfs_cache/`` so the operation is reversible.
    The record always carries ``is_cpp_itfs`` + ``build_dir`` +
    ``module_names`` + ``invalidated_unix`` (even when nothing was cached
    yet) so integrate can later verify a fresh rebuild actually landed.

    Args:
        target_file: The file being patched (gates the operation).
        backup_dir: Directory to move affected cache dirs into.
        build_dir_override: Test-only override for the cpp_itfs build root.

    Returns:
        A status dict (``ok`` / ``skipped`` / ``failed``) recording what was
        moved and the invalidation metadata.
    """
    if not _target_is_in_aiter_cpp_itfs(target_file):
        return {
            "status": "skipped",
            "is_cpp_itfs": False,
            "reason": "target not under aiter/csrc/cpp_itfs/",
        }
    build_dir = build_dir_override or _aiter_cpp_itfs_build_dir()
    module_names = _cpp_itfs_module_names(target_file)
    scope = "module" if module_names else "build_root"
    record: dict[str, Any] = {
        "is_cpp_itfs": True,
        "build_dir": str(build_dir),
        "module_names": module_names,
        "scope": scope,
        "invalidated_at": now_iso(timespec="auto"),
        "invalidated_unix": time.time(),
    }
    if not build_dir.exists():
        # Nothing cached yet -> the re-baseline server builds fresh.
        record.update(
            {
                "status": "skipped",
                "reason": "cpp_itfs build dir does not exist",
                "moved": [],
            }
        )
        return record
    try:
        if module_names:
            to_move: list[Path] = []
            for md in module_names:
                to_move.extend(p for p in build_dir.glob(f"{md}_*") if p.is_dir())
        else:
            to_move = [p for p in build_dir.iterdir() if p.is_dir()]
    except OSError as exc:
        record.update({"status": "failed", "error": f"failed to scan {build_dir}: {exc}"})
        return record
    # De-dup: a shared .cuh can match overlapping MD_NAME globs.
    to_move = sorted({p.resolve() for p in to_move})
    if not to_move:
        record.update(
            {
                "status": "skipped",
                "reason": "no matching cpp_itfs cache entries",
                "moved": [],
            }
        )
        return record
    cache_backup_root = backup_dir / "cpp_itfs_cache"
    moved: list[dict[str, str]] = []
    try:
        cache_backup_root.mkdir(parents=True, exist_ok=True)
        for src in to_move:
            dst = cache_backup_root / src.name
            if dst.exists():
                record.update(
                    {
                        "status": "failed",
                        "error": f"cpp_itfs cache backup path already exists: {dst}",
                        "moved": moved,
                    }
                )
                return record
            shutil.move(str(src), str(dst))
            moved.append({"src": str(src), "backup_path": str(dst)})
    except (OSError, shutil.Error) as exc:
        record.update(
            {
                "status": "failed",
                "error": f"shutil.move failed: {exc}",
                "moved": moved,
            }
        )
        return record
    record.update({"status": "ok", "moved": moved})
    return record


def _restore_aiter_cpp_itfs_cache(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Reverse :func:`_invalidate_aiter_cpp_itfs_cache` (revert path).

    Moves each backed-up cache dir back to its original location, removing
    any dir the re-baseline server regenerated there first so the pre-patch
    runtime cache is restored bit-for-bit.

    Args:
        cache_backup: The backup record from
            :func:`_invalidate_aiter_cpp_itfs_cache`.

    Returns:
        A status dict with ``status`` of ``ok``, ``skipped``, or ``failed``.
    """
    if not isinstance(cache_backup, dict) or cache_backup.get("status") != "ok":
        return {"status": "skipped", "reason": "no cpp_itfs cache backup recorded"}
    moved = cache_backup.get("moved") or []
    if not moved:
        return {"status": "skipped", "reason": "nothing was moved"}
    # ``src`` is an ``rmtree`` target read from an untrusted manifest; only the
    # aiter cpp_itfs runtime build dir is a legitimate restore location.
    build_dir = _aiter_cpp_itfs_build_dir()
    restored: list[str] = []
    for entry in moved:
        src = Path(entry.get("src", ""))  # original cache location
        backup_path = Path(entry.get("backup_path", ""))
        if not str(src) or not str(backup_path) or not backup_path.exists():
            continue
        if not _within_root(src, build_dir):
            log.warning(
                "revert: skipping cpp_itfs cache restore; recorded src %s is outside the "
                "aiter cpp_itfs build dir %s ($AITER_ROOT_DIR/$HOME differ across "
                "apply/revert, or a forged manifest). Cache left invalidated; next call recompiles.",
                src,
                build_dir,
            )
            continue
        if src.exists():
            try:
                shutil.rmtree(src)
            except OSError as exc:
                return {
                    "status": "failed",
                    "error": f"failed to clear regenerated cache dir {src}: {exc}",
                }
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup_path), str(src))
        except (OSError, shutil.Error) as exc:
            return {
                "status": "failed",
                "error": f"shutil.move failed during restore: {exc}",
            }
        restored.append(str(src))
    return {"status": "ok", "restored": restored}


def verify_cpp_itfs_rebuilt(cache_backup: dict[str, Any]) -> dict[str, Any]:
    """Assert the re-baseline actually runtime-recompiled the patched kernel.

    After :func:`_invalidate_aiter_cpp_itfs_cache` moved the stale cache
    aside, a faithful re-baseline must repopulate
    ``<build_dir>/<md_name>_*/lib.so`` (the served binary) with an mtime at
    or after the invalidation. If no such fresh ``lib.so`` exists, the server
    dlopened a stale binary (or the kernel was never exercised) and the
    measured gain is meaningless -- the integrate KEEP/REVERT gate uses this
    to flag/abort instead of scoring a stale binary.

    Args:
        cache_backup: The invalidation record from
            :func:`_invalidate_aiter_cpp_itfs_cache`.

    Returns:
        A dict ``{"verified": bool, ...}``. ``verified`` is ``True`` for
        non-cpp_itfs targets so the caller's gate is a strict no-op off the
        cpp_itfs path.
    """
    if not isinstance(cache_backup, dict) or not cache_backup.get("is_cpp_itfs"):
        return {"verified": True, "status": "skipped", "reason": "non-cpp_itfs target"}
    build_dir_raw = cache_backup.get("build_dir", "")
    since = float(cache_backup.get("invalidated_unix") or 0.0)
    module_names = list(cache_backup.get("module_names") or [])
    # Path("") is Path("."), so an empty/missing build_dir must be rejected
    # before globbing — otherwise verify walks the process CWD.
    if not str(build_dir_raw).strip():
        return {
            "verified": False,
            "status": "stale",
            "reason": f"cpp_itfs build dir absent after re-baseline: {build_dir_raw}",
        }
    build_dir = Path(build_dir_raw)
    if not build_dir.exists():
        return {
            "verified": False,
            "status": "stale",
            "reason": f"cpp_itfs build dir absent after re-baseline: {build_dir}",
        }
    globs = [f"{md}_*/lib.so" for md in module_names] or ["*/lib.so"]
    fresh: list[str] = []
    for pattern in globs:
        for so in build_dir.glob(pattern):
            try:
                mtime = so.stat().st_mtime
            except OSError:
                continue
            # 1s slack absorbs build-dir-create vs lib.so-flush ordering.
            if mtime + 1.0 >= since:
                fresh.append(str(so))
    if fresh:
        return {"verified": True, "status": "ok", "fresh_lib_so": sorted(set(fresh))[:8]}
    return {
        "verified": False,
        "status": "stale",
        "reason": ("no freshly-built cpp_itfs lib.so found after re-baseline; served binary is stale"),
        "build_dir": str(build_dir),
        "module_names": module_names,
    }


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
    """The strategy every layout starts from: nothing compiled, nothing to rebuild."""
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
    """The rebuild strategy for a target inside an editable ``/sgl-workspace`` checkout."""
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
    """The rebuild strategy for a target in an installed framework tree or a FlyDSL checkout."""
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

report = {}
for name in sys.argv[1:]:
    try:
        spec = importlib.util.find_spec(name)
    except BaseException as exc:  # noqa: BLE001 - a half-installed parent raises anything
        report[name] = {"importable": False, "error": f"{type(exc).__name__}: {exc}"}
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


def revert_kernel_patch(manifest_path: str | Path) -> dict[str, Any]:
    """Revert a previously applied kernel patch from its manifest.

    Restores backed-up artifacts and source, clears Python caches when the
    target is Python, restores recorded AITER build state and serving modules,
    and fans out a best-effort revert to multi-node pods. The manifest is updated
    in place with the reverted status.

    Args:
        manifest_path (str | Path): Path to the apply manifest JSON file.

    Returns:
        dict[str, Any]: A result dict with ``status``, ``manifest_path``,
            ``restored_paths``, ``reverted_at`` and, when fan-out ran, a
            ``multinode_revert`` block.
    """
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not a JSON object: {manifest_file}")
    if str(manifest.get("status") or "") == "reverted":
        # An apply that reverted itself and is then unwound again by the stack would
        # otherwise report restoring-from-nothing as an unfinished teardown, on a tree
        # that is in fact clean, and every retry would repeat it.
        return {
            "status": "skipped",
            "manifest_path": str(manifest_file),
            "restored_paths": list(manifest.get("restored_paths") or []),
            "reverted_at": manifest.get("reverted_at"),
            "already_reverted": True,
        }
    # The manifest is untrusted at revert time: confine every copy source to
    # the apply-time backup tree (this manifest's own directory).
    backup_root = manifest_file.resolve().parent
    steps = [
        _restore_backed_up_artifacts(manifest, backup_root),
        _restore_source_backup_set(manifest, backup_root)
        if manifest.get("source_backups")
        else _restore_primary_source_backup(manifest, backup_root),
        _restore_jit_build_state(manifest, backup_root),
        _restore_cpp_itfs_cache_state(manifest),
        _revert_on_patched_pods(manifest),
    ]
    return _write_revert_outcome(manifest_file, manifest, steps)


class _RestoreTally(NamedTuple):
    """One revert step's outcome: what it put back, what it refused to, and what it reports."""

    restored: list[str]
    untrusted: list[dict[str, str]]
    issues: list[dict[str, Any]]
    reported: dict[str, Any]


def _restore_backed_up_artifacts(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
    """Put every backed-up compiled artifact back where the apply found it."""
    restored: list[str] = []
    untrusted: list[dict[str, str]] = []
    for item in manifest.get("artifacts", []):
        src = Path(item["backup_path"])
        dst = Path(item["path"])
        if not _revert_backup_trusted(item.get("backup_path", ""), backup_root):
            untrusted.append(
                {
                    "kind": "artifact",
                    "path": str(item.get("path", "")),
                    "backup_path": str(item.get("backup_path", "")),
                }
            )
            log.warning("revert: skipping artifact with untrusted backup_path %r", item.get("backup_path"))
            continue
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
    return _RestoreTally(restored, untrusted, [], {})


def _restore_source_backup_set(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
    """Undo a multi-file apply from its per-path disposition records, newest write first."""
    restored: list[str] = []
    untrusted: list[dict[str, str]] = []
    cache_cleared = False
    for entry in reversed(manifest["source_backups"]):
        dst = Path(entry["path"])
        if entry.get("disposition") == "added":
            dst.unlink(missing_ok=True)
            restored.append(str(dst))
        else:
            bp = entry.get("backup_path")
            if bp and not _revert_backup_trusted(bp, backup_root):
                untrusted.append(
                    {
                        "kind": "source_backups",
                        "path": str(entry.get("path", "")),
                        "backup_path": str(bp),
                    }
                )
                log.warning("revert: skipping source backup with untrusted backup_path %r", bp)
            elif bp and Path(bp).exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bp, dst, follow_symlinks=False)
                restored.append(str(dst))
        if dst.suffix.lower() in PYTHON_SOURCE_SUFFIXES and not cache_cleared:
            manifest["revert_cache_clear"] = _clear_python_kernel_caches(dst)
            cache_cleared = True
    return _RestoreTally(restored, untrusted, [], {})


def _restore_primary_source_backup(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
    """Undo a whole-file apply from the single ``source_backup`` record it wrote."""
    source_backup = manifest.get("source_backup") or {}
    if not source_backup:
        return _RestoreTally([], [], [], {})
    src = Path(source_backup["backup_path"])
    dst = Path(source_backup["path"])
    if not _revert_backup_trusted(source_backup.get("backup_path", ""), backup_root):
        log.warning(
            "revert: skipping source backup with untrusted backup_path %r",
            source_backup.get("backup_path"),
        )
        untrusted = [
            {
                "kind": "source_backup",
                "path": str(source_backup.get("path", "")),
                "backup_path": str(source_backup.get("backup_path", "")),
            }
        ]
        return _RestoreTally([], untrusted, [], {})
    if not src.exists():
        return _RestoreTally([], [], [], {})
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if dst.suffix.lower() in PYTHON_SOURCE_SUFFIXES:
        manifest["revert_cache_clear"] = _clear_python_kernel_caches(dst)
    return _RestoreTally([str(dst)], [], [], {})


def _restore_jit_build_state(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
    """Put the AITER build state and serving modules back before any multi-node fan-out."""
    jit_build_backup = manifest.get("jit_build_backup") or {}
    if jit_build_backup.get("status") not in {"ok", "clean"}:
        return _RestoreTally([], [], [], {})
    strategy = manifest.get("strategy") or {}
    # Singular key is retained for manifests created by earlier releases.
    pinned = str(strategy.get("jit_build_dir") or "").strip()
    snapshot_jit_dirs = list(strategy.get("jit_build_dirs") or [])
    if not pinned and len(snapshot_jit_dirs) == 1:
        pinned = str(snapshot_jit_dirs[0])
    jit_build_restore = _restore_aiter_jit_build(
        jit_build_backup,
        expected_jit_build_dir=pinned or None,
        backup_root=backup_root,
    )
    reported = {"jit_build_restore": jit_build_restore}
    if jit_build_restore.get("status") != "ok":
        return _RestoreTally([], [], [{"kind": "jit_build_restore", **jit_build_restore}], reported)
    restored_to = jit_build_restore.get("restored_to")
    return _RestoreTally([str(restored_to)] if restored_to else [], [], [], reported)


def _restore_cpp_itfs_cache_state(manifest: dict[str, Any]) -> _RestoreTally:
    """Put the aiter cpp_itfs runtime cache moved aside during apply back in place."""
    cpp_itfs_cache_backup = manifest.get("cpp_itfs_cache_backup") or {}
    if cpp_itfs_cache_backup.get("status") != "ok":
        return _RestoreTally([], [], [], {})
    cpp_itfs_cache_restore = _restore_aiter_cpp_itfs_cache(cpp_itfs_cache_backup)
    manifest["cpp_itfs_cache_restore"] = cpp_itfs_cache_restore
    if cpp_itfs_cache_restore.get("status") != "ok":
        return _RestoreTally([], [], [], {})
    return _RestoreTally(list(cpp_itfs_cache_restore.get("restored", [])), [], [], {})


def _revert_on_patched_pods(manifest: dict[str, Any]) -> _RestoreTally:
    """Fan a best-effort revert out to every pod that received the apply."""
    multinode_info = manifest.get(_MANIFEST_MULTINODE) or {}
    if not (multinode_info.get("records_by_host") or multinode_info.get("host_backup_map")):
        return _RestoreTally([], [], [], {})
    source_backup = manifest.get("source_backup") or {}
    target_path = multinode_info.get("target_path") or (source_backup.get("path") if source_backup else "")
    backup_map = multinode_info.get("host_backup_map") or {}
    records_by_host = multinode_info.get("records_by_host") or {}
    mn_revert: dict[str, Any] = {}
    if records_by_host or (target_path and backup_map):
        try:
            mn_revert = _dispatch_multinode_revert(
                target_path=target_path,
                backup_map=backup_map,
                records_by_host=records_by_host,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            mn_revert = {"status": "failed", "error": str(exc)}
    if not mn_revert:
        return _RestoreTally([], [], [], {})
    failed = [] if mn_revert.get("status") == "ok" else [{"kind": "multinode_revert", **mn_revert}]
    return _RestoreTally([], [], failed, {"multinode_revert": mn_revert})


def _write_revert_outcome(
    manifest_file: Path,
    manifest: dict[str, Any],
    steps: list[_RestoreTally],
) -> dict[str, Any]:
    """Record how complete the revert was in the manifest and report it to the caller."""
    restored = [path for step in steps for path in step.restored]
    untrusted = [entry for step in steps for entry in step.untrusted]
    issues = [issue for step in steps for issue in step.issues]
    reported = {key: record for step in steps for key, record in step.reported.items()}
    reverted_at = now_iso(timespec="auto")
    partial = bool(untrusted or issues)
    manifest["status"] = "reverted_partial" if partial else "reverted"
    manifest["reverted_at"] = reverted_at
    manifest["restored_paths"] = restored
    manifest.update(reported)
    if untrusted:
        manifest["skipped_untrusted_backups"] = untrusted
    if issues:
        manifest["revert_issues"] = issues
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "partial" if partial else "ok",
        "manifest_path": str(manifest_file),
        "restored_paths": restored,
        "reverted_at": reverted_at,
        **reported,
    }
    if untrusted:
        result["skipped_untrusted_backups"] = untrusted
    if issues:
        result["revert_issues"] = issues
    return result


def finalize_kernel_patch(manifest_path: str | Path) -> dict[str, Any]:
    """Delete backups after an integrated patch becomes the new baseline."""
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not a JSON object: {manifest_file}")
    manifest_status = str(manifest.get("status") or "")
    if manifest_status == "finalized":
        return {
            "status": "ok",
            "manifest_path": str(manifest_file),
            "reason": "already finalized",
        }
    if manifest_status not in {"applied", "finalized_partial"}:
        return {
            "status": "failed",
            "manifest_path": str(manifest_file),
            "error": f"cannot finalize manifest in state {manifest_status!r}",
        }
    backup_root = manifest_file.resolve().parent
    issues: list[dict[str, Any]] = []
    multinode = manifest.get(_MANIFEST_MULTINODE) or {}
    records_by_host = dict(multinode.get("records_by_host") or {})
    remote_result: dict[str, Any] = {}
    if records_by_host:
        remote_result = _dispatch_multinode_finalize(records_by_host)
        if remote_result.get("status") != "ok":
            issues.append({"kind": "multinode_finalize", **remote_result})

    deletion = _delete_recorded_backups(_recorded_backup_paths(manifest), backup_root)
    deleted = deletion.deleted
    issues.extend(deletion.issues)

    manifest["status"] = "finalized_partial" if issues else "finalized"
    manifest["finalized_at"] = now_iso(timespec="auto")
    manifest["finalize_deleted"] = deleted
    if remote_result:
        manifest["multinode_finalize"] = remote_result
    if issues:
        manifest["finalize_issues"] = issues
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "partial" if issues else "ok",
        "manifest_path": str(manifest_file),
        "deleted": deleted,
        "issues": issues,
        "multinode_finalize": remote_result,
    }


def _recorded_backup_paths(manifest: dict[str, Any]) -> set[str]:
    """Every backup path this manifest's apply recorded, across all its record shapes."""
    candidates: set[str] = set()
    for entry in manifest.get("artifacts") or []:
        candidates.add(str(entry.get("backup_path") or ""))
    for entry in manifest.get("source_backups") or []:
        candidates.add(str(entry.get("backup_path") or ""))
    source_backup = manifest.get("source_backup") or {}
    candidates.add(str(source_backup.get("backup_path") or ""))
    for key in ("jit_build_backup", "cpp_itfs_cache_backup"):
        record = manifest.get(key) or {}
        candidates.add(str(record.get("backup_path") or ""))
        candidates.add(str(record.get("modules_backup_path") or ""))
    return candidates


def _multi_root_strategies(live_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Compute the set of distinct rebuild strategies across edited files.

    A multi-file patch can touch files in several framework roots (e.g. a
    ``csrc/*.cu`` plus an ``aiter/ops/triton/*.py``). Driving rebuild off the
    primary target alone would skip compilation for companion compiled files, so
    derive ``compiled``/rebuild from the **whole** edited set.

    One root can yield several rebuild modes -- SGLang's ``kernels/jit``
    sources defer to the runtime while its ``kernels/aot/csrc`` sources still
    need the editable install -- so the identity of a rebuild is the root
    *and* the mode. Deduplicating on the root alone would keep whichever file
    the patch happened to list first and silently skip the other's rebuild.

    Args:
        live_paths (Iterable[Path]): The live target paths the patch writes.

    Returns:
        list[dict[str, Any]]: One strategy dict per distinct root and rebuild
            mode that needs a rebuild (compiled roots only), each as returned
            by :func:`_detect_strategy`.
    """
    seen: set[tuple[str, str]] = set()
    strategies: list[dict[str, Any]] = []
    for p in live_paths:
        try:
            strat = _detect_strategy(p)
        except ValueError:
            continue
        if not strat["compiled"]:
            continue
        key = (strat["root"] or str(p.parent), strat["rebuild_mode"])
        if key in seen:
            continue
        seen.add(key)
        strategies.append(strat)
    return strategies


@dataclass(frozen=True)
class KernelPatchRequest:
    """One kernel patch to land: what to apply, where, and how to rebuild.

    Attributes:
        patch_path: The replacement source file, or (snapshot mode) the unified
            diff manifest.
        target_file: The primary file (drives timing/rebuild root).
        backup_root: Root directory for backups and the manifest.
        kernel_id: Identifier for the kernel being patched.
        artifact_paths: Explicit compiled artifacts to back up in addition to
            discovered ones.
        rebuild_command: Override rebuild argv; a string is shlex-split and run
            argv-only (shell=False), never via ``bash -lc``. Commands using
            shell control operators, or invoking ``bash``/``sh`` with
            ``-c``/``-lc``, return ``status='failed'`` with
            ``error_class='invalid_rebuild_command'``.
        rebuild_timeout_sec: Rebuild subprocess timeout in seconds.
        skip_rebuild: When ``True``, skip the rebuild step.
        dry_run: Prepare backups/manifest only, without applying.
        snapshot_dir: When set, enables snapshot mode and holds byte-exact final
            contents mirrored at each write path.
        repo_root: Root the repo-relative snapshot paths resolve against.
        producer_manifest: Canonical producer manifest recorded for deployment
            provenance.
    """

    patch_path: str | Path
    target_file: str | Path
    backup_root: str | Path
    kernel_id: str = ""
    artifact_paths: Iterable[str] | None = None
    rebuild_command: list[str] | str | None = None
    rebuild_timeout_sec: int = 1800
    skip_rebuild: bool = False
    dry_run: bool = False
    snapshot_dir: str | Path | None = None
    repo_root: str | Path | None = None
    producer_manifest: str | Path | None = None


def apply_kernel_patch(request: KernelPatchRequest) -> dict[str, Any]:
    """Apply an optimized kernel file with backup, rebuild, and fan-out.

    Two modes:

    - **snapshot mode** (``snapshot_dir`` given): ``patch_path`` is a unified
      diff used only as a *manifest* of intended changes; the byte-exact final
      contents come from ``snapshot_dir``. The whole multi-file set lands
      atomically (all-or-nothing) or the repo is left untouched — see
      :func:`apply_snapshot`. This is the path that satisfies the "entire patch
      lands byte-for-byte or nothing changes" contract.
    - **legacy full-source mode** (no ``snapshot_dir``): ``patch_path`` is a
      complete replacement file for a single ``target_file`` (whole-file copy).

    Both back up the prior state, rebuild compiled targets, fan out to multi-node
    pods, and revert automatically on rebuild failure.

    Args:
        request (KernelPatchRequest): The patch, its target, and the rebuild options.

    Returns:
        dict[str, Any]: A result dict with ``status`` and, on success, the
            manifest path, target, backup dir, compiled flag, artifact count,
            cache-clear / rebuild / jit-build records, and an optional
            ``multinode`` block.
    """
    try:
        if request.snapshot_dir is not None:
            return _apply_kernel_patch_snapshot(request)
        return _apply_whole_file_patch(request)
    except _PatchRefused as refusal:
        return refusal.result


class _PatchRefused(Exception):
    """An apply that cannot go on; ``result`` is the dict its caller reports."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("error") or "kernel patch refused"))
        self.result = result


def _patch_rebuild_argv(request: KernelPatchRequest) -> list[str]:
    """The argv this call overrides the strategy's rebuild command with."""
    try:
        return _coerce_rebuild_command(request.rebuild_command) if request.rebuild_command else []
    except ValueError as exc:
        raise _PatchRefused({"status": "failed", "error_class": "invalid_rebuild_command", "error": str(exc)}) from exc


@dataclass(frozen=True)
class _PatchDeployment:
    """One apply in progress: the backup tree it owns and the manifest recording it."""

    target: Path
    backup_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]

    def persist(self) -> None:
        """Write the manifest into the backup directory."""
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def record(self, key: str, value: Any) -> None:
        """Add one record to the manifest and persist it, so a later failure can still unwind."""
        self.manifest[key] = value
        self.persist()


@dataclass
class _PodFanout:
    """Per-pod apply records gathered as a patch is pushed to every inference pod."""

    records_by_host: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    backup_map: dict[str, str] = field(default_factory=dict)

    def absorb(self, dispatch_result: dict[str, Any]) -> None:
        """Index one dispatch's per-node entries by pod host."""
        for entry in dispatch_result.get("per_node", []) or []:
            host = (entry.get("host") or "").strip()
            if not host:
                continue
            self.records_by_host.setdefault(host, []).append(dict(entry))
            backup_path = (entry.get("backup_path") or "").strip()
            if backup_path:
                self.backup_map[host] = backup_path

    def block(self, status: str, target: Path, pod_backup_dir: str) -> dict[str, Any]:
        """The manifest ``multinode`` block describing this fan-out."""
        return {
            "status": status,
            "target_path": str(target),
            "backup_dir_on_pod": pod_backup_dir,
            "host_backup_map": self.backup_map,
            "records_by_host": self.records_by_host,
        }


class _RebuildPlan(NamedTuple):
    """How a landed patch is rebuilt: the roots to rebuild and the caches to clear first."""

    strategies: list[dict[str, Any]]
    jit_strategy: dict[str, Any]
    jit_target: Path
    jit_build_dir: Path | None
    cpp_itfs_target: Path
    command_override: list[str]
    timeout_sec: int
    pod_records: dict[str, list[dict[str, Any]]] | None


class _RebuildRecords(NamedTuple):
    """What cache invalidation and rebuild recorded for the manifest."""

    rebuild: dict[str, Any]
    jit_build_backup: dict[str, Any]
    cpp_itfs_cache_backup: dict[str, Any]


def _rebuild_not_run() -> _RebuildRecords:
    """The records an apply carries when nothing had to be rebuilt."""
    return _RebuildRecords(
        rebuild={"status": "skipped", "reason": "source-only patch or skip_rebuild=true"},
        jit_build_backup={"status": "skipped", "reason": "rebuild not run"},
        cpp_itfs_cache_backup={"status": "skipped", "reason": "rebuild not run", "is_cpp_itfs": False},
    )


def _refuse_untrusted_jit_strategies(jit_strategies: Iterable[dict[str, Any]]) -> None:
    """Stop before any mutation when a runtime-JIT strategy has no trustworthy build dir."""
    for strategy in jit_strategies:
        invalidation_error = _runtime_jit_invalidation_error(strategy)
        if invalidation_error:
            raise _PatchRefused(
                {"status": "failed", "error_class": "aiter_jit_invalidation_failed", "error": invalidation_error}
            )


def _invalidate_and_rebuild(deployment: _PatchDeployment, plan: _RebuildPlan) -> _RebuildRecords:
    """Clear every cache the patched roots serve from, then rebuild those roots."""
    jit_build_backup = _invalidate_jit_build(deployment, plan)
    cpp_itfs_cache_backup = _invalidate_cpp_itfs_cache(deployment, plan)
    return _RebuildRecords(
        rebuild=_rebuild_patched_roots(deployment, plan),
        jit_build_backup=jit_build_backup,
        cpp_itfs_cache_backup=cpp_itfs_cache_backup,
    )


def _invalidate_jit_build(deployment: _PatchDeployment, plan: _RebuildPlan) -> dict[str, Any]:
    """Move the AITER JIT build state aside so the next import recompiles the patched kernel."""
    if plan.pod_records is not None:
        # Each pod's record in manifest["multinode"] holds its JIT backup; revert and finalize read it there.
        backup: dict[str, Any] = {"status": "remote"}
        invalidation_error = _remote_jit_invalidation_error(plan.pod_records)
    else:
        backup = _invalidate_aiter_jit_build(plan.jit_target, deployment.backup_dir, jit_build_dir=plan.jit_build_dir)
        invalidation_error = _runtime_jit_invalidation_error(plan.jit_strategy, backup)
    if backup.get("status") in {"ok", "clean"}:
        # Persist the backup record before rebuild so a failure can still restore.
        deployment.record("jit_build_backup", backup)
    if backup.get("status") != "failed" and not invalidation_error:
        return backup
    revert = revert_kernel_patch(deployment.manifest_path)
    detail = invalidation_error or str(backup.get("error") or "") or str(backup.get("reason") or "")
    raise _PatchRefused(
        {
            "status": "failed",
            "error_class": "aiter_jit_invalidation_failed",
            "error": f"AITER JIT cache invalidation failed: {detail}",
            "manifest_path": str(deployment.manifest_path),
            "jit_build_backup": backup,
            "revert": revert,
        }
    )


def _invalidate_cpp_itfs_cache(deployment: _PatchDeployment, plan: _RebuildPlan) -> dict[str, Any]:
    """Move the aiter cpp_itfs runtime cache aside so the re-baseline recompiles from clean state.

    aiter cpp_itfs kernels runtime-compile into ``$HOME/.aiter/build/`` under a
    param-hashed dir name, so pristine and patched collide.
    """
    backup = _invalidate_aiter_cpp_itfs_cache(plan.cpp_itfs_target, deployment.backup_dir)
    if backup.get("status") == "ok":
        # Persist before rebuild so a failure can restore the moved-aside cache.
        deployment.record("cpp_itfs_cache_backup", backup)
        return backup
    if backup.get("status") != "failed":
        return backup
    # The manifest owns both local and per-pod source/JIT backups.
    revert = revert_kernel_patch(deployment.manifest_path)
    raise _PatchRefused(
        {
            "status": "failed",
            "error_class": "aiter_cpp_itfs_invalidation_failed",
            "error": f"aiter cpp_itfs runtime cache invalidation failed: {backup.get('error')}",
            "manifest_path": str(deployment.manifest_path),
            "cpp_itfs_cache_backup": backup,
            "revert": revert,
        }
    )


def _rebuild_patched_roots(deployment: _PatchDeployment, plan: _RebuildPlan) -> dict[str, Any]:
    """Rebuild every distinct compiled root the patch touched."""
    record: dict[str, Any] = {}
    for strategy in plan.strategies:
        record = _run_strategy_rebuild(
            strategy,
            command_override=plan.command_override,
            fallback_cwd=deployment.target.parent,
            timeout_sec=plan.timeout_sec,
        )
        if not _rebuild_ok_to_proceed(record):
            revert = revert_kernel_patch(deployment.manifest_path)
            raise _PatchRefused(
                {
                    "status": "failed",
                    "error": "rebuild failed; original source/artifacts restored",
                    "manifest_path": str(deployment.manifest_path),
                    "rebuild": record,
                    "revert": revert,
                }
            )
    return record


def _finish_apply(
    deployment: _PatchDeployment,
    records: _RebuildRecords,
    cache_clear: dict[str, Any],
    compiled: bool,
    multinode: dict[str, Any],
) -> dict[str, Any]:
    """Mark the manifest applied and report the apply to its caller."""
    manifest = deployment.manifest
    manifest["status"] = "applied"
    manifest["applied_at"] = now_iso(timespec="auto")
    manifest["rebuild"] = records.rebuild
    manifest["cache_clear"] = cache_clear
    # Surface skipped reasons too, for manifest auditing and integrate verification.
    if records.jit_build_backup.get("status") in {"ok", "clean", "remote", "skipped"}:
        manifest["jit_build_backup"] = records.jit_build_backup
    if records.cpp_itfs_cache_backup.get("status") in {"ok", "skipped"}:
        manifest["cpp_itfs_cache_backup"] = records.cpp_itfs_cache_backup
    deployment.persist()
    result: dict[str, Any] = {
        "status": "ok",
        "manifest_path": str(deployment.manifest_path),
        "target_file": str(deployment.target),
        "backup_dir": str(deployment.backup_dir),
        "compiled": compiled,
        "artifact_count": len(deployment.manifest["artifacts"]),
        "cache_clear": cache_clear,
        "rebuild": records.rebuild,
        "jit_build_backup": records.jit_build_backup,
        "cpp_itfs_cache_backup": records.cpp_itfs_cache_backup,
    }
    # Only attach the multinode key when fan-out actually ran.
    if multinode:
        result[_MANIFEST_MULTINODE] = multinode
    return result


def _apply_whole_file_patch(request: KernelPatchRequest) -> dict[str, Any]:
    """Replace one target file with the patch, rebuild its root, and publish it to every pod."""
    patch = Path(request.patch_path).resolve()
    target = Path(request.target_file).resolve()
    if not patch.is_file():
        raise _PatchRefused({"status": "failed", "error": f"patch_path does not exist: {patch}"})
    if not target.is_file():
        raise _PatchRefused({"status": "failed", "error": f"target_file does not exist: {target}"})
    try:
        strategy = _detect_strategy(target)
        _validate_patch_source(patch, target)
    except (ValueError, OSError, py_compile.PyCompileError) as exc:
        raise _PatchRefused({"status": "failed", "error": str(exc)}) from exc
    # Coerce/validate the rebuild_command BEFORE any file/pod mutation so an
    # invalid (shell-control) command fails all-or-nothing instead of aborting
    # after the patch is already copied onto the live target / fanned out.
    command_override = _patch_rebuild_argv(request)
    if not request.skip_rebuild and not _is_multi_node():
        _refuse_untrusted_jit_strategies([strategy])

    deployment = _prepare_whole_file_apply(request, patch, target, strategy)
    if request.dry_run:
        return {"status": "ok", "dry_run": True, "manifest_path": str(deployment.manifest_path)}

    shutil.copy2(patch, target)
    cache_clear = (
        _clear_python_kernel_caches(target)
        if target.suffix.lower() in PYTHON_SOURCE_SUFFIXES
        else {"status": "skipped", "reason": "not a python source target"}
    )
    multinode = _fan_whole_file_out_to_pods(request, deployment, patch, strategy) if _is_multi_node() else {}

    compiled = bool(strategy["compiled"])
    records = _rebuild_not_run()
    if compiled and not request.skip_rebuild:
        plan = _whole_file_rebuild_plan(request, target, strategy, command_override, multinode)
        records = _invalidate_and_rebuild(deployment, plan)
    return _finish_apply(deployment, records, cache_clear, compiled, multinode)


def _prepare_whole_file_apply(
    request: KernelPatchRequest,
    patch: Path,
    target: Path,
    strategy: dict[str, Any],
) -> _PatchDeployment:
    """Back up the target and its compiled artifacts, and write the ``prepared`` manifest."""
    backup_dir = _claim_backup_dir(Path(request.backup_root), request.kernel_id, target)
    source_backup = _copy_to_backup(target, backup_dir, "source")
    artifacts: list[dict[str, str]] = []
    if strategy["compiled"]:
        found = _discover_artifacts(strategy["artifact_roots"], request.artifact_paths)
        artifacts = [_copy_to_backup(path, backup_dir, "artifacts") for path in found]
    manifest: dict[str, Any] = {
        "status": "prepared",
        "kernel_id": request.kernel_id,
        "patch_path": str(patch),
        "target_file": str(target),
        "source_backup": source_backup,
        "artifacts": artifacts,
        "strategy": {
            "compiled": strategy["compiled"],
            "root": strategy["root"],
            "deploy_roots": [str(path) for path in strategy.get("deploy_roots") or []],
            "rebuild_modes": [strategy["rebuild_mode"]],
            "jit_build_dirs": ([strategy["jit_build_dir"]] if strategy.get("jit_build_dir") else []),
        },
        "created_at": now_iso(timespec="auto"),
    }
    if request.producer_manifest:
        manifest["producer_manifest"] = str(Path(request.producer_manifest).resolve())
    backup_dir.mkdir(parents=True, exist_ok=True)
    deployment = _PatchDeployment(target, backup_dir, backup_dir / "manifest.json", manifest)
    deployment.persist()
    return deployment


def _fan_whole_file_out_to_pods(
    request: KernelPatchRequest,
    deployment: _PatchDeployment,
    patch: Path,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    """Push the patched file to every RayJob pod, restoring the sandbox copy if one refuses it."""
    pod_backup_dir = os.environ.get(
        "HYPERLOOM_MN_KERNEL_BACKUP_DIR",
        _MN_POD_BACKUP_DIR_DEFAULT,
    )
    jit_build_dir = str(strategy.get("jit_build_dir") or "").strip()
    remote_jit_dir = jit_build_dir if (strategy["compiled"] and not request.skip_rebuild) else ""
    fanout = _PodFanout()
    try:
        fanout.absorb(
            _dispatch_multinode_apply(
                PodPatchApply(
                    target_file=deployment.target,
                    patch_path=patch,
                    kernel_id=request.kernel_id,
                    backup_dir_on_pod=pod_backup_dir,
                    jit_build_dir=remote_jit_dir,
                )
            )
        )
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        # Pod fan-out failed: revert sandbox copy.
        backup_path = deployment.manifest["source_backup"]["backup_path"]
        try:
            shutil.copy2(backup_path, deployment.target)
        except OSError:
            pass
        raise _PatchRefused(
            {
                "status": "failed",
                "error": (f"multi-node apply fan-out failed; sandbox copy reverted to {backup_path}: {exc}"),
                "manifest_path": str(deployment.manifest_path),
            }
        ) from exc
    # Persist multinode info now so a later rebuild failure reverts pod-side patches too.
    block = fanout.block("ok", deployment.target, pod_backup_dir)
    deployment.record(_MANIFEST_MULTINODE, block)
    return block


def _whole_file_rebuild_plan(
    request: KernelPatchRequest,
    target: Path,
    strategy: dict[str, Any],
    command_override: list[str],
    multinode: dict[str, Any],
) -> _RebuildPlan:
    """How the single patched file's framework root is rebuilt."""
    jit_build_dir = str(strategy.get("jit_build_dir") or "").strip()
    return _RebuildPlan(
        strategies=[strategy],
        jit_strategy=strategy,
        jit_target=target,
        jit_build_dir=Path(jit_build_dir) if jit_build_dir else None,
        cpp_itfs_target=target,
        command_override=command_override,
        timeout_sec=request.rebuild_timeout_sec,
        pod_records=multinode.get("records_by_host") if (multinode and jit_build_dir) else None,
    )


class _SnapshotPlan(NamedTuple):
    """A validated snapshot apply: what it writes, where, and which roots it rebuilds."""

    patch: Path
    target: Path
    descriptors: list[PatchTarget]
    repo_root: Path
    deploy_roots: list[Path]
    write_paths: list[Path]
    rebuild_strategies: list[dict[str, Any]]
    jit_strategies: list[dict[str, Any]]
    command_override: list[str]


def _apply_kernel_patch_snapshot(request: KernelPatchRequest) -> dict[str, Any]:
    """Snapshot-mode apply: land an entire multi-file patch atomically.

    See :func:`apply_kernel_patch` for the contract. The diff at
    ``request.patch_path`` is parsed only as a manifest; contents come from
    ``request.snapshot_dir``. Rebuild, JIT/cpp_itfs invalidation, and
    multi-node fan-out are driven off the **whole** edited set so companion
    compiled files and worker pods are not missed, and all-or-nothing spans
    apply to rebuild (a rebuild failure restores every touched path).

    Raises:
        _PatchRefused: When the patch cannot be validated, landed or rebuilt.
    """
    plan = _snapshot_plan(request)
    deployment = _prepare_snapshot_apply(request, plan)
    if request.dry_run:
        return {"status": "ok", "dry_run": True, "manifest_path": str(deployment.manifest_path)}

    touched = _land_snapshot(request, deployment, plan)
    cache_clear: dict[str, Any] = {"status": "skipped", "reason": "no python source target"}
    for path in plan.write_paths:
        if path.suffix.lower() in PYTHON_SOURCE_SUFFIXES:
            cache_clear = _clear_python_kernel_caches(path)
    multinode = _fan_snapshot_out_to_pods(request, deployment, plan) if _is_multi_node() else {}

    compiled = bool(plan.rebuild_strategies)
    records = _rebuild_not_run()
    if compiled and not request.skip_rebuild:
        records = _invalidate_and_rebuild(deployment, _snapshot_rebuild_plan(request, plan, multinode))
    result = _finish_apply(deployment, records, cache_clear, compiled, multinode)
    result["touched"] = touched
    return result


def _snapshot_plan(request: KernelPatchRequest) -> _SnapshotPlan:
    """Parse the patch manifest, resolve every destination, and pick the roots to rebuild."""
    patch = Path(request.patch_path).resolve()
    target = Path(request.target_file).resolve()
    if not patch.is_file():
        raise _PatchRefused({"status": "failed", "error": f"patch_path does not exist: {patch}"})
    try:
        descriptors = repo_relative_manifest(patch.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        raise _PatchRefused({"status": "failed", "error": f"unparseable patch: {exc}"}) from exc
    try:
        primary_strategy = _detect_strategy(target)
    except ValueError as exc:
        raise _PatchRefused({"status": "failed", "error": str(exc)}) from exc
    # Coerce/validate rebuild_command BEFORE any file/pod mutation (all-or-nothing).
    command_override = _patch_rebuild_argv(request)
    repo_root = _snapshot_repo_root(request, primary_strategy, target)
    # A caller-named repo is a declaration, not the guess rejected above.
    deploy_roots = [Path(path) for path in primary_strategy.get("deploy_roots") or []] or [repo_root]
    write_paths = _snapshot_write_paths(descriptors, repo_root, deploy_roots)
    rebuild_strategies = _multi_root_strategies(write_paths)
    jit_strategies = [
        strategy
        for strategy in rebuild_strategies
        if strategy.get("jit_build_dir") or strategy.get("rebuild_mode") == _REBUILD_MODE_RUNTIME_JIT
    ]
    if len(jit_strategies) > 1:
        raise _PatchRefused({"status": "failed", "error": "snapshot spans multiple AITER JIT roots"})
    if not request.skip_rebuild and not _is_multi_node():
        _refuse_untrusted_jit_strategies(jit_strategies)
    return _SnapshotPlan(
        patch=patch,
        target=target,
        descriptors=descriptors,
        repo_root=repo_root,
        deploy_roots=deploy_roots,
        write_paths=write_paths,
        rebuild_strategies=rebuild_strategies,
        jit_strategies=jit_strategies,
        command_override=command_override,
    )


def _snapshot_repo_root(request: KernelPatchRequest, primary_strategy: dict[str, Any], target: Path) -> Path:
    """The root the patch's repo-relative descriptors resolve against.

    The explicit root, else the strategy root; never ``target.parent`` -- a
    guessed root breaks the byte-for-byte contract.
    """
    resolved_root = str(request.repo_root or "") or primary_strategy["root"]
    if not resolved_root:
        raise _PatchRefused(
            {
                "status": "failed",
                "error": (
                    "snapshot mode requires a known repo root (pass repo_root or a "
                    f"target under a known framework root): {target}"
                ),
            }
        )
    return Path(resolved_root)


def _snapshot_write_paths(
    descriptors: list[PatchTarget],
    repo_root: Path,
    deploy_roots: list[Path],
) -> list[Path]:
    """Resolve every write descriptor to the absolute path it lands on."""
    write_paths: list[Path] = []
    for desc in descriptors:
        try:
            dest = _contained_dest(repo_root, desc.path, deploy_roots=deploy_roots)
        except ValueError as exc:
            raise _PatchRefused({"status": "failed", "error": str(exc), "path": desc.path}) from exc
        if desc.op is PatchOp.WRITE:
            write_paths.append(dest)
    return write_paths


def _prepare_snapshot_apply(request: KernelPatchRequest, plan: _SnapshotPlan) -> _PatchDeployment:
    """Back up the compiled artifacts the patch invalidates and write the ``prepared`` manifest."""
    backup_dir = _claim_backup_dir(Path(request.backup_root), request.kernel_id, plan.target)
    backup_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, str]] = []
    if plan.rebuild_strategies:
        roots: list[Path] = []
        for strategy in plan.rebuild_strategies:
            roots.extend(strategy["artifact_roots"])
        found = _discover_artifacts(roots, request.artifact_paths)
        artifacts = [_copy_to_backup(path, backup_dir, "artifacts") for path in found]
    manifest: dict[str, Any] = {
        "status": "prepared",
        "kernel_id": request.kernel_id,
        "patch_path": str(plan.patch),
        "target_file": str(plan.target),
        "mode": "snapshot",
        "descriptors": [asdict(desc) for desc in plan.descriptors],
        "artifacts": artifacts,
        "strategy": {
            "compiled": bool(plan.rebuild_strategies),
            "root": str(plan.repo_root),
            "deploy_roots": [str(path) for path in plan.deploy_roots],
            "rebuild_modes": sorted({strategy["rebuild_mode"] for strategy in plan.rebuild_strategies}),
            "jit_build_dirs": sorted(
                {str(strategy["jit_build_dir"]) for strategy in plan.jit_strategies if strategy.get("jit_build_dir")}
            ),
        },
        "created_at": now_iso(timespec="auto"),
    }
    if request.producer_manifest:
        manifest["producer_manifest"] = str(Path(request.producer_manifest).resolve())
    deployment = _PatchDeployment(plan.target, backup_dir, backup_dir / "manifest.json", manifest)
    deployment.persist()
    return deployment


def _land_snapshot(request: KernelPatchRequest, deployment: _PatchDeployment, plan: _SnapshotPlan) -> list[str]:
    """Write the whole snapshot onto the live tree and prove every path landed byte-exact."""
    applied = apply_snapshot(
        descriptors=plan.descriptors,
        snapshot_dir=request.snapshot_dir,
        repo_root=plan.repo_root,
        deploy_roots=plan.deploy_roots,
        backup_dir=deployment.backup_dir,
    )
    if applied["status"] != "ok":
        raise _PatchRefused(
            {
                "status": "failed",
                "error": applied.get("error"),
                "path": applied.get("path"),
                "manifest_path": str(deployment.manifest_path),
            }
        )
    deployment.manifest["source_backups"] = applied["source_backups"]
    # Keep a singular source_backup for the primary (legacy manifest readers).
    for entry in applied["source_backups"]:
        if Path(entry["path"]) == deployment.target and entry.get("backup_path"):
            deployment.manifest["source_backup"] = {"path": entry["path"], "backup_path": entry["backup_path"]}
            break
    deployment.persist()
    mismatch = _snapshot_mismatch(request, plan)
    if mismatch:
        revert_kernel_patch(deployment.manifest_path)
        raise _PatchRefused({"status": "failed", "error": mismatch, "manifest_path": str(deployment.manifest_path)})
    return applied["touched"]


def _snapshot_mismatch(request: KernelPatchRequest, plan: _SnapshotPlan) -> str:
    """How the live tree differs from the snapshot it was just given, or ``""``.

    Written paths must be byte-identical to the snapshot and deleted paths gone.
    """
    snap = Path(request.snapshot_dir)
    for desc in plan.descriptors:
        dest = _contained_dest(plan.repo_root, desc.path, deploy_roots=plan.deploy_roots)
        if desc.op is PatchOp.DELETE:
            if dest.exists():
                return f"post-verify: {desc.path} not deleted"
            continue
        src = snap / desc.path
        if src.is_symlink():
            if not dest.is_symlink() or os.readlink(dest) != os.readlink(src):
                return f"post-verify symlink mismatch: {desc.path}"
        elif dest.read_bytes() != src.read_bytes():
            return f"post-verify content mismatch: {desc.path}"
    return ""


def _snapshot_jit_build_dir(plan: _SnapshotPlan) -> str:
    """The AITER JIT build dir the snapshot's single JIT strategy pins, if it has one."""
    jit_strategy = plan.jit_strategies[0] if plan.jit_strategies else {}
    return str(jit_strategy.get("jit_build_dir") or "").strip()


def _fan_snapshot_out_to_pods(
    request: KernelPatchRequest,
    deployment: _PatchDeployment,
    plan: _SnapshotPlan,
) -> dict[str, Any]:
    """Push every written path to every RayJob pod, restoring the whole repo if one refuses."""
    pod_backup_dir = os.environ.get("HYPERLOOM_MN_KERNEL_BACKUP_DIR", _MN_POD_BACKUP_DIR_DEFAULT)
    snap = Path(request.snapshot_dir)
    jit_build_dir = _snapshot_jit_build_dir(plan)
    jit_dispatch_target = next((path for path in plan.write_paths if _target_uses_aiter_jit(path)), None)
    fanout = _PodFanout()
    try:
        for path in plan.write_paths:
            carries_jit = path == jit_dispatch_target and jit_build_dir and not request.skip_rebuild
            fanout.absorb(
                _dispatch_multinode_apply(
                    PodPatchApply(
                        target_file=path,
                        patch_path=snap / Path(path).relative_to(plan.repo_root),
                        kernel_id=request.kernel_id,
                        backup_dir_on_pod=pod_backup_dir,
                        jit_build_dir=jit_build_dir if carries_jit else "",
                    )
                )
            )
            # Persist after every file so a later dispatch failure can
            # revert all already-patched files on every pod.
            deployment.record(_MANIFEST_MULTINODE, fanout.block("partial", plan.target, pod_backup_dir))
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        revert = revert_kernel_patch(deployment.manifest_path)
        raise _PatchRefused(
            {
                "status": "failed",
                "error": f"multi-node apply fan-out failed; repo restored: {exc}",
                "manifest_path": str(deployment.manifest_path),
                "revert": revert,
            }
        ) from exc
    block = fanout.block("ok", plan.target, pod_backup_dir)
    deployment.record(_MANIFEST_MULTINODE, block)
    return block


def _snapshot_rebuild_plan(
    request: KernelPatchRequest,
    plan: _SnapshotPlan,
    multinode: dict[str, Any],
) -> _RebuildPlan:
    """How every compiled root the snapshot touched is rebuilt."""
    jit_build_dir = _snapshot_jit_build_dir(plan)
    jit_target = next((path for path in plan.write_paths if _target_uses_aiter_jit(path)), plan.target)
    cpp_itfs_target = next(
        (path for path in plan.write_paths if _target_is_in_aiter_cpp_itfs(path)),
        plan.target,
    )
    return _RebuildPlan(
        strategies=plan.rebuild_strategies,
        jit_strategy=plan.jit_strategies[0] if plan.jit_strategies else {},
        jit_target=jit_target,
        jit_build_dir=Path(jit_build_dir) if jit_build_dir else None,
        cpp_itfs_target=cpp_itfs_target,
        command_override=plan.command_override,
        timeout_sec=request.rebuild_timeout_sec,
        pod_records=multinode.get("records_by_host") if (multinode and jit_build_dir) else None,
    )
