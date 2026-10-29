#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply an optimized kernel file with source/artifact backup and fast revert."""

from __future__ import annotations

import hashlib
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from hyperloom.common.timeutil import now_iso
from common.unified_diff import PatchOp, PatchTarget, repo_relative_manifest
from hyperloom.inference_optimizer.multi_node.scripts.patch_path_safety import DEFAULT_KERNEL_BACKUP_ROOT

from .patch_jit_cache import (
    _aiter_cpp_itfs_build_dir,
    _invalidate_aiter_cpp_itfs_cache,
    _invalidate_aiter_jit_build,
    _restore_aiter_jit_build,
    _target_is_in_aiter_cpp_itfs,
    _target_uses_aiter_jit,
    _within_root,
)
from .patch_rebuild import (
    COMPILED_SOURCE_SUFFIXES,
    PYTHON_SOURCE_SUFFIXES,
    _REBUILD_MODE_RUNTIME_JIT,
    _detect_strategy,
    _discover_artifacts,
    _rebuild_ok_to_proceed,
    _remote_jit_invalidation_error,
    _run_strategy_rebuild,
    _runtime_jit_invalidation_error,
)

log = logging.getLogger(__name__)


TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".patch", ".diff"}


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
    if any(_UNSAFE_COMMAND_CHARS_RE.search(part) for part in argv):
        raise ValueError("rebuild_command must be argv-like and cannot contain shell control operators")
    if any("\x00" in part for part in argv):
        raise ValueError("rebuild_command contains invalid control characters")
    exe = Path(argv[0]).name.lower()
    if exe in _SHELL_COMMAND_NAMES and any(part in {"-c", "-lc"} for part in argv[1:]):
        raise ValueError("rebuild_command must not invoke a shell command string")
    return argv


_MANIFEST_MULTINODE = "multinode"
_MANIFEST_NAME = "manifest.json"


def _pod_backup_dir() -> str:
    return os.environ.get("HYPERLOOM_MN_KERNEL_BACKUP_DIR", DEFAULT_KERNEL_BACKUP_ROOT)


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


def _safe_name(value: str) -> str:
    """Sanitize a filename component to a safe, short identifier."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "kernel_agent"


def _path_hash(path: Path) -> str:
    """Return a short, stable hash for ``path`` to disambiguate backups."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


_MAX_BACKUP_DIR_ATTEMPTS: int = 200


def _claim_backup_dir(backup_root: Path, kernel_id: str, target: Path) -> Path:
    """Create a backup directory this apply alone owns.

    ``mkdir(exist_ok=False)`` makes the claim atomic against a concurrent caller,
    which a constant or absent ``kernel_id`` would otherwise let reuse a name.
    """
    base = Path(backup_root) / f"{_safe_name(kernel_id or target.stem)}_{_path_hash(target)}"
    for suffix in range(1, _MAX_BACKUP_DIR_ATTEMPTS + 1):
        candidate = base if suffix == 1 else base.with_name(f"{base.name}-{suffix}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"_claim_backup_dir: {base} still taken after {_MAX_BACKUP_DIR_ATTEMPTS} suffixes")


def _copy_to_backup(path: Path, backup_dir: Path, group: str) -> dict[str, str]:
    """Copy a file into the backup tree and return the manifest entry."""
    dst = backup_dir / group / f"{_path_hash(path)}_{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return {"path": str(path), "backup_path": str(dst)}


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
        ValueError: If the patch is incompatible or not a complete source file.
    """
    patch_suffix = patch.suffix.lower()
    target_suffix = target.suffix.lower()
    if patch_suffix in TEXT_ARTIFACT_SUFFIXES:
        raise ValueError(f"patch_path is not a complete source file: {patch}")
    if patch_suffix != target_suffix:
        raise ValueError(
            f"patch suffix {patch_suffix or '<none>'} does not match target suffix {target_suffix or '<none>'}"
        )
    text = patch.read_text(encoding="utf-8", errors="replace")
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


def _contained_dest(
    repo_root: Path,
    rel_path: str,
    *,
    deploy_roots: Iterable[Path] | None = None,
) -> Path:
    """Resolve ``rel_path`` under its coordinate and authorized deploy roots.

    ``deploy_roots=None`` keeps the coordinate-only check for legacy callers; an
    empty iterable rejects every destination.
    """
    if rel_path.startswith("/"):
        raise ValueError(f"absolute path not allowed in patch: {rel_path}")
    if ".." in Path(rel_path).parts:
        raise ValueError(f"'..' not allowed in patch path: {rel_path}")
    root = repo_root.resolve()
    dest = (root / rel_path).resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"patch path escapes repo root {root}: {rel_path}") from exc
    if deploy_roots is not None:
        allowed = [Path(path).resolve() for path in deploy_roots]
        if not any(_within_root(dest, allowed_root) for allowed_root in allowed):
            roots_text = ", ".join(str(path) for path in allowed) or "<none>"
            raise ValueError(
                f"patch path is outside authorized deploy roots: {rel_path}; authorized roots: {roots_text}"
            )
    return dest


def apply_snapshot(
    *,
    descriptors: list[PatchTarget],
    snapshot_dir: str | Path,
    repo_root: str | Path,
    backup_dir: str | Path,
    deploy_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Apply a content-addressed snapshot atomically (all-or-nothing).

    Each descriptor asserts a final filesystem state rather than replaying a diff,
    and the whole set pre-flights before any write: a failure restores every
    touched path from backup, so the repo is never left partially applied and
    there is no fuzzy fallback.
    """
    root = Path(repo_root)
    allowed_roots = None if deploy_roots is None else [Path(path) for path in deploy_roots]
    roots = _SnapshotRoots(
        snapshot_dir=Path(snapshot_dir),
        repo_root=root,
        deploy_roots=allowed_roots,
        symlink_roots=[root] if allowed_roots is None else allowed_roots,
    )
    try:
        staged = _stage_snapshot_writes(descriptors, roots)
    except _PatchRefused as refusal:
        return refusal.result

    backups: list[dict[str, Any]] = []
    touched: list[str] = []
    for write in staged:
        try:
            backups.append(_back_up_before_write(write, Path(backup_dir)))
            _land_staged_write(write, roots)
        except (OSError, ValueError) as exc:
            _restore_live_tree(backups)
            return {
                "status": "failed",
                "error": f"apply failed at {write.descriptor.path}: {exc}; repo restored",
                "path": write.descriptor.path,
            }
        touched.append(str(write.dest))

    return {"status": "ok", "source_backups": backups, "touched": touched}


class _SnapshotRoots(NamedTuple):
    snapshot_dir: Path
    repo_root: Path
    deploy_roots: list[Path] | None
    symlink_roots: list[Path]


class _StagedWrite(NamedTuple):
    descriptor: PatchTarget
    dest: Path
    source: Path | None


def _stage_snapshot_writes(descriptors: list[PatchTarget], roots: _SnapshotRoots) -> list[_StagedWrite]:
    staged: list[_StagedWrite] = []
    try:
        for desc in descriptors:
            dest = _contained_dest(roots.repo_root, desc.path, deploy_roots=roots.deploy_roots)
            source: Path | None = None
            if desc.op is PatchOp.WRITE:
                source = roots.snapshot_dir / desc.path
                _refuse_unwritable_snapshot_source(source, dest, desc.path, roots.symlink_roots)
            staged.append(_StagedWrite(desc, dest, source))
    except ValueError as exc:
        raise _PatchRefused({"status": "failed", "error": str(exc), "path": "<pre-flight>"}) from exc
    return staged


def _refuse_unwritable_snapshot_source(
    source: Path,
    dest: Path,
    rel_path: str,
    symlink_roots: list[Path],
) -> None:
    if not source.is_symlink():
        if not source.exists():
            raise _PatchRefused(
                {"status": "failed", "error": f"snapshot missing content for {rel_path}", "path": rel_path}
            )
        return
    link_target = os.readlink(source)
    resolved_link = (Path(link_target) if os.path.isabs(link_target) else dest.parent / link_target).resolve()
    if any(_within_root(resolved_link, allowed_root) for allowed_root in symlink_roots):
        return
    raise _PatchRefused(
        {
            "status": "failed",
            "error": (f"snapshot symlink target is outside authorized deploy roots: {rel_path} -> {link_target}"),
            "path": rel_path,
        }
    )


def _back_up_before_write(write: _StagedWrite, backup_dir: Path) -> dict[str, Any]:
    dest = write.dest
    existed = dest.exists() or dest.is_symlink()
    if not existed and write.descriptor.op is not PatchOp.DELETE:
        return {"path": str(dest), "backup_path": None, "disposition": "added"}
    backup_path = None
    if existed:
        backup_dest = backup_dir / "source" / f"{_path_hash(dest)}_{dest.name}"
        backup_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, backup_dest, follow_symlinks=False)
        backup_path = str(backup_dest)
    disposition = "deleted" if write.descriptor.op is PatchOp.DELETE else "modified"
    return {"path": str(dest), "backup_path": backup_path, "disposition": disposition}


def _land_staged_write(write: _StagedWrite, roots: _SnapshotRoots) -> None:
    dest = write.dest
    if write.descriptor.op is PatchOp.DELETE:
        dest.unlink(missing_ok=True)
        return
    # Re-validate containment right before the write (close TOCTOU window).
    _contained_dest(roots.repo_root, write.descriptor.path, deploy_roots=roots.deploy_roots)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    if write.source.is_symlink():
        dest.symlink_to(os.readlink(write.source))
        return
    shutil.copy2(write.source, dest, follow_symlinks=False)
    if write.descriptor.mode:
        try:
            dest.chmod(int(write.descriptor.mode, 8))
        except (ValueError, OSError):
            # Restoring mode bits is best-effort.
            pass


def _restore_live_tree(backups: list[dict[str, Any]]) -> None:
    for entry in reversed(backups):
        live = Path(entry["path"])
        if entry["disposition"] == "added":
            live.unlink(missing_ok=True)
            continue
        # modified / deleted -> restore old bytes from backup
        backup_path = entry.get("backup_path")
        if backup_path and Path(backup_path).exists():
            live.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, live, follow_symlinks=False)


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


def _revert_backup_trusted(backup_path: str, backup_root: Path) -> bool:
    """True when a manifest ``backup_path`` is a real backup under ``backup_root``.

    The manifest JSON is untrusted at revert time, so every ``copy2`` source must
    stay inside the apply-time backup tree or a tampered entry could read an
    arbitrary host file.
    """
    if not backup_path:
        return False
    return _within_root(Path(backup_path), backup_root)


class _BackupDeletion(NamedTuple):
    deleted: list[str]
    issues: list[dict[str, Any]]


def _delete_recorded_backups(candidates: set[str], backup_root: Path) -> _BackupDeletion:
    deleted: list[str] = []
    issues: list[dict[str, Any]] = []
    for raw in sorted(path for path in candidates if path):
        path = Path(raw)
        if path.resolve() == backup_root:
            issues.append(
                {
                    "kind": "untrusted_backup",
                    "path": str(path),
                    "error": "backup path equals manifest directory",
                }
            )
            continue
        if not _within_root(path, backup_root):
            issues.append({"kind": "untrusted_backup", "path": str(path)})
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            deleted.append(str(path))
        except OSError as exc:
            issues.append(
                {
                    "kind": "delete_failed",
                    "path": str(path),
                    "error": str(exc),
                }
            )
    return _BackupDeletion(deleted, issues)


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
        _restore_source_backups(manifest, backup_root),
        _restore_jit_build_state(manifest, backup_root),
        _restore_cpp_itfs_cache_state(manifest),
        _revert_on_patched_pods(manifest),
    ]
    return _write_revert_outcome(manifest_file, manifest, steps)


class _RestoreTally(NamedTuple):
    restored: list[str]
    untrusted: list[dict[str, str]]
    issues: list[dict[str, Any]]
    reported: dict[str, Any]


def _restore_backed_up_artifacts(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
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


def _restore_source_backups(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
    entries = list(manifest.get("source_backups") or [])
    # Whole-file apply records one primary backup where snapshot apply records the set.
    kind = "source_backups" if entries else "source_backup"
    if not entries and manifest.get("source_backup"):
        entries = [manifest["source_backup"]]
    restored: list[str] = []
    untrusted: list[dict[str, str]] = []
    cache_cleared = False
    for entry in reversed(entries):
        dst = Path(entry["path"])
        backup_path = entry.get("backup_path")
        if entry.get("disposition") == "added":
            dst.unlink(missing_ok=True)
        elif backup_path and not _revert_backup_trusted(backup_path, backup_root):
            untrusted.append({"kind": kind, "path": str(entry.get("path", "")), "backup_path": str(backup_path)})
            log.warning("revert: skipping source backup with untrusted backup_path %r", backup_path)
            continue
        elif backup_path and Path(backup_path).exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, dst, follow_symlinks=False)
        else:
            continue
        restored.append(str(dst))
        if dst.suffix.lower() in PYTHON_SOURCE_SUFFIXES and not cache_cleared:
            manifest["revert_cache_clear"] = _clear_python_kernel_caches(dst)
            cache_cleared = True
    return _RestoreTally(restored, untrusted, [], {})


def _restore_jit_build_state(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
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
    cpp_itfs_cache_backup = manifest.get("cpp_itfs_cache_backup") or {}
    if cpp_itfs_cache_backup.get("status") != "ok":
        return _RestoreTally([], [], [], {})
    cpp_itfs_cache_restore = _restore_aiter_cpp_itfs_cache(cpp_itfs_cache_backup)
    manifest["cpp_itfs_cache_restore"] = cpp_itfs_cache_restore
    if cpp_itfs_cache_restore.get("status") != "ok":
        return _RestoreTally([], [], [], {})
    return _RestoreTally(list(cpp_itfs_cache_restore.get("restored", [])), [], [], {})


def _revert_on_patched_pods(manifest: dict[str, Any]) -> _RestoreTally:
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
    restored = [path for step in steps for path in step.restored]
    untrusted = [entry for step in steps for entry in step.untrusted]
    issues = [issue for step in steps for issue in step.issues]
    reported = {key: record for step in steps for key, record in step.reported.items()}
    if untrusted:
        reported["skipped_untrusted_backups"] = untrusted
    if issues:
        reported["revert_issues"] = issues
    reverted_at = now_iso(timespec="auto")
    partial = bool(untrusted or issues)
    manifest["status"] = "reverted_partial" if partial else "reverted"
    manifest["reverted_at"] = reverted_at
    manifest["restored_paths"] = restored
    manifest.update(reported)
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "partial" if partial else "ok",
        "manifest_path": str(manifest_file),
        "restored_paths": restored,
        "reverted_at": reverted_at,
        **reported,
    }


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
        strat = _detect_strategy(p)
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
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("error") or "kernel patch refused"))
        self.result = result


def _patch_rebuild_argv(request: KernelPatchRequest) -> list[str]:
    try:
        return _coerce_rebuild_command(request.rebuild_command) if request.rebuild_command else []
    except ValueError as exc:
        raise _PatchRefused({"status": "failed", "error_class": "invalid_rebuild_command", "error": str(exc)}) from exc


@dataclass(frozen=True)
class _PatchDeployment:
    target: Path
    backup_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]

    def persist(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def record(self, key: str, value: Any) -> None:
        self.manifest[key] = value
        self.persist()


@dataclass
class _PodFanout:
    records_by_host: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    backup_map: dict[str, str] = field(default_factory=dict)

    def absorb(self, dispatch_result: dict[str, Any]) -> None:
        for entry in dispatch_result.get("per_node", []) or []:
            host = (entry.get("host") or "").strip()
            if not host:
                continue
            self.records_by_host.setdefault(host, []).append(dict(entry))
            backup_path = (entry.get("backup_path") or "").strip()
            if backup_path:
                self.backup_map[host] = backup_path

    def block(self, status: str, target: Path, pod_backup_dir: str) -> dict[str, Any]:
        return {
            "status": status,
            "target_path": str(target),
            "backup_dir_on_pod": pod_backup_dir,
            "host_backup_map": self.backup_map,
            "records_by_host": self.records_by_host,
        }


class _RebuildPlan(NamedTuple):
    strategies: list[dict[str, Any]]
    jit_strategy: dict[str, Any]
    jit_target: Path
    jit_build_dir: Path | None
    cpp_itfs_target: Path
    command_override: list[str]
    timeout_sec: int
    pod_records: dict[str, list[dict[str, Any]]] | None


class _RebuildRecords(NamedTuple):
    rebuild: dict[str, Any]
    jit_build_backup: dict[str, Any]
    cpp_itfs_cache_backup: dict[str, Any]


def _rebuild_not_run() -> _RebuildRecords:
    return _RebuildRecords(
        rebuild={"status": "skipped", "reason": "source-only patch or skip_rebuild=true"},
        jit_build_backup={"status": "skipped", "reason": "rebuild not run"},
        cpp_itfs_cache_backup={"status": "skipped", "reason": "rebuild not run", "is_cpp_itfs": False},
    )


def _refuse_untrusted_jit_strategies(jit_strategies: Iterable[dict[str, Any]]) -> None:
    for strategy in jit_strategies:
        invalidation_error = _runtime_jit_invalidation_error(strategy)
        if invalidation_error:
            raise _PatchRefused(
                {"status": "failed", "error_class": "aiter_jit_invalidation_failed", "error": invalidation_error}
            )


def _invalidate_and_rebuild(deployment: _PatchDeployment, plan: _RebuildPlan) -> _RebuildRecords:
    jit_build_backup = _invalidate_jit_build(deployment, plan)
    cpp_itfs_cache_backup = _invalidate_cpp_itfs_cache(deployment, plan)
    return _RebuildRecords(
        rebuild=_rebuild_patched_roots(deployment, plan),
        jit_build_backup=jit_build_backup,
        cpp_itfs_cache_backup=cpp_itfs_cache_backup,
    )


def _invalidate_jit_build(deployment: _PatchDeployment, plan: _RebuildPlan) -> dict[str, Any]:
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
        jit_build_dir = str(strategy.get("jit_build_dir") or "").strip()
        records = _invalidate_and_rebuild(
            deployment,
            _RebuildPlan(
                strategies=[strategy],
                jit_strategy=strategy,
                jit_target=target,
                jit_build_dir=Path(jit_build_dir) if jit_build_dir else None,
                cpp_itfs_target=target,
                command_override=command_override,
                timeout_sec=request.rebuild_timeout_sec,
                pod_records=multinode.get("records_by_host") if (multinode and jit_build_dir) else None,
            ),
        )
    return _finish_apply(deployment, records, cache_clear, compiled, multinode)


def _prepare_whole_file_apply(
    request: KernelPatchRequest,
    patch: Path,
    target: Path,
    strategy: dict[str, Any],
) -> _PatchDeployment:
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
    deployment = _PatchDeployment(target, backup_dir, backup_dir / _MANIFEST_NAME, manifest)
    deployment.persist()
    return deployment


def _fan_whole_file_out_to_pods(
    request: KernelPatchRequest,
    deployment: _PatchDeployment,
    patch: Path,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    pod_backup_dir = _pod_backup_dir()
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


class _SnapshotPlan(NamedTuple):
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
        jit_build_dir = _snapshot_jit_build_dir(plan)
        records = _invalidate_and_rebuild(
            deployment,
            _RebuildPlan(
                strategies=plan.rebuild_strategies,
                jit_strategy=plan.jit_strategies[0] if plan.jit_strategies else {},
                jit_target=next((path for path in plan.write_paths if _target_uses_aiter_jit(path)), plan.target),
                jit_build_dir=Path(jit_build_dir) if jit_build_dir else None,
                cpp_itfs_target=next(
                    (path for path in plan.write_paths if _target_is_in_aiter_cpp_itfs(path)), plan.target
                ),
                command_override=plan.command_override,
                timeout_sec=request.rebuild_timeout_sec,
                pod_records=multinode.get("records_by_host") if (multinode and jit_build_dir) else None,
            ),
        )
    result = _finish_apply(deployment, records, cache_clear, compiled, multinode)
    result["touched"] = touched
    return result


def _snapshot_plan(request: KernelPatchRequest) -> _SnapshotPlan:
    patch = Path(request.patch_path).resolve()
    target = Path(request.target_file).resolve()
    if not patch.is_file():
        raise _PatchRefused({"status": "failed", "error": f"patch_path does not exist: {patch}"})
    try:
        descriptors = repo_relative_manifest(patch.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        raise _PatchRefused({"status": "failed", "error": f"unparseable patch: {exc}"}) from exc
    primary_strategy = _detect_strategy(target)
    # Coerce/validate rebuild_command BEFORE any file/pod mutation (all-or-nothing).
    command_override = _patch_rebuild_argv(request)
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
    repo_root = Path(resolved_root)
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


def _snapshot_write_paths(
    descriptors: list[PatchTarget],
    repo_root: Path,
    deploy_roots: list[Path],
) -> list[Path]:
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
    deployment = _PatchDeployment(plan.target, backup_dir, backup_dir / _MANIFEST_NAME, manifest)
    deployment.persist()
    return deployment


def _land_snapshot(request: KernelPatchRequest, deployment: _PatchDeployment, plan: _SnapshotPlan) -> list[str]:
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
    jit_strategy = plan.jit_strategies[0] if plan.jit_strategies else {}
    return str(jit_strategy.get("jit_build_dir") or "").strip()


def _fan_snapshot_out_to_pods(
    request: KernelPatchRequest,
    deployment: _PatchDeployment,
    plan: _SnapshotPlan,
) -> dict[str, Any]:
    pod_backup_dir = _pod_backup_dir()
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
