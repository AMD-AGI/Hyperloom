#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Apply an optimized kernel file with source/artifact backup and fast revert."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Sibling import works whether run as a script or loaded via importlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _io_utils import source_text_looks_complete  # noqa: E402

sys.path.pop(0)

log = logging.getLogger(__name__)


COMPILED_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"}
PYTHON_SOURCE_SUFFIXES = {".py"}
COMPILED_ARTIFACT_SUFFIXES = {".so", ".co", ".hsaco"}
TEXT_ARTIFACT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".patch", ".diff"}
# Fallback when ``inference_optimizer`` is not on ``sys.path`` (standalone CLI).
_FALLBACK_KNOWN_TARGET_ROOTS: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
    "/opt/venv/lib/python3.10/site-packages/aiter/",
    "/opt/venv/lib/python3.10/site-packages/sglang/",
    "/opt/venv/lib/python3.10/site-packages/vllm/",
    "/usr/local/lib/python3.12/dist-packages/aiter/",
    "/usr/local/lib/python3.12/dist-packages/sglang/",
    "/usr/local/lib/python3.12/dist-packages/vllm/",
    "/usr/local/lib/python3.10/dist-packages/aiter/",
    "/usr/local/lib/python3.10/dist-packages/sglang/",
    "/usr/local/lib/python3.10/dist-packages/vllm/",
)

_CACHED_KNOWN_TARGET_ROOTS: tuple[str, ...] | None = None


def known_target_roots() -> tuple[str, ...]:
    """Resolved framework roots (importlib/glob when orchestrator is importable).

    Resolves once and caches the result. Falls back to
    :data:`_FALLBACK_KNOWN_TARGET_ROOTS` when the orchestrator package is
    not importable (standalone CLI use).

    Returns:
        tuple[str, ...]: Absolute path-prefix strings for the recognised
            reusable framework source roots (aiter / sglang / vllm).
    """
    global _CACHED_KNOWN_TARGET_ROOTS
    if _CACHED_KNOWN_TARGET_ROOTS is not None:
        return _CACHED_KNOWN_TARGET_ROOTS
    try:
        from hyperloom.orchestrator.framework.paths import (
            resolve_patch_target_roots,
        )

        _CACHED_KNOWN_TARGET_ROOTS = resolve_patch_target_roots()
    except ImportError:
        _CACHED_KNOWN_TARGET_ROOTS = _FALLBACK_KNOWN_TARGET_ROOTS
    return _CACHED_KNOWN_TARGET_ROOTS


# Backward-compat alias for tests / external imports.
KNOWN_TARGET_ROOTS = _FALLBACK_KNOWN_TARGET_ROOTS


# Pod-local multi-node backup dir; survives sglang restarts.
# Overridable via $HYPERLOOM_MN_KERNEL_BACKUP_DIR for tests.
_MN_POD_BACKUP_DIR_DEFAULT = "/var/kernel_patch_backups"

# Multi-node signal file (nodes >= 2) written by multi_node create-rayjob.
# $MULTI_NODE_STATE_FILE wins; env override keeps test runs isolated.
_MN_STATE_FILE_DEFAULT = "/tmp/multi_node_state.json"


def _mn_state_path() -> Path:
    """Resolve where ``hyperloom.inference_optimizer.multi_node`` dropped its state.

    Honours the ``$MULTI_NODE_STATE_FILE`` override and falls back to
    ``/tmp/multi_node_state.json``.

    Returns:
        Path: The resolved multi-node state-file path.
    """
    return Path(os.environ.get("MULTI_NODE_STATE_FILE", _MN_STATE_FILE_DEFAULT))


# Legacy module attribute kept for direct importers; runtime uses _mn_state_path.
_MN_STATE_FILE = Path(_MN_STATE_FILE_DEFAULT)


def _is_multi_node() -> bool:
    """Report whether a multi-node RayJob is active.

    Returns:
        ``True`` when the state file reports ``nodes >= 2``; ``False`` when the
        state file is missing or unreadable.
    """
    state_path = _mn_state_path()
    try:
        if not state_path.is_file():
            return False
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return int(data.get("nodes") or 0) >= 2
    except (OSError, ValueError):
        return False


def _dispatch_multinode_apply(
    *,
    target_file: Path,
    patch_path: Path,
    kernel_id: str,
    backup_dir_on_pod: str,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """Fan a patch out to every pod (head + workers).

    Args:
        target_file: The file to patch on each pod.
        patch_path: Path to the patch file to apply.
        kernel_id: Identifier of the kernel being patched.
        backup_dir_on_pod: Directory on each pod to store backups.
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
        "hyperloom.inference_optimizer.multi_node",
        "apply-patch",
        "--patch-file",
        str(patch_path),
        "--target-path",
        str(target_file),
        "--backup-dir",
        backup_dir_on_pod,
        "--kernel-id",
        kernel_id or "",
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
    target_path: str,
    backup_map: dict[str, str],
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
        "hyperloom.inference_optimizer.multi_node",
        "revert-patch",
        "--target-path",
        str(target_path),
        "--backup-map-json",
        json.dumps(backup_map, sort_keys=True),
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


def _now() -> str:
    """Return the current UTC timestamp as an ISO8601 string.

    Returns:
        str: The current UTC time formatted as an ISO8601 string.
    """
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Sanitize a filename component to a safe, short identifier.

    Non-alphanumeric characters (other than ``._-``) become underscores and the
    result is truncated to 80 characters.

    Args:
        value (str): The raw string to sanitize.

    Returns:
        str: A filesystem-safe identifier, or ``"kernel_agent"`` when empty.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned[:80] or "kernel_agent"


def _path_hash(path: Path) -> str:
    """Return a short, stable hash for ``path`` to disambiguate backups.

    Args:
        path (Path): The path to hash.

    Returns:
        str: The first 16 hex characters of the SHA-256 of the path string.
    """
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _copy_to_backup(path: Path, backup_dir: Path, group: str) -> dict[str, str]:
    """Copy a file into the backup tree and return the manifest entry.

    Args:
        path (Path): The file to back up.
        backup_dir (Path): Root of the backup tree.
        group (str): Sub-directory grouping (e.g. ``"source"`` / ``"artifacts"``).

    Returns:
        dict[str, str]: A manifest entry with ``path`` (original) and
            ``backup_path`` (backup copy) keys.
    """
    dst = backup_dir / group / f"{_path_hash(path)}_{path.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return {"path": str(path), "backup_path": str(dst)}


# Shared source-completeness heuristic (see _io_utils.source_text_looks_complete).
_source_text_looks_complete = source_text_looks_complete
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


def _unquote_git_path(raw: str) -> str:
    """Decode a git diff header path, handling C-style quoting.

    Git emits paths with special characters as C-quoted strings (e.g.
    ``"b/\\303\\251.py"``). A naive ``..``/absolute check on the still-quoted
    string can be bypassed, so decode it first.

    Args:
        raw (str): The raw token following a header keyword (already
            whitespace-trimmed).

    Returns:
        str: The decoded path, or the input unchanged when it is not quoted.
    """
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        try:
            return raw[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            return raw[1:-1]
    return raw


def _strip_ab_prefix(path: str) -> str:
    """Drop a leading ``a/`` or ``b/`` git diff prefix.

    Args:
        path (str): A diff header path.

    Returns:
        str: The path without its ``a/``/``b/`` prefix.
    """
    return re.sub(r"^[ab]/", "", path)


def parse_patch_manifest(patch_text: str) -> list[dict[str, Any]]:
    """Parse a unified diff into a list of per-path deploy descriptors.

    The patch is read as a *manifest* of intended filesystem changes — the set
    of touched paths and each path's disposition — never as an instruction to be
    replayed. Recognises the full git header vocabulary so nothing is silently
    skipped: plain modifications, ``new file`` / ``--- /dev/null`` additions,
    ``deleted file`` / ``+++ /dev/null`` deletions, ``rename from/to`` (mapped to
    delete-source + add-dest), ``copy from/to``, ``old mode/new mode`` chmod-only
    entries, and ``GIT binary patch`` blocks (flagged, content sourced from the
    snapshot).

    Args:
        patch_text (str): The full unified diff text.

    Returns:
        list[dict[str, Any]]: One descriptor per affected path. Each has
            ``op`` (``"write"`` or ``"delete"``), ``path`` (repo-relative
            target), ``mode`` (octal string or ``""``), ``binary`` (bool), and
            for renames/copies a ``source`` (origin repo-relative path). A
            rename yields two descriptors: a ``delete`` of the source and a
            ``write`` of the dest.

    Raises:
        ValueError: When the patch is empty/unparseable, or a section's
            disposition cannot be determined.
    """
    if not patch_text or not patch_text.strip():
        raise ValueError("empty patch: nothing to apply")

    if re.search(r"(?m)^diff --git ", patch_text):
        blocks = [b for b in re.split(r"(?m)^(?=diff --git )", patch_text) if b.strip()]
    else:
        blocks = [b for b in re.split(r"(?m)^(?=--- )", patch_text) if b.strip()]
    if not blocks:
        raise ValueError("patch contains no file sections")

    descriptors: list[dict[str, Any]] = []
    for block in blocks:
        binary = bool(re.search(r"(?m)^GIT binary patch\b", block))
        new_mode_m = re.search(r"(?m)^new file mode (\d+)$", block)
        chmod_m = re.search(r"(?m)^new mode (\d+)$", block)
        mode = ""
        if new_mode_m:
            mode = new_mode_m.group(1)[-4:]
        elif chmod_m:
            mode = chmod_m.group(1)[-4:]

        rename_to = re.search(r"(?m)^rename to (.+)$", block)
        rename_from = re.search(r"(?m)^rename from (.+)$", block)
        copy_to = re.search(r"(?m)^copy to (.+)$", block)
        if rename_to and rename_from:
            src = _unquote_git_path(rename_from.group(1).strip())
            dst = _unquote_git_path(rename_to.group(1).strip())
            descriptors.append({"op": "delete", "path": src, "mode": "", "binary": False})
            descriptors.append({"op": "write", "path": dst, "mode": mode, "binary": binary})
            continue
        if copy_to:
            dst = _unquote_git_path(copy_to.group(1).strip())
            descriptors.append({"op": "write", "path": dst, "mode": mode, "binary": binary})
            continue

        minus = re.search(r"(?m)^--- (.+)$", block)
        plus = re.search(r"(?m)^\+\+\+ (.+)$", block)
        deleted = bool(re.search(r"(?m)^deleted file mode", block))
        plus_path = _unquote_git_path(plus.group(1).split("\t", 1)[0].strip()) if plus else ""
        minus_path = _unquote_git_path(minus.group(1).split("\t", 1)[0].strip()) if minus else ""

        if deleted or plus_path == "/dev/null":
            target = _strip_ab_prefix(minus_path)
            if not target:
                raise ValueError(f"deletion section missing source path:\n{block[:200]}")
            descriptors.append({"op": "delete", "path": target, "mode": "", "binary": binary})
            continue

        # Addition (--- /dev/null) or modification: dest comes from the +++ line.
        target = _strip_ab_prefix(plus_path)
        if not target:
            # Header-only entries (pure chmod) carry the path on the diff line.
            gitline = re.search(r"(?m)^diff --git a/(.+?) b/(.+)$", block)
            if gitline:
                target = _unquote_git_path(gitline.group(2).strip())
        if not target:
            raise ValueError(f"cannot determine target path for section:\n{block[:200]}")
        descriptors.append({"op": "write", "path": target, "mode": mode, "binary": binary})

    return descriptors


def _contained_dest(repo_root: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` under ``repo_root``, rejecting any escape.

    Rejects absolute paths and any ``..`` traversal, and confirms the resolved
    destination stays inside ``repo_root`` (closes the path-traversal hole the
    review flagged, applied at the deploy boundary).

    Args:
        repo_root (Path): The framework repo root that destinations must stay in.
        rel_path (str): Repo-relative target path from the patch manifest.

    Returns:
        Path: The validated absolute destination path.

    Raises:
        ValueError: If the path is absolute, contains ``..``, or escapes
            ``repo_root``.
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
    return dest


def _within_root(path: Path, root: Path) -> bool:
    """Return True when ``path`` (symlinks resolved) stays inside ``root``.

    Args:
        path (Path): candidate path (need not exist).
        root (Path): the containment root.

    Returns:
        bool: True when the resolved ``path`` equals or is nested under the
            resolved ``root``; False on any resolution error or escape.
    """
    try:
        p = path.resolve()
        r = root.resolve()
    except (OSError, RuntimeError):
        return False
    try:
        return p == r or p.is_relative_to(r)
    except AttributeError:  # pragma: no cover — Python <3.9
        try:
            p.relative_to(r)
            return True
        except ValueError:
            return False


def apply_snapshot(
    *,
    descriptors: list[dict[str, Any]],
    snapshot_dir: str | Path,
    repo_root: str | Path,
    backup_dir: str | Path,
) -> dict[str, Any]:
    """Apply a content-addressed snapshot atomically (all-or-nothing).

    Each descriptor asserts a final filesystem state rather than replaying a
    diff: ``write`` copies the byte-exact file from ``snapshot_dir`` onto the
    live repo, ``delete`` removes the live file. The whole set is staged with a
    full pre-flight (containment + source existence) before any write; if any
    single operation fails, every already-touched path is restored from backup
    and the call returns ``status="failed"`` — the repo is never left partially
    applied. There is no fuzzy fallback.

    Args:
        descriptors (list[dict[str, Any]]): Output of
            :func:`parse_patch_manifest`.
        snapshot_dir (str | Path): Directory holding byte-exact final contents
            for every ``write`` path (mirrored at the same relative path).
        repo_root (str | Path): Framework repo root the targets live under.
        backup_dir (str | Path): Where per-path backups + the revert manifest
            are written.

    Returns:
        dict[str, Any]: ``status="ok"`` with ``source_backups`` (the revert
            manifest entries) and ``touched`` paths, or ``status="failed"`` with
            an ``error`` and the offending path.
    """
    root = Path(repo_root)
    snap = Path(snapshot_dir)
    backups: list[dict[str, Any]] = []

    # Pre-flight: validate everything before touching the live tree.
    staged: list[tuple[dict[str, Any], Path, Path | None]] = []
    try:
        for desc in descriptors:
            dest = _contained_dest(root, desc["path"])
            src: Path | None = None
            if desc["op"] == "write":
                src = snap / desc["path"]
                if not src.is_symlink() and not src.exists():
                    return {
                        "status": "failed",
                        "error": f"snapshot missing content for {desc['path']}",
                        "path": desc["path"],
                    }
            staged.append((desc, dest, src))
    except ValueError as exc:
        return {"status": "failed", "error": str(exc), "path": "<pre-flight>"}

    def _restore_all() -> None:
        """Roll the live tree back to v0 from recorded backups."""
        for entry in reversed(backups):
            disp = entry["disposition"]
            live = Path(entry["path"])
            if disp == "added":
                live.unlink(missing_ok=True)
            else:  # modified / deleted -> restore old bytes from backup
                bp = entry.get("backup_path")
                if bp and Path(bp).exists():
                    live.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bp, live, follow_symlinks=False)

    touched: list[str] = []
    for desc, dest, src in staged:
        try:
            existed = dest.exists() or dest.is_symlink()
            # Back up (disposition drives revert): record before mutating.
            if desc["op"] == "delete" or existed:
                disposition = "deleted" if desc["op"] == "delete" else "modified"
                bp = None
                if existed:
                    bdst = Path(backup_dir) / "source" / f"{_path_hash(dest)}_{dest.name}"
                    bdst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dest, bdst, follow_symlinks=False)
                    bp = str(bdst)
                backups.append({"path": str(dest), "backup_path": bp, "disposition": disposition})
            else:
                backups.append({"path": str(dest), "backup_path": None, "disposition": "added"})

            if desc["op"] == "delete":
                dest.unlink(missing_ok=True)
            else:
                # Re-validate containment on the final dest right before the
                # write (close the TOCTOU window) and never follow a symlink.
                _contained_dest(root, desc["path"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink():
                    dest.unlink()
                if src.is_symlink():
                    link_target = os.readlink(src)
                    dest.symlink_to(link_target)
                else:
                    shutil.copy2(src, dest, follow_symlinks=False)
                    if desc.get("mode"):
                        try:
                            dest.chmod(int(desc["mode"], 8))
                        except (ValueError, OSError):
                            # Restoring mode bits is best-effort: a bad/garbled
                            # mode string (ValueError) or a target filesystem
                            # that rejects chmod (OSError) must not fail the
                            # apply. The copied content is what matters, so keep
                            # the default mode and proceed.
                            pass
            touched.append(str(dest))
        except (OSError, ValueError) as exc:
            _restore_all()
            return {
                "status": "failed",
                "error": f"apply failed at {desc['path']}: {exc}; repo restored",
                "path": desc["path"],
            }

    return {"status": "ok", "source_backups": backups, "touched": touched}


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
    for pattern in ("/tmp/torchinductor_*", "/tmp/triton_*"):
        for path in Path("/tmp").glob(Path(pattern).name):
            if path.exists():
                remove_path(path)
    return {"status": "ok", "removed": removed}


# aiter JIT cache invalidation around rebuilds (setup.py develop won't invalidate jit/build/ .so).
_AITER_CSRC_MARKER = "/aiter/csrc/"


def _target_is_in_aiter_csrc(target_file: Path) -> bool:
    """Report whether a file resides under an ``aiter/csrc/`` tree.

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` if the path is under an ``aiter/csrc/`` directory.
    """
    return _AITER_CSRC_MARKER in str(target_file).replace(os.sep, "/")


def _aiter_jit_build_dir() -> Path | None:
    """Locate the importable aiter package's ``jit/build`` directory.

    Returns:
        The ``<aiter>/jit/build`` path, or ``None`` when aiter is not
        importable.
    """
    try:
        spec = importlib.util.find_spec("aiter")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    aiter_pkg = Path(list(spec.submodule_search_locations)[0])
    return aiter_pkg / "jit" / "build"


def _invalidate_aiter_jit_build(
    target_file: Path,
    backup_dir: Path,
    *,
    jit_build_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Move aiter ``jit/build/`` aside so a post-rebuild import re-JITs.

    Args:
        target_file: The file being patched (gates the operation).
        backup_dir: Directory to move the ``jit/build`` tree into.
        jit_build_dir_override: Test-only override for the jit/build location.

    Returns:
        A status dict with ``status`` of ``ok``, ``skipped``, or ``failed``
        and supporting fields.
    """
    if not _target_is_in_aiter_csrc(target_file):
        return {"status": "skipped", "reason": "target not under aiter/csrc/"}
    jit_build = jit_build_dir_override or _aiter_jit_build_dir()
    if jit_build is None:
        return {"status": "skipped", "reason": "aiter package not importable"}
    if not jit_build.exists():
        return {"status": "skipped", "reason": "aiter jit/build/ does not exist"}
    try:
        is_empty = not any(jit_build.iterdir())
    except OSError as exc:
        return {
            "status": "failed",
            "error": f"failed to scan {jit_build}: {exc}",
            "src": str(jit_build),
        }
    if is_empty:
        return {"status": "skipped", "reason": "aiter jit/build/ is empty"}
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "jit_build"
    if backup_path.exists():
        return {
            "status": "failed",
            "error": f"jit/build backup path already exists: {backup_path}",
            "src": str(jit_build),
        }
    try:
        shutil.move(str(jit_build), str(backup_path))
    except (OSError, shutil.Error) as exc:
        return {
            "status": "failed",
            "error": f"shutil.move failed: {exc}",
            "src": str(jit_build),
            "backup_path": str(backup_path),
        }
    return {
        "status": "ok",
        "src": str(jit_build),
        "backup_path": str(backup_path),
        "moved_at": _now(),
    }


def _restore_aiter_jit_build(jit_build_backup: dict[str, Any]) -> dict[str, Any]:
    """Restore an aiter ``jit/build`` backup, reversing the invalidation.

    Any regenerated ``jit/build`` directory is removed first.

    Args:
        jit_build_backup: The backup record returned by
            :func:`_invalidate_aiter_jit_build`.

    Returns:
        A status dict with ``status`` of ``ok``, ``skipped``, or ``failed``.
    """
    if not isinstance(jit_build_backup, dict) or jit_build_backup.get("status") != "ok":
        return {"status": "skipped", "reason": "no backup recorded"}
    src = Path(jit_build_backup.get("src", ""))
    backup_path = Path(jit_build_backup.get("backup_path", ""))
    if not src or not backup_path:
        return {"status": "skipped", "reason": "incomplete backup record"}
    # The manifest is untrusted at revert time; ``src`` is an ``rmtree`` target.
    # Only the importable aiter jit/build dir is a legitimate destination.
    expected = _aiter_jit_build_dir()
    if expected is None or not _within_root(src, expected):
        log.warning(
            "revert: skipping jit/build restore; recorded src %s does not match the "
            "importable aiter jit/build dir %s (aiter reinstalled/relocated, cross-process "
            "resume, or a forged manifest). jit/build left invalidated; next import re-JITs.",
            src,
            expected,
        )
        return {
            "status": "skipped",
            "reason": f"jit/build src {src} is not the importable aiter jit/build dir",
        }
    if not backup_path.exists():
        return {
            "status": "skipped",
            "reason": f"backup path missing: {backup_path}",
        }
    if src.exists():
        try:
            shutil.rmtree(src)
        except OSError as exc:
            return {
                "status": "failed",
                "error": f"failed to clear regenerated jit/build/: {exc}",
                "src": str(src),
            }
    try:
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(backup_path), str(src))
    except (OSError, shutil.Error) as exc:
        return {
            "status": "failed",
            "error": f"shutil.move failed during restore: {exc}",
            "src": str(src),
            "backup_path": str(backup_path),
        }
    return {"status": "ok", "restored_to": str(src)}


# aiter cpp_itfs kernels are runtime-compiled into parameter-keyed caches.
# Move matching cache dirs aside so patched sources rebuild, then restore on revert.
_AITER_CPP_ITFS_MARKER = "/aiter/csrc/cpp_itfs/"
_MD_NAME_RE = re.compile(r"""(?m)^\s*MD_NAME\s*=\s*["']([^"']+)["']""")


def _target_is_in_aiter_cpp_itfs(target_file: Path) -> bool:
    """Report whether a file lives under an ``aiter/csrc/cpp_itfs/`` tree.

    Strict subset of :func:`_target_is_in_aiter_csrc`: these are the
    runtime-compiled kernels whose served ``.so`` lives in
    ``$HOME/.aiter/build`` rather than in ``<aiter>/jit/build`` or the
    statically-linked wheel. Matches both the editable checkout
    (``/sgl-workspace/aiter/csrc/cpp_itfs/...``) and the dist-packages
    layout (``.../aiter/csrc/cpp_itfs/...``).

    Args:
        target_file: The file path to test.

    Returns:
        ``True`` if the path is under an ``aiter/csrc/cpp_itfs/`` directory.
    """
    return _AITER_CPP_ITFS_MARKER in str(target_file).replace(os.sep, "/")


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
        A status dict recording what was moved and the invalidation metadata.

    Returns one of ``ok`` / ``skipped`` / ``failed`` mirroring
    :func:`_invalidate_aiter_jit_build`. ``build_dir_override`` is a
    test-only hook.
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
        "invalidated_at": _now(),
        "invalidated_unix": time.time(),
    }
    if not build_dir.exists():
        # Nothing cached yet -> the re-baseline server will build fresh into
        # this dir on first kernel call. No stale binary to mask.
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
    to flag/abort instead of scoring a stale binary (GH #458 point 2).

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
    build_dir = Path(cache_backup.get("build_dir", ""))
    since = float(cache_backup.get("invalidated_unix") or 0.0)
    module_names = list(cache_backup.get("module_names") or [])
    if not str(build_dir) or not build_dir.exists():
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


def _detect_strategy(target_file: Path, *, allow_unknown_target: bool) -> dict[str, Any]:
    """Determine the rebuild strategy for a patch target.

    Matches the target against the known framework roots (aiter / sglang /
    vllm) to pick the rebuild command and artifact roots, and decides whether
    the target is a compiled source. Python targets never rebuild.

    Args:
        target_file (Path): The file being patched.
        allow_unknown_target (bool): When ``True``, targets outside the known
            roots are accepted (rooted at the target's parent) instead of
            raising.

    Returns:
        dict[str, Any]: A strategy dict with ``compiled`` (bool), ``root``
            (str), ``rebuild_command`` (list[str]) and ``artifact_roots``
            (list[Path]).

    Raises:
        ValueError: When the target is outside the known roots and
            ``allow_unknown_target`` is ``False``.
    """
    target = str(target_file)
    lower = target.lower()
    roots = known_target_roots()
    if not allow_unknown_target and not any(root in lower for root in roots):
        raise ValueError(f"target_file is outside known reusable source roots: {target_file}")

    suffix = target_file.suffix.lower()
    compiled = suffix in COMPILED_SOURCE_SUFFIXES
    root = None
    rebuild_command: list[str] = []
    artifact_roots: list[Path] = []

    if "/sgl-workspace/aiter/" in lower:
        root = Path("/sgl-workspace/aiter")
        rebuild_command = ["/opt/venv/bin/python", "setup.py", "develop"]
        artifact_roots = [root]
    elif "/sgl-workspace/sglang/sgl-kernel/" in lower:
        root = Path("/sgl-workspace/sglang/sgl-kernel")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."]
        artifact_roots = [root]
    elif "/sgl-workspace/sglang/" in lower:
        root = Path("/sgl-workspace/sglang")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "python"]
        artifact_roots = [root]
    elif "/sgl-workspace/vllm/" in lower:
        root = Path("/sgl-workspace/vllm")
        rebuild_command = ["/opt/venv/bin/python", "-m", "pip", "install", "-e", "."]
        artifact_roots = [root]
    elif allow_unknown_target:
        root = target_file.parent

    if suffix in PYTHON_SOURCE_SUFFIXES:
        compiled = False
        rebuild_command = []
        artifact_roots = []

    return {
        "compiled": compiled,
        "root": str(root) if root else "",
        "rebuild_command": rebuild_command,
        "artifact_roots": artifact_roots,
    }


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


def revert_kernel_patch(manifest_path: str | Path) -> dict[str, Any]:
    """Revert a previously applied kernel patch from its manifest.

    Restores backed-up artifacts and source, clears Python caches when the
    target is Python, restores any aiter ``jit/build`` backup, and fans out a
    best-effort revert to multi-node pods. The manifest is updated in place
    with the reverted status.

    Args:
        manifest_path (str | Path): Path to the apply manifest JSON file.

    Returns:
        dict[str, Any]: A result dict with ``status``, ``manifest_path``,
            ``restored_paths``, ``reverted_at`` and, when fan-out ran, a
            ``multinode_revert`` block.
    """
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    restored: list[str] = []
    for item in manifest.get("artifacts", []):
        src = Path(item["backup_path"])
        dst = Path(item["path"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
    # Disposition-aware multi-file revert (snapshot deploy): added -> unlink,
    # modified/deleted -> restore old bytes. Falls back to the legacy singular
    # ``source_backup`` for manifests written before snapshot deploy.
    source_backups = manifest.get("source_backups")
    cache_cleared = False
    if source_backups:
        for entry in reversed(source_backups):
            disp = entry.get("disposition")
            dst = Path(entry["path"])
            if disp == "added":
                dst.unlink(missing_ok=True)
                restored.append(str(dst))
            else:
                bp = entry.get("backup_path")
                if bp and Path(bp).exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bp, dst, follow_symlinks=False)
                    restored.append(str(dst))
            if dst.suffix.lower() in PYTHON_SOURCE_SUFFIXES and not cache_cleared:
                manifest["revert_cache_clear"] = _clear_python_kernel_caches(dst)
                cache_cleared = True
    source_backup = manifest.get("source_backup") or {}
    if not source_backups and source_backup:
        src = Path(source_backup["backup_path"])
        dst = Path(source_backup["path"])
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(str(dst))
            if dst.suffix.lower() in PYTHON_SOURCE_SUFFIXES:
                manifest["revert_cache_clear"] = _clear_python_kernel_caches(dst)

    # Restore aiter jit/build/ (before multi-node fan-out) if apply moved it aside.
    jit_build_backup = manifest.get("jit_build_backup") or {}
    if jit_build_backup.get("status") == "ok":
        jit_build_restore = _restore_aiter_jit_build(jit_build_backup)
        manifest["jit_build_restore"] = jit_build_restore
        if jit_build_restore.get("status") == "ok" and jit_build_restore.get("restored_to"):
            restored.append(str(jit_build_restore["restored_to"]))

    # Restore the aiter cpp_itfs runtime cache moved aside during apply so a non-KEEP decision serves v0 (only present when apply moved cpp_itfs cache dirs).
    cpp_itfs_cache_backup = manifest.get("cpp_itfs_cache_backup") or {}
    if cpp_itfs_cache_backup.get("status") == "ok":
        cpp_itfs_cache_restore = _restore_aiter_cpp_itfs_cache(cpp_itfs_cache_backup)
        manifest["cpp_itfs_cache_restore"] = cpp_itfs_cache_restore
        if cpp_itfs_cache_restore.get("status") == "ok":
            restored.extend(cpp_itfs_cache_restore.get("restored", []))

    # Multi-node: fan-out a revert to every pod that received the apply (best-effort) so pod-side sglang loads v0 on next restart.
    multinode_info = manifest.get("multinode") or {}
    mn_revert: dict[str, Any] = {}
    if multinode_info and multinode_info.get("host_backup_map"):
        target_path = multinode_info.get("target_path") or (source_backup.get("path") if source_backup else "")
        backup_map = multinode_info.get("host_backup_map") or {}
        if target_path and backup_map:
            try:
                mn_revert = _dispatch_multinode_revert(
                    target_path=target_path,
                    backup_map=backup_map,
                )
            except Exception as exc:  # noqa: BLE001
                mn_revert = {"status": "failed", "error": str(exc)}

    reverted_at = _now()
    manifest["status"] = "reverted"
    manifest["reverted_at"] = reverted_at
    manifest["restored_paths"] = restored
    if mn_revert:
        manifest["multinode_revert"] = mn_revert
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "ok",
        "manifest_path": str(manifest_file),
        "restored_paths": restored,
        "reverted_at": reverted_at,
    }
    # Only attach multinode_revert when fan-out actually ran.
    if mn_revert:
        result["multinode_revert"] = mn_revert
    return result


def _multi_root_strategies(
    live_paths: Iterable[Path],
    *,
    allow_unknown_target: bool,
) -> list[dict[str, Any]]:
    """Compute the set of distinct rebuild strategies across edited files.

    A multi-file patch can touch files in several framework roots (e.g. a
    ``csrc/*.cu`` plus an ``aiter/ops/triton/*.py``). Driving rebuild off the
    primary target alone would skip compilation for companion compiled files, so
    derive ``compiled``/rebuild from the **whole** edited set.

    Args:
        live_paths (Iterable[Path]): The live target paths the patch writes.
        allow_unknown_target (bool): Passed through to :func:`_detect_strategy`.

    Returns:
        list[dict[str, Any]]: One strategy dict per distinct root that needs a
            rebuild (compiled roots only), each as returned by
            :func:`_detect_strategy`.
    """
    seen: set[str] = set()
    strategies: list[dict[str, Any]] = []
    for p in live_paths:
        try:
            strat = _detect_strategy(p, allow_unknown_target=allow_unknown_target)
        except ValueError:
            continue
        if not strat["compiled"]:
            continue
        key = strat["root"] or str(p.parent)
        if key in seen:
            continue
        seen.add(key)
        strategies.append(strat)
    return strategies


def apply_kernel_patch(
    *,
    patch_path: str | Path,
    target_file: str | Path,
    backup_root: str | Path,
    kernel_id: str = "",
    artifact_paths: Iterable[str] | None = None,
    rebuild_command: list[str] | str | None = None,
    rebuild_timeout_sec: int = 1800,
    skip_rebuild: bool = False,
    allow_unknown_target: bool = False,
    dry_run: bool = False,
    snapshot_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
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
        patch_path (str | Path): The replacement source file, or (snapshot mode)
            the unified diff manifest.
        target_file (str | Path): The primary file (drives timing/rebuild root).
        backup_root (str | Path): Root directory for backups and the manifest.
        kernel_id (str): Identifier for the kernel being patched.
        artifact_paths (Iterable[str] | None): Explicit compiled artifacts to
            back up in addition to discovered ones.
        rebuild_command (list[str] | str | None): Override rebuild command; a
            string is run via ``bash -lc``.
        rebuild_timeout_sec (int): Rebuild subprocess timeout in seconds.
        skip_rebuild (bool): When ``True``, skip the rebuild step.
        allow_unknown_target (bool): Allow targets outside the known roots.
        dry_run (bool): Prepare backups/manifest only, without applying.
        snapshot_dir (str | Path | None): When set, enables snapshot mode and
            holds byte-exact final contents mirrored at each write path.

    Returns:
        dict[str, Any]: A result dict with ``status`` and, on success, the
            manifest path, target, backup dir, compiled flag, artifact count,
            cache-clear / rebuild / jit-build records, and an optional
            ``multinode`` block.
    """
    if snapshot_dir is not None:
        return _apply_kernel_patch_snapshot(
            patch_path=patch_path,
            target_file=target_file,
            backup_root=backup_root,
            snapshot_dir=snapshot_dir,
            kernel_id=kernel_id,
            artifact_paths=artifact_paths,
            rebuild_command=rebuild_command,
            rebuild_timeout_sec=rebuild_timeout_sec,
            skip_rebuild=skip_rebuild,
            allow_unknown_target=allow_unknown_target,
            dry_run=dry_run,
            repo_root=repo_root,
        )
    patch = Path(patch_path).resolve()
    target = Path(target_file).resolve()
    if not patch.is_file():
        return {"status": "failed", "error": f"patch_path does not exist: {patch}"}
    if not target.is_file():
        return {"status": "failed", "error": f"target_file does not exist: {target}"}

    try:
        strategy = _detect_strategy(target, allow_unknown_target=allow_unknown_target)
        _validate_patch_source(patch, target)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": str(exc)}

    backup_dir = Path(backup_root) / f"{_safe_name(kernel_id or target.stem)}_{_path_hash(target)}"
    manifest_path = backup_dir / "manifest.json"
    source_backup = _copy_to_backup(target, backup_dir, "source")
    artifacts: list[dict[str, str]] = []
    if strategy["compiled"]:
        found = _discover_artifacts(strategy["artifact_roots"], artifact_paths)
        artifacts = [_copy_to_backup(path, backup_dir, "artifacts") for path in found]

    manifest = {
        "status": "prepared",
        "kernel_id": kernel_id,
        "patch_path": str(patch),
        "target_file": str(target),
        "source_backup": source_backup,
        "artifacts": artifacts,
        "strategy": {
            "compiled": strategy["compiled"],
            "root": strategy["root"],
        },
        "created_at": _now(),
    }
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        return {"status": "ok", "dry_run": True, "manifest_path": str(manifest_path)}

    shutil.copy2(patch, target)
    cache_clear = (
        _clear_python_kernel_caches(target)
        if target.suffix.lower() in PYTHON_SOURCE_SUFFIXES
        else {"status": "skipped", "reason": "not a python source target"}
    )

    # Multi-node: fan-out the patch to every RayJob pod, else hard-revert the sandbox copy.
    multinode_info: dict[str, Any] = {}
    if _is_multi_node():
        pod_backup_dir = os.environ.get(
            "HYPERLOOM_MN_KERNEL_BACKUP_DIR",
            _MN_POD_BACKUP_DIR_DEFAULT,
        )
        try:
            mn_apply = _dispatch_multinode_apply(
                target_file=target,
                patch_path=patch,
                kernel_id=kernel_id,
                backup_dir_on_pod=pod_backup_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Pod fan-out failed: revert sandbox copy to v0.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            return {
                "status": "failed",
                "error": (
                    f"multi-node apply fan-out failed; sandbox copy reverted to {source_backup['backup_path']}: {exc}"
                ),
                "manifest_path": str(manifest_path),
            }
        # Persist per-host backups {hostname: backup_path} so revert can find them.
        backup_map: dict[str, str] = {}
        for entry in mn_apply.get("per_node", []) or []:
            host = (entry.get("host") or "").strip()
            bp = (entry.get("backup_path") or "").strip()
            if host and bp:
                backup_map[host] = bp
        multinode_info = {
            "status": "ok",
            "target_path": str(target),
            "backup_dir_on_pod": pod_backup_dir,
            "host_backup_map": backup_map,
            "per_node": mn_apply.get("per_node", []),
        }
        # Persist multinode info now so a later rebuild failure reverts pod-side patches too.
        manifest["multinode"] = multinode_info
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    command: list[str] = []
    if isinstance(rebuild_command, str):
        command = ["/bin/bash", "-lc", rebuild_command]
    elif rebuild_command:
        command = list(rebuild_command)
    elif not skip_rebuild:
        command = list(strategy["rebuild_command"])

    rebuild = {"status": "skipped", "reason": "source-only patch or skip_rebuild=true"}
    jit_build_backup: dict[str, Any] = {
        "status": "skipped",
        "reason": "rebuild not run",
    }
    cpp_itfs_cache_backup: dict[str, Any] = {
        "status": "skipped",
        "reason": "rebuild not run",
        "is_cpp_itfs": False,
    }
    if strategy["compiled"] and not skip_rebuild:
        # Move aiter jit/build/ aside so post-rebuild import re-JITs cleanly.
        jit_build_backup = _invalidate_aiter_jit_build(target, backup_dir)
        if jit_build_backup.get("status") == "failed":
            # Refuse to rebuild against an inconsistent jit cache: restore v0 and bail.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            return {
                "status": "failed",
                "error_class": "aiter_jit_invalidation_failed",
                "error": (f"aiter jit/build/ invalidation failed: {jit_build_backup.get('error')}"),
                "manifest_path": str(manifest_path),
                "jit_build_backup": jit_build_backup,
            }
        if jit_build_backup.get("status") == "ok":
            # Persist the backup record before rebuild so a failure can still restore.
            manifest["jit_build_backup"] = jit_build_backup
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        # aiter cpp_itfs kernels (e.g. paged_attention -> pa_ragged)
        # are runtime-compiled into $HOME/.aiter/build/<md_name>_<hash>/ on
        # first call, NOT by setup.py develop, and the dir name hashes
        # params (not source) so pristine + patched collide -> the next
        # server reuses the stale .so. Move the affected runtime-cache dirs
        # aside so the re-baseline recompiles from clean state (GH #458).
        # No-op for non-cpp_itfs targets (sglang / vllm / other aiter csrc).
        cpp_itfs_cache_backup = _invalidate_aiter_cpp_itfs_cache(target, backup_dir)
        if cpp_itfs_cache_backup.get("status") == "failed":
            # Refuse to re-baseline against a stale runtime cache: restore
            # source + jit/build (if moved) so on-disk state matches v0,
            # then bail rather than score a possibly-stale binary.
            try:
                shutil.copy2(source_backup["backup_path"], target)
            except OSError:
                pass
            if jit_build_backup.get("status") == "ok":
                _restore_aiter_jit_build(jit_build_backup)
            return {
                "status": "failed",
                "error_class": "aiter_cpp_itfs_invalidation_failed",
                "error": (f"aiter cpp_itfs runtime cache invalidation failed: {cpp_itfs_cache_backup.get('error')}"),
                "manifest_path": str(manifest_path),
                "cpp_itfs_cache_backup": cpp_itfs_cache_backup,
            }
        if cpp_itfs_cache_backup.get("status") == "ok":
            # Persist BEFORE rebuild so a rebuild failure can restore the
            # moved-aside runtime cache via revert_kernel_patch.
            manifest["cpp_itfs_cache_backup"] = cpp_itfs_cache_backup
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        cwd = Path(strategy["root"] or target.parent)
        rebuild = _run_rebuild(command, cwd, rebuild_timeout_sec)
        if rebuild["status"] != "ok":
            revert = revert_kernel_patch(manifest_path)
            return {
                "status": "failed",
                "error": "rebuild failed; original source/artifacts restored",
                "manifest_path": str(manifest_path),
                "rebuild": rebuild,
                "revert": revert,
            }

    manifest["status"] = "applied"
    manifest["applied_at"] = _now()
    manifest["rebuild"] = rebuild
    manifest["cache_clear"] = cache_clear
    if jit_build_backup.get("status") in {"ok", "skipped"}:
        # Surface skipped reason too so manifest readers can audit it.
        manifest["jit_build_backup"] = jit_build_backup
    if cpp_itfs_cache_backup.get("status") in {"ok", "skipped"}:
        # Surface the cpp_itfs record (is_cpp_itfs + build_dir + module_names + invalidated_unix) so integrate can verify a rebuild and revert can restore the runtime cache.
        manifest["cpp_itfs_cache_backup"] = cpp_itfs_cache_backup
    # multinode block already persisted at fan-out time; don't rewrite it here.
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "target_file": str(target),
        "backup_dir": str(backup_dir),
        "compiled": bool(strategy["compiled"]),
        "artifact_count": len(artifacts),
        "cache_clear": cache_clear,
        "rebuild": rebuild,
        "jit_build_backup": jit_build_backup,
        "cpp_itfs_cache_backup": cpp_itfs_cache_backup,
    }
    # Only attach the multinode key when fan-out actually ran.
    if multinode_info:
        result["multinode"] = multinode_info
    return result


def _apply_kernel_patch_snapshot(
    *,
    patch_path: str | Path,
    target_file: str | Path,
    backup_root: str | Path,
    snapshot_dir: str | Path,
    kernel_id: str,
    artifact_paths: Iterable[str] | None,
    rebuild_command: list[str] | str | None,
    rebuild_timeout_sec: int,
    skip_rebuild: bool,
    allow_unknown_target: bool,
    dry_run: bool,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Snapshot-mode apply: land an entire multi-file patch atomically.

    See :func:`apply_kernel_patch` for the contract. The diff at ``patch_path``
    is parsed only as a manifest; contents come from ``snapshot_dir``. Rebuild,
    JIT/cpp_itfs invalidation, and multi-node fan-out are driven off the
    **whole** edited set so companion compiled files and worker pods are not
    missed, and all-or-nothing spans apply → rebuild (a rebuild failure restores
    every touched path).

    Returns:
        dict[str, Any]: Result dict mirroring the legacy mode plus
            ``touched`` (the applied paths).
    """
    patch = Path(patch_path).resolve()
    target = Path(target_file).resolve()
    if not patch.is_file():
        return {"status": "failed", "error": f"patch_path does not exist: {patch}"}

    try:
        descriptors = parse_patch_manifest(patch.read_text(encoding="utf-8", errors="replace"))
    except ValueError as exc:
        return {"status": "failed", "error": f"unparseable patch: {exc}"}
    if not descriptors:
        return {"status": "failed", "error": "patch has no file operations"}

    try:
        primary_strategy = _detect_strategy(target, allow_unknown_target=allow_unknown_target)
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}

    # The repo root is authoritative for resolving the patch's repo-relative
    # paths. Prefer the explicitly-threaded root (captured at verify time where
    # kernel_repo is known); fall back to the strategy root. Never silently guess
    # ``target.parent`` — a wrong root would resolve every manifest path into a
    # nested subdir, violating the byte-for-byte contract.
    resolved_root = str(repo_root or "") or primary_strategy["root"]
    if not resolved_root:
        return {
            "status": "failed",
            "error": (
                "snapshot mode requires a known repo root (pass repo_root or a "
                f"target under a known framework root): {target}"
            ),
        }
    repo_root = Path(resolved_root)
    backup_dir = Path(backup_root) / f"{_safe_name(kernel_id or target.stem)}_{_path_hash(target)}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = backup_dir / "manifest.json"

    # Resolve every write/delete path to absolutes for downstream rebuild logic.
    write_paths: list[Path] = []
    for desc in descriptors:
        try:
            dest = _contained_dest(repo_root, desc["path"])
        except ValueError as exc:
            return {"status": "failed", "error": str(exc), "path": desc["path"]}
        if desc["op"] == "write":
            write_paths.append(dest)

    rebuild_strategies = _multi_root_strategies(
        write_paths, allow_unknown_target=allow_unknown_target
    )
    compiled = bool(rebuild_strategies)

    artifacts: list[dict[str, str]] = []
    if compiled:
        roots: list[Path] = []
        for strat in rebuild_strategies:
            roots.extend(strat["artifact_roots"])
        found = _discover_artifacts(roots, artifact_paths)
        artifacts = [_copy_to_backup(p, backup_dir, "artifacts") for p in found]

    manifest = {
        "status": "prepared",
        "kernel_id": kernel_id,
        "patch_path": str(patch),
        "target_file": str(target),
        "mode": "snapshot",
        "descriptors": descriptors,
        "artifacts": artifacts,
        "strategy": {"compiled": compiled, "root": str(repo_root)},
        "created_at": _now(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        return {"status": "ok", "dry_run": True, "manifest_path": str(manifest_path)}

    applied = apply_snapshot(
        descriptors=descriptors,
        snapshot_dir=snapshot_dir,
        repo_root=repo_root,
        backup_dir=backup_dir,
    )
    if applied["status"] != "ok":
        return {"status": "failed", "error": applied.get("error"), "path": applied.get("path"),
                "manifest_path": str(manifest_path)}

    manifest["source_backups"] = applied["source_backups"]
    # Keep a singular source_backup for the primary so legacy manifest readers work.
    for entry in applied["source_backups"]:
        if Path(entry["path"]) == target and entry.get("backup_path"):
            manifest["source_backup"] = {"path": entry["path"], "backup_path": entry["backup_path"]}
            break
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Post-apply verification: every written path must be byte-identical to its
    # snapshot source, every deleted path gone. Any mismatch -> full restore.
    snap = Path(snapshot_dir)
    for desc in descriptors:
        dest = _contained_dest(repo_root, desc["path"])
        if desc["op"] == "delete":
            if dest.exists():
                revert_kernel_patch(manifest_path)
                return {"status": "failed", "error": f"post-verify: {desc['path']} not deleted",
                        "manifest_path": str(manifest_path)}
            continue
        src = snap / desc["path"]
        if src.is_symlink():
            if not dest.is_symlink() or os.readlink(dest) != os.readlink(src):
                revert_kernel_patch(manifest_path)
                return {"status": "failed", "error": f"post-verify symlink mismatch: {desc['path']}",
                        "manifest_path": str(manifest_path)}
        elif dest.read_bytes() != src.read_bytes():
            revert_kernel_patch(manifest_path)
            return {"status": "failed", "error": f"post-verify content mismatch: {desc['path']}",
                    "manifest_path": str(manifest_path)}

    cache_clear_paths = [p for p in write_paths if p.suffix.lower() in PYTHON_SOURCE_SUFFIXES]
    cache_clear = {"status": "skipped", "reason": "no python source target"}
    for p in cache_clear_paths:
        cache_clear = _clear_python_kernel_caches(p)

    # Multi-node fan-out: push every write path to every pod.
    multinode_info: dict[str, Any] = {}
    if _is_multi_node():
        pod_backup_dir = os.environ.get("HYPERLOOM_MN_KERNEL_BACKUP_DIR", _MN_POD_BACKUP_DIR_DEFAULT)
        per_node_all: list[dict[str, Any]] = []
        backup_map: dict[str, str] = {}
        try:
            for p in write_paths:
                mn_apply = _dispatch_multinode_apply(
                    target_file=p, patch_path=snap / Path(p).relative_to(repo_root),
                    kernel_id=kernel_id, backup_dir_on_pod=pod_backup_dir,
                )
                per_node_all.extend(mn_apply.get("per_node", []) or [])
        except Exception as exc:  # noqa: BLE001
            revert_kernel_patch(manifest_path)
            return {"status": "failed",
                    "error": f"multi-node apply fan-out failed; repo restored: {exc}",
                    "manifest_path": str(manifest_path)}
        for entry in per_node_all:
            host = (entry.get("host") or "").strip()
            bp = (entry.get("backup_path") or "").strip()
            if host and bp:
                backup_map[host] = bp
        multinode_info = {"status": "ok", "target_path": str(target),
                          "backup_dir_on_pod": pod_backup_dir,
                          "host_backup_map": backup_map, "per_node": per_node_all}
        manifest["multinode"] = multinode_info
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    command: list[str] = []
    if isinstance(rebuild_command, str):
        command = ["/bin/bash", "-lc", rebuild_command]
    elif rebuild_command:
        command = list(rebuild_command)

    rebuild: dict[str, Any] = {"status": "skipped", "reason": "source-only patch or skip_rebuild=true"}
    jit_build_backup: dict[str, Any] = {"status": "skipped", "reason": "rebuild not run"}
    cpp_itfs_cache_backup: dict[str, Any] = {"status": "skipped", "reason": "rebuild not run", "is_cpp_itfs": False}
    if compiled and not skip_rebuild:
        jit_build_backup = _invalidate_aiter_jit_build(target, backup_dir)
        if jit_build_backup.get("status") == "failed":
            revert_kernel_patch(manifest_path)
            return {"status": "failed", "error_class": "aiter_jit_invalidation_failed",
                    "error": f"aiter jit/build/ invalidation failed: {jit_build_backup.get('error')}",
                    "manifest_path": str(manifest_path), "jit_build_backup": jit_build_backup}
        if jit_build_backup.get("status") == "ok":
            manifest["jit_build_backup"] = jit_build_backup
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        cpp_itfs_cache_backup = _invalidate_aiter_cpp_itfs_cache(target, backup_dir)
        if cpp_itfs_cache_backup.get("status") == "failed":
            revert_kernel_patch(manifest_path)
            return {"status": "failed", "error_class": "aiter_cpp_itfs_invalidation_failed",
                    "error": f"aiter cpp_itfs runtime cache invalidation failed: {cpp_itfs_cache_backup.get('error')}",
                    "manifest_path": str(manifest_path), "cpp_itfs_cache_backup": cpp_itfs_cache_backup}
        if cpp_itfs_cache_backup.get("status") == "ok":
            manifest["cpp_itfs_cache_backup"] = cpp_itfs_cache_backup
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Rebuild each distinct compiled root. Any failure -> full restore.
        rebuild_records: list[dict[str, Any]] = []
        for strat in rebuild_strategies:
            cmd = command or list(strat["rebuild_command"])
            cwd = Path(strat["root"] or target.parent)
            rec = _run_rebuild(cmd, cwd, rebuild_timeout_sec)
            rebuild_records.append(rec)
            if rec["status"] != "ok":
                revert = revert_kernel_patch(manifest_path)
                return {"status": "failed",
                        "error": "rebuild failed; original source/artifacts restored",
                        "manifest_path": str(manifest_path), "rebuild": rec, "revert": revert}
        rebuild = rebuild_records[-1] if rebuild_records else rebuild

    manifest["status"] = "applied"
    manifest["applied_at"] = _now()
    manifest["rebuild"] = rebuild
    manifest["cache_clear"] = cache_clear
    if jit_build_backup.get("status") in {"ok", "skipped"}:
        manifest["jit_build_backup"] = jit_build_backup
    if cpp_itfs_cache_backup.get("status") in {"ok", "skipped"}:
        manifest["cpp_itfs_cache_backup"] = cpp_itfs_cache_backup
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "status": "ok", "manifest_path": str(manifest_path), "target_file": str(target),
        "backup_dir": str(backup_dir), "compiled": compiled, "artifact_count": len(artifacts),
        "cache_clear": cache_clear, "rebuild": rebuild, "jit_build_backup": jit_build_backup,
        "cpp_itfs_cache_backup": cpp_itfs_cache_backup, "touched": applied["touched"],
    }
    if multinode_info:
        result["multinode"] = multinode_info
    return result


def main() -> int:
    """Run the apply/revert CLI and print the JSON result.

    Parses ``apply`` / ``revert`` subcommands, dispatches to
    :func:`apply_kernel_patch` or :func:`revert_kernel_patch`, and prints the
    result document.

    Returns:
        int: ``0`` when the result status is ``"ok"``, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description="Apply or revert a kernel patch")
    sub = parser.add_subparsers(dest="command", required=True)
    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--patch-path", required=True)
    apply_p.add_argument("--target-file", required=True)
    apply_p.add_argument("--backup-root", required=True)
    apply_p.add_argument("--kernel-id", default="")
    apply_p.add_argument("--artifact-path", action="append", default=[])
    apply_p.add_argument("--rebuild-command", default="")
    apply_p.add_argument("--rebuild-timeout-sec", type=int, default=1800)
    apply_p.add_argument("--skip-rebuild", action="store_true")
    apply_p.add_argument("--allow-unknown-target", action="store_true")
    apply_p.add_argument("--dry-run", action="store_true")

    revert_p = sub.add_parser("revert")
    revert_p.add_argument("--manifest-path", required=True)

    args = parser.parse_args()
    if args.command == "revert":
        result = revert_kernel_patch(args.manifest_path)
    else:
        result = apply_kernel_patch(
            patch_path=args.patch_path,
            target_file=args.target_file,
            backup_root=args.backup_root,
            kernel_id=args.kernel_id,
            artifact_paths=args.artifact_path,
            rebuild_command=args.rebuild_command or None,
            rebuild_timeout_sec=args.rebuild_timeout_sec,
            skip_rebuild=args.skip_rebuild,
            allow_unknown_target=args.allow_unknown_target,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
