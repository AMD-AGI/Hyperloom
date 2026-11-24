# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The restore steps a kernel-patch revert runs from its manifest: artifacts, sources, JIT and cpp_itfs caches, pods."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom_common.timeutil import now_iso

from .patch_jit_cache import _aiter_cpp_itfs_build_dir, _restore_aiter_jit_build, _within_root
from .patch_rebuild import PYTHON_SOURCE_SUFFIXES

log = logging.getLogger(__name__)

MANIFEST_MULTINODE = "multinode"


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


def clear_python_kernel_caches(target: Path) -> dict[str, Any]:
    """Invalidate Python/compiler artifacts, including vLLM graphs referencing Triton.

    Resolve vLLM's root from VLLM_CACHE_ROOT, then XDG_CACHE_HOME, then HOME.
    Return the removed paths; filesystem removal remains best-effort.
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
    home_cache = home / ".cache"
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(home_cache)))
    vllm_cache_root = Path(os.environ.get("VLLM_CACHE_ROOT", str(cache_root / "vllm"))).expanduser()
    for path in (
        home / ".triton" / "cache",
        home_cache / "triton",
        home_cache / "torch" / "inductor",
        vllm_cache_root / "torch_compile_cache",
    ):
        if path.exists():
            remove_path(path)
    temp_root = Path(tempfile.gettempdir())
    for pattern in ("torchinductor_*", "triton_*"):
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


class _RestoreTally(NamedTuple):
    restored: list[str]
    untrusted: list[dict[str, str]]
    issues: list[dict[str, Any]]
    reported: dict[str, Any]


def restore_backed_up_artifacts(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
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


def restore_source_backups(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
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
            manifest["revert_cache_clear"] = clear_python_kernel_caches(dst)
            cache_cleared = True
    return _RestoreTally(restored, untrusted, [], {})


def restore_jit_build_state(manifest: dict[str, Any], backup_root: Path) -> _RestoreTally:
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


def restore_cpp_itfs_cache_state(manifest: dict[str, Any]) -> _RestoreTally:
    cpp_itfs_cache_backup = manifest.get("cpp_itfs_cache_backup") or {}
    if cpp_itfs_cache_backup.get("status") != "ok":
        return _RestoreTally([], [], [], {})
    cpp_itfs_cache_restore = _restore_aiter_cpp_itfs_cache(cpp_itfs_cache_backup)
    manifest["cpp_itfs_cache_restore"] = cpp_itfs_cache_restore
    if cpp_itfs_cache_restore.get("status") != "ok":
        return _RestoreTally([], [], [], {})
    return _RestoreTally(list(cpp_itfs_cache_restore.get("restored", [])), [], [], {})


def revert_on_patched_pods(manifest: dict[str, Any]) -> _RestoreTally:
    multinode_info = manifest.get(MANIFEST_MULTINODE) or {}
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


def write_revert_outcome(
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
