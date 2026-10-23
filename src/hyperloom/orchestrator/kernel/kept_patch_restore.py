# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Put a session's kept kernel patches back on the pods of its multi-node cluster."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from hyperloom.inference_optimizer.multi_node import external_has_server_control

from .apply_kernel_patch import (
    _MANIFEST_NAME,
    PodPatchApply,
    _dispatch_multinode_apply,
    _dispatch_multinode_finalize,
    _is_multi_node,
    _pod_backup_dir,
    _PodFanout,
    _target_uses_aiter_jit,
)

log = logging.getLogger(__name__)

_KEPT_STATUSES = frozenset({"finalized", "finalized_partial"})
_JIT_INVALIDATED_STATUSES = frozenset({"remote", "ok", "clean"})


class KeptPatchRestoreError(RuntimeError):
    """A kept kernel patch could not be put back on the pods."""


class _KeptManifest(NamedTuple):
    path: Path
    data: dict[str, Any]


def restore_kept_patches_on_pods(backup_roots: Iterable[Path]) -> list[Path]:
    """Send every file a kept kernel patch changed to every pod, as this host holds it now.

    A pod created after a KEEP starts from the image's sources, and nothing else
    re-sends a finalized patch. This host's tree is the reference because every
    apply and revert lands here before it fans out, so a file kept more than once
    needs no ordering. A run that is not multi-node has no pods, and a cluster
    without server control (benchmark-only) is left as it is, since no restart
    would load the files.

    Args:
        backup_roots: Apply backup roots; each subdirectory holds one apply's manifest.

    Returns:
        The files sent, each once.

    Raises:
        KeptPatchRestoreError: A manifest is unreadable, a kept file is missing on
            this host, or sending a file to the pods failed.
    """
    if not _is_multi_node():
        return []
    if not external_has_server_control():
        log.warning(
            "no server control on the multi-node cluster: its pods keep their sources, since no restart would load "
            "the kept kernel patches"
        )
        return []
    sends = _kept_file_sends(_kept_manifests(backup_roots))
    fanout = _PodFanout()
    for send in sends:
        try:
            fanout.absorb(_dispatch_multinode_apply(send))
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            raise KeptPatchRestoreError(
                f"sending kept kernel file {send.target_file} to the pods failed: {exc}"
            ) from exc
    if fanout.records_by_host:
        cleanup = _dispatch_multinode_finalize(fanout.records_by_host)
        if cleanup.get("status") != "ok":
            log.warning("kept kernel files restored, but their pod-side backups were not all deleted: %s", cleanup)
    return [send.target_file for send in sends]


def _kept_manifests(backup_roots: Iterable[Path]) -> list[_KeptManifest]:
    kept: list[_KeptManifest] = []
    for manifest_file in sorted(path for root in backup_roots for path in Path(root).glob(f"*/{_MANIFEST_NAME}")):
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KeptPatchRestoreError(f"cannot read kernel patch manifest {manifest_file}: {exc}") from exc
        if not isinstance(data, dict):
            raise KeptPatchRestoreError(f"kernel patch manifest is not a JSON object: {manifest_file}")
        if data.get("status") in _KEPT_STATUSES:
            kept.append(_KeptManifest(manifest_file, data))
    return kept


def _kept_paths(manifest: dict[str, Any]) -> list[Path]:
    if manifest.get("mode") != "snapshot":
        return [Path(manifest["target_file"])]
    return [
        Path(entry["path"]) for entry in manifest.get("source_backups") or [] if entry.get("disposition") != "deleted"
    ]


def _jit_invalidation(manifest: dict[str, Any]) -> tuple[str, Path] | None:
    """The AITER JIT build dir the KEEP's apply invalidated, and the file that carried it to the pods."""
    jit_build_dirs = (manifest.get("strategy") or {}).get("jit_build_dirs") or []
    if not jit_build_dirs or (manifest.get("jit_build_backup") or {}).get("status") not in _JIT_INVALIDATED_STATUSES:
        return None
    paths = _kept_paths(manifest)
    if manifest.get("mode") != "snapshot":
        return str(jit_build_dirs[0]), paths[0]
    carrier = next((path for path in paths if _target_uses_aiter_jit(path)), None)
    return (str(jit_build_dirs[0]), carrier) if carrier else None


def _kept_file_sends(kept: list[_KeptManifest]) -> list[PodPatchApply]:
    kernel_ids: dict[Path, str] = {}
    jit_carriers: dict[str, Path] = {}
    for manifest in kept:
        for path in _kept_paths(manifest.data):
            if not path.is_file():
                raise KeptPatchRestoreError(f"kept kernel file {path} from {manifest.path} is missing on this host")
            kernel_ids.setdefault(path, str(manifest.data.get("kernel_id") or ""))
        invalidation = _jit_invalidation(manifest.data)
        if invalidation:
            jit_carriers.setdefault(*invalidation)
    jit_build_dir_by_path = {path: jit_build_dir for jit_build_dir, path in jit_carriers.items()}
    return [
        PodPatchApply(
            target_file=path,
            patch_path=path,
            kernel_id=kernel_id,
            backup_dir_on_pod=_pod_backup_dir(),
            jit_build_dir=jit_build_dir_by_path.get(path, ""),
        )
        for path, kernel_id in kernel_ids.items()
    ]
