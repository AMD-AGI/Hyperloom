# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-root identity and content capture at the enablement KEEP.

A patch or artifact whose tree is unnamed cannot be replayed, and a non-git root
has no content identity at all, so each contributing root is named by the
operation that bound it, mapped to an anchor a fresh image can resolve, and
captured byte-exact by the shipped snapshot mechanism.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ...source_snapshot import snapshot_source_layer
from .projections import root_id_for

PATCH_APPLY = "patch_apply"
ARTIFACT_INSTALL = "artifact_install"

_PACKAGE_ROOT_PARTS: frozenset[str] = frozenset({"site-packages", "dist-packages"})


def _package_anchor(root: Path) -> tuple[str, str] | None:
    """Return ``(anchor, rel)`` when ``root`` sits under a package directory."""
    parts = root.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] in _PACKAGE_ROOT_PARTS:
            return "site_packages", str(Path(*parts[index + 1 :])) if index + 1 < len(parts) else ""
    return None


def classify_root(root: str, *, session_framework_root: str) -> tuple[str, dict[str, str]]:
    """Return the root's ``kind`` and the ``replay_target`` a consumer resolves.

    ``kind`` is derived from the path alone; ``replay_target`` is the mapping
    contract -- an anchor a fresh image has and a path relative to it, which the
    opaque id could not express for a second package or ``other`` root.
    """
    path = Path(str(root))
    if session_framework_root and path == Path(session_framework_root):
        return "framework_checkout", {"anchor": "framework_root", "rel": ""}
    package = _package_anchor(path)
    if package is not None:
        anchor, rel = package
        return "site_packages", {"anchor": anchor, "rel": rel}
    return "other", {"anchor": "unmappable", "rel": ""}


def build_root_records(
    *,
    contributions: Mapping[str, set[str]],
    base_sha_by_root: Mapping[str, str],
    git_roots: Iterable[str],
    session_framework_root: str,
) -> list[dict[str, Any]]:
    """Build one record per root that contributed to the accepted stack.

    A root carrying a ``patch_apply`` contribution is a build input and one
    carrying ``artifact_install`` is an output target: the input/output split, in
    the only terms the two binding resolvers can produce.
    """
    git = {str(r) for r in git_roots}
    records: list[dict[str, Any]] = []
    for root in sorted(contributions):
        kind, replay_target = classify_root(root, session_framework_root=session_framework_root)
        records.append(
            {
                "id": root_id_for(root),
                "path": root,
                "kind": kind,
                "contributions": sorted(contributions[root]),
                "is_git": root in git,
                # ``null`` by construction for a non-git root, which is exactly
                # why ``is_git`` sits beside it.
                "base_sha": str(base_sha_by_root.get(root) or "") if root in git else "",
                "replay_target": replay_target,
            }
        )
    return records


def accepted_stack_artifacts(
    *,
    inherited: Sequence[Mapping[str, Any]],
    applied: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the artifact set present at this validation, deduped by target.

    The stack a KEEP launched is every artifact the round inherited plus the
    ones it installed; deduplication is by ``target`` with this round's record
    last, matching the durable stacking that hands the next round its base set.
    """
    by_target: dict[str, dict[str, Any]] = {}
    for artifact in (*(inherited or ()), *(applied or ())):
        if not isinstance(artifact, Mapping):
            continue
        target = str(artifact.get("target") or "")
        if target:
            by_target[target] = dict(artifact)
    return list(by_target.values())


def collect_contributions(
    *,
    framework_root: str,
    patch_roots: Mapping[str, str] | None,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Group the accepted stack's bindings by the root each resolver returned."""
    contributions: dict[str, set[str]] = {}
    for root in set((patch_roots or {}).values()) or ({framework_root} if framework_root else set()):
        contributions.setdefault(str(root), set()).add(PATCH_APPLY)
    for artifact in artifacts or ():
        if not isinstance(artifact, Mapping):
            continue
        root = str(artifact.get("root") or framework_root or "")
        if root:
            contributions.setdefault(root, set()).add(ARTIFACT_INSTALL)
    return contributions


def declared_targets(
    *,
    framework_root: str,
    upserted: Sequence[str],
    deleted: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Return ``{root: {rel: expected_op}}`` for the accepted stack's targets.

    Every target is matched against the operation it was *declared* with:
    requiring an upsert everywhere would fail every correctly captured deletion,
    while a presence test would pass a KEEP that ran with its mutation inputs
    stripped and captured the base file.
    """
    targets: dict[str, dict[str, str]] = {}
    for rel in upserted:
        targets.setdefault(framework_root, {})[str(rel)] = "upsert"
    for rel in deleted:
        targets.setdefault(framework_root, {})[str(rel)] = "delete"
    for artifact in artifacts or ():
        if not isinstance(artifact, Mapping) or not artifact.get("rel_target"):
            continue
        root = str(artifact.get("root") or framework_root or "")
        targets.setdefault(root, {})[str(artifact.get("rel_target"))] = "upsert"
    return targets


def capture_root_snapshots(
    *,
    records: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, str]],
    dest_root: Path,
    session_dir: Path,
    import_root: str = "",
) -> list[dict[str, Any]]:
    """Capture each contributing root's declared targets, one snapshot per root.

    ``snapshot_source_layer`` captures paths under exactly one root, so a
    multi-root round needs one invocation per root; a capture that finds nothing
    returns no manifest at all, which the sufficiency rules read as a missing
    snapshot rather than an empty-but-complete one.
    """
    manifests: list[dict[str, Any]] = []
    captured: set[str] = set()
    for record in records:
        root = str(record.get("path") or "")
        declared = dict(targets.get(root) or {})
        if not declared:
            continue
        dest = dest_root / str(record.get("id") or "")
        # The id is a digest of the root path, so every KEEP round of a session
        # captures into one directory; the mechanism overwrites its manifest but
        # never clears ``files/``, which would leave a consumer overlaying a
        # target no longer in the accepted stack.
        shutil.rmtree(dest, ignore_errors=True)
        manifest = snapshot_source_layer(
            framework_root=root,
            base_sha=str(record.get("base_sha") or ""),
            rel_paths=sorted(declared),
            dest_dir=dest,
            provenance="enablement_keep",
            declared_ops=declared,
            import_root=import_root,
        )
        if not manifest:
            continue
        captured.add(str(record.get("id") or ""))
        manifests.append(_portable_manifest(manifest, record=record, session_dir=session_dir))
    _prune_overlay(dest_root, keep=captured)
    return manifests


def _prune_overlay(dest_root: Path, *, keep: set[str]) -> None:
    """Drop overlay directories for roots absent from this KEEP's manifests.

    The delivery selects the overlay by path glob rather than by manifest, so a
    directory an earlier round left behind still ships and the shipped stack
    becomes the union of the rounds instead of the accepted one.
    """
    if not dest_root.is_dir():
        return
    for child in dest_root.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)


def _portable_manifest(
    manifest: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    session_dir: Path,
) -> dict[str, Any]:
    """Replace the manifest's two absolute path fields with portable references."""
    snapshot_dir = Path(str(manifest.get("snapshot_dir") or ""))
    try:
        snapshot_ref = str(snapshot_dir.relative_to(session_dir))
    except ValueError:
        snapshot_ref = snapshot_dir.name
    portable = {k: v for k, v in manifest.items() if k not in ("framework_root", "snapshot_dir")}
    portable["root_id"] = str(record.get("id") or "")
    portable["snapshot_ref"] = snapshot_ref
    return portable
