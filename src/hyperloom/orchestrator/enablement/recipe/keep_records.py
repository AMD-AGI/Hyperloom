# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-root identity and content capture at the enablement KEEP.

A patch or artifact whose tree is unnamed cannot be replayed, and a non-git root
has no content identity at all, so each contributing root is named by the
operation that bound it, mapped to an anchor a fresh image can resolve, and
captured byte-exact by the shipped snapshot mechanism.

The same KEEP records what the tree alone can refute: the accepted levers no
file in the framework reads, and the linked build's compiled extensions the
framework root does not carry.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ...source_snapshot import snapshot_source_layer
from .projections import root_id_for, select_linked_build

if TYPE_CHECKING:
    from hyperloom.orchestrator.state._shared_state.enablement_round import EnablementRound

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
    for root in set((patch_roots or {}).values()):
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


def levers_without_readers(
    enablement: EnablementRound,
    framework_root: Path | None,
    *,
    framework: str,
    effective_config: Mapping[str, Any] | None = None,
) -> list[str] | None:
    """Return accepted env levers in the framework's namespace that nothing reads.

    A lever is accepted because a round that set it advanced, not because
    anything was shown to read it. A knob a specialist introduced in a patch
    that was later superseded leaves its name behind in ``accepted_config``,
    and the recipe then exports an env no code consults -- a replay sets it
    and reproduces nothing, silently.

    Only the framework's own namespace is judged. ``AMD_SERIALIZE_KERNEL``
    is read by the HIP runtime and ``NCCL_*`` by the collective library;
    their absence from the framework tree says nothing about them.

    Every regular file the framework ships is searched, matched as bytes. A
    lever is as likely to be read by a kernel through ``getenv`` or by a
    launch script through shell expansion as by Python, and a reader can sit
    in a file with no extension at all -- a ``Dockerfile``, a ``Makefile``.
    A suffix list is not evidence of absence: skipping a file is what turns
    a working lever into a refusal. A match inside a compiled artifact
    counts too, which can only make this miss a dangling lever, never invent
    one.

    Returns:
        The lever names with no reader, ``[]`` when a scan found none or
        there was nothing to scan for, and ``None`` when the tree was not
        resolved or could not be read -- which is not evidence that every
        lever has one.
    """
    if not framework.strip():
        return []
    # This KEEP's own effective config wins. The standing ``accepted_config``
    # is not replaced with it until the lane re-arms on the result, so a
    # lever this round introduced -- the one the recipe will export -- is
    # not in shared state yet, and scanning only that would check every
    # round's levers except the decisive one.
    envs = {**enablement.accepted_config.get("extra_envs", {}), **(effective_config or {}).get("extra_envs", {})}
    prefix = f"{framework.strip().upper()}_"
    names = sorted({str(k).strip() for k in envs if str(k).strip().startswith(prefix)})
    if not names:
        return []
    if framework_root is None or not framework_root.is_dir():
        # An empty walk over a tree that is not there would report every
        # lever as unread, which is a refusal built out of nothing.
        return None
    needles = {name: name.encode("ascii", "ignore") for name in names}
    unread = set(names)
    try:
        for source in framework_root.rglob("*"):
            if not unread:
                break
            if not source.is_file():
                continue
            blob = source.read_bytes()
            unread -= {name for name in unread if needles[name] in blob}
    except OSError:
        return None
    return sorted(unread)


def _build_output_trees(attempt_root: Path) -> list[Path]:
    """Return the trees a build names as its own output.

    The build records them in its ``result.json`` as the prefixes a runtime
    would import from; that is the build's own statement of where its output
    lives, so it is read rather than guessed at. A result that cannot be
    read, or names no prefix, names no tree.
    """
    result = attempt_root / "result.json"
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    runtime = payload.get("runtime") if isinstance(payload, dict) else None
    prefixes = (runtime or {}).get("pythonpath_prefixes") if isinstance(runtime, dict) else None
    return [Path(str(p)) for p in prefixes if str(p).strip()] if isinstance(prefixes, list) else []


def build_extensions_not_carried(
    enablement: EnablementRound, framework_root: Path | None, *, specialist_task_id: str = ""
) -> list[str] | None:
    """Return the build's compiled extensions the framework root does not have.

    A build does not install itself: its outputs reach the framework root
    only as artifacts a specialist declared, one by one. Declare two of
    three and the round still boots, benchmarks, and is kept -- the gap
    surfaces hours later as an op the loaded extension does not export, on
    whichever code path first needs it.

    Only the extensions built *for this framework package* are judged, and
    only inside the tree the build itself names as its output. An attempt
    root also holds the other repositories a build cloned and, where one was
    provisioned, a virtual environment with its own installed copy of this
    same package -- comparing against those would refuse a recipe over files
    the framework root was never meant to carry. Compared by
    content, so an extension the base image already shipped under the same
    name counts as not carried.

    Every shared object anywhere in the package is considered, not only
    ``.abi3.so`` directly beneath it: an extension built without the
    stable-ABI tag carries an interpreter-specific suffix instead, and one
    belonging to a subpackage sits below the package root. Each is compared
    at its path relative to the package, so a nested module is matched
    against the nested module rather than against a same-named file at the
    top, and the name reported is that relative path.

    Returns:
        The names left behind, ``[]`` only after at least one of the linked
        build's output trees was scanned and nothing was missing (or when no
        build is linked, there being nothing to carry), and ``None`` when a
        build is linked whose outputs could not be read -- a result that
        names no output tree, a named tree that is gone, an unreadable file,
        or no framework root to compare them against. None of those are
        evidence that anything was carried.
    """
    rounds = list(enablement.kept_rounds)
    current = str(specialist_task_id or "").strip()
    if current and not any(str((r or {}).get("task_id") or "").strip() == current for r in rounds):
        rounds.append({"task_id": current})
    state = {
        "build_manifest": list(enablement.build_manifest),
        "last_specialist_task_id": enablement.last_specialist_task_id,
        "kept_rounds": rounds,
    }
    _sentinel, row = select_linked_build(state)
    # Validated as text first: ``Path("")`` is ``Path(".")``, whose
    # ``is_dir()`` is true, so an absent attempt root would otherwise scan
    # the working directory and report whatever it found there.
    attempt_root_text = str((row or {}).get("attempt_root") or "").strip()
    if not attempt_root_text:
        return []
    if framework_root is None:
        return None
    attempt_root = Path(attempt_root_text)
    if not attempt_root.is_dir():
        return None
    missing: list[str] = []
    try:
        package_roots = [
            d for d in (prefix / framework_root.name for prefix in _build_output_trees(attempt_root)) if d.is_dir()
        ]
        if not package_roots:
            # The build named no output tree, or named trees that are gone.
            # Either way this scanned nothing, which is not the same as
            # finding nothing.
            return None
        built_files = sorted((package, built) for package in package_roots for built in package.rglob("*.so"))
    except OSError:
        return None
    for package, built in built_files:
        relative = built.relative_to(package)
        installed = framework_root / relative
        try:
            if not installed.is_file() or installed.read_bytes() != built.read_bytes():
                missing.append(str(relative))
        except OSError:
            return None
    return missing
