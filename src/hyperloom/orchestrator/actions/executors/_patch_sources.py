# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a patch comes from, which tree it applies to, and the stack it belongs to."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hyperloom.common.nofollow import read_text_nofollow
from hyperloom.common.github_urls import repo_slug
from hyperloom.inference_optimizer.framework_paths import (
    framework_apply_tree,
    resolve_framework_tree,
    resolve_kernel_search_roots,
    resolve_session_framework_root,
)

from . import _patch_snapshot
from ...specialists.patch_safety import resolve_patch_apply_root
from ._git import _git_head_sha
from ._integrate_attempt import IntegrateAttempt
from ._nogit_patch import _is_within
from ._patch_artifacts import _resolve_artifact_target

log = logging.getLogger(__name__)


def _acquiring_round(params: dict[str, Any]) -> bool:
    """True for an enablement round, except a launch-only bench.

    A launch-only bench carries a pre-built ``runtime_override`` and may
    acquire nothing of its own.
    """
    return bool(params.get("enablement")) and not params.get("enablement_launch_only")


def _first_ref_in_repo(refs: Iterable[Any], repo_url: str) -> str:
    """The best-ranked candidate ref that points at ``repo_url``.

    Discovery ranks bridge-repo PRs (aiter, HIP, ROCm) alongside the
    framework's own, and a diff is only applicable to the tree it was cut from.
    A bare ``"PR:1234"`` names no repo and is the framework's by construction.
    """
    try:
        want = repo_slug(repo_url)
    except ValueError:
        return ""
    for raw in refs:
        ref = str(raw).strip()
        if not ref:
            continue
        try:
            if repo_slug(ref) == want:
                return ref
        except ValueError:
            return ref
    return ""


def _done_field(done_payload: dict[str, Any] | None, key: str) -> Any:
    """Read ``key`` from a specialist done payload, top level or nested under ``payload``."""
    if not isinstance(done_payload, dict):
        return None
    if done_payload.get(key):
        return done_payload[key]
    inner = done_payload.get("payload")
    return inner.get(key) if isinstance(inner, dict) else None


def _read_done_payload(workspace: Path) -> dict[str, Any] | None:
    """Read and parse ``specialist_done.json`` from a workspace.

    Args:
        workspace (Path): The specialist task workspace directory.

    Returns:
        dict[str, Any] | None: The parsed payload, or ``None`` when the
        file is absent or cannot be parsed.
    """
    done = workspace / "specialist_done.json"
    if not done.exists():
        return None
    try:
        return json.loads(read_text_nofollow(done))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "integrate_patch: failed to parse %s: %r",
            done,
            exc,
        )
        return None


def _candidate_mutation_roots(*, params: dict[str, Any], done_payload: dict[str, Any] | None) -> list[str]:
    """Return every tree this round could mutate, before it mutates any of them.

    Read from the payload alone -- the root the round will apply into, the roots
    the authoring stage bound per patch, and the root each artifact target
    resolves into -- because the resolvers that return them run after the setup
    commands have already installed into those same trees.

    A declared ``framework_source_root`` is resolved here too: it is the tree the
    patches land in whenever it differs from the session's own, and the round
    does not otherwise name it until the stash, by which point setup has run.
    """
    explicit = str(params.get("framework_source_root") or "").strip() or None
    # No patch input, so this returns the declared root when one resolves and
    # the session root otherwise -- the pair the apply itself chooses between.
    effective = _resolve_framework_root(explicit, patch_paths=[])
    roots: list[str] = [str(resolve_session_framework_root() or ""), str(effective or "")]
    roots.extend(str(v) for v in ((done_payload or {}).get("patch_roots") or {}).values())
    entries = params.get("artifacts")
    if not isinstance(entries, list) or not entries:
        entries = (done_payload or {}).get("artifacts_written") or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        resolved = _resolve_artifact_target(str(entry.get("target") or "").strip())
        if resolved is not None:
            roots.append(str(resolved[2]))
    return list(dict.fromkeys(r for r in roots if r))


def _note_pre_mutation_head(
    attempt: IntegrateAttempt,
    root: str | Path | None,
    *,
    enablement: bool = False,
    session_dir: Path | None = None,
) -> None:
    """Record ``root``'s HEAD once, before the first mutation of it.

    Never replaced: a later read would name a tree that has already changed, and
    every patch the recipe carries would replay onto its own result.

    For an enablement round the scope of "once" is the STACK, not the round. A
    KEEP is committed, and an ADVANCED round commits and stacks its patch while
    recording no per-root identity of its own, so by the next round HEAD already
    contains every earlier round's work. Reading it again would name a tree the
    recipe's own earlier patch steps have already been applied to, and a
    consumer replaying that would apply them a second time. The durable map is
    therefore consulted before git and written first-writer-wins -- which also
    carries the reading across a resume, where the per-round map does not
    survive at all.

    Args:
        attempt: The integration owning the per-round map and shared-state reference.
        root: The tree about to be mutated.
        enablement: Whether this round belongs to the enablement stack. An
            ordinary patch round must NOT seed the enablement base: the head
            before an unrelated patch is not the tree the enablement stack
            applies to.
    """
    heads: dict[str, str] = attempt.base_sha_by_root
    key = str(root or "")
    if not key:
        return
    durable = attempt.shared_state.enablement.base_sha_by_root if enablement else {}
    if key not in heads:
        heads[key] = durable.get(key) or _git_head_sha(Path(key))
    if enablement and heads[key]:
        _persist_base_sha_for_root(attempt, key, heads[key], session_dir=session_dir)


def _persist_base_sha_for_root(
    attempt: IntegrateAttempt, root: str, sha: str, *, session_dir: Path | None = None
) -> None:
    """Record this root's pre-mutation head on the durable stack, once.

    Saved here rather than left for the rearm: the reading is only correct
    BEFORE the mutation, and the mutation is the next thing that happens. A
    round that commits its patch and then dies would otherwise resume with the
    entry absent and HEAD already moved, which is exactly the state this map
    exists to prevent.
    """
    shared_state = attempt.shared_state
    if not shared_state.enablement.record_base_sha(root, sha) or session_dir is None:
        return
    try:
        shared_state.save(session_dir)
    except (OSError, AttributeError):
        # The value is on the in-memory state either way, and the rearm saves
        # again; a failed write must not stop the round.
        log.debug("integrate_patch: save after base-sha record failed", exc_info=True)


def resolved_explicit_root(explicit: str) -> Path | None:
    """Resolve a declared framework root, or ``None`` when it is not a directory.

    Args:
        explicit: The declared ``framework_source_root``.

    Returns:
        The resolved directory, or ``None`` when it is unreadable or absent.
    """
    try:
        resolved = Path(explicit).resolve()
    except (OSError, RuntimeError):
        log.warning(
            "integrate_patch: framework_source_root override %r could not be resolved",
            explicit,
        )
        return None
    if not resolved.is_dir():
        log.warning(
            "integrate_patch: framework_source_root override %r does not exist",
            explicit,
        )
        return None
    return resolved


def _read_patch_texts(patch_paths: list[Path] | None) -> list[str]:
    """Return the diff text of every patch that could be read.

    Args:
        patch_paths: Patch files to read.

    Returns:
        The texts that were readable; a shorter list than ``patch_paths`` means
        some patch is missing or unreadable.
    """
    texts: list[str] = []
    for patch in patch_paths or []:
        try:
            texts.append(patch.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return texts


def _sole_patch_root(
    done_payload: dict[str, Any] | None,
    patch_paths: list[Path],
    *,
    specialist_workspace: Path,
) -> str | None:
    """Return the one apply root recorded for every patch, or ``None``.

    Only metadata covering the selected patch set can replace target
    resolution. Unselected patches cannot supply or contradict its root.
    Localization patches outside the specialist workspace deliberately use
    full-set content resolution: the specialist cannot attest their paths.

    Args:
        done_payload: The originating specialist's done payload, if any.
        patch_paths: The complete patch set selected for this integration.
        specialist_workspace: Workspace used to resolve recorded patch paths.

    Returns:
        The sole recorded root, or ``None``.
    """
    raw = (done_payload or {}).get("patch_roots")
    if not isinstance(raw, dict) or not patch_paths:
        return None
    try:
        selected = {patch.resolve() for patch in patch_paths}
    except (OSError, RuntimeError, ValueError):
        return None
    covered: set[Path] = set()
    roots: set[str] = set()
    for recorded_patch, root in raw.items():
        if not isinstance(recorded_patch, str) or not recorded_patch.strip():
            continue
        try:
            resolved = _resolve_patch_paths(
                specialist_workspace=specialist_workspace,
                explicit_patches=[recorded_patch],
                done_payload=None,
            )
        except (OSError, RuntimeError, ValueError):
            return None
        for patch in resolved:
            if patch not in selected:
                continue
            if not isinstance(root, str) or not root.strip():
                return None
            covered.add(patch)
            try:
                root_path = Path(root).expanduser()
                if not root_path.is_absolute():
                    root_path = specialist_workspace / root_path
                roots.add(str(root_path.resolve()))
            except (OSError, RuntimeError, ValueError):
                return None
    return roots.pop() if covered == selected and len(roots) == 1 else None


def _resolve_framework_root(
    explicit: str | None,
    patch_paths: list[Path] | None = None,
    patch_texts: list[str] | None = None,
    recorded_root: str | None = None,
) -> Path | None:
    """Pick one unambiguous framework root under the shared Patch rules.

    A ``recorded_root`` — carried from the authoring stage through
    ``done_payload["patch_roots"]`` — is authoritative and skips probing
    entirely.

    Without a recorded root, the decision falls through to
    :func:`~...specialists.patch_safety.resolve_patch_apply_root`. Without any
    patches to place, the declared root wins, then the session's root, then
    the root the session's framework tree is edited at: its checkout, or the
    install root of a pip-installed package.

    Args:
        explicit: Declared framework root.
        patch_paths: Patch files to place; unreadable ones are skipped.
        patch_texts: Patch diffs already in memory, placed alongside
            ``patch_paths``.
        recorded_root: The apply root recorded at authoring time, if any.

    Returns:
        The resolved root, or ``None`` when the patches name no single tree.
    """
    if recorded_root:
        return resolved_explicit_root(recorded_root)

    explicit_path: Path | None = None
    if explicit:
        explicit_path = resolved_explicit_root(explicit)
        if explicit_path is None:
            return None

    texts = [str(text) for text in (patch_texts or []) if str(text).strip()]
    texts.extend(_read_patch_texts(patch_paths))
    has_patch_input = bool(patch_paths or patch_texts)
    if has_patch_input:
        session_root = resolve_session_framework_root()
        # The search roots do not necessarily hold it: they discover the
        # unprefixed env var, while the session root also answers to
        # <FRAMEWORK>_REPO_PATH and <FRAMEWORK>_DIR. Leaving it out turns the
        # tree under optimisation into a non-candidate, and default_root cannot
        # stand in -- that is consulted only for a create-only set, which has no
        # pre-image to match.
        session_path = Path(session_root) if session_root else None
        roots = [Path(root) for root in resolve_kernel_search_roots()]
        resolution = resolve_patch_apply_root(
            texts,
            explicit_root=explicit_path,
            candidate_roots=tuple(filter(None, [session_path, *roots])),
            default_root=session_path,
        )
        if resolution.root is None:
            log.warning(
                "integrate_patch: Patch root resolution rejected: %s%s",
                resolution.reason,
                (f" matches={[str(root) for root in resolution.matches]!r}" if resolution.matches else ""),
            )
        return resolution.root

    if explicit_path is not None:
        return explicit_path
    session_root = resolve_session_framework_root()
    if session_root and Path(session_root).is_dir():
        return Path(session_root)
    framework = os.environ.get("FRAMEWORK", "")
    if not framework:
        return None
    tree = framework_apply_tree(resolve_framework_tree(framework))
    return tree.root if tree is not None else None


def _resolve_patch_paths(
    *,
    specialist_workspace: Path,
    explicit_patches: list[str] | None,
    done_payload: dict[str, Any] | None,
) -> list[Path]:
    """Resolve the list of patch files to apply.

    Order: ``params.patches`` → ``specialist_done.patches_written`` →
    filesystem scan of ``specialist_workspace/{worktree/,}patches/``.
    Entries normalised to absolute Paths; missing ones logged + dropped.

    Security: a resolved patch path must live inside the specialist workspace
    (or its worktree); an absolute path pointing outside the sandbox is dropped.
    Both sides are ``resolve()``-d first so a symlinked workspace still matches.

    Args:
        specialist_workspace: The specialist task workspace to resolve
            relative paths / scan for patches.
        explicit_patches: Explicit patch paths from params, or ``None``.
        done_payload: The parsed ``specialist_done.json`` payload, or ``None``.

    Returns:
        The resolved, existing patch files as absolute Paths.
    """
    candidates: list[str] = []
    if explicit_patches:
        candidates.extend(str(p) for p in explicit_patches)
    elif done_payload and isinstance(done_payload.get("patches_written"), list):
        candidates.extend(str(p) for p in done_payload["patches_written"] if p)
    else:
        for base in (
            specialist_workspace / "worktree" / "patches",
            specialist_workspace / "patches",
        ):
            if base.is_dir():
                for p in sorted(base.glob("*.patch")):
                    candidates.append(str(p))
                for p in sorted(base.glob("*.diff")):
                    candidates.append(str(p))

    allowed_roots = [
        (specialist_workspace / "worktree").resolve(),
        specialist_workspace.resolve(),
    ]

    out: list[Path] = []
    for c in candidates:
        p = Path(c)
        # Resolve relative paths against the specialist workspace + worktree.
        if not p.is_absolute():
            for base in (
                specialist_workspace / "worktree",
                specialist_workspace,
            ):
                cand = base / c
                if cand.exists():
                    p = cand
                    break
        if not p.exists():
            log.warning(
                "integrate_patch: patch %r not found (specialist_workspace=%s)",
                c,
                specialist_workspace,
            )
            continue
        resolved = p.resolve()
        if not any(_is_within(resolved, root) for root in allowed_roots):
            log.warning(
                "integrate_patch: patch %r resolves outside the specialist workspace (%s); dropping for safety",
                c,
                specialist_workspace,
            )
            continue
        out.append(resolved)
    return out


def _patch_stack_by_root(enablement: Any, patch_roots: dict[str, str], applied: list[Path]) -> dict[str, list[Path]]:
    """Group the accepted patch stack by the tree each patch applies to, in apply order.

    Apply order, not the mapping's: the operation a later patch declares must
    override an earlier one's for the same file.
    """
    accepted = enablement.accepted_stack(applied)
    ordered = [patch for patch in patch_roots if patch not in set(accepted)] + accepted
    by_root: dict[str, list[Path]] = {}
    for patch_path in dict.fromkeys(ordered):
        patch_root = str(patch_roots.get(patch_path) or "")
        if patch_root:
            by_root.setdefault(patch_root, []).append(Path(patch_path))
    return by_root


def _declared_stack_ops(patch_root: str, patches: list[Path], *, base_sha: str) -> dict[str, dict[str, str]]:
    """The file operations each patch declares against ``patch_root``, per patch.

    Proven, not believed: the whole stack is replayed from its base in an isolated
    tree, over the ORDERED stack, because a patch's preimage is the tree its
    predecessors left behind rather than this root's base or the final tree.
    """
    replayed = _patch_snapshot.replayed_stack_ops(Path(patch_root), patches, base_sha=base_sha)
    if replayed is not None:
        return replayed
    if base_sha:
        log.warning(
            "integrate_patch: enablement KEEP cannot replay the stack on %s from %s; "
            "its targets are left undeclared and the recipe is refused",
            patch_root,
            base_sha,
        )
        return {}
    # No base commit: a framework installed from a wheel, which is the ordinary
    # production shape. There is no "check out the base and apply" to prove, but
    # that is not how such a root is replayed -- it is replayed by OVERLAYING the
    # captured files, which is exactly what building an image from this recipe
    # does. Declaring nothing here removes those files from the capture and with
    # them the only replay path that applies, leaving the recipe unusable in the
    # case it is most needed for.
    #
    # What is NOT claimed is that the patch was proven applied: there is no
    # preimage to prove it against. Do not read an empty base_sha as the thing
    # that withholds that claim -- the judge's base_sha refusal is gated on
    # ``is_git``, so it never fires for a wheel root and this path can and does
    # reach ``sufficient``. The protection that does apply is the op map itself:
    # ``overlay_inventory_without_base`` returns None for anything it cannot
    # PROVE, and undeclared targets are what the judge blocks on. The targets are
    # what the overlay must contain, not evidence that the patch applied cleanly.
    overlaid: dict[str, dict[str, str]] = {}
    for patch in patches:
        ops = _patch_snapshot.overlay_inventory_without_base(Path(patch_root), patch)
        if ops is None:
            log.warning(
                "integrate_patch: enablement KEEP cannot resolve %s against the "
                "base-less root %s; its targets are left undeclared",
                patch,
                patch_root,
            )
            continue
        overlaid[str(patch)] = ops
    return overlaid


__all__ = [
    "_read_done_payload",
    "_resolve_framework_root",
    "_resolve_patch_paths",
    "resolved_explicit_root",
]
