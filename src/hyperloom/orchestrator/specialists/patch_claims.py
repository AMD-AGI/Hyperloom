# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sandbox path checks for the patches a specialist claims or leaves behind."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def sibling_checkouts(roots: tuple[str, ...], base: Path | None) -> tuple[Path, ...]:
    """Return the configured source trees other than ``base``.

    Grounding falls back to these when the worktree base does not hold a
    patch's targets, which is the normal case for a specialist that patches a
    framework other than the one its worktree was cut from.

    A tree does not have to be a git checkout to be a candidate: target
    matching only stats files, and ``git apply --check`` runs against a plain
    directory. Requiring ``.git`` here left a pip-installed framework with no
    candidate at all, so every patch it wrote was dropped for a target that was
    on disk the whole time.

    Args:
        roots: The configured framework source roots.
        base: The checkout the specialist worktree branched off, if any.

    Returns:
        The remaining roots that exist, in discovery order.
    """
    base_resolved = base.resolve() if base else None
    out: list[Path] = []
    for raw in roots:
        root = Path(raw)
        if not root.is_dir() or root.resolve() == base_resolved:
            continue
        out.append(root)
    return tuple(out)


def grounding_explicit_root(
    *,
    declared: str,
    patches: list[str],
    patch_roots: dict[str, str],
) -> Path | None:
    """Return the root to ground the whole patch set against, or ``None``.

    ``vet_patches`` grounds one set against one root, so a harvested root can
    stand in only when every patch was harvested and they agree on it. A
    hand-authored patch alongside a harvest has an unknown target tree, and
    grounding it against the harvest root drops it as a mismatch rather than
    matching it against the candidates.

    Args:
        declared: ``framework_source_root`` from the task params, if any.
        patches: The deduplicated patch set about to be vetted.
        patch_roots: Apply roots recorded for harvested patches.

    Returns:
        Path | None: The root, or None to fall back to candidate matching.
    """
    if declared:
        return Path(declared)
    if not patches or any(patch not in patch_roots for patch in patches):
        return None
    roots = set(patch_roots.values())
    return Path(roots.pop()) if len(roots) == 1 else None


def resolve_claimed_patch(p: Any, bases: list[Path]) -> str | None:
    """The first existing sandbox-internal file a claimed patch path names, absolute or under one of ``bases``."""
    raw = Path(str(p))
    candidates = [raw] if raw.is_absolute() else []
    candidates += [base / raw for base in bases]
    for c in candidates:
        try:
            if c.is_file() and patch_path_within_bases(c, bases):
                return str(c)
        except OSError:
            continue
    return None


def patch_path_within_bases(path: Path, bases: list[Path]) -> bool:
    """True when ``path`` resolves inside one of the specialist sandbox bases.

    A claimed patch path (possibly absolute or ``..``-relative) must stay under
    the specialist worktree/workspace before it is read back; only
    sandbox-internal paths are legitimate, and none may pass through a symlink.
    """
    try:
        rp = path.resolve()
    except OSError:
        return False
    lexical = Path(os.path.normpath(path))
    for base in bases:
        try:
            br = base.resolve()
            rel = lexical.relative_to(os.path.normpath(base))
        except (OSError, ValueError):
            continue
        # The sandbox is writable by a demoted child: a link on the way (a planted ``patches`` symlink) would make
        # the path resolve somewhere else inside the base, so what is named must be what is read.
        if rp == br / rel:
            return True
    return False
