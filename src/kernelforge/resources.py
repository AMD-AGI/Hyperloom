# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime access to packaged KernelForge resources and writable state roots.

KernelForge ships inside the Hyperloom distribution, so its knowledge base,
examples and serving patches always live at ``kernelforge/data`` next to the
code -- there is no "repository root" to fall back to. Everything under that
tree is read-only: it may sit in a root-owned ``site-packages`` and is replaced
wholesale on upgrade. Mutable state therefore goes to a separately resolved
writable root, never back into the package.
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _PACKAGE_ROOT / "data"

#: Directory name for mutable state under the writable root.
_STATE_DIR_NAME = "kernelforge"


def packaged_data_root() -> Path:
    """Root of the read-only resource trees shipped inside the package."""
    return _DATA_ROOT


def resource_path(name: str, project_root: str | Path | None = None, *, missing_ok: bool = False) -> Path:
    """Locate a shipped resource directory or file.

    An explicit ``project_root`` is honored first, so an operator can drop their
    own ``knowledge_base``/``local_knowledge`` next to their experiments and have
    it win over the packaged copy. Otherwise the packaged tree is used.

    Raises ``FileNotFoundError`` when nothing resolves. Silently returning a
    non-existent path -- the previous behaviour -- meant a missing data tree
    surfaced as forge-loop running against an empty knowledge base, with no
    error and no log line. Pass ``missing_ok=True`` only where the caller has a
    real fallback for the resource being absent; it returns the packaged
    location so the caller can report a concrete path.
    """
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(Path(project_root) / name)
    candidates.append(_DATA_ROOT / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if missing_ok:
        return candidates[-1]
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"packaged KernelForge resource {name!r} not found; searched: {searched}")


def default_project_root() -> Path:
    """Writable root for mutable artifacts (experiments, caches, learned KB).

    Must never be ``site-packages`` (read-only, wiped on upgrade) nor the process
    working directory (scatters state wherever the caller happened to be). The
    precedence mirrors ``knowledge.experience_store.KnowledgeConfig.from_env``:

    ``$KERNELFORGE_PROJECT_ROOT`` -> ``$USER_DATA_PATH/kernelforge`` ->
    ``~/.cache/hyperloom/kernelforge``
    """
    configured = os.environ.get("KERNELFORGE_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    user_data_path = os.environ.get("USER_DATA_PATH", "").strip()
    if user_data_path:
        return (Path(user_data_path).expanduser() / _STATE_DIR_NAME).resolve()
    return (Path("~/.cache/hyperloom").expanduser() / _STATE_DIR_NAME).resolve()


def writable_knowledge_root() -> Path:
    """Writable destination for knowledge the loop *produces*.

    Postmortem lessons and the tuning DB are written here. The directory is
    created on demand by its callers.

    Note the name: there used to be a packaged, read-only ``knowledge_base``
    tree under ``kernelforge/data`` as well, and the two were easy to confuse.
    That one was removed once an audit found nothing read it. This path is the
    only ``knowledge_base`` left, and it is writable and outside the package.
    """
    return default_project_root() / "knowledge_base"


def assert_sandbox_grant(path: str | Path, *, what: str) -> Path:
    """Validate a directory before it is added to an agent sandbox allowlist.

    Claude's ``add_dirs`` grant is read *and* write, so a knowledge root that
    silently resolved too high up the tree would hand the agent the whole
    KernelForge code tree -- or worse. Before the data trees moved inside the
    package these paths were derived from a repository root, so a wrong answer
    was merely a missing directory; now it can be an over-broad one.

    Returns the resolved path. Raises ``ValueError`` if it does not exist, or if
    it contains the package itself.
    """
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"{what} is not a directory: {resolved}")
    if resolved == _PACKAGE_ROOT or resolved in _PACKAGE_ROOT.parents:
        raise ValueError(
            f"{what} resolved to {resolved}, which contains the kernelforge package itself; "
            "granting it to an agent sandbox would expose the whole installation"
        )
    return resolved
