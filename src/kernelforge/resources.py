# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime access to packaged KernelForge resources and writable state roots."""

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
    """Locate a shipped resource directory or file."""
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
    """Writable root for mutable artifacts (experiments, caches, learned KB)."""
    configured = os.environ.get("KERNELFORGE_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    user_data_path = os.environ.get("USER_DATA_PATH", "").strip()
    if user_data_path:
        return (Path(user_data_path).expanduser() / _STATE_DIR_NAME).resolve()
    return (Path("~/.cache/hyperloom").expanduser() / _STATE_DIR_NAME).resolve()


def writable_knowledge_root() -> Path:
    """Writable destination for knowledge the loop *produces*."""
    return default_project_root() / "knowledge_base"


def assert_sandbox_grant(path: str | Path, *, what: str) -> Path:
    """Validate a directory before it is added to an agent sandbox allowlist."""
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"{what} is not a directory: {resolved}")
    if resolved == _PACKAGE_ROOT or resolved in _PACKAGE_ROOT.parents:
        raise ValueError(
            f"{what} resolved to {resolved}, which contains the kernelforge package itself; "
            "granting it to an agent sandbox would expose the whole installation"
        )
    return resolved
