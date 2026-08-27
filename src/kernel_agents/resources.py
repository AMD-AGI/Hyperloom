# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runtime access to source-tree and packaged KernelForge resources."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_ROOT.parent.parent
_PACKAGED_DATA_ROOT = _PACKAGE_ROOT / "data"


def is_source_checkout(root: Path = _SOURCE_ROOT) -> bool:
    """Return whether ``root`` looks like a KernelForge source checkout."""
    return (root / "pyproject.toml").is_file() and (root / "src" / "kernel_agents").is_dir()


def default_project_root() -> Path:
    """Default runtime root for mutable artifacts.

    In editable/source installs this remains the repository root for backwards
    compatibility. In wheel installs, use the caller's current directory rather
    than writing experiments under site-packages.
    """
    if is_source_checkout():
        return _SOURCE_ROOT
    return Path.cwd().resolve()


def resource_path(name: str, project_root: str | Path | None = None) -> Path:
    """Locate a shipped resource directory or file.

    Source checkouts keep resources at the repository root. Built wheels carry
    the same trees under ``kernel_agents/data``. An explicit ``project_root`` is
    honored first so tests and callers can point at custom resource trees.
    """
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(Path(project_root) / name)
    candidates.append(_SOURCE_ROOT / name)
    candidates.append(_PACKAGED_DATA_ROOT / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
