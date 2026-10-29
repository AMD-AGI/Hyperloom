# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The files a forge-loop campaign keeps in its workspace, and the ones a candidate may edit."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from kernelforge.config.edit_policy import FORGE_LOOP_OUTPUT_DIRNAME, FORGE_PROTECTED_PATHS
from kernelforge.utils.git import git_bytes


def hide_forge_loop_output(workspace: str | Path) -> Path:
    """Create forge-loop's output directory under ``workspace``, hide it from Git, and return it.

    A ``.gitignore`` holding ``*`` ignores everything beside it, itself included, so Git never lists the directory:
    nothing the loop or a session writes there is committed or reported as a new file, whatever the workspace ignores.
    """
    output_root = Path(workspace) / FORGE_LOOP_OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / ".gitignore").write_text("*\n", encoding="utf-8")
    return output_root


def tracked_editable_paths(
    workspace: str | Path,
    *,
    exact_protected_paths: Iterable[str | Path] = (),
    extra_protected_globs: Iterable[str] = (),
) -> set[str]:
    """Return every tracked workspace path outside the protected measurement set."""

    root = Path(workspace).expanduser().resolve()
    result = git_bytes("ls-files", "-z", cwd=root, check=False)
    if result.returncode != 0:
        return set()
    editable: set[str] = set()
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        relative = encoded.decode(errors="surrogateescape")
        if not FORGE_PROTECTED_PATHS.is_protected(
            relative,
            workspace=root,
            exact_paths=exact_protected_paths,
            extra_globs=extra_protected_globs,
        ):
            editable.add(Path(relative).as_posix())
    return editable
