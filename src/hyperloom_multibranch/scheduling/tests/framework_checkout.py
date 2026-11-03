# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A framework git checkout for the scheduling tests, which a session roots on."""

from __future__ import annotations

import subprocess
from pathlib import Path

REMOTE_URL = "https://example.invalid/sglang.git"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True, check=True).stdout.strip()


def framework_checkout(parent: Path, remote: str = REMOTE_URL) -> Path:
    """A git checkout ``parent/framework`` with one commit and, unless ``remote`` is empty, an ``origin`` remote."""

    root = parent / "framework"
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "f.py").write_text("a\n")
    _git(root, "add", "f.py")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "init")
    if remote:
        _git(root, "remote", "add", "origin", remote)
    return root


def head_of(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")
