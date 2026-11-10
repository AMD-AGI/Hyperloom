# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Git checkouts a test needs on disk, such as the reference material a run is given."""

from __future__ import annotations

import subprocess
from pathlib import Path

#: What the knowledge file in a checkout holds, so a test can tell the copy from anything else.
KNOWLEDGE_TEXT = "pad M to the next multiple of 64"


def knowledge_checkout(root: Path, *files: str) -> str:
    """A git checkout at ``root`` holding ``files``, each with :data:`KNOWLEDGE_TEXT` in it; returns its commit."""

    root.mkdir(parents=True)
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(KNOWLEDGE_TEXT, encoding="utf-8")
    git = ("git", "-C", str(root), "-c", "user.email=bench@test", "-c", "user.name=bench")
    for arguments in (("init", "-q"), ("add", "."), ("commit", "-qm", "knowledge")):
        subprocess.run((*git, *arguments), check=True, capture_output=True)
    return subprocess.run((*git, "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()


__all__ = ["KNOWLEDGE_TEXT", "knowledge_checkout"]
