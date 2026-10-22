# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent skills that live outside the packaged knowledge tree, such as the one a kernel repository keeps."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from kernelforge.knowledge.local_wiki.index import _clip, _frontmatter_field

SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class ExternalSkill:
    """One skill folder: where it is, what it says it covers, and the repository its paths may be relative to."""

    root: Path
    name: str
    description: str
    repository_root: Path | None

    @property
    def skill_file(self) -> Path:
        return self.root / SKILL_FILE


def _front_matter(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return lines[1:index]
    return []


def _repository_root(folder: Path) -> Path | None:
    completed = subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        return None
    return Path(completed.stdout.strip()).resolve()


def resolve_external_skill(path: str | Path) -> ExternalSkill:
    """Read the skill folder at ``path``; ``ValueError`` when it holds no ``SKILL.md``."""
    root = Path(path).expanduser().resolve()
    skill_file = root / SKILL_FILE
    if not skill_file.is_file():
        raise ValueError(f"{root} holds no {SKILL_FILE}")
    front = _front_matter(skill_file.read_text(encoding="utf-8", errors="replace"))
    return ExternalSkill(
        root=root,
        name=_frontmatter_field(front, "name") or root.name,
        description=_clip(_frontmatter_field(front, "description")),
        repository_root=_repository_root(root),
    )
