# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent skills that live outside the packaged knowledge tree, such as the one a kernel repository keeps."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def skill_files_opened(
    tool_calls: Iterable[tuple[str, Mapping[str, Any]]], skills: Iterable[ExternalSkill]
) -> list[str]:
    """The files under ``skills`` a session's tool calls named, as ``<skill>/<path>``, each once, first mention first.

    A read through the shell (``cat``, ``sed``, ``grep``) names its file in the command, so every argument counts.
    So does a relative path with a folder in it (``gemm/x.md``, ``.claude/skills/<skill>/gemm/x.md``) that is a file
    of the skill: an agent that has changed into the skill's folder, or works from its repository, reads that way.
    """
    opened: dict[str, None] = {}
    for _tool, arguments in tool_calls:
        text = " ".join(str(value) for value in (arguments or {}).values())
        found: list[tuple[int, str]] = []
        for skill in skills:
            for match in re.finditer(re.escape(str(skill.root)) + r"/([\w./+-]*\w\.\w+)", text):
                found.append((match.start(), f"{skill.name}/{match.group(1)}"))
            for match in _RELATIVE_FILE.finditer(text):
                inside = _inside(skill.root, match.group(0))
                if inside:
                    found.append((match.start(), f"{skill.name}/{inside}"))
        for _position, name in sorted(found):
            opened.setdefault(name, None)
    return list(opened)


# A relative path with at least one folder: not preceded by a path character, so never the tail of an absolute one.
_RELATIVE_FILE = re.compile(r"(?<![\w./+-])[\w.+-]+(?:/[\w.+-]+)+\.\w+")


def _inside(root: Path, path: str) -> str:
    """``path`` as a file of the skill at ``root``, relative to it, or empty when it names none."""
    marker = f"{root.name}/"
    for candidate in (path, path.rsplit(marker, 1)[1] if marker in path else ""):
        if candidate and (root / candidate).is_file() and (root / candidate).resolve().is_relative_to(root.resolve()):
            return candidate
    return ""
