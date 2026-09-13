# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Every declared CLI option must appear in the CLI reference, under its own command."""

from __future__ import annotations

import re
from pathlib import Path

import click

from kernelforge.cli import main

DOC_RELPATH = "docs/kernelforge/reference/cli.md"
ROOT_NAME = "kernelforge"


def _repo_root() -> Path | None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _every_command(group: click.Group, prefix: str = ""):
    """Yield (qualified name, command) for every command, groups included."""
    yield prefix.strip() or ROOT_NAME, group
    for name, cmd in group.commands.items():
        if isinstance(cmd, click.Group):
            yield from _every_command(cmd, f"{prefix}{name} ")
        else:
            yield f"{prefix}{name}", cmd


def _sections(text: str) -> dict[str, str]:
    """Map each markdown heading to its body, a subheading's body included."""
    lines = text.splitlines()
    heads: list[tuple[int, int, str]] = []  # (line index, level, title)
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if match:
            heads.append((index, len(match.group(1)), match.group(2)))

    # The page title is a heading too, so the intro runs to the SECOND one.
    intro_end = heads[1][0] if len(heads) > 1 else len(lines)
    out = {ROOT_NAME: "\n".join(lines[:intro_end])}
    for position, (start, level, title) in enumerate(heads):
        end = len(lines)
        for later_start, later_level, _ in heads[position + 1 :]:
            if later_level <= level:
                end = later_start
                break
        out[title] = out.get(title, "") + "\n" + "\n".join(lines[start:end])
    return out


def _mentions(body: str, opt: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(opt)}(?![\w-])", body) is not None


def test_every_declared_option_is_documented_under_its_own_command():
    repo = _repo_root()
    if repo is None:
        return  # installed without the source tree; nothing to check against
    doc = repo / DOC_RELPATH
    assert doc.is_file(), f"{DOC_RELPATH} is missing"
    sections = _sections(doc.read_text(encoding="utf-8"))

    undocumented: list[str] = []
    for name, cmd in _every_command(main):
        body = sections.get(name)
        if body is None:
            undocumented.append(f"{name} (no section of its own)")
            continue
        for param in cmd.params:
            # secondary_opts (`--no-x`) ride along with their primary spelling.
            for opt in param.opts:
                if opt.startswith("--") and not _mentions(body, opt):
                    undocumented.append(f"{name} {opt}")
    assert not undocumented, f"missing from their own section of {DOC_RELPATH}: " + ", ".join(sorted(undocumented))
