# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Every declared CLI option must appear in the CLI reference doc.

The reference is the only place a caller can learn what an option is called,
and since an undeclared option now costs an exit code rather than a warning
(see ``test_cli_option_wiring``), a flag missing from the doc is a flag nobody
outside this file can discover. A doc written once drifts the moment the next
option lands, so this pins the two together instead of trusting a review to
notice.

The check is one-directional on purpose: it demands every declared option be
documented, but lets the doc mention flags this CLI does not own -- the
``--attention-backend`` inside a ``--bench-extra`` example belongs to
``bench_one_batch``, and the doc has to be able to show it.
"""

from __future__ import annotations

import re
from pathlib import Path

import click

from kernelforge.cli import main

DOC_RELPATH = "docs/kernelforge/reference/cli.md"


def _repo_root() -> Path | None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _walk(group: click.Group, prefix: str = ""):
    """Yield (qualified name, command) for every leaf command under a group."""
    for name, cmd in group.commands.items():
        if isinstance(cmd, click.Group):
            yield from _walk(cmd, f"{prefix}{name} ")
        else:
            yield f"{prefix}{name}", cmd


def _documented(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])--[A-Za-z0-9][A-Za-z0-9-]*", text))


def test_every_declared_option_is_documented():
    repo = _repo_root()
    if repo is None:
        return  # installed without the source tree; nothing to check against
    doc = repo / DOC_RELPATH
    assert doc.is_file(), f"{DOC_RELPATH} is missing"
    documented = _documented(doc.read_text(encoding="utf-8"))

    undocumented: list[str] = []
    for name, cmd in _walk(main):
        for param in cmd.params:
            # secondary_opts (`--no-x`) ride along with their primary spelling.
            for opt in param.opts:
                if opt.startswith("--") and opt not in documented:
                    undocumented.append(f"{name} {opt}")
    assert not undocumented, "options missing from " + DOC_RELPATH + ": " + ", ".join(sorted(undocumented))


def test_every_leaf_command_is_documented():
    repo = _repo_root()
    if repo is None:
        return
    text = (repo / DOC_RELPATH).read_text(encoding="utf-8")
    missing = [name for name, _ in _walk(main) if name.split()[-1] not in text]
    assert not missing, "commands missing from " + DOC_RELPATH + ": " + ", ".join(missing)
