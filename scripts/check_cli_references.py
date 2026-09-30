#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reject commands in agent-facing docs that name a module, subcommand or script that does not exist.

SKILL.md files, agent action docs, orchestrator prompts and the optimizer references are read by LLM
agents, which run the commands in them verbatim. A renamed module there fails only at runtime, inside
an agent turn. Every ``<python> -m <module>`` must be a hyperloom/kernelforge module in the source
tree with a ``__main__`` entry (``hyperloom`` itself with a known command) or an allowed external module, and every
``src/<package>/...py`` script path must exist. Nothing is imported from the checked tree.

Usage:
    python scripts/check_cli_references.py [REPO_ROOT]

Exit code 1 when a reference does not resolve.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_THIS_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_REPO / "src"))

from hyperloom import cli as hyperloom_cli  # noqa: E402

_DOC_GLOBS = (
    "**/SKILL.md",
    "src/hyperloom/agents/**/actions/*.md",
    "src/hyperloom/orchestrator/prompts/**/*.md",
    "src/hyperloom/orchestrator/prompts/**/*.py",
    "src/hyperloom/inference_optimizer/references/**/*.md",
)
# ``-m`` after an interpreter token (python3, /venv/bin/python, "$PYTHON", ${PYTHON}); ``git commit -m`` is not.
_MODULE_RE = re.compile(r"(?:python[\d.]*|PYTHON\w*\}?)\"?\s+-m\s+([\w.]+)(?:\s+([a-z][\w-]*))?(?:\s+([a-z][\w-]*))?")
_SCRIPT_RE = re.compile(r"\bsrc/((?:hyperloom|kernelforge)/[\w/.-]+\.py)\b")
_MAIN_GUARD_RE = re.compile(r"^if __name__ == ['\"]__main__['\"]:", re.MULTILINE)
_EXTERNAL_MODULES = frozenset({"pip"})
_SKIPPED_TOP_DIRS = frozenset({".git", ".venv", "build", "node_modules"})


def _doc_files(root: Path) -> list[Path]:
    found = {
        p for pattern in _DOC_GLOBS for p in root.glob(pattern) if p.relative_to(root).parts[0] not in _SKIPPED_TOP_DIRS
    }
    return sorted(found)


def _module_runnable(root: Path, module: str) -> bool:
    base = root / "src" / Path(*module.split("."))
    if (base / "__main__.py").is_file():
        return True
    source = base.with_suffix(".py")
    return source.is_file() and _MAIN_GUARD_RE.search(source.read_text(encoding="utf-8", errors="replace")) is not None


def _check_module(root: Path, module: str, first: str | None, second: str | None) -> str | None:
    if module.split(".")[0] not in ("hyperloom", "kernelforge"):
        return None if module in _EXTERNAL_MODULES else f"module {module} is not a hyperloom/kernelforge module"
    if module != "hyperloom":
        return None if _module_runnable(root, module) else f"module {module} is not runnable with -m"
    if first is None or first in hyperloom_cli._COMMANDS:
        return None
    if first != "session":
        return f"`hyperloom {first}` is not a hyperloom command"
    if second is None or second in hyperloom_cli._SESSION_COMMANDS:
        return None
    return f"`hyperloom session {second}` is not a session command"


def check(root: Path) -> list[str]:
    """Return ``path:line: message`` for every reference that does not resolve under ``root``."""
    problems: list[str] = []
    for path in _doc_files(root):
        rel = path.relative_to(root)
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for match in _MODULE_RE.finditer(line):
                message = _check_module(root, *match.groups())
                if message:
                    problems.append(f"{rel}:{lineno}: {message}")
            for match in _SCRIPT_RE.finditer(line):
                if not (root / "src" / match.group(1)).is_file():
                    problems.append(f"{rel}:{lineno}: script src/{match.group(1)} does not exist")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]).resolve() if args else _THIS_REPO
    problems = check(root)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
