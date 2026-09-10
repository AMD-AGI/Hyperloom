# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard the enablement relocation.

Two protections, both using the mechanism of
``kernelforge/tests/test_rename_completeness.py``: a ``git ls-files`` sweep, an
allowlist whose every entry carries a written justification, and a self-validating
test that fails when an allowlist entry stops exempting anything.

1. **Old module paths must not creep back.** The relocation left no shims, so any
   reference to a pre-move dotted path is a regression.

2. **``agents/framework/*`` must not import ``orchestrator/*``.** This is the
   direction rule that would have caught the original ``enablement_ops`` reverse
   edge the day it landed. There are zero violations, so the allowlist is empty.
"""

from __future__ import annotations

import ast
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

_SELF = "src/hyperloom/inference_optimizer/tests/test_enablement_relocation_completeness.py"

# Modules that moved, by their pre-move dotted and path spellings.
_OLD_PATH_PATTERN = re.compile(
    r"agents\.framework\.enablement_ops"
    r"|agents/framework/enablement_ops"
    r"|agents\.framework\.enablement\b"
    r"|agents/framework/enablement\.py"
    r"|phases\._enablement_artifacts"
    r"|phases/_enablement_artifacts"
    r"|orchestrator\.framework\.(adapters|stack_actions|localization|build_actions|build_utils|targeted_build|client)"
    r"|orchestrator/framework/(adapters|stack_actions|localization|build_actions|build_utils|targeted_build|client)\.py"
)

_OLD_PATH_ALLOWED: tuple[tuple[str, str, str], ...] = (
    (
        _SELF,
        r".",
        "This file names the old spellings in order to forbid them.",
    ),
    (
        "adjustment.md",
        r".",
        "The relocation log records what moved and from where; rewriting it would "
        "stop telling the reader which spelling to migrate from.",
    ),
    (
        "src/hyperloom/orchestrator/enablement/mandate.py",
        r"Moved from",
        "The module docstring names its own former location so a reader landing here "
        "from an old traceback knows they are in the right place.",
    ),
    (
        "src/hyperloom/common/failure_signature.py",
        r"Moved from|orchestrator\.framework\.adapters",
        "The module docstring names its former location and the two importers whose layering motivated the move.",
    ),
)

# Import-direction rule: nothing under agents/framework may reach into orchestrator.
_ORCHESTRATOR_IMPORT = re.compile(r"^hyperloom\.orchestrator\.")

_AGENTS_FW_ALLOWED: tuple[tuple[str, str, str], ...] = ()


def _repo_root() -> Path | None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _tracked_hits(root: Path, pattern: re.Pattern[str]) -> list[tuple[str, int, str]]:
    """Return ``(rel_path, lineno, line)`` for every pattern match in tracked files."""
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    hits: list[tuple[str, int, str]] = []
    for rel in files:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append((rel, lineno, line.strip()))
    return hits


def _is_allowed(rel: str, line: str, allowed: tuple[tuple[str, str, str], ...]) -> bool:
    for glob, line_re, _why in allowed:
        if (glob == "*" or fnmatchcase(rel, glob)) and re.search(line_re, line):
            return True
    return False


def _agents_fw_orchestrator_imports(root: Path) -> list[tuple[str, int, str]]:
    """Return every ``hyperloom.orchestrator.*`` import under ``agents/framework/``."""
    files = subprocess.run(
        ["git", "ls-files", "src/hyperloom/agents/framework/"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    hits: list[tuple[str, int, str]] = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        try:
            src = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if any(_ORCHESTRATOR_IMPORT.match(n) for n in names):
                hits.append((rel, node.lineno, lines[node.lineno - 1].strip()))
    return hits


def test_every_allowlist_entry_still_exempts_something() -> None:
    """An exemption that matches nothing is a hole nobody is watching."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")

    dead: list[str] = []
    old_hits = _tracked_hits(root, _OLD_PATH_PATTERN)
    for glob, line_re, _why in _OLD_PATH_ALLOWED:
        if not any((glob == "*" or fnmatchcase(rel, glob)) and re.search(line_re, line) for rel, _n, line in old_hits):
            dead.append(f"_OLD_PATH_ALLOWED: {glob!r}  /{line_re}/")

    fw_hits = _agents_fw_orchestrator_imports(root)
    for glob, line_re, _why in _AGENTS_FW_ALLOWED:
        if not any((glob == "*" or fnmatchcase(rel, glob)) and re.search(line_re, line) for rel, _n, line in fw_hits):
            dead.append(f"_AGENTS_FW_ALLOWED: {glob!r}  /{line_re}/")

    assert not dead, "allowlist entries that exempt nothing -- delete them:\n  " + "\n  ".join(dead)


def test_no_stray_pre_relocation_module_paths() -> None:
    """The relocation left no shims, so an old dotted path is a live regression."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")

    stray = [
        f"{rel}:{lineno}: {line}"
        for rel, lineno, line in _tracked_hits(root, _OLD_PATH_PATTERN)
        if not _is_allowed(rel, line, _OLD_PATH_ALLOWED)
    ]
    assert not stray, (
        "references to pre-relocation enablement module paths. Point them at the "
        "canonical location, or add a justified entry to _OLD_PATH_ALLOWED:\n  " + "\n  ".join(stray[:40])
    )


def test_no_agents_framework_imports_orchestrator() -> None:
    """``agents/framework`` must not import ``orchestrator`` -- the reverse-edge rule."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")

    stray = [
        f"{rel}:{lineno}: {line}"
        for rel, lineno, line in _agents_fw_orchestrator_imports(root)
        if not _is_allowed(rel, line, _AGENTS_FW_ALLOWED)
    ]
    assert not stray, (
        "agents/framework imports orchestrator, inverting the layering. Move the "
        "shared code to hyperloom.common, or add a justified entry to "
        "_AGENTS_FW_ALLOWED:\n  " + "\n  ".join(stray[:40])
    )


def test_relocated_modules_are_importable() -> None:
    """The relocation targets are dotted strings in registries; only an import proves them."""
    import importlib

    for module in (
        "hyperloom.common.failure_signature",
        "hyperloom.orchestrator.enablement.mandate",
        "hyperloom.orchestrator.enablement.artifacts",
        "hyperloom.orchestrator.framework.adapter_parsers",
        "hyperloom.orchestrator.enablement.runtime.adapters",
        "hyperloom.orchestrator.enablement.runtime.stack_actions",
        "hyperloom.orchestrator.enablement.runtime.localization",
        "hyperloom.orchestrator.enablement.runtime.build_actions",
        "hyperloom.orchestrator.enablement.runtime.build_utils",
        "hyperloom.orchestrator.enablement.runtime.targeted_build",
    ):
        importlib.import_module(module)


def test_targeted_build_spawn_path_matches_the_module() -> None:
    """``_driver_command`` spawns ``python -m <module>``; a stale string fails only at build time."""
    from hyperloom.orchestrator.enablement.runtime import targeted_build
    from hyperloom.orchestrator.enablement.runtime.build_actions import TargetedBuildAction
    from hyperloom.orchestrator.loop.build_lifecycle import _driver_command

    import tempfile

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="c")
    with tempfile.TemporaryDirectory() as tmp:
        argv = _driver_command(action, tmp)
    assert argv[1] == "-m"
    assert argv[2] == targeted_build.__name__
