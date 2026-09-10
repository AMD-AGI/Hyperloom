# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard the enablement module layout.

Two rules, swept over git-tracked files: no pre-move module path may return, and
``agents/framework`` may not import ``orchestrator``. The second is the direction
rule the ``enablement_ops`` reverse edge escaped.
"""

from __future__ import annotations

import ast
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

_SELF = "src/hyperloom/inference_optimizer/tests/test_enablement_relocation_completeness.py"

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

#: ``(path glob, line regex, why)``. An entry that stops matching is a dead
#: exemption and fails :func:`test_every_allowlist_entry_still_exempts_something`.
_OLD_PATH_ALLOWED: tuple[tuple[str, str, str], ...] = (
    (_SELF, r".", "Names the old spellings in order to forbid them."),
    ("adjustment.md", r".", "Records which spelling to migrate from."),
)

_ORCHESTRATOR_IMPORT = re.compile(r"^hyperloom\.orchestrator\.")


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
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


def _is_allowed(rel: str, line: str) -> bool:
    return any(fnmatchcase(rel, glob) and re.search(line_re, line) for glob, line_re, _why in _OLD_PATH_ALLOWED)


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
    for rel in (f for f in files if f.endswith(".py")):
        src = (root / rel).read_text(encoding="utf-8")
        lines = src.splitlines()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            if any(_ORCHESTRATOR_IMPORT.match(n) for n in names):
                hits.append((rel, node.lineno, lines[node.lineno - 1].strip()))
    return hits


def test_every_allowlist_entry_still_exempts_something() -> None:
    """An exemption that matches nothing is a hole nobody is watching."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")

    hits = _tracked_hits(root, _OLD_PATH_PATTERN)
    dead = [
        f"{glob!r}  /{line_re}/"
        for glob, line_re, _why in _OLD_PATH_ALLOWED
        if not any(fnmatchcase(rel, glob) and re.search(line_re, line) for rel, _n, line in hits)
    ]
    assert not dead, "allowlist entries that exempt nothing -- delete them:\n  " + "\n  ".join(dead)


def test_no_stray_pre_relocation_module_paths() -> None:
    """The move left no shims, so an old dotted path resolves to nothing."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")

    stray = [
        f"{rel}:{lineno}: {line}"
        for rel, lineno, line in _tracked_hits(root, _OLD_PATH_PATTERN)
        if not _is_allowed(rel, line)
    ]
    assert not stray, "references to moved modules:\n  " + "\n  ".join(stray[:40])


def test_no_agents_framework_imports_orchestrator() -> None:
    """``agents/framework`` importing ``orchestrator`` inverts the layering."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")

    stray = [f"{rel}:{lineno}: {line}" for rel, lineno, line in _agents_fw_orchestrator_imports(root)]
    assert not stray, "move the shared code to hyperloom.common instead:\n  " + "\n  ".join(stray[:40])


def test_relocated_modules_are_importable() -> None:
    """The move targets are dotted strings in registries; only an import proves them."""
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
    """``_driver_command`` spawns ``python -m``; a stale path fails only at build time."""
    import tempfile

    from hyperloom.orchestrator.enablement.runtime import targeted_build
    from hyperloom.orchestrator.enablement.runtime.build_actions import TargetedBuildAction
    from hyperloom.orchestrator.loop.build_lifecycle import _driver_command

    action = TargetedBuildAction(gap_id="g", framework="vllm", component="aiter", capability="c")
    with tempfile.TemporaryDirectory() as tmp:
        argv = _driver_command(action, tmp)
    assert argv[1] == "-m"
    assert argv[2] == targeted_build.__name__
