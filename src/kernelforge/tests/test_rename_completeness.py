# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard the moves that folded everything into a single ``kernelforge`` package.

The rename was a bulk text substitution, and the sites it cannot break loudly
are the ones that matter: a module path inside a string, an entry-point group,
a dotted prompt-module registry. Those raise at call time -- often inside an
``except`` branch that silently substitutes a default -- rather than at import.

So this test does what the import graph cannot: it greps the tree and asserts
the surviving occurrences are exactly the ones we decided to keep. Anything
else is a missed rename.
"""

from __future__ import annotations

import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

_PATTERN = re.compile(r"kernel_agents|kernel-agents|KERNEL_AGENTS")

# The second move: the two sibling top-level packages became subpackages, so
# ``forge_llm`` -> ``kernelforge.llm``, ``forge_llm.agent_backends`` ->
# ``kernelforge.agent_backends``, ``forge_gemm_tune`` -> ``kernelforge.gemm_tune``.
# Word boundaries keep unrelated identifiers that merely contain the spelling
# (``resolve_forge_llm_model``, ``_forge_gemm_tune_available``) out of the sweep.
_COLLAPSE_PATTERN = re.compile(r"\bforge_llm\b|\bforge_gemm_tune\b")

_COLLAPSE_ALLOWED: tuple[tuple[str, str, str], ...] = (
    (
        "src/kernelforge/gemm_tune/tune_robustness.py",
        r"~/\.forge_gemm_tune/",
        "A user-home cache directory, not a module path. Renaming it would orphan "
        "every faulted-shape blocklist an operator has already accumulated.",
    ),
    (
        "src/kernelforge/tests/test_rename_completeness.py",
        r".",
        "This file names the old spellings in order to forbid them.",
    ),
)

# Occurrences that are deliberate. Each entry is (path glob, line regex, why).
_ALLOWED: tuple[tuple[str, str, str], ...] = (
    (
        "src/kernelforge/data/*",
        r"kernel-agents",
        "Knowledge-base prose describing historical campaigns -- ${KA_WORKSPACE}/"
        "kernel-agents-workspace/... paths that were real when those runs happened. "
        "Rewriting them would falsify the record.",
    ),
    (
        "*",
        r"KERNEL_AGENTS_MAX_TURNS",
        "Removed environment variable. The literal exists only so Config.from_env "
        "can warn the operator that it is ignored; renaming it silences the warning.",
    ),
    (
        "*",
        r"KERNEL_AGENTS_MODEL",
        "Legacy alias for FORGE_AGENT_MODEL, kept working on purpose. A "
        "back-compat alias that gets renamed is not a back-compat alias.",
    ),
    (
        "src/kernelforge/agent_backends/registry.py",
        r"kernel_agents\.agent_providers",
        "Pre-rename entry-point group, still read so third-party provider plugins "
        "keep loading (with a DeprecationWarning).",
    ),
    (
        "src/kernelforge/tests/test_rename_completeness.py",
        r".",
        "This file names the old spellings in order to forbid them.",
    ),
    (
        "src/kernelforge/tests/test_provider_registry.py",
        r"kernel_agents",
        "Coverage for the deprecated entry-point group's dual-read; the test has to name the group it is asserting on.",
    ),
    (
        "pyproject.toml",
        r"^(kernel-agents = |# Deprecated alias kept for one release)",
        "Deprecated console-script alias (and the comment above it), kept one release so existing scripts and shell history keep working.",
    ),
    (
        "CHANGELOG.md",
        r"kernel_agents|kernel-agents",
        "Historical release notes.",
    ),
)


def _repo_root() -> Path | None:
    root = Path(__file__).resolve()
    for parent in root.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _tracked_hits(root: Path, pattern: re.Pattern[str] = _PATTERN) -> list[tuple[str, int, str]]:
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    hits: list[tuple[str, int, str]] = []
    for rel in files:
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append((rel, lineno, line.strip()))
    return hits


def _is_allowed(rel: str, line: str, allowed: tuple[tuple[str, str, str], ...] = _ALLOWED) -> bool:
    for glob, line_re, _why in allowed:
        # fnmatch's ``*`` crosses "/", which is what we want for tree prefixes.
        if (glob == "*" or fnmatchcase(rel, glob)) and re.search(line_re, line):
            return True
    return False


def test_no_stray_kernel_agents_references() -> None:
    """Every surviving ``kernel_agents`` spelling must be one we chose to keep."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")
    stray = [f"{rel}:{lineno}: {line}" for rel, lineno, line in _tracked_hits(root) if not _is_allowed(rel, line)]
    assert not stray, (
        "unrenamed kernel_agents references; rename them, or add a justified entry "
        "to _ALLOWED:\n  " + "\n  ".join(stray[:40])
    )


def test_no_stray_standalone_package_references() -> None:
    """No path or dotted name may still point at the pre-collapse packages."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")
    stray = [
        f"{rel}:{lineno}: {line}"
        for rel, lineno, line in _tracked_hits(root, _COLLAPSE_PATTERN)
        if not _is_allowed(rel, line, _COLLAPSE_ALLOWED)
    ]
    assert not stray, (
        "references to forge_llm / forge_gemm_tune, which are now kernelforge "
        "subpackages:\n  " + "\n  ".join(stray[:40])
    )


def test_fellow_prompt_modules_are_importable() -> None:
    """The dotted prompt-module registry is strings; only an import proves it."""
    import importlib

    from kernelforge.fellows.constants import FELLOW_PROMPT_MODULES

    for backend, module in FELLOW_PROMPT_MODULES.items():
        assert module.startswith("kernelforge."), backend
        importlib.import_module(module)
