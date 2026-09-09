# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard the moves that folded everything into a single ``kernelforge`` package."""

from __future__ import annotations

import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

_PATTERN = re.compile(r"kernel_agents|kernel-agents|KERNEL_AGENTS")

# The second move: the two sibling top-level packages became subpackages, so ``forge_llm`` -> ``kernelforge.llm``,
# ``forge_llm.agent_backends`` -> ``kernelforge.agent_backends``, ``forge_gemm_tune`` -> ``kernelforge.gemm_tune``.
_COLLAPSE_PATTERN = re.compile(r"\bforge_llm\b|\bforge_gemm_tune\b")

_COLLAPSE_ALLOWED: tuple[tuple[str, str, str], ...] = (
    (
        "CHANGELOG.md",
        r"forge_llm|forge_gemm_tune",
        "Release notes recording what the packages used to be called. An entry "
        "that gets renamed stops telling the reader which spelling to migrate from.",
    ),
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
        "CHANGELOG.md",
        r"kernel_agents|kernel-agents",
        "Historical release notes.",
    ),
)


# The third move: ``fellow`` -> ``kernel_backend``.
_FELLOW_PATTERN = re.compile(r"fellow", re.IGNORECASE)

# The back-compat shims that used to be exempt here are gone: the old spelling is no longer accepted anywhere in code,
# so nothing outside a historical record may name it.
_FELLOW_ALLOWED: tuple[tuple[str, str, str], ...] = (
    (
        "src/kernelforge/data/*.md",
        r"(?i)fellow",
        "Knowledge-base records of campaigns that really did run under the old "
        "vocabulary. The P2 rule stands: paths and commands may be renamed, the "
        "narrative may not, because rewriting it falsifies the record. Scoped to "
        "*.md for the same reason its kernel_agents sibling is: a data/* glob also "
        "swallowed examples/*/run_example.sh, seven of which kept passing a "
        "--fellow flag the CLI no longer declares. forge-loop tolerated unknown "
        "options at the time, so those runs did not fail -- they silently ran an "
        "inferred backend instead of the intended one. That tolerance is gone: an "
        "undeclared option is now an exit code, which is what makes this scope safe.",
    ),
    (
        "src/kernelforge/tests/test_rename_completeness.py",
        r".",
        "This file names the old spelling in order to forbid it.",
    ),
    (
        "CHANGELOG.md",
        r"(?i)fellow",
        "Historical release notes. An entry that gets renamed stops telling the reader which spelling to migrate from.",
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


def test_every_allowlist_entry_still_exempts_something() -> None:
    """An exemption that matches nothing is a hole nobody is watching."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")
    dead: list[str] = []
    for label, allowed, pattern in (
        ("_ALLOWED", _ALLOWED, _PATTERN),
        ("_COLLAPSE_ALLOWED", _COLLAPSE_ALLOWED, _COLLAPSE_PATTERN),
        ("_FELLOW_ALLOWED", _FELLOW_ALLOWED, _FELLOW_PATTERN),
    ):
        hits = _tracked_hits(root, pattern)
        for glob, line_re, _why in allowed:
            if not any(
                (glob == "*" or fnmatchcase(rel, glob)) and re.search(line_re, line) for rel, _lineno, line in hits
            ):
                dead.append(f"{label}: {glob}  /{line_re}/")
    assert not dead, "allowlist entries that exempt nothing -- delete them:\n  " + "\n  ".join(dead)


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


def test_no_stray_fellow_references() -> None:
    """``fellow`` survives only as a deliberate back-compat literal or a record."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")
    stray = [
        f"{rel}:{lineno}: {line}"
        for rel, lineno, line in _tracked_hits(root, _FELLOW_PATTERN)
        if not _is_allowed(rel, line, _FELLOW_ALLOWED)
    ]
    assert not stray, (
        "unrenamed 'fellow' references; rename them to kernel_backend, or add a "
        "justified entry to _FELLOW_ALLOWED:\n  " + "\n  ".join(stray[:40])
    )


def test_the_rename_did_not_space_out_an_unrelated_identifier() -> None:
    """``kernel_backend`` must never appear quoted with a space instead."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a source checkout")
    stray = [
        f"{rel}:{lineno}: {line}" for rel, lineno, line in _tracked_hits(root, re.compile(r'["\']kernel backend["\']'))
    ]
    assert not stray, (
        "a quoted two-word spelling of kernel_backend: the identifier lost its "
        "underscore to the rename and must get it back:\n  " + "\n  ".join(stray[:40])
    )


def test_kernel_backend_prompt_modules_are_importable() -> None:
    """The dotted prompt-module registry is strings; only an import proves it."""
    import importlib

    from kernelforge.kernel_backends.constants import KERNEL_BACKEND_PROMPT_MODULES

    for backend, module in KERNEL_BACKEND_PROMPT_MODULES.items():
        assert module.startswith("kernelforge."), backend
        importlib.import_module(module)
