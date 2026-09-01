# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Non-blocking drift check for AI instruction files.

AI instruction files (``AGENTS.md``, ``.github/copilot-instructions.md``, and any
``.github/instructions/*.instructions.md``) reference concrete repo paths and
``python -m`` module targets. As the tree evolves those references go stale and an
agent then runs the wrong command or looks in the wrong directory. This script
extracts the referenced paths/modules from **code spans only** (fenced blocks and
inline back-ticked tokens, never prose) and reports the ones that no longer resolve.

Advisory by design: it prints a report and exits ``0`` unless invoked with
``--strict``. The CI job runs it non-blocking (``continue-on-error``), so a stale
reference annotates the PR without gating merge.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

INSTRUCTION_FILES: tuple[str, ...] = (
    "AGENTS.md",
    ".github/copilot-instructions.md",
)
INSTRUCTION_GLOBS: tuple[str, ...] = (".github/instructions/*.instructions.md",)

# Inline code spans: `like this`. Fenced blocks are folded into the same stream by
# stripping the ``` fences first, so their content is also scanned token-by-token.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_FENCE_RE = re.compile(r"^```.*$", re.MULTILINE)

# A token is only treated as a repo-path reference when its FIRST component is an
# actual top-level entry of the repo (computed below). This keeps the check to
# fully-qualified paths that are meant to resolve exactly (e.g.
# ``src/hyperloom/.../paths.py``, ``.github/instructions/``) and deliberately skips
# in-prose shorthand (``framework/paths.py``) and absolute container-mount examples
# (``/app/xDiT/``), which are illustrative, not repo references.
_PATH_TOKEN_RE = re.compile(r"^[\w][\w./-]*$")

# ``python -m package.module`` targets.
_PY_M_RE = re.compile(r"python3?\s+-m\s+([\w.]+)")


def _repo_top_level() -> frozenset[str]:
    return frozenset(p.name for p in REPO_ROOT.iterdir())


def _iter_instruction_files() -> list[Path]:
    files: list[Path] = []
    for rel in INSTRUCTION_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            files.append(p)
    for pattern in INSTRUCTION_GLOBS:
        files.extend(sorted(REPO_ROOT.glob(pattern)))
    return files


def _code_spans(text: str) -> list[str]:
    """Return every code span: fenced-block bodies plus inline back-ticked tokens."""
    spans: list[str] = []
    # Fenced blocks: everything between a pair of ``` lines.
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            spans.append(line)
    # Inline code (only outside fences would be ideal, but double-counting is harmless
    # because results are de-duplicated by the caller).
    spans.extend(m.group(1) for m in _INLINE_CODE_RE.finditer(text))
    return spans


def _looks_like_path(token: str, top_level: frozenset[str]) -> bool:
    # Must be multi-component (has a slash) and rooted at a real top-level entry, so
    # only fully-qualified, meant-to-be-exact references are checked.
    if "/" not in token or not _PATH_TOKEN_RE.match(token):
        return False
    first = token.split("/", 1)[0]
    return first in top_level


def _module_to_path(module: str) -> Path | None:
    """Resolve ``a.b.c`` under ``src/`` to a file or package dir, if present."""
    base = REPO_ROOT / "src" / Path(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base
    return None


def check() -> list[tuple[str, str]]:
    """Return a list of ``(source_file, stale_reference)`` pairs."""
    stale: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    top_level = _repo_top_level()
    for f in _iter_instruction_files():
        rel_src = str(f.relative_to(REPO_ROOT))
        text = f.read_text(encoding="utf-8")
        tokens: set[str] = set()
        for span in _code_spans(text):
            tokens.update(span.split())
        for raw in tokens:
            # Trim trailing punctuation only; keep a leading dot (``.github``).
            token = raw.rstrip(".,:;()[]{}\"'").lstrip(",:;()[]{}\"'")
            if not token:
                continue
            if _looks_like_path(token, top_level):
                probe = token.rstrip("/")  # directory references carry a trailing slash
                if not (REPO_ROOT / probe).exists():
                    key = (rel_src, token)
                    if key not in seen:
                        seen.add(key)
                        stale.append(key)
        for m in _PY_M_RE.finditer(text):
            module = m.group(1)
            if _module_to_path(module) is None:
                key = (rel_src, f"python -m {module}")
                if key not in seen:
                    seen.add(key)
                    stale.append(key)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when stale references are found (default: advisory, exit 0)",
    )
    args = parser.parse_args()

    stale = check()
    if not stale:
        print("instruction-drift: no stale path/module references found.")
        return 0

    print("instruction-drift: stale references (verify or update):")
    for src, ref in stale:
        print(f"  ::warning file={src}::stale reference '{ref}' does not resolve in the tree")
    print(f"\n{len(stale)} stale reference(s) across AI instruction files.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
