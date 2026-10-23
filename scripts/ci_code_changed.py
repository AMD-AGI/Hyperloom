# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Say whether a push or pull request changed anything beyond documentation and repository metadata.

Tests with Coverage and Lint start on every event so that their required checks always report; their expensive jobs
then run only when this prints ``code=true``. The patterns are the canonical documentation-only list in
CONTRIBUTING.md, matched as GitHub path filters match them (``*`` stays within one directory, ``**`` crosses them).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable

DOCS_ONLY_PATTERNS = ("**/*.md", "docs/**", "LICENSE*", "COPYRIGHT", "**/CODEOWNERS", ".gitattributes")

# A push that creates a branch reports this as its ``before`` commit.
_NO_COMMIT = "0" * 40


def _glob_regex(pattern: str) -> re.Pattern[str]:
    if any(ch in pattern for ch in "?[]!"):
        raise ValueError(f"unsupported glob syntax in {pattern!r}: only '*' and '**' are translated")
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


_DOCS_ONLY = tuple(_glob_regex(pattern) for pattern in DOCS_ONLY_PATTERNS)


def is_docs_only(path: str) -> bool:
    return any(rx.match(path) for rx in _DOCS_ONLY)


def code_changed(paths: Iterable[str]) -> bool:
    """True unless every path is documentation-only; an empty change list counts as code."""
    paths = list(paths)
    return not paths or not all(is_docs_only(path) for path in paths)


def changed_paths(base: str, head: str) -> list[str]:
    # Without --no-renames a rename lists only its destination, so code moved into docs/ would read as docs-only.
    out = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", base, head], check=True, capture_output=True, text=True
    )
    return [line for line in out.stdout.splitlines() if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="commit the change is measured from; a push's all-zero before counts as code")
    parser.add_argument("head", help="commit the change is measured to")
    args = parser.parse_args(argv)
    changed = args.base == _NO_COMMIT or code_changed(changed_paths(args.base, args.head))
    print(f"code={'true' if changed else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
