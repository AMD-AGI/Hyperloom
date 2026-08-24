# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The breakdown recorder has exactly one writer: the coordinator process.

``Recorder.record_upsert_singleton`` / ``record_upsert_item`` read the current
fragment, merge, and rewrite it while holding an in-process lock only. Two
processes upserting the same (section, producer, key) would lose one side of
the merge, and a file lock cannot fix it here -- the spool lives on a network
filesystem, where advisory locks are unreliable.

The recorder module docstring therefore promises that upserts stay inside one
process. Agent packages and the multi-node helpers run as subprocesses, so this
test fails the moment one of them starts recording fragments, before the
promise silently becomes false.
"""

from __future__ import annotations

import ast
from pathlib import Path

import hyperloom.agents
import hyperloom.inference_optimizer.multi_node

# Matches the recorder package and its ``instrument`` facade, which is how a
# caller would normally reach it.
_FORBIDDEN_FRAGMENT = "breakdown.recorder"


def _subprocess_package_roots() -> list[Path]:
    """Return the package directories that must not write breakdown fragments.

    Returns:
        list[Path]: Absolute roots of the agent and multi-node packages.
    """
    return [
        Path(hyperloom.agents.__file__).resolve().parent,
        Path(hyperloom.inference_optimizer.multi_node.__file__).resolve().parent,
    ]


def _imports_recorder(path: Path) -> bool:
    """Report whether ``path`` imports the breakdown recorder.

    Relative imports are matched on their module suffix: a relative
    ``from ....breakdown.recorder import x`` still carries the fragment.

    Args:
        path (Path): The Python source file to inspect.

    Returns:
        bool: ``True`` when the file imports the recorder in any form.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        # Not our contract to police; the syntax checkers own this.
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_FORBIDDEN_FRAGMENT in alias.name for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if _FORBIDDEN_FRAGMENT in (node.module or ""):
                return True
    return False


def test_subprocess_packages_do_not_write_breakdown_fragments() -> None:
    """No agent or multi-node module may import the breakdown recorder."""
    offenders: list[str] = []
    for root in _subprocess_package_roots():
        for path in sorted(root.rglob("*.py")):
            # Tests may import anything; they do not run as the subprocesses.
            if "tests" in path.parts:
                continue
            if _imports_recorder(path):
                offenders.append(str(path))

    assert not offenders, (
        "breakdown recorder writes must stay in the coordinator process -- "
        "record_upsert_* is not safe across processes (see the recorder module "
        f"docstring). Offending modules: {offenders}"
    )


def test_the_detector_actually_detects(tmp_path: Path) -> None:
    """Guard the guard: a rule that cannot fail protects nothing."""
    absolute = tmp_path / "absolute.py"
    absolute.write_text(
        "from hyperloom.inference_optimizer.breakdown.recorder import instrument\n",
        encoding="utf-8",
    )
    relative = tmp_path / "relative.py"
    relative.write_text(
        "from ...inference_optimizer.breakdown.recorder.recorder import Recorder\n",
        encoding="utf-8",
    )
    plain_import = tmp_path / "plain.py"
    plain_import.write_text(
        "import hyperloom.inference_optimizer.breakdown.recorder as r\n",
        encoding="utf-8",
    )
    innocent = tmp_path / "innocent.py"
    innocent.write_text("from hyperloom.common import io\n", encoding="utf-8")

    assert _imports_recorder(absolute)
    assert _imports_recorder(relative)
    assert _imports_recorder(plain_import)
    assert not _imports_recorder(innocent)
