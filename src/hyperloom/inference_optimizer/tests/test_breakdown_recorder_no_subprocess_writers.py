# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The breakdown recorder has exactly one writer: the coordinator process."""

from __future__ import annotations

import ast
from pathlib import Path

import hyperloom.agents
import hyperloom.inference_optimizer.multi_node
import hyperloom.orchestrator.supervisor

# Matches the recorder package and its ``instrument`` facade, which is how a caller would normally reach it.
_FORBIDDEN_FRAGMENT = "breakdown.recorder"


def _subprocess_package_roots() -> list[Path]:
    """Return the package directories that must not write breakdown fragments."""
    return [
        Path(hyperloom.agents.__file__).resolve().parent,
        Path(hyperloom.inference_optimizer.multi_node.__file__).resolve().parent,
        Path(hyperloom.orchestrator.supervisor.__file__).resolve().parent,
    ]


def _imports_recorder(path: Path) -> bool:
    """Report whether ``path`` imports the breakdown recorder."""
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
            module = node.module or ""
            if _FORBIDDEN_FRAGMENT in module:
                return True
            # ``from ...breakdown import recorder`` names the package in the alias rather than the module path, and
            # reaches the same writers.
            if module.endswith("breakdown") and any(alias.name == "recorder" for alias in node.names):
                return True
    return False


def test_subprocess_packages_do_not_write_breakdown_fragments() -> None:
    """No agent, multi-node or supervisor module may import the recorder."""
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
    reaching = [
        "from hyperloom.inference_optimizer.breakdown.recorder import instrument",
        "from hyperloom.inference_optimizer.breakdown.recorder.recorder import Recorder",
        "import hyperloom.inference_optimizer.breakdown.recorder as r",
        "from hyperloom.inference_optimizer.breakdown import recorder",
        "from hyperloom.inference_optimizer.breakdown import recorder as r",
        "from ...inference_optimizer.breakdown import recorder",
        "from ...breakdown.recorder import instrument",
    ]
    for index, source in enumerate(reaching):
        path = tmp_path / f"reaching_{index}.py"
        path.write_text(source + "\n", encoding="utf-8")
        assert _imports_recorder(path), f"guard missed a real writer: {source}"

    # Neighbouring modules under the same package must not trip it.
    innocent = [
        "from hyperloom.common import io",
        "from hyperloom.inference_optimizer.breakdown import exporter",
        "from hyperloom.inference_optimizer.breakdown import schema",
    ]
    for index, source in enumerate(innocent):
        path = tmp_path / f"innocent_{index}.py"
        path.write_text(source + "\n", encoding="utf-8")
        assert not _imports_recorder(path), f"guard false-positived on: {source}"
