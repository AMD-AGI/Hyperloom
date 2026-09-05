# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""No budget or deadline may be tested for truth.

``if budget:`` and ``budget or default`` read a value of zero as absent. For a
duration that is the difference between "stop now" and "nobody said when", and
the layer that made that mistake replaced an exhausted budget with a ceiling of
its own, measured in days. The bug is not fixable by reviewing the one site: any
new ``if wall_budget:`` reintroduces it, silently, and only under the conditions
nobody tests -- a session that has already run out.

So the shape is banned rather than the instance. This walks the syntax tree of
the modules that carry time budgets and fails on any implicit truth test of a
name that holds one. Comparisons are always allowed: ``budget > 0`` says which
side of zero the author meant, and ``deadline is None`` says absent, and between
them there is nothing left for a falsy test to express.

Ruff has no rule for a truth test on a name matching a pattern and no way to
add one, so the ban is a test, alongside the other tree-shape rules under
``inference_optimizer/tests``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hyperloom

#: The layers a session's wall-clock budget crosses on its way from the
#: dispatcher's arithmetic to the reaper's kill, plus the session clock that
#: produces it. The rule is scoped here rather than repo-wide on purpose: this
#: is the path along which a duration is handed down layer by layer, and
#: therefore the only path on which one layer's falsy test can silently undo
#: the layer above. Patterns rather than filenames, so a module joining one of
#: these layers is covered the day it lands.
_GUARDED = (
    "common/deadline.py",
    "orchestrator/loop/*.py",
    "orchestrator/specialists/*.py",
    "orchestrator/state/shared_state.py",
    "orchestrator/enablement/*.py",
)

#: A name holding a duration or an instant is one this rule covers. Matched as
#: a substring of the identifier, lowercased, so ``wall_budget_sec``,
#: ``_closing_deadline`` and ``session_remaining_seconds`` are all in scope.
_BUDGET_MARKERS = (
    "budget",
    "deadline",
    "timeout",
    "remaining_sec",
    "remaining_seconds",
    "max_seconds",
    "grace_sec",
)

#: Names that carry a marker but are not durations: flags, mappings and text
#: whose truth is exactly what the author meant to test.
_EXEMPT_SUFFIXES = (
    "_exhausted",
    "_blown",
    "_enabled",
    "_disabled",
    "_known",
    "_reason",
    "_error",
    "_kind",
    "_state",
    "_extensions",
    "_timings_sec",
    "_pct",
    "_block",
)


def _is_budget_name(name: str) -> bool:
    """Whether ``name`` holds a duration or an instant this rule covers.

    Args:
        name: The identifier or attribute being tested for truth.

    Returns:
        bool: True when a falsy test on it would confuse zero with absent.
    """
    lowered = name.lower()
    if not any(marker in lowered for marker in _BUDGET_MARKERS):
        return False
    return not lowered.endswith(_EXEMPT_SUFFIXES)


def _tested_names(node: ast.AST) -> list[str]:
    """Collect the names a node tests for truth, recursing through and/or/not.

    Args:
        node: The expression in truth-testing position.

    Returns:
        list[str]: Identifiers whose truth is being taken.
    """
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.BoolOp):
        found: list[str] = []
        for value in node.values:
            found.extend(_tested_names(value))
        return found
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _tested_names(node.operand)
    return []


def _truth_tested(tree: ast.AST) -> list[tuple[int, str]]:
    """Find every implicit truth test of a budget-bearing name.

    Args:
        tree: A parsed module.

    Returns:
        list[tuple[int, str]]: ``(line, name)`` for each offending test.
    """
    offences: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            positions = [node.test]
        elif isinstance(node, ast.BoolOp):
            # ``budget or fallback`` is the assignment form of the same bug.
            positions = list(node.values)
        elif isinstance(node, ast.Assert):
            positions = [node.test]
        else:
            continue
        for position in positions:
            # A comparison states which side of zero it means, so its operands
            # are not truth-tested even when the comparison sits inside a BoolOp.
            if isinstance(position, ast.Compare):
                continue
            for name in _tested_names(position):
                if _is_budget_name(name):
                    offences.append((getattr(position, "lineno", 0), name))
    return offences


def _modules() -> list[Path]:
    """Resolve the guarded patterns, failing loudly if a layer has moved.

    Returns:
        list[Path]: Files to scan.

    Raises:
        AssertionError: When a pattern matches nothing, which means the rule
            has stopped covering the layer it names.
    """
    root = Path(hyperloom.__file__).parent
    found: dict[Path, None] = {}
    unmatched: list[str] = []
    for pattern in _GUARDED:
        matched = sorted(root.glob(pattern))
        if not matched:
            unmatched.append(pattern)
        found.update(dict.fromkeys(matched))
    assert not unmatched, f"guarded layers have moved and the rule no longer covers them: {unmatched}"
    return list(found)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("if wall_budget_sec:\n    pass\n", True),
        ("if wall_budget_sec and wall_budget_sec > 0:\n    pass\n", True),
        ("x = wall_budget_sec or 600.0\n", True),
        ("if not deadline:\n    pass\n", True),
        ("if self._run_deadline:\n    pass\n", True),
        ("if wall_budget_sec > 0:\n    pass\n", False),
        ("if deadline is None:\n    pass\n", False),
        ("if budget_exhausted:\n    pass\n", False),
        ("if name:\n    pass\n", False),
    ],
)
def test_the_rule_recognises_the_shape_it_bans(source, expected):
    assert bool(_truth_tested(ast.parse(source))) is expected


def test_no_budget_is_tested_for_truth():
    offences: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, name in _truth_tested(tree):
            offences.append(f"{path}:{line}: `{name}` tested for truth")
    assert not offences, (
        "a budget or deadline tested for truth reads zero as absent, which is how "
        "an exhausted budget removes its own timeout:\n  " + "\n  ".join(offences)
    )
