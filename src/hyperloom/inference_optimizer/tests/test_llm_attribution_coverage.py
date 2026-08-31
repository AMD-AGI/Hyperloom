# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-tree guard that every LLM call reaches the gateway attributed.

``hyperloom.common.llm_attribution`` can render a header, but a call site that
names nobody drops out of gateway attribution entirely -- which is the
accounting gap the feature exists to close. That is a property of the call
sites, not of the module, so it is checked by statically parsing every
production file rather than by any one module's unit tests.

It lives here rather than beside the module because it reads the component
vocabulary from ``hyperloom.orchestrator``, and ``hyperloom.common`` may not
import a first-party package (see :mod:`test_common_import_lint`).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace.llm_trace import VALID_COMPONENTS

#: Entry points that only tag a call when the caller names a component, so an
#: untagged call site is spend the gateway cannot attribute to anything.
_TAGGED_ENTRY_POINTS = frozenset(
    {
        "achat_completion",
        "aanthropic_completion",
        "aanthropic_messages",
        "anthropic_completion",
        "anthropic_messages",
        "astream_chat_completion_text",
        "chat_completion",
        "claude_sdk_env_options",
        "CodexSession",
        "run_codex_turn",
        "stream_chat_completion_text",
    }
)

_SRC_ROOT = Path(__file__).resolve().parents[2]


def _iter_tagged_calls() -> Iterator[tuple[Path, ast.Call, str]]:
    """Yield every production call to a tagged entry point.

    Yields:
        The path relative to the source root, the call node, and the name it
        was called by. Tests are skipped: they call these entry points to
        exercise them, not to spend against the gateway.
    """
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(_SRC_ROOT)
        if any(part in {"tests", "test"} for part in relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not our files to fix
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _TAGGED_ENTRY_POINTS:
                yield relative, node, name


def _scan_call_sites(field: str) -> tuple[int, list[str]]:
    """Find production calls to a tagged entry point that omit ``field``.

    Args:
        field: The keyword every call site is required to pass.

    Returns:
        The number of call sites seen and one line per offender. The count is
        reported so the guard cannot quietly pass by finding nothing.
    """
    seen = 0
    offenders: list[str] = []
    for relative, node, name in _iter_tagged_calls():
        seen += 1
        if not any(keyword.arg == field for keyword in node.keywords):
            offenders.append(f"{relative}:{node.lineno}: {name}(...) has no {field}=")
    return seen, offenders


@pytest.mark.parametrize("field", ["component", "operation"])
def test_every_llm_entry_point_call_names_its_attribution(field: str) -> None:
    """Every production LLM call must say who is spending, and on what.

    ``component`` and ``operation`` are the two fields the call site alone
    knows -- the rest are filled in from the run's own state.
    """
    seen, offenders = _scan_call_sites(field)
    assert not offenders, f"LLM call sites with no {field}:\n" + "\n".join(offenders)
    # Guard the guard: a rename that emptied _TAGGED_ENTRY_POINTS would
    # otherwise turn this into a test that always passes.
    assert seen >= 15, f"only {seen} LLM call sites found; the scan is no longer finding them"


def test_component_labels_come_from_the_closed_vocabulary() -> None:
    """A misspelled component is worse than a missing one.

    A call site that names no component is caught above and reads as
    unattributed spend. One that misspells it reads as attributed, and quietly
    opens a rollup of its own next to the real component's -- so the gateway
    view and the ledger view of the same producer can no longer be joined.
    Only literals can be judged here; a call site forwarding a variable has no
    name for this to check, and is left to the ledger's own validation.
    """
    offenders: list[str] = []
    checked = 0
    for relative, node, name in _iter_tagged_calls():
        for keyword in node.keywords:
            if keyword.arg != "component" or not isinstance(keyword.value, ast.Constant):
                continue
            checked += 1
            if keyword.value.value not in VALID_COMPONENTS:
                offenders.append(f"{relative}:{node.lineno}: {name}(component={keyword.value.value!r})")
    assert not offenders, "LLM call sites naming an unknown component:\n" + "\n".join(offenders)
    assert checked >= 14, f"only {checked} literal components found; the scan is no longer finding them"
