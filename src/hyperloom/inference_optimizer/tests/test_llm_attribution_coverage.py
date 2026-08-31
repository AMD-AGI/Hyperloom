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
#: ``inject_env`` tags a child process rather than one call, but it names a
#: component the same way and a mislabelled one costs more: it stands for every
#: call the child goes on to make.
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
        "inject_env",
        "run_codex_turn",
        "stream_chat_completion_text",
    }
)

#: Backends that carry their label on a dataclass field instead of passing it at
#: the call site, so the call-site scan below cannot see it.
_COMPONENT_FIELD = "attribution_component"

#: Every first-party package, not just ``hyperloom``: the forge loop spends from
#: ``kernelforge``, so scoping the scan to one package would exempt the tree
#: where a rewrite campaign's whole bill is produced.
_SRC_ROOT = Path(__file__).resolve().parents[3]


def _iter_production_trees() -> Iterator[tuple[Path, ast.AST]]:
    """Yield the parsed tree of every production file under the source root.

    Yields:
        The path relative to the source root and its parsed module. Tests are
        skipped: they name components to exercise them, not to spend.
    """
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(_SRC_ROOT)
        if any(part in {"tests", "test"} for part in relative.parts):
            continue
        try:
            yield relative, ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not our files to fix
            continue


def _imported_under(tree: ast.AST) -> dict[str, str]:
    """Map each aliased import back to the name it was imported under.

    Half the spawn boundaries import ``inject_env`` under a local alias, so
    matching a call by the name written at the call site would exempt exactly
    the sites where a whole child process's spend is labelled.

    Args:
        tree: A parsed module.

    Returns:
        Local name to original name, for the imports that renamed something.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for entry in node.names:
                if entry.asname:
                    aliases[entry.asname] = entry.name.rsplit(".", 1)[-1]
    return aliases


def _iter_tagged_calls() -> Iterator[tuple[Path, ast.Call, str]]:
    """Yield every production call to a tagged entry point.

    Yields:
        The path relative to the source root, the call node, and the name it
        was imported under. Tests are skipped: they call these entry points to
        exercise them, not to spend against the gateway.
    """
    for relative, tree in _iter_production_trees():
        aliases = _imported_under(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            canonical = aliases.get(name, name)
            if canonical in _TAGGED_ENTRY_POINTS:
                yield relative, node, canonical


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


def _iter_component_field_labels() -> Iterator[tuple[Path, int, str]]:
    """Yield every literal value production code gives ``attribution_component``.

    One backend class serves several roles, so it takes its label as a field
    rather than hardcoding one. That moves the label off the call site and out
    of reach of :func:`_iter_tagged_calls`, which only sees keywords passed to
    an entry point -- so the vocabulary has to be checked where it is now set:
    the field's default, and the keyword each role overrides it with.

    Yields:
        The path relative to the source root, the line, and the literal value.
    """
    for relative, tree in _iter_production_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                named = getattr(node.target, "id", "") == _COMPONENT_FIELD
            elif isinstance(node, ast.keyword):
                named = node.arg == _COMPONENT_FIELD
            else:
                continue
            if named and isinstance(node.value, ast.Constant):
                yield relative, node.lineno, node.value.value


def test_backend_component_fields_come_from_the_closed_vocabulary() -> None:
    """A label carried on a field is as binding as one passed at the call site.

    It is in fact worse to get wrong: the field's default stands for every role
    that does not override it, so one unknown value silently reassigns the spend
    of a whole family of backends.
    """
    offenders: list[str] = []
    checked = 0
    for relative, lineno, value in _iter_component_field_labels():
        checked += 1
        if value not in VALID_COMPONENTS:
            offenders.append(f"{relative}:{lineno}: {_COMPONENT_FIELD}={value!r}")
    assert not offenders, f"{_COMPONENT_FIELD} set to an unknown component:\n" + "\n".join(offenders)
    # The default plus at least one role that overrides it; fewer means the
    # field was renamed and this guard is watching nothing.
    assert checked >= 2, f"only {checked} {_COMPONENT_FIELD} literals found; the scan is no longer finding them"
