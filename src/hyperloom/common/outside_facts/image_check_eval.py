# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Evaluators of the ``image_check`` kinds: what each asks of an environment and when it answers pass, fail or nothing.

A check passes when the environment confirms the fact, fails when the environment is the one the fact is about and
contradicts it, and is *unverifiable* in every other case: the project it is about is not installed, a variable that
names a checkout is not set, a command could not be run, a source file cannot be parsed. Unverifiable is never a pass
and never a failure: it says the fact was not looked at.

The same evaluators pin every check to its own examples when a pack is loaded (``check_examples``), so a check that
refuses what the real tool prints, or accepts what it no longer prints, is refused before a session uses it.
"""

from __future__ import annotations

import ast
import posixpath
import re
from collections.abc import Callable, Iterator
from enum import Enum
from typing import NamedTuple

from hyperloom.common.outside_facts.image_check_spec import (
    CheckKind,
    CheckSpec,
    ImageCheckSpecError,
    MatchMode,
)
from hyperloom.common.outside_facts.image_env import ExampleImage, Image, ImageReadError

__all__ = ["Outcome", "Status", "check_examples", "evaluate"]

#: A server prints its flags for ``--help``; one whose ``--help`` is a summary of groups lists them for ``--help=all``.
_HELP_OPTIONS = ("--help", "--help=all")
#: A help text that names fewer long options than this is a summary, not a listing of the flags.
_MIN_LISTED_OPTIONS = 3
_LONG_OPTION = re.compile(r"(?<![\w-])--[A-Za-z][\w-]*")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_DETAIL_CHARS = 300


class Status(Enum):
    """What a check found."""

    PASS = "pass"
    FAIL = "fail"
    UNVERIFIABLE = "unverifiable"


class Outcome(NamedTuple):
    """The answer of one check and, in a line, what it found."""

    status: Status
    detail: str


class _Unverifiable(Exception):
    """Raised inside an evaluator when the environment is not one the check can say anything about."""


def _clean(text: str) -> str:
    return _CONTROL.sub(" ", text).strip()[:_DETAIL_CHARS]


def _anchored(spec: CheckSpec, image: Image) -> None:
    package = spec.package()
    if package is not None and not image.installed(package):
        raise _Unverifiable(f"the Python package {package} is not installed here")


def _target(spec: CheckSpec, image: Image) -> str:
    """The absolute path a path check looks at; unverifiable where the image or the checkout is not there."""
    _anchored(spec, image)
    assert spec.path is not None
    base = ""
    if spec.path.base_env is not None:
        named = image.getenv(spec.path.base_env)
        if named is None:
            raise _Unverifiable(f"{spec.path.base_env} is not set")
        if not posixpath.isabs(named) or image.path_type(named) is None:
            raise _Unverifiable(f"{spec.path.base_env} does not name a directory here")
        base = named
    return posixpath.join(base, *spec.path.parts)


def _source_of(spec: CheckSpec, image: Image) -> str | None:
    """The source a module check reads; ``None`` when the package is there and holds no such module."""
    _anchored(spec, image)
    assert spec.module is not None
    return image.module_source(spec.module)


def _path_exists(spec: CheckSpec, image: Image) -> Outcome:
    target = _target(spec, image)
    found = image.path_type(target)
    if found is None:
        return Outcome(Status.FAIL, f"not found at {target}")
    if spec.file_type is not None and found is not spec.file_type:
        return Outcome(Status.FAIL, f"{target} is a {found.value}, the pack says a {spec.file_type.value}")
    return Outcome(Status.PASS, f"found at {target}")


def _csv_has_columns(spec: CheckSpec, image: Image) -> Outcome:
    target = _target(spec, image)
    header = image.csv_columns(target)
    if header is None:
        return Outcome(Status.FAIL, f"no such file: {target}")
    missing = [column for column in spec.columns if column not in header]
    if missing:
        return Outcome(Status.FAIL, f"{target} lacks the columns {missing}; its header is {list(header)}")
    return Outcome(Status.PASS, f"all {len(spec.columns)} columns are in {target}")


def _contains(source: str, text: str, mode: MatchMode) -> bool:
    if mode is MatchMode.WORD:
        return re.search(rf"(?<!\w){re.escape(text)}(?!\w)", source) is not None
    return text in source


def _source_contains(spec: CheckSpec, image: Image) -> Outcome:
    if spec.module is not None:
        source, name = _source_of(spec, image), "its source"
    else:
        target = _target(spec, image)
        source, name = image.read_text(target), target
    if source is None:
        return Outcome(Status.FAIL, "there is no such source file")
    missing = [text for text in spec.text if not _contains(source, text, spec.match)]
    if missing:
        return Outcome(Status.FAIL, f"{name} does not hold {missing}")
    return Outcome(Status.PASS, f"all {len(spec.text)} texts are in {name}")


def _defined(body: list[ast.stmt]) -> Iterator[tuple[str, ast.stmt]]:
    """The names a block of statements binds, looking into the blocks a module may wrap its definitions in."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            yield from ((target.id, node) for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            yield from ((alias.asname or alias.name.split(".")[0], node) for alias in node.names)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            yield from _defined(node.body)
            yield from _defined(getattr(node, "orelse", []))


def _module_attr(spec: CheckSpec, image: Image) -> Outcome:
    source = _source_of(spec, image)
    if source is None:
        return Outcome(Status.FAIL, "there is no such source file")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise _Unverifiable(f"its source cannot be parsed ({type(exc).__name__})") from exc
    assert spec.attr is not None
    body: list[ast.stmt] = tree.body
    for step, name in enumerate(spec.attr.split(".")):
        node = dict(_defined(body)).get(name)
        if node is None:
            return Outcome(Status.FAIL, f"it defines no {'.'.join(spec.attr.split('.')[: step + 1])}")
        body = node.body if isinstance(node, ast.ClassDef) else []
    return Outcome(Status.PASS, f"it defines {spec.attr}")


def _usable_help(text: str) -> bool:
    return len(set(_LONG_OPTION.findall(text))) >= _MIN_LISTED_OPTIONS


def _lists(text: str, flag: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text) is not None


def _cli_accepts_flag(spec: CheckSpec, image: Image) -> Outcome:
    _anchored(spec, image)
    assert spec.module is not None
    assert spec.flag is not None
    listed = False
    for option in _HELP_OPTIONS:
        try:
            text = image.help_text(spec.module, option)
        except ImageReadError:
            if option == _HELP_OPTIONS[0]:
                raise
            break
        if not _usable_help(text):
            continue
        listed = True
        if _lists(text, spec.flag):
            return Outcome(Status.PASS, f"the help lists {spec.flag}")
    if not listed:
        raise _Unverifiable("the help of the server prints no list of flags")
    return Outcome(Status.FAIL, f"the help does not list {spec.flag}")


_EVALUATORS: dict[CheckKind, Callable[[CheckSpec, Image], Outcome]] = {
    CheckKind.PATH_EXISTS: _path_exists,
    CheckKind.CSV_HAS_COLUMNS: _csv_has_columns,
    CheckKind.CLI_ACCEPTS_FLAG: _cli_accepts_flag,
    CheckKind.SOURCE_CONTAINS: _source_contains,
    CheckKind.MODULE_ATTR: _module_attr,
}


def evaluate(spec: CheckSpec, image: Image) -> Outcome:
    """Ask ``image`` what ``spec`` asks; every detail is one clean line, whatever the environment printed."""
    try:
        outcome = _EVALUATORS[spec.kind](spec, image)
    except _Unverifiable as exc:
        outcome = Outcome(Status.UNVERIFIABLE, str(exc))
    except ImageReadError as exc:
        outcome = Outcome(Status.UNVERIFIABLE, f"could not be read: {exc}")
    return Outcome(outcome.status, _clean(outcome.detail))


def check_examples(spec: CheckSpec) -> None:
    """Pin ``spec`` to its examples: it passes on each of ``passing`` and fails on each of ``failing``.

    Raises:
        ImageCheckSpecError: Naming the example that was judged otherwise, and how.
    """
    for expected, examples in ((Status.PASS, spec.examples.passing), (Status.FAIL, spec.examples.failing)):
        for index, example in enumerate(examples):
            outcome = evaluate(spec, ExampleImage(example))
            if outcome.status is not expected:
                raise ImageCheckSpecError(
                    f"the example examples.{expected.value}[{index}] came out {outcome.status.value} "
                    f"({outcome.detail}), the check must {expected.value} on it"
                )
