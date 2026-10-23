# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a rule pack entry may say about how an image can confirm it: the ``image_check`` vocabulary.

A fact about an outside project stays true only while the image carries the version of the project that states it. An
entry may therefore declare ``image_check``, a list of checks the image can answer mechanically, and every check carries
the examples that pin what it accepts and what it refuses. The kinds are a closed list; a check that needs one more is a
change to this module and ``image_check_eval`` with its own tests, never a new pack.

``path_exists``       the path exists (``type`` file or dir).
``csv_has_columns``   the CSV at the path has every column of ``columns``.
``cli_accepts_flag``  the help of ``python -m <module>`` lists ``flag``.
``source_contains``   the source of ``module`` (or the file at ``path``) holds every ``text``.
``module_attr``       the source of ``module`` defines ``attr`` (``Class.method`` is allowed), without importing it.

A value is a literal string, or ``{ref: <entry>.<role>}`` which reads the name another entry of the pack holds, so that
a fact is still written once. A path is a value or a list of values joined with ``/``; its first element may be
``{env: NAME}``, the directory a variable names. A check about a path says nothing unless the image is the one the fact is
about, so a path check names ``installed`` (a Python package the image must carry) or starts from an ``{env: NAME}``
directory; a module check is anchored by the package of its module. Where the anchor is absent the check is
unverifiable, not failed.

A pack is trusted input: ``cli_accepts_flag`` runs ``python -m <module> --help`` of the module the pack names, and a flag
named anywhere in the help text (a sentence that says it was removed included) counts as listed.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

__all__ = [
    "CheckExamples",
    "CheckKind",
    "CheckSpec",
    "FileType",
    "ImageCheckSpecError",
    "MatchMode",
    "PathSpec",
    "parse_checks",
]

_CHECK = "check"
_EXAMPLES = "examples"
_MODULE = "module"
_PATH = "path"
_TEXT = "text"
_FLAG = "flag"
_ATTR = "attr"
_COLUMNS = "columns"
_INSTALLED = "installed"
_TYPE = "type"
_MATCH = "match"
_REF = "ref"
_ENV = "env"
_PASS = "pass"
_FAIL = "fail"
_HEADER = "header"
_HELP = "help"
_SOURCE = "source"
_TREE = "tree"

_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_PACKAGE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FLAG_NAME = re.compile(r"--?[A-Za-z0-9][A-Za-z0-9_.-]*")


class ImageCheckSpecError(ValueError):
    """An ``image_check`` that cannot be used; the message says which check and why."""


class CheckKind(Enum):
    """The closed list of checks an entry may declare."""

    PATH_EXISTS = "path_exists"
    CSV_HAS_COLUMNS = "csv_has_columns"
    CLI_ACCEPTS_FLAG = "cli_accepts_flag"
    SOURCE_CONTAINS = "source_contains"
    MODULE_ATTR = "module_attr"


class FileType(Enum):
    """What ``path_exists`` requires the path to be."""

    FILE = "file"
    DIR = "dir"


class MatchMode(Enum):
    """How ``source_contains`` finds a text: anywhere, or only as a whole word (an environment variable's name)."""

    SUBSTRING = "substring"
    WORD = "word"


class PathSpec(NamedTuple):
    """A path to look up: ``parts`` joined with ``/``, below the directory ``base_env`` names when it is set."""

    base_env: str | None
    parts: tuple[str, ...]

    def shown(self) -> str:
        """The path as a reader writes it, an environment base as ``$NAME``."""
        head = () if self.base_env is None else (f"${self.base_env}",)
        return posixpath.join(*head, *self.parts)


Example = Mapping[str, Any]


@dataclass(frozen=True)
class CheckExamples:
    """The environments a check is pinned by: it must pass on each of ``passing`` and fail on each of ``failing``."""

    passing: tuple[Example, ...]
    failing: tuple[Example, ...]


@dataclass(frozen=True)
class CheckSpec:
    """One check after its fields have been read, its references resolved and its shape checked."""

    kind: CheckKind
    examples: CheckExamples
    module: str | None = None
    path: PathSpec | None = None
    text: tuple[str, ...] = ()
    flag: str | None = None
    attr: str | None = None
    columns: tuple[str, ...] = ()
    installed: str | None = None
    file_type: FileType | None = None
    match: MatchMode = MatchMode.SUBSTRING

    def package(self) -> str | None:
        """The Python package whose presence says the image is the one this check is about, when there is one."""
        if self.installed is not None:
            return self.installed
        return None if self.module is None else self.module.split(".")[0]

    def describe(self) -> str:
        """One line that says what the check asks, for a report."""
        where = self.module if self.path is None else self.path.shown()
        if self.kind is CheckKind.PATH_EXISTS:
            return f"{where} exists" + ("" if self.file_type is None else f" as a {self.file_type.value}")
        if self.kind is CheckKind.CSV_HAS_COLUMNS:
            return f"{where} has the columns {list(self.columns)}"
        if self.kind is CheckKind.CLI_ACCEPTS_FLAG:
            return f"python -m {where} lists {self.flag}"
        if self.kind is CheckKind.SOURCE_CONTAINS:
            return f"{where} holds {list(self.text)}"
        return f"{where} defines {self.attr}"


Lookup = Callable[[str], str]
_Convert = Callable[[object, Lookup], Any]

_REQUIRED: dict[CheckKind, frozenset[str]] = {
    CheckKind.PATH_EXISTS: frozenset({_PATH}),
    CheckKind.CSV_HAS_COLUMNS: frozenset({_PATH, _COLUMNS}),
    CheckKind.CLI_ACCEPTS_FLAG: frozenset({_MODULE, _FLAG}),
    CheckKind.SOURCE_CONTAINS: frozenset({_TEXT}),
    CheckKind.MODULE_ATTR: frozenset({_MODULE, _ATTR}),
}
_OPTIONAL: dict[CheckKind, frozenset[str]] = {
    CheckKind.PATH_EXISTS: frozenset({_INSTALLED, _TYPE}),
    CheckKind.CSV_HAS_COLUMNS: frozenset({_INSTALLED}),
    CheckKind.CLI_ACCEPTS_FLAG: frozenset(),
    CheckKind.SOURCE_CONTAINS: frozenset({_MODULE, _PATH, _INSTALLED, _MATCH}),
    CheckKind.MODULE_ATTR: frozenset(),
}
#: The one key of an example that carries the content the check reads, per kind; ``env`` may be added to any.
_EXAMPLE_CONTENT: dict[CheckKind, str] = {
    CheckKind.PATH_EXISTS: _TREE,
    CheckKind.CSV_HAS_COLUMNS: _HEADER,
    CheckKind.CLI_ACCEPTS_FLAG: _HELP,
    CheckKind.SOURCE_CONTAINS: _SOURCE,
    CheckKind.MODULE_ATTR: _SOURCE,
}


def _text(what: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImageCheckSpecError(f"{what} must be a non-empty string")
    return value


def _named(what: str, value: object, pattern: re.Pattern[str]) -> str:
    text = _text(what, value)
    if not pattern.fullmatch(text):
        raise ImageCheckSpecError(f"{what} {text!r} is not a valid name")
    return text


def _scalar(what: str, raw: object, lookup: Lookup) -> str:
    if isinstance(raw, Mapping):
        if set(raw) != {_REF}:
            raise ImageCheckSpecError(
                f"{what}: a mapping value holds exactly the key {_REF!r} (or {_ENV!r} first in a path)"
            )
        return lookup(_text(f"{what}.{_REF}", raw[_REF]))
    return _text(what, raw)


def _convert_module(raw: object, _lookup: Lookup) -> str:
    return _named(_MODULE, raw, _DOTTED)


def _convert_attr(raw: object, _lookup: Lookup) -> str:
    return _named(_ATTR, raw, _DOTTED)


def _convert_package(raw: object, _lookup: Lookup) -> str:
    return _named(_INSTALLED, raw, _PACKAGE)


def _convert_flag(raw: object, lookup: Lookup) -> str:
    return _named(_FLAG, _scalar(_FLAG, raw, lookup), _FLAG_NAME)


def _convert_texts(raw: object, lookup: Lookup) -> tuple[str, ...]:
    items = raw if isinstance(raw, list) else [raw]
    if not items:
        raise ImageCheckSpecError(f"{_TEXT} must hold at least one text")
    return tuple(_scalar(_TEXT, item, lookup) for item in items)


def _convert_columns(raw: object, _lookup: Lookup) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ImageCheckSpecError(f"{_COLUMNS} must be a list with at least one column")
    return tuple(_text(_COLUMNS, column) for column in raw)


def _env_base(raw: object) -> str | None:
    if isinstance(raw, Mapping) and set(raw) == {_ENV}:
        return _named(f"{_PATH}.{_ENV}", raw[_ENV], _ENV_NAME)
    return None


def _convert_path(raw: object, lookup: Lookup) -> PathSpec:
    items = raw if isinstance(raw, list) else [raw]
    if not items:
        raise ImageCheckSpecError(f"{_PATH} must hold at least one part")
    base = _env_base(items[0])
    parts = tuple(_scalar(_PATH, item, lookup) for item in (items[1:] if base else items))
    if not parts and base is None:
        raise ImageCheckSpecError(f"{_PATH} must hold at least one part")
    if base is None and not parts[0].startswith("/"):
        raise ImageCheckSpecError(f"{_PATH} {parts[0]!r} must be absolute, or start from an {{{_ENV}: NAME}} directory")
    relative = parts[1:] if base is None else parts
    if any(part.startswith("/") for part in relative):
        raise ImageCheckSpecError(f"{_PATH} has a part after the first that is absolute")
    return PathSpec(base, parts)


def _enum_of(what: str, kind: type[Enum]) -> _Convert:
    def convert(raw: object, _lookup: Lookup) -> Enum:
        values = [member.value for member in kind]
        if raw not in values:
            raise ImageCheckSpecError(f"{what} {raw!r} is not one of {values}")
        return kind(raw)

    return convert


#: Field of the document -> attribute of ``CheckSpec`` and how to read it.
_FIELDS: dict[str, tuple[str, _Convert]] = {
    _MODULE: (_MODULE, _convert_module),
    _PATH: (_PATH, _convert_path),
    _TEXT: (_TEXT, _convert_texts),
    _FLAG: (_FLAG, _convert_flag),
    _ATTR: (_ATTR, _convert_attr),
    _COLUMNS: (_COLUMNS, _convert_columns),
    _INSTALLED: (_INSTALLED, _convert_package),
    _TYPE: ("file_type", _enum_of(_TYPE, FileType)),
    _MATCH: (_MATCH, _enum_of(_MATCH, MatchMode)),
}


def _example_environment(kind: CheckKind, index: int, which: str, raw: object) -> Example:
    where = f"{_EXAMPLES}.{which}[{index}]"
    content = _EXAMPLE_CONTENT[kind]
    if not isinstance(raw, dict) or not {content} <= set(raw) <= {content, _ENV}:
        raise ImageCheckSpecError(f"{where} must be a mapping with the key {content!r} (and optionally {_ENV!r})")
    if kind is CheckKind.PATH_EXISTS:
        if not isinstance(raw[content], list) or not all(isinstance(item, str) for item in raw[content]):
            raise ImageCheckSpecError(f"{where}.{content} must be a list of paths (a directory ends with '/')")
    else:
        _text(f"{where}.{content}", raw[content])
    variables = raw.get(_ENV, {})
    if not isinstance(variables, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in variables.items()
    ):
        raise ImageCheckSpecError(f"{where}.{_ENV} must map variable names to values")
    return raw


def _examples(kind: CheckKind, raw: object) -> CheckExamples:
    if not isinstance(raw, dict) or set(raw) != {_PASS, _FAIL}:
        raise ImageCheckSpecError(f"{_EXAMPLES} must hold the lists {_PASS!r} and {_FAIL!r}")
    held: dict[str, tuple[Example, ...]] = {}
    for which in (_PASS, _FAIL):
        items = raw[which]
        if not isinstance(items, list) or not items:
            raise ImageCheckSpecError(f"{_EXAMPLES}.{which} must be a list with at least one example")
        held[which] = tuple(_example_environment(kind, i, which, item) for i, item in enumerate(items))
    return CheckExamples(held[_PASS], held[_FAIL])


def _check_shape(spec: CheckSpec) -> None:
    uses_path = spec.path is not None
    if spec.kind is CheckKind.SOURCE_CONTAINS and (spec.module is None) == (spec.path is None):
        raise ImageCheckSpecError(f"{CheckKind.SOURCE_CONTAINS.value} names exactly one of {_MODULE!r} and {_PATH!r}")
    if uses_path and spec.installed is None and spec.path.base_env is None:  # type: ignore[union-attr]
        raise ImageCheckSpecError(
            f"a path check must say which image it is about: add {_INSTALLED}: <package>, or start the {_PATH} "
            f"from an {{{_ENV}: NAME}} directory (a path that is absent where the project is absent is not a failure)"
        )
    if not uses_path and spec.installed is not None:
        raise ImageCheckSpecError(f"{_INSTALLED} is for a path check; a module check is anchored by its own package")


def _parse_one(raw: object, lookup: Lookup) -> CheckSpec:
    if not isinstance(raw, dict):
        raise ImageCheckSpecError("must be a mapping")
    try:
        kind = CheckKind(raw.get(_CHECK))
    except ValueError:
        raise ImageCheckSpecError(
            f"{_CHECK} {raw.get(_CHECK)!r} is not one of {[kind.value for kind in CheckKind]}"
        ) from None
    expected = _REQUIRED[kind] | {_CHECK, _EXAMPLES}
    allowed = expected | _OPTIONAL[kind]
    if not expected <= set(raw) <= allowed:
        raise ImageCheckSpecError(
            f"a {kind.value} check holds {sorted(expected)} and may add {sorted(_OPTIONAL[kind])}; found {sorted(map(str, raw))}"
        )
    fields = {attribute: convert(raw[name], lookup) for name, (attribute, convert) in _FIELDS.items() if name in raw}
    spec = CheckSpec(kind=kind, examples=_examples(kind, raw[_EXAMPLES]), **fields)
    _check_shape(spec)
    return spec


def parse_checks(raw: object, lookup: Lookup) -> tuple[CheckSpec, ...]:
    """The checks of an entry's ``image_check``; ``lookup`` turns ``<entry>.<role>`` into the name the pack holds.

    Raises:
        ImageCheckSpecError: When the declaration is not a non-empty list, a check is of an unknown kind, lacks a field
            it needs or holds one it may not, names a reference the pack cannot resolve, or has no way to say which
            image it is about. The message starts with ``image_check[<index>]``.
    """
    if not isinstance(raw, list) or not raw:
        raise ImageCheckSpecError("image_check must be a list with at least one check")
    specs = []
    for index, item in enumerate(raw):
        try:
            specs.append(_parse_one(item, lookup))
        except ImageCheckSpecError as exc:
            raise ImageCheckSpecError(f"image_check[{index}]: {exc}") from exc
    return tuple(specs)
