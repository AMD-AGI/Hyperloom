# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a component's files live: the layout of its upstream repository over the image's install, read from the pack.

A patch names each file by its path in the component's upstream repository. A layout maps those paths onto places in the
image: each *section* is an upstream path prefix, the base it lies over (the installed package, or aiter's meta
directory), and the directory inside that base the section starts at. A path belongs to the section with the longest
matching prefix; a path outside every section cannot be applied to the image. A layout also lists the files whose change
the measured install cannot show, in two lists. ``refuse`` names the files no patch can change in any session: a
compiled file (``*.so``, ``*.co``) is what a build makes, never a source of one. ``native`` names the component's native
sources and build definition, which only a build of the component measures: the package layout refuses them, because
laying a changed C++ file beside the package that runs the old binary would measure old bits, and a native build, where
one is configured, compiles them. A patch is refused naming the pattern. A pattern without a slash names files wherever
they lie; one with a slash names them only below a directory, for a source that is native only where the component's
build compiles it (vLLM's ``csrc/*.cu``), not inside the package, whose runtime may compile a source of its own.

The pack carries the layouts (``component_layout`` entries); this module reads and applies them, and the loader pins each
entry to its own examples with :func:`check_examples`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any

__all__ = [
    "BASES",
    "META_BASE",
    "PACKAGE_BASE",
    "ComponentLayoutSpec",
    "LayoutSection",
    "LayoutSpecError",
    "check_examples",
    "parse_layout",
]

#: The installed package's own directory.
PACKAGE_BASE = "package"
#: The directory aiter reads ``csrc`` and ``hsa`` from (``AITER_META_DIR``).
META_BASE = "meta"
BASES = (PACKAGE_BASE, META_BASE)
_REFUSE, _NATIVE, _GENERATED, _INTO, _PLACE = "refuse", "native", "generated", "into", "place"
_FIELDS = frozenset({"import_name", "sections", _REFUSE, _NATIVE, _GENERATED})
_SECTION_FIELDS = frozenset({"prefix", "base", _INTO})
_ARROW = " -> "
_SLASH = "/"


class LayoutSpecError(ValueError):
    """A layout entry or its examples are not usable; the message says what is wrong."""


@dataclass(frozen=True)
class LayoutSection:
    """The upstream directory ``prefix`` lies over the directory ``into`` of the base ``base``."""

    prefix: str
    base: str = PACKAGE_BASE
    into: str = ""


@dataclass(frozen=True)
class ComponentLayoutSpec:
    """The layout of one component: what its sections are, which files cannot be measured (``refuse`` in any session,
    ``native`` without a native build of the component), and which directories of the package its runtime writes builds
    into."""

    import_name: str
    sections: tuple[LayoutSection, ...]
    refuse: tuple[str, ...] = ()
    generated: tuple[str, ...] = ()
    native: tuple[str, ...] = ()

    def place(self, path: str) -> tuple[LayoutSection, str] | None:
        """The section of the upstream path ``path`` and its path inside the section's base; ``None`` outside all."""
        parts = PurePosixPath(path).parts
        best: tuple[LayoutSection, int] | None = None
        for section in self.sections:
            depth = len(PurePosixPath(section.prefix).parts)
            if len(parts) > depth and parts[:depth] == PurePosixPath(section.prefix).parts:
                if best is None or depth > best[1]:
                    best = (section, depth)
        if best is None:
            return None
        section, depth = best
        return section, str(PurePosixPath(section.into, *parts[depth:]))

    def refused_by(self, path: str) -> str | None:
        """The pattern of ``refuse`` that names ``path``, a file no patch can change; else ``None``."""
        return _named_by(self.refuse, path)

    def native_by(self, path: str) -> str | None:
        """The pattern of ``native`` that names ``path``, a file only a native build measures; else ``None``."""
        return _named_by(self.native, path)


def _named_by(patterns: tuple[str, ...], path: str) -> str | None:
    """The first of ``patterns`` that names the upstream path ``path``.

    A pattern without a slash is matched on the file's name, wherever the file lies. A pattern with a slash is matched
    on the whole upstream path, from the repository's top, where ``*`` also matches a slash: ``csrc/*.cu`` is every
    ``.cu`` file below ``csrc``, at any depth, and no ``.cu`` file elsewhere."""
    upstream = PurePosixPath(path)
    name, whole = upstream.name, upstream.as_posix()
    return next((p for p in patterns if fnmatchcase(whole if _SLASH in p else name, p)), None)


def _strings(where: str, what: str, value: Any, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise LayoutSpecError(f"{where}: {what} must be a list of strings{'' if allow_empty else ' with one or more'}")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LayoutSpecError(f"{where}: {what} holds {item!r}, which is not a non-empty string")
    return tuple(value)


def _relative(where: str, what: str, value: str) -> str:
    parts = PurePosixPath(value).parts
    if value.startswith("/") or ".." in parts or (parts and value != PurePosixPath(*parts).as_posix()):
        raise LayoutSpecError(f"{where}: {what} {value!r} must be a plain relative path without '..'")
    return value


def _patterns(where: str, what: str, value: Any) -> tuple[str, ...]:
    return tuple(
        _relative(where, what, p) if _SLASH in p else p for p in _strings(where, what, value, allow_empty=True)
    )


def _section(where: str, index: int, raw: Any) -> LayoutSection:
    at = f"{where}: sections[{index}]"
    if not isinstance(raw, dict) or not _SECTION_FIELDS - {_INTO} <= set(raw) <= _SECTION_FIELDS:
        raise LayoutSpecError(f"{at} holds 'prefix' and 'base' and may add 'into'; found {raw!r}")
    prefix, base, into = raw["prefix"], raw["base"], raw.get(_INTO, "")
    if not isinstance(prefix, str) or not prefix.strip() or not isinstance(into, str):
        raise LayoutSpecError(f"{at}: prefix must be a non-empty string and into a string")
    if base not in BASES:
        raise LayoutSpecError(f"{at}: base {base!r} is not one of {list(BASES)}")
    return LayoutSection(_relative(at, "prefix", prefix), base, _relative(at, "into", into))


def parse_layout(where: str, body: Mapping[str, Any]) -> ComponentLayoutSpec:
    """The layout a ``component_layout`` entry's fields state; :class:`LayoutSpecError` when they cannot be used."""
    if set(body) != _FIELDS:
        raise LayoutSpecError(f"{where}: holds exactly {sorted(_FIELDS)}; found {sorted(map(str, body))}")
    name = body["import_name"]
    if not isinstance(name, str) or not name.isidentifier():
        raise LayoutSpecError(f"{where}: import_name must be the package's import name, got {name!r}")
    raw = body["sections"]
    if not isinstance(raw, list) or not raw:
        raise LayoutSpecError(f"{where}: sections must be a list with at least one section")
    sections = tuple(_section(where, index, item) for index, item in enumerate(raw))
    if sections[0].base != PACKAGE_BASE:
        raise LayoutSpecError(f"{where}: the first section must lie over the installed package")
    prefixes = [section.prefix for section in sections]
    if len(set(prefixes)) != len(prefixes):
        raise LayoutSpecError(f"{where}: two sections have the same prefix")
    generated = tuple(
        _relative(where, _GENERATED, path) for path in _strings(where, _GENERATED, body[_GENERATED], allow_empty=True)
    )
    refuse, native = _patterns(where, _REFUSE, body[_REFUSE]), _patterns(where, _NATIVE, body[_NATIVE])
    return ComponentLayoutSpec(name, sections, refuse, generated, native)


def _example_list(where: str, examples: Any, key: str) -> tuple[str, ...]:
    if not isinstance(examples, dict) or set(examples) != {_PLACE, _REFUSE}:
        raise LayoutSpecError(f"{where}: examples must hold the lists 'place' and 'refuse'")
    return _strings(where, f"examples.{key}", examples[key], allow_empty=False)


def check_examples(where: str, spec: ComponentLayoutSpec, examples: Any) -> None:
    """Run the layout against its own examples: every ``place`` example ``<upstream path> -> <base>:<path in base>``
    must be placed so, every ``refuse`` example (a path outside the sections, or one a pattern of ``refuse`` or
    ``native`` names) must be refused."""
    for line in _example_list(where, examples, _PLACE):
        path, arrow, expected = line.partition(_ARROW)
        placed = spec.place(path) if arrow else None
        if (
            placed is None
            or spec.refused_by(path)
            or spec.native_by(path)
            or f"{placed[0].base}:{placed[1]}" != expected
        ):
            raise LayoutSpecError(f"{where}: the example {line!r} is not placed as it says by the sections")
    for path in _example_list(where, examples, _REFUSE):
        if spec.place(path) is not None and (spec.refused_by(path) or spec.native_by(path)) is None:
            raise LayoutSpecError(f"{where}: the example {path!r} must be refused, but the layout takes it")
