# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A compile-time knob of a native kernel: where in the component's upstream source it is set, and how a value is
forced, read from the pack.

A kernel picks a launch parameter (the warps of a block, the rows a block computes) per instance at compile time, in a
function of the upstream source. A knob forces one per instance without rewriting that function: a line rendered from
``insert`` goes right after ``anchor``, a line that occurs exactly once in ``file`` (the upstream path), and returns the
value for one instance before the function decides itself. An instance is one *key*: a value of each of ``axes``, in
their order (the weight type and the number of columns of a matrix-vector kernel, say), each key a template instance of
its own, so the value one key is given changes nothing of another's. ``values`` are what a key may be given, ``archs``
the GPU architectures the anchored branch is compiled for, and ``bench`` the microbenchmark case that runs one key:
the operation (one of :data:`_BENCH_PARAMS`) and its parameters, each a template over the axes.

Templates use ``${axis}`` and ``${value}`` (:class:`string.Template`), so a C++ brace needs no escaping. The pack pins
each entry to its examples: every example's ``source`` must hold the anchor once, and the settings it lists, applied
to it (:meth:`KernelKnobSpec.apply`), must give its ``expected`` text exactly.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from string import Template
from typing import Any

__all__ = [
    "KernelKnobSpec",
    "KnobKey",
    "KnobSpecError",
    "check_examples",
    "parse_knob",
]

#: A value of each of a knob's axes, in their order.
KnobKey = tuple[str, ...]
#: The microbenchmark operations a knob's ``bench`` may name, and the parameters each takes.
_BENCH_PARAMS: Mapping[str, frozenset[str]] = {"MUL_MAT": frozenset({"type_a", "n"})}
_VALUE = "value"
_OP = "op"
_FIELDS = frozenset({"component", "file", "anchor", "insert", "axes", "values", "archs", "bench"})
_EXAMPLE_FIELDS = frozenset({"source", "settings", "expected"})


class KnobSpecError(ValueError):
    """A knob entry, its settings or its source cannot be used; the message names the entry."""


@dataclass(frozen=True)
class KernelKnobSpec:
    """One knob (see the module). ``where`` names the entry in errors."""

    where: str
    component: str
    file: str
    anchor: str
    insert: str
    axes: tuple[tuple[str, tuple[str, ...]], ...]
    values: tuple[str, ...]
    archs: tuple[str, ...]
    bench: Mapping[str, str]

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.axes)

    def keys(self) -> list[KnobKey]:
        """Every key, in the order of the axes' values."""
        return list(itertools.product(*(values for _, values in self.axes)))

    def key_of(self, named: Mapping[str, Any]) -> KnobKey:
        """The key ``named`` gives a value of each axis.

        Raises:
            KnobSpecError: It names another axis, misses one, or gives one a value the axis does not hold.
        """
        if set(named) != set(self.axis_names):
            raise KnobSpecError(
                f"{self.where}: a key names exactly the axes {list(self.axis_names)}, not {sorted(named)}"
            )
        for name, held in self.axes:
            if str(named[name]) not in held:
                raise KnobSpecError(f"{self.where}: axis {name} holds {list(held)}, not {named[name]!r}")
        return tuple(str(named[name]) for name in self.axis_names)

    def label(self, key: KnobKey) -> str:
        """``axis=value,...`` of ``key``."""
        return ",".join(f"{name}={value}" for name, value in zip(self.axis_names, key, strict=True))

    def line(self, key: KnobKey, value: str) -> str:
        """The line that forces ``value`` on ``key``."""
        return Template(self.insert).substitute(self._named(key), **{_VALUE: value})

    def bench_case(self, key: KnobKey) -> dict[str, str]:
        """The microbenchmark case of ``key``: ``op`` and its parameters rendered for the key."""
        named = self._named(key)
        return {name: Template(text).substitute(named) for name, text in self.bench.items()}

    def apply(self, source: str, settings: Mapping[KnobKey, str]) -> str:
        """``source`` (the text of :attr:`file`) with one line per setting right after the anchor, in key order.

        Raises:
            KnobSpecError: The anchor is not one line of ``source`` exactly once, or a setting names a key or value the
                knob does not hold.
        """
        lines = source.splitlines(keepends=True)
        found = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == self.anchor]
        if len(found) != 1:
            raise KnobSpecError(
                f"{self.where}: the anchor {self.anchor!r} is {len(found)} lines of {self.file}, not exactly one: the "
                "upstream source moved; the pack entry must follow it"
            )
        order = {key: index for index, key in enumerate(self.keys())}
        for key, value in settings.items():
            if key not in order or value not in self.values:
                raise KnobSpecError(f"{self.where}: no setting {self.label(key) if key in order else key}={value!r}")
        newline = "\r\n" if lines[found[0]].endswith("\r\n") else "\n"
        added = [self.line(key, settings[key]) + newline for key in sorted(settings, key=order.__getitem__)]
        return "".join(lines[: found[0] + 1] + added + lines[found[0] + 1 :])

    def _named(self, key: KnobKey) -> dict[str, str]:
        return dict(zip(self.axis_names, key, strict=True))


def _strings(where: str, what: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise KnobSpecError(f"{where}: {what} must be a non-empty list of strings (quote a number)")
    if len(set(value)) != len(value):
        raise KnobSpecError(f"{where}: {what} lists a value twice")
    return tuple(value)


def _text(where: str, what: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnobSpecError(f"{where}: {what} must be a non-empty string")
    return value


def _placeholders(template: str) -> set[str]:
    return {
        match.group("named") or match.group("braced")
        for match in Template.pattern.finditer(template)
        if match.group("named") or match.group("braced")
    }


def _axes(where: str, raw: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(raw, dict) or not raw:
        raise KnobSpecError(f"{where}: axes must be a non-empty mapping of axis name to its values")
    if _VALUE in raw:
        raise KnobSpecError(f"{where}: an axis may not be called {_VALUE!r}, the name of the forced value")
    return tuple(
        (_text(where, "an axis name", name), _strings(where, f"axis {name}", held)) for name, held in raw.items()
    )


def _bench(where: str, raw: Any, axes: set[str]) -> dict[str, str]:
    if not isinstance(raw, dict) or raw.get(_OP) not in _BENCH_PARAMS:
        raise KnobSpecError(f"{where}: bench must be a mapping whose op is one of {sorted(_BENCH_PARAMS)}")
    params = {name: _text(where, f"bench.{name}", value) for name, value in raw.items()}
    wanted = _BENCH_PARAMS[params[_OP]] | {_OP}
    if set(params) != wanted:
        raise KnobSpecError(f"{where}: a {params[_OP]} bench holds exactly {sorted(wanted)}, not {sorted(params)}")
    named = set().union(*(_placeholders(text) for text in params.values()))
    if named - axes:
        raise KnobSpecError(f"{where}: bench names {sorted(named - axes)}, which are not axes")
    if axes - named:
        raise KnobSpecError(f"{where}: bench does not tell the keys apart: it names no {sorted(axes - named)}")
    return params


def parse_knob(where: str, body: Mapping[str, Any]) -> KernelKnobSpec:
    """The knob a pack entry's fields state.

    Raises:
        KnobSpecError: A field is missing, of the wrong shape, or the templates name something that is not an axis
            (or leave an axis out, so two keys would get the same line).
    """
    if set(body) != _FIELDS:
        raise KnobSpecError(f"{where}: a kernel knob holds exactly {sorted(_FIELDS)}, not {sorted(body)}")
    file = _text(where, "file", body["file"])
    if PurePosixPath(file).is_absolute() or ".." in PurePosixPath(file).parts:
        raise KnobSpecError(f"{where}: file is a path in the upstream tree, not {file!r}")
    axes = _axes(where, body["axes"])
    insert = _text(where, "insert", body["insert"])
    names = {name for name, _ in axes}
    if _placeholders(insert) != names | {_VALUE}:
        raise KnobSpecError(f"{where}: insert must name each axis and ${{{_VALUE}}}, and nothing else: {insert!r}")
    return KernelKnobSpec(
        where=where,
        component=_text(where, "component", body["component"]),
        file=file,
        anchor=_text(where, "anchor", body["anchor"]),
        insert=insert,
        axes=axes,
        values=_strings(where, "values", body["values"]),
        archs=_strings(where, "archs", body["archs"]),
        bench=_bench(where, body["bench"], names),
    )


def check_examples(where: str, spec: KernelKnobSpec, examples: Any) -> None:
    """Pin ``spec`` to its examples (see the module).

    Raises:
        KnobSpecError: An example is malformed, or applying its settings does not give its expected text.
    """
    if not isinstance(examples, list) or not examples:
        raise KnobSpecError(f"{where}: examples must be a non-empty list of {sorted(_EXAMPLE_FIELDS)}")
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or set(example) != _EXAMPLE_FIELDS:
            raise KnobSpecError(f"{where}: example {index} holds exactly {sorted(_EXAMPLE_FIELDS)}")
        settings = example["settings"]
        if not isinstance(settings, list) or not settings or not all(isinstance(item, dict) for item in settings):
            raise KnobSpecError(f"{where}: example {index}: settings must be a non-empty list of mappings")
        applied = {
            spec.key_of({name: value for name, value in item.items() if name != _VALUE}): str(item.get(_VALUE))
            for item in settings
        }
        got = spec.apply(_text(where, f"example {index} source", example["source"]), applied)
        if got != example["expected"]:
            raise KnobSpecError(f"{where}: example {index}: applying its settings gives\n{got}\nnot its expected text")
