# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""JSON for the notebook's frozen dataclasses: fields by name, enums by value, tuples and sets as lists.

A type that defines ``to_document`` and ``from_document`` (the recipe and its delta) travels in its own document form.
A field missing from a stored object takes its default, so a field added later reads older rows.
"""

from __future__ import annotations

import dataclasses
import functools
import types
import typing
from collections.abc import Mapping
from enum import Enum

T = typing.TypeVar("T")

_SCALARS = (str, int, bool)
_SEQUENCES = (tuple, list, frozenset, set)
_MAPPINGS = (dict, Mapping)


def encode(value: object) -> object:
    """``value`` as JSON-ready data."""

    if hasattr(value, "to_document"):
        return value.to_document()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {item.name: encode(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted(encode(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def key_sorted(document: object) -> object:
    """``document`` with every object's keys in order, so a writer that keeps insertion order writes the line
    ``json.dumps(document, sort_keys=True)`` would."""

    if isinstance(document, Mapping):
        return {key: key_sorted(document[key]) for key in sorted(document)}
    if isinstance(document, (tuple, list)):
        return [key_sorted(item) for item in document]
    return document


def decode(kind: type[T], data: object) -> T:
    """Rebuild a ``kind`` from :func:`encode`'s output; raises ``TypeError`` or ``ValueError`` on other data."""

    return typing.cast(T, _decode(kind, data))


@functools.cache
def _fields(kind: type) -> dict[str, object]:
    return typing.get_type_hints(kind)


def _decode(hint: object, data: object) -> object:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        return _decode_optional(hint, data)
    if origin in _SEQUENCES:
        item_hint = typing.get_args(hint)[0]
        return origin(_decode(item_hint, item) for item in _list(data))
    if origin in _MAPPINGS:
        value_hint = typing.get_args(hint)[1]
        return {str(key): _decode(value_hint, item) for key, item in _object(data).items()}
    if hint is object:
        return data
    if isinstance(hint, type):
        return _decode_class(hint, data)
    raise TypeError(f"the notebook cannot decode {hint!r}")


def _decode_optional(hint: object, data: object) -> object:
    members = [member for member in typing.get_args(hint) if member is not type(None)]
    if len(members) != 1:
        raise TypeError(f"the notebook decodes only optional values, not {hint!r}")
    return None if data is None else _decode(members[0], data)


def _decode_class(kind: type, data: object) -> object:
    if hasattr(kind, "from_document"):
        return kind.from_document(data)
    if issubclass(kind, Enum):
        return kind(data)
    if dataclasses.is_dataclass(kind):
        stored = _object(data)
        return kind(**{name: _decode(hint, stored[name]) for name, hint in _fields(kind).items() if name in stored})
    if kind is float and isinstance(data, (int, float)) and not isinstance(data, bool):
        return float(data)
    if kind in _SCALARS and type(data) is kind:
        return data
    raise TypeError(f"expected {kind.__name__}, got {data!r}")


def _list(data: object) -> list[object]:
    if not isinstance(data, list):
        raise TypeError(f"expected a list, got {data!r}")
    return data


def _object(data: object) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise TypeError(f"expected an object, got {data!r}")
    return data


__all__ = ["decode", "encode", "key_sorted"]
