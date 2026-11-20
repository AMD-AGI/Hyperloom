# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What may cross the brain boundary: plain JSON values, and the contract's dataclasses as JSON objects.

Every contract type is a frozen dataclass whose fields are ``str``, ``int``, ``float``, ``bool``, an enum, another
contract dataclass, ``tuple[X, ...]``, ``X | None`` or a JSON mapping. :func:`to_wire` and :func:`from_wire` convert
them; ``from_wire`` refuses unknown and missing keys so a recorded call reads back exactly as it was made.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import json
import types
import typing
from typing import Any, Mapping, TypeVar, Union

T = TypeVar("T")


class WireError(ValueError):
    """A value that cannot cross the brain boundary unchanged."""


def json_value(value: Any, where: str) -> Any:
    """Return ``value`` as it reads back from JSON; refuse anything that would not survive the round trip.

    Tuples become lists, mapping keys must already be strings, and floats must be finite. A live object (a task
    handle, a path, a dataclass) is refused rather than stringified, so nothing leaks through by accident.
    """
    try:
        text = json.dumps(value, allow_nan=False, default=_refuse)
    except (TypeError, ValueError) as exc:
        raise WireError(f"{where}: {exc}") from None
    decoded = json.loads(text)
    if not _same_keys(value, decoded):
        raise WireError(f"{where}: mapping keys must be strings")
    return decoded


def json_mapping(value: Mapping[str, Any], where: str) -> dict[str, Any]:
    """A JSON object field, checked and copied."""
    if not isinstance(value, Mapping):
        raise WireError(f"{where}: expected a mapping, got {type(value).__name__}")
    return json_value(dict(value), where)


def _refuse(obj: object) -> Any:
    raise TypeError(f"{type(obj).__name__} is not a JSON value")


def _same_keys(original: Any, decoded: Any) -> bool:
    if isinstance(original, dict):
        if not all(isinstance(k, str) for k in original):
            return False
        return all(_same_keys(original[k], decoded[k]) for k in original)
    if isinstance(original, (list, tuple)):
        return all(_same_keys(a, b) for a, b in zip(original, decoded))
    return True


def to_wire(obj: Any) -> dict[str, Any]:
    """A contract dataclass as a JSON object."""
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        raise WireError(f"{type(obj).__name__} is not a contract dataclass")
    return {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_wire(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if isinstance(value, Mapping):
        return dict(value)
    return value


def from_wire(cls: type[T], data: Any) -> T:
    """Build ``cls`` from the JSON object :func:`to_wire` made of it."""
    return _decode(cls, data, cls.__name__)


def _decode_dataclass(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise WireError(f"{where}: expected an object, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise WireError(f"{where}: unknown keys {unknown}")
    missing = sorted(n for n, f in fields.items() if n not in data and _required(f))
    if missing:
        raise WireError(f"{where}: missing keys {missing}")
    kwargs = {name: _decode(hints[name], data[name], f"{where}.{name}") for name in fields if name in data}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise WireError(f"{where}: {exc}") from None


def _required(f: dataclasses.Field[Any]) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def _decode(tp: Any, data: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        return _decode_optional(tp, data, where)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _decode_dataclass(tp, data, where)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        try:
            return tp(data)
        except ValueError:
            raise WireError(f"{where}: {data!r} is not a {tp.__name__}") from None
    if origin is tuple:
        if not isinstance(data, list):
            raise WireError(f"{where}: expected a list, got {type(data).__name__}")
        item = typing.get_args(tp)[0]
        return tuple(_decode(item, v, f"{where}[{i}]") for i, v in enumerate(data))
    if origin in (collections.abc.Mapping, dict):
        return json_mapping(data, where)
    return _decode_scalar(tp, data, where)


def _decode_optional(tp: Any, data: Any, where: str) -> Any:
    args = [a for a in typing.get_args(tp) if a is not type(None)]
    if len(args) != 1 or len(typing.get_args(tp)) != 2:
        raise WireError(f"{where}: only X | None unions cross the boundary")
    return None if data is None else _decode(args[0], data, where)


def _decode_scalar(tp: Any, data: Any, where: str) -> Any:
    if tp is float and isinstance(data, int) and not isinstance(data, bool):
        return float(data)
    if tp in (str, int, float, bool) and type(data) is tp:
        return data
    raise WireError(f"{where}: expected {getattr(tp, '__name__', tp)}, got {type(data).__name__}")
