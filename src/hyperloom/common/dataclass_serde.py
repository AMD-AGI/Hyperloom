# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fast dataclass snapshots with the default ``dataclasses.asdict`` contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any


_ATOMIC_TYPES = frozenset({type(None), bool, int, float, str, bytes, complex})


@dataclass
class _Box:
    value: Any


def fast_asdict(root: Any) -> dict[str, Any]:
    """Copy a dataclass instance as ``asdict(root)`` with its default dict factory."""
    if isinstance(root, type) or not is_dataclass(root):
        raise TypeError("asdict() should be called on dataclass instances")
    return {item.name: _convert(getattr(root, item.name)) for item in fields(root)}


def _convert(value: Any) -> Any:
    value_type = type(value)
    if value_type in _ATOMIC_TYPES:
        return value
    if is_dataclass(value_type):
        return {item.name: _convert(getattr(value, item.name)) for item in fields(value)}
    if value_type is dict:
        return {
            (key if type(key) in _ATOMIC_TYPES else _convert(key)): (
                item if type(item) in _ATOMIC_TYPES else _convert(item)
            )
            for key, item in value.items()
        }
    if value_type is list:
        return [item if type(item) in _ATOMIC_TYPES else _convert(item) for item in value]
    if value_type is tuple:
        return tuple(item if type(item) in _ATOMIC_TYPES else _convert(item) for item in value)
    # Stdlib owns namedtuple/subclass reconstruction and per-leaf deepcopy.
    # Plain deepcopy here would leave dataclasses inside containers unconverted.
    return asdict(_Box(value))["value"]
